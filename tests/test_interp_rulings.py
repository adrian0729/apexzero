"""Protocol section 12's pre-tag rulings R1-R13, one regression test (or more) per ruling
that changes behaviour. R6 and R8 are in test_interp_measures.py, R10-R12's decision rules in
test_interp_analysis.py (they reuse its fixtures); this module covers the rest. Nothing is
solved or rolled out: keys and runs are doubles, banks are hand-built strips, and every file
goes to a tmp dir."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.interp import agents, io, measures, stimuli, support
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import (
    Cell,
    CornerSpec,
    Geometry,
    Landmarks,
    Stimulus,
    StimulusBank,
)
from apexzero.tracks.corners import R_DEV


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Bank arrays are float32 as in the rollouts; other modules switch x64 on at import."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def ps():
    import sys

    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


class _NullLog:
    """Stands in for io.StageLog so no test writes runs/*.log or runs/activity.md."""

    def __init__(self, *a, **k):
        self.done = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def step(self, n: int = 1, note: str = ""):
        self.done += n

    def set_total(self, n: int):
        pass


# ------------------------------------------------------------------------- R1


def test_R1_real_laps_run_before_the_tag_everything_else_stays_gated(ps, monkeypatch):
    """R1: `rollouts --real-laps` runs before the tag (it repeats the standard R_DEV
    evaluation); after the tag only on the tagged code. Every other agent rollout still
    needs the tag."""
    calls = []
    monkeypatch.setattr(ps, "_real_laps", lambda args, before_tag=False: calls.append(before_tag))
    monkeypatch.setattr(ps, "_tag_exists", lambda: False)
    ps.main(["rollouts", "--real-laps"])
    assert calls == [True]
    for argv in (["rollouts"], ["rollouts", "--yasmarina"], ["anchored-keys"], ["probes"]):
        with pytest.raises(SystemExit, match="does not exist yet"):
            ps.main(argv)
    monkeypatch.setattr(ps, "_tag_exists", lambda: True)
    monkeypatch.setattr(ps, "_manifest", lambda: agents.Manifest(tag="psychophysics-prereg-v2 x"))
    monkeypatch.setattr(ps, "_code_changed_since_tag", lambda: ["src/apexzero/interp/rollout.py"])
    with pytest.raises(SystemExit, match="frozen code differs"):
        ps.main(["rollouts", "--real-laps"])
    monkeypatch.setattr(ps, "_code_changed_since_tag", list)
    ps.main(["rollouts", "--real-laps"])
    assert calls == [True, False]


def test_R1_cli_help_names_the_ruling(ps, capsys):
    with pytest.raises(SystemExit):
        ps.build_parser().parse_args(["rollouts", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "ruling R1" in out and "standard evaluations" in out


def test_R1_each_real_lap_is_logged_as_a_standard_evaluation(ps, monkeypatch, tmp_path):
    """R1: every real-lap run is a 'standard evaluation (R1)' entry in the manifest, written
    when its lap file is; a resume from the saved laps logs nothing new."""
    path = tmp_path / "manifest.json"
    agents.save_manifest(agents.Manifest(), path)
    real_update = agents.update_manifest
    monkeypatch.setattr(agents, "update_manifest", lambda fn, p=path: real_update(fn, p))
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(io, "STIMULI_DIR", tmp_path / "stimuli")
    monkeypatch.setattr(io, "StageLog", _NullLog)
    monkeypatch.setattr(ps, "_manifest", lambda: agents.load_manifest(path))
    monkeypatch.setattr(ps, "_finals", lambda m: {"G2": SimpleNamespace(path="runs/g2/agent.pkl")})
    monkeypatch.setattr(ps, "_load", lambda entry: None)
    monkeypatch.setattr(stimuli, "real_track", lambda name: SimpleNamespace(ds=2.0))
    monkeypatch.setattr(stimuli, "real_state_requests", lambda reach: [])
    ran = []

    def fake_lap(agent, track):
        ran.append(track)
        x = np.zeros((5, 8))
        x[:, 0] = np.arange(5.0)
        return agents.RealLap(agent="G2", track=track, v_start=50.0, x=x,
                              prev_action=np.zeros((5, 2)), accel=np.zeros((5, 2)),
                              valid=np.ones(5, bool), term_step=-1, cause="", length=4000.0)

    monkeypatch.setattr(agents, "real_lap", lambda ag, track: fake_lap("G2", track))
    ps._real_laps(SimpleNamespace(force=False), before_tag=True)
    m = agents.load_manifest(path)
    ev = m.lists["standard_evaluations"]
    assert len(ev) == len(R_DEV) and {e["track"] for e in ev} == set(R_DEV)
    assert all(e["category"] == ps.R1_CATEGORY == "standard evaluation (R1)" for e in ev)
    assert all(e["agent"] == "G2" and e["before_tag"] and e["outcome"] == "two laps"
               and e["file"].startswith("stimuli/real_laps/") for e in ev)
    assert "pre_tag_agent_rollouts" not in m.lists          # not one of the two exceptions
    ps._real_laps(SimpleNamespace(force=False), before_tag=True)       # resume
    assert len(ran) == len(R_DEV)
    assert len(agents.load_manifest(path).lists["standard_evaluations"]) == len(R_DEV)


def test_R1_a_lap_written_but_not_logged_is_logged_on_resume(ps, monkeypatch, tmp_path):
    """R1 logs every real-lap run. A run that stops between writing its lap file and its
    manifest record leaves a lap that a resume loads instead of re-running: the resume logs
    it (once), marked ``logged_on_resume``."""
    path = tmp_path / "manifest.json"
    agents.save_manifest(agents.Manifest(), path)
    real_update = agents.update_manifest
    monkeypatch.setattr(agents, "update_manifest", lambda fn, p=path: real_update(fn, p))
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(io, "STIMULI_DIR", tmp_path / "stimuli")
    monkeypatch.setattr(io, "StageLog", _NullLog)
    monkeypatch.setattr(ps, "_manifest", lambda: agents.load_manifest(path))
    monkeypatch.setattr(ps, "_finals", lambda m: {"G2": SimpleNamespace(path="runs/g2/agent.pkl")})
    monkeypatch.setattr(ps, "_load", lambda entry: None)
    monkeypatch.setattr(stimuli, "real_track", lambda name: SimpleNamespace(ds=2.0))
    monkeypatch.setattr(stimuli, "real_state_requests", lambda reach: [])
    ran = []

    def fake_lap(ag, track):
        ran.append(track)
        x = np.zeros((5, 8))
        x[:, 0] = np.arange(5.0)
        return agents.RealLap(agent="G2", track=track, v_start=50.0, x=x,
                              prev_action=np.zeros((5, 2)), accel=np.zeros((5, 2)),
                              valid=np.ones(5, bool), term_step=2, cause="off track",
                              length=4000.0)

    monkeypatch.setattr(agents, "real_lap", fake_lap)
    ps._real_laps(SimpleNamespace(force=False), before_tag=True)
    lost = min(R_DEV)
    agents.update_manifest(lambda mm: mm.lists.__setitem__("standard_evaluations", [
        e for e in mm.lists["standard_evaluations"] if e["track"] != lost]))
    ps._real_laps(SimpleNamespace(force=False), before_tag=True)       # resume
    assert len(ran) == len(R_DEV)                                     # nothing re-run
    ev = agents.load_manifest(path).lists["standard_evaluations"]
    assert len(ev) == len(R_DEV) and {e["track"] for e in ev} == set(R_DEV)
    again = [e for e in ev if e["track"] == lost]
    assert len(again) == 1 and again[0]["logged_on_resume"] and again[0]["before_tag"]
    assert again[0]["outcome"] == "terminated: off track"
    assert not any(e.get("logged_on_resume") for e in ev if e["track"] != lost)
    ps._real_laps(SimpleNamespace(force=False), before_tag=True)       # logged once
    assert len(agents.load_manifest(path).lists["standard_evaluations"]) == len(R_DEV)
    assert "200-step margin" in ps.R1_CONVENTION and "diagnose_crashes" in ps.R1_CONVENTION


# ------------------------------------------------------------------------- R2


def _ladder(n_tail: int, ok: int, repro: int | None = None, fid_fail: int = 0,
            n_bulk: int = 3, bulk_context: int = 3) -> dict:
    """G2 tail units (the first ``repro`` reproduced, the first ``ok`` single-corner with
    twin agreement, the last ``fid_fail`` failing the fidelity gate) and G1 bulk units (all
    reproduced, the first ``bulk_context`` context)."""
    repro = ok if repro is None else repro
    rows = []

    def add(agent, unit, set_, version, det, fid=True):
        rows.append(dict(agent=agent, unit=unit, set=set_, version=version, fidelity_ok=fid,
                         det_crashes=det, det_n=4, nat_k=0, nat_n=16, e_ve=0.2,
                         det_overspeed=0))

    for i in range(n_tail):
        u, fid = f"G2/T{i}", i < n_tail - fid_fail
        add("G2", u, "tail", "real", 4 if i < repro else 0, fid)
        add("G2", u, "tail", "isolated", 3 if i < ok else 0, fid)
        add("G2", u, "tail", "twin", 2 if i < ok else 0, fid)
    for i in range(n_bulk):
        u = f"G1/B{i}"
        add("G1", u, "bulk", "real", 4)
        add("G1", u, "bulk", "isolated", 0 if i < bulk_context else 4)
        add("G1", u, "bulk", "twin", 0)
    return A._to_table(rows, A.LADDER_COLUMNS)


def _p8(*a, **k) -> dict:
    return {d.prediction: d for d in A.p8(_ladder(*a, **k))}


def test_R2_p8_thresholds_are_proportional():
    assert A.p8_tail_thresholds(5) == (3, 1)                  # the protocol's original 3 and 1
    assert A.p8_tail_thresholds(3) == (2, 0) and A.p8_tail_thresholds(4) == (3, 0)
    assert A.p8_tail_thresholds(10) == (6, 2) and A.p8_tail_thresholds(7) == (5, 1)
    assert [A.p8_bulk_minimum(n) for n in (3, 4, 6, 8)] == [2, 3, 4, 6]
    # n = 5 reproduces the original rule
    assert _p8(5, 3)["P8-tail"].verdict == "confirm"
    assert _p8(5, 2)["P8-tail"].verdict == "inconclusive"
    assert _p8(5, 1)["P8-tail"].verdict == "refute"
    assert _p8(5, 3, repro=3)["P8-tail"].details["confirm_min"] == 3
    # n = 3 (G2's tail once YasMarina and Zandvoort are bulk): confirm on 2, refute on 0
    assert _p8(3, 2)["P8-tail"].verdict == "confirm"
    assert _p8(3, 1)["P8-tail"].verdict == "inconclusive"
    assert _p8(3, 0)["P8-tail"].verdict == "refute"
    # n < 3: not testable
    d = _p8(2, 2)["P8-tail"]
    assert d.verdict == "not testable" and d.details["n_units"] == 2
    # transplants failing the fidelity gate leave P8 (3.2): 4 units, 1 failed -> n = 3
    d = _p8(4, 2, fid_fail=1)["P8-tail"]
    assert d.verdict == "confirm" and d.details["n_units"] == 3
    assert d.details["fidelity_failed"] == ["G2/T3"]


def test_R2_bulk_context_share_is_proportional():
    assert _p8(3, 2, n_bulk=4, bulk_context=3)["P8-bulk"].verdict == "confirm"
    d = _p8(3, 2, n_bulk=4, bulk_context=2)["P8-bulk"]
    assert d.verdict == "refute" and d.details["confirm_min"] == 3 and d.details["n_context"] == 2


def test_R2_leaves_the_bulk_rule_testable_below_3_units():
    """R2 makes only the tail prediction "not testable" for n < 3. The bulk rule (context in
    >= 2/3 of the reproduced bulk units) still decides on 1 or 2 reproduced units, as
    before R2, and is not testable only with none."""
    d = _p8(3, 2, n_bulk=2, bulk_context=2)["P8-bulk"]
    assert d.verdict == "confirm" and d.details["n_reproduced"] == 2
    assert d.details["confirm_min"] == 2
    assert _p8(3, 2, n_bulk=2, bulk_context=1)["P8-bulk"].verdict == "refute"     # 1/2 < 2/3
    assert _p8(3, 2, n_bulk=1, bulk_context=1)["P8-bulk"].verdict == "confirm"
    assert _p8(3, 2, n_bulk=1, bulk_context=0)["P8-bulk"].verdict == "refute"
    d = _p8(3, 2, n_bulk=0)["P8-bulk"]
    assert d.verdict == "not testable" and d.details["n_reproduced"] == 0


def test_R2_a_gate_with_missing_inputs_keeps_the_unit_in_p8_tail():
    """Protocol 3.2 excludes transplants that fail the fidelity gate. A unit whose gate could
    not be evaluated (no real lap, state index entry, real rung or u = 0 run) did not fail
    it: it stays in P8-tail's n as unreproduced and is listed, so one missing lap cannot
    turn a 3-unit tail "not testable" silently."""
    lad = _ladder(3, 2)
    unit, ver = lad["unit"].astype(str), lad["version"].astype(str)
    lad["fidelity_ok"] = np.where(unit == "G2/T1", False, lad["fidelity_ok"])
    lad["fidelity_missing"] = unit == "G2/T1"
    d = {x.prediction: x for x in A.p8(lad)}["P8-tail"]
    assert d.details["n_units"] == 3 and d.details["fidelity_missing"] == ["G2/T1"]
    assert d.details["fidelity_failed"] == [] and d.details["reproduced"] == 1
    assert d.details["labels"]["G2/T1"] == "fidelity not evaluated (unreproduced)"
    assert d.verdict == "inconclusive"                     # 1 single-with-twin of 3
    # a unit without a real rung at all is missing data too
    keep = ~((unit == "G2/T1") & (ver == "real"))
    d = {x.prediction: x for x in A.p8({k: v[keep] for k, v in lad.items()})}["P8-tail"]
    assert d.details["n_units"] == 3 and d.details["fidelity_missing"] == ["G2/T1"]
    # an evaluated failure still leaves P8 (3.2)
    lad["fidelity_missing"] = np.zeros_like(lad["fidelity_missing"])
    d = {x.prediction: x for x in A.p8(lad)}["P8-tail"]
    assert d.verdict == "not testable" and d.details["fidelity_failed"] == ["G2/T1"]


def test_R2_ladder_table_marks_unevaluated_gates_and_keeps_empty_units():
    rt = A._to_table([dict(cell=0, mode="det", phase=0, crashed=False, crash_type="",
                           e_ve=0.0, s_crash=np.nan)], ("cell", "mode", "phase", "crashed",
                                                        "crash_type", "e_ve", "s_crash"))
    kt = A._to_table([dict(cell=0, s_c0=1000.0)], ("cell", "s_c0"))
    units = [dict(agent="G2", unit=f"G2/X/{i}", set="tail", cells=c)
             for i, c in enumerate(({"real": [0]}, {"real": [0]}, {"real": [0]}, {}))]
    t = A.ladder_table(rt, kt, units, {"G2/X/0": True, "G2/X/1": False, "G2/X/2": None})
    assert t["unit"].tolist() == ["G2/X/0", "G2/X/1", "G2/X/2", "G2/X/3"]
    assert t["fidelity_ok"].tolist() == [True, False, False, False]
    assert t["fidelity_missing"].tolist() == [False, False, True, True]
    assert t["version"][3] == "real" and t["det_n"][3] == 0


def test_R2_crash_corner_regimes_come_from_their_R_seg():
    """R2: every crash corner is the segment corner containing the crash (D15) and its set is
    its R_seg's regime under the agent's frozen q1 / q10 (step 1's frozen values); no unit
    carries a radius label and no named set table remains."""
    assert [f.name for f in dataclasses.fields(stimuli.CrashPair)] == [
        "agent", "track", "lap", "s_crash"]
    assert not hasattr(A, "P8_SETS") and not hasattr(stimuli, "LABEL_MISMATCH")
    reg = {"G1": A.Regime(12.02, 21.32, 8.94), "G2": A.Regime(11.08, 17.67, 7.64)}
    sets = {(p.agent, p.track): A.p8_set(p.agent, stimuli.crash_unit_R_seg(p), reg)
            for p in stimuli.CRASH_PAIRS}
    assert sets[("G2", "YasMarina")] == "bulk" and sets[("G2", "Zandvoort")] == "bulk"
    assert {t for (a, t), v in sets.items() if a == "G2" and v == "tail"} == {
        "Norisring", "Melbourne", "Catalunya"}
    assert sets[("G1", "YasMarina")] == "below"
    assert A.p8_set("G2", 7.0, reg) == "below"                  # clip counts as below
    assert A.p8_set("G3", 15.0, reg) == "other"                 # no frozen regimes yet


# ------------------------------------------------------------------------- R3, R4


def _corner_rows():
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc = stimuli.corner_profile(spec)
    area = float(np.degrees(np.sum(pc.kappa) * 2.0))
    return spec, pc, area


def _geo(theta_eff=90.0, R_seg=20.0, theta_seg=90.0, L_seg=60.0) -> Geometry:
    return Geometry(40.0, theta_eff, 1 / 800, R_seg, theta_seg, L_seg, 0.0, 1030.0, 1, False, 6.0)


def _bank(rows):
    stims = [Stimulus(i, name, kind, ("A",), 1, -1, lm, geo, spec, source=src)
             for i, (name, kind, lm, geo, spec, src, _) in enumerate(rows)]
    k = np.stack([r[6] for r in rows])
    w = np.full_like(k, 6.0)
    return StimulusBank(k, w, w.copy(), stims)


def _kinked_row():
    """Step 1's R6/180 typical row (l_r 13.47 m): its plateau ends between 2 m knots, so its
    knot samples turn 179.56 deg, a 0.44 deg miss from sampling the kinks alone."""
    spec = CornerSpec(R=6.0, theta_deg=180.0, direction=1, width=12.0,
                      l_r=13.465230941772461)
    pc = stimuli.corner_profile(spec)
    return spec, pc, float(np.degrees(np.sum(pc.kappa) * 2.0))


def test_R3_R4_accepted_rows_pass_and_are_listed(ps):
    """R3: a theta_eff miss from knot sampling of the trapezoid's kinks (theta_eff is the
    samples' own area, within the kinks' bound of the nominal angle) is accepted and listed;
    any other miss fails. R4 (as amended, R4a): a twin matching R_seg and the angle within
    the protocol's 2% and 1 deg but shorter than its corner is accepted with its length
    shortfall listed; a miss on R_seg beyond 2%, or a longer twin, fails."""
    spec, pc, area = _corner_rows()
    lm = pc.landmarks
    k_spec, k_pc, k_area = _kinked_row()
    assert 0.3 < 180.0 - k_area < stimuli.theta_sampling_bound_deg(k_spec)
    scaled = pc.kappa * 1.006                                    # sample area 90.54 deg
    a_scaled = float(np.degrees(np.sum(scaled) * 2.0))
    tgt = {"twin_target": {"R_seg": 20.0, "theta_deg": 90.0, "L_seg": 60.0},
           "twin_achieved": {"note": "L_seg exceeds the longest trapezoid"}}
    good = [("syn/sampled/L", "synthetic", k_pc.landmarks, _geo(theta_eff=k_area), k_spec, {},
             k_pc.kappa),
            ("twin/short", "twin", lm, _geo(L_seg=50.0), None, tgt, pc.kappa),
            ("twin/exact", "twin", lm, _geo(L_seg=59.0), None, tgt, pc.kappa)]
    bad = [("syn/off/L", "synthetic", lm, _geo(theta_eff=area + 0.45), spec, {}, pc.kappa),
           ("syn/scaled/L", "synthetic", lm, _geo(theta_eff=a_scaled), spec, {}, scaled),
           ("twin/long", "twin", lm, _geo(L_seg=70.0), None, tgt, pc.kappa),
           ("twin/radius", "twin", lm, _geo(R_seg=21.0, L_seg=50.0), None, tgt, pc.kappa),
           ("twin/far", "twin", lm, _geo(R_seg=20.5, L_seg=50.0), None, tgt, pc.kappa)]
    res = stimuli.check_design_details(_bank(good), [])
    th, tw = res["theta_eff"], res["twin_fit"]
    assert th["ok"] and th["accepted_by"] == "R3" and [r["name"] for r in th["accepted"]] == [
        "syn/sampled/L"]
    assert th["accepted"][0]["sampling_bound_deg"] == pytest.approx(
        np.degrees((1 / 6) * 4.0 / (2 * k_spec.l_r)))
    assert tw["ok"] and tw["accepted_by"] == "R4" and [r["name"] for r in tw["accepted"]] == [
        "twin/short"]
    assert tw["accepted"][0]["length_shortfall"] == pytest.approx(10.0)
    assert tw["length_shortfall"] == pytest.approx({"twin/short": 10.0, "twin/exact": 1.0})
    assert stimuli.check_design(_bank(good), [])["twin_fit"]
    line = ps.check_line("twin_fit", tw)
    assert "PASS" in line and "1 rows accepted per R4 (listed): twin/short" in line
    res = stimuli.check_design_details(_bank(good + bad), [])
    assert not res["theta_eff"]["ok"]
    assert {r["name"]: r["why"] for r in res["theta_eff"]["fails"]} == {
        "syn/off/L": "theta_eff is not the knot samples' area",
        # theta_eff is the samples' area, but 0.54 deg is beyond the kinks' 0.14 deg bound
        "syn/scaled/L": "samples' area beyond the kinks' sampling bound"}
    assert not res["twin_fit"]["ok"]
    assert sorted(r["name"] for r in res["twin_fit"]["fails"]) == [
        "twin/far", "twin/long", "twin/radius"]           # twin/far: 2.5%, beyond 2%
    near = [("twin/near", "twin", lm, _geo(R_seg=20.1, L_seg=50.0), None, tgt, pc.kappa)]
    res = stimuli.check_design_details(_bank(near), [])      # 0.5%: within 2%, accepted (R4a)
    assert res["twin_fit"]["ok"] and [r["name"] for r in res["twin_fit"]["accepted"]] == [
        "twin/near"]
    assert stimuli.twin_length_shortfall(_bank(good).stimuli[1]) == pytest.approx(10.0)
    assert np.isnan(stimuli.twin_length_shortfall(_bank(good).stimuli[0]))


def test_R3_a_spec_that_disagrees_with_its_angle_still_fails(monkeypatch):
    """R3 accepts sampling misses only: a spec whose plateau is 5 m too long builds samples
    that match its own trapezoid (kappa_formula passes) and a theta_eff equal to their area,
    so only the continuous trapezoid's area against the nominal angle catches it."""
    from apexzero.interp.stimuli import _assemble, _Row

    rows = []
    for R, th in ((14.6, 90.0), (40.0, 90.0), (28.6, 180.0)):
        spec = CornerSpec(R=R, theta_deg=th, direction=1, width=12.0, l_r=20.0)
        assert stimuli.spec_area_deg(spec) == pytest.approx(th, abs=1e-9)
    monkeypatch.setattr(CornerSpec, "L_p", property(
        lambda self: max(self.theta / self.kappa_peak - self.l_r, 0.0) + 5.0))
    for R, th in ((14.6, 90.0), (40.0, 90.0), (28.6, 180.0)):
        spec = CornerSpec(R=R, theta_deg=th, direction=1, width=12.0, l_r=20.0)
        rows.append(_Row(f"bug/R{R}/T{th}", stimuli.corner_profile(spec), "synthetic", 1, spec,
                         {"A"}))
    stims, _ = _assemble(rows)
    bank = StimulusBank(np.stack([r.profile.kappa for r in rows]),
                        np.stack([r.profile.w_left for r in rows]),
                        np.stack([r.profile.w_right for r in rows]), stims)
    res = stimuli.check_design_details(bank, [])
    assert res["kappa_formula"]["ok"]
    th = res["theta_eff"]
    assert not th["ok"] and not th["accepted"] and th["spec_area_max_deg"] > 5.0
    assert {r["why"] for r in th["fails"]} == {"spec area is not the nominal angle"}
    assert len(th["fails"]) == 3


def test_R4_shortfall_is_a_twin_transfer_covariate():
    """R4: the twin-transfer table carries each twin's length shortfall, and P8-twin
    reports it as a covariate without changing the verdict."""
    rows = [dict(agent="G2", corner=f"E1/c{i}", source="E1", e_ve_real=0.1 * i,
                 e_ve_isolated=0.0, e_ve_twin=d, e_ve_curve=0.1 * i, length_shortfall=sh)
            for i, (d, sh) in enumerate(((0.0, 0.0), (0.01, 1.0), (0.2, 10.0), (0.3, 30.0)))]
    twins = A._to_table(rows, A.TWIN_COLUMNS)
    assert "length_shortfall" in A.TWIN_COLUMNS
    d = next(x for x in A.p8(_ladder(3, 2), twins) if x.prediction == "P8-twin")
    assert d.details["n_length_short"] == 2 and d.details["length_shortfall_median"] == 5.5
    assert d.details["frac_agree_length_short"] == 0.0
    assert d.details["frac_agree_length_ok"] == 1.0
    assert d.details["spearman_shortfall_disagreement"] == pytest.approx(1.0)
    no_cov = A._to_table([{k: v for k, v in r.items() if k != "length_shortfall"} for r in rows],
                         A.TWIN_COLUMNS)
    d0 = next(x for x in A.p8(_ladder(3, 2), no_cov) if x.prediction == "P8-twin")
    assert d0.verdict == d.verdict


# ------------------------------------------------------------------------- R5, R9, R13


def test_R5_typical_ramp_and_speed_keep_the_10_deg_filter():
    for fn in (support.l_typ_table, support.v_typ_table):
        assert inspect.signature(fn).parameters["min_theta_deg"].default == 10.0
    assert support.TYPICAL_MIN_THETA_DEG == 10.0 and support.V_TYP_MIN_COUNT == 20


def test_R9_r_clip_uses_the_sign_that_clips_first():
    mu, var = np.zeros(200), np.ones(200)
    mu[support.KAPPA_CHANNELS], var[support.KAPPA_CHANNELS] = 0.2, 0.25
    rc = support.r_clip({"mean": mu, "var": var})
    sd = np.sqrt(0.25 + 1e-8)
    np.testing.assert_allclose(rc, 50.0 / (10.0 * sd - 0.2))          # not 50 / (|mu| + 10 sd)


def test_R13_analysis_decisions_are_signed_off():
    doc = A.__doc__
    assert all(f"AD{i} " in doc for i in range(1, 26)) and "R13" in doc


# ------------------------------------------------------------------------- R7


def _cut_design():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells, cid = [], [], 0
    for R in (28.6, 35.8, 44.7, 55.9):
        for th in (90.0, 180.0):
            for d in (1, -1):
                spec = CornerSpec(R=R, theta_deg=th, direction=d, width=12.0, l_r=20.0)
                geo = Geometry(R, th, np.nan, R, th, 40.0, 0.0, 1020.0, 1, False, 6.0)
                i = len(stims)
                stims.append(Stimulus(i, f"syn/R{R}/T{th:.0f}/{d}", "synthetic", ("A",), d, -1,
                                      lm, geo, spec))
                for ap, v0, D in (("A1", 86.0, 600.0), ("A2", 50.0, 200.0)):
                    cells.append(Cell(cid, i, cid // 2, ("A",), ap, v0, D))
                    cid += 1
    return SimpleNamespace(stimuli=stims), cells


def test_R7_rule_2_cuts_keys_of_3_geometries_x_2_approaches_only(ps):
    bank, cells = _cut_design()
    proj = {"cuts": ["rule 2: tier A 180 deg R >= 35.8 dropped"]}
    out, n = ps.flag_budget_cuts(bank, cells, proj)
    cut = [c for c in out if A.has_no_key(c)]
    assert n == 12 and len(cut) == 12 and all("budget_cut" in c.flags for c in cut)
    geos = {(bank.stimuli[c.stim_id].spec.R, c.approach) for c in cut}
    assert geos == {(R, ap) for R in (35.8, 44.7, 55.9) for ap in ("A1", "A2")}
    assert all(bank.stimuli[c.stim_id].spec.theta_deg == 180.0 for c in cut)
    assert ps.flag_budget_cuts(bank, out, proj)[1] == 0                 # idempotent
    assert ps.flag_budget_cuts(bank, cells, {"cuts": []}) == (cells, 0)
    # A later projection without rule 2 solves those keys again (``_cut``), so the flags go:
    # a stale 'no_key' would hide keys the stage now solves.
    back, n_back = ps.flag_budget_cuts(bank, out, {"cuts": []})
    assert n_back == 12 and not any(A.has_no_key(c) or "budget_cut" in c.flags for c in back)
    assert [c.flags for c in back] == [c.flags for c in cells]
    assert len(ps._cut(bank, back, {"cuts": []}, "nominal_A")) == len(back)
    assert ak.DEFAULT_CUTS["nominal_A"] == ak.DEFAULT_CUTS["backoff"] == 6     # one direction
    assert len(ps._cut(bank, out, proj, "nominal_A")) == len(out) - 12
    # keys only: every cut cell is still rolled out
    assert {c.cell_id for c in ps.rollout_cells(out, ("A",))} == {c.cell_id for c in out}
    assert ps.anchored_cells(out, bank, {}, "G1", "P0", True) == [
        c for c in out if c.approach == "A1" and not A.has_no_key(c)]


def _mini_bank():
    """One R40/90 left corner cell at A1 (cut or not) and its null straight."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, pst = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A",), 1, -1, pc.landmarks,
                      geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, pst.landmarks, geo, None)]
    bank = StimulusBank(np.stack([pc.kappa, pst.kappa]), np.stack([pc.w_left, pst.w_left]),
                        np.stack([pc.w_right, pst.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), "A1", 60.0, 300.0, W=200, P=3.0, null_cell=2),
             Cell(1, 0, 1, ("A",), "A1", 60.0, 300.0, W=200, P=3.0, null_cell=2,
                  flags=("budget_cut", A.NO_KEY_FLAG)),
             Cell(2, 1, 2, ("G",), "G", 60.0, 600.0, W=200, P=3.0)]
    return bank, cells


class _Keys:
    """Every cell has a verified nominal key (v_e* 30); cell 0 also a re-anchored one (31)."""

    def nominal(self, cell, phase):
        m = measures.ResponseMetrics(v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, has_brake=True)
        return A.KeyView(None, m, True, "verified")

    def backoff(self, cell):
        return None

    def anchored(self, cell, agent, ts):
        if cell.cell_id != 0:
            return None
        return A.KeyView(None, measures.ResponseMetrics(v_e=31.0), True, "verified")

    def null(self, cell):
        return None

    def socp(self, cell):
        return None

    def no_return(self, *a):
        return None


def _runs(monkeypatch, bank, cells, keys):
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: SimpleNamespace())
    monkeypatch.setattr(measures, "response_metrics", lambda *a, **k: measures.ResponseMetrics(
        v_e=33.0, v_min=25.0, d_b=95.0, has_brake=True))
    monkeypatch.setattr(measures, "key_errors", lambda m, mk, *a, **k: measures.KeyErrors(
        e_ve=m.v_e / mk.v_e - 1.0, dd_b=m.d_b - mk.d_b))
    det = dict(cell=np.array([0, 1]), stim=np.array([0, 0]), pair=np.array([0, 1]),
               phase=np.array([0, 0]), sample=np.array([-1, -1]), steer_sign=np.array([1, 1]),
               W=np.array([200, 200]), s0=np.array([700.0, 700.0]))
    return A.run_table("G1", 1, {"det": det}, cells, bank, keys, None, check_design=False)


def test_R7_cut_cells_carry_no_key_and_leave_key_based_errors(monkeypatch):
    bank, cells = _mini_bank()
    rt = _runs(monkeypatch, bank, cells, _Keys())
    assert rt["e_ve"][0] == pytest.approx(0.1) and np.isnan(rt["e_ve"][1])   # key hidden
    assert not rt["key_at_phase"][1] and np.isnan(rt["e_ve_ra"][1])
    kt = A.key_table(cells, bank, _Keys())
    assert kt["key_status"].tolist() == ["verified", "no_key", "verified"]
    assert np.isnan(kt["v_e_star"][1]) and not kt["braking"][1]
    pv = A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)})
    exc = dict(zip(pv["pair"].tolist(), pv["excluded"].tolist(), strict=True))
    assert exc == {0: "", 1: "no_key"}
    assert A.NO_KEY_FLAG in A.EXCL_KEY and A.NO_KEY_FLAG not in A.EXCL_ALL


# ------------------------------------------------------------------------- R12


def test_R12_run_and_pair_tables_record_the_re_anchored_key(monkeypatch):
    bank, cells = _mini_bank()
    cells = [cells[0], dataclasses.replace(cells[1], flags=()), cells[2]]
    rt = _runs(monkeypatch, bank, cells, _Keys())
    assert rt["ra_key"].tolist() == [True, False]
    assert rt["e_ve_ra"][0] == pytest.approx(33.0 / 31.0 - 1.0)
    assert rt["e_ve_ra"][1] == pytest.approx(33.0 / 30.0 - 1.0)       # nominal fallback
    pv = A.pair_values(rt, A.key_table(cells, bank, _Keys()), {"G1": A.Regime(12.0, 20.0)})
    fb = dict(zip(pv["pair"].tolist(), pv["ra_fallback"].tolist(), strict=True))
    assert fb == {0: 0, 1: 1} and bool(pv["uses_ra"].all())


def _regime_design():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, (R, ap) in enumerate(((7.0, "A1"), (10.0, "A1"), (15.0, "A1"), (30.0, "A1"),
                                 (10.0, "A2"), (11.0, "A1"))):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        stims.append(Stimulus(i, f"s{i}", "synthetic", ("A",), 1, -1, lm, geo, None))
        flags = ("budget_cut", A.NO_KEY_FLAG) if i == 5 else ()
        cells.append(Cell(i, i, 10 + i, ("A",), ap, 86.0, 600.0, flags=flags))
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9)}
    return SimpleNamespace(stimuli=stims), cells, reg


def test_R12_p0_re_anchors_the_non_clipped_below_support_a1_cells(ps):
    """R12: P0 adds the non-clipped below-support A1 cells to A1 tail and bulk (design-time
    clipping without rollouts, the binding run-time one with them); P2 (P6c's checkpoints)
    keeps tail and bulk; a budget-cut cell gets no key."""
    bank, cells, reg = _regime_design()
    ids = lambda xs: [c.cell_id for c in xs]
    # R 7 is clip-saturated by design, R 10 below support, 15 tail, 30 bulk; A2 never.
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P0", True)) == [1, 2, 3]
    # run time: R 10's pair clipped, R 7's not -> R 7 is in P1b's set, R 10 is not
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P0", True, {11})) == [0, 2, 3]
    assert ids(ps.anchored_cells(cells, bank, reg, "G1", "P2", True)) == [2, 3]
    assert ids(ps.anchored_cells(cells, bank, reg, "S1", "P0", False)) == []
    runs = {"mode": np.array(["det", "det", "nat", "det"]),
            "kappa_clipped": np.array([True, False, True, False]),
            "pair": np.array([11, 12, 13, 11])}
    assert A.kappa_clipped_pairs(runs) == {11}


def test_R12_budget_counts_the_extra_anchored_keys(ps, monkeypatch):
    bank, cells, reg = _regime_design()
    monkeypatch.setattr(support, "load_support", lambda d: SimpleNamespace(regimes=reg))
    assert ps._r12_extra_anchored(bank, cells) == 1                    # the R 10 A1 cell
    # The whole P0 count is taken on the design (tail 15, bulk 30, below 10), so R12's cell
    # and the tail and bulk cells are counted alike, not added to the protocol's fixed 96.
    g = ps._anchored_g_count(bank, cells)
    assert g["n"] == 3 and g["source"] == "design" and g["r12_below_support"] == 1
    monkeypatch.setattr(support, "load_support", lambda d: (_ for _ in ()).throw(
        FileNotFoundError(d)))
    assert ps._r12_extra_anchored(bank, cells) == 0                    # nothing frozen yet
    g = ps._anchored_g_count(bank, cells)
    assert g["n"] == ps.ANCHORED_G_PROTOCOL == 96 and g["r12_below_support"] == 0


def test_R7_rule_2_credit_counts_the_cut_keys_in_both_directions(ps):
    """Rule 2's credit is counted on the projection's own item sets: 6 nominal keys with the
    mirror shortcut (one direction), 12 without it, as ``_cut`` drops; phase-matched keys of
    the non-180 deg phase cells likewise."""
    bank, cells = _cut_design()
    # each left-hand cell paired with its mirror (same R, theta, approach), as in the design
    ids: dict = {}
    geo = lambda c: (bank.stimuli[c.stim_id].spec.R, bank.stimuli[c.stim_id].spec.theta_deg,
                     c.approach)
    cells = [dataclasses.replace(c, pair_id=ids.setdefault(geo(c), len(ids))) for c in cells]
    one, both = ps._key_sets(bank, cells, False), ps._key_sets(bank, cells, True)
    assert 2 * len(one["nominal_A"]) == len(both["nominal_A"]) == len(cells)
    c1, c2 = ps._rule2_cut_counts(bank, one), ps._rule2_cut_counts(bank, both)
    proj = {"cuts": ["rule 2: x"]}
    for sets, c in ((one, c1), (both, c2)):
        assert c["nominal_A"] == len(sets["nominal_A"]) - len(
            ps._cut(bank, sets["nominal_A"], proj, "nominal_A"))
    assert c1["nominal_A"] == 6 == ak.DEFAULT_CUTS["nominal_A"] and c2["nominal_A"] == 12
    assert c1["backoff"] == c2["backoff"] == ak.DEFAULT_CUTS["backoff"]
    assert c1["phase"] == c2["phase"] == 0          # no A1 cell with R <= 18.3 here
