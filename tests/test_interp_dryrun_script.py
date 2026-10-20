"""Dry-run rulings R20-R28 (protocol 12, 2026-10-20) wired in scripts/psychophysics.py.

The controls stage's order (c_v re-tuned under R14 after R22 A, c3's tau_c from R26's rule on
the braking keys, c1's shifts in the manifest before c1 rolls out, every control, c1's
viability solves, T64 in its own process), the dry run's required verdicts as R20, R23, R24,
R26, R27 and R28 amend them, and the stimuli stage's R21 loader with the F6 scale kept.

Freeze rule: no agent, no key solve, no rollout. Every stage call runs on a two-cell design
with its heavy steps stubbed, and nothing is written under outputs/ (tmp dirs only).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.interp import agents, controls, io, measures, rollout, stimuli
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import (
    Cell,
    CornerSpec,
    Geometry,
    Stimulus,
    StimulusBank,
)


@pytest.fixture(scope="module")
def ps():
    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


class _Log:
    def __init__(self, *a, **k):
        self.done, self.total, self.lines = 0, int(k.get("total", a[1] if len(a) > 1 else 0)), []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def step(self, n: int = 1, note: str = ""):
        self.done += n

    def set_total(self, n: int):
        self.total = int(n)

    def write(self, msg: str):
        self.lines.append(msg)


def _design():
    """A tier A A1 corner (left), a tier B corner and their null straight, on a real
    StimulusBank (row hashes, landmarks)."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, pst = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A", "B"), 1, -1,
                      pc.landmarks, geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, pst.landmarks, geo,
                      None)]
    bank = StimulusBank(np.stack([pc.kappa, pst.kappa]), np.stack([pc.w_left, pst.w_left]),
                        np.stack([pc.w_right, pst.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), "A1", 86.0, 600.0, W=400, P=4.0, null_cell=2),
             Cell(1, 0, 1, ("B",), "B", 60.0, 250.0, W=400, P=4.0, null_cell=2),
             Cell(2, 1, 2, ("G",), "G", 86.0, 600.0, W=400, P=4.0)]
    return bank, cells


# ------------------------------------------------------------------------- controls stage


def _stub_controls_stage(ps, monkeypatch, tmp_path, bank, cells):
    rec = dict(calls=[], manifest=SimpleNamespace(budget={}, validation={}, lists={}), store={})
    ctl = tmp_path / "controls"
    ctl.mkdir(exist_ok=True)

    def note(*what):
        rec["calls"].append(what)

    def update_manifest(fn):
        fn(rec["manifest"])

    def key_table(cs, bank_, keys):
        ids = [c.cell_id for c in cs]
        return {"cell": np.array(ids), "key_status": np.array(["verified"] * len(ids)),
                "braking": np.array(["G" not in c.tiers for c in cs]),
                "d_b_star": np.full(len(ids), 100.0)}

    def tau_c_k1(bank_, brk, car, c_v):
        note("tau_c_k1", tuple(c.cell_id for c in brk), c_v)
        return 1.2, dict(rows=[], n=4 * len(brk), n_ok=4 * len(brk), n_infeasible=0,
                         n_no_trigger=0, verified=True, bracket=(0.25, 4.0), rule="R26")

    def c1_shifts(bank_, cs, kt, car, c_v):
        note("c1_shifts", tuple(c.cell_id for c in cs))
        return {c.cell_id: dict(shift=15.0 + 20.0, lead=20.0, d_b_q=120.0, d_b_key=100.0,
                                status="ok") for c in cs}

    def run_one(name, spec, bank_, cells_, car, c_v, tau_c, cache, k=1, shifts=None):
        note("run", name, tau_c, None if shifts is None else dict(shifts),
             "c1_shifts" in rec["manifest"].budget)
        return {"stim": np.zeros(1, int)}, dict(agent=f"control-{name}", timesteps=0,
                                                 log_std=[-np.inf, -np.inf], control=name)

    def save(path, groups, meta):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        rec["store"][str(path)] = dict(meta)

    monkeypatch.setattr(ps, "_refuse_after_tag", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_require_steps", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_design", lambda laid_out=False: (bank, cells))
    monkeypatch.setattr(ps, "_run_one_control", run_one)
    monkeypatch.setattr(ps, "_knn_fallback", lambda res, *a: res)
    monkeypatch.setattr(ps, "_c1_viability", lambda *a: note("viability") or dict(jobs=2))
    monkeypatch.setattr(ps, "_t64", lambda *a: note("T64") or "2 runs")
    monkeypatch.setattr(measures, "model_tables", lambda: None)
    monkeypatch.setattr(controls, "car_tables", lambda t: "CAR")
    monkeypatch.setattr(controls, "tune_cv", lambda bank_, tc, car, gate_cells=None: (
        note("tune_cv") or (0.9, {0.95: 0.98, 0.9: 1.0})))
    monkeypatch.setattr(controls, "tau_c_k1", tau_c_k1)
    monkeypatch.setattr(controls, "c1_shifts", c1_shifts)
    monkeypatch.setattr(controls, "completion", lambda arrays, bank_: np.ones(1, bool))
    monkeypatch.setattr(A, "CachedKeys", lambda *a, **k: "KEYS")
    monkeypatch.setattr(A, "key_table", key_table)
    monkeypatch.setattr(ak, "KeyCache", lambda: "CACHE")
    monkeypatch.setattr(ak, "key_for", lambda c, kind, phase, **k: SimpleNamespace(
        status="verified", a=np.zeros((400, 2)), key_id=f"{kind}{c.cell_id}"))
    monkeypatch.setattr(io, "CONTROLS_DIR", ctl)
    monkeypatch.setattr(io, "StageLog", _Log)
    monkeypatch.setattr(io, "code_hash", lambda *a, **k: "code")
    monkeypatch.setattr(io, "rollout_path", lambda agent, ts: tmp_path / "rollouts" / agent / "0.npz")
    monkeypatch.setattr(rollout, "save_rollouts", save)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": {}}, rec["store"][str(p)]))
    monkeypatch.setattr(agents, "update_manifest", update_manifest)
    return rec


def _args(**kw):
    return SimpleNamespace(knn_buffer=False, force=False, workers=1, **kw)


def test_R20_to_R28_controls_stage_order_and_resume(ps, monkeypatch, tmp_path):
    """The controls stage: c_v tuned under R14 with Q's law recorded; tau_c from R26's rule
    on the verified braking tier A/B keys only; c1's shifts in the manifest (and
    controls/c1_shifts.json) before c1 rolls out, and handed to it; c3 at tau_c; then c1's
    viability solves (R28) and T64 (R20). A second run resumes every control; a tuning made
    under the as-built Q law (no 'q_law', the key-fitted tau_c) is redone."""
    bank, cells = _design()
    rec = _stub_controls_stage(ps, monkeypatch, tmp_path, bank, cells)
    ps.stage_controls(_args())
    calls = rec["calls"]
    names = [c[1] for c in calls if c[0] == "run"]
    assert names == list(controls.CONTROLS)
    assert calls[0] == ("tune_cv",)
    assert ("tau_c_k1", (0, 1), 0.9) in calls                     # not the null straight
    i_shift = calls.index(("c1_shifts", (0, 1, 2)))
    c1 = next(c for c in calls if c[:2] == ("run", "c1"))
    assert calls.index(c1) > i_shift and c1[4]                    # in the manifest first
    assert c1[3] == {i: dict(shift=35.0, lead=20.0, d_b_q=120.0, d_b_key=100.0, status="ok")
                     for i in (0, 1, 2)}
    assert next(c for c in calls if c[:2] == ("run", "c3"))[2] == 1.2
    assert [c[0] for c in calls[-2:]] == ["viability", "T64"]
    tun = io.read_json(tmp_path / "controls" / "tuning.json")
    assert tun["q_law"] == controls.Q_LAW and tun["tau_rule"] == controls.TAU_RULE
    assert tun["tau_c"] == 1.2 and "d_b_key" not in tun
    budget = rec["manifest"].budget
    assert budget["c1_shifts"]["status"] == {"ok": 3} and budget["controls_tuning"]["c_v"] == 0.9
    shifts_file = io.read_json(tmp_path / "controls" / "c1_shifts.json")
    assert shifts_file["hash"] == budget["c1_shifts"]["hash"]
    c1_meta = rec["store"][str(tmp_path / "rollouts" / "control-c1" / "0.npz")]
    assert c1_meta["c1_shifts_hash"] == budget["c1_shifts"]["hash"]
    assert c1_meta["q_law"] == controls.Q_LAW
    # resumed: nothing re-tuned or re-run; c1's shifts are recomputed, and the unchanged
    # record keeps its first 'written' time in the file and the manifest (review of the
    # controls wiring, finding 5: R23's "written before c1 is rolled out")
    first = "2000-01-01T00:00:00"
    io.write_json(tmp_path / "controls" / "c1_shifts.json", dict(shifts_file, written=first))
    budget["c1_shifts"]["written"] = first
    calls.clear()
    ps.stage_controls(_args())
    assert not [c for c in calls if c[0] in ("run", "tune_cv", "tau_c_k1")]
    assert [c[0] for c in calls] == ["c1_shifts", "viability", "T64"]
    assert io.read_json(tmp_path / "controls" / "c1_shifts.json")["written"] == first
    assert rec["manifest"].budget["c1_shifts"]["written"] == first
    # an as-built tuning (no q_law, tau_c fitted to the key's d_b) is redone under R22 / R26
    io.write_json(tmp_path / "controls" / "tuning.json",
                  dict(c_v=0.9, rates={"0.9": 1.0}, tau_c=0.788, d_b_key=61.8))
    calls.clear()
    ps.stage_controls(_args())
    assert calls[0] == ("tune_cv",) and any(c[0] == "tau_c_k1" for c in calls)
    tun = io.read_json(tmp_path / "controls" / "tuning.json")
    assert "d_b_key" not in tun and tun["tau_c"] == 1.2
    # a changed shift reruns c1 (and only c1)
    monkeypatch.setattr(controls, "c1_shifts", lambda bank_, cs, kt, car, c_v: {
        c.cell_id: dict(shift=30.0, lead=15.0, d_b_q=115.0, d_b_key=100.0, status="ok")
        for c in cs})
    calls.clear()
    ps.stage_controls(_args())
    assert [c[1] for c in calls if c[0] == "run"] == ["c1"]
    new = io.read_json(tmp_path / "controls" / "c1_shifts.json")
    assert new["written"] != first and rec["manifest"].budget["c1_shifts"] == new
    c1_meta = rec["store"][str(tmp_path / "rollouts" / "control-c1" / "0.npz")]
    assert c1_meta["c1_shifts_hash"] == new["hash"]


def test_R21_controls_and_dryrun_refuse_a_pre_R21_design(ps, monkeypatch):
    """A real-lap cell without its recorded step before x0 (a design built before R21)
    stops the keys, controls and dry-run stages with the rebuild instruction."""
    bank, cells = _design()
    real = Cell(3, 0, 3, ("D",), "real", 60.0, 300.0, W=400, P=4.0,
                x0=(1000.0, 0.1, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0), state_source="S1:lap1@D/1")
    assert real.phase_rule() == "missing"
    with pytest.raises(SystemExit, match="stimuli --real --force"):
        ps._require_r21(cells + [real], "the controls stage")
    ps._require_r21(cells, "the controls stage")                    # designed cells only
    started = Cell(4, 0, 4, ("E1",), "real", 60.0, 300.0, W=400, P=4.0, x0=real.x0,
                   state_source="G1:lap1@E1/x", flags=(stimuli.RUN_START_FLAG,))
    ps._require_r21(cells + [started], "the controls stage")       # run start: s-shift


# ------------------------------------------------------------------------- dry-run verdicts


def _decision(pred, verdict, **det):
    return A.Decision(pred, "x", "", verdict, None, det)


def _stub_verdicts(ps, monkeypatch, tmp_path, verdicts: dict, f6=(1.0,), vacuous=False):
    """Stub every analysis call of ``_control_verdicts``; ``verdicts[agent][test]`` is what
    the rule returns for that control."""
    for name in (*controls.CONTROLS, "T64"):
        p = tmp_path / f"control-{name}.npz"
        p.touch()
    monkeypatch.setattr(ps, "rollout_path_of", lambda agent, ts=0: tmp_path / f"{agent}.npz")
    monkeypatch.setattr(ps, "_manifest", lambda: SimpleNamespace(snapshots=[]))
    monkeypatch.setattr(ps, "_finals", lambda m: {})
    monkeypatch.setattr(io, "CONTROLS_DIR", tmp_path)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: (
        {"det": {}}, dict(c_v=0.9, tau_c=1.2 if "c3" in str(p) else None)))
    monkeypatch.setattr(ps, "_current_control_wants", lambda bank, cells, cache: {
        n: dict(c_v=0.9, tau_c=1.2) if n == "c3" else dict(c_v=0.9)
        for n in (*controls.CONTROLS, "T64")})
    monkeypatch.setattr(ps, "_run_table", lambda agent, *a, **k: {
        "agent": np.array([agent]), "crashed": np.array([False])})
    monkeypatch.setattr(ps, "_f6_scales", lambda bank, cells: list(f6))
    monkeypatch.setattr(A, "pair_values", lambda rt, kt, reg, b: dict(
        agent=str(rt["agent"][0]), q10=next(iter(reg.values())).q10))
    seen = {}

    def rule(test):
        def f(pairs, *a):                      # pairs (P1-P2b) or the run table (P5)
            who = pairs["agent"]
            agent = who if isinstance(who, str) else str(who[0])
            seen.setdefault(agent, {})[test] = pairs.get("q10")
            return _decision(test, verdicts.get(agent[8:], {}).get(test, "not testable"))
        return f

    for fn, test in (("p1a", "P1a"), ("p1b", "P1b"), ("p2", "P2"), ("p2b", "P2b"), ("p5", "P5")):
        monkeypatch.setattr(A, fn, rule(test))
    monkeypatch.setattr(A, "p3_fit", lambda rt, kt, tables: str(rt["agent"][0]))
    monkeypatch.setattr(A, "brake_share", lambda rt, kt: 1.0)
    monkeypatch.setattr(A, "p3", lambda agent, share: _decision(
        "P3", verdicts[agent[8:]].get("P3", "inconclusive"), winner="M_idx"))
    monkeypatch.setattr(A, "cond_table", lambda rt, kt, dood=None: {"agent": rt["agent"]})
    monkeypatch.setattr(A, "p4_classify", lambda cond, gen: SimpleNamespace(
        agent=str(cond["agent"][0]), label="", C={}))
    monkeypatch.setattr(A, "p4", lambda res: _decision(
        "P4", verdicts[res.agent[8:]]["P4"], vacuous=vacuous))
    monkeypatch.setattr(A, "tape_check", lambda rt, kt, prec: dict(
        got=verdicts["T" if prec == "f32" else "T64"], details=dict(crash_cells=[])))
    monkeypatch.setattr(A, "viability_typing", lambda rt, kt: dict(
        passed=verdicts["c1"]["viability"], n_crashed=3,
        n_resolved_overspeed=3 if verdicts["c1"]["viability"] else 2))
    monkeypatch.setattr(controls, "offline_match_runs", lambda *a: dict(passed=True, n=5, n_fail=0))
    monkeypatch.setattr(controls, "c2_planted_law", lambda *a: dict(
        passed=verdicts["c2"]["planted"], n=10, n_fail=0 if verdicts["c2"]["planted"] else 2))
    monkeypatch.setattr(controls, "c3_checks", lambda *a: dict(
        k_le_1="pass", entry="pass", overspeed_crashes="none"))
    monkeypatch.setattr(controls, "lock_radius", lambda tables: 8.16)
    monkeypatch.setattr(controls, "car_tables", lambda tables: "CAR")
    return seen


_AGENTS_OK = {
    "T": dict(errors="within_tau", termination="3 crashes", regret="max rel 2.00e-05"),
    "T64": dict(errors="within_tau", termination="none", regret="zero"),
    "Q": dict(P1a="cautious but lawful", P3="confirm", P4="class 4", P5="confirm"),
    "Qp": {},
    "c1": dict(P2="confirm", P2b="confirm", viability=True),
    "c2": dict(P1a="cautious but lawful", P1b="inconclusive", planted=True),
    "c3": dict(P3="refute"),
    "K": dict(P4="class 2"),
}


def _verdicts(ps, monkeypatch, tmp_path, changes=None, **kw):
    v = {k: dict(d) for k, d in _AGENTS_OK.items()}
    for k, d in (changes or {}).items():
        v[k].update(d)
    seen = _stub_verdicts(ps, monkeypatch, tmp_path, v, **kw)
    sup = SimpleNamespace(regimes={"G1": SimpleNamespace(q1=12.02, q10=21.3)})
    bank, cells = _design()
    out = ps._control_verdicts(bank, cells, {}, "KEYS", None, 0.1, sup, _Log())
    return out, seen


def test_R20_to_R28_dry_run_required_verdicts(ps, monkeypatch, tmp_path):
    """The controls' statuses as the rulings amend protocol 7: T32 blocks on its measure
    check only (crashes and regret reported); T64 needs all three; c2's P1a must equal Q's
    (judged under Q's regime) and its P1b is reported; c3's P3 is expected, a miss a power
    finding; c1 needs R28's viability typing; K's class 1 at the registered F6 asks for the
    halving, which then requires class 2 or 3 (a vacuous class 2 flagged)."""
    out, seen = _verdicts(ps, monkeypatch, tmp_path)
    assert {k: v["status"] for k, v in out.items()} == {k: "pass" for k in _AGENTS_OK}
    assert out["T"]["got"]["termination"] == "3 crashes" and out["T"]["required"] == {
        "errors": ["within_tau"]}
    assert out["c2"]["required"]["P1a"] == ["cautious but lawful"]       # Q's own verdict
    assert seen["control-c2"]["P1a"] == 21.3 and seen["control-c2"]["P1b"] == 20.0
    assert out["c2"]["got"]["P1b"] == "inconclusive" and out["c2"]["reported"] == ["P1b"]
    assert out["c3"]["got"]["P3"] == "refute:M_idx" and "power_finding" not in out["c3"]["details"]
    # misses
    bad, _ = _verdicts(ps, monkeypatch, tmp_path, dict(
        T=dict(errors="1 beyond tau"), T64=dict(regret="max rel 3.00e-06"),
        c2=dict(P1a="confirm"), c3=dict(P3="inconclusive"), c1=dict(viability=False),
        Q=dict(P1a="confirm")))
    st = {k: v["status"] for k, v in bad.items()}
    assert st["T"] == st["T64"] == st["c1"] == "FAIL"
    assert st["c2"] == "pass" and bad["c2"]["required"]["P1a"] == ["confirm"]   # equals Q's
    assert st["c3"] == "pass" and "power finding" in bad["c3"]["details"]["power_finding"]
    assert bad["c1"]["got"]["viability_typing"].startswith("1 of 3")
    other, _ = _verdicts(ps, monkeypatch, tmp_path, dict(c2=dict(P1a="refute")))
    assert other["c2"]["status"] == "FAIL"
    planted, _ = _verdicts(ps, monkeypatch, tmp_path, dict(c2=dict(planted=False)))
    assert planted["c2"]["status"] == "FAIL"
    # K (R27): class 1 at the registered F6 asks for the one halving (blocking)
    k1, _ = _verdicts(ps, monkeypatch, tmp_path, dict(K=dict(P4="class 1")))
    assert k1["K"]["status"] == "halve F6" and "--f6-scale 0.5" in k1["K"]["details"]["P4"]["action"]
    k2, _ = _verdicts(ps, monkeypatch, tmp_path, dict(K=dict(P4="class 2")), f6=(0.5,),
                      vacuous=True)
    assert k2["K"]["status"] == "pass" and "vacuously" in k2["K"]["details"]["P4"]["flag"]
    assert "class-1 path" in k2["K"]["details"]["P4"]["statement"]
    k3, _ = _verdicts(ps, monkeypatch, tmp_path, dict(K=dict(P4="class 1")), f6=(0.5,))
    assert k3["K"]["status"] == "FAIL"


def test_dry_run_judges_only_rollouts_made_with_the_current_inputs(ps, monkeypatch, tmp_path):
    """Review of the controls wiring (finding 2): the dry run compares each control's
    rollout meta with the inputs the controls stage would use now (``_control_want``: c_v,
    design, frozen code, Q's law, tau_c, c1's shift hash, T's tapes). A file left from other
    inputs (a skipped c3 keeps the as-built tau_c 0.788; a failed T64 keeps an older design's
    file) is 'stale rollouts', which blocks; it is never judged."""
    _stub_verdicts(ps, monkeypatch, tmp_path, {k: dict(d) for k, d in _AGENTS_OK.items()})
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": {}}, dict(
        c_v=0.9, tau_c=0.788 if "c3" in str(p) else None,
        design_hash="old" if "T64" in str(p) else "new")))
    monkeypatch.setattr(ps, "_current_control_wants", lambda bank, cells, cache: {
        n: dict(c_v=0.9, tau_c=1.2) if n == "c3" else
        dict(design_hash="new") if n == "T64" else dict(c_v=0.9)
        for n in (*controls.CONTROLS, "T64")})
    judged = []
    run_table = ps._run_table
    monkeypatch.setattr(ps, "_run_table", lambda agent, *a, **k: judged.append(agent) or
                        run_table(agent, *a, **k))
    sup = SimpleNamespace(regimes={"G1": SimpleNamespace(q1=12.02, q10=21.3)})
    bank, cells = _design()
    out = ps._control_verdicts(bank, cells, {}, "KEYS", None, 0.1, sup, _Log())
    assert out["c3"]["status"] == out["T64"]["status"] == "stale rollouts"
    assert out["c3"]["stale"] == {"tau_c": [0.788, 1.2]}
    assert out["T64"]["stale"] == {"design_hash": ["old", "new"]}
    assert "control-c3" not in judged and "control-T64" not in judged
    assert out["Q"]["status"] == out["K"]["status"] == "pass"
    # the controls stage's resume keys and the dry run's are the one helper
    w = ps._control_want("c3", design="d", code="c", c_v=0.9, tau_c=1.2, shifts_hash="h",
                         tape_hash="t")
    assert w == dict(c_v=0.9, design_hash="d", frozen_code_hash="c", q_law=controls.Q_LAW,
                     tau_c=1.2)
    assert ps._control_want("T64", design="d", code="c", tape_hash="t") == dict(
        design_hash="d", frozen_code_hash="c", tape_keys_hash="t")
    assert ps._control_want("c1", design="d", code="c", c_v=0.9, shifts_hash="h")[
        "c1_shifts_hash"] == "h"
    # before any controls stage (no tuning.json): every saved file is stale
    monkeypatch.setattr(ps, "_current_control_wants", lambda bank, cells, cache: {})
    none = ps._control_verdicts(bank, cells, {}, "KEYS", None, 0.1, sup, _Log())
    assert {v["status"] for v in none.values()} == {"stale rollouts"}


def test_knn_fallback_logs_k5_on_the_condition_a_cells_when_k1_passes(ps, monkeypatch):
    """Review of the controls wiring (finding 4): when K at k = 1 passes the gate, its runs
    are kept and K at k = 5 is also run, on the condition (a) cells only, for its logged
    C(a); when k = 1 fails, k = 5 runs on every K cell and its runs are used."""
    bank, cells = _design()
    d = [Cell(10 + i, 0, 10 + i, ("D",), "real", 60.0, 300.0, W=400, P=4.0) for i in range(2)]
    cond = {10: stimuli.COND_A, 11: "f1_other"}
    monkeypatch.setattr(ps, "_control_cells", lambda name, spec, bank_, cells_: d)
    monkeypatch.setattr(ps, "_cond_a_cells", lambda bank_, cs: [c for c in cs
                                                                 if cond[c.cell_id] == stimuli.COND_A])
    monkeypatch.setattr(A, "CachedKeys", lambda *a, **k: "KEYS")
    monkeypatch.setattr(A, "key_table", lambda cs, bank_, keys: {"cell": np.array([10, 11])})
    calls = []

    def run_one(name, spec, bank_, cells_, car, c_v, tau_c, cache, k=1, shifts=None, only=None):
        calls.append((k, None if only is None else sorted(only)))
        return {"k": k}, dict(control=name, k=k)

    monkeypatch.setattr(ps, "_run_one_control", run_one)
    monkeypatch.setattr(ps, "_run_table", lambda agent, ts, groups, *a: groups["det"])
    monkeypatch.setattr(A, "cond_table", lambda rt, kt: rt)
    c_a = {}
    monkeypatch.setattr(A, "completion_by_condition", lambda ct: ({A.COND_A: c_a[ct["k"]]}, {}, []))
    log = _Log()
    c_a.update({1: 0.95, 5: 0.8})
    arrays, meta = ps._knn_fallback(({"k": 1}, dict(control="K", k=1)), None, bank, cells,
                                    "CAR", 0.9, "CACHE", None, log)
    assert arrays == {"k": 1} and calls == [(5, [10])]
    assert meta["C_a_k1"] == 0.95 and meta["C_a_k5"] == 0.8 and not meta["fallback"]
    calls.clear()
    c_a.update({1: 8 / 9, 5: 1.0})
    arrays, meta = ps._knn_fallback(({"k": 1}, dict(control="K", k=1)), None, bank, cells,
                                    "CAR", 0.9, "CACHE", None, log)
    assert arrays == {"k": 5} and calls == [(5, None)] and meta["fallback"] and meta["k"] == 5


# ------------------------------------------------------------------------- stimuli stage


def test_R21_append_real_loads_the_lap_steps_and_keeps_the_f6_scale(ps, monkeypatch, tmp_path):
    """``_append_real`` reads the states with their recorded steps (``load_real_states``)
    and hands them to ``build_real_design``; the F6 scale defaults to the saved design's, so
    a rebuild never undoes protocol 7's halving."""
    got = {}
    (tmp_path / stimuli.REAL_STATES_FILE).write_text("{}")
    monkeypatch.setattr(io, "STIMULI_DIR", tmp_path)
    monkeypatch.setattr(stimuli, "load_real_states", lambda d: got.setdefault("dir", d) and {
        "S1": {"D/1": "attached"}})
    monkeypatch.setattr(stimuli, "build_real_design", lambda bank, cells, states, swap, f6: (
        got.update(states=states, f6=f6) or (bank, cells)))
    ps._append_real("BANK", [], 0.5)
    assert got == dict(dir=tmp_path, states={"S1": {"D/1": "attached"}}, f6=0.5)
    # the saved design's F6 scale
    assert ps._saved_f6_scale() == 1.0                                   # no saved design
    (tmp_path / "cells.json").write_text("{}")
    rows = [SimpleNamespace(source={"f6_scale": 0.5}), SimpleNamespace(source={})]
    monkeypatch.setattr(stimuli, "load_design", lambda d: (SimpleNamespace(stimuli=rows), []))
    assert ps._saved_f6_scale() == 0.5
    rows.append(SimpleNamespace(source={"f6_scale": 1.0}))
    with pytest.raises(SystemExit, match="several scales"):
        ps._saved_f6_scale()
