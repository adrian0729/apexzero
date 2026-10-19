"""Regression tests for the fixes made after the three reviews of the psychophysics harness
(fidelity, numerics, hygiene). Each test names the finding it pins down. No agent is rolled
out: keys and runs are synthetic trajectories, banks are hand-built strips."""

from __future__ import annotations

import dataclasses
import stat
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.interp import agents, controls, io, measures, rollout, stimuli
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import (
    COND_A,
    Cell,
    CornerSpec,
    Geometry,
    Landmarks,
    Stimulus,
    StimulusBank,
)
from apexzero.ocp.window import WindowSpec

B = 0.075
REG = A.Regime(12.0, 20.0)


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollout-side code runs in float32; other test modules switch x64 on at import."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- a mini design


def _mini_bank() -> tuple[StimulusBank, list[Cell]]:
    """One R40/90 left corner at S_C and one width-12 null straight, as real StimulusBank rows
    (so keys are found by ``answer_key.expected_key_id``); the corner cell (A1-like, v0 60
    m/s, D 300 m, W 200) points to the null cell (D 600 m, W 200)."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, ps = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A",), 1, -1, pc.landmarks,
                      geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, ps.landmarks, geo, None)]
    bank = StimulusBank(np.stack([pc.kappa, ps.kappa]), np.stack([pc.w_left, ps.w_left]),
                        np.stack([pc.w_right, ps.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), "A1", 60.0, 300.0, W=200, P=3.0, null_cell=1),
             Cell(1, 1, 1, ("G",), "G", 60.0, 600.0, W=200, P=3.0)]
    return bank, cells


def _drive(s0: float, v0: float, s_brake: float | None, n: int = 201, v_low: float = 40.0,
           decel: float = 15.0) -> np.ndarray:
    """States of a scripted drive at 20 Hz: pedal state 1 until s_brake, then -0.6 while
    braking at ``decel`` down to v_low, then 0.2 (None: never brakes)."""
    x = np.zeros((n, 8))
    s, v, braked = s0, v0, False
    for k in range(n):
        braking = s_brake is not None and s >= s_brake and v > v_low
        braked |= braking
        pedal = -0.6 if braking else (0.2 if braked else 1.0)
        x[k] = [s, 0.0, 0.0, v, 0.0, 0.0, 0.0, pedal]
        if braking:
            v = max(v - decel * 0.05, v_low)
        s += v * 0.05
    return x


def _key_record(bank, cell, kind: str, x: np.ndarray, J: float = 50.0) -> ak.KeyRecord:
    W = int(cell.W)
    a = np.stack([np.zeros(W), x[1:, 7]], axis=1)
    return ak.KeyRecord(key_id=ak.expected_key_id(bank, cell, kind), kind=kind, status="verified",
                        stim_id=cell.stim_id, stim_name=bank.stimuli[cell.stim_id].name,
                        cell_id=cell.cell_id, phase=0, direction=bank.stimuli[cell.stim_id].direction,
                        x0=x[0].copy(), W=W, spec=dataclasses.asdict(WindowSpec(steps=W)),
                        alpha_max=0.17, shrink=0.0, J=J, x=x, a=a,
                        replay={"verified": True, "survives": True, "rel": 0.0},
                        labels={"approach": cell.approach}, measures={"d_resp": np.nan})


def _det_group(rows: list[tuple[Cell, np.ndarray]]) -> dict:
    n = len(rows)
    K = rows[0][1].shape[0]
    x = np.stack([r[1] for r in rows]).astype(np.float32)
    a = np.zeros((n, K - 1, 2), np.float32)
    a[:, :, 1] = x[:, 1:, 7]
    return dict(cell=np.array([c.cell_id for c, _ in rows]), stim=np.array([c.stim_id for c, _ in rows]),
                pair=np.array([c.pair_id for c, _ in rows]), phase=np.zeros(n, int),
                sample=np.full(n, -1), steer_sign=np.ones(n, int), W=np.full(n, K - 1),
                s0=x[:, 0, 0].astype(np.float64), valid=np.ones((n, K), bool), x=x,
                term_step=np.full(n, -1), trunc_step=np.full(n, -1), a=a, a_mean=a.copy(),
                reward=np.zeros((n, K - 1), np.float32))


# ------------------------------------------------------------------------- fidelity B1


def test_B1_keys_get_a_catch_trial_onset_so_dd_resp_is_finite(tmp_path):
    """Fidelity B1: a run table built through CachedKeys has a finite Δd_resp: the key's
    response onset is measured against the null key (same v0, width, W) when the key's
    measures are recomputed from its cached trajectory."""
    bank, (corner, null) = _mini_bank()
    cache = ak.KeyCache(tmp_path / "keys")
    cache.put(_key_record(bank, corner, "nominal", _drive(700.0, 60.0, 880.0)))
    cache.put(_key_record(bank, null, "null", _drive(400.0, 60.0, None)))
    keys = A.CachedKeys(bank, [corner, null], cache)
    view = keys.nominal(corner, 0)
    assert np.isfinite(view.metrics.d_resp)                     # recomputed with the null key
    assert view.metrics.d_resp == pytest.approx(1000.0 - 880.0, abs=4.0)
    det = _det_group([(corner, _drive(700.0, 60.0, 870.0)), (null, _drive(400.0, 60.0, None))])
    rt = A.run_table("G1", 1, {"det": det}, [corner, null], bank, keys, None)
    i = int(np.flatnonzero(rt["cell"] == 0)[0])
    assert np.isfinite(rt["d_resp"][i]) and np.isfinite(rt["dd_resp"][i])
    assert rt["dd_resp"][i] == pytest.approx(10.0, abs=4.0)    # 10 m earlier than the key
    # The run-level column reaches the pair value P2(a) reads.
    kt = A.key_table([corner, null], bank, keys)
    pv = A.pair_values(rt, kt, {"G1": REG}, B)
    # one run in the pair: the 5-of-8 onset rule needs ceil(5/8) = 1 run
    assert np.isfinite(pv["dd_resp"][pv["pair"] == 0][0])


def test_runs_from_another_design_are_refused():
    """Hygiene M1: a run whose recorded W or s0 is not its cell's is not paired with keys."""
    bank, cells = _mini_bank()
    det = _det_group([(cells[0], _drive(700.0, 60.0, 870.0))])
    A.check_runs_match_cells(det, {c.cell_id: c for c in cells})          # matches
    bad = dict(det, s0=det["s0"] + 10.0)
    with pytest.raises(ValueError, match="do not match the current design"):
        A.check_runs_match_cells(bad, {c.cell_id: c for c in cells})
    other = [dataclasses.replace(cells[0], W=400), cells[1]]
    with pytest.raises(ValueError, match="W 200 != 400"):
        A.run_table("G1", 1, {"det": det}, other, bank, None, None)


class _PhaseKeys:
    """Nominal key v_e* 30 at u = 0, 33 at phase 1 (a phase-matched key)."""

    def __init__(self):
        self.cells: dict = {}

    def nominal(self, cell, phase):
        v = 33.0 if phase == 1 else 30.0
        m = measures.ResponseMetrics(v_e=v, v_min=25.0, d_b=100.0 + phase, v_b=60.0,
                                     has_brake=True)
        return A.KeyView(None, m, True, "verified")

    def backoff(self, cell):
        return None

    def anchored(self, cell, agent, ts, phase=0):
        return None

    def null(self, cell):
        return None

    def socp(self, cell):
        return None

    def no_return(self, *a):
        return None


def test_speed_errors_use_the_u0_key_and_onsets_the_phase_key(monkeypatch):
    """Fidelity minors 9-10: e_ve is against the nominal u = 0 key (protocol 2.2), Δd_b and
    the P2b key values against the phase-matched key."""
    bank, cells = _mini_bank()
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: SimpleNamespace())
    monkeypatch.setattr(measures, "response_metrics", lambda *a, **k: measures.ResponseMetrics(
        v_e=33.0, v_min=25.0, d_b=95.0, has_brake=True))
    monkeypatch.setattr(measures, "key_errors", lambda m, mk, *a, **k: measures.KeyErrors(
        e_ve=m.v_e / mk.v_e - 1.0, dd_b=m.d_b - mk.d_b))
    det = dict(cell=np.array([0]), phase=np.array([1]), sample=np.array([-1]),
               steer_sign=np.array([1]), W=np.array([200]), s0=np.array([699.25]))
    rt = A.run_table("G1", 1, {"det": det}, cells, bank, _PhaseKeys(), None, check_design=False)
    assert rt["e_ve"][0] == pytest.approx(33.0 / 30.0 - 1.0)   # u = 0 key
    assert rt["dd_b"][0] == pytest.approx(95.0 - 101.0)        # phase-1 key
    assert rt["v_e_star_k"][0] == 33.0 and rt["d_b_star_k"][0] == 101.0


# ------------------------------------------------------------------------- fidelity B2


def _tier_d_bank():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    geo = Geometry(20.0, 90.0, np.nan, 20.0, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
    stims = [Stimulus(0, "D/3/F1/real/real/real", "hockenheim", ("D",), 1, -1, lm, geo, None,
                      source={"corner_index": 3, "condition": COND_A})]
    bank = SimpleNamespace(stimuli=stims, w_left=np.full((1, 600), 6.0),
                           w_right=np.full((1, 600), 6.0), ds=2.0)
    x0 = (700.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 1.0)
    cells = [Cell(0, 0, 0, ("D",), "real", 30.0, 300.0, W=400, P=1.5, x0=x0,
                  runners=("S1", "G1", "G2", "G3")),
             Cell(1, 0, 1, ("D",), "real", 30.0, 300.0, W=400, P=1.5, x0=x0, runners=("S2",)),
             Cell(2, 0, 2, ("D",), "real", 30.0, 300.0, W=400, P=1.5, x0=x0, runners=("S3",))]
    return bank, cells


class _SocpOnlyS1:
    """Tier D: s-OCP keys solved only from S1's state (protocol 3.3), none for S2 / S3."""

    def nominal(self, cell, phase):
        return None

    def backoff(self, cell):
        return None

    def socp(self, cell):
        if "S1" not in cell.runners:
            return None
        return A.KeyView(None, measures.ResponseMetrics(v_e=29.0, v_min=20.0, d_b=50.0, v_b=30.0,
                                                        has_brake=True))


def test_B2_tier_d_braking_class_and_v_min_star_come_from_the_s1_socp():
    """Fidelity B2: every runner's cell of a tier D strip shares the S1-state s-OCP's braking
    class and v_min*; the condition table never takes the last runner's empty row."""
    bank, cells = _tier_d_bank()
    kt = A.key_table(cells, bank, _SocpOnlyS1())
    assert kt["braking_socp"].tolist() == [True, True, True]
    assert kt["v_min_socp"].tolist() == [20.0, 20.0, 20.0]
    assert kt["socp_cell"].tolist() == [0, 0, 0]
    for agent, cid in (("S2", 1), ("S3", 2)):
        runs = A._to_table([dict(agent=agent, timesteps=0, cell=cid, mode="det", completed=True,
                                 v_min=19.0, v_peak=30.0) for _ in range(4)],
                           ("agent", "timesteps", "cell", "mode", "completed", "v_min", "v_peak"))
        ct = A.cond_table(runs, kt)
        assert ct["braking"].tolist() == [True] and ct["v_min_star"].tolist() == [20.0]
        assert A.completion_by_condition(ct)[2] == [3]           # corner 3 is braking
    # A key table as the old code built it (S3's empty row last) still gives the S1 class.
    old = {k: np.array(v, copy=True) for k, v in kt.items()}
    old["braking_socp"][1:] = False
    old["v_min_socp"][1:] = np.nan
    runs = A._to_table([dict(agent="S3", timesteps=0, cell=2, mode="det", completed=True,
                             v_min=19.0, v_peak=30.0)], ("agent", "timesteps", "cell", "mode",
                                                         "completed", "v_min", "v_peak"))
    ct = A.cond_table(runs, old)
    assert bool(ct["braking"][0]) and ct["v_min_star"][0] == 20.0


# ------------------------------------------------------------------------- fidelity B3


def _cond(c_single: float, c_f6: float, d_single: float, d_f6: float) -> dict:
    rows = []
    for cn in stimuli.D_CONDITIONS:
        c, d = 1.0, 1.0
        if cn == "F1/straight/real/real":
            c, d = c_single, d_single
        elif cn.startswith("F6/"):
            c, d = c_f6, d_f6
        rows.append(dict(agent="S2", timesteps=0, corner=0, condition=cn, braking=True,
                         completes=4 * int(c), complete=bool(c >= 1.0), v_min=20.0, v_peak=80.0,
                         v_min_region=80.0, d_ood=d, v_min_star=20.0, swap_j=-1))
    return A._to_table(rows, A.COND_COLUMNS)


def test_B3_missing_d_ood_is_not_testable_not_class_2():
    """Fidelity B3: with a failing single-factor condition and no d_OOD, P4 is 'not
    testable' (it was class 2 by a vacuous rule); with d_OOD it is classified."""
    no = _cond(0.0, 0.0, np.nan, np.nan)
    r = A.p4_classify(no, None)
    assert r.cls == -1 and A.p4(r).verdict == "not testable"
    assert r.details["d_ood_missing"]
    # The same specialist with d_OOD: every F6 perturbation nearer than the failing
    # condition also fails -> class 1 (fragile), not class 2.
    assert A.p4_classify(_cond(0.0, 0.0, 2.0, 1.0), None).cls == 1
    # Context keying proper: far F6 perturbations complete.
    assert A.p4_classify(_cond(0.0, 1.0, 2.0, 2.5), None).cls == 2
    # Nothing fails: d_OOD is not needed (no class 1 / 2 question).
    ok = _cond(1.0, 1.0, np.nan, np.nan)
    assert A.p4_classify(ok, None).cls >= 3
    # The recipe claim ignores untestable seeds.
    rs = [A.P4Result(a, c, "", {}, 1.0, 0.0, 0.0, 0.0) for a, c in (("S1", 2), ("S2", -1),
                                                                   ("S3", -1))]
    assert A.p4_recipe(rs).verdict == "inconclusive"


# ------------------------------------------------------------------------- fidelity majors


def test_M8_null_straight_completion_is_no_termination_within_W():
    """Fidelity M8: on a tier G straight, completion is surviving the window, not reaching
    the virtual corner's s_c1 + 250 m (1,250 m of a 4 km straight)."""
    bank, (_, null) = _mini_bank()
    st = bank.stimuli[1]
    marks = measures.make_marks(st, bank.w_left[1], bank.w_right[1], 400.0, 400)
    x = _drive(400.0, 60.0, None, n=401)
    alive = measures.Trajectory(source="agent", t=np.arange(401) * 0.05, x=x,
                                valid=np.ones(401, bool))
    assert measures.response_metrics(alive, marks).completed
    k = 300                                                     # a crash past S_C + 250 m
    assert x[k + 1, 0] > 1250.0
    crashed = dataclasses.replace(alive, term_step=k, valid=np.r_[np.ones(k + 2, bool),
                                                                  np.zeros(400 - k - 1, bool)],
                                  flags=np.zeros(400, np.uint8))
    m = measures.response_metrics(crashed, marks)
    assert m.crashed and not m.completed and not m.time_censored
    assert m.v_max == pytest.approx(60.0)                       # v_top covariate source


def _gate_inputs(sims_runs, gates, demote=False):
    keys = sims_runs[1]
    runs = {"G1": {**sims_runs[0], "agent": np.full(A._n(sims_runs[0]), "G1")},
            "G3": {**sims_runs[0], "agent": np.full(A._n(sims_runs[0]), "G3")}}
    return A.AnalysisInputs(b=B, keys=keys, regimes={"G1": REG, "G3": REG}, runs=runs,
                            tables=None, gates=gates, demote_p3=demote)


@pytest.fixture(scope="module")
def lawful():
    import importlib

    keys = importlib.import_module("test_interp_analysis").KEYS     # the synthetic design
    return A.simulate_runs("lawful", keys, A._dry_rng(7), n_native=0, n_matched=0), keys


def test_M9_a_final_that_fails_the_gate_is_not_read(lawful):
    """Fidelity M9: P1-P3 are read only on snapshots that pass protocol 8's gate."""
    out = A.decisions(_gate_inputs(lawful, {"G1": {0: False}, "G3": {0: True}}))
    got = {(d.prediction, d.agent): d for d in out}
    assert got[("P1a", "G1")].verdict == "not testable"
    assert got[("P1a", "G1")].details["reason"] == "checkpoint gate failed"
    assert got[("P1a", "G1")].details["ungated_verdict"] == got[("P1a", "G3")].verdict
    assert got[("P1a", "G3")].verdict != "not testable"
    # No gate given and none computable (no tier G runs): not read either.
    out = A.decisions(_gate_inputs(lawful, {}))
    assert all(d.verdict == "not testable" for d in out if d.prediction in A.CONFIRMATORY_G)


def test_M5_failed_separation_check_demotes_p3(lawful):
    """Fidelity M5: a failed pre-tag P3 separation check moves G3's P3 to secondary."""
    gates = {"G1": {0: True}, "G3": {0: True}}
    p3 = {d.agent: d for d in A.decisions(_gate_inputs(lawful, gates)) if d.prediction == "P3"}
    assert p3["G3"].family == "primary"
    p3 = {d.agent: d for d in A.decisions(_gate_inputs(lawful, gates, demote=True))
          if d.prediction == "P3"}
    assert p3["G3"].family == "secondary" and "demoted" in p3["G3"].details


def test_M6_g3_crash_units_join_p8(tmp_path):
    """Fidelity M6: G3's own R_DEV crashes (from its deterministic real laps) become P8
    units by the same construction, with sets from G3's frozen q1 / q10 (and, by ruling R2,
    every agent's units the same way)."""
    tr = stimuli.real_track("Norisring")
    L = float(tr.length)
    x = np.zeros((50, 8))
    x[:, 0] = np.linspace(0.0, L + 910.0, 50)
    crashed = SimpleNamespace(x=x, valid=np.ones(50, bool), term_step=48, length=L)
    clean = SimpleNamespace(x=x, valid=np.ones(50, bool), term_step=-1, length=L)
    pairs = stimuli.crash_pairs_from_laps("G3", {"Norisring": crashed, "Hockenheim": clean,
                                                 "Austin": crashed})
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.agent, p.track, p.lap) == ("G3", "Norisring", 2)
    assert p.s_crash == pytest.approx(910.0, abs=0.1)
    c = stimuli.crash_corner(tr, 910.0)
    assert stimuli.crash_unit_R_seg(p) == pytest.approx(float(c.r_min))
    f = tmp_path / "extra.json"
    io.write_json(f, dict(pairs=[dict(dataclasses.asdict(p), label_R=99.0)]))  # old label: ignored
    allp = stimuli.crash_pairs(f)
    assert allp[:len(stimuli.CRASH_PAIRS)] == stimuli.CRASH_PAIRS and allp[-1] == p
    rt = {"G3": A.Regime(10.2, 14.5), "G2": A.Regime(11.1, 17.7)}
    assert A.p8_set("G3", 12.0, rt) == "tail"
    assert A.p8_set("G3", 30.0, rt) == "bulk"
    assert A.p8_set("G2", 30.0, rt) == "bulk"                    # R2: no named sets
    assert A.p8_set("S9", 30.0, rt) == "other"


def test_M7_knife_edge_unit_stays_unreproduced():
    """Fidelity M7: 3.2's YasMarina diagnosis labels a knife-edge unit unreproduced in P8."""
    rows = []
    for v, det in (("real", 4), ("isolated", 4), ("twin", 4)):
        rows.append(dict(agent="G2", unit="G2/YasMarina/1369", set="tail", version=v,
                         fidelity_ok=True, det_crashes=det, det_n=4, nat_k=0, nat_n=16,
                         e_ve=0.2, det_overspeed=det))
    lad = A._to_table(rows, A.LADDER_COLUMNS)
    u = A.p8_units(lad)
    assert bool(u["reproduced"][0]) and u["label"][0] == "single-corner"
    u = A.p8_units(lad, knife_edge=("G2/YasMarina/1369",))
    assert not bool(u["reproduced"][0]) and u["label"][0] == "knife-edge (unreproduced)"


def test_M1_controls_T_errors_and_Q_offline_match(monkeypatch):
    """Fidelity M1: T's 'every error within tau' and Q's 'offline_match' are computed."""
    keys = A._to_table([dict(cell=0, key_status="verified", v_e_star=30.0, v_min_star=25.0,
                             d_b_star=100.0, tau_ve=0.2, tau_vmin=0.2, tau_db=1.0)],
                       ("cell", "key_status", "v_e_star", "v_min_star", "d_b_star", "tau_ve",
                        "tau_vmin", "tau_db"))
    runs = A._to_table([dict(cell=0, mode="det", phase=0, v_e=30.1, v_min=25.0, d_b=100.5)],
                       ("cell", "mode", "phase", "v_e", "v_min", "d_b"))
    te = A.tape_errors(runs, keys)
    assert te["passed"] and te["n"] == 1
    far = {**runs, "d_b": np.array([102.0])}
    te = A.tape_errors(far, keys)
    assert not te["passed"] and te["n_beyond"] == 1
    # offline_match: NaN on both sides agrees (flat cell), on one side does not.
    pred = {"e_ve": 0.0, "e_vmin": 0.01, "dd_b": np.nan, "v_b": np.nan}
    assert all(controls.offline_match(pred, {"e_ve": 0.01, "e_vmin": 0.0, "dd_b": np.nan,
                                             "v_b": np.nan}).values())
    assert not controls.offline_match(pred, {"e_ve": 0.0, "e_vmin": 0.0, "dd_b": 3.0,
                                             "v_b": 60.0})["dd_b"]
    # Over a run table: one run matches its prediction, one is 5 points too fast.
    bank, cells = _mini_bank()
    monkeypatch.setattr(controls, "target_profiles", lambda k, car, c_v: np.zeros_like(k))
    monkeypatch.setattr(controls, "offline_prediction", lambda *a, **k: dict(
        v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, e_ve=0.0, e_vmin=0.0, dd_b=0.0))
    kt = A._to_table([dict(cell=0, key_status="verified", v_e_star=30.0, v_min_star=25.0,
                           d_b_star=100.0)], ("cell", "key_status", "v_e_star", "v_min_star",
                                              "d_b_star"))
    rr = A._to_table([dict(cell=0, mode="det", phase=k, crashed=False, s0=700.0, v_e=v,
                           v_min=25.0, d_b=100.0, v_b=60.0) for k, v in ((0, 30.0), (1, 31.5))],
                     ("cell", "mode", "phase", "crashed", "s0", "v_e", "v_min", "d_b", "v_b"))
    om = controls.offline_match_runs(rr, kt, bank, cells, None, 0.95)
    assert om["n"] == 2 and om["n_fail"] == 1 and not om["passed"]
    assert om["fails"][0]["bad"] == ["e_ve"]


# ------------------------------------------------------------------------- minors


def test_stochastic_traces_keep_util_and_beta_f_in_float32():
    """Numerics m9: float16 would round beta_f 0.9999 to 1.0 and util 0.98 to 0.97998."""
    n, T = 2, 3
    tr = rollout.Trace(**{f: None for f in rollout.Trace._fields})
    tr = tr._replace(beta_f=np.full((n, T + 1), 0.9999, np.float32),
                     util_f=np.full((n, T + 1), 0.98, np.float32),
                     pedal=np.full((n, T + 1), 0.5, np.float32))
    out = rollout.trace_arrays(tr, n, "stoch")
    assert out["beta_f"].dtype == np.float32 and out["beta_f"][0, 0] < 1.0
    assert out["util_f"].dtype == np.float32 and out["util_f"][0, 0] >= np.float32(0.98)
    assert out["pedal"].dtype == np.float16


def test_checkpoints_run_the_4_phase_grid_on_p7_cells():
    """Hygiene minor 8: REPLICATION's det_phases is applied; on a 16-phase P7 cell the four
    phases are the 4-phase grid (k = 0, 4, 8, 12), not k = 0..3."""
    assert rollout.det_phase_indices(16, 4) == [0, 4, 8, 12]
    assert rollout.det_phase_indices(16, None) == list(range(16))
    assert rollout.det_phase_indices(4, 4) == [0, 1, 2, 3]
    from apexzero.interp.constants import REPLICATION

    assert REPLICATION["final"]["det_phases"] is None
    assert REPLICATION["last3"]["det_phases"] == 4


def test_knn_buffer_saves_atomically_with_its_provenance(tmp_path):
    """Hygiene minor 4: the buffer keeps its meta (snapshot, timesteps, laps)."""
    buf = controls.KnnBuffer(obs=np.zeros((3, 4), np.float32), act=np.zeros((3, 2), np.float32),
                             lap=np.arange(3), lolo_median_nn=0.5,
                             meta={"path": "runs/x/agent.pkl", "timesteps": 7, "n_laps": 50})
    buf.save(tmp_path / "knn.npz")
    back = controls.KnnBuffer.load(tmp_path / "knn.npz")
    assert back.meta == buf.meta and back.lolo_median_nn == 0.5
    assert stat.S_IMODE((tmp_path / "knn.npz").stat().st_mode) == io._FILE_MODE


def test_p7_spread_uses_one_estimator():
    """Fidelity minor 1: the agent's, Qp's and the key's phase spreads are all ddof 0."""
    v = [100.0, 102.0, 104.0, 106.0]
    assert A.phase_spread(v) == pytest.approx(float(np.std(v, ddof=0)))
    assert np.isnan(A.phase_spread([1.0]))


def test_density_uses_each_grammars_own_sd():
    """Fidelity minor 12: each grammar's features are standardised by its own sample's SD."""
    rng = np.random.default_rng(0)
    narrow, wide = rng.normal(size=(200, 5)), 10.0 * rng.normal(size=(200, 5))
    np.testing.assert_allclose(A.training_sd(wide), np.std(wide, axis=0, ddof=1))
    cs = {"cell": np.array([0]), "R_seg": np.array([20.0]), "theta_deg": np.array([90.0]),
          "l_r": np.array([20.0]), "L_c": np.array([60.0]), "v_peak": np.array([50.0])}
    cells = [SimpleNamespace(cell_id=0, pair_id=3)]
    a = A.density_table(cs, cells, {"G1": "n", "G2": "w"}, {"n": narrow, "w": wide},
                        {"n": narrow[:5], "w": wide[:5]}, None)
    b = A.density_table(cs, cells, {"G1": "n", "G2": "w"}, {"n": narrow, "w": wide},
                        {"n": narrow[:5], "w": wide[:5]}, A.training_sd(narrow))
    assert a["log_support"][1] != b["log_support"][1]           # G2's own SD changes it
    assert a["log_support"][0] == pytest.approx(b["log_support"][0])


def test_p2d_gets_a_qss_null():
    """Fidelity minor 5: the centreline QSS v_min reaches the key table (P2d testable)."""
    bank, cells = _mini_bank()
    q = A.qss_v_min(bank, cells)
    assert 0 in q and 0.0 < q[0]["v_min_qss"] < 60.0
    assert q[1]["v_min_qss"] > 60.0                             # straight: accelerates


def test_cornering_limit_replaces_the_power_band():
    """Numerics m1: v_ss and the infeasibility bound use a_y,max as a cornering limit (the
    table's power-limited entries, 0 at 315 km/h, are not lateral limits)."""
    vl = np.arange(20.0, 316.0, 5.0) / 3.6
    ay = np.where(vl < 270 / 3.6, 15.0 + 0.006 * vl**2, np.linspace(30.0, 0.0, len(vl))[
        np.arange(len(vl))])
    lim = np.where(vl < 270 / 3.6, "grip", "power")
    t = measures.ModelTables(v_dec=vl, a_dec=20.0 + 0.008 * vl**2, pedal_dec=np.full_like(vl, 0.5),
                             v_lat=vl, a_y_max=ay, alpha_max=0.17, limit_y=lim)
    assert np.all(t.a_y_corner[lim == "power"] > ay[lim == "power"])
    assert t.a_y_corner[-1] > 0.0
    np.testing.assert_allclose(t.a_y_corner[lim == "grip"], ay[lim == "grip"])
    assert t.v_ss(300.0) * 3.6 == pytest.approx(315.0)          # flat out on a 300 m arc


# ------------------------------------------------------------------------- hygiene majors


def test_split_nulls_is_idempotent_and_follows_a_W_change():
    """Hygiene M6: after V2 moves a class to W = 800, every corner cell points to a null
    with its own W; re-running changes nothing and makes no duplicate nulls."""
    cells = [Cell(0, 0, 0, ("A",), "A2", 50.0, 200.0, W=400, null_cell=3),
             Cell(1, 1, 0, ("A",), "A2", 50.0, 200.0, W=400, null_cell=3),
             Cell(2, 2, 1, ("A",), "A1", 50.0, 600.0, W=800, null_cell=3),
             Cell(3, 9, 2, ("G",), "G", 50.0, 600.0, W=400)]
    once = stimuli.split_nulls(cells)
    assert len(once) == 5 and once[4].W == 800 and "G" in once[4].tiers
    assert [c.null_cell for c in once[:3]] == [3, 3, 4]
    assert stimuli.split_nulls(once) == once                     # idempotent
    v2 = [dataclasses.replace(c, W=800) if c.approach == "A2" else c for c in once]
    after = stimuli.split_nulls(v2)
    assert len(after) == 5                                      # the W = 800 null is reused
    assert all(after[c.null_cell].W == c.W for c in after if "G" not in c.tiers)


def test_manifest_updates_merge_instead_of_overwriting(tmp_path):
    """Hygiene M2 / M3: updates re-read the manifest under a lock, so one stage's stale copy
    never drops another's fields, and a snapshot is marked before it is rolled out."""
    path = tmp_path / "manifest.json"
    e = agents.SnapshotEntry("G9", "runs/g9/1.pkl", 1, True, "h", 1, (0.0, 0.0), (1.0, 1.0),
                             True, False)
    agents.save_manifest(agents.Manifest(snapshots=[e]), path)
    # Another stage records step 2 and a rollout job marks the snapshot while a long stage
    # holds an early copy; that stage then records its own field through update_manifest.
    agents.update_manifest(lambda m: m.validation.__setitem__("step2", {"passed": True}), path)
    agents.mark_evaluated_locked("runs/g9/1.pkl", "rollouts-det", path)
    agents.update_manifest(lambda m: m.budget.__setitem__("x", 1), path)
    back = agents.load_manifest(path)
    assert back.validation["step2"]["passed"] and back.budget["x"] == 1
    assert back.entry("runs/g9/1.pkl").evaluated_on_stimuli
    assert (tmp_path / "manifest.json.lock").exists()


def test_agent_beating_the_key_is_detected_per_cell(tmp_path, monkeypatch):
    """Fidelity M2 / numerics M1: the anchored-keys stage compares every final's
    deterministic u = 0 return with J* and queues a 6.3 re-solve where one exceeds it."""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        ps = importlib.import_module("psychophysics")
    finally:
        sys.path.pop(0)
    bank, (corner, null) = _mini_bank()
    cache = ak.KeyCache(tmp_path / "keys")
    det = _det_group([(corner, _drive(700.0, 60.0, 870.0)), (null, _drive(400.0, 60.0, None))])
    det["reward"] = np.full((2, 200), 0.3, np.float32)
    traj = measures.trajectory_from_rollout(det, 0, "det")
    j_agent = measures.discounted_return(traj, 200)
    cache.put(_key_record(bank, corner, "nominal", _drive(700.0, 60.0, 880.0), J=j_agent - 1.0))
    cache.put(_key_record(bank, null, "null", _drive(400.0, 60.0, None), J=j_agent + 1.0))
    f = tmp_path / "G9.npz"
    rollout.save_rollouts(f, {"det": det}, dict(agent="G9", timesteps=1, log_std=[0.0, 0.0]))
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    jobs, summary = ps._beaten_jobs(bank, [corner, null], {"G9": SimpleNamespace(timesteps=1)},
                                    cache)
    assert summary["cells_checked"] == 2 and summary["beaten"] == {0: ["G9"]}
    assert len(jobs) == 1 and jobs[0]["cell"].cell_id == 0 and jobs[0]["kind"] == "nominal"
    assert jobs[0]["runs"]["G9"] == (str(f), 0)
