"""Ruling R19 (protocol 12), script side: (a) the two snapshots before each generalist final
('last3', G1-G3 and R18's replicates) also run tier G, the null straights, at 4
deterministic phases and with no stochastic runs, so protocol 8's gate can judge them: the
rollout plan, the rollout job (including a last-3 det group made before the ruling, which
is run again), the agent-rollout projection and the confirmatory stage's input check. The
R19 review's regression tests follow: the resume rule's scope (every kind), and R19a's two
side effects on analyses that read the planned runs (the history snapshots' own null runs,
so their d_resp and Δd_resp are defined for P2; P6 judging the last-3 checkpoints). The
analysis side (the gate judged at a history snapshot, a stable claim reported stable, a P2
claim recurring, and R19c's decision classes in AD21) is in test_interp_analysis.py.
Nothing is solved or rolled out: the policy and the rollout engine are doubles, runs and
keys are scripted trajectories, banks are hand-built and files go to tmp dirs only."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.interp import agents, io, registry, rollout, stimuli, support
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.constants import REPLICATION
from apexzero.interp.stimuli import Cell, CornerSpec, Geometry, Stimulus, StimulusBank
from apexzero.ocp.window import WindowSpec

REPO = Path(__file__).resolve().parents[1]
B = 0.075


@pytest.fixture(scope="module")
def ps():
    path = str(REPO / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


def _design(rows):
    """rows: (tiers, approach, n_phases, W). A hand-built bank and cells, one cell per pair
    (unmirrored), every agent a runner."""
    cells = [Cell(i, i, 10 + i, tiers, ap, 86.0, 600.0, W=W, P=4.0, n_phases=n_ph)
             for i, (tiers, ap, n_ph, W) in enumerate(rows)]
    return SimpleNamespace(stimuli=[None] * len(rows)), cells


A1, P7, TB, G400, G800 = 0, 1, 2, 3, 4
ROWS = [(("A",), "A1", 4, 400), (("A",), "A1", 16, 400), (("B",), "B", 4, 400),
        (("G",), "G", 4, 400), (("G",), "G", 4, 800)]


def test_R19a_last3_rows_run_tier_g_at_four_phases(ps):
    """Protocol 4's last-3 rows: tiers A and B as before, plus tier G at the 4-phase grid;
    the stochastic runs stay on tier A, so tier G has none. The other rows are unchanged."""
    assert REPLICATION["last3"]["det_tiers"] == ("A", "B", "G")
    assert REPLICATION["last3"]["det_phases"] == 4 and REPLICATION["last3"]["native"] == {"A": 8}
    assert "matched" not in REPLICATION["last3"]
    assert REPLICATION["other_generalist"]["det_tiers"] == ("A", "G")
    assert REPLICATION["other_specialist"]["det_tiers"] == ("D", "G")
    assert REPLICATION["final"]["det_phases"] is None
    _, cells = _design(ROWS)
    assert ps._det_phases(cells[G400], "last3") == [0, 1, 2, 3]
    assert ps._det_phases(cells[P7], "last3") == [0, 4, 8, 12]
    assert ps._det_phases(cells[TB], "last3") == [0, 1, 2, 3]
    assert ps._det_phases(cells[TB], "other_generalist") == []
    assert ps._rollout_runs(cells[G400], "last3", "det") == 4
    assert ps._rollout_runs(cells[G400], "last3", "nat") == 0
    assert ps._rollout_runs(cells[G400], "last3", "mat") == 0
    assert ps._rollout_runs(cells[A1], "last3", "nat") == 8
    assert [ps._r19_cell(c) for c in cells] == [False, False, False, True, True]


def _job_stubs(ps, monkeypatch, tmp_path, bank, cells):
    """``_rollout_job`` with the policy, the engine and the manifest stubbed: the engine
    returns the planned runs' index arrays and records each call; files go to tmp_path."""
    calls: list[tuple[str, list]] = []

    def engine(pol, params, rms, bank_, cells_, runs, rec):
        calls.append((rec, list(runs)))
        return dict(cell=np.array([r.cell_id for r in runs], np.int32),
                    phase=np.array([r.phase for r in runs], np.int16),
                    sample=np.array([r.sample for r in runs], np.int32),
                    steer_sign=np.array([r.steer_sign for r in runs], np.int8))

    marked: list[str] = []
    monkeypatch.setattr(ps, "_R", dict(bank=bank, cells=cells, design_hash="h"))
    monkeypatch.setattr(agents, "load_agent", lambda path, agent: SimpleNamespace(
        timesteps=7, cfg=None, actor_params=None, obs_rms=None, log_std=np.zeros(2)))
    monkeypatch.setattr(agents, "mark_evaluated_locked", lambda path, stage: marked.append(stage))
    monkeypatch.setattr(io, "rollout_path", lambda a, t: tmp_path / a / f"{int(t):012d}.npz")
    monkeypatch.setattr(rollout, "agent_policy", lambda cfg: None)
    monkeypatch.setattr(rollout, "run_rollouts", engine)
    return calls, marked


def test_R19a_rollout_job_runs_tier_g_and_reruns_a_pre_r19_det_group(ps, monkeypatch, tmp_path):
    """A last-3 job plans tier G at phases 0-3 beside tiers A and B (P7 cells at 0, 4, 8,
    12), and its native-noise runs on tier A only. A group on disk that holds every planned
    run is kept; a det group without the tier G runs (as made before R19a) is run again,
    while the complete nat group is kept. ``_tier_g_missing`` counts what a det group lacks."""
    bank, cells = _design(ROWS)
    calls, marked = _job_stubs(ps, monkeypatch, tmp_path, bank, cells)
    job = dict(label="G2s1 x last3", agent="G2s1", path="runs/x/ckpt/7.pkl", kind="last3",
               modes=("det", "nat", "mat"), rule3=False, force=False, design_hash="h")
    target = io.rollout_path("G2s1", 7)
    assert ps._tier_g_missing(target, cells, "G2s1") == 8          # no file: all missing
    out = ps._rollout_job(job)
    assert "error" not in out and "rerun" not in out
    assert [rec for rec, _ in calls] == ["full", "stoch"]           # no matched noise
    det = {(r.cell_id, r.phase) for r in calls[0][1]}
    assert det == ({(c, k) for c in (A1, TB, G400, G800) for k in range(4)}
                   | {(P7, k) for k in (0, 4, 8, 12)})
    nat = calls[1][1]
    assert {r.cell_id for r in nat} == {A1, P7} and len(nat) == 16
    assert out["groups"] == ["det:20", "nat:16"] and marked == ["rollouts-det"]
    assert ps._tier_g_missing(target, cells, "G2s1") == 0
    # rerun: both groups complete on disk, nothing is run
    calls.clear()
    out = ps._rollout_job(job)
    assert calls == [] and out["groups"] == [] and "rerun" not in out
    # a det group made before R19a (tiers A and B only) is incomplete: det runs again
    groups, _ = rollout.load_rollouts(target)
    keep = ~np.isin(groups["det"]["cell"], (G400, G800))
    old = {k: v[keep] for k, v in groups["det"].items()}
    target.unlink()
    rollout.save_rollouts(target, {"det": old, "nat": groups["nat"]},
                          dict(agent="G2s1", timesteps=7, log_std=[0.0, 0.0]))
    assert ps._tier_g_missing(target, cells, "G2s1") == 8
    out = ps._rollout_job(job)
    assert out["rerun"] == ["det"] and [rec for rec, _ in calls] == ["full"]
    assert {(r.cell_id, r.phase) for r in calls[0][1]} == det
    assert ps._tier_g_missing(target, cells, "G2s1") == 0
    g2, _ = rollout.load_rollouts(target)
    assert len(g2["nat"]["cell"]) == 16                              # nat kept as it was
    # --force runs a complete group again, without counting it as incomplete
    calls.clear()
    out = ps._rollout_job(dict(job, force=True, modes=("det",)))
    assert [rec for rec, _ in calls] == ["full"] and "rerun" not in out


def _snaps(ps):
    def snaps(spec):
        if spec.final_file == "agent.pkl":
            return [ps.ROOT / spec.run / "agent.pkl"]
        return [ps.ROOT / spec.run / "ckpt" / f"{i:012d}.pkl" for i in range(1, 17)]
    return snaps


def test_R19a_rollout_projection_counts_tier_g_at_the_history_snapshots(ps, monkeypatch):
    """Protocol 4's projection on the design: each generalist's two last-3 snapshots add 4
    deterministic phases on every tier G cell (each run W steps), counted per agent and in
    ``r19``; specialists and the other checkpoints (which ran tier G already) add none."""
    bank, cells = _design(ROWS)
    monkeypatch.setattr(agents, "list_snapshots", _snaps(ps))
    ro = ps._rollout_projection(cells, dict(full=2e4, stoch=1e4))
    by = ro["by_agent"]
    # final: every phase of every cell (4 + 16 + 4 + 4 + 4); last 3: 4 phases on A, B, G
    assert by["G1s1"]["snapshots"] == {"final": 1, "last3": 2}
    assert by["G1s1"]["runs"]["det"] == 32 + 2 * 5 * 4
    assert by["G1s1"]["r19"] == dict(runs=2 * 2 * 4, steps=2 * 4 * (400 + 800))
    # nat: the final's 32 per tier A cell and 16 elsewhere; the last 3's 8 per tier A cell
    assert by["G1s1"]["runs"]["nat"] == (2 * 32 + 3 * 16) + 2 * 2 * 8
    assert by["G1"]["snapshots"] == {"final": 1, "last3": 2, "other_generalist": 13}
    assert by["G1"]["r19"] == by["G1s1"]["r19"]
    assert all(by[a]["r19"] == dict(runs=0, steps=0) for a in registry.specialists())
    r19 = ro["r19"]
    assert r19["cells"] == 2 and r19["rule"].startswith("R19a")
    assert r19["runs"] == 9 * 16 and r19["steps"] == 9 * 9600
    assert r19["steps_without"] == ro["steps_total"] - r19["steps"]
    assert r19["process_h"] == pytest.approx(9 * 9600 / 2e4 / 3600)
    # without R19a's tier G the last-3 det runs would be 2 x 3 x 4 per generalist
    monkeypatch.setitem(REPLICATION, "last3", dict(REPLICATION["last3"], det_tiers=("A", "B")))
    before = ps._rollout_projection(cells, dict(full=2e4, stoch=1e4))
    assert before["by_agent"]["G1s1"]["runs"]["det"] == 32 + 2 * 3 * 4
    assert before["steps_total"] == r19["steps_without"] and before["r19"]["runs"] == 0


def test_R19a_confirmatory_stage_needs_the_history_tier_g_runs(ps, monkeypatch, tmp_path):
    """The analysis stage's one-shot gate reads each history snapshot's det group from its
    rollout file and refuses while one lacks its tier G runs (a det group made before
    R19a), naming it; history snapshots that hold them raise nothing about tier G."""
    bank, cells = _design([(("A",), "A1", 4, 400), (("G",), "G", 4, 400)])
    ents = []
    for a in agents.AGENTS:
        tss = (80, 90, 100) if registry.is_generalist(a) else (100,)
        ents += [agents.SnapshotEntry(a, f"runs/{a}/{t}.pkl", t, t == 100, "h", 1, (0.0, 0.0),
                                      (1.0, 1.0), True, False) for t in tss]
    m = agents.Manifest(band=0.075, snapshots=ents)

    def path(a, t):
        return tmp_path / a / f"{int(t):012d}.npz"

    def write(p, groups, with_g=True):
        p.parent.mkdir(parents=True, exist_ok=True)
        ids = [0, 1] if with_g else [0]
        arrs = {}
        for g in groups:
            arrs[f"{g}_cell"] = np.repeat(ids, 4)
            arrs[f"{g}_phase"] = np.tile(np.arange(4), len(ids))
            arrs[f"{g}_sample"] = np.full(4 * len(ids), -1)
        np.savez(p, **arrs)

    for e in ents:
        write(path(e.agent, e.timesteps), ("det", "nat", "mat") if e.is_final else ("det", "nat"),
              with_g=not (e.agent == "G2s1" and e.timesteps == 80))
    regs = SimpleNamespace(regimes={a: SimpleNamespace(r_clip_max=8.9, r_clip=(8.9,))
                                    for a in agents.AGENTS})
    monkeypatch.setattr(ps, "_require_tag", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_manifest", lambda: m)
    monkeypatch.setattr(ps, "_design", lambda laid_out=False: (bank, cells))
    monkeypatch.setattr(support, "load_support", lambda d: regs)
    monkeypatch.setattr(io, "rollout_path", path)
    with pytest.raises(SystemExit) as ex:
        ps.stage_analyse(SimpleNamespace(confirmatory=True, agent=None, force=False))
    msg = str(ex.value)
    assert ("G2s1: snapshot -2 before the final lacks 4 of its tier G deterministic runs "
            "(ruling R19a") in msg
    assert msg.count("R19a") == 1                     # every other history holds them
    assert ps._tier_g_missing(path("G2s1", 90), cells, "G2s1") == 0


# ------------------------------------------------------------------------- R19 review


def test_R19_review_resume_rule_covers_every_kind(ps, monkeypatch, tmp_path):
    """R19 review finding 2: the resume rule is not limited to the last-3 rows. For every
    snapshot kind and mode, a group on disk that holds every planned run (or more) is kept,
    and one that lacks some is run again whole (all its planned runs, not only the missing
    ones) and replaces the one on disk. ``_rollout_job`` writes whole groups, so on a design
    that has not changed only a group made under an older plan (a last-3 det group from
    before R19a) is ever run again."""
    bank, cells = _design(ROWS)
    calls, _ = _job_stubs(ps, monkeypatch, tmp_path, bank, cells)
    job = dict(label="G1 final", agent="G1", path="runs/x/agent.pkl", kind="final",
               modes=("det", "nat", "mat"), rule3=False, force=False, design_hash="h")
    target = io.rollout_path("G1", 7)
    out = ps._rollout_job(job)
    # final: every phase (4 + 16 + 4 + 4 + 4); native 32 per tier A cell and 16 elsewhere;
    # matched 16 per tier A cell (no tier E here)
    assert out["groups"] == ["det:32", "nat:112", "mat:32"] and "rerun" not in out
    calls.clear()
    out = ps._rollout_job(job)
    assert calls == [] and out["groups"] == [] and "rerun" not in out      # complete: kept
    # a final nat group one run short is incomplete: run again whole, det and mat kept
    groups, _ = rollout.load_rollouts(target)
    target.unlink()
    rollout.save_rollouts(target, dict(groups, nat={k: v[1:] for k, v in groups["nat"].items()}),
                          dict(agent="G1", timesteps=7, log_std=[0.0, 0.0]))
    out = ps._rollout_job(job)
    assert out["rerun"] == ["nat"] and [(rec, len(r)) for rec, r in calls] == [("stoch", 112)]
    g2, _ = rollout.load_rollouts(target)
    assert {k: len(v["cell"]) for k, v in g2.items()} == dict(det=32, nat=112, mat=32)
    # groups that hold more than the plan are kept: the final's cover what an other-checkpoint
    # row plans (tiers A and G at the 4-phase grid, 8 native samples per tier A cell)
    calls.clear()
    out = ps._rollout_job(dict(job, kind="other_generalist"))
    assert calls == [] and out["groups"] == [] and "rerun" not in out


def _strip_design():
    """One R40/90 left corner cell (tier A, A1, 4 phases, D 300 m, W 200) and its width-12
    null straight (tier G, D 600 m, W 200) as real StimulusBank rows (test_interp_review's
    mini design)."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, pst = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A",), 1, -1, pc.landmarks,
                      geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, pst.landmarks, geo,
                      None)]
    bank = StimulusBank(np.stack([pc.kappa, pst.kappa]), np.stack([pc.w_left, pst.w_left]),
                        np.stack([pc.w_right, pst.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), "A1", 60.0, 300.0, W=200, P=3.0, null_cell=1),
             Cell(1, 1, 1, ("G",), "G", 60.0, 600.0, W=200, P=3.0)]
    return bank, cells


def _drive(s0: float, v0: float, s_brake: float | None, n: int = 201) -> np.ndarray:
    """States of a scripted drive at 20 Hz: pedal 1 until s_brake (None: never brakes), then
    -0.6 while braking at 15 m/s^2 down to 40 m/s, then 0.2."""
    x = np.zeros((n, 8))
    s, v, braked = s0, v0, False
    for k in range(n):
        braking = s_brake is not None and s >= s_brake and v > 40.0
        braked |= braking
        x[k] = [s, 0.0, 0.0, v, 0.0, 0.0, 0.0, -0.6 if braking else (0.2 if braked else 1.0)]
        if braking:
            v = max(v - 15.0 * 0.05, 40.0)
        s += v * 0.05
    return x


def _key(bank, cell, kind: str, x: np.ndarray) -> ak.KeyRecord:
    W = int(cell.W)
    a = np.stack([np.zeros(W), x[1:, 7]], axis=1)
    return ak.KeyRecord(key_id=ak.expected_key_id(bank, cell, kind), kind=kind, status="verified",
                        stim_id=cell.stim_id, stim_name=bank.stimuli[cell.stim_id].name,
                        cell_id=cell.cell_id, phase=0,
                        direction=bank.stimuli[cell.stim_id].direction, x0=x[0].copy(), W=W,
                        spec=dataclasses.asdict(WindowSpec(steps=W)), alpha_max=0.17,
                        shrink=0.0, J=50.0, x=x, a=a,
                        replay={"verified": True, "survives": True, "rel": 0.0},
                        labels={"approach": cell.approach}, measures={"d_resp": np.nan})


def _history_det(ps, bank, cells, tiers) -> dict:
    """A history snapshot's det group as ``_rollout_job`` plans it for protocol 4's last-3 row
    with det tiers ``tiers`` (``rollout_cells``, then ``plan_runs`` at the row's 4-phase
    grid): each run a scripted drive from its phase's start, braking at 870 m on the corner
    and never on the null straight."""
    by = {c.cell_id: c for c in cells}
    runs = rollout.plan_runs(ps.rollout_cells(cells, tiers), bank, "det", 0,
                             phases_det=REPLICATION["last3"]["det_phases"], agent="G1")
    x = np.stack([_drive(float(by[r.cell_id].initial_state(r.phase)[0]), by[r.cell_id].v0,
                         None if "G" in by[r.cell_id].tiers else 870.0)
                  for r in runs]).astype(np.float32)
    n, K = x.shape[:2]
    a = np.zeros((n, K - 1, 2), np.float32)
    a[:, :, 1] = x[:, 1:, 7]

    def col(f):
        return np.array([getattr(r, f) for r in runs])

    return dict(cell=col("cell_id"), stim=col("stim_id"), pair=col("pair_id"),
                phase=col("phase"), sample=col("sample"), steer_sign=col("steer_sign"),
                W=col("W"), s0=x[:, 0, 0].astype(np.float64), valid=np.ones((n, K), bool), x=x,
                term_step=np.full(n, -1), trunc_step=np.full(n, -1), a=a, a_mean=a.copy(),
                reward=np.zeros((n, K - 1), np.float32))


def test_R19_review_history_snapshots_get_their_null_runs_so_d_resp_is_defined(ps, tmp_path):
    """R19 review finding 1 (P2 side). The catch-trial onset d_resp needs the driver's own
    null run, which ``run_table`` takes from the snapshot's det group (``_null_runs``: the
    phase-0, unmirrored tier G run). With R19a a history snapshot's det group holds the tier
    G straights, so its corner runs' d_resp and Δd_resp, the pair Δd_resp that P2(a) reads,
    and the corner regret (which also needs the null run) are defined as at the finals.
    Planned with the last-3 row before R19a (tiers A and B), the same snapshot has no null
    run and they are NaN: P2(a) could not hold there."""
    bank, cells = _strip_design()
    corner, null = cells
    cache = ak.KeyCache(tmp_path / "keys")
    cache.put(_key(bank, corner, "nominal", _drive(700.0, 60.0, 880.0)))
    cache.put(_key(bank, null, "null", _drive(400.0, 60.0, None)))
    keys = A.CachedKeys(bank, cells, cache)
    kt = A.key_table(cells, bank, keys)
    by = {c.cell_id: c for c in cells}

    def snapshot(tiers):
        det = _history_det(ps, bank, cells, tiers)
        rt = A.run_table("G1", 90, {"det": det}, cells, bank, keys, None)
        return det, rt, A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)}, B)

    det, rt, pv = snapshot(REPLICATION["last3"]["det_tiers"])
    assert sorted(zip(det["cell"].tolist(), det["phase"].tolist(), strict=True)) == \
        [(c, k) for c in (0, 1) for k in range(4)]
    assert set(A._null_runs(det, by)) == {null.cell_id}
    mine = A._it(rt, "cell") == corner.cell_id
    assert np.isfinite(A._fl(rt, "d_resp")[mine]).all()
    i0 = mine & (A._it(rt, "phase") == 0)
    assert np.isfinite(A._fl(rt, "dd_resp")[i0]).all()
    assert np.isfinite(A._fl(rt, "corner_regret")[i0]).all()
    assert np.isfinite(A._fl(pv, "dd_resp")[A._it(pv, "pair") == corner.pair_id]).all()
    # the last-3 row before R19a: no tier G run, no null run, no catch-trial onset
    det0, rt0, pv0 = snapshot(("A", "B"))
    assert set(det0["cell"].tolist()) == {corner.cell_id} and A._null_runs(det0, by) == {}
    assert np.isnan(A._fl(rt0, "d_resp")).all() and np.isnan(A._fl(rt0, "dd_resp")).all()
    assert np.isnan(A._fl(rt0, "corner_regret")).all()
    assert np.isnan(A._fl(pv0, "dd_resp")[A._it(pv0, "pair") == corner.pair_id]).all()


def _p6_pairs(kind: str, ckpt: int, order: dict[float, int]) -> dict:
    """One seed-0 checkpoint's pair rows as protocol 4's row ``kind`` plans its runs: each
    tier A R level at A2 (all 4 runs complete from checkpoint ``order[R]`` on, else 3 of 4;
    in band), the gentle controls at A1 (complete), and the tier G width-12 null straights
    (complete) when the row's det tiers include G (a final runs every tier)."""
    from apexzero.interp.constants import R_CONTROLS

    tiers = REPLICATION[kind].get("det_tiers")
    base = dict(agent="G1", no_cap=False, excluded="", width=12.0, n_runs=4, osc_index=0.01,
                e_ve=-0.01, e_ve_bo=0.01, uses_ra=False, regime="bulk")
    rows = [dict(base, tiers="A", approach="A2", braking=True, R_nom=R,
                 det_complete=4 if ckpt >= order[R] else 3) for R in order]
    rows += [dict(base, tiers="A", approach="A1", braking=False, R_nom=R, det_complete=4)
             for R in R_CONTROLS]
    if tiers is None or "G" in tiers:
        rows += [dict(base, tiers="G", approach="G", braking=False, R_nom=np.nan,
                      det_complete=4) for _ in range(3)]
    return A._to_table(rows, sorted(rows[0]))


def test_R19_review_p6_judges_the_last3_checkpoints(ps, monkeypatch):
    """R19 review finding 1 (P6 side). P6 counts a checkpoint that fails protocol 8's gate as
    not competent (AD18), and the analysis stage gates every seed-0 snapshot from its own
    runs. With R19a the two checkpoints before the final run tier G, so the gate judges them
    as it judges the other checkpoints (tiers A and G) and t_R is each level's first
    checkpoint of lasting competence. With the last-3 row before R19a (tiers A and B) they
    fail the gate as not evaluable, every t_R moves to the final and P6a is not testable."""
    monkeypatch.setattr(agents, "list_snapshots", lambda spec: [
        ps.ROOT / "runs" / "G1" / "ckpt" / f"{i:012d}.pkl" for i in range(1, 8)])
    kinds = {int(Path(p).stem): k for p, k in ps._snapshot_kinds("G1", "all")}
    assert kinds == {1: "other_generalist", 2: "other_generalist", 3: "other_generalist",
                     4: "other_generalist", 5: "last3", 6: "last3", 7: "final"}
    order = {69.8: 1, 44.7: 2, 35.8: 3, 22.9: 4, 14.6: 5, 11.7: 6}

    def p6a():
        by = {c: _p6_pairs(k, c, order) for c, k in kinds.items()}
        gate = {c: A.checkpoint_gate(p) for c, p in by.items()}      # as stage_analyse does
        return gate, A.p6(by, B, {c: g["passed"] for c, g in gate.items()})[0]

    gate, d = p6a()
    assert all(g["evaluable"] and g["passed"] for g in gate.values())
    assert d.verdict == "confirm" and d.details["t_R"] == {R: float(c) for R, c in order.items()}
    monkeypatch.setitem(REPLICATION, "last3", dict(REPLICATION["last3"], det_tiers=("A", "B")))
    gate, d = p6a()
    assert [c for c, g in gate.items() if not g["evaluable"]] == [5, 6]
    assert set(d.details["t_R"].values()) == {7.0} and d.verdict == "not testable"
