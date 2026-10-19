"""Ruling R17 (protocol 12): (a) re-anchored keys per deterministic phase, each anchored on
that phase's own run, and every run judged against the key of its phase; (b) phase-matched
keys at all 16 phases of P7's cells, with P7's key null kept on phases 0, 4, 8 and 12.
Nothing is solved or rolled out: keys and runs are doubles or hand-built records, banks are
hand-built, and files go to tmp dirs only."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.interp import agents, io, measures, rollout, stimuli, support
from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp.measures import Trajectory
from apexzero.interp.stimuli import (
    Cell,
    CornerSpec,
    Geometry,
    Landmarks,
    Stimulus,
    StimulusBank,
)

V7_FAILED = {"V5": {"passed": False}, "V7": {"passed": False, "name": "V7",
                                              "details": {"cells": {}, "failed": [3]}}}
V7_PASSED = {"V5": {"passed": False}, "V7": {"passed": True, "name": "V7"}}


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Bank arrays are float32 as in the rollouts; other modules switch x64 on at import."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def ps():
    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


def _ids(cs) -> list[int]:
    return [c.cell_id for c in cs]


# ------------------------------------------------------------------------- fixtures


def _bank(n_phases: int = 4, approach: str = "A1"):
    """One R40/90 left corner cell (A1 by default, so the band reads e_ve^RA) and its null
    straight, on a real StimulusBank (row hashes, landmarks)."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, pst = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A",), 1, -1, pc.landmarks,
                      geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, pst.landmarks, geo, None)]
    bank = StimulusBank(np.stack([pc.kappa, pst.kappa]), np.stack([pc.w_left, pst.w_left]),
                        np.stack([pc.w_right, pst.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), approach, 86.0, 600.0, W=400, P=4.0, null_cell=1,
                  n_phases=n_phases, flags=("p7",) if n_phases == 16 else ()),
             Cell(1, 1, 1, ("G",), "G", 86.0, 600.0, W=400, P=4.0)]
    return bank, cells


def _record(kind: str, phase: int, *, key_id: str = "", agent: str = "", snapshot: str = "",
            measures_: dict | None = None, status: str = "verified", x0=None) -> ak.KeyRecord:
    return ak.KeyRecord(
        key_id=key_id or f"{kind}-{agent}-{snapshot}-ph{phase}", kind=kind, status=status,
        stim_id=0, stim_name="syn/R40/T90/W12/typical/L", cell_id=0, phase=phase, direction=1,
        x0=np.zeros(8) if x0 is None else np.asarray(x0, float), W=400, spec={},
        alpha_max=0.17, shrink=0.0, agent=agent,
        anchor={"snapshot": snapshot} if snapshot else {},
        measures=measures_ if measures_ is not None else {"d_b": 100.0, "v_b": 80.0,
                                                          "has_brake": True, "v_e": 30.0})


def _run(s0: float = 600.0, s1: float = 1400.0, n: int = 401, term: int = -1) -> Trajectory:
    x = np.zeros((n, 8))
    x[:, 0] = np.linspace(s0, s1, n)
    x[:, 3] = 80.0
    a = np.tile([0.0, 1.0], (n - 1, 1))
    return Trajectory(source="agent", t=np.arange(n) * 0.05, x=x, valid=np.ones(n, bool), a=a,
                      term_step=term)


# ------------------------------------------------------------------------- (a) answer_key


def test_R17a_anchored_key_identity_carries_the_phase():
    """The phase is part of a re-anchored key's identity: the same state anchored at two
    phases gives two keys; the same phase reproduces the id."""
    bank, cells = _bank()
    base = _record("nominal", 0)
    run = _run()
    a0 = ak.anchor_for(bank, cells[0], base, run, agent="G1", snapshot="7", phase=0)
    a2 = ak.anchor_for(bank, cells[0], base, run, agent="G1", snapshot="7", phase=2)
    assert a0 is not None and a2 is not None and (a0.phase, a2.phase) == (0, 2)
    assert a0.key_id != a2.key_id
    assert ak.anchor_for(bank, cells[0], base, run, agent="G1", snapshot="7",
                         phase=2).key_id == a2.key_id
    np.testing.assert_array_equal(a0.x_prev, a2.x_prev)          # identical states
    # s_L = s_c0 - (d_b* + v_b* 1 s) of the base key; x_L the first state at or after it
    s_L = bank.stimuli[0].landmarks.s_c0 - (100.0 + 80.0)
    assert a2.s_L == pytest.approx(s_L) and run.x[a2.L - 1, 0] < s_L <= run.x[a2.L, 0]
    np.testing.assert_array_equal(a2.a_prev, run.a[a2.L - 1])
    assert a2.key_id == ak.anchored_key_id(bank, run.x[a2.L - 1], run.a[a2.L - 1], 0, "G1",
                                           "7", 2)


def test_R17a_anchor_base_is_the_key_at_the_runs_phase(monkeypatch):
    """The key a phase-k re-anchored key starts from, and falls back to: the phase-matched
    key at u_k when one is cached and verified, else the u = 0 nominal key."""
    nom = _record("nominal", 0)
    ph2 = _record("phase", 2, measures_={"d_b": 98.0, "v_b": 81.0, "has_brake": True})
    ph3 = _record("phase", 3, status="unsolved")
    seen = []

    def fake_key_for(cell, kind, phase=0, agent="", cache=None, bank=None, **kw):
        seen.append((kind, phase))
        return {2: ph2, 3: ph3}.get(phase) if kind == "phase" else None

    monkeypatch.setattr(ak, "key_for", fake_key_for)
    cell = SimpleNamespace(cell_id=0)
    assert ak.anchor_base(cell, 2, nom) is ph2
    assert ak.anchor_base(cell, 1, nom) is nom            # no phase key at u_1
    assert ak.anchor_base(cell, 3, nom) is nom            # unverified phase key
    assert ak.anchor_base(cell, 0, nom) is nom and (("phase", 0) not in seen)


def test_R17a_anchored_key_falls_back_to_the_key_at_that_phase():
    """6.1's fallback at phase k: no sustained brake in the key at that phase, or a run that
    does not reach s_L alive, returns that key itself (nothing is solved)."""
    bank, cells = _bank()
    ph1_flat = _record("phase", 1, measures_={"d_b": np.nan, "v_b": np.nan, "has_brake": False})
    assert ak.anchored_key(bank, cells[0], ph1_flat, "G1", _run(), phase=1) is ph1_flat
    ph1 = _record("phase", 1)
    crashed = _run(term=10)                               # dies before s_L (820 m)
    assert ak.anchored_key(bank, cells[0], ph1, "G1", crashed, phase=1) is ph1
    assert ak.anchor_for(bank, cells[0], ph1, crashed, phase=1) is None


def test_R17a_anchored_key_stores_one_record_per_phase(monkeypatch, tmp_path):
    """``anchored_key`` end to end with the solver stubbed: the same run anchored at two
    phases stores two records, each with its phase, its id and its alias; the anchor records
    the phase and the base key; a repeat at a phase is a cache hit (nothing solved)."""
    bank, cells = _bank()
    cache = ak.KeyCache(tmp_path / "keys")
    run = _run()
    solved = []

    def fake_start(prob, name, guess, extend, stages, max_iter):
        solved.append((prob.phase, name))
        return ak.StartRecord(name, float("nan"), False, False, False, "stub", 0, 0.0), None

    monkeypatch.setattr(ak, "retime", lambda *a, **k: None)
    monkeypatch.setattr(ak, "_run_start", fake_start)
    base0 = _record("nominal", 0, key_id="nom0")
    base2 = _record("phase", 2, key_id="ph2", measures_={"d_b": 98.0, "v_b": 81.0,
                                                          "has_brake": True})
    r0 = ak.anchored_key(bank, cells[0], base0, "G1", run, cache, snapshot="7", phase=0)
    r2 = ak.anchored_key(bank, cells[0], base2, "G1", run, cache, snapshot="7", phase=2)
    assert (r0.kind, r0.phase, r2.kind, r2.phase) == ("anchored", 0, "anchored", 2)
    assert r0.key_id != r2.key_id
    assert r2.anchor["phase"] == 2 and r2.anchor["base_kind"] == "phase"
    assert r2.anchor["base_phase"] == 2 and r2.anchor["nominal"] == "ph2"
    # s_L from the phase key's own d_b* and v_b*
    assert r2.anchor["s_L"] == pytest.approx(bank.stimuli[0].landmarks.s_c0 - (98.0 + 81.0))
    an = ak.anchor_for(bank, cells[0], base2, run, agent="G1", snapshot="7", phase=2)
    assert r2.key_id == an.key_id
    np.testing.assert_array_equal(r2.x0, run.x[an.L - 1])
    for k, r in ((0, r0), (2, r2)):
        hit = ak.key_for(cells[0], "anchored", k, "G1@7", cache, bank)
        assert hit is not None and hit.key_id == r.key_id and hit.phase == k
    assert ak.key_for(cells[0], "anchored", 1, "G1@7", cache, bank) is None
    n = len(solved)
    assert ak.anchored_key(bank, cells[0], base2, "G1", run, cache, snapshot="7",
                           phase=2).key_id == r2.key_id and len(solved) == n


def test_R17a_anchored_key_cache_hit_requires_the_same_base(monkeypatch, tmp_path):
    """Review finding 4: the anchored id holds the problem, not the warm start. A key
    anchored from the u = 0 key (no phase key cached yet) is not reused once the phase key
    exists and lands on the same anchor step: it is re-solved from the phase key and the
    new record replaces the old under the same id and alias."""
    bank, cells = _bank()
    cache = ak.KeyCache(tmp_path / "keys")
    run = _run()
    solved = []

    def fake_start(prob, name, guess, extend, stages, max_iter):
        solved.append(name)
        return ak.StartRecord(name, float("nan"), False, False, False, "stub", 0, 0.0), None

    monkeypatch.setattr(ak, "retime", lambda *a, **k: None)
    monkeypatch.setattr(ak, "_run_start", fake_start)
    m = {"d_b": 98.0, "v_b": 81.0, "has_brake": True}
    u0 = _record("nominal", 0, key_id="nom0", measures_=m)
    ph2 = _record("phase", 2, key_id="ph2", measures_=m)        # same s_L, same step L
    r_old = ak.anchored_key(bank, cells[0], u0, "G1", run, cache, snapshot="7", phase=2)
    n = len(solved)
    assert r_old.anchor["nominal"] == "nom0"
    # same base: a cache hit
    assert ak.anchored_key(bank, cells[0], u0, "G1", run, cache, snapshot="7",
                           phase=2).key_id == r_old.key_id and len(solved) == n
    r_new = ak.anchored_key(bank, cells[0], ph2, "G1", run, cache, snapshot="7", phase=2)
    assert len(solved) > n and solved[n] == "nominal:ph2"
    assert r_new.key_id == r_old.key_id and r_new.anchor["nominal"] == "ph2"
    hit = ak.key_for(cells[0], "anchored", 2, "G1@7", cache, bank)
    assert hit.anchor["nominal"] == "ph2" and hit.anchor["base_kind"] == "phase"


def test_R17a_cache_and_analysis_find_the_key_of_each_phase(tmp_path):
    """A re-anchored record is aliased by its phase (``cell<id>_ph<k>_<agent>@<ts>``):
    ``key_for`` and ``CachedKeys.anchored`` return the key of the asked phase, never
    another phase's."""
    bank, cells = _bank()
    cache = ak.KeyCache(tmp_path / "keys")
    for k in (0, 2):
        cache.put(_record("anchored", k, key_id=f"ra{k}", agent="G1", snapshot="7",
                          measures_={"v_e": 31.0 + k, "has_brake": True}))
    for k in (0, 2):
        rec = ak.key_for(cells[0], "anchored", k, "G1@7", cache, bank)
        assert rec is not None and rec.key_id == f"ra{k}" and rec.phase == k
    assert ak.key_for(cells[0], "anchored", 1, "G1@7", cache, bank) is None
    keys = A.CachedKeys(bank, cells, cache, recompute=False)
    v2 = keys.anchored(cells[0], "G1", 7, 2)
    assert v2.key_id == "ra2" and v2.phase == 2 and v2.metrics.v_e == 33.0
    assert keys.anchored(cells[0], "G1", 7, 1) is None
    assert keys.anchored(cells[0], "G1", 7).key_id == "ra0"        # phase defaults to 0


# ------------------------------------------------------------------------- (a) the stage


def _det(rows: list[tuple[int, int]]) -> dict:
    return dict(cell=np.array([c for c, _ in rows]), phase=np.array([k for _, k in rows]))


def test_R17a_anchored_jobs_one_per_deterministic_phase(ps):
    """One key per (cell, deterministic phase), each anchored on that phase's row: every
    phase at the finals (16 on P7 cells), the 4-phase grid at the checkpoints. A cell's
    phases share one job (review finding 1)."""
    four = Cell(0, 0, 0, ("A",), "A1", 86.0, 600.0, W=400, P=4.0)
    p7 = Cell(1, 1, 1, ("A",), "A1", 86.0, 600.0, W=400, P=4.0, n_phases=16, flags=("p7",))
    # cell 2 has a run but is not selected: no job; cell 3 is selected but has no run
    none = Cell(3, 3, 3, ("A",), "A1", 86.0, 600.0, W=400, P=4.0)
    final = _det([(0, k) for k in range(4)] + [(1, k) for k in range(16)] + [(2, 0)])
    jobs = ps.anchored_jobs([four, p7, none], "f.npz", final, "G1", 7, "P0", force=True)
    got = [(j["cell"].cell_id, k, r) for j in jobs for k, r in j["phases"]]
    assert got == [(0, k, k) for k in range(4)] + [(1, k, 4 + k) for k in range(16)]
    assert all(j["what"] == "anchored" and j["force"] and j["file"] == "f.npz"
               and j["agent"] == "G1" and j["timesteps"] == 7 for j in jobs)
    assert jobs[1]["label"] == "RA G1@7 cell 1 x16 phases (P0)"
    ckpt = _det([(1, k) for k in (0, 4, 8, 12)])
    assert [j["phases"] for j in ps.anchored_jobs([p7], "c.npz", ckpt, "G1", 3, "P2")] \
        == [[(0, 0), (4, 1), (8, 2), (12, 3)]]
    # 6.3's u = 0 rows are unchanged
    assert ps._u0_rows(final, []) == {0: 0, 1: 4, 2: 20}


def test_R17a_anchored_job_solves_a_missing_nominal_key_once_per_cell(ps, monkeypatch):
    """Review finding 1: a cell without a cached nominal key (tier B D = 600) is solved
    once per (agent, cell) job, not once per phase by concurrent workers; a phase that fails
    leaves the others' keys and reports itself."""
    bank, cells = _bank()
    cold = _record("nominal", 0)
    solves, anchored = [], []

    def fake_key_for(cell, kind, phase=0, agent="", cache=None, bank=None, **kw):
        return None

    def fake_full(bank_, cell, cache=None, tables=None, **kw):
        solves.append(cell.cell_id)
        return cold

    def fake_anchored(bank_, cell, base, agent, traj, cache, *, snapshot="", phase=0, **kw):
        if phase == 2:
            raise RuntimeError("solver blew up")
        anchored.append((base.key_id, traj, phase))
        return _record("anchored", phase, agent=agent, snapshot=snapshot)

    monkeypatch.setattr(ps, "_W", dict(bank=bank, tables=None, cache=object()))
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": {}}, {}))
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: f"run{i}")
    monkeypatch.setattr(ak, "key_for", fake_key_for)
    monkeypatch.setattr(ak, "full_recipe", fake_full)
    monkeypatch.setattr(ak, "anchored_key", fake_anchored)
    det = _det([(cells[0].cell_id, k) for k in range(4)])
    (job,) = ps.anchored_jobs([cells[0]], "f.npz", det, "S1", 7, "P1")
    out = ps._anchored_job(job)
    assert solves == [cells[0].cell_id]
    assert anchored == [(cold.key_id, "run0", 0), (cold.key_id, "run1", 1),
                        (cold.key_id, "run3", 3)]
    assert out["phases"] == [0, 1, 3] and [k["phase"] for k in out["keys"]] == [0, 1, 3]
    assert "phase 2:" in out["error"] and "solver blew up" in out["error"]
    assert ps._errors([out])[0].endswith("RuntimeError: solver blew up")


def test_R17a_anchored_job_anchors_on_the_phase_run_from_the_phase_key(ps, monkeypatch):
    """The worker anchors the phase-k job on the run of its row, from the key at phase k
    (the phase-matched key; the u = 0 key where none is cached), and records the phase."""
    bank, cells = _bank()
    nom, ph3 = _record("nominal", 0), _record("phase", 3)
    calls = []

    def fake_key_for(cell, kind, phase=0, agent="", cache=None, bank=None, **kw):
        return {("nominal", 0): nom, ("phase", 3): ph3}.get((kind, phase))

    def fake_anchored(bank_, cell, base, agent, traj, cache, *, snapshot="", phase=0, **kw):
        calls.append((base.key_id, traj, snapshot, phase))
        return _record("anchored", phase, agent=agent, snapshot=snapshot)

    monkeypatch.setattr(ps, "_W", dict(bank=bank, tables=None, cache=object()))
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": {}}, {}))
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: f"run{i}")
    monkeypatch.setattr(ak, "key_for", fake_key_for)
    monkeypatch.setattr(ak, "anchored_key", fake_anchored)
    job = dict(what="anchored", label="RA", agent="G1", timesteps=7, cell=cells[0],
               file="f.npz", phases=[(3, 11)])
    out = ps._anchored_job(job)
    assert "error" not in out and out["phases"] == [3]
    assert calls == [(ph3.key_id, "run11", "7", 3)]
    assert out["keys"][0]["phase"] == 3 and out["keys"][0]["kind"] == "anchored"
    out = ps._anchored_job(dict(job, phases=[(1, 9)]))            # no phase key at u_1
    assert "error" not in out and calls[-1] == (nom.key_id, "run9", "7", 1)


class _NullLog:
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

    def write(self, msg: str):
        pass


def test_R17a_anchored_keys_stage_solves_a_key_per_phase(ps, monkeypatch, tmp_path):
    """The anchored-keys stage at P0 hands the pool one re-anchored job per deterministic
    phase of every selected cell (16 on a P7 cell), each with its own row."""
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, (R, n_ph) in enumerate(((15.0, 4), (30.0, 16), (40.0, 4))):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        stims.append(Stimulus(i, f"s{i}", "synthetic", ("A",), 1, -1, lm, geo, None))
        cells.append(Cell(i, i, 10 + i, ("A",), "A1" if i < 2 else "A2", 86.0, 600.0, W=400,
                          P=4.0, n_phases=n_ph))
    bank = SimpleNamespace(stimuli=stims)
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9)}
    det = _det([(0, k) for k in range(4)] + [(1, k) for k in range(16)]
               + [(2, k) for k in range(4)])
    f = tmp_path / "G1.npz"
    f.write_bytes(b"x")
    pooled = []

    def fake_pool(jobs, workers, log, fn=None, init=None):
        pooled.extend(jobs)
        return [dict(label=j["label"], keys=[], cpu=0.0) for j in jobs]

    monkeypatch.setattr(ps, "_require_tag", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_design", lambda laid_out=False: (bank, cells))
    monkeypatch.setattr(support, "load_support", lambda d: SimpleNamespace(regimes=reg))
    monkeypatch.setattr(ps, "_manifest", lambda: SimpleNamespace(validation=V7_FAILED,
                                                                 budget={}, snapshots=[]))
    monkeypatch.setattr(ps, "_finals", lambda m: {"G1": SimpleNamespace(timesteps=7)})
    monkeypatch.setattr(measures, "model_tables", lambda: None)
    monkeypatch.setattr(ak, "KeyCache", lambda: None)
    monkeypatch.setattr(io, "StageLog", _NullLog)
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": det}, {}))
    monkeypatch.setattr(ps, "_beaten_jobs", lambda *a: ([], {"cells_checked": 0, "beaten": {}}))
    monkeypatch.setattr(A, "CachedKeys", lambda *a, **k: None)
    monkeypatch.setattr(A, "key_table", lambda *a, **k: {})
    monkeypatch.setattr(ps, "_run_table", lambda *a, **k: {"pair": np.array([10])})
    monkeypatch.setattr(A, "kappa_clipped_pairs", lambda rt: set())
    monkeypatch.setattr(A, "viability_targets", lambda t, kt: {})
    monkeypatch.setattr(ps, "_pool", fake_pool)
    monkeypatch.setattr(agents, "update_manifest", lambda fn: None)
    ps.stage_anchored_keys(SimpleNamespace(priority="P0", workers=1, force=False))
    got = [(j["cell"].cell_id, k, r) for j in pooled for k, r in j["phases"]]
    assert len(pooled) == 2                                 # one job per (agent, cell)
    # tail R 15 (4 phases) and bulk R 30 (P7: 16); the A2 cell is never re-anchored
    assert got == [(0, k, k) for k in range(4)] + [(1, k, 4 + k) for k in range(16)]


# ------------------------------------------------------------------------- (a) analysis


class _PhaseRAKeys:
    """u = 0 key v_e* 30 at every phase (no phase keys); re-anchored keys v_e* 31 + k at
    phases 0, 1 and 3 (none at phase 2: that run falls back). ``off`` returns phase 1's
    key at phase 3, to check that a key anchored on another phase's run is counted."""

    def __init__(self, off: bool = False):
        self.off = off

    def nominal(self, cell, phase):
        m = measures.ResponseMetrics(v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, has_brake=True)
        return A.KeyView(None, m, phase == 0, "verified")

    def backoff(self, cell):
        return A.KeyView(None, measures.ResponseMetrics(v_e=28.0), True, "verified")

    def anchored(self, cell, agent, ts, phase=0):
        k = 1 if self.off and phase == 3 else phase
        if k == 2:
            return None
        return A.KeyView(None, measures.ResponseMetrics(v_e=31.0 + k), True, "verified",
                         key_id=f"ra{k}", phase=k)

    def null(self, cell):
        return None

    def socp(self, cell):
        return None

    def no_return(self, *a):
        return None


def _ra_runs(monkeypatch, keys, approach: str = "A1"):
    bank, cells = _bank(approach=approach)
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: SimpleNamespace())
    monkeypatch.setattr(measures, "response_metrics", lambda *a, **k: measures.ResponseMetrics(
        v_e=33.0, v_min=25.0, d_b=95.0, has_brake=True))
    monkeypatch.setattr(measures, "key_errors", lambda m, mk, *a, **k: measures.KeyErrors(
        e_ve=m.v_e / mk.v_e - 1.0, dd_b=m.d_b - mk.d_b))
    n = 4
    det = dict(cell=np.zeros(n, int), stim=np.zeros(n, int), pair=np.zeros(n, int),
               phase=np.arange(n), sample=np.full(n, -1), steer_sign=np.ones(n, int),
               W=np.full(n, 400), s0=np.full(n, 400.0))
    nat = dict(det, phase=np.arange(6) % 4, sample=np.arange(6), cell=np.zeros(6, int),
               stim=np.zeros(6, int), pair=np.zeros(6, int), steer_sign=np.ones(6, int),
               W=np.full(6, 400), s0=np.full(6, 400.0))
    rt = A.run_table("G1", 7, {"det": det, "nat": nat}, cells, bank, keys, None,
                     check_design=False, speed_at_phase=True)
    return rt, A.key_table(cells, bank, keys)


def test_R17a_run_table_judges_each_run_against_its_phases_re_anchored_key(monkeypatch):
    rt, kt = _ra_runs(monkeypatch, _PhaseRAKeys())
    assert set(rt) == set(A.RUN_COLUMNS)
    det, nat = rt["mode"] == "det", rt["mode"] == "nat"
    # deterministic runs at phases 0..3: the key of their own phase; phase 2 falls back
    assert rt["e_ve_ra"][det] == pytest.approx([33 / 31 - 1, 33 / 32 - 1, 33 / 30 - 1,
                                                33 / 34 - 1])
    assert rt["ra_key"][det].tolist() == [True, True, False, True]
    assert rt["ra_phase"][det].tolist() == [0, 1, -1, 3]
    assert rt["ra_key_id"][det].tolist() == ["ra0", "ra1", "", "ra3"]
    assert rt["v_e_star_ra"][det].tolist() == [31.0, 32.0, 30.0, 34.0]
    # stochastic sample j at phase j mod 4 (D9) uses the key of that phase
    assert rt["phase"][nat].tolist() == [0, 1, 2, 3, 0, 1]
    assert rt["ra_phase"][nat].tolist() == [0, 1, -1, 3, 0, 1]
    assert rt["e_ve_ra"][nat] == pytest.approx([33 / 31 - 1, 33 / 32 - 1, 33 / 30 - 1,
                                                33 / 34 - 1, 33 / 31 - 1, 33 / 32 - 1])
    pv = A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)})
    assert bool(pv["uses_ra"][0]) and pv["ra_fallback"][0] == 1
    assert pv["ra_keys"][0] == 3 and pv["ra_off_phase"][0] == 0
    assert pv["v_e_star_ra"][0] == pytest.approx(np.median([31.0, 32.0, 30.0, 34.0]))
    # e_ve_bo_ra: each run's own e_ve_ra scaled by the cell's v_e*/v_e*bo; the fallback run
    # keeps its own e_ve_bo
    ebo = [(33 / 31) * 30 / 28 - 1, (33 / 32) * 30 / 28 - 1, 33 / 28 - 1, (33 / 34) * 30 / 28 - 1]
    assert pv["e_ve_bo_ra"][0] == pytest.approx(np.median(ebo))
    assert pv["e_ve_ra"][0] == pytest.approx(np.median(rt["e_ve_ra"][det]))


def test_R17a_a_key_anchored_on_another_phases_run_is_counted(monkeypatch):
    rt, kt = _ra_runs(monkeypatch, _PhaseRAKeys(off=True))
    det = rt["mode"] == "det"
    assert rt["ra_phase"][det].tolist() == [0, 1, -1, 1]
    pv = A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)})
    assert pv["ra_off_phase"][0] == 1 and pv["ra_keys"][0] == 2


def test_R17a_budget_counts_one_re_anchored_key_per_phase(ps, monkeypatch):
    """P0's re-anchored keys on the design are one per deterministic phase (4; 16 on P7
    cells), about 4x the cells; the specialists', tier B's and checkpoints' protocol
    estimates are multiplied by the 4 phases; both are recorded in the projection."""
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, (R, n_ph, ap) in enumerate(((10.0, 4, "A1"), (15.0, 4, "A1"), (30.0, 16, "A1"),
                                       (30.0, 4, "A2"))):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        stims.append(Stimulus(i, f"s{i}", "synthetic", ("A",), 1, -1, lm, geo, None))
        cells.append(Cell(i, i, 10 + i, ("A",), ap, 86.0, 600.0, W=400, P=4.0, n_phases=n_ph))
    bank = SimpleNamespace(stimuli=stims)
    reg = {"G1": SimpleNamespace(q1=12.0, q10=20.0, r_clip_max=8.9),
           "G2": SimpleNamespace(q1=9.0, q10=12.0, r_clip_max=7.6)}
    monkeypatch.setattr(support, "load_support", lambda d: SimpleNamespace(regimes=reg))
    g = ps._anchored_g_count(bank, cells)
    # G1: below 10, tail 15, bulk 30 (P7); G2: tail 10, bulk 15 and 30 (P7)
    assert g["cells"] == 6 and g["u0_keys"] == 6 and g["n"] == 4 + 4 + 16 + 4 + 4 + 16
    assert g["p7_cells"] == 2 and g["p7_keys"] == 32
    assert g["r12_below_support"] == 1 and g["r12_below_support_keys"] == 4
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: True)
    proj = ps._budget(bank, cells, True, SimpleNamespace(validation=V7_FAILED, budget={}), 1,
                      cache=object())
    it = proj["items"]
    assert it["anchored_G"]["n"] == g["n"] and it["anchored_G"]["item"] == "anchored"
    # review finding 2: P1 and P2 on the design. Specialists: the 3 A1 cells each (4 + 4 +
    # 16 phases); no tier B cell; checkpoints: G1's 2 and G2's 3 tail/bulk cells x 3 x 4
    assert (it["anchored_S"]["n"], it["anchored_B"]["n"], it["anchored_ckpt"]["n"]) == (
        72, 0, 60)
    ra = proj["anchored_phases"]
    assert ra["anchored_G"]["keys"] == g["n"] and ra["anchored_G"]["cells"] == 6
    assert ra["anchored_S"]["u0_keys"] == 9 and ra["anchored_S"]["source"] == "design"
    assert ra["anchored_ckpt"]["keys"] == 60 and ra["anchored_ckpt"]["u0_keys"] == 15
    # before the regimes are frozen the checkpoints fall back to the protocol's 216 x 4
    monkeypatch.setattr(ps, "_frozen_regimes", lambda: None)
    assert ps._anchored_p12_count(bank, cells)["anchored_ckpt"]["keys"] == 864


def test_R17a_budget_counts_p1_anchored_keys_as_the_stage_selects_them(ps):
    """Review finding 2: the specialists' finals re-anchor every A1 and tier B D = 600 cell
    and the generalists' finals every tier B D = 600 cell, at every phase (16 on P7 cells),
    as ``anchored_cells`` selects them; tier B at another D and A2 are not counted."""
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    rows = (("A1", ("A",), 600.0, 4, ()), ("A1", ("A",), 600.0, 16, ("p7",)),
            ("B", ("B",), 600.0, 4, ()), ("B", ("B",), 300.0, 4, ()),
            ("A2", ("A",), 600.0, 4, ()))
    stims, cells = [], []
    for i, (ap, tiers, D, n_ph, flags) in enumerate(rows):
        geo = Geometry(20.0, 90.0, np.nan, 20.0, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        stims.append(Stimulus(i, f"s{i}", "synthetic", tiers, 1, -1, lm, geo, None))
        cells.append(Cell(i, i, 10 + i, tiers, ap, 86.0, D, W=400, P=4.0, n_phases=n_ph,
                          flags=flags))
    got = ps._anchored_p12_count(SimpleNamespace(stimuli=stims), cells)
    assert got["anchored_S"]["keys"] == 3 * (4 + 16 + 4) and got["anchored_S"]["u0_keys"] == 9
    assert got["anchored_S"]["p7_keys"] == 3 * 16
    assert got["anchored_B"]["keys"] == 3 * 4 and got["anchored_B"]["u0_keys"] == 3
    assert got["anchored_B"]["protocol_u0"] == 108
    # the anchored stage solves tier B D = 600's missing nominal keys: budgeted once per cell
    m = SimpleNamespace(validation={}, budget={})
    it = ps._budget(SimpleNamespace(stimuli=stims), cells, True, m, 1)["items"]
    assert it["nominal_B600"]["n"] == 1 and it["nominal_B600"]["priority"] == "P1"


# ------------------------------------------------------------------------- (b) P7


_P7_ROWS = {
    0: ("A1", (9.4, 90.0), 16),       # P7, core (R <= 18.3)
    1: ("A1", (28.6, 90.0), 16),      # P7, not core; flat by its key
    2: ("A1", (28.6, 180.0), 16),     # P7, 180 deg
    3: ("A1", (22.9, 90.0), 4),       # tier A A1, braking
    4: ("A1", (13.1, 90.0), 4),       # core, 4 phases
    5: ("A2", (28.6, 90.0), 4),       # A2, braking
}


def _p7_design():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, (ap, (R, th), n_ph) in _P7_ROWS.items():
        geo = Geometry(R, th, np.nan, R, th, 40.0, 0.0, 1020.0, 1, False, 6.0)
        spec = CornerSpec(R=R, theta_deg=th, direction=1, width=12.0, l_r=20.0)
        stims.append(Stimulus(i, f"syn/R{R}/T{th}/L/{ap}", "synthetic", ("A",), 1, -1, lm, geo,
                              spec))
        cells.append(Cell(i, i, i, ("A",), ap, 86.0, 600.0, W=400, P=4.0, n_phases=n_ph,
                          flags=("p7",) if n_ph == 16 else ()))
    return SimpleNamespace(stimuli=stims), cells


def test_R17b_phase_key_indices_cover_every_phase_and_v7_keeps_u1_u3(ps):
    c4 = Cell(0, 0, 0, ("A",), "A1", 86.0, 600.0, W=400, P=4.0)
    c16 = Cell(1, 1, 1, ("A",), "A1", 86.0, 600.0, W=400, P=4.0, n_phases=16)
    assert ps._phase_key_indices(c4) == [1, 2, 3]
    assert ps._phase_key_indices(c16) == list(range(1, 16))
    assert ps._phase_indices(c4) == [1, 2, 3] and ps._phase_indices(c16) == [4, 8, 12]
    assert ps._p7_null_phases(c16) == [0, 4, 8, 12] and ps._p7_null_phases(c4) == [0, 1, 2, 3]
    assert ps._p7_cell(c16) and not ps._p7_cell(c4)


def test_R17b_validation_v7_still_solves_u1_u3(ps, monkeypatch):
    """Validation V7 (6.5) is unchanged by R17b: phases 4, 8 and 12 of a 16-phase cell."""
    c16 = Cell(1, 1, 1, ("A",), "A1", 86.0, 600.0, W=400, P=None, n_phases=16)
    nom = _record("nominal", 0)
    ks = []
    monkeypatch.setattr(ak, "full_recipe", lambda *a, **k: nom)
    monkeypatch.setattr(ak, "phase_cycle", lambda bank, cell, nominal=None: 4.0)

    def fake_warm(base, bank, cell, kind, phase=0, **kw):
        ks.append(phase)
        return _record("phase", phase)

    monkeypatch.setattr(ak, "warm_key", fake_warm)
    out, P = ps._validate_v7_cell(None, c16, None, None)
    assert ks == [4, 8, 12] and P == 4.0 and len(out["keys"]) == 4


def test_R17b_p7_cells_get_phase_keys_at_every_phase(ps, monkeypatch):
    """P7's cells get phase-matched keys at all 15 phases besides u = 0, whatever V7 says
    and whatever their braking class; other cells keep u_1..u_3."""
    bank, cells = _p7_design()
    braking = {0: True, 1: False, 2: True, 3: True, 4: True, 5: True}
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: braking[c.cell_id])
    monkeypatch.setattr(ps, "_anchored_g_count", lambda bank, cells: dict(
        n=0, source="test", r12_below_support=0, protocol=96))
    cache = object()
    # V7 holds: 6.1's core set plus every P7 cell
    held = ps._key_sets(bank, cells, True)
    assert _ids(held["phase"]) == [0, 1, 2, 4]
    jobs = ps._phase_jobs(bank, held["phase"], cache, {"cuts": []}, "phase", False)
    assert [(_ids(j["cells"]), j["phases"]) for j in jobs] == [
        ([0], list(range(1, 16))), ([1], list(range(1, 16))), ([2], list(range(1, 16))),
        ([4], [1, 2, 3])]
    # V7 failed: the braking candidates, the core cells and every P7 cell (1 is flat)
    failed = ps._key_sets(bank, cells, True, v7_failed=True)
    msgs: list[str] = []
    jobs = ps._phase_jobs(bank, failed["phase"], cache, {"cuts": []}, "phase", True,
                          SimpleNamespace(write=msgs.append))
    assert [(_ids(j["cells"]), len(j["phases"])) for j in jobs] == [
        ([0], 15), ([1], 15), ([2], 15), ([3], 3), ([4], 3), ([5], 3)]
    assert "54 phase-matched keys, 45 of them on 3 P7 cells" in msgs[0]
    # budget: 15 keys per P7 cell, 3 per other cell
    proj = ps._budget(bank, cells, True, SimpleNamespace(validation=V7_FAILED, budget={}), 1,
                      cache=cache)
    pk = proj["phase_keys"]
    assert proj["items"]["phase"]["n"] == 3 * 15 + 3 * 3 == pk["keys"]["phase"]
    assert pk["p7_cells"]["phase"] == 3 and pk["p7_keys"]["phase"] == 45
    assert pk["p7_beyond_u1_u3"]["phase"] == 36
    held_proj = ps._budget(bank, cells, True, SimpleNamespace(validation=V7_PASSED, budget={}),
                           1, cache=cache)
    assert held_proj["items"]["phase"]["n"] == 3 * 15 + 3
    # budget rule 2 keeps 180 deg cells only: its credit is what ``_cut`` drops, 15 per
    # non-180 P7 cell
    sets = ps._key_sets(bank, cells, True, v7_failed=True)
    counted = {k: ps._phase_targets(bank, sets[k], cache, True, unknown=True)
               for k in ps.PHASE_ITEMS}
    credit = ps._rule2_cut_counts(bank, sets, counted)
    rule2 = {"cuts": ["rule 2: x"]}
    kept = ps._cut(bank, counted["phase"], rule2, "phase")
    assert _ids(kept) == [2]
    assert credit["phase"] == sum(len(ps._phase_key_indices(c)) for c in counted["phase"]
                                  if c not in kept) == 15 + 15 + 3 + 3 + 3


def test_R17b_p7_null_keeps_phases_0_4_8_12(ps, monkeypatch):
    """With keys at every phase, P7's key null is still the SD over u = 0, 4, 8 and 12."""
    bank, cells = _p7_design()
    asked = []

    def fake_key_for(c, kind, k=0, **kw):
        asked.append((c.cell_id, kind, k))
        return SimpleNamespace(status="verified", measures={"d_b": 100.0 + k + (k % 4) * 7})

    monkeypatch.setattr(ak, "key_for", fake_key_for)
    kt = {"cell": np.arange(6), "geometry": np.array([f"g{i}" for i in range(6)])}
    sd = ps._key_phase_sd(kt, cells, None, bank)
    assert set(sd) == {"g0"}                       # P7 and R <= 18.3 only
    assert sd["g0"] == pytest.approx(A.phase_spread([100.0, 104.0, 108.0, 112.0]))
    assert {k for _, _, k in asked} == {0, 4, 8, 12}


def test_R17b_analysis_reads_the_key_at_each_of_the_16_phases(monkeypatch, tmp_path):
    """Phase-matched keys solved at all 16 phases are found by ``CachedKeys`` under each
    phase's problem id, so every one of the 16 deterministic runs of a P7 cell is judged
    against the key at its own phase (key_at_phase, v_e_star_k, onset and, under the V7
    remedy, speed errors)."""
    bank, cells = _bank(n_phases=16)
    cache = ak.KeyCache(tmp_path / "keys")
    c = cells[0]
    nom_id = ak.expected_key_id(bank, c, "nominal", 0)
    cache.put(_record("nominal", 0, key_id=nom_id, x0=ak._cell_x0(c, 0),
                      measures_={"v_e": 30.0, "v_min": 25.0, "d_b": 100.0, "v_b": 60.0,
                                 "has_brake": True}))
    for k in range(1, 16):
        cache.put(_record("phase", k, key_id=ak.expected_key_id(bank, c, "phase", k),
                          x0=ak._cell_x0(c, k),
                          measures_={"v_e": 30.0 + k / 10, "v_min": 25.0, "d_b": 100.0 + k,
                                     "v_b": 60.0, "has_brake": True}))
    keys = A.CachedKeys(bank, cells, cache, recompute=False)
    assert all(keys.nominal(c, k).at_phase for k in range(16))
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: SimpleNamespace())
    monkeypatch.setattr(measures, "response_metrics", lambda *a, **k: measures.ResponseMetrics(
        v_e=33.0, v_min=25.0, d_b=95.0, has_brake=True))
    monkeypatch.setattr(measures, "key_errors", lambda m, mk, *a, **k: measures.KeyErrors(
        e_ve=m.v_e / mk.v_e - 1.0, dd_b=m.d_b - mk.d_b))
    n = 16
    det = dict(cell=np.zeros(n, int), stim=np.zeros(n, int), pair=np.zeros(n, int),
               phase=np.arange(n), sample=np.full(n, -1), steer_sign=np.ones(n, int),
               W=np.full(n, 400), s0=np.full(n, 400.0))
    rt = A.run_table("G1", 7, {"det": det}, cells, bank, keys, None, check_design=False,
                     speed_at_phase=True)
    assert rt["key_at_phase"].all()
    assert rt["v_e_star_k"] == pytest.approx(30.0 + np.arange(16) / 10)
    assert rt["dd_b"] == pytest.approx(95.0 - (100.0 + np.arange(16)))
    assert rt["e_ve"] == pytest.approx(33.0 / (30.0 + np.arange(16) / 10) - 1.0)
