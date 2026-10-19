"""Protocol 6.5's V7 remedy: when V7 fails on speeds, phase-matched keys for every braking
cell with an agent-free nominal key, and the speed errors judged against the key at the
run's phase (analysis AD23). Nothing is solved or rolled out: keys and runs are doubles,
banks are hand-built, and no file is written."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.interp import agents, io, measures, stimuli
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

V7_FAILED = {"V5": {"passed": True}, "V7": {"passed": False, "name": "V7",
                                             "details": {"cells": {}, "failed": [12, 136]}}}
V7_PASSED = {"V5": {"passed": True}, "V7": {"passed": True, "name": "V7"}}


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

    def write(self, msg: str):
        pass


# ------------------------------------------------------------------------- design

# cell -> (tiers, approach, name, spec (R, theta) or None, source); one direction each.
_ROWS = {
    0: (("A",), "A1", "syn/R9.4/T90/L", (9.4, 90.0), {}),          # 6.1's core set
    1: (("A",), "A1", "syn/R28.6/T90/L", (28.6, 90.0), {}),        # braking
    2: (("A",), "A2", "syn/R28.6/T90/L2", (28.6, 90.0), {}),       # flat
    3: (("A",), "A1", "syn/R44.7/T180/L", (44.7, 180.0), {}),      # nominal not solved yet
    4: (("A", "B"), "A1", "syn/R14.6/T180/L", (14.6, 180.0), {}),  # core, shared with B
    5: (("E1",), "real", "E1/c/real", None, {}),                   # E1 real rung, braking
    6: (("E1",), "real", "E1/c/twin", (20.0, 90.0), {}),           # E1 twin (P1), braking
    7: (("E2",), "real", "E2/c/twin", (30.0, 90.0), {}),           # E2 (P1), flat
    8: (("E3",), "A2", "E3/s/L", (12.0, 180.0), {}),               # E3 (P1), braking
    9: (("C",), "A2", "C/R22.9/T45/L", (22.9, 45.0), {}),          # tier C (P2), braking
    10: (("B",), "REC", "B/rec/L", (9.4, 180.0), {}),              # recovery (P2), braking
    11: (("G",), "G", "straight/W12", None, {}),                   # null: never
    12: (("D",), "real", "D/3/F1/real/real/real", None,
         {"condition": "F1/real/real/real", "corner_index": 3}),   # s-OCP only: never
    13: (("B",), "B", "B/R9.4/T90/L", (9.4, 90.0), {}),            # tier B: no nominal key
    14: (("V",), "A2", "V/R40/T90/L", (40.0, 90.0), {}),           # validation cell, braking
}
# 2.3's braking class of each cell's nominal key once solved (core cell 0 says False: the
# core set keeps its keys whatever the class).
_BRAKING = {0: False, 1: True, 2: False, 3: True, 4: True, 5: True, 6: True, 7: False,
            8: True, 9: True, 10: True, 14: True}


def _design():
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, (tiers, ap, name, rt, src) in _ROWS.items():
        R, th = rt if rt is not None else (20.0, 90.0)
        geo = Geometry(R, th, np.nan, R, th, 40.0, 0.0, 1020.0, 1, False, 6.0)
        spec = (CornerSpec(R=R, theta_deg=th, direction=1, width=12.0, l_r=20.0)
                if rt is not None else None)
        stims.append(Stimulus(i, name, "synthetic", tiers, 0 if "G" in tiers else 1, -1, lm,
                              geo, spec, source=src))
        cells.append(Cell(i, i, i, tiers, ap, 80.0, 600.0, W=400, P=4.0,
                          runners=("S1",) if "D" in tiers else ()))
    return SimpleNamespace(stimuli=stims), cells


def _ids(cs) -> list[int]:
    return [c.cell_id for c in cs]


def test_v7_verdict_is_read_from_the_manifest_validation(ps):
    assert A.v7_speeds_failed(V7_FAILED) and not A.v7_speeds_failed(V7_PASSED)
    assert not A.v7_speeds_failed({}) and not A.v7_speeds_failed(None)      # not run yet
    assert ps._v7_failed(SimpleNamespace(validation=V7_FAILED))
    assert not ps._v7_failed(SimpleNamespace(validation={"V5": {"passed": False}}))


def test_v7_that_could_not_be_judged_does_not_trigger_the_remedy():
    """Review of the V7 remedy: V7 with no judged cell (no phase-matched validation keys)
    or an error has passed False and fired the remedy, though no speed failure was
    measured. Only a V7 that lists failed cells triggers it."""
    assert not A.v7_speeds_failed({"V7": io._jsonable(ak.v7_phase([]))})
    assert not A.v7_speeds_failed({"V7": {"passed": False, "error": "Traceback ..."}})
    assert not A.v7_speeds_failed({"V7": {"passed": False, "details": {"failed": []}}})
    assert A.v7_speeds_failed(V7_FAILED)


def test_v7_remedy_log_counts_cells_without_a_nominal_key_apart(ps, monkeypatch):
    """A candidate whose nominal job errored has no record: it is reported apart from the
    flat and unsolved cells (a resumed keys run picks it up)."""
    bank, cells = _design()
    solved = set(_BRAKING) - {3}
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: _BRAKING.get(c.cell_id)
                        if c.cell_id in solved else None)
    sets = ps._key_sets(bank, cells, False, v7_failed=True)
    msgs: list[str] = []
    jobs = ps._phase_jobs(bank, sets["phase"], object(), {"cuts": []}, "phase", True,
                          SimpleNamespace(write=msgs.append))
    assert _ids(c for j in jobs for c in j["cells"]) == [0, 1, 4, 5, 14]
    assert len(msgs) == 1 and "5 of 7 candidate cells get keys" in msgs[0]
    assert "1 flat or unsolved skipped" in msgs[0] and "1 without a nominal key" in msgs[0]


def test_v7_remedy_sets_every_nominal_key_cell_and_old_set_when_v7_passes(ps):
    """V7 holds: protocol 6.1's set (A1, R <= 18.3) only. V7 failed: every cell with an
    agent-free nominal key, at its nominal key's priority; nulls, tier D and tier B cells
    without a nominal key never."""
    bank, cells = _design()
    held = ps._key_sets(bank, cells, both=False)
    assert _ids(held["phase"]) == [0, 4] and held["phase_P1"] == held["phase_P2"] == []
    assert ps._key_sets(bank, cells, False, v7_failed=False)["phase"] == held["phase"]
    failed = ps._key_sets(bank, cells, both=False, v7_failed=True)
    assert _ids(failed["phase"]) == [0, 1, 2, 3, 4, 5, 14]      # tier A, E1 real, tier V
    assert _ids(failed["phase_P1"]) == [6, 7, 8]                # E1 twin, E2, E3
    assert _ids(failed["phase_P2"]) == [9, 10]                  # tier C, recovery
    every = {i for k in ps.PHASE_ITEMS for i in _ids(failed[k])}
    assert not every & {11, 12, 13}
    # the other items are untouched by V7
    assert {k: _ids(v) for k, v in held.items() if k not in ps.PHASE_ITEMS} == {
        k: _ids(v) for k, v in failed.items() if k not in ps.PHASE_ITEMS}


def test_v7_remedy_skips_non_braking_and_unsolved_cells(ps, monkeypatch):
    bank, cells = _design()
    solved = set(_BRAKING) - {3}
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: _BRAKING.get(c.cell_id)
                        if c.cell_id in solved else None)
    sets = ps._key_sets(bank, cells, False, v7_failed=True)
    cache = object()
    # Jobs: the core cells (even cell 0, whose class says flat) and the braking candidates;
    # the flat cells 2 and 7 and the unsolved cell 3 get none.
    assert _ids(ps._phase_targets(bank, sets["phase"], cache, True)) == [0, 1, 4, 5, 14]
    assert _ids(ps._phase_targets(bank, sets["phase_P1"], cache, True)) == [6, 8]
    assert _ids(ps._phase_targets(bank, sets["phase_P2"], cache, True)) == [9, 10]
    # The budget's upper bound counts the not-yet-solved cell 3, never the flat ones.
    assert _ids(ps._phase_targets(bank, sets["phase"], cache, True, unknown=True)) == [
        0, 1, 3, 4, 5, 14]
    keep, unk = ps._phase_split(bank, sets["phase"], cache, True)
    assert _ids(unk) == [3]
    # While V7 holds, the old set is kept as it is (no braking filter).
    held = ps._key_sets(bank, cells, False)["phase"]
    assert ps._phase_targets(bank, held, cache, False) == held
    # Jobs at u_1..u_3 (4, 8, 12 on 16-phase cells), no rule 2 cut.
    jobs = ps._phase_jobs(bank, sets["phase"], cache, {"cuts": []}, "phase", True)
    assert [(j["what"], _ids(j["cells"]), j["phases"]) for j in jobs] == [
        ("phase", [c], [1, 2, 3]) for c in (0, 1, 4, 5, 14)]


def _manifest(validation, timings=None):
    return SimpleNamespace(validation=validation, budget={"timings": timings} if timings else {})


def test_v7_remedy_budget_counts_the_expanded_set_and_rule_2_credit(ps, monkeypatch):
    bank, cells = _design()
    solved = set(_BRAKING) - {3}
    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: _BRAKING.get(c.cell_id)
                        if c.cell_id in solved else None)
    monkeypatch.setattr(ps, "_anchored_g_count", lambda bank, cells: dict(
        n=0, source="test", r12_below_support=0, protocol=96))
    cache = object()
    held = ps._budget(bank, cells, False, _manifest(V7_PASSED), 1, cache=cache)
    assert held["items"]["phase"]["n"] == 3 * 2 and "phase_P1" not in held["items"]
    assert held["phase_keys"]["v7_failed"] is False
    proj = ps._budget(bank, cells, False, _manifest(V7_FAILED), 1, cache=cache)
    it = proj["items"]
    # P0: core 0 and 4, braking 1, 5 and 14, unsolved 3 (an upper bound); P1: 6, 8; P2: 9, 10.
    assert it["phase"]["n"] == 18 and it["phase"]["priority"] == "P0"
    assert (it["phase_P1"]["n"], it["phase_P1"]["priority"]) == (6, "P1")
    assert (it["phase_P2"]["n"], it["phase_P2"]["priority"]) == (6, "P2")
    assert it["phase_P1"]["item"] == it["phase_P2"]["item"] == "warm"
    assert proj["phase_keys"]["braking_unknown"] == {"phase": 1, "phase_P1": 0, "phase_P2": 0}
    # Without a cache nothing is known yet: every candidate counts.
    nocache = ps._budget(bank, cells, False, _manifest(V7_FAILED), 1)
    assert nocache["items"]["phase"]["n"] == 3 * 7
    # Rule 2's credit is exactly what ``_cut`` drops from the counted cells: 180 deg cells
    # only, never a budget-cut cell (3: R 44.7 / 180 has no nominal key) or a real road (5).
    sets = ps._key_sets(bank, cells, False, v7_failed=True)
    counted = {k: ps._phase_targets(bank, sets[k], cache, True, unknown=True)
               for k in ps.PHASE_ITEMS}
    rule2 = {"cuts": ["rule 2: x"]}
    credit = ps._rule2_cut_counts(bank, sets, counted)
    for k, cs in counted.items():
        assert credit[k] == 3 * (len(cs) - len(ps._cut(bank, cs, rule2, k)))
    assert _ids(ps._cut(bank, counted["phase"], rule2, "phase")) == [4]
    assert (credit["phase"], credit["phase_P1"], credit["phase_P2"]) == (15, 3, 3)
    # Old behaviour while V7 holds: the credit is the non-180 deg core cells.
    assert ps._rule2_cut_counts(bank, ps._key_sets(bank, cells, False))["phase"] == 3
    # Rule 2 firing (P0 > 35 CPU-h) takes that credit off each phase item.
    cut = ps._budget(bank, cells, False, _manifest(V7_FAILED, {"nominal": 1e5}), 1,
                     cache=cache)
    assert any(c.startswith("rule 2") for c in cut["cuts"])
    assert cut["items"]["phase"]["n"] == 3 * 1 and cut["items"]["phase_P1"]["n"] == 3 * 1


def test_v7_remedy_keys_stage_solves_phase_keys_after_their_nominal_keys(ps, monkeypatch):
    """The keys stage plans each item's phase-matched keys after its nominal keys are
    solved (P0 after tier A and E1, P1 after E1 twins / E2 / E3, P2 after tier C and
    recovery) and skips the flat ones; while V7 holds only the core set is solved."""
    bank, cells = _design()
    solved: set[int] = {14}             # the validation cell's nominal key: validate stage
    calls: list[list[tuple]] = []

    def fake_pool(jobs, workers, log, fn=None, init=None):
        calls.append([(j["what"], tuple(_ids(j.get("cells", [])))) for j in jobs])
        for j in jobs:
            if j["what"] in ("series", "nominal"):
                solved.update(_ids(j["cells"]))
        return [dict(what=j["what"], label=j["label"], keys=[], cpu=0.0) for j in jobs]

    monkeypatch.setattr(ps, "_nominal_braking", lambda b, c, cache: _BRAKING.get(c.cell_id)
                        if c.cell_id in solved else None)
    monkeypatch.setattr(ps, "_anchored_g_count", lambda bank, cells: dict(
        n=0, source="test", r12_below_support=0, protocol=96))
    monkeypatch.setattr(ps, "_pool", fake_pool)
    monkeypatch.setattr(ps, "_refuse_after_tag", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_require_steps", lambda *a, **k: None)
    monkeypatch.setattr(ps, "_design", lambda laid_out=False: (bank, cells))
    monkeypatch.setattr(ps, "_tuned_cv", lambda: None)
    monkeypatch.setattr(ps, "_flag_budget_cuts", lambda bank, cells, proj, log: cells)
    monkeypatch.setattr(ps, "_save_cells", lambda bank, cells, log, why: ("h", cells))
    monkeypatch.setattr(ps, "_tier_d_doses", lambda *a: [])
    monkeypatch.setattr(ps, "_acceptance", lambda *a: None)
    monkeypatch.setattr(measures, "model_tables", lambda: None)
    monkeypatch.setattr(io, "StageLog", _NullLog)
    monkeypatch.setattr(agents, "update_manifest", lambda fn: None)
    monkeypatch.setattr(ak, "KeyCache", lambda: SimpleNamespace(
        get_flag=lambda name, default=None: True))                   # mirror shortcut on
    monkeypatch.setattr(ak, "_x64", lambda: None)
    monkeypatch.setattr(ak, "key_for", lambda *a, **k: None)          # no backed-off keys
    monkeypatch.setattr(ak, "refresh_phase_cycles", lambda bank, cells, cache: cells)
    monkeypatch.setattr(ak, "refresh_stale", lambda *a: {"refreshed": [], "moved": []})
    args = SimpleNamespace(priority="P2", workers=1, smoke=False, force=False,
                           layout_only=False)

    def phase_calls():
        return [sorted(c for w, cs in call if w == "phase" for c in cs) for call in calls]

    monkeypatch.setattr(ps, "_manifest", lambda: _manifest(V7_FAILED))
    ps.stage_keys(args)
    assert len(calls) == 6
    # P0's pool solved cell 3's nominal key too (it brakes): flat 2, 7 and nulls get none
    assert phase_calls() == [[], [0, 1, 3, 4, 5, 14], [], [6, 8], [], [9, 10]]
    # each phase pool comes after the pool that solved its cells' nominal keys
    nominal_in = [{c for w, cs in call if w in ("series", "nominal") for c in cs}
                  for call in calls]
    assert {0, 1, 2, 3, 4, 5} <= nominal_in[0] and {6, 7, 8} <= nominal_in[2]
    assert {9, 10} <= nominal_in[4]

    calls.clear()
    solved.clear()
    solved.add(14)
    monkeypatch.setattr(ps, "_manifest", lambda: _manifest(V7_PASSED))
    ps.stage_keys(args)
    assert phase_calls() == [[], [0, 4], [], []]


# ------------------------------------------------------------------------- run table


def _bank():
    """One R40/90 left corner cell at A2 (no re-anchoring) and its null straight."""
    spec = CornerSpec(R=40.0, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
    pc, pst = stimuli.corner_profile(spec), stimuli.straight_profile(12.0)
    geo = Geometry(40.0, 90.0, 1 / 800, 40.0, 90.0, spec.L_c, 0.0, 1030.0, 1, False, 6.0)
    stims = [Stimulus(0, "syn/R40/T90/W12/typical/L", "synthetic", ("A",), 1, -1, pc.landmarks,
                      geo, spec),
             Stimulus(1, "straight/W12/test", "straight", ("G",), 0, -1, pst.landmarks, geo, None)]
    bank = StimulusBank(np.stack([pc.kappa, pst.kappa]), np.stack([pc.w_left, pst.w_left]),
                        np.stack([pc.w_right, pst.w_right]), stims)
    cells = [Cell(0, 0, 0, ("A",), "A2", 60.0, 300.0, W=200, P=3.0, null_cell=1),
             Cell(1, 1, 1, ("G",), "G", 60.0, 600.0, W=200, P=3.0)]
    return bank, cells


class _V7Keys:
    """u = 0 key: v_e* 30, v_min* 25, e_vmin_line* 0. Phases 1 and 3 have phase-matched keys
    (v_e* 33, v_min* 24, e_vmin_line* 0.1); phase 2 has none (the u = 0 key, not at phase).
    Backed-off: 28 at u = 0, and 27 solved at phase 1 only."""

    def nominal(self, cell, phase):
        if phase in (1, 3):
            m = measures.ResponseMetrics(v_e=33.0, v_min=24.0, d_b=101.0, v_b=60.0,
                                         has_brake=True, e_vmin_line=0.1)
            return A.KeyView(None, m, True, "verified")
        m = measures.ResponseMetrics(v_e=30.0, v_min=25.0, d_b=100.0, v_b=60.0, has_brake=True,
                                     e_vmin_line=0.0)
        return A.KeyView(None, m, phase == 0, "verified")

    def backoff(self, cell):
        return A.KeyView(None, measures.ResponseMetrics(v_e=28.0), True, "verified")

    def backoff_at(self, cell, phase):
        if phase != 1:
            return None
        return A.KeyView(None, measures.ResponseMetrics(v_e=27.0), True, "verified")

    def anchored(self, cell, agent, ts, phase=0):
        return None

    def null(self, cell):
        return None

    def socp(self, cell):
        return None

    def no_return(self, *a):
        return None


def _v7_runs(monkeypatch, speed_at_phase: bool):
    bank, cells = _bank()
    monkeypatch.setattr(measures, "trajectory_from_rollout", lambda arr, i, g: SimpleNamespace())
    monkeypatch.setattr(measures, "response_metrics", lambda *a, **k: measures.ResponseMetrics(
        v_e=33.0, v_min=25.0, d_b=95.0, has_brake=True, e_vmin_line=0.2))
    monkeypatch.setattr(measures, "key_errors", lambda m, mk, *a, **k: measures.KeyErrors(
        e_ve=m.v_e / mk.v_e - 1.0, dd_b=m.d_b - mk.d_b))
    n = 4
    det = dict(cell=np.zeros(n, int), stim=np.zeros(n, int), pair=np.zeros(n, int),
               phase=np.arange(n), sample=np.full(n, -1), steer_sign=np.ones(n, int),
               W=np.full(n, 200), s0=np.full(n, 700.0))
    rt = A.run_table("G1", 1, {"det": det}, cells, bank, _V7Keys(), None, check_design=False,
                     speed_at_phase=speed_at_phase)
    return rt, A.key_table(cells, bank, _V7Keys())


def test_v7_remedy_run_table_judges_speeds_against_the_phase_key(monkeypatch):
    rt, kt = _v7_runs(monkeypatch, speed_at_phase=True)
    assert set(rt) == set(A.RUN_COLUMNS) and rt["speed_at_phase"].all()
    # phases 1 and 3: the phase-matched key; 0 and 2: the u = 0 key (2 has no phase key)
    assert rt["e_ve"] == pytest.approx([33 / 30 - 1, 0.0, 33 / 30 - 1, 0.0])
    assert rt["e_vmin"] == pytest.approx([0.0, 25 / 24 - 1, 0.0, 25 / 24 - 1])
    assert rt["e_vmin_line_excess"] == pytest.approx([0.2, 0.1, 0.2, 0.1])
    assert rt["key_at_phase"].tolist() == [True, True, False, True]
    assert rt["v_e_star_k"].tolist() == [30.0, 33.0, 30.0, 33.0]
    # onsets unchanged: the key at the run's phase
    assert rt["dd_b"] == pytest.approx([-5.0, -6.0, -5.0, -6.0])
    # without re-anchoring, e_ve_ra and v_e*RA follow the key behind e_ve
    assert rt["e_ve_ra"] == pytest.approx(rt["e_ve"]) and not rt["ra_key"].any()
    assert rt["v_e_star_ra"].tolist() == [30.0, 33.0, 30.0, 33.0]
    # e_ve_bo: the backed-off key at the run's phase where one exists (phase 1), else u = 0
    assert rt["e_ve_bo"] == pytest.approx([33 / 28 - 1, 33 / 27 - 1, 33 / 28 - 1, 33 / 28 - 1])
    assert rt["bo_at_phase"].tolist() == [True, True, False, False]
    # S and r's v_e* is the runs' median phase-matched v_e* (AD5 with AD23's remedy)
    pv = A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)})
    assert bool(pv["speed_at_phase"][0]) and not bool(pv["uses_ra"][0])
    assert pv["v_e_star_band"][0] == pytest.approx(31.5) and pv["v_e_star"][0] == 30.0
    assert pv["e_ve"][0] == pytest.approx(np.median(rt["e_ve"]))


def test_v7_holding_keeps_speed_errors_on_the_u0_key(monkeypatch):
    rt, kt = _v7_runs(monkeypatch, speed_at_phase=False)
    assert not rt["speed_at_phase"].any()
    assert rt["e_ve"] == pytest.approx([33 / 30 - 1] * 4)
    assert rt["e_vmin"] == pytest.approx([0.0] * 4)
    assert rt["e_vmin_line_excess"] == pytest.approx([0.2] * 4)
    assert rt["v_e_star_ra"].tolist() == [30.0] * 4
    assert rt["e_ve_bo"] == pytest.approx([33 / 28 - 1] * 4)
    assert rt["bo_at_phase"].tolist() == [True, False, False, False]
    assert rt["dd_b"] == pytest.approx([-5.0, -6.0, -5.0, -6.0])     # onsets phase-matched
    pv = A.pair_values(rt, kt, {"G1": A.Regime(12.0, 20.0)})
    assert not bool(pv["speed_at_phase"][0]) and pv["v_e_star_band"][0] == 30.0


def test_reanchoring_is_judged_against_the_key_behind_e_ve():
    """P6's exception compares v_e*RA with the key the checkpoints would use instead: the
    runs' phase-matched keys under the V7 remedy (a pair whose runs fell back to them has no
    re-anchoring effect), the u = 0 key otherwise."""
    p = {"approach": np.array(["A1", "A1"]), "v_e_star_ra": np.array([33.0, 31.0]),
         "v_e_star": np.array([30.0, 30.0]), "v_e_star_k": np.array([33.0, 31.0]),
         "speed_at_phase": np.array([True, True])}
    assert A.reanchor_immaterial([p])["immaterial"]
    p["speed_at_phase"] = np.array([False, False])
    assert not A.reanchor_immaterial([p])["immaterial"]


def test_v7_remedy_reaches_the_cached_analysis_run_tables(ps, tmp_path, monkeypatch):
    """The analyse stage's cached run tables carry the V7 switch: built with it, and rebuilt
    when it changes."""
    from apexzero.interp import rollout

    f = tmp_path / "roll" / "G1.npz"
    f.parent.mkdir()
    f.write_bytes(b"x")
    monkeypatch.setattr(io, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(io, "rollout_path", lambda a, t: f)
    monkeypatch.setattr(ps, "ROOT", tmp_path)
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({}, {}))
    seen = []

    def fake_rt(*a, **k):
        seen.append(k.get("speed_at_phase"))
        return {"cell": np.array([len(seen)])}

    monkeypatch.setattr(A, "run_table", fake_rt)
    ps._run_table_cached("G1", 1, None, [], None, None, False, "keys")
    ps._run_table_cached("G1", 1, None, [], None, None, False, "keys")
    assert seen == [False]                                   # reused
    ps._run_table_cached("G1", 1, None, [], None, None, False, "keys", True)
    assert seen == [False, True]                             # the V7 switch rebuilds it
    # the other stages' run tables read the verdict from the manifest
    seen.clear()
    ps._run_table("G1", 1, {}, [], None, None, None, SimpleNamespace(validation=V7_FAILED))
    ps._run_table("G1", 1, {}, [], None, None, None, SimpleNamespace(validation=V7_PASSED))
    assert seen == [True, False]


def test_p7_key_phase_sd_stays_on_r_up_to_18_3(ps, monkeypatch):
    """V7's remedy also solves phase keys on P7's R 28.6 and 55.9 cells; P7's key null keeps
    to R <= 18.3 (protocol 2.4, P7)."""
    lm = Landmarks(1000.0, 1010.0, 1020.0, 1030.0, 1040.0)
    stims, cells = [], []
    for i, R in enumerate((14.6, 28.6)):
        geo = Geometry(R, 90.0, np.nan, R, 90.0, 40.0, 0.0, 1020.0, 1, False, 6.0)
        spec = CornerSpec(R=R, theta_deg=90.0, direction=1, width=12.0, l_r=20.0)
        stims.append(Stimulus(i, f"syn/R{R}/L", "synthetic", ("A",), 1, -1, lm, geo, spec))
        cells.append(Cell(i, i, i, ("A",), "A1", 86.0, 600.0, W=400, P=4.0, n_phases=16,
                          flags=("p7",)))
    bank = SimpleNamespace(stimuli=stims)
    kt = {"cell": np.array([0, 1]), "geometry": np.array(["g14.6", "g28.6"])}
    monkeypatch.setattr(ak, "key_for", lambda c, kind, k, **kw: SimpleNamespace(
        status="verified", measures={"d_b": 100.0 + k}))
    assert set(ps._key_phase_sd(kt, cells, None, bank)) == {"g14.6"}
