"""Ruling R30 (protocol 12, 2026-10-20): overspeed defined causally (9.1's primary path).

- Viability starts are judged by their last non-refine round and their replay; frozen keys
  keep protocol 6.2's rule.
- Counterfactual viability records: a qualifying tested state (non-viable, not overruled by a
  later verified one, at or before s_a0 by target, vx above the key's vx*(s) by more than
  tau = 0.5 km/h) is re-solved with (vx, vy, r) scaled to vx*(s), everything else unchanged,
  in s order, stopping at the first that verifies; a crash is overspeed when one verifies.
- Tier B and recovery solves get a distinct second start (Q); the point-mass bound is
  recorded at s_b and s_c0 - 50 only; records cached before R30 are re-judged.
- tau in the ellipse fallback and over-brake; the planted crash controls (OB, LS, LSK,
  LSKF) and their verdicts; the regression counts of T and c2; the controls stage's jobs.
- The two reviews of R30: only a run's own records count (not another rollout's), a
  counterfactual only when made from that record at the current key's speed; a missing solve
  leaves the crash unresolved and listed; the bound only before s_c0; ties in one step; T32's
  crash margins; planted verdicts need complete solves.

Freeze rule: no agent, no IPOPT solve, no rollout through the harness; key solves are test
doubles (``_run_start`` stubbed), and nothing is written under outputs/ (tmp dirs only).
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.interp import analysis as A
from apexzero.interp import answer_key as ak
from apexzero.interp import controls as C
from apexzero.interp import measures as M
from apexzero.interp.constants import CRASH_DV_TAU, DT, kph
from apexzero.interp.measures import Trajectory
from apexzero.ocp.window import WindowResult, WindowSpec

AKT = importlib.import_module("test_interp_answer_key")    # FakeBank, TCell, record, start
MT = importlib.import_module("test_interp_measures")       # traj, crash_run, marks, integrate

V_KEY, V_RUN = 40.0, 45.0      # m/s: the key, and the run 5 m/s faster
S_B = 500.0                    # the run's brake onset (target s_b)


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------- cap rule


def test_R30_a_viability_start_is_judged_by_its_last_non_refine_round(monkeypatch):
    """A first round capped at the +-60 m window bound, then a converged re-centred round:
    a viability start counts as verified (its replay verifies); a frozen key's start keeps
    6.2's rule and counts as capped."""
    rounds = [{"round": 1, "status": "Maximum_Iterations_Exceeded"},
              {"round": 2, "status": "Solve_Succeeded"},
              {"round": "refine", "status": "Maximum_Iterations_Exceeded"}]
    res = SimpleNamespace(meta={"rounds": rounds})
    assert ak._was_capped(res) and not ak._was_capped(res, last_round_only=True)
    res.meta["rounds"] = rounds[1:2] + rounds[:1]
    assert ak._was_capped(res, last_round_only=True)

    W = 200

    def solve(_mk, track, x0, stages, first, extend, **kw):
        xs, acts = first
        return [WindowResult(x=np.asarray(xs), a=np.asarray(acts), value=5.0, success=True,
                             status="Solve_Succeeded", iterations=900, solve_seconds=1.0,
                             rounds=2, meta={"rounds": list(rounds), "cpu_seconds": 2.0})]

    def verify(res, track, x0, ctx=None, **kw):
        res.meta["replay"] = {"verified": True, "survives": True, "rel": 0.0}
        return True

    monkeypatch.setattr(ak, "make_solver", lambda *a, **k: object())
    monkeypatch.setattr(ak, "solve_continuation", solve)
    monkeypatch.setattr(ak, "verify_key", verify)
    monkeypatch.setattr(ak, "_max_slip", lambda x, p: 0.0)
    guess = (np.zeros((W + 1, 8)), np.zeros((W, 2)))
    out = {}
    for kind in ("viability", "nominal", "phase"):
        prob = SimpleNamespace(kind=kind, spec=WindowSpec(), a0_fixed=None, track=None,
                               x0=np.zeros(8), ctx=None, p=ak.F1_2025,
                               metrics=lambda r: dict.fromkeys(ak.START_MEASURES, 1.0))
        sr, _ = ak._run_start(prob, "nominal:k", guess, None, (W,), 800)
        out[kind] = (sr.verified, sr.capped)
    assert out == {"viability": (True, False), "nominal": (False, True), "phase": (False, True)}


# ------------------------------------------------------------------------- viability solves


def _key_record(v: float = V_KEY, W: int = 400, key_id: str = "nomkey", kind="nominal"):
    x = np.zeros((W + 1, 8))
    x[:, 0] = 300.0 + v * DT * np.arange(W + 1)
    x[:, 3] = v
    return AKT.record(kind=kind, key_id=key_id, x=x, a=np.zeros((W, 2)), x0=x[0].copy(), W=W,
                      spec={}, J=10.0)


def _run(v: float = V_RUN, s_crash: float = 700.0, K: int = 200) -> Trajectory:
    """A run at constant v from s = 300 m with non-zero n, xi, vy, r, delta and pedal; it
    terminates at the first state past ``s_crash``."""
    x = np.zeros((K, 8))
    x[:, 0] = 300.0 + v * DT * np.arange(K)
    x[:, 1], x[:, 2], x[:, 3] = 1.5, 0.05, v
    x[:, 4], x[:, 5], x[:, 6], x[:, 7] = 0.3, 0.1, 0.02, -0.4
    kt = int(np.argmax(x[:, 0] >= s_crash))
    a = np.tile([0.1, -0.3], (K - 1, 1))
    return Trajectory(source="agent", t=np.arange(K) * DT, x=x, valid=np.arange(K) <= kt, a=a,
                      term_step=kt - 1)


@pytest.fixture
def solves(monkeypatch):
    """``_run_start`` replaced by a verdict rule on the problem (``rule(prob, name)``),
    each call recorded; Q's rollout guess is a constant-speed straight run."""
    calls, rule = [], {"fn": lambda prob, name: True}

    def run_start(prob, name, first_guess, extend, stages, max_iter):
        ver = bool(rule["fn"](prob, name))
        calls.append(SimpleNamespace(name=name, x0=prob.x0.copy(), a0=prob.a0_fixed,
                                     max_iter=max_iter, kind=prob.kind))
        xs, acts = first_guess
        res = WindowResult(x=np.asarray(xs[: prob.W + 1]), a=np.asarray(acts[: prob.W]),
                           value=10.0, success=ver,
                           status="Solve_Succeeded" if ver else "Maximum_Iterations_Exceeded",
                           iterations=3, solve_seconds=0.0, rounds=1,
                           meta={"lam": (np.zeros(3), np.zeros(4)), "rounds": [],
                                 "cpu_seconds": 0.01,
                                 "replay": {"verified": ver, "survives": True, "rel": 0.0}})
        meas = ak._start_measures(prob.metrics(res)) if ver else {}
        return ak.StartRecord(name=name, J=10.0 if ver else np.nan, success=ver, verified=ver,
                              capped=not ver, status=res.status, iterations=3,
                              cpu_seconds=0.01, measures=meas,
                              rounds=[{"steps": prob.W, "rounds": [
                                  {"round": 1, "status": res.status}]}]), res

    def q_guess(bank, cell, x0, steps, tables, c_v=0.95, phase=0):
        calls.append(SimpleNamespace(name="Q rollout", x0=np.asarray(x0).copy()))
        xs = np.tile(np.asarray(x0, float), (steps + 1, 1))
        xs[:, 0] = x0[0] + x0[3] * DT * np.arange(steps + 1)
        return xs, np.zeros((steps, 2))

    monkeypatch.setattr(ak, "_run_start", run_start)
    monkeypatch.setattr(ak, "q_rollout_guess", q_guess)
    monkeypatch.setattr(ak, "run_metrics", lambda run, marks, *a, **k: {"s_b": S_B})
    monkeypatch.setattr(ak, "key_metrics", lambda res, track, marks, *a, **k: AKT.fake_metrics(
        res.x[:, 0], res.x[:, 3], res.x[:, 1], res.x[:, 7], marks))
    return SimpleNamespace(calls=calls, rule=rule)


def _setup():
    bank = AKT.FakeBank()          # R40/90, s_c0 600 m, s_a0 620 m
    cell = AKT.TCell(cell_id=0, stim_id=0, pair_id=3, tiers=("A",), approach="A1", v0=V_KEY,
                     D=300.0, W=400)
    return bank, cell, _key_record(), _key_record(key_id="bokey", kind="backoff")


def _fast_fails(prob, name, s_ok=(0.0, 610.0)):
    """Verified before s ~ 610 m (s_b, s_c0 - 50, s_c0), or at the key's speed (a
    counterfactual): the run is unrecoverable only at s_a0, where it is 5 m/s too fast."""
    return s_ok[0] < prob.x0[0] < s_ok[1] or prob.x0[3] < V_KEY + 0.5


def test_R30_unrecoverable_at_s_a0_and_recoverable_slowed_is_overspeed(solves):
    """The s_a0 tested state has no verified continuation and is 5 m/s faster than the key;
    the same state slowed to vx*(s) verifies: overspeed (R28's code could never type a point
    of no return found at the s_a0 test overspeed: its own s lies past s_a0)."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert [r.anchor["target"] for r in recs] == ["s_b", "s_c0-50", "s_c0", "s_a0", "s_a0:cf"]
    assert [r.status for r in recs] == ["verified"] * 3 + ["unsolved", "verified"]
    assert all(ak.VIABILITY_RULE_FLAG in r.flags for r in recs)
    lm = bank.stimuli[0].landmarks
    kt = ak.key_trajectory(nom, bank.track(0))
    nr = ak.no_return(recs, run, kt, s_a0=lm.s_a0, key_id=nom.key_id)
    assert nr.target == "s_a0" and nr.s_target == lm.s_a0 and nr.s > lm.s_a0
    assert nr.cf_verified_at == "s_a0" and nr.dv == pytest.approx(V_RUN - V_KEY)
    assert M.no_return_resolved(nr)
    plain, _ = MT.crash_run()                           # a crash the fallback types line/steer
    assert M.crash_type(plain, MT.marks(), MT.traj(plain.s, plain.vx), nr) == "overspeed"
    # without the counterfactual (or scaled to another key) it is not overspeed; the missing
    # counterfactual leaves the crash unresolved and listed (review: not read as a failed one)
    nr_other = ak.no_return(recs, run, kt, s_a0=lm.s_a0, key_id="another")
    assert nr_other.cf_verified_at == "" and nr_other.missing == ("s_a0:cf",)
    assert not M.no_return_resolved(nr_other)
    assert M.crash_type(plain, MT.marks(), MT.traj(plain.s, plain.vx), nr_other) == "line_steer"


def test_R30_counterfactual_keeps_s_n_xi_delta_and_pedal(solves):
    """The slowed state: (vx, vy, r) of x_{L-1} scaled by vx*(s_L) / vx(s_L), s_L the tested
    state; s, n, xi, delta, pedal and a_0 unchanged; its target, scale and key recorded."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    tested, cf = recs[3], recs[4]
    scale = V_KEY / V_RUN
    assert cf.anchor["cf_scale"] == pytest.approx(scale) and cf.anchor["tested"] == "s_a0"
    assert cf.anchor["key"] == nom.key_id and cf.anchor["step"] == tested.anchor["step"]
    np.testing.assert_array_equal(cf.x0[[0, 1, 2, 6, 7]], tested.x0[[0, 1, 2, 6, 7]])
    np.testing.assert_allclose(cf.x0[3:6], tested.x0[3:6] * scale)
    cf_call = next(c for c in solves.calls if c.name.startswith("nominal")
                   and c.x0[3] < V_RUN - 1.0)
    np.testing.assert_array_equal(cf_call.a0, tested.anchor["a_prev"])
    np.testing.assert_array_equal(ak.counterfactual_state(tested.x0, 2.0)[[0, 1, 2, 6, 7]],
                                  tested.x0[[0, 1, 2, 6, 7]])
    assert cf.key_id != tested.key_id


def test_R30_a_later_verified_state_overrules_an_earlier_non_viable_one(solves):
    """s_b fails, s_c0 - 50 and s_c0 verify, s_a0 fails: the s_b verdict is overruled (a
    later state of the same run is recoverable), so s_b gets no counterfactual and the crash
    is decided at s_a0."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = lambda prob, name: _fast_fails(prob, name, (540.0, 610.0))
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert [r.anchor["target"] for r in recs] == ["s_b", "s_c0-50", "s_c0", "s_a0", "s_a0:cf"]
    nr = ak.no_return(recs, run, ak.key_trajectory(nom, bank.track(0)), s_a0=620.0)
    assert nr.overruled == ("s_b",) and nr.target == "s_a0" and nr.cf_verified_at == "s_a0"
    # every tested state verified through s_a0: resolved, not overspeed
    solves.rule["fn"] = lambda prob, name: True
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c2", snapshot="0-ph0")
    nr = ak.no_return(recs, run, ak.key_trajectory(nom, bank.track(0)), s_a0=620.0)
    assert nr.s is None and nr.tested == ("s_b", "s_c0-50", "s_c0", "s_a0")
    assert M.no_return_resolved(nr) and not nr.cf_verified_at


def test_R30_counterfactuals_run_in_s_order_and_stop_at_the_first_verified(solves):
    """s_c0 and s_a0 both fail (s_b, s_c0 - 50 verify): the counterfactual of s_c0 is solved
    first; when it verifies, s_a0's is not solved; when it fails, s_a0's is."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = lambda prob, name: prob.x0[0] < 590.0 or prob.x0[3] < V_KEY + 0.5
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert [r.anchor["target"] for r in recs][4:] == ["s_c0:cf"]
    solves.rule["fn"] = lambda prob, name: prob.x0[0] < 590.0 or (prob.x0[3] < V_KEY + 0.5
                                                                  and prob.x0[0] > 610.0)
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c3", snapshot="0-ph0")
    assert [(r.anchor["target"], r.status) for r in recs][4:] == [
        ("s_c0:cf", "unsolved"), ("s_a0:cf", "verified")]
    nr = ak.no_return(recs, run, ak.key_trajectory(nom, bank.track(0)), s_a0=620.0)
    assert nr.target == "s_c0" and nr.cf_verified_at == "s_a0"


def test_R30_no_counterfactual_within_tau_of_the_key(solves):
    """A run only 0.1 m/s faster than the key (tau = 0.5 km/h = 0.139 m/s) that fails at
    s_a0 gets no counterfactual: not overspeed by the primary path."""
    bank, cell, nom, bo = _setup()
    run = _run(v=V_KEY + 0.1, K=260)
    solves.rule["fn"] = lambda prob, name: prob.x0[0] < 610.0
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert [r.anchor["target"] for r in recs] == ["s_b", "s_c0-50", "s_c0", "s_a0"]
    assert 0.1 < CRASH_DV_TAU == pytest.approx(kph(0.5))


def test_R30_qualifying_states_by_target():
    """The s_a0 test counts as at s_a0 (its own s lies past it); an s_b target after s_a0
    does not qualify; overruled states do not; dv must exceed tau."""
    run = _run()
    kx = _key_record().x

    def rec(target, s_target, step, status="unsolved"):
        return AKT.record(kind="viability", status=status, key_id=f"v{target}",
                          starts=[AKT.start("nominal:k", 1.0, status == "verified")],
                          x0=run.x[step - 1].copy(), flags=[ak.VIABILITY_RULE_FLAG],
                          anchor={"target": target, "s_target": s_target, "step": step,
                                  "s_state": float(run.x[step, 0]),
                                  "a_prev": run.a[step - 1].tolist()})

    def step(s):
        return int(np.argmax(run.x[:, 0] >= s))

    recs = [rec("s_c0", 600.0, step(600.0), "verified"), rec("s_a0", 620.0, step(620.0)),
            rec("s_b", 630.0, step(630.0))]
    q = ak.qualifying_states(recs, run, kx, s_a0=620.0)
    assert [r.anchor["target"] for r, _ in q] == ["s_a0"]
    assert q[0][1] == pytest.approx(V_RUN - V_KEY)
    assert [r.anchor["target"] for r, _ in ak.qualifying_states(recs, run, kx)] == ["s_a0"]
    late_ok = recs + [rec("s_x", 640.0, step(640.0), "verified")]
    assert ak.qualifying_states(late_ok, run, kx, s_a0=620.0) == []        # all overruled


def test_R30_distinct_second_start_for_cells_without_a_backed_off_key(solves, tmp_path):
    """Tier B and the recovery cells have no backed-off key: the second start is Q's rollout
    from the state, never the nominal key twice; a cell with one keeps it."""
    bank, cell, nom, bo = _setup()
    assert [n for n, _ in ak.viability_starts(nom, bo, None)] == ["nominal:nomkey",
                                                                  "backoff:bokey"]
    for second in (None, nom):
        assert [n for n, _ in ak.viability_starts(nom, second, AKT.tables())] == [
            "nominal:nomkey", "Q"]
    assert [n for n, _ in ak.viability_starts(nom, None, None)] == ["nominal:nomkey"]
    solves.rule["fn"] = lambda prob, name: False
    recs = ak.viability_keys(bank, cell, _run(), nom, None, tables=AKT.tables(), agent="c1",
                             snapshot="0-ph0")
    names = [s.name for s in recs[0].starts]
    assert names == ["nominal:nomkey", "Q", "retry"]
    assert "Q start c_v 0.95" in recs[0].flags


def test_R30_point_mass_bound_only_at_s_b_and_s_c0_minus_50(solves):
    """The bound certifies the verified key itself infeasible at s_c0 and s_a0, so it is
    recorded (as evidence, never a verdict) only on the unverified s_b and s_c0 - 50 states."""
    bank, cell, nom, bo = _setup()
    solves.rule["fn"] = lambda prob, name: prob.x0[3] < V_KEY + 0.5
    recs = ak.viability_keys(bank, cell, _run(), nom, bo, tables=AKT.tables(), agent="c1",
                             snapshot="0-ph0")
    got = {r.anchor["target"]: bool(r.bound) for r in recs}
    assert got == {"s_b": True, "s_c0-50": True, "s_c0": False, "s_a0": False, "s_b:cf": False}
    assert all(r.status != "infeasible_bound" for r in recs)


def test_R30_point_mass_bound_not_at_an_s_b_past_s_c0(solves, monkeypatch):
    """Review: a late braker's s_b can lie at or past s_c0 (c1 cells 138 and 139), where the
    point-mass bound certifies the key itself infeasible; it is recorded only before s_c0."""
    bank, cell, nom, bo = _setup()
    monkeypatch.setattr(ak, "run_metrics", lambda run, marks, *a, **k: {"s_b": 605.0})
    solves.rule["fn"] = lambda prob, name: prob.x0[3] < V_KEY + 0.5
    recs = ak.viability_keys(bank, cell, _run(), nom, bo, tables=AKT.tables(), agent="c1",
                             snapshot="0-ph0")
    got = {r.anchor["target"]: bool(r.bound) for r in recs}
    assert got == {"s_c0-50": True, "s_c0": False, "s_b": False, "s_a0": False,
                   "s_c0-50:cf": False}


# ------------------------------------------------------------------------- review: evidence


def _kt(nom, bank):
    return ak.key_trajectory(nom, bank.track(0))


def _nr(recs, run, nom, bank, kt=None, **kw):
    return ak.no_return(recs, run, _kt(nom, bank) if kt is None else kt, s_a0=620.0,
                        s_c0=600.0, key_id=nom.key_id, **kw)


def test_R30_review_records_of_another_rollout_are_left_out(solves):
    """Review finding 1: records are grouped by (agent, snapshot, cell, phase) only, so a
    re-rolled run leaves the old rollout's records in the group. Only records solved from
    this run's own x_{L-1} and a_{L-1} count; a counterfactual counts only when made from
    this run's tested record (``tested_key``) at the current key's speed. (E1c: an old
    verified s_a0 no longer overrules; E1b: an old verified counterfactual no longer types
    the crash overspeed.)"""
    bank, cell, nom, bo = _setup()
    cur = _run()
    plain, _ = MT.crash_run()                           # the fallback types it line/steer
    kp = MT.traj(plain.s, plain.vx)
    # E1c: the old run, viable everywhere, 1 m/s slower; the current one overspeed at s_a0
    solves.rule["fn"] = lambda prob, name: True
    old = ak.viability_keys(bank, cell, _run(v=V_KEY - 1.0, K=260), nom, bo, agent="c1",
                            snapshot="0-ph0")
    solves.rule["fn"] = _fast_fails
    mine = ak.viability_keys(bank, cell, cur, nom, bo, agent="c1", snapshot="0-ph0")
    alone = _nr(mine, cur, nom, bank)
    for mixed in (old + mine, mine + old):
        nr = _nr(mixed, cur, nom, bank)
        assert nr == dataclasses.replace(alone, n_stale=4)
        assert nr.cf_verified_at == "s_a0" and nr.overruled == () and not nr.missing
        assert M.crash_type(plain, MT.marks(), kp, nr) == "overspeed"
    # E1b: the old run 0.5 m/s faster, its s_a0 counterfactual verified; the current run's
    # own counterfactual fails
    solves.rule["fn"] = _fast_fails
    old = ak.viability_keys(bank, cell, _run(v=V_RUN + 0.5), nom, bo, agent="c1",
                            snapshot="0-ph0")
    assert [(r.anchor["target"], r.status) for r in old][3:] == [
        ("s_a0", "unsolved"), ("s_a0:cf", "verified")]
    solves.rule["fn"] = lambda prob, name: 0.0 < prob.x0[0] < 610.0
    mine = ak.viability_keys(bank, cell, cur, nom, bo, agent="c1", snapshot="0-ph0")
    assert [(r.anchor["target"], r.status) for r in mine][3:] == [
        ("s_a0", "unsolved"), ("s_a0:cf", "unsolved")]
    nr = _nr(old + mine, cur, nom, bank)
    assert nr.cf_verified_at == "" and nr.missing == () and nr.n_stale == 4
    assert M.no_return_resolved(nr)                     # its own counterfactual failed
    assert M.crash_type(plain, MT.marks(), kp, nr) == "line_steer"
    # a counterfactual of another state carries that state's id
    assert old[4].anchor["tested_key"] != mine[3].key_id


def test_R30_review_a_missing_solve_is_not_a_failed_one(solves):
    """Review findings 2 and 4: a qualifying state with no counterfactual record (a worker
    error, an interrupted stage), or a tested state of the run with no record (its solve
    errored) after the last viable one, leaves the verdict open: the crash is unresolved
    and listed (``nr_missing``), not typed as if the solve had failed. A crash whose last
    alive state lies before s_a0 has no s_a0 test, so nothing is missing (R30's fallback)."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    tested = [r for r in recs if not r.anchor["target"].endswith(":cf")]
    # E2: the counterfactual never ran
    gap = _nr(tested, run, nom, bank)
    assert gap.target == "s_a0" and gap.missing == ("s_a0:cf",) and gap.cf_verified_at == ""
    assert not M.no_return_resolved(gap)
    solves.rule["fn"] = lambda prob, name: 0.0 < prob.x0[0] < 610.0
    failed = _nr(ak.viability_keys(bank, cell, run, nom, bo, agent="c9", snapshot="0-ph0"),
                 run, nom, bank)
    assert failed.missing == () and M.no_return_resolved(failed)
    cols_gap, cols_failed = A.no_return_columns(gap), A.no_return_columns(failed)
    assert cols_gap != cols_failed
    assert (cols_gap["nr_resolved"], cols_gap["nr_missing"]) == (False, "s_a0:cf")
    assert (cols_failed["nr_resolved"], cols_failed["nr_missing"]) == (True, "")
    # E3: s_b to s_c0 verified, the s_a0 solve errored, the crash after s_a0
    part = [r for r in tested if r.anchor["target"] != "s_a0"]
    nr = _nr(part, run, nom, bank)
    assert nr.s is None and nr.missing == ("s_a0",) and not M.no_return_resolved(nr)
    assert ak.no_return(part, run, _kt(nom, bank), s_a0=620.0).missing == ("s_a0",)
    # a tested state missing before the last viable one is overruled anyway
    no_b = [r for r in recs if r.anchor["target"] != "s_b"]
    assert _nr(no_b, run, nom, bank).missing == () and M.no_return_resolved(
        _nr(no_b, run, nom, bank))
    # a missing s_c0 between viable s_c0 - 50 and non-viable s_a0 (needs s_c0's position)
    no_c0 = [r for r in recs if r.anchor["target"] != "s_c0"]
    assert _nr(no_c0, run, nom, bank).missing == ("s_c0",)
    # the crash straddles s_a0 (last alive state 619.5 m, crash state 621.75 m): no s_a0 test
    strad = _run(s_crash=621.0)
    solves.rule["fn"] = _fast_fails
    st = ak.viability_keys(bank, cell, strad, nom, bo, agent="c1", snapshot="0-ph0")
    assert [r.anchor["target"] for r in st] == ["s_b", "s_c0-50", "s_c0"]
    nr = _nr(st, strad, nom, bank)
    assert nr.s is None and nr.missing == () and not M.no_return_resolved(nr)


def test_R30_review_a_key_re_solved_in_place_voids_its_counterfactuals(solves):
    """Review finding 3: 6.3's re-solve stores a better key under the same id, so the id
    filter cannot see it. A counterfactual counts only when its start is the tested state
    scaled to the current key's vx*(s_L); one scaled to the old key's speed is missing."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert _nr(recs, run, nom, bank).cf_verified_at == "s_a0"
    better = _key_record(v=V_KEY + 1.0)                      # same key_id 'nomkey'
    assert better.key_id == nom.key_id
    nr = _nr(recs, run, nom, bank, kt=_kt(better, bank))
    assert nr.cf_verified_at == "" and nr.missing == ("s_a0:cf",)
    assert not M.no_return_resolved(nr)


def test_R30_review_records_judged_before_R30_are_no_verdict(solves):
    """An unverified tested-state record without R30's flag (judged by the old cap rule,
    the controls stage not re-run) is no verdict under R30: listed as missing."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    old = [dataclasses.replace(r, flags=[]) if r.anchor["target"] == "s_a0" else r
           for r in recs]
    nr = _nr(old, run, nom, bank)
    assert nr.missing == ("s_a0 (judged before R30)",) and nr.tested == ("s_b", "s_c0-50",
                                                                       "s_c0")
    assert not M.no_return_resolved(nr)
    # verified records are the same under both rules
    ok = [dataclasses.replace(r, flags=[]) if r.status == "verified" else r for r in recs]
    assert _nr(ok, run, nom, bank) == _nr(recs, run, nom, bank)


def test_R30_review_two_targets_at_one_step(solves, monkeypatch):
    """Review finding 6: s_b and s_c0 - 50 can share a step (21 c1 runs). They are ordered by
    target (s_b first), whatever the order the cache lists them in, and the shared state's
    counterfactual is solved once, under the first target."""
    bank, cell, nom, bo = _setup()
    monkeypatch.setattr(ak, "run_metrics", lambda run, marks, *a, **k: {"s_b": 550.0})
    run = _run()
    solves.rule["fn"] = lambda prob, name: False
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    assert recs[0].anchor["step"] == recs[1].anchor["step"]
    assert [r.anchor["target"] for r in recs][4:] == ["s_b:cf", "s_c0:cf", "s_a0:cf"]
    for order in (recs, recs[::-1], recs[1:2] + recs[:1] + recs[2:]):
        nr = _nr(order, run, nom, bank)
        assert nr.target == "s_b" and nr.tested[:2] == ("s_b", "s_c0-50") and nr.missing == ()
    assert M.no_return_resolved(nr) and nr.cf_verified_at == ""


def test_R30_review_the_deciding_state_is_reported(solves):
    """Review finding 5 (spec tension, reporting): s_c0 - 50 is non-viable at the key's speed,
    s_c0 non-viable 5 m/s fast with a verified counterfactual: overspeed by R30's "some
    tested state". ``nr_dv`` describes the first non-viable state; ``nr_cf_dv`` the state
    that typed it."""
    bank, cell, nom, bo = _setup()
    run = _run()
    x = run.x.copy()
    x[x[:, 0] < 560.0, 3] = V_KEY
    run = dataclasses.replace(run, x=x)
    solves.rule["fn"] = lambda prob, name: (prob.x0[0] < 540.0
                                           or (prob.x0[3] < V_KEY + 0.5 and prob.x0[0] > 590.0))
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    nr = _nr(recs, run, nom, bank)
    assert nr.target == "s_c0-50" and nr.dv == pytest.approx(0.0, abs=1e-9)
    assert nr.cf_verified_at == "s_c0" and nr.cf_dv == pytest.approx(V_RUN - V_KEY)
    cols = A.no_return_columns(nr)
    assert cols["nr_cf"] == "s_c0" and cols["nr_cf_dv"] == pytest.approx(V_RUN - V_KEY)


def test_R30_review_the_run_table_passes_s_c0_and_s_a0(solves):
    """``CachedKeys.no_return`` gives ``no_return`` the stimulus's s_c0 and s_a0, so a missing
    tested state at s_c0 - 50, s_c0 or s_a0 is seen, and the key id of the run's key."""
    bank, cell, nom, bo = _setup()
    run = _run()
    solves.rule["fn"] = _fast_fails
    recs = ak.viability_keys(bank, cell, run, nom, bo, agent="c1", snapshot="0-ph0")
    kt = _kt(nom, bank)
    fake = SimpleNamespace(viability={("c1", 0, 0, 0, -1): [r for r in recs
                                                            if r.anchor["target"] != "s_c0"]},
                           nominal=lambda c, ph: SimpleNamespace(key_id=nom.key_id, traj=kt),
                           bank=bank)
    nr = A.CachedKeys.no_return(fake, cell, "c1", 0, 0, -1, run, kt)
    assert nr.missing == ("s_c0",) and not M.no_return_resolved(nr)
    fake.viability = {("c1", 0, 0, 0, -1): recs}
    assert A.CachedKeys.no_return(fake, cell, "c1", 0, 0, -1, run, kt).cf_verified_at == "s_a0"


def test_R30_records_cached_before_R30_are_re_judged(solves, tmp_path):
    """An unverified record without R30's flag: starts R30 judges alike are reused (no
    solve), a start whose converged last round was counted as capped is re-run, and a
    record without a backed-off key gets Q's start (the duplicated nominal start dropped);
    verified records and R30 records are kept as they are."""
    bank, cell, nom, bo = _setup()
    cache = ak.KeyCache(tmp_path)
    run = _run()
    solves.rule["fn"] = lambda prob, name: prob.x0[0] < 560.0        # s_b only
    recs = ak.viability_keys(bank, cell, run, nom, bo, cache, agent="c1", snapshot="0-ph0")
    by = {r.anchor["target"]: r for r in recs}
    assert by["s_c0"].status == "unsolved" and len(by["s_c0"].starts) == 3

    def unflag(r, **kw):
        cache.put(dataclasses.replace(r, flags=[f for f in r.flags
                                                if f != ak.VIABILITY_RULE_FLAG], **kw))

    for r in recs:
        unflag(r)
    solves.calls.clear()
    again = ak.viability_keys(bank, cell, run, nom, bo, cache, agent="c1", snapshot="0-ph0")
    assert [r.key_id for r in again] == [r.key_id for r in recs]
    assert [c.name for c in solves.calls] == []                     # every start reused
    assert all(ak.VIABILITY_RULE_FLAG in r.flags for r in again if r.status != "verified")
    assert all(ak.VIABILITY_RULE_FLAG not in r.flags for r in again if r.status == "verified")
    assert any("re-judged" in f for f in cache.get("viability", by["s_c0"].key_id).flags)
    # a capped start whose last round ended otherwise fails under both rules (reused); one
    # whose last round converged with no flag is R30's case (re-run)
    flagged = dataclasses.replace(by["s_c0"].starts[0], capped=True,
                                  status="Solve_Succeeded (window bound active)", rounds=[
                                      {"steps": 200, "rounds": [{"round": 1, "status":
                                                                 "Solve_Succeeded"}]}])
    assert ak._judged_alike(flagged)
    assert not ak._judged_alike(dataclasses.replace(flagged, status="Solve_Succeeded"))
    assert not ak._judged_alike(dataclasses.replace(flagged, capped=False, cpu_seconds=401.0))
    # R30's case: start 1 converged in its last round but was counted capped
    r = by["s_c0"]
    s1 = dataclasses.replace(r.starts[0], capped=True, status="Solve_Succeeded", rounds=[
        {"steps": 200, "rounds": [{"round": 1, "status": "Maximum_Iterations_Exceeded"},
                                  {"round": 2, "status": "Solve_Succeeded"}]}])
    unflag(r, starts=[s1, *r.starts[1:]])
    solves.calls.clear()
    solves.rule["fn"] = lambda prob, name: prob.x0[0] < 560.0 or name.startswith("nominal")
    again = ak.viability_keys(bank, cell, run, nom, bo, cache, agent="c1", snapshot="0-ph0")
    assert [c.name for c in solves.calls] == ["nominal:nomkey"]
    assert {r.anchor["target"]: r.status for r in again}["s_c0"] == "verified"
    # no backed-off key: the old duplicate nominal start is dropped, Q is run, the retry kept
    cache2 = ak.KeyCache(tmp_path / "b")
    solves.rule["fn"] = lambda prob, name: False
    old = ak.viability_keys(bank, cell, run, nom, bo, cache2, agent="c1", snapshot="0-ph0")
    for r in old[:4]:
        unflag_starts = [r.starts[0], dataclasses.replace(r.starts[0]), r.starts[2]]
        cache2.put(dataclasses.replace(r, starts=unflag_starts, flags=[]))
    solves.calls.clear()
    new = ak.viability_keys(bank, cell, run, nom, None, cache2, tables=AKT.tables(),
                            agent="c1", snapshot="0-ph0")
    tested_calls = [c.name for c in solves.calls if c.x0[3] > V_RUN - 0.5]
    assert tested_calls == ["Q rollout", "Q"] * 4
    assert [s.name for s in new[0].starts] == ["nominal:nomkey", "Q", "retry"]


# ------------------------------------------------------------------------- crash typing


def test_R30_fallback_and_over_brake_use_tau():
    """The ellipse fallback types overspeed only when dv > tau (T's crashes: float32 noise at
    the key's own speed); over-brake needs dv <= tau, so a lock 0.1 m/s faster than the key
    is over-brake."""
    K = 401
    key = MT.traj(MT.integrate(np.full(K, 60.0), 300.0), np.full(K, 60.0))
    mk = MT.marks()
    ur = np.full(K, 0.5)
    tr, kt = MT.crash_run()
    ur[kt - 10: kt - 6] = 0.99
    sat, _ = MT.crash_run(util_r=ur)
    assert M.crash_type(sat, mk, key, None) == "other"                 # dv 0: not overspeed
    noise = MT.traj(key.s, np.full(K, 60.0 - 1e-4))                    # the key 1e-4 slower
    assert M.crash_type(sat, mk, noise, None) == "other"
    faster = MT.traj(key.s, np.full(K, 60.0 - 0.5))
    assert M.crash_type(sat, mk, faster, None) == "overspeed"          # dv 0.5 > tau
    # a resolved crash (some tested state non-viable) never reaches the fallback
    nr = M.NoReturn(s=780.0, certified=False, dv=0.5, target="s_a0")
    assert M.crash_type(sat, mk, faster, nr) == "other"
    uf, bf = np.full(K, 0.5), np.zeros(K)
    k45 = MT.crash_run(v=45.0)[1]
    uf[k45 - 5: k45 - 1], bf[k45 - 5: k45 - 1] = 0.99, 1.2
    ob, _ = MT.crash_run(v=45.0, util_f=uf, beta=bf)
    key45 = MT.traj(ob.s, np.full(K, 45.0 - 0.1))
    assert M.crash_type(ob, mk, key45, None) == "overbrake"
    key_slow = MT.traj(ob.s, np.full(K, 45.0 - 1.0))
    assert M.crash_type(ob, mk, key_slow, None) != "overbrake"


# ------------------------------------------------------------------------- planted controls


def _planted_bank():
    """Tier A cells on synthetic corners: the A1 R14.6/90 left cell (site A1) next to an A1
    R13.1 left, an A1 R14.6 right and an A2 R28.6 left and its no-cap twin."""
    def stim(i, R, direction, no_cap=False):
        return SimpleNamespace(direction=direction, spec=SimpleNamespace(
            R=R, theta_deg=90.0, width=12.0, ramp_kind="typical", no_cap=no_cap))

    bank = SimpleNamespace(stimuli=[stim(0, 13.1, 1), stim(1, 14.6, 1), stim(2, 14.6, -1),
                                    stim(3, 28.6, 1, no_cap=True), stim(4, 28.6, 1),
                                    stim(5, 22.9, 1)])

    def cell(cid, sid, approach, tiers=("A",)):
        return SimpleNamespace(cell_id=cid, stim_id=sid, approach=approach, tiers=tiers)

    cells = [cell(10, 0, "A1"), cell(28, 1, "A1", ("A", "B", "E3")), cell(29, 2, "A1"),
             cell(40, 3, "A2"), cell(46, 4, "A2"), cell(47, 5, "A2"), cell(50, 1, "B", ("B",))]
    return bank, cells


def test_R30_planted_cells_are_chosen_by_geometry():
    bank, cells = _planted_bank()
    assert [c.cell_id for c in C.planted_cells(bank, cells, "OB")] == [28, 46]
    assert [c.cell_id for c in C.planted_cells(bank, cells, "LSK")] == [46]
    assert [c.cell_id for c in C.planted_cells(bank, cells, "LSKF")] == [46]
    # the closest R stands in when the exact one is absent
    rest = [c for c in cells if c.cell_id not in (28, 46)]
    assert [c.cell_id for c in C.planted_cells(bank, rest, "LS")] == [10, 47]
    assert set(C.PLANTED) == {"OB", "LS", "LSK", "LSKF"} <= set(C.CONTROLS)
    assert {n: C.CONTROLS[n].required for n in C.PLANTED} == {
        "OB": {"crash_type": ("overbrake",)}, "LS": {"crash_type": ("line_steer",)},
        "LSK": {"crash_type": ("line_steer",)}, "LSKF": {"crash_type": ("not overspeed",)}}
    assert set(C.CONTROL_NAMES) == set(C.CONTROLS)


def test_R30_planted_tapes():
    """LSK: the key's float32 tape with the steering times 0.3 from the first state at or past
    s_c0 + 2 m; LSKF adds the key's braking before s_c0 times 0.9."""
    W = 100
    x = np.zeros((W + 1, 8))
    x[:, 0] = 900.0 + 2.0 * np.arange(W + 1)                   # s_c0 = 1000 at state 50
    a = np.column_stack([np.full(W, 0.5), np.where(np.arange(W) < 60, -0.8, 0.4)])
    lsk = C.planted_tape(x, a, 1000.0, "LSK")
    assert lsk.dtype == np.float32
    np.testing.assert_allclose(lsk[:51, 0], 0.5)
    np.testing.assert_allclose(lsk[51:, 0], 0.15, rtol=1e-6)    # s 1002 m from state 51 on
    np.testing.assert_array_equal(lsk[:, 1], a[:, 1].astype(np.float32))
    lskf = C.planted_tape(x, a, 1000.0, "LSKF")
    np.testing.assert_allclose(lskf[:50, 1], -0.72, rtol=1e-6)  # braking before s_c0 x 0.9
    np.testing.assert_allclose(lskf[50:60, 1], -0.8, rtol=1e-6)
    np.testing.assert_allclose(lskf[60:, 1], 0.4, rtol=1e-6)
    np.testing.assert_array_equal(lskf[:, 0], lsk[:, 0])
    with pytest.raises(ValueError):
        C.planted_tape(x, a, 1000.0, "OB")


def test_R30_planted_q_policies_change_only_their_fault():
    """OB holds the pedal at -1 from s_c0 - 5 m; LS scales the steering by 0.25 from s_c0 +
    2 m; elsewhere both are Q."""
    from apexzero.env import racing
    from apexzero.tracks.generator import strip_arrays

    jax.config.update("jax_enable_x64", False)
    CT = importlib.import_module("test_interp_controls")
    car = C.car_tables(CT._fallback_tables())
    prof = CT.trapezoid(30.0, 90.0, 1, CT.ramp(30.0, 90.0))
    prm = C.q_params_for_kappa(prof.kappa[None], [CT.S_C], car, 0.9)
    tr = strip_arrays(*(jnp.asarray(getattr(prof, f), jnp.float32)
                        for f in ("kappa", "w_left", "w_right")))
    q, ob, ls = C.make_q_policy("Q"), C.make_planted_q_policy("OB"), C.make_planted_q_policy("LS")
    for s, before_ob, before_ls in ((990.0, True, True), (996.0, False, True),
                                    (1010.0, False, False)):
        st = CT.init_state([s, 1.0, 0.05, 50.0, 0.2, 0.1, 0.05, 0.3])
        o = racing.observe(st, tr, racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu))
        mq = np.asarray(q(prm, o, st, tr, 0)[0])
        mo = np.asarray(ob(prm, o, st, tr, 0)[0])
        ml = np.asarray(ls(prm, o, st, tr, 0)[0])
        assert mo[0] == mq[0] and (mo[1] == mq[1] if before_ob else mo[1] == -1.0)
        assert ml[1] == mq[1]
        assert ml[0] == pytest.approx(mq[0] if before_ls else 0.25 * mq[0], rel=1e-6)
    with pytest.raises(ValueError):
        C.make_planted_q_policy("LSK")


def _typed(ct, crashed=None, phase=None, mode="det", tested=None, missing=None):
    n = len(ct)
    return A._to_table([dict(cell=i, mode=mode, phase=0 if phase is None else phase[i],
                             crashed=True if crashed is None else crashed[i], crash_type=ct[i],
                             nr_resolved=i % 2 == 0, nr_target="s_a0", nr_dv=1.0, nr_cf="",
                             nr_tested="s_c0,s_a0" if tested is None else tested[i],
                             nr_missing="" if missing is None else missing[i])
                        for i in range(n)],
                       ("cell", "mode", "phase", "crashed", "crash_type", "nr_resolved",
                        "nr_target", "nr_dv", "nr_cf", "nr_tested", "nr_missing"))


def test_R30_planted_typing_verdicts():
    """Every u = 0 run of a planted control must crash and get its type; LSKF anything but
    overspeed."""
    ob = A.planted_typing(_typed(["overbrake", "overbrake"]), "overbrake")
    assert ob["passed"] and ob["got"] == "overbrake" and len(ob["runs"]) == 2
    miss = A.planted_typing(_typed(["overbrake", "other"]), "overbrake")
    assert not miss["passed"] and miss["got"] == "1 of 2 not overbrake"
    ls = A.planted_typing(_typed(["line_steer", "overspeed"]), "line_steer")
    assert not ls["passed"]
    assert A.planted_typing(_typed(["line_steer", "other"], phase=[0, 1]), "line_steer")["passed"]
    assert A.planted_typing(_typed(["other", "line_steer"]), "not overspeed")["passed"]
    assert not A.planted_typing(_typed(["other", "overspeed"]), "not overspeed")["passed"]
    alive = A.planted_typing(_typed(["", "other"], crashed=[False, True]), "not overspeed")
    assert not alive["passed"] and alive["got"] == "1 of 2 not not overspeed (1 without a crash)"
    none = A.planted_typing(_typed(["other"], phase=[1]), "not overspeed")
    assert not none["passed"] and none["got"] == "no u = 0 run"
    # review: the type must come with the run's viability solves, present and complete (LSK
    # without its solves is typed line/steer by the ellipse fallback, which tests nothing)
    nosolve = A.planted_typing(_typed(["line_steer", "line_steer"], tested=["", "s_a0"]),
                               "line_steer")
    assert not nosolve["passed"] and nosolve["got"] == "1 of 2 not line_steer (1 without " \
        "viability solves)"
    gap = A.planted_typing(_typed(["other", "other"], missing=["", "s_a0:cf"]), "not overspeed")
    assert not gap["passed"] and gap["got"] == "1 of 2 not not overspeed (1 with solves missing)"
    assert gap["runs"][1]["missing"] == "s_a0:cf"
    with pytest.raises(ValueError):
        A.planted_typing(_typed(["other"]), "overspeed")


def test_R30_regression_counts_of_T_and_c2():
    """T's fallback-typed crashes by type (none overspeed expected); c2's crashes on its
    lock-limited cells (raw radius <= r_lock) all overspeed, and at least one, as the
    adjudication expects. Both are reported, never blocking (``CONTROLS``)."""
    assert "crash_types" in C.CONTROLS["T"].reported
    assert "crash_types" not in C.CONTROLS["T"].required
    assert "lock_crashes" in C.CONTROLS["c2"].reported
    assert "lock_crashes" not in C.CONTROLS["c2"].required
    cc = A.crash_type_counts(_typed(["line_steer", "other", "overspeed"]))
    assert cc["n"] == 3 and cc["n_overspeed"] == 1 and cc["n_resolved"] == 2
    bank = SimpleNamespace(kappa=np.array([[0.0, 1 / 6.0], [0.0, 1 / 20.0]]))
    cells = [SimpleNamespace(cell_id=0, stim_id=0), SimpleNamespace(cell_id=1, stim_id=1)]
    rt = A._to_table([dict(cell=c, mode="det", phase=k, crashed=True, crash_type=t)
                      for c, k, t in ((0, 0, "overspeed"), (0, 1, "overspeed"),
                                      (1, 0, "other"))],
                     ("cell", "mode", "phase", "crashed", "crash_type"))
    lk = C.c2_lock_typing(rt, bank, cells, 8.16)
    assert lk["passed"] and lk["n"] == 2 and lk["cells"] == [0]     # cell 1 is not lock-limited
    rt["crash_type"][1] = "other"
    assert not C.c2_lock_typing(rt, bank, cells, 8.16)["passed"]
    assert not C.c2_lock_typing(rt, bank, cells, 5.0)["passed"]      # no lock-limited crash


def test_R30_pair_table_reports_p2b_on_the_solve_typed_subset():
    """P2(b) is also reported over the crashes the viability path typed."""
    pairs = A._to_table([dict(agent="G1", typed_crashes=4, typed_solve=2,
                              overspeed_frac=0.75, overspeed_frac_solve=0.5)],
                        ("agent", "typed_crashes", "typed_solve", "overspeed_frac",
                         "overspeed_frac_solve"))
    assert {"typed_solve", "overspeed_frac_solve"} <= set(A.PAIR_COLUMNS)
    assert {"nr_target", "nr_s_target", "nr_dv", "nr_cf", "nr_overruled"} <= set(A.RUN_COLUMNS)
    assert A._n(pairs) == 1


# ------------------------------------------------------------------------- script wiring


@pytest.fixture(scope="module")
def ps():
    path = str(Path(__file__).resolve().parents[1] / "scripts")
    sys.path.insert(0, path)
    try:
        yield importlib.import_module("psychophysics")
    finally:
        sys.path.remove(path)


def test_R30_controls_stage_solves_c1_and_the_planted_controls(ps, monkeypatch, tmp_path):
    """Step 5 of the controls stage: one viability job per crashed u = 0 run of c1 and of
    each planted control, on cells with a verified u = 0 key (snapshot '0-ph0', Q's c_v for
    the Q start), counts of tested and counterfactual records per control."""
    from apexzero.interp import io, rollout

    have = {"c1", "OB", "LSK"}
    for name in have:
        (tmp_path / f"control-{name}.npz").touch()
    monkeypatch.setattr(io, "rollout_path", lambda agent, ts: tmp_path / f"{agent}.npz")
    monkeypatch.setattr(rollout, "load_rollouts", lambda p: ({"det": {"agent": Path(p).stem}},
                                                             {}))
    monkeypatch.setattr(ps, "_run_table", lambda agent, ts, groups, *a: {"agent": agent})
    crashed = {"control-c1": [(1, 0), (2, 0), (3, 1)], "control-OB": [(28, 0), (46, 0)],
               "control-LSK": [(46, 0)]}
    monkeypatch.setattr(A, "viability_targets", lambda rt, kt, cap: {
        "cell": np.array([c for c, _ in crashed[rt["agent"]]]),
        "phase": np.array([k for _, k in crashed[rt["agent"]]])})
    monkeypatch.setattr(ps, "_u0_rows", lambda det, cells: {1: 0, 2: 1, 28: 0, 46: 1})
    jobs = []

    def pool(js, workers, log, fn=None, init=None):
        jobs.extend(js)
        return [dict(label=j["label"], cpu=1.0, keys=[
            dict(target="s_c0", status="unsolved"), dict(target="s_c0:cf", status="verified")])
            for j in js]

    monkeypatch.setattr(ps, "_pool", pool)
    cells = [SimpleNamespace(cell_id=i) for i in (1, 2, 3, 28, 46)]
    kt = {"cell": np.array([1, 2, 3, 28, 46]),
          "key_status": np.array(["verified", "missing", "verified", "verified", "verified"])}

    class Log:
        total = 0

        def set_total(self, n):
            self.total = n

        def write(self, msg):
            pass

    out = ps._control_viability(None, cells, kt, "KEYS", None, 0.9, 2, Log())
    assert sorted((j["control"], j["cell"].cell_id) for j in jobs) == [
        ("LSK", 46), ("OB", 28), ("OB", 46), ("c1", 1)]
    assert {j["snapshot"] for j in jobs} == {"0-ph0"} and {j["q_cv"] for j in jobs} == {0.9}
    assert out["per_control"]["c1"]["n_no_key"] == 1                   # cell 2: no key
    assert out["per_control"]["OB"]["tested_unsolved"] == 2
    assert out["per_control"]["OB"]["cf_verified"] == 2
    assert out["per_control"]["LS"]["jobs"] == 0 and out["jobs"] == 4


def test_R30_planted_controls_run_through_the_harness(ps, monkeypatch):
    """The controls stage's planted branch end to end on a two-cell design: OB (Q with its
    fault) and LSK (the u = 0 key's float32 tape with its fault) through the float32 harness,
    u = 0 only, on the cell the geometry rule picks; LSK refuses a cell without a verified
    key."""
    jax.config.update("jax_enable_x64", False)
    DS_ = importlib.import_module("test_interp_dryrun_script")
    CT = importlib.import_module("test_interp_controls")
    bank, cells = DS_._design()
    a1 = dataclasses.replace(cells[0], tiers=("A",), approach="A2", D=250.0)   # an A2 site
    cells = [cells[0], cells[1], dataclasses.replace(a1, cell_id=3, pair_id=4), cells[2]]
    car = C.car_tables(CT._fallback_tables())
    W = 400
    x = np.zeros((W + 1, 8))
    x[:, 0] = 1030.0 - 250.0 + 60.0 * DT * np.arange(W + 1)
    x[:, 3] = 60.0
    acts = np.tile([0.0, 0.3], (W, 1))
    key = SimpleNamespace(status="verified", x=x, a=acts, key_id="k")
    monkeypatch.setattr(ak, "key_for", lambda c, kind, phase, **k: key)
    ob = ps._run_one_control("OB", C.CONTROLS["OB"], bank, cells, car, 0.9, None, "CACHE")
    arr, meta = ob
    assert sorted({int(c) for c in arr["cell"]}) == [0, 3] and set(arr["phase"]) == {0}
    assert meta["plant"]["name"] == "OB" and meta["q_law"] == C.Q_LAW
    i = int(np.flatnonzero(arr["cell"] == 0)[0])
    s_c0 = bank.stimuli[0].landmarks.s_c0
    pedal_cmd = arr["a"][i, :, 1]                              # step k's command, at state k
    s = arr["x"][i, : len(pedal_cmd), 0]
    n = int(arr["term_step"][i]) if arr["term_step"][i] >= 0 else len(pedal_cmd)
    on = (s >= s_c0 - 5.0) & (np.arange(len(pedal_cmd)) < n)
    assert on.any() and np.all(pedal_cmd[on] == -1.0)
    assert np.any(pedal_cmd[(s < s_c0 - 5.0)] != -1.0)
    lsk, meta = ps._run_one_control("LSK", C.CONTROLS["LSK"], bank, cells, car, 0.9, None,
                                    "CACHE")
    assert sorted({int(c) for c in lsk["cell"]}) == [3] and meta["n_phases"] == 1
    monkeypatch.setattr(ak, "key_for", lambda c, kind, phase, **k: None)
    assert "no verified u = 0 key" in ps._run_one_control("LSK", C.CONTROLS["LSK"], bank,
                                                          cells, car, 0.9, None, "CACHE")
