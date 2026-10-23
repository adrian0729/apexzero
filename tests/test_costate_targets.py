"""Idea 2's targets (costate.targets): the post-jump costate, the pedal fold, pi, the
switching function and the shadow prices, against finite differences of warm re-solves
(design v2.1, Sections 3.1, 3.4 and 7: K0, K1).

- A synthetic window (step 2's: a left corner on a 400 m strip with varying widths, T = 76,
  solved with the uncompiled solver as tests/test_fields_kkt.py:96-125 does) has an
  approach, braking, a slip contact, inside and outside corridor contacts and a
  full-throttle exit. There:
  - the two routes agree, the x_0 identity and the anchored a_0 companion hold, and the
    action rows hold while their planted bugs fail;
  - the fold leaves stationarity intact and maps two multiplier splits with the same
    stationarity to the same targets;
  - every component of lambda_t at plan states of each kind matches Richardson differences of
    pinned-prefix re-solves (K1's rule), and every planted bug fails;
  - pi_1 on the anchored problem matches its exact-x_1 shift, and its planted bugs fail; every
    estimate carries K1's floor at pi's step;
  - the wall, slip and grip prices match re-solves with the parameter moved, and the road
    field sums to the wall price.
- On step 1's stored keys: the design's example key reads as Section 2 quotes it; K0 holds on
  a W = 400, a W = 800 and an anchored key; lambda_n = Pi^R - Pi^L on the exit straight; a
  mirror pair has mirrored targets; and a few K_P targets match re-solves.

Everything here is CasADi and NumPy (no JAX float64 and no network)."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.costate import data_dirs
from apexzero.costate import targets as TG
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields import perturb as PT
from apexzero.interp import answer_key as ak
from apexzero.interp.stimuli import Landmarks
from apexzero.ocp.window import NA, NX, WindowSpec
from apexzero.tracks.track import Track
from apexzero.vehicle.params import F1_2025

KEYS, STIM = data_dirs()
HAVE_KEYS = (KEYS / "nominal").exists() and (STIM / "cells.json").exists()
needs_keys = pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys are not on this machine")
T_SYN = 76
SOLVE_OPTS = {"mu_strategy": "monotone", "mu_init": 1e-2, "bound_push": 1e-6}
EXAMPLE = "03b6076863e3adfe335e6757"      # design Section 2's example key


# ------------------------------------------------------------------------- the synthetic window


def corner_strip(n: int = 200, ds: float = 2.0, s_c0: float = 100.0, ramp: float = 6.0,
                 arc: float = 18.0, R: float = 15.0) -> tuple[Track, Landmarks]:
    """A left corner (trapezoid curvature to 1/R) on a 400 m strip, 9 m wide with widths that
    vary along s (tests/test_fields_kkt.py:59-75)."""
    s = np.arange(n) * ds
    L = 2 * ramp + arc
    u = s - s_c0
    kappa = np.clip(np.minimum(u / ramp, (L - u) / ramp), 0.0, 1.0) / R
    wl = 4.5 + 0.3 * np.sin(s / 23.0)
    wr = 4.5 + 0.3 * np.cos(s / 19.0)
    th = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
    lm = Landmarks(s_c0, s_c0 + ramp, s_c0 + L / 2, s_c0 + ramp + arc, s_c0 + L)
    tr = Track(name="synthetic-corner", s=s, x=x, y=y, theta=th, kappa=kappa, w_left=wl,
               w_right=wr, closed=False)
    return tr, lm


def kinematic_guess(tr: Track, lm: Landmarks, x0: np.ndarray, T: int, v_c: float = 22.0,
                    a_b: float = 25.0, a_a: float = 8.0, dt: float = 0.05):
    """Centreline states along a brake-to-v_c speed profile (tests/test_fields_kkt.py:78-96)."""
    p = F1_2025
    xs, acts = np.zeros((T + 1, NX)), np.zeros((T, NA))
    xs[0] = x0
    s, v = float(x0[0]), float(x0[3])
    for t in range(T):
        v_new = v + a_a * dt
        if s < lm.s_apex:
            v_new = min(v_new, max(np.sqrt(v_c ** 2 + 2 * a_b * max(lm.s_c0 - s, 0.0)),
                                   v - a_b * dt))
        pedal = 0.5 if v_new >= v else -0.6
        s, v = s + 0.5 * (v + v_new) * dt, v_new
        kap = np.interp(s, tr.s, tr.kappa)
        delta = np.arctan((p.lf + p.lr) * kap)
        xs[t + 1] = [s, 0.0, 0.0, v, 0.0, v * kap, delta, pedal]
        acts[t] = [np.clip(delta / p.delta_max, -1.0, 1.0), pedal]
    return xs, acts


@pytest.fixture(scope="module")
def synth():
    tr, lm = corner_strip()
    spec = WindowSpec(steps=T_SYN, radius=30.0)
    p = F1_2025
    ctx = ak.key_ctx(p)
    x0 = np.array([lm.s_c0 - 45.0, 0.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.3])
    fast = ak.AnchoredSolver(spec, compile=False, ipopt=SOLVE_OPTS)
    res = fast.solve(tr, x0, init=kinematic_guess(tr, lm, x0, T_SYN), ctx=ctx, p=p)
    assert res.success, res.status
    solver = ak.AnchoredSolver(spec, compile=False)
    rs = KK.resolve(KK.problem(solver, tr, res, ctx, p, x0), None)
    assert rs.ok, rs.status
    T = spec.steps
    res = dataclasses.replace(res, x=rs.w[: NX * (T + 1)].reshape(T + 1, NX),
                              a=rs.w[NX * (T + 1):].reshape(T, NA),
                              meta=dict(res.meta, lam=(rs.lam_g, rs.lam_x)))
    prob = K.repolish(KK.problem(solver, tr, res, ctx, p, x0, key_id="synthetic")).problem
    anch = K.repolish(KK.problem(solver, tr, res, ctx, p, x0, kind="anchored",
                                 key_id="synthetic-anchored")).problem
    blocks = TG.dynamics_blocks(prob)
    jac = TG.costate(prob, blocks=blocks)
    return SimpleNamespace(tr=tr, lm=lm, prob=prob, anch=anch, blocks=blocks, jac=jac,
                           cheap=TG.costate(prob, "cheap"))


def _kinds_of_states(cs: TG.Costate) -> dict[str, int]:
    """One plan state of each kind on the synthetic optimum (asserted present)."""
    corr = np.flatnonzero(cs.own_corridor.any(1))
    slip = np.flatnonzero(cs.own_slip.any(1))
    sat = np.flatnonzero(cs.pedal_saturated[:-1])
    brake = np.flatnonzero(cs.a[:, 1] < -0.05)
    assert len(corr) >= 2 and len(slip) and len(sat) and len(brake), "synthetic coverage"
    free = ~(cs.contact | cs.pedal_saturated)
    return {"decision": 0, "approach": 3, "braking": int(brake[len(brake) // 2]),
            "slip": int(slip[-1]), "inside": int(corr[1]), "outside": int(corr[-1]),
            "exit": int(np.flatnonzero(free[: sat[0]])[-1]), "saturated": int(sat[len(sat) // 2])}


@pytest.fixture(scope="module")
def synth_fd(synth):
    """K1-style differences of every component at one state of each kind."""
    cs = synth.jac
    states = _kinds_of_states(cs)
    out = {}
    for name, t in states.items():
        for i in range(NX):
            side = "below" if (i == TG.PEDAL and cs.pedal_saturated[t]) else "central"
            out[(name, t, i)] = TG.fd_costate_k1(synth.prob, t, i, side=side)
    return states, out


# ------------------------------------------------------------------------- K0 on the window


def test_pedal_lag_is_linear_with_the_designs_alpha(synth):
    assert TG.pedal_alpha(synth.prob.spec) == pytest.approx(0.905 ** 10, rel=1e-15)
    A, B = synth.blocks
    chk = TG.fold_check(synth.prob, synth.jac.fold, A, B)
    assert chk["pedal_row_deviation"] < 1e-14
    np.testing.assert_allclose(A[:, TG.PEDAL, TG.PEDAL], 0.905 ** 10, rtol=1e-14)


def test_fold_keeps_stationarity_and_signs(synth):
    prob, fd = synth.prob, synth.jac.fold
    T = prob.T
    A, B = synth.blocks
    chk = TG.fold_check(prob, fd, A, B)
    assert chk["stationarity"] < TG.FOLD_TOL and chk["stationarity_raw"] < TG.FOLD_TOL
    assert chk["command_sign_violation"] <= TG.SIGN_TOL and chk["n_saturated"] >= 5
    assert np.all(fd.lam_x[NX * np.arange(1, T + 1) + TG.PEDAL] == 0.0)
    # c_t = sum_{tau > t} alpha^(tau - t - 1) lam_x[8 tau + 7], the recursion unrolled
    box = prob.lam_x[NX * np.arange(T + 1) + TG.PEDAL]
    tau = np.arange(T + 1)
    for t in (0, 30, T - 1):
        m = tau > t
        assert fd.c[t] == pytest.approx(np.sum(fd.alpha ** (tau[m] - t - 1) * box[m]),
                                        rel=1e-12, abs=1e-18)


def test_fold_makes_the_pedal_split_unique(synth):
    """Another split of the pedal multipliers with the same stationarity (the LICQ failure
    along full throttle) folds to the same targets."""
    prob = synth.prob
    T = prob.T
    rng = np.random.default_rng(0)
    d = np.zeros(T + 1)
    d[1:] = 1e-3 * rng.random(T)                      # extra pedal-box multipliers
    a = TG.pedal_alpha(prob.spec)
    dc = np.zeros(T)
    dc[T - 1] = d[T]
    for t in range(T - 1, 0, -1):
        dc[t - 1] = d[t] + a * dc[t]
    lg, lx = prob.lam_g.copy(), prob.lam_x.copy()
    lx[NX * np.arange(T + 1) + TG.PEDAL] += d
    lg[NX * np.arange(T) + TG.PEDAL] -= dc
    lx[NX * (T + 1) + NA * np.arange(T) + 1] -= (1.0 - a) * dc
    lx[TG.PEDAL] -= a * dc[0]
    other = TG.with_multipliers(prob, lg, lx)
    assert np.max(np.abs(TG.stationarity(other))) < TG.FOLD_TOL
    assert np.max(np.abs(other.lam_g - prob.lam_g)) > 1e-4
    cs2 = TG.costate(other, blocks=synth.blocks)
    np.testing.assert_allclose(cs2.lam, synth.jac.lam, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cs2.switch, synth.jac.switch, rtol=0, atol=1e-12)
    # without the fold, lambda_pedal reads the split: the two splits disagree
    def unfolded(pb):
        box = pb.lam_x[NX * np.arange(T + 1) + TG.PEDAL]
        ident = TG.Fold(alpha=a, c=np.zeros(T), lam_g=pb.lam_g, lam_x=pb.lam_x, pedal_box=box)
        return TG.costate(pb, blocks=synth.blocks, fold=ident).lam[:, TG.PEDAL]

    assert np.max(np.abs(unfolded(other) - unfolded(prob))) > 1e-4


def test_routes_and_identities(synth):
    prob, jac, cheap = synth.prob, synth.jac, synth.cheap
    A, B = synth.blocks
    assert TG.route_gap(jac, cheap).max() < TG.ROUTE_TOL and cheap.end_gap < TG.ROUTE_TOL
    assert TG.x0_identity(prob, A[0]) < TG.X0_TOL
    np.testing.assert_allclose(jac.lam[0], prob.lam_x[:NX] + np.eye(NX)[TG.PEDAL]
                               * jac.fold.alpha * jac.fold.c[0], rtol=0, atol=1e-13)
    assert np.all(jac.lam[-1] == 0.0) and np.all(jac.pi[-1] == 0.0)
    assert TG.action_identity(jac, B, prob.spec) < 1e-10
    for plant in TG.ACTION_PLANTS:
        assert TG.action_identity(jac, B, prob.spec, plant) > 1e-4, plant
    # pi_t = 0.04 (a_t - a_{t-1}) with a_{-1} = 0
    np.testing.assert_allclose(jac.pi[0], 0.04 * prob.a[0], rtol=1e-15)
    np.testing.assert_allclose(jac.pi[5], 0.04 * (prob.a[5] - prob.a[4]), rtol=1e-15)
    # the jump is material only where x_t's own rows are active (a barrier multiplier below
    # ACTIVE_TOL, ~1e-8 next to a contact, moves lambda by as much)
    jumps = np.max(np.abs(jac.jump), axis=1) > 1e-5
    assert np.all(jac.contact[jumps]) and jumps.sum() >= 3
    rec = TG.k0_record(prob, jac, cheap, TG.prices(prob, jac), synth.blocks)
    assert rec["ok"] and rec["finite"], rec


def test_k0_fails_on_bugs_in_sigma_and_pi(synth):
    """Review finding: K0's 'ok' left out the action rows, the straight-road identity and
    the end gap, so a bug in how sigma (C2-sw's target) or pi is assembled passed K0. Each
    injected bug now fails the record."""
    prob, jac, cheap = synth.prob, synth.jac, synth.cheap
    pr = TG.prices(prob, jac)
    rec = TG.k0_record(prob, jac, cheap, pr, synth.blocks)
    assert rec["ok"] and rec["plants_fail"] and rec["switch_gap"] == 0.0
    assert all(v > TG.PLANT_MIN for v in rec["action_plants"].values())
    T = jac.W
    raw_cmd = np.asarray(prob.lam_x, float)[NX * (T + 1):].reshape(T, NA)
    inv = jac.gamma ** -np.arange(T, dtype=float)
    assert np.max(np.abs(raw_cmd - jac.fold.lam_x[NX * (T + 1):].reshape(T, NA))) > 1e-4
    for label, bad in (("raw command", dataclasses.replace(jac, switch=inv[:, None] * raw_cmd)),
                       ("sigma / gamma", dataclasses.replace(jac, switch=jac.switch / jac.gamma)),
                       ("pi scaled", dataclasses.replace(jac, pi=1.5 * jac.pi))):
        r = TG.k0_record(prob, bad, cheap, pr, synth.blocks)
        assert not r["ok"], label
    # the end gap and the straight-road identity are part of the verdict
    assert not TG.k0_record(prob, jac, dataclasses.replace(cheap, end_gap=1e-6), pr,
                            synth.blocks)["ok"]


def test_pedal_ladder_rule_on_planted_differences(monkeypatch):
    """``fd_pedal_below``'s ladder without re-solves: accept the largest converged c whose
    pair agrees; fall back past a failed re-solve; move up when the discrepancy grows as c
    shrinks; never below PEDAL_MIN_H; the noise estimate from the converged rungs."""
    prob = SimpleNamespace(spec=SimpleNamespace(gamma=0.995))

    def fake(table):
        def fd_costate(prob_, t, i, h, side, solver=None):
            v, disc, ok = table[h]
            return TG.FD(value=v, fd_h=v, fd_h2=v, discrepancy=disc, side=side, h=h, ok=ok,
                         iterations=1)
        return fd_costate

    good = {1e-3: (0.03, 1e-6, True), 5e-4: (0.03, 1e-6, True), 2.5e-4: (0.03, 1e-6, True)}
    monkeypatch.setattr(TG, "fd_costate", fake(good))
    fd = TG.fd_pedal_below(prob, 5)
    assert fd.h == 1e-3 and len(fd.ladder) == 1
    failed = {**good, 1e-3: (0.0, 0.0, False)}
    monkeypatch.setattr(TG, "fd_costate", fake(failed))
    fd = TG.fd_pedal_below(prob, 5)
    assert fd.h == 5e-4 and [r[0] for r in fd.ladder] == [1e-3, 5e-4]
    noisy = {1e-3: (0.031, 1e-3, True), 5e-4: (0.04, 1e-2, True), 2.5e-4: (0.5, 0.5, True),
             TG.PEDAL_UP: (0.0306, 1e-6, True)}
    monkeypatch.setattr(TG, "fd_costate", fake(noisy))
    fd = TG.fd_pedal_below(prob, 5)
    assert fd.h == TG.PEDAL_UP and [r[0] for r in fd.ladder] == [1e-3, 5e-4, TG.PEDAL_UP]
    assert fd.noise > 6 * TG.KINK_J_NOISE / (fd.h * 0.995 ** 5)     # the rungs' spread
    assert min(TG.PEDAL_LADDER) >= TG.PEDAL_MIN_H


def test_routes_and_identities_on_the_anchored_structure(synth):
    anch = synth.anch
    A, B = TG.dynamics_blocks(anch)
    jac, cheap = TG.costate(anch, blocks=(A, B)), TG.costate(anch, "cheap")
    assert jac.decision == 1 and jac.unique_from == 1
    assert TG.a0_identity(anch, B[0]) < TG.X0_TOL and TG.x0_identity(anch, A[0]) < TG.X0_TOL
    assert TG.route_gap(jac, cheap)[1:].max() < TG.ROUTE_TOL
    np.testing.assert_allclose(jac.pi[1], 0.04 * (anch.a[1] - anch.a[0]), rtol=1e-15)
    assert TG.k0_record(anch, jac, cheap, TG.prices(anch, jac), (A, B))["ok"]


# ------------------------------------------------------------------------- K1 on the window


def _errors(states, fds, lam):
    out = {i: [] for i in range(NX)}
    for (name, t, i), fd in fds.items():
        assert fd.ok, (name, t, i)
        if not fd.kink:
            out[i].append(TG.fd_error(lam[t, i], fd))
    return {i: np.asarray(v) for i, v in out.items()}


def test_costate_matches_richardson_resolves(synth, synth_fd):
    states, fds = synth_fd
    err = _errors(states, fds, synth.jac.lam)
    for i, e in err.items():
        assert len(e) >= len(states) - 2, (TG.STATE_NAMES[i], len(e))
        assert np.median(e) < TG.K1_MEDIAN and np.quantile(e, 0.95) < TG.K1_P95, \
            (TG.STATE_NAMES[i], e)
    # the pedal-saturated state's one-sided difference matches the folded target
    sat = states["saturated"]
    assert TG.fd_error(synth.jac.lam[sat, TG.PEDAL], fds[("saturated", sat, TG.PEDAL)]) < 1e-3


@pytest.mark.parametrize("plant", TG.PLANTS)
def test_planted_costate_bugs_fail(synth, synth_fd, plant):
    states, fds = synth_fd
    bad = TG.costate(synth.prob, plant=plant, blocks=synth.blocks)
    comps = [TG.S] if plant == "no_es" else ([TG.PEDAL] if plant == "unfolded"
                                             else list(range(4)))
    names = (("inside", "outside", "slip") if plant == "pre_jump"
             else (("saturated", "exit") if plant == "unfolded" else tuple(states)))
    if plant == "unfolded":                  # only lambda_pedal moves, by the fold's c_t
        assert np.array_equal(np.delete(bad.lam, TG.PEDAL, 1),
                              np.delete(synth.jac.lam, TG.PEDAL, 1))
    worst = max(TG.fd_error(bad.lam[t, i], fds[(nm, t, i)]) for nm, t in states.items()
                if nm in names for i in comps if not fds[(nm, t, i)].kink)
    # 'unfolded' moves lambda_pedal by about 1% at this window's saturated state (more on the
    # stored keys, up to 28%): it must fail K1's own p95 band there
    assert worst > (TG.K1_P95 if plant == "unfolded" else 0.05), (plant, worst)


def test_pi_matches_its_exact_shift(synth):
    anch = synth.anch
    pi1 = TG.costate(anch).pi[1]
    # K1's floor as a lambda_1 comparison's, at pi's step (gates.k1_error's zero rule)
    noise = 6.0 * TG.KINK_J_NOISE / (TG.PI_H * float(anch.spec.gamma))
    for j in range(NA):
        fd = TG.fd_pi(anch, j)
        assert fd.ok and abs(fd.value - pi1[j]) < 1e-6 * max(abs(pi1[j]), 1e-3), (j, fd, pi1)
        assert fd.h == TG.PI_H and fd.noise == pytest.approx(noise) and not fd.kink
        for plant in TG.PI_PLANTS:
            bad = TG.fd_pi(anch, j, plant=plant, a_prev=np.array([0.3, -0.2]))
            assert abs(bad.value - pi1[j]) > 1e-3, (j, plant)
            assert bad.noise == fd.noise, (j, plant)


def test_anchored_decision_costate_matches_resolves(synth):
    anch = synth.anch
    lam = TG.costate(anch).lam
    for i in range(4):
        fd = TG.fd_costate(anch, 1, i)
        assert fd.ok and TG.fd_error(lam[1, i], fd) < 1e-5, (i, fd)


# ------------------------------------------------------------------------- prices on the window


def _resolve_P(prob: K.KeyProblem, P: np.ndarray, lbg=None, ubg=None) -> float:
    rs = KK.fd_solver(prob)
    sol = rs(x0=prob.w, p=P, lbx=prob.lbx, ubx=prob.ubx,
             lbg=prob.lbg if lbg is None else lbg, ubg=prob.ubg if ubg is None else ubg,
             lam_g0=prob.lam_g, lam_x0=prob.lam_x)
    assert rs.stats()["return_status"] == "Solve_Succeeded"
    return -float(sol["f"])


def test_prices_match_parameter_resolves(synth):
    prob, cs = synth.prob, synth.jac
    pr = TG.prices(prob, cs)
    assert pr.direction == 1 and pr.inside is pr.left
    assert pr.left[0] > 1e-3 and pr.right[0] > 1e-3 and pr.slip[0] > 1e-3
    assert np.all(pr.left[-1:] == 0.0) and np.all(np.diff(pr.left * cs.gamma ** np.arange(
        prob.T + 1)) <= 1e-10)
    np.testing.assert_array_equal(pr.ctx_path[:, TG.MU], 0.0)   # slip angles ignore mu
    T = prob.T
    sl = K.parameter_slices(T, prob.spec.n_coef)
    h = 1e-4
    # grip: Pi^mu_0 = dJ*/d mu_scale
    i_mu = sl["ctx"].start + TG.MU
    Js = []
    for c in (h, -h):
        P = prob.P.copy()
        P[i_mu] += c
        Js.append(_resolve_P(prob, P))
    assert (Js[0] - Js[1]) / (2 * h) == pytest.approx(pr.mu[0], rel=1e-4)
    # slip: Pi^slip_0 = dJ*/d ln alpha_max, the 1e-4 rad margin held
    Js = []
    for c in (h, -h):
        P = prob.P.copy()
        am = prob.p.alpha_max * np.exp(c)
        P[sl["alpha_max"].start] = am
        lim = 1.0 - prob.spec.slip_safety / am
        lbg, ubg = prob.lbg.copy(), prob.ubg.copy()
        lbg[10 * T:], ubg[10 * T:] = -lim, lim
        Js.append(_resolve_P(prob, P, lbg, ubg))
    assert (Js[0] - Js[1]) / (2 * h) == pytest.approx(pr.slip[0], rel=1e-4)
    # walls: a uniform widening of each side, and the road field's sum
    n = prob.track.n_points
    blocks = KK.param_blocks(prob)
    for side, price in (("w_left", pr.left), ("w_right", pr.right)):
        Js = []
        for c in (h, -h):
            z = PT.zeros(n)._asdict()
            z[side] = np.full(n, c)
            Js.append(KK.resolve(prob, PT.Samples(**z)).J)
        assert (Js[0] - Js[1]) / (2 * h) == pytest.approx(price[0], rel=1e-4), side
        for t in (0, 20, 40):
            f = TG.road_field(prob, cs, t, blocks)
            assert np.sum(getattr(f, side)) == pytest.approx(price[t], rel=1e-10, abs=1e-14)


# ------------------------------------------------------------------------- stored keys


@pytest.fixture(scope="module")
def stored():
    refs = K.a2_refs(KEYS, STIM)
    bank, _ = K.design(STIM)
    ex = next(r for r in refs if r.key_id == EXAMPLE)
    return SimpleNamespace(refs=refs, bank=bank, ex=ex, kt=TG.key_targets(ex, KEYS, bank))


@needs_keys
def test_the_designs_example_key(stored):
    """Design Section 2's numbers for key 03b6... (syn/R13.1/T180/W12/typical/L)."""
    cs, pr = stored.kt.jac, stored.kt.prices
    # quoted to the last digit shown: within half a unit of it
    for got, quoted, unit in zip(cs.lam[1, :4], (-0.0057, -0.053, -0.72, 0.028),
                                 (1e-4, 1e-3, 1e-2, 1e-3), strict=True):
        assert abs(got - quoted) <= 0.5 * unit, (got, quoted)
    assert cs.lam[20, TG.VX] == pytest.approx(-0.077, abs=5e-4)
    assert cs.lam[60, TG.VX] == pytest.approx(0.134, abs=5e-4)
    assert cs.lam[60, TG.XI] == pytest.approx(5.7, abs=0.01)
    flips = np.flatnonzero(np.diff(np.sign(cs.lam[:120, TG.VX])))
    assert flips.tolist() == [6, 52]
    assert cs.lam[55, TG.N] == pytest.approx(0.102, abs=5e-4)
    assert cs.lam_pre[55, TG.N] == pytest.approx(-0.052, abs=5e-4)
    np.testing.assert_allclose(cs.nu[[53, 54], 0], [0.030, 0.117], atol=6e-4)
    assert cs.nu[100, 1] == pytest.approx(0.070, abs=6e-4)
    assert np.flatnonzero(np.abs(cs.nu[:, 3]) > TG.ACTIVE_TOL).tolist() == [43, 44, 45, 46, 47]
    assert abs(cs.lam[100, TG.S]) < 1e-10 and abs(cs.lam[200, TG.S]) < 1e-10
    assert cs.contact[55] and not cs.contact[50] and cs.pedal_saturated[200]
    assert pr.inside[55] < 1e-8 < pr.inside[54]                # last inside contact: row 54


@needs_keys
def test_k0_on_stored_keys(stored):
    refs = stored.refs
    picks = [stored.ex, next(r for r in refs if r.W == 800)]
    anch = sorted((r for r in K.key_index("anchored", KEYS) if r.verified),
                  key=lambda r: r.key_id)[:1]
    for ref in picks + anch:
        kt = stored.kt if ref is stored.ex else TG.key_targets(ref, KEYS, stored.bank)
        rows = np.arange(kt.jac.unique_from, min(200, kt.jac.W))
        rec = TG.k0_record(kt.problem, kt.jac, kt.cheap, kt.prices, kt.blocks, rows)
        assert rec["ok"], rec
        assert rec["route_gap"] < TG.ROUTE_TOL and rec["end_gap"] < TG.ROUTE_TOL
        assert rec["action_identity"] < 1e-9 and rec["pedal_row_deviation"] < 1e-14
        if ref.kind == "anchored":
            assert rec["a0_identity"] < TG.X0_TOL and kt.jac.decision == 1
        else:
            assert rec["straight_identity"] < 1e-12 and rec["n_straight"] > 100
            assert TG.straight_ahead(kt.problem)[-1]


@needs_keys
def test_mirror_partners_have_mirrored_targets(stored):
    ex = stored.ex
    mate = next(r for r in stored.refs if r.pair_id == ex.pair_id and r.phase == ex.phase
                and r.W == ex.W and r.kind == ex.kind and r.direction == -ex.direction)
    a, b = stored.kt, TG.key_targets(mate, KEYS, stored.bank)
    parity = np.array([-1.0 if c in TG.MIRROR_ODD else 1.0 for c in TG.STATE_NAMES])
    scale = np.max(np.abs(a.jac.lam), axis=0)
    assert np.all(np.max(np.abs(a.jac.lam - parity * b.jac.lam), axis=0) <= 0.05 * scale)
    assert np.max(np.abs(a.prices.inside - b.prices.inside)) <= 0.05 * a.prices.inside.max()
    assert b.prices.inside is b.prices.right


@needs_keys
def test_stored_key_costate_matches_resolves(stored):
    prob, lam = stored.kt.problem, stored.kt.jac.lam
    for t in (0, 20, 55):
        for i in range(4):
            fd = TG.fd_costate_k1(prob, t, i)
            assert fd.ok and TG.fd_error(lam[t, i], fd) < 1e-3, (t, i, fd)
