"""TO-OCP: the CasADi step and lookups are the simulator's, to machine precision; the
hand-structured derivatives are CasADi's AD; a solved window replays exactly."""

import os
import shutil
from pathlib import Path

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.ocp.replay import replay_window
from apexzero.ocp.window import (
    WindowResult,
    WindowSolver,
    WindowSpec,
    block_lookup,
    block_select,
    coefficient_windows,
    min_time_guess,
    prefix_multipliers,
    solve_continuation,
    step_function,
    track_scalars,
    window_lookup,
    window_return,
)
from apexzero.tracks.spline import eval_jax, packed_coeffs
from apexzero.tracks.track import Track, load_tumftm
from apexzero.vehicle.params import F1_2025 as P
from apexzero.vehicle.sim import control_step, rk2_step, track_arrays

TUMFTM = Path("data/tracks/tumftm")
jax.config.update("jax_enable_x64", True)


def synthetic(closed: bool, n: int = 700, ds: float = 2.0) -> Track:
    """A smooth strip (or loop) with corners of 40-200 m radius and varying widths."""
    s = np.arange(n) * ds
    ph = 2 * np.pi * s / (n * ds)
    kappa = 0.012 * np.sin(3 * ph) + 0.01 * np.sin(7 * ph + 0.4) + 0.003
    w_left = 6.0 + 0.8 * np.sin(5 * ph)
    w_right = 5.5 + 0.6 * np.cos(4 * ph)
    theta = np.cumsum(kappa) * ds
    return Track(name=f"synthetic-{'loop' if closed else 'strip'}", s=s, x=np.cos(theta).cumsum() * ds,
                 y=np.sin(theta).cumsum() * ds, theta=theta, kappa=kappa, w_left=w_left,
                 w_right=w_right, closed=closed)


def ctx_of(p=P):
    cla = p.cla_f + p.cla_r
    return np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla])


def jax_spline(tr: Track, values, s):
    coef = packed_coeffs(values, tr.closed, tr.n_points)
    return float(eval_jax(jnp.asarray(coef), jnp.asarray(s), tr.ds, jnp.asarray(tr.n_points),
                          jnp.asarray(tr.closed)))


@pytest.mark.parametrize("closed", [True, False])
def test_floor_indexed_lookups_equal_eval_jax(closed):
    tr = synthetic(closed)
    spec = WindowSpec(radius=20.0, ds=tr.ds)
    q = track_scalars(tr)
    s, j0, i0 = ca.SX.sym("s"), ca.SX.sym("j0"), ca.SX.sym("i0")
    W = [ca.SX.sym(f"W{k}", spec.n_coef) for k in range(3)]
    look = ca.Function("look", [s, *W, j0], window_lookup(s, W, j0, q))
    blk, ib = block_select(s, W[0], j0, q, spec.block)
    sel = ca.Function("sel", [s, W[0], j0], [blk, ib])
    b = ca.SX.sym("b", spec.block)
    blook = ca.Function("blook", [s, b, i0], [block_lookup(s, b, i0, q)])
    L = tr.n_points * tr.ds if closed else (tr.n_points - 1) * tr.ds
    # Centres include the start/finish wrap (closed) and both strip ends (open: eval_jax clamps).
    for c in (30.0, 700.0, L - 3.0):
        (Wk, Wl, Wr), jj = coefficient_windows(tr, np.array([c]), spec)
        for sq in np.linspace(c - spec.radius, c + spec.radius + 6.0, 31):
            got = [float(v) for v in look(sq, Wk[:, 0], Wl[:, 0], Wr[:, 0], jj[0])]
            ref = [jax_spline(tr, v, sq) for v in (tr.kappa, tr.w_left, tr.w_right)]
            assert np.max(np.abs(np.array(got) - ref)) < 1e-12
            # Block picked at the node, then substep lookups up to block - 4 intervals ahead.
            bv, iv = sel(sq, Wk[:, 0], jj[0])
            for ahead in np.linspace(0.0, (spec.block - 4) * tr.ds * 0.999, 5):
                if not closed and not (0.0 <= sq + ahead <= L):
                    continue        # beyond a strip end the block's index range shifts
                got = float(blook(sq + ahead, bv, iv))
                assert abs(got - jax_spline(tr, tr.kappa, sq + ahead)) < 1e-12


@pytest.mark.parametrize("closed", [True, False])
def test_step_matches_the_simulator(closed):
    tr = synthetic(closed)
    spec = WindowSpec(radius=30.0, ds=tr.ds)
    phi = step_function(spec)
    ta = track_arrays(tr, dtype=jnp.float64)
    for sigma in (40.0, 650.0, tr.n_points * tr.ds - 60.0):
        (Wk, _, _), jj = coefficient_windows(tr, np.array([sigma]), spec)
        for off in (-spec.radius + 1.0, 0.0, spec.radius - 1.0):
            x = np.array([sigma + off, 0.7, 0.02, 55.0, 0.4, 0.08, 0.03, 0.2])
            a = np.array([0.1, 0.3])
            xc = np.asarray(phi(x, a, Wk[:, 0], jj[0], track_scalars(tr), ctx_of())).ravel()
            xj = control_step(jnp.asarray(x), jnp.asarray(a), ta, P, spec.dt, spec.substeps)
            assert np.max(np.abs(xc - np.asarray(xj))) < 1e-10


@pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
def test_step_matches_the_simulator_on_hockenheim():
    from apexzero.tracks.reference import refine_reference

    tr = refine_reference(load_tumftm(TUMFTM / "Hockenheim.csv"))
    spec = WindowSpec()
    phi = step_function(spec)
    ta = track_arrays(tr, dtype=jnp.float64)
    for sigma in (30.0, 2500.0, tr.length - 3.0):   # includes the start/finish wrap
        (Wk, _, _), jj = coefficient_windows(tr, np.array([sigma]), spec)
        x = np.array([sigma, 0.7, 0.02, 55.0, 0.4, 0.08, 0.03, 0.2])
        a = np.array([0.1, 0.3])
        xc = np.asarray(phi(x, a, Wk[:, 0], jj[0], track_scalars(tr), ctx_of())).ravel()
        xj = jnp.asarray(x)
        for _ in range(spec.substeps):
            xj = rk2_step(xj, jnp.asarray(a), ta, P, spec.dt / spec.substeps)
        assert np.max(np.abs(xc - np.asarray(xj))) < 1e-10


def _point(solver: WindowSolver, tr: Track, seed: int = 0):
    """A perturbed straight-line guess, the solver's parameters and random multipliers."""
    rng = np.random.default_rng(seed)
    T = solver.T
    xs = np.zeros((T + 1, 8))
    xs[:, 0] = 100.0 + 50.0 * solver.spec.dt * np.arange(T + 1)
    xs[:, 3] = 50.0
    xs[:, 1:] += 0.05 * rng.standard_normal((T + 1, 7))
    acts = 0.3 * rng.standard_normal((T, 2))
    w = np.concatenate([xs.reshape(-1), acts.reshape(-1)])
    p = solver.parameters(tr, xs[:-1, 0], ctx_of(), P)
    return w, p, rng.standard_normal(solver.ng), 0.7


def test_structured_derivatives_equal_casadi_ad():
    tr = synthetic(True)
    solver = WindowSolver(WindowSpec(steps=4, radius=20.0), compile=False)
    w, p, lam_g, lam_f = _point(solver, tr)
    ws, ps = ca.MX.sym("w", solver.nw), ca.MX.sym("p", solver.n_param)
    f, g = solver.funcs["nlp"](ws, ps)
    jac = ca.Function("j", [ws, ps], [ca.jacobian(g, ws)])
    hess = ca.Function("h", [ws, ps], [ca.triu(ca.hessian(lam_f * f + ca.dot(ca.DM(lam_g), g), ws)[0])])
    grad = ca.Function("g", [ws, ps], [ca.gradient(f, ws)])
    j_ad, h_ad = ca.DM(jac(w, p)).full(), ca.DM(hess(w, p)).full()
    j_s = ca.DM(solver.funcs["jac_g"](w, p)[1]).full()
    h_s = ca.DM(solver.funcs["hess_lag"](w, p, lam_f, lam_g)).full()
    assert np.max(np.abs(j_s - j_ad)) < 1e-12 * max(1.0, np.max(np.abs(j_ad)))
    assert np.max(np.abs(h_s - h_ad)) < 1e-11 * max(1.0, np.max(np.abs(h_ad)))
    g_s = np.asarray(solver.funcs["grad_f"](w, p)[1]).ravel()
    assert np.max(np.abs(g_s - np.asarray(grad(w, p)).ravel())) < 1e-14
    # The objective is the env's discounted return.
    T = solver.T
    xs = w[: 8 * (T + 1)].reshape(T + 1, 8)
    acts = w[8 * (T + 1):].reshape(T, 2)
    assert abs(-float(solver.funcs["nlp"](w, p)[0]) - window_return(xs, acts, solver.spec)) < 1e-12


@pytest.mark.skipif(shutil.which(os.environ.get("CC", "cc")) is None, reason="no C compiler")
def test_compiled_functions_equal_the_vm(tmp_path):
    tr = synthetic(False)
    spec = WindowSpec(steps=4, radius=20.0)
    vm = WindowSolver(spec, compile=False)
    cc = WindowSolver(spec, compile=True, cache_dir=tmp_path)
    assert cc.compiled
    w, p, lam_g, lam_f = _point(vm, tr, seed=1)
    for name, args in (("nlp", (w, p)), ("jac_g", (w, p)), ("hess_lag", (w, p, lam_f, lam_g))):
        out_vm, out_cc = vm.funcs[name](*args), cc.funcs[name](*args)
        out_vm = out_vm if isinstance(out_vm, (list, tuple)) else [out_vm]
        out_cc = out_cc if isinstance(out_cc, (list, tuple)) else [out_cc]
        for a, b in zip(out_vm, out_cc, strict=True):
            assert np.max(np.abs(ca.DM(a).full() - ca.DM(b).full())) < 1e-12
    # A second solver with the same spec reuses the cached library.
    assert len(list(tmp_path.glob("*.so"))) == 1
    WindowSolver(spec, compile=True, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.so"))) == 1


def test_short_window_solve_replays_exactly():
    tr = synthetic(True)
    spec = WindowSpec(steps=24, radius=30.0)
    solver = WindowSolver(spec, compile=False)
    x0 = np.array([20.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0])
    # Guess: a simple lane-keeping controller at part throttle, simulated (so the guess is
    # dynamically consistent and on track).
    ta = track_arrays(tr, dtype=jnp.float64)
    step = jax.jit(lambda x, a: control_step(x, a, ta, P))
    xs, acts = [x0], []
    for _ in range(spec.steps):
        x = xs[-1]
        kap = np.interp(x[0], tr.s, tr.kappa)
        steer = np.clip(kap * P.wheelbase / P.delta_max - 0.2 * x[1] - 1.0 * x[2], -1.0, 1.0)
        acts.append(np.array([steer, 0.2]))
        xs.append(np.asarray(step(jnp.asarray(x), jnp.asarray(acts[-1]))))
    xs, acts = np.array(xs), np.array(acts)
    assert np.all(np.abs(xs[:, 1]) < 4.0)
    res = solver.solve(tr, x0, init=(xs, acts))
    assert res.success, res.status
    assert res.value > window_return(xs, acts, spec)
    ret, alive, xr = replay_window(tr, x0, res.a)
    assert alive
    assert abs(ret - res.value) <= 1e-6 * abs(res.value)
    assert abs(window_return(res.x, res.a, spec) - res.value) < 1e-9


class _ConstantSpeedOCP:
    """An s-domain 'optimum' at constant speed v, with n = s / 1000 so lookups are visible."""

    def __init__(self, tr: Track, v: float = 50.0):
        self.s = tr.s
        self.t = tr.s / v
        self.lap_time = tr.n_points * tr.ds / v
        self.z = np.zeros((tr.n_points, 7))
        self.z[:, 0] = tr.s / 1000.0
        self.z[:, 2] = v
        self.u = np.tile([0.1, 0.5], (tr.n_points, 1))


@pytest.mark.parametrize("closed", [True, False])
def test_min_time_guess_maps_the_lap_to_time(closed):
    tr = synthetic(closed)
    ocp = _ConstantSpeedOCP(tr)
    L = tr.n_points * tr.ds
    for s0 in ((100.0, L - 20.0, L + 30.0) if closed else (100.0, 1000.0)):
        x_start = np.array([s0, 0.3, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0])
        xs, acts = min_time_guess(ocp, tr, x_start, 30)
        assert xs.shape == (31, 8) and acts.shape == (30, 2)
        np.testing.assert_array_equal(xs[0], x_start)
        np.testing.assert_allclose(np.diff(xs[:, 0]), 2.5, atol=1e-9)   # 50 m/s x 50 ms, unwrapped
        np.testing.assert_allclose(xs[1:, 1], np.mod(xs[1:, 0], L) / 1000.0, atol=1e-9)
        np.testing.assert_allclose(xs[1:, 3], 50.0)
        np.testing.assert_allclose(acts, np.tile([0.1, 0.5], (30, 1)))


def test_continuation_keeps_the_prefix_and_extends_it():
    calls = []

    class Stub:
        def __init__(self, T):
            self.T = T

        def solve(self, track, x0, init, ctx=None, p=None, lam0=None):
            xs, acts = init
            calls.append((self.T, xs.copy(), acts.copy()))
            assert xs.shape == (self.T + 1, 8) and acts.shape == (self.T, 2)
            # The "optimum" shifts the guess, so each stage's output is recognisable.
            return WindowResult(x=xs + 1.0, a=acts + 1.0, value=0.0, success=True, status="",
                                iterations=0, solve_seconds=0.0, rounds=1)

    def extend(state, n):
        return np.repeat(np.asarray(state)[None], n + 1, axis=0), np.full((n, 2), -7.0)

    first = (np.zeros((51, 8)), np.zeros((50, 2)))
    res = solve_continuation(Stub, None, np.zeros(8), [10, 30, 50, 20], first, extend, overlap=4)
    assert [c[0] for c in calls] == [10, 30, 50, 20]
    # Stage 2: stage 1's optimum up to step 6 (= 10 - overlap), then the extension from x_6.
    _, xs2, a2 = calls[1]
    np.testing.assert_array_equal(xs2[:7], res[0].x[:7])
    np.testing.assert_array_equal(a2[:6], res[0].a[:6])
    np.testing.assert_array_equal(a2[6:], -7.0)
    np.testing.assert_array_equal(xs2[7:], np.repeat(res[0].x[6][None], 24, axis=0))
    # A shorter stage re-solves from the longer optimum, cut overlap steps before its own end.
    _, xs4, a4 = calls[3]
    np.testing.assert_array_equal(xs4[:17], res[2].x[:17])
    np.testing.assert_array_equal(a4[:16], res[2].a[:16])
    np.testing.assert_array_equal(a4[16:], -7.0)


def test_prefix_multipliers_follow_the_constraint_layout():
    Tp, N, T = 5, 3, 7
    # Label each multiplier by (block, step, component) so misplacements are visible.
    g = np.concatenate([100 + np.arange(8 * Tp), 200 + np.arange(2 * Tp), 300 + np.arange(2 * Tp)])
    x = np.concatenate([400 + np.arange(8 * (Tp + 1)), 500 + np.arange(2 * Tp)])
    g2, x2 = prefix_multipliers((g, x), Tp, N, T)
    assert g2.shape == (12 * T,) and x2.shape == (8 * (T + 1) + 2 * T,)
    np.testing.assert_array_equal(g2[: 8 * N], 100 + np.arange(8 * N))           # defects
    np.testing.assert_array_equal(g2[8 * T: 8 * T + 2 * N], 200 + np.arange(2 * N))   # corridor
    np.testing.assert_array_equal(g2[10 * T: 10 * T + 2 * N], 300 + np.arange(2 * N))  # slip
    np.testing.assert_array_equal(x2[: 8 * (N + 1)], 400 + np.arange(8 * (N + 1)))
    np.testing.assert_array_equal(x2[8 * (T + 1): 8 * (T + 1) + 2 * N], 500 + np.arange(2 * N))
    assert np.count_nonzero(g2) == 12 * N and np.count_nonzero(x2) == 8 * (N + 1) + 2 * N
