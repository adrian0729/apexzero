"""P1 physics gates: backend parity, terminal speed, coast-down, kinematics, dt convergence."""

import casadi as ca
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from apexzero.tracks.track import from_centreline
from apexzero.vehicle import dynamics
from apexzero.vehicle.ops import CASADI, JAX
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import control_step, track_arrays

jax.config.update("jax_enable_x64", True)
P = F1_2025


def straight(length=20000.0):
    n = int(length / 10) + 1
    xy = np.stack([np.linspace(0, length, n), np.zeros(n)], axis=1)
    w = np.full(n, 6.0)
    return from_centreline("straight", xy, w, w, ds=2.0, smoothing=0.0, closed=False)


def circle(radius):
    t = np.linspace(0, 2 * np.pi, 600, endpoint=False)
    xy = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    w = np.full(600, 6.0)
    return from_centreline("circle", xy, w, w, ds=1.0, smoothing=0.0)


def random_states(n, rng):
    return np.stack([
        rng.uniform(0, 500, n),           # s
        rng.uniform(-5, 5, n),            # n
        rng.uniform(-0.4, 0.4, n),        # xi
        rng.uniform(0.5, 95, n),          # vx (includes the kinematic blend region)
        rng.uniform(-4, 4, n),            # vy
        rng.uniform(-1.5, 1.5, n),        # r
        rng.uniform(-0.45, 0.45, n),      # delta
        rng.uniform(-1, 1, n),            # pedal
    ], axis=1)


def test_jax_casadi_parity():
    rng = np.random.default_rng(0)
    n = 10_000
    xs = random_states(n, rng)
    us = rng.uniform(-1, 1, (n, 2))
    ks = rng.uniform(-0.08, 0.08, n)

    xj = jnp.asarray(xs)
    fj = np.asarray(dynamics.f(xj, jnp.asarray(us), jnp.asarray(ks), P, JAX))

    x = ca.MX.sym("x", 8)
    u = ca.MX.sym("u", 2)
    k = ca.MX.sym("k")
    fun = ca.Function("f", [x, u, k], [dynamics.f(x, u, k, P, CASADI)])
    fc = np.asarray(fun.map(n)(xs.T, us.T, ks[None, :])).T

    scale = np.maximum(np.abs(fc), 1.0)
    assert np.max(np.abs(fj - fc) / scale) < 1e-5


def _run(x0, u, tr, seconds, dt=0.05, substeps=10, method="rk2"):
    trj = track_arrays(tr, dtype=jnp.float64)
    step = jax.jit(lambda x: control_step(x, jnp.asarray(u), trj, P, dt, substeps, method))
    xs = [np.asarray(x0)]
    x = jnp.asarray(x0)
    for _ in range(round(seconds / dt)):
        x = step(x)
        xs.append(np.asarray(x))
    return np.array(xs)


def terminal_speed(p=P):
    """Solve P_max = v * (drag + rolling resistance on weight + downforce)."""
    cla = p.cla_f + p.cla_r

    def res(v):
        return v * (0.5 * p.rho * p.cda * v**2 + p.f_roll * (p.m * p.g + 0.5 * p.rho * cla * v**2)) - p.p_max

    lo, hi = 10.0, 200.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if res(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def test_terminal_speed():
    v_term = terminal_speed()
    assert 85.0 < v_term < 92.0  # ~315 km/h with DRS closed
    x0 = np.array([0, 0, 0, 60.0, 0, 0, 0, 1.0])
    xs = _run(x0, [0.0, 1.0], straight(), seconds=90.0)
    assert xs[-1, 3] == pytest.approx(v_term, rel=5e-3)
    assert abs(xs[-1, 1]) < 1e-6 and abs(xs[-1, 4]) < 1e-6


def test_coast_down_matches_1d_ode():
    p = P
    cla = p.cla_f + p.cla_r

    def rhs(_, v):
        roll = p.f_roll * (p.m * p.g + 0.5 * p.rho * cla * v[0] ** 2) * np.tanh(v[0] / 0.5)
        return [-(0.5 * p.rho * p.cda * v[0] ** 2 + roll) / p.m]

    ref = solve_ivp(rhs, (0, 20.0), [80.0], rtol=1e-10, atol=1e-10, dense_output=True)
    x0 = np.array([0, 0, 0, 80.0, 0, 0, 0, 0.0])
    xs = _run(x0, [0.0, 0.0], straight(), seconds=20.0)
    t = np.arange(len(xs)) * 0.05
    # Neutral pedal is not exactly zero after the smooth relu; allow a small tolerance.
    assert np.max(np.abs(xs[:, 3] - ref.sol(t)[0])) < 0.05


def test_low_speed_is_kinematic():
    # Start off the kinematic manifold (r = 0 with 0.2 rad of steering): within a second
    # at walking pace the yaw rate must settle onto vx * tan(delta) / L.
    x0 = np.array([0, 0, 0, 1.5, 0, 0, 0.2, 0.0])
    xs = _run(x0, [0.2 / P.delta_max, 0.0], circle(100.0), seconds=1.0, dt=0.01, substeps=4)
    vx, r, delta = xs[-1, 3], xs[-1, 5], xs[-1, 6]
    assert r == pytest.approx(vx * np.tan(delta) / P.wheelbase, rel=0.02)


def test_braking_deceleration_is_f1_like():
    from apexzero.vehicle.dynamics import body_derivatives

    for v, lo, hi in [(83.0, 4.0, 7.0), (30.0, 1.8, 3.5)]:
        x = jnp.array([0, 0, 0, v, 0, 0, 0, -1.0])
        dvx = body_derivatives(x, jnp.array([0.0, -1.0]), P, JAX)[0]
        assert lo < -float(dvx) / P.g < hi


def test_dt_convergence_open_loop():
    """RK2 at 5 ms vs RK4 at 1 ms over 20 s of open-loop inputs (no feedback to amplify
    differences): throttle/brake phases crossing zero pedal and weaving at about 1.5 g,
    held per 50 ms control step exactly as a policy's commands are. A straight reference
    (kappa = 0) keeps the open-loop drift away from any Frenet singularity."""
    trj = track_arrays(straight(), dtype=jnp.float64)
    t = np.arange(400) * 0.05
    a_pedal = 0.2 + 0.35 * np.sin(1.0 * t)
    a_steer = 0.03 * np.sin(0.7 * t) / P.delta_max
    us = jnp.asarray(np.stack([a_steer, a_pedal], axis=1))

    def rollout(substeps, method):
        def body(x, u):
            x = control_step(x, u, trj, P, 0.05, substeps, method)
            return x, x
        x0 = jnp.array([0, 0, 0, 40.0, 0, 0, 0, 0.0])
        return np.asarray(jax.lax.scan(body, x0, us)[1])

    a = rollout(10, "rk2")   # 5 ms
    b = rollout(50, "rk4")   # 1 ms
    assert np.all(np.isfinite(a))
    assert b[:, 3].min() > 20.0                                            # stays moving
    assert np.max(np.abs(b[:, 3] * b[:, 5])) / P.g > 1.0                   # real cornering
    assert a[-1, 0] == pytest.approx(b[-1, 0], rel=5e-4)                # distance
    assert np.max(np.abs(a[:, 3] - b[:, 3])) < 0.02                       # vx (m/s)
    assert np.max(np.abs(a[:, 1] - b[:, 1])) < 0.05                       # n (m)
    assert np.max(np.abs(a[:, 5] - b[:, 5])) < 5e-3                       # yaw rate


def test_braking_to_a_stop_never_reverses():
    x0 = np.array([0, 0, 0, 20.0, 0, 0, 0, -1.0])
    xs = _run(x0, [0.0, -1.0], straight(), seconds=6.0)
    assert xs[:, 3].min() > -0.05
    assert abs(xs[-1, 3]) < 0.1
