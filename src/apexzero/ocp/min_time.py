"""Minimum-lap-time optimal control on the shared vehicle ODE (CasADi + IPOPT).

Arc length s is the independent variable. With time-domain dynamics dx/dt = f(x, u, kappa)
and progress rate sdot = f_s, the spatial dynamics are dx/ds = f / sdot and the lap time
is the integral of dt/ds = 1 / sdot. The state excludes s itself:

    z = [n, xi, vx, vy, r, delta, pedal],  u = [a_steer, a_pedal] in [-1, 1]^2

Discretisation: trapezoidal collocation on a uniform s grid, controls piecewise linear,
periodic boundary conditions for closed circuits. The corridor constraint
-(w_right - margin) <= n <= w_left - margin is the same one the RL environment terminates
on. Steering lag and rate limit, and pedal lag, are part of the dynamics, so the optimum
is reachable by a policy acting through the same actuators.

The multipliers of the corridor bounds are returned: they are dT*/d(width) at each
station (how many seconds one more metre of asphalt would be worth there).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from apexzero.oracle.qss import GG, speed_profile
from apexzero.tracks.track import Track
from apexzero.vehicle import dynamics
from apexzero.vehicle.ops import CASADI
from apexzero.vehicle.params import VehicleParams

NZ = 7
NU = 2
Z_NAMES = ("n", "xi", "vx", "vy", "r", "delta", "pedal")


@dataclass
class OCPResult:
    s: np.ndarray            # [N] stations
    z: np.ndarray            # [N, 7]
    u: np.ndarray            # [N, 2]
    t: np.ndarray            # [N] time at each station (t[0] = 0)
    lap_time: float
    success: bool
    status: str
    iterations: int
    solve_seconds: float
    kappa: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    dual_left: np.ndarray    # >= 0, seconds per metre of extra left width
    dual_right: np.ndarray   # >= 0, seconds per metre of extra right width
    meta: dict = field(default_factory=dict)


def resample(track: Track, ds: float):
    """Track arrays on a uniform OCP grid of spacing ~ds."""
    n = round(track.length / ds) if track.closed else int(track.length // ds) + 1
    h = track.length / n if track.closed else ds
    s = np.arange(n) * h
    period = track.length if track.closed else None

    def interp(a):
        if period is not None:
            return np.interp(s, np.r_[track.s, track.length], np.r_[a, a[0]])
        return np.interp(s, track.s, a)

    return s, h, interp(track.kappa), interp(track.w_left), interp(track.w_right)


def spatial_dynamics(p: VehicleParams) -> ca.Function:
    z = ca.SX.sym("z", NZ)
    u = ca.SX.sym("u", NU)
    k = ca.SX.sym("k")
    x = ca.vertcat(0.0, z)
    dxdt = dynamics.f(x, u, k, p, CASADI)
    sdot = dxdt[0]
    return ca.Function("g", [z, u, k], [dxdt[1:] / sdot, 1.0 / sdot], ["z", "u", "k"], ["dzds", "dtds"])


def slip_angles(p: VehicleParams) -> ca.Function:
    """(alpha_f, alpha_r) exactly as the shared tyre model computes them."""
    z = ca.SX.sym("z", NZ)
    fo = dynamics.tyre_forces(ca.vertcat(0.0, z), p, CASADI)
    return ca.Function("alpha", [z], [ca.vertcat(fo.alpha_f, fo.alpha_r)])


def initial_guess(kappa, h, p: VehicleParams, closed: bool):
    v = speed_profile(kappa, h, GG.from_params(p), closed=closed)
    v = np.clip(0.97 * v, 8.0, None)
    z0 = np.zeros((len(kappa), NZ))
    z0[:, 2] = v
    z0[:, 4] = kappa * v
    z0[:, 5] = np.clip(np.arctan(p.lf + p.lr) * 0 + (p.lf + p.lr) * kappa, -p.delta_max, p.delta_max)
    dv = np.gradient(v) / h * v  # dv/dt
    z0[:, 6] = np.clip(np.where(dv > 0, 0.6, dv / 40.0), -1, 1)
    u0 = np.stack([z0[:, 5] / p.delta_max, z0[:, 6]], axis=1)
    return z0, u0


def solve_min_time(
    track: Track,
    p: VehicleParams,
    ds: float = 3.0,
    margin: float = 1.0,
    init: tuple[np.ndarray, np.ndarray] | None = None,
    max_iter: int = 3000,
    tol: float = 1e-6,
    print_level: int = 0,
    u_rate_weight: float = 1e-4,
) -> OCPResult:
    """Solve the periodic minimum-lap-time problem on ``track`` for car ``p``.

    ``u_rate_weight`` adds a tiny penalty on control differences between stations (in
    seconds-equivalent units), which removes chattering in the singular arcs without
    measurably changing the lap time; it is reported in ``meta``.
    """
    import time

    s, h, kap, wl, wr = resample(track, ds)
    n = len(s)
    closed = track.closed
    g = spatial_dynamics(p).map(n, "serial")

    Z = ca.MX.sym("Z", NZ, n)
    U = ca.MX.sym("U", NU, n)
    K = ca.DM(kap).T
    dz, dt = g(Z, U, K)

    if closed:
        nxt = list(range(1, n)) + [0]
        cur = list(range(n))
    else:
        nxt = list(range(1, n))
        cur = list(range(n - 1))
    defects = Z[:, nxt] - Z[:, cur] - 0.5 * h * (dz[:, cur] + dz[:, nxt])
    lap = ca.sum2(0.5 * h * (dt[:, cur] + dt[:, nxt]))
    du = U[:, nxt] - U[:, cur]
    obj = lap + u_rate_weight * ca.sumsqr(du)
    # Loss-of-control limit on tyre slip, identical to the environment's termination.
    alpha = slip_angles(p).map(n, "serial")(Z)

    w = ca.veccat(Z, U)
    lbz = np.array([-np.inf, -1.2, 5.0, -25.0, -5.0, -p.delta_max, -1.0])
    ubz = np.array([np.inf, 1.2, 120.0, 25.0, 5.0, p.delta_max, 1.0])
    lbZ = np.tile(lbz[:, None], (1, n))
    ubZ = np.tile(ubz[:, None], (1, n))
    lbZ[0, :] = -(wr - margin)
    ubZ[0, :] = wl - margin
    lbx = np.concatenate([lbZ.T.reshape(-1), -np.ones(NU * n)])
    ubx = np.concatenate([ubZ.T.reshape(-1), np.ones(NU * n)])

    if init is None:
        z0, u0 = initial_guess(kap, h, p, closed)
    else:
        z0, u0 = init
    x0 = np.concatenate([z0.reshape(-1), u0.reshape(-1)])
    x0 = np.clip(x0, lbx, ubx)

    g_all = ca.vertcat(ca.vec(defects), ca.vec(alpha))
    n_def = defects.numel()
    lbg = np.concatenate([np.zeros(n_def), -p.alpha_max * np.ones(2 * n)])
    ubg = np.concatenate([np.zeros(n_def), p.alpha_max * np.ones(2 * n)])
    nlp = {"x": w, "f": obj, "g": g_all}
    opts = {
        "ipopt.max_iter": max_iter,
        "ipopt.tol": tol,
        "ipopt.print_level": print_level,
        "print_time": False,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.linear_solver": "mumps",
    }
    solver = ca.nlpsol("ocp", "ipopt", nlp, opts)
    t0 = time.perf_counter()
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    elapsed = time.perf_counter() - t0
    stats = solver.stats()

    xopt = np.asarray(sol["x"]).ravel()
    zopt = xopt[: NZ * n].reshape(n, NZ)
    uopt = xopt[NZ * n:].reshape(n, NU)
    lam = np.asarray(sol["lam_x"]).ravel()[: NZ * n].reshape(n, NZ)[:, 0]
    # IPOPT sign convention: lam_x > 0 at an active upper bound, < 0 at an active lower one.
    dual_left = np.clip(lam, 0.0, None)
    dual_right = np.clip(-lam, 0.0, None)

    dtds = np.asarray(spatial_dynamics(p).map(n)(zopt.T, uopt.T, kap[None, :])[1]).ravel()
    seg = 0.5 * h * (dtds[cur] + dtds[nxt])
    t = np.r_[0.0, np.cumsum(seg)][:n]
    return OCPResult(
        s=s, z=zopt, u=uopt, t=t, lap_time=float(seg.sum()), success=bool(stats["success"]),
        status=str(stats["return_status"]), iterations=int(stats["iter_count"]),
        solve_seconds=elapsed, kappa=kap, w_left=wl, w_right=wr, dual_left=dual_left,
        dual_right=dual_right,
        meta={"ds": h, "margin": margin, "u_rate_weight": u_rate_weight, "track": track.name},
    )
