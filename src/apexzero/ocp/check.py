"""Consistency checks between the OCP's discretisation and the time-domain simulator.

``shooting_check`` starts the simulator at every OCP station z_i, applies that interval's
controls (linear in s, as the OCP assumes), integrates with RK4 for the OCP's own interval
time, and measures where it lands relative to z_{i+1}. Nothing is fed back between
intervals, so instability near the limit cannot amplify the error: this isolates the
discretisation mismatch between trapezoidal collocation in s and RK integration in t.

The interval times corrected by the arrival error give an independent lap time.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.ocp.min_time import OCPResult
from apexzero.vehicle.dynamics import f
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import VehicleParams


@dataclass
class ShootingCheck:
    lap_time_ocp: float
    lap_time_shooting: float
    rel_lap_error: float
    max_position_error: float     # m along s at the end of an interval
    max_lateral_error: float      # m
    max_speed_error: float        # m/s
    p99_speed_error: float


def shooting_check(res: OCPResult, p: VehicleParams, substeps: int = 20) -> ShootingCheck:
    jax.config.update("jax_enable_x64", True)
    n = len(res.s)
    h_s = res.meta["ds"]
    L = n * h_s
    s_grid = np.r_[res.s, L]
    k_grid = np.r_[res.kappa, res.kappa[0]]
    z_next = np.vstack([res.z[1:], res.z[:1]])
    u_next = np.vstack([res.u[1:], res.u[:1]])
    t_next = np.r_[res.t[1:], res.lap_time]
    dt_int = t_next - res.t

    def one(s0, s1, z0, u0, u1, k0, k1, dt):
        def kappa(s):
            w = jnp.clip((s - s0) / (s1 - s0), 0.0, 1.0)
            return (1 - w) * k0 + w * k1

        def ctrl(s):
            w = jnp.clip((s - s0) / (s1 - s0), 0.0, 1.0)
            return (1 - w) * u0 + w * u1

        def rhs(x):
            return f(x, ctrl(x[0]), kappa(x[0]), p, JAX)

        def step(x, _):
            h = dt / substeps
            a = rhs(x)
            b = rhs(x + 0.5 * h * a)
            c = rhs(x + 0.5 * h * b)
            d = rhs(x + h * c)
            return x + h / 6 * (a + 2 * b + 2 * c + d), None

        x0 = jnp.concatenate([jnp.array([s0]), z0])
        xe, _ = jax.lax.scan(step, x0, None, length=substeps)
        sdot = rhs(xe)[0]
        return xe, dt - (xe[0] - s1) / sdot

    run = jax.jit(jax.vmap(one))
    xe, t_corr = run(jnp.asarray(res.s), jnp.asarray(s_grid[1:]), jnp.asarray(res.z),
                     jnp.asarray(res.u), jnp.asarray(u_next), jnp.asarray(res.kappa),
                     jnp.asarray(k_grid[1:]), jnp.asarray(dt_int))
    xe, t_corr = np.asarray(xe), np.asarray(t_corr)
    pos_err = np.abs(xe[:, 0] - s_grid[1:])
    lat_err = np.abs(xe[:, 1] - z_next[:, 0])
    v_err = np.abs(xe[:, 3] - z_next[:, 2])
    lap_sh = float(np.sum(t_corr))
    return ShootingCheck(
        lap_time_ocp=res.lap_time, lap_time_shooting=lap_sh,
        rel_lap_error=lap_sh / res.lap_time - 1.0, max_position_error=float(pos_err.max()),
        max_lateral_error=float(lat_err.max()), max_speed_error=float(v_err.max()),
        p99_speed_error=float(np.percentile(v_err, 99)),
    )
