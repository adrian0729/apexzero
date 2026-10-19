"""Replay an OCP solution in the time-domain JAX simulator.

The OCP's controls are scheduled by arc length, u(s), and a light feedback term on the
lateral and heading error keeps the open-loop replay from drifting:

    a_steer = u_steer(s) + k_n (n*(s) - n) + k_xi (xi*(s) - xi)
    a_pedal = u_pedal(s) + k_v (v*(s) - vx)

Controls are recomputed at every integration substep (``hold = False``) to test the
discretisations against each other, or held for 50 ms like a policy (``hold = True``).
The lap time from the replay vs the OCP's lap time is the consistency gate between the
two compilations of the shared ODE.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.ocp.min_time import OCPResult
from apexzero.tracks.track import Track
from apexzero.vehicle.params import VehicleParams
from apexzero.vehicle.sim import rk2_step, track_arrays


@dataclass
class Replay:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    lap_time: float
    completed: bool
    max_lateral_error: float
    max_corridor_violation: float


def replay(res: OCPResult, track: Track, p: VehicleParams, h: float = 0.005,
           hold_steps: int = 1, gains=(0.05, 0.5, 0.05), max_time: float | None = None) -> Replay:
    jax.config.update("jax_enable_x64", True)
    tr = track_arrays(track, dtype=jnp.float64)
    L = float(track.length)
    s_grid = jnp.asarray(np.r_[res.s, L])
    z_grid = jnp.asarray(np.vstack([res.z, res.z[:1]]))
    u_grid = jnp.asarray(np.vstack([res.u, res.u[:1]]))
    wl = jnp.asarray(np.r_[res.w_left, res.w_left[0]])
    wr = jnp.asarray(np.r_[res.w_right, res.w_right[0]])
    k_n, k_xi, k_v = gains

    def ref(s):
        sm = jnp.mod(s, L)
        zi = jax.vmap(lambda col: jnp.interp(sm, s_grid, col), in_axes=1)(z_grid)
        ui = jax.vmap(lambda col: jnp.interp(sm, s_grid, col), in_axes=1)(u_grid)
        return zi, ui, jnp.interp(sm, s_grid, wl), jnp.interp(sm, s_grid, wr)

    def controller(x):
        zr, ur, _, _ = ref(x[0])
        a_steer = ur[0] + k_n * (zr[0] - x[1]) + k_xi * (zr[1] - x[2])
        a_pedal = ur[1] + k_v * (zr[2] - x[3])
        return jnp.clip(jnp.stack([a_steer, a_pedal]), -1.0, 1.0)

    n_steps = int((max_time or 1.5 * res.lap_time) / h)

    @jax.jit
    def run(x0):
        def body(carry, i):
            x, u = carry
            u = jnp.where(i % hold_steps == 0, controller(x), u)
            x = rk2_step(x, u, tr, p, h)
            zr, _, wli, wri = ref(x[0])
            viol = jnp.maximum(x[1] - (wli - res.meta["margin"]), -(wri - res.meta["margin"]) - x[1])
            return (x, u), (x, u, jnp.abs(x[1] - zr[0]), viol)
        u0 = controller(x0)
        _, out = jax.lax.scan(body, (x0, u0), jnp.arange(n_steps))
        return out

    x0 = jnp.asarray(np.r_[0.0, res.z[0]])
    xs, us, lat_err, viol = (np.asarray(a) for a in run(x0))
    t = (np.arange(n_steps) + 1) * h
    crossed = np.nonzero(xs[:, 0] >= L)[0]
    completed = len(crossed) > 0 and np.all(np.isfinite(xs[: crossed[0] + 1]))
    if completed:
        k = crossed[0]
        s_prev = xs[k - 1, 0] if k > 0 else 0.0
        frac = (L - s_prev) / (xs[k, 0] - s_prev)
        lap = float(t[k] - h + frac * h)
        sl = slice(0, k + 1)
    else:
        lap = float("nan")
        sl = slice(None)
    return Replay(t=t[sl], x=xs[sl], u=us[sl], lap_time=lap, completed=bool(completed),
                  max_lateral_error=float(np.max(lat_err[sl])),
                  max_corridor_violation=float(np.max(viol[sl])))


def replay_window(track: Track, x0: np.ndarray, acts: np.ndarray, ctx: np.ndarray | None = None,
                  gamma: float = 0.995, substeps: int = 10) -> tuple[float, bool, np.ndarray]:
    """Open-loop replay of TO-OCP controls through the RL env's own float64 step.

    Returns the discounted return, whether every step survived the env's termination
    checks, and the states [T+1, 8]. This is the TO-OCP's acceptance test: the return must
    reproduce J* and the car must never terminate.
    """
    from apexzero.env import racing

    jax.config.update("jax_enable_x64", True)
    tr = track_arrays(track, dtype=jnp.float64)
    params = racing.EnvParams(window_steps=10**9, substeps=substeps)
    c = racing.default_context() if ctx is None else jnp.asarray(ctx)
    st = racing.EnvState(x=jnp.asarray(x0, jnp.float64), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=c.astype(jnp.float64), hidden_mu=jnp.array(1.0), t=jnp.array(0),
                         slow=jnp.array(0.0), s0=jnp.asarray(x0[0], jnp.float64))

    def body(carry, a):
        st, alive = carry
        _, st2, r, term, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, alive & ~term), (st2.x, r, alive & ~term)

    _, (xs, rs, alive) = jax.lax.scan(body, (st, jnp.array(True)), jnp.asarray(acts, jnp.float64))
    rs = np.asarray(rs)
    ret = float(np.sum(gamma ** np.arange(len(rs)) * rs))
    return ret, bool(np.asarray(alive).all()), np.vstack([np.asarray(x0)[None], np.asarray(xs)])
