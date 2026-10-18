"""Functional, vmappable racing environment (gymnax-style) on the shared vehicle model.

Observation (actor, 222-d, egocentric; nothing that identifies the track or lap position):
    preview  40 points at d_k = d_max (k/40)^1.5 ahead along the reference, with
             d_max = clip(6 s * v, 150, 500) m. Per point: left and right edge (x, y)
             in the body frame / 100 m, and kappa * 50.                          (200)
    ego      vx, vy, r, sideslip, lateral position in the corridor, xi, delta,
             pedal, ax, ay, previous action (2), friction-ellipse use f/r,
             slip angles f/r.                                                    (16)
    context  the six car parameters, normalised to [-1, 1].                      (6)
Action: [a_steer, a_pedal] in [-1, 1]^2 at 20 Hz.
Reward: progress / 10 m - 0.02 * ||a - a_prev||^2; on termination -(5 + 5 (v/50)^2).
Termination: reference point within 1 m of an edge, |xi| > 90 deg, tyre slip beyond
alpha_max (10 deg, loss of control; the OCP enforces the same limit), < 2 m/s for 2 s.
Truncation: end of the training window (or of an open strip); the learner bootstraps.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import CONTEXT_RANGES, F1_2025, VehicleParams, with_context
from apexzero.vehicle.sim import (
    TrackArrays,
    control_step,
    edges_at,
    global_pose,
    kappa_at,
    lookup,
    track_length,
    widths_at,
)

N_PREVIEW = 40
OBS_PREVIEW = 5 * N_PREVIEW
OBS_EGO = 16
OBS_CONTEXT = 6
OBS_DIM = OBS_PREVIEW + OBS_EGO + OBS_CONTEXT
ACTION_DIM = 2

_CTX_LO = jnp.array([CONTEXT_RANGES[k][0] for k in CONTEXT_RANGES])
_CTX_HI = jnp.array([CONTEXT_RANGES[k][1] for k in CONTEXT_RANGES])


class EnvParams(NamedTuple):
    dt: float = 0.05
    substeps: int = 10
    window_steps: int = 1200
    edge_margin: float = 1.0
    slow_speed: float = 2.0
    slow_time: float = 2.0
    action_cost: float = 0.02
    progress_scale: float = 10.0
    obs_noise: float = 0.0
    strip_tail: float = 520.0   # open strips truncate this far before their end


class EnvState(NamedTuple):
    x: jax.Array            # [8] vehicle state
    prev_action: jax.Array  # [2]
    accel: jax.Array        # [2] body-frame (ax, ay) over the last control step
    ctx: jax.Array          # [6] car context (physical units), observed
    hidden_mu: jax.Array    # grip multiplier the policy does not observe (DR jitter)
    t: jax.Array            # control steps since reset
    slow: jax.Array         # seconds spent below slow_speed
    s0: jax.Array           # arc length at reset


def default_context() -> jax.Array:
    p = F1_2025
    cla = p.cla_f + p.cla_r
    return jnp.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla])


def params_from_context(ctx: jax.Array, base: VehicleParams = F1_2025,
                        hidden_mu=1.0) -> VehicleParams:
    return with_context(base, ctx[0], ctx[1], ctx[2], ctx[3], ctx[4] * hidden_mu, ctx[5])


def preview_offsets(v: jax.Array) -> jax.Array:
    d_max = jnp.clip(6.0 * v, 150.0, 500.0)
    k = jnp.arange(1, N_PREVIEW + 1, dtype=v.dtype)
    return d_max * (k / N_PREVIEW) ** 1.5


def observe(state: EnvState, tr: TrackArrays, p: VehicleParams) -> jax.Array:
    x = state.x
    s, n, xi, vx, vy, r, delta, pedal = (x[i] for i in range(8))
    X, Y, psi = global_pose(x, tr)
    c, sn = jnp.cos(psi), jnp.sin(psi)

    sk = s + preview_offsets(jnp.maximum(vx, 0.0))

    def body(px, py):
        dx, dy = px - X, py - Y
        return (c * dx + sn * dy) / 100.0, (-sn * dx + c * dy) / 100.0

    (lx, ly), (rx, ry) = edges_at(sk, tr)
    lxb, lyb = body(lx, ly)
    rxb, ryb = body(rx, ry)
    kap = kappa_at(sk, tr) * 50.0
    preview = jnp.stack([lxb, lyb, rxb, ryb, kap], axis=-1).reshape(-1)

    wl, wr = widths_at(s, tr)
    lateral = (n - 0.5 * (wl - wr)) / (0.5 * (wl + wr))
    fo = tyre_forces(x, p, JAX)
    ego = jnp.stack([
        vx / 50.0, vy / 5.0, r, jnp.arctan2(vy, jnp.maximum(vx, 1.0)), lateral, xi,
        delta / p.delta_max, pedal, state.accel[0] / 50.0, state.accel[1] / 50.0,
        state.prev_action[0], state.prev_action[1],
        jnp.minimum(fo.util_f, 2.0) - 1.0, jnp.minimum(fo.util_r, 2.0) - 1.0,
        fo.alpha_f * 5.0, fo.alpha_r * 5.0,
    ])
    ctx = 2.0 * (state.ctx - _CTX_LO) / (_CTX_HI - _CTX_LO) - 1.0
    return jnp.concatenate([preview, ego, ctx])


def reset(key: jax.Array, tr: TrackArrays, v_ref: jax.Array, ctx: jax.Array,
          params: EnvParams = EnvParams(), hidden_mu=1.0) -> tuple[jax.Array, EnvState]:
    """Random start: s0 uniform, lateral offset within half the corridor, small heading
    error, speed 60-100% of the reference profile at s0."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    length = track_length(tr)
    s_hi = jnp.where(tr.closed, length, jnp.maximum(length - params.strip_tail - 1000.0, 1.0))
    s0 = jax.random.uniform(k1, (), minval=0.0, maxval=s_hi)
    wl, wr = widths_at(s0, tr)
    centre, half = 0.5 * (wl - wr), 0.5 * (wl + wr)
    n0 = centre + 0.5 * (half - params.edge_margin) * jax.random.uniform(k2, (), minval=-1.0, maxval=1.0)
    xi0 = 0.03 * jax.random.normal(k3, ())
    v0 = lookup(v_ref, s0, tr) * jax.random.uniform(k4, (), minval=0.6, maxval=1.0)
    x = jnp.array([s0, n0, xi0, v0, 0.0, 0.0, 0.0, 0.0], dtype=tr.ds.dtype)
    state = EnvState(
        x=x, prev_action=jnp.zeros(2, x.dtype), accel=jnp.zeros(2, x.dtype), ctx=ctx.astype(x.dtype),
        hidden_mu=jnp.asarray(hidden_mu, x.dtype), t=jnp.zeros((), jnp.int32),
        slow=jnp.zeros((), x.dtype), s0=s0,
    )
    p = params_from_context(state.ctx, hidden_mu=state.hidden_mu)
    return observe(state, tr, p), state


def step_no_reset(state: EnvState, action: jax.Array, tr: TrackArrays,
                  params: EnvParams = EnvParams()):
    """One control step. Returns (obs, state, reward, terminated, truncated, info)."""
    p = params_from_context(state.ctx, hidden_mu=state.hidden_mu)
    a = jnp.clip(action, -1.0, 1.0)
    x0 = state.x
    x1 = control_step(x0, a, tr, p, params.dt, params.substeps)

    # Body-frame acceleration from the velocity change (for the observation).
    ax = (x1[3] - x0[3]) / params.dt - x1[4] * x1[5]
    ay = (x1[4] - x0[4]) / params.dt + x1[3] * x1[5]

    s1, n1, xi1, vx1 = x1[0], x1[1], x1[2], x1[3]
    wl, wr = widths_at(s1, tr)
    off_track = (n1 > wl - params.edge_margin) | (n1 < -(wr - params.edge_margin))
    fo1 = tyre_forces(x1, p, JAX)
    slipped = (jnp.abs(fo1.alpha_f) > p.alpha_max) | (jnp.abs(fo1.alpha_r) > p.alpha_max)
    spun = (jnp.abs(xi1) > 0.5 * jnp.pi) | slipped
    slow = jnp.where(vx1 < params.slow_speed, state.slow + params.dt, 0.0)
    stalled = slow > params.slow_time
    bad = ~jnp.all(jnp.isfinite(x1))
    terminated = off_track | spun | stalled | bad

    t = state.t + 1
    length = track_length(tr)
    end_of_strip = (~tr.closed) & (s1 > length - params.strip_tail)
    truncated = ((t >= params.window_steps) | end_of_strip) & ~terminated

    progress = jnp.where(bad, 0.0, s1 - x0[0])
    reward = progress / params.progress_scale - params.action_cost * jnp.sum((a - state.prev_action) ** 2)
    v = jnp.where(bad, 0.0, jnp.hypot(vx1, x1[4]))
    reward = reward - jnp.where(terminated, 5.0 + 5.0 * (v / 50.0) ** 2, 0.0)
    reward = jnp.where(jnp.isfinite(reward), reward, -10.0)

    x1 = jnp.where(bad, x0, x1)
    new = EnvState(x=x1, prev_action=a, accel=jnp.stack([ax, ay]), ctx=state.ctx,
                   hidden_mu=state.hidden_mu, t=t,
                   slow=slow, s0=state.s0)
    obs = observe(new, tr, p)
    info = {"progress": progress, "speed": vx1, "off_track": off_track, "spun": spun,
            "slipped": slipped, "stalled": stalled}
    return obs, new, reward, terminated, truncated, info


def step(key: jax.Array, state: EnvState, action: jax.Array, tr: TrackArrays,
         v_ref: jax.Array, params: EnvParams = EnvParams()):
    """Step with automatic reset. ``info['final_obs']`` holds the pre-reset observation
    so the learner can bootstrap truncated windows."""
    obs, new, reward, term, trunc, info = step_no_reset(state, action, tr, params)
    obs_r, state_r = reset(key, tr, v_ref, state.ctx, params)
    done = term | trunc
    state_out = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_r, new)
    obs_out = jnp.where(done, obs_r, obs)
    info = dict(info, final_obs=obs)
    return obs_out, state_out, reward, term, trunc, info
