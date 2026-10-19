"""Versioned JSON trajectory export: one track and one or more lap "ghosts".

Every demo surface (the static viewer, the matplotlib figures, the optional Rerun log, a
later render mode for the video) reads this one file format, so a figure can be redrawn
from an export without re-running a policy or an optimiser, and an export made today
still loads when the final agents exist.

A *ghost* is one timed lap: struct-of-arrays channels sampled at increasing times, with
``t = 0`` at the start line and ``s`` running from 0 to the lap length. Two sources:

- a policy: :func:`rollout_policy` drives the deterministic policy with the timed-lap
  protocol of ``apexzero.eval.laps.timed_laps`` (rolling start at s = 0, lap 2 is timed)
  and keeps the full vehicle state; :func:`policy_ghost` cuts out the timed lap.
  :func:`ghost_from_lap_trace` accepts ``timed_laps``' own result instead (a thinner
  trace that stops one sample short of the lap end, closed with the harness's lap time).
- the s-domain OCP: :func:`ocp_ghost` maps an ``OCPResult`` from stations in s to time,
  using the OCP's own t(s) (trapezoidal dt/ds), so the ghost's lap time *is*
  ``OCPResult.lap_time``.

Lap boundaries are interpolated exactly as ``timed_laps`` interpolates its crossings, so a
policy ghost's lap time equals the evaluation harness's number for the same run.

Schema ``apexzero.trajectory`` 1.x: additive changes (new optional channels or keys) keep
the major version; anything a 1.x reader could misread bumps it.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from apexzero.tracks.corners import R_TEST
from apexzero.tracks.track import Track
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import Ops
from apexzero.vehicle.params import F1_2025, VehicleParams
from apexzero.viz.sectors import Sector, corner_sectors, sector_deltas

SCHEMA = "apexzero.trajectory"
SCHEMA_VERSION = "1.0.0"
KINDS = ("policy", "ocp", "telemetry")
BADGES = ("CERTIFIED", "MODEL-DEPENDENT", "POST HOC")
VIEWER_PATH = Path(__file__).parent / "viewer" / "index.html"


@dataclass(frozen=True)
class Channel:
    unit: str
    description: str
    decimals: int
    required: bool = False


CHANNELS: dict[str, Channel] = {
    "t": Channel("s", "time since the ghost crossed the start line", 4, True),
    "s": Channel("m", "arc length along the reference line, 0 at the start line", 3, True),
    "n": Channel("m", "lateral offset from the reference line, positive to the left", 3, True),
    "x": Channel("m", "world x of the car's reference point", 2, True),
    "y": Channel("m", "world y of the car's reference point", 2, True),
    "heading": Channel("rad", "yaw angle in the world frame, unwrapped (continuous)", 4, True),
    "speed": Channel("m/s", "ground speed |(vx, vy)|", 3, True),
    "vx": Channel("m/s", "body-frame longitudinal velocity", 3),
    "vy": Channel("m/s", "body-frame lateral velocity", 3),
    "yaw_rate": Channel("rad/s", "yaw rate", 4),
    "steering": Channel("rad", "road-wheel steering angle (actuator state, after lag)", 4),
    "throttle": Channel("1", "throttle, 0..1 (pedal actuator state, after lag)", 3),
    "brake": Channel("1", "brake, 0..1 (pedal actuator state, after lag)", 3),
    "cmd_steer": Channel("1", "steering command in [-1, 1] as applied (policy output "
                              "clipped) from this sample on", 3),
    "cmd_pedal": Channel("1", "pedal command in [-1, 1] (>0 throttle) as applied (policy "
                              "output clipped) from this sample on", 3),
    "util_front": Channel("1", "front-axle force demand / friction ellipse, before the tyre "
                               "saturates (> 1: demand clipped back onto the ellipse)", 3),
    "util_rear": Channel("1", "rear-axle force demand / friction ellipse, before saturation", 3),
    "ellipse_use": Channel("1", "grip actually used on the busier axle, |F| / F_max after "
                                "saturation (1 = at the limit)", 3),
}
REQUIRED = tuple(k for k, c in CHANNELS.items() if c.required)
MIN_GAP = 1e-3   # s; closest two samples may be (t is exported to 0.1 ms)


class SchemaError(ValueError):
    """An export does not conform to the trajectory schema."""


# --------------------------------------------------------------------------- geometry


def _numpy_ops() -> Ops:
    """The vehicle model's ops shim on plain NumPy (float64, no JAX config side effects)."""
    return Ops(
        name="numpy", sin=np.sin, cos=np.cos, tan=np.tan, atan=np.arctan, tanh=np.tanh,
        sqrt=np.sqrt, stack=lambda xs: np.stack(np.broadcast_arrays(*xs), axis=-1),
        index=lambda x, i: x[..., i], where=np.where,
    )


NUMPY: Ops = _numpy_ops()


def _grid_index(track: Track, s: np.ndarray):
    """(i0, i1, frac) of the simulator's piecewise-linear pose lookup at arc length s."""
    n = track.n_points
    i = np.asarray(s, float) / track.ds
    i = np.mod(i, n) if track.closed else np.clip(i, 0.0, n - 1)
    i0 = np.minimum(np.floor(i).astype(int), n - 1)
    frac = i - i0
    i1 = np.mod(i0 + 1, n) if track.closed else np.minimum(i0 + 1, n - 1)
    return i0, i1, frac


def frenet_to_world(track: Track, s, n, xi=None):
    """World (x, y, heading) of Frenet (s, n, xi), exactly as ``vehicle.sim.global_pose``
    computes it (linear pose lookup, wrapped heading difference), in NumPy float64.

    Without ``xi`` the heading is the reference line's.
    """
    i0, i1, frac = _grid_index(track, s)
    cx = (1 - frac) * track.x[i0] + frac * track.x[i1]
    cy = (1 - frac) * track.y[i0] + frac * track.y[i1]
    d = np.mod(track.theta[i1] - track.theta[i0] + np.pi, 2 * np.pi) - np.pi
    th = track.theta[i0] + frac * d
    n = np.asarray(n, float)
    heading = th if xi is None else th + np.asarray(xi, float)
    return cx - n * np.sin(th), cy + n * np.cos(th), heading


def state_channels(track: Track, x: np.ndarray, p: VehicleParams,
                   cmd: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """All ghost channels from full vehicle states ``x`` [T, 8] = (s, n, xi, vx, vy, r,
    delta, pedal), with ``s`` cumulative (unwrapped) along the run."""
    x = np.asarray(x, float)
    s, n, xi, vx, vy, r, delta, pedal = x.T
    wx, wy, heading = frenet_to_world(track, s, n, xi)
    fo = tyre_forces(x, p, NUMPY)
    ch = {
        "s": s, "n": n, "x": wx, "y": wy, "heading": np.unwrap(heading),
        "speed": np.hypot(vx, vy), "vx": vx, "vy": vy, "yaw_rate": r, "steering": delta,
        "throttle": np.clip(pedal, 0.0, 1.0), "brake": np.clip(-pedal, 0.0, 1.0),
        "util_front": np.asarray(fo.util_f, float), "util_rear": np.asarray(fo.util_r, float),
    }
    # The model scales an over-demanded force back by 1 / smax(util, 1): what is left is
    # the grip actually used.
    used = [u / NUMPY.smax(u, 1.0, 5e-3) for u in (ch["util_front"], ch["util_rear"])]
    ch["ellipse_use"] = np.maximum(*used)
    if cmd is not None:
        cmd = np.asarray(cmd, float)
        ch["cmd_steer"], ch["cmd_pedal"] = cmd[:, 0], cmd[:, 1]
    return ch


# --------------------------------------------------------------------------- ghosts


@dataclass
class Ghost:
    id: str
    label: str
    kind: str                       # one of KINDS
    lap_time: float
    channels: dict[str, np.ndarray]
    badge: str | None = None        # one of BADGES, or None when no badge applies
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind {self.kind!r} not in {KINDS}")
        if self.badge is not None and self.badge not in BADGES:
            raise ValueError(f"badge {self.badge!r} not in {BADGES}")
        missing = [k for k in REQUIRED if k not in self.channels]
        if missing:
            raise ValueError(f"ghost {self.id!r} lacks required channels {missing}")

    @property
    def t(self) -> np.ndarray:
        return self.channels["t"]

    @property
    def s(self) -> np.ndarray:
        return self.channels["s"]

    def time_at(self, s) -> np.ndarray:
        """t(s) by linear interpolation (s must be strictly increasing along the lap)."""
        if np.any(np.diff(self.s) <= 0):
            raise ValueError(f"ghost {self.id!r}: s is not strictly increasing")
        return np.interp(s, self.s, self.t)


def lap_window(t: np.ndarray, channels: dict[str, np.ndarray], start: float,
               end: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cut [start, end] in arc length out of a run and re-zero time and distance there.

    Boundary samples are interpolated linearly between the two samples that bracket each
    crossing, exactly as ``timed_laps`` times its laps. ``channels['s']`` must be
    cumulative (unwrapped) and increasing through both crossings.
    """
    t = np.asarray(t, float)
    s = np.asarray(channels["s"], float)

    def crossing(target: float) -> tuple[int, float]:
        idx = np.nonzero(s >= target)[0]
        if not len(idx):
            raise ValueError(f"the run never reaches s = {target:.1f} m (max {s.max():.1f})")
        k = int(idx[0])
        if k == 0:
            if s[0] != target:
                raise ValueError("the run starts past the window start")
            return 0, 0.0
        return k, (target - s[k - 1]) / max(s[k] - s[k - 1], 1e-12)

    def at(k: int, w: float, a: np.ndarray):
        return a[k] if k == 0 else (1.0 - w) * a[k - 1] + w * a[k]

    k0, w0 = crossing(start)
    k1, w1 = crossing(end)
    t_start, t_end = at(k0, w0, t), at(k1, w1, t)
    inner = np.arange(k0, k1)
    # Samples within MIN_GAP of a boundary are dropped so time stays strictly increasing
    # after the export rounds it.
    inner = inner[(s[inner] > start) & (s[inner] < end)
                  & (t[inner] - t_start > MIN_GAP) & (t_end - t[inner] > MIN_GAP)]
    out = {}
    for name, a in dict(channels, t=t).items():
        a = np.asarray(a, float)
        out[name] = np.concatenate([[at(k0, w0, a)], a[inner], [at(k1, w1, a)]])
    t0 = out["t"][0]
    out["t"] = out["t"] - t0
    out["s"] = out["s"] - start
    out["s"][0], out["s"][-1] = 0.0, end - start
    return out["t"], out


@dataclass
class PolicyRollout:
    """One deterministic run with the timed-lap protocol, keeping the full state."""

    t: np.ndarray        # [T + 1] s; t[0] = 0 is the initial state
    x: np.ndarray        # [T + 1, 8] vehicle state
    action: np.ndarray   # [T + 1, 2] command applied from t[k] to t[k] + dt
    alive: np.ndarray    # [T + 1] False from the first terminated step on
    laps: int
    v_start: float
    ctx: np.ndarray      # [6] car context the policy saw
    params: VehicleParams


def load_policy(path: str | Path) -> Callable:
    """Deterministic policy ``obs [B, D] -> action [B, 2]`` from a ``train.py`` agent.pkl
    (actor params + observation normaliser), the same function ``train.py`` evaluates."""
    import jax.numpy as jnp

    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd
    from apexzero.env import racing

    with open(path, "rb") as fh:
        d = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    actor = Actor(cfg, racing.ACTION_DIM, False)
    rms = RunningMeanStd(**{k: jnp.asarray(v) for k, v in d["obs_rms"].items()})
    params = d["actor"]

    def policy(obs):
        return actor.apply(params, rms.normalize(obs))[0]

    return policy


def rollout_policy(policy_fn: Callable, track: Track, v_start: float, laps: int = 2,
                   params=None, ctx=None, n0: float = 0.0,
                   max_steps: int | None = None, unsealed: bool = False) -> PolicyRollout:
    """Run ``policy_fn`` (``obs [B, D] -> action [B, 2]``) from a rolling start at s = 0,
    as ``timed_laps`` does for its first (unjittered) run, but record every state.

    Float32 track arrays and the same step function as the harness, so the lap matches the
    harness's run 0 up to batch-size rounding in the network. The harness only runs in
    float32 (its scan carry mixes float32 state with default-dtype context), so the rollout
    forces float32 even if the caller has enabled ``jax_enable_x64`` (the OCP checks do).
    Refuses lockbox circuits.
    """
    import contextlib

    import jax

    lockbox_guard(track.name, unsealed)
    float32 = getattr(jax, "enable_x64", None)
    with float32(False) if float32 is not None else contextlib.nullcontext():
        return _rollout_policy_f32(policy_fn, track, v_start, laps, params, ctx, n0, max_steps)


def _rollout_policy_f32(policy_fn, track, v_start, laps, params, ctx, n0, max_steps):
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle.sim import track_arrays

    params = racing.EnvParams() if params is None else params
    tr = track_arrays(track)
    ctx = racing.default_context() if ctx is None else jnp.asarray(ctx, jnp.float32)
    L = float(track.length)
    if max_steps is None:
        max_steps = int(laps * 3.0 * L / max(v_start, 20.0) / params.dt) + 200
    eval_params = params._replace(window_steps=10**9)

    x0 = np.zeros(8, np.float32)
    x0[1], x0[3] = n0, v_start
    x0 = jnp.asarray(x0)
    st = racing.EnvState(x=x0, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx,
                         hidden_mu=jnp.array(1.0), t=jnp.zeros((), jnp.int32),
                         slow=jnp.zeros(()), s0=x0[0])
    p = racing.params_from_context(ctx)
    obs = racing.observe(st, tr, p)

    @jax.jit
    def run(st, obs):
        def body(c, _):
            st, ob, alive = c
            a = policy_fn(ob[None])[0]
            ob2, st2, _, term, _, _ = racing.step_no_reset(st, a, tr, eval_params)
            st2 = jax.tree.map(lambda new, old: jnp.where(alive, new, old), st2, st)
            alive2 = alive & ~term
            # Record what the simulator applied (step_no_reset clips to [-1, 1]), not the
            # raw policy mean, which can lie outside the box.
            return (st2, ob2, alive2), (st2.x, jnp.clip(a, -1.0, 1.0), alive2)

        return jax.lax.scan(body, (st, obs, jnp.array(True)), None, max_steps)[1]

    xs, acts, alive = (np.asarray(a) for a in run(st, obs))
    x_all = np.vstack([np.asarray(x0)[None], xs]).astype(float)
    a_all = np.vstack([acts, acts[-1:]]).astype(float)
    alive_all = np.r_[True, alive]
    t = np.arange(max_steps + 1) * params.dt
    p_np = VehicleParams(*[float(np.asarray(v)) for v in p])
    return PolicyRollout(t=t, x=x_all, action=a_all, alive=alive_all, laps=laps,
                         v_start=float(v_start), ctx=np.asarray(ctx, float), params=p_np)


def policy_ghost(ro: PolicyRollout, track: Track, lap: int | None = None, id: str = "agent",
                 label: str = "RL agent", badge: str | None = None,
                 meta: dict | None = None, unsealed: bool = False) -> Ghost:
    """The timed lap (default: the last of ``ro.laps``) of a policy rollout."""
    lockbox_guard(track.name, unsealed)
    lap = ro.laps if lap is None else lap
    L = float(track.length)
    dead = np.nonzero(~ro.alive)[0]
    # Drop the terminating step and everything after: like the harness, a crossing only
    # counts while the car is still alive.
    end = int(dead[0]) if len(dead) else len(ro.t)
    ch = state_channels(track, ro.x[:end], ro.params, cmd=ro.action[:end])
    if ch["s"].max() < lap * L:
        why = "terminated" if len(dead) else "ran out of steps"
        raise ValueError(f"lap {lap} not completed ({why} at s = {ch['s'].max():.0f} m)")
    t, ch = lap_window(ro.t[:end], ch, (lap - 1) * L, lap * L)
    ch["t"] = t
    info = {"source": "policy rollout", "protocol": "rolling start at s = 0, n = 0; "
            f"deterministic mean action; lap {lap} of {ro.laps} timed", "lap": lap,
            "v_start_mps": ro.v_start, "car_context": ro.ctx.tolist(),
            "control_rate_hz": round(1.0 / float(ro.t[1] - ro.t[0]), 6)}
    return Ghost(id, label, "policy", float(t[-1]), ch, badge, dict(info, **(meta or {})))


def ghost_from_lap_trace(result, track: Track, lap: int = 2, lap_time: float | None = None,
                         id: str = "agent", label: str = "RL agent",
                         delta_max: float = F1_2025.delta_max, meta: dict | None = None,
                         unsealed: bool = False) -> Ghost:
    """A ghost from what ``timed_laps`` returns: the ``LapResult`` itself, or its ``.trace``
    dict (keys t, s, n, vx, action) together with ``lap_time``.

    ``timed_laps`` keeps its first run's trace only up to the sample *before* the car
    crosses the end of the timed lap (the crossing sample is cut off), so the trace alone
    cannot close that lap. The lap is closed with the harness's own lap time
    (``LapResult.lap_time[0]``, or ``lap_time``): a final sample at s = lap * L and
    t = (start crossing) + lap_time, which must fall within one control step after the
    last sample; its n, vx and course are extrapolated linearly over that part step.
    ``lap_time`` is not needed for a lap the trace does reach (lap 1 of 2).

    The trace has no heading error, side slip or actuator states, so: ``heading`` is the
    course angle of the path, ``speed`` is vx, throttle/brake/steering are the applied
    *commands* (steering scaled by ``delta_max``), and the friction-ellipse channels are
    absent. Prefer :func:`rollout_policy` + :func:`policy_ghost` when the policy is at hand.
    """
    lockbox_guard(track.name, unsealed)
    if hasattr(result, "trace"):            # a LapResult
        if lap_time is None:
            if not bool(np.asarray(result.completed)[0]):
                raise ValueError("run 0 of the LapResult did not complete its timed lap")
            lap_time = float(np.asarray(result.lap_time)[0])
        trace = result.trace
    else:
        trace = result
    L = float(track.length)
    t = np.r_[0.0, np.asarray(trace["t"], float)]
    s = np.r_[0.0, np.asarray(trace["s"], float)]
    n = np.r_[0.0, np.asarray(trace["n"], float)]
    vx = np.asarray(trace["vx"], float)
    vx = np.r_[vx[0], vx]
    # Row k of the trace is the state after the step driven by action k, so the command
    # active from sample k on is action k (with the initial sample prepended above). The
    # simulator clips actions to [-1, 1]; the channels hold what it applied.
    acts = np.clip(np.asarray(trace["action"], float), -1.0, 1.0)
    cmd = np.vstack([acts, acts[-1:]])
    end, start = lap * L, (lap - 1) * L
    closed_by = "trace"
    if s.max() < end:
        if lap_time is None:
            raise ValueError(
                f"the trace stops at s = {s.max():.2f} m, before the lap ends at {end:.2f} m "
                "(timed_laps cuts the crossing sample off): pass the LapResult itself, or "
                "lap_time=LapResult.lap_time[0]")
        k = int(np.nonzero(s >= start)[0][0])
        t_start = t[k] if s[k] == start else (
            t[k - 1] + (start - s[k - 1]) / (s[k] - s[k - 1]) * (t[k] - t[k - 1]))
        t_end = t_start + float(lap_time)
        dt = float(t[-1] - t[-2])
        if not (t[-1] < t_end <= t[-1] + dt * (1.0 + 1e-6)):
            raise ValueError(
                f"lap_time {lap_time:.4f} s puts the lap end at t = {t_end:.4f} s, not within "
                f"one step ({dt:g} s) after the trace's last sample at {t[-1]:.4f} s")
        w = (t_end - t[-1]) / dt

        def extend(a):
            return np.r_[a, a[-1] + w * (a[-1] - a[-2])]

        t, n, vx = np.r_[t, t_end], extend(n), extend(vx)
        s = np.r_[s, end]
        cmd = np.vstack([cmd, cmd[-1:]])
        closed_by = "harness lap time"
    wx, wy, _ = frenet_to_world(track, s, n)
    course = np.unwrap(np.arctan2(np.gradient(wy), np.gradient(wx)))
    pedal = cmd[:, 1]
    ch = {"s": s, "n": n, "x": wx, "y": wy, "heading": course, "speed": vx, "vx": vx,
          "steering": cmd[:, 0] * delta_max,
          "throttle": np.clip(pedal, 0, 1), "brake": np.clip(-pedal, 0, 1),
          "cmd_steer": cmd[:, 0], "cmd_pedal": cmd[:, 1]}
    tl, ch = lap_window(t, ch, start, end)
    ch["t"] = tl
    info = {"source": "timed_laps trace", "lap": lap, "lap_end_from": closed_by,
            "caveats": "heading is the course angle; steering/throttle/brake are commands; "
                       "no friction-ellipse channels"}
    return Ghost(id, label, "policy", float(tl[-1]), ch, None, dict(info, **(meta or {})))


OCP_TIME_RTOL = 1e-4   # per segment; the OCP and this check evaluate the same formula


def check_ocp_time(res, s_st: np.ndarray, t_st: np.ndarray, z_st: np.ndarray) -> None:
    """Every segment of the OCP's time must be the trapezoid of its own dt/ds.

    dt/ds = (1 - n kappa) / (vx cos xi - vy sin xi) at each station, from the states and
    the OCP's station curvature (``res.kappa``), with the shared model's
    ``frenet_rates``. ``s_st``/``t_st``/``z_st`` include the closing station on closed
    circuits, so this also checks the closing segment ``lap_time - res.t[-1]`` that
    ``ocp_ghost`` would otherwise take on trust: an ``OCPResult`` whose ``t`` and
    ``lap_time`` disagree with its states fails here instead of showing up as a silent
    jump in the last segment.
    """
    from apexzero.vehicle.dynamics import frenet_rates

    kap = np.asarray(res.kappa, float)
    if len(z_st) == len(kap) + 1:           # closing station repeats the first
        kap = np.r_[kap, kap[:1]]
    sdot = frenet_rates(np.c_[np.zeros(len(z_st)), z_st], kap, NUMPY)[0]
    if np.any(sdot <= 0):
        raise ValueError("OCP states do not move forward along the track")
    dtds = 1.0 / sdot
    seg = 0.5 * np.diff(s_st) * (dtds[:-1] + dtds[1:])
    err = np.abs(np.diff(t_st) - seg) / seg
    i = int(np.argmax(err))
    if err[i] > OCP_TIME_RTOL:
        raise ValueError(
            f"OCP time is inconsistent with its states: segment {s_st[i]:.1f}-"
            f"{s_st[i + 1]:.1f} m takes {t_st[i + 1] - t_st[i]:.6f} s, its dt/ds gives "
            f"{seg[i]:.6f} s")


def ocp_ghost(res, track: Track, p: VehicleParams = F1_2025, rate_hz: float | None = 20.0,
              id: str = "ocp", label: str = "Optimal lap (s-domain OCP)",
              badge: str | None = None, meta: dict | None = None,
              unsealed: bool = False) -> Ghost:
    """Map an ``OCPResult`` (stations in s) to time.

    The OCP integrates dt/ds with the trapezoidal rule, so ``res.t`` is its own time at
    each station and ``res.lap_time`` its own lap. Closed circuits get the closing
    station (s = L, t = lap_time, the periodic state) appended; then every channel is
    interpolated linearly in t onto a uniform ``rate_hz`` grid that ends exactly at the
    lap time (``rate_hz=None`` keeps the stations). Position and heading are recomputed
    from the interpolated Frenet state, so the ghost stays on the curved line.
    """
    lockbox_guard(track.name, unsealed)
    if res.meta.get("track") is not None:
        lockbox_guard(res.meta["track"], unsealed)
    if res.meta.get("track") not in (None, track.name):
        raise ValueError(f"OCP is for {res.meta['track']!r}, not {track.name!r}")
    L = float(track.length)
    s_st = np.asarray(res.s, float)
    t_st = np.asarray(res.t, float)
    z_st = np.asarray(res.z, float)
    u_st = np.asarray(res.u, float)
    if track.closed:
        h = float(res.meta.get("ds", s_st[1] - s_st[0]))
        if abs(len(s_st) * h - L) > 1e-6 * L:
            raise ValueError(f"OCP grid spans {len(s_st) * h:.3f} m, track is {L:.3f} m")
        s_st, t_st = np.r_[s_st, L], np.r_[t_st, res.lap_time]
        z_st, u_st = np.vstack([z_st, z_st[:1]]), np.vstack([u_st, u_st[:1]])
    elif not np.isclose(t_st[-1], res.lap_time, rtol=0, atol=1e-9):
        raise ValueError(f"OCP time ends at {t_st[-1]} but lap_time is {res.lap_time}")
    if np.any(np.diff(t_st) <= 0) or np.any(np.diff(s_st) <= 0):
        raise ValueError("OCP time or arc length is not strictly increasing")
    check_ocp_time(res, s_st, t_st, z_st)

    ch_st = state_channels(track, np.c_[s_st, z_st], p, cmd=u_st)
    if rate_hz is None:
        tg = t_st
    else:
        tg = np.arange(0.0, t_st[-1], 1.0 / rate_hz)
        if t_st[-1] - tg[-1] <= MIN_GAP:
            tg = tg[:-1]
        tg = np.r_[tg, t_st[-1]]
    ch = {k: np.interp(tg, t_st, v) for k, v in ch_st.items()}
    # Recompute the pose from the interpolated Frenet state (chords would cut corners).
    xi = np.interp(tg, t_st, z_st[:, 1])
    ch["x"], ch["y"], hd = frenet_to_world(track, ch["s"], ch["n"], xi)
    ch["heading"] = np.unwrap(hd)
    ch["t"] = tg
    ch["s"][0], ch["s"][-1] = s_st[0], s_st[-1]
    info = {"source": "s-domain minimum-time OCP", "solver_status": res.status,
            "success": bool(res.success), "ocp_ds_m": float(res.meta.get("ds", np.nan)),
            "corridor_margin_m": float(res.meta.get("margin", np.nan)),
            "time_mapping": "t(s) from the OCP's trapezoidal dt/ds; channels interpolated "
                            "linearly in t" + (f" onto a uniform {rate_hz:g} Hz grid"
                                               if rate_hz else " (stations kept)"),
            "control": "continuous (piecewise linear in s), not the 20 Hz zero-order hold"}
    return Ghost(id, label, "ocp", float(res.lap_time), ch, badge, dict(info, **(meta or {})))


def delta_time(ghost: Ghost, ref: Ghost, s: np.ndarray) -> np.ndarray:
    """Δt(s) = t_ghost(s) - t_ref(s): positive where the ghost is behind."""
    return ghost.time_at(s) - ref.time_at(s)


# --------------------------------------------------------------------------- document


def _squash(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.casefold())


_LOCKBOX = tuple(_squash(name) for name in R_TEST)


def is_lockbox(track_name: str) -> bool:
    """True if ``track_name`` names an R_TEST circuit or a variant of one.

    Deliberately loose, so a spelling cannot slip past it: directory parts and a file
    suffix are dropped, case and separators are ignored, and any name that *starts* with
    a lockbox circuit counts ("./Monza", "monza", "Monza-mirrored", "Mexico City",
    "Spa_reversed", "SpaFrancorchamps"). No R-dev name starts with an R_TEST name."""
    stem = PurePath(str(track_name)).name
    stem = re.sub(r"\.(csv|pkl|json)$", "", stem, flags=re.IGNORECASE)
    return _squash(stem).startswith(_LOCKBOX)


def lockbox_guard(track_name: str, unsealed: bool = False) -> None:
    """Refuse agent or optimum content on the lockbox circuits (R_TEST), including their
    mirrored and reversed variants, unless the lockbox has been formally unsealed.

    Every entry point that rolls out a policy, maps an OCP or assembles a document calls
    it on the track's name, so a caller cannot skip it by using the library directly."""
    if is_lockbox(track_name) and not unsealed:
        raise PermissionError(
            f"{track_name} is a lockbox (R_TEST) circuit: no agent or optimum laps there "
            "until protocol-v1 unseals it")


def _arr(a, decimals: int) -> list:
    a = np.asarray(a, float)
    if not np.all(np.isfinite(a)):
        raise SchemaError("non-finite value in an exported array")
    return np.round(a, decimals).tolist()


def build_document(track: Track, ghosts: Iterable[Ghost], reference: str | None = None,
                   sectors: list[Sector] | None = None, title: str | None = None,
                   notes: Iterable[str] = (), provenance: dict | None = None,
                   delta_ds: float | None = None, unsealed: bool = False) -> dict[str, Any]:
    """Assemble a schema document. With a ``reference`` ghost, every other ghost gets a
    comparison: Δt(s) on the track's grid and the time lost per corner sector.

    Refuses lockbox circuits (by the track's name) unless ``unsealed``."""
    lockbox_guard(track.name, unsealed)
    ghosts = list(ghosts)
    ids = [g.id for g in ghosts]
    if len(set(ids)) != len(ids):
        raise SchemaError(f"duplicate ghost ids {ids}")
    if reference is not None and reference not in ids:
        raise SchemaError(f"reference {reference!r} is not a ghost")
    L = float(track.length)
    sectors = corner_sectors(track) if sectors is None else sectors
    left, right = track.edges()
    doc: dict[str, Any] = {
        "schema": SCHEMA, "version": SCHEMA_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "apexzero.viz.export",
        "title": title or track.name, "notes": list(notes), "provenance": provenance or {},
        "channels": {k: {"unit": c.unit, "description": c.description, "required": c.required}
                     for k, c in CHANNELS.items()},
        "track": {
            "name": track.name, "closed": bool(track.closed), "length_m": L,
            "ds_m": track.ds, "source": track.meta.get("source", ""),
            "s": _arr(track.s, 3), "x": _arr(track.x, 2), "y": _arr(track.y, 2),
            "heading": _arr(track.theta, 4), "kappa": _arr(track.kappa, 6),
            "w_left": _arr(track.w_left, 2), "w_right": _arr(track.w_right, 2),
            "left": {"x": _arr(left[:, 0], 2), "y": _arr(left[:, 1], 2)},
            "right": {"x": _arr(right[:, 0], 2), "y": _arr(right[:, 1], 2)},
            "sectors": [sec.to_dict() for sec in sectors],
        },
        "reference": reference,
        "ghosts": [],
        "comparisons": [],
    }
    for g in ghosts:
        chans = {k: _arr(v, CHANNELS[k].decimals if k in CHANNELS else 4)
                 for k, v in g.channels.items()}
        doc["ghosts"].append({"id": g.id, "label": g.label, "kind": g.kind,
                              "lap_time_s": float(g.lap_time), "badge": g.badge,
                              "meta": _jsonable(g.meta), "channels": chans})
    if reference is not None:
        ref = ghosts[ids.index(reference)]
        grid = np.r_[track.s, L] if track.closed else np.asarray(track.s, float)
        if delta_ds:
            grid = np.r_[np.arange(0.0, L, delta_ds), L]
        for g in ghosts:
            if g.id == reference:
                continue
            d = delta_time(g, ref, grid)
            doc["comparisons"].append({
                "ghost": g.id, "reference": reference, "s": _arr(grid, 3),
                "delta_t": _arr(d, 4), "lap_delta_s": float(g.lap_time - ref.lap_time),
                "sector_delta_s": _arr(sector_deltas(grid, d, sectors, L), 4),
            })
    validate_document(doc)
    return doc


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def _numeric(a, name: str) -> np.ndarray:
    try:
        arr = np.asarray(a, float)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{name}: not a numeric array") from exc
    _check(arr.ndim == 1, f"{name}: expected a 1-D array")
    _check(bool(np.all(np.isfinite(arr))), f"{name}: non-finite values")
    return arr


def validate_document(doc: dict) -> None:
    """Raise :class:`SchemaError` unless ``doc`` is a readable 1.x trajectory document."""
    _check(isinstance(doc, dict), "document must be a JSON object")
    _check(doc.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    ver = str(doc.get("version", ""))
    _check(ver.split(".")[0] == SCHEMA_VERSION.split(".")[0],
           f"unsupported major version {ver!r} (reader is {SCHEMA_VERSION})")
    tr = doc.get("track")
    _check(isinstance(tr, dict), "missing track")
    for k in ("name", "closed", "length_m", "s", "x", "y", "left", "right", "sectors"):
        _check(k in tr, f"track lacks {k!r}")
    L = float(tr["length_m"])
    _check(L > 0, "track length must be positive")
    n_tr = len(_numeric(tr["s"], "track.s"))
    for k in ("x", "y", "heading", "kappa", "w_left", "w_right"):
        if k in tr:
            _check(len(_numeric(tr[k], f"track.{k}")) == n_tr, f"track.{k}: length mismatch")
    for side in ("left", "right"):
        _check(len(_numeric(tr[side]["x"], f"track.{side}.x")) == len(
            _numeric(tr[side]["y"], f"track.{side}.y")), f"track.{side}: x/y length mismatch")
    for sec in tr["sectors"]:
        _check(0 <= sec["s_start"] < sec["s_end"] <= sec["s_start"] + L + 1e-6,
               f"sector {sec.get('name')}: bad extent")

    ghosts = doc.get("ghosts")
    _check(isinstance(ghosts, list) and len(ghosts) > 0, "no ghosts")
    ids = [g.get("id") for g in ghosts]
    _check(len(set(ids)) == len(ids), "ghost ids must be unique")
    for g in ghosts:
        gid = g.get("id")
        _check(g.get("kind") in KINDS, f"ghost {gid}: kind must be one of {KINDS}")
        _check(g.get("badge") in (None, *BADGES), f"ghost {gid}: unknown badge")
        lap = g.get("lap_time_s")
        _check(isinstance(lap, (int, float)) and np.isfinite(lap) and lap > 0,
               f"ghost {gid}: lap_time_s must be a positive number")
        ch = g.get("channels", {})
        for k in REQUIRED:
            _check(k in ch, f"ghost {gid}: missing required channel {k!r}")
        arrs = {k: _numeric(v, f"ghost {gid}.{k}") for k, v in ch.items()}
        n = len(arrs["t"])
        _check(n >= 2, f"ghost {gid}: needs at least two samples")
        _check(all(len(a) == n for a in arrs.values()), f"ghost {gid}: channel lengths differ")
        _check(bool(np.all(np.diff(arrs["t"]) > 0)), f"ghost {gid}: t must strictly increase")
        _check(abs(arrs["t"][0]) < 1e-6, f"ghost {gid}: t must start at 0")
        _check(abs(arrs["t"][-1] - lap) < 1e-3, f"ghost {gid}: t ends at {arrs['t'][-1]}, "
               f"lap_time_s is {lap}")
        _check(bool(np.all(np.diff(arrs["s"]) >= 0)), f"ghost {gid}: s must not decrease")

    ref = doc.get("reference")
    _check(ref is None or ref in ids, "reference must name a ghost")
    for c in doc.get("comparisons", []):
        _check(c.get("ghost") in ids and c.get("reference") in ids,
               "comparison names an unknown ghost")
        _check(len(_numeric(c["s"], "comparison.s")) == len(
            _numeric(c["delta_t"], "comparison.delta_t")), "comparison: length mismatch")
        _check(len(_numeric(c["sector_delta_s"], "comparison.sector_delta_s"))
               == len(tr["sectors"]), "comparison: one sector delta per sector")


def write_document(doc: dict, path: str | Path) -> Path:
    validate_document(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, allow_nan=False, separators=(",", ":")))
    return path


def read_document(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text())
    validate_document(doc)
    return doc


def ghost_arrays(doc: dict, ghost_id: str) -> dict[str, np.ndarray]:
    """One ghost's channels as NumPy arrays."""
    for g in doc["ghosts"]:
        if g["id"] == ghost_id:
            return {k: np.asarray(v, float) for k, v in g["channels"].items()}
    raise KeyError(ghost_id)


def embed_in_viewer(doc: dict, out_path: str | Path, viewer: str | Path = VIEWER_PATH) -> Path:
    """Write a copy of the static viewer with ``doc`` embedded, so the page works when
    opened straight from disk (browsers block fetch() on file:// URLs)."""
    validate_document(doc)
    html = Path(viewer).read_text()
    marker = '<script type="application/json" id="apexzero-data"></script>'
    if marker not in html:
        raise ValueError("viewer has no embedded-data slot")
    payload = json.dumps(doc, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(
        marker, f'<script type="application/json" id="apexzero-data">{payload}</script>'))
    return out
