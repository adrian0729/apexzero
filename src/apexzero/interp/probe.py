"""Open-loop scans (tier O) and on-policy aliasing probes (protocol 2.4 P7, 3.5, 4).

The preview samples the road at 40 fixed distances ahead, so a short tight corner can fall
between samples, and the sampled peak curvature flickers as the car approaches. Two probes
measure what that does to the policy's command, using forward passes only:

- Open-loop scans: the car in a steady straight-line full-throttle state at speed v (n = xi
  = vy = r = delta = 0, pedal = 1, prev_action = (0, 1), accel = (a_x(v) from one simulation
  step, 0)), placed at distance d before s_c0 for d in [d_b^agent(v) + 50, d_max] in 0.25 m
  steps (only states the agent visits). The pedal-command residual after a cubic smoothing
  spline with 6 degrees of freedom gives Pi (power at 1 cycle per preview index over the
  median power in 0.2-2 cycles per index, index k(d) = 40 (d / d_max)^(2/3)) and A (RMS of
  the residual over d in [40, min(300, d_max)]). Descriptive, no thresholds.
- On-policy aliasing A_alias: the states of the agent's deterministic u = 0 run in
  [t_b - 1 s, t_b], each re-evaluated with the corner translated by delta in {0, 0.25,
  0.5, 0.75} x Delta(d_c) (``stimuli.shift_corner``, exact); fit pedal = alpha + beta delta
  per state; A_alias is the RMS of the residuals. The same for Q (about 0) and Qp (P7 null).

Choices (IMPLEMENTATION.md, decisions C14-C18):
- The smoothing spline is the penalised cubic B-spline of R's ``smooth.spline`` (knots at
  R's ``.nknots.smspl(n)`` quantiles, exact integrated squared second-derivative penalty),
  with lambda solved so the hat-matrix trace is 6. With n of 400-1800 scan points the full
  n-knot spline is indistinguishable at 6 dof and needs an n x n eigenproblem.
- Pi uses the Lomb-Scargle periodogram of the residual against k(d) (k is not uniform in d).
- Scans store the raw policy mean (the network's read-out); A_alias uses the command the
  plant receives (the mean clipped to [-1, 1]), so ripple hidden inside saturation is not
  counted as aliasing and agents compare like with like with Q and Qp, whose commands are
  already clipped. Both are switchable.
- A run's prev_action and accel come from the rollout trace when given (det_prev_action,
  det_accel); otherwise they are rebuilt from the trajectory exactly as the env computes
  them (prev_action = the applied command, accel from consecutive states). Only states
  before the corner start are probed.

Note on A_alias as pre-registered: a state within one shift range (0.75 Delta, about 0.1 s
at 310 km/h) of the brake onset switches from throttle to brake as the corner moves, a step
in delta that the per-state line cannot fit. Q (no aliasing) therefore scores about 0.1,
all from the 2-3 states next to t_b; every earlier state fits to < 1e-5. ``AliasResult``
keeps the per-state residuals so analysis can also report the value without them.

Agent forward passes on tier O stimuli are rollouts on protocol stimuli: post-tag only.
Builders test with scripted controls or randomly initialised networks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import eigh

from apexzero.agents.ppo import RunningMeanStd
from apexzero.env import racing
from apexzero.interp.constants import DS, DT, N_KNOTS, O_D_STEP, OBS_CLIP, P7_DELTAS
from apexzero.interp.measures import Trajectory
from apexzero.interp.rollout import Policy
from apexzero.interp.stimuli import CornerSpec
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.sim import TrackArrays

N_PREVIEW = racing.N_PREVIEW
RIPPLE_BAND = (0.2, 2.0)       # cycles per preview index
RIPPLE_DF = 0.01               # frequency step of the reference band
A_RANGE = (40.0, 300.0)        # m, A is the residual RMS over [40, min(300, d_max)]
ALIAS_WINDOW = 1.0             # s, states in [t_b - 1 s, t_b]
_BUCKET = 256                  # forward-pass batches are padded to a multiple of this

ProfileFn = Callable[[CornerSpec, float], Any]   # (spec, shift m) -> Profile-like


@dataclass
class ScanResult:
    """One tier O scan (one stimulus, one speed)."""

    spec: CornerSpec
    v: float                  # m/s
    d: np.ndarray             # [n] m, distance from the car to s_c0
    pedal: np.ndarray         # [n] mean pedal command (raw policy mean unless clipped)
    steer: np.ndarray         # [n] mean steer command
    residual: np.ndarray      # [n] pedal minus the 6-dof smoothing spline
    k_index: np.ndarray       # [n] preview index k(d)
    Pi: float                 # ripple power ratio at 1 cycle per index
    A: float                  # RMS residual over [40, min(300, d_max)]


@dataclass
class AliasResult:
    """The on-policy aliasing probe of one run (P7 measure 2)."""

    t: np.ndarray             # [m] s, times of the probed states
    s: np.ndarray             # [m] m
    d_c: np.ndarray           # [m] m, distance to the corner start
    spacing: np.ndarray       # [m] m, preview spacing Delta(d_c) at the state's speed
    deltas: np.ndarray        # [q] shifts in units of the spacing
    pedal: np.ndarray         # [m, q] mean pedal command on each shifted strip
    residual: np.ndarray      # [m, q] after the per-state fit pedal = alpha + beta delta
    beta: np.ndarray          # [m] fitted slope per unit of spacing
    A_alias: float            # RMS of ``residual`` (nan without states)


# ------------------------------------------------------------------------- geometry


def d_max_of(v) -> np.ndarray:
    """Preview horizon d_max(v) = clip(6 s v, 150, 500) m (racing.preview_offsets)."""
    return np.clip(6.0 * np.asarray(v, float), 150.0, 500.0)


def preview_index(d: np.ndarray, v: float) -> np.ndarray:
    """k(d) = 40 (d / d_max)^(2/3): the continuous preview index at distance d (d_k = d_max
    (k / 40)^1.5, so k(d_k) = k)."""
    return N_PREVIEW * (np.maximum(np.asarray(d, float), 0.0) / d_max_of(v)) ** (2.0 / 3.0)


def preview_spacing(d: float, v: float) -> float:
    """Spacing Delta(d) of consecutive preview points at distance d (m):
    (1.5 d_max / 40) (d / d_max)^(1/3)."""
    dm = d_max_of(v)
    return 1.5 * dm / N_PREVIEW * (np.maximum(np.asarray(d, float), 0.0) / dm) ** (1.0 / 3.0)


# ------------------------------------------------------------------------- states


def _ctx32(ctx=None) -> np.ndarray:
    """The car context as float32: ``rollout.default_ctx()`` unless given."""
    from apexzero.interp.rollout import default_ctx

    return np.asarray(default_ctx() if ctx is None else ctx, np.float32)


_STRAIGHT: dict[str, TrackArrays] = {}


def _straight() -> TrackArrays:
    if "tr" not in _STRAIGHT:
        n = 64
        z = jnp.zeros(n, jnp.float32)
        _STRAIGHT["tr"] = strip_arrays(z, z + 6.0, z + 6.0)
    return _STRAIGHT["tr"]


def full_throttle_accel(v: float, ctx=None) -> float:
    """a_x(v): the env's body-frame accel after one control step at full throttle from the
    straight steady state at v (pedal state 1, prev_action (0, 1)), float32."""
    tr = _straight()
    st = racing.EnvState(
        x=jnp.asarray([10.0, 0, 0, v, 0, 0, 0, 1.0], jnp.float32),
        prev_action=jnp.asarray([0.0, 1.0], jnp.float32), accel=jnp.zeros(2, jnp.float32),
        ctx=jnp.asarray(_ctx32(ctx)), hidden_mu=jnp.float32(1.0), t=jnp.zeros((), jnp.int32),
        slow=jnp.float32(0.0), s0=jnp.float32(10.0))
    _, new, *_ = racing.step_no_reset(st, jnp.asarray([0.0, 1.0], jnp.float32), tr,
                                      racing.EnvParams(window_steps=10**9))
    return float(new.accel[0])


def scan_states(v: float, d: np.ndarray, s_c0: float, ctx=None) -> racing.EnvState:
    """Batched steady full-throttle states at speed v placed d before s_c0 (float32):
    x = [s_c0 - d, 0, 0, v, 0, 0, 0, 1], prev_action (0, 1), accel (a_x(v), 0), t = 0."""
    d = np.asarray(d, float)
    n = len(d)
    x = np.zeros((n, 8), np.float32)
    x[:, 0] = s_c0 - d
    x[:, 3], x[:, 7] = v, 1.0
    ax = full_throttle_accel(v, ctx)
    c = _ctx32(ctx)
    return racing.EnvState(
        x=jnp.asarray(x), prev_action=jnp.tile(jnp.asarray([0.0, 1.0], jnp.float32), (n, 1)),
        accel=jnp.tile(jnp.asarray([ax, 0.0], jnp.float32), (n, 1)),
        ctx=jnp.tile(jnp.asarray(c), (n, 1)), hidden_mu=jnp.ones(n, jnp.float32),
        t=jnp.zeros(n, jnp.int32), slow=jnp.zeros(n, jnp.float32), s0=jnp.asarray(x[:, 0]))


def scan_state(v: float, d: float, s_c0: float, p=None) -> racing.EnvState:
    """The steady full-throttle state at speed v, placed d before s_c0 (float32). ``p`` is an
    optional car context [6] (default ``racing.default_context()``)."""
    return jax.tree.map(lambda a: a[0], scan_states(v, np.array([d]), s_c0, p))


# ------------------------------------------------------------------------- forward passes

_FWD: dict[tuple, Callable] = {}


def _forward(policy: Policy, batched_tracks: bool, normalise: bool) -> Callable:
    """jit(fn(params, obs_rms, tracks, states [B], aux [B]) -> mean [B, 2])."""
    key = (id(policy), batched_tracks, normalise)
    if key not in _FWD:
        def one(params, obs_rms, tr, st, aux):
            p = racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu)
            o = racing.observe(st, tr, p)
            if normalise:
                o = obs_rms.normalize(o, OBS_CLIP)
            return policy(params, o, st, tr, aux)[0]

        axes = (None, None, 0 if batched_tracks else None, 0, 0)
        _FWD[key] = (jax.jit(jax.vmap(one, in_axes=axes)), policy)   # keep policy alive
    return _FWD[key][0]


def _pad(tree, n: int):
    m = -(-max(n, 1) // _BUCKET) * _BUCKET
    return jax.tree.map(lambda a: jnp.concatenate([a, jnp.repeat(a[-1:], m - n, axis=0)])
                        if m > n else a, tree)


def policy_means(policy: Policy, params: Any, obs_rms: RunningMeanStd | None,
                 tracks: TrackArrays, states: racing.EnvState, aux=None,
                 batched_tracks: bool = False) -> np.ndarray:
    """Policy means [B, 2] at states [B] on one track or one track per state; observations
    normalised and clipped with ``obs_rms`` when given (agents), raw otherwise (controls)."""
    n = int(states.x.shape[0])
    aux = jnp.zeros(n, jnp.int32) if aux is None else jnp.asarray(aux, jnp.int32)
    fn = _forward(policy, batched_tracks, obs_rms is not None)
    st, ax = _pad(states, n), _pad(aux, n)
    tr = _pad(tracks, n) if batched_tracks else tracks
    return np.asarray(fn(params, obs_rms, tr, st, ax))[:n]


def _strips(profiles: list[Any]) -> TrackArrays:
    k = jnp.asarray(np.stack([np.asarray(p.kappa) for p in profiles]), jnp.float32)
    wl = jnp.asarray(np.stack([np.asarray(p.w_left) for p in profiles]), jnp.float32)
    wr = jnp.asarray(np.stack([np.asarray(p.w_right) for p in profiles]), jnp.float32)
    return jax.vmap(strip_arrays)(k, wl, wr)


def _default_profile(spec: CornerSpec, delta: float):
    from apexzero.interp.stimuli import corner_profile, shift_corner

    return corner_profile(spec) if delta == 0.0 else shift_corner(spec, delta, N_KNOTS, DS)


# ------------------------------------------------------------------------- detrending


def _nknots(n: int) -> int:
    """R's ``.nknots.smspl``: knots of smooth.spline for n distinct x."""
    if n < 50:
        return n
    a1, a2, a3, a4 = np.log2([50.0, 100.0, 140.0, 200.0])
    if n < 200:
        return int(np.trunc(2 ** (a1 + (a2 - a1) * (n - 50) / 150)))
    if n < 800:
        return int(np.trunc(2 ** (a2 + (a3 - a2) * (n - 200) / 600)))
    if n < 3200:
        return int(np.trunc(2 ** (a3 + (a4 - a3) * (n - 800) / 2400)))
    return int(np.trunc(200 + (n - 3200) ** 0.2))


def _basis(x: np.ndarray, n_knots: int):
    """Cubic B-spline design matrix [n, m] on knots at quantiles of x, and the exact
    penalty Omega[i, j] = int B_i'' B_j'' (Gauss-Legendre, 2 points per interval)."""
    knots = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_knots)))
    t = np.r_[[knots[0]] * 3, knots, [knots[-1]] * 3]
    m = len(t) - 4
    spl = BSpline(t, np.eye(m), 3, extrapolate=False)
    B = np.nan_to_num(spl(x))
    g = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    a, b = knots[:-1], knots[1:]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    xq = (mid[:, None] + half[:, None] * g[None, :]).ravel()
    wq = np.repeat(half, 2)
    D2 = np.nan_to_num(spl.derivative(2)(xq))
    return B, (D2 * wq[:, None]).T @ D2


def detrend(d: np.ndarray, y: np.ndarray, dof: int = 6) -> np.ndarray:
    """Residual of y after a cubic smoothing spline with ``dof`` effective degrees of
    freedom (smoothing chosen so the hat matrix trace equals dof)."""
    d, y = np.asarray(d, float), np.asarray(y, float)
    n = len(d)
    if n <= dof + 2:
        return y - y.mean() if n else y.copy()
    x = (d - d.min()) / max(d.max() - d.min(), 1e-12)
    B, Om = _basis(x, _nknots(len(np.unique(x))))
    BtB = B.T @ B
    ridge = 1e-10 * np.trace(BtB) / len(BtB)
    mu = np.clip(eigh(Om, BtB + ridge * np.eye(len(BtB)), eigvals_only=True), 0.0, None)

    def trace(log_lam):
        return float(np.sum(1.0 / (1.0 + np.exp(log_lam) * mu)))

    lo, hi = -40.0, 40.0          # trace falls from m (lo) to 2 (hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if trace(mid) > dof else (lo, mid)
    lam = np.exp(0.5 * (lo + hi))
    c = np.linalg.solve(BtB + lam * Om + ridge * np.eye(len(BtB)), B.T @ y)
    return y - B @ c


def hat_trace(d: np.ndarray, dof: int = 6) -> float:
    """Trace of the smoothing spline's hat matrix as ``detrend`` fits it (a check)."""
    d = np.asarray(d, float)
    n = len(d)
    H = np.empty((n, n))
    eye = np.eye(n)
    for j in range(n):
        H[:, j] = eye[:, j] - detrend(d, eye[:, j], dof)
    return float(np.trace(H))


# ------------------------------------------------------------------------- ripple


def lomb_scargle(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Classical Lomb-Scargle power of y(t) at frequencies ``freqs`` (cycles per unit t),
    y mean-removed."""
    t, y = np.asarray(t, float), np.asarray(y, float) - np.mean(y)
    w = 2.0 * np.pi * np.asarray(freqs, float)[:, None]
    tau = np.arctan2(np.sum(np.sin(2 * w * t), axis=1),
                     np.sum(np.cos(2 * w * t), axis=1))[:, None] / (2 * w)
    c, s = np.cos(w * (t - tau)), np.sin(w * (t - tau))
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * ((c @ y) ** 2 / np.sum(c * c, axis=1) + (s @ y) ** 2 / np.sum(s * s, axis=1))


def ripple(residual: np.ndarray, k_index: np.ndarray, d: np.ndarray, d_max: float
           ) -> tuple[float, float]:
    """(Pi, A) of a detrended scan: Pi = Lomb-Scargle power of the residual against k at 1
    cycle per index over its median power on 0.2-2 cycles per index; A = RMS of the residual
    over d in [40, min(300, d_max)] (nan where the scan has no points there)."""
    residual, k_index, d = (np.asarray(a, float) for a in (residual, k_index, d))
    band = np.arange(RIPPLE_BAND[0], RIPPLE_BAND[1] + 1e-9, RIPPLE_DF)
    Pi = np.nan
    if len(residual) > 8 and np.ptp(k_index) > 0:
        p_band = lomb_scargle(k_index, residual, band)
        p1 = lomb_scargle(k_index, residual, np.array([1.0]))[0]
        med = np.median(p_band)
        Pi = float(p1 / med) if med > 0 else np.nan
    sel = (d >= A_RANGE[0]) & (d <= min(A_RANGE[1], d_max))
    A = float(np.sqrt(np.mean(residual[sel] ** 2))) if sel.any() else np.nan
    return Pi, A


# ------------------------------------------------------------------------- the probes


def open_loop_scan(policy: Policy, params: Any, obs_rms: RunningMeanStd | None,
                   spec: CornerSpec, v: float, d_min: float, step: float = O_D_STEP,
                   aux: int = 0, profile_fn: ProfileFn | None = None, clip: bool = False,
                   ctx=None,
                   params_fn: Callable[[np.ndarray], tuple[Any, np.ndarray]] | None = None
                   ) -> ScanResult:
    """Forward passes over d in [d_min, d_max(v)] on the stimulus of ``spec`` (vmapped).
    ``profile_fn(spec, shift)`` builds the strip (default ``stimuli.corner_profile``);
    ``aux`` is the control-table row of that strip (Q), 0 for agents; or
    ``params_fn(kappa [1, N]) -> (params, aux [1])`` builds both from the strip."""
    prof = (profile_fn or _default_profile)(spec, 0.0)
    if params_fn is not None:
        params, aux_rows = params_fn(np.asarray(prof.kappa)[None])
        aux = int(np.asarray(aux_rows)[0])
    tr = _strips([prof])
    tr = jax.tree.map(lambda a: a[0], tr)
    dm = float(d_max_of(v))
    d = np.arange(d_min, dm + 1e-9, step)
    states = scan_states(v, d, float(prof.landmarks.s_c0), ctx)
    mean = policy_means(policy, params, obs_rms, tr, states, np.full(len(d), aux))
    if clip:
        mean = np.clip(mean, -1.0, 1.0)
    pedal, steer = mean[:, 1].astype(float), mean[:, 0].astype(float)
    res = detrend(d, pedal)
    k = preview_index(d, v)
    Pi, A = ripple(res, k, d, dm)
    return ScanResult(spec=spec, v=float(v), d=d, pedal=pedal, steer=steer, residual=res,
                      k_index=k, Pi=Pi, A=A)


def _run_aux(run: Trajectory, prev_action, accel, dt: float):
    """prev_action [K, 2] and accel [K, 2] of a run: given, or rebuilt as the env does
    (prev_action = the applied command of the previous step; accel from consecutive
    states, ax = dvx/dt - vy r, ay = dvy/dt + vx r)."""
    K = len(run.x)
    if prev_action is None:
        prev_action = np.zeros((K, 2))
        prev_action[0] = run.prev_action0
        if run.a is not None:
            prev_action[1:] = run.a[:K - 1]
    if accel is None:
        accel = np.zeros((K, 2))
        accel[0] = run.meta.get("accel0", (0.0, 0.0))
        x = run.x
        accel[1:, 0] = (x[1:, 3] - x[:-1, 3]) / dt - x[1:, 4] * x[1:, 5]
        accel[1:, 1] = (x[1:, 4] - x[:-1, 4]) / dt + x[1:, 3] * x[1:, 5]
    return np.asarray(prev_action, float), np.asarray(accel, float)


def aliasing_detail(policy: Policy, params: Any, obs_rms: RunningMeanStd | None,
                    spec: CornerSpec, run: Trajectory, t_b: float,
                    deltas: tuple[float, ...] = P7_DELTAS,
                    profile_fn: ProfileFn | None = None,
                    params_fn: Callable[[np.ndarray], tuple[Any, np.ndarray]] | None = None,
                    prev_action: np.ndarray | None = None, accel: np.ndarray | None = None,
                    clip: bool = True, ctx=None) -> AliasResult:
    """The aliasing probe with its per-state pieces (see ``aliasing_amplitude``)."""
    deltas_a = np.asarray(deltas, float)
    empty = AliasResult(*(np.zeros(0) for _ in range(4)), deltas_a,
                        np.zeros((0, len(deltas_a))), np.zeros((0, len(deltas_a))),
                        np.zeros(0), np.nan)
    if not np.isfinite(t_b):
        return empty
    dt = run.dt or DT
    pa, ac = _run_aux(run, prev_action, accel, dt)
    s_c0 = float(spec.s_c)
    sel = np.nonzero(run.valid & (run.t >= t_b - ALIAS_WINDOW - 1e-9) & (run.t <= t_b + 1e-9)
                     & (run.x[:, 0] < s_c0))[0]
    if not len(sel):
        return empty
    xs = run.x[sel]
    d_c = s_c0 - xs[:, 0]
    spacing = np.array([preview_spacing(dc, max(v, 0.0)) for dc, v in zip(d_c, xs[:, 3])])
    shifts = spacing[:, None] * deltas_a[None, :]                     # [m, q]
    uniq, inv = np.unique(np.round(shifts.ravel(), 9), return_inverse=True)
    build = profile_fn or _default_profile
    profiles = [build(spec, float(sh)) for sh in uniq]
    tracks = _strips(profiles)
    if params_fn is not None:
        prm, aux_rows = params_fn(np.stack([np.asarray(p.kappa) for p in profiles]))
        aux = np.asarray(aux_rows)[inv]
    else:
        prm, aux = params, np.zeros(len(inv), np.int32)
    m, q = shifts.shape
    c = _ctx32(ctx)
    rep = np.repeat(np.arange(m), q)
    states = racing.EnvState(
        x=jnp.asarray(xs[rep], jnp.float32), prev_action=jnp.asarray(pa[sel][rep], jnp.float32),
        accel=jnp.asarray(ac[sel][rep], jnp.float32), ctx=jnp.tile(jnp.asarray(c), (m * q, 1)),
        hidden_mu=jnp.ones(m * q, jnp.float32), t=jnp.asarray(sel[rep], jnp.int32),
        slow=jnp.zeros(m * q, jnp.float32), s0=jnp.full(m * q, float(run.x[0, 0]), jnp.float32))
    tr_each = jax.tree.map(lambda a: a[jnp.asarray(inv)], tracks)
    mean = policy_means(policy, prm, obs_rms, tr_each, states, aux, batched_tracks=True)
    pedal = mean[:, 1].astype(float).reshape(m, q)
    if clip:
        pedal = np.clip(pedal, -1.0, 1.0)
    X = np.stack([np.ones(q), deltas_a], axis=1)
    coef, *_ = np.linalg.lstsq(X, pedal.T, rcond=None)                 # [2, m]
    resid = pedal - (X @ coef).T
    return AliasResult(t=run.t[sel], s=xs[:, 0], d_c=d_c, spacing=spacing, deltas=deltas_a,
                       pedal=pedal, residual=resid, beta=coef[1],
                       A_alias=float(np.sqrt(np.mean(resid**2))))


def aliasing_amplitude(policy: Policy, params: Any, obs_rms: RunningMeanStd | None,
                       spec: CornerSpec, run: Trajectory, t_b: float,
                       deltas: tuple[float, ...] = P7_DELTAS, **kw) -> float:
    """A_alias of one deterministic run (module docstring); states in [t_b - 1 s, t_b]
    before the corner start, each re-evaluated on the shifted strips with its own
    prev_action and accel. ``params_fn(kappa [S, N]) -> (params, aux [S])`` rebuilds
    per-strip parameters (Q's target rows); keywords as ``aliasing_detail``."""
    return aliasing_detail(policy, params, obs_rms, spec, run, t_b, deltas, **kw).A_alias
