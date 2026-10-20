"""Answer keys: the optima every agent is compared with, solved from the same initial state.

Protocol 6 (key types, recipe, tolerance, active sets, infeasibility, validation V1-V11)
and 3.5 (the A2 layout). Two repo solvers need changes, so they are copied here (the repo
files stay untouched, since a training queue imports them):

- ``solve_min_time_pinned``: ``ocp.min_time.solve_min_time`` with an optional pinned start
  (z[:, i_start] = z_start), a ``speed_profile(..., v_start=v0)`` guess, optional upper bounds
  on vx at given nodes (V8) and an IPOPT CPU cap. With the extras unset it builds the same NLP
  as the repo solver, so the two agree to round-off (V10 asks for 1e-8). The repo solver
  leaves the start free, so on a hairpin it chose 120 m/s at n = -4.75 for itself.
- ``AnchoredSolver``: a ``WindowSolver`` subclass whose ``solve`` copy takes ``a0_fixed``
  (lbx = ubx for a_0). An anchored key is solved from x_{L-1} with a_0 fixed to the agent's
  a_{L-1}, so the free problem starts at x_L with a_{-1} equal to the agent's real previous
  action and the step-0 penalty is a constant. It reuses the compiled solver libraries.
  With ``a0_fixed=None`` it is the parent's code line for line (V10: equal to 1e-10).

The s-OCP strip is cut at s0 - 20 m and resampled on nodes s0 - 20 + 2 i from the full strip's
C2 spline, so a node lands on s0 for any phase offset and every node's kappa and widths equal
the full strip's (asserted to 1e-9). Its result is returned from the pinned node on, with s in
strip coordinates and t = 0 at the pin.

Recipe for agent-free keys (6.2): pinned s-OCP to s_c1 + 600 m -> ``min_time_guess`` extended
with a straight full-throttle (lane-keeping) tail -> continuation over the stages {200, 400,
800} up to W (a stage ending before s_c1 + 300 m on the s-OCP's time line is skipped) from up
to three starts (the s-OCP guess, the neighbouring larger-R key re-timed in s, the agent-free
Q controller's rollout) -> highest J among the starts that pass ``verify_key`` (the
``best_verified`` rule; ``verify_window`` replayed under the key's own reward weights). IPOPT
max_iter 800 per stage and 400 CPU-s per start (each IPOPT call gets the start's remaining
budget, rounded up to 50 s, and a start that runs over counts as capped); a capped start
counts as failed; if every start fails, one retry at max_iter 3000, else "unsolved" (or
"infeasible (bound)" when the point-mass certificate says so). No agent rollout is ever a
start of an agent-free key.

The backed-off key needs no recompile: ``WindowSolver.parameters`` passes ``p.alpha_max`` as the
last solver parameter (the path constraint is alpha / alpha_max, its bound 1 - slip_safety /
alpha_max is computed per solve), and the widths are data (coefficient windows), so narrowing
the corridor by 0.35 m per side is a different Track for the same compiled library. The lock
(delta_max) is a solve-time bound too and stays at the car's value. Only ``WindowSpec`` fields
(gamma, action_cost, steps, ...) are compiled in, so the min-time-in-actuation key
(gamma = 1, action_cost = 1e-4) needs one compile per window length.

Keys run in float64 (``replay`` switches x64 on globally): never in a rollout process. Every
key is solved for the car the rollouts drive: ``racing.default_context()`` as exact float32
values (``key_ctx``), hidden_mu = 1. Measures come from ``measures.response_metrics`` through
``key_metrics`` / ``socp_metrics`` (one definition for agents and keys).

Cache: one record per key under ``outputs/psychophysics/keys/<kind>/<key_id>.{json,npz}``,
``key_id`` a hash of (stimulus samples, x0, W, car parameters, spec, kind, extras); alias files
``<kind>/_alias/<tag>`` map a cell (and a mirror pair) to its key for ``key_for``.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import os
import tempfile
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import casadi as ca
import numpy as np

from apexzero.interp import io
from apexzero.interp import measures as M
from apexzero.interp.constants import (
    A1_D,
    A2_D_MARGIN,
    A2_D_STEP,
    A2_FLAT_D,
    A2_VB_FRACTION,
    A2_VB_TOL,
    ACTIVE_CORRIDOR,
    ACTIVE_SLIP,
    ACTIVE_STEER,
    ACTIVE_WINDOW,
    AGENT_BEATS_KEY_RTOL,
    ANCHOR_LEAD,
    BACKOFF_ALPHA_MAX,
    BACKOFF_SHRINK,
    BAND_EXPECTED,
    BAND_LIMITS,
    BAND_ROUND,
    BOUND_FACTOR,
    BUDGET_P0_CPU_H,
    BUDGET_TOTAL_CPU_H,
    CONT_STAGES,
    DS,
    DT,
    IPOPT_STAGE_MAX_ITER,
    KEY_WORKERS,
    LOGGED_RTOL,
    MIN_ROLLOUT_SPS,
    MULTI_BASIN_FLOORS,
    NEAR_EQUAL_RTOL,
    R_180_MAX,
    R_BRAKING,
    REPLAY_RTOL,
    RETRY_MAX_ITER,
    SOCP_AFTER_EXIT,
    SOCP_BACK,
    SOCP_LEAD_IN,
    STAGE_SKIP_MARGIN,
    START_DISAGREE_J,
    START_DISAGREE_V_KPH,
    START_MAX_CPU,
    V8_EXCESS,
    V8_TIME_COST,
    V_MIN_MARGIN,
    VIABILITY_W,
    W_LONG,
    W_RULE_TIME,
    W_SHORT,
    kph,
)
from apexzero.interp.measures import ModelTables, NoReturn, Trajectory
from apexzero.interp.stimuli import Cell, Landmarks, StimulusBank
from apexzero.ocp.min_time import NU, NZ, OCPResult, resample, slip_angles, spatial_dynamics
from apexzero.ocp.replay import verify_window
from apexzero.ocp.window import (
    MAX_TRAVEL,
    NA,
    NX,
    WindowResult,
    WindowSolver,
    WindowSpec,
    _window_centres,
    block_offset_range,
    coefficient_windows,
    min_time_guess,
    solve_continuation,
    track_scalars,
)
from apexzero.oracle.qss import GG, speed_profile
from apexzero.tracks.spline import packed_coeffs, spline_np
from apexzero.tracks.track import Track
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import Ops
from apexzero.vehicle.params import F1_2025, VehicleParams, with_context
from apexzero.viz.export import lockbox_guard

# The vehicle model on plain NumPy (float64; no JAX config side effects), for slip angles.
NUMPY_OPS = Ops(name="numpy", sin=np.sin, cos=np.cos, tan=np.tan, atan=np.arctan, tanh=np.tanh,
                sqrt=np.sqrt, stack=lambda xs: np.stack(np.broadcast_arrays(*xs), axis=-1),
                index=lambda x, i: x[..., i], where=np.where)

ACTUATION_SPEC = WindowSpec(gamma=1.0, action_cost=1e-4)   # min time in the agent's actuation
START_MEASURES = ("d_b", "v_e", "v_min", "s_vmin", "line_apex")   # multi-basin measures (6.3)
CAPPED_STATUSES = ("Maximum_Iterations_Exceeded", "Maximum_CpuTime_Exceeded")
CPU_BUCKET = 50.0          # s: per-call IPOPT CPU caps are the start's remainder rounded up to this
Q_CV_DEFAULT = 0.95        # Q's speed factor for start (3) unless the tuned c_v is passed
PIN_TOL = 1e-9             # pinned s-OCP: kappa / widths / pinned state (protocol 6.2, V10)

# ------------------------------------------------------------------------- records


@dataclass
class StartRecord:
    """One start of one key solve (protocol 6.2, 6.3)."""

    name: str                 # 'socp' | 'neighbour:<key_id>' | 'Q' | 'warm:<key_id>'
                              # | 'retry' | 'agent:<agent>' (6.3 re-solve) | 'nominal:<key_id>'
    J: float                  # nan if the start failed
    success: bool             # last stage's WindowResult.success (and not capped)
    verified: bool            # verify_key (replay.verify_window, the key's own reward)
    capped: bool              # hit max_iter or the CPU cap: counts as failed
    status: str
    iterations: int
    cpu_seconds: float
    chain: list = field(default_factory=list)       # [(steps, status), ...]
    measures: dict = field(default_factory=dict)    # d_b, v_e, v_min, s_vmin, line_apex, v_b, ...
    rounds: list = field(default_factory=list)      # IPOPT rounds of every stage (solver stats)


@dataclass
class KeyRecord:
    """A TO-OCP key (every kind but 'socp'). Arrays go to the .npz, the rest to the .json."""

    key_id: str
    kind: str                 # 'nominal' | 'backoff' | 'phase' | 'null' | 'anchored'
                              # | 'viability' | 'actuation'
    status: str               # 'verified' | 'unsolved' | 'infeasible_bound' | 'skipped'
    stim_id: int
    stim_name: str
    cell_id: int
    phase: int                # k of u_k (0 for nominal)
    direction: int            # the stimulus direction (+1 left, -1 right, 0 straight); a key
                              # mirrored on the fly carries the flag 'mirrored'
    x0: np.ndarray            # [8] float64 (the float32 harness state, cast)
    W: int                    # steps
    spec: dict                # WindowSpec fields
    alpha_max: float          # rad (backed-off: 8.5 deg)
    shrink: float             # m, corridor narrowing per side (backed-off: 0.35)
    J: float = np.nan         # discounted return of the selected solution
    x: np.ndarray | None = None    # [W+1, 8]
    a: np.ndarray | None = None    # [W, 2]
    replay: dict = field(default_factory=dict)      # res.meta['replay'] of the selection
    starts: list[StartRecord] = field(default_factory=list)
    near_equal: list[str] = field(default_factory=list)   # start names in N
    logged: list[str] = field(default_factory=list)       # starts within (1e-4, 1e-3]
    spread: dict = field(default_factory=dict)      # measure -> max - min across N
    multi_basin: bool = False
    tau: dict = field(default_factory=dict)         # measure -> max(floor, spread)
    measures: dict = field(default_factory=dict)    # ResponseMetrics fields of the selection
    active: dict = field(default_factory=dict)      # slip_/lock_/corridor_limited flags, counts
    active_steps: np.ndarray | None = None          # [W+1, 5] bool (ACTIVE_COLUMNS)
    lam_g: np.ndarray | None = None
    lam_x: np.ndarray | None = None
    agent: str = ""           # anchored / viability: whose state
    anchor: dict = field(default_factory=dict)      # {'step': L, 's_L', 'a_prev', 'snapshot', ...}
    socp_id: str = ""         # pinned s-OCP that seeded the guess
    cpu_seconds: float = 0.0  # all starts (and their replays)
    flags: list[str] = field(default_factory=list)
    code_hash: str = ""
    created: str = ""
    pair_id: int = -1         # the cell's mirror pair (alias for the mirror shortcut)
    labels: dict = field(default_factory=dict)      # R, theta_deg, width, approach, v0, D, ...
    stats: dict = field(default_factory=dict)       # selection's solver statistics
    bound: dict = field(default_factory=dict)       # infeasibility certificate, when computed
    measures_hash: str = ""   # ``measures_hash()`` when ``measures`` was computed (staleness)


ACTIVE_COLUMNS = ("slip_f", "slip_r", "lock", "corridor_left", "corridor_right")


@dataclass
class SocpKey:
    """A pinned s-OCP solution (kind 'socp'), s in strip coordinates, t = 0 at the pin."""

    key_id: str
    stim_id: int
    cell_id: int
    phase: int
    x0: np.ndarray            # [8] float64
    s_end: float              # m
    result: OCPResult         # nodes from the pin on (the 20 m lead-in dropped)
    pin_check: dict = field(default_factory=dict)   # max |dkappa|, |dw| vs the full strip, pin
    vx_bounds: dict = field(default_factory=dict)   # {s: vx_max} (V8, layout cruise)
    measures: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    measures_hash: str = ""   # ``measures_hash()`` when ``measures`` was computed


# ------------------------------------------------------------------------- helpers


def _x64() -> None:
    """Keys run in float64 (as ``replay`` does): switch x64 on in this (key) process."""
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def key_ctx(p: VehicleParams = F1_2025) -> np.ndarray:
    """The car context of every key: ``racing.default_context()`` as the exact float32 values
    the rollouts use (protocol 1.4), cast to float64. alpha_max is not part of it."""
    cla = p.cla_f + p.cla_r
    return np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla],
                    np.float32).astype(np.float64)


def _drive_params(p: VehicleParams) -> VehicleParams:
    """The car as the env and the TO-OCP drive it (context applied to the base car)."""
    c = key_ctx(p)
    return with_context(p, *(float(v) for v in c))


def _guard_track(track: Track) -> None:
    lockbox_guard(track.name)
    for k in ("source", "cut_from", "source_track"):
        v = track.meta.get(k) if isinstance(track.meta, dict) else None
        if isinstance(v, str):
            lockbox_guard(v)


def _guard(stim) -> None:
    """Lockbox: the stimulus name and, for real roads, the source circuit (protocol 0)."""
    lockbox_guard(stim.name)
    src = getattr(stim, "source", {}) or {}
    if src.get("track"):
        lockbox_guard(str(src["track"]))


def _spline_at(values: np.ndarray, track: Track, s: np.ndarray) -> np.ndarray:
    c = packed_coeffs(np.asarray(values, float), track.closed, track.n_points)
    return spline_np(c, np.asarray(s, float), track.ds, track.n_points, track.closed)


def _cell_x0(cell: Cell, phase: int) -> np.ndarray:
    """The harness's float32 initial state at phase k, cast (keys start bit-identically)."""
    return np.asarray(cell.initial_state(phase), np.float32).astype(np.float64)


@functools.cache
def _code_hash() -> str:
    return io.code_hash()


MEASURES_FILES = ("src/apexzero/interp/measures.py", "src/apexzero/interp/stimuli.py",
                  "src/apexzero/interp/constants.py")


@functools.cache
def measures_hash() -> str:
    """Hash of the code a key's cached measures depend on (the measures, the landmarks and
    the constants). A record whose ``measures_hash`` differs has stale measures:
    ``refresh_measures`` recomputes them from its stored solution."""
    return io.code_hash(MEASURES_FILES)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _monotone_s(s: np.ndarray) -> np.ndarray:
    """Strictly increasing copy of an arc-length trace (for interpolation in s)."""
    s = np.maximum.accumulate(np.asarray(s, float))
    return s + 1e-9 * np.arange(len(s))


# ------------------------------------------------------------------------- s-OCP


def pinned_strip(track: Track, s0: float, s_end: float, back: float = SOCP_BACK,
                 ds: float = DS) -> Track:
    """The strip resampled on nodes s0 - back + i ds up to s_end, from the full strip's
    C2 spline (``tracks.spline.spline_np``), so the node at s0 exists for any s0 and every
    node's kappa and widths are the full strip's (checked to 1e-9 by ``socp_from_state``).
    Open Track; s starts at 0 (``meta['s_offset']`` = s0 - back, ``meta['i_pin']`` the s0
    node). Near the strip start the lead-in shrinks to a multiple of ds that fits."""
    s0 = float(s0)
    back = float(min(back, ds * np.floor(max(s0, 0.0) / ds + 1e-9)))
    s_start = s0 - back
    if not track.closed:
        s_end = min(float(s_end), track.length)
    n = int(np.floor((s_end - s_start) / ds + 1e-9)) + 1
    i_pin = round(back / ds)
    if n < i_pin + 3:
        raise ValueError(f"pinned strip from s0 = {s0:.1f} to {s_end:.1f} m is too short")
    s_abs = s_start + np.arange(n) * ds
    k, wl, wr = (_spline_at(a, track, s_abs) for a in (track.kappa, track.w_left, track.w_right))
    from apexzero.audit.qss_distance import strip_xy

    theta, x, y = strip_xy(k, ds)
    meta = {"s_offset": s_start, "i_pin": i_pin, "s0": s0, "source": track.name, "ds": ds}
    return Track(name=f"{track.name}@pin{s0:.3f}", s=np.arange(n) * ds, x=x, y=y, theta=theta,
                 kappa=k, w_left=wl, w_right=wr, closed=False, meta=meta)


def _initial_guess(kappa, h, p: VehicleParams, closed: bool, v_start: float | None = None):
    """``min_time.initial_guess`` with the QSS pass seeded by ``v_start`` (None: identical)."""
    v = speed_profile(kappa, h, GG.from_params(p), closed=closed, v_start=v_start)
    v = np.clip(0.97 * v, 8.0, None)
    z0 = np.zeros((len(kappa), NZ))
    z0[:, 2] = v
    z0[:, 4] = kappa * v
    z0[:, 5] = np.clip(np.arctan(p.lf + p.lr) * 0 + (p.lf + p.lr) * kappa, -p.delta_max, p.delta_max)
    dv = np.gradient(v) / h * v  # dv/dt
    z0[:, 6] = np.clip(np.where(dv > 0, 0.6, dv / 40.0), -1, 1)
    u0 = np.stack([z0[:, 5] / p.delta_max, z0[:, 6]], axis=1)
    return z0, u0


def solve_min_time_pinned(track: Track, p: VehicleParams, ds: float = DS, margin: float = 1.0,
                          init: tuple[np.ndarray, np.ndarray] | None = None,
                          max_iter: int = 3000, tol: float = 1e-6, print_level: int = 0,
                          u_rate_weight: float = 1e-4, z_start: np.ndarray | None = None,
                          i_start: int = 0, v_start: float | None = None,
                          vx_upper: dict[int, float] | None = None,
                          max_cpu_time: float | None = None,
                          vx_lin: list[tuple[int, float, float]] | None = None) -> OCPResult:
    """Copy of ``ocp.min_time.solve_min_time`` (same signature first) with: ``z_start`` [7]
    pinned at node ``i_start`` (lbZ = ubZ), ``v_start`` seeding the QSS guess, ``vx_upper``
    {node: vx_max} bounds, ``vx_lin`` [(i, w, vx_max)] bounds on the speed interpolated
    between nodes i and i + 1, (1 - w) vx_i + w vx_{i+1} <= vx_max (a point between nodes,
    V8's s_c0), and IPOPT ``max_cpu_time``. With the extras unset it builds the repo
    solver's NLP (V10). Protocol 6.2."""
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
    if vx_upper:                                   # V8 / layout: vx <= vx_max at given nodes
        for i, v in vx_upper.items():
            ubZ[2, int(i)] = min(ubZ[2, int(i)], float(v))
    if z_start is not None:                        # the pin: the approach state itself
        z_start = np.asarray(z_start, float).reshape(NZ)
        lbZ[:, i_start] = z_start
        ubZ[:, i_start] = z_start
    lbx = np.concatenate([lbZ.T.reshape(-1), -np.ones(NU * n)])
    ubx = np.concatenate([ubZ.T.reshape(-1), np.ones(NU * n)])

    if init is None:
        z0, u0 = _initial_guess(kap, h, p, closed, v_start)
    else:
        z0, u0 = init
    x0 = np.concatenate([z0.reshape(-1), u0.reshape(-1)])
    x0 = np.clip(x0, lbx, ubx)

    lin = [(1.0 - float(w)) * Z[2, int(i)] + float(w) * Z[2, int(i) + 1]
           for i, w, _ in (vx_lin or [])]
    g_all = ca.vertcat(ca.vec(defects), ca.vec(alpha), *lin)
    n_def = defects.numel()
    lin_ub = [float(v) for _, _, v in (vx_lin or [])]
    lbg = np.concatenate([np.zeros(n_def), -p.alpha_max * np.ones(2 * n), np.full(len(lin), -np.inf)])
    ubg = np.concatenate([np.zeros(n_def), p.alpha_max * np.ones(2 * n), np.asarray(lin_ub, float)])
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
    if max_cpu_time is not None:
        opts["ipopt.max_cpu_time"] = float(max_cpu_time)
    solver = ca.nlpsol("ocp", "ipopt", nlp, opts)
    t0, c0 = time.perf_counter(), time.process_time()
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    elapsed, cpu = time.perf_counter() - t0, time.process_time() - c0
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
        meta={"ds": h, "margin": margin, "u_rate_weight": u_rate_weight, "track": track.name,
              "z_start": None if z_start is None else z_start.tolist(), "i_start": int(i_start),
              "v_start": v_start, "vx_upper": {int(k): float(v) for k, v in (vx_upper or {}).items()},
              "vx_lin": [[int(i), float(w), float(v)] for i, w, v in (vx_lin or [])],
              "max_cpu_time": max_cpu_time, "cpu_seconds": cpu},
    )


def socp_end(landmarks: Landmarks) -> float:
    """Where the pinned s-OCP ends: s_c1 + 600 m (protocol 6.2)."""
    return landmarks.s_c1 + SOCP_AFTER_EXIT


def socp_from_state(track: Track, x0: np.ndarray, s_end: float, p: VehicleParams = F1_2025,
                    vx_upper: dict[float, float] | None = None,
                    max_cpu_time: float | None = None, *, stim_id: int = -1, cell_id: int = -1,
                    phase: int = 0, key_id: str = "", max_iter: int = 3000) -> SocpKey:
    """Pinned s-OCP "from this approach state": resampled with ``pinned_strip`` from x0[0]
    (lead-in ``SOCP_LEAD_IN`` = 0: the s0 node is node 0, z[:, 0] = x0[1:], protocol 6.2),
    ``v_start = x0[3]`` seeding the guess there, solved to ``s_end``; returns nodes from the
    pin on with s in strip coordinates and t = 0 there. A free, time-minimised lead-in
    before the pin would share the pin's control through the last defect and the rate
    penalty, so it would leak into the pinned solution; it is not part of the NLP.
    ``vx_upper`` maps strip s to a vx bound: on a node it bounds that node; between nodes
    it bounds the speed interpolated linearly in s (V8's s_c0 sits off the 2 m grid when
    D is not a multiple of 2 m). Raises if kappa / widths differ from the full strip's by
    more than 1e-9, or a converged solution leaves the pin by more than 1e-9."""
    _guard_track(track)
    x0 = np.asarray(x0, float).reshape(NX)
    cut = pinned_strip(track, x0[0], s_end, back=SOCP_LEAD_IN)
    i_pin, off = int(cut.meta["i_pin"]), float(cut.meta["s_offset"])
    nodes: dict[int, float] = {}
    lin: list[tuple[int, float, float]] = []
    for s, v in (vx_upper or {}).items():
        xi = (float(s) - off) / DS
        i = int(np.floor(xi + 1e-9))
        w = xi - i
        if abs(w) <= 1e-9 or abs(w - 1.0) <= 1e-9:
            i = round(xi)
            if 0 <= i < cut.n_points:
                nodes[i] = min(nodes.get(i, np.inf), float(v))
        elif 0 <= i < cut.n_points - 1:
            lin.append((i, float(w), float(v)))
    res = solve_min_time_pinned(cut, p, ds=DS, z_start=x0[1:], i_start=i_pin, v_start=float(x0[3]),
                                vx_upper=nodes or None, max_cpu_time=max_cpu_time,
                                max_iter=max_iter, vx_lin=lin or None)
    s_abs = off + res.s
    full = [_spline_at(a, track, s_abs) for a in (track.kappa, track.w_left, track.w_right)]
    check = {"kappa": float(np.max(np.abs(res.kappa - full[0]))),
             "w_left": float(np.max(np.abs(res.w_left - full[1]))),
             "w_right": float(np.max(np.abs(res.w_right - full[2]))),
             "z_pin": float(np.max(np.abs(res.z[i_pin] - x0[1:]))),
             "s_pin": float(abs(s_abs[i_pin] - x0[0]))}
    if max(check["kappa"], check["w_left"], check["w_right"], check["s_pin"]) > PIN_TOL:
        raise AssertionError(f"pinned s-OCP strip differs from the full strip: {check}")
    if res.success and check["z_pin"] > PIN_TOL:
        raise AssertionError(f"pinned s-OCP left its pin: {check}")
    sl = slice(i_pin, None)
    t = res.t[sl] - res.t[i_pin]
    out = OCPResult(s=s_abs[sl], z=res.z[sl], u=res.u[sl], t=t, lap_time=float(t[-1]),
                    success=res.success, status=res.status, iterations=res.iterations,
                    solve_seconds=res.solve_seconds, kappa=res.kappa[sl], w_left=res.w_left[sl],
                    w_right=res.w_right[sl], dual_left=res.dual_left[sl],
                    dual_right=res.dual_right[sl],
                    meta=dict(res.meta, s_offset=off, i_pin=i_pin, track=track.name,
                              s0=float(x0[0])))
    return SocpKey(key_id=key_id, stim_id=stim_id, cell_id=cell_id, phase=phase, x0=x0,
                   s_end=float(s_end), result=out, pin_check=check,
                   vx_bounds={float(k): float(v) for k, v in (vx_upper or {}).items()},
                   flags=[] if res.success else [f"socp {res.status}"])


# ------------------------------------------------------------------------- TO-OCP solvers


class AnchoredSolver(WindowSolver):
    """``WindowSolver`` whose ``solve`` copy can fix the first action (protocol 6.1)."""

    def solve(self, track: Track, x0: np.ndarray, init: tuple[np.ndarray, np.ndarray],
              ctx: np.ndarray | None = None, p: VehicleParams = F1_2025,
              max_rounds: int = 6, lam0: tuple[np.ndarray, np.ndarray] | None = None,
              hidden_mu: float = 1.0, refine: bool = True,
              a0_fixed: np.ndarray | None = None) -> WindowResult:
        """``WindowSolver.solve`` with lbx = ubx = a0_fixed for a_0 when given (the actions
        are unscaled variables, so the bound is the action itself). With ``a0_fixed=None``
        this is the parent's code line for line (V10)."""
        spec, T = self.spec, self.T
        if abs(track.ds - spec.ds) > 0.05:
            raise ValueError(f"track ds {track.ds} too far from the nominal {spec.ds} the window is sized for")
        if lam0 is not None and not self._warm_ipopt:
            raise ValueError("lam0 has no effect unless the solver has warm_ipopt options "
                             "(e.g. {'warm_start_init_point': 'yes'})")
        a0 = None
        if a0_fixed is not None:
            a0 = np.asarray(a0_fixed, float).reshape(NA)
            if np.any(np.abs(a0) > 1.0):
                raise ValueError(f"a0_fixed {a0} outside [-1, 1]")
        if ctx is None:
            cla = p.cla_f + p.cla_r
            ctx = np.array([p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla])
        ctx = np.array(ctx, float)
        ctx_drive = ctx.copy()
        ctx_drive[4] *= hidden_mu
        xs, acts = (np.array(v, float) for v in init)
        xs[0] = x0
        if a0 is not None:
            acts[0] = a0
        lam_g, lam_x = lam0 if lam0 is not None else (None, None)
        R = spec.radius
        wall, cpu, iters, log = 0.0, 0.0, 0, []
        for rnd in range(1, max_rounds + 1):
            sigma = _window_centres(xs[:, 0])
            P = self.parameters(track, sigma[:-1], ctx_drive, p)
            o, d = np.zeros(self.nw), self.var_scale
            if self.scaled:
                o = self.offset(sigma)
                P = np.concatenate([P, o])
            lo = np.maximum(sigma[1:], sigma[:-1]) - R
            hi = np.minimum(sigma[1:], sigma[:-1] + MAX_TRAVEL - 0.5) + R
            assert np.all(lo < hi), "empty window bounds"
            lbX = np.tile(np.array([-np.inf, -np.inf, -0.5 * np.pi, 2.0, -25.0, -5.0, -p.delta_max, -1.0])[:, None], (1, T + 1))
            ubX = np.tile(np.array([np.inf, np.inf, 0.5 * np.pi, 120.0, 25.0, 5.0, p.delta_max, 1.0])[:, None], (1, T + 1))
            lbX[0, 1:], ubX[0, 1:] = lo, hi
            lbX[:, 0] = ubX[:, 0] = x0
            lbx = (np.concatenate([lbX.T.reshape(-1), -np.ones(NA * T)]) - o) / d
            ubx = (np.concatenate([ubX.T.reshape(-1), np.ones(NA * T)]) - o) / d
            if a0 is not None:
                ia = NX * (T + 1)
                lbx[ia: ia + NA] = a0
                ubx[ia: ia + NA] = a0
            slip = 1.0 - spec.slip_safety / p.alpha_max        # |alpha| / alpha_max bound
            lbg = np.concatenate([np.zeros(NX * T), np.full(2 * T, -np.inf), np.full(2 * T, -slip)])
            ubg = np.concatenate([np.zeros(NX * T), np.zeros(2 * T), np.full(2 * T, slip)])
            w0 = np.clip((np.concatenate([xs.reshape(-1), acts.reshape(-1)]) - o) / d, lbx, ubx)
            kw = dict(x0=w0, p=P, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            solver = self.solver
            if lam_g is not None:
                kw.update(lam_g0=lam_g, lam_x0=lam_x)
                solver = self.warm_solver()
            t0, c0 = time.perf_counter(), time.process_time()
            sol = solver(**kw)
            wall += time.perf_counter() - t0
            cpu += time.process_time() - c0
            st = solver.stats()
            iters += int(st["iter_count"])
            log.append({"round": rnd, "iterations": int(st["iter_count"]), "status": st["return_status"],
                        **{k: v for k, v in st.items() if k.startswith("t_proc_nlp")}})
            wv = np.asarray(sol["x"]).ravel() * d + o
            xs = wv[: NX * (T + 1)].reshape(T + 1, NX)
            acts = wv[NX * (T + 1):].reshape(T, NA)
            lam_g, lam_x = np.asarray(sol["lam_g"]).ravel(), np.asarray(sol["lam_x"]).ravel()
            at_bound = bool(np.any(xs[1:, 0] < lo + 1.0) or np.any(xs[1:, 0] > hi - 1.0))
            if not at_bound:
                break
        status = str(st["return_status"])
        if refine and status == "Solve_Succeeded" and not at_bound:
            rs = self.refine_solver()
            kw.update(x0=np.asarray(sol["x"]).ravel(), lam_g0=lam_g, lam_x0=lam_x)
            t0, c0 = time.perf_counter(), time.process_time()
            sol_r = rs(**kw)
            wall += time.perf_counter() - t0
            cpu += time.process_time() - c0
            st_r = rs.stats()
            iters += int(st_r["iter_count"])
            log.append({"round": "refine", "iterations": int(st_r["iter_count"]), "status": st_r["return_status"]})
            if st_r["return_status"] == "Solve_Succeeded":     # else keep the unrefined optimum
                sol = sol_r
                wv = np.asarray(sol["x"]).ravel() * d + o
                xs = wv[: NX * (T + 1)].reshape(T + 1, NX)
                acts = wv[NX * (T + 1):].reshape(T, NA)
                lam_g, lam_x = np.asarray(sol["lam_g"]).ravel(), np.asarray(sol["lam_x"]).ravel()
        (Ck, _, _), j0 = coefficient_windows(track, sigma[:-1], spec)
        q_t = np.tile(np.array(track_scalars(track))[:, None], (1, T))
        o_lo, o_hi = self.block_offsets(xs[:T].T, acts.T, Ck, j0[None, :], q_t, np.tile(ctx_drive[:, None], (1, T)))
        off_lo, off_hi = float(np.min(np.asarray(o_lo))), float(np.max(np.asarray(o_hi)))
        rng = block_offset_range(spec.block)
        block_ok = rng[0] <= off_lo and off_hi <= rng[1]
        flags = [f for f, bad in (("window bound active", at_bound), ("kappa block out of range", not block_ok))
                 if bad]
        meta = {"cpu_seconds": cpu, "rounds": log, "lam": (lam_g, lam_x),
                "track": track.name, "ctx": ctx.tolist(), "hidden_mu": float(hidden_mu),
                "block_offsets": (off_lo, off_hi), "spec": spec.__dict__,
                "compiled": self.compiled}
        if a0 is not None:
            meta["a0_fixed"] = a0.tolist()
        return WindowResult(x=xs, a=acts, value=float(-sol["f"]),
                            success=status == "Solve_Succeeded" and not flags,
                            status=status + "".join(f" ({f})" for f in flags),
                            iterations=iters, solve_seconds=wall, rounds=rnd, meta=meta)


_BASE_SOLVERS: dict[tuple, AnchoredSolver] = {}
_SOLVERS: dict[tuple, AnchoredSolver] = {}


def _with_caps(base: AnchoredSolver, max_iter: int, max_cpu_time: float | None) -> AnchoredSolver:
    """A solver sharing ``base``'s compiled functions with other IPOPT caps (a new nlpsol
    on the same external functions: no codegen, no compile)."""
    s = copy.copy(base)
    opts = dict(base._opts)
    opts["ipopt.max_iter"] = int(max_iter)
    opts.pop("ipopt.max_cpu_time", None)
    if max_cpu_time is not None:
        opts["ipopt.max_cpu_time"] = float(max_cpu_time)
    s._opts = opts
    s.solver = ca.nlpsol("to_ocp", "ipopt", s.funcs["nlp"], opts)
    s._refine_solver = None
    s._warm_solver = None
    return s


def make_solver(steps: int, max_iter: int = IPOPT_STAGE_MAX_ITER,
                max_cpu_time: float | None = START_MAX_CPU,
                spec: WindowSpec | None = None, compile: bool = True) -> AnchoredSolver:
    """Cached solver per (steps, caps, spec): WindowSpec(steps=steps) unless ``spec`` is given
    (its steps are replaced; the min-time-in-actuation key uses ``ACTUATION_SPEC``). The
    compiled library comes from the TO-OCP cache (``window.compile_functions`` with
    ``default_cache_dir``): one codegen per (steps, spec) per process, one nlpsol per cap set."""
    spec = dataclasses.replace(spec or WindowSpec(), steps=int(steps))
    cap = None if max_cpu_time is None else float(max_cpu_time)
    key = (spec, int(max_iter), cap, bool(compile))
    if key not in _SOLVERS:
        base = _BASE_SOLVERS.get((spec, bool(compile)))
        if base is None:
            base = AnchoredSolver(spec, max_iter=int(max_iter), compile=compile,
                                  ipopt={} if cap is None else {"max_cpu_time": cap})
            _BASE_SOLVERS[(spec, bool(compile))] = base
            _SOLVERS[key] = base
        else:
            _SOLVERS[key] = _with_caps(base, max_iter, cap)
    return _SOLVERS[key]


def solve_from_state(solver: AnchoredSolver, track: Track, x_prev: np.ndarray,
                     a_prev: np.ndarray | None, init: tuple[np.ndarray, np.ndarray],
                     p: VehicleParams = F1_2025, ctx: np.ndarray | None = None,
                     hidden_mu: float = 1.0) -> WindowResult:
    """An anchored solve from the agent's state x_{L-1} with a_0 fixed to its action a_{L-1}
    (protocol 6.1). The result's first step reproduces the agent's step L-1 -> L. With
    ``a_prev=None`` it is ``WindowSolver.solve`` from x_prev (V10)."""
    xs, acts = (np.array(v, float) for v in init)
    if a_prev is not None:
        acts[0] = np.asarray(a_prev, float).reshape(NA)
    return solver.solve(track, np.asarray(x_prev, float), (xs, acts), ctx=ctx, p=p,
                        hidden_mu=hidden_mu, a0_fixed=a_prev)


class _FixedFirst:
    """``solve_continuation``'s solver with a_0 fixed in every stage (anchored keys)."""

    def __init__(self, solver: AnchoredSolver, a0: np.ndarray):
        self.solver, self.a0 = solver, a0

    def solve(self, *args, **kw) -> WindowResult:
        return self.solver.solve(*args, a0_fixed=self.a0, **kw)


@functools.cache
def _jit_control_step():
    import jax

    from apexzero.vehicle.sim import control_step

    return jax.jit(lambda x, a, tr, p: control_step(x, a, tr, p, DT, 10))


def _track_arrays64(track: Track):
    import jax.numpy as jnp

    from apexzero.vehicle.sim import track_arrays

    _x64()
    return track_arrays(track, dtype=jnp.float64)


def extend_straight(x_end: np.ndarray, n: int, track: Track, p: VehicleParams = F1_2025
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The guess's tail: n steps of the float64 control step from x_end at full throttle with
    a lane-keeping steer (at most 0.3 of the lock): the yaw rate the road's curvature needs,
    kappa(s) vx (feed-forward), plus a critically damped (1 rad/s) loop on the heading error
    and the offset with integral action on the offset (0.3 /s, for the car's understeer):
    r_des = kappa vx - 2 xi - (n + 0.3 int n dt) / vx. On the s-OCP's exit straight it is a
    straight run; on a curved road (tier D, E1 and E2 strips continue past s_c1 + 600 m) it
    stays in the corridor (R 500 m at full throttle: |n| < 3.5 m) instead of settling at the
    heading error -kappa v and leaving it. Returns (xs [n+1, 8], acts [n, 2])."""
    import jax.numpy as jnp

    xs = np.empty((n + 1, NX))
    acts = np.zeros((n, NA))
    x = np.asarray(x_end, float).reshape(NX).copy()
    xs[0] = x
    if n == 0:
        return xs, acts
    tr = _track_arrays64(track)
    step = _jit_control_step()
    pc = _drive_params(p)
    L = p.lf + p.lr
    kc = packed_coeffs(np.asarray(track.kappa, float), track.closed, track.n_points)
    n_int = 0.0
    for k in range(n):
        vx = max(float(x[3]), 5.0)
        kap = float(spline_np(kc, np.array([x[0]]), track.ds, track.n_points, track.closed)[0])
        n_int += float(x[1]) * DT
        r_des = kap * vx - 2.0 * x[2] - (x[1] + 0.3 * n_int) / vx
        a = np.array([np.clip(L * r_des / vx / p.delta_max, -0.3, 0.3), 1.0])
        xn = np.asarray(step(jnp.asarray(x), jnp.asarray(a), tr, pc))
        if not np.all(np.isfinite(xn)):            # hold the speed, advance s
            xn = x.copy()
            xn[0] += max(x[3], 0.0) * DT
        x = xn
        xs[k + 1] = x
        acts[k] = a
    return xs, acts


def socp_guess(socp: SocpKey | OCPResult, track: Track, x0: np.ndarray, steps: int,
               p: VehicleParams = F1_2025) -> tuple[np.ndarray, np.ndarray]:
    """``window.min_time_guess`` from the s-OCP, extended past its end with a straight
    full-throttle tail (``extend_straight``; the repo guess freezes there). Returns
    (xs [steps+1, 8], acts [steps, 2])."""
    ocp = socp.result if isinstance(socp, SocpKey) else socp
    x0 = np.asarray(x0, float).reshape(NX)
    tau0 = float(np.interp(x0[0], ocp.s, ocp.t))
    k_in = int(np.clip(np.floor((ocp.t[-1] - tau0) / DT + 1e-9), 0, steps))
    if k_in > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xs, acts = min_time_guess(ocp, track, x0, k_in)
    else:
        xs, acts = x0[None].copy(), np.zeros((0, NA))
    if k_in < steps:
        xe, ae = extend_straight(xs[-1], steps - k_in, track, p)
        xs = np.vstack([xs, xe[1:]])
        acts = np.vstack([acts, ae])
    return xs, acts


def retime(key: KeyRecord | Trajectory | WindowResult, x0: np.ndarray, steps: int,
           s_shift: float = 0.0, track: Track | None = None, p: VehicleParams = F1_2025
           ) -> tuple[np.ndarray, np.ndarray]:
    """Warm start from another solution re-timed by aligning in s (its s shifted by
    ``s_shift``): the guess's state k is the source's state at the time its s reaches x0's s,
    plus k dt. A start before the source's first s holds x0's speed until it gets there (so a
    phase-shifted start, u_k earlier, follows the nominal key in s); past the source's end
    the guess runs ``extend_straight`` on ``track`` (or holds the last speed without one).
    Used for the neighbouring larger-R key, phases, anchored and viability keys."""
    if getattr(key, "x", None) is None or getattr(key, "a", None) is None or len(key.x) < 2:
        raise ValueError("retime needs a solved source trajectory (x and a)")
    xs_src = np.asarray(key.x, float)
    a_src = np.asarray(key.a, float)
    x0 = np.asarray(x0, float).reshape(NX)
    t_src = np.arange(len(xs_src)) * DT
    s_src = _monotone_s(xs_src[:, 0] + s_shift)
    if x0[0] >= s_src[0]:
        tau0 = float(np.interp(x0[0], s_src, t_src))
    else:
        tau0 = -(s_src[0] - x0[0]) / max(float(x0[3]), 1.0)
    tau = tau0 + np.arange(steps + 1) * DT
    xs = np.empty((steps + 1, NX))
    inside = (tau >= 0.0) & (tau <= t_src[-1])
    for c in range(NX):
        xs[inside, c] = np.interp(tau[inside], t_src, s_src if c == 0 else xs_src[:, c])
    pre = tau < 0.0
    if np.any(pre):
        xs[pre] = x0
        xs[pre, 0] = x0[0] + max(float(x0[3]), 1.0) * (tau[pre] - tau0)
    k_act = np.clip(np.floor(np.maximum(tau[:-1], 0.0) / DT + 1e-9).astype(int), 0, len(a_src) - 1)
    acts = a_src[k_act].copy()
    post = np.flatnonzero(tau > t_src[-1])
    if post.size:
        k = int(post[0])                       # first guess state past the source's end
        if k == 0:                             # x0 lies beyond the source: no overlap at all
            if track is not None:
                return extend_straight(x0, steps, track, p)
            xs[:] = x0
            xs[:, 0] = x0[0] + max(float(x0[3]), 1.0) * np.arange(steps + 1) * DT
            return xs, np.tile(a_src[-1], (steps, 1))
        if track is not None:
            xe, ae = extend_straight(xs[k - 1], steps + 1 - k, track, p)
            xs[k:] = xe[1:]
            acts[k - 1:] = ae
        else:
            last = xs[k - 1]
            xs[k:] = last
            xs[k:, 0] = last[0] + max(float(last[3]), 1.0) * (np.arange(1, steps + 2 - k) * DT)
    xs[0] = x0
    return xs, acts


# ------------------------------------------------------------------------- measures


def key_metrics(res: WindowResult | KeyRecord, track: Track, marks, ctx: np.ndarray | None = None,
                tables: ModelTables | None = None, p: VehicleParams = F1_2025) -> dict:
    """Every protocol 5 measure of a TO-OCP solution: ``measures.trajectory_from_window`` then
    ``measures.response_metrics`` (the single definition shared with the agents)."""
    traj = M.trajectory_from_window(res, track, p=p, ctx=ctx)
    return dataclasses.asdict(M.response_metrics(traj, marks, None, tables, p))


def socp_metrics(socp: SocpKey, marks, tables: ModelTables | None = None,
                 p: VehicleParams = F1_2025) -> dict:
    """Protocol 5 measures of a pinned s-OCP (``measures.trajectory_from_socp``)."""
    traj = M.trajectory_from_socp(socp.result, p)
    return dataclasses.asdict(M.response_metrics(traj, marks, None, tables, p))


def run_metrics(run: Trajectory, marks, tables: ModelTables | None = None,
                p: VehicleParams = F1_2025) -> dict:
    """Protocol 5 measures of an agent's run (viability targets need its s_b)."""
    return dataclasses.asdict(M.response_metrics(run, marks, None, tables, p))


def key_trajectory(rec: KeyRecord, track: Track, p: VehicleParams = F1_2025) -> Trajectory:
    """A stored TO-OCP key as a ``measures.Trajectory`` (its window spec and alpha_max)."""
    res = WindowResult(np.asarray(rec.x, float), np.asarray(rec.a, float), float(rec.J), True,
                       rec.status, 0, 0.0, 0, meta={"spec": dict(rec.spec or {})})
    pp = p._replace(alpha_max=float(rec.alpha_max))
    return M.trajectory_from_window(res, track, p=pp, ctx=key_ctx(pp))


_STALE_FIELDS = ("d_b", "v_e", "v_min", "s_vmin_rel", "line_apex_rel", "v_b", "has_brake")


def refresh_measures(rec: KeyRecord | SocpKey, bank: StimulusBank,
                     tables: ModelTables | None = None, p: VehicleParams = F1_2025,
                     force: bool = False) -> tuple[KeyRecord | SocpKey, bool]:
    """Recompute a cached key's measures from its stored solution with the current code and
    ``tables`` when its ``measures_hash`` is not the current one (or ``force``). Returns
    (record, moved): ``moved`` says a measure of 6.3's multi-basin rule (or the onset)
    changed beyond 1e-9, so the record's tau and multi-basin flag, which come from every
    start's measures (only the selection's solution is stored), are stale and the key must
    be re-solved (``--force``)."""
    if not force and rec.measures_hash == measures_hash():
        return rec, False
    stim = bank.stimuli[rec.stim_id]
    old = dict(rec.measures or {})
    if isinstance(rec, SocpKey):
        marks = _marks(stim, bank, rec.stim_id, float(rec.x0[0]), 0)
        rec.measures = socp_metrics(rec, marks, tables, p)
    elif rec.status == "verified" and rec.x is not None and rec.a is not None:
        marks = _marks(stim, bank, rec.stim_id, float(rec.x0[0]), int(rec.W))
        traj = key_trajectory(rec, bank.track(rec.stim_id), p)
        rec.measures = dataclasses.asdict(M.response_metrics(
            traj, marks, None, tables, p._replace(alpha_max=float(rec.alpha_max))))
    else:
        rec.measures_hash = measures_hash()
        return rec, False
    rec.measures_hash = measures_hash()

    def differs(a, b) -> bool:
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            return bool(a) != bool(b)
        a, b = float(a), float(b)
        if np.isnan(a) and np.isnan(b):
            return False
        return bool(np.isnan(a) or np.isnan(b) or abs(a - b) > 1e-9)

    moved = isinstance(rec, KeyRecord) and any(
        differs(old.get(k, np.nan), rec.measures.get(k, np.nan)) for k in _STALE_FIELDS)
    return rec, bool(moved)


def refresh_stale(cache: KeyCache, bank: StimulusBank, tables: ModelTables | None = None,
                  kinds: tuple[str, ...] = ("nominal", "backoff", "phase", "null", "socp",
                                            "actuation")) -> dict:
    """``refresh_measures`` on every cached agent-free key whose measures are stale, written
    back. Returns {'refreshed': [key ids], 'moved': [key ids whose multi-basin measures
    changed: re-solve them]}."""
    out: dict[str, list[str]] = {"refreshed": [], "moved": []}
    for kind in kinds:
        for rec in cache.records(kind):
            if rec is None or rec.measures_hash == measures_hash():
                continue
            if rec.stim_id >= len(bank):
                continue
            rec, moved = refresh_measures(rec, bank, tables)
            cache.put(rec, aliases=())
            out["refreshed"].append(rec.key_id)
            if moved:
                out["moved"].append(rec.key_id)
    return out


def _start_measures(m: dict) -> dict:
    """The measures of 6.3's multi-basin rule (s_vmin and the line apex relative to s_apex,
    so their spreads are those of the absolute positions), plus what the recipe needs."""
    g = m.get
    return {"d_b": g("d_b", np.nan), "v_e": g("v_e", np.nan), "v_min": g("v_min", np.nan),
            "s_vmin": g("s_vmin_rel", np.nan), "line_apex": g("line_apex_rel", np.nan),
            "v_b": g("v_b", np.nan), "has_brake": bool(g("has_brake", False))}


def _marks(stim, bank: StimulusBank, stim_id: int, s0: float, W: int):
    """``measures.make_marks`` of a bank row (one triangle rule for agents and keys)."""
    return M.make_marks(stim, np.asarray(bank.w_left[stim_id], float),
                        np.asarray(bank.w_right[stim_id], float), float(s0), int(W))


def _window_traj(x: np.ndarray, a: np.ndarray | None = None) -> Trajectory:
    """A key's states as a ``Trajectory`` (t, x, valid; enough for the active set)."""
    x = np.asarray(x, float)
    return Trajectory(source="to_ocp", t=np.arange(len(x)) * DT, x=x,
                      valid=np.ones(len(x), bool), a=None if a is None else np.asarray(a, float))


# ------------------------------------------------------------------------- problems


@dataclass
class _Problem:
    """One key problem: the stimulus, the solved problem (backed-off: narrowed corridor and
    alpha_max) and the start."""

    bank: StimulusBank
    cell: Cell
    stim: object
    kind: str
    phase: int
    track0: Track             # the stimulus the agent drives
    track: Track              # the problem's (narrowed for backed-off keys)
    p: VehicleParams          # the problem's car (alpha_max)
    ctx: np.ndarray
    x0: np.ndarray
    W: int | None
    shrink: float
    spec: WindowSpec          # template; steps replaced per stage
    a0_fixed: np.ndarray | None = None
    tables: ModelTables | None = None

    @property
    def landmarks(self) -> Landmarks:
        return self.stim.landmarks

    @property
    def row_hash(self) -> str:
        return row_hash(self.bank, self.cell.stim_id)

    def marks(self):
        return _marks(self.stim, self.bank, self.cell.stim_id, self.x0[0], self.W or 0)

    def metrics(self, res) -> dict:
        return key_metrics(res, self.track0, self.marks(), self.ctx, self.tables, self.p)


def _problem(bank: StimulusBank, cell: Cell, kind: str, phase: int = 0,
             p: VehicleParams = F1_2025, shrink: float | None = None,
             alpha_max: float | None = None, W: int | None = None,
             spec: WindowSpec | None = None, x0: np.ndarray | None = None,
             a0_fixed: np.ndarray | None = None, tables: ModelTables | None = None) -> _Problem:
    stim = bank.stimuli[cell.stim_id]
    _guard(stim)
    track0 = bank.track(cell.stim_id)
    if kind == "backoff":
        alpha_max = BACKOFF_ALPHA_MAX if alpha_max is None else alpha_max
        shrink = BACKOFF_SHRINK if shrink is None else shrink
    shrink = 0.0 if shrink is None else float(shrink)
    if shrink or (alpha_max is not None and alpha_max != p.alpha_max):
        track, pp = backoff_problem(track0, p, p.alpha_max if alpha_max is None else alpha_max,
                                    shrink)
    else:
        track, pp = track0, p
    x0 = _cell_x0(cell, phase) if x0 is None else np.asarray(x0, float).reshape(NX)
    spec = spec or (ACTUATION_SPEC if kind == "actuation" else WindowSpec())
    return _Problem(bank=bank, cell=cell, stim=stim, kind=kind, phase=int(phase), track0=track0,
                    track=track, p=pp, ctx=key_ctx(pp), x0=x0, W=W, shrink=shrink, spec=spec,
                    a0_fixed=None if a0_fixed is None else np.asarray(a0_fixed, float),
                    tables=tables)


def _labels(prob: _Problem) -> dict:
    spec = getattr(prob.stim, "spec", None)
    out = {"approach": prob.cell.approach, "tiers": list(prob.cell.tiers), "v0": prob.cell.v0,
           "D": prob.cell.D, "cell_flags": list(prob.cell.flags), "stim_flags":
           list(getattr(prob.stim, "flags", ()))}
    if spec is not None:
        out.update(R=spec.R, theta_deg=spec.theta_deg, width=spec.width, l_r=spec.l_r,
                   ramp_kind=spec.ramp_kind, no_cap=spec.no_cap, w_in_corner=spec.w_in_corner)
    geo = getattr(prob.stim, "geometry", None)
    if geo is not None and hasattr(geo, "R_seg"):
        out["R_seg"] = geo.R_seg
    return out


def _socp_for(prob: _Problem, cache: KeyCache | None, s_end: float | None = None,
              vx_upper: dict[float, float] | None = None) -> SocpKey:
    """The problem's pinned s-OCP from its x0 (cached)."""
    s_end = socp_end(prob.landmarks) if s_end is None else s_end
    kid = key_id("socp", prob.row_hash, prob.x0, 0, prob.p, None,
                 socp_extra(s_end, prob.shrink, vx_upper))
    marks = _marks(prob.stim, prob.bank, prob.cell.stim_id, prob.x0[0], prob.W or 0)
    if cache is not None:
        hit = cache.get("socp", kid)
        if hit is not None:
            # The measures follow the current code and tables, not the first caller's.
            hit.measures = socp_metrics(hit, marks, prob.tables, prob.p)
            hit.measures_hash = measures_hash()
            return hit
    soc = socp_from_state(prob.track, prob.x0, s_end, prob.p, vx_upper=vx_upper,
                          max_cpu_time=START_MAX_CPU, stim_id=prob.cell.stim_id,
                          cell_id=prob.cell.cell_id, phase=prob.phase, key_id=kid)
    soc.measures = socp_metrics(soc, marks, prob.tables, prob.p)
    soc.measures_hash = measures_hash()
    if cache is not None:
        nominal_problem = not vx_upper and prob.shrink == 0.0 and prob.p == F1_2025
        cache.put(soc, aliases=(f"cell{prob.cell.cell_id}_ph{prob.phase}_-",) if nominal_problem else ())
    return soc


def socp_extra(s_end: float, shrink: float = 0.0, vx_upper: dict | None = None) -> dict:
    """The s-OCP key id's extras: end, corridor shrink, vx bounds and the formulation (no
    lead-in; bounds between nodes interpolated)."""
    return {"s_end": float(s_end), "shrink": float(shrink),
            "vx_upper": sorted((float(k), float(v)) for k, v in (vx_upper or {}).items()),
            "lead_in": SOCP_LEAD_IN, "vx_rule": "interp"}


def _was_capped(res: WindowResult) -> bool:
    """A solve round hit max_iter or the CPU cap. The polishing refine round is left out: when
    it stops short the solver keeps the converged unrefined optimum (``WindowSolver.solve``),
    and the retry would hit the same 100-iteration refine cap."""
    return any(str(r.get("status")) in CAPPED_STATUSES for r in res.meta.get("rounds", [])
               if r.get("round") != "refine")


def _max_slip(x: np.ndarray, p: VehicleParams) -> float:
    fo = tyre_forces(np.asarray(x, float), p, NUMPY_OPS)
    return float(max(np.max(np.abs(fo.alpha_f)), np.max(np.abs(fo.alpha_r))))


def replay_key(track: Track, x0: np.ndarray, acts: np.ndarray, spec: dict,
               ctx: np.ndarray | None = None, hidden_mu: float = 1.0
               ) -> tuple[float, bool, np.ndarray]:
    """``ocp.replay.replay_window`` with the reward the key was solved for: the env's
    ``action_cost`` and ``progress_scale`` are the spec's (as are gamma and substeps, as
    there). The repo copy builds ``EnvParams`` with the env's default weights, so a key solved
    with another action cost (the min-time-in-actuation key, 1e-4 against 0.02) replays to a
    return ~3e-4 off its J* and never verifies (V3). With the default spec the env is the
    repo's, step for step."""
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle.sim import track_arrays

    _x64()
    spec = {**dataclasses.asdict(WindowSpec()), **(spec or {})}
    params = racing.EnvParams(window_steps=10**9, substeps=int(spec["substeps"]),
                              action_cost=float(spec["action_cost"]),
                              progress_scale=float(spec["progress_scale"]))
    tr = track_arrays(track, dtype=jnp.float64)
    c = racing.default_context() if ctx is None else jnp.asarray(ctx)
    st = racing.EnvState(x=jnp.asarray(x0, jnp.float64), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=c.astype(jnp.float64), hidden_mu=jnp.asarray(hidden_mu, jnp.float64),
                         t=jnp.array(0), slow=jnp.array(0.0), s0=jnp.asarray(x0[0], jnp.float64))

    def body(carry, a):
        st, alive = carry
        _, st2, r, term, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, alive & ~term), (st2.x, r, alive & ~term)

    _, (xs, rs, alive) = jax.lax.scan(body, (st, jnp.array(True)), jnp.asarray(acts, jnp.float64))
    rs = np.asarray(rs)
    ret = float(np.sum(float(spec["gamma"]) ** np.arange(len(rs)) * rs))
    return ret, bool(np.asarray(alive).all()), np.vstack([np.asarray(x0)[None], np.asarray(xs)])


def verify_key(res: WindowResult, track: Track, x0: np.ndarray, ctx: np.ndarray | None = None,
               hidden_mu: float = 1.0, rtol: float = REPLAY_RTOL) -> bool:
    """``ocp.replay.verify_window`` on ``replay_key``: the solve succeeded, the replay under the
    key's own reward survives every step and reproduces J* within ``rtol``. Recorded in
    ``res.meta['replay']`` as there."""
    ret, alive, xr = replay_key(track, x0, res.a, res.meta.get("spec", {}), ctx, hidden_mu)
    rel = (ret - res.value) / abs(res.value)
    ok = bool(res.success and alive and abs(rel) <= rtol)
    res.meta["replay"] = {"return": ret, "rel": rel, "survives": alive, "verified": ok,
                          "max_state_dev": float(np.max(np.abs(xr - res.x)))}
    return ok


def _run_start(prob: _Problem, name: str, first_guess: tuple[np.ndarray, np.ndarray],
               extend: Callable, stages: tuple[int, ...], max_iter: int
               ) -> tuple[StartRecord, WindowResult | None]:
    """One start through the continuation stages (``window.solve_continuation`` one stage at a
    time, so the start's CPU budget is checked between stages), then ``verify_key``."""
    used, iters, chain, rounds = 0.0, 0, [], []
    prev_good: WindowResult | None = None
    last: WindowResult | None = None
    capped = False
    xs0, as0 = (np.asarray(v, float) for v in first_guess)
    if not (np.all(np.isfinite(xs0)) and np.all(np.isfinite(as0))):
        return StartRecord(name=name, J=np.nan, success=False, verified=False, capped=False,
                           status="non-finite first guess (not solved)", iterations=0,
                           cpu_seconds=0.0), None
    for T in stages:
        remaining = START_MAX_CPU - used
        if remaining <= 0.0:
            capped = True
            break
        cap = float(CPU_BUCKET * np.ceil(remaining / CPU_BUCKET))
        solver = make_solver(T, max_iter, cap, spec=prob.spec)
        sf = solver if prob.a0_fixed is None else _FixedFirst(solver, prob.a0_fixed)
        res = solve_continuation(lambda _T, s=sf: s, prob.track, prob.x0, [T], first_guess, extend,
                                 ctx=prob.ctx, p=prob.p, previous=prev_good)[0]
        used += float(res.meta.get("cpu_seconds", 0.0))
        iters += res.iterations
        chain.append((T, res.status))
        rounds.append({"steps": T, "warm_start": res.meta.get("warm_start", ""),
                       "rounds": res.meta.get("rounds", [])})
        if res.success:
            prev_good = res
        last = res
    final = last if last is not None and len(last.a) == stages[-1] else None
    capped = capped or used > START_MAX_CPU or (final is not None and _was_capped(final))
    success = final is not None and final.success and not capped
    verified, replay_cpu, measures = False, 0.0, {}
    if success:
        c0 = time.process_time()
        verified = verify_key(final, prob.track, prob.x0, ctx=prob.ctx)
        replay_cpu = time.process_time() - c0
        slip = _max_slip(final.x, prob.p)
        final.meta["replay"]["max_slip"] = slip
        if verified and slip > prob.p.alpha_max + 1e-6:
            verified = False
            final.meta["replay"]["verified"] = False
        if verified:
            measures = _start_measures(prob.metrics(final))
    status = final.status if final is not None else ("capped before the last stage" if capped
                                                     else "not run")
    sr = StartRecord(name=name, J=float(final.value) if success else np.nan, success=success,
                     verified=bool(verified), capped=bool(capped), status=status,
                     iterations=int(iters), cpu_seconds=float(used + replay_cpu), chain=chain,
                     measures=measures, rounds=rounds)
    return sr, final


def _disagree(a: StartRecord, b: StartRecord) -> bool:
    """Starts (1) and (2) disagree (protocol 6.2: > 2 km/h in v_e or v_min, > 1e-4 in J);
    a failed start counts as a disagreement."""
    if not (a.verified and b.verified):
        return True
    dv = max(abs(a.measures["v_e"] - b.measures["v_e"]), abs(a.measures["v_min"] - b.measures["v_min"]))
    dj = abs(a.J - b.J) / max(abs(a.J), 1e-12)
    return bool(not np.isfinite(dv) or dv > kph(START_DISAGREE_V_KPH) or dj > START_DISAGREE_J)


def _record(prob: _Problem, pairs: list[tuple[StartRecord, WindowResult | None]], *,
            key_id_: str, socp_id: str = "", extra_cpu: float = 0.0, flags: list[str] | None = None,
            agent: str = "", anchor: dict | None = None) -> KeyRecord:
    """Selection (highest J among verified starts: the ``best_verified`` rule), near-equal
    set, multi-basin flag, tau, active set, multipliers and measures."""
    starts = [sr for sr, _ in pairs]
    spec = dataclasses.replace(prob.spec, steps=int(prob.W))
    rec = KeyRecord(key_id=key_id_, kind=prob.kind, status="unsolved", stim_id=prob.cell.stim_id,
                    stim_name=prob.stim.name, cell_id=prob.cell.cell_id, phase=prob.phase,
                    direction=int(prob.stim.direction), x0=prob.x0.copy(), W=int(prob.W),
                    spec=dataclasses.asdict(spec), alpha_max=float(prob.p.alpha_max),
                    shrink=float(prob.shrink), starts=starts, socp_id=socp_id,
                    cpu_seconds=float(sum(s.cpu_seconds for s in starts) + extra_cpu),
                    flags=list(flags or []), agent=agent, anchor=anchor or {},
                    code_hash=_code_hash(), created=_now(), pair_id=int(prob.cell.pair_id),
                    labels=_labels(prob), measures_hash=measures_hash())
    ver = [(sr, r) for sr, r in pairs if sr.verified and r is not None]
    if not ver:
        return rec
    sr, best = max(ver, key=lambda q: q[0].J)
    members, logged = near_equal(starts)
    mb, spread, tau = multi_basin(starts, members)
    act = active_set(_window_traj(best.x), prob.track0, prob.p, prob.shrink,
                     landmarks=prob.landmarks, spec=prob.spec)
    lam_g, lam_x = best.meta["lam"]
    rec.status = "verified"
    rec.J = float(best.value)
    rec.x, rec.a = np.asarray(best.x, float), np.asarray(best.a, float)
    rec.replay = dict(best.meta["replay"])
    rec.near_equal, rec.logged, rec.spread, rec.multi_basin, rec.tau = members, logged, spread, mb, tau
    rec.measures = prob.metrics(best)
    rec.active_steps = act.steps
    rec.active = {"slip_limited": act.slip_limited, "lock_limited": act.lock_limited,
                  "corridor_limited": act.corridor_limited,
                  **{f"n_{c}": int(act.steps[:, i].sum()) for i, c in enumerate(ACTIVE_COLUMNS)},
                  "n_slip_nominal_limit": slip_steps(best.x, F1_2025.alpha_max),
                  "n_slip_backoff_limit": slip_steps(best.x, BACKOFF_ALPHA_MAX)}
    rec.lam_g, rec.lam_x = np.asarray(lam_g, float), np.asarray(lam_x, float)
    rec.stats = {"selected": sr.name, "status": best.status, "iterations": best.iterations,
                 "rounds": best.meta.get("rounds", []), "cpu_seconds": best.meta.get("cpu_seconds"),
                 "warm_start": best.meta.get("warm_start", ""), "compiled": best.meta.get("compiled"),
                 "block_offsets": best.meta.get("block_offsets")}
    return rec


def _store(rec: KeyRecord, cache: KeyCache | None) -> KeyRecord:
    if cache is not None:
        cache.put(rec)
    return rec


def _unsolved_status(prob: _Problem, rec: KeyRecord) -> KeyRecord:
    """A cell with no verified key: 'infeasible_bound' when the point-mass certificate holds
    (needs the model tables), else 'unsolved' (protocol 6.4)."""
    if rec.status == "verified" or prob.tables is None:
        return rec
    b = infeasibility_bound(prob.track0, prob.x0, prob.landmarks, prob.tables)
    rec.bound = dataclasses.asdict(b)
    rec.status = "infeasible_bound" if b.infeasible else "unsolved"
    return rec


# ------------------------------------------------------------------------- recipe


def row_hash(bank: StimulusBank, stim_id: int) -> str:
    """Hash of one bank row's samples (kappa, w_left, w_right)."""
    return io.sha256_arrays(np.asarray(bank.kappa[stim_id], np.float64),
                            np.asarray(bank.w_left[stim_id], np.float64),
                            np.asarray(bank.w_right[stim_id], np.float64))


def key_id(kind: str, bank_row_hash: str, x0: np.ndarray, W: int, p: VehicleParams,
           spec: WindowSpec | None, extra: dict | None = None) -> str:
    """sha256 (hex, first 24 chars) of the key's defining inputs (protocol 6.2): kind, the
    stimulus samples' hash, x0's exact bytes, W, every car parameter, the spec and extras
    (shrink, a0, agent, s_end, ...)."""
    payload = {"kind": kind, "row": bank_row_hash,
               "x0": np.asarray(x0, np.float64).reshape(-1).tobytes().hex(), "W": int(W),
               "p": [float(v) for v in p], "ctx": key_ctx(p).tobytes().hex(),
               "spec": None if spec is None else dataclasses.asdict(spec), "extra": extra or {}}
    text = json.dumps(io._jsonable(payload), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def stages_for(W: int, socp: SocpKey, landmarks: Landmarks,
               candidates: tuple[int, ...] = CONT_STAGES) -> tuple[int, ...]:
    """Continuation stages from ``candidates`` below W, then W itself, skipping a stage whose
    window ends before s_c1 + 300 m on the s-OCP's time line (protocol 6.2; a 600-step window,
    used only for V2, runs 200, 400, 600)."""
    ocp = socp.result
    target = landmarks.s_c1 + STAGE_SKIP_MARGIN
    keep = []
    for T in sorted({int(c) for c in candidates}):
        if T >= W:
            continue
        t_end = T * DT
        if t_end <= ocp.t[-1]:
            s_reach = float(np.interp(t_end, ocp.t, ocp.s))
        else:
            s_reach = float(ocp.s[-1] + ocp.z[-1, 2] * (t_end - ocp.t[-1]))
        if s_reach >= target:
            keep.append(T)
    return (*keep, int(W))


def window_length(socp: SocpKey, landmarks: Landmarks) -> int:
    """W = 400 if the pinned s-OCP reaches s_c1 within 5 s (leaving >= 15 s after the
    corner), else 800 (protocol 4)."""
    ocp = socp.result
    if not ocp.success or landmarks.s_c1 > ocp.s[-1]:
        return W_LONG          # a failed s-OCP gives no time line: the long window
    t_c1 = float(np.interp(landmarks.s_c1, ocp.s, ocp.t))
    return W_SHORT if t_c1 <= W_RULE_TIME else W_LONG


def q_rollout_guess(bank: StimulusBank, cell: Cell, x0: np.ndarray, steps: int,
                    tables: ModelTables, c_v: float = Q_CV_DEFAULT
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Start (3): a float64 rollout of the agent-free Q controller (``controls.make_q_policy``
    with ``controls.q_params``) through ``racing.step_no_reset`` from x0 with the cell's
    prev_action and accel. (xs [steps+1, 8], applied actions [steps, 2])."""
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.interp import controls

    _x64()
    params, aux = controls.q_params(bank, [cell], tables, c_v, "Q")
    policy = controls.make_q_policy("Q")
    tr = _track_arrays64(bank.track(cell.stim_id))
    env = racing.EnvParams(window_steps=10**9)
    st = racing.EnvState(x=jnp.asarray(x0, jnp.float64),
                         prev_action=jnp.asarray(cell.prev_action0, jnp.float64),
                         accel=jnp.asarray(cell.accel0, jnp.float64),
                         ctx=jnp.asarray(key_ctx()), hidden_mu=jnp.asarray(1.0, jnp.float64),
                         t=jnp.zeros((), jnp.int32), slow=jnp.zeros((), jnp.float64),
                         s0=jnp.asarray(x0[0], jnp.float64))
    row = jnp.asarray(np.asarray(aux).reshape(-1)[0], jnp.int32)

    def body(state, _):
        pc = racing.params_from_context(state.ctx, hidden_mu=state.hidden_mu)
        mean, _ = policy(params, racing.observe(state, tr, pc), state, tr, row)
        a = jnp.clip(mean, -1.0, 1.0).astype(jnp.float64)
        _, nxt, *_ = racing.step_no_reset(state, a, tr, env)
        return nxt, (nxt.x, a)

    _, (xs, acts) = jax.lax.scan(body, st, None, length=int(steps))
    return np.vstack([np.asarray(x0, float)[None], np.asarray(xs, float)]), np.asarray(acts, float)


def full_recipe(bank: StimulusBank, cell: Cell, kind: str = "nominal", phase: int = 0,
                p: VehicleParams = F1_2025, shrink: float | None = None,
                neighbour: KeyRecord | None = None, add_q_start: bool | None = None,
                stages: tuple[int, ...] = CONT_STAGES, max_iter: int = IPOPT_STAGE_MAX_ITER,
                retry_max_iter: int = RETRY_MAX_ITER, cache: KeyCache | None = None, *,
                alpha_max: float | None = None, tables: ModelTables | None = None,
                q_guess: Callable | None = None, q_cv: float = Q_CV_DEFAULT,
                W: int | None = None, spec: WindowSpec | None = None, force: bool = False,
                prior: list[tuple[StartRecord, WindowResult | None]] | None = None) -> KeyRecord:
    """Protocol 6.2 for one cell: s-OCP, starts (1)-(3) (Q added for the largest R of each
    series, i.e. without a neighbour, on the validation set, or when starts (1) and (2)
    disagree by > 2 km/h in v_e or v_min or 1e-4 in J; ``add_q_start`` forces it), caps, one
    retry, the best verified start, near-equal set, multi-basin flag, tau, active set and
    measures. Never uses an agent rollout. ``q_guess(bank, cell, x0, steps)`` overrides
    start (3)'s rollout (default: ``q_rollout_guess`` with ``tables``). Imposed and recovery
    cells are screened with the infeasibility bound first (needs ``tables``). ``prior``
    starts (a failed warm start) join the selection."""
    prob = _problem(bank, cell, kind, phase, p, shrink, alpha_max, W, spec, tables=tables)
    flags: list[str] = []
    screened = "imposed" in cell.flags or "recovery" in cell.flags or cell.approach in ("REC", "imposed")
    if screened and tables is not None:
        b = infeasibility_bound(prob.track0, prob.x0, prob.landmarks, tables)
        if b.infeasible:
            prob.W = int(W or cell.W or W_SHORT)
            kid = key_id(kind, prob.row_hash, prob.x0, prob.W, prob.p, prob.spec,
                         {"shrink": prob.shrink})
            rec = _record(prob, [], key_id_=kid, flags=["screened: infeasible (bound)"])
            rec.status, rec.bound = "infeasible_bound", dataclasses.asdict(b)
            return _store(rec, cache)
    c0 = time.process_time()
    soc = _socp_for(prob, cache)
    socp_cpu = time.process_time() - c0
    if not soc.result.success:
        flags.append(f"socp failed: {soc.result.status}")
    prob.W = int(W or cell.W or window_length(soc, prob.landmarks))
    spec_w = dataclasses.replace(prob.spec, steps=prob.W)
    kid = key_id(kind, prob.row_hash, prob.x0, prob.W, prob.p, spec_w, {"shrink": prob.shrink})
    if cache is not None and not force:
        hit = cache.get(kind, kid)
        if hit is not None:
            return hit
    stg = stages_for(prob.W, soc, prob.landmarks, stages)
    first = socp_guess(soc, prob.track, prob.x0, prob.W, prob.p)

    def extend(state, n):
        return socp_guess(soc, prob.track, state, n, prob.p)

    pairs = list(prior or [])
    pairs.append(_run_start(prob, "socp", first, extend, stg, max_iter))
    if neighbour is not None and neighbour.status == "verified" and neighbour.x is not None:
        g2 = retime(neighbour, prob.x0, prob.W, track=prob.track, p=prob.p)
        pairs.append(_run_start(prob, f"neighbour:{neighbour.key_id}", g2, extend, stg, max_iter))
    add_q = add_q_start
    if add_q is None:
        validation = "V" in cell.tiers or cell.approach == "V"
        has_nb = any(sr.name.startswith("neighbour:") for sr, _ in pairs)
        add_q = validation or not has_nb or _disagree(pairs[-2][0], pairs[-1][0])
    if add_q:
        if q_guess is not None:
            gq = q_guess(bank, cell, prob.x0, prob.W)
        elif tables is not None:
            gq = q_rollout_guess(bank, cell, prob.x0, prob.W, tables, q_cv)
            flags.append(f"Q start c_v {q_cv:g}")      # recipe input, not part of the key id
        else:
            gq = None
            flags.append("Q start skipped: no model tables")
        if gq is not None:
            pairs.append(_run_start(prob, "Q", gq, extend, stg, max_iter))
    if not any(sr.verified for sr, _ in pairs):
        pairs.append(_run_start(prob, "retry", first, extend, stg, retry_max_iter))
    rec = _record(prob, pairs, key_id_=kid, socp_id=soc.key_id, extra_cpu=socp_cpu, flags=flags)
    return _store(_unsolved_status(prob, rec), cache)


def series_of(bank: StimulusBank, cells: list[Cell]) -> list[list[Cell]]:
    """Cells grouped into the recipe's series (same theta, width, approach, ramp kind, no-cap
    and E3 inside width, and direction), each sorted by descending R (protocol 6.2: start (2)
    is the neighbouring larger-R key). Cells without a CornerSpec form one-cell series."""
    groups: dict[tuple, list[Cell]] = {}
    for c in cells:
        sp = getattr(bank.stimuli[c.stim_id], "spec", None)
        if sp is None:
            groups[("cell", c.cell_id)] = [c]
            continue
        k = (sp.theta_deg, sp.width, c.approach, sp.ramp_kind, sp.no_cap, sp.w_in_corner,
             sp.direction, c.v0 if c.approach in ("B", "REC", "E-fast", "imposed") else None,
             c.D if c.approach in ("B", "REC", "E-fast", "imposed") else None)
        groups.setdefault(k, []).append(c)
    return [sorted(g, key=lambda c: -bank.stimuli[c.stim_id].spec.R)
            if getattr(bank.stimuli[g[0].stim_id], "spec", None) is not None else g
            for _, g in sorted(groups.items(), key=lambda kv: repr(kv[0]))]


def solve_series(bank: StimulusBank, series: list[Cell], cache: KeyCache | None = None,
                 log=None, **recipe_kw) -> list[KeyRecord]:
    """Nominal keys of one series in descending R, each with the previous (larger-R)
    verified key as start (2) (``full_recipe``)."""
    out, nb = [], None
    for c in series:
        rec = full_recipe(bank, c, neighbour=nb, cache=cache, **recipe_kw)
        out.append(rec)
        if rec.status == "verified":
            nb = rec
        if log is not None:
            log.step(note=f"{rec.stim_name} {c.approach}: {rec.status}")
    return out


def warm_key(base: KeyRecord, bank: StimulusBank, cell: Cell, kind: str, phase: int = 0,
             p: VehicleParams = F1_2025, shrink: float | None = None,
             cache: KeyCache | None = None, *, alpha_max: float | None = None,
             tables: ModelTables | None = None, force: bool = False, **recipe_kw) -> KeyRecord:
    """Backed-off and phase-matched keys: one start warm from ``base`` (the nominal key,
    re-timed in s to the phase's start), one stage at W; if it fails, ``full_recipe`` (with
    the failed warm start in the record)."""
    prob = _problem(bank, cell, kind, phase, p, shrink, alpha_max, base.W, tables=tables)
    spec_w = dataclasses.replace(prob.spec, steps=prob.W)
    kid = key_id(kind, prob.row_hash, prob.x0, prob.W, prob.p, spec_w, {"shrink": prob.shrink})
    if cache is not None and not force:
        hit = cache.get(kind, kid)
        if hit is not None:
            return hit
    prior: list = []
    if base.status == "verified" and base.x is not None:
        guess = retime(base, prob.x0, prob.W, track=prob.track, p=prob.p)

        def extend(state, n):
            return retime(base, state, n, track=prob.track, p=prob.p)

        pair = _run_start(prob, f"warm:{base.key_id}", guess, extend, (prob.W,),
                          IPOPT_STAGE_MAX_ITER)
        if pair[0].verified:
            return _store(_record(prob, [pair], key_id_=kid, socp_id=base.socp_id), cache)
        prior = [pair]
    rec = full_recipe(bank, cell, kind, phase, p, prob.shrink, cache=cache, alpha_max=alpha_max,
                      tables=tables, W=prob.W, force=True, prior=prior, **recipe_kw)
    rec.flags.append("warm start failed: full recipe" if prior else "no verified base: full recipe")
    return _store(rec, cache)


def backoff_problem(track: Track, p: VehicleParams = F1_2025,
                    alpha_max: float = BACKOFF_ALPHA_MAX, shrink: float = BACKOFF_SHRINK
                    ) -> tuple[Track, VehicleParams]:
    """The backed-off problem: widths narrowed by ``shrink`` per side (data: the natural
    spline of w - c is the spline of w minus c, so nothing recompiles) and alpha_max 8.5 deg
    (a runtime solver parameter); the lock is unchanged (the car's physical limit)."""
    name = f"{track.name}/bo{np.degrees(alpha_max):.2f}deg-{shrink:.2f}m"
    t = dataclasses.replace(track, name=name, w_left=np.asarray(track.w_left, float) - shrink,
                            w_right=np.asarray(track.w_right, float) - shrink,
                            meta=dict(track.meta, source_track=track.name, shrink=shrink,
                                      alpha_max=alpha_max))
    return t, p._replace(alpha_max=float(alpha_max))


def null_key(bank: StimulusBank, cell: Cell, cache: KeyCache | None = None, *,
             tables: ModelTables | None = None, p: VehicleParams = F1_2025,
             force: bool = False) -> KeyRecord:
    """Null key on a tier G straight for one (v0, width, W): s-OCP plus one start (the
    s-OCP guess, one stage at W), and the one retry if it fails."""
    prob = _problem(bank, cell, "null", 0, p, tables=tables)
    c0 = time.process_time()
    soc = _socp_for(prob, cache)
    socp_cpu = time.process_time() - c0
    prob.W = int(cell.W or window_length(soc, prob.landmarks))
    spec_w = dataclasses.replace(prob.spec, steps=prob.W)
    kid = key_id("null", prob.row_hash, prob.x0, prob.W, prob.p, spec_w, {})
    if cache is not None and not force:
        hit = cache.get("null", kid)
        if hit is not None:
            return hit
    first = socp_guess(soc, prob.track, prob.x0, prob.W, prob.p)

    def extend(state, n):
        return socp_guess(soc, prob.track, state, n, prob.p)

    pairs = [_run_start(prob, "socp", first, extend, (prob.W,), IPOPT_STAGE_MAX_ITER)]
    if not pairs[0][0].verified:
        pairs.append(_run_start(prob, "retry", first, extend, (prob.W,), RETRY_MAX_ITER))
    rec = _record(prob, pairs, key_id_=kid, socp_id=soc.key_id, extra_cpu=socp_cpu)
    return _store(rec, cache)


def _anchor_index(run: Trajectory, s_target: float) -> int | None:
    """First state reached alive with s >= s_target (None if never): an index the run reaches
    before its terminating transition."""
    valid = np.asarray(run.valid, bool)
    last_alive = (run.term_step if run.term_step >= 0 else len(run.x) - 1)
    idx = np.flatnonzero(valid & (run.x[:, 0] >= s_target))
    idx = idx[idx <= last_alive]
    return int(idx[0]) if idx.size else None


def anchor_base(cell: Cell, phase: int, nominal: KeyRecord | None,
                cache: KeyCache | None = None, bank: StimulusBank | None = None
                ) -> KeyRecord | None:
    """The key a re-anchored key at phase k starts from (ruling R17a): the nominal key at
    that phase, i.e. the phase-matched key at u_k when one is cached and verified, else the
    u = 0 nominal key ``nominal`` (the key at the run's phase when one exists, else the u = 0
    key: the rule every other error follows). Its d_b* and v_b* place s_L, its sustained
    brake gates re-anchoring, it is the warm start, and it is what a run falls back to when
    re-anchoring does not apply (6.1). Looked up, never solved."""
    if phase:
        rec = key_for(cell, "phase", int(phase), cache=cache, bank=bank)
        if rec is not None and rec.status == "verified":
            return rec
    return nominal


@dataclass
class Anchor:
    """A re-anchored problem: the run's state x_{L-1} and action a_{L-1} at the first state
    reached alive with s >= s_L, the cache id and the base key it starts from."""

    key_id: str
    phase: int
    L: int
    s_L: float
    x_prev: np.ndarray
    a_prev: np.ndarray
    base: KeyRecord


def anchored_key_id(bank: StimulusBank, x_prev: np.ndarray, a_prev: np.ndarray, stim_id: int,
                    agent: str, snapshot: str, phase: int, W: int = W_SHORT,
                    p: VehicleParams = F1_2025) -> str:
    """Cache id of a re-anchored key: the agent's state and action, W, the car and spec, and
    (ruling R17a) the deterministic phase whose run it is anchored on, with the agent and
    snapshot. The same state reached at two phases gives two keys."""
    spec_w = dataclasses.replace(WindowSpec(), steps=int(W))
    return key_id("anchored", row_hash(bank, stim_id), np.asarray(x_prev, float), int(W), p,
                  spec_w, {"a0": np.asarray(a_prev, float).tolist(), "agent": agent,
                           "snapshot": snapshot, "phase": int(phase)})


def anchor_for(bank: StimulusBank, cell: Cell, base: KeyRecord, run: Trajectory, *,
               agent: str = "", snapshot: str = "", phase: int = 0, W: int = W_SHORT,
               p: VehicleParams = F1_2025) -> Anchor | None:
    """Where the run at phase k is re-anchored (protocol 6.1, R17a): s_L = s_c0 - (d_b* +
    v_b* 1.0 s) of ``base`` (``anchor_base``: the key at that phase), the run's first state
    reached alive with s >= s_L. None when re-anchoring does not apply: ``base`` unverified
    or without a sustained brake, or the run not reaching s_L alive (at A2 s_L lies before
    the start)."""
    m = base.measures
    if base.status != "verified" or not m.get("has_brake"):
        return None
    lm = bank.stimuli[cell.stim_id].landmarks
    s_L = float(lm.s_c0 - (m["d_b"] + m["v_b"] * ANCHOR_LEAD))
    L = _anchor_index(run, s_L)
    if L is None or L < 1:
        return None
    x_prev = np.asarray(run.x[L - 1], float)
    a_prev = np.asarray(run.a[L - 1], float)
    kid = anchored_key_id(bank, x_prev, a_prev, cell.stim_id, agent, snapshot, phase, W, p)
    return Anchor(kid, int(phase), int(L), s_L, x_prev, a_prev, base)


def anchored_key(bank: StimulusBank, cell: Cell, nominal: KeyRecord, agent: str,
                 run: Trajectory, cache: KeyCache | None = None, *, snapshot: str = "",
                 phase: int = 0, tables: ModelTables | None = None,
                 p: VehicleParams = F1_2025, W: int = W_SHORT, force: bool = False) -> KeyRecord:
    """Re-anchored key (W = 400) from the agent's deterministic run at phase ``phase``
    (ruling R17a: one key per deterministic phase, each anchored on that phase's own run) at
    s_L = s_c0 - (d_b* + v_b* 1.0 s): solved from x_{L-1} with a_0 = a_{L-1}, start (1)
    ``nominal`` re-timed from the matched s (one retry at max_iter 3000 if it fails).
    ``nominal`` is the key at that phase (``anchor_base``: the phase-matched key when one
    exists, else the u = 0 key); it sets s_L. Only when the run reaches s_L alive and that
    key has a sustained brake; else returns ``nominal`` itself (protocol 6.1: the nominal
    key at that phase). The record carries the phase (``KeyRecord.phase``, its alias
    ``cell<id>_ph<k>_<agent>@<snapshot>``, its id). Post-tag."""
    an = anchor_for(bank, cell, nominal, run, agent=agent, snapshot=snapshot, phase=phase,
                    W=W, p=p)
    if an is None:
        return nominal
    x_prev, a_prev, kid = an.x_prev, an.a_prev, an.key_id
    prob = _problem(bank, cell, "anchored", an.phase, p, W=W, x0=x_prev, a0_fixed=a_prev,
                    tables=tables)
    if cache is not None and not force:
        # A hit counts only when it was started from this base key: the id holds the
        # problem (state, action, phase), not the warm start, so a key anchored from the
        # u = 0 key before the phase key existed is re-solved from the phase key once it
        # does (same s_L step, same id; the new record replaces the old).
        hit = cache.get("anchored", kid)
        if hit is not None and hit.anchor.get("nominal", nominal.key_id) == nominal.key_id:
            return hit
    guess = retime(nominal, x_prev, W, track=prob.track, p=prob.p)

    def extend(state, n):
        return retime(nominal, state, n, track=prob.track, p=prob.p)

    pairs = [_run_start(prob, f"nominal:{nominal.key_id}", guess, extend, (W,), IPOPT_STAGE_MAX_ITER)]
    if not pairs[0][0].verified:
        pairs.append(_run_start(prob, "retry", guess, extend, (W,), RETRY_MAX_ITER))
    anchor = {"step": an.L, "s_L": an.s_L, "s_anchor": float(x_prev[0]),
              "a_prev": a_prev.tolist(), "snapshot": snapshot, "nominal": nominal.key_id,
              "phase": an.phase, "base_kind": nominal.kind, "base_phase": int(nominal.phase)}
    rec = _record(prob, pairs, key_id_=kid, socp_id=nominal.socp_id, agent=agent, anchor=anchor)
    return _store(rec, cache)


VIABILITY_TARGETS = ("s_b", "s_c0-50", "s_c0", "s_a0")


def viability_keys(bank: StimulusBank, cell: Cell, run: Trajectory, nominal: KeyRecord,
                   backoff: KeyRecord, cache: KeyCache | None = None, *, agent: str = "",
                   snapshot: str = "", tables: ModelTables | None = None,
                   p: VehicleParams = F1_2025, force: bool = False) -> list[KeyRecord]:
    """Viability solves (W = 200) from the agent's states at s_b, s_c0 - 50, s_c0 and s_a0
    (those before its crash), each from x_{L-1} with a_0 = a_{L-1}; starts: the nominal and
    backed-off keys re-timed (and the one retry). 'verified' means a crash-free continuation
    exists from that state. Post-tag."""
    stim = bank.stimuli[cell.stim_id]
    lm = stim.landmarks
    mr = run_metrics(run, _marks(stim, bank, cell.stim_id, run.x[0, 0], len(run.x) - 1), tables, p)
    targets = {"s_b": mr.get("s_b", np.nan), "s_c0-50": lm.s_c0 - 50.0, "s_c0": lm.s_c0,
               "s_a0": lm.s_a0}
    s_crash = float(run.x[min(run.term_step + 1, len(run.x) - 1), 0]) if run.term_step >= 0 else np.inf
    out = []
    for name in VIABILITY_TARGETS:
        s_t = float(targets[name])
        if not np.isfinite(s_t) or s_t >= s_crash:
            continue
        L = _anchor_index(run, s_t)
        if L is None or L < 1:
            continue
        x_prev = np.asarray(run.x[L - 1], float)
        a_prev = np.asarray(run.a[L - 1], float)
        prob = _problem(bank, cell, "viability", 0, p, W=VIABILITY_W, x0=x_prev, a0_fixed=a_prev,
                        tables=tables)
        spec_w = dataclasses.replace(prob.spec, steps=VIABILITY_W)
        kid = key_id("viability", prob.row_hash, x_prev, VIABILITY_W, prob.p, spec_w,
                     {"a0": a_prev.tolist(), "agent": agent, "snapshot": snapshot, "target": name})
        if cache is not None and not force:
            hit = cache.get("viability", kid)
            if hit is not None:
                out.append(hit)
                continue
        pairs = []
        for src in (nominal, backoff):
            if src is None or src.status != "verified" or src.x is None:
                continue
            guess = retime(src, x_prev, VIABILITY_W, track=prob.track, p=prob.p)

            def extend(state, n, src=src, prob=prob):
                return retime(src, state, n, track=prob.track, p=prob.p)

            pairs.append(_run_start(prob, f"{src.kind}:{src.key_id}", guess, extend,
                                    (VIABILITY_W,), IPOPT_STAGE_MAX_ITER))
        if pairs and not any(sr.verified for sr, _ in pairs):
            src = nominal if nominal.status == "verified" else backoff
            guess = retime(src, x_prev, VIABILITY_W, track=prob.track, p=prob.p)

            def extend_r(state, n, src=src, prob=prob):
                return retime(src, state, n, track=prob.track, p=prob.p)

            pairs.append(_run_start(prob, "retry", guess, extend_r, (VIABILITY_W,), RETRY_MAX_ITER))
        anchor = {"target": name, "s_target": s_t, "step": L, "s_anchor": float(x_prev[0]),
                  "s_state": float(run.x[L, 0]), "a_prev": a_prev.tolist(), "snapshot": snapshot}
        rec = _record(prob, pairs, key_id_=kid, socp_id=nominal.socp_id, agent=agent, anchor=anchor,
                      flags=[] if pairs else ["no verified nominal or backed-off key to start from"])
        out.append(_store(rec, cache))
    return out


def _tested_s(r: KeyRecord) -> float:
    """The tested state's s: the agent's state x_L at or after the target (the solve from
    x_{L-1} with a_0 = a_{L-1} reproduces it exactly)."""
    return float(r.anchor.get("s_state", r.anchor.get("s_anchor", np.inf)))


def no_return(viability: list[KeyRecord], run: Trajectory, key: Trajectory) -> NoReturn:
    """Point of no return from the viability solves: the first tested state (in s) with no
    verified crash-free solution; certified when the tested state before it has one. dv is
    vx - vx* there (the key as a Trajectory). ``s_last_viable`` is the last tested state
    before it with a verified key (all of them viable: s None, and s_last_viable the last
    tested state, which crash typing reads as resolved once it is >= s_a0)."""
    recs = sorted(viability, key=_tested_s)
    last_ok = np.nan
    for i, r in enumerate(recs):
        if r.status != "verified":
            s = _tested_s(r)
            certified = i > 0 and recs[i - 1].status == "verified"
            ok = np.asarray(run.valid, bool)
            v_run = float(np.interp(s, _monotone_s(run.x[ok, 0]), run.x[ok, 3]))
            v_key = float(np.interp(s, _monotone_s(key.x[:, 0]), key.x[:, 3]))
            return NoReturn(s=s, certified=bool(certified), dv=v_run - v_key, s_last_viable=last_ok)
        last_ok = _tested_s(r)
    return NoReturn(s=None, certified=False, dv=np.nan, s_last_viable=last_ok)


# Marks an actuation key whose starts were verified with ``verify_key`` (its own reward,
# K15). Records solved before K15 were replayed with the env's action cost, came back
# 'unsolved', and keep their ids (the id hashes the problem, not the code), so a cached
# actuation record without this flag is re-solved (K16).
ACTUATION_REPLAY_FLAG = "replay under the key's own reward (verify_key, K15)"


def actuation_key(bank: StimulusBank, cell: Cell, cache: KeyCache | None = None, *,
                  tables: ModelTables | None = None, p: VehicleParams = F1_2025,
                  nominal: KeyRecord | None = None, force: bool = False,
                  q_guess: Callable | None = None, q_cv: float = Q_CV_DEFAULT) -> KeyRecord:
    """Min-time in the agent's actuation: TO-OCP with WindowSpec(gamma=1.0, action_cost=1e-4)
    (one compile per window length), validation cells only (V3). Starts (6.2 on the
    validation set, one stage at W: a continuation would compile every stage length, K10):
    (1) the s-OCP guess, (2) the nominal key re-timed (when given or cached), (3) the
    agent-free Q controller's rollout (``q_guess(bank, cell, x0, steps)``, else
    ``q_rollout_guess`` with ``tables`` and ``q_cv``; skipped, and flagged, without either:
    6.2 adds it on the validation set), and the one retry. A cached record without
    ``ACTUATION_REPLAY_FLAG`` predates K15's replay and is re-solved (K16)."""
    prob = _problem(bank, cell, "actuation", 0, p, spec=ACTUATION_SPEC, tables=tables)
    c0 = time.process_time()
    soc = _socp_for(prob, cache)
    socp_cpu = time.process_time() - c0
    prob.W = int(cell.W or window_length(soc, prob.landmarks))
    spec_w = dataclasses.replace(ACTUATION_SPEC, steps=prob.W)
    kid = key_id("actuation", prob.row_hash, prob.x0, prob.W, prob.p, spec_w, {})
    if cache is not None and not force:
        hit = cache.get("actuation", kid)
        if hit is not None and ACTUATION_REPLAY_FLAG in hit.flags:
            return hit
    if nominal is None and cache is not None:
        nominal = key_for(cell, "nominal", 0, cache=cache, bank=bank)
    first = socp_guess(soc, prob.track, prob.x0, prob.W, prob.p)

    def extend(state, n):
        return socp_guess(soc, prob.track, state, n, prob.p)

    flags = [ACTUATION_REPLAY_FLAG]
    pairs = [_run_start(prob, "socp", first, extend, (prob.W,), IPOPT_STAGE_MAX_ITER)]
    if nominal is not None and nominal.status == "verified" and nominal.x is not None:
        g2 = retime(nominal, prob.x0, prob.W, track=prob.track, p=prob.p)
        pairs.append(_run_start(prob, f"nominal:{nominal.key_id}", g2, extend, (prob.W,),
                                IPOPT_STAGE_MAX_ITER))
    if q_guess is not None:
        gq = q_guess(bank, cell, prob.x0, prob.W)
    elif tables is not None:
        gq = q_rollout_guess(bank, cell, prob.x0, prob.W, tables, q_cv)
        flags.append(f"Q start c_v {q_cv:g}")          # recipe input, not part of the key id
    else:
        gq = None
        flags.append("Q start skipped: no model tables")
    if gq is not None:
        pairs.append(_run_start(prob, "Q", gq, extend, (prob.W,), IPOPT_STAGE_MAX_ITER))
    if not any(sr.verified for sr, _ in pairs):
        pairs.append(_run_start(prob, "retry", first, extend, (prob.W,), RETRY_MAX_ITER))
    rec = _record(prob, pairs, key_id_=kid, socp_id=soc.key_id, extra_cpu=socp_cpu, flags=flags)
    return _store(rec, cache)


# ------------------------------------------------------------------------- tolerance


def near_equal(starts: list[StartRecord]) -> tuple[list[str], list[str]]:
    """(N, logged): verified starts with J >= J_best - 1e-4 |J_best|, and those within
    (1e-4, 1e-3] (protocol 6.3)."""
    ver = [s for s in starts if s.verified and np.isfinite(s.J)]
    if not ver:
        return [], []
    jb = max(s.J for s in ver)
    tol_n, tol_l = NEAR_EQUAL_RTOL * abs(jb), LOGGED_RTOL * abs(jb)
    members = [s.name for s in ver if s.J >= jb - tol_n]
    logged = [s.name for s in ver if jb - tol_l <= s.J < jb - tol_n]
    return members, logged


def multi_basin(starts: list[StartRecord], members: list[str]) -> tuple[bool, dict, dict]:
    """(flag, spread, tau): flagged if any of d_b, v_e, v_min, s_vmin, line apex differs
    across N by more than twice its floor; tau = max(floor, spread). A measure defined on
    some members and undefined (nan: no sustained brake) on others is a basin difference."""
    by = {s.name: s for s in starts}
    spread, tau, flag = {}, {}, False
    for m, floor in MULTI_BASIN_FLOORS.items():
        vals = np.array([by[n].measures.get(m, np.nan) for n in members if n in by], float)
        fin = vals[np.isfinite(vals)]
        if fin.size == vals.size:
            sp = float(fin.max() - fin.min()) if fin.size else 0.0
        else:
            sp = np.inf if fin.size else 0.0
        spread[m] = sp
        tau[m] = float(max(floor, sp))
        flag = flag or sp > 2.0 * floor
    return bool(flag), spread, tau


def onset_floor(v_b_star: float) -> float:
    """Floor of onsets compared with a key not solved at that phase: max(1 m, 0.5 v_b* dt)."""
    return max(1.0, 0.5 * v_b_star * DT)


def agent_beats_key(key: KeyRecord, agent_returns: dict[str, float]) -> list[str]:
    """Agents whose deterministic u = 0 return exceeds J* by > 1e-6 |J*| (protocol 6.3):
    the cell is then re-solved warm from every final agent's u = 0 rollout
    (``resolve_beaten_key``, post-tag), the best verified solution taken, errors
    recomputed, and a deviation logged."""
    if key.status != "verified" or not np.isfinite(key.J):
        return []
    lim = key.J + AGENT_BEATS_KEY_RTOL * abs(key.J)
    return sorted(a for a, j in agent_returns.items() if np.isfinite(j) and j > lim)


def resolve_beaten_key(bank: StimulusBank, cell: Cell, key: KeyRecord,
                       agent_runs: dict[str, Trajectory], cache: KeyCache | None = None, *,
                       tables: ModelTables | None = None, p: VehicleParams = F1_2025
                       ) -> tuple[KeyRecord, dict]:
    """The 6.3 re-solve: one start per final agent's u = 0 rollout (its alive prefix, re-timed
    to the window), one stage at W; the result is the best verified solution among these and
    the old key. Returns (record, deviation entry for the findings' Deviations log)."""
    prob = _problem(bank, cell, key.kind, key.phase, p, key.shrink,
                    key.alpha_max if key.alpha_max != p.alpha_max else None, key.W, tables=tables)
    pairs = []
    for agent, run in sorted(agent_runs.items()):
        last = run.term_step if run.term_step >= 0 else len(run.x) - 1
        src = Trajectory(source="agent", t=run.t[: last + 1], x=run.x[: last + 1],
                         valid=np.ones(last + 1, bool), a=run.a[: max(last, 1)])
        guess = retime(src, prob.x0, prob.W, track=prob.track, p=prob.p)

        def extend(state, n, src=src):
            return retime(src, state, n, track=prob.track, p=prob.p)

        pairs.append(_run_start(prob, f"agent:{agent}", guess, extend, (prob.W,),
                                IPOPT_STAGE_MAX_ITER))
    # A key mirrored on the fly ('<id>m') is re-solved as this cell's own problem, stored under
    # the id ``key_for`` looks up first for the cell.
    kid = key.key_id
    if "mirrored" in key.flags:
        kid = expected_key_id(bank, cell, key.kind, key.phase) or key.key_id
    new = _record(prob, pairs, key_id_=kid, socp_id=key.socp_id,
                  flags=["agent beats key: re-solved from agent rollouts"])
    better = new.status == "verified" and new.J > key.J
    out = new if better else key
    if better:
        out.starts = list(key.starts) + list(new.starts)
        _store(out, cache)
    dev = {"date": _now()[:10], "cell_id": cell.cell_id, "kind": key.kind, "key_id": key.key_id,
           "J_old": key.J, "J_new": new.J if new.status == "verified" else np.nan,
           "agents": sorted(agent_runs), "replaced": bool(better),
           "reason": "an agent's deterministic return exceeded J* by > 1e-6 relative (6.3)"}
    return out, dev


@dataclass(frozen=True)
class ActiveSet:
    """Protocol 6.4."""

    steps: np.ndarray         # [W+1, 5] bool, ACTIVE_COLUMNS
    slip_limited: bool        # within +-1 s of v_min
    lock_limited: bool
    corridor_limited: bool


def slip_steps(x: np.ndarray, alpha_limit: float, p: VehicleParams = F1_2025) -> int:
    """States with |alpha_f| or |alpha_r| within 0.5 deg of ``alpha_limit`` (V9 counts against
    the nominal 10 deg limit for both keys)."""
    fo = tyre_forces(np.asarray(x, float), p, NUMPY_OPS)
    near = np.maximum(np.abs(fo.alpha_f), np.abs(fo.alpha_r)) >= alpha_limit - ACTIVE_SLIP
    return int(np.sum(near))


def active_set(traj: Trajectory, track: Track, p: VehicleParams = F1_2025,
               shrink: float = 0.0, landmarks: Landmarks | None = None,
               spec: WindowSpec = WindowSpec()) -> ActiveSet:
    """Slip within 0.5 deg of alpha_max (``p.alpha_max``, front or rear), steering within
    2 deg of delta_max, corridor within 5 cm of the solver's bound (w - shrink - edge margin
    - safety), per state; cell flags from +-1 s around v_min (over [s_c0 - 50, s_c1 + 50]
    when ``landmarks`` are given, else the whole trajectory)."""
    ok = np.asarray(traj.valid, bool)
    x = np.asarray(traj.x, float)
    t = np.asarray(traj.t, float)
    fo = tyre_forces(np.where(ok[:, None], x, 0.0), p, NUMPY_OPS)
    slip_f = np.abs(fo.alpha_f) >= p.alpha_max - ACTIVE_SLIP
    slip_r = np.abs(fo.alpha_r) >= p.alpha_max - ACTIVE_SLIP
    lock = np.abs(x[:, 6]) >= p.delta_max - ACTIVE_STEER
    wl, wr = (_spline_at(w, track, x[:, 0]) for w in (track.w_left, track.w_right))
    edge = spec.edge_margin + spec.safety
    corr_l = (wl - shrink - edge) - x[:, 1] <= ACTIVE_CORRIDOR
    corr_r = x[:, 1] + (wr - shrink - edge) <= ACTIVE_CORRIDOR
    steps = np.stack([slip_f, slip_r, lock, corr_l, corr_r], axis=1) & ok[:, None]
    sel = ok.copy()
    if landmarks is not None:
        sel &= (x[:, 0] >= landmarks.s_c0 - V_MIN_MARGIN) & (x[:, 0] <= landmarks.s_c1 + V_MIN_MARGIN)
    if not sel.any():
        sel = ok
    k = int(np.flatnonzero(sel)[np.argmin(x[sel, 3])])
    win = (np.abs(t - t[k]) <= ACTIVE_WINDOW + 1e-9) & ok
    return ActiveSet(steps=steps, slip_limited=bool(np.any(steps[win, :2])),
                     lock_limited=bool(np.any(steps[win, 2])),
                     corridor_limited=bool(np.any(steps[win, 3:])))


@dataclass(frozen=True)
class Bound:
    """Relaxed point-mass infeasibility certificate (protocol 6.4)."""

    v_reach_apex: float       # m/s, braking from x0 at 1.1 a_dec(v) to s_apex
    v_cap: float              # m/s, sqrt(1.1 a_y,max(v) rho_ub) (first speed above which it fails)
    rho_ub: float             # m, 1.1 x min radius of the min-curvature line (see below)
    infeasible: bool
    rho_lines: dict = field(default_factory=dict)   # min radius (m) per candidate line


def frenet_min_curvature_offsets(kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray,
                                 ds: float, margin: float = 1.0, iters: int = 8,
                                 lam: float = 1e-6) -> np.ndarray:
    """Lateral offsets of the line minimising its own integrated squared curvature,
    int (kappa + n'')^2 / (1 - n kappa) ds (the Frenet curvature (kappa + xi') / sigma' with
    xi ~ n' / e, integrated over the line's arc length), by iteratively reweighted box QPs
    (``audit.qss_distance.box_qp``) with -(w_right - margin) <= n <= w_left - margin.
    Unlike the point-second-difference QP it does not weight the inside of a turn by
    (1 - n kappa)^4, which wraps hairpins round the inner edge."""
    from scipy import sparse

    from apexzero.audit.qss_distance import box_qp
    from apexzero.tracks.reference import second_difference

    k = np.asarray(kappa, float)
    n = len(k)
    d2 = second_difference(n, closed=False) / ds**2
    lo = -(np.asarray(w_right, float) - margin)
    hi = np.asarray(w_left, float) - margin
    mid = 0.5 * (lo + hi)
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
    a = np.clip(np.zeros(n), lo, hi)
    for _ in range(iters):
        w = sparse.diags(1.0 / np.clip(1.0 - a * k, 0.05, None))
        h = d2.T @ w @ d2 + lam * sparse.identity(n)
        a, _ = box_qp(h, -(d2.T @ (w @ k)), lo, hi, x0=a)
    return a


def infeasibility_bound(track: Track, x0: np.ndarray, landmarks: Landmarks,
                        tables: ModelTables, margin: float = 1.0) -> Bound:
    """v_reach,min(s_apex) > v_cap certifies 'infeasible (bound)'; screens imposed D = 100 m
    cells before solving. v_reach,min integrates v dv = -1.1 a_dec(v) ds from x0 to s_apex.
    rho_ub is 1.1 x the smallest radius over [s_c0, s_c1] of the minimum-curvature line
    through the corner cut [s_c0 - 200, s_c1 + 200] (corridor w - margin). The protocol's
    line is ``audit.qss_distance``'s, which wraps a hairpin round its inner edge (R13/180:
    8.0 m), so a feasible hairpin cell could be certified; to keep the certificate relaxed,
    the radius is the largest of that line's, ``frenet_min_curvature_offsets``' line's and the
    centreline's (R13/180: 13.6 m). The cell is infeasible when v^2 > 1.1 a_y,max(v) rho_ub
    at every speed v >= v_reach,min (a_y,max grows with downforce, so every such speed is
    tested)."""
    from apexzero.audit.qss_distance import line_geometry, min_curvature_offsets, strip_xy

    x0 = np.asarray(x0, float).reshape(NX)
    ds, n = track.ds, track.n_points
    i0 = max(int(np.floor((landmarks.s_c0 - 200.0) / ds)), 0)
    i1 = min(int(np.ceil((landmarks.s_c1 + 200.0) / ds)) + 1, n)
    k = np.asarray(track.kappa[i0:i1], float)
    wl, wr = np.asarray(track.w_left[i0:i1], float), np.asarray(track.w_right[i0:i1], float)
    s_nodes = (i0 + np.arange(i1 - i0)) * ds
    corner = (s_nodes >= landmarks.s_c0 - 1e-9) & (s_nodes <= landmarks.s_c1 + 1e-9)
    if not corner.any():
        corner = np.abs(s_nodes - landmarks.s_apex) <= ds
    theta, x, y = strip_xy(k, ds)
    lines = {"audit_min_curvature": min_curvature_offsets(x, y, theta, wl, wr, margin,
                                                          start_offset=None),
             "frenet_min_curvature": frenet_min_curvature_offsets(k, wl, wr, ds, margin),
             "centreline": np.zeros_like(k)}
    rho_lines = {}
    for name, alpha in lines.items():
        _, k_line = line_geometry(k, alpha, ds)
        kmax = float(np.max(np.abs(k_line[corner]))) if corner.any() else 0.0
        rho_lines[name] = 1.0 / kmax if kmax > 0 else np.inf
    rho_ub = BOUND_FACTOR * max(rho_lines.values())
    # v_reach: the lowest speed reachable at s_apex braking at 1.1 a_dec(v) from x0.
    v0, dist = float(x0[3]), float(landmarks.s_apex - x0[0])
    if dist <= 0.0 or v0 <= 0.0:
        v_reach = max(v0, 0.0)
    else:
        vg = np.linspace(0.0, v0, 4001)
        a = BOUND_FACTOR * np.interp(vg, tables.v_dec, tables.a_dec)
        f = vg / np.maximum(a, 1e-9)
        seg = 0.5 * (f[1:] + f[:-1]) * np.diff(vg)
        d_from_top = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]       # distance from v0 down to vg
        v_reach = 0.0 if d_from_top[0] <= dist else float(np.interp(dist, d_from_top[::-1], vg[::-1]))
    if not np.isfinite(rho_ub):
        return Bound(v_reach_apex=v_reach, v_cap=np.inf, rho_ub=rho_ub, infeasible=False,
                     rho_lines=rho_lines)
    vmax = max(float(tables.v_lat[-1]), v_reach) + 1.0
    vg = np.arange(0.0, vmax + 0.05, 0.05)
    # The cornering limit (power-limited entries replaced by tyre grip), so the bound stays
    # a relaxation at top speed (the raw table is 0 at 315 km/h).
    a_y = getattr(tables, "a_y_corner", None)
    a_y = np.asarray(tables.a_y_max if a_y is None else a_y, float)
    gap = BOUND_FACTOR * np.interp(vg, tables.v_lat, a_y) * rho_ub - vg**2
    neg = gap < 0.0
    v_cap = float(vg[np.argmax(neg)]) if neg.any() else np.inf
    above = vg >= v_reach
    infeasible = bool(above.any() and np.all(neg[above]))
    return Bound(v_reach_apex=v_reach, v_cap=v_cap, rho_ub=float(rho_ub), infeasible=infeasible,
                 rho_lines=rho_lines)


# ------------------------------------------------------------------------- layout


def phase_cycle(bank: StimulusBank, cell: Cell, nominal: KeyRecord | None = None,
                socp: SocpKey | None = None, p: VehicleParams = F1_2025) -> float:
    """P = v_sight dt (protocol 4): v_sight from the nominal key, else the pinned s-OCP,
    else the QSS profile from x0 (``rollout.sight_speed``, decision D26)."""
    from apexzero.interp import rollout as R

    lm = bank.stimuli[cell.stim_id].landmarks
    if nominal is not None and nominal.status == "verified" and nominal.x is not None:
        s, vx = nominal.x[:, 0], nominal.x[:, 3]
    elif socp is not None:
        s, vx = socp.result.s, socp.result.z[:, 2]
    else:
        x0 = _cell_x0(cell, 0)
        i0 = int(np.floor(x0[0] / DS))
        k = np.asarray(bank.kappa[cell.stim_id][i0:], float)
        vx = speed_profile(k, DS, GG.from_params(p), closed=False, v_start=float(x0[3]))
        s = (i0 + np.arange(len(k))) * DS
    return float(R.sight_speed(np.asarray(s, float), np.asarray(vx, float), lm.s_c0)) * DT


def refresh_phase_cycles(bank: StimulusBank, cells: list[Cell], cache: KeyCache) -> list[Cell]:
    """P from each cell's nominal key once the nominal keys exist (D26), before the
    phase-matched keys are solved; cells without one keep their P."""
    out = []
    for c in cells:
        nom = key_for(c, "nominal", 0, cache=cache, bank=bank)
        out.append(dataclasses.replace(c, P=phase_cycle(bank, c, nominal=nom)) if nom else c)
    return out


def _with_D(cell: Cell, D: float) -> Cell:
    return dataclasses.replace(cell, D=float(D))


def layout_a2(bank: StimulusBank, cells: list[Cell], tables: ModelTables,
              cache: KeyCache | None = None, p: VehicleParams = F1_2025,
              max_rounds: int = 10, log=None) -> list[Cell]:
    """A2 (protocol 3.5) and every cell's W and P:

    1. For each A2 cell without D, a cruise s-OCP from S_C - 600 m at v0 = v_typ with
       vx <= v0 up to s_c0 gives the brake-demand onset distance from v0; D = d_b + 25 m.
       If it has no sustained brake (or v_min* >= v0 - 0.5 km/h) the cell is flat:
       D = 250 m.
    2. The pinned s-OCP from the cell's start gives v_b; while fewer than 90% of braking A2
       cells have v_b within +-10% of v_typ, each cell outside moves D by 10 m (shorter when
       v_b is too high, longer when too low), at most ``max_rounds`` rounds.
    3. Every cell then gets W (``window_length``) and P (``phase_cycle``) from the pinned
       s-OCP at its start (cached; the nominal recipe reuses it).

    v_b* is read from the pinned s-OCP here (the TO-OCP keys need W and P first); the brief's
    measured gap to the TO-OCP onset is 3-4 m."""
    out = {c.cell_id: c for c in cells}
    a2 = [c for c in cells if c.approach == "A2" and c.D is None]
    braking: dict[int, float] = {}
    for c in a2:
        stim = bank.stimuli[c.stim_id]
        lm = stim.landmarks
        x_far = np.array([lm.s_c0 - A1_D, 0, 0, c.v0, 0, 0, 0, 0], np.float32).astype(float)
        prob = _problem(bank, c, "nominal", 0, p, x0=x_far, tables=tables)
        nodes = lm.s_c0 - A1_D - SOCP_LEAD_IN + DS * np.arange(int((A1_D + SOCP_LEAD_IN) / DS) + 1)
        cruise = {float(s): float(c.v0) for s in nodes if s < lm.s_c0}
        soc = _socp_for(prob, cache, vx_upper=cruise)
        m = soc.measures
        brakes = bool(m.get("has_brake")) and m.get("v_min", np.inf) < c.v0 - kph(0.5)
        if not soc.result.success:
            # No onset distance from a failed solve: listed (flag), laid out flat, and left
            # out of the in-band rule; investigated before the tag.
            out[c.cell_id] = dataclasses.replace(_with_D(c, A2_FLAT_D), flags=tuple(sorted(
                {*c.flags, "flat", "layout_socp_failed"})))
        elif brakes and np.isfinite(m.get("d_b", np.nan)):
            braking[c.cell_id] = float(m["d_b"])
            out[c.cell_id] = dataclasses.replace(_with_D(c, m["d_b"] + A2_D_MARGIN),
                                                 flags=tuple(sorted({*c.flags, "braking"})))
        else:
            out[c.cell_id] = dataclasses.replace(_with_D(c, A2_FLAT_D),
                                                 flags=tuple(sorted({*c.flags, "flat"})))
        if log is not None:
            log.step(note=f"layout cruise {stim.name}")

    def v_b(c: Cell) -> float:
        prob = _problem(bank, c, "nominal", 0, p, tables=tables)
        soc = _socp_for(prob, cache)
        return float(soc.measures.get("v_b", np.nan)) if soc.result.success else np.nan

    vb = {i: v_b(out[i]) for i in braking}
    rounds = 0
    while braking and rounds < max_rounds:
        inband = {i: abs(vb[i] / out[i].v0 - 1.0) <= A2_VB_TOL for i in braking}
        if np.mean(list(inband.values())) >= A2_VB_FRACTION:
            break
        for i, ok in inband.items():
            if ok:
                continue
            c = out[i]
            step = -A2_D_STEP if (not np.isfinite(vb[i]) or vb[i] > c.v0) else A2_D_STEP
            out[i] = _with_D(c, max(c.D + step, A2_D_STEP))
            vb[i] = v_b(out[i])
        rounds += 1
    final = []
    for c in cells:
        c = out[c.cell_id]
        if c.D is None and c.x0 is None:
            final.append(c)
            continue
        prob = _problem(bank, c, "nominal", 0, p, tables=tables)
        soc = _socp_for(prob, cache)
        W = c.W or window_length(soc, prob.landmarks)
        flags = c.flags
        if not soc.result.success:
            flags = tuple(sorted({*flags, "socp_failed"}))
        if c.cell_id in braking:
            ok = abs(vb[c.cell_id] / c.v0 - 1.0) <= A2_VB_TOL
            flags = tuple(sorted({*flags, *(() if ok else ("vb_out_of_band",))}))
        final.append(dataclasses.replace(c, W=int(W), P=phase_cycle(bank, c, socp=soc, p=p),
                                         flags=flags))
        if log is not None:
            log.step(note=f"layout W/P cell {c.cell_id}")
    return final


# ------------------------------------------------------------------------- cache


_KEY_ARRAYS = ("x0", "x", "a", "lam_g", "lam_x", "active_steps")
_RES_ARRAYS = ("s", "z", "u", "t", "kappa", "w_left", "w_right", "dual_left", "dual_right")


def _nan(v):
    return np.nan if v is None else v


def _nan_dict(d: dict) -> dict:
    return {k: (np.nan if v is None else v) for k, v in d.items()}


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _alias_tag(rec: KeyRecord) -> str:
    tag = rec.agent or "-"
    if rec.anchor.get("snapshot"):
        tag += "@" + str(rec.anchor["snapshot"])
    if rec.anchor.get("target"):
        tag += "#" + str(rec.anchor["target"])
    return tag


class KeyCache:
    """keys/<kind>/<key_id>.json + .npz; ``get`` returns None when absent. The .json is written
    last (it marks a complete record). Alias files keys/<kind>/_alias/<tag> name the key of a
    cell (``cell<id>_ph<k>_<agent>``) and of a mirror-pair member
    (``pair<id>_dir<+1|-1>_ph<k>_<agent>``); flags.json holds cache-wide flags (mirror_ok)."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else io.KEYS_DIR

    def _dir(self, kind: str) -> Path:
        if kind not in io.KEY_KINDS:
            raise ValueError(f"unknown key kind {kind!r}; expected one of {io.KEY_KINDS}")
        return self.root / kind

    def get(self, kind: str, key_id: str) -> KeyRecord | SocpKey | None:
        j = self._dir(kind) / f"{key_id}.json"
        if not j.exists():
            return None
        d = io.read_json(j)
        with np.load(j.with_suffix(".npz"), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        return _socp_from(d, arrays) if kind == "socp" else _key_from(d, arrays)

    def put(self, rec: KeyRecord | SocpKey, aliases: tuple[str, ...] | None = None) -> Path:
        if isinstance(rec, SocpKey):
            kind, (d, arrays) = "socp", _socp_to(rec)
            tags = tuple(aliases or ())
        else:
            kind, (d, arrays) = rec.kind, _key_to(rec)
            tags = tuple(aliases) if aliases is not None else (
                f"cell{rec.cell_id}_ph{rec.phase}_{_alias_tag(rec)}",
                f"pair{rec.pair_id}_dir{rec.direction:+d}_ph{rec.phase}_{_alias_tag(rec)}")
        j = self._dir(kind) / f"{rec.key_id}.json"
        _atomic_npz(j.with_suffix(".npz"), arrays)
        io.write_json(j, d)
        for tag in tags:
            self.set_alias(kind, tag, rec.key_id)
        return j

    def set_alias(self, kind: str, tag: str, key_id: str) -> None:
        io.write_json(self._dir(kind) / "_alias" / f"{tag}.json", {"key_id": key_id})

    def alias(self, kind: str, tag: str) -> str | None:
        f = self._dir(kind) / "_alias" / f"{tag}.json"
        return io.read_json(f)["key_id"] if f.exists() else None

    def get_flag(self, name: str, default=None):
        f = self.root / "flags.json"
        return io.read_json(f).get(name, default) if f.exists() else default

    def set_flag(self, name: str, value) -> None:
        f = self.root / "flags.json"
        flags = io.read_json(f) if f.exists() else {}
        flags[name] = value
        io.write_json(f, flags)

    def records(self, kind: str) -> list[KeyRecord | SocpKey]:
        d = self._dir(kind)
        return [self.get(kind, j.stem) for j in sorted(d.glob("*.json"))] if d.exists() else []

    def content_hash(self) -> str:
        """sha256 over (kind/key_id, record JSON, array bytes) in sorted order: the manifest's
        key-cache hash (file containers are not hashed, so it is reproducible). Aliases and
        flags are derived and left out."""
        h = hashlib.sha256()
        for kind in sorted(io.KEY_KINDS):
            d = self.root / kind
            if not d.exists():
                continue
            for j in sorted(d.glob("*.json")):
                h.update(f"{kind}/{j.stem}\n".encode())
                h.update(json.dumps(io.read_json(j), sort_keys=True).encode())
                npz = j.with_suffix(".npz")
                if npz.exists():
                    with np.load(npz, allow_pickle=False) as z:
                        for name in sorted(z.files):
                            h.update(name.encode())
                            h.update(io.sha256_arrays(z[name]).encode())
        return h.hexdigest()


class FreshCache:
    """A ``KeyCache`` view for a forced re-solve that spans several calls (the validate
    stage's ``--force``): a key record cached before the view was made reads as absent, so
    each key is solved once and later reads in the same job get the new record. Writes go
    to the cache. s-OCPs, aliases and flags read through: ``--force`` re-solves keys, as
    ``full_recipe(force=True)`` does, not the s-OCPs they start from."""

    def __init__(self, cache: KeyCache):
        self.cache = cache
        self.fresh: set[tuple[str, str]] = set()

    @property
    def root(self) -> Path:
        return self.cache.root

    def get(self, kind: str, key_id: str) -> KeyRecord | SocpKey | None:
        if kind != "socp" and (kind, key_id) not in self.fresh:
            return None
        return self.cache.get(kind, key_id)

    def put(self, rec: KeyRecord | SocpKey, aliases: tuple[str, ...] | None = None) -> Path:
        self.fresh.add(("socp" if isinstance(rec, SocpKey) else rec.kind, rec.key_id))
        return self.cache.put(rec, aliases)

    def records(self, kind: str) -> list[KeyRecord | SocpKey]:
        return [r for r in self.cache.records(kind)
                if kind == "socp" or (kind, r.key_id) in self.fresh]

    def set_alias(self, kind: str, tag: str, key_id: str) -> None:
        self.cache.set_alias(kind, tag, key_id)

    def alias(self, kind: str, tag: str) -> str | None:
        return self.cache.alias(kind, tag)

    def get_flag(self, name: str, default=None):
        return self.cache.get_flag(name, default)

    def set_flag(self, name: str, value) -> None:
        self.cache.set_flag(name, value)


def _key_to(rec: KeyRecord) -> tuple[dict, dict[str, np.ndarray]]:
    d = {f.name: getattr(rec, f.name) for f in dataclasses.fields(rec) if f.name not in _KEY_ARRAYS}
    d["starts"] = [dataclasses.asdict(s) for s in rec.starts]
    arrays = {k: np.asarray(getattr(rec, k)) for k in _KEY_ARRAYS if getattr(rec, k) is not None}
    return d, arrays


def _key_from(d: dict, arrays: dict[str, np.ndarray]) -> KeyRecord:
    names = {f.name for f in dataclasses.fields(KeyRecord)}
    kw = {k: v for k, v in d.items() if k in names}
    kw["starts"] = [StartRecord(**{**s, "J": _nan(s.get("J")), "measures": _nan_dict(s.get("measures", {}))})
                    for s in d.get("starts", [])]
    kw["J"] = _nan(kw.get("J"))
    for k in ("measures", "spread", "tau"):
        kw[k] = _nan_dict(kw.get(k, {}))
    for k in _KEY_ARRAYS:
        kw[k] = arrays.get(k)
    return KeyRecord(**kw)


def _socp_to(rec: SocpKey) -> tuple[dict, dict[str, np.ndarray]]:
    r = rec.result
    d = {"key_id": rec.key_id, "stim_id": rec.stim_id, "cell_id": rec.cell_id, "phase": rec.phase,
         "s_end": rec.s_end, "pin_check": rec.pin_check,
         "vx_bounds": [[k, v] for k, v in sorted(rec.vx_bounds.items())], "measures": rec.measures,
         "measures_hash": rec.measures_hash,
         "flags": rec.flags, "result": {"lap_time": r.lap_time, "success": r.success,
                                        "status": r.status, "iterations": r.iterations,
                                        "solve_seconds": r.solve_seconds, "meta": r.meta}}
    arrays = {"x0": np.asarray(rec.x0), **{f"res_{k}": np.asarray(getattr(r, k)) for k in _RES_ARRAYS}}
    return d, arrays


def _socp_from(d: dict, arrays: dict[str, np.ndarray]) -> SocpKey:
    r = d["result"]
    res = OCPResult(**{k: arrays[f"res_{k}"] for k in _RES_ARRAYS}, lap_time=r["lap_time"],
                    success=r["success"], status=r["status"], iterations=r["iterations"],
                    solve_seconds=r["solve_seconds"], meta=r["meta"])
    return SocpKey(key_id=d["key_id"], stim_id=d["stim_id"], cell_id=d["cell_id"], phase=d["phase"],
                   x0=arrays["x0"], s_end=d["s_end"], result=res, pin_check=d["pin_check"],
                   vx_bounds={float(k): float(v) for k, v in d["vx_bounds"]},
                   measures=_nan_dict(d["measures"]), flags=d["flags"],
                   measures_hash=str(d.get("measures_hash", "")))


def mirror_state(x: np.ndarray) -> np.ndarray:
    """[s, n, xi, vx, vy, r, delta, pedal] -> [s, -n, -xi, vx, -vy, -r, -delta, pedal]."""
    return np.asarray(x, float) * np.array([1, -1, -1, 1, -1, -1, -1, 1], float)


def mirror_key(rec: KeyRecord, cell: Cell | None = None) -> KeyRecord:
    """Mirror image of a key: states mirrored, steer negated, direction flipped, corridor
    columns of the active set swapped; multipliers dropped (their sign conventions do not
    mirror one to one). Measures are mirror-invariant (eta and the line apex use sigma n)."""
    m = copy.deepcopy(rec)
    m.key_id = rec.key_id + "m"
    m.direction = -rec.direction
    m.x0 = mirror_state(rec.x0)
    if rec.x is not None:
        m.x = mirror_state(rec.x)
    if rec.a is not None:
        m.a = np.asarray(rec.a, float) * np.array([-1.0, 1.0])
    if rec.active_steps is not None:
        m.active_steps = np.asarray(rec.active_steps)[:, [0, 1, 2, 4, 3]]
    m.lam_g = m.lam_x = None
    m.stim_name = rec.stim_name + "~mirror"
    m.flags = [*rec.flags, "mirrored"]
    if cell is not None:
        m.cell_id, m.stim_id, m.pair_id = cell.cell_id, cell.stim_id, cell.pair_id
    return m


AGENT_FREE_KINDS = ("nominal", "backoff", "phase", "null", "actuation")


def expected_key_id(bank: StimulusBank, cell: Cell, kind: str, phase: int = 0,
                    p: VehicleParams = F1_2025) -> str | None:
    """The id under which the recipe stores this cell's agent-free key at the cell's current
    x0 (phase) and W (``full_recipe``, ``warm_key``, ``null_key``, ``actuation_key``), or its
    nominal pinned s-OCP (kind 'socp'); None when it cannot be known yet (no W, no x0)."""
    # The same inputs ``_problem`` gives each recipe (no Track is built: this is a lookup).
    if kind not in (*AGENT_FREE_KINDS, "socp"):
        return None
    try:
        x0 = _cell_x0(cell, phase)
    except ValueError:            # no D (A2 before the layout) or no P (phase > 0)
        return None
    rh = row_hash(bank, cell.stim_id)
    if kind == "socp":
        return key_id("socp", rh, x0, 0, p, None,
                      socp_extra(socp_end(bank.stimuli[cell.stim_id].landmarks)))
    if cell.W is None:
        return None
    W = int(cell.W)
    pp, shrink = (p._replace(alpha_max=float(BACKOFF_ALPHA_MAX)), float(BACKOFF_SHRINK)) \
        if kind == "backoff" else (p, 0.0)
    spec_w = dataclasses.replace(ACTUATION_SPEC if kind == "actuation" else WindowSpec(), steps=W)
    extra = {} if kind in ("null", "actuation") else {"shrink": shrink}
    return key_id(kind, rh, x0, W, pp, spec_w, extra)


def _matches(rec: KeyRecord, cell: Cell, phase: int) -> bool:
    """A mirror-alias hit belongs to this cell's problem: same W, same start s and speed."""
    if cell.W is not None and int(rec.W) != int(cell.W):
        return False
    try:
        x0 = _cell_x0(cell, phase)
    except ValueError:
        return True
    return bool(abs(float(rec.x0[0]) - x0[0]) <= 1e-9 and abs(float(rec.x0[3]) - x0[3]) <= 1e-9)


def key_for(cell: Cell, kind: str = "nominal", phase: int = 0, agent: str = "",
            cache: KeyCache | None = None, bank: StimulusBank | None = None, *,
            snapshot: str = "", solve: bool = False, **recipe_kw) -> KeyRecord | SocpKey | None:
    """The cached key of a cell (mirrored on the fly from the solved left-hand member once
    V5 has passed, i.e. the cache flag ``mirror_ok``); None if it was never solved. With
    ``solve=True`` a missing agent-free key is solved by its recipe (nominal: ``full_recipe``;
    backoff / phase: ``warm_key`` from the nominal; null; actuation; socp) and cached."""
    cache = cache or KeyCache()
    tag = (agent or "-") + (f"@{snapshot}" if snapshot else "")
    want = expected_key_id(bank, cell, kind, phase) if bank is not None and tag == "-" else None
    if want is not None:
        # Agent-free keys are looked up by the id of the cell's current problem (its x0 at
        # this phase and its W), not by the cell alias: an alias names the key solved last
        # for that cell, which V2's second horizon or a layout adjustment can make stale.
        rec = cache.get(kind, want)
        if rec is not None:
            return rec
    else:
        kid = cache.alias(kind, f"cell{cell.cell_id}_ph{phase}_{tag}")
        if kid is not None:
            rec = cache.get(kind, kid)
            if rec is not None:
                return rec
    if bank is not None and kind != "socp" and cache.get_flag("mirror_ok", False):
        stim = bank.stimuli[cell.stim_id]
        if int(stim.direction) == -1:
            rec = None
            if tag == "-" and getattr(stim, "mirror_id", -1) >= 0:
                # The left-hand member's problem: the mirror row from the mirrored state.
                mate = dataclasses.replace(
                    cell, stim_id=int(stim.mirror_id),
                    x0=None if cell.x0 is None else tuple(float(v) for v in mirror_state(cell.x0)))
                want_l = expected_key_id(bank, mate, kind, phase)
                rec = cache.get(kind, want_l) if want_l is not None else None
            if rec is None:
                kid = cache.alias(kind, f"pair{cell.pair_id}_dir+1_ph{phase}_{tag}")
                rec = cache.get(kind, kid) if kid is not None else None
                if rec is not None and not _matches(rec, cell, phase):
                    rec = None
            if rec is not None:
                return mirror_key(rec, cell)
    if not solve:
        return None
    if bank is None:
        raise ValueError("key_for(solve=True) needs the stimulus bank")
    if kind == "nominal":
        return full_recipe(bank, cell, phase=phase, cache=cache, **recipe_kw)
    if kind in ("backoff", "phase"):
        base = key_for(cell, "nominal", 0, cache=cache, bank=bank, solve=True, **recipe_kw)
        return warm_key(base, bank, cell, kind, phase=phase if kind == "phase" else 0, cache=cache,
                        tables=recipe_kw.get("tables"))
    if kind == "null":
        return null_key(bank, cell, cache, tables=recipe_kw.get("tables"))
    if kind == "actuation":
        return actuation_key(bank, cell, cache, tables=recipe_kw.get("tables"))
    if kind == "socp":
        return _socp_for(_problem(bank, cell, "nominal", phase, tables=recipe_kw.get("tables")), cache)
    raise ValueError(f"{kind} keys depend on an agent's run: use anchored_key / viability_keys")


# ------------------------------------------------------------------------- validation


@dataclass
class ValidationResult:
    name: str                 # 'V1' .. 'V11'
    passed: bool
    details: dict = field(default_factory=dict)
    action: str = ""          # what the protocol says to do on failure


def v1_replay(records: list[KeyRecord]) -> ValidationResult:
    """Every key used passes verify_window (return within 1e-6; alive every step). Records
    that are infeasible (bound) or unsolved are not used; they are listed."""
    used = [r for r in records if r.status == "verified"]
    bad = [r.key_id for r in used
           if not (r.replay.get("verified") and r.replay.get("survives")
                   and abs(r.replay.get("rel", np.inf)) <= REPLAY_RTOL)]
    unused = {r.key_id: r.status for r in records if r.status != "verified"}
    rel = [abs(r.replay.get("rel", np.nan)) for r in used]
    return ValidationResult("V1", not bad, {"n_used": len(used), "failed": bad, "not_used": unused,
                                            "max_rel": float(np.nanmax(rel)) if rel else np.nan},
                            "re-solve; otherwise exclude and list")


def _sv(rec: KeyRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _monotone_s(rec.x[:, 0]), rec.x[:, 3], rec.x[:, 1]


def profile_gap(a: KeyRecord, b: KeyRecord, s_hi: float) -> tuple[float, float]:
    """(max |dvx|, max |dn|) between two keys on a 1 m s grid from their common start up to
    ``s_hi`` (V2)."""
    sa, va, na = _sv(a)
    sb, vb, nb = _sv(b)
    hi = min(s_hi, sa[-1], sb[-1])
    grid = np.arange(max(sa[0], sb[0]), hi + 1e-9, 1.0)
    dv = np.abs(np.interp(grid, sa, va) - np.interp(grid, sb, vb))
    dn = np.abs(np.interp(grid, sa, na) - np.interp(grid, sb, nb))
    return float(dv.max()), float(dn.max())


def v2_horizon(bank: StimulusBank, cells: list[Cell], cache: KeyCache | None = None, *,
               tables: ModelTables | None = None, log=None) -> ValidationResult:
    """A2 and imposed cells 400 vs 800 steps; A1 800 vs 600 (one compile): up to s_c1 + 100,
    max |dv| <= 0.1 m/s and |dn| <= 0.05 m. Classes (approaches) that fail need W = 800."""
    det, fail = {}, set()
    for c in cells:
        imposed = c.approach == "imposed" or "imposed" in c.flags
        if c.approach == "A1":
            pair = (W_LONG, 600)
        elif c.approach == "A2" or imposed:
            pair = (W_SHORT, W_LONG)
        else:
            continue
        r1 = full_recipe(bank, c, cache=cache, tables=tables, W=pair[0])
        r2 = full_recipe(bank, c, cache=cache, tables=tables, W=pair[1])
        lm = bank.stimuli[c.stim_id].landmarks
        if r1.status != "verified" or r2.status != "verified":
            det[c.cell_id] = {"windows": pair, "status": (r1.status, r2.status)}
            fail.add("imposed" if imposed else c.approach)
            continue
        dv, dn = profile_gap(r1, r2, lm.s_c1 + 100.0)
        ok = dv <= 0.1 and dn <= 0.05
        det[c.cell_id] = {"windows": pair, "dv": dv, "dn": dn, "passed": ok}
        if not ok:
            fail.add("imposed" if imposed else c.approach)
        if log is not None:
            log.step(note=f"V2 cell {c.cell_id}")
    return ValidationResult("V2", not fail, {"cells": det, "use_W800_for": sorted(fail)},
                            "use W = 800 for that class of stimulus")


def _t_at(rec_or_socp, s: float) -> float:
    if isinstance(rec_or_socp, SocpKey):
        r = rec_or_socp.result
        return float(np.interp(s, r.s, r.t)) if s <= r.s[-1] else np.nan
    x = rec_or_socp.x
    sm = _monotone_s(x[:, 0])
    return float(np.interp(s, sm, np.arange(len(x)) * DT)) if s <= sm[-1] else np.nan


def v3_objective(bank: StimulusBank, cells: list[Cell], cache: KeyCache | None = None, *,
                 tables: ModelTables | None = None, band: float = BAND_EXPECTED,
                 q_cv: float = Q_CV_DEFAULT, log=None) -> ValidationResult:
    """Objective effect (TO-OCP vs min-time in actuation) and discretisation effect (that vs
    s-OCP) on d_b, v_e, v_min and dt (time to s_c1 + 300 m); blocks only if the objective
    effect on v_e (relative) exceeds b / 2, in which case P1 reports both keys
    (``v3_verdict``; the validate stage re-judges every cell at V8's b). A cell whose
    nominal or actuation key is not verified is not judged: it is listed as unevaluated and
    V3 does not pass. Multi-basin keys (6.3) are listed, with the objective effect of every
    start in the actuation key's near-equal set."""
    det = {}
    for c in cells:
        nom = full_recipe(bank, c, cache=cache, tables=tables, q_cv=q_cv)
        act = actuation_key(bank, c, cache, tables=tables, nominal=nom, q_cv=q_cv)
        soc = _socp_for(_problem(bank, c, "nominal", 0, tables=tables), cache)
        if nom.status != "verified" or act.status != "verified":
            det[c.cell_id] = {"status": (nom.status, act.status)}
            continue
        s_ref = bank.stimuli[c.stim_id].landmarks.s_c1 + STAGE_SKIP_MARGIN
        vals = {}
        for name, src in (("nominal", nom), ("actuation", act), ("socp", soc)):
            m = src.measures
            vals[name] = {"d_b": m.get("d_b", np.nan), "v_e": m.get("v_e", np.nan),
                          "v_min": m.get("v_min", np.nan), "t": _t_at(src, s_ref)}
        obj = {k: vals["actuation"][k] - vals["nominal"][k] for k in vals["nominal"]}
        disc = {k: vals["socp"][k] - vals["actuation"][k] for k in vals["nominal"]}
        e_obj = vals["actuation"]["v_e"] / vals["nominal"]["v_e"] - 1.0
        e_n = {s.name: s.measures.get("v_e", np.nan) / vals["nominal"]["v_e"] - 1.0
               for s in act.starts if s.name in act.near_equal}
        det[c.cell_id] = {"values": vals, "objective": obj, "discretisation": disc,
                          "objective_e_ve": e_obj, "objective_e_ve_near_equal": e_n,
                          "multi_basin": {"nominal": bool(nom.multi_basin),
                                          "actuation": bool(act.multi_basin)}}
        if log is not None:
            log.step(note=f"V3 cell {c.cell_id}")
    v = v3_verdict(det, band)
    return ValidationResult("V3", v["passed"], {"cells": det, **{k: v[k] for k in (
        "blocking", "unevaluated", "multi_basin", "band")}},
        "P1 reports both keys (objective effect on v_e > b / 2); unevaluated cells: re-solve")


def v3_verdict(cells: dict, band: float) -> dict:
    """V3's verdict at the band ``band`` over per-cell details (``v3_objective``'s, also
    after a JSON round trip: str keys, None for nan). blocking: cells whose objective effect
    on v_e exceeds band / 2; unevaluated: cells without a finite effect (a key not verified);
    multi_basin: cells either of whose keys is multi-basin (6.3: listed). Passes only when
    some cell was judged, none is unevaluated and none blocks."""
    block, uneval, mb = [], [], []
    for cid, d in cells.items():
        e = (d or {}).get("objective_e_ve")
        e = np.nan if e is None else float(e)
        if not np.isfinite(e):
            uneval.append(cid)
            continue
        if abs(e) > band / 2.0:
            block.append(cid)
        if any(bool(x) for x in ((d or {}).get("multi_basin") or {}).values()):
            mb.append(cid)
    return {"passed": bool(cells) and not uneval and not block, "blocking": block,
            "unevaluated": uneval, "multi_basin": mb, "band": float(band)}


def v4_spread(records: list[KeyRecord]) -> ValidationResult:
    """Spread across starts (reported) and the multi-basin cells (6.3); never blocks."""
    mb = sorted({r.cell_id for r in records if r.multi_basin})
    spreads = {m: [r.spread.get(m, np.nan) for r in records if r.status == "verified"]
               for m in MULTI_BASIN_FLOORS}
    q = {m: (float(np.nanmax(v)) if np.isfinite(v).any() else np.nan) for m, v in
         ((m, np.asarray(v, float)) for m, v in spreads.items()) if len(v)}
    return ValidationResult("V4", True, {"multi_basin_cells": mb, "max_spread": q,
                                         "n_logged": sum(len(r.logged) for r in records),
                                         "n_starts": {r.key_id: len(r.starts) for r in records}},
                            "multi-basin cells leave confirmatory key-based errors")


def v5_mirror(bank: StimulusBank, cells: list[Cell], cache: KeyCache | None = None, *,
              tables: ModelTables | None = None, log=None) -> ValidationResult:
    """Left and right solves: return within 1e-6 relative, measures within 1e-3. Passing sets
    the cache flag that allows ``key_for`` to mirror left-hand keys (the mirror shortcut)."""
    by_pair: dict[int, dict[int, Cell]] = {}
    for c in cells:
        by_pair.setdefault(c.pair_id, {})[int(bank.stimuli[c.stim_id].direction)] = c
    det, bad = {}, []
    for pid, mem in sorted(by_pair.items()):
        if 1 not in mem or -1 not in mem:
            continue
        rl = full_recipe(bank, mem[1], cache=cache, tables=tables)
        rr = full_recipe(bank, mem[-1], cache=cache, tables=tables)
        if rl.status != "verified" or rr.status != "verified":
            det[pid] = {"status": (rl.status, rr.status)}
            bad.append(pid)
            continue
        rel = abs(rl.J - rr.J) / abs(rl.J)
        dm = {}
        for k in ("d_b", "v_e", "v_min", "s_vmin_rel", "line_apex_rel"):
            a, b = rl.measures.get(k, np.nan), rr.measures.get(k, np.nan)
            dm[k] = 0.0 if (np.isnan(a) and np.isnan(b)) else float(abs(a - b))
        ok = rel <= 1e-6 and all(v <= 1e-3 for v in dm.values())
        det[pid] = {"rel_J": rel, "measures": dm, "passed": ok}
        if not ok:
            bad.append(pid)
        if log is not None:
            log.step(note=f"V5 pair {pid}")
    passed = not bad and bool(det)
    if cache is not None:
        cache.set_flag("mirror_ok", passed)
    return ValidationResult("V5", passed, {"pairs": det, "failed": bad},
                            "blocks the mirror shortcut")


def _group_key(r: KeyRecord, measure: str = "v_e") -> tuple:
    """V6 compares keys only at the same approach: theta, width, approach, ramp, no-cap,
    E3 inside width, kind, phase, window and direction; and at the same v0 and D except
    v_e* and v_min* at A2. A2's v0 = v_typ steps with the R bin and its D comes from the
    layout, which moves d_b* with v0 as much as with R, so d_b* is compared only at a fixed
    (v0, D); v_e* and v_min* are set by the corner wherever the key brakes. REC and tier B
    cells hold several v0 per approach."""
    lb = r.labels

    def rnd(x, nd=6):
        return None if x is None else round(float(x), nd)

    key = (lb.get("theta_deg"), lb.get("width"), lb.get("approach"), lb.get("ramp_kind"),
           lb.get("no_cap"), lb.get("w_in_corner"), r.kind, r.phase, int(r.W),
           int(r.direction))
    if measure == "d_b" or lb.get("approach") != "A2":
        key += (rnd(lb.get("v0")), rnd(lb.get("D"), 3))
    return key


def _monotone_violations(records: list[KeyRecord]) -> list[dict]:
    out = []
    for m, sign in (("v_e", 1.0), ("v_min", 1.0), ("d_b", -1.0)):
        groups: dict[tuple, list[KeyRecord]] = {}
        for r in records:
            if r.status == "verified" and r.labels.get("R") is not None:
                groups.setdefault(_group_key(r, m), []).append(r)
        for g, rs in groups.items():
            rs = sorted(rs, key=lambda r: r.labels["R"])
            for a, b in itertools.pairwise(rs):           # a: smaller R, b: larger R
                if abs(float(b.labels["R"]) - float(a.labels["R"])) < 1e-9:
                    continue                              # same R: no order to check
                va, vb = a.measures.get(m, np.nan), b.measures.get(m, np.nan)
                if not (np.isfinite(va) and np.isfinite(vb)):
                    continue
                tau = max(a.tau.get(m, MULTI_BASIN_FLOORS[m]), b.tau.get(m, MULTI_BASIN_FLOORS[m]))
                if sign * (vb - va) < -tau:
                    out.append({"group": list(g), "measure": m, "R_small": a.labels["R"],
                                "R_large": b.labels["R"], "small": a.key_id, "large": b.key_id,
                                "values": (va, vb), "tau": tau})
    return out


def v6_monotone(records: list[KeyRecord], bank: StimulusBank | None = None,
                cells: list[Cell] | None = None, cache: KeyCache | None = None, *,
                tables: ModelTables | None = None) -> tuple[ValidationResult, list[KeyRecord]]:
    """At fixed (theta, w, approach): v_e*, v_min* non-decreasing and d_b* non-increasing in
    R beyond tau. With ``bank`` and ``cells``, each violating pair is re-solved warm from its
    neighbour (the smaller-R cell from the larger-R key and vice versa, keeping the better
    verified record); persistent violations are listed. Returns (result, records after the
    re-solves)."""
    viol = _monotone_violations(records)
    recs = {r.key_id: r for r in records}
    resolved = []
    if viol and bank is not None and cells is not None:
        by_cell = {c.cell_id: c for c in cells}
        for v in viol:
            a, b = recs[v["small"]], recs[v["large"]]
            for tgt, nb in ((a, b), (b, a)):
                c = by_cell.get(tgt.cell_id)
                if c is None:
                    continue
                new = full_recipe(bank, c, tgt.kind, tgt.phase, neighbour=nb, add_q_start=False,
                                  tables=tables, W=tgt.W, force=True, cache=None)
                if new.status == "verified" and new.J > tgt.J + 1e-12 * abs(tgt.J):
                    new.flags.append(f"V6 re-solve warm from {nb.key_id}")
                    new.starts = list(tgt.starts) + list(new.starts)
                    new.key_id = tgt.key_id
                    recs[tgt.key_id] = _store(new, cache)
                    resolved.append(tgt.key_id)
        viol = _monotone_violations(list(recs.values()))
    return (ValidationResult("V6", not viol, {"violations": viol, "re_solved": resolved},
                             "re-solve warm-started from neighbours; list persistent violations"),
            list(recs.values()))


def v7_phase(records: list[KeyRecord]) -> ValidationResult:
    """Phases u_1-u_3: v_e* and v_min* within 0.5 km/h of u_0's; onsets reported (expected
    within one control step)."""
    by: dict[int, dict[int, KeyRecord]] = {}
    for r in records:
        if r.status == "verified" and r.kind in ("nominal", "phase"):
            by.setdefault(r.cell_id, {})[r.phase] = r
    det, bad = {}, []
    for cid, ph in sorted(by.items()):
        if 0 not in ph or len(ph) < 2:
            continue
        r0 = ph[0].measures
        rows = {}
        for k, r in sorted(ph.items()):
            if k == 0:
                continue
            m = r.measures
            rows[k] = {"dv_e": m.get("v_e", np.nan) - r0.get("v_e", np.nan),
                       "dv_min": m.get("v_min", np.nan) - r0.get("v_min", np.nan),
                       "dd_b": m.get("d_b", np.nan) - r0.get("d_b", np.nan)}
        ok = all(abs(v["dv_e"]) <= kph(0.5) and abs(v["dv_min"]) <= kph(0.5) for v in rows.values())
        det[cid] = {"phases": rows, "passed": ok}
        if not ok:
            bad.append(cid)
    return ValidationResult("V7", not bad and bool(det), {"cells": det, "failed": bad},
                            "phase-matched keys for every braking cell")


def band_from_costs(costs: dict[float, float]) -> tuple[float, bool]:
    """b from the median far-field time cost per excess e (V8): the e where the cost reaches
    0.10 s, by linear interpolation through (0, 0) and the measured points, rounded to 0.5
    percentage points and clipped to [5%, 10%]. Returns (b, reached)."""
    es = np.r_[0.0, sorted(costs)]
    cs = np.r_[0.0, [costs[e] for e in sorted(costs)]]
    reached = bool(np.any(cs >= V8_TIME_COST))
    if reached:
        k = int(np.argmax(cs >= V8_TIME_COST))
        e = float(es[k - 1] + (V8_TIME_COST - cs[k - 1]) * (es[k] - es[k - 1]) / (cs[k] - cs[k - 1]))
    else:
        e = BAND_LIMITS[1]
    e = round(e / BAND_ROUND) * BAND_ROUND
    return float(np.clip(e, *BAND_LIMITS)), reached


def v8_band(bank: StimulusBank, cells: list[Cell], cache: KeyCache | None = None, *,
            tables: ModelTables | None = None, p: VehicleParams = F1_2025,
            log=None) -> tuple[ValidationResult, float]:
    """(result, b): on braking cells, the pinned s-OCP with vx(s_c0) <= (1 - e) v_e^sOCP for e
    in {5, 7.5, 10}%; the far-field time cost is the arrival-time difference at the s-OCP's
    end (s_c1 + 600 m); b = the e where the median cost reaches 0.10 s (``band_from_costs``)."""
    det, per_e, failed = {}, {e: [] for e in V8_EXCESS}, {}
    for c in cells:
        prob = _problem(bank, c, "nominal", 0, p, tables=tables)
        free = _socp_for(prob, cache)
        if not free.result.success:
            failed[c.cell_id] = "free s-OCP failed: " + str(free.result.status)
            continue
        if not free.measures.get("has_brake"):
            continue
        lm = prob.landmarks
        v_e = float(np.interp(lm.s_c0, free.result.s, free.result.z[:, 2]))
        row = {"v_e": v_e, "t_free": float(free.result.t[-1]), "cost": {}, "v_e_con": {}}
        for e in V8_EXCESS:
            con = _socp_for(prob, cache, vx_upper={lm.s_c0: (1.0 - e) * v_e})
            if con.result.success:
                row["cost"][e] = float(con.result.t[-1] - free.result.t[-1])
                row["v_e_con"][e] = float(np.interp(lm.s_c0, con.result.s, con.result.z[:, 2]))
            else:
                failed[c.cell_id] = f"e = {e}: {con.result.status}"
        det[c.cell_id] = row
        # A cell enters the medians only with every e solved, so each median is over the
        # same cells; failures are listed.
        if c.cell_id not in failed:
            for e in V8_EXCESS:
                per_e[e].append(row["cost"][e])
        if log is not None:
            log.step(note=f"V8 cell {c.cell_id}")
    med = {e: float(np.median(v)) for e, v in per_e.items() if v}
    if not med:
        return ValidationResult("V8", False, {"cells": det, "failed": failed,
                                              "reason": "no braking cell solved"}), BAND_EXPECTED
    b, reached = band_from_costs(med)
    return ValidationResult("V8", True, {"cells": det, "median_cost": med, "b": b,
                                         "reached": reached, "failed": failed,
                                         "n_cells": len(per_e[V8_EXCESS[0]])}), b


def _slip_count(rec: KeyRecord, limit: float, name: str) -> int:
    """States within 0.5 deg of ``limit``, from the record's active-set counts or its
    states."""
    if name in rec.active:
        return int(rec.active[name])
    return slip_steps(rec.x, limit) if rec.x is not None else 0


def v9_backoff(nominal: list[KeyRecord], backoff: list[KeyRecord]) -> ValidationResult:
    """v_e*bo <= v_e* and v_min*bo <= v_min* on every braking cell; fewer slip-active steps,
    both keys counted on one scale: states with |alpha| within 0.5 deg of the backed-off
    limit 8.5 deg, i.e. >= 8.0 deg (strictly fewer when the nominal key has any, else
    none). Counting against the nominal 10 deg limit could never fail (the backed-off key is
    bounded at 8.5 deg); counts against each key's own limit are reported."""
    nom = {(r.cell_id, r.phase): r for r in nominal if r.status == "verified"}
    det, bad = {}, []
    tol = 1e-6
    for b in backoff:
        n = nom.get((b.cell_id, b.phase))
        if n is None or b.status != "verified" or not n.measures.get("has_brake"):
            continue
        ns = _slip_count(n, BACKOFF_ALPHA_MAX, "n_slip_backoff_limit")
        bs = _slip_count(b, BACKOFF_ALPHA_MAX, "n_slip_backoff_limit")
        own = (_slip_count(n, F1_2025.alpha_max, "n_slip_nominal_limit"),
               int(b.active.get("n_slip_f", 0)) + int(b.active.get("n_slip_r", 0)))
        ok_v = (b.measures["v_e"] <= n.measures["v_e"] + tol
                and b.measures["v_min"] <= n.measures["v_min"] + tol)
        ok_s = bs < ns if ns > 0 else bs == 0
        det[b.cell_id] = {"v_e": (n.measures["v_e"], b.measures["v_e"]),
                          "v_min": (n.measures["v_min"], b.measures["v_min"]),
                          "slip_steps_8deg": (ns, bs), "slip_steps_own_limit": own,
                          "passed": ok_v and ok_s}
        if not (ok_v and ok_s):
            bad.append(b.cell_id)
    return ValidationResult("V9", not bad and bool(det), {"cells": det, "failed": bad}, "investigate")


def _corner_strip(R: float = 40.0, theta_deg: float = 90.0, n: int = 301, s_c: float = 200.0,
                  l_r: float = 20.0, width: float = 12.0) -> tuple[Track, Landmarks]:
    """A short open strip with one trapezoid corner (protocol 3.1 profile, no cap needed at
    these radii), for the solver checks."""
    import jax.numpy as jnp

    from apexzero.tracks.generator import strip_arrays, to_track

    th = np.radians(theta_deg)
    kp = min(1.0 / R, th / l_r)
    L_p = max(th / kp - l_r, 0.0)
    u = np.arange(n) * DS - s_c
    prof = np.clip(np.where(u < l_r, u / l_r, np.where(u < l_r + L_p, 1.0, (2 * l_r + L_p - u) / l_r)), 0, 1)
    k = kp * prof
    w = np.full(n, width / 2)
    _x64()
    tr = strip_arrays(jnp.asarray(k, jnp.float64), jnp.asarray(w, jnp.float64),
                      jnp.asarray(w, jnp.float64), ds=DS)
    L_c = L_p + 2 * l_r
    lm = Landmarks(s_c, s_c + l_r, s_c + L_c / 2, s_c + L_c - l_r, s_c + L_c)
    return to_track(tr, name=f"v10/R{R:g}/T{theta_deg:g}"), lm


def v10_solver_copies(strip_length: float = 600.0, window_steps: int = 24) -> ValidationResult:
    """solve_min_time_pinned(z_start=None) == repo solver to 1e-8 on a 600 m strip;
    AnchoredSolver(a0_fixed=None) == WindowSolver.solve to 1e-10 (a short window in CasADi's
    VM, the same code path as the compiled one); a0_fixed holds exactly; pins hold to 1e-9.
    Validation-sized: well under a CPU-minute plus three small s-OCPs."""
    from apexzero.ocp.min_time import solve_min_time

    _x64()
    det: dict = {}
    track, lm = _corner_strip(n=int(strip_length / DS) + 1)
    ref = solve_min_time(track, F1_2025, ds=DS)
    cp = solve_min_time_pinned(track, F1_2025, ds=DS)
    det["min_time_copy"] = {k: float(np.max(np.abs(np.asarray(getattr(ref, k)) - np.asarray(getattr(cp, k)))))
                            for k in ("z", "u", "t", "kappa", "w_left", "w_right")}
    det["min_time_copy"]["lap_time"] = abs(ref.lap_time - cp.lap_time)
    det["min_time_success"] = (ref.success, cp.success)
    ok_copy = ref.success and cp.success and max(det["min_time_copy"].values()) <= 1e-8
    x0 = np.array([101.37, 0.4, 0.01, 52.0, 0.0, 0.0, 0.0, 0.5])      # off the 2 m grid
    soc = socp_from_state(track, x0, lm.s_c1 + 150.0)
    det["pin"] = dict(soc.pin_check, success=soc.result.success, s0=float(soc.result.s[0]),
                      t0=float(soc.result.t[0]))
    ok_pin = soc.result.success and max(soc.pin_check.values()) <= PIN_TOL
    spec = WindowSpec(steps=window_steps, radius=30.0)
    xw = np.array([150.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0])
    guess = socp_guess(soc, track, xw, window_steps)
    parent = WindowSolver(spec, compile=False).solve(track, xw, guess, ctx=key_ctx())
    child = AnchoredSolver(spec, compile=False)
    copy_ = solve_from_state(child, track, xw, None, guess, ctx=key_ctx())
    det["window_copy"] = {"x": float(np.max(np.abs(parent.x - copy_.x))),
                          "a": float(np.max(np.abs(parent.a - copy_.a))),
                          "J": abs(parent.value - copy_.value), "success": (parent.success, copy_.success)}
    ok_win = parent.success and copy_.success and max(det["window_copy"]["x"], det["window_copy"]["a"],
                                                      det["window_copy"]["J"]) <= 1e-10
    a0 = np.array([0.12, -0.35])
    fixed = solve_from_state(child, track, xw, a0, guess, ctx=key_ctx())
    det["a0_fixed"] = {"da0": float(np.max(np.abs(fixed.a[0] - a0))), "success": fixed.success,
                       "verified": verify_window(fixed, track, xw, ctx=key_ctx())}
    ok_a0 = fixed.success and det["a0_fixed"]["da0"] <= 1e-12 and det["a0_fixed"]["verified"]
    det["passed"] = {"min_time_copy": ok_copy, "pin": ok_pin, "window_copy": ok_win, "a0_fixed": ok_a0}
    return ValidationResult("V10", ok_copy and ok_pin and ok_win and ok_a0, det, "blocks")


def v11_active_set(records: list[KeyRecord]) -> ValidationResult:
    """(13, 180) at A1 is flagged slip-limited and not lock-limited."""
    sel = [r for r in records if r.status == "verified" and r.kind == "nominal" and r.phase == 0
           and r.labels.get("R") is not None and abs(r.labels["R"] - 13.0) < 1e-6
           and abs(r.labels.get("theta_deg", 0.0) - 180.0) < 1e-6 and r.labels.get("approach") == "A1"]
    det = {r.key_id: dict(r.active) for r in sel}
    ok = bool(sel) and all(r.active.get("slip_limited") and not r.active.get("lock_limited") for r in sel)
    return ValidationResult("V11", ok, {"keys": det, "found": len(sel)}, "investigate the logging")


# ------------------------------------------------------------------------- budget

# Protocol 10's key items: label -> (timing item, priority). Timings are CPU-s per unit.
BUDGET_ITEMS = {
    "layout_socp": ("socp", "P0"), "v8_socp": ("socp", "P0"), "validation": ("nominal", "P0"),
    "nominal_A": ("nominal", "P0"), "nominal_E1": ("nominal", "P0"), "null": ("null", "P0"),
    "backoff": ("warm", "P0"), "phase": ("warm", "P0"), "anchored_G": ("anchored", "P0"),
    "viability": ("viability", "P0"),
    "E1_twins": ("nominal", "P1"), "E2": ("nominal", "P1"), "E3": ("nominal", "P1"),
    "tierD_socp": ("socp", "P1"), "anchored_S": ("anchored", "P1"), "anchored_B": ("anchored", "P1"),
    "tierC": ("nominal", "P2"), "recovery": ("nominal", "P2"), "anchored_ckpt": ("anchored", "P2"),
    # V7's remedy (6.5, V7 failed on speeds): phase-matched keys for the braking cells of the
    # P1 and P2 nominal-key items too, each at the priority of its nominal key.
    "phase_P1": ("warm", "P1"), "phase_P2": ("warm", "P2"),
    # Ruling R17c: tier B D = 600 cells (the band reads e_ve^RA there) get their nominal keys
    # in the keys stage (cold, one per cell solved), and under V7's remedy phase-matched keys
    # at their phases (warm from the nominal key), so per-phase re-anchored keys (R17a) start
    # from and fall back to the key at their own phase.
    "nominal_B600": ("nominal", "P1"), "phase_B600": ("warm", "P1"),
    # Ruling R18: the replicates' finals get G1-G3's re-anchored keys (P0: A1; P1: tier B
    # D = 600) and their own viability solves (9.1's cap applied to them apart), never P2's
    # checkpoint keys (P6 stays on seed 0).
    "anchored_R": ("anchored", "P0"), "viability_R": ("viability", "P0"),
    "anchored_B_R": ("anchored", "P1"),
}
# Budget rule 2's cuts: tier A 180 deg cells with R >= 35.8 (3 radii <= 55.9 x A1, A2, each a
# nominal and a backed-off key; ruling R7 corrects the protocol's "5 geometries" and keeps the
# cells in the rollouts with no_key) and the phase-matched keys of the 90 deg cells (A1,
# R <= 18.3: 10 geometries x 3 phases).
_N_180_LARGE = 2 * sum(1 for r in R_BRAKING if 35.8 <= r <= R_180_MAX)
_N_PHASE_90 = 3 * sum(1 for r in R_BRAKING if r <= 18.3)
DEFAULT_CUTS = {"nominal_A": _N_180_LARGE, "backoff": _N_180_LARGE, "phase": _N_PHASE_90}


def project_budget(timings: dict[str, float], counts: dict[str, int | tuple],
                   cut_counts: dict[str, int] | None = None, workers: int = KEY_WORKERS) -> dict:
    """Projected CPU-hours per stage and the budget rule's cuts (protocol 10). ``timings``:
    CPU-s per unit of each timing item ('socp', 'nominal', 'warm', 'null', 'anchored',
    'viability', 'actuation') and optionally 'rollout_sps'; ``counts``: label -> n for the
    labels in ``BUDGET_ITEMS``, or label -> (timing item, priority, n) for others.
    Rule 1: total > 60 CPU-h drops P2. Rule 2: P0 > 35 CPU-h cuts ``cut_counts``
    (default ``DEFAULT_CUTS``). Rule 3: rollouts below 15k steps/s drop checkpoint stochastic
    and tier E matched-noise runs."""
    items = {}
    for label, v in counts.items():
        item, prio, n = (*BUDGET_ITEMS[label], v) if not isinstance(v, tuple) else v
        each = float(timings[item])
        items[label] = {"item": item, "priority": prio, "n": int(n), "cpu_s_each": each,
                        "cpu_h": int(n) * each / 3600.0}

    def totals():
        t = {pr: sum(i["cpu_h"] for i in items.values() if i["priority"] == pr) for pr in ("P0", "P1", "P2")}
        t["total"] = sum(t.values())
        return t

    before = totals()
    cuts = []
    if before["total"] > BUDGET_TOTAL_CPU_H:
        for i in items.values():
            if i["priority"] == "P2":
                i["n"], i["cpu_h"] = 0, 0.0
        cuts.append("rule 1: P2 items dropped (projected total > 60 CPU-h)")
    if before["P0"] > BUDGET_P0_CPU_H:
        for label, n_cut in (cut_counts or DEFAULT_CUTS).items():
            if label in items:
                i = items[label]
                i["n"] = max(i["n"] - int(n_cut), 0)
                i["cpu_h"] = i["n"] * i["cpu_s_each"] / 3600.0
        cuts.append("rule 2: tier A 180 deg R >= 35.8 dropped from the keys (3 geometries x 2 "
                    "approaches, R7: still rolled out, no_key); phase-matched keys for 180 deg "
                    "only")
    if "rollout_sps" in timings and timings["rollout_sps"] < MIN_ROLLOUT_SPS:
        cuts.append("rule 3: checkpoint stochastic and tier E matched-noise runs dropped")
    after = totals()
    return {"items": items, "before": before, "after": after, "cuts": cuts,
            "wall_h": {k: v / max(workers, 1) for k, v in after.items()}, "workers": workers}
