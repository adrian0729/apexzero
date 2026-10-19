"""Response measures: one definition for agents, controls and every kind of key.

Protocol 5 (measures and model tables), 9.1 (crash typing) and the step-3 checks. Every
measure is computed by ``response_metrics(traj, marks, null)`` from a common ``Trajectory``,
built by adapters from an agent or control rollout, a TO-OCP solution (20 Hz), its float64
environment replay, or an s-OCP solution (states on the s grid with t integrated).
Quantities are interpolated linearly in s; the pedal is always the state x[7], the command
only where stated (the decision point s_dec).

Braking is read from the front brake *demand* beta_f = max(0, -fx_f,dem) / fx_max,f, not
from the pedal, so the onset means the same thing at every speed and for agent and key
alike. ``front_brake_demand`` re-implements the demand lines of ``dynamics.tyre_forces``; a
test checks fx_f,dem * scale == tyre_forces(x).fx_f to 1e-10.

Model tables (built once, cached under ``outputs/psychophysics/tables/``):
- a_dec(v): largest straight-line deceleration (tyres + drag), pedal swept over [-1, 0] in
  0.005 steps with the other states at 0, on a 1 km/h grid;
- a_y,max(v): largest steady lateral (centripetal) acceleration |v| r with |alpha_f|,
  |alpha_r| <= alpha_max, by a grid search over (delta, sideslip) with the pedal set for
  zero longitudinal acceleration, on a 5 km/h grid from 20 to 315 km/h. For each grid
  point the yaw rate (yaw-moment balance) and the pedal (dvx = 0) are solved, steady states
  are the sign changes of dvy along the sideslip axis (a Milliken moment-method trim), and
  the best one is polished by Newton on (sideslip, r, pedal) over a fine delta bracket. The
  steering lock and the engine power are part of the model, so the table is lock-limited
  below about 50 km/h and power-limited above about 265 km/h (``limit_y``);
- d_b^phys(v_b, v_e) = int_{v_e}^{v_b} v / a_dec(v) dv;  v_ss(rho) solves v^2 / rho = a_y,max(v).

Trajectory indexing: state k is at t[k] (t[0] = 0, the initial state); step k maps state k
to state k + 1 under command a[k]; per-step arrays have length K - 1. A terminated run keeps
its terminal state (``valid`` is True for it) and nothing after; measures after a crash are
censored, never imputed: a point measure needs its point reached, a window measure needs
the window's end reached. Channels a source does not record (stochastic traces keep only s,
vx, n, xi, pedal, util and beta_f) are NaN, and measures that need them come out NaN.
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass, field, fields
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from apexzero.interp import io
from apexzero.interp.constants import (
    A_DEC_PEDAL_STEP,
    A_DEC_V_STEP_KPH,
    A_Y_V_GRID_KPH,
    BRAKE_MIN_STEPS,
    BRAKE_OFF,
    BRAKE_ON,
    COMPLETE_AFTER_EXIT,
    CRASH_LOOKBACK,
    CRASH_UTIL,
    CRASH_UTIL_STEPS,
    DS,
    DT,
    DV_AFTER,
    DV_BEFORE,
    DV_STEP,
    ETA_OFFSET,
    FF_FRACTION,
    FLAG_BITS,
    FORCED_ONSET,
    KPH,
    LEGACY_PEDAL,
    LEGACY_STEPS,
    LINE_RMS_MARGIN,
    LINE_STEER_UTIL,
    OBS_CLIP,
    OBS_GROUPS,
    OSC_WINDOW,
    RESP_DELTA,
    RESP_STEPS,
    UNSETTLED_N,
    UNSETTLED_PEDAL,
    V_MIN_MARGIN,
    V_TOP_KPH,
)
from apexzero.interp.stimuli import Landmarks
from apexzero.tracks.spline import packed_coeffs, spline_np
from apexzero.vehicle import dynamics
from apexzero.vehicle.ops import JAX, Ops
from apexzero.vehicle.params import F1_2025, VehicleParams, with_context

if TYPE_CHECKING:   # keeps casadi (ocp.*) out of rollout processes that import this module
    from apexzero.ocp.min_time import OCPResult
    from apexzero.ocp.window import WindowResult
    from apexzero.tracks.track import Track

STATE_NAMES = ("s", "n", "xi", "vx", "vy", "r", "delta", "pedal")
EDGE_MARGIN = 1.0             # racing.EnvParams.edge_margin: the "- 1" of line eta
GAMMA = 0.995                 # WindowSpec.gamma / PPO discount of the window return
ACTION_COST = 0.02            # racing.EnvParams.action_cost
PROGRESS_SCALE = 10.0         # racing.EnvParams.progress_scale
TABLES_DIR = io.OUT / "tables"

# NumPy twin of ops.JAX, so the shared vehicle functions run on float64 arrays here.
NP = Ops(name="numpy", sin=np.sin, cos=np.cos, tan=np.tan, atan=np.arctan, tanh=np.tanh,
         sqrt=np.sqrt, stack=lambda xs: np.stack(np.broadcast_arrays(*xs), axis=-1),
         index=lambda x, i: x[..., i], where=np.where)

# ------------------------------------------------------------------------- car and demand


def car_params(p: VehicleParams = F1_2025, ctx: np.ndarray | None = None,
               hidden_mu: float = 1.0) -> VehicleParams:
    """The parameters the env drives with (``racing.params_from_context``) in float64:
    ``ctx`` None means ``racing.default_context()``, i.e. the base car's own values."""
    if ctx is None:
        cla = p.cla_f + p.cla_r
        c = [p.m, p.p_max, cla, p.cda, p.mu_scale, p.cla_f / cla]
    else:
        c = [float(v) for v in np.asarray(ctx, np.float64)]
    return with_context(p, c[0], c[1], c[2], c[3], c[4] * hidden_mu, c[5])


def _mu(mu0, fz_tyre, p: VehicleParams, ops: Ops):
    """dynamics._mu, copied (load-sensitive friction, floored smoothly at 0.3)."""
    return ops.smax(p.mu_scale * (mu0 + p.kmu * (fz_tyre - p.fz0)), 0.3, 1e-2)


def front_brake_demand(x, p: VehicleParams = F1_2025, ops: Ops = JAX):
    """(beta_f, fx_f_dem, fx_max_f) at state(s) x [..., 8]: the front brake demand
    -brake * f_brake_max * brake_bias (brake = relu(-pedal) tanh(vx / 0.5)), the front axle's
    longitudinal capacity mux_f(fz_f / 2) fz_f with the same load transfer as
    ``tyre_forces``, and beta_f = max(0, -fx_f_dem) / fx_max_f (protocol 5). Works with the
    JAX ops (rollouts, vmappable) and ``NP`` (measures). The lines are those of
    ``dynamics.tyre_forces`` verbatim, so fx_f_dem * scale reproduces its fx_f exactly."""
    vx, pedal = ops.index(x, 3), ops.index(x, 7)
    L = p.lf + p.lr
    v2 = vx * vx

    fz_f0 = p.m * p.g * p.lr / L + 0.5 * p.rho * p.cla_f * v2
    fz_r0 = p.m * p.g * p.lf / L + 0.5 * p.rho * p.cla_r * v2
    f_drag = 0.5 * p.rho * p.cda * v2
    f_roll = p.f_roll * (fz_f0 + fz_r0) * ops.tanh(vx / 0.5)

    throttle = ops.relu(pedal)
    brake = ops.relu(-pedal) * ops.tanh(vx / 0.5)
    f_drive = throttle * p.p_max / ops.smax(vx, p.v_drive_floor, 0.5)
    fx_f_dem = -brake * p.f_brake_max * p.brake_bias
    fx_r_dem = f_drive - brake * p.f_brake_max * (1.0 - p.brake_bias)

    mux_f0 = _mu(p.mux_f, 0.5 * fz_f0, p, ops)
    mux_r0 = _mu(p.mux_r, 0.5 * fz_r0, p, ops)
    fx_f1 = ops.softsat(fx_f_dem, mux_f0 * fz_f0)
    fx_r1 = ops.softsat(fx_r_dem, mux_r0 * fz_r0)
    ax_est = (fx_f1 + fx_r1 - f_drag - f_roll) / p.m
    dfz = p.m * ax_est * p.h_cog / L
    fz_f = ops.smax(fz_f0 - dfz, 200.0, 10.0)

    fx_max_f = _mu(p.mux_f, 0.5 * fz_f, p, ops) * fz_f
    demand = ops.where(fx_f_dem < 0.0, -fx_f_dem, 0.0 * fx_f_dem)
    return demand / fx_max_f, fx_f_dem, fx_max_f


def _forces(x: np.ndarray, p: VehicleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(util_f, util_r, beta_f) of states [K, 8] (float64, NumPy)."""
    fo = dynamics.tyre_forces(x, p, NP)
    return (np.asarray(fo.util_f, float), np.asarray(fo.util_r, float),
            np.asarray(front_brake_demand(x, p, NP)[0], float))


# ------------------------------------------------------------------------- model tables


@dataclass(frozen=True)
class ModelTables:
    """Speed-dependent limits of the full model (protocol 5). Grids in m/s, accelerations
    in m/s^2 (positive). The ``*_y`` arrays describe the steady state that attains
    a_y,max at each speed; ``limit_y`` is 'lock', 'power', 'slip', 'grip' (a tyre's peak)
    or 'none' (no steady state with a_y > 0: a_y,max = 0)."""

    v_dec: np.ndarray        # [n] m/s, 1 km/h grid from 0 to 315 km/h
    a_dec: np.ndarray        # [n] largest straight-line deceleration, tyres + drag
    pedal_dec: np.ndarray    # [n] pedal achieving it (front saturates at 0.70 at 310 km/h)
    v_lat: np.ndarray        # [m] m/s, 5 km/h grid from 20 to 315 km/h
    a_y_max: np.ndarray      # [m] largest steady lateral acceleration within alpha_max
    alpha_max: float         # rad, slip limit used for a_y_max
    params: dict = field(default_factory=dict)   # VehicleParams._asdict() of the car
    delta_y: np.ndarray | None = None   # [m] rad, steering of that steady state
    beta_y: np.ndarray | None = None    # [m] rad, sideslip
    r_y: np.ndarray | None = None       # [m] rad/s, yaw rate
    pedal_y: np.ndarray | None = None   # [m] pedal (zero longitudinal acceleration)
    alpha_f_y: np.ndarray | None = None  # [m] rad
    alpha_r_y: np.ndarray | None = None  # [m] rad
    limit_y: np.ndarray | None = None   # [m] str

    def a_dec_at(self, v):
        """a_dec(v), linear interpolation (m/s^2); held constant beyond the grid."""
        return np.interp(np.asarray(v, float), self.v_dec, self.a_dec)

    def a_y_at(self, v):
        """a_y,max(v), linear interpolation (m/s^2); held constant beyond the grid."""
        return np.interp(np.asarray(v, float), self.v_lat, self.a_y_max)

    @cached_property
    def a_y_corner(self) -> np.ndarray:
        """a_y,max on ``v_lat`` as a cornering limit (``cornering_limit``): the power-limited
        entries (no steady state at zero longitudinal acceleration for lack of power, 0 at
        315 km/h) are replaced by the tyres' grip extrapolated with the normal load. v_ss and
        the infeasibility bound use it: the power limit is not a lateral limit, and the
        bound must stay a relaxation."""
        p = VehicleParams(**self.params) if self.params else F1_2025
        return cornering_limit(self, p, np.asarray(self.v_lat, float))

    @cached_property
    def _distance(self) -> tuple[np.ndarray, np.ndarray]:
        """D(v) = int_0^v u / a_dec(u) du on a 0.005 m/s grid (trapezoid; the integrand is
        smooth between the 1 km/h nodes, so this is exact to ~1e-6 relative)."""
        u = np.arange(0.0, 1.15 * self.v_dec[-1], 0.005)
        a = self.a_dec_at(u)
        f = np.where(a > 0.0, u / np.where(a > 0.0, a, 1.0), 0.0)
        D = np.r_[0.0, np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(u))]
        return u, D

    def d_b_phys(self, v_b, v_e):
        """Straight-line braking distance (m) from v_b down to v_e: int v / a_dec(v) dv;
        0 where v_e >= v_b, NaN where either is NaN."""
        vb, ve = np.asarray(v_b, float), np.asarray(v_e, float)
        u, D = self._distance
        out = np.interp(vb, u, D) - np.interp(ve, u, D)
        out = np.where(ve >= vb, 0.0, out)
        out = np.where(np.isfinite(vb) & np.isfinite(ve), out, np.nan)
        return float(out) if out.ndim == 0 else out

    def v_ss(self, rho):
        """Steady-state speed (m/s) on radius rho (m): v^2 / rho = a_y,max(v), the first
        crossing on the piecewise-linear table, solved exactly within its segment. Below the
        grid a_y is held at its 20 km/h value; with no crossing (a_y never falls short of
        v^2 / rho) the result is the top of the grid, 315 km/h. NaN for rho <= 0 or NaN.
        The lateral limit is ``a_y_corner`` (power-limited entries replaced by tyre grip)."""
        rho = np.asarray(rho, float)
        v, a = np.asarray(self.v_lat, float), self.a_y_corner
        with np.errstate(invalid="ignore"):
            g = v**2 - rho[..., None] * a
        pos = g >= 0.0
        idx = np.argmax(pos, axis=-1)
        none = ~pos.any(axis=-1)
        i1 = np.maximum(idx, 1)
        i0 = i1 - 1
        k = (a[i1] - a[i0]) / (v[i1] - v[i0])
        c0 = a[i0] - k * v[i0]
        with np.errstate(invalid="ignore"):
            seg = 0.5 * (rho * k + np.sqrt((rho * k) ** 2 + 4.0 * rho * c0))
            below = np.sqrt(rho * a[0])
        out = np.where(idx == 0, below, seg)
        out = np.where(none, v[-1], out)
        out = np.where(np.isfinite(rho) & (rho > 0.0) | np.isposinf(rho), out, np.nan)
        out = np.where(np.isposinf(rho), v[-1], out)
        return float(out) if out.ndim == 0 else out

    def save(self, path: Path) -> Path:
        cols: dict[str, Any] = {"v_dec": self.v_dec, "a_dec": self.a_dec,
                                "pedal_dec": self.pedal_dec, "v_lat": self.v_lat,
                                "a_y_max": self.a_y_max,
                                "alpha_max": np.asarray(self.alpha_max, float),
                                "params": np.asarray(json.dumps(self.params, sort_keys=True))}
        for name in ("delta_y", "beta_y", "r_y", "pedal_y", "alpha_f_y", "alpha_r_y", "limit_y"):
            if getattr(self, name) is not None:
                cols[name] = np.asarray(getattr(self, name))
        return io._atomic(Path(path), lambda fh: np.savez(fh, **cols))

    @classmethod
    def load(cls, path: Path) -> ModelTables:
        with np.load(Path(path), allow_pickle=False) as d:
            kw = {k: d[k] for k in d.files}
        kw["alpha_max"] = float(kw["alpha_max"])
        kw["params"] = json.loads(str(kw["params"]))
        return cls(**kw)


def cornering_limit(tables: ModelTables, p: VehicleParams = F1_2025,
                    v: np.ndarray | None = None) -> np.ndarray:
    """a_y,max(v) on ``v`` (default: the table's grid) as a cornering limit: the protocol
    table where the tyres or the lock limit it, and beyond the last tyre-limited speed the
    tyres' grip extrapolated with the normal load (mu_eff = a_y m / N linear in N, fitted on
    the top six tyre-limited entries). The table's power-limited entries (``limit_y ==
    'power'``; a_y,max = 0 at 315 km/h) are not cornering limits. Without ``limit_y`` the
    entries beyond the table's maximum are treated as power-limited. (Shared with
    ``controls.lateral_limit``, decision C2.)"""
    vl = np.asarray(tables.v_lat, float)
    ay = np.asarray(tables.a_y_max, float)
    v = vl if v is None else np.asarray(v, float)
    lim = getattr(tables, "limit_y", None)
    if lim is not None and len(lim) == len(vl):
        tyre = np.isin(np.asarray(lim).astype(str), ("grip", "slip", "lock")) & (ay > 0)
    else:
        # No limit labels: only a fall after the table's (last) maximum is taken as power.
        last_max = int(np.flatnonzero(ay >= ay.max())[-1])
        tyre = (np.arange(len(vl)) <= last_max) & (ay > 0)
    k = np.nonzero(tyre)[0]
    if not k.size:
        return np.interp(v, vl, ay)
    out = np.interp(v, vl[k], ay[k])
    beyond = v > vl[k[-1]]
    if beyond.any() and k.size >= 2:
        def load(u):
            return p.m * p.g + 0.5 * p.rho * (p.cla_f + p.cla_r) * u**2

        top = k[-6:]
        c1, c0 = np.polyfit(load(vl[top]), ay[top] * p.m / load(vl[top]), 1)
        n_b = load(v[beyond])
        out[beyond] = np.maximum(c0 + c1 * n_b, 0.0) * n_b / p.m
    return out


def _derivs(vx, vy, r, delta, pedal, p: VehicleParams):
    """(dvx, dvy, dr) of the body dynamics at (vx, vy, r, delta, pedal), broadcast."""
    z = np.zeros(np.broadcast(vx, vy, r, delta, pedal).shape)
    x = np.stack(np.broadcast_arrays(z, z, z, vx, vy, r, delta, pedal), axis=-1)
    dvx, dvy, dr, _, _ = dynamics.body_derivatives(x, np.zeros(2), p, NP)
    return dvx, dvy, dr


def _bisect(fun, lo: np.ndarray, hi: np.ndarray, iters: int = 32):
    """Vectorised bisection for fun = 0 on [lo, hi]; ok where the bracket changes sign."""
    flo, fhi = fun(lo), fun(hi)
    ok = np.isfinite(flo) & np.isfinite(fhi) & (np.sign(flo) != np.sign(fhi))
    inc = fhi > flo
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = (fun(mid) < 0.0) == inc
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi), ok


def _trim_grid(V: float, delta: np.ndarray, beta: np.ndarray, p: VehicleParams):
    """On the (delta, beta) grid at speed V: the yaw rate balancing the yaw moment (dr = 0)
    and the pedal giving dvx = 0, both by bisection then a joint Newton polish; returns
    (D, B, r, pedal, dvy, solved). The yaw-rate bracket keeps both slips within
    alpha_max + 0.05 rad, where the Pacejka curves are nearly monotone."""
    D, B = np.meshgrid(delta, beta, indexing="ij")
    vx, vy = V * np.cos(B), V * np.sin(B)
    wide = p.alpha_max + 0.05
    r_lo = np.maximum((vx * np.tan(D - wide) - vy) / p.lf, (vy - vx * np.tan(wide)) / p.lr)
    r_hi = np.minimum((vx * np.tan(D + wide) - vy) / p.lf, (vy + vx * np.tan(wide)) / p.lr)
    ok = r_hi > r_lo
    r_lo, r_hi = np.where(ok, r_lo, 0.0), np.where(ok, r_hi, 1.0)
    ped = np.zeros_like(D)
    one = np.ones_like(D)
    okr = okp = ok
    for _ in range(2):
        r, okr = _bisect(lambda q, ped=ped: _derivs(vx, vy, q, D, ped, p)[2], r_lo, r_hi)
        ped, okp = _bisect(lambda q, r=r: _derivs(vx, vy, r, D, q, p)[0], -one, one)
    h = 1e-7
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(4):
            dvx0, _, dr0 = _derivs(vx, vy, r, D, ped, p)
            dvx_r, _, dr_r = _derivs(vx, vy, r + h, D, ped, p)
            dvx_p, _, dr_p = _derivs(vx, vy, r, D, ped + h, p)
            j00, j01 = (dr_r - dr0) / h, (dr_p - dr0) / h
            j10, j11 = (dvx_r - dvx0) / h, (dvx_p - dvx0) / h
            det = j00 * j11 - j01 * j10
            step_r = (j11 * dr0 - j01 * dvx0) / det
            step_p = (-j10 * dr0 + j00 * dvx0) / det
            good = np.isfinite(step_r) & np.isfinite(step_p)
            r = np.where(good, r - step_r, r)
            ped = np.clip(np.where(good, ped - step_p, ped), -1.0, 1.0)
    dvx, dvy, dr = _derivs(vx, vy, r, D, ped, p)
    solved = ok & okr & okp & (np.abs(dr) < 1e-6) & (np.abs(dvx) < 1e-6)
    return D, B, r, ped, dvy, solved


def _polish(V: float, delta: np.ndarray, beta: np.ndarray, r: np.ndarray, ped: np.ndarray,
            p: VehicleParams, iters: int = 8):
    """Newton on (beta, r, pedal) for dvy = dr = dvx = 0 at fixed delta (vectorised);
    returns (beta, r, pedal, converged)."""
    z = np.stack([beta, r, ped], axis=-1).astype(float)

    def F(z):
        vx, vy = V * np.cos(z[..., 0]), V * np.sin(z[..., 0])
        dvx, dvy, dr = _derivs(vx, vy, z[..., 1], delta, z[..., 2], p)
        return np.stack([dvy, dr, dvx], axis=-1)

    h = 1e-7
    eye = np.eye(3)
    for _ in range(iters):
        F0 = F(z)
        J = np.stack([(F(z + h * eye[j]) - F0) / h for j in range(3)], axis=-1)
        try:
            step = np.linalg.solve(J, F0[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = (np.linalg.pinv(J) @ F0[..., None])[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        z = z - step
        z[..., 2] = np.clip(z[..., 2], -1.0, 1.0)
    conv = np.all(np.abs(F(z)) < 1e-7, axis=-1)
    return z[..., 0], z[..., 1], z[..., 2], conv


def _slips(V, delta, beta, r, ped, p: VehicleParams):
    z = np.zeros(np.broadcast(delta, beta, r, ped).shape)
    x = np.stack(np.broadcast_arrays(z, z, z, V * np.cos(beta), V * np.sin(beta), r, delta,
                                     ped), axis=-1)
    fo = dynamics.tyre_forces(x, p, NP)
    return np.asarray(fo.alpha_f, float), np.asarray(fo.alpha_r, float)


def steady_lateral_max(V: float, p: VehicleParams = F1_2025, d_step: float = 0.005,
                       b_step: float = 0.005, b_max: float = 0.3, refine: int = 21) -> dict:
    """a_y,max at speed V (m/s): the (delta, sideslip) grid search of protocol 5, then a
    Newton polish of the best trim over delta in [delta* - d_step, delta* + d_step]
    (``refine`` points). Returns the steady state that attains it and its limiting factor."""
    delta = np.linspace(0.0, p.delta_max, round(p.delta_max / d_step) + 1)
    beta = np.linspace(-b_max, b_max, round(2 * b_max / b_step) + 1)
    none = {"a_y": 0.0, "delta": np.nan, "beta": np.nan, "r": np.nan, "pedal": np.nan,
            "alpha_f": np.nan, "alpha_r": np.nan, "limit": "none"}
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        D, B, r, ped, dvy, solved = _trim_grid(V, delta, beta, p)
        e = np.where(solved, dvy, np.nan)
        e0, e1 = e[:, :-1], e[:, 1:]
        cross = np.isfinite(e0) & np.isfinite(e1) & (np.sign(e0) != np.sign(e1))
        w = np.where(cross, e0 / np.where(cross, e0 - e1, 1.0), 0.0)

        def lerp(a):
            return a[:, :-1] + w * (a[:, 1:] - a[:, :-1])

        bc, rc, pc = lerp(B), lerp(r), lerp(ped)
        af, ar = _slips(V, D[:, :-1], bc, rc, pc, p)
        good = cross & (np.abs(af) <= p.alpha_max) & (np.abs(ar) <= p.alpha_max)
        if not good.any():
            return none
        ay = np.where(good, V * rc, -np.inf)
        i, j = np.unravel_index(np.argmax(ay), ay.shape)
        best = {"a_y": float(ay[i, j]), "delta": float(D[i, 0]), "beta": float(bc[i, j]),
                "r": float(rc[i, j]), "pedal": float(pc[i, j]), "alpha_f": float(af[i, j]),
                "alpha_r": float(ar[i, j])}
        dd = np.linspace(max(0.0, delta[i] - d_step), min(p.delta_max, delta[i] + d_step), refine)
        n = len(dd)
        bp, rp, pp, conv = _polish(V, dd, np.full(n, bc[i, j]), np.full(n, rc[i, j]),
                                   np.full(n, pc[i, j]), p)
        afp, arp = _slips(V, dd, bp, rp, pp, p)
        okp = conv & (np.abs(afp) <= p.alpha_max) & (np.abs(arp) <= p.alpha_max)
        if okp.any():   # exact steady states supersede the interpolated grid estimate
            k = int(np.argmax(np.where(okp, V * rp, -np.inf)))
            best = {"a_y": float(V * rp[k]), "delta": float(dd[k]), "beta": float(bp[k]),
                    "r": float(rp[k]), "pedal": float(pp[k]), "alpha_f": float(afp[k]),
                    "alpha_r": float(arp[k])}
    if best["a_y"] <= 0.0:
        return none
    if best["delta"] >= p.delta_max - 1e-9:
        limit = "lock"
    elif best["pedal"] >= 0.98:
        limit = "power"
    elif max(abs(best["alpha_f"]), abs(best["alpha_r"])) >= p.alpha_max - 1e-4:
        limit = "slip"
    else:
        limit = "grip"
    return dict(best, limit=limit)


def build_model_tables(p: VehicleParams = F1_2025, speeds_kph: np.ndarray | None = None,
                       d_step: float = 0.005, b_step: float = 0.005, log=None) -> ModelTables:
    """Build a_dec(v) and a_y,max(v) for car ``p`` (protocol 5). a_dec is a 316 x 201 sweep
    (milliseconds); a_y,max costs about 0.4 CPU-s per speed, about 25 CPU-s for the 60-speed
    grid. ``speeds_kph`` overrides the a_y grid (tests)."""
    v_dec = np.arange(round(V_TOP_KPH / A_DEC_V_STEP_KPH) + 1) * A_DEC_V_STEP_KPH * KPH
    ped = np.linspace(-1.0, 0.0, round(1.0 / A_DEC_PEDAL_STEP) + 1)
    dvx = _derivs(v_dec[:, None], 0.0, 0.0, 0.0, ped[None, :], p)[0]
    j = np.argmax(-dvx, axis=1)
    a_dec = -dvx[np.arange(len(v_dec)), j]
    pedal_dec = ped[j]

    if speeds_kph is None:
        lo, hi, step = A_Y_V_GRID_KPH
        speeds_kph = np.arange(lo, hi + 1e-9, step)
    v_lat = np.asarray(speeds_kph, float) * KPH
    rows = []
    for V in v_lat:
        rows.append(steady_lateral_max(float(V), p, d_step=d_step, b_step=b_step))
        if log is not None:
            log.step(note=f"a_y {V / KPH:.0f} km/h")
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return ModelTables(v_dec=v_dec, a_dec=a_dec, pedal_dec=pedal_dec, v_lat=v_lat,
                       a_y_max=col["a_y"].astype(float), alpha_max=float(p.alpha_max),
                       params=dict(p._asdict()), delta_y=col["delta"].astype(float),
                       beta_y=col["beta"].astype(float), r_y=col["r"].astype(float),
                       pedal_y=col["pedal"].astype(float), alpha_f_y=col["alpha_f"].astype(float),
                       alpha_r_y=col["alpha_r"].astype(float), limit_y=col["limit"].astype(str))


def tables_path(p: VehicleParams = F1_2025) -> Path:
    """``outputs/psychophysics/tables/model_tables_<hash of the car>.npz``."""
    return TABLES_DIR / f"model_tables_{io.sha256_json(dict(p._asdict()))[:12]}.npz"


def model_tables(p: VehicleParams = F1_2025, rebuild: bool = False) -> ModelTables:
    """The cached tables of car ``p``, built and saved on first use."""
    path = tables_path(p)
    if path.exists() and not rebuild:
        return ModelTables.load(path)
    tab = build_model_tables(p)
    tab.save(path)
    return tab


# ------------------------------------------------------------------------- trajectories


@dataclass
class Trajectory:
    """One run or one key solution, on a common footing (protocol 5)."""

    source: str                       # 'agent' | 'control' | 'to_ocp' | 'replay' | 'socp'
    t: np.ndarray                     # [K] s, t[0] = 0
    x: np.ndarray                     # [K, 8] states (STATE_NAMES), float64; NaN = not recorded
    valid: np.ndarray                 # [K] bool, a state actually reached (terminal included)
    a: np.ndarray | None = None       # [K-1, 2] applied command (after the env's clip)
    a_mean: np.ndarray | None = None  # [K-1, 2] policy mean (agents, controls)
    reward: np.ndarray | None = None  # [K-1] env reward of step k
    util_f: np.ndarray | None = None  # [K] front friction-ellipse use (tyre_forces.util_f)
    util_r: np.ndarray | None = None  # [K]
    beta_f: np.ndarray | None = None  # [K] front brake demand
    flags: np.ndarray | None = None   # [K-1] uint8, constants.FLAG_BITS of step k
    term_step: int = -1               # step whose transition terminated (-1: none)
    trunc_step: int = -1              # step that truncated (end of strip), -1: none
    clip_frac: np.ndarray | None = None  # [K, 4] fraction |o| >= 10 per OBS_GROUPS
    clip_max: np.ndarray | None = None   # [K, 4] max |o| per group
    log_std: np.ndarray | None = None    # [2] actor log_std (state-independent)
    dt: float | None = 0.05           # control step; None for the s-OCP (irregular t)
    prev_action0: np.ndarray = field(default_factory=lambda: np.zeros(2))  # a_{-1}
    meta: dict = field(default_factory=dict)  # cell_id, phase, mode, sample, key_id, ...

    @property
    def s(self) -> np.ndarray:
        return self.x[:, 0]

    @property
    def vx(self) -> np.ndarray:
        return self.x[:, 3]

    @property
    def pedal(self) -> np.ndarray:
        return self.x[:, 7]

    @property
    def crashed(self) -> bool:
        return self.term_step >= 0


@dataclass(frozen=True)
class Marks:
    """What a trajectory is measured against: the stimulus's landmarks and corridor, and the
    run's start and window. Widths are evaluated with the same C2 spline the env uses."""

    landmarks: Landmarks
    direction: int            # sigma: +1 left corner, -1 right, 0 straight
    s0: float                 # m, initial arc length of this run
    W: int                    # window (steps)
    w_left: np.ndarray        # [N] knot samples (m) on the DS grid
    w_right: np.ndarray       # [N]
    triangle: bool = False    # no plateau: rho_min over [s_c0, s_c1] instead of [s_a0, s_a1]
    ds: float = DS

    def __post_init__(self):
        wl, wr = np.asarray(self.w_left, float), np.asarray(self.w_right, float)
        object.__setattr__(self, "_coef", (packed_coeffs(wl, False, len(wl)),
                                           packed_coeffs(wr, False, len(wr))))

    def widths(self, s) -> tuple[np.ndarray, np.ndarray]:
        """(w_left(s), w_right(s)) from the strip's spline, as ``sim.widths_at``."""
        cl, cr = self._coef
        n = len(self.w_left)
        s = np.asarray(s, float)
        return spline_np(cl, s, self.ds, n, False), spline_np(cr, s, self.ds, n, False)


def make_marks(stimulus, w_left: np.ndarray, w_right: np.ndarray, s0: float, W: int) -> Marks:
    """Marks of a ``stimuli.Stimulus`` row (its landmarks, direction and corner shape) for a
    run from s0 over W steps; ``w_left``/``w_right`` are the bank's knot samples of the row.
    Real corners count as triangles when their arc is a single knot (s_a1 <= s_a0)."""
    lm = stimulus.landmarks
    spec = getattr(stimulus, "spec", None)
    tri = bool(spec.is_triangle) if spec is not None else bool(lm.s_a1 <= lm.s_a0)
    return Marks(landmarks=lm, direction=int(stimulus.direction), s0=float(s0), W=int(W),
                 w_left=np.asarray(w_left, float), w_right=np.asarray(w_right, float),
                 triangle=tri)


@dataclass(frozen=True)
class NoReturn:
    """Point of no return of a crashed run, from the viability solves (answer_key).

    - certified with s set: no viable key from the state at s, a verified one from the tested
      state before it, so the point of no return lies in (that state, s];
    - s None and s_last_viable >= s_a0: every tested state through s_a0 had a viable key,
      so the point of no return lies after s_a0;
    - anything else (no solves, solver failures, or the first tested state already
      non-viable) is unresolved, and crash typing falls back to the ellipse rule."""

    s: float | None           # m, first tested state with no viable key; None = unresolved
    certified: bool           # the previous tested state has a verified viable key
    dv: float                 # m/s, vx - vx* at s (nan if unresolved)
    s_last_viable: float = np.nan   # m, last tested state with a verified viable key


def _get(trace, key: str, group: str):
    for k in (f"{group}_{key}", key):
        if k in trace:
            return np.asarray(trace[k])
    return None


def trajectory_from_rollout(trace: dict[str, np.ndarray], i: int, group: str = "det",
                            source: str = "agent") -> Trajectory:
    """Run i of a rollout npz group ('det', 'nat', 'mat'; see the ``rollout`` docstring),
    cut at its window W. ``trace`` is the npz (prefixed names ``<group>_<field>``) or one
    group of ``rollout.load_rollouts`` (unprefixed). Stochastic groups fill unrecorded state
    channels with NaN. The command is clipped to [-1, 1] as the env applies it; a_{-1} is the
    run's recorded prev_action at state 0 (0 when not recorded)."""
    W = int(_get(trace, "W", group)[i])
    K = W + 1
    valid = _get(trace, "valid", group)[i, :K].astype(bool)
    xs = _get(trace, "x", group)
    if xs is not None:
        x = xs[i, :K].astype(np.float64)
    else:
        x = np.full((K, 8), np.nan)
        for col, name in ((0, "s"), (1, "n"), (2, "xi"), (3, "vx"), (7, "pedal")):
            arr = _get(trace, name, group)
            if arr is not None:
                x[:, col] = arr[i, :K].astype(np.float64)

    def per_step(name):
        arr = _get(trace, name, group)
        return None if arr is None else arr[i, :W].astype(np.float64)

    def per_state(name):
        arr = _get(trace, name, group)
        return None if arr is None else arr[i, :K].astype(np.float64)

    a = per_step("a")
    term = int(_get(trace, "term_step", group)[i])
    trunc = int(_get(trace, "trunc_step", group)[i])
    pa = _get(trace, "prev_action", group)
    log_std = trace.get("log_std", None)
    meta = {"group": group, "row": int(i)}
    for name in ("cell", "stim", "pair", "phase", "sample", "steer_sign", "W", "s0"):
        arr = _get(trace, name, group)
        if arr is not None:
            meta[name] = arr[i].item()
    flags = _get(trace, "flags", group)
    return Trajectory(
        source=source, t=np.arange(K) * DT, x=x, valid=valid,
        a=None if a is None else np.clip(a, -1.0, 1.0), a_mean=per_step("a_mean"),
        reward=per_step("reward"), util_f=per_state("util_f"), util_r=per_state("util_r"),
        beta_f=per_state("beta_f"),
        flags=None if flags is None else flags[i, :W].astype(np.uint8),
        term_step=term if 0 <= term < W else -1, trunc_step=trunc if 0 <= trunc < W else -1,
        clip_frac=per_state("clip_frac"), clip_max=per_state("clip_max"),
        log_std=None if log_std is None else np.asarray(log_std, np.float64), dt=DT,
        prev_action0=np.zeros(2) if pa is None else pa[i, 0].astype(np.float64), meta=meta)


def trajectory_from_window(res: WindowResult, track: Track, p: VehicleParams = F1_2025,
                           ctx: np.ndarray | None = None, hidden_mu: float = 1.0) -> Trajectory:
    """A TO-OCP solution at 20 Hz: states res.x, commands res.a, rewards from the window
    objective (a_{-1} = 0), util and beta_f from the states; source 'to_ocp'."""
    spec = res.meta.get("spec", {}) if isinstance(res.meta, dict) else {}
    dt = float(spec.get("dt", DT))
    gamma = float(spec.get("gamma", GAMMA))
    cost = float(spec.get("action_cost", ACTION_COST))
    scale = float(spec.get("progress_scale", PROGRESS_SCALE))
    x = np.asarray(res.x, np.float64)
    a = np.asarray(res.a, np.float64)
    prev = np.vstack([np.zeros((1, 2)), a[:-1]])
    reward = (x[1:, 0] - x[:-1, 0]) / scale - cost * np.sum((a - prev) ** 2, axis=1)
    uf, ur, bf = _forces(x, car_params(p, ctx, hidden_mu))
    K = len(x)
    return Trajectory(source="to_ocp", t=np.arange(K) * dt, x=x, valid=np.ones(K, bool), a=a,
                      reward=reward, util_f=uf, util_r=ur, beta_f=bf,
                      flags=np.zeros(K - 1, np.uint8), dt=dt,
                      meta={"gamma": gamma, "action_cost": cost, "J": float(res.value),
                            "success": bool(res.success), "status": str(res.status),
                            "track": getattr(track, "name", "")})


@functools.lru_cache(maxsize=4)
def _replay_scan(substeps: int):
    """jit(fn(tr, state, acts) -> (x, reward, term, trunc, flags) per step): the env's own
    step_no_reset with every flag packed as FLAG_BITS. Cached per substep count, so keys
    replayed in one process share the compile (it is keyed on shapes and the x64 mode)."""
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing

    params = racing.EnvParams(window_steps=10**9, substeps=substeps)
    bits = {k: 1 << v for k, v in FLAG_BITS.items()}

    def run(tr, st, acts):
        def body(st, a):
            _, st2, r, term, trunc, info = racing.step_no_reset(st, a, tr, params)
            other = info["off_track"] | info["spun"] | info["stalled"]
            fl = (jnp.where(info["off_track"], bits["off_track"], 0)
                  + jnp.where(info["slipped"], bits["slipped"], 0)
                  + jnp.where(info["spun"], bits["spun"], 0)
                  + jnp.where(info["stalled"], bits["stalled"], 0)
                  + jnp.where(term & ~other, bits["nonfinite"], 0)
                  + jnp.where(trunc, bits["truncated"], 0))
            return st2, (st2.x, r, term, trunc, fl)

        return jax.lax.scan(body, st, acts)[1]

    return jax.jit(run)


def trajectory_from_replay(track: Track, x0: np.ndarray, acts: np.ndarray,
                           ctx: np.ndarray | None = None, hidden_mu: float = 1.0,
                           p: VehicleParams = F1_2025, gamma: float = GAMMA,
                           substeps: int = 10) -> Trajectory:
    """The float64 environment replay of a key's commands (``ocp.replay.replay_window``
    semantics: prev_action = accel = 0 at the start, the env's own step and reward), with
    every flag recorded; source 'replay'. States after a termination are frozen copies with
    ``valid`` False. Runs inside ``jax.enable_x64(True)``, which restores the global flag
    afterwards, but still never call it in a rollout process (it compiles the env in x64)."""
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle.sim import track_arrays

    with jax.enable_x64(True):
        f64 = jnp.float64
        tr = track_arrays(track, dtype=f64)
        c = racing.default_context() if ctx is None else jnp.asarray(ctx)
        st = racing.EnvState(x=jnp.asarray(x0, f64), prev_action=jnp.zeros(2, f64),
                             accel=jnp.zeros(2, f64), ctx=c.astype(f64),
                             hidden_mu=jnp.asarray(hidden_mu, f64), t=jnp.array(0, jnp.int32),
                             slow=jnp.asarray(0.0, f64), s0=jnp.asarray(x0[0], f64))
        out = _replay_scan(int(substeps))(tr, st, jnp.asarray(acts, f64))
        xs, rs, term, trunc, fl = (np.asarray(o) for o in jax.device_get(out))
    x = np.vstack([np.asarray(x0, np.float64)[None], xs.astype(np.float64)])
    K = len(x)
    T = K - 1
    hit = np.nonzero(term)[0]
    term_step = int(hit[0]) if len(hit) else -1
    tr_hit = np.nonzero(trunc)[0]
    trunc_step = int(tr_hit[0]) if len(tr_hit) and (term_step < 0 or tr_hit[0] < term_step) else -1
    end = term_step if term_step >= 0 else trunc_step
    valid = np.ones(K, bool)
    reward = rs.astype(np.float64)
    if end >= 0:
        valid[end + 2:] = False
        x[end + 2:] = x[end + 1]
        reward[end + 1:] = 0.0
    uf, ur, bf = _forces(x, car_params(p, ctx, hidden_mu))
    a = np.clip(np.asarray(acts, np.float64), -1.0, 1.0)
    return Trajectory(source="replay", t=np.arange(K) * DT, x=x, valid=valid, a=a,
                      reward=reward, util_f=uf, util_r=ur, beta_f=bf,
                      flags=fl.astype(np.uint8)[:T], term_step=term_step,
                      trunc_step=trunc_step, dt=DT,
                      meta={"gamma": gamma, "action_cost": ACTION_COST,
                            "track": getattr(track, "name", "")})


def trajectory_from_socp(res: OCPResult, p: VehicleParams = F1_2025) -> Trajectory:
    """An s-OCP solution (``answer_key.solve_min_time_pinned``; s in strip coordinates, t = 0
    at the pinned node): x = [s, z], t integrated (res.t), a = u at the nodes (the last node's
    control dropped, so a[k] is the control at node k); dt None. 'Steps' of the step-count
    rules (3 consecutive for the onset, 2 for the response) are nodes here."""
    s = np.asarray(res.s, np.float64)
    x = np.column_stack([s, np.asarray(res.z, np.float64)])
    uf, ur, bf = _forces(x, p)
    K = len(s)
    return Trajectory(source="socp", t=np.asarray(res.t, np.float64), x=x,
                      valid=np.ones(K, bool), a=np.asarray(res.u, np.float64)[:-1],
                      util_f=uf, util_r=ur, beta_f=bf, flags=np.zeros(K - 1, np.uint8),
                      dt=None, meta={"lap_time": float(res.lap_time), "success": bool(res.success),
                                     "status": str(res.status)})


def discounted_return(traj: Trajectory, steps: int | None = None) -> float:
    """sum_k gamma^k r_k over the steps actually taken (to the termination or truncation,
    at most ``steps``), with the step-0 action-rate term recomputed against a_{-1} = 0
    (protocol 5: regret uses a_{-1} = 0 for agent and key). NaN without rewards."""
    if traj.reward is None or traj.dt is None:
        return np.nan
    n = len(traj.reward) if steps is None else min(steps, len(traj.reward))
    if traj.term_step >= 0:
        n = min(n, traj.term_step + 1)
    elif traj.trunc_step >= 0:
        n = min(n, traj.trunc_step + 1)
    r = np.asarray(traj.reward[:n], np.float64).copy()
    if n and traj.a is not None:
        cost = float(traj.meta.get("action_cost", ACTION_COST))
        a0 = np.asarray(traj.a[0], np.float64)
        prev = np.asarray(traj.prev_action0, np.float64)
        r[0] += cost * (np.sum((a0 - prev) ** 2) - np.sum(a0**2))
    gamma = float(traj.meta.get("gamma", GAMMA))
    return float(np.sum(gamma ** np.arange(n) * r))


# ------------------------------------------------------------------------- the valid view


class _View:
    """The measured part of a trajectory: cut at W (time-stepped sources), at the first
    invalid state and at the terminal state; ``km`` is the strictly increasing prefix in s
    on which everything 'at s' or 'over [a, b]' is read (first passage)."""

    def __init__(self, traj: Trajectory, W: int | None, p: VehicleParams):
        K = len(traj.t)
        if traj.dt is not None and W is not None and W > 0:
            K = min(K, W + 1)
        valid = np.asarray(traj.valid[:K], bool)
        nv = K if valid.all() else int(np.argmin(valid))
        self.crashed = 0 <= traj.term_step < K - 1
        self.kt = traj.term_step + 1 if self.crashed else -1
        if self.crashed:
            nv = min(nv, self.kt + 1)
            if self.kt >= nv:            # terminal state flagged invalid: treat as not reached
                self.crashed, self.kt = False, -1
        self.traj, self.p, self.n = traj, p, nv
        self.K = K
        self.t = np.asarray(traj.t[:nv], np.float64)
        self.x = np.asarray(traj.x[:nv], np.float64)
        self.s = self.x[:, 0]
        self.vx = self.x[:, 3]
        d = np.diff(self.s)
        bad = np.nonzero(~(d > 0.0))[0]
        self.km = int(bad[0]) + 1 if len(bad) else nv
        if nv and not np.isfinite(self.s[0]):
            self.km = 0

    def _filled(self, name: str) -> np.ndarray:
        arr = getattr(self.traj, name)
        if arr is not None:
            return np.asarray(arr[: self.n], np.float64)
        return self._derived[{"util_f": 0, "util_r": 1, "beta_f": 2}[name]]

    @cached_property
    def _derived(self):
        if self.n and np.all(np.isfinite(self.x[:, 3:8])):
            return _forces(self.x, self.p)
        nan = np.full(self.n, np.nan)
        return nan, nan, nan

    @cached_property
    def util_f(self) -> np.ndarray:
        return self._filled("util_f")

    @cached_property
    def util_r(self) -> np.ndarray:
        return self._filled("util_r")

    @cached_property
    def beta_f(self) -> np.ndarray:
        return self._filled("beta_f")

    @cached_property
    def speed(self) -> np.ndarray:
        return np.hypot(self.x[:, 3], self.x[:, 4])

    @property
    def s_max(self) -> float:
        return float(self.s[self.km - 1]) if self.km else np.nan

    def reached(self, s_star: float) -> bool:
        return bool(self.km and self.s[self.km - 1] >= s_star)

    def at(self, s_star, y: np.ndarray):
        """y at s_star by first passage (linear in s); NaN before the start or if not reached."""
        if not self.km:
            return np.full(np.shape(s_star), np.nan) if np.ndim(s_star) else np.nan
        s = self.s[: self.km]
        out = np.interp(s_star, s, np.asarray(y[: self.km], float), left=np.nan, right=np.nan)
        return float(out) if np.ndim(out) == 0 else out

    def seg(self, a: float, b: float, y: np.ndarray):
        """(s, y) over [a, b] (states inside plus the interpolated ends), or None unless the
        run reached b; a start after a clips the window to the start."""
        if not self.km or not self.reached(b) or np.isnan(a):
            return None
        s = self.s[: self.km]
        yy = np.asarray(y[: self.km], float)
        a = max(a, float(s[0]))
        if a > b:
            return None
        inside = (s > a) & (s < b)
        ends = np.interp([a, b], s, yy)
        return np.r_[a, s[inside], b], np.r_[ends[0], yy[inside], ends[1]]


# ------------------------------------------------------------------------- metrics


@dataclass
class ResponseMetrics:
    """Every per-run measure of protocol 5. Units: m, s, m/s, m/s^2, 1/m; nan = undefined,
    censored or not recorded. Field names are the run-table column names."""

    # completion and crash
    completed: bool = False        # reached s_c1 + 250 m without terminating before it
    crashed: bool = False
    cause: str = ""                # first cause by precedence (constants.CAUSE_PRECEDENCE)
    flags_fired: int = 0           # bitmask of every flag that fired at the terminating step
    crash_side: int = 0            # +1 inside edge (sigma n > 0), -1 outside, 0 none
    crash_phase: str = ""          # 'approach' (< s_c0) | 'entry' (< s_apex) | 'exit' (<= s_c1) | 'post'
    s_crash: float = np.nan
    v_crash: float = np.nan
    t_crash: float = np.nan
    time_censored: bool = False    # alive at W without reaching s_c1 + 250 m
    s_end: float = np.nan          # last valid s
    # first sight (s_c0 - s <= d_max(vx))
    k_sight: int = -1
    s_sight: float = np.nan
    d_sight: float = np.nan        # distance travelled at first sight, s_sight - s0
    n_sight: float = np.nan
    xi_sight: float = np.nan
    pedal_sight: float = np.nan
    unsettled: bool = False        # |n| > 1 m or pedal < 0.9 at first sight
    sight_at_start: bool = False   # the corner start is in the preview at the initial state
    # braking (beta_f)
    has_brake: bool = False        # a sustained brake run starting before s_apex
    s_b: float = np.nan            # onset: start of the last run of >= 3 steps with beta_f > 0.3
    d_b: float = np.nan            # s_c0 - s_b
    v_b: float = np.nan            # vx(s_b)
    t_b: float = np.nan
    forced_onset: bool = False     # within 0.5 s of the start
    n_blips: int = 0               # earlier sustained runs
    blip_s_first: float = np.nan
    blip_s_last: float = np.nan
    blip_dv: float = np.nan        # total speed lost in blips (m/s)
    s_r: float = np.nan            # release: first step after s_b with beta_f < 0.1
    trail: float = np.nan          # s_r - s_c0
    s_dec: float = np.nan          # last pedal *command* crossing of 0 from above before s_b
    s_legacy: float = np.nan       # v1 onset: pedal state <= -0.2 for >= 3 steps
    d_legacy: float = np.nan
    s_resp: float = np.nan         # catch-trial response onset (needs the null run)
    d_resp: float = np.nan         # s_c0 - s_resp
    decel_peak: float = np.nan     # brake-only deceleration (-dvx/dt - drag/m) over [s_b, s_r]
    decel_mean: float = np.nan
    abar: float = np.nan           # (v_b^2 - v_e^2) / (2 d_b)
    # speeds
    v_e: float = np.nan            # vx(s_c0)
    v_e_abs: float = np.nan        # |v|(s_c0)
    v_arc: float = np.nan          # vx(s_a0)
    v_peak: float = np.nan         # max vx before s_b (before s_c0 without a brake)
    v_min: float = np.nan          # min vx over [s_c0 - 50, s_c1 + 50]
    v_min_abs: float = np.nan
    s_vmin_rel: float = np.nan     # s_vmin - s_apex
    v_exit100: float = np.nan      # vx(s_c1 + 100), descriptive
    v_max: float = np.nan          # max vx over the whole measured run (tier G: v_top)
    t_c1: float = np.nan           # time at s_c1
    # friction use and path
    util_f_peak: float = np.nan    # over [s_c0, s_c1]
    util_r_peak: float = np.nan
    rho_min: float = np.nan        # 1 / max kappa_path, kappa_path = (r + beta_dot) / |v|
    e_vmin_line: float = np.nan    # v_min / v_ss(rho_min) - 1
    # line eta (sigma n / (w_in - 1) inside, sigma n / (w_out - 1) outside)
    eta_pre: float = np.nan        # at s_c0 - 30
    eta_apex: float = np.nan       # at s_apex
    eta_post: float = np.nan       # at s_c1 + 30
    line_apex_rel: float = np.nan  # argmax eta - s_apex
    clearance: float = np.nan      # 1 - max eta
    # objective, clipping, oscillation
    J: float = np.nan              # discounted return over W (gamma 0.995), a_{-1} = 0
    kappa_clipped: bool = False    # any preview kappa channel at |o| >= 10 before s_c1
    clip_frac_kappa: float = np.nan
    clip_frac_edges: float = np.nan
    clip_frac_ego: float = np.nan
    clip_frac_context: float = np.nan
    clip_max_kappa: float = np.nan
    clip_max_edges: float = np.nan
    clip_max_ego: float = np.nan
    clip_max_context: float = np.nan
    osc_index: float = np.nan      # mean |Delta steer_mean| per step over the first 5 s


METRIC_COLUMNS = tuple(f.name for f in fields(ResponseMetrics))


def _phase(s: float, lm: Landmarks) -> str:
    if s < lm.s_c0:
        return "approach"
    if s < lm.s_apex:
        return "entry"
    if s <= lm.s_c1:
        return "exit"
    return "post"


def response_metrics(traj: Trajectory, marks: Marks, null: Trajectory | None = None,
                     tables: ModelTables | None = None, p: VehicleParams = F1_2025
                     ) -> ResponseMetrics:
    """Every protocol 5 measure of one trajectory. ``null`` is the same driver's tier G run
    with the same v0, width and W (for s_resp, aligned by distance travelled s - s0);
    ``tables`` gives v_ss for e_vmin_line; ``p`` (the car the run drove) gives the drag
    removed from the depth and any util / beta_f the trajectory does not carry. Measures
    after a crash are censored."""
    m = ResponseMetrics()
    v = _View(traj, marks.W, p)
    if v.n == 0:
        return m
    lm = marks.landmarks
    sig = int(marks.direction)
    s, t, x = v.s, v.t, v.x
    n_state, pedal, vx = x[:, 1], x[:, 7], x[:, 3]

    # completion and crash
    m.s_end = float(s[-1])
    straight = int(marks.direction) == 0 and lm.s_c1 <= lm.s_c0
    if straight:
        # A tier G null straight has no corner: completion is no termination within W
        # (protocol 8's gate), not reaching the virtual corner's s_c1 + 250 m.
        m.completed = not v.crashed
    else:
        m.completed = v.reached(lm.s_c1 + COMPLETE_AFTER_EXIT)
    if v.crashed:
        kt = v.kt
        fl = int(traj.flags[traj.term_step]) if traj.flags is not None else 0
        m.crashed = True
        m.flags_fired = fl & ~(1 << FLAG_BITS["truncated"])
        m.cause = crash_cause(fl)
        m.s_crash, m.v_crash, m.t_crash = float(x[kt, 0]), float(x[kt, 3]), float(t[kt])
        m.crash_side = int(np.sign(sig * x[kt, 1])) if sig else 0
        m.crash_phase = _phase(m.s_crash, lm)
    m.time_censored = (not v.crashed) and not m.completed

    # first sight
    k = first_sight(s, vx, lm.s_c0)
    if k >= 0:
        m.k_sight, m.s_sight, m.d_sight = k, float(s[k]), float(s[k] - s[0])
        m.n_sight, m.xi_sight, m.pedal_sight = float(x[k, 1]), float(x[k, 2]), float(x[k, 7])
        m.unsettled = bool(abs(m.n_sight) > UNSETTLED_N or m.pedal_sight < UNSETTLED_PEDAL)
        m.sight_at_start = k == 0

    # braking
    beta = v.beta_f
    bo = brake_onset(s, t, beta, np.ones(v.n, bool), lm.s_apex, vx)
    if np.isfinite(bo.s_b):
        m.has_brake = True
        m.s_b, m.d_b, m.v_b, m.t_b = bo.s_b, lm.s_c0 - bo.s_b, bo.v_b, bo.t_b
        m.forced_onset = bo.forced
        m.n_blips = len(bo.blips)
        if bo.blips:
            m.blip_s_first, m.blip_s_last = bo.blips[0][0], bo.blips[-1][0]
            m.blip_dv = float(np.sum([b[1] for b in bo.blips]))
        m.s_r = bo.s_r
        m.trail = bo.s_r - lm.s_c0
        cmd = traj.a if traj.a is not None else traj.a_mean
        if cmd is not None and v.n > 1:
            m.s_dec = _decision_point(s, np.asarray(cmd[: v.n - 1, 1], float),
                                      float(np.asarray(traj.prev_action0)[1]), bo.s_b)
        if np.isfinite(bo.s_r):
            m.decel_peak, m.decel_mean = _depth(t, vx, bo, p)
    lg = _sustained(s, t, -pedal, -LEGACY_PEDAL, LEGACY_STEPS, lm.s_apex, vx, inclusive=True)
    if lg is not None:
        m.s_legacy = lg[0]
        m.d_legacy = lm.s_c0 - lg[0]
    if null is not None:
        nv = _View(null, marks.W, p)
        if v.km and nv.km:
            m.s_resp = response_onset(s[: v.km], pedal[: v.km], float(s[0]), nv.s[: nv.km],
                                      nv.x[: nv.km, 7], float(nv.s[0]))
            m.d_resp = lm.s_c0 - m.s_resp

    # speeds
    m.v_e = v.at(lm.s_c0, vx)
    m.v_e_abs = v.at(lm.s_c0, v.speed)
    m.v_arc = v.at(lm.s_a0, vx)
    if m.has_brake:
        sg = v.seg(-np.inf, m.s_b, vx)
    else:
        sg = v.seg(-np.inf, lm.s_c0, vx)
    if sg is not None:
        m.v_peak = float(np.max(sg[1]))
    sg = v.seg(lm.s_c0 - V_MIN_MARGIN, lm.s_c1 + V_MIN_MARGIN, vx)
    if sg is not None:
        j = int(np.nanargmin(sg[1])) if np.isfinite(sg[1]).any() else -1
        if j >= 0:
            m.v_min, m.s_vmin_rel = float(sg[1][j]), float(sg[0][j] - lm.s_apex)
        sa = v.seg(lm.s_c0 - V_MIN_MARGIN, lm.s_c1 + V_MIN_MARGIN, v.speed)
        m.v_min_abs = float(np.min(sa[1]))
    m.v_exit100 = v.at(lm.s_c1 + 100.0, vx)
    if np.isfinite(vx).any():
        m.v_max = float(np.nanmax(vx))
    m.t_c1 = v.at(lm.s_c1, t)
    if m.has_brake and m.d_b > 0.0 and np.isfinite(m.v_e):
        m.abar = (m.v_b**2 - m.v_e**2) / (2.0 * m.d_b)

    # friction use and path
    for name in ("util_f", "util_r"):
        sg = v.seg(lm.s_c0, lm.s_c1, getattr(v, name))
        if sg is not None:
            setattr(m, f"{name}_peak", float(np.max(sg[1])))
    kap = path_curvature(x, t)
    kap = sig * kap if sig else np.abs(kap)
    a0, a1 = (lm.s_c0, lm.s_c1) if marks.triangle or lm.s_a1 <= lm.s_a0 else (lm.s_a0, lm.s_a1)
    sg = v.seg(a0, a1, kap)
    if sg is not None and np.isfinite(sg[1]).all():
        kmax = float(np.max(sg[1]))
        m.rho_min = 1.0 / kmax if kmax > 0.0 else np.inf
        if tables is not None and np.isfinite(m.v_min):
            m.e_vmin_line = m.v_min / tables.v_ss(m.rho_min) - 1.0

    # line
    for name, s_star in (("eta_pre", lm.s_c0 - ETA_OFFSET), ("eta_apex", lm.s_apex),
                         ("eta_post", lm.s_c1 + ETA_OFFSET)):
        nn = v.at(s_star, n_state)
        if np.isfinite(nn):
            setattr(m, name, float(line_eta(np.array([s_star]), np.array([nn]), marks)[0]))
    sg = v.seg(lm.s_c0, lm.s_c1, line_eta(s, n_state, marks))
    if sg is not None and np.isfinite(sg[1]).all():
        j = int(np.argmax(sg[1]))
        m.line_apex_rel = float(sg[0][j] - lm.s_apex)
        m.clearance = float(1.0 - sg[1][j])

    # objective, clipping, oscillation
    m.J = discounted_return(traj, steps=v.K - 1)
    if traj.clip_frac is not None and traj.clip_max is not None:
        cf = np.asarray(traj.clip_frac[: v.n], float)
        cm = np.asarray(traj.clip_max[: v.n], float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN groups (controls)
            for gi, g in enumerate(OBS_GROUPS):
                setattr(m, f"clip_frac_{g}", float(np.nanmean(cf[:, gi])))
                setattr(m, f"clip_max_{g}", float(np.nanmax(cm[:, gi])))
            before = s < lm.s_c1
            m.kappa_clipped = bool(np.any(((cf[:, 0] > 0.0) | (cm[:, 0] >= OBS_CLIP)) & before))
    steer = traj.a_mean if traj.a_mean is not None else traj.a
    if steer is not None and traj.dt is not None:
        m.osc_index = oscillation_index(np.asarray(steer[: max(v.n - 1, 0), 0], float),
                                        traj.dt, OSC_WINDOW)
    return m


def metrics_max_diff(a: ResponseMetrics, b: ResponseMetrics,
                     skip: tuple[str, ...] = ()) -> tuple[float, str]:
    """(largest absolute difference, its field) between two metric records: NaN == NaN,
    NaN vs a number or any mismatch of a flag or label counts as inf. Step 3 (TO-OCP
    solution vs its replay) and V5 (left vs right) use it."""
    worst, where = 0.0, ""
    for f in fields(ResponseMetrics):
        if f.name in skip:
            continue
        u, w = getattr(a, f.name), getattr(b, f.name)
        if isinstance(u, (bool, str, np.bool_)) or f.name in ("k_sight", "n_blips", "crash_side",
                                                             "flags_fired"):
            d = 0.0 if u == w else np.inf
        else:
            u, w = float(u), float(w)
            if np.isnan(u) and np.isnan(w):
                d = 0.0
            elif np.isnan(u) or np.isnan(w):
                d = np.inf
            elif np.isinf(u) or np.isinf(w):
                d = 0.0 if u == w else np.inf
            else:
                d = abs(u - w)
        if d > worst:
            worst, where = d, f.name
    return worst, where


def check_window_replay(res: WindowResult, track: Track, marks: Marks,
                        ctx: np.ndarray | None = None, hidden_mu: float = 1.0,
                        tables: ModelTables | None = None, p: VehicleParams = F1_2025) -> dict:
    """Step 3: a TO-OCP solution and its float64 environment replay must give identical
    measures to 1e-6. Returns the largest measure difference, its field, the largest state
    difference and whether the replay survived. Uses x64 inside a context (key processes)."""
    tw = trajectory_from_window(res, track, p, ctx, hidden_mu)
    tr = trajectory_from_replay(track, np.asarray(res.x[0], float), np.asarray(res.a, float),
                                ctx, hidden_mu, p,
                                gamma=float(tw.meta["gamma"]))
    mw = response_metrics(tw, marks, tables=tables, p=car_params(p, ctx, hidden_mu))
    mr = response_metrics(tr, marks, tables=tables, p=car_params(p, ctx, hidden_mu))
    d, f = metrics_max_diff(mw, mr)
    return {"max_diff": d, "field": f, "max_state_dev": float(np.max(np.abs(tw.x - tr.x))),
            "survives": tr.term_step < 0, "J_window": mw.J, "J_replay": mr.J, "pass": d <= 1e-6}


# ------------------------------------------------------------------------- errors vs keys


@dataclass
class KeyErrors:
    """Errors of one run against one key at the same phase (protocol 2.2, 5). The band
    errors against the backed-off and re-anchored keys are assembled in analysis."""

    e_ve: float = np.nan           # v_e / v_e* - 1
    dd_b: float = np.nan           # d_b - d_b* (negative: brakes later than the key)
    dd_resp: float = np.nan        # d_resp - d_resp*
    e_vmin: float = np.nan         # v_min / v_min* - 1
    e_vmin_line_excess: float = np.nan   # e_vmin_line(agent) - e_vmin_line(key)
    rms_dn: float = np.nan         # RMS(n - n*) over [s_c0 - 100, s_c1 + 100]
    regret: float = np.nan         # J* - J over W (key at this phase only)
    corner_regret: float = np.nan  # regret - regret on the null straight (null runs given)
    dt_corner: float = np.nan      # [t(s_ff) - t*(s_ff)] - same on the null straight (s)
    dt_flag: bool = False          # s_ff not reached within W: last common s used
    dt_corner_socp: float = np.nan  # the same against the pinned s-OCP and its null
    dt_flag_socp: bool = False
    dv_crash: float = np.nan       # vx - vx* at s_crash


KEY_ERROR_COLUMNS = tuple(f.name for f in fields(KeyErrors))


def key_errors(m: ResponseMetrics, mk: ResponseMetrics, traj: Trajectory, key: Trajectory,
               marks: Marks, null: Trajectory | None = None,
               null_key: Trajectory | None = None, key_at_phase: bool = True,
               socp: Trajectory | None = None, null_socp: Trajectory | None = None
               ) -> KeyErrors:
    """Errors of a run (metrics ``m``, trajectory ``traj``) against a key (``mk``, ``key``).
    Regret only when ``key_at_phase`` (protocol 5: a key solved at that phase); corner
    regret also needs the driver's ``null`` run and the ``null_key``. dt_corner subtracts the
    null straight's difference at the same distance travelled; dt_corner_socp repeats it
    against the pinned s-OCP ``socp`` and its null ``null_socp``."""
    e = KeyErrors()
    with np.errstate(divide="ignore", invalid="ignore"):
        e.e_ve = m.v_e / mk.v_e - 1.0
        e.e_vmin = m.v_min / mk.v_min - 1.0
    e.dd_b = m.d_b - mk.d_b
    e.dd_resp = m.d_resp - mk.d_resp
    e.e_vmin_line_excess = m.e_vmin_line - mk.e_vmin_line
    lm = marks.landmarks
    rv, kv = _View(traj, marks.W, F1_2025), _View(key, marks.W, F1_2025)
    grid = np.arange(lm.s_c0 - LINE_RMS_MARGIN, lm.s_c1 + LINE_RMS_MARGIN + 1e-9, 1.0)
    if rv.reached(grid[-1]) and kv.reached(grid[-1]):
        dn = rv.at(grid, rv.x[:, 1]) - kv.at(grid, kv.x[:, 1])
        if np.isfinite(dn).all():
            e.rms_dn = float(np.sqrt(np.mean(dn**2)))
    if key_at_phase:
        e.regret = mk.J - m.J
        if null is not None and null_key is not None:
            e.corner_regret = e.regret - (discounted_return(null_key, marks.W)
                                          - discounted_return(null, marks.W))
    e.dt_corner, e.dt_flag = corner_time_lost(traj, key, marks, null, null_key, m)
    if socp is not None:
        e.dt_corner_socp, e.dt_flag_socp = corner_time_lost(traj, socp, marks, null, null_socp, m)
    if m.crashed:
        e.dv_crash = m.v_crash - kv.at(m.s_crash, kv.vx)
    return e


def corner_time_lost(traj: Trajectory, key: Trajectory, marks: Marks,
                     null: Trajectory | None, null_key: Trajectory | None,
                     m: ResponseMetrics | None = None) -> tuple[float, bool]:
    """(dt_corner, flag), protocol 5: s_ff is the first s > s_c1 (1 m grid) where both the
    run and the key have vx >= 0.97 * 315 km/h; if that is not reached within W, the last
    common s is used and flagged. dt = [t(s_ff) - t*(s_ff)] - [t_null(d) - t*_null(d*)],
    with d = s_ff - s0 and d* = s_ff - s0* (each null aligned by its counterpart's distance
    travelled, so a key solved at another phase is compared fairly). NaN for crashed or
    time-censored runs, or when a null does not cover the distance."""
    if null is None or null_key is None:
        return np.nan, False
    rv, kv = _View(traj, marks.W, F1_2025), _View(key, marks.W, F1_2025)
    if m is None:
        m = response_metrics(traj, marks)
    if m.time_censored or not rv.km or not kv.km:
        return np.nan, False
    lm = marks.landmarks
    thr = FF_FRACTION * V_TOP_KPH * KPH
    s_hi = min(rv.s_max, kv.s_max)
    grid = lm.s_c1 + np.arange(1.0, max(s_hi - lm.s_c1, 0.0) + 1e-9, 1.0)
    hit = np.nonzero((rv.at(grid, rv.vx) >= thr) & (kv.at(grid, kv.vx) >= thr))[0] if len(grid) else []
    if len(hit):
        s_ff, flag = float(grid[hit[0]]), False
    elif rv.crashed or not np.isfinite(s_hi) or s_hi <= lm.s_c1:
        return np.nan, False
    else:
        s_ff, flag = float(s_hi), True
    nv, nkv = _View(null, marks.W, F1_2025), _View(null_key, marks.W, F1_2025)
    d_run = rv.at(s_ff, rv.t) - kv.at(s_ff, kv.t)
    d_null = (nv.at(float(nv.s[0]) + s_ff - float(rv.s[0]), nv.t)
              - nkv.at(float(nkv.s[0]) + s_ff - float(kv.s[0]), nkv.t)) if nv.km and nkv.km else np.nan
    return float(d_run - d_null), flag


def delta_v(traj: Trajectory, key: Trajectory, marks: Marks) -> tuple[np.ndarray, np.ndarray]:
    """(s grid [1 m over [s_c0 - 300, s_c1 + 100]], vx - vx*), NaN after a crash and
    wherever either does not cover the grid point."""
    lm = marks.landmarks
    grid = np.arange(lm.s_c0 - DV_BEFORE, lm.s_c1 + DV_AFTER + 1e-9, DV_STEP)
    rv, kv = _View(traj, marks.W, F1_2025), _View(key, marks.W, F1_2025)
    return grid, np.asarray(rv.at(grid, rv.vx) - kv.at(grid, kv.vx), float)


# ------------------------------------------------------------------------- crash typing


def crash_type(traj: Trajectory, marks: Marks, key: Trajectory | None,
               no_return: NoReturn | None, p: VehicleParams = F1_2025) -> str:
    """Protocol 9.1, for a crashed run ('' otherwise), in this order:
    'overspeed': a certified point of no return at or before s_a0 with dv > 0; or, with the
      point of no return unresolved (see ``NoReturn``), util_f or util_r >= 0.98 for >= 3
      consecutive steps with beta_f < 1 and dv > 0 at the first such step, searched from
      s_c0 - 50 m to the crash (the whole run for a crash before s_c0 - 50 m), so a
      survived earlier corner of a real strip cannot type the crash;
    'overbrake': beta_f >= 1 and util_f >= 0.98 for >= 3 consecutive steps in the last
      second, with dv <= 0 at the first of them;
    'line_steer': max(util_f, util_r) < 0.9 at every state of the last 1.0 s;
    'other': none of the above.
    dv = vx - vx*(s) from ``key`` (NaN without a key, so only line_steer or other can then be
    assigned). Slip and spin terminations are typed by the same rule."""
    v = _View(traj, marks.W, p)
    if not v.crashed:
        return ""
    lm = marks.landmarks
    kt = v.kt
    s, t = v.s, v.t
    uf, ur, bf = v.util_f, v.util_r, v.beta_f
    kv = _View(key, marks.W, p) if key is not None else None

    def dv(k: int) -> float:
        return float(v.vx[k] - kv.at(float(s[k]), kv.vx)) if kv is not None else np.nan

    resolved = False
    if no_return is not None and no_return.certified and no_return.s is not None:
        resolved = True
        if no_return.s <= lm.s_a0 and no_return.dv > 0.0:
            return "overspeed"
    elif (no_return is not None and no_return.s is None
          and np.isfinite(no_return.s_last_viable) and no_return.s_last_viable >= lm.s_a0):
        resolved = True
    if not resolved:
        k0 = 0
        if s[kt] >= lm.s_c0 - V_MIN_MARGIN:
            k0 = int(np.argmax(s >= lm.s_c0 - V_MIN_MARGIN))
        sat = ((uf >= CRASH_UTIL) | (ur >= CRASH_UTIL)) & (bf < 1.0)
        sat[:k0] = False
        runs = [r for r in _runs(sat) if r[1] - r[0] >= CRASH_UTIL_STEPS]
        if runs and dv(runs[0][0]) > 0.0:
            return "overspeed"
    last = t >= t[kt] - CRASH_LOOKBACK - 1e-9
    ob = (bf >= 1.0) & (uf >= CRASH_UTIL) & last
    runs = [r for r in _runs(ob) if r[1] - r[0] >= CRASH_UTIL_STEPS]
    if runs and dv(runs[0][0]) <= 0.0:
        return "overbrake"
    if np.all(np.maximum(uf[last], ur[last]) < LINE_STEER_UTIL):
        return "line_steer"
    return "other"


# ------------------------------------------------------------------------- pieces (tested)


@dataclass(frozen=True)
class BrakeOnset:
    s_b: float                # nan without a sustained brake before s_apex
    k_b: int                  # state index of the onset run's first step (-1)
    forced: bool
    blips: tuple[tuple[float, float], ...]   # (s_start, speed lost) of each earlier run
    s_r: float
    k_r: int
    t_b: float = np.nan
    v_b: float = np.nan       # vx(s_b), interpolated with s_b
    t_r: float = np.nan


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [start, end) of True."""
    mm = np.asarray(mask, bool)
    if not mm.any():
        return []
    d = np.diff(np.r_[0, mm.astype(np.int8), 0])
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist(),
                    strict=True))


def _frac(y: np.ndarray, k: int, thr: float) -> float:
    """Fraction w in [0, 1] from state k - 1 to state k where y crosses thr (0 for k = 0)."""
    if k == 0:
        return 0.0
    den = y[k] - y[k - 1]
    return float(np.clip((thr - y[k - 1]) / den, 0.0, 1.0)) if den != 0 else 1.0


def _lerp(y: np.ndarray, k: int, w: float) -> float:
    return float(y[k]) if k == 0 else float(y[k - 1] + w * (y[k] - y[k - 1]))


def _sustained(s, t, sig, thr, min_steps, s_apex, vx=None, inclusive=False):
    """The last run of >= min_steps states with sig > thr (>= if inclusive) whose
    interpolated start lies before s_apex: (s_start, k, w, earlier runs [(s, k, w, end)])."""
    sig = np.asarray(sig, float)
    with np.errstate(invalid="ignore"):
        on = sig >= thr if inclusive else sig > thr
    runs = []
    for a, b in _runs(on):
        if b - a < min_steps:
            continue
        w = _frac(sig, a, thr)
        sa = _lerp(s, a, w)
        if sa < s_apex:
            runs.append((sa, a, w, b))
    if not runs:
        return None
    sa, a, w, b = runs[-1]
    return sa, a, w, runs[:-1]


def brake_onset(s: np.ndarray, t: np.ndarray, beta_f: np.ndarray, valid: np.ndarray,
                s_apex: float, vx: np.ndarray | None = None) -> BrakeOnset:
    """The start of the last run of >= 3 consecutive states with beta_f > 0.3 that begins
    before s_apex, interpolated at the 0.3 crossing; earlier runs are blips (start s and the
    speed lost from their start to the first state after them; needs ``vx``); release is the
    first state after the onset with beta_f < 0.1, interpolated at the 0.1 crossing
    (protocol 5). Forced: t_b - t[0] <= 0.5 s. Same for agents and keys."""
    s, t = np.asarray(s, float), np.asarray(t, float)
    beta = np.where(np.asarray(valid, bool), np.asarray(beta_f, float), np.nan)
    vx = np.full(len(s), np.nan) if vx is None else np.asarray(vx, float)
    found = _sustained(s, t, beta, BRAKE_ON, BRAKE_MIN_STEPS, s_apex)
    if found is None:
        return BrakeOnset(np.nan, -1, False, (), np.nan, -1)
    s_b, k_b, w, earlier = found
    blips = []
    for sa, a, wa, b in earlier:
        v_end = vx[min(b, len(vx) - 1)]
        blips.append((float(sa), float(_lerp(vx, a, wa) - v_end)))
    t_b = _lerp(t, k_b, w)
    v_b = _lerp(vx, k_b, w)
    with np.errstate(invalid="ignore"):
        rel = np.nonzero(beta[k_b + 1:] < BRAKE_OFF)[0]
    s_r = t_r = np.nan
    k_r = -1
    if len(rel):
        k_r = int(k_b + 1 + rel[0])
        wr = _frac(beta, k_r, BRAKE_OFF)
        s_r, t_r = _lerp(s, k_r, wr), _lerp(t, k_r, wr)
    return BrakeOnset(float(s_b), int(k_b), bool(t_b - t[0] <= FORCED_ONSET + 1e-12),
                      tuple(blips), float(s_r), k_r, float(t_b), float(v_b), float(t_r))


def _decision_point(s: np.ndarray, cmd: np.ndarray, prev0: float, s_b: float) -> float:
    """Last crossing of the pedal command from > 0 to <= 0 at or before s_b, interpolated
    between the two states whose commands bracket it (a_{-1} = prev0 at the start)."""
    c_prev = np.r_[prev0, cmd[:-1]]
    with np.errstate(invalid="ignore"):
        ks = np.nonzero((c_prev > 0.0) & (cmd <= 0.0))[0]
    best = np.nan
    for k in ks:
        if k == 0:
            pos = float(s[0])
        else:
            w = c_prev[k] / (c_prev[k] - cmd[k])
            pos = float(s[k - 1] + w * (s[k] - s[k - 1]))
        if pos <= s_b:
            best = pos
    return best


def _depth(t: np.ndarray, vx: np.ndarray, bo: BrakeOnset, p: VehicleParams) -> tuple[float, float]:
    """(peak, mean) brake-only deceleration -dvx/dt - drag / m over [s_b, s_r]: the peak of
    the central-difference series over the states k_b..k_r, the mean as the exact speed
    drop over the interval minus the time-averaged drag."""
    if len(t) < 2:
        return np.nan, np.nan
    drag = 0.5 * p.rho * p.cda * vx**2 / p.m
    series = -np.gradient(vx, t) - drag
    k_b, k_r = bo.k_b, bo.k_r
    peak = float(np.max(series[k_b: k_r + 1]))
    dt = bo.t_r - bo.t_b
    if not dt > 0.0:
        return peak, float(series[k_b])
    v_r = float(np.interp(bo.t_r, t, vx))
    tt = np.r_[bo.t_b, t[k_b:k_r], bo.t_r]
    vv = np.r_[bo.v_b, vx[k_b:k_r], v_r]
    keep = (tt >= bo.t_b) & (tt <= bo.t_r)
    tt, vv = tt[keep], vv[keep]
    dd = 0.5 * p.rho * p.cda * vv**2 / p.m
    mean_drag = float(np.sum(0.5 * (dd[1:] + dd[:-1]) * np.diff(tt)) / dt)
    return peak, float((bo.v_b - v_r) / dt - mean_drag)


def response_onset(s: np.ndarray, pedal: np.ndarray, s0: float, s_null: np.ndarray,
                   pedal_null: np.ndarray, s0_null: float) -> float:
    """First s with |pedal(s) - pedal_null(s)| > 0.1 for >= 2 consecutive steps, the null
    aligned by distance travelled s - s0 (protocol 5), interpolated at the 0.1 crossing.
    ``s_null`` must be increasing (the null's valid, monotone part); beyond it NaN."""
    s, pedal = np.asarray(s, float), np.asarray(pedal, float)
    d_null = np.asarray(s_null, float) - s0_null
    pn = np.interp(s - s0, d_null, np.asarray(pedal_null, float), left=np.nan, right=np.nan)
    diff = np.abs(pedal - pn)
    with np.errstate(invalid="ignore"):
        on = diff > RESP_DELTA
    runs = [r for r in _runs(on) if r[1] - r[0] >= RESP_STEPS]
    if not runs:
        return np.nan
    k = runs[0][0]
    if k == 0 or not np.isfinite(diff[k - 1]):
        return float(s[k])
    return _lerp(s, k, _frac(diff, k, RESP_DELTA))


def line_eta(s: np.ndarray, n: np.ndarray, marks: Marks) -> np.ndarray:
    """eta = sigma n / (w_in - 1) where sigma n >= 0, else sigma n / (w_out - 1): +1 and -1
    are the inside and outside termination boundaries. A straight (sigma 0) uses sigma = +1
    (left as 'inside')."""
    wl, wr = marks.widths(s)
    sig = 1 if marks.direction >= 0 else -1
    sn = sig * np.asarray(n, float)
    w_in, w_out = (wl, wr) if sig > 0 else (wr, wl)
    with np.errstate(invalid="ignore"):
        return np.where(sn >= 0.0, sn / (w_in - EDGE_MARGIN), sn / (w_out - EDGE_MARGIN))


def path_curvature(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """kappa_path = (r + beta_dot) / |v| with beta = atan2(vy, vx), beta_dot by central
    differences in t (one-sided at the ends; irregular t allowed)."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    if len(t) < 2:
        return np.full(len(t), np.nan)
    vx, vy, r = x[:, 3], x[:, 4], x[:, 5]
    beta = np.unwrap(np.arctan2(vy, vx)) if np.isfinite(vy).all() else np.arctan2(vy, vx)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r + np.gradient(beta, t)) / np.hypot(vx, vy)


def first_sight(s: np.ndarray, vx: np.ndarray, s_c0: float) -> int:
    """First state index with s_c0 - s <= d_max(vx) = clip(6 vx, 150, 500) (the far
    preview point, ``racing.preview_offsets``); -1 if never."""
    d_max = np.clip(6.0 * np.maximum(np.asarray(vx, float), 0.0), 150.0, 500.0)
    with np.errstate(invalid="ignore"):
        hit = np.nonzero(s_c0 - np.asarray(s, float) <= d_max)[0]
    return int(hit[0]) if len(hit) else -1


def oscillation_index(steer_mean: np.ndarray, dt: float = DT, window: float = OSC_WINDOW) -> float:
    """Mean |Delta steer_mean| per step over the first ``window`` seconds (the first
    window / dt commands)."""
    n = round(window / dt)
    sm = np.asarray(steer_mean[:n], float)
    if len(sm) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(sm))))


_CAUSES = (("off_track", "corridor"), ("slipped", "slip"), ("spun", "spin"),
           ("stalled", "stall"), ("nonfinite", "nonfinite"))


def crash_cause(flags: int) -> str:
    """First cause by precedence corridor > slip > spin > stall > non-finite, from one step's
    FLAG_BITS (spun includes slipped in the env, so slip is tested first); '' if none."""
    flags = int(flags)
    for bit, name in _CAUSES:
        if flags >> FLAG_BITS[bit] & 1:
            return name
    return ""
