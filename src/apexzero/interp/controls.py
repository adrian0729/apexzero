"""Scripted positive controls run through the identical harness (protocol 7).

A classifier is only trusted if it returns the planted verdict on a driver whose law is
known. These controls are those drivers. Each is a vmappable JAX function with the
harness's Policy signature ``policy(params, o, state, tr, aux) -> (mean [2], log_std [2])``
(log_std = -inf: no native noise), so it runs through stimulus bank -> rollout -> measures
-> analysis exactly as an agent does. Controls get the raw observation (the rollout is
called with obs_rms = None), except K, which normalises with S1's obs_rms itself.

    T   tape: the nominal key's actions at u = 0, replayed open-loop (aux = tape row)
    Q   QSS on the full road: pure pursuit on the centreline, lookahead max(8 m, 0.6 s v);
        speed target c_v v_QSS(s) from a forward-backward pass with a_y,max(v) and a_dec(v)
        (measures.ModelTables) on the full-resolution kappa; pedal = 0.5 per m/s error plus
        feed-forward, clipped (aux = target-profile row)
    Qp  Q, but the kappa ahead is rebuilt each step from the 40 preview kappa samples only
        (linear in d, last sample held beyond d_max); the target is recomputed every step
    c1  Q with the braking branch of its target moved 15 m towards the corner (late onset)
    c2  Q reading kappa capped at 1/12 1/m (saturating target)
    c3  Q that brakes when the time to the corner <= tau_c, then at a constant fraction of
        the car's braking capacity that reaches Q's own (lawful) entry speed at the corner;
        tau_c fitted so d_b matches the key at R 22.9, 90 deg, A1
    K   1-nearest-neighbour on S1's Hockenheim buffer (normalised obs -> the neighbour's
        action); k = 5 with distance weights if C(a) < 0.9

Implementation choices (IMPLEMENTATION.md, decisions C1-C12; each was forced by a
measured failure of the literal reading, noted there):

- QSS pass: the classic one of ``oracle/qss.py`` (friction ellipse in both passes) with the
  model's tables in place of the point-mass envelope: cornering limit v_ss(1/|kappa|) from
  a_y,max(v), braking a_dec(v) sqrt(1 - lat^2). a_y,max is used as a cornering limit
  (``lateral_limit``): its power-limited entries above 265 km/h (no steady state for lack of
  power, 0 at 315 km/h) are replaced by the tyres' grip extrapolated with the normal load,
  because the forward pass carries the power limit.
- c_v scales the grip-limited part: target = min(v_top, c_v V_grip(s)), V_grip the backward
  pass (unbounded on straights). Scaling the power-limited straight-line speed too would
  make Q brake on every straight above c_v v_top. The braking branches then decelerate at
  about c_v^2 a_dec, the margin a proportional controller needs to track them.
- The forward pass is not part of the policy's target (the pedal saturates at the traction
  limit when the target is out of reach, which is what the forward pass describes); it
  gives Q's offline prediction (``forward_profile``).
- Pedal: 0.5 per m/s on the error between the target read 0.075 s ahead (half the hold plus
  the pedal lag) and the car's speed predicted to that point with its measured
  acceleration, plus the feed-forward pedal for the target's acceleration there (steady
  straight-line pedal map of the model). "Clipped" is to the friction ellipse: between the
  pedal that saturates the brakes and the full-throttle pedal, each scaled by the axle's
  room sqrt(1 - (fy/fy_max)^2) at its current slip angle (``axle_room``).
- Steering: pure pursuit from the car's reference point to the centreline point at arc
  length s + max(8 m, 0.6 s v); the pursuit curvature is turned into the model's
  steady-state steering angle (kinematic angle plus the understeer map
  ``steer_correction``), clipped to the front tyre's peak slip and the lock.
- c3 brakes, once triggered, at the constant fraction of the car's braking capacity that
  reaches Q's lawful entry speed exactly at the corner start, and hands over to Q once Q's
  read-ahead point is in the corner.
- Full-road target profiles are precomputed per stimulus and stacked into a [n_rows, N]
  table indexed by ``aux``; Qp runs its 251-point pass (2 m grid, 500 m) every step.
- K returns the buffer's *mean* action (the specialist's deterministic command at that
  state); the sampled action is kept in the buffer too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.interp.agents import LoadedAgent
from apexzero.interp.constants import (
    COMPLETE_AFTER_EXIT,
    DS,
    DT,
    OBS_CLIP,
    OBS_PREVIEW,
)
from apexzero.interp.measures import ModelTables, cornering_limit
from apexzero.interp.rollout import Policy
from apexzero.interp.stimuli import Cell, Landmarks, StimulusBank
from apexzero.tracks.track import Track
from apexzero.vehicle.dynamics import body_derivatives
from apexzero.vehicle.ops import Ops
from apexzero.vehicle.params import F1_2025, VehicleParams
from apexzero.vehicle.sim import TrackArrays, global_pose, lookup

CONTROL_NAMES = ("T", "Q", "Qp", "c1", "c2", "c3", "K")
Q_VARIANTS = ("Q", "Qp", "c1", "c2", "c3")
CV_CHOICES = (0.95, 0.90, 0.85, 0.80)
LOOKAHEAD_MIN = 8.0          # m
LOOKAHEAD_TIME = 0.6         # s
K_V = 0.5                    # pedal per m/s of speed error
C1_SHIFT = 15.0              # m, late onset
C2_KAPPA_CAP = 1.0 / 12.0    # 1/m
LEAD_TIME = 0.5 * DT + F1_2025.tau_pedal   # s: zero-order hold + pedal lag
V_FREE = 150.0               # m/s: "no cornering limit" (above the car's top speed)
QP_DS = 2.0                  # m, Qp's grid spacing
QP_N = 251                   # Qp grid points: d in [0, 500] m (d_max <= 500)
TAU_C_BRACKET = (0.25, 4.0)  # s, scanned range of c3's tau_c
KNN_K_FALLBACK = 5

# Pedal map grids (uniform): 0 to 120 m/s (so c_v v_ss >= v_top beyond the grid for every
# c_v choice), accelerations -90..+40 m/s^2.
_PM_V = np.arange(0.0, 120.0 + 1e-9, 0.5)
_PM_A = np.arange(-90.0, 40.0 + 1e-9, 0.25)
_PM_P = np.linspace(-1.0, 1.0, 801)
_SAT_TOL = 1e-3              # saturation: within 0.1% of the extreme acceleration
_US_U = np.linspace(0.0, 0.95, 39)   # steady-state steering map: u = a_y / a_y,max(v)
_AY_FLOOR = 1e-3             # m/s^2: a_y,max is 0 at top speed (power-limited); no 0/0

_NP = Ops(name="numpy", sin=np.sin, cos=np.cos, tan=np.tan, atan=np.arctan, tanh=np.tanh,
          sqrt=np.sqrt, stack=lambda xs: np.stack(np.broadcast_arrays(*xs), axis=-1),
          index=lambda x, i: x[..., i], where=np.where)


# ------------------------------------------------------------------------- car tables


def straight_accel(v, pedal, p: VehicleParams = F1_2025) -> np.ndarray:
    """dvx/dt (m/s^2) of the full model driving straight at speed v with a steady pedal
    (state [0, 0, 0, v, 0, 0, 0, pedal], command [0, pedal]); float64, broadcasts."""
    v, pedal = np.broadcast_arrays(np.asarray(v, float), np.asarray(pedal, float))
    x = np.zeros(v.shape + (8,))
    x[..., 3], x[..., 7] = v, pedal
    u = np.stack([np.zeros_like(pedal), pedal], axis=-1)
    return np.asarray(body_derivatives(x, u, p, _NP)[0])


class CarTables(NamedTuple):
    """Speed-dependent limits on one uniform speed grid, as the Q family uses them. NumPy
    float64 when built; the policies get them as float32 JAX arrays (``QParams.car``)."""

    v: Any            # [m] m/s, uniform grid from 0 (_PM_V)
    a_dec: Any        # [m] largest straight-line deceleration (ModelTables.a_dec)
    a_y: Any          # [m] largest steady lateral acceleration (ModelTables.a_y_max)
    a_acc: Any        # [m] full-throttle straight-line acceleration (pedal map)
    kap_ss: Any       # [m'] ascending: the curvature a_y(v) / v^2 whose v_ss is v_ss[i]
    v_ss: Any         # [m'] matching speeds (descending)
    pm_a: Any         # [na] uniform acceleration grid of the pedal map
    pm_pedal: Any     # [m, na] steady straight-line pedal giving acceleration a at v
    us_u: Any         # [nu] uniform grid of u = a_y / a_y,max(v)
    us_dd: Any        # [m, nu] steady-state steering minus the kinematic angle (rad)
    d_brake: Any      # [m] m, straight-line braking distance from v to 0 at a_dec
    v_top: Any        # scalar m/s, a_acc(v_top) = 0
    alpha_peak: Any   # scalar rad, the tyre's peak slip angle (steering limit)


def pedal_map(p: VehicleParams = F1_2025, v_grid: np.ndarray = _PM_V,
              a_grid: np.ndarray = _PM_A) -> tuple[np.ndarray, np.ndarray]:
    """(pedal [nv, na], a_acc [nv]): the steady straight-line pedal giving acceleration a at
    speed v, restricted to the unsaturated range (accelerations beyond what the tyres give
    map to the pedal where the axle saturates, never further); and the full-throttle
    acceleration."""
    acc = straight_accel(v_grid[:, None], _PM_P[None, :], p)        # [nv, np]
    out = np.empty((len(v_grid), len(a_grid)))
    a_acc = acc.max(axis=1)
    i0 = int(np.argmin(np.abs(_PM_P)))
    for j in range(len(v_grid)):
        a = acc[j]
        a_min, a_max, a0 = a.min(), a.max(), a[i0]
        lo = np.nonzero(a <= a_min + _SAT_TOL * max(a0 - a_min, 1e-12))[0].max()
        hi = np.nonzero(a >= a_max - _SAT_TOL * max(a_max - a0, 1e-12))[0].min()
        hi = max(hi, lo + 1)
        sub_a = np.maximum.accumulate(a[lo:hi + 1])
        sub_a = sub_a + 1e-9 * np.arange(len(sub_a))                # strictly increasing
        out[j] = np.interp(a_grid, sub_a, _PM_P[lo:hi + 1])
    return out, a_acc


def steer_correction(a_y: np.ndarray, cruise_pedal: np.ndarray, p: VehicleParams = F1_2025,
                     v_grid: np.ndarray = _PM_V, u_grid: np.ndarray = _US_U) -> np.ndarray:
    """dd(v, u) [nv, nu] (rad): steering of the model's steady turn at lateral acceleration
    u a_y,max(v), minus the kinematic angle atan(L a_y / v^2), i.e. the understeer the pursuit
    law must add. Newton on dvy = dr = 0 (cruise pedal, r = a_y / |v|), continued in u from
    the straight-line solution; rows below 6 m/s fade to 0 (the model turns kinematic below
    4 m/s); a cell that does not converge holds the last converged value along u."""
    L = p.lf + p.lr
    rows = v_grid >= 6.0
    V = v_grid[rows]
    ped = cruise_pedal[rows]
    ay_max = a_y[rows]
    out = np.zeros((len(v_grid), len(u_grid)))
    dd = np.zeros((len(V), len(u_grid)))
    z = np.zeros((len(V), 2))
    ok_prev = np.ones(len(V), bool)

    def resid(z, ay):
        vy = z[:, 1]
        r = ay / np.sqrt(V**2 + vy**2)
        x = np.zeros((len(V), 8))
        x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7] = V, vy, r, z[:, 0], ped
        u = np.stack([z[:, 0] / p.delta_max, ped], axis=-1)
        _, dvy, dr, _, _ = body_derivatives(x, u, p, _NP)
        return np.stack([dvy, dr], axis=-1)

    for j, uj in enumerate(u_grid):
        ay = uj * ay_max
        for _ in range(12):
            f = resid(z, ay)
            J = np.empty((len(V), 2, 2))
            for k in range(2):
                h = np.zeros(2)
                h[k] = 1e-6
                J[:, :, k] = (resid(z + h, ay) - resid(z - h, ay)) / 2e-6
            z = z + np.linalg.solve(J, -f[..., None])[..., 0]
        ok = ok_prev & (np.abs(resid(z, ay)).max(axis=1) < 1e-6)
        kin = np.arctan(L * ay / V**2)
        dd[:, j] = np.where(ok, z[:, 0] - kin, dd[:, j - 1] if j else 0.0)
        ok_prev = ok
    fade = np.clip((V - 4.0) / 2.0, 0.0, 1.0)[:, None]
    out[rows] = dd * np.minimum(fade, 1.0)
    return out


def lateral_limit(tables: ModelTables, p: VehicleParams = F1_2025,
                  v: np.ndarray = _PM_V) -> np.ndarray:
    """a_y,max(v) on ``v`` as a cornering limit (``measures.cornering_limit``): the protocol
    table where the tyres or the lock limit it, the tyres' grip extrapolated with the normal
    load beyond the last tyre-limited speed. The forward pass carries the power limit."""
    return cornering_limit(tables, p, np.asarray(v, float))


def car_tables(tables: ModelTables, p: VehicleParams = F1_2025) -> CarTables:
    """The Q family's tables from the protocol-5 model tables (a_dec; a_y,max as a
    cornering limit, ``lateral_limit``) and the model's pedal map (feed-forward, traction
    limit) and steady-state steering map. Reads the table arrays directly."""
    v = _PM_V
    a_dec = np.interp(v, np.asarray(tables.v_dec, float), np.asarray(tables.a_dec, float))
    a_y = lateral_limit(tables, p, v)
    pm, a_acc = pedal_map(p, v, _PM_A)
    cruise = np.array([np.interp(0.0, _PM_A, row) for row in pm])
    us_dd = steer_correction(a_y, cruise, p, v, _US_U)
    alpha_peak = float(np.tan(np.pi / (2.0 * p.pac_c)) / p.pac_b)
    # v_top: first sign change of the full-throttle acceleration.
    k = int(np.nonzero(a_acc <= 0.0)[0][0]) if np.any(a_acc <= 0.0) else len(v) - 1
    v_top = float(np.interp(0.0, [a_acc[k], a_acc[k - 1]], [v[k], v[k - 1]])) if k > 0 else v[-1]
    # v_ss(kappa) inverts kappa_ss(v) = a_y(v) / v^2 on the lateral table's own range.
    use = v >= float(np.asarray(tables.v_lat)[0]) - 1e-9
    vs = v[use]
    kap = a_y[use] / vs**2
    kap = np.minimum.accumulate(kap) - 1e-12 * np.arange(len(kap))   # strictly decreasing
    integrand = v / np.maximum(a_dec, 1.0)
    d_brake = np.r_[0.0, np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(v))]
    return CarTables(v=v, a_dec=a_dec, a_y=a_y, a_acc=a_acc, kap_ss=kap[::-1].copy(),
                     v_ss=vs[::-1].copy(), pm_a=_PM_A.copy(), pm_pedal=pm,
                     us_u=_US_U.copy(), us_dd=us_dd, d_brake=d_brake, v_top=v_top,
                     alpha_peak=alpha_peak)


def _as_jax(car: CarTables) -> CarTables:
    return CarTables(*(jnp.asarray(a, jnp.float32) for a in car))


# ------------------------------------------------------------------------- JAX pieces


def _uinterp(x, grid, table):
    """Linear interpolation on a uniform grid (clamped at the ends)."""
    x0, dx = grid[0], grid[1] - grid[0]
    n = grid.shape[0]
    f = jnp.clip((x - x0) / dx, 0.0, n - 1.0)
    i = jnp.minimum(jnp.floor(f).astype(jnp.int32), n - 2)
    t = f - i
    return table[i] * (1.0 - t) + table[i + 1] * t


def _bilinear(x, y, xg, yg, table):
    """table[i, j] at (xg[i], yg[j]) on uniform grids, bilinear, clamped at the edges."""
    nx, ny = table.shape
    fx = jnp.clip((x - xg[0]) / (xg[1] - xg[0]), 0.0, nx - 1.0)
    ix = jnp.minimum(jnp.floor(fx).astype(jnp.int32), nx - 2)
    tx = fx - ix
    fy = jnp.clip((y - yg[0]) / (yg[1] - yg[0]), 0.0, ny - 1.0)
    iy = jnp.minimum(jnp.floor(fy).astype(jnp.int32), ny - 2)
    ty = fy - iy
    return ((1 - tx) * ((1 - ty) * table[ix, iy] + ty * table[ix, iy + 1])
            + tx * ((1 - ty) * table[ix + 1, iy] + ty * table[ix + 1, iy + 1]))


def _pedal_for(v, a, car: CarTables):
    """Steady straight-line pedal giving acceleration a at speed v (bilinear)."""
    return _bilinear(v, a, car.v, car.pm_a, car.pm_pedal)


def _v_ss(kap_abs, car: CarTables):
    """Cornering limit v_ss(1 / |kappa|) (m/s); V_FREE where the car is flat out."""
    return jnp.interp(kap_abs, car.kap_ss, car.v_ss, left=V_FREE)


def grip_profile(kap_abs, ds, car: CarTables):
    """Backward QSS pass [n] (m/s): cornering limit v_ss(1/|kappa|), braking with
    a_dec(v) sqrt(1 - lat^2), lat = v^2 |kappa| / a_y,max(v), from the last point's limit.
    The same function serves Q (full road, ds = 2 m knots) and Qp (preview, 2 m grid)."""
    vlim = _v_ss(kap_abs, car)

    def body(v_next, inp):
        k_next, vl = inp
        ay = jnp.maximum(_uinterp(v_next, car.v, car.a_y), _AY_FLOOR)
        lat = jnp.clip(v_next * v_next * k_next / ay, 0.0, 1.0)
        a = _uinterp(v_next, car.v, car.a_dec) * jnp.sqrt(1.0 - lat * lat)
        v = jnp.minimum(vl, jnp.sqrt(v_next * v_next + 2.0 * a * ds))
        return v, v

    _, vb = jax.lax.scan(body, vlim[-1], (kap_abs[1:], vlim[:-1]), reverse=True)
    return jnp.concatenate([vb, vlim[-1:]])


def _target_row(kappa, car: CarTables, c_v, kappa_cap, ds):
    """min(v_top, c_v V_grip) on one strip, and the cornering limit c_v v_ss."""
    kap = jnp.minimum(jnp.abs(kappa), kappa_cap)
    vg = grip_profile(kap, ds, car)
    return jnp.minimum(car.v_top, c_v * vg), jnp.minimum(car.v_top, c_v * _v_ss(kap, car))


_target_rows = jax.jit(jax.vmap(_target_row, in_axes=(0, None, None, None, None)))


def _at(T, x, h):
    """(T(x), T dT/dx) on a uniform grid T[i] at i h: linear value, slope of the interval."""
    n = T.shape[0]
    f = jnp.clip(x / h, 0.0, n - 1.0)
    i = jnp.minimum(jnp.floor(f).astype(jnp.int32), n - 2)
    t = f - i
    return T[i] * (1.0 - t) + T[i + 1] * t, (T[i + 1] ** 2 - T[i] ** 2) / (2.0 * h)


# ------------------------------------------------------------------------- the Q family


class QParams(NamedTuple):
    """Jit parameters of the Q family (Q, Qp, c1, c2, c3)."""

    v_target: jax.Array      # [n_rows, N] m/s, target min(v_top, c_v V_grip) per row (aux)
    s_corner: jax.Array      # [n_rows] m, corner start per row (c3's time-to-corner)
    car: CarTables           # float32 tables (grip profile, feed-forward, clip, v_top)
    c_v: jax.Array           # scalar speed-target factor (Qp applies it per step)
    k_v: jax.Array           # scalar, pedal per m/s
    lookahead_min: jax.Array
    lookahead_time: jax.Array
    lead_time: jax.Array     # s, read-ahead of the target and of the car's speed
    kappa_cap: jax.Array     # 1/m (inf: no cap; c2: 1/12)
    tau_c: jax.Array         # s, c3 brake trigger (0: off)


def pursuit_steer(state: racing.EnvState, tr: TrackArrays, lookahead_min, lookahead_time,
                  car: CarTables, p: VehicleParams = F1_2025):
    """Pure pursuit on the centreline: the arc through the car's reference point, tangent
    to its heading, that reaches the centreline point at arc length s + max(L_min, t_L v)
    (curvature kappa = 2 sin(alpha) / l_d). The steering is the model's steady-state angle
    for that curvature (kinematic atan(L kappa) plus the understeer map), clipped to the
    tyre's peak slip on the front axle and to the lock."""
    x = state.x
    vx, vy, r = x[3], x[4], x[5]
    v = jnp.maximum(vx, 0.0)
    X, Y, psi = global_pose(x, tr)
    c, sn = jnp.cos(psi), jnp.sin(psi)
    st = x[0] + jnp.maximum(lookahead_min, lookahead_time * v)
    dx, dy = lookup(tr.x, st, tr) - X, lookup(tr.y, st, tr) - Y
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    kappa = 2.0 * by / jnp.maximum(bx * bx + by * by, 1e-6)
    u = v * v * jnp.abs(kappa) / jnp.maximum(_uinterp(v, car.v, car.a_y), _AY_FLOOR)
    delta = (jnp.arctan((p.lf + p.lr) * kappa)
             + jnp.sign(kappa) * _bilinear(v, u, car.v, car.us_u, car.us_dd))
    beta_f = jnp.arctan2(vy + p.lf * r, jnp.maximum(vx, 1.0))
    limited = jnp.clip(delta, beta_f - car.alpha_peak, beta_f + car.alpha_peak)
    delta = jnp.where(vx > 4.0, limited, delta)
    return jnp.clip(delta / p.delta_max, -1.0, 1.0)


def axle_room(x, car: CarTables, p: VehicleParams = F1_2025):
    """(front, rear) longitudinal room left by each axle's lateral use: sqrt(1 - u^2) with
    u = fy_pure / fy_max = sin(C atan(B |alpha|)) (1 beyond the peak slip), the per-axle
    friction ellipse of ``dynamics.tyre_forces``."""
    vx, vy, r, delta = x[3], x[4], x[5], x[6]
    v_safe = jnp.maximum(vx, 1.0)
    a_f = jnp.abs(delta - jnp.arctan((vy + p.lf * r) / v_safe))
    a_r = jnp.abs(jnp.arctan((vy - p.lr * r) / v_safe))

    def room(a):
        use = jnp.where(a >= car.alpha_peak, 1.0, jnp.sin(p.pac_c * jnp.arctan(p.pac_b * a)))
        return jnp.sqrt(jnp.clip(1.0 - use * use, 0.0, 1.0))

    return room(a_f), room(a_r)


def pedal_command(state: racing.EnvState, v_t, a_t, params: QParams):
    """Proportional speed control plus feed-forward, clipped to the friction ellipse.

    ``v_t`` and ``a_t`` are the target's speed and acceleration at the read-ahead point
    (``lead_time`` ahead); the proportional term (k_v per m/s) acts on the error against
    the car's speed predicted to that point with its measured acceleration, v + lead ax,
    which absorbs the zero-order hold and the pedal lag. The feed-forward is the steady
    straight-line pedal for a_t. The clip keeps the pedal between the pedal that saturates
    the brakes at a_dec(v) times the front axle's room, and the pedal that gives the
    full-throttle acceleration a_acc(v) times the rear axle's room (``axle_room``)."""
    car = params.car
    x = state.x
    v = jnp.maximum(x[3], 0.0)
    room_f, room_r = axle_room(x, car)
    p_hi = _pedal_for(v, _uinterp(v, car.v, car.a_acc) * room_r, car)
    p_lo = _pedal_for(v, -_uinterp(v, car.v, car.a_dec) * room_f, car)
    v_pred = v + params.lead_time * state.accel[0]
    ff = _pedal_for(v, a_t, car)
    return jnp.clip(params.k_v * (v_t - v_pred) + ff, p_lo, p_hi)


def preview_profile(o, v, params: QParams):
    """Qp's target T [QP_N] on d_j = 2 j (m ahead of the car): kappa rebuilt from the 40 raw
    preview samples (o[5k+4] = 50 kappa_k at d_k), linear in d, held before d_1 and beyond
    d_max; then the same grip pass and scaling as Q."""
    d_k = racing.preview_offsets(v)
    kap_k = o[4:OBS_PREVIEW:5] / 50.0
    d_j = jnp.arange(QP_N, dtype=v.dtype) * QP_DS
    kap = jnp.minimum(jnp.abs(jnp.interp(d_j, d_k, kap_k)), params.kappa_cap)
    return jnp.minimum(params.car.v_top, params.c_v * grip_profile(kap, QP_DS, params.car))


def make_q_policy(variant: str = "Q") -> Policy:
    """The Q family as a Policy ('Q', 'Qp', 'c1', 'c2', 'c3'); params QParams, aux the
    stimulus's target row. Q, c1 and c2 differ only in their precomputed rows; Qp builds its
    target from the observation every step; c3 adds the time-to-corner trigger."""
    if variant not in Q_VARIANTS:
        raise ValueError(f"unknown Q variant {variant!r}")

    def policy(params: QParams, o, state: racing.EnvState, tr: TrackArrays, aux):
        x = state.x
        v = jnp.maximum(x[3], 0.0)
        steer = pursuit_steer(state, tr, params.lookahead_min, params.lookahead_time,
                              params.car)
        ahead = params.lead_time * v
        if variant == "Qp":
            v_t, a_t = _at(preview_profile(o, v, params), ahead, QP_DS)
        else:
            row = params.v_target[aux]
            v_t, a_t = _at(row, x[0] + ahead, tr.ds)
            if variant == "c3":
                v_pred = v + params.lead_time * state.accel[0]
                v_t, a_t = _c3_target(x[0], v, v_pred, ahead, row, params.s_corner[aux],
                                      params.tau_c, v_t, a_t, tr.ds, params.car)
        pedal = pedal_command(state, v_t, a_t, params)
        mean = jnp.stack([steer, pedal]).astype(o.dtype)
        return mean, jnp.full((2,), -jnp.inf, o.dtype)

    return policy


def _c3_target(s, v, v_pred, ahead, row, s_c, tau_c, v_t, a_t, ds, car: CarTables):
    """c3: Q, except on the approach to the corner start s_c. There it is flat out until
    d_c <= tau_c v, then brakes at the constant fraction k of the car's capacity that
    reaches Q's own (lawful) entry speed v_e = row(s_c) exactly at s_c, k = (D(v) - D(v_e))
    / d_c with D the straight-line braking distance at a_dec, as pure feed-forward (the
    target is the car's own predicted speed, so the proportional term is 0). Once Q's
    read-ahead point is past s_c, Q's target applies wherever it is the lower (corners whose
    lawful braking lies inside the ramp, where T(s_c) = v_top). k is recomputed every step,
    so the law is memoryless and corrects its own tracking error."""
    d_c = s_c - s
    v_e, _ = _at(row, s_c, ds)
    before = d_c > 0.0
    trig = before & (d_c <= tau_c * v) & (v > v_e)
    k = ((_uinterp(v, car.v, car.d_brake) - _uinterp(v_e, car.v, car.d_brake))
         / jnp.maximum(d_c, 1e-3))
    vt_c = jnp.where(trig, v_pred, car.v_top)
    at_c = jnp.where(trig, -k * _uinterp(v, car.v, car.a_dec), 0.0)
    # (v_t, a_t) are Q's at the read-ahead point s + ahead; they matter here only once that
    # point is in the corner (before it, Q's approach braking branch is what c3 replaces).
    use_q = (s + ahead >= s_c) & (v_t < vt_c)
    vt = jnp.where(before, jnp.where(use_q, v_t, vt_c), v_t)
    at = jnp.where(before, jnp.where(use_q, a_t, at_c), a_t)
    return vt, at


def q_target_profile(kappa: np.ndarray, tables: ModelTables | CarTables, c_v: float,
                     ds: float = DS, kappa_cap: float = np.inf, brake_shift: float = 0.0
                     ) -> np.ndarray:
    """The policy target [N] (m/s) on one strip's knot samples: min(v_top, c_v V_grip(s)),
    V_grip the backward QSS pass on |kappa| capped at ``kappa_cap`` (c2); ``brake_shift``
    moves every braking branch that many metres towards its corner (c1), keeping the
    cornering limit. The forward pass is in ``forward_profile``."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    rows = target_profiles(np.asarray(kappa, float)[None], car, c_v, ds, kappa_cap,
                           brake_shift)
    return rows[0]


def target_profiles(kappa_rows: np.ndarray, car: CarTables, c_v: float, ds: float = DS,
                    kappa_cap: float = np.inf, brake_shift: float = 0.0) -> np.ndarray:
    """``q_target_profile`` for a stack of strips [B, N] (one compiled pass, float32)."""
    with jax.enable_x64(False):
        T, vlim = _target_rows(jnp.asarray(kappa_rows, jnp.float32), _as_jax(car),
                               jnp.float32(c_v), jnp.float32(kappa_cap), jnp.float32(ds))
    T, vlim = np.asarray(T, float), np.asarray(vlim, float)
    if brake_shift:
        T = np.stack([_shift_braking(t, vl, ds, brake_shift) for t, vl in zip(T, vlim)])
    return T


def _shift_braking(T: np.ndarray, vlim: np.ndarray, ds: float, shift: float) -> np.ndarray:
    """c1: on braking-branch knots (T falling, below the cornering limit) take T(s - shift),
    capped by the cornering limit at s; elsewhere T is unchanged."""
    s = np.arange(len(T)) * ds
    falling = np.r_[T[:-1] > T[1:] + 1e-6, False]
    braking = falling & (T < vlim * (1.0 - 1e-6))
    late = np.minimum(np.interp(s - shift, s, T, left=T[0]), vlim)
    return np.where(braking, np.maximum(late, T), T)


def forward_profile(v_target: np.ndarray, kappa: np.ndarray, s0: float, v0: float,
                    tables: ModelTables | CarTables, ds: float = DS) -> np.ndarray:
    """Q's predicted speed [N] (m/s, NaN before s0): from (s0, v0), the smaller of its target
    and the acceleration envelope a_acc(v) sqrt(1 - lat^2) (the forward QSS pass)."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    n = len(v_target)
    s = np.arange(n) * ds
    out = np.full(n, np.nan)
    i0 = int(np.ceil(s0 / ds - 1e-9))
    v = min(float(v0), float(np.interp(s0, s, v_target)))
    v_prev, s_prev = v, s0
    k_abs = np.abs(np.asarray(kappa, float))
    for i in range(i0, n):
        h = s[i] - s_prev
        ay = max(float(np.interp(v_prev, car.v, car.a_y)), _AY_FLOOR)
        lat = min(v_prev**2 * np.interp(s_prev, s, k_abs) / ay, 1.0)
        a = np.interp(v_prev, car.v, car.a_acc) * np.sqrt(1.0 - lat**2)
        v = min(float(v_target[i]), np.sqrt(max(v_prev**2 + 2.0 * max(a, 0.0) * h, 0.0)))
        out[i] = v
        v_prev, s_prev = v, s[i]
    return out


def offline_prediction(v_target: np.ndarray, kappa: np.ndarray, landmarks: Landmarks,
                       s0: float, v0: float, key_metrics: Any,
                       tables: ModelTables | CarTables, ds: float = DS) -> dict[str, float]:
    """Q's predicted e_ve, e_vmin and dd_b from its own profile against the key (protocol 7:
    the harness values must match within 2 points or one control step). The predicted
    onset is the start of the last falling run of the predicted speed before s_apex.
    ``key_metrics`` has attributes (or keys) v_e, v_min and d_b."""
    v = forward_profile(v_target, kappa, s0, v0, tables, ds)
    s = np.arange(len(v)) * ds
    ok = np.isfinite(v)
    sv, vv = s[ok], v[ok]
    v_e = float(np.interp(landmarks.s_c0, sv, vv))
    win = (sv >= landmarks.s_c0 - 50.0) & (sv <= landmarks.s_c1 + 50.0)
    v_min = float(vv[win].min()) if win.any() else np.nan
    falling = np.r_[vv[1:] < vv[:-1] - 1e-6, False]
    before = sv < landmarks.s_apex
    idx = np.nonzero(falling & before)[0]
    if len(idx):
        k = idx[-1]
        while k > 0 and falling[k - 1]:
            k -= 1
        s_b = float(sv[k])
        d_b, v_b = landmarks.s_c0 - s_b, float(vv[k])
    else:
        d_b, v_b = np.nan, np.nan

    def get(name):
        return float(key_metrics[name] if isinstance(key_metrics, dict)
                     else getattr(key_metrics, name))

    return {"v_e": v_e, "v_min": v_min, "d_b": d_b, "v_b": v_b,
            "e_ve": v_e / get("v_e") - 1.0, "e_vmin": v_min / get("v_min") - 1.0,
            "dd_b": d_b - get("d_b")}


def offline_match(pred: dict[str, float], measured: dict[str, float], dt: float = DT
                  ) -> dict[str, bool]:
    """Protocol 7's Q verdict 'offline_match': harness e_ve and e_vmin within 2 percentage
    points of the prediction, dd_b within one control step (v_b dt). A measure undefined on
    both sides (no braking onset in a flat cell) agrees; undefined on one side does not."""
    vb = measured.get("v_b", pred["v_b"])
    vb = pred["v_b"] if not np.isfinite(vb) else vb
    step = abs(vb) * dt

    def close(name: str, tol: float) -> bool:
        a, b = float(measured[name]), float(pred[name])
        if np.isnan(a) and np.isnan(b):
            return True
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)

    return {"e_ve": close("e_ve", 0.02), "e_vmin": close("e_vmin", 0.02),
            "dd_b": close("dd_b", step)}


def offline_match_runs(runs, keys, bank: StimulusBank, cells: list[Cell], car: CarTables,
                       c_v: float, tiers: tuple[str, ...] = ("A", "B")) -> dict:
    """Q's 'offline_match' over its deterministic tier A and B runs with a verified u = 0
    key: each run's harness e_ve, e_vmin and dd_b (against the key table's u = 0 key) vs
    ``offline_prediction`` from Q's own target profile from the run's start (s0, v0).
    Passed when every compared run matches on all three (protocol 7)."""
    by = {c.cell_id: c for c in cells}
    kidx = {int(c): i for i, c in enumerate(np.asarray(keys["cell"]))}
    rows: dict[int, np.ndarray] = {}
    n, fails = 0, []
    det = np.asarray(runs["mode"]).astype(str) == "det"
    for i in np.flatnonzero(det):
        c = by.get(int(runs["cell"][i]))
        k = kidx.get(int(runs["cell"][i]))
        if c is None or k is None or not (set(c.tiers) & set(tiers)):
            continue
        if str(keys["key_status"][k]) != "verified" or bool(runs["crashed"][i]):
            continue
        km = {"v_e": float(keys["v_e_star"][k]), "v_min": float(keys["v_min_star"][k]),
              "d_b": float(keys["d_b_star"][k])}
        if not (np.isfinite(km["v_e"]) and np.isfinite(km["v_min"])):
            continue
        if c.stim_id not in rows:
            rows[c.stim_id] = target_profiles(np.asarray(bank.kappa[c.stim_id])[None], car, c_v)[0]
        st = bank.stimuli[c.stim_id]
        pred = offline_prediction(rows[c.stim_id], np.asarray(bank.kappa[c.stim_id]),
                                  st.landmarks, float(runs["s0"][i]), float(c.v0), km, car)
        meas = {"e_ve": float(runs["v_e"][i]) / km["v_e"] - 1.0,
                "e_vmin": float(runs["v_min"][i]) / km["v_min"] - 1.0,
                "dd_b": float(runs["d_b"][i]) - km["d_b"], "v_b": float(runs["v_b"][i])}
        ok = offline_match(pred, meas)
        n += 1
        if not all(ok.values()):
            fails.append(dict(cell=int(c.cell_id), phase=int(runs["phase"][i]),
                              bad=[m for m, v in ok.items() if not v],
                              pred={m: pred[m] for m in ("e_ve", "e_vmin", "dd_b")},
                              measured={m: meas[m] for m in ("e_ve", "e_vmin", "dd_b")}))
    return dict(passed=bool(n and not fails), n=n, n_fail=len(fails), fails=fails[:20])


def q_params_for_kappa(kappa_rows: np.ndarray, s_corner: np.ndarray,
                       tables: ModelTables | CarTables, c_v: float, variant: str = "Q",
                       tau_c: float = 0.0, ds: float = DS) -> QParams:
    """QParams from strips' knot samples [n_rows, N] (one target row per strip). Used by
    ``q_params`` and by the probes, which build shifted strips of their own."""
    if variant not in Q_VARIANTS:
        raise ValueError(f"unknown Q variant {variant!r}")
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    cap = C2_KAPPA_CAP if variant == "c2" else np.inf
    shift = C1_SHIFT if variant == "c1" else 0.0
    rows = target_profiles(np.atleast_2d(kappa_rows), car, c_v, ds, cap, shift)
    f32 = jnp.float32
    return QParams(
        v_target=jnp.asarray(rows, f32), s_corner=jnp.asarray(s_corner, f32),
        car=_as_jax(car), c_v=f32(c_v), k_v=f32(K_V), lookahead_min=f32(LOOKAHEAD_MIN),
        lookahead_time=f32(LOOKAHEAD_TIME), lead_time=f32(LEAD_TIME), kappa_cap=f32(cap),
        tau_c=f32(tau_c if variant == "c3" else 0.0))


def q_params(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
             c_v: float, variant: str = "Q", tau_c: float = 0.0
             ) -> tuple[QParams, np.ndarray]:
    """QParams for a run list plus the aux row of each cell (one profile per stimulus):
    rows follow the sorted unique stim_ids of ``cells``; s_corner is each stimulus's s_c0."""
    stims = sorted({c.stim_id for c in cells})
    row_of = {s: i for i, s in enumerate(stims)}
    kappa = np.asarray(bank.kappa)[stims]
    s_corner = np.array([bank.stimuli[s].landmarks.s_c0 for s in stims])
    prm = q_params_for_kappa(kappa, s_corner, tables, c_v, variant, tau_c, bank.ds)
    return prm, np.array([row_of[c.stim_id] for c in cells], np.int32)


# ------------------------------------------------------------------------- T


class TapeParams(NamedTuple):
    acts: jax.Array          # [n_rows, W_max, 2] the nominal key's u = 0 commands


def make_tape_policy() -> Policy:
    """T: returns acts[aux, state.t] (params TapeParams); the last command is held if the
    run outlives its tape."""

    def policy(params: TapeParams, o, state: racing.EnvState, tr: TrackArrays, aux):
        t = jnp.minimum(state.t, params.acts.shape[1] - 1)
        mean = params.acts[aux, t].astype(o.dtype)
        return mean, jnp.full((2,), -jnp.inf, o.dtype)

    return policy


def tape_params(tapes: list[np.ndarray]) -> TapeParams:
    """Stack key command sequences [W_i, 2] into TapeParams, padding each with its last
    command; row i is aux = i."""
    w = max(len(a) for a in tapes)
    out = np.zeros((len(tapes), w, 2), np.float32)
    for i, a in enumerate(tapes):
        a = np.asarray(a, np.float32)
        out[i, :len(a)] = a
        out[i, len(a):] = a[-1]
    return TapeParams(jnp.asarray(out))


# ------------------------------------------------------------------------- K


class KnnParams(NamedTuple):
    obs: jax.Array           # [M, 222] S1-normalised buffer observations
    act: jax.Array           # [M, 2] the buffer's actions (S1's mean commands)
    sq_norm: jax.Array       # [M] |obs|^2 (distance by the dot-product identity)
    obs_rms_mean: jax.Array  # [222] S1's obs_rms
    obs_rms_var: jax.Array   # [222]


def knn_params(buffer: KnnBuffer, obs_rms_mean, obs_rms_var, sampled: bool = False
               ) -> KnnParams:
    """KnnParams of a buffer; ``sampled`` uses the buffer's sampled actions instead of the
    mean commands."""
    obs = jnp.asarray(buffer.obs, jnp.float32)
    act = buffer.act_sampled if sampled else buffer.act
    return KnnParams(obs=obs, act=jnp.asarray(act, jnp.float32),
                     sq_norm=jnp.sum(obs * obs, axis=1),
                     obs_rms_mean=jnp.asarray(obs_rms_mean, jnp.float32),
                     obs_rms_var=jnp.asarray(obs_rms_var, jnp.float32))


def make_knn_policy(k: int = 1) -> Policy:
    """K: normalise o with S1's obs_rms (``RunningMeanStd.normalize``, clip 10), nearest
    neighbour(s) in the buffer (params KnnParams), the neighbour's action as the mean; with
    k > 1 the inverse-distance-weighted mean of the k nearest (the protocol's k = 5
    fallback). ``k`` is static (top_k needs it at trace time)."""

    def policy(params: KnnParams, o, state: racing.EnvState, tr: TrackArrays, aux):
        z = jnp.clip((o - params.obs_rms_mean) / jnp.sqrt(params.obs_rms_var + 1e-8),
                     -OBS_CLIP, OBS_CLIP)
        d2 = params.sq_norm - 2.0 * (params.obs @ z) + jnp.dot(z, z)
        if k == 1:
            mean = params.act[jnp.argmin(d2)]
        else:
            neg, idx = jax.lax.top_k(-d2, k)
            w = 1.0 / jnp.maximum(jnp.sqrt(jnp.maximum(-neg, 0.0)), 1e-6)
            mean = jnp.sum(w[:, None] * params.act[idx], axis=0) / jnp.sum(w)
        return mean.astype(o.dtype), jnp.full((2,), -jnp.inf, o.dtype)

    return policy


@dataclass
class KnnBuffer:
    """A specialist's Hockenheim buffer (protocol 2.4 P4, 7): 40,000 observations subsampled
    from 50 stochastic laps started at random s at 60-100% of v_ref, normalised with its
    obs_rms; also the leave-one-lap-out median nearest-neighbour distance (d_OOD's
    denominator)."""

    obs: np.ndarray          # [M, 222] normalised
    act: np.ndarray          # [M, 2] the policy's mean command at that state
    lap: np.ndarray          # [M] int, lap index (leave-one-lap-out)
    lolo_median_nn: float    # median NN distance with the own lap excluded
    act_sampled: np.ndarray | None = None   # [M, 2] the sampled action that was applied
    agent: str = "S1"
    meta: dict = field(default_factory=dict)

    def save(self, path) -> None:
        """Atomic write, with ``meta`` (snapshot path, timesteps, laps) as JSON: the K
        control and d_OOD keep their provenance."""
        import json
        from pathlib import Path

        from apexzero.interp.io import _atomic, _jsonable

        arrays = dict(obs=self.obs, act=self.act, lap=self.lap,
                      act_sampled=self.act if self.act_sampled is None else self.act_sampled,
                      lolo_median_nn=np.float64(self.lolo_median_nn), agent=np.str_(self.agent),
                      meta_json=np.str_(json.dumps(_jsonable(self.meta), sort_keys=True)))
        _atomic(Path(path), lambda fh: np.savez(fh, **arrays))

    @classmethod
    def load(cls, path) -> KnnBuffer:
        import json

        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
            return cls(obs=z["obs"], act=z["act"], lap=z["lap"],
                       lolo_median_nn=float(z["lolo_median_nn"]),
                       act_sampled=z["act_sampled"], agent=str(z["agent"]), meta=meta)


def nn_distance(query: np.ndarray, ref: np.ndarray, query_group: np.ndarray | None = None,
                ref_group: np.ndarray | None = None, chunk: int = 512) -> np.ndarray:
    """Euclidean distance from each query row to its nearest ref row (float32 matmul,
    chunked); with groups, refs of the query's own group are excluded."""
    q = np.asarray(query, np.float32)
    r = np.asarray(ref, np.float32)
    r2 = np.sum(r * r, axis=1)
    out = np.empty(len(q))
    for a in range(0, len(q), chunk):
        qa = q[a:a + chunk]
        d2 = np.sum(qa * qa, axis=1)[:, None] + r2[None, :] - 2.0 * (qa @ r.T)
        if query_group is not None:
            d2[query_group[a:a + chunk, None] == ref_group[None, :]] = np.inf
        out[a:a + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def assemble_buffer(obs: np.ndarray, act: np.ndarray, lap: np.ndarray,
                    act_sampled: np.ndarray | None = None, n_obs: int = 40_000,
                    rng: np.random.Generator | None = None, agent: str = "S1") -> KnnBuffer:
    """Subsample ``n_obs`` rows uniformly without replacement (all rows if fewer) and compute
    the leave-one-lap-out median NN distance."""
    rng = rng or np.random.default_rng(0)
    n = len(obs)
    keep = np.sort(rng.choice(n, size=min(n_obs, n), replace=False))
    o, a, g = obs[keep], act[keep], np.asarray(lap)[keep]
    nn = nn_distance(o, o, g, g)
    return KnnBuffer(obs=np.asarray(o, np.float32), act=np.asarray(a, np.float32), lap=g,
                     lolo_median_nn=float(np.median(nn[np.isfinite(nn)])),
                     act_sampled=None if act_sampled is None
                     else np.asarray(act_sampled[keep], np.float32), agent=agent)


def collect_laps(actor_fn: Callable, tr: TrackArrays, v_ref: np.ndarray, lap_length: float,
                 n_laps: int, key: jax.Array, obs_rms_mean, obs_rms_var,
                 ctx: np.ndarray | None = None, max_steps: int | None = None,
                 params: racing.EnvParams = racing.EnvParams()) -> dict[str, np.ndarray]:
    """Stochastic laps from ``racing.reset`` starts (random s, lateral offset and heading,
    60-100% of v_ref): each runs until it has covered ``lap_length`` or terminates.
    ``actor_fn(o_norm) -> (mean, log_std)``. Returns normalised obs, mean and sampled
    actions and the lap index of every alive step."""
    ctx = np.asarray(racing.default_context(), np.float32) if ctx is None else ctx
    max_steps = max_steps or int(lap_length / 20.0 / params.dt) + 200
    eval_params = params._replace(window_steps=10**9)
    mean_, var_ = jnp.asarray(obs_rms_mean), jnp.asarray(obs_rms_var)
    v_ref = jnp.asarray(v_ref, jnp.float32)

    def one(k):
        k_reset, k_noise = jax.random.split(k)
        _, st = racing.reset(k_reset, tr, v_ref, jnp.asarray(ctx), eval_params)
        s_start = st.x[0]

        def body(c, t):
            st, alive = c
            p = racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu)
            o = jnp.clip((racing.observe(st, tr, p) - mean_) / jnp.sqrt(var_ + 1e-8),
                         -OBS_CLIP, OBS_CLIP)
            mean, log_std = actor_fn(o)
            eps = jax.random.normal(jax.random.fold_in(k_noise, t), (2,))
            a = mean + jnp.exp(log_std) * eps
            _, st2, _, term, _, _ = racing.step_no_reset(st, a, tr, eval_params)
            done = st2.x[0] - s_start >= lap_length
            keep = alive
            alive2 = alive & ~term & ~done
            st2 = jax.tree.map(lambda n_, o_: jnp.where(alive, n_, o_), st2, st)
            return (st2, alive2), (o, mean, jnp.clip(a, -1.0, 1.0), keep)

        _, out = jax.lax.scan(body, (st, jnp.asarray(True)), jnp.arange(max_steps))
        return out

    keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(n_laps))
    o, mean, a, keep = (np.asarray(z) for z in jax.jit(jax.vmap(one))(keys))
    lap = np.broadcast_to(np.arange(n_laps)[:, None], keep.shape)
    m = keep.astype(bool)
    return {"obs": o[m], "act": mean[m], "act_sampled": a[m], "lap": lap[m]}


def build_knn_buffer(agent: LoadedAgent, n_laps: int = 50, n_obs: int = 40_000,
                     track: Track | None = None) -> KnnBuffer:
    """A specialist's Hockenheim buffer from its own stochastic laps, normalised with its own
    obs_rms. S1's (for K and S1's d_OOD) is the protocol's second allowed pre-tag rollout
    (Integrate phase only); S2's and S3's are built after the tag. Seeds from
    ``io.stream_key('knn_buffer')`` / ``stream_rng`` folded with the agent id; v_ref is the
    training reference (``adapters.reference_speed``)."""
    from apexzero.agents.ppo import Actor
    from apexzero.env.adapters import reference_speed
    from apexzero.interp import io
    from apexzero.interp.stimuli import real_track
    from apexzero.vehicle.sim import track_arrays
    from apexzero.viz.export import lockbox_guard

    track = track if track is not None else real_track("Hockenheim")
    lockbox_guard(track.name)
    io.assert_x64_off()
    tr = track_arrays(track)
    actor = Actor(agent.cfg, 2, False)
    agent_no = sum(ord(c) for c in agent.agent_id)

    def actor_fn(o):
        return actor.apply(agent.actor_params, o)

    key = jax.random.fold_in(io.stream_key("knn_buffer"), agent_no)
    laps = collect_laps(actor_fn, tr, reference_speed(track), float(track.length), n_laps,
                        key, agent.obs_rms.mean, agent.obs_rms.var)
    rng = np.random.default_rng([*io.stream_rng("knn_buffer").integers(0, 2**31, 2),
                                 agent_no])
    buf = assemble_buffer(laps["obs"], laps["act"], laps["lap"], laps["act_sampled"],
                          n_obs, rng, agent.agent_id)
    buf.meta = {"track": track.name, "n_laps": n_laps, "n_steps": len(laps["obs"]),
                "timesteps": int(agent.timesteps), "path": str(agent.path)}
    return buf


def d_ood(obs_norm: np.ndarray, buffer: KnnBuffer) -> np.ndarray:
    """Nearest-neighbour distance of each normalised observation [n, 222] to the buffer,
    divided by the buffer's leave-one-lap-out median (P4)."""
    return nn_distance(obs_norm, buffer.obs) / buffer.lolo_median_nn


# ------------------------------------------------------------------------- specs, tuning


@dataclass(frozen=True)
class ControlSpec:
    """A control and the verdicts it must get (protocol 7, required verdicts block the tag)."""

    name: str
    tiers: tuple[str, ...]
    required: dict[str, tuple[str, ...]] = field(default_factory=dict)  # test -> accepted verdicts


CONTROLS = {
    "T": ControlSpec("T", ("A", "G"), {"regret": ("zero",), "errors": ("within_tau",),
                                       "termination": ("none",)}),
    "Q": ControlSpec("Q", ("A", "B", "D"), {"offline_match": ("pass",),
                                            "P1a": ("confirm", "cautious but lawful"),
                                            "P3": ("confirm",), "P4": ("class 4", "class 5"),
                                            "P5": ("symmetric",)}),
    "Qp": ControlSpec("Qp", ("A", "O"), {"runs": ("clean",)}),
    "c1": ControlSpec("c1", ("A", "B"), {"P2": ("confirm",), "P2b": ("onset_share>0.5",)}),
    "c2": ControlSpec("c2", ("A",), {"P1b": ("confirm",), "P1a": ("confirm",)}),
    "c3": ControlSpec("c3", ("A", "B"), {"P3": ("refute:M_idx",)}),
    "K": ControlSpec("K", ("D",), {"P4": ("class 2", "class 3")}),
}


def run_control(policy: Policy, params: Any, aux_of_cell: np.ndarray, bank: StimulusBank,
                cells: list[Cell], mode: str = "det", record: str = "full",
                samples: int = 16) -> dict[str, np.ndarray]:
    """A control through the harness: every phase of ``cells`` in ``mode`` (raw
    observations), each run's aux set to its cell's row. Returns the flat group arrays of
    ``rollout.run_rollouts`` (rows in plan order)."""
    from apexzero.interp import rollout

    aux = {c.cell_id: int(a) for c, a in zip(cells, aux_of_cell)}
    runs = rollout.plan_runs(cells, bank, mode, samples, aux=aux)
    return rollout.run_rollouts(policy, params, None, bank, cells, runs, record)


def completion(arrays: dict[str, np.ndarray], bank: StimulusBank) -> np.ndarray:
    """Per run: no termination before s_c1 + 250 m (protocol 2.4 P4). ``arrays`` holds 'stim',
    'term_step' and 'x' [R, T+1, 8] (state k + 1 is the post-step state of step k)."""
    stim, term, x = arrays["stim"], arrays["term_step"], arrays["x"]
    out = np.ones(len(stim), bool)
    for r in range(len(stim)):
        if term[r] >= 0:
            s_end = float(x[r, term[r] + 1, 0])
            out[r] = s_end >= bank.stimuli[int(stim[r])].landmarks.s_c1 + COMPLETE_AFTER_EXIT
    return out


def tune_cv(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
            completion_fn: Callable[[float], np.ndarray] | None = None
            ) -> tuple[float | None, dict[float, float]]:
    """The largest c_v in CV_CHOICES giving 100% deterministic completion on ``cells`` (tier A
    and tier D condition (a), every phase; step 4). ``completion_fn(c_v)`` returns the
    per-run completion; by default Q runs through the harness. Returns (c_v or None if no
    choice completes everything, {c_v: completion rate} of the choices tried)."""
    if completion_fn is None:
        car = tables if isinstance(tables, CarTables) else car_tables(tables)
        policy = make_q_policy("Q")

        def completion_fn(c_v: float) -> np.ndarray:
            prm, aux = q_params(bank, cells, car, c_v)
            return completion(run_control(policy, prm, aux, bank, cells), bank)

    rates: dict[float, float] = {}
    for c_v in CV_CHOICES:
        done = np.asarray(completion_fn(c_v), bool)
        rates[c_v] = float(done.mean()) if done.size else float("nan")
        if done.size and done.all():
            return c_v, rates
    return None, rates


def _tau_cell(bank: StimulusBank, cells: list[Cell]) -> Cell:
    """The cell c3 is fitted on: R 22.9 m, 90 deg, A1, left-hand (or the only cell given)."""
    if len(cells) == 1:
        return cells[0]
    for c in cells:
        sp = bank.stimuli[c.stim_id].spec
        if (sp is not None and abs(sp.R - 22.9) < 1e-6 and abs(sp.theta_deg - 90.0) < 1e-6
                and c.approach == "A1" and sp.direction == 1 and sp.ramp_kind == "typical"
                and not sp.no_cap and sp.w_in_corner is None and sp.width == 12.0):
            return c
    raise ValueError("no R 22.9 / 90 deg / A1 / left cell among the cells given")


def fit_tau_c(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
              d_b_key: float, c_v: float, d_b_fn: Callable[[float], float] | None = None,
              tol: float = 0.5, bracket: tuple[float, float] = TAU_C_BRACKET,
              scan_step: float = 0.25, max_iter: int = 40) -> float:
    """tau_c such that c3's d_b matches the key's at R 22.9 m, 90 deg, A1. ``d_b_fn(tau_c)``
    is c3's deterministic u = 0 onset distance (m); by default c3 runs through the harness
    and ``measures`` reads the onset. d_b rises with tau_c only while c3's braking is hard
    enough to pass the beta_f > 0.3 threshold from the trigger on (a long tau_c brakes
    gently, so its measured onset comes late), so the first upward crossing of d_b* on a
    ``scan_step`` grid over ``bracket`` is located and then bisected."""
    if d_b_fn is None:
        cell = _tau_cell(bank, cells)
        car = tables if isinstance(tables, CarTables) else car_tables(tables)
        d_b_fn = _harness_d_b(bank, cell, car, c_v)
    grid = np.arange(bracket[0], bracket[1] + 1e-9, scan_step)
    lo = hi = None
    f_prev = None
    for tau in grid:
        f = d_b_fn(float(tau)) - d_b_key
        if f_prev is not None and np.isfinite(f) and np.isfinite(f_prev) and f_prev < 0 <= f:
            lo, hi = float(tau - scan_step), float(tau)
            break
        if np.isfinite(f) and abs(f) <= tol:
            return float(tau)
        f_prev = f
    if lo is None:
        raise ValueError(f"c3's d_b does not cross d_b* = {d_b_key:.1f} m for tau_c in "
                         f"{bracket} s")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f = d_b_fn(mid) - d_b_key
        if abs(f) <= tol:
            return mid
        lo, hi = (mid, hi) if f < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _harness_d_b(bank: StimulusBank, cell: Cell, car: CarTables, c_v: float
                 ) -> Callable[[float], float]:
    """c3's u = 0 deterministic d_b through the harness and ``measures.response_metrics``."""
    from dataclasses import replace

    from apexzero.interp import measures

    policy = make_q_policy("c3")
    stim = bank.stimuli[cell.stim_id]
    one = replace(cell, n_phases=1, P=cell.P or 0.0)

    def d_b(tau_c: float) -> float:
        prm, aux = q_params(bank, [one], car, c_v, "c3", tau_c)
        arr = run_control(policy, prm, aux, bank, [one])
        traj = measures.trajectory_from_rollout(arr, 0, "det", source="control")
        marks = measures.make_marks(stim, bank.w_left[cell.stim_id], bank.w_right[cell.stim_id],
                                    float(arr["s0"][0]), int(arr["W"][0]))
        return float(measures.response_metrics(traj, marks).d_b)

    return d_b
