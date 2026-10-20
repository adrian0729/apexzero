"""Scripted positive controls run through the identical harness (protocol 7).

A classifier is only trusted if it returns the planted verdict on a driver whose law is
known. These controls are those drivers. Each is a vmappable JAX function with the
harness's Policy signature ``policy(params, o, state, tr, aux) -> (mean [2], log_std [2])``
(log_std = -inf: no native noise), so it runs through stimulus bank -> rollout -> measures
-> analysis exactly as an agent does. Controls get the raw observation (the rollout is
called with obs_rms = None), except K, which normalises with S1's obs_rms itself.

    T   tape: the nominal key's actions at u = 0, replayed open-loop (aux = tape row). Ruling
        R20 splits it: T32 is the float32 harness path the agents use (its measure check on
        runs reaching s_c1 + 250 m blocks; crashes are reported); T64 is the same harness
        code in its own float64 process (``rollout.run_tape_rollouts_x64``), the blocking
        exactness check (regret / |J*| <= 1e-6, errors within tau, no termination)
    Q   QSS on the full road: pure pursuit on the centreline, lookahead max(8 m, 0.6 s v);
        speed target c_v v_QSS(s) from a forward-backward pass with a_y,max(v) and a_dec(v)
        (measures.ModelTables) on the full-resolution kappa; pedal = 0.5 per m/s error plus
        feed-forward, clipped (aux = target-profile row); the speed error is predicted
        through the pedal lag (R22 A, ``law_a``)
    Qp  Q, but the kappa ahead is rebuilt each step from the 40 preview kappa samples only
        (linear in d, last sample held beyond d_max); the target is recomputed every step
    c1  Q with the braking branch of its target moved towards the corner by 15 m plus Q's
        offline-predicted onset lead over the u = 0 nominal key, per cell (R23,
        ``c1_shifts``): 15 m late against the key, about 27-41 m against Q
    c2  Q reading kappa capped at 1/12 1/m (saturating target); judged by P1a = Q's verdict
        and the planted-law check (R24, ``c2_planted_law``)
    c3  a pure time-to-corner trigger: flat out until the time to the corner <= tau_c, then
        braking at the constant fraction k of the car's braking capacity that reaches Q's
        own (lawful) entry speed at the corner; tau_c is the smallest value that keeps
        k <= 1 on every tier A/B braking cell, from Q's offline profile (R26, ``tau_c_k1``)
    K   1-nearest-neighbour on S1's Hockenheim buffer (normalised obs -> the neighbour's
        action); k = 5 with distance weights if C(a) < 0.9, C(a) logged at both k
        (``knn_fallback``)

Ruling R30's planted crash controls (``PLANTED``; deterministic u = 0 runs, one planted
fault each, on cells chosen by geometry: ``planted_cells``), with their viability and
counterfactual solves; they check that overspeed is not a catch-all:

    OB    Q with the pedal held at -1 from s_c0 - 5 m: must type over-brake
    LS    Q with its steering scaled by 0.25 from s_c0 + 2 m: must type line/steer
    LSK   the u = 0 nominal key's float32 tape (T32) with its steering scaled by 0.3 from the
          step it passes s_c0 + 2 m: line/steer at the key's speed
    LSKF  LSK with the key's braking before s_c0 scaled by 0.9 (enters about 0.95 m/s fast,
          recoverable at s_c0, then the same fault): must not type overspeed

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
  gives Q's offline speeds (``forward_profile``). Its acceleration is the model's signed
  longitudinal acceleration in the steady turn at Q's throttle clip (``turn_accel_table``),
  so a power-limited speed loss in a fast corner is predicted (R22 B); below the table
  (30 m/s) or where no steady turn exists it falls back to the clamped envelope
  a_acc(v) sqrt(1 - lat^2).
- Q's offline onset (R22 C) replays Q's own pedal law on the straight-line model (float64,
  ``replay_1d``) and reads the onset with protocol 5's 3-step brake-demand rule,
  implemented here independently of ``measures`` (``brake_demand_1d``, ``onset_1d``). The
  one-step tolerance of ``offline_match`` uses the predicted v_b.
- Pedal: 0.5 per m/s on the error between the target read L = 0.075 s ahead (half the hold
  plus the pedal lag) and the car's speed predicted to that point, plus the feed-forward
  pedal for the target's acceleration there (steady straight-line pedal map of the model).
  The prediction goes through the pedal lag and includes the command being chosen (R22 A,
  no free parameter): v_pred(c) = v + L [(1 - phi) a_ss(v, pedal) + phi a_ss(v, c)], a_ss
  the model's steady straight-line acceleration, phi = 1 - (tau/L)(1 - exp(-L/tau)) =
  0.482 the mean weight of the new command over L, so c = clip(0.5 (v_t - v_pred(c)) + ff)
  is solved by bisection (monotone in c). The as-built predictor (v + L times the measured
  acceleration) fed the pedal back through the env's one-step acceleration: loop gain 2.8,
  |lambda| 1.09 while braking, a 3-step pedal oscillation that the onset rule caught as
  re-applications (dry run). "Clipped" is to the friction ellipse: between the pedal that
  saturates the brakes and the full-throttle pedal, each scaled by the axle's room
  sqrt(1 - (fy/fy_max)^2) at its current slip angle (``axle_room``).
- Steering: pure pursuit from the car's reference point to the centreline point at arc
  length s + max(8 m, 0.6 s v); the pursuit curvature is turned into the model's
  steady-state steering angle (kinematic angle plus the understeer map
  ``steer_correction``), clipped to the front tyre's peak slip and the lock.
- c3 brakes, once triggered, at the constant fraction of the car's braking capacity that
  reaches Q's lawful entry speed exactly at the corner start (pure feed-forward: the
  proportional term is off), and hands over to Q once Q's read-ahead point is in the
  corner. Before the trigger it is flat out (Q's law on the target v_top); Q's own braking
  branch is never used before the hand-over (no "earlier of" rule).
- Full-road target profiles are precomputed per stimulus and stacked into a [n_rows, N]
  table indexed by ``aux``; Qp runs its 251-point pass (2 m grid, 500 m) every step.
- K returns the buffer's *mean* action (the specialist's deterministic command at that
  state); the sampled action is kept in the buffer too.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.interp.agents import LoadedAgent
from apexzero.interp.constants import (
    BRAKE_MIN_STEPS,
    BRAKE_ON,
    COMPLETE_AFTER_EXIT,
    DS,
    DT,
    OBS_CLIP,
    OBS_PREVIEW,
    REPLAY_RTOL,
)
from apexzero.interp.measures import ModelTables, cornering_limit
from apexzero.interp.rollout import Policy
from apexzero.interp.stimuli import Cell, Landmarks, StimulusBank
from apexzero.tracks.track import Track
from apexzero.vehicle.dynamics import _mu, body_derivatives, tyre_forces
from apexzero.vehicle.ops import Ops
from apexzero.vehicle.params import F1_2025, VehicleParams
from apexzero.vehicle.sim import TrackArrays, global_pose, lookup

CONTROL_NAMES = ("T", "Q", "Qp", "c1", "c2", "c3", "K", "OB", "LS", "LSK", "LSKF")
Q_VARIANTS = ("Q", "Qp", "c1", "c2", "c3")
CV_CHOICES = (0.95, 0.90, 0.85, 0.80)
LOOKAHEAD_MIN = 8.0          # m
LOOKAHEAD_TIME = 0.6         # s
K_V = 0.5                    # pedal per m/s of speed error
C1_SHIFT = 15.0              # m, late onset against the u = 0 nominal key (R23)
C2_KAPPA_CAP = 1.0 / 12.0    # 1/m
LEAD_TIME = 0.5 * DT + F1_2025.tau_pedal   # s: zero-order hold + pedal lag
# R22 A: the weight of the command being chosen in the mean acceleration over LEAD_TIME
# through the first-order pedal lag, 1 - (tau/L)(1 - exp(-L/tau)) = 0.482 (no free parameter).
PHI = 1.0 - (F1_2025.tau_pedal / LEAD_TIME) * (1.0 - math.exp(-LEAD_TIME / F1_2025.tau_pedal))
PEDAL_BISECT = 30            # bisection steps of law A (2 / 2^30: below float32 resolution)
V_FREE = 150.0               # m/s: "no cornering limit" (above the car's top speed)
QP_DS = 2.0                  # m, Qp's grid spacing
QP_N = 251                   # Qp grid points: d in [0, 500] m (d_max <= 500)
TAU_C_BRACKET = (0.25, 4.0)  # s, searched range of c3's tau_c
TAU_C_ITERS = 24             # bisection steps of tau_c_k1 (4 s / 2^24)
# Review of R26: tau_c is the bisected threshold plus this guard. At the threshold itself the
# binding run triggers with a margin of ~1e-5 m, below float32 resolution of s (6.1e-5 m at
# 900 m), so whether the float32 harness reproduces k <= 1 was a rounding coin flip (one
# float32 step of tau lower gave k 1.37 on real A1 cells). 1e-3 s is 2% of a control step
# (about 8 cm at 80 m/s); a larger tau only triggers earlier, so k can only fall.
TAU_C_GUARD = 1e-3           # s
KNN_K_FALLBACK = 5
KNN_GATE = 0.9               # protocol 7: K falls back to k = 5 if C(a) < 0.9
# Versions of the Q family's law and of c3's tau_c rule, recorded with the controls stage's
# tuning and runs: saved c_v, tau_c and Q-family rollouts made under another version are
# redone (dry-run rulings R22 A and R26 change both; c_v is re-tuned under R14 after R22 A).
Q_LAW = "R22"
TAU_RULE = "R26+guard"       # R26's smallest tau_c plus TAU_C_GUARD (refits an unguarded one)

# Pedal map grids (uniform): 0 to 120 m/s (so c_v v_ss >= v_top beyond the grid for every
# c_v choice), accelerations -90..+40 m/s^2.
_PM_V = np.arange(0.0, 120.0 + 1e-9, 0.5)
_PM_A = np.arange(-90.0, 40.0 + 1e-9, 0.25)
_PM_P = np.linspace(-1.0, 1.0, 801)
_SAT_TOL = 1e-3              # saturation: within 0.1% of the extreme acceleration
_US_U = np.linspace(0.0, 0.95, 39)   # steady-state steering map: u = a_y / a_y,max(v)
_TURN_V = np.arange(30.0, 90.0 + 1e-9, 0.5)   # steady-turn acceleration table (R22 B)
_AY_FLOOR = 1e-3             # m/s^2: a_y,max is 0 at top speed (power-limited); no 0/0
_ENV = racing.EnvParams()    # dt and substeps of the env step (the 1-D replay's integrator)

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
    pm_p: Any         # [np] uniform pedal grid (_PM_P)
    pm_acc: Any       # [m, np] steady straight-line acceleration a_ss(v, pedal) (R22 A)
    turn_v: Any       # [nt] uniform speed grid of the steady-turn table (_TURN_V)
    turn_u: Any       # [nu] uniform grid of u = a_y / a_y,max(v) (_US_U)
    turn_a: Any       # [nt, nu] signed dvx/dt in the steady turn at Q's throttle clip
    turn_ok: Any      # [nt, nu] 1 where that steady turn exists (Newton converged), else 0


def pedal_map(p: VehicleParams = F1_2025, v_grid: np.ndarray = _PM_V,
              a_grid: np.ndarray = _PM_A) -> tuple[np.ndarray, np.ndarray]:
    """(pedal [nv, na], a_acc [nv]): the steady straight-line pedal giving acceleration a at
    speed v, restricted to the unsaturated range (accelerations beyond what the tyres give
    map to the pedal where the axle saturates, never further); and the full-throttle
    acceleration."""
    out, a_acc, _ = _pedal_tables(p, v_grid, a_grid)
    return out, a_acc


def _pedal_tables(p: VehicleParams, v_grid: np.ndarray, a_grid: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``pedal_map``'s (pedal, a_acc) and the forward map acc [nv, np] = a_ss(v, _PM_P)."""
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
    return out, a_acc, acc


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


def _room_np(alpha: np.ndarray, alpha_peak: float, p: VehicleParams) -> np.ndarray:
    """``axle_room`` of one axle in NumPy: sqrt(1 - u^2), u = sin(C atan(B |alpha|)), 1
    beyond the peak slip."""
    a = np.abs(alpha)
    use = np.where(a >= alpha_peak, 1.0, np.sin(p.pac_c * np.arctan(p.pac_b * a)))
    return np.sqrt(np.clip(1.0 - use * use, 0.0, 1.0))


def turn_accel_table(a_y: np.ndarray, a_acc: np.ndarray, pm_pedal: np.ndarray,
                     alpha_peak: float, p: VehicleParams = F1_2025,
                     v_grid: np.ndarray = _PM_V, a_grid: np.ndarray = _PM_A,
                     turn_v: np.ndarray = _TURN_V, u_grid: np.ndarray = _US_U
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(a [nt, nu], ok [nt, nu]): the model's signed longitudinal acceleration dvx/dt in the
    steady turn at lateral acceleration u a_y,max(v), driven at Q's throttle clip (the
    full-throttle acceleration a_acc(v) scaled by the rear axle's room at the turn's slip,
    as ``pedal_command``'s upper bound), and whether that steady turn exists (R22 B).
    Negative where cornering drag exceeds the power left: the speed loss of a power-limited
    fast corner. Newton on dvy = dr = 0 for (delta, vy) at r = a_y / |v|, continued in u;
    the pedal is the clip at the converged turn, iterated three times, then the turn is
    re-solved at the final pedal. ``ok`` needs a residual below 1e-5."""
    V = np.asarray(turn_v, float)
    ay_max = np.maximum(np.interp(V, v_grid, a_y), _AY_FLOOR)
    acc_top = np.interp(V, v_grid, a_acc)
    out = np.zeros((len(V), len(u_grid)))
    ok = np.zeros((len(V), len(u_grid)), bool)
    z = np.zeros((len(V), 2))                       # (delta, vy)

    def resid(z, ay, ped):
        vy = z[:, 1]
        r = ay / np.sqrt(V**2 + vy**2)
        x = np.zeros((len(V), 8))
        x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7] = V, vy, r, z[:, 0], ped
        u = np.stack([z[:, 0] / p.delta_max, ped], axis=-1)
        dvx, dvy, dr, _, _ = body_derivatives(x, u, p, _NP)
        return np.stack([dvy, dr], axis=-1), np.asarray(dvx, float), x

    def newton(z, ay, ped):
        for _ in range(12):
            f = resid(z, ay, ped)[0]
            J = np.empty((len(V), 2, 2))
            for k in range(2):
                h = np.zeros(2)
                h[k] = 1e-6
                J[:, :, k] = (resid(z + h, ay, ped)[0] - resid(z - h, ay, ped)[0]) / 2e-6
            step = np.einsum("nij,nj->ni", np.linalg.pinv(J + 1e-9 * np.eye(2)), -f)
            z = z + step
        return np.where(np.isfinite(z), z, 0.0)

    def throttle_clip(x):
        v_safe = np.maximum(x[:, 3], 1.0)
        a_r = np.arctan((x[:, 4] - p.lr * x[:, 5]) / v_safe)
        a = acc_top * _room_np(a_r, alpha_peak, p)
        return _bilinear(V, a, v_grid, a_grid, pm_pedal, np)

    for j, uj in enumerate(u_grid):
        ay = uj * ay_max
        ped = np.ones(len(V))
        for _ in range(3):
            z = newton(z, ay, ped)
            ped = throttle_clip(resid(z, ay, ped)[2])
        z = newton(z, ay, ped)
        f, dvx, _ = resid(z, ay, ped)
        out[:, j] = dvx
        ok[:, j] = np.isfinite(dvx) & (np.abs(f).max(axis=1) < 1e-5)
    return np.where(ok, out, 0.0), ok


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
    pm, a_acc, acc = _pedal_tables(p, v, _PM_A)
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
    turn_a, turn_ok = turn_accel_table(a_y, a_acc, pm, alpha_peak, p, v, _PM_A)
    return CarTables(v=v, a_dec=a_dec, a_y=a_y, a_acc=a_acc, kap_ss=kap[::-1].copy(),
                     v_ss=vs[::-1].copy(), pm_a=_PM_A.copy(), pm_pedal=pm,
                     us_u=_US_U.copy(), us_dd=us_dd, d_brake=d_brake, v_top=v_top,
                     alpha_peak=alpha_peak, pm_p=_PM_P.copy(), pm_acc=acc,
                     turn_v=_TURN_V.copy(), turn_u=_US_U.copy(), turn_a=turn_a,
                     turn_ok=turn_ok.astype(float))


def _as_jax(car: CarTables) -> CarTables:
    return CarTables(*(jnp.asarray(a, jnp.float32) for a in car))


def lock_radius(tables: ModelTables, p: VehicleParams = F1_2025) -> float:
    """The largest radius whose steady cornering limit is set by the steering lock: max
    v^2 / a_y,max over the model table's 'lock' entries (``ModelTables.limit_y``); the
    kinematic full-lock radius (lf + lr) / tan(delta_max) without labels. A centreline driver
    cannot hold a corner at or below it at its target (R24's lock-limited cells)."""
    lim = getattr(tables, "limit_y", None)
    if lim is not None:
        lock = np.asarray(lim).astype(str) == "lock"
        ay = np.asarray(tables.a_y_max, float)
        if np.any(lock & (ay > 0)):
            v = np.asarray(tables.v_lat, float)
            return float(np.max(v[lock & (ay > 0)] ** 2 / ay[lock & (ay > 0)]))
    return float((p.lf + p.lr) / np.tan(p.delta_max))


# ------------------------------------------------------------------------- shared pieces
# The interpolators take an array module (jnp for the policies, np for the offline replay),
# so the offline replay evaluates exactly the policy's formulas in float64.


def _uinterp(x, grid, table, xp=jnp):
    """Linear interpolation on a uniform grid (clamped at the ends)."""
    x0, dx = grid[0], grid[1] - grid[0]
    n = grid.shape[0]
    f = xp.clip((x - x0) / dx, 0.0, n - 1.0)
    i = xp.minimum(xp.floor(f).astype(xp.int32), n - 2)
    t = f - i
    return table[i] * (1.0 - t) + table[i + 1] * t


def _bilinear(x, y, xg, yg, table, xp=jnp):
    """table[i, j] at (xg[i], yg[j]) on uniform grids, bilinear, clamped at the edges."""
    nx, ny = table.shape
    fx = xp.clip((x - xg[0]) / (xg[1] - xg[0]), 0.0, nx - 1.0)
    ix = xp.minimum(xp.floor(fx).astype(xp.int32), nx - 2)
    tx = fx - ix
    fy = xp.clip((y - yg[0]) / (yg[1] - yg[0]), 0.0, ny - 1.0)
    iy = xp.minimum(xp.floor(fy).astype(xp.int32), ny - 2)
    ty = fy - iy
    return ((1 - tx) * ((1 - ty) * table[ix, iy] + ty * table[ix, iy + 1])
            + tx * ((1 - ty) * table[ix + 1, iy] + ty * table[ix + 1, iy + 1]))


def _pedal_for(v, a, car: CarTables, xp=jnp):
    """Steady straight-line pedal giving acceleration a at speed v (bilinear)."""
    return _bilinear(v, a, car.v, car.pm_a, car.pm_pedal, xp)


def _a_ss(v, pedal, car: CarTables, xp=jnp):
    """Steady straight-line acceleration of the model at speed v with pedal held (m/s^2)."""
    return _bilinear(v, pedal, car.v, car.pm_p, car.pm_acc, xp)


def law_a(v, pedal_state, v_t, a_t, p_lo, p_hi, car: CarTables, k_v, lead, phi, xp=jnp,
          iters: int = PEDAL_BISECT):
    """Q's pedal law as amended by ruling R22 A: the command c with
    c = clip(k_v (v_t - v_pred(c)) + ff(a_t), p_lo, p_hi), where
    v_pred(c) = v + lead [(1 - phi) a_ss(v, pedal_state) + phi a_ss(v, c)] is the speed at the
    read-ahead point through the first-order pedal lag, the command's own effect included.
    The right-hand side falls with c (a_ss rises with the pedal), so the fixed point is
    unique; it is found by bisection on [p_lo, p_hi]. ``xp`` jnp (policy) or np (replay)."""
    ff = _pedal_for(v, a_t, car, xp)
    base = k_v * (v_t - v - lead * (1.0 - phi) * _a_ss(v, pedal_state, car, xp)) + ff
    gain = k_v * lead * phi

    def h(c):                                   # increasing in c; the root is the command
        return c - base + gain * _a_ss(v, c, car, xp)

    def halve(lo, hi):
        mid = 0.5 * (lo + hi)
        up = h(mid) > 0.0
        return xp.where(up, lo, mid), xp.where(up, mid, hi)

    lo, hi = p_lo, p_hi
    if xp is jnp:
        lo, hi = jax.lax.fori_loop(0, iters, lambda _, lh: halve(*lh), (lo, hi))
    else:
        for _ in range(iters):
            lo, hi = halve(lo, hi)
    root = 0.5 * (lo + hi)
    return xp.where(h(p_lo) >= 0.0, p_lo, xp.where(h(p_hi) <= 0.0, p_hi, root))


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


def _at(T, x, h, xp=jnp):
    """(T(x), T dT/dx) on a uniform grid T[i] at i h: linear value, slope of the interval.
    A stack T [B, N] is read row by row at x [B]."""
    n = T.shape[-1]
    f = xp.clip(x / h, 0.0, n - 1.0)
    i = xp.minimum(xp.floor(f).astype(xp.int32), n - 2)
    t = f - i
    if T.ndim == 1:
        lo, hi = T[i], T[i + 1]
    else:
        r = xp.arange(T.shape[0])
        lo, hi = T[r, i], T[r, i + 1]
    return lo * (1.0 - t) + hi * t, (hi**2 - lo**2) / (2.0 * h)


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
    phi: jax.Array           # law A's weight of the new command over lead_time (PHI)


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


def pedal_command(state: racing.EnvState, v_t, a_t, params: QParams, ff_only=False):
    """Proportional speed control plus feed-forward, clipped to the friction ellipse.

    ``v_t`` and ``a_t`` are the target's speed and acceleration at the read-ahead point
    (``lead_time`` ahead); the proportional term (k_v per m/s) acts on the error against
    the car's speed predicted to that point through the pedal lag, the command's own effect
    included (``law_a``, ruling R22 A; the as-built prediction with the measured
    acceleration oscillated while braking). The feed-forward is the steady straight-line
    pedal for a_t. The clip keeps the pedal between the pedal that saturates the brakes at
    a_dec(v) times the room of the tighter axle, and the pedal that gives the full-throttle
    acceleration a_acc(v) times the rear axle's room (``axle_room``). Braking loads both
    axles, so a rear axle busy cornering (yaw and sideslip, e.g. a start taken mid-corner
    from a real lap) must limit it too, or braking spins the car; min(front, rear) is
    conservative because a_dec is front-limited. ``ff_only`` (c3's triggered brake) drops
    the proportional term: the clipped feed-forward pedal alone."""
    car = params.car
    x = state.x
    v = jnp.maximum(x[3], 0.0)
    room_f, room_r = axle_room(x, car)
    p_hi = _pedal_for(v, _uinterp(v, car.v, car.a_acc) * room_r, car)
    p_lo = _pedal_for(v, -_uinterp(v, car.v, car.a_dec) * jnp.minimum(room_f, room_r), car)
    c = law_a(v, x[7], v_t, a_t, p_lo, p_hi, car, params.k_v, params.lead_time, params.phi)
    if ff_only is False:
        return c
    return jnp.where(ff_only, jnp.clip(_pedal_for(v, a_t, car), p_lo, p_hi), c)


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
        ff_only = False
        if variant == "Qp":
            v_t, a_t = _at(preview_profile(o, v, params), ahead, QP_DS)
        else:
            row = params.v_target[aux]
            v_t, a_t = _at(row, x[0] + ahead, tr.ds)
            if variant == "c3":
                v_t, a_t, ff_only, _ = _c3_target(x[0], v, ahead, row, params.s_corner[aux],
                                                  params.tau_c, v_t, a_t, tr.ds, params.car)
        pedal = pedal_command(state, v_t, a_t, params, ff_only)
        mean = jnp.stack([steer, pedal]).astype(o.dtype)
        return mean, jnp.full((2,), -jnp.inf, o.dtype)

    return policy


def _c3_target(s, v, ahead, row, s_c, tau_c, v_t, a_t, ds, car: CarTables, xp=jnp):
    """c3 (ruling R26, a pure time-to-corner trigger): Q, except on the approach to the
    corner start s_c. There it is flat out (Q's law on the target v_top) until d_c <= tau_c
    v, then brakes at the constant fraction k of the car's capacity that reaches Q's own
    (lawful) entry speed v_e = row(s_c) exactly at s_c, k = (D(v) - D(v_e)) / d_c with D the
    straight-line braking distance at a_dec, as pure feed-forward (``ff_only``: the
    proportional term is off). Once Q's read-ahead point is past s_c, Q's target applies
    wherever it is below the car's speed (or below v_top before the trigger: corners whose
    lawful braking lies inside the ramp, where T(s_c) = v_top). k is recomputed every step,
    so the law is memoryless and corrects its own tracking error. Q's approach braking
    branch is never used (no "earlier of" rule). Returns (v_t, a_t, ff_only, k), k NaN
    where the feed-forward brake is not commanded. ``xp``: jnp (policy) or np (replay; then
    ``row`` is [B, N] and the rest [B])."""
    d_c = s_c - s
    v_e = _at(row, s_c, ds, xp)[0]
    before = d_c > 0.0
    trig = before & (d_c <= tau_c * v) & (v > v_e)
    k = ((_uinterp(v, car.v, car.d_brake, xp) - _uinterp(v_e, car.v, car.d_brake, xp))
         / xp.maximum(d_c, 1e-3))
    use_q = (s + ahead >= s_c) & (v_t < xp.where(trig, v, car.v_top))
    own = before & ~use_q
    ff_only = own & trig
    vt = xp.where(own, xp.where(trig, v, car.v_top), v_t)
    at = xp.where(own, xp.where(trig, -k * _uinterp(v, car.v, car.a_dec, xp), 0.0), a_t)
    return vt, at, ff_only, xp.where(ff_only, k, xp.nan)


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
                    kappa_cap: float = np.inf, brake_shift: float | np.ndarray = 0.0
                    ) -> np.ndarray:
    """``q_target_profile`` for a stack of strips [B, N] (one compiled pass, float32);
    ``brake_shift`` is one value or one per row (c1, ruling R23)."""
    with jax.enable_x64(False):
        T, vlim = _target_rows(jnp.asarray(kappa_rows, jnp.float32), _as_jax(car),
                               jnp.float32(c_v), jnp.float32(kappa_cap), jnp.float32(ds))
    T, vlim = np.asarray(T, float), np.asarray(vlim, float)
    shift = np.broadcast_to(np.asarray(brake_shift, float), (len(T),))
    if np.any(shift):
        T = np.stack([_shift_braking(t, vl, ds, sh) if sh else t
                      for t, vl, sh in zip(T, vlim, shift, strict=True)])
    return T


def _shift_braking(T: np.ndarray, vlim: np.ndarray, ds: float, shift: float) -> np.ndarray:
    """c1: on braking-branch knots (T falling, below the cornering limit) take T(s - shift),
    capped by the cornering limit at s; elsewhere T is unchanged."""
    if shift < 0.0:
        raise ValueError(f"c1's braking shift must be >= 0 m (got {shift:.2f})")
    s = np.arange(len(T)) * ds
    falling = np.r_[T[:-1] > T[1:] + 1e-6, False]
    braking = falling & (T < vlim * (1.0 - 1e-6))
    late = np.minimum(np.interp(s - shift, s, T, left=T[0]), vlim)
    return np.where(braking, np.maximum(late, T), T)


# ------------------------------------------------------------------------- Q offline


def _turn_accel(v, u, car: CarTables) -> tuple[np.ndarray, np.ndarray]:
    """(a, ok) from the steady-turn table at (v, u), bilinear; ok where all four
    neighbours converged and v is on the table's speed range."""
    tv, tu = np.asarray(car.turn_v, float), np.asarray(car.turn_u, float)
    ta, tok = np.asarray(car.turn_a, float), np.asarray(car.turn_ok, float)
    a = _bilinear(v, u, tv, tu, ta, np)
    fx = np.clip((v - tv[0]) / (tv[1] - tv[0]), 0.0, len(tv) - 1.0)
    ix = np.minimum(np.floor(fx).astype(np.int32), len(tv) - 2)
    fy = np.clip((u - tu[0]) / (tu[1] - tu[0]), 0.0, len(tu) - 1.0)
    iy = np.minimum(np.floor(fy).astype(np.int32), len(tu) - 2)
    ok = ((tok[ix, iy] > 0) & (tok[ix + 1, iy] > 0) & (tok[ix, iy + 1] > 0)
          & (tok[ix + 1, iy + 1] > 0) & (v >= tv[0]) & (v <= tv[-1]))
    return a, ok


def forward_profiles(v_target: np.ndarray, kappa: np.ndarray, s0: np.ndarray,
                     v0: np.ndarray, car: CarTables, ds: float = DS) -> np.ndarray:
    """Q's predicted speeds [B, N] (m/s, NaN before each s0; float64): from (s0, v0), knot by
    knot, the smaller of its target and the speed reached with the model's signed
    longitudinal acceleration in the steady turn at Q's throttle clip (``turn_accel_table``
    at u = v^2 |kappa| / a_y,max(v), u <= 0.95), so a power-limited corner loses speed
    (ruling R22 B). Off the table (v < 30 m/s) or where no steady turn exists, the clamped
    envelope max(0, a_acc(v) sqrt(1 - lat^2)) of the as-built pass."""
    T = np.atleast_2d(np.asarray(v_target, float))
    K = np.abs(np.atleast_2d(np.asarray(kappa, float)))
    s0 = np.broadcast_to(np.asarray(s0, float), (len(T),))
    v0 = np.broadcast_to(np.asarray(v0, float), (len(T),))
    n = T.shape[1]
    s = np.arange(n) * ds
    out = np.full(T.shape, np.nan)
    i0 = np.ceil(s0 / ds - 1e-9).astype(int)
    rows = np.arange(len(T))
    v = np.minimum(v0, np.array([np.interp(a, s, t) for a, t in zip(s0, T, strict=True)]))
    s_prev = s0.copy()
    k_prev = np.array([np.interp(a, s, k) for a, k in zip(s0, K, strict=True)])
    for i in range(int(max(i0.min(), 0)), n):
        on = i >= i0
        if not on.any():
            continue
        h = s[i] - s_prev
        ay = np.maximum(np.interp(v, car.v, car.a_y), _AY_FLOOR)
        lat = np.minimum(v * v * k_prev / ay, 1.0)
        env = np.maximum(np.interp(v, car.v, car.a_acc) * np.sqrt(1.0 - lat * lat), 0.0)
        a_turn, ok = _turn_accel(v, np.minimum(lat, float(car.turn_u[-1])), car)
        a = np.where(ok, a_turn, env)
        v_new = np.minimum(T[rows, i], np.sqrt(np.maximum(v * v + 2.0 * a * h, 0.0)))
        out[on, i] = v_new[on]
        v = np.where(on, v_new, v)
        s_prev = np.where(on, s[i], s_prev)
        k_prev = np.where(on, K[:, i], k_prev)
    return out


def forward_profile(v_target: np.ndarray, kappa: np.ndarray, s0: float, v0: float,
                    tables: ModelTables | CarTables, ds: float = DS) -> np.ndarray:
    """``forward_profiles`` of one strip [N]."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    return forward_profiles(np.asarray(v_target, float)[None], np.asarray(kappa, float)[None],
                            [s0], [v0], car, ds)[0]


def _straight_rates(v, pedal, cmd, p: VehicleParams):
    """(dvx/dt, dpedal/dt) of the full model driving straight (zero steer, no lateral
    state): the straight-line ('1-D') model of the offline replay."""
    x = np.zeros(np.shape(v) + (8,))
    x[..., 3], x[..., 7] = v, pedal
    u = np.stack([np.zeros_like(cmd), cmd], axis=-1)
    dvx, _, _, _, dpedal = body_derivatives(x, u, p, _NP)
    return np.asarray(dvx, float), np.asarray(dpedal, float)


def straight_step(s, v, pedal, cmd, p: VehicleParams = F1_2025, dt: float = _ENV.dt,
                  substeps: int = _ENV.substeps):
    """One control step of the 1-D model in float64: the command held for dt, explicit
    midpoint substeps as ``sim.rk2_step`` (ds/dt = vx on a straight)."""
    h = dt / substeps
    for _ in range(substeps):
        dv1, dp1 = _straight_rates(v, pedal, cmd, p)
        vm, pm = v + 0.5 * h * dv1, pedal + 0.5 * h * dp1
        dv2, dp2 = _straight_rates(vm, pm, cmd, p)
        s, v, pedal = s + h * vm, v + h * dv2, pedal + h * dp2
    return s, v, pedal


class Replay(NamedTuple):
    """A 1-D replay [B, K] (states 0..K-1; command and c3 fraction [B, K-1])."""

    s: np.ndarray
    v: np.ndarray
    pedal: np.ndarray
    cmd: np.ndarray
    k: np.ndarray            # c3's commanded braking fraction (NaN when not braking by it)


def replay_1d(rows: np.ndarray, s0, v0, car: CarTables, *, steps: int, ds: float = DS,
              pedal0=0.0, variant: str = "Q", s_corner=None, tau_c=None, s_stop=None,
              k_v: float = K_V, lead: float = LEAD_TIME, phi: float = PHI,
              p: VehicleParams = F1_2025) -> Replay:
    """Q's (or c3's) pedal law replayed on the straight-line model in float64 (ruling R22 C):
    the policy's own formulas (``law_a``, ``_c3_target``) evaluated with NumPy on the
    target rows [B, N], with the friction-ellipse clip of a straight (both axles' room 1),
    the env's [-1, 1] clip and its integrator (``straight_step``). No env, no TrackArrays,
    no harness code. Stops early once every run has s >= s_stop [B] (when given)."""
    T = np.atleast_2d(np.asarray(rows, float))
    B = len(T)
    s = np.broadcast_to(np.asarray(s0, float), (B,)).copy()
    v = np.broadcast_to(np.asarray(v0, float), (B,)).copy()
    ped = np.broadcast_to(np.asarray(pedal0, float), (B,)).copy()
    if variant == "c3":
        s_c = np.broadcast_to(np.asarray(s_corner, float), (B,))
        tau = np.broadcast_to(np.asarray(tau_c, float), (B,))
    S, V, P, C, Kf = [s.copy()], [v.copy()], [ped.copy()], [], []
    stop = None if s_stop is None else np.broadcast_to(np.asarray(s_stop, float), (B,))
    for _ in range(int(steps)):
        vv = np.maximum(v, 0.0)
        ahead = lead * vv
        v_t, a_t = _at(T, s + ahead, ds, np)
        p_hi = _pedal_for(vv, _uinterp(vv, car.v, car.a_acc, np), car, np)
        p_lo = _pedal_for(vv, -_uinterp(vv, car.v, car.a_dec, np), car, np)
        kf = np.full(B, np.nan)
        ff_only = np.zeros(B, bool)
        if variant == "c3":
            v_t, a_t, ff_only, kf = _c3_target(s, vv, ahead, T, s_c, tau, v_t, a_t, ds, car,
                                               np)
        c = law_a(vv, ped, v_t, a_t, p_lo, p_hi, car, k_v, lead, phi, np)
        c = np.where(ff_only, np.clip(_pedal_for(vv, a_t, car, np), p_lo, p_hi), c)
        c = np.clip(c, -1.0, 1.0)
        s, v, ped = straight_step(s, v, ped, c, p)
        S.append(s.copy())
        V.append(v.copy())
        P.append(ped.copy())
        C.append(c)
        Kf.append(kf)
        if stop is not None and np.all(s >= stop):
            break
    stack = lambda xs: np.stack(xs, axis=1)
    return Replay(stack(S), stack(V), stack(P), stack(C) if C else np.zeros((B, 0)),
                  stack(Kf) if Kf else np.zeros((B, 0)))


def brake_demand_1d(v, pedal, p: VehicleParams = F1_2025) -> np.ndarray:
    """Protocol 5's front brake demand on straight-line states, written from the protocol's
    definition and the model's own tyre law, independently of ``measures`` (ruling R22 C):
    beta_f = brake f_brake_max brake_bias / (mu_x,f(fz_f / 2) fz_f), brake = relu(-pedal)
    tanh(vx / 0.5) (the model's brake input), fz_f the front load of ``dynamics.tyre_forces``
    (load transfer included), mu its load-sensitive friction."""
    v, pedal = np.broadcast_arrays(np.asarray(v, float), np.asarray(pedal, float))
    x = np.zeros(v.shape + (8,))
    x[..., 3], x[..., 7] = v, pedal
    fz_f = np.asarray(tyre_forces(x, p, _NP).fz_f, float)
    brake = _NP.relu(-pedal) * np.tanh(v / 0.5)
    return brake * p.f_brake_max * p.brake_bias / (_mu(p.mux_f, 0.5 * fz_f, p, _NP) * fz_f)


def onset_1d(s: np.ndarray, v: np.ndarray, beta: np.ndarray, s_apex) -> tuple[np.ndarray, ...]:
    """Protocol 5's brake-demand onset, implemented here independently of
    ``measures.brake_onset`` (ruling R22 C): per row of [B, K] states, the start of the last
    run of at least BRAKE_MIN_STEPS consecutive states with beta > BRAKE_ON that begins
    before s_apex, interpolated at the BRAKE_ON crossing (a run starting at state 0 starts
    there). Returns (s_b, v_b, k_b) [B]; NaN, NaN, -1 without such a run."""
    s, v, beta = (np.atleast_2d(np.asarray(a, float)) for a in (s, v, beta))
    apex = np.broadcast_to(np.asarray(s_apex, float), (len(s),))
    s_b = np.full(len(s), np.nan)
    v_b = np.full(len(s), np.nan)
    k_b = np.full(len(s), -1, int)
    for r in range(len(s)):
        above = beta[r] > BRAKE_ON
        k, n = 0, len(above)
        while k < n:
            if not above[k]:
                k += 1
                continue
            end = k
            while end < n and above[end]:
                end += 1
            if end - k >= BRAKE_MIN_STEPS:
                if k == 0:
                    sa, va = s[r, 0], v[r, 0]
                else:
                    w = (BRAKE_ON - beta[r, k - 1]) / (beta[r, k] - beta[r, k - 1])
                    sa = s[r, k - 1] + w * (s[r, k] - s[r, k - 1])
                    va = v[r, k - 1] + w * (v[r, k] - v[r, k - 1])
                if sa < apex[r]:
                    s_b[r], v_b[r], k_b[r] = sa, va, k
            k = end
    return s_b, v_b, k_b


def _offline_batch(items: list[dict], car: CarTables, ds: float = DS,
                   steps: int = 800) -> list[dict]:
    """Q's offline speeds and onsets for many starts at once: ``forward_profiles`` and the
    1-D replay of Q's law (``replay_1d``, ``brake_demand_1d``, ``onset_1d``). ``items``
    hold row [N], kappa [N], landmarks, s0, v0 and optionally pedal0 and steps."""
    if not items:
        return []
    rows = np.stack([np.asarray(it["row"], float) for it in items])
    kap = np.stack([np.asarray(it["kappa"], float) for it in items])
    s0 = np.array([float(it["s0"]) for it in items])
    v0 = np.array([float(it["v0"]) for it in items])
    pedal0 = np.array([float(it.get("pedal0", 0.0)) for it in items])
    apex = np.array([float(it["landmarks"].s_apex) for it in items])
    n = max(int(it.get("steps", steps)) for it in items)
    fwd = forward_profiles(rows, kap, s0, v0, car, ds)
    # Stop once every run is BRAKE_MIN_STEPS + 1 steps past its apex (at <= 100 m/s).
    stop = apex + (BRAKE_MIN_STEPS + 1) * 100.0 * _ENV.dt
    rp = replay_1d(rows, s0, v0, car, steps=n, ds=ds, pedal0=pedal0, s_stop=stop)
    beta = brake_demand_1d(rp.v, rp.pedal)
    s_b, v_b, _ = onset_1d(rp.s, rp.v, beta, apex)
    return [dict(v_forward=fwd[i], s_b=float(s_b[i]), v_b=float(v_b[i]))
            for i in range(len(items))]


def offline_prediction(v_target: np.ndarray, kappa: np.ndarray, landmarks: Landmarks,
                       s0: float, v0: float, key_metrics: Any,
                       tables: ModelTables | CarTables, ds: float = DS, *,
                       pedal0: float = 0.0, steps: int = 800,
                       precomputed: dict | None = None) -> dict[str, float]:
    """Q's predicted e_ve, e_vmin and dd_b from its own profile against the key (protocol 7:
    the harness values must match within 2 points or one control step). v_e and v_min come
    from ``forward_profile`` (R22 B); the onset from replaying Q's law on the straight-line
    model with protocol 5's rule (R22 C). ``key_metrics`` has attributes (or keys) v_e,
    v_min and d_b. ``precomputed`` is this start's entry of ``_offline_batch``."""
    if precomputed is None:
        car = tables if isinstance(tables, CarTables) else car_tables(tables)
        precomputed = _offline_batch([dict(row=v_target, kappa=kappa, landmarks=landmarks,
                                           s0=s0, v0=v0, pedal0=pedal0, steps=steps)],
                                     car, ds)[0]
    v = precomputed["v_forward"]
    s = np.arange(len(v)) * ds
    ok = np.isfinite(v)
    sv, vv = s[ok], v[ok]
    v_e = float(np.interp(landmarks.s_c0, sv, vv))
    win = (sv >= landmarks.s_c0 - 50.0) & (sv <= landmarks.s_c1 + 50.0)
    v_min = float(vv[win].min()) if win.any() else np.nan
    s_b, v_b = float(precomputed["s_b"]), float(precomputed["v_b"])
    d_b = landmarks.s_c0 - s_b if np.isfinite(s_b) else np.nan

    def get(name):
        return float(key_metrics[name] if isinstance(key_metrics, dict)
                     else getattr(key_metrics, name))

    return {"v_e": v_e, "v_min": v_min, "d_b": d_b, "v_b": v_b,
            "e_ve": v_e / get("v_e") - 1.0, "e_vmin": v_min / get("v_min") - 1.0,
            "dd_b": d_b - get("d_b")}


def offline_match(pred: dict[str, float], measured: dict[str, float], dt: float = DT
                  ) -> dict[str, bool]:
    """Protocol 7's Q verdict 'offline_match': harness e_ve and e_vmin within 2 percentage
    points of the prediction, dd_b within one control step, v_b dt with the *predicted* v_b
    (ruling R22; the measured v_b where no onset is predicted). A measure undefined on both
    sides (no braking onset in a flat cell) agrees; undefined on one side does not."""
    vb = float(pred.get("v_b", np.nan))
    vb = float(measured.get("v_b", np.nan)) if not np.isfinite(vb) else vb
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
    Passed when every compared run matches on all three (protocol 7). The predictions are
    computed in one batch (``_offline_batch``). The result also lists the residuals of every
    compared run ('residuals': cell, phase, measure differences in points and metres)."""
    by = {c.cell_id: c for c in cells}
    kidx = {int(c): i for i, c in enumerate(np.asarray(keys["cell"]))}
    rows: dict[int, np.ndarray] = {}
    todo = []
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
        # The run's own start (its phase's state: R21 moves a real-lap start in time, so
        # its speed and pedal differ by phase; a designed start differs in s only).
        x0 = np.asarray(c.initial_state(int(runs["phase"][i])), float)
        todo.append((i, c, km, float(x0[7]), float(x0[3])))
    pre = (_offline_batch([dict(row=rows[c.stim_id], kappa=np.asarray(bank.kappa[c.stim_id]),
                                landmarks=bank.stimuli[c.stim_id].landmarks,
                                s0=float(runs["s0"][i]), v0=v0, pedal0=pedal0,
                                steps=int(c.W or 800))
                           for i, c, km, pedal0, v0 in todo], car, bank.ds)
           if isinstance(car, CarTables) else [None] * len(todo))
    n, fails, resid = 0, [], []
    for (i, c, km, pedal0, v0), pc in zip(todo, pre, strict=True):
        st = bank.stimuli[c.stim_id]
        kw = {} if pc is None else {"precomputed": pc}
        pred = offline_prediction(rows[c.stim_id], np.asarray(bank.kappa[c.stim_id]),
                                  st.landmarks, float(runs["s0"][i]), v0, km, car,
                                  **kw)
        meas = {"e_ve": float(runs["v_e"][i]) / km["v_e"] - 1.0,
                "e_vmin": float(runs["v_min"][i]) / km["v_min"] - 1.0,
                "dd_b": float(runs["d_b"][i]) - km["d_b"], "v_b": float(runs["v_b"][i])}
        ok = offline_match(pred, meas)
        n += 1
        resid.append(dict(cell=int(c.cell_id), phase=int(runs["phase"][i]),
                          **{m: meas[m] - pred[m] for m in ("e_ve", "e_vmin", "dd_b")},
                          step=abs(pred["v_b"] if np.isfinite(pred["v_b"]) else meas["v_b"])
                          * DT))
        if not all(ok.values()):
            fails.append(dict(cell=int(c.cell_id), phase=int(runs["phase"][i]),
                              bad=[m for m, v in ok.items() if not v],
                              pred={m: pred[m] for m in ("e_ve", "e_vmin", "dd_b")},
                              measured={m: meas[m] for m in ("e_ve", "e_vmin", "dd_b")}))
    return dict(passed=bool(n and not fails), n=n, n_fail=len(fails), fails=fails[:20],
                residuals=resid)


# ------------------------------------------------------------------------- c1 (R23)


def _key_rows(keys) -> dict[int, int]:
    return {int(c): i for i, c in enumerate(np.asarray(keys["cell"]))}


def c1_shifts(bank: StimulusBank, cells: list[Cell], keys, tables: ModelTables | CarTables,
              c_v: float) -> dict[int, dict]:
    """Ruling R23: c1's braking shift per cell, 15 m plus Q's offline-predicted onset lead
    over the u = 0 nominal key, so c1 is 15 m late against the key. The lead is Q's offline
    d_b (``offline_prediction``'s replay from the cell's phase-0 start) minus the key table's
    d_b* (``keys``: columns cell, key_status, d_b_star). Agent-free: keys and Q's offline
    profile only; to be written to the manifest before c1 is rolled out.

    Returns {cell_id: {shift, lead, d_b_q, d_b_key, status}}. status 'ok'; otherwise the
    shift is C1_SHIFT relative to Q (as built) and status says why no lead exists:
    'no_verified_key', 'key_flat' (no key onset), 'q_flat' (no predicted Q onset), or
    'lead_below_-15m' (Q predicted more than 15 m later than the key: shift 0)."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    kidx = _key_rows(keys)
    stims = sorted({c.stim_id for c in cells})
    T = dict(zip(stims, target_profiles(np.asarray(bank.kappa)[stims], car, c_v, bank.ds),
                 strict=True))
    items = []
    for c in cells:
        x0 = np.asarray(c.initial_state(0), float)
        items.append(dict(row=T[c.stim_id], kappa=np.asarray(bank.kappa[c.stim_id]),
                          landmarks=bank.stimuli[c.stim_id].landmarks, s0=x0[0], v0=x0[3],
                          pedal0=x0[7], steps=int(c.W or 800)))
    pre = _offline_batch(items, car, bank.ds)
    out: dict[int, dict] = {}
    for c, it, pc in zip(cells, items, pre, strict=True):
        s_c0 = it["landmarks"].s_c0
        d_q = s_c0 - pc["s_b"] if np.isfinite(pc["s_b"]) else np.nan
        k = kidx.get(int(c.cell_id))
        verified = k is not None and str(keys["key_status"][k]) == "verified"
        d_key = float(keys["d_b_star"][k]) if verified else np.nan
        lead = d_q - d_key
        if not verified:
            status = "no_verified_key"
        elif not np.isfinite(d_key):
            status = "key_flat"
        elif not np.isfinite(d_q):
            status = "q_flat"
        elif C1_SHIFT + lead < 0.0:
            status = "lead_below_-15m"
        else:
            status = "ok"
        shift = (C1_SHIFT + lead if status == "ok"
                 else 0.0 if status == "lead_below_-15m" else C1_SHIFT)
        out[int(c.cell_id)] = dict(shift=float(shift), lead=float(lead), d_b_q=float(d_q),
                                   d_b_key=float(d_key), status=status)
    return out


def _shift_of(v) -> float:
    return float(v["shift"] if isinstance(v, dict) else v)


def q_params_for_kappa(kappa_rows: np.ndarray, s_corner: np.ndarray,
                       tables: ModelTables | CarTables, c_v: float, variant: str = "Q",
                       tau_c: float = 0.0, ds: float = DS,
                       brake_shift: float | np.ndarray | None = None) -> QParams:
    """QParams from strips' knot samples [n_rows, N] (one target row per strip). Used by
    ``q_params`` and by the probes, which build shifted strips of their own. c1 needs its
    per-row ``brake_shift`` (ruling R23, ``c1_shifts``); other variants take none."""
    if variant not in Q_VARIANTS:
        raise ValueError(f"unknown Q variant {variant!r}")
    if variant == "c1" and brake_shift is None:
        raise ValueError("c1 needs its per-cell braking shift (ruling R23: c1_shifts)")
    if variant != "c1" and brake_shift is not None:
        raise ValueError(f"{variant} takes no braking shift")
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    cap = C2_KAPPA_CAP if variant == "c2" else np.inf
    shift = 0.0 if brake_shift is None else brake_shift
    rows = target_profiles(np.atleast_2d(kappa_rows), car, c_v, ds, cap, shift)
    f32 = jnp.float32
    return QParams(
        v_target=jnp.asarray(rows, f32), s_corner=jnp.asarray(s_corner, f32),
        car=_as_jax(car), c_v=f32(c_v), k_v=f32(K_V), lookahead_min=f32(LOOKAHEAD_MIN),
        lookahead_time=f32(LOOKAHEAD_TIME), lead_time=f32(LEAD_TIME), kappa_cap=f32(cap),
        tau_c=f32(tau_c if variant == "c3" else 0.0), phi=f32(PHI))


def q_params(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
             c_v: float, variant: str = "Q", tau_c: float = 0.0,
             shifts: dict[int, Any] | None = None) -> tuple[QParams, np.ndarray]:
    """QParams for a run list plus the aux row of each cell: one profile per stimulus (c1:
    per stimulus and shift, ``shifts`` being ``c1_shifts``'s {cell_id: entry or metres});
    s_corner is each row's s_c0."""
    if variant == "c1":
        if shifts is None:
            raise ValueError("c1 needs its per-cell braking shift (ruling R23: c1_shifts)")
        keys = [(c.stim_id, _shift_of(shifts[c.cell_id])) for c in cells]
    else:
        keys = [(c.stim_id, 0.0) for c in cells]
    uniq = sorted(set(keys))
    row_of = {k: i for i, k in enumerate(uniq)}
    stims = [k[0] for k in uniq]
    kappa = np.asarray(bank.kappa)[stims]
    s_corner = np.array([bank.stimuli[s].landmarks.s_c0 for s in stims])
    sh = np.array([k[1] for k in uniq]) if variant == "c1" else None
    prm = q_params_for_kappa(kappa, s_corner, tables, c_v, variant, tau_c, bank.ds, sh)
    return prm, np.array([row_of[k] for k in keys], np.int32)


# ------------------------------------------------------------------------- c2 (R24)


def c2_planted_law(c2_runs, q_runs, bank: StimulusBank, cells: list[Cell],
                   tables: ModelTables | CarTables, c_v: float, r_lock: float,
                   tol: float = 0.02) -> dict:
    """Ruling R24's planted-law check of c2 against Q through the harness: per deterministic
    (cell, phase) both ran, the measured c2/Q v_min ratio (run tables: cell, phase, mode,
    v_min, crashed, s0) against the offline prediction, the ratio of c2's and Q's
    ``forward_profile`` minima over [s_c0 - 50, s_c1 + 50] from the run's start. On cells
    with max |kappa| > 1/12 the two agree within ``tol`` (2 points); elsewhere the measured
    ratio is 1 +- ``tol``. Cells whose tightest raw radius is at or below ``r_lock``
    (``lock_radius``: the steering lock, not the grip, sets the cornering limit there, so a
    centreline driver cannot hold its target) are listed, not judged. A crash of either
    control on a judged cell fails it. Returns passed, counts, failures and every row."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    by = {c.cell_id: c for c in cells}

    def index(rt):
        det = np.asarray(rt["mode"]).astype(str) == "det"
        return {(int(rt["cell"][i]), int(rt["phase"][i])): i for i in np.flatnonzero(det)}

    i2, iq = index(c2_runs), index(q_runs)
    pairs = sorted(set(i2) & set(iq))
    stims = sorted({by[c].stim_id for c, _ in pairs if c in by})
    kap = np.asarray(bank.kappa)[stims]
    Tq = dict(zip(stims, target_profiles(kap, car, c_v, bank.ds), strict=True))
    Tc = dict(zip(stims, target_profiles(kap, car, c_v, bank.ds, C2_KAPPA_CAP), strict=True))
    todo = [(c, ph) for c, ph in pairs if c in by]
    s0 = np.array([float(q_runs["s0"][iq[k]]) for k in todo])
    v0 = np.array([float(by[c].initial_state(ph)[3]) for c, ph in todo])   # the phase's start
    rq = np.stack([Tq[by[c].stim_id] for c, _ in todo]) if todo else np.zeros((0, 1))
    rc = np.stack([Tc[by[c].stim_id] for c, _ in todo]) if todo else np.zeros((0, 1))
    kk = np.stack([np.asarray(bank.kappa[by[c].stim_id]) for c, _ in todo]) if todo else rq
    fq = forward_profiles(rq, kk, s0, v0, car, bank.ds) if todo else rq
    fc = forward_profiles(rc, kk, s0, v0, car, bank.ds) if todo else rc
    s = np.arange(rq.shape[1]) * bank.ds
    rows, fails, listed = [], [], []
    for j, (cid, ph) in enumerate(todo):
        c = by[cid]
        lm = bank.stimuli[c.stim_id].landmarks
        win = (s >= lm.s_c0 - 50.0) & (s <= lm.s_c1 + 50.0) & np.isfinite(fq[j])
        pred = float(np.min(fc[j][win]) / np.min(fq[j][win])) if win.any() else np.nan
        k_max = float(np.max(np.abs(kk[j])))
        capped = k_max > C2_KAPPA_CAP
        lock = k_max > 0 and 1.0 / k_max <= r_lock
        a, b = i2[(cid, ph)], iq[(cid, ph)]
        crashed = bool(c2_runs["crashed"][a]) or bool(q_runs["crashed"][b])
        meas = float(c2_runs["v_min"][a]) / float(q_runs["v_min"][b])
        want = pred if capped else 1.0
        row = dict(cell=cid, phase=ph, R_raw=1.0 / k_max if k_max > 0 else np.inf,
                   capped=capped, lock_limited=bool(lock), crashed=crashed, measured=meas,
                   predicted=pred, expected=want, diff=meas - want)
        rows.append(row)
        if lock:
            listed.append(row)
            continue
        if crashed or not (np.isfinite(meas) and np.isfinite(want)) or abs(meas - want) > tol:
            fails.append(row)
    judged = [r for r in rows if not r["lock_limited"]]
    return dict(passed=bool(judged and not fails), n=len(judged),
                n_capped=sum(r["capped"] for r in judged),
                n_uncapped=sum(not r["capped"] for r in judged), n_fail=len(fails),
                fails=fails[:20], lock_limited=listed, rows=rows, r_lock=float(r_lock),
                tol=tol)


# ------------------------------------------------------------------------- c3 (R26)


def c3_fraction(s: np.ndarray, v: np.ndarray, row: np.ndarray, s_c: float, tau_c: float,
                tables: ModelTables | CarTables, ds: float = DS,
                lead: float = LEAD_TIME) -> np.ndarray:
    """c3's commanded braking fraction k at each state of a run (s, v [K]; Q's target row
    [N]; NaN where c3 is not braking by its own law): ``_c3_target`` in NumPy. For the
    pre-P3 check of ruling R26 (k <= 1) on harness traces."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    s, v = np.asarray(s, float), np.maximum(np.asarray(v, float), 0.0)
    row = np.asarray(row, float)
    ahead = lead * v
    v_t, a_t = _at(row, s + ahead, ds, np)
    return _c3_target(s, v, ahead, row, float(s_c), float(tau_c), v_t, a_t, ds, car, np)[3]


def _c3_kmax(rows, s0, v0, pedal0, s_c, steps, tau, car, ds) -> np.ndarray:
    """Largest commanded k of c3's 1-D replay up to the corner start, per run (NaN if c3
    never brakes by its own law there)."""
    rp = replay_1d(rows, s0, v0, car, steps=steps, ds=ds, pedal0=pedal0, variant="c3",
                   s_corner=s_c, tau_c=tau, s_stop=s_c)
    k = rp.k
    out = np.full(len(k), np.nan)
    has = np.isfinite(k).any(axis=1) if k.size else np.zeros(len(k), bool)
    out[has] = np.nanmax(k[has], axis=1)
    return out


def _c3_trigger_margin(rows, s0, v0, pedal0, s_c, steps, tau, car, ds) -> np.ndarray:
    """Per run of c3's 1-D replay at ``tau`` [B]: tau v - d_c (m) at the first step c3 brakes
    by its own law, how far inside the trigger condition d_c <= tau v that step is (NaN if
    it never does). Near 0 the float32 harness may trigger one step later."""
    rp = replay_1d(rows, s0, v0, car, steps=steps, ds=ds, pedal0=pedal0, variant="c3",
                   s_corner=s_c, tau_c=tau, s_stop=s_c)
    s_c = np.broadcast_to(np.asarray(s_c, float), (len(rp.k),))
    tau = np.broadcast_to(np.asarray(tau, float), (len(rp.k),))
    out = np.full(len(rp.k), np.nan)
    for i in range(len(rp.k)):
        hit = np.flatnonzero(np.isfinite(rp.k[i]))
        if hit.size:
            j = int(hit[0])
            out[i] = tau[i] * max(float(rp.v[i, j]), 0.0) - (s_c[i] - float(rp.s[i, j]))
    return out


def tau_c_k1(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
             c_v: float, bracket: tuple[float, float] = TAU_C_BRACKET,
             iters: int = TAU_C_ITERS, guard: float = TAU_C_GUARD) -> tuple[float, dict]:
    """Ruling R26: c3's tau_c, the smallest value that keeps its braking fraction k <= 1 on
    every given cell (the caller passes the tier A/B braking cells), from Q's offline
    profile: Q's target row gives the lawful entry speed v_e = T(s_c0), and c3's own law is
    replayed on the straight-line model (``replay_1d``, float64; flat out from each phase's
    start, the time-to-corner trigger at the control step where d_c <= tau v, the
    feed-forward brake k = (D(v) - D(v_e)) / d_c recomputed every step, the pedal lag
    included) up to the corner start. Agent-free and harness-free; no c3 rollout is read.

    Per run the needed tau is bisected on ``bracket`` (k_max <= 1, or c3 never brakes by its
    own law); the threshold is the largest need, checked on every run at that one value (and
    raised by bisection if a run is not monotone in tau). tau_c is the threshold plus
    ``guard`` (TAU_C_GUARD, review of R26): "smallest" holds at control-step resolution, and
    at the bare threshold the binding run triggers within float32 resolution of s, so the
    float32 harness could trigger it one step later (k far above 1). Runs whose k exceeds 1
    even at the top of the bracket (a start inside the full-capacity braking distance) are
    'infeasible': listed, not used. Returns (tau_c, details) with per-run rows (cell, phase,
    tau_needed, status 'ok' | 'no_trigger' | 'infeasible', k_max at tau_c), the threshold,
    the guard, and the smallest trigger margin tau_c v - d_c (m) of the binding runs."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    stims = sorted({c.stim_id for c in cells})
    T = dict(zip(stims, target_profiles(np.asarray(bank.kappa)[stims], car, c_v, bank.ds),
                 strict=True))
    ids, rows, s0, v0, p0, s_c = [], [], [], [], [], []
    steps = 0
    for c in cells:
        for k in range(c.n_phases):
            x0 = np.asarray(c.initial_state(k), float)
            ids.append((int(c.cell_id), k))
            rows.append(T[c.stim_id])
            s0.append(x0[0])
            v0.append(x0[3])
            p0.append(x0[7])
            s_c.append(bank.stimuli[c.stim_id].landmarks.s_c0)
            steps = max(steps, int(c.W or 800))
    if not ids:
        raise ValueError("no cells given")
    rows, s0, v0, p0, s_c = (np.asarray(a, float) for a in (rows, s0, v0, p0, s_c))
    B = len(ids)

    def kmax(tau):
        return _c3_kmax(rows, s0, v0, p0, s_c, steps, np.broadcast_to(tau, (B,)), car,
                        bank.ds)

    def feasible(km):
        return np.isnan(km) | (km <= 1.0)

    k_hi, k_lo = kmax(bracket[1]), kmax(bracket[0])
    status = np.where(np.isnan(k_hi), "no_trigger",
                      np.where(feasible(k_hi), "ok", "infeasible")).astype(object)
    lo, hi = np.full(B, bracket[0]), np.full(B, bracket[1])
    need = np.where(feasible(k_lo), bracket[0], np.nan)
    open_ = (status == "ok") & ~feasible(k_lo)
    for _ in range(iters if open_.any() else 0):
        mid = 0.5 * (lo + hi)
        f = feasible(kmax(np.where(open_, mid, hi)))
        hi = np.where(open_ & f, mid, hi)
        lo = np.where(open_ & ~f, mid, lo)
    need = np.where(open_, hi, need)
    ok = status == "ok"

    def fits(t: float) -> bool:
        return bool(np.all(feasible(kmax(t)[ok])))

    def raised(t: float) -> float:        # a run not monotone in tau: raise tau to fit all
        lo_g, hi_g = t, float(bracket[1])
        for _ in range(iters):
            mid = 0.5 * (lo_g + hi_g)
            if fits(mid):
                hi_g = mid
            else:
                lo_g = mid
        return hi_g

    thr = float(np.max(need[ok])) if ok.any() else float(bracket[0])
    if ok.any() and not fits(thr):
        thr = raised(thr)
    tau = min(thr + float(guard), float(bracket[1]))
    if ok.any() and not fits(tau):
        tau = min(raised(tau) + float(guard), float(bracket[1]))
    k_at = kmax(tau)
    verified = bool(np.all(feasible(k_at[ok])))
    bind = ok & np.isfinite(need) & (need >= np.max(need[ok]) - 1e-9) if ok.any() else ok
    margin = np.full(B, np.nan)
    if bind.any():
        margin[bind] = _c3_trigger_margin(rows[bind], s0[bind], v0[bind], p0[bind], s_c[bind],
                                          steps, np.full(int(bind.sum()), tau), car, bank.ds)
    table = [dict(cell=cid, phase=ph, tau_needed=float(need[i]), status=str(status[i]),
                  k_max=float(k_at[i])) for i, (cid, ph) in enumerate(ids)]
    return tau, dict(rows=table, n=B, n_ok=int(ok.sum()),
                     n_infeasible=int((status == "infeasible").sum()),
                     n_no_trigger=int((status == "no_trigger").sum()), verified=verified,
                     bracket=tuple(bracket), c_v=float(c_v), threshold=thr,
                     guard=float(guard),
                     trigger_margin_min_m=float(np.nanmin(margin)) if np.isfinite(
                         margin).any() else float("nan"),
                     rule="R26: smallest tau_c with k <= 1 on every tier A/B braking cell, "
                          "from Q's offline profile (1-D replay of c3's law), plus a "
                          f"{guard:g} s guard above float32 resolution (review of R26)")


C3_TIERS = ("A", "B")


def c3_checks(det: dict, runs, keys, bank: StimulusBank, cells: list[Cell],
              tables: ModelTables | CarTables, c_v: float, tau_c: float,
              tiers: tuple[str, ...] = C3_TIERS) -> dict:
    """Ruling R26's checks of c3 before its P3 is read, through the harness.

    ``det``: c3's deterministic rollout arrays (cell, phase, x [R, T+1, 8], W, term_step;
    ``valid`` if recorded); ``runs``: their run table (cell, phase, mode, crashed,
    crash_type, forced_onset, v_e); ``keys``: the key table (cell, tiers, braking,
    key_status). The judged runs are c3's deterministic runs on tier A/B cells whose u = 0
    key is verified and braking (2.3) and whose own onset is not forced (``forced_onset``).

    - k_le_1: c3's commanded braking fraction (``c3_fraction`` on the harness states before
      s_c0, with c3's own target rows from ``q_params``) is at most 1 on every judged run;
    - entry: |v_e - T(s_c0)| <= a_dec(v_e) DT on every judged run, T Q's target row (c3
      reaches Q's lawful entry speed within one control step of braking); a judged run
      without an entry speed (a crash before s_c0) fails;
    - overspeed_crashes: no deterministic run of c3 on a tier A/B cell crashed and typed
      overspeed (any phase, braking or not).

    Returns the three verdicts ('pass' / 'fail', or 'no judged run', which is not a pass;
    'none' / '<n> crashes') with counts, the largest k and entry error (in steps) and the
    failing (cell, phase) runs."""
    car = tables if isinstance(tables, CarTables) else car_tables(tables)
    by = {c.cell_id: c for c in cells}
    kidx = {int(c): i for i, c in enumerate(np.asarray(keys["cell"]))}

    def key_ok(cid: int) -> tuple[bool, bool]:
        i = kidx.get(cid)
        if i is None:
            return False, False
        tier = bool(set(str(keys["tiers"][i]).split(",")) & set(tiers))
        brk = bool(keys["braking"][i]) and str(keys["key_status"][i]) == "verified"
        return tier, brk

    mode = np.asarray(runs["mode"]).astype(str)
    rcell, rph = np.asarray(runs["cell"], int), np.asarray(runs["phase"], int)
    crashed = np.asarray(runs["crashed"], bool)
    ctype = np.asarray(runs["crash_type"]).astype(str)
    forced = np.asarray(runs["forced_onset"], bool)
    v_e = np.asarray(runs["v_e"], float)
    row_of = {(int(rcell[i]), int(rph[i])): i for i in np.flatnonzero(mode == "det")}
    in_tier = np.array([key_ok(int(c))[0] for c in rcell], bool)
    over = (mode == "det") & in_tier & crashed & (ctype == "overspeed")
    judged = [(int(det["cell"][j]), int(det["phase"][j]), j) for j in range(len(det["cell"]))
              if (int(det["cell"][j]), int(det["phase"][j])) in row_of
              and int(det["cell"][j]) in by and all(key_ok(int(det["cell"][j])))
              and not forced[row_of[(int(det["cell"][j]), int(det["phase"][j]))]]]
    jc = [by[c] for c in sorted({c for c, _, _ in judged})]
    prm, aux = q_params(bank, jc, car, c_v, "c3", tau_c) if jc else (None, None)
    row_idx = {c.cell_id: int(a) for c, a in zip(jc, aux if aux is not None else [], strict=True)}
    rows = np.asarray(prm.v_target, float) if prm is not None else np.zeros((0, 1))
    s_grid = np.arange(rows.shape[1]) * bank.ds
    k_bad, e_bad, k_max, e_max, n_trig = [], [], 0.0, 0.0, 0
    for cid, ph, j in judged:
        lm = bank.stimuli[by[cid].stim_id].landmarks
        row = rows[row_idx[cid]]
        W = int(det["W"][j]) if "W" in det else len(det["x"][j]) - 1
        x = np.asarray(det["x"][j][:W + 1], float)
        n = len(x)
        if "valid" in det:
            v = np.asarray(det["valid"][j][:n], bool)
            n = n if v.all() else int(np.argmin(v))
        term = int(det["term_step"][j]) if "term_step" in det else -1
        if term >= 0:
            n = min(n, term + 2)
        x = x[:n]
        pre = x[:, 0] < lm.s_c0
        k = c3_fraction(x[pre, 0], x[pre, 3], row, lm.s_c0, tau_c, car, bank.ds)
        if np.isfinite(k).any():
            n_trig += 1
            km = float(np.nanmax(k))
            k_max = max(k_max, km)
            if not km <= 1.0:
                k_bad.append(dict(cell=cid, phase=ph, k_max=km))
        ve = v_e[row_of[(cid, ph)]]
        T_e = float(np.interp(lm.s_c0, s_grid, row))
        step = float(np.interp(ve, car.v, car.a_dec)) * DT if np.isfinite(ve) else np.nan
        err = abs(ve - T_e) / step if np.isfinite(ve) and step > 0 else np.inf
        e_max = max(e_max, err)
        if not err <= 1.0:
            e_bad.append(dict(cell=cid, phase=ph, v_e=float(ve), target=T_e,
                              steps=float(err)))
    n = len(judged)
    n_over = int(over.sum())
    none = "no judged run"                    # vacuous: never a pass
    return dict(k_le_1=none if not n else "pass" if not k_bad else "fail",
                entry=none if not n else "pass" if not e_bad else "fail",
                overspeed_crashes="none" if not n_over else f"{n_over} crashes",
                n_judged=n, n_triggered=n_trig, n_forced_excluded=int(sum(
                    1 for (c, _), i in row_of.items() if c in by and all(key_ok(c))
                    and forced[i])),
                k_max=k_max, entry_max_steps=e_max, k_fails=k_bad[:20],
                entry_fails=e_bad[:20], n_k_fail=len(k_bad), n_entry_fail=len(e_bad),
                overspeed_cells=sorted({int(c) for c in rcell[over]})[:20], tau_c=float(tau_c))


# ------------------------------------------------------------------------- T


class TapeParams(NamedTuple):
    acts: jax.Array          # [n_rows, W_max, 2] the nominal key's u = 0 commands


@functools.cache
def make_tape_policy() -> Policy:
    """T: returns acts[aux, state.t] (params TapeParams); the last command is held if the
    run outlives its tape. One Policy object per process (cached): T32 and T64 share it,
    and ``rollout.run_tape_rollouts_x64`` accepts no other policy."""

    def policy(params: TapeParams, o, state: racing.EnvState, tr: TrackArrays, aux):
        t = jnp.minimum(state.t, params.acts.shape[1] - 1)
        mean = params.acts[aux, t].astype(o.dtype)
        return mean, jnp.full((2,), -jnp.inf, o.dtype)

    return policy


def tape_params(tapes: list[np.ndarray], dtype=np.float32) -> TapeParams:
    """Stack key command sequences [W_i, 2] into TapeParams, padding each with its last
    command; row i is aux = i. float32 for T32 (the agents' path); float64 (the keys' own
    commands, uncast) only for T64, whose process runs with x64 on (ruling R20)."""
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"tape dtype {dtype} (float32 or float64)")
    if dtype == np.float64 and not jax.config.jax_enable_x64:
        raise RuntimeError("a float64 tape needs x64 on (T64 runs in its own process)")
    w = max(len(a) for a in tapes)
    out = np.zeros((len(tapes), w, 2), dtype)
    for i, a in enumerate(tapes):
        a = np.asarray(a, dtype)
        out[i, :len(a)] = a
        out[i, len(a):] = a[-1]
    return TapeParams(jnp.asarray(out, dtype))


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
    """A control and the verdicts it must get (protocol 7, required verdicts block the tag;
    ``expected`` verdicts do not block: a miss is reported as a power finding)."""

    name: str
    tiers: tuple[str, ...]
    required: dict[str, tuple[str, ...]] = field(default_factory=dict)  # test -> accepted verdicts
    expected: dict[str, tuple[str, ...]] = field(default_factory=dict)  # non-blocking
    reported: tuple[str, ...] = ()                                      # logged, never judged


SAME_AS_Q = "same_as:Q"      # c2's P1a must equal Q's verdict (label and class), R24

CONTROLS = {
    # R20: T is the float32 path (T32): its measure check on runs reaching s_c1 + 250 m
    # blocks; terminations and regret (divided by |J*|) are reported with the keys' margins.
    # R30's regression count (the adjudication's acceptance check, not a ruling): its crashes,
    # typed by the ellipse fallback (no viability solves), by type, expected to include no
    # overspeed now that dv must exceed CRASH_DV_TAU (float32 noise at the key's own speed
    # read as a sign: 6 of 11). Reported: R20 reports T32's crashes, and R30 adds no verdict.
    "T": ControlSpec("T", ("A", "G"), {"errors": ("within_tau",)},
                     reported=("termination", "regret", "crash_types")),
    "Q": ControlSpec("Q", ("A", "B", "D"), {"offline_match": ("pass",),
                                            "P1a": ("confirm", "cautious but lawful"),
                                            "P3": ("confirm",), "P4": ("class 4", "class 5"),
                                            "P5": ("symmetric",)}),
    "Qp": ControlSpec("Qp", ("A", "O"), {"runs": ("clean",)}),
    # R28 as R30 replaces its check: c1's crashed u = 0 runs (cells with a verified u = 0
    # key) get viability and counterfactual solves before the tag; every one is resolved by
    # the viability path, none over-brake or line/steer, and every one unrecoverable at or
    # before s_c0 overspeed (``analysis.viability_typing``).
    "c1": ControlSpec("c1", ("A", "B"), {"P2": ("confirm",), "P2b": ("onset_share>0.5",),
                                         "viability_typing": ("pass",)}),
    # R24: P1a equals Q's verdict; the planted-law check replaces P1b's requirement. R30's
    # regression count (the adjudication's acceptance check): its crashes on the lock-limited
    # cells, expected to stay overspeed (``c2_lock_typing``). Reported: R29(1) lists these
    # cells, it does not judge them.
    "c2": ControlSpec("c2", ("A",), {"P1a": (SAME_AS_Q,), "planted_law": ("pass",)},
                      reported=("P1b", "lock_crashes")),
    # R26: before P3 is read, k <= 1, entry within one step of Q's target and no overspeed
    # crashes on tiers A/B; P3 refuting naming M_idx is expected, a miss a power finding.
    "c3": ControlSpec("c3", ("A", "B"), {"k_le_1": ("pass",), "entry": ("pass",),
                                         "overspeed_crashes": ("none",)},
                      expected={"P3": ("refute:M_idx",)}),
    "K": ControlSpec("K", ("D",), {"P4": ("class 2", "class 3")}),
}


@dataclass(frozen=True)
class PlantSite:
    """Where a planted crash control runs (ruling R30), by geometry: the tier A synthetic cell
    with this approach, direction, angle, width and ramp kind whose R is closest to ``R``
    (the adjudication's cells 28 and 46 of the design they were built on)."""

    R: float
    theta_deg: float
    approach: str
    direction: int = 1
    width: float = 12.0
    ramp_kind: str = "typical"


PLANT_SITES = {"A1": PlantSite(14.6, 90.0, "A1"),     # cell 28: R14.6/T90/W12/typical/L, A1
               "A2": PlantSite(28.6, 90.0, "A2")}     # cell 46: R28.6/T90/W12/typical/L, A2


@dataclass(frozen=True)
class PlantSpec:
    """A planted crash control (ruling R30): Q ('Q') or the u = 0 nominal key's float32 tape
    ('tape', T32's path) with one planted fault, on ``sites``, and the type it must get
    (``analysis.planted_typing``). Offsets are metres from s_c0."""

    name: str
    base: str                          # 'Q' | 'tape'
    sites: tuple[str, ...]
    want: str                          # 'overbrake' | 'line_steer' | 'not overspeed'
    pedal_from: float | None = None    # pedal held at -1 from s_c0 + this (OB)
    steer_from: float | None = None    # steering scaled from s_c0 + this
    steer_gain: float = 1.0
    brake_gain: float | None = None    # the key's braking (pedal < 0) before s_c0 scaled (LSKF)


PLANTED = {
    "OB": PlantSpec("OB", "Q", ("A1", "A2"), "overbrake", pedal_from=-5.0),
    "LS": PlantSpec("LS", "Q", ("A1", "A2"), "line_steer", steer_from=2.0, steer_gain=0.25),
    "LSK": PlantSpec("LSK", "tape", ("A2",), "line_steer", steer_from=2.0, steer_gain=0.3),
    "LSKF": PlantSpec("LSKF", "tape", ("A2",), "not overspeed", steer_from=2.0, steer_gain=0.3,
                      brake_gain=0.9),
}
for _pl in PLANTED.values():
    CONTROLS[_pl.name] = ControlSpec(_pl.name, ("A",), {"crash_type": (_pl.want,)})
del _pl


def planted_cells(bank: StimulusBank, cells: list[Cell], name: str) -> list[Cell]:
    """The cells of planted control ``name``, one per site (``PLANT_SITES``): among the tier A
    cells on synthetic corners with the site's approach, direction, angle, width and ramp kind
    (no no-cap variant), the one whose R is closest in log R (ties: the lower cell id). A site
    with no such cell is left out."""
    out = []
    for site in PLANTED[name].sites:
        st = PLANT_SITES[site]
        best = None
        for c in cells:
            stim = bank.stimuli[c.stim_id]
            sp = getattr(stim, "spec", None)
            if ("A" not in c.tiers or c.approach != st.approach or sp is None
                    or int(stim.direction) != st.direction
                    or not math.isclose(float(sp.theta_deg), st.theta_deg)
                    or not math.isclose(float(sp.width), st.width)
                    or str(getattr(sp, "ramp_kind", "")) != st.ramp_kind
                    or bool(getattr(sp, "no_cap", False))):
                continue
            d = abs(math.log(float(sp.R) / st.R))
            if best is None or (d, c.cell_id) < best[0]:
                best = ((d, c.cell_id), c)
        if best is not None and all(c.cell_id != best[1].cell_id for c in out):
            out.append(best[1])
    return out


@functools.cache
def make_planted_q_policy(name: str) -> Policy:
    """Q with ``PLANTED[name]``'s fault (params QParams, aux Q's target row; s_corner is the
    row's s_c0): the pedal command held at -1 from s_c0 + pedal_from (the friction-ellipse
    clip bypassed, so the fronts lock while turning in: OB), or the steering command scaled
    by steer_gain from s_c0 + steer_from (a weak wheel at Q's own sub-limit speed: LS)."""
    pl = PLANTED[name]
    if pl.base != "Q":
        raise ValueError(f"{name} is not a Q-based planted control")
    q = make_q_policy("Q")

    def policy(params: QParams, o, state: racing.EnvState, tr: TrackArrays, aux):
        mean, ls = q(params, o, state, tr, aux)
        s, s_c0 = state.x[0], params.s_corner[aux]
        steer, pedal = mean[0], mean[1]
        if pl.pedal_from is not None:
            pedal = jnp.where(s >= s_c0 + pl.pedal_from, -1.0, pedal)
        if pl.steer_from is not None:
            steer = jnp.where(s >= s_c0 + pl.steer_from, pl.steer_gain * steer, steer)
        return jnp.stack([steer, pedal]).astype(o.dtype), ls

    return policy


def planted_tape(key_x: np.ndarray, key_a: np.ndarray, s_c0: float, name: str) -> np.ndarray:
    """``PLANTED[name]``'s tape (float32, T32's cast) from the u = 0 nominal key's states
    [W+1, 8] and commands [W, 2] (float64): steering times steer_gain from the first state
    at or past s_c0 + steer_from on, and (LSKF) the braking commands before the first state
    at or past s_c0 times brake_gain."""
    pl = PLANTED[name]
    if pl.base != "tape":
        raise ValueError(f"{name} is not a tape planted control")
    x, acts = np.asarray(key_x, float), np.asarray(key_a, float).copy()
    if pl.steer_from is not None:
        t_f = int(np.argmax(x[:, 0] >= s_c0 + pl.steer_from))
        acts[t_f:, 0] *= pl.steer_gain
    if pl.brake_gain is not None:
        t_c0 = int(np.argmax(x[:, 0] >= s_c0))
        head = acts[:t_c0, 1]
        head[head < 0] *= pl.brake_gain
    return acts.astype(np.float32)


def c2_lock_typing(c2_runs, bank: StimulusBank, cells: list[Cell], r_lock: float) -> dict:
    """R30's regression count of c2 (reported, never blocking: R29(1) lists these cells): its
    crashed deterministic runs on lock-limited cells (the tightest raw radius 1/max|kappa| <=
    ``r_lock``, the cells ``c2_planted_law`` lists) by type; passed when there is at least one
    and all are typed overspeed (the ellipse fallback with dv > CRASH_DV_TAU; c2 has no
    viability solves), as the adjudication expects (48 of 48)."""
    by = {c.cell_id: c for c in cells}
    det = np.asarray(c2_runs["mode"]).astype(str) == "det"
    crashed = np.asarray(c2_runs["crashed"]).astype(bool)
    ct = np.asarray(c2_runs["crash_type"]).astype(str)
    rows = []
    for i in np.flatnonzero(det & crashed):
        c = by.get(int(c2_runs["cell"][i]))
        if c is None:
            continue
        k_max = float(np.max(np.abs(np.asarray(bank.kappa[c.stim_id]))))
        if k_max > 0 and 1.0 / k_max <= r_lock:
            rows.append(dict(cell=c.cell_id, phase=int(c2_runs["phase"][i]), crash_type=str(ct[i]),
                             R_raw=1.0 / k_max))
    n_ov = sum(r["crash_type"] == "overspeed" for r in rows)
    return dict(passed=bool(rows) and n_ov == len(rows), n=len(rows), n_overspeed=n_ov,
                not_overspeed=[r for r in rows if r["crash_type"] != "overspeed"][:20],
                cells=sorted({r["cell"] for r in rows}), r_lock=float(r_lock))

# R20: T64 runs in its own float64 process (``rollout.run_tape_rollouts_x64``), so it is
# not among the in-process CONTROLS. It is blocking.
T64 = ControlSpec("T64", ("A", "G"), {"regret": ("zero",), "errors": ("within_tau",),
                                      "termination": ("none",)})
T64_REGRET_TOL = REPLAY_RTOL  # regret / |J*| <= 1e-6 (= analysis.TAPE_REGRET_RTOL)


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


TUNE_GATE = 0.9          # R14: Q's tier D (a) competence gate (protocol 2.4 P4's C(a) >= 0.9)


def tune_cv(bank: StimulusBank, cells: list[Cell], tables: ModelTables | CarTables,
            completion_fn: Callable[[float], np.ndarray] | None = None,
            gate_cells: list[Cell] | None = None
            ) -> tuple[float | None, dict[float, float]]:
    """The largest c_v in CV_CHOICES for which Q completes every deterministic run of
    ``cells`` (tier A, every phase; step 4) and, by ruling R14, passes the competence gate
    of protocol 2.4 P4 on ``gate_cells`` (tier D condition (a)): the fraction of corners
    whose phase runs complete in at least 3 of 4 (``gate_completion``) is >= P4's 0.9. A
    centreline driver can meet a real start that is outside its own envelope (Hockenheim
    corner 6 from S1's state: 295 km/h, 85 m before a kink whose centreline radius is 19 m),
    so 100% there would test the start, not the driver. ``completion_fn(c_v)`` returns the
    per-run completion over ``cells + gate_cells`` in that order (by default Q through the
    harness). Returns (c_v or None, {c_v: rate}), the rate being tier A's completion, or the
    gate's when tier A is complete."""
    gate_cells = list(gate_cells or [])
    allc = list(cells) + gate_cells
    if completion_fn is None:
        car = tables if isinstance(tables, CarTables) else car_tables(tables)
        policy = make_q_policy("Q")

        def completion_fn(c_v: float) -> np.ndarray:
            prm, aux = q_params(bank, allc, car, c_v)
            arr = run_control(policy, prm, aux, bank, allc)
            done = completion(arr, bank)
            order = {c.cell_id: i for i, c in enumerate(allc)}
            return done[np.argsort([order[int(c)] for c in arr["cell"]], kind="stable")]

    n_gate = sum(c.n_phases for c in gate_cells)
    rates: dict[float, float] = {}
    for c_v in CV_CHOICES:
        done = np.asarray(completion_fn(c_v), bool)
        n_a = len(done) - n_gate              # tier A runs come first
        if not done.size:
            rates[c_v] = float("nan")
            continue
        ok_a = bool(done[:n_a].all()) if n_a else True
        rates[c_v] = float(done[:n_a].mean()) if n_a else 1.0
        if ok_a and gate_cells:
            g = gate_completion(done[n_a:], [c.n_phases for c in gate_cells])
            rates[c_v] = g
            ok_a = g >= TUNE_GATE
        if ok_a:
            return c_v, rates
    return None, rates


def gate_completion(done: np.ndarray, n_phases: list[int]) -> float:
    """Ruling R14 / protocol 2.4 P4's C: the fraction of cells (corners) whose phase runs
    complete in at least 3 of 4 (``ceil(3 n / 4)`` of n), ``done`` in cell-major order."""
    ok, i = [], 0
    for n in n_phases:
        ok.append(int(np.sum(done[i:i + n])) >= int(np.ceil(0.75 * n)))
        i += n
    return float(np.mean(ok)) if ok else float("nan")


def knn_fallback(run_k: Callable[[int], tuple[Any, float]], gate: float = KNN_GATE,
                 c_a_at: Callable[[int], float] | None = None) -> tuple[Any, dict]:
    """Protocol 7's K: k = 1, falling back to k = 5 (distance weights) when C(a) < ``gate``
    (or C(a) is undefined). ``run_k(k)`` runs K with k neighbours and returns (its rollout
    arrays, its C(a)). C(a) is logged at both k whichever is used (dry-run plan step 1; the
    dry run found only k = 1's): when k = 1 passes, k = 5's C(a) comes from ``c_a_at(5)``
    (e.g. K run on the condition (a) cells only), else from ``run_k(5)``. Returns (the
    arrays of the k used, {k, C_a_k1, C_a_k5, gate, fallback})."""
    arr1, c1 = run_k(1)
    rec = dict(k=1, C_a_k1=float(c1), C_a_k5=None, gate=float(gate), fallback=False)
    if np.isfinite(c1) and c1 >= gate:
        c5 = c_a_at(KNN_K_FALLBACK) if c_a_at is not None else run_k(KNN_K_FALLBACK)[1]
        rec["C_a_k5"] = float(c5)
        return arr1, rec
    arr5, c5 = run_k(KNN_K_FALLBACK)
    rec.update(k=KNN_K_FALLBACK, C_a_k5=float(c5), fallback=True)
    return arr5, rec
