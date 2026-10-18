"""Dynamic single-track (bicycle) model in Frenet coordinates.

State  x = [s, n, xi, vx, vy, r, delta, pedal]
    s      arc length along the reference line (m)
    n      lateral offset, positive to the left (m)
    xi     heading relative to the reference line, positive anticlockwise (rad)
    vx, vy body-frame velocity (m/s)
    r      yaw rate (rad/s)
    delta  road-wheel steering angle (rad)
    pedal  combined pedal state in [-1, 1]: >0 throttle, <0 brake
Input  u = [a_steer, a_pedal] in [-1, 1]^2 (commands, passed through lags and limits)

``f(x, u, kappa, p, ops)`` returns dx/dt and is the single source of truth for both the
JAX simulator and the CasADi optimal-control problem. The track enters only through the
reference curvature kappa(s), which the caller evaluates.

Model:
- Pacejka lateral tyre force with E = 0 and per-tyre load sensitivity of friction.
- Friction ellipse per axle, enforced by smoothly scaling (Fx, Fy) back onto it.
- Aerodynamic downforce (front/rear) and drag ~ v^2, rolling resistance on total load.
- Rear-wheel drive limited by P_max / v; brake force split by brake bias.
- Longitudinal load transfer from an algebraic acceleration estimate (no extra state).
- Blend to a kinematic bicycle below ~4 m/s, where slip angles are ill-conditioned.
"""

from __future__ import annotations

from typing import NamedTuple

from apexzero.vehicle.ops import Ops
from apexzero.vehicle.params import VehicleParams

STATE = ("s", "n", "xi", "vx", "vy", "r", "delta", "pedal")
N_STATE = len(STATE)
N_INPUT = 2

V_BLEND_LO = 2.0   # below: fully kinematic
V_BLEND_HI = 4.0   # above: fully dynamic
TAU_KIN = 0.1      # s, relaxation onto the kinematic manifold at low speed


class Forces(NamedTuple):
    fz_f: object
    fz_r: object
    fx_f: object
    fy_f: object
    fx_r: object
    fy_r: object
    alpha_f: object
    alpha_r: object
    util_f: object   # friction-ellipse utilisation before saturation
    util_r: object
    f_drag: object
    f_roll: object


def _mu(mu0, fz_tyre, p: VehicleParams, ops: Ops):
    return ops.smax(p.mu_scale * (mu0 + p.kmu * (fz_tyre - p.fz0)), 0.3, 1e-2)


def tyre_forces(x, p: VehicleParams, ops: Ops) -> Forces:
    """All axle forces for state x (pedal and steering are taken from the state)."""
    vx, vy, r = ops.index(x, 3), ops.index(x, 4), ops.index(x, 5)
    delta, pedal = ops.index(x, 6), ops.index(x, 7)
    L = p.lf + p.lr
    v2 = vx * vx

    # Static + aerodynamic axle loads.
    fz_f0 = p.m * p.g * p.lr / L + 0.5 * p.rho * p.cla_f * v2
    fz_r0 = p.m * p.g * p.lf / L + 0.5 * p.rho * p.cla_r * v2
    f_drag = 0.5 * p.rho * p.cda * v2
    f_roll = p.f_roll * (fz_f0 + fz_r0) * ops.tanh(vx / 0.5)

    # Longitudinal demands.
    throttle = ops.relu(pedal)
    # Brakes oppose motion and fade out at a standstill, so they can never drive the
    # car backwards.
    brake = ops.relu(-pedal) * ops.tanh(vx / 0.5)
    f_drive = throttle * p.p_max / ops.smax(vx, p.v_drive_floor, 0.5)
    fx_f_dem = -brake * p.f_brake_max * p.brake_bias
    fx_r_dem = f_drive - brake * p.f_brake_max * (1.0 - p.brake_bias)

    # Algebraic longitudinal-acceleration estimate for load transfer: saturate the
    # demands on static loads first, so a stamped brake pedal cannot transfer more load
    # than the tyres could ever generate.
    mux_f0 = _mu(p.mux_f, 0.5 * fz_f0, p, ops)
    mux_r0 = _mu(p.mux_r, 0.5 * fz_r0, p, ops)
    fx_f1 = ops.softsat(fx_f_dem, mux_f0 * fz_f0)
    fx_r1 = ops.softsat(fx_r_dem, mux_r0 * fz_r0)
    ax_est = (fx_f1 + fx_r1 - f_drag - f_roll) / p.m
    dfz = p.m * ax_est * p.h_cog / L
    fz_f = ops.smax(fz_f0 - dfz, 200.0, 10.0)
    fz_r = ops.smax(fz_r0 + dfz, 200.0, 10.0)

    # Slip angles (vx floored so they stay finite at a standstill).
    v_safe = ops.smax(vx, 1.0, 0.1)
    alpha_f = delta - ops.atan((vy + p.lf * r) / v_safe)
    alpha_r = -ops.atan((vy - p.lr * r) / v_safe)

    def axle(fx_dem, alpha, fz, mux0, muy0):
        mux = _mu(mux0, 0.5 * fz, p, ops)
        muy = _mu(muy0, 0.5 * fz, p, ops)
        fx_max = mux * fz
        fy_max = muy * fz
        fy_pure = fy_max * ops.sin(p.pac_c * ops.atan(p.pac_b * alpha))
        util = ops.sqrt((fx_dem / fx_max) ** 2 + (fy_pure / fy_max) ** 2 + 1e-8)
        scale = 1.0 / ops.smax(util, 1.0, 5e-3)
        return fx_dem * scale, fy_pure * scale, util

    fx_f, fy_f, util_f = axle(fx_f_dem, alpha_f, fz_f, p.mux_f, p.muy_f)
    fx_r, fy_r, util_r = axle(fx_r_dem, alpha_r, fz_r, p.mux_r, p.muy_r)
    return Forces(fz_f, fz_r, fx_f, fy_f, fx_r, fy_r, alpha_f, alpha_r, util_f, util_r,
                  f_drag, f_roll)


def body_derivatives(x, u, p: VehicleParams, ops: Ops):
    """d/dt of [vx, vy, r, delta, pedal] (everything except the Frenet kinematics)."""
    vx, vy, r = ops.index(x, 3), ops.index(x, 4), ops.index(x, 5)
    delta, pedal = ops.index(x, 6), ops.index(x, 7)
    a_steer, a_pedal = ops.index(u, 0), ops.index(u, 1)
    L = p.lf + p.lr
    fo = tyre_forces(x, p, ops)

    cd, sd = ops.cos(delta), ops.sin(delta)
    ffx = fo.fx_f * cd - fo.fy_f * sd
    ffy = fo.fx_f * sd + fo.fy_f * cd
    fx_ext = ffx + fo.fx_r - fo.f_drag - fo.f_roll

    # Actuators: first-order lag with a smooth rate limit on steering.
    delta_cmd = a_steer * p.delta_max
    ddelta = p.steer_rate_max * ops.tanh((delta_cmd - delta) / (p.tau_steer * p.steer_rate_max))
    dpedal = (a_pedal - pedal) / p.tau_pedal

    # Dynamic single-track model.
    dvx_dyn = fx_ext / p.m + vy * r
    dvy_dyn = (ffy + fo.fy_r) / p.m - vx * r
    dr_dyn = (p.lf * ffy - p.lr * fo.fy_r) / p.iz

    # Kinematic bicycle: yaw rate and side velocity follow r = vx tan(delta) / L and
    # vy = lr * r, by differentiating those relations plus a relaxation term that pulls a
    # state that starts off the kinematic manifold back onto it.
    dvx_kin = (fo.fx_f + fo.fx_r - fo.f_drag - fo.f_roll) / p.m
    tdel = ops.tan(delta)
    d_tan = dvx_kin * tdel + vx * ddelta / (cd * cd)
    r_kin = vx * tdel / L
    dvy_kin = p.lr / L * d_tan + (p.lr * r_kin - vy) / TAU_KIN
    dr_kin = d_tan / L + (r_kin - r) / TAU_KIN

    w = ops.smoothstep((vx - V_BLEND_LO) / (V_BLEND_HI - V_BLEND_LO))
    dvx = w * dvx_dyn + (1.0 - w) * dvx_kin
    dvy = w * dvy_dyn + (1.0 - w) * dvy_kin
    dr = w * dr_dyn + (1.0 - w) * dr_kin
    return dvx, dvy, dr, ddelta, dpedal


def frenet_rates(x, kappa, ops: Ops):
    """(ds/dt, dn/dt, dxi/dt excluding yaw rate) for reference curvature kappa(s)."""
    n, xi = ops.index(x, 1), ops.index(x, 2)
    vx, vy = ops.index(x, 3), ops.index(x, 4)
    ds = (vx * ops.cos(xi) - vy * ops.sin(xi)) / (1.0 - n * kappa)
    dn = vx * ops.sin(xi) + vy * ops.cos(xi)
    return ds, dn, -kappa * ds


def f(x, u, kappa, p: VehicleParams, ops: Ops):
    """dx/dt for the full Frenet-frame state."""
    ds, dn, dxi_track = frenet_rates(x, kappa, ops)
    dvx, dvy, dr, ddelta, dpedal = body_derivatives(x, u, p, ops)
    r = ops.index(x, 5)
    return ops.stack([ds, dn, r + dxi_track, dvx, dvy, dr, ddelta, dpedal])
