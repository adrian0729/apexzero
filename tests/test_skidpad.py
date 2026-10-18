"""Skidpad gate: steady-state cornering limit of the full bicycle model.

On a circle of radius R (kappa = 1/R, n = 0) a steady state needs dvx = dvy = dr = 0,
heading error xi = -atan(vy / vx) so dn/dt = 0, and r = kappa * sdot so dxi/dt = 0.
For each speed we solve for (vy, r, delta, pedal); the skidpad limit is the highest
speed with an equilibrium inside the slip limit. It must sit at or below the point-mass
envelope the QSS oracle uses (the bicycle cannot beat its own friction budget) and not
far below it (the axles are reasonably balanced).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import fsolve

from apexzero.oracle.qss import GG
from apexzero.vehicle.dynamics import body_derivatives, frenet_rates, tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import F1_2025 as P

jax.config.update("jax_enable_x64", True)


def equilibrium(vx, radius, guess):
    k = 1.0 / radius

    def residual(z):
        vy, r, delta, pedal = z
        xi = -np.arctan2(vy, vx)
        x = jnp.array([0.0, 0.0, xi, vx, vy, r, delta, pedal])
        u = jnp.array([delta / P.delta_max, pedal])
        dvx, dvy, dr, _, _ = body_derivatives(x, u, P, JAX)
        sdot, _, _ = frenet_rates(x, k, JAX)
        return [float(dvx), float(dvy), float(dr), float(r - k * sdot)]

    sol, info, ok, _ = fsolve(residual, guess, full_output=True, xtol=1e-12)
    resid = np.max(np.abs(residual(sol)))
    vy, r, delta, pedal = sol
    xi = -np.arctan2(vy, vx)
    fo = tyre_forces(jnp.array([0.0, 0.0, xi, vx, vy, r, delta, pedal]), P, JAX)
    slip_ok = max(abs(float(fo.alpha_f)), abs(float(fo.alpha_r))) <= P.alpha_max
    valid = ok == 1 and resid < 1e-6 and abs(pedal) <= 1.0 and abs(delta) <= P.delta_max and slip_ok
    return valid, sol


def skidpad_limit(radius):
    """(limit speed, pedal at the limit). Pedal near 1 means power-limited, not grip-limited."""
    guess = np.array([0.0, 10.0 / radius, (P.lf + P.lr) / radius, 0.1])
    v_ok, pedal = None, None
    for vx in np.arange(10.0, 95.0, 0.25):
        valid, sol = equilibrium(vx, radius, guess)
        if not valid:
            break
        v_ok, guess, pedal = vx, sol, sol[3]
    return v_ok, pedal


def test_grip_limited_skidpads_sit_just_inside_the_point_mass_envelope():
    gg = GG.from_params(P)
    for radius in (30.0, 50.0, 80.0):
        v, pedal = skidpad_limit(radius)
        assert v is not None and pedal < 0.95       # grip-limited, not power-limited
        a_lat, a_env = v * v / radius, gg.a_lat_max(v)
        # At most the point-mass envelope (+2% for its representative-load
        # approximation) and within 20% of it (reasonably balanced axles).
        assert 0.80 * a_env <= a_lat <= 1.02 * a_env, (radius, v, a_lat / P.g, a_env / P.g)


def test_large_radius_skidpad_is_power_limited_near_top_speed():
    v, pedal = skidpad_limit(400.0)
    assert pedal > 0.95
    assert v > 0.95 * 87.5   # straight-line terminal speed ~87.5 m/s
