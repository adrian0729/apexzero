"""JAX integration of the shared vehicle ODE on a track.

Tracks are stored as fixed-length arrays so that batches of different tracks can be
vmapped together: shorter tracks are padded and carry their true length and point count.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.tracks.spline import eval_jax, packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle.dynamics import f
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import VehicleParams


class TrackArrays(NamedTuple):
    # C2 cubic B-spline coefficients (see tracks/spline.py), length P + 2. The dynamics,
    # corridor and preview evaluate these, identically to the CasADi OCP.
    kappa_c: jax.Array
    w_left_c: jax.Array
    w_right_c: jax.Array
    # Pose arrays (piecewise-linear lookup; observations and plotting only).
    x: jax.Array         # [P]
    y: jax.Array         # [P]
    theta: jax.Array     # [P] unwrapped heading
    ds: jax.Array        # scalar
    n_points: jax.Array  # scalar int, valid points (<= P)
    closed: jax.Array    # scalar bool


def track_arrays(track: Track, pad_to: int | None = None, dtype=jnp.float32) -> TrackArrays:
    n = track.n_points
    pad_to = pad_to or n
    if pad_to < n:
        raise ValueError(f"track {track.name} has {n} points > pad_to={pad_to}")

    def pad(a):
        out = np.full(pad_to, a[-1], dtype=float)
        out[:n] = a
        return jnp.asarray(out, dtype=dtype)

    def coef(a):
        return jnp.asarray(packed_coeffs(np.asarray(a, float), track.closed, pad_to), dtype=dtype)

    return TrackArrays(
        kappa_c=coef(track.kappa), w_left_c=coef(track.w_left), w_right_c=coef(track.w_right),
        x=pad(track.x), y=pad(track.y), theta=pad(track.theta),
        ds=jnp.asarray(track.ds, dtype=dtype), n_points=jnp.asarray(n, dtype=jnp.int32),
        closed=jnp.asarray(track.closed),
    )


def track_length(tr: TrackArrays) -> jax.Array:
    n = tr.n_points.astype(tr.ds.dtype)
    return jnp.where(tr.closed, n * tr.ds, (n - 1) * tr.ds)


def lookup(arr: jax.Array, s: jax.Array, tr: TrackArrays) -> jax.Array:
    """Linear interpolation of a per-point array at arc length s (periodic if closed)."""
    n = tr.n_points
    i = s / tr.ds
    i = jnp.where(tr.closed, jnp.mod(i, n.astype(i.dtype)), jnp.clip(i, 0.0, (n - 1).astype(i.dtype)))
    i0 = jnp.floor(i).astype(jnp.int32)
    frac = i - i0
    i1 = jnp.where(tr.closed, jnp.mod(i0 + 1, n), jnp.minimum(i0 + 1, n - 1))
    i0 = jnp.minimum(i0, n - 1)
    return (1.0 - frac) * arr[i0] + frac * arr[i1]


def spline(coef: jax.Array, s: jax.Array, tr: TrackArrays) -> jax.Array:
    """C2 spline value at arc length s (kappa, widths)."""
    return eval_jax(coef, s, tr.ds, tr.n_points, tr.closed)


def kappa_at(s, tr: TrackArrays):
    return spline(tr.kappa_c, s, tr)


def widths_at(s, tr: TrackArrays):
    return spline(tr.w_left_c, s, tr), spline(tr.w_right_c, s, tr)


def lookup_angle(arr: jax.Array, s: jax.Array, tr: TrackArrays) -> jax.Array:
    """Like ``lookup`` for an unwrapped angle array: interpolates the wrapped difference,
    so the closing segment of a closed track (theta jumps by 2*pi*turns) stays correct."""
    n = tr.n_points
    i = s / tr.ds
    i = jnp.where(tr.closed, jnp.mod(i, n.astype(i.dtype)), jnp.clip(i, 0.0, (n - 1).astype(i.dtype)))
    i0 = jnp.minimum(jnp.floor(i).astype(jnp.int32), n - 1)
    frac = i - i0
    i1 = jnp.where(tr.closed, jnp.mod(i0 + 1, n), jnp.minimum(i0 + 1, n - 1))
    d = arr[i1] - arr[i0]
    d = jnp.mod(d + jnp.pi, 2 * jnp.pi) - jnp.pi
    return arr[i0] + frac * d


def deriv(x, u, tr: TrackArrays, p: VehicleParams):
    return f(x, u, kappa_at(x[..., 0], tr), p, JAX)


def rk2_step(x, u, tr: TrackArrays, p: VehicleParams, dt):
    """Explicit midpoint rule."""
    k1 = deriv(x, u, tr, p)
    k2 = deriv(x + 0.5 * dt * k1, u, tr, p)
    return x + dt * k2


def rk4_step(x, u, tr: TrackArrays, p: VehicleParams, dt):
    k1 = deriv(x, u, tr, p)
    k2 = deriv(x + 0.5 * dt * k1, u, tr, p)
    k3 = deriv(x + 0.5 * dt * k2, u, tr, p)
    k4 = deriv(x + dt * k3, u, tr, p)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def control_step(x, u, tr: TrackArrays, p: VehicleParams, dt: float = 0.05,
                 substeps: int = 10, method: str = "rk2"):
    """Hold the command u for one control period dt, integrating with `substeps` steps."""
    stepper = rk2_step if method == "rk2" else rk4_step
    h = dt / substeps

    def body(xc, _):
        return stepper(xc, u, tr, p, h), None

    x, _ = jax.lax.scan(body, x, None, length=substeps)
    return x


def edges_at(s, tr: TrackArrays):
    """World-frame left and right edge points at arc length(s) s."""
    cx, cy, th = lookup(tr.x, s, tr), lookup(tr.y, s, tr), lookup_angle(tr.theta, s, tr)
    wl, wr = widths_at(s, tr)
    sn, cs = jnp.sin(th), jnp.cos(th)
    return (cx - wl * sn, cy + wl * cs), (cx + wr * sn, cy - wr * cs)


def global_pose(x, tr: TrackArrays):
    """World-frame (X, Y, psi) of the car's reference point."""
    s, n, xi = x[..., 0], x[..., 1], x[..., 2]
    cx, cy, th = lookup(tr.x, s, tr), lookup(tr.y, s, tr), lookup_angle(tr.theta, s, tr)
    return cx - n * jnp.sin(th), cy + n * jnp.cos(th), th + xi
