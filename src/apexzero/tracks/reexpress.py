"""Model (2b) of GW-2: re-express a generated strip on a smoothed reference line.

Every R-dev evaluation drives ``refine_reference(track)``. On 6 of the 12 circuits that
moves the reference line toward hairpin apexes, so the evaluated widths pair the
reference's curvature with an inside edge that is not parallel to it. The generator never
does this: its strips already satisfy the Frenet cap. Model (2b) generates a strip whose
widths are drawn as (2a) draws them, but fitted on the circuits' raw centreline profiles
and without the inside cap, and then re-expresses it exactly where and as
``refine_reference`` would. This module is a JAX port of that step for open strips
(``reexpress``), so it runs inside the training bank.

The port, step by step against ``tracks/reference.py``:
- **Trigger:** ``drivable_margin`` (shrink 1 m) of the output part of the raw strip above
  ``TARGET``; otherwise the strip is kept.
- **Offsets:** ``_offsets_qp``'s box-constrained least squares,
  min ||D2 (c + alpha N)||^2 + lam ||alpha||^2, s.t. -w_right + 3 <= alpha <= w_left - 3,
  with the open second difference (zero end rows). It is solved by FISTA with a fixed
  ``ITERS`` iterations per lam, warm-started from the previous lam (the ladder ``LAMS``
  runs from strong to weak regularisation, as refine tries it). The bounds are widened to
  include 0 where the road is narrower than 6 m (refine never meets such a road). Second
  differences of positions are taken from the headings, not from absolute coordinates,
  so float32 does not cancel them.
- **Reference line:** the offset points' polyline, its discrete curvature (heading change
  over the mean adjacent segment length), resampled at ``ds`` from its start. This
  replaces ``from_centreline``'s 1 cm quintic smoothing spline: the offsets already
  minimise second differences. Linear resampling puts an origin up to l^2 kappa / 8 off
  the arc (5 cm at a 10 m radius, 1 cm at 50 m).
- **Widths:** ``rebase``: rays along the new normals to the raw strip's edge polylines,
  searched within ``WINDOW`` segments of a hint (proportional arc length, refined to the
  nearest raw knot within 3 x ``WINDOW``), with the nearest-vertex fallback.
- **Choice:** refine's: the first lam whose reference radius stays >= ``MIN_RADIUS`` and
  whose drivable margin is <= ``TARGET``; else the valid lam with the lowest margin, if it
  is below the raw strip's; else the raw strip.

The raw strip is generated ``EXTRA_KNOTS`` longer than the output, because the
re-expressed line is shorter (it cuts corners); the output is its first ``n_points``
knots. The generator applies its inside cap afterwards, as for every strip.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

TARGET = 0.6
INNER_MARGIN = 3.0
MIN_RADIUS = 6.0
SHRINK = 1.0
LAMS = (3e-2, 1e-2, 3e-3, 1e-3)
ITERS = 1500
WINDOW = 40
EXTRA_KNOTS = 256


def _d2(v: jax.Array) -> jax.Array:
    """The open second difference (zero first and last rows), as ``second_difference``."""
    z = jnp.zeros(1, v.dtype)
    return jnp.concatenate([z, v[:-2] - 2.0 * v[1:-1] + v[2:], z])


def _d2t(r: jax.Array) -> jax.Array:
    """Its transpose."""
    r = r.at[0].set(0.0).at[-1].set(0.0)
    return (-2.0 * r).at[:-1].add(r[1:]).at[1:].add(r[:-1])


def centreline(kappa: jax.Array, ds: float):
    """Heading, the increments and positions of an open strip from its curvature, as
    ``generator.strip_arrays`` integrates them (trapezoid rule, starting at the origin)."""
    theta = jnp.concatenate([jnp.zeros(1, kappa.dtype),
                             jnp.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])
    dx = 0.5 * (jnp.cos(theta[1:]) + jnp.cos(theta[:-1])) * ds
    dy = 0.5 * (jnp.sin(theta[1:]) + jnp.sin(theta[:-1])) * ds
    z = jnp.zeros(1, kappa.dtype)
    return theta, dx, dy, jnp.concatenate([z, jnp.cumsum(dx)]), jnp.concatenate([z, jnp.cumsum(dy)])


def offsets(d2x, d2y, nx, ny, lo, hi, lam, a0, iters: int = ITERS):
    """argmin ||d2x + D2 (nx a)||^2 + ||d2y + D2 (ny a)||^2 + lam ||a||^2 over lo <= a <= hi,
    by FISTA from a0 (step 1 / L with L = 2 (16 + lam) >= the gradient's Lipschitz bound)."""
    lip = 2.0 * (16.0 + lam)

    def grad(a):
        return 2.0 * (nx * _d2t(d2x + _d2(nx * a)) + ny * _d2t(d2y + _d2(ny * a)) + lam * a)

    def body(_, st):
        a, z, t = st
        a_new = jnp.clip(z - grad(z) / lip, lo, hi)
        t_new = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * t * t))
        return a_new, a_new + ((t - 1.0) / t_new) * (a_new - a), t_new

    a, _, _ = jax.lax.fori_loop(0, iters, body, (a0, a0, jnp.asarray(1.0, a0.dtype)))
    return a


def _wrap(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


def reference_line(dx, dy, alpha, nx, ny, n_out: int, ds: float):
    """The offset points' polyline resampled at ``ds`` from its start: (x, y, theta, kappa)
    [n_out] relative to the strip's origin, and the polyline's arc length at its knots."""
    px = dx + alpha[1:] * nx[1:] - alpha[:-1] * nx[:-1]          # segment increments
    py = dy + alpha[1:] * ny[1:] - alpha[:-1] * ny[:-1]
    seg = jnp.hypot(px, py)
    phi = jnp.arctan2(py, px)
    phi = phi[0] + jnp.concatenate([jnp.zeros(1, phi.dtype), jnp.cumsum(_wrap(jnp.diff(phi)))])
    z = jnp.zeros(1, dx.dtype)
    s_p = jnp.concatenate([z, jnp.cumsum(seg)])
    theta_v = jnp.concatenate([phi[:1], 0.5 * (phi[1:] + phi[:-1]), phi[-1:]])
    kappa_v = jnp.concatenate([z, (phi[1:] - phi[:-1]) / (0.5 * (seg[1:] + seg[:-1])), z])
    x_p = alpha[0] * nx[0] + jnp.concatenate([z, jnp.cumsum(px)])
    y_p = alpha[0] * ny[0] + jnp.concatenate([z, jnp.cumsum(py)])
    s = jnp.arange(n_out, dtype=dx.dtype) * ds
    f = lambda v: jnp.interp(s, s_p, v)
    return f(x_p), f(y_p), f(theta_v), f(kappa_v), s_p


def _ray_distance(ox, oy, nx, ny, ex, ey, hint, window: int = WINDOW):
    """Distance along (nx, ny) from each origin to the polyline (ex, ey), searching the
    segments within ``window`` of ``hint``; the nearest-vertex distance where every ray
    misses (``reference._ray_distances``). A loop over the window keeps memory O(n)."""
    m = ex.shape[0]

    def body(j, carry):
        hit, near = carry
        idx = jnp.clip(hint + j - window, 0, m - 1)
        nxt = jnp.minimum(idx + 1, m - 1)
        ux, uy = ex[nxt] - ex[idx], ey[nxt] - ey[idx]
        wx, wy = ex[idx] - ox, ey[idx] - oy
        denom = nx * uy - ny * ux
        good = jnp.abs(denom) > 1e-12
        safe = jnp.where(good, denom, 1.0)
        t = (wx * uy - wy * ux) / safe
        u = (wx * ny - wy * nx) / safe
        ok = good & (u >= -1e-9) & (u <= 1 + 1e-9) & (t > 0)
        return jnp.minimum(hit, jnp.where(ok, t, jnp.inf)), jnp.minimum(near, jnp.hypot(wx, wy))

    inf = jnp.full(ox.shape, jnp.inf, ox.dtype)
    hit, near = jax.lax.fori_loop(0, 2 * window + 1, body, (inf, inf))
    return jnp.where(jnp.isfinite(hit), hit, near)


def _nearest(ox, oy, cx, cy, hint, window: int):
    """The index of the nearest of (cx, cy) within ``window`` of ``hint`` (first on ties)."""
    m = cx.shape[0]

    def body(j, carry):
        best_d, best_i = carry
        idx = jnp.clip(hint + j - window, 0, m - 1)
        d = jnp.hypot(cx[idx] - ox, cy[idx] - oy)
        better = d < best_d
        return jnp.where(better, d, best_d), jnp.where(better, idx, best_i)

    init = (jnp.full(ox.shape, jnp.inf, ox.dtype), hint)
    return jax.lax.fori_loop(0, 2 * window + 1, body, init)[1]


def drivable_margin(kappa, w_left, w_right, shrink: float = SHRINK):
    """``reference.drivable_margin``: max of (inside width - shrink)+ x |kappa|."""
    inside = jnp.where(kappa > 0, w_left, w_right) - shrink
    return jnp.max(jnp.clip(inside, 0.0, None) * jnp.abs(kappa))


def reexpress(kappa, w_left, w_right, lams, n_out: int, ds: float = 2.0, iters: int = ITERS):
    """Re-express an open raw strip (kappa, w_left, w_right) [n] as ``refine_reference``
    would, and return its first ``n_out`` knots (kappa, w_left, w_right) and the index of
    the lam chosen (-1: kept as it is)."""
    n = kappa.shape[0]
    theta, dx, dy, cx, cy = centreline(kappa, ds)
    nx, ny = -jnp.sin(theta), jnp.cos(theta)
    d2x = jnp.concatenate([jnp.zeros(1, dx.dtype), dx[1:] - dx[:-1], jnp.zeros(1, dx.dtype)])
    d2y = jnp.concatenate([jnp.zeros(1, dy.dtype), dy[1:] - dy[:-1], jnp.zeros(1, dy.dtype)])
    lo = jnp.minimum(-w_right + INNER_MARGIN, 0.0)
    hi = jnp.maximum(w_left - INNER_MARGIN, 0.0)
    lx, ly = cx + w_left * nx, cy + w_left * ny                 # the edges
    rx, ry = cx - w_right * nx, cy - w_right * ny

    def candidate(a0, lam):
        alpha = offsets(d2x, d2y, nx, ny, lo, hi, lam, a0, iters)
        x, y, th, kap, s_p = reference_line(dx, dy, alpha, nx, ny, n_out, ds)
        ox, oy = cx[0] + x, cy[0] + y
        hint = jnp.clip(jnp.round(jnp.arange(n_out) * ds / s_p[-1] * (n - 1)).astype(jnp.int32), 0, n - 1)
        hint = _nearest(ox, oy, cx, cy, hint, 3 * WINDOW)
        mx, my = -jnp.sin(th), jnp.cos(th)
        wl = _ray_distance(ox, oy, mx, my, lx, ly, hint)
        wr = _ray_distance(ox, oy, -mx, -my, rx, ry, hint)
        radius_ok = 1.0 / jnp.max(jnp.abs(kap)) >= MIN_RADIUS
        long_enough = s_p[-1] >= (n_out - 1) * ds
        valid = radius_ok & long_enough
        return alpha, (kap, wl, wr, valid, drivable_margin(kap, wl, wr))

    _, (kap_c, wl_c, wr_c, valid, score) = jax.lax.scan(candidate, jnp.zeros_like(kappa),
                                                       jnp.asarray(lams, kappa.dtype))
    base = drivable_margin(kappa[:n_out], w_left[:n_out], w_right[:n_out])
    hit = valid & (score <= TARGET)
    k = len(lams)
    first_hit = jnp.argmax(hit)
    masked = jnp.where(valid, score, jnp.inf)
    best = jnp.argmin(masked)
    choice = jnp.where(hit.any(), first_hit, jnp.where(masked[best] < base, best, -1))
    choice = jnp.where(base > TARGET, choice, -1)
    pick = lambda c, raw: jnp.where(choice >= 0, c[jnp.clip(choice, 0, k - 1)], raw[:n_out])
    return pick(kap_c, kappa), pick(wl_c, w_left), pick(wr_c, w_right), choice


def lams_array() -> np.ndarray:
    return np.asarray(LAMS, dtype=np.float32)
