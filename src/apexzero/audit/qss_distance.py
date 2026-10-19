"""d_QSS: how far along the centreline a QSS point mass gets in one audit window.

The non-saturating S2 success (plan 2.2; score spec, S2) needs an *upper* reference
distance: a window succeeds under S2 when it completes and covers at least 97% of d_QSS.
The pilot first integrated the QSS profile over the strip's *centreline* curvature. That
reference is too low. The agent drives a racing line inside the corridor, which is less
curved and so allows higher QSS speeds, while progress is measured along the centreline.
Completed windows covered up to 1.23 d_QSS, and S2 collapsed onto S1.

This module defines d_QSS on a racing line instead:

1. **Line.** The minimum-curvature line inside the drivable corridor: lateral offsets
   alpha(s) from the centreline minimising the squared second differences of the line's
   points (the linearised curvature objective of the classic minimum-curvature QP), with
   alpha in [-w_right + margin, w_left - margin] and margin = the env's edge margin (the
   corridor the car's reference point may use before it is off track). The line starts on
   the centreline, as every level does (n = 0 at s = 0).
2. **Speed.** The forward-backward QSS profile (``oracle.qss.speed_profile``) of the level's
   own car (context and hidden grip) along that line, using the line's own segment lengths
   and curvature, starting from x0's speed (0.8 v_QSS(0), capped at the car's limit).
3. **Distance.** The centreline arc length reached after ``seconds`` along the line, which
   is what the env's progress measures (``distance = s_T - s_0``).

A minimum-curvature line is not the minimum-time line, and a point mass on a g-g diagram
evaluated at one representative tyre load is not the full vehicle model. So d_QSS is still
an approximation. ``ratio_quantiles`` (distance / d_QSS over completed windows) is the
diagnostic that says whether it works as an upper reference on a given set of rollouts.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from apexzero.oracle.qss import GG, speed_profile
from apexzero.tracks.reference import second_difference
from apexzero.vehicle.params import VehicleParams

LAMBDA = 1e-6          # weak pull toward the centreline: picks one line on straights
START_TOL = 1e-3       # the line starts on the centreline (|alpha(0)| <= START_TOL, m)


def strip_xy(kappa: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(theta, x, y) of an open strip from its curvature, integrated as in
    ``generator.strip_arrays`` (trapezoid rule from the origin, heading 0)."""
    k = np.asarray(kappa, float)
    theta = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * ds)])
    c, s = np.cos(theta), np.sin(theta)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (c[1:] + c[:-1]) * ds)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * ds)])
    return theta, x, y


def box_qp(h: sparse.spmatrix, c: np.ndarray, lo: np.ndarray, hi: np.ndarray, x0=None,
           tol: float = 1e-9, max_iter: int = 200) -> tuple[np.ndarray, int]:
    """min 0.5 x'Hx - c'x over lo <= x <= hi, for sparse positive-definite H, by projected
    Newton (Bertsekas 1982): a Newton step on the variables not held at a bound, an Armijo
    search along the projection arc, and the epsilon-active set. Returns (x, iterations).

    ``scipy.optimize.lsq_linear`` solves the same problem but needs minutes per 8 km strip
    when the regulariser is weak (the Hessian is badly conditioned); here every iteration is
    one sparse factorisation of a banded matrix."""
    h = sparse.csr_matrix(h)
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    x = np.clip(np.zeros_like(c) if x0 is None else np.asarray(x0, float), lo, hi)

    def f(z):
        return 0.5 * z @ (h @ z) - c @ z

    fx = f(x)
    for it in range(1, max_iter + 1):
        g = h @ x - c
        pg = x - np.clip(x - g, lo, hi)                   # projected gradient
        if np.max(np.abs(pg)) < tol * max(1.0, np.max(np.abs(c))):
            return x, it - 1
        eps = min(1e-3, float(np.linalg.norm(pg)))
        held = ((x <= lo + eps) & (g > 0)) | ((x >= hi - eps) & (g < 0))
        free = np.flatnonzero(~held)
        d = np.zeros_like(x)
        if free.size:
            d[free] = spsolve(h[free][:, free].tocsc(), -g[free])
        d[held] = -g[held]                                # gradient step on held variables
        t = 1.0
        while True:
            xn = np.clip(x + t * d, lo, hi)
            fn = f(xn)
            if fn <= fx + 1e-4 * (g @ (xn - x)) or t < 1e-12:
                break
            t *= 0.5
        if fn >= fx and np.array_equal(xn, x):
            return x, it
        x, fx = xn, fn
    return x, max_iter


def min_curvature_offsets(x, y, theta, w_left, w_right, margin: float, lam: float = LAMBDA,
                          start_offset: float | None = 0.0) -> np.ndarray:
    """Lateral offsets alpha [N] of the minimum-curvature line of an open strip.

    Minimises ||D2 (c + alpha N)||^2 + lam ||alpha||^2 over the box
    -w_right + margin <= alpha <= w_left - margin (centreline points c, left normals N, D2
    the second difference with free ends). Where the corridor is narrower than two margins
    the box collapses to its midpoint. ``start_offset`` pins alpha(0); None leaves it free.
    """
    x, y, theta = (np.asarray(v, float) for v in (x, y, theta))
    n = len(x)
    nx, ny = -np.sin(theta), np.cos(theta)
    d2 = second_difference(n, closed=False)
    a = sparse.vstack([d2 @ sparse.diags(nx), d2 @ sparse.diags(ny),
                       np.sqrt(lam) * sparse.identity(n)]).tocsr()
    b = -np.concatenate([d2 @ x, d2 @ y, np.zeros(n)])
    lo = -np.asarray(w_right, float) + margin
    hi = np.asarray(w_left, float) - margin
    mid = 0.5 * (lo + hi)
    narrow = hi - lo < 2 * START_TOL
    lo = np.where(narrow, mid - START_TOL, lo)
    hi = np.where(narrow, mid + START_TOL, hi)
    if start_offset is not None:
        lo[0], hi[0] = start_offset - START_TOL, start_offset + START_TOL
    alpha, _ = box_qp(a.T @ a, a.T @ b, lo, hi)
    return alpha


def line_geometry(kappa, alpha, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """(segment lengths [N-1], curvature [N]) of the line at lateral offset ``alpha`` from
    a centreline of curvature ``kappa``, in the Frenet frame the env integrates in: with
    e = 1 - alpha kappa and xi = atan2(alpha', e) the heading relative to the centreline,
    the line advances sigma' = sqrt(e^2 + alpha'^2) per metre of centreline and has
    curvature (kappa + xi') / sigma'. On the centreline (alpha = 0) these are exactly
    ``ds`` and ``kappa``."""
    k, n = np.asarray(kappa, float), np.asarray(alpha, float)
    dn = np.gradient(n, ds)
    e = 1.0 - n * k
    xi = np.arctan2(dn, e)
    sig = np.hypot(e, dn)
    k_line = (k + np.gradient(xi, ds)) / sig
    return 0.5 * (sig[1:] + sig[:-1]) * ds, k_line


def progress_in_time(v: np.ndarray, seg: np.ndarray, ds: float, seconds: float) -> float:
    """Centreline arc length reached after ``seconds`` along a line sampled at the
    centreline's nodes (spacing ``ds``), with speeds ``v`` [N] at the nodes and segment
    lengths ``seg`` [N-1] along the line; constant acceleration between nodes. Capped at
    the strip's end."""
    t = np.concatenate([[0.0], np.cumsum(2.0 * seg / (v[:-1] + v[1:]))])
    return float(np.interp(seconds, t, np.arange(len(v)) * ds))


def qss_distance(kappa, w_left, w_right, params: VehicleParams, v0: float, ds: float,
                 seconds: float, margin: float, line: str = "racing", lam: float = LAMBDA) -> float:
    """d_QSS of one level: centreline distance covered in ``seconds`` by the QSS point mass
    of ``params`` on the racing line (``line="racing"``) or on the centreline
    (``line="centre"``, the pilot's first definition but starting from ``v0``)."""
    k = np.asarray(kappa, float)
    if line == "racing":
        theta, x, y = strip_xy(k, ds)
        alpha = min_curvature_offsets(x, y, theta, w_left, w_right, margin, lam)
    elif line == "centre":
        alpha = np.zeros_like(k)
    else:
        raise ValueError(f"line must be 'racing' or 'centre', got {line!r}")
    seg, k_line = line_geometry(k, alpha, ds)
    p = VehicleParams(*(float(f) for f in params))
    v = speed_profile(k_line, seg, GG.from_params(p), closed=False, v_start=float(v0))
    return progress_in_time(v, seg, ds, seconds)


def ratio_quantiles(distance: np.ndarray, completed: np.ndarray, d_qss: np.ndarray,
                    qs=(0.5, 0.9, 0.99, 1.0)) -> dict[str, float]:
    """Quantiles of distance / d_QSS over completed windows ([L, W] arrays, d_qss [L]),
    plus the share of completed windows above 1. A valid upper reference keeps the ratio
    at or below about 1."""
    r = (np.asarray(distance, float) / np.asarray(d_qss, float)[:, None])[np.asarray(completed, bool)]
    if r.size == 0:
        return {**{f"q{q:g}": float("nan") for q in qs}, "above_1": float("nan"), "n": 0}
    out = {f"q{q:g}": float(np.quantile(r, q)) for q in qs}
    out["above_1"] = float(np.mean(r > 1.0))
    out["n"] = int(r.size)
    return out
