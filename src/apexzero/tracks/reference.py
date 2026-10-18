"""Reference lines that keep the Frenet frame non-singular.

The Frenet map (s, n) -> (x, y) is singular where 1 - n * kappa_ref(s) = 0. On tight
hairpins (Austin T1, Shanghai T14, Norisring) the inside edge sits almost at the
centreline's centre of curvature, so the centreline is a poor reference. The reference
line does not have to be the centreline: any smooth curve inside the corridor works, as
long as the corridor edges stay well inside its radius of curvature.

``refine_reference`` solves a box-constrained least-squares problem for lateral offsets
alpha(s) from the centreline:

    min  || D2 (c + alpha * N) ||^2 + lam * || alpha ||^2
    s.t. -w_right + margin <= alpha <= w_left - margin

and lowers ``lam`` until max(n_inside * |kappa_ref|) <= target. The edges are kept
exactly; only the reference line and the widths measured from it change.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from apexzero.tracks.track import Track, from_centreline


def second_difference(n: int, closed: bool) -> sparse.csr_matrix:
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    d2 = sparse.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="lil")
    if closed:
        d2[0, n - 1] = 1.0
        d2[n - 1, 0] = 1.0
    else:
        d2[0, :] = 0.0
        d2[n - 1, :] = 0.0
    return d2.tocsr()


def _offsets_qp(track: Track, lam: float, margin: float) -> np.ndarray:
    n = track.n_points
    nx, ny = -np.sin(track.theta), np.cos(track.theta)
    d2 = second_difference(n, track.closed)
    a = sparse.vstack([
        d2 @ sparse.diags(nx),
        d2 @ sparse.diags(ny),
        np.sqrt(lam) * sparse.identity(n),
    ]).tocsr()
    b = -np.concatenate([d2 @ track.x, d2 @ track.y, np.zeros(n)])
    lo = -track.w_right + margin
    hi = track.w_left - margin
    res = lsq_linear(a, b, bounds=(lo, hi), method="trf", lsmr_tol="auto", max_iter=500)
    return res.x


def _window(hint: np.ndarray, window: int, m: int, closed: bool) -> np.ndarray:
    idx = hint[:, None] + np.arange(-window, window + 1)[None, :]
    return idx % m if closed else np.clip(idx, 0, m - 1)


def _ray_distances(
    origins: np.ndarray, normals: np.ndarray, poly: np.ndarray, hint: np.ndarray,
    window: int, closed: bool,
) -> np.ndarray:
    """Distance along +normal from each origin to the polyline, searching near ``hint``.

    Falls back to the nearest-vertex distance where the ray misses the window.
    """
    m = len(poly)
    idx = _window(hint, window, m, closed)
    nxt = (idx + 1) % m if closed else np.minimum(idx + 1, m - 1)
    p0, p1 = poly[idx], poly[nxt]
    e = p1 - p0
    w = p0 - origins[:, None, :]
    d = normals[:, None, :]
    denom = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / denom
        u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / denom
    ok = (np.abs(denom) > 1e-12) & (u >= -1e-9) & (u <= 1 + 1e-9) & (t > 0)
    t = np.where(ok, t, np.inf)
    hit = t.min(axis=1)
    miss = ~np.isfinite(hit)
    if miss.any():
        hit[miss] = np.linalg.norm(w[miss], axis=2).min(axis=1)
    return hit


def _nearest_index(points: np.ndarray, poly: np.ndarray, hint: np.ndarray, window: int,
                   closed: bool) -> np.ndarray:
    idx = _window(hint, window, len(poly), closed)
    dist = np.linalg.norm(poly[idx] - points[:, None, :], axis=2)
    return idx[np.arange(len(points)), dist.argmin(axis=1)]


def rebase(track: Track, ref: Track, window: int = 40) -> Track:
    """Re-measure ``track``'s edges as left/right widths from a new reference line."""
    left, right = track.edges()
    nx, ny = -np.sin(ref.theta), np.cos(ref.theta)
    normals = np.stack([nx, ny], axis=1)
    origins = np.stack([ref.x, ref.y], axis=1)
    # Search hint: proportional arc length, refined to the nearest centreline sample.
    centre = np.stack([track.x, track.y], axis=1)
    hint = np.round(ref.s / ref.length * track.n_points).astype(int) % track.n_points
    hint = _nearest_index(origins, centre, hint, 3 * window, track.closed)
    wl = _ray_distances(origins, normals, left, hint, window, track.closed)
    wr = _ray_distances(origins, -normals, right, hint, window, track.closed)
    return Track(
        name=track.name, s=ref.s, x=ref.x, y=ref.y, theta=ref.theta, kappa=ref.kappa,
        w_left=wl, w_right=wr, closed=track.closed,
        meta=dict(track.meta, reference="refined"),
    )


def drivable_margin(track: Track, shrink: float = 1.0) -> float:
    """max over s of (w_inside - shrink) * |kappa|: the Frenet margin the car can reach.

    ``shrink`` is the corridor reduction used for termination (car half-width), so this is
    the largest n * kappa the vehicle's reference point can ever see.
    """
    inside = np.where(track.kappa > 0, track.w_left, track.w_right) - shrink
    return float(np.max(np.clip(inside, 0.0, None) * np.abs(track.kappa)))


def refine_reference(
    track: Track,
    target: float = 0.6,
    inner_margin: float = 3.0,
    min_radius: float = 6.0,
    lams: tuple[float, ...] = (3e-2, 1e-2, 3e-3, 1e-3),
) -> Track:
    """Re-express ``track`` on a smoother reference line if its drivable margin > target.

    Tries progressively weaker regularisation toward the centreline and keeps the
    candidate with the lowest drivable margin whose reference radius stays above
    ``min_radius`` (very weak regularisation wraps the line around the inside apex
    like a string round a peg, which creates a kink). Tracks already within the
    target are returned unchanged.
    """
    base = drivable_margin(track)
    if base <= target:
        return track
    best = (base, track, None)
    nx, ny = -np.sin(track.theta), np.cos(track.theta)
    for lam in lams:
        alpha = _offsets_qp(track, lam, inner_margin)
        xy = np.stack([track.x + alpha * nx, track.y + alpha * ny], axis=1)
        ref = from_centreline(
            track.name, xy, np.ones(len(xy)), np.ones(len(xy)), ds=track.ds,
            smoothing=0.01, closed=track.closed,
        )
        cand = rebase(track, ref)
        if 1.0 / np.abs(cand.kappa).max() < min_radius:
            continue
        score = drivable_margin(cand)
        if score < best[0]:
            best = (score, cand, lam)
        if score <= target:
            break
    score, cand, lam = best
    if cand is not track:
        cand.meta.update(reference_lambda=lam, drivable_margin=score)
    return cand
