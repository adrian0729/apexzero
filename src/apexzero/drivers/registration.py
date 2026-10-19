"""Register a session's position feed onto the TUMFTM centreline.

OpenF1 positions live in a circuit-local frame of the timing system (decimetres, an
arbitrary origin and orientation). The TUMFTM centreline lives in its own local frame.
A similarity transform (scale, rotation, translation, optionally a reflection) maps one
onto the other:

1. A coarse grid over rotations (and both handednesses), after matching centroids and
   scale, picks the basin; a few trimmed ICP iterations score each candidate.
2. Full trimmed ICP (Umeyama's closed form on nearest-point correspondences) refines it.

The gate (plan section 6): reject a session if the RMS distance of the registered points
from the centreline exceeds 3 m, or if the two track models disagree on lap length by
more than 1%. The length check is what catches an old TUMFTM layout whose changed
section is short enough to hide inside the RMS.

Caveat found on the 2025 data: the feed's track model is not the centreline. In tight
corners it sits 1.5-2.8 m to the inside and its lap is 0.4-1.7% shorter, i.e. it looks
like a racing line. Both gate statistics therefore carry a racing-line component, and
the gate as written also rejects some current layouts. ``corridor_excursion`` (how far
outside the TUMFTM track edges the feed goes) separates an old layout much more sharply
and is reported alongside; the gate itself is left as the plan specifies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.drivers.geometry import Centreline

RMS_MAX_M = 3.0
LENGTH_TOL = 0.01


@dataclass(frozen=True)
class Similarity:
    """x_tumftm = scale * R @ x_openf1 + t (R may be a reflection when det R = -1)."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return self.scale * np.asarray(xy) @ self.rotation.T + self.translation

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy) - self.translation) @ self.rotation / self.scale

    @property
    def reflected(self) -> bool:
        return bool(np.linalg.det(self.rotation) < 0)

    @property
    def angle_deg(self) -> float:
        r = self.rotation
        return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = False) -> Similarity:
    """Least-squares similarity transform taking src onto dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if not allow_reflection and np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    rot = u @ np.diag(d) @ vt
    var_s = (a**2).sum() / len(src)
    scale = float((sig * d).sum() / var_s)
    return Similarity(scale, rot, mu_d - scale * rot @ mu_s)


def _rotation(angle: float, reflect: bool) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return rot @ np.diag([1.0, -1.0]) if reflect else rot


def icp(points: np.ndarray, line: Centreline, init: Similarity, iters: int = 50,
        trim: float = 0.95, tol: float = 1e-7) -> tuple[Similarity, float]:
    """Trimmed point-to-point ICP; returns (transform, RMS over the kept fraction)."""
    tf = init
    prev = np.inf
    keep = max(3, int(trim * len(points)))
    reflect = init.reflected
    rms = np.inf
    for _ in range(iters):
        moved = tf.apply(points)
        s, dist = line.nearest(moved)
        order = np.argsort(dist)[:keep]
        target = line.point(s[order])
        # Solve for the full transform from the original points, keeping the handedness.
        cand = umeyama(points[order], target, allow_reflection=True)
        if cand.reflected != reflect:
            cand = _fix_handedness(points[order], target, reflect)
        tf = cand
        rms = float(np.sqrt(np.mean(dist[order] ** 2)))
        if abs(prev - rms) < tol * max(rms, 1.0):
            break
        prev = rms
    return tf, rms


def _fix_handedness(src: np.ndarray, dst: np.ndarray, reflect: bool) -> Similarity:
    flip = np.diag([1.0, -1.0])
    if reflect:
        inner = umeyama(src @ flip.T, dst)
        return Similarity(inner.scale, inner.rotation @ flip, inner.translation)
    return umeyama(src, dst)


@dataclass(frozen=True)
class Registration:
    transform: Similarity
    rms_m: float              # over all points, no trimming
    p95_m: float
    max_m: float
    n_points: int
    length_openf1_m: float    # the feed's own lap length, measured in its frame (metres)
    length_tumftm_m: float
    rms_max_m: float = RMS_MAX_M
    length_tol: float = LENGTH_TOL

    @property
    def length_mismatch(self) -> float:
        """Relative lap-length disagreement once the fitted scale is applied."""
        return self.transform.scale * self.length_openf1_m / self.length_tumftm_m - 1.0

    @property
    def accepted(self) -> bool:
        return self.rms_m <= self.rms_max_m and abs(self.length_mismatch) <= self.length_tol

    @property
    def reason(self) -> str:
        why = []
        if self.rms_m > self.rms_max_m:
            why.append(f"RMS {self.rms_m:.2f} m > {self.rms_max_m:g} m")
        if abs(self.length_mismatch) > self.length_tol:
            why.append(f"length mismatch {100 * self.length_mismatch:+.2f}% "
                       f"(limit {100 * self.length_tol:g}%)")
        return "; ".join(why) or "accepted"


def register(points: np.ndarray, line: Centreline, length_openf1: float,
             rotation_step_deg: float = 5.0, allow_reflection: bool = True,
             max_points: int = 6000, coarse_points: int = 500, seed: int = 0) -> Registration:
    """Register OpenF1 positions (metres, feed frame) onto ``line``.

    ``points`` should be on-track samples (flying laps): pit-lane samples are what the
    95% trim is for, but they also inflate the reported (untrimmed) RMS.
    ``length_openf1`` is the feed's lap length in its own frame (``lap_path_length``).
    """
    pts = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    fit = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    coarse = fit[rng.choice(len(fit), min(coarse_points, len(fit)), replace=False)]
    mu_p = fit.mean(axis=0)
    mu_l = line.xy.mean(axis=0)
    scale0 = line.length / length_openf1
    best = (np.inf, None)
    for reflect in (False, True) if allow_reflection else (False,):
        for ang in np.radians(np.arange(0.0, 360.0, rotation_step_deg)):
            rot = _rotation(ang, reflect)
            init = Similarity(scale0, rot, mu_l - scale0 * rot @ mu_p)
            tf, rms = icp(coarse, line, init, iters=3, trim=0.8)
            if rms < best[0]:
                best = (rms, tf)
    tf, _ = icp(fit, line, best[1], iters=60, trim=0.95, tol=1e-6)
    _, dist = line.nearest(tf.apply(pts))
    return Registration(
        transform=tf, rms_m=float(np.sqrt(np.mean(dist**2))), p95_m=float(np.percentile(dist, 95)),
        max_m=float(dist.max()), n_points=len(pts), length_openf1_m=float(length_openf1),
        length_tumftm_m=line.length,
    )


def corridor_excursion(points: np.ndarray, line: Centreline, transform: Similarity,
                       q: float = 99.0) -> tuple[float, float]:
    """(share of points outside the TUMFTM edges, q-th percentile of the excursion in m).

    Excursion is the signed distance beyond the nearer edge (negative = inside the
    track). A racing line stays inside the corridor up to survey error; a section of a
    different layout leaves it by tens of metres.
    """
    xy = transform.apply(points)
    s, dist = line.nearest(xy)
    tan = line.tangent(s)
    rel = xy - line.point(s)
    n = np.sign(tan[:, 0] * rel[:, 1] - tan[:, 1] * rel[:, 0]) * dist
    tr = line.track
    grid = np.r_[tr.s, tr.length] * (line.length / tr.length)
    wl = np.interp(s, grid, np.r_[tr.w_left, tr.w_left[0]])
    wr = np.interp(s, grid, np.r_[tr.w_right, tr.w_right[0]])
    exc = np.maximum(n - wl, -n - wr)
    return float(np.mean(exc > 0)), float(np.percentile(exc, q))


def lap_path_length(t: np.ndarray, xy: np.ndarray, t0: float, duration: float,
                    dt: float = 0.01) -> float:
    """Distance covered between t0 and t0 + duration, in the feed's own frame.

    A cubic spline in time through the samples (pass samples a couple of seconds beyond
    both ends) so 3.7 Hz chords do not cut corners: the sagitta of a 20 m chord on a
    100 m radius is 0.5 m, and it adds up over a lap. An error in t0 shifts both ends
    of the window equally, so it cancels at equal line speeds.
    """
    from scipy.interpolate import CubicSpline

    keep = np.r_[True, np.diff(t) > 1e-3]
    t, xy = t[keep], xy[keep]
    if t[0] > t0 or t[-1] < t0 + duration:
        raise ValueError("samples must cover the whole lap window")
    sx, sy = CubicSpline(t, xy[:, 0]), CubicSpline(t, xy[:, 1])
    tt = np.arange(t0, t0 + duration + 0.5 * dt, dt)
    return float(np.hypot(sx(tt, 1), sy(tt, 1)).sum() * dt)
