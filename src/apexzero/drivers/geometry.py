"""A closed centreline as a dense polyline: projection, arc length and lap distance.

Everything in the driver study lives on one coordinate, the arc length s along the TUMFTM
centreline. The public position feed is snapped to a track model (no lateral line, see
docs/findings/snapping-audit.md), so projecting a sample onto the centreline keeps all the
information it has. Projection is point-to-segment on a ~0.5 m polyline, so s is exact to
millimetres rather than quantised to the vertex spacing.

Two projections are provided. ``nearest`` is global (a KD-tree), which is ambiguous where
the circuit crosses itself (Suzuka) or runs back alongside itself; ``project_window``
searches only near a predicted s, which is what a moving car needs.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from apexzero.tracks.track import Track


class Centreline:
    def __init__(self, track: Track, step: float = 0.5) -> None:
        if not track.closed:
            raise ValueError("the driver study needs a closed circuit")
        n_sub = max(1, int(np.ceil(track.ds / step)))
        s_track = np.arange(track.n_points * n_sub) * (track.length / (track.n_points * n_sub))
        ext_s = np.r_[track.s, track.length]
        self.name = track.name
        self.xy = np.stack([np.interp(s_track, ext_s, np.r_[track.x, track.x[0]]),
                            np.interp(s_track, ext_s, np.r_[track.y, track.y[0]])], axis=1)
        # Arc length of the polyline itself, so projections and lengths agree exactly.
        seg = np.roll(self.xy, -1, axis=0) - self.xy
        seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.r_[0.0, np.cumsum(seg_len)[:-1]]
        self.length = float(seg_len.sum())
        self._seg = seg
        self._seg_len = seg_len
        self._tree = cKDTree(self.xy)
        self.track = track

    @property
    def n(self) -> int:
        return len(self.s)

    def _refine(self, xy: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact projection onto the two segments either side of vertex ``idx``."""
        best_d = np.full(len(xy), np.inf)
        best_s = np.zeros(len(xy))
        for j in ((idx - 1) % self.n, idx):
            a = self.xy[j]
            e = self._seg[j]
            u = np.clip(np.einsum("ij,ij->i", xy - a, e) / self._seg_len[j] ** 2, 0.0, 1.0)
            d = np.linalg.norm(a + u[:, None] * e - xy, axis=1)
            better = d < best_d
            best_d = np.where(better, d, best_d)
            best_s = np.where(better, self.s[j] + u * self._seg_len[j], best_s)
        return best_s % self.length, best_d

    def nearest(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global projection: (s, distance) for each point."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        _, idx = self._tree.query(xy)
        return self._refine(xy, idx)

    def project_window(self, xy: np.ndarray, s_guess: np.ndarray,
                       half_window: float) -> tuple[np.ndarray, np.ndarray]:
        """Project each point onto the part of the line within ``half_window`` of s_guess."""
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        s_guess = np.broadcast_to(np.asarray(s_guess, dtype=float), (len(xy),))
        step = self.length / self.n
        k = int(np.ceil(half_window / step))
        centre = np.round((s_guess % self.length) / step).astype(int)
        cand = (centre[:, None] + np.arange(-k, k + 1)[None, :]) % self.n
        d2 = ((self.xy[cand] - xy[:, None, :]) ** 2).sum(axis=2)
        idx = cand[np.arange(len(xy)), d2.argmin(axis=1)]
        return self._refine(xy, idx)

    def point(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float) % self.length
        ext = np.r_[self.s, self.length]
        return np.stack([np.interp(s, ext, np.r_[self.xy[:, 0], self.xy[0, 0]]),
                         np.interp(s, ext, np.r_[self.xy[:, 1], self.xy[0, 1]])], axis=-1)

    def tangent(self, s: np.ndarray) -> np.ndarray:
        """Unit tangent at s (of the segment containing s)."""
        s = np.asarray(s, dtype=float) % self.length
        j = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, self.n - 1)
        return self._seg[j] / self._seg_len[j][..., None]

    def curvature(self, s: np.ndarray) -> np.ndarray:
        """Signed curvature of the underlying spline track, interpolated at s."""
        tr = self.track
        scale = tr.length / self.length
        ext = np.r_[tr.s, tr.length]
        return np.interp((np.asarray(s) * scale) % tr.length, ext, np.r_[tr.kappa, tr.kappa[0]])


def wrap(ds: np.ndarray, length: float) -> np.ndarray:
    """Signed distance difference folded into [-L/2, L/2)."""
    return (np.asarray(ds) + 0.5 * length) % length - 0.5 * length
