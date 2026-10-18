"""Track geometry on a uniform arc-length grid.

A track is a centreline parameterised by arc length s, with heading theta(s), signed
curvature kappa(s) (positive = turning left) and the distances from the centreline to
the left and right edges, measured relative to the driving direction. The vehicle model
lives in Frenet coordinates (s, n) on this centreline, with n positive to the left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep

DEFAULT_DS = 2.0


@dataclass(frozen=True)
class Track:
    name: str
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    kappa: np.ndarray
    w_left: np.ndarray
    w_right: np.ndarray
    closed: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def ds(self) -> float:
        return float(self.s[1] - self.s[0])

    @property
    def length(self) -> float:
        """Closed tracks: the full lap length (the grid does not repeat s=0 at the end)."""
        return float(self.s[-1] + self.ds) if self.closed else float(self.s[-1])

    @property
    def n_points(self) -> int:
        return len(self.s)

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """(left, right) edge polylines, each of shape [N, 2]."""
        nx, ny = -np.sin(self.theta), np.cos(self.theta)
        left = np.stack([self.x + self.w_left * nx, self.y + self.w_left * ny], axis=1)
        right = np.stack([self.x - self.w_right * nx, self.y - self.w_right * ny], axis=1)
        return left, right

    def frenet_singularity_margin(self) -> float:
        """max over s of w_side(s) * |kappa(s)| on the inside of each turn.

        The Frenet map is singular where 1 - n*kappa = 0. Values well below 1 are safe;
        procedural tracks are filtered to <= 0.5.
        """
        inside = np.where(self.kappa > 0, self.w_left, self.w_right)
        return float(np.max(inside * np.abs(self.kappa)))

    def mirrored(self) -> Track:
        """Reflect across the x axis: left turns become right turns."""
        return Track(
            name=f"{self.name}-mirrored", s=self.s.copy(), x=self.x.copy(), y=-self.y,
            theta=-self.theta, kappa=-self.kappa, w_left=self.w_right.copy(),
            w_right=self.w_left.copy(), closed=self.closed, meta=dict(self.meta, mirrored=True),
        )

    def reversed(self) -> Track:
        """Drive the circuit the other way round."""
        if not self.closed:
            raise ValueError("reversing an open strip is not supported")
        x, y = self.x[::-1], self.y[::-1]
        wl, wr = self.w_right[::-1], self.w_left[::-1]
        return from_centreline(
            f"{self.name}-reversed", np.stack([x, y], 1), wl, wr, ds=self.ds, smoothing=0.0,
            meta=dict(self.meta, reversed=True),
        )


def from_centreline(
    name: str,
    xy: np.ndarray,
    w_left: np.ndarray,
    w_right: np.ndarray,
    ds: float = DEFAULT_DS,
    smoothing: float = 0.05,
    closed: bool = True,
    meta: dict | None = None,
) -> Track:
    """Fit a smoothing spline through centreline samples and resample at uniform ds.

    ``smoothing`` is the RMS centreline deviation (m) the spline may absorb; it is what
    keeps curvature from amplifying survey noise. Widths are smoothed more heavily.
    """
    xy = np.asarray(xy, dtype=float)
    wl = np.asarray(w_left, dtype=float)
    wr = np.asarray(w_right, dtype=float)
    if closed and np.linalg.norm(xy[0] - xy[-1]) < 1e-6:
        xy, wl, wr = xy[:-1], wl[:-1], wr[:-1]
    pts = np.vstack([xy, xy[:1]]) if closed else xy
    wls = np.append(wl, wl[0]) if closed else wl
    wrs = np.append(wr, wr[0]) if closed else wr
    chord = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = chord / chord[-1]
    per = 1 if closed else 0
    m = len(u)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u, s=m * smoothing**2, per=per, k=5)
    tck_w, _ = splprep([wls, wrs], u=u, s=m * 0.05**2, per=per, k=3)

    # Arc length of the fitted spline, then uniform resampling in s.
    dense_u = np.linspace(0.0, 1.0, 40 * m + 1)
    dx, dy = splev(dense_u, tck, der=1)
    speed = np.hypot(dx, dy)
    arc = np.r_[0.0, np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(dense_u))]
    total = arc[-1]
    if closed:
        n = int(round(total / ds))
        ds_eff = total / n
        s = np.arange(n) * ds_eff
    else:
        n = int(np.floor(total / ds)) + 1
        s = np.arange(n) * ds
    us = np.interp(s, arc, dense_u)

    x, y = splev(us, tck)
    d1x, d1y = splev(us, tck, der=1)
    d2x, d2y = splev(us, tck, der=2)
    theta = np.unwrap(np.arctan2(d1y, d1x))
    kappa = (d1x * d2y - d1y * d2x) / np.power(d1x**2 + d1y**2, 1.5)
    wl_s, wr_s = splev(us, tck_w)
    return Track(
        name=name, s=s, x=np.asarray(x), y=np.asarray(y), theta=theta, kappa=kappa,
        w_left=np.asarray(wl_s), w_right=np.asarray(wr_s), closed=closed, meta=meta or {},
    )


def load_tumftm(path: str | Path, ds: float = DEFAULT_DS, smoothing: float = 0.05) -> Track:
    """Load a TUMFTM racetrack-database CSV: x_m, y_m, w_tr_right_m, w_tr_left_m."""
    path = Path(path)
    data = np.loadtxt(path, delimiter=",", comments="#")
    return from_centreline(
        path.stem, data[:, :2], w_left=data[:, 3], w_right=data[:, 2], ds=ds,
        smoothing=smoothing, closed=True, meta={"source": "TUMFTM racetrack-database"},
    )


def load_all_tumftm(directory: str | Path = "data/tracks/tumftm", **kw) -> dict[str, Track]:
    return {p.stem: load_tumftm(p, **kw) for p in sorted(Path(directory).glob("*.csv"))}
