"""Segment a track into corners and straights, and summarise its geometry.

A corner is a maximal run where |kappa| exceeds ``k_on`` (radius below 1/k_on), extended
outward while |kappa| stays above ``k_off`` (hysteresis), with runs closer than
``merge_gap`` and of the same sign merged. Everything else is straight. These statistics
fit the procedural generator and measure how F1-like generated tracks are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.tracks.track import Track

# Development circuits: generator fitting, hyperparameters, early stopping.
R_DEV = ["BrandsHatch", "Hockenheim", "MoscowRaceway", "Norisring", "Nuerburgring",
         "Oschersleben", "Sepang", "Sochi", "YasMarina", "Melbourne", "Catalunya", "Zandvoort"]
# Lockbox: evaluated once per final model; never used for fitting or tuning.
R_TEST = ["Austin", "Budapest", "MexicoCity", "Montreal", "Monza", "Sakhir", "SaoPaulo",
          "Shanghai", "Silverstone", "Spa", "Spielberg", "Suzuka"]
EXCLUDED = ["IMS"]  # banked oval


@dataclass
class Corner:
    s_start: float
    s_end: float
    angle: float       # signed turning angle (rad), integral of kappa
    r_min: float       # minimum radius (m)
    length: float

    @property
    def sign(self) -> int:
        return 1 if self.angle > 0 else -1


@dataclass
class Segmentation:
    corners: list[Corner]
    straights: np.ndarray   # lengths (m) between consecutive corners (cyclic for closed)
    length: float

    @property
    def corners_per_km(self) -> float:
        return 1000.0 * len(self.corners) / self.length

    def alternation(self) -> np.ndarray:
        """1 where corner i+1 turns the opposite way to corner i."""
        signs = np.array([c.sign for c in self.corners])
        return (signs[1:] != signs[:-1]).astype(int)


def smooth(k: np.ndarray, ds: float, window_m: float, closed: bool) -> np.ndarray:
    w = max(1, round(window_m / ds))
    kern = np.ones(w) / w
    if closed:
        ext = np.concatenate([k[-w:], k, k[:w]])
        return np.convolve(ext, kern, mode="same")[w:-w]
    return np.convolve(k, kern, mode="same")


def segment(track: Track, k_on: float = 1 / 400, k_off: float = 1 / 800,
            merge_gap: float = 20.0, smooth_m: float = 10.0) -> Segmentation:
    k = smooth(track.kappa, track.ds, smooth_m, track.closed)
    n = len(k)
    ds = track.ds
    above_on = np.abs(k) > k_on
    above_off = np.abs(k) > k_off
    # Rotate closed tracks to start in a straight so no corner wraps around s = 0.
    shift = 0
    if track.closed and above_off.any() and not above_off.all():
        shift = int(np.argmin(np.abs(k)))
        k = np.roll(k, -shift)
        above_on = np.roll(above_on, -shift)
        above_off = np.roll(above_off, -shift)
    runs = []
    i = 0
    while i < n:
        if above_on[i]:
            a = i
            while a > 0 and above_off[a - 1] and np.sign(k[a - 1]) == np.sign(k[i]):
                a -= 1
            b = i
            while b + 1 < n and above_off[b + 1] and np.sign(k[b + 1]) == np.sign(k[i]):
                b += 1
            runs.append([a, b])
            i = b + 1
        else:
            i += 1
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * ds < merge_gap and np.sign(k[a]) == np.sign(k[merged[-1][0]]):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    corners = []
    for a, b in merged:
        seg = k[a:b + 1]
        corners.append(Corner(
            s_start=((a + shift) % n) * ds, s_end=((b + shift) % n) * ds,
            angle=float(seg.sum() * ds), r_min=float(1.0 / max(np.abs(seg).max(), 1e-9)),
            length=float((b - a + 1) * ds)))
    starts = np.array([a for a, _ in merged])
    ends = np.array([b for _, b in merged])
    if len(merged) > 1:
        gaps = (starts[1:] - ends[:-1] - 1) * ds
        if track.closed:
            gaps = np.append(gaps, (n - ends[-1] - 1 + starts[0]) * ds)
    else:
        gaps = np.array([n * ds])
    return Segmentation(corners, np.maximum(gaps, 0.0), track.length)


def geometry_stats(segs: list[Segmentation]) -> dict[str, np.ndarray]:
    """Pooled per-corner and per-straight samples across tracks."""
    return {
        "r_min": np.array([c.r_min for s in segs for c in s.corners]),
        "abs_angle": np.array([abs(c.angle) for s in segs for c in s.corners]),
        "corner_length": np.array([c.length for s in segs for c in s.corners]),
        "straight": np.concatenate([s.straights for s in segs]),
        "corners_per_km": np.array([s.corners_per_km for s in segs]),
        "alternation": np.concatenate([s.alternation() for s in segs]),
    }
