"""Additive segmentation of a lap into corner complexes and phases.

A lap is cut into corner complexes (the plan asks for 12-20) at the midpoints of the
straights between them, so every metre of the lap belongs to exactly one complex and
complex times sum exactly to the lap time. Corners come from the centreline curvature
(``apexzero.tracks.corners``). Corners less than ``hard_gap`` apart always merge (a
chicane is one complex: at 3.7 Hz, one sample every 6-25 m, its halves cannot be timed
separately); up to ``merge_gap`` they merge while more than ``n_min`` complexes remain;
then the closest pairs merge until at most ``n_max`` remain. A circuit with fewer natural
complexes than ``n_min`` keeps what it has (Monza has about ten): splitting a chicane to
reach a quota creates phases a few samples long, which is noise, not resolution.

Each complex is split into three phases at reference positions taken from the session's
pooled speed profile, not from any driver's lap:

- entry: from the cut to where the reference speed first falls within ``apex_frac`` of
  the complex's speed range above its minimum (approach, braking and turn-in);
- apex: the low-speed part around the minimum;
- exit: from there to the next cut (traction and the first half of the straight).

Every phase is at least ``min_phase`` long.

Corner type comes from the observed envelope apex speed: slow < 120 km/h,
medium 120-200 km/h, fast > 200 km/h.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.drivers.geometry import Centreline
from apexzero.tracks.corners import segment as curvature_segments

PHASES = ("entry", "apex", "exit")
TYPE_EDGES_KMH = (120.0, 200.0)


def corner_type(v_min_ms: float) -> str:
    kmh = 3.6 * v_min_ms
    if kmh < TYPE_EDGES_KMH[0]:
        return "slow"
    return "medium" if kmh < TYPE_EDGES_KMH[1] else "fast"


@dataclass(frozen=True)
class Complex:
    index: int
    x_start: float          # lap distance of the cut before it
    x_end: float            # lap distance of the cut after it
    x_entry_end: float      # entry -> apex
    x_apex_end: float       # apex -> exit
    x_min: float            # reference minimum-speed point
    v_min: float            # reference minimum speed (m/s)
    kind: str               # slow / medium / fast
    corners: tuple[tuple[float, float], ...]   # curvature runs inside, lap distance

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_start, self.x_entry_end, self.x_apex_end, self.x_end)

    @property
    def name(self) -> str:
        return f"C{self.index + 1:02d}"


@dataclass(frozen=True)
class Segmentation:
    length: float
    complexes: tuple[Complex, ...]

    @property
    def cuts(self) -> np.ndarray:
        """All phase boundaries in lap distance, from 0 to L (3 phases per complex)."""
        b = [c.bounds[:3] for c in self.complexes]
        return np.r_[np.concatenate(b), self.length]

    def labels(self) -> list[tuple[str, str, str]]:
        """(complex name, phase, corner type) for each phase interval of ``cuts``."""
        return [(c.name, p, c.kind) for c in self.complexes for p in PHASES]

    def phase_times(self, elapsed: np.ndarray) -> np.ndarray:
        """Per-phase times from elapsed times evaluated at ``cuts`` (sums to the lap)."""
        return np.diff(elapsed)

    def corner_windows(self, before: float = 150.0, after: float = 100.0) -> list[tuple]:
        """Where traffic matters: from ``before`` m ahead of the apex phase to ``after`` m
        beyond it, clipped to the complex."""
        return [(max(c.x_start, c.x_entry_end - before), min(c.x_end, c.x_apex_end + after))
                for c in self.complexes]


def _corner_runs(line: Centreline, s_line: float, **kw) -> list[tuple[float, float]]:
    """Curvature corners as (start, end) in lap distance, sorted, non-wrapping."""
    seg = curvature_segments(line.track, **kw)
    scale = line.length / line.track.length
    runs = []
    for c in seg.corners:
        a = ((c.s_start * scale) - s_line) % line.length
        b = a + c.length * scale
        runs.append((a, b))
    runs.sort()
    # A corner straddling the timing line is split; keep the larger piece's start.
    out = []
    for a, b in runs:
        if b > line.length:
            if line.length - a >= b - line.length:
                out.append((a, line.length - 1e-6))
            else:
                out.insert(0, (0.0, b - line.length))
        else:
            out.append((a, b))
    return sorted(out)


def build(line: Centreline, s_line: float, v_ref_x: np.ndarray, v_ref: np.ndarray,
          n_min: int = 12, n_max: int = 20, hard_gap: float = 80.0, merge_gap: float = 150.0,
          apex_frac: float = 0.15, min_phase: float = 30.0,
          edge_gap: float = 150.0, **corner_kw) -> Segmentation:
    """Segment one circuit, given the timing-line position and a reference v(x).

    ``v_ref`` is a pooled reference speed (centreline rate) on the lap-distance grid
    ``v_ref_x``. Cuts within ``edge_gap`` of the timing line snap to it, so no complex is
    a sliver at the start or end of the lap. Corners closer than ``hard_gap`` always
    merge; closer than ``merge_gap`` they merge while more than ``n_min`` complexes remain.
    """
    length = line.length
    groups = [[r] for r in _corner_runs(line, s_line, **corner_kw)]
    if not groups:
        raise ValueError("no corners found")

    def gap(i: int) -> float:
        return groups[i + 1][0][0] - groups[i][-1][1]

    while len(groups) > 1:
        gaps = np.array([gap(i) for i in range(len(groups) - 1)])
        i = int(np.argmin(gaps))
        if (len(groups) > n_max or gaps[i] < hard_gap
                or (gaps[i] < merge_gap and len(groups) > n_min)):
            groups[i] = groups[i] + groups.pop(i + 1)
        else:
            break
    # Cuts at straight midpoints; the timing line is always a cut.
    mids = [0.5 * (groups[i][-1][1] + groups[i + 1][0][0]) for i in range(len(groups) - 1)]
    cuts = [0.0] + [m for m in mids if edge_gap < m < length - edge_gap] + [length]
    complexes = []
    for j in range(len(cuts) - 1):
        a, b = cuts[j], cuts[j + 1]
        runs = tuple(r for g in groups for r in g if a <= 0.5 * (r[0] + r[1]) < b)
        m = (v_ref_x >= a) & (v_ref_x < b)
        xs, vs = v_ref_x[m], v_ref[m]
        if runs:
            lo, hi = runs[0][0], runs[-1][1]
            inside = (xs >= lo) & (xs <= hi)
            if inside.any():
                xs_c, vs_c = xs[inside], vs[inside]
            else:
                xs_c, vs_c = xs, vs
        else:
            xs_c, vs_c = xs, vs
        k = int(np.argmin(vs_c))
        x_min, v_min = float(xs_c[k]), float(vs_c[k])
        # Apex phase: the contiguous stretch around the minimum within the band.
        band = vs <= v_min + apex_frac * (vs.max() - v_min)
        i0 = int(np.searchsorted(xs, x_min))
        i0 = min(i0, len(xs) - 1)
        lo_i = i0
        while lo_i > 0 and band[lo_i - 1]:
            lo_i -= 1
        hi_i = i0
        while hi_i < len(xs) - 1 and band[hi_i + 1]:
            hi_i += 1
        e_end, a_end = float(xs[lo_i]), float(xs[hi_i])
        if a_end - e_end < min_phase:
            pad = 0.5 * (min_phase - (a_end - e_end))
            e_end, a_end = e_end - pad, a_end + pad
        e_end = float(np.clip(e_end, a + min_phase, b - 2 * min_phase))
        a_end = float(np.clip(a_end, e_end + min_phase, b - min_phase))
        complexes.append(Complex(j, a, b, e_end, a_end, x_min, v_min, corner_type(v_min), runs))
    return Segmentation(length, tuple(complexes))
