"""Corner complexes that partition a lap, and the time each one costs.

A per-corner "receipt" only adds up if the pieces tile the lap. ``tracks.corners.segment``
finds corners (curvature runs) but leaves straights between them, so a braking zone would
fall outside every corner. Here each corner complex owns half of the straight before it
and half of the straight after it (the split is at the straight's midpoint), so its
sector holds the whole event: braking, turn-in, apex and the traction exit. Sector times
then sum exactly to the lap delta.

Corners closer than ``merge_gap`` are one complex (a chicane is one decision, not two).
Names are ``C1, C2, ...`` in lap order (``C3-C4`` for a merged complex). They are this
segmentation's indices, deliberately not the circuit's official turn numbers, which
count kinks and complexes differently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.tracks.corners import segment
from apexzero.tracks.track import Track


@dataclass
class Sector:
    name: str
    s_start: float       # in [0, L)
    s_end: float         # in (s_start, s_start + L]; above L for the sector that wraps s = 0
    s_apex: float        # station of the complex's minimum radius, in [0, L)
    r_min: float         # m
    angle: float         # signed turning angle (rad)
    corners: list[int]   # indices of the merged corners, in lap order
    s_entry: float = 0.0  # where the complex's curvature starts (inside the sector)
    s_exit: float = 0.0   # where it ends, unwrapped like s_end

    def to_dict(self) -> dict:
        return {"name": self.name, "s_start": self.s_start, "s_end": self.s_end,
                "s_entry": self.s_entry, "s_exit": self.s_exit, "s_apex": self.s_apex,
                "r_min": self.r_min if np.isfinite(self.r_min) else None,
                "angle": self.angle, "corners": list(self.corners)}


def corner_sectors(track: Track, merge_gap: float = 40.0, **segment_kw) -> list[Sector]:
    """Corner-complex sectors that tile [0, L) (closed) or [0, L] (open strips)."""
    L = float(track.length)
    seg = segment(track, **segment_kw)
    if not seg.corners:
        return [Sector("lap", 0.0, L, 0.0, float("inf"), 0.0, [], 0.0, L)]

    # Lap order, with each corner's end unwrapped past its start.
    spans = []
    for c in seg.corners:
        a, b = c.s_start, c.s_end
        if b < a:
            b += L
        spans.append((a, b, c))
    spans.sort(key=lambda x: x[0])

    # Apex: the station of peak |kappa| inside the tightest corner of the complex.
    k = np.abs(track.kappa)

    def apex(a: float, b: float) -> float:
        idx = (np.arange(round(a / track.ds), round(b / track.ds) + 1)) % track.n_points
        return float(track.s[idx[np.argmax(k[idx])]])

    # Merge neighbours separated by less than merge_gap into complexes.
    groups: list[list[int]] = [[0]]
    for i in range(1, len(spans)):
        if spans[i][0] - spans[groups[-1][-1]][1] < merge_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    if track.closed and len(groups) > 1:
        gap_wrap = spans[groups[0][0]][0] + L - spans[groups[-1][-1]][1]
        if gap_wrap < merge_gap:
            last = groups.pop()
            groups[0] = last + groups[0]

    starts = []
    ends = []
    for g in groups:
        a = spans[g[0]][0]
        b = spans[g[-1]][1]
        if b < a:      # a complex merged across the wrap
            b += L
        starts.append(a)
        ends.append(b)

    m = len(groups)
    if track.closed:
        # Boundary after complex i: the midpoint of the straight to the next complex, in
        # coordinates unwrapped forward from the complex's end (so it may exceed L).
        bounds = []
        for i in range(m):
            nxt = starts[(i + 1) % m]
            while nxt < ends[i]:
                nxt += L
            bounds.append(0.5 * (ends[i] + nxt))
        lo, hi = [], []
        for i in range(m):
            a = bounds[i - 1] % L
            span = (bounds[i] - bounds[i - 1]) % L
            if m == 1 or span < 1e-9:
                span = L
            lo.append(a)
            hi.append(a + span)
    else:
        mids = [0.5 * (ends[i] + starts[i + 1]) for i in range(m - 1)]
        lo = [0.0] + mids
        hi = mids + [L]

    sectors = []
    for i, g in enumerate(groups):
        a, b = lo[i], hi[i]
        entry = starts[i]
        while track.closed and entry < a - 1e-9:
            entry += L
        names = [f"C{j + 1}" for j in g]
        corners = [spans[j][2] for j in g]
        rmin_idx = int(np.argmin([c.r_min for c in corners]))
        ap = apex(spans[g[rmin_idx]][0], spans[g[rmin_idx]][1])
        sectors.append(Sector(
            name=names[0] if len(names) == 1 else f"{names[0]}-{names[-1]}",
            s_start=float(a), s_end=float(b), s_apex=ap,
            r_min=float(min(c.r_min for c in corners)),
            angle=float(sum(c.angle for c in corners)), corners=list(g),
            s_entry=float(entry), s_exit=float(entry + ends[i] - starts[i])))
    return sectors


def sector_deltas(s: np.ndarray, delta_t: np.ndarray, sectors: list[Sector],
                  length: float) -> np.ndarray:
    """Time gained (< 0) or lost (> 0) in each sector, from a cumulative delta Δt(s).

    ``s`` must span [0, length]. A sector that wraps past the line is split into its two
    pieces on the same lap, so the entries sum to Δt(L) - Δt(0) exactly.
    """
    s = np.asarray(s, float)
    d = np.asarray(delta_t, float)

    def D(x):
        return float(np.interp(x, s, d))

    out = []
    for sec in sectors:
        if sec.s_end <= length + 1e-9:
            out.append(D(min(sec.s_end, length)) - D(sec.s_start))
        else:
            out.append(D(length) - D(sec.s_start) + D(sec.s_end - length) - D(0.0))
    return np.array(out)
