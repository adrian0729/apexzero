"""Corner complexes that partition a lap, and the time each one costs.

A per-corner "receipt" only adds up if the pieces tile the lap. ``tracks.corners.segment``
finds corners (curvature runs) but leaves straights between them, so a braking zone would
fall outside every corner. Here each corner complex owns half of the straight before it
and half of the straight after it (the split is at the straight's midpoint), so its
sector holds the braking zone, turn-in, apex and the start of the traction exit. Sector
times then sum exactly to the lap delta.

What a sector *measures* is not what it *costs*. A slow exit keeps costing time on the
straight after the corner, well past the midpoint, so the next sector's clock includes a
speed deficit carried in from upstream: in the Hockenheim demo both cars are flat out
through all of C2 (a 253 m-radius bend), the agent enters it 33 km/h slower after braking
too hard for C1, and all of C2's measured +0.27 s is C1's. :func:`carry_over_split` charges
time accrued while *both* cars are flat out (nobody is deciding anything) to the sector of
the last point where either car was not, which gives a per-corner cost that still sums to
the lap delta. :func:`sector_speeds` gives the entry, exit and minimum speeds to read it by.

Corners closer than ``merge_gap`` are one complex (a chicane is one decision, not two).
Names are ``C1, C2, ...`` in lap order (``C3-C4`` for a merged complex). They are this
segmentation's indices, deliberately not the circuit's official turn numbers, which
count kinks and complexes differently. ``r_min`` is the minimum radius of the *reference
line*, which a car may straighten (a kink with a curvature spike), so speeds are the
better guide to how slow a corner is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLAT_THROTTLE = 0.99   # throttle (pedal state) at or above this counts as flat out

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


def sector_of(x: np.ndarray, sectors: list[Sector], length: float) -> np.ndarray:
    """Index of the sector that holds each station ``x`` in [0, length) (-1 if none)."""
    x = np.asarray(x, float)
    out = np.full(x.shape, -1, int)
    for i, sec in enumerate(sectors):
        rel = np.mod(x - sec.s_start, length)
        out[(out < 0) & (rel < sec.s_end - sec.s_start)] = i
    return out


@dataclass
class CarryOver:
    """A per-sector split of Δt (all arrays [n_sectors], seconds, > 0: time lost)."""

    own: np.ndarray          # accrued in the sector, on stretches that began in it
    carried_in: np.ndarray   # accrued in the sector on flat-out stretches that began upstream
    carried_out: np.ndarray  # accrued downstream on flat-out stretches that began here

    @property
    def measured(self) -> np.ndarray:
        """What the clock shows inside each sector (equals :func:`sector_deltas`)."""
        return self.own + self.carried_in

    @property
    def charged(self) -> np.ndarray:
        """What each corner costs: its own time plus its exit's cost downstream."""
        return self.own + self.carried_out


def carry_over_split(s: np.ndarray, delta_t: np.ndarray, flat: np.ndarray,
                     sectors: list[Sector], length: float, closed: bool = True) -> CarryOver:
    """Charge time accrued while both cars are flat out to the corner that caused it.

    ``s`` spans [0, length] (increasing), ``delta_t`` is Δt(s) on it and ``flat[i]`` is
    True where *both* cars are flat out at ``s[i]``. Where both are flat out, neither is
    making a decision, so any time gained or lost there comes from the state they carried
    in (mostly a speed deficit from a slower exit). Each maximal flat-out stretch is
    charged to the sector of the last interval before it, where at least one car was not
    flat out; on a closed lap a stretch that starts at the line continues the one that
    ends the lap. Everything else stays where it was measured. Sector boundaries are put
    on the grid first, so ``measured`` equals :func:`sector_deltas` and both ``measured``
    and ``charged`` sum to Δt(L) - Δt(0).
    """
    s = np.asarray(s, float)
    d = np.asarray(delta_t, float)
    f = np.asarray(flat, bool)
    if not (len(s) == len(d) == len(f)):
        raise ValueError("s, delta_t and flat must have the same length")
    cuts = np.mod([sec.s_start for sec in sectors], length)
    grid = np.union1d(s, cuts[(cuts > s[0]) & (cuts < s[-1])])
    dg = np.interp(grid, s, d)
    fg = np.interp(grid, s, f.astype(float)) > 1.0 - 1e-9     # flat at both neighbours
    inc = np.diff(dg)
    home = sector_of(0.5 * (grid[1:] + grid[:-1]), sectors, length)
    if np.any(home < 0):
        raise ValueError("the sectors do not tile the lap")
    flat_iv = fg[:-1] & fg[1:]
    m = len(inc)

    runs = []                                   # maximal runs of flat intervals [a, b]
    j = 0
    while j < m:
        if flat_iv[j]:
            k = j
            while k + 1 < m and flat_iv[k + 1]:
                k += 1
            runs.append((j, k))
            j = k + 1
        else:
            j += 1
    owner = home.copy()
    wraps = closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == m - 1
    for a, b in runs:
        if a == 0 and b == m - 1:
            continue                            # flat out all lap: nothing to move
        if a > 0:
            owner[a:b + 1] = home[a - 1]
        elif wraps:                             # the last run starts after the first
            owner[a:b + 1] = home[runs[-1][0] - 1]
    n = len(sectors)
    own = np.zeros(n)
    carried_in = np.zeros(n)
    carried_out = np.zeros(n)
    stay = owner == home
    np.add.at(own, home[stay], inc[stay])
    np.add.at(carried_in, home[~stay], inc[~stay])
    np.add.at(carried_out, owner[~stay], inc[~stay])
    return CarryOver(own, carried_in, carried_out)


def sector_speeds(s: np.ndarray, v: np.ndarray, sectors: list[Sector], length: float,
                  closed: bool = True) -> dict[str, np.ndarray]:
    """Speed at each sector's entry (``s_start``) and exit (``s_end``) and the minimum in
    between, from one ghost's samples (``s`` in [0, length], non-decreasing)."""
    s = np.asarray(s, float)
    v = np.asarray(v, float)

    def at(x):          # x in [0, 2L): a sector's end may lie past the line
        return float(np.interp(x - length if closed and x > length else x, s, v))

    entry, exit_, vmin = [], [], []
    for sec in sectors:
        span = sec.s_end - sec.s_start
        rel = np.mod(s - sec.s_start, length) if closed else s - sec.s_start
        inside = v[(rel >= 0) & (rel <= span)]
        a, b = at(sec.s_start), at(sec.s_end)
        entry.append(a)
        exit_.append(b)
        vmin.append(min(a, b, *inside.tolist()))
    return {"entry": np.array(entry), "exit": np.array(exit_), "min": np.array(vmin)}
