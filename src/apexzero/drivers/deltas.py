"""Per-phase lap tables, the R1 envelope, teammate deltas and the reliability checks.

Every number here is a time in the distance domain: how long a lap spent between two
fixed lap distances. Because phases tile the lap, a teammate's phase deltas add up
exactly to the lap-time delta of the same laps; ``pair_delta`` asserts this.

- **R1** (plan section 6) is the empirical envelope: per car-session, the best time in
  each phase across both teammates' laps. No model is involved. Teammate deltas do not
  depend on it (the reference cancels); the per-lap residual y = t - R1
  (``r1_residuals``) is the input to the later hierarchical model.
- **Teammate delta** for a car-session: inside each Q segment both drivers have clean
  laps in, the mean phase times of driver B's top-n laps *in that segment* minus driver
  A's; the per-segment deltas are then averaged with inverse-variance weights
  1 / (1/n_A + 1/n_B). A lap is only ever compared with laps of the same Q segment, so
  track evolution and tyre state between segments cannot enter the delta.
- **Placebo**: shuffle the driver labels among the car-session's laps within each Q
  segment. What survives with the true labels is driver, setup *or* any per-car
  measurement offset; the shuffle cannot tell those apart.
- **Split-half**: the same delta from two disjoint halves of each driver's laps, split
  inside each Q segment; Spearman rho across cells is the reliability, with a
  permutation p-value that flips whole pairs (cells of one pair are not independent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr


@dataclass
class LapTable:
    session: str
    keys: list[tuple[int, int]]       # (driver, lap)
    driver: np.ndarray                # [n]
    team: list[str]                   # [n]
    q_segment: np.ndarray             # [n]
    lap_time: np.ndarray              # [n] official
    phase_t: np.ndarray               # [n, P] seconds, rows sum to lap_time
    labels: list[tuple[str, str, str]]   # [P] (complex, phase, corner type)
    acronym: dict[int, str]
    new_set: np.ndarray | None = None    # [n] first flying lap on a fresh set (covariate)

    def rows(self, mask: np.ndarray) -> LapTable:
        idx = np.flatnonzero(mask)
        return LapTable(self.session, [self.keys[i] for i in idx], self.driver[idx],
                        [self.team[i] for i in idx], self.q_segment[idx], self.lap_time[idx],
                        self.phase_t[idx], self.labels, self.acronym,
                        None if self.new_set is None else self.new_set[idx])

    def team_mask(self, team: str) -> np.ndarray:
        return np.array([t == team for t in self.team], dtype=bool)


def r1_envelope(tab: LapTable) -> dict[str, np.ndarray]:
    """Best phase time per car-session (team), across both teammates' laps."""
    out = {}
    for team in sorted(set(tab.team)):
        out[team] = tab.phase_t[tab.team_mask(team)].min(axis=0)
    return out


def r1_residuals(tab: LapTable) -> np.ndarray:
    """[n, P] y = t - R1 for every lap, against its own car-session's envelope (s)."""
    r1 = r1_envelope(tab)
    return tab.phase_t - np.stack([r1[t] for t in tab.team]) if tab.keys else tab.phase_t


@dataclass
class PairDelta:
    session: str
    team: str
    a: int                      # reference driver (number)
    b: int
    n_a: int                    # laps used, over all Q segments
    n_b: int
    q_segments: tuple[int, ...]  # Q segments that contributed
    delta: np.ndarray           # [P] seconds, weighted mean over segments of mean(B) - mean(A)
    lap_delta: float            # the same for the lap time
    segments: tuple[tuple[int, int, int], ...] = ()   # (Q segment, n_A, n_B)
    var_factor: float = 1.0     # var(delta) relative to 2 laps each in one segment, for
    #                             independent laps: 1 / sum_q 1/(1/n_A + 1/n_B). 1 for 2 + 2
    #                             laps, 2 for 1 + 1, 0.5 for 2 + 2 in each of two segments

    def label(self, acr: dict[int, str]) -> str:
        return f"{acr.get(self.b, self.b)} - {acr.get(self.a, self.a)}"


def _select(tab: LapTable, idx: np.ndarray, top_n: int | None) -> np.ndarray:
    if top_n is None or len(idx) <= top_n:
        return idx
    return idx[np.argsort(tab.lap_time[idx], kind="stable")[:top_n]]


def pair_delta(tab: LapTable, team: str, top_n: int | None = 2,
               drivers: np.ndarray | None = None) -> PairDelta | None:
    """Teammate delta for one car-session, matched within Q segments (module docstring).

    ``top_n`` laps per driver *per Q segment*; ``drivers`` overrides the labels (placebo).
    """
    drv = tab.driver if drivers is None else drivers
    m_team = tab.team_mask(team)
    pair = sorted(set(drv[m_team].tolist()))
    if len(pair) != 2:
        return None
    a, b = pair
    parts = []
    for q in sorted(set(tab.q_segment[m_team].tolist())):
        in_q = m_team & (tab.q_segment == q)
        ia = _select(tab, np.flatnonzero(in_q & (drv == a)), top_n)
        ib = _select(tab, np.flatnonzero(in_q & (drv == b)), top_n)
        if len(ia) and len(ib):
            parts.append((int(q), ia, ib))
    if not parts:
        return None
    w = np.array([1.0 / (1.0 / len(ia) + 1.0 / len(ib)) for _, ia, ib in parts])
    wn = w / w.sum()
    delta = np.zeros(tab.phase_t.shape[1])
    lap_delta = 0.0
    for wq, (_, ia, ib) in zip(wn, parts, strict=True):
        delta += wq * (tab.phase_t[ib].mean(axis=0) - tab.phase_t[ia].mean(axis=0))
        lap_delta += wq * float(tab.lap_time[ib].mean() - tab.lap_time[ia].mean())
    # Additivity: the phases tile the lap, so the deltas must sum to the lap delta.
    assert abs(delta.sum() - lap_delta) < 1e-6, (delta.sum(), lap_delta)
    return PairDelta(tab.session, team, int(a), int(b), sum(len(p[1]) for p in parts),
                     sum(len(p[2]) for p in parts), tuple(p[0] for p in parts), delta,
                     lap_delta, tuple((q, len(ia), len(ib)) for q, ia, ib in parts),
                     float(1.0 / w.sum()))


def teammate_deltas(tab: LapTable, top_n: int | None = 2) -> list[PairDelta]:
    out = []
    for team in sorted(set(tab.team)):
        d = pair_delta(tab, team, top_n)
        if d is not None:
            out.append(d)
    return out


def aggregate(delta: np.ndarray, labels: list[tuple[str, str, str]], by: str) -> dict:
    """Sum phase deltas into complexes (by='complex') or corner type x phase (by='type')."""
    out: dict = {}
    for d, (cx, ph, kind) in zip(delta, labels, strict=True):
        key = cx if by == "complex" else (kind, ph)
        out[key] = out.get(key, 0.0) + float(d)
    return out


def shuffle_within(rng: np.random.Generator, labels: np.ndarray,
                   strata: np.ndarray) -> np.ndarray:
    """Permute ``labels`` separately inside each stratum (lap counts per stratum kept)."""
    out = labels.copy()
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        out[idx] = labels[idx][rng.permutation(len(idx))]
    return out


@dataclass
class PlaceboResult:
    real: np.ndarray              # [cells] true-label deltas, (pair, phase) flattened
    null: np.ndarray              # [shuffles, cells] the same under shuffled labels
    cells: list[tuple[str, str]]  # (corner type, phase) of each cell
    var_factor: np.ndarray        # [cells] its pair's PairDelta.var_factor
    pair: np.ndarray              # [cells] pair index
    ss_p_value: float             # permutation p of sum(delta^2) over all cells
    n_pairs: int
    pairs: list = field(default_factory=list)   # the true-label PairDelta per pair

    @property
    def n_cells(self) -> int:
        return len(self.real)

    @property
    def n_shuffles(self) -> int:
        return self.null.shape[0]

    @property
    def real_abs_mean(self) -> float:
        return float(np.abs(self.real).mean())

    @property
    def placebo_abs_mean(self) -> float:
        return float(np.abs(self.null).mean())

    @property
    def cell_null_sd(self) -> np.ndarray:
        return self.null.std(axis=0)

    def exceed_share(self, limits: dict) -> tuple[float, float, int]:
        """(shuffled share, real share, cells) of cells whose |delta| exceeds ``limits``
        [(type, phase)] scaled to the pair's lap counts (x sqrt(var_factor)).

        The limits come from synthetic data at 2 laps per driver in one segment. On
        shuffled real data about 5% should exceed a 2-sigma limit if the synthetic budget
        is as wide as the real null spread: this is a check that can fail.
        """
        lim = np.array([limits.get(c, np.nan) for c in self.cells]) * np.sqrt(self.var_factor)
        ok = np.isfinite(lim)
        if not ok.any():
            return float("nan"), float("nan"), 0
        return (float(np.mean(np.abs(self.null[:, ok]) > lim[ok])),
                float(np.mean(np.abs(self.real[ok]) > lim[ok])), int(ok.sum()))

    def abs_ratio_by_cell(self) -> dict[tuple[str, str], tuple[float, int]]:
        """(type, phase) -> (mean |real delta| / mean |shuffled delta|, cells)."""
        out = {}
        for c in sorted(set(self.cells)):
            m = np.array([x == c for x in self.cells])
            out[c] = (float(np.abs(self.real[m]).mean() / np.abs(self.null[:, m]).mean()),
                      int(m.sum()))
        return out


def placebo(tabs: list[LapTable], top_n: int | None = 2, n_shuffles: int = 200,
            seed: int = 0) -> PlaceboResult:
    """Shuffle driver labels within each (car-session, Q segment), lap counts preserved.

    ``ss_p_value`` compares the sum of squared deltas over every cell under the true
    labels with the same sum under shuffled labels (one shuffle per car-session and
    segment, drawn independently). Under the true labels any per-car measurement offset
    counts as an effect too, so a small p says the labels carry structure, not that it
    is the driver's. A per-cell test is not offered: with two laps per driver there are
    only three distinct splits, so no single cell can reach p < 0.05. The signed mean of
    shuffled deltas is not reported: it is about zero by symmetry for any pipeline.
    """
    rng = np.random.default_rng(seed)
    real, null, cells, var, pair, pairs = [], [], [], [], [], []
    for tab in tabs:
        for team in sorted(set(tab.team)):
            d = pair_delta(tab, team, top_n)
            if d is None:
                continue
            sub = tab.rows(tab.team_mask(team) & np.isin(tab.q_segment, d.q_segments))
            sims = []
            for _ in range(n_shuffles):
                p = pair_delta(sub, team, top_n,
                               drivers=shuffle_within(rng, sub.driver, sub.q_segment))
                if p is not None:
                    sims.append(p.delta)
            if not sims:
                continue
            real.append(d.delta)
            null.append(np.array(sims))
            cells.extend((kind, ph) for _, ph, kind in tab.labels)
            var.append(np.full(len(d.delta), d.var_factor))
            pair.append(np.full(len(d.delta), len(pairs)))
            pairs.append(d)
    if not real:
        raise ValueError("no teammate pairs")
    real_arr = np.concatenate(real)
    s_min = min(len(n) for n in null)
    null_arr = np.concatenate([n[:s_min] for n in null], axis=1)  # [shuffles, cells]
    ss_null = (null_arr**2).sum(axis=1)
    ss_real = float((real_arr**2).sum())
    return PlaceboResult(
        real=real_arr, null=null_arr, cells=cells, var_factor=np.concatenate(var),
        pair=np.concatenate(pair),
        ss_p_value=float((1 + np.sum(ss_null >= ss_real)) / (1 + len(ss_null))),
        n_pairs=len(pairs), pairs=pairs,
    )


@dataclass
class SplitHalf:
    rho: float
    p_value: float      # one-sided permutation p, flipping whole pairs (see _rho_test)
    n_cells: int
    n_pairs: int
    level: str
    p_naive: float = float("nan")   # scipy's p, treating every cell as independent


def _rho_test(xs: list[np.ndarray], ys: list[np.ndarray], level: str, n_perm: int = 2000,
              seed: int = 0) -> SplitHalf:
    """Spearman rho of the concatenated halves, with a pair-level sign-flip permutation.

    Under no driver effect, swapping the labels of one pair's second half flips the sign
    of its whole second-half delta vector and leaves its first half alone, so flipping
    pairs at random keeps every within-pair dependence (shared laps, session-level
    errors) that makes the cells non-independent.
    """
    n_cells = int(sum(len(x) for x in xs))
    if n_cells < 3:
        return SplitHalf(float("nan"), float("nan"), n_cells, len(xs), level)
    x, y = np.concatenate(xs), np.concatenate(ys)
    r = spearmanr(x, y)
    rho = float(r.statistic)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        sg = rng.choice((-1.0, 1.0), len(ys))
        yp = np.concatenate([s * yi for s, yi in zip(sg, ys, strict=True)])
        hits += spearmanr(x, yp).statistic >= rho
    return SplitHalf(rho, float((1 + hits) / (1 + n_perm)), n_cells, len(xs), level,
                     float(r.pvalue))


def _halves(tab: LapTable, team: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Each driver's laps alternately into halves by lap-time rank, inside each Q
    segment where both drivers have 2+ laps (so both halves are segment-matched)."""
    m_team = tab.team_mask(team)
    pair = sorted(set(tab.driver[m_team].tolist()))
    if len(pair) != 2:
        return None
    h1, h2 = [], []
    for q in sorted(set(tab.q_segment[m_team].tolist())):
        idx = [np.flatnonzero(m_team & (tab.q_segment == q) & (tab.driver == d)) for d in pair]
        if min(len(i) for i in idx) < 2:
            continue
        for i in idx:
            i = i[np.argsort(tab.lap_time[i], kind="stable")]
            h1.extend(i[0::2])
            h2.extend(i[1::2])
    if not h1:
        return None
    return np.array(h1), np.array(h2)


def split_half_within(tabs: list[LapTable], level: str = "phase", n_perm: int = 2000,
                      seed: int = 0) -> SplitHalf:
    """Within-session reliability: deltas from odd- vs even-ranked laps of each driver,
    split inside each Q segment.

    ``level`` is 'phase' (every phase cell) or 'type' (corner type x phase sums).
    """
    xs, ys = [], []
    for tab in tabs:
        for team in sorted(set(tab.team)):
            h = _halves(tab, team)
            if h is None:
                continue
            m1 = np.zeros(len(tab.keys), bool)
            m1[h[0]] = True
            m2 = np.zeros(len(tab.keys), bool)
            m2[h[1]] = True
            d1 = pair_delta(tab.rows(m1), team, top_n=None)
            d2 = pair_delta(tab.rows(m2), team, top_n=None)
            if d1 is None or d2 is None:
                continue
            if level == "type":
                a1 = aggregate(d1.delta, tab.labels, "type")
                a2 = aggregate(d2.delta, tab.labels, "type")
                keys = sorted(a1)
                xs.append(np.array([a1[k] for k in keys]))
                ys.append(np.array([a2[k] for k in keys]))
            else:
                xs.append(d1.delta)
                ys.append(d2.delta)
    return _rho_test(xs, ys, level, n_perm, seed)


def split_half_across(tabs: list[LapTable], top_n: int | None = 2, n_perm: int = 2000,
                      seed: int = 0) -> SplitHalf:
    """Across-session reliability: alternate sessions into halves, average each driver
    pair's corner type x phase deltas within a half, correlate pairs present in both."""
    halves: list[dict] = [{}, {}]
    for i, tab in enumerate(tabs):
        for d in teammate_deltas(tab, top_n):
            agg = aggregate(d.delta, tab.labels, "type")
            for k, v in agg.items():
                halves[i % 2].setdefault((d.a, d.b), {}).setdefault(k, []).append(v)
    xs, ys = [], []
    for ab in sorted(set(halves[0]) & set(halves[1])):
        keys = sorted(set(halves[0][ab]) & set(halves[1][ab]))
        if keys:
            xs.append(np.array([np.mean(halves[0][ab][k]) for k in keys]))
            ys.append(np.array([np.mean(halves[1][ab][k]) for k in keys]))
    return _rho_test(xs, ys, "across sessions", n_perm, seed)
