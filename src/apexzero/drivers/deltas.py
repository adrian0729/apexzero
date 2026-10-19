"""Per-phase lap tables, the R1 envelope, teammate deltas and the reliability checks.

Every number here is a time in the distance domain: how long a lap spent between two
fixed lap distances. Because phases tile the lap, a teammate's phase deltas add up
exactly to the lap-time delta of the same laps; ``teammate_deltas`` asserts this.

- **R1** (plan section 6) is the empirical envelope: per car-session, the best time in
  each phase across both teammates' laps. No model is involved. Teammate deltas do not
  depend on it (the reference cancels); it is reported because the per-lap residual
  y = t - R1 is the input to the later hierarchical model.
- **Teammate delta** for a car-session: mean phase times of driver B's top-n laps minus
  driver A's, using only Q segments both drivers took part in (same track evolution).
- **Placebo**: shuffle the driver labels among the car-session's laps; any effect that
  survives is an artefact of selection or the pipeline.
- **Split-half**: the same delta from two disjoint halves of each driver's laps; Spearman
  rho across cells is the reliability.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def rows(self, mask: np.ndarray) -> LapTable:
        idx = np.flatnonzero(mask)
        return LapTable(self.session, [self.keys[i] for i in idx], self.driver[idx],
                        [self.team[i] for i in idx], self.q_segment[idx], self.lap_time[idx],
                        self.phase_t[idx], self.labels, self.acronym)


def r1_envelope(tab: LapTable) -> dict[str, np.ndarray]:
    """Best phase time per car-session (team), across both teammates' laps."""
    out = {}
    for team in sorted(set(tab.team)):
        m = np.array([t == team for t in tab.team])
        out[team] = tab.phase_t[m].min(axis=0)
    return out


@dataclass
class PairDelta:
    session: str
    team: str
    a: int                      # reference driver (number)
    b: int
    n_a: int                    # laps used
    n_b: int
    q_segments: tuple[int, ...]
    delta: np.ndarray           # [P] seconds, mean(B) - mean(A)
    lap_delta: float            # mean lap time B - A over the same laps

    def label(self, acr: dict[int, str]) -> str:
        return f"{acr.get(self.b, self.b)} - {acr.get(self.a, self.a)}"


def _select(tab: LapTable, idx: np.ndarray, top_n: int | None) -> np.ndarray:
    if top_n is None or len(idx) <= top_n:
        return idx
    return idx[np.argsort(tab.lap_time[idx], kind="stable")[:top_n]]


def pair_delta(tab: LapTable, team: str, top_n: int | None = 2,
               drivers: np.ndarray | None = None) -> PairDelta | None:
    """Teammate delta for one car-session; ``drivers`` overrides the labels (placebo)."""
    drv = tab.driver if drivers is None else drivers
    m_team = np.array([t == team for t in tab.team])
    pair = sorted(set(drv[m_team].tolist()))
    if len(pair) != 2:
        return None
    a, b = pair
    qa = set(tab.q_segment[m_team & (drv == a)].tolist())
    qb = set(tab.q_segment[m_team & (drv == b)].tolist())
    common = sorted(qa & qb)
    if not common:
        return None
    in_q = np.isin(tab.q_segment, common)
    ia = _select(tab, np.flatnonzero(m_team & in_q & (drv == a)), top_n)
    ib = _select(tab, np.flatnonzero(m_team & in_q & (drv == b)), top_n)
    delta = tab.phase_t[ib].mean(axis=0) - tab.phase_t[ia].mean(axis=0)
    lap_delta = float(tab.lap_time[ib].mean() - tab.lap_time[ia].mean())
    # Additivity: the phases tile the lap, so the deltas must sum to the lap delta.
    assert abs(delta.sum() - lap_delta) < 1e-6, (delta.sum(), lap_delta)
    return PairDelta(tab.session, team, int(a), int(b), len(ia), len(ib), tuple(common), delta,
                     lap_delta)


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


@dataclass
class PlaceboResult:
    real_abs_mean: float          # mean |delta| over (pair, phase) cells, true labels
    placebo_abs_mean: float       # same, averaged over shuffles
    placebo_mean: float           # signed mean over shuffles and cells (should be ~0)
    placebo_mean_se: float
    ss_p_value: float             # permutation p of sum(delta^2) over all cells, real vs shuffled
    n_cells: int
    n_shuffles: int
    cell_null_sd: np.ndarray      # [cells] placebo sd per (pair, phase), flattened


def placebo(tabs: list[LapTable], top_n: int | None = 2, n_shuffles: int = 200,
            seed: int = 0) -> PlaceboResult:
    """Shuffle driver labels within each car-session (lap counts per driver preserved).

    Two readings. Shuffled deltas must average zero (the pipeline and the top-n
    selection create no driver effect by themselves). And the real labels should explain
    more phase-level variance than shuffled ones: ``ss_p_value`` compares the sum of
    squared deltas over every cell with the same sum under shuffled labels (one shuffle
    per car-session, drawn independently). A per-cell permutation test is not offered:
    with two laps per driver there are only three distinct splits, so no single cell can
    reach p < 0.05.
    """
    rng = np.random.default_rng(seed)
    real, null = [], []
    for tab in tabs:
        for team in sorted(set(tab.team)):
            d = pair_delta(tab, team, top_n)
            if d is None:
                continue
            sub = tab.rows(np.array([t == team for t in tab.team])
                           & np.isin(tab.q_segment, d.q_segments))
            sims = []
            for _ in range(n_shuffles):
                p = pair_delta(sub, team, top_n, drivers=rng.permutation(sub.driver))
                if p is not None:
                    sims.append(p.delta)
            if sims:
                real.append(d.delta)
                null.append(np.array(sims))
    if not real:
        raise ValueError("no teammate pairs")
    real_arr = np.concatenate(real)
    s_min = min(len(n) for n in null)
    null_arr = np.concatenate([n[:s_min] for n in null], axis=1)  # [shuffles, cells]
    ss_null = (null_arr**2).sum(axis=1)
    ss_real = float((real_arr**2).sum())
    flat = null_arr.mean(axis=1)
    return PlaceboResult(
        real_abs_mean=float(np.abs(real_arr).mean()),
        placebo_abs_mean=float(np.abs(null_arr).mean()),
        placebo_mean=float(null_arr.mean()),
        placebo_mean_se=float(flat.std(ddof=1) / np.sqrt(len(flat))),
        ss_p_value=float((1 + np.sum(ss_null >= ss_real)) / (1 + len(ss_null))),
        n_cells=len(real_arr),
        n_shuffles=s_min,
        cell_null_sd=null_arr.std(axis=0),
    )


@dataclass
class SplitHalf:
    rho: float
    p_value: float
    n_cells: int
    level: str


def _halves(tab: LapTable, team: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Each driver's laps (common Q segments) alternately into halves by lap-time rank."""
    m_team = np.array([t == team for t in tab.team])
    pair = sorted(set(tab.driver[m_team].tolist()))
    if len(pair) != 2:
        return None
    q = set(tab.q_segment[m_team & (tab.driver == pair[0])].tolist()) & set(
        tab.q_segment[m_team & (tab.driver == pair[1])].tolist())
    in_q = np.isin(tab.q_segment, sorted(q))
    h1, h2 = [], []
    for drv in pair:
        idx = np.flatnonzero(m_team & in_q & (tab.driver == drv))
        if len(idx) < 2:
            return None
        idx = idx[np.argsort(tab.lap_time[idx], kind="stable")]
        h1.extend(idx[0::2])
        h2.extend(idx[1::2])
    return np.array(h1), np.array(h2)


def split_half_within(tabs: list[LapTable], level: str = "phase") -> SplitHalf:
    """Within-session reliability: deltas from odd- vs even-ranked laps of each driver.

    ``level`` is 'phase' (every phase cell) or 'type' (corner type x phase sums).
    """
    x, y = [], []
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
                a1, a2 = aggregate(d1.delta, tab.labels, "type"), aggregate(d2.delta, tab.labels,
                                                                            "type")
                keys = sorted(a1)
                x.extend(a1[k] for k in keys)
                y.extend(a2[k] for k in keys)
            else:
                x.extend(d1.delta)
                y.extend(d2.delta)
    if len(x) < 3:
        return SplitHalf(float("nan"), float("nan"), len(x), level)
    r = spearmanr(x, y)
    return SplitHalf(float(r.statistic), float(r.pvalue), len(x), level)


def split_half_across(tabs: list[LapTable], top_n: int | None = 2) -> SplitHalf:
    """Across-session reliability: alternate sessions into halves, average each driver
    pair's corner type x phase deltas within a half, correlate pairs present in both."""
    halves: list[dict] = [{}, {}]
    for i, tab in enumerate(tabs):
        for d in teammate_deltas(tab, top_n):
            agg = aggregate(d.delta, tab.labels, "type")
            for k, v in agg.items():
                halves[i % 2].setdefault((d.a, d.b, k), []).append(v)
    common = sorted(set(halves[0]) & set(halves[1]))
    x = [np.mean(halves[0][k]) for k in common]
    y = [np.mean(halves[1][k]) for k in common]
    if len(x) < 3:
        return SplitHalf(float("nan"), float("nan"), len(x), "across sessions")
    r = spearmanr(x, y)
    return SplitHalf(float(r.statistic), float(r.pvalue), len(x), "across sessions")
