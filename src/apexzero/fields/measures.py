"""Measures of the receptive fields (design Section 6) and their summaries per mirror pair and
per agent.

Everything here is NumPy on impulse responses that ``network.fields`` (the network's VJP) or
``kkt.fields`` (the optimum's adjoints) already computed: an impulse response is
d outcome / d sample, one [N] array per feature (``perturb.Samples``, perturb.py:71). No
network, solver or JAX is touched.

**Edge-tracking gains and buffer** (design Section 4). Along the trapezoid shift of one side's
width (``perturb.edge_shift``, perturb.py:113; +1 widens that side):

- G = 1 - dc_min/dDelta_in at the base run's argmin s* (``edge_gain``; ``kkt.edge_gain`` is the
  same formula on the optimum's Fields). c_min reads the inside width at s* directly
  (outcomes.py:281-290), so dc_min/dDelta = e(s*) - sigma dn(s*)/dDelta, with e(s*) the edge's
  own move at the pick (``edge_move``: the spline of the sampled trapezoid at s*, or its
  interpolation in s between the pick's states at a window end). G = sigma dn/dDelta only
  where e(s*) = 1: c_min's window [s_c0 - 50, s_c1 + 50] is wider than the trapezoid's plateau
  [s_c0 - 20, s_c1 + 20], and a width-blind line at an argmin with e(s*) < 1 would score
  G = 1 - e(s*) instead of 0, biasing H2' toward 'lawful'. Rule fixed at the tag (option (b)
  of the review): G = 1 - dc_min/dDelta is kept, and a key with e(s*) < ``EDGE_PLATEAU`` = 0.99
  is excluded for the reason 'argmin off plateau' (``EdgeMeasures.on_plateau``), on both sides.
  G* = 1 on an active inside contact; G = 0 for a width-blind line on the plateau.
- G_out = dc_min/dDelta_out = -sigma dn(s*)/dDelta_out (``outside_gain``): how far the line
  moves outward at the inside argmin s* per metre the outside edge moves outward. c_min does
  not read the outside width, so this is the line's move alone. The design leaves G_out
  undefined beyond "the same trapezoid on the outside edge"; ``1 - dc_min/dDelta_out``
  (``kkt.edge_gain(side='outside')``) is 1 for every line that ignores the outside edge, the
  optimum included, so it is not used. An outside-clearance outcome would need a fifth
  outcome in both pipelines (open issue of this stage).
- B = c_min at the base point (the buffer); to first order an inside narrowing Delta crashes
  the car when (1 - G) Delta > B. An invalid c_min (B NaN) makes every edge measure NaN.
- r_in = (dv_apex/dDelta_in)_net / (dv_apex/dDelta_in)_opt (``speed_ratio``).

**Distance-resolved fields.** ``bump_field`` contracts impulse responses with unit-peak
Gaussian bumps centred at s_1 + d (``perturb.bump_profiles`` and ``combine``,
perturb.py:135-146), on the d grid from -20 m in 2 m steps (``perturb.d_grid``) to the end of
the support (``support_end``: max_k s_k + clip(6 vx_k, 150, 500) over the states the outcome
reads, plus the spline's reach). For two fields F_net, F_opt on the same grid:

- gain g = <F_net, F_opt> / <F_opt, F_opt> (``gain``); a near-zero network field is a low
  gain, not missing data;
- shape rho = cosine similarity (``cosine``);
- location = centroid of |F_net| - centroid of |F_opt| (m; ``location_shift``);
- extent W90 = width of the shortest run of grid points holding 90% of sum |F| (``w90``);
- reach E90 = the d at which the cumulative trapezoid integral of |F| from d = -20 m first
  reaches 90% of its total, linearly interpolated (``e90``). A Gaussian bump blurs reach by
  1.2816 sigma_b (the normal 0.9 quantile), so reaches are compared only at equal sigma_b;
- H1-typical's shape criterion is a matched-versus-mismatched specificity test
  (``shape_specificity``), not a threshold on an extreme order statistic: per key k,
  delta_rho_k = rho(F_net,k, F_opt,k) - median_j rho(F_net,k, F_opt,j) over keys j whose
  geometry is far from k's (``far_geometry``: |ln(R_j / R_k)| >= 0.4 or |theta_j - theta_k|
  >= 45 deg; nearer geometries count as the same), on fields recoded toward the turn
  (``perturb.recode``) and laid on one grid aligned at s_c0 (``NULL_D``), per outcome. The key's
  criterion is delta_rho_k > 0 (``h1_criteria``); an agent's is the cluster-bootstrap CI of the
  pair values of delta_rho lying above 0. The cross-geometry null (``cross_geometry_null``: rho,
  g and location between the optimum's fields at keys of far geometry) and the phase ceiling
  (``phase_ceiling``: the optimum's fields across one cell's phases) are reported beside it,
  as their medians (descriptive; nothing depends on how many keys are passed).

**Turn recoding** (``turn_fields``): curvature signed toward the turn, widths as inside and
outside (``perturb.recode``, perturb.py:148), and for the steer outcome the outcome itself
flipped on right-hand corners (``perturb.turn_steer``).

**Mirror pairs** (protocol 9.1, docs/interp/psychophysics-protocol.md:858-866, and design
Section 6). The unit is the mirror pair (geometry x approach): 2 directions x 4 phases = 8
keys. ``pair_value`` is the median over the pair's keys with both directions weighted
equally (``weighted_median``: weight 1 / (2 n_dir) per key, the mean of the lower and upper
weighted medians, so equal weights give ``np.median``); a pair is undefined if more than 3
of its 8 keys are excluded (more than floor(3/8 n) of n in general, so 1 of 4 when one
direction is reported alone). P7's 16-phase
cells first take the median over phases within each direction; the pair value is then the
mean of the two direction medians. Excluded keys are NaN. The asymmetry index is
m_L - m_R over mirror partners (same pair, approach and phase; ``asymmetry``).

**Agents** (``agent_summary``): the median over geometries (R, theta) of the pair values (a
geometry's pairs, e.g. tier C's widths 9 / 12 / 15 m at one (R, theta), first take their
median), with a 95% percentile interval from a bootstrap that resamples geometries
(``cluster_bootstrap``; seeded from step 1's ROOT_SEED on a stream of its own,
``FIELDS_STREAMS``). Reported for left keys, right keys and pooled (``by_direction``).
``h2_decision`` applies H2''s bands (design Section 7) to such an interval.

**Coverage** (``coverage``, design Section 8): exclusions per (agent, snapshot), outcome and
reason (``EXCLUSION_REASONS``, assigned in that order by ``exclusion_reason``: invalid outcome,
then gate 3, then numerical (the optimum's key did not re-polish or failed its adjoint check),
then inertia, then kappa clipping, then argmin off plateau); above 30% excluded,
the agent is descriptive for that outcome. A pipeline failure (``PIPELINE_FAILURES``: the loop
exhausted) is never an exclusion: ``coverage`` refuses it, so the stage that produced it fails.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp.constants import DS, ROOT_SEED, STREAM_OFFSET
from apexzero.interp.stimuli import Landmarks
from apexzero.tracks.spline import packed_coeffs, spline_np

MASS = 0.9                  # W90 / E90: the share of sum |F|
E90_BLUR = 1.2815515655446004   # the normal 0.9 quantile: a bump at d_k has E90 d_k + this sb
PAIR_KEYS = 8               # protocol 9.1: 2 directions x 4 phases
PAIR_MAX_EXCLUDED = 3       # of 8 (protocol 9.1; design Section 6)
P7_PHASES = 16              # P7 cells: median over phases first (design Section 6)
SPLINE_REACH = 2 * DS       # m: the cubic B-spline's support beyond the last sample read
BOOTSTRAP_N = 20_000        # as step 1's descriptive bands (analysis.py:488)
LEVEL = 0.95
FIELDS_STREAMS = {"bootstrap": 101, "gate1_sample": 102, "dose_sample": 103,
                  "permutation": 104}   # seed streams of step 2
H2_DEFICIT_HI = 0.5         # H2': confirm (deficit) if the CI's upper end is <= 0.5
H2_LAWFUL = (0.8, 1.2)      # H2': lawful if the CI lies within
H2_OVER_LO = 1.2            # H2': over-tracking if the CI's lower end is >= 1.2
H1_GAIN = (0.8, 1.25)       # H1-typical: gain band (design Section 7)
H1_LOCATION = 12.0          # m, H1-typical: location within +- the preview spacing (12-14 m)
R_IN_BAND = (0.5, 2.0)      # H2-speed: under / lawful / over
COVERAGE_MAX = 0.30         # excluded share above which an agent is descriptive for an outcome
EXCLUSION_REASONS = ("invalid outcome", "gate 3", "numerical", "inertia", "kappa clipping",
                     "argmin off plateau")   # assigned in this order (``exclusion_reason``)
LOOP_EXHAUSTED = "loop length exhausted"     # network.EXHAUSTED: a pipeline failure
PIPELINE_FAILURES = (LOOP_EXHAUSTED,)
EDGE_PLATEAU = 0.99         # e(s*) below this: 'argmin off plateau' (G is not sigma dn/dDelta)
FAR_LOG_R = 0.4             # shape specificity: far geometry if |ln(R_j / R_k)| >= this ...
FAR_THETA = 45.0            # ... or |theta_j - theta_k| >= this (deg)
SHAPE_DELTA_MIN = 0.0       # H1-typical shape: delta_rho (and its CI's lower end) above this


def stream_rng(name: str) -> np.random.Generator:
    """A NumPy generator of one of step 2's seed streams (disjoint from step 1's
    ``io.stream_rng`` streams, constants.py:57-67)."""
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + FIELDS_STREAMS[name]])


# ------------------------------------------------------------------------- edge gains


def along(impulse: Samples, direction: Samples) -> float:
    """d outcome / d amplitude along a sample-space direction: sum over features of
    <impulse_f, direction_f>."""
    return float(sum(np.dot(np.asarray(getattr(impulse, f), float),
                            np.asarray(getattr(direction, f), float)) for f in PT.FEATURES))


def edge_derivative(impulse: Samples, landmarks: Landmarks, sigma: int, side: str = "inside",
                    ds: float = DS) -> float:
    """dy/dDelta along the trapezoid shift of the ``side`` width (+1 widens it)."""
    n = len(np.asarray(impulse.kappa))
    return along(impulse, PT.edge_shift(n, landmarks, sigma, side, 1.0, ds))


def edge_gain(impulse_c_min: Samples, landmarks: Landmarks, sigma: int, ds: float = DS) -> float:
    """G = 1 - dc_min/dDelta_in (module docstring); NaN for NaN impulses (invalid c_min: both
    ``network.fields`` and ``kkt.fields`` give NaN rows there)."""
    return 1.0 - edge_derivative(impulse_c_min, landmarks, sigma, "inside", ds)


def edge_move(landmarks: Landmarks, n: int, s_states, pick, ds: float = DS) -> float:
    """e(s*): the inside edge's own move at c_min's pick per unit trapezoid amplitude, i.e.
    dc_min/dDelta with the states held. The spline of the sampled trapezoid (the arrays'
    natural cubic spline, ``packed_coeffs`` and ``spline_np``, as ``outcomes.clearance_np``
    reads the widths) at the pick state's s, or interpolated linearly in s to s_star between
    states i0 and i1 for a window-end pick (``outcomes.read``'s arithmetic). ``pick`` is an
    ``outcomes.Pick``; NaN for None."""
    if pick is None:
        return float("nan")
    s_states = np.asarray(s_states, float)
    h = PT.trapezoid(PT.sample_s(int(n), ds), landmarks.s_c0, landmarks.s_c1)
    c = packed_coeffs(h, False, int(n))
    i0, i1 = int(np.asarray(pick.i0)), int(np.asarray(pick.i1))
    e0, e1 = spline_np(c, np.array([s_states[i0], s_states[i1]]), ds, int(n), False)
    if not bool(np.asarray(pick.interp)):
        return float(e0)
    s0, s1 = s_states[i0], s_states[i1]
    return float(e0 + (float(np.asarray(pick.s_star)) - s0) * (e1 - e0) / (s1 - s0))


def outside_gain(impulse_c_min: Samples, landmarks: Landmarks, sigma: int, ds: float = DS
                 ) -> float:
    """G_out = dc_min/dDelta_out: the line's outward move at the inside argmin per metre the
    outside edge moves outward (module docstring)."""
    return edge_derivative(impulse_c_min, landmarks, sigma, "outside", ds)


def speed_ratio(dv_net: float, dv_opt: float, floor: float = 1e-9) -> float:
    """r_in = dv_net / dv_opt (NaN when the optimum's response is below ``floor``)."""
    return float(dv_net / dv_opt) if abs(dv_opt) > floor else float("nan")


@dataclass(frozen=True)
class EdgeMeasures:
    """G, G_out and B of one key, plus the raw derivatives they come from and the plateau
    test (``e_star``; NaN when no pick was given)."""

    G: float
    G_out: float
    B: float
    dc_in: float              # dc_min / dDelta_in
    dc_out: float             # dc_min / dDelta_out
    dv_in: float              # dv_apex / dDelta_in (r_in's numerator or denominator)
    e_star: float = float("nan")     # the edge's own move at the pick (``edge_move``)

    @property
    def on_plateau(self) -> bool:
        """e(s*) >= EDGE_PLATEAU: G is the line's move sigma dn/dDelta (False: the key is
        excluded, 'argmin off plateau')."""
        return bool(np.isfinite(self.e_star) and self.e_star >= EDGE_PLATEAU)


def edge_measures(impulse_c_min: Samples, impulse_v_apex: Samples, c_min: float,
                  landmarks: Landmarks, sigma: int, ds: float = DS, s_states=None,
                  pick=None) -> EdgeMeasures:
    """The edge-tracking gains, the buffer and the inside-shift speed response of one key;
    with the run's states' s and c_min's pick, also e(s*). Every measure is NaN when c_min is
    (an invalid outcome), whatever the impulse rows hold."""
    if not np.isfinite(c_min):
        nan = float("nan")
        return EdgeMeasures(nan, nan, nan, nan, nan, nan, nan)
    dc_in = edge_derivative(impulse_c_min, landmarks, sigma, "inside", ds)
    dc_out = edge_derivative(impulse_c_min, landmarks, sigma, "outside", ds)
    e = float("nan")
    if s_states is not None and pick is not None:
        e = edge_move(landmarks, len(np.asarray(impulse_c_min.kappa)), s_states, pick, ds)
    return EdgeMeasures(G=1.0 - dc_in, G_out=dc_out, B=float(c_min), dc_in=dc_in, dc_out=dc_out,
                        dv_in=edge_derivative(impulse_v_apex, landmarks, sigma, "inside", ds),
                        e_star=e)


# ------------------------------------------------------------------------- bump fields


def support_end(x: np.ndarray, k_end: int | None = None) -> float:
    """The last arc length (m) the closed loop can read up to state k_end (all states by
    default): max_k s_k + clip(6 vx_k, 150, 500) (the preview horizon, racing.py:86-90), plus
    the spline's reach (design Section 4, 'Impulse responses')."""
    x = np.asarray(x, float)
    k = len(x) if k_end is None else int(k_end) + 1
    s, vx = x[:k, 0], x[:k, 3]
    return float(np.max(s + np.clip(6.0 * np.maximum(vx, 0.0), 150.0, 500.0)) + SPLINE_REACH)


def d_grid_to(s_end: float, s1: float) -> np.ndarray:
    """The design's d grid, -20 m to s_end - s_1 in 2 m steps (``perturb.d_grid``)."""
    return PT.d_grid(float(s_end) - float(s1))


def bump_field(impulse, s1: float, d: np.ndarray, sigma_b: float = PT.SIGMA_B[0],
               ds: float = DS) -> np.ndarray:
    """F(d) [..., M]: the outcome's derivative along unit-peak Gaussian bumps centred at
    s_1 + d of one feature, from that feature's impulse responses [..., N]."""
    imp = np.asarray(impulse, float)
    prof = PT.bump_profiles(PT.sample_s(imp.shape[-1], ds), s1, np.asarray(d, float), sigma_b)
    return PT.combine(imp, prof)


def turn_fields(impulse: Samples, sigma: int, outcome: str = "") -> dict[str, np.ndarray]:
    """Impulse responses with respect to (kappa_turn, w_in, w_out) (``perturb.recode``); for
    ``outcome`` 'steer' the outcome is also signed toward the turn (``perturb.turn_steer``)."""
    rec = PT.recode(impulse, sigma)
    if outcome == "steer":
        rec = {k: PT.turn_steer(v, sigma) for k, v in rec.items()}
    return rec


def _abs(field) -> np.ndarray:
    return np.abs(np.asarray(field, float))


def gain(net, opt) -> float:
    """g = <net, opt> / <opt, opt> (NaN when the reference field is all zero)."""
    a, b = np.asarray(net, float), np.asarray(opt, float)
    den = float(b @ b)
    return float(a @ b / den) if den > 0 else float("nan")


def cosine(net, opt) -> float:
    """rho = <net, opt> / (|net| |opt|) (NaN when either field is all zero)."""
    a, b = np.asarray(net, float), np.asarray(opt, float)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 0 else float("nan")


def centroid(field, d) -> float:
    """sum d |F| / sum |F| (m; NaN for a zero field)."""
    w = _abs(field)
    tot = float(w.sum())
    return float(np.asarray(d, float) @ w / tot) if tot > 0 else float("nan")


def location_shift(net, opt, d) -> float:
    """Centroid of |net| minus centroid of |opt| (m)."""
    return centroid(net, d) - centroid(opt, d)


def peak(field, d) -> float:
    """The grid d of max |F| (NaN for a zero field)."""
    w = _abs(field)
    if w.max(initial=0.0) <= 0:
        return float("nan")
    return float(np.asarray(d, float)[int(np.argmax(w))])


def w90(field, d, mass: float = MASS) -> float:
    """Width (m) of the shortest run of consecutive grid points whose |F| sums to at least
    ``mass`` of the total: (j - i) step for the shortest window [d_i, d_j]."""
    w = _abs(field)
    dd = np.asarray(d, float)
    tot = float(w.sum())
    if tot <= 0:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(w)])
    best = np.inf
    j = 0
    for i in range(len(w)):                      # two pointers: c[j + 1] - c[i] >= mass tot
        j = max(j, i)
        while j < len(w) and c[j + 1] - c[i] < mass * tot * (1 - 1e-12):
            j += 1
        if j == len(w):
            break
        best = min(best, dd[j] - dd[i])
    return float(best)


def e90(field, d, mass: float = MASS) -> float:
    """Reach (m): the d where the cumulative trapezoid integral of |F| from the grid's start
    reaches ``mass`` of its total, interpolated linearly between grid points."""
    w = _abs(field)
    dd = np.asarray(d, float)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(dd))])
    tot = float(cum[-1])
    if tot <= 0:
        return float("nan")
    target = mass * tot
    j = int(np.searchsorted(cum, target, side="left"))
    j = min(max(j, 1), len(dd) - 1)
    c0, c1 = cum[j - 1], cum[j]
    t = 0.0 if c1 == c0 else (target - c0) / (c1 - c0)
    return float(dd[j - 1] + t * (dd[j] - dd[j - 1]))


@dataclass(frozen=True)
class Shape:
    """The location, extent and reach of one distance-resolved field."""

    amplitude: float          # max |F|
    peak: float               # m
    centroid: float           # m
    w90: float                # m
    e90: float                # m

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in self.__dataclass_fields__}


def shape(field, d) -> Shape:
    w = _abs(field)
    return Shape(amplitude=float(w.max(initial=0.0)), peak=peak(field, d),
                 centroid=centroid(field, d), w90=w90(field, d), e90=e90(field, d))


def compare(net, opt, d) -> dict[str, float]:
    """The design's comparison of a network field with the optimum's on one d grid: g, rho,
    location shift, and each field's W90 and E90."""
    return {"g": gain(net, opt), "rho": cosine(net, opt), "location": location_shift(net, opt, d),
            "w90_net": w90(net, d), "w90_opt": w90(opt, d), "e90_net": e90(net, d),
            "e90_opt": e90(opt, d)}


def h1_criteria(cmp: dict[str, float], rho_far: float, spacing: float = H1_LOCATION
                ) -> dict[str, bool]:
    """H1-typical's three criteria on one key's comparison (design Section 7): gain within
    [0.8, 1.25], location within +- the preview spacing, and shape: the matched rho above
    ``rho_far``, the median rho of the same network field against the optimum's fields at
    far geometries (``shape_specificity``; delta_rho > SHAPE_DELTA_MIN)."""
    g, loc, rho = cmp["g"], cmp["location"], cmp["rho"]
    return {"gain": bool(np.isfinite(g) and H1_GAIN[0] <= g <= H1_GAIN[1]),
            "location": bool(np.isfinite(loc) and abs(loc) <= spacing),
            "shape": bool(np.isfinite(rho) and np.isfinite(rho_far)
                          and rho - rho_far > SHAPE_DELTA_MIN)}


NULL_D = (-150.0, 400.0)    # m, the common d grid relative to s_c0 (nulls and specificity)


def far_geometry(ga, gb, log_r: float = FAR_LOG_R, theta: float = FAR_THETA) -> bool:
    """Geometries (R, theta) count as different when |ln(R_a / R_b)| >= ``log_r`` or
    |theta_a - theta_b| >= ``theta`` deg; nearer ones (R 13.0 tier V against R 13.1 tier A)
    count as the same."""
    (ra, ta), (rb, tb) = ga, gb
    return bool(abs(np.log(float(ra) / float(rb))) >= log_r or abs(float(ta) - float(tb)) >= theta)


def aligned_field(impulse: Samples, s_c0: float, sigma: int, outcome: str = "",
                  feature: str = "kappa_turn", sigma_b: float = PT.SIGMA_B[0],
                  d_range: tuple[float, float] = NULL_D) -> np.ndarray:
    """One recoded feature's bump field (``turn_fields``: kappa_turn, w_in or w_out; steer
    signed toward the turn) on the common grid d (``NULL_D``) aligned at the corner start:
    bumps centred at s_c0 + d."""
    d = PT.d_grid(d_range[1], d_range[0])
    rec = turn_fields(impulse, sigma, outcome)
    return bump_field(rec[feature], s_c0, d, sigma_b)


def _aligned(fields: dict, outcome: str, feature: str, sigma_b: float, d_range) -> dict:
    """{label: (geometry, F)} from {label: (geometry, s_c0, sigma, impulse Samples or None)}."""
    return {k: (tuple(g), aligned_field(imp, s_c0, sig, outcome, feature, sigma_b, d_range))
            for k, (g, s_c0, sig, imp) in fields.items() if imp is not None}


def cross_geometry_null(fields: dict, outcome: str = "", sigma_b: float = PT.SIGMA_B[0],
                        feature: str = "kappa_turn", d_range: tuple[float, float] = NULL_D
                        ) -> dict:
    """The cross-geometry null (design Section 8, descriptive): rho, g and location between
    the optimum's recoded fields at keys of far geometry (``far_geometry``), aligned at
    s_c0. ``fields`` maps a label to (geometry (R, theta), s_c0, sigma, impulse Samples or
    None). Reported as the pairs and their median rho (no extreme order statistic)."""
    d = PT.d_grid(d_range[1], d_range[0])
    aligned = _aligned(fields, outcome, feature, sigma_b, d_range)
    rows = []
    keys = list(aligned)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (ga, Fa), (gb, Fb) = aligned[a], aligned[b]
            if not far_geometry(ga, gb):
                continue
            c = compare(Fa, Fb, d)
            rows.append({"a": str(a), "b": str(b), "rho": c["rho"], "g": c["g"],
                         "location": c["location"]})
    rho = np.array([r["rho"] for r in rows], float)
    return {"pairs": rows, "n_pairs": len(rows),
            "rho_median": float(np.nanmedian(rho)) if len(rho) else float("nan")}


def phase_ceiling(fields: dict, outcome: str = "", sigma_b: float = PT.SIGMA_B[0],
                  feature: str = "kappa_turn", d_range: tuple[float, float] = NULL_D) -> dict:
    """The phase ceiling (design Section 8): rho, g and location between the optimum's
    recoded fields at the phases of one cell (``fields``: {phase: (s_c0, sigma, impulse
    Samples or None)}), aligned at s_c0: how much of a field's shape survives a change of
    start phase alone. Pairs, and the median and minimum rho."""
    d = PT.d_grid(d_range[1], d_range[0])
    aligned = {k: aligned_field(imp, s_c0, sig, outcome, feature, sigma_b, d_range)
               for k, (s_c0, sig, imp) in fields.items() if imp is not None}
    rows = []
    keys = sorted(aligned)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            c = compare(aligned[a], aligned[b], d)
            rows.append({"a": a, "b": b, "rho": c["rho"], "g": c["g"], "location": c["location"]})
    rho = np.array([r["rho"] for r in rows], float)
    return {"pairs": rows, "n_pairs": len(rows),
            "rho_median": float(np.nanmedian(rho)) if len(rho) else float("nan"),
            "rho_min": float(np.nanmin(rho)) if len(rho) else float("nan")}


def shape_specificity(net: dict, opt: dict, outcome: str = "", sigma_b: float = PT.SIGMA_B[0],
                      feature: str = "kappa_turn", d_range: tuple[float, float] = NULL_D
                      ) -> list[dict]:
    """H1-typical's shape criterion per key (module docstring): for each label k in both
    ``net`` and ``opt`` ({label: (geometry, s_c0, sigma, impulse Samples or None)}, the
    network's and the optimum's fields of one outcome at the same keys), rho_match =
    rho(F_net,k, F_opt,k), rho_far = the median of rho(F_net,k, F_opt,j) over the optimum's
    fields at far geometries j, and delta_rho = rho_match - rho_far (NaN when k has no far
    key or a field is missing)."""
    fn = _aligned(net, outcome, feature, sigma_b, d_range)
    fo = _aligned(opt, outcome, feature, sigma_b, d_range)
    rows = []
    for k, (g, Fn) in fn.items():
        if k not in fo:
            continue
        match = cosine(Fn, fo[k][1])
        far = [cosine(Fn, Fj) for j, (gj, Fj) in fo.items() if j != k and far_geometry(g, gj)]
        rho_far = float(np.nanmedian(far)) if len(far) else float("nan")
        rows.append({"label": k, "geometry": list(g), "rho_match": match, "rho_far": rho_far,
                     "n_far": len(far), "delta_rho": match - rho_far})
    return rows


# ------------------------------------------------------------------------- mirror pairs


@dataclass(frozen=True)
class KeyMeasure:
    """One key's value of one measure, with what it is aggregated by. NaN = excluded."""

    key_id: str
    pair_id: int
    approach: str
    direction: int            # +1 left, -1 right
    phase: int
    n_phases: int
    R: float
    theta_deg: float
    value: float
    reason: str = ""          # why excluded ('' when valid)

    @property
    def geometry(self) -> tuple[float, float]:
        return self.R, self.theta_deg


def weighted_median(values, weights) -> float:
    """The weighted median: the mean of the lower median (the first sorted value whose
    cumulative weight reaches half the total) and the upper median (the first that passes
    it), so equal weights give ``np.median``. NaN values are dropped; NaN if none is left."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if not len(v):
        return float("nan")
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cum = np.cumsum(w) / w.sum()
    eps = 1e-12
    lo = int(np.argmax(cum >= 0.5 - eps))
    hi = int(np.argmax(cum > 0.5 + eps))
    return float(0.5 * (v[lo] + v[hi]))


@dataclass(frozen=True)
class PairValue:
    """One mirror pair's value of one measure (protocol 9.1)."""

    pair_id: int
    approach: str
    R: float
    theta_deg: float
    value: float              # NaN when undefined
    n_keys: int               # keys expected (2 directions x n_phases, or one direction's)
    n_excluded: int
    defined: bool
    directions: tuple[int, ...]

    @property
    def geometry(self) -> tuple[float, float]:
        return self.R, self.theta_deg


def max_excluded(n_keys: int) -> int:
    """Exclusions a pair of n_keys keys may have and stay defined: 3 of 8, floor(3 n / 8)."""
    return (PAIR_MAX_EXCLUDED * int(n_keys)) // PAIR_KEYS


def pair_value(rows: Sequence[KeyMeasure], directions: Iterable[int] = (1, -1)) -> PairValue:
    """The pair value of one mirror pair's keys (all one pair and approach), over the given
    directions (both: pooled; one: that direction's report)."""
    dirs = tuple(int(s) for s in directions)
    rows = [r for r in rows if int(r.direction) in dirs]
    if not rows:
        raise ValueError("no keys for this pair and these directions")
    first = rows[0]
    if any((r.pair_id, r.approach) != (first.pair_id, first.approach) for r in rows):
        raise ValueError("rows of more than one (pair, approach)")
    seen = {(r.direction, r.phase) for r in rows}
    if len(seen) != len(rows):
        raise ValueError(f"pair {first.pair_id}: two keys share (direction, phase) (a W twin?)")
    n_ph = max(int(r.n_phases) for r in rows)
    n_keys = len(dirs) * n_ph
    vals = np.array([r.value for r in rows], float)
    n_excl = n_keys - int(np.isfinite(vals).sum())
    defined = n_excl <= max_excluded(n_keys)
    value = float("nan")
    if defined:
        sig = np.array([int(r.direction) for r in rows])
        if n_ph >= P7_PHASES:          # P7: per-direction median over phases, then the mean
            meds = [float(np.nanmedian(vals[sig == s])) for s in dirs
                    if np.isfinite(vals[sig == s]).any()]
            value = float(np.mean(meds)) if meds else float("nan")
        else:
            present = [s for s in dirs if np.isfinite(vals[sig == s]).any()]
            n_dir = {s: int(np.isfinite(vals[sig == s]).sum()) for s in present}
            w = np.array([1.0 / (len(present) * n_dir[s]) if s in n_dir else 0.0 for s in sig])
            value = weighted_median(vals, w)
    return PairValue(pair_id=int(first.pair_id), approach=str(first.approach), R=float(first.R),
                     theta_deg=float(first.theta_deg), value=value, n_keys=n_keys,
                     n_excluded=n_excl, defined=bool(defined), directions=dirs)


def group_pairs(rows: Iterable[KeyMeasure]) -> dict[tuple[int, str], list[KeyMeasure]]:
    out: dict[tuple[int, str], list[KeyMeasure]] = {}
    for r in rows:
        out.setdefault((int(r.pair_id), str(r.approach)), []).append(r)
    return dict(sorted(out.items()))


def pair_values(rows: Iterable[KeyMeasure], directions: Iterable[int] = (1, -1)
                ) -> list[PairValue]:
    """Every pair's value over the given directions (pairs with no key in them are left out)."""
    dirs = tuple(directions)
    out = []
    for rs in group_pairs(rows).values():
        if any(int(r.direction) in dirs for r in rs):
            out.append(pair_value(rs, dirs))
    return out


def asymmetry(rows: Iterable[KeyMeasure]) -> list[dict]:
    """Delta m = m_L - m_R over mirror partners (same pair, approach and phase), one row per
    partnered phase with both values defined."""
    by: dict[tuple, dict[int, float]] = {}
    for r in rows:
        by.setdefault((int(r.pair_id), str(r.approach), int(r.phase)), {})[int(r.direction)] = \
            float(r.value)
    out = []
    for (pid, app, ph), v in sorted(by.items()):
        if 1 in v and -1 in v and np.isfinite(v[1]) and np.isfinite(v[-1]):
            out.append({"pair_id": pid, "approach": app, "phase": ph, "delta": v[1] - v[-1]})
    return out


def asymmetry_pairs(rows: Iterable[KeyMeasure]) -> dict[tuple[int, str], float]:
    """The median asymmetry index of each pair over its partnered phases."""
    by: dict[tuple[int, str], list[float]] = {}
    for a in asymmetry(rows):
        by.setdefault((a["pair_id"], a["approach"]), []).append(a["delta"])
    return {k: float(np.median(v)) for k, v in sorted(by.items())}


# ------------------------------------------------------------------------- agents


def geometry_values(pairs: Iterable[PairValue]) -> dict[tuple[float, float], float]:
    """The median of the defined pair values at each geometry (R, theta)."""
    by: dict[tuple[float, float], list[float]] = {}
    for p in pairs:
        if p.defined and np.isfinite(p.value):
            by.setdefault(p.geometry, []).append(p.value)
    return {g: float(np.median(v)) for g, v in sorted(by.items())}


def cluster_bootstrap(values, clusters, stat: Callable = np.nanmedian, n: int = BOOTSTRAP_N,
                      level: float = LEVEL, rng: np.random.Generator | None = None
                      ) -> tuple[float, float, float]:
    """(estimate, lo, hi): ``stat`` of the values and its percentile interval when whole
    clusters are resampled with replacement (each draw takes as many clusters as there are,
    and every value of a drawn cluster)."""
    v = np.asarray(values, float)
    c = np.asarray(clusters)
    if v.shape[0] != c.shape[0]:
        raise ValueError("values and clusters differ in length")
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    rng = stream_rng("bootstrap") if rng is None else rng
    labels, inv = np.unique(c, return_inverse=True, axis=0)
    members = [v[inv.ravel() == k] for k in range(len(labels))]
    est = float(stat(v))
    draws = rng.integers(0, len(members), size=(int(n), len(members)))
    sizes = np.array([len(m) for m in members])
    if np.all(sizes == sizes[0]):                 # equal clusters: one vectorised pass
        M = np.stack(members)
        s = stat(M[draws].reshape(int(n), -1), axis=1)
    else:
        s = np.array([stat(np.concatenate([members[k] for k in row])) for row in draws])
    q = (100 * (1 - level) / 2, 100 * (1 + level) / 2)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(s, q) if np.isfinite(s).any() else (np.nan, np.nan)
    return est, float(lo), float(hi)


@dataclass(frozen=True)
class AgentSummary:
    """One agent's measure over geometries (design Section 6)."""

    directions: tuple[int, ...]
    n_pairs: int
    n_defined: int
    n_geometries: int
    median: float             # median over geometries of the geometry values
    lo: float                 # 95% cluster-bootstrap interval over geometries
    hi: float
    pairs: tuple[PairValue, ...]

    def as_dict(self) -> dict:
        return {"directions": list(self.directions), "n_pairs": self.n_pairs,
                "n_defined": self.n_defined, "n_geometries": self.n_geometries,
                "median": self.median, "lo": self.lo, "hi": self.hi}


def agent_summary(rows: Sequence[KeyMeasure], directions: Iterable[int] = (1, -1),
                  n: int = BOOTSTRAP_N, rng: np.random.Generator | None = None) -> AgentSummary:
    """Pair values -> geometry medians -> their median, with the bootstrap over geometries."""
    dirs = tuple(directions)
    pv = pair_values(rows, dirs)
    gv = geometry_values(pv)
    vals = np.array(list(gv.values()), float)
    geos = np.array(list(gv.keys()), float).reshape(-1, 2)
    est, lo, hi = cluster_bootstrap(vals, geos, np.nanmedian, n, LEVEL, rng)
    return AgentSummary(directions=dirs, n_pairs=len(pv), n_defined=sum(p.defined for p in pv),
                        n_geometries=len(gv), median=est, lo=lo, hi=hi, pairs=tuple(pv))


def by_direction(rows: Sequence[KeyMeasure], n: int = BOOTSTRAP_N) -> dict[str, AgentSummary]:
    """The summary for left keys, right keys and pooled (design Section 4, 'Turn
    direction'), each from a fresh start of the bootstrap stream (reproducible)."""
    return {name: agent_summary(rows, dirs, n, stream_rng("bootstrap"))
            for name, dirs in (("left", (1,)), ("right", (-1,)), ("pooled", (1, -1)))}


def h2_decision(lo: float, hi: float) -> str:
    """H2''s band rule on the median G's interval (design Section 7): 'confirm' (deficit) if
    hi <= 0.5, 'lawful' if the interval lies within [0.8, 1.2], 'over-tracking' if
    lo >= 1.2, else 'inconclusive'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi <= H2_DEFICIT_HI:
        return "confirm"
    if H2_LAWFUL[0] <= lo and hi <= H2_LAWFUL[1]:
        return "lawful"
    if lo >= H2_OVER_LO:
        return "over-tracking"
    return "inconclusive"


# ------------------------------------------------------------------------- coverage


def exclusion_reason(invalid: bool, gate3_failed: bool = False, inertia_failed: bool = False,
                     kappa_clipped: bool = False, off_plateau: bool = False,
                     why_invalid: str = "", numerical: bool = False) -> str:
    """The one reason a key is excluded for an outcome, the first that applies in
    ``EXCLUSION_REASONS``' order ('' when kept). ``numerical``: the optimum's key failed to
    re-polish or its adjoint check (decided before the tag: excluded as a numerical failure,
    never computed at the stored point). ``why_invalid`` is the plan's reason: a pipeline
    failure (``PIPELINE_FAILURES``) raises instead of becoming an exclusion."""
    if invalid and why_invalid in PIPELINE_FAILURES:
        raise ValueError(f"pipeline failure, not an exclusion: {why_invalid}")
    flags = (invalid, gate3_failed, numerical, inertia_failed, kappa_clipped, off_plateau)
    for flag, reason in zip(flags, EXCLUSION_REASONS, strict=True):
        if flag:
            return reason
    return ""


def coverage(records: Iterable[dict]) -> dict:
    """Exclusions per (agent, snapshot) and outcome: ``records`` are dicts with agent,
    snapshot, outcome and reason ('' for a kept key). Returns, per 'agent@snapshot' and
    outcome, n, the excluded count per reason, the excluded share and whether it exceeds 30%
    (the agent is then descriptive for that outcome). A record whose reason is a pipeline
    failure (``PIPELINE_FAILURES``) raises: it is never counted as the network's exclusion."""
    out: dict[str, dict[str, dict]] = {}
    for r in records:
        if r.get("reason") in PIPELINE_FAILURES:
            raise ValueError(f"{r.get('agent')}@{r.get('snapshot')} {r.get('outcome')}: "
                             f"{r['reason']} is a pipeline failure, not an exclusion")
        who = f"{r['agent']}@{r['snapshot']}"
        c = out.setdefault(who, {}).setdefault(str(r["outcome"]), {"n": 0, "excluded": {}})
        c["n"] += 1
        if r.get("reason"):
            c["excluded"][r["reason"]] = c["excluded"].get(r["reason"], 0) + 1
    for per in out.values():
        for c in per.values():
            k = sum(c["excluded"].values())
            c["share_excluded"] = k / c["n"] if c["n"] else float("nan")
            c["descriptive"] = bool(c["share_excluded"] > COVERAGE_MAX)
    return out
