"""Fidelity of level scores against regret: the E1 audit metrics (plan sections 2.2-2.3).

The audit asks how well a level score S (PVL, MaxMC, learnability, ...) ranks levels by
regret, Reg = J_opt - J^pi. Everything here works on per-level arrays, so the same code
grades scores against *proxy* labels now and against TO-OCP labels (J*_20) later:

- **targets**: J^pi from cross-fitted target windows; J_opt is either the real J*_20 or,
  until labels exist, the proxy J_best (best mean return over the audited checkpoints);
  absolute and normalised regret;
- **fidelity**: Spearman rho against both regrets, RC@10%, P@q and the two-loading
  regression rank(S) ~ a rank(J_opt) + b rank(J^pi), with bootstrap CIs over levels
  (one shared resample for every score, so differences between scores are paired);
- **reliability**: split-half correlation of each score and of the target, Spearman-Brown
  step-up, disattenuation, and the window budget needed for a reliability goal;
- **phase**: window completion rate, binned as in the plan (<50%, 50-90%, >=90%).

Window conventions follow the plan's "Window assignment" (section 2.2), numbered from 1:
windows 1-4 feed the operational scores and windows 5+ the target; the high-budget scores
are cross-fitted on odd windows (score) against even windows (target). In 0-based array
indices the odd windows are 0, 2, 4, ...

All rank metrics act on the *last* axis and broadcast over leading axes, so a stack of
bootstrap resamples [B, L] is graded in one call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import NamedTuple

import numpy as np
from scipy.stats import norm, rankdata

N_OPERATIONAL = 4
PHASE_EDGES = (0.5, 0.9)            # window-completion bins: early | mid | late


# --------------------------------------------------------------------------- windows


class WindowPlan(NamedTuple):
    """0-based window indices for each role (plan 2.2, frozen in `audit-v1`)."""
    operational: np.ndarray   # windows 1-4: the operational (in-training) score
    op_target: np.ndarray     # windows 5+: the target for the operational score
    hb_score: np.ndarray      # odd windows (1, 3, 5, ...): the high-budget score
    hb_target: np.ndarray     # even windows (2, 4, 6, ...): its cross-fitted target


def window_plan(n_windows: int, n_operational: int = N_OPERATIONAL) -> WindowPlan:
    if n_windows < n_operational + 2:
        raise ValueError(f"need at least {n_operational + 2} windows, got {n_windows}")
    w = np.arange(n_windows)
    return WindowPlan(w[:n_operational], w[n_operational:], w[0::2], w[1::2])


def odd_even(windows: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Split a window set by position (1st, 3rd, ... vs 2nd, 4th, ...). Windows are i.i.d.
    draws, so any disjoint split is valid; alternating positions keep the halves equal."""
    w = np.asarray(list(windows))
    return w[0::2], w[1::2]


# --------------------------------------------------------------------------- targets


class Targets(NamedTuple):
    j_pi: np.ndarray       # [C, L] mean discounted return over the target windows
    j_opt: np.ndarray      # [L]    J*_20, or the proxy J_best
    reg: np.ndarray        # [C, L] J_opt - J^pi
    reg_norm: np.ndarray   # [C, L] Reg / J_opt (NaN where J_opt <= norm_floor)
    proxy: bool            # True when J_opt is the best observed mean return
    attained: np.ndarray   # [C, L] checkpoint c attains the proxy J_best (all False otherwise)


def regret_targets(g_disc: np.ndarray, windows: Iterable[int], j_opt: np.ndarray | None = None,
                   norm_floor: float = 1.0) -> Targets:
    """Regret targets for C checkpoints from discounted window returns ``g_disc`` [C, L, W].

    Without ``j_opt`` the optimum is the proxy J_best = max_c mean_w G^gamma over the *same*
    target windows. That makes it a certified lower bound on J*: every realised window is
    a feasible control sequence of the deterministic level (only policy sampling and
    observation noise differ between windows), so its return, and hence any mean of such
    returns, is at most the optimum. Two consequences of the proxy, both by construction:
    Reg >= 0, with Reg = 0 at the checkpoint that attains J_best (so the newest checkpoint's
    labels degenerate as training improves), and regret is measured against what *this run*
    reached, not against what is reachable. Taking J_best from the target windows (never the
    score windows) keeps the cross-fit clean: score and label share no window noise.

    Pass the real J*_20 [L] as ``j_opt`` once TO-OCP labels exist; nothing else changes.
    ``norm_floor`` (reward units) guards normalised regret: a level whose J_opt is at or below
    it (the best window crashed almost at once) has no meaningful fraction of optimum.
    """
    g = np.asarray(g_disc, float)
    if g.ndim != 3:
        raise ValueError(f"g_disc must be [C, L, W], got shape {g.shape}")
    w = np.asarray(list(windows))
    j_pi = g[..., w].mean(-1)
    if j_opt is None:
        opt = j_pi.max(0)
        proxy = True
        attained = j_pi == opt[None, :]
    else:
        opt = np.asarray(j_opt, float)
        if opt.shape != j_pi.shape[1:]:
            raise ValueError(f"j_opt must be [L] = {j_pi.shape[1:]}, got {opt.shape}")
        proxy = False
        attained = np.zeros(j_pi.shape, bool)
    reg = opt[None, :] - j_pi
    ok = opt > norm_floor
    reg_norm = np.where(ok[None, :], reg / np.where(ok, opt, 1.0)[None, :], np.nan)
    return Targets(j_pi, opt, reg, reg_norm, proxy, attained)


# --------------------------------------------------------------------------- rank metrics


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks along the last axis (ties share their mean rank, as in Spearman)."""
    return rankdata(np.asarray(x, float), axis=-1)


def _pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    den = np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    num = (x * y).sum(-1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho along the last axis (NaN when either side is constant)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    return _pearson(ranks(x), ranks(y))


def top_count(n: int, frac: float) -> int:
    """Size of the top ``frac`` of ``n`` levels (at least one; robust to 0.1 * 150 = 15.000...02)."""
    return max(1, math.ceil(frac * n - 1e-9))


def top_weights(score: np.ndarray, frac: float) -> np.ndarray:
    """Membership of each level in the score's top ``frac`` (along the last axis).

    Levels strictly above the cut get weight 1. Levels tied *at* the cut share the remaining
    slots equally, which is the expected membership under random tie-breaking; the weights
    therefore always sum to exactly k = top_count(L, frac), and heavily tied scores
    (learnability, MNA's zero gate) are graded without an arbitrary order.
    """
    s = np.asarray(score, float)
    k = top_count(s.shape[-1], frac)
    cut = -np.sort(-s, axis=-1)[..., k - 1:k]
    above = s > cut
    tied = s == cut
    n_above = above.sum(-1, keepdims=True)
    n_tied = tied.sum(-1, keepdims=True)
    return np.where(above, 1.0, np.where(tied, (k - n_above) / np.maximum(n_tied, 1), 0.0))


def regret_captured(score: np.ndarray, reg: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """RC@frac: share of the total regret held by the score's top ``frac`` of levels.

    A random score captures exactly k/L (= ``frac`` up to rounding) in expectation; a perfect
    one captures the oracle share ``regret_captured(reg, reg, frac)``, which is close to k/L
    when regret is spread evenly over levels. The raw share therefore cannot reach the
    plan's provisional thresholds (faithful >= 0.6) unless regret is concentrated, so
    ``regret_captured_lift`` is reported beside it. Assumes Reg >= 0 (true for the proxy;
    TO-OCP labels are lower bounds, so negative regret flags a label problem); NaN when the
    total regret is not positive.
    """
    reg = np.asarray(reg, float)
    w = top_weights(score, frac)
    tot = np.broadcast_to(reg, w.shape).sum(-1)
    num = (w * reg).sum(-1)
    return np.where(tot > 0, num / np.where(tot > 0, tot, 1.0), np.nan)


def regret_captured_lift(score: np.ndarray, reg: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """RC@frac rescaled so that a random score is 0 and the regret-sorted oracle is 1:
    (RC - k/L) / (RC_oracle - k/L). Dividing by the oracle share alone is misleading when
    regret is nearly uniform (random then scores ~k/L / RC_oracle, close to 1), which is
    exactly the proxy pilot's situation. NaN when the oracle cannot beat random (all regret
    equal)."""
    reg = np.asarray(reg, float)
    rc = regret_captured(score, reg, frac)
    n = np.shape(rc)
    L = np.broadcast_shapes(np.shape(score), reg.shape)[-1]
    base = top_count(L, frac) / L
    span = np.broadcast_to(regret_captured(reg, reg, frac), n) - base
    ok = span > 1e-12
    return np.where(ok, (rc - base) / np.where(ok, span, 1.0), np.nan)


def precision_at(score: np.ndarray, reg: np.ndarray, q: float = 0.1) -> np.ndarray:
    """P@q: share of the score's top-q levels that are also in the top-q by regret (ties at
    either cut weighted as in ``top_weights``). Random scores give q in expectation."""
    ws = top_weights(score, q)
    wr = top_weights(np.broadcast_to(np.asarray(reg, float), ws.shape), q)
    return (ws * wr).sum(-1) / top_count(ws.shape[-1], q)


def _zrank(x: np.ndarray) -> np.ndarray:
    r = ranks(x)
    r = r - r.mean(-1, keepdims=True)
    sd = r.std(-1, keepdims=True)
    return np.where(sd > 0, r / np.where(sd > 0, sd, 1.0), np.nan)


def two_loading(score: np.ndarray, j_opt: np.ndarray, j_pi: np.ndarray,
                max_collinearity: float = 0.999) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardised OLS of rank(S) on rank(J_opt) and rank(J^pi): returns (a, b, r12).

    Since Reg = J_opt - J^pi, a faithful score loads positively on the optimum and negatively
    on the agent's return (a > 0, b < 0); a score that only tracks how badly the agent does
    has a ~ 0, one that only tracks level "size" has b ~ 0. With z-scored ranks the normal
    equations are 2 x 2 with off-diagonal r12 = corr(rank J_opt, rank J^pi), returned as the
    collinearity diagnostic: the proxy J_best is nearly a copy of J^pi late in training, and
    (a, b) are unidentified as |r12| -> 1 (NaN beyond ``max_collinearity``).
    """
    shape = np.broadcast_shapes(np.shape(score), np.shape(j_opt), np.shape(j_pi))
    y, x1, x2 = (_zrank(np.broadcast_to(np.asarray(v, float), shape)) for v in (score, j_opt, j_pi))
    r12 = (x1 * x2).mean(-1)
    s1y = (x1 * y).mean(-1)
    s2y = (x2 * y).mean(-1)
    det = 1.0 - r12**2
    ok = np.abs(r12) < max_collinearity
    det = np.where(ok, det, np.nan)
    a = (s1y - r12 * s2y) / det
    b = (s2y - r12 * s1y) / det
    return a, b, r12


# --------------------------------------------------------------------------- bootstrap tables


class Estimate(NamedTuple):
    value: float
    lo: float
    hi: float


class FidelityTable(NamedTuple):
    rows: dict[str, dict[str, Estimate]]   # score -> metric -> estimate
    n_levels: int                          # levels graded (finite targets)
    n_dropped: int                         # levels dropped for non-finite targets
    skipped: list[str]                     # scores with non-finite values on graded levels


METRICS = ("rho_abs", "rho_norm", "rc", "rc_lift", "p_at_q", "a_opt", "b_pi")
PAIRED = ("rho_abs", "rho_norm", "rc_lift")


def fidelity_metrics(score, j_opt, j_pi, reg, reg_norm, frac: float = 0.1,
                     q: float = 0.1) -> dict[str, np.ndarray]:
    """Every fidelity metric of one score, along the last axis."""
    a, b, _ = two_loading(score, j_opt, j_pi)
    return {
        "rho_abs": spearman(score, reg),
        "rho_norm": spearman(score, reg_norm),
        "rc": regret_captured(score, reg, frac),
        "rc_lift": regret_captured_lift(score, reg, frac),
        "p_at_q": precision_at(score, reg, q),
        "a_opt": a,
        "b_pi": b,
    }


def bootstrap_indices(n_levels: int, n_boot: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_levels, (n_boot, n_levels))


def _estimate(point: float, samples: np.ndarray, alpha: float) -> Estimate:
    s = samples[np.isfinite(samples)]
    if not np.isfinite(point) or s.size < 10:
        return Estimate(float(point), np.nan, np.nan)
    lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(float(point), float(lo), float(hi))


def fidelity_table(scores: dict[str, np.ndarray], j_opt: np.ndarray, j_pi: np.ndarray,
                   reg: np.ndarray, reg_norm: np.ndarray, n_boot: int = 1000, seed: int = 0,
                   frac: float = 0.1, q: float = 0.1, alpha: float = 0.05,
                   reference: str | None = None) -> FidelityTable:
    """Point estimates and percentile bootstrap CIs over levels for every score.

    Every score is graded on the same resamples, so ``delta_*`` rows (score minus
    ``reference``, e.g. the agent-only baseline) carry paired CIs, as the plan's
    "not above the best baseline (paired CI)" rule requires. Levels with a non-finite target
    (normalised regret undefined) are dropped for all scores alike.
    """
    j_opt, j_pi, reg, reg_norm = (np.asarray(v, float) for v in (j_opt, j_pi, reg, reg_norm))
    keep = np.isfinite(j_opt) & np.isfinite(j_pi) & np.isfinite(reg) & np.isfinite(reg_norm)
    n = int(keep.sum())
    idx = bootstrap_indices(n, n_boot, seed)
    tg = [v[keep] for v in (j_opt, j_pi, reg, reg_norm)]
    tg_b = [v[idx] for v in tg]
    rows, skipped, boot = {}, [], {}
    for name, s in scores.items():
        s = np.asarray(s, float)[keep]
        if not np.all(np.isfinite(s)):
            skipped.append(name)
            continue
        point = fidelity_metrics(s, *tg, frac=frac, q=q)
        samples = fidelity_metrics(s[idx], *tg_b, frac=frac, q=q)
        boot[name] = samples
        rows[name] = {m: _estimate(float(point[m]), samples[m], alpha) for m in METRICS}
    if reference is not None and reference in rows:
        ref = {m: rows[reference][m].value for m in PAIRED}
        for name, row in rows.items():
            for m in PAIRED:
                d = boot[name][m] - boot[reference][m]
                row[f"delta_{m}"] = _estimate(row[m].value - ref[m], d, alpha)
    return FidelityTable(rows, n, int((~keep).sum()), skipped)


# --------------------------------------------------------------------------- reliability


def tie_share(x: np.ndarray) -> float:
    """Share of levels holding the most common value (1 for a constant score)."""
    _, counts = np.unique(np.asarray(x, float), return_counts=True)
    return float(counts.max() / counts.sum()) if counts.size else float("nan")


def correlation(x: np.ndarray, y: np.ndarray, method: str = "spearman",
                max_tie_share: float | None = None) -> float:
    """Correlation across levels over pairwise-finite entries (NaN if fewer than 3).

    With ``max_tie_share``, NaN when either side has more than that share of levels on one
    value: two mostly-zero halves (learnability when almost no window succeeds) agree on
    their ties and give a spuriously high split-half r that says nothing about the levels
    that matter."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    if max_tie_share is not None and max(tie_share(x[ok]), tie_share(y[ok])) > max_tie_share:
        return float("nan")
    f = spearman if method == "spearman" else _pearson
    return float(f(x[ok], y[ok]))


def spearman_brown(r, k):
    """Reliability of a measure made k times longer (k windows-multiples), from reliability r."""
    r, k = np.asarray(r, float), np.asarray(k, float)
    return k * r / (1.0 + (k - 1.0) * r)


def windows_for_reliability(r, n_windows, goal: float = 0.9):
    """Windows needed for reliability ``goal``, given reliability r measured at ``n_windows``
    (Spearman-Brown prophecy). Infinite when r <= 0: more windows of an unreliable score
    cannot be extrapolated. Assumes the score averages exchangeable windows; for max-based
    scores (MaxMC, the max-return term of MaxMC-unit) it is an approximation."""
    r = np.asarray(r, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = goal * (1.0 - r) / (r * (1.0 - goal))
    return np.where(r > 0, np.maximum(n_windows * k, 1.0), np.inf)


def disattenuate(rho, rel_x, rel_y):
    """Correlation corrected for unreliability of both sides: rho / sqrt(rel_x rel_y).
    The observed rho can reach at most sqrt(rel_x rel_y); NaN when a reliability is <= 0."""
    rho, rx, ry = (np.asarray(v, float) for v in (rho, rel_x, rel_y))
    den = np.sqrt(np.where((rx > 0) & (ry > 0), rx * ry, np.nan))
    return rho / den


ScoreFn = Callable[[np.ndarray], dict[str, np.ndarray]]


def split_half(fn: ScoreFn, windows: Iterable[int], method: str = "spearman",
               max_tie_share: float | None = None) -> dict[str, float]:
    """Odd/even split-half correlation across levels of every quantity ``fn`` returns.

    ``fn(window_indices)`` maps a window subset to per-level values (e.g. ``level_scores``
    or a target). The result is the reliability of a quantity measured on half the windows;
    ``spearman_brown(r, 2)`` steps it up to the full set. Spearman by default, matching the
    rank-based fidelity endpoints and robust to the crash-penalty tails of the returns.
    ``max_tie_share`` as in ``correlation``.
    """
    a, b = odd_even(windows)
    va, vb = fn(a), fn(b)
    return {k: correlation(va[k], vb[k], method, max_tie_share) for k in va}


def reliability_curve(fn: ScoreFn, windows: Iterable[int], sizes: Iterable[int],
                      n_splits: int = 20, seed: int = 0, method: str = "spearman",
                      max_tie_share: float | None = None) -> dict[str, np.ndarray]:
    """Empirical reliability against window count: for each size m, the mean correlation
    between ``fn`` on two disjoint random m-window subsets, over ``n_splits`` draws. Checks
    the Spearman-Brown prophecy directly instead of trusting it."""
    w = np.asarray(list(windows))
    sizes = list(sizes)
    if max(sizes) > len(w) // 2:
        raise ValueError(f"sizes must be <= {len(w) // 2} for {len(w)} windows")
    rng = np.random.default_rng(seed)
    acc: dict[str, list[list[float]]] = {}
    for j, m in enumerate(sizes):
        for _ in range(n_splits):
            p = rng.permutation(w)
            va, vb = fn(np.sort(p[:m])), fn(np.sort(p[m:2 * m]))
            for k in va:
                acc.setdefault(k, [[] for _ in sizes])[j].append(
                    correlation(va[k], vb[k], method, max_tie_share))
    return {k: np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vals])
            for k, vals in acc.items()}


def spearman_mde(n_levels: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest Spearman rho distinguishable from 0 (two-sided ``alpha``, given ``power``)
    with ``n_levels`` levels, using the Fieller-Hartley-Pearson variance of Fisher's z for
    Spearman, 1.06 / (n - 3). A difference between two independent rhos needs sqrt(2) more
    in z; within one audit the scores share levels, so the paired bootstrap is tighter."""
    z = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * math.sqrt(1.06 / (n_levels - 3))
    return math.tanh(z)


# --------------------------------------------------------------------------- phase and checks


def completion_rate(completed: np.ndarray) -> float:
    """Phase variable: the share of audited windows that end by truncation (no crash)."""
    return float(np.mean(completed))


def phase_of(completion: float) -> str:
    lo, hi = PHASE_EDGES
    return "early" if completion < lo else ("mid" if completion < hi else "late")


def value_calibration(v_first: np.ndarray, g_disc: np.ndarray) -> dict[str, float]:
    """Units check for critic values: per-level means over windows of V(o_0) against the
    discounted window return. In raw reward units (V = sigma * V_hat, spec DEV-6) the ratio
    of means and the slope of G on V sit near 1 for a calibrated critic; values left in
    normalised units would be off by the factor sigma. (V bootstraps beyond the window and
    G does not; with gamma = 0.995 the tail is gamma^1200 ~ 0.25% of the value.)"""
    v = np.asarray(v_first, float).mean(-1)
    g = np.asarray(g_disc, float).mean(-1)
    vc = v - v.mean()
    slope = float((vc * (g - g.mean())).sum() / (vc * vc).sum()) if np.any(vc) else float("nan")
    return {"mean_v": float(v.mean()), "mean_g": float(g.mean()),
            "ratio": float(v.mean() / g.mean()), "slope": slope,
            "r": correlation(v, g, method="pearson")}


def distance_in_time(v: np.ndarray, ds: float, seconds: float) -> np.ndarray:
    """Distance covered in ``seconds`` along speed profiles ``v`` [..., P] sampled every
    ``ds`` metres, with constant acceleration between nodes (dt = 2 ds / (v_i + v_i+1), as
    in the QSS lap time). Capped at the profile's length if it is shorter than ``seconds``."""
    v = np.asarray(v, float)
    flat = v.reshape(-1, v.shape[-1])
    s = np.arange(v.shape[-1]) * ds
    t = np.concatenate([np.zeros((len(flat), 1)),
                        np.cumsum(2.0 * ds / (flat[:, :-1] + flat[:, 1:]), axis=1)], axis=1)
    out = np.array([np.interp(seconds, ti, s) for ti in t])
    return out.reshape(v.shape[:-1])
