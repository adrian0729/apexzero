"""Fidelity of a level score to true regret: the plan's co-primary endpoints (section 2.3).

- Spearman rho between score and regret (average ranks for ties).
- RC@q: the share of total regret held by the score's top q of levels. Scores tie often
  (learnability is 0 on every never- or always-solved level, MNA is 0 on unsolved ones), so
  the level set at the cut-off is not well defined. We use the *expected* capture under
  uniformly random tie-breaking: levels strictly above the cut-off count fully and the tied
  group at the cut-off shares the remaining slots equally. This is deterministic, and a
  constant score captures exactly k/n (about q, the random-selection value) instead of a
  number that depends on the level order.
- Percentile bootstrap over levels (paired: every score is resampled with the same level
  indices, so differences between scores get valid CIs too).
- The plan's 'uninformative' test compares a score with the *best* of the agent-only and
  policy-agnostic baselines (paired CI): per resample, the difference to the largest baseline
  rho in that resample. Its two-sided bootstrap p-value is what ``holm`` adjusts across the
  family of headline scores.

Both endpoints depend on the scores only through their order, so -inf (JaxUED's score for a
level on which no episode completed) is allowed and ranks below every finite score; NaN and
+inf are errors.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import rankdata


def top_k(n: int, frac: float) -> int:
    return max(1, round(frac * n))


def _rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    den = np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def spearman(score: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Spearman rho along the last axis (nan where either side is constant)."""
    return _rowwise_pearson(rankdata(score, axis=-1), rankdata(target, axis=-1))


def capture_weights(score: np.ndarray, k: int) -> np.ndarray:
    """Selection weight of each level in the top k by score, along the last axis (any leading
    axes are independent sets); the tied group at the cut-off shares the leftover slots."""
    s = np.asarray(score, float)
    thr = -np.sort(-s, axis=-1)[..., k - 1:k]
    above = s > thr
    tied = s == thr
    left = (k - above.sum(-1, keepdims=True)) / tied.sum(-1, keepdims=True)
    return above + tied * left


def regret_capture(score: np.ndarray, regret: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """RC@frac along the last axis: sum of regret over the score's top levels / total regret."""
    k = top_k(np.shape(score)[-1], frac)
    w = capture_weights(score, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * regret).sum(-1) / regret.sum(-1)


def rankable(name: str, score: np.ndarray) -> np.ndarray:
    """``score`` with -inf replaced by a finite value below every other score (ties kept), so
    rank statistics treat it as 'last'. NaN and +inf raise."""
    s = np.asarray(score, float)
    if np.any(np.isnan(s) | (s == np.inf)):
        raise ValueError(f"score {name!r} has NaN or +inf values")
    low = s == -np.inf
    if low.any():
        fin = s[~low]
        floor = fin.min() - 1.0 - abs(fin.min()) if fin.size else 0.0
        s = np.where(low, floor, s)
    return s


def boot_p(d: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for H0: d = 0, with the (count + 1) / (B + 1)
    correction so it is never 0."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    lo = (np.sum(d <= 0) + 1) / (d.size + 1)
    hi = (np.sum(d >= 0) + 1) / (d.size + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def fidelity(scores: dict[str, np.ndarray], regret: np.ndarray, frac: float = 0.1,
             n_boot: int = 2000, seed: int = 0,
             baselines: Sequence[str] = ("agent_only", "j_star", "bfs_steps"),
             level: float = 0.95) -> dict[str, dict[str, float | str]]:
    """rho and RC@frac with percentile-bootstrap CIs for every score, plus paired differences
    in rho: ``d_<b>`` against each baseline b present in ``scores``, and ``d_best`` against the
    best baseline (per resample; the point estimate uses the best point rho), with its
    bootstrap p-value ``p_best``. Also reports the regret-ranked ceiling under the key
    ``"oracle_regret"`` (RC of regret itself; its rho is 1 by definition)."""
    regret = np.asarray(regret, float)
    if not np.all(np.isfinite(regret)):
        raise ValueError("regret has non-finite values (mask the levels first)")
    n = len(regret)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    reg_b = regret[idx]
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci = lambda x: (float(np.nanpercentile(x, lo_q)), float(np.nanpercentile(x, hi_q)))
    rho_b = {}
    out = {}
    for name, s in {**scores, "oracle_regret": regret}.items():
        s = rankable(name, s)
        rb = spearman(s[idx], reg_b)
        cb = regret_capture(s[idx], reg_b, frac)
        rho_b[name] = rb
        r_lo, r_hi = ci(rb)
        c_lo, c_hi = ci(cb)
        out[name] = dict(rho=float(spearman(s, regret)), rho_lo=r_lo, rho_hi=r_hi,
                         rc=float(regret_capture(s, regret, frac)), rc_lo=c_lo, rc_hi=c_hi)
    base = [b for b in baselines if b in scores]
    if base:
        best_b = np.fmax.reduce([rho_b[b] for b in base])          # nan-ignoring max
        best = max(base, key=lambda b: out[b]["rho"])
        for name in scores:
            for b in base:
                d_lo, d_hi = ci(rho_b[name] - rho_b[b])
                out[name].update({f"d_{b}": out[name]["rho"] - out[b]["rho"],
                                  f"d_{b}_lo": d_lo, f"d_{b}_hi": d_hi})
            d = rho_b[name] - best_b
            d_lo, d_hi = ci(d)
            out[name].update(d_best=out[name]["rho"] - out[best]["rho"], d_best_lo=d_lo,
                             d_best_hi=d_hi, p_best=boot_p(d), best_baseline=best)
    return out
