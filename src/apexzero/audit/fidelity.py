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
"""

from __future__ import annotations

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
    """Selection weight of each level in the top k by score (rows = independent sets);
    the tied group at the cut-off shares the leftover slots."""
    s = np.atleast_2d(score)
    thr = -np.sort(-s, axis=-1)[:, k - 1:k]
    above = s > thr
    tied = s == thr
    left = (k - above.sum(-1, keepdims=True)) / tied.sum(-1, keepdims=True)
    w = above + tied * left
    return w.reshape(np.shape(score))


def regret_capture(score: np.ndarray, regret: np.ndarray, frac: float = 0.1) -> np.ndarray:
    """RC@frac along the last axis: sum of regret over the score's top levels / total regret."""
    k = top_k(np.shape(score)[-1], frac)
    w = capture_weights(score, k)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * regret).sum(-1) / regret.sum(-1)


def fidelity(scores: dict[str, np.ndarray], regret: np.ndarray, frac: float = 0.1,
             n_boot: int = 2000, seed: int = 0, baseline: str | None = "agent_only",
             level: float = 0.95) -> dict[str, dict[str, float]]:
    """rho and RC@frac with percentile-bootstrap CIs for every score, plus the paired
    difference in rho against ``baseline`` (if given). Also reports the regret-ranked ceiling
    under the key ``"oracle_regret"`` (RC of regret itself; its rho is 1 by definition)."""
    n = len(regret)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    reg_b = regret[idx]
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    ci = lambda x: (float(np.nanpercentile(x, lo_q)), float(np.nanpercentile(x, hi_q)))
    rho_b = {}
    out = {}
    for name, s in {**scores, "oracle_regret": regret}.items():
        s = np.asarray(s, float)
        if not np.all(np.isfinite(s)):
            raise ValueError(f"score {name!r} has non-finite values")
        rb = spearman(s[idx], reg_b)
        cb = regret_capture(s[idx], reg_b, frac)
        rho_b[name] = rb
        r_lo, r_hi = ci(rb)
        c_lo, c_hi = ci(cb)
        out[name] = dict(rho=float(spearman(s, regret)), rho_lo=r_lo, rho_hi=r_hi,
                         rc=float(regret_capture(s, regret, frac)), rc_lo=c_lo, rc_hi=c_hi)
    if baseline is not None and baseline in rho_b:
        for name in scores:
            d_lo, d_hi = ci(rho_b[name] - rho_b[baseline])
            out[name].update(d_rho=out[name]["rho"] - out[baseline]["rho"], d_rho_lo=d_lo,
                             d_rho_hi=d_hi)
    return out
