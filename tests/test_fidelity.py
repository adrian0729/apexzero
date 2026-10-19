"""Fidelity endpoints: Spearman rho and RC@q (tie-aware), with paired bootstrap CIs."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from apexzero.audit.fidelity import capture_weights, fidelity, regret_capture, spearman, top_k


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 200).astype(float)            # heavy ties
    y = x + rng.normal(0, 2, 200)
    assert np.isclose(spearman(x, y), spearmanr(x, y).statistic, atol=1e-12)
    rows = rng.normal(size=(3, 50))
    got = spearman(rows, y[:50][None].repeat(3, 0))
    assert np.allclose(got, [spearmanr(r, y[:50]).statistic for r in rows], atol=1e-12)
    assert np.isnan(spearman(np.ones(10), y[:10]))     # constant score: undefined, not 0


def test_regret_capture_by_hand():
    regret = np.array([5.0, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    assert top_k(10, 0.1) == 1
    assert np.isclose(regret_capture(regret, regret), 0.5)           # the ceiling
    assert np.isclose(regret_capture(-regret, regret), 0.0)          # picks a zero-regret level
    assert np.isclose(regret_capture(np.zeros(10), regret), 0.1)     # all tied: k/n
    tie = np.array([1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0])                 # two tied for one slot
    assert np.allclose(capture_weights(tie, 1), [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(regret_capture(tie, regret), (5 + 1) / 2 / 10)


def test_capture_weights_always_fill_exactly_k_slots():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 4, (20, 37)).astype(float)
    for k in (1, 4, 10, 37):
        w = capture_weights(s, k)
        assert np.allclose(w.sum(-1), k) and np.all((w >= 0) & (w <= 1))


def test_fidelity_table_and_paired_differences():
    rng = np.random.default_rng(2)
    regret = rng.gamma(1.0, 1.0, 400)
    scores = {"perfect": regret.copy(), "noisy": regret + rng.normal(0, 1, 400),
              "random": rng.random(400), "agent_only": regret + rng.normal(0, 0.5, 400)}
    fid = fidelity(scores, regret, n_boot=400, seed=0)
    assert np.isclose(fid["perfect"]["rho"], 1.0)
    assert np.isclose(fid["perfect"]["rc"], fid["oracle_regret"]["rc"])
    for v in fid.values():
        assert v["rho_lo"] <= v["rho"] + 1e-12 and v["rho"] <= v["rho_hi"] + 1e-12
        assert v["rc_lo"] <= v["rc"] + 1e-12 and v["rc"] <= v["rc_hi"] + 1e-12
    assert abs(fid["random"]["rho"]) < 0.15                    # n = 400: SD of rho ~ 0.05
    assert fid["random"]["rho_hi"] - fid["random"]["rho_lo"] < 0.3
    assert fid["agent_only"]["d_rho"] == 0.0
    assert fid["perfect"]["d_rho_lo"] > 0                      # beats the baseline, paired CI
    assert fid["noisy"]["d_rho_hi"] < 0                        # worse than it
    with pytest.raises(ValueError):
        fidelity({"bad": np.r_[np.inf, regret[1:]]}, regret, n_boot=10)
