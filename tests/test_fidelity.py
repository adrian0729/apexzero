"""Fidelity endpoints: Spearman rho and RC@q (tie-aware), with paired bootstrap CIs."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from apexzero.audit.fidelity import (
    boot_p,
    capture_weights,
    fidelity,
    holm,
    rankable,
    regret_capture,
    spearman,
    top_k,
)


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


def test_capture_weights_work_along_the_last_axis_of_any_rank():
    # e.g. seeds x bootstrap resamples x levels for a two-way bootstrap.
    rng = np.random.default_rng(3)
    s = rng.integers(0, 5, (3, 4, 30)).astype(float)
    w = capture_weights(s, 6)
    assert w.shape == s.shape
    for i in range(3):
        for j in range(4):
            np.testing.assert_array_equal(w[i, j], capture_weights(s[i, j], 6))
    reg = rng.gamma(1.0, 1.0, s.shape)
    rc = regret_capture(s, reg, 0.2)
    assert rc.shape == (3, 4)
    assert np.isclose(rc[2, 1], regret_capture(s[2, 1], reg[2, 1], 0.2))


def test_minus_inf_ranks_last_and_nan_is_refused():
    s = np.array([0.3, -np.inf, 0.1, -np.inf, 2.0])
    r = rankable("s", s)
    assert np.all(np.isfinite(r)) and (r[1] == r[3]) and r[1] < 0.1
    np.testing.assert_array_equal(np.argsort(r, kind="stable"), np.argsort(s, kind="stable"))
    regret = np.array([1.0, 3.0, 2.0, 4.0, 0.5])
    fid = fidelity({"s": s}, regret, n_boot=50, baselines=())
    assert np.isclose(fid["s"]["rho"], spearmanr(r, regret).statistic)
    assert np.isclose(rankable("all", np.full(3, -np.inf)), 0.0).all()
    for bad in (np.nan, np.inf):
        with pytest.raises(ValueError):
            rankable("bad", np.r_[bad, 1.0])
    with pytest.raises(ValueError):
        fidelity({"s": s}, np.r_[np.nan, regret[1:]], n_boot=10)


def test_holm_and_bootstrap_p_by_hand():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])            # capped, monotone
    assert boot_p(np.ones(99)) == pytest.approx(2 / 100)                # never 0
    assert boot_p(np.r_[np.ones(50), -np.ones(50)]) == 1.0


def test_fidelity_table_and_paired_differences():
    rng = np.random.default_rng(2)
    regret = rng.gamma(1.0, 1.0, 400)
    scores = {"perfect": regret.copy(), "noisy": regret + rng.normal(0, 1, 400),
              "random": rng.random(400), "agent_only": regret + rng.normal(0, 0.5, 400),
              "j_star": -regret + rng.normal(0, 0.2, 400)}       # a strongly wrong-signed baseline
    fid = fidelity(scores, regret, n_boot=400, seed=0)
    assert np.isclose(fid["perfect"]["rho"], 1.0)
    assert np.isclose(fid["perfect"]["rc"], fid["oracle_regret"]["rc"])
    for v in fid.values():
        assert v["rho_lo"] <= v["rho"] + 1e-12 and v["rho"] <= v["rho_hi"] + 1e-12
        assert v["rc_lo"] <= v["rc"] + 1e-12 and v["rc"] <= v["rc_hi"] + 1e-12
    assert abs(fid["random"]["rho"]) < 0.15                    # n = 400: SD of rho ~ 0.05
    assert fid["random"]["rho_hi"] - fid["random"]["rho_lo"] < 0.3
    assert fid["agent_only"]["d_agent_only"] == 0.0
    assert fid["perfect"]["d_agent_only_lo"] > 0               # beats the baseline, paired CI
    assert fid["noisy"]["d_agent_only_hi"] < 0                 # worse than it
    # The best baseline is agent-only here (j_star has rho ~ -1), absent bfs_steps is skipped.
    assert fid["noisy"]["best_baseline"] == "agent_only" and "d_bfs_steps" not in fid["noisy"]
    assert fid["noisy"]["d_best"] == fid["noisy"]["d_agent_only"]
    assert fid["noisy"]["d_j_star_lo"] > 0                     # beats the wrong-signed one
    assert fid["perfect"]["d_best_lo"] > 0 and fid["perfect"]["p_best"] < 0.01
    assert fid["random"]["d_best_hi"] < 0 and fid["random"]["p_best"] < 0.01
    with pytest.raises(ValueError):
        fidelity({"bad": np.r_[np.inf, regret[1:]]}, regret, n_boot=10)
