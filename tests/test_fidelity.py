"""Fidelity, reliability and phase metrics (audit/fidelity.py) on synthetic data with known
answers: hand-computed cases, exact identities, and population values for large samples."""

import math

import numpy as np
import pytest
from scipy.stats import spearmanr

from apexzero.audit import fidelity as F
from apexzero.audit.estimators import level_scores, window_stats
from apexzero.audit.rollouts import WindowRollouts

# --------------------------------------------------------------------------- windows


def test_window_plan_follows_the_plan_numbering():
    p = F.window_plan(8)
    assert list(p.operational) == [0, 1, 2, 3]            # windows 1-4
    assert list(p.op_target) == [4, 5, 6, 7]              # windows 5+
    assert list(p.hb_score) == [0, 2, 4, 6]               # odd windows 1, 3, 5, 7
    assert list(p.hb_target) == [1, 3, 5, 7]              # even windows 2, 4, 6, 8
    assert not set(p.operational) & set(p.op_target)
    assert not set(p.hb_score) & set(p.hb_target)
    with pytest.raises(ValueError):
        F.window_plan(5)


def test_odd_even_splits_by_position():
    a, b = F.odd_even([4, 5, 6, 7, 8])
    assert list(a) == [4, 6, 8] and list(b) == [5, 7]


# --------------------------------------------------------------------------- targets


def _returns():
    # [C=2, L=3, W=4]; target windows 0 and 2.
    return np.array([
        [[10, 99, 20, 99], [40, 0, 40, 0], [0.5, 9, 0.5, 9]],
        [[30, -9, 50, -9], [10, 0, 30, 0], [0.2, 9, 0.6, 9]],
    ], float)


def test_proxy_targets_hand_computed():
    t = F.regret_targets(_returns(), [0, 2], norm_floor=1.0)
    assert t.proxy
    assert np.allclose(t.j_pi, [[15, 40, 0.5], [40, 20, 0.4]])
    assert np.allclose(t.j_opt, [40, 40, 0.5])            # best mean over checkpoints
    assert np.allclose(t.reg, [[25, 0, 0], [0, 20, 0.1]])
    assert np.all(t.reg >= 0)
    assert np.array_equal(t.attained, [[False, True, True], [True, False, False]])
    assert np.allclose(t.reg_norm[:, :2], [[25 / 40, 0], [0, 0.5]])
    assert np.all(np.isnan(t.reg_norm[:, 2]))              # J_opt = 0.5 <= floor


def test_proxy_uses_only_the_target_windows():
    g = _returns()
    g2 = g.copy()
    g2[..., [1, 3]] = 1e6                                  # score windows: must not leak
    a, b = F.regret_targets(g, [0, 2]), F.regret_targets(g2, [0, 2])
    assert np.array_equal(a.reg, b.reg) and np.array_equal(a.j_opt, b.j_opt)


def test_proxy_is_a_lower_bound_on_any_window_return_max():
    rng = np.random.default_rng(0)
    g = rng.normal(50, 10, (5, 30, 16))
    t = F.regret_targets(g, range(16))
    assert np.all(t.j_opt <= g.max(axis=(0, 2)) + 1e-12)


def test_real_optimum_replaces_the_proxy():
    j_star = np.array([60.0, 45.0, 2.0])
    t = F.regret_targets(_returns(), [0, 2], j_opt=j_star)
    assert not t.proxy and not t.attained.any()
    assert np.allclose(t.reg, j_star - t.j_pi)
    assert np.allclose(t.reg_norm, t.reg / j_star)
    with pytest.raises(ValueError):
        F.regret_targets(_returns(), [0, 2], j_opt=np.ones(4))


# --------------------------------------------------------------------------- rank metrics


def test_spearman_matches_scipy_with_ties_and_batches():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 5, (3, 40)).astype(float)          # heavy ties
    y = x + rng.normal(size=(3, 40))
    ours = F.spearman(x, y)
    assert np.allclose(ours, [spearmanr(x[i], y[i])[0] for i in range(3)])
    assert np.isnan(F.spearman(np.ones(10), np.arange(10.0)))


def test_top_count_and_tie_weights():
    assert F.top_count(150, 0.1) == 15 and F.top_count(5, 0.1) == 1
    w = F.top_weights(np.array([5.0, 4, 3, 2, 1, 0, 0, 0, 0, 0]), 0.2)
    assert np.array_equal(w, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    w = F.top_weights(np.array([9.0, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 0.2)
    assert np.allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 0, 0, 0, 0, 0, 0])
    assert np.allclose(F.top_weights(np.zeros((4, 20)), 0.1).sum(-1), 2.0)


def test_regret_captured_hand_computed():
    reg = np.array([5.0, 3, 2, 0, 0, 0, 0, 0, 0, 0])
    assert np.isclose(F.regret_captured(reg, reg, 0.2), 0.8)          # oracle share
    assert np.isclose(F.regret_captured(-reg, reg, 0.2), 0.0)
    assert np.isclose(F.regret_captured(np.zeros(10), reg, 0.2), 0.2)  # all tied = random
    score = np.array([0.0, 9, 8, 0, 0, 0, 0, 0, 0, 0])                # picks levels 1 and 2
    assert np.isclose(F.regret_captured(score, reg, 0.2), 0.5)
    assert np.isnan(F.regret_captured(score, np.zeros(10), 0.2))


def test_regret_captured_random_scores_capture_frac_in_expectation():
    rng = np.random.default_rng(2)
    reg = rng.exponential(size=50)
    scores = rng.normal(size=(4000, 50))
    assert abs(F.regret_captured(scores, reg, 0.1).mean() - 0.1) < 0.005
    assert abs(F.regret_captured_lift(scores, reg, 0.1).mean()) < 0.02     # random -> 0


def test_regret_captured_lift_hand_computed():
    reg = np.array([5.0, 3, 2, 0, 0, 0, 0, 0, 0, 0])                   # k = 2, random = 0.2
    assert np.isclose(F.regret_captured_lift(reg, reg, 0.2), 1.0)     # oracle -> 1
    score = np.array([0.0, 9, 8, 0, 0, 0, 0, 0, 0, 0])                # RC 0.5
    assert np.isclose(F.regret_captured_lift(score, reg, 0.2), (0.5 - 0.2) / (0.8 - 0.2))
    assert np.isclose(F.regret_captured_lift(-reg, reg, 0.2), -0.2 / 0.6)
    near_uniform = np.array([1.0, 1.0, 1.0, 1.0, 1.1])               # oracle barely beats random
    assert F.regret_captured(np.arange(5.0)[::-1], near_uniform, 0.2) / \
        F.regret_captured(near_uniform, near_uniform, 0.2) > 0.9      # ratio flatters a bad score
    assert F.regret_captured_lift(np.arange(5.0)[::-1], near_uniform, 0.2) < 0   # lift does not
    assert np.isnan(F.regret_captured_lift(score, np.ones(10), 0.2))   # all equal: undefined


def test_precision_at_q():
    reg = np.arange(20.0)
    assert np.isclose(F.precision_at(reg, reg, 0.1), 1.0)
    assert np.isclose(F.precision_at(-reg, reg, 0.1), 0.0)
    assert np.isclose(F.precision_at(np.zeros(20), reg, 0.1), 0.1)    # all tied = random
    s = reg.copy()
    s[19] = -1                                                         # demote one of the top 2
    assert np.isclose(F.precision_at(s, reg, 0.1), 0.5)


def test_two_loading_known_answers():
    rng = np.random.default_rng(3)
    j_opt, j_pi = rng.normal(size=500), rng.normal(size=500)
    a, b, r12 = F.two_loading(j_opt, j_opt, j_pi)                     # tracks the optimum only
    assert np.isclose(a, 1.0) and np.isclose(b, 0.0, atol=1e-12)
    a, b, _ = F.two_loading(-j_pi, j_opt, j_pi)                       # tracks the agent only
    assert np.isclose(a, 0.0, atol=1e-12) and np.isclose(b, -1.0)
    big_opt, big_pi = rng.normal(size=20000), rng.normal(size=20000)
    a, b, r12 = F.two_loading(big_opt - big_pi, big_opt, big_pi)      # regret itself
    assert a > 0.6 and b < -0.6 and abs(a + b) < 0.03 and abs(r12) < 0.03
    a, b, r12 = F.two_loading(j_pi, j_pi, j_pi)                       # collinear: unidentified
    assert np.isnan(a) and np.isnan(b) and np.isclose(r12, 1.0)


# --------------------------------------------------------------------------- bootstrap tables


def _synthetic_targets(n, rho, seed):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.normal(size=(2, n))
    reg = np.exp(z1)                                     # monotone in z1: Spearman-invariant
    j_opt = 60.0 + rng.uniform(0, 20, n)
    j_pi = j_opt - reg
    return reg, j_opt, j_pi, reg / j_opt, rho * z1 + math.sqrt(1 - rho**2) * z2


def test_fidelity_table_perfect_score_and_population_rho():
    reg, j_opt, j_pi, reg_norm, noisy = _synthetic_targets(400, 0.6, seed=4)
    tab = F.fidelity_table({"oracle": reg, "noisy": noisy}, j_opt, j_pi, reg, reg_norm,
                           n_boot=500, seed=0, reference="noisy")
    o = tab.rows["oracle"]
    assert o["rho_abs"].value == pytest.approx(1.0) and o["rc_lift"].value == pytest.approx(1.0)
    assert o["p_at_q"].value == pytest.approx(1.0)
    rho_s = 6 / math.pi * math.asin(0.6 / 2)             # Spearman of a Gaussian pair
    n = tab.rows["noisy"]
    assert n["rho_abs"].lo < rho_s < n["rho_abs"].hi
    assert 0.08 < n["rho_abs"].hi - n["rho_abs"].lo < 0.25
    assert n["delta_rho_abs"].value == 0.0 and n["delta_rho_abs"].lo == n["delta_rho_abs"].hi == 0.0
    assert o["delta_rho_abs"].lo > 0                      # paired: oracle beats the noisy score
    assert tab.n_levels == 400 and tab.n_dropped == 0 and not tab.skipped


def test_fidelity_table_drops_undefined_targets_and_skips_nan_scores():
    reg, j_opt, j_pi, reg_norm, noisy = _synthetic_targets(100, 0.5, seed=5)
    reg_norm[:7] = np.nan
    bad = noisy.copy()
    bad[50] = np.nan
    tab = F.fidelity_table({"noisy": noisy, "bad": bad}, j_opt, j_pi, reg, reg_norm, n_boot=50)
    assert tab.n_levels == 93 and tab.n_dropped == 7 and tab.skipped == ["bad"]
    direct = F.spearman(noisy[7:], reg[7:])
    assert np.isclose(tab.rows["noisy"]["rho_abs"].value, direct)


def test_fidelity_table_is_deterministic_given_the_seed():
    args = _synthetic_targets(80, 0.4, seed=6)
    t1 = F.fidelity_table({"s": args[4]}, *args[1:3], args[0], args[3], n_boot=100, seed=9)
    t2 = F.fidelity_table({"s": args[4]}, *args[1:3], args[0], args[3], n_boot=100, seed=9)
    assert t1.rows == t2.rows


# --------------------------------------------------------------------------- reliability


def test_spearman_brown_and_window_budget():
    assert np.isclose(F.spearman_brown(0.5, 2), 2 / 3)
    assert np.isclose(F.spearman_brown(0.37, 1), 0.37)
    assert np.isclose(F.spearman_brown(F.spearman_brown(0.4, 2), 0.5), 0.4)
    assert np.isclose(F.windows_for_reliability(0.5, 10, 0.9), 90.0)
    n = F.windows_for_reliability(0.3, 16, 0.9)
    assert np.isclose(F.spearman_brown(0.3, n / 16), 0.9)
    assert np.isinf(F.windows_for_reliability(0.0, 16)) and np.isinf(F.windows_for_reliability(-0.2, 16))


def test_disattenuation():
    assert np.isclose(F.disattenuate(0.4, 0.64, 1.0), 0.5)
    assert np.isnan(F.disattenuate(0.4, 0.0, 0.9))


def _window_values(n_levels, n_windows, noise_var, seed):
    """Classical test theory: x_lw = mu_l + e_lw, var(mu) = 1, so a mean of n windows has
    reliability 1 / (1 + noise_var / n)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n_levels, 1))
    return mu + math.sqrt(noise_var) * rng.normal(size=(n_levels, n_windows))


def test_split_half_recovers_classical_reliability():
    x = _window_values(4000, 16, 4.0, seed=7)
    fn = lambda w: {"mean": x[:, w].mean(-1)}
    r = F.split_half(fn, range(16), method="pearson")["mean"]
    assert abs(r - 1 / (1 + 4 / 8)) < 0.02                       # 8 windows per half
    assert abs(F.spearman_brown(r, 2) - 1 / (1 + 4 / 16)) < 0.02  # stepped up to 16
    assert abs(F.windows_for_reliability(r, 8, 0.9) - 36) < 4     # 1 / (1 + 4/36) = 0.9
    rs = F.split_half(fn, range(16))["mean"]                      # Spearman of a Gaussian pair
    assert abs(rs - 6 / math.pi * math.asin(r / 2)) < 0.02


def test_tie_guard_rejects_mostly_constant_halves():
    rng = np.random.default_rng(13)
    a = np.zeros(100)
    b = np.zeros(100)
    a[:3], b[:3] = rng.uniform(0.1, 0.2, 3), rng.uniform(0.1, 0.2, 3)  # 97% tied at zero
    assert F.correlation(a, b) > 0.9                                   # spuriously "reliable"
    assert np.isnan(F.correlation(a, b, max_tie_share=0.8))
    assert F.tie_share(a) == pytest.approx(0.97) and F.tie_share(np.arange(4.0)) == 0.25
    x = rng.normal(size=100)
    assert np.isfinite(F.correlation(x, x + rng.normal(size=100), max_tie_share=0.8))
    rel = F.split_half(lambda w: {"s": np.where(np.arange(100) < 3, 1.0, 0.0) * len(w)},
                       range(8), max_tie_share=0.8)
    assert np.isnan(rel["s"])


def test_reliability_curve_matches_theory():
    x = _window_values(3000, 16, 4.0, seed=8)
    sizes = [1, 2, 4, 8]
    curve = F.reliability_curve(lambda w: {"mean": x[:, w].mean(-1)}, range(16), sizes,
                                n_splits=6, method="pearson")["mean"]
    assert np.allclose(curve, [1 / (1 + 4 / m) for m in sizes], atol=0.03)
    with pytest.raises(ValueError):
        F.reliability_curve(lambda w: {"m": x[:, w].mean(-1)}, range(16), [9])


def test_spearman_mde_formula_and_type_one_error():
    assert np.isclose(F.spearman_mde(1000), math.tanh(2.80158 * math.sqrt(1.06 / 997)), atol=1e-4)
    assert F.spearman_mde(150) > F.spearman_mde(1000)
    rng = np.random.default_rng(10)                               # the 1.06 / (n - 3) variance
    n = 200
    rho = F.spearman(rng.normal(size=(4000, n)), rng.normal(size=(4000, n)))
    z = np.arctanh(rho) / math.sqrt(1.06 / (n - 3))
    assert 0.035 < np.mean(np.abs(z) > 1.96) < 0.065


# --------------------------------------------------------------------------- phase and checks


def test_phase_bins():
    assert F.phase_of(0.49) == "early" and F.phase_of(0.5) == "mid"
    assert F.phase_of(0.899) == "mid" and F.phase_of(0.9) == "late"
    assert F.completion_rate(np.array([[True, False], [True, True]])) == 0.75


def test_value_calibration_detects_unscaled_values():
    rng = np.random.default_rng(11)
    g = rng.normal(60, 8, (300, 1)) + rng.normal(0, 3, (300, 12))
    v = g.mean(-1, keepdims=True) + rng.normal(0, 1, (300, 12))
    ok = F.value_calibration(v, g)
    assert abs(ok["ratio"] - 1) < 0.01 and abs(ok["slope"] - 1) < 0.05 and ok["r"] > 0.95
    sigma = 16.5
    bad = F.value_calibration(v / sigma, g)                       # forgot V = sigma * V_hat
    assert abs(bad["ratio"] - 1 / sigma) < 0.01


def test_distance_in_time():
    assert np.isclose(F.distance_in_time(np.full(100, 10.0), 2.0, 5.0), 50.0)
    v = np.where(np.arange(200) <= 50, 10.0, 20.0)               # 10 m/s to 100 m, then 20 m/s
    # 10 s to 100 m; the 100-102 m node pair averages 15 m/s (0.1333 s); then 20 m/s.
    expected = 102.0 + 20.0 * (12.0 - 10.0 - 2.0 * 2.0 / 30.0)
    assert np.isclose(F.distance_in_time(v, 2.0, 12.0), expected)
    assert np.isclose(F.distance_in_time(np.full(10, 10.0), 2.0, 60.0), 18.0)   # capped
    assert F.distance_in_time(np.full((3, 4, 10), 1.0), 2.0, 1.0).shape == (3, 4)


# --------------------------------------------------------------------------- with the estimators


def test_split_half_runs_on_level_scores_end_to_end():
    rng = np.random.default_rng(12)
    L, W, T = 12, 8, 260
    skill = rng.uniform(0.25, 0.35, (L, 1, 1))
    reward = np.abs(skill + rng.normal(0, 0.05, (L, W, T)))
    value = rng.normal(40, 3, (L, W, T))
    lengths = rng.integers(150, T + 1, (L, W))
    alive = (np.arange(T)[None, None, :] < lengths[..., None]).astype(float)
    ro = WindowRollouts(reward * alive, value, rng.normal(40, 1, (L, W, T, 5)), alive, lengths < T,
                        rng.normal(40, 3, (L, W)), rng.uniform(3000, 4000, (L, W)), np.zeros((L, W, T)))
    st = window_stats(ro, phases=rng.integers(0, 128, (L, W)))
    rel = F.split_half(lambda w: level_scores(st, w), range(W))
    assert {"pvl", "maxmc", "learnability_s1", "mna", "disagreement", "agent_only"} <= rel.keys()
    assert rel["agent_only"] > 0.5                                 # the planted level effect
    plan = F.window_plan(W)
    g = np.stack([st.G_disc, st.G_disc + 1.0])                     # two "checkpoints"
    t = F.regret_targets(g, plan.hb_target)
    tab = F.fidelity_table(level_scores(st, plan.hb_score), t.j_opt, t.j_pi[0], t.reg[0],
                           t.reg_norm[0], n_boot=50, reference="agent_only")
    assert np.allclose(t.reg[0], 1.0) and np.allclose(t.reg[1], 0.0)
    assert "pvl" in tab.rows and "delta_rho_abs" in tab.rows["pvl"]
