"""Estimator unit tests from docs/paper/score-spec.md section 4.1, plus JaxUED parity."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxued.utils import positive_value_loss as ref_pvl

import apexzero.compat  # noqa: F401  (must precede jaxued)
from apexzero.audit.estimators import (
    level_scores,
    mna_terms,
    mna_terms_batch,
    scoring_gae,
    window_stats,
)
from apexzero.audit.rollouts import WindowRollouts

F = np.zeros
B = np.array


def test_E1_one_chunk_with_trailing_incomplete_episode():
    r, v = B([1.0, 0, 2, 5]), B([1.0, 2, 1, 3])
    term, trunc = B([0, 0, 1, 0], bool), F(4, bool)
    cut = B([0, 0, 0, 1], bool)                       # chunk ends after t = 3, b = V_4 = 4
    v_ext = np.r_[v, 4.0]
    a = scoring_gae(np.r_[r, 0], v_ext, np.r_[term, False], np.r_[trunc, False], 0.0,
                    np.r_[cut, False], 0.5, 0.5)[:4]
    assert np.allclose(a, [0.6875, -1.25, 1.0, 4.0])
    ep = a[:3]
    assert np.isclose(np.maximum(ep, 0).mean(), 0.5625)
    assert np.isclose(np.abs(ep).mean(), 0.979167, atol=1e-6)
    assert np.isclose(ep.mean(), 0.145833, atol=1e-6)
    assert np.isclose(max(0.0, r[:3].sum()) - v[:3].mean(), 1.666667, atol=1e-6)   # MaxMC
    assert np.isclose((1 + 0.5 * 0 + 0.25 * 2) - v[0], 0.5)                          # MaxMC-unit


def test_E2_chunk_cut_inside_the_episode():
    r, v = B([1.0, 0, 2, 5, 0]), B([1.0, 2, 1, 3, 4])
    term = B([0, 0, 1, 0, 0], bool)
    cut = B([0, 1, 0, 1, 0], bool)
    a = scoring_gae(r, v, term, F(5, bool), 0.0, cut, 0.5, 0.5)[:4]
    assert np.allclose(a, [0.625, -1.5, 1.0, 4.0])
    assert np.isclose(np.maximum(a[:3], 0).mean(), 0.541667, atol=1e-6)
    assert np.isclose(np.abs(a[:3]).mean(), 1.041667, atol=1e-6)


def test_E3_window_end_modes():
    r, v = B([1.0, 1, 1]), B([2.0, 2, 2])
    trunc = B([0, 0, 1], bool)
    aF = scoring_gae(r, v, F(3, bool), trunc, 2.0, F(3, bool), 0.5, 0.5, mode="F")
    aB = scoring_gae(r, v, F(3, bool), trunc, 2.0, F(3, bool), 0.5, 0.5, mode="B")
    assert np.allclose(aF, [-0.0625, -0.25, -1.0])
    assert np.allclose(aB, 0.0)


def test_E4_units_scaling_is_identity_after_sigma():
    sigma = 2.0
    v_hat = B([0.5, 1, 0.5, 1.5, 2.0])
    r = B([1.0, 0, 2, 5, 0])
    term = B([0, 0, 1, 0, 0], bool)
    cut = B([0, 0, 0, 1, 0], bool)
    a = scoring_gae(r, sigma * v_hat, term, F(5, bool), 0.0, cut, 0.5, 0.5)[:4]
    assert np.allclose(a, [0.6875, -1.25, 1.0, 4.0])


def test_E5_E6_maxmc_multi_episode_and_floor():
    eps = [([-1, -1], [1, 1]), ([3], [2]), ([0, 1, 0], [0, 0.5, 1]), ([0, 0], [1, 1])]
    G = [sum(r) for r, _ in eps]
    vbar = np.mean([np.mean(v) for _, v in eps])
    assert np.isclose(max(0, max(G)) - vbar, 1.875)
    eps = [([-1, -1], [1, 1]), ([-3], [2])]
    G = [sum(r) for r, _ in eps]
    vbar = np.mean([np.mean(v) for _, v in eps])
    assert np.isclose(max(0, max(G)) - vbar, -1.5)


def test_E8_maxmc_unit_has_no_floor():
    g, v0 = B([1.5, -2, 0.7]), B([1.0, 1.2, 0.8])
    assert np.isclose((g.max() - v0).mean(), 0.5)
    g, v0 = B([-2.0, -3]), B([0.0, 0])
    assert np.isclose((g.max() - v0).mean(), -2.0)


def test_E9_learnability_indicators():
    s1 = B([1, 0, 1, 1], bool)
    p = s1.mean()
    assert np.isclose(p * (1 - p), 0.1875)
    dist, d_qss = B([4000, 3900, 4100, 3950.0]), 4100.0
    s2 = s1 & (dist >= 0.97 * d_qss)
    assert list(s2) == [True, False, True, False]
    assert np.isclose(s2.mean() * (1 - s2.mean()), 0.25)


def test_E10_E11_E13_E14_mna_closed_form():
    r, v = B([0.0, 2, -1]), B([1.0, 0, 0])
    term = B([0, 0, 1], bool)
    g = mna_terms(r, v, term, F(3, bool), 0.0, F(3, bool), 1.0, 0.5)
    assert np.allclose(g, [0.75, 0.5, 1.0])
    assert np.isclose(g.sum() / 3, 0.75)
    g = mna_terms(r, v, term, F(3, bool), 0.0, B([0, 1, 0], bool), 1.0, 0.5)    # cut after t = 1
    assert np.allclose(g, [0.5, 0.0, 1.0])
    g = mna_terms(r, v, F(3, bool), B([0, 0, 1], bool), 2.0, F(3, bool), 1.0, 0.5, mode="B")
    assert np.allclose(g, [0.5, 0.0, 0.0])
    g = mna_terms(r, v, term, F(3, bool), 0.0, F(3, bool), 1.0, 1.0)
    assert np.allclose(g, [1.0, 1.0, 1.0])


def _mna_reference_port(rewards, values, dones, last_value, gamma, lambd):
    """Line-for-line float64 port of MNA's compute_clipped_gae(use_max_value=True) for one
    env (maze_plr.py L85-143 at the pinned commit), returning G_hat = -min(adv, 0)."""
    T = len(rewards)
    ext = np.r_[values, last_value]
    deltas = rewards + gamma * ext[1:] * (1 - dones) - ext[:-1]
    out = np.zeros(T)
    for s in range(T):
        td_total, gae_sum, is_done = 0.0, 0.0, 0.0
        for i in range(T - s):
            t = s + i
            td_total = td_total + gamma**i * deltas[t]
            clipped = min(td_total, 0.0)
            td_total = clipped
            gae_sum += (lambd**i * clipped * (1 - is_done)
                        + (lambd ** (i + 1)) / (1 - lambd) * clipped * dones[t] * (1 - is_done))
            is_done = max(is_done, dones[t])
        adv = (1 - lambd) * (gae_sum + (lambd ** (T - s)) / (1 - lambd) * td_total * (1 - is_done))
        out[s] = -min(adv, 0.0)
    return out


@pytest.mark.parametrize("lam", [0.9, 0.95, 0.98])
def test_E15_mna_closed_form_equals_reference_port(lam):
    rng = np.random.default_rng(int(lam * 100))
    for _ in range(5):
        T = 12
        r, v = rng.normal(size=T), rng.normal(size=T)
        term = np.zeros(T, bool)
        term[rng.integers(4, T)] = True
        end = int(np.argmax(term)) + 1
        ours = mna_terms(r[:end], v[:end], term[:end], np.zeros(end, bool), 0.0,
                         np.zeros(end, bool), 0.99, lam)
        ref = _mna_reference_port(r[:end], v[:end], term[:end].astype(float), 0.0, 0.99, lam)
        assert np.allclose(ours, ref, atol=1e-10)


def test_E16_disagreement_population_std():
    heads = B([[1.0, 2, 3, 4, 5], [2.0, 2, 2, 2, 2]])
    assert np.isclose(heads.std(-1).mean(), 0.707107, atol=1e-6)
    assert not np.isclose(heads.std(-1, ddof=1).mean(), 0.707107, atol=1e-3)


def test_E19_pvl_identity():
    rng = np.random.default_rng(3)
    r, v = rng.normal(size=50), rng.normal(size=50)
    a = scoring_gae(r, v, np.r_[np.zeros(49, bool), True], np.zeros(50, bool), 0.0,
                    np.zeros(50, bool), 0.995, 0.95)
    assert np.isclose(np.maximum(a, 0).mean(), 0.5 * np.abs(a).mean() + 0.5 * a.mean(), atol=1e-12)


def test_pvl_matches_jaxued_on_completed_episodes():
    rng = np.random.default_rng(0)
    T = 40
    r, v = rng.normal(size=T), rng.normal(size=T)
    dones = np.zeros(T, bool)
    dones[[9, 23, 31]] = True
    adv = np.zeros(T)
    start = 0
    for end in [10, 24, 32, T]:
        term = np.zeros(end - start, bool)
        if end in (10, 24, 32):
            term[-1] = True
        adv[start:end] = scoring_gae(r[start:end], v[start:end], term, np.zeros(end - start, bool),
                                     0.0, np.zeros(end - start, bool), 0.99, 0.95)
        start = end
    ref = float(ref_pvl(jnp.asarray(dones[:, None]), jnp.asarray(adv[:, None]))[0])
    ours = np.mean([np.maximum(adv[a:b], 0).mean() for a, b in [(0, 10), (10, 24), (24, 32)]])
    assert np.isclose(ours, ref, atol=1e-6)


def test_window_stats_and_level_scores_end_to_end():
    rng = np.random.default_rng(1)
    L, W, T = 3, 4, 300
    reward = np.abs(rng.normal(0.3, 0.05, (L, W, T)))
    value = rng.normal(40, 3, (L, W, T))
    lengths = rng.integers(150, T + 1, (L, W))
    alive = (np.arange(T)[None, None, :] < lengths[..., None]).astype(float)
    term = lengths < T
    ro = WindowRollouts(reward * alive, value, np.zeros((L, W, T, 0)), alive, term,
                        rng.normal(40, 3, (L, W)), rng.uniform(3000, 4000, (L, W)), np.zeros((L, W, T)))
    st = window_stats(ro, phases=rng.integers(0, 128, (L, W)))
    sc = level_scores(st, range(4), qss_distance=np.full(L, 3900.0))
    assert np.allclose(sc["pvl"], 0.5 * sc["l1"] + 0.5 * sc["mean_a"])     # E2 identity, aggregated
    assert np.all(sc["learnability_s1"] <= 0.25) and np.all(sc["learnability_s2"] <= 0.25)
    assert np.all(np.isfinite(sc["mna"]))


def test_scoring_gae_batches_over_leading_axes():
    rng = np.random.default_rng(5)
    N, T = 6, 40
    r, v, vf = rng.normal(size=(N, T)), rng.normal(size=(N, T)), rng.normal(size=(N, T))
    term, trunc, cut = rng.random((N, T)) < 0.05, rng.random((N, T)) < 0.05, rng.random((N, T)) < 0.05
    trunc &= ~term
    for mode in ("F", "B"):
        batched = scoring_gae(r, v, term, trunc, vf, cut, 0.99, 0.9, mode)
        for i in range(N):
            one = scoring_gae(r[i], v[i], term[i], trunc[i], vf[i], cut[i], 0.99, 0.9, mode)
            assert np.array_equal(batched[i], one)


@pytest.mark.parametrize("lam", [0.9, 0.95, 1.0])
def test_mna_terms_batch_equals_the_per_trajectory_closed_form(lam):
    # Several episodes and cuts per row, both modes; also the E-10/E-11/E-14 hand vectors.
    rng = np.random.default_rng(int(lam * 100) + 1)
    N, T = 7, 60
    r, v, vf = rng.normal(size=(N, T)), rng.normal(size=(N, T)), rng.normal(size=(N, T))
    term, trunc, cut = rng.random((N, T)) < 0.06, rng.random((N, T)) < 0.04, rng.random((N, T)) < 0.04
    trunc &= ~term
    for mode in ("F", "B"):
        batched = mna_terms_batch(r, v, term, trunc, vf, cut, 0.99, lam, mode, block=3)
        for i in range(N):
            one = mna_terms(r[i], v[i], term[i], trunc[i], vf[i], cut[i], 0.99, lam, mode)
            assert np.allclose(batched[i], one, rtol=0, atol=1e-12)
    r1, v1, t1 = B([[0.0, 2, -1]]), B([[1.0, 0, 0]]), B([[0, 0, 1]], bool)
    assert np.allclose(mna_terms_batch(r1, v1, t1, F((1, 3), bool), 0.0, F((1, 3), bool), 1.0, 0.5),
                       [[0.75, 0.5, 1.0]])
    assert np.allclose(mna_terms_batch(r1, v1, t1, F((1, 3), bool), 0.0, B([[0, 1, 0]], bool), 1.0, 0.5),
                       [[0.5, 0.0, 1.0]])
    assert np.allclose(mna_terms_batch(r1, v1, t1, F((1, 3), bool), 0.0, F((1, 3), bool), 1.0, 1.0),
                       [[1.0, 1.0, 1.0]])
