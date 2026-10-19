"""The in-training chunk accumulator (spec 0.4): hand vectors E-1/E-2/E-4 through the
scorer, JaxUED parity on concatenated per-chunk advantages (E-17), and equality with the
offline per-window audit under identical chunk cuts (E-18)."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxued.utils import accumulate_rollout_stats

import apexzero.compat  # noqa: F401  (restores jax.tree_map before jaxued runs)
from apexzero.audit.estimators import LAMBDAS, window_stats
from apexzero.audit.rollouts import WindowRollouts
from apexzero.ued.scoring import ChunkData, ChunkScorer, variant_key


def chunk(r, v, term, trunc=None, vf=None, b=None, heads=None, level=None, progress=None):
    """ChunkData from slot-major lists (one row per slot); returns time-major arrays."""
    r = np.atleast_2d(np.asarray(r, float))
    N, T = r.shape
    f = lambda a, fill=0.0: np.full((N, T), fill) if a is None else np.atleast_2d(np.asarray(a, float))
    term = np.atleast_2d(np.asarray(term, bool))
    trunc = np.zeros((N, T), bool) if trunc is None else np.atleast_2d(np.asarray(trunc, bool))
    hd = np.zeros((T, N, 0)) if heads is None else np.asarray(heads, float).transpose(1, 0, 2)
    lv = np.zeros((N, T), np.int32) if level is None else np.atleast_2d(level)
    return ChunkData(r.T, f(v).T, hd, term.T, trunc.T, f(vf).T,
                     np.zeros(N) if b is None else np.atleast_1d(np.asarray(b, float)), lv.T,
                     f(progress).T, np.zeros((T, N), bool))


def test_E1_one_chunk_through_the_scorer():
    sc = ChunkScorer(1, gamma=0.5, lambdas=(0.5,), headline_lambda=0.5, modes=("F",))
    rec, adv = sc.score(chunk([1.0, 0, 2, 5], [1.0, 2, 1, 3], [0, 0, 1, 0], b=4.0), sigma=1.0, chunk=0)
    assert np.allclose(adv[:, 0], [0.6875, -1.25, 1.0, 4.0])
    assert rec["slot"].tolist() == [0] and rec["t"].tolist() == [2] and rec["T"].tolist() == [3]
    assert rec["pvl"][0] == pytest.approx(0.5625)
    assert rec["l1"][0] == pytest.approx(0.979167, abs=1e-6)
    assert rec["mean_a"][0] == pytest.approx(0.145833, abs=1e-6)
    assert max(0.0, rec["G"][0]) - rec["V_bar"][0] == pytest.approx(1.666667, abs=1e-6)   # MaxMC
    assert rec["G_disc"][0] - rec["V_first"][0] == pytest.approx(0.5)                     # MaxMC-unit
    assert rec["crashed"].tolist() == [True] and rec["s1"].tolist() == [False]
    assert sc.acc.n[0] == 1                                # the step-3 episode carries over


def test_E2_cut_inside_the_episode_carries_the_sums():
    sc = ChunkScorer(1, gamma=0.5, lambdas=(0.5,), headline_lambda=0.5, modes=("F",))
    rec0, a0 = sc.score(chunk([1.0, 0], [1.0, 2], [0, 0], b=1.0), sigma=1.0, chunk=0)
    rec1, a1 = sc.score(chunk([2.0, 5], [1.0, 3], [1, 0], b=4.0), sigma=1.0, chunk=1)
    assert rec0["slot"].size == 0                          # nothing is scored at a chunk end
    assert np.allclose(np.r_[a0[:, 0], a1[:, 0]], [0.625, -1.5, 1.0, 4.0])
    assert rec1["T"].tolist() == [3] and rec1["chunk"].tolist() == [1]
    assert rec1["pvl"][0] == pytest.approx(0.541667, abs=1e-6)
    assert rec1["l1"][0] == pytest.approx(1.041667, abs=1e-6)
    assert rec1["G"][0] == 3.0 and rec1["V_first"][0] == 1.0      # MaxMC unchanged


def test_E4_sigma_rescales_values_and_bootstrap():
    sc = ChunkScorer(1, gamma=0.5, lambdas=(0.5,), headline_lambda=0.5, modes=("F",))
    rec, adv = sc.score(chunk([1.0, 0, 2, 5], [0.5, 1, 0.5, 1.5], [0, 0, 1, 0], b=2.0), sigma=2.0, chunk=0)
    assert np.allclose(adv[:, 0], [0.6875, -1.25, 1.0, 4.0])
    assert rec["pvl"][0] == pytest.approx(0.5625)


def test_E3_window_end_modes_and_variants():
    sc = ChunkScorer(1, gamma=0.5, lambdas=(0.5,), headline_lambda=0.5, modes=("F", "B"))
    rec, _ = sc.score(chunk([1.0, 1, 1], [2.0, 2, 2], [0, 0, 0], trunc=[0, 0, 1], vf=[0, 0, 2.0]),
                      sigma=1.0, chunk=0)
    assert rec[variant_key("l1", 0.5, "F")][0] == pytest.approx(0.4375)
    assert rec[variant_key("pvl", 0.5, "F")][0] == pytest.approx(0.0)
    assert rec[variant_key("l1", 0.5, "B")][0] == pytest.approx(0.0)
    assert rec["s1"].tolist() == [True]


def _random_stream(rng, N, T_total, p_done=0.03, E=0):
    r = rng.normal(0.3, 1.0, (N, T_total))
    v = rng.normal(5.0, 2.0, (N, T_total + 1))         # V_{T_total} = final chunk bootstrap
    vf = rng.normal(5.0, 2.0, (N, T_total))
    done = rng.random((N, T_total)) < p_done
    term = done & (rng.random((N, T_total)) < 0.5)
    trunc = done & ~term
    heads = rng.normal(5.0, 1.0, (N, T_total, E))
    return r, v, vf, term, trunc, heads


def test_E17_per_episode_scores_equal_jaxued_on_concatenated_chunk_advantages():
    rng = np.random.default_rng(0)
    N, CH, K = 5, 16, 8
    r, v, vf, term, trunc, _ = _random_stream(rng, N, CH * K, p_done=0.06)
    sc = ChunkScorer(N, gamma=0.99, lambdas=(0.95,), headline_lambda=0.95, modes=("F",))
    recs, advs = [], []
    for k in range(K):
        s = slice(k * CH, (k + 1) * CH)
        d = chunk(r[:, s], v[:, s], term[:, s], trunc[:, s], vf[:, s], b=v[:, (k + 1) * CH])
        rec, a = sc.score(d, sigma=1.0, chunk=k)
        recs.append(rec)
        advs.append(a)
    rec = {key: np.concatenate([x[key] for x in recs]) for key in recs[0]}
    adv = np.concatenate(advs)                                      # [T, N]
    dones = jnp.asarray((term | trunc).T)
    for stat, f in (("pvl", lambda a: np.maximum(a, 0)), ("l1", np.abs), ("mean_a", lambda a: a)):
        mean, mx, count = accumulate_rollout_stats(dones, jnp.asarray(f(adv)), time_average=True)
        for j in range(N):
            ours = rec[stat][rec["slot"] == j]
            assert ours.size == int(count[j])
            if ours.size:
                assert np.isclose(ours.mean(), float(mean[j]), atol=1e-5)
                assert np.isclose(ours.max(), float(mx[j]), atol=1e-5)
    assert rec["slot"].size > 10


def test_E18_online_accumulator_equals_offline_window_audit():
    rng = np.random.default_rng(1)
    CH, K, E = 128, 6, 3
    r, v, vf, term, trunc, heads = _random_stream(rng, 1, CH * K, p_done=0.008, E=E)
    sc = ChunkScorer(1, gamma=0.995)
    recs = []
    for k in range(K):
        s = slice(k * CH, (k + 1) * CH)
        d = chunk(r[:, s], v[:, s], term[:, s], trunc[:, s], vf[:, s], b=v[:, (k + 1) * CH],
                  heads=heads[:, s])
        recs.append(sc.score(d, sigma=1.0, chunk=k)[0])
    rec = {key: np.concatenate([x[key] for x in recs]) for key in recs[0]}

    # The same episodes as offline windows, phased against the chunk boundaries.
    ends = np.nonzero((term | trunc)[0])[0]
    starts = np.r_[0, ends[:-1] + 1]
    W, Tm = ends.size, int((ends - starts).max() + 1)
    ro = {k: np.zeros((1, W, Tm)) for k in ("reward", "value", "alive")}
    hd = np.zeros((1, W, Tm, E))
    for w, (a, b) in enumerate(zip(starts, ends, strict=True)):
        n = b - a + 1
        ro["reward"][0, w, :n], ro["value"][0, w, :n], ro["alive"][0, w, :n] = r[0, a:b + 1], v[0, a:b + 1], 1
        hd[0, w, :n] = heads[0, a:b + 1]
    wr = WindowRollouts(ro["reward"], ro["value"], hd, ro["alive"], term[0, ends][None],
                        vf[0, ends][None], np.zeros((1, W)), np.zeros((1, W, Tm)))
    phases = ((-starts) % CH)[None]
    assert rec["slot"].size == W
    for mode in ("F", "B"):
        st = window_stats(wr, gamma=0.995, mode=mode, phases=phases)
        for k, lam in enumerate(LAMBDAS):
            for stat, arr in (("pvl", st.pvl), ("l1", st.l1), ("mean_a", st.mean_a)):
                assert np.allclose(rec[variant_key(stat, lam, mode)], arr[k, 0], atol=1e-9), (stat, lam, mode)
    assert np.allclose(rec["G"], st.G[0]) and np.allclose(rec["G_disc"], st.G_disc[0])
    assert np.allclose(rec["V_first"], st.V_first[0]) and np.allclose(rec["V_bar"], st.V_bar[0])
    assert np.allclose(rec["dis"], st.dis[0]) and np.array_equal(rec["T"], st.T[0])
    assert np.array_equal(rec["s1"], st.completed[0])
    assert W >= 4 and (ends - starts).max() > CH            # some windows span chunks


def test_record_levels_distance_and_no_heads():
    sc = ChunkScorer(2, gamma=0.9)
    d = chunk([[1.0, 1, 1], [0.0, 0, 0]], [[0.0, 0, 0], [0.0, 0, 0]], [[0, 1, 0], [0, 0, 0]],
              level=[[7, 7, 3], [5, 5, 5]], progress=[[10.0, 20, 30], [1.0, 1, 1]])
    rec, _ = sc.score(d, sigma=1.0, chunk=4)
    assert rec["level"].tolist() == [7] and rec["distance"].tolist() == [30.0]
    assert np.isnan(rec["dis"][0]) and rec["chunk"].tolist() == [4]
    assert sc.acc.n.tolist() == [1, 3] and sc.acc.distance.tolist() == [30.0, 3.0]
