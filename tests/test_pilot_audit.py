"""Helpers of scripts/pilot_audit.py on hand-built inputs: crash-window means, baselines and
the reference row, window-budget medians and formatting, the Spearman-Brown deviation, and
the checkpoint cache key."""

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot_audit as P

from apexzero.audit import fidelity as F
from apexzero.audit.estimators import WindowStats


def test_window_mean_all_and_completed_only():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, -50.0, 7.0, -50.0]])
    done = np.array([[True, True, False, True], [True, False, True, False]])
    assert np.allclose(P._window_mean(x)([0, 1])["s"], [1.5, -22.5])
    assert np.allclose(P._window_mean(x, done)([0, 1, 2])["s"], [1.5, 6.0])
    assert np.isnan(P._window_mean(x, done)([1, 3])["s"][1])          # no completed window


def test_crash_split_uses_the_same_levels_for_both_columns():
    rng = np.random.default_rng(0)
    L, W = 60, 8
    x = rng.normal(size=(L, 1)) + 0.1 * rng.normal(size=(L, W))
    done = np.ones((L, W), bool)
    done[:5, 0::2] = False                                             # 5 levels: no completed odd window
    d = P.crash_split(x, done, range(W))
    assert d["n"] == L - 5 and d["n_total"] == L
    a, b = F.odd_even(range(W))
    keep = slice(5, None)
    assert d["all"] == pytest.approx(F.correlation(x[keep][:, a].mean(-1), x[keep][:, b].mean(-1)))


def _stats(g_disc, distance=None):
    L, W = g_disc.shape
    z = np.zeros((L, W))
    z3 = np.zeros((4, L, W))
    return WindowStats(T=np.full((L, W), 1200.0), G=g_disc.copy(), G_disc=g_disc, V_first=z,
                       V_bar=z, pvl=z3, l1=z3, mean_a=z3, pvl_disc=z, l1_disc=z, mna_sum=z,
                       dis=z, completed=np.ones((L, W), bool),
                       distance=np.full((L, W), 3000.0) if distance is None else distance)


def test_scores_for_baselines_and_reference_use_only_the_score_windows():
    rng = np.random.default_rng(1)
    g = rng.normal(50, 5, (2, 10, 6))                                  # [C, L, W]
    st = _stats(g[0])
    d_qss = rng.uniform(3000, 4000, 10)
    w = [0, 2, 4]
    s = P.scores_for(st, w, d_qss, g, 0, rng_seed=7)
    j_best = g[..., w].mean(-1).max(0)
    assert np.allclose(s["j_best"], j_best)
    assert np.allclose(s["qss_slow"], -d_qss)
    assert np.allclose(s[P.REFERENCE], j_best - g[0][:, w].mean(-1))
    assert np.allclose(s["agent_only"], -g[0][:, w].mean(-1))
    assert np.array_equal(s["random"], P.scores_for(st, w, d_qss, g, 0, rng_seed=7)["random"])
    g2 = g.copy()
    g2[..., [1, 3, 5]] = 1e6                                           # target windows: no leak
    s2 = P.scores_for(_stats(g2[0]), w, d_qss, g2, 0, rng_seed=7)
    assert all(np.allclose(s[k], s2[k]) for k in ("j_best", P.REFERENCE, "agent_only"))
    j_star = j_best + 3.0
    s3 = P.scores_for(st, w, d_qss, g, 0, rng_seed=7, j_opt=j_star)
    assert np.allclose(s3["j_best"], j_star) and np.allclose(s3[P.REFERENCE], j_star - g[0][:, w].mean(-1))


def test_median_budget_drops_withheld_and_counts_them():
    med, used, withheld = P.median_budget([10, np.nan, 30, np.nan, np.inf, 20])
    assert (used, withheld) == (4, 2) and med == 25.0
    assert P.median_budget([np.nan, np.nan])[1:] == (0, 2)


def test_window_formatting():
    assert P._windows(np.nan) == "-" and P._windows(np.inf) == "never"
    assert P._windows(np.inf, 16) == ">16" and P._windows(3.2) == "4"


def _res(curves, rel, sizes=(1, 2, 4, 8, 16)):
    return {"sizes": list(sizes), "curves": curves, "rel_s": rel, "rel_t": rel}


def test_budgets_and_sb_deviation_on_a_constructed_result():
    sizes = np.array([1, 2, 4, 8, 16])
    r16 = 0.8
    sb = F.spearman_brown(r16, sizes / 16)
    curves = {"pearson": {"l1": sb.copy()}, "spearman": {"l1": sb + np.array([0.3, 0.2, 0.1, 0.05, 0])}}
    rel = {"pearson": {"l1": r16}, "spearman": {"l1": r16}}
    r = _res(curves, rel)
    pb, ext, sbud = P.budgets(r, "l1", 16)
    assert ext and pb == pytest.approx(F.windows_for_reliability(0.8, 16)) and np.isinf(sbud)
    assert np.allclose(P.sb_deviation(r, "l1", "pearson", 16), 0.0)
    assert np.allclose(P.sb_deviation(r, "l1", "spearman", 16), [0.3, 0.2, 0.1, 0.05])
    assert np.all(np.isnan(P.sb_deviation(_res(curves, {"pearson": {"l1": np.nan},
                                                        "spearman": {}}), "l1", "pearson", 16)))


def test_checkpoint_cache_key_rejects_a_replaced_checkpoint(tmp_path):
    ck = tmp_path / "000000000001.pkl"
    ck.write_bytes(b"first")
    a = P.checkpoint_id(ck)
    ck.write_bytes(b"other")
    assert a != P.checkpoint_id(ck) and a["bytes"] == 5


def test_legacy_cache_needs_an_older_checkpoint(tmp_path):
    """A cache without the checkpoint hash is reused only when the checkpoint file predates it."""
    L, W = 3, 6
    st = _stats(np.ones((L, W)))
    ck = tmp_path / "000000000001.pkl"
    ck.write_bytes(b"ckpt")
    cache = tmp_path / "cache"
    cache.mkdir()
    base = {"levels": L, "windows": W, "seed": P.ROOT_SEED, "mode": "F"}
    out = cache / f"{ck.stem}.npz"
    np.savez_compressed(out, meta=json.dumps(base), **st._asdict(), sigma=1.0, timesteps=1,
                        wall=0.0, cpu=0.0)
    os.utime(ck, (1_000, 1_000))                                       # checkpoint older
    got, info = P.audit_checkpoint(ck, cache, None, None, np.zeros((L, 8)), W, None)
    assert info["legacy_cache"] and np.array_equal(got.G_disc, st.G_disc)
    os.utime(ck, None)                                                 # checkpoint now newer
    os.utime(out, (1_000, 1_000))
    with pytest.raises(pickle.UnpicklingError):                        # cache rejected: loads the ckpt
        P.audit_checkpoint(ck, cache, None, None, np.zeros((L, 8)), W, None)
