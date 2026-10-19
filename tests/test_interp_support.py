"""Training-support statistics (protocol 1.4, 2.3, 3.5): the generator copy, the QSS copy,
corner tables, l_typ, v_typ, regimes, R_clip and the kNN support density."""

import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import RunningMeanStd
from apexzero.interp import support as S
from apexzero.interp.constants import L_TYP_R_NODES, V_TYP_CHOICES_KPH
from apexzero.oracle.qss import GG, speed_profile, speed_profile_jax
from apexzero.tracks.generator import generate


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollouts, controls and stimuli arrays run in float32. tests/test_dynamics.py and
    tests/test_env.py switch x64 on globally when they are imported, so in a full-suite run
    this module switches it off for its own tests and restores it afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


from apexzero.vehicle.params import F1_2025

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base_grammar():
    return S.grammar(S.GRAMMARS["base"])


@pytest.fixture(scope="module")
def small_base_table():
    return S.corner_table(S.GRAMMARS["base"], n_strips=24)


def test_generate_with_corners_equals_generate(base_grammar):
    g, _ = base_grammar
    key = jax.random.PRNGKey(3)
    k1, wl1, wr1 = generate(key, g)
    k2, wl2, wr2, gc = S.generate_with_corners(key, g)
    for a, b in ((k1, k2), (wl1, wl2), (wr1, wr2)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The corner record describes the strip: each generated corner turns sign * angle.
    kap = np.asarray(k2, np.float64)
    s = np.arange(len(kap)) * 2.0
    for c in range(5):
        a0 = float(gc.start[c])
        sel = (s >= a0) & (s <= a0 + float(gc.length[c]))
        assert np.all(np.abs(kap[sel]) <= float(gc.k_peak[c]) * (1 + 1e-5))
        area = np.sum(kap[sel]) * 2.0
        assert area == pytest.approx(float(gc.sign[c] * gc.angle[c]), abs=0.08)
    assert np.all(np.asarray(gc.l_r) >= 10.0 - 1e-5)
    np.testing.assert_allclose(np.asarray(gc.length), np.asarray(gc.l_c + 2 * gc.l_r), rtol=1e-6)


def test_qss_copy_equals_repo_and_takes_a_start_speed(base_grammar):
    g, _ = base_grammar
    kap, _, _ = generate(jax.random.PRNGKey(5), g)
    ref = speed_profile_jax(kap, 2.0, F1_2025)
    np.testing.assert_array_equal(np.asarray(S.qss_profile_jax(kap, 2.0, F1_2025)), np.asarray(ref))
    v = np.asarray(S.qss_profile_jax(kap, 2.0, F1_2025, v_start=40.0))
    assert v[0] == pytest.approx(40.0)
    # Against the NumPy envelope with the same start speed (float32 vs float64).
    v_np = speed_profile(np.asarray(kap, float), 2.0, GG.from_params(F1_2025), closed=False,
                         v_start=40.0)
    np.testing.assert_allclose(v, v_np, rtol=2e-4)


def test_stimulus_v_peak_starts_from_v0():
    kap = np.zeros(2560)
    s = np.arange(2560) * 2.0
    kap[(s >= 1000) & (s < 1030)] = 1.0 / 13.0          # a plain 30 m arc at s = 1000
    v_np = speed_profile(kap[200:], 2.0, GG.from_params(F1_2025), closed=False, v_start=60.0)
    expect = v_np[: (1000 - 400) // 2 + 1].max()
    got = S.stimulus_v_peak(kap, 400.0, 60.0, 1000.0)
    assert got == pytest.approx(expect, rel=2e-4)
    # The road before s0 is irrelevant: a corner there does not change the answer.
    kap2 = kap.copy()
    kap2[(s >= 100) & (s < 120)] = 0.2
    assert S.stimulus_v_peak(kap2, 400.0, 60.0, 1000.0) == pytest.approx(got, rel=1e-6)
    batch = S.stimulus_v_peak_batch(np.stack([kap, kap]), np.array([400.0, 400.0]),
                                    np.array([60.0, 80.0]), np.array([1000.0, 1000.0]))
    assert batch[0] == pytest.approx(got) and batch[1] > batch[0]


def test_training_strips_follow_the_mixture():
    st = S.training_strips(S.GRAMMARS["v3mix"], n_strips=64)
    share = float(np.mean(np.asarray(st.use_alt)))
    assert 0.3 < share < 0.7
    base = S.training_strips(S.GRAMMARS["base"], n_strips=8)
    assert not np.any(np.asarray(base.use_alt))
    assert st.kappa.shape == (64, 4096) and st.v_qss.shape == (64, 4096)
    # Default keys are the root-seed support stream: reproducible.
    again = S.training_strips(S.GRAMMARS["base"], n_strips=8)
    np.testing.assert_array_equal(np.asarray(base.kappa), np.asarray(again.kappa))


def test_corner_table(small_base_table):
    t = small_base_table
    n = len(t["R_seg"])
    assert set(t) == set(S.CORNER_COLUMNS)
    assert n > 24 * 20 and all(len(v) == n for v in t.values())
    assert np.all(t["s_start"] < 8190.0 - 520.0)
    assert np.all(t["l_r"] >= 10.0 - 1e-4)
    # The exact samples never exceed the generated peak: R_eff >= 1 / k_peak of the corner.
    k_peak = t["onset_rate"] * t["l_r"]
    assert np.mean(t["R_eff"] >= 1.0 / k_peak * (1 - 1e-4)) > 0.9
    assert np.all(t["v_apex"] <= t["v_peak"] + 1e-6) and np.all(t["v_peak"] <= 87.6)
    assert np.all(t["R_seg"] >= t["R_eff"] * 0.999)          # the 10 m smoothing only widens
    assert set(np.unique(t["source"])) == {"base"} and np.all(t["strip_length"] == 7670.0)


def test_l_typ_table_and_interpolation():
    R = np.array([10.0, 10.5, 15.0, 15.5, 20.0, 25.0, 30.0, 50.0, 80.0, 90.0, 300.0])
    th = np.array([90.0, 90, 90, 90, 90, 90, 90, 90, 90, 90, 3.0])
    lr = np.array([20.0, 22, 30, 32, 40, 40, 44, 44, 50, 52, 10.0])
    t = {"R_seg": R, "theta_deg": th, "l_r": lr}
    lt = S.l_typ_table(t)
    assert lt.l_nodes == (21.0, 31.0, 40.0, 44.0, 51.0)       # the 3 deg kink is left out
    assert lt.l_nodes_all[-1] == 50.0 and lt.counts == (2, 2, 2, 2, 2)
    for r, l_ in zip(L_TYP_R_NODES, lt.l_nodes):
        assert S.l_typ(r, lt) == pytest.approx(l_)
    assert S.l_typ(5.0, lt) == 21.0 and S.l_typ(500.0, lt) == 51.0       # constant outside
    mid = math.exp(0.5 * (math.log(9.4) + math.log(14.6)))
    assert S.l_typ(mid, lt) == pytest.approx(26.0)                      # linear in log R
    with pytest.raises(ValueError):
        S.l_typ_table({"R_seg": np.array([10.0]), "theta_deg": np.array([90.0]),
                       "l_r": np.array([20.0])})


def test_v_typ_rounds_to_the_a2_choices():
    rng = np.random.default_rng(0)
    R = np.r_[np.full(30, 10.0), np.full(30, 13.0), np.full(5, 13.0), np.full(30, 20.0),
              np.full(30, 40.0), np.full(30, 80.0)]
    th = np.r_[np.full(30, 90.0), np.full(30, 90.0), np.full(5, 170.0), np.full(30, 150.0),
               np.full(30, 90.0), np.full(30, 90.0)]
    v = np.r_[np.full(30, 158 / 3.6), np.full(30, 168 / 3.6), np.full(5, 300 / 3.6),
              np.full(30, 240 / 3.6), np.full(30, 330 / 3.6), rng.uniform(60, 80, 30)]
    vt = S.v_typ_table({"R_seg": R, "theta_deg": th, "v_peak": v})
    assert vt.choice_kph["0|lt120"] == 160.0 and vt.choice_kph["1|lt120"] == 160.0
    assert "1|ge120" in vt.pooled and vt.choice_kph["1|ge120"] == 160.0  # 5 corners: pooled
    assert vt.choice_kph["2|ge120"] == 250.0 and vt.choice_kph["3|lt120"] == 250.0
    assert set(vt.choice_kph.values()) <= set(V_TYP_CHOICES_KPH)
    assert S.v_typ(10.0, 90.0, vt) == pytest.approx(160 / 3.6)
    assert S.theta_class(119.9) == "lt120" and S.theta_class(120.0) == "ge120"


def test_r_clip_is_the_worse_sign():
    mu = np.zeros(222)
    var = np.ones(222)
    k = S.KAPPA_CHANNELS
    mu[k] = np.linspace(-0.2, 0.2, 40)
    var[k] = np.linspace(0.3, 0.6, 40) ** 2
    rms = RunningMeanStd(jnp.asarray(mu, jnp.float32), jnp.asarray(var, jnp.float32), jnp.asarray(1.0))
    rc = S.r_clip(rms)
    sd = np.sqrt(var[k] + 1e-8)
    np.testing.assert_allclose(rc, 50.0 / (10.0 * sd - np.abs(mu[k])), rtol=1e-5)
    np.testing.assert_allclose(S.r_clip({"mean": mu, "var": var}), rc, rtol=1e-5)
    # Semantics: just below R_clip the worse-signed curvature clips, just above it does not.
    for j in (0, 39):
        sign = -np.sign(mu[k][j]) or 1.0
        for factor, clipped in ((0.999, True), (1.001, False)):
            obs = np.zeros(222, np.float32)
            obs[k[j]] = 50.0 * sign / (rc[j] * factor)
            raw = (obs[k[j]] - mu[k][j]) / sd[j]
            assert (abs(raw) >= 10.0) == clipped
    var[k[3]] = 1e-6
    assert np.isinf(S.r_clip({"mean": mu + 1.0, "var": var})[3])


@pytest.mark.skipif(not (REPO / "runs/dr-pilot-200M/ckpt/000200000000.pkl").exists(),
                    reason="G1 final snapshot not present")
def test_r_clip_of_the_finals_matches_the_protocol():
    """Reading obs_rms is not an evaluation (protocol 2.3): no forward pass, no rollout."""
    expect = {"runs/dr-pilot-200M/ckpt/000200000000.pkl": (6.4, 8.9),
              "runs/dr-pilot-v2-tilt30/ckpt/000199884800.pkl": (5.7, 7.6)}
    for path, (lo, hi) in expect.items():
        p = REPO / path
        if not p.exists():
            continue
        with open(p, "rb") as fh:
            rc = S.r_clip(pickle.load(fh)["obs_rms"])
        assert rc.min() == pytest.approx(lo, abs=0.05) and rc.max() == pytest.approx(hi, abs=0.05)


def test_regime_boundaries():
    rt = S.RegimeTable("G1", "base", q1=11.4, q10=20.2, r_clip=(6.4, 8.9), r_clip_max=8.9)
    assert [S.regime(r, rt) for r in (6.0, 8.9, 9.4, 11.4, 18.3, 20.2, 87.3)] == [
        "clip", "clip", "below", "tail", "tail", "bulk", "bulk"]
    no_clip = S.RegimeTable("G3", "v3mix", q1=9.4, q10=13.4)
    assert S.regime(6.0, no_clip) == "below"
    t = {"R_seg": np.linspace(1.0, 100.0, 100)}
    rt2 = S.regime_table("G1", S.GRAMMARS["base"], t, None)
    assert rt2.q1 == pytest.approx(np.quantile(t["R_seg"], 0.01)) and np.isnan(rt2.r_clip_max)


def test_knn_log_density():
    rng = np.random.default_rng(1)
    train = rng.uniform(0.0, 1.0, (20000, 5))
    q = rng.uniform(0.3, 0.7, (200, 5))
    ld = S.knn_log_density(train, q, np.ones(5))
    assert abs(np.median(ld)) < 0.15                 # unit density in the unit cube
    # Feature scaling: densities are per standardised unit, so scaling by sd cancels.
    ld2 = S.knn_log_density(train * 3.0, q * 3.0, np.full(5, 3.0))
    np.testing.assert_allclose(ld, ld2, rtol=1e-9, atol=1e-9)
    # exclude_self drops the zero distance of a training row to itself.
    self_in = S.knn_log_density(train[:50], train[:5], np.ones(5), k=3)
    self_ex = S.knn_log_density(train[:50], train[:5], np.ones(5), k=3, exclude_self=True)
    assert np.all(self_ex < self_in + 1e-12)
    far = S.knn_log_density(train, np.full((1, 5), 5.0), np.ones(5))
    assert far[0] < np.min(ld) - 5.0
    assert np.isnan(S.knn_log_density(train, np.array([[np.nan, 0, 0, 0, 0]]), np.ones(5))[0])
    f = S.support_features([10.0], [90.0], [20.0], [40.0], [50.0])
    assert f.shape == (1, 5) and f[0, 0] == pytest.approx(math.log(10.0))


def test_tight_fast_rate():
    t = {"source": np.array(["a", "a", "a", "b"]), "strip": np.array([0, 0, 1, -1]),
         "strip_length": np.array([50e3, 50e3, 50e3, 100e3]),
         "R_seg": np.array([10.0, 30.0, 12.0, 15.0]),
         "v_peak": np.array([260 / 3.6, 270 / 3.6, 200 / 3.6, 300 / 3.6])}
    # 2 tight-and-fast corners (strip a/0 and b) over 200 km.
    assert S.tight_fast_rate(t) == pytest.approx(1.0)
    assert S.tight_fast_rate(t, v_min_kph=0.0) == pytest.approx(1.5)


def test_agent_grammars_match_the_agent_registry():
    from apexzero.interp.agents import AGENTS

    for agent, g in S.AGENT_GRAMMAR.items():
        assert AGENTS[agent].grammar == S.GRAMMARS[g]


def test_build_and_load_support(tmp_path):
    rms = {"G1": ("dummy.pkl", {"mean": np.zeros(222), "var": np.full(222, 0.36)}),
           "S1": ("dummy-s1.pkl", {"mean": np.zeros(222), "var": np.full(222, 0.3)})}
    sup = S.build_support(tmp_path, rms, n_strips=16)
    assert (tmp_path / "support.json").exists() and (tmp_path / "corners_rdev.npz").exists()
    assert set(sup.regimes) == {"G1", "G2", "G3", "S1"}
    assert sup.regimes["G1"].r_clip_max == pytest.approx(50.0 / (10.0 * math.sqrt(0.36 + 1e-8)))
    assert np.isnan(sup.regimes["G2"].r_clip_max) and sup.regimes["S1"].grammar == ""
    rdev = sup.table("rdev")
    assert set(np.unique(rdev["source"])) == set(S.R_DEV)
    back = S.load_support(tmp_path)
    assert back.l_typ == sup.l_typ and back.v_typ == sup.v_typ
    assert back.feature_sd == sup.feature_sd and back.hashes == sup.hashes
    for a, r in sup.regimes.items():
        b = back.regimes[a]
        assert (b.q1, b.q10, b.r_clip, b.grammar) == (r.q1, r.q10, r.r_clip, r.grammar)
        assert b.r_clip_max == r.r_clip_max or (np.isnan(b.r_clip_max) and np.isnan(r.r_clip_max))
    assert len(sup.feature_sd) == len(S.SUPPORT_FEATURES) and all(x > 0 for x in sup.feature_sd)
