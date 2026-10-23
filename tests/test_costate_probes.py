"""Idea 2's probes (costate.probes; design v2.1, Sections 4-7), on random-init networks of G1's
architecture and on constructed data with known answers:

- the extractor: h1..h3, m, V and the heads against flax's ``Actor.apply`` / ``Critic.apply``
  (float32 here; float64 to 1e-10 in a subprocess, K3), heads paired by construction order,
  trained networks refused before the tag; PR-4's planted readout in h2 and h3;
- the two-penalty ridge against direct solves of its normal equations (alpha_H = inf: the
  baseline); the frozen folds (``states``' deal), strata subsets, and no held-out row in any
  fit or penalty choice; by-fold targets [F, n, m] trained and scored on slice f;
- R2_wd (within-direction SST for mirror-odd components), the direction-only floor, the
  cluster bootstrap against repeated rows, weighted Spearman;
- K4 plants on a synthetic world (48 geometries, mirror pairs, smooth trajectories): PR-1a
  (one unit per target), PR-1b inside C2 (the plant is recovered and C2 confirms), PR-6 (a
  decoy-only layer does not pass C2's residualised test; the units' biases), PR-2 (m in the
  layer adds nothing), the paired increment of K4's no-gain plants (a host's chance offset
  cannot fail one; a planted gain still does),
  PR-3 (no leak on the grouped split; the mis-grouped split leaks), PR-7 (scale invariance),
  PR-8 (C2-sw's planted switching unit);
- K5: permuted targets (wiring null: held-out rows kept, the training rows permuted across
  the training folds, mirror-odd columns within direction), the wiring check's 10-draw rule
  passing and failing on each planted fault (a fold's model on unpermuted targets, an odd
  column left unpermuted, test-set tuning), the control task's calibration and selectivity;
  the FWL two-stage check; the MLP probe; the decision bands;
- C2's control pool on a population of random networks whose gains vary: the two-level CI
  wider than the geometry-only one, a gain planted 0.08 over the population's median
  confirmed and one at the median not, the CI's coverage of the offset from the population's
  median near nominal (a Monte Carlo over fresh worlds and pools), the network's rank among
  the controls, and the pool and its resampling deterministic on their own stream.

No trained network is loaded or evaluated (design Section 1: nothing before the tag
costate-prereg-v1); only G1's config.json is read. NumPy float64 in this process (x64 stays
off); JAX float64 runs in its own process."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import Actor, Critic, PPOConfig, RunningMeanStd
from apexzero.costate import probes as P
from apexzero.costate import states as ST
from apexzero.costate.targets import K_P
from apexzero.env import racing
from apexzero.interp import io
from apexzero.interp.registry import AGENTS

SRC = Path(P.__file__).resolve().parents[2]
G1_RUN = io.REPO / AGENTS["G1"].run
HAVE_G1_CONFIG = (G1_RUN / "config.json").exists()
# G1's architecture (runs/dr-pilot-200M/config.json), used only if the run is not on disk
G1_LIKE = PPOConfig(network="mlp", actor_width=256, critic_width=256, mlp_layers=3,
                    activation="tanh", ensemble_heads=5)
GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)    # a coarser grid than the frozen one, for speed
H_GRID = (*GRID, np.inf)
NAMES = tuple(f"lam_{c}" for c in K_P)
RAW = tuple(f"raw_{c}" for c in K_P)


@pytest.fixture(scope="module")
def cfg() -> PPOConfig:
    return P.g1_config() if HAVE_G1_CONFIG else G1_LIKE


# ------------------------------------------------------------------------- the synthetic world


def _world(n_geo: int = 48, keys: int = 4, steps: int = 20, seed: int = 0):
    """A table like T's: n_geo geometries (R, theta), ``keys`` keys each in mirror pairs
    (direction +1 / -1, phase offset per pair), ``steps`` states per key along s - s_apex;
    eps [n, 4] smooth key-specific series independent of the decoys and of z's state part."""
    rng = np.random.default_rng(seed)
    R = np.round(np.exp(np.linspace(np.log(6.0), np.log(300.0), n_geo)), 2)
    th = np.array([45.0, 60.0, 90.0, 135.0, 180.0])[rng.integers(0, 5, n_geo)]
    cols = {k: [] for k in ("key_id", "pair_id", "R", "theta", "direction", "width",
                            "ds_apex", "vx", "t")}
    eps, kid = [], 0
    for g in range(n_geo):
        vmin = min(np.sqrt(9.8 * 2.0 * R[g]), 80.0)
        for j in range(keys):
            ds = np.linspace(-150.0, 100.0, steps) + 15.0 * (j // 2)
            vx = vmin + (85.0 - vmin) * (1.0 - np.exp(-(np.abs(ds) / 60.0) ** 2))
            for c, v in (("key_id", f"k{kid}"), ("pair_id", f"g{g}p{j // 2}"), ("R", R[g]),
                         ("theta", th[g]), ("direction", 1 if j % 2 == 0 else -1),
                         ("width", 12.0)):
                cols[c].append(np.full(steps, v))
            cols["ds_apex"].append(ds)
            cols["vx"].append(vx)
            cols["t"].append(5 * np.arange(steps))
            e = np.cumsum(rng.normal(0.0, 1.0, (steps, 4)), axis=0)
            eps.append(e / np.sqrt(np.arange(1, steps + 1))[:, None])
            kid += 1
    tab = {k: np.concatenate(v) for k, v in cols.items()}
    eps = np.concatenate(eps)
    n = len(eps)
    ds, vx, sig, lR = tab["ds_apex"], tab["vx"], tab["direction"], np.log(tab["R"])
    phi = np.stack([ds / 100, vx / 50, sig, np.tanh(ds / 60), sig * np.tanh(ds / 60), lR / 3,
                    tab["theta"] / 180, np.exp(-(ds / 50) ** 2),
                    sig * np.exp(-(ds / 50) ** 2) / np.sqrt(tab["R"] / 10)], 1)
    _, key_idx = np.unique(tab["key_id"], return_inverse=True)
    key_eff = rng.normal(0.0, 1.0, (kid, 3))[key_idx.ravel()]
    A = rng.normal(0.0, 0.5, (phi.shape[1] + 3, racing.OBS_DIM))
    z = np.hstack([phi, key_eff]) @ A + 0.3 * rng.normal(0.0, 1.0, (n, racing.OBS_DIM))
    z[:, -6:] = 0.37                                  # constant context channels: dropped
    z = np.clip(z, -10.0, 10.0)
    g = np.stack([np.exp(-((ds + 20) / 40) ** 2) * (1 + 0.3 * lR),
                  sig * np.tanh(ds / 30) * np.exp(-(ds / 120) ** 2),
                  sig * np.exp(-(ds / 25) ** 2) * (vx / 50),
                  np.tanh((ds + 60) / 20) * (vx / 80)], 1)
    return tab, z, eps, g + 0.25 * eps


@pytest.fixture(scope="module")
def world(cfg):
    tab, z, eps, lam = _world()
    w = ST.weights(tab)
    fo = P.folds(tab)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, w)
    trained = P.hidden(P.random_actor(cfg, 10), z)     # the network under test (random-init)
    rand = [P.hidden(P.random_actor(cfg, s), z)["h3"] for s in range(P.N_RANDOM_ACTORS)]
    return dict(tab=tab, z=z, eps=eps, lam=lam, w=w, fo=fo, res=res, h=trained, rand=rand,
                sig=tab["direction"], base=np.hstack([z, trained["m"]]))


@pytest.fixture(scope="module")
def c2run(world):
    """C2's probes on 8 target columns at once (lambda_res by fold, then raw lambda): the
    base [z, m], the 5 random-init actors, the network's own h3, a decoy layer (PR-6), a PR-1b
    plant at SNR 4 and PR-2's layer with m appended."""
    W = world
    Y = np.concatenate([W["res"].by_fold, np.broadcast_to(W["lam"], W["res"].by_fold.shape)], 2)
    names = NAMES + RAW
    run = lambda H: P.ridge_cv(W["base"], Y, W["fo"], W["w"], H, names, GRID, H_GRID)
    base = run(None)
    rand = [run(h) for h in W["rand"]]
    h3 = W["h"]["h3"]
    own = run(h3)
    tab = W["tab"]
    dec = P.decoy_units(np.stack([tab["ds_apex"], tab["vx"], W["sig"], np.log(tab["R"])], 1),
                        W["w"], np.random.default_rng(7), 32)
    decoy = run(np.hstack([h3, dec]))
    r_k = own.y[:, :4] - own.pred[:, :4]           # lambda_res residualised on the own baseline
    plant = run(P.plant_units(h3, r_k, 4.0, W["w"], np.random.default_rng(9)))
    with_m = run(np.hstack([h3, W["h"]["m"]]))
    share = ST.residual_share(W["res"].oof, W["lam"], W["w"], W["sig"], K_P)
    return dict(Y=Y, base=base, rand=rand, own=own, decoy=decoy, plant=plant, with_m=with_m,
                passing=np.r_[[ST.power_floor(share)[c] for c in K_P], np.zeros(4, bool)],
                raw_cols=np.r_[np.zeros(4, bool), np.ones(4, bool)])


@pytest.fixture(scope="module")
def f64_runs(tmp_path_factory):
    """K3's float64 extractor check in two x64 processes (G1's config with its 5 heads, and
    the head-free critic of the specialists), started early and read at the end."""
    if not HAVE_G1_CONFIG:
        return None
    out = tmp_path_factory.mktemp("probes")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    procs = {}
    for name, extra in (("g1", []), ("no_heads", ["--heads", "0"])):
        procs[name] = (out / f"{name}.json", subprocess.Popen(
            [sys.executable, "-m", "apexzero.costate.probes", "extractor-check", "--out",
             str(out / f"{name}.json"), "--run", str(G1_RUN), *extra], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    return procs


# ------------------------------------------------------------------------- extractor


def test_extractor_matches_flax_and_pairs_heads_by_construction_order(cfg, f64_runs):
    assert not jax.config.jax_enable_x64
    assert (cfg.network, cfg.actor_width, cfg.critic_width, cfg.mlp_layers, cfg.activation,
            cfg.ensemble_heads) == ("mlp", 256, 256, 3, "tanh", 5)
    z = np.clip(np.random.default_rng(0).normal(0, 2, (32, racing.OBS_DIM)), -10, 10)
    zf = jnp.asarray(z, jnp.float32)
    a, c = P.random_actor(cfg, 0), P.random_critic(cfg, 0)
    ha, hc = P.hidden(a, z), P.hidden(c, z)
    assert ha.out.dtype == np.float64 and [h.shape for h in ha.layers] == [(32, 256)] * 3
    mean = np.asarray(Actor(cfg, 2, False).apply(P.random_actor(cfg, 0).params, zf)[0])
    np.testing.assert_allclose(ha["m"], mean, atol=2e-5)
    v, heads = Critic(cfg).apply(P.random_critic_params(cfg, 0), zf, with_heads=True)
    np.testing.assert_allclose(hc["V"], np.asarray(v), atol=2e-5)
    assert hc.heads.shape == (32, 5) and hc.u.shape == (5, 32, 64)
    np.testing.assert_allclose(hc.heads, np.asarray(heads), atol=2e-5)
    # head i is (hidden Dense_{2i+2}, output Dense_{2i+1}), not the sorted names' order
    pairs = P.head_layers(c.params, cfg)
    tree = c.params["params"]
    for i, ((Wh, _), (Wo, _)) in enumerate(pairs):
        np.testing.assert_array_equal(Wh, tree[f"Dense_{2 * i + 2}"]["kernel"])
        np.testing.assert_array_equal(Wo, tree[f"Dense_{2 * i + 1}"]["kernel"])
    np.testing.assert_allclose(hc["u4"], np.maximum(hc["h3"] @ tree["Dense_10"]["kernel"]
                                                    + tree["Dense_10"]["bias"], 0))
    # the head-free critic of the specialists; the actor has no V; SimBa is refused
    c0 = P.random_critic(cfg.with_(ensemble_heads=0), 0)
    assert P.n_heads(c0.params) == 0 and P.hidden(c0, z).heads is None
    with pytest.raises(KeyError):
        ha["V"]
    with pytest.raises(ValueError):
        P.random_actor(cfg.with_(network="simba"), 0)
    # the normaliser: ppo.py's clip on the exact upcast of obs_rms
    rms = RunningMeanStd(jnp.full(222, 0.5, jnp.float32), jnp.full(222, 0.01, jnp.float32),
                         jnp.float32(10.0))
    o = np.random.default_rng(1).normal(0, 1, (4, 222))
    zz, clipped = P.normalise(o, rms)
    var = np.float64(np.float32(0.01))
    np.testing.assert_allclose(zz, np.clip((o - 0.5) / np.sqrt(var + 1e-8), -10, 10),
                               rtol=1e-12)
    assert clipped.any() and np.all(np.abs(zz[clipped]) == 10)


def test_trained_networks_wait_for_the_tag(cfg, monkeypatch):
    a = P.random_actor(cfg, 0)
    net = P.Net("actor", a.params, cfg, "G1@final", trained=True)
    with pytest.raises(PermissionError):
        P.hidden(net, np.zeros((1, racing.OBS_DIM)))
    with pytest.raises(ValueError):
        P.plant_readout(net, np.zeros(racing.OBS_DIM))
    monkeypatch.setattr(P, "_TRAINED_OK", True)      # what unlock_trained does after the tag
    assert P.hidden(net, np.zeros((1, racing.OBS_DIM))).out.shape == (1, 2)


def test_planted_readout_in_h2_and_h3_is_recovered(cfg, world):
    """PR-4: h2 holds tanh(z v + c), h3 holds it linearly; a probe on h2 alone or h3 alone
    recovers the readout's own R2 for its target within 0.02 (actor and critic trunk)."""
    W = world
    z, w = W["z"], W["w"]
    rng = np.random.default_rng(11)
    v = np.r_[rng.normal(0, 1, racing.OBS_DIM - 6), np.zeros(6)]
    v /= np.sqrt(ST.wvar(z @ v, w))
    r = np.tanh(z @ v + 0.3)
    y = r + 0.35 * np.sqrt(ST.wvar(r, w)) * rng.standard_normal(len(r))
    b = ST.wmean((r - ST.wmean(r, w)) * (y - ST.wmean(y, w)), w) / ST.wvar(r, w)
    r2_readout = 1 - ST.wvar(y - b * r, w) / ST.wvar(y, w)
    for net in (P.random_actor(cfg, 3), P.random_critic(cfg, 3)):
        hh = P.hidden(P.plant_readout(net, v, 0.3, eps=1e-4), z)
        np.testing.assert_allclose(hh["h2"][:, 0], r, atol=1e-7)
        np.testing.assert_allclose(hh["h3"][:, 0] / 1e-4, r, atol=1e-7)
        if net.kind == "actor":
            for L in ("h2", "h3"):
                got = P.probe_r2(P.ridge_cv(hh[L], y, W["fo"], w, None, ("y",), GRID, H_GRID),
                                 w)[0]
                assert abs(got - r2_readout) < 0.02, (L, got, r2_readout)


# ------------------------------------------------------------------------- ridge and folds


def _direct(X, Y, w, pb, ab, ah):
    """The two-penalty weighted ridge with an unpenalised intercept, by one dense solve."""
    wt = w / w.sum()
    Xa = np.hstack([np.ones((len(X), 1)), X])
    pen = np.r_[0.0, np.full(pb, ab), np.full(X.shape[1] - pb, ah)]
    keep = np.isfinite(pen)
    A = (Xa[:, keep] * wt[:, None]).T @ Xa[:, keep] + np.diag(pen[keep])
    coef = np.zeros((Xa.shape[1], Y.shape[1]))
    coef[keep] = np.linalg.solve(A, (Xa[:, keep] * wt[:, None]).T @ Y)
    return coef


def test_ridge_path_matches_direct_solves():
    rng = np.random.default_rng(0)
    n, pb, ph = 300, 7, 5
    X = rng.normal(0, 1, (n, pb + ph)) @ rng.normal(0, 1, (pb + ph, pb + ph)) * 0.4
    Y = X @ rng.normal(0, 1, (pb + ph, 3)) + rng.normal(0, 1, (n, 3))
    w = rng.uniform(0.2, 2.0, n)
    mu, yb, G, c = P._sums(X, Y, w).moments()
    a_b, a_h = (1e-3, 0.5, 30.0), (1e-2, 2.0, np.inf)
    beta = P.ridge_path(G, c, pb, a_b, a_h)
    for i, ab in enumerate(a_b):
        for j, ah in enumerate(a_h):
            d = _direct(X, Y, w, pb, ab, ah)
            np.testing.assert_allclose(beta[i, j], d[1:], atol=1e-10)
            pred = P._predict(X[:5], mu, yb, beta)[:, i, j]
            np.testing.assert_allclose(pred, d[0] + X[:5] @ d[1:], atol=1e-10)
    # inf on the hidden block is the base-only ridge; no base block: the hidden-only ridge
    np.testing.assert_array_equal(beta[:, 2, pb:], 0.0)
    only_h = P.ridge_path(G[pb:, pb:], c[pb:], 0, (np.nan,), (0.7, np.inf))
    np.testing.assert_allclose(only_h[0, 0], _direct(X[:, pb:], Y, w, 0, 0, 0.7)[1:], atol=1e-10)
    np.testing.assert_array_equal(only_h[0, 1], 0.0)


def test_folds_are_the_states_deal_and_strata_keep_them(world):
    W = world
    fo, tab = W["fo"], W["tab"]
    np.testing.assert_array_equal(fo.outer, ST.row_folds(tab))
    fo.check()
    np.testing.assert_array_equal(fo.cluster, ST.geometry_index(tab))
    assert fo.n_folds == 6 and len(np.unique(fo.cluster)) == 48
    assert all(len(np.unique(fo.cluster[fo.outer == f])) == 8 for f in range(6))
    # mirror partners share a fold; the inner folds are 5 over the training geometries
    for pair in np.unique(tab["pair_id"])[:20]:
        assert len(np.unique(fo.outer[tab["pair_id"] == pair])) == 1
    for f in range(6):
        tr = fo.outer != f
        assert set(np.unique(fo.inner[f][tr])) == set(range(5))
        for g in np.unique(fo.cluster[tr]):
            assert len(np.unique(fo.inner[f][fo.cluster == g])) == 1
    # a stratum keeps T's folds (never re-dealt on its own rows)
    m = tab["ds_apex"] < -40
    sub = fo.subset(m)
    np.testing.assert_array_equal(sub.outer, fo.outer[m])
    np.testing.assert_array_equal(sub.inner, fo.inner[:, m])
    with pytest.raises(ValueError):
        P.Folds(fo.outer, np.zeros_like(fo.inner), fo.cluster).check()
    # a different partition deals differently
    assert not np.array_equal(P.folds(tab, 1).outer, fo.outer)


def test_held_out_rows_never_enter_a_fit_or_a_penalty_choice(world):
    W = world
    fo, w = W["fo"], W["w"]
    Y = W["lam"][:, :2]
    base = W["z"][:, :40]
    H = W["h"]["h3"][:, :30]
    pr = P.ridge_cv(base, Y, fo, w, H, None, GRID, H_GRID)
    # targets as [n, m]: changing fold 2's targets leaves fold 2's predictions unchanged
    Y2 = Y.copy()
    held = fo.outer == 2
    Y2[held] = np.random.default_rng(0).normal(0, 100, (held.sum(), 2))
    pr2 = P.ridge_cv(base, Y2, fo, w, H, None, GRID, H_GRID)
    np.testing.assert_allclose(pr2.pred[held], pr.pred[held], atol=1e-10)
    assert not np.allclose(pr2.pred[~held], pr.pred[~held])
    # targets by fold: row f of the array is only scored on fold f's rows
    Yf = np.array(np.broadcast_to(Y, (6, *Y.shape)))
    Yf[2][held] = 1e6
    pr3 = P.ridge_cv(base, Yf, fo, w, H, None, GRID, H_GRID)
    np.testing.assert_allclose(pr3.pred, pr.pred, atol=1e-10)
    assert np.all(pr3.y[held] == 1e6) and np.array_equal(pr3.y[~held], pr.y[~held])
    # a column constant on the training rows (the context channels) is dropped
    assert np.all(pr.kept[:, 0] <= 40) and pr.alpha.shape == (6, 2, 2)
    assert P.ridge_cv(W["z"], Y, fo, w, None, None, GRID, H_GRID).kept[0, 0] == 216


def test_by_fold_targets_train_and_score_fold_f_on_slice_f(world):
    """Targets [F, n, m] (C2's cross-fitted lambda_res): while fold f is held out the probe
    trains on slice f's training rows and scores fold f's rows against slice f. With slice f
    a distinct known function of f, fold f's predictions, penalties and scored targets are
    those of the same probe fitted on slice f alone (K4's PR-1b uses one array, so this is
    the by-fold path's own check)."""
    W = world
    fo, w = W["fo"], W["w"]
    y0 = W["lam"][:, :2]
    base, H = W["z"][:, :40], W["h"]["h3"][:, :30]
    Yf = np.stack([(1.0 + f) * y0 + 10.0 * f for f in range(fo.n_folds)])
    pr = P.ridge_cv(base, Yf, fo, w, H, None, GRID, H_GRID)
    for f in range(fo.n_folds):
        held = fo.outer == f
        one = P.ridge_cv(base, Yf[f], fo, w, H, None, GRID, H_GRID)
        np.testing.assert_allclose(pr.pred[held], one.pred[held], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(pr.alpha[f], one.alpha[f])
        np.testing.assert_array_equal(pr.y[held], Yf[f][held])
        # and not slice f + 1's: the folds' slices differ by construction
        assert not np.allclose(pr.pred[held], P.ridge_cv(
            base, Yf[(f + 1) % fo.n_folds], fo, w, H, None, GRID, H_GRID).pred[held])


def test_decoy_units_biases():
    """PR-6's units: tanh(gain (Ds A + b)), b ~ N(0, bias^2) drawn after A; bias 0 gives odd
    functions of the centred decoys, a bias does not."""
    rng = np.random.default_rng(0)
    D, w = rng.normal(size=(200, 4)), np.full(200, 1 / 200)
    Ds = P.scaler(D, w).apply(D)
    g = np.random.default_rng(5)
    A = g.normal(0.0, 0.5, (4, 8))
    b = g.normal(0.0, 1.0, 8)
    np.testing.assert_allclose(P.decoy_units(D, w, np.random.default_rng(5), 8, 2.0),
                               np.tanh(2.0 * (Ds @ A + b)), atol=1e-14)
    U0 = P.decoy_units(D, w, np.random.default_rng(5), 8, 2.0, bias=0.0)
    np.testing.assert_allclose(U0, np.tanh(2.0 * Ds @ A), atol=1e-14)
    np.testing.assert_allclose(P.decoy_units(-D, w, np.random.default_rng(5), 8, 2.0, 0.0), -U0,
                               atol=1e-14)
    assert not np.allclose(P.decoy_units(-D, w, np.random.default_rng(5), 8, 2.0),
                           -P.decoy_units(D, w, np.random.default_rng(5), 8, 2.0))


# ------------------------------------------------------------------------- metrics, bootstrap


def test_direction_aware_r2_and_the_direction_floor(world):
    W = world
    w, sig = W["w"], W["sig"]
    y = 0.8 * sig + W["lam"][:, 1]                       # a mirror-odd target with a sign
    dmean = {d: ST.wmean(y[sig == d], w[sig == d]) for d in (-1, 1)}
    pred_dir = np.where(sig > 0, dmean[1], dmean[-1])
    raw, wd = P.r2(y, pred_dir, w, None, ("lam_n",))[0], P.r2(y, pred_dir, w, sig, ("lam_n",))[0]
    assert raw > 0.5 and abs(wd) < 1e-12               # direction alone earns nothing in R2_wd
    # an even component is scored raw either way; R2_wd's SST is about the direction means
    assert P.r2(y, pred_dir, w, sig, ("lam_vx",))[0] == pytest.approx(raw)
    p = pred_dir + 0.3 * W["lam"][:, 1]
    sst_wd = ST.wmean((y - pred_dir) ** 2, w)
    assert P.r2(y, p, w, sig, ("lam_n",))[0] == pytest.approx(
        1 - ST.wmean((y - p) ** 2, w) / sst_wd, rel=1e-12)
    assert P.is_odd("lam_xi") and P.is_odd("sigma_steer") and P.is_odd("lam_pre_delta")
    assert not P.is_odd("lam_vx") and not P.is_odd("Pi_in") and not P.is_odd(None)
    # the direction-only floor out of fold
    Y = np.stack([W["lam"][:, 0], y], 1)
    dp = P.direction_only(Y, W["fo"], w, sig, ("lam_s", "lam_n"))
    r_wd, r_raw = P.probe_r2(dp, w, sig), P.probe_r2(dp, w)
    assert r_raw[1] > 0.5 and abs(r_wd[1]) < 0.02 and r_raw[0] == pytest.approx(r_wd[0])
    rf = P.fold_r2(dp, W["fo"], w, sig)
    assert rf.shape == (6, 2)


def test_cluster_bootstrap_equals_repeated_rows():
    rng = np.random.default_rng(0)
    n, G = 120, 12
    cl = rng.integers(0, G, n)
    d = rng.choice([-1, 1], n)
    y = rng.normal(0, 1, (n, 2)) + d[:, None] * np.array([0.0, 2.0])
    preds = np.stack([y + rng.normal(0, s, (n, 2)) for s in (0.5, 1.0)])
    w = rng.uniform(0.5, 1.5, n)
    names = ("lam_vx", "lam_n")
    bs = P.boot_sums(y, preds, w, cl, d, names)
    r0 = P.boot_r2(bs, np.ones((1, G)))[0]
    for k in range(2):
        np.testing.assert_allclose(r0[k], P.r2(y, preds[k], w, d, names), rtol=1e-12)
    counts = P.cluster_counts(G, 5, np.random.default_rng(3))
    assert counts.shape == (5, G) and np.all(counts.sum(1) == G)
    rb = P.boot_r2(bs, counts)
    for b in range(5):
        rows = np.concatenate([np.flatnonzero(cl == g) for g in range(G)
                               for _ in range(counts[b, g])])
        for k in range(2):
            np.testing.assert_allclose(rb[b, k], P.r2(y[rows], preds[k][rows], w[rows], d[rows],
                                                      names), rtol=1e-10)
    lo, hi = P.ci(rb[:, 0])
    assert np.all(lo <= hi)
    assert np.array_equal(P.cluster_counts(G, 5, P.boot_rng(1)),
                          P.cluster_counts(G, 5, P.boot_rng(1)))


def test_weighted_spearman():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, 50).astype(float)          # ties
    y = x + rng.normal(0, 3, 50)
    assert P.wspearman(x, y) == pytest.approx(spearmanr(x, y)[0], abs=1e-12)
    k = rng.integers(1, 4, 50)                          # integer weights = repeated rows
    assert P.wspearman(x, y, k.astype(float)) == pytest.approx(
        spearmanr(np.repeat(x, k), np.repeat(y, k))[0], abs=1e-12)


# ------------------------------------------------------------------------- K4 plants


def test_planted_units_are_recovered_pr1a(cfg):
    """PR-1a: random-init h3 plus one unit per target, targets independent of z and of each
    other (i.i.d. here), SNR 0.25, 1 and 4: held-out R2 and Delta2 within 0.03 of
    SNR / (1 + SNR), on a T-sized world (9,600 rows)."""
    tab, z, _, _ = _world(keys=8, steps=25, seed=2)
    w = ST.weights(tab)
    fo = P.folds(tab)
    hh = P.hidden(P.random_actor(cfg, 10), z)
    snr = np.array([0.25, 1.0, 4.0])
    y = np.random.default_rng(3).standard_normal((len(w), 3))
    H = P.plant_units(hh["h3"], y, snr, w, np.random.default_rng(5))
    base = np.hstack([z, hh["m"]])
    rb = P.probe_r2(P.ridge_cv(base, y, fo, w, None, None, GRID, H_GRID), w)
    rf = P.probe_r2(P.ridge_cv(base, y, fo, w, H, None, GRID, H_GRID), w)
    want = snr / (1 + snr)
    assert np.all(np.abs(rf - want) < 0.03), (rf, want)
    assert np.all(np.abs(rf - rb - want) < 0.03), (rf - rb, want)


def test_decoy_layer_does_not_pass_c2_and_a_plant_does(world, c2run):
    """PR-6 and PR-1b on one pipeline. A layer of random h3 plus 32 tanh units of random
    mixes of (s - s_apex, vx, sigma, log R) gains on raw lambda but not on the cross-fitted
    lambda_res, and C2's statistic (minus the median random-init actor) is within 0.01 of 0,
    not a confirm; a unit u_k = r_k + eta (SNR 4, r_k = lambda_res residualised on the
    plant's own baseline) is recovered: partial R2 within 0.03 of 0.8, C2 confirms."""
    W, C = world, c2run
    gd = P.gain(C["base"], C["decoy"], C["rand"], W["w"], W["fo"], W["sig"], C["passing"])
    assert np.all(C["passing"][:4])                 # every K_P residual passes the floor here
    assert abs(gd.stat) <= 0.01 and gd.decision != "confirm", gd.record()
    own_res, own_raw = gd.raw[:4].mean(), gd.raw[4:].mean()
    assert own_raw >= 0.04 and own_res <= 0.02, (gd.raw,)
    gp = P.gain(C["base"], C["plant"], C["rand"], W["w"], W["fo"], W["sig"], C["passing"])
    assert gp.decision == "confirm" and gp.ci[0] > 0.3, gp.record()
    # PR-1b's expectations against the plant's own baseline R2_b (random h3 + z + m). This
    # 192-key world attenuates them by up to ~0.045 at SNR 4 (h3's 256 units share the plant's
    # penalty and the targets are smooth along each trajectory), so 0.05 here; K4's 0.03 is
    # checked on T (and by PR-1a above with i.i.d. targets on a T-sized world).
    r_b = P.probe_r2(C["own"], W["w"], W["sig"])[:4]
    r_f = P.probe_r2(C["plant"], W["w"], W["sig"])[:4]
    partial = (r_f - r_b) / (1 - r_b)
    assert np.all(np.abs(partial - P.expected_r2(4.0)) < 0.05), partial
    assert abs(gp.stat - np.mean((1 - r_b) * P.expected_r2(4.0))) < 0.05, gp.stat
    # the statistic is the network's gain minus the median random gain, over passing columns
    g_r = C["rand"]
    med = np.median([P.probe_r2(p, W["w"], W["sig"]) for p in g_r], axis=0)
    b = P.probe_r2(C["base"], W["w"], W["sig"])
    f = P.probe_r2(C["plant"], W["w"], W["sig"])
    assert gp.stat == pytest.approx(((f - b) - (med - b))[:4].mean(), abs=1e-12)
    assert len(gp.boot) == P.N_BOOT
    with pytest.raises(ValueError):
        P.gain(C["base"], C["plant"], C["rand"], W["w"], W["fo"], W["sig"], np.zeros(8, bool))
    # states.power_floor's verdicts, keyed by component, select the same columns
    floor = {**ST.power_floor(np.full(4, 0.5)), **{r: False for r in RAW}}
    gk = P.gain(C["base"], C["plant"], C["rand"], W["w"], W["fo"], W["sig"], floor)
    assert gk.stat == gp.stat and np.array_equal(gk.passing, C["passing"])


def test_the_output_in_the_layer_adds_nothing_pr2(world, c2run):
    """PR-2: random-init h3 plus its own m (m is in the base): Delta2 within 0.01 of 0, and
    the same as h3 alone."""
    W, C = world, c2run
    g_m = P.gain(C["base"], C["with_m"], C["rand"], W["w"], W["fo"], W["sig"], C["passing"])
    g_h = P.gain(C["base"], C["own"], C["rand"], W["w"], W["fo"], W["sig"], C["passing"])
    assert abs(g_m.stat) <= 0.01 and abs(g_m.stat - g_h.stat) <= 0.005, (g_m.gain, g_h.gain)


def test_paired_increment_is_the_change_in_the_same_network(world):
    """K4's no-gain plants are scored by the paired increment Delta2(host + plant) -
    Delta2(host): one fit of the base and the controls, the same rows and draws, so the
    controls' median cancels. A host with a chance offset against the controls (a weak unit
    of lambda_res) and a plant that adds nothing (its own m, already in the base): the
    unpaired Delta2 carries the offset past 0.01, the paired one stays near 0. A plant that
    adds information still raises the paired increment past 0.01."""
    W = world
    w, fo, sig = W["w"], W["fo"], W["sig"]
    Y = W["res"].by_fold
    n = len(w)
    own = Y[fo.outer, np.arange(n)]
    h3, m = W["h"]["h3"], W["h"]["m"]
    host = P.plant_units(h3, own[:, :1], 0.3, w, np.random.default_rng(11))      # 'lucky'
    counts = P.cluster_counts(int(fo.cluster.max()) + 1, 300, np.random.default_rng(12))

    def run(layer, h=host):
        return P.c2_paired(layer, h, W["rand"], W["z"], m, Y, fo, w, sig, NAMES, None, None,
                           counts, GRID, H_GRID)

    nothing = run(np.hstack([host, m]))
    # the paired increment is the plant's gain minus the host's, both c2's own gains
    for g, layer in ((nothing.plant, np.hstack([host, m])), (nothing.host, host)):
        ref = P.c2(layer, W["rand"], W["z"], m, Y, fo, w, sig, NAMES, None, None, counts,
                   GRID, H_GRID)
        assert g.stat == ref.stat and np.array_equal(g.boot, ref.boot)
    assert nothing.stat == pytest.approx(nothing.plant.stat - nothing.host.stat, abs=1e-15)
    np.testing.assert_allclose(nothing.gain, nothing.plant.r2_full - nothing.host.r2_full,
                               atol=1e-15)
    np.testing.assert_array_equal(nothing.boot, nothing.plant.boot - nothing.host.boot)
    assert nothing.ci == tuple(float(v) for v in P.ci(nothing.boot))
    rec = nothing.record()
    assert rec["stat"] == nothing.stat and rec["host"]["stat"] == nothing.host.stat
    assert rec["unpaired"]["stat"] == nothing.plant.stat
    # the host's offset alone cannot fail the no-gain tolerance; unpaired, it does
    assert nothing.host.stat > 0.01 and nothing.plant.stat > 0.01, nothing.host.record()
    assert abs(nothing.stat) <= 0.005, nothing.gain
    # a planted gain (a unit of xi's residual, SNR 1) still exceeds it
    gain = run(P.plant_units(host, own[:, 2:3], 1.0, w, np.random.default_rng(13)))
    assert gain.stat > 0.01 and gain.ci[0] > 0.0, gain.gain
    # two gains of different controls are not paired
    other = P.c2(host, W["rand"][:2], W["z"], m, Y, fo, w, sig, NAMES, None, None, counts,
                 GRID, H_GRID)
    with pytest.raises(ValueError):
        P.paired(nothing.plant, other)


def test_scale_invariance_pr7(world, c2run):
    """PR-7: h3 scaled by 0.1 and 10, z channels by random factors in [0.2, 5]: Delta2 and
    R2_z unchanged to 1e-6."""
    W, C = world, c2run
    sc = np.random.default_rng(4).uniform(0.2, 5.0, racing.OBS_DIM)
    base_s = np.hstack([W["z"] * sc, W["h"]["m"]])
    b = P.ridge_cv(base_s, C["Y"], W["fo"], W["w"], None, NAMES + RAW, GRID, H_GRID)
    r_b = P.probe_r2(C["base"], W["w"], W["sig"])
    np.testing.assert_allclose(P.probe_r2(b, W["w"], W["sig"]), r_b, atol=1e-6)
    r_f = P.probe_r2(C["own"], W["w"], W["sig"])
    for k in (0.1, 10.0):
        f = P.ridge_cv(base_s, C["Y"], W["fo"], W["w"], k * W["h"]["h3"], NAMES + RAW, GRID,
                       H_GRID)
        np.testing.assert_allclose(P.probe_r2(f, W["w"], W["sig"]) - r_b, r_f - r_b, atol=1e-6)


def test_grouped_split_has_no_leak_and_a_misgrouped_one_does_pr3(world):
    """PR-3: one-hot geometry and pair codes. On the frozen grouped split the codes gain
    <= 0.01; dealt by key (mirror partners separated) they leak."""
    W = world
    tab, w = W["tab"], W["w"]
    geo = W["fo"].cluster
    eff = np.random.default_rng(6).normal(0, 1, geo.max() + 1)[geo]
    y = eff + 0.5 * np.random.default_rng(7).standard_normal(len(w))     # an even target
    codes = np.hstack([P.one_hot(geo), P.one_hot(tab["pair_id"])])
    base = W["z"][:, :20]
    gains = {}
    for name, fo in (("grouped", W["fo"]),
                     ("by_key", P.grouped_folds(tab["key_id"], np.random.default_rng(8),
                                                cluster=geo))):
        rb = P.probe_r2(P.ridge_cv(base, y, fo, w, None, ("lam_s",), GRID, H_GRID), w)[0]
        rf = P.probe_r2(P.ridge_cv(base, y, fo, w, codes, ("lam_s",), GRID, H_GRID), w)[0]
        gains[name] = rf - rb
    assert gains["grouped"] <= 0.01 and gains["by_key"] > 0.3, gains


# ------------------------------------------------------------------------- K5, FWL, C2-sw, MLP


def _rows(a: np.ndarray) -> np.ndarray:
    """a's rows in lexicographic order: two arrays hold the same multiset of rows iff equal."""
    a = np.asarray(a)
    return a[np.lexsort(a.T[::-1])]


def test_permuted_targets_are_a_wiring_null(world):
    """While fold f is held out its rows keep their targets; the training rows are permuted
    state by state across the training folds (never within each one), the mirror-odd columns
    (n, xi) within each turn direction and s, vx across both; a draw is reproducible, two
    draws differ; without the direction nothing is permuted."""
    W = world
    fo, lam, sig = W["fo"], W["lam"], W["sig"]
    odd = np.array([P.is_odd(nm) for nm in NAMES])
    assert odd.tolist() == [False, True, True, False]
    Yf = P.permuted_targets(lam, fo, sig, NAMES)
    assert Yf.shape == (6, *lam.shape)
    for f in range(6):
        held, train = fo.outer == f, fo.outer != f
        np.testing.assert_array_equal(Yf[f][held], lam[held])
        np.testing.assert_array_equal(_rows(Yf[f][train][:, ~odd]), _rows(lam[train][:, ~odd]))
        for d in (-1, 1):
            rows = train & (sig == d)
            np.testing.assert_array_equal(_rows(Yf[f][rows][:, odd]), _rows(lam[rows][:, odd]))
            # s and vx cross the directions
            assert not np.array_equal(np.sort(Yf[f][rows][:, ~odd], 0),
                                      np.sort(lam[rows][:, ~odd], 0))
        for g in set(range(6)) - {f}:          # every column crosses the training folds
            rows = fo.outer == g
            for k in range(4):
                assert not np.array_equal(np.sort(Yf[f][rows, k]), np.sort(lam[rows, k]))
    np.testing.assert_array_equal(P.permuted_targets(lam, fo, sig, NAMES), Yf)
    other = P.permuted_targets(lam, fo, sig, NAMES, draw=1)
    assert not np.array_equal(other[0][fo.outer != 0], Yf[0][fo.outer != 0])
    with pytest.raises(ValueError, match="direction"):
        P.permuted_targets(lam, fo, None, NAMES)


def test_wiring_null_passes_and_its_planted_faults_fail(world, monkeypatch):
    """K5's wiring check (``WIRING_DRAWS`` = 10 draws averaged; a component passes with its
    R2_wd CI's lower end <= 0 and its point <= 0.01): the correct pipeline passes, each
    draw's point recorded. Each planted fault fails on the components z decodes (s, n, vx;
    xi's real R2 here is low): one outer fold's model fitted on unpermuted targets; lam_n
    left unpermuted for one model (n alone fails); test-set tuning, every penalty chosen by
    the held-out SSE against the real targets. Without the direction it raises."""
    W = world
    fo, lam, w, sig = W["fo"], W["lam"], W["w"], W["sig"]
    counts = P.cluster_counts(48, 400, P.boot_rng(0))

    def check():
        return P.wiring_null(W["z"], lam, fo, w, None, NAMES, sig, counts)

    out = check()
    assert out["pass"] and all(out["pass_by_component"]), out
    assert out["n_draws"] == P.WIRING_DRAWS == 10 and out["tol"] == P.WIRING_TOL == 0.01
    assert np.shape(out["per_draw"]) == (10, 4) and len(set(map(tuple, out["per_draw"]))) == 10
    real = P.probe_r2(P.ridge_cv(W["z"], lam, fo, w, None, NAMES), w, sig)
    assert np.all(real[[0, 1, 3]] > 0.5)                 # the same pipeline on real targets

    perm, ridge = P.permuted_targets, P.ridge_cv

    def fold0_unpermuted(Y, fo, direction, names=None, draw=0):
        Yf = perm(Y, fo, direction, names, draw)
        Yf[0] = Y
        return Yf

    def n_unpermuted(Y, fo, direction, names=None, draw=0):
        Yf = perm(Y, fo, direction, names, draw)
        Yf[0][:, 1] = Y[:, 1]
        return Yf

    def tuned_on_test_set(base, Y, fo, w, hidden=None, names=None, grid=P.GRID,
                          h_grid=P.H_GRID):     # on this file's coarser GRID, for speed
        fits = [ridge(base, Y, fo, w, hidden, names, (a,), (h,)) for a in GRID
                for h in (H_GRID if hidden is not None else (np.inf,))]
        pred = np.array(fits[0].pred)
        for f in range(fo.n_folds):
            te = fo.outer == f            # Probe.y: the real held-out targets
            sse = np.stack([w[te] @ (p.y[te] - p.pred[te]) ** 2 for p in fits])
            pred[te] = np.stack([p.pred[te] for p in fits])[sse.argmin(0), :, np.arange(4)].T
        return P.Probe(pred, fits[0].y, fits[0].alpha, fits[0].kept, fits[0].names)

    for name, fake, fails in (("permuted_targets", fold0_unpermuted, [0, 1, 3]),
                              ("permuted_targets", n_unpermuted, [1]),
                              ("ridge_cv", tuned_on_test_set, [0, 1, 3])):
        with monkeypatch.context() as mp:
            mp.setattr(P, name, fake)
            out = check()
        ok = np.array(out["pass_by_component"])
        assert not out["pass"] and not ok[fails].any(), (fake.__name__, out)
        if fake is n_unpermuted:
            assert ok.tolist() == [True, False, True, True], out
    with pytest.raises(ValueError, match="direction"):
        P.wiring_null(W["z"], lam, fo, w, None, NAMES, None, counts)


def test_two_stage_fwl_reproduces_the_full_model(world):
    W = world
    base, H, Y = W["z"][:, :30], W["h"]["h3"][:, :40], W["lam"][:, :2]
    full = P.ridge_cv(base, Y, W["fo"], W["w"], H, None, (1e-9,), (0.05,))
    two = P.fwl(base, H, Y, W["fo"], W["w"], full.alpha)
    np.testing.assert_allclose(two.pred, full.pred, atol=1e-6)
    # at a real penalty the two differ, slightly
    full = P.ridge_cv(base, Y, W["fo"], W["w"], H, None, (1.0,), (0.05,))
    two = P.fwl(base, H, Y, W["fo"], W["w"], full.alpha)
    d = np.abs(P.probe_r2(two, W["w"]) - P.probe_r2(full, W["w"]))
    assert np.all(d < 0.05) and not np.allclose(two.pred, full.pred)


def test_c2_sw_recovers_a_planted_switching_unit_pr8(world):
    """PR-8: on the pedal-saturated rows (here: the approach), a unit holding sigma^ped plus
    noise (SNR 4 against its residual on z): C2-sw's gain within 0.03 of (1 - R2_z) 0.8, a
    confirm. The pre-clip margin's Spearman."""
    W = world
    ds = W["tab"]["ds_apex"]
    sat = ds < -40
    sig_ped = -0.002 * (-40 - ds[sat]) ** 1.2 + 0.05 * W["eps"][sat, 3]
    fs = W["fo"].subset(sat)
    ws = ST.weights({k: v[sat] for k, v in W["tab"].items()})
    z = W["z"][sat]
    grid = (1e-2, 1e-1, 1.0)
    pz = P.ridge_cv(z, sig_ped, fs, ws, None, ("sigma_pedal",), grid, (*grid, np.inf))
    r2z = P.probe_r2(pz, ws)[0]
    H = P.plant_units(W["h"]["h3"][sat], pz.y[:, 0] - pz.pred[:, 0], 4.0, ws,
                      np.random.default_rng(13))
    rand = [h[sat] for h in W["rand"]]
    g = P.c2_sw(H, rand, z, sig_ped, fs, ws, grid=grid, h_grid=(*grid, np.inf))
    assert g.decision == "confirm" and abs(g.stat - (1 - r2z) * 0.8) < 0.03, g.record()
    assert g.r2_base[0] == pytest.approx(r2z, abs=1e-12)
    m_ped = 1.0 + 0.5 * sig_ped + 0.01 * np.random.default_rng(14).standard_normal(sat.sum())
    assert P.wspearman(m_ped - 1.0, sig_ped, ws) > 0.9


def test_control_task_calibrates_its_ceiling(world):
    W = world
    z, w, fo = W["z"], W["w"], W["fo"]
    f0, f1 = P.control_function(z, w, 0), P.control_function(z, w, 1)
    np.testing.assert_array_equal(f0, P.control_function(z, w, 0))
    assert not np.allclose(f0, f1)
    assert abs(ST.wmean(f0, w)) < 1e-12 and ST.wvar(f0, w) == pytest.approx(1.0)
    c = P.control_task(z, w, 0, 0.3)                    # no ceiling: s^2 = (1 - R2*) / R2*
    assert c.s == pytest.approx(np.sqrt(0.7 / 0.3)) and np.isnan(c.ceiling)

    def ceiling(y):                                      # a cheap ceiling: ridge on z
        return P.probe_r2(P.ridge_cv(z, y, fo, w, None, ("y",), GRID, H_GRID), w)[0]

    c = P.control_task(z, w, 0, 0.3, ceiling)
    assert c.ok and abs(c.ceiling - 0.3) <= P.CONTROL_TOL and c.evaluations <= P.CONTROL_STEPS
    assert ST.wvar(c.y, w) == pytest.approx(1.0)
    np.testing.assert_allclose(P.selectivity([0.5, 0.2], [0.3, 0.3]), [0.2, -0.1])
    with pytest.raises(ValueError):
        P.control_task(z, w, 0, 1.0)


def test_mlp_probe_reads_a_nonlinear_target(world):
    W = world
    rng = np.random.default_rng(0)
    n = len(W["w"])
    X = rng.normal(0, 1, (n, 6))
    y = np.sin(2 * X[:, 0]) * X[:, 1] + 0.1 * rng.standard_normal(n)
    mlp = P.mlp_cv(X, y, W["fo"], W["w"], ("y",), alphas=(1e-4,), maxiter=300)
    lin = P.ridge_cv(X, y, W["fo"], W["w"], None, ("y",), GRID, H_GRID)
    r_m, r_l = P.probe_r2(mlp, W["w"])[0], P.probe_r2(lin, W["w"])[0]
    assert r_m > 0.8 and r_l < 0.1, (r_m, r_l)
    again = P.mlp_cv(X, y, W["fo"], W["w"], ("y",), alphas=(1e-4,), maxiter=300)
    np.testing.assert_array_equal(again.pred, mlp.pred)  # seeded: deterministic
    # the inner CV picks the penalty on the training geometries only
    few = W["fo"].cluster < 6
    m2 = P.mlp_cv(X[few], y[few], W["fo"].subset(few), W["w"][few], ("y",), alphas=(1e-4, 1.0),
                  maxiter=40)
    assert set(np.unique(m2.alpha[:, 0, 0][np.isfinite(m2.alpha[:, 0, 0])])) <= {1e-4, 1.0}


def test_decision_bands():
    assert P.decide(0.06, 0.01, 0.1) == "confirm"
    assert P.decide(0.04, 0.01, 0.1) == "inconclusive"        # point below the band
    assert P.decide(0.06, -0.01, 0.1) == "inconclusive"
    assert P.decide(0.01, -0.02, 0.049) == "refute"
    assert P.expected_r2(1.0) == 0.5


# ------------------------------------------------------------------------- K3 in float64


@pytest.mark.skipif(not HAVE_G1_CONFIG, reason="G1's config.json is not on this machine")
def test_extractor_float64_matches_flax_to_1e10(f64_runs):
    for name, (path, proc) in f64_runs.items():
        log, _ = proc.communicate(timeout=300)
        assert proc.returncode == 0, f"{name}:\n{log[-3000:]}"
        rec = json.loads(path.read_text())
        assert rec["pass"] and rec["max_error"] <= P.EXTRACTOR_TOL, rec
        assert all(r["dtype"] == "float64" for r in rec["rows"])
        heads = 0 if name == "no_heads" else 5
        assert all(r["heads"] == heads for r in rec["rows"])
        assert ("critic_heads" in rec["rows"][0]) == bool(heads)
        assert all(r["grad_v"] <= P.EXTRACTOR_TOL for r in rec["rows"])
        if heads:   # Critic.apply stop-gradients the heads; the extractor does not
            assert all(r["grad_heads_apply_max"] == 0.0 < r["grad_heads_extractor_max"]
                       for r in rec["rows"])


def test_a_component_mask_scores_each_component_on_its_own_rows():
    """Review finding (exclusions never reached C2): ``boot_sums``' per-component mask gives
    each component exactly the R2 of its own subset of rows (the probes fitted on all of
    them), on every bootstrap draw."""
    rng = np.random.default_rng(31)
    n, G = 240, 24
    cl = np.arange(n) // 10
    w = rng.random(n) + 0.5
    d = np.where(np.arange(n) % 2 == 0, 1, -1)
    y = rng.normal(size=(n, 2)) + d[:, None] * np.array([0.0, 1.0])
    pred = y[None] + rng.normal(size=(3, n, 2))
    mask = np.ones((n, 2), bool)
    mask[: n // 2, 1] = False
    counts = P.cluster_counts(G, 50, np.random.default_rng(32))
    bs = P.boot_sums(y, pred, w, cl, d, ("lam_s", "lam_n"), mask)
    got = P.boot_r2(bs, counts)
    for k in range(2):
        m = mask[:, k]
        sub = P.boot_sums(y[m, k:k + 1], pred[:, m, k:k + 1], w[m], cl[m], d[m],
                           (("lam_s", "lam_n")[k],))
        # the subset's own clusters: zero-count clusters are absent there, so pad the counts
        present = np.unique(cl[m])
        want = P.boot_r2(sub, counts[:, present])
        np.testing.assert_allclose(got[:, :, k], want[:, :, 0], rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------------- C2's control pool


POP_G, POP_ROWS = 48, 10        # the population world: 48 geometries of 10 rows each
POP_K2 = 0.5                    # the base's residual share: R2(base) is about 1 - POP_K2
POP_MU, POP_TAU, POP_EPS = 0.5, 0.06, 0.10   # each network's share of the base's residual


def _population(rng: np.random.Generator, mu_net: float, n_pool: int = P.N_POOL):
    """C2's statistic on a world whose random 'networks' vary (the lead's decision: the
    control pool and the two-level CI). One target column; the base's prediction leaves the
    residual r = sqrt(POP_K2) y. A network with parameter mu removes the share u_g = mu +
    eps_g (eps_g ~ N(0, POP_EPS^2), fresh per network and geometry) of r's squared error in
    geometry g (its prediction base + t r, (1 - t)^2 = 1 - u), so its gain over the base is
    about POP_K2 mu in expectation over geometries. The pool's controls draw mu from N(POP_MU,
    POP_TAU^2): their gains vary with SD POP_K2 POP_TAU = 0.03 (C2's null had 0.028), and the
    population's median gain is POP_K2 POP_MU. Returns the base, the network's and the
    controls' probes (the network first), the folds and the weights."""
    n = POP_G * POP_ROWS
    cl = np.repeat(np.arange(POP_G), POP_ROWS)
    y = rng.standard_normal((n, 1))
    base = y * (1.0 - np.sqrt(POP_K2))
    mus = np.r_[mu_net, POP_MU + POP_TAU * rng.standard_normal(n_pool)]
    preds = []
    for mu in mus:
        u = np.clip(mu + POP_EPS * rng.standard_normal(POP_G), 0.0, 1.0)[cl]
        preds.append(base + (1.0 - np.sqrt(1.0 - u))[:, None] * (y - base))

    def probe(p):
        return P.Probe(p, y, np.zeros((1, 1, 2)), np.zeros((1, 2), int), ("lam_s",))

    fo = P.Folds(np.zeros(n, int), np.zeros((1, n), int), cl)
    return probe(base), [probe(p) for p in preds], fo, np.ones(n)


def _pool_gain(rng, mu_net, n_boot=500, pool=True):
    b, fs, fo, w = _population(rng, mu_net)
    counts = P.cluster_counts(POP_G, n_boot, rng)
    draws = P.pool_draws(P.N_POOL, n_boot, rng) if pool else None
    return P.gain(b, fs[0], fs[1:], w, fo, counts=counts, pool=draws)


def test_the_two_level_ci_carries_the_controls_spread():
    """(a) On a population of random networks whose gains vary (``_population``), the
    two-level CI (the geometry clusters and the 20 controls resampled together) is wider
    than the same point's geometry-only CI, which it records beside it; the point is the
    median over all 20 controls either way, and with every draw's controls the whole pool
    the two-level CI is the geometry-only one exactly."""
    g = _pool_gain(np.random.default_rng(1), POP_MU)
    geo = _pool_gain(np.random.default_rng(1), POP_MU, pool=False)
    assert g.stat == geo.stat and g.ci_geometry == geo.ci
    assert g.ci[1] - g.ci[0] > geo.ci[1] - geo.ci[0]
    assert g.ci[0] < geo.ci[0] and g.ci[1] > geo.ci[1]
    rec = g.record()
    assert rec["ci_kind"] == "two-level" and rec["ci_geometry"] == list(geo.ci)
    assert "ci_kind" not in geo.record() and geo.record().keys() == {
        "names", "passing", "r2_base", "r2_full", "r2_random", "gain", "raw", "partial",
        "stat", "ci", "decision", "band"}       # K4's records are unchanged
    # draw b's statistic: the network's gain minus the median over its resampled controls
    b, fs, fo, w = _population(np.random.default_rng(2), POP_MU)
    counts = P.cluster_counts(POP_G, 40, np.random.default_rng(3))
    draws = P.pool_draws(P.N_POOL, 40, np.random.default_rng(4))
    got = P.gain(b, fs[0], fs[1:], w, fo, counts=counts, pool=draws)
    bs = P.boot_sums(b.y, np.stack([b.pred] + [f.pred for f in fs]), w, fo.cluster)
    rb = P.boot_r2(bs, counts)[:, :, 0]
    want = [rb[i, 1] - rb[i, 0] - np.median(rb[i, 2 + draws[i]] - rb[i, 0])
            for i in range(40)]
    np.testing.assert_allclose(got.boot, want, rtol=0, atol=1e-14)
    whole = np.tile(np.arange(P.N_POOL), (40, 1))
    same = P.gain(b, fs[0], fs[1:], w, fo, counts=counts, pool=whole)
    assert same.ci == same.ci_geometry
    with pytest.raises(ValueError, match="indices of the 20 controls"):
        P.gain(b, fs[0], fs[1:], w, fo, counts=counts, pool=draws[:, :] + P.N_POOL)
    with pytest.raises(ValueError, match="indices of the 20 controls"):
        P.gain(b, fs[0], fs[1:], w, fo, counts=counts, pool=draws[:10])


def test_a_planted_gain_over_the_population_median_is_confirmed():
    """(b) A network planted with a true gain of 0.08 over the population's median gain is
    confirmed (CI lower end > 0, point >= 0.05), and one at the median is not; its rank
    among the 20 controls' own gains is reported."""
    hi = _pool_gain(np.random.default_rng(5), POP_MU + 0.08 / POP_K2)
    mid = _pool_gain(np.random.default_rng(5), POP_MU)
    assert hi.decision == "confirm" and hi.ci[0] > 0 and hi.stat >= 0.05, hi.record()
    assert mid.decision != "confirm" and abs(mid.stat) < 0.03, mid.record()
    assert hi.pool["rank"] <= 2 and hi.pool["of"] == P.N_POOL + 1     # the controls' SD 0.03
    assert 1 < mid.pool["rank"] < P.N_POOL + 1
    ctl = np.asarray(mid.pool["control_gains"])
    assert ctl.shape == (P.N_POOL,) and mid.pool["network_gain"] == pytest.approx(mid.raw[0])
    np.testing.assert_allclose(ctl, (mid.r2_random - mid.r2_base)[:, 0], rtol=0, atol=1e-15)
    assert mid.pool["rank"] == 1 + int((ctl > mid.raw[0]).sum())
    assert mid.pool["n_above"] + mid.pool["n_below"] + mid.pool["n_tied"] == P.N_POOL
    # over fresh worlds and pools: the planted network is confirmed nearly always, the one at
    # the median never
    rng = np.random.default_rng(6)
    dec_hi = [_pool_gain(rng, POP_MU + 0.08 / POP_K2, 300).decision for _ in range(60)]
    dec_mid = [_pool_gain(rng, POP_MU, 300).decision for _ in range(60)]
    assert dec_hi.count("confirm") >= 55 and "confirm" not in dec_mid


def test_the_two_level_ci_covers_the_offset_from_the_population_median():
    """(c) A cheap Monte Carlo: over 400 fresh worlds and pools of 20 controls, the two-level
    95% CI covers the network's true offset from the population's median gain (0 at the
    median, 0.08 planted) near nominally; the geometry-only CI, which holds the controls
    fixed, under-covers (about 0.85 in a 2,000-draw check) and is narrower on average."""
    for mu, seed in ((POP_MU, 7), (POP_MU + 0.08 / POP_K2, 8)):
        truth = POP_K2 * (mu - POP_MU)
        rng = np.random.default_rng(seed)
        two = geo = 0
        width = []
        for _ in range(400):
            g = _pool_gain(rng, mu)
            two += g.ci[0] <= truth <= g.ci[1]
            geo += g.ci_geometry[0] <= truth <= g.ci_geometry[1]
            width.append((g.ci[1] - g.ci[0]) / (g.ci_geometry[1] - g.ci_geometry[0]))
        assert 0.92 <= two / 400 <= 0.99, (mu, two / 400)
        assert geo / 400 <= two / 400 - 0.05, (mu, geo / 400, two / 400)
        assert np.mean(width) > 1.2 and np.min(width) >= 1.0, (mu, np.mean(width))


def test_the_pool_and_its_resampling_are_deterministic():
    """(d) The pool is seeds 0-19 (the 5 controls among them); its resampling draws on its own
    named stream ('costate_c2_pool'), the same for every call and independent of the
    geometry bootstrap's, so a record repeats exactly."""
    from apexzero.costate import STREAMS, stream_rng

    assert P.POOL_SEEDS == tuple(range(20)) and P.N_POOL == 20
    assert set(range(P.N_RANDOM_ACTORS)) <= set(P.POOL_SEEDS)
    assert STREAMS["costate_c2_pool"] == P.POOL_STREAM
    assert list(STREAMS.values()).count(P.POOL_STREAM) == 1
    a, b = P.pool_draws(P.N_POOL), P.pool_draws(P.N_POOL)
    assert a.shape == (P.N_BOOT, P.N_POOL) and a.min() >= 0 and a.max() < P.N_POOL
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        a, stream_rng("costate_c2_pool").integers(0, P.N_POOL, size=(P.N_BOOT, P.N_POOL)))
    # the geometry bootstrap's stream draws other numbers
    assert not np.array_equal(a, P.pool_draws(P.N_POOL, rng=P.boot_rng()))
    b0, fs, fo, w = _population(np.random.default_rng(9), POP_MU)
    r1 = P.gain(b0, fs[0], fs[1:], w, fo, pool=P.pool_draws(P.N_POOL)).record()
    r2 = P.gain(b0, fs[0], fs[1:], w, fo, pool=P.pool_draws(P.N_POOL)).record()
    assert json.dumps(r1) == json.dumps(r2)
