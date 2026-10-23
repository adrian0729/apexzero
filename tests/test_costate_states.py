"""Idea 2's state sets, strata, folds, decoys, observability and weights (costate.states;
design v2.1, Sections 3.2, 4 and 12), on constructed inputs with known answers and on step 1's
stored keys:

- T's steps (multiples of 5 up to the first t past s_c1 + 50 m), K2's two horizon rules, the
  interior window (k steps ahead, rows, clearance, slip margin), the phase labels and the
  sign-change count;
- the serpentine deal, the geometry folds (6 x 8, whole geometries, deterministic per
  partition, different across partitions) and the inner folds;
- the three weighting schemes and the small-geometry flag;
- the kNN decoy model (against a brute-force mean), cross-fitting that never sees a row's own
  fold, f_D's k by grouped CV (the smallest for a rough decoy, the largest for a smooth one,
  fold f's choice blind to fold f's targets, ties to the smaller k), residual shares (with
  the within-direction convention) and fold stability;
- the observability rule (tau_s, column residuals, clipped rows, thresholds);
- on the stored keys: T = 9,967 states, 48 geometries whose keys (mirror partners, phases,
  widths, twins) share a fold, 459 braking keys, the example key's rows, and N = 8,136.

CasADi and NumPy only (no JAX float64, no network)."""

from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.costate import data_dirs, stream_rng
from apexzero.costate import states as ST
from apexzero.costate import targets as TG
from apexzero.fields import keys as K
from apexzero.interp import io
from apexzero.interp.stimuli import Landmarks

KEYS, STIM = data_dirs()
HAVE_KEYS = (KEYS / "nominal").exists() and (STIM / "cells.json").exists()
needs_keys = pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys are not on this machine")
EXAMPLE = "03b6076863e3adfe335e6757"
LM = Landmarks(s_c0=100.0, s_a0=110.0, s_apex=120.0, s_a1=130.0, s_c1=140.0)


# ------------------------------------------------------------------------- steps and strata


def test_t_end_steps_and_horizon_rules():
    x = np.zeros((401, 8))
    x[:, 0] = 2.0 * np.arange(401)                          # s = 2t
    assert ST.t_end(x, LM) == 95                             # first s >= 190
    np.testing.assert_array_equal(ST.t_steps(x, LM, 400), np.arange(0, 96, 5))
    np.testing.assert_array_equal(ST.t_steps(x, LM, 400, t_H=40), np.arange(0, 41, 5))
    # remaining horizon: a W = 800 key keeps t <= 400 + t_H; 'absolute' cuts every key
    np.testing.assert_array_equal(ST.horizon_keep(np.array([90, 500, 600]), 800, 120),
                                  [True, True, False])
    np.testing.assert_array_equal(ST.horizon_keep(np.array([90, 500]), 800, 120, "absolute"),
                                  [True, False])
    assert ST.horizon_keep(np.arange(5), 400, None).all()
    with pytest.raises(ValueError):
        ST.t_end(x[:50], LM)


def _fake_cs(W: int = 100):
    """The attributes ``interior`` reads, all clear."""
    return SimpleNamespace(own_corridor=np.zeros((W + 1, 2), bool),
                           own_slip=np.zeros((W + 1, 2), bool),
                           clearance=np.full((W + 1, 2), 3.0), slip_margin=np.full(W + 1, 0.1))


def test_interior_window():
    cs = _fake_cs()
    cs.own_corridor[20, 0] = True                            # a wall contact at x_20
    cs.clearance[40, 1] = ST.CLEARANCE_MIN - 1e-3            # close to the edge at x_40
    cs.slip_margin[70] = ST.SLIP_MARGIN_MIN - 1e-4           # close to the slip limit at x_70
    for k in (10, 20):
        inner = ST.interior(cs, k)
        bad = np.zeros(101, bool)
        for tau in (20, 40, 70):
            bad[max(tau - k, 0): tau + 1] = True
        np.testing.assert_array_equal(inner, ~bad)
    assert ST.interior(cs, 10)[95]                           # the window stops at W
    cs.own_slip[100, 1] = True
    assert not ST.interior(cs, 10)[95]


def test_phases_and_sign_changes():
    s = np.array([50.0, 89.9, 90.0, 119.0, 120.0, 139.9, 140.0, 200.0])
    np.testing.assert_array_equal(
        ST.phases(s, LM, s_b=90.0),
        ["approach", "approach", "braking", "braking", "corner", "corner", "exit", "exit"])
    # no sustained brake: the braking phase starts at s_c0
    assert ST.phases(np.array([95.0, 100.0]), LM, np.nan).tolist() == ["approach", "braking"]
    assert ST.sign_changes(np.array([1.0, 2.0, -1.0, 0.0, -3.0, 4.0])) == 2
    assert ST.sign_changes(np.array([1e-13, -1e-13, 1.0]), atol=1e-12) == 0


# ------------------------------------------------------------------------- folds


def _geometries():
    """48 (R, theta) pairs shaped like the A2 set: 23 at 90, 17 at 180, 4 at 135, 3 at 45,
    1 at 60 (design Section 4)."""
    Rs = np.geomspace(6.0, 300.0, 23)
    R = np.r_[Rs, Rs[:17], Rs[5:9], Rs[4:7], Rs[12]]
    th = np.r_[np.full(23, 90.0), np.full(17, 180.0), np.full(4, 135.0), np.full(3, 45.0), 60.0]
    return R, th


def test_serpentine_and_deal():
    np.testing.assert_array_equal(ST._serpentine(14, 6), [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0,
                                                          0, 1])
    R, th = _geometries()
    f0 = ST.geometry_folds(R, th, 0)
    np.testing.assert_array_equal(np.bincount(f0), np.full(6, 8))
    np.testing.assert_array_equal(f0, ST.geometry_folds(R, th, 0))            # deterministic
    np.testing.assert_array_equal(f0, ST.geometry_folds(R[::-1], th[::-1], 0)[::-1])  # order-free
    others = [ST.geometry_folds(R, th, p) for p in range(1, ST.N_PARTITIONS)]
    assert all(not np.array_equal(f0, f) for f in others)
    with pytest.raises(ValueError):
        ST.geometry_folds(R, th, ST.N_PARTITIONS)
    # every stratum (theta class x R tercile) is spread over the folds
    cls, ter = ST.theta_class(th), ST.r_tercile(R, th)
    assert np.bincount(ter).tolist() == [16, 16, 16] and set(cls.tolist()) == {0, 1, 2}
    for c in range(3):
        for q in range(3):
            m = (cls == c) & (ter == q)
            if m.sum() >= 6:
                cnt = np.bincount(f0[m], minlength=6)
                assert cnt.max() - cnt.min() <= 2, (c, q, cnt)
    inner = ST.inner_folds(R, th, f0, held_out=2)
    assert np.all(inner[f0 == 2] == -1)
    np.testing.assert_array_equal(np.bincount(inner[f0 != 2]), np.full(5, 8))
    assert stream_rng("costate_folds", 0).integers(1 << 30) != \
        stream_rng("costate_inner", 0).integers(1 << 30)


# ------------------------------------------------------------------------- weights


def _toy_table() -> io.Table:
    """Keys A (2 rows), B (4 rows) in geometry (10, 90) with pairs 1 and 2; C (1 row) in
    (20, 180)."""
    key = np.array(["A"] * 2 + ["B"] * 4 + ["C"])
    return {"key_id": key, "R": np.array([10.0] * 6 + [20.0]),
            "theta": np.array([90.0] * 6 + [180.0]), "pair_id": np.array([1, 1, 2, 2, 2, 2, 3])}


def test_weights_known_answers():
    tab = _toy_table()
    np.testing.assert_allclose(ST.weights(tab, "key"), [1 / 6] * 2 + [1 / 12] * 4 + [1 / 3])
    np.testing.assert_allclose(ST.weights(tab, "geometry"), [1 / 8] * 2 + [1 / 16] * 4 + [1 / 2])
    np.testing.assert_allclose(ST.weights(tab, "state"), np.full(7, 1 / 7))
    sub = io.select(tab, np.array([True, False, True, True, False, False, True]))
    np.testing.assert_allclose(ST.weights(sub), [1 / 3, 1 / 6, 1 / 6, 1 / 3])
    np.testing.assert_array_equal(ST.small_geometry(tab, 1), [False] * 6 + [True])
    np.testing.assert_array_equal(ST.small_geometry(tab, 2), np.ones(7, bool))
    with pytest.raises(ValueError):
        ST.weights(tab, "pair")
    # a stratum renormalises the key weights on its own rows (every statistic on a stratum)
    m = np.array([True, False, True, True, False, False, True])
    sw = ST.stratum_weights(tab, m)
    np.testing.assert_allclose(sw[m], ST.weights(sub))
    np.testing.assert_array_equal(sw[~m], 0.0)
    assert sw.sum() == pytest.approx(1.0)


def test_weighted_moments_and_winsorising():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 1.0, 5.0])
    assert ST.wmean(y, w) == pytest.approx(2.25)
    assert ST.wvar(y, w) == pytest.approx(np.sum(w * (y - 2.25) ** 2) / 8)
    g = np.array([0, 0, 1, 1])
    assert ST.wvar(y, w, g) == pytest.approx((0.25 + 0.25 + 5 * (3 - 17 / 6) ** 2
                                              + (2 - 17 / 6) ** 2) / 8)
    assert ST.wquantile(np.arange(101.0), np.ones(101), 0.5) == pytest.approx(50.0)
    lo, hi = ST.winsor_bounds(np.arange(1001.0), np.ones(1001))
    assert lo == pytest.approx(5.0, abs=0.6) and hi == pytest.approx(995.0, abs=0.6)
    np.testing.assert_array_equal(ST.winsorise(np.array([-5.0, 500.0, 2000.0]), (lo, hi)),
                                  [lo, 500.0, hi])


# ------------------------------------------------------------------------- decoys


def test_knn_mean_against_brute_force():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)) * np.array([1.0, 10.0, 0.1])
    Y = rng.standard_normal((200, 2))
    w = rng.random(200) + 0.5
    m = ST.knn_fit(X, Y, w, k=7)
    q = rng.standard_normal((5, 3)) * np.array([1.0, 10.0, 0.1])
    mu = (w[:, None] * X).sum(0) / w.sum()
    sd = np.sqrt((w[:, None] * (X - mu) ** 2).sum(0) / w.sum())
    for i, z in enumerate(q):
        d = np.linalg.norm((X - z) / sd, axis=1)
        nn = np.argsort(d)[:7]
        np.testing.assert_allclose(m.predict(z[None])[0],
                                   (w[nn, None] * Y[nn]).sum(0) / w[nn].sum(), rtol=1e-12)
    # a constant feature keeps unit scale
    Xc = np.c_[X, np.ones(200)]
    assert ST.knn_fit(Xc, Y).sd[-1] == 1.0


def test_cross_fit_never_sees_the_rows_own_fold():
    """Six folds holding the same feature points; target column g is 1000 on fold g's rows
    and 0 elsewhere. A model that saw fold g would predict something for its rows; every
    model that predicts them (held out, or cross-fitted inside another fold's training set)
    predicts exactly 0."""
    rng = np.random.default_rng(1)
    X = np.tile(rng.standard_normal((40, 2)), (6, 1))
    fold = np.repeat(np.arange(6), 40)
    Y = 1000.0 * (fold[:, None] == np.arange(6)[None, :])
    res = ST.cross_fit(X, Y, fold, k=5)
    assert res.by_fold.shape == (6, 240, 6) and np.all(np.isfinite(res.by_fold))
    for g in range(6):
        np.testing.assert_array_equal(res.by_fold[:, fold == g, g], 1000.0)
    np.testing.assert_array_equal(res.oof, res.by_fold[fold, np.arange(240)])
    np.testing.assert_array_equal(res.training(2), res.by_fold[2])


def _decoy_world(per: int = 40, seed: int = 3):
    """``_geometries``' 48 geometries with ``per`` rows each at x ~ U(-1, 1), a decoy
    coordinate every geometry shares; partition 0's outer and inner folds per row; random
    weights. Two targets: a rough function of x with little noise (the kNN's bias dominates:
    the smallest k is best) and a smooth one with much noise (its variance dominates: the
    largest)."""
    R, th = _geometries()
    rng = np.random.default_rng(seed)
    geo = np.repeat(np.arange(len(R)), per)
    x = rng.uniform(-1.0, 1.0, len(geo))
    outer_g = ST.geometry_folds(R, th, 0)
    inner = np.stack([ST.inner_folds(R, th, outer_g, f, 0)[geo] for f in range(ST.N_FOLDS)])
    Y = np.c_[np.sin(150.0 * x) + 0.02 * rng.standard_normal(len(x)),
              x + rng.standard_normal(len(x))]
    return x[:, None], Y, outer_g[geo], inner, rng.random(len(x)) + 0.5, rng


def test_decoy_k_is_small_for_a_rough_decoy_and_large_for_a_smooth_one():
    X, Y, fold, inner, w, _ = _decoy_world()
    k, sse = ST.knn_k_cv(X, Y, fold, inner, w)
    assert k.shape == (6, 2) and sse.shape == (6, len(ST.KNN_K_GRID), 2)
    np.testing.assert_array_equal(k[:, 0], 3)
    np.testing.assert_array_equal(k[:, 1], 30)
    # the rule inside cross_fit: the same choice and curve, recorded
    res = ST.cross_fit(X, Y, fold, w, inner=inner)
    np.testing.assert_array_equal(res.k, k)
    np.testing.assert_array_equal(res.cv, sse)
    rec = res.record()
    assert rec["grid"] == list(ST.KNN_K_GRID) and rec["k"] == k.tolist()
    assert np.array(rec["sse"]).shape == sse.shape
    # every fit serving fold f uses fold f's k, so slice f of each column is the fixed-k
    # cross-fit's at that k; each row's own-fold residual uses its own fold's k
    fixed = {kk: ST.cross_fit(X, Y, fold, w, k=kk) for kk in ST.KNN_K_GRID}
    for f in range(6):
        for j in range(2):
            np.testing.assert_array_equal(res.by_fold[f, :, j],
                                          fixed[int(k[f, j])].by_fold[f, :, j])
            np.testing.assert_array_equal(res.oof[fold == f, j],
                                          fixed[int(k[f, j])].oof[fold == f, j])
    assert fixed[15].k.tolist() == [[15, 15]] * 6 and fixed[15].cv is None
    assert fixed[15].record()["sse"] is None


def test_decoy_k_never_sees_the_held_out_fold():
    """Fold f's targets replaced by large noise: fold f's curve and k are bit for bit the
    same, and so are slice f's training residuals and its held-out rows' predictions; every
    other fold's curve moves (fold f's rows are in its training set)."""
    X, Y, fold, inner, w, rng = _decoy_world()
    k0, sse0 = ST.knn_k_cv(X, Y, fold, inner, w)
    r0 = ST.cross_fit(X, Y, fold, w, inner=inner)
    moved = False
    for f in (0, 3, 5):
        Y2 = Y.copy()
        Y2[fold == f] = 100.0 * rng.standard_normal(((fold == f).sum(), 2))
        k2, sse2 = ST.knn_k_cv(X, Y2, fold, inner, w)
        np.testing.assert_array_equal(k2[f], k0[f])
        np.testing.assert_array_equal(sse2[f], sse0[f])
        assert all(not np.allclose(sse2[g], sse0[g]) for g in range(6) if g != f)
        moved |= bool(np.any(k2 != k0))
        r2 = ST.cross_fit(X, Y2, fold, w, inner=inner)
        tr = fold != f
        np.testing.assert_array_equal(r2.k[f], r0.k[f])
        np.testing.assert_array_equal(r2.by_fold[f][tr], r0.by_fold[f][tr])
        np.testing.assert_allclose((Y2 - r2.by_fold[f])[~tr], (Y - r0.by_fold[f])[~tr],
                                   rtol=0, atol=1e-9)
    assert moved                                   # the perturbation does move other folds


def test_decoy_k_arguments():
    X, Y, fold, inner, w, _ = _decoy_world(per=12)
    with pytest.raises(ValueError, match="inner folds"):
        ST.cross_fit(X, Y, fold, w)                # the rule needs the inner folds
    with pytest.raises(ValueError):
        ST.cross_fit(X, Y, fold, w, k=np.full((6, 3), 5))
    with pytest.raises(ValueError):
        ST.knn_k_cv(X, Y, fold, inner[:, :-1], w)
    bad = inner.copy()
    bad[0, fold == 0] = 1                          # held-out rows given an inner fold
    with pytest.raises(ValueError, match="training rows"):
        ST.knn_k_cv(X, Y, fold, bad, w)
    with pytest.raises(ValueError, match="increasing"):
        ST.knn_k_cv(X, Y, fold, inner, w, grid=(5, 3))
    # an exact tie (a zero target: every SSE is 0) goes to the smaller k
    k, sse = ST.knn_k_cv(X, np.c_[np.zeros(len(X)), Y[:, 1]], fold, inner, w)
    assert np.all(sse[:, :, 0] == 0.0) and np.all(k[:, 0] == ST.KNN_K_GRID[0])
    # a given [F, m] choice is used fold by fold
    kk = np.array([[3, 30], [30, 3], [5, 8], [8, 5], [15, 15], [3, 3]])
    r = ST.cross_fit(X, Y, fold, w, k=kk)
    np.testing.assert_array_equal(r.k, kk)
    for f in range(6):
        for j in range(2):
            np.testing.assert_array_equal(
                r.by_fold[f, :, j], ST.cross_fit(X, Y, fold, w, k=int(kk[f, j])).by_fold[f, :, j])


def test_residual_share_and_stability():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.uniform(-1, 1, (n, 1))
    sig = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    smooth = np.sin(3 * X[:, 0])
    noise = 0.3 * rng.standard_normal(n)
    Y = np.c_[smooth + noise, sig * (2.0 + smooth) + noise]       # even, then mirror-odd
    fold = rng.integers(0, 6, n)
    w = np.full(n, 1.0 / n)
    res = ST.cross_fit(np.c_[X, sig], Y, fold, w, k=30)
    share = ST.residual_share(res.oof, Y, w, sig, names=("vx", "n"))
    expect = 0.09 / np.array([np.var(Y[:, 0]), ST.wvar(Y[:, 1], w, sig)])
    np.testing.assert_allclose(share, expect, rtol=0.15)
    raw = ST.residual_share(res.oof, Y, w, None, names=("vx", "n"))
    assert raw[1] < 0.2 * share[1]                       # the direction split inflates raw SST
    stab = ST.knn_stability(res, Y, w)
    assert np.all(stab > 0) and np.all(stab < 0.2)
    assert ST.power_floor(np.array([0.2, 0.09, 0.1, 0.5])) == {"s": True, "n": False,
                                                                "xi": True, "vx": True}


# ------------------------------------------------------------------------- observability


def test_observability_rule():
    rng = np.random.default_rng(3)
    m = 222
    J = rng.standard_normal((4, m, 8))
    J[1, :, 0] *= 1e-6                                       # s structurally unobservable
    J[2, :, 0] *= 1e-9                                       # the smallest column, then
    J[2, :, 0] *= 1e-3 * np.median(np.linalg.norm(J[2], axis=0)) / np.linalg.norm(J[2, :, 0])
                                                             # ratio 1e-3: kept
    J[3, :, 4] = 2.0 * J[3, :, 3] - J[3, :, 5]               # vy's column in the others' span
    ob = ST.observability(J)
    assert ob.s_unobservable.tolist() == [False, True, False, False]
    assert ob.s_ratio[2] == pytest.approx(1e-3, rel=1e-9)
    assert np.isnan(ob.resid_sq[1, 0]) and not ob.observable[1, 0]
    assert ob.resid_sq[3, 4] < 1e-20 * np.sum(J[3, :, 4] ** 2)
    # an orthonormal set: each column's residual is its whole norm
    Q, _ = np.linalg.qr(rng.standard_normal((m, 8)))
    oq = ST.observability(Q)
    np.testing.assert_allclose(oq.resid_sq[0], 1.0, rtol=1e-10)
    np.testing.assert_allclose(oq.rel_resid[0], 1.0, rtol=1e-10)
    # dropping s: the others are projected on the remaining six columns only
    for k in (1, 3, 7):
        A = np.delete(J[1], [0, k], axis=1)
        b = J[1, :, k]
        r = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
        assert ob.resid_sq[1, k] == pytest.approx(r @ r, rel=1e-9)
    # clipped rows are zeroed
    clip = np.zeros(m, bool)
    clip[:m - 3] = True
    oc = ST.observability(J[0], clipped=clip)
    Jc = J[0].copy()
    Jc[clip] = 0.0
    np.testing.assert_allclose(oc.s_ratio, ST.observability(Jc).s_ratio, rtol=1e-12)
    assert oc.resid_sq[0, 3] < 1e-12                         # 3 rows cannot hold 8 columns
    # per-component thresholds on the squared residual
    th = ST.observability(J, thresholds={"vy": 1e-6})
    assert th.observable[:3, 4].all() and not th.observable[3, 4]


# ------------------------------------------------------------------------- stored keys


@pytest.fixture(scope="module")
def design():
    bank, _ = K.design(STIM)
    refs = K.a2_refs(KEYS, STIM)
    return SimpleNamespace(bank=bank, refs=refs)


@needs_keys
def test_t_set_size_and_geometry_folds_on_the_a2_keys(design):
    rows, n_T, mb_keys, mb_rows = [], 0, 0, 0
    for r in design.refs:
        st = K.strata(r, STIM)
        rec = K.load_key(r, root=KEYS)
        lm = design.bank.stimuli[r.stim_id].landmarks
        n = len(ST.t_steps(np.asarray(rec.x), lm, r.W))
        n_T += n
        if getattr(rec, "multi_basin", False):
            mb_keys, mb_rows = mb_keys + 1, mb_rows + n
        rows.append({"key_id": np.array([r.key_id]), "R": np.array([st.R]),
                     "theta": np.array([st.theta_deg]), "pair_id": np.array([st.pair_id]),
                     "W": np.array([r.W]), "direction": np.array([st.direction])})
    assert n_T == 9967                                       # design Section 4
    # the 4 multi-basin keys are excluded (Section 3.2): T as built holds the rest
    assert mb_keys == 4 and mb_rows == 80 and n_T - mb_rows == 9887
    tab = ST.concat(rows)
    G = ST.geometry_keys(tab)
    uniq = np.unique(G, axis=0)
    assert len(uniq) == 48
    assert int(ST.small_geometry(tab).sum()) and len(np.unique(
        tab["geometry"][ST.small_geometry(tab)])) == 11
    for p in (0, 3):
        fold = ST.row_folds(tab, p)
        gf = ST.geometry_folds(uniq[:, 0], uniq[:, 1], p)
        np.testing.assert_array_equal(np.bincount(gf), np.full(6, 8))
        for g in np.unique(tab["geometry"]):                 # whole geometries
            assert len(np.unique(fold[tab["geometry"] == g])) == 1
        for pid in np.unique(tab["pair_id"]):                # mirror partners and twins
            m = tab["pair_id"] == pid
            assert len(np.unique(fold[m])) == 1
            assert set(tab["direction"][m].tolist()) == {-1, 1}
    w = ST.weights(tab, "geometry")
    for g in np.unique(tab["geometry"]):
        assert np.sum(w[tab["geometry"] == g]) == pytest.approx(1 / 48)


@needs_keys
def test_rows_of_the_example_key(design):
    ref = next(r for r in design.refs if r.key_id == EXAMPLE)
    kt = TG.key_targets(ref, KEYS, design.bank)
    lm = design.bank.stimuli[ref.stim_id].landmarks
    rows = ST.key_rows(kt.jac, kt.prices, ref, K.strata(ref, STIM), lm)
    t = rows["t"]
    assert t[0] == 0 and np.all(np.diff(t) == 5) and t[-1] <= rows["t_end"][0] < t[-1] + 5
    assert io.check_table(rows) == len(t) and set(ST.TARGET_COLUMNS) <= set(rows)
    i55 = int(np.flatnonzero(t == 55)[0])
    assert rows["contact"][i55] and rows["near_limit"][i55] and not rows["interior"][i55]
    assert rows["lam_n"][i55] == pytest.approx(0.102, abs=5e-4)
    assert rows["lam_pre_n"][i55] == pytest.approx(-0.052, abs=5e-4)
    assert rows["braking_key"].all() and rows["direction"][0] == 1
    order = [ST.PHASES.index(p) for p in rows["phase"]]
    assert order == sorted(order) and set(rows["phase"]) == set(ST.PHASES)
    assert np.isfinite(rows["s_b"][0]) and rows["s_b"][0] < lm.s_apex
    np.testing.assert_array_equal(rows["pedal_saturated"], kt.jac.pedal_saturated[t])
    # C2-sw's stratum: sigma^ped > 0 (unique, the target's support); the raw flag a subset
    sw = rows[ST.SW_STRATUM]
    np.testing.assert_array_equal(sw, kt.jac.switch[t, 1] > TG.ACTIVE_TOL)
    assert np.all(sw[rows["pedal_saturated"]]) and sw.sum() >= rows["pedal_saturated"].sum()
    assert np.all(rows["sigma_pedal"][sw] > 0) and np.all(rows["sigma_pedal"][~sw] <= TG.ACTIVE_TOL)
    np.testing.assert_array_equal(rows["interior20"] & ~rows["interior"], False)
    np.testing.assert_allclose(rows["Pi_in"], kt.prices.inside[t])
    D = ST.decoy_d1(rows)
    np.testing.assert_allclose(D[:, 0], rows["s"] - lm.s_apex)
    np.testing.assert_allclose(D[:, 2], np.log(13.1), rtol=1e-6)
    assert D[0, 3] == 180.0 and D[0, 4] == 12.0 and D[0, 5] == 1.0
    # N's row: an anchored key at t = 1
    aref = next(r for r in K.key_index("anchored", KEYS) if r.verified)
    akt = TG.key_targets(aref, KEYS, design.bank)
    arow = ST.key_rows(akt.jac, akt.prices, aref, K.strata(aref, STIM),
                       design.bank.stimuli[aref.stim_id].landmarks, steps=[1])
    assert arow["t"].tolist() == [1] and arow["agent"][0] == aref.agent
    np.testing.assert_allclose(arow["lam_vx"], akt.jac.lam[1, TG.VX])


@needs_keys
def test_n_set_size():
    try:
        nref = ST.n_refs(KEYS)
    except (OSError, ValueError, KeyError) as e:          # runs not on this machine
        pytest.skip(f"no final snapshots: {e}")
    assert len(nref) == 12 and sum(len(v) for v in nref.values()) == 8136
