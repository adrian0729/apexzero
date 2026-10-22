"""Idea 2's decision rules (costate.analysis) on synthetic data with known answers (design
v2.1, Section 6 with Section 12's decisions).

- The Huber calibration: exact on clean lines, robust to outliers, the slope clamped at >= 0
  (location only), per fold without the held-out rows.
- The bootstrap helpers against probes' own (the same resamples over a fixed cluster count),
  the vectorised weighted Spearman against ``probes.wspearman``, the within-direction rule.
- C1's bands: a gradient equal to lambda (plus noise) confirms beyond its nulls and earns the
  label at scale; 10 lambda confirms without the label; -lambda and pure noise refute; a null
  as good as the gradient blocks the confirm; C1-pattern, C1-heads (planted heads with a
  closed form), C1-limit's sign, C1-dev's 2-of-3 rule, the value-accuracy threshold.
- C3-w: the stratified permutation p against exact enumeration, the shift-inverted upper bound
  and its decisions; C3-int's double centring and a planted pre-signed association; R18's
  variant rule; C2 'not testable' without a component above the power floor.

Pure NumPy: no network, no key file."""

import itertools

import numpy as np
import pytest

from apexzero.costate import analysis as AN
from apexzero.costate import probes as PB
from apexzero.costate import states as ST
from apexzero.costate.targets import K_P


def _world(n_geo: int = 24, keys: int = 3, steps: int = 12, seed: int = 0):
    """Rows over geometries x keys x steps with a direction per key and K_P-shaped targets
    (smooth along each key, varying across geometries)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geo):
        R, th = 10.0 + 7.0 * g, (90.0, 180.0, 45.0)[g % 3]
        for k in range(keys):
            d = 1 if k % 2 == 0 else -1
            phase = rng.uniform(0, 2 * np.pi)
            for t in range(steps):
                u = t / steps
                lam = np.array([np.sin(3 * u + phase) + 0.02 * g, d * (np.cos(2 * u) + 0.03 * g),
                                d * (u - 0.5 + 0.01 * g), np.cos(4 * u + phase) - 0.01 * g])
                rows.append((g, R, th, f"g{g}k{k}", d, lam))
    tab = {"R": np.array([r[1] for r in rows]), "theta": np.array([r[2] for r in rows]),
           "key_id": np.array([r[3] for r in rows]), "direction": np.array([r[4] for r in rows]),
           "pair_id": np.array([r[0] for r in rows])}
    lam = np.stack([r[5] for r in rows])
    return tab, lam


@pytest.fixture(scope="module")
def world():
    tab, lam = _world()
    w = ST.weights(tab, "key")
    cl, G = AN.geometry_clusters(tab)
    outer = ST.row_folds(tab)
    counts = PB.cluster_counts(G, 300, np.random.default_rng(1))
    return tab, lam, w, cl, G, outer, counts


def test_huber_fit_clean_robust_and_clamped():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 0.3 + 1.7 * x
    h = AN.huber_fit(x, y)
    assert h.a == pytest.approx(0.3, abs=1e-9) and h.b == pytest.approx(1.7, abs=1e-9)
    y2 = y.copy()
    y2[:20] += 50.0                                   # 5% gross outliers
    h2 = AN.huber_fit(x, y2)
    ls = np.polyfit(x, y2, 1)
    assert abs(h2.b - 1.7) < 0.05 and abs(h2.b - 1.7) < abs(ls[0] - 1.7) + 1e-12
    assert abs(h2.a - 0.3) < 0.1 < abs(ls[1] - 0.3)
    # a negative slope is clamped: the location-only Huber fit
    yn = 0.5 - 2.0 * x + 0.01 * rng.normal(size=400)
    h3 = AN.huber_fit(x, yn)
    assert h3.clamped and h3.b == 0.0
    loc = AN.huber_fit(np.zeros(400), yn, nonneg=False)
    assert h3.a == pytest.approx(loc.a, rel=1e-6)
    # weights: a zero weight removes a row
    w = np.ones(400)
    w[:20] = 0.0
    assert AN.huber_fit(x, y2, w).b == pytest.approx(1.7, abs=1e-9)


def test_calibration_never_sees_the_held_out_fold(world):
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(2)
    g = lam + 0.1 * rng.normal(size=lam.shape)
    a = AN.calibrate(g, lam, outer, w)
    lam2 = lam.copy()
    lam2[outer == 0] += 100.0                         # changes only fold 0's targets
    b = AN.calibrate(g, lam2, outer, w)
    assert np.allclose(a.pred[outer == 0], b.pred[outer == 0])
    assert np.allclose(a.a[0], b.a[0]) and np.allclose(a.b[0], b.b[0])
    assert not np.allclose(a.b[1:], b.b[1:])


def test_bootstrap_helpers_match_probes(world):
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(3)
    pred = lam[:, 1] + 0.3 * rng.normal(size=len(w))
    d = tab["direction"]
    pt, dr = AN.boot_r2(lam[:, 1], [pred], w, cl, G, counts, d, True)
    bs = PB.boot_sums(lam[:, 1:2], pred[:, None], w, cl, d, ("lam_n",))
    assert pt[0] == pytest.approx(PB.boot_r2(bs, np.ones((1, G)))[0, 0, 0], rel=1e-12)
    assert np.allclose(dr[:, 0], PB.boot_r2(bs, counts)[:, 0, 0])
    assert pt[0] == pytest.approx(PB.r2(lam[:, 1], pred, w, d, ("lam_n",))[0], rel=1e-12)
    # weighted Spearman: the point equals probes.wspearman, each draw equals it on the
    # resampled weights; the odd rule is the direction-weighted within-direction rho
    x = pred
    p0, draws = AN.spearman_boot(x, lam[:, 1], w, cl, counts)
    assert p0 == pytest.approx(PB.wspearman(x, lam[:, 1], w), abs=1e-12)
    wb = w * counts[5][cl]
    assert draws[5] == pytest.approx(PB.wspearman(x[wb > 0], lam[wb > 0, 1], wb[wb > 0]),
                                     abs=1e-12)
    po, _ = AN.spearman_boot(x, lam[:, 1], w, cl, counts, d, True)
    want = sum(w[d == s].sum() * PB.wspearman(x[d == s], lam[d == s, 1], w[d == s])
               for s in (1, -1)) / w.sum()
    assert po == pytest.approx(want, abs=1e-12)


def test_c1_bands_and_the_label(world):
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(4)
    rows = np.ones(len(w), bool)
    d = tab["direction"]
    noise = rng.normal(size=lam.shape)
    sd = lam.std(0)
    good = lam + 0.15 * sd * noise
    weak = {"value_ridge": lam + 2.0 * sd * rng.normal(size=lam.shape),
            "random_critics": [rng.normal(size=lam.shape) for _ in range(4)]}
    c = AN.c1(good, lam, outer, w, cl, G, d, rows, weak, None, K_P, counts)
    assert c.decision == "confirm" and c.label and c.ci[0] > 0.6
    assert all(v[1] > 0 for v in c.null_diff.values())
    assert np.all(np.abs(c.slope - 1.0) < 0.1)
    c10 = AN.c1(10.0 * good, lam, outer, w, cl, G, d, rows, weak, None, K_P, counts)
    assert c10.decision == "confirm" and not c10.label
    neg = AN.c1(-good, lam, outer, w, cl, G, d, rows, None, None, K_P, counts)
    assert neg.decision == "refute" and neg.ci[1] < 0.2 and not neg.label
    assert np.all(neg.calibrated.b == 0.0)           # every slope clamped
    rnd = AN.c1(noise, lam, outer, w, cl, G, d, rows, None, None, K_P, counts)
    assert rnd.decision == "refute"
    # a null as good as the gradient: its paired CI reaches 0, so no confirm
    same = AN.c1(good, lam, outer, w, cl, G, d, rows, {"qss": good}, None, K_P, counts)
    assert same.null_diff["qss"][1] <= 0.0 and same.decision != "confirm"
    # one sign-flipped component blocks the confirm (its Spearman CI is below 0)
    flip = good.copy()
    flip[:, 2] *= -1
    assert AN.c1(flip, lam, outer, w, cl, G, d, rows, None, None, K_P, counts).decision != \
        "confirm"
    # a component's valid mask leaves its rows out of both the fit and the score
    valid = np.ones_like(lam, bool)
    valid[: len(w) // 2, 0] = False
    cv = AN.c1(good, lam, outer, w, cl, G, d, rows, None, valid, K_P, counts)
    assert cv.n_rows[0] == len(w) - len(w) // 2 and cv.n_rows[1] == len(w)


def test_c1_secondaries(world):
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(5)
    n = len(w)
    d = tab["direction"]
    names = (*K_P, "vy", "r", "delta", "pedal")
    lam8 = np.hstack([lam, rng.normal(size=(n, 4))])
    g8 = lam8 + np.hstack([0.1 * rng.normal(size=(n, 4)), 3.0 * rng.normal(size=(n, 4))])
    pat = AN.c1_pattern(g8, lam8, outer, w, cl, G, d, np.ones(n, bool), np.ones((n, 8), bool),
                        names, counts=counts)
    assert pat["decision"] == "confirm" and pat["diff"] > 0.5
    # C1-heads: heads share V; the gradients differ by c_k du with a closed form
    c = np.linspace(-1, 1, 5)
    du = rng.normal(size=(n, 8))
    gq = rng.normal(size=(n, 8))
    q = {"V": np.repeat(rng.normal(size=n)[:, None], 5, 1)}
    for i, nm in enumerate(names):
        scale = 0.1 if nm in K_P else 2.0
        q[f"g_{nm}"] = gq[:, i:i + 1] + scale * c[None] * du[:, i:i + 1]
    heads = AN.c1_heads(q, w, cl, G, np.ones(n, bool), counts)
    assert heads["D"]["V"] <= 1e-20
    i = 2
    within = np.var(c, ddof=1) * (0.1 * du[:, i]) ** 2
    mean = gq[:, i] + 0.1 * c.mean() * du[:, i]
    ww = w / w.sum()
    assert heads["D"]["g_xi"] == pytest.approx((ww @ within) / (ww @ (mean - ww @ mean) ** 2),
                                               rel=1e-10)
    assert heads["grad_KP_minus_V"]["decision"] == "confirm"
    assert heads["grad_secondary_minus_KP"]["decision"] == "confirm"
    # C1-limit: at contacts the calibrated gradient points away from the binding wall
    g = lam + 0.05 * rng.normal(size=lam.shape)
    cal = AN.calibrate(g, lam, outer, w)
    side = np.where(np.arange(n) % 7 == 0, np.where(np.arange(n) % 2 == 0, 1, -1), 0)
    away = g.copy()
    away[:, 1] -= 0.5 * side                          # left wall: more negative g_n
    away[:, 2] -= 0.5 * side
    sd = lam.std(0)
    lim = AN.c1_limit(cal, away, lam, outer, w, cl, G, side, sd, K_P, counts)
    assert lim["decision"] == "confirm" and lim["n"] == int(np.sum(side != 0))
    toward = g.copy()
    toward[:, 1] += 0.5 * side
    toward[:, 2] += 0.5 * side
    assert AN.c1_limit(cal, toward, lam, outer, w, cl, G, side, sd, K_P, counts)["decision"] \
        == "not confirmed"


def test_c1_dev_and_value_accuracy():
    up = [(10 ** (7 + i / 4), 0.1 * i) for i in range(16)]
    down = [(t, -r) for t, r in up]
    rec = {"G1": up, "G1s1": up, "G1s2": down, "G2": up, "G2s1": down, "G2s2": down}
    out = AN.c1_dev(rec)
    assert out["rho"]["G1"] == pytest.approx(1.0) and out["rho"]["G2s1"] == pytest.approx(-1.0)
    assert out["variants"]["v1"] == "confirm"
    assert out["variants"]["v2"] == "not confirmed" and out["variants"]["v3"] == "not testable"
    rng = np.random.default_rng(6)
    v = rng.normal(size=500)
    assert AN.value_accuracy(v + 0.1 * rng.normal(size=500), v, np.ones(500))["pass"]
    assert not AN.value_accuracy(v + 1.0 * rng.normal(size=500), v, np.ones(500))["pass"]


def _exact_c3(x, y, strata):
    """The stratified permutation distribution by enumeration (tiny strata)."""
    groups = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)
    N = len(x)
    per = []
    for idx in groups.values():
        rx = AN._centred_ranks(np.array([x[i] for i in idx]))
        ry = AN._centred_ranks(np.array([y[i] for i in idx]))
        den = np.sqrt((rx @ rx) * (ry @ ry))
        per.append([len(idx) / N * (rx @ ry[list(p)]) / den
                    for p in itertools.permutations(range(len(idx)))])
    return np.array([sum(t) for t in itertools.product(*per)])


def test_c3_w_against_enumeration_and_its_bound():
    strata = {f"n{i}": f"s{i // 4}" for i in range(12)}
    rng = np.random.default_rng(7)
    x = {k: float(rng.normal()) for k in strata}
    y = {k: x[k] + 0.3 * float(rng.normal()) for k in strata}
    res = AN.c3_w(x, y, strata, n_perm=60_000, rng=np.random.default_rng(8))
    names = list(strata)
    exact = _exact_c3([x[k] for k in names], [y[k] for k in names], [strata[k] for k in names])
    obs = res["rho"]
    assert obs == pytest.approx(np.mean([v["rho"] for v in res["strata"].values()]))
    p_exact = np.mean(exact >= obs - 1e-12)
    assert abs(res["p"] - p_exact) < 0.005
    assert res["upper_bound"] == pytest.approx(obs - np.quantile(exact, 0.05), abs=0.03)
    assert res["decision"] == "confirm"
    # a strong negative association: no reversed refute, but the bound rules a strong one out
    yn = {k: -v for k, v in y.items()}
    neg = AN.c3_w(x, yn, strata, n_perm=20_000, rng=np.random.default_rng(9))
    assert neg["decision"] == "rules out a strong association" and neg["p"] > 0.5
    # ties in y (completion 1.0) and a constant stratum contribute rho = 0
    yt = {k: (1.0 if strata[k] == "s0" else y[k]) for k in strata}
    tie = AN.c3_w(x, yt, strata, n_perm=2_000, rng=np.random.default_rng(10))
    assert tie["strata"]["s0"]["rho"] == 0.0
    # the design's strata: 31 networks, 7 strata; covariates as partial associations
    xs = {k: float(rng.normal()) for k in AN.C3_STRATA}
    ys = {k: float(rng.normal()) for k in AN.C3_STRATA}
    full = AN.c3_w(xs, ys, n_perm=20_000, rng=np.random.default_rng(11),
                   covariates={"competence": {k: float(rng.normal()) for k in AN.C3_STRATA}})
    assert full["n"] == 31 and len(full["strata"]) == 7 and not full["missing"]
    assert 0.15 < full["null_sd"] < 0.25
    assert "competence" in full["partial"]


def test_c3_int_double_centring_and_a_planted_pairing():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(6, 9))
    dc = AN.double_centre(A)
    want = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    assert np.allclose(dc, want, atol=1e-12)
    B = A.copy()
    B[0, 0] = np.nan
    dB = AN.double_centre(B)
    assert np.isnan(dB[0, 0]) and np.allclose(np.nanmean(dB, 1), 0, atol=1e-9)
    n_net, n_geo = 8, 24
    blocks = np.arange(n_geo) % 3
    net_eff = rng.normal(size=(n_net, 1))
    geo_eff = rng.normal(size=(1, n_geo))
    inter = rng.normal(size=(n_net, n_geo))
    res = {"n": net_eff + geo_eff + inter, "vx": rng.normal(size=(n_net, n_geo))}
    beh = {"G_minus_1": -inter + 0.5 * rng.normal(size=(n_net, n_geo)) + 3 * net_eff,
           "B": rng.normal(size=(n_net, n_geo))}
    out = AN.c3_int(res, beh, blocks, n_perm=400, rng=np.random.default_rng(13))
    assert out["pairs"]["n~G_minus_1"] > 0.7 and out["decision"] == "confirm"
    null = AN.c3_int({"n": rng.normal(size=(n_net, n_geo))},
                     {"G_minus_1": rng.normal(size=(n_net, n_geo))}, blocks, n_perm=400,
                     rng=np.random.default_rng(14))
    assert null["decision"] == "not confirmed"


def test_variant_rule_and_untestable_c2():
    per = {"G1": {"C1": "confirm", "C2": "refute", "C2-sw": "inconclusive"},
           "G1s1": {"C1": "confirm", "C2": "inconclusive", "C2-sw": "inconclusive"},
           "G1s2": {"C1": "refute", "C2": "confirm", "C2-sw": "missing"},
           "G2": {"C1": "refute"}, "G2s1": {"C1": "refute"}, "G2s2": {"C1": "refute"}}
    v = AN.variant_decisions(per)
    assert v["v1"]["C1"]["verdict"] == "confirm" and v["v1"]["C1"]["seeds_agreeing"] == 2
    assert v["v1"]["C2"]["verdict"] == "seed-specific"
    assert v["v1"]["C2-sw"]["verdict"] == "inconclusive"
    assert v["v2"]["C1"]["verdict"] == "refute" and v["v2"]["C1"]["seeds_agreeing"] == 3
    assert v["v3"]["C1"]["verdict"] == "not testable"
    assert AN.network_decisions(None, {"decision": "confirm"}, None) == \
        {"C1": "missing", "C2": "confirm", "C2-sw": "missing"}
    rec = AN.c2_network(None, [], None, None, None, None, None, None, K_P,
                        dict.fromkeys(K_P, False), np.ones(3, bool))
    assert rec["decision"] == "not testable"
    assert AN.network_of("G1") == "runs/dr-pilot-200M" and len(AN.GENERALISTS) == 31
    assert set(AN.C3_STRATA.values()) == {"base-v1", "base-v2", "base-v3", "real-v1",
                                          "real-v3", "loo-f1", "loo-f2"}
    assert {AN.network_of(a) for a in AN.DECISION_NETWORKS} <= set(AN.GENERALISTS)


# ------------------------------------------------------------------------- the calibration's floor


def _skewed_world(n_geo: int = 36, keys: int = 4, steps: int = 12, seed: int = 20):
    """A world whose targets are skewed (lambda_s: a heavy right tail, median about 0) and
    direction-dependent (the odd components' means differ by direction), with little
    between-geometry mean variation: where a pooled Huber calibration floors an
    uninformative gradient well below 0 (review finding on C1's calibration)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_geo):
        R, th = 10.0 + 5.0 * g, (90.0, 180.0, 45.0)[g % 3]
        for k in range(keys):
            d = 1 if k % 2 == 0 else -1
            for t in range(steps):
                e = rng.normal(size=8)
                lam = np.array([np.exp(1.2 * e[0]) - 1.0, d * (0.6 + 0.3 * e[1]),
                                d * (1.0 + 0.4 * e[2]), 0.5 * e[3],
                                d * (0.8 + 0.3 * e[4]), d * (0.5 + 0.2 * e[5]),
                                d * (0.3 + 0.2 * e[6]), np.exp(0.8 * e[7]) - 1.0])
                rows.append((g, R, th, f"g{g}k{k}", d, lam))
    tab = {"R": np.array([r[1] for r in rows]), "theta": np.array([r[2] for r in rows]),
           "key_id": np.array([r[3] for r in rows]), "direction": np.array([r[4] for r in rows]),
           "pair_id": np.array([r[0] for r in rows])}
    lam = np.stack([r[5] for r in rows])
    w = ST.weights(tab, "key")
    lam = np.stack([ST.winsorise(lam[:, k], ST.winsor_bounds(lam[:, k], w))
                    for k in range(8)], 1)
    return tab, lam, w


NAMES8 = (*K_P, "vy", "r", "delta", "pedal")


def test_uninformative_gradients_score_about_zero_and_never_confirm_c1_pattern():
    """Review finding (C1's calibration floor): with the pooled Huber calibration a zero,
    pure-noise or -lambda gradient scored R2_c of -0.16 on T's K_P mean and C1-pattern
    confirmed on a zero gradient. Within-direction intercepts and the training-fold mean
    intercept put every uninformative gradient at about 0 on every component."""
    tab, lam, w = _skewed_world()
    cl, G = AN.geometry_clusters(tab)
    outer = ST.row_folds(tab)
    counts = PB.cluster_counts(G, 400, np.random.default_rng(21))
    d = tab["direction"]
    n = len(w)
    rows = np.ones(n, bool)
    rng = np.random.default_rng(22)
    bounds = np.stack([lam.min(0), lam.max(0)], 1)
    dirg = np.repeat(np.asarray(d, float)[:, None], 8, 1)
    for label, g in (("zero", np.zeros_like(lam)), ("noise", rng.normal(size=lam.shape)),
                     ("negated", -lam), ("direction", dirg)):
        c = AN.c1(g[:, :4], lam[:, :4], outer, w, cl, G, d, rows, None, None, K_P, counts,
                  bounds=bounds[:4])
        assert np.all(c.r2c >= -0.02) and np.all(c.r2c <= 0.005), (label, c.r2c)
        assert c.decision == "refute", label
        pat = AN.c1_pattern(g, lam, outer, w, cl, G, d, rows, np.ones((n, 8), bool), NAMES8,
                            counts=counts, bounds=bounds)
        assert pat["decision"] == "not confirmed", (label, pat)
        assert pat["ci"][0] <= 0.0 <= pat["ci"][1], (label, pat)
    # the old pooled Huber location would floor the skewed and the direction-dependent ones
    old = AN.huber_fit(np.zeros(n), lam[:, 2], w)
    pooled = 1.0 - (w @ (lam[:, 2] - old.a) ** 2) / ST.wvar(lam[:, 2], w, d) / w.sum()
    assert pooled < -0.3
    # a planted positive pattern still confirms
    g8 = lam + np.hstack([0.1 * rng.normal(size=(n, 4)), 3.0 * rng.normal(size=(n, 4))])
    pat = AN.c1_pattern(g8, lam, outer, w, cl, G, d, rows, np.ones((n, 8), bool), NAMES8,
                        counts=counts, bounds=bounds)
    assert pat["decision"] == "confirm"


def _snr_noise(lam, w, d, snr, rng):
    odd = np.array([PB.is_odd(f"lam_{c}") for c in K_P])
    sd = np.array([np.sqrt(ST.wvar(lam[:, k], w, d if odd[k] else None)) for k in range(4)])
    return rng.normal(size=lam.shape) * sd / np.sqrt(snr)


def test_the_scale_label_is_the_frisch_bracket(world):
    """Review finding (the 'at scale' slope): the slope of lambda on g is about R2 / c, so a
    half-scale gradient got the label. The label now needs b_gl (g on lambda) >= 0.5 and
    1 / b_lg <= 2 on every component."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(23)
    d = tab["direction"]
    rows = np.ones(len(w), bool)
    eta = _snr_noise(lam, w, d, 4.0, rng)
    half = AN.c1(0.45 * (lam + eta), lam, outer, w, cl, G, d, rows, None, None, K_P, counts)
    assert half.decision == "confirm" and not half.label
    assert np.all(np.abs(half.slope_gl - 0.45) < 0.08)
    one = AN.c1(lam + eta, lam, outer, w, cl, G, d, rows, None, None, K_P, counts)
    assert one.decision == "confirm" and one.label
    assert np.all(one.slope_gl_ci[:, 0] >= 0.5) and np.all(one.inv_slope_ci[:, 1] <= 2.0)
    rec = one.record()
    assert rec["slope_gl"] and rec["inv_slope_lg_ci"] and rec["calibration"]["slope"] == "wls"


def test_a_few_high_leverage_gradients_cannot_wreck_c1(world):
    """Review finding (leverage): four rows of one key with a 100 SD gradient spike crushed
    the calibration slope of every fold that trains on them (0.9 -> 0.28 here; a null's R2_c
    went to -18 on the dry-run world). The leverage weights keep every fold's slope and the
    bounds clip the spiked rows' own predictions: away from the spiked key C1 is unchanged."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(24)
    d = tab["direction"]
    rows = np.ones(len(w), bool)
    g = lam + _snr_noise(lam, w, d, 9.0, rng)
    bounds = np.stack([lam.min(0), lam.max(0)], 1)
    key = np.asarray(tab["key_id"])
    hit = np.flatnonzero(key == key[0])[:4]
    spiked = g.copy()
    spiked[hit] += 100.0 * lam.std(0)
    odd = np.array([PB.is_odd(f"lam_{c}") for c in K_P])
    a = AN.calibrate(g, lam, outer, w, direction=d, odd=odd, bounds=bounds)
    b = AN.calibrate(spiked, lam, outer, w, direction=d, odd=odd, bounds=bounds)
    assert np.max(np.abs(a.b - b.b)) < 0.1, (a.b, b.b)      # bounded: K_LEV^2 s^2 per row
    assert np.all(b.clip_share <= a.clip_share + 4 / len(w) + 0.01)
    tr = outer != (outer[hit[0]] + 1) % 6                      # a fold training on the spike
    plain = AN._wls(spiked[tr, 0], lam[tr, 0], w[tr])[1]      # no leverage weights
    assert plain < 0.3
    rest = key != key[0]
    clean = AN.c1(g, lam, outer, w, cl, G, d, rest, None, None, K_P, counts, bounds=bounds,
                  fit_rows=rows)
    c = AN.c1(spiked, lam, outer, w, cl, G, d, rest, None, None, K_P, counts, bounds=bounds,
              fit_rows=rows)
    assert c.decision == "confirm" and abs(c.stat - clean.stat) < 0.03, (c.stat, clean.stat)
    # without the bounds the spiked rows' own predictions are unbounded, but finite
    raw = AN.c1(spiked, lam, outer, w, cl, G, d, rows, None, None, K_P, counts)
    assert np.all(np.isfinite(raw.r2c))


def test_a_component_mask_excludes_cells_from_the_decision(world):
    """Review finding (K2b and multi-basin never reached the decisions): a gradient equal to
    lambda outside the excluded cells and anti-aligned inside them confirms only when the
    decision mask (``states.decision_mask``) is applied."""
    tab, lam, w, cl, G, outer, counts = world
    n = len(w)
    rng = np.random.default_rng(25)
    phase = np.where(np.arange(n) % 12 < 4, "corner", "approach")
    t2 = {**tab, "phase": phase, "multi_basin": np.zeros(n, bool)}
    mask = ST.decision_mask(t2, None, {"corner": ["n", "xi"], "approach": []})
    assert not mask[phase == "corner", 1].any() and mask[phase == "corner", 0].all()
    g = lam + 0.05 * rng.normal(size=lam.shape)
    bad = phase == "corner"
    g[bad, 1] = -5.0 * lam[bad, 1]
    g[bad, 2] = -5.0 * lam[bad, 2]
    d = tab["direction"]
    rows = np.ones(n, bool)
    assert AN.c1(g, lam, outer, w, cl, G, d, rows, None, mask, K_P, counts).decision == "confirm"
    assert AN.c1(g, lam, outer, w, cl, G, d, rows, None, None, K_P, counts).decision != "confirm"
    mb = {**t2, "multi_basin": np.arange(n) < 12}
    assert not ST.decision_mask(mb)[:12].any() and ST.decision_mask(mb)[12:].all()
