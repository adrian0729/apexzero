"""Idea 2's decision rules (costate.analysis) on synthetic data with known answers (design
v2.1, Section 6 with Section 12's decisions).

- The Huber calibration: exact on clean lines, robust to outliers, the slope clamped at >= 0
  (location only), per fold without the held-out rows.
- The bootstrap helpers against probes' own (the same resamples over a fixed cluster count),
  the vectorised weighted Spearman against ``probes.wspearman``, the within-direction rule.
- C1's bands: a gradient equal to lambda (plus noise) confirms beyond its nulls and earns the
  label at scale; 10 lambda confirms without the label; -lambda and pure noise refute; a null
  as good as the gradient blocks the confirm; C1-pattern, C1-heads (planted heads with a
  closed form), C1-limit's sign, C1-dev's 2-of-3 rule.
- Value accuracy (out of fold on the stratum, the frozen folds; planted values pass, random,
  anti-aligned and weak ones fail) and C1-own's standing; the strata tables' rows.
- C3-w: the stratified permutation p against exact enumeration, the shift-inverted upper bound
  and its decisions; C3-int's double centring and a planted pre-signed association; R18's
  variant rule; C2 'not testable' without a component above the power floor.

- C1's per-fold nulls: a regressor [F, n, m] calibrated fold by fold on its own slice, the
  far-geometry lambda and a value fit's pullback (stubbed) per held-out fold: perturbing fold
  f's targets cannot change fold f's calibrated null predictions (the old one-predictor
  layout let them); rows without an out-of-fold calibration dropped, not zero-filled, and a
  one-geometry stratum 'not testable'; C3-w's partial covariate renormalised over the strata
  it scores.

Pure NumPy (the nulls module only for its far-geometry lambda and, with the pullback stubbed,
its per-fold wiring): no network, no key file."""

import itertools
from types import SimpleNamespace

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
    # Post-tag rehearsal finding (K2b's frozen exclusions leave xi 2 contact rows on T, n
    # none): a component whose contact rows some bootstrap draw leaves out has no CI and
    # drops out (it read 'not confirmed' on a NaN CI); a row without an out-of-fold
    # calibration is dropped, not scored at a zero excess
    one = np.flatnonzero((side != 0) & (cl == cl[side != 0][0]))
    valid = np.zeros((n, 4), bool)
    valid[one, 2] = True
    lim1 = AN.c1_limit(cal, away, lam, outer, w, cl, G, side, sd, K_P, counts, valid=valid)
    assert lim1["decision"] == "not testable" and lim1["per_component"]["xi"]["ci"] is None
    assert lim1["per_component"]["xi"]["geometries"] == 1
    assert lim1["per_component"]["n"]["n"] == 0
    cal_nan = AN.calibrate(g, lam, outer, w)
    cal_nan.a[0] = np.nan                       # fold 0 without a calibration
    lim2 = AN.c1_limit(cal_nan, away, lam, outer, w, cl, G, side, sd, K_P, counts)
    dropped = int(np.sum((side != 0) & (outer == 0)))
    assert dropped and lim2["per_component"]["xi"]["n_uncalibrated"] == dropped
    assert lim2["per_component"]["xi"]["n"] == lim["n"] - dropped
    assert lim2["decision"] == "confirm"


def test_c1_dev_and_value_accuracy():
    up = [(10 ** (7 + i / 4), 0.1 * i) for i in range(16)]
    down = [(t, -r) for t, r in up]
    rec = {"G1": up, "G1s1": up, "G1s2": down, "G2": up, "G2s1": down, "G2s2": down}
    out = AN.c1_dev(rec)
    assert out["rho"]["G1"] == pytest.approx(1.0) and out["rho"]["G2s1"] == pytest.approx(-1.0)
    assert out["variants"]["v1"] == "confirm"
    assert out["variants"]["v2"] == "not confirmed" and out["variants"]["v3"] == "not testable"


def test_value_accuracy_is_out_of_fold_on_the_stratum(world):
    """Value accuracy (Section 3.3 'Value accuracy', Section 12 decision 7, Section 13 'Built
    before the tag'): the out-of-fold R2 of sigma_R V_c against a target (the deterministic
    policy's own return in the decision, ``VALUE_TARGET``) on the requested rows, with the
    frozen folds and C1's affine calibration. A planted critic, the target plus small noise,
    passes, also scaled and offset (the calibration absorbs both, the raw R2 does not);
    anti-aligned, pure-noise and weak values fail; the cut is 0.8 by behaviour, on each side
    of it, and the comparison reads ``threshold``; the held-out fold's target never enters its
    own calibration, through value_accuracy itself; rows outside the stratum do not count; a
    requested row without a finite critic value or target, or without an out-of-fold
    prediction, fails it; a missing target fails. C1-own's standing follows, NumPy booleans
    counting as booleans."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(6)
    n = len(w)
    vstar = 60.0 + 10.0 * lam[:, 3] + 3.0 * lam[:, 0] + 0.5 * rng.normal(size=n)
    rows = rng.random(n) < 0.6
    wi = ST.stratum_weights(tab, rows)
    sd = vstar[rows].std()

    def va(v, target=vstar, **kw):
        return AN.value_accuracy(v, target, outer, wi, rows, cl, G, counts, **kw)

    v_good = vstar + 0.1 * sd * rng.normal(size=n)
    good = va(v_good)
    assert good["pass"] and good["r2"] > 0.97 and good["ci"][0] > 0.95
    assert good["n"] == good["n_scored"] == good["n_requested"] == rows.sum()
    assert good["n_missing"] == good["n_missing_target"] == 0 and "reason" not in good
    assert good["threshold"] == 0.8 and AN.VALUE_ACCURACY_R2 == 0.8
    assert good["target"] == AN.VALUE_TARGET == "v_own_det" and good["rows"] == "own"
    vs = va(v_good, target_name="v_star_inf", rows_name="interior")
    assert vs["target"] == "v_star_inf" and vs["rows"] == "interior" and vs["pass"]
    assert np.allclose(good["slope"], 1.0, atol=0.05)
    scaled = va(0.4 * (vstar + 0.1 * sd * rng.normal(size=n)) - 15.0)
    assert scaled["pass"] and scaled["raw_r2"] < 0.0 and np.allclose(scaled["slope"], 2.5,
                                                                      rtol=0.05)
    # an in-sample affine fit cannot score below 0 (a clamped slope predicts the weighted
    # mean); out of fold the anti-aligned value does
    anti = va(-vstar)
    assert not anti["pass"] and anti["r2"] < -1e-3 and all(anti["clamped"])
    assert not va(rng.normal(size=n))["pass"]
    weak = va(vstar + 1.0 * sd * rng.normal(size=n))
    assert not weak["pass"] and 0.3 < weak["r2"] < 0.7
    # the cut by behaviour: target plus a sd x noise has R2 about 1 / (1 + a^2), 0.862 at
    # a = 0.4 (passes) and 0.768 at a = 0.55 (fails); the comparison reads threshold
    e = np.random.default_rng(11).normal(size=n)
    hi, lo = va(vstar + 0.4 * sd * e), va(vstar + 0.55 * sd * e)
    assert hi["pass"] and 0.82 < hi["r2"] < 0.9, hi["r2"]
    assert not lo["pass"] and 0.72 < lo["r2"] < 0.8, lo["r2"]
    assert va(vstar + 0.55 * sd * e, threshold=0.7)["pass"]
    r = lo["r2"]
    assert va(vstar + 0.55 * sd * e, threshold=r)["pass"]
    assert not va(vstar + 0.55 * sd * e, threshold=np.nextafter(r, 2.0))["pass"]
    # out of fold, through value_accuracy: fold 0's calibration never sees fold 0's targets
    t2 = vstar.copy()
    t2[outer == 0] += 1e3 * rng.normal(size=int((outer == 0).sum()))
    a, b = va(v_good), va(v_good, target=t2)
    assert np.isclose(a["slope"][0], b["slope"][0]) and np.isclose(a["intercept"][0],
                                                                   b["intercept"][0])
    assert not np.isclose(a["slope"][1], b["slope"][1])
    a, b = (AN.calibrate(vstar[:, None], t[:, None], outer, wi, rows, rows[:, None])
            for t in (vstar, t2))
    assert np.allclose(a.pred[outer == 0], b.pred[outer == 0])
    # outside the stratum nothing counts
    v = v_good
    bad = v.copy()
    bad[~rows] = 1e6
    assert va(bad)["r2"] == pytest.approx(va(v)["r2"], rel=1e-12)
    out_only = v.copy()
    out_only[~rows] = np.nan
    assert va(out_only)["pass"] and va(out_only)["n_missing"] == 0
    # every requested row scored: a good value with holes on the stratum fails, the holes
    # counted; so does a target with holes
    holes = v.copy()
    hit = np.flatnonzero(rows)[::3]
    holes[hit] = np.nan
    h = va(holes)
    assert not h["pass"] and h["n_missing"] == len(hit) and h["r2"] > 0.95
    assert h["n_requested"] == rows.sum() and "lack a finite critic value" in h["reason"]
    th = vstar.copy()
    th[hit[:2]] = np.nan
    ht = va(v, target=th)
    assert not ht["pass"] and ht["n_missing_target"] == 2 and "finite target" in ht["reason"]
    # ... and so does a fold that cannot be calibrated (fold 0 trained on 2 rows), even with
    # no threshold at all
    r2m = outer == 0
    r2m[np.flatnonzero(outer == 1)[:2]] = True
    o = AN.value_accuracy(v, vstar, outer, ST.stratum_weights(tab, r2m), r2m, threshold=-np.inf)
    assert o["n_scored"] < o["n"] and np.isfinite(o["r2"]) and not o["pass"]
    assert "out-of-fold prediction" in o["reason"]
    miss = AN.value_accuracy(v, None, outer, wi, rows)
    assert not miss["pass"] and "v_own_det" in miss["reason"]
    # C1-own: a secondary only with both preconditions; NumPy booleans are booleans, and
    # anything else (None, NaN, numbers, strings, arrays) is a failed precondition
    assert AN.c1_own_status(True, True) == "secondary"
    assert AN.c1_own_status(True, False) == "descriptive (value accuracy)"
    assert AN.c1_own_status(False, True) == "descriptive (K3b)"
    assert AN.c1_own_status(False, None) == "descriptive (K3b, value accuracy)"
    assert AN.c1_own_status(np.True_, np.bool_(True)) == "secondary"
    assert AN.c1_own_status(np.array(True), True) == "secondary"
    assert AN.c1_own_status(np.True_, np.bool_(False)) == "descriptive (value accuracy)"
    for odd in (float("nan"), 1.0, 1, "yes", np.array([True, True])):
        assert AN.c1_own_status(odd, True) == "descriptive (K3b)", odd
    g = lam + 0.1 * rng.normal(size=lam.shape)
    d = tab["direction"]
    allr = np.ones(n, bool)
    for k3b, vok, want, valid in ((True, True, "secondary", True),
                                  (True, False, "descriptive (value accuracy)", False),
                                  (True, None, "descriptive (value accuracy)", None),
                                  (np.True_, np.bool_(True), "secondary", True),
                                  (np.True_, float("nan"), "descriptive (value accuracy)", False)):
        rec = AN.c1_own(g, lam, outer, w, cl, G, d, allr, k3b, K_P, counts,
                        value_accuracy_pass=vok)
        assert rec["status"] == want and rec["validity"]["value_accuracy"] is valid
        assert rec["validity"]["k3b_trained"] is True
    few = AN.c1_own(g, lam, outer, w, cl, G, d, np.arange(n) < 5, True, K_P, counts,
                    value_accuracy_pass=False)
    assert few["decision"] == "not testable" and few["status"] == "descriptive (value accuracy)"


def test_strata_rows_and_their_table_records(world):
    """The strata tables' row expressions ('&'-joined columns, '~', 'phase:', 'direction:',
    'W:') and a table row's fields: the statistic, its CI and per-component values, the rows,
    keys and geometries, the band (descriptive) and the primary mark."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(8)
    n = len(w)
    t = {**tab, "t": np.arange(n), "interior": rng.random(n) < 0.5,
         "braking_key": np.repeat(rng.random(n // 12) < 0.7, 12),
         "phase": rng.choice(np.array(["approach", "braking", "corner", "exit"]), n),
         "W": rng.choice([400, 800], n)}
    m = AN.stratum_rows(t, "interior&~braking_key&phase:exit")
    assert np.array_equal(m, t["interior"] & ~t["braking_key"] & (t["phase"] == "exit"))
    assert np.array_equal(AN.stratum_rows(t, "direction:-1&W:800"),
                          (t["direction"] == -1) & (t["W"] == 800))
    with pytest.raises(KeyError, match="no column"):
        AN.stratum_rows(t, "interior&nonsense")
    assert set(AN.strata_rows(t, ("interior", "interior&braking_key"))) == {
        "interior", "interior&braking_key"}
    for h, names in AN.STRATA.items():
        assert names[0] == AN.PRIMARY_STRATUM[h] and len(set(names)) == len(names)
        assert set(AN.strata_rows({**t, "near_limit": ~t["interior"], "contact": t["interior"],
                                   "interior20": t["interior"], "corner_band": t["interior"],
                                   "pedal_saturated": t["interior"], ST.SW_STRATUM: t["interior"],
                                   "kappa_clipped": t["interior"]}, names)) == set(names)
    rows = t["interior"]
    g = lam + 0.2 * rng.normal(size=lam.shape)
    c = AN.c1(g, lam, outer, ST.stratum_weights(t, rows), cl, G, t["direction"], rows, None,
              None, K_P, counts, slope_boot="always")
    assert c.decision == "confirm" and np.all(np.isfinite(c.slope_ci))
    r = AN.stratum_record("C1", c.record(), rows, t, primary=True)
    assert r["primary"] and not r["descriptive"] and r["band"] == "confirm"
    assert r["n"] == rows.sum() and r["stat"] == c.stat and r["keys"] == len(
        np.unique(t["key_id"][rows])) and r["geometries"] == G
    assert len(r["slope_ci"]) == 4 and "calibration" not in r
    # without 'always' a non-confirm leaves the slope CIs empty (the label's only); with it
    # the same non-confirm carries them (Section 6: 'slope CIs per component and stratum')
    noise = rng.normal(size=lam.shape)
    c0 = AN.c1(noise, lam, outer, ST.stratum_weights(t, rows), cl, G,
               t["direction"], rows, None, None, K_P, counts)
    assert c0.decision != "confirm" and np.all(np.isnan(c0.slope_ci))
    ca = AN.c1(noise, lam, outer, ST.stratum_weights(t, rows), cl, G, t["direction"], rows, None,
               None, K_P, counts, slope_boot="always")
    assert ca.decision == c0.decision != "confirm" and not ca.label
    assert np.all(np.isfinite(ca.slope_ci)) and np.all(np.isfinite(ca.slope_gl_ci))
    e = AN.stratum_record("C2", {"decision": "not testable", "reason": "x"}, np.zeros(n, bool), t)
    assert e["descriptive"] and e["band"] == "not testable" and e["n"] == 0 and e["reason"] == "x"
    # a record computed on other rows than its stratum's is never written under it
    assert c.record()["n_stratum"] == rows.sum()
    with pytest.raises(ValueError, match="written under a stratum"):
        AN.stratum_record("C1", c.record(), ~rows, t)
    with pytest.raises(ValueError, match="written under a stratum"):
        AN.stratum_record("C2-sw", {"decision": "refute", "n_stratum": 3}, rows, t)
    # the per-network kappa-clipped column: absent, its strata are None ('not testable');
    # strata_tab adds it from one network's flags
    assert AN.PER_NETWORK_COLUMNS == ("kappa_clipped",)
    got = AN.strata_rows(t, ("interior", "interior&kappa_clipped"))
    assert got["interior&kappa_clipped"] is None and AN.missing_columns(
        t, "interior&kappa_clipped") == ["kappa_clipped"]
    nt = AN.stratum_record("C1", {"reason": "no kappa"}, None, t)
    assert nt["band"] == "not testable" and nt["n"] is None and not nt["primary"]
    kc = rng.random(n) < 0.1
    ts = AN.strata_tab(t, {"kappa_clipped": kc, "observability": None})
    assert np.array_equal(AN.stratum_rows(ts, "interior&kappa_clipped"), t["interior"] & kc)
    assert "kappa_clipped" not in t and set(ts) == set(t) | {"kappa_clipped"}
    assert AN.strata_counts(t, ("interior", "interior&kappa_clipped")) == {
        "interior": int(t["interior"].sum()), "interior&kappa_clipped": None}
    with pytest.raises(KeyError, match="no column"):
        AN.strata_rows(t, ("interior&nonsense",))     # only per-network columns may be absent


def test_the_frozen_strata_lists():
    """The strata tables' lists, pinned literally and in order (frozen into the record at the
    tag; the C1 rows' bootstrap streams are STRATUM_SEED + the row's index, so order matters).
    Sources: Section 12 item 4 (interior20, the corner band); Section 4 'Strata' (near-limit,
    contact, the phases, left and right, W 400 and 800, pedal-saturated, kappa-clipped) and
    'Sub-scores' (the braking and non-braking keys); Section 6 (C2-sw's approach and exit
    reported separately: all four phase parts, which split its stratum); Section 13 (C2-sw's
    pedal-switching stratum, the raw pedal-saturated flag descriptive)."""
    kp = ("interior&direction:1", "interior&direction:-1", "interior&W:400", "interior&W:800",
          "interior&pedal_saturated", "interior&kappa_clipped")
    ph = ("interior&phase:approach", "interior&phase:braking", "interior&phase:corner",
          "interior&phase:exit")
    sw = "pedal_switching"
    assert AN.STRATA == {
        "C1": ("interior", "interior20", "near_limit", "contact", "corner_band",
               "interior&corner_band", *ph, "interior&braking_key", "interior&~braking_key",
               *kp),
        "C2": ("interior", "interior20", "near_limit", "contact", "corner_band",
               "interior&corner_band", "interior&braking_key", "interior&~braking_key", *ph,
               *kp),
        "C2-sw": (sw, f"{sw}&phase:approach", f"{sw}&phase:braking", f"{sw}&phase:corner",
                  f"{sw}&phase:exit", "pedal_saturated", f"{sw}&braking_key",
                  f"{sw}&~braking_key", f"{sw}&direction:1", f"{sw}&direction:-1",
                  f"{sw}&W:400", f"{sw}&W:800", f"{sw}&kappa_clipped")}
    assert AN.PRIMARY_STRATUM == {"C1": "interior", "C2": "interior", "C2-sw": sw}
    assert AN.STRATUM_SEED == 50 and ST.SW_STRATUM == sw
    # C2-sw's four phase rows are disjoint and add up to its stratum on any T
    rng = np.random.default_rng(9)
    n = 500
    t = {"t": np.arange(n), sw: rng.random(n) < 0.4,
         "phase": rng.choice(np.array(ST.PHASES), n)}
    parts = [AN.stratum_rows(t, f"{sw}&phase:{p}") for p in ST.PHASES]
    assert np.array_equal(np.sum(parts, 0), t[sw].astype(int))
    assert set(ST.PHASES) == {"approach", "braking", "corner", "exit"}


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
    assert tie["strata"]["s0"]["rho"] == 0.0 and tie["n_varying"] == 8
    # Post-tag rehearsal finding: with every stratum tied (x the same where the calibration
    # slope clamps at 0, or one network per stratum) the null is the point 0 and its bound
    # of 0 read 'rules out a strong association'; it is not testable
    xt = {k: 0.5 for k in strata}
    flat = AN.c3_w(xt, y, strata, n_perm=2_000, rng=np.random.default_rng(10))
    assert flat["decision"] == "not testable" and flat["n_varying"] == 0 and flat["reason"]
    assert AN.c3_w({}, {}, strata, n_perm=100)["decision"] == "not testable"
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


def test_c1_pattern_drops_the_components_without_a_decision_row():
    """Post-tag rehearsal finding: K2b's frozen exclusions leave n and vx no decision row on
    T, and C1-pattern averaged their NaN R2_c into K_P's mean, so its difference and CI were
    NaN and it read 'not confirmed' for every network. As in C1, such a component drops out
    of its side's mean; with a side empty C1-pattern is 'not testable'. C1's record gives a
    dropped component's null R2_c and Spearman as NaN (they read 0)."""
    tab, lam, w = _skewed_world()
    cl, G = AN.geometry_clusters(tab)
    outer = ST.row_folds(tab)
    counts = PB.cluster_counts(G, 400, np.random.default_rng(21))
    d = tab["direction"]
    n = len(w)
    rows = np.ones(n, bool)
    rng = np.random.default_rng(23)
    g8 = lam + np.hstack([0.1 * rng.normal(size=(n, 4)), 3.0 * rng.normal(size=(n, 4))])
    valid = np.ones((n, 8), bool)
    valid[:, [K_P.index("n"), K_P.index("vx")]] = False
    pat = AN.c1_pattern(g8, lam, outer, w, cl, G, d, rows, valid, NAMES8, counts=counts)
    assert pat["dropped"] == ["n", "vx"] and pat["n_rows"]["n"] == 0
    assert pat["n_rows"]["s"] == n and np.isfinite(pat["diff"]) and np.all(np.isfinite(pat["ci"]))
    assert pat["decision"] == "confirm"
    kept = (pat["r2c"]["s"] + pat["r2c"]["xi"]) / 2 - np.mean(
        [pat["r2c"][c] for c in ("vy", "r", "delta", "pedal")])
    assert pat["diff"] == pytest.approx(kept, abs=1e-12)
    c = AN.c1(g8[:, :4], lam[:, :4], outer, w, cl, G, d, rows, {"zero": np.zeros((n, 4))},
              valid[:, :4], K_P, counts)
    assert list(c.n_rows[[1, 3]]) == [0, 0] and np.isfinite(c.stat)
    assert np.all(np.isnan(c.null_r2c["zero"][[1, 3]]))
    assert np.all(np.isfinite(c.null_r2c["zero"][[0, 2]]))
    assert np.all(np.isnan(c.spearman[[1, 3]])) and np.all(np.isnan(c.spearman_ci[[1, 3]]))
    valid[:, :4] = False
    pat = AN.c1_pattern(g8, lam, outer, w, cl, G, d, rows, valid, NAMES8, counts=counts)
    assert pat["decision"] == "not testable" and "K_P" in pat["reason"]


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


def test_a_per_fold_regressor_keeps_the_held_out_fold_out_of_its_calibration(world):
    """Review finding (fold-dependent C1 nulls): a null [F, n, m] per held-out fold is
    calibrated fold by fold on its own slice, so fold f's fit and predictions read slice f
    and the other folds' targets only; the old layout (one predictor per row, from its own
    fold's slice) let the other slices into fold f's calibration."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(30)
    n, F = len(w), ST.N_FOLDS
    d, odd = tab["direction"], AN._names_odd(K_P)
    X = lam[None] * rng.uniform(0.5, 2.0, size=(F, 1, 4)) + 0.3 * rng.normal(size=(F, n, 4))
    cal = AN.calibrate(X, lam, outer, w, direction=d, odd=odd)
    assert np.all(np.isfinite(cal.pred)) and np.all(cal.b > 0)
    for f in range(F):
        lam2, X2 = lam.copy(), X.copy()
        lam2[outer == f] += 50.0 * rng.normal(size=((outer == f).sum(), 4))
        X2[np.arange(F) != f] = rng.normal(size=(F - 1, n, 4))
        b = AN.calibrate(X2, lam2, outer, w, direction=d, odd=odd)
        assert np.allclose(b.pred[outer == f], cal.pred[outer == f]), f
        assert np.allclose(b.a[f], cal.a[f]) and np.allclose(b.b[f], cal.b[f]), f
    # a 2-D regressor is the same in every slice
    two = AN.calibrate(X[0], lam, outer, w, direction=d, odd=odd)
    same = AN.calibrate(np.broadcast_to(X[0], X.shape), lam, outer, w, direction=d, odd=odd)
    assert np.array_equal(two.pred, same.pred) and np.array_equal(two.b, same.b)
    # the old layout: fold 0's rows keep their predictor, the training rows' change, and so
    # does fold 0's calibration
    X3 = X.copy()
    X3[1:] = rng.normal(size=(F - 1, n, 4))
    old = AN.calibrate(X[outer, np.arange(n)], lam, outer, w, direction=d, odd=odd)
    old3 = AN.calibrate(X3[outer, np.arange(n)], lam, outer, w, direction=d, odd=odd)
    assert not np.allclose(old.pred[outer == 0], old3.pred[outer == 0])
    with pytest.raises(ValueError, match="per-fold regressor"):
        AN.calibrate(X[:3], lam, outer, w)
    # c1 takes it as a null: scored like a 2-D null when every slice is the same
    g = lam + 0.2 * lam.std(0) * rng.normal(size=lam.shape)
    rows = np.ones(n, bool)
    c3 = AN.c1(g, lam, outer, w, cl, G, d, rows, {"far_geometry": np.broadcast_to(X[0], X.shape)},
               None, K_P, counts)
    c2 = AN.c1(g, lam, outer, w, cl, G, d, rows, {"far_geometry": X[0]}, None, K_P, counts)
    assert c3.null_diff == c2.null_diff and np.array_equal(c3.null_r2c["far_geometry"],
                                                           c2.null_r2c["far_geometry"])


def _phase_tab(tab):
    """The toy T with the columns the far-geometry lambda's phase coordinate reads (12 steps
    of 10 m per key: approach to the brake onset at 30 m, braking to the apex at 60 m, the
    corner to 80 m, then the exit)."""
    t = np.tile(np.arange(12), len(tab["R"]) // 12)
    n = len(t)
    return {**tab, "t": t, "s": 10.0 * t, "s_b": np.full(n, 30.0), "s_c0": np.full(n, 25.0),
            "s_apex": np.full(n, 60.0), "s_c1": np.full(n, 80.0)}


def test_the_far_geometry_null_cannot_carry_the_held_out_folds_targets(world):
    """Review finding (fold-dependent C1 nulls): slice f of the far-geometry lambda takes no
    geometry of fold f (and, for a training row, none of its own fold, as
    ``states.cross_fit``), so perturbing fold f's targets changes neither slice f nor fold
    f's calibrated null predictions; under the old construction (each row's own-fold slice)
    it changed them."""
    from apexzero.costate import nulls as NL
    from apexzero.costate.targets import STATE_NAMES

    tab, lam, w, cl, G, outer, counts = world
    t = _phase_tab(tab)
    rng = np.random.default_rng(31)
    n, F = len(w), ST.N_FOLDS
    lam8 = np.concatenate([lam, rng.normal(size=(n, 4))], 1)        # K_P first, then 4 more
    assert tuple(STATE_NAMES[:4]) == K_P
    d, odd = tab["direction"], AN._names_odd(K_P)
    far = NL.far_lambda(t, lam8, outer, w)
    assert far.shape == (F, n, 8) and np.isfinite(far[outer, np.arange(n)]).all()
    cal = AN.calibrate(far[..., :4], lam, outer, w, direction=d, odd=odd)
    old = AN.calibrate(far[outer, np.arange(n), :4], lam, outer, w, direction=d, odd=odd)
    for f in range(F):
        lam2 = lam8.copy()
        lam2[outer == f] += 10.0 * rng.normal(size=((outer == f).sum(), 8))
        far2 = NL.far_lambda(t, lam2, outer, w)
        assert np.allclose(far2[f], far[f], equal_nan=True), f
        assert not np.allclose(far2, far, equal_nan=True)
        # a training row's slice leaves its own fold out too: fold g's targets never reach
        # slice f on fold g's rows
        g_ = (f + 1) % F
        lam3 = lam8.copy()
        lam3[outer == g_] += 10.0 * rng.normal(size=((outer == g_).sum(), 8))
        far3 = NL.far_lambda(t, lam3, outer, w)
        assert np.allclose(far3[f][outer == g_], far[f][outer == g_], equal_nan=True), f
        b = AN.calibrate(far2[..., :4], lam2[:, :4], outer, w, direction=d, odd=odd)
        assert np.allclose(b.pred[outer == f], cal.pred[outer == f]), f
        ob = AN.calibrate(far2[outer, np.arange(n), :4], lam2[:, :4], outer, w, direction=d,
                          odd=odd)
        assert not np.allclose(ob.pred[outer == f], old.pred[outer == f]), f


def test_a_value_fit_null_is_pulled_back_through_each_folds_own_fit(world, monkeypatch):
    """Review finding (fold-dependent C1 nulls): slice f of a value fit's gradient is every
    row pulled back through the fit made without fold f (``nulls.fold_gradients``; the
    pullback stubbed as the linear critic's closed form w'J), so fold f's value targets and
    its lambda change neither slice f nor fold f's calibrated null predictions; under the
    old construction (each row through its own fold's fit) they did."""
    from apexzero.costate import critic as CR
    from apexzero.costate import nulls as NL

    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(32)
    n, F, p = len(w), ST.N_FOLDS, 12
    fo = PB.folds(tab)
    assert np.array_equal(fo.outer, outer)
    z = rng.normal(size=(n, p))
    J = rng.normal(size=(n, p, 8)) + 0.2 * lam[:, None, [0, 1, 2, 3, 0, 1, 2, 3]]

    def pullback(net, tr, inp, oj):
        g = np.einsum("kc,ncd->nkd", net.params[0][:, :p], inp)
        return SimpleNamespace(value=np.zeros((len(inp), 1)), g=g, pi=np.zeros((len(inp), 1, 2)),
                               g_held=g)

    monkeypatch.setattr(CR, "pullback", pullback)
    key = tab["key_id"]
    keys = [NL.KeyInput(k, np.flatnonzero(key == k), J[key == k], None) for k in np.unique(key)]
    v = z @ rng.uniform(0.5, 1.5, p) + 0.1 * rng.normal(size=n)
    zz = np.concatenate([z, np.zeros((n, CR.OBS_DIM - p))], 1)
    d, odd = tab["direction"], AN._names_odd(K_P)

    def nulls_of(v_, lam_):
        fr = NL.fit_value_ridge(zz, v_, fo, w, None)
        g3 = NL.fold_gradients(fr, keys, n, F)[..., :4]
        return g3, AN.calibrate(g3, lam_, outer, w, direction=d, odd=odd), AN.calibrate(
            g3[outer, np.arange(n)], lam_, outer, w, direction=d, odd=odd)

    g3, cal, old = nulls_of(v, lam)
    assert g3.shape == (F, n, 4) and not np.allclose(g3[0], g3[1])
    for f in range(F):
        v2, lam2 = v.copy(), lam.copy()
        v2[outer == f] += 5.0 * rng.normal(size=(outer == f).sum())
        lam2[outer == f] += 5.0 * rng.normal(size=((outer == f).sum(), 4))
        h3, b, ob = nulls_of(v2, lam2)
        assert np.array_equal(h3[f], g3[f]), f
        assert np.allclose(b.pred[outer == f], cal.pred[outer == f]), f
        assert not np.allclose(ob.pred[outer == f], old.pred[outer == f]), f


def test_rows_without_an_out_of_fold_calibration_are_dropped_not_zero_filled(world):
    """Review finding (one-geometry strata scored against zero-filled predictions): a fold
    with fewer than 3 training rows has no calibration, and its rows are dropped from the
    scoring (and counted) instead of scored against 0; a stratum inside one geometry spans one
    outer fold, so nothing is left and C1 is 'not testable' with that reason."""
    tab, lam, w, cl, G, outer, counts = world
    rng = np.random.default_rng(33)
    n = len(w)
    d = tab["direction"]
    g = lam + 0.3 * lam.std(0) * rng.normal(size=lam.shape)
    nl = {"qss": rng.normal(size=lam.shape)}
    one = cl == 5
    c = AN.c1(g, lam, outer, w, cl, G, d, one, nl, None, K_P, counts)
    assert c.decision == "not testable" and "1 outer fold" in c.reason
    assert np.all(c.n_rows == 0) and np.all(c.n_uncalibrated == one.sum())
    r = AN.stratum_record("C1", c.record(), one, tab)
    assert r["band"] == "not testable" and "1 outer fold" in r["reason"] and r["geometries"] == 1
    # two geometries in two folds; fold A's calibration may train on 2 of B's rows only: A's
    # rows are dropped, B's are scored by the fit on A's rows
    ga = 5
    gb = int(cl[np.flatnonzero(outer != outer[cl == ga][0])[0]])
    A, B = cl == ga, cl == gb
    fit = A | (B & (np.cumsum(B) <= 2))
    rows = A | B
    odd = AN._names_odd(K_P)
    with np.errstate(invalid="ignore", divide="ignore"):    # draws without geometry B: nan
        c2 = AN.c1(g, lam, outer, w, cl, G, d, rows, nl, None, K_P, counts, fit_rows=fit)
        assert np.all(c2.n_rows == B.sum()) and np.all(c2.n_uncalibrated == A.sum())
        assert np.all(np.isnan(c2.calibrated.pred[A]))
        assert np.all(np.isfinite(c2.calibrated.pred[B]))
        for k in range(4):
            pt, _ = AN.boot_r2(lam[B, k], [c2.calibrated.pred[B, k]], w[B], cl[B], G, counts,
                               d[B], bool(odd[k]))
            assert c2.r2c[k] == pytest.approx(pt[0], rel=1e-12), k
            zero = np.where(B, c2.calibrated.pred[:, k], 0.0)
            pz, _ = AN.boot_r2(lam[rows, k], [zero[rows]], w[rows], cl[rows], G, counts,
                               d[rows], bool(odd[k]))
            assert pz[0] != pytest.approx(pt[0]), k                # the old zero-filled score
    # rows spanning two folds with no fold calibrated: the generic reason
    few = (A & (np.cumsum(A) <= 2)) | (B & (np.cumsum(B) <= 2))
    c3 = AN.c1(g, lam, outer, w, cl, G, d, few, nl, None, K_P, counts)
    assert c3.decision == "not testable" and "no component keeps" in c3.reason
    # the primary record has no reason and counts no dropped row
    full = AN.c1(g, lam, outer, w, cl, G, d, np.ones(n, bool), nl, None, K_P, counts)
    assert "reason" not in full.record() and full.record()["n_uncalibrated"] == [0, 0, 0, 0]


def test_the_partial_covariate_skips_and_renormalises_strata_without_it():
    """Review finding (C3-w's overlap covariate is missing for 2 of the 31 generalists): a
    stratum holding a network without the covariate is left out of the partial, the size
    weights are renormalised over the strata scored (they were divided by all 31), and the
    networks without it are listed."""
    rng = np.random.default_rng(34)
    xs = {k: float(rng.normal()) for k in AN.C3_STRATA}
    ys = {k: xs[k] + float(rng.normal()) for k in AN.C3_STRATA}
    gone = ("runs/dr-pilot-v2-tilt30-s3", "runs/dr-pilot-v2-tilt30-s4")
    ov = {k: float(rng.normal()) for k in AN.C3_STRATA if k not in gone}
    const = dict.fromkeys(AN.C3_STRATA, 1.0)      # no residualisation: the plain stratum rho
    res = AN.c3_w(xs, ys, n_perm=200, rng=np.random.default_rng(35),
                  covariates={"overlap": ov, "constant": const,
                              "constant_gap": {k: 1.0 for k in ov}})
    p = res["partial"]["overlap"]
    assert p["missing"] == list(gone) and p["strata_skipped"] == ["base-v2"]
    assert p["n"] == 26 and not p["complete"]
    keep = {k: s for k, s in AN.C3_STRATA.items() if s != "base-v2"}
    sub = AN.c3_w({k: xs[k] for k in keep}, {k: ys[k] for k in keep}, keep, n_perm=200,
                  rng=np.random.default_rng(36), covariates={"overlap": ov})
    assert p["rho"] == pytest.approx(sub["partial"]["overlap"]["rho"], abs=1e-12)
    assert sub["partial"]["overlap"]["complete"] and sub["partial"]["overlap"]["missing"] == []
    # a constant covariate gives the plain size-weighted rho, renormalised when a stratum goes
    per = res["strata"]
    assert res["partial"]["constant"]["rho"] == pytest.approx(res["rho"], abs=1e-12)
    want = sum(v["n"] * v["rho"] for s, v in per.items() if s != "base-v2") / 26
    assert res["partial"]["constant_gap"]["rho"] == pytest.approx(want, abs=1e-12)
    none = AN.c3_w(xs, ys, n_perm=100, rng=np.random.default_rng(37), covariates={"x": {}})
    assert np.isnan(none["partial"]["x"]["rho"]) and none["partial"]["x"]["n"] == 0


def test_only_the_designs_conditions_are_not_testable():
    """Review finding: the primary C2 and C2-sw caught any ValueError as 'not testable'.
    ``probes.gain`` raises ``probes.NotTestable`` (a ValueError subclass) for the design's
    two conditions only; a stratum whose folds leave a held-out fold's training rows in one
    inner fold is 'not testable' with that reason (``probe_cv_problem``), not a LinAlgError
    from an empty inner fit; ``guarded`` records an error in place of a descriptive."""
    assert issubclass(PB.NotTestable, ValueError)
    y = np.zeros((4, 1))
    p = PB.Probe(y, y, np.zeros((6, 1, 2)), np.zeros((6, 2), int), ("lam_vx",))
    with pytest.raises(PB.NotTestable, match="no component passes the power floor"):
        PB.gain(p, p, [], np.ones(4), None, passing=[False])
    with pytest.raises(PB.NotTestable, match="has a scored row"):
        PB.gain(p, p, [], np.ones(4), None, passing=[True], mask=np.zeros((4, 1), bool))
    q = PB.Probe(y + 1.0, y + 1.0, np.zeros((6, 1, 2)), np.zeros((6, 2), int), ("lam_vx",))
    with pytest.raises(ValueError, match="same targets") as e:
        PB.gain(p, q, [], np.ones(4), None)
    assert not isinstance(e.value, PB.NotTestable)
    tab, _ = _world()
    fo = PB.folds(tab)
    geo = ST.geometry_index(tab)
    assert AN.probe_cv_problem(fo, np.ones(len(geo), bool)) is None
    two = np.isin(geo, np.unique(geo)[:2])
    why = AN.probe_cv_problem(fo, two)
    assert why is not None and "inner cross-validation" in why
    rec = AN.c2_sw_network(None, [], None, None, None, fo, None, two)
    assert rec == {"decision": "not testable", "reason": why, "n_stratum": int(two.sum())}
    none = AN.c2_network(None, [], None, None, None, fo, None, None, K_P,
                         dict.fromkeys(K_P, True), np.zeros(len(geo), bool))
    assert none["decision"] == "not testable" and none["reason"] == "no row"
    assert AN.guarded(lambda: 1 / 0) == {"error": "ZeroDivisionError: division by zero"}
    assert AN.guarded(dict, a=1) == {"a": 1}
    r = AN.stratum_record("C2", {"error": "KeyError: 'x'"}, two, tab)
    assert r["band"] == "error" and r["error"] == "KeyError: 'x'" and r["n"] == int(two.sum())
