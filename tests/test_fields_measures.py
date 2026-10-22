"""Step 2's measures (fields.measures) on synthetic fields with known answers (design Section 6).

- Edge gains: a line that holds its clearance (dn = h at the argmin) has G = 1, a width-blind
  line G = 0; G_out is the line's outward move at the argmin; r_in is the speed ratio.
- Bump fields of a single-sample impulse response are the bump itself; a Gaussian field's
  peak, centroid, W90 and E90 are its centre, its centre, 3.29 sigma and centre + 1.28 sigma;
  gain, cosine and location shift of scaled and shifted copies are the scale, 1 and the shift.
- Pairs: the median of 8 keys with both directions weighted equally, undefined beyond 3
  exclusions; P7's 16-phase cells take the median over phases first; the asymmetry index.
- Agents: the geometry medians and a cluster bootstrap that resamples whole geometries.
Pure NumPy; no network or solver."""

import numpy as np
import pytest

from apexzero.fields import measures as MS
from apexzero.fields import perturb as PT
from apexzero.interp.stimuli import Landmarks

LM = Landmarks(400.0, 420.0, 440.0, 460.0, 480.0)
N = 400


def _impulse(w_left=None, w_right=None, kappa=None) -> PT.Samples:
    z = np.zeros(N)
    return PT.Samples(z if kappa is None else kappa, z if w_left is None else w_left,
                      z if w_right is None else w_right)


def _point(j: int, value: float = 1.0) -> np.ndarray:
    e = np.zeros(N)
    e[j] = value
    return e


def test_edge_gains_of_planted_lines():
    s = PT.sample_s(N)
    j_star = int(LM.s_apex / 2.0)              # the argmin sits in the trapezoid's plateau
    assert PT.trapezoid(s, LM.s_c0, LM.s_c1)[j_star] == 1.0
    for sigma in (1, -1):
        inside = PT.side_feature(sigma, "inside")
        outside = PT.side_feature(sigma, "outside")
        # c = w_in - 1 - sigma n read at the argmin: the edge's own move is +1 per metre
        edge = {inside: _point(j_star)}
        blind = _impulse(**edge)
        assert MS.edge_gain(blind, LM, sigma) == pytest.approx(0.0, abs=1e-12)
        # a line that moves with the edge (sigma dn = dDelta at s*) cancels it: G = 1
        holder = _impulse(**{inside: np.zeros(N)})
        assert MS.edge_gain(holder, LM, sigma) == pytest.approx(1.0, abs=1e-12)
        # a line that follows the edge by 0.3: dc/dDelta = 0.7
        part = _impulse(**{inside: 0.7 * _point(j_star)})
        assert MS.edge_gain(part, LM, sigma) == pytest.approx(0.3, abs=1e-12)
        # G_out: the line's outward move at s* per metre of outside widening
        out = _impulse(**{inside: _point(j_star), outside: 0.25 * _point(j_star)})
        em = MS.edge_measures(out, _impulse(**{inside: 2.0 * _point(j_star)}), 0.4, LM, sigma)
        assert em.G == pytest.approx(0.0, abs=1e-12) and em.G_out == pytest.approx(0.25)
        assert em.B == 0.4 and em.dv_in == pytest.approx(2.0)
    # outside the trapezoid the edge shift does not reach the argmin
    far = _impulse(w_left=_point(10))
    assert MS.edge_gain(far, LM, 1) == pytest.approx(1.0)
    assert MS.speed_ratio(0.5, 2.0) == 0.25 and np.isnan(MS.speed_ratio(1.0, 0.0))
    nan = PT.Samples(*(np.full(N, np.nan) for _ in range(3)))
    assert np.isnan(MS.edge_gain(nan, LM, 1))


def test_corridor_translation_gain_T():
    """Design Section 15: T = G + G_out is the line's inward move when the whole corridor
    translates inward (inside widened, outside narrowed). A line held at a fixed fraction f
    of the road from the inside (inside, centre or outside anchored) reads dc/dDelta_in = f
    and dc/dDelta_out = f, so G = 1 - f, G_out = f and T = 1; a width-blind (fixed-n) line has
    G = G_out = T = 0; and T equals the directional derivative of the line's move along the
    translation (Delta_in = +1, Delta_out = -1)."""
    j_star = int(LM.s_apex / 2.0)
    for sigma in (1, -1):
        inside = PT.side_feature(sigma, "inside")
        outside = PT.side_feature(sigma, "outside")
        for f in (0.0, 0.3, 0.5, 1.0):
            imp = _impulse(**{inside: f * _point(j_star), outside: f * _point(j_star)})
            em = MS.edge_measures(imp, imp, 1.0, LM, sigma)
            assert em.G == pytest.approx(1 - f) and em.G_out == pytest.approx(f)
            assert em.T == pytest.approx(1.0)
            shift = PT.edge_shift(N, LM, sigma, "inside", 1.0)
            back = PT.edge_shift(N, LM, sigma, "outside", -1.0)
            corridor = PT.Samples(*(np.asarray(a) + np.asarray(b) for a, b in zip(shift, back,
                                                                                    strict=True)))
            # the line's move sigma dn = e(s*) - dc/dDelta along the translation (e = 1 here)
            assert em.T == pytest.approx(1.0 - MS.along(imp, corridor))
        blind = MS.edge_measures(_impulse(**{inside: _point(j_star)}), _impulse(), 1.0, LM, sigma)
        assert blind.G == pytest.approx(0.0) and blind.G_out == pytest.approx(0.0)
        assert blind.T == pytest.approx(0.0)


def test_bump_field_of_a_point_impulse_is_the_bump():
    s1, sb = 300.0, 8.0
    d = PT.d_grid(80.0)
    j = 170                                      # s_j = 340 = s1 + 40
    f = MS.bump_field(_point(j, 2.0), s1, d, sb)
    np.testing.assert_allclose(f, 2.0 * PT.gaussian_bump(d, 40.0, sb), rtol=0, atol=1e-15)
    assert MS.peak(f, d) == 40.0
    # stacked impulses ([outcomes, N]) map to [outcomes, M]
    two = MS.bump_field(np.stack([_point(j), _point(j + 5)]), s1, d, sb)
    assert two.shape == (2, len(d)) and MS.peak(two[1], d) == 50.0
    assert MS.support_end(np.array([[100.0, 0, 0, 30.0, 0, 0, 0, 0],
                                    [110.0, 0, 0, 90.0, 0, 0, 0, 0]])) == 110.0 + 500.0 + 4.0
    assert MS.d_grid_to(354.0, 300.0)[-1] == 54.0


@pytest.mark.parametrize("sigma_b", PT.SIGMA_B)
def test_shape_of_a_gaussian_field(sigma_b):
    d = PT.d_grid(400.0)
    mu = 120.0
    f = -3.0 * PT.gaussian_bump(d, mu, sigma_b)
    sh = MS.shape(f, d)
    assert sh.amplitude == pytest.approx(3.0) and sh.peak == mu
    assert sh.centroid == pytest.approx(mu, abs=1e-9)
    assert abs(sh.e90 - (mu + MS.E90_BLUR * sigma_b)) <= 0.25           # well within a bin
    assert abs(sh.w90 - 2 * 1.6448536 * sigma_b) <= PT.D_STEP           # a grid's resolution
    # reach is blurred by the bump: the same field at another width moves E90 by 1.28 dsigma
    g = PT.gaussian_bump(d, mu, 2 * sigma_b)
    assert MS.e90(g, d) - MS.e90(f, d) == pytest.approx(MS.E90_BLUR * sigma_b, abs=0.3)


def test_comparisons_of_scaled_and_shifted_fields():
    d = PT.d_grid(300.0)
    opt = PT.gaussian_bump(d, 60.0, 8.0) - 0.4 * PT.gaussian_bump(d, 110.0, 16.0)
    c = MS.compare(0.7 * opt, opt, d)
    assert c["g"] == pytest.approx(0.7) and c["rho"] == pytest.approx(1.0)
    assert c["location"] == pytest.approx(0.0, abs=1e-12)
    shifted = PT.gaussian_bump(d, 75.0, 8.0) - 0.4 * PT.gaussian_bump(d, 125.0, 16.0)
    assert MS.location_shift(shifted, opt, d) == pytest.approx(15.0, abs=0.01)   # 2 m grid
    assert MS.cosine(shifted, opt) < 0.9
    # a zero network field is a low gain (0), not missing; a zero reference is undefined
    assert MS.gain(np.zeros_like(d), opt) == 0.0
    assert np.isnan(MS.gain(opt, np.zeros_like(d))) and np.isnan(MS.cosine(opt, 0 * opt))
    crit = MS.h1_criteria(c, rho_far=0.8)
    assert crit == {"gain": False, "location": True, "shape": True}
    assert MS.h1_criteria(MS.compare(opt, opt, d), 0.8) == {"gain": True, "location": True,
                                                           "shape": True}
    # shape is specificity: a matched rho at or below the far-geometry median fails
    assert not MS.h1_criteria(MS.compare(opt, opt, d), 1.0)["shape"]
    assert not MS.h1_criteria(c, float("nan"))["shape"]


def test_turn_recoding_of_fields():
    k, wl, wr = _point(5), _point(6, 2.0), _point(7, 3.0)
    imp = PT.Samples(k, wl, wr)
    left = MS.turn_fields(imp, 1)
    right = MS.turn_fields(imp, -1)
    np.testing.assert_array_equal(left["w_in"], wl)
    np.testing.assert_array_equal(right["w_in"], wr)
    np.testing.assert_array_equal(right["kappa_turn"], -k)
    steer = MS.turn_fields(imp, -1, "steer")
    np.testing.assert_array_equal(steer["w_in"], -wr)          # the steer is flipped too
    np.testing.assert_array_equal(steer["kappa_turn"], k)


def _rows(values_left, values_right, pair=7, R=20.0, theta=90.0, n_phases=4, approach="A2"):
    out = []
    for sig, vals in ((1, values_left), (-1, values_right)):
        for ph, v in enumerate(vals):
            out.append(MS.KeyMeasure(f"k{pair}{sig}{ph}", pair, approach, sig, ph, n_phases, R,
                                     theta, float(v)))
    return out


def test_weighted_median():
    v = np.array([3.0, 1.0, 2.0, 10.0])
    assert MS.weighted_median(v, np.ones(4)) == np.median(v)
    assert MS.weighted_median(v[:3], np.ones(3)) == 2.0
    # two equally weighted values: their midpoint
    assert MS.weighted_median([1.0, 5.0], [0.5, 0.5]) == 3.0
    # one direction with 4 keys, the other with 1: the lone key carries half the weight
    vals = [0.0, 0.0, 0.0, 0.0, 10.0]
    w = [0.125] * 4 + [0.5]
    assert MS.weighted_median(vals, w) == pytest.approx(5.0)
    assert np.isnan(MS.weighted_median([np.nan], [1.0]))


def test_pair_values_and_exclusions():
    rows = _rows([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
    pv = MS.pair_value(rows)
    assert pv.defined and pv.value == 4.5 and pv.n_keys == 8 and pv.n_excluded == 0
    # 3 exclusions: defined, both directions weighted equally (right has 1 key left)
    rows3 = _rows([1.0, 2.0, 3.0, 4.0], [np.nan, np.nan, np.nan, 20.0])
    pv3 = MS.pair_value(rows3)
    assert pv3.defined and pv3.n_excluded == 3
    w = [1 / 8] * 4 + [1 / 2]
    assert pv3.value == pytest.approx(MS.weighted_median([1, 2, 3, 4, 20], w))
    assert pv3.value > np.median([1, 2, 3, 4, 20])
    # 4 exclusions: undefined
    rows4 = _rows([1.0, np.nan, 3.0, 4.0], [np.nan, np.nan, np.nan, 20.0])
    assert not MS.pair_value(rows4).defined and np.isnan(MS.pair_value(rows4).value)
    # one direction reported alone: 4 keys, at most 1 excluded
    assert MS.pair_value(rows3, (1,)).value == 2.5
    assert MS.pair_value(_rows([1.0, np.nan, 3.0, 4.0], [0] * 4), (1,)).defined
    assert not MS.pair_value(_rows([1.0, np.nan, np.nan, 4.0], [0] * 4), (1,)).defined
    # P7's 16-phase cells: the median over phases first, then the two directions' mean
    left = np.arange(16.0)
    right = 100 + np.arange(16.0)
    p7 = MS.pair_value(_rows(left, right, n_phases=16))
    assert p7.value == pytest.approx(0.5 * (np.median(left) + np.median(right)))
    with pytest.raises(ValueError, match="twin"):
        MS.pair_value(rows + rows[:1])
    with pytest.raises(ValueError, match="more than one"):
        MS.pair_value(rows + _rows([1] * 4, [1] * 4, pair=8))


def test_pair_completeness_counts_the_listed_keys():
    """Design Section 15: a pair's expected keys are its keys in the frozen list (the rows
    passed), not 2 x n_phases; undefined above 3/8 of them excluded; directions weighted
    equally; a direction with no valid key leaves the pair to the other."""
    nominal = _rows([1.0], [3.0])                          # a nominal-only pair: 2 keys
    pv = MS.pair_value(nominal)
    assert pv.defined and pv.n_keys == 2 and pv.n_excluded == 0 and pv.value == 2.0
    one = MS.pair_value(_rows([1.0], [np.nan]))            # 1 of 2 excluded: undefined
    assert not one.defined and np.isnan(one.value)
    assert MS.pair_value(nominal, (1,)).defined and MS.pair_value(nominal, (1,)).n_keys == 1
    assert not MS.pair_value(_rows([np.nan], [3.0]), (1,)).defined
    assert [MS.max_excluded(n) for n in (1, 2, 4, 5, 8, 32)] == [0, 0, 1, 1, 3, 12]
    # pair 189's shape: 4 left keys and 1 right key
    five = _rows([0.0, 0.0, 0.0, 0.0], [10.0])
    pv5 = MS.pair_value(five)
    assert pv5.n_keys == 5 and pv5.defined
    assert pv5.value == pytest.approx(MS.weighted_median([0, 0, 0, 0, 10], [1 / 8] * 4 + [1 / 2]))
    assert pv5.value == pytest.approx(5.0)                 # the lone right key carries half
    lost_right = MS.pair_value(_rows([0.0, 1.0, 2.0, 3.0], [np.nan]))
    assert lost_right.defined and lost_right.value == pytest.approx(1.5)   # left alone
    assert not MS.pair_value(_rows([0.0, np.nan, 2.0, 3.0], [np.nan])).defined
    # a full 8-key pair is unchanged: 3 exclusions defined, 4 undefined
    assert MS.pair_value(_rows([1, 2, 3, 4], [np.nan, np.nan, np.nan, 20])).defined


def test_asymmetry_index():
    rows = _rows([1.0, 2.0, np.nan, 4.0], [0.5, 1.0, 1.0, 1.0])
    asym = MS.asymmetry(rows)
    assert [a["phase"] for a in asym] == [0, 1, 3]
    assert [a["delta"] for a in asym] == [0.5, 1.0, 3.0]
    assert MS.asymmetry_pairs(rows) == {(7, "A2"): 1.0}
    # a duplicate partner (two keys claiming one pair, approach, phase and direction) raises
    with pytest.raises(ValueError, match="duplicate mirror partner"):
        MS.asymmetry(rows + rows[:1])


def test_asymmetry_summary_over_pairs():
    """The index every summary carries (analysis.asymmetry_summary): per pair the median delta
    over its partnered phases, then the median over pairs with a cluster bootstrap over
    geometries; a duplicate partner leaves it undefined with the reason, never silently."""
    from apexzero.fields import analysis as A

    rows = (_rows([1.0, 2.0, np.nan, 4.0], [0.5, 1.0, 1.0, 1.0], pair=7, R=20.0)
            + _rows([0.0] * 4, [2.0] * 4, pair=8, R=40.0)
            + _rows([3.0] * 4, [0.0] * 4, pair=9, R=80.0))
    a = A.asymmetry_summary(rows, n=500)
    assert a["n_pairs"] == 3 and a["n_phases"] == 11
    assert a["est"] == pytest.approx(1.0)                 # median of {1, -2, 3}
    assert a["lo"] <= a["est"] <= a["hi"]
    assert A.summarise(rows, None, 500)["asymmetry"] == a
    assert "asymmetry" not in A.summarise(rows, None, 500, asym=False)
    one_side = [r for r in rows if r.direction == 1]
    assert A.asymmetry_summary(one_side, n=500)["n_pairs"] == 0
    bad = A.asymmetry_summary(rows + rows[:1], n=500)
    assert bad["n_pairs"] == 0 and "duplicate" in bad["error"] and np.isnan(bad["est"])


def test_cluster_bootstrap_resamples_whole_clusters():
    rng = np.random.default_rng(0)
    # one cluster: every draw is the same data, so the interval collapses on the estimate
    est, lo, hi = MS.cluster_bootstrap(rng.normal(size=12), np.zeros(12), n=500, rng=rng)
    assert lo == hi == est
    # clusters of identical members: the statistic only takes the cluster values
    v = np.repeat([1.0, 2.0, 3.0, 4.0], 3)
    c = np.repeat([0, 1, 2, 3], 3)
    est, lo, hi = MS.cluster_bootstrap(v, c, n=4000, rng=np.random.default_rng(1))
    assert est == 2.5 and 1.0 <= lo <= est <= hi <= 4.0
    # unequal cluster sizes take the slow path and agree with an equal-size reference
    est2, lo2, hi2 = MS.cluster_bootstrap(np.r_[v, 4.0], np.r_[c, 3], n=4000,
                                          rng=np.random.default_rng(1))
    assert 1.0 <= lo2 <= hi2 <= 4.0
    # reproducible from the step 2 stream
    a = MS.cluster_bootstrap(v, c, n=200, rng=MS.stream_rng("bootstrap"))
    b = MS.cluster_bootstrap(v, c, n=200, rng=MS.stream_rng("bootstrap"))
    assert a == b


def test_agent_summary_over_geometries_and_directions():
    rows = []
    geoms = [(10.0, 90.0), (20.0, 90.0), (40.0, 90.0), (20.0, 180.0), (40.0, 180.0)]
    for i, (R, th) in enumerate(geoms):
        g = 0.2 + 0.1 * i
        rows += _rows([g] * 4, [g + 0.05] * 4, pair=i, R=R, theta=th)
    # a second pair at the first geometry (another width): the geometry takes their median
    rows += _rows([0.4] * 4, [0.4] * 4, pair=99, R=10.0, theta=90.0)
    summ = MS.agent_summary(rows, n=2000, rng=np.random.default_rng(3))
    assert summ.n_pairs == 6 and summ.n_defined == 6 and summ.n_geometries == 5
    gv = MS.geometry_values(summ.pairs)
    assert gv[(10.0, 90.0)] == pytest.approx(np.median([0.225, 0.4]))
    assert summ.median == pytest.approx(np.median(list(gv.values())))
    assert summ.lo <= summ.median <= summ.hi
    dirs = MS.by_direction(rows, n=500)
    assert dirs["left"].median < dirs["pooled"].median < dirs["right"].median
    assert dirs["left"].as_dict()["directions"] == [1]
    # an undefined pair drops out of the geometry values
    rows_bad = rows + _rows([np.nan] * 4, [1.0] * 4, pair=50, R=80.0, theta=90.0)
    s2 = MS.agent_summary(rows_bad, n=200)
    assert s2.n_pairs == 7 and s2.n_defined == 6 and s2.n_geometries == 5


def test_h2_bands():
    assert MS.h2_decision(0.1, 0.45) == "confirm"
    assert MS.h2_decision(0.85, 1.1) == "lawful"
    assert MS.h2_decision(1.25, 1.6) == "over-tracking"
    assert MS.h2_decision(0.4, 0.9) == "inconclusive"
    assert MS.h2_decision(np.nan, 0.2) == "inconclusive"


def _corner(s_c0, width, sigma=1, shift=30.0):
    s = PT.sample_s(N)
    return PT.Samples(sigma * PT.gaussian_bump(s, s_c0 + shift, width), np.zeros(N), np.zeros(N))


def test_cross_geometry_null():
    """Recoded fields aligned at s_c0: keys of near geometry are not compared; a corner-shaped
    field shifted by the corner's own position is the same field after alignment, and a
    right-hand mirror (kappa negated) is the same field after recoding. The null reports its
    median, not an extreme order statistic."""
    fields = {"a": ((10.0, 90.0), 300.0, 1, _corner(300.0, 10.0)),
              "b": ((20.0, 90.0), 400.0, -1, _corner(400.0, 10.0, sigma=-1)),
              "c": ((20.5, 90.0), 350.0, 1, _corner(350.0, 30.0)),
              "x": ((40.0, 90.0), 300.0, 1, None)}
    assert MS.far_geometry((10.0, 90.0), (20.0, 90.0))
    assert not MS.far_geometry((20.0, 90.0), (20.5, 90.0))       # merged: the same geometry
    assert MS.far_geometry((20.0, 90.0), (20.5, 135.0))           # 45 deg apart
    null = MS.cross_geometry_null(fields, sigma_b=8.0)
    pairs = {(p["a"], p["b"]): p for p in null["pairs"]}
    assert set(pairs) == {("a", "b"), ("a", "c")}                 # b, c share a geometry
    assert pairs[("a", "b")]["rho"] == pytest.approx(1.0)         # the mirror, recoded
    assert pairs[("a", "b")]["g"] == pytest.approx(1.0)
    assert pairs[("a", "c")]["rho"] < 0.95
    assert null["rho_median"] == pytest.approx(np.median([p["rho"] for p in null["pairs"]]))
    assert np.isnan(MS.cross_geometry_null({"a": fields["a"]})["rho_median"])


def test_shape_specificity_and_its_controls():
    """delta_rho = rho(net_k, opt_k) - median rho(net_k, opt_j) over far geometries j: the
    optimum's own field at another phase (a positive control) passes, a far geometry's field
    substituted for the network's (a negative control) fails."""
    geo = {"k1": (10.0, 90.0), "k2": (20.0, 90.0), "k3": (40.0, 90.0), "k4": (80.0, 90.0)}
    width = {"k1": 6.0, "k2": 12.0, "k3": 24.0, "k4": 48.0}
    opt = {k: (geo[k], 300.0, 1, _corner(300.0, width[k], shift=10.0 + width[k]))
           for k in geo}
    # the optimum at another phase: the same field, slightly shifted and scaled
    phase = {k: (geo[k], 300.0, 1, PT.Samples(*(0.9 * np.roll(np.asarray(a), 1)
                                                   for a in opt[k][3])))
             for k in geo}
    rows = MS.shape_specificity(phase, opt, sigma_b=8.0)
    assert {r["label"] for r in rows} == set(geo)
    assert all(r["delta_rho"] > 0.05 and r["n_far"] >= 1 for r in rows), rows
    # a far geometry's field in place of the network's: no specificity
    swapped = {"k1": (geo["k1"], 300.0, 1, opt["k4"][3]), "k4": (geo["k4"], 300.0, 1, opt["k1"][3])}
    bad = {r["label"]: r for r in MS.shape_specificity(swapped, opt, sigma_b=8.0)}
    assert bad["k1"]["delta_rho"] < 0 and bad["k4"]["delta_rho"] < 0
    # the phase ceiling of one cell: its phases' fields are near copies
    pc = MS.phase_ceiling({0: (300.0, 1, opt["k2"][3]), 1: (300.0, 1, phase["k2"][3]),
                           2: (300.0, 1, None)}, sigma_b=8.0)
    assert pc["n_pairs"] == 1 and pc["rho_median"] > 0.99 and pc["rho_min"] == pc["rho_median"]


def test_edge_move_and_the_plateau_rule():
    """e(s*), the edge's own move at c_min's pick, is the spline of the sampled trapezoid: 1
    on the plateau, below 1 on its ramps (G = 1 - dc/dDelta is then not the line's move and
    the key is excluded), interpolated between the pick's states at a window end."""
    from apexzero.fields import outcomes as O

    s_states = np.array([300.0, 330.0, 360.0, 380.0, 400.0, 440.0, 470.0, 495.0, 515.0, 525.0])
    on = O.Pick(5, 5, 0.0, False)                         # s = 440, inside [380, 500]
    assert MS.edge_move(LM, N, s_states, on) == pytest.approx(1.0, abs=1e-12)
    ramp = O.Pick(8, 8, 0.0, False)                       # s = 515: h = 0.25 on the ramp
    e_ramp = MS.edge_move(LM, N, s_states, ramp)
    assert 0.2 < e_ramp < 0.3
    end = O.Pick(7, 8, 505.0, True)                        # interpolated at s* = 505
    e7, e8 = (MS.edge_move(LM, N, s_states, O.Pick(k, k, 0.0, False)) for k in (7, 8))
    assert MS.edge_move(LM, N, s_states, end) == pytest.approx(e7 + (505 - 495) / 20 * (e8 - e7))
    assert np.isnan(MS.edge_move(LM, N, s_states, None))
    j_star = 220                                           # s = 440 (sample index)
    blind = _impulse(w_left=_point(j_star))
    em = MS.edge_measures(blind, blind, 0.5, LM, 1, s_states=s_states, pick=on)
    assert em.on_plateau and em.e_star == pytest.approx(1.0)
    off = MS.edge_measures(blind, blind, 0.5, LM, 1, s_states=s_states, pick=ramp)
    assert not off.on_plateau and off.e_star == pytest.approx(e_ramp)
    # an invalid c_min (NaN buffer) makes every measure NaN, whatever the rows hold
    nanm = MS.edge_measures(blind, blind, float("nan"), LM, 1, s_states=s_states, pick=on)
    assert all(np.isnan(getattr(nanm, f)) for f in ("G", "G_out", "B", "dc_in", "dv_in"))


def test_exclusion_reasons_in_order_and_pipeline_failures():
    assert MS.exclusion_reason(False) == ""
    assert MS.exclusion_reason(True, True, True, True, True) == "invalid outcome"
    assert MS.exclusion_reason(False, True, True) == "gate 3"
    assert MS.exclusion_reason(False, False, True, True) == "inertia"
    assert MS.exclusion_reason(False, False, False, True, True) == "kappa clipping"
    assert MS.exclusion_reason(False, off_plateau=True) == "argmin off plateau"
    assert MS.exclusion_reason(False, False, True, numerical=True) == "numerical"
    assert MS.exclusion_reason(False, True, numerical=True) == "gate 3"
    # Gate 4 is its own reason, after inertia and before kappa clipping
    assert MS.exclusion_reason(False, gate4_failed=True) == "gate 4"
    assert MS.exclusion_reason(False, False, True, gate4_failed=True) == "inertia"
    assert MS.exclusion_reason(False, kappa_clipped=True, gate4_failed=True) == "gate 4"
    # design Section 15: a pipeline failure is an exclusion of that key, counted first
    assert MS.exclusion_reason(True, why_invalid=MS.LOOP_EXHAUSTED) == MS.PIPELINE_LOOP
    assert MS.exclusion_reason(False, loop_exhausted=True) == MS.PIPELINE_LOOP
    assert MS.exclusion_reason(True, True, no_optimum=True) == MS.PIPELINE_NO_OPT
    assert MS.exclusion_reason(True, True, branch_failed=True) == MS.PIPELINE_BRANCH
    assert MS.exclusion_reason(False, no_optimum=True, branch_failed=True) == MS.PIPELINE_NO_OPT
    assert MS.EXCLUSION_REASONS[:3] == MS.PIPELINE_REASONS
    cov = MS.coverage([{"agent": "G1", "snapshot": "1", "outcome": "c_min",
                        "reason": MS.PIPELINE_LOOP, "key_id": "k1"},
                       {"agent": "G1", "snapshot": "1", "outcome": "c_min", "reason": "",
                        "key_id": "k2"}])["G1@1"]["c_min"]
    assert cov["excluded"] == {MS.PIPELINE_LOOP: 1} and cov["pipeline_keys"] == ["k1"]
    assert cov["share_excluded"] == 0.5 and cov["descriptive"]


def test_coverage_counts_and_the_30_percent_rule():
    recs = ([{"agent": "S1", "snapshot": "1", "outcome": "c_min", "reason": "invalid outcome"}] * 7
            + [{"agent": "S1", "snapshot": "1", "outcome": "c_min", "reason": ""}] * 3
            + [{"agent": "G1", "snapshot": "2", "outcome": "c_min", "reason": "gate 3"}]
            + [{"agent": "G1", "snapshot": "2", "outcome": "c_min", "reason": ""}] * 9)
    cov = MS.coverage(recs)
    s1, g1 = cov["S1@1"]["c_min"], cov["G1@2"]["c_min"]
    assert s1["n"] == 10 and s1["excluded"] == {"invalid outcome": 7} and s1["descriptive"]
    assert g1["share_excluded"] == pytest.approx(0.1) and not g1["descriptive"]
    # the boundary: exactly 30% excluded is not descriptive ('exceeds 30%'), 40% is
    for k, want in ((3, False), (4, True)):
        rr = ([{"agent": "A", "snapshot": "1", "outcome": "G", "reason": "gate 3"}] * k
              + [{"agent": "A", "snapshot": "1", "outcome": "G", "reason": ""}] * (10 - k))
        c = MS.coverage(rr)["A@1"]["G"]
        assert c["share_excluded"] == k / 10 and c["descriptive"] is want
