"""d_QSS on a racing line (audit/qss_distance.py): the box QP against scipy, the Frenet line
geometry on circles, the minimum-curvature line on a corner, and the distance itself."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from apexzero.audit import qss_distance as Q
from apexzero.vehicle.params import F1_2025

DS = 2.0


def _corner_strip(n=1500, start=400, radius=60.0, angle=np.pi / 2, ramp=40.0, width=12.0):
    """A straight, a clothoid-ramped left corner of the given radius and angle, a straight."""
    k_peak = 1.0 / radius
    arc = max(angle / k_peak - ramp, 0.0)
    s = np.arange(n) * DS - start
    prof = np.clip(np.minimum(s / ramp, (2 * ramp + arc - s) / ramp), 0.0, 1.0)
    kappa = k_peak * prof
    half = np.full(n, width / 2)
    return kappa, half.copy(), half.copy()


def test_box_qp_matches_lsq_linear():
    rng = np.random.default_rng(0)
    n = 60
    a = sparse.csr_matrix(rng.normal(size=(90, n)))
    b = rng.normal(size=90)
    lo, hi = -0.2 * np.ones(n), 0.3 * np.ones(n)
    ref = lsq_linear(a.toarray(), b, bounds=(lo, hi), method="bvls").x
    x, _ = Q.box_qp(a.T @ a, a.T @ b, lo, hi)
    obj = lambda z: np.sum((a @ z - b) ** 2)
    assert obj(x) <= obj(ref) + 1e-9
    assert np.allclose(x, ref, atol=1e-6)
    assert np.all(x >= lo) and np.all(x <= hi)


def test_line_geometry_centreline_and_offset_circle():
    kappa = np.full(200, 1 / 50.0)
    seg, k = Q.line_geometry(kappa, np.zeros(200), DS)
    assert np.allclose(seg, DS) and np.allclose(k, kappa)
    seg, k = Q.line_geometry(kappa, np.full(200, 5.0), DS)    # 5 m to the left = inside
    assert np.allclose(k, 1 / 45.0) and np.allclose(seg, DS * 45 / 50)
    seg, k = Q.line_geometry(kappa, np.full(200, -5.0), DS)   # outside
    assert np.allclose(k, 1 / 55.0) and np.allclose(seg, DS * 55 / 50)


def test_min_curvature_line_cuts_the_corner_inside_the_corridor():
    kappa, wl, wr = _corner_strip()
    theta, x, y = Q.strip_xy(kappa, DS)
    alpha = Q.min_curvature_offsets(x, y, theta, wl, wr, margin=1.0)
    assert np.all(alpha <= wl - 1.0 + 1e-9) and np.all(alpha >= -wr + 1.0 - 1e-9)
    assert abs(alpha[0]) <= Q.START_TOL + 1e-12                      # starts on the centreline
    apex = int(np.argmax(kappa))
    assert alpha[apex] > 3.0                                          # left turn: apex inside
    _, k_line = Q.line_geometry(kappa, alpha, DS)
    inner = slice(100, -100)                                          # away from the free ends
    assert np.abs(k_line[inner]).max() < kappa.max()                  # straighter than centre
    assert np.sum(k_line[inner] ** 2) < 0.8 * np.sum(kappa[inner] ** 2)


def test_qss_distance_racing_line_beats_centreline_and_v0_matters():
    kappa, wl, wr = _corner_strip()
    kw = dict(ds=DS, seconds=20.0, margin=1.0)                       # ends before the strip
    centre = Q.qss_distance(kappa, wl, wr, F1_2025, 60.0, line="centre", **kw)
    racing = Q.qss_distance(kappa, wl, wr, F1_2025, 60.0, **kw)
    assert racing > centre
    assert Q.qss_distance(kappa, wl, wr, F1_2025, 30.0, line="centre", **kw) < centre
    straight = np.zeros(1500)
    d_c = Q.qss_distance(straight, wl, wr, F1_2025, 60.0, line="centre", **kw)
    d_r = Q.qss_distance(straight, wl, wr, F1_2025, 60.0, **kw)
    assert d_r == pytest.approx(d_c, rel=1e-9)                        # nothing to cut
    with pytest.raises(ValueError):
        Q.qss_distance(kappa, wl, wr, F1_2025, 60.0, line="apex", **kw)


def test_progress_in_time_constant_speed():
    v = np.full(101, 10.0)
    assert Q.progress_in_time(v, np.full(100, DS), DS, 5.0) == pytest.approx(50.0)
    # A line half as long as the centreline: twice the centreline progress in the same time.
    assert Q.progress_in_time(v, np.full(100, DS / 2), DS, 5.0) == pytest.approx(100.0)


def test_ratio_quantiles():
    dist = np.array([[90.0, 110.0], [50.0, 200.0]])
    done = np.array([[True, True], [True, False]])
    q = Q.ratio_quantiles(dist, done, np.array([100.0, 100.0]))
    assert q["n"] == 3 and q["q1"] == pytest.approx(1.1) and q["above_1"] == pytest.approx(1 / 3)
    assert q["q0.5"] == pytest.approx(0.9)
    empty = Q.ratio_quantiles(dist, np.zeros_like(done), np.array([100.0, 100.0]))
    assert empty["n"] == 0 and np.isnan(empty["q1"])
