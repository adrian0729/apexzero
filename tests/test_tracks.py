from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from apexzero.tracks.reference import drivable_margin, refine_reference
from apexzero.tracks.track import from_centreline, load_tumftm

TUMFTM = Path("data/tracks/tumftm")
needs_tumftm = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")


def circle(radius=100.0, width=10.0, n=400):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    w = np.full(n, width / 2)
    return from_centreline("circle", xy, w, w, ds=2.0)


def densify(poly, step=0.05):
    closed = np.vstack([poly, poly[:1]])
    c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    t = np.arange(0, c[-1], step)
    return np.stack([np.interp(t, c, closed[:, 0]), np.interp(t, c, closed[:, 1])], axis=1)


def test_circle_geometry():
    tr = circle()
    assert tr.length == pytest.approx(2 * np.pi * 100.0, rel=1e-4)
    assert np.allclose(tr.kappa, 1 / 100.0, rtol=1e-3)
    assert np.allclose(np.diff(tr.s), tr.ds)
    left, _ = tr.edges()
    assert np.allclose(np.hypot(left[:, 0], left[:, 1]), 95.0, atol=1e-2)


def test_mirror_and_reverse_flip_curvature():
    tr = circle()
    assert np.allclose(tr.mirrored().kappa, -tr.kappa)
    rev = tr.reversed()
    assert rev.length == pytest.approx(tr.length, rel=1e-4)
    assert np.allclose(rev.kappa, -1 / 100.0, rtol=1e-3)


@needs_tumftm
@pytest.mark.parametrize("name,length", [("Monza", 5793), ("Spa", 7004), ("Suzuka", 5807)])
def test_real_circuit_lengths(name, length):
    assert load_tumftm(TUMFTM / f"{name}.csv").length == pytest.approx(length, rel=0.002)


@needs_tumftm
def test_suzuka_figure_eight_has_zero_net_turn():
    tr = load_tumftm(TUMFTM / "Suzuka.csv")
    assert abs(tr.theta[-1] + tr.kappa[-1] * tr.ds - tr.theta[0]) < 0.05


@needs_tumftm
@pytest.mark.parametrize("name", ["Austin", "Shanghai", "Norisring"])
def test_refined_reference_keeps_edges_and_lowers_margin(name):
    tr = load_tumftm(TUMFTM / f"{name}.csv")
    ref = refine_reference(tr)
    assert drivable_margin(ref) < min(0.75, drivable_margin(tr))
    for old, new in zip(tr.edges(), ref.edges()):
        err = cKDTree(densify(old)).query(new)[0]
        assert err.max() < 0.05
