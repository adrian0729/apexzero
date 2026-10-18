"""P2 gates on a small oval: convergence, constraint satisfaction, simulator consistency."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.ocp.check import shooting_check
from apexzero.ocp.min_time import solve_min_time
from apexzero.oracle.qss import GG, lap_time, speed_profile
from apexzero.tracks.track import from_centreline
from apexzero.vehicle.dynamics import tyre_forces
from apexzero.vehicle.ops import JAX
from apexzero.vehicle.params import F1_2025 as P


def oval(straight=300.0, radius=60.0, width=12.0):
    pts = []
    for x in np.linspace(0, straight, 60, endpoint=False):
        pts.append((x, -radius))
    for a in np.linspace(-np.pi / 2, np.pi / 2, 60, endpoint=False):
        pts.append((straight + radius * np.cos(a), radius * np.sin(a)))
    for x in np.linspace(straight, 0, 60, endpoint=False):
        pts.append((x, radius))
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 60, endpoint=False):
        pts.append((radius * np.cos(a), radius * np.sin(a)))
    xy = np.array(pts)
    w = np.full(len(xy), width / 2)
    return from_centreline("oval", xy, w, w, ds=2.0, smoothing=0.05)


@pytest.fixture(scope="module")
def solved():
    tr = oval()
    return tr, solve_min_time(tr, P, ds=3.0, max_iter=2000)


def test_ocp_converges_and_beats_centreline_qss(solved):
    tr, res = solved
    assert res.success, res.status
    qss = lap_time(speed_profile(tr.kappa, tr.ds, GG.from_params(P)), tr.ds)
    # Using the track width must be faster than driving the centreline.
    assert res.lap_time < qss


def test_ocp_respects_corridor_and_slip_limits(solved):
    _, res = solved
    n = res.z[:, 0]
    assert np.all(n <= res.w_left - 1.0 + 1e-4)
    assert np.all(n >= -(res.w_right - 1.0) - 1e-4)
    jax.config.update("jax_enable_x64", True)
    fo = tyre_forces(jnp.asarray(np.c_[res.s, res.z]), P, JAX)
    assert np.abs(np.asarray(fo.alpha_f)).max() <= P.alpha_max + 1e-4
    assert np.abs(np.asarray(fo.alpha_r)).max() <= P.alpha_max + 1e-4
    assert np.all(np.abs(res.u) <= 1.0 + 1e-6)


def test_ocp_line_uses_the_width_in_corners(solved):
    """Out-in-out: the optimal line should approach both edges somewhere on the lap."""
    _, res = solved
    n = res.z[:, 0]
    assert n.max() > res.w_left.max() - 1.0 - 0.5
    assert n.min() < -(res.w_right.max() - 1.0) + 0.5


def test_ocp_is_consistent_with_the_simulator(solved):
    _, res = solved
    c = shooting_check(res, P)
    assert abs(c.rel_lap_error) < 1e-3          # plan gate is 0.3%
    assert c.max_position_error < 0.2
