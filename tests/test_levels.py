from pathlib import Path

import jax
import numpy as np
import pytest

from apexzero.oracle.qss import GG
from apexzero.tracks.corners import R_DEV, geometry_stats, segment
from apexzero.tracks.generator import fit_grammar, to_track
from apexzero.tracks.levels import build, make_levels
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025

TUMFTM = Path("data/tracks/tumftm")
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")


@pytest.fixture(scope="module")
def grammar():
    return fit_grammar([load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV])


def test_level_streams_are_disjoint_and_deterministic():
    root = jax.random.PRNGKey(7)
    a = {tuple(k) for k in np.asarray(make_levels(root, "a_dr").keys)}
    b = {tuple(k) for k in np.asarray(make_levels(root, "pool").keys)}
    c = {tuple(k) for k in np.asarray(make_levels(root, "a_tail").keys)}
    assert not (a & b) and not (a & c) and not (b & c)
    k1 = np.asarray(make_levels(root, "a_dr", n=5).keys)
    k2 = np.asarray(make_levels(root, "a_dr", n=5).keys)
    assert np.array_equal(k1, k2)


def test_level_sets_have_their_intended_character(grammar):
    root = jax.random.PRNGKey(7)
    stats, widths = {}, {}
    for stream in ("a_dr", "a_tail", "a_ood"):
        tr, _, x0 = build(make_levels(root, stream, n=60), grammar)
        tracks = [to_track(jax.tree.map(lambda a, i=i: a[i], tr)) for i in range(60)]
        stats[stream] = np.median(geometry_stats([segment(t) for t in tracks])["r_min"])
        widths[stream] = np.median(np.concatenate([t.w_left + t.w_right for t in tracks]))
        top = GG.from_params(F1_2025).top_speed()
        assert np.allclose(np.asarray(x0[:, 3]), 0.8 * top, rtol=1e-3)   # opening straight
        assert np.all(np.asarray(x0[:, [1, 2, 4, 5, 6, 7]]) == 0.0)
    assert stats["a_tail"] < 0.7 * stats["a_dr"]      # hairpin-heavy
    assert widths["a_ood"] < 0.8 * widths["a_dr"]     # narrower
