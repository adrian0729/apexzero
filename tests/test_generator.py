from pathlib import Path

import jax
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from apexzero.tracks.corners import R_DEV, R_TEST, geometry_stats, segment
from apexzero.tracks.generator import N_POINTS, fit_grammar, generate_strip, to_track
from apexzero.tracks.reference import drivable_margin
from apexzero.tracks.track import load_tumftm

TUMFTM = Path("data/tracks/tumftm")
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")


@pytest.fixture(scope="module")
def grammar():
    return fit_grammar([load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV])


def test_dev_and_lockbox_are_disjoint():
    assert not set(R_DEV) & set(R_TEST)


def test_strips_are_deterministic_and_well_formed(grammar):
    gen = jax.jit(jax.vmap(lambda k: generate_strip(k, grammar)))
    a = gen(jax.random.split(jax.random.PRNGKey(0), 50))
    b = gen(jax.random.split(jax.random.PRNGKey(0), 50))
    assert np.array_equal(np.asarray(a.kappa_c), np.asarray(b.kappa_c))
    assert a.x.shape == (50, N_POINTS)
    for i in range(50):
        tr = to_track(jax.tree.map(lambda x, i=i: x[i], a))
        assert np.all(np.isfinite(tr.kappa)) and np.all(np.isfinite(tr.x))
        assert 1.0 / np.abs(tr.kappa).max() >= 8.0 - 1e-3          # radius floor
        assert drivable_margin(tr) <= 0.6 + 1e-3                   # Frenet safety cap
        assert np.all(tr.w_left > 1.5) and np.all(tr.w_right > 1.5)
        assert np.abs(tr.kappa[:250]).max() < 1e-6                  # starts on a >= 500 m straight


def test_corner_statistics_match_development_circuits(grammar):
    a = jax.jit(jax.vmap(lambda k: generate_strip(k, grammar)))(jax.random.split(jax.random.PRNGKey(1), 200))
    gen = geometry_stats([segment(to_track(jax.tree.map(lambda x, i=i: x[i], a))) for i in range(200)])
    dev = geometry_stats([segment(load_tumftm(TUMFTM / f"{n}.csv")) for n in R_DEV])
    for key in ("r_min", "abs_angle", "corner_length"):
        assert wasserstein_distance(gen[key], dev[key]) / np.median(dev[key]) < 0.3, key
    assert abs(gen["alternation"].mean() - dev["alternation"].mean()) < 0.1


def test_mixed_bank_draws_from_both_grammars(grammar):
    from apexzero.env.adapters import generate_bank
    from apexzero.tracks.generator import tilted

    bank = generate_bank(jax.random.PRNGKey(3), grammar, 120, tilted(grammar), alt_fraction=0.5)
    tracks = [to_track(jax.tree.map(lambda a, i=i: a[i], bank.tracks)) for i in range(120)]
    r_med = np.array([np.median(geometry_stats([segment(t)])["r_min"]) for t in tracks])
    plain = generate_bank(jax.random.PRNGKey(3), grammar, 120)
    r_plain = np.array([np.median(geometry_stats([segment(to_track(jax.tree.map(lambda a, i=i: a[i], plain.tracks)))])["r_min"]) for i in range(120)])
    assert np.median(r_med) < 0.85 * np.median(r_plain)          # the mixture is tighter
    assert np.all(np.isfinite(np.asarray(bank.v_ref)))
