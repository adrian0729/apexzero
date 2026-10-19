"""SFL sampler: spec section 4.2 vectors S-1..S-4."""

import jax
import numpy as np
import pytest

from apexzero.ued.sfl import SOURCE_BUFFER, SOURCE_POOL, slot_sources, top_k, window_success


def test_S1_top_k_by_learnability():
    ell = [0.25, 0.0, 0.1875, 0.0, 0.24]
    assert set(top_k(ell, 2).tolist()) == {4, 0}
    assert top_k(ell, 2).tolist() == [4, 0]          # argsort(l)[-K:], ascending


def test_S2_ties_reference_takes_the_highest_index_ours_is_random():
    assert top_k([0.0, 0.0, 0.0], 1).tolist() == [2]
    picks = set()
    for s in range(30):
        perm = np.asarray(jax.random.permutation(jax.random.PRNGKey(s), 3))
        picks.add(int(top_k([0.0, 0.0, 0.0], 1, perm)[0]))
    assert picks == {0, 1, 2}
    # A shuffle never changes which levels win when there are no ties.
    perm = np.asarray(jax.random.permutation(jax.random.PRNGKey(0), 5))
    assert set(top_k([0.25, 0.0, 0.1875, 0.0, 0.24], 2, perm).tolist()) == {4, 0}


def test_S3_nan_learnability_raises():
    # The reference's argsort sorts NaN last, so top-1 would select index 1.
    assert int(np.argsort([0.25, np.nan, 0.1])[-1]) == 1
    with pytest.raises(ValueError):
        top_k([0.25, np.nan, 0.1], 1)


def test_window_success_and_the_end_of_strip_check():
    crashed = np.array([[False, True], [True, False]])
    alive = np.array([[100, 37], [5, 100]])
    assert window_success(crashed, alive, 100).tolist() == [[True, False], [False, True]]
    with pytest.raises(AssertionError, match="end of its strip"):
        window_success(crashed, np.array([[100, 37], [5, 99]]), 100)   # stopped short, not crashed


def test_S4_slot_split():
    src = slot_sources(8, 0.5)
    assert src.tolist() == [SOURCE_POOL] * 4 + [SOURCE_BUFFER] * 4
    assert slot_sources(8, 0.0).tolist() == [SOURCE_POOL] * 8
    assert slot_sources(2048, 0.5).sum() == 1024
