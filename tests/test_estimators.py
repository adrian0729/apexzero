"""Estimators: hand-checkable cases and a differential test against JaxUED's own code."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxued.utils import compute_max_returns
from jaxued.utils import max_mc as ref_max_mc
from jaxued.utils import positive_value_loss as ref_pvl

import apexzero.compat  # noqa: F401  (must precede jaxued)
from apexzero.audit import estimators as E
from apexzero.audit.rollouts import WindowRollouts


def make(reward, value, alive, terminated, final_value, heads=None):
    reward, value, alive = (np.asarray(a, float) for a in (reward, value, alive))
    L, W, T = reward.shape
    heads = np.zeros((L, W, T, 0)) if heads is None else np.asarray(heads, float)
    return WindowRollouts(reward, value, heads, alive, np.asarray(terminated, bool),
                          np.asarray(final_value, float), np.zeros((L, W)), np.zeros((L, W, T)))


def test_gae_hand_example_with_truncation_bootstrap():
    # One level, one window of 3 steps, truncated with bootstrap value 2.
    ro = make([[[1.0, 1.0, 1.0]]], [[[0.5, 0.5, 0.5]]], [[[1, 1, 1]]], [[False]], [[2.0]])
    g, lam = 0.9, 0.5
    d2 = 1 + g * 2.0 - 0.5
    d1 = 1 + g * 0.5 - 0.5
    d0 = 1 + g * 0.5 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    assert np.allclose(E.gae(ro, g, lam)[0, 0], [a0, a1, a2])


def test_termination_does_not_bootstrap_and_masks_after_the_end():
    ro = make([[[1.0, 1.0, 0.0]]], [[[0.5, 0.5, 9.0]]], [[[1, 1, 0]]], [[True]], [[5.0]])
    adv = E.gae(ro, 0.9, 1.0)[0, 0]
    assert adv[2] == 0.0
    assert np.isclose(adv[1], 1.0 - 0.5)                 # no bootstrap after termination
    assert np.isclose(adv[0], (1 + 0.9 * 0.5 - 0.5) + 0.9 * adv[1])


def test_pvl_identity_with_l1():
    rng = np.random.default_rng(0)
    ro = make(rng.normal(size=(3, 4, 20)), rng.normal(size=(3, 4, 20)), np.ones((3, 4, 20)),
              np.zeros((3, 4), bool), rng.normal(size=(3, 4)))
    adv = E.gae(ro, 0.99, 0.95)
    mean_a = (adv.sum(-1) / 20).mean(-1)
    assert np.allclose(E.positive_value_loss(ro, 0.99, 0.95),
                       0.5 * E.l1_value_loss(ro, 0.99, 0.95) + 0.5 * mean_a)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pvl_and_maxmc_match_jaxued_reference(seed):
    """Lay each level's windows end to end as one rollout with done flags, as a PLR loop
    sees them, and compare with JaxUED's positive_value_loss / max_mc."""
    rng = np.random.default_rng(seed)
    L, W, T = 5, 3, 12
    reward = rng.normal(size=(L, W, T))
    value = rng.normal(size=(L, W, T))
    lengths = rng.integers(4, T + 1, size=(L, W))
    alive = (np.arange(T)[None, None, :] < lengths[..., None]).astype(float)
    terminated = lengths < T
    ro = make(reward, value, alive, terminated, rng.normal(size=(L, W)))
    adv = E.gae(ro, 0.99, 0.95)

    # Flatten each level's windows into a [time, level] rollout with done at episode ends,
    # padded at the end with not-done steps (an incomplete trailing episode, which the
    # reference ignores).
    tot = lengths.sum(1).max() + 3

    def flat(a):
        cols = [np.concatenate([a[i, w, : lengths[i, w]] for w in range(W)]) for i in range(L)]
        return np.stack([np.r_[c, np.full(tot - len(c), 0.7)] for c in cols], 1)

    cols = [np.concatenate([np.r_[np.zeros(lengths[i, w] - 1), 1.0] for w in range(W)]) for i in range(L)]
    dones = np.stack([np.r_[c, np.zeros(tot - len(c))] for c in cols], 1)
    dones = dones.astype(bool)  # the reference loop passes boolean dones
    pvl_ref = np.asarray(ref_pvl(jnp.asarray(dones), jnp.asarray(flat(adv))))
    assert np.allclose(E.positive_value_loss(ro, 0.99, 0.95), pvl_ref, atol=1e-5)

    max_ret = np.asarray(compute_max_returns(jnp.asarray(dones), jnp.asarray(flat(reward))))
    mm_ref = np.asarray(ref_max_mc(jnp.asarray(dones), jnp.asarray(flat(value)), jnp.asarray(max_ret)))
    assert np.allclose(E.maxmc_implemented(ro), mm_ref, atol=1e-5)


def test_learnability_and_disagreement():
    succ = np.array([[True, True, False, False], [True, True, True, True]])
    assert np.allclose(E.learnability(succ), [0.25, 0.0])
    heads = np.zeros((1, 1, 2, 3))
    heads[0, 0, :, :] = [[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]]
    ro = make(np.zeros((1, 1, 2)), np.zeros((1, 1, 2)), np.ones((1, 1, 2)), [[False]], [[0.0]], heads)
    assert np.isclose(E.disagreement(ro)[0], 0.5 * np.std([0.0, 1.0, 2.0]))
