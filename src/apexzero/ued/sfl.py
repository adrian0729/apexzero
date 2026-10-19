"""Racing SFL, "Sampling For Learnability" (spec 2.3-2.4).

Every ``refresh_every`` chunks, with the parameters frozen at that chunk boundary, the
whole pool is scored: M independent stochastic windows per level from its straight start
(``audit.rollouts.rollout_windows``, 1% observation noise, the audit's own rollouts), the
success of a window is S1 (it was completed, not crashed; spec 1.5), p = mean S1 and the
learnability is p (1 - p). The learnable set D is the top K by learnability. The pool is
shuffled first, so the stable sort breaks ties at random, as fresh random levels do in the
reference (DEV-21). D is fully replaced at every refresh (no staleness, no re-scoring in
between), as in the reference.

Between refreshes the last round(rho N) slots draw uniformly with replacement from D at
every episode reset and the others draw uniformly from the pool; every transition is
trained on (SFL has no PLR-perp masking).

Departures from the reference code (DEV-21): N is the shuffled pool instead of fresh
levels, M windows instead of one L-step AutoReplay stream (DEV-10), draws at each reset
instead of each update, T counted in chunks, and NaN learnabilities raise instead of being
selected by ``argsort``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import numpy as np

from apexzero.agents.ppo import PPOConfig
from apexzero.audit.rollouts import Snapshot, rollout_windows
from apexzero.env import racing
from apexzero.ued.pool import PoolBank

SOURCE_POOL, SOURCE_BUFFER = 0, 1


@dataclass(frozen=True)
class SFLConfig:
    refresh_every: int = 50      # T, in chunks (PPO updates)
    windows: int = 4             # M windows per level and refresh
    buffer_size: int = 1000      # K
    rho: float = 0.5             # fraction of slots drawing from D
    obs_noise: float = 0.01      # as in the audit rollouts
    level_chunk: int = 256       # levels scored in parallel


class Refresh(NamedTuple):
    buffer: np.ndarray        # [K] pool indices (D)
    learnability: np.ndarray  # [L] p (1 - p), pool order
    success: np.ndarray       # [L, M] S1 per window
    distance: np.ndarray      # [L, M] metres (S2 needs the QSS distance, applied offline)
    perm: np.ndarray          # [L] the shuffle used for tie-breaking


def top_k(learnability, k: int, perm=None) -> np.ndarray:
    """Indices of the K most learnable levels. Without ``perm`` this is the reference's
    ``argsort(l)[-K:]`` (stable ascending sort: ties go to the higher index); with a random
    ``perm`` the ties are broken at random. NaN raises (the reference would select it)."""
    ell = np.asarray(learnability, float)
    if np.isnan(ell).any():
        raise ValueError("NaN learnability: every level needs its scoring windows")
    perm = np.arange(ell.size) if perm is None else np.asarray(perm)
    return perm[np.argsort(ell[perm], kind="stable")[-k:]]


def slot_sources(n_envs: int, rho: float) -> np.ndarray:
    """Per slot: SOURCE_BUFFER for the last round(rho N) slots, SOURCE_POOL for the rest."""
    n_d = round(rho * n_envs)
    src = np.full(n_envs, SOURCE_POOL, np.int32)
    src[n_envs - n_d:] = SOURCE_BUFFER
    return src


def refresh(snap: Snapshot, cfg: PPOConfig, bank: PoolBank, key: jax.Array, scfg: SFLConfig,
            params: racing.EnvParams = racing.EnvParams()) -> Refresh:
    """Score the pool with frozen parameters and pick D.

    The windows run in pool order and the scores are then read in shuffled order; since
    every window draws its own noise, this equals scoring in shuffled order."""
    kp, kr = jax.random.split(key)
    L = bank.size
    perm = np.asarray(jax.random.permutation(kp, L))
    ro = rollout_windows(snap, cfg, bank.tracks, bank.x0, bank.ctx, bank.hidden_mu, scfg.windows, kr,
                         steps=params.window_steps, obs_noise=scfg.obs_noise,
                         level_chunk=scfg.level_chunk, params=params)
    s1 = ~np.asarray(ro.terminated, bool)
    if np.any(s1 & (ro.alive.sum(-1) < params.window_steps)):
        raise AssertionError("a scoring window reached the end of its strip")
    p = s1.mean(-1)
    ell = p * (1.0 - p)
    return Refresh(top_k(ell, scfg.buffer_size, perm), ell, s1, np.asarray(ro.distance), perm)
