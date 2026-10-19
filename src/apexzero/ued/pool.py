"""The curriculum pool as a PPO ``Env``: every slot plays the pool level its sampler chose.

Curriculum C (plan 2.4) trains every arm on one labelled pool of paper levels
(``apexzero.tracks.levels``, stream "pool", 4,000 levels). A level fixes the strip, the car
context, the hidden grip and the straight start x0, so an episode is one window
(1,200 steps) from x0, and windows of one level differ only in the policy's randomness, as
in the audit rollouts.

Which level a slot plays next is the sampler's decision, carried in the per-slot state:
- ``source < 0``: reset to ``next_level``, which the trainer pre-draws once per chunk
  (PLR-perp, spec 2.2). A slot that finishes twice in one chunk replays it (AutoReplay).
- ``source = s >= 0``: a fresh uniform draw from table ``s`` of ``PoolBank.sources`` at
  every reset (DR-pool: the whole pool; SFL: the pool or its learnable set D, spec 2.4).
The loss-mask flag travels the same way (``mask`` for the running episode, ``next_mask``
taken at the next reset), which is how PLR-perp tells the learner which windows to train on.

The level pytree stored in the PLR buffer is ``{keys, ctx, hidden_mu, level_id}``: the
compact paper encoding plus the pool index.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from apexzero.agents.ppo import Env
from apexzero.env import racing
from apexzero.tracks import levels as paper_levels
from apexzero.tracks.generator import Grammar
from apexzero.vehicle.sim import TrackArrays


class PoolBank(NamedTuple):
    """Device-resident pool, shared by all slots (the env-parameter carry)."""
    tracks: TrackArrays     # every field stacked along axis 0 (L levels)
    x0: jax.Array           # [L, 8] straight start
    ctx: jax.Array          # [L, 6] car context (observed)
    hidden_mu: jax.Array    # [L] grip multiplier (not observed)
    obs0: jax.Array         # [L, obs_dim] the start observation (x0 is deterministic)
    sources: jax.Array      # [S, L] int32 level-id tables for per-reset uniform draws
    source_len: jax.Array   # [S] int32 valid entries in each table

    @property
    def size(self) -> int:
        return self.x0.shape[0]


class PoolState(NamedTuple):
    env: racing.EnvState
    level: jax.Array        # pool index of the level being played
    next_level: jax.Array   # level taken at the next reset when source < 0
    source: jax.Array       # -1: next_level; s >= 0: uniform draw from sources[s]
    mask: jax.Array         # train on the running episode (PLR-perp: replay windows only)
    next_mask: jax.Array    # mask taken at the next reset


def start_state(x0: jax.Array, ctx: jax.Array, hidden_mu: jax.Array) -> racing.EnvState:
    """The env state at a level's straight start (same as the audit's ``rollout_windows``)."""
    z = jnp.zeros(2, x0.dtype)
    return racing.EnvState(x=x0, prev_action=z, accel=z, ctx=ctx.astype(x0.dtype),
                           hidden_mu=jnp.asarray(hidden_mu, x0.dtype), t=jnp.zeros((), jnp.int32),
                           slow=jnp.zeros((), x0.dtype), s0=x0[0])


def make_bank(levels: paper_levels.Levels, g: Grammar, n_sources: int = 2,
              params: racing.EnvParams = racing.EnvParams()) -> PoolBank:
    """Regenerate the pool's strips and start observations on device. Every table of
    ``sources`` starts as the whole pool."""
    tracks, _, x0 = jax.jit(lambda keys: paper_levels.build(levels._replace(keys=keys), g))(levels.keys)
    ctx = jnp.asarray(levels.ctx, x0.dtype)
    mu = jnp.asarray(levels.hidden_mu, x0.dtype)

    def obs_at_start(tr, x, c, m):
        st = start_state(x, c, m)
        return racing.observe(st, tr, racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu))

    obs0 = jax.jit(jax.vmap(obs_at_start))(tracks, x0, ctx, mu)
    n = x0.shape[0]
    sources = jnp.tile(jnp.arange(n, dtype=jnp.int32), (n_sources, 1))
    return PoolBank(tracks, x0, ctx, mu, obs0, sources, jnp.full((n_sources,), n, jnp.int32))


def with_source(bank: PoolBank, s: int, level_ids) -> PoolBank:
    """Replace table ``s`` by ``level_ids`` (e.g. SFL's learnable set D after a refresh)."""
    ids = jnp.asarray(level_ids, jnp.int32)
    row = jnp.zeros(bank.size, jnp.int32).at[: ids.shape[0]].set(ids)
    return bank._replace(sources=bank.sources.at[s].set(row),
                         source_len=bank.source_len.at[s].set(ids.shape[0]))


def level_pytree(levels: paper_levels.Levels) -> dict:
    """Per-level buffer entries, stacked [L]: ``{keys, ctx, hidden_mu, level_id}``."""
    n = levels.ctx.shape[0]
    return {"keys": jnp.asarray(levels.keys, jnp.uint32), "ctx": jnp.asarray(levels.ctx, jnp.float32),
            "hidden_mu": jnp.asarray(levels.hidden_mu, jnp.float32),
            "level_id": jnp.arange(n, dtype=jnp.int32)}


def reset_to(level: jax.Array, bank: PoolBank) -> tuple[jax.Array, racing.EnvState]:
    return bank.obs0[level], start_state(bank.x0[level], bank.ctx[level], bank.hidden_mu[level])


def pool_env(params: racing.EnvParams = racing.EnvParams()) -> Env:
    """``Env`` over a ``PoolBank``. ``reset(key, bank)`` is a uniform draw from the pool (so
    the env also works with ``make_train``); curriculum trainers set the slots' state with
    ``init_slots`` instead."""

    def reset(key, bank):
        lvl = jax.random.randint(key, (), 0, bank.size)
        obs, st = reset_to(lvl, bank)
        return obs, PoolState(st, lvl, lvl, jnp.int32(0), jnp.bool_(True), jnp.bool_(True))

    def step(key, state, action, bank):
        tr = jax.tree.map(lambda a: a[state.level], bank.tracks)
        obs, st, r, term, trunc, info = racing.step_no_reset(state.env, action, tr, params)
        done = term | trunc
        src = jnp.maximum(state.source, 0)
        drawn = bank.sources[src, jax.random.randint(key, (), 0, bank.source_len[src])]
        new_level = jnp.where(state.source < 0, state.next_level, drawn)
        obs_r, st_r = reset_to(new_level, bank)
        env_out = jax.tree.map(lambda a, b: jnp.where(done, a, b), st_r, st)
        out = PoolState(env_out, jnp.where(done, new_level, state.level), state.next_level,
                        state.source, jnp.where(done, state.next_mask, state.mask), state.next_mask)
        # Open strips also truncate near their end; pool windows must never get there.
        eos = trunc & (st.t < params.window_steps)
        info = dict(info, final_obs=obs, end_of_strip=eos)
        return jnp.where(done, obs_r, obs), out, r, term, trunc, info

    return Env(reset, step, racing.OBS_DIM, racing.ACTION_DIM, False)


def init_slots(bank: PoolBank, level, next_level, source, mask, next_mask):
    """Batched start state: slot j plays ``level[j]`` from its start."""
    obs, st = jax.vmap(lambda lv: reset_to(lv, bank))(jnp.asarray(level, jnp.int32))
    i32 = lambda a: jnp.asarray(a, jnp.int32)
    return obs, PoolState(st, i32(level), i32(next_level), i32(source),
                          jnp.asarray(mask, bool), jnp.asarray(next_mask, bool))


def trace(state: PoolState, info: dict) -> dict:
    """Per-step record for the scorer (``make_rollout``'s ``trace_fn``)."""
    return {"level": state.level, "progress": info["progress"], "end_of_strip": info["end_of_strip"]}


def loss_mask(state: PoolState) -> jax.Array:
    """``make_rollout``'s ``mask_fn``: train on the transitions of masked-in episodes."""
    return state.mask
