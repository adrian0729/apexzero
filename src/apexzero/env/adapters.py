"""Build PPO-ready ``Env``s from tracks: a single-track specialist or a bank of tracks."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Env
from apexzero.env import racing
from apexzero.oracle.qss import GG, speed_profile
from apexzero.tracks.track import Track
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import TrackArrays, track_arrays


class TrackBank(NamedTuple):
    tracks: TrackArrays   # every field stacked along axis 0
    v_ref: jax.Array      # [T, P]


def reference_speed(track: Track, scale: float = 1.0) -> np.ndarray:
    return scale * speed_profile(track.kappa, track.ds, GG.from_params(F1_2025), closed=track.closed)


def make_bank(tracks: list[Track], pad_to: int | None = None) -> TrackBank:
    pad_to = pad_to or max(t.n_points for t in tracks)
    arrs = [track_arrays(t, pad_to=pad_to) for t in tracks]
    stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *arrs)
    vref = []
    for t in tracks:
        v = np.full(pad_to, 1.0)
        v[: t.n_points] = reference_speed(t)
        vref.append(v)
    return TrackBank(stacked, jnp.asarray(np.stack(vref), jnp.float32))


class BankState(NamedTuple):
    env: racing.EnvState
    track_id: jax.Array


def bank_env(bank: TrackBank, params: racing.EnvParams = racing.EnvParams(),
             ctx: jax.Array | None = None) -> Env:
    """Each reset draws a track uniformly from the bank (domain randomisation over tracks)."""
    n_tracks = bank.v_ref.shape[0]
    ctx = racing.default_context() if ctx is None else ctx

    def get(i):
        return jax.tree.map(lambda a: a[i], bank.tracks), bank.v_ref[i]

    def reset(key):
        k1, k2 = jax.random.split(key)
        tid = jax.random.randint(k1, (), 0, n_tracks)
        tr, v = get(tid)
        obs, st = racing.reset(k2, tr, v, ctx, params)
        return obs, BankState(st, tid)

    def step(key, state, action):
        tr, _ = get(state.track_id)
        obs, st, r, term, trunc, info = racing.step_no_reset(state.env, action, tr, params)
        done = term | trunc
        obs_r, state_r = reset(key)
        out_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_r, BankState(st, state.track_id))
        out_obs = jnp.where(done, obs_r, obs)
        return out_obs, out_state, r, term, trunc, dict(info, final_obs=obs)

    return Env(reset, step, racing.OBS_DIM, racing.ACTION_DIM, False)
