"""PPO-ready ``Env``s: a fixed bank of tracks, or procedural domain randomisation (E0).

Both keep the tracks in the environment *parameter carry* (not in per-env state), so a
batch of thousands of envs shares one device-resident bank and each env only stores an
index into it.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Env
from apexzero.env import racing
from apexzero.oracle.qss import GG, speed_profile, speed_profile_jax
from apexzero.tracks.generator import Grammar, generate
from apexzero.tracks.generator import strip_arrays as _strip_arrays
from apexzero.tracks.track import Track
from apexzero.vehicle.params import CONTEXT_RANGES, F1_2025
from apexzero.vehicle.sim import TrackArrays, track_arrays

_CTX_LO = np.array([CONTEXT_RANGES[k][0] for k in CONTEXT_RANGES])
_CTX_HI = np.array([CONTEXT_RANGES[k][1] for k in CONTEXT_RANGES])


class TrackBank(NamedTuple):
    tracks: TrackArrays   # every field stacked along axis 0
    v_ref: jax.Array      # [B, P] QSS reference speed (for start speeds)


class BankState(NamedTuple):
    env: racing.EnvState
    track_id: jax.Array


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


def _get(bank: TrackBank, i):
    return jax.tree.map(lambda a: a[i], bank.tracks), bank.v_ref[i]


def _bank_step(reset_fn, params):
    def step(key, state, action, bank):
        tr, _ = _get(bank, state.track_id)
        obs, st, r, term, trunc, info = racing.step_no_reset(state.env, action, tr, params)
        done = term | trunc
        obs_r, state_r = reset_fn(key, bank)
        out_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), state_r, BankState(st, state.track_id))
        out_obs = jnp.where(done, obs_r, obs)
        return out_obs, out_state, r, term, trunc, dict(info, final_obs=obs)

    return step


def bank_env(bank: TrackBank, params: racing.EnvParams = racing.EnvParams(),
             ctx: jax.Array | None = None) -> Env:
    """Fixed tracks and a fixed car: each reset draws a track uniformly from the bank."""
    n_tracks = bank.v_ref.shape[0]
    ctx = racing.default_context() if ctx is None else ctx

    def reset(key, bank):
        k1, k2 = jax.random.split(key)
        tid = jax.random.randint(k1, (), 0, n_tracks)
        tr, v = _get(bank, tid)
        obs, st = racing.reset(k2, tr, v, ctx, params)
        return obs, BankState(st, tid)

    return Env(reset, _bank_step(reset, params), racing.OBS_DIM, racing.ACTION_DIM, False,
               init_params=lambda key: bank)


def generate_bank(key: jax.Array, g: Grammar, size: int, g_alt: Grammar | None = None,
                  alt_fraction: float = 0.0) -> TrackBank:
    """``size`` fresh strips with their QSS reference speeds, entirely on-device.

    With ``g_alt``, each strip comes from ``g_alt`` with probability ``alt_fraction``
    (a per-strip mixture; the grammars share array shapes, so the choice is a select).
    """

    def one(k):
        if g_alt is not None:
            k, kc = jax.random.split(k)
            use_alt = jax.random.uniform(kc) < alt_fraction
            gg = jax.tree.map(lambda a, b: jnp.where(use_alt, b, a), g, g_alt)
        else:
            gg = g
        kappa, wl, wr = generate(k, gg)
        tr = _strip_arrays(kappa, wl, wr)
        return tr, speed_profile_jax(kappa, 2.0, F1_2025)

    tracks, v = jax.vmap(one)(jax.random.split(key, size))
    return TrackBank(tracks, v)


def sample_context(key: jax.Array) -> jax.Array:
    """Car context uniform within the plan's training ranges."""
    u = jax.random.uniform(key, (6,))
    return jnp.asarray(_CTX_LO) + u * jnp.asarray(_CTX_HI - _CTX_LO)


def procedural_env(g: Grammar, bank_size: int = 4096, refresh_per_update: int = 256,
                   params: racing.EnvParams = racing.EnvParams(), randomize_car: bool = True,
                   mu_jitter: float = 0.02, g_alt: Grammar | None = None,
                   alt_fraction: float = 0.0) -> Env:
    """Domain randomisation over generated strips (E0).

    The bank holds ``bank_size`` strips; before every PPO update ``refresh_per_update`` of
    them (a rotating slice) are regenerated, so over a long run the agent sees millions of
    distinct tracks without ever generating one inside the environment step. Each reset
    draws a strip, a car context (if ``randomize_car``) and a hidden grip multiplier
    U(1 - mu_jitter, 1 + mu_jitter) that the policy does not observe. ``g_alt`` /
    ``alt_fraction`` mix in a second grammar (e.g. the tilted, hairpin-heavy one), which
    the first DR pilot showed is needed: it crashed in corners tighter than ~90% of the
    plain generator's.
    """
    default_ctx = racing.default_context()

    def reset(key, bank):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        tid = jax.random.randint(k1, (), 0, bank_size)
        tr, v = _get(bank, tid)
        ctx = sample_context(k2) if randomize_car else default_ctx
        mu = jax.random.uniform(k3, (), minval=1.0 - mu_jitter, maxval=1.0 + mu_jitter)
        obs, st = racing.reset(k4, tr, v, ctx, params, hidden_mu=mu)
        return obs, BankState(st, tid)

    def init_params(key):
        return generate_bank(key, g, bank_size, g_alt, alt_fraction)

    def refresh(key, bank, update):
        start = (update * refresh_per_update) % bank_size
        fresh = generate_bank(key, g, refresh_per_update, g_alt, alt_fraction)
        idx = (start + jnp.arange(refresh_per_update)) % bank_size
        return jax.tree.map(lambda old, new: old.at[idx].set(new), bank, fresh)

    return Env(reset, _bank_step(reset, params), racing.OBS_DIM, racing.ACTION_DIM, False,
               init_params=init_params, refresh=refresh)
