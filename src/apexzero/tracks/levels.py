"""Levels for the paper: theta = (strip, car context, hidden grip, start state).

A level is stored compactly as the generator key of its strip plus the car numbers, and
regenerated deterministically on demand. Every paper level starts on the strip's opening
straight: s = 0, n = 0, xi = 0, v = 0.8 v_QSS(0), previous action 0.

Level sets come from disjoint, named seed streams (the `audit-v1` partition), so no
training draw can coincide with an audit or pool level:

    train    the DR training stream (E0), keys folded from stream id 0
    a_dr     1,000 levels from the training generator
    a_tail   300 levels from the tilted (hairpin-heavy) generator
    a_ood    100 levels from the shifted out-of-distribution generator
    pool     4,000 levels for the curriculum comparison (C)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.oracle.qss import speed_profile_jax
from apexzero.tracks.generator import Grammar, generate, out_of_distribution, strip_arrays, tilted
from apexzero.vehicle.params import F1_2025

STREAMS = {"train": 0, "a_dr": 1, "a_tail": 2, "a_ood": 3, "pool": 4}
SIZES = {"a_dr": 1000, "a_tail": 300, "a_ood": 100, "pool": 4000}
START_SPEED_FRACTION = 0.8


class Levels(NamedTuple):
    keys: jax.Array        # [L, 2] uint32 generator keys
    ctx: jax.Array         # [L, 6] car context
    hidden_mu: jax.Array   # [L] grip multiplier (not observed)
    generator: str         # "train" | "tilted" | "ood"


def stream_key(root: jax.Array, stream: str) -> jax.Array:
    return jax.random.fold_in(root, STREAMS[stream])


def make_levels(root: jax.Array, stream: str, n: int | None = None, randomize_car: bool = True,
                mu_jitter: float = 0.02) -> Levels:
    from apexzero.env.adapters import sample_context
    from apexzero.env.racing import default_context

    n = n or SIZES[stream]
    base = stream_key(root, stream)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(jnp.arange(n))
    kc = jax.vmap(lambda k: jax.random.fold_in(k, 1))(keys)
    km = jax.vmap(lambda k: jax.random.fold_in(k, 2))(keys)
    ctx = jax.vmap(sample_context)(kc) if randomize_car else jnp.tile(default_context(), (n, 1))
    mu = jax.vmap(lambda k: jax.random.uniform(k, (), minval=1 - mu_jitter, maxval=1 + mu_jitter))(km)
    gen = {"a_tail": "tilted", "a_ood": "ood"}.get(stream, "train")
    return Levels(jax.vmap(lambda k: jax.random.fold_in(k, 0))(keys), ctx, mu, gen)


def grammar_for(g: Grammar, generator: str) -> Grammar:
    return {"train": g, "tilted": tilted(g), "ood": out_of_distribution(g)}[generator]


def build(levels: Levels, g: Grammar):
    """(TrackArrays stacked [L, ...], v_ref [L, P], x0 [L, 8]) for a level set."""
    gg = grammar_for(g, levels.generator)

    def one(key):
        kappa, wl, wr = generate(key, gg)
        tr = strip_arrays(kappa, wl, wr)
        v = speed_profile_jax(kappa, 2.0, F1_2025)
        x0 = jnp.array([0.0, 0.0, 0.0, START_SPEED_FRACTION * v[0], 0.0, 0.0, 0.0, 0.0])
        return tr, v, x0

    return jax.vmap(one)(levels.keys)


def save(levels: Levels, path) -> None:
    np.savez(path, keys=np.asarray(jax.random.key_data(levels.keys)) if jnp.issubdtype(levels.keys.dtype, jax.dtypes.prng_key) else np.asarray(levels.keys),
             ctx=np.asarray(levels.ctx), hidden_mu=np.asarray(levels.hidden_mu),
             generator=np.array(levels.generator))


def load(path) -> Levels:
    d = np.load(path)
    return Levels(jnp.asarray(d["keys"]), jnp.asarray(d["ctx"]), jnp.asarray(d["hidden_mu"]),
                  str(d["generator"]))
