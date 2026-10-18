"""Audit rollouts: many stochastic windows per level from a frozen checkpoint.

For each (level, window) pair the policy drives one 1,200-step window from the level's
straight start x0, with stochastic actions and 1% observation noise (windows differ only
in that randomness). Per-step rewards, critic values, ensemble-head values and an alive
mask are kept (values in *raw reward units*: the critic's output times sigma, the return
normaliser's std, spec DEV-6), so every level score (any GAE lambda, discounted or not, any success
definition) can be computed afterwards from the same rollouts, and targets can be
cross-fitted on disjoint windows.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import Actor, Critic, PPOConfig, RunningMeanStd, gaussian_logp
from apexzero.env import racing
from apexzero.vehicle.sim import TrackArrays


class Snapshot(NamedTuple):
    actor: dict
    critic: dict
    obs_rms: RunningMeanStd
    ret_rms: RunningMeanStd   # return normaliser: critic outputs are in units of reward / sigma

    @property
    def sigma(self):
        return jnp.sqrt(self.ret_rms.var + 1e-8)


class WindowRollouts(NamedTuple):
    reward: np.ndarray       # [L, W, T]
    value: np.ndarray        # [L, W, T]   V(o_t) of the main critic
    heads: np.ndarray        # [L, W, T, E] ensemble heads (E = 0 if none)
    alive: np.ndarray        # [L, W, T]   1 while the step happened
    terminated: np.ndarray   # [L, W]      crashed / left the corridor / spun / stalled
    final_value: np.ndarray  # [L, W]      V(o_T) when truncated (bootstrap)
    distance: np.ndarray     # [L, W]      s_T - s_0
    logp: np.ndarray         # [L, W, T]   behaviour log-probabilities

    def returns(self, gamma: float) -> np.ndarray:
        T = self.reward.shape[-1]
        disc = gamma ** np.arange(T)
        return np.sum(self.reward * self.alive * disc, axis=-1)


def snapshot_from_pickle(d: dict) -> Snapshot:
    def rms(x):
        return RunningMeanStd(**{k: jnp.asarray(v) for k, v in x.items()}) if isinstance(x, dict) else x

    return Snapshot(d["actor"], d["critic"], rms(d["obs_rms"]), rms(d["ret_rms"]))


def rollout_windows(snap: Snapshot, cfg: PPOConfig, tracks: TrackArrays, x0: jax.Array,
                    ctx: jax.Array, hidden_mu: jax.Array, windows: int, key: jax.Array,
                    steps: int = 1200, obs_noise: float = 0.01, level_chunk: int = 64,
                    params: racing.EnvParams = racing.EnvParams()) -> WindowRollouts:
    """``tracks``, ``x0`` [L, 8], ``ctx`` [L, 6], ``hidden_mu`` [L] describe L levels."""
    actor = Actor(cfg, racing.ACTION_DIM, False)
    critic = Critic(cfg)
    E = cfg.ensemble_heads
    params = params._replace(window_steps=steps)

    def one_window(tr, x0_l, ctx_l, mu_l, k):
        st = racing.EnvState(x=x0_l, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx_l,
                             hidden_mu=mu_l, t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()),
                             s0=x0_l[0])
        obs = racing.observe(st, tr, racing.params_from_context(ctx_l, hidden_mu=mu_l))

        def body(c, k_t):
            st, ob, alive = c
            k1, k2 = jax.random.split(k_t)
            o = snap.obs_rms.normalize(ob)
            o = o + obs_noise * jax.random.normal(k1, o.shape)
            mean, log_std = actor.apply(snap.actor, o)
            a = mean + jnp.exp(log_std) * jax.random.normal(k2, mean.shape)
            lp = gaussian_logp(a, mean, log_std)
            if E:
                v, hs = critic.apply(snap.critic, o, with_heads=True)
            else:
                v, hs = critic.apply(snap.critic, o), jnp.zeros((0,))
            ob2, st2, r, term, trunc, _ = racing.step_no_reset(st, a, tr, params)
            alive2 = alive & ~term & ~trunc
            st2 = jax.tree.map(lambda n, old: jnp.where(alive, n, old), st2, st)
            ob2 = jnp.where(alive, ob2, ob)
            return (st2, ob2, alive2), (r * alive, v, hs, alive, term & alive, lp)

        (stT, obT, _), (r, v, hs, al, term, lp) = jax.lax.scan(
            body, (st, obs, jnp.array(True)), jax.random.split(k, steps))
        oT = snap.obs_rms.normalize(obT)
        vT = critic.apply(snap.critic, oT)
        sig = snap.sigma
        v, hs, vT = v * sig, hs * sig, vT * sig
        crashed = term.any()
        return (r, v, hs, al.astype(jnp.float32), crashed, jnp.where(crashed, 0.0, vT),
                stT.x[0] - x0_l[0], lp)

    per_level = jax.vmap(one_window, in_axes=(None, None, None, None, 0))
    batched = jax.jit(jax.vmap(per_level, in_axes=(0, 0, 0, 0, 0)))

    L = x0.shape[0]
    outs = []
    for start in range(0, L, level_chunk):
        sl = slice(start, min(start + level_chunk, L))
        n = sl.stop - sl.start
        keys = jax.random.split(jax.random.fold_in(key, start), n * windows).reshape(n, windows, -1)
        tr = jax.tree.map(lambda a, sl=sl: a[sl], tracks)
        out = batched(tr, x0[sl], ctx[sl], hidden_mu[sl], keys)
        outs.append(jax.tree.map(np.asarray, out))
    cat = [np.concatenate([o[i] for o in outs]) for i in range(8)]
    r, v, hs, al, crashed, vT, dist, lp = cat
    if not E:
        hs = np.zeros(r.shape + (0,), np.float32)
    return WindowRollouts(r, v, hs, al, crashed, vT, dist, lp)
