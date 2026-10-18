"""Proximal Policy Optimisation, written from scratch in the PureJaxRL style.

The whole training loop (rollouts, GAE, minibatch epochs) is one jitted function, so a
vmap over seeds trains several independent agents in parallel.

Implementation details, following "The 37 Implementation Details of PPO":
- separate actor and critic networks; orthogonal init; small final-layer init for the
  policy mean; state-independent log-std;
- running observation normalisation (frozen inside a rollout, updated after it);
- reward scaling by the running std of the discounted return;
- GAE(lambda) with correct handling of truncation: a truncated step bootstraps from
  V(final_obs), the pre-reset observation, never from the next episode's first state;
- clipped surrogate, value loss with the same clipping, entropy bonus with a linear
  schedule, global gradient clipping, Adam with linear learning-rate decay;
- per-minibatch advantage normalisation.

Environments are given as an ``Env`` of pure functions; ``from_gymnax`` adapts gymnax.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 2048
    num_steps: int = 64
    total_timesteps: int = 50_000_000
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_value: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    ent_coef_final: float = 0.0
    lr: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    normalize_obs: bool = True
    normalize_reward: bool = True
    log_std_init: float = -0.5
    network: str = "simba"          # "simba" or "mlp"
    actor_width: int = 256
    critic_width: int = 384
    mlp_layers: int = 3
    activation: str = "tanh"        # for "mlp"
    obs_noise: float = 0.0          # Gaussian noise on (normalised) observations in training
    ensemble_heads: int = 0         # extra value heads on a stop-gradient trunk (disagreement score)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size

    def with_(self, **kw) -> PPOConfig:
        return replace(self, **kw)


# --------------------------------------------------------------------------- envs


class Env(NamedTuple):
    """Pure-function environment. ``step`` auto-resets and returns
    (obs, state, reward, terminated, truncated, info) with ``info['final_obs']``.

    ``reset(key, params)`` and ``step(key, state, action, params)`` take an environment
    parameter carry (for example a bank of tracks). ``init_params(key)`` creates it and
    ``refresh(key, params, update)``, if given, updates it between PPO updates (for
    example regenerating part of the track bank).
    """

    reset: Callable[[jax.Array, Any], tuple[jax.Array, Any]]
    step: Callable[[jax.Array, Any, jax.Array, Any], tuple]
    obs_dim: int
    action_dim: int
    discrete: bool
    init_params: Callable[[jax.Array], Any] | None = None
    refresh: Callable[[jax.Array, Any, jax.Array], Any] | None = None


def from_gymnax(name: str, truncation_as_termination: bool = True, **kw) -> Env:
    """Adapt a gymnax environment (both the < 1.0 and >= 1.0 step signatures).

    With ``truncation_as_termination`` (the PureJaxRL / rejax convention) time-limit ends
    are treated as terminal, so results are comparable to those references. Set it to
    False to bootstrap through time limits from ``final_observation``."""
    import gymnax

    env, params = gymnax.make(name, **kw)
    space = env.action_space(params)
    discrete = hasattr(space, "n")
    action_dim = space.n if discrete else int(np.prod(space.shape))
    obs_dim = int(np.prod(env.observation_space(params).shape))

    def reset(key, _p=None):
        return env.reset(key, params)

    def step(key, state, action, _p=None):
        out = env.step(key, state, action, params)
        if len(out) == 5:  # gymnax < 1.0: (obs, state, reward, done, info), no truncation flag
            obs, state, reward, done, info = out
            term, trunc = done.astype(bool), jnp.zeros_like(done, bool)
            info = {"final_observation": obs}
        else:
            obs, state, reward, term, trunc, info = out
            term, trunc = term.astype(bool), trunc.astype(bool)
        if truncation_as_termination:
            term, trunc = term | trunc, jnp.zeros_like(trunc)
        return obs, state, reward, term, trunc, {"final_obs": info["final_observation"]}

    return Env(reset, step, obs_dim, action_dim, discrete)


# --------------------------------------------------------------------------- networks


SQRT2 = math.sqrt(2.0)


def _dense(features, scale=SQRT2):
    return nn.Dense(features, kernel_init=nn.initializers.orthogonal(scale),
                    bias_init=nn.initializers.zeros)


class MLPEncoder(nn.Module):
    width: int
    layers: int
    activation: str

    @nn.compact
    def __call__(self, x):
        act = nn.tanh if self.activation == "tanh" else nn.relu
        for _ in range(self.layers):
            x = act(_dense(self.width)(x))
        return x


class SimBaEncoder(nn.Module):
    """Dense -> 2 x [LayerNorm -> Dense(2w) -> ReLU -> Dense(w) + skip] -> LayerNorm."""

    width: int
    blocks: int = 2

    @nn.compact
    def __call__(self, x):
        x = _dense(self.width, 1.0)(x)
        for _ in range(self.blocks):
            h = nn.LayerNorm()(x)
            h = nn.relu(_dense(2 * self.width)(h))
            h = _dense(self.width, 1.0)(h)
            x = x + h
        return nn.LayerNorm()(x)


def _encoder(cfg: PPOConfig, width: int) -> nn.Module:
    if cfg.network == "simba":
        return SimBaEncoder(width)
    return MLPEncoder(width, cfg.mlp_layers, cfg.activation)


class Actor(nn.Module):
    cfg: PPOConfig
    action_dim: int
    discrete: bool

    @nn.compact
    def __call__(self, obs):
        h = _encoder(self.cfg, self.cfg.actor_width)(obs)
        out = _dense(self.action_dim, 0.01)(h)
        if self.discrete:
            return out, None
        log_std = self.param("log_std", nn.initializers.constant(self.cfg.log_std_init),
                             (self.action_dim,))
        return out, log_std


class Critic(nn.Module):
    """Value network. With ``ensemble_heads`` = E > 0 it also owns E small heads that read
    the trunk through a stop-gradient: trained on the same targets, they cannot change the
    main critic, and their spread is the ensemble-disagreement level score."""

    cfg: PPOConfig

    @nn.compact
    def __call__(self, obs, with_heads: bool = False):
        h = _encoder(self.cfg, self.cfg.critic_width)(obs)
        v = _dense(1, 1.0)(h).squeeze(-1)
        if self.cfg.ensemble_heads and (with_heads or self.is_initializing()):
            hs = jax.lax.stop_gradient(h)
            heads = jnp.stack([
                _dense(1, 1.0)(nn.relu(_dense(64)(hs))).squeeze(-1)
                for _ in range(self.cfg.ensemble_heads)
            ], axis=-1)
            return (v, heads) if with_heads else v
        return v


# --------------------------------------------------------------------------- distributions


def gaussian_logp(a, mean, log_std):
    z = (a - mean) / jnp.exp(log_std)
    return jnp.sum(-0.5 * z * z - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)


def gaussian_entropy(log_std, batch_shape):
    ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    return jnp.broadcast_to(ent, batch_shape)


def categorical_logp(a, logits):
    logp = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(logp, a[..., None].astype(jnp.int32), axis=-1).squeeze(-1)


def categorical_entropy(logits):
    logp = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(logp) * logp, axis=-1)


# --------------------------------------------------------------------------- normalisers


class RunningMeanStd(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, shape=()):
        return cls(jnp.zeros(shape), jnp.ones(shape), jnp.asarray(1e-4))

    def update(self, x):
        """Parallel (Chan et al.) merge of a batch along axis 0."""
        b_mean, b_var, b_n = jnp.mean(x, 0), jnp.var(x, 0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        return RunningMeanStd(mean, m2 / tot, tot)

    def normalize(self, x, clip=10.0):
        return jnp.clip((x - self.mean) / jnp.sqrt(self.var + 1e-8), -clip, clip)


# --------------------------------------------------------------------------- agent state


class AgentState(NamedTuple):
    actor: TrainState
    critic: TrainState
    obs_rms: RunningMeanStd
    ret_rms: RunningMeanStd


class Transition(NamedTuple):
    obs: jax.Array          # normalised
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array       # scaled
    terminated: jax.Array
    truncated: jax.Array
    final_value: jax.Array  # V(final_obs), used when truncated
    raw_reward: jax.Array


def policy_apply(actor_params, actor: Actor, obs, discrete):
    out, log_std = actor.apply(actor_params, obs)
    return out, log_std


def act(agent: AgentState, actor: Actor, obs_raw, key, discrete, deterministic=False,
        normalize=True):
    obs = agent.obs_rms.normalize(obs_raw) if normalize else obs_raw
    out, log_std = actor.apply(agent.actor.params, obs)
    if discrete:
        if deterministic:
            return jnp.argmax(out, -1)
        return jax.random.categorical(key, out)
    if deterministic:
        return out
    return out + jnp.exp(log_std) * jax.random.normal(key, out.shape)


# --------------------------------------------------------------------------- training


def make_train(cfg: PPOConfig, env: Env, log_fn: Callable[[dict], None] | None = None,
               checkpoint_fn: Callable[[Any, Any], None] | None = None):
    """``checkpoint_fn(snapshot, update)`` is called on the host after every update with the
    actor/critic params and normalisers; it decides itself which updates to keep."""
    actor = Actor(cfg, env.action_dim, env.discrete)
    critic = Critic(cfg)
    n_up = cfg.num_updates
    steps_per_epoch = cfg.num_minibatches * cfg.update_epochs

    def lr_schedule(count):
        frac = 1.0 - (count // steps_per_epoch) / n_up
        return cfg.lr * (frac if cfg.anneal_lr else 1.0)

    def make_tx():
        return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                           optax.adam(learning_rate=lr_schedule, eps=1e-5))

    def init(key):
        ka, kc, ke, kp = jax.random.split(key, 4)
        env_params = env.init_params(kp) if env.init_params is not None else None
        dummy = jnp.zeros((1, env.obs_dim))
        a_params = actor.init(ka, dummy)
        c_params = critic.init(kc, dummy)
        agent = AgentState(
            actor=TrainState.create(apply_fn=actor.apply, params=a_params, tx=make_tx()),
            critic=TrainState.create(apply_fn=critic.apply, params=c_params, tx=make_tx()),
            obs_rms=RunningMeanStd.create((env.obs_dim,)),
            ret_rms=RunningMeanStd.create(()),
        )
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(ke, cfg.num_envs), env_params)
        if cfg.normalize_obs:
            agent = agent._replace(obs_rms=agent.obs_rms.update(obs))
        return agent, env_state, obs, env_params

    def norm_obs(agent, obs):
        return agent.obs_rms.normalize(obs) if cfg.normalize_obs else obs

    def train(key):
        key, k0 = jax.random.split(key)
        agent, env_state, obs, env_params = init(k0)
        ret_acc = jnp.zeros(cfg.num_envs)
        ep_ret = jnp.zeros(cfg.num_envs)
        ep_len = jnp.zeros(cfg.num_envs)

        def update(carry, upd):
            agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params = carry
            if env.refresh is not None:
                key, kr = jax.random.split(key)
                env_params = env.refresh(kr, env_params, upd)

            def env_step(c, _):
                agent, env_state, obs, ret_acc, ep_ret, ep_len, key = c
                key, k_act, k_noise, k_env = jax.random.split(key, 4)
                o = norm_obs(agent, obs)
                o_in = o + cfg.obs_noise * jax.random.normal(k_noise, o.shape) if cfg.obs_noise else o
                out, log_std = actor.apply(agent.actor.params, o_in)
                if env.discrete:
                    action = jax.random.categorical(k_act, out)
                    logp = categorical_logp(action, out)
                else:
                    action = out + jnp.exp(log_std) * jax.random.normal(k_act, out.shape)
                    logp = gaussian_logp(action, out, log_std)
                value = critic.apply(agent.critic.params, o_in)

                nobs, env_state, reward, term, trunc, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None))(
                    jax.random.split(k_env, cfg.num_envs), env_state, action, env_params)
                final_value = critic.apply(agent.critic.params, norm_obs(agent, info["final_obs"]))
                done = term | trunc

                ret_acc = ret_acc * cfg.gamma + reward
                ret_rms = agent.ret_rms
                if cfg.normalize_reward:
                    ret_rms = agent.ret_rms.update(ret_acc)
                    scaled = reward / jnp.sqrt(ret_rms.var + 1e-8)
                else:
                    scaled = reward
                agent = agent._replace(ret_rms=ret_rms)
                ret_acc = jnp.where(done, 0.0, ret_acc)

                ep_ret = ep_ret + reward
                ep_len = ep_len + 1
                stats = (jnp.where(done, ep_ret, 0.0), jnp.where(done, ep_len, 0.0), done)
                ep_ret = jnp.where(done, 0.0, ep_ret)
                ep_len = jnp.where(done, 0.0, ep_len)

                tr = Transition(o_in, action, logp, value, scaled, term, trunc, final_value, reward)
                return (agent, env_state, nobs, ret_acc, ep_ret, ep_len, key), (tr, stats, obs)

            (agent, env_state, obs, ret_acc, ep_ret, ep_len, key), (traj, stats, raw_obs) = jax.lax.scan(
                env_step, (agent, env_state, obs, ret_acc, ep_ret, ep_len, key), None, cfg.num_steps)

            last_value = critic.apply(agent.critic.params, norm_obs(agent, obs))

            # GAE with truncation bootstrapping.
            def gae_step(c, t):
                adv_next, v_next = c
                done = t.terminated | t.truncated
                next_v = jnp.where(done, jnp.where(t.terminated, 0.0, t.final_value), v_next)
                delta = t.reward + cfg.gamma * next_v - t.value
                adv = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * adv_next
                return (adv, t.value), adv

            _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_value), last_value), traj,
                                  reverse=True)
            targets = adv + traj.value

            # PPO epochs over shuffled minibatches.
            def epoch(c, _):
                agent, key = c
                key, kp = jax.random.split(key)
                flat = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]),
                                    (traj, adv, targets))
                perm = jax.random.permutation(kp, cfg.batch_size)
                mbs = jax.tree.map(
                    lambda x: jnp.take(x, perm, axis=0).reshape(
                        (cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]), flat)

                def minibatch(agent, mb):
                    t, a, tgt = mb
                    frac = upd / n_up
                    ent_coef = cfg.ent_coef + frac * (cfg.ent_coef_final - cfg.ent_coef)

                    def actor_loss(params):
                        out, log_std = actor.apply(params, t.obs)
                        if env.discrete:
                            logp = categorical_logp(t.action, out)
                            ent = categorical_entropy(out)
                        else:
                            logp = gaussian_logp(t.action, out, log_std)
                            ent = gaussian_entropy(log_std, logp.shape)
                        ratio = jnp.exp(logp - t.logp)
                        an = (a - a.mean()) / (a.std() + 1e-8)
                        pg = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * an)
                        loss = pg.mean() - ent_coef * ent.mean()
                        approx_kl = ((ratio - 1) - (logp - t.logp)).mean()
                        clipfrac = (jnp.abs(ratio - 1) > cfg.clip_eps).mean()
                        return loss, (pg.mean(), ent.mean(), approx_kl, clipfrac)

                    def critic_loss(params):
                        if cfg.ensemble_heads:
                            v, heads = critic.apply(params, t.obs, with_heads=True)
                            head_loss = ((heads - tgt[:, None]) ** 2).mean()
                        else:
                            v, head_loss = critic.apply(params, t.obs), 0.0
                        vl = (v - tgt) ** 2
                        if cfg.clip_value:
                            vc = t.value + jnp.clip(v - t.value, -cfg.clip_eps, cfg.clip_eps)
                            vl = jnp.maximum(vl, (vc - tgt) ** 2)
                        return cfg.vf_coef * 0.5 * (vl.mean() + head_loss)

                    (_, aux), ga = jax.value_and_grad(actor_loss, has_aux=True)(agent.actor.params)
                    vloss, gc = jax.value_and_grad(critic_loss)(agent.critic.params)
                    agent = agent._replace(actor=agent.actor.apply_gradients(grads=ga),
                                           critic=agent.critic.apply_gradients(grads=gc))
                    return agent, (vloss,) + aux

                agent, m = jax.lax.scan(minibatch, agent, mbs)
                return (agent, key), m

            (agent, key), m = jax.lax.scan(epoch, (agent, key), None, cfg.update_epochs)

            if cfg.normalize_obs:
                agent = agent._replace(obs_rms=agent.obs_rms.update(raw_obs.reshape(-1, env.obs_dim)))

            ret_sum, len_sum, dones = stats
            n_done = jnp.maximum(dones.sum(), 1)
            var_t = jnp.var(targets)
            metrics = {
                "update": upd,
                "timesteps": (upd + 1) * cfg.batch_size,
                "episode_return": ret_sum.sum() / n_done,
                "episode_length": len_sum.sum() / n_done,
                "episodes": dones.sum(),
                "reward_mean": traj.raw_reward.mean(),
                "terminated_frac": traj.terminated.mean(),
                "value_loss": m[0].mean(),
                "policy_loss": m[1].mean(),
                "entropy": m[2].mean(),
                "approx_kl": m[3].mean(),
                "clip_frac": m[4].mean(),
                "explained_var": 1.0 - jnp.var(targets - traj.value) / (var_t + 1e-8),
            }
            if log_fn is not None:
                jax.debug.callback(log_fn, metrics)
            if checkpoint_fn is not None:
                snap = {"actor": agent.actor.params, "critic": agent.critic.params,
                        "obs_rms": agent.obs_rms, "ret_rms": agent.ret_rms}
                jax.debug.callback(checkpoint_fn, snap, upd)
            return (agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params), metrics

        carry = (agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params)
        carry, metrics = jax.lax.scan(update, carry, jnp.arange(n_up))
        return carry[0], metrics

    return train, actor, critic
