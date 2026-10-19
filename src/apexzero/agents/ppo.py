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

    def update_masked(self, x, w):
        """``update`` restricted to the rows of ``x`` where ``w`` [B] is set (PLR-perp trains
        only on replayed levels, so only those transitions may move the normalisers). An
        all-zero ``w`` returns the statistics bit for bit: the merge itself would not, since
        var * count / count need not round back to var."""
        w = w.astype(bool)
        wb = w.reshape(w.shape + (1,) * (x.ndim - 1))
        b_n = w.sum().astype(x.dtype)
        safe = jnp.maximum(b_n, 1.0)
        b_mean = jnp.where(wb, x, 0.0).sum(0) / safe
        b_var = jnp.where(wb, (x - b_mean) ** 2, 0.0).sum(0) / safe
        delta = b_mean - self.mean
        tot = self.count + b_n
        mean = self.mean + delta * b_n / tot
        m2 = self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot
        merged = RunningMeanStd(mean, m2 / tot, tot)
        return jax.tree.map(lambda new, old: jnp.where(b_n > 0, new, old), merged, self)

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
    heads: jax.Array        # [E] ensemble-head values (E may be 0)
    final_heads: jax.Array  # [E] heads at final_obs


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
#
# ``make_train`` is the whole loop in one jitted function. Its three stages are also exposed
# on their own (``make_rollout``, ``make_learner``, ``init_agent``) so a curriculum trainer
# (apexzero.ued) can interleave them with host-side level-buffer operations between chunks,
# using exactly the same networks, losses and normalisers.


def make_tx(cfg: PPOConfig, num_updates: int):
    """Adam with global-norm clipping and a linear learning-rate decay over ``num_updates``
    PPO updates. The schedule counts optimiser steps, so (as in JaxUED) updates that apply
    no gradient do not advance it."""
    steps_per_update = cfg.num_minibatches * cfg.update_epochs

    def lr_schedule(count):
        frac = 1.0 - (count // steps_per_update) / num_updates
        return cfg.lr * (frac if cfg.anneal_lr else 1.0)

    return optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                       optax.adam(learning_rate=lr_schedule, eps=1e-5))


def init_agent(cfg: PPOConfig, actor: Actor, critic: Critic, obs_dim: int, key_actor: jax.Array,
               key_critic: jax.Array, num_updates: int) -> AgentState:
    dummy = jnp.zeros((1, obs_dim))
    a_params = actor.init(key_actor, dummy)
    c_params = critic.init(key_critic, dummy)
    return AgentState(
        actor=TrainState.create(apply_fn=actor.apply, params=a_params, tx=make_tx(cfg, num_updates)),
        critic=TrainState.create(apply_fn=critic.apply, params=c_params, tx=make_tx(cfg, num_updates)),
        obs_rms=RunningMeanStd.create((obs_dim,)),
        ret_rms=RunningMeanStd.create(()),
    )


class Rollout(NamedTuple):
    traj: Transition          # [T, N] per field
    stats: tuple              # (episode return, episode length, done) per step, for logging
    raw_obs: jax.Array        # [T, N, obs_dim] un-normalised observations (normaliser update)
    last_value: jax.Array     # [N] V(o_T) with the rollout's parameters (GAE bootstrap)
    last_heads: jax.Array     # [N, E]
    mask: jax.Array | None    # [T, N] transitions to train on (``mask_fn``), else None
    trace: Any                # ``trace_fn`` output per step, else None


def make_rollout(cfg: PPOConfig, env: Env, actor: Actor, critic: Critic,
                 mask_fn: Callable[[Any], jax.Array] | None = None,
                 trace_fn: Callable[[Any, dict], Any] | None = None):
    """``rollout(agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params)`` runs
    ``cfg.num_steps`` steps with frozen parameters and returns (new carry, ``Rollout``).

    Hooks, both optional and evaluated on the per-env state *before* each step:
    - ``mask_fn(env_state) -> [N] bool`` marks the transitions that will be trained on
      (PLR-perp trains only on replayed levels). The return normaliser is then updated
      from those envs only, and the mask is returned for ``learn``.
    - ``trace_fn(env_state, info) -> pytree`` records extra per-step data (level ids,
      progress) next to the trajectory.
    Without hooks the computation is exactly the one ``make_train`` always ran.
    """

    def norm_obs(agent, obs):
        return agent.obs_rms.normalize(obs) if cfg.normalize_obs else obs

    def rollout(agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params):
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
            if cfg.ensemble_heads:
                value, heads = critic.apply(agent.critic.params, o_in, with_heads=True)
            else:
                value, heads = critic.apply(agent.critic.params, o_in), jnp.zeros((cfg.num_envs, 0))

            prev_state = env_state
            mask = mask_fn(prev_state) if mask_fn is not None else None
            nobs, env_state, reward, term, trunc, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_env, cfg.num_envs), env_state, action, env_params)
            if cfg.ensemble_heads:
                final_value, final_heads = critic.apply(
                    agent.critic.params, norm_obs(agent, info["final_obs"]), with_heads=True)
            else:
                final_value = critic.apply(agent.critic.params, norm_obs(agent, info["final_obs"]))
                final_heads = jnp.zeros((cfg.num_envs, 0))
            done = term | trunc

            ret_acc = ret_acc * cfg.gamma + reward
            ret_rms = agent.ret_rms
            if cfg.normalize_reward:
                ret_rms = (agent.ret_rms.update(ret_acc) if mask is None
                           else agent.ret_rms.update_masked(ret_acc, mask))
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

            tr = Transition(o_in, action, logp, value, scaled, term, trunc, final_value, reward,
                            heads, final_heads)
            trace = trace_fn(prev_state, info) if trace_fn is not None else None
            return (agent, env_state, nobs, ret_acc, ep_ret, ep_len, key), (tr, stats, obs, mask, trace)

        carry, (traj, stats, raw_obs, mask, trace) = jax.lax.scan(
            env_step, (agent, env_state, obs, ret_acc, ep_ret, ep_len, key), None, cfg.num_steps)
        agent, obs = carry[0], carry[2]
        if cfg.ensemble_heads:
            last_value, last_heads = critic.apply(agent.critic.params, norm_obs(agent, obs), with_heads=True)
        else:
            last_value = critic.apply(agent.critic.params, norm_obs(agent, obs))
            last_heads = jnp.zeros((cfg.num_envs, 0))
        return carry, Rollout(traj, stats, raw_obs, last_value, last_heads, mask, trace)

    return rollout


def _masked_mean(x, w, n):
    """Mean of ``x`` over the rows where ``w`` [B] is set (and all trailing axes); ``n`` is
    max(sum(w), 1). ``where`` rather than a product, so masked-out rows contribute exactly 0."""
    wb = w.reshape(w.shape + (1,) * (x.ndim - 1))
    return jnp.where(wb, x, 0.0).sum() / (n * (x.size // w.size))


def make_learner(cfg: PPOConfig, env: Env, actor: Actor, critic: Critic, num_updates: int):
    """``learn(agent, traj, last_value, last_heads, key, upd, mask=None)``: GAE and the PPO
    epochs on one rollout; returns (agent, key, (advantages, targets, minibatch metrics)).

    With ``mask`` [T, N] (spec DEV-16, the per-transition form of PLR-perp's "no gradient
    on exploratory levels"): every loss term (policy, entropy, value, ensemble heads) is a
    mean over masked-in transitions only; minibatch advantage normalisation uses only them;
    and a minibatch without any leaves the networks and optimiser state untouched. GAE is
    computed over all transitions; the mask is constant within an episode and the lambda-sum
    is cut at every episode end, so masked-out episodes cannot leak into masked-in targets.
    """

    def gae(traj, last_value, last_heads):
        # GAE with truncation bootstrapping. Ensemble heads get their own GAE targets
        # from their own values (as in minimax), over the same rewards and episode ends.
        def gae_step(c, t):
            adv_next, v_next, hadv_next, h_next = c
            done = t.terminated | t.truncated
            next_v = jnp.where(done, jnp.where(t.terminated, 0.0, t.final_value), v_next)
            delta = t.reward + cfg.gamma * next_v - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * adv_next
            d, tm = done[:, None], t.terminated[:, None]
            next_h = jnp.where(d, jnp.where(tm, 0.0, t.final_heads), h_next)
            hdelta = t.reward[:, None] + cfg.gamma * next_h - t.heads
            hadv = hdelta + cfg.gamma * cfg.gae_lambda * (1.0 - d) * hadv_next
            return (adv, t.value, hadv, t.heads), (adv, hadv)

        init = (jnp.zeros_like(last_value), last_value, jnp.zeros_like(last_heads), last_heads)
        _, (adv, head_adv) = jax.lax.scan(gae_step, init, traj, reverse=True)
        return adv, adv + traj.value, head_adv + traj.heads

    def learn(agent, traj, last_value, last_heads, key, upd, mask=None):
        adv, targets, head_targets = gae(traj, last_value, last_heads)
        data = (traj, adv, targets, head_targets) if mask is None else (
            traj, adv, targets, head_targets, mask)

        # PPO epochs over shuffled minibatches.
        def epoch(c, _):
            agent, key = c
            key, kp = jax.random.split(key)
            flat = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), data)
            perm = jax.random.permutation(kp, cfg.batch_size)
            mbs = jax.tree.map(
                lambda x: jnp.take(x, perm, axis=0).reshape(
                    (cfg.num_minibatches, cfg.minibatch_size) + x.shape[1:]), flat)

            def minibatch(agent, mb):
                if mask is None:
                    (t, a, tgt, htgt), w = mb, None
                else:
                    t, a, tgt, htgt, w = mb
                    n = jnp.maximum(w.sum(), 1).astype(a.dtype)
                frac = upd / num_updates
                ent_coef = cfg.ent_coef + frac * (cfg.ent_coef_final - cfg.ent_coef)
                mean = (lambda x: x.mean()) if w is None else (lambda x: _masked_mean(x, w, n))

                def actor_loss(params):
                    out, log_std = actor.apply(params, t.obs)
                    if env.discrete:
                        logp = categorical_logp(t.action, out)
                        ent = categorical_entropy(out)
                    else:
                        logp = gaussian_logp(t.action, out, log_std)
                        ent = gaussian_entropy(log_std, logp.shape)
                    ratio = jnp.exp(logp - t.logp)
                    if w is None:
                        an = (a - a.mean()) / (a.std() + 1e-8)
                    else:
                        a_mean = _masked_mean(a, w, n)
                        a_std = jnp.sqrt(_masked_mean((a - a_mean) ** 2, w, n))
                        an = (a - a_mean) / (a_std + 1e-8)
                    pg = -jnp.minimum(ratio * an, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * an)
                    loss = mean(pg) - ent_coef * mean(ent)
                    approx_kl = mean((ratio - 1) - (logp - t.logp))
                    clipped = jnp.abs(ratio - 1) > cfg.clip_eps
                    clipfrac = clipped.mean() if w is None else mean(clipped.astype(a.dtype))
                    return loss, (mean(pg), mean(ent), approx_kl, clipfrac)

                def critic_loss(params):
                    if cfg.ensemble_heads:
                        v, heads = critic.apply(params, t.obs, with_heads=True)
                        head_loss = mean((heads - htgt) ** 2)
                    else:
                        v, head_loss = critic.apply(params, t.obs), 0.0
                    vl = (v - tgt) ** 2
                    if cfg.clip_value:
                        vc = t.value + jnp.clip(v - t.value, -cfg.clip_eps, cfg.clip_eps)
                        vl = jnp.maximum(vl, (vc - tgt) ** 2)
                    return cfg.vf_coef * 0.5 * (mean(vl) + head_loss)

                (_, aux), ga = jax.value_and_grad(actor_loss, has_aux=True)(agent.actor.params)
                vloss, gc = jax.value_and_grad(critic_loss)(agent.critic.params)
                new = agent._replace(actor=agent.actor.apply_gradients(grads=ga),
                                     critic=agent.critic.apply_gradients(grads=gc))
                if w is not None:
                    keep = w.any()
                    new = jax.tree.map(lambda x, y: jnp.where(keep, x, y), new, agent)
                return new, (vloss,) + aux

            agent, m = jax.lax.scan(minibatch, agent, mbs)
            return (agent, key), m

        (agent, key), m = jax.lax.scan(epoch, (agent, key), None, cfg.update_epochs)
        return agent, key, (adv, targets, m)

    return learn


def make_train(cfg: PPOConfig, env: Env, log_fn: Callable[[dict], None] | None = None,
               checkpoint_fn: Callable[[Any, Any], None] | None = None):
    """``checkpoint_fn(snapshot, update)`` is called on the host after every update with the
    actor/critic params and normalisers; it decides itself which updates to keep."""
    actor = Actor(cfg, env.action_dim, env.discrete)
    critic = Critic(cfg)
    n_up = cfg.num_updates
    rollout = make_rollout(cfg, env, actor, critic)
    learn = make_learner(cfg, env, actor, critic, n_up)

    def init(key):
        ka, kc, ke, kp = jax.random.split(key, 4)
        env_params = env.init_params(kp) if env.init_params is not None else None
        agent = init_agent(cfg, actor, critic, env.obs_dim, ka, kc, n_up)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            jax.random.split(ke, cfg.num_envs), env_params)
        if cfg.normalize_obs:
            agent = agent._replace(obs_rms=agent.obs_rms.update(obs))
        return agent, env_state, obs, env_params

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

            (agent, env_state, obs, ret_acc, ep_ret, ep_len, key), ro = rollout(
                agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params)
            traj = ro.traj
            agent, key, (_, targets, m) = learn(agent, traj, ro.last_value, ro.last_heads, key, upd)

            if cfg.normalize_obs:
                agent = agent._replace(obs_rms=agent.obs_rms.update(ro.raw_obs.reshape(-1, env.obs_dim)))

            ret_sum, len_sum, dones = ro.stats
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
