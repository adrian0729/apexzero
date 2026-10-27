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

An optional value-gradient term in the critic's loss (``ValueGradient``, VG-0; see the
section of that name below) is off unless a ``ValueGradient`` is passed to ``make_train``,
``make_rollout`` and ``make_learner``; without one they trace exactly the computation they
always did.
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


@dataclass(frozen=True)
class ValueGradient:
    """The value-gradient term in the critic's loss (VG-0, follow-through plan 7.3).

    GDHP-style (Werbos; Prokhorov & Wunsch 1997): besides its value regression the critic
    fits its gradient with respect to the environment's continuous state y, dV/dy, to
    VGL(lambda) targets G' taken through the simulator (Fairbank & Alonso 2012):

        G'_t = D_y [ r_t / sigma_t + gamma ( lam G'_{t+1} + (1 - lam) dV/dy(y_{t+1}) ) ]

    with D_y the total derivative of the recorded transition, the action's dependence on y
    through the policy mean at the recorded noise; 0 after a termination, dV/dy at the
    pre-reset state after a truncation and at the rollout's end (``vgl_scan``). The term is

        coef * mean_rows mean_k ((dV/dy_k(y_t) - G'_{t,k}) / scale_k)^2

    over the rows with finite targets (and inside the loss mask), after each component's
    targets are winsorised at their ``winsor`` and 1 - ``winsor`` quantiles over the rollout,
    with scale_k^2 = Var(G'_k) + E[(dV/dy_k)^2] over the rollout (the rollout critic's own
    gradient; ``vg_standardise``): per-component normalised squared errors, Omega diagonal
    and equal over the components after standardisation. Its derivative in the critic's
    parameters is a double backprop.

    ``coef`` 0 builds every capture, target and metric but leaves the loss and the update
    exactly the plain critic's (tests/test_value_gradient.py). The settings are not
    ``PPOConfig`` fields: the frozen scripts/h4r.py compares ``asdict(PPOConfig(...))`` with
    the H4R runs' config.json, so any new field would change that harness's verdicts.
    """

    coef: float = 0.0       # beta
    lam: float = 0.95       # VGL(lambda)
    winsor: float = 0.005   # per-component two-sided winsorisation quantile of the targets (0: off)

    def __post_init__(self):
        if not self.coef >= 0.0:
            raise ValueError(f"ValueGradient.coef must be >= 0, got {self.coef}")
        if not 0.0 <= self.lam <= 1.0:
            raise ValueError(f"ValueGradient.lam must be in [0, 1], got {self.lam}")
        if not 0.0 <= self.winsor < 0.5:
            raise ValueError(f"ValueGradient.winsor must be in [0, 0.5), got {self.winsor}")


class VGModel(NamedTuple):
    """An environment's differentiable side, for the value-gradient term. Per env
    (unbatched), with ``state`` the env's own per-env state and ``params`` its parameter carry:

    - ``get(state) -> y`` [dim]: the continuous Markov state the gradient is taken in;
    - ``obs(y, state, params) -> ob``: the raw (un-normalised) observation at y, the other
      per-env quantities (track, car, timers) read from ``state``; at y = get(state) it is the
      observation the env returned for ``state``;
    - ``step(y, action, state, params) -> (y_next, reward)``: the env's transition without a
      reset, the reward unscaled; at y = get(state) it is the env's own step, and it is
      differentiable in y and the action almost everywhere.
    """

    get: Callable[[Any], jax.Array]
    obs: Callable[[jax.Array, Any, Any], jax.Array]
    step: Callable[[jax.Array, jax.Array, Any, Any], tuple[jax.Array, jax.Array]]
    dim: int
    names: tuple[str, ...] = ()


# --------------------------------------------------------------------------- envs


class Env(NamedTuple):
    """Pure-function environment. ``step`` auto-resets and returns
    (obs, state, reward, terminated, truncated, info) with ``info['final_obs']``.

    ``reset(key, params)`` and ``step(key, state, action, params)`` take an environment
    parameter carry (for example a bank of tracks). ``init_params(key)`` creates it and
    ``refresh(key, params, update)``, if given, updates it between PPO updates (for
    example regenerating part of the track bank).

    ``vg_model``, if given, is the environment's differentiable side (``VGModel``), which
    only the value-gradient term reads; its ``step`` must also put the pre-reset state in
    ``info['final_state']``.
    """

    reset: Callable[[jax.Array, Any], tuple[jax.Array, Any]]
    step: Callable[[jax.Array, Any, jax.Array, Any], tuple]
    obs_dim: int
    action_dim: int
    discrete: bool
    init_params: Callable[[jax.Array], Any] | None = None
    refresh: Callable[[jax.Array, Any, jax.Array], Any] | None = None
    vg_model: VGModel | None = None


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
    vg: VGCapture | None = None   # the value-gradient term's per-step capture, else None


class VGCapture(NamedTuple):
    """What the value-gradient term records per step ([T, N] leading axes)."""

    state: Any                # the env state before the step
    final_state: Any          # the env state after the step, before any reset (info['final_state'])
    reward_scale: jax.Array   # the divisor of the step's reward: sqrt(ret_rms.var + 1e-8), or 1


def _check_vg(env: Env, vg: ValueGradient | None) -> None:
    if vg is None:
        return
    if env.discrete:
        raise ValueError("the value-gradient term needs continuous actions")
    if env.vg_model is None:
        raise ValueError("the value-gradient term needs an Env with a vg_model")


def make_rollout(cfg: PPOConfig, env: Env, actor: Actor, critic: Critic,
                 mask_fn: Callable[[Any], jax.Array] | None = None,
                 trace_fn: Callable[[Any, dict], Any] | None = None,
                 vg: ValueGradient | None = None):
    """``rollout(agent, env_state, obs, ret_acc, ep_ret, ep_len, key, env_params)`` runs
    ``cfg.num_steps`` steps with frozen parameters and returns (new carry, ``Rollout``).

    Hooks, both optional and evaluated on the per-env state *before* each step:
    - ``mask_fn(env_state) -> [N] bool`` marks the transitions that will be trained on
      (PLR-perp trains only on replayed levels). The return normaliser is then updated
      from those envs only, and the mask is returned for ``learn``.
    - ``trace_fn(env_state, info) -> pytree`` records extra per-step data (level ids,
      progress) next to the trajectory.
    With ``vg`` the rollout also records ``Rollout.vg`` (``VGCapture``) for the
    value-gradient term; it changes nothing else.
    Without hooks the computation is exactly the one ``make_train`` always ran.
    """
    _check_vg(env, vg)

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
            if vg is not None:
                if "final_state" not in info:
                    raise ValueError("the value-gradient term needs info['final_state'] "
                                     "(the pre-reset state) from env.step")
                scale = (jnp.sqrt(ret_rms.var + 1e-8) if cfg.normalize_reward
                         else jnp.ones((), reward.dtype))
                cap = VGCapture(prev_state, info["final_state"],
                                jnp.broadcast_to(scale, reward.shape).astype(reward.dtype))
                return ((agent, env_state, nobs, ret_acc, ep_ret, ep_len, key),
                        (tr, stats, obs, mask, trace, cap))
            return (agent, env_state, nobs, ret_acc, ep_ret, ep_len, key), (tr, stats, obs, mask, trace)

        carry, ys = jax.lax.scan(
            env_step, (agent, env_state, obs, ret_acc, ep_ret, ep_len, key), None, cfg.num_steps)
        if vg is not None:
            traj, stats, raw_obs, mask, trace, cap = ys
        else:
            (traj, stats, raw_obs, mask, trace), cap = ys, None
        agent, obs = carry[0], carry[2]
        if cfg.ensemble_heads:
            last_value, last_heads = critic.apply(agent.critic.params, norm_obs(agent, obs), with_heads=True)
        else:
            last_value = critic.apply(agent.critic.params, norm_obs(agent, obs))
            last_heads = jnp.zeros((cfg.num_envs, 0))
        return carry, Rollout(traj, stats, raw_obs, last_value, last_heads, mask, trace, cap)

    return rollout


def _masked_mean(x, w, n):
    """Mean of ``x`` over the rows where ``w`` [B] is set (and all trailing axes); ``n`` is
    max(sum(w), 1). ``where`` rather than a product, so masked-out rows contribute exactly 0."""
    wb = w.reshape(w.shape + (1,) * (x.ndim - 1))
    return jnp.where(wb, x, 0.0).sum() / (n * (x.size // w.size))


# --------------------------------------------------------------------------- value gradient
#
# The pieces of the value-gradient term (``ValueGradient``). All take one env's quantities
# (unbatched) unless stated; the learner vmaps them. y is the env's continuous state
# (``VGModel.get``), dV/dy is in the critic's units (reward / sigma_R) per unit of y.


def _norm_fn(cfg: PPOConfig):
    if cfg.normalize_obs:
        return lambda rms, ob: rms.normalize(ob)
    return lambda rms, ob: ob


def state_value_fn(cfg: PPOConfig, critic: Critic, model: VGModel):
    """``value(critic_params, obs_rms, y, state, env_params)``: the critic's main value at the
    normalised (clipped) observation of y, as the rollout evaluates it (without obs noise)."""
    norm = _norm_fn(cfg)

    def value(critic_params, obs_rms, y, state, env_params):
        return critic.apply(critic_params, norm(obs_rms, model.obs(y, state, env_params)))

    return value


def state_gradient_fn(cfg: PPOConfig, critic: Critic, model: VGModel):
    """dV/dy, the gradient of ``state_value_fn`` in y (same arguments)."""
    return jax.grad(state_value_fn(cfg, critic, model), argnums=2)


def transition_fn(cfg: PPOConfig, actor: Actor, model: VGModel):
    """``h(y, action, state, actor_params, obs_rms, env_params) -> (reward, y_next)`` of one
    recorded transition, with the action's dependence on y through the policy mean at the
    recorded noise: a(y) = action + (mu(o(y)) - stop_gradient(mu(o(y)))), so a equals the
    recorded action exactly and da/dy = dmu/dy (the reparametrised stochastic policy). The
    mean is taken at the noise-free observation, also when ``cfg.obs_noise`` is set."""
    norm = _norm_fn(cfg)

    def h(y, action, state, actor_params, obs_rms, env_params):
        mean, _ = actor.apply(actor_params, norm(obs_rms, model.obs(y, state, env_params)))
        a = action + (mean - jax.lax.stop_gradient(mean))
        y1, r = model.step(y, a, state, env_params)
        return r, y1

    return h


def vgl_scan(jr, jy, g_next, reward_scale, terminated, truncated, gamma: float, lam: float):
    """VGL(lambda) targets [T, N, Y] by a reverse scan over the rollout.

    jr [T, N, Y] is D_y r_t (unscaled), jy [T, N, Y, Y] is D_y y_{t+1} (rows: y_{t+1}'s
    components), g_next [T, N, Y] is dV/dy at the state after step t before any reset, and
    reward_scale [T, N] the step's reward divisor. With c_t the bootstrap of step t,

        c_t = 0 after a termination; g_next_t after a truncation and at t = T - 1;
              lam G'_{t+1} + (1 - lam) g_next_t otherwise,
        G'_t = jr_t / reward_scale_t + gamma jy_t' c_t.
    """

    def body(g_after, t):
        jr_t, jy_t, gn_t, rs_t, te_t, tr_t = t
        mixed = lam * g_after + (1.0 - lam) * gn_t
        c = jnp.where(te_t[:, None], 0.0, jnp.where(tr_t[:, None], gn_t, mixed))
        g = jr_t / rs_t[:, None] + gamma * jnp.einsum("nij,ni->nj", jy_t, c)
        return g, g

    _, G = jax.lax.scan(body, g_next[-1], (jr, jy, g_next, reward_scale, terminated, truncated),
                        reverse=True)
    return G


def vg_standardise(G, valid, winsor: float, g_critic):
    """(winsorised G, scale [Y]). Each component of the targets G [..., Y] is clipped to its
    ``winsor`` and 1 - ``winsor`` quantiles over the ``valid`` rows (``winsor`` 0: not
    clipped). Its scale is sqrt(Var(G_k) + E[g_k^2]) over those rows, with g_critic [..., Y]
    the rollout critic's own dV/dy: the targets' spread plus the critic's gradient size, so
    a component whose targets barely vary (the accel channels, about 1e-5) cannot weigh 1e5
    times the others while the critic's gradient there is still far off (1 where undefined
    or 0)."""
    Y = G.shape[-1]
    flat = lambda g: jnp.where(valid[..., None], g, jnp.nan).reshape(-1, Y)
    if winsor > 0:
        lo = jnp.nanquantile(flat(G), winsor, axis=0)
        hi = jnp.nanquantile(flat(G), 1.0 - winsor, axis=0)
        G = jnp.clip(G, lo, hi)
    var = jnp.nanvar(flat(G), axis=0) + jnp.nanmean(flat(g_critic) ** 2, axis=0)
    ok = jnp.isfinite(var) & (var > 0)
    return G, jnp.sqrt(jnp.where(ok, var, 1.0)).astype(G.dtype)


class VGTargets(NamedTuple):
    y: jax.Array          # [T, N, Y] the state before each step
    target: jax.Array     # [T, N, Y] winsorised targets, 0 on invalid rows
    scale: jax.Array      # [Y] each component's standardisation scale
    valid: jax.Array      # [T, N] finite targets (and inside the loss mask)
    raw: jax.Array        # [T, N, Y] the targets before winsorisation


def vg_targets(cfg: PPOConfig, vg: ValueGradient, actor: Actor, critic: Critic, model: VGModel,
               agent: AgentState, traj: Transition, cap: VGCapture, env_params,
               mask: jax.Array | None = None) -> VGTargets:
    """The term's targets for one rollout, from the rollout's own networks and normalisers
    (as GAE's targets use the rollout's values). Jacobians one time step at a time (bounded
    memory): jacfwd of ``transition_fn`` and the critic's dV/dy at the pre-reset states."""
    h = transition_fn(cfg, actor, model)
    grad_y = state_gradient_fn(cfg, critic, model)
    ap, cp, rms = agent.actor.params, agent.critic.params, agent.obs_rms

    def per_step(args):
        st, fst, act = args
        y = jax.vmap(model.get)(st)
        y1 = jax.vmap(model.get)(fst)
        jr, jy = jax.vmap(lambda q, a, s: jax.jacfwd(h)(q, a, s, ap, rms, env_params))(y, act, st)
        g1 = jax.vmap(lambda q, s: grad_y(cp, rms, q, s, env_params))(y1, fst)
        return y, jr, jy, g1

    y, jr, jy, g1 = jax.lax.map(per_step, (cap.state, cap.final_state, traj.action))
    raw = vgl_scan(jr, jy, g1, cap.reward_scale, traj.terminated, traj.truncated, cfg.gamma, vg.lam)
    valid = jnp.all(jnp.isfinite(raw), -1)
    if mask is not None:
        valid = valid & mask.astype(bool)
    target, scale = vg_standardise(raw, valid, vg.winsor, g1)
    target = jnp.where(valid[..., None], target, 0.0)
    return VGTargets(y, target, scale, valid, raw)


def vg_loss_fn(cfg: PPOConfig, critic: Critic, model: VGModel):
    """``loss(critic_params, obs_rms, y [B, Y], state [B], target [B, Y], scale [Y], w [B],
    env_params)``: the mean over the rows in ``w`` of mean_k ((dV/dy_k - target_k) /
    scale_k)^2, the term before ``coef``. Its gradient in critic_params is a double backprop."""
    grad_y = state_gradient_fn(cfg, critic, model)

    def loss(critic_params, obs_rms, y, state, target, scale, w, env_params):
        g = jax.vmap(grad_y, in_axes=(None, None, 0, 0, None))(critic_params, obs_rms, y, state,
                                                               env_params)
        e = (g - target) / scale
        per = jnp.mean(e * e, -1)
        n = jnp.maximum(w.sum(), 1).astype(per.dtype)
        return jnp.where(w, per, 0.0).sum() / n

    return loss


def make_learner(cfg: PPOConfig, env: Env, actor: Actor, critic: Critic, num_updates: int,
                 vg: ValueGradient | None = None):
    """``learn(agent, traj, last_value, last_heads, key, upd, mask=None)``: GAE and the PPO
    epochs on one rollout; returns (agent, key, (advantages, targets, minibatch metrics)).

    With ``vg`` (``ValueGradient``) ``learn`` also takes ``vg_data`` (the rollout's
    ``VGCapture``) and ``env_params``; it computes the term's targets once per rollout, adds
    coef times the term to the critic's loss, and appends the term and the share of valid
    rows to the minibatch metrics. The value loss reported stays the plain one.

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

    _check_vg(env, vg)
    if vg is not None:
        vg_term = vg_loss_fn(cfg, critic, env.vg_model)

    def learn(agent, traj, last_value, last_heads, key, upd, mask=None, vg_data=None,
              env_params=None):
        adv, targets, head_targets = gae(traj, last_value, last_heads)
        data = (traj, adv, targets, head_targets) if mask is None else (
            traj, adv, targets, head_targets, mask)
        if vg is not None:
            if vg_data is None:
                raise ValueError("learn: the value-gradient term needs the rollout's VGCapture")
            vt = vg_targets(cfg, vg, actor, critic, env.vg_model, agent, traj, vg_data,
                            env_params, mask)
            valid_share = vt.valid.mean()
            data = data + (vt.y, vg_data.state, vt.target, vt.valid)

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
                if vg is not None:
                    mb, (yb, sb, gb, vb) = mb[:-4], mb[-4:]
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
                if vg is None:
                    vloss, gc = jax.value_and_grad(critic_loss)(agent.critic.params)
                else:
                    vw = vb if w is None else vb & w.astype(bool)

                    def critic_loss_vg(params):
                        base = critic_loss(params)
                        term = vg_term(params, agent.obs_rms, yb, sb, gb, vt.scale, vw, env_params)
                        return (base + vg.coef * term if vg.coef else base), (base, term)

                    (_, (vloss, vgl)), gc = jax.value_and_grad(critic_loss_vg, has_aux=True)(
                        agent.critic.params)
                new = agent._replace(actor=agent.actor.apply_gradients(grads=ga),
                                     critic=agent.critic.apply_gradients(grads=gc))
                if w is not None:
                    keep = w.any()
                    new = jax.tree.map(lambda x, y: jnp.where(keep, x, y), new, agent)
                if vg is not None:
                    return new, (vloss,) + aux + (vgl, valid_share)
                return new, (vloss,) + aux

            agent, m = jax.lax.scan(minibatch, agent, mbs)
            return (agent, key), m

        (agent, key), m = jax.lax.scan(epoch, (agent, key), None, cfg.update_epochs)
        return agent, key, (adv, targets, m)

    return learn


def make_train(cfg: PPOConfig, env: Env, log_fn: Callable[[dict], None] | None = None,
               checkpoint_fn: Callable[[Any, Any], None] | None = None,
               vg: ValueGradient | None = None):
    """``checkpoint_fn(snapshot, update)`` is called on the host after every update with the
    actor/critic params and normalisers; it decides itself which updates to keep.

    ``vg`` adds the value-gradient term (``ValueGradient``) and the metrics ``vg_loss`` (the
    term before its coefficient) and ``vg_valid`` (the share of rows with finite targets)."""
    actor = Actor(cfg, env.action_dim, env.discrete)
    critic = Critic(cfg)
    n_up = cfg.num_updates
    rollout = make_rollout(cfg, env, actor, critic, vg=vg)
    learn = make_learner(cfg, env, actor, critic, n_up, vg=vg)

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
            if vg is None:
                agent, key, (_, targets, m) = learn(agent, traj, ro.last_value, ro.last_heads, key, upd)
            else:
                agent, key, (_, targets, m) = learn(agent, traj, ro.last_value, ro.last_heads, key,
                                                    upd, vg_data=ro.vg, env_params=env_params)

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
            if vg is not None:
                metrics["vg_loss"] = m[5].mean()
                metrics["vg_valid"] = m[6].mean()
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
