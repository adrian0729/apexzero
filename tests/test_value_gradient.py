"""The value-gradient term in the critic's loss (VG-0, follow-through plan 7.3).

Off by default, the training path is the one it always was: the racing bank env's default
update reproduces the values recorded before the term existed, and ``info['final_state']``
changes nothing. With the term on and beta = 0 the update is the plain one bit for bit, and
at beta > 0 only the critic moves. The model side reproduces the env's own step and
observation bit for bit; the state gradient, the one-step Jacobians, the VGL(lambda) targets
(lambda 0, 0.5 and 1, across a termination and a truncation) and the term's double-backprop
parameter gradient agree with finite differences in float64; and on a 1-D braking problem
with a fixed policy, the critic trained with the term has a state gradient nearer the exact
value gradient than the critic trained without it, on the same rollouts."""

import hashlib
import importlib.util
import json
import pickle
import platform
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import (
    Actor,
    Critic,
    Env,
    PPOConfig,
    RunningMeanStd,
    ValueGradient,
    VGModel,
    init_agent,
    make_learner,
    make_rollout,
    make_train,
    state_gradient_fn,
    state_value_fn,
    transition_fn,
    vg_loss_fn,
    vg_standardise,
    vg_targets,
)
from apexzero.env import racing
from apexzero.env.adapters import VG_NAMES, TrackBank, bank_env, make_bank
from apexzero.tracks.track import from_centreline

REPO = Path(__file__).resolve().parents[1]
TUMFTM = REPO / "data" / "tracks" / "tumftm"
needs_tracks = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")


@pytest.fixture
def float32():
    """Some test modules switch JAX to float64 when imported; training runs in float32."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture
def float64():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


def wavy_track(n=1500, mirror=False):
    """test_env.py's closed track with left and right corners and asymmetric widths."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 600 + 80 * np.sin(3 * t) + 30 * np.cos(5 * t)
    xy = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    wl = 6.0 + 1.0 * np.sin(2 * t)
    wr = 5.0 + 0.5 * np.cos(4 * t)
    tr = from_centreline("wavy", xy, wl, wr, ds=2.0)
    return tr.mirrored() if mirror else tr


def racing_env(window_steps=20):
    bank = make_bank([wavy_track(), wavy_track(mirror=True)])
    return bank_env(bank, racing.EnvParams(window_steps=window_steps)), bank


TINY = dict(num_envs=8, num_steps=16, total_timesteps=8 * 16 * 4, num_minibatches=4,
            update_epochs=2, network="mlp", actor_width=16, critic_width=16, mlp_layers=2,
            ensemble_heads=2)


def _leaves_equal(a, b) -> bool:
    return all(np.asarray(x).tobytes() == np.asarray(y).tobytes()
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def _abs_sum(p) -> float:
    return float(sum(np.abs(np.asarray(x, np.float64)).sum() for x in jax.tree.leaves(p)))


# ------------------------------------------------------------------ the default path is unchanged

# A tiny default-path run on the racing bank env, recorded at 3323a0e (before the term, the
# capture and info['final_state'] existed), float32, seed 0, on the M3 Pro's CPU with JAX
# 0.11.2. The tolerant values hold anywhere; the sha256 of every final parameter and
# normaliser byte is checked only on that platform and JAX version.
GOLDEN_RACING = dict(
    actor_abs=344.8066236387392, critic_abs=649.9713385994328, ret_var=15.910029411315918,
    value_loss=[2.8532683849334717, 2.1403110027313232, 1.6629852056503296, 1.2155201435089111],
    entropy=[1.8382138013839722, 1.838789701461792, 1.839264154434204, 1.8393899202346802],
    episode_return=[-11.99829387664795, -10.745559692382812, -9.210911750793457,
                    -7.941428184509277],
    terminated_frac=[0.0859375, 0.1171875, 0.125, 0.0703125],
)
GOLDEN_SHA = "3ab357585dd13592f880fe8a7a53b1095cf0fd01c598b17f3217eae74b0051e0"
GOLDEN_PLATFORM = ("0.11.2", "Darwin", "arm64")


def _run_tiny(vg=None, env=None):
    env = racing_env()[0] if env is None else env
    train, _, _ = make_train(PPOConfig(**TINY), env, vg=vg)
    return jax.jit(train)(jax.random.PRNGKey(0))


def _sha(agent) -> str:
    h = hashlib.sha256()
    for x in jax.tree.leaves((agent.actor.params, agent.critic.params, agent.obs_rms, agent.ret_rms)):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def test_default_racing_update_reproduces_the_pre_term_values(float32):
    agent, m = _run_tiny()
    got = dict(actor_abs=_abs_sum(agent.actor.params), critic_abs=_abs_sum(agent.critic.params),
               ret_var=float(agent.ret_rms.var),
               **{k: np.asarray(m[k]) for k in ("value_loss", "entropy", "episode_return",
                                                "terminated_frac")})
    for k, v in GOLDEN_RACING.items():
        assert np.allclose(got[k], v, rtol=1e-5, atol=1e-7), k
    assert "vg_loss" not in m and "vg_valid" not in m
    if (jax.__version__, platform.system(), platform.machine()) == GOLDEN_PLATFORM:
        assert _sha(agent) == GOLDEN_SHA                    # bit for bit on the recording platform


def test_default_rollout_records_no_capture_and_defaults_are_off(float32):
    env, bank = racing_env()
    cfg = PPOConfig(**TINY)
    actor, critic = Actor(cfg, 2, False), Critic(cfg)
    agent = init_agent(cfg, actor, critic, racing.OBS_DIM, jax.random.PRNGKey(0), jax.random.PRNGKey(1), 4)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(3), 8), bank)
    z = jnp.zeros(8)
    _, ro = make_rollout(cfg, env, actor, critic)(agent, st, obs, z, z, z, jax.random.PRNGKey(4), bank)
    assert ro.vg is None
    assert ValueGradient() == ValueGradient(coef=0.0, lam=0.95, winsor=0.005)
    assert "vg" not in {f for f in PPOConfig.__dataclass_fields__}     # not a PPOConfig field
    for bad in (dict(coef=-1.0), dict(lam=1.5), dict(winsor=0.5)):
        with pytest.raises(ValueError):
            ValueGradient(**bad)
    with pytest.raises(ValueError, match="vg_model"):
        make_rollout(cfg, env._replace(vg_model=None), actor, critic, vg=ValueGradient())
    with pytest.raises(ValueError, match="continuous"):
        make_learner(cfg, env._replace(discrete=True), actor, critic, 4, vg=ValueGradient())


# ------------------------------------------------------------------ beta = 0 and beta > 0


def test_beta_zero_reproduces_the_plain_training_bit_for_bit(float32):
    env, _ = racing_env()
    a0, m0 = _run_tiny(None, env)
    a1, m1 = _run_tiny(ValueGradient(coef=0.0), env)
    assert _leaves_equal((a0.actor.params, a0.critic.params, a0.obs_rms, a0.ret_rms),
                         (a1.actor.params, a1.critic.params, a1.obs_rms, a1.ret_rms))
    for k in m0:
        assert np.asarray(m0[k]).tobytes() == np.asarray(m1[k]).tobytes(), k
    assert np.all(np.isfinite(np.asarray(m1["vg_loss"])))
    assert np.all(np.asarray(m1["vg_valid"]) > 0.9)


def _one_rollout(env, bank, cfg, vg, seed=0, warm=1):
    actor, critic = Actor(cfg, 2, False), Critic(cfg)
    agent = init_agent(cfg, actor, critic, racing.OBS_DIM, jax.random.PRNGKey(seed),
                       jax.random.PRNGKey(seed + 1), 10)
    n = cfg.num_envs
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed + 3), n), bank)
    agent = agent._replace(obs_rms=agent.obs_rms.update(obs))
    z = jnp.zeros(n)
    ro_fn = jax.jit(make_rollout(cfg, env, actor, critic, vg=vg))
    carry = (agent, st, obs, z, z, z)
    for k in range(warm + 1):
        (agent, st, obs, ret, epr, epl, _), ro = ro_fn(*carry, jax.random.PRNGKey(seed + 10 + k), bank)
        if k < warm:
            agent = agent._replace(obs_rms=agent.obs_rms.update(ro.raw_obs.reshape(-1, racing.OBS_DIM)))
            carry = (agent, st, obs, ret, epr, epl)
    return actor, critic, agent, ro


def test_the_term_moves_only_the_critic(float32):
    env, bank = racing_env()
    cfg = PPOConfig(**{**TINY, "num_envs": 16})
    vg0, vg1 = ValueGradient(coef=0.0), ValueGradient(coef=1.0)
    actor, critic, agent, ro = _one_rollout(env, bank, cfg, vg0)
    plain = jax.jit(make_learner(cfg, env, actor, critic, 10))
    off = jax.jit(make_learner(cfg, env, actor, critic, 10, vg=vg0))
    on = jax.jit(make_learner(cfg, env, actor, critic, 10, vg=vg1))
    args = (agent, ro.traj, ro.last_value, ro.last_heads, jax.random.PRNGKey(7), 0)
    p, _, (_, _, mp) = plain(*args)
    a, _, (_, _, ma) = off(*args, vg_data=ro.vg, env_params=bank)
    b, _, (_, _, mb) = on(*args, vg_data=ro.vg, env_params=bank)
    assert _leaves_equal((p.actor, p.critic), (a.actor, a.critic))       # beta 0: the plain update
    for x, y in zip(mp, ma[:5], strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()          # same metrics
    assert len(ma) == 7 and np.all(np.isfinite(np.asarray(ma[5]))) and float(ma[6].mean()) > 0.9
    assert _leaves_equal(a.actor, b.actor)                 # the actor never sees the term
    assert not _leaves_equal(a.critic.params, b.critic.params)
    assert float(mb[5].mean()) < float(ma[5].mean())       # beta 1 lowers the term it optimises
    with pytest.raises(ValueError, match="VGCapture"):
        off(*args)


# ------------------------------------------------------------------ the model is the env


def test_model_reproduces_the_env_step_and_observation(float32):
    env, bank = racing_env(window_steps=12)
    cfg = PPOConfig(**{**TINY, "num_envs": 16, "num_steps": 32})
    actor, critic, agent, ro = _one_rollout(env, bank, cfg, ValueGradient())
    model, cap, tr = env.vg_model, ro.vg, ro.traj
    assert model.names == VG_NAMES and model.dim == 12
    done = np.asarray(tr.terminated | tr.truncated)
    assert np.asarray(tr.terminated).any() and np.asarray(tr.truncated).any() and (~done).any()
    get = jax.vmap(jax.vmap(model.get))
    y, y1 = get(cap.state), get(cap.final_state)
    step = jax.vmap(jax.vmap(lambda q, a, s: model.step(q, a, s, bank)))
    y_next, r = step(y, tr.action, cap.state)
    # The same functions as the env's step; XLA fuses them differently inside the rollout's
    # scan, so they agree to float32 rounding (the worst case is printed on failure).
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b)) /
                                    (np.abs(np.asarray(b)) + 1.0)))
    assert rel(y_next, y1) < 1e-5, rel(y_next, y1)
    assert rel(r, tr.raw_reward) < 1e-5, rel(r, tr.raw_reward)
    ob = jax.vmap(jax.vmap(lambda q, s: model.obs(q, s, bank)))(y, cap.state)
    assert rel(ob, ro.raw_obs) < 1e-5, rel(ob, ro.raw_obs)
    # Not done: the pre-reset state is the next step's state. Done: it is the old episode's.
    nxt = np.asarray(y[1:])
    assert np.array_equal(np.asarray(y1[:-1])[~done[:-1]], nxt[~done[:-1]])
    assert not np.array_equal(np.asarray(y1[:-1])[done[:-1]], nxt[done[:-1]])
    assert np.array_equal(np.asarray(cap.final_state.track_id), np.asarray(cap.state.track_id))
    # The reward divisor is the rollout's: scaled reward = raw / divisor.
    assert np.allclose(np.asarray(tr.reward), np.asarray(tr.raw_reward / cap.reward_scale), rtol=1e-6)


# ------------------------------------------------------------------ finite differences (float64)


def _bank64(bank: TrackBank) -> TrackBank:
    up = lambda a: a.astype(jnp.float64) if jnp.issubdtype(a.dtype, jnp.floating) else a
    return jax.tree.map(up, bank)


def _racing64(seed=0):
    bank = _bank64(make_bank([wavy_track(), wavy_track(mirror=True)]))
    env = bank_env(bank, racing.EnvParams())
    cfg = PPOConfig(num_envs=4, num_steps=8, network="mlp", actor_width=32, critic_width=32,
                    mlp_layers=2)
    actor, critic = Actor(cfg, 2, False), Critic(cfg)
    agent = init_agent(cfg, actor, critic, racing.OBS_DIM, jax.random.PRNGKey(seed),
                       jax.random.PRNGKey(seed + 1), 10)
    # A spread-out normaliser and a scaled actor, so every channel and the policy matter.
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(5), 64), bank)
    rms = RunningMeanStd(jnp.mean(obs, 0), jnp.full(racing.OBS_DIM, 0.5), jnp.asarray(100.0))
    agent = agent._replace(obs_rms=rms)              # no channel clipped at these states
    a = agent.actor.params
    a = jax.tree_util.tree_map_with_path(
        lambda p, x: x * 30.0 if "Dense_0" in jax.tree_util.keystr(p) and "Encoder" not in
        jax.tree_util.keystr(p) else x, a)
    agent = agent._replace(actor=agent.actor.replace(params=a))
    s0 = jax.tree.map(lambda x: x[0], st)
    env_state = s0.env._replace(x=jnp.array([101.3, 0.7, 0.04, 50.0, 0.3, 0.05, 0.01, 0.4]),
                                prev_action=jnp.array([0.1, 0.3]), accel=jnp.array([2.0, -3.0]))
    return env, bank, cfg, actor, critic, agent, s0._replace(env=env_state)


def _fd_grad(f, y, h):
    cols = []
    for k in range(y.shape[0]):
        e = jnp.zeros_like(y).at[k].set(h[k])
        cols.append((f(y + e) - f(y - e)) / (2 * h[k]))
    return jnp.stack(cols, -1)


# Central-difference steps per component (s, n, xi, vx, vy, r, delta, pedal, prev_action,
# accel): small enough that no preview point crosses a 2 m knot of the pose lookup.
STEP = np.array([1e-5, 1e-6, 1e-7, 1e-6, 1e-6, 1e-7, 1e-8, 1e-6, 1e-6, 1e-6, 1e-5, 1e-5])


def test_state_gradient_matches_finite_differences(float64):
    env, bank, cfg, actor, critic, agent, s0 = _racing64()
    model = env.vg_model
    y = model.get(s0)
    v = state_value_fn(cfg, critic, model)
    g = state_gradient_fn(cfg, critic, model)(agent.critic.params, agent.obs_rms, y, s0, bank)
    fd = _fd_grad(lambda q: v(agent.critic.params, agent.obs_rms, q, s0, bank), y, STEP)
    assert np.all(np.abs(np.asarray(g)) > 0)
    assert np.allclose(np.asarray(g), np.asarray(fd), rtol=1e-5, atol=1e-9)


def test_one_step_jacobians_match_finite_differences(float64):
    env, bank, cfg, actor, critic, agent, s0 = _racing64()
    model = env.vg_model
    h = transition_fn(cfg, actor, model)
    y, action = model.get(s0), jnp.array([0.2, 0.5])
    args = (action, s0, agent.actor.params, agent.obs_rms, bank)
    r0, y1 = h(y, *args)
    y_env, r_env = model.step(y, action, s0, bank)
    assert float(r0) == float(r_env) and np.array_equal(np.asarray(y1), np.asarray(y_env))
    jr, jy = jax.jacfwd(h)(y, *args)
    # h's forward value never sees the policy (mean - stop_gradient(mean) is 0), so the finite
    # differences are of the reparametrised transition itself: a(q) = a + mu(q) - mu(y).
    mu = lambda q: actor.apply(agent.actor.params, agent.obs_rms.normalize(model.obs(q, s0, bank)))[0]
    mu0 = mu(y)
    step = jax.jit(lambda q: model.step(q, action + (mu(q) - mu0), s0, bank))
    fd_r = _fd_grad(lambda q: step(q)[1], y, STEP)
    fd_y = _fd_grad(lambda q: step(q)[0], y, STEP)
    assert np.allclose(np.asarray(jr), np.asarray(fd_r), rtol=1e-4, atol=1e-8)   # FD rounding
    assert np.allclose(np.asarray(jy), np.asarray(fd_y), rtol=1e-4, atol=1e-8)
    # The policy path matters here: without it dr/dy and dy'/dy are measurably different.
    fd_plain = _fd_grad(jax.jit(lambda q: model.step(q, action, s0, bank)[1]), y, STEP)
    assert not np.allclose(np.asarray(jr), np.asarray(fd_plain), rtol=1e-2, atol=1e-6)


def test_double_backprop_gradient_matches_finite_differences(float64):
    env, bank, cfg, actor, critic, agent, s0 = _racing64()
    model = env.vg_model
    B = 6
    rng = np.random.default_rng(0)
    y0 = model.get(s0)
    ys = jnp.asarray(y0 + rng.normal(size=(B, 12)) * np.array([5, .3, .02, 3, .2, .02, .005, .1,
                                                                .1, .1, 2, 2]))
    states = jax.tree.map(lambda x: jnp.stack([x] * B), s0)
    target = jnp.asarray(rng.normal(size=(B, 12)) * 0.01)
    scale = jnp.asarray(rng.uniform(0.005, 0.05, 12))
    w = jnp.array([True, True, False, True, True, True])
    loss = vg_loss_fn(cfg, critic, model)
    f = lambda p: loss(p, agent.obs_rms, ys, states, target, scale, w, bank)
    p0 = agent.critic.params
    g = jax.grad(f)(p0)
    d = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape)), p0)
    eps = 1e-5
    shift = lambda s: jax.tree.map(lambda x, dx: x + s * dx, p0, d)
    fd = (f(shift(eps)) - f(shift(-eps))) / (2 * eps)
    ad = sum(float(jnp.vdot(a, b)) for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(d), strict=True))
    assert abs(ad - float(fd)) <= 1e-6 * max(1.0, abs(ad))
    # A masked-out row cannot move the loss.
    t2 = target.at[2].set(1e6)
    assert float(loss(p0, agent.obs_rms, ys, states, t2, scale, w, bank)) == float(f(p0))


# ------------------------------------------------------------------ a 1-D braking problem
#
# A point mass approaches a corner at S_C. Reward: progress / 10 up to the corner, less a
# quadratic penalty above a braking-curve speed limit, both per metre travelled, so the return
# is continuous in the state; the episode ends at the corner. Its env, with a VGModel, for the
# VGL tests below. (With the penalty per step instead, one more penalised step appears
# whenever the crossing moves a step later: the return jumps, the pathwise VGL targets miss
# the jumps, and no critic fits both its values and their gradients.)

S_C, V_MIN, A_MAX, DT = 150.0, 10.0, 12.0, 0.05
S0_MIN, S0_MAX = 90.0, 145.0      # starts: at most 3 s from the corner


class BState(NamedTuple):
    y: jax.Array      # (s, v)
    t: jax.Array


def v_limit(s):
    return jnp.sqrt(20.0 ** 2 + 2.0 * 15.0 * jnp.maximum(S_C - s, 0.0))


def b_core(y, a):
    s, v = y[0], y[1]
    v1 = jnp.maximum(v + jnp.clip(a[0], -1.0, 1.0) * A_MAX * DT, V_MIN)
    s1 = s + 0.5 * (v + v1) * DT
    over = jnp.maximum(v1 - v_limit(s1), 0.0)
    # Per metre travelled before the corner, so the return is continuous in the state: when the
    # crossing moves to the next step, that step's distance, and so its reward, tends to 0.
    r = (jnp.minimum(s1, S_C) - s) / 10.0 * (1.0 - 0.005 * over ** 2)
    return jnp.stack([s1, v1]), r


def b_obs(y):
    return jnp.stack([(y[0] - S_C) / 100.0, y[1] / 50.0])


def braking_env(window=400):
    def reset(key, _p=None):
        k1, k2 = jax.random.split(key)
        y = jnp.stack([jax.random.uniform(k1, (), minval=S0_MIN, maxval=S0_MAX),
                       jax.random.uniform(k2, (), minval=20.0, maxval=60.0)])
        return b_obs(y), BState(y, jnp.zeros((), jnp.int32))

    def step(key, state, action, _p=None):
        y1, r = b_core(state.y, action)
        t1 = state.t + 1
        term = y1[0] >= S_C
        trunc = (t1 >= window) & ~term
        final = BState(y1, t1)
        obs_r, st_r = reset(key)
        done = term | trunc
        out = jax.tree.map(lambda a, b: jnp.where(done, a, b), st_r, final)
        return (jnp.where(done, obs_r, b_obs(y1)), out, r, term, trunc,
                {"final_obs": b_obs(y1), "final_state": final})

    model = VGModel(lambda st: st.y, lambda y, st, p: b_obs(y),
                    lambda y, a, st, p: b_core(y, a), 2, ("s", "v"))
    return Env(reset, step, 2, 1, False, vg_model=model)


def _fixed_policy(cfg, actor, key, gain=40.0):
    """A random-init actor whose output layer is scaled up: a smooth, state-dependent policy."""
    p = actor.init(key, jnp.zeros((1, 2)))
    return jax.tree_util.tree_map_with_path(
        lambda path, x: x * gain if "Dense_0" in jax.tree_util.keystr(path) and "Encoder" not in
        jax.tree_util.keystr(path) else x, p)


def _lambda_return(cfg, actor, critic, agent, y_rec, actions, scales, term, trunc, lam):
    """R^lambda_0 as a function of the start state, re-simulated through the model with the
    recorded noise (a_k(y) = a_k + mu(y_k) - mu(y_k recorded)) to the episode's first end,
    terminations and truncations held at their recorded steps; the bootstrap value
    V(y_{k+1}) at the pre-reset state, and in full at the rollout's last step."""
    mu = lambda y: actor.apply(agent.actor.params, b_obs(y))[0]
    V = lambda y: critic.apply(agent.critic.params, b_obs(y))
    horizon = len(actions)

    def f(y_start):
        parts, y = [], y_start
        for k in range(horizon):
            a = actions[k] + (mu(y) - mu(y_rec[k]))
            y1, r = b_core(y, a)
            parts.append((r / scales[k], y1))
            y = y1
            if term[k] or trunc[k]:
                break
        R = None
        for k in reversed(range(len(parts))):
            r, y1 = parts[k]
            if term[k]:
                boot = 0.0
            elif trunc[k] or k == horizon - 1:
                boot = V(y1)
            else:
                boot = (1 - lam) * V(y1) + lam * R
            R = r + cfg.gamma * boot
        return R

    return f


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_vgl_targets_match_finite_differences_of_the_lambda_return(float64, lam):
    env = braking_env(window=9)
    cfg = PPOConfig(num_envs=32, num_steps=12, gamma=0.97, network="mlp", actor_width=16,
                    critic_width=16, mlp_layers=2, normalize_obs=False, normalize_reward=True,
                    log_std_init=-1.0)
    vg = ValueGradient(coef=1.0, lam=lam, winsor=0.0)
    actor, critic = Actor(cfg, 1, False), Critic(cfg)
    agent = init_agent(cfg, actor, critic, 2, jax.random.PRNGKey(0), jax.random.PRNGKey(1), 10)
    agent = agent._replace(actor=agent.actor.replace(params=_fixed_policy(cfg, actor, jax.random.PRNGKey(2))))
    # Starts near the corner, so some envs terminate inside the rollout and others truncate.
    st = BState(jnp.stack([jnp.linspace(120.0, 149.0, 32), jnp.linspace(20.0, 50.0, 32)], -1),
                jnp.zeros(32, jnp.int32))
    obs = jax.vmap(b_obs)(st.y)
    z = jnp.zeros(32)
    _, ro = make_rollout(cfg, env, actor, critic, vg=vg)(agent, st, obs, z, z, z,
                                                          jax.random.PRNGKey(4), None)
    vt = vg_targets(cfg, vg, actor, critic, env.vg_model, agent, ro.traj, ro.vg, None)
    tr = ro.traj
    te, tc = np.asarray(tr.terminated), np.asarray(tr.truncated)
    assert te.any() and tc.any()
    checked = 0
    for n in range(32):
        T = tr.reward.shape[0]
        f = _lambda_return(cfg, actor, critic, agent, ro.vg.state.y[:, n],
                           [tr.action[k, n] for k in range(T)],
                           [float(ro.vg.reward_scale[k, n]) for k in range(T)], te[:, n], tc[:, n],
                           lam)
        y0 = ro.vg.state.y[0, n]
        fd = _fd_grad(f, y0, np.array([1e-4, 1e-5]))
        assert np.allclose(np.asarray(vt.raw[0, n]), np.asarray(fd), rtol=1e-5, atol=1e-8), n
        checked += 1
    assert checked == 32


def test_standardise_winsorises_and_bounds_the_scale(float64):
    rng = np.random.default_rng(0)
    G = jnp.asarray(np.c_[rng.normal(0, 1, 1000), rng.normal(0, 1e-5, 1000)])
    G = G.at[0, 0].set(1e6)
    valid = jnp.ones(1000, bool).at[5].set(False)
    gc = jnp.asarray(np.c_[np.zeros(1000), np.full(1000, 1e-2)])
    Gw, scale = vg_standardise(G, valid, 0.005, gc)
    assert float(Gw[0, 0]) < 5.0                                    # the outlier is clipped
    assert abs(float(scale[0]) - float(np.std(np.asarray(Gw[valid, 0])))) < 1e-9
    assert abs(float(scale[1]) - 1e-2) < 1e-5                       # bounded by the critic's size
    _, s0 = vg_standardise(G, valid, 0.0, jnp.zeros_like(G))
    assert float(s0[0]) > 1e3                                       # unwinsorised: the outlier
    _, s1 = vg_standardise(jnp.zeros_like(G), valid, 0.005, jnp.zeros_like(G))
    assert np.array_equal(np.asarray(s1), np.ones(2))               # undefined: 1


def _exact_gradient(cfg, actor, actor_params, ys, steps=400):
    """dV^mu/dy of the deterministic policy by autodiff through its rollout to the corner."""
    def ret(y):
        def body(c, _):
            y, alive, disc, total = c
            a = actor.apply(actor_params, b_obs(y))[0]
            y1, r = b_core(y, a)
            total = total + jnp.where(alive, disc * r, 0.0)
            alive = alive & (y1[0] < S_C)
            return (y1, alive, disc * cfg.gamma, total), None
        (_, alive, _, total), _ = jax.lax.scan(body, (y, jnp.array(True), 1.0, 0.0), None, steps)
        return total
    return jax.vmap(jax.grad(ret))(ys)


def test_braking_critic_gradient_lands_nearer_the_exact_value_gradient(float32):
    """A fixed policy on the 1-D braking problem; the same rollouts train a plain critic and
    one with the term (beta 1, VGL(0.95)), from the same initialisation. The term's critic has
    the smaller gradient error in both components, and its value loss is no worse."""
    env = braking_env()
    cfg = PPOConfig(num_envs=64, num_steps=64, gamma=0.98, gae_lambda=0.95, network="mlp",
                    actor_width=32, critic_width=64, mlp_layers=2, normalize_obs=False,
                    normalize_reward=True, log_std_init=-6.0, ent_coef=0.0, ent_coef_final=0.0,
                    lr=1e-3, anneal_lr=False, clip_value=False, update_epochs=4, num_minibatches=4)
    actor, critic = Actor(cfg, 1, False), Critic(cfg)
    a_params = _fixed_policy(cfg, actor, jax.random.PRNGKey(2), gain=200.0)
    rng = np.random.default_rng(1)
    ys = jnp.asarray(np.c_[rng.uniform(S0_MIN, S0_MAX, 256), rng.uniform(20, 60, 256)], jnp.float32)
    exact = np.asarray(_exact_gradient(cfg, actor, a_params, ys))
    assert np.all(np.isfinite(exact)) and np.all(np.std(exact, 0) > 0)

    def train_critic(vg, n_updates=150):
        agent = init_agent(cfg, actor, critic, 2, jax.random.PRNGKey(0), jax.random.PRNGKey(1), n_updates)
        agent = agent._replace(actor=agent.actor.replace(params=a_params))
        actor_state = agent.actor
        rollout = jax.jit(make_rollout(cfg, env, actor, critic, vg=vg))
        learn = jax.jit(make_learner(cfg, env, actor, critic, n_updates, vg=vg))
        obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(3), 64), None)
        z = jnp.zeros(64)
        carry, key = (agent, st, obs, z, z, z), jax.random.PRNGKey(4)
        for upd in range(n_updates):
            key, kr, kl = jax.random.split(key, 3)
            (agent, st, obs, ret, epr, epl, _), ro = rollout(*carry, kr, None)
            kw = {} if vg is None else dict(vg_data=ro.vg, env_params=None)
            agent, _, (_, _, m) = learn(agent, ro.traj, ro.last_value, ro.last_heads, kl, upd, **kw)
            agent = agent._replace(actor=actor_state)          # the policy stays fixed
            carry = (agent, st, obs, ret, epr, epl)
        sigma = float(jnp.sqrt(agent.ret_rms.var + 1e-8))      # the critic's units: reward / sigma
        g = jax.vmap(jax.grad(lambda y: critic.apply(agent.critic.params, b_obs(y))))(ys)
        return sigma * np.asarray(g), float(m[0].mean())

    def rel_err(g):
        return np.sqrt(((g - exact) ** 2).mean(0)) / np.sqrt((exact ** 2).mean(0))

    (plain, v_plain), (with_term, v_term) = train_critic(None), train_critic(ValueGradient(coef=1.0))
    e_plain, e_term = rel_err(plain), rel_err(with_term)
    # Measured at the commit: plain (0.22, 0.11), with the term (0.12, 0.03); value loss
    # 3.1e-4 against 1.6e-5.
    assert np.all(e_term < 0.7 * e_plain), (e_term, e_plain)
    assert v_term <= v_plain, (v_term, v_plain)


# ------------------------------------------------------------------ train_dr.py


def _load_train_dr(monkeypatch):
    scripts = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("train_dr_vg", scripts / "train_dr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_train_dr(mod, tmp_path, monkeypatch, extra):
    seen = {}
    real_train = mod.make_train

    def spy_train(cfg, env, *a, **kw):
        seen["vg"] = kw.get("vg")
        return real_train(cfg, env, *a, **kw)

    class Laps:
        completion_rate, median_lap = 0.0, float("nan")

    monkeypatch.setattr(mod, "make_train", spy_train)
    monkeypatch.setattr(mod, "timed_laps", lambda *a, **k: Laps())
    monkeypatch.setattr(mod, "refine_reference", lambda tr: tr)
    if not (tmp_path / "data").exists():
        (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "smoke", "--steps", "128",
                                      "--num-envs", "8", "--num-steps", "8", "--bank", "8",
                                      "--refresh", "4", "--ensemble-heads", "2", *extra])
    mod.main()
    return seen


@needs_tracks
def test_train_dr_default_path_passes_no_term(tmp_path, monkeypatch, float32):
    mod = _load_train_dr(monkeypatch)
    seen = _run_train_dr(mod, tmp_path, monkeypatch, [])
    assert seen["vg"] is None
    out = tmp_path / "runs" / "smoke"
    cfg = json.loads((out / "config.json").read_text())
    assert "vg" not in cfg and set(cfg) == {"ppo", "args", "env"}
    assert cfg["args"]["vg_coef"] == 0.0
    assert "vg_loss" not in (out / "metrics.csv").read_text().splitlines()[0]
    with open(max((out / "ckpt").glob("*.pkl")), "rb") as fh:
        assert "vg" not in pickle.load(fh)


@needs_tracks
def test_train_dr_value_gradient_smoke(tmp_path, monkeypatch, float32):
    """Two PPO updates at 8 envs on CPU with the term on."""
    mod = _load_train_dr(monkeypatch)
    seen = _run_train_dr(mod, tmp_path, monkeypatch, ["--vg-coef", "0.3", "--vg-lambda", "0.9"])
    assert seen["vg"] == ValueGradient(coef=0.3, lam=0.9, winsor=0.005)
    out = tmp_path / "runs" / "smoke"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["vg"] == {"coef": 0.3, "lam": 0.9, "winsor": 0.005}
    head = (out / "metrics.csv").read_text().splitlines()
    assert "vg_loss" in head[0] and "vg_valid" in head[0] and len(head) >= 2
    with open(max((out / "ckpt").glob("*.pkl")), "rb") as fh:
        assert pickle.load(fh)["vg"] == {"coef": 0.3, "lam": 0.9, "winsor": 0.005}


@needs_tracks
@pytest.mark.parametrize("extra, msg", [
    (["--vg-lambda", "0.9"], "needs --vg-coef > 0"),
    (["--vg-winsor", "0.01"], "needs --vg-coef > 0"),
    (["--vg-coef", "-1"], "must be >= 0"),
])
def test_train_dr_refuses_bad_vg_flags(tmp_path, monkeypatch, extra, msg):
    mod = _load_train_dr(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "bad", *extra])
    with pytest.raises(SystemExit, match=msg):
        mod.main()


def _run_train_dr_rdev(mod, tmp_path, monkeypatch, name, extra):
    """One tiny train_dr.py run with timed_laps recorded (circuit, start speed, runs)."""
    from apexzero.env.adapters import reference_speed

    calls = []

    class Laps:
        completion_rate, median_lap = 1.0, 80.0

    def laps(policy, tr, v0, runs):
        calls.append((tr.name, v0, runs))
        assert np.isfinite(np.asarray(policy(np.zeros(racing.OBS_DIM, np.float32)))).all()
        assert v0 == float(reference_speed(tr)[0]) * 0.9
        return Laps()

    monkeypatch.setattr(mod, "timed_laps", laps)
    monkeypatch.setattr(mod, "refine_reference", lambda tr: tr)
    if not (tmp_path / "data").exists():
        (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", name, "--steps", "128",
                                      "--num-envs", "8", "--num-steps", "8", "--bank", "8",
                                      "--refresh", "4", "--ensemble-heads", "2", *extra])
    mod.main()
    return calls


@needs_tracks
def test_train_dr_skip_eval_rdev_writes_no_rdev_result_and_the_default_is_unchanged(
        tmp_path, monkeypatch, capsys, float32):
    """VG-1's pilot option (docs/prereg/vg1-value-gradient.md, Section 4.1). Off by default,
    every R-dev circuit is evaluated as before (timed_laps once per circuit in R_DEV's order,
    from 0.9 x its reference start speed, 10 runs), written to eval_rdev.json and printed.
    With --skip-eval-rdev no lap is run and no R-dev result is written or printed, and the
    training is the same bit for bit: checkpoints, metrics and config but for the flag."""
    from apexzero.tracks.corners import R_DEV

    mod = _load_train_dr(monkeypatch)
    calls = _run_train_dr_rdev(mod, tmp_path, monkeypatch, "default", [])
    out_default = capsys.readouterr().out
    skipped = _run_train_dr_rdev(mod, tmp_path, monkeypatch, "skip", ["--skip-eval-rdev"])
    out_skip = capsys.readouterr().out
    a, b = tmp_path / "runs" / "default", tmp_path / "runs" / "skip"

    assert [c[0] for c in calls] == list(R_DEV) and {c[2] for c in calls} == {10}
    rows = json.loads((a / "eval_rdev.json").read_text())
    assert list(rows) == list(R_DEV)
    assert all(r == {"completion": 1.0, "median_lap_s": 80.0} for r in rows.values())
    assert all(f'{n} {{"completion": 1.0' in out_default for n in R_DEV)

    assert skipped == [] and not (b / "eval_rdev.json").exists()
    assert sorted(p.name for p in b.iterdir()) == ["ckpt", "config.json", "metrics.csv"]
    low = out_skip.casefold()
    assert not any(n.casefold() in low for n in R_DEV) and "completion" not in low
    assert "lap" not in low.replace("--skip-eval-rdev: no r-dev evaluation", "")

    ca, cb = (json.loads((d / "config.json").read_text()) for d in (a, b))
    assert ca["args"]["skip_eval_rdev"] is False and cb["args"]["skip_eval_rdev"] is True
    for c in (ca, cb):
        del c["args"]["name"], c["args"]["skip_eval_rdev"]
    assert ca == cb
    cks = [sorted((d / "ckpt").glob("*.pkl")) for d in (a, b)]
    assert [p.name for p in cks[0]] == [p.name for p in cks[1]] and cks[0]
    for pa, pb in zip(*cks, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), pa.name
    ma, mb = ((d / "metrics.csv").read_text().splitlines() for d in (a, b))
    head = ma[0].split(",")
    keep = [i for i, k in enumerate(head) if k not in ("wall_s", "sps")]
    assert ma[0] == mb[0] and len(ma) == len(mb) >= 2
    for ra, rb in zip(ma[1:], mb[1:], strict=True):
        fa, fb = ra.split(","), rb.split(",")
        assert [fa[i] for i in keep] == [fb[i] for i in keep]


# ------------------------------------------------------------------ scripts/vgl.py


def _load_vgl(monkeypatch):
    scripts = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setenv("XLA_FLAGS", __import__("os").environ.get("XLA_FLAGS", ""))
    spec = importlib.util.spec_from_file_location("vgl_script", scripts / "vgl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vgl_residual_on_a_saved_snapshot(tmp_path, monkeypatch, float32):
    """The residual check (M0) reads a checkpoint in the shared format and scores its critic's
    dV/dy against the term's targets; a critic trained with the term scores lower."""
    from apexzero.agents.checkpoints import save_snapshot

    mod = _load_vgl(monkeypatch)
    env, bank = racing_env(window_steps=40)
    cfg = PPOConfig(**{**TINY, "num_envs": 16, "total_timesteps": 16 * 16 * 12})
    vg = ValueGradient(coef=1.0)
    res = {}
    for name, v in (("plain", None), ("term", vg)):
        train, _, _ = make_train(cfg, env, vg=v)
        agent, _ = jax.jit(train)(jax.random.PRNGKey(0))
        snap = {"actor": agent.actor.params, "critic": agent.critic.params,
                "obs_rms": agent.obs_rms, "ret_rms": agent.ret_rms}
        path = save_snapshot(tmp_path / name, snap, cfg, cfg.total_timesteps)
        cfg2, loaded = mod.load_snapshot(path)
        assert cfg2 == cfg
        res[name] = mod.residual(cfg2, loaded, env, ValueGradient(coef=1.0), rollouts=2,
                                 num_envs=16, num_steps=16, seed=3)
    for r in res.values():
        assert r["components"] == list(VG_NAMES) and r["rows"] == 2 * 16 * 16
        assert r["valid_rows"] > 0.9 * r["rows"] and r["action_pairs"] > 0.5 * r["rows"]
        assert all(np.isfinite(x) for x in r["normalised_residual"].values())
        assert all(np.isfinite(x) for x in r["action_projected_residual"].values())
        for a in ("steer", "pedal"):
            for k in ("action_projection_share_target", "action_projection_share_critic"):
                assert abs(sum(r[k][a].values()) - 1.0) < 1e-6
    assert res["term"]["normalised_residual_mean"] < res["plain"]["normalised_residual_mean"]
