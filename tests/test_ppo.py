"""Fast PPO smoke tests (full verification against rejax: scripts/verify_ppo.py)."""

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import PPOConfig, RunningMeanStd, from_gymnax, make_train


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (1000, 4))
    rms = RunningMeanStd.create((4,))
    for chunk in np.split(data, 10):
        rms = rms.update(jnp.asarray(chunk))
    assert np.allclose(rms.mean, data.mean(0), atol=1e-3)
    assert np.allclose(rms.var, data.var(0), rtol=1e-2)


def test_ppo_learns_cartpole_quickly():
    cfg = PPOConfig(num_envs=16, num_steps=128, total_timesteps=150_000, update_epochs=4,
                    num_minibatches=4, gamma=0.99, ent_coef=0.01, ent_coef_final=0.01,
                    lr=2.5e-4, normalize_obs=False, normalize_reward=False, network="mlp",
                    actor_width=64, critic_width=64, mlp_layers=2)
    train, _, _ = make_train(cfg, from_gymnax("CartPole-v1"))
    _, metrics = jax.jit(train)(jax.random.PRNGKey(0))
    ret = np.asarray(metrics["episode_return"])
    assert ret[-5:].mean() > 3 * ret[:5].mean()
    assert ret[-5:].mean() > 100


def test_simba_network_runs():
    cfg = PPOConfig(num_envs=8, num_steps=16, total_timesteps=8 * 16 * 2, network="simba",
                    actor_width=32, critic_width=48)
    train, _, _ = make_train(cfg, from_gymnax("Pendulum-v1"))
    agent, metrics = jax.jit(train)(jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(metrics["value_loss"])))


def test_ensemble_heads_train_without_touching_the_main_critic():
    import flax.linen as nn  # noqa: F401

    from apexzero.agents.ppo import Critic

    cfg = PPOConfig(network="mlp", critic_width=32, mlp_layers=2, ensemble_heads=5)
    critic = Critic(cfg)
    params = critic.init(jax.random.PRNGKey(0), jnp.zeros((1, 4)))
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 4))
    v, heads = critic.apply(params, obs, with_heads=True)
    assert v.shape == (8,) and heads.shape == (8, 5)
    # Gradient of the head loss must not reach the shared trunk.
    g = jax.grad(lambda p: ((critic.apply(p, obs, with_heads=True)[1] - 1.0) ** 2).mean())(params)
    trunk = [leaf for path, leaf in jax.tree_util.tree_leaves_with_path(g["params"])
             if "MLPEncoder" in jax.tree_util.keystr(path)]
    assert trunk and all(float(jnp.abs(x).max()) == 0.0 for x in trunk)


def test_checkpoint_callback_receives_snapshots():
    seen = []
    cfg = PPOConfig(num_envs=8, num_steps=16, total_timesteps=8 * 16 * 3, network="mlp",
                    actor_width=16, critic_width=16, mlp_layers=1)
    train, _, _ = make_train(cfg, from_gymnax("CartPole-v1"),
                             checkpoint_fn=lambda snap, upd: seen.append(int(upd)))
    jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(0)))
    assert seen == [0, 1, 2]


def test_training_with_per_head_targets_runs_and_heads_learn():
    # Value clipping off: with unnormalised CartPole rewards PPO's +/-0.2 value clip makes the
    # main critic lag the true returns, which the (unclipped) heads would then "disagree" with.
    cfg = PPOConfig(num_envs=16, num_steps=64, total_timesteps=16 * 64 * 30, network="mlp",
                    actor_width=32, critic_width=32, mlp_layers=2, ensemble_heads=3,
                    normalize_obs=False, normalize_reward=False, gamma=0.99, clip_value=False)
    train, _, critic = make_train(cfg, from_gymnax("CartPole-v1"))
    agent, metrics = jax.jit(train)(jax.random.PRNGKey(0))
    assert np.all(np.isfinite(np.asarray(metrics["value_loss"])))
    obs = jax.random.normal(jax.random.PRNGKey(2), (64, 4)) * 0.05
    _, heads = critic.apply(agent.critic.params, obs, with_heads=True)
    per_head = jnp.mean(heads, 0)
    # Heads trained on their own GAE targets learn the return scale (start-state returns
    # are tens here) and agree with each other, which the disagreement score relies on.
    assert float(per_head.min()) > 5.0
    assert float(per_head.std() / per_head.mean()) < 0.3


def test_log_spaced_checkpoints_are_written_in_the_shared_format(tmp_path):
    import pickle

    from apexzero.agents.checkpoints import list_checkpoints, log_spaced, make_checkpoint_fn
    from apexzero.audit.rollouts import snapshot_from_pickle

    assert log_spaced(200e6, 10e6, 5)[0] == 10_000_000 and log_spaced(200e6, 10e6, 5)[-1] == 200_000_000
    cfg = PPOConfig(num_envs=8, num_steps=16, total_timesteps=8 * 16 * 6, network="mlp",
                    actor_width=16, critic_width=16, mlp_layers=1)
    train, _, _ = make_train(cfg, from_gymnax("CartPole-v1"),
                             checkpoint_fn=make_checkpoint_fn(tmp_path, cfg, first=8 * 16 * 2, n=3))
    jax.block_until_ready(jax.jit(train)(jax.random.PRNGKey(0)))
    files = list_checkpoints(tmp_path)
    assert len(files) >= 2 and files[-1].stem == f"{8 * 16 * 6:012d}"
    with open(files[-1], "rb") as fh:
        snap = snapshot_from_pickle(pickle.load(fh))
    assert float(snap.sigma) > 0


# --------------------------------------------------------------------------- loss mask (UED)


def test_masked_normaliser_update_uses_only_masked_rows():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(2.0, 3.0, (64, 3)))
    w = jnp.asarray(rng.random(64) < 0.4)
    rms = RunningMeanStd.create((3,)).update(jnp.asarray(rng.normal(size=(16, 3))))
    ref = rms.update(x[np.asarray(w)])
    got = rms.update_masked(x, w)
    for a, b in zip(ref, got, strict=True):
        assert np.allclose(a, b, rtol=1e-5)
    same = rms.update_masked(x, jnp.zeros(64, bool))
    for a, b in zip(rms, same, strict=True):
        assert np.array_equal(a, b)


def _synthetic_rollout(seed, T=8, N=4, obs_dim=3, E=2):
    from apexzero.agents.ppo import Transition

    rng = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
    term = np.zeros((T, N), bool)
    term[3, 3] = True                       # slot 3: an episode boundary after t = 3
    tr = Transition(f(T, N, obs_dim), f(T, N, 2), f(T, N), f(T, N), f(T, N), jnp.asarray(term),
                    jnp.zeros((T, N), bool), f(T, N), f(T, N), f(T, N, E), f(T, N, E))
    mask = np.ones((T, N), bool)
    mask[:, 2] = False                      # a whole exploratory slot
    mask[4:, 3] = False                     # the episode after the boundary
    return tr, f(N), f(N, E), jnp.asarray(mask)


def _learner(E=2):
    from apexzero.agents.ppo import Actor, Critic, Env, init_agent, make_learner

    cfg = PPOConfig(num_envs=4, num_steps=8, num_minibatches=2, update_epochs=2, network="mlp",
                    actor_width=8, critic_width=8, mlp_layers=1, ensemble_heads=E)
    env = Env(None, None, 3, 2, False)
    actor, critic = Actor(cfg, 2, False), Critic(cfg)
    agent = init_agent(cfg, actor, critic, 3, jax.random.PRNGKey(0), jax.random.PRNGKey(1), 10)
    return jax.jit(make_learner(cfg, env, actor, critic, 10)), agent


def _same(a, b) -> bool:
    return all(jax.tree.leaves(jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)))


def test_masked_out_transitions_cannot_move_the_networks():
    learn, agent = _learner()
    tr, lv, lh, mask = _synthetic_rollout(0)
    out, *_ = learn(agent, tr, lv, lh, jax.random.PRNGKey(2), 0, mask)
    # Replace every masked-out entry (and the masked-out slot's bootstrap) by other data.
    tr2, lv2, lh2, _ = _synthetic_rollout(1)
    keep = lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 2)), a, b)
    fields = {k: keep(getattr(tr, k), getattr(tr2, k)) for k in tr._fields
              if k not in ("terminated", "truncated")}
    out_mod, *_ = learn(agent, tr._replace(**fields), lv.at[2].set(lv2[2]), lh.at[2].set(lh2[2]),
                        jax.random.PRNGKey(2), 0, mask)
    assert _same(out.actor.params, out_mod.actor.params)
    assert _same(out.critic.params, out_mod.critic.params)
    # ...while changing masked-in data does move them.
    out_in, *_ = learn(agent, tr._replace(reward=tr.reward + 1.0), lv, lh, jax.random.PRNGKey(2), 0, mask)
    assert not _same(out.critic.params, out_in.critic.params)


def test_fully_masked_update_leaves_agent_and_optimiser_untouched():
    learn, agent = _learner()
    tr, lv, lh, _ = _synthetic_rollout(0)
    out, _, (_, _, m) = learn(agent, tr, lv, lh, jax.random.PRNGKey(2), 0, jnp.zeros((8, 4), bool))
    assert _same((agent.actor, agent.critic), (out.actor, out.critic))
    assert all(np.all(np.asarray(x) == 0.0) for x in m)


def test_rollout_hooks_mask_the_return_normaliser_and_trace_steps():
    from apexzero.agents.ppo import Actor, Critic, init_agent, make_rollout

    cfg = PPOConfig(num_envs=8, num_steps=16, network="mlp", actor_width=8, critic_width=8, mlp_layers=1)
    env = from_gymnax("CartPole-v1")
    actor, critic = Actor(cfg, env.action_dim, True), Critic(cfg)
    agent = init_agent(cfg, actor, critic, env.obs_dim, jax.random.PRNGKey(0), jax.random.PRNGKey(1), 4)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(3), 8), None)
    z = jnp.zeros(8)
    args = (agent, st, obs, z, z, z, jax.random.PRNGKey(4), None)
    plain = make_rollout(cfg, env, actor, critic)
    none = make_rollout(cfg, env, actor, critic, mask_fn=lambda s: jnp.zeros(8, bool),
                        trace_fn=lambda s, info: info["final_obs"][:, 0])
    full = make_rollout(cfg, env, actor, critic, mask_fn=lambda s: jnp.ones(8, bool))
    (a0, *_), r0 = plain(*args)
    (a1, *_), r1 = none(*args)
    (a2, *_), r2 = full(*args)
    assert r0.mask is None and r0.trace is None
    assert r1.mask.shape == (16, 8) and not bool(r1.mask.any()) and r1.trace.shape == (16, 8)
    assert np.array_equal(a1.ret_rms.var, agent.ret_rms.var)            # nothing masked in
    assert np.allclose(a2.ret_rms.var, a0.ret_rms.var, rtol=1e-5)        # everything masked in
    assert np.array_equal(r1.traj.raw_reward, r0.traj.raw_reward)        # same trajectory
