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
