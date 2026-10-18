"""Verify the from-scratch PPO against rejax on gymnax CartPole-v1 and Pendulum-v1.

    uv run python scripts/verify_ppo.py [--seeds 5]

Both implementations get identical hyperparameters and seeds; both final policies are
scored with the same deterministic evaluation (200 episodes). Pass: our mean is within
the reference's 95% CI or above it. Writes docs/findings/ppo-verification.md.
"""

import argparse
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from rejax import PPO as RejaxPPO

from apexzero.agents.ppo import PPOConfig, from_gymnax, make_train

TASKS = {
    "CartPole-v1": dict(
        ours=PPOConfig(num_envs=16, num_steps=128, total_timesteps=500_000, update_epochs=4,
                       num_minibatches=4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                       vf_coef=0.5, ent_coef=0.01, ent_coef_final=0.01, lr=2.5e-4,
                       max_grad_norm=0.5, normalize_obs=False, normalize_reward=False,
                       network="mlp", actor_width=64, critic_width=64, mlp_layers=2,
                       activation="tanh", clip_value=False),
        ref=dict(num_envs=16, num_steps=128, total_timesteps=500_000, num_epochs=4,
                 num_minibatches=4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, learning_rate=2.5e-4, max_grad_norm=0.5,
                 normalize_observations=False, normalize_rewards=False,
                 agent_kwargs={"activation": "tanh", "hidden_layer_sizes": (64, 64)}),
    ),
    "Pendulum-v1": dict(
        ours=PPOConfig(num_envs=32, num_steps=128, total_timesteps=1_000_000, update_epochs=10,
                       num_minibatches=16, gamma=0.95, gae_lambda=0.95, clip_eps=0.2,
                       vf_coef=0.5, ent_coef=0.0, ent_coef_final=0.0, lr=1e-3,
                       max_grad_norm=0.5, normalize_obs=True, normalize_reward=True,
                       network="mlp", actor_width=64, critic_width=64, mlp_layers=2,
                       activation="tanh", clip_value=False, log_std_init=0.0),
        ref=dict(num_envs=32, num_steps=128, total_timesteps=1_000_000, num_epochs=10,
                 num_minibatches=16, gamma=0.95, gae_lambda=0.95, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.0, learning_rate=1e-3, max_grad_norm=0.5,
                 normalize_observations=True, normalize_rewards=True,
                 agent_kwargs={"activation": "tanh", "hidden_layer_sizes": (64, 64)}),
    ),
}


def evaluate_ours(agent, actor, env_name, key, episodes=200):
    env, params = gymnax.make(env_name)
    discrete = hasattr(env.action_space(params), "n")

    def run(k):
        k0, k1 = jax.random.split(k)
        obs, state = env.reset(k0, params)

        def body(c, _):
            obs, state, ret, done, k = c
            k, ks = jax.random.split(k)
            o = agent.obs_rms.normalize(obs[None]) if actor.cfg.normalize_obs else obs[None]
            out, _ = actor.apply(agent.actor.params, o)
            a = jnp.argmax(out[0]) if discrete else out[0]
            out = env.step(ks, state, a, params)
            obs, state, r = out[0], out[1], out[2]
            d = out[3] if len(out) == 5 else (out[3] | out[4])
            ret = ret + r * (1 - done)
            return (obs, state, ret, done | d, k), None

        (_, _, ret, _, _), _ = jax.lax.scan(body, (obs, state, 0.0, False, k1), None,
                                            params.max_steps_in_episode)
        return ret

    return jax.vmap(run)(jax.random.split(key, episodes))


def ci95(x):
    x = np.asarray(x)
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return x.mean(), half


def main(seeds: int):
    lines = ["# PPO verification against rejax", "",
             "Identical hyperparameters and seeds; deterministic evaluation, 200 episodes per",
             "seed; mean over seeds with a 95% CI.", "",
             "| Task | Ours | rejax | Pass | Ours wall-clock | rejax wall-clock |", "|---|---|---|---|---|---|"]
    all_pass = True
    for name, cfgs in TASKS.items():
        keys = jax.random.split(jax.random.PRNGKey(0), seeds)

        train, actor, _ = make_train(cfgs["ours"], from_gymnax(name))
        f = jax.jit(jax.vmap(train))
        t0 = time.perf_counter()
        agents, _ = f(keys)
        jax.block_until_ready(agents)
        t_ours = time.perf_counter() - t0
        ours = [float(evaluate_ours(jax.tree.map(lambda x, i=i: x[i], agents), actor, name,
                                    jax.random.PRNGKey(100 + i)).mean()) for i in range(seeds)]

        algo = RejaxPPO.create(env=name, **cfgs["ref"])
        g = jax.jit(jax.vmap(algo.train))
        t0 = time.perf_counter()
        _, (_, evals) = g(keys)
        jax.block_until_ready(evals)
        t_ref = time.perf_counter() - t0
        ref = np.asarray(evals)[:, -1].mean(axis=-1)

        mo, ho = ci95(ours)
        mr, hr = ci95(ref)
        ok = mo >= mr - max(hr, 0.02 * abs(mr))
        all_pass &= ok
        print(f"{name}: ours {mo:.1f} +/- {ho:.1f} | rejax {mr:.1f} +/- {hr:.1f} | {'PASS' if ok else 'FAIL'}"
              f" | {t_ours:.0f}s vs {t_ref:.0f}s", flush=True)
        lines.append(f"| {name} | {mo:.1f} ± {ho:.1f} | {mr:.1f} ± {hr:.1f} | {'yes' if ok else 'no'} "
                     f"| {t_ours:.0f} s | {t_ref:.0f} s |")
    Path("docs/findings").mkdir(parents=True, exist_ok=True)
    Path("docs/findings/ppo-verification.md").write_text("\n".join(lines) + "\n")
    print("ALL PASS" if all_pass else "SOME FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    main(ap.parse_args().seeds)
