"""Train a PPO racing agent on one or more tracks and evaluate timed laps.

    uv run python scripts/train.py --tracks Hockenheim --steps 50e6 --name hockenheim-specialist
    uv run python scripts/train.py --tracks Hockenheim --steps 2e6 --name smoke --network mlp

Writes runs/<name>/{config.json, metrics.csv, agent.msgpack, eval.json}.
"""

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import numpy as np
from flax import serialization

from apexzero.agents.checkpoints import make_checkpoint_fn
from apexzero.agents.ppo import PPOConfig, make_train
from apexzero.env import racing
from apexzero.env.adapters import bank_env, make_bank, reference_speed
from apexzero.eval.laps import timed_laps
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["Hockenheim"])
    ap.add_argument("--steps", type=float, default=50e6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    name = args.name or f"{'-'.join(args.tracks)}-{args.network}-{int(args.steps / 1e6)}M"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    tracks = [refine_reference(load_tumftm(f"data/tracks/tumftm/{t}.csv")) for t in args.tracks]
    env = bank_env(make_bank(tracks))
    cfg = PPOConfig(
        num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=int(args.steps),
        lr=args.lr, network=args.network, gamma=0.995, gae_lambda=0.95,
        actor_width=256, critic_width=256 if args.network == "mlp" else 384,
        mlp_layers=3, activation="tanh",
    )
    (out / "config.json").write_text(json.dumps({"ppo": asdict(cfg), "tracks": args.tracks,
                                                 "env": racing.EnvParams()._asdict(),
                                                 "seed": args.seed}, indent=1))

    log = open(out / "metrics.csv", "w", newline="")  # noqa: SIM115  (closed after training; written from a JAX callback)
    writer = None
    t_start = time.perf_counter()

    def log_fn(m):
        nonlocal writer
        m = {k: float(np.asarray(v)) for k, v in m.items()}
        if int(m["update"]) % args.log_every and int(m["update"]) != cfg.num_updates - 1:
            return
        m["wall_s"] = time.perf_counter() - t_start
        m["sps"] = m["timesteps"] / max(m["wall_s"], 1e-6)
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(m))
            writer.writeheader()
        writer.writerow(m)
        log.flush()
        print(f"[{m['wall_s']:6.0f}s] {m['timesteps'] / 1e6:7.2f}M steps  {m['sps'] / 1e3:5.1f}k sps  "
              f"ep_ret {m['episode_return']:8.2f}  ep_len {m['episode_length']:7.1f}  "
              f"term {m['terminated_frac'] * 100:5.2f}%  ent {m['entropy']:6.3f}  "
              f"kl {m['approx_kl']:.4f}  ev {m['explained_var']:.2f}", flush=True)

    # Log-spaced snapshots from 5M steps, so learning can be compared across checkpoints.
    ckpt_fn = make_checkpoint_fn(out, cfg, first=5e6)
    train, actor, _ = make_train(cfg, env, log_fn, ckpt_fn)
    agent, _ = jax.jit(train)(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(agent)
    log.close()
    (out / "agent.msgpack").write_bytes(serialization.to_bytes(agent))
    with open(out / "agent.pkl", "wb") as fh:  # plain arrays: params and normalisers
        pickle.dump(jax.device_get({
            "actor": agent.actor.params, "critic": agent.critic.params,
            "obs_rms": agent.obs_rms._asdict(), "ret_rms": agent.ret_rms._asdict(),
            "config": asdict(cfg),
        }), fh)

    def policy(obs):
        return actor.apply(agent.actor.params, agent.obs_rms.normalize(obs))[0]

    results = {}
    for tr in tracks:
        v0 = float(reference_speed(tr)[0]) * 0.9
        res = timed_laps(policy, tr, v0, runs=20)
        results[tr.name] = {"completion": res.completion_rate, "median_lap_s": res.median_lap,
                            "lap_times": [None if np.isnan(x) else float(x) for x in res.lap_time]}
        ocp = Path("outputs/ocp") / f"{tr.name}.pkl"
        if ocp.exists():
            with open(ocp, "rb") as fh:
                t_ocp = pickle.load(fh).lap_time
            results[tr.name]["ocp_lap_s"] = t_ocp
            results[tr.name]["gap"] = res.median_lap / t_ocp - 1.0
        print(tr.name, json.dumps({k: v for k, v in results[tr.name].items() if k != "lap_times"}))
    (out / "eval.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
