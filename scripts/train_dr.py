"""Domain-randomisation (E0-style) training on procedural strips, with checkpoints.

    uv run python scripts/train_dr.py --steps 200e6 --name dr-pilot

Tracks: a rotating on-device bank of generated strips (grammar fitted on R-dev only).
Cars: context sampled per reset in the plan's ranges, plus hidden +/-2% grip.
Checkpoints at log-spaced timesteps (snapshot = params + both normalisers).
Evaluation: timed laps on the R-dev circuits only (the lockbox is never touched), with
the default car, against the s-domain OCP lap where one exists.
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

from apexzero.agents.ppo import Actor, PPOConfig, make_train
from apexzero.env import racing
from apexzero.env.adapters import procedural_env, reference_speed
from apexzero.eval.laps import timed_laps
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm


def log_spaced(total: float, first: float = 10e6, n: int = 16) -> list[int]:
    return sorted({int(x) for x in np.geomspace(first, total, n)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=200e6)
    ap.add_argument("--name", default="dr-pilot")
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=64)
    ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
    ap.add_argument("--bank", type=int, default=4096)
    ap.add_argument("--refresh", type=int, default=256)
    ap.add_argument("--fixed-car", action="store_true")
    ap.add_argument("--ensemble-heads", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path("runs") / args.name
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    dev = [load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV]
    g = fit_grammar(dev)
    env = procedural_env(g, bank_size=args.bank, refresh_per_update=args.refresh,
                         randomize_car=not args.fixed_car)
    cfg = PPOConfig(num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=int(args.steps),
                    network=args.network, gamma=0.995, gae_lambda=0.95, actor_width=256,
                    critic_width=256 if args.network == "mlp" else 384, mlp_layers=3,
                    activation="tanh", ensemble_heads=args.ensemble_heads)
    (out / "config.json").write_text(json.dumps({"ppo": asdict(cfg), "args": vars(args),
                                                 "env": racing.EnvParams()._asdict()}, indent=1))
    ckpt_steps = log_spaced(args.steps)
    ckpt_updates = {max(0, s // cfg.batch_size - 1): s for s in ckpt_steps}
    ckpt_updates[cfg.num_updates - 1] = int(args.steps)

    log = open(out / "metrics.csv", "w", newline="")  # noqa: SIM115  (written from a JAX callback)
    writer = None
    t0 = time.perf_counter()

    def log_fn(m):
        nonlocal writer
        m = {k: float(np.asarray(v)) for k, v in m.items()}
        if int(m["update"]) % 25 and int(m["update"]) != cfg.num_updates - 1:
            return
        m["wall_s"] = time.perf_counter() - t0
        m["sps"] = m["timesteps"] / max(m["wall_s"], 1e-6)
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(m))
            writer.writeheader()
        writer.writerow(m)
        log.flush()

    def checkpoint_fn(snap, upd):
        upd = int(upd)
        if upd in ckpt_updates:
            snap = jax.device_get(snap)
            d = {"actor": snap["actor"], "critic": snap["critic"],
                 "obs_rms": snap["obs_rms"]._asdict(), "ret_rms": snap["ret_rms"]._asdict(),
                 "config": asdict(cfg), "timesteps": ckpt_updates[upd]}
            with open(out / "ckpt" / f"{ckpt_updates[upd]:012d}.pkl", "wb") as fh:
                pickle.dump(d, fh)

    train, _, _ = make_train(cfg, env, log_fn, checkpoint_fn)
    agent, _ = jax.jit(train)(jax.random.PRNGKey(args.seed))
    jax.block_until_ready(agent)
    log.close()

    pol_actor = Actor(cfg, 2, False)

    def policy(obs):
        return pol_actor.apply(agent.actor.params, agent.obs_rms.normalize(obs))[0]

    results = {}
    for name in R_DEV:
        tr = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
        res = timed_laps(policy, tr, float(reference_speed(tr)[0]) * 0.9, runs=10)
        row = {"completion": res.completion_rate, "median_lap_s": res.median_lap}
        ocp = Path("outputs/ocp") / f"{name}.pkl"
        if ocp.exists():
            with open(ocp, "rb") as fh:
                t_ocp = pickle.load(fh).lap_time
            row.update(ocp_lap_s=t_ocp, gap=res.median_lap / t_ocp - 1.0)
        results[name] = row
        print(name, json.dumps(row), flush=True)
    (out / "eval_rdev.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
