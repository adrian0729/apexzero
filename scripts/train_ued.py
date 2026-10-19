"""Curriculum comparison C (plan 2.4): one arm on the labelled pool, with checkpoints.

    uv run python scripts/train_ued.py --arm plr-pvl --steps 1e9 --name c/plr-pvl-s0 --seed 0
    uv run python scripts/train_ued.py --arm plr-oracle --labels outputs/pool_jstar.npy ...

Arms: dr-pool, plr-pvl, plr-maxmc, plr-oracle, sfl (``apexzero.ued.trainer``). Defaults are
the plan's: 4,000 pool levels, 1,200-step windows, 128-step chunks, JaxUED's PLR defaults
(buffer 1,000, replay 0.8, staleness 0.3, rank beta 0.3, fill 0.5) and SFL with T = 50,
M = 4, K = 1,000, rho = 0.5. PPO matches ``train_dr.py`` except the chunk length.

The pool is the "pool" stream of ``apexzero.tracks.levels``: loaded from ``--pool-file``
(``levels.save`` format, the frozen labelled pool) or regenerated from ``--levels-root``.
``--labels`` is a .npy of J*_20 per pool level, in pool order (plr-oracle only).

Outputs in runs/<name>/:
    config.json, metrics.csv (one row per chunk; wall and process CPU time)
    ckpt/<timesteps>.pkl     params + normalisers (train_dr.py's format), log-spaced
    sampler/<timesteps>.pkl  the PLR buffer or SFL's D and learnabilities, same timesteps
    records/<chunk>.npz      every completed episode with its scores (``ued.scoring``)
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

from apexzero.agents.ppo import PPOConfig
from apexzero.env import racing
from apexzero.tracks import levels as paper_levels
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar
from apexzero.tracks.track import load_tumftm
from apexzero.ued import pool
from apexzero.ued.plr import PLRConfig
from apexzero.ued.sfl import SFLConfig
from apexzero.ued.trainer import ARMS, UEDConfig, UEDTrainer


def checkpoint_chunks(total: float, batch: int, n: int, first: float = 10e6) -> set[int]:
    """Chunks closest to n log-spaced timesteps from 10M (or total/100 for short runs) to
    the end; files are named by the timesteps actually done at that chunk."""
    first = min(first, total / 100)
    n_chunks = max(int(total) // batch, 1)
    return {min(max(round(x / batch) - 1, 0), n_chunks - 1) for x in np.geomspace(first, total, n)} | {
        n_chunks - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--steps", type=float, default=1e9)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=128, help="chunk length")
    ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
    ap.add_argument("--ensemble-heads", type=int, default=5)
    ap.add_argument("--window-steps", type=int, default=1200)
    ap.add_argument("--pool-file", type=Path)
    ap.add_argument("--levels-root", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=paper_levels.SIZES["pool"])
    ap.add_argument("--labels", type=Path, help="J*_20 per pool level (.npy), for plr-oracle")
    ap.add_argument("--capacity", type=int, default=1000)
    ap.add_argument("--replay-prob", type=float, default=0.8)
    ap.add_argument("--staleness", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--min-fill", type=float, default=0.5)
    ap.add_argument("--sfl-every", type=int, default=50)
    ap.add_argument("--sfl-windows", type=int, default=4)
    ap.add_argument("--sfl-k", type=int, default=1000)
    ap.add_argument("--sfl-rho", type=float, default=0.5)
    ap.add_argument("--checkpoints", type=int, default=8)
    ap.add_argument("--records-every", type=int, default=50, help="chunks per records shard")
    args = ap.parse_args()

    out = Path("runs") / args.name
    for sub in ("ckpt", "sampler", "records"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if args.pool_file:
        lv = paper_levels.load(args.pool_file)
    else:
        lv = paper_levels.make_levels(jax.random.PRNGKey(args.levels_root), "pool", n=args.pool_size)
    oracle_j = None
    if args.arm == "plr-oracle":
        if args.labels is None:
            ap.error("plr-oracle needs --labels (J*_20 of every pool level)")
        oracle_j = np.load(args.labels)
        if oracle_j.shape != (lv.ctx.shape[0],):
            ap.error(f"--labels has shape {oracle_j.shape}, the pool has {lv.ctx.shape[0]} levels")

    params = racing.EnvParams(window_steps=args.window_steps)
    g = fit_grammar([load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV])
    bank = pool.make_bank(lv, g, params=params)
    cfg = PPOConfig(num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=int(args.steps),
                    network=args.network, gamma=0.995, gae_lambda=0.95, actor_width=256,
                    critic_width=256 if args.network == "mlp" else 384, mlp_layers=3,
                    activation="tanh", ensemble_heads=args.ensemble_heads)
    ued = UEDConfig(arm=args.arm,
                    plr=PLRConfig(capacity=args.capacity, replay_prob=args.replay_prob,
                                  staleness_coeff=args.staleness, temperature=args.temperature,
                                  minimum_fill_ratio=args.min_fill),
                    sfl=SFLConfig(refresh_every=args.sfl_every, windows=args.sfl_windows,
                                  buffer_size=args.sfl_k, rho=args.sfl_rho))
    (out / "config.json").write_text(json.dumps(
        {"ppo": asdict(cfg), "ued": asdict(ued), "args": {k: str(v) for k, v in vars(args).items()},
         "env": params._asdict(), "pool_levels": int(lv.ctx.shape[0])}, indent=1))

    n_chunks = max(cfg.num_updates, 1)
    ckpt_chunks = checkpoint_chunks(args.steps, cfg.batch_size, args.checkpoints)

    trainer = UEDTrainer(cfg, ued, bank, pool.level_pytree(lv), params, oracle_j=oracle_j)
    t0, c0 = time.perf_counter(), time.process_time()
    st = trainer.init(jax.random.PRNGKey(args.seed))
    log = open(out / "metrics.csv", "w", newline="")  # noqa: SIM115  (written chunk by chunk)
    writer, shard, shard_start = None, [], 0

    def flush(upto: int):
        nonlocal shard, shard_start
        if shard:
            cat = {k: np.concatenate([r[k] for r in shard]) for k in shard[0]}
            cat = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in cat.items()}
            np.savez_compressed(out / "records" / f"{shard_start:07d}.npz", **cat)
        shard, shard_start = [], upto

    for k in range(n_chunks):
        st, chunk_log = trainer.step(st)
        m = dict(chunk_log.metrics, wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0)
        m["sps"] = m["timesteps"] / max(m["wall_s"], 1e-6)
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(m), extrasaction="ignore", restval="")
            writer.writeheader()
        writer.writerow(m)
        log.flush()
        shard.append(chunk_log.records)
        if (k + 1) % args.records_every == 0:
            flush(k + 1)
        if k in ckpt_chunks:
            ts = (k + 1) * cfg.batch_size
            snap = dict(trainer.snapshot(st), config=asdict(cfg), timesteps=ts, arm=args.arm)
            with open(out / "ckpt" / f"{ts:012d}.pkl", "wb") as fh:
                pickle.dump(snap, fh)
            with open(out / "sampler" / f"{ts:012d}.pkl", "wb") as fh:
                pickle.dump(dict(trainer.sampler_snapshot(st), timesteps=ts), fh)
        if k % 25 == 0 or k == n_chunks - 1:
            keys = ("chunk", "timesteps", "train_frac", "episodes", "buf_size", "sps", "cpu_s")
            print(json.dumps({x: round(m[x], 3) if isinstance(m[x], float) else m[x]
                              for x in keys if x in m}), flush=True)
    flush(n_chunks)
    log.close()


if __name__ == "__main__":
    main()
