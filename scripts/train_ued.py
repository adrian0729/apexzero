"""Curriculum comparison C (plan 2.4): one arm on the labelled pool, with checkpoints.

    uv run python scripts/train_ued.py --arm plr-pvl --steps 1e9 --name c/plr-pvl-s0 --seed 0
    uv run python scripts/train_ued.py --arm plr-oracle --pool-file outputs/pool.npz \
        --labels outputs/pool_jstar.npz ...

Arms: dr-pool, plr-pvl, plr-maxmc, plr-oracle, sfl (``apexzero.ued.trainer``). Defaults are
the plan's: 4,000 pool levels, 1,200-step windows, 128-step chunks, JaxUED's PLR defaults
(buffer 1,000, replay 0.8, staleness 0.3, rank beta 0.3, fill 0.5) and SFL with T = 50,
M = 4, K = 1,000, rho = 0.5. PPO matches ``train_dr.py`` except the chunk length.

The pool is the "pool" stream of ``apexzero.tracks.levels``: loaded from ``--pool-file``
(``levels.save`` format, the frozen labelled pool) or regenerated from ``--levels-root``.
plr-oracle needs ``--pool-file`` and ``--labels``, a .npz of J*_20 per pool level written by
``ued.pool.save_oracle_labels``; it stores the pool's level encoding, and labels from any
other pool are refused.

Outputs in <runs-dir>/<name>/:
    config.json, metrics.csv (one row per chunk; wall and process CPU time)
    ckpt/<timesteps>.pkl     params + normalisers, ``agents.checkpoints`` (shared format and
                             naming), 8 log-spaced from 10M to the end, plus the last
    sampler/<timesteps>.pkl  the PLR buffer or SFL's D and learnabilities, same timesteps
    records/<chunk>.npz      every completed episode with its scores (``ued.scoring``) and
                             the write of its group (write_row / write_outcome / write_score);
                             PLR arms also every committed buffer operation with its outcome,
                             score, R_max and buffer index (``ev_*``, ``ued.plr.event_rows``),
                             joined to the records on (chunk, row)
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

from apexzero.agents.checkpoints import checkpoint_schedule, save_snapshot
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--steps", type=float, default=1e9)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=128, help="chunk length")
    ap.add_argument("--network", default="mlp", choices=["mlp", "simba"])
    ap.add_argument("--width", type=int, help="actor and critic width (default: train_dr.py's)")
    ap.add_argument("--ensemble-heads", type=int, default=5)
    ap.add_argument("--window-steps", type=int, default=1200)
    ap.add_argument("--pool-file", type=Path)
    ap.add_argument("--levels-root", type=int, default=0)
    ap.add_argument("--pool-size", type=int, default=paper_levels.SIZES["pool"])
    ap.add_argument("--labels", type=Path, help="J*_20 of the pool (save_oracle_labels .npz), plr-oracle")
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
    ap.add_argument("--checkpoint-first", type=float, default=10e6)
    ap.add_argument("--records-every", type=int, default=50, help="chunks per records shard")
    ap.add_argument("--no-mna", action="store_true", help="skip MNA's O(T^2) term in the records")
    args = ap.parse_args(argv)
    if args.arm == "plr-oracle" and (args.pool_file is None or args.labels is None):
        ap.error("plr-oracle needs --pool-file (the frozen labelled pool) and --labels (its J*_20)")
    return args


def main(argv=None):
    args = parse_args(argv)
    out = args.runs_dir / args.name
    for sub in ("ckpt", "sampler", "records"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if args.pool_file:
        lv = paper_levels.load(args.pool_file)
    else:
        lv = paper_levels.make_levels(jax.random.PRNGKey(args.levels_root), "pool", n=args.pool_size)
    oracle_j = pool.load_oracle_labels(args.labels, lv) if args.arm == "plr-oracle" else None

    params = racing.EnvParams(window_steps=args.window_steps)
    g = fit_grammar([load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV])
    bank = pool.make_bank(lv, g, params=params)
    width = args.width or 256
    cfg = PPOConfig(num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=int(args.steps),
                    network=args.network, gamma=0.995, gae_lambda=0.95, actor_width=width,
                    critic_width=args.width or (256 if args.network == "mlp" else 384), mlp_layers=3,
                    activation="tanh", ensemble_heads=args.ensemble_heads)
    ued = UEDConfig(arm=args.arm,
                    plr=PLRConfig(capacity=args.capacity, replay_prob=args.replay_prob,
                                  staleness_coeff=args.staleness, temperature=args.temperature,
                                  minimum_fill_ratio=args.min_fill),
                    sfl=SFLConfig(refresh_every=args.sfl_every, windows=args.sfl_windows,
                                  buffer_size=args.sfl_k, rho=args.sfl_rho), mna=not args.no_mna)
    (out / "config.json").write_text(json.dumps(
        {"ppo": asdict(cfg), "ued": asdict(ued), "args": {k: str(v) for k, v in vars(args).items()},
         "env": params._asdict(), "pool_levels": int(lv.ctx.shape[0])}, indent=1))

    # One chunk is one PPO update, so the shared schedule (and file naming) applies as is.
    n_chunks = max(cfg.num_updates, 1)
    ckpt_at = checkpoint_schedule(cfg.total_timesteps, cfg.batch_size, args.checkpoint_first, args.checkpoints)

    trainer = UEDTrainer(cfg, ued, bank, pool.level_pytree(lv), params, oracle_j=oracle_j)
    t0, c0 = time.perf_counter(), time.process_time()
    st = trainer.init(jax.random.PRNGKey(args.seed))
    log = open(out / "metrics.csv", "w", newline="")  # noqa: SIM115  (written chunk by chunk)
    writer, shard, shard_start = None, [], 0

    def flush(upto: int):
        nonlocal shard, shard_start
        if shard:
            cols = {}
            for prefix, i in (("", 0), ("ev_", 1)):
                parts = [s[i] for s in shard if s[i] is not None]
                if parts:
                    cols.update({prefix + k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
            cols = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in cols.items()}
            np.savez_compressed(out / "records" / f"{shard_start:07d}.npz", **cols)
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
        shard.append((chunk_log.records, chunk_log.event_rows))
        if (k + 1) % args.records_every == 0:
            flush(k + 1)
        if k in ckpt_at:
            ts, done = ckpt_at[k], (k + 1) * cfg.batch_size
            save_snapshot(out, trainer.snapshot(st), cfg, ts, timesteps_done=done, arm=args.arm)
            with open(out / "sampler" / f"{ts:012d}.pkl", "wb") as fh:
                pickle.dump(dict(trainer.sampler_snapshot(st), timesteps=ts, timesteps_done=done), fh)
        if k % 25 == 0 or k == n_chunks - 1:
            keys = ("chunk", "timesteps", "train_frac", "episodes", "buf_size", "sps", "cpu_s")
            print(json.dumps({x: round(m[x], 3) if isinstance(m[x], float) else m[x]
                              for x in keys if x in m}), flush=True)
    flush(n_chunks)
    log.close()
    return out


if __name__ == "__main__":
    main()
