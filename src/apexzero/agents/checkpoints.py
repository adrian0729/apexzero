"""Log-spaced training snapshots, shared by every training script.

A snapshot holds the actor and critic params, both normalisers and the PPO config, as
plain arrays, in ``<run>/ckpt/<timesteps:012d>.pkl``. The audit, the checkpoint
comparisons (lap times, ghosts, per-corner learning) and the demo all read this format
(``apexzero.audit.rollouts.snapshot_from_pickle``).
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from apexzero.agents.ppo import PPOConfig


def log_spaced(total: float, first: float = 10e6, n: int = 16) -> list[int]:
    first = min(first, total)
    return sorted({int(x) for x in np.geomspace(first, total, n)})


def make_checkpoint_fn(out_dir: Path, cfg: PPOConfig, first: float = 10e6, n: int = 16):
    """A ``checkpoint_fn`` for ``make_train`` saving ~n log-spaced snapshots plus the last."""
    ckpt_dir = Path(out_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    steps = log_spaced(cfg.total_timesteps, first, n)
    at_update = {max(0, s // cfg.batch_size - 1): s for s in steps}
    at_update[cfg.num_updates - 1] = cfg.num_updates * cfg.batch_size

    def checkpoint_fn(snap, upd):
        upd = int(upd)
        if upd not in at_update:
            return
        snap = jax.device_get(snap)
        d = {"actor": snap["actor"], "critic": snap["critic"],
             "obs_rms": snap["obs_rms"]._asdict(), "ret_rms": snap["ret_rms"]._asdict(),
             "config": asdict(cfg), "timesteps": at_update[upd]}
        with open(ckpt_dir / f"{at_update[upd]:012d}.pkl", "wb") as fh:
            pickle.dump(d, fh)

    return checkpoint_fn


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted((Path(run_dir) / "ckpt").glob("*.pkl"))
