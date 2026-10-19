"""Log-spaced training snapshots, shared by every training script.

A snapshot holds the actor and critic params, both normalisers and the PPO config, as
plain arrays, in ``<run>/ckpt/<timesteps:012d>.pkl``. The audit, the checkpoint
comparisons (lap times, ghosts, per-corner learning) and the demo all read this format
(``apexzero.audit.rollouts.snapshot_from_pickle``).

One naming rule for every trainer (``checkpoint_schedule``): a snapshot is taken after
the last update that does not pass a log-spaced target and is named by that target, and
the final update is named by the timesteps it ends on.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np

from apexzero.agents.ppo import PPOConfig


def log_spaced(total: float, first: float = 10e6, n: int = 16) -> list[int]:
    if first >= total:      # geomspace(t, t, n) has interior points just below t: int() -> t - 1
        return [int(total)]
    return sorted({int(x) for x in np.geomspace(first, total, n)})


def checkpoint_schedule(total_timesteps: float, batch_size: int, first: float = 10e6,
                        n: int = 16) -> dict[int, int]:
    """{update index: timesteps the snapshot is named by} for ~n log-spaced targets from
    ``first`` to ``total_timesteps``, plus the final update. Targets closer together than
    one update share a snapshot (short runs get fewer than n)."""
    num_updates = max(int(total_timesteps) // batch_size, 1)
    at_update = {max(0, s // batch_size - 1): s for s in log_spaced(total_timesteps, first, n)}
    at_update[num_updates - 1] = num_updates * batch_size
    return at_update


def save_snapshot(out_dir: Path, snap: dict, cfg: PPOConfig, timesteps: int, **extra) -> Path:
    """Write ``snap`` (actor, critic, obs_rms, ret_rms; normalisers as NamedTuples or dicts)
    to ``<out_dir>/ckpt/<timesteps:012d>.pkl``; ``extra`` keys are stored alongside."""
    snap = jax.device_get(snap)
    rms = lambda r: r if isinstance(r, dict) else r._asdict()
    d = {"actor": snap["actor"], "critic": snap["critic"], "obs_rms": rms(snap["obs_rms"]),
         "ret_rms": rms(snap["ret_rms"]), "config": asdict(cfg), "timesteps": int(timesteps), **extra}
    path = Path(out_dir) / "ckpt" / f"{int(timesteps):012d}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(d, fh)
    return path


def make_checkpoint_fn(out_dir: Path, cfg: PPOConfig, first: float = 10e6, n: int = 16):
    """A ``checkpoint_fn`` for ``make_train`` saving ~n log-spaced snapshots plus the last."""
    (Path(out_dir) / "ckpt").mkdir(parents=True, exist_ok=True)
    at_update = checkpoint_schedule(cfg.total_timesteps, cfg.batch_size, first, n)

    def checkpoint_fn(snap, upd):
        upd = int(upd)
        if upd in at_update:
            save_snapshot(out_dir, snap, cfg, at_update[upd])

    return checkpoint_fn


def list_checkpoints(run_dir: Path) -> list[Path]:
    return sorted((Path(run_dir) / "ckpt").glob("*.pkl"))
