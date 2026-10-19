"""Timed laps on the R-dev circuits for any saved snapshot (the lockbox is never touched).

    uv run python scripts/eval_rdev.py runs/ued-plr-maxmc-pilot            # its last checkpoint
    uv run python scripts/eval_rdev.py runs/dr-pilot-200M/ckpt/000050000000.pkl

Same protocol as the end of scripts/train_dr.py: default car, 10 runs from 0.9 x the
reference start speed, gap against the s-domain OCP lap. Writes <run>/eval_rdev.json for a
run directory (refusing to overwrite one written by training) or <ckpt>.eval_rdev.json
next to a named checkpoint.
"""

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.agents.checkpoints import list_checkpoints
from apexzero.agents.ppo import Actor, PPOConfig
from apexzero.audit.rollouts import snapshot_from_pickle
from apexzero.env.adapters import reference_speed
from apexzero.eval.laps import timed_laps
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path, help="a run directory (uses its last checkpoint) or a checkpoint")
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    if args.path.is_dir():
        ckpt = list_checkpoints(args.path)[-1]
        out = args.path / "eval_rdev.json"
        if out.exists():
            raise SystemExit(f"{out} exists (written by training); evaluate a checkpoint file instead")
    else:
        ckpt, out = args.path, args.path.with_suffix(".eval_rdev.json")
    with open(ckpt, "rb") as fh:
        d = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    snap = snapshot_from_pickle(d)
    actor = Actor(cfg, 2, False)
    print(f"{ckpt} ({d['timesteps'] / 1e6:.1f}M steps)", flush=True)

    def policy(obs):
        return actor.apply(snap.actor, snap.obs_rms.normalize(obs))[0]

    results = {"checkpoint": str(ckpt), "timesteps": int(d["timesteps"])}
    for name in R_DEV:
        tr = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
        res = timed_laps(policy, tr, float(reference_speed(tr)[0]) * 0.9, runs=args.runs)
        row = {"completion": res.completion_rate, "median_lap_s": res.median_lap}
        ocp = Path("outputs/ocp") / f"{name}.pkl"
        if ocp.exists():
            with open(ocp, "rb") as fh:
                t_ocp = pickle.load(fh).lap_time
            row.update(ocp_lap_s=t_ocp, gap=res.median_lap / t_ocp - 1.0)
        results[name] = row
        print(name, json.dumps(row), flush=True)
    out.write_text(json.dumps(results, indent=1))
    done = [n for n in R_DEV if results[n]["completion"] > 0.5]
    print(f"completes {len(done)}/{len(R_DEV)}: {', '.join(done)}")


if __name__ == "__main__":
    main()
