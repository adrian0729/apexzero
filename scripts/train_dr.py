"""Domain-randomisation (E0-style) training on procedural strips, with checkpoints.

    uv run python scripts/train_dr.py --steps 200e6 --name dr-pilot

Tracks: a rotating on-device bank of generated strips (grammar fitted on R-dev only).
With --real-widths each strip's per-side widths are a wrapped segment of an R-dev
circuit's own width profile, measured from its refine_reference line as in the evaluation
below, instead of the median-width model (curvature unchanged). With --stat-widths they are
drawn from the statistical width model (GW-2's 2a, tracks/widths.py) fitted on those
profiles, which copies no circuit (with --reexpress, model 2b: fitted on the raw profiles
and re-expressed as refine_reference would); all are off by default. Only training changes:
the audit level sets (tracks/levels.py, pilot_audit.py, interp/) keep the default width
model, as they already ignore --tilt-*, so they stay fixed across runs.
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

from apexzero.agents.checkpoints import make_checkpoint_fn
from apexzero.agents.ppo import Actor, PPOConfig, make_train
from apexzero.env import racing
from apexzero.env.adapters import procedural_env, reference_speed
from apexzero.eval.laps import timed_laps
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar, tilted
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm


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
    ap.add_argument("--tilt-fraction", type=float, default=0.0,
                    help="share of strips from the tilted (hairpin-heavy) generator")
    ap.add_argument("--tilt-alpha", type=float, default=1.0,
                    help="tilted generator's corner weights are (1 / r_min)^alpha")
    ap.add_argument("--real-widths", action="store_true",
                    help="strip widths are segments of the R-dev circuits' per-side width profiles "
                         "(measured from the refine_reference line, as evaluated)")
    ap.add_argument("--real-widths-from", default="",
                    help="with --real-widths: comma-separated R-dev circuits whose width profiles "
                         "are used (default all), for leave-circuits-out tests")
    ap.add_argument("--stat-widths", action="store_true",
                    help="strip widths drawn from the statistical width model (tracks/widths.py) "
                         "fitted on the R-dev circuits' evaluated width profiles; copies none")
    ap.add_argument("--stat-widths-from", default="",
                    help="with --stat-widths: comma-separated R-dev circuits the model is fitted "
                         "on (default all), for leave-circuits-out tests")
    ap.add_argument("--reexpress", action="store_true",
                    help="with --stat-widths: model 2b, fitted on the raw centreline profiles and "
                         "re-expressed on a smoothed reference as refine_reference would")
    ap.add_argument("--ensemble-heads", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path("runs") / args.name
    out.mkdir(parents=True, exist_ok=True)
    dev = [load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV]
    width_from = [n for n in args.real_widths_from.split(",") if n]
    unknown = sorted(set(width_from) - set(R_DEV))
    if unknown:
        raise SystemExit(f"--real-widths-from: not R-dev circuits: {unknown}")
    if width_from and not args.real_widths:
        raise SystemExit("--real-widths-from needs --real-widths")
    stat_from = [n for n in args.stat_widths_from.split(",") if n]
    unknown = sorted(set(stat_from) - set(R_DEV))
    if unknown:
        raise SystemExit(f"--stat-widths-from: not R-dev circuits: {unknown}")
    if stat_from and not args.stat_widths:
        raise SystemExit("--stat-widths-from needs --stat-widths")
    if args.stat_widths and args.real_widths:
        raise SystemExit("--stat-widths and --real-widths are alternatives; choose one")
    if args.reexpress and not args.stat_widths:
        raise SystemExit("--reexpress needs --stat-widths")
    width_from = width_from or stat_from
    width_tracks = [d for n, d in zip(R_DEV, dev) if n in width_from] if width_from else None
    if args.stat_widths:
        g = fit_grammar(dev, width_tracks=width_tracks, stat_widths=True,
                        **({"reexpress": True} if args.reexpress else {}))
    else:
        g = fit_grammar(dev, real_widths=args.real_widths, width_tracks=width_tracks)
    env = procedural_env(g, bank_size=args.bank, refresh_per_update=args.refresh,
                         randomize_car=not args.fixed_car, g_alt=tilted(g, alpha=args.tilt_alpha),
                         alt_fraction=args.tilt_fraction)
    cfg = PPOConfig(num_envs=args.num_envs, num_steps=args.num_steps, total_timesteps=int(args.steps),
                    network=args.network, gamma=0.995, gae_lambda=0.95, actor_width=256,
                    critic_width=256 if args.network == "mlp" else 384, mlp_layers=3,
                    activation="tanh", ensemble_heads=args.ensemble_heads)
    (out / "config.json").write_text(json.dumps({"ppo": asdict(cfg), "args": vars(args),
                                                 "env": racing.EnvParams()._asdict()}, indent=1))
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

    train, _, _ = make_train(cfg, env, log_fn, make_checkpoint_fn(out, cfg))
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
