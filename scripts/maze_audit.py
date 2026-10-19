"""E5 pilot: JaxUED's own maze DR and PLR-MaxMC, briefly trained, scores audited against BFS regret.

    uv run python scripts/maze_audit.py                     # train (skips finished runs), audit, report
    uv run python scripts/maze_audit.py --stage audit       # re-audit existing checkpoints
    uv run python scripts/maze_audit.py --stage report      # tables + figures from saved audits

Training runs third_party/jaxued_examples/maze_plr.py unmodified (through run_jaxued_maze.py)
with the working directory --out (default runs/maze-audit/pilot-v2/), so its checkpoints land
in <out>/checkpoints/<run>/<seed>/ and wandb stays offline there. Both runs use the example's
defaults, including its 30,000-update linear learning-rate schedule, and differ from a default
run only in being stopped early (after --stop-after 200 updates, where the learning rate is
still >= 99% of its initial value) and in the checkpoint cadence (one every 50 updates):
- dr:        --replay_prob 0 --exploratory_grad_updates   (every update samples fresh
             generator levels and trains on them; the unused buffer still fills);
- plr-maxmc: --score_function MaxMC                       (PLR-perp at JaxUED's defaults).
The machine is shared, so every JAX process runs single-threaded XLA at nice 10, and the
script records the training runs' wall and CPU time in <out>/timing.json (each audit's own
cost is stored in its npz). Finished training runs and audited checkpoints are skipped on a
rerun; a run directory trained with another schedule is an error, not reused.

Writes <out>/audit/<run>-step<k>.npz (per-level scores, J*, J^pi, absolute and normalised
regret), <out>/audit/fidelity.json, and docs/findings/maze-audit-pilot.{md,png} plus
docs/findings/maze-audit-pilot-rho.png.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apexzero.audit.fidelity import fidelity, holm, spearman, top_k
from apexzero.maze import audit as maze

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "maze-audit" / "pilot-v2"
OUT = DEFAULT_OUT                                       # set from --out in main()
DOCS = ROOT / "docs" / "findings"
RUNS = {
    "dr": ["--replay_prob", "0", "--exploratory_grad_updates"],
    "plr-maxmc": ["--score_function", "MaxMC"],
}
# Headline scores (JaxUED's PVL / MaxMC computed by its own functions) and baselines.
MAIN = ["pvl", "maxmc", "l1", "maxmc_unit", "learnability", "mna"]
BASE = ["agent_only", "j_star", "random", "oracle_operational"]
LABEL = {
    "pvl": "PVL", "maxmc": "MaxMC", "l1": "L1 value loss", "maxmc_unit": "MaxMC-unit",
    "learnability": "Learnability p(1-p)", "mna": "MNA", "agent_only": "Agent-only -J^pi",
    "j_star": "J* (policy-agnostic)", "bfs_steps": "BFS steps", "random": "Random",
    "oracle_operational": "Oracle-operational", "oracle_regret": "Regret itself (ceiling)",
}
COLOR = {"dr": "#2a78d6", "plr-maxmc": "#eb6834"}     # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def child_cpu() -> float:
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return ru.ru_utime + ru.ru_stime


def load_timing() -> dict:
    p = OUT / "timing.json"
    return json.loads(p.read_text()) if p.exists() else {}


def record_timing(key: str, entry: dict) -> None:
    """Merge one entry into timing.json (re-read first: stages may run concurrently)."""
    t = load_timing()
    t[key] = entry
    (OUT / "timing.json").write_text(json.dumps(t, indent=1))


def run_dir(name: str, seed: int) -> Path:
    return OUT / "checkpoints" / name / str(seed)


def shown_out() -> str:
    """OUT as the docs show it: relative to the repo when it lies inside it."""
    try:
        return str(OUT.relative_to(ROOT))
    except ValueError:
        return str(OUT)


# --------------------------------------------------------------------------- train


def committed_steps(d: Path) -> int:
    """Checkpoints whose orbax save has finished (models/<k>/commit_success.txt)."""
    return len(list((d / "models").glob("*/commit_success.txt"))) if d.exists() else 0


def train(args) -> None:
    """Run maze_plr.py with its ``--num_updates`` = ``--updates`` and stop it once
    ``--stop-after`` updates are checkpointed. ``--updates`` only sets the length of the linear
    learning-rate decay (frac = 1 - optimizer_updates / num_updates, maze_plr.py
    ``linear_schedule``), so the default 30,000 gives the first 200 updates exactly the
    learning rate of a default JaxUED run (>= 99.3% of 1e-4). A short schedule would anneal it
    within the pilot, and unevenly across arms: PLR's schedule counts only the replay updates'
    optimizer steps, DR's counts every update."""
    OUT.mkdir(parents=True, exist_ok=True)
    stop = min(args.stop_after or args.updates, args.updates)
    n_ckpt = stop // args.eval_freq
    for name in args.runs:
        flags = RUNS[name]
        d = run_dir(name, args.seed)
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
            if cfg["num_updates"] != args.updates:
                raise SystemExit(f"{d} was trained with a {cfg['num_updates']}-update schedule, "
                                 f"not --updates {args.updates}; use another --out")
            if committed_steps(d) < n_ckpt:
                raise SystemExit(f"{d} holds an unfinished run ({committed_steps(d)} of {n_ckpt} "
                                 "checkpoints); maze_plr.py cannot resume it, delete it to retrain")
        if committed_steps(d) >= n_ckpt:
            print(f"{name}: {n_ckpt} checkpoints exist, skipping training", flush=True)
            continue
        # `nice` rather than a preexec_fn: this process has JAX threads, so no Python may run
        # between fork and exec.
        cmd = ["nice", "-n", "10", sys.executable, str(ROOT / "scripts" / "run_jaxued_maze.py"),
               "--run_name", name, "--seed", str(args.seed), "--num_updates", str(args.updates),
               "--eval_freq", str(args.eval_freq), "--checkpoint_save_interval", "1", *flags]
        print(" ".join(cmd[4:]), f"(stopped after {stop} updates)", flush=True)
        t0, c0 = time.time(), child_cpu()
        with open(OUT / f"train-{name}.log", "w") as log:
            p = subprocess.Popen(cmd, cwd=OUT, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, PYTHONUNBUFFERED="1"))
            while p.poll() is None and committed_steps(d) < n_ckpt:
                time.sleep(2)
            if p.poll() is None:
                p.terminate()
            rc = p.wait()
        if committed_steps(d) < n_ckpt:
            raise RuntimeError(f"{name}: training exited ({rc}) before {n_ckpt} checkpoints")
        entry = dict(wall_s=time.time() - t0, cpu_s=child_cpu() - c0,
                     schedule_updates=args.updates, updates=stop, flags=flags)
        record_timing(f"train/{name}", entry)
        print(f"{name}: {entry['wall_s']:.0f} s wall, {entry['cpu_s']:.0f} s CPU", flush=True)


# --------------------------------------------------------------------------- audit


def run_config(name: str, seed: int) -> dict:
    return json.loads((run_dir(name, seed) / "config.json").read_text())


def shared_level_config(names, seed: int) -> dict:
    """The generator / discount / time limit every audited run trained with. One level set
    and one J* serve all runs, so they must agree."""
    lcs = {name: maze.level_config(run_config(name, seed)) for name in names}
    lc = next(iter(lcs.values()))
    if any(v != lc for v in lcs.values()):
        raise SystemExit(f"the runs differ in generator, discount or time limit, so they cannot "
                         f"share audit levels and J*: {lcs}")
    return lc


def audit(args) -> None:
    os.nice(10)
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    lc = shared_level_config(args.runs, args.seed)
    levels = maze.generate_levels(args.levels, jax.random.PRNGKey(args.level_seed), lc["n_walls"])
    oracle = maze.bfs_optimum(levels, lc["gamma"], lc["max_steps"])
    print(f"{args.levels} levels ({lc}), {np.mean(oracle[0] > 0):.1%} solvable, "
          f"mean J* {oracle[0].mean():.3f}", flush=True)
    for name in args.runs:
        r = list(RUNS).index(name)          # the key depends on the run, not on --runs
        d = run_dir(name, args.seed)
        steps = [s for s in args.audit_steps if s in maze.checkpoint_steps(d)]
        for s in steps:
            if (OUT / "audit" / f"{name}-step{s}.npz").exists() and not args.force:
                print(f"{name} step {s}: audited already (--force to redo)", flush=True)
                continue
            t0, c0 = time.time(), time.process_time()
            ck = maze.load_checkpoint(d, s)
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 1000), 100 * r + s)
            res = maze.audit_checkpoint(ck, levels, key, args.episodes, args.score_episodes,
                                        oracle=oracle)
            wall, cpu = time.time() - t0, time.process_time() - c0
            np.savez_compressed(OUT / "audit" / f"{name}-step{s}.npz", **res,
                                num_updates=ck.num_updates, step=s, wall_s=wall, cpu_s=cpu)
            print(f"{name} step {s} ({ck.num_updates} updates): solve {res['success_rate'].mean():.1%}, "
                  f"regret {res['regret'].mean():.3f}, stream {int(res['stream_steps'][0])} steps, "
                  f"{wall:.0f} s wall / {cpu:.0f} s CPU", flush=True)


# --------------------------------------------------------------------------- report

# Score families. MAIN are the six headline estimators (the Holm family); NATIVE are PVL and
# MaxMC as PLR stores them at JaxUED's defaults (one 256-step rollout); BEST_OF are the
# agent-only and policy-agnostic baselines of the plan's 'uninformative' test.
NATIVE = ["pvl_native", "maxmc_native"]
BEST_OF = ("agent_only", "j_star", "bfs_steps")
TARGETS = {"abs": "regret", "norm": "normalised regret"}
LABEL.update({"pvl_native": "PVL, native 256-step", "maxmc_native": "MaxMC, native 256-step"})


def load_audits(seed_runs=RUNS) -> dict:
    out = {}
    for name in seed_runs:
        files = sorted((OUT / "audit").glob(f"{name}-step*.npz"),
                       key=lambda p: int(p.stem.split("step")[-1]))
        out[name] = [dict(np.load(f)) for f in files]
    return out


def key(run: str, a: dict, target: str = "abs") -> str:
    return f"{run}/{int(a['step'])}/{target}"


def fidelities(audits: dict, n_boot: int) -> dict:
    """rho / RC per checkpoint against absolute regret (all levels) and normalised regret (the
    levels with a reachable goal, where it is defined), each with the same level resampling
    across scores; Holm-adjusted p-values of 'differs from the best baseline' over MAIN."""
    fid = {}
    for name, rows in audits.items():
        for a in rows:
            scores = {k: a[f"s_{k}"] for k in maze.score_names(a)}
            m = a["j_star"] > 0
            for target, reg, sel in (("abs", a["regret"], slice(None)), ("norm", a["regret_norm"], m)):
                sc = {k: v[sel] for k, v in scores.items()}
                if target == "norm":        # the reference estimates its own target
                    sc["oracle_operational"] = sc["oracle_operational"] / a["j_star"][sel]
                f = fidelity(sc, reg[sel], n_boot=n_boot, seed=int(a["step"]), baselines=BEST_OF)
                for s, p in zip(MAIN, holm([f[s]["p_best"] for s in MAIN])):
                    f[s]["p_best_holm"] = float(p)
                fid[key(name, a, target)] = f
    return fid


def fmt_ci(v: dict, k: str, digits: int = 2) -> str:
    return f"{v[k]:+.{digits}f} [{v[k + '_lo']:+.{digits}f}, {v[k + '_hi']:+.{digits}f}]"


def fmt_rc(v: dict) -> str:
    return f"{v['rc']:.2f} [{v['rc_lo']:.2f}, {v['rc_hi']:.2f}]"


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def style(ax) -> None:
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def scatter_figure(audits: dict, fid: dict, path: Path) -> None:
    names = ["pvl", "maxmc", "l1", "learnability", "mna", "agent_only"]
    fig, axes = plt.subplots(len(audits), len(names), figsize=(2.6 * len(names), 2.5 * len(audits)),
                             squeeze=False)
    for r, (run, rows) in enumerate(audits.items()):
        a = rows[-1]
        f = fid[key(run, a)]
        for c, s in enumerate(names):
            ax = axes[r, c]
            ax.scatter(a[f"s_{s}"], a["regret"], s=9, color=COLOR[run], alpha=0.55, linewidths=0)
            ax.set_title(f"{LABEL[s]}\nrho {f[s]['rho']:+.2f}, RC@10% {f[s]['rc']:.2f}",
                         fontsize=8, color=INK)
            style(ax)
            if c == 0:
                ax.set_ylabel(f"{run}, {int(a['num_updates'])} updates\nregret J* - J^pi",
                              fontsize=8, color=INK)
            if r == len(audits) - 1:
                ax.set_xlabel("score (operational episodes 1-4)", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rho_figure(audits: dict, fid: dict, path: Path) -> None:
    """Small multiples: one row per regret target, one column per score."""
    names = MAIN + ["agent_only"]
    fig, axes = plt.subplots(len(TARGETS), len(names), figsize=(2.2 * len(names), 2.4 * len(TARGETS)),
                             sharey=True, squeeze=False)
    for r, (target, tlabel) in enumerate(TARGETS.items()):
        for c, s in enumerate(names):
            ax = axes[r, c]
            for run, rows in audits.items():
                x = [int(a["num_updates"]) for a in rows]
                f = [fid[key(run, a, target)][s] for a in rows]
                ax.fill_between(x, [v["rho_lo"] for v in f], [v["rho_hi"] for v in f],
                                color=COLOR[run], alpha=0.12, lw=0)
                ax.plot(x, [v["rho"] for v in f], "-o", color=COLOR[run], lw=2, ms=4, label=run)
            ax.axhline(0, color=MUTED, lw=0.8)
            if r == 0:
                ax.set_title(LABEL[s], fontsize=8, color=INK)
            if r == len(TARGETS) - 1:
                ax.set_xlabel("PPO updates", fontsize=7, color=MUTED)
            style(ax)
        axes[r, 0].set_ylabel(f"Spearman rho vs\n{tlabel}", fontsize=8, color=INK)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(cols: list, rows: list[tuple[str, list[str]]]) -> list[str]:
    head = "| | " + " | ".join(cols) + " |"
    return [head, "|---|" + "---|" * len(cols), *(f"| {r} | " + " | ".join(c) + " |" for r, c in rows)]


def report(args) -> None:
    audits = {k: v for k, v in load_audits().items() if v}
    if not audits:
        raise SystemExit("no audits found; run --stage audit first")
    t0, c0 = time.time(), time.process_time()
    fid = fidelities(audits, args.n_boot)
    (OUT / "audit" / "fidelity.json").write_text(json.dumps(fid, indent=1))
    DOCS.mkdir(parents=True, exist_ok=True)
    scatter_figure(audits, fid, DOCS / "maze-audit-pilot.png")
    rho_figure(audits, fid, DOCS / "maze-audit-pilot-rho.png")
    record_timing("report", dict(wall_s=time.time() - t0, cpu_s=time.process_time() - c0))
    timing = load_timing()

    allrows = [a for rows in audits.values() for a in rows]
    first = allrows[0]
    n_lv, k = len(first["regret"]), top_k(len(first["regret"]), 0.1)
    reach = first["j_star"] > 0
    n_reach, k_reach = int(reach.sum()), top_k(int(reach.sum()), 0.1)
    d_pvl = max(np.abs(a["s_pvl"] - a["s_pvl_ours"]).max() for a in allrows)
    d_mmc = max(np.abs(a["s_maxmc"] - a["s_maxmc_ours"]).max() for a in allrows)
    n_unsolvable = n_lv - n_reach
    reg_min = min(a["regret"][a["j_star"] > 0].min() for a in allrows)
    nat_eps = np.concatenate([a["native_episodes"] for a in allrows])
    n_neg_inf = sum(int(np.isneginf(a["s_maxmc_native"]).sum()) for a in allrows)
    train = {k_[6:]: v for k_, v in timing.items() if k_.startswith("train/")}
    train_cpu = sum(v["cpu_s"] for v in train.values())
    train_wall = sum(v["wall_s"] for v in train.values())
    aud_cpu = sum(float(a["cpu_s"]) for a in allrows)
    aud_wall = sum(float(a["wall_s"]) for a in allrows)
    total_cpu = train_cpu + aud_cpu + timing["report"]["cpu_s"]
    cfgs = {run: run_config(run, args.seed) for run in audits}
    cfg = next(iter(cfgs.values()))
    schedules = {c["num_updates"] for c in cfgs.values()}
    per_update = cfg["num_train_envs"] * cfg["num_steps"]
    stop = max(int(a["num_updates"]) for a in allrows)
    lr_end = 1 - stop / cfg["num_updates"]          # DR's LR fraction at the last checkpoint
    ck_cols = lambda rows: [f"@{int(a['num_updates'])} updates" for a in rows]
    cols = [(run, a) for run, rows in audits.items() for a in rows]
    col_names = [f"{run} @{int(a['num_updates'])}" for run, a in cols]
    default_schedule = schedules == {30000}
    max_steps = maze.level_config(cfg)["max_steps"]

    L = ["# E5 maze audit, pilot: do PVL and MaxMC track exact regret on JaxUED's own agents?", ""]
    L += [
        "JaxUED's maze example (`third_party/jaxued_examples/maze_plr.py`, vendored unmodified) was",
        f"trained twice with its own code, seed {args.seed}, "
        + ("at its defaults, including its 30,000-update linear learning-rate schedule, and stopped"
           if default_schedule else
           f"at its defaults except a {'/'.join(map(str, sorted(schedules)))}-update learning-rate "
           "schedule (a deviation), and stopped"),
        f"after {stop} updates ({stop * per_update / 1e6:.1f}M env steps; the default run is 30,000 updates,",
        f"so this is {stop / 30000:.1%} of it; the learning rate is still at least {lr_end:.1%} of its initial value):",
        "- **dr**: `--replay_prob 0 --exploratory_grad_updates`: every update trains on fresh generator levels;",
        "- **plr-maxmc**: `--score_function MaxMC`: PLR-perp with replay 0.8, rank prioritisation (beta 0.3),",
        "  staleness 0.3 and a 4,000-level buffer. It takes no gradient step on new levels, and replay starts",
        "  only once the buffer is half full (about 63 updates), so it learns from fewer updates than dr.",
        "",
        (f"Every audited checkpoint plays the same {n_lv} generator mazes (13x13, {cfg['n_walls']} walls, "
        f"{cfg['agent_view_size']}x{cfg['agent_view_size']} view;"),
        f"{n_reach} have a reachable goal) for {args.episodes} episodes each: one AutoReplay stream per maze with",
        "the stochastic policy, step for step as JaxUED's `sample_trajectories_rnn` (checked in the tests).",
        f"- **Scores** use episodes 1-{args.score_episodes}: what JaxUED's new-level branch computes on a rollout that",
        f"  ends after {args.score_episodes} episodes. PVL and MaxMC come from JaxUED's own `compute_gae` and",
        f"  `jaxued.utils` (gamma {cfg['gamma']}, lambda {cfg['gae_lambda']}, time limits terminal). L1, MaxMC-unit,",
        "  learnability p(1-p) (success = goal reached) and MNA use our primitives and the score spec's",
        "  aggregation.",
        f"- **Native** PVL and MaxMC are what PLR at JaxUED's defaults stores for a new level: one {cfg['num_steps']}-step",
        "  rollout from a reset with a fresh LSTM carry (a second, independent stream), scored on every",
        f"  episode that completes in it ({np.median(nat_eps):.0f} at the median, {nat_eps.min()} to {nat_eps.max()}).",
        (f"  No level scored -inf (JaxUED's value when no episode completes): with the {max_steps}-step time"
         f" limit every {cfg['num_steps']}-step rollout completes one." if n_neg_inf == 0 else
         f"  {n_neg_inf} level-checkpoint pairs scored -inf (no episode completed); fidelity ranks them last."),
        f"- **Targets**: J^pi is the mean discounted return of episodes {args.score_episodes + 1}-{args.episodes}, so the scores and",
        "  the target share no sampling noise. J* is the exact BFS optimum (`apexzero.maze.oracle`) with the",
        "  run's own gamma and time limit. The plan's two primary targets are absolute regret J* - J^pi (all",
        f"  {n_lv} mazes) and normalised regret (J* - J^pi) / J* (the {n_reach} mazes with J* > 0; it is undefined",
        "  on the others).",
        "- **Baselines**: agent-only (-J^pi on the scored episodes), J* and BFS path length (policy-agnostic),",
        "  random, and oracle-operational (J* - J^pi on the scored episodes, divided by J* for the normalised",
        "  target), a reference, not a ceiling.",
        "",
        "Checks on every audited checkpoint:",
        f"- our PVL and MaxMC equal JaxUED's to {max(d_pvl, d_mmc):.1e} (float32 vs float64);",
        "- no episode beat the BFS optimum, so regret >= 0 for every episode, not just on average.",
        f"  The {n_unsolvable} mazes without a reachable goal have J* = J^pi = 0; on the rest the smallest",
        f"  per-maze regret is {reg_min:.3f}.",
        "",
    ]
    for target, tlabel in TARGETS.items():
        sub = "" if target == "abs" else f", on the {n_reach} mazes with J* > 0"
        L += [f"## Spearman rho with {tlabel}{sub} [95% bootstrap CI over levels]", ""]
        for run, rows in audits.items():
            L += [f"**{run}**", ""]
            body = []
            if target == "abs":
                body += [("*solve rate*", [f"{a['success_rate'].mean():.0%}" for a in rows]),
                         ("*mean regret*", [f"{a['regret'].mean():.3f}" for a in rows])]
            else:
                body += [("*mean normalised regret*",
                          [f"{np.nanmean(a['regret_norm']):.3f}" for a in rows])]
            body += [(f"*{tlabel} held by the top 10%*",
                      [f"{fid[key(run, a, target)]['oracle_regret']['rc']:.2f}" for a in rows])]
            body += [(LABEL[s], [fmt_ci(fid[key(run, a, target)][s], "rho") for a in rows])
                     for s in MAIN + NATIVE + BASE + ["bfs_steps"]]
            L += table(ck_cols(rows), body) + [""]
    for target, tlabel in TARGETS.items():
        kk, nn = (k, n_lv) if target == "abs" else (k_reach, n_reach)
        L += [f"## RC@10% for {tlabel}: share of the total in the score's top {kk} of {nn} levels", ""]
        if target == "abs":
            for run, rows in audits.items():
                L += [f"**{run}**", ""]
                body = [(LABEL[s], [fmt_rc(fid[key(run, a)][s]) for a in rows])
                        for s in MAIN + NATIVE + BASE + ["oracle_regret"]]
                L += table(ck_cols(rows), body) + [""]
            L += ["*Regret itself (ceiling)* ranks levels by the target regret: no score can capture more.",
                  "Ties at the cut-off share the remaining slots, so a constant score captures exactly 0.10.", ""]
        else:
            body = [(LABEL[s], [f"{fid[key(run, a, target)][s]['rc']:.2f}" for run, a in cols])
                    for s in MAIN + NATIVE + ["agent_only", "j_star", "oracle_regret"]]
            L += table(col_names, body) + ["", "(Point estimates; the CIs are in fidelity.json.)", ""]

    L += ["## Against the best baseline: paired difference in rho [95% CI], Holm-adjusted p", "",
          "The plan calls a score uninformative if it is not above the best of the agent-only and",
          "policy-agnostic baselines (agent-only -J^pi, J*, BFS steps; paired CI). Each cell is",
          "rho(score) - rho(best baseline), where the best baseline is taken per bootstrap resample, and",
          "the two-sided bootstrap p-value of 'no difference', Holm-adjusted over the six headline scores",
          "of that checkpoint and target. The native scores are listed without adjustment.", ""]
    for target, tlabel in TARGETS.items():
        for run, rows in audits.items():
            L += [f"**{run}, {tlabel}**", ""]
            body = [("*best baseline (point estimate)*",
                     [LABEL[fid[key(run, a, target)]["pvl"]["best_baseline"]] for a in rows])]
            for s in MAIN + NATIVE:
                cells = []
                for a in rows:
                    v = fid[key(run, a, target)][s]
                    p = f"p_H {fmt_p(v['p_best_holm'])}" if s in MAIN else f"p {fmt_p(v['p_best'])}"
                    cells.append(f"{fmt_ci(v, 'd_best')}; {p}")
                body.append((LABEL[s], cells))
            L += table(ck_cols(rows), body) + [""]

    L += ["## Scores against regret at the last audited checkpoint", "",
          "![scatter](maze-audit-pilot.png)", "",
          "Spearman rho across checkpoints, against absolute (top) and normalised (bottom) regret",
          "(shaded: 95% bootstrap CI):", "",
          "![rho over training](maze-audit-pilot-rho.png)", "",
          "## Lambda sensitivity of PVL and L1 (rho with regret)", ""]
    lam = cfg["gae_lambda"]
    body = [({"pvl": f"PVL, lambda {lam} (headline)", "l1": f"L1, lambda {lam} (headline)"}.get(
        s, s.replace("_lam", ", lambda ").replace("pvl", "PVL").replace("l1", "L1")),
        [f"{fid[key(run, a)][s]['rho']:+.2f}" for run, a in cols])
        for s in ["pvl_lam0.9", "pvl_lam0.95", "pvl", "pvl_lam1.0", "l1_lam0.95", "l1", "l1_lam1.0"]]
    L += table(col_names, body) + [""]
    diag = {key(run, a): diagnostics(a, fid[key(run, a)], args) for run, a in cols}
    m_tgt = args.episodes - args.score_episodes
    L += ["## Diagnostics: what the scores track", ""]
    body = [(f"rho({LABEL[s]}, solve rate in episodes 1-{args.score_episodes})",
             [f"{diag[key(run, a)]['succ'][s]:+.2f}" for run, a in cols]) for s in DIAG]
    body += [(f"rho({LABEL[s]}, regret), mazes solved at least once in 1-{args.score_episodes}",
              [f"{diag[key(run, a)]['within'][s]:+.2f}" for run, a in cols]) for s in DIAG]
    body += [(f"mazes solved at least once in 1-{args.score_episodes}",
              [str(diag[key(run, a)]["n_solved"]) for run, a in cols]),
             ("median / 90th percentile regret",
              [f"{diag[key(run, a)]['q50']:.2f} / {diag[key(run, a)]['q90']:.2f}" for run, a in cols]),
             (f"episodes in the native {cfg['num_steps']}-step rollout (mean)",
              [f"{a['native_episodes'].mean():.2f}" for run, a in cols]),
             (f"reliability of a {m_tgt}-episode regret estimate",
              [f"{diag[key(run, a)]['rel']:.2f}" for run, a in cols]),
             (f"projected for {TARGET_EPISODES} episodes (Spearman-Brown)",
              [f"{diag[key(run, a)]['rel_k']:.2f}" for run, a in cols])]
    L += table(col_names, body) + [""]
    L += [f"The reliability row is the rank correlation of two independent {m_tgt}-episode estimates",
          "of the same regret (the oracle-operational row above). A score's observable rho with the",
          "target is capped near the square root of it (the plan's sqrt(reliability) ceiling).", ""]
    L += reading(audits, fid, diag, args)
    L += ["", "## Compute", "",
          f"- Training: {train_wall / 60:.1f} min wall, {train_cpu / 60:.1f} CPU-min for the two runs ("
          + ", ".join(f"{r} {v['cpu_s'] / 60:.1f}" for r, v in sorted(train.items())) + ").",
          *[f"  {r}: {v['note']}." for r, v in sorted(train.items()) if "note" in v],
          (f"- Audits: {aud_wall / 60:.1f} min wall, {aud_cpu / 60:.1f} CPU-min for {len(allrows)} "
           f"checkpoints ({n_lv} mazes x {args.episodes} episodes each, plus the native rollout)."),
          f"- Total: **{total_cpu / 60:.1f} CPU-min**. Every JAX process ran single-threaded XLA at nice 10 on a",
          "  machine shared with other jobs (load average far above the core count), so wall times are",
          "  inflated and CPU times are probably too: niced work lands on the efficiency cores.",
          "", "## Deviations from JaxUED and from the plan, and limits of this pilot", "",
          "- **Episodes per score.** The headline scores fix the scored episodes at",
          f"  {args.score_episodes} for every maze and checkpoint (spec DEV-5), so their noise does not depend on the",
          f"  agent's speed. JaxUED scores whatever completes in one {cfg['num_steps']}-step rollout; the native rows",
          "  report exactly that score, so the deviation's effect is visible in the tables.",
          "- **Training length.** "
          + (f"Each run is stopped after {stop} of JaxUED's default 30,000 updates; nothing else in"
             " the training differs from a default run (the learning rate has barely decayed)."
             if default_schedule else
             f"The learning-rate schedule is {'/'.join(map(str, sorted(schedules)))} updates instead of"
             f" 30,000, so it anneals within the pilot (DR to {lr_end:.0%} of its initial value; PLR, whose"
             " schedule counts only replay updates, less), which confounds the dr / plr comparison."),
          "- **MNA** is normalised per step (DEV-11). Its gate 1[R_max > 0] equals 'any success' here,",
          "  because the maze reward is positive exactly when the goal is reached.",
          "- **Statistics.** CIs are percentile bootstraps over levels only (one seed, so not the plan's",
          "  two-way seed x level bootstrap). Holm is applied over the six headline scores within each",
          "  checkpoint and target, not over checkpoints, runs or the two targets.",
          "- **Fresh-score audit only.** The frozen checkpoint rescores fresh generator levels. The",
          "  scores stored in PLR's buffer (the in-buffer audit) are not analysed yet, although every",
          "  checkpoint carries the buffer.",
          (f"- **Pilot scale.** One seed, {stop} updates, the {cfg['agent_view_size']}x{cfg['agent_view_size']}"
          " view only. SFL is not run: its maze code is"),
          "  not vendored. These numbers size and debug the full E5 (5 seeds x 16 checkpoints x 2,000",
          "  mazes); they are not results.",
          "",
          f"Reproduce: `uv run python scripts/maze_audit.py` (artefacts in `{shown_out()}/`).", ""]
    (DOCS / "maze-audit-pilot.md").write_text("\n".join(L))
    print(f"wrote {DOCS / 'maze-audit-pilot.md'}; total {total_cpu / 60:.1f} CPU-min")


DIAG = ["pvl", "maxmc", "l1"]
TARGET_EPISODES = 32


def diagnostics(a: dict, f: dict, args) -> dict:
    """Why a score does or does not track regret: its correlation with the scored episodes'
    solve rate, its rho within the mazes solved at least once there, and how noisy the target
    is (split-half reliability = rho of the oracle-operational reference, projected to
    TARGET_EPISODES target episodes by Spearman-Brown, which is approximate for ranks)."""
    sr, reg = a["success_rate_op"], a["regret"]
    m = sr > 0
    rel = f["oracle_operational"]["rho"]
    n = TARGET_EPISODES / (args.episodes - args.score_episodes)
    return dict(succ={s: float(spearman(a[f"s_{s}"], sr)) for s in DIAG},
                within={s: float(spearman(a[f"s_{s}"][m], reg[m])) for s in DIAG},
                n_solved=int(m.sum()), q50=float(np.median(reg)), q90=float(np.quantile(reg, 0.9)),
                succ_regret=float(spearman(sr, reg)),
                rel=rel, rel_k=n * rel / (1 + (n - 1) * rel))


def span(xs, signed: bool = True) -> str:
    xs = list(xs)
    f = "+.2f" if signed else ".2f"
    return f"{min(xs):{f}} to {max(xs):{f}}"


def names_and(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def reading(audits: dict, fid: dict, diag: dict, args) -> list[str]:
    """The interpretation. Every number comes from the tables above, and every sentence whose
    wording depends on the numbers (a sign, an ordering, 'most', a ceiling) is chosen by a
    check on them, with neutral wording otherwise, so a rerun cannot print a stale claim."""
    last = {run: rows[-1] for run, rows in audits.items()}
    lk = [key(run, a) for run, a in last.items()]
    every = list(diag)
    vb = ["pvl", "maxmc", "l1", "maxmc_unit"]
    rho_vb = [fid[k][s]["rho"] for k in lk for s in vb]
    rho_six = [fid[k][s]["rho"] for k in lk for s in MAIN]
    best_d_hi = max(fid[k][s]["d_best_hi"] for k in lk for s in MAIN)
    n_last = max(int(a["num_updates"]) for a in last.values())
    solve_last = {run: a["success_rate"].mean() for run, a in last.items()}
    solve = ", ".join(f"{run} {v:.0%}" for run, v in solve_last.items())
    m_tgt = args.episodes - args.score_episodes
    within = [diag[k]["within"][s] for k in every if diag[k]["n_solved"] >= 100 for s in DIAG]
    succ = [diag[k]["succ"][s] for k in every for s in DIAG]
    L = ["## Reading", ""]
    above = [s for s in MAIN if all(fid[k][s]["d_best_lo"] > 0 for k in lk)]
    below = [s for s in MAIN if all(fid[k][s]["d_best_hi"] < 0 for k in lk)]
    L += [("- **In this phase the value-based scores rank mazes the wrong way round.**"
           if max(rho_vb) < 0 else "- **Value-based scores at the last checkpoints.**"),
          f"  At the last checkpoint of each run ({n_last} updates; solve rate {solve}), PVL, MaxMC,",
          f"  L1 and MaxMC-unit have rho {span(rho_vb)} with regret."]
    if len(below) == len(MAIN):
        L += ["  All six headline scores are below the best baseline at both checkpoints, every paired CI",
              "  excluding 0" + (" (and every Holm-adjusted p < 0.05)." if all(
                  fid[k][s]["p_best_holm"] < 0.05 for k in lk for s in MAIN) else ".")]
    else:
        L += [f"  Above the best baseline at both (paired CI): {names_and([LABEL[s] for s in above]) if above else 'none'};",
              (f"  below it at both: {names_and([LABEL[s] for s in below]) if below else 'none'}"
              f" (largest paired upper CI {best_d_hi:+.2f}).")]
    if max(rho_six) < 0:
        order = sorted(MAIN, key=lambda s: -np.mean([fid[k][s]["rho"] for k in lk]))
        L += ["  None of the six is positively correlated with regret there; the least negative on",
              f"  average are {LABEL[order[0]]} and {LABEL[order[1]]}."]
    fails = np.mean(list(solve_last.values())) < 0.5
    sr_reg = [diag[k]["succ_regret"] for k in every]
    if min(succ) > 0.5 and max(sr_reg) < 0:
        L += ["- **Why: the scores detect success, and success is what lowers regret here.**"
              + (" The agent fails most episodes, so most regret is the cost of never reaching the goal."
                 if fails else ""),
              "  A success surprises the critic (a reward after a long run of zeros), so it produces large",
              "  positive advantages, and MaxMC's R_max stays 0 unless an episode succeeded. So PVL, MaxMC",
              f"  and L1 rise with the scored episodes' solve rate (rho {span(succ)} across checkpoints),",
              f"  while regret falls with it (rho {span(sr_reg)})."]
    else:
        L += [f"- **Scores and success.** PVL, MaxMC and L1 have rho {span(succ)} with the scored episodes'",
              f"  solve rate across checkpoints, and that solve rate has rho {span(sr_reg)} with regret."]
    if within:
        L += ["  Restricted to mazes solved at least once in the scored episodes (checkpoints with 100 or",
              f"  more such mazes), rho with regret is {span(within)}."]
    # The plan's pre-registered prediction on normalised regret (section 2.5).
    mm = [(fid[k]["maxmc"]["rho"], fid[k.replace("/abs", "/norm")]["maxmc"]["rho"]) for k in every]
    mm_nat = [(fid[k]["maxmc_native"]["rho"], fid[k.replace("/abs", "/norm")]["maxmc_native"]["rho"])
              for k in every]
    a_lo = min(x for x, _ in mm + mm_nat)
    if a_lo >= 0.3 and max(abs(y) for _, y in mm + mm_nat) <= 0.1:
        verdict = "as the plan predicts (high against absolute, near zero against normalised)."
    elif max(x for x, _ in mm + mm_nat) < 0.2:
        verdict = ("so the prediction's premise, a high rho against absolute regret, does not hold"
                   " at this stage, and the prediction cannot be tested yet.")
    else:
        verdict = "which is not the pattern the plan predicts (high against absolute, near zero against normalised)."
    L += ["- **Normalised regret (the plan's prediction for MaxMC).** Across checkpoints MaxMC has rho",
          f"  {span(x for x, _ in mm)} with absolute and {span(y for _, y in mm)} with normalised regret; native",
          f"  MaxMC {span(x for x, _ in mm_nat)} and {span(y for _, y in mm_nat)}, {verdict}"]
    nat_vs = [fid[k][f"{s}_native"]["rho"] - fid[k][s]["rho"] for k in every for s in ("pvl", "maxmc")]
    L += ["- **Native vs 4-episode scores.** JaxUED's own 256-step score differs from the fixed-4-episode",
          f"  version by {span(nat_vs)} in rho across checkpoints"
          + (", so the DEV-5 choice does not change the picture." if max(abs(x) for x in nat_vs) < 0.1
             else "; see the rho tables.")]
    ceil = [fid[k]["oracle_regret"]["rc"] for k in every]
    jq = np.quantile(next(iter(last.values()))["j_star"][next(iter(last.values()))["j_star"] > 0], [0.1, 0.9])
    if max(ceil) < 0.3:
        L += [f"- **RC@10% cannot discriminate yet.** Regret is compressed (median {span((diag[k]['q50'] for k in every), False)},",
              f"  90th percentile {span((diag[k]['q90'] for k in every), False)})"
              + (", because the agent fails most mazes" if fails else "")
              + f" and J* lies between {jq[0]:.2f} and {jq[1]:.2f} on 80% of the solvable mazes.",
              f"  Ranking by regret itself captures only {span(ceil, False)} of it in the top 10%, so the plan's",
              "  provisional 'faithful' bar (RC@10% >= 0.6) is out of reach for any score here. RC has to be",
              "  read against its ceiling, and says something only once regret concentrates on a minority of mazes."]
    else:
        L += [f"- **RC@10%.** Ranking by regret itself captures {span(ceil, False)} of it in the top 10%;",
              "  scores have to be read against that ceiling."]
    L += [f"- **Target reliability.** {m_tgt} target episodes estimate regret with reliability",
          f"  {span((diag[k]['rel'] for k in every), False)}, so even a score equal to the true regret would show a rho of",
          f"  only about {span((np.sqrt(diag[k]['rel']) for k in every), False)} (the square root) against it. With",
          f"  {TARGET_EPISODES} target episodes per maze the projected reliability is",
          f"  {span((diag[k]['rel_k'] for k in every), False)}; more mazes do not change it."]
    L += [f"- **Scope.** One seed, {n_last / 30000:.1%} of a default run"
          + (", and an agent that fails most mazes. The full E5 has to show whether the sign changes once"
             " the agent solves most mazes, when a solved maze's regret is about speed rather than success."
             if fails else ".")]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "train", "audit", "report"], default="all")
    ap.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="run artefacts (checkpoints, audits)")
    ap.add_argument("--updates", type=int, default=30000,
                    help="maze_plr --num_updates: the LR schedule's length (JaxUED's default)")
    ap.add_argument("--stop-after", type=int, default=200, help="stop training after this many updates")
    ap.add_argument("--eval-freq", type=int, default=50, help="checkpoint every this many updates")
    ap.add_argument("--audit-steps", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3],
                    help="orbax steps to audit (step k = (k + 1) * eval-freq updates)")
    ap.add_argument("--levels", type=int, default=500)
    ap.add_argument("--level-seed", type=int, default=2026)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--score-episodes", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--force", action="store_true", help="re-audit checkpoints audited already")
    args = ap.parse_args()
    global OUT
    OUT = args.out.absolute()       # not resolve(): runs/ may be a symlink out of the repo
    if args.stage in ("all", "train"):
        train(args)
    if args.stage in ("all", "audit"):
        audit(args)
    if args.stage in ("all", "report"):
        report(args)


if __name__ == "__main__":
    main()
