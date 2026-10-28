"""E0's run list (docs/plan/e0.md, Section 1), for scripts/gpu_round.sh.

    python scripts/e0_runs.py                                       # the 5 runs, one JSON line each
    bash scripts/gpu_round.sh e0 scripts/e0_runs.py --phase main    # on the box, at e0-train-v1

Seeds 50-54, every run ``scripts/train_dr.py --steps 1e9 --seed <s> --name e0-s<s>
--real-widths`` with every other argument train_dr.py's default: ``--real-widths`` with no
``--real-widths-from`` is generator-v1 (copied R-dev width profiles, all 12), no ``--tilt-*``
flag is v1 curvature, and ``--vg-coef`` stays 0. Each run records its arguments in
runs/<name>/config.json. Standard library only (round_manifest.py's contract: no JAX, no
numpy). E0 has one phase: ``--phase main`` is accepted and changes nothing, and the box's
queue, pull-round, ``round_manifest.py check`` and scripts/e0_gate.py's manifest must all use
the same form (e0_gate.py passes ``--phase main``).
"""

import argparse
import json
import sys

TRAINER = "scripts/train_dr.py"
STEPS = "1e9"
SEEDS = tuple(range(50, 55))
PHASES = ("main",)


def run_name(seed: int) -> str:
    return f"e0-s{int(seed)}"


def command(seed: int) -> list[str]:
    """One run's cmd: the trainer, --steps, --seed and --name, then --real-widths."""
    return [TRAINER, "--steps", STEPS, "--seed", str(int(seed)), "--name", run_name(seed),
            "--real-widths"]


def runs(phase: str = "main") -> list[dict]:
    """The 5 runs in seed order, each {name, seed, cmd, config}."""
    if phase not in PHASES:
        raise ValueError(f"E0 has one phase, 'main', not {phase!r}")
    return [{"name": run_name(s), "seed": s, "cmd": command(s),
             "config": f"runs/{run_name(s)}/config.json"} for s in SEEDS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=PHASES, default="main",
                    help="E0 has one phase (accepted; every box, pull-round and check must use "
                         "the same form)")
    a = ap.parse_args(argv)
    for run in runs(a.phase):
        sys.stdout.write(json.dumps(run) + "\n")


if __name__ == "__main__":
    main()
