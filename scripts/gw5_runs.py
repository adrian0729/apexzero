"""GW-5 and GW-7's run list (docs/findings/gw5-plan.md), for scripts/gpu_round.sh.

    python scripts/gw5_runs.py                                # the 20 runs, one JSON line each
    bash scripts/gpu_round.sh gw5 scripts/gw5_runs.py         # on the box, at gw5-v1

Four arms x seeds 70-74, every run ``scripts/gw5_train.py --steps 200e6 --seed <s> --name <name>
--real-widths <arm flags>``: E0's recipe (copied R-dev widths, v1, MLP 3 x 256, every other
argument train_dr.py's default) at 200M, with

- ``ctrl``: no lever;
- ``margin``: ``--train-edge-margin 1.4`` (GW-5; evaluation stays at 1.0);
- ``bumps``: ``--local-bumps`` (GW-5; local Gaussian width bumps in training);
- ``simba``: ``--network simba`` (GW-7).

The order is seed-major, the four arms rotated left by the seed index. Names are
gw5-<arm>-s<seed>. The runs carry no "config" entry: round_manifest.py knows only
train_dr.py's and train.py's config formats, so it checks the cmd but not the recorded
arguments (each run records its levers in runs/<name>/gw5.json). Standard library only.
"""

import argparse
import json
import sys

TRAINER = "scripts/gw5_train.py"
STEPS = "200e6"
SEEDS = tuple(range(70, 75))
ARMS = ("ctrl", "margin", "bumps", "simba")
ARM_FLAGS = {"ctrl": (), "margin": ("--train-edge-margin", "1.4"), "bumps": ("--local-bumps",),
             "simba": ("--network", "simba")}
PHASES = ("main",)


def run_name(arm: str, seed: int) -> str:
    return f"gw5-{arm}-s{int(seed)}"


def command(arm: str, seed: int) -> list[str]:
    return [TRAINER, "--steps", STEPS, "--seed", str(int(seed)), "--name", run_name(arm, seed),
            "--real-widths", *ARM_FLAGS[arm]]


def runs(phase: str = "main") -> list[dict]:
    if phase not in PHASES:
        raise ValueError(f"GW-5 has one phase, 'main', not {phase!r}")
    out = []
    for i, s in enumerate(SEEDS):
        for p in range(len(ARMS)):
            arm = ARMS[(p + i) % len(ARMS)]
            out.append({"name": run_name(arm, s), "seed": s, "cmd": command(arm, s), "arm": arm})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="main")
    a = ap.parse_args(argv)
    for r in runs(a.phase):
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
