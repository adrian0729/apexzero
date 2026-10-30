"""GW-8's run list, the training-margin sweep (docs/findings/gw8-plan.md), for scripts/gpu_round.sh.

    python scripts/gw8_runs.py                                # the 15 runs, one JSON line each
    bash scripts/gpu_round.sh gw8 scripts/gw8_runs.py         # on the box, at gw8-v1

Three margins x seeds 90-94, every run ``scripts/gw5_train.py --steps 200e6 --seed <s> --name
<name> --real-widths --train-edge-margin <M>``: GW-5's margin arm (E0's recipe at 200M with a
training-only termination margin; evaluation stays at 1.0 m) at M = 1.2, 1.4 and 1.6 m.

The order is seed-major, the three arms rotated left by the seed index. Names are
gw8-m<M>-s<seed>. No "config" entry, as in gw5_runs.py: each run records its margin in
runs/<name>/gw5.json. Standard library only.
"""

import argparse
import json
import sys

TRAINER = "scripts/gw5_train.py"
STEPS = "200e6"
SEEDS = tuple(range(90, 95))
MARGINS = ("1.2", "1.4", "1.6")
PHASES = ("main",)


def run_name(margin: str, seed: int) -> str:
    return f"gw8-m{margin}-s{int(seed)}"


def command(margin: str, seed: int) -> list[str]:
    return [TRAINER, "--steps", STEPS, "--seed", str(int(seed)), "--name", run_name(margin, seed),
            "--real-widths", "--train-edge-margin", margin]


def runs(phase: str = "main") -> list[dict]:
    if phase not in PHASES:
        raise ValueError(f"GW-8 has one phase, 'main', not {phase!r}")
    out = []
    for i, s in enumerate(SEEDS):
        for p in range(len(MARGINS)):
            m = MARGINS[(p + i) % len(MARGINS)]
            out.append({"name": run_name(m, s), "seed": s, "cmd": command(m, s), "margin": m})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="main")
    a = ap.parse_args(argv)
    for r in runs(a.phase):
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
