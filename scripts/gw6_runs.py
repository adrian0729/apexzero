"""GW-6's run list (docs/findings/gw5-plan.md, "GW-6: the rule for the 1B pilot"), for
scripts/gpu_round.sh.

    python scripts/gw6_runs.py                                # the 5 runs, one JSON line each
    bash scripts/gpu_round.sh gw6 scripts/gw6_runs.py         # on the box, at gw6-v1

GW-5's rule chose the ``margin`` arm (outputs/gw5/completion.json: +2.6 circuits over ``ctrl``,
one-sided p 0.016). GW-6 is that recipe at 1B: ``scripts/gw5_train.py --steps 1e9 --seed <s>
--name gw6-margin-s<s> --real-widths --train-edge-margin 1.4``, seeds 80-84, the comparison
being E0's five 1B runs (``--steps 1e9 --real-widths``, the same recipe without the margin).
No "config" entry, as in gw5_runs.py. Standard library only.
"""

import argparse
import json
import sys

TRAINER = "scripts/gw5_train.py"
STEPS = "1e9"
SEEDS = tuple(range(80, 85))
ARM = "margin"
FLAGS = ("--real-widths", "--train-edge-margin", "1.4")
PHASES = ("main",)


def run_name(seed: int) -> str:
    return f"gw6-{ARM}-s{int(seed)}"


def runs(phase: str = "main") -> list[dict]:
    if phase not in PHASES:
        raise ValueError(f"GW-6 has one phase, 'main', not {phase!r}")
    return [{"name": run_name(s), "seed": s, "arm": ARM,
             "cmd": [TRAINER, "--steps", STEPS, "--seed", str(s), "--name", run_name(s), *FLAGS]}
            for s in SEEDS]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="main")
    a = ap.parse_args(argv)
    for r in runs(a.phase):
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
