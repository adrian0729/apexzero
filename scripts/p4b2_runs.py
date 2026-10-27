"""P4-B2's run list (docs/prereg/p4-b2-specialists.md, Section 2), for scripts/gpu_round.sh.

    python scripts/p4b2_runs.py                     # the 5 runs, one JSON line each
    GPUS=1 LANES_PER_GPU=5 bash scripts/gpu_round.sh p4b2 scripts/p4b2_runs.py   # on the box

Five fresh Hockenheim specialists, S4-S8, seeds 4-8, in that order. Every run is
``scripts/train.py --tracks Hockenheim --steps 200e6 --seed <k> --name
hockenheim-specialist-mlp-200M-s<k>`` with every other argument train.py's default, as for
S1-S3, and records its arguments in runs/<name>/config.json. The names are the draft's, so they
do not start with ``p4b2-``: round_manifest.py check notes that it cannot look for stray
directories of theirs (S1's, S2's and S3's run directories share the prefix and are not strays).
Standard library only (round_manifest.py's contract: no JAX, no numpy). P4-B2 has no options.
"""

import argparse
import json
import sys

TRAINER = "scripts/train.py"
TRACK = "Hockenheim"
STEPS = "200e6"
AGENTS = ("S4", "S5", "S6", "S7", "S8")
SEEDS = (4, 5, 6, 7, 8)                 # S1-S3 used 1-3, the excluded 50M run 0


def run_name(seed: int) -> str:
    return f"hockenheim-specialist-mlp-200M-s{int(seed)}"


def command(seed: int) -> list[str]:
    """One run's cmd: the trainer, its track, --steps, --seed and --name (Section 2)."""
    return [TRAINER, "--tracks", TRACK, "--steps", STEPS, "--seed", str(int(seed)), "--name",
            run_name(seed)]


def runs() -> list[dict]:
    """The 5 runs in the pre-registered order, each {name, seed, cmd, config, agent}
    (round_manifest.py keeps and ignores ``agent``)."""
    return [{"name": run_name(s), "seed": s, "cmd": command(s),
             "config": f"runs/{run_name(s)}/config.json", "agent": a}
            for a, s in zip(AGENTS, SEEDS, strict=True)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    for run in runs():
        sys.stdout.write(json.dumps(run) + "\n")


if __name__ == "__main__":
    main()
