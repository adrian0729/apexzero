"""Maze E5's run lists for scripts/gpu_round.sh (docs/prereg/maze-e5.md Sections 2 and 9).

    uv run python scripts/maze_e5_runs.py --phase main            # the 30 runs
    uv run python scripts/maze_e5_runs.py --phase timing --lanes-per-gpu 4 --gpus 4
    uv run python scripts/maze_e5_runs.py --phase timing-audit --lanes-per-gpu 4 --gpus 4
    uv run python scripts/maze_e5_runs.py --phase capture-check

Each phase is its own round, and the round names do not prefix each other, so
round_manifest.py's stray-directory check of one never flags another's runs:

- ``main`` (round ``maze-e5``): 5 seeds (1-5) x {DR, PLR-MaxMC, SFL} x views {13, 5}, 30,000
  updates each, named ``maze-e5-<arm>-v<view>-s<seed>``. The order is fixed: the SFL runs
  first, then PLR-MaxMC, then DR (the expected longest first, so that with 10 or more lanes,
  or one seed per box, no lane holds two SFL runs), within an arm seed by seed, view 13
  before 5. ``--drop k``
  applies the first k steps of the budget drop order (Section 9): 1, SFL at view 13; 2, view
  13 in every arm; 3, seed 5; 4, seed 4 (at least 3 seeds stay).
- ``timing`` (round ``maze-e5t<L>``): the timing calibration at L lanes per GPU on G GPUs:
  G x L runs of seed 99, 500 updates each (SFL through 5 refreshes), the six cells in the
  main list's order, repeated; named ``maze-e5t<L>-<j>-<arm>-v<view>``. If G x L is below the
  number of cells, whole waves are added until every cell runs (8 runs for G = 1, L = 4). Run
  it with the same LANES_PER_GPU and GPUS, so each GPU carries L runs as the main queue would.
- ``timing-audit`` (round ``maze-e5at``): one timing-only audit (no scores written) of the
  last checkpoint (update 400) of the first timing run of each cell in round
  ``maze-e5t<L>``.
- ``audit`` (round ``maze-e5a``, after the tag): one job per trained run of ``main`` (the same
  ``--drop``), named ``maze-e5a-<arm>-v<view>-s<seed>``, auditing its 16 checkpoints.
- ``capture-check`` (round ``maze-e5c``): three 200-update SFL runs (view 5, seed 99), two
  with D captured and one without, for the GPU check that the capture leaves the parameters
  bit-identical (DEV-E5-2).

``--arms`` limits the timing phases to some arms (e.g. ``dr,plr`` while SFL's network file is
not vendored). Standard library only, so the queue can run it without JAX.
"""

import argparse
import json
import sys

ARMS = ("sfl", "plr", "dr")              # the run list's order
VIEWS = (13, 5)
SEEDS = (1, 2, 3, 4, 5)
UPDATES = 30_000
TIMING_SEED = 99
TIMING_UPDATES = 500
CAPTURE_UPDATES = 200
TRAIN = "scripts/maze_e5.py"
DROP_ORDER = ("SFL at view 13", "view 13 in every arm", "seed 5", "seed 4")


def cells(arms=ARMS) -> list[tuple[str, int]]:
    return [(a, v) for a in ARMS if a in arms for v in VIEWS]


def train_cmd(name: str, arm: str, view: int, seed: int, updates: int, extra=()) -> list[str]:
    return [TRAIN, "train", "--arm", arm, "--view", str(view), "--seed", str(seed),
            "--updates", str(updates), *extra, "--name", name]


def dropped(arm: str, view: int, seed: int, drop: int) -> bool:
    return ((drop >= 1 and arm == "sfl" and view == 13) or (drop >= 2 and view == 13)
            or (drop >= 3 and seed == 5) or (drop >= 4 and seed == 4))


def main_runs(drop: int = 0) -> list[dict]:
    if not 0 <= drop <= len(DROP_ORDER):
        raise ValueError(f"--drop must be 0..{len(DROP_ORDER)}")
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            for view in VIEWS:
                if dropped(arm, view, seed, drop):
                    continue
                name = f"maze-e5-{arm}-v{view}-s{seed}"
                out.append({"name": name, "seed": seed, "arm": arm, "view": view,
                            "updates": UPDATES,
                            "cmd": train_cmd(name, arm, view, seed, UPDATES)})
    return out


def timing_runs(lanes_per_gpu: int, gpus: int, arms=ARMS) -> list[dict]:
    if lanes_per_gpu < 1 or gpus < 1:
        raise ValueError("--lanes-per-gpu and --gpus must be positive")
    cs = cells(arms)
    if not cs:
        raise ValueError("no arms")
    wave = gpus * lanes_per_gpu
    out = []
    for j in range(wave * -(-len(cs) // wave)):        # whole waves, every cell at least once
        arm, view = cs[j % len(cs)]
        name = f"maze-e5t{lanes_per_gpu}-{j:02d}-{arm}-v{view}"
        out.append({"name": name, "seed": TIMING_SEED, "arm": arm, "view": view,
                    "updates": TIMING_UPDATES,
                    "cmd": train_cmd(name, arm, view, TIMING_SEED, TIMING_UPDATES)})
    return out


def timing_audit_runs(lanes_per_gpu: int, gpus: int, arms=ARMS) -> list[dict]:
    out = []
    seen = set()
    for run in timing_runs(lanes_per_gpu, gpus, arms):
        cell = (run["arm"], run["view"])
        if cell in seen:
            continue
        seen.add(cell)
        name = f"maze-e5at-{run['arm']}-v{run['view']}"
        out.append({"name": name, "seed": TIMING_SEED, "arm": run["arm"], "view": run["view"],
                    "cmd": [TRAIN, "audit", "--timing-only", "--source", run["name"],
                            "--update", "400", "--name", name]})
    return out


def audit_runs(drop: int = 0) -> list[dict]:
    out = []
    for run in main_runs(drop):
        name = "maze-e5a-" + run["name"].removeprefix("maze-e5-")
        out.append({"name": name, "seed": run["seed"], "arm": run["arm"], "view": run["view"],
                    "source": run["name"],
                    "cmd": [TRAIN, "audit", "--run", run["name"], "--name", name]})
    return out


def capture_runs() -> list[dict]:
    out = []
    for tag, extra in (("on-a", ()), ("on-b", ()), ("off", ("--no-capture",))):
        name = f"maze-e5c-sfl-v5-{tag}"
        out.append({"name": name, "seed": TIMING_SEED, "arm": "sfl", "view": 5,
                    "updates": CAPTURE_UPDATES,
                    "cmd": train_cmd(name, "sfl", 5, TIMING_SEED, CAPTURE_UPDATES,
                                     (*extra, "--save", str(CAPTURE_UPDATES)))})
    return out


def runs(phase: str = "main", drop: int = 0, lanes_per_gpu: int = 4, gpus: int = 4,
         arms=ARMS) -> list[dict]:
    if phase == "main":
        return main_runs(drop)
    if phase == "timing":
        return timing_runs(lanes_per_gpu, gpus, arms)
    if phase == "timing-audit":
        return timing_audit_runs(lanes_per_gpu, gpus, arms)
    if phase == "audit":
        return audit_runs(drop)
    if phase == "capture-check":
        return capture_runs()
    raise ValueError(f"unknown phase {phase!r}")


def round_name(phase: str, lanes_per_gpu: int = 4) -> str:
    return {"main": "maze-e5", "timing": f"maze-e5t{lanes_per_gpu}", "timing-audit": "maze-e5at",
            "audit": "maze-e5a", "capture-check": "maze-e5c"}[phase]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", choices=["main", "timing", "timing-audit", "audit",
                                        "capture-check"], default="main")
    ap.add_argument("--drop", type=int, default=0, help="budget drop steps applied (0-4)")
    ap.add_argument("--lanes-per-gpu", type=int, default=4)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS), help="timing phases only")
    a = ap.parse_args(argv)
    arms = tuple(x for x in a.arms.split(",") if x)
    if not set(arms) <= set(ARMS) or (not a.phase.startswith("timing")
                                        and set(arms) != set(ARMS)):
        raise SystemExit(f"--arms must be a subset of {','.join(ARMS)} (timing phases only)")
    for run in runs(a.phase, a.drop, a.lanes_per_gpu, a.gpus, arms):
        print(json.dumps(run))


if __name__ == "__main__":
    sys.exit(main())
