"""VG-1's run lists (docs/prereg/vg1-value-gradient.md, Section 4), for scripts/gpu_round.sh.

    uv run python scripts/vg1_runs.py --phase pilot     # round vg1p: 6 runs, before the tag
    uv run python scripts/vg1_runs.py --phase main      # round vg1: 40 runs, at vg1-prereg-v1

Run as a script it prints one JSON object per line, in the pre-registered order (the run-list
contract of scripts/round_manifest.py). It uses the standard library only.

**Pilot** (Section 4.1; round ``vg1p``): beta in {0.1, 0.3, 1.0} x seeds 15 and 16, named
``vg1p-b<beta>-s<seed>``, each ``scripts/train_dr.py --steps 200e6 --seed <seed> --name <name>
--vg-coef <beta> --skip-eval-rdev``. Seed 15's three runs come first, then seed 16's, each in
the grid's order: at ``GPUS=1 LANES_PER_GPU=3`` each lane is one beta, seed 15 in the first
round and seed 16 in the second; at ``GPUS=2 LANES_PER_GPU=3`` all six run at once, each GPU
one run of each beta. Every pilot run passes --skip-eval-rdev (``runs`` refuses a list without
it), so no R-dev completion exists before the tag.

**Main** (Section 4.2; round ``vg1``): 2 arms x seeds 30-49 = 40 runs, ``vg1-base-s<seed>``
(no VG flag: the default path) and ``vg1-vgl-s<seed>`` (``--vg-coef beta*``). beta* is read
from the pilot record outputs/vg1/pilot.json, which must be tracked and unchanged against
HEAD, and must name an eligible beta of the grid; with none eligible the main run does not
start. The order goes by blocks of 4 seeds: the block's 4 base runs, then its 4 vgl runs
(30-33 base, 30-33 vgl, 34-37 base, ...). gpu_round.sh deals job j to lane j mod L and lane
l to GPU l mod G. On one 4-GPU box at 5 lanes per GPU (the plan, if the draft's memory check
passes) lane l runs jobs l and l + 20, one base and one vgl run, so the lanes end together; at
4 lanes per GPU GPU g runs both arms of seeds 30 + g, 34 + g, ... on 2 base lanes and 2 vgl
lanes; SHARD=i/K on K single-GPU boxes
(whole seeds) gives each box the same 2 + 2 lanes when K is 2, 4, 5 or 10, so that every box
takes the same number of seeds from each block (another K keeps both runs of a seed on one
GPU, but its lanes mix the arms). No arm has a GPU, lane set or box of its own, and the two
runs of a seed always share a GPU.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = {"pilot": "vg1p", "main": "vg1"}
TRAINER = "scripts/train_dr.py"
STEPS = "200e6"
PILOT_BETAS = ("0.1", "0.3", "1.0")       # Section 4.1's grid, as the names and flags spell it
PILOT_SEEDS = (15, 16)
MAIN_SEEDS = tuple(range(30, 50))
BLOCK = 4                                 # seeds per block of the main run's order
ARMS = ("base", "vgl")
SKIP = "--skip-eval-rdev"
PILOT_RECORD = "outputs/vg1/pilot.json"


def _run(name: str, seed: int, extra: list[str], **keys) -> dict:
    cmd = [TRAINER, "--steps", STEPS, "--seed", str(seed), "--name", name, *extra]
    return {"name": name, "seed": seed, "cmd": cmd, "config": f"runs/{name}/config.json", **keys}


def pilot_runs() -> list[dict]:
    """Section 4.1's 6 runs, seed 15's then seed 16's, each in the grid's order."""
    out = [_run(f"vg1p-b{b}-s{s}", s, ["--vg-coef", b, SKIP], arm="pilot", beta=float(b))
           for s in PILOT_SEEDS for b in PILOT_BETAS]
    bad = [r["name"] for r in out if r["cmd"].count(SKIP) != 1]
    if bad:
        raise SystemExit(f"refusing the pilot: {bad} do not pass {SKIP} (Section 4.1)")
    return out


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def beta_star(root: Path = ROOT, record: str = PILOT_RECORD) -> str:
    """beta* from the committed pilot record, spelt as the grid spells it (Section 4.2)."""
    path = Path(root) / record
    if not path.is_file():
        raise SystemExit(f"{record} not found: the pilot chooses beta* first (Section 4.1)")
    if _git(root, "ls-files", "--error-unmatch", "--", record).returncode:
        raise SystemExit(f"{record} is not committed (force-add it, Section 9 step 3)")
    if _git(root, "diff", "--quiet", "HEAD", "--", record).returncode:
        raise SystemExit(f"{record} differs from HEAD: commit it first")
    rec = json.loads(path.read_text())
    if rec.get("record") != "vg1-pilot":
        raise SystemExit(f"{record} is not VG-1's pilot record")
    b = rec.get("beta_star")
    if b is None:
        raise SystemExit(f"{record}: no beta is eligible, so the main run does not start "
                         "(Section 4.1)")
    match = [g for g in PILOT_BETAS if float(g) == b and not isinstance(b, bool)]
    if not match:
        raise SystemExit(f"{record}: beta* {b!r} is not in the grid {PILOT_BETAS}")
    if not (rec.get("eligible") or {}).get(match[0]):
        raise SystemExit(f"{record}: beta* {match[0]} is not marked eligible")
    return match[0]


def main_runs(root: Path = ROOT, record: str = PILOT_RECORD) -> list[dict]:
    """Section 4.2's 40 runs in blocks of 4 seeds: the block's base runs, then its vgl runs."""
    b = beta_star(root, record)
    out = []
    for i in range(0, len(MAIN_SEEDS), BLOCK):
        block = MAIN_SEEDS[i:i + BLOCK]
        out += [_run(f"vg1-base-s{s}", s, [], arm="base", beta=None) for s in block]
        out += [_run(f"vg1-vgl-s{s}", s, ["--vg-coef", b], arm="vgl", beta=float(b))
                for s in block]
    return out


def runs(phase: str = "pilot", root: Path = ROOT, record: str = PILOT_RECORD) -> list[dict]:
    if phase == "pilot":
        return pilot_runs()
    if phase == "main":
        return main_runs(root, record)
    raise SystemExit(f"--phase is pilot or main, not {phase!r}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", choices=sorted(ROUNDS), required=True)
    a = ap.parse_args(argv)
    for run in runs(a.phase):
        print(json.dumps(run))


if __name__ == "__main__":
    main(sys.argv[1:])
