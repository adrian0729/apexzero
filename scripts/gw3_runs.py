"""GW-3's run list (docs/prereg/gw3-width-round.md, Section 2), for scripts/gpu_round.sh.

    python scripts/gw3_runs.py                                # the 120 runs, one JSON line each
    bash scripts/gpu_round.sh gw3 scripts/gw3_runs.py         # on the box, at gw3-prereg-v1

Six arms x two variants x seeds 20-29, every run ``scripts/train_dr.py --steps 200e6 --seed <s>
--name <name> <variant flags> <arm flags>`` with every other argument train_dr.py's default:

- arms: ``base`` (no width flag); ``cl-f1`` and ``cl-f2``, ``--real-widths --real-widths-from
  <fold f>`` (copied leave-out); ``sl-f1`` and ``sl-f2``, ``--stat-widths --stat-widths-from
  <fold f>`` (model (2a) fitted on fold f; no ``--reexpress``); ``sa``, ``--stat-widths``
  (fitted on all 12);
- folds: the committed leave-out folds, f1 = R_DEV[0::2] and f2 = R_DEV[1::2], written out
  (tests/test_gw3_analysis.py checks them against R_DEV);
- variants: v1 (no flag) and v3 (``--tilt-fraction 0.5 --tilt-alpha 2``).

The order is seed-major: for seed s (seed index i = s - 20) the 12 cells of ``CELLS`` rotated
left by i, 120 runs in all. Names are gw3-<variant>-<arm>-s<seed>; each run records its
arguments in runs/<name>/config.json. Standard library only (round_manifest.py's contract: no
JAX, no numpy). GW-3 has one phase: ``--phase main`` is accepted and changes nothing.
"""

import argparse
import json
import sys

TRAINER = "scripts/train_dr.py"
STEPS = "200e6"
SEEDS = tuple(range(20, 30))
FOLDS = {"f1": ("BrandsHatch", "MoscowRaceway", "Nuerburgring", "Sepang", "YasMarina",
                "Catalunya"),
         "f2": ("Hockenheim", "Norisring", "Oschersleben", "Sochi", "Melbourne", "Zandvoort")}
VARIANT_FLAGS = {"v1": (), "v3": ("--tilt-fraction", "0.5", "--tilt-alpha", "2")}
ARM_FLAGS = {
    "base": (),
    "cl-f1": ("--real-widths", "--real-widths-from", ",".join(FOLDS["f1"])),
    "cl-f2": ("--real-widths", "--real-widths-from", ",".join(FOLDS["f2"])),
    "sl-f1": ("--stat-widths", "--stat-widths-from", ",".join(FOLDS["f1"])),
    "sl-f2": ("--stat-widths", "--stat-widths-from", ",".join(FOLDS["f2"])),
    "sa": ("--stat-widths",),
}
# Section 2's cell list; seed index i takes it rotated left by i
CELLS = ("v1-base", "v3-base", "v1-cl-f1", "v3-cl-f1", "v1-sl-f1", "v3-sl-f1",
         "v1-cl-f2", "v3-cl-f2", "v1-sl-f2", "v3-sl-f2", "v1-sa", "v3-sa")
PHASES = ("main",)


def run_name(variant: str, arm: str, seed: int) -> str:
    return f"gw3-{variant}-{arm}-s{int(seed)}"


def command(variant: str, arm: str, seed: int) -> list[str]:
    """One run's cmd: the trainer, --steps, --seed and --name, then the variant's and the
    arm's flags (Section 2's order)."""
    return [TRAINER, "--steps", STEPS, "--seed", str(int(seed)), "--name",
            run_name(variant, arm, seed), *VARIANT_FLAGS[variant], *ARM_FLAGS[arm]]


def cell_order(seed: int) -> list[str]:
    """The 12 cells in seed ``seed``'s order: CELLS rotated left by seed - 20."""
    i = SEEDS.index(int(seed))
    return [CELLS[(p + i) % len(CELLS)] for p in range(len(CELLS))]


def runs(phase: str = "main") -> list[dict]:
    """The 120 runs in the pre-registered order, each {name, seed, cmd, config, variant,
    arm} (round_manifest.py keeps and ignores the last two)."""
    if phase not in PHASES:
        raise ValueError(f"GW-3 has one phase, 'main', not {phase!r}")
    out = []
    for s in SEEDS:
        for cell in cell_order(s):
            variant, arm = cell.split("-", 1)
            name = run_name(variant, arm, s)
            out.append({"name": name, "seed": s, "cmd": command(variant, arm, s),
                        "config": f"runs/{name}/config.json", "variant": variant, "arm": arm})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=PHASES, default="main",
                    help="GW-3 has one phase (accepted; the draft's commands pass none, and "
                         "every box, pull-round and check must use the same form)")
    a = ap.parse_args(argv)
    for run in runs(a.phase):
        sys.stdout.write(json.dumps(run) + "\n")


if __name__ == "__main__":
    main()
