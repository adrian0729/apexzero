"""Run JaxUED's own maze_plr.py (vendored, unmodified) with a compatibility shim.

    uv run python scripts/run_jaxued_maze.py --run_name dr-s0 --seed 0 --score_function MaxMC ...

All arguments are passed through to third_party/jaxued_examples/maze_plr.py. wandb runs in
offline mode (nothing is uploaded).
"""

import os
import runpy
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb

import apexzero.compat  # noqa: F401  (restores jax.tree_map before jaxued is imported)

# Newer wandb requires a `format` for numpy videos; the eval animations are not needed
# offline, so make them no-ops instead of editing the vendored script.
wandb.Video = lambda *args, **kwargs: None

SCRIPT = Path(__file__).resolve().parents[1] / "third_party" / "jaxued_examples" / "maze_plr.py"

if __name__ == "__main__":
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")
