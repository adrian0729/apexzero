"""Archive every 2023-2026 OpenF1 qualifying and sprint-qualifying session.

    uv run python scripts/fetch_openf1.py            # resumable; re-run to continue
    uv run python scripts/fetch_openf1.py --years 2025

About 5k requests at <=29/min, so a full run takes roughly 3 hours.
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.data.openf1 import run

# The snapping audit runs on these first, so they are fetched before everything else.
PRIORITY = [(2025, "Monza"), (2025, "Suzuka"), (2025, "Silverstone")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/openf1"))
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    args = ap.parse_args()
    run(args.root, args.years, PRIORITY)
