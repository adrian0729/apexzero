"""VG-1's descriptive snapshots (docs/prereg/vg1-value-gradient.md, Section 5.3), run around a
defect in the tagged ``scripts/vg1_c1.py`` (a deviation, recorded in docs/findings/vg1.md).

    uv run --no-sync python scripts/vg1_snapshots.py

At ``vg1-prereg-v1``, ``vg1_c1.py snapshots`` cannot run: ``cmd_snapshots`` imports
``apexzero.costate.analysis`` before ``start()`` puts the frozen worktree's src/ on sys.path,
so the import fails on main (no ``apexzero`` there), and with the worktree's src/ on PYTHONPATH
``enter_worktree`` refuses because ``apexzero`` is already loaded. Every other subcommand
imports inside helpers that run after ``start()``.

VG-1's code stays the tag's byte for byte (its tag gate checks that, and this file is not one of
``vg1_analysis.CODE_PATHS``). This script takes ``cmd_snapshots``' own source, moves that one
import to just after ``start()``, checks that nothing else differs, and redefines the function
in ``vg1_c1``'s namespace in memory only, then runs ``vg1_c1.main(["snapshots"])``: the same
pin, freeze, import and tag checks, wiring gate and write-once output
(outputs/vg1/c1/snapshots.json). Descriptive: it decides nothing.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vg1_c1 as V

IMPORT = "    from apexzero.costate import analysis as AN\n"
ANCHOR = '    run = start("snapshots", Path(a.worktree))\n'


def fixed_source() -> str:
    src = inspect.getsource(V.cmd_snapshots)
    if src.count(IMPORT + "\n") != 1 or src.count(ANCHOR) != 1:
        raise SystemExit("cmd_snapshots is not the tagged one: refusing")
    out = src.replace(IMPORT + "\n", "", 1).replace(ANCHOR, ANCHOR + IMPORT, 1)
    if (sorted(ln for ln in out.splitlines() if ln) != sorted(ln for ln in src.splitlines() if ln)
            or out.count("\n") != src.count("\n") - 1):
        raise SystemExit("the moved import changed more than its own line: refusing")
    return out


def main() -> None:
    exec(compile(fixed_source(), V.__file__, "exec"), V.__dict__)  # noqa: S102 (the tagged source, one import moved)
    V.main(["snapshots"])


if __name__ == "__main__":
    main()
