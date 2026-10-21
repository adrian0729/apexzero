"""Post-hoc check of P5 (mirror symmetry) against the A2 layout (not pre-registered).

    uv run python scripts/p5_layout_check.py

The re-run harness gate passes (scripts/mirror_gate_rerun.py), but at A2 each member of a
mirror pair has its own start distance D, laid out from its own key, so 43 of the 68 A2 pairs
start at slightly different places (4 by 10 m or more). P5 compares brake points and speed
errors between members, so unequal starts could look like a learned asymmetry. This recomputes
P5 with the frozen analysis code (``interp.analysis.p5``) on the cached run tables of the
confirmatory inputs, three ways: all pairs (must reproduce decisions_all.json), without the
pairs that start 5 m or more apart, and on exactly mirrored pairs only.
Writes outputs/psychophysics/post/p5_layout_check.json.
"""

import datetime
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _bootstrap  # noqa: F401  (makes src/ importable)

_spec = importlib.util.spec_from_file_location("psychophysics", ROOT / "scripts" / "psychophysics.py")
psy = importlib.util.module_from_spec(_spec)
sys.modules["psychophysics"] = psy
_spec.loader.exec_module(psy)


def pair_offsets(bank, cells) -> dict[int, float]:
    """pair_id -> largest |x0_left * mirror - x0_right| over the state, for 2-member pairs."""
    from apexzero.interp import rollout

    by: dict[int, list] = {}
    for c in cells:
        if c.pair_id is not None and c.pair_id >= 0:
            by.setdefault(c.pair_id, []).append(c)
    out = {}
    for pid, cs in by.items():
        if len(cs) == 2:
            a, b = sorted(cs, key=lambda c: -bank.stimuli[c.stim_id].direction)
            xa, xb = rollout._cell_x0(a), rollout._cell_x0(b)
            out[pid] = float(np.max(np.abs(xa * rollout._MIRROR - xb)))
    return out


def main() -> int:
    from apexzero.interp import analysis as A
    from apexzero.interp import io

    bank, cells = psy._design()
    off = pair_offsets(bank, cells)
    finals = psy._finals(psy._manifest())
    kt = io.load_table(io.ANALYSIS_DIR / "key_table.npz")
    recorded = {d["agent"]: d for d in json.loads(
        (io.ANALYSIS_DIR / "decisions_all.json").read_text())["decisions"] if d["prediction"] == "P5"}
    subsets = {"all": lambda o: True, "without >=5 m": lambda o: o < 5.0,
               "exact only": lambda o: o == 0.0}
    rows = []
    for agent, e in sorted(finals.items()):
        p = io.ANALYSIS_DIR / "runs" / f"{agent}_{int(e.timesteps):012d}.npz"
        if not p.exists():
            continue
        rt = io.load_table(p)
        pair = np.asarray(rt["pair"]).astype(int)
        row = {"agent": agent, "recorded": recorded.get(agent, {}).get("details", {}).get("rule_verdict")
               or recorded.get(agent, {}).get("verdict")}
        for name, keep in subsets.items():
            mask = np.array([keep(off.get(int(q), np.inf)) if int(q) in off else name == "all"
                             for q in pair])
            d = A.p5(io.select(rt, mask), kt)
            row[name] = dict(verdict=d.verdict, **{k: d.details.get(k) for k in
                             ("median_abs_dve", "median_abs_ddb", "concordance", "n_pairs")})
        rows.append(row)
        print(f"{agent:5s} recorded {row['recorded']:8s} | " + " | ".join(
            f"{n}: {row[n]['verdict']} dve {row[n]['median_abs_dve']:.3f} "
            f"ddb {row[n]['median_abs_ddb']:.1f} conc {row[n]['concordance']:.2f} "
            f"n {row[n]['n_pairs']}" for n in subsets))
    out = ROOT / "outputs" / "psychophysics" / "post"
    out.mkdir(parents=True, exist_ok=True)
    n_off = {k: sum(1 for o in off.values() if f(o)) for k, f in subsets.items()}
    (out / "p5_layout_check.json").write_text(json.dumps(dict(
        created=datetime.datetime.now().isoformat(timespec="seconds"), pre_registered=False,
        pairs_in_design=len(off), pairs_kept=n_off, agents=rows), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
