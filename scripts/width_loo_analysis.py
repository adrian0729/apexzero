"""Analysis for the leave-circuits-out width test (docs/findings/width-test-loo-plan.md).

    uv run python scripts/width_loo_analysis.py

Reads each run's eval_rdev.json and scores it only on the circuits whose widths the
leave-out runs of that fold never saw (the other fold). Three groups, all v3 configuration:

- baseline:  dr-pilot-v3-tilt50a2 (seeds 0-4), generated widths;
- leave-out: dr-realw-loo-f{1,2}-s{0..3}, real widths from one fold only;
- in-sample: dr-realw-v3-s{0..3}, real widths from all 12 circuits.

Primary test: leave-out vs baseline on held-out completions, one-sided exact permutation
test on the difference in means, with labels exchanged within each fold only (the folds'
held-out circuits differ in difficulty, so values are only exchangeable inside a fold).
Writes outputs/width-loo/analysis.json.
"""

import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.tracks.corners import R_DEV

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
FOLDS = {"f1": R_DEV[0::2], "f2": R_DEV[1::2]}     # widths seen by the leave-out runs
HELD_OUT = {"f1": FOLDS["f2"], "f2": FOLDS["f1"]}   # circuits each fold is judged on
BASELINE = ["dr-pilot-v3-tilt50a2"] + [f"dr-pilot-v3-tilt50a2-s{s}" for s in range(1, 5)]
IN_SAMPLE = [f"dr-realw-v3-s{s}" for s in range(4)]
LEAVE_OUT = {f: [f"dr-realw-loo-{f}-s{s}" for s in range(4)] for f in FOLDS}


def load(name: str) -> dict | None:
    p = RUNS / name / "eval_rdev.json"
    return json.loads(p.read_text()) if p.exists() else None


def completed(ev: dict, circuits: list[str]) -> list[str]:
    """Circuits completed by the standard rule (more than half the runs finish)."""
    return [c for c in circuits if ev[c]["completion"] > 0.5]


def stratified_perm_p(groups: dict[str, tuple[list[float], list[float]]]) -> tuple[float, float, int]:
    """One-sided exact permutation p for mean(treated) - mean(control) > 0.

    groups maps a stratum to (treated, control) values; labels are exchanged only within a
    stratum, keeping each stratum's group sizes. Returns (observed difference, p, number of
    relabellings). Ties count towards p.
    """
    per = []     # per stratum: the treated group's sum under every relabelling
    for treated, control in groups.values():
        pooled = treated + control
        per.append([sum(pooled[i] for i in idx)
                    for idx in itertools.combinations(range(len(pooled)), len(treated))])
    n_t = sum(len(t) for t, _ in groups.values())
    n_c = sum(len(c) for _, c in groups.values())
    total = sum(sum(t) + sum(c) for t, c in groups.values())

    def diff(sum_t: float) -> float:
        return sum_t / n_t - (total - sum_t) / n_c

    observed = diff(sum(sum(t) for t, _ in groups.values()))
    hits = count = 0
    for combo in itertools.product(*per):
        count += 1
        hits += diff(sum(combo)) >= observed - 1e-12
    return observed, hits / count, count


def main() -> int:
    missing = [n for n in BASELINE + IN_SAMPLE + LEAVE_OUT["f1"] + LEAVE_OUT["f2"] if load(n) is None]
    if missing:
        print(f"missing eval_rdev.json (excluded): {', '.join(missing)}")

    rows: dict[str, dict] = {}       # group -> fold -> list of held-out completion counts
    gaps: dict[str, list[float]] = {"baseline": [], "leave-out": [], "in-sample": []}
    per_circuit: dict[str, dict[str, list[int]]] = {c: {g: [] for g in gaps} for c in R_DEV}
    for f, held in HELD_OUT.items():
        for group, names in (("baseline", BASELINE), ("leave-out", LEAVE_OUT[f]),
                             ("in-sample", IN_SAMPLE)):
            vals = []
            for n in names:
                ev = load(n)
                if ev is None:
                    continue
                done = completed(ev, held)
                vals.append(len(done))
                gaps[group] += [ev[c]["gap"] for c in done]
                for c in held:
                    per_circuit[c][group].append(int(c in done))
            rows.setdefault(group, {})[f] = vals

    def mean(group: str) -> float:
        v = rows[group]["f1"] + rows[group]["f2"]
        return statistics.mean(v) if v else math.nan

    strata = {f: ([float(x) for x in rows["leave-out"][f]], [float(x) for x in rows["baseline"][f]])
              for f in FOLDS if rows["leave-out"][f] and rows["baseline"][f]}
    if strata:
        obs, p, n_perm = stratified_perm_p(strata)
    else:
        obs, p, n_perm = math.nan, math.nan, 0
        print("no fold has both leave-out and baseline runs: primary test not run")
    base, loo, ins = mean("baseline"), mean("leave-out"), mean("in-sample")
    retained = (loo - base) / (ins - base) if ins != base else math.nan

    print(f"held-out completions per run (of {len(HELD_OUT['f1'])}), by fold f1 | f2:")
    for g in ("baseline", "leave-out", "in-sample"):
        print(f"  {g:10s} {rows[g]['f1']} | {rows[g]['f2']}   mean {mean(g):.2f}")
    print(f"primary: leave-out - baseline = {obs:+.2f}, one-sided p = {p:.4f} "
          f"(exact, within-fold, {n_perm} relabellings)")
    print(f"leakage: in-sample - leave-out = {ins - loo:+.2f}; "
          f"share of the in-sample gain kept without seeing the widths = {retained:.2f}")
    print("per circuit (share of runs completing it when held out):")
    for c in R_DEV:
        cells = "  ".join(f"{g} {sum(v)}/{len(v)}" for g, v in per_circuit[c].items())
        print(f"  {c:14s} {cells}")
    med = {g: (statistics.median(v) if v else math.nan) for g, v in gaps.items()}
    print("median gap on completed held-out circuits: "
          + ", ".join(f"{g} {100 * m:.1f}%" for g, m in med.items()))

    out = ROOT / "outputs" / "width-loo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps({
        "folds": FOLDS, "held_out": HELD_OUT, "missing": missing, "completions": rows,
        "means": {"baseline": base, "leave-out": loo, "in-sample": ins},
        "primary": {"diff": obs, "p_one_sided": p, "relabellings": n_perm},
        "leakage": ins - loo, "gain_retained": retained,
        "per_circuit": per_circuit, "median_gap": med}, indent=1))
    print(f"wrote {out / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
