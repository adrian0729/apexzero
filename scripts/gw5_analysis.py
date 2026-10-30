"""GW-5, GW-6 and GW-7's analysis (docs/findings/gw5-plan.md): EXPLORATORY.

    uv run python scripts/gw5_analysis.py completion   # primary outcome and GW-6's rule
    uv run python scripts/gw5_analysis.py edges        # short-scale edge following (X-2's machinery)
    uv run python scripts/gw5_analysis.py gw6          # GW-6's 1B runs against E0's

``completion`` reads each GW-5 run's end-of-run eval_rdev.json (R-dev circuits completed,
completion > 0.5, of 12; the median gap on the circuits it completes) and metrics.csv (steps
per second), compares each lever arm with ``ctrl`` by the exact permutation test of the mean
difference (5 against 5: ``analysis.permutation_test`` with one stratum), one-sided arm > ctrl
with Holm over the three arms, and applies the plan's GW-6 rule. ``edges`` loads each network's
final checkpoint and measures, at the 12 real R-dev crash unit-phases, L/e for the sigma_b = 4
and 8 m bumps, the trapezoid G and the real change's L/e (``x2x3_exploratory.unit_phase``),
each network's value the median over its unit-phases. Added after the first analysis, on
review: the lap gap on the circuits both same-seed runs complete (``completion``) and the base
run's c_min at the unit-phases (``edges``). ``gw6`` compares GW-6's runs with E0's
five. Outputs: outputs/gw5/{completion,edges,gw6}.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import gw5_runs as R
import numpy as np

from apexzero.fields import analysis as A

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = ROOT / "outputs" / "gw5"
ARMS = R.ARMS
LEVERS = ("margin", "bumps", "simba")
TIE_ORDER = ("bumps", "margin", "simba")
MIN_GAIN, MAX_P = 1.0, 0.10            # GW-6's rule
ALPHA = 0.05
FINAL = 199_884_800
E0 = tuple(f"e0-s{s}" for s in range(50, 55))
GW6_SEEDS = tuple(range(80, 85))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def run_stats(name: str) -> dict:
    """One run's circuits completed, median gap on them and last logged steps per second."""
    ev = json.loads((RUNS / name / "eval_rdev.json").read_text())
    done = [c for c, r in ev.items() if r["completion"] > 0.5]
    gaps = [ev[c]["gap"] for c in done if ev[c].get("gap") is not None
            and np.isfinite(ev[c]["gap"])]
    with (RUNS / name / "metrics.csv").open() as f:
        last = list(csv.DictReader(f))[-1]
    return {"circuits": len(done), "completed": sorted(done),
            "gaps": {c: float(ev[c]["gap"]) for c in done if ev[c].get("gap") is not None
                     and np.isfinite(ev[c]["gap"])},
            "median_gap": float(np.median(gaps)) if gaps else float("nan"),
            "mean_completion": float(np.mean([r["completion"] for r in ev.values()])),
            "sps": float(last["sps"]), "timesteps": float(last["timesteps"]),
            "eval_sha256": sha256(RUNS / name / "eval_rdev.json")}


def compare(a: list[float], b: list[float]) -> dict:
    res = A.permutation_test({"all": (a, b)})
    return {"mean_diff": res["T"], "p_greater": res["p"], "p_less": res["p_opposite"],
            "p_two_sided": float(min(1.0, 2 * min(res["p"], res["p_opposite"]))),
            "n_relabellings": res["n_relabellings"]}


def holm(p: dict[str, float]) -> dict:
    out, alive = {}, True
    for i, k in enumerate(sorted(p, key=p.get)):
        level = ALPHA / (len(p) - i)
        alive = alive and p[k] <= level
        out[k] = {"p": p[k], "level": level, "reject": alive}
    return out


def stage_completion() -> dict:
    runs = R.runs()
    per = {r["name"]: {**run_stats(r["name"]), "arm": r["arm"], "seed": r["seed"],
                       "levers": json.loads((RUNS / r["name"] / "gw5.json").read_text())["levers"]}
           for r in runs}
    for name, v in per.items():                         # the lever each run recorded
        lv, arm = v["levers"], v["arm"]
        want = {"margin": lv["train_edge_margin"] == 1.4, "bumps": lv["local_bumps"],
                "ctrl": lv["train_edge_margin"] is None and not lv["local_bumps"],
                "simba": lv["train_edge_margin"] is None and not lv["local_bumps"]}[arm]
        if not want or v["timesteps"] < FINAL:
            raise SystemExit(f"{name}: levers {lv} or timesteps {v['timesteps']} not as planned")
    by = {a: [per[R.run_name(a, s)] for s in R.SEEDS] for a in ARMS}
    arms = {a: {"circuits": [x["circuits"] for x in by[a]],
                "mean_circuits": float(np.mean([x["circuits"] for x in by[a]])),
                "median_circuits": float(np.median([x["circuits"] for x in by[a]])),
                "mean_median_gap": float(np.nanmean([x["median_gap"] for x in by[a]])),
                "mean_sps": float(np.mean([x["sps"] for x in by[a]]))} for a in ARMS}
    ctrl = arms["ctrl"]["circuits"]
    tests = {a: compare(arms[a]["circuits"], ctrl) for a in LEVERS}
    gap_tests = {a: compare([x["median_gap"] for x in by[a]],
                            [x["median_gap"] for x in by["ctrl"]]) for a in LEVERS}
    # added after the first analysis (review): the gap on circuits both same-seed runs complete
    common = {}
    for a in LEVERS:
        per_seed, pooled = {}, []
        for s in R.SEEDS:
            ga, gc = per[R.run_name(a, s)]["gaps"], per[R.run_name("ctrl", s)]["gaps"]
            d = [ga[c] - gc[c] for c in sorted(set(ga) & set(gc))]
            per_seed[str(s)] = {"n_circuits": len(d), "mean_diff": float(np.mean(d)) if d else None}
            pooled += d
        common[a] = {"per_seed": per_seed, "n_circuit_pairs": len(pooled),
                     "pooled_mean_diff": float(np.mean(pooled)),
                     "share_positive": float(np.mean(np.asarray(pooled) > 0))}
    h = holm({a: tests[a]["p_greater"] for a in LEVERS})
    cand = min(LEVERS, key=lambda a: (-tests[a]["mean_diff"], tests[a]["p_greater"],
                                         TIE_ORDER.index(a)))
    chosen = tests[cand]["mean_diff"] >= MIN_GAIN and tests[cand]["p_greater"] <= MAX_P
    return {"runs": per, "arms": arms, "tests": tests, "holm": h, "gap_tests": gap_tests,
            "common_circuit_gap": common,
            "gw6": {"candidate": cand, "gain": tests[cand]["mean_diff"],
                    "p_greater": tests[cand]["p_greater"], "chosen": cand if chosen else None,
                    "rule": f"gain >= {MIN_GAIN} circuit and one-sided p <= {MAX_P}"}}


def stage_edges() -> dict:
    import x2x3_exploratory as X

    X.unlock()
    bank, cells, ups = X.design()
    nets, failed = {}, []
    for r in R.runs():
        name = r["name"]
        ck = max((RUNS / name / "ckpt").glob("*.pkl"))
        pin = {"path": str(ck.relative_to(ROOT)), "sha256": sha256(ck), "timesteps": FINAL}
        try:
            net = X.load_net(name, pin)
        except Exception as e:  # noqa: BLE001  (a network the fields code cannot load)
            failed.append({"name": name, "error": repr(e)})
            continue
        vals = []
        for cid, k in ups:
            u = X.unit_phase(net, bank, cells, cid, k)
            u.pop("_ctx", None)
            vals.append(u)
        ok = [u for u in vals if "pipeline" not in u]
        nets[name] = {"arm": r["arm"], "seed": r["seed"], "checkpoint_sha256": pin["sha256"],
                      "n_unit_phases": len(ok),
                      "pipeline": [(u["cell"], u["phase"], u["pipeline"]) for u in vals
                                   if "pipeline" in u],
                      "L_per_e_sigma4": X.med(u["forms"]["L_per_e"]["sigma_b=4"] for u in ok),
                      "L_per_e_sigma8": X.med(u["forms"]["L_per_e"]["sigma_b=8"] for u in ok),
                      "trapezoid_G": X.med(u["forms"]["G_loc"]["trapezoid"] for u in ok),
                      "L_per_e_real": X.med(u["L_per_e_real"] for u in ok),
                      # added after the first analysis (review): clearance at the base run
                      "base_c_min": X.med(u["base_c_min"] for u in ok)}
        print(f"{name}: {len(ok)} of {len(vals)} unit-phases", flush=True)
    out = {"networks": nets, "not_loaded": failed, "tests": {}}
    for m in ("L_per_e_sigma4", "L_per_e_sigma8", "trapezoid_G", "L_per_e_real", "base_c_min"):
        ctrl = [nets[R.run_name("ctrl", s)][m] for s in R.SEEDS if R.run_name("ctrl", s) in nets]
        out["tests"][m] = {"ctrl_median": float(np.median(ctrl))}
        for a in LEVERS:
            v = [nets[R.run_name(a, s)][m] for s in R.SEEDS if R.run_name(a, s) in nets]
            if len(v) == len(R.SEEDS) and all(np.isfinite(v)):
                out["tests"][m][a] = {"median": float(np.median(v)), **compare(v, ctrl)}
            else:
                out["tests"][m][a] = {"not_measured": f"{len(v)} of {len(R.SEEDS)} networks"}
    return out


def stage_gw6() -> dict:
    comp = json.loads((OUT / "completion.json").read_text())
    arm = comp["gw6"]["chosen"]
    if arm is None:
        return {"chosen": None, "note": "no lever chosen: E0 is the 1B pilot of the recipe"}
    gw6 = [f"gw6-{arm}-s{s}" for s in GW6_SEEDS]
    a, b = [run_stats(n) for n in gw6], [run_stats(n) for n in E0]
    ca, cb = [x["circuits"] for x in a], [x["circuits"] for x in b]
    return {"chosen": arm, "runs": dict(zip(gw6, a, strict=True)), "e0": dict(zip(E0, b, strict=True)),
            "gw6_circuits": ca, "e0_circuits": cb, "gw6_median": float(np.median(ca)),
            "e0_median": float(np.median(cb)), "reaches_gate": bool(np.median(ca) >= 10),
            "test": compare(ca, cb),
            "gap": {"gw6_mean_median_gap": float(np.nanmean([x["median_gap"] for x in a])),
                    "e0_mean_median_gap": float(np.nanmean([x["median_gap"] for x in b]))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("completion", "edges", "gw6"))
    a = ap.parse_args()
    t0 = time.time()
    res = {"completion": stage_completion, "edges": stage_edges, "gw6": stage_gw6}[a.stage]()
    res["status"] = "EXPLORATORY (docs/findings/gw5-plan.md)"
    res["provenance"] = {"script": "scripts/gw5_analysis.py",
                         "script_sha256": sha256(Path(__file__)), "git_head": git("rev-parse", "HEAD"),
                         "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.stage}.json").write_text(json.dumps(clean(res), indent=1) + "\n")
    show = {k: res[k] for k in ("arms", "tests", "holm", "gw6") if k in res} or res
    print(json.dumps(clean(show), indent=1)[:4000])


if __name__ == "__main__":
    sys.exit(main())
