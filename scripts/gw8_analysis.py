"""GW-8's analysis, the training-margin sweep (docs/findings/gw8-plan.md): EXPLORATORY.

    uv run python scripts/gw8_analysis.py completion   # primary outcome, lap gap, the rule
    uv run python scripts/gw8_analysis.py edges        # clearance and edge following (X-2's machinery)

``completion`` reads each run's end-of-run eval_rdev.json and metrics.csv (gw5_analysis.run_stats),
checks that it recorded its margin and no bumps and reached 200M, and compares the 1.2 and 1.6 m
arms with 1.4 m by the exact permutation test of the mean difference (5 against 5), two-sided:
circuits completed, the runs' median gaps, and the gap on the circuits both same-seed runs
complete. It then applies the plan's recommendation rule. ``edges`` measures each network at the
12 real R-dev crash unit-phases as gw5_analysis's ``edges`` stage does (base c_min, L/e for the
sigma_b = 4 and 8 m bumps, trapezoid G, the real change's L/e). Written while the round trained,
before any of its runs came back. Outputs: outputs/gw8/{completion,edges}.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gw5_analysis as G
import gw8_runs as R
import numpy as np

ROOT = G.ROOT
RUNS = G.RUNS
OUT = ROOT / "outputs" / "gw8"
REF = "1.4"
OTHERS = ("1.2", "1.6")
FINAL = G.FINAL                         # 200e6 rounded down to whole updates
MORE_FOR_WIDER = 1.0                    # the rule: 1.6 m if it completes this many more
LESS_FOR_NARROWER = -0.4                # 1.2 m if it completes at most this many fewer ...
GAP_FOR_NARROWER = -0.003               # ... and its same-seed gap is this much lower (0.3 points)
MEASURES = ("base_c_min", "L_per_e_sigma4", "L_per_e_sigma8", "trapezoid_G", "L_per_e_real")


def stage_completion() -> dict:
    per = {}
    for r in R.runs():
        name, m = r["name"], r["margin"]
        lv = json.loads((RUNS / name / "gw5.json").read_text())["levers"]
        st = G.run_stats(name)
        if lv["train_edge_margin"] != float(m) or lv["local_bumps"] or st["timesteps"] < FINAL:
            raise SystemExit(f"{name}: levers {lv} or timesteps {st['timesteps']} not as planned")
        per[name] = {**st, "margin": m, "seed": r["seed"]}
    by = {m: [per[R.run_name(m, s)] for s in R.SEEDS] for m in R.MARGINS}
    arms = {m: {"circuits": [x["circuits"] for x in by[m]],
                "mean_circuits": float(np.mean([x["circuits"] for x in by[m]])),
                "median_circuits": float(np.median([x["circuits"] for x in by[m]])),
                "median_gaps": [x["median_gap"] for x in by[m]],
                "mean_median_gap": float(np.nanmean([x["median_gap"] for x in by[m]])),
                "mean_sps": float(np.mean([x["sps"] for x in by[m]]))} for m in R.MARGINS}
    tests = {m: G.compare(arms[m]["circuits"], arms[REF]["circuits"]) for m in OTHERS}
    gap_tests = {m: G.compare(arms[m]["median_gaps"], arms[REF]["median_gaps"]) for m in OTHERS}
    common = {}
    for m in OTHERS:
        per_seed, pooled = {}, []
        for s in R.SEEDS:
            ga, gr = per[R.run_name(m, s)]["gaps"], per[R.run_name(REF, s)]["gaps"]
            d = [ga[c] - gr[c] for c in sorted(set(ga) & set(gr))]
            per_seed[str(s)] = {"n_circuits": len(d), "mean_diff": float(np.mean(d)) if d else None}
            pooled += d
        common[m] = {"per_seed": per_seed, "n_circuit_pairs": len(pooled),
                     "pooled_mean_diff": float(np.mean(pooled)),
                     "share_slower": float(np.mean(np.asarray(pooled) > 0))}
    d16 = round(arms["1.6"]["mean_circuits"] - arms[REF]["mean_circuits"], 9)
    d12 = round(arms["1.2"]["mean_circuits"] - arms[REF]["mean_circuits"], 9)
    g12 = round(common["1.2"]["pooled_mean_diff"], 9)
    if d16 >= MORE_FOR_WIDER:
        pick = "1.6"
    elif d12 >= LESS_FOR_NARROWER and g12 <= GAP_FOR_NARROWER:
        pick = "1.2"
    else:
        pick = REF
    gw5 = json.loads((G.OUT / "completion.json").read_text())["arms"]
    return {"runs": per, "arms": arms, "tests": tests, "gap_tests": gap_tests,
            "common_circuit_gap": common,
            "rule": {"recommended": pick, "d_circuits_1.6": d16, "d_circuits_1.2": d12,
                     "common_gap_1.2": g12,
                     "text": (f"1.6 if its mean circuits is >= {MORE_FOR_WIDER} above 1.4's; else "
                              f"1.2 if its mean is >= {LESS_FOR_NARROWER} relative to 1.4's and its "
                              f"same-seed gap difference <= {GAP_FOR_NARROWER}; else 1.4")},
            "reference_gw5": {a: {k: gw5[a][k] for k in ("mean_circuits", "mean_median_gap")}
                              for a in ("ctrl", "margin")}}


def stage_edges() -> dict:
    import x2x3_exploratory as X

    X.unlock()
    bank, cells, ups = X.design()
    nets, failed = {}, []
    for r in R.runs():
        name = r["name"]
        ck = max((RUNS / name / "ckpt").glob("*.pkl"))
        pin = {"path": str(ck.relative_to(ROOT)), "sha256": G.sha256(ck), "timesteps": FINAL}
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
        nets[name] = {"margin": r["margin"], "seed": r["seed"], "checkpoint_sha256": pin["sha256"],
                      "n_unit_phases": len(ok),
                      "pipeline": [(u["cell"], u["phase"], u["pipeline"]) for u in vals
                                   if "pipeline" in u],
                      "base_c_min": X.med(u["base_c_min"] for u in ok),
                      "L_per_e_sigma4": X.med(u["forms"]["L_per_e"]["sigma_b=4"] for u in ok),
                      "L_per_e_sigma8": X.med(u["forms"]["L_per_e"]["sigma_b=8"] for u in ok),
                      "trapezoid_G": X.med(u["forms"]["G_loc"]["trapezoid"] for u in ok),
                      "L_per_e_real": X.med(u["L_per_e_real"] for u in ok)}
        print(f"{name}: {len(ok)} of {len(vals)} unit-phases", flush=True)
    out = {"networks": nets, "not_loaded": failed, "tests": {}}
    for k in MEASURES:
        v = {m: [nets[R.run_name(m, s)][k] for s in R.SEEDS if R.run_name(m, s) in nets]
             for m in R.MARGINS}
        out["tests"][k] = {m: {"median": float(np.median(v[m])) if v[m] else None, "n": len(v[m])}
                           for m in R.MARGINS}
        for m in OTHERS:
            if len(v[m]) == len(v[REF]) == len(R.SEEDS) and np.all(np.isfinite(v[m] + v[REF])):
                out["tests"][k][m].update(G.compare(v[m], v[REF]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("completion", "edges"))
    a = ap.parse_args()
    t0 = time.time()
    res = {"completion": stage_completion, "edges": stage_edges}[a.stage]()
    res["status"] = "EXPLORATORY (docs/findings/gw8-plan.md)"
    res["provenance"] = {"script": "scripts/gw8_analysis.py",
                         "script_sha256": G.sha256(Path(__file__)),
                         "git_head": G.git("rev-parse", "HEAD"),
                         "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.stage}.json").write_text(json.dumps(G.clean(res), indent=1) + "\n")
    show = {k: res[k] for k in ("arms", "tests", "rule") if k in res} or res
    print(json.dumps(G.clean(show), indent=1)[:4000])


if __name__ == "__main__":
    sys.exit(main())
