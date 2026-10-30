"""X-1, locality of edge tracking on H4R's 20 same-code networks: EXPLORATORY
(docs/findings/locality-h4r-plan.md, committed in 3e71926 before anything here was computed).
Every number this script writes is exploratory.

    uv run python scripts/locality_h4r_exploratory.py

``locality.md``'s 24 profiles and three forms (``scripts/locality_exploratory.py``'s own
functions, imported read-only) on H4R's stored per-key impulse responses, with H4R's keys and
exclusions (``h4r.h4r_key_measures``: step 2's primary non-capped A2 keys plus H4R's
gate2-trained exclusion). Real-width against baseline, per profile and form, on the networks'
pooled medians: H4R's frozen permutation test (``analysis.permutation_test``, stratified by
variant, 5 against 5: 63,504 relabellings) and its paired sign-flip test
(``h4r.sign_flip_test``, 10 same-seed pairs), each two-sided as twice the smaller one-sided p.
The primary outcome is L/e for the sigma_b = 4 and 8 m bumps at s* (Holm at 0.05). No
network, closed loop, gradient or solver is run.

**Checks** (the script stops and writes nothing if one fails): the arrays against their rows'
sha256; the trapezoid's G_loc against each kept key's G and each network's pooled summary
against H4R's decisions.json (1e-9); the trapezoid's permutation T and p against H4R-G's;
``locality.md``'s s* recovery checks; the frozen code unchanged from HEAD.

**Outputs**: outputs/locality-h4r/results.json (with a provenance block) and results.csv (one
row per network, profile and form).
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
import h4r as H  # H4R's frozen key rule, names and paired test, read-only
import locality_exploratory as LX  # locality.md's profiles, forms and summaries, read-only
import numpy as np

from apexzero.fields import analysis as A
from apexzero.fields import measures as MS
from apexzero.fields import stages as S

ROOT = LX.ROOT
H4R_OUT = ROOT / "outputs" / "h4r"
OPT_ROWS = ROOT / "outputs" / "fields" / "opt-fields" / "rows.jsonl"
PREREG = ROOT / "scripts" / "h4r_prereg.json"
DECISIONS = H4R_OUT / "analyse" / "decisions.json"
DECISIONS_SHA256 = "f3bcbed6292d19c58f5c6f6c972526bc8bb317540d7314d9f8506bd79207eec9"
OUT = ROOT / "outputs" / "locality-h4r"
PLAN = ROOT / "docs" / "findings" / "locality-h4r-plan.md"
PLAN_COMMIT = "3e71926"
FROZEN = ("src", "scripts/fields.py", "scripts/h4r.py", "scripts/locality_exploratory.py",
          "scripts/_bootstrap.py")
TOL = 1e-9
PRIMARY = ("sigma_b=4", "sigma_b=8")
PRIMARY_FORM = "L_per_e"
ALPHA = 0.05


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise LX.CheckFailed(1)


def two_sided(p: float, q: float) -> float:
    return float(min(1.0, 2.0 * min(p, q)))


def compare(stat: dict[str, float]) -> dict:
    """Real-width against baseline on per-network values: the stratified permutation test and
    the paired sign-flip test (H4R's), both two-sided."""
    g = {v: ([stat[H.run_name(v, H.REAL, s)] for s in H.SEEDS],
             [stat[H.run_name(v, H.BASE, s)] for s in H.SEEDS]) for v in H.VARIANTS}
    res = A.permutation_test(g)
    diffs = [stat[H.run_name(v, H.REAL, s)] - stat[H.run_name(v, H.BASE, s)]
             for v in H.VARIANTS for s in H.SEEDS]
    sf = H.sign_flip_test(diffs, -1)
    return {"mean_diff": res["T"] / len(g), "T": res["T"],
            "per_variant": {v: x["T"] for v, x in res["strata"].items()},
            "p_two_sided": two_sided(res["p"], res["p_opposite"]),
            "p_real_gt_base": res["p"], "p_real_lt_base": res["p_opposite"],
            "n_relabellings": res["n_relabellings"],
            "paired": {"mean_diff": sf["mean_diff"], "n_pairs": sf["n_pairs"],
                       "p_two_sided": two_sided(sf["p"], sf["p_opposite"]),
                       "real_lower_pairs": int(sum(d < 0 for d in diffs))},
            "real_range": [min(x for r, _ in g.values() for x in r),
                           max(x for r, _ in g.values() for x in r)],
            "baseline_range": [min(x for _, b in g.values() for x in b),
                               max(x for _, b in g.values() for x in b)]}


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    out, order = {}, sorted(pvals, key=pvals.get)
    still = True
    for i, k in enumerate(order):
        level = alpha / (len(order) - i)
        still = still and pvals[k] <= level
        out[k] = {"p": pvals[k], "level": level, "reject": still}
    return out


def monotone(res: dict, labs: list[str]) -> dict:
    """Networks whose G_loc median rises with every step of the family's scale."""
    ok = {n: bool(np.all(np.diff([res[n][x]["G_loc"]["median"] for x in labs]) > 0))
          for n in res}
    return {"members": labs, "n_monotone": int(sum(ok.values())), "n": len(ok),
            "not_monotone": sorted(n for n, v in ok.items() if not v)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=MS.BOOTSTRAP_N)
    args = ap.parse_args(argv)
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%S")
    changed = git("status", "--porcelain", "--", *FROZEN)
    if changed:
        fail(f"frozen code differs from HEAD: {changed.splitlines()}")
    raw_dec = DECISIONS.read_bytes()
    if hashlib.sha256(raw_dec).hexdigest() != DECISIONS_SHA256:
        fail("H4R's decisions.json is not the committed confirmatory file")
    dec = json.loads(raw_dec)
    rec = json.loads(PREREG.read_text())
    a2 = {k for _, k in rec["keys"]["a2"]["ids"]}
    opt = {r["key_id"]: r for r in S.read_rows(OPT_ROWS) if r["key_id"] in a2}
    if a2 - set(opt):
        fail(f"{len(a2 - set(opt))} A2 keys without an optimum row")
    g2 = json.loads((H4R_OUT / "gate2-trained" / "gate2_trained.json").read_text())
    failing = {n: H.gate_verdict(v)["failing_keys"] for n, v in g2["networks"].items()}
    inputs = {LX.rel(p): LX.sha256_file(p) for p in
              (OPT_ROWS, PREREG, DECISIONS, H4R_OUT / "gate2-trained" / "gate2_trained.json")}

    # ---- per network: H4R's keys, the per-key pass, the summaries and checks 1-3, 5
    res, checks, kept_n = {}, {}, {}
    dec_max = 0.0
    for name in H.NETWORKS:
        p = H4R_OUT / "fields" / f"{H._file(name)}.jsonl"
        rows = S.read_rows(p)
        inputs[LX.rel(p)] = LX.sha256_file(p)
        km = H.h4r_key_measures(rows, opt, "G", failing.get(name, ()))
        kept = {k.key_id for k in km if not k.reason}
        vals, c = LX.network_pass(name, rows, S.arrays_dir(H4R_OUT / "fields" / "arrays",
                                                           rows[0]["net"]), opt, kept)
        if c["sha_mismatch"]:
            fail(f"{name}: arrays differ from their rows' sha256: {c['sha_mismatch'][:3]}")
        if kept - set(vals):
            fail(f"{name}: kept keys without values {sorted(kept - set(vals))[:3]}")
        for k in km:
            if not k.reason and abs(vals[k.key_id]["G_loc"][LX.TRAP] - k.value) > TOL:
                fail(f"{name} {k.key_id}: G_loc(trapezoid) differs from the key's G")
        if c["trap_max_abs"] > TOL:
            fail(f"{name}: G_loc(trapezoid) differs from the rows' G by {c['trap_max_abs']:.3e}")
        if max(c["basis_sum_max_abs"], c["e_star_max_abs"], c["other_features_max_abs"]) > TOL:
            fail(f"{name}: s* recovery check {c}")
        res[name] = LX.summaries(km, vals, args.n_boot)
        if args.n_boot == MS.BOOTSTRAP_N:
            ref, got = dec["networks"][name]["G"]["pooled"], res[name]["trapezoid"]["G_loc"]
            d = max(abs(got[x] - ref[x]) for x in ("median", "lo", "hi"))
            d = max(d, abs(got["median"] - dec["statistics"]["G"][name]))
            if d > TOL:
                fail(f"{name}: trapezoid summary differs from H4R's G by {d:.3e}")
            dec_max = max(dec_max, d)
        checks[name] = c
        kept_n[name] = {"kept": len(kept), "keys": len(km),
                        "gate2_excluded": sum(k.reason == H.GATE2_REASON for k in km)}
        print(f"{name}: {len(kept)} kept of {len(km)}; {time.time() - t0:.0f} s", flush=True)

    # ---- comparisons, check 4, the primary outcome and the readings
    comp = {}
    for lab in LX.LABELS:
        comp[lab] = {}
        for form in LX.FORMS:
            if form in res[H.NETWORKS[0]][lab]:
                comp[lab][form] = compare({n: res[n][lab][form]["median"] for n in H.NETWORKS})
    trap, ref = comp["trapezoid"]["G_loc"], dec["H4R_G"]
    t_diff = abs(trap["T"] - ref["T"])
    p_diff = max(abs(trap["p_real_lt_base"] - ref["p"]), abs(trap["p_real_gt_base"]
                                                             - ref["p_opposite"]))
    if args.n_boot == MS.BOOTSTRAP_N and max(t_diff, p_diff) > TOL:
        fail(f"the trapezoid's test differs from H4R-G (T by {t_diff:.2e}, p by {p_diff:.2e})")
    h = holm({lab: comp[lab][PRIMARY_FORM]["p_two_sided"] for lab in PRIMARY})
    changed_ = any(v["reject"] for v in h.values())
    primary = {"form": PRIMARY_FORM, "members": list(PRIMARY), "holm": h,
               "tests": {lab: comp[lab][PRIMARY_FORM] for lab in PRIMARY},
               "trapezoid_G_loc_mean_diff": trap["mean_diff"],
               "reading": ("short-scale following is changed by real-width training"
                           if changed_ else "no change detected at short scales")}
    kernel = {n: max((lab for lab in LX.LABELS if lab.startswith("d=")),
                     key=lambda x, n=n: res[n][x]["G_loc"]["median"]) for n in H.NETWORKS}
    groups = {}
    for v in H.VARIANTS:
        for arm in (H.REAL, H.BASE):
            names = [H.run_name(v, arm, s) for s in H.SEEDS]
            groups[f"{v}-{arm}"] = {lab: {form: {
                "median": float(np.median([res[n][lab][form]["median"] for n in names])),
                "range": [float(min(res[n][lab][form]["median"] for n in names)),
                          float(max(res[n][lab][form]["median"] for n in names))]}
                for form in comp[lab]} for lab in LX.LABELS}
    for arm in (H.REAL, H.BASE):
        names = [H.run_name(v, arm, s) for v in H.VARIANTS for s in H.SEEDS]
        groups[arm] = {lab: {form: {
            "median": float(np.median([res[n][lab][form]["median"] for n in names])),
            "range": [float(min(res[n][lab][form]["median"] for n in names)),
                      float(max(res[n][lab][form]["median"] for n in names))]}
            for form in comp[lab]} for lab in LX.LABELS}
    readings = {
        "monotone": {f: monotone(res, labs) for f, labs in LX.FAMILY_SCALES.items()},
        "kernel_peak": {"per_network": kernel,
                        "counts": {lab: sum(v == lab for v in kernel.values())
                                   for lab in sorted(set(kernel.values()))}},
    }

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    result = {
        "status": "EXPLORATORY: every number here is exploratory (locality-h4r-plan.md)",
        "plan": {"path": LX.rel(PLAN), "commit": git("log", "-1", "--format=%H", "--", LX.rel(PLAN)),
                 "expected_commit_prefix": PLAN_COMMIT, "sha256": LX.sha256_file(PLAN)},
        "primary": primary, "comparisons": comp, "readings": readings, "groups": groups,
        "networks": {n: {"keys": kept_n[n], "summaries": res[n]} for n in H.NETWORKS},
        "checks": {
            "arrays_checked": sum(c["n_arrays"] for c in checks.values()),
            "trapezoid_vs_row_G_max_abs": max(c["trap_max_abs"] for c in checks.values()),
            "trapezoid_summary_vs_decisions_max_abs": dec_max,
            "trapezoid_test_vs_H4R_G": {"T_abs_diff": t_diff, "p_abs_diff": p_diff},
            "basis_sum_max_abs": max(c["basis_sum_max_abs"] for c in checks.values()),
            "e_star_max_abs": max(c["e_star_max_abs"] for c in checks.values()),
            "s_star_beyond_plateau_max_m": max(c["s_star_beyond_plateau_max_m"]
                                               for c in checks.values()),
            "n_boot": args.n_boot},
        "inputs_sha256": dict(sorted(inputs.items())),
        "provenance": {"script": LX.rel(script), "script_sha256": LX.sha256_file(script),
                       "git_head": git("rev-parse", "HEAD"), "start": start,
                       "seconds": round(time.time() - t0, 1), "python": sys.version.split()[0]},
    }
    (OUT / "results.json").write_text(json.dumps(LX.clean(result), indent=1) + "\n")
    with (OUT / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "variant", "arm", "seed", "profile", "form", "median", "lo", "hi"])
        for n in H.NETWORKS:
            v, arm, seed = H.parse_name(n)
            for lab in LX.LABELS:
                for form, s in res[n][lab].items():
                    w.writerow([n, v, arm, seed, lab, form, s["median"], s["lo"], s["hi"]])
    print(json.dumps(LX.clean(primary), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
