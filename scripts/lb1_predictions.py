"""LB-1's calibrated predictions (docs/prereg/protocol-v1.md Section 9), from the R-dev dry run.

    uv run python scripts/lb1_predictions.py --root outputs/lockbox-dryrun/lb1-rdev-e0 \\
        --out docs/prereg/protocol-v1-predictions.json

Reads only the dry run's own ``freeze.json`` and ``analysis.json`` (R-dev standing in for
R_TEST); it never reads outputs/lockbox/, outputs/sealed/ or any R_TEST file. Every statistic
is scripts/lockbox.py's own (``family_summary``, ``iqm``, ``_med``, ``_pct``), so the point
values equal the dry run's ``headline_median_subject`` and ``iqm_gap`` (checked before anything
is written).

**The prediction.** Each point value is the dry run's value. Its interval is a circuit
bootstrap: the 12 R-dev circuits are resampled with replacement, each draw keeping all five
subjects' rows of a circuit and its three variants together (so the mirrored and reversed
differences are paired), and every statistic is recomputed exactly as ``analyse`` computes it;
the interval is the nearest-rank 2.5th to 97.5th percentile over ``N_BOOT`` draws. It treats
R_TEST as a fresh draw of 12 circuits like R-dev's, with no further widening: a lockbox value
outside its interval is evidence that R_TEST differs from that model (the R-dev numbers are
in-sample for widths; the R_TEST numbers are not). A prediction holds when the lockbox value
lies within its interval, ends included, both rounded to ``HELD_DECIMALS`` places (so a value
equal to a bound in exact arithmetic is never lost to float rounding); a missing value does not
hold. The completion differences are in circuits (integers), not shares.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lockbox as LB  # the harness's own statistics

N_BOOT = 4000
SEED = 2029
HELD_DECIMALS = 12
FAMILIES = ("original", "mirrored", "reversed")
STATS = ("laps_completed_share", "circuits_completed", "circuits_completed_share", "median_gap")


def headline(rows: list[dict], subjects: list[str], cmin: float) -> dict:
    """``analyse``'s headline_median_subject for these rows: per family, the median over
    subjects of each subject's ``family_summary``."""
    out = {}
    for f in FAMILIES:
        per = [LB.family_summary([r for r in rows if r["subject"] == s and r["variant"] == f],
                                 cmin) for s in subjects]
        out[f] = {k: LB._med([p.get(k, np.nan) for p in per]) for k in STATS}
    return out


def iqm_gap(rows: list[dict], subjects: list[str], family: str) -> float:
    """``analyse``'s completed-only IQM of the nominal gap over subject x circuit."""
    g = [LB.unclean(r["gap"]["+0.0"]) for r in rows
         if r["variant"] == family and r["subject"] in subjects and r["circuit_completed"]]
    return LB.iqm(np.asarray(g, float))


def predictions(rows: list[dict], subjects: list[str], cmin: float) -> dict:
    h = headline(rows, subjects, cmin)
    o, m, v = h["original"], h["mirrored"], h["reversed"]
    return {
        "laps_completed_share": o["laps_completed_share"],
        "circuits_completed": o["circuits_completed"],
        "median_gap": o["median_gap"],
        "iqm_gap_completed_only": iqm_gap(rows, subjects, "original"),
        "mirrored_minus_original_circuits_completed":
            m["circuits_completed"] - o["circuits_completed"],
        "reversed_minus_original_circuits_completed":
            v["circuits_completed"] - o["circuits_completed"],
        "mirrored_minus_original_median_gap": m["median_gap"] - o["median_gap"],
        "reversed_minus_original_median_gap": v["median_gap"] - o["median_gap"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="the dry run's root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    rec = json.loads((root / "freeze.json").read_text())
    if rec["mode"] != "dryrun":
        raise SystemExit("not a dry run: predictions come from R-dev only")
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists: write-once")
    raw = (root / "analysis.json").read_bytes()
    an = json.loads(raw)
    subjects, circuits = an["subjects"], an["circuits"]
    cmin = rec["protocol"]["laps"]["completion_min"]
    rows = an["rows"]

    point = predictions(rows, subjects, cmin)
    hd = an["headline_median_subject"]
    mine = headline(rows, subjects, cmin)
    for f in FAMILIES:                      # every headline input of every prediction
        for k in STATS:
            got, w = mine[f][k], LB.unclean(hd[f][k])
            if not (got == w or np.isclose(got, w, rtol=0, atol=1e-12)):
                raise SystemExit(f"{f}.{k}: recomputed {got} but analysis.json has {w}")
    w = LB.unclean(an["iqm_gap"]["original"]["completed_only"]["iqm"])
    if not np.isclose(point["iqm_gap_completed_only"], w, rtol=0, atol=1e-12):
        raise SystemExit(f"iqm_gap: recomputed {point['iqm_gap_completed_only']}, file has {w}")

    by_circuit = {c: [r for r in rows if r["circuit"] == c] for c in circuits}
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in point}
    for _ in range(N_BOOT):
        picked = rng.integers(0, len(circuits), len(circuits))
        boot = []
        for j, i in enumerate(picked):          # a circuit drawn twice is two circuits
            boot += [{**r, "circuit": f"{r['circuit']}#{j}"} for r in by_circuit[circuits[i]]]
        for k, x in predictions(boot, subjects, cmin).items():
            draws[k].append(x)
    lo, hi = LB.PROTOCOL["stats"]["ci"]
    out = {"record": "lb1-predictions", "label": "from the R-dev dry run (in-sample for widths)",
           "root": str(root), "freeze_id": rec["freeze_id"],
           "analysis_sha256": hashlib.sha256(raw).hexdigest(),
           "method": "circuit bootstrap, paired over variants; nearest-rank percentiles",
           "n_boot": N_BOOT, "seed": SEED, "interval": [lo, hi],
           "holds_when": ("the lockbox value lies within the interval, ends included, both "
                          f"rounded to {HELD_DECIMALS} decimal places; a missing value (or a "
                          "difference with a missing term) does not hold"),
           "predictions": {k: {"point": point[k],
                               "interval": [LB._pct(np.asarray(v), lo),
                                            LB._pct(np.asarray(v), hi)]}
                           for k, v in draws.items()}}
    Path(a.out).write_text(json.dumps(LB.clean(out), indent=1) + "\n")
    print(json.dumps(out["predictions"], indent=1))


if __name__ == "__main__":
    main()
