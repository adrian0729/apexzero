"""Interpretability leftovers IN-1 to IN-4: EXPLORATORY and DESCRIPTIVE only
(docs/findings/interp-leftovers-plan.md, committed in 916edaa before anything here was
computed; amendment 1 in 79977af, after IN-1's two built-in checks and before any result).

    uv run python scripts/interp_leftovers.py in1    # step 2: pair-weighted anchored-key G
    uv run python scripts/interp_leftovers.py in2    # step 2: H2' on the common A2 keys
    uv run python scripts/interp_leftovers.py in3    # step 1: outputs never written up
    uv run python scripts/interp_leftovers.py in4    # H4R: lap gap where both arms complete

Every number comes from files already on disk: no network, closed loop, gradient or solver is
run, and R_TEST stays sealed (IN-4 reads only the 12 R-dev circuits, each through
``viz.export.lockbox_guard``). Step 2's frozen code (src/apexzero/fields) and step 1's
(src/apexzero/interp) are imported read-only; every subcommand refuses to run when src/,
scripts/fields.py or scripts/psychophysics.py differ from HEAD. Nothing is written under
outputs/fields/, outputs/psychophysics/ or outputs/h4r/: results go to
outputs/interp-leftovers/<item>.json, each with a provenance block.

**Inputs, checked before use** (plan, Section 1): step 2's decisions.json and step 1's
decisions_all.json by their sha256; step 2's row files against their provenance sidecars; the
key lists against their embedded content hash; H4R's evaluations against the manifest's
eval_sha256 and the manifest by its sha256.

**IN-1.** Each anchored row (finals and earlier snapshots) is copied with w_is_cell set to
true (``copy_rows``; an A2 horizon twin, which no anchored row is, would keep its value) and
passed to the frozen ``stages.anchored_edges``: G, T and B with step 2's exclusions, pair
weighting, cluster bootstrap and coverage. H2''s band rule labels each G (descriptive).
Check (i): the same copy-and-call on the A2 rows reproduces decisions.json's H2 and
edge_descriptive exactly. Check (ii), as amended: plain per-key medians with step 2's hand
rule (the frozen exclusions plus the row's overall branch verdict) reproduce step 2's
Sections 7.8 and 7.9.

**IN-2.** K = the primary non-capped A2 keys with no exclusion in any of the nine
generalists (expected 306). Primary: keys outside K excluded ('not common'), then step 2's
pair rule, ``analysis.summarise`` with ``measures.h2_decision`` and R18's variant rule.
Sensitivities: (a) keys outside K dropped from the pairs; (b) G3s2's failing gate2-trained key
also left out of K.

**IN-3.** Step 1's P2d, P3 ratio, P6d, P8-F and per-checkpoint P6 curves, tabulated from
decisions_all.json, with one figure.

**IN-4.** The lap gap (median completed lap over the s-OCP lap, minus 1, in points) of H4R's
20 networks on circuits both arms complete: per-circuit arm medians, paired by seed (sign
flips and a bootstrap over pairs), GW-4's per-network statistic, and the pooled median.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "interp-leftovers"
FIELDS = ROOT / "outputs" / "fields"
PSY = ROOT / "outputs" / "psychophysics" / "analysis"
H4R = ROOT / "outputs" / "h4r"
PLAN = ROOT / "docs" / "findings" / "interp-leftovers-plan.md"
FIGURE = ROOT / "docs" / "findings" / "interp-leftovers-p6.png"
PLAN_COMMIT = "916edaa"
AMENDMENT_COMMIT = "79977af"
FROZEN = ("src", "scripts/fields.py", "scripts/psychophysics.py")

DECISIONS_SHA256 = "7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf"
DECISIONS_ALL_SHA256 = "14c674863e78b68cf01986bddcaec34480700ea21811aa0971550711feca4ea1"
H4R_MANIFEST_SHA256 = "f4731356a19a4196f9fbf2a4b742f3f0fe54c1b2b2ffb1fe4a3cae869ab4f3a3"

STATUS = {"in1": "EXPLORATORY (descriptive labels): IN-1 of interp-leftovers-plan.md",
          "in2": "EXPLORATORY (all labels descriptive): IN-2 of interp-leftovers-plan.md",
          "in3": "DESCRIPTIVE: IN-3 of interp-leftovers-plan.md (read from decisions_all.json)",
          "in4": "DESCRIPTIVE: IN-4 of interp-leftovers-plan.md"}

# ---- IN-1: step 2's tables (receptive-fields-step2.md) that check (ii) reproduces
# Section 7.9: agent -> (keys, G, B, T), A1 and tier B pooled; G to 3 decimals, B and T to 2.
TABLE_7_9 = {"G1": (662, 0.787, 0.98, 0.72), "G1s1": (565, 0.815, 0.99, 0.78),
             "G1s2": (662, 0.814, 1.44, 0.72), "G2": (681, 0.833, 0.97, 0.75),
             "G2s1": (687, 0.802, 1.31, 0.73), "G2s2": (669, 0.787, 1.24, 0.70),
             "G3": (567, 0.851, 0.72, 0.79), "G3s1": (687, 0.744, 1.43, 0.72),
             "G3s2": (687, 0.730, 1.22, 0.75)}
# Section 7.8: snapshot (or '<final>/A1') -> (keys, G), A1 keys only.
TABLE_7_8 = {"G1@73680629": (160, 0.787), "G1@109856054": (159, 0.827),
             "G1@134140317": (160, 0.822), "G2@134140317": (155, 0.870),
             "G2@163792745": (167, 0.834), "G3@109856054": (191, 0.893),
             "G3@134140317": (190, 0.891), "G3@163792745": (148, 0.848),
             "G1/A1": (425, 0.793), "G2/A1": (443, 0.834), "G3/A1": (376, 0.844)}
E1_TOL = 0.03                   # E1.1: pair-weighted G within this of the per-key median
E1_RANGE = (0.78, 0.90)         # E1.2 (amendment 1)
DEFICIT_HI = 0.5                # a CI's upper end at or below this would be a deficit
A1_AGENTS = ("G1", "G2", "G3")  # amendment 1: their finals' A1-only summary

# ---- IN-2
K_EXPECTED = 306
NOT_COMMON = "not common"
SECTION_5_1 = {"G2s1": 0.848, "G2": 0.842, "G1s2": 0.841, "G3": 0.840, "G2s2": 0.826,
               "G1": 0.817, "G1s1": 0.816, "G3s2": 0.811, "G3s1": 0.794}
LAWFUL_LO = 0.8
MOST = 5                        # amendment 1: 'most' of nine

# ---- IN-3
IN3_ITEMS = ("P2d", "P3-ratio", "P3", "P6d", "P8-F", "P6a", "P6b", "P6c")
SLOPE_MAX = 0.2
B_SHIFT_MAX = 0.05
F_MAX = 0.2

# ---- IN-4
COMPLETION_MIN = 0.5            # scripts/h4r.py's COMPLETION_MIN: completed when above it
PAIR_BOOT_N = 20_000
PAIR_BOOT_SEED = 0
GAP_TOL = 1e-12
WIDTH_TEST_DELTA = {"v1": 0.30, "v3": 0.57}   # width-test.md, Section 5.3 (points)
E4_BAND = (0.2, 0.8)
E4_MAX = 1.0
POINTS = 100.0


class CheckFailed(SystemExit):
    """A built-in or input check failed: the item stops and writes nothing."""


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise CheckFailed(1)


# ------------------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def frozen_problems() -> list[str]:
    """Paths under the frozen harnesses that differ from HEAD (untracked files included)."""
    return [ln for ln in git("status", "--porcelain", "--", *FROZEN).splitlines() if ln.strip()]


def clean(x):
    """JSON-safe: NaN as null and infinities as 'inf' / '-inf', NumPy values as Python."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def provenance(item: str, argv: list[str], t0: float, start: str, inputs: dict) -> dict:
    script = Path(__file__).resolve()
    return {"item": item, "script": rel(script), "script_sha256": sha256_file(script),
            "plan": {"path": rel(PLAN), "sha256": sha256_file(PLAN),
                     "commit": git("log", "-1", "--format=%H", "--", rel(PLAN)),
                     "plan_commit": PLAN_COMMIT, "amendment_1_commit": AMENDMENT_COMMIT},
            "git_head": git("rev-parse", "HEAD"), "frozen_code_clean": True,
            "inputs_sha256": inputs, "argv": [rel(script), *argv], "start": start,
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system()}}


def write(name: str, obj: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(clean(obj), indent=1, allow_nan=False))
    print(f"wrote {rel(path)}", flush=True)
    return path


# ------------------------------------------------------------------------- checked inputs


def checked_json(path: Path, sha: str, what: str) -> dict:
    raw = Path(path).read_bytes()
    got = sha256_bytes(raw)
    if got != sha:
        fail(f"{what} is not the committed file ({got})")
    return json.loads(raw)


def checked_rows(path: Path, inputs: dict) -> list[dict]:
    """A row file, checked against the sha256 its provenance sidecar records."""
    side = Path(f"{path}.provenance.json")
    want = json.loads(side.read_text())["files"].get(Path(path).name)
    raw = Path(path).read_bytes()
    got = sha256_bytes(raw)
    if want is None or got != want:
        fail(f"{rel(path)}: sha256 {got} is not its sidecar's {want}")
    inputs[rel(path)] = got
    return [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]


def key_lists(inputs: dict) -> dict:
    """outputs/fields/keys/lists.json, checked against its embedded content sha256."""
    from apexzero.interp import io

    p = FIELDS / "keys" / "lists.json"
    obj = json.loads(p.read_text())
    want = obj["provenance"]["content_sha256"]
    got = sha256_bytes(io.json_bytes({k: v for k, v in obj.items() if k != "provenance"}))
    if got != want:
        fail(f"keys/lists.json: content sha256 {got} is not its stamp's {want}")
    inputs[rel(p)] = sha256_file(p)
    return obj


def step2_inputs(inputs: dict) -> tuple[dict, dict, dict]:
    """(decisions.json, the optimum's rows by key id, the key lists)."""
    dec = checked_json(FIELDS / "analyse" / "decisions.json", DECISIONS_SHA256,
                       "outputs/fields/analyse/decisions.json")
    inputs["outputs/fields/analyse/decisions.json"] = DECISIONS_SHA256
    opt = {r["key_id"]: r for r in checked_rows(FIELDS / "opt-fields" / "rows.jsonl", inputs)}
    return dec, opt, key_lists(inputs)


def a2_rows(names, inputs: dict) -> dict[str, list[dict]]:
    return {n: checked_rows(FIELDS / "fields-a2" / f"{n.replace('/', '_')}.jsonl", inputs)
            for n in names}


def gate2(inputs: dict) -> dict:
    p = FIELDS / "gate2-trained" / "gate2_trained.json"
    obj = json.loads(p.read_text())
    inputs[rel(p)] = sha256_file(p)
    return obj


# ------------------------------------------------------------------------- IN-1


def copy_rows(rows: list[dict], twins: set[str]) -> list[dict]:
    """Each row copied with w_is_cell true, except an A2 horizon twin's (kept as it is); the
    input rows are not changed."""
    return [r if r["key_id"] in twins else dict(r, w_is_cell=True) for r in rows]


def overall_branch_ok(r: dict) -> bool:
    """The row's overall branch verdict (step 2's hand rule); a row without one passes."""
    b = r.get("branch")
    return True if not b else bool(b["ok"])


def per_key(km, rows: list[dict], approach: str | None = None) -> dict:
    """Plain medians over kept keys (not pair-weighted, no CI) under the frozen rule and under
    step 2's hand rule (amendment 1), with the keys where the two differ."""
    byid = {r["key_id"]: r for r in rows}
    ks = [k for k in km if approach is None or k.approach == approach]
    frozen = [k for k in ks if not k.reason]
    hand = [k for k in frozen if overall_branch_ok(byid[k.key_id])]

    def med(x):
        return float(np.median([k.value for k in x])) if x else float("nan")

    return {"frozen": {"n": len(frozen), "median": med(frozen)},
            "hand": {"n": len(hand), "median": med(hand)},
            "differ": sorted({k.key_id for k in frozen} - {k.key_id for k in hand})}


def fmt_eq(x: float, ref: float, nd: int) -> bool:
    return f"{x:.{nd}f}" == f"{ref:.{nd}f}"


def check_tables(pk: dict[str, dict]) -> dict:
    """Check (ii) as amended: ``pk`` maps a table row's name to {'G': per_key(...), 'B': ...,
    'T': ...} (TABLE_7_9) or {'G': per_key(...)} (TABLE_7_8). Returns the comparison; fails on
    any mismatch under the hand rule."""
    rows, bad = {}, []
    for name, (n, g, b, t) in TABLE_7_9.items():
        got = {m: pk["7.9"][name][m]["hand"] for m in ("G", "B", "T")}
        ok = (got["G"]["n"] == n and fmt_eq(got["G"]["median"], g, 3)
              and fmt_eq(got["B"]["median"], b, 2) and fmt_eq(got["T"]["median"], t, 2))
        rows[f"7.9 {name}"] = {"table": [n, g, b, t], "hand": [got["G"]["n"], got["G"]["median"],
                                                               got["B"]["median"],
                                                               got["T"]["median"]],
                               "frozen": [pk["7.9"][name]["G"]["frozen"]["n"],
                                          pk["7.9"][name]["G"]["frozen"]["median"]],
                               "match": ok}
        bad += [] if ok else [f"7.9 {name}"]
    for name, (n, g) in TABLE_7_8.items():
        got = pk["7.8"][name]["G"]["hand"]
        ok = got["n"] == n and fmt_eq(got["median"], g, 3)
        rows[f"7.8 {name}"] = {"table": [n, g], "hand": [got["n"], got["median"]],
                               "frozen": [pk["7.8"][name]["G"]["frozen"]["n"],
                                          pk["7.8"][name]["G"]["frozen"]["median"]],
                               "match": ok}
        bad += [] if ok else [f"7.8 {name}"]
    if bad:
        fail(f"check (ii): per-key medians do not reproduce step 2's tables: {bad}")
    return {"pass": True, "rule": "frozen exclusions plus the row's overall branch verdict "
                                  "(amendment 1)", "rows": rows}


def json_norm(x):
    from apexzero.interp import io

    return json.loads(io.json_bytes(x))


def check_i(rows_a2: dict[str, list[dict]], opt: dict, twins: set[str], gate2_ok: dict,
            dec: dict, n_boot: int) -> dict:
    """Check (i): copy-and-call on the A2 rows changes no row and reproduces decisions.json's
    H2 and edge_descriptive exactly (serialised as step 2 wrote them)."""
    from apexzero.fields import stages as S

    copied = {n: copy_rows(r, twins) for n, r in rows_a2.items()}
    changed = sum(a != b for n in rows_a2 for a, b in zip(rows_a2[n], copied[n], strict=True))
    if changed:
        fail(f"check (i): the copy changed {changed} A2 rows")
    out = S.analyse_edges(copied, opt, gate2_ok=gate2_ok, n=n_boot, decide=True)
    res = {"n_rows_changed": 0, "n_boot": n_boot}
    for k in ("H2", "edge_descriptive"):
        got, ref = json_norm(out[k]), dec[k]
        diff = sorted(n for n in set(got) | set(ref) if got.get(n) != ref.get(n))
        res[k] = {"n_entries": len(got), "equal": not diff, "differ": diff}
        if diff:
            fail(f"check (i): {k} differs from decisions.json for {diff[:5]}")
    res["pass"] = True
    return res


def overlap(a: dict, b: dict) -> bool:
    """Two intervals share a point (amendment 1)."""
    return bool(max(a["lo"], b["lo"]) <= min(a["hi"], b["hi"]))


def wholly_above(a: dict, b: dict) -> bool:
    """Interval a lies wholly above interval b."""
    return bool(a["lo"] > b["hi"])


def snapshot_order(earlier: list[str]) -> dict[str, list[str]]:
    """{agent: its earlier snapshots in timestep order}."""
    by: dict[str, list[tuple[int, str]]] = {}
    for n in earlier:
        a, ts = n.split("@")
        by.setdefault(a, []).append((int(ts), n))
    return {a: [n for _, n in sorted(v)] for a, v in sorted(by.items())}


def e1_readings(res: dict, a1: dict, pk: dict, dec: dict, generalists, specialists,
                order: dict[str, list[str]]) -> dict:
    """E1.1-E1.3 (plan, Section 2, with amendment 1's readings)."""
    e11 = {}
    for n in generalists:
        g = res[n]["G"]["pooled"]
        ref = dec["H2"][n]["pooled"]
        pkm = pk[n]["hand"]["median"]
        e11[n] = {"pair_weighted": g["median"], "lo": g["lo"], "hi": g["hi"],
                  "per_key_median_7_9": pkm, "per_key_median_frozen": pk[n]["frozen"]["median"],
                  "diff": g["median"] - pkm, "within_tol": abs(g["median"] - pkm) <= E1_TOL,
                  "hi_above_0.5": g["hi"] > DEFICIT_HI,
                  "a2": {k: ref[k] for k in ("median", "lo", "hi")},
                  "overlaps_a2": overlap(g, ref),
                  "wholly_below_a2": bool(g["hi"] < ref["lo"])}
    e12 = {}
    for agent, snaps in order.items():
        seq = [(s, res[s]["G"]["pooled"]) for s in snaps]
        if agent in a1:
            seq.append((f"{agent}/A1", a1[agent]["pooled"]))
        first = seq[0][1]
        e12[agent] = {"sequence": {s: {k: v[k] for k in ("median", "lo", "hi")} for s, v in seq},
                      "earlier_in_range": {s: bool(E1_RANGE[0] <= res[s]["G"]["pooled"]["median"]
                                                   <= E1_RANGE[1]) for s in snaps},
                      "later_ci_wholly_above_earliest": [s for s, v in seq[1:]
                                                         if wholly_above(v, first)]}
        e12[agent]["holds"] = (all(e12[agent]["earlier_in_range"].values())
                               and not e12[agent]["later_ci_wholly_above_earliest"])
    e13 = {n: {"share_excluded": res[n]["coverage"]["share_excluded"],
               "descriptive": bool(res[n]["coverage"]["descriptive"])} for n in specialists}
    return {"E1.1": {"per_agent": e11,
                     "holds": all(v["within_tol"] and v["hi_above_0.5"] and v["overlaps_a2"]
                                  for v in e11.values()),
                     "n_within_tol": sum(v["within_tol"] for v in e11.values()),
                     "n_hi_above_0.5": sum(v["hi_above_0.5"] for v in e11.values()),
                     "n_overlap_a2": sum(v["overlaps_a2"] for v in e11.values()),
                     "wholly_below_a2": [n for n, v in e11.items() if v["wholly_below_a2"]]},
            "E1.2": {"per_agent": e12, "holds": all(v["holds"] for v in e12.values())},
            "E1.3": {"per_agent": e13, "holds": all(v["descriptive"] for v in e13.values())}}


def run_in1(args) -> int:
    from apexzero.fields import analysis as A
    from apexzero.fields import measures as MS
    from apexzero.fields import stages as S
    from apexzero.interp import registry

    t0, start, inputs = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S"), {}
    dec, opt, lists = step2_inputs(inputs)
    twins = set(lists["a2"]["w_twins"])
    finals, earlier = list(lists["anchored_finals"]), list(lists["anchored_earlier"])
    generalists = tuple(registry.generalists())
    specialists = tuple(a for a in finals if a not in generalists)
    rows = {n: checked_rows(FIELDS / "fields-anchored" / f"{n}.jsonl", inputs) for n in finals}
    rows |= {n: checked_rows(FIELDS / "fields-earlier" / f"{n}.jsonl", inputs) for n in earlier}

    # ---- structure (plan, Method): no twin among the anchored rows; one key per (pair,
    # approach, direction, phase) within each agent
    for n, rs in rows.items():
        if tw := twins & {r["key_id"] for r in rs}:
            fail(f"{n}: anchored rows include A2 horizon twins {sorted(tw)}")
        cells = [(r["pair_id"], r["approach"], r["direction"], r["phase"]) for r in rs]
        if len(set(cells)) != len(cells):
            fail(f"{n}: two anchored keys share (pair, approach, direction, phase)")
        if any(r.get("w_is_cell", True) for r in rs):
            fail(f"{n}: an anchored row already has w_is_cell true")

    # ---- check (i)
    g2 = gate2(inputs)
    gate2_ok = {n: bool(v["pass"]) for n, v in g2["networks"].items()}
    ra2 = a2_rows(A.A2_NETWORKS, inputs)
    print("check (i): the frozen analyse_edges on the copied A2 rows ...", flush=True)
    chk_i = check_i(ra2, opt, twins, gate2_ok, dec, args.n_boot)
    print(f"check (i) passed ({time.time() - t0:.0f} s)", flush=True)

    # ---- per-key medians and check (ii)
    copied = {n: copy_rows(rs, twins) for n, rs in rows.items()}
    km = {n: {m: S.edge_key_measures(copied[n], opt, m) for m in ("G", "B", "T")}
          for n in rows}
    pk = {n: {m: per_key(km[n][m], rows[n]) for m in ("G", "B", "T")} for n in rows}
    pk_tab = {"7.9": {n: pk[n] for n in TABLE_7_9},
              "7.8": {n: {"G": pk[n]["G"]} for n in earlier}
              | {f"{a}/A1": {"G": per_key(km[a]["G"], rows[a], "A1")} for a in A1_AGENTS}}
    chk_ii = check_tables(pk_tab)
    chk_ii["hand_rule_extra_keys"] = {n: pk[n]["G"]["differ"] for n in rows
                                      if pk[n]["G"]["differ"]}
    print("check (ii) passed (amended rule)", flush=True)

    # ---- the summaries (frozen anchored_edges on the copies) and labels
    res, a1 = {}, {}
    for n in [*finals, *earlier]:
        r = S.anchored_edges(copied[n], opt, args.n_boot)
        g = r["G"]["pooled"]
        r["G"]["h2_band_label"] = MS.h2_decision(g["lo"], g["hi"])
        r["descriptive"] = bool(r["coverage"]["descriptive"])
        r["n_rows"] = len(rows[n])
        r["per_key"] = {m: {k: pk[n][m][k] for k in ("frozen", "hand")} for m in ("G", "B", "T")}
        res[n] = r
        print(f"{n}: G {g['median']:.3f} [{g['lo']:.3f}, {g['hi']:.3f}] on "
              f"{g['n_geometries']} geometries, {g['n_defined']}/{g['n_pairs']} pairs; "
              f"excluded {r['coverage']['share_excluded']:.1%}; {time.time() - t0:.0f} s",
              flush=True)
    for a in A1_AGENTS:                      # amendment 1: the finals' A1-only summary
        rows_a1 = [k for k in km[a]["G"] if k.approach == "A1"]
        a1[a] = A.summarise(rows_a1, None, args.n_boot)
        a1[a]["h2_band_label"] = MS.h2_decision(a1[a]["pooled"]["lo"], a1[a]["pooled"]["hi"])
        a1[a]["per_key"] = pk_tab["7.8"][f"{a}/A1"]["G"]

    readings = e1_readings(res, a1, {n: pk[n]["G"] for n in rows}, dec, generalists,
                           specialists, snapshot_order(earlier))
    out = {"status": STATUS["in1"],
           "note": "The frozen stages.anchored_edges on copies of the anchored rows with "
                   "w_is_cell true; every other rule is step 2's. Labels are descriptive.",
           "checks": {"structure": {"pass": True, "twins_absent": True,
                                    "unique_pair_approach_direction_phase": True},
                      "check_i": chk_i, "check_ii": chk_ii},
           "agents": res, "finals_A1_only": a1, "readings": readings,
           "provenance": provenance("in1", sys.argv[2:], t0, start, inputs)}
    write("in1.json", out)
    return 0


# ------------------------------------------------------------------------- IN-2


def common_keys(km: dict[str, list]) -> set[str]:
    """The keys kept (no exclusion of any reason) in every network's key measures."""
    sets = [{k.key_id for k in rows if not k.reason} for rows in km.values()]
    return set.intersection(*sets) if sets else set()


def mask_outside(rows: list, keep: set[str]) -> list:
    """Primary: every key outside ``keep`` excluded ('not common'; a key already excluded
    keeps its own reason); the pairs keep all their listed keys."""
    return [k if (k.key_id in keep or k.reason)
            else dataclasses.replace(k, value=float("nan"), reason=NOT_COMMON) for k in rows]


def drop_outside(rows: list, keep: set[str]) -> list:
    """Sensitivity (a): only the keys in ``keep`` are listed."""
    return [k for k in rows if k.key_id in keep]


def h2_on(rows_by_net: dict[str, list], n_boot: int, gate2_ok: dict | None = None) -> dict:
    """Per network: step 2's summarise with H2''s bands and h2_network's coverage label;
    then R18's variant rule over the nine."""
    from apexzero.fields import analysis as A
    from apexzero.fields import measures as MS
    from apexzero.interp import registry

    per = {}
    for n, rows in rows_by_net.items():
        summ = A.summarise(rows, MS.h2_decision, n_boot)
        cov = MS.coverage(A.coverage_records(rows, n, "final", "G"))[f"{n}@final"]["G"]
        per[n] = A.h2_network(summ, cov, (gate2_ok or {}).get(n))
        per[n]["excluded"] = cov["excluded"]
    return {"networks": per, "variants": {v: A.h2_variant(per, v) for v in registry.variants()}}


def e2_readings(res: dict, dec: dict) -> dict:
    meds = {n: v["pooled"]["median"] for n, v in res["networks"].items()}
    lo = {n: v["pooled"]["lo"] for n, v in res["networks"].items()}
    hi = {n: v["pooled"]["hi"] for n, v in res["networks"].items()}
    others = {n: m for n, m in meds.items() if n != "G3"}
    in_range = min(others.values()) <= meds["G3"] <= max(others.values())
    others_clear = [n for n in others if lo[n] >= LAWFUL_LO]
    higher = {n: meds[n] - dec["H2"][n]["pooled"]["median"] for n in meds}
    return {"E2.1": {"G3_median": meds["G3"], "others_range": [min(others.values()),
                                                               max(others.values())],
                     "G3_within_others_range": bool(in_range),
                     "G3_ci_clears_0.8": bool(lo["G3"] >= LAWFUL_LO),
                     "others_ci_clearing_0.8": others_clear,
                     "holds": bool(in_range and (lo["G3"] < LAWFUL_LO or others_clear))},
            "E2.2": {"median_minus_confirmatory": higher,
                     "n_higher": sum(v > 0 for v in higher.values()),
                     "holds": sum(v > 0 for v in higher.values()) >= MOST},
            "E2.3": {"min_hi": min(hi.values()),
                     "holds": all(v > DEFICIT_HI for v in hi.values())},
            "lawful": sorted(n for n, v in res["networks"].items()
                             if v["band_decision"] == "lawful"),
            "labels": {n: v["decision"] for n, v in res["networks"].items()}}


def run_in2(args) -> int:
    from apexzero.fields import analysis as A
    from apexzero.fields import stages as S

    t0, start, inputs = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S"), {}
    dec, opt, _ = step2_inputs(inputs)
    nets = tuple(A.H2_NETWORKS)
    rows = a2_rows(nets, inputs)
    km = {n: S.edge_key_measures(rows[n], opt) for n in nets}
    ids = {n: {k.key_id for k in km[n]} for n in nets}
    if len({frozenset(v) for v in ids.values()}) != 1:
        fail("the nine generalists' primary non-capped key lists differ")
    K = common_keys(km)
    print(f"|K| = {len(K)} of {len(ids[nets[0]])}", flush=True)
    if len(K) != K_EXPECTED:
        fail(f"|K| = {len(K)}, not {K_EXPECTED}: the item stops for an amendment")

    # ---- check: plain per-key medians on K reproduce Section 5.1
    plain = {n: float(np.median([k.value for k in km[n] if k.key_id in K])) for n in nets}
    bad = [n for n in nets if not fmt_eq(plain[n], SECTION_5_1[n], 3)]
    if bad:
        fail(f"per-key medians on K do not reproduce Section 5.1: {bad}")
    check = {"pass": True, "plain_medians_on_K": plain, "section_5_1": SECTION_5_1}

    g2 = gate2(inputs)
    gate2_ok = {n: bool(v["pass"]) for n, v in g2["networks"].items()}
    g3s2_fail = sorted(r["key_id"] for r in g2["networks"]["G3s2"]["rows"] if not r["pass"])
    Kb = K - set(g3s2_fail)

    primary = h2_on({n: mask_outside(km[n], K) for n in nets}, args.n_boot)
    step2_g3s2 = h2_on({n: mask_outside(km[n], K) for n in nets}, args.n_boot,
                       {n: gate2_ok.get(n) for n in nets})
    sens_a = h2_on({n: drop_outside(km[n], K) for n in nets}, args.n_boot)
    sens_b = h2_on({n: mask_outside(km[n], Kb) for n in nets}, args.n_boot)
    for name, r in (("primary", primary), ("(a)", sens_a), ("(b)", sens_b)):
        for n, v in r["networks"].items():
            p = v["pooled"]
            print(f"{name} {n}: {v['decision']} {p['median']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]"
                  f" {p['n_geometries']} geometries", flush=True)
    out = {"status": STATUS["in2"],
           "note": "Keys outside K excluded ('not common'), then step 2's pair rule, "
                   "analysis.summarise with measures.h2_decision, h2_network's coverage label "
                   "and R18's variant rule. Every label is descriptive; the confirmatory H2' "
                   "decisions stand as decided.",
           "K": {"n": len(K), "n_listed": len(ids[nets[0]]), "keys": sorted(K)},
           "check": check,
           "primary": primary,
           "primary_variants_with_G3s2_descriptive": step2_g3s2["variants"],
           "sensitivity_a_dropped": sens_a,
           "sensitivity_b": {"G3s2_failing_gate2_keys": g3s2_fail,
                             "n_removed_from_K": len(K) - len(Kb), "n": len(Kb), **sens_b},
           "confirmatory": {n: {"decision": dec["H2"][n]["decision"],
                                **{k: dec["H2"][n]["pooled"][k] for k in ("median", "lo", "hi",
                                                                         "n_geometries")}}
                            for n in nets},
           "readings": {"primary": e2_readings(primary, dec),
                        "sensitivity_a": e2_readings(sens_a, dec),
                        "sensitivity_b": e2_readings(sens_b, dec)},
           "provenance": provenance("in2", sys.argv[2:], t0, start, inputs)}
    write("in2.json", out)
    return 0


# ------------------------------------------------------------------------- IN-3


def by_prediction(decisions: list[dict]) -> dict[str, dict[str, dict]]:
    """{prediction: {agent: decision}} for IN-3's items."""
    out: dict[str, dict[str, dict]] = {}
    for d in decisions:
        if d["prediction"] in IN3_ITEMS:
            out.setdefault(d["prediction"], {})[d["agent"]] = d
    return out


def as_float(x) -> float:
    """A decisions_all value as a float ('inf' strings and None included)."""
    if x is None:
        return float("nan")
    return float(x)


def trend(ts: list[int], vals: list[float]) -> dict:
    """Spearman rho against timesteps, first and final values (amendment 1's reading)."""
    from scipy.stats import spearmanr

    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    rho = float(spearmanr(np.asarray(ts)[ok], v[ok]).statistic) if ok.sum() >= 3 else np.nan
    return {"rho": rho, "first": float(v[0]), "final": float(v[-1]),
            "rising": bool(rho > 0 and v[-1] > v[0]), "falling": bool(rho < 0 and v[-1] < v[0])}


def p6_curves(p6a: dict, p6b: dict, p6c: dict, b: float) -> dict:
    """Per agent: the checkpoints, tail e_ve^RA at A1, v_top (m/s and km/h), exp(log_std)
    (steer, pedal), t_R per tier A radius, and the trend readings."""
    out = {}
    for agent in sorted(p6c):
        tail = {int(k): as_float(v) for k, v in p6c[agent]["details"]["tail_e_ve_ra"].items()}
        std = {int(k): [float(x) for x in v] for k, v in p6b[agent]["details"]["std"].items()}
        vtop = {int(k): float(v) for k, v in p6b[agent]["details"]["v_top"].items()}
        cks = sorted(tail)
        if sorted(std) != cks or sorted(vtop) != cks:
            fail(f"{agent}: P6b's and P6c's checkpoints differ")
        tR = {float(r): as_float(v) for r, v in p6a[agent]["details"]["t_R"].items()}
        out[agent] = {
            "checkpoints": cks,
            "tail_e_ve_ra_A1": [tail[c] for c in cks],
            "v_top_mps": [vtop[c] for c in cks], "v_top_kph": [3.6 * vtop[c] for c in cks],
            "std_steer": [std[c][0] for c in cks], "std_pedal": [std[c][1] for c in cks],
            "t_R": {f"{r:g}": tR[r] for r in sorted(tR)},
            "readings": {"tail_below_b_every_checkpoint": bool(all(tail[c] < b for c in cks)),
                         "tail_max": max(tail[c] for c in cks),
                         "v_top": trend(cks, [vtop[c] for c in cks]),
                         "std_steer": trend(cks, [std[c][0] for c in cks]),
                         "std_pedal": trend(cks, [std[c][1] for c in cks])}}
    return out


def in3_tables(P: dict, b: float, generalists) -> dict:
    p2d = {a: P["P2d"][a]["details"] for a in generalists}
    p3r = {a: {**P["P3-ratio"][a]["details"], "b_P3": P["P3"][a]["details"]["b"],
               "P3_verdict": P["P3"][a]["verdict"], "family": P["P3-ratio"][a]["family"]}
           for a in generalists}
    for v in p3r.values():
        v["b_shift"] = v["b_without_310"] - v["b_P3"]
    p6d = {a: {k: as_float(v) if k != "large_first" else v
               for k, v in P["P6d"][a]["details"].items()} for a in sorted(P["P6d"])}
    p8f = {a: P["P8-F"][a]["details"] for a in sorted(P["P8-F"])}
    read = {
        "P2d": {"n_frac_nearer_qss_below_0.5": sum(v["frac_nearer_qss"] < 0.5
                                                   for v in p2d.values()),
                "n_line_excess_within_b": sum(abs(v["median_line_excess"]) <= b
                                              for v in p2d.values()),
                "outside_b": [a for a, v in p2d.items() if abs(v["median_line_excess"]) > b]},
        "P3-ratio": {"n_abs_slope_below_0.2": sum(abs(v["ratio_slope_vs_log_vb"]) < SLOPE_MAX
                                                  for v in p3r.values()),
                     "n_b_shift_within_0.05": sum(abs(v["b_shift"]) <= B_SHIFT_MAX
                                                  for v in p3r.values()),
                     "b_shift_outside_0.05": [a for a, v in p3r.items()
                                              if abs(v["b_shift"]) > B_SHIFT_MAX]},
        "P6d": {"n_large_first_false": sum(v["large_first"] is False for v in p6d.values())},
        "P8-F": {"n_F_at_most_0.2": sum(as_float(v["F"]) <= F_MAX for v in p8f.values())}}
    read["P2d"]["holds"] = (read["P2d"]["n_frac_nearer_qss_below_0.5"] >= MOST
                            and not read["P2d"]["outside_b"])
    read["P3-ratio"]["holds"] = (read["P3-ratio"]["n_abs_slope_below_0.2"] == len(p3r)
                                 and not read["P3-ratio"]["b_shift_outside_0.05"])
    read["P6d"]["holds"] = read["P6d"]["n_large_first_false"] == len(p6d)
    read["P8-F"]["holds"] = read["P8-F"]["n_F_at_most_0.2"] == len(p8f)
    return {"P2d": p2d, "P3-ratio": p3r, "P6d": p6d, "P8-F": p8f, "readings": read}


SERIES = {"G1": ("#2a78d6", "o"), "G2": ("#eb6834", "s"), "G3": ("#1baf7a", "^")}


def p6_figure(curves: dict, b: float, paths: list[Path]) -> None:
    """One figure, four panels on a log timestep axis (one y-scale each): tail e_ve^RA at A1
    against +b, v_top, and exp(log_std) for steer and pedal. Colour follows the agent, with a
    marker shape and a direct label at the line's end as secondary encoding."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    panels = (("tail_e_ve_ra_A1", "Tail entry-speed error at A1, e_ve^RA (fraction)"),
              ("v_top_kph", "Tier G top speed v_top (km/h)"),
              ("std_steer", "Policy std, steer: exp(log_std)"),
              ("std_pedal", "Policy std, pedal: exp(log_std)"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (key, title) in zip(axes.ravel(), panels, strict=True):
        ax.set_facecolor("#fcfcfb")
        for agent, c in curves.items():
            col, mk = SERIES[agent]
            x = np.array(c["checkpoints"], float) / 1e6
            y = np.array(c[key], float)
            ax.plot(x, y, color=col, lw=2, marker=mk, ms=5, mec="#fcfcfb", mew=1, label=agent)
            ax.annotate(agent, (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=ink)
        if key == "tail_e_ve_ra_A1":
            ax.axhline(b, color=muted, lw=1, ls="--")
            ax.annotate(f"+b = {b:g}", (10.5, b), xytext=(0, 3), textcoords="offset points",
                        fontsize=8, color=muted)
            ax.axhline(0.0, color=grid, lw=1)
        ax.set_xscale("log")
        ax.set_xlim(9, 260)
        ax.set_xticks([10, 20, 50, 100, 200])
        ax.set_xticklabels(["10", "20", "50", "100", "200"])
        ax.set_title(title, fontsize=9, color=ink, loc="left")
        ax.grid(True, color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Training steps (millions, log scale)", fontsize=8, color=muted)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Step 1's per-checkpoint P6 curves, G1-G3 (descriptive; decisions_all.json)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_in3(args) -> int:
    from apexzero.interp import registry

    t0, start, inputs = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S"), {}
    path = PSY / "decisions_all.json"
    d = checked_json(path, DECISIONS_ALL_SHA256, rel(path))
    inputs[rel(path)] = DECISIONS_ALL_SHA256
    if d.get("confirmatory") is not False:
        fail("decisions_all.json is not the non-confirmatory run")
    b = float(d["band"])
    P = by_prediction(d["decisions"])
    generalists = tuple(registry.generalists())
    for item, agents in (("P2d", generalists), ("P3-ratio", generalists), ("P3", generalists),
                         ("P6d", A1_AGENTS), ("P8-F", A1_AGENTS), ("P6a", A1_AGENTS),
                         ("P6b", A1_AGENTS), ("P6c", A1_AGENTS)):
        if set(P.get(item, {})) != set(agents):
            fail(f"{item}: agents {sorted(P.get(item, {}))}, expected {sorted(agents)}")
    tables = in3_tables(P, b, generalists)
    curves = p6_curves(P["P6a"], P["P6b"], P["P6c"], b)
    tables["readings"]["P6_curves"] = {a: c["readings"] for a, c in curves.items()}
    tables["readings"]["P6_curves"]["holds"] = {
        "tail_below_b": all(c["readings"]["tail_below_b_every_checkpoint"]
                            for c in curves.values()),
        "v_top_rising": all(c["readings"]["v_top"]["rising"] for c in curves.values()),
        "std_falling": all(c["readings"][k]["falling"] for c in curves.values()
                           for k in ("std_steer", "std_pedal"))}
    verdicts = {item: {a: (v["family"], v["verdict"]) for a, v in P[item].items()}
                for item in IN3_ITEMS}
    fig_out = OUT / "p6_curves.png"
    p6_figure(curves, b, [fig_out, FIGURE])
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "in3_p6_curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["agent", "checkpoint", "tail_e_ve_ra_A1", "v_top_kph", "std_steer",
                    "std_pedal"])
        for a, c in curves.items():
            for i, ck in enumerate(c["checkpoints"]):
                w.writerow([a, ck, f"{c['tail_e_ve_ra_A1'][i]:.4f}", f"{c['v_top_kph'][i]:.2f}",
                            f"{c['std_steer'][i]:.4f}", f"{c['std_pedal'][i]:.4f}"])
    out = {"status": STATUS["in3"], "band_b": b, "families_and_verdicts": verdicts,
           **tables, "P6_curves": curves,
           "figure": [rel(fig_out), rel(FIGURE)],
           "provenance": provenance("in3", sys.argv[2:], t0, start, inputs)}
    write("in3.json", out)
    return 0


# ------------------------------------------------------------------------- IN-4


def completed_gaps(ev: dict, circuits) -> dict[str, float]:
    """{circuit: gap in points} for the circuits a network completes (completion above
    COMPLETION_MIN; H4R's rule)."""
    out = {}
    for c in circuits:
        row = ev[c]
        if float(row["completion"]) > COMPLETION_MIN:
            g = float(row["gap"])
            if not np.isfinite(g):
                fail(f"a completed circuit ({c}) has no finite gap")
            out[c] = POINTS * g
    return out


def primary_delta(base: list[dict[str, float]], real: list[dict[str, float]]) -> dict:
    """Measure 1: per circuit both arms complete (at least one network each), each arm's
    median gap over its completing networks and delta = real - base; their median and the
    number of circuits with delta > 0."""
    circ = sorted(set().union(*base) & set().union(*real))
    per = {}
    for c in circ:
        mb = float(np.median([g[c] for g in base if c in g]))
        mr = float(np.median([g[c] for g in real if c in g]))
        per[c] = {"base": mb, "real": mr, "delta": mr - mb,
                  "n_base": sum(c in g for g in base), "n_real": sum(c in g for g in real)}
    d = [v["delta"] for v in per.values()]
    return {"circuits": circ, "n_circuits": len(circ), "per_circuit": per,
            "median_delta": float(np.median(d)) if d else float("nan"),
            "n_positive": int(sum(x > 0 for x in d))}


def sign_flip_p(x) -> tuple[float, int]:
    """Two-sided p from every sign assignment of the values (amendment 1): the share whose
    mean's absolute value is at least the observed one's, the observed one included."""
    x = np.asarray(x, float)
    obs = abs(x.mean())
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(x))))
    means = np.abs(signs @ x / len(x))
    return float(np.mean(means >= obs - GAP_TOL)), len(signs)


def boot_mean_ci(x, n: int = PAIR_BOOT_N, seed: int = PAIR_BOOT_SEED) -> tuple[float, float]:
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def paired(pairs: dict[str, tuple[dict[str, float], dict[str, float]]]) -> dict:
    """Measure 2: per same-seed pair, the mean over circuits both complete of real - base."""
    per = {}
    for k, (gb, gr) in pairs.items():
        cs = sorted(set(gb) & set(gr))
        per[k] = {"circuits": cs, "n": len(cs),
                  "mean_diff": float(np.mean([gr[c] - gb[c] for c in cs])) if cs
                  else float("nan")}
    used = [k for k, v in per.items() if v["n"]]
    x = [per[k]["mean_diff"] for k in used]
    p, n_flips = sign_flip_p(x)
    lo, hi = boot_mean_ci(x)
    return {"per_pair": per, "n_pairs": len(used), "left_out": sorted(set(per) - set(used)),
            "mean": float(np.mean(x)), "p_two_sided": p, "n_sign_flips": n_flips,
            "ci95": [lo, hi], "n_positive": int(sum(v > 0 for v in x))}


def per_network_median(gaps: dict[str, float], circuits=None) -> float:
    v = [g for c, g in gaps.items() if circuits is None or c in circuits]
    return float(np.median(v)) if v else float("nan")


def arm_stats(base: list[float], real: list[float]) -> dict:
    b, r = np.asarray(base, float), np.asarray(real, float)
    sd_b, sd_r = float(np.std(b, ddof=1)), float(np.std(r, ddof=1))
    return {"base_mean": float(b.mean()), "real_mean": float(r.mean()),
            "diff": float(r.mean() - b.mean()), "base_sd": sd_b, "real_sd": sd_r,
            "n_base": len(b), "n_real": len(r),
            "se_diff": float(np.sqrt(sd_b ** 2 / len(b) + sd_r ** 2 / len(r)))}


def run_in4(args) -> int:
    from apexzero.tracks.corners import R_DEV, R_TEST
    from apexzero.viz.export import lockbox_guard

    t0, start, inputs = time.time(), time.strftime("%Y-%m-%dT%H:%M:%S"), {}
    man = checked_json(H4R / "manifest.json", H4R_MANIFEST_SHA256, "outputs/h4r/manifest.json")
    inputs["outputs/h4r/manifest.json"] = H4R_MANIFEST_SHA256
    circuits = list(R_DEV)
    for c in circuits:
        lockbox_guard(c)
        if c in R_TEST:
            fail(f"{c} is a lockbox circuit")
    ocp = {}
    for c in circuits:
        p = ROOT / "outputs" / "ocp" / f"{c}.pkl"
        with open(p, "rb") as fh:
            ocp[c] = float(pickle.load(fh).lap_time)
        inputs[rel(p)] = sha256_file(p)
    nets = man["networks"]
    if len(nets) != 20:
        fail(f"the manifest lists {len(nets)} networks, not 20")
    gaps, meta, chk = {}, {}, {"ocp_lap_max_abs": 0.0, "gap_max_abs": 0.0, "n_rows": 0}
    for name, m in sorted(nets.items()):
        p = ROOT / m["eval_path"]
        raw = p.read_bytes()
        if sha256_bytes(raw) != m["eval_sha256"]:
            fail(f"{m['eval_path']} differs from the manifest's eval_sha256")
        inputs[rel(p)] = m["eval_sha256"]
        ev = json.loads(raw)
        if sorted(k for k in ev if k in circuits) != sorted(circuits) or \
                set(ev) - set(circuits) - {"checkpoint", "timesteps"}:
            fail(f"{name}: its evaluation's circuits are not the 12 R-dev circuits")
        for c in circuits:
            row = ev[c]
            if float(row["completion"]) != float(m["completion"][c]):
                fail(f"{name} {c}: completion differs from the manifest")
            chk["ocp_lap_max_abs"] = max(chk["ocp_lap_max_abs"],
                                         abs(float(row["ocp_lap_s"]) - ocp[c]))
            if np.isfinite(float(row["median_lap_s"])):
                chk["gap_max_abs"] = max(chk["gap_max_abs"], abs(
                    float(row["gap"]) - (float(row["median_lap_s"]) / ocp[c] - 1.0)))
            chk["n_rows"] += 1
        gaps[name] = completed_gaps(ev, circuits)
        if len(gaps[name]) != int(m["completed"]):
            fail(f"{name}: {len(gaps[name])} completed circuits, the manifest says "
                 f"{m['completed']}")
        meta[name] = {"variant": m["variant"], "arm": m["arm"], "seed": int(m["seed"])}
    if chk["ocp_lap_max_abs"] != 0.0:
        fail(f"an evaluation's ocp_lap_s differs from outputs/ocp: {chk['ocp_lap_max_abs']}")
    if chk["gap_max_abs"] > GAP_TOL:
        fail(f"an evaluation's gap differs from median / ocp - 1 by {chk['gap_max_abs']}")
    chk["pass"] = True

    def group(v, arm):
        return [n for n in sorted(meta) if meta[n]["variant"] == v and meta[n]["arm"] == arm]

    variants = sorted({m["variant"] for m in meta.values()})
    prim, gw4, pooled, pairs = {}, {}, {}, {}
    for v in variants:
        nb, nr = group(v, "base"), group(v, "realw")
        if len(nb) != 5 or len(nr) != 5:
            fail(f"{v}: {len(nb)} baselines and {len(nr)} real-width networks, not 5 and 5")
        prim[v] = primary_delta([gaps[n] for n in nb], [gaps[n] for n in nr])
        prim[v]["width_test_delta"] = WIDTH_TEST_DELTA[v]
        common = set(prim[v]["circuits"])
        per_net = {n: {"all": per_network_median(gaps[n]),
                       "common": per_network_median(gaps[n], common),
                       "n_completed": len(gaps[n]),
                       "n_completed_common": sum(c in common for c in gaps[n])}
                   for n in nb + nr}
        gw4[v] = {"per_network": per_net,
                  "all_circuits": arm_stats([per_net[n]["all"] for n in nb],
                                            [per_net[n]["all"] for n in nr]),
                  "common_circuits": arm_stats([per_net[n]["common"] for n in nb],
                                               [per_net[n]["common"] for n in nr])}
        pb = [g for n in nb for g in gaps[n].values()]
        pr = [g for n in nr for g in gaps[n].values()]
        pooled[v] = {"base": float(np.median(pb)), "real": float(np.median(pr)),
                     "n_base": len(pb), "n_real": len(pr),
                     "diff": float(np.median(pr) - np.median(pb)),
                     "note": "mixes circuit sets (the seed study's pooled statistic)"}
        for s in sorted({meta[n]["seed"] for n in nb}):
            b = next(n for n in nb if meta[n]["seed"] == s)
            r = next(n for n in nr if meta[n]["seed"] == s)
            pairs[f"{v}/s{s}"] = (gaps[b], gaps[r])
    gw4["pooled_over_variants"] = {
        k: arm_stats([gw4[v]["per_network"][n][k.split("_")[0]] for v in variants
                      for n in group(v, "base")],
                     [gw4[v]["per_network"][n][k.split("_")[0]] for v in variants
                      for n in group(v, "realw")])
        for k in ("all_circuits", "common_circuits")}
    pr = paired(pairs)
    if len(pairs) != 10:
        fail(f"{len(pairs)} same-seed pairs, not 10")
    rises = {f"primary {v}": prim[v]["median_delta"] for v in variants}
    rises["paired mean"] = pr["mean"]
    for v in [*variants, "pooled_over_variants"]:
        for k in ("all_circuits", "common_circuits"):
            rises[f"GW-4 {v} {k}"] = gw4[v][k]["diff"]
    readings = {
        "E4.1": {"primary_in_band": {v: bool(E4_BAND[0] <= prim[v]["median_delta"] <= E4_BAND[1])
                                     for v in variants},
                 "paired_mean_above_0": bool(pr["mean"] > 0)},
        "E4.2": {"rises": rises, "all_below_1": bool(all(x < E4_MAX for x in rises.values())),
                 "at_or_above_1": [k for k, x in rises.items() if x >= E4_MAX]}}
    readings["E4.1"]["holds"] = bool(all(readings["E4.1"]["primary_in_band"].values())
                                     and readings["E4.1"]["paired_mean_above_0"])
    for v in variants:
        print(f"{v}: primary median delta {prim[v]['median_delta']:+.3f} points on "
              f"{prim[v]['n_circuits']} circuits ({prim[v]['n_positive']} positive)", flush=True)
    print(f"paired: mean {pr['mean']:+.3f} [{pr['ci95'][0]:+.3f}, {pr['ci95'][1]:+.3f}] "
          f"p {pr['p_two_sided']:.4f}", flush=True)
    out = {"status": STATUS["in4"], "units": "points (100 x (median lap / s-OCP lap - 1))",
           "completion_min": COMPLETION_MIN, "checks": chk, "ocp_lap_s": ocp,
           "networks": {n: {**meta[n], "gaps": gaps[n]} for n in sorted(meta)},
           "primary": prim, "paired": pr, "gw4": gw4, "pooled": pooled,
           "readings": readings,
           "provenance": provenance("in4", sys.argv[2:], t0, start, inputs)}
    write("in4.json", out)
    return 0


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    from apexzero.fields import measures as MS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("item", choices=("in1", "in2", "in3", "in4"))
    ap.add_argument("--n-boot", type=int, default=MS.BOOTSTRAP_N,
                    help="bootstrap draws for IN-1 and IN-2 (step 2's 20,000; check (i) "
                         "needs it)")
    args = ap.parse_args(argv)
    if probs := frozen_problems():
        fail(f"frozen code differs from HEAD: {probs}")
    if args.n_boot != MS.BOOTSTRAP_N and args.item in ("in1", "in2"):
        print(f"warning: --n-boot {args.n_boot} is not step 2's {MS.BOOTSTRAP_N}", flush=True)
    return {"in1": run_in1, "in2": run_in2, "in3": run_in3, "in4": run_in4}[args.item](args)


if __name__ == "__main__":
    sys.exit(main())
