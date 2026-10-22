"""Locality of edge tracking: EXPLORATORY (docs/findings/locality-plan.md, committed in 291128d
before anything here was computed). Every number this script writes is exploratory.

    uv run python scripts/locality_exploratory.py

How far a line follows an inside-edge change, as a function of the change's length scale and
placement, from step 2's stored per-key impulse responses only. No network, closed loop,
gradient or solver is run: by linearity the first-order response of c_min to an inside-width
profile h(s) is the dot product of h's samples with c_min's inside-width impulse row
(``perturb.combine``). Inside is w_left on left turns (sigma = +1) and w_right on right turns.

**Inputs** (read only): step 2's per-key rows (outputs/fields/fields-a2/*.jsonl, the 35
network entries; outputs/fields/opt-fields/rows.jsonl), the impulse arrays
(fields-a2/arrays/<net>/<key>.npz, each checked against its row's 'arrays_sha256';
opt-fields/arrays/<key>.npz) and the committed confirmatory decisions (checked by sha256).
Step 2's frozen code (src/apexzero/fields) is imported read-only; the script refuses to run
when src/ or scripts/fields.py differ from HEAD.

**s*** is the c_min argmin of the base run. The optimum's is its row's 's_star'. The network
rows do not store it, so it is recovered from the stored explicit term of c_min
('explicit_c_min': the edge's own move with the states held). Its inside row b is the arrays'
cubic-spline basis at the pick (or the linear interpolation between two states at a window
end). That spline reproduces constants and linear functions, so sum_j b_j = 1 and
sum_j b_j s_j = s*. Both are checked per key, with b . trapezoid against the row's e_star, and
for the optimum the recovered s* against its row's s_star.

**Measure** (plan, 'Measure'). G_loc(h) = H - dc_min/dDelta, with H the height of h at s*:
1 in families 1-3 and for step 2's trapezoid, and exp(-d^2 / 2 sigma_b^2) in family 4, which
is reported per unit of the bump's peak. For the trapezoid this is step 2's
G = 1 - dc_min/dDelta, checked per key to 1e-9 on every row with a finite G (the script fails
otherwise), and its pooled summary is checked against decisions.json for every network. The
line-move form L(h) = e_h(s*) - dc_min/dDelta = sigma dn(s*)/dDelta, with e_h(s*) = b . h the
edge's own move at s* as c_min reads it through the arrays' spline of the sampled h, is
reported beside G_loc. The two agree where that spline reproduces h at s*; the gap H - e_h(s*)
is reported per family member. Where they differ (sharp profiles: the X = 5 m ramp's corner at
s* reads as about 0.95, so a line that did not move at all would score G_loc of about 0.05), L
is the line's actual move. In families 1-3 the ratio L / e_h(s*) (form 'L_per_e': the line's
move per metre of the edge's move as c_min reads it, like H3's line / |edge| ratio) is also
summarised. Added after an independent verification of the first run (the first run's G_loc
and L are unchanged; the note says so).

**Families** (all with height 1 at s* except family 4):
1. Gaussian bumps centred at s*, sigma_b in {2, 4, 8, 16, 32, 64} m.
2. Ramps ending at s*: 0 at s* - X, linear to 1 at s*, flat to s_c1 + 20 m, X in
   {5, 10, 20, 40, 80} m. Beyond s_c1 + 20 m the plan does not say; here the ramp falls to 0
   over 20 m as the trapezoid's does, and if s* lies beyond s_c1 + 20 m the fall starts at s*.
3. Plateaus centred at s*, L in {10, 20, 40, 80} m, 20 m ramps (``perturb.trapezoid`` with
   inner L/2 and outer L/2 + 20 about s*), and step 2's corner-wide trapezoid as L_max.
4. Bumps of sigma_b = 8 m centred at s* + d, d in {-80, -60, -40, -20, -10, 0, +10, +20} m.

**Keys, exclusions and summaries** (plan, 'Data' and 'Summaries'): step 2's frozen
``stages.edge_key_measures`` gives each network's primary non-capped A2 keys and its exclusion
reasons (all of them, 'argmin off plateau' included), so every family member uses the keys
step 2's G used for that network. Each member is summarised by ``measures.agent_summary``
(pooled directions): mirror-pair values with step 2's pair rules, the median over geometries,
and the cluster bootstrap over geometries on step 2's 'bootstrap' stream, restarted for every
summary. The optimum's keys are ``stages.optimum_edges``' (primary non-capped A2 keys, a
'numerical' exclusion for a key that failed to re-polish).

**E4** (descriptive): per family member, H4's arms (real-width v1 and v3 seeds 0-3 against
baseline seeds 1-4) on the per-network medians, with the frozen within-variant exact
permutation statistic (``analysis.permutation_test``, 4,900 relabellings) and a two-sided p
(twice the smaller one-sided p, at most 1); the leave-out runs against the v3 baselines
likewise (495). The two gate2-trained failures are kept: everything here is exploratory.

**Outputs**: outputs/locality/results.json (with a provenance block), results.csv (one row
per entry and family member, all three forms) and per_key.npz (per-key G_loc, L, e_h(s*)
and s*). The 'readings' block holds the numbers E1-E4 are read from.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import io
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

from apexzero.fields import analysis as A
from apexzero.fields import measures as MS
from apexzero.fields import perturb as PT
from apexzero.fields import stages as S

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ROOT / "outputs" / "fields"
OUT = ROOT / "outputs" / "locality"
PLAN = ROOT / "docs" / "findings" / "locality-plan.md"
PLAN_COMMIT = "291128d"
DECISIONS = FIELDS / "analyse" / "decisions.json"
DECISIONS_SHA256 = "7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf"
FROZEN = ("src", "scripts/fields.py")
FROZEN_MODULES = ("keys", "measures", "perturb", "stages", "analysis")

SIGMA_B = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)            # family 1 (m)
RAMP_X = (5.0, 10.0, 20.0, 40.0, 80.0)                  # family 2 (m)
PLATEAU_L = (10.0, 20.0, 40.0, 80.0)                    # family 3 (m)
OFFSET_D = (-80.0, -60.0, -40.0, -20.0, -10.0, 0.0, 10.0, 20.0)   # family 4 (m)
OFFSET_SIGMA_B = 8.0
RAMP_LEN = PT.TRAPEZOID_OUTER - PT.TRAPEZOID_INNER      # 20 m, the trapezoid's ramps
CHECK_TOL = 1e-9          # the plan's required agreement with step 2's G
BASIS_TOL = 1e-9          # sum b = 1, b . trapezoid = e_star
S_STAR_TOL = 1e-6         # m, recovered s* against the optimum's row
C_MIN = S.NET_OUTCOMES.index("c_min")
SHORT = ("sigma_b=2", "sigma_b=4", "X=5", "X=10", "L=10", "L=20")   # E1 / E4's short scales
SHORT_WIDE = ("sigma_b=2", "sigma_b=4", "sigma_b=8", "sigma_b=16", "X=5", "X=10", "X=20",
              "X=40", "L=10", "L=20")                   # a wider reading of E4's short scales
FORMS = ("G_loc", "L", "L_per_e")
REAL_ROAD_HI = 0.2        # the plan's 'real-road values of about 0.1-0.2' (E1)
FAMILY_SCALES = {"bump": [f"sigma_b={g:g}" for g in SIGMA_B],
                 "ramp": [f"X={x:g}" for x in RAMP_X],
                 "plateau": [f"L={x:g}" for x in PLATEAU_L] + ["trapezoid"]}

MEMBERS: tuple[tuple[str, str, float | None], ...] = (
    *(("bump", f"sigma_b={g:g}", g) for g in SIGMA_B),
    *(("ramp", f"X={x:g}", x) for x in RAMP_X),
    *(("plateau", f"L={x:g}", x) for x in PLATEAU_L),
    ("plateau", "trapezoid", None),
    *(("offset", f"d={d:+g}", d) for d in OFFSET_D),
)
LABELS = tuple(m[1] for m in MEMBERS)
TRAP = LABELS.index("trapezoid")
RATIO = np.array([f != "offset" for f, _, _ in MEMBERS])   # members with e_h(s*) close to 1


class CheckFailed(SystemExit):
    """A built-in check failed: the run stops and writes nothing."""


# ------------------------------------------------------------------------- profiles


def ramp_profile(s, s_star: float, x: float, s_c1: float) -> np.ndarray:
    """Family 2: 0 before s* - x, linear to 1 at s*, flat to max(s*, s_c1 + 20), then down to
    0 over 20 m (the trapezoid's exit; module docstring)."""
    s = np.asarray(s, float)
    up = np.clip((s - (s_star - x)) / x, 0.0, 1.0)
    end = max(float(s_star), float(s_c1) + PT.TRAPEZOID_INNER)
    down = np.clip(((end + RAMP_LEN) - s) / RAMP_LEN, 0.0, 1.0)
    return np.minimum(up, down)


def plateau_profile(s, s_star: float, length: float) -> np.ndarray:
    """Family 3: 1 on [s* - L/2, s* + L/2] with 20 m ramps to 0 on each side."""
    half = 0.5 * float(length)
    return PT.trapezoid(s, s_star, s_star, outer=half + RAMP_LEN, inner=half)


def profiles(s, s_star: float, s_c0: float, s_c1: float) -> tuple[np.ndarray, np.ndarray]:
    """([M, N] sampled profiles of every member, [M] their height H at s*), in MEMBERS'
    order."""
    s = np.asarray(s, float)
    P = np.empty((len(MEMBERS), len(s)))
    H = np.ones(len(MEMBERS))
    for m, (fam, _, p) in enumerate(MEMBERS):
        if fam == "bump":
            P[m] = PT.gaussian_bump(s, s_star, p)
        elif fam == "ramp":
            P[m] = ramp_profile(s, s_star, p, s_c1)
        elif fam == "plateau" and p is None:
            P[m] = PT.trapezoid(s, s_c0, s_c1)
        elif fam == "plateau":
            P[m] = plateau_profile(s, s_star, p)
        else:
            P[m] = PT.gaussian_bump(s, s_star + p, OFFSET_SIGMA_B)
            H[m] = float(PT.gaussian_bump(0.0, p, OFFSET_SIGMA_B))
    return P, H


def s_star_from_basis(b, s) -> tuple[float, float]:
    """(s*, sum b - 1) from c_min's explicit inside-width row b (module docstring)."""
    b = np.asarray(b, float)
    tot = float(b.sum())
    return float(b @ np.asarray(s, float) / tot), tot - 1.0


def key_values(imp_in, b, s, s_star: float, s_c0: float, s_c1: float) -> dict[str, np.ndarray]:
    """One key's G_loc, L, L / e_h(s*) (families 1-3; NaN in family 4), H and e_h(s*) for every
    member: imp_in is c_min's inside-width impulse row, b its explicit row (the edge's own move
    at s*)."""
    P, H = profiles(s, s_star, s_c0, s_c1)
    dc = P @ np.asarray(imp_in, float)
    e = P @ np.asarray(b, float)
    ratio = np.full(len(MEMBERS), np.nan)
    ratio[RATIO] = (e - dc)[RATIO] / e[RATIO]
    return {"G_loc": H - dc, "L": e - dc, "L_per_e": ratio, "H": H, "e": e}


def inside_index(sigma: int) -> int:
    return PT.FEATURES.index(PT.side_feature(int(sigma), "inside"))


# ------------------------------------------------------------------------- per-key pass


def network_pass(name: str, rows: list[dict], arr_dir: Path, opt_rows: dict[str, dict],
                 kept: set[str]) -> tuple[dict, dict]:
    """Per kept key {key_id: key_values + s*}, and the checks over every row with arrays."""
    out, chk = {}, {"n_arrays": 0, "sha_mismatch": [], "n_trap_checked": 0,
                    "trap_max_abs": 0.0, "n_basis_checked": 0, "basis_sum_max_abs": 0.0,
                    "e_star_max_abs": 0.0, "other_features_max_abs": 0.0,
                    "s_star_beyond_plateau_max_m": 0.0, "n_s_star_beyond_s_c1_plus_20": 0}
    for r in rows:
        if r.get("pipeline"):
            continue
        key = r["key_id"]
        raw = (arr_dir / f"{key.replace('/', '_')}.npz").read_bytes()
        chk["n_arrays"] += 1
        if hashlib.sha256(raw).hexdigest() != r["arrays_sha256"]:
            chk["sha_mismatch"].append(key)
            continue
        with np.load(io.BytesIO(raw)) as z:
            imp, ex = z["impulse"][C_MIN], z["explicit_c_min"]
        o = opt_rows[key]
        fi = inside_index(r["direction"])
        s = PT.sample_s(imp.shape[-1])
        if r.get("G") is not None:
            g_trap = 1.0 - float(imp[fi] @ PT.trapezoid(s, o["s_c0"], o["s_c1"]))
            chk["trap_max_abs"] = max(chk["trap_max_abs"], abs(g_trap - float(r["G"])))
            chk["n_trap_checked"] += 1
        if key not in kept:
            continue
        b = ex[fi]
        s_star, dsum = s_star_from_basis(b, s)
        chk["n_basis_checked"] += 1
        chk["basis_sum_max_abs"] = max(chk["basis_sum_max_abs"], abs(dsum))
        e_trap = float(b @ PT.trapezoid(s, o["s_c0"], o["s_c1"]))
        chk["e_star_max_abs"] = max(chk["e_star_max_abs"], abs(e_trap - float(r["e_star"])))
        chk["other_features_max_abs"] = max(chk["other_features_max_abs"],
                                            float(np.abs(np.delete(ex, fi, 0)).max()))
        lo_p, hi_p = o["s_c0"] - PT.TRAPEZOID_INNER, o["s_c1"] + PT.TRAPEZOID_INNER
        chk["s_star_beyond_plateau_max_m"] = max(chk["s_star_beyond_plateau_max_m"],
                                                 lo_p - s_star, s_star - hi_p)
        chk["n_s_star_beyond_s_c1_plus_20"] += int(s_star > hi_p)
        out[key] = {**key_values(imp[fi], b, s, s_star, o["s_c0"], o["s_c1"]), "s_star": s_star}
    return out, chk


def optimum_pass(opt_a2: list[dict], arr_dir: Path) -> tuple[dict, dict, str]:
    """The optimum's per-key values on every A2 key (s* from its row), its checks, and a
    sha256 over the (key, file sha256) manifest of the arrays read."""
    out, chk = {}, {"n_keys": 0, "trap_max_abs": 0.0, "s_star_max_abs": 0.0,
                    "basis_sum_max_abs": 0.0}
    manifest = []
    for o in sorted(opt_a2, key=lambda r: r["key_id"]):
        if o.get("failed"):
            continue
        key = o["key_id"]
        raw = (arr_dir / f"{key.replace('/', '_')}.npz").read_bytes()
        manifest.append(f"{key} {hashlib.sha256(raw).hexdigest()}")
        with np.load(io.BytesIO(raw)) as z:
            imp, ex = z["impulse"][C_MIN], z["explicit"][C_MIN]
        fi = inside_index(o["direction"])
        s = PT.sample_s(imp.shape[-1])
        g_trap = 1.0 - float(imp[fi] @ PT.trapezoid(s, o["s_c0"], o["s_c1"]))
        chk["trap_max_abs"] = max(chk["trap_max_abs"], abs(g_trap - float(o["G_star"])))
        s_rec, dsum = s_star_from_basis(ex[fi], s)
        chk["s_star_max_abs"] = max(chk["s_star_max_abs"], abs(s_rec - float(o["s_star"])))
        chk["basis_sum_max_abs"] = max(chk["basis_sum_max_abs"], abs(dsum))
        chk["n_keys"] += 1
        out[key] = {**key_values(imp[fi], ex[fi], s, float(o["s_star"]), o["s_c0"], o["s_c1"]),
                    "s_star": float(o["s_star"])}
    digest = hashlib.sha256("\n".join(manifest).encode()).hexdigest()
    return out, chk, digest


# ------------------------------------------------------------------------- summaries


def summaries(km: list[MS.KeyMeasure], vals: dict[str, dict], n_boot: int) -> dict:
    """{member label: {form: summary}} (pooled ``agent_summary``), 'L_per_e' in families 1-3
    only."""
    out = {}
    for m, lab in enumerate(LABELS):
        out[lab] = {}
        for form in FORMS if RATIO[m] else FORMS[:2]:
            rows = [dataclasses.replace(k, value=(float(vals[k.key_id][form][m]) if not k.reason
                                                  else float("nan"))) for k in km]
            res = MS.agent_summary(rows, (1, -1), n_boot, MS.stream_rng("bootstrap")).as_dict()
            out[lab][form] = {k: res[k] for k in ("median", "lo", "hi", "n_geometries",
                                                  "n_defined", "n_pairs")}
    return out


def edge_read_gap(per_entry: dict[str, dict]) -> dict:
    """Per member, the distribution over kept network keys (distinct checkpoints) of
    H - e_h(s*): how far the arrays' spline of the sampled profile misses its height at s*."""
    gaps = np.array([v["H"] - v["e"] for per_key in per_entry.values()
                     for v in per_key.values()])
    if not len(gaps):
        return {}
    a = np.abs(gaps)
    return {lab: {"median": float(np.median(gaps[:, m])), "p95_abs": float(np.percentile(a[:, m], 95)),
                  "max_abs": float(a[:, m].max())} for m, lab in enumerate(LABELS)}


def role(name: str) -> str:
    if name in A.H2_NETWORKS:
        return "generalist"
    for v, d in A.H4_VARIANTS.items():
        if name in d["real"]:
            return f"real-width {v}"
        if name in d["baseline"]:
            return f"baseline {v}"
    if name in A.H4_TRANSFER["real"]:
        return "leave-out"
    return "other"


def two_sided(res: dict) -> float:
    return float(min(1.0, 2.0 * min(res["p"], res["p_opposite"])))


def e4(medians: dict[str, dict[str, float]]) -> dict:
    """Per member, H4's arms and the leave-out runs against the baselines (module
    docstring)."""
    out = {}
    for lab in LABELS:
        st = {n: medians[n][lab] for n in medians}
        groups = {v: ([st[r] for r in d["real"]],
                      [st[r] for r in d["baseline"] if not A.is_baseline_seed0(r)])
                  for v, d in A.H4_VARIANTS.items()}
        loo = {"transfer": ([st[r] for r in A.H4_TRANSFER["real"]],
                            [st[r] for r in A.H4_TRANSFER["baseline"]
                             if not A.is_baseline_seed0(r)])}
        row = {}
        for key, g in (("H4_arms", groups), ("leave_out", loo)):
            try:
                res = A.permutation_test(g)
            except ValueError as e:
                row[key] = {"error": str(e)}
                continue
            row[key] = {"T": res["T"], "mean_diff_per_stratum": res["T"] / len(g),
                        "p_two_sided": two_sided(res), "p_real_gt_base": res["p"],
                        "p_real_lt_base": res["p_opposite"], "n_relabellings": res["n_relabellings"],
                        "strata": {k: {"T": v["T"], "p_two_sided": two_sided(v)}
                                   for k, v in res["strata"].items()},
                        "real_range": [min(x for r, _ in g.values() for x in r),
                                       max(x for r, _ in g.values() for x in r)],
                        "baseline_range": [min(x for _, b in g.values() for x in b),
                                           max(x for _, b in g.values() for x in b)]}
        out[lab] = row
    return out


def line_forms(res_net: dict, distinct: list[str]) -> dict:
    """E1 in the three forms, families 1-3: per member the median and range over the distinct
    networks' medians, and the networks at or below the plan's real-road 0.2."""
    out = {}
    for form in FORMS:
        out[form] = {}
        for lab in (x for labs in FAMILY_SCALES.values() for x in labs):
            v = {n: res_net[n][lab][form]["median"] for n in distinct}
            a = np.array(list(v.values()))
            out[form][lab] = {"median_over_networks": float(np.median(a)),
                              "range": [float(a.min()), float(a.max())],
                              f"at_or_below_{REAL_ROAD_HI:g}": {
                                  n: x for n, x in sorted(v.items(), key=lambda t: t[1])
                                  if x <= REAL_ROAD_HI}}
    return out


def groups() -> dict[str, tuple[str, ...]]:
    """The note's groups: step 1's generalists, H4's baseline arms (seeds 1-4; four of them are
    generalists too), the real-width arms and the leave-out runs."""
    return {"generalists": tuple(A.H2_NETWORKS),
            "baselines": tuple(r for d in A.H4_VARIANTS.values() for r in d["baseline"]
                               if not A.is_baseline_seed0(r)),
            "real-width": tuple(r for d in A.H4_VARIANTS.values() for r in d["real"]),
            "leave-out": tuple(A.H4_TRANSFER["real"])}


def group_summaries(res_all: dict) -> dict:
    """Per group, member and form: the median and range of the networks' medians."""
    out = {}
    for g, names in groups().items():
        out[g] = {"networks": list(names)}
        for lab in LABELS:
            out[g][lab] = {}
            for form in FORMS:
                if form not in res_all[names[0]][lab]:
                    continue
                a = np.array([res_all[n][lab][form]["median"] for n in names])
                out[g][lab][form] = {"median": float(np.median(a)),
                                     "range": [float(a.min()), float(a.max())]}
    return out


def e4_span(e4_res: dict, labs: tuple[str, ...]) -> dict:
    """E4 over a set of members: the ranges of the mean difference, the per-variant
    differences and the two-sided p, for H4's arms and the leave-out runs."""
    out = {"members": list(labs)}
    for key in ("H4_arms", "leave_out"):
        rows = [e4_res[x][key] for x in labs]
        per_var = [v["T"] for r in rows for v in r["strata"].values()]
        out[key] = {"mean_diff_range": [min(r["mean_diff_per_stratum"] for r in rows),
                                        max(r["mean_diff_per_stratum"] for r in rows)],
                    "per_stratum_diff_range": [min(per_var), max(per_var)],
                    "p_two_sided_range": [min(r["p_two_sided"] for r in rows),
                                          max(r["p_two_sided"] for r in rows)]}
    return out


def readings(res_net: dict, opt: dict, opt_vals: dict, distinct: list[str],
             e4_res: dict) -> dict:
    """The numbers E1-E4 are read from (exploratory; the words are in the findings note)."""
    med = {n: {lab: res_net[n][lab]["G_loc"]["median"] for lab in LABELS} for n in res_net}
    r: dict = {"distinct_networks": len(distinct)}
    fam = {}
    for f, labs in FAMILY_SCALES.items():
        seqs = {n: [med[n][x] for x in labs] for n in distinct}
        mono = [n for n, q in seqs.items() if all(b >= a for a, b in itertools.pairwise(q))]
        first = np.array([q[0] for q in seqs.values()])
        last = np.array([q[-1] for q in seqs.values()])
        fam[f] = {"scales": labs, "n_monotone_nondecreasing": len(mono),
                  "not_monotone": sorted(set(distinct) - set(mono)),
                  "shortest_scale_range": [float(first.min()), float(first.max())],
                  "shortest_scale_median": float(np.median(first)),
                  "longest_scale_range": [float(last.min()), float(last.max())],
                  "median_over_networks": {x: float(np.median([med[n][x] for n in distinct]))
                                           for x in labs},
                  "n_at_or_below_0.2": {x: int(sum(med[n][x] <= 0.2 for n in distinct))
                                        for x in labs}}
    r["E1"] = fam
    r["E1_forms"] = line_forms(res_net, distinct)
    f13 = [lab for labs in FAMILY_SCALES.values() for lab in labs]
    i13 = [LABELS.index(x) for x in f13]
    i4 = [m for m, (f, _, _) in enumerate(MEMBERS) if f == "offset"]
    og = np.array([v["G_loc"] for v in opt_vals.values()])
    oh = np.array([v["H"] for v in opt_vals.values()])
    r["E2"] = {"optimum_median_range_families_1_3": [
        min(opt[x]["G_loc"]["median"] for x in f13), max(opt[x]["G_loc"]["median"] for x in f13)],
        "optimum_ci_range_families_1_3": [min(opt[x]["G_loc"]["lo"] for x in f13),
                                          max(opt[x]["G_loc"]["hi"] for x in f13)],
        "optimum_family4_median": {x: opt[x]["G_loc"]["median"] for x in LABELS
                                   if x.startswith("d=")},
        "family4_height_at_s_star": {x: float(PT.gaussian_bump(0.0, p, OFFSET_SIGMA_B))
                                     for f, x, p in MEMBERS if f == "offset"},
        "optimum_per_key_G_loc_range_families_1_3": [float(og[:, i13].min()),
                                                     float(og[:, i13].max())],
        "optimum_per_key_max_abs_G_loc_minus_H_family4": float(
            np.abs(og[:, i4] - oh[:, i4]).max()),
        "optimum_L_median_families_1_3": {x: opt[x]["L"]["median"] for x in f13},
        "n_optimum_keys_per_key": len(opt_vals)}
    offs = [x for f, x, _ in MEMBERS if f == "offset"]
    pre = [x for x in offs if float(x[2:]) < 0]
    post = [x for x in offs if float(x[2:]) >= 0]
    arg = {n: max(offs, key=lambda x, n=n: med[n][x]) for n in distinct}
    amin = {n: min(offs, key=lambda x, n=n: med[n][x]) for n in distinct}
    r["E3"] = {"argmax_d": arg,
               "n_argmax_before_s_star": int(sum(float(a[2:]) < 0 for a in arg.values())),
               "n_max_before_exceeds_max_at_or_after": int(sum(
                   max(med[n][x] for x in pre) > max(med[n][x] for x in post) for n in distinct)),
               "n_before_gt_mirror_after": {f"d=-{a:g} vs d=+{a:g}": int(sum(
                   med[n][f"d=-{a:g}"] > med[n][f"d=+{a:g}"] for n in distinct))
                   for a in (10.0, 20.0)},
               "argmin_d": amin,
               "n_argmin_before_s_star": int(sum(float(a[2:]) < 0 for a in amin.values())),
               "n_below_d=+20": {x: int(sum(med[n][x] < med[n]["d=+20"] for n in distinct))
                                 for x in ("d=-80", "d=-60", "d=-40")},
               "median_over_networks": {x: float(np.median([med[n][x] for n in distinct]))
                                        for x in offs}}
    r["E4"] = {x: {k: {kk: v.get(kk) for kk in ("mean_diff_per_stratum", "p_two_sided",
                                                  "real_range", "baseline_range")}
                       for k, v in e4_res[x].items()} for x in LABELS}
    ps = [(x, k, v["p_two_sided"]) for x in LABELS for k, v in e4_res[x].items()
          if "p_two_sided" in v]
    r["E4_spans"] = {"short": e4_span(e4_res, SHORT), "short_wide": e4_span(e4_res, SHORT_WIDE),
                     "n_tests": len(ps),
                     "p_two_sided_below_0.05": [f"{x} {k} {p:.4f}" for x, k, p in ps if p < 0.05]}
    return r


# ------------------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def frozen_problems() -> list[str]:
    changed = git("status", "--porcelain", "--", *FROZEN)
    return [ln for ln in changed.splitlines() if ln.strip()]


# ------------------------------------------------------------------------- main


def clean(x):
    """JSON-safe: NaN and infinities as null, NumPy scalars as Python numbers."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise CheckFailed(1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=MS.BOOTSTRAP_N)
    args = ap.parse_args(argv)
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%S")
    if probs := frozen_problems():
        fail(f"frozen code differs from HEAD: {probs}")
    raw_dec = DECISIONS.read_bytes()
    if hashlib.sha256(raw_dec).hexdigest() != DECISIONS_SHA256:
        fail("decisions.json is not the committed confirmatory run")
    dec = json.loads(raw_dec)

    opt_rows = {r["key_id"]: r for r in S.read_rows(FIELDS / "opt-fields" / "rows.jsonl")}
    net_rows, net_dirs, inputs = {}, {}, {}
    for name in A.A2_NETWORKS:
        p = FIELDS / "fields-a2" / f"{name.replace('/', '_')}.jsonl"
        net_rows[name] = S.read_rows(p)
        net_dirs[name] = S.arrays_dir(FIELDS / "fields-a2" / "arrays", net_rows[name][0]["net"])
        inputs[rel(p)] = sha256_file(p)
    inputs[rel(FIELDS / "opt-fields" / "rows.jsonl")] = sha256_file(
        FIELDS / "opt-fields" / "rows.jsonl")
    inputs[rel(DECISIONS)] = DECISIONS_SHA256

    # ---- networks: step 2's keys and exclusions, then the per-key pass
    km, vals, checks = {}, {}, {}
    for name in A.A2_NETWORKS:
        km[name] = S.edge_key_measures(net_rows[name], opt_rows)
        kept = {k.key_id for k in km[name] if not k.reason}
        vals[name], checks[name] = network_pass(name, net_rows[name], net_dirs[name], opt_rows,
                                                kept)
        if kept - set(vals[name]):
            fail(f"{name}: kept keys without values {sorted(kept - set(vals[name]))[:3]}")
        for k in km[name]:          # the kept keys' G is the row's G: G_loc(trapezoid) equals it
            if not k.reason and abs(vals[name][k.key_id]["G_loc"][TRAP] - k.value) > CHECK_TOL:
                fail(f"{name} {k.key_id}: G_loc(trapezoid) {vals[name][k.key_id]['G_loc'][TRAP]}"
                     f" != step 2's G {k.value}")
        c = checks[name]
        print(f"{name}: {len(kept)} kept of {len(km[name])}; trapezoid max |G_loc - G| "
              f"{c['trap_max_abs']:.1e} over {c['n_trap_checked']} keys; "
              f"{time.time() - t0:.0f} s", flush=True)
        if c["sha_mismatch"]:
            fail(f"{name}: arrays differ from their rows' sha256: {c['sha_mismatch'][:3]}")
        if c["trap_max_abs"] > CHECK_TOL:
            fail(f"{name}: G_loc(trapezoid) differs from step 2's per-key G by "
                 f"{c['trap_max_abs']:.3e} > {CHECK_TOL}")
        if max(c["basis_sum_max_abs"], c["e_star_max_abs"], c["other_features_max_abs"]) \
                > BASIS_TOL:
            fail(f"{name}: s* recovery check {c}")

    # ---- the optimum (stages.optimum_edges' keys)
    opt_all = [r for r in opt_rows.values() if r.get("kind") != "anchored"]
    opt_vals, opt_chk, opt_manifest = optimum_pass(opt_all, FIELDS / "opt-fields" / "arrays")
    if opt_chk["trap_max_abs"] > CHECK_TOL:
        fail(f"optimum: G_loc(trapezoid) differs from G* by {opt_chk['trap_max_abs']:.3e}")
    if opt_chk["s_star_max_abs"] > S_STAR_TOL or opt_chk["basis_sum_max_abs"] > BASIS_TOL:
        fail(f"optimum: s* recovery check {opt_chk}")
    opt_a2 = [r for r in opt_all if r.get("w_is_cell") and not r.get("cap_binds")]
    opt_km = [A.key_measure(S._meta_of(r), float(r["G_star"]) if not r.get("failed")
                            else float("nan"), {"numerical": bool(r.get("failed"))})
              for r in opt_a2]
    inputs["outputs/fields/opt-fields/arrays (A2 keys read: key sha256 manifest)"] = opt_manifest
    print(f"optimum: {opt_chk['n_keys']} A2 keys, {len(opt_km)} primary non-capped; trapezoid "
          f"max |G_loc - G*| {opt_chk['trap_max_abs']:.1e}; {time.time() - t0:.0f} s", flush=True)

    # ---- summaries
    res_net = {}
    for name in A.A2_NETWORKS:
        res_net[name] = summaries(km[name], vals[name], args.n_boot)
    opt_sum = summaries(opt_km, opt_vals, args.n_boot)
    print(f"summaries done; {time.time() - t0:.0f} s", flush=True)

    # ---- check 2: the trapezoid's summaries reproduce decisions.json
    dec_check = {"n_boot": args.n_boot, "max_abs": 0.0, "compared": 0}
    if args.n_boot == MS.BOOTSTRAP_N:
        for name in A.A2_NETWORKS:
            ref = dec["edge_descriptive"][name]["G"]["pooled"]
            got = res_net[name]["trapezoid"]["G_loc"]
            for k in ("median", "lo", "hi"):
                dec_check["max_abs"] = max(dec_check["max_abs"], abs(got[k] - ref[k]))
            dec_check["compared"] += 1
        ref = dec["optimum"]["G_star"]["pooled"]
        got = opt_sum["trapezoid"]["G_loc"]
        dec_check["optimum_max_abs"] = max(abs(got[k] - ref[k]) for k in ("median", "lo", "hi"))
        if max(dec_check["max_abs"], dec_check["optimum_max_abs"]) > CHECK_TOL:
            fail(f"trapezoid summaries differ from decisions.json: {dec_check}")
    else:
        dec_check["skipped"] = "n_boot differs from step 2's"

    # ---- distinct checkpoints, readings
    seen, distinct, dup = {}, [], {}
    for name in A.A2_NETWORKS:
        ck = net_rows[name][0]["ckpt_sha256"]
        if ck in seen:
            dup[name] = seen[ck]
        else:
            seen[ck] = name
            distinct.append(name)
    med = {n: {lab: res_net[n][lab]["G_loc"]["median"] for lab in LABELS} for n in A.A2_NETWORKS}
    e4_res = e4(med)
    opt_used = {k: opt_vals[k] for k in (r["key_id"] for r in opt_a2) if k in opt_vals}
    read = readings({n: res_net[n] for n in distinct}, opt_sum, opt_used, distinct, e4_res)
    read["groups"] = group_summaries(res_net)
    gap = edge_read_gap({n: vals[n] for n in distinct})
    opt_gap = edge_read_gap({"optimum": opt_used})

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    result = {
        "status": "EXPLORATORY: every number here is exploratory (locality-plan.md)",
        "plan": {"path": rel(PLAN), "commit": git("log", "-1", "--format=%H", "--", rel(PLAN)),
                 "expected_commit_prefix": PLAN_COMMIT, "sha256": sha256_file(PLAN)},
        "members": [{"family": f, "label": x, "param": p,
                     "height_at_s_star": (float(PT.gaussian_bump(0.0, p, OFFSET_SIGMA_B))
                                          if f == "offset" else 1.0)} for f, x, p in MEMBERS],
        "checks": {
            "trapezoid_vs_step2_G_per_key": {
                "tol": CHECK_TOL, "pass": True,
                "max_abs": max(c["trap_max_abs"] for c in checks.values()),
                "n_rows_checked": int(sum(c["n_trap_checked"] for c in checks.values())),
                "optimum_max_abs": opt_chk["trap_max_abs"],
                "optimum_rows_checked": opt_chk["n_keys"]},
            "trapezoid_summary_vs_decisions": dec_check | {"pass": True},
            "arrays_sha256_vs_rows": {"n_files": int(sum(c["n_arrays"] for c in checks.values())),
                                      "n_mismatch": 0},
            "s_star_recovery": {
                "networks_basis_sum_max_abs": max(c["basis_sum_max_abs"] for c in checks.values()),
                "networks_e_star_max_abs": max(c["e_star_max_abs"] for c in checks.values()),
                "networks_other_features_max_abs": max(c["other_features_max_abs"]
                                                       for c in checks.values()),
                "optimum_s_star_vs_row_max_abs_m": opt_chk["s_star_max_abs"],
                "optimum_basis_sum_max_abs": opt_chk["basis_sum_max_abs"],
                "networks_s_star_beyond_trapezoid_plateau_max_m": max(
                    c["s_star_beyond_plateau_max_m"] for c in checks.values()),
                "networks_n_s_star_beyond_s_c1_plus_20": int(sum(
                    c["n_s_star_beyond_s_c1_plus_20"] for c in checks.values()))},
            "per_network": checks},
        "edge_read_gap_H_minus_e": {"networks": gap, "optimum": opt_gap},
        "networks": {n: {"role": role(n), "net": net_rows[n][0]["net"],
                         "ckpt_sha256": net_rows[n][0]["ckpt_sha256"],
                         "duplicate_of": dup.get(n), "n_keys": len(km[n]),
                         "n_kept": int(sum(not k.reason for k in km[n])),
                         "members": res_net[n]} for n in A.A2_NETWORKS},
        "optimum": {"n_keys": len(opt_km), "n_kept": int(sum(not k.reason for k in opt_km)),
                    "members": opt_sum},
        "E4": e4_res,
        "readings": read,
        "provenance": {
            "script": rel(script), "script_sha256": sha256_file(script),
            "git_head": git("rev-parse", "HEAD"), "frozen_code_clean": True,
            "frozen_modules_sha256": {m: sha256_file(ROOT / "src" / "apexzero" / "fields"
                                                     / f"{m}.py") for m in FROZEN_MODULES},
            "inputs_sha256": inputs, "argv": [rel(script), *(argv or sys.argv[1:])],
            "n_boot": args.n_boot, "start": start, "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "seconds": round(time.time() - t0, 1),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system()}},
    }
    (OUT / "results.json").write_text(json.dumps(clean(result), indent=1, allow_nan=False))
    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "role", "duplicate_of", "family", "member", "n_geometries",
                    "n_pairs_defined",
                    *(f"{form}_{k}" for form in FORMS for k in ("median", "lo", "hi"))])
        entries = [(n, role(n), dup.get(n) or "", res_net[n]) for n in A.A2_NETWORKS]
        entries.append(("optimum", "optimum", "", opt_sum))
        for n, rl, dp, res in entries:
            for f, lab, _ in MEMBERS:
                g = res[lab]["G_loc"]
                w.writerow([n, rl, dp, f, lab, g["n_geometries"], g["n_defined"],
                            *(f"{res[lab][form][k]:.4f}" if form in res[lab] else ""
                              for form in FORMS for k in ("median", "lo", "hi"))])
    per_key = {}
    for n, vv in [*((n, vals[n]) for n in distinct), ("optimum", opt_vals)]:
        ks = sorted(vv)
        tag = n.replace("/", "_")
        per_key[f"{tag}__key_id"] = np.array(ks)
        per_key[f"{tag}__G_loc"] = np.array([vv[k]["G_loc"] for k in ks])
        per_key[f"{tag}__L"] = np.array([vv[k]["L"] for k in ks])
        per_key[f"{tag}__e"] = np.array([vv[k]["e"] for k in ks])
        per_key[f"{tag}__s_star"] = np.array([vv[k]["s_star"] for k in ks])
    per_key["members"] = np.array(LABELS)
    np.savez_compressed(OUT / "per_key.npz", **per_key)
    print(f"checks passed; wrote {rel(OUT / 'results.json')}, results.csv, per_key.npz "
          f"({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
