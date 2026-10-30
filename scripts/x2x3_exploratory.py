"""X-2 and X-3, locality at G2's real R-dev crash corners: EXPLORATORY (docs/findings/x2-plan.md,
committed in 8d81ef2 before anything here was computed). Every number this script writes is
exploratory.

    uv run python scripts/x2x3_exploratory.py

At the 12 unit-phases of step 2's H3 (cells 584, 605 and 612 x phases 0-3: Norisring 874,
Melbourne 2531 and Catalunya 4168, all R-dev), with H3's own start, road, width change and
512-step loop (``stages.h3_lambda``), G2's base run, one VJP of c_min (``network.fields``) and
the explicit term (``network.explicit_c_min``) give:

- **X-2 (first order):** the real inside-width change's L/e (H3's inside-only ratio),
  ``locality.md``'s 24 profiles in its three forms (``scripts/locality_exploratory.py``'s own
  functions, s* from the explicit row's basis, the start's landmarks for the ramp's end and the
  trapezoid), and the bump sigma_b whose L/e equals the real change's;
- **X-3 (finite amplitude):** closed-loop runs (``mode="secant"``) with the inside width
  changed by A h for the sigma_b = 4 m bump, the X = 5 m ramp and the trapezoid,
  A in {-1, -0.5, -0.2, +0.2, +0.5, +1} m, and the finite L/e(A) against the first-order one.

``unit_phase`` is reused by GW-5's analysis (docs/findings/gw5-plan.md) for other networks.
The frozen code is imported read-only; the script stops and writes nothing if a check fails.
Output: outputs/x2x3/results.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import locality_exploratory as LX  # locality.md's profiles and forms, read-only
import numpy as np

from apexzero.fields import analysis as A
from apexzero.fields import keys as K
from apexzero.fields import network as N
from apexzero.fields import perturb as PT
from apexzero.fields import stages as S

ROOT = LX.ROOT
H3_JSON = ROOT / "outputs" / "fields" / "h3-lambda" / "h3.json"
PREREG = ROOT / "src" / "apexzero" / "fields" / "prereg.json"
PLAN = ROOT / "docs" / "findings" / "x2-plan.md"
PLAN_COMMIT = "8d81ef2"
OUT = ROOT / "outputs" / "x2x3"
STEP2_TAG = "receptive-fields-prereg-v1"
FROZEN = ("src", "scripts/fields.py", "scripts/locality_exploratory.py", "scripts/_bootstrap.py")
AMPS = (-1.0, -0.5, -0.2, 0.2, 0.5, 1.0)
X3_PROFILES = ("sigma_b=4", "X=5", "trapezoid")
SHORT = ("sigma_b=2", "sigma_b=4")
SIGMA_LABELS = tuple(f"sigma_b={g:g}" for g in LX.SIGMA_B)
REL, ABS = 1e-6, 1e-9


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise LX.CheckFailed(1)


def unlock() -> None:
    """Check 4, then ``network.unlock_trained`` (as scripts/costate_exploratory.py does) and
    x64 on (the float64 loop, as scripts/fields.py's post-tag stages switch it)."""
    import jax

    changed = git("status", "--porcelain", "--", *FROZEN)
    if changed:
        fail(f"frozen code differs from HEAD: {changed.splitlines()}")
    if not git("rev-parse", "-q", "--verify", f"refs/tags/{STEP2_TAG}"):
        fail(f"step 2's tag {STEP2_TAG} is not here")
    jax.config.update("jax_enable_x64", True)
    N.unlock_trained()


def load_net(name: str, pin: dict):
    return S.load_network(name, pin, ROOT)


def design():
    bank, cells = K.design()
    return bank, cells, [(int(c), k) for c in A.H3_CELLS for k in range(cells[int(c)].n_phases)]


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(ABS, REL * max(abs(a), abs(b)))


def equivalent_sigma(le: dict[str, float], target: float) -> float:
    """The bump sigma_b whose L/e equals ``target``: log-linear between grid points at the
    first crossing; -inf below the grid's first value, +inf above its last."""
    xs = np.log(np.asarray(LX.SIGMA_B))
    ys = np.asarray([le[x] for x in SIGMA_LABELS])
    if target < ys[0]:
        return float("-inf")
    for i in range(len(ys) - 1):
        lo, hi = sorted((ys[i], ys[i + 1]))
        if lo <= target <= hi and ys[i + 1] != ys[i]:
            return float(np.exp(xs[i] + (target - ys[i]) * (xs[i + 1] - xs[i])
                                / (ys[i + 1] - ys[i])))
    return float("inf")


def unit_phase(net, bank, cells, cid: int, phase: int, steps: int = S.H3_STEPS) -> dict:
    """X-2's first-order measures at one unit-phase, and the context X-3 needs (``_ctx``)."""
    pairs = A.h3_pairs(bank, cells, (cid,))
    st = N.cell_start(cells[cid], phase, bank)
    rd = N.road(bank, st.stim_id)
    N.road(bank, pairs[cid])                       # the twin through step 1's guard too
    dw = A.width_change(bank, st.stim_id, pairs[cid])
    base = N.run(net, rd, st, steps=steps)
    out = {"cell": cid, "phase": phase, "net": net.name, "sigma": int(st.sigma)}
    if base.exhausted:
        return {**out, "pipeline": "loop exhausted"}
    if not base.plans["c_min"].valid:
        return {**out, "pipeline": "c_min invalid"}
    fl = N.fields(net, rd, base, steps)
    ex = N.explicit_c_min(rd, base)
    imp = fl.of("c_min")
    side = PT.side_feature(int(st.sigma), "inside")
    other = "w_left" if side == "w_right" else "w_right"
    imp_in, b = np.asarray(getattr(imp, side), float), np.asarray(getattr(ex, side), float)
    s = PT.sample_s(len(b))
    s_star, dsum = LX.s_star_from_basis(b, s)
    lm = st.landmarks
    kv = LX.key_values(imp_in, b, s, s_star, lm.s_c0, lm.s_c1)
    h = np.asarray(getattr(dw, side), float)
    e_real, dc_real = float(b @ h), float(imp_in @ h)
    dc = {f: float(np.asarray(getattr(imp, f), float) @ np.asarray(getattr(dw, f), float))
          for f in ("w_left", "w_right")}
    edge = sum(float(np.asarray(getattr(ex, f), float) @ np.asarray(getattr(dw, f), float))
               for f in ("w_left", "w_right"))
    forms = {form: {lab: float(kv[form][m]) for m, lab in enumerate(LX.LABELS)
                    if np.isfinite(kv[form][m])} for form in LX.FORMS}
    le_real = (e_real - dc_real) / e_real
    eq = equivalent_sigma(forms["L_per_e"], le_real)
    eq_label = (f"below {LX.SIGMA_B[0]:g} m" if eq == float("-inf")
                else f"above {LX.SIGMA_B[-1]:g} m" if eq == float("inf") else f"{eq:.2f} m")
    return {**out, "inside": side, "equivalent_sigma_b_label": eq_label, "s_star": s_star, "s_c0": float(lm.s_c0),
            "s_c1": float(lm.s_c1), "c_min0": float(fl.values[0]),
            "base_c_min": float(base.values()[0]), "e_real": e_real, "dc_real": dc_real,
            "L_per_e_real": le_real, "dc": dc, "edge_move": edge,
            "equivalent_sigma_b_m": equivalent_sigma(forms["L_per_e"], le_real),
            "forms": forms, "basis_sum_err": dsum,
            "explicit_other_max_abs": float(np.abs(np.asarray(getattr(ex, other))).max()),
            "_ctx": {"rd": rd, "st": st, "imp_in": imp_in, "b": b, "s": s, "side": side,
                     "steps": steps}}


def finite(net, row: dict) -> dict:
    """X-3 at one unit-phase: the zero change and every profile x amplitude, closed loop."""
    c = row["_ctx"]
    n = len(c["s"])
    zero = N.run(net, c["rd"], c["st"], PT.zeros(n), c["steps"], mode="secant")
    c0 = float(zero.values()[0])
    P, _ = LX.profiles(c["s"], row["s_star"], row["s_c0"], row["s_c1"])
    out = {"c_min0_secant": c0, "profiles": {}}
    for lab in X3_PROFILES:
        h = P[LX.LABELS.index(lab)]
        e_h = float(c["b"] @ h)
        lin = row["forms"]["L_per_e"][lab]
        runs = []
        for amp in AMPS:
            d = {"kappa": np.zeros(n), "w_left": np.zeros(n), "w_right": np.zeros(n)}
            d[c["side"]] = amp * h
            r = N.run(net, c["rd"], c["st"], PT.Samples(**d), c["steps"], mode="secant")
            crashed = bool(r.term_step >= 0 and not r.plans["c_min"].valid)
            valid = bool(r.plans["c_min"].valid and not r.exhausted)
            cm = float(r.values()[0]) if valid else float("nan")
            le = (e_h * amp - (cm - c0)) / (e_h * amp) if valid else float("nan")
            # added after the first run (a diagnostic, not a measure): where the closest point
            # lies, to see whether it moves away from s* when the edge widens locally
            s_at = float(r.plans["c_min"].s_at) if valid else float("nan")
            runs.append({"A": amp, "crashed": crashed, "exhausted": bool(r.exhausted),
                         "valid": valid, "c_min": cm, "L_per_e": le,
                         "diff": le - lin if valid else float("nan"),
                         "s_at": s_at, "argmin_shift_m": s_at - row["s_star"]})
        out["profiles"][lab] = {"e_h": e_h, "L_per_e_lin": lin, "runs": runs}
    return out


def med(x) -> float:
    a = np.asarray([v for v in x if np.isfinite(v)], float)
    return float(np.median(a)) if len(a) else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args(argv)
    t0 = time.time()
    start = time.strftime("%Y-%m-%dT%H:%M:%S")
    unlock()
    pin = json.loads(PREREG.read_text())["networks"][A.H3_AGENT]
    net = load_net(A.H3_AGENT, pin)
    h3 = {(r["cell"], r["phase"]): r for r in json.loads(H3_JSON.read_text())}
    bank, cells, ups = design()
    rows = []
    for cid, k in ups:
        r = unit_phase(net, bank, cells, cid, k)
        if "pipeline" in r:
            fail(f"cell {cid} phase {k}: {r['pipeline']}")
        ref = h3[(cid, k)]
        for got, want in ((r["dc"]["w_left"], ref["dc_left"]), (r["dc"]["w_right"], ref["dc_right"]),
                          (r["edge_move"], ref["edge_move"])):
            if not close(got, float(want)):
                fail(f"cell {cid} phase {k}: {got} does not reproduce h3.json's {want}")
        if abs(r["basis_sum_err"]) > ABS or r["explicit_other_max_abs"] > ABS:
            fail(f"cell {cid} phase {k}: explicit row check ({r['basis_sum_err']:.2e}, "
                 f"{r['explicit_other_max_abs']:.2e})")
        r["x3"] = finite(net, r)
        if not close(r["x3"]["c_min0_secant"], r["base_c_min"]):
            fail(f"cell {cid} phase {k}: the zero change's c_min {r['x3']['c_min0_secant']} "
                 f"!= the base run's {r['base_c_min']}")
        r.pop("_ctx")
        rows.append(r)
        print(f"cell {cid} phase {k}: L/e real {r['L_per_e_real']:+.3f}, trapezoid G "
              f"{r['forms']['G_loc']['trapezoid']:.3f}; {time.time() - t0:.0f} s", flush=True)

    # ---- X-2 summaries and reading
    def per_unit(fn):
        return {str(cid): med(fn(r) for r in rows if r["cell"] == cid) for cid in A.H3_CELLS}

    x2 = {"L_per_e_real": {"median": med(r["L_per_e_real"] for r in rows),
                           "per_unit": per_unit(lambda r: r["L_per_e_real"])},
          "equivalent_sigma_b_m": {"median": med(r["equivalent_sigma_b_m"] for r in rows),
                                   "values": [r["equivalent_sigma_b_m"] for r in rows]},
          "profiles": {form: {lab: {"median": med(r["forms"][form][lab] for r in rows),
                                    "per_unit": per_unit(lambda r, f=form, x=lab:
                                                         r["forms"][f][x])}
                              for lab in LX.LABELS if lab in rows[0]["forms"][form]}
                       for form in LX.FORMS}}
    T = x2["profiles"]["G_loc"]["trapezoid"]["median"]
    Sv = med(r["forms"]["L_per_e"][x] for r in rows for x in SHORT)
    x2["reading"] = {"T_median_trapezoid_G": T, "S_median_short_bump_L_per_e": Sv,
                     "reading": ("locality explains the low real-road ratio" if T >= 0.5 and Sv <= 0.2
                                 else "the corners or states explain it" if T < 0.3 else "mixed")}

    # ---- X-3 summaries and reading
    x3 = {}
    for lab in X3_PROFILES:
        by = {}
        for amp in AMPS:
            runs = [next(u for u in r["x3"]["profiles"][lab]["runs"] if u["A"] == amp) for r in rows]
            lins = [r["x3"]["profiles"][lab]["L_per_e_lin"] for r in rows]
            by[f"{amp:+g}"] = {
                "median_diff": med(u["diff"] for u in runs),
                "median_abs_diff": med(abs(u["diff"]) for u in runs),
                "median_ratio": med(u["L_per_e"] / li for u, li in zip(runs, lins, strict=True)
                                    if u["valid"] and li != 0),
                "n_valid": sum(u["valid"] for u in runs),
                "n_crashed": sum(u["crashed"] for u in runs),
                "median_abs_argmin_shift_m": med(abs(u["argmin_shift_m"]) for u in runs),
                "n_argmin_moved_over_5m": sum(bool(abs(u["argmin_shift_m"]) > 5.0)
                                              for u in runs if u["valid"])}
        small = max(by[f"{a:+g}"]["median_abs_diff"] for a in AMPS if abs(a) <= 0.5)
        big = max(by[f"{a:+g}"]["median_abs_diff"] for a in AMPS if abs(a) == 1.0)
        holds = small <= 0.05 and big <= 0.10
        x3[lab] = {"by_amplitude": by, "max_median_abs_diff_small": small,
                   "max_median_abs_diff_1m": big,
                   "reading": "first order holds" if holds else "first order fails"}

    OUT.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    result = {
        "status": "EXPLORATORY: every number here is exploratory (x2-plan.md)",
        "plan": {"path": LX.rel(PLAN), "commit": git("log", "-1", "--format=%H", "--", LX.rel(PLAN)),
                 "expected_commit_prefix": PLAN_COMMIT},
        "network": {"name": net.name, "pin": pin}, "x2": x2, "x3": x3, "rows": rows,
        "checks": {"h3_reproduced": True, "explicit_rows_ok": True, "zero_change_ok": True,
                   "max_basis_sum_err": max(abs(r["basis_sum_err"]) for r in rows),
                   "max_explicit_other": max(r["explicit_other_max_abs"] for r in rows)},
        "inputs_sha256": {LX.rel(p): LX.sha256_file(p) for p in (H3_JSON, PREREG)},
        "provenance": {"script": LX.rel(script), "script_sha256": LX.sha256_file(script),
                       "git_head": git("rev-parse", "HEAD"), "start": start,
                       "seconds": round(time.time() - t0, 1)}}
    (OUT / "results.json").write_text(json.dumps(LX.clean(result), indent=1) + "\n")
    print(json.dumps(LX.clean({"x2": x2["reading"], "x3": {k: v["reading"] for k, v in x3.items()}}),
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
