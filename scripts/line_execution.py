"""The line / execution split on the lockbox laps (docs/findings/line-execution-plan.md): POST HOC.

    uv run --no-sync python scripts/line_execution.py solve --root outputs/lockbox --workers 4
    uv run --no-sync python scripts/line_execution.py analyse --root outputs/lockbox

Descriptive. For each completed original circuit of each subject, the lap's minimum-time
problem is solved again with the lateral offset held within eps of the car's own line (one
representative run): T_eps is the fastest lap within eps of that line, and
R(eps) = (T_a - T_eps) / (T_a - T*) is the share of the car's lost time that better driving
recovers there. Everything is read from the run's own root (``analysis.json``, ``corners.csv``,
``ocp/`` and ``laps/``, each data file checked against its record's sha256); no track file is
opened. ``solve`` is resumable and never overwrites; ``analyse`` writes ``split.json`` once.
Run it from the repository root (its paths are relative).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import csv
import io
import json
import multiprocessing
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import lockbox as LB  # the harness's helpers; sets the BLAS / XLA thread counts first
import numpy as np

SCRIPT = Path(__file__).resolve()
ROOTS = {"lockbox": "R_TEST, held out", "lb1-rdev-e0": "R-dev, in-sample for widths"}
OUT = Path("outputs/posthoc/line-execution")
EPS = (0.05, 0.25, 0.5, 1.0, 2.0)
CORNER_EPS = (0.05, 0.5)
TYPES = ("slow", "medium", "fast")
REL_TOL = 1e-4          # the consistency checks' slack (relative)
CLOSURE_LIMIT = 1e-6    # s
NICE = 15
MAX_WORKERS = 4


def provenance(t_start: float) -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True,
                          text=True, cwd=SCRIPT.parent).stdout.strip()
    return {"script": "scripts/line_execution.py", "script_sha256": LB.sha256_file(SCRIPT),
            "git_head": head, "seconds": round(time.time() - t_start, 1)}


def check_root(root: Path) -> str:
    if root.name not in ROOTS or not (root / "analysis.lock").exists():
        raise SystemExit(f"{root}: not an analysed lockbox root ({', '.join(ROOTS)})")
    return ROOTS[root.name]


def completed_rows(root: Path) -> tuple[dict, list[dict]]:
    """The rows behind the headline median gap: originals with ``circuit_completed``."""
    an = json.loads((root / "analysis.json").read_text())
    return an, [r for r in an["rows"] if r["variant"] == "original" and r["circuit_completed"]]


def reference(root: Path, circuit: str):
    key = f"{circuit}__w+0.0"
    rec = json.loads((root / "ocp" / f"{key}.json").read_text())
    raw = (root / "ocp" / f"{key}.pkl").read_bytes()
    if LB.sha256_bytes(raw) != rec["pkl_sha256"]:
        raise SystemExit(f"{root}/ocp/{key}.pkl does not match its record")
    return pickle.loads(raw), rec["pkl_sha256"]


def car_lap(root: Path, subject: str, name: str) -> dict:
    """The representative run: the completed run whose lap time is closest to the median
    (ties: the lowest index), with its lap-2 profile, line and speed at the stations."""
    rec = json.loads((root / "laps" / subject / f"{name}.json").read_text())
    raw = (root / "laps" / subject / f"{name}.npz").read_bytes()
    if LB.sha256_bytes(raw) != rec["npz_sha256"]:
        raise SystemExit(f"{root}/laps/{subject}/{name}.npz does not match its record")
    lt = np.asarray([LB.unclean(x) for x in rec["lap_time_s"]], float)
    done = np.nonzero(np.asarray(rec["completed"], bool))[0]
    med = LB._med(lt[done])
    r = int(done[np.argmin(np.abs(lt[done] - med))])
    with np.load(io.BytesIO(raw)) as z:
        out = {"run": r, "T_a": float(lt[r]), "median_lap": med, "stations": z["stations"],
               "tau": z["tau"][r], "n": z["path_n"][r], "npz_sha256": rec["npz_sha256"]}
    if out["tau"][-1] != out["T_a"]:
        raise SystemExit(f"{subject} {name}: the lap profile misses the lap time")
    return out


def line_on_grid(res, lap: dict) -> np.ndarray:
    return np.interp(res.s, lap["stations"], lap["n"])


def band(res, n_line: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """[lb, ub]: within eps of the line and inside the reference's corridor, never excluding
    the line itself."""
    m = res.meta["margin"]
    lo, hi = -(res.w_right - m), res.w_left - m
    return (np.maximum(n_line - eps, np.minimum(lo, n_line)),
            np.minimum(n_line + eps, np.maximum(hi, n_line)))


def grid_track(res, w_left: np.ndarray, w_right: np.ndarray, name: str):
    """A track on the reference's own grid: its curvature, the given widths. x, y and heading
    are zeros; the solver does not use them."""
    from apexzero.tracks.track import Track

    z = np.zeros(len(res.s))
    return Track(name=name, s=res.s.copy(), x=z, y=z, theta=z, kappa=res.kappa.copy(),
                 w_left=np.asarray(w_left, float), w_right=np.asarray(w_right, float))


def solve_like_lockbox(track, op: dict):
    """The lockbox's solve: the default initial guess, then one retry from the failed iterate
    at tol 1e-5."""
    from apexzero.ocp.min_time import solve_min_time
    from apexzero.vehicle.params import F1_2025

    attempts = []
    res = solve_min_time(track, F1_2025, ds=op["ds"], margin=op["margin"], tol=op["tol"],
                         max_iter=op["max_iter"])
    attempts.append({"init": "qss", "tol": op["tol"], "success": res.success,
                     "status": res.status, "iterations": res.iterations,
                     "solve_s": round(res.solve_seconds, 1)})
    if not res.success:
        res = solve_min_time(track, F1_2025, ds=op["ds"], margin=op["margin"], tol=1e-5,
                             max_iter=op["max_iter"], init=(res.z, res.u))
        attempts.append({"init": "failed iterate", "tol": 1e-5, "success": res.success,
                         "status": res.status, "iterations": res.iterations,
                         "solve_s": round(res.solve_seconds, 1)})
    return res, attempts


def shooting(res) -> dict:
    from apexzero.ocp.check import shooting_check
    from apexzero.vehicle.params import F1_2025

    try:
        chk = shooting_check(res, F1_2025)
        return {"shooting_rel_err": float(chk.rel_lap_error), "shooting_error": None}
    except Exception as exc:  # noqa: BLE001  (recorded; analyse treats it as failed)
        return {"shooting_rel_err": float("nan"), "shooting_error": repr(exc)}


def solve_key(subject: str, name: str, eps: float) -> str:
    return f"{subject}__{name}__e{eps:.2f}"


def fresh(js: Path, pk: Path | None = None) -> bool:
    """True if the task still has to run; a data file left without its record stops it."""
    if js.exists():
        return False
    if pk is not None and pk.exists():
        raise SystemExit(f"{pk} has no record: move it to the Trash and resume")
    return True


def recon_task(root: str, out: str, op: dict, circuit: str) -> dict:
    """The solver's noise floor: the reference re-solved from its own pickle's arrays."""
    t_start = time.time()
    root, out = Path(root), Path(out)
    js = out / "recon" / f"{circuit}.json"
    if not fresh(js):
        return {"task": f"recon {circuit}", "skipped": True}
    res, sha = reference(root, circuit)
    sol, attempts = solve_like_lockbox(grid_track(res, res.w_left, res.w_right, circuit), op)
    LB.write_json_new(js, {
        "circuit": circuit, "ref_pkl_sha256": sha, "ref_lap_s": res.lap_time,
        "lap_s": sol.lap_time, "rel_diff": sol.lap_time / res.lap_time - 1.0,
        "success": sol.success, "attempts": attempts,
        "grid_equal": len(sol.s) == len(res.s) and bool(np.max(np.abs(sol.s - res.s)) < 1e-9),
        "max_abs_input_diff": float(max(np.max(np.abs(sol.kappa - res.kappa)),
                                        np.max(np.abs(sol.w_left - res.w_left)),
                                        np.max(np.abs(sol.w_right - res.w_right)))),
        **shooting(sol), "provenance": provenance(t_start)})
    return {"task": f"recon {circuit}", "success": sol.success,
            "seconds": round(time.time() - t_start)}


def band_task(root: str, out: str, op: dict, subject: str, circuit: str, name: str,
              eps: float) -> dict:
    """One fixed-line solve: the lap's reference problem with n held within eps of the line."""
    t_start = time.time()
    root, out = Path(root), Path(out)
    key = solve_key(subject, name, eps)
    js, pk = out / "solves" / f"{key}.json", out / "solves" / f"{key}.pkl"
    if not fresh(js, pk):
        return {"task": key, "skipped": True}
    res, ref_sha = reference(root, circuit)
    lap = car_lap(root, subject, name)
    n_line = line_on_grid(res, lap)
    lb, ub = band(res, n_line, eps)
    m = op["margin"]
    sol, attempts, error = None, [], None
    try:      # a failing solve is recorded as failed, never fatal
        sol, attempts = solve_like_lockbox(grid_track(res, ub + m, m - lb, name), op)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    rec = {"key": key, "subject": subject, "circuit": circuit, "name": name, "eps_m": eps,
           "run": lap["run"], "T_a_s": lap["T_a"], "ref_lap_s": res.lap_time,
           "ref_pkl_sha256": ref_sha, "lap_npz_sha256": lap["npz_sha256"],
           "band_holds_line": bool(np.all((lb <= n_line) & (n_line <= ub))),
           "min_band_width_m": float(np.min(ub - lb)), "error": error, "attempts": attempts}
    if sol is not None:
        n = sol.z[:, 0]
        rec.update({
            "success": sol.success, "status": sol.status, "lap_time_s": sol.lap_time,
            "grid_equal": len(sol.s) == len(res.s) and bool(np.max(np.abs(sol.s - res.s)) < 1e-9),
            "max_band_violation_m": float(max(np.max(lb - n), np.max(n - ub), 0.0)),
            **shooting(sol),
            "pkl_sha256": LB.write_new(pk, pickle.dumps(sol))})
    else:
        rec.update({"success": False, "status": "error", "lap_time_s": float("nan"),
                    "grid_equal": False, "max_band_violation_m": float("nan"),
                    "shooting_rel_err": float("nan"), "shooting_error": "no solution",
                    "pkl_sha256": None})
    rec["provenance"] = provenance(t_start)
    LB.write_json_new(js, rec)
    return {"task": key, "success": rec["success"], "seconds": round(time.time() - t_start)}


def stage_solve(root: Path, workers: int) -> None:
    check_root(root)
    out = OUT / root.name
    op = json.loads((root / "freeze.json").read_text())["protocol"]["ocp"]
    _, rows = completed_rows(root)
    tasks = [(recon_task, (str(root), str(out), op, c))
             for c in sorted({r["circuit"] for r in rows})
             if fresh(out / "recon" / f"{c}.json")]
    tasks += [(band_task, (str(root), str(out), op, r["subject"], r["circuit"], r["name"], e))
              for r in rows for e in EPS
              if fresh(out / "solves" / f"{solve_key(r['subject'], r['name'], e)}.json",
                       out / "solves" / f"{solve_key(r['subject'], r['name'], e)}.pkl")]
    print(f"{len(tasks)} solves to run", flush=True)
    if not tasks:
        return
    ctx = multiprocessing.get_context("spawn")
    workers = max(1, min(workers, MAX_WORKERS, len(tasks)))
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=LB.lower_priority,
                             initargs=(NICE,)) as pool:
        futs = [pool.submit(fn, *a) for fn, a in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"[{i}/{len(tasks)}] {json.dumps(LB.clean(fut.result()))}", flush=True)


# ------------------------------------------------------------------------- analyse


def corner_sectors(root: Path) -> dict:
    """Per original circuit, the lockbox's sectors and their types (corners.csv), which must be
    the same for every subject."""
    from apexzero.viz.sectors import Sector

    by = {}
    with (root / "corners.csv").open() as f:
        for r in csv.DictReader(f):
            if r["variant"] == "original":
                by.setdefault(r["circuit"], {}).setdefault(r["subject"], []).append(
                    (r["corner"], float(r["s_start_m"]), float(r["s_end_m"]), r["type"]))
    out = {}
    for c, subs in by.items():
        lists = list(subs.values())
        if any(x != lists[0] for x in lists):
            raise SystemExit(f"{c}: the sectors differ between subjects")
        out[c] = ([Sector(name=n, s_start=a, s_end=b, s_apex=a, r_min=float("nan"), angle=0.0,
                          corners=[]) for n, a, b, _ in lists[0]],
                  [t for *_, t in lists[0]])
    return out


def median_subject(rows: list[dict], key, subjects: list[str]) -> dict:
    """Each subject's median over its rows, then the median over subjects (the headline's
    aggregation); NaN (missing) is dropped."""
    per = {s: LB._med([key(r) for r in rows if r["subject"] == s]) for s in subjects}
    return {"median_subject": LB._med(list(per.values())), "per_subject": per}


def eps50(R: list[float]) -> float:
    """The band at which R first reaches 0.5, linear between grid points; -inf below the grid,
    +inf above it, NaN if a value it needs is missing."""
    for i, x in enumerate(R):
        if not np.isfinite(x):
            return float("nan")
        if x >= 0.5:
            if i == 0:
                return float("-inf")
            return float(EPS[i - 1] + (0.5 - R[i - 1]) * (EPS[i] - EPS[i - 1]) / (x - R[i - 1]))
    return float("inf")


def load_solve(out: Path, key: str):
    """The solve's record and, if it passed (converged, on the reference's grid, shooting error
    below the lockbox's P2 limit), its result."""
    rec = json.loads((out / "solves" / f"{key}.json").read_text())
    ok = (rec["success"] and rec["grid_equal"] and rec["shooting_error"] is None
          and abs(LB.unclean(rec["shooting_rel_err"])) < LB.P2_SHOOTING_LIMIT)
    if not ok:
        return rec, None
    raw = (out / "solves" / f"{key}.pkl").read_bytes()
    if LB.sha256_bytes(raw) != rec["pkl_sha256"]:
        raise SystemExit(f"{key}.pkl does not match its record")
    return rec, pickle.loads(raw)


def stage_analyse(root: Path) -> None:
    from apexzero.viz.sectors import sector_deltas, sector_of

    t_start = time.time()
    label = check_root(root)
    out = OUT / root.name
    path = out / "split.json"
    if path.exists():
        raise SystemExit(f"{path} exists: write-once")
    an, rows = completed_rows(root)
    subjects = an["subjects"]
    sectors = corner_sectors(root)
    inputs = {str(root / f): LB.sha256_file(root / f)
              for f in ("analysis.json", "corners.csv", "freeze.json")}
    circuits = sorted({r["circuit"] for r in rows})
    recon = {c: json.loads((out / "recon" / f"{c}.json").read_text()) for c in circuits}
    refs = {c: reference(root, c) for c in circuits}
    for c, (_, sha) in refs.items():
        inputs[str(root / "ocp" / f"{c}__w+0.0.pkl")] = sha

    per_row, failed, viol = [], [], {"below_optimum": [], "not_monotone": [], "car_beats_line": []}
    corner = {e: {t: {"exec_s": 0.0, "line_s": 0.0, "n_sectors": 0} for t in TYPES}
              for e in CORNER_EPS}
    dev_by_type = {t: [] for t in TYPES}
    closure, gap_diff, band_viol, band_ok = 0.0, 0.0, 0.0, True
    for r in rows:
        res, _ = refs[r["circuit"]]
        T_star = res.lap_time
        if T_star != LB.unclean(r["t_ocp_s"]["+0.0"]):
            raise SystemExit(f"{r['circuit']}: the reference is not the lockbox's")
        lap = car_lap(root, r["subject"], r["name"])
        inputs[str(root / "laps" / r["subject"] / f"{r['name']}.npz")] = lap["npz_sha256"]
        T_a, lost = lap["T_a"], lap["T_a"] - T_star
        L = len(res.s) * res.meta["ds"]
        n_line = line_on_grid(res, lap)
        dev = np.abs(n_line - res.z[:, 0])
        secs, types = sectors[r["circuit"]]
        idx = sector_of(res.s, secs, L)
        for i, t in enumerate(types):
            if t in dev_by_type:
                dev_by_type[t].append(dev[idx == i])
        gap_diff = max(gap_diff, abs(T_a / T_star - 1.0 - LB.unclean(r["gap"]["+0.0"])))
        T, sols = {}, {}
        for e in EPS:
            key = solve_key(r["subject"], r["name"], e)
            rec, sol = load_solve(out, key)
            band_ok &= rec["band_holds_line"]
            if sol is None:
                failed.append({"key": key, "status": rec["status"], "error": rec["error"],
                               "shooting_rel_err": rec["shooting_rel_err"]})
                T[e] = float("nan")
                continue
            band_viol = max(band_viol, rec["max_band_violation_m"])
            T[e], sols[e] = sol.lap_time, sol
            if sol.lap_time < T_star * (1 - REL_TOL):
                viol["below_optimum"].append({"key": key, "T_eps": sol.lap_time, "T_star": T_star})
        Ts = [T[e] for e in EPS]
        for a, b, e in zip(Ts, Ts[1:], EPS[1:], strict=False):
            if np.isfinite(a) and np.isfinite(b) and b > a * (1 + REL_TOL):
                viol["not_monotone"].append({"row": f"{r['subject']} {r['circuit']}", "eps": e,
                                             "T_prev": a, "T": b})
        if np.isfinite(T[EPS[0]]) and T_a < T[EPS[0]] * (1 - REL_TOL):
            viol["car_beats_line"].append({"row": f"{r['subject']} {r['circuit']}",
                                           "T_a": T_a, "T_eps": T[EPS[0]]})
        R = [(T_a - T[e]) / lost for e in EPS]
        st = lap["stations"]
        t_star = LB.ocp_time_at(res, st)
        for e in CORNER_EPS:
            if e not in sols:
                continue
            t_e = LB.ocp_time_at(sols[e], st)
            ex = sector_deltas(st, lap["tau"] - t_e, secs, L)
            ln = sector_deltas(st, t_e - t_star, secs, L)
            closure = max(closure, abs(ex.sum() - (T_a - T[e])), abs(ln.sum() - (T[e] - T_star)))
            for i, t in enumerate(types):
                if t in corner[e]:
                    corner[e][t]["exec_s"] += float(ex[i])
                    corner[e][t]["line_s"] += float(ln[i])
                    corner[e][t]["n_sectors"] += 1
        per_row.append({
            "subject": r["subject"], "circuit": r["circuit"], "run": lap["run"],
            "T_a_s": T_a, "T_star_s": T_star, "G": T_a / T_star - 1.0,
            "T_eps_s": {f"{e}": T[e] for e in EPS},
            "R": {f"{e}": x for e, x in zip(EPS, R, strict=True)},
            "G_exec": {f"{e}": (T_a - T[e]) / T_star for e in EPS},
            "G_line": {f"{e}": (T[e] - T_star) / T_star for e in EPS},
            "eps50_m": eps50(R),
            "dev_median_m": float(np.median(dev)), "dev_p90_m": float(np.percentile(dev, 90)),
            "wrap_jump_m": float(abs(lap["n"][-1] - lap["n"][0]))})
    if closure > CLOSURE_LIMIT:
        raise SystemExit(f"the sector parts miss the lap totals by {closure:.2e} s")

    lost_all = np.array([x["T_a_s"] - x["T_star_s"] for x in per_row])
    recovery, gap_split = {}, {}
    for e in EPS:
        k = f"{e}"
        Re = np.array([x["R"][k] for x in per_row])
        rec_e = np.array([x["T_a_s"] - x["T_eps_s"][k] for x in per_row])
        ok = np.isfinite(rec_e)
        recovery[k] = {**median_subject(per_row, lambda x, k=k: x["R"][k], subjects),
                       "pooled": float(rec_e[ok].sum() / lost_all[ok].sum()),
                       "min": float(np.nanmin(Re)), "max": float(np.nanmax(Re)),
                       "n_rows": int(ok.sum())}
        gap_split[k] = {
            "G": median_subject(per_row, lambda x: x["G"], subjects)["median_subject"],
            "G_exec": median_subject(per_row, lambda x, k=k: x["G_exec"][k],
                                     subjects)["median_subject"],
            "G_line": median_subject(per_row, lambda x, k=k: x["G_line"][k],
                                     subjects)["median_subject"]}
    e50 = np.array([x["eps50_m"] for x in per_row])
    for e in CORNER_EPS:
        for t in TYPES:
            c = corner[e][t]
            c["lost_s"] = c["exec_s"] + c["line_s"]
            c["exec_share"] = c["exec_s"] / c["lost_s"] if c["lost_s"] else float("nan")
    dev_all = np.concatenate([d for t in TYPES for d in dev_by_type[t]])
    r005 = recovery[f"{EPS[0]}"]["median_subject"]
    reading = ("mostly line" if r005 < 1 / 3 else "mostly execution" if r005 > 2 / 3
               else "mixed")
    out_obj = {
        "record": "line-execution-split", "label": f"POST HOC, descriptive; {label}",
        "plan": "docs/findings/line-execution-plan.md", "root": str(root),
        "freeze_id": an["freeze_id"], "eps_m": list(EPS), "rows_n": len(per_row),
        "reading": {"rule": "R(0.05) at the median subject: < 1/3 mostly line, > 2/3 mostly "
                            "execution, otherwise mixed", "R_0.05": r005, "reading": reading},
        "recovery": recovery, "gap_split": gap_split,
        "eps50_m": {**median_subject(per_row, lambda x: x["eps50_m"], subjects),
                    "min": float(np.min(e50)), "max": float(np.max(e50))},
        "corner_types": {f"{e}": corner[e] for e in CORNER_EPS},
        "lateral_deviation_m": {
            "per_row_median": median_subject(per_row, lambda x: x["dev_median_m"], subjects),
            "per_row_p90": median_subject(per_row, lambda x: x["dev_p90_m"], subjects),
            "by_type": {t: {"median": float(np.median(np.concatenate(dev_by_type[t]))),
                            "p90": float(np.percentile(np.concatenate(dev_by_type[t]), 90)),
                            "stations": int(sum(len(d) for d in dev_by_type[t]))}
                        for t in TYPES if dev_by_type[t]},
            "all": {"median": float(np.median(dev_all)),
                    "p90": float(np.percentile(dev_all, 90))}},
        "checks": {
            "recon_rel_diff": {c: v["rel_diff"] for c, v in recon.items()},
            "recon_max_abs_rel_diff": max(abs(v["rel_diff"]) for v in recon.values()),
            "recon_all_converged": all(v["success"] for v in recon.values()),
            "recon_max_abs_input_diff": max(v["max_abs_input_diff"] for v in recon.values()),
            "failed_solves": failed, "violations": viol,
            "band_holds_line": band_ok, "max_band_violation_m": band_viol,
            "max_wrap_jump_m": max(x["wrap_jump_m"] for x in per_row),
            "max_gap_diff_vs_lockbox": gap_diff, "sector_closure_s": closure},
        "rows": per_row, "inputs_sha256": dict(sorted(inputs.items())),
        "provenance": provenance(t_start)}
    LB.write_json_new(path, out_obj)
    print(json.dumps(LB.clean({k: out_obj[k] for k in ("reading", "recovery", "eps50_m")}),
                     indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("solve", "analyse"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    a = ap.parse_args()
    root = Path(a.root)
    if a.stage == "solve":
        stage_solve(root, a.workers)
    else:
        stage_analyse(root)


if __name__ == "__main__":
    main()
