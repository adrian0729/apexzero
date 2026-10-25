"""Solve the minimum-lap-time OCP on the R-dev circuits and check it against the simulator.

    uv run python scripts/solve_all_ocp.py --workers 4

Writes outputs/ocp/<track>.pkl and outputs/ocp/summary.csv. With no --tracks it solves R_DEV
(``apexzero.tracks.corners``). Every requested circuit goes through ``viz.export.lockbox_guard``,
so no optimum is solved on a lockbox (R_TEST) circuit until protocol-v1 unseals it (STATUS,
'Protocol deviation'; the first all-circuit batch is sealed). The original P2 gate was
convergence on at least 23 of 25 circuits with shooting consistency under 0.3%.
"""

import argparse
import csv
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)


def solve_one(name: str, ds: float, out: str) -> dict:
    import numpy as np

    from apexzero.ocp.check import shooting_check
    from apexzero.ocp.min_time import solve_min_time
    from apexzero.oracle.qss import GG, lap_time, speed_profile
    from apexzero.tracks.reference import drivable_margin, refine_reference
    from apexzero.tracks.track import load_tumftm
    from apexzero.vehicle.params import F1_2025

    t0 = time.perf_counter()
    tr = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
    res = solve_min_time(tr, F1_2025, ds=ds)
    if not res.success:
        # Retry once from the failed iterate with a looser tolerance.
        res = solve_min_time(tr, F1_2025, ds=ds, init=(res.z, res.u), tol=1e-5)
    chk = shooting_check(res, F1_2025)
    qss = lap_time(speed_profile(tr.kappa, tr.ds, GG.from_params(F1_2025)), tr.ds)
    with open(Path(out) / f"{name}.pkl", "wb") as fh:
        pickle.dump(res, fh)
    return {
        "track": name, "length_m": round(tr.length, 1), "success": res.success,
        "status": res.status, "iterations": res.iterations,
        "solve_s": round(res.solve_seconds, 1), "lap_time_s": round(res.lap_time, 3),
        "centreline_qss_s": round(qss, 3), "shooting_rel_err": f"{chk.rel_lap_error:+.2e}",
        "max_dv_mps": round(chk.max_speed_error, 3),
        "v_min_kph": round(float(np.min(res.z[:, 2])) * 3.6, 1),
        "v_max_kph": round(float(np.max(res.z[:, 2])) * 3.6, 1),
        "drivable_margin": round(drivable_margin(tr), 3),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ds", type=float, default=3.0)
    ap.add_argument("--out", default="outputs/ocp")
    ap.add_argument("--tracks", nargs="*")
    args = ap.parse_args()
    from apexzero.tracks.corners import R_DEV
    from apexzero.viz.export import lockbox_guard

    names = args.tracks or list(R_DEV)
    for n in names:
        lockbox_guard(n)   # refuses R_TEST circuits (and their variants) before any solve
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(solve_one, n, args.ds, args.out): n for n in names}
        for fut in as_completed(futs):
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001  (report and keep going)
                row = {"track": futs[fut], "success": False, "status": f"error: {exc!r}"}
            rows.append(row)
            print({k: row.get(k) for k in ("track", "success", "lap_time_s", "shooting_rel_err", "solve_s")}, flush=True)
    rows.sort(key=lambda r: r["track"])
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(Path(args.out) / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    ok = sum(bool(r.get("success")) for r in rows)
    print(f"converged {ok}/{len(rows)}")
