"""The lockbox's provenance comparison (docs/prereg/protocol-v1.md Section 14): DESCRIPTIVE.

    uv run python scripts/lockbox_provenance.py --out outputs/lockbox-provenance/provenance.json

Run only after docs/findings/lockbox.md is committed. It opens ``outputs/sealed/`` for the first
time and compares the old optima there (the first all-circuit s-OCP batch of 18 Oct 2026 and the
stray Monza solves) with the lockbox's re-solved nominal references of the 12 originals
(``outputs/lockbox/ocp/<circuit>__w+0.0.pkl``, each checked against its record's sha256). It
decides nothing and changes no number in the lockbox findings. The TO-OCP speed benchmarks in
``outputs/sealed/to_ocp/`` are 60 s windows, not laps; they are listed with their sha256s and
compared with nothing.
Run it from the repository root (its paths are relative).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from apexzero.ocp.min_time import Z_NAMES
from apexzero.tracks.corners import R_TEST

SEALED = Path("outputs/sealed")
LOCKBOX = Path("outputs/lockbox")
VX = Z_NAMES.index("vx")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def new_reference(circuit: str):
    key = f"{circuit}__w+0.0"
    rec = json.loads((LOCKBOX / "ocp" / f"{key}.json").read_text())
    pk = LOCKBOX / "ocp" / f"{key}.pkl"
    if sha(pk) != rec["pkl_sha256"]:
        raise SystemExit(f"{pk}: sha256 differs from its record")
    return load(pk), rec


def compare(old, new) -> dict:
    """The old optimum against the new: lap time, settings, grid, geometry and profiles."""
    out = {"old_lap_s": old.lap_time, "new_lap_s": new.lap_time,
           "lap_rel_diff": new.lap_time / old.lap_time - 1.0,
           "old_success": bool(old.success), "old_status": old.status,
           "old_iterations": old.iterations, "new_iterations": new.iterations,
           "old_meta": old.meta, "meta_equal": old.meta == new.meta,
           "n_old": len(old.s), "n_new": len(new.s)}
    if len(old.s) == len(new.s):
        out.update({
            "max_abs_ds_m": float(np.max(np.abs(old.s - new.s))),
            "max_abs_dkappa": float(np.max(np.abs(old.kappa - new.kappa))),
            "max_abs_dw_left_m": float(np.max(np.abs(old.w_left - new.w_left))),
            "max_abs_dw_right_m": float(np.max(np.abs(old.w_right - new.w_right))),
            "max_abs_dvx_mps": float(np.max(np.abs(old.z[:, VX] - new.z[:, VX]))),
            "max_abs_dn_m": float(np.max(np.abs(old.z[:, 0] - new.z[:, 0]))),
            "max_abs_dt_s": float(np.max(np.abs(old.t - new.t)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = Path(a.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: write-once")
    if not (Path("docs/findings/lockbox.md").exists() and (LOCKBOX / "analysis.lock").exists()):
        raise SystemExit("the lockbox findings come first (protocol-v1 Section 14)")

    files = sorted(p for p in SEALED.rglob("*") if p.is_file())
    out = {"record": "lockbox-provenance", "label": "DESCRIPTIVE, after the lockbox findings",
           "sealed_files_sha256": {str(p): sha(p) for p in files}, "batch": {}, "stray": {}}

    for c in R_TEST:
        new, rec = new_reference(c)
        out["batch"][c] = {**compare(load(SEALED / "ocp" / f"{c}.pkl"), new),
                           "new_pkl_sha256": rec["pkl_sha256"]}

    stray = SEALED / "stray-2026-10-25"
    monza_new, _ = new_reference("Monza")
    for name in ("monza_ocp.pkl", "monza_ocp_ds2.pkl"):
        out["stray"][name] = {**compare(load(stray / name), monza_new),
                              "bytes_equal_batch_monza": (stray / name).read_bytes()
                              == (SEALED / "ocp" / "Monza.pkl").read_bytes()}
    with (stray / "ocp_summary_all_circuits.csv").open() as f:
        rows = {r["track"]: r for r in csv.DictReader(f)}
    out["stray"]["ocp_summary_all_circuits.csv"] = {
        "tracks": len(rows), "r_test_tracks": sorted(c for c in rows if c in R_TEST),
        "lap_s_vs_batch_pkl": {c: float(rows[c]["lap_time_s"]) - out["batch"][c]["old_lap_s"]
                               for c in R_TEST if c in rows}}

    rel = [v["lap_rel_diff"] for v in out["batch"].values()]
    out["summary"] = {"circuits": len(rel), "median_lap_rel_diff": float(np.median(rel)),
                      "min_lap_rel_diff": float(min(rel)), "max_lap_rel_diff": float(max(rel)),
                      "all_meta_equal": all(v["meta_equal"] for v in out["batch"].values()),
                      "all_old_succeeded": all(v["old_success"] for v in out["batch"].values())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
