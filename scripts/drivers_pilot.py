"""Drivers-lite pilot on 2025 qualifying -> outputs/drivers-lite/, docs/findings/drivers-pilot.md

    nice -n 15 uv run python scripts/drivers_pilot.py registration   # every mapped 2025 session
    nice -n 15 uv run python scripts/drivers_pilot.py registration --year 2024   # gate check
    nice -n 15 uv run python scripts/drivers_pilot.py registration --year 2023
    nice -n 15 uv run python scripts/drivers_pilot.py pilot          # pipeline on pilot circuits
    nice -n 15 uv run python scripts/drivers_pilot.py tune           # smoother, R-dev synthetics
    nice -n 15 uv run python scripts/drivers_pilot.py pilot          # again, tuned smoother
    nice -n 15 uv run python scripts/drivers_pilot.py injection      # calibrate, inject, recover
    nice -n 15 uv run python scripts/drivers_pilot.py persistent     # persistent per-car errors
    nice -n 15 uv run python scripts/drivers_pilot.py report         # writes the findings page

The first ``pilot`` pass (default smoother) only supplies the feed's sampling intervals
and position noise to ``tune``. Stages cache their results under outputs/drivers-lite/,
so the report can be rebuilt without recomputing. Single process; run it niced. Pilot
circuits are the 2025 sessions that pass the registration gate (at most four). The 2023
and 2024 registrations are only used to check the proposed corridor-excursion gate on
seasons it was not chosen on.
Measured on the busy M3 Pro: registration ~2 s CPU per session, pilot ~80 s CPU for four
sessions, tune ~2 min, injection ~3 min, persistent ~3 min, report ~10 s.

Lockbox: the pilot circuits are real F1 circuits, several of them R-test. Only telemetry
is registered and timed there. No agent, OCP or QSS lap is computed on them: the
injection-recovery laps on pilot circuits use the session's own pooled speed profile,
and the QSS-profile check and the smoother tuning run on R-dev geometry only.

Data: OpenF1 (CC BY-NC-SA 4.0). Only aggregates and per-segment deltas are written.
"""

import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

from apexzero.drivers import recovery
from apexzero.drivers.circuits import CIRCUITS
from apexzero.drivers.geometry import Centreline
from apexzero.drivers.pipeline import register_session, registration_points, run_session
from apexzero.drivers.registration import corridor_excursion, inside_offset
from apexzero.drivers.session import load
from apexzero.drivers.smoother import DEFAULT_CONFIG
from apexzero.drivers.synth import SynthConfig, qss_model
from apexzero.tracks.track import load_tumftm

ROOT = Path("data/raw/openf1")
TRACKS = Path("data/tracks/tumftm")
OUT = Path("outputs/drivers-lite")
YEAR = 2025
MAX_PILOT = 4
SYS_OFFSET_M = 1.0      # per-car along-track position offset in the persistent-error scenario
TUNE_CIRCUITS = ["Catalunya", "Hockenheim"]      # R-dev geometry only
QSS_CHECK_CIRCUITS = ["Sepang", "Nuerburgring"]  # R-dev, not used for tuning


def sessions(year=YEAR, name="Qualifying"):
    out = []
    for meta_path in sorted((ROOT / str(year)).glob("*/session.json")):
        meta = json.loads(meta_path.read_text())
        if meta["name"] == name:
            out.append((meta, meta_path.parent))
    return sorted(out, key=lambda m: m[0]["date_start"])


def save(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(name):
    with open(OUT / name, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- stages


def stage_registration(year=YEAR):
    rows = []
    for meta, d in sessions(year):
        c = CIRCUITS.get(meta["circuit"])
        if c is None or c.tumftm is None:
            rows.append({"circuit": meta["circuit"], "key": meta["key"], "tumftm": None,
                         "note": c.note if c else "unknown"})
            continue
        t0 = time.process_time()
        sess = load(d)
        line = Centreline(load_tumftm(TRACKS / f"{c.tumftm}.csv"))
        reg = register_session(sess, line)
        pts = registration_points(sess)[0]
        out_frac, exc99 = corridor_excursion(pts, line, reg.transform)
        inside = inside_offset(pts, line, reg.transform)
        rows.append({
            "circuit": meta["circuit"], "key": meta["key"], "dir": str(d), "tumftm": c.tumftm,
            "layout_matches": c.layout_matches, "lockbox": c.lockbox, "note": c.note,
            "rms_m": reg.rms_m, "p95_m": reg.p95_m, "max_m": reg.max_m,
            "scale": reg.transform.scale, "angle_deg": reg.transform.angle_deg,
            "reflected": reg.transform.reflected, "length_openf1_m": reg.length_openf1_m,
            "length_tumftm_m": reg.length_tumftm_m, "length_mismatch": reg.length_mismatch,
            "outside_corridor": out_frac, "excursion_p99_m": exc99,
            "inside_offset_m": inside,
            "accepted": reg.accepted, "reason": reg.reason, "n_points": reg.n_points,
            "cpu_s": time.process_time() - t0,
        })
        r = rows[-1]
        print(f"{meta['circuit']:>20} -> {c.tumftm:<12} RMS {r['rms_m']:.2f} m  "
              f"length {100 * r['length_mismatch']:+.2f}%  scale {r['scale']:.4f}  "
              f"outside {100 * out_frac:.1f}%  p99 exc {exc99:+.1f} m  inside {inside:+.2f} m  "
              f"{r['reason']}  "
              f"[{r['cpu_s']:.0f} s cpu]", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "registration.json" if year == YEAR else f"registration_{year}.json"
    (OUT / name).write_text(json.dumps(rows, indent=1))


def pilot_sessions():
    rows = json.loads((OUT / "registration.json").read_text())
    ok = [r for r in rows if r.get("accepted") and r.get("layout_matches")]
    return ok[:MAX_PILOT]


def stage_pilot():
    """Run with the tuned smoother if ``tune`` has run, else the defaults (the first pass
    exists only to measure the feed's noise for the tuning synthetics)."""
    cfg = tuned_config() if (OUT / "tune.json").exists() else DEFAULT_CONFIG
    print("smoother:", cfg, flush=True)
    for r in pilot_sessions():
        t0, c0 = time.time(), time.process_time()
        res = run_session(r["dir"], TRACKS, cfg)
        res.config = cfg
        res.cpu_s = time.process_time() - c0
        res.wall_s = time.time() - t0
        save(res, f"pilot_{r['tumftm']}.pkl")
        print(f"  [{res.cpu_s:.0f} s cpu, {res.wall_s:.0f} s wall]", flush=True)


def _feed_properties():
    """Sampling intervals and position noise pooled over the pilot sessions (feed
    properties, not circuit properties)."""
    res = [load_pickle(f"pilot_{r['tumftm']}.pkl") for r in pilot_sessions()]
    dl = np.concatenate([x.dt_location for x in res])
    dc = np.concatenate([x.dt_car for x in res])
    sigma = float(np.median([x.pos_sigma for x in res]))
    return dl, dc, sigma


# The first grid (jerk 100-2e4, speed sigma 0.35-1.2) put the optimum at its corner on
# both axes, so it now extends below and above it.
TUNE_GRID = [dict(jerk_psd=q, sigma_speed=sv) for q in (25.0, 50.0, 100.0, 400.0, 3e3, 2e4)
             for sv in (0.35, 0.7, 1.2, 2.0, 3.5, 6.0)]


def stage_tune():
    dl, dc, sigma = _feed_properties()
    scfg = SynthConfig(pos_sigma=sigma)
    models = [qss_model(name, TRACKS, dl, dc) for name in TUNE_CIRCUITS]
    rows = []
    for g in TUNE_GRID:
        cfg = replace(DEFAULT_CONFIG, **g)
        c0 = time.process_time()
        recs = [recovery.run(m, effects={}, pairs=4, laps_per_driver=3, reps=1, cfg=cfg,
                             synth_cfg=scfg, seed=11) for m in models]
        lap = np.concatenate([r.lap_err for r in recs], axis=1)
        pair = np.concatenate([r.pair_err for r in recs], axis=1)
        row = dict(g, lap_rms_ms=1e3 * float(np.sqrt(np.mean(lap**2))),
                   lap_bias_rms_ms=1e3 * float(np.sqrt(np.mean(lap.mean(axis=0) ** 2))),
                   lap_sd_ms=1e3 * float(np.sqrt(np.mean(lap.var(axis=0)))),
                   pair_rms_ms=1e3 * float(np.sqrt(np.mean(pair**2))),
                   cpu_s=time.process_time() - c0)
        rows.append(row)
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    best = min(rows, key=lambda r: r["pair_rms_ms"])
    print("best:", best)
    (OUT / "tune.json").write_text(json.dumps({"rows": rows, "best": best,
                                               "circuits": TUNE_CIRCUITS,
                                               "pos_sigma": sigma}, indent=1))


def tuned_config():
    best = json.loads((OUT / "tune.json").read_text())["best"]
    return replace(DEFAULT_CONFIG, jerk_psd=best["jerk_psd"], sigma_speed=best["sigma_speed"])


def stage_injection(pairs=10, reps=2, laps_per_driver=3):
    """Calibrate the synthetic feed to each pilot session, then inject and recover.

    Calibration matches the real held-out S2 error and within-driver phase spread (see
    ``recovery.calibrate``); it sets noise levels only and never touches the pipeline.
    """
    cfg = tuned_config()
    dl, dc, sigma = _feed_properties()
    out = {"config": asdict(cfg), "pilot": {}, "qss": {}, "calibration": {}}
    for r in pilot_sessions():
        res = load_pickle(f"pilot_{r['tumftm']}.pkl")
        c0 = time.process_time()
        model = res.model()
        cal = recovery.calibrate(model, res.held_out_s2, res.table, cfg,
                                 SynthConfig(pos_sigma=res.pos_sigma))
        out["calibration"][r["tumftm"]] = cal
        print(f"{r['tumftm']}: wobble {cal.wobble_sd} (held-out real {1e3 * cal.real_held_sd:.1f}"
              f" / synthetic {1e3 * cal.synth_held_sd:.1f} ms; tried "
              f"{ {k: round(1e3 * v, 1) for k, v in cal.tried.items()} })", flush=True)
        for cell, (real, meas) in cal.by_cell.items():
            print(f"    {cell}: real spread {1e3 * real:.0f} ms, synthetic measurement "
                  f"{1e3 * meas:.0f} ms", flush=True)
        rec = recovery.run(model, recovery.DEFAULT_EFFECTS, pairs=pairs,
                           laps_per_driver=laps_per_driver, reps=reps, cfg=cfg,
                           synth_cfg=cal.config(SynthConfig(pos_sigma=res.pos_sigma)), seed=100,
                           held_out=True, natural=cal.natural)
        out["pilot"][r["tumftm"]] = rec
        print(f"  {rec.pair_est.shape[0]} pairs [{time.process_time() - c0:.0f} s cpu]",
              flush=True)
    cals = list(out["calibration"].values())
    qss_cfg = SynthConfig(pos_sigma=sigma,
                          wobble_sd=float(np.median([c.wobble_sd for c in cals])))
    # Natural variation for the QSS check: median over pilot circuits per type x phase.
    nat_cell = {}
    for name, c in out["calibration"].items():
        labels = out["pilot"][name].labels
        for i, (_, ph, kind) in enumerate(labels):
            nat_cell.setdefault((kind, ph), []).append(c.natural[i])
    nat_cell = {k: float(np.median(v)) for k, v in nat_cell.items()}
    out["qss_natural"] = nat_cell
    for name in QSS_CHECK_CIRCUITS:
        c0 = time.process_time()
        model = qss_model(name, TRACKS, dl, dc)
        nat = np.array([nat_cell.get((kind, ph), 0.0) for _, ph, kind in model.seg.labels()])
        rec = recovery.run(model, recovery.DEFAULT_EFFECTS,
                           pairs=pairs, laps_per_driver=laps_per_driver, reps=reps, cfg=cfg,
                           synth_cfg=qss_cfg, seed=200, natural=nat)
        out["qss"][name] = rec
        print(f"{name} (QSS profile): [{time.process_time() - c0:.0f} s cpu]", flush=True)
    out["qss_config"] = qss_cfg
    save(out, "injection.pkl")


def stage_persistent(pairs=10, reps=2, laps_per_driver=3):
    """Persistent per-car errors (a scenario, not a measurement).

    The injection runs again with, in turn, a line pattern shared by all of a driver's
    laps (as large as one lap's calibrated wobble), a per-car along-track position
    offset of sd SYS_OFFSET_M, and both. Seeds, effects and natural variation are the
    injection stage's and the persistent terms come from a separate random stream, so
    the difference from the injection run is what they add.
    """
    cfg = tuned_config()
    inj = load_pickle("injection.pkl")
    out = {"config": {}, "runs": {}}
    for r in pilot_sessions():
        res = load_pickle(f"pilot_{r['tumftm']}.pkl")
        cal = inj["calibration"][r["tumftm"]]
        base = cal.config(SynthConfig(pos_sigma=res.pos_sigma))
        for variant, (wob, off) in {"line": (cal.wobble_sd, 0.0), "offset": (0.0, SYS_OFFSET_M),
                                    "both": (cal.wobble_sd, SYS_OFFSET_M)}.items():
            c0 = time.process_time()
            scfg = replace(base, driver_wobble_sd=wob, car_offset_sd=off)
            out["runs"].setdefault(variant, {})[r["tumftm"]] = recovery.run(
                res.model(), recovery.DEFAULT_EFFECTS, pairs=pairs,
                laps_per_driver=laps_per_driver, reps=reps, cfg=cfg, synth_cfg=scfg, seed=100,
                held_out=True, natural=cal.natural)
            out["config"].setdefault(variant, {})[r["tumftm"]] = scfg
            print(f"{r['tumftm']} ({variant}): [{time.process_time() - c0:.0f} s cpu]",
                  flush=True)
    save(out, "persistent.pkl")


def stage_report():
    from drivers_report import write_report

    write_report(OUT, pilot_sessions())


STAGES = {"registration": stage_registration, "pilot": stage_pilot, "tune": stage_tune,
          "injection": stage_injection, "persistent": stage_persistent, "report": stage_report}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--year", type=int, default=YEAR, help="registration stage only")
    args = ap.parse_args()
    if args.stage == "registration":
        stage_registration(args.year)
    else:
        STAGES[args.stage]()
