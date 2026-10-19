"""Snapping audit: does the public F1 position feed carry each driver's racing line?

For one session, every flying lap's location samples are compared with a reference curve
through the pole lap (a cubic spline in time through its samples, so corner chords do
not masquerade as lateral offsets). If positions were genuinely measured, different
drivers' lines through a corner would differ by of order a metre. If they are projected
onto a fixed track model, the distances collapse to centimetres.

Positive controls make a null result meaningful:
- pit-lane samples (physically 10-20 m from the racing line) must show large distances;
- a flying lap shifted laterally by a known 1-3 m offset must be recovered.

Source: OpenF1 (CC BY-NC-SA 4.0). Positions are in decimetres in the feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from apexzero.data.openf1 import read_json_gz

DM_TO_M = 0.1


def _t(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


@dataclass
class Lap:
    driver: int
    lap_number: int
    t0: float
    duration: float
    pit_out: bool


@dataclass
class SessionPositions:
    key: int
    year: int
    circuit: str
    laps: list[Lap]
    loc: dict[int, tuple[np.ndarray, np.ndarray]]   # driver -> (t [N], xy [N, 2] metres)


def load_session(directory: str | Path) -> SessionPositions:
    import json

    d = Path(directory)
    meta = json.loads((d / "session.json").read_text())
    laps = []
    for row in read_json_gz(d / "laps.json.gz"):
        if row.get("lap_duration") is None or row.get("date_start") is None:
            continue
        laps.append(Lap(int(row["driver_number"]), int(row["lap_number"]), _t(row["date_start"]),
                        float(row["lap_duration"]), bool(row.get("is_pit_out_lap"))))
    loc = {}
    for path in sorted((d / "location").glob("*.json.gz")):
        rows = [r for r in read_json_gz(path) if not (r["x"] == 0 and r["y"] == 0)]
        if not rows:
            continue
        t = np.array([_t(r["date"]) for r in rows])
        xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float) * DM_TO_M
        order = np.argsort(t)
        loc[int(path.name.split(".")[0])] = (t[order], xy[order])
    return SessionPositions(meta["key"], meta["year"], meta["circuit"], laps, loc)


def flying_laps(sess: SessionPositions, within: float = 1.07) -> list[Lap]:
    best = min(lap.duration for lap in sess.laps if not lap.pit_out)
    return [lap for lap in sess.laps if not lap.pit_out and lap.duration <= within * best
            and lap.driver in sess.loc]


def lap_samples(sess: SessionPositions, lap: Lap) -> tuple[np.ndarray, np.ndarray]:
    t, xy = sess.loc[lap.driver]
    m = (t >= lap.t0) & (t <= lap.t0 + lap.duration)
    return t[m], xy[m]


def flag_broken_streams(sess: SessionPositions, laps: list[Lap], tree, factor: float = 3.0,
                        floor_m: float = 0.5) -> set[int]:
    """Drivers whose positions sit far from everyone else's (a broken coordinate stream,
    e.g. 2023 Jeddah car 21 at ~3 m everywhere, straights included)."""
    med = {}
    for lap in laps:
        _, xy = lap_samples(sess, lap)
        if len(xy):
            med.setdefault(lap.driver, []).append(tree.query(xy)[0])
    per = {d: float(np.median(np.concatenate(v))) for d, v in med.items()}
    session = float(np.median(list(per.values())))
    return {d for d, m in per.items() if m > max(floor_m, factor * session)}


def reference_curve(t: np.ndarray, xy: np.ndarray, dt: float = 0.005) -> np.ndarray:
    """Dense curve through a lap's samples: cubic spline in time, evaluated every dt."""
    keep = np.r_[True, np.diff(t) > 1e-3]
    t, xy = t[keep], xy[keep]
    sx, sy = CubicSpline(t, xy[:, 0]), CubicSpline(t, xy[:, 1])
    tt = np.arange(t[0], t[-1], dt)
    return np.stack([sx(tt), sy(tt)], axis=1)


def lateral_shift(t: np.ndarray, xy: np.ndarray, offset: float) -> np.ndarray:
    """Shift a lap's samples by ``offset`` metres along the left normal of its path."""
    sx, sy = CubicSpline(t, xy[:, 0]), CubicSpline(t, xy[:, 1])
    vx, vy = sx(t, 1), sy(t, 1)
    nrm = np.stack([-vy, vx], axis=1) / np.hypot(vx, vy)[:, None]
    return xy + offset * nrm


@dataclass
class AuditResult:
    session: str
    n_laps: int
    n_drivers: int
    n_samples: int
    median_m: float
    p95_m: float
    p99_m: float
    per_driver_median_m: dict[int, float]
    pit_lane_p95_m: float
    pit_lane_max_m: float
    injected: dict[float, float]          # offset -> median recovered distance
    pole: tuple[int, int, float]          # driver, lap, time
    flagged: tuple[int, ...] = ()         # drivers excluded for a broken coordinate stream


def audit(sess: SessionPositions) -> AuditResult:
    laps = flying_laps(sess)
    pole = min(laps, key=lambda lap: lap.duration)
    tp, xyp = lap_samples(sess, pole)
    tree = cKDTree(reference_curve(tp, xyp))
    flagged = flag_broken_streams(sess, laps, tree)
    laps = [lap for lap in laps if lap.driver not in flagged]

    dists, per_driver = [], {}
    for lap in laps:
        if lap is pole:
            continue
        _, xy = lap_samples(sess, lap)
        if len(xy) < 50:
            continue
        d = tree.query(xy)[0]
        dists.append(d)
        per_driver.setdefault(lap.driver, []).append(d)
    alld = np.concatenate(dists)

    # Positive control 1: samples well inside pit in/out laps, near the pit lane.
    pit = []
    for lap in sess.laps:
        if lap.pit_out and lap.driver in sess.loc:
            _, xy = lap_samples(sess, lap)
            if len(xy) > 20:
                pit.append(tree.query(xy[:15])[0])  # first seconds of an out-lap
    pit = np.concatenate(pit) if pit else np.array([np.nan])

    # Positive control 2: inject a known lateral offset into another driver's best lap.
    other = min((lap for lap in laps if lap.driver != pole.driver), key=lambda lap: lap.duration)
    to, xyo = lap_samples(sess, other)
    injected = {}
    for off in (1.0, 2.0, 3.0):
        injected[off] = float(np.median(tree.query(lateral_shift(to, xyo, off))[0]))

    return AuditResult(
        session=f"{sess.year} {sess.circuit} (key {sess.key})", n_laps=len(dists),
        n_drivers=len(per_driver), n_samples=len(alld), median_m=float(np.median(alld)),
        p95_m=float(np.percentile(alld, 95)), p99_m=float(np.percentile(alld, 99)),
        per_driver_median_m={k: float(np.median(np.concatenate(v))) for k, v in per_driver.items()},
        pit_lane_p95_m=float(np.nanpercentile(pit, 95)), pit_lane_max_m=float(np.nanmax(pit)),
        injected=injected, pole=(pole.driver, pole.lap_number, pole.duration),
        flagged=tuple(sorted(flagged)),
    )


@dataclass
class DriverSignal:
    median_icc: float              # intraclass correlation of signed offsets, by station
    frac_bins_significant: float   # share of 50 m stations with a driver effect at p < 0.01
    p95_driver_offset_m: float     # 95th pct over (station, driver) of |driver mean - station mean|
    max_driver_offset_m: float
    planted_median_icc: float      # same statistics after planting a 0.5 m line for one driver
    planted_frac_significant: float
    planted_p95_driver_offset_m: float
    n_bins: int


def _signed_offsets(tree, curve, s_curve, xy):
    d, idx = tree.query(xy)
    idx = np.clip(idx, 1, len(curve) - 2)
    tan = curve[idx + 1] - curve[idx - 1]
    rel = xy - curve[idx]
    sign = np.sign(tan[:, 0] * rel[:, 1] - tan[:, 1] * rel[:, 0])
    return s_curve[idx], sign * d


def driver_signal(sess: SessionPositions, bin_m: float = 50.0, plant_m: float = 0.5,
                  min_laps: int = 3) -> DriverSignal:
    """Is a driver's lateral offset from the reference consistent across their own laps?

    For every 50 m station, compares the variance of drivers' mean signed offsets with the
    lap-to-lap variance within drivers (one-way ANOVA). Real racing lines would show
    consistent per-driver offsets (ICC well above 0). A planted per-driver line of
    ``plant_m`` metres for one driver is the positive control.
    """
    from scipy.stats import f as f_dist

    laps = flying_laps(sess)
    pole = min(laps, key=lambda lap: lap.duration)
    tp, xyp = lap_samples(sess, pole)
    curve = reference_curve(tp, xyp)
    s_curve = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    tree = cKDTree(curve)
    flagged = flag_broken_streams(sess, laps, tree)
    laps = [lap for lap in laps if lap.driver not in flagged]
    counts = {}
    for lap in laps:
        counts[lap.driver] = counts.get(lap.driver, 0) + 1
    drivers = [d for d, c in counts.items() if c >= min_laps and d != pole.driver]
    if len(drivers) < 5:
        raise ValueError(f"only {len(drivers)} drivers with >= {min_laps} flying laps")
    planted = drivers[0]
    nb = int(s_curve[-1] // bin_m)

    def stats(plant: float):
        table = {}  # (bin, driver) -> list of per-lap mean offsets
        for lap in laps:
            if lap.driver not in drivers:
                continue
            t, xy = lap_samples(sess, lap)
            if len(xy) < 50:
                continue
            if lap.driver == planted and plant:
                xy = lateral_shift(t, xy, plant)
            s, off = _signed_offsets(tree, curve, s_curve, xy)
            b = (s // bin_m).astype(int)
            for k in np.unique(b):
                if 0 <= k < nb:
                    table.setdefault((k, lap.driver), []).append(off[b == k].mean())
        iccs, sig, effects = [], [], []
        for k in range(nb):
            groups = [np.array(table[(k, d)]) for d in drivers if len(table.get((k, d), [])) >= 2]
            if len(groups) < 5:
                continue
            means = np.array([g.mean() for g in groups])
            effects.extend(np.abs(means - means.mean()))
            n_i = np.array([len(g) for g in groups])
            grand = np.concatenate(groups).mean()
            ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
            ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
            dfb, dfw = len(groups) - 1, n_i.sum() - len(groups)
            msb, msw = ssb / dfb, max(ssw / dfw, 1e-12)
            n0 = (n_i.sum() - (n_i**2).sum() / n_i.sum()) / dfb
            iccs.append((msb - msw) / (msb + (n0 - 1) * msw))
            sig.append(f_dist.sf(msb / msw, dfb, dfw) < 0.01)
        eff = np.array(effects)
        return (float(np.median(iccs)), float(np.mean(sig)), float(np.percentile(eff, 95)),
                float(eff.max()), len(iccs))

    icc, frac, p95, mx, n = stats(0.0)
    icc_p, frac_p, p95_p, _, _ = stats(plant_m)
    return DriverSignal(icc, frac, p95, mx, icc_p, frac_p, p95_p, n)
