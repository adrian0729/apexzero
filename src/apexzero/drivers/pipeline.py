"""The drivers-lite pipeline end to end, for a real session or a synthetic one.

    registration -> lap checks 1-7 -> s(t) per lap -> segmentation -> traffic (check 8)
    -> per-phase lap table -> teammate deltas

Real and synthetic sessions go through the same functions (``fit_laps``), which is the
point of the injection-recovery gate: the error it measures is the error of this code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from apexzero.drivers import laps as lapf
from apexzero.drivers import segments
from apexzero.drivers.circuits import Circuit, circuit_for
from apexzero.drivers.deltas import LapTable
from apexzero.drivers.geometry import Centreline
from apexzero.drivers.progress import (
    LapInput,
    SessionTiming,
    fit_session,
    held_out_sector,
    lap_inputs,
)
from apexzero.drivers.registration import Registration, lap_path_length, register
from apexzero.drivers.session import SessionData, load
from apexzero.drivers.smoother import DEFAULT_CONFIG, POS, SmootherConfig
from apexzero.drivers.synth import CircuitModel, SynthSession, empirical_model
from apexzero.tracks.track import load_tumftm

TRACKS = Path("data/tracks/tumftm")


def registration_points(sess: SessionData, within: float = 1.07) -> tuple[np.ndarray, list]:
    """On-track samples of the session's flying laps (no out-laps, within 107%)."""
    timed = [lap for lap in sess.laps if lap.timed and not lap.pit_out
             and lap.driver in sess.location]
    if not timed:
        raise ValueError("no timed laps")
    best = min(lap.duration for lap in timed)
    fly = sorted([lap for lap in timed if lap.duration <= within * best],
                 key=lambda lap: lap.duration)
    pts = []
    for lap in fly:
        t, xy = sess.location[lap.driver]
        m = (t >= lap.date_start) & (t <= lap.date_start + lap.duration)
        pts.append(xy[m])
    return np.concatenate(pts), fly


def register_session(sess: SessionData, line: Centreline, n_length_laps: int = 8) -> Registration:
    pts, fly = registration_points(sess)
    lengths = []
    for lap in fly[:n_length_laps]:
        t, xy = sess.location[lap.driver]
        m = (t > lap.date_start - 3.0) & (t < lap.date_start + lap.duration + 3.0)
        try:
            lengths.append(lap_path_length(t[m], xy[m], lap.date_start, lap.duration))
        except ValueError:
            continue
    return register(pts, line, float(np.median(lengths)))


def reference_speed(timing: SessionTiming, keys=None, step: float = 5.0):
    """Pooled (median) centreline rate on a lap-distance grid: the reference profile."""
    x = np.arange(0.0, timing.length, step)
    use = keys if keys is not None else list(timing.laps)
    v = np.median(np.stack([timing.laps[k].speed(x) for k in use]), axis=0)
    return x, v


def car_speed(timing: SessionTiming, x: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """Centreline rate -> the car's own speed (what corner types are defined on):
    rate x path ratio x the median per-lap speed scale."""
    k = float(np.median([lf.k_speed for lf in timing.laps.values()]))
    r = np.interp(x % timing.length, np.r_[timing.ratio_x, timing.length],
                  np.r_[timing.ratio, timing.ratio[0]])
    return rate * r * k


def phase_times(timing: SessionTiming, seg: segments.Segmentation, keys) -> np.ndarray:
    """[laps, phases] times. The ends are pinned to 0 and the official lap time (the
    anchored fit already passes within microseconds of them), so phases add up to the
    official lap time exactly."""
    cuts = seg.cuts
    out = []
    for k in keys:
        e = timing.laps[k].elapsed(cuts)
        e[0], e[-1] = 0.0, timing.laps[k].duration
        out.append(np.diff(e))
    return np.stack(out)


def position_noise(timing: SessionTiming) -> float:
    """Robust sigma of position residuals after the final fit (all laps pooled)."""
    r = []
    for lf in timing.laps.values():
        m = (lf.fit.kind == POS) & lf.fit.used
        r.append(lf.fit.z[m] - lf.fit.x[m, 0])
    r = np.concatenate(r)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def sampling_intervals(inputs: list[LapInput]) -> tuple[np.ndarray, np.ndarray]:
    """Empirical location and car_data sampling intervals over the fitted lap windows."""
    dl = np.concatenate([np.diff(li.t_pos) for li in inputs])
    dc = np.concatenate([np.diff(li.t_spd) for li in inputs])
    return dl[dl > 0], dc[dc > 0]


@dataclass
class SessionResult:
    label: str
    circuit: Circuit
    line: Centreline
    registration: Registration
    timing: SessionTiming
    seg: segments.Segmentation
    statuses: dict
    table: LapTable                 # laps passing every check
    table_all: LapTable             # every fitted lap (checks 1-7 passed before fitting)
    held_out_s2: np.ndarray         # predicted - official S1+S2 crossing (s), per lap
    v_ref_x: np.ndarray
    v_ref: np.ndarray
    dt_location: np.ndarray
    dt_car: np.ndarray
    pos_sigma: float
    dropped: dict

    def model(self) -> CircuitModel:
        return empirical_model(self.label, self.line, self.registration.transform,
                               self.timing, self.v_ref_x, self.v_ref, self.seg,
                               self.dt_location, self.dt_car)


def lap_table(label: str, sess: SessionData, timing: SessionTiming, seg: segments.Segmentation,
              statuses: dict, keys) -> LapTable:
    keys = list(keys)
    return LapTable(
        session=label, keys=keys, driver=np.array([k[0] for k in keys], dtype=int),
        team=[sess.drivers[k[0]].team if k[0] in sess.drivers else "?" for k in keys],
        q_segment=np.array([statuses[k].q_segment for k in keys], dtype=int),
        lap_time=np.array([timing.laps[k].duration for k in keys]),
        phase_t=phase_times(timing, seg, keys) if keys else np.zeros((0, len(seg.labels()))),
        labels=seg.labels(), acronym={n: d.acronym for n, d in sess.drivers.items()},
    )


def run_session(directory: str | Path, tracks_dir: str | Path = TRACKS,
                cfg: SmootherConfig = DEFAULT_CONFIG, log=print) -> SessionResult:
    sess = load(directory)
    label = sess.label
    circuit = circuit_for(sess.circuit)
    line = Centreline(load_tumftm(Path(tracks_dir) / f"{circuit.tumftm}.csv"))
    reg = register_session(sess, line)
    log(f"{label}: registration RMS {reg.rms_m:.2f} m, length {100 * reg.length_mismatch:+.2f}%"
        f" -> {reg.reason}")
    if not reg.accepted:
        raise ValueError(f"{label}: registration rejected ({reg.reason})")
    statuses = lapf.classify(sess)
    cand = [lap for lap in sess.laps if statuses[(lap.driver, lap.number)].ok]
    inputs, dropped = lap_inputs(sess, reg.transform, line, cand)
    for key, why in dropped.items():
        statuses[key].reasons.append(f"tracking: {why}")
    timing = fit_session(inputs, line, cfg)
    x, v = reference_speed(timing)
    seg = segments.build(line, timing.s_lines[0], x, car_speed(timing, x, v))
    others = lapf.car_tracks(sess, reg.transform, line)
    lapf.traffic_check(statuses, timing, others, seg)
    fitted = sorted(timing.laps)
    ok = [k for k in fitted if statuses[k].ok]
    held = held_out_sector(inputs, line, timing)
    dl, dc = sampling_intervals(inputs)
    res = SessionResult(
        label=label, circuit=circuit, line=line, registration=reg, timing=timing, seg=seg,
        statuses=statuses, table=lap_table(label, sess, timing, seg, statuses, ok),
        table_all=lap_table(label, sess, timing, seg, statuses, fitted), held_out_s2=held,
        v_ref_x=x, v_ref=v, dt_location=dl, dt_car=dc, pos_sigma=position_noise(timing),
        dropped=dropped,
    )
    log(f"{label}: {len(fitted)} laps fitted, {len(ok)} pass every check, "
        f"{len(seg.complexes)} complexes, held-out S2 robust sd "
        f"{1000 * 1.4826 * np.median(np.abs(held - np.median(held))):.1f} ms")
    return res


# ----------------------------------------------------------------------------- synthetic


@dataclass
class SynthResult:
    est: np.ndarray        # [n_laps, P] estimated phase times
    true: np.ndarray       # [n_laps, P]
    table: LapTable        # estimated, ready for teammate deltas
    table_true: LapTable   # same laps, true phase times
    injected: dict[int, np.ndarray]
    held_out_s2: np.ndarray | None = None   # same statistic as on real sessions


def fit_synthetic(ss: SynthSession, model: CircuitModel, cfg: SmootherConfig = DEFAULT_CONFIG,
                  held_out: bool = False) -> SynthResult:
    """Run the real pipeline's s(t) and phase steps on a synthetic session.

    Lap checks are not exercised (every synthetic lap is a clean push lap); the
    segmentation is the circuit model's, as it would be fixed per circuit in real use.
    ``held_out`` also runs the held-out S2 anchor check, which can be compared with the
    real sessions' value: that is the test of whether the synthetic feed is as bad as the
    real one.
    """
    sess = ss.session
    inputs, dropped = lap_inputs(sess, model.transform, model.line, sess.laps)
    if dropped:
        raise RuntimeError(f"synthetic laps dropped: {dropped}")
    timing = fit_session(inputs, model.line, cfg)
    keys = [(lap.driver, lap.number) for lap in ss.laps]
    est = phase_times(timing, model.seg, keys)
    true = np.stack([lap.phase_t for lap in ss.laps])
    lap_time = np.array([timing.laps[k].duration for k in keys])

    def table(pt):
        return LapTable("synthetic", keys, np.array([k[0] for k in keys]),
                        [sess.drivers[k[0]].team for k in keys], np.ones(len(keys), int),
                        lap_time, pt, model.seg.labels(),
                        {n: d.acronym for n, d in sess.drivers.items()})

    # The truth table must use the official (rounded) lap time so additivity holds.
    true_adj = true + ((lap_time - true.sum(axis=1)) / true.shape[1])[:, None]
    held = held_out_sector(inputs, model.line, timing, cfg) if held_out else None
    return SynthResult(est, true, table(est), table(true_adj), ss.injected, held)
