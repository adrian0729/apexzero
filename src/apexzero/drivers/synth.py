"""Injection-recovery: synthetic OpenF1-like laps with known per-phase effects.

The gate (plan section 6) asks what the pipeline can recover from data as poor as the
feed. Synthetic laps are built with a known lap-distance -> time map, degraded the way
the feed degrades real laps, and pushed through the unchanged pipeline:

- truth: dt/dx = 1 / v_ref(x) plus, for each phase, an effect (natural lap-to-lap
  variation, plus the injected effect for the "B" driver) spread over the phase as a
  Hann bump, so the effect sits entirely inside its phase and the phase time changes by
  exactly the effect;
- clocks: location and car_data sample times drawn independently from the session's
  empirical sampling-interval distributions;
- positions: centreline point at the true s, Student-t along-track noise at the scale
  measured on the real session, 13 cm lateral noise, mapped into the feed's frame and
  rounded to decimetres;
- speed: centreline rate x path ratio x a per-lap scale x a smooth per-lap line wobble,
  rounded to whole km/h; brake binarised (100 above a deceleration threshold, else 0);
- timing: lap and sector durations to the millisecond, start time off by up to 0.1 s.

Two speed profiles are supported. ``empirical_model`` uses the real session's pooled
reference speed (the realistic case, and no optimum is computed). ``qss_model`` uses a
quasi-steady-state profile on an R-dev centreline, as a check that the error budget does
not hinge on one profile shape; it refuses R-test circuits (the lockbox).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apexzero.drivers.geometry import Centreline
from apexzero.drivers.registration import Similarity
from apexzero.drivers.segments import Segmentation
from apexzero.drivers.session import LapRecord, SessionData, Telemetry
from apexzero.tracks.corners import R_TEST


@dataclass(frozen=True)
class SynthConfig:
    pos_sigma: float = 1.9         # robust sigma of along-track position noise (m)
    pos_df: float = 3.0            # Student-t degrees of freedom (the real tails are heavy)
    lat_sd: float = 0.13           # lateral scatter (snapping audit)
    speed_scale_sd: float = 0.0012  # per-lap speed-sensor scale
    start_jitter: float = 0.1      # date_start error, uniform +/- (s)
    wobble_sd: float = 0.004       # per-lap smooth relative change of the path ratio
    wobble_len: float = 150.0      # its correlation length (m)
    natural_sd: float = 0.010      # natural per-phase lap-to-lap variation (s)
    brake_decel: float = 5.0       # m/s^2
    dx: float = 0.5                # synthesis grid (m)
    pad_s: float = 5.0             # driving before and after each timed lap (s)


DEFAULT_SYNTH = SynthConfig()


@dataclass
class CircuitModel:
    """What the synthesis needs to know about one circuit."""

    name: str
    line: Centreline
    transform: Similarity
    s_lines: np.ndarray           # [3] timing line, S1 and S2 ends (centreline s)
    v_ref_x: np.ndarray
    v_ref: np.ndarray             # centreline rate (m/s)
    ratio_x: np.ndarray
    ratio: np.ndarray             # speed / centreline rate
    seg: Segmentation
    dt_location: np.ndarray       # empirical sampling intervals (s)
    dt_car: np.ndarray

    @property
    def length(self) -> float:
        return self.line.length

    @property
    def x_sectors(self) -> np.ndarray:
        return (self.s_lines[1:] - self.s_lines[0]) % self.length


def _interp_periodic(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, length: float) -> np.ndarray:
    return np.interp(np.asarray(x) % length, np.r_[xp, length], np.r_[fp, fp[0]])


def phase_bumps(seg: Segmentation, x: np.ndarray) -> np.ndarray:
    """[P, len(x)] Hann bumps, each integrating to 1 over its phase interval."""
    cuts = seg.cuts
    dx = x[1] - x[0]
    out = np.zeros((len(cuts) - 1, len(x)))
    for p in range(len(cuts) - 1):
        a, b = cuts[p], cuts[p + 1]
        u = (x - a) / (b - a)
        w = np.where((u > 0) & (u < 1), np.sin(np.pi * u) ** 2, 0.0)
        out[p] = w / (w.sum() * dx)
    return out


def _smooth_noise(rng: np.random.Generator, n: int, dx: float, corr_len: float) -> np.ndarray:
    """Unit-variance Gaussian noise with a Gaussian correlation of length corr_len."""
    w = rng.standard_normal(n + 1)
    k = max(1, int(corr_len / dx))
    ker = np.exp(-0.5 * (np.arange(-3 * k, 3 * k + 1) / k) ** 2)
    sm = np.convolve(np.r_[w[-3 * k:], w, w[:3 * k]], ker, mode="same")[3 * k:-3 * k]
    sm = sm[:n]
    return sm / sm.std()


def _draw_clock(rng: np.random.Generator, t0: float, t1: float, dts: np.ndarray) -> np.ndarray:
    n = int((t1 - t0) / max(np.median(dts), 1e-3) * 1.5) + 10
    t = t0 + rng.uniform(0, np.median(dts)) + np.cumsum(rng.choice(dts, n))
    return t[t < t1]


@dataclass
class SynthLap:
    driver: int
    number: int
    t_start: float
    phase_t: np.ndarray           # true per-phase times (s), sum = lap time
    lap_time: float
    effects: np.ndarray           # per-phase injected + natural offsets (s)


@dataclass
class SynthSession:
    session: SessionData
    laps: list[SynthLap] = field(default_factory=list)
    injected: dict[int, np.ndarray] = field(default_factory=dict)   # driver -> per-phase


def synth_session(model: CircuitModel, pairs: int, laps_per_driver: int,
                  injected: np.ndarray, cfg: SynthConfig = DEFAULT_SYNTH,
                  seed: int = 0) -> SynthSession:
    """``pairs`` teams of two; driver 2i+1 ("B") carries ``injected`` (per phase, s)."""
    rng = np.random.default_rng(seed)
    length = model.length
    s_line = model.s_lines[0]
    x = (np.arange(int(np.ceil(length / cfg.dx))) + 0.5) * cfg.dx
    x = x[x < length]
    base = 1.0 / _interp_periodic(x, model.v_ref_x, model.v_ref, length)
    bumps = phase_bumps(model.seg, x)
    ratio = _interp_periodic(x, model.ratio_x, model.ratio, length)
    cuts = model.seg.cuts
    drivers, teams = {}, {}
    sess = SessionData(key=-1, year=0, name="Synthetic", circuit=model.name, drivers={}, laps=[],
                       stints=[], weather_t=np.zeros(0), rainfall=np.zeros(0),
                       race_control=[{"t": 0.0, "category": "SessionStatus", "flag": None,
                                      "scope": None, "sector": None, "qualifying_phase": 1,
                                      "message": "SESSION STARTED"}])
    out = SynthSession(sess)
    from apexzero.drivers.session import DriverInfo

    clock = 1000.0
    for pair in range(pairs):
        for j in range(2):
            drv = 2 * pair + j + 1
            drivers[drv] = DriverInfo(drv, f"S{drv:02d}", f"T{pair:02d}")
            teams[drv] = f"T{pair:02d}"
            eff_inj = injected if j == 1 else np.zeros(len(cuts) - 1)
            out.injected[drv] = eff_inj
            t_pos, xy_pos, t_car, v_car, b_car = [], [], [], [], []
            for lap_no in range(1, laps_per_driver + 1):
                eff = eff_inj + cfg.natural_sd * rng.standard_normal(len(cuts) - 1)
                pace = base + eff @ bumps
                # Before and after the timed lap the car runs the base profile.
                n_pad = int(cfg.pad_s * 95.0 / cfg.dx)
                pace_ext = np.r_[np.tile(base, 2)[-n_pad:], pace, np.tile(base, 2)[:n_pad]]
                x_ext = (np.arange(len(pace_ext)) - n_pad + 0.5) * cfg.dx
                t_ext = np.r_[0.0, np.cumsum(pace_ext * cfg.dx)]
                xe = np.r_[x_ext - 0.5 * cfg.dx, x_ext[-1] + 0.5 * cfg.dx]
                t_line = np.interp(0.0, xe, t_ext)
                t_ext = t_ext - t_line + clock
                t_at = np.interp(np.r_[cuts], xe, t_ext)
                phase_t = np.diff(t_at)
                lap_time = float(t_at[-1] - t_at[0])
                k = 1.0 + cfg.speed_scale_sd * rng.standard_normal()
                wob = 1.0 + cfg.wobble_sd * _smooth_noise(rng, len(x), cfg.dx, cfg.wobble_len)
                ratio_ext = np.r_[np.tile(ratio, 2)[-n_pad:], ratio * wob, np.tile(ratio, 2)[:n_pad]]
                # Clocks.
                tp = _draw_clock(rng, t_ext[0], t_ext[-1], model.dt_location)
                tc = _draw_clock(rng, t_ext[0], t_ext[-1], model.dt_car)
                # Positions.
                xp = np.interp(tp, t_ext, xe)
                noise = cfg.pos_sigma / 1.4826 / _t_mad(cfg.pos_df) * rng.standard_t(
                    cfg.pos_df, len(tp))
                sp = s_line + xp + noise
                tan = model.line.tangent(sp)
                nrm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
                xy = model.line.point(sp) + nrm * (cfg.lat_sd * rng.standard_normal(len(tp)))[:, None]
                xy = np.round(model.transform.inverse(xy) * 10.0) / 10.0
                # Speed and brake.
                xc = np.interp(tc, t_ext, xe)
                idx = np.clip(np.searchsorted(xe, xc) - 1, 0, len(pace_ext) - 1)
                rate = 1.0 / pace_ext[idx]
                v_meas = k * ratio_ext[idx] * rate
                accel = np.gradient(1.0 / pace_ext, cfg.dx)[idx] * rate
                t_pos.append(tp)
                xy_pos.append(xy)
                t_car.append(tc)
                v_car.append(np.round(v_meas * 3.6))
                b_car.append(np.where(accel < -cfg.brake_decel, 100.0, 0.0))
                # Official timing: durations to 1 ms, start time jittered.
                dur = round(lap_time, 3)
                s1 = round(float(np.interp(model.x_sectors[0], xe, t_ext) - t_at[0]), 3)
                s2 = round(float(np.interp(model.x_sectors[1], xe, t_ext) - t_at[0]), 3) - s1
                s2 = round(s2, 3)
                start = float(t_at[0] + rng.uniform(-cfg.start_jitter, cfg.start_jitter))
                sess.laps.append(LapRecord(drv, lap_no, start, dur,
                                           (s1, s2, round(dur - s1 - s2, 3)), False))
                out.laps.append(SynthLap(drv, lap_no, float(t_at[0]), phase_t, lap_time, eff))
                clock = float(t_ext[-1]) + 60.0
            sess.location[drv] = (np.concatenate(t_pos), np.concatenate(xy_pos))
            tcar = np.concatenate(t_car)
            zero = np.zeros(len(tcar))
            sess.car[drv] = Telemetry(tcar, np.concatenate(v_car), zero, np.concatenate(b_car),
                                      zero, zero)
            sess.stints.append({"driver_number": drv, "lap_start": 1, "lap_end": laps_per_driver,
                                "compound": "SOFT", "tyre_age_at_start": 0})
    sess.drivers = drivers
    return out


def _t_mad(df: float) -> float:
    """Median absolute deviation of a standard Student-t (for scaling to a robust sigma)."""
    from scipy.stats import t as t_dist

    return float(t_dist.ppf(0.75, df))


def empirical_model(name: str, line: Centreline, transform: Similarity, timing, v_ref_x, v_ref,
                    seg: Segmentation, dt_location, dt_car) -> CircuitModel:
    """A circuit model from a real session's own estimates (no optimum involved)."""
    return CircuitModel(name, line, transform, np.asarray(timing.s_lines), np.asarray(v_ref_x),
                        np.asarray(v_ref), timing.ratio_x, timing.ratio, seg,
                        np.asarray(dt_location), np.asarray(dt_car))


def qss_model(track_name: str, tracks_dir, dt_location: np.ndarray, dt_car: np.ndarray,
              ratio: float = 1.0 / 1.006, **seg_kw) -> CircuitModel:
    """A circuit model from a QSS speed profile on an R-dev TUMFTM centreline.

    The profile only shapes synthetic telemetry; its lap time is never reported. R-test
    circuits are refused, so the lockbox stays closed even to synthetic test data.
    """
    from pathlib import Path

    from apexzero.drivers import segments
    from apexzero.oracle.qss import GG, speed_profile
    from apexzero.tracks.track import load_tumftm
    from apexzero.vehicle.params import F1_2025

    if track_name in R_TEST:
        raise ValueError(f"{track_name} is an R-test (lockbox) circuit")
    track = load_tumftm(Path(tracks_dir) / f"{track_name}.csv")
    line = Centreline(track)
    v = speed_profile(track.kappa, track.ds, GG.from_params(F1_2025), closed=True)
    xs = track.s * (line.length / track.length)
    seg = segments.build(line, 0.0, xs, v, **seg_kw)
    lines = np.array([0.0, line.length / 3.0, 2.0 * line.length / 3.0])
    eye = Similarity(1.0, np.eye(2), np.zeros(2))
    return CircuitModel(track_name, line, eye, lines, xs, v, xs, np.full(len(xs), ratio), seg,
                        np.asarray(dt_location), np.asarray(dt_car))
