"""Progress along the lap, s(t), for every lap of a session.

What the data look like (2025 Monza qualifying, measured while building this):

- Projected positions scatter ~2.2 m along the track (robust sigma), at every speed, on
  top of a ~3 m pattern that repeats for every car at the same place (track-model
  mapping and the racing line). The 13 cm of the snapping audit is the *lateral* scatter.
- Speed is whole km/h and smooth; there is no measurable lag between the location and
  car_data clocks (best lag within +/-0.05 s).
- Centreline progress / integrated speed = 1.006 +/- 0.001 per lap: the cars drive a line
  ~0.6% shorter than the TUMFTM centreline, and that ratio varies along the lap.
- ``date_start`` is off by up to ~0.1 s, but lap and sector durations are exact.

So, per session:

1. Track each lap's samples with a windowed projection (no jumps at crossovers).
2. Smooth without anchors; estimate where the timing lines are on the centreline
   (circular median over laps) and each lap's start-time correction delta, alternating
   twice. The mean of delta is not identifiable from the data (shifting every start time
   just moves the lines), so the median delta is pinned to zero.
3. Smooth with hard anchors, positions only; pool the speed / progress ratio over all
   laps into a path-ratio profile r(x) and a per-lap speed scale k.
4. Smooth with hard anchors, positions and speed (speed = k r(x) ds/dt).

Lap distance x runs from 0 at the timing line to the centreline length L, so segment
times taken from t(x) sum exactly to the official lap time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apexzero.drivers.geometry import Centreline, wrap
from apexzero.drivers.registration import Similarity
from apexzero.drivers.session import KMH_TO_MS, LapRecord, SessionData
from apexzero.drivers.smoother import (
    ANCHOR,
    DEFAULT_CONFIG,
    POS,
    SPEED,
    Smoothed,
    SmootherConfig,
    smooth,
)


def track_samples(line: Centreline, t: np.ndarray, xy: np.ndarray, v0: float = 50.0,
                  s_first: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unwrapped s and distance from the line for one car's consecutive samples.

    Each sample is projected within a window around the previous one moved on at the
    last observed rate, so a figure-eight crossing (Suzuka) or a hairpin's return leg can
    never capture the car.
    """
    n = len(t)
    s = np.empty(n)
    dist = np.empty(n)
    if s_first is None:
        s0, d0 = line.nearest(xy[:1])
    else:
        s0, d0 = line.project_window(xy[:1], s_first, 150.0)
    s[0], dist[0] = s0[0], d0[0]
    v = v0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        guess = s[i - 1] + v * dt
        si, di = line.project_window(xy[i:i + 1], guess, 60.0 + 100.0 * dt)
        s[i] = guess + wrap(si[0] - guess, line.length)
        dist[i] = di[0]
        if dt > 0.05:
            v = float(np.clip((s[i] - s[i - 1]) / dt, 5.0, 100.0))
    return s, dist


@dataclass
class LapInput:
    """One lap's raw samples, already projected (s unwrapped, continuous)."""

    driver: int
    number: int
    date_start: float
    duration: float
    sectors: tuple[float, float, float]
    t_pos: np.ndarray
    s_pos: np.ndarray
    d_pos: np.ndarray
    t_spd: np.ndarray
    v_spd: np.ndarray        # m/s, as measured (whole km/h)

    @property
    def key(self) -> tuple[int, int]:
        return (self.driver, self.number)

    @property
    def taus(self) -> np.ndarray:
        """Elapsed times of the four line crossings: start, S1 end, S2 end, finish."""
        s1, s2, _ = self.sectors
        return np.array([0.0, s1, s1 + s2, self.duration])


def lap_inputs(sess: SessionData, tf: Similarity, line: Centreline, laps: list[LapRecord],
               pad: float = 2.5, max_gap: float = 2.0) -> tuple[list[LapInput], dict]:
    """Project every requested timed lap; return the inputs and {key: reason} for drops."""
    out, dropped = [], {}
    for lap in laps:
        if not lap.timed:
            dropped[(lap.driver, lap.number)] = "untimed"
            continue
        if lap.driver not in sess.location or lap.driver not in sess.car:
            dropped[(lap.driver, lap.number)] = "no telemetry"
            continue
        t_all, xy_all = sess.location[lap.driver]
        lo, hi = lap.date_start - pad, lap.date_start + lap.duration + pad
        m = (t_all >= lo) & (t_all <= hi)
        car = sess.car[lap.driver]
        mc = (car.t >= lo) & (car.t <= hi)
        if m.sum() < 20 or mc.sum() < 20:
            dropped[(lap.driver, lap.number)] = "too few samples"
            continue
        t = t_all[m]
        gaps = np.diff(np.r_[lo, t, hi])
        if gaps.max() > max_gap + pad:
            dropped[(lap.driver, lap.number)] = f"position gap {gaps.max():.1f} s"
            continue
        s, d = track_samples(line, t, tf.apply(xy_all[m]))
        out.append(LapInput(lap.driver, lap.number, lap.date_start, float(lap.duration),
                            tuple(float(x) for x in lap.sectors), t, s, d,
                            car.t[mc], car.speed_kmh[mc] * KMH_TO_MS))
    return out, dropped


def _measurements(li: LapInput, use_speed: bool, anchors: tuple[np.ndarray, np.ndarray] | None,
                  gain: np.ndarray | None = None):
    t = [li.t_pos]
    kind = [np.full(len(li.t_pos), POS)]
    z = [li.s_pos]
    g = [np.ones(len(li.t_pos))]
    if use_speed:
        t.append(li.t_spd)
        kind.append(np.full(len(li.t_spd), SPEED))
        z.append(li.v_spd)
        g.append(np.ones(len(li.t_spd)) if gain is None else gain)
    if anchors is not None:
        t.append(anchors[0])
        kind.append(np.full(len(anchors[0]), ANCHOR))
        z.append(anchors[1])
        g.append(np.ones(len(anchors[0])))
    return np.concatenate(t), np.concatenate(kind), np.concatenate(z), np.concatenate(g)


def _circ_median(s: np.ndarray, length: float) -> float:
    ref = s[0]
    return float((ref + np.median(wrap(s - ref, length))) % length)


@dataclass
class LapFit:
    driver: int
    number: int
    t_start: float           # corrected lap start (date_start + delta)
    delta: float
    duration: float
    s_start: float           # unwrapped s of the timing line at t_start
    k_speed: float
    fit: Smoothed
    n_rejected: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.driver, self.number)

    def elapsed(self, x: np.ndarray) -> np.ndarray:
        """Time since the lap started at which lap distance x was reached."""
        return self.fit.t_at(self.s_start + np.asarray(x, dtype=float)) - self.t_start

    def speed(self, x: np.ndarray) -> np.ndarray:
        """Centreline progress rate ds/dt at lap distance x."""
        return self.fit.v_at(self.fit.t_at(self.s_start + np.asarray(x, dtype=float)))

    def sigma_t(self, x: np.ndarray) -> np.ndarray:
        """Approximate 1-sigma timing uncertainty at x (position sigma / speed)."""
        tt = self.fit.t_at(self.s_start + np.asarray(x, dtype=float))
        return self.fit.sigma_s_at(tt) / np.maximum(self.fit.v_at(tt), 1.0)


@dataclass
class SessionTiming:
    length: float
    s_lines: np.ndarray            # [3] centreline s of the timing line, S1 and S2 ends
    ratio_x: np.ndarray            # lap-distance grid of the path-ratio profile
    ratio: np.ndarray              # speed / centreline rate
    laps: dict[tuple[int, int], LapFit] = field(default_factory=dict)
    delta_spread: float = 0.0      # robust sd of the start-time corrections

    @property
    def x_sectors(self) -> np.ndarray:
        """Lap distance of the S1 and S2 lines."""
        return (self.s_lines[1:] - self.s_lines[0]) % self.length


def _crossing_times(fit: Smoothed, s_targets: np.ndarray) -> np.ndarray:
    return fit.t_at(s_targets)


def _unwrap_lines(s_lines: np.ndarray, near: float, length: float) -> np.ndarray:
    """Unwrapped anchor positions [start, S1, S2, finish] with start nearest ``near``."""
    s0 = near + wrap(s_lines[0] - near, length)
    off = (s_lines[1:] - s_lines[0]) % length
    return np.array([s0, s0 + off[0], s0 + off[1], s0 + length])


def fit_session(inputs: list[LapInput], line: Centreline, cfg: SmootherConfig = DEFAULT_CONFIG,
                ratio_bin: float = 10.0, ratio_smooth: float = 50.0, iterations: int = 2,
                drop_anchor: int | None = None,
                fixed: SessionTiming | None = None) -> SessionTiming:
    """Estimate s(t) for every lap; see the module docstring for the steps.

    ``drop_anchor`` (1 or 2) leaves out one sector line for the held-out-anchor gate.
    ``fixed`` reuses the line positions and path-ratio profile of an earlier fit (so a
    held-out refit changes only what it should).
    """
    length = line.length
    if not inputs:
        raise ValueError("no laps to fit")
    # Step 2: unanchored smoothing, then timing lines and start corrections.
    free = {}
    for li in inputs:
        t, kind, z, g = _measurements(li, use_speed=True, anchors=None,
                                      gain=np.full(len(li.t_spd), 1.0 / 1.006))
        free[li.key] = smooth(t, kind, z, gain=g, cfg=cfg)
    if fixed is None:
        delta = {li.key: 0.0 for li in inputs}
        for _ in range(iterations):
            s_lines = np.array([
                _circ_median(np.array([free[li.key].s_at(li.date_start + delta[li.key] + li.taus[k])
                                       for li in inputs]) % length, length)
                for k in range(3)])
            for li in inputs:
                fit = free[li.key]
                near = float(fit.s_at(li.date_start))
                targets = _unwrap_lines(s_lines, near, length)
                delta[li.key] = float(np.mean(_crossing_times(fit, targets)
                                              - li.date_start - li.taus))
            med = float(np.median(list(delta.values())))
            delta = {k: v - med for k, v in delta.items()}
    else:
        s_lines = fixed.s_lines
        delta = {}
        for li in inputs:
            fit = free[li.key]
            targets = _unwrap_lines(s_lines, float(fit.s_at(li.date_start)), length)
            keep = [k for k in range(4) if k != drop_anchor]
            delta[li.key] = float(np.mean((_crossing_times(fit, targets) - li.date_start
                                           - li.taus)[keep]))

    def anchors(li: LapInput) -> tuple[np.ndarray, np.ndarray, float, float]:
        t0 = li.date_start + delta[li.key]
        s_un = _unwrap_lines(s_lines, float(free[li.key].s_at(t0)), length)
        keep = [k for k in range(4) if k != drop_anchor]
        return (t0 + li.taus)[keep], s_un[keep], t0, float(s_un[0])

    # Step 3: anchored, positions only -> path-ratio profile and per-lap speed scale.
    if fixed is None:
        xs, rs, per_lap = [], [], {}
        for li in inputs:
            ta, sa, t0, s0 = anchors(li)
            fit = smooth(*_measurements(li, use_speed=False, anchors=(ta, sa))[:3], cfg=cfg)
            ok = (li.t_spd > li.t_pos[0]) & (li.t_spd < li.t_pos[-1])
            rate = fit.v_at(li.t_spd[ok])
            x = (fit.s_at(li.t_spd[ok]) - s0) % length
            good = rate > 5.0
            xs.append(x[good])
            rs.append(li.v_spd[ok][good] / rate[good])
            per_lap[li.key] = (x[good], rs[-1])
        x_all, r_all = np.concatenate(xs), np.concatenate(rs)
        grid = np.arange(0.0, length, ratio_bin)
        idx = np.minimum((x_all // ratio_bin).astype(int), len(grid) - 1)
        med = np.array([np.median(r_all[idx == i]) if np.any(idx == i) else np.nan
                        for i in range(len(grid))])
        med = np.where(np.isfinite(med), med, np.nanmedian(med))
        w = max(1, round(ratio_smooth / ratio_bin))
        ext = np.r_[med[-w:], med, med[:w]]
        ratio = np.convolve(ext, np.ones(w) / w, mode="same")[w:-w]
    else:
        grid, ratio = fixed.ratio_x, fixed.ratio
        per_lap = None

    def r_at(x):
        return np.interp(x % length, np.r_[grid, length], np.r_[ratio, ratio[0]])

    timing = SessionTiming(length, s_lines, grid, ratio)
    # Step 4: anchored, positions and speed.
    for li in inputs:
        ta, sa, t0, s0 = anchors(li)
        if per_lap is not None:
            x, r = per_lap[li.key]
            k = float(np.median(r / r_at(x))) if len(x) else 1.0
        else:
            fit0 = smooth(*_measurements(li, use_speed=False, anchors=(ta, sa))[:3], cfg=cfg)
            ok = (li.t_spd > li.t_pos[0]) & (li.t_spd < li.t_pos[-1])
            rate = fit0.v_at(li.t_spd[ok])
            x = (fit0.s_at(li.t_spd[ok]) - s0) % length
            good = rate > 5.0
            k = float(np.median(li.v_spd[ok][good] / rate[good] / r_at(x[good])))
        # Path ratio is evaluated at the positions-and-anchors estimate of s (fixed point).
        pre = free[li.key]
        x_spd = (pre.s_at(li.t_spd) - s0) % length
        t, kind, z, g = _measurements(li, use_speed=True, anchors=(ta, sa),
                                      gain=k * r_at(x_spd))
        fit = smooth(t, kind, z, gain=g, cfg=cfg)
        timing.laps[li.key] = LapFit(li.driver, li.number, t0, delta[li.key], li.duration, s0,
                                     k, fit, int((~fit.used).sum()))
    d = np.array(list(delta.values()))
    timing.delta_spread = float(1.4826 * np.median(np.abs(d - np.median(d))))
    return timing


def held_out_sector(inputs: list[LapInput], line: Centreline, timing: SessionTiming,
                    cfg: SmootherConfig = DEFAULT_CONFIG, which: int = 2) -> np.ndarray:
    """Refit without one sector line and return predicted - official crossing time (s).

    ``which`` = 2 drops the S2/S3 line (the plan's gate); the prediction is the time the
    refitted s(t) reaches that line, and the truth is the official S1 + S2.
    """
    refit = fit_session(inputs, line, cfg, drop_anchor=which, fixed=timing)
    x_line = timing.x_sectors[which - 1]
    err = []
    for li in inputs:
        lf = refit.laps[li.key]
        err.append(float(lf.elapsed(np.array([x_line]))[0] - li.taus[which]))
    return np.array(err)
