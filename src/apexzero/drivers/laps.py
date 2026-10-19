"""Which qualifying laps are comparable push laps.

A lap enters the teammate comparison only if every check passes; each failure is
recorded with its reason so the filter can be audited lap by lap:

1. timed: start, duration and all three sectors present;
2. not an out-lap (``is_pit_out_lap``) and not an in-lap (the driver's next lap is an
   out-lap, so this one ended in the pit lane);
3. not deleted by race control (track limits);
4. dry: no rainfall reading during the lap, and not on intermediates or wets;
5. SOFT tyres, from the stints table (whether the set is new is kept as a covariate,
   ``LapStatus.new_set``; the compound is not cross-checked against a second source);
6. no yellow, double-yellow or red flag anywhere on track during the lap;
7. within 102% of the driver's best lap in the same Q segment (among laps passing 1-6);
8. no traffic: no other car within 1.0 s ahead of the driver in any corner window.

Traffic (8) needs every car's position, so it runs after s(t) estimation: the driver's
own time to reach the car ahead's current position is ``elapsed(x_other) - elapsed(x)``
on the driver's smoothed lap. The tow flag (a car within 2 s ahead at the start of a
straight) is kept as a covariate, not a filter, as in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from apexzero.drivers import session as sx
from apexzero.drivers.geometry import Centreline, wrap
from apexzero.drivers.registration import Similarity

PUSH_WITHIN = 1.02
TRAFFIC_GAP_S = 1.0
TOW_GAP_S = 2.0
NEW_SET_MAX_AGE = 1     # laps on the set before this one (the stints table's tyre age)


@dataclass
class LapStatus:
    driver: int
    number: int
    q_segment: int
    compound: str | None
    tyre_age: int | None
    duration: float | None
    reasons: list[str] = field(default_factory=list)
    tow: bool = False

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def new_set(self) -> bool | None:
        """First flying lap on a fresh set: at most the out-lap already done on it (the
        plan's new-set covariate; a used set is back out after a run, age 3 or more)."""
        return None if self.tyre_age is None else self.tyre_age <= NEW_SET_MAX_AGE

    @property
    def key(self) -> tuple[int, int]:
        return (self.driver, self.number)


def classify(sess: sx.SessionData, within: float = PUSH_WITHIN) -> dict[tuple[int, int], LapStatus]:
    """Checks 1-7 (everything that does not need positions)."""
    deleted = sx.deleted_laps(sess)
    flags = sx.flag_intervals(sess)
    by_driver: dict[int, dict[int, sx.LapRecord]] = {}
    for lap in sess.laps:
        by_driver.setdefault(lap.driver, {})[lap.number] = lap
    out = {}
    for lap in sess.laps:
        t0 = lap.date_start
        comp, age = sx.compound(sess, lap.driver, lap.number)
        st = LapStatus(lap.driver, lap.number, sx.q_segment(sess, t0) if t0 else 0, comp, age,
                       lap.duration)
        if not lap.timed:
            st.reasons.append("untimed")
        if lap.pit_out:
            st.reasons.append("out-lap")
        nxt = by_driver[lap.driver].get(lap.number + 1)
        if nxt is not None and nxt.pit_out:
            st.reasons.append("in-lap")
        if (lap.driver, lap.number) in deleted:
            st.reasons.append("deleted")
        if lap.timed:
            t1 = t0 + lap.duration
            if sx.rain_during(sess, t0, t1) or comp in ("INTERMEDIATE", "WET"):
                st.reasons.append("wet")
            if any(a < t1 and b > t0 for a, b, _ in flags):
                st.reasons.append("flag")
        if comp != "SOFT":
            st.reasons.append(f"tyre {comp}")
        out[st.key] = st
    best: dict[tuple[int, int], float] = {}
    for st in out.values():
        if st.ok:
            k = (st.driver, st.q_segment)
            best[k] = min(best.get(k, np.inf), st.duration)
    for st in out.values():
        if st.ok and st.duration > within * best[(st.driver, st.q_segment)]:
            st.reasons.append("not push")
    return out


@dataclass
class CarTrack:
    """A car's whole-session positions as (t, s mod L), on-track samples only."""

    t: np.ndarray
    s: np.ndarray


def plausible_steps(t: np.ndarray, s: np.ndarray, length: float, v_max: float = 100.0,
                    slack: float = 30.0, reacquire: int = 3) -> np.ndarray:
    """Mask of samples whose step from the last kept sample a car could have driven.

    The global projection jumps between branches where a circuit crosses itself
    (Suzuka) and interpolating across such a jump sweeps a phantom car along the track.
    A sample is dropped if it is more than ``v_max * dt + slack`` from the last kept one;
    after ``reacquire`` consecutive drops the next sample is kept anyway, so one bad
    reference sample cannot lock the track out.
    """
    keep = np.ones(len(t), bool)
    last, rejects = 0, 0
    for i in range(1, len(t)):
        step = abs(float(wrap(s[i] - s[last], length)))
        if step <= v_max * (t[i] - t[last]) + slack or rejects >= reacquire:
            last, rejects = i, 0
        else:
            keep[i] = False
            rejects += 1
    return keep


def car_tracks(sess: sx.SessionData, tf: Similarity, line: Centreline,
               max_offset: float = 7.0) -> dict[int, CarTrack]:
    """Every car's samples projected globally; pit-lane and garage samples (more than
    ``max_offset`` m from the centreline) are dropped so they cannot count as traffic,
    and so are branch jumps at crossovers (``plausible_steps``)."""
    out = {}
    for drv, (t, xy) in sess.location.items():
        s, d = line.nearest(tf.apply(xy))
        keep = d <= max_offset
        t, s = t[keep], s[keep]
        ok = plausible_steps(t, s, line.length)
        out[drv] = CarTrack(t[ok], s[ok])
    return out


def _position_at(ct: CarTrack, tq: np.ndarray, length: float, max_dt: float = 0.6) -> np.ndarray:
    """Interpolated s (mod L) at times tq, NaN where no sample is within ``max_dt``."""
    if len(ct.t) < 2:
        return np.full(len(tq), np.nan)
    i = np.clip(np.searchsorted(ct.t, tq), 1, len(ct.t) - 1)
    t0, t1 = ct.t[i - 1], ct.t[i]
    s0 = ct.s[i - 1]
    s1 = s0 + wrap(ct.s[i] - s0, length)
    u = np.clip((tq - t0) / np.maximum(t1 - t0, 1e-6), 0.0, 1.0)
    out = (s0 + u * (s1 - s0)) % length
    ok = (tq >= t0 - max_dt) & (tq <= t1 + max_dt) & (t1 - t0 < 2 * max_dt)
    return np.where(ok, out, np.nan)


def gap_ahead(lap_fit, s_line: float, others: dict[int, CarTrack], length: float,
              x_points: np.ndarray) -> np.ndarray:
    """Smallest time gap (s) to any other car ahead, at each lap distance in x_points.

    At the moment the driver is at x, a car at x + d is ``elapsed(x + d) - elapsed(x)``
    seconds ahead on the driver's own speed profile. Only cars less than half a lap ahead
    count; where no car is ahead the gap is inf.
    """
    t_here = lap_fit.t_start + lap_fit.elapsed(x_points)
    best = np.full(len(x_points), np.inf)
    for drv, ct in others.items():
        if drv == lap_fit.driver:
            continue
        s_other = _position_at(ct, t_here, length)
        ok = np.isfinite(s_other)
        if not ok.any():
            continue
        d = (s_other - s_line - x_points) % length
        ahead = ok & (d > 0) & (d < 0.5 * length)
        if not ahead.any():
            continue
        x_ahead = x_points[ahead] + d[ahead]
        # Beyond the lap end, extrapolate at the finish-line speed.
        inside = x_ahead <= length
        gap = np.empty(ahead.sum())
        gap[inside] = (lap_fit.elapsed(x_ahead[inside])
                       - lap_fit.elapsed(x_points[ahead][inside]))
        if (~inside).any():
            v_end = max(float(lap_fit.speed(np.array([length - 1.0]))[0]), 1.0)
            gap[~inside] = (lap_fit.duration - lap_fit.elapsed(x_points[ahead][~inside])
                            + (x_ahead[~inside] - length) / v_end)
        cur = best[ahead]
        best[ahead] = np.minimum(cur, gap)
    return best


def traffic_check(statuses: dict, timing, others: dict[int, CarTrack], seg,
                  gap_s: float = TRAFFIC_GAP_S, tow_s: float = TOW_GAP_S,
                  step: float = 10.0) -> None:
    """Checks 8 (traffic) and the tow covariate, in place, for laps that have a fit."""
    length = timing.length
    s_line = timing.s_lines[0]
    windows = seg.corner_windows()
    xw = np.concatenate([np.arange(a, b, step) for a, b in windows])
    x_straight = np.array([c.x_apex_end for c in seg.complexes])
    for key, lf in timing.laps.items():
        st = statuses.get(key)
        if st is None:
            continue
        g = gap_ahead(lf, s_line, others, length, xw)
        if np.any(g < gap_s):
            st.reasons.append("traffic")
        gt = gap_ahead(lf, s_line, others, length, x_straight)
        st.tow = bool(np.any(gt < tow_s))
