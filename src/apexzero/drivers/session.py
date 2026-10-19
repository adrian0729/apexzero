"""Load one archived OpenF1 qualifying session into arrays the driver pipeline can use.

What each endpoint contributes, and what it does not:

- ``location``: x, y in decimetres at ~3.7 Hz, snapped to a track model, so it carries
  progress along the lap but no lateral line.
- ``car_data``: speed in whole km/h, throttle, binary-ish brake, gear, DRS, on its own
  ~3.7 Hz clock (not the location clock).
- ``laps``: lap and sector durations are official timing (1 ms). ``date_start`` is not:
  consecutive laps disagree with ``start[n+1] - start[n] = duration[n]`` by up to
  +/-0.2 s, so it is treated as a noisy prior and re-estimated from positions.
- ``race_control``: Q-segment starts, deleted laps (track limits), yellow and red flags.
- ``stints``: tyre compound per lap range; ``weather``: rainfall per minute.

Timestamps are converted to float seconds since the epoch (UTC).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from apexzero.data.openf1 import read_json_gz

DM_TO_M = 0.1
KMH_TO_MS = 1.0 / 3.6


def parse_times(stamps: list[str]) -> np.ndarray:
    """ISO-8601 UTC stamps -> float epoch seconds (vectorised for the common '+00:00' form)."""
    if all(s.endswith("+00:00") for s in stamps):
        arr = np.array([s[:-6] for s in stamps], dtype="datetime64[us]")
        return arr.astype("int64") / 1e6
    return np.array([datetime.fromisoformat(s).timestamp() for s in stamps])


@dataclass(frozen=True)
class LapRecord:
    driver: int
    number: int
    date_start: float | None        # noisy (+/-0.1 s); see module docstring
    duration: float | None          # official, 1 ms
    sectors: tuple[float | None, float | None, float | None]
    pit_out: bool

    @property
    def timed(self) -> bool:
        return (self.date_start is not None and self.duration is not None
                and all(x is not None for x in self.sectors))


@dataclass(frozen=True)
class DriverInfo:
    number: int
    acronym: str
    team: str


@dataclass
class Telemetry:
    t: np.ndarray            # [N] epoch seconds
    speed_kmh: np.ndarray    # [N] integer km/h
    throttle: np.ndarray
    brake: np.ndarray        # 0 or 100 in practice
    gear: np.ndarray
    drs: np.ndarray


@dataclass
class SessionData:
    key: int
    year: int
    name: str
    circuit: str
    drivers: dict[int, DriverInfo]
    laps: list[LapRecord]
    stints: list[dict]
    weather_t: np.ndarray
    rainfall: np.ndarray
    race_control: list[dict]
    location: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    car: dict[int, Telemetry] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.year} {self.circuit} {self.name}"

    def lap(self, driver: int, number: int) -> LapRecord | None:
        for lap in self.laps:
            if lap.driver == driver and lap.number == number:
                return lap
        return None


def load(directory: str | Path, telemetry: bool = True) -> SessionData:
    d = Path(directory)
    meta = json.loads((d / "session.json").read_text())
    drivers = {}
    for row in read_json_gz(d / "drivers.json.gz"):
        n = int(row["driver_number"])
        drivers[n] = DriverInfo(n, row.get("name_acronym") or str(n), row.get("team_name") or "?")
    laps = []
    for row in read_json_gz(d / "laps.json.gz"):
        start = row.get("date_start")
        laps.append(LapRecord(
            driver=int(row["driver_number"]), number=int(row["lap_number"]),
            date_start=None if start is None else float(parse_times([start])[0]),
            duration=row.get("lap_duration"),
            sectors=tuple(row.get(f"duration_sector_{i}") for i in (1, 2, 3)),
            pit_out=bool(row.get("is_pit_out_lap")),
        ))
    laps.sort(key=lambda lap: (lap.driver, lap.number))
    weather = read_json_gz(d / "weather.json.gz")
    rc = read_json_gz(d / "race_control.json.gz")
    if rc:
        for m, t in zip(rc, parse_times([m["date"] for m in rc]), strict=True):
            m["t"] = float(t)
        rc.sort(key=lambda m: m["t"])
    sess = SessionData(
        key=int(meta["key"]), year=int(meta["year"]), name=meta["name"], circuit=meta["circuit"],
        drivers=drivers, laps=laps, stints=read_json_gz(d / "stints.json.gz"),
        weather_t=parse_times([w["date"] for w in weather]) if weather else np.zeros(0),
        rainfall=np.array([float(w.get("rainfall") or 0) for w in weather]),
        race_control=rc,
    )
    if telemetry:
        for path in sorted((d / "location").glob("*.json.gz")):
            rows = [r for r in read_json_gz(path) if not (r["x"] == 0 and r["y"] == 0)]
            if rows:
                t = parse_times([r["date"] for r in rows])
                xy = np.array([[r["x"], r["y"]] for r in rows], dtype=float) * DM_TO_M
                o = np.argsort(t, kind="stable")
                sess.location[int(path.name.split(".")[0])] = (t[o], xy[o])
        for path in sorted((d / "car_data").glob("*.json.gz")):
            rows = read_json_gz(path)
            if rows:
                t = parse_times([r["date"] for r in rows])
                o = np.argsort(t, kind="stable")
                col = {k: np.array([r.get(k) or 0 for r in rows], dtype=float)[o]
                       for k in ("speed", "throttle", "brake", "n_gear", "drs")}
                sess.car[int(path.name.split(".")[0])] = Telemetry(
                    t[o], col["speed"], col["throttle"], col["brake"], col["n_gear"], col["drs"])
    return sess


# ----------------------------------------------------------------------------- race control

_DELETED = re.compile(r"CAR (\d+) \(\w+\) (?:TIME \S+ )?(?:LAP )?DELETED\b.*?\bLAP (\d+)\b")


def deleted_laps(sess: SessionData) -> set[tuple[int, int]]:
    """(driver, lap) pairs whose time was deleted (track limits and the like)."""
    out = set()
    for m in sess.race_control:
        hit = _DELETED.search(m.get("message") or "")
        if hit and "REINSTATED" not in m["message"]:
            out.add((int(hit.group(1)), int(hit.group(2))))
    return out


def phase_starts(sess: SessionData) -> list[tuple[float, int]]:
    """(time, Q segment) for every 'SESSION STARTED', in time order."""
    out = []
    for m in sess.race_control:
        if m.get("category") == "SessionStatus" and "STARTED" in (m.get("message") or ""):
            out.append((m["t"], int(m.get("qualifying_phase") or len(out) + 1)))
    return out


def q_segment(sess: SessionData, t: float) -> int:
    """Q segment in progress at time t (laps started after the flag still count)."""
    seg = 1
    for start, phase in phase_starts(sess):
        if t + 1e-6 >= start:
            seg = phase
    return seg


def flag_intervals(sess: SessionData) -> list[tuple[float, float, str]]:
    """Periods with a yellow, double yellow or red flag anywhere on track.

    A sector yellow lasts until a CLEAR for the same sector (or a track-wide CLEAR or
    GREEN); a red lasts until the next GREEN. Open intervals run to the end of the feed.
    """
    open_: dict[object, tuple[float, str]] = {}
    out = []
    end = sess.race_control[-1]["t"] if sess.race_control else 0.0
    for m in sess.race_control:
        flag, scope, t = m.get("flag"), m.get("scope"), m["t"]
        if m.get("category") != "Flag":
            continue
        if flag in ("YELLOW", "DOUBLE YELLOW", "RED"):
            key = m.get("sector") if scope == "Sector" else ("track", flag)
            open_.setdefault(key, (t, flag))
        elif flag == "CLEAR" and scope == "Sector":
            if m.get("sector") in open_:
                t0, f = open_.pop(m.get("sector"))
                out.append((t0, t, f))
        elif flag in ("CLEAR", "GREEN"):
            for key in list(open_):
                t0, f = open_.pop(key)
                out.append((t0, t, f))
    out += [(t0, end, f) for t0, f in open_.values()]
    return sorted(out)


def compound(sess: SessionData, driver: int, lap: int) -> tuple[str | None, int | None]:
    """(compound, tyre age at the lap) from the stints table."""
    for st in sess.stints:
        if st["driver_number"] == driver and st["lap_start"] <= lap <= (st["lap_end"] or 10**6):
            age0 = st.get("tyre_age_at_start")
            return st.get("compound"), None if age0 is None else age0 + lap - st["lap_start"]
    return None, None


def rain_during(sess: SessionData, t0: float, t1: float, pad: float = 60.0) -> bool:
    """Any rainfall reading within ``pad`` seconds of [t0, t1] (readings are per minute)."""
    m = (sess.weather_t >= t0 - pad) & (sess.weather_t <= t1 + pad)
    return bool(np.any(sess.rainfall[m] > 0))
