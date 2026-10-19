"""Resumable, rate-limited archiver for OpenF1 qualifying data.

Raw responses are stored gzipped exactly as received, one file per request, under
``<root>/<year>/<session_key>/``. Every file written is recorded in ``manifest.jsonl``
with its SHA-256, so the archive can be verified later and a run can resume after
any interruption by skipping files that already exist.

OpenF1 data is CC BY-NC-SA 4.0. It is fetched by script and never committed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE_URL = "https://api.openf1.org/v1"
USER_AGENT = "apexzero-research/0.1"

# Endpoints fetched once per session, and once per driver per session.
SESSION_ENDPOINTS = ("drivers", "laps", "stints", "weather", "race_control", "session_result")
DRIVER_ENDPOINTS = ("car_data", "location")


class RateLimiter:
    """Sliding-window limiter that enforces both a per-second and a per-minute cap."""

    def __init__(self, per_second: int = 3, per_minute: int = 29) -> None:
        self.per_second = per_second
        self.per_minute = per_minute
        self._stamps: deque[float] = deque()

    def wait(self) -> None:
        while True:
            now = time.monotonic()
            while self._stamps and now - self._stamps[0] >= 60.0:
                self._stamps.popleft()
            in_last_second = sum(1 for t in self._stamps if now - t < 1.0)
            if len(self._stamps) < self.per_minute and in_last_second < self.per_second:
                self._stamps.append(now)
                return
            delays = [0.01]
            if len(self._stamps) >= self.per_minute:
                delays.append(self._stamps[0] + 60.0 - now)
            if in_last_second >= self.per_second:
                delays.append(self._stamps[-self.per_second] + 1.0 - now)
            time.sleep(max(delays))


class OpenF1Client:
    def __init__(self, limiter: RateLimiter | None = None, max_retries: int = 8) -> None:
        self.limiter = limiter or RateLimiter()
        self.max_retries = max_retries
        self._http = httpx.Client(
            base_url=BASE_URL, headers={"User-Agent": USER_AGENT}, timeout=120.0
        )

    def close(self) -> None:
        self._http.close()

    def get(self, endpoint: str, **params: object) -> list[dict]:
        """GET one endpoint. A 404 means "no results" in OpenF1 and returns []."""
        backoff = 5.0
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                resp = self._http.get(f"/{endpoint}", params=params)
            except httpx.TransportError as exc:
                _log(f"  transport error on {endpoint} {params}: {exc!r}; retry in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 300.0)
                continue
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                return []
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get("Retry-After", backoff))
                _log(f"  HTTP {resp.status_code} on {endpoint} {params}; retry in {wait:.0f}s")
                time.sleep(wait)
                backoff = min(backoff * 2, 300.0)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"giving up on {endpoint} {params} after {self.max_retries} attempts")


@dataclass(frozen=True)
class Session:
    key: int
    year: int
    name: str
    circuit: str
    date_start: str
    date_end: str
    meeting_key: int

    @classmethod
    def from_api(cls, row: dict) -> Session:
        return cls(
            key=int(row["session_key"]),
            year=int(row["year"]),
            name=row["session_name"],
            circuit=row["circuit_short_name"],
            date_start=row["date_start"],
            date_end=row["date_end"],
            meeting_key=int(row["meeting_key"]),
        )


class Archive:
    """Content-addressed-ish file store with an append-only manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.jsonl"

    def path(self, *parts: str | int) -> Path:
        return self.root.joinpath(*map(str, parts))

    def has(self, path: Path) -> bool:
        return path.exists()

    def write(self, path: Path, records: list[dict], url: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = gzip.compress(json.dumps(records, separators=(",", ":")).encode(), mtime=0)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
        entry = {
            "path": str(path.relative_to(self.root)),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "records": len(records),
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self.manifest.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def read_json_gz(path: Path) -> list[dict]:
    return json.loads(gzip.decompress(path.read_bytes()))


def list_sessions(client: OpenF1Client, years: Iterable[int]) -> list[Session]:
    """All completed, non-cancelled qualifying-type sessions (includes sprint qualifying)."""
    now = datetime.now(UTC)
    out: list[Session] = []
    for year in years:
        for row in client.get("sessions", year=year, session_type="Qualifying"):
            if row.get("is_cancelled"):
                continue
            if datetime.fromisoformat(row["date_end"]) > now:
                continue
            out.append(Session.from_api(row))
    return out


def order_sessions(sessions: list[Session], priority: Iterable[tuple[int, str]]) -> list[Session]:
    """Priority (year, circuit) sessions first, main qualifying before sprint, then by date."""
    prio = {(y, c.lower()): i for i, (y, c) in enumerate(priority)}

    def key(s: Session) -> tuple:
        rank = prio.get((s.year, s.circuit.lower()), len(prio))
        return (rank, s.name != "Qualifying", s.date_start)

    return sorted(sessions, key=key)


def planned_requests(archive: Archive, session: Session, drivers: list[int]) -> Iterator[tuple]:
    base = (session.year, session.key)
    for ep in SESSION_ENDPOINTS:
        if ep == "drivers":
            continue
        yield archive.path(*base, f"{ep}.json.gz"), ep, {"session_key": session.key}
    for ep in DRIVER_ENDPOINTS:
        for n in drivers:
            params = {"session_key": session.key, "driver_number": n}
            yield archive.path(*base, ep, f"{n}.json.gz"), ep, params


def fetch_session(client: OpenF1Client, archive: Archive, session: Session) -> int:
    """Fetch everything for one session. Returns the number of new requests made."""
    made = 0
    drivers_path = archive.path(session.year, session.key, "drivers.json.gz")
    if not archive.has(drivers_path):
        rows = client.get("drivers", session_key=session.key)
        archive.write(drivers_path, rows, f"{BASE_URL}/drivers?session_key={session.key}")
        made += 1
    drivers = sorted({int(r["driver_number"]) for r in read_json_gz(drivers_path)})
    meta_path = archive.path(session.year, session.key, "session.json")
    if not meta_path.exists():
        meta_path.write_text(json.dumps(session.__dict__, indent=1))
    for path, ep, params in planned_requests(archive, session, drivers):
        if archive.has(path):
            continue
        rows = client.get(ep, **params)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        archive.write(path, rows, f"{BASE_URL}/{ep}?{query}")
        made += 1
    return made


def _log(msg: str) -> None:
    stamp = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def run(root: Path, years: list[int], priority: list[tuple[int, str]]) -> None:
    archive = Archive(root)
    client = OpenF1Client()
    try:
        sessions = order_sessions(list_sessions(client, years), priority)
        (root / "sessions.json").write_text(json.dumps([s.__dict__ for s in sessions], indent=1))
        _log(f"{len(sessions)} qualifying-type sessions to archive for {years}")
        total = 0
        for i, s in enumerate(sessions, 1):
            t0 = time.monotonic()
            made = fetch_session(client, archive, s)
            total += made
            _log(
                f"{i}/{len(sessions)} {s.year} {s.circuit} {s.name} (key {s.key}): "
                f"{made} new requests in {time.monotonic() - t0:.0f}s"
            )
        _log(f"done: {total} new requests")
    finally:
        client.close()
