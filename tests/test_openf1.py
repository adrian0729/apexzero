import time

from apexzero.data.openf1 import Archive, RateLimiter, Session, order_sessions, read_json_gz


def test_rate_limiter_enforces_per_second_cap():
    limiter = RateLimiter(per_second=3, per_minute=1000)
    t0 = time.monotonic()
    for _ in range(7):
        limiter.wait()
    # 7 calls at <=3/s need at least two full one-second windows.
    assert time.monotonic() - t0 >= 2.0


def _session(key, year, circuit, name="Qualifying", date="2025-01-01"):
    return Session(key, year, name, circuit, date, date, 0)


def test_order_sessions_puts_priority_first_and_main_qualifying_before_sprint():
    sessions = [
        _session(1, 2024, "Spa", date="2024-07-27"),
        _session(2, 2025, "Monza", name="Sprint Qualifying", date="2025-09-05"),
        _session(3, 2025, "Monza", date="2025-09-06"),
        _session(4, 2023, "Sakhir", date="2023-03-04"),
    ]
    ordered = [s.key for s in order_sessions(sessions, [(2025, "monza")])]
    assert ordered == [3, 2, 4, 1]


def test_archive_roundtrip_and_manifest(tmp_path):
    archive = Archive(tmp_path)
    path = archive.path(2025, 123, "laps.json.gz")
    rows = [{"lap_number": 1, "lap_duration": 80.123}]
    archive.write(path, rows, "https://example/laps")
    assert read_json_gz(path) == rows
    assert archive.has(path)
    assert '"records": 1' in archive.manifest.read_text()
