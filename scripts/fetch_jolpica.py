"""Official qualifying results 2023-2026 from jolpica-f1 (the Ergast successor).

    uv run python scripts/fetch_jolpica.py

Writes data/raw/jolpica/qualifying_<year>.json (all pages merged) plus a SHA-256
manifest. A handful of requests, spaced well under jolpica's limits.
"""

import hashlib
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import httpx

BASE = "https://api.jolpi.ca/ergast/f1"
UA = "apexzero-research/0.1"


def fetch_year(http: httpx.Client, year: int) -> list[dict]:
    races, offset, limit = {}, 0, 100
    while True:
        resp = http.get(f"{BASE}/{year}/qualifying.json", params={"limit": limit, "offset": offset})
        resp.raise_for_status()
        data = resp.json()["MRData"]
        for race in data["RaceTable"]["Races"]:
            key = race["round"]
            if key in races:
                races[key]["QualifyingResults"].extend(race["QualifyingResults"])
            else:
                races[key] = race
        offset += limit
        time.sleep(1.0)
        if offset >= int(data["total"]):
            break
    return [races[k] for k in sorted(races, key=int)]


def main(out: Path = Path("data/raw/jolpica")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with httpx.Client(headers={"User-Agent": UA}, timeout=60.0) as http:
        for year in (2023, 2024, 2025, 2026):
            races = fetch_year(http, year)
            path = out / f"qualifying_{year}.json"
            path.write_text(json.dumps(races, indent=1))
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            n = sum(len(r["QualifyingResults"]) for r in races)
            print(f"{year}: {len(races)} rounds, {n} results", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
