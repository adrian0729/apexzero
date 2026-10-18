"""Fetch the TUMFTM racetrack-database centrelines (LGPL-3.0) into data/tracks/tumftm/.

    uv run python scripts/fetch_tracks.py

The files are fetched by script and never vendored into this repository.
"""

import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import httpx

REPO = "TUMFTM/racetrack-database"
REF = "master"
NAMES = [
    "Austin", "BrandsHatch", "Budapest", "Catalunya", "Hockenheim", "IMS", "Melbourne",
    "MexicoCity", "Montreal", "Monza", "MoscowRaceway", "Norisring", "Nuerburgring",
    "Oschersleben", "Sakhir", "SaoPaulo", "Sepang", "Shanghai", "Silverstone", "Sochi", "Spa",
    "Spielberg", "Suzuka", "YasMarina", "Zandvoort",
]


def main(out: Path = Path("data/tracks/tumftm")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with httpx.Client(timeout=60.0) as http:
        for name in NAMES:
            path = out / f"{name}.csv"
            if not path.exists():
                url = f"https://raw.githubusercontent.com/{REPO}/{REF}/tracks/{name}.csv"
                resp = http.get(url)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps({"repo": REPO, "ref": REF, "sha256": manifest}, indent=1))
    print(f"{len(manifest)} tracks in {out}")


if __name__ == "__main__":
    main()
