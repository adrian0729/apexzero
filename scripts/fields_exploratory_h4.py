"""Step 2, EXPLORATORY (not confirmatory): H4's frozen test with the gate2-trained failures ignored.

    uv run python scripts/fields_exploratory_h4.py

The confirmatory H4 and H4-transfer are 'not testable (gate2-trained)' (outputs/fields/analyse/
decisions.json). This runs the frozen test functions (``analysis.h4_test``, ``h4_transfer``)
on the confirmatory run's per-network statistics with ``descriptive=None``, for G (H4's
statistic) and for the buffer B (not pre-registered as an H4 statistic). It reads only the
committed decisions file and writes outputs/fields/exploratory/h4_exploratory.json. See
docs/findings/receptive-fields-step2.md, Section 6.
"""

import hashlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)

from apexzero.fields import analysis as A

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "outputs" / "fields" / "analyse" / "decisions.json"
EXPECTED_SHA256 = "7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf"


def main() -> int:
    raw = DECISIONS.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != EXPECTED_SHA256:
        raise SystemExit(f"decisions.json is not the committed confirmatory run ({sha})")
    d = json.loads(raw)
    G = {k: float(v) for k, v in d["statistics"].items()}
    B = {k: float(v["B"]["pooled"]["median"]) for k, v in d["edge_descriptive"].items()}
    out = {"note": "EXPLORATORY: gate2-trained failures (runs/dr-pilot-v3-tilt50a2-s2, "
                   "runs/dr-realw-v1-s1) ignored; the confirmatory H4 is 'not testable'",
           "source_sha256": sha}
    for name, stats in (("G", G), ("B", B)):
        h4 = A.h4_test(stats, descriptive=None)
        tr = A.h4_transfer(stats, descriptive=None)
        out[name] = {"H4": h4, "H4_transfer": tr}
        m, s = h4["main"], h4["sensitivity"]
        print(f"{name}: H4 seed-0-free T={m['T']:+.4f} p={m['p']:.5f} p_opposite={m['p_opposite']:.5f} "
              f"({m['n_relabellings']}); with seed 0 p={s['p']:.5f} p_opposite={s['p_opposite']:.5f}")
        tm, ts = tr["main"], tr["sensitivity"]
        print(f"   H4-transfer T={tm['T']:+.4f} p={tm['p']:.5f} p_opposite={tm['p_opposite']:.5f} "
              f"({tm['n_relabellings']}); with seed 0 p={ts['p']:.5f} p_opposite={ts['p_opposite']:.5f}")
    dst = ROOT / "outputs" / "fields" / "exploratory" / "h4_exploratory.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
