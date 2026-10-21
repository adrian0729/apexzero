"""Re-run of the P5 harness mirror gate after P5 failed (protocol P5: "If this fails, the
harness gate is re-run. If the gate still passes, the asymmetry is reported as learned").

    uv run python scripts/mirror_gate_rerun.py

Uses the frozen harness (``interp.rollout.mirror_gate``, unchanged since the tag) on the same
two pairs as the pre-tag gate (tier A R 22.9 m / 90 deg at A1, R 44.7 m / 90 deg at A2), and,
as a descriptive extension, on every other tier A mirror pair. The gate replays the scripted
lane keeper's actions (plus a seeded steering wobble) on both members; no trained agent runs.
Writes outputs/psychophysics/post/mirror_gate_rerun.json; nothing of step 1's outputs changes.
"""

import dataclasses
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _bootstrap  # noqa: F401  (makes src/ importable)

_spec = importlib.util.spec_from_file_location("psychophysics", ROOT / "scripts" / "psychophysics.py")
psy = importlib.util.module_from_spec(_spec)
sys.modules["psychophysics"] = psy
_spec.loader.exec_module(psy)

GATE_TOL = 1e-5          # protocol P5: mirrored states to 1e-5 over 400 steps


def tier_a_pairs(bank, cells) -> dict[tuple, tuple]:
    """(R, theta, approach) -> (left, right) cells of every tier A mirror pair, laid out as
    the pre-tag gate laid out its pairs (W = 400, P = v0 dt, D = 200 m when unset)."""
    groups: dict[tuple, list] = {}
    for c in cells:
        st = bank.stimuli[c.stim_id]
        if "A" not in c.tiers or st.spec is None or st.spec.no_cap:
            continue
        key = (float(st.spec.R), float(st.spec.theta_deg), c.approach)
        groups.setdefault(key, []).append(dataclasses.replace(
            c, W=400, P=round(c.v0 * 0.05, 3), D=c.D if c.D is not None else 200.0))
    out = {}
    for key, cs in groups.items():
        cs.sort(key=lambda c: -bank.stimuli[c.stim_id].direction)
        if len(cs) == 2 and cs[0].pair_id == cs[1].pair_id:
            out[key] = tuple(cs)
    return out


def main() -> int:
    from apexzero.interp import rollout

    bank, cells = psy._design()
    pairs = tier_a_pairs(bank, cells)
    gate_keys = [(22.9, 90.0, "A1"), (44.7, 90.0, "A2")]
    rows = []
    for key in sorted(pairs, key=lambda k: (k not in gate_keys, k)):
        try:
            err = float(rollout.mirror_gate(bank, pairs[key], steps=400))
            note = ""
        except (ValueError, RuntimeError) as e:     # report, do not hide, a pair it cannot run
            err, note = float("nan"), f"{type(e).__name__}: {e}"
        rows.append(dict(R=key[0], theta_deg=key[1], approach=key[2], gate_pair=key in gate_keys,
                         max_abs=err, ok=bool(err <= GATE_TOL), note=note))
        print(f"{'GATE ' if key in gate_keys else 'extra'} R{key[0]}/{key[1]:.0f} {key[2]}: "
              f"max |dx| {err:.3g} {'ok' if err <= GATE_TOL else 'FAIL'} {note}")
    gate = [r for r in rows if r["gate_pair"]]
    if len(gate) != 2:
        raise SystemExit(f"the pre-tag gate's two pairs were not both found: {gate}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=False,
                          text=True).stdout.strip()
    res = dict(created=datetime.datetime.now().isoformat(timespec="seconds"), git_head=head,
               tolerance=GATE_TOL, steps=400, modes=["det", "mat (mirrored noise, CRN)"],
               gate_ok=all(r["ok"] for r in gate),
               extension_ok=all(r["ok"] for r in rows if not r["gate_pair"]),
               n_extension=sum(not r["gate_pair"] for r in rows), pairs=rows)
    out = ROOT / "outputs" / "psychophysics" / "post"
    out.mkdir(parents=True, exist_ok=True)
    (out / "mirror_gate_rerun.json").write_text(json.dumps(res, indent=1))
    print(f"gate {'PASS' if res['gate_ok'] else 'FAIL'}; extension "
          f"{'all pass' if res['extension_ok'] else 'has failures'} ({res['n_extension']} pairs)")
    return 0 if res["gate_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
