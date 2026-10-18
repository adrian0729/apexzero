"""Snapping audit over every downloaded qualifying session -> docs/findings/snapping-audit.md

    uv run python scripts/run_audit.py [--max-sessions N]

Data: OpenF1 (CC BY-NC-SA 4.0). Only aggregate statistics are written to the report.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from apexzero.data.audit import (
    audit,
    driver_signal,
    flying_laps,
    lap_samples,
    load_session,
    reference_curve,
)

ROOT = Path("data/raw/openf1")
OUT = Path("docs/findings")


def sessions(max_n=None):
    out = []
    for meta_path in sorted(ROOT.glob("*/*/session.json")):
        meta = json.loads(meta_path.read_text())
        d = meta_path.parent
        if meta["name"] != "Qualifying" or not (d / "location").exists():
            continue
        if len(list((d / "location").glob("*.json.gz"))) < 18:
            continue  # still downloading
        out.append((meta, d))
    return out[:max_n] if max_n else out


def main(max_n=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rows, example, skipped = [], None, []
    for meta, d in sessions(max_n):
        sess = load_session(d)
        try:
            a, s = audit(sess), driver_signal(sess)
        except ValueError as exc:
            skipped.append(f"{meta['year']} {meta['circuit']}: {exc}")
            print("skip", skipped[-1], flush=True)
            continue
        rows.append((meta, a, s))
        if example is None or meta["circuit"] == "Monza":
            example = (meta, sess)
        print(f"{a.session}: median {a.median_m*100:.1f} cm, ICC {s.median_icc:+.3f}, driver offset p95 {s.p95_driver_offset_m*100:.1f} cm (max {s.max_driver_offset_m*100:.0f}), planted ICC {s.planted_median_icc:+.3f} p95 {s.planted_p95_driver_offset_m*100:.0f} cm", flush=True)

    # Figure: distance distribution for one session, plus ICC real vs planted for all.
    meta, sess = example
    laps = flying_laps(sess)
    pole = min(laps, key=lambda lap: lap.duration)
    tree = cKDTree(reference_curve(*lap_samples(sess, pole)))
    dists = np.concatenate([tree.query(lap_samples(sess, lap)[1])[0] for lap in laps if lap is not pole])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].hist(np.clip(dists, 0, 2.0), bins=80, color="#4c72b0")
    ax[0].set_xlabel("distance from the pole lap's curve (m), all drivers' flying laps")
    ax[0].set_ylabel("samples")
    ax[0].set_title(f"{meta['year']} {meta['circuit']} qualifying")
    x = np.arange(len(rows))
    ax[1].bar(x - 0.2, [r[2].median_icc for r in rows], 0.4, label="real data", color="#4c72b0")
    ax[1].bar(x + 0.2, [r[2].planted_median_icc for r in rows], 0.4, label="with a planted 0.5 m line", color="#dd8452")
    ax[1].set_xticks(x, [f"{r[0]['circuit']}\n{r[0]['year']}" for r in rows], fontsize=7)
    ax[1].set_ylabel("median driver ICC of lateral offset")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "snapping-audit.png", dpi=150)

    lines = [
        "# Snapping audit: the public F1 position feed has no racing-line information",
        "",
        "Every flying lap's 3.7 Hz location samples are compared with a cubic-spline curve through",
        "the pole lap. Two tests: (1) distance from the pole lap's curve; (2) whether a driver's",
        "*signed* lateral offset is consistent across their own laps, per 50 m station",
        "(one-way ANOVA, intraclass correlation). Positive controls: pit-lane samples, laps",
        "shifted by known 1-3 m offsets, and a planted 0.5 m line for one driver.",
        "",
        "| Session | Laps | Median dist. | p95 | Pit-lane p95 | Injected 2 m -> | Driver ICC | Stations p<0.01 | Driver offset p95 | Planted ICC | Planted detected | Planted offset p95 | Excluded cars |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for meta, a, s in rows:
        lines.append(
            f"| {meta['year']} {meta['circuit']} | {a.n_laps} | {a.median_m*100:.1f} cm | {a.p95_m*100:.0f} cm "
            f"| {a.pit_lane_p95_m:.1f} m | {a.injected[2.0]:.2f} m | {s.median_icc:+.3f} "
            f"| {s.frac_bins_significant*100:.1f}% | {s.p95_driver_offset_m*100:.1f} cm | {s.planted_median_icc:+.2f} "
            f"| {s.planted_frac_significant*100:.0f}% | {s.planted_p95_driver_offset_m*100:.0f} cm "
            f"| {', '.join(map(str, a.flagged)) or '-'} |"
        )
    lines += [
        "",
        "![audit](snapping-audit.png)",
        "",
        "**Reading.** Every driver sits ~13 cm from the pole lap's curve, with no driver-specific",
        "offset (ICC ~ 0, significant stations at about the false-positive rate), while the same",
        "test detects a planted 0.5 m line at almost every station and recovers injected offsets",
        "exactly. The feed can represent off-line positions (the pit lane is metres away), so",
        "the positions are projected onto a single track model. Consequence: driver comparisons",
        "use speed, throttle, brake and time as functions of distance, never lateral lines.",
        "",
        "*Excluded cars* had a broken coordinate stream: positions far from every other car's",
        "everywhere on the lap, straights included (for example 2023 Jeddah car 21, ~3 m). That",
        "is a per-car data fault, not line information, and is reported rather than hidden.",
        "",
        "Data: OpenF1 (CC BY-NC-SA 4.0), aggregate statistics only.",
    ]
    follow = [(m, a, sg) for m, a, sg in rows
              if sg.frac_bins_significant > 0.05 or sg.p95_driver_offset_m > 0.2]
    if follow:
        lines += ["", "**Flagged for follow-up** (more than 5% of stations significant, or",
                  "driver offsets above 20 cm at p95). Effects of a few centimetres are per-car",
                  "artefacts such as antenna position or latency, far below racing-line scale",
                  "(0.5-2 m); larger ones are open items, not conclusions:"]
        for m, a, sg in follow:
            lines.append(f"- {m['year']} {m['circuit']}: {sg.frac_bins_significant*100:.1f}% of stations "
                         f"significant, driver offset p95 {sg.p95_driver_offset_m*100:.1f} cm, ICC {sg.median_icc:+.3f}")
    if skipped:
        lines += ["", "Skipped sessions:", *[f"- {x}" for x in skipped]]
    (OUT / "snapping-audit.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'snapping-audit.md'} ({len(rows)} sessions)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int)
    main(ap.parse_args().max_sessions)
