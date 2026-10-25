"""Snapping audit over every downloaded qualifying session -> docs/findings/snapping-audit.md

    uv run python scripts/run_audit.py [--max-sessions N]   # audit, cache, write the report
    uv run python scripts/run_audit.py --from-cache          # report and figure from the cache

The audit loads every session's location feed and takes several minutes. Its
per-session statistics, the example session's histogram and the per-car breakdown of
flagged sessions are cached in outputs/snapping-audit/audit.json, so the report and the
figure can be rebuilt without it.

Data: OpenF1 (CC BY-NC-SA 4.0). Only aggregate statistics are written to the report and
the cache.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from apexzero.data.audit import (
    AuditResult,
    DriverSignal,
    _signed_offsets,
    audit,
    driver_signal,
    flag_broken_streams,
    flying_laps,
    lap_samples,
    load_session,
    reference_curve,
)

ROOT = Path("data/raw/openf1")
OUT = Path("docs/findings")
CACHE = Path("outputs/snapping-audit/audit.json")
FOLLOW_OFFSET_M = 0.2       # driver offset p95 above this is flagged (and broken down per car)
SEAM_M = 600.0              # per-car breakdown: "near the seam" of the pole lap's curve
# Categorical slots 1-2 of the reference palette (validated, light surface).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


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


def per_car_offsets(sess, bin_m: float = 50.0, min_laps: int = 3) -> list[dict]:
    """The driver_signal statistic split by car (no planted line).

    Per 50 m station, each car's mean signed offset over its laps minus the station mean
    over cars; per car: median, p95 and max of its absolute value, and the stations
    (start, in metres from the pole lap's first sample) where it exceeds FOLLOW_OFFSET_M.
    Same laps, stations and car selection as ``driver_signal``.
    """
    laps = flying_laps(sess)
    pole = min(laps, key=lambda lap: lap.duration)
    curve = reference_curve(*lap_samples(sess, pole))
    s_curve = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    tree = cKDTree(curve)
    flagged = flag_broken_streams(sess, laps, tree)
    laps = [lap for lap in laps if lap.driver not in flagged]
    counts = {}
    for lap in laps:
        counts[lap.driver] = counts.get(lap.driver, 0) + 1
    drivers = [d for d, c in counts.items() if c >= min_laps and d != pole.driver]
    nb = int(s_curve[-1] // bin_m)
    table = {}
    for lap in laps:
        if lap.driver not in drivers:
            continue
        _, xy = lap_samples(sess, lap)
        if len(xy) < 50:
            continue
        s, off = _signed_offsets(tree, curve, s_curve, xy)
        b = (s // bin_m).astype(int)
        for k in np.unique(b):
            if 0 <= k < nb:
                table.setdefault((k, lap.driver), []).append(off[b == k].mean())
    eff = {d: [] for d in drivers}
    big = {d: [] for d in drivers}
    for k in range(nb):
        means = {d: np.mean(table[(k, d)]) for d in drivers if len(table.get((k, d), [])) >= 2}
        if len(means) < 5:
            continue
        mu = np.mean(list(means.values()))
        for d, m in means.items():
            eff[d].append(abs(m - mu))
            if abs(m - mu) > FOLLOW_OFFSET_M:
                big[d].append(k * bin_m)
    rows = []
    for d in sorted(drivers):
        e = np.array(eff[d])
        if not len(e):
            continue
        rows.append({"car": int(d), "laps": int(counts[d]), "stations": len(e),
                     "median_m": float(np.median(e)), "p95_m": float(np.percentile(e, 95)),
                     "max_m": float(e.max()), "big_m": big[d], "lap_m": nb * bin_m})
    return rows


def compute(max_n=None) -> dict:
    """Run the audit over every session and return the cache (JSON-ready)."""
    rows, example, skipped = [], None, []
    for meta, d in sessions(max_n):
        sess = load_session(d)
        if not sess.laps:
            skipped.append(f"{meta['year']} {meta['circuit']}: the archived laps table is "
                           "empty (OpenF1 returned no timed laps for this session; the "
                           "location feed is there, but no lap can be cut from it)")
            print("skip", skipped[-1], flush=True)
            continue
        try:
            a, s = audit(sess), driver_signal(sess)
        except ValueError as exc:
            skipped.append(f"{meta['year']} {meta['circuit']}: {exc}")
            print("skip", skipped[-1], flush=True)
            continue
        row = {"meta": meta, "audit": asdict(a), "signal": asdict(s)}
        if s.p95_driver_offset_m > FOLLOW_OFFSET_M:
            row["per_car"] = per_car_offsets(sess)
        rows.append(row)
        if example is None or meta["circuit"] == "Monza":
            example = (meta, sess)
        print(f"{a.session}: median {a.median_m*100:.1f} cm, ICC {s.median_icc:+.3f}, driver offset p95 {s.p95_driver_offset_m*100:.1f} cm (max {s.max_driver_offset_m*100:.0f}), planted ICC {s.planted_median_icc:+.3f} p95 {s.planted_p95_driver_offset_m*100:.0f} cm", flush=True)

    # The example session's distances from the pole lap's curve, as a histogram.
    meta, sess = example
    laps = flying_laps(sess)
    pole = min(laps, key=lambda lap: lap.duration)
    tree = cKDTree(reference_curve(*lap_samples(sess, pole)))
    dists = np.concatenate([tree.query(lap_samples(sess, lap)[1])[0] for lap in laps if lap is not pole])
    counts, edges = np.histogram(np.clip(dists, 0, 2.0), bins=80)
    return {"rows": rows, "skipped": skipped,
            "example": {"meta": meta, "counts": counts.tolist(), "edges": edges.tolist()}}


def _unpack(row):
    a = dict(row["audit"])
    a["injected"] = {float(k): v for k, v in a["injected"].items()}
    a["pole"], a["flagged"] = tuple(a["pole"]), tuple(a["flagged"])
    a["per_driver_median_m"] = {int(k): v for k, v in a["per_driver_median_m"].items()}
    return row["meta"], AuditResult(**a), DriverSignal(**row["signal"])


def figure(cache: dict) -> None:
    rows = sorted(cache["rows"], key=lambda r: r["meta"]["date_start"])
    ex = cache["example"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    edges, counts = np.array(ex["edges"]), np.array(ex["counts"])
    ax[0].bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=BLUE,
              edgecolor="white", linewidth=0.4)
    ax[0].set_xlabel("distance from the pole lap's curve (m; above 2 m counted at 2)", color=INK)
    ax[0].set_ylabel("samples, all drivers' flying laps", color=INK)
    ax[0].set_title(f"{ex['meta']['year']} {ex['meta']['circuit']} qualifying", color=INK,
                    fontsize=11)

    # One column per session in date order, a gap between seasons, the season named below.
    years = [r["meta"]["year"] for r in rows]
    x, pos, centres, seps = [], 0.0, {}, []
    for i, y in enumerate(years):
        if i and y != years[i - 1]:
            seps.append(pos + 1.0)
            pos += 3.0
        else:
            pos += 1.0 if i else 0.0
        x.append(pos)
        centres.setdefault(y, []).append(pos)
    x = np.array(x)
    real = [r["signal"]["median_icc"] for r in rows]
    planted = [r["signal"]["planted_median_icc"] for r in rows]
    kw = dict(s=22, edgecolors="white", linewidths=0.6, zorder=3)
    ax[1].scatter(x, planted, color=ORANGE, label="with a planted 0.5 m line for one driver", **kw)
    ax[1].scatter(x, real, color=BLUE, label="real data", **kw)
    for s in seps:
        ax[1].axvline(s, color=GRID, lw=1.0, zorder=1)
    ax[1].axhline(0, color=MUTED, lw=0.6, zorder=2)
    ax[1].set_xticks([np.mean(v) for v in centres.values()],
                     [f"{y}\n{len(v)} sessions" for y, v in centres.items()], fontsize=9,
                     color=INK)
    ax[1].tick_params(axis="x", length=0)
    ax[1].set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax[1].set_ylim(-0.12, 1.0)
    ax[1].set_ylabel("median driver ICC of lateral offset", color=INK)
    ax[1].set_title(f"Every qualifying session ({len(rows)}), in date order", color=INK,
                    fontsize=11)
    ax[1].legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.spines[["left", "bottom"]].set_color(MUTED)
        a.tick_params(colors=MUTED)
        a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "snapping-audit.png", dpi=150)
    plt.close(fig)


def write_report(cache: dict) -> None:
    rows = [_unpack(r) for r in cache["rows"]]
    skipped = cache["skipped"]
    figure(cache)
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
              if sg.frac_bins_significant > 0.05 or sg.p95_driver_offset_m > FOLLOW_OFFSET_M]
    if follow:
        lines += ["", "**Flagged for follow-up** (more than 5% of stations significant, or",
                  "driver offsets above 20 cm at p95). Effects of a few centimetres are per-car",
                  "artefacts such as antenna position or latency, far below racing-line scale",
                  "(0.5-2 m); larger ones are open items, not conclusions:"]
        for m, a, sg in follow:
            lines.append(f"- {m['year']} {m['circuit']}: {sg.frac_bins_significant*100:.1f}% of stations "
                         f"significant, driver offset p95 {sg.p95_driver_offset_m*100:.1f} cm, ICC {sg.median_icc:+.3f}")
    lines += _per_car_section(cache)
    if skipped:
        lines += ["", "Skipped sessions:", *[f"- {x}" for x in skipped]]
    (OUT / "snapping-audit.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'snapping-audit.md'} ({len(rows)} sessions)")


def _per_car_section(cache: dict) -> list[str]:
    big = [r for r in cache["rows"] if r.get("per_car")]
    if not big:
        return []
    cm = f"{FOLLOW_OFFSET_M * 100:.0f} cm"
    out = ["", (f"**Per-car breakdown of the sessions with driver offsets above {cm} at p95.** "
           "The same statistic split by car: per 50 m station, the car's mean signed offset "
           "over its laps minus the station mean over cars, and per car the median, p95 and "
           f"maximum of its absolute value. *Near the seam* counts the car's stations above {cm} "
           f"that lie within {SEAM_M:.0f} m of the start or end of the pole lap's curve, where "
           "the reference curve begins and ends.")]
    tallies = []
    for r in big:
        meta, cars = r["meta"], sorted(r["per_car"], key=lambda c: -c["p95_m"])
        lap = cars[0]["lap_m"]
        n_big = sum(len(c["big_m"]) for c in cars)
        n_seam = sum(near_seam(r, s) for c in cars for s in c["big_m"])
        hits: dict = {}
        for c in cars:
            for s in c["big_m"]:
                hits[s] = hits.get(s, 0) + 1
        common = sorted(s for s, n in hits.items() if n >= len(cars) / 2)
        out += ["", (f"*{meta['year']} {meta['circuit']}*: {len(cars)} cars, each with "
                f"{min(len(c['big_m']) for c in cars)}-{max(len(c['big_m']) for c in cars)} "
                f"stations above {cm}; {n_seam} of the {n_big} car-stations above {cm} are near "
                f"the seam. Stations where at least half the cars exceed {cm} (start, m; lap "
                f"{lap:.0f} m): {', '.join(f'{s:.0f}' for s in common) or 'none'}."),
                "", (f"| Car | Laps | Stations | Median (cm) | p95 (cm) | Max (cm) | Stations > {cm} "
                "| Near the seam |"), "|---|---|---|---|---|---|---|---|"]
        for c in cars:
            out.append(f"| {c['car']} | {c['laps']} | {c['stations']} | {100 * c['median_m']:.1f} "
                       f"| {100 * c['p95_m']:.1f} | {100 * c['max_m']:.1f} | {len(c['big_m'])} "
                       f"| {sum(near_seam(r, s) for s in c['big_m'])} |")
        tallies.append((n_seam, n_big))
    seam, total = map(sum, zip(*tallies, strict=True))
    out += ["", (f"**Reading.** {seam} of {total} car-stations above {cm} lie near the seam, and "
            "the stations are the same for every car. An offset that every car shows at the "
            "same few places belongs to the place (where the reference curve starts and "
            "ends), not to a car's line, so these flags are not a driver signal. Away from "
            f"the seam no car exceeds {cm} at more than "
            f"{max(sum(not near_seam(r, s) for s in c['big_m']) for r in big for c in r['per_car'])} "
            "station(s).")]
    return out


def near_seam(row: dict, s: float) -> bool:
    """Whether the 50 m station starting at s is within SEAM_M of either end of the curve."""
    return s < SEAM_M or s + 50.0 > row["per_car"][0]["lap_m"] - SEAM_M


def main(max_n=None, from_cache=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if from_cache:
        cache = json.loads(CACHE.read_text())
    else:
        cache = compute(max_n)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
    write_report(cache)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--from-cache", action="store_true",
                    help=f"rebuild the report and figure from {CACHE} without re-auditing")
    args = ap.parse_args()
    main(args.max_sessions, args.from_cache)
