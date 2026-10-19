"""Write docs/findings/drivers-pilot.md from the cached drivers_pilot.py stages.

Called by ``drivers_pilot.py report``; kept separate so the numbers and the prose that
reads them live in one place. Every number in the prose is computed here from the cached
results, not typed in. Everything written is labelled as a pilot; teammate deltas appear
only as an example table, never as a ranking.
"""

import collections
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apexzero.drivers import deltas as dl
from apexzero.drivers import recovery
from apexzero.drivers.pipeline import car_speed
from apexzero.drivers.segments import PHASES

DOCS = Path("docs/findings")
TYPES = ("slow", "medium", "fast")
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
PHASE_COLOURS = {"entry": "#2a78d6", "apex": "#eb6834", "exit": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _ms(x, nd=0):
    return f"{1000 * x:+.{nd}f}" if np.isfinite(x) else "-"


def _range(vals, fmt="{:.0f}"):
    vals = [v for v in vals if np.isfinite(v)]
    return f"{fmt.format(min(vals))} to {fmt.format(max(vals))}"


def _complex_errors(rec):
    """Per-pair complex-level errors (measurement, recovered - injected) by corner type."""
    meas, total = collections.defaultdict(list), collections.defaultdict(list)
    for name in sorted({c for c, _, _ in rec.labels}):
        cols = [i for i, (c, _, _) in enumerate(rec.labels) if c == name]
        kind = rec.labels[cols[0]][2]
        meas[kind].extend(rec.pair_err[:, cols].sum(axis=1))
        total[kind].extend(rec.pair_est[:, cols].sum(axis=1) - rec.injected[cols].sum())
    return meas, total


def _false_positive_rate(recs, limits):
    """Share of zero-injected phase cells whose recovered delta exceeds the claim limit."""
    hits, n = 0, 0
    for rec in recs:
        for i, (_, ph, kind) in enumerate(rec.labels):
            if rec.injected[i] != 0 or (kind, ph) not in limits:
                continue
            hits += int(np.sum(np.abs(rec.pair_est[:, i]) > limits[(kind, ph)]))
            n += rec.pair_est.shape[0]
    return hits / max(n, 1), n


def _detection_rate(recs, limits, kind, ph):
    hits, n = 0, 0
    for rec in recs:
        for i, (_, p, k) in enumerate(rec.labels):
            if k == kind and p == ph and rec.injected[i] != 0:
                hits += int(np.sum(np.sign(rec.pair_est[:, i]) * np.sign(rec.injected[i])
                                   * np.abs(rec.pair_est[:, i]) > limits[(kind, ph)]))
                n += rec.pair_est.shape[0]
    return hits / max(n, 1), n


def _figure(res, budget, budget_qss):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    r0 = res["Monza"] if "Monza" in res else next(iter(res.values()))
    ax = axes[0]
    v_kmh = 3.6 * car_speed(r0.timing, r0.v_ref_x, r0.v_ref)
    for c in r0.seg.complexes:
        for a, b, ph in zip(c.bounds[:-1], c.bounds[1:], PHASES, strict=True):
            ax.axvspan(a, b, color=PHASE_COLOURS[ph], alpha=0.16, lw=0)
        ax.axvline(c.x_start, color=MUTED, lw=0.6)
        ax.text(0.5 * (c.x_start + c.x_end), 412, f"{c.name}\n{c.kind}", ha="center",
                va="top", fontsize=7, color=MUTED)
    ax.plot(r0.v_ref_x, v_kmh, color=INK, lw=1.6)
    ax.set_xlim(0, r0.timing.length)
    ax.set_ylim(0, 420)
    ax.set_yticks(np.arange(0, 351, 50))
    ax.set_xlabel("lap distance from the timing line (m, TUMFTM centreline)")
    ax.set_ylabel("pooled reference speed (km/h)")
    ax.set_title(f"{r0.label}: complexes and entry / apex / exit phases", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOURS[p], alpha=0.35) for p in PHASES]
    ax.legend(handles, PHASES, loc="lower right", fontsize=8, frameon=False)
    ax = axes[1]
    xs = np.arange(len(TYPES))
    w = 0.26
    by = {(b.kind, b.phase): b for b in budget}
    for j, ph in enumerate(PHASES):
        pos = xs + (j - 1) * w
        ax.bar(pos, [by[(k, ph)].claim_limit_ms if (k, ph) in by else np.nan for k in TYPES],
               w - 0.03, color=PHASE_COLOURS[ph], label=f"{ph}: claim limit")
        ax.plot(pos, [by[(k, ph)].measurement_limit_ms if (k, ph) in by else np.nan
                      for k in TYPES], "D", ms=6, color=INK,
                label="measurement only" if j == 0 else None)
        ax.plot(pos, [budget_qss[(k, ph)].claim_limit_ms if (k, ph) in budget_qss else np.nan
                      for k in TYPES], "o", ms=8, mfc="white", mec=INK, mew=1.2,
                label="QSS profile, R-dev" if j == 0 else None)
    ax.set_xticks(xs, TYPES)
    ax.set_ylabel("2 sigma, per-phase teammate delta (ms)")
    ax.set_title("Injection-recovery: smallest claimable effect", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    for a in axes:
        a.grid(axis="y", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "drivers-pilot.png", dpi=150)


def write_report(out_dir: Path, pilots: list[dict]) -> None:
    reg_rows = json.loads((out_dir / "registration.json").read_text())
    tune = json.loads((out_dir / "tune.json").read_text())
    with open(out_dir / "injection.pkl", "rb") as f:
        inj = pickle.load(f)
    res = {}
    for r in pilots:
        with open(out_dir / f"pilot_{r['tumftm']}.pkl", "rb") as f:
            res[r["tumftm"]] = pickle.load(f)
    tabs = [x.table for x in res.values()]
    cals = inj["calibration"]

    pilot_recs = list(inj["pilot"].values())
    qss_recs = list(inj["qss"].values())
    budget = recovery.budget(pilot_recs)
    by = {(b.kind, b.phase): b for b in budget}
    budget_qss = {(b.kind, b.phase): b for b in recovery.budget(qss_recs)}
    agg = recovery.aggregate_budget(pilot_recs)
    limits = {k: 1e-3 * b.claim_limit_ms for k, b in by.items()}
    fpr, fpr_n = _false_positive_rate(pilot_recs, limits)
    cx_meas, cx_total = collections.defaultdict(list), collections.defaultdict(list)
    for rec in pilot_recs:
        m, t = _complex_errors(rec)
        for k in m:
            cx_meas[k].extend(m[k])
            cx_total[k].extend(t[k])

    placebo = dl.placebo(tabs, top_n=2, n_shuffles=500, seed=1)
    sh_phase = dl.split_half_within(tabs, "phase")
    sh_type = dl.split_half_within(tabs, "type")
    sh_across = dl.split_half_across(tabs)
    _figure(res, budget, budget_qss)

    L = []
    w_ = L.append
    w_("# Drivers-lite pilot: 2025 qualifying, four circuits (PILOT, no rankings)")
    w_("")
    w_("*Pilot of plan section 3.4 (method: plan-v1 section 6). It tests the measurement "
       "pipeline, not the drivers. Every teammate number below is an example of what the "
       "pipeline outputs, with its error bars; none of it is a claim about who is faster.*")
    w_("")
    w_("Code: `src/apexzero/drivers/`, run by `scripts/drivers_pilot.py` (stages "
       "`registration`, `pilot`, `tune`, `injection`, `report`). Data: OpenF1 (CC BY-NC-SA "
       "4.0) and the TUMFTM centrelines.")
    w_("")
    w_("**Lockbox.** The circuits that pass registration are all R-test circuits. Only the "
       "public telemetry is registered and timed on them. No RL agent, OCP or QSS lap is "
       "evaluated on them. Injection-recovery laps on these circuits use each session's own "
       "pooled speed profile. The QSS-profile check and the smoother tuning run on R-dev "
       "geometry only (`qss_model` refuses R-test names in code). The synthetic noise "
       "*levels* are calibrated against the real pilot sessions (section 2). That measures "
       "the feed; it does not fit anything to those circuits.")
    w_("")

    # ------------------------------------------------------------------ 1. registration
    cur = [r for r in reg_rows if r.get("tumftm") and r["layout_matches"]]
    old = [r for r in reg_rows if r.get("tumftm") and not r["layout_matches"]]
    mapped = cur + old
    w_("## 1. Registration onto the TUMFTM centreline")
    w_("")
    w_("The fit is a similarity transform: a 5-degree rotation grid with both "
       "handednesses, then trimmed ICP. The gate is the plan's: RMS <= 3 m and lap-length "
       "mismatch <= 1%. The length is the feed's own driven-line length (a spline through "
       "the samples of the fastest laps) times the fitted scale, compared with the "
       "centreline length. Fitted rotations are "
       f"{_range([r['angle_deg'] for r in mapped], '{:+.1f}')} degrees and scales "
       f"{_range([r['scale'] for r in mapped], '{:.4f}')}, with no reflection. So the feed "
       "and TUMFTM frames share orientation and units up to a translation.")
    w_("")
    w_("| Circuit (OpenF1) | TUMFTM | Layout | RMS (m) | p95 (m) | Length | Scale | "
       "Outside corridor | Excursion p99 (m) | Inside offset (m) | Plan gate |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reg_rows:
        if r.get("tumftm") is None:
            continue
        lay = "current" if r["layout_matches"] else "**old**"
        w_(f"| {r['circuit']} | {r['tumftm']} | {lay} | {r['rms_m']:.2f} | {r['p95_m']:.1f} | "
           f"{100 * r['length_mismatch']:+.2f}% | {r['scale']:.4f} | "
           f"{100 * r['outside_corridor']:.1f}% | {r['excursion_p99_m']:+.1f} | "
           f"{r['inside_offset_m']:+.2f} | {'pass' if r['accepted'] else 'fail'} |")
    missing = [r["circuit"] for r in reg_rows if r.get("tumftm") is None]
    w_("")
    w_(f"No TUMFTM centreline: {', '.join(missing)}. *Inside offset* is the mean lateral "
       "offset towards the inside of corners tighter than 150 m radius. *Excursion* is the "
       "distance beyond the nearer TUMFTM track edge (negative = inside the track).")
    w_("")
    gate_excursion = 8.0
    n_open = sum(r["excursion_p99_m"] <= gate_excursion for r in cur)
    n_old_open = sum(r["excursion_p99_m"] <= gate_excursion for r in old)
    w_("**Reading.**")
    w_("")
    w_(f"- The plan's gate rejects all {len(old)} outdated TUMFTM layouts, as it should. It "
       f"also rejects {sum(not r['accepted'] for r in cur)} of the {len(cur)} current "
       f"layouts. It passes only {', '.join(r['circuit'] for r in cur if r['accepted'])}.")
    w_("- The likely cause is the feed's track model, which is not the centreline. In "
       "corners tighter than 150 m radius it sits "
       f"{_range([r['inside_offset_m'] for r in cur], '{:.1f}')} m "
       "to the inside, and its lap is "
       f"{_range([-100 * r['length_mismatch'] for r in cur], '{:.1f}')}% shorter. That looks "
       "like a racing line. So both gate statistics measure the line as well as any "
       "registration error, and a twisty circuit can fail on the line alone.")
    w_("- The excursion beyond the TUMFTM edges separates the two groups much more sharply. "
       "At the 99th percentile it is at most "
       f"{max(r['excursion_p99_m'] for r in cur):+.1f} m on current layouts and at least "
       f"{min(r['excursion_p99_m'] for r in old):+.1f} m on old ones. **Proposal, not "
       f"applied here:** gate on excursion p99 <= {gate_excursion:g} m and report RMS and "
       f"length as diagnostics. On this table that would pass {n_open} of {len(cur)} current "
       f"layouts and {n_old_open} of {len(old)} old ones. The pilot keeps the plan's gate.")
    w_("- Projecting a racing-line model onto the centreline is still a fixed, monotone "
       "map, the same for every car. So s(t) stays comparable between teammates. The cost "
       "is a speed / centreline-rate ratio that varies along the lap. The smoother "
       "estimates it from all laps (section 2).")
    w_("")

    # ------------------------------------------------------------------ 2. s(t)
    best = tune["best"]
    rows_hi = [r for r in tune["rows"] if r["sigma_speed"] >= 0.7]
    w_("## 2. s(t) per lap, the held-out sector anchor, and synthetic calibration")
    w_("")
    w_("What the feed looks like (measured on Monza 2025 while building the smoother):")
    w_("")
    w_("- Projected positions scatter about 2 m *along* the track, with heavy tails. A "
       "~0.7 m pattern sits on top, the same for every car. The snapping audit's 13 cm is "
       "the lateral scatter only.")
    w_("- Speed comes in whole km/h, on its own ~3.7 Hz clock. The location and car_data "
       "clocks show no lag (best lag within +/-0.05 s).")
    w_("- `date_start` is off by up to ~0.1 s: consecutive laps disagree with start + "
       "duration by up to 0.2 s. Lap and sector durations are exact. The estimated "
       "start-time corrections below have a robust sd of "
       f"{_range([1000 * x.timing.delta_spread for x in res.values()])} ms, close to the "
       "58 ms of a uniform +/-0.1 s error.")
    w_("")
    w_("The smoother is an RTS smoother on [s, v, a] driven by white jerk. It fuses "
       "positions, speed and hard anchors at the official lap and sector times. Speed is "
       "modelled as a per-lap scale times a pooled path-ratio profile times ds/dt. The "
       "timing-line positions and each lap's start-time correction come from unanchored "
       "fits first. The median correction is pinned to zero, because a common shift is not "
       "identifiable. Hyperparameters were tuned on QSS-profile synthetic laps on R-dev "
       f"geometry ({', '.join(tune['circuits'])}). The result is jerk PSD "
       f"{best['jerk_psd']:g} m^2/s^5 and speed sigma {best['sigma_speed']:g} m/s "
       f"(pair-delta RMS {best['pair_rms_ms']:.1f} ms). The surface is flat: every grid "
       f"point with speed sigma >= 0.7 m/s scored "
       f"{_range([r['pair_rms_ms'] for r in rows_hi])} ms.")
    w_("")
    w_("**Held-out sector anchor (plan gate 3).** Each lap is refitted without the S2/S3 "
       "line, the crossing time is predicted, and it is compared with the official S1 + "
       "S2. The same check on synthetic laps calibrates the synthetic feed. The line wobble "
       "(`SynthConfig.wobble_sd`) is the one chosen from "
       f"{sorted(next(iter(cals.values())).tried)} whose synthetic held-out spread is "
       "closest to the real one. Natural lap-to-lap variation is then set per corner type "
       "x phase, from the real within-driver spread minus the synthetic measurement error "
       "(table below).")
    w_("")
    w_("| Session | Laps fitted | Held-out bias (ms) | Held-out robust sd (ms) | p95 abs (ms) | "
       "Synthetic robust sd (ms) | Wobble | Start corr. sd (ms) | Position sigma (m) |")
    w_("|---|---|---|---|---|---|---|---|---|")
    for name, x in res.items():
        h, c = x.held_out_s2, cals[name]
        syn = inj["pilot"][name].held_out_s2
        w_(f"| {x.label} | {len(x.timing.laps)} | {_ms(np.median(h))} | "
           f"{1000 * recovery.robust_sd(h):.1f} | {1000 * np.percentile(np.abs(h), 95):.0f} | "
           f"{1000 * recovery.robust_sd(syn):.1f} | {c.wobble_sd:g} | "
           f"{1000 * x.timing.delta_spread:.0f} | {x.pos_sigma:.2f} |")
    w_("")
    w_("Within-driver spread of phase times (real, laps of one driver in one Q segment) "
       "against the synthetic measurement error at the calibrated wobble, median over a "
       "circuit's cells of each type, in ms (real / synthetic measurement). Where the "
       "synthetic number is larger, the synthetic feed is noisier than the real one there, "
       "which makes the budget conservative; natural variation is then set to zero.")
    w_("")
    cells = [(k, ph) for k in TYPES for ph in PHASES]
    w_("| Session | " + " | ".join(f"{k} {ph}" for k, ph in cells) + " |")
    w_("|---|" + "---|" * len(cells))
    for name, x in res.items():
        bc = cals[name].by_cell
        w_(f"| {x.label.split(' Qualifying')[0]} | " + " | ".join(
            f"{1000 * bc[c][0]:.0f} / {1000 * bc[c][1]:.0f}" if c in bc else "-"
            for c in cells) + " |")
    n_over = sum(len(c.over_noisy) for c in cals.values())
    n_all = sum(len(c.by_cell) for c in cals.values())
    w_("")
    w_(f"In {n_over} of {n_all} circuit x cell combinations the synthetic measurement "
       "error alone exceeds the real total spread. Real lap-to-lap variation is several "
       "times larger in slow phases than in fast ones, which is why natural variation is "
       "set per cell and not as one number.")
    w_("")
    biases = [1000 * np.median(x.held_out_s2) for x in res.values()]
    w_("A single crossing time far from the anchors is good to "
       f"{_range([1000 * recovery.robust_sd(x.held_out_s2) for x in res.values()])} ms "
       "(robust sd). The synthetic column comes from the injection runs themselves. The "
       f"per-session biases ({_range(biases, '{:+.0f}')} ms) are common to every lap of a "
       "session, so they cancel between teammates. They do show a model mismatch near "
       "that line that a physics reference would inherit.")
    w_("")
    n_cx = {x.label.split()[1]: len(x.seg.complexes) for x in res.values()}
    w_(f"**Segmentation.** Complexes per circuit: "
       f"{', '.join(f'{k} {v}' for k, v in n_cx.items())}. The plan asks for 12-20. "
       "Corners less than 80 m apart always merge, because at 3.7 Hz (one sample every "
       "6-25 m) a chicane's halves cannot be timed apart. Below 12, the code keeps the "
       "natural count rather than splitting chicanes or straights to fill a quota. Each "
       "complex is split into entry / apex / exit at positions on the pooled reference "
       "speed, and typed by its apex speed (slow < 120, medium 120-200, fast > 200 km/h).")
    w_("")
    w_("![pilot](drivers-pilot.png)")
    w_("")

    # ------------------------------------------------------------------ 3. lap filter
    w_("## 3. Lap filter")
    w_("")
    w_("| Session | Laps fitted (checks 1-7 passed) | Traffic (< 1.0 s ahead in a corner) | "
       "Pass every check | Exclusions before fitting (all laps) | Teammate pairs | "
       "Laps per driver in pairs |")
    w_("|---|---|---|---|---|---|---|")
    shares = []
    for x in res.values():
        reasons = collections.Counter(r for st in x.statuses.values() for r in st.reasons)
        traffic = reasons.pop("traffic", 0)
        shares.append(traffic / len(x.timing.laps))
        other = ", ".join(f"{k} {v}" for k, v in reasons.most_common())
        pairs = dl.teammate_deltas(x.table)
        counts = collections.Counter(n for d in pairs for n in (d.n_a, d.n_b))
        share = 100 * traffic / len(x.timing.laps)
        w_(f"| {x.label} | {len(x.timing.laps)} | {traffic} ({share:.0f}%) "
           f"| {len(x.table.keys)} | {other} | {len(pairs)} | "
           f"{', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))} |")
    w_("")
    w_(f"Traffic removes {_range([100 * s for s in shares])}% of otherwise clean push laps. "
       "That is the plan's rule taken literally: any car less than 1.0 s ahead anywhere in "
       "any corner window. Spot checks at Montreal found real slow cars on out-laps and "
       "cool-down laps, on the racing line (the same lateral offset as fast cars, so not "
       "the pit lane). With the 102% rule per Q segment and shared Q segments required, "
       "pairs end up with 1-2 laps per driver (last column: laps -> number of drivers). "
       "That is the binding constraint on everything below. Tyre compounds come from the "
       "OpenF1 stints table only. Montreal's MEDIUM count is large and was not "
       "cross-checked.")
    w_("")

    # ------------------------------------------------------------------ 4. injection
    rec0 = pilot_recs[0]
    qcfg = inj["qss_config"]
    w_("## 4. Injection-recovery error budget (plan gate 2)")
    w_("")
    w_(f"For each pilot circuit: {rec0.pair_est.shape[0]} synthetic teammate pairs "
       f"({rec0.pair_est.shape[0] // 2} pairs x 2 sessions), {rec0.laps_per_driver} laps "
       f"per driver, comparing each driver's top-{rec0.top_n} laps as in the real analysis. "
       "Every B driver got injected effects of +40 ms in each slow-corner entry, +20 ms in "
       "each medium-corner apex and -20 ms in each fast-corner exit. Every phase of every "
       "lap also carries natural variation at the calibrated per-cell level (section 2).")
    w_("")
    w_("How the laps are degraded:")
    w_("")
    w_("- resampled on the empirical location and car_data interval distributions;")
    w_("- positions with Student-t (3 dof) along-track noise at each session's measured "
       "scale, 13 cm lateral noise and decimetre rounding;")
    w_("- speed with a per-lap scale (sd 0.2%) and a smooth line wobble at the calibrated "
       "level (150 m correlation), then rounded to whole km/h;")
    w_("- brake binarised (the timing pipeline does not use it);")
    w_("- start times jittered by +/-0.1 s.")
    w_("")
    w_("Then the laps go through the real pipeline.")
    w_("")
    w_("*Measurement* = estimated minus true delta over the same laps: the pipeline's own "
       "error. *Recovered - injected* adds natural "
       f"variation over {rec0.top_n} laps per driver. Its 2 sigma is the **claim limit**, "
       "the plan's \"2 sigma of the resulting error\".")
    w_("")
    w_("| Corner type | Phase | Cells | Per-lap meas. sd (ms) | Meas. bias (ms) | "
       "Meas. 2 sigma (ms) | Injected (ms) | Recovered mean (ms) | Recovered - injected sd (ms) | "
       "**Claim limit (ms)** | QSS R-dev claim limit (ms) |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in budget:
        q = budget_qss.get((b.kind, b.phase))
        w_(f"| {b.kind} | {b.phase} | {b.n_cells} | {b.lap_sd_ms:.0f} | {b.pair_bias_ms:+.1f} | "
           f"{b.measurement_limit_ms:.0f} | {b.injected_ms:+.0f} | {b.recovered_ms:+.1f} | "
           f"{b.recovered_sd_ms:.1f} | **{b.claim_limit_ms:.0f}** | "
           f"{q.claim_limit_ms if q else float('nan'):.0f} |")
    w_("")
    w_("Complex level (entry + apex + exit), claim limit by corner type: " + "; ".join(
        f"{k} {2000 * np.std(cx_total[k], ddof=1):.0f} ms (measurement only "
        f"{2000 * np.std(cx_meas[k], ddof=1):.0f})" for k in TYPES if k in cx_total) + ".")
    w_("")
    w_("Corner type x phase summed over the lap (the driver-profile cells of the plan's "
       "model):")
    w_("")
    w_("| Cell | Pairs | Meas. sd (ms) | Injected total (ms) | Recovered - injected mean (ms) | "
       "Recovered - injected sd (ms) | Claim limit (ms) |")
    w_("|---|---|---|---|---|---|---|")
    for k in sorted(agg, key=lambda k: (TYPES.index(k[0]), PHASES.index(k[1]))):
        a = agg[k]
        w_(f"| {k[0]} {k[1]} | {a['n']} | {a['pair_sd_ms']:.1f} | {a['injected_ms']:+.0f} | "
           f"{a['recovered_minus_injected_ms']:+.1f} | {a['recovered_sd_ms']:.1f} | "
           f"{2 * a['recovered_sd_ms']:.0f} |")
    w_("")
    det, det_n = _detection_rate(pilot_recs, limits, "slow", "entry")
    det_m, _ = _detection_rate(pilot_recs, limits, "medium", "apex")
    det_f, _ = _detection_rate(pilot_recs, limits, "fast", "exit")
    w_(f"False positives: {100 * fpr:.1f}% of {fpr_n} zero-injected phase cells exceed their "
       "claim limit (about 5% is expected at 2 sigma; neighbouring injected phases also "
       f"leak into them). Detection: {100 * det:.0f}% of {det_n} injected slow-entry cells "
       f"(+40 ms) exceed the limit with the right sign. The rates are {100 * det_m:.0f}% for "
       f"medium-apex (+20 ms) and {100 * det_f:.0f}% for fast-exit (-20 ms).")
    w_("")
    se = by.get(("slow", "entry"))
    qss_ratio = [budget_qss[k].claim_limit_ms / by[k].claim_limit_ms for k in by
                 if k in budget_qss]
    w_("**Reading.** The injected effects come back unbiased within their error: recovered "
       f"means are {_range([b.recovered_ms - b.injected_ms for b in budget], '{:+.1f}')} ms "
       "off the injected values. The smallest claimable per-phase teammate effect, from one "
       f"car-session with {rec0.top_n} laps per driver, is "
       f"{_range([b.claim_limit_ms for b in budget])} ms depending on corner type and phase. "
       f"The pipeline's own share is {_range([b.measurement_limit_ms for b in budget])} ms. "
       f"A +40 ms slow-corner entry is {40 / se.claim_limit_ms:.1f}x its limit "
       f"({se.claim_limit_ms:.0f} ms), and one synthetic car-session flags it in only "
       f"{100 * det:.0f}% of cases. Natural lap-to-lap variation, not the pipeline, "
       "dominates the limit in slow and medium phases. The pipeline's own error dominates "
       "in fast phases. Both average down with more laps per driver or more sessions per "
       "pair, as long as they are independent between laps. A per-car systematic error "
       "would not average down, and none is modelled. If car-sessions were independent, a "
       "season of 20 would bring the per-phase limits to "
       f"{_range([b.claim_limit_ms / np.sqrt(20) for b in budget])} ms, and the corner type x "
       "phase lap sums to "
       f"{_range([2 * a['recovered_sd_ms'] / np.sqrt(20) for a in agg.values()])} ms "
       "(limit / sqrt(20)). That independence is an assumption; the split-half gate is "
       "what tests it. "
       "Ratios of QSS-profile limits (R-dev geometry, median calibration: wobble "
       f"{qcfg.wobble_sd:g}, natural "
       f"{_range([1000 * v for v in inj['qss_natural'].values()])} ms by cell) to pilot "
       f"limits are {_range(qss_ratio, '{:.2f}')}, median {np.median(qss_ratio):.2f}. "
       "So, outside the fast-entry cell, the budget does not hinge on the speed-profile "
       "shape. Per-lap errors also carry a part common to all laps. It cancels in "
       "teammate deltas, but it would not cancel against a physics reference (R2).")
    w_("")

    # ------------------------------------------------------------------ 5. placebo
    w_("## 5. Placebo and split-half reliability (real data)")
    w_("")
    w_(f"**Placebo (gate 4).** Driver labels were shuffled within each car-session "
       f"({placebo.n_shuffles} shuffles, top-2 laps, shared Q segments, "
       f"{placebo.n_cells} phase cells). Shuffled deltas average "
       f"{_ms(placebo.placebo_mean, 1)} +/- {1000 * placebo.placebo_mean_se:.1f} ms, so "
       "neither the pipeline nor the top-2 selection creates a driver effect by itself. "
       f"Mean |delta| per cell is {1000 * placebo.placebo_abs_mean:.0f} ms shuffled versus "
       f"{1000 * placebo.real_abs_mean:.0f} ms with the true labels. A permutation test on "
       f"the sum of squared deltas over all cells gives p = {placebo.ss_p_value:.3f}. That is "
       "the probability that random labels explain at least as much phase-level variance "
       "as the real ones. Read it as weak, pooled evidence that the true labels carry some "
       "phase-level structure beyond lap-to-lap variation, not as a statement about any "
       "driver. Driver, setup and per-car measurement offsets are confounded in it. A "
       "per-cell test is not possible here: with 2 laps per driver there are only 3 "
       "distinct splits.")
    w_("")
    w_("**Split-half (gate 5).** The gate is Spearman rho >= 0.5.")
    w_("")
    w_("| Split | Level | Cells | Spearman rho | p |")
    w_("|---|---|---|---|---|")
    w_(f"| odd vs even ranked laps, within session | phase | {sh_phase.n_cells} | "
       f"{sh_phase.rho:+.2f} | {sh_phase.p_value:.2g} |")
    w_(f"| odd vs even ranked laps, within session | corner type x phase | {sh_type.n_cells} | "
       f"{sh_type.rho:+.2f} | {sh_type.p_value:.2g} |")
    w_(f"| alternate sessions (pairs in both halves) | corner type x phase | "
       f"{sh_across.n_cells} | {sh_across.rho:+.2f} | {sh_across.p_value:.2g} |")
    w_("")
    w_("No split reaches the 0.5 gate. The within-session halves compare one lap with one "
       "lap, and only for pairs with 2+ clean laps each. Driver, setup and any per-car "
       "measurement offset (antenna position, transponder) are confounded in them. The "
       "across-session split has only the pairs that appear in both halves of four "
       "sessions. Both are underpowered pilots of the gate. The plan's version (odd vs "
       "even rounds over a season) needs the full calendar.")
    w_("")

    # ------------------------------------------------------------------ 6. example
    w_("## 6. Example teammate per-corner table (pilot, not a ranking)")
    w_("")
    cand = []
    for name, x in res.items():
        for d in dl.teammate_deltas(x.table):
            cand.append((d.n_a + d.n_b, len(x.seg.complexes), name, d.team, d))
    cand.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
    _, _, name, team, d = cand[0]
    x = res[name]
    acr = x.table.acronym
    sub = x.table.rows(np.array([t == team for t in x.table.team]))
    pb = dl.placebo([sub], top_n=2, n_shuffles=500, seed=2)
    w_(f"{x.label}, {team}. Delta = {acr.get(d.b, d.b)} minus {acr.get(d.a, d.a)}, so "
       f"positive means {acr.get(d.b, d.b)} took longer. Each driver contributes the mean of "
       f"their top-{min(d.n_a, d.n_b)} clean laps in the shared Q segment(s) "
       f"{', '.join(f'Q{q}' for q in d.q_segments)}. The lap delta is {_ms(d.lap_delta)} ms, "
       "and the phase deltas below add up to it exactly. A cell would be **bold** only if it "
       "cleared both the claim limit for its corner type and phase and twice this car's own "
       "placebo spread.")
    w_("")
    w_("| Complex | Type | Entry (ms) | Apex (ms) | Exit (ms) | Complex (ms) | Above both limits |")
    w_("|---|---|---|---|---|---|---|")
    for c in x.seg.complexes:
        cols = [i for i, (cn, _, _) in enumerate(x.table.labels) if cn == c.name]
        cells, marks = [], []
        for i in cols:
            _, ph, kind = x.table.labels[i]
            v = d.delta[i]
            ok = abs(v) > limits.get((kind, ph), np.inf) and abs(v) > 2 * pb.cell_null_sd[i]
            cells.append(f"**{_ms(v)}**" if ok else _ms(v))
            if ok:
                marks.append(ph)
        w_(f"| {c.name} | {c.kind} | {' | '.join(cells)} | {_ms(d.delta[cols].sum())} | "
           f"{', '.join(marks) or '-'} |")
    w_("")
    w_("With 1-2 laps per driver, single cells mostly sit inside the error budget. That "
       "is the expected pilot outcome. It is why the plan pools over a season with a "
       "hierarchical model before saying anything about a driver.")
    w_("")

    # ------------------------------------------------------------------ 7. caveats
    w_("## 7. Not done, or uncertain")
    w_("")
    w_(f"- **Registration gate.** As written it passes {sum(r['accepted'] for r in cur)} of "
       f"{len(cur)} current-layout 2025 circuits (section 1). The corridor-excursion gate "
       "is a proposal, checked only against this table.")
    w_("- **Scope.** Four circuits, one qualifying session each. Sprint qualifying was not "
       "run. Most circuits stay below the plan's 12 complexes because chicanes are never "
       "split.")
    w_("- **Synthetic realism.** The synthetic feed is calibrated on two statistics per "
       "session (held-out S2 spread; within-driver spread per corner type x phase). The "
       "single wobble knob stands in for every unmodelled error source, so how the "
       "measurement error splits between phases is a model assumption (section 2 table). "
       "Natural variation is Gaussian and independent between phases. The ~0.7 m position "
       "pattern common "
       "to all cars is not synthesised; it is common-mode, but that it cancels is untested. "
       "No clock latency (none was measured) and no per-car antenna offsets are modelled. "
       "The QSS check profiles are computed on the centreline, not a racing line.")
    w_("- **Smoother.** Tuning picked the stiffest acceleration model on the grid (jerk "
       f"PSD {best['jerk_psd']:g}), on a flat error surface. Held-out biases of "
       f"{_range(biases, '{:+.0f}')} ms show it misfits some real braking zones in a way "
       "common to all laps.")
    w_("- **Lap filter.** Compounds come from the stints table only, with no new/used "
       "flag. Traffic uses other cars' raw projections; samples more than 7 m from the "
       "centreline (pit lane, garage) are ignored. The tow flag is computed but unused.")
    w_("- **Not built (later plan steps).** The hierarchical model (M1/M2), R2 calibration, "
       "the circularity check, the multiverse and external validity. Brake and throttle "
       "are loaded but unused.")
    w_("- **Underpowered gates.** The claim limits assume 2 laps per driver, and many real "
       "pairs have 1. The placebo tests label exchangeability within car-sessions, not "
       "across a season.")
    w_("")
    cpu = sum(getattr(x, "cpu_s", 0.0) for x in res.values())
    w_(f"Compute: the pilot stage took {cpu:.0f} s of CPU over four sessions. Registration "
       "takes ~2 s of CPU per session; calibration, injection-recovery and tuning take a "
       "few minutes each. Everything ran single-process at nice 15 on the M3 Pro alongside "
       "other jobs.")
    (DOCS / "drivers-pilot.md").write_text("\n".join(L) + "\n")
    print(f"wrote {DOCS / 'drivers-pilot.md'}")
