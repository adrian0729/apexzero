"""Write docs/findings/drivers-pilot.md from the cached drivers_pilot.py stages.

Called by ``drivers_pilot.py report``; kept separate so the numbers and the prose that
reads them live in one place. Every number in the prose is computed here from the cached
results, not typed in, except the first version's numbers quoted in the change log
(``V1``, from outputs/drivers-lite/archive-v1/). Everything written is labelled as a
pilot; teammate deltas appear only as an example table, never as a ranking.
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
from apexzero.drivers.registration import EXCURSION_MAX_M
from apexzero.drivers.segments import PHASES, corner_type

DOCS = Path("docs/findings")
TYPES = ("slow", "medium", "fast")
SEASON = 20                 # car-sessions per pair in a season, for the projection
GATE_EXCURSION_M = EXCURSION_MAX_M
# Categorical slots 1-3 of the reference palette (validated all-pairs, light surface).
PHASE_COLOURS = {"entry": "#2a78d6", "apex": "#eb6834", "exit": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# The first version's headline numbers (docs/findings/drivers-pilot.md before review,
# archived with its outputs in outputs/drivers-lite/archive-v1/), quoted in section 9.
V1 = {"placebo_p": 0.027, "rho_phase": 0.29, "rho_type": 0.39, "rho_across": 0.11,
      "mismatched_pairs": 4, "pairs": 25, "fpr": 5.8, "jerk": 100, "sigma_speed": 1.2}


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
    ax.plot(r0.v_ref_x, v_kmh, color=INK, lw=1.6, label="pooled median lap")
    if r0.v_env is not None:
        ax.plot(r0.v_ref_x, 3.6 * r0.v_env, color=INK, lw=0.9, ls="--",
                label="envelope (95th pct., types)")
    ax.set_xlim(0, r0.timing.length)
    ax.set_ylim(0, 420)
    ax.set_yticks(np.arange(0, 351, 50))
    ax.set_xlabel("lap distance from the timing line (m, TUMFTM centreline)")
    ax.set_ylabel("car speed (km/h)")
    ax.set_title(f"{r0.label}: complexes and entry / apex / exit phases", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLOURS[p], alpha=0.35) for p in PHASES]
    lines = ax.get_legend_handles_labels()
    ax.legend(handles + lines[0], list(PHASES) + lines[1], loc="lower right", fontsize=7.5,
              frameon=False, ncol=2)
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


def _grid_edges(rows, best):
    """Which grid edges the tuned point sits on, as words."""
    out = []
    for key, name in (("jerk_psd", "jerk PSD"), ("sigma_speed", "speed sigma")):
        vals = sorted({r[key] for r in rows})
        if best[key] == vals[0]:
            out.append(f"the lowest {name} tried ({vals[0]:g})")
        elif best[key] == vals[-1]:
            out.append(f"the highest {name} tried ({vals[-1]:g})")
    return out


def _widening(pb, limits, target=0.05):
    """Factor by which the synthetic limits must be widened so that ``target`` of the
    shuffled real cells exceed them (bisection; 1 = the budget already fits)."""
    lo, hi = 0.25, 8.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        share = pb.exceed_share({k: mid * v for k, v in limits.items()})[0]
        lo, hi = (mid, hi) if share > target else (lo, mid)
    return 0.5 * (lo + hi)


def _r1_summary(x):
    """Per session: car-sessions, median (best clean lap - R1 lap) and median per-lap
    y = t - R1 summed over each corner type's phases (s)."""
    tab = x.table
    y = dl.r1_residuals(tab)
    gaps = []
    for team, r1 in (x.r1 or dl.r1_envelope(tab)).items():
        m = tab.team_mask(team)
        if m.sum() >= 2:
            gaps.append(float(tab.lap_time[m].min() - r1.sum()))
    by_type = {}
    for kind in TYPES:
        cols = [i for i, (_, _, k) in enumerate(tab.labels) if k == kind]
        by_type[kind] = float(np.median(y[:, cols].sum(axis=1))) if cols else float("nan")
    return len(set(tab.team)), gaps, by_type


def write_report(out_dir: Path, pilots: list[dict]) -> None:
    reg_rows = json.loads((out_dir / "registration.json").read_text())
    reg_other = {int(p.stem.split("_")[1]): json.loads(p.read_text())
                 for p in sorted(out_dir.glob("registration_*.json"))}
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
    with open(out_dir / "persistent.pkl", "rb") as f:
        pers = pickle.load(f)
    budget = recovery.budget(pilot_recs)
    by = {(b.kind, b.phase): b for b in budget}
    budget_qss = {(b.kind, b.phase): b for b in recovery.budget(qss_recs)}
    agg = recovery.aggregate_budget(pilot_recs)
    limits = {k: 1e-3 * b.claim_limit_ms for k, b in by.items()}
    rates = recovery.held_out_rates(pilot_recs)
    sys_by = {v: recovery.systematic_error(pilot_recs, [runs[k] for k in inj["pilot"]])
              for v, runs in pers["runs"].items()}
    sys_err = sys_by["both"]
    cx_meas, cx_total = collections.defaultdict(list), collections.defaultdict(list)
    for rec in pilot_recs:
        m, t = _complex_errors(rec)
        for k in m:
            cx_meas[k].extend(m[k])
            cx_total[k].extend(t[k])

    placebo = dl.placebo(tabs, top_n=2, n_shuffles=500, seed=1)
    shuffled_share, real_share, share_cells = placebo.exceed_share(limits)
    widen = _widening(placebo, limits)
    ratios = placebo.abs_ratio_by_cell()
    sh_phase = dl.split_half_within(tabs, "phase")
    sh_type = dl.split_half_within(tabs, "type")
    sh_across = dl.split_half_across(tabs)
    pairs_all = [d for x in res.values() for d in dl.teammate_deltas(x.table)]
    multi_q = [d for d in pairs_all if len(d.q_segments) > 1]
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
       "4.0) and the TUMFTM centrelines. This is the second version, after code review; "
       "section 9 lists what changed.")
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
       "handednesses, then trimmed ICP. The gate is the corridor excursion: the 99th "
       "percentile of the registered points' distance beyond the TUMFTM track edges must be "
       f"<= {GATE_EXCURSION_M:g} m. RMS and lap-length mismatch are reported as diagnostics. "
       "They were the plan's gate (RMS <= 3 m, length mismatch <= 1%), which chose the four "
       "pilot circuits before the excursion gate was adopted (see the reading below). The "
       "length is the feed's own driven-line length (a spline through "
       "the samples of the fastest laps) times the fitted scale, compared with the "
       "centreline length. Fitted rotations are "
       f"{_range([r['angle_deg'] for r in mapped], '{:+.1f}')} degrees and scales "
       f"{_range([r['scale'] for r in mapped], '{:.4f}')}, with no reflection. So the feed "
       "and TUMFTM frames share orientation and units up to a translation.")
    w_("")
    w_("| Circuit (OpenF1) | TUMFTM | Layout | RMS (m) | p95 (m) | Length | Scale | "
       "Outside corridor | Excursion p99 (m) | Inside offset (m) | Plan gate (diagnostic) | "
       "Gate |")
    w_("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reg_rows:
        if r.get("tumftm") is None:
            continue
        lay = "current" if r["layout_matches"] else "**old**"
        w_(f"| {r['circuit']} | {r['tumftm']} | {lay} | {r['rms_m']:.2f} | {r['p95_m']:.1f} | "
           f"{100 * r['length_mismatch']:+.2f}% | {r['scale']:.4f} | "
           f"{100 * r['outside_corridor']:.1f}% | {r['excursion_p99_m']:+.1f} | "
           f"{r['inside_offset_m']:+.2f} | {'pass' if r['plan_gate'] else 'fail'} | "
           f"{'pass' if r['accepted'] else 'fail'} |")
    missing = [r["circuit"] for r in reg_rows if r.get("tumftm") is None]
    w_("")
    w_(f"No TUMFTM centreline: {', '.join(missing)}. *Inside offset* is the mean lateral "
       "offset towards the inside of corners tighter than 150 m radius. *Excursion* is the "
       "distance beyond the nearer TUMFTM track edge (negative = inside the track).")
    w_("")
    n_open = sum(r["accepted"] for r in cur)
    n_old_open = sum(r["accepted"] for r in old)
    pilot_names = [r["circuit"] for r in pilots]
    added = [r["circuit"] for r in cur if r["accepted"] and r["circuit"] not in pilot_names]
    w_("**Reading.**")
    w_("")
    w_(f"- The plan's gate rejects all {len(old)} outdated TUMFTM layouts, as it should. It "
       f"also rejects {sum(not r['plan_gate'] for r in cur)} of the {len(cur)} current "
       f"layouts. It passes only {', '.join(r['circuit'] for r in cur if r['plan_gate'])}, "
       "and that alone chose the four pilot circuits. They are now pinned by session key "
       "(`PILOT_KEYS` in `drivers_pilot.py`), so the change of gate does not move the "
       "pilot.")
    w_("- The likely cause is the feed's track model, which is not the centreline. In "
       "corners tighter than 150 m radius it sits "
       f"{_range([r['inside_offset_m'] for r in cur], '{:.1f}')} m "
       "to the inside, and its lap is "
       f"{_range([-100 * r['length_mismatch'] for r in cur], '{:.1f}')}% shorter. That looks "
       "like a racing line (an inference from these two statistics; not checked against "
       "an outside source). The 1% length test compares that line's length with the "
       "centreline's, so it is biased by design, and the 3 m RMS mixes in the lateral "
       "line offset. A twisty circuit can fail on the line alone.")
    w_("- The excursion beyond the TUMFTM edges separates the two groups much more sharply. "
       "At the 99th percentile it is at most "
       f"{max(r['excursion_p99_m'] for r in cur):+.1f} m on current layouts and at least "
       f"{min(r['excursion_p99_m'] for r in old):+.1f} m on old ones. **This is now the "
       f"gate** (docs/STATUS.md, Decisions taken): excursion p99 <= {GATE_EXCURSION_M:g} m, "
       f"with RMS and length as diagnostics. On this table it passes {n_open} of {len(cur)} "
       f"current layouts and {n_old_open} of {len(old)} old ones. The threshold was picked "
       "after looking at this table, so it is checked on other seasons below.")
    w_("- Projecting a racing-line model onto the centreline is still a fixed, monotone "
       "map, the same for every car. So s(t) stays comparable between teammates. The cost "
       "is a speed / centreline-rate ratio that varies along the lap. The smoother "
       "estimates it from all laps (section 2).")
    w_("")
    if reg_other:
        w_(f"**The gate on seasons it was not chosen on.** The same registration on "
           "every mapped qualifying session of other years (same code, same "
           f"{GATE_EXCURSION_M:g} m threshold):")
        w_("")
        w_("| Season | Current layouts | Plan gate passes | Excursion gate passes | "
           "Max excursion p99, current (m) | Old layouts | Excursion gate passes (old) | "
           "Min excursion p99, old (m) |")
        w_("|---|---|---|---|---|---|---|---|")
        for year, rows in sorted(reg_other.items()):
            c = [r for r in rows if r.get("tumftm") and r["layout_matches"]]
            o = [r for r in rows if r.get("tumftm") and not r["layout_matches"]]
            exc_c = [r["excursion_p99_m"] for r in c]
            exc_o = [r["excursion_p99_m"] for r in o]
            w_(f"| {year} | {len(c)} | {sum(r['plan_gate'] for r in c)} | "
               f"{sum(r['accepted'] for r in c)} | "
               f"{max(exc_c) if exc_c else float('nan'):+.1f} | {len(o)} | "
               f"{sum(r['accepted'] for r in o)} | "
               f"{min(exc_o) if exc_o else float('nan'):+.1f} |")
        w_("")
    w_("The pilot's four sessions were chosen under the plan's gate and are kept; every "
       "number in sections 2-7 comes from them. The 2025 sessions the excursion gate adds ("
       f"{', '.join(added) or 'none'}) are not run in this pilot. Checking the TUMFTM length "
       "against the official circuit length (which would catch layout changes without the "
       "line bias) was not done: it needs an official-length table that is not in the "
       "repository.")
    w_("")

    # ------------------------------------------------------------------ 2. s(t)
    best = tune["best"]
    edges = _grid_edges(tune["rows"], best)
    near = [r for r in tune["rows"] if r["pair_rms_ms"] <= 1.1 * best["pair_rms_ms"]]
    jerks = sorted({r["jerk_psd"] for r in tune["rows"]})
    sigmas = sorted({r["sigma_speed"] for r in tune["rows"]})
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
       f"geometry ({', '.join(tune['circuits'])}), over jerk PSD "
       f"{', '.join(f'{j:g}' for j in jerks)} m^2/s^5 x speed sigma "
       f"{', '.join(f'{s:g}' for s in sigmas)} m/s. The best point is jerk PSD "
       f"{best['jerk_psd']:g} and speed sigma {best['sigma_speed']:g} m/s (pair-delta RMS "
       f"{best['pair_rms_ms']:.1f} ms)" + (f", which is {' and '.join(edges)}, so the "
                                         "optimum may lie outside the grid" if edges else
                                         ", inside the grid on both axes") +
       f". The surface is flat: {len(near)} of {len(tune['rows'])} grid points score within "
       f"10% of the best ({_range([r['pair_rms_ms'] for r in near], '{:.1f}')} ms).")
    w_("")
    w_("**Held-out sector anchor (plan gate 3).** Each lap is refitted without the S2/S3 "
       "anchor, the time it reaches the S2/S3 line is predicted, and it is compared with "
       "the official S1 + S2. For each lap the line's position is the median over the "
       "*other* laps, and the path-ratio profile is re-estimated without any S2 anchor; "
       "the start and S1 lines and the start-time corrections still use every lap. The "
       "same check on synthetic laps calibrates the synthetic feed. The line wobble "
       "(`SynthConfig.wobble_sd`) is the one chosen from "
       f"{sorted(next(iter(cals.values())).tried)} whose synthetic held-out spread is "
       "closest to the real one. Natural lap-to-lap variation is then set per corner type "
       "x phase, from the real within-driver spread minus the synthetic measurement error "
       "(table below).")
    w_("")
    w_("| Session | Laps fitted | Median (ms; line offset, not a bias) | Held-out robust sd "
       "(ms) | p95 abs (ms) | Synthetic robust sd (ms) | Wobble | Start corr. sd (ms) | "
       "Position sigma (m) |")
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
    medians = [1000 * np.median(x.held_out_s2) for x in res.values()]
    w_("A single crossing time far from the anchors is good to "
       f"{_range([1000 * recovery.robust_sd(x.held_out_s2) for x in res.values()])} ms "
       "(robust sd); that spread is the held-out statistic. The synthetic column comes from "
       f"the injection runs themselves. The per-session medians ({_range(medians, '{:+.0f}')}"
       " ms) are not an absolute bias and say nothing about a physics reference: the line "
       "position itself comes from the unanchored fits, so the median only measures how "
       "far the anchored and unanchored fits disagree about where the S2 line is. It is "
       "common to all laps of a session and cancels between teammates.")
    w_("")
    n_cx = {x.label.split()[1]: len(x.seg.complexes) for x in res.values()}
    retyped = sum(c.kind != corner_type(c.v_min) for x in res.values() for c in x.seg.complexes)
    n_complex = sum(len(x.seg.complexes) for x in res.values())
    w_(f"**Segmentation.** Complexes per circuit: "
       f"{', '.join(f'{k} {v}' for k, v in n_cx.items())}. The plan asks for 12-20. "
       "Corners less than 80 m apart always merge, because at 3.7 Hz (one sample every "
       "6-25 m) a chicane's halves cannot be timed apart. Below 12, the code keeps the "
       "natural count rather than splitting chicanes or straights to fill a quota. Each "
       "complex is split into entry / apex / exit at positions on the pooled (median) "
       "reference speed. It is typed (slow < 120, medium 120-200, fast > 200 km/h) by the "
       "observed envelope apex speed, as the plan specifies: the 95th percentile of the "
       "clean laps' own speed, a percentile rather than the maximum so that one bad fit "
       f"cannot move a corner across a threshold. {retyped} of {n_complex} complexes get a "
       "different type than the median lap would give them.")
    w_("")
    w_("![pilot](drivers-pilot.png)")
    w_("")

    # ------------------------------------------------------------------ 3. lap filter
    w_("## 3. Lap filter")
    w_("")
    w_("| Session | Laps fitted (checks 1-7 passed) | Traffic (< 1.0 s ahead in a corner) | "
       "Pass every check | New set (of those) | Exclusions before fitting (all laps) | "
       "Teammate pairs | Laps per driver in pairs |")
    w_("|---|---|---|---|---|---|---|---|")
    shares = []
    lap_counts = [n for d in pairs_all for n in (d.n_a, d.n_b)]
    for x in res.values():
        reasons = collections.Counter(r for st in x.statuses.values() for r in st.reasons)
        traffic = reasons.pop("traffic", 0)
        shares.append(traffic / len(x.timing.laps))
        other = ", ".join(f"{k} {v}" for k, v in reasons.most_common())
        pairs = dl.teammate_deltas(x.table)
        counts = collections.Counter(n for d in pairs for n in (d.n_a, d.n_b))
        share = 100 * traffic / len(x.timing.laps)
        new = int(np.sum(x.table.new_set)) if x.table.new_set is not None else 0
        w_(f"| {x.label} | {len(x.timing.laps)} | {traffic} ({share:.0f}%) "
           f"| {len(x.table.keys)} | {new} | {other} | {len(pairs)} | "
           f"{', '.join(f'{k}: {v}' for k, v in sorted(counts.items()))} |")
    w_("")
    w_(f"Traffic removes {_range([100 * s for s in shares])}% of otherwise clean push laps. "
       "That is the plan's rule taken literally: any car less than 1.0 s ahead anywhere in "
       "any corner window. Spot checks at Montreal found real slow cars on out-laps and "
       "cool-down laps, on the racing line (the same lateral offset as fast cars, so not "
       "the pit lane). Other cars' positions are projected globally; samples more than 7 m "
       "from the centreline (pit lane, garage) and samples further from the previous one "
       "than a car can drive (branch jumps at Suzuka's crossover) are dropped. With the "
       "102% rule per Q segment and laps compared only within a Q segment, pairs end up "
       f"with {_range(lap_counts)} laps per driver (last column: laps -> number of "
       "drivers), most with 1 or 2. That is the "
       "binding constraint on everything below. *New set* is a covariate (the stints "
       "table's tyre age <= 1), not a filter. Tyre compounds come from the OpenF1 stints "
       "table only; Montreal's MEDIUM count is large and was not cross-checked against a "
       "second source.")
    w_("")
    w_("Deleted laps: race control's 'LAP n' is one ahead of OpenF1's lap numbering (in "
       "2023-25 qualifying, 483 of the 489 deletion messages that give both a lap number "
       "and the deleted time give OpenF1 lap n - 1's time to the millisecond). The parser "
       "now matches the deleted time to a lap duration first, falls back to n - 1, and "
       "handles messages with neither (track-limit '(NEXT LAP PIT)' and double-yellow "
       "deletions) and reinstatements.")
    w_("")

    # ------------------------------------------------------------------ 4. injection
    rec0 = pilot_recs[0]
    qcfg = inj["qss_config"]
    w_("## 4. Injection-recovery error budget (plan gate 2)")
    w_("")
    n_reps = len(set(rec0.rep.tolist()))
    w_(f"For each pilot circuit: {rec0.pair_est.shape[0]} synthetic teammate pairs "
       f"({rec0.pair_est.shape[0] // n_reps} pairs x {n_reps} sessions), "
       f"{rec0.laps_per_driver} laps per driver in one Q segment, comparing each driver's "
       f"top-{rec0.top_n} laps as in the real analysis. Every B driver got injected effects "
       "of +40 ms in each slow-corner entry, +20 ms in each medium-corner apex and -20 ms "
       "in each fast-corner exit. Every phase of every lap also carries natural variation "
       "at the calibrated per-cell level (section 2).")
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
    w_("The laps then go through the real s(t), phase and teammate-delta code. **Held "
       "fixed, so not part of this budget:** registration (the session's transform is "
       "reused), segmentation and corner types (the session's), the lap filter, traffic "
       "and the Q-segment structure (every synthetic lap is a clean push lap in a single "
       "segment), and the track model (synthetic positions scatter about the TUMFTM "
       "centreline, not about a racing-line-like model as the real feed does). The line "
       "wobble is drawn independently for every lap; persistent per-car errors are added "
       "only in the sensitivity run below.")
    w_("")
    w_("*Measurement* = estimated minus true delta over the same laps: the pipeline's own "
       "error. *Recovered - injected* adds natural "
       f"variation over {rec0.top_n} laps per driver. Its 2 sigma is the **claim limit**, "
       "the plan's \"2 sigma of the resulting error\". It applies to a pair with 2 laps "
       "per driver in one Q segment; for other lap counts the example in section 6 scales "
       "it by sqrt(var factor) (`PairDelta.var_factor`), which assumes independent laps.")
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
    fpr, fpr_n = rates["fpr"]
    det = rates["detection"]

    def _det(k):
        r, n = det.get(k, (float("nan"), 0))
        return f"{100 * r:.0f}% of {n}"

    w_("**False positives and detection, out of sample.** The claim limits are fitted on "
       f"one synthetic session per circuit and scored on the other (and the reverse), so "
       f"the rate is not 5% by construction. {100 * fpr:.1f}% of {fpr_n} zero-injected "
       "phase cells exceed their limit (about 5% is expected at 2 sigma; neighbouring "
       "injected phases also leak into them). Detection, right sign and above the limit: "
       f"slow entry (+40 ms) {_det(('slow', 'entry'))}, medium apex (+20 ms) "
       f"{_det(('medium', 'apex'))}, fast exit (-20 ms) {_det(('fast', 'exit'))} cells.")
    w_("")
    se = by.get(("slow", "entry"))
    ratio_by = {k: budget_qss[k].claim_limit_ms / by[k].claim_limit_ms for k in by
                if k in budget_qss}
    qss_ratio = list(ratio_by.values())
    worst = max(ratio_by.items(), key=lambda kv: abs(np.log(kv[1])))
    w_("**Reading.** The injected effects come back unbiased within their error: recovered "
       f"means are {_range([b.recovered_ms - b.injected_ms for b in budget], '{:+.1f}')} ms "
       "off the injected values. The smallest claimable per-phase teammate effect, from one "
       f"car-session with {rec0.top_n} laps per driver, is "
       f"{_range([b.claim_limit_ms for b in budget])} ms depending on corner type and phase. "
       f"The pipeline's own share is {_range([b.measurement_limit_ms for b in budget])} ms. "
       f"A +40 ms slow-corner entry is {40 / se.claim_limit_ms:.1f}x its limit "
       f"({se.claim_limit_ms:.0f} ms), and one synthetic car-session flags it in "
       f"{_det(('slow', 'entry')).split(' of')[0]} of cases. Natural lap-to-lap variation, "
       "not the pipeline, dominates the limit in slow and medium phases. The pipeline's own "
       "error dominates in fast phases. "
       "Ratios of QSS-profile limits (R-dev geometry, median calibration: wobble "
       f"{qcfg.wobble_sd:g}, natural "
       f"{_range([1000 * v for v in inj['qss_natural'].values()])} ms by cell) to pilot "
       f"limits are {_range(qss_ratio, '{:.2f}')}, median {np.median(qss_ratio):.2f}. "
       f"{sum(0.745 <= q <= 1.255 for q in qss_ratio)} of {len(qss_ratio)} cells agree "
       f"within 25%. The largest disagreement is {worst[0][0]} {worst[0][1]} "
       f"({worst[1]:.2f}), where the calibrated natural variation is "
       f"{1000 * inj['qss_natural'].get(worst[0], float('nan')):.0f} ms, so the limit is "
       "mostly measurement error, and that does depend on the profile.")
    w_("")
    scfg = next(iter(pers["config"]["both"].values()))
    sys_wob = _range([c.driver_wobble_sd for c in pers["config"]["both"].values()], "{:g}")
    w_("**Persistent per-car errors (sensitivity run).** The per-lap budget above lets "
       "every error average down over laps. Two that would not were added to the same "
       "synthetic sessions (same seeds; the persistent terms come from a separate random "
       "stream, so the true phase times and every other draw are identical): a line "
       "pattern shared by all of a driver's laps, as large as the calibrated per-lap "
       f"wobble ({sys_wob}), as if a driver always took a different line than the pooled "
       "path-ratio profile assumes; and an along-track position offset per car of sd "
       f"{scfg.car_offset_sd:g} m (antenna or transponder position). Each was run alone and "
       "then both together. These sizes are a scenario, not a measurement. The extra "
       "error they add to a teammate delta (sd over pairs and cells):")
    w_("")
    w_("| Corner type | Phase | Line pattern only (ms) | Position offset only (ms) | "
       "Both (ms) | Both, 2 sigma (ms) | Both, lap sum of the cell (ms) | "
       "Claim limit per car-session (ms) |")
    w_("|---|---|---|---|---|---|---|---|")
    for k in sorted(sys_err, key=lambda k: (TYPES.index(k[0]), PHASES.index(k[1]))):
        sd_cell, sd_sum, _ = sys_err[k]
        w_(f"| {k[0]} | {k[1]} | {1000 * sys_by['line'][k][0]:.1f} | "
           f"{1000 * sys_by['offset'][k][0]:.1f} | {1000 * sd_cell:.1f} | {2000 * sd_cell:.0f} | "
           f"{1000 * sd_sum:.1f} | {by[k].claim_limit_ms if k in by else float('nan'):.0f} |")
    w_("")
    line_ms = np.median([1000 * v[0] for v in sys_by["line"].values()])
    off_ms = np.median([1000 * v[0] for v in sys_by["offset"].values()])
    w_(f"Median over cells: {line_ms:.1f} ms from the line pattern, {off_ms:.1f} ms from "
       f"the {scfg.car_offset_sd:g} m position offset. Unlike the per-lap errors, both are "
       "the same in every lap of a car, so more laps do not reduce them.")
    w_("")
    proj = [np.sqrt((b.claim_limit_ms / np.sqrt(SEASON)) ** 2
                    + (2000 * sys_err[(b.kind, b.phase)][0]) ** 2)
            for b in budget if (b.kind, b.phase) in sys_err]
    proj_naive = [b.claim_limit_ms / np.sqrt(SEASON) for b in budget]
    proj_sum = [np.sqrt((2 * a["recovered_sd_ms"] / np.sqrt(SEASON)) ** 2
                        + (2000 * sys_err[k][1]) ** 2) for k, a in agg.items() if k in sys_err]
    w_(f"**Season projection, qualified.** If {SEASON} car-sessions per pair were "
       "independent and nothing about a car persisted, the per-phase limits would shrink "
       f"to {_range(proj_naive)} ms (limit / sqrt({SEASON})). A persistent per-car error "
       "does not shrink with more sessions. With the scenario above treated as such a "
       f"floor, the per-phase limits only reach {_range(proj)} ms, and the corner type x "
       f"phase lap sums {_range(proj_sum)} ms. Neither number is a measured property of "
       "the feed: the persistent error sizes are assumed. Split-half across sessions "
       "(section 5) is the data's own test of what persists.")
    w_("")

    # ------------------------------------------------------------------ 5. placebo
    w_("## 5. Placebo and split-half reliability (real data)")
    w_("")
    w_(f"**Placebo (gate 4).** Driver labels were shuffled within each car-session *and Q "
       f"segment* ({placebo.n_shuffles} shuffles, top-2 laps per Q segment, "
       f"{placebo.n_pairs} pairs, {placebo.n_cells} phase cells). The signed mean of "
       "shuffled deltas is not reported: it is about zero by symmetry for any pipeline, "
       "so it tests nothing. Three statistics that can fail:")
    w_("")
    w_(f"- *Synthetic limits against the real null.* {100 * shuffled_share:.1f}% of shuffled "
       "cells exceed the synthetic claim limit for their corner type and phase (scaled to "
       f"each pair's lap counts; {share_cells} cells). If the synthetic budget were as "
       "wide as the real null spread this would be about 5%; more means the synthetic "
       "budget is too narrow for the real data, less means it is too wide. Shuffled real "
       "data still contain any real effect, mixed between the labels, which pushes this "
       f"share up. With the true labels the share is {100 * real_share:.1f}%. For 5% of "
       f"shuffled cells to exceed them, the limits would have to be {widen:.2f}x wider; "
       "until that is understood, the section 4 limits should be read as optimistic.")
    w_("- *Real against shuffled |delta|, per corner type x phase* (1 = the true labels "
       "explain nothing beyond lap-to-lap variation):")
    w_("")
    w_("| Cell | " + " | ".join(f"{k} {ph}" for k, ph in cells) + " |")
    w_("|---|" + "---|" * len(cells))
    w_("| Mean abs. real / shuffled | " + " | ".join(
        f"{ratios[c][0]:.2f}" if c in ratios else "-" for c in cells) + " |")
    w_("| Cells | " + " | ".join(str(ratios[c][1]) if c in ratios else "-" for c in cells)
       + " |")
    w_("")
    w_(f"- *Pooled permutation test.* The sum of squared deltas over all cells under the "
       f"true labels against the same sum under shuffled labels gives p = "
       f"{placebo.ss_p_value:.3f}: the probability that random labels explain at least as "
       "much phase-level variance as the real ones.")
    w_("")
    w_("A small p or a ratio above 1 says the labels carry structure. It does not say "
       "that structure is the driver's: setup and any per-car measurement offset (antenna "
       "position, a car that always takes a different line than the pooled profile "
       "assumes) are confounded with the label, and a within-car shuffle cannot tell them "
       "apart. A per-cell test is not possible here: with 2 laps per driver there are only "
       "3 distinct splits.")
    w_("")
    w_("**Split-half (gate 5).** The gate is Spearman rho >= 0.5. Halves are cut inside "
       "each Q segment. The p-value flips the sign of whole pairs' second halves at "
       "random (2000 draws, one-sided), because cells of one pair share laps and "
       "session-level errors and are not independent; the naive p that treats every cell "
       "as independent is shown for comparison.")
    w_("")
    w_("| Split | Level | Pairs | Cells | Spearman rho | p (pair-level permutation) | "
       "p (naive) |")
    w_("|---|---|---|---|---|---|---|")
    for sh, split, lev in ((sh_phase, "odd vs even ranked laps, within session", "phase"),
                           (sh_type, "odd vs even ranked laps, within session",
                            "corner type x phase"),
                           (sh_across, "alternate sessions (pairs in both halves)",
                            "corner type x phase")):
        w_(f"| {split} | {lev} | {sh.n_pairs} | {sh.n_cells} | {sh.rho:+.2f} | "
           f"{sh.p_value:.3f} | {sh.p_naive:.2g} |")
    w_("")
    w_("No split reaches the 0.5 gate. The within-session halves need 2+ clean laps per "
       "driver in the same Q segment, so few pairs qualify and most compare one lap with "
       "one lap. Driver, setup and any per-car measurement offset are confounded in them. "
       "The across-session split has only the pairs that appear in both halves of four "
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
    sub = x.table.rows(x.table.team_mask(team))
    pb = dl.placebo([sub], top_n=2, n_shuffles=500, seed=2)
    segs = "; ".join(f"Q{q}: {acr.get(d.a, d.a)} {na}, {acr.get(d.b, d.b)} {nb}"
                     for q, na, nb in d.segments)
    w_(f"{x.label}, {team}. Delta = {acr.get(d.b, d.b)} minus {acr.get(d.a, d.a)}, so "
       f"positive means {acr.get(d.b, d.b)} took longer. Laps are compared only within a "
       f"Q segment (laps used: {segs}); each segment's delta is the difference of the two "
       "drivers' mean phase times there (top-2 per driver), and the segments are averaged "
       f"with weights 1 / (1/n_A + 1/n_B). The lap delta is {_ms(d.lap_delta)} ms, and the "
       "phase deltas below add up to it exactly. A cell would be **bold** only if it "
       "cleared both the claim limit for its corner type and phase, scaled to these lap "
       f"counts (x {np.sqrt(d.var_factor):.2f}), and twice this car's own placebo spread.")
    w_("")
    w_("| Complex | Type | Entry (ms) | Apex (ms) | Exit (ms) | Complex (ms) | Above both limits |")
    w_("|---|---|---|---|---|---|---|")
    survive, fall = [], []
    for c in x.seg.complexes:
        cols = [i for i, (cn, _, _) in enumerate(x.table.labels) if cn == c.name]
        cells_, marks = [], []
        for i in cols:
            _, ph, kind = x.table.labels[i]
            v = d.delta[i]
            lim = limits.get((kind, ph), np.inf) * np.sqrt(d.var_factor)
            ok = abs(v) > lim and abs(v) > 2 * pb.cell_null_sd[i]
            cells_.append(f"**{_ms(v)}**" if ok else _ms(v))
            if ok:
                marks.append(ph)
                floor = np.hypot(lim, 2 * sys_err.get((kind, ph), (np.nan,))[0])
                (survive if abs(v) > floor else fall).append(f"{c.name} {ph}")
        w_(f"| {c.name} | {c.kind} | {' | '.join(cells_)} | {_ms(d.delta[cols].sum())} | "
           f"{', '.join(marks) or '-'} |")
    w_("")
    if survive or fall:
        w_("Neither bold rule includes persistent per-car errors or the section 5 finding "
           "that the synthetic limits are too narrow for the real null. With the section 4 "
           "persistent-error scenario added to the limit (in quadrature, 2 sigma), "
           + (f"{', '.join(survive)} would still clear it" if survive else "no bold cell "
              "would still clear it")
           + (f" and {', '.join(fall)} would not" if survive and fall else "") + ". The "
           f"table has {len(d.delta)} cells, and on shuffled real data "
           f"{100 * shuffled_share:.0f}% of cells exceed the limit (section 5), so about "
           f"{len(d.delta) * shuffled_share:.1f} cells above it are expected with no driver "
           "effect at all.")
        w_("")
    w_(f"{len(multi_q)} of {len(pairs_all)} pilot pairs have clean laps from both drivers "
       "in more than one Q segment. With this few laps per driver, single cells mostly sit "
       "inside the error budget. That is the expected pilot outcome. It is why the plan "
       "pools over a season with a hierarchical model before saying anything about a "
       "driver.")
    w_("")

    # ------------------------------------------------------------------ 7. R1
    w_("## 7. R1 envelope (input to the later model)")
    w_("")
    w_("R1 is, per car-session, the best time in each phase across both teammates' clean "
       "laps; the plan's model works on y = t - R1 per lap and phase. Summary only (no "
       "per-driver numbers). *Best lap - R1* is how far a car-session's best clean lap is "
       "from the sum of its best phases (car-sessions with 2+ clean laps); the last three "
       "columns are the median per-lap y summed over each corner type's phases.")
    w_("")
    w_("| Session | Car-sessions with clean laps | Median best lap - R1 (ms) | "
       "Range (ms) | Median y, slow (ms) | Median y, medium (ms) | Median y, fast (ms) |")
    w_("|---|---|---|---|---|---|---|")
    for x in res.values():
        n_cs, gaps, by_type = _r1_summary(x)
        w_(f"| {x.label} | {n_cs} | {1000 * np.median(gaps):.0f} | "
           f"{_range([1000 * g for g in gaps])} | "
           + " | ".join(f"{1000 * by_type[k]:.0f}" for k in TYPES) + " |")
    w_("")
    w_("The R1 envelope is a minimum over a handful of noisy laps, so it is biased low "
       "(more laps, lower envelope), and with one clean lap a car-session's y is zero "
       "everywhere. Teammate deltas do not depend on it.")
    w_("")

    # ------------------------------------------------------------------ 8. caveats
    w_("## 8. Not done, or uncertain")
    w_("")
    w_(f"- **Registration gate.** The plan's gate passed {sum(r['plan_gate'] for r in cur)} "
       f"of {len(cur)} current-layout 2025 circuits (section 1), and that is what limited "
       "the pilot to four circuits. The corridor-excursion gate that replaced it passes "
       f"{n_open} of {len(cur)}; the pilot was not widened to the others.")
    w_("- **Scope.** Four circuits, one qualifying session each. Sprint qualifying was not "
       "run. Most circuits stay below the plan's 12 complexes because chicanes are never "
       "split.")
    w_("- **Synthetic realism.** The synthetic feed is calibrated on two statistics per "
       "session (held-out S2 spread; within-driver spread per corner type x phase). The "
       "single wobble knob stands in for every unmodelled error source, so how the "
       "measurement error splits between phases is a model assumption (section 2 table). "
       "Natural variation is Gaussian and independent between phases. Persistent per-car "
       "errors are only a sensitivity scenario with assumed sizes (section 4). The ~0.7 m "
       "position pattern common to all cars is not synthesised; it is common-mode, but "
       "that it cancels is untested. No clock latency (none was measured). The QSS check "
       "profiles are computed on the centreline, not a racing line.")
    w_(f"- **The synthetic budget looks too narrow.** On shuffled real data "
       f"{100 * shuffled_share:.0f}% of cells exceed the claim limits instead of about 5% "
       f"(section 5); the limits would need to be {widen:.2f}x wider. Part of that may be "
       "real effects mixed into the shuffles, part unmodelled real error.")
    w_("- **Smoother.** " + (f"The tuned point is {' and '.join(edges)} of the extended "
                             "grid, so the optimum may lie further out. " if edges else
                             "The tuned point is inside the extended grid. ") +
       "The error surface is flat near the optimum.")
    w_("- **Lap filter.** Compounds come from the stints table only (no second source). "
       "The new-set flag is a covariate and the tow flag is computed, but neither enters "
       "the teammate deltas yet.")
    w_("- **Not built (later plan steps).** The hierarchical model (M1/M2), R2 calibration, "
       "the circularity check, the multiverse and external validity. Brake and throttle "
       "are loaded but unused.")
    w_("- **Underpowered gates.** Many real pairs have 1 lap per driver in a segment. The "
       "placebo tests label exchangeability within car-sessions, not across a season.")
    w_("")
    cpu = sum(getattr(x, "cpu_s", 0.0) for x in res.values())
    w_(f"Compute: the pilot stage took {cpu:.0f} s of CPU over four sessions. Registration "
       "takes ~2 s of CPU per session; calibration, injection-recovery and tuning take a "
       "few minutes each. Everything ran single-process at nice 15 on the M3 Pro alongside "
       "other jobs.")
    w_("")

    # ------------------------------------------------------------------ 9. changes
    w_("## 9. Changes after code review")
    w_("")
    w_("The first version (archived with its outputs in `outputs/drivers-lite/archive-v1/`) "
       "had these problems; every number above is from the corrected code.")
    w_("")
    w_(f"- **Teammate laps from different Q segments.** Each driver's top-2 laps were taken "
       f"from all shared Q segments together, so in {V1['mismatched_pairs']} of "
       f"{V1['pairs']} pairs (including the old Spa example table) one driver's laps came "
       "from a different segment than the other's, and track evolution entered the delta. "
       "Laps are now compared only within a segment; a test checks that a Q-segment-wide "
       "offset leaves every delta unchanged. The placebo and split-half are stratified by "
       f"Q segment too. The first version's placebo p = {V1['placebo_p']} and split-half "
       f"rho {V1['rho_phase']} / {V1['rho_type']} / {V1['rho_across']} are superseded.")
    w_("- **Deleted laps off by one.** Race control's 'LAP n' was read as OpenF1 lap n; it "
       "is lap n - 1. Every deletion flagged the lap after the deleted one, and two truly "
       "deleted push laps (Spa, Monza) were in the clean tables.")
    w_("- **Placebo statistic.** The signed mean of shuffled deltas (zero by symmetry) was "
       "read as evidence; it is replaced by the statistics in section 5.")
    w_(f"- **In-sample false-positive rate.** The {V1['fpr']}% was scored against limits "
       "fitted on the same synthetic cells; it is now out of sample.")
    w_("- **Held-out S2 not fully held out**, and its median read as a model bias (section "
       "2). **Smoother grid** stopped at the optimum on both axes (jerk "
       f"{V1['jerk']}, speed sigma {V1['sigma_speed']}); it is wider now. **Calibration "
       "test** replayed one candidate's seed and wobble. **Interior position gaps** of up "
       "to 4.5 s were accepted. **Traffic** could see phantom cars at Suzuka's crossover. "
       "**Corner types** came from the median lap, not the envelope. **R1** was built but "
       "not reported.")
    (DOCS / "drivers-pilot.md").write_text("\n".join(L) + "\n")
    print(f"wrote {DOCS / 'drivers-pilot.md'}")
