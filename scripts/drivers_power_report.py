"""Write docs/findings/drivers-power.md from outputs/drivers-lite/power/power.json.

Called by ``drivers_power.py``; every number in the prose is computed here from the cached
result. EXPLORATORY: the plan, model and go/no-go rule are in drivers-power-plan.md.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from drivers_power import (
    BLUE,
    DOCS,
    GATE,
    GRID,
    INK,
    KS,
    MUTED,
    ORANGE,
    SCHEMES,
    SCOPES,
    TYPES,
    _first_cross,
)

from apexzero.drivers.segments import PHASES

STAT_NAME = {"profile": "profile", "raw": "raw"}
SCHEME_NAME = {"3": "3 phases", "2a": "2 phases (entry + apex | exit)",
               "2b": "2 phases (entry | apex + exit)"}


def _pp(res, scope, scheme, stat, k, persist=False):
    return np.array(res["sweep"]["p_pass"][scope][f"{scheme}|{stat}|{k}|{int(persist)}"])


def _fmt_tau(x):
    return "> 40" if x is None else f"{x:.0f}"


def _fmt_p(x):
    return "-" if x is None else f"{x:.2f}"


def _figure(res, path):
    taus = np.array(res["taus_ms"])
    cal = res["calibration"]["by_k"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, sharey=True)
    for i, stat in enumerate(("profile", "raw")):
        for j, two in enumerate((False, True)):
            ax = axes[i, j]

            def curve(scope, k, two=two, stat=stat):
                if not two:
                    return _pp(res, scope, "3", stat, k)
                return np.minimum(_pp(res, scope, "2a", stat, k), _pp(res, scope, "2b", stat, k))

            ax.plot(taus, curve("q", 1.36), color=BLUE, lw=2, marker="o", ms=4,
                    label="35 qualifying sessions, noise x1.36")
            ax.plot(taus, curve("q", 1.0), color=BLUE, lw=2, ls="--", marker="o", ms=4,
                    label="35 qualifying sessions, noise x1")
            ax.plot(taus, curve("q+sq+2026", 1.36), color=ORANGE, lw=2, marker="o", ms=4,
                    label="plus sprint qualifying and 2026, noise x1.36")
            ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
            t80 = cal["1.36"][stat]["tau_80_ms"]
            th = cal["1.36"][stat]["tau_hat_ms"]
            if t80 is not None:
                ax.axvline(t80, color=INK, lw=0.9)
                ax.text(t80 + 0.6, 0.04, "pilot\n80% bound", fontsize=8, color=INK)
            if th is not None:
                ax.axvline(th, color=INK, lw=0.9, ls="--")
            ax.set_title(f"{STAT_NAME[stat]} statistic, "
                         f"{'2 phases (worse merge)' if two else '3 phases'}", fontsize=10,
                         color=INK)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(MUTED)
            ax.tick_params(colors=MUTED)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            if i == 1:
                ax.set_xlabel("tau: sd of the per-complex teammate-label effect (ms)", color=INK)
            if j == 0:
                ax.set_ylabel(f"P(split-half rho >= {GATE:g})", color=INK)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    axes[0, 0].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _reading(res: dict) -> list[str]:
    taus = np.array(res["taus_ms"])
    dec = res["decision"]
    cal = res["calibration"]["by_k"]["1.36"]
    t80, th = cal["profile"]["tau_80_ms"], cal["profile"]["tau_hat_ms"]

    def need(scope, scheme, stat="profile", k=1.36, persist=False):
        return _first_cross(taus, _pp(res, scope, scheme, stat, k, persist), 0.5)

    base = need("q", "3")
    wide = need("q+sq+2026", "3")
    two = [need("q", s) for s in ("2a", "2b")]
    out = ["## 5. Reading", ""]
    if base is not None and t80:
        out.append(f"- **The gap.** On the 35 qualifying sessions with 3 phases and noise x1.36, "
                   f"the profile gate passes half the time at tau = {base:.0f} ms per complex. "
                   f"The pilot's own data put tau at about {th:.0f} ms, with an 80% upper bound "
                   f"of {t80:.0f} ms: the gate needs an effect {base / t80:.1f} times the "
                   "bound. Read on the lap, tau is per complex of a type and phase, so on a "
                   "circuit with four fast complexes the fast-entry lap sum has an sd of about "
                   f"{4 * base:.0f} ms at the gate's tau.")
    if wide is not None and base is not None:
        out.append(f"- **More sessions {'barely help' if base - wide < 3 else 'help, not enough'}."
                   f"** Adding sprint qualifying and 2026 moves the 50% point to {wide:.0f} ms "
                   f"(from {base:.0f}), and that scope is optimistic: neither set is registered "
                   "and sprint qualifying has fewer laps. More sessions shrink the noise of each "
                   "half only as the square root of their number.")
    if None not in two and base is not None:
        same = max(abs(t - base) for t in two) < 1.0
        out.append(f"- **Two phases** {'leave the 50% point where it is' if same else 'move the 50% point'}"
                   f": {two[0]:.0f} ms (entry + apex | exit) and {two[1]:.0f} ms (entry | apex + "
                   f"exit), against {base:.0f} ms with three. Merging cancels the timing error at "
                   "the dropped boundary, but a merged cell also sums two effects and two "
                   "phases' natural variation, so the choice of 2 or 3 phases does not decide "
                   "the gate.")
    raw = need("q", "3", "raw")
    if raw is not None and base is not None:
        out.append(f"- **Raw against profile.** The raw statistic's 50% point is {raw:.0f} ms. "
                   "Most of a teammate gap changes from session to session (section 1), so "
                   "the overall gap adds noise to the raw statistic about as much as it adds "
                   "reliable signal; the raw statistic does not pass for free.")
    fp_q = _pp(res, "q", "3", "profile", 1.36, True)[0]
    fp = max(_pp(res, s, sch, "profile", 1.36, True)[0] for s in SCOPES for sch in SCHEMES)
    pc = res["calibration"]["persistent_tau0"]["1.36"]["profile"]
    out.append(f"- **A pass would not mean much on its own.** With the persistent-error scenario "
               f"held across sessions, the profile gate passes with no driver effect in "
               f"{100 * fp_q:.0f}% of seasons on the 35 sessions with 3 phases, and up to "
               f"{100 * fp:.0f}% with more sessions or 2 phases (section 4). The pilot argues "
               "against errors of that size repeating: had they repeated, its profile rho would "
               f"have had a median of {pc['median']:+.2f}, and the observed one or lower in "
               f"{100 * pc['p_at_or_below_observed']:.1f}% of cases (a post hoc check). A "
               "smaller per-car line or antenna offset that does repeat is still "
               "indistinguishable from a driver effect in this design.")
    out += ["", "## 6. What it means for DV-B", ""]
    if dec["go"] is False:
        out += [
            ("**No-go.** Under every choice decision #18 leaves open (2 or 3 phases, raw or "
            "profile, sprint qualifying and 2026 in or out) the gate would most likely fail "
            "at the effect sizes the pilot allows, so a pre-registered run would spend 15-25 h "
            "to confirm a failure the simulation already predicts. DV-B is not run. The "
            "two-track plan's fallback stands: 2-3 teammate-pair case studies, labelled as "
            "case studies, each cell above its claim limit and its placebo spread. The #18 "
            "choices need not be made unless something below changes."),
            "",
            ("What would change the verdict: a real effect several times larger than the pilot's "
            "bound (the pilot has nine pairs, so this cannot be ruled out, only made unlikely); "
            "a lower per-car-session noise (more laps per driver, or a better timing model); or "
            "a much larger pool of sessions with the same pairs, which 2023-25 does not have. "
            "A case study that pools one long-running pair over all its sessions is the "
            "nearest thing to the gate's question that the data can carry."),
        ]
    elif dec["go"]:
        out += [
            ("**Go.** At least one choice reaches the bar. Before B1, the user chooses (decision "
            "#18): 2 or 3 phases; raw or profile statistic; blind behind a pre-registration "
            "(recommended) or open; whether sprint qualifying and 2026 are in; whether "
            "`layout_matches` stays. The pass would still need section 4's caveat: it would "
            "show teammate-label effects, not driver skill."),
        ]
    else:
        out += ["**Undecided.** The rule cannot be read; see the verdict."]
    out += [
        "",
        "## 7. Caveats",
        "",
        ("- **The noise is synthetic.** It is the injection-recovery budget of four circuits, "
        "calibrated to the real feed on two statistics per session, and scaled by 1.36 because "
        "the real placebo spread was wider. Real sessions at other circuits may be noisier or "
        "quieter; each simulated session borrows one of the four."),
        ("- **The effect model is simple.** tau is the same for every corner type and phase, "
        "independent between cells, and constant across circuits and seasons. A driver whose "
        "effect is concentrated in one cell, or that changes with the car, is not modelled."),
        ("- **Two phases are approximated** by merging recorded phases; a real 2-phase cut at "
        "the minimum-speed point sits between the two merges, which bracket it."),
        ("- **The gap model rests on 27 pilot lap deltas**, 9 pairs of them repeated. Its sds are "
        "rough; they matter only for the raw statistic."),
        ("- **Yield and lap counts** are the pilot's (four 2025 sessions). Other seasons, and "
        "sprint qualifying above all, may give fewer clean laps."),
        ("- **The calibration has nine pairs.** tau_80 is an 80% bound from a small design; "
        "the true effect could be larger. The verdict says the gate is unlikely to pass, not "
        "that it cannot."),
        "",
        "## 8. Deviations from the plan",
        "",
        ("- The report is written by a second script, `scripts/drivers_power_report.py`, as "
        "`drivers_report.py` does for the pilot; the plan named only `drivers_power.py`."),
        ("- The profile statistic is also added to the library "
        "(`deltas.split_half_across(..., profile=True)`, with tests), so that a pre-registration "
        "could use it as is. The simulation implements the same statistic on its own arrays."),
        ("- **Post hoc, added after seeing section 4:** the pilot's exact design at tau = 0 "
        "with the persistent errors on (`pilot_persistent`), to ask whether the pilot would "
        "have seen them. It does not enter the decision rule."),
    ]
    return out


def write_report(res: dict) -> None:
    taus = np.array(res["taus_ms"])
    cal = res["calibration"]
    dec = res["decision"]
    inp = res["inputs"]
    designs = res["sweep"]["designs"]
    _figure(res, DOCS / "drivers-power.png")
    L = []
    w_ = L.append
    w_("# Drivers DV-B0: power of the split-half gate (EXPLORATORY)")
    w_("")
    w_("*Follow-through item DV-B0. The plan, the model and the go/no-go rule were committed "
       "before the first run: [drivers-power-plan.md](drivers-power-plan.md). This is a "
       "simulation from the noise budgets recorded in [drivers-pilot.md](drivers-pilot.md). "
       "It reads no laps, positions or timing from any session outside the four pilot "
       "sessions; for the season designs it reads team rosters only.*")
    w_("")
    w_("Code: `scripts/drivers_power.py` (simulation) and `scripts/drivers_power_report.py` "
       "(this page), from the cached `drivers_pilot.py` stages. Output: "
       f"`outputs/drivers-lite/power/power.json`. {res['n_sims']:,} simulated seasons per grid "
       f"point, seed {res['seed']}; {res['cpu_s'] / 60:.0f} CPU-min on the M3 Pro.")
    w_("")

    # ------------------------------------------------------------------ verdict
    t80 = dec["tau_80_ms"]
    w_("## Verdict")
    w_("")
    if dec["go"] is None:
        w_("**Undecided by the rule.** The pilot does not bound tau below 40 ms, the top of the "
           "sweep, so P(pass) at tau_80 cannot be read.")
    else:
        best = max(dec["checks"], key=lambda c: max(c["p3"], min(c["p2a"], c["p2b"])))
        w_(f"**{'GO' if dec['go'] else 'NO-GO'} for DV-B.** The rule: with the profile "
           f"statistic, noise x1.36 and no persistent errors, P(rho >= {GATE:g}) must reach "
           f"0.5 at the pilot's 80% upper bound on tau, tau_80 = {t80:.1f} ms, for 3 phases "
           "or for both 2-phase merges, in some scope. P(pass) there:")
        w_("")
        w_("| Scope | 3 phases | 2 phases (entry + apex / exit) | 2 phases (entry / apex + exit) "
           "| Reaches 0.5 |")
        w_("|---|---|---|---|---|")
        for c in dec["checks"]:
            w_(f"| {SCOPES[c['scope']]} | {_fmt_p(c['p3'])} | {_fmt_p(c['p2a'])} | "
               f"{_fmt_p(c['p2b'])} | {'yes' if c['pass3'] or c['pass2'] else 'no'} |")
        w_("")
        w_(f"The best case is {SCOPES[best['scope']]}.")
    w_("")

    # ------------------------------------------------------------------ inputs
    w_("## 1. What the simulation is built from")
    w_("")
    w_("The model and its parameters are fixed in the plan. The numbers it took from disk:")
    w_("")
    vf = np.array(inp["var_factors"])
    w_(f"- **Yield.** {inp['pilot_pairs']} pair deltas from {inp['car_sessions']} pilot "
       f"car-sessions ({100 * inp['yield']:.1f}%): a car-session gives a teammate delta only "
       "if both drivers have clean laps in a shared Q segment.")
    w_(f"- **Lap counts.** The pilot pairs' variance factors (1 = 2 laps per driver in one "
       f"segment) have median {np.median(vf):.2f} and range {vf.min():.2f} to {vf.max():.2f}; "
       f"{int(np.sum(vf >= 2))} of {len(vf)} pairs compare one lap with one lap.")
    names = list(inp["noise_sd_ms"])
    med = np.median(np.array([inp["noise_sd_ms"][n] for n in names]), axis=0)
    cells = [f"{t} {p}" for t in TYPES for p in PHASES]
    w_("- **Noise per car-session** (recorded injection-recovery runs, lap sum per corner "
       "type x phase at 2 laps per driver; median over the four circuits of the sd, ms): "
       + ", ".join(f"{c} {m:.0f}" for c, m in zip(cells, med, strict=True)) + ".")
    g1, g2 = res["gap"]["1.0"], res["gap"]["1.36"]
    w_(f"- **Overall gap.** From the pilot's {g1['pair_sessions']} lap deltas "
       f"({g1['pairs_repeated']} pairs seen in more than one session): rms lap delta "
       f"{1e3 * np.sqrt(g1['total_ms2']):.0f} ms, within-pair spread between sessions "
       f"{1e3 * np.sqrt(g1['within_ms2']):.0f} ms. After the lap-delta noise, the constant gap g "
       f"has sd {1e3 * g1['sd_g']:.0f} ms and the per-session part eta "
       f"{1e3 * g1['sd_eta']:.0f} ms at noise x1 ({1e3 * g2['sd_g']:.0f} and "
       f"{1e3 * g2['sd_eta']:.0f} ms at x1.36). Most of a teammate gap therefore changes from "
       "session to session, which is noise for the raw statistic.")
    w_("")
    w_("| Scope | Sessions | Rounds | Teammate pairs | Pair-sessions | Pairs in both halves (mean) "
       "| Usable sessions per pair, median | Usable sessions, two longest pairs (mean) |")
    w_("|---|---|---|---|---|---|---|---|")
    for scope, d in designs.items():
        w_(f"| {d['label']} | {d['sessions']} | {d['meetings']} | {d['pairs']} | "
           f"{d['pair_sessions']} | {d['mean_pairs_in_both_halves']:.1f} | "
           f"{d['median_usable_sessions_per_pair']:.0f} | "
           f"{d['mean_usable_sessions_top2_pairs']:.1f} |")
    w_("")
    w_("*Pair-sessions* are every car-session in the rosters; each yields a delta with the "
       "pilot's probability, so *usable sessions* are after that. Sprint qualifying shares "
       "its round, and so its half, with the round's qualifying.")
    w_("")

    # ------------------------------------------------------------------ calibration
    w_("## 2. How large tau could be: the pilot's own split-half")
    w_("")
    obs = cal["observed"]
    w_(f"The pilot's across-session split-half ({cal['pairs_in_both_halves']} pairs in both "
       f"halves) gives rho = {obs['raw']:+.2f} on the raw deltas (as in the pilot report) and "
       f"{obs['profile']:+.2f} on the profile. The same model on the pilot's exact design "
       "turns these into bounds on tau (3 phases, no persistent errors, so anything "
       "persistent in the pilot counts as signal):")
    w_("")
    w_("| Statistic | Noise | tau_hat (ms) | tau_80 (ms) |")
    w_("|---|---|---|---|")
    for stat in ("profile", "raw"):
        for k in KS:
            c = cal["by_k"][str(k)][stat]
            w_(f"| {stat} | x{k:g} | {_fmt_tau(c['tau_hat_ms'])} | {_fmt_tau(c['tau_80_ms'])} |")
    w_("")
    w_("tau_hat is where the simulated pilot's median rho equals the observed one; tau_80 is "
       "where the observed rho falls at the 20th percentile, an 80% upper bound. With nine "
       "pairs and two sessions per half the pilot bounds tau only loosely.")
    w_("")

    # ------------------------------------------------------------------ power
    w_("## 3. Power")
    w_("")
    w_("![power](drivers-power.png)")
    w_("")
    w_("*Solid vertical line: the pilot's tau_80; dashed: tau_hat (noise x1.36, the panel's "
       "statistic). 2-phase panels show the lower of the two merges. No persistent errors.*")
    w_("")
    w_("tau at which P(pass) reaches 0.5 and 0.8 (ms, interpolated on the 5 ms grid; no "
       "persistent errors):")
    w_("")
    w_("| Scope | Phases | Statistic | x1: 0.5 | x1: 0.8 | x1.36: 0.5 | x1.36: 0.8 |")
    w_("|---|---|---|---|---|---|---|")
    for scope in SCOPES:
        for scheme in SCHEMES:
            for stat in ("profile", "raw"):
                v = []
                for k in KS:
                    pp = _pp(res, scope, scheme, stat, k)
                    v += [_first_cross(taus, pp, 0.5), _first_cross(taus, pp, 0.8)]
                w_(f"| {SCOPES[scope]} | {SCHEME_NAME[scheme]} | {stat} | "
                   + " | ".join(_fmt_tau(x) for x in v) + " |")
    w_("")
    w_("P(pass) on the 35 qualifying sessions, noise x1.36, no persistent errors, by tau (ms):")
    w_("")
    w_("| Phases | Statistic | " + " | ".join(f"{t:.0f}" for t in taus) + " |")
    w_("|---|---|" + "---|" * len(taus))
    for scheme in SCHEMES:
        for stat in ("profile", "raw"):
            w_(f"| {SCHEME_NAME[scheme]} | {stat} | "
               + " | ".join(f"{p:.2f}" for p in _pp(res, "q", scheme, stat, 1.36)) + " |")
    w_("")

    # ------------------------------------------------------------------ persistent
    w_("## 4. False passes from persistent per-car errors")
    w_("")
    w_("With the pilot's persistent-error scenario (a line pattern and a 1 m position offset "
       "per car) held constant across a pair's sessions, the gate can pass with no driver "
       "effect at all. P(pass) at tau = 0, noise x1.36:")
    w_("")
    w_("| Scope | Phases | Raw | Profile |")
    w_("|---|---|---|---|")
    for scope in SCOPES:
        for scheme in SCHEMES:
            w_(f"| {SCOPES[scope]} | {SCHEME_NAME[scheme]} | "
               f"{_pp(res, scope, scheme, 'raw', 1.36, True)[0]:.2f} | "
               f"{_pp(res, scope, scheme, 'profile', 1.36, True)[0]:.2f} |")
    w_("")
    pc = res["calibration"]["persistent_tau0"]
    w_("**Would the pilot have seen them?** (post hoc.) The same scenario on the pilot's exact "
       "design at tau = 0 gives a split-half rho with median "
       + "; ".join(f"{pc[str(k)]['profile']['median']:+.2f} (profile) and "
                   f"{pc[str(k)]['raw']['median']:+.2f} (raw) at noise x{k:g}" for k in KS)
       + f". The pilot observed {cal['observed']['profile']:+.2f} and "
       f"{cal['observed']['raw']:+.2f}; a value that low comes up in "
       + "; ".join(f"{100 * pc[str(k)]['profile']['p_at_or_below_observed']:.1f}% (profile) and "
                   f"{100 * pc[str(k)]['raw']['p_at_or_below_observed']:.1f}% (raw) of "
                   f"simulated pilots at x{k:g}" for k in KS)
       + ". So errors of the scenario's size most likely do not repeat across sessions. The "
       "table above is a worst case; the pilot's bound on tau already covers anything "
       "persistent, driver or car.")
    w_("")
    L += _reading(res)
    (DOCS / "drivers-power.md").write_text("\n".join(L) + "\n")
    print(f"wrote {DOCS / 'drivers-power.md'}")
