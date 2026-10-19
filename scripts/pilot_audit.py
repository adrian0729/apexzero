"""Pilot audit (plan phase 11) on the DR pilot's checkpoints, with PROXY labels.

    nice -n 15 uv run python scripts/pilot_audit.py [--run runs/dr-pilot-200M] [--levels 150]
                                                    [--windows 32] [--report-only]

Exercises the E1 machinery end to end before TO-OCP labels exist: rollouts of every
checkpoint on the first ``--levels`` A_DR levels (root key PRNGKey(2026); grammar fitted on
R-dev; the lockbox circuits are never touched), window statistics (mode F, one random chunk
phase per window), level scores, and the fidelity, reliability and phase metrics of
``apexzero.audit.fidelity``. The regret target uses the proxy optimum J_best (best mean
return over the audited checkpoints), so every number here grades the machinery, not the
scores. When J*_20 labels arrive, pass them as ``j_opt`` to ``regret_targets``.

Window statistics are cached per checkpoint under outputs/pilot-audit/, so re-running after
training writes more checkpoints only rolls out the new ones (``--report-only`` rolls out
nothing). Writes docs/findings/pilot-audit.md and docs/findings/pilot-audit-*.png.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from apexzero.agents.ppo import PPOConfig
from apexzero.audit import fidelity as F
from apexzero.audit.estimators import WindowStats, level_scores, window_stats
from apexzero.audit.rollouts import rollout_windows, snapshot_from_pickle
from apexzero.env import racing
from apexzero.oracle.qss import speed_profile_jax
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import DS, fit_grammar, generate
from apexzero.tracks.levels import build, grammar_for, make_levels
from apexzero.tracks.track import load_tumftm

ROOT_SEED = 2026
HEADLINE = ["pvl", "l1", "maxmc", "maxmc_unit", "learnability_s1", "learnability_s2", "mna",
            "disagreement"]
BASELINES = ["agent_only", "j_best", "qss_slow", "random"]
REFERENCE = "proxy_regret"           # proxy regret on the score windows ("oracle-operational")
LABELS = {
    "pvl": "PVL", "l1": "L1 value loss", "maxmc": "MaxMC (as implemented)",
    "maxmc_unit": "MaxMC (unit-consistent)", "learnability_s1": "learnability, S1",
    "learnability_s2": "learnability, S2", "mna": "MNA", "disagreement": "disagreement (5 heads)",
    "agent_only": "agent-only, -J^pi_score", "j_best": "J_best, score windows",
    "qss_slow": "geometric: -d_QSS(60 s)", "random": "random",
    "proxy_regret": "proxy regret, score windows (reference)",
}
# Reference categorical palette (dataviz skill, light surface), fixed order per score.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STATS_FIELDS = WindowStats._fields


# --------------------------------------------------------------------------- levels


def pilot_levels(n: int):
    """The first ``n`` A_DR levels, their strips and start states, and d_QSS: the distance the
    QSS profile of the level's *own* car (context and hidden grip) covers in one window."""
    dev = [load_tumftm(f"data/tracks/tumftm/{name}.csv") for name in R_DEV]
    g = fit_grammar(dev)
    lv = make_levels(jax.random.PRNGKey(ROOT_SEED), "a_dr", n=n)
    tracks, _, x0 = build(lv, g)
    gg = grammar_for(g, lv.generator)

    def qss(key, ctx, mu):
        kappa = generate(key, gg)[0]
        return speed_profile_jax(kappa, DS, racing.params_from_context(ctx, hidden_mu=mu))

    v = np.asarray(jax.jit(jax.vmap(qss))(lv.keys, lv.ctx, lv.hidden_mu))
    p = racing.EnvParams()
    d_qss = F.distance_in_time(v, DS, p.window_steps * p.dt)
    return lv, tracks, x0, d_qss


# --------------------------------------------------------------------------- rollouts (cached)


def audit_checkpoint(path: Path, cache: Path, lv, tracks, x0, n_windows: int, phases: np.ndarray):
    """WindowStats for one checkpoint, from the cache when its settings match."""
    meta = {"levels": int(x0.shape[0]), "windows": n_windows, "seed": ROOT_SEED, "mode": "F"}
    out = cache / f"{path.stem}.npz"
    if out.exists():
        d = np.load(out)
        if json.loads(str(d["meta"])) == meta:
            return WindowStats(*(d[k] for k in STATS_FIELDS)), dict(
                sigma=float(d["sigma"]), timesteps=int(d["timesteps"]), wall=float(d["wall"]),
                cpu=float(d["cpu"]))
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    snap = snapshot_from_pickle(ck)
    cfg = PPOConfig(**ck["config"])
    t0, c0 = time.perf_counter(), time.process_time()
    key = jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), 1)     # same draws for every ckpt
    ro = rollout_windows(snap, cfg, tracks, x0, lv.ctx, lv.hidden_mu, n_windows, key)
    st = window_stats(ro, phases=phases)                             # mode F, chunked
    info = dict(sigma=float(snap.sigma), timesteps=int(ck["timesteps"]),
                wall=time.perf_counter() - t0, cpu=time.process_time() - c0)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **st._asdict(), **info)
    return st, info


# --------------------------------------------------------------------------- analysis


def scores_for(st: WindowStats, windows, d_qss, g_all, c, rng_seed):
    """Every graded score from ``windows``: the estimators plus the baselines, all computed
    from the score windows only (so none shares noise with the target windows)."""
    s = level_scores(st, windows, qss_distance=d_qss)
    best = F.regret_targets(g_all, windows)                    # J_best and J^pi on score windows
    s["j_best"] = best.j_opt
    s["qss_slow"] = -d_qss
    s["random"] = np.random.default_rng(rng_seed).normal(size=d_qss.shape)
    s[REFERENCE] = best.reg[c]
    keep = HEADLINE + BASELINES + [REFERENCE]
    return {k: s[k] for k in keep if k in s}


def analyse(stats, infos, d_qss, n_boot):
    W = stats[0].G_disc.shape[-1]
    plan = F.window_plan(W)
    g_all = np.stack([st.G_disc for st in stats])               # [C, L, W]
    t_hb = F.regret_targets(g_all, plan.hb_target)
    t_op = F.regret_targets(g_all, plan.op_target)
    half = W // 2
    sizes = [m for m in (1, 2, 4, 8, 16, 32) if m <= half]
    res = []
    for c, (st, info) in enumerate(zip(stats, infos)):
        comp = F.completion_rate(st.completed)
        sc_hb = scores_for(st, plan.hb_score, d_qss, g_all, c, rng_seed=c)
        sc_op = scores_for(st, plan.operational, d_qss, g_all, c, rng_seed=c)
        fid_hb = F.fidelity_table(sc_hb, t_hb.j_opt, t_hb.j_pi[c], t_hb.reg[c], t_hb.reg_norm[c],
                                  n_boot=n_boot, seed=c, reference="agent_only")
        fid_op = F.fidelity_table(sc_op, t_op.j_opt, t_op.j_pi[c], t_op.reg[c], t_op.reg_norm[c],
                                  n_boot=n_boot, seed=c, reference="agent_only")

        def score_fn(w, st=st, c=c):
            return scores_for(st, w, d_qss, g_all, c, rng_seed=c)

        def target_fn(w, c=c):
            t = F.regret_targets(g_all, w)
            return {"reg": t.reg[c], "reg_norm": t.reg_norm[c], "j_pi": t.j_pi[c]}

        rel_s = F.split_half(score_fn, range(W))
        rel_t = F.split_half(target_fn, range(W))
        curve = F.reliability_curve(lambda w, sf=score_fn, tf=target_fn: {**sf(w), **tf(w)},
                                    range(W), sizes, n_splits=10, seed=c)
        r12 = F.spearman(t_hb.j_opt, t_hb.j_pi[c])
        res.append(dict(
            info=info, completion=comp, phase=F.phase_of(comp),
            j_pi=float(t_hb.j_pi[c].mean()), reg_mean=float(t_hb.reg[c].mean()),
            reg_median=float(np.median(t_hb.reg[c])), attained=float(t_hb.attained[c].mean()),
            r12=float(r12), fid_hb=fid_hb, fid_op=fid_op, rel_s=rel_s, rel_t=rel_t, curve=curve,
            sizes=sizes, calib=F.value_calibration(st.V_first, st.G_disc),
            s2_rate=float(np.mean(st.completed & (st.distance >= 0.97 * d_qss[:, None]))),
        ))
    return plan, res


# --------------------------------------------------------------------------- figures


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _log_steps(ax, x):
    """Log axis of training steps in millions, labelled with plain round numbers."""
    ax.set_xscale("log")
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000) if x.min() * 0.9 <= t <= x.max() * 1.1]
    ax.set_xticks(ticks, [f"{t}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def fig_fidelity(res, path: Path):
    """Small multiples: one panel per headline score, rho against absolute and normalised
    regret with 95% bootstrap bands, the agent-only baseline dashed for comparison."""
    x = np.array([r["info"]["timesteps"] / 1e6 for r in res])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes.ravel(), HEADLINE):
        _style(ax)
        for metric, color, label in (("rho_abs", PALETTE[0], "vs Reg"),
                                     ("rho_norm", PALETTE[1], "vs Reg / J_opt")):
            est = [r["fid_hb"].rows.get(name, {}).get(metric) for r in res]
            y = np.array([e.value if e else np.nan for e in est])
            lo = np.array([e.lo if e else np.nan for e in est])
            hi = np.array([e.hi if e else np.nan for e in est])
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(x, y, "-o", color=color, lw=1.5, ms=4, label=label)
        base = [r["fid_hb"].rows["agent_only"]["rho_abs"].value for r in res]
        ax.plot(x, base, "--", color=INK2, lw=1.2, label="agent-only, vs Reg")
        ax.axhline(0, color=INK2, lw=0.8)
        _log_steps(ax, x)
        ax.set_ylim(-1, 1)
        ax.set_title(LABELS[name], fontsize=9, color=INK)
    for ax in axes[1]:
        ax.set_xlabel("training steps (M)", fontsize=8, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Spearman rho", fontsize=8, color=INK2)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("PROXY-LABEL PILOT: high-budget fidelity (odd-window scores vs even-window "
                 "regret against J_best), 95% bootstrap bands over levels", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_reliability(res, path: Path, W: int):
    """Left: split-half reliability (score on half the windows) across training. Right: the
    empirical reliability curve against window count at the last checkpoint, with the
    Spearman-Brown prophecy from the split-half value and the 0.9 goal."""
    x = np.array([r["info"]["timesteps"] / 1e6 for r in res])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for a in ax:
        _style(a)
    for i, name in enumerate(HEADLINE):
        ax[0].plot(x, [r["rel_s"].get(name, np.nan) for r in res], "-o", color=PALETTE[i],
                   lw=1.5, ms=4, label=LABELS[name])
    ax[0].plot(x, [r["rel_t"]["reg"] for r in res], "-s", color=INK, lw=2, ms=5,
               label="target: regret")
    ax[0].axhline(0.9, color=INK2, lw=0.8, ls=":")
    _log_steps(ax[0], x)
    ax[0].set_ylim(-0.2, 1)
    ax[0].set_xlabel("training steps (M)", fontsize=8, color=INK2)
    ax[0].set_ylabel(f"split-half reliability ({W // 2} windows)", fontsize=8, color=INK2)
    ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    # The newest checkpoint's proxy labels are degenerate (it attains J_best on most levels),
    # so the curve is shown for the one before it.
    last = res[-2] if len(res) > 2 else res[-1]
    sizes = np.array(last["sizes"])
    grid = np.geomspace(1, 128, 100)
    shown = HEADLINE + ["reg"]
    for i, name in enumerate(shown):
        color = INK if name == "reg" else PALETTE[i]
        emp = last["curve"].get(name)
        if emp is None or not np.isfinite(emp).any():
            continue
        r_half = last["rel_t"]["reg"] if name == "reg" else last["rel_s"].get(name, np.nan)
        ax[1].plot(sizes, emp, "o", color=color, ms=5)
        if np.isfinite(r_half) and r_half > 0:
            ax[1].plot(grid, F.spearman_brown(r_half, grid / (W // 2)), "-", color=color, lw=1.2)
    ax[1].axhline(0.9, color=INK2, lw=0.8, ls=":")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_ylim(-0.2, 1)
    ax[1].set_xlabel("windows per level", fontsize=8, color=INK2)
    ax[1].set_ylabel("reliability", fontsize=8, color=INK2)
    ax[1].set_title(f"{last['info']['timesteps'] / 1e6:.0f}M checkpoint: empirical (dots) vs "
                    "Spearman-Brown from the split half (lines)", fontsize=9, color=INK)
    fig.suptitle("PROXY-LABEL PILOT: reliability of scores and target", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- report


def _ci(e: F.Estimate | None, digits: int = 2, signed: bool = True) -> str:
    if e is None or not np.isfinite(e.value):
        return "-"
    f = f"{'+' if signed else ''}.{digits}f"
    if not np.isfinite(e.lo):
        return f"{e.value:{f}}"
    return f"{e.value:{f}} [{e.lo:{f}}, {e.hi:{f}}]"


def _num(v, fmt="{:.2f}") -> str:
    return "-" if v is None or not np.isfinite(v) else fmt.format(v)


def _windows(v) -> str:
    if v is None or not np.isfinite(v):
        return "never" if v is not None and np.isinf(v) else "-"
    return f"{int(np.ceil(v))}"


def report(args, plan, res, d_qss, out: Path, skipped_ckpts):
    L = len(d_qss)
    W = len(plan.hb_score) + len(plan.hb_target)
    half = W // 2
    wall = sum(r["info"]["wall"] for r in res)
    cpu = sum(r["info"]["cpu"] for r in res)
    lines = [
        "# Pilot audit: level-score fidelity on the DR pilot's early checkpoints",
        "",
        "> **PROXY-LABEL PILOT.** Regret here is measured against **J_best**, the best mean",
        "> return any audited checkpoint reached on the level, not against the TO-OCP optimum",
        "> J\\*_20. These numbers test the E1 machinery (plan phase 11) and must not be read as",
        "> findings about the scores. The audit is ready for real labels: pass J\\*_20 as",
        "> `j_opt` to `apexzero.audit.fidelity.regret_targets` and re-run.",
        "",
        f"Generated by `scripts/pilot_audit.py` from `{args.run}`.",
        "",
        "## Setup",
        "",
        f"- **Checkpoints:** {len(res)} DR-pilot checkpoints, "
        + ", ".join(f"{r['info']['timesteps'] / 1e6:.1f}M" for r in res)
        + " (single seed; training was still running, so these are early checkpoints only)."
        + (f" Skipped (unreadable, probably mid-write): {', '.join(skipped_ckpts)}." if skipped_ckpts else ""),
        (f"- **Levels:** the first {L} A_DR levels (`make_levels(PRNGKey({ROOT_SEED}), 'a_dr')`),"
        " grammar fitted on R-dev only. No lockbox (R-test) circuit is used anywhere."),
        (f"- **Windows:** {W} stochastic 1,200-step windows per level and checkpoint, with 1%"
        " observation noise; one chunk phase per window drawn from U{0..127} (seed"
        f" {ROOT_SEED}); mode F scoring (window end terminal), λ = 0.95 headline. The same"
        " random draws are reused at every checkpoint (common random numbers)."),
        (f"- **Window roles (plan 2.2):** high-budget scores use the odd windows (1, 3, ...,"
        f" {W - 1}), the target the even windows; operational scores use windows 1-4 and their"
        f" target windows 5-{W}."),
        ("- **Target:** J^π = mean discounted return over the target windows; J_best ="
        " max over checkpoints of that mean (from the target windows only, so scores and labels"
        " share no window noise); Reg = J_best − J^π, Reg_norm = Reg / J_best. J_best is a"
        " certified lower bound on J\\*: every realised window is a feasible control sequence"
        " of the deterministic level."),
        ("- **Phase:** window completion rate over all windows of the audited levels"
        " (<50% early, 50-90% mid, ≥90% late)."),
        (f"- **Statistics:** percentile bootstrap over levels, {args.boot} resamples shared by all"
        " scores (Δ columns are paired against the agent-only baseline). One seed, so no"
        " seed-level variance is included: CIs are too narrow for the plan's two-way bootstrap."),
        (f"- **Compute:** rollouts + window statistics {wall / 60:.1f} min wall, {cpu / 60:.1f} min"
        " CPU in total (nice 15, on a machine shared with the DR run)."),
        "",
        ("Baselines: *agent-only* = −J^π on the score windows; *J_best, score windows* = the proxy"
        " optimum recomputed from the score windows (policy-agnostic, in the role J\\*_20 will"
        " play); *geometric* = −d_QSS, the distance the QSS profile of the level's own car covers"
        " in 60 s (shorter = slower, more technical strip); *random*. The *proxy regret, score"
        " windows* row is the oracle-operational reference: the regret itself measured on the"
        " score windows. It is a reference, not a ceiling."),
        "",
        "## Units check and phase",
        "",
        ("Critic values must be in raw reward units (V = σ·V̂, spec DEV-6; `rollout_windows`"
        " multiplies by σ from the checkpoint's `ret_rms`). A calibrated critic then has"
        " mean V(o₀) ≈ mean G^γ (the *ratio* column is the units check); values left"
        " normalised would be off by the factor σ. *slope* and *r* (G on V across levels)"
        " measure something else: how well V(o₀) tells levels apart, which is limited because"
        " o₀ previews only the first few hundred metres of a ~3.5 km window."),
        "",
        ("| Checkpoint | σ | mean V(o₀) | mean G^γ | ratio | slope | r | completion | phase"
        " | S2 rate | mean J^π | mean Reg | median Reg | attains J_best | corr(J_best, J^π) |"),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in res:
        k = r["calib"]
        lines.append(
            f"| {r['info']['timesteps'] / 1e6:.1f}M | {r['info']['sigma']:.2f} | {k['mean_v']:.1f}"
            f" | {k['mean_g']:.1f} | {k['ratio']:.3f} | {k['slope']:.2f} | {k['r']:.2f}"
            f" | {r['completion'] * 100:.1f}% | {r['phase']} | {r['s2_rate'] * 100:.1f}%"
            f" | {r['j_pi']:.1f} | {r['reg_mean']:.2f} | {r['reg_median']:.2f}"
            f" | {r['attained'] * 100:.0f}% | {r['r12']:+.2f} |")
    lines += [
        "",
        ("*attains J_best*: share of levels on which this checkpoint *is* the proxy optimum, so"
        " its regret is exactly 0 there. *corr(J_best, J^π)*: rank correlation of the two"
        " loadings of the two-loading regression; near ±1 they are unidentified."),
        "",
        "## Fidelity across training (high budget)",
        "",
        "Spearman ρ of each score (odd windows) against absolute regret (even windows), 95% CI.",
        "",
        "| Score | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M ({r['phase']})" for r in res) + " |",
        "|---|" + "---|" * len(res),
    ]
    for name in HEADLINE + BASELINES + [REFERENCE]:
        cells = [_ci(r["fid_hb"].rows.get(name, {}).get("rho_abs")) for r in res]
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
    lines += ["", "![fidelity](pilot-audit-fidelity.png)", "",
              "## Operational scores (windows 1-4 against windows 5+)", "",
              ("Spearman ρ against absolute regret, and RC@10% (share of total regret in the"
              " score's top 10% of levels; random = 0.10)."), "",
              "| Score | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M ρ / RC" for r in res) + " |",
              "|---|" + "---|" * len(res)]
    for name in HEADLINE + BASELINES + [REFERENCE]:
        cells = []
        for r in res:
            row = r["fid_op"].rows.get(name)
            cells.append("-" if row is None else
                         f"{_num(row['rho_abs'].value, '{:+.2f}')} / {_num(row['rc'].value)}")
        lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Reliability and power",
        "",
        (f"Split-half reliability: each score computed on the odd and on the even windows"
        f" ({half} each), Spearman correlation across levels. That is the reliability of the"
        f" high-budget score as used ({half} windows); Spearman-Brown steps it to other budgets."
        " *Windows for 0.9* is the Spearman-Brown prophecy (the G1 gate asks for ≥ 0.9 late and"
        " ≥ 0.8 mid within 64 windows). *Operational* is the prophecy at 4 windows."),
        "",
        "| Quantity | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M r / for 0.9 / op." for r in res) + " |",
        "|---|" + "---|" * len(res),
    ]
    for name in HEADLINE + ["agent_only", "reg", "reg_norm", "j_pi"]:
        cells = []
        for r in res:
            v = r["rel_t"].get(name) if name in ("reg", "reg_norm", "j_pi") else r["rel_s"].get(name)
            ok = v is not None and np.isfinite(v)
            n = float(F.windows_for_reliability(v, half)) if ok else None
            op = float(F.spearman_brown(v, 4 / half)) if ok else None
            cells.append(f"{_num(v)} / {_windows(n)} / {_num(op)}")
        label = {"reg": "**target: Reg**", "reg_norm": "**target: Reg/J_best**",
                 "j_pi": "**target: J^π**"}.get(name, LABELS.get(name, name))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    mde = {n: F.spearman_mde(n) for n in (L, 1000, 1412)}
    lines += [
        "",
        "![reliability](pilot-audit-reliability.png)",
        "",
        "**Level power.** Smallest Spearman ρ distinguishable from 0 at α = 0.05 with 80% power"
        " (Fisher z, variance 1.06/(n−3)): "
        + ", ".join(f"{mde[n]:.3f} with {n} levels" for n in mde)
        + ". A difference between two independent ρs needs √2 more in z; scores graded on the"
        " same levels are paired, so the bootstrap Δ columns are tighter than that.",
        "",
        ("Caveats on the reliability numbers: Spearman-Brown assumes the score averages"
        " exchangeable windows. MaxMC (and the max-return term of MaxMC-unit) take a maximum"
        " over windows, so their prophecy is approximate; the right panel of the figure"
        " compares the prophecy with the empirical curve. A reliability near 0 or negative"
        " (constant or near-constant scores, e.g. learnability when every window completes)"
        " cannot be extrapolated (\"never\")."),
        "",
        "## Per-checkpoint tables (high budget)",
        "",
        ("ρ = Spearman against Reg and against Reg/J_best; RC = RC@10% (share of total regret in"
        " the score's top 10%) and RC/oracle (divided by the share of the regret-sorted top 10%);"
        " P@10% = overlap of the top 10% sets; a, b = standardised loadings of rank(S) on"
        " rank(J_best) and rank(J^π) (faithful: a > 0, b < 0); Δρ = ρ minus the agent-only"
        " baseline's ρ, paired; ρ* = ρ disattenuated by the split-half reliabilities of score"
        " and target (it can exceed 1 in magnitude when those reliabilities are small, since"
        " both are noisy estimates)."),
    ]
    for r in res:
        tab = r["fid_hb"]
        lines += [
            "",
            f"### {r['info']['timesteps'] / 1e6:.1f}M steps: completion {r['completion'] * 100:.1f}%"
            f" ({r['phase']}), {tab.n_levels} levels graded"
            + (f", {tab.n_dropped} dropped (J_best ≤ 1)" if tab.n_dropped else ""),
            "",
            "| Score | ρ vs Reg | ρ vs Reg/J_best | RC@10% | RC/oracle | P@10% | a | b | Δρ vs agent-only | ρ* |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in HEADLINE + BASELINES + [REFERENCE]:
            row = tab.rows.get(name)
            if row is None:
                lines.append(f"| {LABELS[name]} | " + " | ".join(["-"] * 9) + " |")
                continue
            rel = r["rel_s"].get(name, np.nan) if name != REFERENCE else np.nan
            rho_star = F.disattenuate(row["rho_abs"].value, rel, r["rel_t"]["reg"])
            lines.append(
                f"| {LABELS[name]} | {_ci(row['rho_abs'])} | {_ci(row['rho_norm'])} | {_ci(row['rc'], signed=False)}"
                f" | {_num(row['rc_rel'].value)} | {_num(row['p_at_q'].value)}"
                f" | {_num(row['a_opt'].value, '{:+.2f}')} | {_num(row['b_pi'].value, '{:+.2f}')}"
                f" | {_ci(row.get('delta_rho_abs'))} | {_num(float(rho_star), '{:+.2f}')} |")
        if tab.skipped:
            lines.append(f"\nSkipped (non-finite on graded levels): {', '.join(tab.skipped)}.")

    lines += [
        "",
        "## What the proxy can and cannot say",
        "",
        ("- **Degenerate at the newest checkpoint.** J_best is attained by the best checkpoint"
        " on each level, so that checkpoint's regret is exactly 0 there (see *attains J_best*)."
        " As training improves monotonically the last checkpoint's labels collapse toward 0 and"
        " its fidelity rows are uninformative; the earlier checkpoints are graded against what"
        " this run later reached, which is regret relative to the run, not to the optimum."),
        ("- **Collinear loadings.** J_best is close to the late J^π, so corr(J_best, J^π) is"
        " high and the two-loading coefficients are poorly identified; with J\\*_20 (a property"
        " of the level, not of the run) the regression becomes meaningful."),
        ("- **Lower bound, not the optimum.** Reg ≥ 0 by construction and understates true regret"
        " everywhere the run never got close to J\\*. Normalised regret divides by a"
        " run-dependent number."),
        ("- **Pilot scale.** One seed, early checkpoints only (no late phase yet unless the table"
        f" says so), {L} levels instead of 1,000 A_DR + 300 A_tail + 100 A_OOD, {W} windows"
        " without the adaptive raise to 64, no Holm correction, no seed-level bootstrap."),
        ("- **d_QSS** (geometric baseline and the S2 success threshold) integrates the level"
        " car's QSS profile from s = 0 and ignores the 0.8·v_QSS start, a deficit of a few tens"
        " of metres in ~3-4 km."),
        "",
        ("What carries over unchanged to the real audit: the window roles and cross-fitting, the"
        " metrics and their tie handling, the paired bootstrap, the split-half reliability,"
        " disattenuation and window-budget calculations, and the units check."),
    ]
    (out / "pilot-audit.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/dr-pilot-200M")
    ap.add_argument("--levels", type=int, default=150)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max-ckpts", type=int, default=None, help="audit at most this many (newest last)")
    ap.add_argument("--cache", default="outputs/pilot-audit")
    ap.add_argument("--out", default="docs/findings")
    ap.add_argument("--report-only", action="store_true", help="use cached checkpoints only")
    args = ap.parse_args()

    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    lv, tracks, x0, d_qss = pilot_levels(args.levels)
    print(f"levels ready ({time.perf_counter() - t0:.0f} s); median d_QSS {np.median(d_qss):.0f} m",
          flush=True)
    phases = np.random.default_rng(ROOT_SEED).integers(0, 128, (args.levels, args.windows))
    paths = sorted(Path(args.run, "ckpt").glob("*.pkl"))
    if args.max_ckpts:
        paths = paths[:args.max_ckpts]
    stats, infos, skipped = [], [], []
    for p in paths:
        if args.report_only and not (cache / f"{p.stem}.npz").exists():
            continue
        try:
            st, info = audit_checkpoint(p, cache, lv, tracks, x0, args.windows, phases)
        except (EOFError, pickle.UnpicklingError) as exc:
            skipped.append(f"{p.stem} ({type(exc).__name__})")
            continue
        stats.append(st)
        infos.append(info)
        print(f"{p.stem}: completion {st.completed.mean():.3f}, wall {info['wall']:.0f} s, "
              f"cpu {info['cpu']:.0f} s", flush=True)
    if len(stats) < 2:
        raise SystemExit("need at least two audited checkpoints")
    plan, res = analyse(stats, infos, d_qss, args.boot)
    fig_fidelity(res, out / "pilot-audit-fidelity.png")
    fig_reliability(res, out / "pilot-audit-reliability.png", args.windows)
    report(args, plan, res, d_qss, out, skipped)
    print(f"wrote {out / 'pilot-audit.md'} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
