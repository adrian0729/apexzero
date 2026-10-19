"""Pilot audit (plan phase 11) on the DR pilot's checkpoints, with PROXY labels.

    nice -n 15 uv run python scripts/pilot_audit.py [--run runs/dr-pilot-200M] [--levels 150]
                                                    [--windows 32] [--report-only]
                                                    [--sub-ckpts 000040471316,000060341763]

Exercises the E1 machinery end to end before TO-OCP labels exist: rollouts of every
checkpoint on the first ``--levels`` A_DR levels (root key PRNGKey(2026); grammar fitted on
R-dev; the lockbox circuits are never touched), window statistics (mode F, one random chunk
phase per window), level scores, and the fidelity, reliability and phase metrics of
``apexzero.audit.fidelity``. The regret target uses the proxy optimum J_best (best mean
return over the audited checkpoints), so every number here grades the machinery, not the
scores. When J*_20 labels arrive, pass them with ``--j-opt``.

A **64-window sub-pilot** (``--sub-levels`` levels, ``--sub-windows`` windows, the
checkpoints in ``--sub-ckpts``, independent random draws) observes the reliability curves
up to 32 windows per half, beyond the main pilot's 16. Its regret target uses a *fixed*
optimum per level (the main pilot's J_best over all its windows and checkpoints, or the
J*_20 labels), as the real audit will.

Window statistics and d_QSS are cached under outputs/pilot-audit/ (keyed by the
checkpoint file's hash and the statistics version), so re-running after training writes
more checkpoints only rolls out the new ones (``--report-only`` rolls out no new main-pilot
checkpoint; the sub-pilot rolls out only what it has not cached). Writes
docs/findings/pilot-audit.md and docs/findings/pilot-audit-*.png.
"""

from __future__ import annotations

import argparse
import hashlib
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
from scipy.stats import norm

from apexzero.agents.ppo import PPOConfig
from apexzero.audit import fidelity as F
from apexzero.audit import qss_distance as Q
from apexzero.audit.estimators import WindowStats, level_scores, window_stats
from apexzero.audit.rollouts import rollout_windows, snapshot_from_pickle
from apexzero.env import racing
from apexzero.oracle.qss import speed_profile_jax
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import DS, fit_grammar, generate
from apexzero.tracks.levels import build, grammar_for, make_levels
from apexzero.tracks.track import load_tumftm

ROOT_SEED = 2026
STATS_VERSION = 1      # bump when window_stats / rollout_windows change what they compute
DQSS_VERSION = "racing-line-v1"
MAX_TIE_SHARE = 0.8    # split-half r is not reported when > 80% of levels tie on one value
GRADABLE = 0.1         # proxy labels: a checkpoint is graded only if it attains J_best on < 10%
GOAL = 0.9
HEADLINE = ["pvl", "l1", "maxmc", "maxmc_unit", "learnability_s1", "learnability_s2", "mna",
            "disagreement"]
MAX_BASED = {"maxmc", "maxmc_unit"}  # a max over windows: Spearman-Brown is approximate
# Quantities that are (close to) plain means over windows, where Spearman-Brown is an identity
# for Pearson reliability. MaxMC (max), MNA (ratio of sums with a zero gate) and learnability
# (p(1 - p) of a mean) are not; the target Reg is a mean up to the per-half J_best.
MEAN_BASED = ["pvl", "l1", "disagreement", "agent_only", "reg"]
BASELINES = ["agent_only", "j_best", "qss_slow", "random"]
BEST_OF = ["agent_only", "j_best", "qss_slow"]   # plan 2.3: agent-only and policy-agnostic
REFERENCE = "proxy_regret"           # proxy regret on the score windows ("oracle-operational")
TARGETS = ["reg", "reg_norm", "j_pi"]
LABELS = {
    "pvl": "PVL", "l1": "L1 value loss", "maxmc": "MaxMC (as implemented)",
    "maxmc_unit": "MaxMC (unit-consistent)", "learnability_s1": "learnability, S1",
    "learnability_s2": "learnability, S2", "mna": "MNA", "disagreement": "disagreement (5 heads)",
    "agent_only": "agent-only, -J^pi_score", "j_best": "J_best, score windows",
    "qss_slow": "geometric: -d_QSS(60 s)", "random": "random",
    "proxy_regret": "proxy regret, score windows (reference)",
    "reg": "**target: Reg**", "reg_norm": "**target: Reg/J_best**", "j_pi": "**target: J^π**",
}
# Reference categorical palette (dataviz skill, light surface), fixed order per score.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STATS_FIELDS = WindowStats._fields


# --------------------------------------------------------------------------- levels


def pilot_levels(n: int, cache: Path | None = None):
    """The first ``n`` A_DR levels, their strips and start states, and two versions of
    d_QSS [n]: ``d_qss``, the QSS point mass of the level's own car on the minimum-curvature
    line from x0 (``audit.qss_distance``; the S2 reference and the geometric baseline), and
    ``d_qss_centre``, the pilot's first definition (QSS on the centreline curvature from
    v_QSS(0)), kept only for the diagnostic that replaced it."""
    dev = [load_tumftm(f"data/tracks/tumftm/{name}.csv") for name in R_DEV]
    g = fit_grammar(dev)
    lv = make_levels(jax.random.PRNGKey(ROOT_SEED), "a_dr", n=n)
    tracks, _, x0 = build(lv, g)
    gg = grammar_for(g, lv.generator)
    p = racing.EnvParams()
    seconds = p.window_steps * p.dt

    def qss_centre(key, ctx, mu):
        kappa = generate(key, gg)[0]
        return speed_profile_jax(kappa, DS, racing.params_from_context(ctx, hidden_mu=mu))

    v = np.asarray(jax.jit(jax.vmap(qss_centre))(lv.keys, lv.ctx, lv.hidden_mu))
    d_centre = F.distance_in_time(v, DS, seconds)
    d_qss = racing_d_qss(lv, gg, np.asarray(x0), cache)
    return lv, tracks, x0, d_qss, d_centre


def racing_d_qss(lv, gg, x0: np.ndarray, cache: Path | None = None) -> np.ndarray:
    """d_QSS on the racing line for every level (about 0.3 s per level), cached."""
    n = int(x0.shape[0])
    p = racing.EnvParams()
    meta = {"levels": n, "seed": ROOT_SEED, "version": DQSS_VERSION, "lam": Q.LAMBDA,
            "margin": p.edge_margin, "seconds": p.window_steps * p.dt}
    out = None if cache is None else cache / f"d_qss_{n}.npz"
    if out is not None and out.exists():
        d = np.load(out)
        if json.loads(str(d["meta"])) == meta:
            return d["d_qss"]
    kappa, wl, wr = (np.asarray(a, float) for a in jax.jit(jax.vmap(lambda k: generate(k, gg)))(lv.keys))
    ctx, mu = np.asarray(lv.ctx, float), np.asarray(lv.hidden_mu, float)
    d = np.array([Q.qss_distance(kappa[i], wl[i], wr[i],
                                 racing.params_from_context(ctx[i], hidden_mu=mu[i]),
                                 float(x0[i, 3]), DS, meta["seconds"], p.edge_margin)
                  for i in range(n)])
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, meta=json.dumps(meta), d_qss=d)
    return d


# --------------------------------------------------------------------------- rollouts (cached)


def checkpoint_id(path: Path) -> dict:
    data = path.read_bytes()
    return {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()[:16]}


def audit_checkpoint(path: Path, cache: Path, lv, tracks, x0, n_windows: int, phases: np.ndarray,
                     key_fold: int = 1):
    """WindowStats for one checkpoint, from the cache when its settings match.

    The cache key covers the levels, windows, seed, scoring mode, the rollout key, the
    statistics version and the checkpoint file's hash. Caches written before the hash was
    recorded (``legacy``) are accepted only for the main pilot's key, at statistics version
    1, and only when the checkpoint file is older than the cache file, so it cannot have
    been replaced since the statistics were computed."""
    base = {"levels": int(x0.shape[0]), "windows": n_windows, "seed": ROOT_SEED, "mode": "F"}
    meta = {**base, "key_fold": key_fold, "stats_version": STATS_VERSION, "ckpt": checkpoint_id(path)}
    out = cache / f"{path.stem}.npz"
    if out.exists():
        d = np.load(out)
        m = json.loads(str(d["meta"]))
        legacy = (m == base and key_fold == 1 and STATS_VERSION == 1
                  and path.stat().st_mtime < out.stat().st_mtime)
        if m == meta or legacy:
            return WindowStats(*(d[k] for k in STATS_FIELDS)), dict(
                sigma=float(d["sigma"]), timesteps=int(d["timesteps"]), wall=float(d["wall"]),
                cpu=float(d["cpu"]), legacy_cache=legacy, stem=path.stem)
    with open(path, "rb") as fh:
        ck = pickle.load(fh)
    snap = snapshot_from_pickle(ck)
    cfg = PPOConfig(**ck["config"])
    t0, c0 = time.perf_counter(), time.process_time()
    key = jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), key_fold)   # same draws per ckpt
    ro = rollout_windows(snap, cfg, tracks, x0, lv.ctx, lv.hidden_mu, n_windows, key)
    st = window_stats(ro, phases=phases)                                 # mode F, chunked
    info = dict(sigma=float(snap.sigma), timesteps=int(ck["timesteps"]),
                wall=time.perf_counter() - t0, cpu=time.process_time() - c0)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, meta=json.dumps(meta), **st._asdict(), **info)
    return st, {**info, "legacy_cache": False, "stem": path.stem}


# --------------------------------------------------------------------------- analysis


def scores_for(st: WindowStats, windows, d_qss, g_all, c, rng_seed, j_opt=None,
               floor_with_best=False):
    """Every graded score from ``windows``: the estimators plus the baselines, all computed
    from the score windows only (so none shares noise with the target windows). With real
    labels ``j_opt`` (J*_20), the policy-agnostic baseline is J*_20 itself and the reference
    row is the plan's oracle-operational score J*_20 - G on the score windows."""
    s = level_scores(st, windows, qss_distance=d_qss)
    best = F.regret_targets(g_all, windows, j_opt=j_opt, floor_with_best=floor_with_best,
                            warn=False)                         # J_opt and J^pi on score windows
    s["j_best"] = best.j_opt
    s["qss_slow"] = -d_qss
    s["random"] = np.random.default_rng(rng_seed).normal(size=d_qss.shape)
    s[REFERENCE] = best.reg[c]
    keep = HEADLINE + BASELINES + [REFERENCE]
    return {k: s[k] for k in keep if k in s}


def _window_mean(x: np.ndarray, mask: np.ndarray | None = None):
    """Per-level mean of a per-window quantity over a window subset, optionally over the
    completed windows only (NaN for a level with none): the crash-window diagnostic."""
    def fn(w):
        if mask is None:
            return {"s": x[:, w].mean(-1)}
        m = mask[:, w]
        return {"s": np.where(m.any(-1), (x[:, w] * m).sum(-1) / np.maximum(m.sum(-1), 1), np.nan)}
    return fn


def crash_split(x: np.ndarray, completed: np.ndarray, windows) -> dict:
    """Odd/even split-half Spearman r of the per-level window mean over all windows and over
    completed windows only, both on the *same* levels: those with a completed window in each
    half (the others, the hardest levels, are dropped from both columns and counted)."""
    a, b = F.odd_even(windows)
    ca, cb = _window_mean(x, completed)(a)["s"], _window_mean(x, completed)(b)["s"]
    xa, xb = _window_mean(x)(a)["s"], _window_mean(x)(b)["s"]
    same = np.isfinite(ca) & np.isfinite(cb)
    return {"all": F.correlation(xa[same], xb[same]), "completed": F.correlation(ca[same], cb[same]),
            "n": int(same.sum()), "n_total": len(x)}


def analyse(stats, infos, d_qss, d_centre, n_boot, j_opt=None, floor_with_best=False):
    W = stats[0].G_disc.shape[-1]
    plan = F.window_plan(W)
    g_all = np.stack([st.G_disc for st in stats])               # [C, L, W]
    kw = dict(j_opt=j_opt, floor_with_best=floor_with_best, warn=False)
    t_hb = F.regret_targets(g_all, plan.hb_target, **kw)
    t_op = F.regret_targets(g_all, plan.op_target, **kw)
    half = W // 2
    sizes = [m for m in (1, 2, 4, 8, 16, 32) if m <= half]
    res = []
    for c, (st, info) in enumerate(zip(stats, infos)):
        comp = F.completion_rate(st.completed)
        sc_hb = scores_for(st, plan.hb_score, d_qss, g_all, c, c, j_opt, floor_with_best)
        sc_op = scores_for(st, plan.operational, d_qss, g_all, c, c, j_opt, floor_with_best)
        # One bootstrap seed for every checkpoint: the levels are shared, so resamples are
        # paired across checkpoints too (used for the between-checkpoint MDE).
        fid_hb = F.fidelity_table(sc_hb, t_hb.j_opt, t_hb.j_pi[c], t_hb.reg[c], t_hb.reg_norm[c],
                                  n_boot=n_boot, seed=0, reference="agent_only", best_of=BEST_OF,
                                  keep_samples=True)
        fid_op = F.fidelity_table(sc_op, t_op.j_opt, t_op.j_pi[c], t_op.reg[c], t_op.reg_norm[c],
                                  n_boot=n_boot, seed=0, reference="agent_only", best_of=BEST_OF)

        def score_fn(w, st=st, c=c):
            return scores_for(st, w, d_qss, g_all, c, c, j_opt, floor_with_best)

        def target_fn(w, c=c):
            t = F.regret_targets(g_all, w, **kw)
            return {"reg": t.reg[c], "reg_norm": t.reg_norm[c], "j_pi": t.j_pi[c]}

        rel_s = F.split_halves(score_fn, range(W), max_tie_share=MAX_TIE_SHARE)
        rel_t = F.split_halves(target_fn, range(W), max_tie_share=MAX_TIE_SHARE)
        curves = F.reliability_curves(lambda w, sf=score_fn, tf=target_fn: {**sf(w), **tf(w)},
                                      range(W), sizes, n_splits=10, seed=c,
                                      max_tie_share=MAX_TIE_SHARE)
        s1 = level_scores(st, plan.hb_score, qss_distance=d_qss)
        res.append(dict(
            info=info, completion=comp, phase=F.phase_of(comp),
            j_pi=float(t_hb.j_pi[c].mean()), reg_mean=float(t_hb.reg[c].mean()),
            reg_median=float(np.median(t_hb.reg[c])), attained=float(t_hb.attained[c].mean()),
            negative=int(t_hb.negative[c].sum()),
            r12=float(F.spearman(t_hb.j_opt, t_hb.j_pi[c])), fid_hb=fid_hb, fid_op=fid_op,
            rel_s=rel_s, rel_t=rel_t, curves=curves,
            rc_oracle=float(F.regret_captured(t_hb.reg[c], t_hb.reg[c], 0.1)),
            crash_l1=crash_split(st.l1[1], st.completed, range(W)),
            crash_g=crash_split(st.G_disc, st.completed, range(W)),
            crash=float(1.0 - st.completed.mean()), sizes=sizes,
            calib=F.value_calibration(st.V_first, st.G_disc),
            s2_rate=float(np.mean(st.completed & (st.distance >= 0.97 * d_qss[:, None]))),
            s2_rate_centre=float(np.mean(st.completed & (st.distance >= 0.97 * d_centre[:, None]))),
            ratio=Q.ratio_quantiles(st.distance, st.completed, d_qss),
            ratio_centre=Q.ratio_quantiles(st.distance, st.completed, d_centre),
            rho_s1_s2=float(F.spearman(s1["learnability_s1"], s1["learnability_s2"])),
        ))
    return plan, res


def graded(res):
    return [r for r in res if r["attained"] < GRADABLE]


def checkpoint_diff_mde(res, names, alpha: float = 0.05, power: float = 0.8):
    """MDE of a between-checkpoint difference in rho (last minus first gradable checkpoint)
    from the paired level bootstrap: (z_alpha/2 + z_power) * SD of the paired difference.
    None when fewer than two gradable checkpoints share their graded levels."""
    g = graded(res)
    if len(g) < 2:
        return None
    first, last = g[0], g[-1]
    if not np.array_equal(first["fid_hb"].keep, last["fid_hb"].keep):
        return None
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    rows = {}
    for name in names:
        a, b = first["fid_hb"].samples.get(name), last["fid_hb"].samples.get(name)
        if a is None or b is None:
            continue
        ok = np.isfinite(a["rho_abs"]) & np.isfinite(b["rho_abs"])
        if ok.sum() < 10:
            continue
        sd = [float(np.nanstd(r["fid_hb"].samples[name]["rho_abs"])) for r in g
              if name in (r["fid_hb"].samples or {})]
        sd_d = float(np.std(b["rho_abs"][ok] - a["rho_abs"][ok]))
        rows[name] = dict(sd_rho=float(np.median(sd)), sd_diff=sd_d, mde_diff=z * sd_d,
                          corr=float(np.corrcoef(a["rho_abs"][ok], b["rho_abs"][ok])[0, 1]))
    return dict(first=first, last=last, n_levels=first["fid_hb"].n_levels, z=z, rows=rows)


# --------------------------------------------------------------------------- 64-window sub-pilot


def sub_pilot(stats_by_stem, j_ref, d_qss, n_levels):
    """Reliability curves to 32 windows per half from the sub-pilot's window statistics.
    ``j_ref`` [n_levels] is the fixed optimum per level (see the module docstring)."""
    out = {}
    for stem, (st, info) in stats_by_stem.items():
        W = st.G_disc.shape[-1]
        half = W // 2
        sizes = [m for m in (1, 2, 4, 8, 16, 32) if m <= half]
        dq = d_qss[:n_levels]

        def fn(w, st=st, dq=dq):
            s = level_scores(st, w, qss_distance=dq)
            j_pi = st.G_disc[:, w].mean(-1)
            return {**{k: s[k] for k in HEADLINE + ["agent_only"] if k in s},
                    "reg": j_ref - j_pi, "j_pi": j_pi}

        out[stem] = dict(info=info, sizes=sizes, half=half,
                         rel=F.split_halves(fn, range(W), max_tie_share=MAX_TIE_SHARE),
                         curves=F.reliability_curves(fn, range(W), sizes, n_splits=10, seed=0,
                                                     max_tie_share=MAX_TIE_SHARE),
                         completion=F.completion_rate(st.completed), levels=n_levels, windows=W)
    return out


# --------------------------------------------------------------------------- reliability helpers


def budgets(r: dict, name: str, half: int, goal: float = GOAL) -> tuple[float, bool, float]:
    """(Pearson SB budget bracketed by the empirical Pearson curve, extrapolated?,
    empirical Spearman budget) for one quantity of one checkpoint's results."""
    rel = r["rel_t"] if name in TARGETS else r["rel_s"]
    rp = rel["pearson"].get(name, np.nan)
    cp = r["curves"]["pearson"].get(name)
    cs = r["curves"]["spearman"].get(name)
    if cp is None or not np.isfinite(rp):
        pb, ext = float("nan"), False
    else:
        pb, ext = F.window_budget(rp, half, cp, r["sizes"], goal)
    sb = F.empirical_windows(cs, r["sizes"], goal) if cs is not None else float("nan")
    return pb, ext, sb


def median_budget(values) -> tuple[float, int, int]:
    """Median of measured budgets, dropping withheld (NaN) ones: (median, n_used,
    n_withheld). Infinite budgets (reliability <= 0, or never reached on the curve) count."""
    v = np.asarray(values, float)
    used = v[~np.isnan(v)]
    return (float(np.median(used)) if used.size else float("nan")), int(used.size), int(np.isnan(v).sum())


def sb_deviation(r: dict, name: str, method: str, half: int) -> np.ndarray:
    """Empirical minus Spearman-Brown reliability at each measured size below ``half``,
    the prophecy anchored on the split-half r at ``half`` windows."""
    rel = r["rel_t"] if name in TARGETS else r["rel_s"]
    rh = rel[method].get(name, np.nan)
    emp = r["curves"][method].get(name)
    sizes = np.array(r["sizes"], float)
    if emp is None or not np.isfinite(rh) or rh <= 0:
        return np.full(len(sizes[sizes < half]), np.nan)
    pred = F.spearman_brown(rh, sizes / half)
    return (emp - pred)[sizes < half]


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


def fig_fidelity(res, path: Path, tag: str, opt: str):
    """Small multiples: one panel per headline score, rho against absolute and normalised
    regret with 95% bootstrap bands, the agent-only baseline dashed for comparison."""
    x = np.array([r["info"]["timesteps"] / 1e6 for r in res])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes.ravel(), HEADLINE):
        _style(ax)
        for metric, color, label in (("rho_abs", PALETTE[0], "vs Reg"),
                                     ("rho_norm", PALETTE[1], f"vs Reg / {opt}")):
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
    fig.suptitle(f"{tag}: high-budget fidelity (odd-window scores vs even-window regret against"
                 f" {opt}), 95% bootstrap bands over levels", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_reliability(res, sub, path: Path, W: int, tag: str):
    """Left: split-half Spearman reliability (half the windows) across training. Middle and
    right: empirical reliability against window count at the newest gradable checkpoint,
    on the raw (Pearson) and rank (Spearman) scale, with the Spearman-Brown prophecy from
    the split half; open squares are the 64-window sub-pilot (50 levels, up to 32 windows)."""
    half = W // 2
    x = np.array([r["info"]["timesteps"] / 1e6 for r in res])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), facecolor=SURFACE)
    for a in ax:
        _style(a)
    for i, name in enumerate(HEADLINE):
        ax[0].plot(x, [r["rel_s"]["spearman"].get(name, np.nan) for r in res], "-o",
                   color=PALETTE[i], lw=1.5, ms=4, label=LABELS[name])
    ax[0].plot(x, [r["rel_t"]["spearman"]["reg"] for r in res], "-s", color=INK, lw=2, ms=5,
               label="target: regret")
    ax[0].axhline(GOAL, color=INK2, lw=0.8, ls=":")
    _log_steps(ax[0], x)
    ax[0].set_ylim(-0.2, 1)
    ax[0].set_xlabel("training steps (M)", fontsize=8, color=INK2)
    ax[0].set_ylabel(f"split-half Spearman r ({half} windows)", fontsize=8, color=INK2)
    ax[0].legend(fontsize=7, frameon=False, ncol=2, loc="lower right")
    g = graded(res)
    last = g[-1] if g else res[-1]
    sp = (sub or {}).get(last["info"].get("stem"))
    grid = np.geomspace(1, 64, 100)
    for a, method, title in ((ax[1], "pearson", "Pearson (raw values)"),
                             (ax[2], "spearman", "Spearman (ranks)")):
        for i, name in enumerate(HEADLINE + ["reg"]):
            color = INK if name == "reg" else PALETTE[i]
            emp = last["curves"][method].get(name)
            if emp is None or not np.isfinite(emp).any():
                continue
            rel = last["rel_t"] if name == "reg" else last["rel_s"]
            r_half = rel[method].get(name, np.nan)
            a.plot(last["sizes"], emp, "o", color=color, ms=5)
            if np.isfinite(r_half) and r_half > 0:
                a.plot(grid, F.spearman_brown(r_half, grid / half), "-" if method == "pearson" else "--",
                       color=color, lw=1.1)
            if sp is not None and name in sp["curves"][method]:
                a.plot(sp["sizes"], sp["curves"][method][name], "s", mfc="none", color=color, ms=6)
        a.axhline(GOAL, color=INK2, lw=0.8, ls=":")
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_ylim(-0.2, 1)
        a.set_xlabel("windows per level", fontsize=8, color=INK2)
        a.set_ylabel("reliability", fontsize=8, color=INK2)
        a.set_title(f"{last['info']['timesteps'] / 1e6:.0f}M checkpoint, {title}", fontsize=9,
                    color=INK)
    fig.suptitle(f"{tag}: reliability of scores and target. Middle and right: dots = main pilot,"
                 " open squares = 64-window sub-pilot, lines = Spearman-Brown from the 16-window"
                 " split half (dashed on ranks, where it has no basis)", fontsize=9, color=INK)
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


def _windows(v, cap: int | None = None) -> str:
    """A window budget: '-' when withheld (NaN), 'never' when reliability <= 0, '>cap' when
    the empirical curve never reached the goal (cap = the largest measured size)."""
    if v is None or np.isnan(v):
        return "-"
    if np.isinf(v):
        return f">{cap}" if cap is not None else "never"
    return f"{int(np.ceil(v))}"


def _range(vals, fmt="{:.2f}") -> str:
    v = [x for x in vals if np.isfinite(x)]
    return "-" if not v else f"{fmt.format(min(v))}-{fmt.format(max(v))}"


def _colmean(d: np.ndarray) -> np.ndarray:
    """Column means over the finite entries of a 2-D array (NaN where a column has none)."""
    ok = np.isfinite(d)
    n = ok.sum(0)
    return np.where(n > 0, np.where(ok, d, 0.0).sum(0) / np.maximum(n, 1), np.nan)


def sb_check(res, half: int) -> list[str]:
    """Empirical minus Spearman-Brown reliability, Pearson vs Spearman, over the gradable
    checkpoints: the prophecy is an identity for the Pearson reliability of a window mean,
    so a deviation that appears only on the rank scale is a property of the metric, not
    evidence that the windows are not exchangeable."""
    g = graded(res) or res
    sizes = [m for m in g[0]["sizes"] if m < half]
    head = " | ".join(f"{s} m={m}" for s in ("P", "S") for m in sizes)
    lines = [
        (f"**Spearman-Brown check** ({len(g)} gradable checkpoints). Mean over checkpoints of"
         " empirical minus prophesied reliability at m windows, the prophecy anchored on the"
         f" split-half r at {half} windows: *P* = Pearson on raw values, *S* = Spearman on"
         " ranks (where the prophecy has no basis; shown for contrast)."),
        "",
        f"| Quantity | {head} |",
        "|---|" + "---|" * (2 * len(sizes)),
    ]
    dev = {}
    for name in HEADLINE + ["agent_only", "reg"]:
        cells = []
        for method in ("pearson", "spearman"):
            m = _colmean(np.array([sb_deviation(r, name, method, half) for r in g]))
            dev[(name, method)] = m
            cells += [_num(v, "{:+.2f}") for v in m]
        mark = "" if name in MEAN_BASED else " †"
        lines.append(f"| {LABELS.get(name, name)}{mark} | " + " | ".join(cells) + " |")

    def mean_abs(method):
        v = np.concatenate([np.abs(dev[(n, method)]) for n in MEAN_BASED])
        return float(v[np.isfinite(v)].mean())

    lines += [
        "",
        ("† not a mean over windows (MaxMC: a maximum; MNA: a ratio of sums with a zero gate;"
         " learnability: p(1 − p) of a mean), so the prophecy is approximate on either scale."),
        "",
        (f"Mean |empirical − prophecy| over the window means (PVL, L1, disagreement, agent-only,"
         f" target Reg): Pearson {mean_abs('pearson'):.3f}, Spearman {mean_abs('spearman'):.3f}."
         " The windows are i.i.d. draws by construction, so the Pearson reliability of a window"
         " mean must follow the prophecy, and within sampling noise it does. Rank reliability"
         " has no such identity: for L1 and the target it starts well above the prophecy at one"
         " window and rises more slowly. Why is not established here; one candidate, heavy"
         " crash tails in the per-window values, reproduces the pattern in a synthetic test"
         " (`tests/test_fidelity.py`). Spearman-Brown is therefore not used for rank budgets:"
         " those are read off the empirical curve, and the sub-pilot below observes it to 32"
         " windows."),
    ]
    return lines


def crash_table(res, half: int) -> list[str]:
    """Split-half reliability of window means over all windows vs completed windows only,
    on the same levels."""
    head = " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M" for r in res)

    def cell(d):
        return f"{_num(d['all'])} / {_num(d['completed'])} (n={d['n']})"

    rows = [
        ("crash share of windows", [f"{r['crash'] * 100:.0f}%" for r in res]),
        ("L1 (λ 0.95): all / completed only", [cell(r["crash_l1"]) for r in res]),
        ("G^γ (= J^π): all / completed only", [cell(r["crash_g"]) for r in res]),
    ]
    return [
        ("**Crash windows and reliability.** Split-half Spearman r (odd vs even windows,"
         f" {half} each) of the per-level window mean, over all windows and over completed"
         " windows only. Both columns use the same levels: those with at least one completed"
         f" window in each half (n of {res[0]['crash_l1']['n_total']}; the dropped levels are"
         " the hardest ones). Where dropping crashed windows raises r, crashed windows add"
         " level-independent noise. That is suggestive for the plan's hurdle split (crash"
         " probability, and regret given completion) and for robust aggregation, but it is a"
         " descriptive comparison, not a test, and it does not bear on the Spearman-Brown"
         " question above. The headline scores keep the reference definitions (means over"
         " all windows)."),
        "",
        f"| Quantity | {head} |",
        "|---|" + "---|" * len(res),
        *[f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows],
    ]


def summary(res, half: int, sub, proxy: bool) -> list[str]:
    """Computed bullets for the top of the report: thresholds only, no interpretation that a
    re-run on more checkpoints could silently contradict."""
    ratios = [r["calib"]["ratio"] for r in res]
    sig = [r["info"]["sigma"] for r in res]
    comp = [r["completion"] for r in res]
    g = graded(res) if proxy else res
    raw = all(0.8 < x < 1.25 for x in ratios)
    verdict = ("so critic values are in raw reward units" if raw else
               "**outside 0.8-1.25: check the value units before trusting value-based scores**")
    out = [
        (f"- **Units:** σ = {min(sig):.1f}-{max(sig):.1f}; mean V(o₀) / mean G^γ ="
         f" {min(ratios):.3f}-{max(ratios):.3f} across checkpoints, {verdict} (normalised values"
         f" would give ~1/σ ≈ {1 / max(sig):.2f})."),
        (f"- **Phase:** window completion {min(comp) * 100:.0f}%-{max(comp) * 100:.0f}%; phases"
         f" covered: {', '.join(p for p in ('early', 'mid', 'late') if p in {r['phase'] for r in res})}."),
    ]
    if proxy:
        bad = [r for r in res if r["attained"] >= GRADABLE]
        out.append(
            f"- **Gradable checkpoints:** {len(g)} of {len(res)}. With proxy labels a checkpoint"
            f" that attains J_best on {GRADABLE:.0%} of levels or more has that share of its"
            " regret at exactly 0, which dominates every rank metric; it is excluded from these"
            " bullets, the Spearman-Brown check and the reliability figure"
            + (f" (excluded: {', '.join(f'{r['info']['timesteps'] / 1e6:.0f}M ({r['attained']:.0%})' for r in bad)})."
               if bad else "."))
        out.append(
            "- **Regret concentration:** the regret-sorted oracle's RC@10% is"
            f" {_range([r['rc_oracle'] for r in g])} on the gradable checkpoints and"
            f" {_range([r['rc_oracle'] for r in bad])} on the excluded ones. It rises with"
            " training because the proxy's exact zeros pile up (late regret is concentrated"
            " on the few levels the newest checkpoint has not mastered), so raw RC@10% is"
            " phase-dependent and, late in this run, a proxy artefact; compare scores by the"
            " RC lift." if bad else
            "- **Regret concentration:** the regret-sorted oracle's RC@10% is"
            f" {_range([r['rc_oracle'] for r in g])}; compare scores by the RC lift.")
    else:
        neg = max((r["negative"] for r in res), default=0)
        out.append(f"- **Negative regret** (J^π above J\\*_20, a label problem): up to {neg} levels"
                   " at one checkpoint.")
    if not g:
        return out
    # Target reliability.
    rs = [r["rel_t"]["spearman"]["reg"] for r in g]
    rp = [r["rel_t"]["pearson"]["reg"] for r in g]
    pb = [budgets(r, "reg", half)[0] for r in g]
    sb = [budgets(r, "reg", half)[2] for r in g]
    reach = sum(np.isfinite(v) for v in sb)
    out.append(
        f"- **Target reliability (Reg, {half} windows per half):** split-half Spearman r"
        f" {_range(rs)} (median {np.median(rs):.2f}), Pearson r {_range(rp)} (median"
        f" {np.median(rp):.2f}). Windows for {GOAL} on the raw scale (Pearson, Spearman-Brown"
        f" bracketed by the empirical curve): median {_windows(np.median(pb))}, range"
        f" {_windows(min(pb))}-{_windows(max(pb))}. On the rank scale the empirical curve"
        f" reaches {GOAL} within {half} windows at {reach} of {len(g)} checkpoints; budgets beyond"
        f" {half} are not observed in the main pilot.")
    if sub:
        cells = []
        for s in sub.values():
            cells.append(f"{s['info']['timesteps'] / 1e6:.0f}M: Spearman"
                         f" {_num(s['rel']['spearman'].get('reg'))}, Pearson"
                         f" {_num(s['rel']['pearson'].get('reg'))}")
        out.append(f"- **64-window sub-pilot** (split-half r at {next(iter(sub.values()))['half']}"
                   " windows per half, fixed optimum, regret target): " + "; ".join(cells) + ".")
    need = []
    for name in HEADLINE:
        vals = [budgets(r, name, half) for r in g]
        med, used, withheld = median_budget([v[0] for v in vals])
        if not used:
            continue
        smed, _, _ = median_budget([v[2] for v in vals])
        ext = sum(v[1] for v in vals if np.isfinite(v[0]))
        note = []
        if withheld:
            note.append(f"{withheld} withheld")
        if ext:
            note.append(f"{ext} beyond the curve")
        if name in MAX_BASED:
            note.append("max-based")
        need.append(f"{LABELS[name]} {_windows(med)} / {_windows(smed, half)}"
                    + (f" ({', '.join(note)})" if note else ""))
    out.append(f"- **Score reliability**, median windows for {GOAL} over gradable checkpoints,"
               f" raw scale (Spearman-Brown, bracketed) / rank scale (empirical, >{half} = not"
               " reached on the curve): " + "; ".join(need) + ".")
    pos, beat, best = [], [], []
    for name in HEADLINE:
        rows = [r["fid_hb"].rows.get(name) for r in g]

        def k_above(key, rows=rows):
            return sum(1 for row in rows if row and key in row and np.isfinite(row[key].lo)
                       and row[key].lo > 0)

        pos.append(f"{LABELS[name]} {k_above('rho_abs')}/{len(g)}")
        beat.append(f"{LABELS[name]} {k_above('delta_rho_abs')}/{len(g)}")
        best.append(f"{LABELS[name]} {k_above('delta_best_rho_abs')}/{len(g)}")
    out += [
        "- **ρ vs Reg with CI above 0** (high budget), checkpoints: " + "; ".join(pos) + ".",
        "- **ρ above the agent-only baseline** (paired CI above 0), checkpoints: "
        + "; ".join(beat) + ".",
        "- **ρ above the best of the agent-only and policy-agnostic baselines** (per-resample"
        " best, paired CI above 0), checkpoints: " + "; ".join(best) + ".",
    ]
    rq = [r["ratio"] for r in g]
    out.append(
        f"- **d_QSS (S2 reference):** completed windows cover a median"
        f" {_range([q['q0.5'] for q in rq], '{:.3f}')} of d_QSS on the racing line (99th"
        f" percentile {_range([q['q0.99'] for q in rq], '{:.3f}')}, maximum"
        f" {_range([q['q1'] for q in rq], '{:.3f}')}; share above 1:"
        f" {_range([q['above_1'] * 100 for q in rq], '{:.0f}')}%). Against the centreline"
        f" definition it replaced the maximum was {_range([r['ratio_centre']['q1'] for r in g], '{:.3f}')}."
        f" S2 rate / completion: {_range([r['s2_rate'] / r['completion'] for r in g])};"
        f" Spearman(learnability S1, S2): {_range([r['rho_s1_s2'] for r in g])}.")
    return out


def g1_bullet(sub) -> str:
    """The G1 reading and, when the sub-pilot ran, what it gives at the 64-window budget."""
    out = ("- **G1 reliability metric (this report's reading, needs sign-off):** the plan's gate"
           " (reliability ≥ 0.9 late, ≥ 0.8 mid, within 64 windows) does not name the metric."
           " Here it is read on the rank scale: the Spearman reliability of the score and of the"
           " target at the window budget the audit uses, measured on the empirical curve, not"
           " prophesied, because the fidelity endpoints are rank correlations. Pearson"
           " reliability with Spearman-Brown is used only to size budgets beyond the curve.")
    if not sub:
        return out + " The 64-window sub-pilot has not been run, so the gate is not evaluated."
    cells, pass_s, pass_p, low = [], 0, 0, set()
    for s in sub.values():
        j = len(s["sizes"]) - 1
        rs, rp = s["curves"]["spearman"]["reg"][j], s["curves"]["pearson"]["reg"][j]
        pass_s += rs >= 0.8
        pass_p += rp >= 0.8
        cells.append(f"{s['info']['timesteps'] / 1e6:.0f}M: {rs:.2f} rank / {rp:.2f} raw")
        for n in HEADLINE:
            c = s["curves"]["spearman"].get(n)
            if c is not None and np.isfinite(c[j]) and c[j] < 0.8:
                low.add(LABELS[n])
    m = next(iter(sub.values()))["sizes"][-1]
    return out + (
        f" Read this way, at the 64-window budget ({m} windows per half; sub-pilot, mean of 10"
        f" random splits, {next(iter(sub.values()))['levels']} levels) the target Reg reaches"
        f" {'; '.join(cells)}, so the mid-phase gate (0.8) holds at {pass_s} of {len(sub)}"
        f" checkpoints on the rank scale and at {pass_p} of {len(sub)} on the raw scale: the"
        " choice of metric decides the verdict for the target. Scores below 0.8 on the rank"
        f" scale at {m} windows at some sub-pilot checkpoint: {', '.join(sorted(low)) or 'none'}."
        " No checkpoint is late-phase yet, so the 0.9 late gate is untested.")


def report(args, plan, res, d_qss, d_centre, sub, out: Path, skipped_ckpts):
    L = len(d_qss)
    W = len(plan.hb_score) + len(plan.hb_target)
    half = W // 2
    wall = sum(r["info"]["wall"] for r in res)
    cpu = sum(r["info"]["cpu"] for r in res)
    proxy = args.j_opt is None
    opt = "J_best" if proxy else "J\\*_20"
    labels = dict(LABELS, reg_norm=f"**target: Reg/{opt}**")
    if proxy:
        banner = [
            "> **PROXY-LABEL PILOT.** Regret here is measured against **J_best**, the best mean",
            "> return any audited checkpoint reached on the level, not against the TO-OCP optimum",
            "> J\\*_20. These numbers test the E1 machinery (plan phase 11) and must not be read as",
            "> findings about the scores. The audit is ready for real labels: re-run with",
            "> `--j-opt labels.npy` (J\\*_20 for the same levels, in level order).",
        ]
        target = ("- **Target:** J^π = mean discounted return over the target windows; J_best ="
                  " max over checkpoints of that mean (from the target windows only, so scores and"
                  " labels share no window noise); Reg = J_best − J^π, Reg_norm = Reg / J_best."
                  " J_best is a lower bound on the env's optimal expected return (every window,"
                  " crashed or not, is a trajectory of the env), not necessarily on the J\\*_20"
                  " label.")
    else:
        floor = (" The optimum is max(J\\*_20, J_best) (`--floor-with-best`)."
                 if args.floor_with_best else "")
        banner = [(f"> **TO-OCP LABELS** from `{args.j_opt}` (J\\*_20 per level). Pilot scale:"
                   " one seed, the levels and checkpoints listed below.")]
        target = ("- **Target:** J^π = mean discounted return over the target windows;"
                  " Reg = J\\*_20 − J^π, Reg_norm = Reg / J\\*_20 (J\\*_20 is a best-known optimum,"
                  " so Reg is a lower bound on regret and can be negative where a checkpoint"
                  " beats the label)." + floor)
    legacy = [r["info"]["stem"] for r in res if r["info"].get("legacy_cache")]
    lines = [
        "# Pilot audit: level-score fidelity on the DR pilot's early checkpoints",
        "",
        *banner,
        "",
        f"Generated by `scripts/pilot_audit.py` from `{args.run}`.",
        "",
        f"## Summary (computed; {'proxy' if proxy else 'TO-OCP'} labels)",
        "",
        *summary(res, half, sub, proxy),
        g1_bullet(sub),
        "",
        "## Setup",
        "",
        f"- **Checkpoints:** {len(res)} DR-pilot checkpoints, "
        + ", ".join(f"{r['info']['timesteps'] / 1e6:.1f}M" for r in res)
        + " (single seed)."
        + (f" Skipped: {', '.join(skipped_ckpts)}." if skipped_ckpts else ""),
        (f"- **Levels:** the first {L} A_DR levels (`make_levels(PRNGKey({ROOT_SEED}), 'a_dr')`),"
         " grammar fitted on R-dev only. No lockbox (R-test) circuit is used anywhere."),
        (f"- **Windows:** {W} stochastic 1,200-step windows per level and checkpoint, with 1%"
         " observation noise; one chunk phase per window drawn from U{0..127} (seed"
         f" {ROOT_SEED}); mode F scoring (window end terminal), λ = 0.95 headline. The same"
         " random draws are reused at every checkpoint (common random numbers)."),
        (f"- **Window roles (plan 2.2):** high-budget scores use the odd windows (1, 3, ...,"
         f" {W - 1}), the target the even windows; operational scores use windows 1-4 and their"
         f" target windows 5-{W}."),
        target,
        ("- **d_QSS** (S2 success threshold and the geometric baseline): the centreline distance"
         " the QSS point mass of the level's own car (context and hidden grip) covers in 60 s on"
         " the minimum-curvature line inside the corridor (edge margin 1 m, as the env's"
         " off-track test), starting from x0 (n = 0, 0.8·v_QSS(0)); `audit.qss_distance`. It"
         " replaces the centreline-curvature definition, which completed windows beat by up to"
         " 23% (see *d_QSS check*)."),
        ("- **Phase:** window completion rate over all windows of the audited levels"
         " (<50% early, 50-90% mid, ≥90% late)."),
        (f"- **Statistics:** percentile bootstrap over levels, {args.boot} resamples shared by all"
         " scores and all checkpoints (Δ columns are paired). One seed, so no seed-level"
         " variance is included: CIs are too narrow for the plan's two-way bootstrap."),
        (f"- **Compute:** rollouts + window statistics {wall / 60:.1f} min wall, {cpu / 60:.1f} min"
         " CPU in total (nice 15, on a machine shared with training runs)."
         + (f" Cached statistics from before the cache recorded checkpoint hashes, accepted"
            f" because each checkpoint file predates its cache: {len(legacy)} checkpoints."
            if legacy else "")),
        "",
        (f"Baselines: *agent-only* = −J^π on the score windows; *{LABELS['j_best']}* = the"
         " policy-agnostic optimum baseline"
         + (" (recomputed from the score windows, in the role J\\*_20 will play)" if proxy else "")
         + "; *geometric* = −d_QSS (shorter = slower, more technical strip); *random*. *Best"
         " baseline* = the best of agent-only, J_best and geometric within each bootstrap"
         f" resample. The *{LABELS[REFERENCE]}* row is the oracle-operational reference,"
         f" {opt} − J^π on the score windows: a reference, not a ceiling."),
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
         f" | S2 rate | mean J^π | mean Reg | median Reg | attains {opt} | corr({opt}, J^π) |"),
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
        ((f"*attains {opt}*: share of levels on which this checkpoint *is* the optimum, so its"
          " regret is exactly 0 there. ") if proxy or args.floor_with_best else "")
        + (f"*corr({opt}, J^π)*: rank correlation of the two loadings of the two-loading"
           " regression; near ±1 they are unidentified. *S2 rate*: windows that complete and"
           " cover at least 97% of d_QSS (the non-saturating success)."),
        "",
        "### d_QSS check",
        "",
        ("distance / d_QSS over completed windows (median / 99th percentile / maximum, and the"
         " share above 1), for the racing-line d_QSS used here and the centreline definition it"
         " replaced. A valid upper reference keeps the ratio at or below about 1; S2 can only"
         " differ from S1 where it is."),
        "",
        ("| Checkpoint | racing line: median / p99 / max | above 1 | centreline: median / p99 / max"
         " | S2 rate (racing / centreline) | S1 rate | ρ(learnability S1, S2) |"),
        "|---|---|---|---|---|---|---|",
    ]
    for r in res:
        q, qc = r["ratio"], r["ratio_centre"]
        lines.append(
            f"| {r['info']['timesteps'] / 1e6:.1f}M | {q['q0.5']:.3f} / {q['q0.99']:.3f} / {q['q1']:.3f}"
            f" | {q['above_1'] * 100:.1f}% | {qc['q0.5']:.3f} / {qc['q0.99']:.3f} / {qc['q1']:.3f}"
            f" | {r['s2_rate'] * 100:.1f}% / {r['s2_rate_centre'] * 100:.1f}%"
            f" | {r['completion'] * 100:.1f}% | {_num(r['rho_s1_s2'], '{:+.2f}')} |")
    worst = max(r["ratio"]["q1"] for r in res)
    over = [r for r in res if r["ratio"]["above_1"] > 0.1]
    lines += [
        "",
        (("**Verdict:** the racing-line d_QSS is an upper reference on these rollouts (maximum"
          f" ratio {worst:.3f}).") if worst <= 1.02 else
         (f"**Verdict: d_QSS is close to, but not strictly, an upper reference** (maximum ratio"
          f" {worst:.3f}, against {max(r['ratio_centre']['q1'] for r in res):.3f} for the"
          " centreline definition)."
          + (f" From {over[0]['info']['timesteps'] / 1e6:.0f}M on, more than 10% of completed"
             f" windows beat it (up to {max(r['ratio']['above_1'] for r in res):.0%}), and S2"
             " starts to saturate again (S2 rate approaching the completion rate, ρ(S1, S2)"
             f" up to {max(r['rho_s1_s2'] for r in over):.2f})." if over else "")
          + " A minimum-curvature line is not the minimum-time line, and the point-mass g-g"
          " model evaluates tyre friction at one representative load; which of these lets the"
          " agent beat d_QSS has not been tested. The reference (or the 0.97 fraction) needs"
          " a decision before `audit-v1` is frozen.")),
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
              ("Spearman ρ against absolute regret, 95% CI; then the RC@10% lift (share of total"
               " regret in the score's top 10% of levels, rescaled so that random = 0 and the"
               " regret-sorted oracle = 1); then Δρ against the best of the agent-only and"
               " policy-agnostic baselines, per resample (paired CI; the plan's"
               " 'uninformative' rule)."), ""]
    for metric, title, signed in (("rho_abs", "ρ vs Reg", True), ("rc_lift", "RC lift", True),
                                  ("delta_best_rho_abs", "Δρ vs best baseline", True)):
        lines += [f"**{title}**", "",
                  "| Score | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M" for r in res) + " |",
                  "|---|" + "---|" * len(res)]
        for name in HEADLINE + BASELINES + [REFERENCE]:
            cells = [_ci(r["fid_op"].rows.get(name, {}).get(metric), signed=signed) for r in res]
            lines.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")
        lines.append("")

    sizes = res[0]["sizes"]
    lines += [
        "## Reliability and power",
        "",
        (f"Split-half reliability: each quantity computed on the odd and on the even windows"
         f" ({half} each) and correlated across levels, on the rank scale (Spearman, the scale"
         " of the fidelity endpoints) and on the raw scale (Pearson, the scale on which"
         " Spearman-Brown holds for a window mean). Cells: Spearman r / Pearson r."),
        "",
        "| Quantity | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M" for r in res) + " |",
        "|---|" + "---|" * len(res),
    ]
    for name in HEADLINE + ["agent_only"] + TARGETS:
        cells = []
        for r in res:
            rel = r["rel_t"] if name in TARGETS else r["rel_s"]
            cells.append(f"{_num(rel['spearman'].get(name))} / {_num(rel['pearson'].get(name))}")
        lines.append(f"| {labels.get(name, name)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (f"Windows for {GOAL}. *raw*: the Spearman-Brown prophecy from the Pearson split half,"
         " bracketed by the empirical Pearson curve (above every measured size that missed the"
         " goal, not above the first that reached it; `*` = beyond the measured curve, prophecy"
         f" only). *rank*: the smallest measured size whose empirical Spearman reliability"
         f" reaches {GOAL} (>{half}: not reached). *op.*: empirical Spearman reliability at the"
         f" operational budget, {len(plan.operational)} windows (scores only; the target's"
         f" {len(plan.op_target)} operational windows lie beyond the main pilot's curve)."
         " Cells: raw / rank / op."),
        "",
        "| Quantity | " + " | ".join(f"{r['info']['timesteps'] / 1e6:.0f}M" for r in res) + " |",
        "|---|" + "---|" * len(res),
    ]
    i_op = sizes.index(len(plan.operational)) if len(plan.operational) in sizes else None
    for name in HEADLINE + ["agent_only"] + TARGETS:
        cells = []
        for r in res:
            pb, ext, sb = budgets(r, name, half)
            emp = r["curves"]["spearman"].get(name)
            op = emp[i_op] if (emp is not None and i_op is not None and name not in TARGETS) else np.nan
            cells.append(f"{_windows(pb)}{'*' if ext and np.isfinite(pb) else ''}"
                         f" / {_windows(sb, half)} / {_num(op)}")
        lines.append(f"| {labels.get(name, name)} | " + " | ".join(cells) + " |")
    mde = {n: F.spearman_mde(n) for n in (L, 1000, 1412)}
    lines += [
        "",
        "![reliability](pilot-audit-reliability.png)",
        "",
        *sb_check(res, half),
        "",
    ]
    if sub:
        any_sub = next(iter(sub.values()))
        h2 = any_sub["half"]
        lines += [
            (f"**64-window sub-pilot.** {any_sub['levels']} levels (the first of the {L}),"
             f" {any_sub['windows']} windows per level and checkpoint, independent random draws;"
             f" regret against a fixed optimum per level ({'the main pilot J_best over all its windows and checkpoints' if proxy else 'J*_20'})."
             f" Empirical reliability at m windows (mean of 10 random disjoint pairs); m = {h2}"
             f" is the high-budget score at the plan's 64-window maximum. Cells: Spearman /"
             " Pearson; the Spearman-Brown prophecy from the main pilot's 16-window Pearson"
             " split half is in brackets where it applies."),
            "",
        ]
        for stem, s in sub.items():
            main = next((r for r in res if r["info"].get("stem") == stem), None)
            lines += [
                f"*{s['info']['timesteps'] / 1e6:.1f}M (completion {s['completion'] * 100:.0f}%)*",
                "",
                "| Quantity | " + " | ".join(f"m={m}" for m in s["sizes"]) + " |",
                "|---|" + "---|" * len(s["sizes"]),
            ]
            for name in HEADLINE + ["agent_only", "reg", "j_pi"]:
                cs, cp = s["curves"]["spearman"].get(name), s["curves"]["pearson"].get(name)
                if cs is None:
                    continue
                rp16 = np.nan
                if main is not None:
                    rel = main["rel_t"] if name in TARGETS else main["rel_s"]
                    rp16 = rel["pearson"].get(name, np.nan)
                cells = []
                mean_based = name in MEAN_BASED + ["j_pi"] and name != "reg"
                for j, m in enumerate(s["sizes"]):
                    pro = (f" [{F.spearman_brown(rp16, m / half):.2f}]"
                           if mean_based and np.isfinite(rp16) and rp16 > 0 else "")
                    cells.append(f"{_num(cs[j])} / {_num(cp[j])}{pro}")
                mark = "" if name in MEAN_BASED + ["j_pi"] else " †"
                lines.append(f"| {labels.get(name, name)}{mark} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.append(
            "Brackets only for window means (†: not a window mean). They come from the main"
            " pilot's 150 levels and draws, so level sampling adds to any gap. The sub-pilot's"
            " target uses a fixed optimum and the main pilot's a J_best recomputed per half, so"
            " the target rows carry no bracket.")
        lines.append("")
    else:
        lines += ["**64-window sub-pilot:** not run (no cached sub-pilot statistics).", ""]
    dm = checkpoint_diff_mde(res, HEADLINE + ["agent_only"])
    lines += [
        "**Level power.** Smallest Spearman ρ distinguishable from 0 at α = 0.05 with 80% power"
        " (Fisher z, variance 1.06/(n−3)): "
        + ", ".join(f"{mde[n]:.3f} with {n} levels" for n in mde)
        + ". Scores graded on the same levels are paired, so the bootstrap Δ columns are"
        " tighter than an independent-samples bound.",
        "",
    ]
    if dm and dm["rows"]:
        scale = np.sqrt(dm["n_levels"] / 1000)
        lines += [
            (f"**Between-checkpoint MDE** (plan 2.3, row B: a later-minus-earlier difference in"
             f" ρ). The level bootstrap is paired across checkpoints (same levels, same"
             f" resamples), so SD(ρ_late − ρ_early) comes straight from it, for"
             f" {dm['last']['info']['timesteps'] / 1e6:.0f}M minus"
             f" {dm['first']['info']['timesteps'] / 1e6:.0f}M on {dm['n_levels']} levels; MDE ="
             f" {dm['z']:.2f}·SD (α = 0.05 two-sided, 80% power). Scaled to 1,000 levels by"
             f" √({dm['n_levels']}/1000). One seed: seed-level variance is not included."),
            "",
            ("| Score | SD of ρ (median over gradable) | SD of Δρ | corr(ρ_early, ρ_late) | MDE of Δρ |"
             " MDE at 1,000 levels |"),
            "|---|---|---|---|---|---|",
        ]
        for name, row in dm["rows"].items():
            lines.append(f"| {LABELS[name]} | {row['sd_rho']:.3f} | {row['sd_diff']:.3f}"
                         f" | {row['corr']:+.2f} | {row['mde_diff']:.3f} | {row['mde_diff'] * scale:.3f} |")
        lines.append("")
    lines += [
        ("Caveats on the reliability numbers: Spearman-Brown is applied only on the raw scale,"
         " where it is an identity for a mean of i.i.d. windows. MaxMC (and the max-return"
         " term of MaxMC-unit) take a maximum over windows, MNA is a ratio of sums with a zero"
         " gate and learnability is p(1 − p) of a mean, so their raw budgets are only"
         " indicative (MNA's raw-scale r is also dominated by a few outlying levels); read"
         " their rank budgets off the curve. A reliability near 0 or negative cannot be"
         " extrapolated (\"never\")."
         " Split-half r is not reported (\"-\") when more than"
         f" {MAX_TIE_SHARE:.0%} of levels share one value in either half (e.g. S2 learnability"
         " while almost no window succeeds): the halves then agree on their ties and r is"
         " spuriously high. ρ* below disattenuates a Spearman ρ by Spearman split-half"
         " reliabilities, which is an approximation (the correction is exact for Pearson"
         " correlations under classical test theory)."),
        "",
        *crash_table(res, half),
        "",
        "## Per-checkpoint tables (high budget)",
        "",
        (f"ρ = Spearman against Reg and against Reg/{opt}; RC@10% = share of total regret in"
         " the score's top 10% of levels (random = k/L = 0.10); lift = (RC − k/L) / (RC_oracle −"
         " k/L), so random = 0 and the regret-sorted oracle = 1. The oracle's own share (in each"
         " heading) depends on how concentrated regret is: early it is spread almost evenly and"
         " the oracle captures little more than k/L, so the raw share barely moves; late, the"
         " proxy's exact zeros concentrate it on a few levels. Dividing by the oracle share"
         " alone would rate a random score close to 1 early on; the lift does not."
         " P@10% = overlap of the top 10% sets; a, b = standardised loadings of rank(S) on"
         f" rank({opt}) and rank(J^π) (faithful: a > 0, b < 0); Δρ = ρ minus the agent-only"
         " baseline's ρ, paired; Δρ best = ρ minus the best baseline's ρ per resample, paired;"
         " ρ* = ρ disattenuated by the Spearman split-half reliabilities of score and target (an"
         " approximation; it can exceed 1 in magnitude when those reliabilities are small)."),
    ]
    for r in res:
        tab = r["fid_hb"]
        lines += [
            "",
            f"### {r['info']['timesteps'] / 1e6:.1f}M steps: completion {r['completion'] * 100:.1f}%"
            f" ({r['phase']}), {tab.n_levels} levels graded"
            + (f", {tab.n_dropped} dropped ({opt} ≤ 1)" if tab.n_dropped else "")
            + f", oracle RC@10% {r['rc_oracle']:.2f}"
            + (f", attains {opt} on {r['attained']:.0%} (not gradable)" if proxy and r["attained"] >= GRADABLE else ""),
            "",
            (f"| Score | ρ vs Reg | ρ vs Reg/{opt} | RC@10% | RC lift | P@10% | a | b | Δρ vs agent-only"
             " | Δρ vs best baseline | ρ* |"),
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in HEADLINE + BASELINES + [REFERENCE]:
            row = tab.rows.get(name)
            if row is None:
                lines.append(f"| {LABELS[name]} | " + " | ".join(["-"] * 10) + " |")
                continue
            rel = r["rel_s"]["spearman"].get(name, np.nan) if name != REFERENCE else np.nan
            rho_star = F.disattenuate(row["rho_abs"].value, rel, r["rel_t"]["spearman"]["reg"])
            lines.append(
                f"| {LABELS[name]} | {_ci(row['rho_abs'])} | {_ci(row['rho_norm'])} | {_ci(row['rc'], signed=False)}"
                f" | {_ci(row['rc_lift'])} | {_ci(row['p_at_q'], signed=False)}"
                f" | {_ci(row['a_opt'])} | {_ci(row['b_pi'])}"
                f" | {_ci(row.get('delta_rho_abs'))} | {_ci(row.get('delta_best_rho_abs'))}"
                f" | {_num(float(rho_star), '{:+.2f}')} |")
        if tab.skipped:
            lines.append(f"\nSkipped (non-finite on graded levels): {', '.join(tab.skipped)}.")

    lines += [
        "",
        "## Limits of this pilot",
        "",
    ]
    if proxy:
        lines += [
            ("- **Degenerate late.** J_best is attained by the best checkpoint on each level, so"
             " that checkpoint's regret is exactly 0 there (see *attains J_best*). As training"
             " improves, the newest checkpoints' labels collapse toward 0; checkpoints that"
             f" attain J_best on {GRADABLE:.0%} of levels or more are reported but not graded."
             " The earlier checkpoints are graded against what this run later reached, which is"
             " regret relative to the run, not to the optimum."),
            ("- **Collinear loadings.** J_best is close to the late J^π, so corr(J_best, J^π) is"
             " high and the two-loading coefficients are poorly identified; with J\\*_20 (a"
             " property of the level, not of the run) the regression becomes meaningful."),
            ("- **Lower bound, not the optimum.** Reg ≥ 0 by construction and understates true"
             " regret everywhere the run never got close to J\\*. Normalised regret divides by a"
             " run-dependent number."),
        ]
    lines += [
        ("- **Pilot scale.** One seed, mid-phase checkpoints only (no late phase yet unless the"
         f" table says so), {L} levels instead of 1,000 A_DR + 300 A_tail + 100 A_OOD, {W}"
         " windows (64 only in the sub-pilot) without the adaptive raise, no Holm correction,"
         " no seed-level bootstrap."),
        ("- **d_QSS** is a point-mass reference on a minimum-curvature line, not a minimum-time"
         " trajectory of the full model; the *d_QSS check* table says how often the agent beats"
         " it. The start deficit (0.8·v_QSS(0)) is now included."),
    ]
    if proxy:
        lines += [
            "",
            ("What carries over unchanged to the real audit: the window roles and cross-fitting,"
             " the metrics and their tie handling, the paired bootstrap (including the best-"
             "baseline rule), the split-half reliability on both scales, the empirical window"
             " budgets, d_QSS, and the units check."),
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
    ap.add_argument("--report-only", action="store_true",
                    help="main pilot: use cached checkpoints only (the sub-pilot still rolls out"
                         " any of --sub-ckpts it has not cached; pass --sub-ckpts '' to skip it)")
    ap.add_argument("--j-opt", default=None,
                    help=".npy of J*_20 for the same levels, in level order (replaces the proxy)")
    ap.add_argument("--floor-with-best", action="store_true",
                    help="with --j-opt: use max(J*_20, J_best) as the optimum")
    ap.add_argument("--sub-levels", type=int, default=50)
    ap.add_argument("--sub-windows", type=int, default=64)
    ap.add_argument("--sub-ckpts", default="000040471316,000060341763",
                    help="comma-separated checkpoint stems for the 64-window sub-pilot ('' = none)")
    args = ap.parse_args()

    out, cache = Path(args.out), Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    j_opt = None
    if args.j_opt:
        j_opt = np.load(args.j_opt)
        if j_opt.shape != (args.levels,):
            raise SystemExit(f"--j-opt has shape {j_opt.shape}, expected ({args.levels},)")
        LABELS.update({"j_best": "J\\*_20 (policy-agnostic)",          # tables only
                       REFERENCE: "oracle-operational J\\*_20 − G (reference)"})
    lv, tracks, x0, d_qss, d_centre = pilot_levels(args.levels, cache)
    print(f"levels ready ({time.perf_counter() - t0:.0f} s); median d_QSS {np.median(d_qss):.0f} m"
          f" (centreline {np.median(d_centre):.0f} m)", flush=True)
    phases = np.random.default_rng(ROOT_SEED).integers(0, 128, (args.levels, args.windows))
    paths = sorted(Path(args.run, "ckpt").glob("*.pkl"))
    if args.max_ckpts:
        paths = paths[:args.max_ckpts]
    stats, infos, skipped = [], [], []
    for p in paths:
        if args.report_only and not (cache / f"{p.stem}.npz").exists():
            skipped.append(f"{p.stem} (not audited yet: --report-only)")
            continue
        try:
            st, info = audit_checkpoint(p, cache, lv, tracks, x0, args.windows, phases)
        except (EOFError, pickle.UnpicklingError) as exc:
            skipped.append(f"{p.stem} (unreadable, probably mid-write: {type(exc).__name__})")
            continue
        stats.append(st)
        infos.append(info)
        print(f"{p.stem}: completion {st.completed.mean():.3f}, wall {info['wall']:.0f} s, "
              f"cpu {info['cpu']:.0f} s{' (legacy cache)' if info['legacy_cache'] else ''}", flush=True)
    if len(stats) < 2:
        raise SystemExit("need at least two audited checkpoints")
    plan, res = analyse(stats, infos, d_qss, d_centre, args.boot, j_opt=j_opt,
                        floor_with_best=args.floor_with_best)
    print(f"analysed ({time.perf_counter() - t0:.0f} s)", flush=True)

    sub = {}
    stems = [s for s in args.sub_ckpts.split(",") if s]
    if stems:
        n_sub = args.sub_levels
        sub_cache = cache / f"sub{args.sub_windows}"
        g_all = np.stack([st.G_disc for st in stats])
        j_ref = (j_opt if j_opt is not None else g_all.mean(-1).max(0))[:n_sub]
        sub_lv = lv._replace(keys=lv.keys[:n_sub], ctx=lv.ctx[:n_sub], hidden_mu=lv.hidden_mu[:n_sub])
        sub_tracks = jax.tree.map(lambda a: a[:n_sub], tracks)
        sub_phases = np.random.default_rng(ROOT_SEED + 1).integers(0, 128, (n_sub, args.sub_windows))
        got = {}
        for stem in stems:
            p = Path(args.run, "ckpt", f"{stem}.pkl")
            if not p.exists():
                continue
            got[stem] = audit_checkpoint(p, sub_cache, sub_lv, sub_tracks, x0[:n_sub],
                                         args.sub_windows, sub_phases, key_fold=2)
            print(f"sub-pilot {stem}: wall {got[stem][1]['wall']:.0f} s", flush=True)
        sub = sub_pilot(got, j_ref, d_qss, n_sub)
    tag = "PROXY-LABEL PILOT" if j_opt is None else "TO-OCP LABELS (pilot scale)"
    fig_fidelity(res, out / "pilot-audit-fidelity.png", tag, "J_best" if j_opt is None else "J*_20")
    fig_reliability(res, sub, out / "pilot-audit-reliability.png", args.windows, tag)
    report(args, plan, res, d_qss, d_centre, sub, out, skipped)
    print(f"wrote {out / 'pilot-audit.md'} ({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
