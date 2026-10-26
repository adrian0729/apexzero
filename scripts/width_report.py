"""GW-2's width report: generated strip widths against the R-dev circuits as evaluated.

    uv run python scripts/width_report.py                       # default, copied and 2a
    uv run python scripts/width_report.py --models 2b --tag 2b  # model 2b, same bands

Exploratory. The statistics and the bands (what counts as matching R-dev) are fixed in
docs/findings/width-model-plan.md, which must be committed, unchanged, before this runs
(``check_plan``). Only the 12 R-dev circuits are read, each through ``lockbox_guard``.

**Folds.** The committed leave-out folds (width-test-loo-plan.md): fold f1's widths are
R_DEV[0::2] and fold f2's R_DEV[1::2]. A width model fitted on (or copying) fold f's
circuits F is judged against the other fold's circuits H, which it never saw. The
reference is F itself: a statistic matches when the generator is no further from H than F
is, plus a fixed slack (``SLACK``; ``judge``). Curvature always comes from the grammar fitted
on all 12 circuits, as in training. Both variants are judged: v1 (base grammar) and v3
(every odd strip from ``tilted(g, alpha=2)``, the other half from the base grammar). The
same keys are used for every model, so all models share each strip's curvature. The fit on
all 12 circuits (the candidate for ``generator-v1``) is reported in-sample, not judged.

**Statistics** (2 m grid; real circuits as ``refine_reference`` re-expresses them):
- W1 of per-side widths over all knots (sides pooled), and of the inside and outside widths
  at tight-corner knots (|kappa| >= 1/50 1/m; inside = the side the knot turns toward);
- the share of tight-corner knots whose inside width ranges over >= 1 m, and >= 2 m, in the
  100 m up to and including the knot (51 knots; open strips skip their first 50 knots);
- apex dip and outside widening by radius bin: per ``corners.segment`` corner, the inside
  (outside) width at the apex (max |kappa| smoothed over 10 m) minus that side's median over
  the 100 m before and the 100 m after the corner (corners of open strips whose base would
  leave the strip are skipped); the median per bin (<25, 25-50, 50-100, 100-200, >=200 m);
- per-side width spectra in octave bands from 8 m to 1,024 m (Welch: 2,048 m segments every
  1,024 m, wrapped on closed circuits, linear detrend, periodic Hann window, both sides
  averaged; band power is variance in m^2);
- the no-copy check: structural (no array of the width model has more than 64 values) and
  empirical (the share of non-trivial 200 m windows of the strips, both sides, that lie
  within 0.05 m RMS of any 200 m window of any R-dev circuit in either side order).

Writes outputs/width-report/report.json (or report-<tag>.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import wasserstein_distance

from apexzero.tracks.corners import R_DEV, segment, smooth
from apexzero.tracks.generator import fit_grammar, generate, tilted
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.tracks.widths import grid_profile, max_leaf_size, profile_track
from apexzero.viz.export import lockbox_guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs" / "findings" / "width-model-plan.md"
OUT = ROOT / "outputs" / "width-report"
TUMFTM = ROOT / "data" / "tracks" / "tumftm"

FOLDS = {"f1": R_DEV[0::2], "f2": R_DEV[1::2]}      # widths a fold's model is fitted on
HELD_OUT = {"f1": FOLDS["f2"], "f2": FOLDS["f1"]}   # circuits it is judged on
VARIANTS = ("v1", "v3")
TILT_ALPHA = 2.0
SEED = 20261026
N_STRIPS = 400
DS = 2.0

K_TIGHT = 1.0 / 50.0
RANGE_KNOTS = 50                                    # 100 m before the knot
THRESHOLDS = (1.0, 2.0)
RADIUS_EDGES = (25.0, 50.0, 100.0, 200.0)
RADIUS_LABELS = ("<25", "25-50", "50-100", "100-200", ">=200")
BASE_KNOTS = 50
SEG, HOP = 1024, 512
SPEC_EDGES = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
SPEC_LABELS = ("8-16", "16-32", "32-64", "64-128", "128-256", "256-512", "512-1024")
JUDGED_SPEC_BANDS = 6                               # 8-512 m; 512-1,024 m is reported only

NOCOPY_LEN, NOCOPY_HOP, NOCOPY_STRIPS = 100, 100, 100
NOCOPY_RMS, NOCOPY_MIN_SD, NOCOPY_FLOOR = 0.05, 0.25, 0.001
REAL_QUERY_HOP = 25
MAX_LEAF = 64

MIN_BIN_CORNERS_H, MIN_BIN_CORNERS_X = 5, 20
SLACK = {"w1_all": 0.25, "w1_inside_tight": 0.25, "w1_outside_tight": 0.25,
         "share_1m": 0.10, "share_2m": 0.10, "apex_dip": 0.5, "apex_widen": 0.5,
         "spectrum": 1.0}


class Profile(NamedTuple):
    kappa: np.ndarray
    wl: np.ndarray
    wr: np.ndarray
    closed: bool
    name: str


# --------------------------------------------------------------------------- statistics


def window_range(w: np.ndarray, closed: bool, knots: int = RANGE_KNOTS) -> np.ndarray:
    """max - min of w over the knots + 1 knots ending at each knot (NaN where an open
    profile has fewer)."""
    w = np.asarray(w, float)
    if closed:
        v = sliding_window_view(np.r_[w[-knots:], w], knots + 1)
        return v.max(axis=1) - v.min(axis=1)
    v = sliding_window_view(w, knots + 1)
    return np.r_[np.full(knots, np.nan), v.max(axis=1) - v.min(axis=1)]


def side_samples(profiles: list[Profile]) -> dict[str, np.ndarray]:
    """Per-side widths at all knots (sides pooled), and inside and outside widths at
    tight-corner knots."""
    allk, w_in, w_out = [], [], []
    for p in profiles:
        allk += [p.wl, p.wr]
        t = np.abs(p.kappa) >= K_TIGHT
        left_in = p.kappa[t] > 0
        w_in.append(np.where(left_in, p.wl[t], p.wr[t]))
        w_out.append(np.where(left_in, p.wr[t], p.wl[t]))
    cat = lambda a: np.concatenate(a) if a else np.zeros(0)
    return {"all": cat(allk), "inside_tight": cat(w_in), "outside_tight": cat(w_out)}


def range_share(profiles: list[Profile]) -> dict:
    """Share of tight-corner knots whose inside width ranges over >= each threshold in the
    100 m up to and including the knot."""
    hits, total = np.zeros(len(THRESHOLDS)), 0
    for p in profiles:
        r_in = np.where(p.kappa > 0, window_range(p.wl, p.closed), window_range(p.wr, p.closed))
        m = (np.abs(p.kappa) >= K_TIGHT) & np.isfinite(r_in)
        total += int(m.sum())
        hits += [np.sum(r_in[m] >= t) for t in THRESHOLDS]
    share = hits / total if total else np.full(len(THRESHOLDS), np.nan)
    return {"n_knots": total, "share_1m": float(share[0]), "share_2m": float(share[1])}


def apex_rows(p: Profile) -> np.ndarray:
    """Rows (r_min, inside dip, outside widening) for the corners of one profile."""
    n = len(p.kappa)
    ks = np.abs(smooth(p.kappa, DS, 10.0, p.closed))
    rows = []
    for c in segment(profile_track(p.kappa, p.wl, p.wr, p.closed, DS)).corners:
        a, b = round(c.s_start / DS), round(c.s_end / DS)
        if p.closed:
            if b < a:
                b += n
            idx = np.arange(a, b + 1) % n
            around = np.r_[(a - np.arange(1, BASE_KNOTS + 1)) % n, (b + np.arange(1, BASE_KNOTS + 1)) % n]
        else:
            if a - BASE_KNOTS < 0 or b + BASE_KNOTS >= n:
                continue
            idx = np.arange(a, b + 1)
            around = np.r_[np.arange(a - BASE_KNOTS, a), np.arange(b + 1, b + 1 + BASE_KNOTS)]
        apex = idx[int(np.argmax(ks[idx]))]
        w_in, w_out = (p.wl, p.wr) if c.sign > 0 else (p.wr, p.wl)
        rows.append((c.r_min, w_in[apex] - np.median(w_in[around]),
                     w_out[apex] - np.median(w_out[around])))
    return np.asarray(rows, float).reshape(-1, 3)


def apex_table(profiles: list[Profile]) -> dict:
    """Median apex dip and outside widening, and corner counts, per radius bin."""
    rows = np.concatenate([apex_rows(p) for p in profiles]) if profiles else np.zeros((0, 3))
    b = np.searchsorted(np.asarray(RADIUS_EDGES), rows[:, 0], side="right")
    out = {}
    for i, lab in enumerate(RADIUS_LABELS):
        sel = rows[b == i]
        out[lab] = {"n": len(sel),
                    "dip": float(np.median(sel[:, 1])) if len(sel) else None,
                    "widen": float(np.median(sel[:, 2])) if len(sel) else None}
    return out


def _segments(x: np.ndarray, closed: bool) -> np.ndarray:
    n = len(x)
    starts = range(0, n, HOP) if closed else range(0, n - SEG + 1, HOP)
    return np.stack([x[(s + np.arange(SEG)) % n] for s in starts]) if len(starts) else np.zeros((0, SEG))


def band_power(segs: np.ndarray) -> np.ndarray:
    """Mean over segments of the variance per octave band (``SPEC_EDGES``): linear
    detrend, periodic Hann window, power normalised so a broadband signal's bands sum to
    its variance."""
    if len(segs) == 0:
        return np.full(len(SPEC_LABELS), np.nan)
    t = np.arange(SEG, dtype=float)
    tc = t - t.mean()
    x = segs - segs.mean(axis=1, keepdims=True)
    x = x - np.outer(x @ tc / (tc @ tc), tc)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * t / SEG)
    p = 2.0 * np.abs(np.fft.rfft(x * w, axis=1)) ** 2 / (SEG**2 * np.mean(w**2))
    p[:, -1] /= 2.0
    k = np.arange(p.shape[1])
    lam = np.where(k > 0, SEG * DS / np.maximum(k, 1), np.inf)
    band = np.searchsorted(np.asarray(SPEC_EDGES), lam, side="right") - 1
    ok = (k > 0) & (band >= 0) & (band < len(SPEC_LABELS))
    per = np.stack([np.bincount(band[ok], weights=row[ok], minlength=len(SPEC_LABELS)) for row in p])
    return per.mean(axis=0)


def spectra(profiles: list[Profile]) -> dict[str, list[float]]:
    """Per-side spectrum (both sides' segments pooled), and descriptive spectra of the
    total width and of half the left-right difference."""
    side = [_segments(w, p.closed) for p in profiles for w in (p.wl, p.wr)]
    tot = [_segments(p.wl + p.wr, p.closed) for p in profiles]
    dif = [_segments(0.5 * (p.wl - p.wr), p.closed) for p in profiles]
    cat = lambda a: np.concatenate(a) if a else np.zeros((0, SEG))
    return {"side": band_power(cat(side)).tolist(), "total": band_power(cat(tot)).tolist(),
            "half_difference": band_power(cat(dif)).tolist()}


def statistics(profiles: list[Profile]) -> dict:
    return {"samples": side_samples(profiles), "range": range_share(profiles),
            "apex": apex_table(profiles), "spectra": spectra(profiles)}


# ---------------------------------------------------------------------------- no-copy


def windows(p: Profile, starts, length: int = NOCOPY_LEN) -> np.ndarray:
    """[len(starts), 2, length] windows of (wl, wr); closed profiles wrap."""
    n = len(p.kappa)
    idx = (np.asarray(list(starts))[:, None] + np.arange(length)[None, :])
    idx = idx % n if p.closed else idx
    return np.stack([p.wl[idx], p.wr[idx]], axis=1)


def reference_windows(profiles: list[Profile], length: int = NOCOPY_LEN) -> np.ndarray:
    """Every window of every closed profile, in both side orders."""
    w = np.concatenate([windows(p, range(len(p.kappa)), length) for p in profiles])
    return np.concatenate([w, w[:, ::-1]])


def nearest_rms(queries: np.ndarray, refs: np.ndarray, chunk: int = 200) -> np.ndarray:
    """Per query, the smallest RMS difference to any reference window (both sides)."""
    q = queries.reshape(len(queries), -1).astype(float)
    r = refs.reshape(len(refs), -1).astype(float)
    rn = np.einsum("ij,ij->i", r, r)
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        qc = q[i:i + chunk]
        d2 = np.einsum("ij,ij->i", qc, qc)[:, None] + rn[None, :] - 2.0 * (qc @ r.T)
        out[i:i + chunk] = np.sqrt(np.clip(d2.min(axis=1), 0.0, None) / q.shape[1])
    return out


def nontrivial(w: np.ndarray) -> np.ndarray:
    return w.std(axis=2).max(axis=1) >= NOCOPY_MIN_SD


def copy_share(queries: np.ndarray, refs: np.ndarray) -> dict:
    keep = nontrivial(queries)
    d = nearest_rms(queries[keep], refs) if keep.any() else np.zeros(0)
    n = int(keep.sum())
    return {"n_windows": len(queries), "n_nontrivial": n,
            "share_within": float(np.mean(d < NOCOPY_RMS)) if n else float("nan"),
            "median_nearest_rms": float(np.median(d)) if n else float("nan")}


# ------------------------------------------------------------------------------ bands


def judge(kind: str, x: float, f: float, h: float | None, slack: float) -> bool:
    """Band rule. "w1": x and f are distances to H, x <= f + slack. "abs": x, f, h are
    values, |x - h| <= |f - h| + slack. "log2": band powers, |log2(x / h)| <= |log2(f / h)| +
    slack."""
    if kind == "w1":
        return bool(x <= f + slack)
    if kind == "abs":
        return bool(abs(x - h) <= abs(f - h) + slack)
    if kind == "log2":
        lg = lambda a: math.log2(max(a, 1e-12) / max(h, 1e-12))   # no power: far off
        return bool(abs(lg(x)) <= abs(lg(f)) + slack)
    raise ValueError(kind)


def band_checks(sx: dict, sf: dict, sh: dict) -> list[dict]:
    """Every banded check of generator statistics sx against held-out sh, with the fitting
    fold sf as the reference (plan, Section 4)."""
    out = []

    def add(stat, kind, x, f, h, judged=True, **extra):
        ok = judge(kind, x, f, h, SLACK[stat.split(":")[0]]) if judged else None
        out.append({"stat": stat, "rule": kind, "generator": x, "fitting_fold": f,
                    "held_out": h, "slack": SLACK[stat.split(":")[0]], "judged": judged,
                    "ok": ok, **extra})

    for key in ("all", "inside_tight", "outside_tight"):
        x = wasserstein_distance(sx["samples"][key], sh["samples"][key])
        f = wasserstein_distance(sf["samples"][key], sh["samples"][key])
        add(f"w1_{key}", "w1", float(x), float(f), None)
    for key in ("share_1m", "share_2m"):
        add(key, "abs", sx["range"][key], sf["range"][key], sh["range"][key])
    for lab in RADIUS_LABELS:
        bx, bf, bh = sx["apex"][lab], sf["apex"][lab], sh["apex"][lab]
        judged = bh["n"] >= MIN_BIN_CORNERS_H and bx["n"] >= MIN_BIN_CORNERS_X
        for key, stat in (("dip", "apex_dip"), ("widen", "apex_widen")):
            # A bin the fitting fold lacks has no reference distance: slack only (f = h).
            f = bf[key] if bf["n"] else bh[key]
            add(f"{stat}:{lab}", "abs", bx[key], f, bh[key], judged=judged and bx[key] is not None,
                n_generator=bx["n"], n_fitting=bf["n"], n_held_out=bh["n"])
    for i, lab in enumerate(SPEC_LABELS):
        add(f"spectrum:{lab}", "log2", sx["spectra"]["side"][i], sf["spectra"]["side"][i],
            sh["spectra"]["side"][i], judged=i < JUDGED_SPEC_BANDS)
    return out


# --------------------------------------------------------------------------- inputs


def real_profiles(names: list[str], cache: dict) -> list[Profile]:
    out = []
    for name in names:
        lockbox_guard(name)
        if name not in cache:
            tr = refine_reference(load_tumftm(TUMFTM / f"{name}.csv"))
            cache[name] = Profile(*grid_profile(tr, DS), True, name)
        out.append(cache[name])
    return out


def strips(g, variant: str, n: int = N_STRIPS, seed: int = SEED) -> list[Profile]:
    """n strips of a grammar: v1 all from g; v3 the odd ones from tilted(g, 2)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    # Model 2b solves a QP per strip, so its strips are generated in batches of 25.
    if g.reexpress_lams is None:
        gen = lambda gg: [np.asarray(a, float) for a in jax.jit(jax.vmap(lambda k: generate(k, gg)))(keys)]
    else:
        gen = lambda gg: [np.asarray(a, float) for a in jax.jit(
            lambda ks: jax.lax.map(lambda k: generate(k, gg), ks, batch_size=25))(keys)]
    k, wl, wr = gen(g)
    if variant == "v3":
        kt, wlt, wrt = gen(tilted(g, alpha=TILT_ALPHA))
        odd = (np.arange(n) % 2 == 1)[:, None]
        k, wl, wr = np.where(odd, kt, k), np.where(odd, wlt, wl), np.where(odd, wrt, wr)
    return [Profile(k[i], wl[i], wr[i], False, f"strip{i}") for i in range(n)]


def grammar(model: str, dev: list, width_tracks: list | None):
    """The grammar of a model: curvature from ``dev`` (raw), widths from ``width_tracks``
    (default all of ``dev``): evaluated tracks for copied and 2a, raw ones for 2b."""
    if model == "default":
        return fit_grammar(dev)
    if model == "copied":
        return fit_grammar(dev, real_widths=True, width_tracks=width_tracks)
    if model == "2a":
        return fit_grammar(dev, stat_widths=True, width_tracks=width_tracks)
    if model == "2b":
        return fit_grammar(dev, stat_widths=True, width_tracks=width_tracks, reexpress=True)
    raise SystemExit(f"unknown model {model!r}")


def model_leaf_size(g) -> int:
    if g.width_model is not None:
        return max_leaf_size(g.width_model)
    if g.real_w_left is not None:
        return int(np.size(g.real_w_left))
    return int(np.size(g.width))


def fitted_summary(g) -> dict | None:
    if g.width_model is None:
        return None
    return {f: np.asarray(v).tolist() for f, v in g.width_model._asdict().items()}


# ------------------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_plan() -> str:
    """The plan must be committed and unchanged; returns the commit that last touched it."""
    if not PLAN.exists() or not git("ls-files", str(PLAN.relative_to(ROOT))):
        raise SystemExit(f"{PLAN.relative_to(ROOT)} is not committed: fix the bands first")
    if git("status", "--porcelain", "--", str(PLAN.relative_to(ROOT))):
        raise SystemExit(f"{PLAN.relative_to(ROOT)} has uncommitted changes")
    return git("log", "-1", "--format=%H", "--", str(PLAN.relative_to(ROOT)))


def provenance(start: float, plan_commit: str, inputs: dict) -> dict:
    return {
        "label": "exploratory (docs/findings/width-model-plan.md)",
        "plan": str(PLAN.relative_to(ROOT)), "plan_commit": plan_commit,
        "plan_sha256": sha256_file(PLAN),
        "script": str(SCRIPT.relative_to(ROOT)), "script_sha256": sha256_file(SCRIPT),
        "git_head": git("rev-parse", "HEAD"),
        "changed_paths": [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines()
                          if ln.strip()],
        "src_clean": not git("status", "--porcelain", "--", "src").strip(),
        "inputs_sha256": dict(sorted(inputs.items())), "argv": sys.argv[1:],
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - start, 1),
        "environment": {"python": platform.python_version(), "jax": jax.__version__,
                        "numpy": np.__version__, "machine": platform.machine(),
                        "system": platform.system(), "nice": os.nice(0)},
    }


def clean(x):
    """JSON-safe: NaN as null, NumPy values as Python ones; sample arrays dropped."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items() if k != "samples"}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------------------- main


def timing(dev, models) -> dict:
    """Seconds to generate a 256-strip bank refresh (the training bank's per-update work)
    on this machine, jitted, median of 5 after a warm-up, for each model reported."""
    from apexzero.env.adapters import generate_bank

    out = {}
    for model in models:
        g = grammar(model, dev, None)
        f = jax.jit(lambda k, g=g: generate_bank(k, g, 256, tilted(g, alpha=TILT_ALPHA), 0.5))
        jax.block_until_ready(f(jax.random.PRNGKey(0)))
        ts = []
        for i in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(f(jax.random.PRNGKey(i + 1)))
            ts.append(time.perf_counter() - t0)
        out[model] = float(np.median(ts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="default,copied,2a")
    ap.add_argument("--tag", default="", help="output report-<tag>.json instead of report.json")
    ap.add_argument("--strips", type=int, default=N_STRIPS)
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args(argv)
    start = time.time()
    plan_commit = check_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / (f"report-{args.tag}.json" if args.tag else "report.json")
    if out_path.exists():
        raise SystemExit(f"{out_path.relative_to(ROOT)} exists: the report is computed once")
    models = [m for m in args.models.split(",") if m]

    for name in R_DEV:
        lockbox_guard(name)
    inputs = {f"data/tracks/tumftm/{n}.csv": sha256_file(TUMFTM / f"{n}.csv") for n in R_DEV}
    dev = [load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV]
    raw_by_name = dict(zip(R_DEV, dev, strict=True))
    cache: dict = {}
    real_all = real_profiles(R_DEV, cache)
    evaluated = {p.name: refine_reference(load_tumftm(TUMFTM / f"{p.name}.csv")) for p in real_all}
    sets = {"f1": FOLDS["f1"], "f2": FOLDS["f2"], "all": list(R_DEV)}
    real_stats = {k: statistics(real_profiles(v, cache)) for k, v in sets.items()}
    per_circuit = {p.name: {"range": range_share([p]), "apex": apex_table([p]),
                            "length_m": len(p.kappa) * DS} for p in real_all}
    refs_all = reference_windows(real_all)

    # The real reference for the no-copy rate: held-out circuits' windows against the
    # fitting fold's (for "all": each circuit against the other 11).
    real_copy = {}
    for f in ("f1", "f2"):
        q = np.concatenate([windows(p, range(0, len(p.kappa), REAL_QUERY_HOP))
                            for p in real_profiles(HELD_OUT[f], cache)])
        real_copy[f] = copy_share(q, reference_windows(real_profiles(FOLDS[f], cache)))
    d, n_all = [], 0
    for p in real_all:
        q = windows(p, range(0, len(p.kappa), REAL_QUERY_HOP))
        n_all += len(q)
        keep = nontrivial(q)
        d.append(nearest_rms(q[keep], reference_windows([o for o in real_all if o.name != p.name])))
    d = np.concatenate(d)
    real_copy["all"] = {"n_windows": n_all, "n_nontrivial": len(d),
                        "share_within": float(np.mean(d < NOCOPY_RMS)),
                        "median_nearest_rms": float(np.median(d))}

    results, fits, verdict = {}, {}, {}
    for model in models:
        results[model], fits[model] = {}, {}
        failed, checked = [], 0
        for fold in ("f1", "f2", "all"):
            source = raw_by_name if model == "2b" else evaluated   # 2b fits raw profiles
            wt = None if fold == "all" else [source[n] for n in FOLDS[fold]]
            g = grammar(model, dev, wt)
            fits[model][fold] = fitted_summary(g)
            results[model][fold] = {}
            for v in VARIANTS:
                prof = strips(g, v, args.strips)
                sx = statistics(prof)
                q = np.concatenate([windows(p, range(0, len(p.kappa) - NOCOPY_LEN + 1, NOCOPY_HOP))
                                    for p in prof[:NOCOPY_STRIPS]])
                cs = copy_share(q, refs_all)
                leaf = model_leaf_size(g)
                ref_rate = real_copy[fold]["share_within"]
                nocopy_ok = bool(leaf <= MAX_LEAF and cs["share_within"] <= max(NOCOPY_FLOOR, ref_rate))
                entry = {"stats": sx, "nocopy": {**cs, "max_leaf_size": leaf,
                                                 "real_reference_share": ref_rate, "ok": nocopy_ok}}
                if fold == "all":
                    sr = real_stats["all"]
                    entry["in_sample"] = {
                        "w1": {k: float(wasserstein_distance(sx["samples"][k], sr["samples"][k]))
                               for k in ("all", "inside_tight", "outside_tight")},
                        "range": {k: (sx["range"][k], sr["range"][k]) for k in ("share_1m", "share_2m")}}
                else:
                    entry["checks"] = band_checks(sx, real_stats[fold], real_stats[
                        "f2" if fold == "f1" else "f1"])
                    for c in entry["checks"]:
                        if c["judged"]:
                            checked += 1
                            if not c["ok"]:
                                failed.append(f"{fold}/{v}/{c['stat']}")
                checked += 1
                if not nocopy_ok:
                    failed.append(f"{fold}/{v}/nocopy")
                results[model][fold][v] = entry
                print(f"{model} {fold} {v}: share_1m {sx['range']['share_1m']:.3f} "
                      f"nocopy {cs['share_within']:.4f}", flush=True)
        verdict[model] = {"meets_bands": not failed, "n_checks": checked, "failed": failed}

    timing_s = None if args.no_timing else timing(dev, models)
    report = {"label": "exploratory", "folds": FOLDS, "held_out": HELD_OUT,
              "constants": {"seed": SEED, "n_strips": args.strips, "k_tight": K_TIGHT,
                            "range_knots": RANGE_KNOTS, "thresholds": THRESHOLDS,
                            "radius_edges": RADIUS_EDGES, "spec_edges": SPEC_EDGES,
                            "judged_spec_bands": JUDGED_SPEC_BANDS, "slack": SLACK,
                            "min_bin_corners": [MIN_BIN_CORNERS_H, MIN_BIN_CORNERS_X],
                            "nocopy": [NOCOPY_LEN, NOCOPY_HOP, NOCOPY_STRIPS, NOCOPY_RMS,
                                       NOCOPY_MIN_SD, NOCOPY_FLOOR, MAX_LEAF]},
              "real": {"sets": real_stats, "per_circuit": per_circuit, "copy_reference": real_copy},
              "models": results, "fits": fits, "verdict": verdict,
              "bank_refresh_seconds": timing_s,
              "provenance": provenance(start, plan_commit, inputs)}
    out_path.write_text(json.dumps(clean(report), indent=1))
    print(json.dumps(clean(verdict), indent=1))
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
