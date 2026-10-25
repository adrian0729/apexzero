"""PS-P4, option A: step 1's P4 made testable. EXPLORATORY (not confirmatory).

    nice -n 15 uv run python scripts/p4_exploratory.py <stage> [--workers 4]

Plan: docs/findings/p4-plan.md, committed in 916edaa before anything here was computed, and
its Amendment 1 (implementation details), committed with this script before the first stage
ran. Every number this script writes is exploratory. It cannot change step 1's confirmatory
P4 verdicts or the post-tag rerun (psychophysics-step1.md, Sections 4.1 and 4.4).

**Question.** P4's class 2 needs every F6 ("irrelevant") perturbation at least as far
off-manifold (d_OOD) as a failing single-factor F1 condition to complete; step 1's F6 rows
reach d_OOD of only 0.7-1.0, so S2's and S3's class 2 held vacuously. This builds larger
perturbations that are certified irrelevant (the corner's minimum-time optimum is unchanged
and the generalists complete them), runs the specialists on them, and reapplies P4's ordered
rules with the F6 set extended (``classify``, a copy of the frozen ``p4_classify`` with the F6
set as a parameter).

**Stages** (run in this order; each refuses to run before the ones it needs):

- ``build``: the 38 candidate conditions on Hockenheim's 20 tier D corners (F6s: the frozen
  builder at f6_scale 1-8; K-str: kappa noise on straights; W+: widening only; W+str: the
  contingency's widening on K-str's footprint, built now and used only if the contingency
  triggers), the registered tier D rows copied from step 1's bank, cells copying each owner's
  condition (a) cell, and the built-in check that the frozen builder at f6_scale 0.5
  reproduces step 1's F6 strips bit for bit.
- ``certify-socp``: the pinned s-OCP of each braking corner and condition from S1's tier D
  state (frozen ``answer_key``, float64 workers). Condition (a) must reproduce step 1's key to
  a relative 1e-6; the first 10 solves are timed (plan Section 9's cut rule).
- ``certify-generalists``: the 9 generalist finals on (a) and the isolated road (reference),
  then on each condition's s-OCP-certified corners; the median C_G rule (with its fallback,
  fixed on (a) first) and W+'s contingency rule.
- ``freeze``: outputs/p4/certified.json (to be committed before ``specialists``).
- ``specialists``: S1-S3 rerun the 23 registered conditions (the reproduction check against
  decisions_all.json), then every certified condition on its certified corners; d_OOD from
  each specialist's own buffer.
- ``classify``: P4's ordered rules with the extended F6 set, per family, the recipe claim.
- ``report``: the readings of expected directions E1-E6 and the C against d_OOD tables.

**Integrity.** src/apexzero/interp and scripts/psychophysics.py are imported read-only; every
stage refuses to run when either differs from HEAD or from the tag psychophysics-prereg-v2.
Nothing is written under outputs/psychophysics/; outputs go to outputs/p4/. Only Hockenheim
(R_DEV) is built or driven, through ``viz.export.lockbox_guard``; outputs/sealed/ is never
read. Rollout chunks are padded to step 1's full chunk of 2048 runs (XLA's float32 results
depend on the batch shape; at 2048 the stored runs are reproduced bit for bit).
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
from dataclasses import replace
from pathlib import Path

# XLA reads its flags at import: 2 threads per process, as step 1's harness ran.
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=2")

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

from apexzero.interp import analysis as A
from apexzero.interp import io as pio
from apexzero.interp import registry
from apexzero.interp import stimuli as ST
from apexzero.interp.constants import (
    CHUNK,
    DS,
    F6_KAPPA_NOISE,
    F6_WOBBLE_AMPL,
    F6_WOBBLE_WAVELENGTH,
    N_KNOTS,
    ROOT_SEED,
    STREAM_OFFSET,
    STREAMS,
)
from apexzero.interp.stimuli import COND_A, COND_ISO, D_CONDITIONS, D_TRACK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p4"
BUILD, SOCP, GEN, SPEC = OUT / "build", OUT / "socp", OUT / "generalists", OUT / "specialists"
CERTIFIED = OUT / "certified.json"
PLAN = ROOT / "docs" / "findings" / "p4-plan.md"
PLAN_COMMIT = "916edaa"
TAG = "psychophysics-prereg-v2"
FROZEN = ("src/apexzero/interp", "scripts/psychophysics.py")
DECISIONS_ALL = pio.ANALYSIS_DIR / "decisions_all.json"
DECISIONS_ALL_SHA256 = "14c674863e78b68cf01986bddcaec34480700ea21811aa0971550711feca4ea1"
KEY_TABLE = pio.ANALYSIS_DIR / "key_table.npz"
SWAP = pio.STIMULI_DIR / "swap.json"

PREFIX = "P4A"
F6_TAILS = tuple(c.split("/", 1)[1] for c in A.F6_CONDS)    # wobble0.3 ... knoise1e-3
F6_NOMINAL = (*F6_WOBBLE_AMPL, *F6_KAPPA_NOISE)              # m, m, 1/m, 1/m
F6S_SCALES = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
F6S_SCALES_CUT = (1.0, 2.0, 4.0, 8.0)                       # plan Section 9's timing cut
KSTR_LEVELS = (1e-3, 2e-3, 4e-3, 8e-3)                      # 1/m, peak
KSTR_TAGS = ("1e-3", "2e-3", "4e-3", "8e-3")
WPLUS_AMPL = (0.5, 1.0, 2.0, 3.0, 4.0)                      # m
FAMILIES = ("F6s", "Kstr", "Wplus", "WplusStr")
FAMILY_INDEX = {"Kstr": 1, "Wplus": 2, "WplusStr": 3}       # the new stream's family id
P4_STREAM = 401              # STREAM_OFFSET + 401: differs from every frozen stream offset
STRAIGHT_MARGIN = 50.0       # m, each segment() corner extended on both sides
STRAIGHT_MIN = 200.0         # m, a straight run must be longer (the band's long end)
WOBBLE_FAMILIES = ("wobble", "widening")

SOCP_TOL = 0.01              # |v*_min(cond) / v*_min(a) - 1| <= 1%
SOCP_REPRO_RTOL = 1e-6       # condition (a) re-solve vs step 1's key (relative)
GEN_MEDIAN_MIN = 0.9         # median C_G >= 0.9
GEN_FALLBACK = 0.1           # else median C_G(cond) >= median C_G(a) - 0.1
MIN_CORNERS = 7              # certified at >= 7 of the 9 braking corners
TIMING_N, TIMING_MAX_S = 10, 120.0
DOOD_TOL = 1e-6              # registered d_OOD vs decisions_all.json (absolute)
MAX_WORKERS = 4
NICE = 15
OWNERS = ("S1", "S2", "S3")  # whose tier D state a cell starts from (S1's: the generalists')


class CheckFailed(SystemExit):
    """A built-in check failed: the item stops for an amendment (plan Section 10)."""


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise CheckFailed(1)


# ------------------------------------------------------------------------- provenance


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def frozen_problems() -> list[str]:
    """The frozen harness must equal HEAD and the tag (plan Section 9)."""
    out = []
    st = git("status", "--porcelain", "--", *FROZEN)
    if st:
        out.append(f"differs from HEAD: {st}")
    r = subprocess.run(["git", "diff", "--quiet", TAG, "HEAD", "--", *FROZEN], cwd=ROOT,
                       check=False)
    if r.returncode != 0:
        out.append(f"changed since {TAG}")
    return out


def require_frozen() -> None:
    probs = frozen_problems()
    if probs:
        raise SystemExit("frozen harness changed: " + "; ".join(probs))


def provenance(stage: str, argv: list[str], t0: float, inputs: dict | None = None) -> dict:
    script = Path(__file__).resolve()
    return {"stage": stage, "script": rel(script), "script_sha256": sha256_file(script),
            "git_head": git("rev-parse", "HEAD"),
            "changed_paths": [ln for ln in git("status", "--porcelain").splitlines() if ln],
            "frozen_clean": not frozen_problems(), "argv": argv,
            "plan": {"path": rel(PLAN), "commit_prefix": PLAN_COMMIT,
                     "sha256": sha256_file(PLAN)},
            "inputs_sha256": inputs or {}, "seconds": round(time.time() - t0, 1),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system(),
                            "XLA_FLAGS": os.environ.get("XLA_FLAGS", "")}}


def clean(x):
    """JSON-safe: NaN and infinities as null, NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=1, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def be_nice() -> None:
    try:
        cur = os.nice(0)
        if cur < NICE:
            os.nice(NICE - cur)
    except OSError:
        pass


# ------------------------------------------------------------------------- the conditions


def condition_catalogue() -> list[dict]:
    """The 38 candidate conditions of plan Section 3, in job order (F6s by scale, then K-str,
    W+, and W+str, the contingency's, last)."""
    out = []
    for s in F6S_SCALES:
        for tail, nom in zip(F6_TAILS, F6_NOMINAL, strict=True):
            wob = tail.startswith("wobble")
            out.append(dict(name=f"{PREFIX}/F6s/{tail}/x{s:g}", family="F6s",
                            ladder=f"F6s/{tail}", kind="wobble" if wob else "knoise",
                            amplitude=nom * s, unit="m" if wob else "1/m", f6_scale=s,
                            level_index=-1, contingency=False))
    for li, (lv, tag) in enumerate(zip(KSTR_LEVELS, KSTR_TAGS, strict=True)):
        out.append(dict(name=f"{PREFIX}/Kstr/{tag}", family="Kstr", ladder="Kstr",
                        kind="knoise", amplitude=lv, unit="1/m", f6_scale=None,
                        level_index=li, contingency=False))
    for fam in ("Wplus", "WplusStr"):
        for li, a in enumerate(WPLUS_AMPL):
            out.append(dict(name=f"{PREFIX}/{fam}/{a:g}m", family=fam, ladder=fam,
                            kind="widening", amplitude=a, unit="m", f6_scale=None,
                            level_index=li, contingency=fam == "WplusStr"))
    return out


CATALOGUE = condition_catalogue()
BY_NAME = {c["name"]: c for c in CATALOGUE}
MAIN_CATALOGUE = CATALOGUE

# Amendment 2 (post hoc): the plan's one ladder extension, K-str's rungs continued.
EXT_KSTR_LEVELS = (16e-3, 32e-3, 64e-3)
EXT_KSTR_TAGS = ("16e-3", "32e-3", "64e-3")
EXTENSION_DIR = OUT / "extension"


def extension_catalogue() -> list[dict]:
    """K-str at 16e-3, 32e-3 and 64e-3 1/m: positions 4-6 of the extended K-str ladder."""
    n0 = len(KSTR_LEVELS)
    return [dict(name=f"{PREFIX}/Kstr/{tag}", family="Kstr", ladder="Kstr", kind="knoise",
                 amplitude=lv, unit="1/m", f6_scale=None, level_index=n0 + li,
                 contingency=False)
            for li, (lv, tag) in enumerate(zip(EXT_KSTR_LEVELS, EXT_KSTR_TAGS, strict=True))]


EXTENSION = False
OUT_DIR = OUT


def configure(extension: bool) -> None:
    """Point every stage at the planned run (outputs/p4/) or at Amendment 2's ladder
    extension (outputs/p4/extension/, its own build, certification, freeze and runs)."""
    global EXTENSION, OUT_DIR, BUILD, SOCP, GEN, SPEC, CERTIFIED, CATALOGUE, BY_NAME
    EXTENSION = bool(extension)
    OUT_DIR = EXTENSION_DIR if EXTENSION else OUT
    BUILD, SOCP, GEN, SPEC = (OUT_DIR / d for d in ("build", "socp", "generalists",
                                                     "specialists"))
    CERTIFIED = OUT_DIR / "certified.json"
    CATALOGUE = extension_catalogue() if EXTENSION else MAIN_CATALOGUE
    BY_NAME = {c["name"]: c for c in CATALOGUE}


def p4_rng(corner: int, family: str, level_index: int) -> np.random.Generator:
    """The new stream (plan Section 3): never the frozen 'f6' stream's draws."""
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + P4_STREAM, int(corner),
                                  FAMILY_INDEX[family], int(level_index)])


def straight_runs(n: int, corner_spans: list[tuple[float, float]], lead1: int, i_end: int,
                  ds: float = DS, margin: float = STRAIGHT_MARGIN,
                  min_len: float = STRAIGHT_MIN) -> list[tuple[int, int]]:
    """K-str's straights: maximal runs [a, b] of strip knots after the lead-in (> lead1) and
    up to ``i_end`` (the frozen kappa noise's range end, the excision's last knot) that lie
    outside every segment() corner span (s_start, s_end + ds) extended by ``margin`` m on both
    sides, kept when (b - a) ds > ``min_len``."""
    i = np.arange(n)
    s = i * ds
    ok = (i > lead1) & (i <= i_end)
    for a, b in corner_spans:
        ok &= ~((s >= a - margin) & (s <= b + margin))
    runs = []
    j = 0
    while j < n:
        if not ok[j]:
            j += 1
            continue
        k = j
        while k + 1 < n and ok[k + 1]:
            k += 1
        if (k - j) * ds > min_len:
            runs.append((int(j), int(k)))
        j = k + 1
    return runs


def kstr_noise(rng: np.random.Generator, n: int, runs: list[tuple[int, int]],
               level: float) -> np.ndarray:
    """The frozen ``_kappa_noise`` applied separately on each straight run (a, b) (its own
    zero-net-heading window, 20 m tapers inside the run, peak ``level``), runs drawn in order
    of position from one generator."""
    i = np.arange(n)
    out = np.zeros(n)
    for a, b in runs:
        out += ST._kappa_noise(rng, i, a, b, level)
    return out


def run_window(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    """K-str's footprint as a weight: the frozen kappa noise's window on each run (0 at its
    ends, 20 m ramps, 1 inside), 0 elsewhere."""
    i = np.arange(n)
    w = np.zeros(n)
    for a, b in runs:
        ww = ST._ramp_up(i, a + ST.TAPER_KNOTS) * ST._ramp_down(i, b - ST.TAPER_KNOTS)
        w = np.maximum(w, np.where((i > a) & (i < b), ww, 0.0))
    return w


def widening(n: int, amplitude: float, phi: float, footprint: np.ndarray) -> np.ndarray:
    """A max(0, sin(2 pi s / 60 + phi)) on the footprint: the frozen wobble's wavelength and
    knot positions (s = i ds) with its narrowing half removed."""
    s = np.arange(n) * DS
    return amplitude * np.maximum(0.0, np.sin(2.0 * np.pi * s / F6_WOBBLE_WAVELENGTH + phi)) \
        * footprint


def widen(kappa: np.ndarray, w_left: np.ndarray, w_right: np.ndarray, add: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both edges moved outward by ``add`` >= 0, then the frozen inside cap on the modified
    knots (add > 0). Raises if any knot ends narrower than the road it widens."""
    if np.any(add < 0):
        raise ValueError("widening only: negative amplitude")
    wl2, wr2, cap = ST.apply_inside_cap(kappa, w_left + add, w_right + add, mask=add > 0)
    if np.any(wl2 < w_left) or np.any(wr2 < w_right):
        raise ValueError("the inside cap narrowed a widened knot below its road")
    return wl2, wr2, cap


def hockenheim_track(name: str = D_TRACK):
    """The only circuit this item builds or drives: ``lockbox_guard`` first, then Hockenheim."""
    from apexzero.viz.export import lockbox_guard

    lockbox_guard(name)
    if name != D_TRACK:
        raise ValueError(f"PS-P4 uses {D_TRACK} only, not {name!r}")
    return ST.real_track(name)


def strip_track(kappa, w_left, w_right, name: str):
    """The NumPy Track of one strip as the bank gives it (``StimulusBank.track``)."""
    import jax.numpy as jnp

    from apexzero.tracks.generator import strip_arrays, to_track

    tr = strip_arrays(jnp.asarray(kappa, jnp.float32), jnp.asarray(w_left, jnp.float32),
                      jnp.asarray(w_right, jnp.float32), DS)
    return to_track(tr, name)


def new_conditions(dc: ST.DCorner, track, swap_j: int | None) -> tuple[dict, dict]:
    """Profiles of the 38 candidate conditions of one corner and their footprint record."""
    from apexzero.tracks.corners import segment

    i = dc.index
    rr = ST._RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
    k, wl, wr = rr.real()
    lm = rr.landmarks(k)
    w_lead = rr.w_lead()
    mod = w_lead < 1.0
    n = N_KNOTS
    profiles: dict[str, ST.Profile] = {}
    info: dict = {"lead1": int(rr.lead1), "i_b": int(rr.i_b), "c0": int(rr.c0),
                  "c1": int(rr.c1)}
    for s in F6S_SCALES:
        conds, _ = ST.hockenheim_conditions(i, track, swap_j, float(s))
        for tail in F6_TAILS:
            profiles[f"{PREFIX}/F6s/{tail}/x{s:g}"] = conds[f"F6/{tail}"]
    corners = segment(strip_track(k, wl, wr, f"D/{i}/{COND_A}")).corners
    spans = [(float(c.s_start), float(c.s_end + DS)) for c in corners]
    runs = straight_runs(n, spans, rr.lead1, rr.i_b)
    win = run_window(n, runs)
    lo, hi = lm.s_c0 - STRAIGHT_MARGIN, lm.s_c1 + STRAIGHT_MARGIN
    if np.any(win[(np.arange(n) * DS >= lo) & (np.arange(n) * DS <= hi)] > 0):
        fail(f"corner {i}: K-str's footprint reaches the scored corner's region")
    info.update(straight_runs=[[a, b] for a, b in runs],
                straight_m=[(b - a) * DS for a, b in runs],
                segment_corners=len(corners))
    for li, lv in enumerate(KSTR_LEVELS):
        kk = k + kstr_noise(p4_rng(i, "Kstr", li), n, runs, lv)
        wl2, wr2, cap = ST.apply_inside_cap(kk, wl, wr, mask=mod)
        profiles[f"{PREFIX}/Kstr/{KSTR_TAGS[li]}"] = ST.Profile(kk, wl2, wr2, lm, cap)
    phis = {}
    for fam, foot in (("Wplus", 1.0 - w_lead), ("WplusStr", (1.0 - w_lead) * win)):
        for li, a in enumerate(WPLUS_AMPL):
            phi = float(p4_rng(i, fam, li).uniform(0.0, 2.0 * np.pi))
            phis[f"{fam}/{a:g}"] = phi
            try:
                wl2, wr2, cap = widen(k, wl, wr, widening(n, a, phi, foot))
            except ValueError as e:
                fail(f"corner {i} {fam} {a:g} m: {e}")
            profiles[f"{PREFIX}/{fam}/{a:g}m"] = ST.Profile(k, wl2, wr2, lm, cap)
    info["phi"] = phis
    if set(profiles) != {c["name"] for c in MAIN_CATALOGUE}:
        fail("the builder and the catalogue disagree")
    return profiles, info


def kstr_profile(corner: int, kappa, w_left, w_right, lm, mod, runs, level: float,
                 level_index: int) -> ST.Profile:
    """One K-str strip: kappa noise on the straight runs from the new stream, then the
    frozen inside cap on the modified knots (as ``new_conditions`` builds it)."""
    kk = kappa + kstr_noise(p4_rng(corner, "Kstr", level_index), len(kappa), runs, level)
    wl2, wr2, cap = ST.apply_inside_cap(kk, w_left, w_right, mask=mod)
    return ST.Profile(kk, wl2, wr2, lm, cap)


def extension_conditions(dc: ST.DCorner, track) -> tuple[dict, dict]:
    """Amendment 2's K-str rungs on one corner: the planned K-str footprint, noise and
    stream (level index = position in the extended ladder)."""
    from apexzero.tracks.corners import segment

    i = dc.index
    rr = ST._RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
    k, wl, wr = rr.real()
    lm = rr.landmarks(k)
    mod = rr.w_lead() < 1.0
    corners = segment(strip_track(k, wl, wr, f"D/{i}/{COND_A}")).corners
    spans = [(float(c.s_start), float(c.s_end + DS)) for c in corners]
    runs = straight_runs(N_KNOTS, spans, rr.lead1, rr.i_b)
    profiles = {c["name"]: kstr_profile(i, k, wl, wr, lm, mod, runs, c["amplitude"],
                                        c["level_index"]) for c in extension_catalogue()}
    info = {"lead1": int(rr.lead1), "i_b": int(rr.i_b), "c0": int(rr.c0), "c1": int(rr.c1),
            "straight_runs": [[a, b] for a, b in runs],
            "straight_m": [(b - a) * DS for a, b in runs], "segment_corners": len(corners)}
    return profiles, info


# ------------------------------------------------------------------------- inputs


def stored_design():
    return ST.load_design(pio.STIMULI_DIR)


def decisions_p4() -> dict[str, dict]:
    raw = DECISIONS_ALL.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != DECISIONS_ALL_SHA256:
        fail(f"decisions_all.json is not step 1's post-tag rerun ({sha})")
    d = json.loads(raw)
    return {r["agent"]: r for r in d["decisions"] if r["prediction"] in ("P4", "P4-recipe")}


def braking_from_keys(kt: dict) -> list[int]:
    """The braking corners as ``cond_table`` reads them: the s-OCP brakes in condition (a)."""
    corner, cond, b = A._it(kt, "corner"), A._st(kt, "condition"), A._bo(kt, "braking_socp")
    return sorted({int(c) for c, n, x in zip(corner, cond, b, strict=True)
                   if c >= 0 and n == COND_A and x})


def finals() -> dict:
    from apexzero.interp import agents

    m = agents.load_manifest(pio.MANIFEST_PATH)
    return {e.agent: e for e in m.snapshots if e.is_final and not e.error}


def load_build():
    if not (BUILD / "cells.json").exists():
        raise SystemExit("run the build stage first")
    bank, cells = ST.load_design(BUILD)
    meta = read_json(BUILD / "cells_meta.json")["cells"]
    return bank, cells, meta


def owner_cells(meta: list[dict], owner: str, conds, corners) -> list[int]:
    cs, cr = set(conds), set(corners)
    return [m["cell"] for m in meta if m["owner"] == owner and m["condition"] in cs
            and m["corner"] in cr]


# ------------------------------------------------------------------------- build


def stage_build(args) -> None:
    import jax.numpy as jnp  # noqa: F401  (float32 process)

    t0 = time.time()
    require_frozen()
    if (BUILD / "cells.json").exists() and not args.force:
        raise SystemExit("already built (--force rebuilds; certification depends on it)")
    sbank, scells = stored_design()
    kt = pio.load_table(KEY_TABLE)
    braking = braking_from_keys(kt)
    track = hockenheim_track()
    swap = {int(k): int(v) for k, v in pio.read_json(SWAP).items()}
    dcs = ST.hockenheim_corners(track)
    reg_rows: list[int] = []                 # stored stim ids, in bank order
    new_rows: list[ST._Row] = []
    checks = {"f6_scale_0.5_vs_stored": {}, "registered_vs_stored": {}, "a_road": {}}
    footprint = {}
    for dc in dcs:
        i = dc.index
        sw = swap.get(i)
        conds05, _ = ST.hockenheim_conditions(i, track, sw, 0.5)
        for cond in D_CONDITIONS:
            name = f"D/{i}/{cond}"
            if cond not in conds05:
                continue
            st = sbank.by_name(name)
            p = conds05[cond]
            eq = all(np.array_equal(getattr(p, f), getattr(sbank, f)[st.stim_id])
                     for f in ("kappa", "w_left", "w_right"))
            key = "f6_scale_0.5_vs_stored" if cond.startswith("F6/") else "registered_vs_stored"
            checks[key][name] = bool(eq)
            reg_rows.append(st.stim_id)
        if EXTENSION:
            profiles, info = extension_conditions(dc, track)
        else:
            profiles, info = new_conditions(dc, track, sw)
        a = sbank.by_name(f"D/{i}/{COND_A}").stim_id
        rr = ST._RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
        k, wl, wr = rr.real()
        checks["a_road"][str(i)] = bool(np.array_equal(k, sbank.kappa[a])
                                        and np.array_equal(wl, sbank.w_left[a])
                                        and np.array_equal(wr, sbank.w_right[a]))
        footprint[str(i)] = info
        for c in CATALOGUE:
            p = profiles[c["name"]]
            src = rr.source_info(condition=c["name"], corner_index=i, D_gap=dc.D_gap,
                                 key=f"D/{i}", family=c["family"], amplitude=c["amplitude"],
                                 unit=c["unit"], p4_exploratory=True,
                                 **({"f6_scale": c["f6_scale"]} if c["f6_scale"] else {}))
            new_rows.append(ST._Row(f"D/{i}/{c['name']}", p, "hockenheim", dc.corner.sign,
                                    None, {"D"}, src, {"cap_binds"} if p.capped.any() else set()))
    if not all(checks["f6_scale_0.5_vs_stored"].values()):
        bad = [k for k, v in checks["f6_scale_0.5_vs_stored"].items() if not v]
        fail(f"the frozen builder at f6_scale 0.5 does not reproduce step 1's F6 strips: {bad}")
    if not all(checks["a_road"].values()):
        fail("condition (a)'s road differs from step 1's bank")
    print(f"built {len(new_rows)} new strips; F6 at 0.5 reproduces step 1's "
          f"{len(checks['f6_scale_0.5_vs_stored'])} F6 strips bit for bit", flush=True)

    # The bank: step 1's tier D rows (copied), then the new rows.
    idmap = {s: j for j, s in enumerate(reg_rows)}
    reg_stims = [replace(sbank.stimuli[s], stim_id=idmap[s],
                         mirror_id=idmap.get(sbank.stimuli[s].mirror_id, -1)) for s in reg_rows]
    new_stims, _ = ST._assemble(new_rows, start_id=len(reg_rows))
    bank = ST.StimulusBank(
        np.concatenate([sbank.kappa[reg_rows], np.stack([r.profile.kappa for r in new_rows])]),
        np.concatenate([sbank.w_left[reg_rows], np.stack([r.profile.w_left for r in new_rows])]),
        np.concatenate([sbank.w_right[reg_rows],
                        np.stack([r.profile.w_right for r in new_rows])]),
        reg_stims + new_stims, sbank.ds)

    # Cells: per braking corner and owner, step 1's registered cells and copies of the
    # owner's condition (a) cell on every new strip. Order: corner, owner (S1 first, as step
    # 1's key table), condition.
    owner_runners = {"S1": ST.RUNNERS_FROM_S1, "S2": ("S2",), "S3": ("S3",)}
    by_name_row = {s.name: s.stim_id for s in bank.stimuli}
    cells, meta = [], []
    pair = max(c.pair_id for c in scells) + 1
    for i in braking:
        for owner in OWNERS:
            mine = {sbank.stimuli[c.stim_id].name: c for c in scells
                    if "D" in c.tiers and tuple(c.runners) == tuple(owner_runners[owner])
                    and int(sbank.stimuli[c.stim_id].source.get("corner_index", -1)) == i}
            base = mine[f"D/{i}/{COND_A}"]
            for cond in D_CONDITIONS:
                c = mine.get(f"D/{i}/{cond}")
                if c is None:
                    continue
                cid = len(cells)
                cells.append(replace(c, cell_id=cid, stim_id=idmap[c.stim_id]))
                meta.append(dict(cell=cid, owner=owner, corner=i, condition=cond,
                                 registered=True, stored_cell=int(c.cell_id)))
            for cat in CATALOGUE:
                sid = by_name_row[f"D/{i}/{cat['name']}"]
                cid = len(cells)
                flags = tuple(sorted(set(base.flags) | set(bank.stimuli[sid].flags)))
                cells.append(replace(base, cell_id=cid, stim_id=sid, pair_id=pair,
                                     mirrored=False, flags=flags))
                meta.append(dict(cell=cid, owner=owner, corner=i, condition=cat["name"],
                                 registered=False, stored_cell=-1))
                pair += 1
    strips = {}
    for s in bank.stimuli:
        strips[s.name] = pio.sha256_arrays(bank.kappa[s.stim_id], bank.w_left[s.stim_id],
                                           bank.w_right[s.stim_id])
    bank.save(BUILD)
    ST.save_cells(BUILD / "cells.json", cells)
    write_json(BUILD / "cells_meta.json", {"cells": meta, "braking": braking})
    cap_new = {r.name: int(r.profile.capped.sum()) for r in new_rows if r.profile.capped.any()}
    write_json(BUILD / "build.json", {
        "status": "EXPLORATORY (p4-plan.md): stimuli only; no agent or solver has run",
        "catalogue": CATALOGUE, "braking_corners": braking, "n_new_rows": len(new_rows),
        "n_registered_rows": len(reg_rows), "n_cells": len(cells),
        "design_hash": ST.design_hash(bank, cells), "checks": checks,
        "footprints": footprint, "strip_sha256": strips, "capped_knots_new": cap_new,
        "provenance": provenance("build", sys.argv[1:], t0, {
            "bank.npz": sha256_file(pio.STIMULI_DIR / "bank.npz"),
            "cells.json": sha256_file(pio.STIMULI_DIR / "cells.json"),
            "swap.json": sha256_file(SWAP), "key_table.npz": sha256_file(KEY_TABLE)})})
    print(f"wrote {rel(BUILD)}: {len(bank)} rows, {len(cells)} cells; braking corners "
          f"{braking}", flush=True)


# ------------------------------------------------------------------------- pools


_W: dict = {}


def _pool(jobs: list[dict], fn, init, workers: int, on_result=None) -> list[dict]:
    """A spawned pool of at most 4 processes; each worker is configured like this process
    (the planned run or Amendment 2's extension) before its initializer loads the build."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not jobs:
        return []
    out = []
    n = max(1, min(int(workers), MAX_WORKERS, len(jobs)))
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"),
                             initializer=init, initargs=(EXTENSION,)) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            out.append(r)
            if on_result is not None:
                on_result(r)
    return out


# ------------------------------------------------------------------------- s-OCP


def _socp_init(extension: bool = False) -> None:
    from apexzero.interp import answer_key as ak
    from apexzero.interp import measures

    configure(extension)
    be_nice()
    ak._x64()
    if not measures.tables_path().exists():
        raise RuntimeError("step 1's model tables are missing (they would be built under "
                           "outputs/psychophysics/)")
    bank, cells = ST.load_design(BUILD)
    _W.update(bank=bank, cells={c.cell_id: c for c in cells}, tables=measures.model_tables())


def _socp_job(job: dict) -> dict:
    """One pinned s-OCP from the cell's phase-0 state (``answer_key.key_for``'s socp recipe,
    without a cache)."""
    import traceback

    from apexzero.interp import answer_key as ak

    out = dict(job)
    c0, t0 = time.process_time(), time.time()
    try:
        cell = _W["cells"][job["cell"]]
        prob = ak._problem(_W["bank"], cell, "nominal", 0, tables=_W["tables"])
        try:
            soc = ak._socp_for(prob, None)
        except AssertionError as e:            # the pin or strip check failed
            out.update(success=False, checks_ok=False, error=f"check: {e}")
        else:
            res = soc.result
            out.update(success=bool(res.success), checks_ok=True, status=str(res.status),
                       iterations=int(res.iterations), solve_seconds=float(res.solve_seconds),
                       ipopt_cpu=float(res.meta.get("cpu_seconds", np.nan)),
                       v_min=float(soc.measures.get("v_min", np.nan)),
                       has_brake=bool(soc.measures.get("has_brake", False)),
                       pin_check=soc.pin_check, key_id=soc.key_id)
    except Exception:  # noqa: BLE001  (reported to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"], out["wall"] = time.process_time() - c0, time.time() - t0
    return out


def socp_results(socp_dir: Path | None = None) -> dict[tuple[int, str], dict]:
    p = (socp_dir or SOCP) / "results.jsonl"
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            out[(int(r["corner"]), r["condition"])] = r
    return out


def finished(r: dict | None) -> bool:
    """A solve that ran to an answer (a failed pin or strip check is an answer; an exception
    is retried on the next run)."""
    return r is not None and (not r.get("error") or str(r["error"]).startswith("check:"))


def contingency_state() -> dict | None:
    p = GEN / "contingency.json"
    return read_json(p) if p.exists() else None


def timing_state() -> dict | None:
    p = SOCP / "timing.json"
    return read_json(p) if p.exists() else None


def active_conditions(timing: dict | None, cont: dict | None) -> list[str]:
    """Catalogue conditions still in play: F6s scales cut by the timing rule are out, and
    W+str is in only once the contingency has triggered."""
    cut = bool(timing and timing.get("cut"))
    trig = bool(cont and cont.get("triggered"))
    out = []
    for c in CATALOGUE:
        if c["family"] == "F6s" and cut and c["f6_scale"] not in F6S_SCALES_CUT:
            continue
        if c["contingency"] and not trig:
            continue
        out.append(c["name"])
    return out


def socp_certified(v_min: float, v_min_a: float, success: bool, checks_ok: bool) -> bool:
    """Plan Section 4.1: converged, passed its checks, and v*_min within 1% of (a)'s."""
    if not (success and checks_ok and np.isfinite(v_min) and np.isfinite(v_min_a)
            and v_min_a > 0):
        return False
    return abs(v_min / v_min_a - 1.0) <= SOCP_TOL


def stage_certify_socp(args) -> None:
    t0 = time.time()
    require_frozen()
    _, _, meta = load_build()
    kt = pio.load_table(KEY_TABLE)
    braking = read_json(BUILD / "cells_meta.json")["braking"]
    cell_of = {(m["corner"], m["condition"]): m["cell"] for m in meta if m["owner"] == "S1"}
    stored_cell = {m["cell"]: m["stored_cell"] for m in meta}
    done = socp_results()
    SOCP.mkdir(parents=True, exist_ok=True)

    def job(i, cond):
        return dict(corner=int(i), condition=cond, cell=int(cell_of[(i, cond)]),
                    label=f"s-OCP corner {i} {cond}")

    def record(r):
        with open(SOCP / "results.jsonl", "a") as fh:
            fh.write(json.dumps(clean(r), sort_keys=True) + "\n")
        done[(int(r["corner"]), r["condition"])] = r
        st = "error" if r.get("error") else ("ok" if r.get("success") else "not converged")
        v, t = r.get("v_min"), r.get("solve_seconds")
        print(f"  {r['label']}: {st}, v_min {float('nan') if v is None else v:.4f}, "
              f"{float('nan') if t is None else t:.1f} s", flush=True)

    # The first 10 solves, in job order: (a) on the 9 braking corners, then the first F6s
    # member on the first braking corner. They are timed, and (a) is checked.
    cont, timing = contingency_state(), timing_state()
    order = [job(i, COND_A) for i in braking]
    order += [job(i, c) for c in active_conditions(None, None) for i in braking]
    first = order[:TIMING_N]
    todo = [j for j in first if not finished(done.get((j["corner"], j["condition"])))]
    if todo:
        print(f"first {len(todo)} solves (timed)", flush=True)
        _pool(todo, _socp_job, _socp_init, args.workers, record)
    errs = [done[(j["corner"], j["condition"])] for j in first
            if done[(j["corner"], j["condition"])].get("error")]
    if errs:
        fail(f"s-OCP errors in the first solves: {[e['label'] for e in errs]}")
    # Built-in check: condition (a) reproduces step 1's key (the S1-state cell's v_min_socp).
    kidx = {int(c): j for j, c in enumerate(kt["cell"])}
    rep = {}
    for i in braking:
        r = done[(i, COND_A)]
        ref = float(kt["v_min_socp"][kidx[stored_cell[r["cell"]]]])
        rel_err = abs(r["v_min"] / ref - 1.0)
        rep[str(i)] = dict(v_min=r["v_min"], stored=ref, rel_err=rel_err,
                           success=r["success"], ok=bool(r["success"]
                                                         and rel_err <= SOCP_REPRO_RTOL))
    if not all(v["ok"] for v in rep.values()):
        write_json(SOCP / "check_a.json", rep)
        fail(f"condition (a)'s s-OCP does not reproduce step 1's keys: {rep}")
    if timing is None:
        secs = [float(done[(j["corner"], j["condition"])].get(
            "solve_seconds", done[(j["corner"], j["condition"])]["wall"])) for j in first]
        mean = float(np.mean(secs))
        timing = dict(n=len(secs), solve_seconds=secs, mean_s=mean, threshold_s=TIMING_MAX_S,
                      cut=bool(mean > TIMING_MAX_S),
                      cpu_s=[float(done[(j["corner"], j["condition"])]["cpu"]) for j in first],
                      rule="plan Section 9: mean wall seconds of the first 10 IPOPT solves "
                           "> 120 s cuts F6s to f6_scale 1, 2, 4, 8 before any further solve")
        write_json(SOCP / "timing.json", timing)
        print(f"timing: mean {mean:.1f} s per solve -> F6s cut: {timing['cut']}", flush=True)
    rest = [job(i, c) for c in active_conditions(timing, cont) for i in braking]
    rest = [j for j in rest if not finished(done.get((j["corner"], j["condition"])))]
    print(f"{len(rest)} solves to go", flush=True)
    _pool(rest, _socp_job, _socp_init, args.workers, record)
    errs = [r["label"] for r in done.values() if r.get("error") and not
            str(r["error"]).startswith("check:")]
    v_a = {i: done[(i, COND_A)]["v_min"] for i in braking}
    table = {}
    for c in active_conditions(timing, cont):
        per = {}
        for i in braking:
            r = done.get((i, c))
            if r is None:
                continue
            ratio = r.get("v_min", np.nan) / v_a[i] if r.get("v_min") is not None else np.nan
            per[str(i)] = dict(v_min=r.get("v_min"), ratio=ratio, success=r.get("success"),
                               checks_ok=r.get("checks_ok"),
                               certified=socp_certified(r.get("v_min") or np.nan, v_a[i],
                                                        bool(r.get("success")),
                                                        bool(r.get("checks_ok"))))
        table[c] = dict(corners=per, n_certified=sum(v["certified"] for v in per.values()),
                        certified_corners=[int(k) for k, v in per.items() if v["certified"]])
    write_json(SOCP / "socp.json", {
        "status": "EXPLORATORY (p4-plan.md Section 4.1)", "braking": braking,
        "check_a": rep, "timing": timing, "errors": errs, "conditions": table,
        "provenance": provenance("certify-socp", sys.argv[1:], t0,
                                 {"build.json": sha256_file(BUILD / "build.json")})})
    missing = [c for c in active_conditions(timing, cont) for i in braking
               if (i, c) not in done]
    print(f"wrote {rel(SOCP / 'socp.json')}; errors {len(errs)}; missing {len(missing)}",
          flush=True)


# ------------------------------------------------------------------------- rollouts


def _rollout_init(extension: bool = False) -> None:
    from apexzero.interp import measures

    configure(extension)
    be_nice()
    pio.assert_x64_off()
    if not measures.tables_path().exists():
        raise RuntimeError("step 1's model tables are missing")
    bank, cells = ST.load_design(BUILD)
    _W.update(bank=bank, cells=cells, by_id={c.cell_id: c for c in cells},
              tables=measures.model_tables(), design=ST.design_hash(bank, cells))


def run_padded(pol, params, obs_rms, bank, cells, runs, chunk: int = CHUNK) -> dict:
    """``rollout.run_rollouts`` with every chunk padded to exactly ``chunk`` runs (copies of
    the chunk's own runs, dropped afterwards): step 1 ran every tier D run in a 2048-run
    batch, and XLA's float32 results depend on the batch shape."""
    from apexzero.interp import rollout

    by_w: dict[int, list[int]] = {}
    for k, r in enumerate(runs):
        by_w.setdefault(int(r.W), []).append(k)
    T = max(by_w)
    parts = []
    for W in sorted(by_w):
        idx = by_w[W]
        for start in range(0, len(idx), chunk):
            sel = idx[start:start + chunk]
            pad = [runs[sel[j % len(sel)]] for j in range(chunk)]
            arr = rollout.run_rollouts(pol, params, obs_rms, bank, cells, pad, "full", chunk)
            arr = rollout._pad_time({k: v[:len(sel)] for k, v in arr.items()}, T)
            parts.append((np.asarray(sel), arr))
    order = np.argsort(np.concatenate([p[0] for p in parts]), kind="stable")
    keys = parts[0][1].keys()
    return {k: np.concatenate([p[1][k] for p in parts])[order] for k in keys}


def run_rows(agent: str, timesteps: int, det: dict, bank, by_id: dict, tables) -> dict:
    """The run table columns ``cond_table`` reads, from ``measures.response_metrics`` as
    ``analysis.run_table`` computes them (tier D cells have no null run)."""
    from apexzero.interp import measures

    A.check_runs_match_cells(det, by_id)
    rows = []
    for i in range(len(det["cell"])):
        cell = by_id[int(det["cell"][i])]
        stim = bank.stimuli[cell.stim_id]
        traj = measures.trajectory_from_rollout(det, i, "det")
        marks = A._marks(stim, bank, float(det["s0"][i]), int(det["W"][i]))
        m = measures.response_metrics(traj, marks, None, tables)
        rows.append(dict(agent=agent, timesteps=int(timesteps), mode="det", cell=int(cell.cell_id),
                         stim=int(cell.stim_id), phase=int(det["phase"][i]),
                         completed=bool(m.completed), crashed=bool(m.crashed),
                         v_min=float(m.v_min), v_peak=float(m.v_peak), s_crash=float(m.s_crash),
                         s_end=float(m.s_end), cause=str(m.cause)))
    return A._to_table(rows, ("agent", "timesteps", "mode", "cell", "stim", "phase", "completed",
                              "crashed", "v_min", "v_peak", "s_crash", "s_end", "cause"))


def _stored_bitwise(agent: str, timesteps: int, det: dict, stored_of: dict) -> dict:
    """Registered runs against step 1's stored deterministic runs: states and termination
    step bit for bit (a check beyond the plan's)."""
    have = [i for i in range(len(det["cell"])) if stored_of.get(int(det["cell"][i]), -1) >= 0]
    if not have:
        return {"n": 0}
    with np.load(pio.rollout_path(agent, timesteps), allow_pickle=False) as z:
        sd = {k: z[f"det_{k}"] for k in ("cell", "phase", "term_step", "x")}
    look = {(int(c), int(p)): j for j, (c, p) in enumerate(zip(sd["cell"], sd["phase"],
                                                              strict=True))}
    same, n = 0, 0
    worst = 0.0
    for i in have:
        j = look[(stored_of[int(det["cell"][i])], int(det["phase"][i]))]
        T = min(det["x"].shape[1], sd["x"].shape[1])
        a, b = det["x"][i][:T], sd["x"][j][:T]
        ok = np.array_equal(a, b) and int(det["term_step"][i]) == int(sd["term_step"][j])
        same += ok
        n += 1
        if not ok:
            worst = max(worst, float(np.nanmax(np.abs(a - b))))
    return {"n": n, "bitwise_equal": same, "max_abs_diff_unequal": worst}


def _rollout_job(job: dict) -> dict:
    """One agent's deterministic runs (4 phases) on a list of cells; the run table, the
    phase-0 d_OOD of specialists (``psychophysics._d_ood_from_det``, frozen) and, for
    registered cells, the bitwise comparison with step 1's runs."""
    import traceback

    from apexzero.interp import agents, controls, rollout

    out = dict(label=job["label"], agent=job["agent"], batch=job["batch"], cells=job["cells"])
    c0, t0 = time.process_time(), time.time()
    try:
        path = ROOT / job["path"]
        if sha256_file(path) != job["sha256"]:
            raise RuntimeError(f"{job['path']} is not the manifest's final snapshot")
        bank, cells, by_id = _W["bank"], _W["cells"], _W["by_id"]
        ag = agents.load_agent(path, job["agent"])
        cs = [by_id[c] for c in job["cells"]]
        runs = rollout.plan_runs(cs, bank, "det", 0, agent=job["agent"])
        if len(runs) != 4 * len(cs):
            raise RuntimeError(f"{len(runs)} runs for {len(cs)} cells")
        pol = rollout.agent_policy(ag.cfg)
        det = run_padded(pol, ag.actor_params, ag.obs_rms, bank, cells, runs)
        dst = Path(job["out"])
        rollout.save_rollouts(dst, {"det": det}, dict(
            agent=job["agent"], timesteps=ag.timesteps, log_std=ag.log_std.tolist(),
            path=job["path"], sha256=job["sha256"], design_hash=_W["design"],
            note="p4_exploratory.py (EXPLORATORY); chunks padded to 2048 runs"))
        rt = run_rows(job["agent"], ag.timesteps, det, bank, by_id, _W["tables"])
        pio.save_table(dst.with_name(dst.stem + "_runs.npz"), rt)
        out["n_runs"] = len(runs)
        out["timesteps"] = int(ag.timesteps)
        if job.get("d_ood"):
            import psychophysics as PS

            buf = controls.KnnBuffer.load(pio.CONTROLS_DIR / f"knn_{job['agent']}.npz")
            dd = PS._d_ood_from_det(det, job["agent"], bank, cells, ag.obs_rms, buf)
            out["d_ood"] = [[int(c), float(v)] for (_, c), v in dd.items()]
        if job.get("stored_of"):
            stored = {int(k): int(v) for k, v in job["stored_of"].items()}
            out["stored"] = _stored_bitwise(job["agent"], ag.timesteps, det, stored)
    except Exception:  # noqa: BLE001  (reported to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"], out["wall"] = time.process_time() - c0, time.time() - t0
    return out


def rollout_batch(batch_dir: Path, agents_: list[str], cells_of: dict[str, list[int]],
                  meta: list[dict], workers: int, d_ood: bool = False,
                  compare_stored: bool = False) -> dict[str, dict]:
    """Run (or resume) one batch of rollouts, one job per agent; returns per-agent results
    (run table, d_OOD, checks), cached in ``batch_dir``/results.json."""
    fin = finals()
    res_path = batch_dir / "results.json"
    have = read_json(res_path)["agents"] if res_path.exists() else {}
    stored_of = {m["cell"]: m["stored_cell"] for m in meta}
    jobs = []
    for a in agents_:
        cs = sorted(cells_of[a])
        if a in have and not have[a].get("error"):
            if have[a].get("cells") != cs:
                raise SystemExit(f"{batch_dir.name}/{a} on disk ran other cells")
            continue
        e = fin[a]
        jobs.append(dict(label=f"{a} {batch_dir.name}", agent=a, batch=batch_dir.name,
                         path=e.path, sha256=e.sha256, cells=cs,
                         out=str(batch_dir / f"{a}.npz"), d_ood=d_ood,
                         stored_of={str(c): stored_of[c] for c in cs} if compare_stored
                         else None))

    def done(r):
        st = "ERROR" if r.get("error") else f"{r.get('n_runs')} runs"
        extra = f", stored {r['stored']}" if r.get("stored") else ""
        print(f"  {r['label']}: {st} ({r['wall']:.0f} s){extra}", flush=True)
        have[r["agent"]] = {k: v for k, v in r.items() if k != "label"}
        write_json(res_path, {"agents": have})

    batch_dir.mkdir(parents=True, exist_ok=True)
    _pool(jobs, _rollout_job, _rollout_init, workers, done)
    errs = {a: r["error"] for a, r in have.items() if r.get("error")}
    if errs:
        for a, e in errs.items():
            print(f"{a}: {e}", flush=True)
        raise SystemExit(f"rollout errors in {batch_dir.name}: {sorted(errs)}")
    return have


def keys_table(meta: list[dict], kt: dict, socp_vmin: dict[tuple[int, str], float]) -> dict:
    """The key columns ``cond_table`` reads, per cell of this design: step 1's key table rows
    for registered cells (by their stored cell id); for new cells the S1-state s-OCP's v_min
    on S1's cell (``_share_tier_d_socp``'s rule: one s-OCP per strip, from S1's state)."""
    kidx = {int(c): j for j, c in enumerate(kt["cell"])}
    rows = []
    for m in meta:
        if m["registered"]:
            j = kidx[m["stored_cell"]]
            rows.append(dict(cell=m["cell"], corner=m["corner"], condition=m["condition"],
                             v_min_socp=float(kt["v_min_socp"][j]),
                             braking_socp=bool(kt["braking_socp"][j]),
                             swap_j=int(kt["swap_j"][j])))
        else:
            v = socp_vmin.get((m["corner"], m["condition"]), np.nan) if m["owner"] == "S1" \
                else np.nan
            rows.append(dict(cell=m["cell"], corner=m["corner"], condition=m["condition"],
                             v_min_socp=float(v), braking_socp=False, swap_j=-1))
    return A._to_table(rows, ("cell", "corner", "condition", "v_min_socp", "braking_socp",
                              "swap_j"))


def cond_tables(batch_dirs: list[Path], agents_: list[str], keys: dict,
                with_d_ood: bool = False) -> dict[str, dict]:
    """Per agent, the frozen ``cond_table`` over its runs in these batches."""
    out = {}
    for a in agents_:
        rts, dd = [], {}
        for b in batch_dirs:
            p = b / f"{a}_runs.npz"
            if p.exists():
                rts.append(pio.load_table(p))
            if with_d_ood and (b / "results.json").exists():
                r = read_json(b / "results.json")["agents"].get(a, {})
                dd.update({(a, int(c)): float(v) for c, v in r.get("d_ood", [])})
        if rts:
            out[a] = A.cond_table(pio.concat_tables(rts), keys, dd if with_d_ood else None)
    return out


def completion_over(cond: dict, name: str, corners) -> float:
    """C of one condition over the given corners: share with >= 3 of 4 phases complete."""
    m = (A._st(cond, "condition") == name) & np.isin(A._it(cond, "corner"), list(corners))
    return A._frac(A._bo(cond, "complete")[m])


def enters_p4(passes_generalists: bool, certified_corners) -> bool:
    """Plan Section 4.3: a condition the generalists pass, certified at >= 7 of the 9
    braking corners."""
    return bool(passes_generalists) and len(certified_corners) >= MIN_CORNERS


def generalist_criterion(median_a: float) -> dict:
    """Plan Section 4.2, fixed on condition (a) before any new condition is run."""
    if median_a >= GEN_MEDIAN_MIN:
        return dict(rule="median C_G >= 0.9", threshold=GEN_MEDIAN_MIN, fallback=False,
                    median_C_G_a=median_a)
    return dict(rule="fallback: median C_G >= median C_G(a) - 0.1",
                threshold=median_a - GEN_FALLBACK, fallback=True, median_C_G_a=median_a)


def contingency_rule(status: dict[str, dict]) -> dict:
    """W+ is rebuilt on K-str's footprint if it is certified at fewer than 7 braking corners
    at every A >= 1 m (a member's count: its s-OCP-certified corners if it passes the
    generalists' rule, else 0)."""
    counts = {}
    for a in WPLUS_AMPL:
        if a < 1.0:
            continue
        s = status.get(f"{PREFIX}/Wplus/{a:g}m", {})
        counts[f"{a:g}"] = len(s.get("socp_corners", [])) if s.get("passes_generalists") else 0
    return dict(counts=counts, triggered=bool(counts) and all(v < MIN_CORNERS
                                                             for v in counts.values()))


def stage_certify_generalists(args) -> None:
    t0 = time.time()
    require_frozen()
    _, _, meta = load_build()
    braking = read_json(BUILD / "cells_meta.json")["braking"]
    kt = pio.load_table(KEY_TABLE)
    sj = SOCP / "socp.json"
    if not sj.exists():
        raise SystemExit("run certify-socp first")
    socp = read_json(sj)
    timing, cont = socp["timing"], contingency_state()
    conds = active_conditions(timing, cont)
    miss = [c for c in conds if c not in socp["conditions"]
            or len(socp["conditions"][c]["corners"]) < len(braking)]
    if miss:
        raise SystemExit(f"s-OCP results missing for {miss}: run certify-socp")
    gens = list(registry.generalists())
    res = socp_results()
    keys = keys_table(meta, kt, {k: float(v.get("v_min", np.nan)) for k, v in res.items()})

    # Reference: (a) and the isolated road on the 9 braking corners (C_G(a) for the rule and
    # its fallback, both for P4's memorisation index); read before any new condition is run.
    ref_cells = owner_cells(meta, "S1", (COND_A, COND_ISO), braking)
    print(f"reference batch: {len(gens)} generalists x {len(ref_cells)} cells", flush=True)
    ref = rollout_batch(GEN / "reference", gens, {a: ref_cells for a in gens}, meta,
                        args.workers, compare_stored=True)
    ct = cond_tables([GEN / "reference"], gens, keys)
    C_a = {a: completion_over(ct[a], COND_A, braking) for a in gens}
    crit_path = GEN / "criterion.json"
    if crit_path.exists():
        crit = read_json(crit_path)
    else:
        crit = generalist_criterion(float(np.median(list(C_a.values()))))
        crit.update(C_G_a=C_a, stored_bitwise={a: ref[a].get("stored") for a in gens})
        write_json(crit_path, crit)
    print(f"C_G(a): {C_a}; rule: {crit['rule']} (threshold {crit['threshold']:.3f})",
          flush=True)
    if EXTENSION:
        main = read_json(OUT / "generalists" / "criterion.json")
        if (main["rule"], main["threshold"], main["C_G_a"]) != (crit["rule"], crit["threshold"],
                                                                 crit["C_G_a"]):
            fail("the extension's generalist criterion differs from the planned run's")

    def evaluate(names: list[str], batch: str) -> dict:
        per_gen = {a: [] for a in gens}
        for c in names:
            corners = socp["conditions"][c]["certified_corners"]
            for a in gens:
                per_gen[a] += owner_cells(meta, "S1", (c,), corners)
        if not any(per_gen.values()):
            return {}
        print(f"{batch} batch: {len(gens)} generalists x {len(per_gen[gens[0]])} cells",
              flush=True)
        rollout_batch(GEN / batch, gens, per_gen, meta, args.workers)
        cts = cond_tables([GEN / batch], gens, keys)
        out = {}
        for c in names:
            corners = socp["conditions"][c]["certified_corners"]
            if not corners:
                out[c] = dict(socp_corners=[], C_G={}, median_C_G=None,
                              passes_generalists=False)
                continue
            cg = {a: completion_over(cts[a], c, corners) for a in gens}
            med = float(np.median(list(cg.values())))
            out[c] = dict(socp_corners=corners, C_G=cg, median_C_G=med,
                          passes_generalists=bool(med >= crit["threshold"] - 1e-12))
        return out

    primary = [c for c in conds if not BY_NAME[c]["contingency"]]
    status = evaluate(primary, "new")
    if cont is None:
        cont = (dict(applicable=False, triggered=False,
                     note="the extension has no W+ members (Amendment 2)") if EXTENSION
                else contingency_rule(status))
        write_json(GEN / "contingency.json", cont)
        print(f"contingency: {cont}", flush=True)
    if cont["triggered"]:
        extra = [c["name"] for c in CATALOGUE if c["contingency"]]
        if any(c not in socp["conditions"] for c in extra):
            print("the contingency triggered: run certify-socp, then certify-generalists "
                  "again (W+str)", flush=True)
            return
        status.update(evaluate(extra, "contingency"))
    write_json(GEN / "generalists.json", {
        "status": "EXPLORATORY (p4-plan.md Section 4.2)", "criterion": crit,
        "contingency": cont, "conditions": status, "C_G_a": C_a,
        "provenance": provenance("certify-generalists", sys.argv[1:], t0,
                                 {"socp.json": sha256_file(sj)})})
    print(f"wrote {rel(GEN / 'generalists.json')}", flush=True)


# ------------------------------------------------------------------------- freeze


def certified_list() -> dict:
    socp = read_json(SOCP / "socp.json")
    gen = read_json(GEN / "generalists.json")
    build = read_json(BUILD / "build.json")
    strips = build["strip_sha256"]
    out = {}
    for c in CATALOGUE:
        name = c["name"]
        s = gen["conditions"].get(name)
        sc = socp["conditions"].get(name)
        entry = dict(family=c["family"], ladder=c["ladder"], kind=c["kind"],
                     amplitude=c["amplitude"], unit=c["unit"], f6_scale=c["f6_scale"])
        if sc is None:
            why = ("cut by the timing rule" if c["family"] == "F6s" and socp["timing"]["cut"]
                   else "contingency not triggered" if c["contingency"] else "not solved")
            entry.update(evaluated=False, reason=why, enters_p4=False, certified_corners=[])
        else:
            passes = bool(s and s["passes_generalists"])
            corners = sc["certified_corners"] if passes else []
            entry.update(evaluated=True, socp_corners=sc["certified_corners"],
                         socp_ratio={k: v["ratio"] for k, v in sc["corners"].items()},
                         median_C_G=s["median_C_G"] if s else None,
                         passes_generalists=passes, certified_corners=corners,
                         enters_p4=enters_p4(passes, corners))
        rows = [f"D/{i}/{name}" for i in entry["certified_corners"]]
        entry["strips_sha256"] = hashlib.sha256(
            "".join(strips[r] for r in rows).encode()).hexdigest() if rows else None
        entry["strip_sha256_per_corner"] = {r.split("/")[1]: strips[r] for r in rows}
        out[name] = entry
    status = ("EXPLORATORY and POST HOC (p4-plan.md Amendment 2): frozen before any "
              "specialist is rolled out on an extension rung" if EXTENSION else
              "EXPLORATORY (p4-plan.md Section 4.4): frozen before any specialist is rolled "
              "out on a new condition")
    return dict(status=status,
                criterion=gen["criterion"], contingency=gen["contingency"],
                timing=socp["timing"], braking=socp["braking"], conditions=out,
                entering=[n for n, e in out.items() if e["enters_p4"]],
                design_hash=build["design_hash"],
                inputs_sha256={"socp.json": sha256_file(SOCP / "socp.json"),
                               "generalists.json": sha256_file(GEN / "generalists.json"),
                               "build.json": sha256_file(BUILD / "build.json")})


def stage_freeze(args) -> None:
    require_frozen()
    if not (GEN / "generalists.json").exists():
        raise SystemExit("run certify-generalists first")
    cert = certified_list()
    if CERTIFIED.exists() and not args.force:
        old = read_json(CERTIFIED)
        if clean(cert) != {k: v for k, v in old.items() if k != "frozen"}:
            raise SystemExit("certified.json exists and differs (--force rewrites it)")
        print("certified.json unchanged", flush=True)
        return
    cert["frozen"] = dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), git_head=git("rev-parse",
                                                                                "HEAD"))
    write_json(CERTIFIED, cert)
    n = len(cert["entering"])
    print(f"wrote {rel(CERTIFIED)}: {n} conditions enter P4. Commit it before the "
          "specialists stage: git add -f outputs/p4/certified.json", flush=True)


def require_committed_freeze() -> dict:
    if not CERTIFIED.exists():
        raise SystemExit("no certified.json: run freeze")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel(CERTIFIED)], cwd=ROOT,
                             capture_output=True, check=False).returncode == 0
    if not tracked or git("status", "--porcelain", "--", rel(CERTIFIED)):
        raise SystemExit("certified.json must be committed, unchanged, before any specialist "
                         "is rolled out on a new condition (plan Section 4.4)")
    cert = read_json(CERTIFIED)
    if clean(certified_list()) != {k: v for k, v in cert.items() if k != "frozen"}:
        raise SystemExit("certified.json does not match the certification outputs")
    cert["frozen"]["commit"] = git("log", "-1", "--format=%H", "--", rel(CERTIFIED))
    return cert


# ------------------------------------------------------------------------- classification


def classify(cond: dict, generalists: dict | None, f6: tuple[str, ...] = A.F6_CONDS,
             tier_a: dict | None = None, tier_g: dict | None = None) -> A.P4Result:
    """P4's ordered class rules (first match wins), the frozen ``analysis.p4_classify`` with
    the F6 set as a parameter. With ``f6`` = the registered four it is ``p4_classify``. The
    details also name the F6 conditions that decide class 1 or block class 2."""
    agent = A._agent(cond)
    C, D, braking = A.completion_by_condition(cond)
    S_spec, phi, m, nidx = A._spec_indices(cond, braking)
    MI = A.memorisation_index(cond, generalists)
    eff = A._main_effects(C)
    det: dict = dict(d_ood=D, main_effects=eff, n_braking=len(braking), **nidx)
    tier_g_crash = bool((A._it(tier_g, "det_crashes", 0) > 0).any()) if tier_g is not None \
        and A._n(tier_g) else None
    det["tier_g_crash"] = tier_g_crash

    def res(cls: int, keys_: tuple[str, ...] = (), label: str | None = None) -> A.P4Result:
        return A.P4Result(agent, cls, label or A.P4_LABELS[cls], C, S_spec, phi, m, MI, keys_,
                          det)

    def c(name: str) -> float:
        return C.get(name, float("nan"))

    if not (c(COND_A) >= A.P4_GATE):
        return res(0)
    f6v = [(f, c(f), D.get(f, np.nan)) for f in f6 if f in C]
    single = [s for s in A.F1_SINGLE.values() if c(s) <= A.P4_FAIL]
    if single:
        missing = [s for s in single if not np.isfinite(D.get(s, np.nan))]
        missing += [f for f in f6 if f in C and not np.isfinite(D.get(f, np.nan))]
        if missing or not f6v:
            det.update(failing_conditions=single, d_ood_missing=missing or list(f6))
            return res(-1)
    for s in single:
        near = [f for f, cf, df in f6v if df <= D.get(s, np.nan) and cf <= A.P4_FAIL]
        if near:
            det["failing_condition"] = s
            det["p4a_class1_witnesses"] = near
            return res(1)
    for s in single:
        far = [(f, cf) for f, cf, df in f6v if df >= D.get(s, np.nan)]
        if all(cf >= A.P4_PASS for _, cf in far):
            keys_ = tuple(f for f in A.F1_FACTORS if np.isfinite(eff[f])
                          and eff[f] >= A.P4_MAIN_EFFECT)
            det.update(failing_condition=s, vacuous=not far, n_f6_at_or_above=len(far))
            label = A.P4_LABELS[2] + (" (width-keyed)" if keys_ == ("width",) else "")
            return res(2, keys_, label)
        det.setdefault("p4a_class2_blockers", {})[s] = [f for f, cf in far if cf < A.P4_PASS]
    all8 = all(c(x) >= A.P4_PASS for x in A.F1_CONDS)
    if all8 and (phi >= A.P4_PHI_MEM or m >= A.P4_M_MEM or S_spec < A.P4_S_MEM
                 or c(A.COND_MIRROR) <= A.P4_FAIL):
        return res(3)
    cls5 = (all8 and c(A.COND_MIRROR) >= A.P4_PASS
            and all(c(x) >= A.P4_PASS for x in A.LAMBDA_CONDS)
            and A._in(S_spec, A.P4_S_READ) and phi < A.P4_PHI_READ and m < A.P4_M_READ)
    if cls5:
        comp_a = float("nan")
        if tier_a is not None and A._n(tier_a):
            ma = A._mask(tier_a, approach="A2", braking=None, exclude=A.EXCL_ALL) & \
                (A._fl(tier_a, "R_seg") >= A.P4_TIER_A_R)
            nr = A._it(tier_a, "n_runs", 0)[ma].sum()
            comp_a = float(A._it(tier_a, "det_complete", 0)[ma].sum() / nr) if nr else np.nan
        det["tier_a_completion"] = comp_a
        if np.isfinite(comp_a) and comp_a >= A.P4_TIER_A_COMPLETION and tier_g_crash is False:
            return res(4)
        return res(5)
    return res(6)


def tier_g_table(crash: bool) -> dict:
    return A._to_table([dict(tiers="G", det_crashes=int(bool(crash)))], ("tiers", "det_crashes"))


def result_dict(r: A.P4Result) -> dict:
    d = A.p4(r)
    return dict(cls=r.cls, verdict=d.verdict, label=r.label, key_factors=list(r.key_factors),
                C=r.C, S_spec=r.S_spec, phi=r.phi, m=r.m, MI=r.MI, details=r.details)


def reproduction_check(res: dict[str, A.P4Result], dec: dict[str, dict]) -> dict:
    """Plan Section 6's check: with the registered F6 set the classifier reproduces
    decisions_all.json's P4 rows (class, label, failing condition, vacuous,
    n_f6_at_or_above, C exactly; d_OOD to 1e-6); S_spec, phi, m and MI are reported."""
    out = {}
    for a, r in res.items():
        ref = dec[a]
        rd = ref["details"]
        C_ok = r.C == {k: float(v) for k, v in rd["C"].items()}
        dd = [abs(r.details["d_ood"][k] - float(v)) for k, v in rd["d_ood"].items()]
        same = dict(
            verdict=A.p4(r).verdict == ref["verdict"], label=r.label == rd["label"],
            failing_condition=r.details.get("failing_condition") == rd.get("failing_condition"),
            vacuous=r.details.get("vacuous") == rd.get("vacuous"),
            n_f6_at_or_above=r.details.get("n_f6_at_or_above") == rd.get("n_f6_at_or_above"),
            C=C_ok, d_ood=bool(max(dd) <= DOOD_TOL) and set(r.details["d_ood"]) == set(rd["d_ood"]))
        diffs = {}
        for k in ("S_spec", "phi", "m", "MI"):
            mine, ref_v = float(getattr(r, k)), rd.get(k)
            if ref_v is None:
                diffs[k] = None if not np.isfinite(mine) else "differs (stored null)"
            else:
                diffs[k] = abs(mine - float(ref_v)) if np.isfinite(mine) else "differs (nan)"
        out[a] = dict(ok=all(same.values()), fields=same, d_ood_max_abs=max(dd),
                      reported_abs_diffs=diffs)
    return out


# ------------------------------------------------------------------------- specialists


def stage_specialists(args) -> None:
    t0 = time.time()
    require_frozen()
    cert = require_committed_freeze()
    _, _, meta = load_build()
    braking = cert["braking"]
    kt = pio.load_table(KEY_TABLE)
    dec = decisions_p4()
    res = socp_results()
    keys = keys_table(meta, kt, {k: float(v.get("v_min", np.nan)) for k, v in res.items()})
    gens = list(registry.seed0_generalists())
    gen_ct = cond_tables([GEN / "reference"], gens, keys)
    gen_table = pio.concat_tables([gen_ct[a] for a in gens])

    # 1. The registered conditions through this path: the reproduction check.
    reg = {a: owner_cells(meta, a, D_CONDITIONS, braking) for a in OWNERS}
    print(f"registered batch: {[len(v) for v in reg.values()]} cells", flush=True)
    out = rollout_batch(SPEC / "registered", list(OWNERS), reg, meta, args.workers,
                        d_ood=True, compare_stored=True)
    ct = cond_tables([SPEC / "registered"], list(OWNERS), keys, with_d_ood=True)
    rr = {a: classify(ct[a], gen_table, tier_g=tier_g_table(dec[a]["details"]["tier_g_crash"]))
          for a in OWNERS}
    chk = reproduction_check(rr, dec)
    write_json(SPEC / "check.json", {
        "status": "EXPLORATORY (p4-plan.md Sections 5 and 6): built-in reproduction check",
        "check": chk, "stored_bitwise": {a: out[a].get("stored") for a in OWNERS},
        "classes": {a: result_dict(r) for a, r in rr.items()}})
    if not all(v["ok"] for v in chk.values()):
        fail(f"the registered conditions do not reproduce decisions_all.json: {chk}")
    print("registered conditions reproduce decisions_all.json's P4 rows", flush=True)

    # 2. Every certified condition on its certified corners.
    new = {a: [] for a in OWNERS}
    for name in cert["entering"]:
        corners = cert["conditions"][name]["certified_corners"]
        for a in OWNERS:
            new[a] += owner_cells(meta, a, (name,), corners)
    print(f"new batch: {[len(v) for v in new.values()]} cells", flush=True)
    rollout_batch(SPEC / "new", list(OWNERS), new, meta, args.workers, d_ood=True)
    write_json(SPEC / "specialists.json", {
        "status": "EXPLORATORY (p4-plan.md Section 5)", "entering": cert["entering"],
        "freeze_commit": cert["frozen"]["commit"],
        "provenance": provenance("specialists", sys.argv[1:], t0, {
            "certified.json": sha256_file(CERTIFIED),
            "decisions_all.json": DECISIONS_ALL_SHA256,
            **{f"knn_{a}.npz": sha256_file(pio.CONTROLS_DIR / f"knn_{a}.npz")
               for a in OWNERS}})})
    print(f"wrote {rel(SPEC)}", flush=True)


def stage_classify(args) -> None:
    t0 = time.time()
    require_frozen()
    cert = require_committed_freeze()
    if not (SPEC / "specialists.json").exists():
        raise SystemExit("run the specialists stage first")
    _, _, meta = load_build()
    kt = pio.load_table(KEY_TABLE)
    dec = decisions_p4()
    res = socp_results()
    keys = keys_table(meta, kt, {k: float(v.get("v_min", np.nan)) for k, v in res.items()})
    gens = list(registry.seed0_generalists())
    gen_ct = cond_tables([GEN / "reference"], gens, keys)
    gen_table = pio.concat_tables([gen_ct[a] for a in gens])
    ct = cond_tables([SPEC / "registered", SPEC / "new"], list(OWNERS), keys, with_d_ood=True)
    entering = list(cert["entering"])
    info = {n: cert["conditions"][n] for n in entering}
    sets = {"registered": A.F6_CONDS}
    if EXTENSION:
        # Amendment 2: the planned certified conditions (their runs on the planned design)
        # plus the extension's rungs.
        main_cert = read_json(OUT / "certified.json")
        m_meta = read_json(OUT / "build" / "cells_meta.json")["cells"]
        m_res = socp_results(OUT / "socp")
        m_keys = keys_table(m_meta, kt, {k: float(v.get("v_min", np.nan))
                                         for k, v in m_res.items()})
        m_ct = cond_tables([OUT / "specialists" / "new"], list(OWNERS), m_keys,
                           with_d_ood=True)
        ct = {a: pio.concat_tables([ct[a], m_ct[a]]) for a in OWNERS}
        sets["planned"] = (*A.F6_CONDS, *main_cert["entering"])
        info.update({n: main_cert["conditions"][n] for n in main_cert["entering"]})
        entering = list(main_cert["entering"]) + entering
    sets["extended"] = (*A.F6_CONDS, *entering)
    fam_of = {n: info[n]["family"] for n in entering}
    for fam in FAMILIES:
        members = tuple(n for n in entering if fam_of[n] == fam)
        if members:
            sets[f"family:{fam}"] = (*A.F6_CONDS, *members)
    out, results = {}, {}
    for a in OWNERS:
        tg = tier_g_table(dec[a]["details"]["tier_g_crash"])
        per = {}
        for sname, f6 in sets.items():
            r = classify(ct[a], gen_table, f6, tier_g=tg)
            if r.cls in (4, 5):
                raise SystemExit("class 4/5 needs the tier A table, which this script does "
                                 "not load (plan Amendment 1)")
            per[sname] = result_dict(r)
            if sname == "extended":
                results[a] = r
        C, D, _ = A.completion_by_condition(ct[a])
        table = [dict(condition=n, family=fam_of[n], ladder=info[n]["ladder"],
                      amplitude=info[n]["amplitude"], unit=info[n]["unit"], C=C.get(n),
                      d_ood=D.get(n), n_corners=len(info[n]["certified_corners"]),
                      extension=EXTENSION and n in cert["conditions"])
                 for n in entering]
        table += [dict(condition=f, family="F6 (registered)", ladder=f"F6s/{f.split('/')[1]}",
                       amplitude=nom * 0.5, unit="m" if f.startswith("F6/wobble") else "1/m",
                       C=C.get(f), d_ood=D.get(f), n_corners=9)
                  for f, nom in zip(A.F6_CONDS, F6_NOMINAL, strict=True)]
        nearest, reach = {}, {}
        ext = sets["extended"]
        for s in A.F1_SINGLE.values():
            if C.get(s, np.nan) <= A.P4_FAIL:
                cand = [(abs(D[n] - D[s]), n) for n in entering if np.isfinite(D.get(n, np.nan))]
                if cand:
                    _, n = min(cand)
                    nearest[s] = dict(failing_C=C[s], failing_d_ood=D[s], nearest=n,
                                      C=C[n], d_ood=D[n])
                above = [f for f in ext if f in C and D.get(f, np.nan) >= D[s]]
                reach[s] = dict(failing_C=C[s], failing_d_ood=D[s],
                                n_f6_at_or_above=len(above),
                                failing_at_or_below=[f for f in ext if f in C
                                                     and D.get(f, np.nan) <= D[s]
                                                     and C[f] <= A.P4_FAIL],
                                incomplete_at_or_above=[f for f in above if C[f] < A.P4_PASS])
        fc = per["extended"]["details"].get("failing_condition")
        out[a] = dict(classes=per, table=table, nearest=nearest, reach=reach,
                      non_vacuous=(reach[fc]["n_f6_at_or_above"] >= 1) if fc in reach
                      else None)
    recipe = A.p4_recipe([results[a] for a in OWNERS])
    recipe_reg = A.p4_recipe([classify(ct[a], gen_table,
                                       tier_g=tier_g_table(dec[a]["details"]["tier_g_crash"]))
                              for a in OWNERS])
    write_json(OUT_DIR / "classes.json", {
        "status": ("EXPLORATORY and POST HOC (p4-plan.md Amendment 2: the ladder extension)"
                   if EXTENSION else
                   "EXPLORATORY (p4-plan.md Section 6): every class here is exploratory"),
        "sets": {k: list(v) for k, v in sets.items()}, "specialists": out,
        "recipe": dict(verdict=recipe.verdict, details=recipe.details),
        "recipe_registered": dict(verdict=recipe_reg.verdict, details=recipe_reg.details),
        "provenance": provenance("classify", sys.argv[1:], t0,
                                 {"certified.json": sha256_file(CERTIFIED)})})
    for a in OWNERS:
        e = out[a]["classes"]["extended"]
        print(f"{a}: extended {e['verdict']} ({e['label']}); per family "
              f"{ {k: v['verdict'] for k, v in out[a]['classes'].items()} }", flush=True)
    print(f"recipe: {recipe.verdict}", flush=True)


# ------------------------------------------------------------------------- report


def _spearman(x, y) -> float:
    from scipy import stats

    if len(x) < 2:
        return float("nan")
    if len(x) == 2:
        return float(np.sign((x[1] - x[0]) * (y[1] - y[0])))
    return float(stats.spearmanr(x, y).statistic)


def readings(cert: dict, classes: dict) -> dict:
    """E1-E6 as Amendment 1 operationalises them."""
    conds = cert["conditions"]
    ent = set(cert["entering"])

    def top(pred) -> float:
        v = [conds[n]["f6_scale"] if conds[n]["family"] == "F6s"
             else conds[n]["amplitude"] / 1e-3 for n in ent if pred(conds[n])]
        return max(v) if v else 0.0

    kn_f6 = top(lambda e: e["family"] == "F6s" and e["kind"] == "knoise")
    kn_k = top(lambda e: e["family"] == "Kstr")
    wob = top(lambda e: e["family"] == "F6s" and e["kind"] == "wobble")
    e1a = dict(top_scale_F6s_knoise=kn_f6, top_scale_equiv_Kstr=kn_k, top_scale_F6s_wobble=wob,
               holds=bool(kn_f6 > wob and kn_k > wob))
    wp = {a: conds[f"{PREFIX}/Wplus/{a:g}m"] for a in WPLUS_AMPL}
    n_s = {f"{a:g}": len(wp[a].get("socp_corners", [])) for a in WPLUS_AMPL}
    e1b = dict(enters={f"{a:g}": wp[a]["enters_p4"] for a in WPLUS_AMPL}, socp_corners=n_s,
               holds=bool(wp[0.5]["enters_p4"] and wp[1.0]["enters_p4"]
                          and n_s["4"] < n_s["0.5"]))
    big = [n for n, e in conds.items() if e["family"] == "F6s" and e["kind"] == "wobble"
           and e["amplitude"] >= 1.8 - 1e-9 and e["evaluated"]]
    nb = len(cert["braking"])
    e1c = dict(members={n: dict(socp_failed=nb - len(conds[n]["socp_corners"]),
                                enters=conds[n]["enters_p4"]) for n in big},
               holds=bool(big) and all(nb - len(conds[n]["socp_corners"]) >= 3
                                       and not conds[n]["enters_p4"] for n in big))
    # E2a: d_OOD rises with amplitude in every sub-ladder (>= 2 members entering).
    e2a = {}
    for a, sp in classes["specialists"].items():
        rows = [r for r in sp["table"] if r["condition"] in ent and r["d_ood"] is not None]
        for lad in sorted({r["ladder"] for r in rows}):
            rr = sorted((r["amplitude"], r["d_ood"]) for r in rows if r["ladder"] == lad)
            if len(rr) >= 2:
                e2a[f"{a} {lad}"] = _spearman([x for x, _ in rr], [y for _, y in rr])
    e2a_holds = bool(e2a) and all(v > 0 for v in e2a.values())
    e2b = {}
    for a, thr in (("S2", 2.37), ("S3", 2.75)):
        rows = [r for r in classes["specialists"][a]["table"] if r["condition"] in ent
                and r["d_ood"] is not None and r["d_ood"] >= thr]
        e2b[a] = dict(threshold=thr, reaching=[(r["condition"], r["d_ood"]) for r in rows],
                      all_wobble_or_widening=all(
                          conds[r["condition"]]["kind"] in WOBBLE_FAMILIES for r in rows))
    e2b_holds = all(v["reaching"] and v["all_wobble_or_widening"] for v in e2b.values())
    cls = {a: classes["specialists"][a]["classes"]["extended"]["cls"] for a in OWNERS}
    s2 = classes["specialists"]["S2"]["classes"]["extended"]
    e3 = dict(cls=cls["S2"], witnesses=s2["details"].get("p4a_class1_witnesses"),
              failing_condition=s2["details"].get("failing_condition"), holds=cls["S2"] == 1)
    e4 = dict(cls=cls["S3"], holds=cls["S3"] in (1, 6))
    e5 = dict(cls=cls["S1"], holds=cls["S1"] == 6)
    e6 = dict(recipe=classes["recipe"]["verdict"],
              applicable=cls["S2"] == 1 and cls["S3"] == 1,
              holds=(classes["recipe"]["verdict"] == "class 1") if cls["S2"] == 1
              and cls["S3"] == 1 else None)
    return {"E1a": e1a, "E1b": e1b, "E1c": e1c,
            "E2a": dict(spearman=e2a, holds=e2a_holds), "E2b": dict(per=e2b, holds=e2b_holds),
            "E3": e3, "E4": e4, "E5": e5, "E6": e6}


def stage_report(args) -> None:
    import csv

    t0 = time.time()
    require_frozen()
    cert = require_committed_freeze()
    classes = read_json(OUT_DIR / "classes.json")
    if EXTENSION:
        # The readings over the planned and the extension's certified conditions together.
        main_cert = read_json(OUT / "certified.json")
        cert = dict(main_cert, conditions={**main_cert["conditions"], **cert["conditions"]},
                    entering=list(main_cert["entering"]) + list(cert["entering"]))
    read = readings(cert, classes)
    with open(OUT_DIR / "c_vs_dood.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specialist", "condition", "family", "ladder", "amplitude", "unit", "C",
                    "d_ood", "n_corners"])
        for a, sp in classes["specialists"].items():
            for r in sp["table"]:
                w.writerow([a, r["condition"], r["family"], r["ladder"], f"{r['amplitude']:g}",
                            r["unit"], "" if r["C"] is None else f"{r['C']:.4f}",
                            "" if r["d_ood"] is None else f"{r['d_ood']:.4f}", r["n_corners"]])
    write_json(OUT_DIR / "report.json", {
        "status": ("EXPLORATORY and POST HOC (p4-plan.md Amendment 2: the ladder extension)"
                   if EXTENSION else
                   "EXPLORATORY: every reading here is exploratory (p4-plan.md Section 7)"),
        "readings": read,
        "certification": {n: {k: e.get(k) for k in ("family", "amplitude", "unit",
                                                     "socp_corners", "median_C_G",
                                                     "passes_generalists", "enters_p4")}
                          for n, e in cert["conditions"].items()},
        "classes": {a: {k: v["verdict"] for k, v in sp["classes"].items()}
                    for a, sp in classes["specialists"].items()},
        "recipe": classes["recipe"],
        "provenance": provenance("report", sys.argv[1:], t0, {
            "certified.json": sha256_file(CERTIFIED),
            "classes.json": sha256_file(OUT_DIR / "classes.json")})})
    for k, v in read.items():
        print(f"{k}: holds={v.get('holds')}", flush=True)
    print(f"wrote {rel(OUT_DIR / 'report.json')}, c_vs_dood.csv", flush=True)


# ------------------------------------------------------------------------- main


STAGES = {"build": stage_build, "certify-socp": stage_certify_socp,
          "certify-generalists": stage_certify_generalists, "freeze": stage_freeze,
          "specialists": stage_specialists, "classify": stage_classify, "report": stage_report}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=tuple(STAGES))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help=f"process pool size (at most {MAX_WORKERS})")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extension", action="store_true",
                    help="Amendment 2's post hoc K-str ladder extension (outputs/p4/extension/)")
    args = ap.parse_args(argv)
    configure(args.extension)
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} processes (another workflow shares the Mac)")
    be_nice()
    assert P4_STREAM + STREAM_OFFSET not in {STREAM_OFFSET + v for v in STREAMS.values()}
    STAGES[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
