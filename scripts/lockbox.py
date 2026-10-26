"""The lockbox harness (LB-0, follow-through plan Section 7.7): ENGINEERING, no claim.

Four stages, run in order on one root; each refuses what the one before it did not leave:

    uv run python scripts/lockbox.py freeze --mode dryrun --root outputs/lockbox-dryrun/NAME \\
        --circuits BrandsHatch Hockenheim Norisring \\
        --h4r-subjects h4r-v3-base-s12 h4r-v1-realw-s12 h4r-v1-base-s11
    uv run python scripts/lockbox.py ocp --root outputs/lockbox-dryrun/NAME --workers 3
    uv run python scripts/lockbox.py laps --root outputs/lockbox-dryrun/NAME --workers 3
    uv run python scripts/lockbox.py analyse --root outputs/lockbox-dryrun/NAME

- ``freeze`` binds everything before any number exists: the circuits and their mirrored and
  reversed variants, the protocol constants, the subjects (checkpoint paths and sha256s, each
  checked), the circuit files' sha256s and a fresh ``freeze_id``. It refuses a root that
  exists and is not empty, so every later file in the root is new.
- ``ocp`` re-solves the s-domain minimum-time references under the current code: every
  variant at its nominal widths and with both sides' widths moved by -0.5 m and +0.5 m (plan
  v1's geometry error bar), with a fixed convergence fallback.
- ``laps`` rolls every subject out on every variant (``eval.laps.timed_laps``'s protocol, line
  for line, checked against ``timed_laps`` itself on each subject's first variant), keeps each
  completed run's lap-2 time at every station (per-corner timing) and run 0's OOD distance at
  every step. It prints no outcome.
- ``analyse`` runs once. It refuses if its lock or any output exists, checks every input, then
  takes an exclusive lock (``analysis.lock``, never removed by the harness) before computing
  any number. It reads references only from the root's own ``ocp/``, each file checked against
  its record's sha256 and the root's ``freeze_id``; it never reads outputs/ocp/ or
  outputs/sealed/. If it fails after taking the lock, rerunning it needs the lock removed by
  hand, which on the lockbox root is a protocol deviation to log.

**Modes.** ``freeze --mode`` sets the mode; every later stage reads it from the root's
``freeze.json`` and checks it against the root.

- ``dryrun``: R-dev stands in for R_TEST. Circuits must be R-dev (a subset is allowed) and every
  name passes ``viz.export.lockbox_guard`` first, so R_TEST names and their variants raise
  PermissionError at every stage. ``--root`` is required (use a new directory under
  outputs/lockbox-dryrun/); it may not be, contain or sit inside outputs/lockbox/ or
  outputs/sealed/.
- ``lockbox``: all 12 R_TEST circuits, root outputs/lockbox/ only. Every stage first calls
  ``tag_gate``, which raises PermissionError unless the annotated tag ``protocol-v1`` exists on
  origin, the local tag is the same tag object, HEAD descends from its commit, and the metric
  code (``METRIC_PATHS``, untracked files included) in the working tree is the tagged code.
  Only behind that check does ``lockbox_guard(name, unsealed=True)`` run; it appears nowhere
  else in this file (a test checks). The subjects file must be tracked and unchanged since
  the tag.

**Definitions** (``freeze`` writes the constants into ``freeze.json``; LB-1's pre-registration
fixes the final values and may change any of them).

- Track: ``refine_reference(variant(load_tumftm(csv)))``, variant first, as the environment
  sees it. A width re-solve moves both sides of that track by delta; the reference line stays.
- Reference: ``ocp.min_time.solve_min_time`` (ds 3 m, margin 1 m, tol 1e-6); if it fails, one
  retry from the failed iterate at tol 1e-5 (``scripts/solve_all_ocp.py``'s rule). A reference
  that still fails is reported, and every gap against it is missing.
- Laps: default car, grip 1, no observation noise; rolling start at s = 0 from 0.9 x the QSS
  start speed; ``--runs`` runs (default 10: run 0 centred, the rest jittered as ``timed_laps``);
  2 laps, lap 2 timed. A variant is completed when MORE than half its runs finish (the gate's
  rule). G = median completed lap / T_OCP - 1.
- Per-corner timing: ``viz.sectors.corner_sectors`` on the variant's track; the time lost in a
  sector is ``viz.sectors.sector_deltas`` of t_agent(s) - t_OCP(s) on lap 2, so the sectors sum
  to the lap delta (checked). Corner type by the nominal OCP's apex speed, its minimum v_x
  between the complex's entry and exit: slow < 120, medium 120-200, fast > 200 km/h (plan v1).
- OOD distance: the generalist analogue of step 1's d_OOD (the frozen
  ``interp.controls.assemble_buffer``, ``nn_distance`` and ``d_ood``, imported read-only). Each
  subject's buffer holds 40,000 of its normalised observations (subsampled, seeded) from one
  deterministic 1,200-step window on each of 100 generated levels (the training generator
  fitted on raw R-dev centrelines, default car, grip 1, root seed 7707, which no other study
  uses); the denominator is the buffer's leave-one-level-out median nearest-neighbour
  distance. A variant's d_OOD is the median over run 0's steps on its two laps; a corner's,
  over run 0's steps in its sector. LB-1 must set the generator to the subjects' own training
  generator (generator-v1) if that differs from the default.

**Workers.** At most 4 processes (``--workers``, clamped); each lowers its priority to nice 15
and runs with 2 XLA and BLAS threads (``--threads``). Another workflow shares the Mac.

Outputs (under the root): freeze.json; ocp/<key>.pkl and .json; laps/<subject>/buffer.npz and
.json, laps/<subject>/<variant>.npz and .json; analysis.lock, analysis.json, corners.csv.
"""

from __future__ import annotations

import os
import sys
import time

_T0 = time.time()


def _threads(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return None


_T = _threads(sys.argv)
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or 2}")
for _v in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(_T or 2))

import argparse
import csv
import dataclasses
import hashlib
import io
import json
import math
import multiprocessing
import pickle
import platform
import re
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

from apexzero.tracks.corners import R_DEV, R_TEST
from apexzero.viz.export import lockbox_guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
LOCKBOX_ROOT = ROOT / "outputs" / "lockbox"
DRYRUN_PARENT = ROOT / "outputs" / "lockbox-dryrun"
SEALED = ROOT / "outputs" / "sealed"
TRACK_DIR = ROOT / "data" / "tracks" / "tumftm"
H4R_MANIFEST = ROOT / "outputs" / "h4r" / "manifest.json"

TAG = "protocol-v1"
METRIC_PATHS = ("scripts/lockbox.py", "scripts/_bootstrap.py", "src/apexzero")
MODES = ("dryrun", "lockbox")
VARIANTS = ("original", "mirrored", "reversed")
WIDTH_DELTAS = (0.0, -0.5, 0.5)      # m, both sides; 0.0 is the nominal reference
MAX_WORKERS = 4
NICE = 15
SUBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CAUSES = ("corridor", "slip", "spin", "stall")
CORNER_TYPES = ("slow", "medium", "fast", "unknown")
P2_SHOOTING_LIMIT = 3e-3          # the P2 gate's shooting consistency (0.3% of the lap)

PROTOCOL = {
    "laps": {"runs": 10, "laps": 2, "start_fraction": 0.9, "seed": 0, "completion_min": 0.5,
             "completed_when": "completion fraction > completion_min",
             "context": "default car (racing.default_context), grip 1, no observation noise"},
    "ocp": {"ds": 3.0, "margin": 1.0, "tol": 1e-6, "max_iter": 3000,
            "fallback": "one retry from the failed iterate at tol 1e-5"},
    "widths": {"deltas_m": list(WIDTH_DELTAS), "applied_to": "w_left and w_right, each"},
    "sectors": {"merge_gap_m": 40.0, "type_kph": [120.0, 200.0],
                "types": ["slow (< 120)", "medium (120-200)", "fast (> 200)"],
                "apex_speed": "the nominal OCP's minimum v_x between the complex's entry and exit"},
    "ood": {"root_seed": 7707, "stream": "a_dr", "levels": 100, "steps": 1200,
            "n_obs": 40_000, "subsample_seed": 7707, "chunk": 25,
            "generator": "train grammar, fit_grammar on raw R-dev centrelines",
            "denominator": "leave-one-level-out median NN distance",
            "variant_statistic": "median (and 90th percentile) over run 0's steps, both laps"},
    "stats": {"n_boot": 2000, "boot_seed": 0, "iqm_trim": 0.25, "ci": [2.5, 97.5],
              "ci_method": "nearest rank (no interpolation, so +inf entries are safe)",
              "h1": {"laps_completed_min": 0.90, "median_gap_max": 0.04,
                     "source": "plan-v1 H1 as written; LB-1 may replace it"}},
}


class CheckFailed(SystemExit):
    """A built-in check failed: the stage stops and writes nothing more."""


# ------------------------------------------------------------------------- provenance


def git_run(*args: str) -> subprocess.CompletedProcess:
    """Every git call goes through here (the tests replace it)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False)


def git(*args: str) -> str:
    out = git_run(*args)
    return out.stdout.strip() if out.returncode == 0 else ""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


SCRIPT_SHA256 = sha256_file(SCRIPT)     # the code this process loaded


def rel(path: Path) -> str:
    """Repo-relative where possible, so no output carries an absolute local path."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return Path(path).name if Path(path).is_absolute() else str(path)


def environment() -> dict:
    import jax

    return {"python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__, "machine": platform.machine(),
            "system": platform.system(), "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "nice": os.nice(0), "cpu_count": os.cpu_count()}


def provenance(start: float, inputs: dict[str, str] | None = None) -> dict:
    return {
        "script": rel(SCRIPT), "script_sha256": SCRIPT_SHA256,
        "git_head": git("rev-parse", "HEAD"),
        "changed_paths": [ln for ln in git("status", "--porcelain", "--untracked-files=no")
                          .splitlines() if ln.strip()],
        "inputs_sha256": dict(sorted((inputs or {}).items())),
        "argv": [rel(SCRIPT), *sys.argv[1:]],
        "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - start, 1),
        "environment": environment(),
    }


def clean(x):
    """JSON-safe: NaN as null, infinities as "inf" / "-inf", NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def unclean(x) -> float:
    """``clean``'s inverse for one number: null is NaN, "inf" / "-inf" are infinities."""
    if x is None:
        return float("nan")
    if isinstance(x, str):
        return float(x)
    return float(x)


def write_new(path: Path, data: bytes) -> str:
    """Write ``data`` to a path that must not exist yet (atomic); returns its sha256."""
    path = Path(path)
    if path.exists():
        raise CheckFailed(f"{rel(path)} exists: the harness never overwrites an output")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)        # fails if the path appeared meanwhile
    finally:
        tmp.unlink()
    return sha256_bytes(data)


def write_json_new(path: Path, obj) -> str:
    return write_new(path, json.dumps(clean(obj), indent=1, allow_nan=False).encode())


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def lower_priority(level: int = NICE) -> None:
    cur = os.nice(0)
    if cur < level:
        os.nice(level - cur)


# ------------------------------------------------------------------------- the lockbox gate


def tag_gate() -> dict:
    """PermissionError unless the annotated tag protocol-v1 exists on origin, the local tag is
    the same tag object, HEAD descends from its commit, and the metric code in the working
    tree (``METRIC_PATHS``, untracked files included) is the tagged code."""
    ls = git_run("ls-remote", "--tags", "origin", TAG)
    if ls.returncode != 0:
        raise PermissionError(f"cannot list origin's tags ({ls.stderr.strip()}): the lockbox "
                              "stays sealed")
    refs = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    obj, peeled = refs.get(f"refs/tags/{TAG}"), refs.get(f"refs/tags/{TAG}^{{}}")
    if obj is None:
        raise PermissionError(f"{TAG} is not on origin: the lockbox stays sealed")
    if peeled is None or peeled == obj:
        raise PermissionError(f"{TAG} on origin is not an annotated tag")
    local = git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}")
    if local.returncode != 0 or local.stdout.strip() != obj:
        raise PermissionError(f"the local {TAG} is missing or is not origin's tag object")
    kind = git_run("cat-file", "-t", obj)
    if kind.returncode != 0 or kind.stdout.strip() != "tag":
        raise PermissionError(f"{TAG} ({obj}) is not a tag object")
    commit = git_run("rev-parse", "--verify", "--quiet", f"{obj}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.strip() != peeled:
        raise PermissionError(f"{TAG}'s commit does not match origin's")
    anc = git_run("merge-base", "--is-ancestor", peeled, "HEAD")
    if anc.returncode != 0:
        raise PermissionError(f"HEAD does not descend from {TAG} ({peeled})")
    diff = git_run("diff", "--quiet", peeled, "--", *METRIC_PATHS)
    if diff.returncode != 0:
        raise PermissionError(f"the metric code ({', '.join(METRIC_PATHS)}) differs from "
                              f"{TAG}'s")
    untracked = git_run("ls-files", "--others", "--exclude-standard", "--", *METRIC_PATHS)
    if untracked.returncode != 0 or untracked.stdout.strip():
        raise PermissionError(f"untracked files under the metric code: {untracked.stdout!r}")
    head = git_run("rev-parse", "HEAD")
    return {"tag": TAG, "tag_object": obj, "commit": peeled, "head": head.stdout.strip()}


def variant_name(circuit: str, variant: str) -> str:
    """The Track name ``tracks.track`` gives the variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return circuit if variant == "original" else f"{circuit}-{variant}"


@dataclasses.dataclass(frozen=True)
class Clearance:
    """What ``clear`` admitted: only these circuits and variant names may load."""
    mode: str
    circuits: tuple[str, ...]
    names: frozenset[str]
    tag: dict | None = None


def clear(circuits, mode: str) -> Clearance:
    """The only door to circuit geometry. ``dryrun``: every name through ``lockbox_guard``
    first (R_TEST names and their variants raise PermissionError), then R-dev only.
    ``lockbox``: ``tag_gate`` first (PermissionError without the tag), then all 12 R_TEST
    circuits, then ``lockbox_guard(name, unsealed=True)``, its only call in this file."""
    circuits = tuple(str(c) for c in circuits)
    names = [variant_name(c, v) for c in circuits for v in VARIANTS]
    if mode == "dryrun":
        for n in (*circuits, *names):
            lockbox_guard(n)
        bad = [c for c in circuits if c not in R_DEV]
        if bad:
            raise ValueError(f"the dry run takes R-dev circuits only, not {bad}")
        if len(set(circuits)) != len(circuits) or not circuits:
            raise ValueError(f"circuits must be distinct and non-empty: {circuits}")
        return Clearance(mode, circuits, frozenset((*circuits, *names)))
    if mode == "lockbox":
        tag = tag_gate()
        if sorted(circuits) != sorted(R_TEST) or len(circuits) != len(R_TEST):
            raise ValueError("the lockbox run takes all 12 R_TEST circuits, no subset")
        for n in (*circuits, *names):
            lockbox_guard(n, unsealed=True)    # reached only behind tag_gate
        return Clearance(mode, circuits, frozenset((*circuits, *names)), tag)
    raise ValueError(f"mode must be one of {MODES}, not {mode!r}")


def _admit(name: str, clearance: Clearance) -> None:
    if name not in clearance.names:
        raise PermissionError(f"{name} was not cleared for this run")
    if clearance.mode == "dryrun":
        lockbox_guard(name)
    elif clearance.mode != "lockbox" or not clearance.tag:
        raise PermissionError(f"{name}: no tag clearance")


def load_variant(circuit: str, variant: str, clearance: Clearance):
    """The refined Track of a cleared variant: variant first, then ``refine_reference``."""
    from apexzero.tracks.reference import refine_reference
    from apexzero.tracks.track import load_tumftm

    _admit(circuit, clearance)
    name = variant_name(circuit, variant)
    _admit(name, clearance)
    base = load_tumftm(TRACK_DIR / f"{circuit}.csv")
    raw = (base if variant == "original" else base.mirrored() if variant == "mirrored"
           else base.reversed())
    track = refine_reference(raw)
    if track.name != name:
        raise CheckFailed(f"variant track is named {track.name}, not {name}")
    return track


def widen(track, delta: float, margin: float = PROTOCOL["ocp"]["margin"]):
    """The track with both sides' widths moved by ``delta`` m (the reference line stays)."""
    if delta == 0.0:
        return track
    wl, wr = track.w_left + delta, track.w_right + delta
    if min(wl.min(), wr.min()) <= margin:
        raise CheckFailed(f"{track.name}: widths {delta:+.1f} m leave no corridor")
    return dataclasses.replace(track, w_left=wl, w_right=wr,
                               meta=dict(track.meta, width_delta_m=delta))


# ------------------------------------------------------------------------- roots


def _inside(path: Path, parent: Path) -> bool:
    p, q = Path(path).resolve(), Path(parent).resolve()
    return p == q or q in p.parents


def resolve_root(mode: str, root: str | Path | None) -> Path:
    if mode == "lockbox":
        if root is not None and Path(root).resolve() != LOCKBOX_ROOT.resolve():
            raise CheckFailed(f"the lockbox run writes to {rel(LOCKBOX_ROOT)} only")
        return LOCKBOX_ROOT
    if mode != "dryrun":
        raise ValueError(f"mode must be one of {MODES}")
    if root is None:
        raise CheckFailed(f"a dry run needs --root (a new directory under {rel(DRYRUN_PARENT)})")
    out = Path(root)
    if _inside(out, LOCKBOX_ROOT) or _inside(LOCKBOX_ROOT, out):
        raise CheckFailed(f"a dry run may not write to, inside or above {rel(LOCKBOX_ROOT)}")
    if _inside(out, SEALED) or _inside(SEALED, out):
        raise CheckFailed("a dry run may not write to, inside or above outputs/sealed/")
    return out


def safe_path(root: Path, *parts: str) -> Path:
    """A path the harness may read or write: inside the root (symlinks resolved), never in
    outputs/sealed/."""
    p = Path(root).joinpath(*parts)
    if not _inside(p, root):
        raise PermissionError(f"{p} is outside the run's root")
    if _inside(p, SEALED):
        raise PermissionError("outputs/sealed/ is never read")
    return p


def load_freeze(root) -> tuple[dict, Clearance]:
    """The root's freeze record and its clearance; the mode must match the root."""
    if root is None:
        raise CheckFailed("give --root")
    root = Path(root)
    path = safe_path(root, "freeze.json")
    if not path.exists():
        raise CheckFailed(f"{rel(path)} is missing: run freeze first")
    rec = json.loads(path.read_text())
    mode = rec.get("mode")
    if mode not in MODES:
        raise CheckFailed(f"freeze.json has no valid mode ({mode!r})")
    is_lb = root.resolve() == LOCKBOX_ROOT.resolve()
    if (mode == "lockbox") != is_lb:
        raise CheckFailed(f"mode {mode} does not belong in {rel(root)}")
    if mode == "dryrun":
        resolve_root(mode, root)
    clearance = clear(rec["circuits"], mode)
    if mode == "lockbox" and clearance.tag["commit"] != (rec.get("tag") or {}).get("commit"):
        raise CheckFailed(f"{TAG} now points at {clearance.tag['commit']}, not the frozen "
                          f"{(rec.get('tag') or {}).get('commit')}")
    return rec, clearance


# ------------------------------------------------------------------------- subjects


def subjects_from_h4r(names) -> list[dict]:
    """Dry-run subjects: H4R finals with the sha256s bound in outputs/h4r/manifest.json, and
    each one's stored end-of-training R-dev evaluation (its eval_rdev.json, sha-checked), which
    ``analyse`` compares with the original variants (Mac against A6000; GW-1 found completion
    equal on 334 of 336 calls and no lap time equal bit for bit)."""
    man = json.loads(H4R_MANIFEST.read_text())["networks"]
    out = []
    for n in names:
        ent = man.get(f"runs/{n}")
        if ent is None:
            raise CheckFailed(f"{n} is not in {rel(H4R_MANIFEST)}")
        stored = None
        ev = ROOT / ent.get("eval_path", "")
        if ent.get("eval_path") and ev.exists():
            raw = ev.read_bytes()
            if sha256_bytes(raw) != ent["eval_sha256"]:
                raise CheckFailed(f"{rel(ev)} does not match the manifest's sha256")
            stored = {c: {"completion": r.get("completion"), "median_lap_s": r.get("median_lap_s")}
                      for c, r in json.loads(raw).items() if isinstance(r, dict)}
        out.append({"name": n, "checkpoint": ent["path"], "sha256": ent["sha256"],
                    "source": rel(H4R_MANIFEST), "stored_eval_rdev": stored})
    return out


def subjects_from_file(path: Path) -> list[dict]:
    d = json.loads(Path(path).read_text())
    subs = d["subjects"] if isinstance(d, dict) else d
    return [{"name": s["name"], "checkpoint": s["checkpoint"], "sha256": s["sha256"],
             "source": rel(path)} for s in subs]


def checkpoint_file(sub: dict) -> Path:
    p = Path(sub["checkpoint"])
    return p if p.is_absolute() else ROOT / p


def verify_subjects(subs: list[dict]) -> list[dict]:
    """Each subject's checkpoint must hash to its declared sha256; names must be distinct and
    filesystem-safe; no checkpoint may sit in outputs/sealed/."""
    names = [s["name"] for s in subs]
    if not subs or len(set(names)) != len(names):
        raise CheckFailed(f"subjects must be distinct and non-empty: {names}")
    out = []
    for s in subs:
        if not SUBJECT_NAME.match(s["name"]):
            raise CheckFailed(f"subject name {s['name']!r} is not filesystem-safe")
        f = checkpoint_file(s)
        if _inside(f, SEALED):
            raise PermissionError("outputs/sealed/ is never read")
        raw = f.read_bytes()
        sha = sha256_bytes(raw)
        if sha != s["sha256"]:
            raise CheckFailed(f"{s['name']}: {s['checkpoint']} hashes to {sha}, not the "
                              f"declared {s['sha256']}")
        d = pickle.loads(raw)
        out.append(dict(s, checkpoint=rel(f) if _inside(f, ROOT) else str(f),
                        timesteps=int(d["timesteps"])))
    return out


def load_subject(sub: dict):
    """(policy, cfg, snapshot) of a frozen subject, refusing a changed checkpoint. The policy
    is scripts/eval_rdev.py's: the actor's mean on the snapshot's normalised observation."""
    from apexzero.agents.ppo import Actor, PPOConfig
    from apexzero.audit.rollouts import snapshot_from_pickle

    raw = checkpoint_file(sub).read_bytes()
    if sha256_bytes(raw) != sub["sha256"]:
        raise CheckFailed(f"{sub['name']}: checkpoint changed since freeze")
    d = pickle.loads(raw)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    snap = snapshot_from_pickle(d)
    actor = Actor(cfg, 2, False)

    def policy(obs):
        return actor.apply(snap.actor, snap.obs_rms.normalize(obs))[0]

    return policy, cfg, snap


# ------------------------------------------------------------------------- freeze


def stage_freeze(args) -> Path:
    t_start = time.time()
    mode = args.mode
    if mode == "lockbox" and args.circuits:
        raise CheckFailed("the lockbox run takes all 12 R_TEST circuits; --circuits is for "
                          "the dry run")
    circuits = list(R_TEST) if mode == "lockbox" else list(args.circuits or R_DEV)
    clearance = clear(circuits, mode)            # the gate (lockbox) or the guard (dry run)
    root = resolve_root(mode, args.root)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise CheckFailed(f"{rel(root)} exists and is not empty: freeze needs a fresh root")
    if bool(args.subjects) == bool(args.h4r_subjects):
        raise CheckFailed("give exactly one of --subjects FILE and --h4r-subjects NAMES")
    if mode == "lockbox" and not args.subjects:
        raise CheckFailed("the lockbox run takes its subjects from a committed file")
    if args.runs < 3:
        raise CheckFailed("--runs must be at least 3")
    if args.subjects:
        spath = Path(args.subjects)
        if mode == "lockbox":
            r = rel(spath)
            tracked = git_run("ls-files", "--error-unmatch", "--", r)
            same = git_run("diff", "--quiet", clearance.tag["commit"], "--", r)
            if tracked.returncode != 0 or same.returncode != 0:
                raise CheckFailed(f"{r} must be tracked and unchanged since {TAG}")
        subs = subjects_from_file(spath)
    else:
        subs = subjects_from_h4r(args.h4r_subjects)
    subs = verify_subjects(subs)
    files = {rel(TRACK_DIR / f"{c}.csv"): sha256_file(TRACK_DIR / f"{c}.csv")
             for c in clearance.circuits}
    protocol = json.loads(json.dumps(PROTOCOL))
    protocol["laps"]["runs"] = int(args.runs)
    rec = {
        "label": ("DRY RUN: R-dev stands in for R_TEST; not a lockbox result"
                  if mode == "dryrun" else f"LOCKBOX ({TAG})"),
        "mode": mode, "freeze_id": uuid.uuid4().hex,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "circuits": list(clearance.circuits), "variants": list(VARIANTS),
        "protocol": protocol, "subjects": subs, "circuit_files_sha256": files,
        "tag": clearance.tag, "provenance": provenance(t_start),
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json_new(root / "freeze.json", rec)
    print(f"froze {len(subs)} subjects x {len(circuits)} circuits x {len(VARIANTS)} variants "
          f"into {rel(root)} ({rec['label']})")
    return root


def check_circuit_files(rec: dict) -> None:
    for path, sha in rec["circuit_files_sha256"].items():
        if sha256_file(ROOT / path) != sha:
            raise CheckFailed(f"{path} changed since freeze")


# ------------------------------------------------------------------------- ocp


def ocp_key(circuit: str, variant: str, delta: float) -> str:
    return f"{variant_name(circuit, variant)}__w{delta:+.1f}"


def ocp_tasks(rec: dict) -> list[tuple[str, str, float]]:
    return [(c, v, float(d)) for c in rec["circuits"] for v in rec["variants"]
            for d in rec["protocol"]["widths"]["deltas_m"]]


def ocp_done(root: Path, key: str) -> dict | None:
    """The record of a finished solve (its pickle checked), or None if not written yet."""
    js = safe_path(root, "ocp", f"{key}.json")
    if not js.exists():
        return None
    r = json.loads(js.read_text())
    pk = safe_path(root, "ocp", f"{key}.pkl")
    if not pk.exists() or sha256_file(pk) != r["pkl_sha256"]:
        raise CheckFailed(f"{rel(pk)} is missing or does not match its record")
    return r


def solve_task(root: str, rec: dict, clearance: Clearance, circuit: str, variant: str,
               delta: float) -> dict:
    from apexzero.ocp.check import shooting_check
    from apexzero.ocp.min_time import solve_min_time
    from apexzero.tracks.reference import drivable_margin
    from apexzero.vehicle.params import F1_2025

    lower_priority()
    t_start = time.time()
    root = Path(root)
    key = ocp_key(circuit, variant, delta)
    op = rec["protocol"]["ocp"]
    track = widen(load_variant(circuit, variant, clearance), delta, op["margin"])
    attempts = []
    res = solve_min_time(track, F1_2025, ds=op["ds"], margin=op["margin"], tol=op["tol"],
                         max_iter=op["max_iter"])
    attempts.append({"init": "qss", "tol": op["tol"], "success": res.success,
                     "status": res.status, "iterations": res.iterations,
                     "solve_s": round(res.solve_seconds, 1)})
    if not res.success:
        res = solve_min_time(track, F1_2025, ds=op["ds"], margin=op["margin"], tol=1e-5,
                             max_iter=op["max_iter"], init=(res.z, res.u))
        attempts.append({"init": "failed iterate", "tol": 1e-5, "success": res.success,
                         "status": res.status, "iterations": res.iterations,
                         "solve_s": round(res.solve_seconds, 1)})
    try:
        chk = shooting_check(res, F1_2025)
        shoot = {"shooting_rel_err": float(chk.rel_lap_error),
                 "max_dv_mps": float(chk.max_speed_error), "shooting_error": None}
    except Exception as exc:  # noqa: BLE001  (recorded; analyse reports it)
        shoot = {"shooting_rel_err": float("nan"), "max_dv_mps": float("nan"),
                 "shooting_error": repr(exc)}
    pkl_sha = write_new(safe_path(root, "ocp", f"{key}.pkl"), pickle.dumps(res))
    out = {"key": key, "circuit": circuit, "variant": variant, "name": track.name,
           "width_delta_m": delta, "freeze_id": rec["freeze_id"], "success": bool(res.success),
           "status": res.status, "attempts": attempts, "lap_time_s": float(res.lap_time),
           "length_m": float(track.length), **shoot,
           "min_width_m": float(min(track.w_left.min(), track.w_right.min())),
           "drivable_margin": float(drivable_margin(track)),
           "pkl": f"ocp/{key}.pkl", "pkl_sha256": pkl_sha,
           "provenance": provenance(t_start)}
    write_json_new(safe_path(root, "ocp", f"{key}.json"), out)
    return {"key": key, "converged": out["success"],
            "solve_s": round(sum(a["solve_s"] for a in attempts), 1)}


def run_pool(fn, tasks: list[tuple], workers: int) -> list:
    workers = max(1, min(workers, MAX_WORKERS, len(tasks)))
    out = []
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=lower_priority) as pool:
        futs = {pool.submit(fn, *t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            out.append(r)
            print(json.dumps(clean(r)), flush=True)
    return out


def stage_ocp(args) -> None:
    rec, clearance = load_freeze(args.root)
    root = Path(args.root)
    check_circuit_files(rec)
    todo = [t for t in ocp_tasks(rec) if ocp_done(root, ocp_key(*t)) is None]
    print(f"{len(todo)} of {len(ocp_tasks(rec))} solves to run", flush=True)
    if todo:
        run_pool(solve_task, [(str(root), rec, clearance, *t) for t in todo], args.workers)


# ------------------------------------------------------------------------- laps


@dataclasses.dataclass
class LapRollout:
    """``eval.laps.LapResult``'s numbers plus every run's arrays and run 0's observations."""
    completed: np.ndarray      # [R]
    lap_time: np.ndarray       # [R]
    s_reached: np.ndarray      # [R]
    t: np.ndarray              # [T] time after each step
    s: np.ndarray              # [T, R] post-step s
    alive: np.ndarray          # [T, R] after the step
    term: np.ndarray           # [T, R]
    causes: np.ndarray         # [T, R, 4] off_track, slipped, spun, stalled
    s_pre0: np.ndarray | None  # [T] run 0's s before each step
    alive_pre0: np.ndarray | None
    obs0: np.ndarray | None    # [T, D] run 0's normalised observation before each step
    length: float
    laps: int
    dt: float

    @property
    def completion_rate(self) -> float:
        return float(self.completed.mean())


def _cross(s: np.ndarray, t: np.ndarray, k: int, target: float, dt: float):
    """``eval.laps.timed_laps``'s crossing arithmetic at sample k, with its operand types
    (``target`` and ``dt`` Python floats, ``s`` float32), so results are bit for bit equal."""
    s_prev = s[k - 1] if k else 0.0
    return t[k] - dt + (target - s_prev) / max(s[k] - s_prev, 1e-6) * dt


def crossing_time(s: np.ndarray, t: np.ndarray, target: float, dt: float):
    """``timed_laps``'s ``cross``: the time s first reaches ``target``, linear between the
    samples either side (s before the first sample is 0); None if never."""
    idx = np.nonzero(s >= target)[0]
    if not len(idx):
        return None
    return _cross(s, t, int(idx[0]), float(target), float(dt))


def lap_profile(s: np.ndarray, t: np.ndarray, stations: np.ndarray, length: float, lap: int,
                dt: float) -> np.ndarray:
    """Time since the start of lap ``lap`` (1-based) at each station in [0, length], each one
    ``crossing_time`` exactly, so the last entry equals the lap time bit for bit."""
    base = float((lap - 1) * length)
    t0 = crossing_time(s, t, base, dt)
    out = np.empty(len(stations))
    for i, x in enumerate(stations):
        out[i] = crossing_time(s, t, float(base + x), dt) - t0
    return out


def rollout_laps(policy_fn, track, v_start: float, runs: int = 20, laps: int = 2, params=None,
                 ctx=None, seed: int = 0, obs_fn=None) -> LapRollout:
    """``eval.laps.timed_laps``, line for line, keeping every run's s, alive and termination
    flags and, with ``obs_fn``, run 0's ``obs_fn(obs)``, s and alive before every step."""
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle.sim import track_arrays

    params = racing.EnvParams() if params is None else params
    tr = track_arrays(track)
    ctx = racing.default_context() if ctx is None else ctx
    L = float(track.length)
    max_steps = int(laps * 3.0 * L / max(v_start, 20.0) / params.dt) + 200
    eval_params = params._replace(window_steps=10**9)

    rng = np.random.default_rng(seed)
    n0 = np.where(np.arange(runs) == 0, 0.0, rng.uniform(-1.0, 1.0, runs))
    v0 = v_start * np.where(np.arange(runs) == 0, 1.0, rng.uniform(0.95, 1.05, runs))
    x0 = np.zeros((runs, 8), np.float32)
    x0[:, 1], x0[:, 3] = n0, v0

    def init_state(x):
        return racing.EnvState(x=x, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx,
                               hidden_mu=jnp.array(1.0),
                               t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()), s0=x[0])

    states = jax.vmap(init_state)(jnp.asarray(x0))
    p = racing.params_from_context(ctx)
    obs = jax.vmap(lambda st: racing.observe(st, tr, p))(states)
    record = obs_fn is not None

    @jax.jit
    def run(states, obs):
        def body(c, _):
            st, ob, alive = c
            a = policy_fn(ob)
            ob2, st2, _, term, _, info = jax.vmap(
                lambda s, aa: racing.step_no_reset(s, aa, tr, eval_params))(st, a)
            alive2 = alive & ~term
            extra = (st.x[0, 0], alive[0], obs_fn(ob[0])) if record else ()
            st2 = jax.tree.map(lambda new, old: jnp.where(
                alive.reshape((-1,) + (1,) * (new.ndim - 1)), new, old), st2, st)
            flags = jnp.stack([info["off_track"], info["slipped"], info["spun"],
                               info["stalled"]], axis=-1)
            return (st2, ob2, alive2), (st2.x[:, 0], alive2, term, flags, *extra)
        _, out = jax.lax.scan(body, (states, obs, jnp.ones(runs, bool)), None, max_steps)
        return out

    out = [np.asarray(a) for a in run(states, obs)]
    s, alive, term, flags = out[:4]
    t = (np.arange(max_steps) + 1) * params.dt
    completed = np.zeros(runs, bool)
    lap_time = np.full(runs, np.nan)
    for r in range(runs):
        t1 = crossing_time(s[:, r], t, (laps - 1) * L, params.dt)
        t2 = crossing_time(s[:, r], t, laps * L, params.dt)
        if t1 is not None and t2 is not None and alive[np.nonzero(s[:, r] >= laps * L)[0][0], r]:
            completed[r] = True
            lap_time[r] = t2 - t1
    return LapRollout(completed, lap_time, s.max(axis=0), t, s, alive, term, flags,
                      out[4] if record else None, out[5] if record else None,
                      out[6] if record else None, L, laps, float(params.dt))


def outcomes(ro: LapRollout) -> dict:
    """Per run: crash (terminated without completing), timeout (neither), and for a crash the
    terminating step's s (mod L), lap and cause (scripts/diagnose_crashes.py's order)."""
    R = ro.s.shape[1]
    prev = np.vstack([np.ones((1, R), bool), ro.alive[:-1]])
    hit = ro.term & prev
    out = {"crash": np.zeros(R, bool), "timeout": np.zeros(R, bool),
           "term_s": np.full(R, np.nan), "term_lap": np.full(R, -1), "cause": [""] * R}
    for r in range(R):
        if ro.completed[r]:
            continue
        ks = np.nonzero(hit[:, r])[0]
        if ks.size:
            k = int(ks[0])
            out["crash"][r] = True
            out["term_s"][r] = float(ro.s[k, r] % ro.length)
            out["term_lap"][r] = int(ro.s[k, r] // ro.length) + 1
            f = ro.causes[k, r]
            out["cause"][r] = next((c for c, v in zip(CAUSES, f, strict=True) if v),
                                   "nonfinite")
        else:
            out["timeout"][r] = True
    return out


def ood_levels(spec: dict):
    """(stacked TrackArrays, x0) of the OOD reference levels: the training generator fitted on
    raw R-dev centrelines, as scripts/pilot_audit.py and GW-1 build A_DR, from the harness's
    own root seed."""
    import jax

    from apexzero.tracks.generator import fit_grammar
    from apexzero.tracks.levels import build, make_levels
    from apexzero.tracks.track import load_tumftm

    for name in R_DEV:
        lockbox_guard(name)
    g = fit_grammar([load_tumftm(TRACK_DIR / f"{n}.csv") for n in R_DEV])
    lv = make_levels(jax.random.PRNGKey(spec["root_seed"]), spec["stream"], n=spec["levels"],
                     randomize_car=False)
    tracks, _, x0 = build(lv, g)
    return tracks, x0


def buffer_window(cfg, steps: int):
    """One deterministic window from a level's start (the mean action, default car, grip 1):
    the normalised observation before every step and whether the car was alive then."""
    import jax
    import jax.numpy as jnp

    from apexzero.agents.ppo import Actor
    from apexzero.env import racing

    actor = Actor(cfg, racing.ACTION_DIM, False)
    params = racing.EnvParams()._replace(window_steps=steps)

    def one(snap, tr, x0_l):
        ctx, mu = racing.default_context(), jnp.array(1.0)
        st = racing.EnvState(x=x0_l, prev_action=jnp.zeros(2), accel=jnp.zeros(2), ctx=ctx,
                             hidden_mu=mu, t=jnp.zeros((), jnp.int32), slow=jnp.zeros(()),
                             s0=x0_l[0])
        obs = racing.observe(st, tr, racing.params_from_context(ctx, hidden_mu=mu))

        def body(c, _):
            st, ob, alive = c
            o = snap.obs_rms.normalize(ob)
            mean, _ = actor.apply(snap.actor, o)
            ob2, st2, _, term, trunc, _ = racing.step_no_reset(st, mean, tr, params)
            alive2 = alive & ~term & ~trunc
            st2 = jax.tree.map(lambda n, old: jnp.where(alive, n, old), st2, st)
            ob2 = jnp.where(alive, ob2, ob)
            return (st2, ob2, alive2), (o, alive)

        _, (o, alive) = jax.lax.scan(body, (st, obs, jnp.array(True)), None, steps)
        return o, alive

    return one


def build_buffer(snap, cfg, spec: dict, name: str = "subject"):
    """The subject's KnnBuffer (``interp.controls.assemble_buffer``, levels as the groups)."""
    import jax

    from apexzero.interp.controls import assemble_buffer

    tracks, x0 = ood_levels(spec)
    one = jax.jit(jax.vmap(buffer_window(cfg, spec["steps"]), in_axes=(None, 0, 0)))
    n_lv = int(x0.shape[0])
    obs, group = [], []
    for a in range(0, n_lv, spec["chunk"]):
        sl = slice(a, min(a + spec["chunk"], n_lv))
        tr = jax.tree.map(lambda v, sl=sl: v[sl], tracks)
        o, alive = (np.asarray(z) for z in one(snap, tr, x0[sl]))
        for i in range(o.shape[0]):
            obs.append(o[i][alive[i]])
            group.append(np.full(int(alive[i].sum()), a + i))
    obs, group = np.concatenate(obs), np.concatenate(group)
    return assemble_buffer(obs, np.zeros((len(obs), 2), np.float32), group,
                           n_obs=spec["n_obs"], rng=np.random.default_rng(spec["subsample_seed"]),
                           agent=name)


def subject_buffer(root: Path, rec: dict, sub: dict, snap, cfg):
    """The subject's buffer, built once and saved (a resumed run reloads it, checked)."""
    from apexzero.interp.controls import KnnBuffer

    d = safe_path(root, "laps", sub["name"])
    js, nz = d / "buffer.json", d / "buffer.npz"
    if js.exists():
        meta = json.loads(js.read_text())
        if not nz.exists() or sha256_file(nz) != meta["npz_sha256"]:
            raise CheckFailed(f"{rel(nz)} does not match its record")
        with np.load(nz) as z:
            buf = KnnBuffer(obs=z["obs"], act=np.zeros((len(z["obs"]), 2), np.float32),
                            lap=z["group"], lolo_median_nn=float(z["lolo_median_nn"]),
                            agent=sub["name"])
        return buf, meta["npz_sha256"]
    t_start = time.time()
    buf = build_buffer(snap, cfg, rec["protocol"]["ood"], sub["name"])
    sha = write_new(nz, npz_bytes(obs=buf.obs, group=buf.lap,
                                  lolo_median_nn=np.float64(buf.lolo_median_nn)))
    write_json_new(js, {"subject": sub["name"], "freeze_id": rec["freeze_id"],
                        "n_obs": len(buf.obs), "n_levels": len(np.unique(buf.lap)),
                        "lolo_median_nn": buf.lolo_median_nn, "npz_sha256": sha,
                        "spec": rec["protocol"]["ood"], "provenance": provenance(t_start)})
    return buf, sha


def laps_done(root: Path, subject: str, name: str) -> dict | None:
    js = safe_path(root, "laps", subject, f"{name}.json")
    if not js.exists():
        return None
    r = json.loads(js.read_text())
    nz = safe_path(root, "laps", subject, f"{name}.npz")
    if not nz.exists() or sha256_file(nz) != r["npz_sha256"]:
        raise CheckFailed(f"{rel(nz)} is missing or does not match its record")
    return r


def parity_check(policy, track, v_start: float, lp: dict, ro: LapRollout) -> None:
    """``rollout_laps`` must reproduce ``eval.laps.timed_laps`` on the same inputs exactly."""
    from apexzero.eval.laps import timed_laps

    ref = timed_laps(policy, track, v_start, runs=lp["runs"], laps=lp["laps"], seed=lp["seed"])
    same = (np.array_equal(ref.completed, ro.completed)
            and np.array_equal(ref.lap_time, ro.lap_time, equal_nan=True)
            and np.array_equal(ref.s_reached, ro.s_reached))
    if not same:
        raise CheckFailed(f"{track.name}: the harness's rollout differs from timed_laps")


def laps_subject(root: str, rec: dict, clearance: Clearance, sub: dict) -> dict:
    """Every variant for one subject (one worker per subject)."""
    from apexzero.env.adapters import reference_speed
    from apexzero.interp.controls import d_ood

    lower_priority()
    root = Path(root)
    lp = rec["protocol"]["laps"]
    todo = [(c, v) for c in rec["circuits"] for v in rec["variants"]
            if laps_done(root, sub["name"], variant_name(c, v)) is None]
    if not todo:
        return {"subject": sub["name"], "variants_run": 0}
    policy, cfg, snap = load_subject(sub)
    buf, buf_sha = subject_buffer(root, rec, sub, snap, cfg)
    t_all = time.time()
    for i, (circuit, variant) in enumerate(todo):
        t_start = time.time()
        track = load_variant(circuit, variant, clearance)
        name = track.name
        v_start = float(reference_speed(track)[0]) * lp["start_fraction"]
        ro = rollout_laps(policy, track, v_start, runs=lp["runs"], laps=lp["laps"],
                          seed=lp["seed"], obs_fn=snap.obs_rms.normalize)
        if i == 0:
            parity_check(policy, track, v_start, lp, ro)
        oc = outcomes(ro)
        L = ro.length
        stations = np.r_[track.s, L]
        tau = np.full((lp["runs"], len(stations)), np.nan)
        for r in np.nonzero(ro.completed)[0]:
            tau[r] = lap_profile(ro.s[:, r], ro.t, stations, L, lp["laps"], ro.dt)
        if ro.completed.any() and not np.array_equal(tau[ro.completed, -1],
                                                     ro.lap_time[ro.completed]):
            raise CheckFailed(f"{sub['name']} {name}: the lap profile misses the lap time")
        keep = ro.alive_pre0 & (ro.s_pre0 < lp["laps"] * L)
        d = d_ood(ro.obs0[keep], buf).astype(np.float32)
        npz = npz_bytes(stations=stations, tau=tau, ood_s=ro.s_pre0[keep].astype(np.float64),
                        ood_d=d)
        sha = write_new(safe_path(root, "laps", sub["name"], f"{name}.npz"), npz)
        write_json_new(safe_path(root, "laps", sub["name"], f"{name}.json"), {
            "subject": sub["name"], "circuit": circuit, "variant": variant, "name": name,
            "freeze_id": rec["freeze_id"], "checkpoint_sha256": sub["sha256"],
            "buffer_sha256": buf_sha, "v_start": v_start, "length_m": L,
            "max_steps": int(ro.s.shape[0]), "completed": ro.completed,
            "lap_time_s": ro.lap_time, "s_reached": ro.s_reached, **oc,
            "parity_checked": i == 0,
            "npz": f"laps/{sub['name']}/{name}.npz", "npz_sha256": sha,
            "seconds": round(time.time() - t_start, 2), "provenance": provenance(t_start)})
        print(f"{sub['name']} {name} written", flush=True)    # no outcome is printed
    return {"subject": sub["name"], "variants_run": len(todo),
            "seconds": round(time.time() - t_all, 1)}


def stage_laps(args) -> None:
    rec, clearance = load_freeze(args.root)
    check_circuit_files(rec)
    run_pool(laps_subject, [(str(args.root), rec, clearance, s) for s in rec["subjects"]],
             args.workers)


# ------------------------------------------------------------------------- analysis helpers


def corner_type(v_kph: float, edges=(120.0, 200.0)) -> str:
    """slow < 120, medium 120-200 (both ends), fast > 200 km/h."""
    if not np.isfinite(v_kph):
        return "unknown"
    return "slow" if v_kph < edges[0] else "fast" if v_kph > edges[1] else "medium"


def apex_speed_kph(res, sector, length: float) -> float:
    """The OCP's minimum v_x between the complex's entry and exit (km/h)."""
    s = np.asarray(res.s, float)
    span = sector.s_exit - sector.s_entry
    m = np.mod(s - sector.s_entry, length) <= span
    if not m.any():                       # a complex shorter than the OCP's station spacing
        off = np.abs(np.mod(s - sector.s_apex + length / 2, length) - length / 2)
        m = np.arange(len(s)) == int(np.argmin(off))
    return float(np.min(res.z[m, 2]) * 3.6)


def ocp_time_at(res, stations: np.ndarray) -> np.ndarray:
    """The OCP's t(s) at the stations (t = lap time at s = L)."""
    L = float(res.s[-1] + res.meta["ds"])
    return np.interp(stations, np.r_[res.s, L], np.r_[res.t, res.lap_time])


def sector_dt(stations: np.ndarray, tau: np.ndarray, t_ocp: np.ndarray, sectors,
              length: float) -> np.ndarray:
    """[R, n_sectors] time lost (> 0) per sector on the timed lap; rows of runs that did not
    complete stay NaN."""
    from apexzero.viz.sectors import sector_deltas

    out = np.full((tau.shape[0], len(sectors)), np.nan)
    for r in range(tau.shape[0]):
        if np.all(np.isfinite(tau[r])):
            out[r] = sector_deltas(stations, tau[r] - t_ocp, sectors, length)
    return out


def iqm(x, keep_inf: bool = False) -> float:
    """rliable's IQM, ``scipy.stats.trim_mean(x, 0.25)`` written out: sort, cut
    int(0.25 n) from each end, average the rest. NaN is missing and dropped; with
    ``keep_inf``, +inf (an uncompleted circuit) stays in, sorts last, and makes the IQM +inf
    when it reaches the middle half."""
    x = np.asarray(x, float).ravel()
    x = x[~np.isnan(x)] if keep_inf else x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    x = np.sort(x)
    k = int(PROTOCOL["stats"]["iqm_trim"] * len(x))
    return float(np.mean(x[k:len(x) - k]))


def _pct(x: np.ndarray, q: float) -> float:
    """Nearest-rank percentile (no interpolation, so +inf entries are safe)."""
    x = np.sort(np.asarray(x, float)[~np.isnan(np.asarray(x, float))])
    if not len(x):
        return float("nan")
    k = min(len(x) - 1, max(0, math.ceil(q / 100.0 * len(x)) - 1))
    return float(x[k])


def iqm_stratified(M: np.ndarray, n_boot: int, seed: int, keep_inf: bool = False) -> dict:
    """IQM over a [subjects, variants] score matrix (NaN = missing) with a stratified
    bootstrap: subjects resampled with replacement within each variant (rliable)."""
    M = np.asarray(M, float)
    R, K = M.shape
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, R, (R, K))
        boots[b] = iqm(M[idx, np.arange(K)[None, :]], keep_inf)
    lo, hi = PROTOCOL["stats"]["ci"]
    return {"iqm": iqm(M, keep_inf), "ci95": [_pct(boots, lo), _pct(boots, hi)],
            "n_scores": int((~np.isnan(M)).sum()), "n_inf": int(np.isinf(M).sum()),
            "n_missing": int(np.isnan(M).sum()), "n_boot": n_boot, "seed": seed}


def spearman(x, y) -> dict:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return {"rho": None, "n": int(m.sum())}
    return {"rho": float(spearmanr(x[m], y[m]).statistic), "n": int(m.sum())}


def _nanmax(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmax(x)) if np.isfinite(x).any() else float("nan")


def _med(x) -> float:
    """Median with NaN (missing) dropped; infinities are kept."""
    x = np.asarray(x, float).ravel()
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


GAP_TAGS = {"+0.0": "", "-0.5": "_narrow", "+0.5": "_wide"}


def family_summary(rows: list[dict], cmin: float) -> dict:
    """Over one subject's rows of one family: the share of laps completed, circuits completed
    (completion > cmin), the median G over completed circuits, and the median with an
    uncompleted circuit counted as +inf; each also against the -0.5 m and +0.5 m references.
    A missing reference leaves the +inf statistic missing, never smaller."""
    if not rows:
        return {}
    laps_ok = sum(int(np.sum(r["completed"])) for r in rows)
    laps_all = sum(len(r["completed"]) for r in rows)
    done = [r for r in rows if r["completion"] > cmin]
    out = {"n_circuits": len(rows), "laps_completed_share": laps_ok / laps_all,
           "circuits_completed": len(done), "circuits_completed_share": len(done) / len(rows)}
    for d, tag in GAP_TAGS.items():
        g = [unclean(r["gap"][d]) for r in done]
        g_inf = np.array([np.inf if r["completion"] <= cmin else unclean(r["gap"][d])
                          for r in rows])
        out[f"median_gap{tag}"] = _med(g)
        out[f"median_gap_crash_inf{tag}"] = (float(np.median(g_inf))
                                             if not np.isnan(g_inf).any() else float("nan"))
    return out


# ------------------------------------------------------------------------- analyse


def acquire_lock(root: Path, freeze_id: str) -> Path:
    """analyse runs once: an exclusive lock file, never removed by the harness."""
    lock = safe_path(root, "analysis.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise CheckFailed(f"{rel(lock)} exists: analyse has already run (or started) on "
                          "this root, and it runs once") from None
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps({"freeze_id": freeze_id, "pid": os.getpid(),
                             "start": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    return lock


def load_inputs(root: Path, rec: dict) -> tuple[dict, dict]:
    """Every reference and lap record, checked: present, its data file's sha256, the root's
    freeze_id. Nothing outside the root is read."""
    refs, laps = {}, {}
    for t in ocp_tasks(rec):
        key = ocp_key(*t)
        r = ocp_done(root, key)
        if r is None:
            raise CheckFailed(f"reference {key} is missing: run ocp first")
        if r["freeze_id"] != rec["freeze_id"]:
            raise CheckFailed(f"reference {key} belongs to another freeze")
        refs[key] = r
    for sub in rec["subjects"]:
        for c in rec["circuits"]:
            for v in rec["variants"]:
                name = variant_name(c, v)
                r = laps_done(root, sub["name"], name)
                if r is None:
                    raise CheckFailed(f"laps {sub['name']} {name} are missing: run laps first")
                if r["freeze_id"] != rec["freeze_id"] or r["checkpoint_sha256"] != sub["sha256"]:
                    raise CheckFailed(f"laps {sub['name']} {name} belong to another freeze")
                laps[(sub["name"], name)] = r
    return refs, laps


def stored_eval_agreement(rec: dict, rows: list[dict]) -> dict | None:
    """Dry run with H4R subjects: the original variants against each subject's stored
    end-of-training evaluation (A6000), recorded at freeze. Descriptive, not a check."""
    out = {}
    for sub in rec["subjects"]:
        st = sub.get("stored_eval_rdev")
        if not st:
            continue
        same, n, dlap = 0, 0, []
        for r in rows:
            if r["subject"] != sub["name"] or r["variant"] != "original" or r["circuit"] not in st:
                continue
            n += 1
            s = st[r["circuit"]]
            same += int(s["completion"] is not None and float(s["completion"]) == r["completion"])
            a, b = unclean(s["median_lap_s"]), r["median_lap_s"]
            if np.isfinite(a) and np.isfinite(b):
                dlap.append(abs(a - b))
        out[sub["name"]] = {"circuits": n, "completion_equal": same,
                            "max_abs_median_lap_diff_s": max(dlap) if dlap else None}
    return out or None


def stage_analyse(args) -> Path:
    t_start = time.time()
    if args.root is None:
        raise CheckFailed("give --root")
    root = Path(args.root)
    for f in ("analysis.lock", "analysis.json", "corners.csv"):
        if safe_path(root, f).exists():
            raise CheckFailed(f"{rel(safe_path(root, f))} exists: analyse runs once")
    rec, clearance = load_freeze(root)
    check_circuit_files(rec)
    refs, laps = load_inputs(root, rec)        # checks only; the lock comes before any number
    acquire_lock(root, rec["freeze_id"])
    from apexzero.viz.sectors import corner_sectors, sector_of

    P = rec["protocol"]
    cmin = P["laps"]["completion_min"]
    edges = tuple(P["sectors"]["type_kph"])
    deltas = [f"{d:+.1f}" for d in P["widths"]["deltas_m"]]
    inputs = {rel(safe_path(root, "freeze.json")): sha256_file(safe_path(root, "freeze.json"))}
    ocp_res = {}
    for key, r in refs.items():
        pk = safe_path(root, "ocp", f"{key}.pkl")
        raw = pk.read_bytes()
        if sha256_bytes(raw) != r["pkl_sha256"]:
            raise CheckFailed(f"{rel(pk)} changed after it was checked")
        inputs[rel(pk)] = r["pkl_sha256"]
        ocp_res[key] = pickle.loads(raw) if r["success"] else None

    rows, corner_rows = [], []
    closure = 0.0
    for c in rec["circuits"]:
        for v in rec["variants"]:
            track = load_variant(c, v, clearance)
            L = float(track.length)
            sectors = corner_sectors(track, merge_gap=P["sectors"]["merge_gap_m"])
            nominal = ocp_res[ocp_key(c, v, 0.0)]
            v_apex = ([apex_speed_kph(nominal, sec, L) for sec in sectors] if nominal
                      else [float("nan")] * len(sectors))
            types = [corner_type(x, edges) for x in v_apex]
            t_ocp = {d: (ocp_res[ocp_key(c, v, float(d))].lap_time
                         if ocp_res[ocp_key(c, v, float(d))] is not None else float("nan"))
                     for d in deltas}
            for sub in rec["subjects"]:
                lr = laps[(sub["name"], track.name)]
                nz = safe_path(root, lr["npz"])
                raw = nz.read_bytes()
                if sha256_bytes(raw) != lr["npz_sha256"]:
                    raise CheckFailed(f"{rel(nz)} changed after it was checked")
                inputs[rel(nz)] = lr["npz_sha256"]
                with np.load(io.BytesIO(raw)) as z:
                    stations, tau = z["stations"], z["tau"]
                    ood_s, ood_d = z["ood_s"], z["ood_d"].astype(float)
                completed = np.asarray(lr["completed"], bool)
                lt = np.asarray([unclean(x) for x in lr["lap_time_s"]], float)
                frac = float(completed.mean())
                med_lap = _med(lt[completed]) if completed.any() else float("nan")
                gap = {d: (med_lap / t_ocp[d] - 1.0
                           if np.isfinite(t_ocp[d]) and np.isfinite(med_lap) else float("nan"))
                       for d in deltas}
                if nominal is not None and completed.any():
                    dtc = sector_dt(stations, tau, ocp_time_at(nominal, stations), sectors, L)
                    lap_delta = lt - nominal.lap_time
                    closure = max(closure, float(np.max(np.abs(
                        np.sum(dtc[completed], axis=1) - lap_delta[completed]))))
                    dt_med = np.median(dtc[completed], axis=0)
                else:
                    dt_med = np.full(len(sectors), np.nan)
                sec_idx = sector_of(np.mod(ood_s, L), sectors, L)
                term_s = np.asarray([unclean(x) for x in lr["term_s"]], float)
                crash = np.asarray(lr["crash"], bool)
                crash_sec = sector_of(np.mod(term_s[crash], L), sectors, L)
                rows.append({
                    "subject": sub["name"], "circuit": c, "variant": v, "name": track.name,
                    "completion": frac, "circuit_completed": frac > cmin,
                    "completed": completed.tolist(), "n_completed": int(completed.sum()),
                    "median_lap_s": med_lap, "lap_sd_s": (float(np.std(lt[completed], ddof=1))
                                                          if completed.sum() > 1 else float("nan")),
                    "gap": gap, "t_ocp_s": t_ocp,
                    "d_ood_median": _med(ood_d),
                    "d_ood_p90": float(np.percentile(ood_d, 90)) if len(ood_d) else float("nan"),
                    "n_ood_steps": len(ood_d),
                    "crashes": int(crash.sum()), "timeouts": int(np.sum(lr["timeout"])),
                    "causes": {k: lr["cause"].count(k) for k in CAUSES}})
                for i, sec in enumerate(sectors):
                    corner_rows.append({
                        "subject": sub["name"], "circuit": c, "variant": v, "corner": sec.name,
                        "type": types[i], "v_apex_kph": v_apex[i], "r_min_m": sec.r_min,
                        "s_start_m": sec.s_start, "s_end_m": sec.s_end,
                        "circuit_completed": frac > cmin, "n_completed": int(completed.sum()),
                        "dt_median_s": float(dt_med[i]),
                        "d_ood_median": _med(ood_d[sec_idx == i]),
                        "n_ood_steps": int(np.sum(sec_idx == i)),
                        "crashes_here": int(np.sum(crash_sec == i))})
    if closure > 1e-6:
        raise CheckFailed(f"sector times miss the lap delta by {closure:.2e} s")

    subjects = [s["name"] for s in rec["subjects"]]
    fams = {"original": ["original"], "mirrored": ["mirrored"], "reversed": ["reversed"],
            "all": list(rec["variants"])}
    h1 = P["stats"]["h1"]
    per_subject = {}
    for s in subjects:
        per_subject[s] = {f: family_summary([r for r in rows if r["subject"] == s
                                             and r["variant"] in vs], cmin)
                          for f, vs in fams.items()}
        o = per_subject[s]["original"]
        per_subject[s]["h1_as_written"] = {
            "laps_ok": o["laps_completed_share"] >= h1["laps_completed_min"],
            "gap_ok": bool(np.isfinite(o["median_gap"])
                           and o["median_gap"] <= h1["median_gap_max"])}
    keys = list(per_subject[subjects[0]]["original"])
    headline = {f: {k: _med([per_subject[s][f].get(k, np.nan) for s in subjects])
                    for k in keys} for f in fams}
    ho = headline["original"]
    headline["h1_as_written_on_median_subject"] = {
        "laps_completed_share": ho["laps_completed_share"], "median_gap": ho["median_gap"],
        "holds": bool(ho["laps_completed_share"] >= h1["laps_completed_min"]
                      and np.isfinite(ho["median_gap"])
                      and ho["median_gap"] <= h1["median_gap_max"]),
        "subjects_meeting_both": sum(per_subject[s]["h1_as_written"]["laps_ok"]
                                     and per_subject[s]["h1_as_written"]["gap_ok"]
                                     for s in subjects)}

    iqms = {}
    for f, vs in fams.items():
        names = [variant_name(c, v) for c in rec["circuits"] for v in vs]
        M = np.full((len(subjects), len(names)), np.nan)
        M_inf = M.copy()
        for i, s in enumerate(subjects):
            for j, n in enumerate(names):
                r = next(r for r in rows if r["subject"] == s and r["name"] == n)
                g = r["gap"]["+0.0"]
                if r["circuit_completed"]:
                    M[i, j] = M_inf[i, j] = g
                else:
                    M_inf[i, j] = np.inf
        iqms[f] = {"completed_only": iqm_stratified(M, P["stats"]["n_boot"],
                                                    P["stats"]["boot_seed"]),
                   "uncompleted_as_inf": iqm_stratified(M_inf, P["stats"]["n_boot"],
                                                        P["stats"]["boot_seed"], keep_inf=True)}

    by_type = {}
    for s in subjects:
        by_type[s] = {}
        for f, vs in fams.items():
            cr = [r for r in corner_rows if r["subject"] == s and r["variant"] in vs
                  and r["circuit_completed"] and np.isfinite(r["dt_median_s"])]
            by_type[s][f] = {t: {"n_corners": sum(r["type"] == t for r in cr),
                                 "dt_sum_s": float(sum(r["dt_median_s"] for r in cr
                                                       if r["type"] == t)),
                                 "dt_median_s": _med([r["dt_median_s"] for r in cr
                                                      if r["type"] == t])}
                             for t in CORNER_TYPES}

    mirror = {}
    for c in rec["circuits"]:
        a, b = ocp_res[ocp_key(c, "original", 0.0)], ocp_res[ocp_key(c, "mirrored", 0.0)]
        mirror[c] = (b.lap_time / a.lap_time - 1.0) if a is not None and b is not None else None
    width_change = {d: _med([r["gap"][d] - r["gap"]["+0.0"] for r in rows
                             if r["circuit_completed"]]) for d in deltas if d != "+0.0"}
    t_ocp_change = {}
    for d in deltas:
        if d == "+0.0":
            continue
        ch = []
        for c in rec["circuits"]:
            for v in rec["variants"]:
                a, b = ocp_res[ocp_key(c, v, 0.0)], ocp_res[ocp_key(c, v, float(d))]
                ch.append(b.lap_time / a.lap_time - 1.0 if a is not None and b is not None
                          else float("nan"))
        t_ocp_change[d] = {"median": _med(ch), "min": float(np.nanmin(ch)) if
                           np.isfinite(ch).any() else float("nan"),
                           "max": float(np.nanmax(ch)) if np.isfinite(ch).any()
                           else float("nan")}
    ood = {"per_variant": {s: {r["name"]: {"median": r["d_ood_median"], "p90": r["d_ood_p90"],
                                           "n": r["n_ood_steps"]}
                               for r in rows if r["subject"] == s} for s in subjects},
           "spearman_completion_vs_d_ood": {s: spearman(
               [r["d_ood_median"] for r in rows if r["subject"] == s],
               [r["completion"] for r in rows if r["subject"] == s]) for s in subjects},
           "spearman_corner_dt_vs_d_ood": {s: spearman(
               [r["d_ood_median"] for r in corner_rows if r["subject"] == s
                and r["circuit_completed"]],
               [r["dt_median_s"] for r in corner_rows if r["subject"] == s
                and r["circuit_completed"]]) for s in subjects},
           "crash_corners_vs_others": {s: {
               "median_d_ood_crash_corners": _med([r["d_ood_median"] for r in corner_rows
                                                   if r["subject"] == s and r["crashes_here"]]),
               "median_d_ood_other_corners": _med([r["d_ood_median"] for r in corner_rows
                                                   if r["subject"] == s
                                                   and not r["crashes_here"]])}
               for s in subjects}}
    unconverged = [k for k, r in refs.items() if not r["success"]]
    out = {
        "label": rec["label"], "mode": rec["mode"], "freeze_id": rec["freeze_id"],
        "subjects": subjects, "circuits": rec["circuits"], "variants": rec["variants"],
        "checks": {"references_converged": f"{len(refs) - len(unconverged)}/{len(refs)}",
                   "unconverged": unconverged,
                   "retried": [k for k, r in refs.items() if len(r["attempts"]) > 1],
                   "max_abs_shooting_rel_err": _nanmax([abs(unclean(r["shooting_rel_err"]))
                                                        for r in refs.values()]),
                   "shooting_over_p2_limit": [k for k, r in refs.items() if not abs(
                       unclean(r["shooting_rel_err"])) < P2_SHOOTING_LIMIT],
                   "mirror_ocp_lap_rel_diff": mirror, "sector_sum_closure_s": closure,
                   "stored_eval_agreement": stored_eval_agreement(rec, rows)},
        "per_subject": per_subject, "headline_median_subject": headline, "iqm_gap": iqms,
        "corner_types": by_type,
        "geometry_error_bar": {"median_change_in_gap": width_change,
                               "relative_change_in_t_ocp": t_ocp_change},
        "d_ood": ood, "rows": rows, "provenance": provenance(t_start, inputs),
    }
    sio = io.StringIO()
    w = csv.DictWriter(sio, fieldnames=list(corner_rows[0]) if corner_rows else ["subject"])
    w.writeheader()
    w.writerows([clean(r) for r in corner_rows])
    write_new(safe_path(root, "corners.csv"), sio.getvalue().encode())
    write_json_new(safe_path(root, "analysis.json"), out)
    print(f"analysis written to {rel(safe_path(root, 'analysis.json'))} ({rec['label']})")
    return safe_path(root, "analysis.json")


# ------------------------------------------------------------------------- CLI


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--mode", choices=MODES, required=True)
    f.add_argument("--root", help="dry run: required; lockbox: outputs/lockbox (the default)")
    f.add_argument("--circuits", nargs="*", help="dry run only: R-dev circuits (default all)")
    f.add_argument("--subjects", help="JSON: {'subjects': [{name, checkpoint, sha256}]}")
    f.add_argument("--h4r-subjects", nargs="*", help="dry run: H4R finals by run name")
    f.add_argument("--runs", type=int, default=PROTOCOL["laps"]["runs"])
    for name in ("ocp", "laps", "analyse"):
        p = sub.add_parser(name)
        p.add_argument("--root", required=True)
        if name != "analyse":
            p.add_argument("--workers", type=int, default=3, help=f"at most {MAX_WORKERS}")
    for p in sub.choices.values():
        p.add_argument("--threads", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    lower_priority()
    {"freeze": stage_freeze, "ocp": stage_ocp, "laps": stage_laps,
     "analyse": stage_analyse}[args.stage](args)


if __name__ == "__main__":
    main()
