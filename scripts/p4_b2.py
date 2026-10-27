"""P4, option B2: five fresh Hockenheim specialists, S4-S8 (docs/prereg/p4-b2-specialists.md,
DRAFT until the tag p4-b2-prereg-v1 is placed).

    nice -n 15 uv run --no-sync python scripts/p4_b2.py <stage> [--workers 4]

The five runs train on a GPU box as the round ``p4b2`` (scripts/p4b2_runs.py through
scripts/gpu_round.sh, at the training tag p4-b2-train-v1) and come back with ``scripts/gpu_sync.sh
<box> pull-round p4b2 scripts/p4b2_runs.py``. Every stage here runs on the Mac, at most 4
processes, each at nice 15.

**Pre-tag stages** (refused once the tag exists; no S4-S8 network sees a protocol stimulus):

- ``manifest``: scripts/round_manifest.py check of the round at the training tag (a run that
  failed twice in the queue is listed as missing; any other problem stops the stage), then per
  run its config.json, agent.pkl (sha256, timesteps 199,884,800, a forward pass), the 16
  checkpoints' sha256, eval.json (reported beside K4's gate; it decides nothing) and training
  log, and the round's box manifest, queue log and status file. The agent.pkl and checkpoints
  must match the sha256 listed on the box before it was deleted (runs/p4b2-box-sha256-*.txt,
  ``BOX_SHA256_CMD``), and the load (all five at once on one GPU, Section 2) is recorded with
  any deviation from it. outputs/p4b2/pre/manifest.json.
- ``buffers``: each specialist's d_OOD buffer, the frozen ``controls.build_knn_buffer`` (50
  stochastic laps of real Hockenheim, 40,000 observations, its own obs_rms, protocol exception
  2). The agent ids S4-S8 give stream indices 135-139, distinct from S1-S3's 132-134 (checked).
  outputs/p4b2/pre/buffers/. ``--check`` rebuilds S1-S3's into outputs/p4b2/check/buffers/ and
  compares them with step 1's.
- ``starts``: each specialist's deterministic two-lap run of real Hockenheim (the frozen
  ``agents.real_lap``, protocol exception 1, ruling R1) and its tier D states at s_c - D_gap -
  100 on lap 2 (``agents.real_state``, the rule of ``stimuli.real_state_requests``), written in
  step 1's layout so that the frozen ``stimuli.load_real_states`` reads them back (ruling R21's
  attachment checked). A specialist whose run ends before the lap-2 start of a braking corner is
  class 0 and gets no stimuli (Section 9). outputs/p4b2/pre/starts/. ``--check`` does S1-S3 and
  compares them with step 1's stored laps and states.
- ``build``: the bank and the cells. The bank holds step 1's 23 registered tier D conditions on
  the 20 corners, rebuilt by the frozen builder at f6_scale 0.5 and equal to step 1's strips bit
  for bit, and the 11 certified conditions on their certified corners, built by
  scripts/p4_exploratory.py's builders, each strip's sha256 that of the frozen lists. The cells
  are built for every owner at the 9 braking corners (below). ``--owners check`` builds S1-S3's
  from step 1's states into outputs/p4b2/check/build/ and checks them against step 1's and
  option A's cells; the default builds S4-S8's into outputs/p4b2/build/.
- ``reproduce``: S1-S3 through this pipeline and Section 7's checks (a)-(d), plus their tier G
  and tier A A2 runs against step 1's. outputs/p4b2/check/reproduce.json.
- ``cm-dryrun``: the corner-matched reading on S1-S3, a DRY RUN: exploratory and post hoc for
  those networks, it changes nothing in Section 4.2. outputs/p4b2/check/cm_dryrun.json.
- ``record``: every input hash in scripts/p4b2_prereg.json, for the lead to commit alone, tag
  and push.

**Post-tag stages** (refused unless ``tag_gate`` passes and every recorded input still hashes
to its record):

- ``rollouts``: S4-S8 on tier D (their own cells), tier G and tier A A2 (step 1's cells, whose
  starts are agent-independent), 4 deterministic phases per cell, every chunk padded to 2,048
  runs (option A, Amendment 1, item 1). outputs/p4b2/rollouts/.
- ``classify``: once, into outputs/p4b2/classify/decisions.json: Section 4.1's classes, the
  corner-matched reading, each specialist's reading (Section 4.3), the recipe (Section 4.4) and
  the wording it fixes (Section 5).

**Cells.** An owner's cell at braking corner i and registered condition q starts from that
owner's lap-2 state, mapped into strip coordinates through the corner's excision by the frozen
``stimuli._strip_state``, as ``stimuli.build_real_design`` builds tier D. That function names
S1-S3 only, so its tier D loop is repeated here for any owner, and checked on S1-S3 against
step 1's cells. No OCP is solved: W and P are those of step 1's S1-state cell of the same strip.
W is equal across S1, S2 and S3 at all 449 of step 1's tier D strips, and P is read only by a
state at its run's own start, while every lap-2 state is on ruling R21's lap rule (checked). A
certified condition's cell copies the owner's condition (a) cell at that corner, as option A
did. Key rows (braking corners, v*_min, swap partners) are step 1's S1-state rows of the same
strip, the values ``analysis._share_tier_d_socp`` gives every runner.

**Integrity.** src/apexzero/interp, scripts/psychophysics.py and scripts/p4_exploratory.py are
imported read-only, and every stage refuses to run when any of them differs from its pinned git
blob or tree and sha256 (``FROZEN``). Nothing is written under outputs/psychophysics/ or
outputs/p4/, and no file under src/ is added. Only Hockenheim (R_DEV) is built or driven
(``viz.export.lockbox_guard`` first); outputs/sealed/ is never read.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

# XLA reads its flags at import: 2 threads per process, as step 1's harness and option A ran.
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               "intra_op_parallelism_threads=2")

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np
import p4_exploratory as P4A  # read-only: option A's builders, classifier and run tables
import p4b2_runs

from apexzero.interp import analysis as A
from apexzero.interp import io as pio
from apexzero.interp import registry
from apexzero.interp import stimuli as ST
from apexzero.interp.constants import N_KNOTS, S_C
from apexzero.interp.stimuli import COND_A, D_CONDITIONS, D_TRACK

SCRIPT = Path(__file__).resolve()
CODE_ROOT = SCRIPT.parents[1]           # where the imported modules live (never moved)
ROOT = CODE_ROOT                        # the repository every git call and path reads
OUT = ROOT / "outputs" / "p4b2"
PRE, CHECK, BUILD, ROLL = OUT / "pre", OUT / "check", OUT / "build", OUT / "rollouts"
DECISIONS = OUT / "classify" / "decisions.json"
DRAFT_REL = "docs/prereg/p4-b2-specialists.md"
RECORD_REL = "scripts/p4b2_prereg.json"
TAG = "p4-b2-prereg-v1"
TRAIN_TAG = "p4-b2-train-v1"            # the training round's tag (gpu_round.sh runs at it)
REMOTE = "origin"
ROUND = "p4b2"
RUNS_SCRIPT = "scripts/p4b2_runs.py"

OLD = ("S1", "S2", "S3")                # step 1's specialists: checks and the CM dry run only
NEW = tuple(p4b2_runs.AGENTS)           # S4-S8, Section 2
RUNS = {r["agent"]: r for r in p4b2_runs.runs()}
FINAL_TIMESTEPS = 199_884_800           # train.py's last whole update at 200M steps
N_CHECKPOINTS = 16
K4_COMPLETION, K4_GAP = 1.0, 0.08       # K4's gate, reported beside eval.json (decides nothing)
F6_SCALE = 0.5                          # step 1's F6 amplitudes after ruling R27's halving
MAX_WORKERS = 4
NICE = 15

# The frozen code, pinned (Section 3, Code): git blob or tree at HEAD and the file's sha256.
FROZEN = {
    "scripts/psychophysics.py": {
        "blob": "53d34e7966c4546088dd65a8a04a033cf12df798",
        "sha256": "40ac5e91109bed2a2009779ec72c04a0cafe5f60037ae5bbb5a56ab8090456a6"},
    "scripts/p4_exploratory.py": {
        "blob": "f29eec606e0c65bc8cefbad29fb5b27ad478c88c",
        "sha256": "95183edf892cc94f5000420ebf3ba4c747e10f9712e1781b57998fdaca1face8"},
    "src/apexzero/interp": {"tree": "e4c456fe3d8aacde5d0a1132f54d5fde552c8d7e"},
}
# Option A's frozen certified lists (Section 3) and step 1's stored inputs, by sha256.
CERT_LISTS = {
    "planned": ("outputs/p4/certified.json",
                "f9ab567a508f5886a294e4341704b1ca29453ac107a10161b4e74710c1e40b55"),
    "extension": ("outputs/p4/extension/certified.json",
                  "5d0d9b3134ddbf5aaa6f97a43f8868d08d66d450a260b642ea25709c4f1db8e8"),
}
STEP1 = {
    "bank.npz": (pio.STIMULI_DIR / "bank.npz",
                 "c124cb39866dbd5dd0937e9e99eef1fe582c7a3f4bf417608dafb87cdc69e1ae"),
    "key_table.npz": (pio.ANALYSIS_DIR / "key_table.npz",
                      "938fb380e2134b01a79ce500e03e8a0e840b5f817222654046bf98e3782f9e4e"),
    "decisions_all.json": (P4A.DECISIONS_ALL, P4A.DECISIONS_ALL_SHA256),
}
# Option A's stored files that P4's memorisation index reads (the generalists' reference runs on
# conditions (a) and iso, with the design they ran on).
GEN_REFERENCE = P4A.OUT / "generalists" / "reference"
OPTION_A_BUILD = P4A.OUT / "build" / "cells_meta.json"

# Section 3's F6 set: the braking corners, the 11 certified conditions and their corners.
BRAKING = (0, 1, 3, 4, 7, 8, 14, 16, 17)
CERTIFIED_CORNERS = {
    "P4A/F6s/knoise0.5e-3/x1": BRAKING,
    "P4A/F6s/knoise1e-3/x1": (0, 1, 4, 7, 8, 14, 17),
    "P4A/F6s/knoise0.5e-3/x2": BRAKING,
    "P4A/F6s/knoise0.5e-3/x3": (0, 1, 3, 4, 7, 8, 16, 17),
    "P4A/Kstr/1e-3": BRAKING,
    "P4A/Kstr/2e-3": (0, 1, 3, 4, 7, 8, 16, 17),
    "P4A/Kstr/4e-3": (0, 1, 3, 4, 7, 8, 16, 17),
    "P4A/Kstr/8e-3": BRAKING,
    "P4A/Wplus/0.5m": (0, 1, 3, 4, 7, 8, 17),
    "P4A/Wplus/1m": (0, 1, 3, 4, 8, 16, 17),
    "P4A/Kstr/16e-3": BRAKING,
}
EXTENSION_CONDS = ("P4A/Kstr/16e-3",)   # from outputs/p4/extension/certified.json
FAMILIES = ("F6s", "Kstr", "Wplus")
B2_F6 = (*A.F6_CONDS, *CERTIFIED_CORNERS)     # Section 4.1: the 4 registered and the 11

# Sections 4.2-5: the corner-matched reading, the readings and the wording they fix.
CM_LABELS = {"CM-1": "fragile (CM)", "CM-2": "context-keyed (CM)", "CM-6": "mixed (CM)",
             "not testable": "CM not testable"}
READINGS = ("fragile", "context-keyed", "corner-keyed", "reads the road", "mixed", "undecided",
            "not competent")
RECIPE_MIN = 3                          # of 5: at most one reading can reach it
KEY_PHRASES = {"width": "constant width", "after": "the straightened road after the corner",
               "before": "the straightened road before the corner"}
NO_KEY = "no key named"
WORDING = {   # (the wording, the note beside it), Section 5's table
    "fragile": (
        ("The Hockenheim specialists are fragile off their training manifold. "
         "Certified-irrelevant road changes no further from it break them as the failing "
         "single-factor change does, so memorisation cannot be diagnosed."),
        "The demo does not call them memorisers."),
    "context-keyed": (
        ("Context-keyed memorisation, keyed by [factor]. Certified-irrelevant changes at least "
         "as far off the manifold leave them intact, while one real-road factor breaks them."),
        "Their gap from the optimum is a fact about Hockenheim only."),
    "corner-keyed": ("Corner-keyed memorisation.", "The protocol's interpretation table."),
    "reads the road": ("Reads the road within its envelope.", ""),
    "reads the road, beyond": ("Reads the road beyond its envelope.", ""),
    "mixed": ("No single label.", "The specialists are described per network and per corner."),
    "undecided": (
        ("Tied to Hockenheim's context (the single-factor failures and MI), with memorisation "
         "and fragility not separable by certified-irrelevant road changes."), ""),
    "not competent": (
        "The recipe does not reliably give specialists competent from these states.",
        "P4 is not answerable."),
    "none": ("No recipe-level description.", "Each specialist's reading is reported."),
}
PAIR_COLUMNS = ("agent", "timesteps", "pair", "tiers", "approach", "R_seg", "no_cap",
                "braking", "excluded", "n_runs", "det_complete", "det_crashes")


class CheckFailed(SystemExit):
    """A built-in check failed: the item stops for an amendment (Section 7)."""


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", flush=True)
    raise CheckFailed(1)


# ------------------------------------------------------------------------- provenance


def git_run(*args: str) -> subprocess.CompletedProcess:
    """Every git call goes through here (the tests point ROOT at a scratch repository)."""
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=False)


def git(*args: str) -> str:
    out = git_run(*args)
    return out.stdout.strip() if out.returncode == 0 else ""


def tree_blobs(rev: str, paths=()) -> dict[str, str] | None:
    """{path: blob id} of every file of ``rev`` (a commit, or a tree id with paths relative to
    it) under ``paths``; None when git cannot list it."""
    proc = git_run("ls-tree", "-r", "-z", rev, "--", *paths)
    if proc.returncode != 0:
        return None
    out = {}
    for entry in proc.stdout.split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            if meta.split()[1] == "blob":
                out[path] = meta.split()[2]
    return out


def byte_mismatches(want: dict[str, str]) -> list[str]:
    """The paths whose bytes in the working tree are not the blob ``want`` names, or that are
    missing. git diff and git status trust the index for a file marked assume-unchanged or
    skip-worktree, so the bytes are hashed here (``git hash-object --no-filters``)."""
    present = [p for p in want if (ROOT / p).is_file()]
    have = {}
    if present:
        proc = subprocess.run(["git", "hash-object", "--no-filters", "--stdin-paths"], cwd=ROOT,
                              input="".join(p + "\n" for p in present), capture_output=True,
                              text=True, check=False)
        ids = proc.stdout.split()
        if proc.returncode != 0 or len(ids) != len(present):
            return [f"git hash-object failed ({proc.stderr.strip()})"]
        have = dict(zip(present, ids, strict=True))
    return [p for p, b in want.items() if have.get(p) != b]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path: Path | str) -> str:
    """Repo-relative, so no output carries an absolute local path."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT.resolve()))
    except ValueError:
        return Path(path).name if Path(path).is_absolute() else str(path)


def clean(x):
    """JSON-safe: NaN and infinities as null, NumPy values as Python ones (option A's)."""
    return P4A.clean(x)


def json_bytes(obj) -> bytes:
    return (json.dumps(clean(obj), indent=1, sort_keys=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, obj) -> str:
    """Atomic write; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    data = json_bytes(obj)
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return sha256_bytes(data)


def write_new(path: Path, obj) -> str:
    """Write a JSON file that must not exist yet (atomic: a hard link fails if the path
    appeared meanwhile); returns its sha256."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{rel(path)} exists: this stage runs once")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(obj)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return sha256_bytes(data)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def be_nice() -> None:
    try:
        cur = os.nice(0)
        if cur < NICE:
            os.nice(NICE - cur)
    except OSError:
        pass


def provenance(stage: str, t0: float, inputs: dict | None = None, gate: dict | None = None
               ) -> dict:
    return {"stage": stage, "script": rel(SCRIPT), "script_sha256": sha256_file(SCRIPT),
            "git_head": git("rev-parse", "HEAD"),
            "changed_paths": [ln for ln in git("status", "--porcelain", "--untracked-files=no")
                              .splitlines() if ln.strip()],
            "frozen_clean": not frozen_problems(), "argv": sys.argv[1:],
            "draft": {"path": DRAFT_REL, "sha256": _sha_or_none(ROOT / DRAFT_REL)},
            "tag_gate": gate, "inputs_sha256": dict(sorted((inputs or {}).items())),
            "seconds": round(time.time() - t0, 1), "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system(),
                            "XLA_FLAGS": os.environ.get("XLA_FLAGS", "")}}


def _sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if Path(path).is_file() else None


# ------------------------------------------------------------------------- freeze and lockbox


def frozen_problems(pins: dict | None = None) -> list[str]:
    """Every way the frozen code differs from its pins (Section 3): the git blob or tree at
    HEAD, the working file's sha256 (a tree's files: their bytes against the pinned tree's
    blobs), a working-tree change or an untracked file, and the imported modules' files being
    elsewhere."""
    out = []
    for path, pin in (FROZEN if pins is None else pins).items():
        got = git("rev-parse", f"HEAD:{path}")
        want = pin.get("tree") or pin.get("blob")
        if got != want:
            out.append(f"{path}: {got[:12] or 'missing'} at HEAD, pinned {want[:12]}")
        st = git_run("status", "--porcelain", "--untracked-files=all", "--", path)
        if st.returncode or st.stdout.strip():
            out.append(f"{path}: changed or untracked files in the working tree")
        if "sha256" in pin:
            p = ROOT / path
            if not p.is_file() or sha256_file(p) != pin["sha256"]:
                out.append(f"{path}: sha256 is not the pinned {pin['sha256'][:12]}")
        if "tree" in pin:
            blobs = tree_blobs(pin["tree"])
            bad = (["the pinned tree is not in this repository"] if blobs is None else
                   byte_mismatches({f"{path}/{p}": b for p, b in blobs.items()}))
            if bad:
                out.append(f"{path}: not the pinned tree's files byte for byte ({bad[:3]})")
    if pins is None:
        here = {Path(P4A.__file__).resolve(): CODE_ROOT / "scripts" / "p4_exploratory.py",
                Path(A.__file__).resolve().parent: CODE_ROOT / "src" / "apexzero" / "interp"}
        out += [f"{k} is imported instead of {v}" for k, v in here.items() if k != v]
    return out


def require_frozen() -> None:
    probs = frozen_problems()
    if probs:
        raise SystemExit("the frozen code differs from its pins: " + "; ".join(probs))


def tag_exists() -> bool:
    return git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}").returncode == 0


def refuse_after_tag(what: str) -> None:
    """Pre-tag stages write what the record binds: once the tag exists they would change it."""
    if tag_exists():
        raise SystemExit(f"refusing to run {what} after the tag {TAG}")


def hockenheim(name: str = D_TRACK):
    """The only circuit this item builds or drives (option A's ``hockenheim_track``:
    ``lockbox_guard`` first, then Hockenheim only)."""
    return P4A.hockenheim_track(name)


def stream_id(agent: str) -> int:
    """``controls.build_knn_buffer``'s stream index for an agent: its id's character codes
    summed (S1 132, ..., S8 139)."""
    return sum(ord(c) for c in agent)


def check_streams() -> dict[str, int]:
    ids = {a: stream_id(a) for a in (*OLD, *NEW)}
    if [ids[a] for a in NEW] != list(range(135, 140)) or len(set(ids.values())) != len(ids):
        fail(f"the buffer streams are not distinct or not 135-139 for S4-S8: {ids}")
    return ids


def step1_inputs() -> dict[str, str]:
    """Step 1's stored inputs, each checked against its sha256."""
    for name, (path, sha) in STEP1.items():
        got = sha256_file(path)
        if got != sha:
            fail(f"{name} is not step 1's ({got[:12]}, want {sha[:12]})")
    return {name: sha for name, (_, sha) in STEP1.items()}


def certified() -> dict[str, dict]:
    """The 11 certified conditions, read from option A's two frozen lists (each tracked,
    unchanged and of its pinned sha256): name -> family, corners and per-corner strip sha256.
    They must be Section 3's table, in its order."""
    out = {}
    for key, (path, sha) in CERT_LISTS.items():
        p = ROOT / path
        if not p.is_file() or sha256_file(p) != sha:
            fail(f"{path} is not option A's frozen list ({sha[:12]})")
        tracked = git_run("ls-files", "--error-unmatch", path).returncode == 0
        if not tracked or git("status", "--porcelain", "--", path):
            fail(f"{path} must be tracked and unchanged")
        d = read_json(p)
        for n in d["entering"]:
            e = d["conditions"][n]
            out[n] = dict(list=key, family=e["family"], ladder=e["ladder"], kind=e["kind"],
                          amplitude=e["amplitude"], unit=e["unit"], f6_scale=e["f6_scale"],
                          corners=tuple(int(c) for c in e["certified_corners"]),
                          strips={str(k): v for k, v in e["strip_sha256_per_corner"].items()})
    if list(out) != list(CERTIFIED_CORNERS) or any(out[n]["corners"] != c for n, c in
                                                    CERTIFIED_CORNERS.items()):
        fail("the frozen lists do not give Section 3's 11 conditions and corners")
    if [n for n, e in out.items() if e["list"] == "extension"] != list(EXTENSION_CONDS):
        fail("the extension's list does not give K-str 16e-3 alone")
    return out


def comparators(cert: dict[str, dict]) -> dict[int, tuple[str, ...]]:
    """Section 4.2: the certified conditions certified at each braking corner, in Section 3's
    order (the registered F6 conditions are not comparators)."""
    return {i: tuple(n for n, e in cert.items() if i in e["corners"]) for i in BRAKING}


def key_table() -> dict:
    kt = pio.load_table(STEP1["key_table.npz"][0])
    if tuple(P4A.braking_from_keys(kt)) != BRAKING:
        fail(f"step 1's braking corners are {P4A.braking_from_keys(kt)}, not {BRAKING}")
    return kt


# ------------------------------------------------------------------------- snapshots


def old_finals() -> dict[str, tuple[str, str]]:
    """S1-S3's finals in step 1's manifest: (repo-relative path, sha256)."""
    fin = P4A.finals()
    return {a: (fin[a].path, fin[a].sha256) for a in OLD}


def read_manifest() -> dict:
    p = PRE / "manifest.json"
    if not p.exists():
        raise SystemExit("no outputs/p4b2/pre/manifest.json: run the manifest stage")
    return read_json(p)


def new_finals(man: dict | None = None) -> dict[str, tuple[str, str]]:
    """S4-S8's agent.pkl in the manifest (present runs only): (path, sha256)."""
    man = read_manifest() if man is None else man
    return {a: (e["final"]["path"], e["final"]["sha256"]) for a, e in man["networks"].items()}


def load_snapshot(agent: str, path: str, sha: str):
    """``agents.load_agent`` on a repo-relative path (so no record holds an absolute one),
    after checking its sha256."""
    from apexzero.interp import agents

    if sha256_file(ROOT / path) != sha:
        fail(f"{path} is not the recorded snapshot of {agent}")
    return agents.load_agent(Path(path), agent)


# ------------------------------------------------------------------------- manifest


def shard_manifests() -> list[tuple[Path, dict]]:
    return [(p, read_json(p)) for p in sorted((ROOT / "runs").glob(f"{ROUND}-manifest-*of*.json"))]


def round_check(train_tag: str) -> tuple[int, str]:
    """scripts/round_manifest.py check of this round at the training tag: (exit code, report)."""
    proc = subprocess.run([sys.executable, "scripts/round_manifest.py", "check", "--round",
                           ROUND, "--runs-script", RUNS_SCRIPT, "--tag", train_tag], cwd=ROOT,
                          capture_output=True, text=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def check_problems(report: str) -> list[str] | None:
    """The problem lines of a failed check report (those after its 'FAILED:' line); None when
    the report has no such line (the check did not run to its end)."""
    lines = report.splitlines()
    for k, ln in enumerate(lines):
        if ln.startswith("FAILED:"):
            return [x.strip() for x in lines[k + 1:] if x.strip()]
    return None


def unexplained(problems: list[str] | None, failed: set[str]) -> list[str]:
    """Problems that are not about a run that failed twice (Section 9: such a run is missing,
    with its queue events); a check that did not finish is unexplained as a whole."""
    if problems is None:
        return ["round_manifest.py check did not finish"]
    return [p for p in problems if not any(p.startswith(f"{n}:") for n in failed)]


BOX_SHA256 = f"{ROUND}-box-sha256-*of*.txt"     # in runs/, listed from the box (Section 8, 2)
BOX_SHA256_CMD = ("ssh <box> 'cd ~/apexzero && sha256sum runs/hockenheim-specialist-mlp-200M-s[4-8]"
                  "/agent.pkl runs/hockenheim-specialist-mlp-200M-s[4-8]/ckpt/*.pkl' > "
                  f"runs/{ROUND}-box-sha256-0of1.txt")


def box_sha256_files() -> list[Path]:
    return sorted((ROOT / "runs").glob(BOX_SHA256))


def read_sha256sum(text: str) -> dict[str, str]:
    """``sha256sum``'s lines (the hash, a space, a mode character, the path): path -> hash."""
    out = {}
    for ln in text.splitlines():
        h, _, path = ln.partition(" ")
        if len(h) == 64 and path[:1] in (" ", "*"):
            out[path[1:]] = h
    return out


def box_sha256_problems(networks: dict[str, dict], listed: dict[str, str]) -> tuple[int, list[str]]:
    """Each present run's agent.pkl and checkpoints against the sha256 listed on the box before
    it was deleted (Section 8, step 2): (files that match, problems)."""
    n, probs = 0, []
    for e in networks.values():
        for f in (e["final"], *e["checkpoints"]):
            want = listed.get(f["path"])
            if want is None:
                probs.append(f"{f['path']}: not in the box's sha256 list")
            elif want != f["sha256"]:
                probs.append(f"{f['path']}: not the box's file (sha256)")
            else:
                n += 1
    return n, probs


def load_summary(shards: list[tuple[Path, dict]], recs: dict[str, dict]) -> dict:
    """Section 2's load: all five runs at once on one GPU (GPUS=1 LANES_PER_GPU=5). Any other
    schedule (the x4 box's fallback, or the lanes left at gpu_round.sh's default of 4, which
    puts S8 after S4) is a deviation, recorded here and in the findings."""
    boxes = [{k: m.get(k) for k in ("shard", "box_type", "gpu_count", "gpus", "lanes_per_gpu")}
             for _, m in shards]
    runs = {a: {k: recs.get(RUNS[a]["name"], {}).get(k) for k in ("gpu", "lane")} for a in NEW}
    slots = {(r["gpu"], r["lane"]) for r in runs.values()}
    decided = (len(boxes) == 1 and boxes[0]["gpus"] == 1 and boxes[0]["lanes_per_gpu"] == len(NEW)
               and len(slots) == len(NEW) and all(g == 0 for g, _ in slots))
    return dict(rule="all five at once on one GPU: GPUS=1 LANES_PER_GPU=5 (Section 2)",
                boxes=boxes, runs=runs, decided=bool(decided),
                deviation=None if decided else "not the decided load (Section 2): a deviation")


def manifest_entry(agent: str, run: dict, rec: dict) -> tuple[dict | None, list[str]]:
    """One present run: its config, final snapshot (loaded, one forward pass), checkpoints,
    evaluation and logs, or the problems that keep it out."""
    from apexzero.interp import agents

    name = run["name"]
    d = ROOT / "runs" / name
    files = {"config": d / "config.json", "final": d / "agent.pkl", "eval": d / "eval.json",
             "log": ROOT / "runs" / f"{name}.log"}
    probs = [f"{name}: {rel(p)} is missing" for p in files.values() if not p.is_file()]
    if probs:
        return None, probs
    ag = agents.load_agent(Path(rel(files["final"])), agent)
    if ag.timesteps != FINAL_TIMESTEPS:
        probs.append(f"{name}: agent.pkl has {ag.timesteps} timesteps, not {FINAL_TIMESTEPS}")
    if not agents.forward_pass_ok(ag):
        probs.append(f"{name}: agent.pkl gives no finite forward pass")
    ckpts = sorted((d / "ckpt").glob("*.pkl"))
    if len(ckpts) != N_CHECKPOINTS or ckpts[-1].name != f"{FINAL_TIMESTEPS:012d}.pkl":
        probs.append(f"{name}: {len(ckpts)} checkpoints, not {N_CHECKPOINTS} ending at "
                     f"{FINAL_TIMESTEPS}")
    cfg = read_json(files["config"])
    if cfg.get("tracks") != [D_TRACK] or cfg.get("seed") != run["seed"]:
        probs.append(f"{name}: config.json records tracks {cfg.get('tracks')}, seed "
                     f"{cfg.get('seed')}")
    ev = read_json(files["eval"]).get(D_TRACK, {})
    gap, comp = ev.get("gap"), ev.get("completion")
    rms = ag.obs_rms
    log1 = ROOT / "runs" / f"{name}.log.1"
    return dict(
        agent=agent, name=name, seed=run["seed"], run=f"runs/{name}", status=rec.get("status"),
        queue={k: rec.get(k) for k in ("start", "end", "lane", "gpu")},
        config=dict(path=rel(files["config"]), sha256=sha256_file(files["config"]),
                    tracks=cfg.get("tracks"), seed=cfg.get("seed"), ppo=cfg.get("ppo"),
                    env=cfg.get("env")),
        final=dict(path=rel(files["final"]), sha256=sha256_file(files["final"]),
                   size_bytes=files["final"].stat().st_size, timesteps=int(ag.timesteps),
                   log_std=[float(v) for v in ag.log_std], forward_pass_ok=True,
                   obs_rms_sha256=pio.sha256_arrays(*(np.asarray(a) for a in
                                                      (rms.mean, rms.var, rms.count)))),
        checkpoints=[dict(path=rel(p), sha256=sha256_file(p)) for p in ckpts],
        eval=dict(path=rel(files["eval"]), sha256=sha256_file(files["eval"]), completion=comp,
                  median_lap_s=ev.get("median_lap_s"), ocp_lap_s=ev.get("ocp_lap_s"), gap=gap,
                  k4_gate=bool(comp is not None and gap is not None and comp >= K4_COMPLETION
                               and gap <= K4_GAP)),
        log=dict(path=rel(files["log"]), sha256=sha256_file(files["log"])),
        first_attempt_log=dict(path=rel(log1), sha256=sha256_file(log1)) if log1.is_file()
        else None), probs


def stage_manifest(args) -> None:
    t0 = time.time()
    require_frozen()
    refuse_after_tag("manifest")
    out = PRE / "manifest.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{rel(out)} exists (--force rewrites it before the tag)")
    rc, report = round_check(args.train_tag)
    shards = shard_manifests()
    recs = {r["name"]: r for _, m in shards for r in m.get("runs", [])}
    failed = {n for n, r in recs.items() if r.get("status") == "failed"}
    if rc != 0:
        bad = unexplained(check_problems(report), failed)
        if bad:
            print(report, flush=True)
            raise SystemExit("round_manifest.py check failed other than by runs that failed "
                             "twice: " + "; ".join(bad))
    networks, missing, probs = {}, {}, []
    for a in NEW:
        run = RUNS[a]
        rec = recs.get(run["name"], {})
        if run["name"] in failed:
            logs = {s: sha256_file(ROOT / "runs" / f"{run['name']}{s}")
                    for s in (".log", ".log.1") if (ROOT / "runs" / f"{run['name']}{s}").is_file()}
            missing[a] = dict(name=run["name"], reason="failed twice in the queue (Section 9)",
                              queue=rec, logs_sha256=logs)
            continue
        e, p = manifest_entry(a, run, rec)
        probs += p
        if e is not None:
            networks[a] = e
    if probs:
        raise SystemExit("runs not as Section 2 needs them: " + "; ".join(probs))
    # The final checkpoints' (and every checkpoint's) sha256 against the box's (Section 8, 2).
    lists = box_sha256_files()
    if not lists:
        raise SystemExit(f"no runs/{BOX_SHA256}: the box's sha256 list is written before the box "
                         f"is deleted (Section 8, step 2): {BOX_SHA256_CMD}")
    listed = {}
    for p in lists:
        listed.update(read_sha256sum(p.read_text()))
    n_box, box_probs = box_sha256_problems(networks, listed)
    if box_probs:
        raise SystemExit("the pulled runs are not the box's: " + "; ".join(box_probs[:10]))
    load = load_summary(shards, recs)
    if not load["decided"]:
        print(f"note: {load['deviation']}: {load['boxes']}", flush=True)
    commit = git("rev-parse", "--verify", "--quiet", f"{args.train_tag}^{{commit}}")
    logs = {rel(p): sha256_file(p) for p in sorted((ROOT / "runs").glob(f"{ROUND}-*"))
            if p.is_file() and p.suffix in (".log", ".tsv", ".json", ".txt")}
    boxes = [dict(path=rel(p), sha256=sha256_file(p),
                  **{k: m.get(k) for k in ("shard", "head", "tree_clean", "tags", "box_type",
                                           "gpu_model", "gpu_count", "gpus", "lanes_per_gpu",
                                           "xla_mem_fraction", "written", "run_list_sha256")})
             for p, m in shards]
    k4 = {a: dict(completion=e["eval"]["completion"], gap=e["eval"]["gap"],
                  passes=e["eval"]["k4_gate"]) for a, e in networks.items()}
    sha = write_json(out, {
        "status": "P4-B2 pre-registration DRAFT: the five runs as trained; no S4-S8 network "
                  "has seen a protocol stimulus",
        "round": ROUND, "runs_script": RUNS_SCRIPT, "train_tag": args.train_tag,
        "train_commit": commit, "check": dict(returncode=rc, report=report.splitlines()),
        "boxes": boxes, "load": load, "round_files_sha256": logs,
        "box_sha256": dict(lists=[rel(p) for p in lists], files_equal=n_box),
        "networks": networks, "missing": missing,
        "k4_gate": dict(rule="completion 100% and gap <= 8% (reported, decides nothing)",
                        per_network=k4),
        "final_timesteps": FINAL_TIMESTEPS,
        "provenance": provenance("manifest", t0, {RUNS_SCRIPT: sha256_file(ROOT / RUNS_SCRIPT)})})
    print(f"wrote {rel(out)} ({sha[:12]}): {len(networks)} networks, {len(missing)} missing. "
          f"Commit it: git add -f {rel(out)}", flush=True)


# ------------------------------------------------------------------------- buffers


def stage_buffers(args) -> None:
    from apexzero.interp import controls

    t0 = time.time()
    require_frozen()
    refuse_after_tag("buffers")
    ids = check_streams()
    if args.check:
        snaps, out_dir = old_finals(), CHECK / "buffers"
    else:
        snaps, out_dir = new_finals(), PRE / "buffers"
    track = hockenheim()
    rec = {}
    for a, (path, sha) in snaps.items():
        dst = out_dir / f"knn_{a}.npz"
        if dst.exists() and not args.force:
            raise SystemExit(f"{rel(dst)} exists (--force rebuilds it before the tag)")
        ag = load_snapshot(a, path, sha)
        buf = controls.build_knn_buffer(ag, track=track)
        dst.parent.mkdir(parents=True, exist_ok=True)
        buf.save(dst)
        rec[a] = dict(path=rel(dst), sha256=sha256_file(dst), stream=ids[a], snapshot=path,
                      snapshot_sha256=sha, n_obs=len(buf.obs),
                      n_steps=int(buf.meta["n_steps"]), lolo_median_nn=buf.lolo_median_nn)
        if args.check:
            ref = controls.KnnBuffer.load(pio.CONTROLS_DIR / f"knn_{a}.npz")
            rec[a]["equals_step1"] = bool(
                all(np.array_equal(getattr(buf, f), getattr(ref, f)) for f in
                    ("obs", "act", "lap", "act_sampled"))
                and buf.lolo_median_nn == ref.lolo_median_nn)
        print(f"  {a}: buffer of {len(buf.obs)} observations (stream {ids[a]}, "
              f"{buf.meta['n_steps']} steps)", flush=True)
    sha = write_json(out_dir / "buffers.json", {
        "status": ("CHECK: S1-S3's buffers rebuilt and compared with step 1's" if args.check
                   else "P4-B2 pre-tag d_OOD buffers (protocol exception 2)"),
        "streams": ids, "buffers": rec,
        "provenance": provenance("buffers", t0)})
    if args.check and not all(r["equals_step1"] for r in rec.values()):
        fail(f"rebuilt buffers differ from step 1's: {[a for a, r in rec.items()]}")
    print(f"wrote {rel(out_dir / 'buffers.json')} ({sha[:12]})", flush=True)


# ------------------------------------------------------------------------- starts


def lap_states(rl, track) -> tuple[dict, dict, dict]:
    """The tier D states of one real lap: key 'D/<i>' -> RealState fields and trace index, at
    s_c - D_gap - 100 on lap 2 (``real_state_requests``' rule), and the corners it does not
    reach, with the reason."""
    import psychophysics as PS

    from apexzero.interp import agents

    states, index, unreachable = {}, {}, {}
    for dc in ST.hockenheim_corners(track):
        key, s_t = f"D/{dc.index}", float(dc.corner.s_start - dc.before)
        try:
            st = agents.real_state(rl, s_t, 2)
        except ValueError as e:
            unreachable[key] = str(e)
            continue
        states[key] = dataclasses.asdict(st)
        index[key] = int(PS._state_index(rl, s_t, 2))
    return states, index, unreachable


def stage_starts(args) -> None:
    from apexzero.interp import agents

    t0 = time.time()
    require_frozen()
    refuse_after_tag("starts")
    if args.check:
        snaps, out_dir = old_finals(), CHECK / "starts"
    else:
        snaps, out_dir = new_finals(), PRE / "starts"
    if (out_dir / ST.REAL_STATES_FILE).exists() and not args.force:
        raise SystemExit(f"{rel(out_dir)} exists (--force rebuilds it before the tag)")
    track = hockenheim()
    states, index, summary = {}, {}, {}
    for a, (path, sha) in snaps.items():
        ag = load_snapshot(a, path, sha)
        rl = agents.real_lap(ag, D_TRACK)
        f = out_dir / ST.REAL_LAPS_DIR / f"{a}_{D_TRACK}.npz"
        f.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(x=rl.x, prev_action=rl.prev_action, accel=rl.accel, valid=rl.valid,
                      term_step=rl.term_step, cause=rl.cause, length=rl.length,
                      v_start=rl.v_start, ds=float(track.ds))
        pio._atomic(f, lambda fh, arrays=arrays: np.savez(fh, **arrays))
        states[a], index[a], lost = lap_states(rl, track)
        braking_lost = [i for i in BRAKING if f"D/{i}" in lost]
        summary[a] = dict(snapshot=path, snapshot_sha256=sha, lap=rel(f), lap_sha256=sha256_file(f),
                          term_step=int(rl.term_step), cause=rl.cause, n_states=len(states[a]),
                          unreachable=lost, braking_unreachable=braking_lost,
                          class0=bool(braking_lost))
        print(f"  {a}: {'two laps' if rl.term_step < 0 else rl.cause}; {len(states[a])} of 20 "
              f"states" + (f"; class 0 (corners {braking_lost})" if braking_lost else ""),
              flush=True)
    write_json(out_dir / ST.REAL_STATES_FILE, states)
    write_json(out_dir / ST.REAL_STATES_INDEX_FILE, index)
    loaded = ST.load_real_states(out_dir)          # R21's attachment, checked by the frozen code
    rules = {a: sorted({"lap" if st.trace_index > 0 else "run start" for st in v.values()})
             for a, v in loaded.items()}
    if any(r != ["lap"] for r in rules.values() if r):
        fail(f"a lap-2 state is its run's own start: {rules}")
    check = None
    if args.check:
        check = starts_check(out_dir, loaded)
    files = {rel(p): sha256_file(p) for p in sorted(out_dir.rglob("*"))
             if p.is_file() and p.name != "starts.json"}
    sha = write_json(out_dir / "starts.json", {
        "status": ("CHECK: S1-S3's real laps and tier D states recomputed" if args.check else
                   "P4-B2 pre-tag start states (protocol exception 1, R1)"),
        "rule": "s_c - D_gap - 100 m on lap 2 (stimuli.real_state_requests), agents.real_state",
        "braking": list(BRAKING), "agents": summary, "files_sha256": files, "check": check,
        "provenance": provenance("starts", t0)})
    if check is not None and not check["ok"]:
        fail(f"S1-S3's recomputed starts differ from step 1's: {check}")
    print(f"wrote {rel(out_dir / 'starts.json')} ({sha[:12]})", flush=True)


def starts_check(out_dir: Path, loaded: dict) -> dict:
    """S1-S3's recomputed laps and tier D states against step 1's, bit for bit."""
    ref = ST.load_real_states(pio.STIMULI_DIR)
    out = {}
    for a in OLD:
        mine = {k: v for k, v in loaded.get(a, {}).items()}
        theirs = {k: v for k, v in ref.get(a, {}).items() if k.startswith("D/")}
        lap = {}
        with np.load(out_dir / ST.REAL_LAPS_DIR / f"{a}_{D_TRACK}.npz") as z, \
                np.load(pio.STIMULI_DIR / ST.REAL_LAPS_DIR / f"{a}_{D_TRACK}.npz") as y:
            for k in ("x", "prev_action", "accel", "valid", "term_step"):
                lap[k] = bool(np.array_equal(z[k], y[k]))
        out[a] = dict(states_equal=mine == theirs, n_states=len(mine), n_step1=len(theirs),
                      lap_equal=lap)
    ok = all(v["states_equal"] and all(v["lap_equal"].values()) for v in out.values())
    return dict(ok=bool(ok), per_agent=out)


# ------------------------------------------------------------------------- the design


def build_bank(track, sbank, swap: dict[int, int], cert: dict) -> tuple[ST.StimulusBank, dict]:
    """Step 1's registered tier D rows (rebuilt by the frozen builder at f6_scale 0.5 and
    compared with step 1's bit for bit, then copied with their records), and the certified
    conditions on their certified corners (option A's builders; strip sha256 against the
    frozen lists), assembled as option A assembled its bank."""
    cat = {c["name"]: c for c in (*P4A.MAIN_CATALOGUE, *P4A.extension_catalogue())}
    reg, rows = [], []
    checks = {"registered_vs_step1": {}, "certified_vs_lists": {}}
    for dc in ST.hockenheim_corners(track):
        i = dc.index
        conds, _ = ST.hockenheim_conditions(i, track, swap.get(i), F6_SCALE)
        for cond in D_CONDITIONS:
            if cond not in conds:
                continue
            st = sbank.by_name(f"D/{i}/{cond}")
            p = conds[cond]
            checks["registered_vs_step1"][st.name] = bool(all(
                np.array_equal(getattr(p, f), getattr(sbank, f)[st.stim_id])
                for f in ("kappa", "w_left", "w_right")))
            reg.append(st.stim_id)
        here = [n for n, e in cert.items() if i in e["corners"]]
        if not here:
            continue
        prof = {}
        if any(cert[n]["list"] == "planned" for n in here):
            prof.update(P4A.new_conditions(dc, track, swap.get(i))[0])
        if any(cert[n]["list"] == "extension" for n in here):
            prof.update(P4A.extension_conditions(dc, track)[0])
        rr = ST._RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS, track)
        for n in here:
            p, c = prof[n], cat[n]
            checks["certified_vs_lists"][f"D/{i}/{n}"] = bool(
                pio.sha256_arrays(p.kappa, p.w_left, p.w_right) == cert[n]["strips"][str(i)])
            src = rr.source_info(condition=n, corner_index=i, D_gap=dc.D_gap, key=f"D/{i}",
                                 family=c["family"], amplitude=c["amplitude"], unit=c["unit"],
                                 p4_exploratory=True,
                                 **({"f6_scale": c["f6_scale"]} if c["f6_scale"] else {}))
            rows.append(ST._Row(f"D/{i}/{n}", p, "hockenheim", dc.corner.sign, None, {"D"}, src,
                                {"cap_binds"} if p.capped.any() else set()))
    idmap = {s: j for j, s in enumerate(reg)}
    reg_stims = [replace(sbank.stimuli[s], stim_id=idmap[s],
                         mirror_id=idmap.get(sbank.stimuli[s].mirror_id, -1)) for s in reg]
    new_stims, _ = ST._assemble(rows, start_id=len(reg))

    def stack(f):
        return np.concatenate([getattr(sbank, f)[reg],
                               np.stack([getattr(r.profile, f) for r in rows])])

    bank = ST.StimulusBank(stack("kappa"), stack("w_left"), stack("w_right"),
                           reg_stims + new_stims, sbank.ds)
    return bank, checks


def step1_s1_cells(sbank, scells) -> dict[tuple[int, str], ST.Cell]:
    """Step 1's tier D cells run from S1's state, by (corner, condition)."""
    out = {}
    for c in scells:
        if "D" in c.tiers and tuple(c.runners) == tuple(ST.RUNNERS_FROM_S1):
            src = sbank.stimuli[c.stim_id].source
            out[(int(src["corner_index"]), str(src["condition"]))] = c
    return out


def owner_cells(owner: str, states: dict, bank, s1: dict, excisions: dict, cert: dict,
                cell0: int, pair0: int) -> tuple[list[ST.Cell], list[dict], int]:
    """One owner's tier D cells at the braking corners (module docstring, Cells): the
    registered conditions from the owner's lap-2 state as ``build_real_design`` makes them
    (condition (a) and the mirror share a pair, every other condition has its own), then each
    certified condition as a copy of the owner's (a) cell. Returns the cells, their meta rows
    and the next free pair id."""
    by_row = {s.name: s.stim_id for s in bank.stimuli}
    cells, meta, pair = [], [], pair0
    for i in BRAKING:
        st = states[f"D/{i}"]
        pair_a, pair = pair, pair + 1
        base = None
        for cond in D_CONDITIONS:
            ref = s1.get((i, cond))
            if ref is None:
                continue                    # swap A exists only where the corner has a partner
            sid = by_row[f"D/{i}/{cond}"]
            mir = cond == "F4/mirror"
            ss = ST._strip_state(excisions[i], st, mir)
            if cond in (COND_A, "F4/mirror"):
                pid = pair_a
            else:
                pid, pair = pair, pair + 1
            flags = ({ST.RUN_START_FLAG} if ss.run_start else set()) | set(bank.stimuli[sid].flags)
            c = ST.Cell(cell_id=cell0 + len(cells), stim_id=sid, pair_id=pid, tiers=("D",),
                        approach="real", v0=float(ss.x0[3]), D=float(S_C - ss.x0[0]), W=ref.W,
                        P=ref.P, n_phases=ref.n_phases, x0=ss.x0, prev_action0=ss.pa,
                        accel0=ss.ac, state_source=f"{owner}:lap{st.lap}@D/{i}",
                        flags=tuple(sorted(flags)), mirrored=mir, runners=(owner,),
                        x0_before=ss.x0_before, prev_action_before=ss.pa_before,
                        accel_before=ss.ac_before)
            cells.append(c)
            meta.append(dict(cell=c.cell_id, owner=owner, corner=i, condition=cond,
                             registered=True, family="registered", s1_cell=int(ref.cell_id)))
            if cond == COND_A:
                base = c
        for n, e in cert.items():
            if i not in e["corners"]:
                continue
            sid = by_row[f"D/{i}/{n}"]
            c = replace(base, cell_id=cell0 + len(cells), stim_id=sid, pair_id=pair,
                        mirrored=False,
                        flags=tuple(sorted(set(base.flags) | set(bank.stimuli[sid].flags))))
            pair += 1
            cells.append(c)
            meta.append(dict(cell=c.cell_id, owner=owner, corner=i, condition=n,
                             registered=False, family=e["family"],
                             s1_cell=int(s1[(i, COND_A)].cell_id)))
    return cells, meta, pair


def design_keys(meta: list[dict], kt: dict) -> dict:
    """The key columns ``cond_table`` reads, per cell: a registered cell takes step 1's key row
    of the S1-state cell of the same strip; a certified one has no s-OCP (v*_min is read only
    on (a) and the doses, braking only on (a))."""
    kidx = {int(c): j for j, c in enumerate(kt["cell"])}
    rows = []
    for m in meta:
        if m["registered"]:
            j = kidx[m["s1_cell"]]
            rows.append(dict(cell=m["cell"], corner=m["corner"], condition=m["condition"],
                             v_min_socp=float(kt["v_min_socp"][j]),
                             braking_socp=bool(kt["braking_socp"][j]),
                             swap_j=int(kt["swap_j"][j])))
        else:
            rows.append(dict(cell=m["cell"], corner=m["corner"], condition=m["condition"],
                             v_min_socp=float("nan"), braking_socp=False, swap_j=-1))
    return A._to_table(rows, ("cell", "corner", "condition", "v_min_socp", "braking_socp",
                              "swap_j"))


CELL_FIELDS = ("tiers", "approach", "v0", "D", "W", "n_phases", "x0", "prev_action0", "accel0",
               "state_source", "null_cell", "flags", "mirrored", "x0_before",
               "prev_action_before", "accel_before")


def cell_differences(mine: ST.Cell, ref: ST.Cell, with_p: bool) -> list[str]:
    fields = (*CELL_FIELDS, "P") if with_p else CELL_FIELDS
    return [f for f in fields if getattr(mine, f) != getattr(ref, f)]


def check_cells(cells: list, meta: list[dict], bank, sbank, scells) -> dict:
    """S1-S3's cells against step 1's (registered conditions) and option A's (certified
    conditions): every field but the ids and, for S2 and S3, P (this build gives every owner
    the S1-state cell's P; step 1 gave each its own, read only by a run-start state)."""
    runners = {"S1": tuple(ST.RUNNERS_FROM_S1), "S2": ("S2",), "S3": ("S3",)}
    step1 = {}
    for c in scells:
        if "D" in c.tiers:
            src = sbank.stimuli[c.stim_id].source
            step1[(tuple(c.runners), int(src["corner_index"]), str(src["condition"]))] = c
    option_a = {}
    for d in (P4A.OUT / "build", P4A.OUT / "extension" / "build"):
        _, acells = ST.load_design(d)
        for m in read_json(d / "cells_meta.json")["cells"]:
            if not m["registered"]:
                option_a[(m["owner"], m["corner"], m["condition"])] = acells[m["cell"]]
    diffs, n = {}, Counter()
    for c, m in zip(cells, meta, strict=True):
        a = m["owner"]
        ref = (step1.get((runners[a], m["corner"], m["condition"])) if m["registered"]
               else option_a.get((a, m["corner"], m["condition"])))
        if ref is None:
            diffs[f"{a} D/{m['corner']}/{m['condition']}"] = ["no reference cell"]
            continue
        d = cell_differences(c, ref, with_p=a == "S1")
        n["registered" if m["registered"] else "certified"] += 1
        if d:
            diffs[f"{a} D/{m['corner']}/{m['condition']}"] = d
    return dict(ok=not diffs, compared=dict(n), differences=diffs,
                note="P compared for S1 only (see check_cells)")


def stage_build(args) -> None:
    t0 = time.time()
    require_frozen()
    refuse_after_tag("build")
    check = args.owners == "check"
    out = CHECK / "build" if check else BUILD
    if (out / "cells.json").exists() and not args.force:
        raise SystemExit(f"{rel(out)} is built (--force rebuilds it before the tag)")
    inputs = step1_inputs()
    sbank, scells = ST.load_design(pio.STIMULI_DIR)
    kt = key_table()
    track = hockenheim()
    swap = {int(k): int(v) for k, v in pio.read_json(P4A.SWAP).items()}
    cert = certified()
    bank, checks = build_bank(track, sbank, swap, cert)
    for k, v in checks.items():
        bad = [n for n, ok in v.items() if not ok]
        if bad:
            fail(f"{k}: {len(bad)} strips differ ({bad[:5]})")
    print(f"bank: {len(bank)} rows; {len(checks['registered_vs_step1'])} registered strips equal "
          f"step 1's, {len(checks['certified_vs_lists'])} certified strips match the lists",
          flush=True)
    excluded = {}
    if check:
        owners = list(OLD)
        states = ST.load_real_states(pio.STIMULI_DIR)
    else:
        starts = read_json(PRE / "starts" / "starts.json")["agents"]
        owners = [a for a in NEW if a in starts and not starts[a]["class0"]]
        excluded = {a: "class 0: the real lap ends before a braking corner's start (Section 9)"
                    for a in NEW if a in starts and starts[a]["class0"]}
        states = ST.load_real_states(PRE / "starts")
    s1 = step1_s1_cells(sbank, scells)
    excisions = {dc.index: ST._RealRoad(D_TRACK, dc.corner, dc.before, dc.after, N_KNOTS,
                                        track).exc
                 for dc in ST.hockenheim_corners(track) if dc.index in BRAKING}
    cells, meta, pair = [], [], 0
    for a in owners:
        cs, ms, pair = owner_cells(a, states[a], bank, s1, excisions, cert, len(cells), pair)
        cells += cs
        meta += ms
    rules = Counter(c.phase_rule() for c in cells)
    if set(rules) != {"lap"}:
        fail(f"every cell must be on R21's lap rule: {dict(rules)}")
    cell_check = check_cells(cells, meta, bank, sbank, scells) if check else None
    if cell_check is not None and not cell_check["ok"]:
        fail(f"S1-S3's cells differ from step 1's or option A's: {cell_check['differences']}")
    out.mkdir(parents=True, exist_ok=True)
    bank.save(out)
    ST.save_cells(out / "cells.json", cells)
    write_json(out / "cells_meta.json", {"cells": meta, "braking": list(BRAKING),
                                         "owners": owners, "excluded": excluded})
    pio.save_table(out / "keys.npz", design_keys(meta, kt))
    strips = {s.name: pio.sha256_arrays(bank.kappa[s.stim_id], bank.w_left[s.stim_id],
                                        bank.w_right[s.stim_id]) for s in bank.stimuli}
    bank_hash = bank.content_hash()
    check_build = CHECK / "build" / "build.json"
    if not check and check_build.exists() and read_json(check_build)["bank_hash"] != bank_hash:
        fail("the S4-S8 bank differs from the bank S1-S3's checks ran on")
    for name in ("cells.json", "bank.npz"):
        inputs[f"step1 {name}"] = sha256_file(pio.STIMULI_DIR / name)
    inputs.update({path: sha for path, sha in CERT_LISTS.values()})
    inputs["swap.json"] = sha256_file(P4A.SWAP)
    if not check:
        inputs["pre/starts/starts.json"] = sha256_file(PRE / "starts" / "starts.json")
    sha = write_json(out / "build.json", {
        "status": ("CHECK: S1-S3's cells, for the reproduction and the CM dry run" if check
                   else "P4-B2 pre-tag design: stimuli only; no S4-S8 network has run on it"),
        "owners": owners, "excluded": excluded, "n_rows": len(bank), "n_cells": len(cells),
        "cells_per_owner": dict(Counter(m["owner"] for m in meta)),
        "design_hash": ST.design_hash(bank, cells), "bank_hash": bank_hash,
        "checks": {k: dict(n=len(v), all_equal=all(v.values())) for k, v in checks.items()},
        "cell_check": cell_check, "phase_rules": dict(rules), "strip_sha256": strips,
        "rule": "W and P from step 1's S1-state cell of the same strip; certified cells copy "
                "the owner's (a) cell (module docstring, Cells)",
        "provenance": provenance("build", t0, inputs)})
    print(f"wrote {rel(out)}: {len(bank)} rows, {len(cells)} cells for {owners} ({sha[:12]})",
          flush=True)


# ------------------------------------------------------------------------- rollouts


_W: dict = {}


def design_dir(name: str) -> Path:
    return {"check": CHECK / "build", "main": BUILD, "step1": pio.STIMULI_DIR}[name]


def _worker_init() -> None:
    from apexzero.interp import measures

    be_nice()
    pio.assert_x64_off()
    if not measures.tables_path().exists():
        raise RuntimeError("step 1's model tables are missing")
    _W.update(tables=measures.model_tables(), designs={})


def _design(name: str):
    if name not in _W["designs"]:
        bank, cells = ST.load_design(design_dir(name))
        _W["designs"][name] = (bank, cells, {c.cell_id: c for c in cells},
                               ST.design_hash(bank, cells))
    return _W["designs"][name]


def bitwise(det: dict, stored: Path, cell_map: dict[int, int]) -> dict:
    """Runs on the cells of ``cell_map`` (this design's id -> the stored file's id) against the
    stored file's deterministic runs: states and termination step bit for bit."""
    with np.load(stored, allow_pickle=False) as z:
        sd = {k: z[f"det_{k}"] for k in ("cell", "phase", "term_step", "x")}
    look = {(int(c), int(p)): j for j, (c, p) in enumerate(zip(sd["cell"], sd["phase"],
                                                              strict=True))}
    n = same = absent = 0
    worst = 0.0
    for i in range(len(det["cell"])):
        sc = cell_map.get(int(det["cell"][i]))
        if sc is None:
            continue
        j = look.get((sc, int(det["phase"][i])))
        if j is None:
            absent += 1
            continue
        T = min(det["x"].shape[1], sd["x"].shape[1])
        a, b = det["x"][i][:T], sd["x"][j][:T]
        ok = np.array_equal(a, b) and int(det["term_step"][i]) == int(sd["term_step"][j])
        n += 1
        same += ok
        if not ok:
            worst = max(worst, float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))))
    return {"n": n, "bitwise_equal": same, "absent": absent, "max_abs_diff_unequal": worst}


def _rollout_job(job: dict) -> dict:
    """One agent's deterministic runs (4 phases) on a list of cells of one design: the run
    table, the phase-0 d_OOD against a buffer (``psychophysics._d_ood_from_det``, frozen), and
    bitwise comparisons with stored runs."""
    import traceback

    from apexzero.interp import agents, controls, rollout

    out = dict(label=job["label"], agent=job["agent"], batch=job["batch"],
               design=job["design"], cells=job["cells"])
    c0, t0 = time.process_time(), time.time()
    try:
        if sha256_file(ROOT / job["path"]) != job["sha256"]:
            raise RuntimeError(f"{job['path']} is not the recorded snapshot")
        bank, cells, by_id, dh = _design(job["design"])
        ag = agents.load_agent(Path(job["path"]), job["agent"])
        cs = [by_id[c] for c in job["cells"]]
        runs = rollout.plan_runs(cs, bank, "det", 0, agent=job["agent"])
        if len(runs) != 4 * len(cs):
            raise RuntimeError(f"{len(runs)} runs for {len(cs)} cells")
        det = P4A.run_padded(rollout.agent_policy(ag.cfg), ag.actor_params, ag.obs_rms, bank,
                             cells, runs)
        dst = ROOT / job["out"]
        rollout.save_rollouts(dst, {"det": det}, dict(
            agent=job["agent"], timesteps=ag.timesteps, log_std=ag.log_std.tolist(),
            path=job["path"], sha256=job["sha256"], design_hash=dh,
            note="p4_b2.py; chunks padded to 2048 runs"))
        rt = P4A.run_rows(job["agent"], ag.timesteps, det, bank, by_id, _W["tables"])
        pio.save_table(dst.with_name(dst.stem + "_runs.npz"), rt)
        out.update(n_runs=len(runs), timesteps=int(ag.timesteps), design_hash=dh)
        if job.get("buffer"):
            import psychophysics as PS

            buf = controls.KnnBuffer.load(ROOT / job["buffer"])
            dd = PS._d_ood_from_det(det, job["agent"], bank, cells, ag.obs_rms, buf)
            out["d_ood"] = [[int(c), float(v)] for (_, c), v in dd.items()]
            out["buffer"] = dict(path=job["buffer"], sha256=sha256_file(ROOT / job["buffer"]))
        if job.get("compare"):
            out["compare"] = [dict(bitwise(det, ROOT / cmp["file"],
                                           {int(k): int(v) for k, v in cmp["map"].items()}),
                                   file=cmp["file"]) for cmp in job["compare"]]
    except Exception:  # noqa: BLE001  (reported to the main process)
        out["error"] = traceback.format_exc()
    out["cpu"], out["wall"] = time.process_time() - c0, time.time() - t0
    return out


def _pool(jobs: list[dict], workers: int, on_result) -> None:
    """A spawned pool of at most 4 processes, each loading the designs it needs."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if not jobs:
        return
    n = max(1, min(int(workers), MAX_WORKERS, len(jobs)))
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"),
                             initializer=_worker_init) as ex:
        for f in as_completed([ex.submit(_rollout_job, j) for j in jobs]):
            on_result(f.result())


def rollout_batch(batch_dir: Path, design: str, cells_of: dict[str, list[int]],
                  snaps: dict[str, tuple[str, str]], workers: int,
                  buffers: dict[str, str] | None = None,
                  compare: dict[str, list[dict]] | None = None) -> dict[str, dict]:
    """Run (or resume) one batch, one job per agent, cached in ``batch_dir``/results.json in
    option A's layout (so its ``cond_tables`` reads it)."""
    res_path = batch_dir / "results.json"
    have = read_json(res_path)["agents"] if res_path.exists() else {}
    jobs = []
    for a, cs in cells_of.items():
        cs = sorted(int(c) for c in cs)
        if a in have and not have[a].get("error"):
            if have[a].get("cells") != cs or have[a].get("design") != design:
                raise SystemExit(f"{rel(batch_dir)}/{a} on disk ran other cells")
            continue
        path, sha = snaps[a]
        jobs.append(dict(label=f"{a} {batch_dir.name}", agent=a, batch=batch_dir.name,
                         design=design, path=path, sha256=sha, cells=cs,
                         out=rel(batch_dir / f"{a}.npz"), buffer=(buffers or {}).get(a),
                         compare=(compare or {}).get(a)))

    def done(r):
        st = "ERROR" if r.get("error") else f"{r.get('n_runs')} runs"
        print(f"  {r['label']}: {st} ({r['wall']:.0f} s)", flush=True)
        have[r["agent"]] = {k: v for k, v in r.items() if k != "label"}
        write_json(res_path, {"agents": have})

    batch_dir.mkdir(parents=True, exist_ok=True)
    _pool(jobs, workers, done)
    errs = {a: r["error"] for a, r in have.items() if r.get("error")}
    if errs:
        for a, e in errs.items():
            print(f"{a}: {e}", flush=True)
        raise SystemExit(f"rollout errors in {rel(batch_dir)}: {sorted(errs)}")
    return have


def tier_ga_cells(scells) -> list[int]:
    """Step 1's tier G null straights and tier A A2 cells (every agent runs them)."""
    return sorted(c.cell_id for c in scells if "G" in c.tiers
                  or ("A" in c.tiers and c.approach == "A2"))


# ------------------------------------------------------------------------- tables


def tier_table(rt: dict, kt: dict, pair_of: dict[int, int]) -> dict:
    """Per pair, the columns of ``analysis.pair_values`` that P4's class 4 and the tier G rule
    read, by ``_pair_row``'s rules: the key row of the pair (its first cell with direction > 0,
    else its first), with its tiers, approach, R_seg, no_cap and braking; the exclusion (no key,
    a status in EXCL_ALL, or multi-basin); and the deterministic runs' n_runs, det_complete and
    det_crashes."""
    kidx = {int(c): i for i, c in enumerate(A._it(kt, "cell"))}
    kdir = A._it(kt, "direction", 0)
    det = A._st(rt, "mode") == "det"
    agent, ts, cell = A._st(rt, "agent"), A._it(rt, "timesteps", 0), A._it(rt, "cell")
    completed, crashed = A._bo(rt, "completed"), A._bo(rt, "crashed")
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(det):
        groups.setdefault((str(agent[i]), int(ts[i]), int(pair_of[int(cell[i])])), []).append(i)
    rows = []
    for (a, t, p), idx in sorted(groups.items()):
        idx = np.asarray(idx)
        kr = None
        for c in sorted({int(c) for c in cell[idx] if int(c) in kidx}):
            kr = kidx[c] if kr is None else kr
            if int(kdir[kidx[c]]) > 0:
                kr = kidx[c]
                break

        def K(name, default, kr=kr):
            if kr is None or name not in kt:
                return default
            v = kt[name][kr]
            return v.item() if hasattr(v, "item") else v

        status = str(K("key_status", "missing"))
        if kr is None or status in ("missing", A.NO_KEY_FLAG):
            excluded = A.NO_KEY_FLAG
        elif status in A.EXCL_ALL:
            excluded = status
        elif bool(K("multi_basin", False)):
            excluded = "multi_basin"
        else:
            excluded = ""
        rows.append(dict(agent=a, timesteps=t, pair=p, tiers=str(K("tiers", "")),
                         approach=str(K("approach", "")), R_seg=float(K("R_seg", np.nan)),
                         no_cap=bool(K("no_cap", False)), braking=bool(K("braking", False)),
                         excluded=excluded, n_runs=int(idx.size),
                         det_complete=int(completed[idx].sum()),
                         det_crashes=int(crashed[idx].sum())))
    return A._to_table(rows, PAIR_COLUMNS)


def tier_tables(rt: dict, kt: dict, scells) -> tuple[dict, dict]:
    """(tier A, tier G) pair tables, as step 1's ``decisions`` selects them."""
    t = tier_table(rt, kt, {c.cell_id: c.pair_id for c in scells})
    return pio.select(t, A.has_tier(t, "A")), pio.select(t, A.has_tier(t, "G"))


def tier_a_completion(tier_a: dict) -> float:
    """P4's class 4 statistic, as ``p4_classify`` computes it: tier A A2 completion over pairs
    with R_seg >= 10.7 m (no-cap and EXCL_ALL pairs out)."""
    ma = A._mask(tier_a, approach="A2", braking=None, exclude=A.EXCL_ALL) & \
        (A._fl(tier_a, "R_seg") >= A.P4_TIER_A_R)
    nr = A._it(tier_a, "n_runs", 0)[ma].sum()
    return float(A._it(tier_a, "det_complete", 0)[ma].sum() / nr) if nr else float("nan")


def generalist_table(kt: dict) -> dict:
    """The seed-0 generalists' cond tables on (a) and iso from option A's reference runs (step
    1's finals, checked bit for bit against step 1 there): P4's memorisation index."""
    meta = read_json(OPTION_A_BUILD)["cells"]
    keys = P4A.keys_table(meta, kt, {})
    gens = list(registry.seed0_generalists())
    ct = P4A.cond_tables([GEN_REFERENCE], gens, keys)
    return pio.concat_tables([ct[a] for a in gens])


def option_a_inputs() -> dict[str, str]:
    files = [OPTION_A_BUILD, *(GEN_REFERENCE / f"{a}_runs.npz"
                               for a in registry.seed0_generalists())]
    return {rel(p): sha256_file(p) for p in files}


def corner_cells(ct: dict) -> dict[tuple[int, str], tuple[bool, float]]:
    """done(i, q) and d(i, q) per braking corner and condition (Section 4.2's inputs)."""
    return {(int(c), str(n)): (bool(k), float(d)) for c, n, k, d in zip(
        A._it(ct, "corner"), A._st(ct, "condition"), A._bo(ct, "complete"),
        A._fl(ct, "d_ood"), strict=True)}


# ------------------------------------------------------------------------- Section 4


def failing_conditions(C: dict[str, float]) -> list[str]:
    """P4's failing list: the single-factor F1 conditions with C <= 0.5."""
    return [s for s in A.F1_SINGLE.values() if C.get(s, float("nan")) <= A.P4_FAIL]


def p4_class(ct: dict, generalists: dict | None, f6: tuple[str, ...] = B2_F6,
             tier_a: dict | None = None, tier_g: dict | None = None) -> dict:
    """Section 4.1: P4's ordered rules with the given F6 set (option A's ``classify``, the
    frozen ``p4_classify`` with the F6 set as a parameter), as a JSON-ready dict with its
    failing conditions and vacuity."""
    r = P4A.classify(ct, generalists, f6, tier_a, tier_g)
    d = P4A.result_dict(r)
    d["failing"] = failing_conditions(r.C)
    d["vacuous"] = d["details"].get("vacuous") if r.cls == 2 else None
    return d


def cm_type(d_f: float, comps: list[tuple[str, float, bool]]) -> str:
    """Section 4.2's type of one failing cell (first match wins), from its comparators'
    (name, d, done): fragile, keyed, mixed or unmatched."""
    if any(d <= d_f and not ok for _, d, ok in comps):
        return "fragile"
    above = [ok for _, d, ok in comps if d >= d_f]
    if above and all(above):
        return "keyed"
    if above:
        return "mixed"
    return "unmatched"


def cm_class(counts: dict[str, int]) -> dict:
    """The specialist's CM class from its cell types: n = fragile + keyed + mixed; not
    testable if n < 3, CM-1 if fragile >= n / 2, CM-2 if keyed >= 0.8 n, else CM-6."""
    fr, ke, mx = (int(counts.get(k, 0)) for k in ("fragile", "keyed", "mixed"))
    n = fr + ke + mx
    if n < 3:
        cls = "not testable"
    elif 2 * fr >= n:
        cls = "CM-1"
    elif 5 * ke >= 4 * n:
        cls = "CM-2"
    else:
        cls = "CM-6"
    return dict(cm=cls, label=CM_LABELS[cls], n=n)


def cm_reading(cells: dict[tuple[int, str], tuple[bool, float]], failing: list[str],
               comps: dict[int, tuple[str, ...]], braking: tuple[int, ...] = BRAKING) -> dict:
    """Section 4.2 for one specialist: every failing cell (a braking corner and failing
    condition, not completed, with a finite d) typed against the certified conditions
    certified at its corner. A comparator without a run or a finite d is listed as unused."""
    out = []
    for f in failing:
        for i in braking:
            if (i, f) not in cells:
                continue
            done_f, d_f = cells[(i, f)]
            if done_f or not math.isfinite(d_f):
                continue
            used, unused = [], []
            for g in comps.get(i, ()):
                ok, d = cells.get((i, g), (None, float("nan")))
                if ok is None or not math.isfinite(d):
                    unused.append(g)
                else:
                    used.append((g, d, ok))
            out.append(dict(corner=i, condition=f, d=d_f, type=cm_type(d_f, used),
                            comparators=[dict(condition=g, d=d, done=ok) for g, d, ok in used],
                            unused=unused))
    counts = Counter(c["type"] for c in out)
    return dict(cells=out, counts={k: int(counts.get(k, 0)) for k in
                                   ("fragile", "keyed", "mixed", "unmatched")},
                **cm_class(counts))


def reading_of(res: dict | None, cm: dict | None = None, missing: str | None = None) -> dict:
    """Section 4.3: a specialist's reading from its Section 4.1 result (``p4_class``) and its
    CM reading. 'via' is P4 or CM; class -1 is a deviation and reads 'undecided' (Section 9)."""
    if missing is not None:
        return dict(reading="missing", label="missing", via=None, why=missing, cls=None,
                    keys=[], class4=False)
    cls = int(res["cls"])
    keys = list(res.get("key_factors") or [])

    def r(reading, label=None, via="P4", **kw):
        return dict(reading=reading, label=label or reading, via=via, cls=cls,
                    keys=keys if reading == "context-keyed" else [], class4=cls == 4, **kw)

    if cls == 0:
        return r("not competent")
    if cls == -1:
        return r("undecided", "undecided (P4 not testable: a deviation, Section 9)")
    if cls == 1:
        return r("fragile")
    if cls == 2 and not res.get("vacuous"):
        return r("context-keyed")
    if cls == 2:
        c = (cm or {}).get("cm", "not testable")
        kind = {"CM-1": "fragile", "CM-2": "context-keyed", "CM-6": "mixed"}.get(c, "undecided")
        return r(kind, CM_LABELS.get(c, "undecided") if kind != "undecided" else "undecided",
                 via="CM", cm=c)
    if cls == 3:
        return r("corner-keyed")
    if cls in (4, 5):
        return r("reads the road", f"reads the road ({'generally' if cls == 4 else 'locally'})")
    return r("mixed")


def key_text(keys: list[tuple[str, int]], m: int) -> str:
    """Section 4.4's [factor]: every key held in the majority, with its count, most held first
    ('constant width (3 of 4) and the straightened road after the corner (1 of 4)')."""
    parts = [f"{KEY_PHRASES.get(k, NO_KEY)} ({n} of {m})" for k, n in keys]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def recipe(readings: dict[str, dict], n_total: int = len(NEW)) -> dict:
    """Section 4.4: a reading is claimed when at least 3 of the 5 hold it. CM readings count
    with their kind, classes 4 and 5 together, 'undecided' on its own; a missing specialist
    counts against every reading (the denominator stays 5)."""
    counts = Counter(r["reading"] for r in readings.values() if r["reading"] != "missing")
    held = [k for k in READINGS if counts.get(k, 0) >= RECIPE_MIN]
    if len(held) > 1:
        raise AssertionError(f"two readings reach {RECIPE_MIN}: {held}")
    claimed = held[0] if held else None
    members = [a for a, r in readings.items() if r["reading"] == claimed]
    out = dict(claimed=claimed, counts={k: int(counts.get(k, 0)) for k in READINGS},
               members=members, n_total=n_total, n_via_cm=sum(readings[a]["via"] == "CM"
                                                             for a in members),
               missing=[a for a, r in readings.items() if r["reading"] == "missing"],
               keys=[], class4=sum(bool(r.get("class4")) for r in readings.values()))
    if claimed == "context-keyed":
        kc = Counter()
        for a in members:
            kc.update(readings[a]["keys"] or [""])
        order = [*A.F1_FACTORS, ""]
        out["keys"] = sorted(kc.items(), key=lambda kv: (-kv[1], order.index(kv[0])))
    out["wording"] = wording(out)
    return out


def wording(rec: dict) -> dict:
    """Section 5: the wording the recipe reading fixes for protocol-v1 and the demo."""
    claimed = rec["claimed"]
    if claimed is None:
        key = "none"
    elif claimed == "reads the road" and rec["class4"] >= RECIPE_MIN:
        key = "reads the road, beyond"
    else:
        key = claimed
    text, note = WORDING[key]
    if claimed == "context-keyed":
        text = text.replace("[factor]", key_text(rec["keys"], len(rec["members"])))
    return dict(key=key, text=text, note=note)


REGISTERED_KIND = {0: "not competent", -1: "not testable", 1: "fragile", 2: "context-keyed",
                   3: "corner-keyed", 4: "reads the road", 5: "reads the road", 6: "mixed"}


def registered_recipe(results: dict[str, dict | None], n_total: int = len(NEW)) -> dict:
    """Section 4.4's last item: the same 3-of-5 rule on the Section 4.1 classes alone, a vacuous
    class 2 flagged (reported beside the recipe reading; it decides nothing)."""
    kinds = {a: ("missing" if r is None else REGISTERED_KIND[int(r["cls"])])
             for a, r in results.items()}
    counts = Counter(k for k in kinds.values() if k != "missing")
    held = [k for k, n in counts.items() if n >= RECIPE_MIN]
    claimed = held[0] if held else None
    members = [a for a, k in kinds.items() if k == claimed]
    return dict(claimed=claimed, counts=dict(counts), members=members, n_total=n_total,
                n_vacuous=sum(bool(results[a] and results[a].get("vacuous")) for a in members),
                classes={a: (None if r is None else int(r["cls"])) for a, r in results.items()})


def expected_directions(per: dict[str, dict], rec: dict, k4: dict) -> dict:
    """Section 6's EB1-EB7, read off the results (each decides nothing). EB7's first part,
    'well above the generalists', has no threshold in the draft: its numbers are reported and
    it is not judged."""
    present = {a: p for a, p in per.items() if p.get("p4") is not None}
    fails = {a: p["p4"]["failing"] for a, p in present.items() if p["p4"]["failing"]}
    width_after = {A.F1_SINGLE["width"], A.F1_SINGLE["after"]}
    eb1 = dict(k4=k4, C_a={a: p["p4"]["C"].get(COND_A) for a, p in present.items()},
               holds=bool(len(present) == len(NEW) and all(v.get("passes") for v in k4.values())
                          and all((p["p4"]["C"].get(COND_A) or 0) >= A.P4_GATE
                                  for p in present.values())))
    eb2 = dict(failing=fails, holds=sum(bool(set(f) & width_after) for f in fails.values())
               >= RECIPE_MIN)
    eb3 = dict(holds=bool(fails) and all(present[a]["p4"]["cls"] == 2
                                         and present[a]["p4"]["vacuous"] for a in fails))
    eb4 = dict(holds=all(p["p4"]["cls"] == 6 for a, p in present.items() if a not in fails))
    cms = {a: p["cm"]["cm"] for a, p in present.items() if p.get("cm")}
    eb5 = dict(cm=cms, holds=bool(cms) and all(v == "CM-6" for v in cms.values()))
    eb6 = dict(claimed=rec["claimed"], holds=rec["claimed"] in (None, "undecided"))
    crashes = {a: p["p4"]["details"].get("tier_g_crash") for a, p in present.items()}
    eb7 = dict(MI={a: p["p4"]["MI"] for a, p in present.items()}, tier_g_crash=crashes,
               holds_mi=None, holds_tier_g=sum(bool(v) for v in crashes.values()) >= 2)
    return {"EB1": eb1, "EB2": eb2, "EB3": eb3, "EB4": eb4, "EB5": eb5, "EB6": eb6, "EB7": eb7}


# ------------------------------------------------------------------------- S1-S3 checks


def option_a_maps(meta: list[dict]) -> dict[str, list[dict]]:
    """Per S1-S3: this design's cells against option A's stored runs (registered, planned and
    extension batches), by (owner, corner, condition)."""
    mine = {(m["owner"], m["corner"], m["condition"]): m["cell"] for m in meta}
    out = {a: [] for a in OLD}
    for build, batches in ((P4A.OUT / "build", (P4A.OUT / "specialists" / "registered",
                                                P4A.OUT / "specialists" / "new")),
                           (P4A.OUT / "extension" / "build",
                            (P4A.OUT / "extension" / "specialists" / "new",))):
        theirs = {(m["owner"], m["corner"], m["condition"]): m["cell"]
                  for m in read_json(build / "cells_meta.json")["cells"]}
        for d in batches:
            res = read_json(d / "results.json")["agents"]
            for a in OLD:
                ran = set(res[a]["cells"])
                cmap = {mine[k]: c for k, c in theirs.items() if k[0] == a and c in ran
                        and k in mine}
                out[a].append(dict(file=rel(d / f"{a}.npz"), batch=rel(d),
                                   map={str(k): v for k, v in cmap.items()}))
    return out


def d_ood_vs_option_a(have: dict, maps: dict[str, list[dict]]) -> dict:
    """Each run's d_OOD against option A's for the same cell (same buffers, same code)."""
    out = {}
    for a in OLD:
        mine = {int(c): float(v) for c, v in have[a]["d_ood"]}
        n, worst = 0, 0.0
        for cmp in maps[a]:
            theirs = {int(c): float(v) for c, v in
                      read_json(ROOT / cmp["batch"] / "results.json")["agents"][a]["d_ood"]}
            for k, v in cmp["map"].items():
                if v in theirs:
                    n += 1
                    worst = max(worst, abs(mine[int(k)] - theirs[v]))
        out[a] = dict(n=n, max_abs_diff=worst)
    return out


def _json_eq(a, b) -> bool:
    return json.loads(json.dumps(clean(a), sort_keys=True)) == json.loads(
        json.dumps(clean(b), sort_keys=True))


def classes_check(ct: dict, gen: dict, dec: dict, stored: dict, sets: dict) -> dict:
    """S1-S3's classes under option A's sets against a stored classes.json (classes per set,
    the recipe and the registered recipe), with option A's tier G flag and no tier A table."""
    out, results, reg = {}, {}, {}
    for a in OLD:
        tg = P4A.tier_g_table(dec[a]["details"]["tier_g_crash"])
        per = {}
        for name, f6 in sets.items():
            r = P4A.classify(ct[a], gen, f6, tier_g=tg)
            per[name] = P4A.result_dict(r)
            if name == "extended":
                results[a] = r
            if name == "registered":
                reg[a] = r
        want = stored["specialists"][a]["classes"]
        out[a] = {name: _json_eq(per[name], want.get(name)) for name in sets}
        out[a]["sets_equal"] = sorted(sets) == sorted(want)
    rc = A.p4_recipe([results[a] for a in OLD])
    rr = A.p4_recipe([reg[a] for a in OLD])
    out["recipe"] = _json_eq(dict(verdict=rc.verdict, details=rc.details), stored["recipe"])
    out["recipe_registered"] = _json_eq(dict(verdict=rr.verdict, details=rr.details),
                                        stored["recipe_registered"])
    ok = all(all(v.values()) if isinstance(v, dict) else v for v in out.values())
    return dict(ok=bool(ok), per=out)


def option_a_sets(cert: dict, with_extension: bool) -> dict[str, tuple[str, ...]]:
    """Option A's classification sets: the planned run's (registered, extended, per family) or
    the extension's (registered, planned, extended, per family)."""
    planned = tuple(n for n in cert if n not in EXTENSION_CONDS)
    ent = (*planned, *EXTENSION_CONDS) if with_extension else planned
    sets = {"registered": A.F6_CONDS}
    if with_extension:
        sets["planned"] = (*A.F6_CONDS, *planned)
    sets["extended"] = (*A.F6_CONDS, *ent)
    for fam in FAMILIES:
        members = tuple(n for n in ent if cert[n]["family"] == fam)
        if members:
            sets[f"family:{fam}"] = (*A.F6_CONDS, *members)
    return sets


def _same_value(x, y) -> bool:
    if isinstance(x, float | np.floating) and isinstance(y, float | np.floating):
        return bool((np.isnan(x) and np.isnan(y)) or x == y)
    return bool(x == y)


def pairs_check(tier_a: dict, tier_g: dict, agent: str, ts: int) -> dict:
    """This pipeline's tier A A2 and tier G pair columns against step 1's stored pair table."""
    stored = pio.load_table(pio.ANALYSIS_DIR / f"{agent}_pairs.npz")
    look = {int(p): i for i, (p, t) in enumerate(zip(A._it(stored, "pair"),
                                                     A._it(stored, "timesteps", 0),
                                                     strict=True)) if int(t) == ts}
    cols = ("tiers", "approach", "R_seg", "no_cap", "excluded", "n_runs", "det_complete",
            "det_crashes")
    n, bad = 0, []
    for t in (tier_a, tier_g):
        for i in range(A._n(t)):
            j = look.get(int(t["pair"][i]))
            if j is None:
                bad.append(f"pair {int(t['pair'][i])} not in step 1's table")
                continue
            n += 1
            for c in cols:
                x, y = t[c][i], stored[c][j]
                if not _same_value(x, y):
                    bad.append(f"pair {int(t['pair'][i])} {c}: {x!r} != {y!r}")
    sm = np.isin(A._it(stored, "pair"), list(A._it(tier_a, "pair"))) & \
        (A._it(stored, "timesteps", 0) == ts)
    return dict(n=n, differences=bad[:20], n_differences=len(bad),
                tier_a_completion=tier_a_completion(tier_a),
                tier_a_completion_step1=tier_a_completion(pio.select(stored, sm)),
                tier_g_crash=bool((A._it(tier_g, "det_crashes", 0) > 0).any()))


def stage_reproduce(args) -> None:
    t0 = time.time()
    require_frozen()
    refuse_after_tag("reproduce")
    out = CHECK / "reproduce.json"
    if not (CHECK / "build" / "cells.json").exists():
        raise SystemExit("run build --owners check first")
    inputs = step1_inputs()
    dec = P4A.decisions_p4()
    kt = key_table()
    cert = certified()
    meta = read_json(CHECK / "build" / "cells_meta.json")["cells"]
    snaps = old_finals()
    _, scells = ST.load_design(pio.STIMULI_DIR)
    maps = option_a_maps(meta)
    buffers = {a: rel(pio.CONTROLS_DIR / f"knn_{a}.npz") for a in OLD}
    d_dir, ga_dir = CHECK / "rollouts" / "D", CHECK / "rollouts" / "GA"
    print("S1-S3 tier D on this design:", flush=True)
    have_d = rollout_batch(d_dir, "check", {a: [m["cell"] for m in meta if m["owner"] == a]
                                            for a in OLD}, snaps, args.workers, buffers, maps)
    ga = tier_ga_cells(scells)
    step1_runs = {a: [dict(file=rel(pio.rollout_path(a, FINAL_TIMESTEPS)),
                           map={str(c): c for c in ga})] for a in OLD}
    print(f"S1-S3 tier G and tier A A2 ({len(ga)} of step 1's cells):", flush=True)
    have_ga = rollout_batch(ga_dir, "step1", {a: ga for a in OLD}, snaps, args.workers,
                            compare=step1_runs)
    keys = pio.load_table(CHECK / "build" / "keys.npz")
    ct = P4A.cond_tables([d_dir], list(OLD), keys, with_d_ood=True)
    gen = generalist_table(kt)
    ct_planned = {a: pio.select(t, ~np.isin(A._st(t, "condition"), EXTENSION_CONDS))
                  for a, t in ct.items()}
    # (a) on the 23 registered conditions alone, as decisions_all.json holds them (its C and
    # d_OOD dicts have no certified condition).
    ct_registered = {a: pio.select(t, np.isin(A._st(t, "condition"), D_CONDITIONS))
                     for a, t in ct.items()}
    rr = {a: P4A.classify(ct_registered[a], gen, A.F6_CONDS,
                          tier_g=P4A.tier_g_table(dec[a]["details"]["tier_g_crash"]))
          for a in OLD}
    check_a = P4A.reproduction_check(rr, dec)
    check_b = classes_check(ct_planned, gen, dec, read_json(P4A.OUT / "classes.json"),
                            option_a_sets(cert, False))
    check_c = classes_check(ct, gen, dec, read_json(P4A.OUT / "extension" / "classes.json"),
                            option_a_sets(cert, True))
    check_d = {a: have_d[a]["compare"] for a in OLD}
    d_ok = all(c["n"] > 0 and c["bitwise_equal"] == c["n"] and not c["absent"]
               for v in check_d.values() for c in v)
    dood = d_ood_vs_option_a(have_d, maps)
    dood_ok = all(v["n"] > 0 and v["max_abs_diff"] <= P4A.DOOD_TOL for v in dood.values())
    tiers = {}
    for a in OLD:
        rt = pio.load_table(ga_dir / f"{a}_runs.npz")
        ta, tg = tier_tables(rt, kt, scells)
        tiers[a] = pairs_check(ta, tg, a, FINAL_TIMESTEPS)
        tiers[a]["tier_g_crash_step1"] = bool(dec[a]["details"]["tier_g_crash"])
        tiers[a]["step1_bitwise"] = have_ga[a]["compare"]
    tiers_ok = all(v["n_differences"] == 0 and v["tier_g_crash"] == v["tier_g_crash_step1"]
                   and _json_eq(v["tier_a_completion"], v["tier_a_completion_step1"])
                   for v in tiers.values())
    build = read_json(CHECK / "build" / "build.json")
    checks = {
        "cells_vs_step1_and_option_a": bool(build["cell_check"]["ok"]),
        "a_decisions_all": all(v["ok"] for v in check_a.values()),
        "b_option_a_classes": check_b["ok"], "c_option_a_extension_classes": check_c["ok"],
        "d_runs_bitwise_option_a": bool(d_ok), "d_ood_option_a": bool(dood_ok),
        "tier_g_and_a_vs_step1": bool(tiers_ok)}
    inputs.update(option_a_inputs())
    inputs.update({rel(P4A.OUT / "classes.json"): sha256_file(P4A.OUT / "classes.json"),
                   rel(P4A.OUT / "extension" / "classes.json"):
                       sha256_file(P4A.OUT / "extension" / "classes.json"),
                   **{b: sha256_file(ROOT / b) for b in buffers.values()},
                   "check/build/build.json": sha256_file(CHECK / "build" / "build.json")})
    sha = write_json(out, {
        "status": "CHECK (Section 7, Pipeline on S1-S3): S1-S3 through this pipeline",
        "ok": all(checks.values()), "checks": checks,
        "a": check_a, "b": check_b, "c": check_c, "d": check_d, "d_ood": dood,
        "tiers": tiers,
        "note": "tier G and tier A A2 are compared through the pair columns P4 reads and the "
                "tier G flag; their states against step 1's runs are reported (step 1 ran "
                "them in chunks of other sizes)",
        "provenance": provenance("reproduce", t0, inputs)})
    for k, v in checks.items():
        print(f"  {k}: {'ok' if v else 'FAILED'}", flush=True)
    print(f"wrote {rel(out)} ({sha[:12]})", flush=True)
    if not all(checks.values()):
        fail(f"the pipeline does not reproduce S1-S3: {[k for k, v in checks.items() if not v]}")


def stage_cm_dryrun(args) -> None:
    t0 = time.time()
    require_frozen()
    refuse_after_tag("cm-dryrun")
    rep = CHECK / "reproduce.json"
    if not rep.exists() or not read_json(rep)["ok"]:
        raise SystemExit("run reproduce first (and it must pass)")
    out = CHECK / "cm_dryrun.json"
    if out.exists() and not args.force:
        raise SystemExit(f"{rel(out)} exists: the dry run is computed once")
    kt = key_table()
    cert = certified()
    comps = comparators(cert)
    _, scells = ST.load_design(pio.STIMULI_DIR)
    keys = pio.load_table(CHECK / "build" / "keys.npz")
    ct = P4A.cond_tables([CHECK / "rollouts" / "D"], list(OLD), keys, with_d_ood=True)
    gen = generalist_table(kt)
    per = {}
    for a in OLD:
        rt = pio.load_table(CHECK / "rollouts" / "GA" / f"{a}_runs.npz")
        ta, tg = tier_tables(rt, kt, scells)
        res = p4_class(ct[a], gen, B2_F6, ta, tg)
        cm = cm_reading(corner_cells(ct[a]), res["failing"], comps) if res["failing"] else None
        per[a] = dict(p4=res, cm=cm, reading=reading_of(res, cm),
                      tier_a_completion=tier_a_completion(ta))
    sha = write_json(out, {
        "status": "DRY RUN (Section 7, CM on S1-S3): exploratory and post hoc for S1-S3, whose "
                  "option A results were seen. It is not a result and decides nothing; any "
                  "change to Section 4.2 after it is a dated amendment that states what was seen",
        "f6_set": list(B2_F6), "comparators": {str(i): list(v) for i, v in comps.items()},
        "specialists": per,
        "recipe": "not computed: Section 4.4's rule is for S4-S8 (3 of 5)",
        "provenance": provenance("cm-dryrun", t0, {"check/reproduce.json": sha256_file(rep)})})
    for a, p in per.items():
        cm = p["cm"]
        print(f"  {a}: class {p['p4']['cls']} ({p['p4']['label']}"
              f"{', vacuous' if p['p4']['vacuous'] else ''}); CM "
              f"{cm['cm'] + ' ' + str(cm['counts']) if cm else 'not computed (no failing F1)'}"
              f"; reading {p['reading']['label']}", flush=True)
    print(f"wrote {rel(out)} ({sha[:12]}): a dry run, not a result", flush=True)


# ------------------------------------------------------------------------- record and tag


PRE_REL = "outputs/p4b2/pre"
BOUND = ("scripts/p4_b2.py", "scripts/p4b2_runs.py", RECORD_REL, DRAFT_REL,
         "tests/test_p4_b2.py", "scripts/round_manifest.py", *FROZEN,
         *(p for p, _ in CERT_LISTS.values()), PRE_REL)


def tracked_clean(paths) -> list[str]:
    """Paths that are not tracked or differ from HEAD (directories: any file under them),
    and files git ignores under them (outputs/ is ignored, so an untracked file in
    outputs/p4b2/pre/ would not show as untracked)."""
    out = []
    for p in paths:
        if git_run("ls-files", "--error-unmatch", p).returncode != 0:
            out.append(f"{p}: not tracked")
        elif (git("status", "--porcelain", "--untracked-files=all", "--", p)
              or byte_mismatches(tree_blobs("HEAD", (p,)) or {})):
            out.append(f"{p}: differs from HEAD")
        if p == PRE_REL:
            ignored = git("ls-files", "--others", "--ignored", "--exclude-standard", "--", p)
            if ignored:
                out.append(f"{p}: files not added: {ignored.split()[:5]}")
    return out


def record_inputs() -> dict:
    """Every input the post-tag stages read, with its sha256 (paths repo-relative)."""
    files = [ROOT / DRAFT_REL, SCRIPT, ROOT / RUNS_SCRIPT, ROOT / "tests" / "test_p4_b2.py",
             ROOT / "scripts" / "round_manifest.py",
             *(ROOT / p for p, _ in CERT_LISTS.values()), *(p for p, _ in STEP1.values()),
             pio.STIMULI_DIR / "cells.json", P4A.SWAP, OPTION_A_BUILD,
             *(GEN_REFERENCE / f"{a}_runs.npz" for a in registry.seed0_generalists()),
             BUILD / "build.json", BUILD / "bank.npz", BUILD / "cells.json",
             BUILD / "cells_meta.json", BUILD / "keys.npz", CHECK / "reproduce.json",
             CHECK / "cm_dryrun.json", measures_tables_path()]
    files += sorted(p for p in PRE.rglob("*") if p.is_file())
    return {rel(p): sha256_file(p) for p in files}


def measures_tables_path() -> Path:
    from apexzero.interp import measures

    return measures.tables_path()


def stage_record(args) -> None:
    t0 = time.time()
    require_frozen()
    refuse_after_tag("record")
    need = [PRE / "manifest.json", PRE / "buffers" / "buffers.json",
            PRE / "starts" / "starts.json", BUILD / "build.json", CHECK / "reproduce.json",
            CHECK / "cm_dryrun.json"]
    missing = [rel(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"run the pre-tag stages first: {missing}")
    if not read_json(CHECK / "reproduce.json")["ok"]:
        raise SystemExit("the S1-S3 reproduction did not pass")
    probs = tracked_clean([b for b in BOUND if b != RECORD_REL])
    if probs:
        raise SystemExit("commit these first (git add -f for outputs/p4b2/pre): "
                         + "; ".join(probs))
    man = read_manifest()
    for path, sha in new_finals(man).values():
        if sha256_file(ROOT / path) != sha:
            fail(f"{path} no longer matches the manifest")
    inputs = record_inputs()
    rec = {
        "version": 1, "tag": TAG, "draft": DRAFT_REL, "base_commit": git("rev-parse", "HEAD"),
        "train_tag": man["train_tag"], "train_commit": man["train_commit"],
        "frozen": FROZEN, "certified_lists": {k: dict(path=p, sha256=s)
                                              for k, (p, s) in CERT_LISTS.items()},
        "f6_set": list(B2_F6), "certified_corners": {n: list(c) for n, c in
                                                     CERTIFIED_CORNERS.items()},
        "braking": list(BRAKING), "streams": check_streams(),
        "networks": {a: e["final"] for a, e in man["networks"].items()},
        "missing": man["missing"],
        "design_hash": read_json(BUILD / "build.json")["design_hash"],
        "inputs_sha256": inputs,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    sha = write_json(ROOT / RECORD_REL, rec)
    print(f"wrote {RECORD_REL} ({sha[:12]}). Then, alone: git add {RECORD_REL} && git commit -m "
          f"'P4-B2 pre-registration record' && git tag -a {TAG} -m 'P4-B2 pre-registration' && "
          f"git push {REMOTE} HEAD {TAG}", flush=True)


def tag_gate() -> dict:
    """PermissionError unless the annotated tag p4-b2-prereg-v1 exists on origin, the local tag
    is the same tag object, it points at the commit that last changed the record, HEAD descends
    from it, and every bound path (the code, the draft, the record, the frozen code and lists,
    outputs/p4b2/pre/) in the working tree is the tagged one byte for byte, with no untracked
    file. Both refs are named in full, as scripts/lockbox.py's gate names them."""
    ls = git_run("ls-remote", REMOTE, f"refs/tags/{TAG}", f"refs/tags/{TAG}^{{}}")
    if ls.returncode != 0:
        raise PermissionError(f"cannot list {REMOTE}'s tags ({ls.stderr.strip()})")
    refs = {}
    for line in ls.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            refs[parts[1]] = parts[0]
    obj, peeled = refs.get(f"refs/tags/{TAG}"), refs.get(f"refs/tags/{TAG}^{{}}")
    if obj is None:
        raise PermissionError(f"{TAG} is not on {REMOTE}")
    if peeled is None or peeled == obj:
        raise PermissionError(f"{TAG} on {REMOTE} is not an annotated tag")
    local = git_run("rev-parse", "--verify", "--quiet", f"refs/tags/{TAG}")
    if local.returncode != 0 or local.stdout.strip() != obj:
        raise PermissionError(f"the local {TAG} is missing or is not {REMOTE}'s tag object")
    kind = git_run("cat-file", "-t", obj)
    if kind.returncode != 0 or kind.stdout.strip() != "tag":
        raise PermissionError(f"{TAG} ({obj}) is not a tag object")
    commit = git_run("rev-parse", "--verify", "--quiet", f"{obj}^{{commit}}")
    if commit.returncode != 0 or commit.stdout.strip() != peeled:
        raise PermissionError(f"{TAG}'s commit does not match {REMOTE}'s")
    recorded = git("log", "-1", "--format=%H", peeled, "--", RECORD_REL)
    if recorded != peeled:
        raise PermissionError(f"{TAG} points at {peeled[:12]}, not at the commit that recorded "
                              f"{RECORD_REL} ({recorded[:12] or 'none'})")
    if git_run("merge-base", "--is-ancestor", peeled, "HEAD").returncode != 0:
        raise PermissionError(f"HEAD does not descend from {TAG} ({peeled[:12]})")
    if git_run("diff", "--quiet", peeled, "--", *BOUND).returncode != 0:
        raise PermissionError(f"a bound path differs from {TAG}'s")
    # git diff trusts the index for a file marked assume-unchanged or skip-worktree: the bytes.
    tagged = tree_blobs(peeled, BOUND)
    bad = ["cannot list the tag's files"] if tagged is None else byte_mismatches(tagged)
    if bad:
        raise PermissionError(f"bound files are not {TAG}'s byte for byte: {bad[:5]}")
    # outputs/ is ignored: an extra file in outputs/p4b2/pre/ shows only with --ignored (which
    # would also list __pycache__ under src/, so it is asked of that directory alone).
    for extra, paths in (((), BOUND), (("--ignored",), (PRE_REL,))):
        untracked = git_run("ls-files", "--others", *extra, "--exclude-standard", "--", *paths)
        if untracked.returncode != 0 or untracked.stdout.strip():
            raise PermissionError(f"untracked files under the bound paths: "
                                  f"{untracked.stdout!r}")
    return {"tag": TAG, "tag_object": obj, "commit": peeled,
            "head": git("rev-parse", "HEAD")}


def record_problems(rec: dict) -> list[str]:
    """Every recorded input that no longer hashes to its record, and pins that moved."""
    out = [f"{p}: missing" for p in rec["inputs_sha256"] if not (ROOT / p).is_file()]
    out += [f"{p}: sha256 changed" for p, s in rec["inputs_sha256"].items()
            if (ROOT / p).is_file() and sha256_file(ROOT / p) != s]
    if rec["frozen"] != FROZEN or list(rec["f6_set"]) != list(B2_F6):
        out.append("the record's pins or F6 set are not this script's")
    return out


def _post_tag() -> tuple[dict, dict]:
    gate = tag_gate()
    require_frozen()
    rec = read_json(ROOT / RECORD_REL)
    probs = record_problems(rec)
    if probs:
        raise SystemExit("the record no longer holds: " + "; ".join(probs[:10]))
    return gate, rec


def stage_rollouts(args) -> None:
    t0 = time.time()
    gate, rec = _post_tag()
    meta = read_json(BUILD / "cells_meta.json")
    owners = meta["owners"]
    snaps = {a: (e["path"], e["sha256"]) for a, e in rec["networks"].items() if a in owners}
    _, scells = ST.load_design(pio.STIMULI_DIR)
    buffers = {a: rel(PRE / "buffers" / f"knn_{a}.npz") for a in owners}
    print(f"S4-S8 tier D ({owners}):", flush=True)
    rollout_batch(ROLL / "D", "main", {a: [m["cell"] for m in meta["cells"] if m["owner"] == a]
                                       for a in owners}, snaps, args.workers, buffers)
    print("S4-S8 tier G and tier A A2:", flush=True)
    rollout_batch(ROLL / "GA", "step1", {a: tier_ga_cells(scells) for a in owners}, snaps,
                  args.workers)
    sha = write_json(ROLL / "rollouts.json", {
        "status": "P4-B2 post-tag rollouts", "owners": owners,
        "provenance": provenance("rollouts", t0, {RECORD_REL: _sha_or_none(ROOT / RECORD_REL)},
                                 gate)})
    print(f"wrote {rel(ROLL / 'rollouts.json')} ({sha[:12]})", flush=True)


def committed_ever(path: Path) -> bool:
    return bool(git("log", "--all", "--format=%H", "--", rel(path)))


def rollout_problems(rec: dict, owners: list[str]) -> list[str]:
    """The tier D rollouts classify reads must be each owner's, on the recorded design, with
    d_OOD against its recorded buffer (the rollouts stage writes both into results.json)."""
    res_path = ROLL / "D" / "results.json"
    have = read_json(res_path)["agents"] if res_path.exists() else {}
    out = []
    for a in owners:
        r = have.get(a)
        if not r or r.get("error"):
            out.append(f"{a}: no tier D rollouts")
            continue
        if r.get("design_hash") != rec.get("design_hash"):
            out.append(f"{a}: tier D rollouts of design {str(r.get('design_hash'))[:12]}, not the "
                       "record's")
        buf = r.get("buffer") or {}
        if not buf.get("sha256") or buf["sha256"] != rec.get("inputs_sha256", {}).get(
                buf.get("path")):
            out.append(f"{a}: d_OOD against {buf.get('path')}, not its recorded buffer")
    return out


def stage_classify(args) -> None:
    t0 = time.time()
    gate, rec = _post_tag()
    if DECISIONS.exists() or committed_ever(DECISIONS):
        raise SystemExit(f"{rel(DECISIONS)} exists or was committed: classify runs once")
    if not (ROLL / "rollouts.json").exists():
        raise SystemExit("run the rollouts stage first")
    kt = key_table()
    cert = certified()
    comps = comparators(cert)
    man = read_manifest()
    meta = read_json(BUILD / "cells_meta.json")
    owners = meta["owners"]
    probs = rollout_problems(rec, owners)
    if probs:
        raise SystemExit("the rollouts are not the recorded design's: " + "; ".join(probs))
    _, scells = ST.load_design(pio.STIMULI_DIR)
    keys = pio.load_table(BUILD / "keys.npz")
    ct = P4A.cond_tables([ROLL / "D"], owners, keys, with_d_ood=True)
    gen = generalist_table(kt)
    per, readings, results = {}, {}, {}
    for a in NEW:
        if a in man["missing"]:
            per[a] = dict(p4=None, cm=None, missing=man["missing"][a]["reason"])
            readings[a] = reading_of(None, missing=per[a]["missing"])
            results[a] = None
            continue
        if a in meta["excluded"]:
            res = dict(cls=0, label=A.P4_LABELS[0], failing=[], vacuous=None, key_factors=[],
                       C={}, MI=None, details={"reason": meta["excluded"][a]})
            per[a] = dict(p4=res, cm=None)
            readings[a], results[a] = reading_of(res), res
            continue
        rt = pio.load_table(ROLL / "GA" / f"{a}_runs.npz")
        ta, tg = tier_tables(rt, kt, scells)
        res = p4_class(ct[a], gen, B2_F6, ta, tg)
        fam = {f"family:{f}": p4_class(ct[a], gen, (*A.F6_CONDS, *(n for n in cert
                                                                   if cert[n]["family"] == f)),
                                       ta, tg)["cls"] for f in FAMILIES}
        fam["registered"] = p4_class(ct[a], gen, A.F6_CONDS, ta, tg)["cls"]
        cm = cm_reading(corner_cells(ct[a]), res["failing"], comps) if res["failing"] else None
        per[a] = dict(p4=res, cm=cm, per_family=fam, tier_a_completion=tier_a_completion(ta),
                      tier_g_rule=("crashes on a null straight: evidence for class 1 on "
                                   "synthetic stimuli (reported; the class is unchanged)")
                      if res["details"].get("tier_g_crash") else None,
                      cm_decides=res["cls"] == 2 and bool(res["vacuous"]))
        readings[a], results[a] = reading_of(res, cm), res
    rc = recipe(readings)
    k4 = man["k4_gate"]["per_network"]
    sha = write_new(DECISIONS, {
        "status": "P4-B2 confirmatory classification (docs/prereg/p4-b2-specialists.md at "
                  f"{TAG})", "f6_set": list(B2_F6),
        "specialists": per, "readings": readings, "recipe": rc,
        "recipe_registered": registered_recipe(results), "expected": expected_directions(
            per, rc, k4), "k4_gate": man["k4_gate"],
        "provenance": provenance("classify", t0, {RECORD_REL: _sha_or_none(ROOT / RECORD_REL)},
                                 gate)})
    for a in NEW:
        print(f"  {a}: {readings[a]['label']}", flush=True)
    print(f"recipe: {rc['claimed'] or 'no reading reaches 3 of 5'}; wording: "
          f"{rc['wording']['text']}", flush=True)
    print(f"wrote {rel(DECISIONS)} ({sha[:12]}). Commit it: git add -f {rel(DECISIONS)}",
          flush=True)


# ------------------------------------------------------------------------- main


PRE_TAG = {"manifest": stage_manifest, "buffers": stage_buffers, "starts": stage_starts,
           "build": stage_build, "reproduce": stage_reproduce, "cm-dryrun": stage_cm_dryrun,
           "record": stage_record}
POST_TAG = {"rollouts": stage_rollouts, "classify": stage_classify}
STAGES = {**PRE_TAG, **POST_TAG}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=tuple(STAGES))
    ap.add_argument("--workers", type=int, default=MAX_WORKERS,
                    help=f"process pool size (at most {MAX_WORKERS})")
    ap.add_argument("--force", action="store_true", help="rewrite a pre-tag output")
    ap.add_argument("--check", action="store_true",
                    help="buffers, starts: S1-S3 against step 1, into outputs/p4b2/check/")
    ap.add_argument("--owners", choices=("new", "check"), default="new",
                    help="build: S4-S8 (default) or S1-S3's check design")
    ap.add_argument("--train-tag", default=TRAIN_TAG, help="manifest: the training round's tag")
    args = ap.parse_args(argv)
    if args.workers > MAX_WORKERS:
        raise SystemExit(f"at most {MAX_WORKERS} processes (another workflow shares the Mac)")
    os.chdir(ROOT)
    be_nice()
    STAGES[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
