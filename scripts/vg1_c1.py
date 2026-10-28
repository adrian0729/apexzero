"""VG-1's C1 through the frozen costate-prereg-v1 pipeline (docs/prereg/vg1-value-gradient.md,
Section 5.1): ENGINEERING until the tag, the primary's measurement after it.

    uv run --no-sync python scripts/vg1_c1.py worktree           # once: ../RL-racing-vg1-c1
    uv run --no-sync python scripts/vg1_c1.py verify             # every bound file; writes nothing
    uv run --no-sync python scripts/vg1_c1.py gate [--out FILE]  # the wiring gate alone
    uv run --no-sync python scripts/vg1_c1.py measure --phase pilot|pilot2|main
    uv run --no-sync python scripts/vg1_c1.py snapshots          # descriptive, after the main run

**The worktree.** Main's src/ is not the tag's (VG-0 changed agents/ppo.py and env/adapters.py),
so the frozen code runs from a git worktree next to the repo, ``../RL-racing-vg1-c1``, checked
out (detached) at ``WORKTREE_COMMIT``: the last commit on main whose src/, scripts/costate.py,
scripts/_bootstrap.py, scripts/fields.py, scripts/psychophysics.py and the costate design
document equal the tag's, and which holds the confirmatory decisions.json and tag.json (at the
tag commit itself neither is committed, and the freeze check requires both at HEAD). ``worktree``
creates it with links to this repo's .venv, data/, runs/ and the bound costate inputs
(outputs/costate/inputs, gates, nulls and post); nothing is written there. This script stays on
main: it puts the worktree's scripts/ and src/ first on sys.path, imports scripts/costate.py and
scripts/fields.py from there, and checks that apexzero is the worktree's.

**Lane A's rules, pinned** (Section 5.1). R2-R4 of docs/findings/costate-followups-plan.md (on the
branch costate-exploratory, blob ``PINNED_PLAN_BLOB`` at ``PINNED_COMMIT``) are implemented there
by scripts/costate_exploratory.py (blob ``PINNED_SOURCE_BLOB``). The functions in ``COPIED`` are
copied from that blob byte for byte (the section 'Copied'): freeze_problems, bound_hashes,
check_inputs and verify, the helpers they call, and the frozen C1's wiring helpers. A test checks
the copies against the pinned blob, and every subcommand checks first that the pinned commit
exists, that the plan's and the source's blobs there are the recorded ones, and that the copies
still equal the blob (``pinned_problems``).

**Checks before anything** (``preflight``, then ``check_inputs``): the pin; the copied
``freeze_problems`` on the worktree (the tag is the annotated object tag.json pins and an ancestor
of HEAD; src/ and the frozen scripts are the tag's, git and byte by byte; step 1's and step 2's
frozen code; the record's code hash ``d22e670c...``; decisions.json's committed sha256; the
package versions); the imports (apexzero and every module from inside the worktree are frozen
code, and nothing comes from this repo's src/); and the sha256 of inputs/world.npz,
inputs/summary.json and the decision networks' critic files. Only then are trained networks
unlocked (the copied ``unlock``). ``measure --phase main`` and ``snapshots`` also need VG-1's
own tag gate (scripts/vg1_analysis.py's ``tag_gate``: vg1-prereg-v1 on origin, HEAD descends
from it, VG-1's code unchanged).

**The wiring gate** (``wiring_gate``), before any VG-1 network: G1 (runs/dr-pilot-200M) from its
pinned checkpoint, its pullback at T recomputed (scripts/costate.py's ``critic_of`` loop over T's
keys, the trained critic only: ``pullback_arrays``) and compared with its bound
post/critic/runs_dr-pilot-200M.npz (g, pi, g_held, value and accel_sens, bit for bit); K3 after
the tag recomputed and compared with the bound .k3.json; its decision mask rebuilt against
decisions.json's sha256; and C1 recomputed from the fresh pullback (``frozen_c1``) against
decisions.json: r2c, stat, ci, the per-fold a and b of s and xi, the Spearman rho of s and xi
and the scored rows, to ``GATE_TOL`` = 1e-12. Nothing is measured unless all hold.

**Per network** (``measure_network``; the networks of the phase's manifest,
outputs/vg1/manifest-<phase>.json from scripts/vg1_analysis.py ``manifest``, each final
checkpoint loaded through its manifest pin, sha256 and timesteps checked):
- whether every parameter and normaliser value is finite;
- K3 after the tag (``gates.k3_trained`` on D's raw observations, as the critic stage) with
  ``VG1_HEADS`` = 5 expected heads: the frozen ``gates.expected_heads`` has no rule for VG-1's
  names. A heads failure with the head counts read is a wiring error and stops the run; an
  extractor or float32 failure, or a measurement that raised, is recorded (the network is a
  failed network in the analysis);
- the main V's pullback at T and C1 from it (``frozen_c1``: ``analysis.c1`` on ``c1_frame``'s
  calibration, rows, decision mask, bounds and bootstrap, without nulls): R2_c (C1's stat, the
  mean over s and xi), per-component R2_c and Spearman rho for s and xi and their mean (S2), the
  CIs, the scored rows and the per-fold calibration. A computation that raises is recorded as
  an error (a non-finite R2_c).

``snapshots`` (Section 5.3, descriptive): R2_c at every checkpoint of seeds 30-33 in each arm
and C1-dev's rho over log timesteps (``analysis.c1_dev``).

**Outputs** under outputs/vg1/c1/ only (never outputs/costate or outputs/sealed), write-once:
gate.json (``gate --out`` elsewhere), pilot.json, pilot2.json (the second pilot, Amendment 1:
the same as pilot, no tag gate), main.json, snapshots.json. Each carries a
provenance stamp: this repo's HEAD and changed paths, the worktree's HEAD, the costate tag, the
code hash, the pin, the gate's record, every input's sha256, the arguments (repo-relative) and the
environment. Non-finite numbers are written as null or "inf" / "-inf".

**Compute**: one process, XLA on --threads threads (default 2, as scripts/costate.py) with
eigen's own threading off, reniced to --nice (default 10). Another workflow shares the Mac.
Nothing here reads R_TEST or anything under outputs/sealed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # this repository (main)
WORKTREE = ROOT.parent / "RL-racing-vg1-c1"
WORKTREE_COMMIT = "348f392a5b0b4143017c16a3d8736ea53c11fe55"
WORKTREE_LINKS = (".venv", "data", "runs", "outputs/costate/inputs", "outputs/costate/gates",
                  "outputs/costate/nulls", "outputs/costate/post")

PINNED_COMMIT = "2056a3d5148807a76d734aa5b0bf32601cd02e2f"   # costate-exploratory, 26 Oct 2026
PINNED_SOURCE = "scripts/costate_exploratory.py"
PINNED_SOURCE_BLOB = "30befd8c35df1c317afa03d7dfe7fe835d4d3d07"
PINNED_PLAN = "docs/findings/costate-followups-plan.md"
PINNED_PLAN_BLOB = "e05eb306432f0b9b41a5a74bea8da6a1f0429ff3"
COPIED = ("_git", "_sha256_bytes", "sha256_file", "_changed", "_json_at", "freeze_problems",
          "bound_hashes", "decision_files", "group_inputs", "unbound_problems", "mask_sha256",
          "mask_problems", "load_context", "check_inputs", "unlock", "verify", "load_world",
          "c1_reference", "frozen_c1", "c1_wiring")

# The pinned source's constants, which its copied functions read (there from scripts/costate.py;
# ``enter_worktree`` checks them against the worktree's costate.py).
TAG = "costate-prereg-v1"
OUT_REL = "outputs/costate"
DECISIONS_REL = f"{OUT_REL}/analyse/decisions.json"
DECISIONS_SHA256 = "8119f91f7c1a24186fad011a83ef931e441626d7493355272b762e8aa97b0ee4"
RECORD_CODE_HASH = "d22e670cd059063af38c1af74c039a0ae11933b50443ab40f14256bb49dbfff3"
FROZEN_PATHS = ("src", "scripts/costate.py", "scripts/_bootstrap.py")
ITEM_GROUPS = ("step1", "step2", "keys:n", "keys:a2", "checkpoints", "snapshots")
ALLOWED_SCRIPTS = ("costate.py", "fields.py", "_bootstrap.py")

G1 = "runs/dr-pilot-200M"
GATE_TOL = 1e-12
GATE_FIELDS = ("g", "pi", "g_held", "value", "accel_sens")
VG1_HEADS = 5
SX = ("s", "xi")
K_SX = (0, 2)                          # their columns in K_P
OUT_DIR = "outputs/vg1/c1"
MANIFEST = "outputs/vg1/manifest-{phase}.json"
SNAPSHOT_SEEDS = (30, 31, 32, 33)
DEFAULT_THREADS = 2
DEFAULT_NICE = 10
LABEL = {"pilot": "exploratory (VG-1's pilot, before the tag)",
         "pilot2": "exploratory (VG-1's second pilot, before the tag)",
         "main": "confirmatory input (VG-1, at vg1-prereg-v1)",
         "snapshots": "descriptive (VG-1, Section 5.3)", "gate": "engineering (the wiring gate)"}

CS = None   # scripts/costate.py, the worktree's (enter_worktree)
F = None    # scripts/fields.py, the worktree's


# ------------------------------------------------------------------------- the pin


def _git_main(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def function_sources(text: str, names=COPIED) -> dict[str, str | None]:
    """{name: the top-level function's source segment, or None} of a module's text."""
    tree = ast.parse(text)
    by = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    return {k: (ast.get_source_segment(text, by[k]) if k in by else None) for k in names}


def pinned_text() -> str | None:
    r = _git_main("show", f"{PINNED_COMMIT}:{PINNED_SOURCE}")
    return r.stdout if r.returncode == 0 else None


def copy_problems(pinned: str, here: str) -> list[str]:
    """Each copied function whose source here is not the pinned blob's, byte for byte."""
    a, b = function_sources(pinned), function_sources(here)
    out = []
    for k in COPIED:
        if a[k] is None:
            out.append(f"{k}: not in the pinned source")
        elif b[k] is None:
            out.append(f"{k}: not copied here")
        elif a[k] != b[k]:
            out.append(f"{k}: differs from the pinned source")
    return out


def pinned_problems() -> list[str]:
    """The pin (module docstring): the commit, the plan's and the source's blobs there, and the
    copies against the source."""
    if _git_main("cat-file", "-e", f"{PINNED_COMMIT}^{{commit}}").returncode:
        return [f"the pinned commit {PINNED_COMMIT[:12]} is not in this repository"]
    out = []
    for rel, want in ((PINNED_SOURCE, PINNED_SOURCE_BLOB), (PINNED_PLAN, PINNED_PLAN_BLOB)):
        got = _git_main("rev-parse", f"{PINNED_COMMIT}:{rel}").stdout.strip()
        if got != want:
            out.append(f"{rel} at {PINNED_COMMIT[:12]} is blob {got[:12] or 'missing'}, not "
                       f"{want[:12]}")
    text = pinned_text()
    if text is None:
        return out + [f"cannot read {PINNED_SOURCE} at {PINNED_COMMIT[:12]}"]
    return out + copy_problems(text, Path(__file__).read_text())


# ------------------------------------------------------------------------- the worktree


def make_worktree(path: Path = WORKTREE, commit: str = WORKTREE_COMMIT) -> Path:
    """``git worktree add --detach <path> <commit>`` and the links (module docstring)."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path.name} exists already: nothing done")
    r = _git_main("worktree", "add", "--detach", str(path), commit)
    if r.returncode:
        raise SystemExit(f"git worktree add failed: {r.stderr.strip()}")
    for rel in WORKTREE_LINKS:
        link, target = path / rel, ROOT / rel
        if not target.exists():
            raise SystemExit(f"{rel} is missing in this repository: cannot link it")
        if link.exists() or link.is_symlink():
            raise SystemExit(f"{rel} exists in the worktree: the commit tracks it")
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, link.parent))
    print(f"worktree {path.name} at {commit[:12]}, links: {', '.join(WORKTREE_LINKS)}")
    return path


def enter_worktree(path: Path = WORKTREE) -> Path:
    """The worktree's scripts/ and src/ first on sys.path (this repo's taken off), then its
    scripts/costate.py and scripts/fields.py imported as ``CS`` and ``F``."""
    global CS, F
    wt = Path(path).resolve()
    if not (wt / "scripts" / "costate.py").is_file():
        raise SystemExit(f"no worktree at {Path(path).name}: run 'vg1_c1.py worktree' first")
    loaded = sorted(m for m in sys.modules if m == "apexzero" or m.startswith("apexzero."))
    if loaded or "costate" in sys.modules or "fields" in sys.modules:
        raise SystemExit(f"refusing: modules already imported from elsewhere: {loaded[:4]}")
    mine = {(ROOT / "scripts").resolve(), (ROOT / "src").resolve()}
    sys.path[:] = [p for p in sys.path if not p or Path(p).resolve() not in mine]
    sys.path[:0] = [str(wt / "scripts"), str(wt / "src")]
    import costate as cs_mod
    import fields as f_mod

    CS, F = cs_mod, f_mod
    want = {"TAG": TAG, "OUT_REL": OUT_REL, "DECISIONS_REL": DECISIONS_REL,
            "CODE_PATHS": FROZEN_PATHS}
    bad = [k for k, v in want.items() if getattr(CS, k) != v]
    if bad or Path(CS.__file__).resolve().parent != wt / "scripts":
        raise SystemExit(f"the worktree's scripts/costate.py is not the pinned one's ({bad})")
    return wt


def import_problems(root: Path) -> list[str]:
    """apexzero from the worktree's src/; every module loaded from inside the worktree frozen
    code (scripts/costate.py, fields.py, _bootstrap.py, src/apexzero); none from this repo's
    src/ (this script and scripts/vg1_analysis.py are this repo's, outside the worktree)."""
    import apexzero

    root = Path(root).resolve()
    want = (root / "src" / "apexzero").resolve()
    got = Path(apexzero.__file__).resolve().parent
    out = [] if got == want else ["apexzero is not imported from the worktree's src/"]
    mine = (ROOT / "src").resolve()
    allowed = {(root / "scripts" / n).resolve() for n in ALLOWED_SCRIPTS}
    for name, mod in sorted(dict(sys.modules).items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        fp = Path(f).resolve()
        if mine in fp.parents:
            out.append(f"{name} imported from this repository's src/, not the worktree's")
        elif root in fp.parents and not (want in fp.parents or fp in allowed):
            out.append(f"{name} imported from {fp.relative_to(root)}, outside the frozen code")
    return out


def preflight(root: Path) -> list[str]:
    """The pin, the freeze (the copied ``freeze_problems``) and the imports."""
    return pinned_problems() + freeze_problems(root) + import_problems(root)


class Item:
    """What a subcommand reads (``check_inputs``' groups)."""

    def __init__(self, id_: str, reads: tuple[str, ...]):
        self.id, self.reads = id_, reads


ITEMS = {"gate": Item("vg1-gate", ("world", "critic:decision")),
         "measure": Item("vg1-c1", ("world", "critic:decision", "checkpoints")),
         "snapshots": Item("vg1-c1-snapshots", ("world", "critic:decision", "snapshots"))}


class Run:
    """One subcommand's run on the worktree: the record, decisions.json, the bound sha256 map
    and the inputs read (sha256 checked and recorded), as the pinned source's ``Run`` reads."""

    def __init__(self, item: Item, root: Path):
        self.item, self.root = item, Path(root)
        self.costate = self.root / OUT_REL
        self.rec, self.decisions, self.bound = load_context(self.root)
        self.inputs: dict[str, str] = {}

    def read(self, rel: str, kind: str = "bound") -> Path:
        path = self.costate / rel
        if kind != "bound":
            raise ValueError(f"{rel}: only bound files are read here")
        if not path.exists():
            raise SystemExit(f"{self.item.id}: its input {OUT_REL}/{rel} is missing")
        want = (self.bound or {}).get(rel)
        if want is None:
            raise SystemExit(f"{rel} is not bound by decisions.json or the record")
        got = sha256_file(path)
        if got != want:
            raise SystemExit(f"{rel} is not the file decisions.json or the record binds "
                             f"({got[:12]} against {want[:12]})")
        self.inputs[f"{OUT_REL}/{rel}"] = got
        return path

    def note_input(self, rel: str, sha256: str) -> None:
        self.inputs[rel] = sha256


def start(item: str, worktree: Path) -> Run:
    """The checks (module docstring), then a Run whose inputs are checked."""
    wt = enter_worktree(worktree)
    bad = preflight(wt)
    if bad:
        raise SystemExit(f"refusing {item}: " + "; ".join(bad))
    run = Run(ITEMS[item], wt)
    done = check_inputs(run)
    print(f"{item}: the pin, the freeze and the imports pass; {len(done)} bound inputs checked",
          flush=True)
    return run


# ------------------------------------------------------------------------- copied
#
# Copied byte for byte from scripts/costate_exploratory.py at PINNED_COMMIT (blob
# PINNED_SOURCE_BLOB); tests/test_vg1.py and pinned_problems check them. Do not edit.

_HASHED: dict[tuple, str] = {}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    """A file's sha256, cached per (resolved path, size, mtime) within this process."""
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _HASHED:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        _HASHED[key] = h.hexdigest()
    return _HASHED[key]


def _changed(root: Path, rev: str, paths: Sequence[str]) -> list[str]:
    return F.changed_since(rev, tuple(paths), root)


def _json_at(root: Path, rev: str, rel: str) -> dict | None:
    r = _git(root, "show", f"{rev}:{rel}")
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


def freeze_problems(root: Path = ROOT, record_code_hash: str = RECORD_CODE_HASH,
                    decisions_sha256: str = DECISIONS_SHA256) -> list[str]:
    """Why no item may run on this tree (module docstring 'the freeze')."""
    from apexzero.costate import prereg as PR

    root = Path(root)
    pin_p = CS.pin_path_of(root)
    if not pin_p.exists():
        return [f"no {pin_p.relative_to(root)}: the tag was never pinned"]
    pin = json.loads(pin_p.read_text())
    t = F._tag_object(TAG, root)
    if t is None:
        return [f"the tag {TAG} does not exist"]
    out = []
    commit = F._tag_commit(TAG, root)
    if t[1] != "tag":
        out.append(f"{TAG} is not an annotated tag")
    if t[0] != pin.get("object") or commit != pin.get("commit"):
        out.append(f"{TAG} is not the object tag.json pins")
    if _git(root, "merge-base", "--is-ancestor", commit or TAG, "HEAD").returncode != 0:
        out.append(f"{TAG} is not an ancestor of HEAD")
    if _json_at(root, "HEAD", f"{OUT_REL}/tag.json") != pin:
        out.append("tag.json is not committed at HEAD as it is on disk")
    ch = _changed(root, TAG, FROZEN_PATHS + (CS.DESIGN_DOC,))
    if ch:
        out.append(f"frozen code or the design document differs from {TAG}: {ch[:6]}")
    try:
        hidden = sorted(set(F.byte_changes(TAG, FROZEN_PATHS, root)) - {c.split(" ")[0]
                                                                        for c in ch})
        if hidden:
            out.append(f"frozen code differs from {TAG} byte by byte: {hidden[:6]}")
    except RuntimeError as e:
        out.append(f"git failed reading {TAG}'s code: {e}")
    out += [str(p) for p in CS.frozen_steps_problems(root)]
    rec = CS.tagged_record(TAG, root)
    if rec is None:
        return out + [f"no record {CS.PREREG_REL} at {TAG}"]
    out += [str(p) for p in CS.step_tags_problems(rec, root)]
    if rec.get("code_hash") != record_code_hash:
        out.append(f"the record's code hash is {str(rec.get('code_hash'))[:12]}, not "
                   f"{record_code_hash[:12]}")
    now = PR.code_hash(root)
    if now != rec.get("code_hash"):
        out.append(f"the tree's costate code hash {now[:12]} is not the record's")
    design = root / CS.DESIGN_DOC
    if (not design.exists() or sha256_file(design) != rec.get("design_doc_sha256")
            or CS.tagged_doc_sha256(TAG, root) != rec.get("design_doc_sha256")):
        out.append("the design document is not the one the record registers")
    rp = root / CS.PREREG_REL
    tag_rec = _git(root, "show", f"{TAG}:{CS.PREREG_REL}")
    if not rp.exists() or tag_rec.returncode != 0 or rp.read_bytes() != tag_rec.stdout:
        out.append(f"{CS.PREREG_REL} on disk is not the tag's")
    dp = root / DECISIONS_REL
    if not dp.exists() or sha256_file(dp) != decisions_sha256:
        out.append(f"{DECISIONS_REL} is missing or not the committed confirmatory decisions")
    else:
        at_head = _git(root, "show", f"HEAD:{DECISIONS_REL}")
        if at_head.returncode != 0 or _sha256_bytes(at_head.stdout) != decisions_sha256:
            out.append(f"{DECISIONS_REL} is not committed at HEAD as those bytes")
        pv = (json.loads(dp.read_text()).get("provenance") or {})
        if (pv.get("tag") or {}).get("object") != pin.get("object"):
            out.append("decisions.json was not written under the pinned tag")
    env = F.env_problems(rec.get("environment"), F.environment(root))
    if env:
        out.append(f"package versions differ from the record's: {env[:6]}")
    return out


def bound_hashes(rec: dict, decisions: dict) -> dict[str, str]:
    """{path under outputs/costate: sha256} of every bound file: the record's ``outputs`` and
    decisions.json's ``inputs`` (the 9 null arrays are in both, and must agree)."""
    out = {k: v["sha256"] for k, v in (rec.get("outputs") or {}).items()}
    for k, v in (decisions.get("inputs") or {}).items():
        if k in out and out[k] != v:
            raise SystemExit(f"{k}: the record and decisions.json bind different sha256")
        out[k] = v
    return out


def decision_files(bound: dict[str, str]) -> list[str]:
    """The 9 decision networks' file stems (their own arrays are bound: post/own/<f>.npz)."""
    return sorted(Path(k).stem for k in bound if k.startswith("post/own/") and k.endswith(".npz"))


def group_inputs(group: str, bound: dict[str, str], out_root: Path) -> list[tuple[str, str]]:
    """[(path under outputs/costate, 'bound' or 'stamped')] of a checked input group."""
    dec = decision_files(bound)
    if group == "world":
        rels = ["inputs/world.npz", "inputs/summary.json"]
    elif group == "critic:decision":
        rels = [f"post/critic/{f}{x}" for f in dec for x in (".npz", ".k3.json")]
    elif group == "critic:all":
        rels = sorted(k for k in bound if k.startswith("post/critic/"))
    elif group == "activations:decision":
        rels = [f"post/activations/{f}.npz" for f in dec]
    elif group == "activations:other":
        d = Path(out_root) / "post" / "activations"
        have = sorted(p.name for p in d.glob("*.npz")) if d.exists() else []
        return [(f"post/activations/{n}", "stamped") for n in have
                if f"post/activations/{n}" not in bound]
    elif group == "own":
        rels = [f"post/own/{f}{x}" for f in dec for x in (".npz", ".k3b.json")]
    elif group == "probes":
        rels = [f"post/probes/{f}.json" for f in dec]
    elif group == "nulls":
        rels = [f"nulls/{f}.npz" for f in dec] + ["nulls/summary.json"]
    elif group.startswith("gates:"):
        rels = [f"gates/{group.split(':', 1)[1]}.json"]
    else:
        raise KeyError(f"{group} is not a checked input group")
    return [(r, "bound") for r in rels]


def unbound_problems(path: Path, rec: dict, tag: dict | None) -> list[str]:
    """Why an unbound post-tag file may not be read (module docstring): the confirmatory
    analysis's own stamp rule (scripts/fields.py's ``input_stamp_problems``)."""
    return F.input_stamp_problems([Path(path)], rec.get("code_hash"), None, tag,
                                  rec.get("environment"))


def mask_sha256(valid) -> str:
    """scripts/costate.py's ``decision_mask_sha256`` of a decision mask [n, 4]."""
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(np.asarray(valid, bool)).tobytes()).hexdigest()


def mask_problems(valid, run_name: str, decisions: dict) -> list[str]:
    """A rebuilt decision mask against decisions.json's ``decision_mask_sha256`` (where the
    network has one: the 9 decision networks)."""
    want = ((decisions.get("networks") or {}).get(run_name) or {}).get("decision_mask_sha256")
    if want is None:
        return []
    got = mask_sha256(valid)
    if got == want:
        return []
    msg = (f"{run_name}: the rebuilt decision mask ({got[:12]}) is not the confirmatory one "
           f"({want[:12]})")
    return [msg]


def load_context(root: Path = ROOT) -> tuple[dict, dict, dict[str, str]]:
    """(the record at the tag, decisions.json, the bound sha256 map)."""
    rec = CS.tagged_record(TAG, root)
    if rec is None:
        raise SystemExit(f"no record at {TAG}")
    decisions = json.loads((Path(root) / DECISIONS_REL).read_text())
    return rec, decisions, bound_hashes(rec, decisions)


def check_inputs(run: Run) -> list[str]:
    """Read (check and record) every file of the item's checked input groups."""
    done = []
    for g in run.item.reads:
        if g in ITEM_GROUPS:
            continue
        for rel, kind in group_inputs(g, run.bound or {}, run.costate):
            run.read(rel, kind)
            done.append(rel)
    return done


def unlock() -> None:
    """Trained networks allowed in this process (plan R4), as scripts/costate.py's
    ``_post_tag`` does, after the checks."""
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    CR.unlock_trained()
    PB.unlock_trained()
    NW.unlock_trained()


def verify(root: Path = ROOT) -> list[str]:
    """The plan, the freeze, the imports, every bound file's sha256 and every unbound
    post-tag activation file's stamp; writes nothing."""
    bad = preflight(root)
    rec, _decisions, bound = load_context(root)
    costate = Path(root) / OUT_REL
    for rel, want in sorted(bound.items()):
        p = costate / rel
        if not p.exists():
            bad.append(f"{rel} is missing")
        elif sha256_file(p) != want:
            bad.append(f"{rel} is not the bound file")
    tag = F.tag_identity(TAG, root)
    for rel, _ in group_inputs("activations:other", bound, costate):
        bad += [f"{rel}: {b}" for b in unbound_problems(costate / rel, rec, tag)]
    print(f"verify: {len(bound)} bound files and the unbound activations checked; "
          f"{len(bad)} problems", flush=True)
    return bad


def load_world(run: Run):
    """T's world (``scripts/costate.py``'s ``_load_world_frozen``: the bound world.npz, T's
    (key, t) list, geometries and folds against the record), read through ``run``."""
    from types import SimpleNamespace

    CS._x64()
    run.read("inputs/world.npz")
    return CS._load_world_frozen(SimpleNamespace(out=str(run.costate)), run.rec)


def c1_reference(decisions: dict, name: str) -> dict:
    """decisions.json's record of a network (``networks`` or ``descriptive_networks``)."""
    for part in ("networks", "descriptive_networks"):
        rec = (decisions.get(part) or {}).get(name)
        if rec is not None and "C1" in rec:
            return rec
    raise SystemExit(f"decisions.json has no C1 record of {name}")


def frozen_c1(fr, g8, counts, rows=None, w=None):
    """C1's statistic without nulls, slope CIs or leave-one-geometry-out sets (``analysis.c1``
    on ``c1_frame``'s calibration, rows, mask, bounds and bootstrap) of a pullback g8 [n, 8];
    ``rows`` and ``w`` replace the interior and its weights (a stratum: its fit rows too)."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import targets as TG

    rows = fr.inner if rows is None else rows
    w = fr.w_in if w is None else w
    return AN.c1(g8[:, fr.idx], fr.lam, fr.fo.outer, w, fr.cl, fr.G, fr.tab["direction"], rows,
                 None, fr.valid, TG.K_P, counts, slope_boot=False, bounds=fr.bnd)


def c1_wiring(c1, want: dict, ks=K_SX) -> dict:
    """{field: max |difference|} of a recomputed C1 against its decisions.json record (r2c of
    s and xi, stat, ci, the per-fold a and b of s and xi; plan Section 3 'Reuse')."""
    import numpy as np

    ks = list(ks)
    got = {"r2c": [c1.r2c[k] for k in ks], "stat": [c1.stat], "ci": list(c1.ci),
           "a": c1.calibrated.a[:, ks].ravel(), "b": c1.calibrated.b[:, ks].ravel()}
    ref = {"r2c": [want["r2c"][k] for k in ks], "stat": [want["stat"]], "ci": want["ci"],
           "a": np.asarray(want["calibration"]["a"], float)[:, ks].ravel(),
           "b": np.asarray(want["calibration"]["b"], float)[:, ks].ravel()}
    return {k: float(np.max(np.abs(np.asarray(got[k], float) - np.asarray(ref[k], float))))
            for k in got}


# ------------------------------------------------------------------------- one network


def pullback_arrays(agent, world) -> dict:
    """scripts/costate.py's ``critic_of`` loop for the trained critic alone (no random-init
    critics, no width derivative): g, pi, g_held, value and accel_sens at T, every head."""
    import numpy as np

    from apexzero.costate import critic as CR

    net = CR.trained_critic(agent, getattr(agent, "checkpoint", ""))
    n = world.n
    E = net.n_heads
    g = np.full((n, 1 + E, 8), np.nan)
    pi = np.full((n, 1 + E, 2), np.nan)
    gh = np.full((n, 1 + E, 8), np.nan)
    y = np.full((n, 1 + E), np.nan)
    acc = np.full((n, 1 + E, 2), np.nan)
    for k in world.keys:
        pb = CR.pullback(net, None, k.inp, k.oj)
        g[k.rows], pi[k.rows], gh[k.rows] = pb.g, pb.pi, pb.g_held
        y[k.rows], acc[k.rows] = pb.value, pb.accel_sens
    return {"g": g, "pi": pi, "g_held": gh, "value": y, "accel_sens": acc,
            "sigma": float(net.sigma)}


def k3_vg1(agent, world, heads: int = VG1_HEADS) -> dict:
    """scripts/costate.py's ``k3_trained`` (D's raw observations) with ``heads`` expected."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT

    ob = np.full((world.n, CR.OBS_DIM), np.nan)
    for k in world.keys:
        ob[k.rows] = k.oj.ob
    rows = CS.d_rows(world)
    have = rows[np.isfinite(ob[rows]).all(axis=1)]
    out = GT.k3_trained(agent, ob[have], heads)
    out.update(rows="D: T's t = 0 rows", n_d=len(rows),
               n_d_without_observation=int(len(rows) - len(have)))
    return out


def params_finite(agent) -> bool:
    """Every floating value of the snapshot (actor, critic, both normalisers) is finite."""
    import jax
    import numpy as np

    leaves = [np.asarray(x) for x in jax.tree.leaves(agent.snapshot)]
    return all(np.isfinite(x).all() for x in leaves if np.issubdtype(x.dtype, np.floating))


def c1_summary(c1) -> dict:
    """VG-1's numbers from a frozen C1 (``frozen_c1``): s and xi only (n and vx have no
    decision row on T)."""
    import numpy as np

    k = list(K_SX)
    sp = np.asarray(c1.spearman, float)[k]
    return {"stat": c1.stat, "ci": list(c1.ci), "decision_band": c1.decision,
            "reason": c1.reason,
            "r2c": dict(zip(SX, np.asarray(c1.r2c, float)[k].tolist(), strict=True)),
            "spearman": dict(zip(SX, sp.tolist(), strict=True)),
            "spearman_mean": float(np.mean(sp)),
            "spearman_ci": dict(zip(SX, np.asarray(c1.spearman_ci, float)[k].tolist(),
                                    strict=True)),
            "n_rows": dict(zip(SX, np.asarray(c1.n_rows)[k].tolist(), strict=True)),
            "n_uncalibrated": (None if c1.n_uncalibrated is None else
                               dict(zip(SX, np.asarray(c1.n_uncalibrated)[k].tolist(),
                                        strict=True))),
            "calibration": {"a": c1.calibrated.a[:, k].tolist(),
                            "b": c1.calibrated.b[:, k].tolist(),
                            "clamped": np.asarray(c1.calibrated.clamped)[:, k].tolist()}}


def network_c1(run: Run, world, counts, name: str, agent) -> tuple[dict, object]:
    """(the summary, the C1) of a loaded network's main V at T; an error is recorded."""
    from apexzero.costate import inputs as IN
    from apexzero.fields import network as NW

    try:
        arrays = pullback_arrays(agent, world)
        flags = IN.flags(world, NW.upcast32(agent.obs_rms))
        fr = CS.c1_frame(world, run.rec["frozen"], flags)
        c1 = frozen_c1(fr, arrays["g"][:, 0], counts)
    except Exception as e:  # noqa: BLE001  (recorded: the network's R2_c is not computable)
        return {"stat": None, "error": f"{type(e).__name__}: {e}"}, None
    out = c1_summary(c1)
    out["decision_rows"] = {c: int((fr.inner & fr.valid[:, k]).sum())
                            for c, k in zip(SX, K_SX, strict=True)}
    out["decision_mask_sha256"] = mask_sha256(fr.valid)
    return out, c1


def measure_network(run: Run, world, counts, ent: dict) -> dict:
    """One VG-1 network (module docstring 'Per network')."""
    t0 = time.time()
    name = f"runs/{ent['name']}"
    pin = dict(ent["final"])
    agent = CS.load_agent(name, pin, run.root)
    run.note_input(pin["path"], pin["sha256"])
    finite = params_finite(agent)
    k3 = k3_vg1(agent, world)
    failed = list(k3["decision"]["failed"])
    measured = all("error" not in (k3.get(k) or {}) for k in ("critic", "extractor"))
    if "heads" in failed and measured:
        # the head counts were read, and are not 5: the manifest holds every config at 5
        raise SystemExit(f"{ent['name']}: K3's heads check failed ({k3['decision']['reason']}): "
                         "a wiring error, which stops the analysis for an amendment "
                         "(Section 5.1)")
    c1, _ = network_c1(run, world, counts, name, agent)
    rec = {"name": ent["name"], "network": name, "arm": ent.get("arm"), "seed": ent.get("seed"),
           "beta": ent.get("beta"), "checkpoint": pin, "params_finite": finite,
           "k3": {"pass": k3["decision"]["pass"], "failed": failed,
                  "reason": k3["decision"]["reason"], "checks": k3["checks"],
                  "expected_heads": VG1_HEADS, "critic": k3.get("critic"),
                  "extractor": k3.get("extractor"), "float32": k3.get("float32")},
           "c1": c1, "seconds": time.time() - t0}
    st = c1.get("stat")
    print(f"{ent['name']}: R2_c {st if st is None else f'{st:.4f}'}, K3 "
          f"{'pass' if k3['decision']['pass'] else failed}, finite {finite} "
          f"({rec['seconds']:.0f} s)", flush=True)
    return rec


# ------------------------------------------------------------------------- the wiring gate


def _max_abs(a, b) -> float | None:
    import numpy as np

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return None
    both = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[both] - b[both]), initial=0.0))


def wiring_gate(run: Run, world, counts) -> dict:
    """G1's pullback, K3 and C1 recomputed against its bound files and decisions.json (module
    docstring 'The wiring gate'); raises SystemExit unless every check holds."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.fields import network as NW

    t0 = time.time()
    pin = run.rec["networks"][G1]
    agent = CS.load_agent(G1, pin, run.root)
    run.note_input(pin["path"], pin["sha256"])
    f = PR.net_file(G1)
    with np.load(run.read(f"post/critic/{f}.npz")) as d:
        stored = {k: d[k].copy() for k in GATE_FIELDS}
        stored_pass = bool(d["k3_trained_pass"])
    stored_k3 = json.loads(run.read(f"post/critic/{f}.k3.json").read_text())
    fresh = pullback_arrays(agent, world)
    pull = {k: {"equal": bool(fresh[k].dtype == stored[k].dtype
                              and np.array_equal(fresh[k], stored[k], equal_nan=True)),
                "max_abs_diff": _max_abs(fresh[k], stored[k])} for k in GATE_FIELDS}
    k3 = CS.k3_trained(agent, world, G1)
    k3_same = {"checks": k3["checks"] == stored_k3["checks"],
               "decision": k3["decision"] == stored_k3["decision"],
               "critic": k3["critic"] == stored_k3["critic"],
               "extractor": k3["extractor"] == stored_k3["extractor"],
               "float32": k3["float32"] == stored_k3["float32"],
               "pass": k3["decision"]["pass"] is True and stored_pass}
    flags = IN.flags(world, NW.upcast32(agent.obs_rms))
    fr = CS.c1_frame(world, run.rec["frozen"], flags)
    mask = mask_problems(fr.valid, G1, run.decisions)
    c1 = frozen_c1(fr, fresh["g"][:, 0], counts)
    want = c1_reference(run.decisions, G1)["C1"]
    diff = c1_wiring(c1, want)
    diff["spearman"] = float(np.max(np.abs(np.asarray(c1.spearman, float)[list(K_SX)]
                                           - np.asarray(want["spearman"], float)[list(K_SX)])))
    rows_same = [int(c1.n_rows[k]) for k in K_SX] == [int(want["n_rows"][k]) for k in K_SX]
    checks = {"pullback_bit_for_bit": all(v["equal"] for v in pull.values()),
              "k3_reproduced": all(k3_same.values()), "decision_mask": not mask,
              "c1_within_tol": all(v <= GATE_TOL for v in diff.values()),
              "c1_rows": rows_same}
    rec = {"network": G1, "checkpoint": pin, "pullback": pull, "k3": k3_same,
           "decision_mask_problems": mask, "c1_max_abs_diff": diff, "tolerance": GATE_TOL,
           "c1": c1_summary(c1), "c1_decisions_json": {"stat": want["stat"], "r2c": want["r2c"],
                                                       "n_rows": want["n_rows"]},
           "checks": checks, "pass": all(checks.values()), "seconds": time.time() - t0}
    worst = max(diff.values())
    same = "bit for bit" if checks["pullback_bit_for_bit"] else "DIFFERS"
    print(f"wiring gate (G1): pullback {same}"
          f", K3 {'reproduced' if checks['k3_reproduced'] else 'DIFFERS'}, C1 max |diff| "
          f"{worst:.3g} (tolerance {GATE_TOL:g}), stat {c1.stat:.15g} against "
          f"{want['stat']:.15g}: {'PASS' if rec['pass'] else 'FAIL'}", flush=True)
    if not rec["pass"]:
        raise SystemExit(f"the wiring gate failed: {[k for k, v in checks.items() if not v]}; "
                         "nothing is measured")
    return rec


# ------------------------------------------------------------------------- outputs


def clean(x):
    """JSON-safe: NaN as null, infinities as "inf" / "-inf", NumPy values as Python ones."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [clean(v) for v in x]
    if hasattr(x, "tolist") and not isinstance(x, str):
        return clean(x.tolist())
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def refuse_out(path: Path) -> Path:
    """No output under outputs/costate or outputs/sealed (links followed)."""
    p = Path(path).resolve()
    for d in (ROOT / "outputs" / "costate", ROOT / "outputs" / "sealed", WORKTREE / "outputs"):
        d = d.resolve()
        if p == d or d in p.parents:
            raise SystemExit(f"refusing to write {path}: never under {d.name}")
    if any(q.name == "sealed" and q.parent.name == "outputs" for q in (p, *p.parents)):
        raise SystemExit(f"refusing {path}: nothing under outputs/sealed is read or written")
    return Path(path)


def write_once(path: Path, obj: dict) -> Path:
    path = refuse_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(clean(obj), indent=1, allow_nan=False) + "\n").encode()
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {path}: outputs are write-once") from None
    print(f"wrote {_rel(path)}", flush=True)
    return path


def _rel(path: Path) -> str:
    p = Path(path).resolve()
    for base in (ROOT, WORKTREE):
        try:
            return p.relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    return f"<outside>/{p.name}"


def safe_argv(argv: list[str]) -> list[str]:
    out = ["scripts/vg1_c1.py"]
    for a in argv[1:]:
        pre, val = ("", a) if not (a.startswith("--") and "=" in a) else a.split("=", 1)
        val = _rel(Path(val)) if os.path.isabs(val) else val
        out.append(f"{pre}={val}" if pre else val)
    return out


def stamp(run: Run, label: str, start_t: float, extra: dict | None = None) -> dict:
    """The provenance stamp (module docstring 'Outputs')."""
    from apexzero.costate import prereg as PR

    head = _git_main("rev-parse", "HEAD").stdout.strip()
    changed = [ln for ln in _git_main("status", "--porcelain", "--untracked-files=no")
               .stdout.splitlines() if ln.strip()]
    wt_head = _git(run.root, "rev-parse", "HEAD").stdout.decode().strip()
    return {"label": label, "script": "scripts/vg1_c1.py",
            "script_sha256": sha256_file(Path(__file__)), "head": head, "changed": changed,
            "worktree_head": wt_head, "worktree_commit": WORKTREE_COMMIT,
            "costate_tag": F.tag_identity(TAG, run.root), "code_hash": PR.code_hash(run.root),
            "pin": {"commit": PINNED_COMMIT, "source": PINNED_SOURCE,
                    "source_blob": PINNED_SOURCE_BLOB, "plan": PINNED_PLAN,
                    "plan_blob": PINNED_PLAN_BLOB, "copied": list(COPIED)},
            "inputs": dict(sorted(run.inputs.items())), "argv": safe_argv(sys.argv),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start_t)),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - start_t, 1),
            "environment": {**F.environment(run.root), "xla_flags": os.environ.get("XLA_FLAGS"),
                            "nice": os.getpriority(os.PRIO_PROCESS, 0),
                            "processor": platform.processor()}, **(extra or {})}


# ------------------------------------------------------------------------- subcommands


def read_manifest(phase: str, other: str = "") -> tuple[dict, str]:
    """(the phase's manifest, its sha256): outputs/vg1/manifest-<phase>.json from
    scripts/vg1_analysis.py ``manifest``, the main run's committed as it is; or ``other``, a
    planted one (a partial run, as the tests use)."""
    rel = MANIFEST.format(phase=phase)
    path = ROOT / rel
    if other:
        path, rel, phase = Path(other), _rel(Path(other)), f"other:{phase}"
    if not path.is_file():
        raise SystemExit(f"{rel} not found: run scripts/vg1_analysis.py manifest first")
    m = json.loads(path.read_text())
    if m.get("record") != "vg1-manifest" or m.get("phase") != phase.split(":")[-1]:
        raise SystemExit(f"{rel} is not VG-1's {phase} manifest")
    if phase == "main":
        if _git_main("ls-files", "--error-unmatch", "--", rel).returncode:
            raise SystemExit(f"{rel} is not committed (Section 9, step 6)")
        if _git_main("diff", "--quiet", "HEAD", "--", rel).returncode:
            raise SystemExit(f"{rel} differs from HEAD")
    return m, sha256_file(path)


def _analysis():
    """scripts/vg1_analysis.py (this repo's; stdlib and NumPy at import)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vg1_analysis", ROOT / "scripts" /
                                                  "vg1_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _world_and_counts(run: Run):
    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    world = load_world(run)
    G = AN.geometry_clusters(world.tab)[1]
    return world, PB.cluster_counts(G)


def cmd_worktree(a) -> None:
    make_worktree(Path(a.worktree))


def cmd_verify(a) -> None:
    wt = enter_worktree(Path(a.worktree))
    bad = verify(wt)
    for b in bad:
        print(f"  {b}")
    if bad:
        raise SystemExit(1)
    print("verify: OK")


def cmd_gate(a) -> None:
    t0 = time.time()
    run = start("gate", Path(a.worktree))
    unlock()
    world, counts = _world_and_counts(run)
    gate = wiring_gate(run, world, counts)
    if a.out:
        write_once(Path(a.out), {"record": "vg1-c1-gate", "gate": gate,
                                 "provenance": stamp(run, LABEL["gate"], t0)})


def cmd_measure(a) -> None:
    t0 = time.time()
    tag = _analysis().tag_gate() if a.phase == "main" and not a.manifest else None
    manifest, msha = read_manifest(a.phase, a.manifest)
    run = start("measure", Path(a.worktree))
    run.note_input(_rel(Path(a.manifest)) if a.manifest else MANIFEST.format(phase=a.phase),
                   msha)
    out = Path(a.out) if a.out else ROOT / OUT_DIR / f"{a.phase}.json"
    refuse_out(out)
    if out.exists():
        raise SystemExit(f"refusing: {_rel(out)} exists (write-once)")
    unlock()
    world, counts = _world_and_counts(run)
    gate = wiring_gate(run, world, counts)
    nets = [e for e in manifest["runs"] if e.get("final")]
    want = set(a.network or [])
    if want:
        nets = [e for e in nets if e["name"] in want]
    recs = {e["name"]: measure_network(run, world, counts, e) for e in nets}
    skipped = [e["name"] for e in manifest["runs"] if not e.get("final")]
    write_once(out, {"record": "vg1-c1", "phase": a.phase,
                     "partial": bool(want or a.out or a.manifest),
                     "networks": recs, "no_final_checkpoint": skipped, "gate": gate,
                     "manifest_sha256": msha, "vg1_tag": tag,
                     "provenance": stamp(run, LABEL[a.phase], t0)})


def cmd_snapshots(a) -> None:
    from apexzero.costate import analysis as AN

    t0 = time.time()
    tag = _analysis().tag_gate()
    manifest, msha = read_manifest("main")
    run = start("snapshots", Path(a.worktree))
    run.note_input(MANIFEST.format(phase="main"), msha)
    out = Path(a.out) if a.out else ROOT / OUT_DIR / "snapshots.json"
    refuse_out(out)
    if out.exists():
        raise SystemExit(f"refusing: {_rel(out)} exists (write-once)")
    unlock()
    world, counts = _world_and_counts(run)
    gate = wiring_gate(run, world, counts)
    per, series = {}, {}
    for e in manifest["runs"]:
        if e.get("seed") not in SNAPSHOT_SEEDS or not e.get("checkpoints"):
            continue
        name = f"runs/{e['name']}"
        rows = []
        for pin in e["checkpoints"]:
            agent = CS.load_agent(name, dict(pin), run.root)
            run.note_input(pin["path"], pin["sha256"])
            c1, _ = network_c1(run, world, counts, name, agent)
            rows.append({"timesteps": pin["timesteps"], **c1})
        per[e["name"]] = rows
        series[e["name"]] = [(r["timesteps"], r["stat"]) for r in rows
                             if r.get("stat") is not None and math.isfinite(r["stat"])]
        print(f"{e['name']}: {len(rows)} snapshots", flush=True)
    dev = AN.c1_dev(series)
    write_once(out, {"record": "vg1-c1-snapshots", "label": "descriptive", "networks": per,
                     "c1_dev_rho": dev["rho"], "gate": gate, "manifest_sha256": msha,
                     "vg1_tag": tag, "provenance": stamp(run, LABEL["snapshots"], t0)})


def lower_priority(level: int) -> None:
    try:
        if os.getpriority(os.PRIO_PROCESS, 0) < level:
            os.setpriority(os.PRIO_PROCESS, 0, level)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worktree", default=str(WORKTREE))
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("worktree", help="create the worktree and its links")
    sub.add_parser("verify", help="the pin, the freeze and every bound file (writes nothing)")
    g = sub.add_parser("gate", help="the wiring gate on G1")
    g.add_argument("--out", default="")
    m = sub.add_parser("measure", help="C1 of the phase's networks")
    m.add_argument("--phase", choices=["pilot", "pilot2", "main"], required=True)
    m.add_argument("--network", action="append",
                   help="only this run (repeatable; a partial run needs --out)")
    m.add_argument("--manifest", default="",
                   help="a planted manifest instead of the phase's (a partial run, with --out)")
    m.add_argument("--out", default="")
    s = sub.add_parser("snapshots", help="R2_c over the snapshots of seeds 30-33 (descriptive)")
    s.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if getattr(a, "network", None) or getattr(a, "manifest", None):
        out = Path(a.out).resolve() if a.out else None
        if out is None or (ROOT / OUT_DIR).resolve() in (out, *out.parents):
            raise SystemExit("--network and --manifest make a partial run: give --out outside "
                             "outputs/vg1/c1")
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                       f"intra_op_parallelism_threads={a.threads}")
    lower_priority(a.nice)
    bad = pinned_problems()
    if bad:
        raise SystemExit("refusing: the pin does not hold: " + "; ".join(bad))
    {"worktree": cmd_worktree, "verify": cmd_verify, "gate": cmd_gate, "measure": cmd_measure,
     "snapshots": cmd_snapshots}[a.cmd](a)


if __name__ == "__main__":
    main()
