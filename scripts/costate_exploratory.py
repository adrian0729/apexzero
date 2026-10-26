"""Idea 2's follow-ups (lane A of docs/plan/follow-through.md): EXPLORATORY.

The plan, docs/findings/costate-followups-plan.md, was committed alone in ``PLAN_COMMIT``
before anything here was computed. It fixes every item's question, computation, rows and
expected directions, and any later change to it is an appended, dated amendment (its Section
7). Every number this script writes is exploratory. None changes a decision of
costate-prereg-v1 (outputs/costate/analyse/decisions.json).

    uv run --no-sync python scripts/costate_exploratory.py <item> [options]
    uv run --no-sync python scripts/costate_exploratory.py verify
    uv run --no-sync python scripts/costate_exploratory.py status

**Items** (one subcommand each; plan Section 4; ``ITEMS``): cs-1 to cs-15 and bc-a0. An
item's computation is registered with ``computes``. Until it is, the item's subcommand runs
every check below, lists the inputs it verified, and writes nothing. ``--check`` stops after
the checks for an implemented item too. ``verify`` checks the plan, the freeze and every bound
file; ``status`` lists the items and their outputs. Neither writes anything.

**Checks before any item** (``preflight``; plan R2 and R3):
- **the plan** (``plan_problems``):
  - committed, first in ``PLAN_COMMIT``, an ancestor of HEAD, unchanged against HEAD;
  - each version on HEAD's history an append-only extension of the one before;
  - the appended text made only of ``### Amendment N (YYYY-MM-DD)`` subsections, numbered on
    from the last;
- **the freeze** (``freeze_problems``):
  - the tag is annotated and is the object tag.json pins, and it is an ancestor of HEAD;
  - src/, scripts/costate.py and scripts/_bootstrap.py equal the tag's (git and byte by byte),
    and so does the design document;
  - step 1's and step 2's frozen code equal their tags', and those tags are the objects the
    record registers (scripts/costate.py's ``frozen_steps_problems`` and
    ``step_tags_problems``);
  - the costate code hash of the tree is the record's, and prereg.json is the tag's;
  - decisions.json has its committed sha256 and is committed at HEAD as those bytes;
  - the package versions are the record's;
  - the tag is not looked up on the remote: the confirmatory runs established that, and an
    exploratory run needs no network;
- **the imports** (``import_problems``): every module loaded from inside the repository is
  frozen code (src/apexzero, scripts/costate.py, scripts/fields.py, scripts/_bootstrap.py)
  or this script;
- **the inputs** (``Run.read``):
  - every bound file the item reads has the sha256 that decisions.json's ``inputs`` or the
    record's ``outputs`` gives it (``bound_hashes``);
  - a post-tag file nothing binds (the 25 non-decision networks' activations) is read only if
    its stamp binds its sha256, names the record's code hash and the tag, and is neither
    partial nor under a deviation (``unbound_problems``);
  - checkpoints are loaded only through the record's pins (scripts/costate.py's
    ``load_agent``);
  - every sha256 read is recorded in the item's stamps.

Only after these checks, and only for an item that computes trained networks
(``Item.trained``), does ``unlock`` call apexzero.costate.critic's, apexzero.costate.probes'
and apexzero.fields.network's ``unlock_trained``, as scripts/costate.py's ``_post_tag``
does. The worker pools mirror those locks (``apexzero.costate.workers``).

**Rows** (plan R5; decision #8). ``guard_rows`` refuses any row outside the approach and exit
phases, any multi-basin row, and, for a K_P component, any row in a phase where K2b's frozen
rule excludes it or outside the network's frozen decision mask. Every score of a network output
against a window-problem target goes through it. ``mask_problems`` compares a rebuilt decision
mask with decisions.json's ``decision_mask_sha256``.

**Outputs** (plan R8; ``Run``):
- **where:** under outputs/costate/exploratory/<item>/ (and /<--run>/);
- **write-once:** nothing is overwritten or deleted, so a re-run takes a new ``--run``;
- **stamps:** each output carries a stamp, embedded in a JSON object or as a
  <file>.provenance.json sidecar, labelled 'exploratory'. It names the item, the plan (path,
  first commit, sha256, amendments), HEAD, the paths changed against it, the code hash
  (re-checked before writing), the tag, the inputs' sha256, the arguments
  (repository-relative), the environment and the pool;
- **partial runs:** --out, --network and --limit make a run partial, and a partial run writes
  only outside outputs/costate;
- **never:** nothing is written under outputs/costate's inputs, gates, nulls, post, logs or
  analyse, and nothing is read or written under outputs/sealed.

**Compute** (plan R9): pools of ``DEFAULT_WORKERS`` (4) workers × ``DEFAULT_THREADS`` (2) XLA
threads (``pool``), BLAS pinned to one thread for a pooled item before numpy loads, and the
process reniced by ``--nice`` (default 10) before any item runs.

**The lockbox** (plan R7): this script never reads the lockbox circuits or outputs/sealed.

**Implemented items.**
- CS-1 (``cs1``, the section 'CS-1: the xi tail'; plan Section 4 with amendment 1): the tail
  reproduced from the findings, where it sits, g_xi split by a fresh VJP into channel groups,
  and R2_c under the Huber and rank calibrations after their plants.
- CS-2 to CS-5 (plan Section 4 with amendment 2; the sections 'CS-2 to CS-5: shared' and one
  per item): the descriptives pack (``cs2``), C3-int on s and xi (``cs3``), Delta C1, R2_z and
  the MLP-on-z ceiling (``cs4``, pooled) and W1 on the Pi targets (``cs5``, pooled). None
  evaluates a trained network, so none unlocks.
- CS-10 (ii) and CS-11 on D (plan Section 4 with amendment 3; the section 'CS-10, CS-11: own
  runs'): each decision network's deterministic policy driven from D's 477 keys for 800 steps
  or to the strip's end; C1-own on every 5th state of its own runs, with its own interior
  flag (``cs10``, pooled), and the policy's own inside-width price Pi^in,own by a VJP through
  the own loop on the perturbed road, with W1's three comparisons (``cs11``, pooled). Both
  unlock. CS-10 (iii) and N's CS-11 wait for CS-9.
- CS-7 and CS-8 (plan Section 4 with amendment 7; the sections 'CS-7, CS-8: shared', 'CS-7:
  critic layers' and 'CS-8: C2-sw and C2'): ridge probes of lambda and the frozen lambda_res
  on z and z + each of the critic's layers and head units against 20 random-init critics,
  with the actor's h3 against the pool as the comparator (``cs7``, pooled); C2-sw's
  random-only null on the approach and exit pedal-switching rows (G1's scaling first, per
  network under R6's 3 h rule), C2-sw on sigma^ped residualised on f_D's decoys, and C2 with
  a decoy chosen on targets alone to fit geometry 6 (``cs8``, pooled). Neither evaluates a
  trained network, so neither unlocks.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
import itertools
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path


def _flag(argv: Sequence[str], flag: str) -> str | None:
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _command(argv: Sequence[str]) -> str | None:
    """The subcommand on the command line (the first argument that is not an option or an
    option's value)."""
    skip = False
    for a in list(argv)[1:]:
        if skip:
            skip = False
        elif a in ("--threads", "--workers", "--nice", "--out", "--run", "--network",
                   "--limit"):
            skip = True
        elif not a.startswith("-"):
            return a
    return None


# The items with worker pools (plan Section 1, "pooled"), and the BLAS libraries' thread
# variables (scripts/costate.py's BLAS_THREADS): a pooled item pins BLAS to one thread before
# numpy loads, in this process and so in every worker it spawns.
POOLED_ITEMS = ("cs-4", "cs-5", "cs-7", "cs-8", "cs-9", "cs-10", "cs-11", "cs-12", "bc-a0")
BLAS_THREADS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS")
DEFAULT_THREADS = 2
_T = _flag(sys.argv, "--threads")
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or DEFAULT_THREADS}")
if _command(sys.argv) in POOLED_ITEMS:
    os.environ.update(dict.fromkeys(BLAS_THREADS, "1"))

import _bootstrap  # noqa: F401  (makes src/ importable)
import costate as CS  # scripts/costate.py, frozen: read, never edited
import fields as F  # scripts/fields.py, step 2's helpers, frozen at its tag

ROOT = Path(__file__).resolve().parents[1]
TAG = CS.TAG
OUT_REL = CS.OUT_REL                                   # outputs/costate
EXP_REL = f"{OUT_REL}/exploratory"
PROTECTED = ("inputs", "gates", "nulls", "post", "logs", "analyse")   # never written here
SCRIPT_REL = "scripts/costate_exploratory.py"
PLAN_REL = "docs/findings/costate-followups-plan.md"
PLAN_COMMIT = "eaf1e80e92e065634333a0299c62dbe6d14d5b9f"   # the plan, committed alone
AMENDMENTS_HEADING = "## 7. Amendments"
AMENDMENT_RE = re.compile(r"^### Amendment (\d+) \((\d{4}-\d{2}-\d{2})\)\s*$")
DECISIONS_REL = CS.DECISIONS_REL                       # outputs/costate/analyse/decisions.json
DECISIONS_SHA256 = "8119f91f7c1a24186fad011a83ef931e441626d7493355272b762e8aa97b0ee4"
RECORD_CODE_HASH = "d22e670cd059063af38c1af74c039a0ae11933b50443ab40f14256bb49dbfff3"
FROZEN_PATHS = CS.CODE_PATHS                           # src, scripts/costate.py, _bootstrap
ALLOWED_SCRIPTS = ("costate.py", "fields.py", "_bootstrap.py", "costate_exploratory.py")
LABEL = "exploratory"
SCORED_PHASES = ("approach", "exit")                   # plan R5: the only phases scored
DEFAULT_WORKERS = 4                                    # plan R9 (another workflow shares the Mac)
DEFAULT_NICE = 10


# ------------------------------------------------------------------------- the items


@dataclasses.dataclass(frozen=True)
class Item:
    """One follow-up (plan Section 4). ``reads``: its input groups (``group_inputs``);
    ``trained``: it computes trained networks (plan R4: ``unlock`` after the checks);
    ``solves``: it re-solves window problems (targets only); ``needs``: items whose outputs it
    reads."""

    id: str
    title: str
    reads: tuple[str, ...]
    trained: bool = False
    solves: bool = False
    needs: tuple[str, ...] = ()

    @property
    def cmd(self) -> str:
        return self.id.lower()

    @property
    def pooled(self) -> bool:
        return self.cmd in POOLED_ITEMS


ITEMS: tuple[Item, ...] = (
    Item("CS-1", "The xi tail: where it sits, critic against map, other calibrations",
         ("world", "critic:decision", "activations:decision", "gates:k4", "checkpoints"),
         trained=True),
    Item("CS-2", "Descriptives pack", ("world", "critic:all", "checkpoints")),
    Item("CS-3", "C3-int on s and xi (network x geometry)",
         ("world", "critic:all", "step1", "step2", "checkpoints"), needs=("CS-1",)),
    Item("CS-4", "Delta C1, R2_z and the MLP-on-z ceiling",
         ("world", "critic:decision", "activations:decision", "checkpoints"), needs=("CS-1",)),
    Item("CS-5", "W1 on the Pi targets",
         ("world", "critic:decision", "activations:decision", "checkpoints"), needs=("CS-1",)),
    Item("CS-6", "C1-dev over the 16 snapshots", ("world", "snapshots"), trained=True,
         needs=("CS-1",)),
    Item("CS-7", "Critic-layer and head-unit probes",
         ("world", "activations:decision", "checkpoints"), needs=("CS-1",)),
    Item("CS-8", "C2-sw's random-only null; C2-sw residualised; C2 with a decoy that fits "
         "geometry 6", ("world", "activations:decision", "gates:c2_null", "checkpoints")),
    Item("CS-9", "N: per-agent replication of C1, C2 and C2-sw, with K2b on N",
         ("world", "keys:n", "gates:k0", "gates:k2b", "checkpoints"), trained=True,
         solves=True),
    Item("CS-10", "C1-own on own-run states, then on N",
         ("world", "critic:decision", "own", "checkpoints"), trained=True, needs=("CS-9",)),
    Item("CS-11", "The policy's own width price Pi^in,own",
         ("world", "critic:decision", "checkpoints"), trained=True, needs=("CS-9",)),
    Item("CS-12", "C2 and C2-sw beyond the 9 decision networks (primary only)",
         ("world", "activations:other", "checkpoints")),
    Item("CS-13", "C2-pairs", ("world", "activations:decision"), needs=("CS-9",)),
    Item("CS-14", "q_c, the action-projected critic signal, with K2b for q_c",
         ("world", "gates:k2b", "checkpoints"), trained=True, solves=True),
    Item("CS-15", "The FWL check, actor h1 and h2 probes, C2 on D",
         ("world", "activations:decision")),
    Item("BC-a0", "Braking and corner targets pilot (targets only)",
         ("world", "gates:k2b", "keys:a2"), solves=True),
)
BY_CMD = {it.cmd: it for it in ITEMS}

# Input groups checked when an item runs (``group_inputs``); the others are checked by the
# item's own code when it reads them (their own stamps, step 1's and step 2's records, the
# key files' content hashes, the checkpoints' pins).
CHECKED_GROUPS = ("world", "critic:decision", "critic:all", "activations:decision",
                  "activations:other", "own", "probes", "nulls")
ITEM_GROUPS = ("step1", "step2", "keys:n", "keys:a2", "checkpoints", "snapshots")

COMPUTE: dict[str, Callable] = {}


def computes(item_id: str):
    """Register ``fn(run)`` as an item's computation (plan Section 4)."""
    if item_id not in {it.id for it in ITEMS}:
        raise KeyError(f"{item_id} is not an item of the plan")

    def deco(fn):
        COMPUTE[item_id] = fn
        return fn

    return deco


# ------------------------------------------------------------------------- git helpers


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


_HASHED: dict[tuple, str] = {}


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


def history(rel: str, root: Path = ROOT, rev: str = "HEAD") -> list[tuple[str, bytes | None]]:
    """[(commit, the file's bytes there, or None where it is deleted)] of the commits on
    ``rev``'s history that touched ``rel``, oldest first. Raises RuntimeError when git fails."""
    r = _git(root, "log", "--reverse", "--format=%H", rev, "--", rel)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip()[:200] or "git log failed")
    out = []
    for c in r.stdout.decode().split():
        b = _git(root, "show", f"{c}:{rel}")
        out.append((c, b.stdout if b.returncode == 0 else None))
    return out


def _changed(root: Path, rev: str, paths: Sequence[str]) -> list[str]:
    return F.changed_since(rev, tuple(paths), root)


# ------------------------------------------------------------------------- the plan


def _amendment_problems(text: str, first_n: int) -> tuple[int, list[str]]:
    """(the number of the last amendment, problems) of text appended after the plan's first
    version: every heading must be a dated amendment numbered on from ``first_n``."""
    import datetime

    n, out = first_n, []
    for ln in text.splitlines():
        if not ln.startswith("#"):
            continue
        m = AMENDMENT_RE.match(ln)
        if m is None:
            out.append(f"the appended heading {ln.strip()!r} is not '### Amendment N "
                       "(YYYY-MM-DD)'")
            continue
        k, date = int(m.group(1)), m.group(2)
        if k != n + 1:
            out.append(f"amendment {k} follows amendment {n}: they are numbered on, from 1")
        try:
            datetime.date.fromisoformat(date)
        except ValueError:
            out.append(f"amendment {k}'s date {date} is not a date")
        n = k
    return n, out


def _amendments_in(text: str) -> int:
    i = text.find(AMENDMENTS_HEADING)
    return 0 if i < 0 else len([ln for ln in text[i:].splitlines() if AMENDMENT_RE.match(ln)])


def plan_problems(root: Path = ROOT, plan_commit: str = PLAN_COMMIT,
                  rel: str = PLAN_REL) -> list[str]:
    """Why no item may run under the plan (module docstring; plan Section 7's rule)."""
    root = Path(root)
    path = root / rel
    if not path.exists():
        return [f"the plan {rel} is missing"]
    try:
        vers = history(rel, root)
    except RuntimeError as e:
        return [f"git failed reading the plan's history: {e}"]
    if not vers or vers[0][1] is None:
        return [f"the plan {rel} is not committed"]
    out = []
    first_commit, first = vers[0]
    if not first_commit.startswith(plan_commit):
        out.append(f"the plan was first committed in {first_commit[:12]}, not {plan_commit[:12]}")
    if _git(root, "merge-base", "--is-ancestor", plan_commit, "HEAD").returncode != 0:
        out.append(f"the plan's commit {plan_commit[:12]} is not an ancestor of HEAD")
    if AMENDMENTS_HEADING.encode() not in first:
        out.append(f"the plan's first version has no '{AMENDMENTS_HEADING}' section")
    ch = _changed(root, "HEAD", (rel,))
    if ch:
        out.append(f"the plan differs from HEAD ({ch[0]}): commit it first")
    prev = first
    for c, b in vers[1:]:
        if b is None:
            out.append(f"the plan is deleted in {c[:12]}")
            continue
        if not b.startswith(prev):
            out.append(f"the plan was edited in place in {c[:12]}: only appended amendments are "
                       "allowed (plan Section 7)")
        prev = b
    disk = path.read_bytes()
    if not disk.startswith(first):
        out.append("the plan on disk is not an append-only extension of its first version")
    else:
        try:
            tail = disk[len(first):].decode()
        except UnicodeDecodeError:
            return out + ["the plan's appended text is not UTF-8"]
        _, bad = _amendment_problems(tail, _amendments_in(first.decode()))
        out += bad
    return out


def plan_identity(root: Path = ROOT, rel: str = PLAN_REL) -> dict:
    """The plan as the stamps name it: path, first commit, sha256 on disk, amendments."""
    path = Path(root) / rel
    text = path.read_text()
    vers = history(rel, root)
    return {"path": rel, "first_commit": vers[0][0] if vers else None,
            "sha256": sha256_file(path), "amendments": _amendments_in(text)}


# ------------------------------------------------------------------------- the freeze


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


def import_problems(root: Path = ROOT, modules: dict | None = None) -> list[str]:
    """Every module loaded from inside the repository must be frozen code or this script
    (scripts/costate.py's rule, with this script allowed)."""
    import apexzero

    root = Path(root).resolve()
    want = (root / "src" / "apexzero").resolve()
    got = Path(apexzero.__file__).resolve().parent
    out = [] if got == want else [f"apexzero is imported from {got}, not {want}"]
    prefixes = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)}
    allowed = {(root / "scripts" / n).resolve() for n in ALLOWED_SCRIPTS}
    for name, mod in sorted((modules if modules is not None else dict(sys.modules)).items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        fp = Path(f).resolve()
        if root not in fp.parents or any(p == fp or p in fp.parents for p in prefixes):
            continue
        if want in fp.parents or fp in allowed:
            continue
        out.append(f"{name} imported from {fp.relative_to(root)}, outside the frozen code")
    return out


# ------------------------------------------------------------------------- inputs


def _is_sealed(path: Path) -> bool:
    """Whether a path lies under any outputs/sealed (plan R7), its links followed."""
    p = Path(path).resolve()
    return any(q.name == "sealed" and q.parent.name == "outputs" for q in (p, *p.parents))


def refuse_sealed(path: Path) -> Path:
    if _is_sealed(path):
        raise SystemExit(f"refusing {path}: nothing under outputs/sealed is read or written "
                         "(plan R7)")
    return Path(path)


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


# ------------------------------------------------------------------------- rows (plan R5)


class RowsRefused(SystemExit):
    """A score was asked on rows decision #8 or K2b's frozen rule keeps out (plan R5)."""


def guard_rows(tab, rows, component: str | None = None, k2b_excluded: dict | None = None,
               mask=None):
    """``rows`` as a boolean array, after refusing (``RowsRefused``) any row outside the
    approach and exit phases, any multi-basin row, and, for a K_P component, any row in a
    phase where K2b's frozen rule excludes it (``k2b_excluded``, the record's
    frozen.k2b_excluded) or outside ``mask`` (the component's column of the network's frozen
    decision mask). Every score of a network output against a window-problem target passes
    its rows through here (plan R5)."""
    import numpy as np

    from apexzero.costate import targets as TG

    rows = np.asarray(rows, bool)
    ph = np.asarray(tab["phase"]).astype(str)
    if rows.shape != ph.shape:
        raise ValueError(f"rows {rows.shape} do not match the table's {ph.shape}")
    bad = rows & ~np.isin(ph, SCORED_PHASES)
    if bad.any():
        got = sorted(set(ph[bad].tolist()))
        raise RowsRefused(f"refusing a score on {int(bad.sum())} rows of phase {got}: only "
                          f"{', '.join(SCORED_PHASES)} rows are scored (decision #8, plan R5)")
    if "multi_basin" in tab and (rows & np.asarray(tab["multi_basin"], bool)).any():
        raise RowsRefused("refusing a score on multi-basin rows (design Section 3.2)")
    if component is not None and component in TG.K_P:
        if k2b_excluded is None:
            raise RowsRefused(f"a score of lambda_{component} needs K2b's frozen exclusions")
        excl = [p for p, comps in k2b_excluded.items() if component in comps]
        hit = rows & np.isin(ph, excl)
        if hit.any():
            raise RowsRefused(f"refusing lambda_{component} on {int(hit.sum())} rows of a phase "
                              f"K2b excludes it from ({sorted(excl)}; plan R5)")
        if mask is not None:
            out = rows & ~np.asarray(mask, bool)
            if out.any():
                raise RowsRefused(f"refusing lambda_{component} on {int(out.sum())} rows outside "
                                  "the network's frozen decision mask (plan R5)")
    return rows


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


# ------------------------------------------------------------------------- outputs


def _abs_out(path: Path) -> Path:
    return Path(path).resolve()


def safe_argv(argv: Sequence[str], root: Path = ROOT) -> list[str]:
    """The arguments as a stamp records them: paths inside the repository relative to it,
    other absolute paths reduced to '<outside>/<name>' (no local paths; plan R8)."""
    root = Path(root).resolve()
    out = []
    for i, a in enumerate(argv):
        if i == 0:
            out.append(SCRIPT_REL)
            continue
        val, pre = a, ""
        if a.startswith("--") and "=" in a:
            pre, val = a.split("=", 1)
            pre += "="
        if os.path.isabs(val):
            p = Path(val).resolve()
            try:
                val = p.relative_to(root).as_posix()
            except ValueError:
                val = f"<outside>/{p.name}"
        out.append(pre + val)
    return out


def savez_bytes(arrays: dict) -> bytes:
    """scripts/costate.py's ``savez`` (one deflated '<name>.npy' member per array, every zip
    timestamp fixed at 1980-01-01) into memory: the same arrays give the same bytes."""
    import numpy as np
    from numpy.lib import format as npf

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as zf:
        for name, v in arrays.items():
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(zi, "w", force_zip64=True) as fh:
                npf.write_array(fh, np.asanyarray(v), allow_pickle=False)
    return buf.getvalue()


def _write_once(path: Path, data: bytes) -> Path:
    path = refuse_sealed(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as fh:
            fh.write(data)
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {path.name}: outputs are write-once (plan R8); "
                         "give a new --run") from None
    return path


class Run:
    """One item's run: its output directory, the inputs it read (sha256 checked and
    recorded), its pool and its stamps (module docstring 'Outputs')."""

    def __init__(self, item: Item, args, root: Path = ROOT, rec: dict | None = None,
                 decisions: dict | None = None, bound: dict | None = None):
        self.item, self.args, self.root = item, args, Path(root)
        self.costate = self.root / OUT_REL
        self.partial = [f for f, v in (("--out", getattr(args, "out", None)),
                                       ("--network", getattr(args, "network", None)),
                                       ("--limit", getattr(args, "limit", None)))
                        if v is not None]
        self.out = self._out_dir()
        self.rec, self.decisions, self.bound = rec, decisions, bound
        self.inputs: dict[str, str] = {}
        self.workers: int | None = None
        self.start = F._now()
        self.code_hash: str | None = None

    def _out_dir(self) -> Path:
        args = self.args
        canon = self.root / EXP_REL
        run = getattr(args, "run", None)
        if run is not None and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run)):
            raise SystemExit(f"--run {run!r}: a plain name (letters, digits, '.', '_', '-')")
        if getattr(args, "out", None) is None:
            if self.partial:
                raise SystemExit(f"{', '.join(self.partial)} makes a partial run: give --out "
                                 f"outside {OUT_REL}")
            base = canon
        else:
            base = Path(args.out)
            res = _abs_out(base)
            # outputs/costate here, and wherever its exploratory/ link leads
            for d in {_abs_out(self.root / OUT_REL), _abs_out(canon), _abs_out(canon).parent}:
                if res == d or d in res.parents:
                    raise SystemExit(f"--out {args.out}: a partial run writes outside "
                                     f"{OUT_REL}")
        refuse_sealed(base)
        res = _abs_out(base)
        for d in PROTECTED:
            pd = _abs_out(self.root / OUT_REL / d)
            if res == pd or pd in res.parents:
                raise SystemExit(f"refusing to write under {OUT_REL}/{d} (plan R8)")
        out = base / self.item.cmd
        return out / run if run else out

    # -- inputs

    def read(self, rel: str, kind: str = "bound") -> Path:
        """A file under outputs/costate, checked (module docstring 'the inputs') and recorded."""
        path = refuse_sealed(self.costate / rel)
        if not path.exists():
            raise SystemExit(f"{self.item.id}: its input {OUT_REL}/{rel} is missing")
        if kind == "bound":
            want = (self.bound or {}).get(rel)
            if want is None:
                raise SystemExit(f"{rel} is not bound by decisions.json or the record")
            got = sha256_file(path)
            if got != want:
                raise SystemExit(f"{rel} is not the file decisions.json or the record binds "
                                 f"({got[:12]} against {want[:12]})")
        elif kind == "stamped":
            bad = unbound_problems(path, self.rec or {}, F.tag_identity(TAG, self.root))
            if bad:
                raise SystemExit(f"{rel}: {'; '.join(bad)}")
            got = sha256_file(path)
        else:
            raise ValueError(kind)
        self.inputs[f"{OUT_REL}/{rel}"] = got
        return path

    def note_input(self, rel: str, sha256: str) -> None:
        """Record an input checked by the item's own code (a key file, step 2's rows, a
        checkpoint through its pin)."""
        self.inputs[rel] = sha256

    # -- the pool

    def pool(self, fn: Callable, units: Sequence, state=None, names=None) -> list:
        """``apexzero.costate.workers.run`` with this run's workers (at most one per unit);
        recorded in every stamp the run writes."""
        from apexzero.costate import workers as WK

        n = max(1, min(int(getattr(self.args, "workers", None) or DEFAULT_WORKERS), len(units)))
        self.workers = n
        return WK.run(fn, list(units), n, state, names)

    # -- stamps and writes

    def stamp(self, files: dict | None = None, content_sha256: str | None = None) -> dict:
        from apexzero.costate import prereg as PR
        from apexzero.costate import workers as WK

        now = PR.code_hash(self.root)
        if self.code_hash is not None and now != self.code_hash:
            raise SystemExit(f"refusing to write {self.item.id}'s output: the code changed "
                             "while it ran")
        bad = import_problems(self.root)
        if bad:
            raise SystemExit(f"refusing to write {self.item.id}'s output: {'; '.join(bad[:4])}")
        out = {"label": LABEL, "item": self.item.id, "title": self.item.title,
               "plan": plan_identity(self.root), "head": F._head(self.root),
               "changed_vs_head": _changed(self.root, "HEAD", (SCRIPT_REL, PLAN_REL)
                                           + FROZEN_PATHS + ("scripts/fields.py",)),
               "code_hash": now, "script_sha256": sha256_file(self.root / SCRIPT_REL),
               "tag": F.tag_identity(TAG, self.root), "decisions_sha256": DECISIONS_SHA256,
               "inputs": dict(sorted(self.inputs.items())), "argv": safe_argv(sys.argv,
                                                                              self.root),
               "partial": list(self.partial), "start": self.start, "end": F._now(),
               "environment": F.environment(self.root), "rows_rule": "plan R5 (decision #8)",
               "files": dict(files or {})}
        if self.workers is not None:
            out["workers"], out["threads"] = int(self.workers), WK.thread_env()
        if content_sha256 is not None:
            out["content_sha256"] = content_sha256
        return out

    def write_json(self, name: str, obj: dict) -> Path:
        """A JSON object output, write-once, its stamp embedded ('provenance', binding the
        content's sha256 as scripts/fields.py's ``content_sha256`` does)."""
        from apexzero.interp import io as IO

        if not isinstance(obj, dict) or "provenance" in obj:
            raise ValueError("a JSON output is an object without its own 'provenance'")
        body = {**obj, "label": LABEL}
        data = IO.json_bytes({**body, "provenance": self.stamp(
            content_sha256=F.content_sha256(body))})
        p = _write_once(self.out / f"{name}.json", data)
        print(f"wrote {p}", flush=True)
        return p

    def write_npz(self, name: str, arrays: dict) -> Path:
        """An npz output, write-once, with its <file>.provenance.json sidecar binding its
        sha256."""
        from apexzero.fields import prereg as FP

        data = savez_bytes(arrays)
        path = self.out / f"{name}.npz"
        st = self.stamp({path.name: _sha256_bytes(data)})
        p = _write_once(path, data)
        _write_once(FP.stamp_path(p), (json.dumps(st, indent=1, sort_keys=True) + "\n").encode())
        print(f"wrote {p}", flush=True)
        return p


# ------------------------------------------------------------------------- running an item


def load_context(root: Path = ROOT) -> tuple[dict, dict, dict[str, str]]:
    """(the record at the tag, decisions.json, the bound sha256 map)."""
    rec = CS.tagged_record(TAG, root)
    if rec is None:
        raise SystemExit(f"no record at {TAG}")
    decisions = json.loads((Path(root) / DECISIONS_REL).read_text())
    return rec, decisions, bound_hashes(rec, decisions)


def preflight(root: Path = ROOT) -> list[str]:
    """The plan, the freeze and the imports (module docstring)."""
    return plan_problems(root) + freeze_problems(root) + import_problems(root)


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


def run_item(item: Item, args, root: Path = ROOT) -> Run:
    """The checks, then (for an implemented item) its computation (module docstring)."""
    from apexzero.costate import prereg as PR

    bad = preflight(root)
    if bad:
        raise SystemExit(f"refusing to run {item.id}: " + "; ".join(bad))
    rec, decisions, bound = load_context(root)
    run = Run(item, args, root, rec, decisions, bound)
    run.code_hash = PR.code_hash(root)
    read = check_inputs(run)
    later = [g for g in item.reads if g in ITEM_GROUPS]
    print(f"{item.id}: the plan, the freeze and the imports pass; {len(read)} inputs checked"
          + (f"; checked by the item itself: {', '.join(later)}" if later else ""), flush=True)
    fn = COMPUTE.get(item.id)
    if getattr(args, "check", False):
        return run
    if fn is None:
        raise SystemExit(f"{item.id}: its computation is not implemented yet; the checks passed "
                         "and nothing was written")
    if item.trained:
        unlock()
    fn(run)
    return run


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


def status(root: Path = ROOT) -> list[dict]:
    """Each item, whether its computation is implemented, and its outputs (writes nothing)."""
    from apexzero.fields import prereg as FP

    rows = []
    base = Path(root) / EXP_REL
    for it in ITEMS:
        d = base / it.cmd
        outs = []
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file() and not p.name.endswith(FP.STAMP_SUFFIX):
                    st = FP.read_stamp(p) or {}
                    outs.append({"file": p.relative_to(base).as_posix(),
                                 "label": st.get("label"), "head": st.get("head"),
                                 "plan_sha256": (st.get("plan") or {}).get("sha256")})
        rows.append({"item": it.id, "title": it.title, "implemented": it.id in COMPUTE,
                     "trained": it.trained, "solves": it.solves, "pooled": it.pooled,
                     "needs": list(it.needs), "outputs": outs})
    return rows


# ------------------------------------------------------------------------- CS-1: the xi tail
#
# Plan Section 4 CS-1 with amendment 1 (its operational details). On the 9 decision networks:
# (1) the tail of g_xi on lambda_xi's decision rows (findings Section 7.1), its counts and the
# findings table's other columns reproduced first; (2) where it sits (geometry, s - s_apex,
# observation and kappa clipping, the preview-spacing and tyre-channel kinks); (3) g_xi split
# into channel groups by a fresh VJP of the main V (a trained-network computation, plan R4);
# (4) R2_c under the Huber and rank calibrations beside the frozen one, after their plants.

CS1_ORDER = ("G1", "G1s1", "G1s2", "G2", "G2s1", "G2s2", "G3", "G3s1", "G3s2")
# findings Section 7.1's table at its printed rounding (amendment 1 item 1): the fold-mean
# slope b_xi, the tail's rows, its weight share, its share of xi's squared error, bulk R2_c xi,
# the sign-opposite shares on the tail and the bulk (rows), the within-direction variance ratio
# of the predictions to lambda_xi; and geometry 38's share of xi's squared error (compared,
# descriptive: amendment 1 does not make it a stop)
CS1_FINDINGS_FIELDS = (("slope_fold_mean", 4), ("tail_rows", 0), ("tail_weight_share", 3),
                       ("tail_sqerr_share", 3), ("bulk_r2c_xi", 3), ("opposite_tail", 2),
                       ("opposite_bulk", 2), ("var_ratio", 2), ("geometry38_sqerr_share", 3))
CS1_FINDINGS_STOP = CS1_FINDINGS_FIELDS[:-1]
CS1_FINDINGS = {
    "G1": (0.0035, 286, 0.216, 0.099, -0.017, 0.88, 0.52, 0.07, 0.013),
    "G1s1": (0.0319, 247, 0.179, 0.649, 0.078, 0.92, 0.47, 1.79, 0.102),
    "G1s2": (0.0325, 259, 0.199, 0.774, 0.065, 0.88, 0.41, 3.24, 0.213),
    "G2": (0.0037, 283, 0.226, 0.093, -0.017, 0.90, 0.56, 0.05, 0.009),
    "G2s1": (0.0359, 315, 0.239, 0.862, 0.065, 0.89, 0.36, 5.37, 0.193),
    "G2s2": (0.0146, 229, 0.171, 0.313, -0.012, 0.89, 0.41, 0.54, 0.146),
    "G3": (0.0394, 240, 0.192, 0.775, 0.092, 0.87, 0.51, 3.22, 0.245),
    "G3s1": (0.0303, 336, 0.244, 0.864, 0.065, 0.90, 0.54, 5.60, 0.222),
    "G3s2": (0.0337, 294, 0.198, 0.776, 0.053, 0.86, 0.43, 3.19, 0.208),
}
XI = 2                                   # targets.STATE_NAMES.index('xi')
N_PREVIEW = 40                           # racing.N_PREVIEW: 5 channels per preview point
EDGE_NEAR = tuple(5 * k + c for k in range(N_PREVIEW // 2) for c in range(4))
EDGE_FAR = tuple(5 * k + c for k in range(N_PREVIEW // 2, N_PREVIEW) for c in range(4))
EDGE = EDGE_NEAR + EDGE_FAR              # the 160 edge-pose channels (lx, ly, rx, ry)
LATERAL_XI = (204, 205)                  # racing.observe's ego channels 4 (lateral) and 5 (xi)
XI_SET = EDGE + LATERAL_XI               # the plan's 'channels xi moves' for the clip split
TYRE_UTIL = (212, 213)                   # min(util, 2) - 1, front and rear
GROUPS = ("near_edge", "far_edge", "lateral_xi", "other_channels", "accel_term")
KINK_VX = {"kink_vx25": 25.0, "kink_vx83": 500.0 / 6.0}   # d_max = clip(6 vx, 150, 500)
KINK_BAND = 1.0                          # m/s
TYRE_STEP = 1e-3                         # rad: the tyre-kink flag's step in xi
PHASE_BIN = 25.0                         # m: bins of s - s_apex
CS1_GEOMETRY = 38                        # (44.7 m, 180 deg), frozen.geometries[38]
SUM_TOL = 1e-10                          # the groups against the stored g_xi (relative)
SUM_FLOOR = 1e-12                        # ... with |g_xi| floored at this x sum |c_k|
WIRING_TOL = 1e-8                        # the frozen C1 against decisions.json (plan 'Reuse')
E_COUNT = 6                              # amendment 1 item 5: 'at least 6 of 9'
RULE_GAIN, RULE_NETWORKS = 0.1, 5        # plan CS-1 'Rule for later items'
CALIBRATIONS = ("frozen", "huber", "rank")
PLANTS = ("zero", "noise", "negated", "direction_only")
CS1_COMPONENTS = ("s", "xi")             # K_P's columns 0 and 2


def xi_tail(g, w, direction):
    """[n] findings Section 7.1's tail: the calibration's leverage cut (``analysis``'s
    ``leverage_weights`` < 1, K_LEV = 3) of g centred on its weighted median within each turn
    direction (``analysis._robust_centre``), on all the given rows at once."""
    import numpy as np

    from apexzero.costate import analysis as AN

    g, w = np.asarray(g, float), np.asarray(w, float)
    grp = AN._groups(np.asarray(direction), True, len(g))
    return AN.leverage_weights(AN._robust_centre(g, w, grp), w) < 1.0


def _ratio(a: float, b: float) -> float:
    """a / b, inf where only b is 0, nan where both are (amendment 1 item 2)."""
    if b > 0:
        return float(a / b)
    return float("inf") if a > 0 else float("nan")


def flag_table(flag, tail, w, err) -> dict:
    """A flag on the tail and the bulk: rows, the shares on rows and weights, their
    enrichments (tail share / bulk share) and the flag's share of the tail's squared error
    (``err`` per row: w (pred - lambda)^2)."""
    import numpy as np

    flag, tail = np.asarray(flag, bool), np.asarray(tail, bool)
    w, err = np.asarray(w, float), np.asarray(err, float)

    def share(m, v):
        tot = v[m].sum()
        return float(v[m & flag].sum() / tot) if tot > 0 else float("nan")

    one = np.ones(len(flag))
    st, sb, wt, wb = share(tail, one), share(~tail, one), share(tail, w), share(~tail, w)
    e = err[tail].sum()
    return {"tail_rows": int((flag & tail).sum()), "bulk_rows": int((flag & ~tail).sum()),
            "tail_share": st, "bulk_share": sb, "enrichment_rows": _ratio(st, sb),
            "tail_share_w": wt, "bulk_share_w": wb, "enrichment_weight": _ratio(wt, wb),
            "tail_sqerr_share": float(err[tail & flag].sum() / e) if e > 0 else float("nan")}


def category_table(cat, tail, w, err, values=None) -> dict:
    """Column arrays over the values of a category (a geometry, a bin): the tail's and the
    bulk's rows and weights, the tail's squared error share and the enrichments."""
    import numpy as np

    cat, tail = np.asarray(cat), np.asarray(tail, bool)
    w, err = np.asarray(w, float), np.asarray(err, float)
    vals = np.unique(cat) if values is None else np.asarray(values)
    cols: dict[str, list] = {k: [] for k in ("value", "tail_rows", "bulk_rows", "tail_w",
                                             "bulk_w", "tail_sqerr_share", "enrichment_rows",
                                             "enrichment_weight")}
    nt, nb, wt, wb, et = tail.sum(), (~tail).sum(), w[tail].sum(), w[~tail].sum(), err[tail].sum()
    for v in vals:
        m = cat == v
        cols["value"].append(v.item() if hasattr(v, "item") else v)
        cols["tail_rows"].append(int((m & tail).sum()))
        cols["bulk_rows"].append(int((m & ~tail).sum()))
        cols["tail_w"].append(float(w[m & tail].sum()))
        cols["bulk_w"].append(float(w[m & ~tail].sum()))
        cols["tail_sqerr_share"].append(float(err[m & tail].sum() / et) if et > 0 else np.nan)
        cols["enrichment_rows"].append(_ratio((m & tail).sum() / max(nt, 1),
                                              (m & ~tail).sum() / max(nb, 1)))
        cols["enrichment_weight"].append(_ratio(w[m & tail].sum() / wt if wt > 0 else 0.0,
                                                w[m & ~tail].sum() / wb if wb > 0 else 0.0))
    return cols


def half_set(values, err) -> list:
    """The smallest set of values, in decreasing order of ``err``, holding at least half of
    its sum (amendment 1 item 2, E1.1)."""
    import numpy as np

    err = np.asarray(err, float)
    if not err.sum() > 0:
        return []
    order = np.argsort(-err, kind="mergesort")
    cum = np.cumsum(err[order]) / err.sum()
    k = int(np.argmax(cum >= 0.5)) + 1
    return [np.asarray(values)[i].item() for i in order[:k]]


# -- the Huber calibration with direction intercepts (amendment 1 item 4)


def _sides(direction, n: int) -> list[tuple[int, object]]:
    """[(side, mask)]: side 1 is sigma > 0 (``analysis.Calibrated``'s intercept index); one
    group of side 0 without a direction."""
    import numpy as np

    if direction is None:
        return [(0, np.ones(n, bool))]
    d = np.asarray(direction) > 0
    return [(s, m) for s, m in ((0, ~d), (1, d)) if m.any()]


def _wls_groups(x, y, w, groups):
    """Weighted least squares with one intercept per group and a shared slope, in
    ``analysis._wls``'s arithmetic (one group gives its bits)."""
    import numpy as np

    mx = np.zeros(len(x))
    my = np.zeros(len(x))
    cen = []
    for m in groups:
        sw = w[m].sum()
        cx, cy = w[m] @ x[m] / sw, w[m] @ y[m] / sw
        mx[m], my[m] = cx, cy
        cen.append((cx, cy))
    sxx = w @ ((x - mx) ** 2)
    b = float(w @ ((x - mx) * (y - my)) / sxx) if sxx > 0 else 0.0
    return np.array([float(cy - b * cx) for cx, cy in cen]), b


def _loc_groups(y, w, groups):
    import numpy as np

    return np.array([float(w[m] @ y[m] / w[m].sum()) for m in groups]), 0.0


def _irls_groups(x, y, w, groups, c: float, slope: bool):
    """``analysis._irls`` with one intercept per group (the scale pooled over the groups)."""
    import numpy as np

    from apexzero.costate import analysis as AN

    def fit(wt):
        return _wls_groups(x, y, wt, groups) if slope else _loc_groups(y, wt, groups)

    def per_row(a):
        A = np.zeros(len(x))
        for i, m in enumerate(groups):
            A[m] = a[i]
        return A

    a, b = fit(w)
    s = 0.0
    for it in range(1, AN.HUBER_ITERS + 1):
        r = y - per_row(a) - b * x
        s = AN.wmedian(np.abs(r - AN.wmedian(r, w)), w) / AN.MAD_NORMAL
        if not s > 0:
            return a, b, 0.0, it
        u = np.abs(r) / s
        wt = w * np.minimum(1.0, c / np.maximum(u, 1e-300))
        a1, b1 = fit(wt)
        done = (np.abs(a1 - a).sum() + abs(b1 - b)
                <= AN.HUBER_TOL * (1.0 + np.abs(a).sum() + abs(b)))
        a, b = a1, b1
        if done:
            return a, b, s, it
    return a, b, s, AN.HUBER_ITERS


def huber_groups(x, y, w=None, direction=None, c: float | None = None, nonneg: bool = True):
    """(a [2] per side (nan where a side has no row), b, scale, clamped, iterations): the
    weighted Huber fit of y on x with one intercept per turn direction and a shared slope
    (``analysis.huber_fit`` exactly with one direction), the slope clamped at >= 0 by the
    location-only fit."""
    import numpy as np

    from apexzero.costate import analysis as AN

    c = AN.HUBER_C if c is None else float(c)
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    d = None if direction is None else np.asarray(direction)
    ok = np.isfinite(x) & np.isfinite(y) & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]
    d = None if d is None else d[ok]
    a2 = np.full(2, np.nan)
    if not len(x):
        return a2, float("nan"), float("nan"), False, 0
    sides = _sides(d, len(x))
    groups = [m for _, m in sides]
    a, b, s, it = _irls_groups(x, y, w, groups, c, True)
    clamped = False
    if nonneg and b < 0.0:
        a, _, s, it2 = _irls_groups(x, y, w, groups, c, False)
        b, clamped, it = 0.0, True, it + it2
    for (side, _), v in zip(sides, a, strict=True):
        a2[side] = v
    return a2, float(b), float(s), clamped, int(it)


def calibrate_huber(g, lam, outer, w, fit_rows, valid, direction, odd, bounds,
                    n_folds: int | None = None) -> dict:
    """Out-of-fold predictions [n, m] of the Huber calibration (amendment 1 item 4): even
    components ``analysis.huber_fit``, odd ones ``huber_groups`` with direction intercepts;
    clipped to ``bounds`` [m, 2]; with the per-fold intercepts, slopes and clamps."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import states as ST

    F = ST.N_FOLDS if n_folds is None else int(n_folds)
    g, lam = np.asarray(g, float), np.asarray(lam, float)
    n, m = lam.shape
    outer, w, d = np.asarray(outer, int), np.asarray(w, float), np.asarray(direction)
    fit, val = np.asarray(fit_rows, bool), np.asarray(valid, bool)
    pred = np.full((n, m), np.nan)
    a = np.full((F, m, 2), np.nan)
    b = np.full((F, m), np.nan)
    clamped = np.zeros((F, m), bool)
    side = (d > 0).astype(int)
    for f in range(F):
        held = outer == f
        for k in range(m):
            tr = (outer != f) & fit & val[:, k] & np.isfinite(g[:, k]) & np.isfinite(lam[:, k])
            if tr.sum() < 3:
                continue
            if odd[k]:
                ak, bk, _, ck, _ = huber_groups(g[tr, k], lam[tr, k], w[tr], d[tr])
            else:
                h = AN.huber_fit(g[tr, k], lam[tr, k], w[tr], nonneg=True)
                ak, bk, ck = np.array([h.a, h.a]), h.b, h.clamped
            a[f, k], b[f, k], clamped[f, k] = ak, bk, ck
            pred[held, k] = ak[side[held]] + bk * g[held, k]
    bnd = np.asarray(bounds, float).reshape(m, 2)
    return {"pred": np.clip(pred, bnd[:, 0], bnd[:, 1]), "a": a, "b": b, "clamped": clamped}


# -- the rank calibration (amendment 1 item 4)


def rank_map(g_tr, lam_tr, w_tr, g_new):
    """Each new g's weighted mid-rank in the training g, F = (sum w [g_i < g] + 1/2 sum w
    [g_i = g]) / sum w, mapped to the training lambda's weighted quantile at F
    (``states.wquantile``)."""
    import numpy as np

    from apexzero.costate import states as ST

    g_tr, lam_tr, w_tr = (np.asarray(v, float) for v in (g_tr, lam_tr, w_tr))
    o = np.argsort(g_tr, kind="mergesort")
    gs = g_tr[o]
    cw = np.concatenate([[0.0], np.cumsum(w_tr[o])])
    g_new = np.asarray(g_new, float)
    lo = np.searchsorted(gs, g_new, side="left")
    hi = np.searchsorted(gs, g_new, side="right")
    F = (cw[lo] + 0.5 * (cw[hi] - cw[lo])) / cw[-1]
    return np.asarray(ST.wquantile(lam_tr, w_tr, F), float)


def calibrate_rank(g, lam, outer, w, fit_rows, valid, direction, bounds,
                   n_folds: int | None = None) -> dict:
    """Out-of-fold predictions [n, m] of the rank calibration (amendment 1 item 4): per fold,
    component and turn direction, ``rank_map`` on the training rows of that direction, or
    their weighted mean of lambda where those rows are fewer than 3 or their weighted
    Spearman of (g, lambda) is <= 0 or undefined; clipped to ``bounds``. With the per-fold
    Spearman [F, m, 2] and where the fallback was taken."""
    import numpy as np

    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    F = ST.N_FOLDS if n_folds is None else int(n_folds)
    g, lam = np.asarray(g, float), np.asarray(lam, float)
    n, m = lam.shape
    outer, w = np.asarray(outer, int), np.asarray(w, float)
    d = np.asarray(direction) > 0
    fit, val = np.asarray(fit_rows, bool), np.asarray(valid, bool)
    pred = np.full((n, m), np.nan)
    rho = np.full((F, m, 2), np.nan)
    fallback = np.zeros((F, m, 2), bool)
    for f in range(F):
        for k in range(m):
            for side in (0, 1):
                ds = d == bool(side)
                held = (outer == f) & ds
                tr = ((outer != f) & fit & val[:, k] & np.isfinite(g[:, k])
                      & np.isfinite(lam[:, k]) & ds)
                if not held.any() or not tr.any():
                    continue
                x, y, w0 = g[tr, k], lam[tr, k], w[tr]
                # a constant g or lambda has no Spearman: probes.wspearman would return the
                # round-off of its centred ranks, not 'undefined'
                if tr.sum() >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rho[f, k, side] = PB.wspearman(x, y, w0)
                if not rho[f, k, side] > 0:
                    fallback[f, k, side] = True
                    pred[held, k] = w0 @ y / w0.sum()
                else:
                    pred[held, k] = rank_map(x, y, w0, g[held, k])
    bnd = np.asarray(bounds, float).reshape(m, 2)
    return {"pred": np.clip(pred, bnd[:, 0], bnd[:, 1]), "spearman": rho, "fallback": fallback}


# -- the critic against the map (amendment 1 item 3)


def xi_contributions(dy0, clipped, J, sd, sigma: float, J_acc, Mx, reset):
    """[n, 5] the channel groups' contributions to g_xi (``GROUPS``), in ``critic.pullback``'s
    terms: w = dV_c/do with the clipped channels zeroed, / sd (dV/dob); a channel's term
    sigma_R w_j J[j, xi] (= sigma_R dV_c/dz_j J_z[j, xi]); the accel term sigma_R (J_acc' w)'
    M_x[:, xi] on rows with t >= 1. They sum to the pullback's g_xi."""
    import numpy as np

    w = np.where(np.asarray(clipped, bool), 0.0, np.asarray(dy0, float)) / np.asarray(sd)[None]
    cc = sigma * w * np.asarray(J)[:, :, XI]
    rest = np.ones(cc.shape[1], bool)
    rest[list(EDGE + LATERAL_XI)] = False
    acc = sigma * np.einsum("nc,nce->ne", w, np.asarray(J_acc))
    accx = np.where(np.asarray(reset, bool), 0.0, np.einsum("ne,ne->n", acc, np.asarray(Mx)[:, :, XI]))
    return np.stack([cc[:, list(EDGE_NEAR)].sum(1), cc[:, list(EDGE_FAR)].sum(1),
                     cc[:, list(LATERAL_XI)].sum(1), cc[:, rest].sum(1), accx], 1)


def map_and_critic_sizes(dy0, clipped, J, sd, sigma: float):
    """([n] ||J_z e_xi|| over the unclipped channels, [n] sigma_R ||dV_c/dz|| over the
    unclipped channels with J_z[j, xi] != 0, [n] the channels' part of g_xi)."""
    import numpy as np

    live = ~np.asarray(clipped, bool)
    jz = np.asarray(J)[:, :, XI] / np.asarray(sd)[None]
    jz = np.where(live, jz, 0.0)
    moved = live & (jz != 0.0)
    dv = np.where(moved, np.asarray(dy0, float), 0.0)
    return (np.sqrt((jz ** 2).sum(1)), sigma * np.sqrt((dv ** 2).sum(1)),
            sigma * (dv * jz).sum(1))


def group_shares(c, g, lam, w) -> dict:
    """The groups' shares on a set of rows (amendment 1 item 3): of |g| signed (sum w c_k
    sgn(g) / sum w |g|) and unsigned, the weighted shares of rows where sgn(c_k) = sgn(g) and
    where sgn(c_k) = -sgn(lambda)."""
    import numpy as np

    c, g, lam, w = (np.asarray(v, float) for v in (c, g, lam, w))
    sw = w.sum()
    sg = np.sign(g)
    return {"share_signed": (w @ (c * sg[:, None]) / (w @ np.abs(g))).tolist(),
            "share_unsigned": (w @ np.abs(c) / (w @ np.abs(c).sum(1))).tolist(),
            "sign_agree": (w @ (np.sign(c) == sg[:, None]) / sw).tolist(),
            "opposite_lambda": (w @ (np.sign(c) == -np.sign(lam)[:, None]) / sw).tolist()}


@functools.lru_cache(maxsize=1)
def _util_jit():
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle.dynamics import tyre_forces
    from apexzero.vehicle.ops import JAX

    def one(x, ctx):
        fo = tyre_forces(x, racing.params_from_context(ctx, hidden_mu=1.0), JAX)
        return jnp.stack([fo.util_f, fo.util_r])

    return jax.jit(jax.vmap(one, in_axes=(0, None)))


def tyre_region(x):
    """[n, 3] the tyre and slip channels' piecewise region (``critic._signature``'s parts for
    channels 212-215): util_f >= 2, util_r >= 2, vx <= 1, at states x [n, 8]."""
    import numpy as np

    from apexzero.fields import network as NW

    x = np.asarray(x, np.float64)
    util = np.asarray(_util_jit()(x, NW.ctx64()))
    return np.concatenate([util >= 2.0, (x[:, 3] <= 1.0)[:, None]], 1)


def tyre_kink_xi(x, step: float = TYRE_STEP):
    """[n] the region of channels 212-215 changes within +-step of xi (plan CS-1 step 2)."""
    import numpy as np

    x = np.asarray(x, np.float64)
    r0 = tyre_region(x)
    out = np.zeros(len(x), bool)
    for sgn in (-1.0, 1.0):
        xs = x.copy()
        xs[:, XI] += sgn * step
        out |= (tyre_region(xs) != r0).any(1)
    return out


def _wmedian(v, w) -> float:
    from apexzero.costate import analysis as AN

    return AN.wmedian(v, w) if len(v) else float("nan")


def _wmean(v, w) -> float:
    import numpy as np

    return float(np.asarray(w) @ np.asarray(v) / np.sum(w)) if len(v) else float("nan")


def _r2_scores(tab, comp: str, lam, preds: dict, w, cl, G: int, counts, rows, k2b, mask,
               odd: bool) -> dict:
    """{calibration: (point, draws [B])} R2 (R2_wd for an odd component) on ``rows``, after
    the row guard (plan R5), on the frozen bootstrap counts."""
    import numpy as np

    from apexzero.costate import analysis as AN

    sel = guard_rows(tab, rows, comp, k2b, mask)
    names = list(preds)
    P = [np.asarray(preds[k], float)[sel] for k in names]
    if not sel.any():
        return {k: (float("nan"), np.full(len(counts), np.nan)) for k in names}
    if not all(np.isfinite(p).all() for p in P):
        raise SystemExit(f"CS-1: a calibration left rows of lambda_{comp} without a prediction")
    pt, dr = AN.boot_r2(np.asarray(lam)[sel], P, np.asarray(w)[sel], np.asarray(cl)[sel], G,
                        counts, np.asarray(tab["direction"])[sel], odd)
    return {k: (float(pt[i]), dr[:, i]) for i, k in enumerate(names)}


def _ci_rec(pt: float, draws) -> dict:
    from apexzero.costate import analysis as AN

    lo, hi = AN._ci(draws)
    return {"r2c": pt, "ci": [lo, hi]}


def _cs1_network(run: Run, label: str, world, A: dict, frozen: dict, counts, noise) -> tuple:
    """One network's CS-1 (module section header): (its summary, its row arrays)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG
    from apexzero.fields import network as NW

    t0 = time.time()
    tab, k2b = world.tab, frozen["k2b_excluded"]
    name = AN.network_of(label)
    f = PR.net_file(name)
    pin = run.rec["networks"][name]
    agent = CS.load_agent(name, pin, run.root)
    run.note_input(pin["path"], pin["sha256"])
    rms = NW.upcast32(agent.obs_rms)
    flags = IN.flags(world, rms)
    fr = CS.c1_frame(world, frozen, flags)
    bad = mask_problems(fr.valid, name, run.decisions)
    if bad:
        raise SystemExit(f"CS-1: {'; '.join(bad)}")
    with np.load(run.read(f"post/critic/{f}.npz")) as d:
        g8, k3_pass = d["g"][:, 0].copy(), bool(d["k3_trained_pass"])
    k3 = json.loads(run.read(f"post/critic/{f}.k3.json").read_text())
    if not k3_pass or not (k3.get("decision") or {}).get("pass"):
        raise SystemExit(f"CS-1: {label} failed K3 after the tag")
    with np.load(run.read(f"post/activations/{f}.npz")) as d:
        clipped = d["clipped"].copy()
    sd = np.sqrt(np.asarray(rms.var, np.float64) + CR.RMS_EPS)
    z = (A["ob"] - np.asarray(rms.mean, np.float64)) / sd
    if not np.array_equal(clipped, np.abs(z) >= CR.OBS_CLIP):
        raise SystemExit(f"CS-1: {label}'s stored clip flags are not its obs_rms's")
    direction = np.asarray(tab["direction"])
    w_in, outer, cl, G = fr.w_in, fr.fo.outer, fr.cl, fr.G
    g = g8[:, fr.idx]

    # -- the frozen C1 first, against decisions.json (plan Section 3 'Reuse')
    c1 = AN.c1(g, fr.lam, outer, w_in, cl, G, direction, fr.inner, None, fr.valid, TG.K_P,
               counts, slope_boot=False, bounds=fr.bnd)
    want = run.decisions["networks"][name]["C1"]
    ks = [TG.K_P.index(c) for c in CS1_COMPONENTS]
    got = {"r2c": [c1.r2c[k] for k in ks], "stat": [c1.stat], "ci": list(c1.ci),
           "a": c1.calibrated.a[:, ks].ravel(), "b": c1.calibrated.b[:, ks].ravel()}
    ref = {"r2c": [want["r2c"][k] for k in ks], "stat": [want["stat"]], "ci": want["ci"],
           "a": np.asarray(want["calibration"]["a"], float)[:, ks].ravel(),
           "b": np.asarray(want["calibration"]["b"], float)[:, ks].ravel()}
    wiring = {k: float(np.max(np.abs(np.asarray(got[k], float) - np.asarray(ref[k], float))))
              for k in got}
    if not all(v <= WIRING_TOL for v in wiring.values()):
        raise SystemExit(f"CS-1: {label}'s frozen C1 does not reproduce decisions.json: {wiring}")

    rows = {c: guard_rows(tab, fr.inner & fr.valid[:, k], c, k2b, fr.valid[:, k])
            for c, k in zip(CS1_COMPONENTS, ks, strict=True)}
    rx = rows["xi"]
    ix = np.flatnonzero(rx)
    gx, lx, wx, dx = g[ix, XI], fr.lam[ix, XI], w_in[ix], direction[ix]
    tail = xi_tail(gx, wx, dx)
    pred = {"frozen": c1.calibrated.pred[:, ks]}
    px = pred["frozen"][ix, 1]
    err = wx * (px - lx) ** 2

    # -- step 1: the findings' table
    def r2wd(sel):
        return float(AN.boot_r2(lx[sel], [px[sel]], wx[sel], cl[ix][sel], G, np.ones((1, G)),
                                dx[sel], True)[0][0])

    geo = ST.geometry_index(tab)[ix]
    table = {"slope_fold_mean": float(np.mean(c1.calibrated.b[:, XI])),
             "tail_rows": int(tail.sum()), "tail_weight_share": float(wx[tail].sum() / wx.sum()),
             "tail_sqerr_share": float(err[tail].sum() / err.sum()),
             "bulk_r2c_xi": r2wd(~tail),
             "opposite_tail": float(np.mean(np.sign(gx[tail]) == -np.sign(lx[tail]))),
             "opposite_bulk": float(np.mean(np.sign(gx[~tail]) == -np.sign(lx[~tail]))),
             "var_ratio": float(ST.wvar(px, wx, dx) / ST.wvar(lx, wx, dx)),
             "geometry38_sqerr_share": float(err[geo == CS1_GEOMETRY].sum() / err.sum())}
    printed = dict(zip([k for k, _ in CS1_FINDINGS_FIELDS], CS1_FINDINGS[label], strict=True))
    diff = {}
    for k, dec in CS1_FINDINGS_FIELDS:
        tol = 0.0 if dec == 0 else 0.5 * 10.0 ** -dec + 1e-12
        diff[k] = {"computed": table[k], "printed": printed[k],
                   "agrees": bool(abs(table[k] - printed[k]) <= tol)}
    if int(tail.sum()) != printed["tail_rows"]:
        raise SystemExit(f"CS-1: {label}'s tail has {int(tail.sum())} rows, the findings "
                         f"{printed['tail_rows']} (plan CS-1 step 1)")
    off = [k for k, _ in CS1_FINDINGS_STOP if not diff[k]["agrees"]]
    if off:
        raise SystemExit(f"CS-1: {label} does not reproduce the findings' table: "
                         + "; ".join(f"{k} {diff[k]['computed']:.6g} against "
                                     f"{diff[k]['printed']}" for k in off))

    # -- step 2: where the tail sits
    ob = A["ob"][ix]
    vx = np.asarray(tab["vx"], float)[ix]
    xs = A["x"][ix]
    cx = clipped[ix]
    in_set = np.zeros(cx.shape[1], bool)
    in_set[list(XI_SET)] = True
    region0 = tyre_region(xs)
    ob_region = np.concatenate([ob[:, list(TYRE_UTIL)] >= 1.0, (xs[:, 3] <= 1.0)[:, None]], 1)
    fl = {"clip_any": cx.any(1), "clip_xi_channels": cx[:, in_set].any(1),
          "clip_other_channels": cx[:, ~in_set].any(1),
          "kappa_clipped": np.asarray(flags["kappa_clipped"], bool)[ix]}
    for k, v in KINK_VX.items():
        fl[k] = np.abs(vx - v) < KINK_BAND
    fl["kink_vx_either"] = fl["kink_vx25"] | fl["kink_vx83"]
    fl["tyre_kink_xi"] = tyre_kink_xi(xs)
    fl["tyre_clip_region"] = ob_region.any(1)
    util_max = ob[:, list(TYRE_UTIL)].max(1) + 1.0
    bins = np.floor(np.asarray(tab["ds_apex"], float)[ix] / PHASE_BIN).astype(int)
    geo_tab = category_table(geo, tail, wx, err, values=np.arange(G))
    bin_tab = category_table(bins, tail, wx, err)
    hs = half_set(geo_tab["value"], geo_tab["tail_sqerr_share"])
    labels = AN.geometry_labels(tab)
    top_bins = np.argsort(-np.nan_to_num(np.asarray(bin_tab["tail_sqerr_share"])),
                          kind="mergesort")[:5]
    where = {
        "flags": {k: flag_table(v, tail, wx, err) for k, v in fl.items()},
        "tyre_region_agrees_with_observation": bool(np.array_equal(region0, ob_region)),
        "vx": {"median_tail": _wmedian(vx[tail], wx[tail]),
               "median_bulk": _wmedian(vx[~tail], wx[~tail]),
               "mean_tail": _wmean(vx[tail], wx[tail]), "mean_bulk": _wmean(vx[~tail], wx[~tail])},
        "util_max": {"median_tail": _wmedian(util_max[tail], wx[tail]),
                     "median_bulk": _wmedian(util_max[~tail], wx[~tail]),
                     "mean_tail": _wmean(util_max[tail], wx[tail]),
                     "mean_bulk": _wmean(util_max[~tail], wx[~tail])},
        "ds_apex": {"median_tail": _wmedian(np.asarray(tab["ds_apex"])[ix][tail], wx[tail]),
                    "median_bulk": _wmedian(np.asarray(tab["ds_apex"])[ix][~tail], wx[~tail])},
        "half_set": [{"geometry": int(v), "R": float(labels[v][0]), "theta": float(labels[v][1]),
                      "tail_sqerr_share": geo_tab["tail_sqerr_share"][v],
                      "tail_rows": geo_tab["tail_rows"][v], "bulk_rows": geo_tab["bulk_rows"][v],
                      "enrichment_rows": geo_tab["enrichment_rows"][v]} for v in hs],
        "half_set_size": len(hs), "geometry38_in_half_set": CS1_GEOMETRY in hs,
        "geometry38": {k: geo_tab[k][CS1_GEOMETRY] for k in geo_tab if k != "value"},
        "geometries_with_tail_rows": int(np.sum(np.asarray(geo_tab["tail_rows"]) > 0)),
        "top_phase_bins": [{"bin_m": int(bin_tab["value"][i]) * PHASE_BIN,
                            **{k: bin_tab[k][i] for k in bin_tab if k != "value"}}
                           for i in top_bins]}

    # -- step 3: the critic against the map, by a fresh VJP at the xi decision rows (plan R4)
    net = CR.trained_critic(agent, pin["sha256"])
    inp = CR.Inputs(*(A[f_][ix] for f_ in CR.Inputs._fields))
    oj = CR.ObsJac(**{f_: A[f_][ix] for f_ in IN.OBSJAC_FIELDS})
    pb = CR.pullback(net, None, inp, oj)
    zz, cl_x, sd_x = oj.z(net.obs_rms)
    gradf = CR._grad_jit(net.fn)
    _, dy = CR._batched(lambda o_: gradf(net.params, o_), len(zz),
                        np.clip(zz, -CR.OBS_CLIP, CR.OBS_CLIP))
    dy0 = dy[:, 0, :]
    stored = g8[ix]
    with np.errstate(invalid="ignore", divide="ignore"):
        pb_rel = float(np.nanmax(np.abs(pb.g[:, 0] - stored)
                                 / np.maximum(np.abs(stored), 1e-300)))
    contrib = xi_contributions(dy0, cl_x, oj.J, sd_x, net.sigma, oj.J_acc, oj.Mx, oj.reset)
    denom = np.maximum(np.abs(stored[:, XI]), SUM_FLOOR * np.abs(contrib).sum(1))
    sum_rel = float(np.max(np.abs(contrib.sum(1) - stored[:, XI]) / denom))
    if not (pb_rel <= SUM_TOL and sum_rel <= SUM_TOL):
        raise SystemExit(f"CS-1: {label}'s fresh VJP does not reproduce the stored g "
                         f"(pullback {pb_rel:.3g}, groups {sum_rel:.3g}; plan CS-1 step 3)")
    map_size, critic_size, chan = map_and_critic_sizes(dy0, cl_x, oj.J, sd_x, net.sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = chan / (critic_size * map_size)
    jz_any = np.flatnonzero((np.abs(oj.J[:, :, XI]) > 0).any(0))

    def sizes(sel):
        return {"map_median": _wmedian(map_size[sel], wx[sel]),
                "map_mean": _wmean(map_size[sel], wx[sel]),
                "critic_median": _wmedian(critic_size[sel], wx[sel]),
                "critic_mean": _wmean(critic_size[sel], wx[sel]),
                "cosine_median": _wmedian(cosine[sel], wx[sel]),
                "abs_g_median": _wmedian(np.abs(gx[sel]), wx[sel])}

    cvm = {"groups": list(GROUPS),
           "tail": {**group_shares(contrib[tail], gx[tail], lx[tail], wx[tail]), **sizes(tail)},
           "bulk": {**group_shares(contrib[~tail], gx[~tail], lx[~tail], wx[~tail]),
                    **sizes(~tail)},
           "check": {"pullback_max_rel": pb_rel, "groups_max_rel": sum_rel,
                     "accel_term_max_abs": float(np.max(np.abs(contrib[:, -1]))),
                     "other_channels_max_abs": float(np.max(np.abs(contrib[:, 3])))},
           "channels_xi_moves": jz_any.tolist(), "sigma_R": float(net.sigma)}
    cvm["ratio"] = {k: _ratio(cvm["tail"][f"{k}_median"], cvm["bulk"][f"{k}_median"])
                    for k in ("map", "critic")}
    cvm["ratio_mean"] = {k: _ratio(cvm["tail"][f"{k}_mean"], cvm["bulk"][f"{k}_mean"])
                         for k in ("map", "critic")}

    # -- step 4: the other calibrations, after the plants
    lam2, bnd2, val2 = fr.lam[:, ks], fr.bnd[ks], fr.valid[:, ks]
    odd2 = np.array([False, True])

    def all_calibrations(gg):
        val = val2 & np.isfinite(lam2) & np.isfinite(gg)
        fz = AN.calibrate(gg, lam2, outer, w_in, fr.inner, val, direction=direction, odd=odd2,
                          bounds=bnd2).pred
        hu = calibrate_huber(gg, lam2, outer, w_in, fr.inner, val, direction, odd2, bnd2)
        rk = calibrate_rank(gg, lam2, outer, w_in, fr.inner, val, direction, bnd2)
        return {"frozen": fz, "huber": hu, "rank": rk}

    cal = all_calibrations(g[:, ks])
    if not np.array_equal(cal["frozen"], pred["frozen"], equal_nan=True):
        raise SystemExit(f"CS-1: {label}: the frozen calibration on (s, xi) is not C1's")
    pred.update(huber=cal["huber"]["pred"], rank=cal["rank"]["pred"])
    one = np.ones((1, G))
    plants = {"zero": np.zeros((len(g), 2)), "noise": noise[:, ks], "negated": -lam2,
              "direction_only": np.repeat(direction.astype(float)[:, None], 2, 1)}
    plant_r2 = {c: {} for c in CALIBRATIONS}
    for pname, pg in plants.items():
        pc = all_calibrations(pg)
        pp = {"frozen": pc["frozen"], "huber": pc["huber"]["pred"], "rank": pc["rank"]["pred"]}
        for j, comp in enumerate(CS1_COMPONENTS):
            sc = _r2_scores(tab, comp, lam2[:, j], {k: v[:, j] for k, v in pp.items()}, w_in,
                            cl, G, one, rows[comp], k2b, val2[:, j], bool(odd2[j]))
            for k in CALIBRATIONS:
                plant_r2[k].setdefault(pname, {})[comp] = sc[k][0]
    from apexzero.costate import gates as GT

    lo_f, hi_f = GT.K4_FLOOR
    tail_full = np.zeros(len(g), bool)
    tail_full[ix] = tail
    sets = {"all": None, "bulk": ~tail_full, "tail": tail_full}
    cals = {}
    for k in CALIBRATIONS:
        ok = all(lo_f <= v <= hi_f for p in plant_r2[k].values() for v in p.values())
        cals[k] = {"plants": plant_r2[k], "plants_pass": bool(ok)}
    for sname, extra in sets.items():
        per = {}
        for j, comp in enumerate(CS1_COMPONENTS):
            r = rows[comp] if extra is None else rows[comp] & extra
            per[comp] = _r2_scores(tab, comp, lam2[:, j], {k: v[:, j] for k, v in pred.items()},
                                   w_in, cl, G, counts, r, k2b, val2[:, j], bool(odd2[j]))
            for k in CALIBRATIONS:
                cals[k].setdefault(sname, {})[comp] = {**_ci_rec(*per[comp][k]),
                                                       "rows": int(r.sum())}
        for k in CALIBRATIONS:
            pts = [per[c][k][0] for c in CS1_COMPONENTS]
            dr = np.mean([per[c][k][1] for c in CS1_COMPONENTS], axis=0)
            cals[k][sname]["mean"] = _ci_rec(float(np.mean(pts)), dr)
    if abs(cals["frozen"]["all"]["mean"]["r2c"] - c1.stat) > WIRING_TOL:
        raise SystemExit(f"CS-1: {label}'s rescored frozen R2_c is not C1's")
    cals["huber"]["fold_b"] = cal["huber"]["b"].tolist()
    cals["huber"]["fold_clamped"] = cal["huber"]["clamped"].tolist()
    cals["rank"]["fold_spearman"] = cal["rank"]["spearman"].tolist()
    cals["rank"]["fold_fallback"] = cal["rank"]["fallback"].tolist()

    summary = {"run": name, "decision_rows": {c: int(r.sum()) for c, r in rows.items()},
               "wiring": {"c1_max_abs_diff": wiring, "findings_table": diff},
               "tail": table, "where": where, "critic_vs_map": cvm, "calibrations": cals,
               "seconds": time.time() - t0}
    arrays = {"rows_xi": ix, "rows_s": np.flatnonzero(rows["s"]), "tail": tail,
              "tail_s": tail_full[rows["s"]], "g_xi": gx, "lam_xi": lx, "w_xi": wx,
              "contributions": contrib, "map_size": map_size, "critic_size": critic_size,
              "cosine": cosine, "geometry": geo, "ds_apex_bin": bins, "vx": vx,
              "util_max": util_max, **{f"flag/{k}": v for k, v in fl.items()},
              **{f"geometry_table/{k}": np.asarray(v, float) for k, v in geo_tab.items()},
              **{f"bin_table/{k}": np.asarray(v, float) for k, v in bin_tab.items()}}
    for k in CALIBRATIONS:
        for j, comp in enumerate(CS1_COMPONENTS):
            arrays[f"pred_{k}_{comp}"] = pred[k][rows[comp], j]
    print(f"CS-1 {label}: tail {int(tail.sum())}; R2_c xi frozen "
          f"{cals['frozen']['all']['xi']['r2c']:.3f}, huber {cals['huber']['all']['xi']['r2c']:.3f}"
          f", rank {cals['rank']['all']['xi']['r2c']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return summary, arrays


def _count(nets: dict, fn) -> dict:
    per = {k: bool(fn(v)) for k, v in nets.items()}
    return {"per_network": per, "count": int(sum(per.values())), "of": len(per)}


def _not_enriched(rec: dict) -> bool:
    """A flag is not enriched when its row enrichment is below 1.5, or undefined because no
    row carries it (amendment 1 item 5)."""
    import numpy as np

    e = rec["enrichment_rows"]
    return bool(np.isnan(e) or e < 1.5)


def cs1_expected(nets: dict) -> dict:
    """E1.1-E1.4 read as amendment 1 item 5 reads them, from the per-network summaries."""

    def holds(*parts, need=E_COUNT):
        return bool(all(p["count"] >= min(need, p["of"]) for p in parts))

    e11a = _count(nets, lambda v: v["where"]["half_set_size"] <= 6)
    e11b = _count(nets, lambda v: v["where"]["geometry38_in_half_set"])
    e11 = _count(nets, lambda v: v["where"]["half_set_size"] <= 6
                 and v["where"]["geometry38_in_half_set"])

    def fl(v, k):
        return v["where"]["flags"][k]

    e12 = {"higher_vx": _count(nets, lambda v: v["where"]["vx"]["median_tail"]
                               > v["where"]["vx"]["median_bulk"]),
           "kink_vx83_enriched": _count(nets, lambda v: fl(v, "kink_vx83")["enrichment_rows"]
                                        > 1.5)}
    for k in ("clip_any", "clip_xi_channels", "clip_other_channels", "kappa_clipped",
              "tyre_kink_xi"):
        e12[f"{k}_not_enriched"] = _count(nets, lambda v, k=k: _not_enriched(fl(v, k)))

    def cvm(v):
        return v["critic_vs_map"]

    far = GROUPS.index("far_edge")
    e13 = {"far_over_half_tail": _count(nets, lambda v: cvm(v)["tail"]["share_signed"][far] > 0.5),
           "far_under_half_bulk": _count(nets, lambda v: cvm(v)["bulk"]["share_signed"][far]
                                         < 0.5),
           "critic_ratio_over_2": _count(nets, lambda v: cvm(v)["ratio"]["critic"] > 2.0),
           "map_ratio_under_2": _count(nets, lambda v: cvm(v)["ratio"]["map"] < 2.0)}

    def e14_of(k: str) -> dict:
        def cal(v):
            return v["calibrations"][k]["all"]

        def s_move(v):
            return abs(cal(v)["s"]["r2c"] - v["calibrations"]["frozen"]["all"]["s"]["r2c"])

        out = {"xi_at_least_-0.05": _count(nets, lambda v: cal(v)["xi"]["r2c"] >= -0.05),
               "xi_below_0.2": _count(nets, lambda v: cal(v)["xi"]["r2c"] < 0.2),
               "xi_ci_upper_below_0.2": _count(nets, lambda v: cal(v)["xi"]["ci"][1] < 0.2),
               "s_moves_under_0.03": _count(nets, lambda v: s_move(v) < 0.03)}
        out["holds"] = holds(out["xi_at_least_-0.05"], out["xi_below_0.2"],
                             out["s_moves_under_0.03"], need=len(nets))
        return out

    e14 = {k: e14_of(k) for k in ("huber", "rank")}
    return {
        "E1.1": {"half_set_at_most_6": e11a, "geometry38_in_half_set": e11b, "both": e11,
                 "holds": holds(e11)},
        "E1.2": {**e12, "holds": holds(*e12.values()),
                 "note": "tyre_kink_xi cannot be set (channels 212-215 do not read xi): "
                         "'not enriched' holds by construction (amendment 1 item 2)"},
        "E1.3": {**e13, "holds": holds(*e13.values())},
        "E1.4": {**e14, "holds": bool(all(v["holds"] for v in e14.values()))},
        "reading": "amendment 1 item 5: a condition without a count holds in at least 6 of 9; "
                   "E1.4 in all 9, on point estimates"}


def cs1_rule(nets: dict) -> dict:
    """The rule for later items (plan CS-1; amendment 1 item 6)."""
    out = {}
    for k in ("huber", "rank"):
        def fires(v, k=k):
            c = v["calibrations"]
            return (c[k]["plants_pass"]
                    and c[k]["all"]["xi"]["r2c"] - c["frozen"]["all"]["xi"]["r2c"] > RULE_GAIN)

        cnt = _count(nets, fires)
        out[k] = {**cnt, "fires": cnt["count"] >= RULE_NETWORKS}
    out["co_report"] = [k for k in ("huber", "rank") if out[k]["fires"]]
    out["rule"] = ("in at least 5 of the 9 networks the calibration passes its plants and "
                   "raises xi's R2_c over the frozen one by more than 0.1: later items "
                   "co-report it beside the frozen one, which stays the headline")
    return out


@computes("CS-1")
def cs1(run: Run) -> None:
    """CS-1 (plan Section 4, amendment 1): per decision network ``_cs1_network``, then the
    expected directions read (``cs1_expected``) and the rule for later items (``cs1_rule``);
    outputs summary.json and rows.npz."""
    from types import SimpleNamespace

    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT
    from apexzero.costate import inputs as IN
    from apexzero.costate import probes as PB
    from apexzero.costate import targets as TG

    CS._x64()
    run.read("inputs/world.npz")
    world = CS._load_world_frozen(SimpleNamespace(out=str(run.costate)), run.rec)
    frozen = run.rec["frozen"]
    n = world.n
    # every key's Inputs and ObsJac fields as T-length arrays (one name space: 'reset' is in
    # both and must agree)
    A: dict = {}
    covered = np.zeros(n, bool)
    for k in world.keys:
        if not np.array_equal(np.asarray(k.inp.reset), np.asarray(k.oj.reset)):
            raise SystemExit(f"CS-1: key {k.key_id}'s inputs and ObsJac disagree on reset rows")
        covered[k.rows] = True
        for src, names in ((k.inp, CR.Inputs._fields), (k.oj, IN.OBSJAC_FIELDS)):
            for f in names:
                v = np.asarray(getattr(src, f))
                A.setdefault(f, np.zeros((n, *v.shape[1:]), v.dtype))[k.rows] = v
    if not covered.all():
        raise SystemExit("CS-1: T has rows without network inputs")
    labels = AN.geometry_labels(world.tab)
    if (labels[CS1_GEOMETRY].tolist() != frozen["geometries"][CS1_GEOMETRY]
            or labels[CS1_GEOMETRY].tolist() != [44.7, 180.0]):
        raise SystemExit("CS-1: geometry 38 is not (44.7 m, 180 deg)")
    G = AN.geometry_clusters(world.tab)[1]
    counts = PB.cluster_counts(G)
    rng = GT.k4_rng(7)                    # gates.pr5's stream: its second draw is 'noise'
    rng.standard_normal((n, len(TG.K_P)))
    noise = rng.standard_normal((n, len(TG.K_P)))
    want = getattr(run.args, "network", None)
    nets = [lb for lb in CS1_ORDER if want is None or want in (lb, AN.network_of(lb))]
    if not nets:
        raise SystemExit(f"CS-1: --network {want} is not a decision network")
    res, arrays = {}, {}
    for lb in nets:
        res[lb], arr = _cs1_network(run, lb, world, A, frozen, counts, noise)
        arrays.update({f"{lb}/{k}": v for k, v in arr.items()})
    out = {"item": "CS-1", "networks": res, "expected": cs1_expected(res),
           "rule_for_later_items": cs1_rule(res),
           "definitions": "plan Section 4 CS-1 and amendment 1",
           "groups": list(GROUPS), "components": list(CS1_COMPONENTS),
           "calibrations": list(CALIBRATIONS), "plants": list(PLANTS),
           "plant_interval": list(GT.K4_FLOOR), "n_boot": len(counts)}
    run.write_npz("rows", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-2 to CS-5: shared
#
# Plan Section 4 CS-2 to CS-5 with amendment 2 (their operational details). Each network's
# checkpoint is loaded through its pin for its obs_rms and PPOConfig only (no trained network
# is evaluated, so these items do not unlock); its frame and decision mask are rebuilt and its
# frozen C1 recomputed and checked against decisions.json before anything else (amendment 2
# item 1).

SX = ("s", "xi")                         # C1's components with decision rows
K_SX = (0, 2)                            # their columns in K_P (and in the 8 state components)
DESCRIPTIVE_SPECIALISTS = ("S1", "S2", "S3")


def load_world(run: Run):
    """T's world (``scripts/costate.py``'s ``_load_world_frozen``: the bound world.npz, T's
    (key, t) list, geometries and folds against the record), read through ``run``."""
    from types import SimpleNamespace

    CS._x64()
    run.read("inputs/world.npz")
    return CS._load_world_frozen(SimpleNamespace(out=str(run.costate)), run.rec)


def network_frame(run: Run, name: str, world) -> tuple:
    """(agent, flags, C1's frame) of one network: its checkpoint through its pin (recorded),
    its flags under its own obs_rms, ``c1_frame`` and its decision mask, checked against
    decisions.json where it has one (amendment 2 item 1)."""
    from apexzero.costate import inputs as IN
    from apexzero.fields import network as NW

    pin = run.rec["networks"][name]
    agent = CS.load_agent(name, pin, run.root)
    run.note_input(pin["path"], pin["sha256"])
    flags = IN.flags(world, NW.upcast32(agent.obs_rms))
    fr = CS.c1_frame(world, run.rec["frozen"], flags)
    bad = mask_problems(fr.valid, name, run.decisions)
    if bad:
        raise SystemExit(f"{run.item.id}: {'; '.join(bad)}")
    return agent, flags, fr


def read_critic(run: Run, name: str, fields=("g", "g_held", "accel_sens", "dV_dwin")) -> dict:
    """A network's bound critic arrays, after its K3 check after the tag passed."""
    import numpy as np

    from apexzero.costate import prereg as PR

    f = PR.net_file(name)
    with np.load(run.read(f"post/critic/{f}.npz")) as d:
        out = {k: d[k].copy() for k in fields if k in d.files}
        ok = bool(d["k3_trained_pass"])
    k3 = json.loads(run.read(f"post/critic/{f}.k3.json").read_text())
    if not ok or not (k3.get("decision") or {}).get("pass"):
        raise SystemExit(f"{run.item.id}: {name} failed K3 after the tag")
    return out


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


def checked_c1(run: Run, name: str, fr, g8, counts):
    """The frozen C1 of a network, recomputed and required to reproduce decisions.json to
    ``WIRING_TOL`` (amendment 2 item 1): (C1, the differences)."""
    c1 = frozen_c1(fr, g8, counts)
    wiring = c1_wiring(c1, c1_reference(run.decisions, name)["C1"])
    if not all(v <= WIRING_TOL for v in wiring.values()):
        raise SystemExit(f"{run.item.id}: {name}'s frozen C1 does not reproduce decisions.json: "
                         f"{wiring}")
    return c1, wiring


def c1_rows(fr, k2b: dict) -> dict:
    """{component: [n] C1's rows} of s and xi (the interior within the decision mask), through
    the row guard (plan R5)."""
    return {c: guard_rows(fr.tab, fr.inner & fr.valid[:, k], c, k2b, fr.valid[:, k])
            for c, k in zip(SX, K_SX, strict=True)}


def r2_ci(y, preds, w, cl, G: int, counts, direction=None, odd: bool = False) -> list[dict]:
    """[{'r2', 'ci'}] R2 (R2_wd when ``odd``) of each prediction vector on the given rows, with
    the cluster bootstrap's 95% CI (``analysis.boot_r2``)."""
    from apexzero.costate import analysis as AN

    pt, dr = AN.boot_r2(y, list(preds), w, cl, G, counts, direction, odd)
    return [{"r2": float(pt[i]), "ci": list(AN._ci(dr[:, i]))} for i in range(len(pt))]


def network_label(name: str) -> str:
    """A decision network's registry label (G1, ...), else the name itself."""
    from apexzero.costate import analysis as AN

    for lb in CS1_ORDER:
        if AN.network_of(lb) == name:
            return lb
    return name


def all_networks(rec: dict) -> list[str]:
    """The record's networks: the 31 generalists in ``analysis.GENERALISTS``' order, then the
    specialists (amendment 2 item 1)."""
    from apexzero.costate import analysis as AN

    pins = list(rec["networks"])
    gen = [n for n in AN.GENERALISTS if n in pins]
    return gen + [n for n in pins if n not in gen]


def _count_of(per: dict, fn) -> dict:
    """{per_network, count, of} of a condition over a {name: record} map (NaN-safe)."""
    out = {}
    for k, v in per.items():
        try:
            out[k] = bool(fn(v))
        except (TypeError, KeyError, ValueError):
            out[k] = False
    return {"per_network": out, "count": int(sum(out.values())), "of": len(out)}


# ------------------------------------------------------------------------- CS-2: descriptives
#
# Plan Section 4 CS-2 with amendment 2 item 2, on all 34 networks (heads: the 31
# generalists): unwinsorised R2, raw R2 and the direction-only floor, per-head R2_c, the heads'
# gradient reliability rho_k, the spread over folds, C1 on D, and the accel term.

UTIL_BINS = (0.0, 0.25, 0.5, 0.75, 1.0, 2.0)   # util_max bins, the last closed at 2
RHO_E21 = 0.9                            # E2.1
E22_BAND, E22_AGREE = 0.2, 0.1           # E2.2: CI upper end, agreement with the main V
E23_S, E24_FLOOR, E26_SHARE = 0.05, 0.1, 0.1


def direction_floor(lam, outer, w, fit, direction, n_folds: int | None = None):
    """[n] the direction-only floor (amendment 2 item 2): per outer fold and turn direction,
    the w-weighted mean of lam over the training folds' ``fit`` rows, at the held-out fold's
    rows of that direction (NaN where there is none)."""
    import numpy as np

    from apexzero.costate import states as ST

    F = ST.N_FOLDS if n_folds is None else int(n_folds)
    lam, w = np.asarray(lam, float), np.asarray(w, float)
    outer, fit, d = np.asarray(outer, int), np.asarray(fit, bool), np.asarray(direction) > 0
    pred = np.full(len(lam), np.nan)
    for f in range(F):
        for side in (False, True):
            tr = (outer != f) & fit & (d == side) & np.isfinite(lam)
            held = (outer == f) & (d == side)
            if tr.any() and held.any():
                pred[held] = w[tr] @ lam[tr] / w[tr].sum()
    return pred


def per_fold_r2(y, pred, w, cl, G: int, outer, direction=None, odd: bool = False,
                n_folds: int | None = None):
    """[F] R2 (R2_wd when ``odd``) on each outer fold's rows (NaN where a fold has under 2)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import states as ST

    F = ST.N_FOLDS if n_folds is None else int(n_folds)
    y, pred, w, outer = (np.asarray(v) for v in (y, pred, w, outer))
    d = None if direction is None else np.asarray(direction)
    out = np.full(F, np.nan)
    one = np.ones((1, G))
    for f in range(F):
        m = outer == f
        if m.sum() >= 2:
            out[f] = AN.boot_r2(y[m], [pred[m]], w[m], np.asarray(cl)[m], G, one,
                                None if d is None else d[m], odd)[0][0]
    return out


def wcorr(x, y, w, direction=None) -> float:
    """Weighted Pearson correlation; with ``direction``, the direction-weight-weighted mean of
    the two directions' own (R2_wd's convention, amendment 2 item 2)."""
    import numpy as np

    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    groups = ([np.ones(len(x), bool)] if direction is None
              else [np.asarray(direction) > 0, np.asarray(direction) <= 0])
    num = den = 0.0
    for m in groups:
        if m.sum() < 2:
            continue
        ww = w[m]
        xc, yc = x[m] - ww @ x[m] / ww.sum(), y[m] - ww @ y[m] / ww.sum()
        sxy, sxx, syy = ww @ (xc * yc), ww @ (xc * xc), ww @ (yc * yc)
        if sxx > 0 and syy > 0:
            num += ww.sum() * sxy / np.sqrt(sxx * syy)
            den += ww.sum()
    return float(num / den) if den > 0 else float("nan")


def reliability_within(q, w, direction) -> float:
    """``critic.head_reliability`` within each turn direction, their direction-weight-weighted
    mean (amendment 2 item 2: xi's companion)."""
    import numpy as np

    from apexzero.costate import critic as CR

    q, w = np.asarray(q, float), np.asarray(w, float)
    num = den = 0.0
    for m in (np.asarray(direction) > 0, np.asarray(direction) <= 0):
        if m.sum() < 2:
            continue
        num += w[m].sum() * CR.head_reliability(q[m], w[m])
        den += w[m].sum()
    return float(num / den) if den > 0 else float("nan")


def accel_share(g_vx, g_held_vx):
    """[n] the accel term's share of g_vx per state, |g - g_held| / (|g_held| + |g - g_held|),
    0 where both are 0 (``critic.Pullback.accel_share``'s formula)."""
    import numpy as np

    t = np.abs(np.asarray(g_vx, float) - np.asarray(g_held_vx, float))
    d = np.abs(np.asarray(g_held_vx, float)) + t
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 0, t / d, 0.0)


def accel_record(g_vx, g_held_vx, sens, util, rows, w) -> dict:
    """The accel term's share of g_vx and the accel-channel sensitivity against util_max on
    one row set under its weights ``w`` (amendment 2 item 2, step 7)."""
    import numpy as np

    from apexzero.costate import probes as PB

    rows = np.asarray(rows, bool)
    if not rows.any():
        return {"rows": 0}
    share = accel_share(g_vx, g_held_vx)[rows]
    t = np.abs(np.asarray(g_vx, float) - np.asarray(g_held_vx, float))[rows]
    d = np.abs(np.asarray(g_held_vx, float))[rows] + t
    ww = np.asarray(w, float)[rows]
    s, u = np.asarray(sens, float)[rows], np.asarray(util, float)[rows]
    bins = []
    for i, (lo, hi) in enumerate(itertools.pairwise(UTIL_BINS)):
        m = (u >= lo) & ((u < hi) if i < len(UTIL_BINS) - 2 else (u <= hi))
        bins.append({"bin": [lo, hi], "rows": int(m.sum()),
                     "weight": float(ww[m].sum() / ww.sum()),
                     "sens_median": _wmedian(s[m], ww[m])})
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = PB.wspearman(s, u, ww) if np.ptp(s) > 0 and np.ptp(u) > 0 else float("nan")
    return {"rows": int(rows.sum()), "share_mean": _wmean(share, ww),
            "share_median": _wmedian(share, ww),
            "share_aggregate": float(ww @ t / (ww @ d)) if ww @ d > 0 else float("nan"),
            "sens_median": _wmedian(s, ww), "sens_mean": _wmean(s, ww),
            "sens_util_spearman": float(rho), "util_bins": bins}


def _cs2_network(run: Run, name: str, world, util, counts) -> dict:
    """One network's CS-2 (the section header; amendment 2 item 2)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    t0 = time.time()
    _, _, fr = network_frame(run, name, world)
    crit = read_critic(run, name, ("g", "g_held", "accel_sens"))
    g = crit["g"]
    c1, wiring = checked_c1(run, name, fr, g[:, 0], counts)
    tab, k2b = fr.tab, run.rec["frozen"]["k2b_excluded"]
    direction = np.asarray(tab["direction"])
    outer, cl, G, w_in = fr.fo.outer, fr.cl, fr.G, fr.w_in
    rows = c1_rows(fr, k2b)
    out = {"run": name, "label": network_label(name), "wiring": {"c1_max_abs_diff": wiring},
           "decision_rows": {c: int(r.sum()) for c, r in rows.items()},
           "frozen": {"r2c": {c: float(c1.r2c[k]) for c, k in zip(SX, K_SX, strict=True)},
                      "stat": float(c1.stat), "ci": list(c1.ci)}}
    # steps 1, 2 and 5: unwinsorised R2, raw R2 and the floor, the spread over folds
    comps = {}
    for c, k in zip(SX, K_SX, strict=True):
        odd = c == "xi"
        p = c1.calibrated.pred[:, k]
        sel = rows[c] & np.isfinite(p)
        if sel.sum() != rows[c].sum():
            raise SystemExit(f"CS-2: {name}: rows of lambda_{c} without a calibration")
        y, y_raw = fr.lam[:, k], np.asarray(tab[f"lam_{c}"], float)
        floor = direction_floor(y, outer, w_in, rows[c], direction)
        if not np.isfinite(floor[sel]).all():
            raise SystemExit(f"CS-2: {name}: rows of lambda_{c} without a direction-only floor")
        a = (w_in[sel], cl[sel], G, counts)
        frozen_r2 = r2_ci(y[sel], [p[sel]], *a, direction[sel], odd)[0]
        if abs(frozen_r2["r2"] - c1.r2c[k]) > WIRING_TOL:
            raise SystemExit(f"CS-2: {name}: the rescored R2_c of lambda_{c} is not C1's")
        raw = r2_ci(y[sel], [p[sel], floor[sel]], *a)
        folds = per_fold_r2(y[sel], p[sel], w_in[sel], cl[sel], G, outer[sel], direction[sel],
                            odd)
        comps[c] = {"rows": int(sel.sum()), "r2c": frozen_r2,
                    "unwinsorised": r2_ci(y_raw[sel], [p[sel]], *a, direction[sel], odd)[0],
                    "raw": raw[0], "direction_floor": raw[1],
                    "raw_minus_floor": raw[0]["r2"] - raw[1]["r2"],
                    "folds": {"r2": folds.tolist(), "sd": float(np.nanstd(folds)),
                              "min": float(np.nanmin(folds)), "max": float(np.nanmax(folds))}}
    fm = np.nanmean([comps[c]["folds"]["r2"] for c in SX], axis=0)
    out["components"] = comps
    out["folds_mean"] = {"r2": fm.tolist(), "sd": float(np.nanstd(fm)),
                         "min": float(np.nanmin(fm)), "max": float(np.nanmax(fm))}
    # steps 3 and 4: the heads (generalists)
    E = g.shape[1] - 1
    if E >= 2:
        want = c1_reference(run.decisions, name).get("C1-heads") or {}
        one = np.ones((1, G))
        dw = {}
        for c, j in zip(SX, K_SX, strict=True):
            got = AN.head_disagreement_boot(g[fr.inner, 1:, j], w_in[fr.inner], cl[fr.inner],
                                            one)[0]
            dw[c] = float(abs(got - (want.get("D") or {}).get(f"g_{c}", np.nan)))
        if not all(v <= WIRING_TOL for v in dw.values()):
            raise SystemExit(f"CS-2: {name}'s heads do not reproduce C1-heads' D: {dw}")
        out["wiring"]["heads_D_abs_diff"] = dw
        heads = []
        for i in range(1, E + 1):
            ch = frozen_c1(fr, g[:, i], counts)
            heads.append({"r2c": {c: float(ch.r2c[k]) for c, k in zip(SX, K_SX, strict=True)},
                          "stat": float(ch.stat), "ci": list(ch.ci),
                          "minus_main": float(ch.stat - c1.stat)})
        rel = {}
        for c, k in zip(SX, K_SX, strict=True):
            sel, odd = rows[c], c == "xi"
            q, ww = g[sel, 1:, k], w_in[sel]
            rho = CR.head_reliability(q, ww)
            corr_h = [wcorr(g[sel, i, k], fr.lam[sel, k], ww, direction[sel] if odd else None)
                      for i in range(1, E + 1)]
            ceiling = float(np.sqrt(rho)) if rho >= 0 else float("nan")
            rel[c] = {"rho": float(rho), "sqrt_rho": ceiling,
                      "rho_within_direction": (reliability_within(q, ww, direction[sel])
                                               if odd else None),
                      "corr_heads": corr_h,
                      "corr_main": wcorr(g[sel, 0, k], fr.lam[sel, k], ww,
                                         direction[sel] if odd else None),
                      "ceiling_binds": bool(any(abs(v) > ceiling for v in corr_h))}
        inner = fr.inner
        rel["interior_all_components"] = {
            nm: float(CR.head_reliability(g[inner, 1:, j], w_in[inner]))
            for j, nm in enumerate(TG.STATE_NAMES)}
        out["heads"] = heads
        out["reliability"] = rel
    # step 6: C1 on D
    D = np.zeros(world.n, bool)
    D[CS.d_rows(world)] = True
    Dr = D & fr.inner
    for c, k in zip(SX, K_SX, strict=True):
        guard_rows(tab, Dr & fr.valid[:, k], c, k2b, fr.valid[:, k])
    wD = ST.stratum_weights(tab, Dr)
    cD = frozen_c1(fr, g[:, 0], counts, rows=Dr, w=wD)
    fitted = {}
    for c, k in zip(SX, K_SX, strict=True):
        s = Dr & fr.valid[:, k] & np.isfinite(c1.calibrated.pred[:, k])
        fitted[c] = r2_ci(fr.lam[s, k], [c1.calibrated.pred[s, k]], wD[s], cl[s], G, counts,
                          direction[s], c == "xi")[0]
        fitted[c]["rows"] = int(s.sum())
    out["c1_on_d"] = {"rows": int(Dr.sum()), "n_rows": {c: int(cD.n_rows[k]) for c, k in
                                                        zip(SX, K_SX, strict=True)},
                      "r2c": {c: float(cD.r2c[k]) for c, k in zip(SX, K_SX, strict=True)},
                      "stat": float(cD.stat), "ci": list(cD.ci), "band": cD.decision,
                      "interior_fitted": fitted}
    # step 7: the accel term and the accel-channel sensitivity (no target: any row, plan R5)
    t1 = np.asarray(tab["t"]) >= 1
    union = rows["s"] | rows["xi"]
    acc = {}
    for sname, m in (("interior", fr.inner), ("c1_rows", union),
                     ("interior_t1", fr.inner & t1), ("c1_rows_t1", union & t1)):
        acc[sname] = accel_record(g[:, 0, 3], crit["g_held"][:, 0, 3],
                                  np.abs(crit["accel_sens"][:, 0, 1]), util, m,
                                  ST.stratum_weights(tab, m))
    out["accel"] = acc
    out["seconds"] = time.time() - t0
    print(f"CS-2 {out['label']}: R2_c {c1.stat:.3f}, D {cD.stat:.3f} "
          f"({out['seconds']:.0f} s)", flush=True)
    return out


def cs2_expected(nets: dict) -> dict:
    """E2.1-E2.6 read as amendment 2 items 1 and 2 read them."""
    from apexzero.costate import analysis as AN

    dec = {k: v for k, v in nets.items() if v["label"] in CS1_ORDER}
    gen = {k: v for k, v in nets.items() if "reliability" in v}
    others = {k: v for k, v in nets.items() if k not in dec}

    def both(fn, sub):
        return _count_of(sub, fn)

    e21 = {"rho_at_least_0.9": both(lambda v: all(v["reliability"][c]["rho"] >= RHO_E21
                                                  for c in SX), gen),
           "ceiling_never_binds": both(lambda v: not any(v["reliability"][c]["ceiling_binds"]
                                                         for c in SX), gen)}
    e21["holds"] = all(p["count"] == p["of"] for p in e21.values())
    heads = [(k, i, h) for k, v in gen.items() for i, h in enumerate(v.get("heads", []))]
    band = [(k, i) for k, i, h in heads if h["ci"][1] < E22_BAND]
    agree = [(k, i) for k, i, h in heads if abs(h["minus_main"]) <= E22_AGREE]
    e22 = {"heads": len(heads), "ci_upper_below_0.2": len(band), "within_0.1_of_main": len(agree),
           "failing": sorted({f"{nets[k]['label']}/head{i}" for k, i, _ in heads
                              if (k, i) not in set(band) or (k, i) not in set(agree)})}
    e22["holds"] = bool(heads) and len(band) == len(agree) == len(heads)

    def e23(v):
        c = v["components"]
        return (abs(c["s"]["unwinsorised"]["r2"] - c["s"]["r2c"]["r2"]) <= E23_S
                and c["xi"]["unwinsorised"]["r2"] < c["xi"]["r2c"]["r2"])

    def read(fn):
        r = {"decision": both(fn, dec), "others": both(fn, others)}
        r["holds"] = r["decision"]["count"] >= E_COUNT
        return r

    return {"E2.1": e21, "E2.2": e22, "E2.3": read(e23),
            "E2.4": read(lambda v: abs(v["components"]["xi"]["raw_minus_floor"]) <= E24_FLOOR),
            "E2.5": read(lambda v: v["c1_on_d"]["ci"][1] < AN.C1_REFUTE),
            "E2.6": read(lambda v: v["accel"]["c1_rows"]["share_mean"] < E26_SHARE),
            "reading": "amendment 2 item 1: without a count, at least 6 of the 9 decision "
                       "networks; E2.1 and E2.2 over every generalist and head"}


def world_arrays(world, fields=("ob",)) -> dict:
    """{field: [n, ...]} of the world's per-key ObsJac fields as T-length arrays."""
    import numpy as np

    out: dict = {}
    for k in world.keys:
        for f in fields:
            v = np.asarray(getattr(k.oj, f))
            out.setdefault(f, np.full((world.n, *v.shape[1:]), np.nan))[k.rows] = v
    return out


@computes("CS-2")
def cs2(run: Run) -> None:
    """CS-2 (plan Section 4, amendment 2 item 2): ``_cs2_network`` on every network, then the
    expected directions read (``cs2_expected``); outputs summary.json."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    world = load_world(run)
    ob = world_arrays(world)["ob"]
    if not np.isfinite(ob).all():
        raise SystemExit("CS-2: T has rows without a raw observation")
    util = ob[:, list(TYRE_UTIL)].max(1) + 1.0
    G = AN.geometry_clusters(world.tab)[1]
    counts = PB.cluster_counts(G)
    want = getattr(run.args, "network", None)
    names = [n for n in all_networks(run.rec)
             if want is None or want in (n, network_label(n))]
    if getattr(run.args, "limit", None) is not None:
        names = names[:max(1, int(run.args.limit))]
    if not names:
        raise SystemExit(f"CS-2: --network {want} is not a network of the record")
    nets = {n: _cs2_network(run, n, world, util, counts) for n in names}
    out = {"item": "CS-2", "networks": nets, "expected": cs2_expected(nets),
           "definitions": "plan Section 4 CS-2 and amendment 2 items 1 and 2",
           "components": list(SX), "util_bins": list(UTIL_BINS), "n_boot": len(counts)}
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-3: C3-int on s, xi
#
# Plan Section 4 CS-3 with amendment 2 item 3: per network x geometry, the signed calibrated
# critic residual of s and xi on C1's rows and four behavioural deviations (step 2's G - 1
# and B, step 1's entry-speed ratio - 1 and brake-onset residual), double-centred over
# networks and geometries; each of the 8 pairs' correlation with a two-sided permutation p
# (C3-int's blocked permutation of the geometry labels within each network) and Holm's
# adjustment.

CS3_MEASURES = ("G_minus_1", "B", "entry_speed_ratio_minus_1", "brake_onset_residual")
STEP2_MEASURES = {"G_minus_1": "G", "B": "B"}
STEP1_MEASURES = ("entry_speed_ratio_minus_1", "brake_onset_residual")
E31_R, E31_P = 0.15, 0.05
PERM_TIE = 1e-12                         # analysis.c3_int's tie allowance


def geometry_means(v, w, geo, rows, G: int):
    """[G] the w-weighted mean of v over each geometry's ``rows`` (NaN where it has none)."""
    import numpy as np

    v, w, geo, rows = np.asarray(v, float), np.asarray(w, float), np.asarray(geo), np.asarray(
        rows, bool)
    out = np.full(G, np.nan)
    for gg in range(G):
        m = rows & (geo == gg) & np.isfinite(v)
        if m.any() and w[m].sum() > 0:
            out[gg] = w[m] @ v[m] / w[m].sum()
    return out


def geometry_medians(values: dict, key_geo: dict, G: int):
    """[G] the median over each geometry's keys of {key_id: value}, finite values only."""
    import numpy as np

    by: dict[int, list] = {}
    for k, v in values.items():
        if k in key_geo and np.isfinite(v):
            by.setdefault(int(key_geo[k]), []).append(float(v))
    out = np.full(G, np.nan)
    for gg, vs in by.items():
        out[gg] = float(np.median(vs))
    return out


def geometry_blocks(labels):
    """[G] C3-int's permutation blocks of T's geometries: R tercile x theta class
    (``states.r_tercile``, ``states.theta_class``)."""
    import numpy as np

    from apexzero.costate import states as ST

    labels = np.asarray(labels, float)
    return (ST.r_tercile(labels[:, 0], labels[:, 1]) * (len(ST.THETA_CLASSES) + 1)
            + ST.theta_class(labels[:, 1]))


def holm(p) -> list:
    """Holm's step-down adjusted p-values (NaN kept as NaN)."""
    import numpy as np

    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok], kind="mergesort")]
    m, run_max = len(order), 0.0
    for j, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - j) * p[i]))
        out[i] = run_max
    return out.tolist()


def c3_pairs_test(res: dict, beh: dict, pairs, blocks, n_perm: int, rng) -> dict:
    """Each pair's correlation over present cells of its double-centred tables and its
    two-sided permutation p (amendment 2 item 3). ``res`` {component: [N, G]}, ``beh``
    {measure: [N, G]} over one network list; ``pairs`` [(name, component, measure, network
    indices)]; each draw permutes the geometry labels within each of the N networks within
    ``blocks`` (as ``analysis.c3_int`` draws them) and serves every pair."""
    import warnings

    import numpy as np

    from apexzero.costate import analysis as AN

    # a geometry no network of a pair holds is an all-missing column: double_centre's
    # nanmean warns there and leaves it missing
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    blocks = np.asarray(blocks)
    N, G = next(iter(beh.values())).shape
    groups = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    R = {nm: AN.double_centre(np.asarray(res[c], float)[idx]) for nm, c, _, idx in pairs}
    obs = {}
    for nm, _, b, idx in pairs:
        obs[nm] = AN._corr_present(R[nm].ravel(), AN.double_centre(
            np.asarray(beh[b], float)[idx]).ravel())
    null = np.full((int(n_perm), len(pairs)), np.nan)
    rows = np.arange(N)[:, None]
    for d in range(int(n_perm)):
        perm = np.empty((N, G), int)
        for i in range(N):
            p = np.arange(G)
            for gidx in groups:
                p[gidx] = gidx[rng.permutation(len(gidx))]
            perm[i] = p
        cache: dict = {}
        for j, (nm, _, b, idx) in enumerate(pairs):
            key = (b, tuple(np.asarray(idx).tolist()))
            if key not in cache:
                cache[key] = AN.double_centre(np.asarray(beh[b], float)[rows, perm][idx])
            null[d, j] = AN._corr_present(R[nm].ravel(), cache[key].ravel())
    out = {}
    for j, (nm, c, b, idx) in enumerate(pairs):
        z = null[:, j][np.isfinite(null[:, j])]
        r = obs[nm]
        p = (float((1 + np.sum(np.abs(z) >= abs(r) - PERM_TIE)) / (1 + len(z)))
             if np.isfinite(r) else float("nan"))
        present = np.isfinite(R[nm]) & np.isfinite(np.asarray(beh[b], float)[idx])
        out[nm] = {"component": c, "measure": b, "networks": len(idx), "r": float(r), "p": p,
                   "cells": int(present.sum()), "n_perm": len(z),
                   "null_sd": float(np.std(z)) if len(z) else float("nan"),
                   "null_q": np.quantile(z, [0.025, 0.975]).tolist() if len(z) else None}
    return {"pairs": out, "null": null}


def step2_tables(run: Run, names, key_geo: dict, G: int) -> tuple[dict, dict]:
    """({measure: {network: [G]}}, per-network notes) of step 2's G - 1 and B (amendment 2
    item 3): each rows file through the record's ``c3w_inputs`` binding and its stamp, the
    optimum's rows through their stamp, every row's checkpoint the network's pin; per key
    ``fields.stages.edge_key_measures`` (step 2's own exclusions), per geometry the median."""
    import numpy as np

    from apexzero.costate import prereg as PR
    from apexzero.fields import prereg as FP
    from apexzero.fields import stages as S

    d = CS.fields_out(run.root)
    refuse_sealed(d)

    def stamped(p: Path) -> str:
        sha = sha256_file(p)
        st = FP.read_stamp(p) or {}
        if (st.get("files") or {}).get(p.name) != sha:
            raise SystemExit(f"CS-3: {p.name} is not the file its step 2 stamp binds")
        run.note_input(PR._repo_rel(p, run.root), sha)
        return sha

    opt_p = d / "opt-fields" / "rows.jsonl"
    stamped(opt_p)
    opt = {r["key_id"]: r for r in S.read_rows(opt_p)}
    tabs: dict = {m: {} for m in STEP2_MEASURES}
    notes = {}
    for name in names:
        ci = (run.rec.get("c3w_inputs") or {}).get(name) or {}
        if not ci.get("overlap_file"):
            continue
        p = d / "fields-a2" / Path(ci["overlap_file"]).name
        if stamped(p) != ci.get("overlap_sha256"):
            raise SystemExit(f"CS-3: {p.name} is not the file the record's c3w_inputs bind")
        rows = S.read_rows(p)
        pin = run.rec["networks"][name]["sha256"]
        bad = [r.get("key_id") for r in rows if r.get("ckpt_sha256") not in (None, pin)]
        if bad or not any(r.get("ckpt_sha256") == pin for r in rows):
            raise SystemExit(f"CS-3: {p.name}'s rows are not of {name}'s pinned checkpoint")
        note = {}
        for m, field in STEP2_MEASURES.items():
            km = S.edge_key_measures(rows, opt, field)
            vals = {k.key_id: k.value for k in km}
            t = geometry_medians(vals, key_geo, G)
            tabs[m][name] = t - 1.0 if m == "G_minus_1" else t
            note[m] = {"keys": len(km), "keys_in_T": int(sum(k in key_geo for k in vals)),
                       "finite_in_T": int(sum(k in key_geo and np.isfinite(v)
                                              for k, v in vals.items())),
                       "geometries": int(np.isfinite(t).sum())}
        notes[name] = note
    return tabs, notes


def psychophysics_out(root: Path = ROOT) -> Path:
    """Step 1's outputs (outputs/psychophysics, or through a worktree's outputs/outputs
    link), as ``scripts/costate.py``'s ``fields_out`` finds step 2's."""
    d = Path(root) / "outputs" / "psychophysics"
    if (d / "analysis").exists():
        return d
    return Path(root) / "outputs" / "outputs" / "psychophysics"


def step1_tables(run: Run, names, tab, key_geo: dict, G: int) -> tuple[dict, dict]:
    """({measure: {network: [G]}}, per-network notes) of step 1's entry-speed ratio - 1 and
    brake-onset residual (amendment 2 item 3): each agent's analysis run table at its pinned
    final, the 'det' run at each T key's (cell, phase) with ``w_is_cell``, its stim and W
    checked; e_ve and -dd_b, finite only; per geometry the median."""
    import numpy as np

    from apexzero.costate import prereg as PR

    base = psychophysics_out(run.root) / "analysis" / "runs"
    refuse_sealed(base)
    keys: dict = {}
    for i in range(len(tab["key_id"])):
        k = str(tab["key_id"][i])
        if k not in keys:
            keys[k] = (int(tab["cell_id"][i]), int(tab["key_phase"][i]), int(tab["stim_id"][i]),
                       int(tab["W"][i]), bool(tab["w_is_cell"][i]))
    tabs: dict = {m: {} for m in STEP1_MEASURES}
    notes = {}
    for name in names:
        label = network_label(name)
        ts = int(run.rec["networks"][name]["timesteps"])
        p = base / f"{label}_{ts:012d}.npz"
        if not p.exists():
            raise SystemExit(f"CS-3: no step 1 run table {p.name}")
        run.note_input(PR._repo_rel(p, run.root), sha256_file(p))
        with np.load(p, allow_pickle=False) as d:
            if not np.all(d["timesteps"] == ts):
                raise SystemExit(f"CS-3: {p.name} is not of {name}'s pinned final")
            det = np.flatnonzero(d["mode"] == "det")
            col = {c: d[c][det] for c in ("cell", "phase", "stim", "W", "e_ve", "dd_b",
                                          "crashed")}
        at = {(int(c), int(ph)): j for j, (c, ph) in enumerate(zip(col["cell"], col["phase"],
                                                                   strict=True))}
        vals: dict = {m: {} for m in STEP1_MEASURES}
        crashed = missing = 0
        for k, (cell, ph, stim, W, w_is_cell) in keys.items():
            if not w_is_cell:
                continue
            j = at.get((cell, ph))
            if j is None:
                missing += 1
                continue
            if int(col["stim"][j]) != stim or int(col["W"][j]) != W:
                raise SystemExit(f"CS-3: {p.name}'s det run at key {k} is not the key's")
            crashed += bool(col["crashed"][j])
            vals["entry_speed_ratio_minus_1"][k] = float(col["e_ve"][j])
            vals["brake_onset_residual"][k] = -float(col["dd_b"][j])
        note = {"keys": sum(w for *_, w in keys.values()), "missing": missing,
                "crashed": int(crashed)}
        for m in STEP1_MEASURES:
            t = geometry_medians(vals[m], key_geo, G)
            tabs[m][name] = t
            note[m] = {"finite": int(sum(np.isfinite(v) for v in vals[m].values())),
                       "geometries": int(np.isfinite(t).sum())}
        notes[name] = note
    return tabs, notes


def _cs3_residuals(run: Run, name: str, world, counts) -> tuple:
    """([2, G] the signed calibrated residual of s and xi per geometry, the network's notes)
    (amendment 2 item 3)."""
    import numpy as np

    from apexzero.costate import states as ST

    _, _, fr = network_frame(run, name, world)
    g = read_critic(run, name, ("g",))["g"]
    c1, wiring = checked_c1(run, name, fr, g[:, 0], counts)
    rows = c1_rows(fr, run.rec["frozen"]["k2b_excluded"])
    geo = ST.geometry_index(fr.tab)
    sigma = np.asarray(fr.tab["direction"], float)
    out = np.full((len(SX), fr.G), np.nan)
    for j, (c, k) in enumerate(zip(SX, K_SX, strict=True)):
        r = c1.calibrated.pred[:, k] - fr.lam[:, k]
        if c == "xi":
            r = sigma * r
        out[j] = geometry_means(r, fr.w_in, geo, rows[c], fr.G)
    return out, {"wiring": wiring, "geometries": [int(np.isfinite(v).sum()) for v in out],
                 "decision_rows": {c: int(r.sum()) for c, r in rows.items()}}


@computes("CS-3")
def cs3(run: Run) -> None:
    """CS-3 (plan Section 4, amendment 2 item 3): the residual tables of every network, step
    2's and step 1's behaviour tables, the 8 primary pairs (generalists) with Holm's
    adjustment and step 1's pairs with S1-S3 added (descriptive); outputs summary.json and
    tables.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import stream_rng

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    labels = AN.geometry_labels(tab)
    G = len(labels)
    counts = PB.cluster_counts(G)
    geo = ST.geometry_index(tab)
    key_geo = {str(k): int(g_) for k, g_ in zip(tab["key_id"], geo, strict=True)}
    nets = all_networks(run.rec)
    if getattr(run.args, "network", None) is not None:
        raise SystemExit("CS-3 is across networks: it takes no --network (--limit N keeps the "
                         "first N networks, a partial run)")
    if getattr(run.args, "limit", None) is not None:
        nets = nets[:max(1, int(run.args.limit))]
    res = np.full((len(nets), len(SX), G), np.nan)
    notes = {}
    for i, n in enumerate(nets):
        res[i], notes[n] = _cs3_residuals(run, n, world, counts)
        print(f"CS-3 {network_label(n)}: residual tables ({time.time() - t0:.0f} s)", flush=True)
    gen = [n for n in nets if n in AN.GENERALISTS]
    step1 = [n for n in [AN.network_of(lb) for lb in CS1_ORDER] + list(DESCRIPTIVE_SPECIALISTS)
             if n in nets]
    t2, n2 = step2_tables(run, gen, key_geo, G)
    t1, n1 = step1_tables(run, step1, tab, key_geo, G)
    beh = {m: np.full((len(nets), G), np.nan) for m in CS3_MEASURES}
    for m, d in {**t2, **t1}.items():
        for n, v in d.items():
            beh[m][nets.index(n)] = v
    has = {m: [i for i in range(len(nets)) if np.isfinite(beh[m][i]).any()] for m in beh}
    pairs, desc = [], []
    for c in SX:
        for m in CS3_MEASURES:
            idx = np.array([i for i in has[m] if nets[i] in AN.GENERALISTS], int)
            pairs.append((f"{c}~{m}", c, m, idx))
            if m in STEP1_MEASURES:
                desc.append((f"{c}~{m}+S", c, m, np.array(has[m], int)))
    resd = {c: res[:, j] for j, c in enumerate(SX)}
    blocks = geometry_blocks(labels)
    n_perm = int(run.rec["constants"]["analysis"]["c3_int_perm"])
    if n_perm != AN.C3_INT_PERMUTATIONS:
        raise SystemExit("CS-3: the record's c3_int_perm is not analysis.C3_INT_PERMUTATIONS")
    test = c3_pairs_test(resd, beh, pairs + desc, blocks, n_perm, stream_rng("costate_c3", 1))
    prim = {k: test["pairs"][k] for k, *_ in pairs}
    adj = holm([v["p"] for v in prim.values()])
    for v, a in zip(prim.values(), adj, strict=True):
        v["p_holm"] = a
    descr = {k: test["pairs"][k] for k, *_ in desc}
    e31 = {"max_abs_r": float(np.nanmax([abs(v["r"]) for v in prim.values()])),
           "all_abs_r_below_0.15": bool(all(abs(v["r"]) < E31_R for v in prim.values())),
           "none_holm_at_or_below_0.05": bool(all(not (v["p_holm"] <= E31_P)
                                                  for v in prim.values()))}
    e31["holds"] = e31["all_abs_r_below_0.15"] and e31["none_holm_at_or_below_0.05"]
    out = {"item": "CS-3", "networks": nets, "pairs": prim, "descriptive_with_specialists": descr,
           "expected": {"E3.1": e31},
           "networks_per_measure": {m: [nets[i] for i in has[m]] for m in CS3_MEASURES},
           "residual_notes": notes, "step2_notes": n2, "step1_notes": n1,
           "geometries": labels.tolist(), "blocks": blocks.tolist(), "n_perm": n_perm,
           "stream": ["costate_c3", 1], "definitions": "plan Section 4 CS-3 and amendment 2 "
           "items 1 and 3", "seconds": time.time() - t0}
    run.write_npz("tables", {"residual": res, **{f"behaviour/{m}": v for m, v in beh.items()},
                             "null": test["null"], "pairs": np.array([k for k, *_ in pairs + desc]),
                             "networks": np.array(nets)})
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-4: Delta C1, R2_z, MLP
#
# Plan Section 4 CS-4 with amendment 2 item 4, on the 9 decision networks and each
# component's C1 rows: R2_z (ridge on the network's clipped z), R2(z + g) and Delta C1 (g the
# main V's 8 pulled-back components as the hidden block; ``analysis.delta_c1`` must agree),
# the MLP-on-z ceiling (``probes.mlp_cv``) and R2_c / R2_MLP. One pool unit per (network,
# component).

DELTA_TOL = 1e-12                        # analysis.delta_c1 against the probes here
E41 = {"s": 0.2, "xi": 0.3}
E42_GAIN, E42_XI = 0.05, 0.4
E43_RATIO, E44_DELTA = 0.1, 0.02


def cs4_unit(state, unit) -> dict:
    """One (network, component) of CS-4 in a worker: the ridge probes on z and on z + g, the
    check against ``analysis.delta_c1`` and the MLP probe, out of fold on the unit's rows.
    ``unit``: the activations file (sha256 checked by the parent), the rows, targets, weights,
    folds (partition 0 subset), turn directions and g [rows, 8]."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    t0 = time.time()
    with np.load(unit["path"]) as d:
        z = np.clip(d["z"][unit["idx"]], -10.0, 10.0)
    fo, w, comp = unit["fo"], unit["w"], unit["comp"]
    Y = np.asarray(unit["lam"], float)[:, None]
    names = [f"lam_{comp}"]
    why = AN.probe_cv_problem(fo, np.ones(len(w), bool))
    if why:
        raise RuntimeError(f"CS-4 {unit['label']}/{comp}: {why}")
    d_dir = unit["direction"] if comp == "xi" else None
    base = PB.ridge_cv(z, Y, fo, w, None, names)
    full = PB.ridge_cv(z, Y, fo, w, unit["g8"], names)
    dc = AN.delta_c1(z, unit["g8"], Y, fo, w, unit["direction"], [comp])
    r0 = float(PB.probe_r2(base, w, d_dir)[0])
    r1 = float(PB.probe_r2(full, w, d_dir)[0])
    diff = max(abs(dc["r2_z"][0] - r0), abs(dc["r2_z_g"][0] - r1))
    if not diff <= DELTA_TOL:
        raise RuntimeError(f"CS-4 {unit['label']}/{comp}: analysis.delta_c1 differs ({diff:.3g})")
    t1 = time.time()
    mlp = PB.mlp_cv(z, Y, fo, w, names)
    return {"pred_z": base.pred[:, 0], "pred_zg": full.pred[:, 0], "pred_mlp": mlp.pred[:, 0],
            "y": base.y[:, 0], "alpha_z": base.alpha[:, 0, 0], "alpha_zg": full.alpha[:, 0],
            "alpha_mlp": mlp.alpha[:, 0, 0], "delta_c1": dc, "delta_c1_check": float(diff),
            "seconds_ridge": t1 - t0, "seconds_mlp": time.time() - t1}


@computes("CS-4")
def cs4(run: Run) -> None:
    """CS-4 (plan Section 4, amendment 2 item 4): the frames, the frozen C1 checked, one pool
    unit per (network, component) (``cs4_unit``), then the CIs, the ratio and the expected
    directions; outputs summary.json and preds.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    direction = np.asarray(tab["direction"])
    G = AN.geometry_clusters(tab)[1]
    counts = PB.cluster_counts(G)
    want = getattr(run.args, "network", None)
    labels = [lb for lb in CS1_ORDER if want is None or want in (lb, AN.network_of(lb))]
    if not labels:
        raise SystemExit(f"CS-4: --network {want} is not a decision network")
    units, frames = [], {}
    for lb in labels:
        name = AN.network_of(lb)
        _, _, fr = network_frame(run, name, world)
        g = read_critic(run, name, ("g",))["g"][:, 0]
        c1, wiring = checked_c1(run, name, fr, g, counts)
        rows = c1_rows(fr, run.rec["frozen"]["k2b_excluded"])
        act = run.read(f"post/activations/{PR.net_file(name)}.npz")
        frames[lb] = (fr, c1, wiring, rows)
        for c, k in zip(SX, K_SX, strict=True):
            idx = np.flatnonzero(rows[c])
            units.append({"label": lb, "comp": c, "path": str(act), "idx": idx,
                          "lam": fr.lam[idx, k], "w": fr.w_in[idx], "g8": g[idx],
                          "fo": fr.fo.subset(rows[c]), "direction": direction[idx]})
    print(f"CS-4: {len(units)} units on {min(run.args.workers or DEFAULT_WORKERS, len(units))} "
          f"workers ({time.time() - t0:.0f} s)", flush=True)
    got = run.pool(cs4_unit, units, names=[f"{u['label']}/{u['comp']}" for u in units])
    nets: dict = {}
    arrays = {}
    for u, r in zip(units, got, strict=True):
        lb, c = u["label"], u["comp"]
        fr, c1, wiring, rows = frames[lb]
        k = K_SX[SX.index(c)]
        odd = c == "xi"
        if not np.array_equal(r["y"], u["lam"]):
            raise SystemExit(f"CS-4 {lb}/{c}: the probes' targets are not the unit's")
        cl = fr.cl[u["idx"]]
        P = [r["pred_z"], r["pred_zg"], r["pred_mlp"]]
        if not all(np.isfinite(p).all() for p in P):
            raise SystemExit(f"CS-4 {lb}/{c}: rows without an out-of-fold prediction")
        pt, dr = AN.boot_r2(u["lam"], P, u["w"], cl, G, counts, u["direction"], odd)
        if abs(pt[0] - r["delta_c1"]["r2_z"][0]) > DELTA_TOL:
            raise SystemExit(f"CS-4 {lb}/{c}: the bootstrap's R2_z is not the probe's")
        r2c = float(c1.r2c[k])
        rec = {"rows": len(u["idx"]),
               "r2_z": {"r2": float(pt[0]), "ci": list(AN._ci(dr[:, 0]))},
               "r2_z_g": {"r2": float(pt[1]), "ci": list(AN._ci(dr[:, 1]))},
               "r2_mlp": {"r2": float(pt[2]), "ci": list(AN._ci(dr[:, 2]))},
               "delta_c1": {"delta": float(pt[1] - pt[0]), "ci": list(AN._ci(dr[:, 1] - dr[:, 0]))},
               "mlp_minus_z": {"diff": float(pt[2] - pt[0]),
                               "ci": list(AN._ci(dr[:, 2] - dr[:, 0]))},
               "r2c": r2c, "ratio_to_ceiling": AN.ratio_to_ceiling(r2c, float(pt[2])),
               "alpha_z": r["alpha_z"].tolist(), "alpha_zg": r["alpha_zg"].tolist(),
               "alpha_mlp": r["alpha_mlp"].tolist(), "delta_c1_check": r["delta_c1_check"],
               "seconds_ridge": r["seconds_ridge"], "seconds_mlp": r["seconds_mlp"]}
        nets.setdefault(lb, {"run": AN.network_of(lb), "wiring": {"c1_max_abs_diff": wiring},
                             "c1_stat": float(c1.stat), "components": {}})["components"][c] = rec
        arrays.update({f"{lb}/{c}/rows": u["idx"], f"{lb}/{c}/pred_z": r["pred_z"],
                       f"{lb}/{c}/pred_zg": r["pred_zg"], f"{lb}/{c}/pred_mlp": r["pred_mlp"]})
    for v in nets.values():
        cc = v["components"]
        v["mean"] = {k: float(np.mean([cc[c][k]["r2"] for c in SX]))
                     for k in ("r2_z", "r2_z_g", "r2_mlp")}
        v["mean"]["ratio_to_ceiling"] = AN.ratio_to_ceiling(v["c1_stat"], v["mean"]["r2_mlp"])

    def cnt(fn):
        return _count_of(nets, fn)

    def comp(v, c):
        return v["components"][c]

    e41 = {c: cnt(lambda v, c=c: comp(v, c)["r2_z"]["r2"] >= E41[c]) for c in SX}
    e42 = {c: cnt(lambda v, c=c: comp(v, c)["mlp_minus_z"]["diff"] >= E42_GAIN) for c in SX}
    e42["xi_mlp_above_0.4"] = cnt(lambda v: comp(v, "xi")["r2_mlp"]["r2"] > E42_XI)
    e43 = {c: cnt(lambda v, c=c: comp(v, c)["ratio_to_ceiling"] <= E43_RATIO) for c in SX}
    e44 = {c: cnt(lambda v, c=c: comp(v, c)["delta_c1"]["delta"] <= E44_DELTA
                  and comp(v, c)["delta_c1"]["ci"][0] <= 0.0 <= comp(v, c)["delta_c1"]["ci"][1])
           for c in SX}
    n = len(nets)
    expected = {
        "E4.1": {**e41, "holds": all(p["count"] >= min(E_COUNT, n) for p in e41.values())},
        "E4.2": {**e42, "holds": all(p["count"] >= min(E_COUNT, n) for p in e42.values())},
        "E4.3": {**e43, "holds": all(p["count"] == n for p in e43.values())},
        "E4.4": {**e44, "holds": all(p["count"] == n for p in e44.values())},
        "reading": "amendment 2 item 4"}
    out = {"item": "CS-4", "networks": nets, "expected": expected,
           "definitions": "plan Section 4 CS-4 and amendment 2 items 1 and 4",
           "components": list(SX), "n_boot": len(counts), "seconds": time.time() - t0}
    run.write_npz("preds", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-5: W1 on the Pi targets
#
# Plan Section 4 CS-5 with amendment 2 item 5, on the 9 decision networks and T's interior
# approach- and exit-phase rows: ridge probes of Pi^in, Pi^out, Pi^slip and Pi^mu on z, on z +
# the actor's h3 and on z + the critic's h3; each hidden layer's gain beyond z against 20
# random-init controls (the pool actors stored, 20 random-init critics computed here) with the
# two-level CI; the probes' R2 and Spearman; and W1's critic width price against Pi^in on the
# same rows. One pool unit per network.

PI_TARGETS = ("Pi_in", "Pi_out", "Pi_slip", "Pi_mu")
LAYERS = ("actor", "critic")
RANDOM_TOL = 1e-12                       # the pool actor of seed 0 rebuilt against its h3
E51_R2, E52_GAIN = 0.3, 0.05


def cs5_rows(tab):
    """[n] CS-5's rows: T's interior rows of the approach and exit phases (amendment 2 item
    5), through the row guard."""
    import numpy as np

    rows = np.asarray(tab["interior"], bool) & np.isin(np.asarray(tab["phase"]).astype(str),
                                                         SCORED_PHASES)
    return guard_rows(tab, rows)


def cs5_unit(state, unit) -> dict:
    """One network's CS-5 probes in a worker: z, z + actor h3, z + critic h3, and z + each
    control's h3 (the 20 pool actors stored; 20 random-init critics built from the network's
    PPOConfig, seeds 0-19), out of fold on the unit's rows, the four Pi targets per call."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB

    t0 = time.time()
    idx, cfg = unit["idx"], unit["cfg"]
    with np.load(unit["path"]) as d:
        z = np.clip(d["z"][idx], -10.0, 10.0)
        h = {"actor": d["actor_h3"][idx], "critic": d["critic_h3"][idx]}
        pool_a = [d[f"random{s}_h3"][idx] for s in PB.POOL_SEEDS]
    h0 = PB.hidden(PB.random_actor(cfg, 0), z)["h3"]
    gap = float(np.max(np.abs(h0 - pool_a[0])))
    if not gap <= RANDOM_TOL:
        raise RuntimeError(f"CS-5 {unit['label']}: the pool actor of seed 0 does not rebuild "
                           f"its stored h3 ({gap:.3g})")
    pool_c = [PB.hidden(PB.random_critic(cfg, s), z)["h3"] for s in CR.RANDOM_SEEDS]
    Y, w, fo = unit["Y"], unit["w"], unit["fo"]
    names = list(PI_TARGETS)

    def probe(hh=None):
        return PB.ridge_cv(z, Y, fo, w, hh, names)

    base = probe()
    out = {"z": base.pred, "y": base.y, "alpha_z": base.alpha[:, :, 0]}
    for k in LAYERS:
        p = probe(h[k])
        out[k], out[f"alpha_{k}"] = p.pred, p.alpha
    out["pool_actor"] = np.stack([probe(hh).pred for hh in pool_a])
    out["pool_critic"] = np.stack([probe(hh).pred for hh in pool_c])
    out["random0_gap"] = gap
    out["seconds"] = time.time() - t0
    return out


def control_gain(dr_full, dr_controls, dr_base, pool) -> dict:
    """The trained gain minus the controls' median gain (C2's arithmetic, amendment 2 item 5):
    ``dr_*`` R2 points [1 + B] (row 0 the point, then the bootstrap draws) of the full probe,
    each control's [J, 1 + B] and the base; ``pool`` [B, J] the two-level draws' control
    indices. Its two-level CI, its geometry-only CI and the network's rank."""
    import numpy as np

    from apexzero.costate import analysis as AN

    gt = np.asarray(dr_full) - np.asarray(dr_base)                   # [1 + B]
    gc = np.asarray(dr_controls) - np.asarray(dr_base)[None]         # [J, 1 + B]
    stat = float(gt[0] - np.median(gc[:, 0]))
    B = len(gt) - 1
    two = gt[1:] - np.median(gc[:, 1:][np.asarray(pool), np.arange(B)[:, None]], axis=1)
    geo = gt[1:] - np.median(gc[:, 1:], axis=0)
    above = int((gc[:, 0] > gt[0]).sum())
    return {"stat": stat, "ci": list(AN._ci(two)), "ci_geometry": list(AN._ci(geo)),
            "gain": float(gt[0]), "control_median": float(np.median(gc[:, 0])),
            "control_gains": gc[:, 0].tolist(), "rank": 1 + above, "of": len(gc) + 1}


@computes("CS-5")
def cs5(run: Run) -> None:
    """CS-5 (plan Section 4, amendment 2 item 5): one pool unit per network (``cs5_unit``),
    then the probes' R2 and Spearman, the gains against the controls with their two-level
    CIs, W1's calibrated agreement on the same rows and the expected directions; outputs
    summary.json and preds.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    frozen = run.rec["frozen"]
    rows = cs5_rows(tab)
    idx = np.flatnonzero(rows)
    w_all = ST.stratum_weights(tab, rows)
    w = w_all[idx]
    fo = PB.folds(tab)
    cl_all, G = AN.geometry_clusters(tab)
    cl = cl_all[idx]
    counts = PB.cluster_counts(G)
    pool = PB.pool_draws(PB.N_POOL)
    bnd = {t: frozen["winsor_bounds"][t] for t in PI_TARGETS}
    Y = np.stack([ST.winsorise(np.asarray(tab[t], float)[idx], bnd[t]) for t in PI_TARGETS], 1)
    want = getattr(run.args, "network", None)
    labels = [lb for lb in CS1_ORDER if want is None or want in (lb, AN.network_of(lb))]
    if not labels:
        raise SystemExit(f"CS-5: --network {want} is not a decision network")
    units, dv = [], {}
    for lb in labels:
        name = AN.network_of(lb)
        pin = run.rec["networks"][name]
        agent = CS.load_agent(name, pin, run.root)
        run.note_input(pin["path"], pin["sha256"])
        dv[lb] = read_critic(run, name, ("dV_dwin",))["dV_dwin"]
        act = run.read(f"post/activations/{PR.net_file(name)}.npz")
        units.append({"label": lb, "path": str(act), "cfg": agent.cfg, "idx": idx, "Y": Y,
                      "w": w, "fo": fo.subset(rows)})
    print(f"CS-5: {len(units)} units, {len(idx)} rows ({time.time() - t0:.0f} s)", flush=True)
    got = run.pool(cs5_unit, units, names=labels)
    nets, arrays = {}, {}
    for u, r in zip(units, got, strict=True):
        lb = u["label"]
        if not np.array_equal(r["y"], Y):
            raise SystemExit(f"CS-5 {lb}: the probes' targets are not the rows'")
        P = np.concatenate([r["z"][None], r["actor"][None], r["critic"][None], r["pool_actor"],
                            r["pool_critic"]])                       # [3 + 2J, rows, 4]
        if not np.isfinite(P).all():
            raise SystemExit(f"CS-5 {lb}: rows without an out-of-fold prediction")
        J = len(r["pool_actor"])
        per = {}
        for t, name_t in enumerate(PI_TARGETS):
            pt, dr = AN.boot_r2(Y[:, t], list(P[:, :, t]), w, cl, G, counts)
            allr = np.vstack([pt[None], dr])                          # [1 + B, K]
            rec = {}
            for i, pname in enumerate(("z", "actor", "critic")):
                rho, rd = AN.spearman_boot(P[i, :, t], Y[:, t], w, cl, counts)
                rec[f"r2_{pname}"] = {"r2": float(pt[i]), "ci": list(AN._ci(dr[:, i]))}
                rec[f"spearman_{pname}"] = {"rho": float(rho), "ci": list(AN._ci(rd))}
            for i, layer in enumerate(LAYERS, start=1):
                ctl = allr[:, 3 + (i - 1) * J:3 + i * J].T
                rec[f"gain_{layer}"] = control_gain(allr[:, i], ctl, allr[:, 0], pool)
            per[name_t] = rec
        w1 = {}
        for tgt, y_t in (("raw", np.asarray(tab["Pi_in"], float)),
                         ("winsorised", ST.winsorise(np.asarray(tab["Pi_in"], float),
                                                     bnd["Pi_in"]))):
            w1[tgt] = AN.calibrated_agreement(dv[lb], y_t, fo.outer, w_all, cl_all, G, rows,
                                              counts, bounds=[bnd["Pi_in"]])
        nets[lb] = {"run": AN.network_of(lb), "targets": per, "w1_pi_in": w1,
                    "random0_gap": r["random0_gap"], "seconds": r["seconds"],
                    "alpha_z": r["alpha_z"].tolist()}
        arrays.update({f"{lb}/z": r["z"], f"{lb}/actor": r["actor"], f"{lb}/critic": r["critic"],
                       f"{lb}/pool_actor": r["pool_actor"], f"{lb}/pool_critic": r["pool_critic"]})

    def cnt(fn):
        return _count_of(nets, fn)

    def tg(v, t):
        return v["targets"][t]

    e51 = {t: cnt(lambda v, t=t: tg(v, t)["r2_z"]["r2"] >= E51_R2) for t in ("Pi_in", "Pi_out")}
    e52 = {f"{layer}:{t}": cnt(lambda v, t=t, layer=layer: tg(v, t)[f"gain_{layer}"]["stat"]
                               <= E52_GAIN) for layer in LAYERS for t in PI_TARGETS}
    named = [f"{lb}/{layer}:{t}" for lb, v in nets.items() for layer in LAYERS for t in PI_TARGETS
             if tg(v, t)[f"gain_{layer}"]["stat"] > E52_GAIN
             and tg(v, t)[f"gain_{layer}"]["ci"][0] > 0]
    e53 = {t: cnt(lambda v, t=t: tg(v, t)["r2_z"]["r2"] < tg(v, "Pi_in")["r2_z"]["r2"])
           for t in ("Pi_slip", "Pi_mu")}
    n = len(nets)
    expected = {
        "E5.1": {**e51, "holds": all(p["count"] >= min(E_COUNT, n) for p in e51.values())},
        "E5.2": {**e52, "holds": all(p["count"] == n for p in e52.values()),
                 "above_0.05_with_ci_above_0": named},
        "E5.3": {**e53, "holds": all(p["count"] >= min(E_COUNT, n) for p in e53.values())},
        "reading": "amendment 2 item 5"}
    out = {"item": "CS-5", "networks": nets, "expected": expected, "rows": int(rows.sum()),
           "rows_by_phase": {p: int((rows & (np.asarray(tab["phase"]).astype(str) == p)).sum())
                             for p in SCORED_PHASES},
           "targets": list(PI_TARGETS), "winsor_bounds": bnd, "controls": {
               "actor": "the pool actors, seeds 0-19 (stored h3)",
               "critic": "random-init critics, seeds 0-19 (h3 computed here)"},
           "delta_pi_sd_t": "BC-a0 has not run: reported with BC-a0 (amendment 2 item 5)",
           "definitions": "plan Section 4 CS-5 and amendment 2 items 1 and 5",
           "n_boot": len(counts), "seconds": time.time() - t0}
    run.write_npz("preds", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-10, CS-11: own runs
#
# Plan Section 4 CS-10 (ii) and CS-11 on D, with amendment 3 (their operational details). Each
# of the 9 decision networks drives its deterministic policy from each of D's 477 keys (x_0,
# a_-1 = 0, accel 0) for 800 steps or to the strip's end (``own_horizon``); a run that
# terminates is dropped and counted. CS-10 sweeps the run once for lambda^own at every state
# (``own.own_costate``), pulls the critic back at every 5th state up to the run's own t_end,
# flags the run's own interior and scores C1-own there (``analysis.c1_own``). CS-11
# differentiates the run's return with respect to the road's width samples (``width_vjp``)
# for Pi^in,own at H = W and H = 800, and makes W1's three comparisons on D. Pooled: one unit
# per (network, quarter of D's keys). N's parts wait for CS-9.

OWN_STEPS = 800                          # H = 800 for every D key (design Section 3.3)
OWN_QUARTERS = 4                         # units per network (amendment 3 item 1)
RESTART_AT = (0.25, 0.6)                 # CS-10's restart checks, as shares of H (K3b's)
PULLBACK_TOL = 1e-10                     # the pullback at state 0 against the stored g
J_TOL = 1e-6                             # the width loop's J at d = 0 against own_return's
JVP_TOL = 1e-6                           # the width VJP's contraction against the JVP (both
                                         # amendment 6: two programs over an 800-step loop,
                                         # K3's AD tolerance, critic.FD_TOL)
DV_TOL = 1e-9                            # dV_dwin recomputed against the stored one, over
                                         # the network's largest |dV_dwin| on D (amendment 5)
FD_STEP = 1e-3                           # m: the descriptive Richardson differences
E10_ROWS, E10_WIDTH = 10.0, 0.5          # E10.1: rows x 10, CI at most half as wide
E11_R2 = 0.2                             # E11.3


@functools.lru_cache(maxsize=1)
def _bank():
    """Step 1's stimulus bank, once per process (a worker's own copy)."""
    from apexzero.costate import inputs as IN

    return IN.design()[0]


def d_keys(world) -> list[dict]:
    """D's keys in T's key order: each key's D row (its t = 0 row, a reset row), x_0 and the
    key-level columns the own-run tables copy."""
    import numpy as np

    tab = world.tab
    out = []
    for k in world.keys:
        i0 = int(k.rows[0])
        if int(tab["t"][i0]) != 0 or not bool(np.asarray(k.inp.reset)[0]):
            raise SystemExit(f"key {k.key_id}: its first T row is not its reset t = 0 row")
        out.append({"key_id": str(k.key_id), "row": i0, "x0": np.asarray(k.inp.x[0], float),
                    "stim_id": int(tab["stim_id"][i0]), "direction": int(tab["direction"][i0]),
                    "W": int(tab["W"][i0]), "s_b": float(tab["s_b"][i0]),
                    "lm": tuple(float(tab[c][i0]) for c in ("s_c0", "s_apex", "s_c1"))})
    return out


def own_start(x0, steps: int = OWN_STEPS):
    """The loop inputs of a D key's own run: a reset start (a_-1 = accel = 0), no forced
    command, s_stop = +inf, the harness's context (amendment 3 item 1)."""
    import numpy as np

    from apexzero.fields import network as NW

    return NW.LoopIn(np.asarray(x0, float), np.zeros(2), np.zeros(2), np.zeros((steps, 2)),
                     np.int32(0), np.float64(np.inf), NW.ctx64(), NW.context_channels())


def own_horizon(net, tr, li, steps: int = OWN_STEPS) -> dict:
    """``own.own_return`` from the start of ``li``: H (``steps``, or fewer at the strip's end),
    whether the run terminated or reached the strip's end within them, and its J."""
    import numpy as np

    from apexzero.costate import own as OW

    z = np.zeros((1, 2))
    r = OW.own_return(net, tr, li, np.asarray(li.x0, float)[None], z, z, steps)
    return {"H": int(r.H[0]), "terminated": bool(r.terminated[0]),
            "strip_end": bool(r.strip_end[0]), "J_return": float(r.J[0])}


def landmarks_of(bank, kd: dict):
    """The key's stimulus landmarks, required to equal T's columns (amendment 3 item 2)."""
    import numpy as np

    lm = bank.stimuli[kd["stim_id"]].landmarks
    got = (lm.s_c0, lm.s_apex, lm.s_c1)
    if not np.allclose(got, kd["lm"], rtol=0.0, atol=1e-6):
        raise RuntimeError(f"key {kd['key_id']}: its landmarks {got} are not T's {kd['lm']}")
    return lm


def own_states(x, lm, H: int, every: int | None = None) -> tuple:
    """(the run's scored steps, its own t_end or -1): k = 0, ``every``, ... up to the first k
    with s_k >= s_c1 + 50 m (``states.t_end``'s rule on the run), and k < H."""
    import numpy as np

    from apexzero.costate import states as ST

    every = ST.EVERY if every is None else int(every)
    try:
        te = ST.t_end(np.asarray(x)[:H + 1], lm)
    except ValueError:
        te = -1
    last = te if te >= 0 else H - 1
    ks = np.arange(0, last + 1, every)
    return ks[ks < H], te


@functools.lru_cache(maxsize=1)
def _margins_jit():
    import jax
    import jax.numpy as jnp

    from apexzero.env import racing
    from apexzero.vehicle import sim
    from apexzero.vehicle.dynamics import tyre_forces
    from apexzero.vehicle.ops import JAX

    edge = racing.EnvParams().edge_margin

    def one(tr, x, ctx):
        wl, wr = sim.widths_at(x[0], tr)
        p = racing.params_from_context(ctx, hidden_mu=1.0)
        fo = tyre_forces(x, p, JAX)
        clear = jnp.stack([wl - edge - x[1], wr - edge + x[1]])
        return clear, p.alpha_max - jnp.maximum(jnp.abs(fo.alpha_f), jnp.abs(fo.alpha_r))

    return jax.jit(jax.vmap(one, in_axes=(None, 0, None)))


def run_margins(tr, x) -> tuple:
    """(clearance [n, 2] to the env's termination boundary, left and right; slip margin [n])
    at a run's states (amendment 3 item 2: racing.py's off-track and slip rules, mu = 1)."""
    import jax.numpy as jnp
    import numpy as np

    from apexzero.fields import network as NW

    c, s = _margins_jit()(tr, jnp.asarray(np.asarray(x, float)), jnp.asarray(NW.ctx64()))
    return np.asarray(c), np.asarray(s)


def own_interior(clear, slip, k: int | None = None) -> tuple:
    """(interior [n], contact [n], ok [n]) of a run's states (amendment 3 item 2): ok is no
    contact (a clearance below the OCP's corridor margin or a slip margin below its slip
    margin), both clearances >= CLEARANCE_MIN and the slip margin >= SLIP_MARGIN_MIN; the
    interior at j is ok over [j, min(j + k, n - 1)]."""
    import numpy as np

    from apexzero.costate import states as ST
    from apexzero.ocp.window import WindowSpec

    spec = WindowSpec()
    k = ST.INTERIOR_K if k is None else int(k)
    cmin = np.min(np.asarray(clear, float), axis=1)
    slip = np.asarray(slip, float)
    contact = (cmin < spec.safety) | (slip < spec.slip_safety)
    ok = ~contact & (cmin >= ST.CLEARANCE_MIN) & (slip >= ST.SLIP_MARGIN_MIN)
    return ST._window_all(ok, k), contact, ok


def run_inputs(x, a, ks):
    """``critic.Inputs`` at a run's steps ks: (x_k, x_{k-1}, a_{k-1}), a reset row at k = 0."""
    import numpy as np

    from apexzero.costate import critic as CR

    ks = np.asarray(ks, int)
    km = np.maximum(ks - 1, 0)
    reset = ks == 0
    x, a = np.asarray(x, float), np.asarray(a, float)
    return CR.Inputs(x[ks].copy(), np.where(reset[:, None], x[ks], x[km]),
                     np.where(reset[:, None], 0.0, a[km]), reset)


def restart_error(net, tr, li, oc, H: int, at=RESTART_AT, steps: int = OWN_STEPS) -> float:
    """CS-10's restart check (amendment 3 item 2): own.own_costate from the run's x_k (x_{k-1},
    clip(a_{k-1}), accel_of(x_k, x_{k-1}), count H - k) against the sweep's lambda^own_k and
    lambda^own_held,k; the largest error relative to the row's largest |lambda|."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import own as OW

    err = 0.0
    for q in at:
        k = int(q * H)
        if k < 1 or k >= H:
            continue
        xk, xp = np.asarray(oc.x[k], float), np.asarray(oc.x[k - 1], float)
        lk = li._replace(x0=xk, prev_action=np.clip(np.asarray(oc.a[k - 1], float), -1.0, 1.0),
                         accel=np.asarray(CR.accel_of(xk, xp)))
        rk = OW.own_costate(net, tr, lk, steps, H - k, x_prev=xp)
        for got, want in ((rk.lam[0], oc.lam[k]), (rk.lam_held[0], oc.lam_held[k])):
            scale = max(float(np.max(np.abs(want))), 1e-300)
            err = max(err, float(np.max(np.abs(got - want))) / scale)
    return err


def own_nets(agent) -> tuple:
    """(the actor as a NetPolicy, the critic as a CriticNet), both marked trained (R4)."""
    from apexzero.costate import critic as CR
    from apexzero.fields import network as NW

    return NW.agent_net(agent), CR.trained_critic(agent)


def cs10_unit(state, unit) -> dict:
    """One (network, quarter of D's keys) of CS-10 in a worker (amendment 3 item 2): per key
    the run's horizon, its sweep, its scored states with their phase inputs, own interior
    flag, observability, lambda^own and the critic's pullback; the restart check on the
    unit's first kept run."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import own as OW
    from apexzero.costate import states as ST
    from apexzero.fields import network as NW

    t0 = time.time()
    net, crit = own_nets(unit["agent"])
    bank = _bank()
    out, restart = [], None
    for i in unit["keys"]:
        kd = state["keys"][i]
        rd = NW.road(bank, kd["stim_id"])
        lm = landmarks_of(bank, kd)
        li = own_start(kd["x0"])
        rec = {"key": int(i), **own_horizon(net, rd.base, li)}
        if rec["terminated"]:
            rec.update(kept=False, ks=np.array([0]), t_end=-1,
                       x=np.asarray(kd["x0"], float)[None])
            out.append(rec)
            continue
        H = rec["H"]
        oc = OW.own_costate(net, rd.base, li, OWN_STEPS, H, reset=True, all_states=True)
        if not oc.valid_all:
            raise RuntimeError(f"key {kd['key_id']}: a run own_return counts is not valid on "
                               "its H steps")
        ks, te = own_states(oc.x, lm, H)
        clear, slip = run_margins(rd.base, oc.x)       # every state (one shape), then 0..H
        inner, contact, _ok = own_interior(clear[:H + 1], slip[:H + 1])
        inp = run_inputs(oc.x, oc.a, ks)
        oj = CR.obs_jacobians(rd.base, inp)
        pb = CR.pullback(crit, None, inp, oj)
        jz, cl = oj.jz(crit.obs_rms)
        obs = ST.observability(jz, cl)
        rec.update(kept=True, ks=ks, t_end=int(te), x=oc.x[ks], lam=oc.lam[ks],
                   lam_held=oc.lam_held[ks], mapped=oc.mapped[ks], g=pb.g[:, 0],
                   g_held=pb.g_held[:, 0], value=pb.value[:, 0],
                   s_unobservable=obs.s_unobservable, interior=inner[ks], contact=contact[ks],
                   clearance=clear[ks], slip_margin=slip[ks], J=oc.J, J_env=oc.J_env,
                   interior_share=float(inner[:H + 1].mean()))
        if restart is None:
            restart = {"key": int(i), "H": H, "error": restart_error(net, rd.base, li, oc, H)}
        out.append(rec)
    return {"label": unit["label"], "keys": out, "restart": restart,
            "seconds": time.time() - t0}


def own_table(keys: list[dict], recs: list[dict], tab) -> tuple[dict, dict]:
    """(the own-run table, its per-row arrays) of one network over every D key, in T's key
    order: key-level columns from T's D row, state-level ones from the runs (amendment 3
    item 2). A dropped run keeps its state 0, so every D key, and so every geometry, is in
    the table."""
    from types import SimpleNamespace

    import numpy as np

    from apexzero.costate import states as ST
    from apexzero.interp import io

    cols = ("key_id", "stim_id", "pair_id", "direction", "R", "theta", "W", "width",
            "s_apex", "s_c0", "s_c1", "s_b")
    T, A = {c: [] for c in (*cols, "k", "s", "n", "vx", "kept", "t_end_run", "phase")}, {}
    fields = {"lam": 8, "lam_held": 8, "g": 8, "g_held": 8, "clearance": 2}
    flat = ("value", "slip_margin")
    flags = ("interior", "contact", "mapped", "s_unobservable")
    for f in (*fields, *flat, *flags):
        A[f] = []
    for kd, r in zip(keys, recs, strict=True):
        m = len(r["ks"])
        for c in cols:
            T[c].append(np.repeat(np.asarray(tab[c])[kd["row"]], m))
        x = np.asarray(r["x"], float)
        T["k"].append(np.asarray(r["ks"], int))
        T["s"].append(x[:, 0])
        T["n"].append(x[:, 1])
        T["vx"].append(x[:, 3])
        T["kept"].append(np.full(m, bool(r["kept"])))
        T["t_end_run"].append(np.full(m, int(r["t_end"])))
        lm = SimpleNamespace(s_c0=kd["lm"][0], s_apex=kd["lm"][1], s_c1=kd["lm"][2])
        T["phase"].append(ST.phases(x[:, 0], lm, kd["s_b"]))
        for f, d in fields.items():
            A[f].append(np.asarray(r[f], float) if r["kept"] else np.full((m, d), np.nan))
        for f in flat:
            A[f].append(np.asarray(r[f], float) if r["kept"] else np.full(m, np.nan))
        for f in flags:
            A[f].append(np.asarray(r[f], bool) if r["kept"] else np.zeros(m, bool))
    out = {c: np.concatenate(v) for c, v in T.items()}
    out["multi_basin"] = np.zeros(len(out["k"]), bool)
    out["interior"] = np.concatenate(A.pop("interior")) & out["kept"]
    io.check_table(out)
    out["geometry"] = ST.geometry_index(out)
    arr = {f: np.concatenate(v) for f, v in A.items()}
    return out, arr


def own_frame_problems(tab_own, frozen: dict) -> list[str]:
    """The own-run table's geometries and partition 0's folds against the record's
    (amendment 3 item 1)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    bad = []
    labels = AN.geometry_labels(tab_own)
    if labels.tolist() != frozen.get("geometries"):
        bad.append("the own-run table's geometries are not the frozen ones")
    else:
        fo = PB.folds(tab_own)
        g = ST.geometry_index(tab_own)
        want = np.asarray(frozen["fold_partition0"], int)[g]
        if not np.array_equal(fo.outer, want):
            bad.append("the own-run table's folds are not partition 0's")
    return bad


def c1_own_wiring(rec: dict, want: dict, ks=K_SX) -> dict:
    """{field: max |difference|} of a recomputed C1-own record against decisions.json's (r2c
    of s and xi, stat, ci, the per-fold a and b of s and xi; amendment 3 item 2)."""
    import numpy as np

    ks = list(ks)

    def pick(r):
        return {"r2c": [r["r2c"][k] for k in ks], "stat": [r["stat"]], "ci": list(r["ci"]),
                "a": np.asarray(r["calibration"]["a"], float)[:, ks].ravel(),
                "b": np.asarray(r["calibration"]["b"], float)[:, ks].ravel()}

    got, ref = pick(rec), pick(want)
    return {k: float(np.nanmax(np.abs(np.asarray(got[k], float) - np.asarray(ref[k], float))))
            for k in got}


def confirmatory_c1_own(fr, crit: dict, own: dict, counts, k2b: dict, k3b: bool,
                        va: bool) -> dict:
    """The confirmatory C1-own (scripts/costate.py's ``analyse_network``) on T's restarts,
    recomputed: the critic's g against lambda^own on the restarts with a valid run, each
    component's scored rows through the row guard (plan R5)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    tab = fr.tab
    lo = np.asarray(own["lam_own"], float)[:, fr.idx]
    ok = np.isfinite(lo).all(1) & np.asarray(own["valid"], bool)
    for c, k in zip(SX, K_SX, strict=True):
        guard_rows(tab, ok & fr.valid[:, k], c, k2b, fr.valid[:, k])
    return AN.c1_own(np.asarray(crit["g"])[:, 0, fr.idx], lo, fr.fo.outer,
                     ST.stratum_weights(tab, ok), fr.cl, fr.G, tab["direction"], ok, k3b,
                     TG.K_P, counts, fr.valid, value_accuracy_pass=va)


def own_c1_records(tab_own, arr: dict, rows, valid, frozen: dict, counts, k3b: bool,
                   va: bool) -> dict:
    """CS-10's C1-own on own-run states and its two companions (amendment 3 item 2)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    idx = [TG.STATE_NAMES.index(c) for c in TG.K_P]
    k2b = frozen["k2b_excluded"]
    guard_rows(tab_own, rows)
    for c, k in zip(SX, K_SX, strict=True):
        guard_rows(tab_own, rows & valid[:, k], c, k2b, valid[:, k])
    fo = PB.folds(tab_own)
    cl, G = AN.geometry_clusters(tab_own)
    w = ST.stratum_weights(tab_own, rows)
    d = np.asarray(tab_own["direction"])
    g, lam = arr["g"][:, idx], arr["lam"][:, idx]
    out = {"primary": AN.c1_own(g, lam, fo.outer, w, cl, G, d, rows, k3b, TG.K_P, counts, valid,
                                value_accuracy_pass=va)}
    bnd = CS._bounds(frozen, TG.K_P)
    lam_w = np.stack([ST.winsorise(lam[:, j], tuple(bnd[j])) for j in range(len(TG.K_P))], 1)
    c = AN.c1(g, lam_w, fo.outer, w, cl, G, d, rows, None, valid, TG.K_P, counts,
              slope_boot=False, bounds=bnd)
    out["winsorised"] = {**c.record(), "descriptive": True}
    held = AN.c1_own(arr["g_held"][:, idx], arr["lam_held"][:, idx], fo.outer, w, cl, G, d,
                     rows, k3b, TG.K_P, counts, valid, value_accuracy_pass=va)
    out["held"] = {**held, "descriptive": True, "status": "descriptive (the accel-held form)"}
    return out


def _brief(rec: dict) -> dict:
    """A C1 record's headline fields (s and xi)."""
    ks = list(K_SX)

    def at(v):
        return None if v is None else [v[k] for k in ks]

    return {"decision": rec.get("decision"), "stat": rec.get("stat"), "ci": rec.get("ci"),
            "r2c": at(rec.get("r2c")), "n_rows": at(rec.get("n_rows")),
            "spearman": at(rec.get("spearman")), "spearman_ci": at(rec.get("spearman_ci")),
            "slope": at(rec.get("slope")), "slope_gl": at(rec.get("slope_gl"))}


def cs10_expected(nets: dict) -> dict:
    """E10.1 and E10.2 (amendment 3 item 2)."""

    def width(ci):
        return float(ci[1] - ci[0]) if ci and None not in ci else float("nan")

    parts = {
        "refute": lambda v: v["c1_own"]["decision"] == "refute",
        "rows_x10": lambda v: all(
            v["c1_own"]["n_rows"][j] >= E10_ROWS * v["confirmatory"]["n_rows"][j]
            for j in range(len(SX))),
        "ci_half_as_wide": lambda v: (width(v["c1_own"]["ci"])
                                      <= E10_WIDTH * width(v["confirmatory"]["ci"])),
    }
    e1 = {k: _count_of(nets, fn) for k, fn in parts.items()}
    n = len(nets)
    e2 = _count_of(nets, lambda v: v["c1_own"]["spearman_ci"][SX.index("xi")][0] > 0)
    return {"E10.1": {**e1, "holds": all(p["count"] == n for p in e1.values())},
            "E10.2": {**e2, "holds": e2["count"] >= min(E_COUNT, n)},
            "ci_width_ratio": {k: float(width(v["c1_own"]["ci"]) / width(v["confirmatory"]["ci"]))
                               for k, v in nets.items()},
            "reading": "amendment 3 items 1 and 2", "n_networks": n}


def network_units(run: Run, labels, nkeys: int, quarters: int = OWN_QUARTERS) -> tuple:
    """(units, names, agents): one unit per (network, quarter of D's keys); each network's
    checkpoint through its pin (recorded)."""
    import numpy as np

    from apexzero.costate import analysis as AN

    units, names, agents = [], [], {}
    for lb in labels:
        name = AN.network_of(lb)
        pin = run.rec["networks"][name]
        agents[lb] = CS.load_agent(name, pin, run.root)
        run.note_input(pin["path"], pin["sha256"])
        for q, part in enumerate(np.array_split(np.arange(nkeys), quarters)):
            units.append({"label": lb, "quarter": q, "agent": agents[lb],
                          "keys": [int(i) for i in part]})
            names.append(f"{lb}/q{q}")
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, names = units[:int(lim)], names[:int(lim)]
    return units, names, agents


def decision_labels(run: Run) -> list[str]:
    from apexzero.costate import analysis as AN

    want = getattr(run.args, "network", None)
    labels = [lb for lb in CS1_ORDER if want is None or want in (lb, AN.network_of(lb))]
    if not labels:
        raise SystemExit(f"{run.item.id}: --network {want} is not a decision network")
    return labels


def _gather(units, got) -> dict:
    """{label: (key records in key order, restart records, seconds)} from the pool."""
    out: dict = {}
    for u, r in zip(units, got, strict=True):
        o = out.setdefault(u["label"], {"keys": [], "restart": [], "seconds": 0.0})
        o["keys"] += r["keys"]
        if r.get("restart") is not None:
            o["restart"].append(r["restart"])
        o["seconds"] += float(r["seconds"])
        for k in ("checks", "fd"):
            if r.get(k) is not None:
                o.setdefault(k, []).append(r[k])
    for o in out.values():
        o["keys"].sort(key=lambda r: r["key"])
    return out


@computes("CS-10")
def cs10(run: Run) -> None:
    """CS-10 (ii) (plan Section 4, amendment 3 items 1 and 2): per decision network the
    confirmatory C1-own recomputed first (wiring), then the pooled own runs (``cs10_unit``),
    the own-run table, the wiring of the pullbacks and restarts, C1-own on the own-run states
    with its companions, and the expected directions; outputs summary.json and rows.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import own as OW
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab, frozen = world.tab, run.rec["frozen"]
    keys = d_keys(world)
    G = AN.geometry_clusters(tab)[1]
    counts = PB.cluster_counts(G)
    labels = decision_labels(run)
    pre: dict = {}
    for lb in labels:
        name = AN.network_of(lb)
        f = PR.net_file(name)
        _agent, _flags, fr = network_frame(run, name, world)
        crit = read_critic(run, name, ("g", "g_held"))
        with np.load(run.read(f"post/own/{f}.npz")) as d:
            own = {k: d[k].copy() for k in ("lam_own", "valid", "k3b_pass")}
        k3b = json.loads(run.read(f"post/own/{f}.k3b.json").read_text())
        if not bool(own["k3b_pass"]) or not (k3b.get("decision") or {}).get("pass"):
            raise SystemExit(f"CS-10: {lb} failed K3b's trained analogue")
        dec = run.decisions["networks"][name]
        va = bool(dec["value_accuracy"]["pass"])
        conf = confirmatory_c1_own(fr, crit, own, counts, frozen["k2b_excluded"], True, va)
        wiring = c1_own_wiring(conf, dec["C1-own"])
        if not all(v <= WIRING_TOL for v in wiring.values()):
            raise SystemExit(f"CS-10: {lb}'s confirmatory C1-own does not reproduce "
                             f"decisions.json: {wiring}")
        d0 = np.asarray([kd["row"] for kd in keys])
        pre[lb] = {"name": name, "conf": conf, "wiring": wiring, "va": va,
                   "g_d": np.asarray(crit["g"])[d0, 0], "fr": fr}
    units, names, _ = network_units(run, labels, len(keys))
    print(f"CS-10: {len(units)} units over {len(labels)} networks, {len(keys)} D keys "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = _gather(units, run.pool(cs10_unit, units, state={"keys": keys}, names=names))
    pool_s = time.time() - t1
    nets, arrays = {}, {}
    for lb, o in got.items():
        p = pre[lb]
        recs = o["keys"]
        if [r["key"] for r in recs] != list(range(len(keys))) and getattr(
                run.args, "limit", None) is None:
            raise SystemExit(f"CS-10 {lb}: the pool did not return every D key")
        ks_keys = [keys[r["key"]] for r in recs]
        # wiring: the pullback at every kept run's state 0 against the stored g at D
        pb_err = 0.0
        for r in recs:
            if r["kept"]:
                want = p["g_d"][r["key"]]
                pb_err = max(pb_err, float(np.max(np.abs(np.asarray(r["g"])[0] - want))
                                           / max(float(np.max(np.abs(want))), 1e-300)))
        if not pb_err <= PULLBACK_TOL:
            raise SystemExit(f"CS-10 {lb}: the pullback at state 0 is not the stored g "
                             f"({pb_err:.3g})")
        rs_err = max((x["error"] for x in o["restart"]), default=float("nan"))
        if not rs_err <= OW.RESTART_RTOL:
            raise SystemExit(f"CS-10 {lb}: a restart does not reproduce the sweep ({rs_err:.3g})")
        tab_own, arr = own_table(ks_keys, recs, tab)
        bad = own_frame_problems(tab_own, frozen) if getattr(run.args, "limit", None) is None \
            else []
        if bad:
            raise SystemExit(f"CS-10 {lb}: {'; '.join(bad)}")
        phase = np.asarray(tab_own["phase"]).astype(str)
        fin = np.isfinite(arr["lam"]).all(1) & np.isfinite(arr["g"]).all(1)
        rows = (np.asarray(tab_own["kept"], bool) & np.asarray(tab_own["interior"], bool)
                & fin & np.isin(phase, SCORED_PHASES))
        valid = ST.decision_mask(tab_own, ~arr["s_unobservable"], frozen["k2b_excluded"])
        recs_c1 = own_c1_records(tab_own, arr, rows, valid, frozen, counts, True, p["va"])
        kept = np.array([r["kept"] for r in recs])
        H = np.array([r["H"] for r in recs])
        term = np.array([r["terminated"] for r in recs])
        se = np.array([r["strip_end"] for r in recs])
        te = np.array([r["t_end"] for r in recs])
        sh = np.array([r.get("interior_share", np.nan) for r in recs])
        on_d = np.isin(phase, SCORED_PHASES) & np.asarray(tab_own["kept"], bool)
        w_x = ST.stratum_weights(tab_own, rows & valid[:, K_SX[1]])
        fr = p["fr"]
        tx = fr.inner & fr.valid[:, K_SX[1]]
        desc = {
            "runs": {"keys": len(recs), "kept": int(kept.sum()), "terminated": int(term.sum()),
                     "strip_end": int((se & kept).sum()),
                     "t_end_not_reached": int((kept & (te < 0)).sum()),
                     "H": {"min": int(H[kept].min()) if kept.any() else None,
                           "median": float(np.median(H[kept])) if kept.any() else None,
                           "max": int(H[kept].max()) if kept.any() else None},
                     "interior_share_median": float(np.nanmedian(sh))},
            "states": {"all": int(np.asarray(tab_own["kept"], bool).sum()),
                       "by_phase": {ph: int((np.asarray(tab_own["kept"], bool)
                                             & (phase == ph)).sum()) for ph in ST.PHASES},
                       "interior_by_phase": {ph: int((np.asarray(tab_own["interior"], bool)
                                                      & (phase == ph)).sum())
                                             for ph in ST.PHASES},
                       "scored_rows": int(rows.sum()),
                       "contact_rows_scored_phases": int((arr["contact"] & on_d).sum()),
                       "s_unobservable_rows": int((rows & arr["s_unobservable"]).sum())},
            "xi_rows_vs_T": {
                "own_ds_apex_median": float(AN.wmedian(
                    (tab_own["s"] - tab_own["s_apex"])[w_x > 0], w_x[w_x > 0])),
                "T_ds_apex_median": float(AN.wmedian(np.asarray(tab["ds_apex"])[tx],
                                                     fr.w_in[tx])),
                "own_vx_median": float(AN.wmedian(tab_own["vx"][w_x > 0], w_x[w_x > 0])),
                "T_vx_median": float(AN.wmedian(np.asarray(tab["vx"])[tx], fr.w_in[tx]))}}
        nets[lb] = {"run": p["name"], "c1_own": _brief(recs_c1["primary"]),
                    "c1_own_record": recs_c1["primary"],
                    "winsorised": _brief(recs_c1["winsorised"]),
                    "held": _brief(recs_c1["held"]),
                    "confirmatory": _brief(p["conf"]),
                    "wiring": {"confirmatory_c1_own": p["wiring"], "pullback_state0": pb_err,
                               "restart": rs_err, "restart_keys": o["restart"]},
                    **desc, "seconds": o["seconds"]}
        arrays.update({f"{lb}/{k}": np.asarray(v) for k, v in tab_own.items()
                       if k in ("k", "s", "n", "vx", "kept", "interior", "geometry", "W",
                                "direction", "t_end_run")})
        arrays[f"{lb}/key"] = np.repeat(np.arange(len(recs)), [len(r["ks"]) for r in recs])
        arrays[f"{lb}/phase"] = phase
        arrays[f"{lb}/rows"] = rows
        arrays[f"{lb}/valid"] = valid
        for k, v in arr.items():
            arrays[f"{lb}/{k}"] = v
    out = {"item": "CS-10", "part": "(ii) on D's own runs; (iii) on N waits for CS-9",
           "networks": nets, "expected": cs10_expected(nets), "steps": OWN_STEPS,
           "every": ST.EVERY, "interior_k": ST.INTERIOR_K, "components": list(SX),
           "definitions": "plan Section 4 CS-10 and amendment 3 items 1 and 2",
           "n_boot": len(counts), "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_npz("rows", arrays)
    run.write_json("summary", plain(out))


@functools.lru_cache(maxsize=8)
def _width_vjp_jit(fn, steps: int):
    """jit((params, obs_rms, base, samples, li, d, count_W, count_H) -> ([J_W, J_H], ([J_env,W,
    J_env,H], x [steps+1, 8], valid [steps+1]), dJ_W/dd, dJ_H/dd)): ``own.own_loop`` on the
    road perturbed by the samples' tangent along d (amendment 3 item 3), one VJP, two
    reverse sweeps."""
    import jax
    import jax.numpy as jnp

    from apexzero.costate import own as OW
    from apexzero.fields import perturb as PT

    def run(params, obs_rms, base, samples, li, d0, c_w, c_h):
        def f(d):
            tr = PT.perturbed_arrays(base, samples, d)
            z = jnp.zeros((steps, 8), li.x0.dtype)
            jw, jew, _ = OW.own_loop(fn, params, obs_rms, tr, li, steps, z, c_w)
            jh, jeh, outs = OW.own_loop(fn, params, obs_rms, tr, li, steps, z, c_h)
            return jnp.stack([jw, jh]), (jnp.stack([jew, jeh]), outs["x"], outs["valid"])

        J, vjp, aux = jax.vjp(f, d0, has_aux=True)
        one = jnp.eye(2, dtype=J.dtype)
        return J, aux, vjp(one[0])[0], vjp(one[1])[0]

    return jax.jit(run)


@functools.lru_cache(maxsize=8)
def _width_jvp_jit(fn, steps: int):
    """jit(... -> d[J_W, J_H] along a tangent of the samples): the JVP the VJP is checked
    against (amendment 3 item 3)."""
    import jax
    import jax.numpy as jnp

    from apexzero.costate import own as OW
    from apexzero.fields import perturb as PT

    def run(params, obs_rms, base, samples, li, d0, tangent, c_w, c_h):
        def f(d):
            tr = PT.perturbed_arrays(base, samples, d)
            z = jnp.zeros((steps, 8), li.x0.dtype)
            jw = OW.own_loop(fn, params, obs_rms, tr, li, steps, z, c_w)[0]
            jh = OW.own_loop(fn, params, obs_rms, tr, li, steps, z, c_h)[0]
            return jnp.stack([jw, jh])

        return jax.jvp(f, (d0,), (tangent,))[1]

    return jax.jit(run)


@functools.lru_cache(maxsize=8)
def _width_secant_jit(fn, steps: int):
    """jit(... -> (J, J_env, valid)) on the road plus the secant of the samples along d
    (the descriptive Richardson differences, amendment 3 item 3)."""
    import jax
    import jax.numpy as jnp

    from apexzero.costate import own as OW
    from apexzero.fields import perturb as PT

    def run(params, obs_rms, base, samples, li, d, count):
        tr = PT.perturbed_arrays(base, samples, d, mode="secant")
        z = jnp.zeros((steps, 8), li.x0.dtype)
        J, Je, outs = OW.own_loop(fn, params, obs_rms, tr, li, steps, z, count)
        return J, Je, outs["valid"]

    return jax.jit(run)


def _scaled(sm, c: float):
    from apexzero.fields import perturb as PT

    return PT.Samples(*(c * a for a in sm))


def inside_field(field, sigma: int):
    """The inside width part of a samples field (w_left on a left corner, w_right on a right
    one): its sum is the field contracted with the uniform inside widening
    (``inputs.inside_widening``), Pi^in,own for dJ/dd."""
    import numpy as np

    return np.asarray(field.w_left if int(sigma) > 0 else field.w_right, float)


def width_vjp(net, rd, li, c_w: int, c_h: int, steps: int = OWN_STEPS) -> dict:
    """Pi^in,own at counts W and H of one run (amendment 3 item 3): J, J_env, the run's
    states and validity, and the inside-width fields dJ/dw_in(s_j) at both counts."""
    import jax.numpy as jnp
    import numpy as np

    from apexzero.costate import own as OW
    from apexzero.fields import network as NW
    from apexzero.fields import perturb as PT

    NW._refuse_trained(net, "an own width price")
    NW.require_x64()
    li = OW.inputs_inf(li)
    f = _width_vjp_jit(net.fn, int(steps))
    J, (Je, x, valid), gw, gh = f(net.params, net.obs_rms, rd.base, rd.samples, li,
                                   PT.zeros(rd.n), jnp.int32(c_w), jnp.int32(c_h))
    return {"J": np.asarray(J), "J_env": np.asarray(Je), "x": np.asarray(x),
            "valid": np.asarray(valid), "field_W": gw, "field_H": gh}


def width_jvp(net, rd, li, sigma: int, c_w: int, c_h: int, steps: int = OWN_STEPS):
    """d[J_W, J_H]/dDelta along the uniform inside widening (the VJP's check)."""
    import jax.numpy as jnp
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW
    from apexzero.fields import perturb as PT

    f = _width_jvp_jit(net.fn, int(steps))
    return np.asarray(f(net.params, net.obs_rms, rd.base, rd.samples, OW.inputs_inf(li),
                        PT.zeros(rd.n), IN.inside_widening(rd.n, int(sigma)), jnp.int32(c_w),
                        jnp.int32(c_h)))


def width_fd(net, rd, li, sigma: int, count: int, h: float = FD_STEP,
             steps: int = OWN_STEPS) -> dict:
    """Descriptive: Richardson central differences (h, h/2) of J and J_env along the uniform
    inside widening (secant), at ``count`` steps (amendment 3 item 3)."""
    import jax.numpy as jnp
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW

    f = _width_secant_jit(net.fn, int(steps))
    tang = IN.inside_widening(rd.n, int(sigma))
    li = OW.inputs_inf(li)
    vals, alive = {}, True
    for c in (-1.0, -0.5, 0.5, 1.0):
        J, Je, v = f(net.params, net.obs_rms, rd.base, rd.samples, li, _scaled(tang, c * h),
                     jnp.int32(count))
        vals[c] = (float(J), float(Je))
        alive &= bool(np.asarray(v)[:count + 1].all())
    out = {"h": h, "alive": alive}
    for j, nm in enumerate(("J", "J_env")):
        f1 = (vals[1.0][j] - vals[-1.0][j]) / (2 * h)
        f2 = (vals[0.5][j] - vals[-0.5][j]) / h
        out[nm] = {"fd_h": f1, "fd_h2": f2, "richardson": (4 * f2 - f1) / 3}
    return out


def buffer_of(x, lm, sigma: int, tr, upto: int) -> tuple[float, str]:
    """(B, why invalid): step 2's c_min (``outcomes.c_min_plan``) on a run's states 0..upto;
    NaN with the reason when the run is not alive through s_c1 + 50 m there."""
    import numpy as np

    from apexzero.fields import outcomes as O

    pl = O.c_min_plan(O.trajectory(np.asarray(x, float)[:upto + 1]), lm, int(sigma), tr)
    return (float(pl.value), "") if pl.valid else (float("nan"), pl.reason)


def drop_reason(kept: bool, alive: bool, why_b: str) -> str:
    """Why a run is not kept at a horizon (amendment 3 item 3): '' when kept."""
    if kept:
        return ""
    if not alive:
        return "not alive through the horizon"
    return why_b or "B <= 0"


def cs11_unit(state, unit) -> dict:
    """One (network, quarter of D's keys) of CS-11 in a worker (amendment 3 item 3): per key
    the critic's dV_dwin recomputed at D, the run's horizon, Pi^in,own at W and H with the
    run's buffers, the J check; the JVP check and the descriptive differences on the unit's
    first kept key."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.fields import network as NW

    t0 = time.time()
    net, crit = own_nets(unit["agent"])
    bank = _bank()
    out, checks, fd = [], None, None
    for i in unit["keys"]:
        kd = state["keys"][i]
        sig, W = kd["direction"], kd["W"]
        rd = NW.road(bank, kd["stim_id"])
        lm = landmarks_of(bank, kd)
        x0 = np.asarray(kd["x0"], float)
        inp = CR.Inputs(x0[None], x0[None], np.zeros((1, 2)), np.array([True]))
        rec = {"key": int(i), "dV_dwin": float(IN.width_derivative(crit, rd, inp, sig)[0])}
        rec["dv_gap"] = abs(rec["dV_dwin"] - unit["dv"][str(i)]) / unit["dv_scale"]
        if not rec["dv_gap"] <= DV_TOL:                       # amendment 5: stop at once
            raise RuntimeError(f"key {kd['key_id']}: dV_dwin recomputed is not the stored one "
                               f"({rec['dv_gap']:.3g})")
        li = own_start(x0)
        rec.update(own_horizon(net, rd.base, li))
        H = rec["H"]
        if rec["terminated"]:
            rec.update(kept_W=False, kept_H=False, why_W="terminated", why_H="terminated")
            out.append(rec)
            continue
        cw = min(W, H)
        r = width_vjp(net, rd, li, cw, H)
        valid = r["valid"]
        alive_w = bool(H >= W and valid[:W + 1].all())
        b_w, why_w = buffer_of(r["x"], lm, sig, rd.base, W)
        b_h, why_h = buffer_of(r["x"], lm, sig, rd.base, H)
        kept_w = alive_w and np.isfinite(b_w) and b_w > 0
        kept_h = bool(valid[:H + 1].all()) and np.isfinite(b_h) and b_h > 0
        fw, fh = inside_field(r["field_W"], sig), inside_field(r["field_H"], sig)
        rec.update(J_W=float(r["J"][0]), J_H=float(r["J"][1]), J_env_W=float(r["J_env"][0]),
                   J_env_H=float(r["J_env"][1]), B_W=b_w, B_H=b_h, kept_W=bool(kept_w),
                   kept_H=bool(kept_h), pi_own_W=float(fw.sum()), pi_own_H=float(fh.sum()),
                   field_W=fw.astype(np.float32), field_H=fh.astype(np.float32),
                   why_W=drop_reason(kept_w, alive_w, why_w),
                   why_H=drop_reason(kept_h, bool(valid[:H + 1].all()), why_h),
                   j_gap=abs(float(r["J"][1]) - rec["J_return"])
                   / max(abs(rec["J_return"]), 1e-300))
        if not rec["j_gap"] <= J_TOL:
            raise RuntimeError(f"key {kd['key_id']}: the width loop's J is not own_return's "
                               f"({rec['j_gap']:.3g})")
        if checks is None and kept_w and kept_h:
            jv = width_jvp(net, rd, li, sig, cw, H)
            got = np.array([rec["pi_own_W"], rec["pi_own_H"]])
            checks = {"key": int(i), "jvp": jv.tolist(), "vjp": got.tolist(),
                      "error": float(np.max(np.abs(jv - got))
                                     / max(float(np.max(np.abs(jv))), 1e-300))}
            if not checks["error"] <= JVP_TOL:
                raise RuntimeError(f"key {kd['key_id']}: the width VJP is not the JVP "
                                   f"({checks['error']:.3g})")
            fd = {"key": int(i), "ad": rec["pi_own_W"], **width_fd(net, rd, li, sig, cw)}
            ref = fd["J_env"]["richardson"]
            fd["error_J_env"] = abs(fd["ad"] - ref) / max(abs(ref), 1e-300)
        out.append(rec)
    return {"label": unit["label"], "keys": out, "checks": checks, "fd": fd,
            "seconds": time.time() - t0}


def agreement(x, target, rows, w, outer, cl, G: int, counts, bounds=None) -> dict:
    """W1's comparison (amendment 3 item 3): ``analysis.calibrated_agreement`` with the pooled
    slopes b_lg (target on x) and b_gl (x on target) and their CIs on the frozen counts."""
    import numpy as np

    from apexzero.costate import analysis as AN

    rec = AN.calibrated_agreement(x, target, outer, w, cl, G, rows, counts, bounds=bounds)
    sel = np.asarray(rows, bool) & np.isfinite(x) & np.isfinite(target)
    if sel.sum() >= 3:
        ps = AN.pooled_slopes(np.asarray(x, float)[sel], np.asarray(target, float)[sel],
                              w[sel], cl[sel], counts)
        rec.update(b_lg=ps["lg"], b_lg_ci=list(AN._ci(ps["lg_draws"])), b_gl=ps["gl"],
                   b_gl_ci=list(AN._ci(ps["gl_draws"])))
    return rec


def cs11_expected(nets: dict) -> dict:
    """E11.1 to E11.3 (amendment 3 item 3), on the common rows."""
    n = len(nets)

    def c(v, k):
        return v["common"][k]

    e1 = _count_of(nets, lambda v: c(v, "critic_own")["r2c"] > c(v, "critic_optimum")["r2c"])
    e2 = _count_of(nets, lambda v: (c(v, "own_optimum")["b_gl"] < 1.0
                                    and c(v, "own_optimum")["b_gl_ci"][1] < 1.0))
    e3 = _count_of(nets, lambda v: c(v, "own_optimum")["r2c"] < E11_R2)
    need = min(E_COUNT, n)
    return {"E11.1": {**e1, "holds": e1["count"] >= need},
            "E11.2": {**e2, "holds": e2["count"] >= need},
            "E11.3": {**e3, "holds": e3["count"] >= need},
            "reading": "amendment 3 items 1 and 3", "n_networks": n}


@computes("CS-11")
def cs11(run: Run) -> None:
    """CS-11 on D (plan Section 4, amendment 3 items 1 and 3): per decision network the
    pooled width VJPs (``cs11_unit``), the wiring of dV_dwin, J and the JVP, W1's three
    comparisons on the common rows with their companions, and the expected directions;
    outputs summary.json and rows.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab, frozen = world.tab, run.rec["frozen"]
    keys = d_keys(world)
    d0 = np.asarray([kd["row"] for kd in keys])
    fo = PB.folds(tab)
    cl, G = AN.geometry_clusters(tab)
    counts = PB.cluster_counts(G)
    bnd = [frozen["winsor_bounds"]["Pi_in"]]
    pi_in = np.asarray(tab["Pi_in"], float)
    labels = decision_labels(run)
    stored = {lb: read_critic(run, AN.network_of(lb), ("dV_dwin",))["dV_dwin"] for lb in labels}
    units, names, _ = network_units(run, labels, len(keys))
    scale = {lb: float(np.max(np.abs(v[d0]))) for lb, v in stored.items()}
    for u in units:
        u["dv"] = {str(i): float(stored[u["label"]][d0[i]]) for i in u["keys"]}
        u["dv_scale"] = scale[u["label"]]
    print(f"CS-11: {len(units)} units over {len(labels)} networks, {len(keys)} D keys "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = _gather(units, run.pool(cs11_unit, units, state={"keys": keys}, names=names))
    pool_s = time.time() - t1
    nets, arrays = {}, {}
    for lb, o in got.items():
        recs = o["keys"]
        if [r["key"] for r in recs] != list(range(len(keys))) and getattr(
                run.args, "limit", None) is None:
            raise SystemExit(f"CS-11 {lb}: the pool did not return every D key")
        idx = d0[[r["key"] for r in recs]]
        dv = stored[lb]
        dv_err = max(abs(r["dV_dwin"] - dv[i]) / scale[lb]
                     for r, i in zip(recs, idx, strict=True))
        if not dv_err <= DV_TOL:
            raise SystemExit(f"CS-11 {lb}: dV_dwin recomputed is not the stored one "
                             f"({dv_err:.3g})")
        j_gap = max((r["j_gap"] for r in recs if "j_gap" in r), default=float("nan"))
        if not j_gap <= J_TOL:
            raise SystemExit(f"CS-11 {lb}: the width loop's J is not own_return's ({j_gap:.3g})")
        jvp = max((c["error"] for c in o.get("checks", [])), default=float("nan"))
        if not jvp <= JVP_TOL:
            raise SystemExit(f"CS-11 {lb}: the width VJP is not the JVP ({jvp:.3g})")

        def col(k, default=np.nan, recs=recs, idx=idx):
            v = np.full(world.n, default, float)
            v[idx] = [r.get(k, default) for r in recs]
            return v

        kw, kh = col("kept_W", 0.0) > 0, col("kept_H", 0.0) > 0
        own_w, own_h = col("pi_own_W"), col("pi_own_H")
        sets = {"common": kw & kh, "kept_H": kh, "kept_W": kw}
        per = {}
        for nm, rows in sets.items():
            guard_rows(tab, rows)
            w = ST.stratum_weights(tab, rows)
            per[nm] = {"n": int(rows.sum())}
            if nm in ("common", "kept_H"):
                per[nm]["critic_own"] = agreement(dv, own_h, rows, w, fo.outer, cl, G, counts)
            if nm in ("common", "kept_W"):
                per[nm]["own_optimum"] = agreement(own_w, pi_in, rows, w, fo.outer, cl, G,
                                                   counts, bnd)
            if nm == "common":
                per[nm]["critic_optimum"] = agreement(dv, pi_in, rows, w, fo.outer, cl, G,
                                                      counts, bnd)
                per[nm]["medians"] = {k: float(AN.wmedian(v[rows], w[rows])) for k, v in (
                    ("pi_own_W", own_w), ("pi_own_H", own_h), ("Pi_in", pi_in),
                    ("dV_dwin", dv))}
                per[nm]["zero_share"] = {"pi_own_W": float(np.mean(own_w[rows] == 0.0)),
                                         "pi_own_H": float(np.mean(own_h[rows] == 0.0))}
        why = {}
        for h in ("W", "H"):
            c = {}
            for r in recs:
                if not r.get(f"kept_{h}"):
                    c[r.get(f"why_{h}") or "?"] = c.get(r.get(f"why_{h}") or "?", 0) + 1
            why[h] = c
        nets[lb] = {"run": AN.network_of(lb), **per,
                    "runs": {"keys": len(recs), "terminated": int(sum(r["terminated"]
                                                                      for r in recs)),
                             "kept_W": int(kw.sum()), "kept_H": int(kh.sum()),
                             "strip_end": int(sum(r["strip_end"] for r in recs)),
                             "dropped": why},
                    "B": {h: float(np.nanmedian([r.get(f"B_{h}", np.nan) for r in recs]))
                          for h in ("W", "H")},
                    "wiring": {"dV_dwin": dv_err, "dV_dwin_scale": scale[lb], "J": j_gap,
                               "jvp": jvp,
                               "jvp_keys": o.get("checks", [])},
                    "fd_descriptive": o.get("fd", []), "seconds": o["seconds"]}
        m = max(len(r.get("field_W", [])) for r in recs)
        fw = np.full((len(recs), m), np.nan, np.float32)
        fh = np.full((len(recs), m), np.nan, np.float32)
        for j, r in enumerate(recs):
            if "field_W" in r:
                fw[j, :len(r["field_W"])], fh[j, :len(r["field_H"])] = r["field_W"], r["field_H"]
        arrays.update({f"{lb}/row": idx, f"{lb}/kept_W": kw[idx], f"{lb}/kept_H": kh[idx],
                       f"{lb}/pi_own_W": own_w[idx], f"{lb}/pi_own_H": own_h[idx],
                       f"{lb}/dV_dwin": dv[idx], f"{lb}/Pi_in": pi_in[idx],
                       f"{lb}/H": col("H")[idx], f"{lb}/B_W": col("B_W")[idx],
                       f"{lb}/B_H": col("B_H")[idx], f"{lb}/J_W": col("J_W")[idx],
                       f"{lb}/J_H": col("J_H")[idx], f"{lb}/field_W": fw,
                       f"{lb}/field_H": fh})
    out = {"item": "CS-11", "part": "on D; N's CS-11 waits for CS-9", "networks": nets,
           "expected": cs11_expected(nets), "steps": OWN_STEPS, "pi_in_bounds": bnd[0],
           "definitions": "plan Section 4 CS-11 and amendment 3 items 1 and 3",
           "n_boot": len(counts), "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_npz("rows", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-7, CS-8: shared
#
# Plan Section 4 CS-7 and CS-8 with amendment 7 (their operational details). Both read the 9
# decision networks' bound activations and compute random-init networks only, so neither
# unlocks. The frozen lambda_res (``frozen_lam_res``) is ``probes_of``'s, checked against the
# record's interior residual shares before anything else (amendment 7 item 1).

RES_TOL = 1e-10                          # the frozen lambda_res's shares against the record
FIT_TOL = 1e-12                          # the cross-fit scheme against states.cross_fit


def frozen_lam_res(tab, frozen: dict) -> tuple:
    """(lam [n, 4] the winsorised K_P targets, lambda_res by fold [F, n, 4], each row's
    own-fold residual [n, 4], partition 0's folds): ``probes_of``'s frozen f_D cross-fit on
    every row of T (targets only; amendment 7 item 1)."""
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    lam = CS._winsorised(tab, frozen, TG.K_P)
    fo = PB.folds(tab)
    k = PR.decoy_k(frozen)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, ST.weights(tab, "key"),
                       k=ST.KNN_K if k is None else k)
    return lam, res.by_fold, res.oof, fo


def residual_share_gap(tab, lam, oof, frozen: dict) -> dict:
    """{component: |the interior residual share of ``oof`` - the record's|}
    (``states.decoy_shares``; the record's ``frozen.residual_share_interior``)."""
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    _, inner = ST.decoy_shares(tab, lam, oof)
    want = frozen["residual_share_interior"]
    return {c: abs(float(inner[i]) - float(want[c])) for i, c in enumerate(TG.K_P)}


def checked_lam_res(run: Run, tab) -> tuple:
    """``frozen_lam_res`` and its gaps, after its wiring check: the item stops on a gap above
    ``RES_TOL`` (amendment 7 item 1)."""
    frozen = run.rec["frozen"]
    lam, by_fold, oof, fo = frozen_lam_res(tab, frozen)
    gap = residual_share_gap(tab, lam, oof, frozen)
    if not all(v <= RES_TOL for v in gap.values()):
        raise SystemExit(f"{run.item.id}: the frozen lambda_res does not reproduce the record's "
                         f"interior residual shares: {gap}")
    return lam, by_fold, oof, fo, gap


def pool_actor_gap(cfg, z, stored) -> float:
    """max |h3 of ``probes.random_actor(cfg, 0)`` on z - the stored ``random0_h3``| (amendment
    7 item 1: the pool actor's rebuild)."""
    import numpy as np

    from apexzero.costate import probes as PB

    return float(np.max(np.abs(PB.hidden(PB.random_actor(cfg, 0), z)["h3"] - stored)))


def control_draws(dr_full, dr_controls, dr_base, pool) -> tuple:
    """``control_gain``'s statistic with its two-level and geometry-only draws [B], so that a
    mean over components can be taken on the same draws."""
    import numpy as np

    gt = np.asarray(dr_full) - np.asarray(dr_base)
    gc = np.asarray(dr_controls) - np.asarray(dr_base)[None]
    B = len(gt) - 1
    two = gt[1:] - np.median(gc[:, 1:][np.asarray(pool), np.arange(B)[:, None]], axis=1)
    geo = gt[1:] - np.median(gc[:, 1:], axis=0)
    return float(gt[0] - np.median(gc[:, 0])), two, geo


def mean_of_draws(parts) -> dict:
    """{stat, ci, ci_geometry} of the mean of several ``control_draws`` on the same draws."""
    import numpy as np

    from apexzero.costate import analysis as AN

    stat = float(np.mean([p[0] for p in parts]))
    two = np.mean([p[1] for p in parts], axis=0)
    geo = np.mean([p[2] for p in parts], axis=0)
    return {"stat": stat, "ci": list(AN._ci(two)), "ci_geometry": list(AN._ci(geo))}


def trim_logo(rec: dict, focus: int = 6, top: int = 3) -> dict:
    """A C2 or C2-sw record with its leave-one-geometry-out sets summarised: the sets'
    decisions counted, the most influential geometry, and each component's shares of the
    weight and of the base's squared error at geometry ``focus`` and at the ``top`` largest
    (the full sets stay in no output: the summary would carry 48 per record)."""
    import math

    lg = rec.get("logo") if isinstance(rec, dict) else None
    if not isinstance(lg, dict):
        return rec
    out = {k: v for k, v in rec.items() if k != "logo"}
    ids = [int(g) for g in (lg.get("geometry") or [])]
    dec: dict = {}
    for d in lg.get("decision") or []:
        dec[str(d)] = dec.get(str(d), 0) + 1

    def key(v):
        return float(v) if v is not None and math.isfinite(float(v)) else -math.inf

    shares = {}
    for comp, sh in (lg.get("shares") or {}).items():
        sse, wt = list(sh.get("sse_base") or []), list(sh.get("weight") or [])
        at = ids.index(focus) if focus in ids else None
        order = sorted(range(len(sse)), key=lambda i: -key(sse[i]))[:top]
        shares[comp] = {
            f"geometry{focus}": None if at is None else {"weight": wt[at], "sse_base": sse[at]},
            "top_sse_base": [{"geometry": ids[i], "sse_base": sse[i], "weight": wt[i]}
                             for i in order]}
    out["logo"] = {"kind": lg.get("kind"), "n_sets": len(ids), "decisions": dec,
                   "most_influential": lg.get("most_influential"), "shares": shares}
    return out


def ci_width(ci) -> float:
    return float(ci[1]) - float(ci[0])


# ------------------------------------------------------------------------- CS-7: critic layers
#
# Plan Section 4 CS-7 with amendment 7 item 2, on the 9 decision networks and C1's rows per
# component: ridge probes of lambda_k and of the frozen lambda_res,k on z and on z + each of
# the critic's h1, h2, h3 and head units (320), the trained gain against the median of 20
# random-init critics' same layer, and the actor's h3 against the 20 pool actors as the
# comparator. Pooled: per (network, component) its 106 fits in 4 chunks (``cs7_chunks``).

CS7_LAYERS = ("h1", "h2", "h3", "heads")
CS7_TARGETS = ("lam", "res")             # lambda_k winsorised; the frozen lambda_res,k
CS7_ACTOR = "actor_h3"
E71_GAIN = 0.05


def heads_block(u):
    """Head units [E, n, 64] as one block [n, 64 E], in head order (amendment 7 item 2)."""
    import numpy as np

    return np.concatenate(list(np.asarray(u)), axis=1)


def cs7_chunks() -> list[list[tuple]]:
    """The 106 probe fits of one (network, component) in 4 chunks of similar cost: the base,
    the trained critic's 4 layers, the actor's h3 and the 20 pool actors; then the 20
    random-init critics (4 layers each) in three runs of seeds."""
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB

    first = [("base",), *[("critic", L) for L in CS7_LAYERS], ("actor",),
             *[("pool", int(s)) for s in PB.POOL_SEEDS]]
    rc = [("random", int(s)) for s in CR.RANDOM_SEEDS]
    return [first, rc[:7], rc[7:14], rc[14:]]


def cs7_keys() -> list[str]:
    """Every prediction set of one (network, component), in the scoring order: the base, the
    trained critic's layers, the actor's h3, the random critics' layers, the pool actors."""
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB

    return ["base", *[f"critic_{L}" for L in CS7_LAYERS], "actor",
            *[f"random{s}_{L}" for s in CR.RANDOM_SEEDS for L in CS7_LAYERS],
            *[f"pool{s}" for s in PB.POOL_SEEDS]]


def cs7_unit(state, unit) -> dict:
    """One chunk of one (network, component)'s CS-7 probes in a worker (``cs7_chunks``): the
    ridge probe on z and on z + each layer of the chunk, out of fold, both targets per call.
    Random critics are built from the network's PPOConfig on its z; the pool actor of seed 0
    and the random critic of seed 0 are checked where the chunk reads them."""
    import numpy as np

    from apexzero.costate import probes as PB

    t0 = time.time()
    idx, cfg, jobs, comp = unit["idx"], unit["cfg"], unit["jobs"], unit["comp"]
    kinds = {j[0] for j in jobs}
    with np.load(unit["path"]) as d:
        z = np.clip(d["z"][idx], -10.0, 10.0)
        trained = {}
        if "critic" in kinds:
            trained = {L: d[f"critic_{L}"][idx] for L in ("h1", "h2", "h3")}
            trained["heads"] = heads_block(d["head_units"][:, idx])
        if "actor" in kinds:
            trained["actor"] = d["actor_h3"][idx]
        pool = {j[1]: d[f"random{j[1]}_h3"][idx] for j in jobs if j[0] == "pool"}
    Y, fo, w = unit["Y"], unit["fo"], unit["w"]
    names = (f"lam_{comp}", f"res_{comp}")
    preds: dict = {}
    checks: dict = {}
    y = [None]
    tag = f"CS-7 {unit['label']}/{comp}/{unit['chunk']}"

    def fit(key: str, h) -> None:
        p = PB.ridge_cv(z, Y, fo, w, h, names)
        if y[0] is None:
            y[0] = p.y
        elif not np.array_equal(y[0], p.y, equal_nan=True):
            raise RuntimeError(f"{tag}: the probes were not fitted on the same targets")
        preds[key] = p.pred

    if 0 in pool:
        checks["random0_gap"] = pool_actor_gap(cfg, z, pool[0])
        if not checks["random0_gap"] <= RANDOM_TOL:
            raise RuntimeError(f"{tag}: the pool actor of seed 0 does not rebuild its stored h3 "
                               f"({checks['random0_gap']:.3g})")
    for job in jobs:
        kind = job[0]
        if kind == "base":
            fit("base", None)
        elif kind == "critic":
            fit(f"critic_{job[1]}", trained[job[1]])
        elif kind == "actor":
            fit("actor", trained["actor"])
        elif kind == "pool":
            fit(f"pool{job[1]}", pool[job[1]])
        elif kind == "random":
            s = int(job[1])
            net = PB.random_critic(cfg, s)
            if s == 0:
                gap = PB.extractor_gap(PB.random_actor(cfg, 0), net, z[:64])
                checks["extractor_seed0"] = gap
                worst = max(gap["actor_mean"], gap["critic_v"], gap.get("critic_heads", np.inf))
                if not worst <= PB.EXTRACTOR_TOL:
                    raise RuntimeError(f"{tag}: the random critic of seed 0 does not match "
                                       f"Critic.apply ({worst:.3g})")
            hid = PB.hidden(net, z)
            if hid.u is None or hid.u.shape[0] != PB.N_HEADS_GENERALIST:
                raise RuntimeError(f"{tag}: the random critic of seed {s} has no 5 heads")
            for L in ("h1", "h2", "h3"):
                fit(f"random{s}_{L}", hid[L])
            fit(f"random{s}_heads", heads_block(hid.u))
        else:
            raise ValueError(f"{tag}: unknown job {job}")
    return {"preds": preds, "y": y[0], "checks": checks, "seconds": time.time() - t0}


def cs7_component(preds: dict, y, w, cl, G: int, counts, pool, direction=None,
                  odd: bool = False) -> tuple[dict, dict]:
    """(record, draws) of one (network, component): per target, R2(z) and per layer the
    trained gain against the controls (``control_gain``: the critic's layers against the
    random critics' same layer, the actor's h3 against the pool actors), with R2(z + h);
    ``draws`` {(target, layer): ``control_draws``} for the mean over components."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import probes as PB

    keys = cs7_keys()
    missing = [k for k in keys if k not in preds]
    if missing:
        raise ValueError(f"CS-7: prediction sets missing: {missing[:5]}")
    rec, draws = {}, {}
    for t, tname in enumerate(CS7_TARGETS):
        P = [np.asarray(preds[k])[:, t] for k in keys]
        if not all(np.isfinite(p).all() for p in P):
            raise ValueError(f"CS-7: rows without an out-of-fold prediction ({tname})")
        pt, dr = AN.boot_r2(np.asarray(y)[:, t], P, w, cl, G, counts, direction, odd)
        allr = np.vstack([pt[None], dr])                                 # [1 + B, K]
        col = {k: allr[:, i] for i, k in enumerate(keys)}
        out = {"r2_base": {"r2": float(pt[0]), "ci": list(AN._ci(dr[:, 0]))}, "layers": {}}
        sets = [(L, f"critic_{L}", [f"random{s}_{L}" for s in CR.RANDOM_SEEDS])
                for L in CS7_LAYERS]
        sets.append((CS7_ACTOR, "actor", [f"pool{s}" for s in PB.POOL_SEEDS]))
        for L, full, ctl_keys in sets:
            ctl = np.stack([col[k] for k in ctl_keys])
            g = control_gain(col[full], ctl, col["base"], pool)
            g["r2_full"] = {"r2": float(col[full][0]), "ci": list(AN._ci(col[full][1:]))}
            g["controls"] = ("random-init critics 0-19" if L != CS7_ACTOR
                             else "the pool actors 0-19")
            out["layers"][L] = g
            draws[(tname, L)] = control_draws(col[full], ctl, col["base"], pool)
        rec[tname] = out
    return rec, draws


def cs7_expected(nets: dict) -> dict:
    """E7.1 to E7.3 (amendment 7 item 2) over the networks' records."""
    def layer(v, c, t, L):
        return v["components"][c][t]["layers"][L]

    named, named_geo = [], []
    for lb, v in nets.items():
        for c in SX:
            for t in CS7_TARGETS:
                for L in CS7_LAYERS:
                    g = layer(v, c, t, L)
                    if g["stat"] > E71_GAIN and g["ci"][0] > 0:
                        named.append(f"{lb}/{c}/{t}/{L}")
                    if g["stat"] > E71_GAIN and g["ci_geometry"][0] > 0:
                        named_geo.append(f"{lb}/{c}/{t}/{L}")
    e72 = {f"{t}:{c}": _count_of(nets, lambda v, c=c, t=t: layer(v, c, t, "h3")["stat"]
                                 >= layer(v, c, t, CS7_ACTOR)["stat"])
           for t in CS7_TARGETS for c in SX}
    e73 = {f"{t}:{c}": _count_of(nets, lambda v, c=c, t=t: layer(v, c, t, "heads")["stat"]
                                 <= layer(v, c, t, "h3")["stat"])
           for t in CS7_TARGETS for c in SX}
    n = len(nets)
    return {
        "E7.1": {"above_0.05_with_ci_above_0": named,
                 "above_0.05_with_geometry_ci_above_0": named_geo, "holds": not named},
        "E7.2": {**e72, "holds": all(p["count"] >= min(E_COUNT, n) for p in e72.values())},
        "E7.3": {**e73, "holds": all(p["count"] >= min(E_COUNT, n) for p in e73.values())},
        "reading": "amendment 7 item 2"}


@computes("CS-7")
def cs7(run: Run) -> None:
    """CS-7 (plan Section 4, amendment 7 item 2): the frozen lambda_res checked; per network
    its frame and decision mask; per (network, component) 4 pool units (``cs7_unit``); then
    the gains against the controls, their means over s and xi and the expected directions;
    outputs summary.json and preds.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    frozen = run.rec["frozen"]
    lam, by_fold, oof, _fo, res_gap = checked_lam_res(run, tab)
    direction = np.asarray(tab["direction"])
    G = AN.geometry_clusters(tab)[1]
    counts = PB.cluster_counts(G)
    pool = PB.pool_draws(PB.N_POOL)
    chunks = cs7_chunks()
    units, names, frames = [], [], {}
    for lb in decision_labels(run):
        name = AN.network_of(lb)
        agent, _, fr = network_frame(run, name, world)
        if not np.array_equal(fr.lam, lam):
            raise SystemExit(f"CS-7 {lb}: C1's frame's lambda is not the frozen lambda_res's")
        rows = c1_rows(fr, frozen["k2b_excluded"])
        act = run.read(f"post/activations/{PR.net_file(name)}.npz")
        frames[lb] = (fr, rows, name)
        for c, k in zip(SX, K_SX, strict=True):
            idx = np.flatnonzero(rows[c])
            F = by_fold.shape[0]
            Y = np.concatenate([np.broadcast_to(fr.lam[idx, k][None, :, None], (F, len(idx), 1)),
                                by_fold[:, idx, k][:, :, None]], axis=2)
            for i, jobs in enumerate(chunks):
                units.append({"label": lb, "comp": c, "chunk": i, "path": str(act),
                              "cfg": agent.cfg, "idx": idx, "Y": Y, "w": fr.w_in[idx],
                              "fo": fr.fo.subset(rows[c]), "jobs": jobs})
                names.append(f"{lb}/{c}/{i}")
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, names = units[:int(lim)], names[:int(lim)]
    print(f"CS-7: {len(units)} units ({time.time() - t0:.0f} s)", flush=True)
    t_pool = time.time()
    got = run.pool(cs7_unit, units, names=names)
    pool_s = time.time() - t_pool
    by: dict = {}
    for u, r in zip(units, got, strict=True):
        o = by.setdefault((u["label"], u["comp"]), {"preds": {}, "y": None, "checks": {},
                                                    "seconds": 0.0, "chunks": 0, "unit": u})
        if o["y"] is None:
            o["y"] = r["y"]
        elif not np.array_equal(o["y"], r["y"], equal_nan=True):
            raise SystemExit(f"CS-7 {u['label']}/{u['comp']}: the chunks' targets differ")
        o["preds"].update(r["preds"])
        o["checks"].update(r["checks"])
        o["seconds"] += float(r["seconds"])
        o["chunks"] += 1
    nets, arrays, partial_units, draws_of = {}, {}, [], {}
    for (lb, c), o in by.items():
        if o["chunks"] != len(chunks):
            partial_units.append(f"{lb}/{c}")
            continue
        fr, rows, name = frames[lb]
        u = o["unit"]
        idx = u["idx"]
        k = K_SX[SX.index(c)]
        y = np.asarray(o["y"])
        if not (np.array_equal(y[:, 0], fr.lam[idx, k]) and np.array_equal(y[:, 1], oof[idx, k])):
            raise SystemExit(f"CS-7 {lb}/{c}: the probes' targets are not lambda and the "
                             "frozen lambda_res's own-fold residual")
        odd = c == "xi"
        rec, dr = cs7_component(o["preds"], y, u["w"], fr.cl[idx], G, counts, pool,
                                direction[idx] if odd else None, odd)
        draws_of[(lb, c)] = dr
        c2g = ((run.decisions.get("networks") or {}).get(name) or {}).get("C2", {}).get("gain")
        net = nets.setdefault(lb, {"run": name, "components": {}, "checks": {}})
        net["components"][c] = {**rec, "rows": len(idx), "seconds": o["seconds"],
                                "c2_confirmatory_gain": None if c2g is None else c2g[k]}
        net["checks"][c] = o["checks"]
        keys = cs7_keys()
        arrays[f"{lb}/{c}/rows"] = idx
        arrays[f"{lb}/{c}/y"] = y
        for kk in keys[:6]:
            arrays[f"{lb}/{c}/{kk}"] = np.asarray(o["preds"][kk])
        arrays[f"{lb}/{c}/random"] = np.stack([o["preds"][kk] for kk in keys[6:6 + 80]]
                                              ).astype(np.float32)
        arrays[f"{lb}/{c}/pool"] = np.stack([o["preds"][kk] for kk in keys[86:]]
                                            ).astype(np.float32)
    for lb, net in nets.items():
        if all((lb, c) in draws_of for c in SX):
            net["mean_over_s_xi"] = {
                t: {L: mean_of_draws([draws_of[(lb, c)][(t, L)] for c in SX])
                    for L in (*CS7_LAYERS, CS7_ACTOR)} for t in CS7_TARGETS}
    complete = {lb: v for lb, v in nets.items() if set(v["components"]) == set(SX)}
    out = {"item": "CS-7", "networks": nets,
           "expected": cs7_expected(complete) if complete else None,
           "incomplete": partial_units, "layers": list(CS7_LAYERS), "targets": {
               "lam": "lambda_k winsorised at its frozen bounds",
               "res": "the frozen lambda_res,k (f_D, k = 8), by fold"},
           "controls": {"critic": "random-init critics, seeds 0-19 (computed here)",
                        "actor_h3": "the pool actors, seeds 0-19 (stored h3)"},
           "residual_share_gap": res_gap,
           "definitions": "plan Section 4 CS-7 and amendment 7 items 1 and 2",
           "n_boot": len(counts), "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_npz("preds", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-8: C2-sw and C2
#
# Plan Section 4 CS-8 with amendment 7 item 3. (a) C2-sw's random-only null: hosts 20-39
# scored as the network against the pool on the 2,507 approach- and exit-phase
# pedal-switching rows, G1's scaling first, per network under R6's 3 h rule, and the trained
# networks' C2-sw on the same rows. (b) C2-sw on sigma^ped residualised on f_D's decoys. (c) C2
# with a decoy that fits geometry 6, chosen on targets alone, on the interior's approach and
# exit rows, against f_D on the same rows. The probes of (a) and (b) are fitted once per
# scaling and every record is assembled from them (``sw_record``).

CS8_TARGETS = ("sigma", "res")           # sigma^ped winsorised; sigma_res (f_D, k = 8)
CS8_CHUNKS = 4
R6_BUDGET = 3 * 3600.0                   # s: R6's 'about 3 h'
E8A_MEAN = 0.005                         # the null's mean rounds to 0.00 or less
E8A_SD = (0.005, 0.025)                  # its SD rounds to 0.01 or 0.02
CONFIRMATORY_RANK1 = ("G1", "G1s1", "G1s2", "G2", "G2s2", "G3s1", "G3s2")
E8B_HALF, E8B_RANK1_MAX, E8B_SURVIVE = 0.5, 4, 7
E8C_SHARE, E8C_NARROW = 0.3, 2.0 / 3.0
DECOYS = ("same_theta_direction", "log_r_x3", "spline_ridge")   # the plan's order
G6 = 6                                   # geometry 6: R 9.4 m, theta 90 deg (design 17.6)
G6_DESIGN = {("lam", "s"): (11.8, 1), ("lam", "xi"): (9.0, 1),
             ("res", "s"): (70.0, 0), ("res", "xi"): (65.0, 0)}  # design 17.6: %, decimals
LOG_R_COL, THETA_COL, DIR_COL = 2, 3, 5  # in states.decoy_d1
LOG_R_WEIGHT = 3.0
SPLINE_KNOTS, SPLINE_DEGREE = 5, 3


def cs8_rows(tab):
    """[n] (a)'s and (b)'s rows: T's pedal-switching rows (``states.SW_STRATUM``) in the
    approach and exit phases (amendment 7 item 3), through the row guard."""
    import numpy as np

    from apexzero.costate import states as ST

    rows = np.asarray(tab[ST.SW_STRATUM], bool) & np.isin(
        np.asarray(tab["phase"]).astype(str), SCORED_PHASES)
    return guard_rows(tab, rows)


def sw_record(base, full, controls, w, fo, draws, labels, m_pedal, y, seeds) -> dict:
    """``analysis.c2_sw_network``'s record from fitted probes on the stratum's rows: the gain
    of ``full`` over ``base`` against the median of ``controls`` (``probes.gain`` with the
    two-level draws and the single-geometry rule's sets, as ``probes.c2_sw`` calls it), the
    stratum's size, rho(m_pedal - 1, y) and the controls' label."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    g = PB.gain(base, full, list(controls), w, fo, None, None, None, None, pool=draws,
                logo="rule", labels=labels)
    rec = g.record()
    rec["n_stratum"] = len(w)
    rec["spearman_margin"] = PB.wspearman(np.asarray(m_pedal) - 1.0, np.asarray(y), w)
    rec.update({"controls": AN.controls_label(seeds, True)})
    return rec


def null_entry(seed: int, rec: dict) -> dict:
    """A host's entry as ``gates.c2_null`` records it (its summary reads these fields)."""
    from apexzero.costate import probes as PB

    ci_g = rec.get("ci_geometry")
    return {"seed": int(seed), **{k: rec.get(k) for k in (
        "stat", "ci", "ci_geometry", "gain", "raw", "decision", "decision_all_geometries",
        "geometry_sensitive", "reason", "n_stratum", "pool", "spearman_margin")},
        "decision_geometry_ci": None if ci_g is None else PB.decide(
            float(rec["stat"]), float(ci_g[0]), float(ci_g[1]))}


def share_at_or_above(stats, stat: float) -> float | None:
    """``analysis.c2_null_note``'s share of the null's finite statistics at or above ``stat``."""
    import numpy as np

    st = np.asarray([v for v in stats if v is not None], float)
    st = st[np.isfinite(st)]
    return float(np.mean(st >= float(stat))) if st.size else None


def cs8_jobs(hosts: bool) -> list[tuple]:
    """One scaling's probe fits (amendment 7 item 3): the base, the trained h3 and the 20
    pool actors on sigma^ped and on sigma_res, and, with ``hosts``, the 20 null hosts on
    sigma^ped."""
    from apexzero.costate import gates as GT
    from apexzero.costate import probes as PB

    jobs = [("base", t) for t in CS8_TARGETS] + [("trained", t) for t in CS8_TARGETS]
    jobs += [("pool", int(s), t) for t in CS8_TARGETS for s in PB.POOL_SEEDS]
    if hosts:
        jobs += [("host", int(s), "sigma") for s in GT.C2_NULL_SEEDS]
    return jobs


def chunked(jobs: Sequence, n: int) -> list[list]:
    """``jobs`` dealt round-robin into ``n`` chunks (similar cost, the cheap base fits
    spread)."""
    return [list(jobs[i::n]) for i in range(n) if jobs[i::n]]


def cs8_sw_unit(state, unit) -> dict:
    """One chunk of one scaling's (a) and (b) probes in a worker: each job's ridge probe on z
    (``probes.ridge_cv``; base None, the trained h3, a pool actor's stored h3, or a null
    host's h3 built from the scaling's PPOConfig, ``probes.random_actor``), on sigma^ped or
    on sigma_res by fold. The pool actor of seed 0 is rebuilt and checked first."""
    import numpy as np

    from apexzero.costate import probes as PB

    t0 = time.time()
    idx, cfg, jobs = unit["idx"], unit["cfg"], unit["jobs"]
    seeds = sorted({int(j[1]) for j in jobs if j[0] == "pool"} | {0})
    with np.load(unit["path"]) as d:
        z = np.clip(d["z"][idx], -10.0, 10.0)
        h3 = d["actor_h3"][idx]
        m_tr = d["m"][idx, 1]
        pool = {s: d[f"random{s}_h3"][idx] for s in seeds}
    tag = f"CS-8 {unit['label']}/{unit['chunk']}"
    gap = pool_actor_gap(cfg, z, pool[0])
    if not gap <= RANDOM_TOL:
        raise RuntimeError(f"{tag}: the pool actor of seed 0 does not rebuild its stored h3 "
                           f"({gap:.3g})")
    Y = {"sigma": unit["sigma"], "res": unit["res"]}
    names = {"sigma": ("sigma_pedal",), "res": ("sigma_res",)}
    probes, m_host = {}, {}
    for job in jobs:
        kind, t = job[0], job[-1]
        if kind == "base":
            h = None
        elif kind == "trained":
            h = h3
        elif kind == "pool":
            h = pool[int(job[1])]
        elif kind == "host":
            hid = PB.hidden(PB.random_actor(cfg, int(job[1])), z)
            h = hid["h3"]
            m_host[int(job[1])] = hid["m"][:, 1]
        else:
            raise ValueError(f"{tag}: unknown job {job}")
        probes[job] = PB.ridge_cv(z, Y[t], unit["fo"], unit["w"], h, names[t])
    return {"probes": probes, "m_host": m_host, "m_trained": m_tr, "random0_gap": gap,
            "seconds": time.time() - t0}


def cs8_scaling(probes: dict, m_host: dict, m_trained, sigma, res_y, w, fo, draws, labels,
                hosts: bool) -> dict:
    """One scaling's records from its probes: the trained h3's C2-sw on sigma^ped and on
    sigma_res, and, with ``hosts``, every null host's record and the null's summary
    (``gates.c2_null_summary``)."""
    from apexzero.costate import gates as GT
    from apexzero.costate import probes as PB

    seeds = tuple(int(s) for s in PB.POOL_SEEDS)
    out: dict = {}
    for t, y in (("sigma", sigma), ("res", res_y)):
        ctl = [probes[("pool", s, t)] for s in seeds]
        out[t] = sw_record(probes[("base", t)], probes[("trained", t)], ctl, w, fo, draws,
                           labels, m_trained, y, seeds)
    if hosts:
        ctl = [probes[("pool", s, "sigma")] for s in seeds]
        nets = []
        for s in GT.C2_NULL_SEEDS:
            rec = sw_record(probes[("base", "sigma")], probes[("host", int(s), "sigma")], ctl,
                            w, fo, draws, labels, m_host[int(s)], sigma, seeds)
            nets.append({**null_entry(int(s), rec), "most_influential": (
                rec.get("logo") or {}).get("most_influential")})
        out["null"] = {"seeds": [int(s) for s in GT.C2_NULL_SEEDS], "networks": nets,
                       "summary": GT.c2_null_summary(nets)}
    return out


# -- (c): the decoy candidates (amendment 7 item 3)


def cross_fit_with(X, Y, fold, w, fitpred) -> tuple:
    """(by fold [F, n, m], info): ``states.cross_fit``'s scheme with any decoy model
    ``fitpred(X_train, Y_train, w_train, fold_train, X_pred) -> (pred, info)``: outer fold f's
    rows from a model fitted on the other folds, and the training rows of fold g != f from a
    model fitted on the folds other than f and g. ``info`` sums the models' integer counts
    (rows that fell back)."""
    import numpy as np

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Y = Y[:, None] if Y.ndim == 1 else Y
    fold = np.asarray(fold, int)
    w = np.asarray(w, float)
    F = int(fold.max()) + 1
    out = np.full((F, *Y.shape), np.nan)
    info: dict = {"fits": 0, "fallback_rows": 0, "fallback_rows_held_out": 0}

    def one(fit, pred, held_out: bool):
        p, inf = fitpred(X[fit], Y[fit], w[fit], fold[fit], X[pred])
        info["fits"] += 1
        fb = int((inf or {}).get("fallback_rows", 0))
        info["fallback_rows"] += fb
        if held_out:
            info["fallback_rows_held_out"] += fb
        return Y[pred] - p

    for f in range(F):
        test, train = fold == f, fold != f
        out[f, test] = one(train, test, True)
        for g in range(F):
            if g == f:
                continue
            out[f, fold == g] = one(train & (fold != g), fold == g, False)
    return out, info


def knn_fitpred(k: int):
    """f_D's own model (``states.knn_fit``) as a ``cross_fit_with`` model."""
    from apexzero.costate import states as ST

    def fp(Xtr, Ytr, wtr, ftr, Xte):
        return ST.knn_fit(Xtr, Ytr, wtr, k).predict(Xte), {}

    return fp


def group_knn_fitpred(k: int, log_r_weight: float = 1.0):
    """Candidates 1 and 2: f_D's kNN standardised by all the fit's training rows (as
    ``states.knn_fit``), each predicted row's neighbours searched among the training rows of
    its own theta and turn direction (fewer than k there: of its turn direction), with the
    standardised log R coordinate times ``log_r_weight``."""
    import numpy as np
    from scipy.spatial import cKDTree

    from apexzero.costate import states as ST

    def fp(Xtr, Ytr, wtr, ftr, Xte):
        mean = ST.wmean(Xtr, wtr)
        sd = np.sqrt(ST.wvar(Xtr, wtr))
        sd = np.where(sd > 0, sd, 1.0)
        sd[LOG_R_COL] = sd[LOG_R_COL] / float(log_r_weight)
        out = np.full((len(Xte), Ytr.shape[1]), np.nan)
        fallback = 0
        keys_te = np.stack([Xte[:, THETA_COL], Xte[:, DIR_COL]], 1)
        for th, dr in np.unique(keys_te, axis=0):
            te = (Xte[:, THETA_COL] == th) & (Xte[:, DIR_COL] == dr)
            tr = (Xtr[:, THETA_COL] == th) & (Xtr[:, DIR_COL] == dr)
            if tr.sum() < k:
                tr = Xtr[:, DIR_COL] == dr
                fallback += int(te.sum())
            if tr.sum() < k:
                raise ValueError(f"fewer than {k} training rows of direction {dr}")
            knn = ST.KNN(mean=mean, sd=sd, tree=cKDTree((Xtr[tr] - mean) / sd), y=Ytr[tr],
                         w=wtr[tr], k=int(k))
            out[te] = knn.predict(Xte[te])
        return out, {"fallback_rows": fallback}

    return fp


def bspline_basis(u, lo: float, hi: float, interior, degree: int = SPLINE_DEGREE):
    """[n, len(interior) + degree + 1] the cubic B-spline basis on [lo, hi] with the given
    interior knots, u clipped to [lo, hi] (amendment 7 item 3, candidate 3)."""
    import numpy as np
    from scipy.interpolate import BSpline

    t = np.r_[np.full(degree + 1, lo), np.asarray(interior, float), np.full(degree + 1, hi)]
    return BSpline.design_matrix(np.clip(np.asarray(u, float), lo, hi), t, degree).toarray()


def spline_spec(Xtr, wtr) -> list:
    """[(lo, hi, interior knots)] of s - s_apex and vx on a fit's training rows: their range
    and the weighted quantiles 1/6, ..., 5/6."""
    import numpy as np

    from apexzero.costate import states as ST

    qs = np.arange(1, SPLINE_KNOTS + 1) / (SPLINE_KNOTS + 1)
    out = []
    for j in (0, 1):                     # decoy_d1's ds_apex and vx
        u = np.asarray(Xtr[:, j], float)
        out.append((float(u.min()), float(u.max()),
                    [float(v) for v in ST.wquantile(u, wtr, qs)]))
    return out


def spline_design(X, spec) -> object:
    """[n, 486] candidate 3's columns: the tensor product of the two bases (81), each times
    (1, log R, theta class 1, theta class 2, w, sigma)."""
    import numpy as np

    from apexzero.costate import states as ST

    X = np.asarray(X, float)
    n = len(X)
    B1 = bspline_basis(X[:, 0], *spec[0])
    B2 = bspline_basis(X[:, 1], *spec[1])
    T = (B1[:, :, None] * B2[:, None, :]).reshape(n, -1)
    tc = ST.theta_class(X[:, THETA_COL])
    M = np.stack([np.ones(n), X[:, LOG_R_COL], (tc == 1).astype(float), (tc == 2).astype(float),
                  X[:, 4], X[:, DIR_COL]], 1)
    return (T[:, :, None] * M[:, None, :]).reshape(n, -1)


def wridge_path(Xtr, Ytr, wtr, Xte, alphas):
    """[A, n_te, m] weighted ridge predictions for every penalty: columns z-scored on the
    training rows (an SD below ``probes.MIN_SD`` dropped), the target centred, the weights
    normalised to sum 1."""
    import numpy as np

    from apexzero.costate import probes as PB

    w = np.asarray(wtr, float) / np.sum(wtr)
    mu = w @ Xtr
    sd = np.sqrt(w @ (Xtr - mu) ** 2)
    keep = sd >= PB.MIN_SD
    Xs = (Xtr[:, keep] - mu[keep]) / sd[keep]
    Xt = (Xte[:, keep] - mu[keep]) / sd[keep]
    yb = w @ Ytr
    Gm = Xs.T @ (Xs * w[:, None])
    c = Xs.T @ ((Ytr - yb) * w[:, None])
    e, V = np.linalg.eigh(Gm)
    e = np.clip(e, 0.0, None)
    Vc, XtV = V.T @ c, Xt @ V
    return np.stack([yb + XtV @ (Vc / (e[:, None] + float(a))) for a in alphas])


def choose_penalty(sse) -> object:
    """[m] per column, the index of the smallest SSE [A, m]; near-ties (``probes.TIE_RTOL``)
    to the larger penalty (``probes._choose``'s rule)."""
    import numpy as np

    from apexzero.costate import probes as PB

    sse = np.asarray(sse, float)
    out = np.empty(sse.shape[1], int)
    for j in range(sse.shape[1]):
        ok = sse[:, j] <= sse[:, j].min() * (1.0 + PB.TIE_RTOL) + 1e-300
        out[j] = int(np.flatnonzero(ok).max())
    return out


def spline_fitpred(Xtr, Ytr, wtr, ftr, Xte):
    """Candidate 3 as a ``cross_fit_with`` model: the spline design from the fit's training
    rows, each column's penalty from ``probes.GRID`` by leave-one-outer-fold-out over the
    fit's training folds, then refitted on all of them."""
    import numpy as np

    from apexzero.costate import probes as PB

    spec = spline_spec(Xtr, wtr)
    Atr, Ate = spline_design(Xtr, spec), spline_design(Xte, spec)
    alphas = PB.GRID
    sse = np.zeros((len(alphas), Ytr.shape[1]))
    for h in np.unique(ftr):
        tr, te = ftr != h, ftr == h
        P = wridge_path(Atr[tr], Ytr[tr], wtr[tr], Atr[te], alphas)
        sse += np.einsum("n,anm->am", wtr[te], (Ytr[te][None] - P) ** 2)
    pick = choose_penalty(sse)
    P = wridge_path(Atr, Ytr, wtr, Ate, alphas)
    return np.stack([P[pick[j], :, j] for j in range(Ytr.shape[1])], 1), {
        "penalties": [float(alphas[i]) for i in pick]}


def decoy_models(k: int) -> dict:
    """{name: ``cross_fit_with`` model} of the candidates, in the plan's order."""
    return {"same_theta_direction": group_knn_fitpred(k),
            "log_r_x3": group_knn_fitpred(k, LOG_R_WEIGHT),
            "spline_ridge": spline_fitpred}


def geometry_share(v, w, geo, direction=None, odd: bool = False, g: int = G6) -> float:
    """Geometry g's share of v's weighted sum of squares about its weighted mean (about each
    turn direction's when ``odd``: R2_wd's denominator; amendment 7 item 3)."""
    import numpy as np

    v, w = np.asarray(v, float), np.asarray(w, float)
    if odd:
        d = np.asarray(direction)
        mu = np.zeros_like(v)
        for s in np.unique(d):
            m = d == s
            mu[m] = np.sum(w[m] * v[m]) / np.sum(w[m])
    else:
        mu = np.sum(w * v) / np.sum(w)
    q = w * (v - mu) ** 2
    return float(q[np.asarray(geo) == g].sum() / q.sum())


def g6_shares(lam, oof, valid, rows, w, geo, direction) -> dict:
    """{('lam' | 'res', component): geometry 6's share} of lambda and of the own-fold residual
    on ``rows`` within each component's decision rows, under the weights ``w`` [n]."""
    import numpy as np

    out = {}
    for c, k in zip(SX, K_SX, strict=True):
        r = np.asarray(rows, bool) & np.asarray(valid, bool)[:, k]
        odd = c == "xi"
        for name, v in (("lam", lam), ("res", oof)):
            out[(name, c)] = geometry_share(np.asarray(v)[r, k], np.asarray(w)[r],
                                            np.asarray(geo)[r], np.asarray(direction)[r], odd)
    return out


def design_g6_problems(shares: dict) -> list[str]:
    """Where f_D's shares do not reproduce design 17.6 at its printed rounding."""
    bad = []
    for key, (want, dec) in G6_DESIGN.items():
        got = round(100.0 * shares[key], dec)
        if got != want:
            bad.append(f"{key[0]}_{key[1]}: {100.0 * shares[key]:.3f}% against {want}%")
    return bad


def passes_g6(shares: dict) -> bool:
    """The rule: lambda_res's geometry-6 shares at most twice lambda's, for s and for xi."""
    return all(shares[("res", c)] <= 2.0 * shares[("lam", c)] for c in SX)


def _shares_json(shares: dict) -> dict:
    return {f"{a}_{c}": float(v) for (a, c), v in shares.items()}


def cs8_c2_unit(state, unit) -> dict:
    """One (network, decoy)'s C2 in a worker: ``analysis.c2_network`` as the confirmatory
    primary (base [z, m], the trained h3 against the 20 pool actors with the two-level draws,
    the by-fold lambda_res, the frozen power floor, the decision mask, the rule's sets) on
    (c)'s stratum. The pool actor of seed 0 is rebuilt and checked first."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import targets as TG

    t0 = time.time()
    rows = np.asarray(unit["rows"], bool)
    with np.load(unit["path"]) as d:
        z = np.clip(d["z"], -10.0, 10.0)
        h3, m = d["actor_h3"], d["m"]
        pool = [d[f"random{s}_h3"] for s in PB.POOL_SEEDS]
    gap = pool_actor_gap(unit["cfg"], z[rows], pool[0][rows])
    if not gap <= RANDOM_TOL:
        raise RuntimeError(f"CS-8 {unit['label']}/{unit['decoy']}: the pool actor of seed 0 "
                           f"does not rebuild its stored h3 ({gap:.3g})")
    try:
        rec = AN.c2_network(h3, pool, z, m, unit["lam_res"], unit["fo"], None,
                            unit["direction"], TG.K_P, unit["power_floor"], rows,
                            weights=unit["w"][rows], mask=unit["valid"],
                            pool=PB.pool_draws(len(pool)), controls=PB.POOL_SEEDS, logo="rule",
                            labels=unit["labels"])
    except PB.NotTestable as e:
        rec = {"decision": "not testable", "reason": str(e)}
    return {"record": rec, "random0_gap": gap, "seconds": time.time() - t0}


def cs8_expected(sw: dict, null_g1: dict | None, own_null: dict, c2: dict | None) -> dict:
    """E8a.1, E8b.1 and E8c.1 (amendment 7 item 3)."""
    from apexzero.costate import probes as PB

    out: dict = {"reading": "amendment 7 item 3"}
    stat = {lb: v["sigma"]["stat"] for lb, v in sw.items()}
    rank1 = [lb for lb, v in sw.items() if (v["sigma"].get("pool") or {}).get("rank") == 1]
    if null_g1 is not None:
        s = null_g1["summary"]
        mx = s.get("max")
        above = {lb: (lb in stat and mx is not None and stat[lb] > mx) for lb in
                 CONFIRMATORY_RANK1}
        out["E8a.1"] = {
            "null_mean": s.get("mean"), "mean_part": bool(s.get("mean") is not None
                                                          and s["mean"] < E8A_MEAN),
            "null_sd": s.get("sd"), "sd_part": bool(s.get("sd") is not None
                                                    and E8A_SD[0] <= s["sd"] < E8A_SD[1]),
            "null_max": mx, "confirmatory_rank1_above_every_host": above,
            "rank1_part": bool(all(above.values())),
            "rank1_on_these_rows": rank1,
            "rank1_on_these_rows_above_every_host": {
                lb: bool(mx is not None and stat[lb] > mx) for lb in rank1},
            "own_scaling_above_every_host": {
                lb: bool(v["summary"].get("max") is not None and lb in stat
                         and stat[lb] > v["summary"]["max"]) for lb, v in own_null.items()}}
        e = out["E8a.1"]
        e["holds"] = bool(e["mean_part"] and e["sd_part"] and e["rank1_part"])
    half, nonpos, r1, survive = {}, [], {}, {}
    for lb, v in sw.items():
        a, b = v["sigma"]["stat"], v["res"]["stat"]
        if a is None or b is None:
            continue
        if a <= 0:
            nonpos.append(lb)
        else:
            half[lb] = bool(b <= E8B_HALF * a)
        rk = (v["res"].get("pool") or {}).get("rank")
        r1[lb] = rk == 1
        survive[lb] = bool(rk == 1 and v["res"]["ci"][0] > 0)
    out["E8b.1"] = {"at_most_half": half, "count_at_most_half": int(sum(half.values())),
                    "sigma_stat_not_positive": nonpos, "rank1_on_res": r1,
                    "count_rank1_on_res": int(sum(r1.values())),
                    "rank1_with_ci_above_0": survive,
                    "survives": bool(sum(survive.values()) >= E8B_SURVIVE),
                    "holds": bool(sum(half.values()) >= E_COUNT
                                  and sum(r1.values()) <= E8B_RANK1_MAX)}
    if c2 is not None and c2.get("networks"):
        p1, p2, p3 = {}, {}, {}
        for lb, v in c2["networks"].items():
            ch, ref = v.get("chosen") or {}, v.get("frozen") or {}
            g6 = ((ch.get("logo") or {}).get("shares") or {})
            sh = [((g6.get(f"lam_{c}") or {}).get(f"geometry{G6}") or {}).get("sse_base")
                  for c in SX]
            p1[lb] = bool(all(x is not None and x < E8C_SHARE for x in sh))
            if ch.get("ci") is not None and ref.get("ci") is not None:
                p2[lb] = bool(ci_width(ch["ci"]) <= E8C_NARROW * ci_width(ref["ci"]))
            p3[lb] = bool(ch.get("stat") is not None and ch["stat"] < PB.C2_BAND)
        n = len(c2["networks"])
        out["E8c.1"] = {"geometry6_below_0.3": p1, "count_geometry6": int(sum(p1.values())),
                        "narrower_by_a_third": p2, "count_narrower": int(sum(p2.values())),
                        "point_below_0.05": p3, "count_below_0.05": int(sum(p3.values())),
                        "holds": bool(sum(p1.values()) >= min(E_COUNT, n)
                                      and sum(p2.values()) >= min(E_COUNT, n)
                                      and sum(p3.values()) == n)}
    elif c2 is not None:
        out["E8c.1"] = {"holds": None, "reason": c2.get("reason")}
    return out


@computes("CS-8")
def cs8(run: Run) -> None:
    """CS-8 (plan Section 4, amendment 7 item 3): the frozen lambda_res and sigma_res; (a) and
    (b)'s probes per scaling in pool units (``cs8_sw_unit``), G1's first and timed, the other
    networks' with their hosts only under R6's 3 h rule; (c)'s decoy chosen on targets alone
    (``cross_fit_with``), then C2 per (network, decoy) in pool units (``cs8_c2_unit``);
    outputs summary.json and preds.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import gates as GT
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    frozen = run.rec["frozen"]
    lam, by_fold, oof, fo, res_gap = checked_lam_res(run, tab)
    null_conf = json.loads(run.read("gates/c2_null.json").read_text())
    direction = np.asarray(tab["direction"])
    geo = AN.geometry_clusters(tab)[0]
    labels_geo = AN.geometry_labels(tab)
    draws = PB.pool_draws(PB.N_POOL)
    k_d = PR.decoy_k(frozen)
    k_d = ST.KNN_K if k_d is None else k_d
    w_key = ST.weights(tab, "key")
    D1 = ST.decoy_d1(tab)
    # -- (a) and (b): rows, sigma^ped and sigma_res (targets only, every row of T)
    rs = cs8_rows(tab)
    idx = np.flatnonzero(rs)
    ws = ST.stratum_weights(tab, rs)[idx]
    fo_s = fo.subset(rs)
    sig = ST.winsorise(np.asarray(tab["sigma_pedal"], float),
                       frozen["winsor_bounds"]["sigma_pedal"])
    sres = ST.cross_fit(D1, sig, fo.outer, w_key, k=k_d)
    sres_rows = sres.by_fold[:, idx, :]                              # [F, n_rows, 1]
    sres_y = sres.oof[idx, 0]
    share_sres = float(ST.wvar(sres_y, ws) / ST.wvar(sig[idx], ws))
    labels = decision_labels(run)
    frames, agents = {}, {}
    for lb in dict.fromkeys(["G1", *labels]):
        name = AN.network_of(lb)
        agents[lb], _, frames[lb] = network_frame(run, name, world)
    acts = {lb: run.read(f"post/activations/{PR.net_file(AN.network_of(lb))}.npz")
            for lb in labels}

    def sw_units(lb: str, hosts: bool) -> tuple:
        jobs = cs8_jobs(hosts)
        us = [{"label": lb, "chunk": i, "path": str(acts[lb]), "cfg": agents[lb].cfg,
               "idx": idx, "sigma": sig[idx], "res": sres_rows, "w": ws, "fo": fo_s,
               "jobs": ch} for i, ch in enumerate(chunked(jobs, CS8_CHUNKS))]
        return us, [f"{lb}/{u['chunk']}" for u in us]

    def gather(us, got) -> dict:
        out: dict = {}
        for u, r in zip(us, got, strict=True):
            o = out.setdefault(u["label"], {"probes": {}, "m_host": {}, "seconds": 0.0,
                                            "random0_gap": 0.0, "m_trained": r["m_trained"]})
            o["probes"].update(r["probes"])
            o["m_host"].update(r["m_host"])
            o["seconds"] += float(r["seconds"])
            o["random0_gap"] = max(o["random0_gap"], float(r["random0_gap"]))
        return out

    lim = getattr(run.args, "limit", None)
    phases: dict = {}
    got_by: dict = {}
    first = [lb for lb in labels if lb == "G1"]
    rest = [lb for lb in labels if lb != "G1"]
    t_g1 = None
    if first:
        us, nm = sw_units("G1", True)
        if lim is not None:
            us, nm = us[:int(lim)], nm[:int(lim)]
        print(f"CS-8: G1's scaling, {len(us)} units ({time.time() - t0:.0f} s)", flush=True)
        t1 = time.time()
        got_by.update(gather(us, run.pool(cs8_sw_unit, us, names=nm)))
        t_g1 = time.time() - t1
    projection = None if t_g1 is None else 8.0 * t_g1
    per_network = bool(projection is not None and projection < R6_BUDGET)
    phases["r6"] = {"g1_wall_seconds": t_g1, "projected_other_8_seconds": projection,
                    "budget_seconds": R6_BUDGET, "per_network": per_network,
                    "rule": "plan R6 and amendment 7 item 3: per network when 8 x G1's "
                            "measured pooled time is under about 3 h"}
    if rest and lim is None:
        us, nm = [], []
        for lb in rest:
            u1, n1 = sw_units(lb, per_network)
            us += u1
            nm += n1
        print(f"CS-8: the other scalings, {len(us)} units (hosts: {per_network}; "
              f"{time.time() - t0:.0f} s)", flush=True)
        t2 = time.time()
        got_by.update(gather(us, run.pool(cs8_sw_unit, us, names=nm)))
        phases["other_wall_seconds"] = time.time() - t2
    sw, own_null, arrays = {}, {}, {}
    null_g1 = None
    complete_jobs = {lb: set(cs8_jobs(lb == "G1" or per_network)) for lb in got_by}
    for lb, o in got_by.items():
        if set(o["probes"]) != complete_jobs[lb]:
            continue                                     # a partial run's incomplete scaling
        pr = o["probes"]
        if not (np.array_equal(pr[("base", "sigma")].y[:, 0], sig[idx])
                and np.array_equal(pr[("base", "res")].y[:, 0], sres_y)):
            raise SystemExit(f"CS-8 {lb}: the probes' targets are not sigma^ped and sigma_res")
        hosts = ("host", int(GT.C2_NULL_SEEDS[0]), "sigma") in pr
        rec = cs8_scaling(pr, o["m_host"], o["m_trained"], sig[idx], sres_y, ws, fo_s, draws,
                          labels_geo, hosts)
        name = AN.network_of(lb)
        conf = ((run.decisions.get("networks") or {}).get(name) or {}).get("C2-sw") or {}
        sw[lb] = {"run": name, "sigma": trim_logo(rec["sigma"]), "res": trim_logo(rec["res"]),
                  "random0_gap": o["random0_gap"], "seconds": o["seconds"],
                  "confirmatory_c2sw": {k: conf.get(k) for k in ("stat", "ci", "decision",
                                                                 "n_stratum")}
                  | {"rank": (conf.get("pool") or {}).get("rank")}}
        if hosts:
            nl = rec["null"]
            if lb == "G1":
                null_g1 = nl
            else:
                own_null[lb] = nl
        arrays[f"{lb}/sigma"] = np.stack(
            [pr[("base", "sigma")].pred[:, 0], pr[("trained", "sigma")].pred[:, 0],
             *[pr[("pool", int(s), "sigma")].pred[:, 0] for s in PB.POOL_SEEDS]]
        ).astype(np.float32)
        arrays[f"{lb}/res"] = np.stack(
            [pr[("base", "res")].pred[:, 0], pr[("trained", "res")].pred[:, 0],
             *[pr[("pool", int(s), "res")].pred[:, 0] for s in PB.POOL_SEEDS]]
        ).astype(np.float32)
        if hosts:
            arrays[f"{lb}/hosts"] = np.stack([pr[("host", int(s), "sigma")].pred[:, 0]
                                              for s in GT.C2_NULL_SEEDS]).astype(np.float32)
    if null_g1 is not None:
        for lb, v in sw.items():
            v["share_of_g1_hosts_at_or_above"] = share_at_or_above(
                [r["stat"] for r in null_g1["networks"]], v["sigma"]["stat"])
    for lb, nl in own_null.items():
        sw[lb]["share_of_own_hosts_at_or_above"] = share_at_or_above(
            [r["stat"] for r in nl["networks"]], sw[lb]["sigma"]["stat"])
    arrays["rows_sw"] = idx
    arrays["sigma"] = sig[idx]
    arrays["sigma_res"] = sres_y
    # -- (c): the decoy, chosen on targets alone before any network's probe is fitted
    c2 = None
    if lim is None:
        inner = np.asarray(tab["interior"], bool)
        w_conf = ST.stratum_weights(tab, inner)
        rows_c = cs5_rows(tab)
        w_c = ST.stratum_weights(tab, rows_c)
        valid_g1 = frames["G1"].valid
        sh_frozen = g6_shares(lam, oof, valid_g1, inner, w_conf, geo, direction)
        bad = design_g6_problems(sh_frozen)
        if bad:
            raise SystemExit(f"CS-8: f_D's geometry-6 shares do not reproduce design 17.6: "
                             f"{'; '.join(bad)}")
        by_knn, _ = cross_fit_with(D1, lam, fo.outer, w_key, knn_fitpred(k_d))
        fit_gap = float(np.max(np.abs(by_knn - by_fold)))
        if not fit_gap <= FIT_TOL:
            raise SystemExit(f"CS-8: the cross-fit scheme does not reproduce states.cross_fit "
                             f"({fit_gap:.3g})")
        tried, chosen = [], None
        for nm_d, model in decoy_models(k_d).items():
            t3 = time.time()
            bf, info = cross_fit_with(D1, lam, fo.outer, w_key, model)
            oof_d = bf[fo.outer, np.arange(len(fo.outer))]
            sh = g6_shares(lam, oof_d, valid_g1, inner, w_conf, geo, direction)
            sh_c = g6_shares(lam, oof_d, valid_g1, rows_c, w_c, geo, direction)
            share_t, share_in = ST.decoy_shares(tab, lam, oof_d)
            rec_d = {"decoy": nm_d, "shares": _shares_json(sh), "shares_on_stratum": _shares_json(
                sh_c), "passes": passes_g6(sh), "fallback": info,
                "residual_share_interior": dict(zip(("s", "n", "xi", "vx"), share_in.tolist())),
                "residual_share_T": dict(zip(("s", "n", "xi", "vx"), share_t.tolist())),
                "power_floor": ST.power_floor(share_in), "seconds": time.time() - t3}
            tried.append(rec_d)
            print(f"CS-8 (c): {nm_d}: passes {rec_d['passes']}", flush=True)
            if rec_d["passes"]:
                chosen = (nm_d, bf)
                break
        c2 = {"frozen_shares": _shares_json(sh_frozen),
              "frozen_shares_on_stratum": _shares_json(
                  g6_shares(lam, oof, valid_g1, rows_c, w_c, geo, direction)),
              "thresholds": {c: 2.0 * sh_frozen[("lam", c)] for c in SX},
              "cross_fit_gap": fit_gap, "tried": tried,
              "chosen": None if chosen is None else chosen[0],
              "stratum_rows": int(rows_c.sum())}
        if chosen is None:
            c2["reason"] = "no candidate fits geometry 6: nothing more is computed"
        else:
            arrays["c/lam_res_chosen"] = chosen[1].astype(np.float32)
            us, nm = [], []
            for lb in labels:
                fr = frames[lb]
                for c, k in zip(SX, K_SX, strict=True):
                    guard_rows(tab, rows_c & fr.valid[:, k], c, frozen["k2b_excluded"],
                               fr.valid[:, k])
                for dname, lr in (("frozen", by_fold), ("chosen", chosen[1])):
                    us.append({"label": lb, "decoy": dname, "path": str(acts[lb]),
                               "cfg": agents[lb].cfg, "rows": rows_c, "lam_res": lr, "fo": fo,
                               "direction": direction, "power_floor": frozen["power_floor"],
                               "w": w_c, "valid": fr.valid, "labels": labels_geo})
                    nm.append(f"{lb}/{dname}")
            print(f"CS-8 (c): {len(us)} units ({time.time() - t0:.0f} s)", flush=True)
            t4 = time.time()
            got = run.pool(cs8_c2_unit, us, names=nm)
            phases["c_wall_seconds"] = time.time() - t4
            nets_c: dict = {}
            for u, r in zip(us, got, strict=True):
                v = nets_c.setdefault(u["label"], {"run": AN.network_of(u["label"])})
                v[u["decoy"]] = trim_logo(r["record"])
                v[f"{u['decoy']}_seconds"] = r["seconds"]
                v["random0_gap"] = max(v.get("random0_gap", 0.0), r["random0_gap"])
            for lb, v in nets_c.items():
                conf = ((run.decisions.get("networks") or {}).get(v["run"]) or {}).get("C2") or {}
                v["confirmatory_c2"] = {k: conf.get(k) for k in ("stat", "ci", "decision",
                                                                 "n_stratum", "scored_rows")}
                ch, ref = v.get("chosen") or {}, v.get("frozen") or {}
                if ch.get("ci") is not None and ref.get("ci") is not None:
                    v["ci_width"] = {"chosen": ci_width(ch["ci"]), "frozen": ci_width(ref["ci"]),
                                     "confirmatory": None if conf.get("ci") is None
                                     else ci_width(conf["ci"])}
            c2["networks"] = nets_c
    out = {"item": "CS-8", "rows": {"sw": len(idx), "sw_by_phase": {
               p: int((rs & (np.asarray(tab["phase"]).astype(str) == p)).sum())
               for p in SCORED_PHASES}},
           "sigma_res_share": share_sres, "residual_share_gap": res_gap,
           "a_b": {"networks": sw, "null_g1": null_g1, "null_own": own_null,
                   "confirmatory_c2_null": {"summary": (null_conf.get("summary")),
                                            "rows": null_conf.get("rows"),
                                            "statistic": "C2 (not C2-sw), its 3,951 rows"}},
           "c": c2, "phases": phases, "expected": cs8_expected(sw, null_g1, own_null, c2),
           "definitions": "plan Section 4 CS-8 and amendment 7 items 1 and 3",
           "partial": lim is not None, "seconds": time.time() - t0}
    run.write_npz("preds", arrays)
    run.write_json("summary", plain(out))


def plain(obj):
    """numpy scalars and arrays as Python values, so the JSON writer turns every non-finite
    float into null or 'inf' (``interp.io._jsonable`` returns a numpy scalar's NaN as is)."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# ------------------------------------------------------------------------- the command line


def build_parser():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for it in ITEMS:
        p = sub.add_parser(it.cmd, help=f"{it.id}: {it.title}")
        p.add_argument("--check", action="store_true",
                       help="run the checks only (plan R2, R3) and write nothing")
        p.add_argument("--run", default=None,
                       help="a run name: outputs go to exploratory/<item>/<run>/")
        p.add_argument("--out", default=None,
                       help="a root outside outputs/costate (a partial run)")
        p.add_argument("--network", default=None,
                       help="one network only (a partial run; give --out)")
        p.add_argument("--limit", type=int, default=None,
                       help="the first N units only (a partial run; give --out)")
        p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                       help=f"worker processes of a pooled item (default {DEFAULT_WORKERS})")
        p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                       help=f"XLA intra-op threads (default {DEFAULT_THREADS})")
        p.add_argument("--nice", type=int, default=DEFAULT_NICE,
                       help=f"renice this process by this much first (default {DEFAULT_NICE})")
    sub.add_parser("verify", help="check the plan, the freeze and every bound file")
    sub.add_parser("status", help="list the items and their outputs")
    return ap


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.cmd == "verify":
        bad = verify()
        if bad:
            raise SystemExit("verify: " + "; ".join(bad[:20]))
        return
    if args.cmd == "status":
        print(json.dumps(status(), indent=1), flush=True)
        return
    if args.nice:
        os.nice(int(args.nice))
    t0 = time.time()
    run_item(BY_CMD[args.cmd], args)
    print(f"{args.cmd}: done in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
