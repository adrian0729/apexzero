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
- CS-9 with CS-10 (iii), N's CS-11 and N's CS-13 (plan Section 4 with amendment 9; the
  section 'CS-9: N'), in four stages (``--stage``): N's world from step 1's 12 finals'
  anchored keys at x_1, less the 6 K0 keys (targets only, pooled); K2b on 26 of N's keys
  (targets only, pooled) and N's decision exclusions; the networks at N (the critic's and 20
  random critics' pullbacks, the width derivative, lambda^own from x_1 over W - 1 steps and
  Pi^in,own; pooled, unlocks); and the analyses (C1, C2, C2-sw, C1-own, W1's comparisons and
  C2-pairs between N's states and T's plan states of the same stimulus; pooled).
- CS-6, CS-12, CS-14 and CS-15 (plan Section 4 with amendment 10; a section each): C1 without
  nulls at each decision network's 16 snapshots under its own observability, with the trends
  in log timesteps (``cs6``, one process, unlocks); the primary C2 and C2-sw of the 25
  networks outside the decision set against 20 pool actors computed on each one's z
  (``cs12``, pooled); q_c's window rule from the key plans and its stop rule, applied before
  K2b for q_c and any network (``cs14``, pooled, targets only); and the FWL check of the
  primary C2's probes, C2 with the actor's h1 and h2, and C2 on D (``cs15``, pooled). CS-12
  and CS-15 read CS-8's summary through its stamp (``item_output``); neither unlocks.
- BC-a0 (plan Section 4 with amendment 11; the section 'BC-a0: road beyond the preview'):
  targets only, no network. K2b's 26 keys re-solved by ``gates.k2b_solve`` on 8 road
  continuations drawn from the v1 training grammar beyond K2b's point s2, and 50 braking- and
  corner-phase interior states on continuations 1-3 placed 20 m beyond their own preview
  reach; the unseen-road shares U with key-bootstrap CIs, K2b's p90 beside them, and the
  route each points to (``bc_a0``, pooled). It does not unlock.
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
                   "--limit", "--stage"):
            skip = True
        elif not a.startswith("-"):
            return a
    return None


# The items with worker pools (plan Section 1, "pooled"), and the BLAS libraries' thread
# variables (scripts/costate.py's BLAS_THREADS): a pooled item pins BLAS to one thread before
# numpy loads, in this process and so in every worker it spawns.
POOLED_ITEMS = ("cs-4", "cs-5", "cs-7", "cs-8", "cs-9", "cs-10", "cs-11", "cs-12", "cs-14",
                "cs-15", "bc-a0")
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
    Item("CS-6", "C1-dev over the 16 snapshots", ("world", "critic:decision", "snapshots"),
         trained=True, needs=("CS-1",)),
    Item("CS-7", "Critic-layer and head-unit probes",
         ("world", "activations:decision", "checkpoints"), needs=("CS-1",)),
    Item("CS-8", "C2-sw's random-only null; C2-sw residualised; C2 with a decoy that fits "
         "geometry 6", ("world", "activations:decision", "gates:c2_null", "checkpoints")),
    Item("CS-9", "N: per-agent replication of C1, C2 and C2-sw, with K2b on N",
         ("world", "keys:n", "gates:k0", "gates:k2b", "critic:all", "own",
          "activations:decision", "checkpoints"), trained=True, solves=True),
    Item("CS-10", "C1-own on own-run states, then on N",
         ("world", "critic:decision", "own", "checkpoints"), trained=True, needs=("CS-9",)),
    Item("CS-11", "The policy's own width price Pi^in,own",
         ("world", "critic:decision", "checkpoints"), trained=True, needs=("CS-9",)),
    Item("CS-12", "C2 and C2-sw beyond the 9 decision networks (primary only)",
         ("world", "activations:other", "checkpoints"), needs=("CS-8",)),
    Item("CS-13", "C2-pairs", ("world", "activations:decision"), needs=("CS-9",)),
    Item("CS-14", "q_c, the action-projected critic signal, with K2b for q_c",
         ("world", "gates:k2b", "checkpoints"), trained=True, solves=True),
    Item("CS-15", "The FWL check, actor h1 and h2 probes, C2 on D",
         ("world", "activations:decision", "checkpoints"), needs=("CS-8",)),
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
                if p.is_file() and not p.name.endswith(FP.STAMP_SUFFIX) \
                        and BC_CACHE not in p.relative_to(d).parts:
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


# ------------------------------------------------------------------------- CS-9: N
#
# Plan Section 4 CS-9 with amendment 9 (its operational details), with CS-10 (iii), N's CS-11
# and N's CS-13. Four stages (``--stage``), each writing its own outputs under
# exploratory/cs-9/ and reading an earlier one's only through its stamp (``stage_file``):
# - build: N's world, step 1's 12 finals' anchored keys at x_1 (less the 6 K0 keys) with their
#   targets and network inputs (targets only; ``cs9_build``);
# - k2b: K2b on 26 of N's keys, a second corner beyond x_1's preview (targets only;
#   ``cs9_k2b``), and N's decision exclusions (T's frozen ones with N's own);
# - networks: every agent's critic pullbacks, the 20 random critics', its width derivative,
#   its own costate from x_1 (CS-10 (iii)) and its own width price (N's CS-11) at N
#   (``cs9_networks``, trained);
# - analyse: C1, C2 and C2-sw on N, C1-own on N, W1's comparisons on N and C2-pairs between N's
#   states and T's plan states of the same stimulus (``cs9_analyse``).

N_SEED = 20261025                        # plan R10: default_rng([N_SEED, item, unit])
N_ITEM = 9
N_K2B_KEYS = 26                          # amendment 9 item 3
N_BLOCK = 128                            # build units: at most this many keys of a stimulus
N_QUARTERS = 4                           # networks stage: units per agent (amendment 9 item 4)
N_STAGES = ("build", "k2b", "networks", "analyse")
N_STEP = 1                               # N's state: x_1 of an anchored key
N_AGENTS = CS1_ORDER + DESCRIPTIVE_SPECIALISTS
BUILD_TOL = 1e-12                        # the build's code at T's steps against T's world
K2B_TOL = 1e-6                           # k2b_rows at T's rows against gates/k2b.json (SD units)
ACTOR_TOL = 1e-12                        # the trained actor's h3 against the bound actor_h3
PRED_TOL = 1e-9                          # CS-13: T's predictions, the union fit against T's own
PAIR_DS, PAIR_DV = 2.5, 1.0              # CS-13's matched decoys: m, m/s (design Section 6)
E13_RHO = 0.4                            # E13.1's 'modest'
E13_COVER = 7                            # E13.2's 'at least 7 of 9'
SX_COLS = (0, 2)                         # s and xi among the K_P columns


def agent_pin(rec: dict, agent: str) -> str:
    """The record's network key of one of step 1's finals (S1-S3 are pinned by label)."""
    from apexzero.costate import analysis as AN

    return agent if agent in rec["networks"] else AN.network_of(agent)


def stage_rel(run: Run, name: str) -> str:
    """A stage output's path as a stamp records it (repository-relative, no local paths)."""
    base = run.root / EXP_REL
    try:
        return (Path(EXP_REL) / run.out.relative_to(base) / name).as_posix()
    except ValueError:
        return f"<outside>/{run.item.cmd}/{name}"


def stage_file(run: Run, name: str) -> Path:
    """An earlier stage's output of this run (amendment 9 item 1 'Stages'): its stamp must
    name CS-9 and the label exploratory, bind the file's bytes (a sidecar's sha256, or an
    embedded stamp's content_sha256), and not be partial unless this run is; its sha256 is
    recorded among the inputs."""
    from apexzero.fields import prereg as FP

    path = refuse_sealed(run.out / name)
    if not path.exists():
        raise SystemExit(f"{run.item.id}: {name} is missing: run its stage first")
    st = FP.read_stamp(path) or {}
    sha = sha256_file(path)
    if path.suffix == ".json":
        bound = st.get("content_sha256") == F.content_sha256(json.loads(path.read_text()))
    else:
        bound = (st.get("files") or {}).get(path.name) == sha
    bad = [why for why, ok in (
        ("its stamp is not an exploratory CS-9 stamp",
         st.get("label") == LABEL and st.get("item") == run.item.id),
        ("its stamp does not bind its bytes", bound),
        ("it is a partial run's output", bool(run.partial) or not st.get("partial")))
        if not ok]
    if bad:
        raise SystemExit(f"{run.item.id}: {name}: {'; '.join(bad)}")
    run.note_input(stage_rel(run, name), sha)
    return path


def rel_gap(got, want) -> float:
    """max |got - want| / max |want| over numeric arrays (NaN equal to NaN); 0 or inf for
    other arrays (equal or not)."""
    import numpy as np

    g, w = np.asarray(got), np.asarray(want)
    if g.shape != w.shape:
        return float("inf")
    if g.dtype.kind in "fc" and w.dtype.kind in "fciub":
        g, w = g.astype(float), w.astype(float)
        both = np.isnan(g) & np.isnan(w)
        if np.any(np.isnan(g) != np.isnan(w)):
            return float("inf")
        d = np.where(both, 0.0, np.abs(g - w))
        scale = float(np.max(np.abs(np.where(both, 0.0, w)), initial=0.0))
        return float(np.max(d, initial=0.0)) / max(scale, 1e-300)
    return 0.0 if np.array_equal(g, w) else float("inf")


def oj_take(oj, idx):
    """An ObsJac's rows ``idx``."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN

    idx = np.asarray(idx, int)
    return CR.ObsJac(**{f: np.asarray(getattr(oj, f))[idx] for f in IN.OBSJAC_FIELDS})


def inputs_concat(parts):
    import numpy as np

    from apexzero.costate import critic as CR

    return CR.Inputs(*(np.concatenate([np.asarray(getattr(p, f)) for p in parts])
                       for f in CR.Inputs._fields))


def oj_concat(parts):
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN

    return CR.ObsJac(**{f: np.concatenate([np.asarray(getattr(p, f)) for p in parts])
                        for f in IN.OBSJAC_FIELDS})


# -- build


def key_data_at(ref, bank, steps=None, t_H: int | None = None, kt=None) -> tuple:
    """(rows, ``critic.Inputs``, its KeyTargets, the routes' gap at the rows) of one key:
    ``inputs.key_data``'s rows and network inputs at ``steps`` (N: [1]; None: T's steps after
    K2's cut at t_H), without the ObsJac, which the caller evaluates per stimulus
    (amendment 9 item 2)."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import nulls as NL
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    cs, pr = kt.jac, kt.prices
    lm = bank.stimuli[ref.stim_id].landmarks
    rows = ST.key_rows(cs, pr, ref, ST.strata(ref), lm, steps=steps, t_H=t_H)
    t = np.asarray(rows["t"], int)
    rows["wall_side"] = IN.wall_side(cs, t)
    rows["delta_bound"] = cs.own_box[t, TG.DELTA]
    gam = float(kt.problem.spec.gamma)
    rows["v_star"] = NL.tail_values(cs.x, cs.a, gam)[t]
    rows["v_star_inf"] = NL.horizon_values(cs.x, cs.a, gam)[t]
    rows["a_steer"], rows["a_pedal"] = cs.a[t, 0], cs.a[t, 1]
    prev = np.where((t > 0)[:, None], cs.a[np.maximum(t - 1, 0)], 0.0)
    rows["a_prev_steer"], rows["a_prev_pedal"] = prev[:, 0], prev[:, 1]
    inp = CR.key_inputs(cs.x, cs.a, t, anchored=ref.kind == "anchored")
    gap = float(np.max(np.abs(cs.lam[t] - kt.cheap.lam[t])))
    return rows, inp, kt, gap


def build_wiring(world, bank, t_H: int | None, n_keys: int = 2) -> dict:
    """The build's code at T's steps on T's first keys against the bound world (amendment 9
    item 2): {key: the largest relative gap over the rows' columns, Inputs and ObsJac}."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import states as ST
    from apexzero.fields import network as NW

    a2 = {r.key_id: r for r in ST.a2_refs()}
    tab = world.tab
    out = {}
    for k in world.keys[:n_keys]:
        ref = a2[str(k.key_id)]
        rows, inp, _kt, _gap = key_data_at(ref, bank, None, t_H)
        oj = CR.obs_jacobians(NW.road(bank, ref.stim_id).base, inp)
        idx = np.asarray(k.rows, int)
        gaps = [rel_gap(v, np.asarray(tab[c])[idx]) if c in tab else float("inf")
                for c, v in rows.items()]
        gaps += [rel_gap(getattr(inp, f), getattr(k.inp, f)) for f in CR.Inputs._fields]
        gaps += [rel_gap(getattr(oj, f), getattr(k.oj, f)) for f in IN.OBSJAC_FIELDS]
        out[str(k.key_id)] = max(gaps)
    return out


def geometry_T(tab, geometries):
    """[n] each row's geometry index in T (``frozen.geometries``); SystemExit for a geometry
    outside T's (amendment 9 item 1)."""
    import numpy as np

    at = {(float(r), float(t)): i for i, (r, t) in enumerate(geometries)}
    out = np.empty(len(tab["R"]), int)
    for i, (r, t) in enumerate(zip(np.asarray(tab["R"], float), np.asarray(tab["theta"], float),
                                   strict=True)):
        j = at.get((float(r), float(t)))
        if j is None:
            raise SystemExit(f"CS-9: geometry (R {r}, theta {t}) is not among T's 48")
        out[i] = j
    return out


def cs9_build_unit(state, unit) -> dict:
    """One stimulus block of N's build in a worker (amendment 9 item 2): per key its targets
    and row at x_1; the block's Inputs and their ObsJac on the stimulus's road."""
    from apexzero.costate import critic as CR
    from apexzero.costate import targets as TG
    from apexzero.fields import network as NW

    t0 = time.time()
    bank = _bank()
    rows, inps, failed, mb, gaps = [], [], [], [], []
    for ref in unit["refs"]:
        try:
            r, inp, kt, gap = key_data_at(ref, bank, [N_STEP])
        except (TG.RepolishFailed, ValueError) as e:
            failed.append({"key_id": ref.key_id, "agent": ref.agent,
                           "error": f"{type(e).__name__}: {e}"})
            continue
        if bool(kt.jac.multi_basin):
            mb.append({"key_id": ref.key_id, "agent": ref.agent})
            continue
        rows.append(r)
        inps.append(inp)
        gaps.append(gap)
    out = {"stim_id": unit["stim_id"], "rows": rows, "failed": failed, "multi_basin": mb,
           "gap": gaps}
    if rows:
        inp = inputs_concat(inps)
        out["inp"] = inp
        out["oj"] = CR.obs_jacobians(NW.road(bank, unit["stim_id"]).base, inp)
    out["seconds"] = time.time() - t0
    return out


def n_refs_checked(run: Run) -> dict:
    """{agent: N's refs, by key id} (amendment 9 item 2): the index's lists against the
    record's key sets, their files against its content hashes (recorded as inputs)."""
    from apexzero.costate import keys_root
    from apexzero.costate import states as ST
    from apexzero.fields import prereg as FP

    refs = ST.n_refs()
    if sorted(refs) != sorted(N_AGENTS):
        raise SystemExit(f"CS-9: N's agents are {sorted(refs)}, not step 1's 12 finals")
    out = {}
    for agent in N_AGENTS:
        k = f"n:{agent}"
        want = [tuple(t) for t in run.rec["key_sets"][k]]
        got = sorted((r.kind, r.key_id) for r in refs[agent])
        if got != want:
            raise SystemExit(f"CS-9: the index's {k} is not the record's key set")
        h = FP.content_hash(want, keys_root())
        if h != run.rec["content"][k]:
            raise SystemExit(f"CS-9: the key files of {k} differ from the tagged ones")
        run.note_input(f"keys/{k}", h)
        out[agent] = sorted(refs[agent], key=lambda r: r.key_id)
    return out


def n_world_arrays(world) -> dict:
    """``inputs.save_world``'s arrays of a world (its layout; ``inputs.load_world`` reads them
    back)."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN

    arrays = {f"tab/{k}": np.asarray(v) for k, v in world.tab.items()}
    meta = []
    for i, k in enumerate(world.keys):
        meta.append(k.key_id)
        arrays[f"key/{i}/rows"] = np.asarray(k.rows)
        for f, v in zip(CR.Inputs._fields, k.inp, strict=True):
            arrays[f"key/{i}/inp/{f}"] = np.asarray(v)
        for f in IN.OBSJAC_FIELDS:
            arrays[f"key/{i}/oj/{f}"] = np.asarray(getattr(k.oj, f))
    arrays["keys.json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    return arrays


def build_descriptives(tab, gaps) -> dict:
    """Keys and rows per agent, by phase; the interior, contact and pedal-switching shares;
    geometries per agent; the routes' gap (amendment 9 item 2)."""
    import numpy as np

    from apexzero.costate import states as ST

    agent = np.asarray(tab["agent"]).astype(str)
    phase = np.asarray(tab["phase"]).astype(str)
    out = {}
    for a in N_AGENTS:
        m = agent == a
        out[a] = {"rows": int(m.sum()),
                  "by_phase": {p: int((m & (phase == p)).sum()) for p in ST.PHASES},
                  "interior_by_phase": {p: int((m & (phase == p)
                                                & np.asarray(tab["interior"], bool)).sum())
                                        for p in ST.PHASES},
                  "interior_share": float(np.mean(np.asarray(tab["interior"], bool)[m]))
                  if m.any() else None,
                  "contact_share": float(np.mean(np.asarray(tab["contact"], bool)[m]))
                  if m.any() else None,
                  "pedal_switching_share": float(np.mean(
                      np.asarray(tab[ST.SW_STRATUM], bool)[m])) if m.any() else None,
                  "geometries": len(np.unique(np.asarray(tab["geometry_T"])[m]))}
    g = np.asarray(gaps, float)
    out["route_gap"] = {"max": float(g.max()) if len(g) else None,
                        "median": float(np.median(g)) if len(g) else None}
    return out


def cs9_build(run: Run, world) -> None:
    """N's world (amendment 9 item 2): the wiring at T's steps, the keys checked against the
    record, the K0 keys dropped, the pooled build and its descriptives; build.json and
    n_world.npz."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import nulls as NL
    from apexzero.costate import states as ST

    t0 = time.time()
    frozen = run.rec["frozen"]
    bank = _bank()
    wiring = build_wiring(world, bank, frozen.get("t_H"))
    if not all(v <= BUILD_TOL for v in wiring.values()):
        raise SystemExit(f"CS-9: the build's code does not reproduce T's world: {wiring}")
    refs = n_refs_checked(run)
    k0 = set(json.loads(run.read("gates/k0.json").read_text())["summary"]["not_ok_descriptive"])
    keep, dropped = [], []
    for a in N_AGENTS:
        for r in refs[a]:
            (dropped if r.key_id in k0 else keep).append(r)
    if sorted(r.key_id for r in dropped) != sorted(k0):
        raise SystemExit("CS-9: a K0 key is not among N's keys")
    by_stim: dict = {}
    for r in keep:
        by_stim.setdefault(int(r.stim_id), []).append(r)
    units, names = [], []
    for sid in sorted(by_stim):
        rs = by_stim[sid]
        for b, i in enumerate(range(0, len(rs), N_BLOCK)):
            units.append({"stim_id": sid, "refs": rs[i:i + N_BLOCK]})
            names.append(f"stim{sid}/{b}")
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, names = units[:int(lim)], names[:int(lim)]
    print(f"CS-9 build: {sum(len(u['refs']) for u in units)} keys in {len(units)} units "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = run.pool(cs9_build_unit, units, names=names)
    pool_s = time.time() - t1
    tables, keys, at, gaps, failed, mb = [], [], 0, [], [], []
    for name, g in zip(names, got, strict=True):
        failed += g["failed"]
        mb += g["multi_basin"]
        gaps += g["gap"]
        if not g["rows"]:
            continue
        n = len(g["rows"])
        tables += g["rows"]
        keys.append(NL.KeyInput(name, np.arange(at, at + n), g["inp"], g["oj"]))
        at += n
    tab = ST.concat(tables)
    tab["geometry_T"] = geometry_T(tab, frozen["geometries"])
    wn = IN.World(tab, keys)
    per_agent_keys = {a: {"index": len(refs[a]),
                          "k0_dropped": sum(r.agent == a for r in dropped),
                          "multi_basin": sum(m["agent"] == a for m in mb),
                          "failed": sum(f["agent"] == a for f in failed)} for a in N_AGENTS}
    out = {"item": "CS-9", "stage": "build", "keys": per_agent_keys,
           "k0_dropped": sorted(r.key_id for r in dropped), "failed": failed,
           "multi_basin": mb, "n_rows": int(wn.n), "n_entries": len(keys),
           "n_stimuli": len(by_stim), "descriptives": build_descriptives(tab, gaps),
           "wiring": {"build_vs_T": wiring, "tol": BUILD_TOL},
           "definitions": "plan Section 4 CS-9 and amendment 9 items 1 and 2",
           "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_npz("n_world", n_world_arrays(wn))
    run.write_json("build", plain(out))


def load_n_world(run: Run):
    """N's world from the build stage's output (its stamp checked)."""
    from apexzero.costate import inputs as IN

    CS._x64()
    return IN.load_world(stage_file(run, "n_world.npz"))


def n_tab_problems(tab, frozen: dict) -> list[str]:
    """N's table against the frozen geometries: every row's geometry_T is its (R, theta)."""
    import numpy as np

    geo = np.asarray(frozen["geometries"], float)[np.asarray(tab["geometry_T"], int)]
    ok = (np.array_equal(geo[:, 0], np.asarray(tab["R"], float))
          and np.array_equal(geo[:, 1], np.asarray(tab["theta"], float)))
    return [] if ok else ["N's geometry_T column is not its rows' (R, theta) in T"]


# -- K2b on N


def k2b_rows(ref, sd, bank, t=None, kt=None, severities=None) -> dict:
    """``gates.k2b_key`` on any rows t of one key (amendment 9 item 3): a second corner of the
    key's direction 20 m beyond the rows' furthest preview point, both severities, by
    ``gates.k2b_solve``, and |Delta lambda_k| / SD_T on K_P at the rows with their phases.
    t None gives T's rows as ``k2b_key`` reads them (``states.t_steps`` without K2's cut);
    N's row is t = [1]."""
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG
    from apexzero.env import racing

    severities = GT.K2B_SEVERITIES if severities is None else severities
    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    prob, cs = kt.problem, kt.jac
    lm = bank.stimuli[ref.stim_id].landmarks
    t = ST.t_steps(cs.x, lm, cs.W) if t is None else np.asarray(t, int)
    reach = float(np.max(cs.x[t, 0] + np.asarray(racing.preview_offsets(
        np.maximum(cs.x[t, 3], 0.0)[:, None]))[:, -1]))
    s2 = reach + GT.K2B_MARGIN
    n = len(np.asarray(prob.track.kappa))
    sigma = int(ST.strata(ref).direction)
    ph = ST.phases(cs.x[t, 0], lm, ST.brake_onset(cs, lm))
    out = {"key_id": ref.key_id, "agent": ref.agent, "t": t.tolist(), "s2": s2, "reach": reach,
           "severities": {}}
    for name, (R, th) in severities.items():
        delta = GT.second_corner(n, s2, R, th, sigma)
        new, log = GT.k2b_solve(prob, delta)
        if new is None:
            out["severities"][name] = {"ok": False, **log}
            continue
        cs2 = TG.costate(new, "jac")
        d = GT.k2_compare(cs, cs2, sd, t)
        out["severities"][name] = {
            "ok": True, "max": d.max(0).tolist(),
            "by_phase": {p: np.quantile(d[ph == p], 0.9, axis=0).tolist()
                         for p in ST.PHASES if np.any(ph == p)},
            "rows": d.tolist(), "phase": ph.tolist(), "n_solves": len(log["log"])}
    return out


def k2b_gap(got: dict, want: dict) -> float:
    """The largest difference of ``k2b_rows``' record from a stored ``gates.k2b_key`` record:
    reach and s2 (m) and every severity's rows (SD units); inf when a severity's success or
    phases differ."""
    import numpy as np

    gaps = [abs(float(got["reach"]) - float(want["reach"])),
            abs(float(got["s2"]) - float(want["s2"]))]
    for name, w in want["severities"].items():
        g = got["severities"].get(name) or {}
        if bool(g.get("ok")) != bool(w.get("ok")):
            return float("inf")
        if w.get("ok"):
            if list(g["phase"]) != list(w["phase"]):
                return float("inf")
            a, b = np.asarray(g["rows"], float), np.asarray(w["rows"], float)
            if a.shape != b.shape:
                return float("inf")
            gaps.append(float(np.max(np.abs(a - b), initial=0.0)))
    return max(gaps)


def k2b_picks(tab, n_keys: int = N_K2B_KEYS, item: int = N_ITEM) -> list[dict]:
    """K2b on N's keys (amendment 9 item 3): pick i takes the R tercile x theta class stratum
    i mod (their number) of N's distinct geometries, sorted, and a row of it drawn with
    default_rng([N_SEED, item, i]); a key already picked is skipped. [{'pick', 'row',
    'key_id', 'stratum'}]."""
    import numpy as np

    from apexzero.costate import states as ST

    key = np.asarray(tab["key_id"]).astype(str)
    geo = np.stack([np.asarray(tab["R"], float), np.asarray(tab["theta"], float)], 1)
    uniq, inv = np.unique(geo, axis=0, return_inverse=True)
    stratum = ST.r_tercile(uniq[:, 0], uniq[:, 1]) * 10 + ST.theta_class(uniq[:, 1])
    per_row = stratum[inv.ravel()]
    groups = sorted(set(per_row.tolist()))
    n_keys = min(int(n_keys), len(np.unique(key)))
    out, seen, i = [], set(), 0
    while len(out) < n_keys:
        if i > 100 * max(n_keys, 1) + 1000:
            raise RuntimeError("K2b on N: no new key in its strata")
        g = groups[i % len(groups)]
        idx = np.flatnonzero(per_row == g)
        j = int(idx[np.random.default_rng([N_SEED, int(item), i]).integers(len(idx))])
        if key[j] not in seen:
            seen.add(key[j])
            out.append({"pick": i, "row": j, "key_id": str(key[j]), "stratum": int(g)})
        i += 1
    return out


def cs9_k2b_unit(state, unit) -> dict:
    """One key of K2b on N in a worker (or the wiring key at T's rows)."""
    t0 = time.time()
    rec = k2b_rows(unit["ref"], state["sd"], _bank(), unit["t"])
    rec["seconds"] = time.time() - t0
    return rec


def union_exclusions(frozen_excl: dict, own: dict) -> dict:
    """{phase: components} T's frozen K2b exclusions together with N's own (the union, in
    K_P's order; amendment 9 item 3)."""
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    return {p: [c for c in TG.K_P if c in set(frozen_excl.get(p, [])) | set(own.get(p, []))]
            for p in ST.PHASES}


def cs9_k2b(run: Run, world) -> None:
    """K2b on N (amendment 9 item 3): the wiring key, the 26 picks, pooled re-solves, N's rule
    and the union with T's frozen exclusions; k2b.json."""
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import states as ST

    t0 = time.time()
    frozen = run.rec["frozen"]
    wn = load_n_world(run)
    bad = n_tab_problems(wn.tab, frozen)
    if bad:
        raise SystemExit(f"CS-9: {'; '.join(bad)}")
    sd = GT.lam_sd(world.tab)
    lim = getattr(run.args, "limit", None)
    picks = k2b_picks(wn.tab, N_K2B_KEYS if lim is None else min(N_K2B_KEYS, int(lim)))
    refs = {r.key_id: r for v in n_refs_checked(run).values() for r in v}
    stored = json.loads(run.read("gates/k2b.json").read_text())
    w0 = stored["keys"][0]
    a2 = {r.key_id: r for r in ST.a2_refs()}
    units = [{"ref": a2[w0["key_id"]], "t": None}]
    units += [{"ref": refs[p["key_id"]], "t": [N_STEP]} for p in picks]
    names = [f"wiring/{w0['key_id']}"] + [f"pick{p['pick']}/{p['key_id']}" for p in picks]
    print(f"CS-9 k2b: {len(picks)} keys and the wiring key ({time.time() - t0:.0f} s)",
          flush=True)
    t1 = time.time()
    got = run.pool(cs9_k2b_unit, units, state={"sd": sd}, names=names)
    pool_s = time.time() - t1
    gap = k2b_gap(got[0], w0)
    if not gap <= K2B_TOL:
        raise SystemExit(f"CS-9: k2b_rows does not reproduce gates/k2b.json's first key "
                         f"({gap:.3g})")
    recs = got[1:]
    rule = GT.k2b_rule(recs)
    phases = sorted({p for r in recs for v in r["severities"].values() if v.get("ok")
                     for p in v["phase"]}, key=ST.PHASES.index)
    union = union_exclusions(frozen["k2b_excluded"], rule["excluded"])
    failed = {name: sum(not (r["severities"].get(name) or {}).get("ok") for r in recs)
              for name in GT.K2B_SEVERITIES}
    out = {"item": "CS-9", "stage": "k2b", "picks": picks, "keys": recs, "rule": rule,
           "phases_with_rows": phases, "excluded_T": frozen["k2b_excluded"],
           "excluded_N": rule["excluded"], "excluded_union": union,
           "failed_severities": failed, "sd_T": np.asarray(sd).tolist(),
           "wiring": {"key": w0["key_id"], "gap": gap, "tol": K2B_TOL,
                      "seconds": got[0]["seconds"]},
           "E9.4": {"approach_excluded_N": rule["excluded"].get("approach", []),
                    "holds": bool({"n", "vx"} <= set(rule["excluded"].get("approach", [])))},
           "definitions": "plan Section 4 CS-9 'K2b on N' and amendment 9 item 3",
           "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_json("k2b", plain(out))


# -- the networks at N


def own_restart(net, tr, x, prev_action, accel, maps0, count: int):
    """``own.own_costate`` from a restart state (``own_of``'s T restarts; N's x_1 with clip(a_0)
    and the accel of x_0 -> x_1): count steps, the start's on-manifold maps ``maps0``."""
    import numpy as np

    from apexzero.costate import own as OW
    from apexzero.fields import network as NW

    li = NW.LoopIn(np.asarray(x, float), np.asarray(prev_action, float),
                   np.asarray(accel, float), np.zeros((int(count), 2)), np.int32(0),
                   np.float64(np.inf), NW.ctx64(), NW.context_channels())
    return OW.own_costate(net, tr, li, int(count), int(count), reset=False, maps0=maps0)


def loop_from(x, prev_action, accel, steps: int = OWN_STEPS):
    """The loop inputs of a run from a restart state (N's x_1; amendment 9 item 4)."""
    import numpy as np

    from apexzero.fields import network as NW

    return NW.LoopIn(np.asarray(x, float), np.asarray(prev_action, float),
                     np.asarray(accel, float), np.zeros((int(steps), 2)), np.int32(0),
                     np.float64(np.inf), NW.ctx64(), NW.context_channels())


def own_horizon_from(net, tr, li, steps: int = OWN_STEPS) -> dict:
    """``own_horizon`` from a restart start (its prev_action and accel, not a reset)."""
    import numpy as np

    from apexzero.costate import own as OW

    r = OW.own_return(net, tr, li, np.asarray(li.x0, float)[None],
                      np.asarray(li.prev_action, float)[None], np.asarray(li.accel, float)[None],
                      steps)
    return {"H": int(r.H[0]), "terminated": bool(r.terminated[0]),
            "strip_end": bool(r.strip_end[0]), "J_return": float(r.J[0])}


def width_record(net, rd, li, lm, sig: int, horizon: int, hz: dict) -> dict:
    """Amendment 3 item 3's per-run record from the start ``li``, with ``horizon`` steps (W at
    D, W - 1 at N) for the comparison with Pi^in: Pi^in,own at min(horizon, H) and at H, the
    buffers, the kept rules and the J check (amendment 9 item 4)."""
    import numpy as np

    rec = dict(hz)
    H = rec["H"]
    if rec["terminated"]:
        rec.update(kept_W=False, kept_H=False, why_W="terminated", why_H="terminated")
        return rec
    cw = min(int(horizon), H)
    r = width_vjp(net, rd, li, cw, H)
    valid = r["valid"]
    alive_w = bool(H >= horizon and valid[:horizon + 1].all())
    b_w, why_w = buffer_of(r["x"], lm, sig, rd.base, horizon)
    b_h, why_h = buffer_of(r["x"], lm, sig, rd.base, H)
    kept_w = alive_w and np.isfinite(b_w) and b_w > 0
    kept_h = bool(valid[:H + 1].all()) and np.isfinite(b_h) and b_h > 0
    fw, fh = inside_field(r["field_W"], sig), inside_field(r["field_H"], sig)
    rec.update(J_W=float(r["J"][0]), J_H=float(r["J"][1]), J_env_W=float(r["J_env"][0]),
               J_env_H=float(r["J_env"][1]), B_W=b_w, B_H=b_h, kept_W=bool(kept_w),
               kept_H=bool(kept_h), pi_own_W=float(fw.sum()), pi_own_H=float(fh.sum()),
               why_W=drop_reason(kept_w, alive_w, why_w),
               why_H=drop_reason(kept_h, bool(valid[:H + 1].all()), why_h),
               j_gap=abs(float(r["J"][1]) - rec["J_return"]) / max(abs(rec["J_return"]),
                                                                    1e-300), cw=cw)
    return rec


NET_FIELDS = ("g", "g_held", "value", "random_g", "dV_dwin", "lam_own", "lam_own_held",
              "own_valid")
CS11_FIELDS = ("H", "terminated", "strip_end", "J_return", "kept_W", "kept_H", "pi_own_W",
               "pi_own_H", "B_W", "B_H", "J_W", "J_H", "j_gap")


def net_wiring(unit, net, crit, rcs, bank) -> dict:
    """The networks stage's wiring on T's first key (amendment 9 item 4), each gap against
    its tolerance (a failure stops the unit at once): the pullback and the random critic of
    seed 0 against the bound g and random_g[0] (per row, of its largest |g|), dV_dwin (of the
    network's largest |dV_dwin| on D) and, for the 9, lambda^own at T's first interior
    restart (of its largest |lambda|)."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import own as OW
    from apexzero.fields import network as NW

    w = unit["wiring"]
    pb = CR.pullback(crit, None, w["inp"], w["oj"])
    rg0 = CR.pullback(rcs[0], None, w["inp"], w["oj"]).g[:, 0]

    def per_row(got, want):
        sc = np.maximum(np.max(np.abs(want), axis=1), 1e-300)
        return float(np.max(np.max(np.abs(got - want), axis=1) / sc))

    rd = NW.road(bank, w["stim_id"])
    dv = IN.width_derivative(crit, rd, w["inp"], w["direction"])
    out = {"pullback": per_row(pb.g[:, 0], w["g"]), "random0": per_row(rg0, w["random0"]),
           "dV_dwin": float(np.max(np.abs(dv - w["dV_dwin"]))) / w["dv_scale"]}
    if not (out["pullback"] <= PULLBACK_TOL and out["random0"] <= PULLBACK_TOL
            and out["dV_dwin"] <= DV_TOL):
        raise RuntimeError(f"{unit['label']}: the critic at T's first key is not the stored "
                           f"one: {out}")
    r = w.get("restart")
    if r is not None:
        rd_r = NW.road(bank, r["stim_id"])
        oc = own_restart(net, rd_r.base, r["x"], r["prev_action"], r["accel"],
                         (r["Mx"], r["Ma"]), r["count"])
        want = np.asarray(r["lam_own"], float)
        out["lam_own"] = float(np.max(np.abs(oc.lam[0] - want))
                               / max(float(np.max(np.abs(want))), 1e-300))
        if not out["lam_own"] <= OW.RESTART_RTOL:
            raise RuntimeError(f"{unit['label']}: lambda^own at T's first restart is not the "
                               f"stored one ({out['lam_own']:.3g})")
    return out


def cs9_net_unit(state, unit) -> dict:
    """One (agent, quarter of its N keys) of the networks stage in a worker (amendment 9
    item 4): per stimulus block the critic's and the 20 random critics' pullbacks and the
    width derivative; per key lambda^own over W - 1 steps and the own width record; the
    checks on the unit's first kept key; the wiring on the agent's first unit."""
    import numpy as np

    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.fields import network as NW

    t0 = time.time()
    agent = unit["agent"]
    net, crit = own_nets(agent)
    rcs = CR.random_critics(agent.cfg, crit.obs_rms, crit.sigma, CR.RANDOM_SEEDS)
    bank = _bank()
    wiring = net_wiring(unit, net, crit, rcs, bank) if unit.get("wiring") else None
    cols: dict = {f: [] for f in ("rows", *NET_FIELDS, *CS11_FIELDS, "why_W", "why_H")}
    checks = fd = None
    for b in unit["blocks"]:
        rd = NW.road(bank, b["stim_id"])
        inp, oj = b["inp"], b["oj"]
        pb = CR.pullback(crit, None, inp, oj)
        rg = np.stack([CR.pullback(c, None, inp, oj).g[:, 0] for c in rcs], 1)  # [n, 20, 8]
        dv = IN.width_derivative(crit, rd, inp, b["direction"])
        lm = bank.stimuli[b["stim_id"]].landmarks
        if not np.allclose((lm.s_c0, lm.s_apex, lm.s_c1), b["lm"], rtol=0.0, atol=1e-6):
            raise RuntimeError(f"stimulus {b['stim_id']}: its landmarks are not N's columns")
        pa, acc = inp.prev_action, inp.accel
        for j, row in enumerate(b["rows"]):
            W = int(b["W"][j])
            oc = own_restart(net, rd.base, inp.x[j], pa[j], acc[j], (oj.Mx[j], oj.Ma[j]),
                             W - N_STEP)
            li = loop_from(inp.x[j], pa[j], acc[j])
            hz = own_horizon_from(net, rd.base, li)
            rec = width_record(net, rd, li, lm, b["direction"], W - N_STEP, hz)
            if "j_gap" in rec and not rec["j_gap"] <= J_TOL:            # amendment 5
                raise RuntimeError(f"row {row}: the width loop's J is not own_return's "
                                   f"({rec['j_gap']:.3g})")
            if checks is None and rec.get("kept_W") and rec.get("kept_H"):
                jv = width_jvp(net, rd, li, b["direction"], rec["cw"], rec["H"])
                gotp = np.array([rec["pi_own_W"], rec["pi_own_H"]])
                checks = {"row": int(row), "jvp": jv.tolist(), "vjp": gotp.tolist(),
                          "error": float(np.max(np.abs(jv - gotp))
                                         / max(float(np.max(np.abs(jv))), 1e-300))}
                if not checks["error"] <= JVP_TOL:
                    raise RuntimeError(f"row {row}: the width VJP is not the JVP "
                                       f"({checks['error']:.3g})")
                fd = {"row": int(row), "ad": rec["pi_own_W"],
                      **width_fd(net, rd, li, b["direction"], rec["cw"])}
                ref = fd["J_env"]["richardson"]
                fd["error_J_env"] = abs(fd["ad"] - ref) / max(abs(ref), 1e-300)
            cols["rows"].append(int(row))
            cols["g"].append(pb.g[j, 0])
            cols["g_held"].append(pb.g_held[j, 0])
            cols["value"].append(float(pb.value[j, 0]))
            cols["random_g"].append(rg[j])
            cols["dV_dwin"].append(float(dv[j]))
            cols["lam_own"].append(oc.lam[0])
            cols["lam_own_held"].append(oc.lam_held[0])
            cols["own_valid"].append(bool(oc.valid_all))
            for f in CS11_FIELDS:
                cols[f].append(rec.get(f, np.nan))
            cols["why_W"].append(rec.get("why_W", ""))
            cols["why_H"].append(rec.get("why_H", ""))
    out = {k: np.asarray(v) for k, v in cols.items()}
    return {"label": unit["label"], "cols": out, "checks": checks, "fd": fd, "wiring": wiring,
            "seconds": time.time() - t0}


def width_check_problems(units: list[dict], j_gap) -> list[str]:
    """The parent's re-check of the networks stage's width checks for one agent (amendment 9
    item 4): every finite J gap within ``J_TOL``, and the JVP check within ``JVP_TOL`` on each
    unit that kept a key at both horizons. A unit that kept none has no JVP check (its runs
    all terminated or lost their buffer), which is no failure."""
    import numpy as np

    bad = []
    jg = np.asarray(j_gap, float)
    fin = jg[np.isfinite(jg)]
    if fin.size and not float(fin.max()) <= J_TOL:
        bad.append(f"the width loop's J is not own_return's ({float(fin.max()):.3g})")
    for u in units:
        c = u.get("checks")
        kept = np.asarray(u["cols"]["kept_W"], bool) & np.asarray(u["cols"]["kept_H"], bool)
        if c is None and kept.any():
            bad.append(f"{u.get('label')}: a unit kept a key but ran no JVP check")
        elif c is not None and not c["error"] <= JVP_TOL:
            bad.append(f"the width VJP is not the JVP ({c['error']:.3g})")
    return bad


def n_blocks(wn, rows) -> list[dict]:
    """The world entries' parts on the given rows (an agent's): per stimulus entry its rows,
    Inputs, ObsJac, W, direction and landmark columns."""
    import numpy as np

    tab = wn.tab
    keep = np.zeros(wn.n, bool)
    keep[np.asarray(rows, int)] = True
    out = []
    for k in wn.keys:
        r = np.asarray(k.rows, int)
        j = np.flatnonzero(keep[r])
        if not len(j):
            continue
        rr = r[j]
        sid = np.unique(np.asarray(tab["stim_id"])[rr])
        dirs = np.unique(np.asarray(tab["direction"])[rr])
        if len(sid) != 1 or len(dirs) != 1:
            raise SystemExit(f"CS-9: world entry {k.key_id} spans stimuli {sid.tolist()}")
        lm = np.unique(np.stack([np.asarray(tab[c], float)[rr]
                                 for c in ("s_c0", "s_apex", "s_c1")], 1), axis=0)
        if len(lm) != 1:
            raise SystemExit(f"CS-9: world entry {k.key_id} has several landmarks")
        out.append({"stim_id": int(sid[0]), "direction": int(dirs[0]), "rows": rr,
                    "inp": k.inp.take(j), "oj": oj_take(k.oj, j),
                    "W": np.asarray(tab["W"], int)[rr], "lm": tuple(lm[0].tolist())})
    return out


def split_blocks(blocks: list[dict], parts: int) -> list[list[dict]]:
    """Whole blocks dealt into ``parts`` contiguous groups of about equal rows."""
    import numpy as np

    sizes = np.array([len(b["rows"]) for b in blocks])
    if not len(sizes):
        return []
    start = np.r_[0, np.cumsum(sizes)[:-1]]
    q = np.minimum((start * parts) // max(int(sizes.sum()), 1), parts - 1)
    return [[b for b, qq in zip(blocks, q, strict=True) if qq == i] for i in range(parts)
            if np.any(q == i)]


def t_wiring_inputs(run: Run, world, name: str, bank) -> dict:
    """The networks stage's wiring inputs of one network on T's first key (amendment 9 item
    4): its rows' Inputs and ObsJac with the bound g, random_g[0] and dV_dwin, and for a
    decision network T's first valid interior restart with the bound lambda^own."""
    import numpy as np

    from apexzero.costate import prereg as PR

    crit = read_critic(run, name, ("g", "random_g", "dV_dwin"))
    tab = world.tab
    k = world.keys[0]
    rows = np.asarray(k.rows, int)
    d0 = CS.d_rows(world)
    out = {"inp": k.inp, "oj": k.oj, "stim_id": int(tab["stim_id"][rows[0]]),
           "direction": int(tab["direction"][rows[0]]), "g": crit["g"][rows, 0],
           "random0": crit["random_g"][0][rows],
           "dV_dwin": crit["dV_dwin"][rows],
           "dv_scale": float(np.max(np.abs(crit["dV_dwin"][d0])))}
    f = PR.net_file(name)
    if f"post/own/{f}.npz" in (run.bound or {}):
        with np.load(run.read(f"post/own/{f}.npz")) as d:
            lam, valid, rr = d["lam_own"], d["valid"], d["rows"]
        i = next(int(i) for i in rr if valid[i] and np.isfinite(lam[i]).all())
        kk = next(q for q in world.keys if i in set(np.asarray(q.rows).tolist()))
        j = int(np.flatnonzero(np.asarray(kk.rows) == i)[0])
        inp = kk.inp.take([j])
        out["restart"] = {"row": i, "stim_id": int(tab["stim_id"][i]), "x": inp.x[0],
                          "prev_action": inp.prev_action[0], "accel": inp.accel[0],
                          "Mx": kk.oj.Mx[j], "Ma": kk.oj.Ma[j],
                          "count": int(tab["W"][i]) - int(tab["t"][i]), "lam_own": lam[i]}
    return out


def cs9_networks(run: Run, world) -> None:
    """The networks at N (amendment 9 item 4): per agent its units (the first carrying the
    wiring on T's first key), pooled; networks.npz (per agent, its rows' arrays in N's row
    order) and networks.json (the checks, drops and timing)."""
    import numpy as np

    t0 = time.time()
    wn = load_n_world(run)
    tab = wn.tab
    bank = _bank()
    agent_col = np.asarray(tab["agent"]).astype(str)
    want = getattr(run.args, "network", None)
    agents = [a for a in N_AGENTS if want is None or want in (a, agent_pin(run.rec, a))]
    if not agents:
        raise SystemExit(f"CS-9: --network {want} is not one of step 1's finals")
    units, names = [], []
    for a in agents:
        name = agent_pin(run.rec, a)
        pin = run.rec["networks"][name]
        ag = CS.load_agent(name, pin, run.root)
        run.note_input(pin["path"], pin["sha256"])
        wire = t_wiring_inputs(run, world, name, bank)
        parts = split_blocks(n_blocks(wn, np.flatnonzero(agent_col == a)), N_QUARTERS)
        for q, part in enumerate(parts):
            units.append({"label": a, "quarter": q, "agent": ag, "blocks": part,
                          "wiring": wire if q == 0 else None})
            names.append(f"{a}/q{q}")
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, names = units[:int(lim)], names[:int(lim)]
    print(f"CS-9 networks: {len(units)} units over {len(agents)} agents "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = run.pool(cs9_net_unit, units, names=names)
    pool_s = time.time() - t1
    arrays, nets = {}, {}
    by: dict = {}
    for u, g in zip(units, got, strict=True):
        by.setdefault(u["label"], []).append(g)
    for a, gs in by.items():
        cols = {k: np.concatenate([g["cols"][k] for g in gs]) for k in gs[0]["cols"]}
        order = np.argsort(cols["rows"], kind="mergesort")
        cols = {k: v[order] for k, v in cols.items()}
        if lim is None and not np.array_equal(cols["rows"], np.flatnonzero(agent_col == a)):
            raise SystemExit(f"CS-9 {a}: the pool did not return every N row of the agent")
        wiring = [g["wiring"] for g in gs if g["wiring"] is not None]
        checks = [g["checks"] for g in gs if g["checks"] is not None]
        jg = cols["j_gap"].astype(float)
        bad = width_check_problems(gs, jg)
        if lim is None and bad:
            raise SystemExit(f"CS-9 {a}: {'; '.join(bad)}")
        jvp = max((c["error"] for c in checks), default=None)
        why = {}
        for h in ("W", "H"):
            c: dict = {}
            for kept, wy in zip(cols[f"kept_{h}"], cols[f"why_{h}"], strict=True):
                if not (kept is True or kept == 1.0):
                    c[str(wy) or "?"] = c.get(str(wy) or "?", 0) + 1
            why[h] = c
        j_max = float(np.max(jg[np.isfinite(jg)], initial=0.0))
        nets[a] = {"run": agent_pin(run.rec, a), "rows": len(cols["rows"]),
                   "own_valid": int(np.sum(cols["own_valid"])),
                   "terminated": int(np.nansum(cols["terminated"].astype(float))),
                   "strip_end": int(np.nansum(cols["strip_end"].astype(float))),
                   "kept_W": int(np.nansum(cols["kept_W"].astype(float))),
                   "kept_H": int(np.nansum(cols["kept_H"].astype(float))), "dropped": why,
                   "wiring": wiring, "j_gap_max": j_max,
                   "j_gap_rows": int(np.isfinite(jg).sum()), "jvp": jvp,
                   "jvp_note": None if checks else "no unit kept a key at both horizons",
                   "jvp_checks": checks,
                   "fd_descriptive": [g["fd"] for g in gs if g["fd"] is not None],
                   "seconds": float(sum(g["seconds"] for g in gs))}
        for k, v in cols.items():
            if k in ("why_W", "why_H"):
                v = v.astype(str)
            elif k in ("terminated", "strip_end", "kept_W", "kept_H"):
                v = np.asarray([bool(x) and not np.isnan(float(x)) for x in v.tolist()])
            arrays[f"{a}/{k}"] = v
    out = {"item": "CS-9", "stage": "networks", "agents": nets, "steps": OWN_STEPS,
           "own_count": "W - 1 (399)",
           "tolerances": {"pullback": PULLBACK_TOL, "dV_dwin": DV_TOL, "J": J_TOL,
                          "jvp": JVP_TOL},
           "definitions": "plan Section 4 CS-9 and amendment 9 item 4",
           "pool_seconds": pool_s, "seconds": time.time() - t0}
    run.write_npz("networks", arrays)
    run.write_json("networks", plain(out))


# -- the analyses on N


def n_folds(T_tab, tabN):
    """N's folds (amendment 9 item 1): each row its geometry's outer and inner folds of
    partition 0 on T (``probes.folds``), the cluster T's geometry index."""
    import numpy as np

    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    foT = PB.folds(T_tab)
    gT = ST.geometry_index(T_tab)
    G = int(gT.max()) + 1
    outer_g = np.full(G, -1, int)
    inner_g = np.full((foT.n_folds, G), -9, int)
    outer_g[gT] = foT.outer
    inner_g[:, gT] = foT.inner
    g = np.asarray(tabN["geometry_T"], int)
    fo = PB.Folds(outer_g[g], inner_g[:, g], g, 0)
    fo.check()
    return fo, outer_g


def present_clusters(g, rows) -> tuple:
    """(cluster [n] re-indexed over the geometries present in ``rows``, their number, their T
    indices) (amendment 9 item 1 'Bootstraps')."""
    import numpy as np

    g = np.asarray(g, int)
    present = np.unique(g[np.asarray(rows, bool)])
    remap = np.full(int(g.max()) + 1 if len(g) else 1, -1)
    remap[present] = np.arange(len(present))
    return np.where(remap[g] >= 0, remap[g], 0), max(len(present), 1), present


def cross_fit_external(X, Y, fold, w, Xe, Ye, fold_e, k: int):
    """[F, n_e, m] the residuals of external rows under ``states.cross_fit``'s scheme on (X, Y,
    fold, w) (amendment 9 item 7): an external row of fold g gets, when g is held out, Ye minus
    the fit on the folds other than g, and when f != g is held out, the fit on the folds other
    than f and g. On the fit's own rows it is ``cross_fit``'s by-fold residual."""
    import numpy as np

    from apexzero.costate import states as ST

    X, Y, Xe, Ye = (np.asarray(a, float) for a in (X, Y, Xe, Ye))
    Y = Y[:, None] if Y.ndim == 1 else Y
    Ye = Ye[:, None] if Ye.ndim == 1 else Ye
    fold, fold_e, w = np.asarray(fold, int), np.asarray(fold_e, int), np.asarray(w, float)
    F = int(fold.max()) + 1
    out = np.full((F, len(Xe), Y.shape[1]), np.nan)
    for f in range(F):
        train = fold != f
        te = fold_e == f
        if te.any():
            out[f, te] = Ye[te] - ST.knn_fit(X[train], Y[train], w[train], k).predict(Xe[te])
        for g in range(F):
            pe = fold_e == g
            if g == f or not pe.any():
                continue
            fit = train & (fold != g)
            out[f, pe] = Ye[pe] - ST.knn_fit(X[fit], Y[fit], w[fit], k).predict(Xe[pe])
    return out


def match_pairs(n_side: dict, t_side: dict, ds_tol: float = PAIR_DS, dv_tol: float = PAIR_DV
                ) -> tuple:
    """(I, J): N rows and T rows of the same stimulus and direction with |delta (s - s_apex)|
    < ds_tol and |delta vx| < dv_tol (amendment 9 item 7)."""
    import numpy as np

    I, J = [], []
    sn, st = np.asarray(n_side["stim_id"]), np.asarray(t_side["stim_id"])
    for s in np.unique(sn):
        i = np.flatnonzero(sn == s)
        j = np.flatnonzero(st == s)
        if not len(j):
            continue
        ok = ((np.abs(n_side["ds_apex"][i][:, None] - t_side["ds_apex"][j][None]) < ds_tol)
              & (np.abs(n_side["vx"][i][:, None] - t_side["vx"][j][None]) < dv_tol)
              & (n_side["direction"][i][:, None] == t_side["direction"][j][None]))
        a, b = np.nonzero(ok)
        I += i[a].tolist()
        J += j[b].tolist()
    return np.asarray(I, int), np.asarray(J, int)


def pairs_record(pred_n: dict, pred_t: dict, res_n, res_t, valid_n, valid_t, w_n, w_t,
                 geo_n, I, J, cols=SX_COLS) -> dict:
    """N's CS-13 statistic (amendment 9 item 7): per component, on the pairs whose two rows
    are in their masks and differ in lambda_res, the weighted Spearman of the predicted
    against the true delta lambda_res (N minus T) for the base and the full probe, and their
    paired difference, bootstrapped over the pairs' geometries."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    out = {"n_pairs": len(I), "components": {}}
    for k, nm in zip(range(len(cols)), SX, strict=True):
        kk = cols[k]
        keep = valid_n[I, kk] & valid_t[J, kk]
        dl = res_n[I, k] - res_t[J, k]
        keep &= np.abs(dl) > 0.0
        rec = {"n": int(keep.sum()), "geometries": 0}
        if keep.sum() < 3:
            out["components"][nm] = {**rec, "reason": "fewer than 3 pairs"}
            continue
        ii, jj = I[keep], J[keep]
        pw = w_n[ii] * w_t[jj]
        cl, G, _ = present_clusters(geo_n[ii], np.ones(len(ii), bool))
        counts = PB.cluster_counts(G)
        res = {}
        for p in ("base", "full"):
            x = pred_n[p][ii, k] - pred_t[p][jj, k]
            res[p] = AN.spearman_boot(x, dl[keep], pw, cl, counts)
        diff = res["full"][1] - res["base"][1]
        rec.update(geometries=int(G),
                   base={"rho": res["base"][0], "ci": list(AN._ci(res["base"][1]))},
                   full={"rho": res["full"][0], "ci": list(AN._ci(res["full"][1]))},
                   difference={"rho": float(res["full"][0] - res["base"][0]),
                               "ci": list(AN._ci(diff))})
        out["components"][nm] = rec
    return out


def union_fit(t_side: dict, n_side: dict, names=("lam_s", "lam_xi")) -> dict:
    """N's CS-13 probes (amendment 9 item 7): base [z, m] and full [z, m, h3] ridge probes on
    T's rows and N's rows at weight 0, T's folds; with the fit on T's rows alone, and the
    largest gap of T's predictions between the two."""
    import numpy as np

    from apexzero.costate import probes as PB

    nT = len(t_side["w"])
    B = np.vstack([np.hstack([t_side["z"], t_side["m"]]), np.hstack([n_side["z"], n_side["m"]])])
    Hh = np.vstack([t_side["h3"], n_side["h3"]])
    Y = np.concatenate([t_side["res_by_fold"], n_side["res_by_fold"]], axis=1)
    w = np.concatenate([t_side["w"], np.zeros(len(n_side["z"]))])
    fo = PB.Folds(np.concatenate([t_side["outer"], n_side["outer"]]),
                  np.concatenate([t_side["inner"], n_side["inner"]], axis=1),
                  np.concatenate([t_side["geo"], n_side["geo"]]), 0)
    foT = PB.Folds(t_side["outer"], t_side["inner"], t_side["geo"], 0)
    BT = B[:nT]
    out, gaps = {}, {}
    for p, hid in (("base", None), ("full", Hh)):
        u = PB.ridge_cv(B, Y, fo, w, hid, names)
        t = PB.ridge_cv(BT, t_side["res_by_fold"], foT, t_side["w"],
                        None if hid is None else hid[:nT], names)
        gaps[p] = rel_gap(u.pred[:nT], t.pred)
        out[p] = (u.pred[:nT], u.pred[nT:])
    return {"pred_t": {p: v[0] for p, v in out.items()},
            "pred_n": {p: v[1] for p, v in out.items()}, "gap": gaps}


def cs13_expected(nets: dict) -> dict:
    """E13.1 and E13.2 read on N's pairs (descriptions; amendment 9 item 7)."""
    def comps(v):
        return [c for c in v["components"].values() if "base" in c]

    e1 = _count_of(nets, lambda v: bool(comps(v)) and all(
        0.0 < c[p]["rho"] < E13_RHO for c in comps(v) for p in ("base", "full")))
    e2 = _count_of(nets, lambda v: bool(comps(v)) and all(
        c["difference"]["ci"][0] <= 0.0 <= c["difference"]["ci"][1] for c in comps(v)))
    n = len(nets)
    return {"E13.1": {**e1, "holds": e1["count"] >= min(E_COUNT, n)},
            "E13.2": {**e2, "holds": e2["count"] >= min(E13_COVER, n)},
            "reading": "amendment 9 item 7 (E13.1 and E13.2 read on N's pairs, descriptions)",
            "n_networks": n}


def cs9_expected(nets: dict, k2b: dict) -> dict:
    """E9.1 to E9.5 (amendment 9 item 8) on the 9 decision networks."""
    nine = {k: v for k, v in nets.items() if k in CS1_ORDER}
    n = len(nine)
    e1 = _count_of(nine, lambda v: v["C1"]["decision"] == "refute")
    e2 = _count_of(nine, lambda v: v["C2"]["decision"] in ("inconclusive", "refute"))
    nt = [k for k, v in nine.items() if v["C2"]["decision"] == "not testable"]
    e3 = _count_of(nine, lambda v: v["C2-sw"]["decision"] == "refute")
    e5 = _count_of(nine, lambda v: v["C1-own"]["primary"].get("decision") == "refute")
    return {"E9.1": {**e1, "holds": e1["count"] == n},
            "E9.2": {**e2, "holds": e2["count"] == n, "not_testable": nt},
            "E9.3": {**e3, "holds": e3["count"] >= min(E_COUNT, n)},
            "E9.4": k2b.get("E9.4"),
            "E9.5": {**e5, "holds": e5["count"] >= min(E_COUNT, n)},
            "reading": "amendment 9 item 8", "n_networks": n}


def cs9_analyse_unit(state, unit) -> dict:
    """One agent's analyses on N in a worker (amendment 9 items 5 to 7): its mask, the actor
    at N, C1, C2, C2-sw, C1-own, W1's comparisons and (the 9) C2-pairs."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    t0 = time.time()
    frozen, labels, union = state["frozen"], state["labels"], state["union"]
    tab, arr = unit["tab"], unit["arrays"]
    agent = unit["agent"]
    n = len(tab["key_id"])
    fo = PB.Folds(unit["outer"], unit["inner"], np.asarray(tab["geometry_T"], int), 0)
    gT = np.asarray(tab["geometry_T"], int)
    d = np.asarray(tab["direction"])
    idx = [TG.STATE_NAMES.index(c) for c in TG.K_P]
    lam = CS._winsorised(tab, frozen, TG.K_P)
    bnd = CS._bounds(frozen, TG.K_P)
    valid = ST.decision_mask(tab, ~unit["s_unobservable"], union)
    phase = np.asarray(tab["phase"]).astype(str)
    scored = np.isin(phase, SCORED_PHASES)
    inner = np.asarray(tab["interior"], bool) & scored
    guard_rows(tab, inner)
    for c, k in zip(SX, K_SX, strict=True):
        guard_rows(tab, inner & valid[:, k], c, union, valid[:, k])
    # the actor at N (trained), and the pool on N's z
    o = np.clip(unit["z"], -10.0, 10.0)
    actor, _ = PB.agent_nets(agent)
    ha = PB.hidden(actor, o)
    h3, m = ha["h3"], ha["m"]
    pool = [PB.hidden(PB.random_actor(agent.cfg, s), o)["h3"] for s in PB.POOL_SEEDS]
    wire = {}
    if unit.get("t_side") is not None:
        ts = unit["t_side"]
        wire["actor_h3"] = float(np.max(np.abs(PB.hidden(actor, ts["z64"])["h3"]
                                               - ts["actor_h3_64"])))
        wire["pool_seed0"] = pool_actor_gap(agent.cfg, ts["z64"], ts["random0_64"])
        if not (wire["actor_h3"] <= ACTOR_TOL and wire["pool_seed0"] <= RANDOM_TOL):
            raise RuntimeError(f"{unit['label']}: the actor at T is not the bound one: {wire}")
    out: dict = {"run": unit["name"], "descriptive": unit["label"] not in CS1_ORDER,
                 "wiring": wire, "rows": int(n),
                 "decision_rows": {c: int((inner & valid[:, k]).sum())
                                   for c, k in zip(SX, K_SX, strict=True)}}
    # C1 on N (step 2)
    g8 = np.asarray(arr["g"], float)
    g = g8[:, idx]
    w1 = ST.stratum_weights(tab, inner)
    cl, G, present = present_clusters(gT, inner)
    counts = PB.cluster_counts(G)
    nulls = {"random_critics": [np.asarray(arr["random_g"], float)[:, s][:, idx]
                                for s in range(np.asarray(arr["random_g"]).shape[1])]}
    c1 = AN.c1(g, lam, fo.outer, w1, cl, G, d, inner, nulls, valid, TG.K_P, counts,
               slope_boot="always", bounds=bnd, logo="rule", labels=labels, geometries=present)
    out["C1"] = {**trim_c1_logo(c1.record()), "geometries": int(G),
                 "kappa_clipped_rows": int((inner & np.asarray(unit["kappa_clipped"],
                                                               bool)).sum())}
    # C2 on N (step 3)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, ST.weights(tab, "key"), k=8)
    sh_all, sh_in = ST.decoy_shares(tab, lam, res.oof)
    out["residual_shares"] = {"all": dict(zip(TG.K_P, sh_all.tolist(), strict=True)),
                              "interior": dict(zip(TG.K_P, sh_in.tolist(), strict=True)),
                              "power_floor_implied": ST.power_floor(sh_in),
                              "power_floor_used": frozen["power_floor"]}
    pdraws = PB.pool_draws(len(pool))
    w2 = ST.stratum_weights(tab, inner)
    try:
        c2 = AN.c2_network(h3, pool, o, m, res.by_fold, fo, None, d, TG.K_P,
                           frozen["power_floor"], inner, weights=w2[inner], mask=valid,
                           pool=pdraws, controls=PB.POOL_SEEDS, logo="rule", labels=labels)
    except PB.NotTestable as e:
        c2 = {"decision": "not testable", "reason": str(e)}
    out["C2"] = trim_logo(c2)
    # C2-sw on N (step 4)
    rs = np.asarray(tab[ST.SW_STRATUM], bool) & scored
    guard_rows(tab, rs)
    sig = ST.winsorise(np.asarray(tab["sigma_pedal"], float),
                       frozen["winsor_bounds"]["sigma_pedal"])
    ws = ST.stratum_weights(tab, rs)
    try:
        sw = AN.c2_sw_network(h3, pool, o, sig, m[:, 1], fo, ws[rs], rs, pool=pdraws,
                              controls=PB.POOL_SEEDS, logo="rule", labels=labels)
    except PB.NotTestable as e:
        sw = {"decision": "not testable", "reason": str(e)}
    out["C2-sw"] = trim_logo(sw)
    # C1-own on N (step 5, CS-10 (iii))
    lo = np.asarray(arr["lam_own"], float)[:, idx]
    loh = np.asarray(arr["lam_own_held"], float)[:, idx]
    ok = (inner & np.asarray(arr["own_valid"], bool) & np.isfinite(lo).all(1)
          & np.isfinite(g).all(1))
    guard_rows(tab, ok)
    for c, k in zip(SX, K_SX, strict=True):
        guard_rows(tab, ok & valid[:, k], c, union, valid[:, k])
    wo = ST.stratum_weights(tab, ok)
    clo, Go, _po = present_clusters(gT, ok)
    co = PB.cluster_counts(Go)
    own = {"primary": AN.c1_own(g, lo, fo.outer, wo, clo, Go, d, ok, unit["k3b"], TG.K_P, co,
                                valid, value_accuracy_pass=unit["va"])}
    lw = np.stack([ST.winsorise(lo[:, j], tuple(bnd[j])) for j in range(len(TG.K_P))], 1)
    cw = AN.c1(g, lw, fo.outer, wo, clo, Go, d, ok, None, valid, TG.K_P, co, slope_boot=False,
               bounds=bnd)
    own["winsorised"] = {**cw.record(), "descriptive": True}
    held = AN.c1_own(np.asarray(arr["g_held"], float)[:, idx], loh, fo.outer, wo, clo, Go, d,
                     ok, unit["k3b"], TG.K_P, co, valid, value_accuracy_pass=unit["va"])
    own["held"] = {**held, "descriptive": True, "status": "descriptive (the accel-held form)"}
    own["rows"] = {"scored": int(ok.sum()), "geometries": int(Go),
                   "invalid_runs_on_c1_rows": int((inner & ~np.asarray(arr["own_valid"],
                                                                       bool)).sum())}
    out["C1-own"] = own
    # N's CS-11 (step 6)
    ap = np.asarray(tab["interior"], bool) & (phase == "approach")
    kw = np.asarray(arr["kept_W"], bool)
    kh = np.asarray(arr["kept_H"], bool)
    dv = np.asarray(arr["dV_dwin"], float)
    own_w, own_h = np.asarray(arr["pi_own_W"], float), np.asarray(arr["pi_own_H"], float)
    pi_in = np.asarray(tab["Pi_in"], float)
    pb_in = [frozen["winsor_bounds"]["Pi_in"]]
    cs11: dict = {}
    for nm, rows in (("common", ap & kw & kh), ("kept_H", ap & kh), ("kept_W", ap & kw)):
        guard_rows(tab, rows)
        w = ST.stratum_weights(tab, rows)
        c_, G_, _p = present_clusters(gT, rows)
        cnt = PB.cluster_counts(G_)
        rec: dict = {"n": int(rows.sum()), "geometries": int(G_)}
        if rows.sum() >= 3:
            if nm in ("common", "kept_H"):
                rec["critic_own"] = agreement(dv, own_h, rows, w, fo.outer, c_, G_, cnt)
            if nm in ("common", "kept_W"):
                rec["own_optimum"] = agreement(own_w, pi_in, rows, w, fo.outer, c_, G_, cnt,
                                               pb_in)
            if nm == "common":
                rec["critic_optimum"] = agreement(dv, pi_in, rows, w, fo.outer, c_, G_, cnt,
                                                  pb_in)
                rec["medians"] = {k: float(AN.wmedian(v[rows], w[rows])) for k, v in (
                    ("pi_own_W", own_w), ("pi_own_H", own_h), ("Pi_in", pi_in),
                    ("dV_dwin", dv))}
                rec["zero_share"] = {"pi_own_W": float(np.mean(own_w[rows] == 0.0)),
                                     "pi_own_H": float(np.mean(own_h[rows] == 0.0))}
        cs11[nm] = rec
    out["CS-11"] = cs11
    arrays = {"row": unit["rows"], "valid": valid, "c1_rows": inner, "own_rows": ok,
              "sw_rows": rs, "c1_pred": c1.calibrated.pred.astype(np.float32),
              "cs11_common": ap & kw & kh, "res_oof": res.oof.astype(np.float32)}
    # N's CS-13 (step 7; the 9)
    if unit.get("t_side") is not None:
        ts = unit["t_side"]
        nr = np.flatnonzero(inner)
        ns = {"z": o[nr], "m": m[nr], "h3": h3[nr],
              "res_by_fold": unit["res_ext"][:, nr][:, :, list(SX_COLS)],
              "outer": fo.outer[nr], "inner": fo.inner[:, nr], "geo": gT[nr]}
        fit = union_fit(ts, ns)
        if not all(v <= PRED_TOL for v in fit["gap"].values()):
            raise RuntimeError(f"{unit['label']}: CS-13's union fit does not reproduce T's "
                               f"own: {fit['gap']}")
        n_side = {"stim_id": np.asarray(tab["stim_id"])[nr],
                  "ds_apex": np.asarray(tab["ds_apex"], float)[nr],
                  "vx": np.asarray(tab["vx"], float)[nr], "direction": d[nr]}
        I, J = match_pairs(n_side, ts)
        f_own = fo.outer[nr]
        res_n = unit["res_ext"][f_own, nr][:, list(SX_COLS)]
        wN = ST.stratum_weights(tab, inner)[nr]
        pr = pairs_record(fit["pred_n"], fit["pred_t"], res_n, ts["res_oof"], valid[nr],
                          ts["valid"], wN, ts["w"], gT[nr], I, J)
        pr["wiring"] = {"union_vs_T": fit["gap"], "tol": PRED_TOL}
        out["CS-13"] = pr
        arrays.update({"pairs_I": nr[I] if len(I) else I, "pairs_J": J})
    out["seconds"] = time.time() - t0
    return {"label": unit["label"], "record": out, "arrays": arrays}


def t_side_of(run: Run, world, name: str, fr, lam_t, by_fold, oof, stratum) -> dict:
    """CS-13's T side of one decision network (amendment 9 item 7): its stratum rows' z, h3,
    m (the bound activations), the frozen lambda_res of s and xi, its T mask, folds, stimuli,
    decoy columns and weights; and the first 64 rows for the actor's wiring."""
    import numpy as np

    from apexzero.costate import prereg as PR
    from apexzero.costate import states as ST

    f = PR.net_file(name)
    tab = world.tab
    r = np.flatnonzero(stratum)
    with np.load(run.read(f"post/activations/{f}.npz")) as d:
        z, h3, m = d["z"], d["actor_h3"], d["m"]
        r0 = d["random0_h3"][:64]
    gT = ST.geometry_index(tab)
    cols = list(SX_COLS)
    return {"z": np.clip(z[r], -10.0, 10.0), "h3": h3[r], "m": m[r],
            "res_by_fold": by_fold[:, r][:, :, cols], "res_oof": oof[r][:, cols],
            "valid": fr.valid[r], "outer": fr.fo.outer[r], "inner": fr.fo.inner[:, r],
            "geo": gT[r], "w": ST.stratum_weights(tab, stratum)[r],
            "stim_id": np.asarray(tab["stim_id"])[r],
            "ds_apex": np.asarray(tab["ds_apex"], float)[r],
            "vx": np.asarray(tab["vx"], float)[r],
            "direction": np.asarray(tab["direction"])[r],
            "z64": np.clip(z[:64], -10.0, 10.0), "actor_h3_64": h3[:64], "random0_64": r0}


def trim_c1_logo(rec: dict) -> dict:
    """A C1 record with its leave-one-geometry-out sets summarised: their number, their
    decisions counted and the most influential geometry."""
    lg = rec.get("logo")
    if not isinstance(lg, dict):
        return rec
    dec: dict = {}
    for d in lg.get("decision") or []:
        dec[str(d)] = dec.get(str(d), 0) + 1
    return {**rec, "logo": {"kind": lg.get("kind"), "n_sets": len(lg.get("geometry") or []),
                            "decisions": dec, "most_influential": lg.get("most_influential")}}


def brief_c2(rec: dict) -> dict:
    """A C2 or C2-sw record's headline fields."""
    keys = ("decision", "decision_all_geometries", "stat", "ci", "ci_geometry", "gain",
            "rank", "n_stratum", "scored_rows", "spearman_margin", "reason")
    return {k: rec.get(k) for k in keys if k in rec}


def cs9_analyse(run: Run, world) -> None:
    """The analyses on N (amendment 9 items 5 to 8): per agent, pooled; summary.json,
    cs10-iii.json, cs11-n.json, cs13-n.json and rows.npz."""
    import numpy as np

    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG
    from apexzero.fields import network as NW

    t0 = time.time()
    frozen = run.rec["frozen"]
    wn = load_n_world(run)
    tabN = wn.tab
    bad = n_tab_problems(tabN, frozen)
    if bad:
        raise SystemExit(f"CS-9: {'; '.join(bad)}")
    build = json.loads(stage_file(run, "build.json").read_text())
    k2b = json.loads(stage_file(run, "k2b.json").read_text())
    union = k2b["excluded_union"]
    with np.load(stage_file(run, "networks.npz")) as d:
        net_arr = {k: d[k] for k in d.files}
    foN, outer_g = n_folds(world.tab, tabN)
    if outer_g.tolist() != frozen["fold_partition0"]:
        raise SystemExit("CS-9: T's folds are not partition 0's")
    labels = np.asarray(frozen["geometries"], float)
    # T's side of CS-13: the frozen lambda_res and N's residuals by T's scheme
    lam_t, by_fold, oof, foT, res_gap = checked_lam_res(run, world.tab)
    k = PR.decoy_k(frozen) or ST.KNN_K
    Xt, wt = ST.decoy_d1(world.tab), ST.weights(world.tab, "key")
    own_t = cross_fit_external(Xt, lam_t, foT.outer, wt, Xt, lam_t, foT.outer, k)
    ext_gap = float(np.max(np.abs(own_t - by_fold)))
    if not ext_gap <= FIT_TOL:
        raise SystemExit(f"CS-9: the external cross-fit does not reproduce T's ({ext_gap:.3g})")
    lam_n = CS._winsorised(tabN, frozen, TG.K_P)
    res_ext = cross_fit_external(Xt, lam_t, foT.outer, wt, ST.decoy_d1(tabN), lam_n,
                                 foN.outer, k)
    phase_t = np.asarray(world.tab["phase"]).astype(str)
    stratum_t = np.asarray(world.tab["interior"], bool) & np.isin(phase_t, SCORED_PHASES)
    guard_rows(world.tab, stratum_t)
    agent_col = np.asarray(tabN["agent"]).astype(str)
    want = getattr(run.args, "network", None)
    agents = [a for a in N_AGENTS if want is None or want in (a, agent_pin(run.rec, a))]
    units = []
    for a in agents:
        name = agent_pin(run.rec, a)
        if f"{a}/rows" not in net_arr:
            raise SystemExit(f"CS-9: the networks stage has no arrays of {a}")
        rows = np.asarray(net_arr[f"{a}/rows"], int)
        mine = np.flatnonzero(agent_col == a)
        if not (np.array_equal(rows, mine) or (run.partial and np.isin(rows, mine).all())):
            raise SystemExit(f"CS-9 {a}: the networks stage's rows are not N's")
        pin = run.rec["networks"][name]
        ag = CS.load_agent(name, pin, run.root)
        run.note_input(pin["path"], pin["sha256"])
        rms = NW.upcast32(ag.obs_rms)
        z, cl, jz = IN.z_and_jz(wn, rms)
        ob = ST.observability(jz[rows], cl[rows])
        sub = {c: np.asarray(v)[rows] for c, v in tabN.items()}
        fl = {"kappa_clipped": IN.kappa_clipped(cl[rows])}
        k3b, va, ts = None, None, None
        if a in CS1_ORDER:
            f = PR.net_file(name)
            with np.load(run.read(f"post/own/{f}.npz")) as d:
                k3b = bool(d["k3b_pass"])
            k3 = json.loads(run.read(f"post/own/{f}.k3b.json").read_text())
            k3b = bool(k3b and (k3.get("decision") or {}).get("pass"))
            va = bool(run.decisions["networks"][name]["value_accuracy"]["pass"])
            _ag, _flags, fr = network_frame(run, name, world)
            ts = t_side_of(run, world, name, fr, lam_t, by_fold, oof, stratum_t)
        arrays = {kk.split("/", 1)[1]: v for kk, v in net_arr.items() if kk.startswith(a + "/")}
        units.append({"label": a, "name": name, "agent": ag, "tab": sub, "arrays": arrays,
                      "rows": rows, "z": z[rows], "s_unobservable": ob.s_unobservable,
                      "kappa_clipped": fl["kappa_clipped"], "outer": foN.outer[rows],
                      "inner": foN.inner[:, rows], "k3b": k3b, "va": va, "t_side": ts,
                      "res_ext": res_ext[:, rows]})
    print(f"CS-9 analyse: {len(units)} agents ({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    state = {"frozen": frozen, "labels": labels, "union": union}
    got = run.pool(cs9_analyse_unit, units, state=state, names=[u["label"] for u in units])
    pool_s = time.time() - t1
    nets = {g["label"]: g["record"] for g in got}
    rows_arr = {f"{g['label']}/{kk}": np.asarray(v) for g in got for kk, v in g["arrays"].items()}
    dec = run.decisions.get("networks") or {}

    def quoted(a: str, h: str) -> dict | None:
        r = (dec.get(agent_pin(run.rec, a)) or {}).get(h)
        if r is None:
            return None
        return _brief(r) if h in ("C1", "C1-own") else brief_c2(r)

    summary = {"item": "CS-9", "stage": "analyse",
               "agents": {a: {"run": v["run"], "descriptive": v["descriptive"],
                              "rows": v["rows"], "decision_rows": v["decision_rows"],
                              "C1": _brief(v["C1"]), "C1_record": v["C1"],
                              "C2": v["C2"], "C2-sw": v["C2-sw"],
                              "residual_shares": v["residual_shares"],
                              "wiring": v["wiring"],
                              "confirmatory_T": {"C1": quoted(a, "C1"), "C2": quoted(a, "C2"),
                                                 "C2-sw": quoted(a, "C2-sw")},
                              "seconds": v["seconds"]} for a, v in nets.items()},
               "k2b": {kk: k2b[kk] for kk in ("excluded_T", "excluded_N", "excluded_union",
                                              "phases_with_rows", "failed_severities", "E9.4")},
               "build": {kk: build[kk] for kk in ("keys", "n_rows", "descriptives")},
               "wiring": {"lam_res_shares": res_gap, "external_cross_fit": ext_gap},
               "definitions": "plan Section 4 CS-9 and amendment 9 items 1, 5, 6 and 8",
               "n_boot": PB.N_BOOT, "pool_seconds": pool_s,
               "seconds": time.time() - t0}
    summary["expected"] = cs9_expected(
        {a: {"C1": v["C1"], "C2": v["C2"], "C2-sw": v["C2-sw"], "C1-own": v["C1-own"]}
         for a, v in nets.items()}, k2b)
    own = {a: {"run": v["run"], "descriptive": v["descriptive"],
               "c1_own": _brief(v["C1-own"]["primary"]), "c1_own_record": v["C1-own"]["primary"],
               "winsorised": _brief(v["C1-own"]["winsorised"]),
               "held": _brief(v["C1-own"]["held"]), "rows": v["C1-own"]["rows"],
               "confirmatory": quoted(a, "C1-own")} for a, v in nets.items()}
    cs10 = {"item": "CS-10", "part": "(iii) on N (CS-9 step 5)", "agents": own,
            "count": "W - 1 (399) steps from x_1",
            "definitions": "plan Section 4 CS-9 step 5, CS-10, and amendment 9 items 4 and 5"}
    cs11 = {"item": "CS-11", "part": "on N (CS-9 step 6)",
            "agents": {a: v["CS-11"] for a, v in nets.items()},
            "expected": cs11_expected({a: v["CS-11"] for a, v in nets.items()
                                       if a in CS1_ORDER and "critic_own" in v["CS-11"]["common"]}),
            "reading": "E11.1 to E11.3 on N's common rows, descriptions (amendment 9 item 5)",
            "definitions": "plan Section 4 CS-9 step 6, CS-11, and amendment 9 items 4 and 5"}
    c13 = {a: v["CS-13"] for a, v in nets.items() if "CS-13" in v}
    cs13 = {"item": "CS-13", "part": "N's pairs (CS-9 step 7)", "networks": c13,
            "expected": cs13_expected(c13), "pair_tolerances": {"ds_apex": PAIR_DS,
                                                                "vx": PAIR_DV},
            "definitions": "plan Section 4 CS-9 step 7, CS-13, and amendment 9 item 7"}
    run.write_npz("rows", rows_arr)
    run.write_json("summary", plain(summary))
    run.write_json("cs10-iii", plain(cs10))
    run.write_json("cs11-n", plain(cs11))
    run.write_json("cs13-n", plain(cs13))


@computes("CS-9")
def cs9(run: Run) -> None:
    """CS-9 with CS-10 (iii), N's CS-11 and N's CS-13 (plan Section 4, amendment 9): the stages
    named by --stage (default all, in order)."""
    stage = getattr(run.args, "stage", None) or "all"
    stages = N_STAGES if stage == "all" else (stage,)
    world = load_world(run)
    for s in stages:
        t0 = time.time()
        {"build": cs9_build, "k2b": cs9_k2b, "networks": cs9_networks,
         "analyse": cs9_analyse}[s](run, world)
        print(f"CS-9 {s}: {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------- other items' outputs


def item_output(run: Run, item_id: str, name: str, base: Path | None = None) -> Path:
    """Another item's canonical output (amendment 10 item 1), read only through its stamp: the
    label exploratory, the item ``item_id``, the stamp binding the file's bytes (a sidecar's
    sha256, or an embedded stamp's content_sha256), and not partial unless this run is. Its
    sha256 is recorded among the inputs. ``base``: the exploratory root (default
    outputs/costate/exploratory)."""
    from apexzero.fields import prereg as FP

    root = Path(base) if base is not None else run.root / EXP_REL
    path = refuse_sealed(root / item_id.lower() / name)
    rel = f"{EXP_REL}/{item_id.lower()}/{name}"
    if not path.exists():
        raise SystemExit(f"{run.item.id}: {rel} is missing: run {item_id} first")
    st = FP.read_stamp(path) or {}
    sha = sha256_file(path)
    if path.suffix == ".json":
        bound = st.get("content_sha256") == F.content_sha256(json.loads(path.read_text()))
    else:
        bound = (st.get("files") or {}).get(path.name) == sha
    bad = [why for why, ok in (
        (f"its stamp is not an exploratory {item_id} stamp",
         st.get("label") == LABEL and st.get("item") == item_id),
        ("its stamp does not bind its bytes", bound),
        ("it is a partial run's output", bool(run.partial) or not st.get("partial")))
        if not ok]
    if bad:
        raise SystemExit(f"{run.item.id}: {rel}: {'; '.join(bad)}")
    run.note_input(rel, sha)
    return path


# ------------------------------------------------------------------------- CS-6: C1-dev
#
# Plan Section 4 CS-6 with amendment 10 item 2, on the 9 decision networks' 16 pinned
# snapshots each: the main V's on-manifold pullback at T, C1 without nulls under the snapshot's
# own observability of s (``c1_bare``), and the trends with log timesteps
# (``analysis.c1_dev``) of the statistic, R2_c and the Spearman of s and xi. The last
# snapshot (the network's own pin) is computed first and checked against the stored g, the
# decision mask and decisions.json's C1. One process, no pool (its measured cost is minutes).

CS6_SERIES = ("stat", "r2c_s", "r2c_xi", "spearman_s", "spearman_xi")
E61_HI = 0.2                             # analysis.C1_REFUTE: the CI upper end below it
PIN_FIELDS = ("path", "sha256", "timesteps")


def snapshot_pins(rec: dict, label: str) -> list[tuple[str, dict]]:
    """[(key, pin)] of a decision network's snapshots (the record's ``snapshots.<label>``), by
    timesteps."""
    pins = (rec.get("snapshots") or {}).get(label)
    if not pins:
        raise SystemExit(f"CS-6: the record pins no snapshot of {label}")
    return sorted(pins.items(), key=lambda kv: int(kv[1]["timesteps"]))


def main_v_pullback(net, world):
    """[n, 8] the main V's on-manifold g at T: ``critic.pullback`` per key on the bound world's
    Inputs and ObsJac, as the critic stage computed it (``critic_of``)."""
    import numpy as np

    from apexzero.costate import critic as CR

    g = np.full((world.n, 8), np.nan)
    for k in world.keys:
        g[k.rows] = CR.pullback(net, None, k.inp, k.oj).g[:, 0]
    return g


def row_gap(got, want) -> float:
    """max over rows of max |got - want| / that row's largest |want| (NaN equal to NaN; inf
    where the shapes or the NaNs differ)."""
    import numpy as np

    g, w = np.asarray(got, float), np.asarray(want, float)
    if g.shape != w.shape or g.ndim != 2 or np.any(np.isnan(g) != np.isnan(w)):
        return float("inf")
    d = np.where(np.isnan(w), 0.0, np.abs(g - w)).max(axis=1)
    sc = np.maximum(np.where(np.isnan(w), 0.0, np.abs(w)).max(axis=1), 1e-300)
    return float(np.max(d / sc, initial=0.0))


def c1_point(c1) -> dict:
    """A snapshot's C1 in brief (amendment 10 item 2): the statistic with its CI and band, and
    per component (s, xi) R2_c, the Spearman with its CI, the scored rows and the clamped
    folds."""
    import numpy as np

    cl = np.asarray(c1.calibrated.clamped)
    cl = cl.reshape(cl.shape[0], -1) if cl.ndim else cl.reshape(1, 1)
    return {"stat": float(c1.stat), "ci": [float(v) for v in c1.ci], "decision": c1.decision,
            "r2c": {c: float(c1.r2c[k]) for c, k in zip(SX, K_SX, strict=True)},
            "spearman": {c: float(c1.spearman[k]) for c, k in zip(SX, K_SX, strict=True)},
            "spearman_ci": {c: [float(v) for v in np.asarray(c1.spearman_ci)[k]]
                            for c, k in zip(SX, K_SX, strict=True)},
            "n_rows": {c: int(c1.n_rows[k]) for c, k in zip(SX, K_SX, strict=True)},
            "clamped_folds": {c: int(np.sum(cl[:, k])) if cl.shape[1] > k else None
                              for c, k in zip(SX, K_SX, strict=True)}}


def series_value(pt: dict, series: str) -> float:
    """One of ``CS6_SERIES`` from a snapshot's ``c1_point``."""
    if series == "stat":
        return float(pt["stat"])
    kind, comp = series.rsplit("_", 1)
    return float(pt[kind][comp])


def cs6_trends(points: dict) -> dict:
    """{series: ``analysis.c1_dev`` of the networks' (timesteps, value) pairs} (amendment 10
    item 2); a non-finite value is left out of its network's pairs."""
    import math

    from apexzero.costate import analysis as AN

    out = {}
    for sname in CS6_SERIES:
        pairs = {}
        for lb, pts in points.items():
            pr = [(int(p["timesteps"]), series_value(p, sname)) for p in pts]
            pairs[lb] = [(t, v) for t, v in pr if math.isfinite(v)]
        out[sname] = AN.c1_dev(pairs)
    return out


def cs6_expected(points: dict, trends: dict) -> dict:
    """E6.1 to E6.3 (amendment 10 item 2)."""
    below = {lb: sum(1 for p in pts if p["ci"][1] < E61_HI) for lb, pts in points.items()}
    total = {lb: len(pts) for lb, pts in points.items()}
    var_xi = trends["spearman_xi"]["variants"]
    return {
        "E6.1": {"snapshots_below_0.2": below, "snapshots": total,
                 "count": int(sum(below.values())), "of": int(sum(total.values())),
                 "holds": bool(all(below[lb] == total[lb] for lb in below))},
        "E6.2": {"rho": trends["spearman_xi"]["rho"], "variants": var_xi,
                 "holds": bool(var_xi and all(v == "confirm" for v in var_xi.values()))},
        "E6.3": {"rho": trends["stat"]["rho"], "variants": trends["stat"]["variants"],
                 "holds": None, "reading": "not predicted (plan E6.3): reported"},
        "reading": "amendment 10 item 2"}


@computes("CS-6")
def cs6(run: Run) -> None:
    """CS-6 (plan Section 4, amendment 10 item 2): per decision network, its last snapshot
    first (checked against the stored g, decisions.json's mask and C1), then the other 15;
    per snapshot the main V's pullback at T and C1 without nulls under its own observability;
    then the trends and the expected directions; outputs summary.json and g.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import probes as PB
    from apexzero.fields import network as NW

    t0 = time.time()
    world = load_world(run)
    frozen = run.rec["frozen"]
    k2b = frozen["k2b_excluded"]
    counts = PB.cluster_counts(AN.geometry_clusters(world.tab)[1])
    lim = getattr(run.args, "limit", None)
    points, wiring, arrays = {}, {}, {}
    for lb in decision_labels(run):
        name = AN.network_of(lb)
        stored = read_critic(run, name, ("g",))["g"][:, 0]
        pins = snapshot_pins(run.rec, lb)
        last_key, last = pins[-1]
        net_pin = run.rec["networks"][name]
        if any(str(last.get(f)) != str(net_pin.get(f)) for f in PIN_FIELDS):
            raise SystemExit(f"CS-6 {lb}: its last snapshot's pin is not the network's")
        order = [pins[-1], *pins[:-1]]
        if lim is not None:
            order = order[:int(lim)]
        pts = []
        for key, pin in order:
            t1 = time.time()
            ag = CS.load_agent(name, pin, run.root)
            run.note_input(pin["path"], pin["sha256"])
            net = CR.trained_critic(ag, pin["path"])
            g = main_v_pullback(net, world)
            flags = IN.flags(world, NW.upcast32(ag.obs_rms))
            fr = CS.c1_frame(world, frozen, flags)
            c1_rows(fr, k2b)                                     # the row guard (plan R5)
            c1 = CS.c1_bare(fr, g, counts)
            pt = {"snapshot": key, "timesteps": int(pin["timesteps"]), **c1_point(c1)}
            if key == last_key:
                gap = row_gap(g, stored)
                bad = mask_problems(fr.valid, name, run.decisions)
                cw = c1_wiring(c1, c1_reference(run.decisions, name)["C1"])
                wiring[lb] = {"snapshot": key, "g_gap": gap, "mask_sha256": mask_sha256(fr.valid),
                              "c1": cw, "tolerances": {"g": PULLBACK_TOL, "c1": WIRING_TOL}}
                if not (gap <= PULLBACK_TOL and not bad
                        and all(v <= WIRING_TOL for v in cw.values())):
                    raise SystemExit(f"CS-6 {lb}: the last snapshot does not reproduce the "
                                     f"stored network: g {gap:.3g}, mask {bad}, C1 {cw}")
            pt["seconds"] = time.time() - t1
            pts.append(pt)
            arrays[f"{lb}/{int(pin['timesteps'])}/g"] = g.astype(np.float32)
            arrays[f"{lb}/{int(pin['timesteps'])}/valid"] = np.asarray(fr.valid, bool)
            print(f"CS-6 {lb} @ {int(pin['timesteps'])}: {time.time() - t1:.1f} s", flush=True)
        points[lb] = sorted(pts, key=lambda p: p["timesteps"])
    trends = cs6_trends(points)
    out = {"item": "CS-6", "networks": {lb: {"run": AN.network_of(lb), "snapshots": pts,
                                             "wiring": wiring.get(lb)}
                                        for lb, pts in points.items()},
           "trends": trends, "series": list(CS6_SERIES),
           "expected": cs6_expected(points, trends),
           "definitions": "plan Section 4 CS-6 and amendment 10 items 1 and 2",
           "n_boot": len(counts), "partial": lim is not None, "seconds": time.time() - t0}
    run.write_npz("g", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-12: beyond the 9
#
# Plan Section 4 CS-12 with amendment 10 item 3, on the 22 generalists outside the decision set
# and S1-S3: the primary C2 on the interior's approach and exit rows and the primary C2-sw on
# the approach and exit pedal-switching rows, each against the 20 pool actors computed on the
# network's own z (seeds 0-4 checked against the stored controls). One pool unit per
# (network, statistic). No trained network is evaluated, so the item does not unlock.

CS12_KINDS = ("C2", "C2-sw")
E121_MIN, E122_GEN, E122_SPEC = 22, 15, 1


def other_networks(rec: dict) -> list[str]:
    """The record's networks outside the decision set: the 22 other generalists in
    ``analysis.GENERALISTS``' order, then S1-S3 (amendment 10 item 3)."""
    from apexzero.costate import analysis as AN

    dec = {AN.network_of(lb) for lb in CS1_ORDER}
    return [n for n in all_networks(rec) if n not in dec]


def pool_actors(cfg, o, stored: dict | None = None) -> tuple[list, dict]:
    """(the 20 pool actors' h3 on ``o``, {seed: gap}): ``probes.random_actor(cfg, s)``, s =
    0-19, and for each seed in ``stored`` ({seed: (h3, m)}) the largest |difference| of its
    rebuilt h3 and m from the stored ones (amendment 10 item 3)."""
    import numpy as np

    from apexzero.costate import probes as PB

    pool, gaps = [], {}
    for s in PB.POOL_SEEDS:
        hid = PB.hidden(PB.random_actor(cfg, int(s)), o)
        pool.append(hid["h3"])
        if stored and int(s) in stored:
            h, m = stored[int(s)]
            gaps[int(s)] = float(max(np.max(np.abs(hid["h3"] - h)), np.max(np.abs(hid["m"] - m))))
    return pool, gaps


def cs12_unit(state, unit) -> dict:
    """One (network, statistic) of CS-12 in a worker: the network's activations on the unit's
    rows, the 20 pool actors on its z (seeds 0-4 checked against the stored controls first),
    then ``analysis.c2_network`` (C2) or ``analysis.c2_sw_network`` (C2-sw) as the
    confirmatory primary, on the rows given (arrays already subset to them)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import targets as TG

    t0 = time.time()
    idx = unit["idx"]
    with np.load(unit["path"]) as d:
        o = np.clip(d["z"][idx], -10.0, 10.0)
        h3, m = d["actor_h3"][idx], d["m"][idx]
        stored = {s: (d[f"random{s}_h3"][idx], d[f"random{s}_m"][idx])
                  for s in range(PB.N_RANDOM_ACTORS)}
    pool, gaps = pool_actors(unit["cfg"], o, stored)
    worst = max(gaps.values())
    if not worst <= RANDOM_TOL:
        raise RuntimeError(f"CS-12 {unit['label']}/{unit['kind']}: the pool actors do not "
                           f"rebuild the stored controls ({worst:.3g})")
    draws = PB.pool_draws(len(pool))
    every = np.ones(len(idx), bool)
    try:
        if unit["kind"] == "C2":
            rec = AN.c2_network(h3, pool, o, m, unit["lam_res"], unit["fo"], None,
                                unit["direction"], TG.K_P, unit["power_floor"], every,
                                weights=unit["w"], mask=unit["valid"], pool=draws,
                                controls=PB.POOL_SEEDS, logo="rule", labels=unit["labels"])
        else:
            rec = AN.c2_sw_network(h3, pool, o, unit["sigma"], m[:, 1], unit["fo"], unit["w"],
                                   every, pool=draws, controls=PB.POOL_SEEDS, logo="rule",
                                   labels=unit["labels"])
    except PB.NotTestable as e:
        rec = {"decision": "not testable", "reason": str(e)}
    return {"record": trim_logo(rec), "random_gaps": gaps, "seconds": time.time() - t0}


def cs12_expected(nets: dict, specialists=DESCRIPTIVE_SPECIALISTS) -> dict:
    """E12.1 and E12.2 (amendment 10 item 3) on the ruled decisions and the pool ranks."""
    c2 = {k: (v.get("C2") or {}).get("decision") for k, v in nets.items()}
    ok = {k: d in ("inconclusive", "refute") for k, d in c2.items()}
    confirm = [k for k, d in c2.items() if d == "confirm"]
    untestable = [k for k, d in c2.items() if d == "not testable"]

    def rank1(v) -> bool:
        return ((v.get("C2-sw") or {}).get("pool") or {}).get("rank") == 1

    gen = {k: rank1(v) for k, v in nets.items() if k not in specialists}
    spec = {k: rank1(v) for k, v in nets.items() if k in specialists}
    return {
        "E12.1": {"decisions": c2, "inconclusive_or_refute": int(sum(ok.values())),
                  "of": len(c2), "confirm": confirm, "not_testable": untestable,
                  "holds": bool(sum(ok.values()) >= min(E121_MIN, len(c2)) and not confirm)},
        "E12.2": {"rank1_generalists": gen, "count_generalists": int(sum(gen.values())),
                  "of_generalists": len(gen), "rank1_specialists": spec,
                  "count_specialists": int(sum(spec.values())),
                  "holds": bool(sum(gen.values()) >= min(E122_GEN, len(gen))
                                and sum(spec.values()) <= E122_SPEC)},
        "reading": "amendment 10 item 3"}


def cs8_quotes(run: Run) -> dict:
    """The 9 decision networks' C2-sw on sigma^ped ((a)) and f_D reference C2 ((c)) on the same
    rows, from CS-8's summary read through its stamp (amendment 10 items 1 and 3)."""
    s = json.loads(item_output(run, "CS-8", "summary.json").read_text())
    sw = (s.get("a_b") or {}).get("networks") or {}
    c2 = ((s.get("c") or {}).get("networks")) or {}
    return {lb: {"C2-sw": brief_c2(sw[lb]["sigma"]) | {
                 "rank": ((sw[lb]["sigma"].get("pool") or {}).get("rank"))}
                 if lb in sw else None,
                 "C2": brief_c2(c2[lb]["frozen"]) if lb in c2 and "frozen" in c2[lb] else None}
            for lb in CS1_ORDER}


@computes("CS-12")
def cs12(run: Run) -> None:
    """CS-12 (plan Section 4, amendment 10 item 3): per network its frame and decision mask, its
    stamped activations, then one pool unit per (network, statistic) (``cs12_unit``); the
    decision networks' records quoted from CS-8; outputs summary.json."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    frozen = run.rec["frozen"]
    k2b = frozen["k2b_excluded"]
    _lam, by_fold, _oof, fo, res_gap = checked_lam_res(run, tab)
    labels = AN.geometry_labels(tab)
    direction = np.asarray(tab["direction"])
    rows_c = cs5_rows(tab)
    ic = np.flatnonzero(rows_c)
    w_c = ST.stratum_weights(tab, rows_c)[ic]
    rows_s = cs8_rows(tab)
    i_s = np.flatnonzero(rows_s)
    w_s = ST.stratum_weights(tab, rows_s)[i_s]
    sig = ST.winsorise(np.asarray(tab["sigma_pedal"], float),
                       frozen["winsor_bounds"]["sigma_pedal"])
    quotes = cs8_quotes(run)
    want = getattr(run.args, "network", None)
    names = [n for n in other_networks(run.rec) if want is None or want == n]
    if not names:
        raise SystemExit(f"CS-12: --network {want} is not one of the 25 networks outside the "
                         "decision set")
    units, unames, frames = [], [], {}
    for name in names:
        agent, _flags, fr = network_frame(run, name, world)
        for c, k in zip(SX, K_SX, strict=True):
            guard_rows(tab, rows_c & fr.valid[:, k], c, k2b, fr.valid[:, k])
        act = run.read(f"post/activations/{PR.net_file(name)}.npz", "stamped")
        frames[name] = {"decision_rows": {c: int((rows_c & fr.valid[:, k]).sum())
                                          for c, k in zip(SX, K_SX, strict=True)}}
        units.append({"label": name, "kind": "C2", "path": str(act), "cfg": agent.cfg,
                      "idx": ic, "lam_res": by_fold[:, ic], "fo": fo.subset(rows_c),
                      "direction": direction[ic], "power_floor": frozen["power_floor"],
                      "w": w_c, "valid": fr.valid[ic], "labels": labels})
        units.append({"label": name, "kind": "C2-sw", "path": str(act), "cfg": agent.cfg,
                      "idx": i_s, "sigma": sig[i_s], "fo": fo.subset(rows_s), "w": w_s,
                      "labels": labels})
        unames += [f"{name}/C2", f"{name}/C2-sw"]
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, unames = units[:int(lim)], unames[:int(lim)]
    print(f"CS-12: {len(units)} units ({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = run.pool(cs12_unit, units, names=unames)
    pool_s = time.time() - t1
    nets: dict = {}
    for u, r in zip(units, got, strict=True):
        v = nets.setdefault(u["label"], {"descriptive_specialist":
                                         u["label"] in DESCRIPTIVE_SPECIALISTS,
                                         **frames[u["label"]], "random_gaps": {}, "seconds": {}})
        v[u["kind"]] = r["record"]
        v["random_gaps"][u["kind"]] = max(r["random_gaps"].values())
        v["seconds"][u["kind"]] = r["seconds"]
    complete = {k: v for k, v in nets.items() if all(h in v for h in CS12_KINDS)}
    out = {"item": "CS-12", "networks": nets,
           "expected": cs12_expected(complete) if complete else None,
           "decision_networks_cs8": quotes,
           "rows": {"C2_stratum": int(rows_c.sum()), "C2-sw": int(rows_s.sum())},
           "controls": "the 20 pool actors, seeds 0-19, computed on each network's z (seeds "
                       "0-4 checked against the stored controls)",
           "residual_share_gap": res_gap, "random_tol": RANDOM_TOL,
           "definitions": "plan Section 4 CS-12 and amendment 10 items 1 and 3",
           "n_boot": PB.N_BOOT, "pool_seconds": pool_s, "partial": lim is not None,
           "seconds": time.time() - t0}
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-14: q_c
#
# Plan Section 4 CS-14 with amendment 10 item 4. The window rule (step 4) is computed first,
# from the key plans alone (targets only): a T interior row of the approach or exit phase
# passes when its plan states x_t ... x_{t+64} all lie in its own phase. K2b for q_c can only
# remove rows, so the stop rule (step 8, fewer than 200 rows) is applied to this count before
# K2b for q_c and before any network; the item then writes its counts and stops.

Q_WINDOW = 64                            # GAE's cut: the window t ... t + 64
Q_MIN_ROWS = 200                         # the plan's stop rule (step 8)
Q_BLOCK = 32                             # keys per pool unit
S_TOL = 1e-12                            # the plan's s_t against T's column (relative)


def window_pass(x, lm, s_b: float, t: int, W: int, phase: str,
                window: int = Q_WINDOW) -> tuple[bool, str, float]:
    """(passes, why not, s_{t+window} - s_t) of one T row under CS-14's window rule
    (amendment 10 item 4): t + window <= W and every plan state x_t ... x_{t+window} in the
    row's own phase (``states.phases``)."""
    import numpy as np

    from apexzero.costate import states as ST

    if t + window > W:
        return False, "beyond_W", float("nan")
    ph = ST.phases(np.asarray(x, float)[t:t + window + 1, 0], lm, s_b)
    length = float(x[t + window, 0] - x[t, 0])
    if np.all(ph == phase):
        return True, "", length
    return False, "phase", length


def cs14_unit(state, unit) -> dict:
    """One block of T's keys under CS-14's window rule in a worker (amendment 10 item 4): per
    key ``targets.key_targets`` (the plan T's targets read), its landmarks and brake onset,
    the wiring of its T rows against T's columns, then each scored row's window."""
    import numpy as np

    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    t0 = time.time()
    bank = _bank()
    cols: dict = {f: [] for f in ("row", "passes", "why", "length")}
    wiring = {"s": 0.0, "phase": 0, "s_b": 0, "W": 0}
    for kd in unit["keys"]:
        kt = TG.key_targets(kd["ref"], bank=bank)
        cs = kt.jac
        x, W = np.asarray(cs.x, float), int(cs.W)
        lm = bank.stimuli[kd["ref"].stim_id].landmarks
        s_b = ST.brake_onset(cs, lm)
        t = np.asarray(kd["t"], int)
        wiring["s"] = max(wiring["s"], rel_gap(x[t, 0], kd["s"]))
        wiring["phase"] += int(np.sum(ST.phases(x[t, 0], lm, s_b) != kd["phase"]))
        same_sb = (np.isnan(s_b) and np.isnan(kd["s_b"])) or s_b == kd["s_b"]
        wiring["s_b"] += int(not same_sb)
        wiring["W"] += int(W != kd["W"])
        for j in np.flatnonzero(kd["scored"]):
            ok, why, length = window_pass(x, lm, s_b, int(t[j]), W, str(kd["phase"][j]))
            cols["row"].append(int(kd["rows"][j]))
            cols["passes"].append(bool(ok))
            cols["why"].append(why)
            cols["length"].append(length)
    return {"cols": {k: np.asarray(v) for k, v in cols.items()}, "wiring": wiring,
            "seconds": time.time() - t0}


@computes("CS-14")
def cs14(run: Run) -> None:
    """CS-14 (plan Section 4, amendment 10 item 4): the window rule on T's interior approach
    and exit rows from the key plans, pooled by blocks of keys (``cs14_unit``), its wiring
    against T's columns, then the stop rule on its count; outputs summary.json. With fewer
    than ``Q_MIN_ROWS`` rows the item is 'not scorable under #8' and stops there (and, by the
    amendment, with more too, until a further amendment)."""
    import numpy as np

    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    ph = np.asarray(tab["phase"]).astype(str)
    scored = np.asarray(tab["interior"], bool) & np.isin(ph, SCORED_PHASES)
    guard_rows(tab, scored)
    a2 = {r.key_id: r for r in ST.a2_refs()}
    keys = []
    for k in world.keys:
        rr = np.asarray(k.rows, int)
        ref = a2.get(str(k.key_id))
        if ref is None:
            raise SystemExit(f"CS-14: T's key {k.key_id} is not an A2 key")
        keys.append({"ref": ref, "rows": rr, "t": np.asarray(tab["t"], int)[rr],
                     "s": np.asarray(tab["s"], float)[rr], "phase": ph[rr],
                     "s_b": float(np.asarray(tab["s_b"], float)[rr[0]]),
                     "W": int(np.asarray(tab["W"], int)[rr[0]]), "scored": scored[rr]})
    lim = getattr(run.args, "limit", None)
    units = [{"keys": keys[i:i + Q_BLOCK]} for i in range(0, len(keys), Q_BLOCK)]
    if lim is not None:
        units = units[:int(lim)]
    print(f"CS-14: {len(units)} units over {sum(len(u['keys']) for u in units)} keys "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = run.pool(cs14_unit, units, names=[f"keys{i}" for i in range(len(units))])
    pool_s = time.time() - t1
    cols = {k: np.concatenate([g["cols"][k] for g in got]) for k in got[0]["cols"]}
    wiring = {"s": max(g["wiring"]["s"] for g in got),
              **{f: int(sum(g["wiring"][f] for g in got)) for f in ("phase", "s_b", "W")},
              "s_tol": S_TOL}
    if not (wiring["s"] <= S_TOL and wiring["phase"] == 0 and wiring["s_b"] == 0
            and wiring["W"] == 0):
        raise SystemExit(f"CS-14: the key plans are not T's: {wiring}")
    rows = cols["row"].astype(int)
    passes = cols["passes"].astype(bool)
    if lim is None and not np.array_equal(np.sort(rows), np.flatnonzero(scored)):
        raise SystemExit("CS-14: the pool did not return every scored row")
    by_phase = {p: {"rows": int(np.sum(ph[rows] == p)),
                    "pass": int(np.sum(passes & (ph[rows] == p))),
                    "beyond_W": int(np.sum((cols["why"] == "beyond_W") & (ph[rows] == p)))}
                for p in SCORED_PHASES}
    n_pass = int(passes.sum())
    length = cols["length"].astype(float)
    lp = length[passes & np.isfinite(length)]
    la = length[np.isfinite(length)]
    kid = np.asarray(tab["key_id"]).astype(str)
    stop = n_pass < Q_MIN_ROWS
    out = {"item": "CS-14", "window": Q_WINDOW, "rows_scored_phases": len(rows),
           "rows_pass": n_pass, "by_phase": by_phase,
           "keys_with_a_passing_row": len(np.unique(kid[rows[passes]])),
           "window_length_m": {"all": {"median": float(np.median(la)) if la.size else None,
                                       "min": float(la.min()) if la.size else None,
                                       "max": float(la.max()) if la.size else None},
                               "passing": {"median": float(np.median(lp)) if lp.size else None,
                                           "min": float(lp.min()) if lp.size else None,
                                           "max": float(lp.max()) if lp.size else None}},
           "stop_rule": {"min_rows": Q_MIN_ROWS, "fires": bool(stop)},
           "decision": ("not scorable under #8" if stop else
                        "stopped before K2b for q_c: a further amendment fixes the rest"),
           "k2b_for_q_c": "not run: it can only remove rows (amendment 10 item 4)",
           "networks": "none evaluated",
           "expected": {e: {"holds": None, "reading": "not read: the item stopped at the "
                                                     "window rule (amendment 10 item 4)"}
                        for e in ("E14.1", "E14.2", "E14.3")},
           "wiring": wiring,
           "definitions": "plan Section 4 CS-14 (steps 4 and 8) and amendment 10 item 4",
           "pool_seconds": pool_s, "partial": lim is not None, "seconds": time.time() - t0}
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- CS-15
#
# Plan Section 4 CS-15 with amendment 10 item 5, on the 9 decision networks: (1) the primary
# C2 on the interior's approach and exit rows from its own probes (checked against CS-8 (c)'s
# f_D reference record), and the same with every full probe replaced by its two-stage (FWL)
# fit at its own penalties; (2) C2 with the actor's h1 or h2 against the pool's h1 or h2,
# computed here; (3) C2 on D's rows. Four pool units per network. No trained network is
# evaluated, so the item does not unlock.

CS15_PARTS = ("h3", "h1", "h2", "D")
CS8_REF_TOL = 1e-10                      # step 1's record against CS-8 (c)'s f_D reference
CS8_REF_FIELDS = ("stat", "ci", "ci_geometry", "gain", "raw", "r2_base", "r2_full")
E151_TOL = 0.01


def c2_probes(hs, z, m, Y, fo, w, names) -> tuple:
    """(the base probe, the full probes): ``probes.c2``'s conditional probes, base [z, m] and
    full [z, m, h] for each h in ``hs``, on the rows given."""
    import numpy as np

    from apexzero.costate import probes as PB

    n = len(w)
    base = np.hstack([np.asarray(z).reshape(n, -1), np.asarray(m).reshape(n, -1)])
    return PB.conditional(list(hs), base, Y, fo, w, names)


def c2_from_probes(pb, fulls, w, fo, direction, passing, mask, pool=None, controls=None,
                   labels=None, logo: str | None = "rule") -> tuple[dict, object]:
    """(record, ``probes.Gain``) of C2 from fitted probes (``fulls[0]`` the network's, the rest
    its controls), assembled as ``analysis.c2_network`` assembles its record on the rows
    given (amendment 10 item 5)."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB

    mk = np.asarray(mask, bool).reshape(len(w), -1)
    gn = PB.gain(pb, fulls[0], list(fulls[1:]), w, fo, direction, passing, None, None,
                 mask=mk, pool=pool, logo=logo, labels=labels)
    rec = gn.record()
    rec["n_stratum"] = len(w)
    rec["scored_rows"] = mk.sum(0).tolist()
    if controls is not None:
        rec["controls"] = AN.controls_label(controls, pool is not None)
    return rec, gn


def raw_gain(pb, full, w, fo, direction, passing, mask) -> object:
    """``probes.gain`` without controls: the network's own gain R2(full) - R2(base), its mean
    over the passing components and the geometry-bootstrap draws of that mean."""
    import numpy as np

    from apexzero.costate import probes as PB

    mk = np.asarray(mask, bool).reshape(len(w), -1)
    return PB.gain(pb, full, [], w, fo, direction, passing, None, None, mask=mk)


def one_component(passing: dict, comp: str) -> dict:
    """The power floor's verdicts with every component but ``comp`` off."""
    return {c: bool(v) and c == comp for c, v in passing.items()}


def paired(a, b) -> dict:
    """{point, ci} of b - a on the same draws (two ``probes.Gain``)."""
    import numpy as np

    from apexzero.costate import analysis as AN

    d = np.asarray(b.boot, float) - np.asarray(a.boot, float)
    return {"point": float(b.stat - a.stat), "ci": list(AN._ci(d))}


def fwl_record(pb, fulls, Yb, base_x, hs, w, fo, direction, passing, mask, pool, controls,
               labels, names) -> dict:
    """Step 1's FWL part (amendment 10 item 5): every full probe refitted by ``probes.fwl`` at
    its own penalties; the trained h3's raw gain one-stage and FWL (per component and their
    mean) with the paired difference, and the C2 statistic under FWL beside the one-stage one
    with the paired two-level difference."""
    from apexzero.costate import probes as PB

    fw = [PB.fwl(base_x, h, Yb, fo, w, f.alpha, names) for h, f in zip(hs, fulls, strict=True)]
    rec_one, g_one = c2_from_probes(pb, fulls, w, fo, direction, passing, mask, pool, controls,
                                    labels, None)
    rec_fwl, g_fwl = c2_from_probes(pb, fw, w, fo, direction, passing, mask, pool, controls,
                                    labels, None)
    raw = {}
    for nm, ps in (("mean", passing), *((c, one_component(passing, c)) for c in SX)):
        a = raw_gain(pb, fulls[0], w, fo, direction, ps, mask)
        b = raw_gain(pb, fw[0], w, fo, direction, ps, mask)
        raw[nm] = {"one_stage": float(a.stat), "fwl": float(b.stat), "difference": paired(a, b)}
    return {"raw_gain": raw,
            "statistic": {"one_stage": rec_one["stat"], "fwl": rec_fwl["stat"],
                          "fwl_ci": rec_fwl["ci"], "difference": paired(g_one, g_fwl)},
            "fwl_preds": fw[0].pred}


def cs8_ref_gap(rec: dict, ref: dict) -> float:
    """The largest |difference| of a C2 record from CS-8 (c)'s reference over
    ``CS8_REF_FIELDS`` (None as NaN, NaN equal to NaN; inf on a missing or misshapen field)
    and the pool rank (inf when it differs)."""
    import numpy as np

    worst = 0.0
    for f in CS8_REF_FIELDS:
        if f not in rec or f not in ref:
            return float("inf")
        a = np.asarray([np.nan if v is None else v for v in np.atleast_1d(rec[f]).tolist()],
                       float)
        b = np.asarray([np.nan if v is None else v for v in np.atleast_1d(ref[f]).tolist()],
                       float)
        if a.shape != b.shape or np.any(np.isnan(a) != np.isnan(b)):
            return float("inf")
        worst = max(worst, float(np.max(np.where(np.isnan(a), 0.0, np.abs(a - b)),
                                        initial=0.0)))
    if (rec.get("pool") or {}).get("rank") != (ref.get("pool") or {}).get("rank"):
        return float("inf")
    return worst


def cs15_unit(state, unit) -> dict:
    """One (network, part) of CS-15 in a worker (amendment 10 item 5): 'h3' the primary C2 from
    its own probes (checked against CS-8 (c)'s record at once) with its FWL part; 'h1' and
    'h2' C2 with that layer against the pool's same layer, computed here; 'D' the primary C2
    on D's rows. Arrays are already subset to the part's rows."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import probes as PB
    from apexzero.costate import targets as TG

    t0 = time.time()
    part, idx, cfg = unit["part"], unit["idx"], unit["cfg"]
    tag = f"CS-15 {unit['label']}/{part}"
    with np.load(unit["path"]) as d:
        o = np.clip(d["z"][idx], -10.0, 10.0)
        m = d["m"][idx]
        h = d["actor_h3" if part in ("h3", "D") else f"actor_{part}"][idx]
        stored0 = d["random0_h3"][idx]
        pool = ([d[f"random{s}_h3"][idx] for s in PB.POOL_SEEDS] if part in ("h3", "D")
                else None)
    if pool is None:                       # h1 or h2: the pool's same layer, computed here
        pool = []
        for s in PB.POOL_SEEDS:
            hid = PB.hidden(PB.random_actor(cfg, int(s)), o)
            if int(s) == 0:
                gap = float(np.max(np.abs(hid["h3"] - stored0)))
            pool.append(hid[part])
    else:
        gap = pool_actor_gap(cfg, o, stored0)
    if not gap <= RANDOM_TOL:
        raise RuntimeError(f"{tag}: the pool actor of seed 0 does not rebuild its stored h3 "
                           f"({gap:.3g})")
    draws = PB.pool_draws(len(pool))
    w, fo, direction, valid = unit["w"], unit["fo"], unit["direction"], unit["valid"]
    passing = unit["power_floor"]
    out: dict = {"label": unit["label"], "part": part, "random0_gap": gap}
    if part == "h3":
        names = tuple(f"lam_{c}" for c in TG.K_P)
        Yb = unit["lam_res"]
        hs = [h, *pool]
        pb, fulls = c2_probes(hs, o, m, Yb, fo, w, names)
        rec, _g = c2_from_probes(pb, fulls, w, fo, direction, passing, valid, draws,
                                 PB.POOL_SEEDS, unit["labels"])
        ref = unit.get("cs8_ref")
        if ref is not None:
            out["cs8_gap"] = cs8_ref_gap(rec, ref)
            if not out["cs8_gap"] <= CS8_REF_TOL:
                raise RuntimeError(f"{tag}: C2 does not reproduce CS-8 (c)'s reference "
                                   f"({out['cs8_gap']:.3g})")
        n = len(w)
        base_x = np.hstack([o.reshape(n, -1), m.reshape(n, -1)])
        fw = fwl_record(pb, fulls, Yb, base_x, hs, w, fo, direction, passing, valid, draws,
                        PB.POOL_SEEDS, unit["labels"], names)
        out["record"] = trim_logo(rec)
        out["fwl"] = {k: v for k, v in fw.items() if k != "fwl_preds"}
        out["preds"] = {"base": pb.pred, "full": fulls[0].pred, "fwl": fw["fwl_preds"],
                        "y": pb.y}
    else:
        try:
            rec = AN.c2_network(h, pool, o, m, unit["lam_res"], fo, None, direction, TG.K_P,
                                passing, np.ones(len(idx), bool), weights=w, mask=valid,
                                pool=draws, controls=PB.POOL_SEEDS, logo="rule",
                                labels=unit["labels"])
        except PB.NotTestable as e:
            rec = {"decision": "not testable", "reason": str(e)}
        out["record"] = trim_logo(rec)
    out["seconds"] = time.time() - t0
    return out


def cs15_expected(nets: dict) -> dict:
    """E15.1 to E15.3 (amendment 10 item 5)."""
    def get(v, part):
        return (v.get(part) or {}).get("record") or {}

    e1 = _count_of(nets, lambda v: abs(v["h3"]["fwl"]["raw_gain"]["mean"]["difference"]["point"])
                   <= E151_TOL)
    e1s = {lb: v["h3"]["fwl"]["statistic"]["difference"]["point"] for lb, v in nets.items()}
    e2 = {L: _count_of(nets, lambda v, L=L: get(v, L)["stat"] <= get(v, "h3")["stat"])
          for L in ("h1", "h2")}
    e3 = _count_of(nets, lambda v: get(v, "D").get("decision") in ("not testable",
                                                                     "inconclusive"))
    n = len(nets)
    return {
        "E15.1": {**e1, "statistic_difference": e1s,
                  "holds": bool(e1["count"] >= min(E_COUNT, n))},
        "E15.2": {**e2, "holds": bool(all(p["count"] >= min(E_COUNT, n) for p in e2.values()))},
        "E15.3": {**e3, "decisions": {lb: get(v, "D").get("decision") for lb, v in nets.items()},
                  "holds": bool(e3["count"] >= min(E_COUNT, n))},
        "reading": "amendment 10 item 5"}


@computes("CS-15")
def cs15(run: Run) -> None:
    """CS-15 (plan Section 4, amendment 10 item 5): the frozen lambda_res; CS-8 (c)'s reference
    records; per network its frame and decision mask; 4 pool units per network
    (``cs15_unit``); outputs summary.json and preds.npz."""
    import numpy as np

    from apexzero.costate import analysis as AN
    from apexzero.costate import prereg as PR
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST

    t0 = time.time()
    world = load_world(run)
    tab = world.tab
    frozen = run.rec["frozen"]
    k2b = frozen["k2b_excluded"]
    _lam, by_fold, _oof, fo, res_gap = checked_lam_res(run, tab)
    labels_geo = AN.geometry_labels(tab)
    direction = np.asarray(tab["direction"])
    s8 = json.loads(item_output(run, "CS-8", "summary.json").read_text())
    ref8 = ((s8.get("c") or {}).get("networks")) or {}
    rows_c = cs5_rows(tab)
    d_mask = np.zeros(world.n, bool)
    d_mask[CS.d_rows(world)] = True
    rows_d = guard_rows(tab, d_mask)
    strata = {"h3": rows_c, "h1": rows_c, "h2": rows_c, "D": rows_d}
    wts = {k: ST.stratum_weights(tab, r) for k, r in (("C", rows_c), ("D", rows_d))}
    units, names = [], []
    for lb in decision_labels(run):
        name = AN.network_of(lb)
        agent, _flags, fr = network_frame(run, name, world)
        act = run.read(f"post/activations/{PR.net_file(name)}.npz")
        ref = (ref8.get(lb) or {}).get("frozen")
        if ref is None:
            raise SystemExit(f"CS-15 {lb}: CS-8 (c) holds no f_D reference record")
        for part in CS15_PARTS:
            rows = strata[part]
            for c, k in zip(SX, K_SX, strict=True):
                guard_rows(tab, rows & fr.valid[:, k], c, k2b, fr.valid[:, k])
            idx = np.flatnonzero(rows)
            units.append({"label": lb, "part": part, "path": str(act), "cfg": agent.cfg,
                          "idx": idx, "lam_res": by_fold[:, idx], "fo": fo.subset(rows),
                          "direction": direction[idx], "valid": fr.valid[idx],
                          "w": wts["D" if part == "D" else "C"][idx],
                          "power_floor": frozen["power_floor"], "labels": labels_geo,
                          "cs8_ref": ref if part == "h3" else None})
            names.append(f"{lb}/{part}")
    lim = getattr(run.args, "limit", None)
    if lim is not None:
        units, names = units[:int(lim)], names[:int(lim)]
    print(f"CS-15: {len(units)} units ({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got = run.pool(cs15_unit, units, names=names)
    pool_s = time.time() - t1
    nets: dict = {}
    arrays: dict = {}
    for u, r in zip(units, got, strict=True):
        lb, part = r["label"], r["part"]
        v = nets.setdefault(lb, {"run": AN.network_of(lb)})
        v[part] = {k: x for k, x in r.items() if k not in ("label", "part", "preds")}
        v[part]["rows"] = len(u["idx"])
        if part == "h3":
            if not r.get("cs8_gap", float("inf")) <= CS8_REF_TOL:
                raise SystemExit(f"CS-15 {lb}: C2 does not reproduce CS-8 (c)'s reference")
            arrays[f"{lb}/rows"] = u["idx"]
            for k, x in r["preds"].items():
                arrays[f"{lb}/{k}"] = np.asarray(x, np.float32)
    for lb, v in nets.items():
        conf = ((run.decisions.get("networks") or {}).get(v["run"]) or {}).get("C2") or {}
        v["confirmatory_c2"] = {k: conf.get(k) for k in ("stat", "ci", "decision", "n_stratum")}
    complete = {lb: v for lb, v in nets.items() if all(p in v for p in CS15_PARTS)}
    out = {"item": "CS-15", "networks": nets,
           "expected": cs15_expected(complete) if complete else None,
           "rows": {"stratum": int(rows_c.sum()), "D": int(rows_d.sum())},
           "residual_share_gap": res_gap, "cs8_ref_tol": CS8_REF_TOL,
           "definitions": "plan Section 4 CS-15 and amendment 10 items 1 and 5",
           "n_boot": PB.N_BOOT, "pool_seconds": pool_s, "partial": lim is not None,
           "seconds": time.time() - t0}
    run.write_npz("preds", arrays)
    run.write_json("summary", plain(out))


# ------------------------------------------------------------------------- BC-a0: road beyond the preview
#
# Plan Section 4 BC-a0 with amendment 11. Targets only: no network output of any network is
# computed or read (plan R5 then leaves its rows unrestricted). K2b's 26 keys
# (gates/k2b.json) are re-solved by K2b's amplitude continuation and re-centring
# (``gates.k2b_solve``) on road continuations drawn from the v1 training grammar
# (``tracks.generator.fit_grammar`` on the R-dev circuits, as scripts/train_dr.py builds it):
# - per key, continuations 1-8 from s_next = max(s2, s_c1 + g_j), beyond every T state's
#   preview (``bc_units``, placement 'key');
# - per state, continuations 1-3 from 20 m beyond the state's own preview reach, on 50
#   braking- and corner-phase interior states (``bc_state_picks``, placement 'state').
# The unseen-road share U of each component and phase (``bc_share``), K2b's p90 beside it, and
# the route each share points to (``bc_route``: option (i) or (ii), a recommendation under
# decision #6).

BC_ITEM = 100                            # R10's item number for BC-a0 (amendment 11 item 1)
BC_J = 8                                 # continuations per key, j = 1..8
BC_J_STATE = 3                           # continuations per state, j = 1..3
BC_PICK_UNIT = 100                       # R10's unit of key i's state picks: 100 + i
BC_BOOT_UNIT = {"key": 1000, "state": 1001}   # R10's units of the key bootstraps
BC_ROUTE_BOUND = 0.1                     # U's CI upper end at most this: option (i)
BC_ROUTE_PHASES = ("braking", "corner")
BC_STATE_PHASES = ("braking", "corner")
BC_STATE_NAMES = ("s", "n", "xi", "vx", "vy", "r", "delta", "pedal")   # targets.STATE_NAMES
BC_K_P = ("s", "n", "xi", "vx")          # targets.K_P
BC_MIRROR_ODD = ("n", "xi", "vy", "r", "delta", "steer")   # targets.MIRROR_ODD
BC_LAM = tuple(f"lam_{c}" for c in BC_STATE_NAMES)
BC_PI = ("Pi_in", "Pi_out", "Pi_slip", "Pi_mu")
BC_SIGMA = ("sigma_steer", "sigma_pedal")
BC_COLS = BC_LAM + BC_PI + BC_SIGMA      # the 14 targets compared
BC_DX = ("s", "n", "vx")                 # the state's own move, descriptive
BC_DX_IDX = (0, 1, 3)
BC_TARGET_TOL = 1e-12                    # the re-polished targets against the bound world (rel.)
BC_DROP = {"key": 4, "state": 1}         # more failed re-solves than this: dropped (half)
BC_CAP_MARGIN = 0.6                      # the generator's inside-width cap (w_in - 1)|kappa|
BC_CACHE = "unit-cache"                  # finished units' records, for a resume (amendment 14)


def bc_odd(col: str) -> bool:
    """A compared target is mirror-odd (its variance is taken within direction): lam_c for c in
    targets.MIRROR_ODD, and sigma_steer (steer is odd)."""
    if col.startswith("lam_"):
        return col[4:] in BC_MIRROR_ODD
    return col == "sigma_steer"


def bc_grammar(run: Run | None = None, root: Path = ROOT) -> tuple:
    """(the v1 training grammar, {path: sha256} of its circuits): step 1's
    ``interp.support.grammar`` for the 'base' spec (G1's v1 grammar: ``fit_grammar`` with its
    defaults on the 12 raw R-dev centrelines, each name through ``lockbox_guard`` first, as
    scripts/train_dr.py builds it for a run without --real-widths; amendment 11 item 2). The
    circuits' files' sha256 are recorded."""
    from apexzero.interp import support as SU

    g, g_alt = SU.grammar(SU.GRAMMARS["base"])
    if g_alt is not None:
        raise SystemExit("BC-a0: the base grammar has a tilted share")
    files = {}
    for tr in SU.dev_centrelines():
        rel = f"data/tracks/tumftm/{tr.name}.csv"
        files[rel] = sha256_file(refuse_sealed(Path(root) / rel))
    if run is not None:
        for rel, h in files.items():
            run.note_input(rel, h)
    return g, files


def bc_strip(g, j: int, item: int = BC_ITEM) -> dict:
    """Continuation j's generated strip (amendment 11 item 2): ``generator.generate`` from a
    JAX key seeded by default_rng([N_SEED, item, j]), with the draws ``generate`` makes from the
    same key that the placement reads, replicated: the run-up 'first' (where its first corner
    token starts), gap[0] before ``generate`` overwrites it with the run-up (the grammar's gap
    draw g_j), gap[1], and the uniform u behind the first corner's direction s0."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from apexzero.tracks import generator as GN

    seed = int(np.random.default_rng([N_SEED, int(item), int(j)]).integers(0, 2**31 - 1))
    key = jax.random.PRNGKey(seed)
    kappa, _wl, _wr = GN.generate(key, g)
    keys = jax.random.split(key, 10)
    m = GN.MAX_SEGMENTS
    si = jax.random.randint(keys[4], (m,), 0, g.straight_len.shape[0])
    gap = g.gap_scale * g.straight_len[si] * jnp.exp(g.jitter * jax.random.normal(keys[5], (m,)))
    first = jax.random.uniform(keys[6], (), minval=500.0, maxval=900.0)
    u = float(jax.random.uniform(keys[8], ()))
    return {"j": int(j), "seed": seed, "kappa": np.asarray(kappa, float), "ds": float(GN.DS),
            "first": float(first), "gap0": float(gap[0]), "gap1": float(gap[1]), "u": u,
            "s0": -1 if u < 0.5 else 1}


def bc_strip_problems(strip: dict) -> list[str]:
    """The replicated draws against the strip ``generate`` returned: its first nonzero knot is
    the first one beyond 'first', of sign s0."""
    import numpy as np

    k = np.asarray(strip["kappa"], float)
    nz = np.flatnonzero(k != 0.0)
    if not len(nz):
        return [f"strip {strip['j']}: no corner"]
    want = int(np.floor(strip["first"] / strip["ds"])) + 1
    bad = []
    if int(nz[0]) != want:
        bad.append(f"strip {strip['j']}: first corner at knot {int(nz[0])}, not {want}")
    if int(np.sign(k[nz[0]])) != int(strip["s0"]):
        bad.append(f"strip {strip['j']}: first corner's sign is not s0")
    return bad


def bc_p_alt(gap: float, p_short: float, p_long: float) -> float:
    """The grammar's alternation probability after a gap (``generator.generate``): 1 below
    20 m, p_alt_short below 60 m, p_alt_long beyond."""
    return 1.0 if gap < 20.0 else (float(p_short) if gap < 60.0 else float(p_long))


def bc_delta(strip: dict, n: int, ds: float, s_next: float, s_c1: float, sigma: int,
             p_short: float, p_long: float, w_left=None, w_right=None) -> tuple:
    """(the sample perturbation, its record) of one continuation (amendment 11 item 2): the
    strip's knot curvatures as generated (signs included) from its first corner token on,
    shifted by a whole number J of knots so that the token starts at s_start = first + J ds,
    the first such point at or after s_next; widths unchanged. Knots at or before s_start are
    unchanged (the strip is 0 there). Recorded beside it: whether its first corner turns the
    key's own way (direction sigma), and the grammar's alternation probability for the gap
    s_start - s_c1 (``bc_p_alt``; p_alt_long beyond 60 m), descriptively."""
    import numpy as np

    from apexzero.fields import perturb as PT

    if abs(float(ds) - float(strip["ds"])) > 1e-12:
        raise ValueError(f"the strip's knot spacing {strip['ds']} is not the key's {ds}")
    J = int(np.ceil((float(s_next) - float(strip["first"])) / float(ds)))
    s_start = float(strip["first"]) + J * float(ds)
    gap = s_start - float(s_c1)
    src = np.asarray(strip["kappa"], float)
    i = np.arange(int(n)) - J
    ok = (i >= 0) & (i < len(src))
    k = np.zeros(int(n))
    k[ok] = src[i[ok]]
    rec = {"j": int(strip["j"]), "s_next": float(s_next), "s_start": s_start, "shift": J,
           "gap": gap, "p_alt": bc_p_alt(gap, p_short, p_long),
           "first_direction": int(strip["s0"]), "same_as_key": int(strip["s0"]) == int(sigma),
           "kappa_max": float(np.max(np.abs(k), initial=0.0))}
    if w_left is not None and w_right is not None:
        w_in = np.where(k > 0, np.asarray(w_left, float), np.asarray(w_right, float))
        ratio = np.where(k != 0, (w_in - 1.0) * np.abs(k), 0.0)
        rec["cap_ratio_max"] = float(np.max(ratio, initial=0.0))
        rec["cap_exceeded"] = bool(rec["cap_ratio_max"] > BC_CAP_MARGIN)
    return PT.Samples(k, np.zeros(int(n)), np.zeros(int(n))), rec


def bc_reach(s, vx):
    """[n] s + d_max(vx), the furthest preview point (``racing.preview_offsets``' last), as
    ``gates.k2b_key`` reads it."""
    import numpy as np

    from apexzero.env import racing

    s, vx = np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(vx, float))
    return s + np.asarray(racing.preview_offsets(np.maximum(vx, 0.0)[:, None]))[:, -1]


def bc_targets(cs, pr, t) -> dict:
    """The 14 compared targets and the state at plan steps t of one solution: lam [n, 8], Pi
    [n, 4] (in, out, slip, mu), sigma [n, 2] (steer, pedal) and x [n, 3] (s, n, vx), as
    ``states.key_rows`` reads them."""
    import numpy as np

    t = np.asarray(t, int)
    return {"lam": np.asarray(cs.lam, float)[t],
            "Pi": np.stack([np.asarray(pr.inside, float)[t], np.asarray(pr.outside, float)[t],
                            np.asarray(pr.slip, float)[t], np.asarray(pr.mu, float)[t]], 1),
            "sigma": np.asarray(cs.switch, float)[t],
            "x": np.asarray(cs.x, float)[t][:, list(BC_DX_IDX)]}


def bc_world_targets(tab, rows) -> dict:
    """The same targets from the bound world's columns at ``rows``."""
    import numpy as np

    rows = np.asarray(rows, int)

    def cols(names):
        return np.stack([np.asarray(tab[c], float)[rows] for c in names], 1)

    return {"lam": cols(BC_LAM), "Pi": cols(BC_PI), "sigma": cols(BC_SIGMA)}


def bc_log_summary(log: Sequence[dict], ok: bool) -> dict:
    """``gates.k2b_solve``'s log in brief (amendment 13): its solves, their IPOPT iterations
    (total, largest), the statuses counted, the last status, the most re-centring rounds at one
    amplitude, and for a failed re-solve the amplitude it stopped at and the last one it
    passed (0 if none)."""
    from collections import Counter

    log = list(log)
    amps = [float(e["amp"]) for e in log]
    rounds = Counter(amps)
    out = {"solves": len(log), "iterations": int(sum(int(e["iterations"]) for e in log)),
           "iterations_max": max((int(e["iterations"]) for e in log), default=0),
           "statuses": dict(sorted(Counter(str(e["status"]) for e in log).items())),
           "last_status": str(log[-1]["status"]) if log else None,
           "rounds_max": max(rounds.values(), default=0)}
    if not ok:
        stop = amps[-1] if amps else None
        out["amp_failed"] = stop
        out["amp_passed"] = max((a for a in amps if stop is not None and a < stop), default=0.0)
    return out


def bc_unit(state, unit) -> dict:
    """One BC-a0 re-solve in a worker: one key's continuation j read at its T rows ('key'),
    or one state's continuation j read at its row ('state'); or K2b's own first key ('wiring',
    ``cs9_k2b_unit``). The re-polished stored optimum's targets must equal the bound world's
    at the rows (``BC_TARGET_TOL``), or the unit stops the pool. A re-solve that fails, or
    raises, is recorded as failed."""
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import targets as TG

    if unit["kind"] == "wiring":
        return cs9_k2b_unit(state, unit)
    t0 = time.time()
    kt = TG.key_targets(unit["ref"], bank=_bank())
    t = np.asarray(unit["t"], int)
    base = bc_targets(kt.jac, kt.prices, t)
    gap = max(rel_gap(base[k], unit["want"][k]) for k in ("lam", "Pi", "sigma"))
    if not gap <= BC_TARGET_TOL:
        raise RuntimeError(f"BC-a0 {unit['key_id']}: the re-polished targets differ from the "
                           f"bound world's ({gap:.3g})")
    tr = kt.problem.track
    strip = state["strips"][int(unit["j"])]
    delta, rec = bc_delta(strip, len(np.asarray(tr.kappa)), float(tr.ds), unit["s_next"],
                          unit["s_c1"], unit["sigma"], *state["p_alt"], tr.w_left, tr.w_right)
    rec.update({"kind": unit["kind"], "key": int(unit["key"]), "key_id": unit["key_id"],
                "t": t.tolist(), "target_gap": gap})
    try:
        new, log = GT.k2b_solve(kt.problem, delta)
        err = None
    except Exception as e:  # noqa: BLE001 (a failed re-solve is counted, plan BC-a0 step 4)
        new, log, err = None, {"log": []}, f"{type(e).__name__}: {e}"
    rec["n_solves"] = len(log.get("log", []))
    rec["log"] = bc_log_summary(log.get("log", []), new is not None)
    if new is None:
        rec.update({"ok": False, "error": err})
    else:
        cs2 = TG.costate(new, "jac")
        got = bc_targets(cs2, TG.prices(new, cs2), t)
        rec.update({"ok": True, "d_lam": got["lam"] - base["lam"], "d_Pi": got["Pi"] - base["Pi"],
                    "d_sigma": got["sigma"] - base["sigma"], "d_x": got["x"] - base["x"]})
    rec["seconds"] = time.time() - t0
    print(f"BC-a0 {unit['kind']} {unit['key_id'][:8]} t={t.tolist()[:1]} j={unit['j']}: "
          f"{'ok' if rec['ok'] else 'failed'} in {rec['seconds']:.0f} s", flush=True)
    return rec


def bc_cache_name(code_hash: str | None, script_sha: str, name: str, unit: dict) -> str:
    """A unit's cache file name (amendment 14 item 1): the sha256 of the code hash, the
    script's sha256, the unit's name and its definition (kind, key, continuation, plan steps,
    s_next)."""
    import numpy as np

    t = unit.get("t")
    d = {"kind": unit["kind"], "key_id": unit.get("key_id"), "j": unit.get("j"),
         "t": None if t is None else np.asarray(t, int).tolist(),
         "s_next": None if unit.get("s_next") is None else float(unit["s_next"])}
    blob = json.dumps({"code": code_hash, "script": script_sha, "name": name, "unit": d},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def bc_unit_cached(state, unit) -> dict:
    """``state['fn']`` (``bc_unit``) on the unit, its record then written to the unit's cache
    file (``unit['cache']``) by an atomic rename (amendment 14 item 1)."""
    import pickle

    path = Path(unit["cache"])
    rec = globals()[state["fn"]](state, {k: v for k, v in unit.items() if k != "cache"})
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rec, fh, protocol=5)
    os.replace(tmp, path)
    return rec


def bc_pool_cached(run: Run, fn: Callable, units: Sequence[dict], names: Sequence[str],
                   state: dict, cache_dir: Path) -> tuple[list, int]:
    """(each unit's record in order, how many were read from the cache): the units with a
    cache file named for this code and unit are read; the others go to the pool, each record
    cached as it finishes (amendment 14 item 1)."""
    import pickle

    cache_dir = refuse_sealed(Path(cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)
    script = sha256_file(run.root / SCRIPT_REL)
    paths = [cache_dir / f"{bc_cache_name(run.code_hash, script, n, u)}.pkl"
             for n, u in zip(names, units, strict=True)]
    got: list = [None] * len(units)
    todo = []
    for i, path in enumerate(paths):
        if path.exists():
            with open(path, "rb") as fh:
                got[i] = pickle.load(fh)
        else:
            todo.append(i)
    if todo:
        res = run.pool(bc_unit_cached, [{**units[i], "cache": str(paths[i])} for i in todo],
                       state={**state, "fn": fn.__name__}, names=[names[i] for i in todo])
        for i, r in zip(todo, res, strict=True):
            got[i] = r
    return got, len(units) - len(todo)


def a2_refs_checked(run: Run) -> dict:
    """{key id: A2 ref}: the index's list against the record's 'a2' key set, its files against
    the record's content hash (recorded as an input)."""
    from apexzero.costate import keys_root
    from apexzero.costate import states as ST
    from apexzero.fields import prereg as FP

    refs = ST.a2_refs()
    want = [tuple(t) for t in run.rec["key_sets"]["a2"]]
    if sorted((r.kind, r.key_id) for r in refs) != want:
        raise SystemExit("BC-a0: the index's A2 keys are not the record's key set")
    h = FP.content_hash(want, keys_root())
    if h != run.rec["content"]["a2"]:
        raise SystemExit("BC-a0: the A2 key files differ from the tagged ones")
    run.note_input("keys/a2", h)
    return {r.key_id: r for r in refs}


def bc_state_picks(tab, keys: list[tuple[int, str, object]], item: int = BC_ITEM) -> list[dict]:
    """The per-state placement's states (amendment 11 item 3). On K2b's key i (``keys``: (i,
    key id, its T rows)), the interior braking- and corner-phase rows: with both phases, one of
    each; with one phase, two of it (one if it has one); drawn with
    default_rng([N_SEED, item, BC_PICK_UNIT + i]) (braking first)."""
    import numpy as np

    ph = np.asarray(tab["phase"]).astype(str)
    inner = np.asarray(tab["interior"], bool)
    out = []
    for i, kid, rows in keys:
        rows = np.asarray(rows, int)
        cand = {p: rows[inner[rows] & (ph[rows] == p)] for p in BC_STATE_PHASES}
        have = [p for p in BC_STATE_PHASES if len(cand[p])]
        rng = np.random.default_rng([N_SEED, int(item), BC_PICK_UNIT + int(i)])
        if len(have) == 2:
            pick = [int(cand[p][rng.integers(len(cand[p]))]) for p in BC_STATE_PHASES]
        elif len(have) == 1:
            c = cand[have[0]]
            pick = [int(r) for r in c[np.sort(rng.choice(len(c), min(2, len(c)),
                                                         replace=False))]]
        else:
            pick = []
        out += [{"key": int(i), "key_id": str(kid), "row": r, "t": int(tab["t"][r]),
                 "phase": str(ph[r])} for r in pick]
    return out


def bc_t_sd(tab, cols=BC_COLS) -> dict:
    """{col: SD_T}: K2b's scale (``gates.lam_sd``): the key-weighted SD over all of T's rows,
    within direction for a mirror-odd target."""
    import numpy as np

    from apexzero.costate import states as ST

    w = ST.weights(tab)
    d = np.asarray(tab["direction"])
    return {c: float(np.sqrt(ST.wvar(np.asarray(tab[c], float), w, d if bc_odd(c) else None)))
            for c in cols}


def bc_t_var(tab, cols=BC_COLS) -> dict:
    """{phase: {col: var_{T,p}}}: the key-weighted variance over T's interior rows of the
    phase (weights renormalised on them), within direction for a mirror-odd target."""
    import numpy as np

    from apexzero.costate import states as ST

    ph = np.asarray(tab["phase"]).astype(str)
    inner = np.asarray(tab["interior"], bool)
    d = np.asarray(tab["direction"])
    out = {}
    for p in ST.PHASES:
        rows = inner & (ph == p)
        w = ST.stratum_weights(tab, rows)[rows]
        out[p] = {c: float(ST.wvar(np.asarray(tab[c], float)[rows], w,
                                   d[rows] if bc_odd(c) else None)) if rows.any() else None
                  for c in cols}
    return out


def bc_key_means(v, key, mask, n_keys: int):
    """[n_keys] each key's mean of v over its rows in mask (nan for a key without one)."""
    import numpy as np

    v, key, mask = np.asarray(v, float), np.asarray(key, int), np.asarray(mask, bool)
    s = np.bincount(key[mask], weights=v[mask], minlength=n_keys)
    c = np.bincount(key[mask], minlength=n_keys)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def bc_share(D, key, mask, var: float | None, counts) -> dict:
    """The unseen-road share of one target on the rows in ``mask`` (amendment 11 item 4).
    D [J, n]: Delta of each continuation at each row (nan where it failed). Per row m2 =
    mean_j Delta^2 over its successful continuations, split as (mean_j Delta)^2 (bias) plus the
    continuations' spread (var). Each key weighs equally, its rows equally: U = the mean over
    keys of their mean m2, over var_{T,p}. CI: the keys resampled with ``counts`` [B, K]
    (a draw with no key that has a row is left out and counted)."""
    import numpy as np

    from apexzero.costate import probes as PB

    D = np.asarray(D, float)
    fin = np.isfinite(D)
    mask = np.asarray(mask, bool) & fin.any(0)
    n_keys = np.asarray(counts).shape[1]
    n_ok = np.maximum(fin.sum(0), 1)
    mean_d = np.where(fin, D, 0.0).sum(0) / n_ok
    m2 = (np.where(fin, D, 0.0) ** 2).sum(0) / n_ok
    b2 = mean_d ** 2
    out = {"rows": int(mask.sum()), "keys": 0, "var_T": var, "U": None, "U_bias": None,
           "U_spread": None, "ci": [None, None], "n_draws": 0}
    if not mask.any() or var is None or not var > 0:
        return out
    mk = bc_key_means(m2, key, mask, n_keys)
    bk = bc_key_means(b2, key, mask, n_keys)
    have = np.isfinite(mk)
    c = np.asarray(counts, float)[:, have]
    tot = c.sum(1)
    ok = tot > 0
    draws = (c[ok] @ mk[have]) / tot[ok] / var
    lo, hi = PB.ci(draws) if len(draws) else (np.nan, np.nan)
    U = float(np.mean(mk[have]) / var)
    Ub = float(np.mean(bk[have]) / var)
    out.update({"keys": int(have.sum()), "U": U, "U_bias": Ub, "U_spread": U - Ub,
                "ci": [float(lo), float(hi)], "n_draws": int(ok.sum())})
    return out


def bc_p90(D, mask, sd: float) -> dict:
    """K2b's statistic beside U: |Delta| / SD_T pooled over the rows in ``mask`` and the
    successful continuations: its p90, median and largest, and the worst continuation's p90."""
    import numpy as np

    D = np.abs(np.asarray(D, float)[:, np.asarray(mask, bool)]) / float(sd)
    v = D[np.isfinite(D)]
    if not len(v):
        return {"n": 0, "p90": None, "median": None, "max": None, "worst_j_p90": None}
    per_j = [float(np.quantile(r[np.isfinite(r)], 0.9)) for r in D if np.isfinite(r).any()]
    return {"n": len(v), "p90": float(np.quantile(v, 0.9)), "median": float(np.median(v)),
            "max": float(v.max()), "worst_j_p90": max(per_j)}


def bc_counts(n_keys: int, placement: str, n_boot: int | None = None, item: int = BC_ITEM):
    """[B, K] the key bootstrap's multiplicities (R10's stream, unit ``BC_BOOT_UNIT``)."""
    import numpy as np

    from apexzero.costate import probes as PB

    rng = np.random.default_rng([N_SEED, int(item), BC_BOOT_UNIT[placement]])
    B = PB.N_BOOT if n_boot is None else int(n_boot)
    return rng.multinomial(int(n_keys), np.full(int(n_keys), 1.0 / n_keys), size=B)


def bc_stats(D, DX, key, phase, interior, kept, sd: dict, var: dict, counts,
             phases=None) -> dict:
    """Per phase and target: U (on the interior rows of kept units) and K2b's p90 (on those
    rows, and on every row of the kept units), and the state's own move |Delta x| at the
    interior rows (median, largest). D [J, n, 14] in ``BC_COLS``' order; DX [J, n, 3]."""
    import numpy as np

    from apexzero.costate import states as ST

    phase = np.asarray(phase).astype(str)
    interior = np.asarray(interior, bool)
    kept = np.asarray(kept, bool)
    out = {}
    for p in (ST.PHASES if phases is None else phases):
        inner = kept & interior & (phase == p)
        every = kept & (phase == p)
        rec = {"rows_interior": int(inner.sum()), "rows_all": int(every.sum()), "targets": {}}
        for ci, c in enumerate(BC_COLS):
            rec["targets"][c] = {
                "share": bc_share(D[:, :, ci], key, inner, (var.get(p) or {}).get(c), counts),
                "p90_interior": bc_p90(D[:, :, ci], inner, sd[c]),
                "p90_all": bc_p90(D[:, :, ci], every, sd[c])}
        dx = np.abs(np.asarray(DX, float)[:, inner])
        rec["dx"] = {name: ({"median": float(np.nanmedian(dx[..., i])),
                             "max": float(np.nanmax(dx[..., i]))}
                            if np.isfinite(dx[..., i]).any() else None)
                     for i, name in enumerate(BC_DX)}
        out[p] = rec
    return out


def bc_route(per_state: dict, per_key: dict, bound: float = BC_ROUTE_BOUND) -> dict:
    """The route each K_P component and phase points to (plan BC-a0 'What the outcomes mean';
    a recommendation under decision #6): read from the per-state share's CI upper end, at most
    ``bound``: option (i), with that upper end as the stated bound; above it: option (ii); no
    share: none. The per-key share is beside it as the lower bound."""
    out = {}
    for p in BC_ROUTE_PHASES:
        for c in BC_K_P:
            col = f"lam_{c}"
            st = per_state.get(p, {}).get("targets", {}).get(col, {}).get("share", {})
            ky = per_key.get(p, {}).get("targets", {}).get(col, {}).get("share", {})
            hi = (st.get("ci") or [None, None])[1]
            if hi is None or st.get("U") is None:
                opt = None
            else:
                opt = "(i)" if hi <= bound else "(ii)"
            out[f"{p}/{c}"] = {"option": opt, "U_state": st.get("U"), "ci_state": st.get("ci"),
                               "stated_bound": hi if opt == "(i)" else None,
                               "U_key": ky.get("U"), "ci_key": ky.get("ci")}
    return out


def bc_expected(per_state: dict, per_key: dict) -> dict:
    """Ea0.1-Ea0.3 (amendment 11 item 5), point estimates, read on the per-state shares (the
    estimand) and reported on the per-key shares too."""
    def U(s, p, c):
        return ((s.get(p) or {}).get("targets", {}).get(f"lam_{c}", {}).get("share") or {}).get("U")

    def gt(a, b):
        return a is not None and b is not None and a > b

    out = {}
    for name, s in (("state", per_state), ("key", per_key)):
        e1 = {c: U(s, "corner", c) for c in ("s", "vx")}
        e2 = {c: {"braking": U(s, "braking", c), "corner": U(s, "corner", c)} for c in BC_K_P}
        out[name] = {
            "Ea0.1": {"corner_U": e1, "holds": all(v is not None and v > BC_ROUTE_BOUND
                                                    for v in e1.values())},
            "Ea0.2": {"U": e2, "count": sum(gt(v["corner"], v["braking"]) for v in e2.values()),
                      "of": len(BC_K_P),
                      "holds": all(gt(v["corner"], v["braking"]) for v in e2.values())}}
    cmp = {f"{p}/{c}": {"state": U(per_state, p, c), "key": U(per_key, p, c)}
           for p in BC_ROUTE_PHASES for c in BC_K_P}
    n = sum(gt(v["state"], v["key"]) for v in cmp.values())
    out["Ea0.3"] = {"U": cmp, "count": n, "of": len(cmp), "holds": n == len(cmp)}
    out["reading"] = "amendment 11 item 5"
    return out


def bc_units(tab, stored_keys: list[dict], rows_of: dict, a2: dict, strips: dict,
             picks: list[dict]) -> tuple[list[dict], list[str], list[dict]]:
    """(units, names, per-key facts) of both placements (amendment 11 items 1 and 3). Per key:
    its T rows, s_c1, direction and W from the bound world; s_next = max(s2, s_c1 + g_j); the
    key's T rows must all lie 20 m or more before s2's preview point (K2b's construction)."""
    import numpy as np

    from apexzero.costate import gates as GT

    units, names, facts = [], [], []
    for i, krec in enumerate(stored_keys):
        kid = str(krec["key_id"])
        rows = np.asarray(rows_of[kid], int)
        r0 = int(rows[0])
        s_c1, sigma = float(tab["s_c1"][r0]), int(tab["direction"][r0])
        W, s2 = int(tab["W"][r0]), float(krec["s2"])
        reach = float(np.max(bc_reach(np.asarray(tab["s"], float)[rows],
                                      np.asarray(tab["vx"], float)[rows])))
        if reach > s2 - GT.K2B_MARGIN + 1e-6:
            raise SystemExit(f"BC-a0 {kid}: a T row's preview reaches {reach:.1f} m, beyond "
                             f"s2 - {GT.K2B_MARGIN:g} m")
        want = bc_world_targets(tab, rows)
        facts.append({"key": i, "key_id": kid, "W": W, "s2": s2, "s_c1": s_c1,
                      "direction": sigma, "rows": len(rows), "reach_T": reach,
                      "R": float(tab["R"][r0]), "theta": float(tab["theta"][r0])})
        base = {"key": i, "key_id": kid, "ref": a2[kid], "s_c1": s_c1, "sigma": sigma, "W": W}
        for j in range(1, BC_J + 1):
            units.append({**base, "kind": "key", "j": j, "rows": rows,
                          "t": np.asarray(tab["t"], int)[rows],
                          "s_next": max(s2, s_c1 + strips[j]["gap0"]), "want": want})
            names.append(f"key/{kid}/j{j}")
    for pk in picks:
        kid, r = pk["key_id"], int(pk["row"])
        f = facts[pk["key"]]
        reach = float(bc_reach(tab["s"][r], tab["vx"][r])[0])
        base = {"key": pk["key"], "key_id": kid, "ref": a2[kid], "s_c1": f["s_c1"],
                "sigma": f["direction"], "W": f["W"]}
        for j in range(1, BC_J_STATE + 1):
            units.append({**base, "kind": "state", "j": j, "rows": np.array([r]),
                          "t": np.array([pk["t"]]), "s_next": reach + 20.0,
                          "want": bc_world_targets(tab, [r]), "reach": reach})
            names.append(f"state/{kid}/t{pk['t']}/j{j}")
    order = sorted(range(len(units)), key=lambda u: -units[u]["W"])     # longest first
    return [units[u] for u in order], [names[u] for u in order], facts


def bc_assemble(recs: list[dict], kind: str, n_rows_of: dict, J: int) -> tuple:
    """(D [J, n, 14], DX [J, n, 3], ok [unit, J]) of one placement from its unit records, with
    each unit's rows laid out in ``n_rows_of``'s order ({unit id: (start, n)})."""
    import numpy as np

    n = sum(v[1] for v in n_rows_of.values())
    D = np.full((J, n, len(BC_COLS)), np.nan)
    DX = np.full((J, n, len(BC_DX)), np.nan)
    ok = {u: np.zeros(J, bool) for u in n_rows_of}
    for r in recs:
        if r["kind"] != kind:
            continue
        uid = r["key"] if kind == "key" else (r["key"], r["t"][0])
        start, m = n_rows_of[uid]
        j = int(r["j"]) - 1
        if r["ok"]:
            D[j, start:start + m] = np.concatenate([r["d_lam"], r["d_Pi"], r["d_sigma"]], 1)
            DX[j, start:start + m] = r["d_x"]
            ok[uid][j] = True
    return D, DX, ok


@computes("BC-a0")
def bc_a0(run: Run) -> None:
    """BC-a0 (plan Section 4, amendment 11): the bound world, K2b's keys and the A2 key set
    checked; the grammar and the 8 strips (their draws checked); the per-state picks; the
    wiring key (K2b's first key, both severities, against gates/k2b.json); one pooled unit per
    key and continuation and per state and continuation; the shares, K2b's p90, the routes and
    the expected directions; outputs rows.npz and summary.json."""
    import jax
    import numpy as np

    from apexzero.costate import gates as GT
    from apexzero.costate import probes as PB
    from apexzero.costate import targets as TG

    t0 = time.time()
    CS._x64()
    if (BC_STATE_NAMES, BC_K_P, BC_MIRROR_ODD) != (TG.STATE_NAMES, TG.K_P, TG.MIRROR_ODD):
        raise SystemExit("BC-a0: its component lists are not targets.py's")
    world = load_world(run)
    tab = world.tab
    stored = json.loads(run.read("gates/k2b.json").read_text())
    a2 = a2_refs_checked(run)
    rows_of = {str(k.key_id): np.asarray(k.rows, int) for k in world.keys}
    lim = getattr(run.args, "limit", None)
    keys = stored["keys"] if lim is None else stored["keys"][:int(lim)]
    missing = [k["key_id"] for k in keys if k["key_id"] not in rows_of]
    if missing:
        raise SystemExit(f"BC-a0: K2b keys without T rows: {missing}")
    sd = bc_t_sd(tab)
    lam_sd = np.asarray(GT.lam_sd(tab, BC_STATE_NAMES), float)
    if not np.array_equal(np.array([sd[c] for c in BC_LAM]), lam_sd):
        raise SystemExit("BC-a0: SD_T is not gates.lam_sd's")
    var = bc_t_var(tab)
    g, files = bc_grammar(run, run.root)
    strips = {j: bc_strip(g, j) for j in range(1, BC_J + 1)}
    bad = [b for s in strips.values() for b in bc_strip_problems(s)]
    if bad:
        raise SystemExit(f"BC-a0: {'; '.join(bad)}")
    p_alt = (float(g.p_alt_short), float(g.p_alt_long))
    picks = bc_state_picks(tab, [(i, k["key_id"], rows_of[k["key_id"]])
                                 for i, k in enumerate(keys)])
    units, names, facts = bc_units(tab, keys, rows_of, a2, strips, picks)
    w0 = stored["keys"][0]
    units = [{"kind": "wiring", "ref": a2[w0["key_id"]], "t": None}] + units
    names = [f"wiring/{w0['key_id']}"] + names
    print(f"BC-a0: {len(keys)} keys, {len(picks)} states, {len(units)} units "
          f"({time.time() - t0:.0f} s)", flush=True)
    t1 = time.time()
    got, resumed = bc_pool_cached(run, bc_unit, units, names,
                                  {"sd": GT.lam_sd(tab), "strips": strips, "p_alt": p_alt},
                                  run.out / BC_CACHE)
    print(f"BC-a0: {resumed} of {len(units)} units read from the cache", flush=True)
    pool_s = time.time() - t1
    gap = k2b_gap(got[0], w0)
    if not gap <= K2B_TOL:
        raise SystemExit(f"BC-a0: gates.k2b_solve does not reproduce gates/k2b.json's first key "
                         f"({gap:.3g})")
    recs = got[1:]
    # the per-key placement: every T row of each key, in K2b's key order
    key_rows, start = {}, 0
    for f in facts:
        key_rows[f["key"]] = (start, f["rows"])
        start += f["rows"]
    all_rows = np.concatenate([rows_of[f["key_id"]] for f in facts])
    Dk, DXk, okk = bc_assemble(recs, "key", key_rows, BC_J)
    st_rows = {(pk["key"], pk["t"]): (i, 1) for i, pk in enumerate(picks)}
    Ds, DXs, oks = bc_assemble(recs, "state", st_rows, BC_J_STATE)
    fail_k = {f["key_id"]: int(BC_J - okk[f["key"]].sum()) for f in facts}
    fail_s = {f"{pk['key_id']}/t{pk['t']}": int(BC_J_STATE - oks[(pk["key"], pk["t"])].sum())
              for pk in picks}
    kept_key = np.concatenate([np.full(f["rows"], fail_k[f["key_id"]] <= BC_DROP["key"])
                               for f in facts])
    kept_state = np.array([fail_s[f"{pk['key_id']}/t{pk['t']}"] <= BC_DROP["state"]
                           for pk in picks], bool)
    key_k = np.concatenate([np.full(f["rows"], f["key"]) for f in facts])
    key_s = np.array([pk["key"] for pk in picks], int)
    ph = np.asarray(tab["phase"]).astype(str)
    inner = np.asarray(tab["interior"], bool)
    st_idx = np.array([pk["row"] for pk in picks], int)
    per_key = bc_stats(Dk, DXk, key_k, ph[all_rows], inner[all_rows], kept_key, sd, var,
                       bc_counts(len(facts), "key"))
    per_state = bc_stats(Ds, DXs, key_s, ph[st_idx], inner[st_idx], kept_state, sd, var,
                         bc_counts(len(facts), "state"), phases=BC_STATE_PHASES)
    frozen_p90 = (stored.get("rule") or {}).get("p90")
    unit_recs = [{k: v for k, v in r.items() if not k.startswith("d_")} for r in recs]
    out = {
        "item": "BC-a0", "targets_only": True,
        "keys": facts, "states": picks,
        "strips": [{k: v for k, v in s.items() if k != "kappa"} for s in strips.values()],
        "grammar": {"circuits": sorted(files), "p_alt_short": p_alt[0], "p_alt_long": p_alt[1],
                    "corners": int(g.corner_angle.shape[0]),
                    "straights": int(g.straight_len.shape[0]),
                    "x64": bool(jax.config.jax_enable_x64)},
        "units": unit_recs,
        "failed": {"key": fail_k, "state": fail_s,
                   "dropped_keys": [k for k, v in fail_k.items() if v > BC_DROP["key"]],
                   "dropped_states": [k for k, v in fail_s.items() if v > BC_DROP["state"]],
                   "errors": sorted({str(r.get("error")) for r in recs
                                     if not r["ok"] and r.get("error")})},
        "sd_T": sd, "var_T_interior": var,
        "per_key": per_key, "per_state": per_state,
        "route": bc_route(per_state, per_key),
        "expected": bc_expected(per_state, per_key),
        "k2b_frozen_p90": frozen_p90,
        "wiring": {"k2b_key": w0["key_id"], "k2b_gap": gap, "k2b_tol": K2B_TOL,
                   "k2b_seconds": got[0]["seconds"],
                   "target_gap_max": max(float(r["target_gap"]) for r in recs),
                   "target_tol": BC_TARGET_TOL, "sd_equals_lam_sd": True},
        "definitions": "plan Section 4 BC-a0 and amendment 11",
        "n_boot": PB.N_BOOT, "partial": lim is not None,
        "unit_seconds": {"median": float(np.median([r["seconds"] for r in recs])),
                         "max": float(np.max([r["seconds"] for r in recs])),
                         "sum": float(np.sum([r["seconds"] for r in recs]))},
        "pool_seconds": pool_s, "resumed_from_cache": resumed,
        "seconds": time.time() - t0}
    arrays = {"key/D": Dk, "key/DX": DXk, "key/rows": all_rows, "key/key": key_k,
              "key/kept": kept_key, "state/D": Ds, "state/DX": DXs, "state/rows": st_idx,
              "state/key": key_s, "state/kept": kept_state,
              "strips/kappa": np.stack([strips[j]["kappa"] for j in range(1, BC_J + 1)]),
              "cols": np.array(BC_COLS), "dx_cols": np.array(BC_DX)}
    run.write_npz("rows", arrays)
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
        if it.id == "CS-9":
            p.add_argument("--stage", choices=(*N_STAGES, "all"), default="all",
                           help="CS-9's stage (amendment 9 item 1; default all, in order)")
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
