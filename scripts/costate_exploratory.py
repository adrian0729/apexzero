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

**Implemented items.** CS-1 (``cs1``, the section 'CS-1: the xi tail'; plan Section 4 with
amendment 1): the tail reproduced from the findings, where it sits, g_xi split by a fresh VJP
into channel groups, and R2_c under the Huber and rank calibrations after their plants.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import io
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
POOLED_ITEMS = ("cs-4", "cs-7", "cs-8", "cs-9", "cs-10", "cs-11", "cs-12", "bc-a0")
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
    Item("CS-2", "Descriptives pack", ("world", "critic:all", "activations:decision")),
    Item("CS-3", "C3-int on s and xi (network x geometry)",
         ("world", "critic:all", "step1", "step2"), needs=("CS-1",)),
    Item("CS-4", "Delta C1, R2_z and the MLP-on-z ceiling",
         ("world", "critic:decision", "activations:decision"), needs=("CS-1",)),
    Item("CS-5", "W1 on the Pi targets", ("world", "critic:decision", "activations:decision"),
         needs=("CS-1",)),
    Item("CS-6", "C1-dev over the 16 snapshots", ("world", "snapshots"), trained=True,
         needs=("CS-1",)),
    Item("CS-7", "Critic-layer and head-unit probes", ("world", "activations:decision"),
         needs=("CS-1",)),
    Item("CS-8", "C2-sw's random-only null; C2-sw residualised; C2 with a decoy that fits "
         "geometry 6", ("world", "activations:decision", "probes", "gates:c2_null")),
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
