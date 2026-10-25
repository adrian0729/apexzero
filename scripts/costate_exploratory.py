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
"""

from __future__ import annotations

import dataclasses
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
         ("world", "critic:decision", "activations:decision", "gates:k4"), trained=True),
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
