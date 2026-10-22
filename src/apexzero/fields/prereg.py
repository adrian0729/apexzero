"""Step 2's pre-registration record (design Sections 9 and 15): what the tag fixes besides the
code.

The record is ``src/apexzero/fields/prereg.json``. It lies inside the frozen paths, so git
freezes it with the code at the tag receptive-fields-prereg-v1, and scripts/fields.py's freeze
check reads it from the tagged commit (``git show <tag>:<path>``), not from the working tree.
It is written last before the tag, from a clean tree (``build_record`` by the 'prereg' stage),
and holds:

- the key lists (``key_sets``): the A2 set (all 481 keys, the 6 horizon twins listed apart),
  the anchored finals per agent and the earlier snapshots per (agent, snapshot), as sorted key
  ids; a content hash of each set (``content_hash``: sha256 over each listed key's .json and
  .npz bytes, in id order); each agent's final timesteps;
- the re-polish movement per key (compact rows: max_dx, dJ_rel, flagged, ok) and their hash;
- step 1's stimulus design hash (``stimuli.design_hash``), the design document's sha256, the
  commit the record was built on (``prereg_base_commit``) and ``code_hash`` over src/ (all of
  it, so that nothing importable from src/ lies outside the hash), scripts/fields.py and
  scripts/_bootstrap.py (every .py file: the imported ocp, vehicle, tracks, env and agents
  code too; the record itself is JSON, so it is never part of the hash). ``tree_code_hash``
  is the same rule on a commit's blobs (git), so the freeze check can compare the record with
  the code the tag holds;
- every network a post-tag stage loads (``networks``): its checkpoint's repo-relative path,
  sha256 and timesteps (post-tag loaders load the recorded path and check its hash), and each
  registry agent's stored final rollouts (``rollouts``: the det group's hash, ``det_hash``);
- the per-regime A2 cell sets of every agent and H4's typing per variant, with the sha256 of
  step 1's support.json (``regimes``);
- the gate2-trained and dose key lists;
- the stamped, passing pre-tag outputs (``bind_outputs``): keys/Gate 0 (and the loop audit),
  re-polish, Gates 1-5, the controls, the calibration, the Qp reader fields and opt-fields,
  each with its sha256 and its stamp's code hash; for opt-fields the sha256 of rows.jsonl, a
  content hash of arrays/ (sorted names and bytes), its exact key coverage and the exclusion
  lists (design Sections 5 and 8: a failed re-polish, an inertia or adjoint failure and a G*
  outside Gate 4's band are listed exclusions, not failures of the stage), checked against
  Gate 4's 'outside' list and re-polish's 'not_ok' list. 'Passing' means complete: every key
  processed, no crash (``_passes``); Gate 2 must hold exactly its fixed rows
  (``gates.gate2_labels``) and Gate 3 every registry agent with its every-key check;
- the calibrated branch-consistency tolerance with its file's sha256 (``calibration``): what
  every post-tag stage uses (``network.BRANCH_TOL`` is only the pre-tag default);
- the tolerances and bands the gates and measures use (the code constants, copied for the
  reader; the code hash is what binds them).

``verify_record`` recomputes each item from disk and lists every mismatch as a ``Problem``
(a str with a ``code``): a missing or edited key file, a set that no longer matches the index,
a re-polish run whose movement differs from the recorded one by more than
``REPOLISH_RECHECK_TOL``, a changed design, design document, bound output, calibration file,
support file or checkpoint, or code changed after the record was written. Only the codes in
``WAIVABLE`` ('code_changed', 'design_doc_changed') can be waived by a logged deviation.
Post-tag stages iterate ``frozen_ids``, never a fresh glob.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Self

import numpy as np

PREREG_NAME = "prereg.json"
PREREG_PATH = Path(__file__).with_name(PREREG_NAME)
PREREG_REL = f"src/apexzero/fields/{PREREG_NAME}"
DESIGN_DOC = "docs/interp/receptive-fields-design.md"
CODE_ROOTS = ("src", "scripts/fields.py", "scripts/_bootstrap.py")
REPOLISH_RECHECK_TOL = 1e-6      # m: a re-polish elsewhere may move a key this much more
VERSION = 2
WAIVABLE = ("code_changed", "design_doc_changed")   # what --post-tag-deviation may waive
STAMP_SUFFIX = ".provenance.json"                   # sidecar stamp of a non-dict output
# The pre-tag outputs the record binds (design Section 15), relative to the output root, and
# what 'passing' means for each (``_passes``).
BOUND_OUTPUTS = ("keys/gate0.json", "keys/loop_audit.json", "repolish/summary.json",
                 "gates/gate1.json", "gates/gate2.json", "gates/gate3.json", "gates/gate4.json",
                 "gates/gate5.json", "controls/controls.json", "calibration/branch_tol.json",
                 "opt-fields/summary.json", "qp-fields/summary.json")
CALIBRATION = "calibration/branch_tol.json"


class Problem(str):
    """A freeze problem: its message, with a category ``code`` (``WAIVABLE`` or not)."""

    code: str

    def __new__(cls, code: str, message: str) -> Self:
        obj = str.__new__(cls, message)
        obj.code = str(code)
        return obj


def problem_code(p) -> str:
    """A problem's code ('' for a plain string: never waivable)."""
    return getattr(p, "code", "")


def waivable(problems) -> bool:
    """Whether a deviation may waive every one of ``problems``."""
    return all(problem_code(p) in WAIVABLE for p in problems)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def content_hash(ids: list[tuple[str, str]], keys_root: Path) -> str:
    """sha256 over (kind, id, the .json bytes' hash, the .npz bytes' hash) of each listed key
    in order; a missing file hashes as 'missing'."""
    h = hashlib.sha256()
    root = Path(keys_root)
    for kind, kid in ids:
        h.update(f"{kind}/{kid}".encode())
        for ext in (".json", ".npz"):
            p = root / kind / f"{kid}{ext}"
            h.update((sha256_file(p) if p.exists() else "missing").encode())
    return h.hexdigest()


def dir_hash(path: Path, pattern: str = "*.npz") -> tuple[str, int]:
    """(sha256 over the sorted (name, sha256 of bytes) of the files matching ``pattern``,
    their number); ('missing', 0) for no directory."""
    path = Path(path)
    if not path.is_dir():
        return "missing", 0
    files = sorted(path.glob(pattern))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest(), len(files)


def det_hash(path: Path) -> str:
    """sha256 over a rollout file's det group (each det_* member's name, dtype, shape and
    bytes, in name order): the stored runs Gate 3, branch consistency and the anchored
    restarts read, independent of how the file was re-saved."""
    h = hashlib.sha256()
    with np.load(Path(path), allow_pickle=False) as z:
        for k in sorted(n for n in z.files if n.startswith("det_")):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}|{a.dtype}|{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


def key_sets(keys_root: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """The sets the tag fixes, as sorted (kind, key_id) lists: 'a2', 'a2_twins',
    'finals:<agent>', 'earlier:<agent>@<snapshot>' (``keys``' index: JSON only)."""
    from apexzero.fields import keys as K

    out = {"a2": sorted((r.kind, r.key_id) for r in K.a2_refs(keys_root)),
           "a2_twins": sorted((r.kind, r.key_id) for r in K.a2_twins(keys_root))}
    for a, refs in K.anchored_finals(keys_root).items():
        out[f"finals:{a}"] = sorted((r.kind, r.key_id) for r in refs)
    for (a, snap), refs in K.anchored_earlier(keys_root).items():
        out[f"earlier:{a}@{snap}"] = sorted((r.kind, r.key_id) for r in refs)
    return out


def repolish_rows(rows: list[dict]) -> dict[str, list]:
    """{key_id: [max_dx, dJ_rel, flagged, ok]} from the repolish stage's rows."""
    return {str(r["key_id"]): [float(r.get("max_dx", float("nan"))),
                               float(r.get("dJ_rel", float("nan"))),
                               bool(r.get("flagged")), bool(r.get("ok"))]
            for r in sorted(rows, key=lambda r: str(r["key_id"]))}


def _canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def tolerances(calibrated: dict | None = None) -> dict:
    """The gate and measure constants (a copy for the reader; the code hash binds them), with
    the calibrated branch tolerance (``calibrated``: branch_tol.json's 'tolerance') beside the
    pre-tag provisional default."""
    from apexzero.fields import gates as G
    from apexzero.fields import kkt as KK
    from apexzero.fields import measures as MS
    from apexzero.fields import network as N

    return {"gate0": G.GATE0_TOL, "gate1": [KK.GATE1_MEDIAN, KK.GATE1_P95, KK.ZERO_ATOL],
            "gate1_fd_h": KK.FD_H, "gate1_resolve_tol": KK.GATE1_TOL,
            "gate2": [N.GATE2_TOL, N.ZERO_ATOL, N.RICH_RTOL, N.KINK_FRACTION],
            "gate2_h_max": N.H_MAX, "gate3_mean": N.MEAN_TOL,
            "gate3_step": [float(v) for v in N.STEP_TOL[1:]], "gate3_accel": N.ACCEL_TOL,
            "branch": dict(calibrated) if calibrated else None,
            "branch_provisional_pre_tag": N.BRANCH_TOL, "gate4": list(KK.GATE4_BAND),
            "gate5": [G.GATE5_TOL, G.GATE5_STATE_TOL, G.GATE5_BOUND_TOL, G.GATE5_TIE],
            "adjoint_rtol": KK.ADJOINT_RTOL, "loop_steps": N.STEPS,
            "horizon_margin": N.HORIZON_MARGIN, "edge_plateau": MS.EDGE_PLATEAU,
            "far_geometry": [MS.FAR_LOG_R, MS.FAR_THETA],
            "descriptive_reference": {"h1_gain": list(MS.H1_GAIN),
                                      "h1_location": MS.H1_LOCATION,
                                      "t_blind_hi": MS.T_BLIND_HI,
                                      "t_anchored": list(MS.T_ANCHORED)},
            "h2": [MS.H2_DEFICIT_HI, *MS.H2_LAWFUL, MS.H2_OVER_LO],
            "coverage_max": MS.COVERAGE_MAX, "gate2_trained_n": 12, "harness_chunk": 64,
            "hybrid_gain": N.HYBRID_GAIN}


def build_record(repo: Path, sets: dict[str, list[tuple[str, str]]], keys_root: Path,
                 repolish: list[dict], design_hash: str = "", final_timesteps: dict | None = None,
                 tols: dict | None = None, extra: dict | None = None) -> dict:
    """The record (module docstring) from the computed sets, the re-polish rows and the
    design hash; ``repo`` is the repository root (design document and code hash); ``extra``
    holds the items the prereg stage computes (networks, regimes, key lists, bound outputs,
    calibration, base commit). A missing design document is an error."""
    repo = Path(repo)
    if not (repo / DESIGN_DOC).exists():
        raise FileNotFoundError(f"{DESIGN_DOC} is missing: the record registers it")
    rp = repolish_rows(repolish)
    missing = sorted({k for v in sets.values() for _, k in v} - set(rp))
    rec = {"version": VERSION,
           "key_sets": {k: [list(t) for t in v] for k, v in sets.items()},
           "key_counts": {k: len(v) for k, v in sets.items()},
           "content": {k: content_hash(v, keys_root) for k, v in sets.items()},
           "final_timesteps": dict(final_timesteps or {}),
           "repolish": rp, "repolish_hash": _canon(rp), "repolish_missing": missing,
           "design_hash": design_hash,
           "design_doc_sha256": sha256_file(repo / DESIGN_DOC),
           "code_hash": code_hash(repo), "tolerances": dict(tols or {})}
    rec.update(extra or {})
    return rec


def _hash_items(items) -> str:
    """sha256 over (repo-relative path, bytes) pairs in the given order (``code_hash``)."""
    h = hashlib.sha256()
    for rel, data in items:
        h.update(str(rel).encode())
        h.update(data)
    return h.hexdigest()


def code_files(repo: Path, roots: tuple[str, ...] = CODE_ROOTS) -> list[str]:
    """``code_hash``'s files on disk, in its order, as repo-relative POSIX paths: per root (in
    ``roots`` order) the root itself when it is a file, else every .py file under it
    (``rglob``, sorted as paths); the record never (it is JSON)."""
    repo = Path(repo)
    out: list[str] = []
    for r in roots:
        p = repo / r
        files = sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])
        out += [f.relative_to(repo).as_posix() for f in files if f.name != PREREG_NAME]
    return out


def code_hash(repo: Path, roots: tuple[str, ...] = CODE_ROOTS) -> str:
    """``io.code_hash``'s rule (every .py file under roots: path and bytes) on a given
    repository root. The record (JSON) is never part of it."""
    repo = Path(repo)
    return _hash_items((rel, (repo / rel).read_bytes()) for rel in code_files(repo, roots))


def _git_out(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    r = subprocess.run(["git", *args], cwd=repo, input=stdin, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: "
                           f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def tree_blobs(repo: Path, rev: str, paths: tuple[str, ...]) -> dict[str, bytes]:
    """{repo-relative path: blob bytes} of every file under ``paths`` in commit ``rev``
    (``git ls-tree -r`` and ``git cat-file --batch``: raw bytes, never decoded). Raises
    RuntimeError when git fails (an unknown revision, no repository)."""
    repo = Path(repo)
    out = _git_out(repo, "ls-tree", "-r", "-z", rev, "--", *paths)
    entries = []
    for ent in out.split(b"\0"):
        if not ent:
            continue
        head, _, path = ent.partition(b"\t")
        _, typ, sha = head.split()
        if typ == b"blob":
            entries.append((path.decode(), sha.decode()))
    if not entries:
        return {}
    raw = _git_out(repo, "cat-file", "--batch",
                   stdin="".join(f"{sha}\n" for _, sha in entries).encode())
    blobs, i = {}, 0
    for path, sha in entries:
        nl = raw.index(b"\n", i)
        got, typ, size = raw[i:nl].split()
        if got.decode() != sha or typ != b"blob":
            raise RuntimeError(f"git cat-file: unexpected {raw[i:nl][:80]!r}")
        n = int(size)
        blobs[path] = raw[nl + 1: nl + 1 + n]
        i = nl + 1 + n + 1
    return blobs


def tree_files(blobs: dict[str, bytes], roots: tuple[str, ...] = CODE_ROOTS) -> list[str]:
    """``code_files``' selection and order over a commit's paths (``tree_blobs``)."""
    out: list[str] = []
    for r in roots:
        if r in blobs:
            sel = [r]
        else:
            sel = sorted((p for p in blobs if p.startswith(r.rstrip("/") + "/")
                          and p.endswith(".py")), key=PurePosixPath)
        out += [p for p in sel if PurePosixPath(p).name != PREREG_NAME]
    return out


def tree_code_hash(repo: Path, rev: str, roots: tuple[str, ...] = CODE_ROOTS) -> str:
    """``code_hash`` of the code a commit holds (its blobs; ``tree_files``' order is
    ``code_files``', so on a clean checkout the two agree). Raises RuntimeError when git
    fails."""
    blobs = tree_blobs(repo, rev, roots)
    return _hash_items((p, blobs[p]) for p in tree_files(blobs, roots))


# ------------------------------------------------------------------------- stamps and outputs


def stamp_path(path: Path) -> Path:
    """The sidecar stamp of an output that is not a JSON object (<file>.provenance.json)."""
    path = Path(path)
    return path.with_name(path.name + STAMP_SUFFIX)


def read_stamp(path: Path) -> dict | None:
    """The provenance stamp of an output: embedded in a JSON object ('provenance'), else its
    sidecar; None when there is none."""
    path = Path(path)
    if path.suffix == ".json" and path.exists():
        try:
            d = json.loads(path.read_text())
        except ValueError:
            d = None
        if isinstance(d, dict) and isinstance(d.get("provenance"), dict):
            return d["provenance"]
    sc = stamp_path(path)
    return json.loads(sc.read_text()) if sc.exists() else None


def _gate3_complete(d) -> bool:
    """Gate 3 on every registry agent (``agents.AGENTS``), each with its every-key check and at
    least the default planted sample checked (``gates.GATE3_N_ANCHORED`` + ``GATE3_N_A2``
    rows not skipped), and passing."""
    from apexzero.fields import gates as G
    from apexzero.interp import agents

    per = d.get("agents") or {}
    if set(per) != set(agents.AGENTS):
        return False
    for v in per.values():
        done = [r for r in v.get("rows", []) if "skipped" not in r]
        if (not v.get("pass") or v.get("every_key") is None
                or len(done) < G.GATE3_N_ANCHORED + G.GATE3_N_A2):
            return False
    return bool(d.get("pass"))


def _passes(rel: str, d) -> bool:
    """Whether a bound output reports a pass (module docstring)."""
    if rel == "keys/gate0.json":
        return bool(d["summary"]["pass"])
    if rel == "keys/loop_audit.json":
        return bool(d["ok"] and d.get("h3", {}).get("ok", False))
    if rel == "gates/gate1.json":
        return bool(d["decision"]["pass"])
    if rel == "gates/gate2.json":            # exactly the fixed rows, each passing (not the flag)
        from apexzero.fields import gates as G

        rows = d.get("rows") or []
        return bool(rows and {r["label"] for r in rows} == G.gate2_labels()
                    and all(r["pass"] is True for r in rows))
    if rel == "gates/gate3.json":
        return _gate3_complete(d)
    if rel == "gates/gate4.json":            # complete: its outside keys are listed exclusions
        return bool(d.get("pass")) and not d.get("crashed")
    if rel == "controls/controls.json":
        return bool(d["pass"]) and all(bool(v) for v in d["pass"].values())
    if rel == CALIBRATION:
        tol = d.get("tolerance", {})
        return bool(d.get("controls_branch_ok") and not d.get("agents_missing_from_gate3")
                    and tol and all(np.isfinite(float(v)) for v in tol.values()))
    if rel in ("opt-fields/summary.json", "qp-fields/summary.json"):
        return bool(d.get("n")) and not d.get("crashed")       # failed rows are exclusions
    if rel == "repolish/summary.json":
        return bool(d.get("pass")) and not d.get("crashed")
    return bool(d.get("pass"))


def _ids(sets: dict, names) -> set[tuple[str, str]]:
    return {tuple(t) for n in names for t in sets.get(n, [])}


def _rows_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def bind_rows_dir(out_root: Path, stage: str, want_ids: set[str]) -> tuple[dict, list[str]]:
    """The binding of a per-key rows stage (opt-fields, qp-fields): sha256 of rows.jsonl, the
    content hash of arrays/, and its coverage: the rows' key ids exactly ``want_ids`` (no
    duplicates), one arrays/<id>.npz per row that has arrays, no extra .npz files."""
    root = Path(out_root) / stage
    probs: list[str] = []
    rp = root / "rows.jsonl"
    if not rp.exists():
        return {}, [f"{stage}: no rows.jsonl"]
    rows = _rows_jsonl(rp)
    ids = [str(r["key_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        probs.append(f"{stage}: {len(ids) - len(set(ids))} duplicate key rows")
    if set(ids) != set(want_ids):
        probs.append(f"{stage}: covers {len(set(ids))} keys, not exactly the record's "
                     f"{len(want_ids)} ({len(set(want_ids) - set(ids))} missing, "
                     f"{len(set(ids) - set(want_ids))} extra)")
    arr = root / "arrays"
    if stage == "qp-fields":
        arr = arr / "reader-Qp"
    want_npz = {f"{r['key_id'].replace('/', '_')}.npz" for r in rows
                if not r.get("failed") and not r.get("pipeline")}
    have_npz = {p.name for p in arr.glob("*.npz")} if arr.is_dir() else set()
    if want_npz - have_npz:
        probs.append(f"{stage}: {len(want_npz - have_npz)} rows without arrays")
    if have_npz - want_npz:
        probs.append(f"{stage}: {len(have_npz - want_npz)} array files of no row")
    h, n = dir_hash(arr)
    return {"rows_sha256": sha256_file(rp), "n_rows": len(rows), "arrays_hash": h,
            "n_arrays": n}, probs


def opt_exclusions(rows: list[dict]) -> dict[str, list[str]]:
    """The exclusion lists the optimum's rows fix (design Sections 5, 8 and 15): on the A2
    keys a failed re-polish ('numerical'), an adjoint or inertia failure, G* outside Gate 4's
    band and the argmin off plateau; on the anchored keys the numerical, adjoint and inertia
    failures."""
    a2 = [r for r in rows if r.get("kind") != "anchored"]
    an = [r for r in rows if r.get("kind") == "anchored"]
    ok = [r for r in a2 if not r.get("failed")]
    an_ok = [r for r in an if not r.get("failed")]
    return {"numerical_failed": sorted(r["key_id"] for r in a2 if r.get("failed")),
            "adjoint_failed": sorted(r["key_id"] for r in ok if r.get("adjoint_ok") is False),
            "inertia_failed": sorted(r["key_id"] for r in ok if r.get("inertia_failed")),
            "outside_gate4": sorted(r["key_id"] for r in ok if not r.get("in_band")),
            "off_plateau": sorted(r["key_id"] for r in ok if r["valid"]["c_min"]
                                  and not r.get("on_plateau")),
            "anchored_numerical_failed": sorted(r["key_id"] for r in an if r.get("failed")),
            "anchored_adjoint_failed": sorted(r["key_id"] for r in an_ok
                                              if r.get("adjoint_ok") is False),
            "anchored_inertia_failed": sorted(r["key_id"] for r in an_ok
                                              if r.get("inertia_failed"))}


def bind_outputs(out_root: Path, code_hash_now: str, sets: dict, qp_ids: set[str] | None = None
                 ) -> tuple[dict, list[str]]:
    """({rel: {sha256, code_hash}}, the opt-fields and qp-fields bindings, the exclusion
    lists; problems) for the pre-tag outputs under ``out_root``: each ``BOUND_OUTPUTS`` file
    must exist, carry a stamp with ``code_hash_now`` and no partial options, and pass
    (``_passes``: complete, no crash); Gate 0 must cover every listed key and Gate 4 every A2
    key; opt-fields must cover exactly every listed key (qp-fields exactly ``qp_ids``, the
    primary A2 keys); Gate 4's 'outside' list must equal the opt-fields A2 rows that failed to
    re-polish, fall outside the band or fail inertia or the adjoint check, and re-polish's
    'not_ok' list the opt-fields rows (every kind) that failed to re-polish."""
    root = Path(out_root)
    out: dict = {"files": {}}
    probs: list[str] = []
    every = _ids(sets, sets)
    for rel in BOUND_OUTPUTS:
        p = root / rel
        if not p.exists():
            probs.append(f"{rel} is missing")
            continue
        st = read_stamp(p)
        if st is None:
            probs.append(f"{rel} has no provenance stamp")
        else:
            if st.get("code_hash") != code_hash_now:
                probs.append(f"{rel} was written by other code (stamp {str(st.get('code_hash'))[:12]})")
            if st.get("partial"):
                probs.append(f"{rel} is a partial run ({st['partial']})")
        d = json.loads(p.read_text())
        try:
            ok = _passes(rel, d)
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            probs.append(f"{rel} does not pass")
        out["files"][rel] = {"sha256": sha256_file(p),
                             "code_hash": None if st is None else st.get("code_hash")}
        if rel == "keys/gate0.json":
            got = {(r["kind"], r["key_id"]) for r in d.get("rows", [])}
            if got != every:
                probs.append(f"Gate 0 covers {len(got)} keys, not the record's {len(every)}")
        if rel == "gates/gate4.json":
            got = {r["key_id"] for r in d.get("rows", [])}
            if got != {k for _, k in sets.get("a2", [])}:
                probs.append("Gate 4 does not cover exactly the A2 set")
    for rel in ("repolish/repolish.jsonl", "phase-ceiling/phase_ceiling.json"):
        p = root / rel
        if p.exists():
            out["files"][rel] = {"sha256": sha256_file(p),
                                 "code_hash": (read_stamp(p) or {}).get("code_hash")}
    b, pr = bind_rows_dir(root, "opt-fields", {k for _, k in every})
    out["opt_fields"], probs = b, probs + pr
    if (root / "opt-fields" / "rows.jsonl").exists():
        rows = _rows_jsonl(root / "opt-fields" / "rows.jsonl")
        out["exclusions"] = opt_exclusions(rows)
        g4 = root / "gates" / "gate4.json"
        if g4.exists():
            outside = {r["key_id"] for r in json.loads(g4.read_text()).get("outside", [])}
            ex = out["exclusions"]
            mine = (set(ex["numerical_failed"]) | set(ex["adjoint_failed"])
                    | set(ex["inertia_failed"]) | set(ex["outside_gate4"]))
            if outside != mine:
                probs.append(f"Gate 4's outside list ({len(outside)}) is not opt-fields' A2 "
                             f"exclusions ({len(mine)})")
        rs = root / "repolish" / "summary.json"
        if rs.exists():
            not_ok = set(json.loads(rs.read_text()).get("not_ok", []))
            failed = {r["key_id"] for r in rows if r.get("failed")}
            if not_ok != failed:
                probs.append(f"re-polish's not_ok list ({len(not_ok)}) is not opt-fields' failed "
                             f"rows ({len(failed)})")
    if qp_ids is not None:
        b, pr = bind_rows_dir(root, "qp-fields", set(qp_ids))
        out["qp_fields"], probs = b, probs + pr
    return out, probs


def calibration_binding(out_root: Path, gate3_sha: str | None = None) -> tuple[dict, list[str]]:
    """The calibrated branch tolerance the record binds: the file's sha256, tolerance and
    trained error; problems if the file is missing, fails, or was derived from another
    gates/gate3.json (``gate3_sha``)."""
    p = Path(out_root) / CALIBRATION
    if not p.exists():
        return {}, [f"{CALIBRATION} is missing (run the calibrate stage)"]
    d = json.loads(p.read_text())
    probs = [] if _passes(CALIBRATION, d) else [f"{CALIBRATION} does not pass"]
    if gate3_sha is not None and d.get("gate3_sha256") != gate3_sha:
        probs.append(f"{CALIBRATION} was derived from another gates/gate3.json")
    return {"path": CALIBRATION, "sha256": sha256_file(p),
            "tolerance": {k: float(v) for k, v in d.get("tolerance", {}).items()},
            "trained_mean_err": d.get("trained_mean_err"), "factor": d.get("factor")}, probs


def record_tolerance(rec: dict) -> dict[str, float] | None:
    """The calibrated branch tolerance the record binds (None when absent: a post-tag stage
    then refuses)."""
    tol = (rec.get("calibration") or {}).get("tolerance")
    return {k: float(v) for k, v in tol.items()} if tol else None


# ------------------------------------------------------------------------- verification


def verify_record(rec: dict, repo: Path, keys_root: Path,
                  current_sets: Callable[[], dict] | None = None,
                  repolish: list[dict] | None = None, design_hash: str | None = None,
                  final_timesteps: dict | None = None, out_root: Path | None = None,
                  networks: Callable[[dict], list[str]] | None = None,
                  support_sha256: str | None = None,
                  rollouts: Callable[[dict], list[str]] | None = None) -> list[Problem]:
    """Every way the disk differs from the tagged record (module docstring); [] when it
    matches. ``current_sets()`` recomputes the sets from the key index; ``repolish`` are this
    machine's re-polish rows (None: not checked); ``design_hash`` the current step-1 design's;
    ``out_root`` the output root whose bound outputs, opt-fields and qp-fields rows and arrays
    and calibration file are re-hashed; ``networks(pins)`` lists checkpoint problems;
    ``support_sha256`` the current support.json's; ``rollouts(recorded)`` lists the stored
    rollouts whose det group changed."""
    out: list[Problem] = []
    repo = Path(repo)
    if int(rec.get("version", -1)) != VERSION:
        out.append(Problem("record", f"pre-registration record version {rec.get('version')} "
                           f"is not {VERSION}"))
    sets = {k: [tuple(t) for t in v] for k, v in rec.get("key_sets", {}).items()}
    for k, ids in sets.items():
        h = content_hash(ids, keys_root)
        if h != rec["content"].get(k):
            out.append(Problem("keys", f"key set {k}: the key files differ from the tagged ones"))
    if current_sets is not None:
        now = current_sets()
        for k in sorted(set(now) | set(sets)):
            if [tuple(t) for t in now.get(k, [])] != sets.get(k, []):
                out.append(Problem("keys", f"key set {k}: the index no longer gives the "
                                   "tagged list"))
    if final_timesteps is not None and dict(final_timesteps) != rec.get("final_timesteps", {}):
        out.append(Problem("checkpoints", "an agent's final snapshot differs from the tagged one"))
    if repolish is not None:
        now = repolish_rows(repolish)
        if _canon(now) != rec.get("repolish_hash"):
            recd = rec.get("repolish", {})
            bad = [k for k, v in recd.items()
                   if k not in now or bool(now[k][3]) != bool(v[3])
                   or not abs(float(now[k][0]) - float(v[0])) <= REPOLISH_RECHECK_TOL]
            if bad:
                out.append(Problem("repolish", f"re-polish differs from the recorded movement "
                                   f"on {len(bad)} keys (e.g. {bad[:3]})"))
    if design_hash is not None and design_hash != rec.get("design_hash"):
        out.append(Problem("design", "step 1's stimulus design differs from the tagged one"))
    doc = repo / DESIGN_DOC
    if not rec.get("design_doc_sha256"):
        out.append(Problem("record", "the record registers no design document"))
    elif not doc.exists() or sha256_file(doc) != rec["design_doc_sha256"]:
        out.append(Problem("design_doc_changed", f"{DESIGN_DOC} differs from the tagged design"))
    if code_hash(repo) != rec.get("code_hash"):
        out.append(Problem("code_changed", "code changed after the pre-registration record was "
                           "written"))
    if out_root is not None:
        out += _verify_outputs(rec, Path(out_root))
    if networks is not None:
        pins = rec.get("networks")
        if not pins:
            out.append(Problem("record", "the record pins no network checkpoint"))
        else:
            out += [Problem("checkpoints", m) for m in networks(pins)]
    if support_sha256 is not None and support_sha256 != (rec.get("regimes") or {}).get(
            "support_sha256"):
        out.append(Problem("support", "step 1's support.json differs from the tagged one"))
    if rollouts is not None:
        if not rec.get("rollouts"):
            out.append(Problem("record", "the record pins no stored rollouts"))
        else:
            out += [Problem("rollouts", m) for m in rollouts(rec["rollouts"])]
    return out


def _verify_outputs(rec: dict, root: Path) -> list[Problem]:
    out = []
    b = rec.get("outputs")
    if not b:
        return [Problem("record", "the record binds no pre-tag outputs")]
    for rel, v in b.get("files", {}).items():
        p = root / rel
        if not p.exists() or sha256_file(p) != v["sha256"]:
            out.append(Problem("outputs", f"{rel} differs from the tagged output"))
    for stage, key in (("opt-fields", "opt_fields"), ("qp-fields", "qp_fields")):
        want = b.get(key)
        if not want:
            continue
        rp = root / stage / "rows.jsonl"
        if not rp.exists() or sha256_file(rp) != want["rows_sha256"]:
            out.append(Problem("outputs", f"{stage}/rows.jsonl differs from the tagged rows"))
        arr = root / stage / "arrays"
        if stage == "qp-fields":
            arr = arr / "reader-Qp"
        if dir_hash(arr)[0] != want["arrays_hash"]:
            out.append(Problem("outputs", f"{stage}/arrays differ from the tagged arrays"))
    cal = rec.get("calibration") or {}
    if not cal.get("tolerance"):
        out.append(Problem("calibration", "the record holds no calibrated branch tolerance"))
    else:
        p = root / cal.get("path", CALIBRATION)
        if not p.exists() or sha256_file(p) != cal["sha256"]:
            out.append(Problem("calibration", f"{CALIBRATION} differs from the tagged calibration"))
    return out


def frozen_ids(rec: dict, name: str) -> list[tuple[str, str]]:
    """The tagged (kind, key_id) list of one set: what post-tag stages iterate."""
    return [tuple(t) for t in rec["key_sets"][name]]


def save(rec: dict, path: Path = PREREG_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=_default) + "\n")
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")
