"""Step 2's pre-registration record (design Section 9): what the tag fixes besides the code.

The record is ``src/apexzero/fields/prereg.json``. It lies inside the frozen paths, so git
freezes it with the code at the tag receptive-fields-prereg-v1, and scripts/fields.py's freeze
check reads it from the tagged commit (``git show <tag>:<path>``), not from the working tree.
It is written last before the tag (``build_record`` by the 'prereg' stage) and holds:

- the key lists (``key_sets``): the A2 set (all 481 keys, the 6 horizon twins listed apart),
  the anchored finals per agent and the earlier snapshots per (agent, snapshot), as sorted key
  ids; a content hash of each set (``content_hash``: sha256 over each listed key's .json and
  .npz bytes, in id order); each agent's final timesteps;
- the re-polish movement per key (compact rows: max_dx, dJ_rel, flagged, ok) and their hash;
- step 1's stimulus design hash (``stimuli.design_hash``), the design document's sha256, and
  ``io.code_hash`` over src/apexzero and scripts/fields.py (every .py file: the imported ocp,
  vehicle, tracks, env and agents code too; the record itself is not .py);
- the tolerances and bands the gates and measures use (the code constants, copied for the
  reader; the code hash is what binds them).

``verify_record`` recomputes each item from disk and lists every mismatch: a missing or
edited key file, a set that no longer matches the index (e.g. H3's new anchored keys leaking
into the anchored finals), a re-polish run whose movement differs from the recorded one by
more than ``REPOLISH_RECHECK_TOL``, a changed design or design document, or code changed after
the record was written. Post-tag stages iterate ``frozen_ids``, never a fresh glob.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

PREREG_NAME = "prereg.json"
PREREG_PATH = Path(__file__).with_name(PREREG_NAME)
PREREG_REL = f"src/apexzero/fields/{PREREG_NAME}"
DESIGN_DOC = "docs/interp/receptive-fields-design.md"
CODE_ROOTS = ("src/apexzero", "scripts/fields.py")
REPOLISH_RECHECK_TOL = 1e-6      # m: a re-polish elsewhere may move a key this much more
VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def tolerances() -> dict:
    """The gate and measure constants (a copy for the reader; the code hash binds them)."""
    from apexzero.fields import gates as G
    from apexzero.fields import kkt as KK
    from apexzero.fields import measures as MS
    from apexzero.fields import network as N

    return {"gate0": G.GATE0_TOL, "gate1": [KK.GATE1_MEDIAN, KK.GATE1_P95, KK.ZERO_ATOL],
            "gate1_fd_h": KK.FD_H, "gate1_resolve_tol": KK.GATE1_TOL,
            "gate2": [N.GATE2_TOL, N.ZERO_ATOL, N.RICH_RTOL, N.KINK_FRACTION],
            "gate2_h_max": N.H_MAX, "gate3_mean": N.MEAN_TOL,
            "gate3_step": [float(v) for v in N.STEP_TOL[1:]], "gate3_accel": N.ACCEL_TOL,
            "branch": N.BRANCH_TOL, "gate4": list(KK.GATE4_BAND),
            "gate5": [G.GATE5_TOL, G.GATE5_STATE_TOL, G.GATE5_BOUND_TOL, G.GATE5_TIE],
            "adjoint_rtol": KK.ADJOINT_RTOL, "loop_steps": N.STEPS,
            "horizon_margin": N.HORIZON_MARGIN, "edge_plateau": MS.EDGE_PLATEAU,
            "far_geometry": [MS.FAR_LOG_R, MS.FAR_THETA], "h1_gain": list(MS.H1_GAIN),
            "h1_location": MS.H1_LOCATION, "h2": [MS.H2_DEFICIT_HI, *MS.H2_LAWFUL, MS.H2_OVER_LO],
            "coverage_max": MS.COVERAGE_MAX}


def build_record(repo: Path, sets: dict[str, list[tuple[str, str]]], keys_root: Path,
                 repolish: list[dict], design_hash: str = "", final_timesteps: dict | None = None,
                 tols: dict | None = None) -> dict:
    """The record (module docstring) from the computed sets, the re-polish rows and the
    design hash; ``repo`` is the repository root (design document and code hash)."""
    repo = Path(repo)
    rp = repolish_rows(repolish)
    missing = sorted({k for v in sets.values() for _, k in v} - set(rp))
    return {"version": VERSION,
            "key_sets": {k: [list(t) for t in v] for k, v in sets.items()},
            "key_counts": {k: len(v) for k, v in sets.items()},
            "content": {k: content_hash(v, keys_root) for k, v in sets.items()},
            "final_timesteps": dict(final_timesteps or {}),
            "repolish": rp, "repolish_hash": _canon(rp), "repolish_missing": missing,
            "design_hash": design_hash,
            "design_doc_sha256": sha256_file(repo / DESIGN_DOC)
            if (repo / DESIGN_DOC).exists() else "",
            "code_hash": code_hash(repo), "tolerances": dict(tols or {})}


def code_hash(repo: Path, roots: tuple[str, ...] = CODE_ROOTS) -> str:
    """``io.code_hash``'s rule (every .py file under roots: path and bytes) on a given
    repository root."""
    h = hashlib.sha256()
    repo = Path(repo)
    files: list[Path] = []
    for r in roots:
        p = repo / r
        files += sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])
    for f in files:
        h.update(str(f.relative_to(repo)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def verify_record(rec: dict, repo: Path, keys_root: Path,
                  current_sets: Callable[[], dict] | None = None,
                  repolish: list[dict] | None = None, design_hash: str | None = None,
                  final_timesteps: dict | None = None) -> list[str]:
    """Every way the disk differs from the tagged record (module docstring); [] when it
    matches. ``current_sets()`` recomputes the sets from the key index; ``repolish`` are this
    machine's re-polish rows (None: not checked); ``design_hash`` the current step-1 design's."""
    out = []
    repo = Path(repo)
    if int(rec.get("version", -1)) != VERSION:
        out.append(f"pre-registration record version {rec.get('version')} is not {VERSION}")
    sets = {k: [tuple(t) for t in v] for k, v in rec.get("key_sets", {}).items()}
    for k, ids in sets.items():
        h = content_hash(ids, keys_root)
        if h != rec["content"].get(k):
            out.append(f"key set {k}: the key files differ from the tagged ones")
    if current_sets is not None:
        now = current_sets()
        for k in sorted(set(now) | set(sets)):
            if [tuple(t) for t in now.get(k, [])] != sets.get(k, []):
                out.append(f"key set {k}: the index no longer gives the tagged list")
    if final_timesteps is not None and dict(final_timesteps) != rec.get("final_timesteps", {}):
        out.append("an agent's final snapshot differs from the tagged one")
    if repolish is not None:
        now = repolish_rows(repolish)
        if _canon(now) != rec.get("repolish_hash"):
            recd = rec.get("repolish", {})
            bad = [k for k, v in recd.items()
                   if k not in now or bool(now[k][3]) != bool(v[3])
                   or not abs(float(now[k][0]) - float(v[0])) <= REPOLISH_RECHECK_TOL]
            if bad:
                out.append(f"re-polish differs from the recorded movement on {len(bad)} keys "
                           f"(e.g. {bad[:3]})")
    if design_hash is not None and design_hash != rec.get("design_hash"):
        out.append("step 1's stimulus design differs from the tagged one")
    doc = repo / DESIGN_DOC
    if rec.get("design_doc_sha256") and (not doc.exists()
                                         or sha256_file(doc) != rec["design_doc_sha256"]):
        out.append(f"{DESIGN_DOC} differs from the tagged design")
    if code_hash(repo) != rec.get("code_hash"):
        out.append("code changed after the pre-registration record was written")
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
