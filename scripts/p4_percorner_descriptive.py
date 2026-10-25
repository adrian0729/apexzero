"""PS-P4, per-corner C and d_OOD at chosen corners: DESCRIPTIVE and POST HOC. Added after an
independent review of docs/findings/p4.md, when every P4 result had been seen. It is not part
of P4: P4 classifies on a condition's C and its D, the median d_OOD over the condition's
certified corners (docs/findings/p4-plan.md, Sections 5 and 6). Nothing here changes a
certification, a class or a reading.

    uv run python scripts/p4_percorner_descriptive.py [--corners 7 14]

**Inputs** (read only): the stored outputs of scripts/p4_exploratory.py, for the planned run
(outputs/p4/) and Amendment 2's extension (outputs/p4/extension/): build/cells_meta.json,
build/stimuli.json, certified.json, classes.json, specialists/{registered,new}/results.json
and S*_runs.npz, specialists/new/S*.npz (det_cell, det_stim, det_phase and det_s0 only), and
generalists/{reference,new}/G*_runs.npz. No agent, solver or rollout is run, and nothing under
outputs/psychophysics/ or outputs/sealed/ is read. Every built track name passes
``viz.export.lockbox_guard``.

**Per (specialist, condition, corner):** the deterministic phases completed (of 4), 'complete'
as the frozen ``interp.analysis.cond_table`` counts it (at least ceil(3n/4) of n), the cell's
d_OOD (its phase-0 run's, as results.json stores it) and whether the corner is certified for
the condition. **Per (condition, corner):** the number of the nine generalists completing it,
on S1's tier D states, as P4 runs them.

**Built-in check.** Averaging 'complete' over each condition's braking corners present, and
taking the median d_OOD over them (``interp.analysis.completion_by_condition``'s rule),
reproduces classes.json's C and D for every specialist and condition, in the planned run and
in the extension. For a new condition the corners present must equal its certified corners. A
failure stops the script.

**The d_OOD window.** Per braking corner, the length from a phase-0 run's start (det_s0) to
s_c0 + 50 m (the stimulus's landmark), over the specialists' new runs.

Writes outputs/p4/percorner.json (with a provenance block) and prints the tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p4"
EXT = OUT / "extension"
SPECIALISTS = ("S1", "S2", "S3")
GENERALISTS = ("G1", "G1s1", "G1s2", "G2", "G2s1", "G2s2", "G3", "G3s1", "G3s2")
COND_A = "F1/real/real/real"
# Condition (a), step 1's failing single-factor F1 conditions (S2: road after straightened and
# constant width; S3: constant width), and every K-str rung, planned and extension.
CONDITIONS = (
    (COND_A, "(a)"),
    ("F1/real/straight/real", "road after straightened (F1)"),
    ("F1/real/real/const", "constant width (F1)"),
    ("P4A/Kstr/1e-3", "K-str 1e-3"),
    ("P4A/Kstr/2e-3", "K-str 2e-3"),
    ("P4A/Kstr/4e-3", "K-str 4e-3"),
    ("P4A/Kstr/8e-3", "K-str 8e-3"),
    ("P4A/Kstr/16e-3", "K-str 16e-3 (extension)"),
    ("P4A/Kstr/32e-3", "K-str 32e-3 (extension)"),
    ("P4A/Kstr/64e-3", "K-str 64e-3 (extension)"),
)
TOL = 1e-12


class CheckFailed(RuntimeError):
    pass


# ------------------------------------------------------------------------- reading


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_meta(base: Path) -> dict[int, dict]:
    return {int(c["cell"]): c for c in read_json(base / "build" / "cells_meta.json")["cells"]}


def per_cell(runs: dict, d_ood: dict[int, float]) -> dict[int, dict]:
    """Per cell of one agent's runs: deterministic phases completed and run, 'complete' by the
    frozen cond_table's rule (>= ceil(3n/4) of n), and the cell's stored d_OOD (or None)."""
    det = np.asarray(runs["mode"]).astype(str) == "det"
    cell = np.asarray(runs["cell"]).astype(int)
    done = np.asarray(runs["completed"]).astype(bool)
    out = {}
    for c in np.unique(cell[det]):
        m = det & (cell == c)
        n, k = int(m.sum()), int(done[m].sum())
        out[int(c)] = {"completes": k, "n": n, "complete": bool(k >= math.ceil(3 * n / 4)),
                       "d_ood": d_ood.get(int(c))}
    return out


def load_agents(batch_dir: Path, agents, with_d_ood: bool, read: dict) -> dict[str, dict]:
    """Per agent present in one batch directory: its per-cell records."""
    res = {}
    rj = batch_dir / "results.json"
    if with_d_ood and rj.exists():
        read[rel(rj)] = sha256_file(rj)
        res = read_json(rj)["agents"]
    out = {}
    for a in agents:
        p = batch_dir / f"{a}_runs.npz"
        if not p.exists():
            continue
        read[rel(p)] = sha256_file(p)
        with np.load(p, allow_pickle=False) as z:
            runs = {k: z[k] for k in ("mode", "cell", "completed")}
        dd = {int(c): float(v) for c, v in res.get(a, {}).get("d_ood", [])}
        out[a] = per_cell(runs, dd)
    return out


def keyed(cells: dict[int, dict], meta: dict[int, dict]) -> dict[tuple[str, int], dict]:
    """Per-cell records keyed (condition, corner)."""
    out = {}
    for c, r in cells.items():
        m = meta[c]
        key = (m["condition"], int(m["corner"]))
        if key in out:
            raise CheckFailed(f"two cells for {key}")
        out[key] = r
    return out


# ------------------------------------------------------------------------- the check


def by_condition(recs: dict[tuple[str, int], dict], braking) -> dict[str, dict]:
    """completion_by_condition's rule on per-corner records: C is the mean of 'complete' over
    the braking corners present, D the median d_OOD over them (missing values skipped)."""
    out: dict[str, dict] = {}
    for (cond, corner), r in recs.items():
        if corner not in braking:
            continue
        e = out.setdefault(cond, {"corners": [], "complete": [], "d_ood": []})
        e["corners"].append(corner)
        e["complete"].append(r["complete"])
        e["d_ood"].append(np.nan if r["d_ood"] is None else r["d_ood"])
    for e in out.values():
        d = np.asarray(e["d_ood"], float)
        d = d[~np.isnan(d)]
        e["C"] = float(np.mean(e["complete"]))
        e["D"] = float(np.median(d)) if d.size else float("nan")
        e["corners"] = sorted(e["corners"])
    return out


def check_against_classes(recs: dict[tuple[str, int], dict], braking, classes_ext: dict,
                          certified: dict[str, list[int]]) -> int:
    """Compare by_condition's C and D with one specialist's classes.json 'extended' entry.
    Returns the number of conditions compared; raises CheckFailed on any difference."""
    mine = by_condition(recs, braking)
    C, D = classes_ext["C"], classes_ext["details"]["d_ood"]
    n = 0
    for cond, e in mine.items():
        if cond not in C:
            continue
        if cond in certified and e["corners"] != sorted(certified[cond]):
            raise CheckFailed(f"{cond}: corners {e['corners']} against certified "
                              f"{sorted(certified[cond])}")
        if abs(e["C"] - C[cond]) > TOL:
            raise CheckFailed(f"{cond}: C {e['C']} against classes.json {C[cond]}")
        ref = D.get(cond)
        if math.isnan(e["D"]) != (ref is None) or (ref is not None
                                                    and abs(e["D"] - ref) > TOL):
            raise CheckFailed(f"{cond}: d_OOD {e['D']} against classes.json {ref}")
        n += 1
    return n


# ------------------------------------------------------------------------- windows


def windows(base: Path, agents, braking, read: dict) -> dict[int, list[float]]:
    """Per braking corner: [min, max] over the specialists' phase-0 new runs of
    s_c0 + 50 m - det_s0, the length of the stretch d_OOD is read over."""
    stim = read_json(base / "build" / "stimuli.json")["stimuli"]
    read[rel(base / "build" / "stimuli.json")] = sha256_file(base / "build" / "stimuli.json")
    s_c0 = {int(s["stim_id"]): float(s["landmarks"]["s_c0"]) for s in stim}
    meta = cell_meta(base)
    lens: dict[int, list[float]] = {}
    for a in agents:
        p = base / "specialists" / "new" / f"{a}.npz"
        if not p.exists():
            continue
        read[rel(p)] = sha256_file(p)
        with np.load(p, allow_pickle=False) as z:
            cell, st, ph, s0 = (z[k] for k in ("det_cell", "det_stim", "det_phase", "det_s0"))
        for c, s, f, x in zip(cell, st, ph, s0, strict=True):
            corner = int(meta[int(c)]["corner"])
            if int(f) == 0 and corner in braking:
                lens.setdefault(corner, []).append(s_c0[int(s)] + 50.0 - float(x))
    return {c: [min(v), max(v)] for c, v in sorted(lens.items())}


# ------------------------------------------------------------------------- main


def rel(path: Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return r.stdout.strip()


def guard_tracks(*bases: Path) -> list[str]:
    from apexzero.viz.export import lockbox_guard

    names = set()
    for b in bases:
        for s in read_json(b / "build" / "stimuli.json")["stimuli"]:
            names.add(str((s.get("source") or {}).get("track", "")))
    for n in sorted(names):
        lockbox_guard(n)
    return sorted(names)


def fmt(r: dict | None) -> str:
    """d_OOD to 2 decimals, rounded once from the stored value, and phases completed."""
    if r is None:
        return "-"
    d = "-" if r["d_ood"] is None else f"{r['d_ood']:.2f}"
    return f"{d} ({r['completes']}/{r['n']})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corners", type=int, nargs="+", default=[7, 14])
    args = ap.parse_args(argv)
    t0 = time.time()
    tracks = guard_tracks(OUT, EXT)
    read: dict[str, str] = {}
    for p in (OUT / "certified.json", EXT / "certified.json", OUT / "classes.json",
              EXT / "classes.json", OUT / "build" / "cells_meta.json",
              EXT / "build" / "cells_meta.json"):
        read[rel(p)] = sha256_file(p)
    cert, cert_x = read_json(OUT / "certified.json"), read_json(EXT / "certified.json")
    braking = [int(c) for c in cert["braking"]]
    if [int(c) for c in cert_x["braking"]] != braking:
        raise CheckFailed("the planned run and the extension have different braking corners")
    certified = {n: e["certified_corners"] for n, e in cert["conditions"].items()}
    certified_x = {n: e["certified_corners"] for n, e in cert_x["conditions"].items()}
    meta, meta_x = cell_meta(OUT), cell_meta(EXT)

    reg = load_agents(OUT / "specialists" / "registered", SPECIALISTS, True, read)
    new = load_agents(OUT / "specialists" / "new", SPECIALISTS, True, read)
    new_x = load_agents(EXT / "specialists" / "new", SPECIALISTS, True, read)
    spec = {a: {**keyed(reg[a], meta), **keyed(new[a], meta)} for a in SPECIALISTS}
    spec_x = {a: keyed(new_x[a], meta_x) for a in SPECIALISTS}

    classes, classes_x = read_json(OUT / "classes.json"), read_json(EXT / "classes.json")
    n_checked = {}
    for a in SPECIALISTS:
        n_checked[f"planned {a}"] = check_against_classes(
            spec[a], braking, classes["specialists"][a]["classes"]["extended"], certified)
        n_checked[f"extension {a}"] = check_against_classes(
            {**spec[a], **spec_x[a]}, braking,
            classes_x["specialists"][a]["classes"]["extended"], {**certified, **certified_x})

    # The extension's generalist reference runs repeat the planned run's (a), so they are
    # not read again.
    gen = {}
    for base, mm, batch in ((OUT, meta, "reference"), (OUT, meta, "new"),
                            (EXT, meta_x, "new")):
        for g, cells in load_agents(base / "generalists" / batch, GENERALISTS, False,
                                    read).items():
            for key, r in keyed(cells, mm).items():
                gen.setdefault(key, {})[g] = r["complete"]

    table = []
    for corner in args.corners:
        for cond, label in CONDITIONS:
            cc = certified.get(cond, certified_x.get(cond))
            row = {"corner": corner, "condition": cond, "label": label,
                   "certified": None if cc is None else corner in cc, "specialists": {}}
            for a in SPECIALISTS:
                r = spec[a].get((cond, corner)) or spec_x[a].get((cond, corner))
                row["specialists"][a] = r
            g = gen.get((cond, corner))
            row["generalists_completing"] = None if g is None else int(sum(g.values()))
            row["generalists_run"] = None if g is None else len(g)
            table.append(row)

    win = {}
    for base in (OUT, EXT):
        for c, (lo, hi) in windows(base, SPECIALISTS, braking, read).items():
            w = win.setdefault(c, [lo, hi])
            win[c] = [min(w[0], lo), max(w[1], hi)]

    out = {"status": ("DESCRIPTIVE and POST HOC: per-corner values from stored P4 outputs; "
                      "not part of P4 and changes no certification, class or reading"),
           "corners": args.corners, "braking": braking, "tracks": tracks,
           "check": {"conditions_compared": n_checked, "passed": True},
           "table": table,
           "d_ood_window_m": {str(c): v for c, v in sorted(win.items())},
           "provenance": {
               "script": rel(Path(__file__)), "script_sha256": sha256_file(Path(__file__)),
               "git_head": git("rev-parse", "HEAD"),
               "changed_paths": [ln for ln in git("status", "--porcelain").splitlines() if ln],
               "argv": sys.argv[1:] if argv is None else argv, "inputs_sha256": read,
               "seconds": round(time.time() - t0, 1),
               "end": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "environment": {"python": platform.python_version(),
                               "numpy": np.__version__, "machine": platform.machine(),
                               "system": platform.system()}}}
    dest = OUT / "percorner.json"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(dest)

    print(f"check passed: {n_checked}")
    for corner in args.corners:
        print(f"\ncorner {corner}: S1 | S2 | S3 as d_OOD (phases completed); certified; "
              "generalists completing")
        for row in (r for r in table if r["corner"] == corner):
            cells = [fmt(row["specialists"][a]) for a in SPECIALISTS]
            g, n, c = (row["generalists_completing"], row["generalists_run"],
                       row["certified"])
            cert_s = "registered" if c is None else ("yes" if c else "no")
            gen_s = "-" if g is None else f"{g} of {n}"
            print(f"  {row['label']:<30} " + " | ".join(cells) + f" | {cert_s} | {gen_s}")
    print("\nd_OOD window (m) per braking corner:",
          {c: [round(v[0], 1), round(v[1], 1)] for c, v in sorted(win.items())})
    print(f"wrote {rel(dest)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", flush=True)
        raise SystemExit(1) from None
