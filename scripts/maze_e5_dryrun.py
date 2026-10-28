"""Maze E5 dry run (docs/prereg/maze-e5.md Section 8, step 4): the machinery on the pilot's checkpoints.

    uv run python scripts/maze_e5_dryrun.py reproduce             # Section 3's pilot reproduction
    uv run python scripts/maze_e5_dryrun.py analyse --n-boot 200  # audit, then analyse, one seed

A machinery check only; nothing it writes is a result. ``maze_e5.py``'s ``audit`` and ``analyse``
sit behind the tag gate, and ``analyse`` runs once and writes outputs/maze-e5/, so neither can
run before the tag. This script imports ``maze_e5`` and ``maze_audit`` as modules and calls their
functions, so the real machinery is what runs. It is not in ``maze_e5.CODE_PATHS``, writes only
under runs/maze-e5-dryrun/ (git-ignored), reads the pilot (runs/maze-audit/pilot-v2/) and
nothing else, and never touches outputs/. Run it at low priority (it lowers its own to 15).

- ``reproduce``: the pilot's own ``maze_audit.audit`` and ``maze_audit.report`` run on a copy of
  the pilot's checkpoints (500 mazes, level seed 2026, 8 episodes, 4 score episodes, 2,000
  bootstrap draws, steps 0-3) with one change: each ``audit_checkpoint`` call is replaced by
  ``e5_audit.audit_policy(..., sfl_native=False)`` at the run's ``e5_audit.settings`` (the
  mapping tests/test_maze_e5.py makes: dr -> dr, plr-maxmc -> plr). E5 adds three arrays
  (``success_rate_tgt``, ``j_pi_odd``, ``j_pi_even``); they are dropped, so each audit file
  holds exactly the keys the pilot's did. The new fidelity.json is then compared with the
  pilot's: the same keys, every number exactly equal (NaN equal to NaN), and whether the bytes
  are identical. Writes runs/maze-e5-dryrun/reproduce/{audit/, docs/, compare.json}; the
  report's tables and figures go to its docs/, not to docs/findings/.
- ``analyse``: DR and PLR-MaxMC at seed 0, four checkpoints each, through ``maze_e5``'s own
  ``audit_levels`` and ``audit_one`` (2,000 fresh mazes; the in-buffer audit for PLR), written
  with ``npz_bytes`` as ``stage_audit`` lays them out (u<update>-<part>.npz, audit-run.json
  with sha256 records), a scratch manifest shaped like ``stage_manifest``'s, then
  ``load_audits``, ``e5.point``, ``e5.bootstrap`` (``--n-boot``), ``e5.analyse``,
  ``e5.describe`` and ``decisions_md``. Writes runs/maze-e5-dryrun/{audit-*, manifest.json,
  analyse/{analysis.json, decisions.md, dryrun.json}}. Four checkpoints of one seed leave
  most tests untestable; the aim is that the whole path runs and writes well-formed output.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scripts/ (maze_e5, maze_audit)
import _bootstrap  # noqa: F401  (makes src/ importable)
import maze_audit
import maze_e5

ROOT = maze_e5.ROOT
PILOT = ROOT / "runs" / "maze-audit" / "pilot-v2"
OUT = ROOT / "runs" / "maze-e5-dryrun"
ARM = {"dr": "dr", "plr-maxmc": "plr"}           # the pilot's run names -> E5's arms
E5_ONLY = ("success_rate_tgt", "j_pi_odd", "j_pi_even")   # what audit_policy adds to the pilot's keys
PILOT_ARGS = dict(runs=["dr", "plr-maxmc"], seed=0, levels=500, level_seed=2026, episodes=8,
                  score_episodes=4, n_boot=2000, audit_steps=[0, 1, 2, 3], force=False)
VIEW = 5                                         # the pilot's agent view


def under_out(path: Path) -> Path:
    """``path``, which must lie inside runs/maze-e5-dryrun/ (the only place this writes)."""
    path = Path(path).resolve()
    if OUT.resolve() not in path.parents:
        raise SystemExit(f"{path} is outside {OUT}")
    return path


def fresh_dir(path: Path) -> Path:
    """An empty directory: a rerun deletes the previous dry run's copy (E5 itself never would)."""
    path = under_out(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def pilot_copy() -> Path:
    """A copy of the pilot's checkpoints and timing under OUT, so nothing under runs/maze-audit/
    is opened for writing (orbax manages the directories it reads). Checked by sha256."""
    dst = OUT / "pilot"
    if not dst.exists():
        dst.mkdir(parents=True)
        shutil.copytree(PILOT / "checkpoints", dst / "checkpoints")
        shutil.copy(PILOT / "timing.json", dst / "timing.json")
    for name in PILOT_ARGS["runs"]:
        a, b = (maze_e5.tree_sha256(p / "checkpoints" / name) for p in (PILOT, dst))
        if a != b:
            raise SystemExit(f"the copy of {name}'s checkpoints differs from the pilot's")
    return dst


# --------------------------------------------------------------------------- reproduce


def differences(a, b, path="") -> list[str]:
    """Where two parsed JSON values differ: keys, types, and numbers compared exactly (no
    tolerance; NaN equals NaN)."""
    here = path or "<root>"
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{here}: key only in the pilot: {k}" for k in a.keys() - b.keys()]
        out += [f"{here}: key only in the new file: {k}" for k in b.keys() - a.keys()]
        for k in a.keys() & b.keys():
            out += differences(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{here}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{path}[{i}]")]
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool | type(None)):
        same = a == b or (math.isnan(a) and math.isnan(b))
        return [] if same else [f"{here}: {a!r} vs {b!r}"]
    return [] if type(a) is type(b) and a == b else [f"{here}: {a!r} vs {b!r}"]


def e5_audit_checkpoint(ck, levels, key, episodes, score_episodes, oracle=None):
    """``maze.audit_checkpoint``'s signature and keys, computed by E5's ``audit_policy``."""
    from apexzero.maze import audit as maze
    from apexzero.maze import e5_audit as ea

    arm = ARM[ck.config["run_name"]]
    res = ea.audit_policy(maze.network_policy(ck.params), ea.settings(arm, ck.config), levels,
                          oracle, key, episodes, score_episodes, sfl_native=False)
    for k in E5_ONLY:
        res.pop(k)
    return res


class PilotModule:
    """``maze_audit.maze`` with ``audit_checkpoint`` swapped for E5's; the rest is the module."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return e5_audit_checkpoint if name == "audit_checkpoint" else getattr(self._module, name)


def cmd_reproduce(a) -> int:
    t0 = time.time()
    work = fresh_dir(OUT / "reproduce")
    src = pilot_copy()
    (work / "checkpoints").symlink_to(src / "checkpoints")
    shutil.copy(src / "timing.json", work / "timing.json")
    args = SimpleNamespace(**{**PILOT_ARGS, "n_boot": a.n_boot})
    maze_audit.OUT, maze_audit.DOCS = work, work / "docs"
    real = maze_audit.maze
    maze_audit.maze = PilotModule(real)
    try:
        t1 = time.time()
        maze_audit.audit(args)                  # the pilot's own flow, E5's audit_policy inside
        t2 = time.time()
    finally:
        maze_audit.maze = real
    maze_audit.report(args)                     # fidelities() and fidelity.json, as the pilot did
    t3 = time.time()

    old_bytes = (PILOT / "audit" / "fidelity.json").read_bytes()
    new_bytes = (work / "audit" / "fidelity.json").read_bytes()
    diffs = differences(json.loads(old_bytes), json.loads(new_bytes))
    # Why a difference arises: the per-checkpoint arrays, which the pilot saved.
    import numpy as np

    array_diffs = []
    for p in sorted((PILOT / "audit").glob("*.npz")):
        o, n = dict(np.load(p)), dict(np.load(work / "audit" / p.name))
        if o.keys() != n.keys():
            array_diffs.append(f"{p.name}: keys {sorted(o.keys() ^ n.keys())}")
        array_diffs += [f"{p.name}: {k}" for k in o.keys() & n.keys()
                        if k not in ("wall_s", "cpu_s") and not np.array_equal(o[k], n[k],
                                                                              equal_nan=True)]
    result = dict(passed=not diffs, bytes_identical=old_bytes == new_bytes, n_differences=len(diffs),
                  first_differences=diffs[:20], audit_array_differences=array_diffs[:20],
                  n_boot=a.n_boot, wall_s=dict(audit=t2 - t1, report=t3 - t2, total=t3 - t0))
    (work / "compare.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"reproduce: {'PASS' if not diffs else 'FAIL'}; bytes identical: "
          f"{result['bytes_identical']}; {len(diffs)} differences; audit {t2 - t1:.0f} s, "
          f"report {t3 - t2:.0f} s")
    for d in diffs[:10]:
        print("  ", d)
    for d in array_diffs[:10]:
        print("   array:", d)
    return 0 if not diffs else 1


# --------------------------------------------------------------------------- analyse


def timed(timing: dict, stage: str, fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    timing[stage] = timing.get(stage, 0.0) + time.time() - t0
    print(f"  {stage}: {timing[stage]:.1f} s", flush=True)
    return out


def cmd_analyse(a) -> int:
    from apexzero.maze import audit as maze
    from apexzero.maze import e5

    timing: dict = {}
    src = pilot_copy()
    OUT.mkdir(parents=True, exist_ok=True)
    names = {n: f"dryrun-{ARM[n]}-v{VIEW}-s0" for n in PILOT_ARGS["runs"]}
    audit_names = {v: f"maze-e5-dryrun/audit-{v}" for v in names.values()}
    for d in audit_names.values():
        fresh_dir(maze_e5.RUNS / d)
    out = fresh_dir(OUT / "analyse")

    # The scratch manifest: stage_manifest's entry for each run, from the pilot's checkpoints.
    entries, loaded = {}, {}
    for pilot_name, name in names.items():
        arm, ckdir = ARM[pilot_name], maze_e5.jaxued_dir(src, pilot_name, 0)
        cks = []
        for step in maze.checkpoint_steps(ckdir):
            update = maze.load_checkpoint(ckdir, step).num_updates
            cks.append({"update": update, "step": step, "path": maze_e5.rel(ckdir / "models" / str(step)),
                        "sha256": maze_e5.tree_sha256(ckdir / "models" / str(step))})
        entries[name] = {"arm": arm, "view": VIEW, "seed": 0, "updates": cks[-1]["update"],
                         "status": "trained", "config": maze_e5.rel(ckdir / "config.json"),
                         "config_sha256": maze_e5.sha256_bytes((ckdir / "config.json").read_bytes()),
                         "checkpoints": cks}
        loaded[name] = pilot_name
    manifest = {"round": "maze-e5", "dryrun": True, "drop": 0, "written": maze_e5.now(),
                "vendored": maze_e5.sfl.check_vendored(), "missing": [], "runs": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    # stage_audit's body for each run, without the tag gate.
    first = {n: maze_e5.load_any(e["arm"], e["checkpoints"][0], src, 0, loaded[n])
             for n, e in entries.items()}
    levels, oracles = timed(timing, "audit_levels", maze_e5.audit_levels,
                            [e["arm"] for e in entries.values()], [first[n][2] for n in entries])
    inbuf = {}
    for name, entry in entries.items():
        out_dir, files = maze_e5.RUNS / audit_names[name], {}
        for ckpt in entry["checkpoints"]:
            t0 = time.time()
            ck, policy, cfg = maze_e5.load_any(entry["arm"], ckpt, src, 0, loaded[name])
            res = timed(timing, "audit_one", maze_e5.audit_one, entry["arm"], ck, policy, cfg,
                        list(entries).index(name), ckpt["update"], levels, oracles)
            for part, arrays in res.items():
                path = out_dir / f"u{ckpt['update']:05d}-{part}.npz"
                files[path.name] = maze_e5.write_new(path, maze_e5.npz_bytes(
                    **arrays, update=ckpt["update"], checkpoint_sha256=ckpt["sha256"],
                    wall_s=time.time() - t0))
            inbuf[name, ckpt["update"]] = "inbuf_stored" in res["fresh"]
            f = res["fresh"]
            print(f"{name} update {ckpt['update']}: solve "
                  f"{f['success_rate_tgt'][f['j_star'] > 0].mean():.1%}, in-buffer audit "
                  f"{'ran (' + str(len(f['inbuf_stored'])) + ' levels)' if inbuf[name, ckpt['update']] else 'not run'}",
                  flush=True)
        maze_e5.write_new(out_dir / "audit-run.json", (json.dumps(
            {"run": name, "arm": entry["arm"], "view": entry["view"], "seed": 0,
             "tag": {"dryrun": True}, "files": files, "written": maze_e5.now()}, indent=1)
            + "\n").encode())

    # stage_analyse's body after the gate.
    runs = timed(timing, "load_audits", maze_e5.load_audits, manifest, audit_names)
    pt = timed(timing, "point", e5.point, runs)
    boot = timed(timing, "bootstrap", e5.bootstrap, runs, a.n_boot, e5.BOOT_SEED,
                 min(a.workers, maze_e5.MAX_WORKERS))
    res = timed(timing, "analyse", e5.analyse, runs, boot, pt)
    res["describe"] = timed(timing, "describe", e5.describe, runs, pt)
    res["provenance"] = {"dryrun": True, "n_boot": a.n_boot, "boot_seed": e5.BOOT_SEED,
                         "runs": len(runs), "missing": manifest["missing"]}
    maze_e5.write_json_new(out / "analysis.json", res)
    text = timed(timing, "decisions_md", maze_e5.decisions_md, res)
    maze_e5.write_new(out / "decisions.md", text.encode())
    json.loads((out / "analysis.json").read_text())           # well-formed (allow_nan=False)

    summary = {"n_boot": a.n_boot, "workers": min(a.workers, maze_e5.MAX_WORKERS),
               "wall_s": timing, "inbuffer_ran": {f"{k[0]}/{k[1]}": v for k, v in inbuf.items()},
               "rows": [{"arm": r["arm"], "score": r["score"], "rows": r["rows"]} for r in res["rows"]],
               "labels": {arm: {"|".join(k): v["label"] for k, v in x["labels"].items()}
                          for arm, x in res["arms"].items()}}
    (out / "dryrun.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"analyse: n_boot {a.n_boot}; wall s per stage "
          + ", ".join(f"{k} {v:.1f}" for k, v in timing.items()))
    print("in-buffer audit ran:", {k: v for k, v in summary["inbuffer_ran"].items()})
    print("rows:", summary["rows"])
    print("labels:", summary["labels"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("reproduce", help="Section 3's pilot reproduction")
    p.add_argument("--n-boot", type=int, default=PILOT_ARGS["n_boot"], help="the pilot's: 2000")
    p = sub.add_parser("analyse", help="audit and analyse the pilot's checkpoints, one seed")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=4, help="processes (at most 4)")
    a = ap.parse_args(argv)
    maze_e5.lower_priority(15)
    return {"reproduce": cmd_reproduce, "analyse": cmd_analyse}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
