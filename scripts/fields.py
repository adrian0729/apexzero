"""Receptive fields, step 2: the stages of the design (docs/interp/receptive-fields-design.md).

    uv run python scripts/fields.py <stage> [options]

Pre-tag stages (design Sections 8 and 10; no trained network's field is computed):

    keys            the key lists (A2 set, anchored finals and earlier snapshots) with their
                    strata and the loop audit (stored horizons against the loop length),
                    and Gate 0 on them (no solves)
    repolish        one REFINE_IPOPT re-polish per key, its movement recorded (Section 5);
                    passes only when every key re-polished
    gate1           KKT fields against Richardson FD of warm re-solves on the stratified
                    sample, with the planted bugs
    gate2           network AD against FD for a random actor, the planted edge keeper and the
                    linear driver
    gate3           teacher-forced wiring of trained snapshots (no derivatives) with the
                    planted wiring errors; --all checks every key with a det row
    gate4           G* of the optimum on every A2 key (all 481; --no-twins leaves out the 6
                    horizon twins)
    gate5           the KKT replay through the network-side loop, and CasADi against JAX
                    state tangents, on the Gate 1 sample
    controls        RF-C1..C5 and the unlawful readers through the network pipeline
    opt-fields      the optimum's fields per key (A2 and anchored), persisted with their
                    exclusion flags (agent-independent on the A2 keys)
    phase-ceiling   the optimum's fields across each A2 cell's phases (from opt-fields)
    prereg          the pre-registration record src/apexzero/fields/prereg.json (key lists
                    and their hashes, re-polish movement, design and code hashes): the last
                    step before the tag

Post-tag stages (refused unless the freeze check passes; ``--post-tag-deviation REASON``
logs a dated deviation and lets a changed-code run proceed):

    fields-a2       the networks' fields on the A2 keys (H2', H4, H4-transfer, H1-typical)
    fields-anchored the networks' fields on the anchored finals (H1-fast)
    dose            the width dose-response subsample, optimum and networks
    h3-keys         H3's new key solves (step 2's own key cache)
    h3-lambda       G2's lambda-grid rollouts and lambda*_lin at H3's units
    analyse         the analysis (once: --confirmatory refuses a second run)

Each stage writes under outputs/fields/<stage>/ (``--out`` moves the root) and logs to
runs/fields-<stage>.log (``--log-dir``), with a done/total line at least once a minute and a
note in runs/activity.md while it runs (only when the log is under runs/). Network stages run
in float64: this script switches x64 on before a pre-tag stage runs (a post-tag stage after
its freeze check), and never runs step 1's float32 harness (the optimum's stages read float64
either way). XLA is held to 2 threads unless --threads says otherwise.

Freeze rule (design Section 9; ``freeze_problems``): the tag receptive-fields-prereg-v1 must
exist, and so must step 1's psychophysics-prereg-v2 (fail closed); nothing under src/apexzero
or scripts/fields.py may differ from the tag (working tree, later commits or untracked
files: later commits touching only docs or findings are fine); step 1's frozen code must be
its own tag's; and the pre-registration record at the tag (``fields.prereg``) must match the
disk: the key files and lists, the re-polish movement, step 1's design, the design document
and the code hash. Post-tag stages iterate the record's frozen key lists, never a fresh glob.
After the check, ``network.unlock_trained`` allows trained networks' fields.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _threads(argv: list[str]) -> int | None:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return None


_T = _threads(sys.argv)
if _T is not None or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_T or 2}")

import _bootstrap  # noqa: F401  (makes src/ importable)

ROOT = Path(__file__).resolve().parents[1]
TAG = "receptive-fields-prereg-v1"
FROZEN_PATHS = ("src/apexzero/fields", "scripts/fields.py")
STEP1_TAG = "psychophysics-prereg-v2"
STEP1_FROZEN = ("src/apexzero/interp", "scripts/psychophysics.py")
CODE_PATHS = ("src/apexzero", "scripts/fields.py")   # nothing here may change after the tag
PRE_TAG = ("keys", "repolish", "gate1", "gate2", "gate3", "gate4", "gate5", "controls",
           "opt-fields", "phase-ceiling", "calibrate", "prereg")
# calibrate builds float32 references in this process (the harness needs x64 off) and runs the
# float64 dry runs in subprocesses, so it is the one pre-tag stage that leaves x64 off here.
X64_OFF = ("calibrate",)
# Its controls: the planted linear driver on the A2 cells where it survives the window
# (mirror pairs), and step 1's lane keeper (generic-callable path), as in tests/.
CAL_LINEAR_CELLS = (344, 345, 62, 63, 66, 67)
CAL_LINEAR_STEPS = 200
CAL_KEEPER_STARTS = ((380, 0), (51, 1))
CAL_RANDOM_STARTS = ((62, 0), (345, 1))
POST_TAG = ("fields-a2", "fields-anchored", "dose", "h3-keys", "h3-lambda", "analyse")
STAGES = PRE_TAG + POST_TAG
OUT = ROOT / "outputs" / "fields"
PREREG_REL = "src/apexzero/fields/prereg.json"


# ------------------------------------------------------------------------- git and the freeze


def _git(*args: str, root: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _tag_commit(tag: str, root: Path = ROOT) -> str | None:
    r = _git("rev-list", "-n", "1", tag, root=root)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def changed_since(tag: str, paths: tuple[str, ...], root: Path = ROOT) -> list[str]:
    """Files under ``paths`` that differ from the tagged commit (working tree or later
    commits), plus untracked ones; [] when they are the tagged code. A git failure is itself
    reported (as '<git error: ...>'), never read as 'no change'."""
    r = _git("diff", "--name-only", tag, "--", *paths, root=root)
    if r.returncode != 0:
        return [f"<git error: {r.stderr.strip()[:200]}>"]
    changed = [ln for ln in r.stdout.splitlines() if ln.strip()]
    u = _git("ls-files", "--others", "--exclude-standard", "--", *paths, root=root)
    if u.returncode != 0:
        return changed + [f"<git error: {u.stderr.strip()[:200]}>"]
    return changed + [f"{ln} (untracked)" for ln in u.stdout.splitlines() if ln.strip()]


def tagged_record(tag: str = TAG, root: Path = ROOT) -> dict | None:
    """The pre-registration record as the tagged commit holds it (None if absent)."""
    r = _git("show", f"{tag}:{PREREG_REL}", root=root)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return json.loads(r.stdout)


def _verify_on_disk(rec: dict, root: Path) -> list[str]:
    """``prereg.verify_record`` against this repository's key store, re-polish rows, design
    and agents."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg
    from apexzero.interp import io, stimuli

    rows_path = OUT / "repolish" / "repolish.jsonl"
    rows = ([json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
            if rows_path.exists() else None)
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    ft = {a: K.final_timesteps(a) for a in rec.get("final_timesteps", {})}
    probs = prereg.verify_record(rec, root, io.KEYS_DIR, lambda: prereg.key_sets(),
                                 rows, stimuli.design_hash(bank, cells), ft)
    if rows is None:
        probs.append(f"no re-polish rows at {rows_path} to check against the record")
    return probs


def freeze_problems(tag: str = TAG, root: Path = ROOT, verify=_verify_on_disk) -> list[str]:
    """Why a post-tag stage may not run on what is checked out (module docstring): a missing
    tag (step 2's or step 1's), any change since the tag under src/apexzero or
    scripts/fields.py (or a git failure), step 1's frozen code differing from its tag, a
    missing pre-registration record at the tag, or a record that the disk no longer matches
    (``verify(record, root)``)."""
    if _tag_commit(tag, root) is None:
        return [(f"the git tag {tag} does not exist yet (design Section 9: the fields of "
                 "trained networks are computed only after it)")]
    out = []
    ch = changed_since(tag, CODE_PATHS, root)
    if ch:
        out.append(f"code differs from the tag {tag}: {ch[:8]}")
    if _tag_commit(STEP1_TAG, root) is None:
        out.append(f"step 1's tag {STEP1_TAG} is missing: its frozen code cannot be checked")
    else:
        ch1 = changed_since(STEP1_TAG, STEP1_FROZEN, root)
        if ch1:
            out.append(f"step 1's frozen code differs from {STEP1_TAG}: {ch1[:8]}")
    rec = tagged_record(tag, root)
    if rec is None:
        out.append(f"no pre-registration record {PREREG_REL} at the tag {tag}")
    elif verify is not None:
        out.extend(verify(rec, root))
    return out


def require_tag(what: str, deviation: str | None = None, log_root: Path | None = None) -> None:
    """Refuse a post-tag stage unless ``freeze_problems`` is empty. With ``deviation`` (a
    reason), the problems are logged as a dated deviation (<log_root>/deviations.jsonl) and
    the stage proceeds, except when the tag itself is missing."""
    probs = freeze_problems()
    if not probs:
        return
    if deviation and _tag_commit(TAG) is not None:
        rec = {"date": time.strftime("%Y-%m-%d"), "time": time.strftime("%H:%M:%S"),
               "stage": what, "reason": deviation, "problems": probs}
        path = Path(log_root or OUT) / "deviations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"post-tag deviation logged in {path}: {deviation}", flush=True)
        return
    raise SystemExit(f"refusing to run {what}: " + "; ".join(probs))


# ------------------------------------------------------------------------- logs and outputs


def stage_log(args, stage: str, total: int, note: str = ""):
    """io.StageLog writing runs/fields-<stage>.log (or under --log-dir)."""
    from apexzero.interp import io

    class FieldsLog(io.StageLog):
        def __init__(self):
            super().__init__(stage, total, note=note)
            self.log_dir = Path(args.log_dir)
            self.path = self.log_dir / f"fields-{stage}.log"
            self._note = f"fields {stage} [pid {os.getpid()}]"

        def _activity(self, text):
            if self.log_dir.resolve() == io.RUNS_DIR.resolve():
                super()._activity(text)

    return FieldsLog()


def _out(args, stage: str) -> Path:
    d = Path(args.out) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(path: Path, obj) -> Path:
    from apexzero.interp import io

    io.write_json(path, obj)
    print(f"wrote {path}", flush=True)
    return path


def _x64() -> None:
    import jax

    jax.config.update("jax_enable_x64", True)


def _limit(refs: list, n: int | None, spread: bool = True) -> list:
    """The first n keys, or n spread evenly over the list (``spread``)."""
    if n is None or n >= len(refs):
        return list(refs)
    if not spread:
        return list(refs[:n])
    import numpy as np

    return [refs[i] for i in np.linspace(0, len(refs) - 1, int(n)).astype(int)]


def _refs(kinds: list[str]):
    """Key refs of the requested kinds: 'a2', 'finals', 'earlier' (anchored)."""
    from apexzero.fields import keys as K

    out = []
    if "a2" in kinds:
        out += list(K.a2_refs())
    if "finals" in kinds:
        out += [r for refs in K.anchored_finals().values() for r in refs]
    if "earlier" in kinds:
        out += [r for refs in K.anchored_earlier().values() for r in refs]
    return out


# ------------------------------------------------------------------------- pre-tag stages


def stage_keys(args) -> None:
    """The key lists with their strata, then Gate 0 on the requested kinds."""
    import dataclasses

    from apexzero.fields import gates as G
    from apexzero.fields import keys as K

    out = _out(args, "keys")
    a2 = K.a2_refs()
    finals = K.anchored_finals()
    earlier = K.anchored_earlier()
    rows = []
    for r in a2:
        st = K.strata(r)
        rows.append({**dataclasses.asdict(r), **{f"strata_{k}": v for k, v in
                                                 dataclasses.asdict(st).items()}})
    lists = {"a2": {"n": len(a2), "by_kind_W": _count(a2, lambda r: f"{r.kind}/{r.W}"),
                    "w_twins": [r.key_id for r in a2 if not K.strata(r).w_is_cell],
                    "cap_binds": [r.key_id for r in a2 if K.strata(r).cap_binds],
                    "keys": rows},
             "anchored_finals": {a: {"n": len(v), "snapshot": v[0].snapshot if v else "",
                                     "keys": [r.key_id for r in v]} for a, v in finals.items()},
             "anchored_earlier": {f"{a}@{s}": len(v) for (a, s), v in earlier.items()}}
    lists["n_finals"] = sum(v["n"] for v in lists["anchored_finals"].values())
    lists["a2_primary"] = [r.key_id for r in K.a2_primary_refs()]
    _write(out / "lists.json", lists)
    if not args.no_audit:
        from apexzero.fields import network as N
        from apexzero.fields import stages as S

        audit = N.loop_audit()
        audit["h3"] = S.h3_loop_audit()
        _write(out / "loop_audit.json", audit)
        print(f"loop audit: max stored horizon {audit['max']} (limit {audit['limit']}), "
              f"over {audit['n_over']}, open {audit['n_open']}, ok {audit['ok']}; H3 max "
              f"{audit['h3']['max']} (limit {audit['h3']['limit']}), ok {audit['h3']['ok']}")
        if not (audit["ok"] and audit["h3"]["ok"]):
            raise SystemExit("the loop is too short for a stored horizon: raise network.STEPS "
                             "or stages.H3_STEPS")
    if args.no_gate0:
        return
    refs = _limit(_refs(args.kinds), args.limit)
    with stage_log(args, "keys", len(refs), note=f"gate 0 on {len(refs)} keys") as log:
        res = G.gate0(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(out / "gate0.json", res)
    s = res["summary"]
    print(f"Gate 0: {s['n_ok']}/{s['n']} ok, max |r_free| {s['max_r_free']:.2e}, "
          f"pass {s['pass']}")


def _count(refs, key) -> dict:
    out: dict = {}
    for r in refs:
        out[key(r)] = out.get(key(r), 0) + 1
    return dict(sorted(out.items()))


def _repolish_chunk(refs) -> list[dict]:
    from apexzero.fields import keys as K

    bank, _ = K.design()
    out = []
    for r in refs:
        try:
            out.append(K.summary(K.repolish(K.rebuild(K.load_key(r), bank))))
        except Exception as e:  # noqa: BLE001
            out.append({"key_id": r.key_id, "kind": r.kind, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
    return out


def stage_repolish(args) -> None:
    """One re-polish per key; resumes from outputs/fields/repolish/repolish.jsonl."""
    from apexzero.fields import gates as G

    out = _out(args, "repolish")
    path = out / "repolish.jsonl"
    done = set()
    if path.exists() and not args.force:
        done = {json.loads(ln)["key_id"] for ln in path.read_text().splitlines() if ln.strip()}
    elif path.exists():
        path.unlink()
    refs = [r for r in _limit(_refs(args.kinds), args.limit) if r.key_id not in done]
    with stage_log(args, "repolish", len(refs)) as log, open(path, "a") as fh:
        def progress(n):
            log.step(n)

        for i in range(0, len(refs), 50 * max(args.workers, 1)):
            rows = G.map_keys(_repolish_chunk, refs[i: i + 50 * max(args.workers, 1)],
                              args.workers, chunk=50, progress=progress)
            fh.writelines(json.dumps(_jsonl(row)) + "\n" for row in rows)
            fh.flush()
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    summ = {"n": len(rows), "n_ok": sum(bool(r.get("ok")) for r in rows),
            "n_flagged": sum(bool(r.get("flagged")) for r in rows),
            "max_dx": max((r.get("max_dx", 0.0) for r in rows), default=0.0),
            "max_dJ_rel": max((abs(r.get("dJ_rel", 0.0)) for r in rows), default=0.0),
            "max_iterations": max((r.get("iterations", 0) for r in rows), default=0),
            "not_ok": [r["key_id"] for r in rows if not r.get("ok")]}
    summ["pass"] = bool(rows and summ["n_ok"] == summ["n"])
    _write(out / "summary.json", summ)
    print(f"re-polish: {summ['n_ok']}/{summ['n']} ok, {summ['n_flagged']} flagged, "
          f"pass {summ['pass']}")


def _jsonl(row: dict) -> dict:
    from apexzero.interp import io

    return json.loads(io.json_bytes(row))


def _gate1_refs(args):
    from apexzero.fields import gates as G

    sample = G.gate1_sample()
    if args.n is None or args.n >= len(sample):
        return sample
    # round-robin over the strata so a small run still spans them
    by: dict[str, list] = {}
    for r in sample:
        by.setdefault(G.family(r), []).append(r)
    out, k = [], 0
    while len(out) < args.n:
        for s in by.values():
            if k < len(s) and len(out) < args.n:
                out.append(s[k])
        k += 1
    return out


def stage_gate1(args) -> None:
    from apexzero.fields import gates as G

    refs = _gate1_refs(args)
    with stage_log(args, "gate1", len(refs)) as log:
        res = G.gate1(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate1.json", res)
    d = res["decision"]
    print(f"Gate 1 on {len(refs)} keys: median {d['clean']['median']:.2e}, p95 "
          f"{d['clean']['p95']:.2e}, pass {d['pass']}; plants fail "
          f"{ {p: v['fails'] for p, v in d['plants'].items()} }")


def stage_gate2(args) -> None:
    from apexzero.fields import gates as G

    rows = G.gate2_policies(tuple(args.random), tuple(args.keeper), tuple(args.linear))
    with stage_log(args, "gate2", len(rows)) as log:
        res = []
        for label, net, rd, st, steps in rows:
            res.append(G.gate2_row(net, rd, st, steps, label=label))
            log.step(note=label)
    out = {"rows": res, "tol": G.GATE2_TOL, "pass": all(r["pass"] for r in res)}
    _write(_out(args, "gates") / "gate2.json", out)
    for r in res:
        print(f"Gate 2 {r['label']}: rel_err {r['rel_err']} share {r['share_counted']}")


def stage_gate3(args) -> None:
    from apexzero.fields import gates as G

    agents = args.agent or ["G2"]
    with stage_log(args, "gate3", len(agents)) as log:
        res = {}
        for a in agents:
            res[a] = G.gate3_agent(a, args.n_anchored, args.n_a2, all_keys=args.all,
                                   limit=args.limit)
            log.step(note=a)
    out = {"agents": res, "pass": all(v["pass"] for v in res.values())}
    _write(_out(args, "gates") / "gate3.json", out)
    for a, v in res.items():
        errs = [r.get("mean_err") for r in v["rows"]]
        print(f"Gate 3 {a}: pass {v['pass']}, mean errors {errs}")


def stage_gate4(args) -> None:
    from apexzero.fields import gates as G
    from apexzero.fields import keys as K

    refs = _limit(list(K.a2_primary_refs() if args.no_twins else K.a2_refs()), args.limit)
    with stage_log(args, "gate4", len(refs)) as log:
        res = G.gate4(refs, workers=args.workers, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate4.json", res)
    print(f"Gate 4 on {res['n']} A2 keys: {res['n_outside']} outside, max |G* - 1| "
          f"{res['max_abs_dev']:.2e}, pass {res['pass']}")


def stage_gate5(args) -> None:
    from apexzero.fields import gates as G

    refs = _gate1_refs(args)
    with stage_log(args, "gate5", len(refs)) as log:
        res = G.gate5(refs, progress=lambda n: log.step(n))
    _write(_out(args, "gates") / "gate5.json", res)
    for r in res["rows"]:
        print(f"Gate 5 {r['key_id']} ({r['kind']}, W {r['W']}): err {r['err']}, state "
              f"{r['state_tangent_err']:.1e}, pass {r['pass']}")


def stage_controls(args) -> None:
    from apexzero.fields import controls as C

    which = tuple(args.which)
    cells = C.QUICK_CELLS if args.quick else None
    with stage_log(args, "controls", len(which)) as log:
        res = {}
        for name in which:
            res.update({k: v for k, v in C.run_all((name,), cells).items() if k != "pass"})
            log.step(note=name)
    res["pass"] = {k: bool(v["pass"]) for k, v in res.items()}
    _write(_out(args, "controls") / "controls.json", C.jsonable(res))
    print(f"controls: {res['pass']}")


def _opt_chunk(items) -> list[dict]:
    """``stages.opt_fields`` on [(arrays dir, key), ...] (one map_keys chunk)."""
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S

    bank, _ = K.design()
    return [row for out_dir, ref in items for row in S.opt_fields([ref], Path(out_dir), bank)]


def stage_opt_fields(args) -> None:
    """The optimum's fields per key (pre-tag; ``stages.opt_fields_key``)."""
    from apexzero.fields import gates as G
    from apexzero.fields import stages as S

    out = _out(args, "opt-fields")
    refs = _limit(_refs(args.kinds), args.limit)
    with stage_log(args, "opt-fields", len(refs)) as log:
        items = [(str(out / "arrays"), r) for r in refs]
        rows = G.map_keys(_opt_chunk, items, args.workers, chunk=10,
                          progress=lambda n: log.step(n))
    S.write_rows(out / "rows.jsonl", rows)
    bad = [r["key_id"] for r in rows if r.get("failed")]
    summ = {"n": len(rows), "n_failed": len(bad), "failed": bad,
            "n_outside_gate4": sum(not r.get("in_band", False) for r in rows
                                   if r["kind"] != "anchored" and not r.get("failed")),
            "n_inertia_failed": sum(bool(r.get("inertia_failed")) for r in rows),
            "n_adjoint_failed": sum(r.get("adjoint_ok") is False for r in rows),
            "n_off_plateau": sum(not r.get("on_plateau", False) for r in rows
                                 if not r.get("failed") and r["valid"]["c_min"])}
    _write(out / "summary.json", summ)
    print(f"opt-fields: {summ}")


def stage_calibrate(args) -> None:
    """Branch consistency's value tolerance (``network.branch_tolerance``; design Section 14),
    derived before the tag without running a trained network's free loop: the controls'
    float32/float64 outcome spread per unit of teacher-forced action error, times the trained
    agents' largest teacher-forced error (``gate3 --all`` on every agent, read from
    gates/gate3.json), times BRANCH_FACTOR. Writes calibration/branch_tol.json; the value
    is then fixed in ``network.BRANCH_TOL`` before the tag."""
    import jax
    import numpy as np

    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.interp import rollout

    if jax.config.jax_enable_x64:
        raise SystemExit("calibrate builds float32 references: x64 must be off in this process")
    g3_path = _out(args, "gates") / "gate3.json"
    if not g3_path.exists():
        raise SystemExit("run `fields.py gate3 --all --agent ...` on every agent first")
    g3 = json.loads(g3_path.read_text())
    per_agent = {a: max((r["mean_err"] for r in v["rows"] if r.get("mean_err") is not None),
                        default=float("nan")) for a, v in g3["agents"].items()}
    from apexzero.interp import agents as A

    missing = sorted(set(A.AGENTS) - set(per_agent))
    out = _out(args, "calibration")
    with stage_log(args, "calibrate", 4) as log:
        bank, cells = K.design()
        g1 = N._final_agent("G1")
        starts = [(cells[c], k) for c, k in CAL_RANDOM_STARTS]
        det = N.float32_reference(rollout.agent_policy(g1.cfg), N.random_actor_params(g1.cfg, 0),
                                  g1.obs_rms, bank, starts, 200)
        N.save_det(out / "random.npz", det, {"policy": "random", "seed": 0, "loop_steps": 200})
        starts = [(cells[c], 0) for c in CAL_LINEAR_CELLS]
        Ws, bs = zip(*(N.linear_driver(bank.stimuli[c.stim_id].direction) for c, _ in starts),
                     strict=True)
        det = N.float32_reference(N.linear_policy, (np.stack(Ws), np.stack(bs)), None, bank,
                                  starts, CAL_LINEAR_STEPS, aux=list(range(len(starts))))
        det["lin_W"], det["lin_b"] = np.stack(Ws), np.stack(bs)
        N.save_det(out / "linear.npz", det, {"policy": "linear_driver",
                                              "loop_steps": CAL_LINEAR_STEPS})
        starts = [(cells[c], k) for c, k in CAL_KEEPER_STARTS]
        det = N.float32_reference(rollout._lane_keeper, N.KEEPER_PARAMS, None, bank, starts, 300)
        N.save_det(out / "keeper.npz", det, {"policy": "lane_keeper", "loop_steps": 300})
        log.step(note="float32 references")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]))
        cmds = {"linear": ["--f32", str(out / "linear.npz")],
                "random": ["--f32", str(out / "random.npz"), "--keeper", str(out / "keeper.npz")]}
        procs = {k: subprocess.Popen([sys.executable, "-m", "apexzero.fields.network", "dry-run",
                                      k, *a, "--out", str(out / k)], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for k, a in cmds.items()}
        reps = {}
        for k, p in procs.items():
            txt, _ = p.communicate(timeout=3600)
            if p.returncode != 0:
                raise SystemExit(f"dry-run {k} failed:\n{txt[-4000:]}")
            reps[k] = json.loads((out / f"{k}.json").read_text())
        log.step(note="float64 dry runs")
        rows = [r for r in reps["linear"]["rows"] + reps["random"].get("keeper", [])]
        control_rows = [{"source": r["source"], "values": r["branch"]["values"],
                         "mean_err": r["wiring"]["mean_err"], "branch_ok": r["branch"]["ok"]}
                        for r in rows]
        trained = max(v for v in per_agent.values() if np.isfinite(v))
        tol = N.branch_tolerance(control_rows, trained)
        log.step(note="tolerance")
        res = {"tolerance": tol["tol"], "factor": tol["factor"], "trained_mean_err": trained,
               "trained_mean_err_by_agent": per_agent, "agents_missing_from_gate3": missing,
               "gate3_rows": {a: len(v["rows"]) for a, v in g3["agents"].items()},
               "control_rows": control_rows,
               "slopes": tol["slopes"], "provisional_BRANCH_TOL": dict(N.BRANCH_TOL),
               "controls_branch_ok": all(r["branch_ok"] for r in control_rows)}
        _write(out / "branch_tol.json", res)
        log.step(note="written")
    print(f"calibrate: tolerance {tol['tol']} (trained max teacher-forced error {trained:.2e}; "
          f"{len(control_rows)} control rows; missing from gate3: {missing})")


def stage_phase_ceiling(args) -> None:
    from apexzero.fields import stages as S

    src = Path(args.out) / "opt-fields"
    rows = S.read_rows(src / "rows.jsonl")
    res = S.phase_ceiling_rows(rows, src / "arrays")
    _write(_out(args, "phase-ceiling") / "phase_ceiling.json", res)
    print(f"phase ceiling on {len(res)} cells")


def stage_prereg(args) -> None:
    """The pre-registration record (``fields.prereg``): the last step before the tag. It
    needs the re-polish rows of every key."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg
    from apexzero.interp import agents, io, stimuli

    rows_path = Path(args.out) / "repolish" / "repolish.jsonl"
    if not rows_path.exists():
        raise SystemExit(f"no re-polish rows at {rows_path}: run the repolish stage first")
    rows = [json.loads(ln) for ln in rows_path.read_text().splitlines() if ln.strip()]
    sets = prereg.key_sets()
    bank, cells = stimuli.load_design(io.STIMULI_DIR)
    ft = {a: K.final_timesteps(a) for a in agents.AGENTS
          if agents.final_snapshot(agents.AGENTS[a]) is not None}
    rec = prereg.build_record(ROOT, sets, io.KEYS_DIR, rows, stimuli.design_hash(bank, cells),
                              ft, prereg.tolerances())
    if rec["repolish_missing"] and not args.force:
        raise SystemExit(f"{len(rec['repolish_missing'])} listed keys have no re-polish row "
                         "(--force writes the record anyway)")
    path = prereg.save(rec)
    print(f"wrote {path}: {rec['key_counts']}")


# ------------------------------------------------------------------------- post-tag stages


def _post_tag(args, what: str) -> None:
    """The freeze check, then float64 and trained networks' fields allowed."""
    require_tag(what, getattr(args, "post_tag_deviation", None), Path(args.out))
    _x64()
    from apexzero.fields import network as N

    N.unlock_trained()


def _frozen_refs(names: tuple[str, ...]):
    """The tagged record's key lists (never a fresh glob), as KeyRefs."""
    from apexzero.fields import keys as K
    from apexzero.fields import prereg

    rec = tagged_record()
    if rec is None:
        raise SystemExit("no pre-registration record at the tag")
    want = {tuple(t) for name in names for t in prereg.frozen_ids(rec, name)}
    by = {(r.kind, r.key_id): r for kind in K.KINDS for r in K.key_index(kind)}
    missing = sorted(want - set(by))
    if missing:
        raise SystemExit(f"{len(missing)} tagged keys are not in the key store: {missing[:3]}")
    return [by[t] for t in sorted(want)]


def _det_of(agent_id: str | None, own: bool = False):
    """det groups of stored rollouts, loaded on first use: of each anchored key's own
    snapshot (``own`` False), or of ``agent_id``'s final snapshot for every key (``own``
    True: A2 keys, the network's own runs). None for a network without step 1 rollouts."""
    from apexzero.fields import keys as K
    from apexzero.fields import network as N
    from apexzero.interp import io

    cache: dict = {}

    def get(ref):
        a = agent_id if own else (ref.agent or agent_id)
        if a is None or a not in K.agents.AGENTS:
            return None
        ts = K.final_timesteps(a) if own or not ref.snapshot else int(ref.snapshot)
        path = io.rollout_path(a, ts)
        if not path.exists():
            return None
        if (a, ts) not in cache:
            cache[(a, ts)] = N.load_det(path)
        return cache[(a, ts)]

    return get


def _networks(args, default) -> list[str]:
    return list(args.network or default)


def stage_fields_a2(args) -> None:
    _post_tag(args, "the networks' fields on the A2 keys")
    from apexzero.fields import analysis as A
    from apexzero.fields import stages as S

    refs = _limit(_frozen_refs(("a2",)), args.limit)
    names = _networks(args, list(A.H2_NETWORKS)
                      + [r for d in A.H4_VARIANTS.values() for r in d["real"] + d["baseline"]]
                      + list(A.H4_TRANSFER["real"]))
    out = _out(args, "fields-a2")
    with stage_log(args, "fields-a2", len(refs) * len(names)) as log:
        for name in dict.fromkeys(names):
            net = S.load_network(name)
            rows = S.network_fields(net, refs, out / "arrays", det_of=_det_of(name, own=True),
                                    progress=lambda n: log.step(n))
            S.write_rows(out / f"{name.replace('/', '_')}.jsonl", rows)


def stage_fields_anchored(args) -> None:
    _post_tag(args, "the networks' fields on the anchored finals")
    from apexzero.fields import stages as S

    agents_ = args.agent or None
    out = _out(args, "fields-anchored")
    rec = tagged_record()
    sets = [k for k in rec["key_sets"] if k.startswith("finals:")
            and (agents_ is None or k.split(":", 1)[1] in agents_)]
    for name in sets:
        agent = name.split(":", 1)[1]
        refs = _limit(_frozen_refs((name,)), args.limit)
        net = S.load_network(agent)
        with stage_log(args, f"fields-anchored-{agent}", len(refs)) as log:
            rows = S.network_fields(net, refs, out / "arrays", det_of=_det_of(agent),
                                    blind=False, progress=lambda n: log.step(n))
        S.write_rows(out / f"{agent}.jsonl", rows)


def stage_dose(args) -> None:
    _post_tag(args, "the dose-response subsample")
    from apexzero.fields import gates as G
    from apexzero.fields import keys as K
    from apexzero.fields import kkt as KK
    from apexzero.fields import network as N
    from apexzero.fields import stages as S

    out = _out(args, "dose")
    refs = _limit(S.dose_sample(), args.limit)
    bank, cells = K.design()
    with stage_log(args, "dose", len(refs)) as log:
        opt = []
        for ref in refs:
            try:
                prob, stim, _ = G.problem(ref, bank)
                opt.append(KK.dose_optimum(prob, stim.landmarks, int(stim.direction),
                                           K.decision_step(prob.rec)))
            except Exception as e:  # noqa: BLE001  (recorded; the key is a numerical exclusion)
                opt.append({"key_id": ref.key_id, "failed": True, "error": str(e)})
            log.step()
    _write(out / "optimum.json", opt)
    nets = {r.agent or "G2" for r in refs if r.kind == "anchored"} | set(args.network or ["G2"])
    res = {}
    for name in sorted(nets):
        net = S.load_network(name)
        det = _det_of(name)
        rows = []
        for ref in refs:
            if ref.kind == "anchored" and ref.agent != name:
                continue
            rec = K.load_key(ref)
            st = S.network_start(ref, rec, bank, cells, det(ref) if ref.kind == "anchored"
                                 else None)
            rd = N.road(bank, st.stim_id)
            rows.append({"key_id": ref.key_id, **S.dose_network(net, rd, st)})
        res[name] = rows
    _write(out / "networks.json", res)


def stage_h3_keys(args) -> None:
    _post_tag(args, "H3's key solves")
    from apexzero.fields import stages as S

    cache = S.h3_cache(_out(args, "h3-keys") / "keys")
    base = S.h3_base_keys(cache)
    anch = S.h3_anchored_keys(cache)
    _write(_out(args, "h3-keys") / "keys.json",
           {"base": [r.key_id for r in base], "anchored": [r.key_id for r in anch],
            "status": {r.key_id: r.status for r in base + anch}})


def stage_h3_lambda(args) -> None:
    _post_tag(args, "H3's lambda-grid rollouts")
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S

    net = S.load_network(A.H3_AGENT)
    _, cells = K.design()
    res = [S.h3_lambda(net, cid, k) for cid in A.H3_CELLS for k in range(cells[cid].n_phases)]
    _write(_out(args, "h3-lambda") / "h3.json", res)


def stage_analyse(args) -> None:
    _post_tag(args, "the analysis")
    from apexzero.fields import analysis as A
    from apexzero.fields import keys as K
    from apexzero.fields import stages as S
    from apexzero.interp import io, support

    out = _out(args, "analyse")
    path = S.confirmatory_guard(out) if args.confirmatory else out / "decisions_dry.json"
    root = Path(args.out)
    opt = {r["key_id"]: r for r in S.read_rows(root / "opt-fields" / "rows.jsonl")}
    opt_dir = root / "opt-fields" / "arrays"
    net_rows, net_dirs = {}, {}
    for f in sorted((root / "fields-a2").glob("*.jsonl")):
        rows = S.read_rows(f)
        if rows:
            name = (rows[0]["net"].split("@")[0] if not f.stem.startswith("runs_")
                    else f.stem.replace("runs_", "runs/", 1))
            net_rows[name] = rows
            net_dirs[name] = (root / "fields-a2" / "arrays"
                              / rows[0]["net"].replace("/", "_").replace("@", "_"))
    res = S.analyse_edges(net_rows, opt)
    bank, cells = K.design()
    regimes = support.load_support(io.SUPPORT_DIR).regimes
    a2 = {r.cell_id for r in K.a2_primary_refs()}
    res["fields"] = {}
    for name in A.H2_NETWORKS:
        if name in net_rows:
            bulk = set(A.a2_cells_by_regime(name, bank, cells, regimes, a2).get("bulk", ()))
            res["fields"][name] = S.analyse_fields(net_rows[name], opt, net_dirs[name], opt_dir,
                                                   bulk)
    res["H1_fast"] = {}
    for f in sorted((root / "fields-anchored").glob("*.jsonl")):
        rows = S.read_rows(f)
        if rows:
            d = root / "fields-anchored" / "arrays" / rows[0]["net"].replace("/", "_").replace(
                "@", "_")
            res["H1_fast"][f.stem] = S.h1_fast_rows(rows, opt, d, opt_dir, f.stem)
    _write(path, res)


# ------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=2, help="XLA intra-op threads")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name in STAGES:
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(OUT), help="output root (default outputs/fields)")
        p.add_argument("--log-dir", default=str(ROOT / "runs"),
                       help="where fields-<stage>.log goes (default runs/)")
        p.add_argument("--workers", type=int, default=1, help="key worker processes")
        p.add_argument("--force", action="store_true", help="redo instead of resuming")
        if name in ("keys", "repolish", "opt-fields"):
            p.add_argument("--kinds", nargs="+", default=["a2", "finals", "earlier"],
                           choices=("a2", "finals", "earlier"))
        if name in ("keys", "repolish", "gate3", "gate4", "opt-fields", "fields-a2",
                    "fields-anchored", "dose"):
            p.add_argument("--limit", type=int, default=None,
                           help="this many keys, spread evenly over the list")
        if name == "keys":
            p.add_argument("--no-gate0", action="store_true", help="the lists only")
            p.add_argument("--no-audit", action="store_true", help="skip the loop audit")
        if name == "gate4":
            p.add_argument("--no-twins", action="store_true",
                           help="leave out the 6 tier V horizon twins (default: all 481)")
        if name in POST_TAG:
            p.add_argument("--post-tag-deviation", default=None, metavar="REASON",
                           help="log the freeze check's problems as a dated deviation and run")
        if name in ("fields-a2", "dose"):
            p.add_argument("--network", action="append", default=None,
                           help="network (registry id or runs/<dir>; repeatable)")
        if name in ("gate1", "gate5"):
            p.add_argument("--n", type=int, default=None,
                           help="the first n keys of the stratified sample (round-robin over "
                                "strata); default all 50")
        if name in ("gate3", "fields-anchored", "analyse"):
            p.add_argument("--agent", action="append", default=None,
                           help="agent id (repeatable); gate3 default G2")
        if name == "gate3":
            p.add_argument("--n-anchored", type=int, default=3)
            p.add_argument("--n-a2", type=int, default=2)
            p.add_argument("--all", action="store_true",
                           help="the clean check on every key with a det row (exclusions)")
        if name == "controls":
            p.add_argument("--which", nargs="+",
                           default=["RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers"],
                           choices=("RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers"))
            p.add_argument("--quick", action="store_true",
                           help="the multi-cell controls on fewer cells (controls.QUICK_CELLS)")
        if name == "gate2":
            p.add_argument("--random", nargs="*", type=int, default=[62, 345],
                           help="A2 cells of the random-init actor (steer)")
            p.add_argument("--keeper", nargs="*", type=int, default=[344],
                           help="cells of the planted edge keeper (plan-level outcomes)")
            p.add_argument("--linear", nargs="*", type=int, default=[344, 345],
                           help="cells of the planted linear driver (plan-level outcomes)")
        if name == "analyse":
            p.add_argument("--confirmatory", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage in PRE_TAG and args.stage not in X64_OFF:
        _x64()                       # post-tag stages switch it on after the freeze check
    t0 = time.time()
    globals()["stage_" + args.stage.replace("-", "_")](args)
    print(f"{args.stage}: {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
