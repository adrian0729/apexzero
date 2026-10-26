"""Paper labels (follow-through PA-1): J*_20 per paper level, resumable, in a worker pool.

    nice -n 15 uv run python scripts/label_levels.py run --stream a_dr --root 2026 --levels 0:3 \\
        --agent runs/dr-pilot-200M/ckpt/000200000000.pkl --workers 3 --out outputs/labels/pa1-e2e
    uv run python scripts/label_levels.py status --out outputs/labels/pa1-e2e
    uv run python scripts/label_levels.py export-j-opt --out DIR --to DIR/j_opt.npy --n 150
    uv run python scripts/label_levels.py export-oracle --out DIR --to DIR/pool_jstar.npz \\
        --pool-file outputs/pool.npz

``run`` labels levels ``--levels`` (ranges ``a:b`` and single indices) of
``make_levels(PRNGKey(--root), --stream)``, optionally only shard ``--shard i/n`` (indices with
index % n == i, so shards on different machines are disjoint). The method, the retry policy,
the schema and the exports are ``apexzero.ocp.labels``'s. Every level writes
``<out>/records/<id>.json`` and ``.npz`` when it is done; a run of the same ``--out`` skips
levels that have an outcome (verified or unsolved), labels again those that ended in an error,
and resumes an interrupted level from its cache (``<out>/work/<id>/``).

``<out>/run.json`` pins the stream, root, starts, policy, checkpoint hash and code hash of the
first run. A later run with any of them different is refused (a new ``--out`` is a new label
set); ``--allow-code-change`` lets the code hash differ, recorded in run.json and per record,
and the exports then refuse the mix unless told otherwise.

Workers are spawned processes, each with one thread for BLAS, MUMPS and XLA, so process CPU
time is the budget. The agent start needs ``--agent`` (a checkpoint); without it, pass
``--starts socp qss`` explicitly. No lockbox circuit is read: levels are generated from the
grammar fitted on R-dev.

``status`` prints the outcome counts, CPU per level and every start's verification rate, then
one row per level. ``export-j-opt`` writes the .npy ``scripts/pilot_audit.py --j-opt`` reads
(levels 0..n-1 of a_dr at root 2026 by default, pilot_audit's own); ``export-oracle`` writes
``ued.pool.save_oracle_labels``'s .npz for ``scripts/train_ued.py --labels``.
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # single-threaded MUMPS / BLAS: CPU time is the budget
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax

jax.config.update("jax_enable_x64", True)   # before apexzero loads: float64 guesses and replays

import numpy as np

from apexzero.ocp import labels as L

PINNED = ("schema", "stream", "root", "starts", "policy", "agent_sha256")


# ----------------------------------------------------------------------------- arguments


def parse_levels(specs: list[str]) -> list[int]:
    out: set[int] = set()
    for spec in specs:
        if ":" in spec:
            a, b = spec.split(":")
            out |= set(range(int(a), int(b)))
        else:
            out.add(int(spec))
    return sorted(out)


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    i, n = spec.split("/")
    return int(i), int(n)


def policy_from_args(args) -> L.LabelPolicy:
    base = L.LabelPolicy()
    kw = dict(seconds=args.seconds, replicates=args.replicates, agent_windows=args.agent_windows,
              retry=not args.no_retry)
    for name, attr in (("max_iter", "max_iter"), ("start_budget", "start_cpu_budget_s"),
                       ("retry_max_iter", "retry_max_iter"), ("retry_budget", "retry_cpu_budget_s"),
                       ("call_cap", "call_cpu_cap_s")):
        if getattr(args, name) is not None:
            kw[attr] = getattr(args, name)
    if args.stages:
        kw["stages_s"] = tuple(float(s) for s in args.stages)
    if args.retry_stages:
        kw["retry_stages_s"] = tuple(float(s) for s in args.retry_stages)
    return dataclasses.replace(base, **kw)


# ----------------------------------------------------------------------------- workers

_W: dict = {}


def _init(cfg: dict) -> None:
    """Per process: the grammar, the solvers (built on first use), the checkpoint."""
    _W["cfg"] = cfg
    _W["grammar"] = L.r_dev_grammar()
    _W["solvers"] = L.SolverCache(radius=cfg["policy"].radius, compile=not cfg["no_compile"])
    p = cfg["policy"]
    _W["agent"] = (L.AgentStart(cfg["agent"], p.agent_windows, p.agent_obs_noise, p.agent_seed)
                   if cfg["agent"] else None)


def _label(index: int) -> dict:
    cfg = _W["cfg"]
    out = Path(cfg["out"])
    c0 = time.process_time()
    level = L.make_paper_levels(cfg["stream"], cfg["root"], [index], _W["grammar"])[0]

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')} pid {os.getpid()}] {msg}", flush=True)

    try:
        lab = L.label_level(level, cfg["policy"], _W["solvers"], _W["agent"], cfg["starts"],
                            work=out / "work" / level.id, code=cfg["code"], log=log)
    except Exception as exc:  # noqa: BLE001  (an error is a record; the next run relabels it)
        lab = L.error_record(level, exc, cfg["policy"], cfg["code"], time.process_time() - c0)
        log(f"{level.id}: ERROR {type(exc).__name__}: {exc}")
    L.write_record(out, lab)
    rec = lab.record
    cpu = rec["cpu"]
    return {"id": level.id, "status": rec["label"]["status"], "j_star": rec["label"].get("j_star"),
            "start": rec["label"].get("start"), "cpu_s": cpu.get("level_s", cpu["level_process_s"]),
            "wall_s": cpu.get("wall_s")}


# ----------------------------------------------------------------------------- commands


def check_run_json(out: Path, meta: dict, allow_code_change: bool) -> None:
    path = out / "run.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=1))
        return
    old = json.loads(path.read_text())
    diff = [k for k in PINNED if old.get(k) != meta.get(k)]
    if diff:
        raise SystemExit(f"{path} was made with other {', '.join(diff)}: a different label set "
                         f"needs its own --out")
    if old["code"]["sha256"] != meta["code"]["sha256"]:
        if not allow_code_change:
            raise SystemExit(f"{path}: the code hash changed ({old['code']['sha256'][:12]} -> "
                             f"{meta['code']['sha256'][:12]}); files: "
                             + ", ".join(f for f in sorted(set(old['code']['files']) | set(meta['code']['files']))
                                         if old['code']['files'].get(f) != meta['code']['files'].get(f))
                             + ". Use a new --out, or --allow-code-change to mix (recorded).")
        old.setdefault("code_changes", []).append({"at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                   "code": meta["code"]})
        path.write_text(json.dumps(old, indent=1))


def cmd_run(args) -> None:
    out = Path(args.out)
    starts = L._primary_order(args.starts)
    if "agent" in starts and not args.agent:
        raise SystemExit("the agent start needs --agent <checkpoint> (or pass --starts socp qss)")
    policy = policy_from_args(args)
    indices = L.select_shard(parse_levels(args.levels), parse_shard(args.shard))
    code = L.code_identity()
    agent_sha = L.AgentStart(args.agent).sha256 if "agent" in starts else None
    meta = {"schema": L.SCHEMA, "stream": args.stream, "root": args.root, "starts": starts,
            "policy": policy.to_json(), "agent_sha256": agent_sha,
            "agent": None if not args.agent else Path(args.agent).name, "code": code,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    check_run_json(out, meta, args.allow_code_change)
    ids = {i: L.level_id(args.stream, args.root, i) for i in indices}
    todo = [i for i in indices if not L.is_done(L.load_record(out, ids[i]))]
    print(f"{len(indices)} levels selected, {len(indices) - len(todo)} done, {len(todo)} to label; "
          f"code {code['sha256'][:12]} at {str(code['git_commit'])[:12]}"
          f"{' (dirty: ' + ', '.join(code['git_dirty']) + ')' if code['git_dirty'] else ''}; "
          f"sim-v1 {code['sim_v1'] or 'not tagged'}", flush=True)
    cfg = {"out": str(out), "stream": args.stream, "root": args.root, "starts": starts, "policy": policy,
           "agent": args.agent if "agent" in starts else None, "code": code, "no_compile": args.no_compile}
    t0 = time.perf_counter()
    workers = max(1, min(args.workers, len(todo)))
    lost = []           # levels whose worker died (no record): the next run labels them
    if workers == 1 and todo:
        _init(cfg)
        for i in todo:
            print(json.dumps(_label(i)), flush=True)
    elif todo:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init,
                                 initargs=(cfg,)) as ex:
            futs = {ex.submit(_label, i): i for i in todo}
            for f in as_completed(futs):
                try:
                    print(json.dumps(f.result()), flush=True)
                except Exception as exc:  # noqa: BLE001  (a dead worker: BrokenProcessPool)
                    lost.append(futs[f])
                    print(f"{ids[futs[f]]}: no record ({type(exc).__name__}: {exc})", flush=True)
    recs = [r for r in (L.load_record(out, ids[i]) for i in indices) if r is not None]
    print(json.dumps({"wall_s": time.perf_counter() - t0, "lost": [ids[i] for i in sorted(lost)],
                      **L.summarise(recs)}, indent=1))
    if lost or any(r["label"]["status"] == "error" for r in recs):
        sys.exit(1)     # errors and lost levels are relabelled by the next run of this --out


def _fmt(v, f="{:.5f}"):
    return "-" if v is None else f.format(v)


def cmd_status(args) -> None:
    recs = list(L.iter_records(Path(args.out)))
    print(json.dumps(L.summarise(recs), indent=1))
    print("| Level | Status | J* | From | Verified (primary) | Retried | Spread | Agent best survivor "
          "| s-OCP | Level CPU | Solves | Wall |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        lab, cpu = r["label"], r.get("cpu", {})
        socp = r.get("socp") or {}
        print(f"| {r['level']['id']} | {lab['status']} | {_fmt(lab.get('j_star'))} | {lab.get('start') or '-'} | "
              f"{lab.get('n_verified_primary', '-')}/{lab.get('n_primary', '-')} | {lab.get('retried', '-')} | "
              f"{_fmt(lab.get('primary_spread'), '{:.2e}')} | {_fmt(lab.get('j_agent_survivor_best'))} | "
              f"{socp.get('status', '-')} | {_fmt(cpu.get('level_s', cpu.get('level_process_s')), '{:.0f} s')} | "
              f"{_fmt(cpu.get('solves_s'), '{:.0f} s')} | {_fmt(cpu.get('wall_s'), '{:.0f} s')} |")


def cmd_export_j_opt(args) -> None:
    try:
        js = L.export_j_opt(Path(args.out), Path(args.to), args.n, args.stream, args.root,
                            args.allow_mixed_code)
    except L.LabelError as e:
        raise SystemExit(f"not exported: {e}") from e
    print(f"wrote {args.to} ({len(js)} labels, J* {js.min():.3f}-{js.max():.3f}) and {args.to}.json")


def cmd_export_oracle(args) -> None:
    from apexzero.tracks import levels as paper_levels

    if args.pool_file:
        lv = paper_levels.load(args.pool_file)
    else:
        lv = L.float32_levels("pool", args.root, args.n or paper_levels.SIZES["pool"])
    try:
        js = L.export_oracle(Path(args.out), Path(args.to), lv, "pool", args.root, args.allow_mixed_code)
    except L.LabelError as e:
        raise SystemExit(f"not exported: {e}") from e
    print(f"wrote {args.to} ({len(js)} labels) and {args.to}.json")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="label levels (resumable)")
    r.add_argument("--stream", default="a_dr")
    r.add_argument("--root", type=int, default=2026, help="PRNGKey of the level set (pilot_audit's: 2026)")
    r.add_argument("--levels", nargs="+", required=True, help="indices: a:b ranges and single values")
    r.add_argument("--shard", default=None, help="i/n: only indices with index %% n == i")
    r.add_argument("--out", required=True)
    r.add_argument("--agent", default=None, help="checkpoint of the agent start (agents.checkpoints)")
    r.add_argument("--starts", nargs="+", default=list(L.STARTS), choices=L.STARTS)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seconds", type=float, default=L.LabelPolicy.seconds)
    r.add_argument("--stages", type=float, nargs="+", default=None, help="continuation stages (s)")
    r.add_argument("--replicates", type=int, default=0, help="extra rounds of the starts (label noise)")
    r.add_argument("--agent-windows", type=int, default=L.LabelPolicy.agent_windows)
    r.add_argument("--max-iter", type=int, default=None)
    r.add_argument("--start-budget", type=float, default=None, help="CPU-s per start")
    r.add_argument("--call-cap", type=float, default=None, help="CPU-s per IPOPT call")
    r.add_argument("--no-retry", action="store_true")
    r.add_argument("--retry-stages", type=float, nargs="+", default=None)
    r.add_argument("--retry-max-iter", type=int, default=None)
    r.add_argument("--retry-budget", type=float, default=None, help="CPU-s per retried start")
    r.add_argument("--allow-code-change", action="store_true")
    r.add_argument("--no-compile", action="store_true", help="CasADi's VM instead of compiled C")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("status", help="outcomes, CPU and one row per level")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_status)

    j = sub.add_parser("export-j-opt", help="labels.npy for pilot_audit.py --j-opt")
    j.add_argument("--out", required=True)
    j.add_argument("--to", required=True)
    j.add_argument("--n", type=int, required=True, help="pilot_audit's --levels")
    j.add_argument("--stream", default="a_dr")
    j.add_argument("--root", type=int, default=2026)
    j.add_argument("--allow-mixed-code", action="store_true")
    j.set_defaults(fn=cmd_export_j_opt)

    o = sub.add_parser("export-oracle", help="pool J* .npz for train_ued.py --labels")
    o.add_argument("--out", required=True)
    o.add_argument("--to", required=True)
    o.add_argument("--pool-file", default=None, help="levels.save file of the pool (else --root/--n)")
    o.add_argument("--root", type=int, default=0)
    o.add_argument("--n", type=int, default=None)
    o.add_argument("--allow-mixed-code", action="store_true")
    o.set_defaults(fn=cmd_export_oracle)

    args = ap.parse_args(argv)
    np.set_printoptions(precision=6)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
