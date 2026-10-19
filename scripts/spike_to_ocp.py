"""K2a spike: how fast does the training-objective OCP solve, and does it replay?

    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 10 20 60 --polish
    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --s0 1000 2500 --seconds 60 --first ocp
    uv run python scripts/spike_to_ocp.py --tracks --levels pool:0 pool:1 --seconds 60

Cases: circuits (``--tracks``, refined reference, x0 at each ``--s0`` with n = 0, xi = 0,
v = 0.8 v_QSS(s0)) and production-like levels (``--levels pool:i``: level i of a paper
level set, ``tracks.levels``, keys from PRNGKey(``--level-root``): a generated 8 km strip,
a random car context and a hidden grip multiplier, x0 as every paper level's).

Method (default ``continuation``): the first 10 s are solved from the first guess (the
trained policy's rollout, or the s-domain optimum mapped to time with ``--first ocp``),
then each longer window (20, 40, 60, ... s) from the previous optimum, continued by the
s-domain minimum-time optimum. ``--first policy ocp`` runs one such chain per first guess
(independent starts, as the label plan's multi-start budget) and adds a "best of all
starts" row. ``--polish`` adds a start for the target window: solve the next longer window
from the first chain's optimum and re-solve the target from it. ``--method direct`` is one
solve from the first guess (the first spike's method, kept for comparison).

Continuation cannot run without the s-domain optimum, so its cost is part of a level's:
on a level it is solved here (on the part of the strip the longest window can reach,
``window.reachable_strip``; ``--socp-whole`` for the whole strip) and its CPU time is
reported next to the TO-OCP's. A circuit's lap s-OCP serves every window on it; it is
loaded from outputs/ocp/<track>.pkl when cached (its original solve's wall time is then
reported instead) unless ``--solve-ocp``.

Every start's optimum is replayed open-loop in the float64 JAX environment (termination
checks, hidden grip included); a row reports the best *verified* optimum
(``replay.best_verified``: solve succeeded, replay survives and reproduces J* within
1e-6), or none. CPU is process time, single-threaded; compiling the NLP functions (once
per window length, then cached) and the guesses are reported separately. Pickles and a
markdown table go to --out.
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # single-threaded MUMPS / BLAS: CPU time is the budget

import argparse
import pickle
import resource
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.env.adapters import reference_speed
from apexzero.ocp.min_time import solve_min_time
from apexzero.ocp.replay import best_verified
from apexzero.ocp.window import (
    WindowSolver,
    WindowSpec,
    min_time_guess,
    reachable_strip,
    solve_continuation,
    window_return,
)
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import track_arrays

jax.config.update("jax_enable_x64", True)
STAGES = (10, 20, 40, 60, 80, 100, 120)     # s; continuation schedule


def policy_rollout(track, x0, steps, agent_path, ctx, hidden_mu):
    """Dynamically consistent warm start: the trained policy driven from x0."""
    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd

    with open(agent_path, "rb") as fh:
        d = pickle.load(fh)
    actor = Actor(PPOConfig(**d["config"]), 2, False)
    rms = RunningMeanStd(**{k: jnp.asarray(v) for k, v in d["obs_rms"].items()})
    tr = track_arrays(track, dtype=jnp.float64)
    params = racing.EnvParams(window_steps=10**9)
    st = racing.EnvState(x=jnp.asarray(x0), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=jnp.asarray(ctx, jnp.float64), hidden_mu=jnp.asarray(hidden_mu, jnp.float64),
                         t=jnp.array(0), slow=jnp.array(0.0), s0=jnp.asarray(x0[0]))
    obs = racing.observe(st, tr, racing.params_from_context(st.ctx, hidden_mu=st.hidden_mu))

    def body(c, _):
        st, ob = c
        a = jnp.clip(actor.apply(d["actor"], rms.normalize(ob.astype(jnp.float32)))[0], -1, 1).astype(jnp.float64)
        ob2, st2, _, _, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, ob2), (st2.x, a)

    _, (xs, acts) = jax.lax.scan(body, (st, obs), None, steps)
    return np.vstack([x0, np.asarray(xs)]), np.asarray(acts)


def steps(sec: float) -> int:
    return round(sec / 0.05)


def next_stage(sec: float) -> float:
    """The default polish window: the next longer stage (or 20 s more past the schedule)."""
    return next((s for s in STAGES if s > sec), sec + 20.0)


def polish_windows(sec: float, polish_to: float | None) -> list[float]:
    """The longer windows a polish start solves, in order, before re-solving ``sec``: up
    the continuation schedule to ``polish_to`` (default: just the next stage)."""
    top = polish_to if polish_to is not None and polish_to > sec else next_stage(sec)
    return sorted({s for s in STAGES if sec < s < top} | {top})


def timed_min_time(track, p):
    c0, t0 = time.process_time(), time.perf_counter()
    ocp = solve_min_time(track, p)
    return ocp, dict(cpu_s=time.process_time() - c0, wall_s=time.perf_counter() - t0,
                     note=f"s-OCP on {track.length:.0f} m: {ocp.status}, {ocp.iterations} it")


def circuit_cases(names, s0s, solve_ocp: bool):
    """Windows on the circuits; the lap s-OCP (cached or solved once) serves all of them."""
    for name in names:
        track = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
        v_ref = reference_speed(track)
        ocp_path = Path("outputs/ocp") / f"{name}.pkl"
        if ocp_path.exists() and not solve_ocp:
            with open(ocp_path, "rb") as fh:
                ocp = pickle.load(fh)
            socp = dict(cpu_s=None, note=f"cached lap s-OCP ({ocp.solve_seconds:.0f} s wall when solved)")
        else:
            ocp, socp = timed_min_time(track, F1_2025)
            socp["note"] = "lap " + socp["note"] + " (shared by every window on the circuit)"
            print(f"{name}: {socp['note']}, cpu={socp['cpu_s']:.1f}s", flush=True)
        for s0 in s0s:
            x0 = np.array([s0, 0.0, 0.0, 0.8 * float(np.interp(s0, track.s, v_ref)), 0.0, 0.0, 0.0, 0.0])
            label = name if s0 == 0 else f"{name} s={s0:.0f}"
            yield dict(name=label, track=track, x0=x0, ctx=np.asarray(racing.default_context()),
                       hidden_mu=1.0, p=F1_2025, ocp=ocp, socp=socp, first="policy")


def level_cases(specs, root: int):
    from apexzero.tracks.corners import R_DEV
    from apexzero.tracks.generator import fit_grammar, to_track
    from apexzero.tracks.levels import build, make_levels

    g = fit_grammar([load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV])
    for spec in specs:
        stream, idx = spec.split(":")
        idx = int(idx)
        lv = make_levels(jax.random.PRNGKey(root), stream, n=idx + 1)
        tr_arr, _, x0 = build(lv, g)
        track = to_track(jax.tree.map(lambda a, i=idx: a[i], tr_arr), name=f"{stream}-{idx}")
        ctx, mu = np.asarray(lv.ctx[idx], float), float(lv.hidden_mu[idx])
        p = racing.params_from_context(jnp.asarray(ctx), hidden_mu=mu)
        p = p._replace(**{k: float(getattr(p, k)) for k in p._fields})
        yield dict(name=f"{stream}:{idx}", track=track, x0=np.asarray(x0[idx], float), ctx=ctx, hidden_mu=mu,
                   p=p, ocp=None, socp=None, first="ocp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", default=["Hockenheim"])
    ap.add_argument("--s0", type=float, nargs="+", default=[0.0], help="window starts on the circuits (m)")
    ap.add_argument("--levels", nargs="*", default=[], help="paper levels, e.g. pool:0 a_dr:3")
    ap.add_argument("--level-root", type=int, default=0, help="PRNGKey of the level sets")
    ap.add_argument("--seconds", type=float, nargs="+", default=[10, 20, 60])
    ap.add_argument("--method", choices=["continuation", "direct"], default="continuation")
    ap.add_argument("--polish", action="store_true", help="second start from the next longer window")
    ap.add_argument("--polish-to", type=float, default=None,
                    help="polish from this longer window (s), continuing up the schedule to it")
    ap.add_argument("--first", choices=["policy", "ocp"], nargs="+", default=None,
                    help="first-stage guess of each start (default: policy on circuits, s-OCP on levels)")
    ap.add_argument("--agent", default="runs/hockenheim-specialist-mlp/agent.pkl")
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--overlap", type=int, default=20, help="steps of each optimum not reused")
    ap.add_argument("--solve-ocp", action="store_true", help="re-solve (and time) cached circuit s-OCPs")
    ap.add_argument("--socp-whole", action="store_true", help="levels: s-OCP on the whole strip")
    ap.add_argument("--no-compile", action="store_true", help="CasADi's VM instead of compiled C")
    ap.add_argument("--tag", default=None, help="file name tag (default: from the cases)")
    ap.add_argument("--out", default="outputs/to_ocp/speed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    solvers, compile_s = {}, {}

    def solver_for(T):
        if T not in solvers:
            t0 = time.perf_counter()
            solvers[T] = WindowSolver(WindowSpec(steps=T, radius=args.radius), compile=not args.no_compile)
            compile_s[T] = time.perf_counter() - t0
        return solvers[T]

    longest_s = max(args.seconds) if not args.polish else max(polish_windows(s, args.polish_to)[-1]
                                                              for s in args.seconds)
    cases = [*circuit_cases(args.tracks, args.s0, args.solve_ocp), *level_cases(args.levels, args.level_root)]
    rows = []
    for case in cases:
        name, track, x0, ctx, mu, p = (case[k] for k in ("name", "track", "x0", "ctx", "hidden_mu", "p"))
        ocp, socp, guess_track = case["ocp"], case["socp"], track
        if ocp is None:         # a level: its own strip and car, so its own s-OCP
            guess_track = track if args.socp_whole else reachable_strip(track, x0, longest_s, p)
            ocp, socp = timed_min_time(guess_track, p)
            socp["note"] += f" (strip {track.length:.0f} m)"
            print(f"{name}: {socp['note']}, cpu={socp['cpu_s']:.1f}s", flush=True)
        firsts = args.first or [case["first"]]
        t0 = time.perf_counter()
        guesses = {f: policy_rollout(track, x0, steps(longest_s), args.agent, ctx, mu) if f == "policy"
                   else min_time_guess(ocp, guess_track, x0, steps(longest_s)) for f in firsts}
        guess_s = time.perf_counter() - t0

        def extend(state, n, ocp=ocp, guess_track=guess_track):
            return min_time_guess(ocp, guess_track, state, n)

        for sec in args.seconds:
            H = steps(sec)
            sched = [H] if args.method == "direct" else [steps(s) for s in STAGES if s < sec] + [H]
            kw = dict(overlap=args.overlap, ctx=ctx, p=p, hidden_mu=mu)
            runs = {f: solve_continuation(solver_for, track, x0, sched, guesses[f], extend, **kw) for f in firsts}
            if args.polish:
                longer = [steps(s) for s in polish_windows(sec, args.polish_to)]
                runs["polish"] = solve_continuation(solver_for, track, x0, [*longer, H], None, extend,
                                                    previous=runs[firsts[0]][-1], **kw)
            spec = WindowSpec(steps=H)
            # One row per start alone; then (with --polish) the first start with its polish,
            # and (with several starts) all of them. A multi-start row reports the best
            # *verified* optimum of its starts and is charged with all of their cost.
            variants = [(args.method + (f" from {f}" if len(firsts) > 1 else ""), [f]) for f in firsts]
            if args.polish:
                variants.append((args.method + (f" from {firsts[0]}" if len(firsts) > 1 else "") + "+polish",
                                 [firsts[0], "polish"]))
            if len(firsts) > 1:
                variants.append(("best of all starts", list(runs)))
            for method, keys in variants:
                stages = [r for k in keys for r in runs[k]]
                best = best_verified([runs[k][-1] for k in keys], track, x0, ctx, mu)
                best_key = next((k for k in keys if runs[k][-1] is best), None)
                cpu = sum(r.meta["cpu_seconds"] for r in stages)
                rp = best.meta["replay"] if best is not None else {}
                row = dict(case=name, track=track.name, x0=x0.tolist(), ctx=ctx.tolist(), hidden_mu=mu,
                           seconds=sec, method=method,
                           stages=" ".join(f"{len(r.a) * 0.05:.0f}" for r in stages),
                           status=best.status if best is not None else "no verified optimum",
                           stage_status=[(len(r.a) * 0.05, r.status) for r in stages],
                           best_start=best_key,
                           J_first={f: window_return(g[0][: H + 1], g[1][:H], spec) for f, g in guesses.items()},
                           iterations=sum(r.iterations for r in stages), rounds=sum(r.rounds for r in stages),
                           wall_s=sum(r.solve_seconds for r in stages), cpu_s=cpu,
                           socp_cpu_s=socp["cpu_s"], socp_note=socp["note"],
                           level_cpu_s=None if socp["cpu_s"] is None else cpu + socp["cpu_s"],
                           J=best.value if best is not None else float("nan"),
                           J_starts={k: runs[k][-1].value for k in keys},
                           replays={k: runs[k][-1].meta.get("replay") for k in keys},
                           J_check=window_return(best.x, best.a, spec) if best is not None else float("nan"),
                           J_replay=rp.get("return", float("nan")), replay_rel=rp.get("rel", float("nan")),
                           replay_survives=rp.get("survives", False), verified=best is not None,
                           max_state_dev=rp.get("max_state_dev", float("nan")),
                           peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                           guess_s=guess_s,
                           stage_log=[dict(seconds=len(r.a) * 0.05, iterations=r.iterations, rounds=r.rounds,
                                           cpu_s=r.meta["cpu_seconds"], J=r.value, status=r.status,
                                           warm_start=r.meta.get("warm_start"))
                                      for r in stages])
                rows.append(row)
                print(f"{name} {sec:.0f}s [{method}; stages {row['stages']}]: {row['status']} "
                      f"it={row['iterations']} rounds={row['rounds']} wall={row['wall_s']:.1f}s "
                      f"cpu={cpu:.1f}s J*={row['J']:.5f} "
                      f"(starts {', '.join(f'{k} {v:.5f}' for k, v in row['J_starts'].items())}) "
                      f"replay {row['replay_rel']:+.1e} survives={row['replay_survives']}", flush=True)
                slug = (f"{args.tag}-" if args.tag else "") + name.replace(" ", "_").replace(":", "-").replace("=", "")
                with open(out / f"{slug}-{int(sec)}s-{method}.pkl", "wb") as fh:   # candidates: for diagnosis
                    pickle.dump({"result": best, "row": row, "candidates": {k: runs[k][-1] for k in keys},
                                 "stages": {k: runs[k][:-1] for k in keys}}, fh)
            for st in row["stage_log"]:
                print(f"    {st['seconds']:.0f}s: it={st['iterations']} rounds={st['rounds']} "
                      f"cpu={st['cpu_s']:.1f}s J*={st['J']:.5f} {st['status']} (from {st['warm_start']})",
                      flush=True)
        print(f"{name}: guess {guess_s:.1f}s; compile (cached after the first run) "
              + ", ".join(f"{T * 0.05:.0f}s window {v:.1f}s" for T, v in sorted(compile_s.items())), flush=True)

    lines = [("| Case | Window | Method | Stages (s) | Iter | Rounds | Solve wall | CPU | s-OCP CPU | "
              "Level CPU | J* | Replay diff | Survives |"), "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        socp = "cached" if r["socp_cpu_s"] is None else f"{r['socp_cpu_s']:.0f} s"
        level = "-" if r["level_cpu_s"] is None else f"{r['level_cpu_s']:.0f} s"
        lines.append(f"| {r['case']} | {r['seconds']:.0f} s | {r['method']} | {r['stages']} | {r['iterations']} | "
                     f"{r['rounds']} | {r['wall_s']:.0f} s | {r['cpu_s']:.0f} s | {socp} | {level} | "
                     f"{r['J']:.5f} | {r['replay_rel']:+.1e} | {r['replay_survives']} |")
    table = "\n".join(lines) + "\n"
    print(table)
    s0s = [] if args.s0 == [0.0] or not args.tracks else ["s" + "_".join(f"{s:.0f}" for s in args.s0)]
    tag = args.tag or "-".join([*args.tracks, *s0s, *args.levels]).replace(":", "")
    polish = ("-polish" + (f"{args.polish_to:.0f}" if args.polish_to else "")) if args.polish else ""
    tag += f"-{args.method}{polish}" + ("-" + "-".join(args.first) if args.first else "")
    (out / f"table-{tag}.md").write_text(table)
    with open(out / f"rows-{tag}.pkl", "wb") as fh:
        pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
