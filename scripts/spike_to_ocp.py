"""K2a spike: how fast does the training-objective OCP solve, and does it replay?

    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 10 20 60 --polish

Problem: x0 at s = 0 on the refined reference (n = 0, xi = 0, v = 0.8 v_QSS), the RL env's
own window problem (apexzero.ocp.window). Method (default ``continuation``): the first 10 s
are solved from the trained policy's rollout, then each longer window (20, 40, 60, ... s)
from the previous optimum, continued by the s-domain minimum-time optimum mapped to time
(outputs/ocp/<track>.pkl, solved if missing). ``--polish`` adds a second start for the
target window: solve the next longer window and re-solve the target from it; the better
optimum is kept. ``--method direct`` is one solve from the policy rollout (the first
spike's method, kept for comparison).

Every reported optimum's controls are replayed open-loop in the float64 JAX environment,
termination checks included. CPU is the process time of the IPOPT solves, single-threaded;
compiling the NLP functions (once per window length, then cached) and the guesses (a JAX
rollout) are reported separately. Pickles and a markdown table go to --out.
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
from apexzero.ocp.replay import replay_window
from apexzero.ocp.window import (
    WindowSolver,
    WindowSpec,
    min_time_guess,
    solve_continuation,
    window_return,
)
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import track_arrays

jax.config.update("jax_enable_x64", True)
STAGES = (10, 20, 40, 60, 80, 100, 120)     # s; continuation schedule


def policy_rollout(track, x0, steps, agent_path):
    """Dynamically consistent warm start: the trained policy driven from x0."""
    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd

    with open(agent_path, "rb") as fh:
        d = pickle.load(fh)
    actor = Actor(PPOConfig(**d["config"]), 2, False)
    rms = RunningMeanStd(**{k: jnp.asarray(v) for k, v in d["obs_rms"].items()})
    tr = track_arrays(track, dtype=jnp.float64)
    params = racing.EnvParams(window_steps=10**9)
    st = racing.EnvState(x=jnp.asarray(x0), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=racing.default_context().astype(jnp.float64),
                         hidden_mu=jnp.array(1.0), t=jnp.array(0),
                         slow=jnp.array(0.0), s0=jnp.asarray(x0[0]))
    obs = racing.observe(st, tr, racing.params_from_context(st.ctx))

    def body(c, _):
        st, ob = c
        a = jnp.clip(actor.apply(d["actor"], rms.normalize(ob.astype(jnp.float32)))[0], -1, 1).astype(jnp.float64)
        ob2, st2, _, _, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, ob2), (st2.x, a)

    _, (xs, acts) = jax.lax.scan(body, (st, obs), None, steps)
    return np.vstack([x0, np.asarray(xs)]), np.asarray(acts)


def steps(sec: float) -> int:
    return round(sec / 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["Hockenheim"])
    ap.add_argument("--seconds", type=float, nargs="+", default=[10, 20, 60])
    ap.add_argument("--method", choices=["continuation", "direct"], default="continuation")
    ap.add_argument("--polish", action="store_true", help="second start from the next longer window")
    ap.add_argument("--first", choices=["policy", "ocp"], default="policy", help="first-stage guess")
    ap.add_argument("--agent", default="runs/hockenheim-specialist-mlp/agent.pkl")
    ap.add_argument("--radius", type=float, default=60.0)
    ap.add_argument("--overlap", type=int, default=20, help="steps of each optimum not reused")
    ap.add_argument("--no-compile", action="store_true", help="CasADi's VM instead of compiled C")
    ap.add_argument("--out", default="outputs/to_ocp/speed")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = F1_2025
    solvers, compile_s = {}, {}

    def solver_for(T):
        if T not in solvers:
            t0 = time.perf_counter()
            solvers[T] = WindowSolver(WindowSpec(steps=T, radius=args.radius), compile=not args.no_compile)
            compile_s[T] = time.perf_counter() - t0
        return solvers[T]

    rows = []
    for name in args.tracks:
        track = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
        ocp_path = Path("outputs/ocp") / f"{name}.pkl"
        if ocp_path.exists():
            with open(ocp_path, "rb") as fh:
                ocp = pickle.load(fh)
        else:
            ocp = solve_min_time(track, p)
        v0 = 0.8 * float(reference_speed(track)[0])
        x0 = np.array([0.0, 0.0, 0.0, v0, 0.0, 0.0, 0.0, 0.0])
        longest = steps(max(args.seconds))
        t0 = time.perf_counter()
        if args.first == "policy":
            first = policy_rollout(track, x0, longest, args.agent)
        else:
            first = min_time_guess(ocp, track, x0, longest)
        guess_s = time.perf_counter() - t0

        def extend(state, n, ocp=ocp, track=track):
            return min_time_guess(ocp, track, state, n)

        for sec in args.seconds:
            H = steps(sec)
            sched = [H] if args.method == "direct" else [steps(s) for s in STAGES if s < sec] + [H]
            for T in sched:
                solver_for(T)
            runs = {"forward": solve_continuation(solver_for, track, x0, sched, first, extend,
                                                  overlap=args.overlap, p=p)}
            if args.polish:
                nxt = steps(next(s for s in STAGES if s > sec))
                solver_for(nxt)
                runs["polish"] = solve_continuation(solver_for, track, x0, [nxt, H], None, extend,
                                                    overlap=args.overlap, p=p, previous=runs["forward"][-1])
            spec = WindowSpec(steps=H)
            # One row per method: the forward run alone, and (with --polish) the better of
            # the two starts, charged with both runs' cost.
            variants = [(args.method, ["forward"])]
            if args.polish:
                variants.append((args.method + "+polish", ["forward", "polish"]))
            for method, keys in variants:
                stages = [r for k in keys for r in runs[k]]
                best_key = max(keys, key=lambda k: runs[k][-1].value)
                best = runs[best_key][-1]
                ret, alive, xr = replay_window(track, x0, best.a)
                row = dict(track=name, seconds=sec, method=method,
                           stages=" ".join(f"{len(r.a) * 0.05:.0f}" for r in stages), status=best.status,
                           best_start=best_key, J_first=window_return(first[0][: H + 1], first[1][:H], spec),
                           iterations=sum(r.iterations for r in stages), rounds=sum(r.rounds for r in stages),
                           wall_s=sum(r.solve_seconds for r in stages),
                           cpu_s=sum(r.meta["cpu_seconds"] for r in stages),
                           J=best.value, J_starts={k: runs[k][-1].value for k in keys},
                           J_check=window_return(best.x, best.a, spec), J_replay=ret,
                           replay_rel=(ret - best.value) / abs(best.value), replay_survives=alive,
                           max_state_dev=float(np.max(np.abs(xr - best.x))),
                           peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                           stage_log=[dict(seconds=len(r.a) * 0.05, iterations=r.iterations, rounds=r.rounds,
                                           cpu_s=r.meta["cpu_seconds"], J=r.value, status=r.status)
                                      for r in stages])
                rows.append(row)
                print(f"{name} {sec:.0f}s [{method}; stages {row['stages']}]: {row['status']} "
                      f"it={row['iterations']} rounds={row['rounds']} wall={row['wall_s']:.1f}s "
                      f"cpu={row['cpu_s']:.1f}s J*={row['J']:.5f} "
                      f"(starts {', '.join(f'{k} {v:.5f}' for k, v in row['J_starts'].items())}) "
                      f"replay={ret:.5f} ({row['replay_rel']:+.1e}) survives={alive}", flush=True)
                with open(out / f"{name}-{int(sec)}s-{method}.pkl", "wb") as fh:
                    pickle.dump({"result": best, "row": row}, fh)
            for st in row["stage_log"]:
                print(f"    {st['seconds']:.0f}s: it={st['iterations']} rounds={st['rounds']} "
                      f"cpu={st['cpu_s']:.1f}s J*={st['J']:.5f} {st['status']}", flush=True)
        print(f"{name}: guess {guess_s:.1f}s; compile (cached after the first run) "
              + ", ".join(f"{T * 0.05:.0f}s window {v:.1f}s" for T, v in sorted(compile_s.items())), flush=True)

    lines = [("| Track | Window | Method | Stages (s) | Iter | Rounds | Solve wall | CPU | J* | Replay diff | "
              "Survives | Peak RSS |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['track']} | {r['seconds']:.0f} s | {r['method']} | {r['stages']} | {r['iterations']} | "
                     f"{r['rounds']} | {r['wall_s']:.0f} s | {r['cpu_s']:.0f} s | {r['J']:.5f} | "
                     f"{r['replay_rel']:+.1e} | {r['replay_survives']} | {r['peak_rss_gb']:.1f} GB |")
    table = "\n".join(lines) + "\n"
    print(table)
    tag = "-".join(args.tracks) + f"-{args.method}" + ("-polish" if args.polish else "")
    (out / f"table-{tag}.md").write_text(table)
    with open(out / f"rows-{tag}.pkl", "wb") as fh:
        pickle.dump(rows, fh)


if __name__ == "__main__":
    main()
