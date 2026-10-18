"""K2a spike: does the training-objective OCP converge, how fast, and does it replay?

    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 20 40 60

For each track and window length: x0 at s = 0 on the reference (n = 0, xi = 0,
v = 0.8 v_QSS), warm start = the s-domain OCP mapped to time, solve the TO-OCP, then
replay its controls open-loop in the float64 JAX simulator (env termination checks
included) and compare returns. Writes docs/findings/to-ocp-spike.md.
"""

import argparse
import os
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
from apexzero.ocp.window import WindowSolver, WindowSpec, window_return
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import track_arrays

jax.config.update("jax_enable_x64", True)


def warm_start(track, ocp, x0, spec):
    """Map the s-domain optimum to the time grid, starting from x0's speed."""
    t_grid = ocp.t
    L = track.length
    T = spec.steps
    t = np.arange(T + 1) * spec.dt
    # Time along the OCP lap, shifted so the guess starts at s = 0 (x0's position).
    s = np.interp(np.mod(t, ocp.lap_time), t_grid, ocp.s) + L * np.floor(t / ocp.lap_time)
    z = np.stack([np.interp(np.mod(si, L), ocp.s, ocp.z[:, k]) for k in range(7) for si in [s]][0:7])
    xs = np.vstack([s, z]).T
    xs[0] = x0
    u = np.stack([np.interp(np.mod(si, L), ocp.s, ocp.u[:, k]) for k in range(2) for si in [s[:-1]]][0:2]).T
    return xs, u


def policy_rollout(track, x0, spec, agent_path):
    """Dynamically consistent warm start: the trained policy driven from x0."""
    import pickle as _pk

    from apexzero.agents.ppo import Actor, PPOConfig, RunningMeanStd

    with open(agent_path, "rb") as fh:
        d = _pk.load(fh)
    actor = Actor(PPOConfig(**d["config"]), 2, False)
    rms = RunningMeanStd(**{k: jnp.asarray(v) for k, v in d["obs_rms"].items()})
    tr = track_arrays(track, dtype=jnp.float64)
    params = racing.EnvParams(window_steps=10**9, substeps=spec.substeps)
    st = racing.EnvState(x=jnp.asarray(x0), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=racing.default_context().astype(jnp.float64),
                         hidden_mu=jnp.array(1.0), t=jnp.array(0),
                         slow=jnp.array(0.0), s0=jnp.asarray(x0[0]))
    p = racing.params_from_context(st.ctx)
    obs = racing.observe(st, tr, p)

    def body(c, _):
        st, ob = c
        a = jnp.clip(actor.apply(d["actor"], rms.normalize(ob.astype(jnp.float32)))[0], -1, 1).astype(jnp.float64)
        ob2, st2, _, _, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, ob2), (st2.x, a)

    _, (xs, acts) = jax.lax.scan(body, (st, obs), None, spec.steps)
    return np.vstack([x0, np.asarray(xs)]), np.asarray(acts)


def replay(track, p, x0, acts, spec):
    tr = track_arrays(track, dtype=jnp.float64)
    params = racing.EnvParams(window_steps=10**9, substeps=spec.substeps)
    st = racing.EnvState(x=jnp.asarray(x0), prev_action=jnp.zeros(2), accel=jnp.zeros(2),
                         ctx=racing.default_context().astype(jnp.float64),
                         hidden_mu=jnp.array(1.0), t=jnp.array(0),
                         slow=jnp.array(0.0), s0=jnp.asarray(x0[0]))

    def body(c, a):
        st, alive = c
        _, st2, r, term, _, _ = racing.step_no_reset(st, a, tr, params)
        return (st2, alive & ~term), (st2.x, r, alive & ~term)

    (_, _), (xs, rs, alive) = jax.lax.scan(body, (st, jnp.array(True)), jnp.asarray(acts))
    xs, rs, alive = np.asarray(xs), np.asarray(rs), np.asarray(alive)
    ret = float(np.sum(spec.gamma ** np.arange(len(rs)) * rs))
    return ret, bool(alive.all()), np.vstack([x0, xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["Hockenheim"])
    ap.add_argument("--seconds", type=float, nargs="+", default=[20, 40, 60])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--init", choices=["ocp", "policy"], default="policy")
    ap.add_argument("--agent", default="runs/hockenheim-specialist-mlp/agent.pkl")
    ap.add_argument("--radius", type=float, default=15.0)
    args = ap.parse_args()
    p = F1_2025
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
        for sec in args.seconds:
            spec = WindowSpec(steps=round(sec / 0.05), radius=args.radius)
            if args.init == "policy":
                xs0, a0 = policy_rollout(track, x0, spec, args.agent)
            else:
                xs0, a0 = warm_start(track, ocp, x0, spec)
            j_init = window_return(xs0, a0, spec)
            t0 = time.perf_counter()
            solver = WindowSolver(spec, threads=args.threads)
            build = time.perf_counter() - t0
            t0 = time.perf_counter()
            res = solver.solve(track, x0, init=(xs0, a0))
            wall = time.perf_counter() - t0
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
            ret_ocp = window_return(res.x, res.a, spec)
            ret_rep, alive, xr = replay(track, p, x0, res.a, spec)
            rows.append(dict(track=name, seconds=sec, steps=spec.steps, status=res.status, init=args.init,
                             J_init=j_init, radius=args.radius,
                             iterations=res.iterations, rounds=res.rounds, build_s=build,
                             solve_s=res.solve_seconds, wall_s=wall,
                             peak_rss_gb=rss_gb, J=res.value, J_check=ret_ocp, J_replay=ret_rep,
                             replay_rel=(ret_rep - res.value) / abs(res.value),
                             replay_survives=alive, max_state_dev=float(np.max(np.abs(xr - res.x)))))
            r = rows[-1]
            print(f"{name} {sec:.0f}s: {r['status']} it={r['iterations']} rounds={r['rounds']} build={build:.0f}s solve={r['solve_s']:.0f}s "
                  f"J_init={j_init:.4f} J*={r['J']:.4f} replay={r['J_replay']:.4f} ({r['replay_rel']*100:+.4f}%) "
                  f"survives={alive} maxdev={r['max_state_dev']:.2e} peakRSS={rss_gb:.1f}GB", flush=True)
            out = Path("outputs/to_ocp"); out.mkdir(parents=True, exist_ok=True)
            with open(out / f"{name}-{int(sec)}s.pkl", "wb") as fh:
                pickle.dump(res, fh)
    lines = ["# TO-OCP spike (K2a)", "",
             "Training-objective OCP in window mode on the RL environment's exact problem (RK2 x 10 at",
             "5 ms, 20 Hz zero-order hold, corridor and slip limits tightened by 3 cm, discounted reward).",
             "Warm start: the s-domain minimum-time optimum mapped to time. Replay: the optimal controls",
             "applied open-loop in the float64 JAX environment, with its termination checks.", "",
             "| Track | Window | Status | Iter (rounds) | Build | Solve | Peak RSS | J* | Replay | Diff | Survives |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['track']} | {r['seconds']:.0f} s | {r['status']} | {r['iterations']} ({r['rounds']}) | "
                     f"{r['build_s']:.0f} s | {r['solve_s']:.0f} s | {r['peak_rss_gb']:.1f} GB | {r['J']:.3f} | {r['J_replay']:.3f} | "
                     f"{r['replay_rel']*100:+.4f}% | {r['replay_survives']} |")
    lines += ["", f"Machine: {os.cpu_count()} CPU threads (Apple M3 Pro); IPOPT + MUMPS, single process."]
    Path("docs/findings").mkdir(parents=True, exist_ok=True)
    Path("docs/findings/to-ocp-spike.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
