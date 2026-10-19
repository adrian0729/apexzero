"""Where and why does a policy crash? Per-circuit crash diagnostics (R-dev only).

    uv run python scripts/diagnose_crashes.py --run runs/dr-pilot-200M [--ckpt LAST]

For each circuit, drives the checkpoint deterministically from a rolling start (as in
eval/laps.py) and, at the first termination, records the arc length, the termination
cause (corridor, spin, tyre slip, stall), speed, the local corner geometry (minimum
radius within +/-60 m, track width) and where that corner sits in the generator's
training distribution (percentile of corner radius among generated strips). Writes
docs/findings/<run>-crashes.md.
"""

import argparse
import pickle
from pathlib import Path

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.checkpoints import list_checkpoints
from apexzero.agents.ppo import Actor, PPOConfig
from apexzero.audit.rollouts import snapshot_from_pickle
from apexzero.env import racing
from apexzero.env.adapters import reference_speed
from apexzero.tracks.corners import R_DEV, geometry_stats, segment
from apexzero.tracks.generator import fit_grammar, generate_strip, to_track
from apexzero.tracks.reference import refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.sim import track_arrays


def first_crash(policy, track, v0, laps=2, params=racing.EnvParams()):
    tr = track_arrays(track)
    ctx = racing.default_context()
    st = racing.EnvState(x=jnp.array([0.0, 0.0, 0.0, v0, 0.0, 0.0, 0.0, 0.0]), prev_action=jnp.zeros(2),
                         accel=jnp.zeros(2), ctx=ctx, hidden_mu=jnp.array(1.0), t=jnp.array(0),
                         slow=jnp.array(0.0), s0=jnp.array(0.0))
    p = racing.params_from_context(ctx)
    obs = racing.observe(st, tr, p)
    steps = int(laps * 3 * track.length / max(v0, 20.0) / params.dt)
    params = params._replace(window_steps=10**9)

    @jax.jit
    def run(st, obs):
        def body(c, _):
            st, ob = c
            a = policy(ob[None])[0]
            ob2, st2, _, term, _, info = racing.step_no_reset(st, a, tr, params)
            return (st2, ob2), (st2.x, term, info["off_track"], info["spun"], info["slipped"], info["stalled"])
        return jax.lax.scan(body, (st, obs), None, steps)[1]

    xs, term, off, spun, slip, stall = (np.asarray(a) for a in run(st, obs))
    if not term.any():
        return None
    k = int(np.argmax(term))
    cause = "corridor" if off[k] else "slip" if slip[k] else "spin" if spun[k] else "stall" if stall[k] else "?"
    return {"s": float(xs[k, 0] % track.length), "lap": int(xs[k, 0] // track.length) + 1,
            "v_kph": float(xs[k, 3] * 3.6), "n": float(xs[k, 1]), "cause": cause}


def local_geometry(track, s, half=60.0):
    d = np.abs(((track.s - s + track.length / 2) % track.length) - track.length / 2)
    m = d <= half
    k = np.abs(track.kappa[m]).max()
    i = np.argmin(d)
    return {"r_min": float(1 / max(k, 1e-9)), "width": float(track.w_left[i] + track.w_right[i])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/dr-pilot-200M"))
    ap.add_argument("--ckpt", default="LAST")
    args = ap.parse_args()
    ckpts = list_checkpoints(args.run)
    path = ckpts[-1] if args.ckpt == "LAST" else Path(args.ckpt)
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    snap = snapshot_from_pickle(d)
    actor = Actor(PPOConfig(**d["config"]), 2, False)

    def policy(obs):
        return actor.apply(snap.actor, snap.obs_rms.normalize(obs))[0]

    dev = [load_tumftm(f"data/tracks/tumftm/{n}.csv") for n in R_DEV]
    g = fit_grammar(dev)
    arr = jax.jit(jax.vmap(lambda k: generate_strip(k, g)))(jax.random.split(jax.random.PRNGKey(7), 300))
    gen = geometry_stats([segment(to_track(jax.tree.map(lambda a, i=i: a[i], arr))) for i in range(300)])
    gen_r = np.sort(gen["r_min"])

    rows = []
    for name in R_DEV:
        tr = refine_reference(load_tumftm(f"data/tracks/tumftm/{name}.csv"))
        v0 = float(reference_speed(tr)[0]) * 0.9
        c = first_crash(policy, tr, v0)
        if c is None:
            rows.append((name, None, None))
            print(f"{name:13s} no crash in 2 laps", flush=True)
            continue
        geo = local_geometry(tr, c["s"])
        pct = 100.0 * np.searchsorted(gen_r, geo["r_min"]) / len(gen_r)
        rows.append((name, c, dict(geo, pct=pct)))
        print(f"{name:13s} lap {c['lap']} s={c['s']:6.0f} m  {c['cause']:8s} {c['v_kph']:5.0f} km/h  "
              f"corner R_min {geo['r_min']:6.1f} m (generator pct {pct:5.1f})  width {geo['width']:4.1f} m", flush=True)

    lines = [f"# Crash diagnostics: {args.run.name} ({path.stem.lstrip('0')} steps)", "",
             "Deterministic policy, default car, rolling start, two laps; first termination per",
             "circuit (R-dev only). 'Generator pct' is the percentile of the local corner's minimum",
             "radius among corners of 300 generated training strips (low = tighter than most).", "",
             "| Circuit | Crash | Lap | s (m) | Cause | Speed (km/h) | Corner R_min (m) | Generator pct | Width (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, c, geo in rows:
        if c is None:
            lines.append(f"| {name} | no | - | - | - | - | - | - | - |")
        else:
            lines.append(f"| {name} | yes | {c['lap']} | {c['s']:.0f} | {c['cause']} | {c['v_kph']:.0f} | "
                         f"{geo['r_min']:.1f} | {geo['pct']:.1f} | {geo['width']:.1f} |")
    out = Path("docs/findings") / f"{args.run.name}-crashes.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
