"""The value-gradient term (VG-0): its cost, and its residual on trained networks. ENGINEERING.

    nice -n 10 uv run python scripts/vgl.py overhead [--num-envs 128 --num-steps 64 ...]
    nice -n 10 uv run python scripts/vgl.py residual runs/<run> [--checkpoint <file>] ...

**overhead.** The wall time of one PPO update (bank refresh, rollout, learner, normaliser
update) of ``scripts/train_dr.py``'s recipe at a chosen small size, with and without the term
(``ppo.ValueGradient`` at --vg-coef; its cost does not depend on the coefficient as long as it
is above 0), on the procedural env with the default grammar fitted on R-dev. Each arm is one
``make_train`` call of --updates updates, compiled first and then run --repeats times; the
per-update time is the median run's wall time over --updates. Writes a JSON record (the sizes,
both times, their ratio, the compile times, the environment and HEAD) to stdout and to --out.

**residual** (the pre-registration draft's manipulation check M0, docs/prereg/
vg1-value-gradient.md). For one checkpoint: --rollouts rollouts of --num-steps steps of its
stochastic policy at --num-envs envs on fresh default-grammar strips (seeded by --seed, the
car randomised as in training), the term's VGL(--vg-lambda) targets from the network's own
actor and critic (``ppo.vg_targets``, winsorised at --vg-winsor), and the critic's own
dV/dy. It reports, per component of y and pooled (mean and median over the components), the
normalised residual E[(dV/dy_k - G'_k)^2] / Var(G'_k) over the valid rows, and the share of
valid rows. It also
reports the one-step action projection, with B_t = dy_{t+1}/da_t on steps that continue the
episode: the normalised residual of B_t' dV/dy(y_{t+1}) against B_t' G'_{t+1} per action
(steer, pedal), and each component's share of the projection, mean |B_kj G'_k| over the sum
of those over k (and the same for the critic's own gradient): how much s and xi weigh in what
one action changes. The same numbers come from a network trained with or without the term.
Nothing is evaluated on any circuit; R_TEST is never read.

Another workflow shares the Mac: the script lowers its own priority (--nice, default 10) and
runs XLA on --threads intra-op threads (default 4).
"""

from __future__ import annotations

import os
import sys


def _threads(argv: list[str]) -> int:
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 4


if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                               f"intra_op_parallelism_threads={_threads(sys.argv)}")

import argparse
import dataclasses
import hashlib
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple

import _bootstrap  # noqa: F401  (makes src/ importable)
import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import (
    Actor,
    AgentState,
    Critic,
    PPOConfig,
    RunningMeanStd,
    ValueGradient,
    make_rollout,
    make_train,
    state_gradient_fn,
    vg_targets,
)
from apexzero.env import racing
from apexzero.env.adapters import VG_NAMES, procedural_env
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar, tilted
from apexzero.tracks.track import load_tumftm

ROOT = Path(__file__).resolve().parents[1]


def lower_priority(level: int) -> None:
    cur = os.nice(0)
    if cur < level:
        os.nice(level - cur)


def environment() -> dict:
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src",
                                     "scripts"], capture_output=True, text=True,
                                    check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        head, dirty = None, None
    return {"python": platform.python_version(), "jax": jax.__version__, "numpy": np.__version__,
            "machine": platform.machine(), "system": platform.system(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "xla_flags": os.environ.get("XLA_FLAGS", ""), "nice": os.nice(0),
            "devices": [str(d) for d in jax.devices()], "head": head, "src_or_scripts_dirty": dirty}


def default_grammar():
    """train_dr.py's default grammar: fitted on the R-dev circuits, default widths."""
    dev = [load_tumftm(str(ROOT / "data" / "tracks" / "tumftm" / f"{n}.csv")) for n in R_DEV]
    return fit_grammar(dev, real_widths=False, width_tracks=None)


def train_cfg(num_envs: int, num_steps: int, total: int) -> PPOConfig:
    """train_dr.py's PPOConfig (MLP 3 x 256 tanh, 5 heads), at another size."""
    return PPOConfig(num_envs=num_envs, num_steps=num_steps, total_timesteps=total,
                     network="mlp", gamma=0.995, gae_lambda=0.95, actor_width=256,
                     critic_width=256, mlp_layers=3, activation="tanh", ensemble_heads=5)


# ------------------------------------------------------------------------- overhead


def time_arm(cfg: PPOConfig, env, vg: ValueGradient | None, repeats: int, seed: int) -> dict:
    train, _, _ = make_train(cfg, env, vg=vg)
    fn = jax.jit(train)
    t0 = time.perf_counter()
    compiled = fn.lower(jax.random.PRNGKey(seed)).compile()
    t_compile = time.perf_counter() - t0
    runs = []
    for r in range(repeats):
        t0 = time.perf_counter()
        agent, metrics = compiled(jax.random.PRNGKey(seed + r))
        jax.block_until_ready((agent, metrics))
        runs.append(time.perf_counter() - t0)
    finite = bool(np.all(np.isfinite(np.asarray(metrics["value_loss"]))))
    out = {"compile_s": t_compile, "run_s": runs, "per_update_s": float(np.median(runs)) / cfg.num_updates,
           "value_loss_finite": finite}
    if vg is not None:
        out["vg_loss_last"] = float(np.asarray(metrics["vg_loss"])[-1])
        out["vg_valid_last"] = float(np.asarray(metrics["vg_valid"])[-1])
    return out


def cmd_overhead(a) -> dict:
    g = default_grammar()
    env = procedural_env(g, bank_size=a.bank, refresh_per_update=a.refresh, g_alt=tilted(g),
                         alt_fraction=0.0)
    cfg = train_cfg(a.num_envs, a.num_steps, a.num_envs * a.num_steps * a.updates)
    vg = ValueGradient(coef=a.vg_coef, lam=a.vg_lambda, winsor=a.vg_winsor)
    base = time_arm(cfg, env, None, a.repeats, a.seed)
    term = time_arm(cfg, env, vg, a.repeats, a.seed)
    return {"what": "per-update wall time of train_dr.py's recipe, default path and with the "
                    "value-gradient term (VG-0)",
            "sizes": {"num_envs": a.num_envs, "num_steps": a.num_steps, "updates": a.updates,
                      "repeats": a.repeats, "bank": a.bank, "refresh": a.refresh,
                      "minibatches": cfg.num_minibatches, "epochs": cfg.update_epochs,
                      "transitions_per_update": cfg.batch_size},
            "vg": dataclasses.asdict(vg), "default": base, "with_term": term,
            "ratio": term["per_update_s"] / base["per_update_s"],
            "environment": environment()}


# ------------------------------------------------------------------------- residual


class Params(NamedTuple):
    """A checkpoint's network, where the rollout and the targets read ``.params``."""

    params: Any


def load_snapshot(path: Path) -> tuple[PPOConfig, AgentState]:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    known = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg = PPOConfig(**{k: v for k, v in d["config"].items() if k in known})
    rms = lambda x: RunningMeanStd(**{k: jnp.asarray(v) for k, v in x.items()})
    agent = AgentState(actor=Params(d["actor"]), critic=Params(d["critic"]),
                       obs_rms=rms(d["obs_rms"]), ret_rms=rms(d["ret_rms"]))
    return cfg, agent


def residual(cfg: PPOConfig, agent: AgentState, env, vg: ValueGradient, rollouts: int,
             num_envs: int, num_steps: int, seed: int) -> dict:
    """The normalised residual of the critic's dV/dy against the term's targets on rollouts of
    the network's own stochastic policy (module docstring, 'residual')."""
    cfg = dataclasses.replace(cfg, num_envs=num_envs, num_steps=num_steps)
    actor, critic = Actor(cfg, racing.ACTION_DIM, False), Critic(cfg)
    rollout = jax.jit(make_rollout(cfg, env, actor, critic, vg=vg))
    targets = jax.jit(lambda ag, tr, cap, p: vg_targets(cfg, vg, actor, critic, env.vg_model,
                                                        ag, tr, cap, p))
    grad_y = state_gradient_fn(cfg, critic, env.vg_model)
    grads = jax.jit(jax.vmap(jax.vmap(lambda cp, rms, y, s, p: grad_y(cp, rms, y, s, p),
                                      in_axes=(None, None, 0, 0, None)),
                             in_axes=(None, None, 0, 0, None)))
    step = env.vg_model.step
    b_one = lambda y, a, s, p: jax.jacfwd(lambda q: step(y, q, s, p)[0])(a)       # [Y, 2]
    action_jac = jax.jit(jax.vmap(jax.vmap(b_one, in_axes=(0, 0, 0, None)),
                                  in_axes=(0, 0, 0, None)))
    key = jax.random.PRNGKey(seed)
    key, kp, kr = jax.random.split(key, 3)
    params = env.init_params(kp)
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(kr, num_envs), params)
    z = jnp.zeros(num_envs)
    carry = (agent, st, obs, z, z, z)
    valid_n, total_n = 0, 0
    G_all, g_all, qt_all, qc_all, ct_all, cc_all = [], [], [], [], [], []
    for _ in range(rollouts):
        key, k = jax.random.split(key)
        (ag, st, obs, ret, epr, epl, _), ro = rollout(*carry, k, params)
        vt = targets(agent, ro.traj, ro.vg, params)
        g = np.asarray(grads(agent.critic.params, agent.obs_rms, vt.y, ro.vg.state, params))
        G = np.asarray(vt.target)
        v = np.asarray(vt.valid)
        G_all.append(G[v])
        g_all.append(g[v])
        valid_n += int(v.sum())
        total_n += v.size
        # The one-step action projection: B_t = dy_{t+1}/da_t, on steps that continue the
        # episode, against the next row's target and the critic's own gradient there.
        B = np.asarray(action_jac(vt.y, ro.traj.action, ro.vg.state, params))       # [T, N, Y, 2]
        done = np.asarray(ro.traj.terminated | ro.traj.truncated)
        pair = (~done[:-1]) & v[:-1] & v[1:]
        Bp, Gn, gn = B[:-1][pair], G[1:][pair], g[1:][pair]
        qt_all.append(np.einsum("pya,py->pa", Bp, Gn))
        qc_all.append(np.einsum("pya,py->pa", Bp, gn))
        ct_all.append(np.abs(Bp * Gn[..., None]))                   # [P, Y, 2] per component
        cc_all.append(np.abs(Bp * gn[..., None]))
        carry = (agent._replace(ret_rms=ag.ret_rms), st, obs, ret, epr, epl)
    G, g = np.concatenate(G_all), np.concatenate(g_all)
    var = G.var(0)
    nres = ((g - G) ** 2).mean(0) / np.where(var > 0, var, np.nan)
    qt, qc = np.concatenate(qt_all), np.concatenate(qc_all)
    qvar = qt.var(0)
    qres = ((qc - qt) ** 2).mean(0) / np.where(qvar > 0, qvar, np.nan)
    def share(c):                                                   # [Y, 2], columns sum to 1
        m = np.concatenate(c).mean(0)
        return m / m.sum(0, keepdims=True)

    sh_t, sh_c = share(ct_all), share(cc_all)
    acts = ("steer", "pedal")
    by = lambda arr: {n: float(x) for n, x in zip(VG_NAMES, arr, strict=True)}
    return {"components": list(VG_NAMES),
            "normalised_residual": by(nres),
            "normalised_residual_mean": float(np.nanmean(nres)),
            "normalised_residual_median": float(np.nanmedian(nres)),
            "target_sd": by(np.sqrt(var)),
            "critic_rms": by(np.sqrt((g ** 2).mean(0))),
            "action_projected_residual": {a: float(x) for a, x in zip(acts, qres, strict=True)},
            "action_projection_share_target": {a: by(sh_t[:, j]) for j, a in enumerate(acts)},
            "action_projection_share_critic": {a: by(sh_c[:, j]) for j, a in enumerate(acts)},
            "action_pairs": int(qt.shape[0]), "valid_rows": valid_n, "rows": total_n}


def cmd_residual(a) -> dict:
    run = Path(a.run)
    ck = Path(a.checkpoint) if a.checkpoint else max((run / "ckpt").glob("*.pkl"))
    cfg, agent = load_snapshot(ck)
    g = default_grammar()
    env = procedural_env(g, bank_size=a.bank, refresh_per_update=a.refresh, g_alt=tilted(g),
                         alt_fraction=0.0)
    vg = ValueGradient(coef=1.0, lam=a.vg_lambda, winsor=a.vg_winsor)
    out = residual(cfg, agent, env, vg, a.rollouts, a.num_envs, a.num_steps, a.seed)
    return {"what": "the value-gradient term's normalised residual on a checkpoint (M0)",
            "run": run.name, "checkpoint": ck.name,
            "checkpoint_sha256": hashlib.sha256(ck.read_bytes()).hexdigest(),
            "sizes": {"rollouts": a.rollouts, "num_envs": a.num_envs, "num_steps": a.num_steps,
                      "bank": a.bank, "seed": a.seed},
            "vg": {"lam": a.vg_lambda, "winsor": a.vg_winsor}, **out,
            "environment": environment()}


# ------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=4, help="XLA intra-op threads (default 4)")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--out", default="", help="also write the JSON record here")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("overhead")
    o.add_argument("--num-envs", type=int, default=128)
    o.add_argument("--num-steps", type=int, default=64)
    o.add_argument("--updates", type=int, default=6)
    o.add_argument("--repeats", type=int, default=3)
    o.add_argument("--bank", type=int, default=256)
    o.add_argument("--refresh", type=int, default=16)
    o.add_argument("--vg-coef", type=float, default=0.1)
    o.add_argument("--vg-lambda", type=float, default=0.95)
    o.add_argument("--vg-winsor", type=float, default=0.005)
    o.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("residual")
    r.add_argument("run")
    r.add_argument("--checkpoint", default="")
    r.add_argument("--rollouts", type=int, default=4)
    r.add_argument("--num-envs", type=int, default=64)
    r.add_argument("--num-steps", type=int, default=64)
    r.add_argument("--bank", type=int, default=256)
    r.add_argument("--refresh", type=int, default=16)
    r.add_argument("--vg-lambda", type=float, default=0.95)
    r.add_argument("--vg-winsor", type=float, default=0.005)
    r.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args(argv)
    lower_priority(a.nice)
    jax.config.update("jax_enable_x64", False)
    rec = cmd_overhead(a) if a.cmd == "overhead" else cmd_residual(a)
    text = json.dumps(rec, indent=1)
    print(text, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
