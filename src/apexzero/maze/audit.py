"""E5 external check: level scores of JaxUED maze checkpoints against exact BFS regret.

The subject is JaxUED's own maze training code (``third_party/jaxued_examples/maze_plr.py``,
vendored unmodified and imported here as a module), so nothing about the agent is ours:
its ``ActorCritic`` (conv + ResetRNN LSTM), its orbax checkpoints, its ``Maze`` env wrapped
in ``AutoReplayWrapper`` exactly as ``main()`` builds it, and its level generator.

Protocol per checkpoint (a fresh-score audit, the maze analogue of the racing E1):
- N levels from the training generator (``make_level_generator(13, 13, n_walls)``), a fixed
  audit key, so every checkpoint and run sees the same levels;
- one AutoReplay stream per level with the stochastic policy, run until every level has
  completed W episodes. This is ``sample_trajectories_rnn`` step for step (the LSTM resets on
  the previous ``done``; the time limit is terminal, maze ``env.py``), only longer, because
  JaxUED's fixed 256-step rollout would give a trained agent ten episodes and an untrained
  one a single episode. Episodes are i.i.d.: the level is deterministic and the carry resets;
- operational scores from episodes 1..M of the stream (M = 4: JaxUED's new-level branch
  on a rollout that ended after M episodes), and the regret target from episodes M+1..W,
  so score and target share no sampling noise (as windows 1-4 / 5+ in the racing spec);
- J* from BFS (``apexzero.maze.oracle``); J^pi = mean discounted return of the target
  episodes; regret = J* - J^pi, which is >= 0 for every episode, not just on average,
  because the env is deterministic and J* is the maximum over action sequences (asserted).

Scores, all on the same operational episodes:
- PVL and MaxMC exactly as JaxUED computes them: its ``compute_gae`` (from maze_plr.py) on
  the stream with the PPO's gamma and lambda (0.995, 0.98), then ``jaxued.utils``
  ``positive_value_loss`` / ``max_mc(compute_max_returns)``, with ``done`` masked after the
  M-th episode so later steps form one trailing incomplete episode, which JaxUED drops;
- the same two, plus L1, MaxMC-unit, learnability p(1 - p) with success = goal reached, MNA,
  and the lambda variants, from our per-episode primitives (``scoring_gae``, ``mna_terms``)
  and the spec's aggregation (``estimators.level_scores``). The JaxUED and ours columns
  must agree (tests/test_maze_audit.py);
- baselines: agent-only -J^pi, policy-agnostic J* and BFS path length, the oracle-operational
  reference J* - J^pi(operational episodes), and random.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jaxued.environments import Maze
from jaxued.environments.maze import Level, make_level_generator
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

import apexzero.compat  # noqa: F401  (restores jax.tree_map, which jaxued calls)
from apexzero.audit.estimators import LAMBDAS, WindowStats, level_scores, mna_terms, scoring_gae
from apexzero.maze.oracle import min_actions_to_goal, optimal_plan, optimal_return

MAZE_PLR = Path(__file__).resolve().parents[3] / "third_party" / "jaxued_examples" / "maze_plr.py"
MAZE_SIZE = 13            # maze_plr.main: Maze(max_height=13, max_width=13, ...)


def float32(fn):
    """Run ``fn`` with JAX's default 32-bit types, as JaxUED trains. Some processes (the racing
    tests) enable x64 globally; there the maze observation ``uint8 / 10.0`` would become
    float64, the LSTM carry would change dtype inside ``scan``, and ``jax.random.choice`` in
    the level generator would draw different levels."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with jax.enable_x64(False):
            return fn(*args, **kwargs)
    return wrapped


@lru_cache(maxsize=1)
def maze_plr():
    """JaxUED's examples/maze_plr.py as a module (no edits; its ``main`` is not run)."""
    spec = importlib.util.spec_from_file_location("jaxued_maze_plr", MAZE_PLR)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=4)
def make_env(agent_view_size: int = 5):
    """The training env exactly as ``maze_plr.main`` builds it (AutoReplay on one level)."""
    env = Maze(max_height=MAZE_SIZE, max_width=MAZE_SIZE, agent_view_size=agent_view_size,
               normalize_obs=True)
    return AutoReplayWrapper(env), env.default_params


@float32
def generate_levels(n: int, key: jax.Array, n_walls: int = 25) -> Level:
    """n levels from the DR generator the training run samples from (batched Level)."""
    gen = make_level_generator(MAZE_SIZE, MAZE_SIZE, n_walls)
    return jax.vmap(gen)(jax.random.split(key, n))


def bfs_optimum(levels: Level, gamma: float = 0.995, max_steps: int = 250):
    """Per level: J* and the BFS action count k* (0 where the goal is unreachable, J* = 0)."""
    walls, pos = np.asarray(levels.wall_map, bool), np.asarray(levels.agent_pos)
    dirs, goal = np.asarray(levels.agent_dir), np.asarray(levels.goal_pos)
    j = np.array([optimal_return(walls[i], pos[i], dirs[i], goal[i], gamma, max_steps)
                  for i in range(len(walls))])
    k = np.array([min_actions_to_goal(walls[i], pos[i], dirs[i], goal[i], max_steps) or 0
                  for i in range(len(walls))])
    return j, k


# --------------------------------------------------------------------------- checkpoints


class MazeCheckpoint(NamedTuple):
    params: Any          # flax variables {"params": ...} for maze_plr.ActorCritic
    config: dict         # the run's config.json
    step: int            # orbax step = maze_plr's eval_step index
    num_updates: int     # PPO updates (DR + replay + mutation) at save time
    sampler: dict        # the PLR level buffer (LevelSampler state)


def checkpoint_steps(run_dir: str | Path) -> list[int]:
    with ocp.CheckpointManager(Path(run_dir).resolve() / "models",
                               item_handlers=ocp.StandardCheckpointHandler()) as mgr:
        return list(mgr.all_steps())


def load_checkpoint(run_dir: str | Path, step: int | None = None) -> MazeCheckpoint:
    """``run_dir`` = <cwd of the training run>/checkpoints/<run_name>/<seed>. Restored the way
    maze_plr's own ``eval_checkpoint`` does it (StandardCheckpointHandler, no target)."""
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "config.json").read_text())
    with ocp.CheckpointManager(run_dir / "models",
                               item_handlers=ocp.StandardCheckpointHandler()) as mgr:
        step = mgr.latest_step() if step is None else step
        ck = mgr.restore(step)
    n = int(ck["num_dr_updates"]) + int(ck["num_replay_updates"]) + int(ck["num_mutation_updates"])
    return MazeCheckpoint(ck["params"], config, int(step), n, ck["sampler"])


# --------------------------------------------------------------------------- policies


class Policy(NamedTuple):
    """A batched policy: ``init(n)`` -> carry; ``act(params, key, carry, obs, last_done)`` ->
    (carry, action [n], value [n]). ``act`` must be a module-level function (it keys the jit cache)."""
    params: Any
    init: Callable[[int], Any]
    act: Callable


def _network_act(params, key, hstate, obs, last_done):
    # sample_trajectories_rnn's sample_step, line for line (maze_plr.py L111-119).
    net = maze_plr().ActorCritic(7)
    x = jax.tree.map(lambda a: a[None, ...], (obs, last_done))
    hstate, pi, value = net.apply(params, x, hstate)
    action = pi.sample(seed=key)
    return hstate, action.squeeze(0), value.squeeze(0)


def network_policy(params) -> Policy:
    """The checkpoint's stochastic policy, with a fresh LSTM carry as in every JaxUED rollout."""
    return Policy(params, lambda n: maze_plr().ActorCritic.initialize_carry((n,)), _network_act)


def _plan_act(plans, key, t, obs, last_done):
    t = jnp.where(last_done, 0, t)               # a new episode restarts the plan
    a = plans[jnp.arange(plans.shape[0]), jnp.minimum(t, plans.shape[1] - 1)]
    return t + 1, a, jnp.zeros(plans.shape[0])


def plan_policy(plans: np.ndarray) -> Policy:
    """Open-loop action sequences [n, L] (padded with any action), replayed every episode."""
    return Policy(jnp.asarray(plans, jnp.int32), lambda n: jnp.zeros(n, jnp.int32), _plan_act)


def bfs_plans(levels: Level, max_steps: int = 250) -> np.ndarray:
    """BFS-optimal plans [n, max_steps]; unreachable goals get 'turn left' forever."""
    walls, pos = np.asarray(levels.wall_map, bool), np.asarray(levels.agent_pos)
    dirs, goal = np.asarray(levels.agent_dir), np.asarray(levels.goal_pos)
    out = np.zeros((len(walls), max_steps), np.int32)
    for i in range(len(walls)):
        plan = optimal_plan(walls[i], pos[i], dirs[i], goal[i], max_steps)
        if plan:
            out[i, :len(plan)] = plan
    return out


# --------------------------------------------------------------------------- rollouts


class MazeRollouts(NamedTuple):
    """Time-major AutoReplay streams, one per level, as sample_trajectories_rnn returns them."""
    reward: np.ndarray       # [T, N]
    value: np.ndarray        # [T, N]   V(o_t) before acting
    done: np.ndarray         # [T, N]   bool (goal reached or time limit)
    last_value: np.ndarray   # [N]      V(o_T), JaxUED's GAE bootstrap for the open tail


@lru_cache(maxsize=16)
def _chunk_runner(act, env, env_params, n: int, chunk: int):
    def run(params, carry, t0, key):
        def step(c, t):
            pc, obs, state, last_done = c
            k_act, k_env = jax.random.split(jax.random.fold_in(key, t))
            pc, action, value = act(params, k_act, pc, obs, last_done)
            obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_env, n), state, action, env_params)
            return (pc, obs, state, done), (reward, value, done)

        return jax.lax.scan(step, carry, t0 + jnp.arange(chunk))

    return jax.jit(run)


@float32
def collect(policy: Policy, levels: Level, episodes: int, key: jax.Array,
            agent_view_size: int = 5, chunk: int = 250) -> MazeRollouts:
    """Run every level's stream until it has completed ``episodes`` episodes. Step keys are
    fold_in(key, t), so the result does not depend on ``chunk`` (only the trimmed length does)."""
    env, env_params = make_env(agent_view_size)
    n = int(levels.wall_map.shape[0])
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
        jax.random.split(k_reset, n), levels, env_params)
    run = _chunk_runner(policy.act, env, env_params, n, chunk)
    carry = (policy.init(n), obs, state, jnp.zeros(n, bool))
    outs, count = [], np.zeros(n, int)
    t_max = episodes * env_params.max_steps_in_episode      # every episode ends by the time limit
    for t0 in range(0, t_max, chunk):
        carry, out = run(policy.params, carry, t0, k_run)
        outs.append(jax.tree.map(np.asarray, out))
        count += outs[-1][2].sum(0)
        if (count >= episodes).all():
            break
    pc, obs, _, last_done = carry
    _, _, last_value = policy.act(policy.params, k_run, pc, obs, last_done)
    reward, value, done = (np.concatenate([o[i] for o in outs]) for i in range(3))
    assert (done.sum(0) >= episodes).all()
    return MazeRollouts(reward, value, done.astype(bool), np.asarray(last_value))


def first_dones(done: np.ndarray, k: int) -> np.ndarray:
    """``done`` with every episode end after the k-th removed: what a rollout that stopped
    after k episodes shows JaxUED's accumulators (later steps = one dropped open tail)."""
    return done & (np.cumsum(done, axis=0) <= k)


def episode_bounds(done: np.ndarray, episodes: int) -> np.ndarray:
    """[N, W, 2] (start, stop) step indices of each level's first W episodes."""
    out = np.zeros((done.shape[1], episodes, 2), int)
    for i in range(done.shape[1]):
        ends = np.nonzero(done[:, i])[0][:episodes]
        out[i, :, 1] = ends + 1
        out[i, 1:, 0] = ends[:-1] + 1
    return out


# --------------------------------------------------------------------------- scores


@float32
def jaxued_scores(ro: MazeRollouts, k: int, gamma: float, lam: float) -> dict[str, np.ndarray]:
    """PVL and MaxMC exactly as maze_plr.py's new-level branch computes them (L545-547) on a
    rollout holding the first k episodes of each stream."""
    adv, _ = maze_plr().compute_gae(gamma, lam, jnp.asarray(ro.last_value), jnp.asarray(ro.value),
                                    jnp.asarray(ro.reward), jnp.asarray(ro.done))
    dones = jnp.asarray(first_dones(ro.done, k))
    values, rewards = jnp.asarray(ro.value), jnp.asarray(ro.reward)
    max_returns = compute_max_returns(dones, rewards)
    return {"pvl": np.asarray(positive_value_loss(dones, adv)),
            "maxmc": np.asarray(max_mc(dones, values, max_returns)),
            "max_return": np.asarray(max_returns),
            "advantages": np.asarray(adv)}


def episode_stats(ro: MazeRollouts, episodes: int, gamma: float, lambdas=LAMBDAS,
                  headline: int = 2) -> WindowStats:
    """Per (level, episode) records from our primitives, in the racing audit's WindowStats
    layout so ``estimators.level_scores`` aggregates them unchanged. Maze specifics:
    every episode end is terminal (mode F = mode B, no chunk cuts inside counted episodes),
    and the success slot (``completed``, S1 in racing) holds 'goal reached' (G_e > 0)."""
    bounds = episode_bounds(ro.done, episodes)
    N, W = bounds.shape[:2]
    shape = (N, W)
    T, G, G_disc, V_first, V_bar, mna_sum = (np.zeros(shape) for _ in range(6))
    pvl, l1, mean_a = (np.zeros((len(lambdas), *shape)) for _ in range(3))
    pvl_disc, l1_disc = np.zeros(shape), np.zeros(shape)
    for i in range(N):
        for w in range(W):
            a0, a1 = bounds[i, w]
            r = ro.reward[a0:a1, i].astype(float)
            v = ro.value[a0:a1, i].astype(float)
            n = len(r)
            term, none = np.r_[np.zeros(n - 1, bool), True], np.zeros(n, bool)
            disc = gamma ** np.arange(n)
            for k, lam in enumerate(lambdas):
                a = scoring_gae(r, v, term, none, 0.0, none, gamma, lam)
                pvl[k, i, w], l1[k, i, w], mean_a[k, i, w] = (
                    np.maximum(a, 0).mean(), np.abs(a).mean(), a.mean())
                if k == headline:
                    pvl_disc[i, w] = np.sum(disc * np.maximum(a, 0)) / disc.sum()
                    l1_disc[i, w] = np.sum(disc * np.abs(a)) / disc.sum()
            mna_sum[i, w] = mna_terms(r, v, term, none, 0.0, none, gamma, lambdas[headline]).sum()
            T[i, w], G[i, w], G_disc[i, w] = n, r.sum(), np.sum(disc * r)
            V_first[i, w], V_bar[i, w] = v[0], v.mean()
    return WindowStats(T, G, G_disc, V_first, V_bar, pvl, l1, mean_a, pvl_disc, l1_disc, mna_sum,
                       np.full(shape, np.nan), G > 0, np.zeros(shape))


def audit_rollouts(ro: MazeRollouts, j_star: np.ndarray, k_star: np.ndarray, episodes: int = 8,
                   score_episodes: int = 4, gamma: float = 0.995, lam: float = 0.98,
                   seed: int = 0) -> dict[str, np.ndarray]:
    """Everything the fidelity analysis needs, per level. Score keys are prefixed ``s_``."""
    if lam not in LAMBDAS:
        raise ValueError(f"PPO lambda {lam} must be one of the stored variants {LAMBDAS}")
    headline = LAMBDAS.index(lam)
    st = episode_stats(ro, episodes, gamma, headline=headline)
    if np.any(st.G_disc > j_star[:, None] + 1e-5):
        raise AssertionError("an episode beat the BFS optimum: env/oracle mismatch")
    op, tgt = range(score_episodes), range(score_episodes, episodes)
    if not len(tgt):
        tgt = op                                          # in-sample (used by tests only)
    j_pi = st.G_disc[:, list(tgt)].mean(-1)
    ours = level_scores(st, op, headline=headline)
    ref = jaxued_scores(ro, score_episodes, gamma, lam)
    rng = np.random.default_rng(seed)
    scores = {
        "pvl": ref["pvl"], "maxmc": ref["maxmc"],                     # JaxUED's own functions
        "pvl_ours": ours["pvl"], "maxmc_ours": ours["maxmc"],
        "l1": ours["l1"], "maxmc_unit": ours["maxmc_unit"],
        "learnability": ours["learnability_s1"], "mna": ours["mna"],
        "agent_only": ours["agent_only"],
        "j_star": j_star, "bfs_steps": k_star.astype(float),
        "oracle_operational": j_star + ours["agent_only"],
        "random": rng.random(len(j_star)),
    }
    scores.update({k: v for k, v in ours.items() if k.startswith(("pvl_lam", "l1_lam"))})
    out = {f"s_{k}": np.asarray(v, float) for k, v in scores.items()}
    out.update(
        j_star=j_star, k_star=k_star, j_pi=j_pi, regret=j_star - j_pi,
        j_pi_all=st.G_disc.mean(-1), success_rate=st.completed.mean(-1),
        success_rate_op=st.completed[:, list(op)].mean(-1),
        mean_steps=st.T.mean(-1), stream_steps=np.full(len(j_star), ro.done.shape[0]),
    )
    return out


def audit_checkpoint(ck: MazeCheckpoint, levels: Level, key: jax.Array, episodes: int = 8,
                     score_episodes: int = 4, oracle: tuple | None = None) -> dict[str, np.ndarray]:
    """Audit one checkpoint on ``levels`` (pass ``oracle`` = bfs_optimum(levels) to reuse it)."""
    gamma, lam = float(ck.config["gamma"]), float(ck.config["gae_lambda"])
    j_star, k_star = oracle if oracle is not None else bfs_optimum(levels, gamma)
    ro = collect(network_policy(ck.params), levels, episodes, key,
                 agent_view_size=int(ck.config["agent_view_size"]))
    return audit_rollouts(ro, j_star, k_star, episodes, score_episodes, gamma, lam)


def score_names(audit: dict[str, np.ndarray]) -> Sequence[str]:
    return [k[2:] for k in audit if k.startswith("s_")]
