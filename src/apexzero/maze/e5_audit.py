"""Maze E5's audit (docs/prereg/maze-e5.md Section 3): the pilot's protocol at E5's size.

It calls the pilot's functions in ``apexzero.maze.audit`` unchanged (``collect``,
``audit_rollouts``, ``episode_stats``), so with E5's additions off and the pilot's settings
it computes exactly what ``audit_checkpoint`` does (tests/test_maze_e5.py). The additions:

- **Settings** from the run's own config (``settings``): JaxUED's config.json or SFL's
  composed config; J* uses the run's γ and the 250-step limit, the scores its γ and λ.
- **The target's solve rate** per maze (episodes M+1..W), which fixes a checkpoint's phase,
  and the odd and even target episodes' mean returns (the reliability split).
- **SFL's native learnability** (``sfl_native``): one independent 2,000-step AutoReplay
  stream per maze from a reset with a fresh carry, the first 10 completed episodes counted
  and the open last one dropped (``apexzero.maze.sfl.native_learnability``), checked against
  the line-for-line port of SFL's counting on the first maze streams.
- **The in-buffer audit** (``inbuffer_levels``, ``audit_inbuffer``): PLR's valid buffer
  entries (a fixed-key uniform 1,000 if there are more) with their stored scores and
  staleness, or SFL's learnable set D with its learnability and the score age; their current
  regret from BFS and 36-episode streams, as for the fresh mazes.
"""

from __future__ import annotations

import time

import jax
import numpy as np

from apexzero.maze import audit as maze
from apexzero.maze import sfl

LEVELS = 2000
LEVEL_SEED = 2027
EPISODES = 36
SCORE_EPISODES = 4
SFL_ROLLOUT = 2000
SFL_MAX_EPISODES = 10
INBUF_MAX = 1000
AUDIT_SEED = 20271
JAXUED_WALLS = 25
CHECK_LEVELS = 64          # streams compared with the line-for-line port of SFL's counting


def settings(arm: str, config: dict) -> dict:
    """gamma, lam, num_steps, agent_view_size, n_walls and max_steps of a run."""
    if arm == "sfl":
        s = sfl.run_settings(config.get("config", config))
        out = dict(gamma=s["gamma"], lam=s["gae_lambda"], num_steps=s["num_steps"],
                   agent_view_size=s["agent_view_size"], n_walls=s["n_walls"])
    else:
        out = dict(gamma=float(config["gamma"]), lam=float(config["gae_lambda"]),
                   num_steps=int(config["num_steps"]),
                   agent_view_size=int(config["agent_view_size"]),
                   n_walls=int(config["n_walls"]))
    _, env_params = maze.make_env(out["agent_view_size"])
    out["max_steps"] = int(env_params.max_steps_in_episode)
    return out


def fresh_levels(n: int = LEVELS, seed: int = LEVEL_SEED, n_walls: int = JAXUED_WALLS):
    """The shared audit mazes: JaxUED's default generator (25 walls), level seed 2027. SFL's
    own generator (60 walls) gives its secondary set."""
    return maze.generate_levels(n, jax.random.PRNGKey(seed), n_walls)


def audit_key(run_index: int, update: int) -> jax.Array:
    return jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(AUDIT_SEED), run_index),
                              update)


def target_stats(ro: maze.MazeRollouts, episodes: int, score_episodes: int,
                 gamma: float) -> dict[str, np.ndarray]:
    st = maze.episode_stats(ro, episodes, gamma, scored=[])
    tgt = slice(score_episodes, episodes)
    return dict(success_rate_tgt=st.completed[:, tgt].mean(-1),
                j_pi_odd=st.G_disc[:, score_episodes::2].mean(-1),
                j_pi_even=st.G_disc[:, score_episodes + 1::2].mean(-1))


def sfl_native_scores(policy: maze.Policy, levels, key, view: int,
                      chunk: int) -> dict[str, np.ndarray]:
    """SFL's own learnability for every level, from its own independent stream."""
    ro = maze.collect(policy, levels, 0, key, agent_view_size=view, chunk=chunk,
                      min_steps=SFL_ROLLOUT)
    nat = sfl.native_learnability(ro.done, ro.reward, SFL_ROLLOUT, SFL_MAX_EPISODES)
    m = min(CHECK_LEVELS, ro.done.shape[1])
    ref = sfl.sfl_outcomes_reference(SFL_ROLLOUT, ro.done[:SFL_ROLLOUT, :m],
                                     ro.reward[:SFL_ROLLOUT, :m])
    if not (np.array_equal(ref["num_episodes"], nat["episodes"][:m])
            and np.allclose(ref["learnability"], nat["learnability"][:m], atol=1e-6)):
        raise AssertionError("SFL's native learnability differs from the port of its counting")
    return dict(s_learnability_native=nat["learnability"],
                native_sfl_episodes=nat["episodes"].astype(float),
                native_sfl_success=nat["success_rate"])


def audit_policy(policy: maze.Policy, st: dict, levels, oracle, key, episodes: int = EPISODES,
                 score_episodes: int = SCORE_EPISODES, sfl_native: bool = True) -> dict:
    """One checkpoint on ``levels``: ``audit_checkpoint``'s streams and scores, plus the E5
    additions. With ``sfl_native`` off and the pilot's settings, the ``audit_rollouts`` part
    is exactly ``audit_checkpoint``'s."""
    j_star, k_star = oracle
    kw = dict(agent_view_size=st["agent_view_size"], chunk=st["num_steps"] + 1)
    ro = maze.collect(policy, levels, episodes, key, **kw)
    native = maze.collect(policy, levels, 0, jax.random.fold_in(key, 1),
                          min_steps=st["num_steps"] + 1, **kw)
    out = maze.audit_rollouts(ro, j_star, k_star, episodes, score_episodes, st["gamma"],
                              st["lam"], native=native, num_steps=st["num_steps"])
    if max(np.abs(out["s_pvl"] - out["s_pvl_ours"]).max(),
           np.abs(out["s_maxmc"] - out["s_maxmc_ours"]).max()) > 1e-6:
        raise AssertionError("PVL or MaxMC differs from JaxUED's functions by more than 1e-6")
    out.update(target_stats(ro, episodes, score_episodes, st["gamma"]))
    if sfl_native:
        out.update(sfl_native_scores(policy, levels, jax.random.fold_in(key, 2),
                                     st["agent_view_size"], kw["chunk"]))
    return out


def inbuffer_levels(arm: str, ck, run_index: int, update: int):
    """(levels, stored score, age) of the buffer in force at a checkpoint, or None for DR.
    PLR: the valid entries of the sampler state, a fixed-key uniform 1,000 if there are more;
    age is the staleness (the sampler's counter minus the entry's timestamp). SFL: D, its
    learnability from the last refresh, and the updates since that refresh."""
    if arm == "dr":
        return None
    if arm == "sfl":
        d = ck.learnable
        n = len(d["scores"])
        return (sfl.learnable_levels(d), np.asarray(d["scores"], float),
                np.full(n, float(int(d["update"]) - int(d["refresh_update"]))))
    s = ck.sampler
    size = int(np.asarray(s["size"]))
    idx = np.arange(size)
    if size > INBUF_MAX:
        rng = np.random.default_rng([AUDIT_SEED, 2, run_index, update])
        idx = np.sort(rng.choice(size, INBUF_MAX, replace=False))
    levels = jax.tree.map(lambda x: np.asarray(x)[idx], s["levels"])
    from jaxued.environments.maze import Level

    lv = Level(**{k: jax.numpy.asarray(levels[k]) for k in sfl.LEVEL_FIELDS})
    stale = float(np.asarray(s["episode_count"])) - np.asarray(s["timestamps"], float)[idx]
    return lv, np.asarray(s["scores"], float)[idx], stale


def audit_inbuffer(policy: maze.Policy, st: dict, levels, stored, age, key,
                   episodes: int = EPISODES, score_episodes: int = SCORE_EPISODES) -> dict:
    """Current regret on the buffer's levels (BFS and 36-episode streams, the target from
    episodes M+1..W), with each level's stored score and age."""
    j_star, k_star = maze.bfs_optimum(levels, st["gamma"], st["max_steps"])
    ro = maze.collect(policy, levels, episodes, key, agent_view_size=st["agent_view_size"],
                      chunk=st["num_steps"] + 1)
    es = maze.episode_stats(ro, episodes, st["gamma"], scored=[])
    if np.any(es.G_disc > j_star[:, None] + 1e-5):
        raise AssertionError("an in-buffer episode beat the BFS optimum")
    j_pi = es.G_disc[:, score_episodes:episodes].mean(-1)
    reach = j_star > 0
    regret = j_star - j_pi
    return dict(inbuf_stored=np.asarray(stored, float), inbuf_age=np.asarray(age, float),
                inbuf_j_star=j_star, inbuf_k_star=k_star.astype(float), inbuf_j_pi=j_pi,
                inbuf_regret=regret,
                inbuf_regret_norm=np.where(reach, regret / np.where(reach, j_star, 1.0), np.nan),
                inbuf_reachable=reach, inbuf_stream_steps=np.full(len(j_star), ro.done.shape[0]),
                inbuf_success_tgt=es.completed[:, score_episodes:episodes].mean(-1))


def timed(fn, *args, **kwargs):
    t0, c0 = time.time(), time.process_time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0, time.process_time() - c0
