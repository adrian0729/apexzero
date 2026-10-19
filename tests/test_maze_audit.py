"""E5 maze audit: JaxUED-parity of the scores on collected rollouts, and exact regret.

The policies here are a randomly initialised copy of maze_plr.py's own ActorCritic (with a
critic bias so values are not ~0) and the BFS-optimal open-loop plan; the levels are the
training generator's plus two hand-made ones (an unreachable goal and a corridor).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from jaxued.environments.maze import Level

import apexzero.compat  # noqa: F401  (must precede jaxued calls)
from apexzero.audit.estimators import scoring_gae
from apexzero.maze import audit as maze
from apexzero.maze.oracle import min_actions_to_goal, optimal_plan

ENCLOSED = """
.............
.>...........
.............
.............
.............
.............
.......#.....
......#G#....
.......#.....
.............
.............
.............
.............
"""
CORRIDOR = """
#############
#############
#############
#############
#############
#############
#<.........G#
#############
#############
#############
#############
#############
#############
"""


def _levels(n_generated: int, seed: int = 3) -> Level:
    gen = maze.generate_levels(n_generated, jax.random.PRNGKey(seed))
    extra = [Level.from_str(s) for s in (ENCLOSED, CORRIDOR)]
    extra = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *extra)
    return jax.tree.map(lambda a, b: jnp.concatenate([jnp.asarray(a), b.astype(jnp.asarray(a).dtype)]),
                        gen, extra)


@maze.float32
def _random_params(seed: int = 0, value_bias: float = 0.3):
    key = jax.random.PRNGKey(seed)
    mod = maze.maze_plr()
    env, ep = maze.make_env()
    lvl = maze.generate_levels(1, key)
    obs, _ = jax.vmap(env.reset_to_level, (0, 0, None))(jax.random.split(key, 1), lvl, ep)
    x = (jax.tree.map(lambda a: a[None], obs), jnp.zeros((1, 1), bool))
    params = mod.ActorCritic(7).init(key, x, mod.ActorCritic.initialize_carry((1,)))
    bias = params["params"]["critic1"]["bias"]
    params["params"]["critic1"]["bias"] = bias + value_bias
    return params


@pytest.fixture(scope="module")
def random_rollouts():
    levels = _levels(10)
    ro = maze.collect(maze.network_policy(_random_params()), levels, 3, jax.random.PRNGKey(5))
    return levels, ro


def test_optimal_plan_matches_bfs_length():
    walls = np.zeros((3, 5), bool)
    assert optimal_plan(walls, (0, 1), 0, (4, 1)) == [2, 2, 2, 2]
    assert optimal_plan(walls, (0, 1), 2, (4, 1)) == [0, 0, 2, 2, 2, 2]
    walls[:, 2] = True
    assert optimal_plan(walls, (0, 1), 0, (4, 1)) is None
    lv = maze.generate_levels(40, jax.random.PRNGKey(11))
    for i in range(40):
        args = (np.asarray(lv.wall_map[i]), np.asarray(lv.agent_pos[i]), int(lv.agent_dir[i]),
                np.asarray(lv.goal_pos[i]))
        plan, k = optimal_plan(*args), min_actions_to_goal(*args)
        assert (plan is None and k is None) or len(plan) == k


def test_episode_masks():
    done = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1]], bool)
    assert (maze.first_dones(done, 2).sum(0) == [2, 2]).all()
    assert not maze.first_dones(done, 2)[5].any()                 # third episode ends dropped
    b = maze.episode_bounds(done, 2)
    assert b[0].tolist() == [[0, 2], [2, 4]] and b[1].tolist() == [[0, 1], [1, 3]]


def test_stream_has_w_complete_episodes_and_resets(random_rollouts):
    levels, ro = random_rollouts
    assert ro.reward.shape == ro.value.shape == ro.done.shape
    assert (ro.done.sum(0) >= 3).all()
    b = maze.episode_bounds(ro.done, 3)
    lengths = b[..., 1] - b[..., 0]
    assert lengths.max() <= 250 and lengths.min() >= 1
    # A reward is earned only on the step that ends an episode (sparse goal reward).
    assert np.all(ro.reward[~ro.done] == 0)
    # The corridor is solved only by turning around (two turns + 10 forward moves).
    assert lengths[-1].min() >= 12


def test_chunking_does_not_change_the_stream():
    levels = _levels(4)
    pol = maze.network_policy(_random_params())
    a = maze.collect(pol, levels, 2, jax.random.PRNGKey(9), chunk=250)
    b = maze.collect(pol, levels, 2, jax.random.PRNGKey(9), chunk=60)
    T = min(a.done.shape[0], b.done.shape[0])
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x[:T], y[:T])


def test_global_x64_does_not_change_levels_or_rollouts():
    # Other test modules (and possibly a combined script) switch x64 on process-wide.
    pol = maze.network_policy(_random_params())
    with jax.enable_x64(False):
        lv32 = maze.generate_levels(3, jax.random.PRNGKey(8))
        ro32 = maze.collect(pol, lv32, 1, jax.random.PRNGKey(2))
    with jax.enable_x64(True):
        lv64 = maze.generate_levels(3, jax.random.PRNGKey(8))
        ro64 = maze.collect(pol, lv64, 1, jax.random.PRNGKey(2))
    jax.tree.map(np.testing.assert_array_equal, lv32, lv64)
    for x, y in zip(ro32, ro64):
        np.testing.assert_array_equal(x, y)
        assert x.dtype == y.dtype


def test_scores_equal_jaxued_utils_on_the_collected_rollouts(random_rollouts):
    levels, ro = random_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    for m in (1, 2):
        ref = maze.jaxued_scores(ro, m, 0.995, 0.98)
        res = maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=m)
        # PVL / MaxMC from our per-episode primitives = JaxUED's compute_gae + jaxued.utils.
        np.testing.assert_allclose(res["s_pvl_ours"], ref["pvl"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(res["s_maxmc_ours"], ref["maxmc"], rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(res["s_pvl"], ref["pvl"])
        assert np.all(np.isfinite(ref["pvl"])) and np.all(np.isfinite(ref["maxmc"]))
    # Per step: maze_plr.compute_gae over the stream = scoring_gae per completed episode.
    adv = maze.jaxued_scores(ro, 3, 0.995, 0.98)["advantages"]
    for i, eps in enumerate(maze.episode_bounds(ro.done, 3)):
        for a0, a1 in eps:
            n = a1 - a0
            term = np.r_[np.zeros(n - 1, bool), True]
            ours = scoring_gae(ro.reward[a0:a1, i], ro.value[a0:a1, i], term, np.zeros(n, bool),
                               0.0, np.zeros(n, bool), 0.995, 0.98)
            np.testing.assert_allclose(adv[a0:a1, i], ours, rtol=1e-4, atol=1e-5)
    # Our aggregation identities hold on real rollouts too.
    assert np.allclose(res["s_pvl_ours"], 0.5 * res["s_l1"] + 0.5 * maze.level_scores(
        maze.episode_stats(ro, 3, 0.995), range(2), headline=2)["mean_a"])
    assert np.all((res["s_learnability"] >= 0) & (res["s_learnability"] <= 0.25))


def test_random_policy_regret_is_nonnegative(random_rollouts):
    levels, ro = random_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    res = maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=1)
    assert np.all(res["regret"] >= -1e-6)
    assert j_star[-2] == 0.0 and res["regret"][-2] == 0.0          # unreachable goal: no regret


def test_bfs_plan_policy_attains_j_star_exactly():
    levels = _levels(25)
    j_star, k_star = maze.bfs_optimum(levels)
    assert (j_star == 0).sum() >= 1 and (j_star > 0).sum() >= 20
    ro = maze.collect(maze.plan_policy(maze.bfs_plans(levels)), levels, 4, jax.random.PRNGKey(1))
    res = maze.audit_rollouts(ro, j_star, k_star, episodes=4, score_episodes=2)
    np.testing.assert_allclose(res["j_pi"], j_star, atol=1e-6)
    np.testing.assert_allclose(res["j_pi_all"], j_star, atol=1e-6)
    assert np.all(np.abs(res["regret"]) < 1e-6) and np.all(res["regret"] >= -1e-6)
    lengths = np.diff(maze.episode_bounds(ro.done, 4), axis=-1)[..., 0]
    assert np.all(lengths[j_star > 0] == k_star[j_star > 0, None])
    assert np.all(res["success_rate"][j_star > 0] == 1) and np.all(res["success_rate"][j_star == 0] == 0)


def test_checkpoint_round_trip(tmp_path):
    params = _random_params(4)
    state = dict(params=params, num_dr_updates=7, num_replay_updates=5, num_mutation_updates=0,
                 sampler=dict(scores=jnp.zeros(3), size=jnp.asarray(0)))
    (tmp_path / "config.json").write_text(json.dumps(dict(gamma=0.995, gae_lambda=0.98,
                                                          agent_view_size=5)))
    mgr = ocp.CheckpointManager(tmp_path / "models",
                                options=ocp.CheckpointManagerOptions(save_interval_steps=1))
    for step in (0, 1):
        mgr.save(step, args=ocp.args.StandardSave(state))
    mgr.wait_until_finished()
    mgr.close()
    assert maze.checkpoint_steps(tmp_path) == [0, 1]
    ck = maze.load_checkpoint(tmp_path)
    assert ck.step == 1 and ck.num_updates == 12 and ck.config["gae_lambda"] == 0.98
    jax.tree.map(np.testing.assert_array_equal, ck.params, params)
    # The restored params drive the vendored ActorCritic like the originals do.
    levels = _levels(2)
    a = maze.collect(maze.network_policy(ck.params), levels, 1, jax.random.PRNGKey(2))
    b = maze.collect(maze.network_policy(params), levels, 1, jax.random.PRNGKey(2))
    np.testing.assert_array_equal(a.value, b.value)
