"""E5 maze audit: JaxUED-parity of the scores on collected rollouts, and exact regret.

The policies here are a randomly initialised copy of maze_plr.py's own ActorCritic (with a
critic bias so values are not ~0), the BFS-optimal open-loop plan, a per-episode mixture of
the two (so the scored episodes hold successes and failures, R_max > 0 and 0 < p < 1), and
an always-forward copy of the network (deterministic, so its stream can be compared with
maze_plr's own ``sample_trajectories_rnn`` step for step). The levels are the training
generator's plus hand-made ones (an unreachable goal, a corridor, a three-step straight).
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax.training.train_state import TrainState
from jaxued.environments.maze import Level
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss

import apexzero.compat  # noqa: F401  (must precede jaxued calls)
from apexzero.audit.estimators import mna_terms, scoring_gae
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
STRAIGHT = """
.............
.............
.............
.............
.............
.............
.>..G........
.............
.............
.............
.............
.............
.............
"""
GAMMA, LAM = 0.995, 0.98


def _levels(n_generated: int, seed: int = 3, extra=(ENCLOSED, CORRIDOR)) -> Level:
    gen = maze.generate_levels(n_generated, jax.random.PRNGKey(seed))
    extra = [Level.from_str(s) for s in extra]
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


def _mixed_act(params, key, carry, obs, last_done):
    """Per episode, with probability 1/2, the BFS plan's actions, else the network's sampled
    ones; the values are always the network's, so the critic is scored on both outcomes."""
    net_params, plans = params
    hstate, t, follow = carry
    k_net, k_coin = jax.random.split(key)
    hstate, a_net, value = maze._network_act(net_params, k_net, hstate, obs, last_done)
    t = jnp.where(last_done, 0, t)
    follow = jnp.where(t == 0, jax.random.bernoulli(k_coin, 0.5, t.shape), follow)
    a_plan = plans[jnp.arange(plans.shape[0]), jnp.minimum(t, plans.shape[1] - 1)]
    return (hstate, t + 1, follow), jnp.where(follow, a_plan, a_net), value


def _mixed_policy(levels: Level, seed: int = 2) -> maze.Policy:
    init = lambda n: (maze.maze_plr().ActorCritic.initialize_carry((n,)), jnp.zeros(n, jnp.int32),
                      jnp.zeros(n, bool))
    return maze.Policy((_random_params(seed), jnp.asarray(maze.bfs_plans(levels))), init, _mixed_act)


def _episode_returns(ro, episodes):
    """Undiscounted return of each (level, episode), by a plain loop."""
    b = maze.episode_bounds(ro.done, episodes)
    return np.array([[ro.reward[a0:a1, i].sum() for a0, a1 in b[i]] for i in range(len(b))])


@pytest.fixture(scope="module")
def random_rollouts():
    levels = _levels(10)
    ro = maze.collect(maze.network_policy(_random_params()), levels, 3, jax.random.PRNGKey(5))
    return levels, ro


@pytest.fixture(scope="module")
def mixed_rollouts():
    levels = _levels(30, seed=7)
    ro = maze.collect(_mixed_policy(levels), levels, 6, jax.random.PRNGKey(12))
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


def test_level_config_comes_from_the_run():
    cfg = dict(n_walls=60, gamma=0.99, agent_view_size=5, gae_lambda=0.98)
    assert maze.level_config(cfg) == dict(n_walls=60, gamma=0.99, max_steps=250, agent_view_size=5)


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
    long = maze.collect(maze.network_policy(_random_params()), _levels(1), 1, jax.random.PRNGKey(5),
                        min_steps=600)
    assert len(long.done) >= 600


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


@maze.float32
def test_collect_is_sample_trajectories_rnn_step_for_step():
    # The always-forward network is deterministic (the forward logit is 1e4 above the rest), so
    # maze_plr's own rollout and ours must agree exactly although their random keys differ:
    # STRAIGHT ends an episode every 3 steps, the other levels hit the time limit twice.
    params = _random_params(1)
    a1 = params["params"]["actor1"]
    params["params"]["actor1"] = {"kernel": a1["kernel"], "bias": a1["bias"].at[2].set(1e4)}
    levels = _levels(3, extra=(ENCLOSED, CORRIDOR, STRAIGHT))
    n, T, key = int(levels.wall_map.shape[0]), 512, jax.random.PRNGKey(4)
    ro = maze.collect(maze.network_policy(params), levels, 1, key, min_steps=T + 1)
    mod = maze.maze_plr()
    env, ep = maze.make_env()
    obs, state = jax.vmap(env.reset_to_level, (0, 0, None))(
        jax.random.split(jax.random.split(key)[0], n), levels, ep)
    ts = TrainState.create(apply_fn=mod.ActorCritic(7).apply, params=params, tx=optax.identity())
    (_, _, _, _, _, last_value), traj = mod.sample_trajectories_rnn(
        jax.random.PRNGKey(99), env, ep, ts, mod.ActorCritic.initialize_carry((n,)), obs, state, n, T)
    reward, done, value = (np.asarray(traj[i]) for i in (2, 3, 5))
    assert done[:, -1].sum() >= 150 and done[:, :-1].sum(0).min() >= 2
    np.testing.assert_array_equal(done, ro.done[:T])
    np.testing.assert_array_equal(reward, ro.reward[:T])
    assert reward[:, -1].sum() > 100                              # the straight's goal rewards
    np.testing.assert_allclose(value, ro.value[:T], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(last_value), ro.value[T], rtol=1e-5, atol=1e-6)
    # Native scores = on_new_levels on a 256-step rollout. sample_trajectories_rnn is
    # prefix-consistent (same key chain), so its first 256 steps are that rollout, and its
    # value at step 256 is the rollout's bootstrap V(o_256).
    S = 256
    adv, _ = mod.compute_gae(GAMMA, LAM, jnp.asarray(value[S]), jnp.asarray(value[:S]),
                             jnp.asarray(reward[:S]), jnp.asarray(done[:S]))
    d, mr = jnp.asarray(done[:S]), compute_max_returns(jnp.asarray(done[:S]), jnp.asarray(reward[:S]))
    nat = maze.jaxued_native_scores(ro, S, GAMMA, LAM)
    np.testing.assert_allclose(nat["pvl"], positive_value_loss(d, adv), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(nat["maxmc"], max_mc(d, jnp.asarray(value[:S]), mr), rtol=1e-5,
                               atol=1e-7)
    assert np.all(nat["episodes"] >= 1) and nat["episodes"][-1] >= 80


def test_per_step_gae_and_identities_on_random_rollouts(random_rollouts):
    levels, ro = random_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    for m in (1, 2):
        ref = maze.jaxued_scores(ro, m, GAMMA, LAM)
        res = maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=m)
        np.testing.assert_allclose(res["s_pvl_ours"], ref["pvl"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(res["s_maxmc_ours"], ref["maxmc"], rtol=1e-4, atol=1e-6)
        assert np.all(np.isfinite(ref["pvl"])) and np.all(np.isfinite(ref["maxmc"]))
    # Per step: maze_plr.compute_gae over the stream = scoring_gae per completed episode.
    adv = maze.jaxued_scores(ro, 3, GAMMA, LAM)["advantages"]
    for i, eps in enumerate(maze.episode_bounds(ro.done, 3)):
        for a0, a1 in eps:
            n = a1 - a0
            term = np.r_[np.zeros(n - 1, bool), True]
            ours = scoring_gae(ro.reward[a0:a1, i], ro.value[a0:a1, i], term, np.zeros(n, bool),
                               0.0, np.zeros(n, bool), GAMMA, LAM)
            np.testing.assert_allclose(adv[a0:a1, i], ours, rtol=1e-4, atol=1e-5)
    # Our aggregation identities hold on real rollouts too.
    assert np.allclose(res["s_pvl_ours"], 0.5 * res["s_l1"] + 0.5 * maze.level_scores(
        maze.episode_stats(ro, 3, GAMMA), range(2), headline=2)["mean_a"])


def test_scores_equal_jaxued_with_successes_and_failures(mixed_rollouts):
    levels, ro = mixed_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    bounds = maze.episode_bounds(ro.done, 6)
    G = _episode_returns(ro, 6)
    for m in (2, 4):
        ref = maze.jaxued_scores(ro, m, GAMMA, LAM)
        res = maze.audit_rollouts(ro, j_star, k_star, episodes=6, score_episodes=m)
        p = res["success_rate_op"]
        mixed = (p > 0) & (p < 1)
        # The regime that matters: R_max > 0 on many levels, mixed outcomes, and never-solved
        # levels (MaxMC's floor) all present.
        assert (ref["max_return"] > 0).sum() >= 15 and mixed.sum() >= 10 and (p == 0).sum() >= 1
        # JaxUED's R_max is the max *undiscounted* return over the scored episodes, floored at 0.
        np.testing.assert_allclose(ref["max_return"], np.maximum(G[:, :m].max(1), 0), rtol=1e-6)
        np.testing.assert_allclose(res["s_maxmc_ours"], ref["maxmc"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(res["s_pvl_ours"], ref["pvl"], rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(res["s_maxmc"], ref["maxmc"])
        # Learnability p(1 - p) and MNA (gate: any success) by hand, level by level.
        for i in range(len(p)):
            succ = G[i, :m] > 0
            assert res["s_learnability"][i] == pytest.approx(succ.mean() * (1 - succ.mean()))
            g_hat, steps = 0.0, 0
            for a0, a1 in bounds[i, :m]:
                n = a1 - a0
                term, none = np.r_[np.zeros(n - 1, bool), True], np.zeros(n, bool)
                g_hat += mna_terms(ro.reward[a0:a1, i], ro.value[a0:a1, i], term, none, 0.0, none,
                                   GAMMA, LAM).sum()
                steps += n
            assert res["s_mna"][i] == pytest.approx(g_hat / steps if succ.any() else 0.0,
                                                    rel=1e-9, abs=1e-12)
        assert np.all(res["s_learnability"][mixed] > 0) and np.all(res["s_learnability"] <= 0.25)
        assert np.all(res["s_mna"][p == 0] == 0) and np.any(res["s_mna"][p > 0] != 0)
    # Normalised regret: on reachable goals only.
    reach = j_star > 0
    np.testing.assert_allclose(res["regret_norm"][reach], res["regret"][reach] / j_star[reach])
    assert np.isnan(res["regret_norm"][~reach]).all() and (~reach).sum() >= 1


def test_masking_dones_equals_a_rollout_that_stopped_after_k_episodes(mixed_rollouts):
    # jaxued_scores masks every done after the k-th; JaxUED on a rollout that really ended at
    # the k-th episode end must give the same PVL and MaxMC (its last value is then unused).
    levels, ro = mixed_rollouts
    mod = maze.maze_plr()
    n = ro.done.shape[1]
    with jax.enable_x64(False):
        for k in (1, 3):
            res = maze.jaxued_scores(ro, k, GAMMA, LAM)
            for i in [*range(5), n - 2, n - 1]:
                stop = np.nonzero(ro.done[:, i])[0][k - 1] + 1
                v, r, d = (jnp.asarray(a[:stop, i:i + 1]) for a in (ro.value, ro.reward, ro.done))
                adv, _ = mod.compute_gae(GAMMA, LAM, jnp.full(1, 123.0), v, r, d)
                assert res["pvl"][i] == pytest.approx(float(positive_value_loss(d, adv)[0]),
                                                      rel=1e-6, abs=1e-7)
                assert res["maxmc"][i] == pytest.approx(
                    float(max_mc(d, v, compute_max_returns(d, r))[0]), rel=1e-6, abs=1e-7)


def test_native_scores_and_minus_inf(mixed_rollouts):
    levels, ro = mixed_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    # A 100-step window: plan-following first episodes finish in it, timed-out ones do not.
    nat = maze.jaxued_native_scores(ro, 100, GAMMA, LAM)
    none = nat["episodes"] == 0
    assert none.sum() >= 3 and (~none).sum() >= 3
    assert np.all(np.isneginf(nat["pvl"]) == none) and np.all(np.isneginf(nat["maxmc"]) == none)
    np.testing.assert_array_equal(nat["episodes"], ro.done[:100].sum(0))
    res = maze.audit_rollouts(ro, j_star, k_star, episodes=6, score_episodes=4, native=ro,
                              num_steps=100)
    np.testing.assert_array_equal(res["s_pvl_native"], nat["pvl"])
    np.testing.assert_array_equal(res["native_episodes"], nat["episodes"])
    with pytest.raises(ValueError, match="num_steps"):
        maze.jaxued_native_scores(ro, ro.done.shape[0], GAMMA, LAM)


def test_episode_stats_vectorised_returns_match_a_per_episode_loop(mixed_rollouts):
    _, ro = mixed_rollouts
    st = maze.episode_stats(ro, 6, GAMMA, scored=[0, 1])
    full = maze.episode_stats(ro, 6, GAMMA)
    for i, eps in enumerate(maze.episode_bounds(ro.done, 6)):
        for w, (a0, a1) in enumerate(eps):
            r, v = ro.reward[a0:a1, i].astype(float), ro.value[a0:a1, i].astype(float)
            assert st.T[i, w] == a1 - a0 and st.G[i, w] == pytest.approx(r.sum(), abs=1e-12)
            assert st.G_disc[i, w] == pytest.approx(np.sum(GAMMA ** np.arange(a1 - a0) * r),
                                                    rel=1e-12, abs=1e-14)
            assert st.V_first[i, w] == v[0]
            assert st.V_bar[i, w] == pytest.approx(v.mean(), rel=1e-12, abs=1e-12)
    assert np.isnan(st.pvl[:, :, 2:]).all() and np.isnan(st.mna_sum[:, 2:]).all()
    np.testing.assert_array_equal(st.pvl[:, :, :2], full.pvl[:, :, :2])
    np.testing.assert_array_equal(st.mna_sum[:, :2], full.mna_sum[:, :2])


def test_random_policy_regret_is_nonnegative(random_rollouts):
    levels, ro = random_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    res = maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=1)
    assert np.all(res["regret"] >= -1e-6)
    assert j_star[-2] == 0.0 and res["regret"][-2] == 0.0          # unreachable goal: no regret


def test_audit_refuses_an_in_sample_target(random_rollouts):
    levels, ro = random_rollouts
    j_star, k_star = maze.bfs_optimum(levels)
    for m in (0, 3, 4):
        with pytest.raises(ValueError, match="score_episodes"):
            maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=m)


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
    # The env/oracle guard: claim a J* 0.01 below what the plan attains on one level.
    bad = j_star.copy()
    bad[np.argmax(j_star > 0)] -= 0.01
    with pytest.raises(AssertionError, match="beat the BFS optimum"):
        maze.audit_rollouts(ro, bad, k_star, episodes=4, score_episodes=2)


def test_a_detour_costs_exactly_its_discounted_time():
    # Left, right (back to the start state), then the BFS plan: k + 2 actions, so
    # J^pi = gamma^(k+1) (1 - 0.9 (k+2) / 250) and regret = J* - J^pi > 0, exactly.
    levels = _levels(20, seed=5)
    j_star, k_star = maze.bfs_optimum(levels)
    plans = maze.bfs_plans(levels)
    detour = np.concatenate([np.tile([0, 1], (len(plans), 1)), plans[:, :-2]], 1)
    ro = maze.collect(maze.plan_policy(detour), levels, 3, jax.random.PRNGKey(6))
    res = maze.audit_rollouts(ro, j_star, k_star, episodes=3, score_episodes=1)
    ok = j_star > 0
    k = k_star[ok]
    expected = j_star[ok] - GAMMA ** (k + 1) * (1 - 0.9 * (k + 2) / 250)
    np.testing.assert_allclose(res["regret"][ok], expected, atol=1e-6)
    assert np.all(res["regret"][ok] > 1e-3) and np.all(res["regret"][~ok] == 0)


def test_checkpoint_round_trip_and_audit(tmp_path):
    params = _random_params(4)
    state = dict(params=params, num_dr_updates=7, num_replay_updates=5, num_mutation_updates=0,
                 sampler=dict(scores=jnp.zeros(3), size=jnp.asarray(0)))
    cfg = dict(gamma=0.99, gae_lambda=0.98, agent_view_size=5, n_walls=25, num_steps=256)
    (tmp_path / "config.json").write_text(json.dumps(cfg))
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
    # End to end: J* uses the run's gamma (0.99, not the default), native scores are present.
    res = maze.audit_checkpoint(ck, levels, jax.random.PRNGKey(3), episodes=2, score_episodes=1)
    np.testing.assert_array_equal(res["j_star"], maze.bfs_optimum(levels, 0.99)[0])
    assert not np.allclose(res["j_star"], maze.bfs_optimum(levels)[0])
    assert np.all(res["native_episodes"] >= 1) and np.all(np.isfinite(res["s_maxmc_native"]))
