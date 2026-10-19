"""End-to-end smoke runs of the curriculum C trainer on a tiny pool (64 levels, 64 slots,
short windows) of the real racing env, checking the sampler invariants chunk by chunk:
no non-finite score in the buffer, scores written only from completed episodes,
exploratory windows masked out of every loss and normaliser, SFL's D in use."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.agents.ppo import Actor, Critic, PPOConfig, RunningMeanStd, init_agent
from apexzero.audit.rollouts import Snapshot, rollout_windows
from apexzero.env import racing
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar
from apexzero.tracks.levels import make_levels
from apexzero.tracks.track import load_tumftm
from apexzero.ued import pool, sfl
from apexzero.ued.plr import (
    KIND_DR,
    KIND_EXPLORE,
    KIND_REPLAY,
    KIND_SFL,
    OP_INSERT,
    OP_NOOP,
    OP_UPDATE,
    OUT_DROPPED,
    OUT_DUPLICATE,
    OUT_INSERTED,
    OUT_UPDATED,
    Assignment,
    PLRConfig,
)
from apexzero.ued.sfl import SFLConfig
from apexzero.ued.trainer import UEDConfig, UEDTrainer

TUMFTM = Path("data/tracks/tumftm")
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
N_LEVELS, N_ENVS, CHUNK, WINDOW = 64, 64, 16, 40
PARAMS = racing.EnvParams(window_steps=WINDOW)
CFG = PPOConfig(num_envs=N_ENVS, num_steps=CHUNK, total_timesteps=N_ENVS * CHUNK * 8, network="mlp",
                actor_width=32, critic_width=32, mlp_layers=2, ensemble_heads=2, num_minibatches=4,
                update_epochs=2)


@pytest.fixture(scope="module", autouse=True)
def float32_like_training():
    """Several test modules switch JAX to float64 when imported; training runs in float32."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture(scope="module")
def pool_setup():
    g = fit_grammar([load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV])
    lv = make_levels(jax.random.PRNGKey(7), "pool", n=N_LEVELS)
    return pool.make_bank(lv, g, params=PARAMS), pool.level_pytree(lv)


def _same(a, b) -> bool:
    return all(jax.tree.leaves(jax.tree.map(lambda x, y: bool(jnp.array_equal(x, y)), a, b)))


def test_pool_env_resets_to_the_assigned_level(pool_setup):
    bank, _ = pool_setup
    st0 = pool.start_state(bank.x0[3], bank.ctx[3], bank.hidden_mu[3])
    tr = jax.tree.map(lambda a: a[3], bank.tracks)
    direct = racing.observe(st0, tr, racing.params_from_context(st0.ctx, hidden_mu=st0.hidden_mu))
    assert np.allclose(pool.reset_to(3, bank)[0], direct, atol=1e-5)

    bank = pool.with_source(bank, 1, [5])
    env = pool.pool_env(PARAMS)
    obs, st = pool.init_slots(bank, [0, 1, 2], [9, 9, 9], [-1, 1, 0], [True, False, True],
                              [False, True, True])
    st = st._replace(env=st.env._replace(t=jnp.full(3, WINDOW - 1, jnp.int32)))   # truncate next step
    out = jax.vmap(env.step, in_axes=(0, 0, 0, None))(jax.random.split(jax.random.PRNGKey(0), 3), st,
                                                      jnp.zeros((3, 2)), bank)
    nobs, nst, r, term, trunc, info = out
    assert bool(trunc.all()) and not bool(term.any()) and not bool(info["end_of_strip"].any())
    assert nst.level[0] == 9 and nst.level[1] == 5 and 0 <= int(nst.level[2]) < N_LEVELS
    assert nst.mask.tolist() == [False, True, True]                  # took next_mask
    assert np.allclose(nobs, bank.obs0[nst.level]) and int(nst.env.t.max()) == 0
    assert not np.allclose(info["final_obs"], nobs)


def _check_plr_chunk(k, before, cur, st, log, trainer):
    rec, mask, nxt = log.records, log.mask, log.nxt
    buf = st.buffer
    size = int(buf["size"])
    scores = np.asarray(buf["scores"])
    # No -inf (or NaN) ever admitted; empty entries keep -inf; levels are unique.
    assert np.all(np.isfinite(scores[:size])) and np.all(scores[size:] == -np.inf)
    ids = np.asarray(buf["levels"]["level_id"][:size])
    assert len(set(ids.tolist())) == size

    # Every transition of a replay window is trained on, no other.
    for j in range(N_ENVS):
        ts = np.sort(rec["t"][rec["slot"] == j])
        first = CHUNK if ts.size == 0 else ts[0] + 1
        assert np.all(mask[:first, j] == (cur.kind[j] == KIND_REPLAY))
        assert np.all(mask[first:, j] == (nxt.kind[j] == KIND_REPLAY))

    # One write per finished write group, built only from completed episodes of this chunk.
    n_rec = np.bincount(rec["slot"], minlength=N_ENVS)
    writes = np.isin(log.events.op, (OP_INSERT, OP_UPDATE))
    assert writes.sum() == (n_rec >= 1).sum() + (n_rec >= 2).sum()
    assert log.events.n[writes].sum() == rec["slot"].size

    # Entries whose score, level or extras changed were written in this chunk, by a write.
    changed = np.asarray(before["levels"]["level_id"]) != np.asarray(buf["levels"]["level_id"])
    changed |= np.asarray(before["scores"]) != scores
    for key in buf["levels_extra"]:
        a, b = np.asarray(before["levels_extra"][key]), np.asarray(buf["levels_extra"][key])
        changed |= ~((a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else (a == b))
    ok = np.isin(log.outcomes["outcome"], (OUT_INSERTED, OUT_DUPLICATE, OUT_UPDATED))
    written = set(log.outcomes["index"][ok].tolist())
    assert set(np.nonzero(changed)[0].tolist()) <= written
    assert np.all(np.asarray(buf["levels_extra"]["score_chunk"])[list(written)] == k)
    for i in written:                                   # the last write to an index wins
        last = np.nonzero(ok & (log.outcomes["index"] == i))[0][-1]
        assert scores[i] == pytest.approx(log.outcomes["score"][last])

    # The admission log: every record points to its group's write, and the per-event
    # transition counts add up to the records'.
    rows = log.event_rows
    assert rows["row"].size == int((log.events.op != OP_NOOP).sum())
    assert np.all(np.isin(rows["op"][rec["write_row"]], (OP_INSERT, OP_UPDATE)))
    assert np.array_equal(rec["write_outcome"], rows["outcome"][rec["write_row"]])
    assert np.array_equal(rec["write_score"], rows["score"][rec["write_row"]], equal_nan=True)
    assert np.all(rows["level_id"][rec["write_row"]] == rec["level"])
    assert rows["steps"].sum() == rec["T"].sum()
    replay = rec["kind"] == KIND_REPLAY
    m = log.metrics
    assert m["replay_steps_scored"] + m["replay_steps_dropped"] == rec["T"][replay].sum()
    dropped = rec["write_outcome"] == OUT_DROPPED
    assert m["replay_steps_dropped"] == rec["T"][dropped].sum() and np.all(replay[dropped])


def _agent_parts(st):
    return (st.agent.actor, st.agent.critic, st.agent.obs_rms, st.agent.ret_rms)


def _evict(st, j, tree):
    """Overwrite the buffer entry slot j is replaying with a pool level not in the buffer
    (what an exploratory insert does), keeping its score."""
    buf, i = st.buffer, int(st.cur.buf_idx[j])
    ids = set(np.asarray(buf["levels"]["level_id"][: int(buf["size"])]).tolist())
    new = next(lv for lv in range(N_LEVELS) if lv not in ids)
    levels = jax.tree.map(lambda a, b: a.at[i].set(b[new]), buf["levels"], tree)
    st.buffer = {**buf, "levels": levels}


def test_plr_perp_smoke(pool_setup):
    bank, tree = pool_setup
    ued = UEDConfig(arm="plr-pvl", plr=PLRConfig(capacity=16))
    trainer = UEDTrainer(CFG, ued, bank, tree, PARAMS)
    st = trainer.init(jax.random.PRNGKey(0))
    agent0 = st.agent
    saw_replay = saw_update = False
    evicted = None                                     # (slot, level) of a forced eviction
    for k in range(8):
        if evicted is None and k >= 3 and (st.cur.kind == KIND_REPLAY).any():
            j = int(np.nonzero(st.cur.kind == KIND_REPLAY)[0][0])
            evicted = (j, int(st.cur.level_id[j]))
            _evict(st, j, tree)
        before, cur, parts = st.buffer, st.cur, _agent_parts(st)
        st, log = trainer.step(st)
        _check_plr_chunk(k, before, cur, st, log, trainer)
        if not log.mask.any():
            # Every window exploratory (chunk 0: the buffer starts empty): PLR-perp neither
            # trains nor moves the normalisers.
            assert _same(parts, _agent_parts(st))
        if evicted is not None and evicted[0] in log.records["slot"]:
            # DEV-18 at the trainer: the evicted replay was played and trained on to its
            # end, and its score write was dropped.
            j, lv = evicted
            first = np.nonzero(log.records["slot"] == j)[0]
            first = first[np.argmin(log.records["t"][first])]
            if log.records["level"][first] == lv and log.records["kind"][first] == KIND_REPLAY:
                assert log.mask[: log.records["t"][first] + 1, j].all()
                assert log.records["write_outcome"][first] == OUT_DROPPED
                assert log.metrics["replay_steps_dropped"] >= log.records["T"][first]
                evicted = (-1, -1)
        saw_replay |= bool((log.records["kind"] == KIND_REPLAY).any())
        saw_update |= bool((log.outcomes["outcome"] == OUT_UPDATED).any())
        m = log.metrics
        assert np.isfinite(m["explained_var"]) == bool(log.mask.any())
        for key, v in m.items():
            per_kind = f"{key.split('_')[0]}_episodes"
            if isinstance(v, float) and key != "explained_var":
                assert np.isfinite(v) == (m.get(per_kind, 1) > 0), key
    assert saw_replay and saw_update and int(st.buffer["size"]) == 16
    assert evicted == (-1, -1)                       # the forced eviction was checked
    assert not _same(agent0.actor.params, st.agent.actor.params)
    snap = trainer.sampler_snapshot(st)
    assert snap["level_id"].size == 16 and np.all(np.isfinite(snap["scores"]))

    # An all-exploratory chunk from a trained state: still bit-identical networks, optimiser
    # state and normalisers. The counts here are integers, for which the plain merge happens
    # to round-trip, so the normalisers get a count it does not (var * c / c != var).
    c = jnp.float32(729655.7)
    st.agent = st.agent._replace(obs_rms=st.agent.obs_rms._replace(count=c),
                                 ret_rms=RunningMeanStd(jnp.float32(-2.3250308), jnp.float32(0.4324972), c))
    n = N_ENVS
    explore = Assignment(np.full(n, KIND_EXPLORE, np.int32), np.arange(n, dtype=np.int32) % N_LEVELS,
                         np.full(n, -1, np.int32))
    st.cur = explore._replace(level_id=st.cur.level_id)
    st.env_state = st.env_state._replace(mask=jnp.zeros(n, bool), next_mask=jnp.zeros(n, bool))
    trainer.plr.predraw = lambda buf, key, n: explore
    before, cur, parts = st.buffer, st.cur, _agent_parts(st)
    st, log = trainer.step(st)
    assert not log.mask.any()
    _check_plr_chunk(8, before, cur, st, log, trainer)
    assert _same(parts, _agent_parts(st))
    assert not _same(agent0.ret_rms, st.agent.ret_rms)


def test_sfl_smoke(pool_setup):
    bank, tree = pool_setup
    ued = UEDConfig(arm="sfl", sfl=SFLConfig(refresh_every=2, windows=2, buffer_size=8, level_chunk=64))
    trainer = UEDTrainer(CFG, ued, bank, tree, PARAMS)
    st = trainer.init(jax.random.PRNGKey(1))
    seen_d = set(st.refresh.buffer.tolist())
    for k in range(5):
        st, log = trainer.step(st)
        d = st.refresh.buffer
        assert d.size == 8 and len(set(d.tolist())) == 8
        assert np.array_equal(np.asarray(st.bank.sources[1][:8]), d) and int(st.bank.source_len[1]) == 8
        assert st.refresh_chunk == 2 * (k // 2)
        ell = st.refresh.learnability
        assert np.all((ell >= 0) & (ell <= 0.25)) and ell.shape == (N_LEVELS,)
        seen_d |= set(d.tolist())
        rec = log.records
        sfl_rows = rec["kind"] == KIND_SFL
        assert np.all(rec["slot"][sfl_rows] >= N_ENVS // 2) and np.all(rec["slot"][~sfl_rows] < N_ENVS // 2)
        assert set(rec["level"][sfl_rows].tolist()) <= seen_d
        assert np.all(rec["kind"][~sfl_rows] == KIND_DR)
        assert log.mask.all() and log.metrics["train_frac"] == 1.0
    assert trainer.sampler_snapshot(st)["refresh_chunk"] == 4


def test_sfl_scorer_is_the_audit_rollout_and_compiles_once(pool_setup):
    bank, _ = pool_setup
    actor, critic = Actor(CFG, racing.ACTION_DIM, False), Critic(CFG)
    key = jax.random.PRNGKey(3)
    compiled = []
    for s in range(2):                                   # two checkpoints, one compilation
        a = init_agent(CFG, actor, critic, racing.OBS_DIM, jax.random.PRNGKey(s), jax.random.PRNGKey(9 + s), 4)
        snap = Snapshot(a.actor.params, a.critic.params, a.obs_rms, a.ret_rms._replace(var=a.ret_rms.var * 4))
        crashed, alive, dist = sfl.score_windows(snap, CFG, bank, 2, key, WINDOW, 0.01, 32, PARAMS)
        compiled.append(sfl._window_scorer(CFG, WINDOW, 0.01, PARAMS)._cache_size())
        ro = rollout_windows(snap, CFG, bank.tracks, bank.x0, bank.ctx, bank.hidden_mu, 2, key, steps=WINDOW,
                             obs_noise=0.01, level_chunk=32, params=PARAMS)
        assert np.array_equal(crashed, ro.terminated) and np.array_equal(dist, ro.distance)
        assert np.array_equal(alive, ro.alive.sum(-1)) and alive.dtype.kind == "i"
    assert compiled[1] == compiled[0]


@pytest.mark.parametrize("arm", ["dr-pool", "plr-maxmc", "plr-oracle"])
def test_other_arms_run(pool_setup, arm):
    bank, tree = pool_setup
    oracle = np.linspace(20.0, 40.0, N_LEVELS) if arm == "plr-oracle" else None
    trainer = UEDTrainer(CFG, UEDConfig(arm=arm, plr=PLRConfig(capacity=16)), bank, tree, PARAMS,
                         oracle_j=oracle)
    st = trainer.init(jax.random.PRNGKey(2))
    for _ in range(2):
        before, cur = st.buffer, st.cur
        st, log = trainer.step(st)
        if trainer.plr is not None:
            _check_plr_chunk(st.chunk - 1, before, cur, st, log, trainer)
        else:
            assert np.all(log.records["kind"] == KIND_DR) and log.mask.all()
    if arm == "plr-oracle":
        snap = trainer.sampler_snapshot(st)
        # Exploratory insert: S = J* - mean G^gamma of the group, so S >= J* - max G^gamma.
        assert np.all(snap["scores"] >= oracle[snap["level_id"]] - snap["max_return_disc"] - 1e-3)
