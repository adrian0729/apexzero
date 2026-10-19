"""Racing PLR-perp: spec section 4.2 vectors P-1..P-9 (several run JaxUED's LevelSampler
directly, to pin down our reading of it) plus the asynchronous-window rules of spec 2.2
(E-7, DEV-17, DEV-18, DEV-19, write groups and event order)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.tracks.levels import Levels
from apexzero.ued import pool
from apexzero.ued.plr import (
    KIND_EXPLORE,
    KIND_REPLAY,
    OP_FIRST_USE,
    OP_INSERT,
    OP_NOOP,
    OP_UPDATE,
    OUT_DROPPED,
    OUT_DUPLICATE,
    OUT_INSERTED,
    OUT_REJECTED,
    OUT_STALE_DRAW,
    OUT_TIMESTAMP,
    OUT_UPDATED,
    Assignment,
    Events,
    PLRConfig,
    RacingPLR,
    init_buffer,
    make_sampler,
)

L = 500


def _levels(n=L):
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(3), i))(jnp.arange(n))
    rng = np.random.default_rng(0)
    return pool.level_pytree(Levels(keys, rng.uniform(-1, 1, (n, 6)), rng.uniform(0.98, 1.02, n), "train"))


LEVELS = _levels()


def _buf(sampler, level_ids, scores, timestamps, c, extras=None):
    """A buffer holding pool levels ``level_ids`` at indices 0..n-1."""
    buf = init_buffer(sampler)
    n = len(level_ids)
    ids = jnp.asarray(level_ids)
    levels = {k: buf["levels"][k].at[:n].set(LEVELS[k][ids]) for k in LEVELS}
    ex = dict(buf["levels_extra"])
    for k, v in (extras or {}).items():
        ex[k] = ex[k].at[:n].set(jnp.asarray(v, ex[k].dtype))
    return {**buf, "levels": levels, "levels_extra": ex,
            "scores": buf["scores"].at[:n].set(jnp.asarray(scores, jnp.float32)),
            "timestamps": buf["timestamps"].at[:n].set(jnp.asarray(timestamps, jnp.int32)),
            "size": jnp.asarray(n, buf["size"].dtype), "episode_count": jnp.asarray(c, buf["size"].dtype)}


def _events(rows, n_slots=4):
    """rows: dicts with op, level_id, buf_idx and group stats; padded to 3 n_slots."""
    E = 3 * n_slots
    cols = {f: np.zeros(E, dt) for f, dt in zip(Events._fields, (
        np.int32, np.int32, np.int32, np.int32, np.float32, np.float32, np.float32, np.float32,
        np.float32, bool, np.int32, np.int32), strict=True)}
    cols["buf_idx"][:] = -1
    for e, r in enumerate(rows):
        r = {"n": 1, **r}
        for k, v in r.items():
            cols[k][e] = v
    return Events(**cols)


def _level(i):
    return jax.tree.map(lambda a: a[i], LEVELS)


EXTRA = {"max_return": jnp.float32(0), "max_return_disc": jnp.float32(0), "ever_s1": jnp.bool_(False),
         "score_chunk": jnp.int32(0)}


# --------------------------------------------------------------------------- P-1..P-9


def p1_state():
    s = make_sampler(PLRConfig(capacity=3, staleness_coeff=0.3, temperature=0.3))
    return s, _buf(s, [10, 11, 12], [0.5, 2.0, 1.0], [2, 9, 5], c=10)


def test_P1_weights():
    s, buf = p1_state()
    ranks = np.array([3, 1, 2])
    w = (1.0 / ranks) ** (1 / 0.3)
    assert np.allclose(s.score_weights(buf), w / w.sum(), atol=1e-6)
    assert np.allclose(s.score_weights(buf), [0.022829, 0.888974, 0.088197], atol=1e-6)
    assert np.allclose(s.staleness_weights(buf), [0.571429, 0.071429, 0.357143], atol=1e-6)
    assert np.allclose(s.level_weights(buf), [0.187409, 0.643710, 0.168881], atol=1e-6)


def test_P2_admission_against_lowest_combined_weight():
    s, buf = p1_state()
    assert int(jnp.argmin(s.level_weights(buf))) == 2       # not index 0, the lowest score
    out, idx = s.insert(buf, _level(20), 0.6, EXTRA)
    assert int(idx) == -1                                     # 1.0 < 0.6 is False
    assert np.array_equal(out["scores"], buf["scores"]) and int(out["episode_count"]) == 11
    out, idx = s.insert(buf, _level(20), 1.5, EXTRA)
    assert int(idx) == 2
    assert float(out["scores"][2]) == 1.5 and int(out["levels"]["level_id"][2]) == 20
    assert int(out["timestamps"][2]) == 11 and int(out["episode_count"]) == 11


def test_P3_minus_inf_is_never_admitted():
    s = make_sampler(PLRConfig(capacity=3))
    buf = _buf(s, [10, 11], [0.5, 2.0], [1, 2], c=4)
    assert int(s._get_next_idx(buf)) == 2
    out, idx = s.insert(buf, _level(20), -jnp.inf, EXTRA)
    assert int(idx) == -1 and int(out["size"]) == 2 and int(out["episode_count"]) == 5


def test_P4_ties_rank_the_lower_index_first():
    s = make_sampler(PLRConfig(capacity=3, temperature=0.3))
    buf = _buf(s, [10, 11, 12], [1.0, 1.0, 0.0], [0, 0, 0], c=1)
    w = (1.0 / np.array([1, 2, 3])) ** (1 / 0.3)
    assert np.allclose(s.score_weights(buf), w / w.sum(), atol=1e-6)


def test_P5_staleness_falls_back_to_uniform():
    s = make_sampler(PLRConfig(capacity=3))
    buf = _buf(s, [10, 11], [1.0, 2.0], [7, 7], c=7)
    assert np.allclose(s.staleness_weights(buf), [0.5, 0.5, 0.0])


def test_P6_duplicate_overwrites_score_and_extras():
    plr = RacingPLR(PLRConfig(capacity=4, score="maxmc"), LEVELS)
    buf = _buf(plr.sampler, [10, 11, 12], [0.5, 2.0, 1.0], [2, 9, 5], c=10,
               extras={"max_return": [1, 5, 1], "max_return_disc": [1, 4, 1], "ever_s1": [0, 1, 0]})
    ev = _events([dict(op=OP_INSERT, level_id=11, g_max=2.0, v_bar=1.0, g_disc_max=-3.0, any_s1=False)])
    out, res = plr.apply_events(buf, ev, chunk=7)
    assert res["outcome"][0] == OUT_DUPLICATE and res["index"][0] == 1
    assert float(out["scores"][1]) == pytest.approx(1.0)       # R_max 2 - V 1 (E-7, second half)
    ex = jax.tree.map(lambda a: a[1], out["levels_extra"])
    assert float(ex["max_return"]) == 2.0 and float(ex["max_return_disc"]) == -3.0
    assert not bool(ex["ever_s1"]) and int(ex["score_chunk"]) == 7
    assert int(out["timestamps"][1]) == 11 and int(out["episode_count"]) == 11
    assert int(out["size"]) == 3


def test_P7_no_replay_below_the_fill_ratio():
    plr = RacingPLR(PLRConfig(capacity=10), LEVELS)
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    low = _buf(plr.sampler, list(range(4)), np.arange(4.0), np.zeros(4), c=4)
    full = _buf(plr.sampler, list(range(5)), np.arange(5.0), np.zeros(5), c=5)
    dec = jax.vmap(plr.sampler.sample_replay_decision, in_axes=(None, 0))
    assert not bool(dec(low, keys).any())
    assert 0.7 < float(dec(full, keys).mean()) < 0.9
    a = plr.predraw(low, jax.random.PRNGKey(1), 400)
    assert np.all(a.kind == KIND_EXPLORE) and np.all(a.buf_idx == -1)
    assert a.level_id.min() >= 0 and a.level_id.max() < L
    assert np.all(plr.predraw(plr.init(), jax.random.PRNGKey(1), 50).kind == KIND_EXPLORE)


def test_P8_replay_write_after_eviction_is_dropped():
    plr = RacingPLR(PLRConfig(capacity=10, score="pvl"), LEVELS)
    ids = [100 + i for i in range(10)]
    ids[7] = 123
    scores = np.arange(10, 20.0)
    scores[7] = 0.0                                            # lowest score...
    ts = np.zeros(10)
    ts[7] = 50                                                 # ...and freshest: lowest weight
    buf = _buf(plr.sampler, ids, scores, ts, c=50)
    assert int(jnp.argmin(plr.sampler.level_weights(buf))) == 7
    ev = _events([dict(op=OP_INSERT, level_id=456, pvl=5.0),             # evicts index 7
                  dict(op=OP_UPDATE, level_id=123, buf_idx=7, pvl=99.0)])  # slot replaying 123
    out, res = plr.apply_events(buf, ev, chunk=3)
    assert res["outcome"][0] == OUT_INSERTED and res["index"][0] == 7
    assert res["outcome"][1] == OUT_DROPPED
    assert int(out["levels"]["level_id"][7]) == 456 and float(out["scores"][7]) == 5.0


@pytest.mark.parametrize("beta,mass", [(0.3, 0.8716), (1.0, 0.1336), (0.1, 0.9990)])
def test_P9_rank_mass_of_the_top_level(beta, mass):
    s = make_sampler(PLRConfig(capacity=1000, temperature=beta))
    buf = init_buffer(s)
    buf = {**buf, "scores": jnp.linspace(1.0, 0.0, 1000), "size": jnp.asarray(1000, buf["size"].dtype)}
    assert float(s.score_weights(buf).max()) == pytest.approx(mass, abs=1e-4)


# --------------------------------------------------------------------------- spec 2.2 rules


def test_E7_rmax_history_replay_then_duplicate_hit():
    plr = RacingPLR(PLRConfig(capacity=4, score="maxmc"), LEVELS)
    buf = _buf(plr.sampler, [30], [0.0], [0], c=1, extras={"max_return": [5.0]})
    ev = _events([dict(op=OP_UPDATE, level_id=30, buf_idx=0, g_max=3.0, v_bar=1.0),
                  dict(op=OP_INSERT, level_id=30, g_max=2.0, v_bar=1.0)])
    out, res = plr.apply_events(buf, ev, chunk=0)
    assert list(res["outcome"][:2]) == [OUT_UPDATED, OUT_DUPLICATE]
    assert res["score"][0] == pytest.approx(4.0) and res["score"][1] == pytest.approx(1.0)
    assert float(out["levels_extra"]["max_return"][0]) == 2.0


def test_DEV19_concurrent_replay_writes_read_the_latest_rmax():
    plr = RacingPLR(PLRConfig(capacity=4, score="maxmc"), LEVELS)
    buf = _buf(plr.sampler, [30], [0.0], [0], c=1,
               extras={"max_return": [5.0], "max_return_disc": [-jnp.inf], "ever_s1": [False]})
    ev = _events([dict(op=OP_UPDATE, level_id=30, buf_idx=0, g_max=7.0, v_bar=1.0, g_disc_max=-4.0),
                  dict(op=OP_UPDATE, level_id=30, buf_idx=0, g_max=6.0, v_bar=2.0, g_disc_max=-6.0,
                       any_s1=True)])
    out, res = plr.apply_events(buf, ev, chunk=2)
    assert res["score"][0] == pytest.approx(6.0) and res["score"][1] == pytest.approx(5.0)
    ex = jax.tree.map(lambda a: a[0], out["levels_extra"])
    assert float(ex["max_return"]) == 7.0
    assert float(ex["max_return_disc"]) == -4.0        # discounted max: no floor, kept
    assert bool(ex["ever_s1"])
    assert int(out["episode_count"]) == 1              # replay writes do not touch c


def test_first_use_of_a_replay_draw_and_of_a_stale_draw():
    plr = RacingPLR(PLRConfig(capacity=4), LEVELS)
    buf = _buf(plr.sampler, [30, 31], [1.0, 2.0], [1, 2], c=5)
    ev = _events([dict(op=OP_FIRST_USE, level_id=31, buf_idx=1),
                  dict(op=OP_FIRST_USE, level_id=99, buf_idx=0)])     # level 99 left index 0
    out, res = plr.apply_events(buf, ev, chunk=0)
    assert list(res["outcome"][:3]) == [OUT_TIMESTAMP, OUT_STALE_DRAW, 0]
    assert list(np.asarray(out["timestamps"][:2])) == [1, 6]
    assert int(out["episode_count"]) == 7
    assert np.array_equal(out["scores"], buf["scores"])


def test_explore_insert_extras_and_rejection():
    plr = RacingPLR(PLRConfig(capacity=2, score="pvl"), LEVELS)
    buf = _buf(plr.sampler, [30, 31], [1.0, 2.0], [1, 2], c=2)
    ev = _events([dict(op=OP_INSERT, level_id=40, pvl=0.5, g_max=-2.0, g_disc_max=-1.5, any_s1=True),
                  dict(op=OP_INSERT, level_id=41, pvl=3.0, g_max=-2.0, g_disc_max=-1.5, any_s1=True),
                  dict(op=OP_NOOP)])
    out, res = plr.apply_events(buf, ev, chunk=4)
    assert list(res["outcome"][:3]) == [OUT_REJECTED, OUT_INSERTED, 0]
    i = int(res["index"][1])
    ex = jax.tree.map(lambda a: a[i], out["levels_extra"])
    assert float(ex["max_return"]) == 0.0                # R_max floored at 0 (JaxUED)
    assert float(ex["max_return_disc"]) == -1.5          # R^gamma_max is not floored
    assert bool(ex["ever_s1"]) and int(ex["score_chunk"]) == 4
    assert int(out["episode_count"]) == 4


def test_oracle_latest_score_uses_the_level_label():
    j = np.linspace(0.0, 49.9, L)
    plr = RacingPLR(PLRConfig(capacity=4, score="oracle"), LEVELS, oracle_j=j)
    ev = _events([dict(op=OP_INSERT, level_id=200, g_disc_mean=7.0)])
    out, res = plr.apply_events(plr.init(), ev, chunk=0)
    assert res["score"][0] == pytest.approx(j[200] - 7.0, rel=1e-6)
    with pytest.raises(ValueError):
        RacingPLR(PLRConfig(score="oracle"), LEVELS)
    with pytest.raises(ValueError):
        RacingPLR(PLRConfig(score="oracle"), LEVELS, oracle_j=np.full(L, np.nan))


def test_non_finite_scores_never_reach_the_buffer():
    plr = RacingPLR(PLRConfig(capacity=4, score="pvl"), LEVELS)
    with pytest.raises(AssertionError):
        plr.apply_events(plr.init(), _events([dict(op=OP_INSERT, level_id=3, pvl=np.nan)]), chunk=0)


def test_predraw_uses_frozen_buffer_weights():
    plr = RacingPLR(PLRConfig(capacity=3, staleness_coeff=0.3, temperature=0.3, replay_prob=1.0,
                              minimum_fill_ratio=0.5), LEVELS)
    buf = _buf(plr.sampler, [10, 11, 12], [0.5, 2.0, 1.0], [2, 9, 5], c=10)
    a = plr.predraw(buf, jax.random.PRNGKey(5), 20000)
    assert np.all(a.kind == KIND_REPLAY)
    assert np.array_equal(a.level_id, np.array([10, 11, 12])[a.buf_idx])
    freq = np.bincount(a.buf_idx, minlength=3) / a.buf_idx.size
    assert np.allclose(freq, [0.187409, 0.643710, 0.168881], atol=0.015)


def _records(rows):
    cols = ("slot", "t", "level", "pvl", "G", "V_bar", "G_disc", "s1")
    return {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}


def test_chunk_events_group_and_order():
    # slot 0: carried explore window ends at t=5, then its replay draw ends twice (t=9, t=20);
    # slot 1: carried replay window ends at t=5, its explore draw is still running;
    # slot 2: no episode ends in the chunk.
    rec = _records([(0, 20, 11, 0.3, 4.0, 1.0, 2.0, True), (1, 5, 12, 0.7, 1.0, 1.0, 1.0, False),
                    (0, 5, 10, 0.2, 1.0, 2.0, 0.5, False), (0, 9, 11, 0.1, 2.0, 3.0, 1.0, False)])
    cur = Assignment(np.array([KIND_EXPLORE, KIND_REPLAY, KIND_EXPLORE]), np.array([10, 12, 14]),
                     np.array([-1, 4, -1]))
    nxt = Assignment(np.array([KIND_REPLAY, KIND_EXPLORE, KIND_REPLAY]), np.array([11, 13, 15]),
                     np.array([3, -1, 6]))
    ev, kinds = RacingPLR.chunk_events(rec, cur, nxt)
    assert ev.op.shape == (64,)
    got = [(int(ev.t[e]), int(ev.slot[e]), int(ev.op[e]), int(ev.level_id[e]), int(ev.buf_idx[e]))
           for e in range(4)]
    assert got == [(5, 0, OP_INSERT, 10, -1), (5, 0, OP_FIRST_USE, 11, 3), (5, 1, OP_UPDATE, 12, 4),
                   (20, 0, OP_UPDATE, 11, 3)]
    assert np.all(ev.op[4:] == OP_NOOP)
    assert ev.n[3] == 2 and ev.pvl[3] == pytest.approx(0.2) and ev.g_max[3] == 4.0
    assert ev.v_bar[3] == pytest.approx(2.0) and ev.g_disc_max[3] == 2.0
    assert ev.g_disc_mean[3] == pytest.approx(1.5) and bool(ev.any_s1[3])
    assert list(kinds) == [KIND_REPLAY, KIND_REPLAY, KIND_EXPLORE, KIND_REPLAY]
    rec["level"][2] = 99                                   # a record on the wrong level
    with pytest.raises(AssertionError):
        RacingPLR.chunk_events(rec, cur, nxt)


def _chunk_events_loop(rec, cur, nxt):
    """Straightforward per-slot reference for chunk_events: (t, slot, sub, op, level, idx, n, stats)."""
    rows = []
    for j in np.unique(rec["slot"]):
        idx = np.nonzero(rec["slot"] == j)[0]
        idx = idx[np.argsort(rec["t"][idx], kind="stable")]
        groups = [(idx[:1], cur)] + ([(idx[1:], nxt)] if idx.size > 1 else [])
        for g, a in groups:
            op = OP_UPDATE if a.kind[j] == KIND_REPLAY else OP_INSERT
            rows.append((rec["t"][g[-1]], j, 0, op, a.level_id[j], a.buf_idx[j], g.size,
                         rec["pvl"][g].mean(), rec["G"][g].max(), rec["V_bar"][g].mean(),
                         rec["G_disc"][g].max(), rec["G_disc"][g].mean(), rec["s1"][g].any()))
        if nxt.kind[j] == KIND_REPLAY:
            rows.append((rec["t"][idx[0]], j, 1, OP_FIRST_USE, nxt.level_id[j], nxt.buf_idx[j], 0,
                         0, 0, 0, 0, 0, False))
    return sorted(rows, key=lambda r: r[:3])


def test_chunk_events_matches_a_per_slot_loop():
    rng = np.random.default_rng(4)
    for trial in range(30):
        N = int(rng.integers(1, 12))
        n_rec = 0 if trial == 0 else int(rng.integers(1, 40))
        cur = Assignment(rng.integers(0, 2, N), rng.integers(0, L, N), rng.integers(-1, 16, N))
        nxt = Assignment(rng.integers(0, 2, N), rng.integers(0, L, N), rng.integers(-1, 16, N))
        slot = rng.integers(0, N, n_rec)
        t = np.empty(n_rec, int)
        for j in range(N):                        # distinct episode end times per slot
            sel = slot == j
            t[sel] = rng.choice(128, sel.sum(), replace=False)
        rec = {"slot": slot, "t": t, "pvl": rng.normal(size=n_rec), "G": rng.normal(size=n_rec),
               "V_bar": rng.normal(size=n_rec), "G_disc": rng.normal(size=n_rec),
               "s1": rng.random(n_rec) < 0.5}
        order = np.lexsort((t, slot))
        first = np.zeros(n_rec, bool)
        first[order[np.r_[True, slot[order][1:] != slot[order][:-1]]] if n_rec else []] = True
        rec["level"] = np.where(first, cur.level_id[slot], nxt.level_id[slot])
        ev, _ = RacingPLR.chunk_events(rec, cur, nxt)
        ref = _chunk_events_loop(rec, cur, nxt)
        assert (ev.op != OP_NOOP).sum() == len(ref)
        for e, r in enumerate(ref):
            got = (ev.t[e], ev.slot[e], ev.op[e], ev.level_id[e], ev.buf_idx[e], ev.n[e], ev.pvl[e],
                   ev.g_max[e], ev.v_bar[e], ev.g_disc_max[e], ev.g_disc_mean[e], ev.any_s1[e])
            assert np.allclose(np.array(got, float), np.array(r[:2] + r[3:], float), atol=1e-6)


def test_duplicate_check_by_full_level_equals_check_by_level_id():
    # LevelSampler.find compares the whole pytree; within one pool, level_id determines
    # (keys, ctx, hidden_mu), so this is the spec's duplicate check by level_id.
    flat = np.concatenate([np.asarray(LEVELS["keys"], np.float64), np.asarray(LEVELS["ctx"]),
                           np.asarray(LEVELS["hidden_mu"])[:, None]], axis=1)
    assert len(np.unique(flat, axis=0)) == L
    s = make_sampler(PLRConfig(capacity=8))
    buf = _buf(s, [5, 6, 7], [1.0, 1.0, 1.0], [0, 0, 0], c=0)
    assert [int(s.find(buf, _level(i))) for i in (5, 6, 7, 8)] == [0, 1, 2, -1]
