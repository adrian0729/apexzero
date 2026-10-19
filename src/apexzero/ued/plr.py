"""Racing PLR-perp (spec 2.2): JaxUED's ``LevelSampler`` driven by asynchronous windows.

The buffer is JaxUED's own ``LevelSampler`` (rank prioritisation, staleness mix, admission
against the lowest *combined* weight, duplicate check); nothing here reimplements it. The
level pytree is ``{keys, ctx, hidden_mu, level_id}`` (``ued.pool.level_pytree``) and the
per-level extras are

    max_return       R_max, undiscounted, floored at 0 (JaxUED's ``compute_max_returns``)
    max_return_disc  R^gamma_max, discounted, -inf initialised, no floor (DEV-22)
    ever_s1          any window on this level completed (MNA's gate, DEV-12)
    score_chunk      chunk of the last score write (score age for the in-buffer audit)

What differs from JaxUED's synchronous loop is when things happen. Windows are 1,200 steps
and chunks 128, so every slot is somewhere inside its own window when a chunk starts:

A. ``predraw`` (chunk start): one replay decision per slot (DEV-15) and a replay index
   drawn from the weights frozen for the chunk (DEV-17). A slot takes its draw at its next
   reset; if it finishes again within the chunk it replays it.
B/C. The trainer rolls out, scores (``ued.scoring``) and trains with loss mask
   1[window is a replay] (DEV-16).
D. ``chunk_events`` turns the chunk's completed episodes into buffer operations, which
   ``apply_events`` commits in (t, slot) order inside one jitted scan:
   - first use of a replay draw: c += 1 and the level's timestamp <- c (unused draws are
     discarded; if the level was evicted since the draw, the timestamp is left alone);
   - a finished exploratory write group: S from the group, then ``LevelSampler.insert``
     (a duplicate hit by level overwrites score and extras, as in JaxUED);
   - a finished replay write group: dropped if the level has left that index (DEV-18),
     otherwise R_max <- max(stored, group) read from the latest buffer state (DEV-19) and
     ``LevelSampler.update``.

A write group is the episodes of one slot on one assignment that end in the same chunk:
the slot's first episode (its assignment from before the chunk) and then all later ones
(its draw for this chunk). Scores (episode-weighted means, spec 1.8):

    pvl      mean_e PVL_e                     (lambda_PPO, mode F)
    maxmc    R_max - mean_e Vbar_e            (as implemented in JaxUED)
    oracle   J*_20[level] - mean_e G^gamma_e  (oracle-latest; needs the pool's labels)

Every group has at least one completed episode, so no -inf "nothing completed" score can
arise; ``apply_events`` still checks every score it writes and raises on a non-finite one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxued.level_sampler import LevelSampler

import apexzero.compat  # noqa: F401  (restores jax.tree_map before jaxued runs)

KIND_EXPLORE, KIND_REPLAY, KIND_DR, KIND_SFL = 0, 1, 2, 3
SCORES = ("pvl", "maxmc", "oracle")

OP_NOOP, OP_FIRST_USE, OP_INSERT, OP_UPDATE = 0, 1, 2, 3
# apply_events outcomes per event
OUT_NONE, OUT_TIMESTAMP, OUT_STALE_DRAW, OUT_INSERTED, OUT_DUPLICATE, OUT_REJECTED, \
    OUT_UPDATED, OUT_DROPPED = range(8)


@dataclass(frozen=True)
class PLRConfig:
    """Curriculum C defaults (spec 2.1 table, "Ours" column)."""
    capacity: int = 1000
    replay_prob: float = 0.8
    staleness_coeff: float = 0.3
    temperature: float = 0.3          # rank prioritisation, beta
    minimum_fill_ratio: float = 0.5
    duplicate_check: bool = True      # by level (level_id identifies a pool level)
    score: str = "pvl"                # "pvl" | "maxmc" | "oracle"


class Assignment(NamedTuple):
    """What each slot plays (host arrays [N])."""
    kind: np.ndarray       # KIND_EXPLORE | KIND_REPLAY
    level_id: np.ndarray   # pool index
    buf_idx: np.ndarray    # buffer index for replays, -1 otherwise


class Events(NamedTuple):
    """Buffer operations of one chunk in commit order, padded with OP_NOOP (arrays [E])."""
    op: np.ndarray
    level_id: np.ndarray
    buf_idx: np.ndarray
    n: np.ndarray            # episodes in the write group
    pvl: np.ndarray          # mean_e PVL_e
    g_max: np.ndarray        # max_e G_e
    v_bar: np.ndarray        # mean_e Vbar_e
    g_disc_max: np.ndarray   # max_e G^gamma_e
    g_disc_mean: np.ndarray  # mean_e G^gamma_e
    any_s1: np.ndarray
    slot: np.ndarray         # for logging only
    t: np.ndarray


def make_sampler(cfg: PLRConfig) -> LevelSampler:
    return LevelSampler(cfg.capacity, replay_prob=cfg.replay_prob, staleness_coeff=cfg.staleness_coeff,
                        minimum_fill_ratio=cfg.minimum_fill_ratio, prioritization="rank",
                        prioritization_params={"temperature": cfg.temperature},
                        duplicate_check=cfg.duplicate_check)


def placeholder_level() -> dict:
    return {"keys": jnp.zeros(2, jnp.uint32), "ctx": jnp.zeros(6, jnp.float32),
            "hidden_mu": jnp.ones((), jnp.float32), "level_id": jnp.int32(-1)}


def placeholder_extra() -> dict:
    return {"max_return": jnp.float32(-jnp.inf), "max_return_disc": jnp.float32(-jnp.inf),
            "ever_s1": jnp.bool_(False), "score_chunk": jnp.int32(-1)}


def init_buffer(sampler: LevelSampler) -> dict:
    """``LevelSampler.initialize`` with strongly typed counters, so the buffer can be the
    carry of a jitted scan. They take JAX's default integer width (int64 under x64), as the
    Python ints JaxUED starts from do; ``staleness_weights`` only type-checks that way."""
    buf = sampler.initialize(placeholder_level(), placeholder_extra())
    int_ = jax.dtypes.canonicalize_dtype(jnp.int64)
    return {**buf, "size": jnp.zeros((), int_), "episode_count": jnp.zeros((), int_)}


class RacingPLR:
    """Buffer operations for one PLR-perp run over a pool of levels.

    ``levels``: the pool's level pytree [L] (``ued.pool.level_pytree``).
    ``oracle_j``: [L] J*_20 per pool level, required for the oracle-latest score.
    """

    def __init__(self, cfg: PLRConfig, levels: dict, oracle_j=None):
        if cfg.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        if cfg.score == "oracle":
            if oracle_j is None:
                raise ValueError("the oracle-latest score needs J* labels for the pool")
            if not np.all(np.isfinite(np.asarray(oracle_j))):
                raise ValueError("J* labels must be finite")
        self.cfg = cfg
        self.sampler = make_sampler(cfg)
        self.levels = levels
        self.pool_size = int(levels["level_id"].shape[0])
        self.oracle_j = jnp.zeros(self.pool_size, jnp.float32) if oracle_j is None else \
            jnp.asarray(oracle_j, jnp.float32)
        self._predraw = jax.jit(self._predraw_impl, static_argnums=2)
        self._apply = jax.jit(self._apply_impl)

    def init(self) -> dict:
        return init_buffer(self.sampler)

    # ------------------------------------------------------------------ A. pre-draw

    def _predraw_impl(self, buf, key, n):
        kr, ki, ke = jax.random.split(key, 3)
        replay = jax.vmap(self.sampler.sample_replay_decision, in_axes=(None, 0))(
            buf, jax.random.split(kr, n))
        ok = buf["size"] > 0
        w = self.sampler.level_weights(buf)
        w = jnp.where(ok, w, 1.0 / self.cfg.capacity)
        idx = jax.random.choice(ki, self.cfg.capacity, (n,), p=w)
        replay = replay & ok
        explore = jax.random.randint(ke, (n,), 0, self.pool_size)
        level = jnp.where(replay, buf["levels"]["level_id"][idx], explore)
        return replay, jnp.where(replay, idx, -1), level

    def predraw(self, buf: dict, key: jax.Array, n: int) -> Assignment:
        """Next assignment of every slot, from the buffer as it is at the chunk start."""
        replay, idx, level = jax.device_get(self._predraw(buf, key, n))
        kind = np.where(replay, KIND_REPLAY, KIND_EXPLORE).astype(np.int32)
        return Assignment(kind, level.astype(np.int32), idx.astype(np.int32))

    # ------------------------------------------------------------------ D. events

    @staticmethod
    def chunk_events(records: dict, cur: Assignment, nxt: Assignment) -> tuple[Events, np.ndarray]:
        """Buffer operations of one chunk, plus the kind of every record.

        Per slot with completed episodes: write group 0 is its first episode (on ``cur``),
        committed at that episode's end; then the first use of ``nxt`` if it is a replay;
        write group 1 is every later episode (on ``nxt``), committed at the last one's end.
        Events are sorted by (t, slot, write before first use) and padded with no-ops to a
        power of two (at least 64), so the jitted scan compiles for a few sizes only."""
        order = np.lexsort((records["t"], records["slot"]))
        s, t = records["slot"][order], records["t"][order]
        first = np.r_[True, s[1:] != s[:-1]] if s.size else np.zeros(0, bool)
        kinds = np.empty(s.size, np.int32)
        kinds[order] = np.where(first, cur.kind[s], nxt.kind[s])
        assigned = np.where(first, cur.level_id[s], nxt.level_id[s])
        if np.any(records["level"][order] != assigned):
            raise AssertionError("a record's level does not match its slot's assignment")

        # Write groups are contiguous runs in (slot, t) order: a slot's first record, then the
        # rest of that slot's records (if any).
        heads = np.nonzero(first)[0]
        seconds = heads + 1
        seconds = seconds[seconds < s.size]
        seconds = seconds[~first[seconds]]
        start = np.sort(np.r_[heads, seconds])
        gs, is_first = s[start], first[start]
        red = lambda f, key: f.reduceat(records[key][order], start) if start.size else np.zeros(0)
        n = np.diff(np.r_[start, s.size])
        src = lambda a: np.where(is_first, getattr(cur, a)[gs], getattr(nxt, a)[gs])
        kind_g = src("kind")
        w = {"op": np.where(kind_g == KIND_REPLAY, OP_UPDATE, OP_INSERT), "level_id": src("level_id"),
             "buf_idx": src("buf_idx"), "n": n, "pvl": red(np.add, "pvl") / np.maximum(n, 1),
             "g_max": red(np.maximum, "G"), "v_bar": red(np.add, "V_bar") / np.maximum(n, 1),
             "g_disc_max": red(np.maximum, "G_disc"), "g_disc_mean": red(np.add, "G_disc") / np.maximum(n, 1),
             "any_s1": red(np.logical_or, "s1").astype(bool), "slot": gs,
             "t": np.maximum.reduceat(t, start) if start.size else np.zeros(0), "sub": np.zeros(start.size)}
        # First use of a replay draw, at the end of the slot's first episode.
        fu = np.nonzero(first & (nxt.kind[s] == KIND_REPLAY))[0]
        u = {k: np.zeros(fu.size) for k in w}
        u.update(op=np.full(fu.size, OP_FIRST_USE), level_id=nxt.level_id[s[fu]],
                 buf_idx=nxt.buf_idx[s[fu]], slot=s[fu], t=t[fu], sub=np.ones(fu.size))
        cat = {k: np.concatenate([w[k], u[k]]) for k in w}
        o = np.lexsort((cat["sub"], cat["slot"], cat["t"]))
        n_ev = o.size
        E = max(64, 1 << max(n_ev - 1, 0).bit_length())
        dtypes = dict(op=np.int32, level_id=np.int32, buf_idx=np.int32, n=np.int32, pvl=np.float32,
                      g_max=np.float32, v_bar=np.float32, g_disc_max=np.float32, g_disc_mean=np.float32,
                      any_s1=bool, slot=np.int32, t=np.int32)
        ev = {}
        for k, dt in dtypes.items():
            ev[k] = np.full(E, -1 if k == "buf_idx" else 0, dt)
            ev[k][:n_ev] = cat[k][o]
        return Events(**ev), kinds

    def group_score(self, ev, r_max, oracle_j):
        """S for a write group (JAX; ``r_max`` already combined with the stored value)."""
        if self.cfg.score == "pvl":
            return ev.pvl
        if self.cfg.score == "maxmc":
            return r_max - ev.v_bar
        return oracle_j[ev.level_id] - ev.g_disc_mean

    def _apply_impl(self, buf, events: Events, levels, oracle_j, chunk):
        s = self.sampler
        nan = jnp.float32(jnp.nan)

        def step(buf, ev):
            def noop(buf):
                return buf, (jnp.int32(OUT_NONE), jnp.int32(-1), nan)

            def first_use(buf):
                c = buf["episode_count"] + 1
                here = buf["levels"]["level_id"][ev.buf_idx] == ev.level_id
                ts = buf["timestamps"]
                ts = ts.at[ev.buf_idx].set(jnp.where(here, c, ts[ev.buf_idx]).astype(ts.dtype))
                out = jnp.where(here, OUT_TIMESTAMP, OUT_STALE_DRAW).astype(jnp.int32)
                return {**buf, "timestamps": ts, "episode_count": c}, (out, ev.buf_idx, nan)

            def insert(buf):
                level = jax.tree.map(lambda a: a[ev.level_id], levels)
                r_max = jnp.maximum(0.0, ev.g_max)
                extra = {"max_return": r_max, "max_return_disc": ev.g_disc_max,
                         "ever_s1": ev.any_s1, "score_chunk": chunk}
                score = self.group_score(ev, r_max, oracle_j)
                dup = s.find(buf, level) >= 0 if s.duplicate_check else jnp.bool_(False)
                buf, idx = s.insert(buf, level, score, extra)
                out = jnp.where(dup, OUT_DUPLICATE, jnp.where(idx >= 0, OUT_INSERTED, OUT_REJECTED))
                return buf, (out.astype(jnp.int32), idx.astype(jnp.int32), score)

            def update(buf):
                i = ev.buf_idx
                here = buf["levels"]["level_id"][i] == ev.level_id
                old = s.get_levels_extra(buf, i)
                r_max = jnp.maximum(old["max_return"], jnp.maximum(0.0, ev.g_max))
                extra = {"max_return": r_max,
                         "max_return_disc": jnp.maximum(old["max_return_disc"], ev.g_disc_max),
                         "ever_s1": old["ever_s1"] | ev.any_s1, "score_chunk": chunk}
                score = self.group_score(ev, r_max, oracle_j)
                new = s.update(buf, i, score, extra)
                buf = jax.tree.map(lambda a, b: jnp.where(here, a, b), new, buf)
                out = jnp.where(here, OUT_UPDATED, OUT_DROPPED).astype(jnp.int32)
                return buf, (out, i, score)

            return jax.lax.switch(ev.op, [noop, first_use, insert, update], buf)

        return jax.lax.scan(step, buf, Events(*(jnp.asarray(a) for a in events)))

    def apply_events(self, buf: dict, events: Events, chunk: int) -> tuple[dict, dict]:
        """Commit one chunk's events; returns the buffer and per-event (outcome, index,
        score) as host arrays. Raises if a non-finite score was written."""
        buf, (out, idx, score) = self._apply(buf, events, self.levels, self.oracle_j, jnp.int32(chunk))
        out, idx, score = (np.asarray(a) for a in jax.device_get((out, idx, score)))
        written = np.isin(out, (OUT_INSERTED, OUT_DUPLICATE, OUT_UPDATED, OUT_REJECTED))
        if not np.all(np.isfinite(score[written])):
            raise AssertionError("a non-finite level score reached the buffer")
        return buf, {"outcome": out, "index": idx, "score": score}


def buffer_snapshot(buf: dict) -> dict:
    """Host copy of the buffer's live entries (for checkpoints and the in-buffer audit)."""
    b = jax.device_get(buf)
    n = int(b["size"])
    out = {"size": n, "episode_count": int(b["episode_count"]),
           "level_id": np.asarray(b["levels"]["level_id"][:n]), "scores": np.asarray(b["scores"][:n]),
           "timestamps": np.asarray(b["timestamps"][:n])}
    out.update({k: np.asarray(v[:n]) for k, v in b["levels_extra"].items()})
    return out
