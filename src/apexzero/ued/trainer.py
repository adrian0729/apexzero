"""Curriculum C trainer: PPO on the labelled pool with a level sampler in the loop.

Arms (plan 2.4, row C), all on the same pool, PPO and chunk length:

    dr-pool     a uniform pool draw at every reset; train on every transition
    plr-pvl     racing PLR-perp (``ued.plr``) scored by PVL (lambda_PPO, mode F)
    plr-maxmc   racing PLR-perp scored by MaxMC as implemented in JaxUED
    plr-oracle  racing PLR-perp scored by oracle-latest, J*_20 - mean G^gamma (pool labels)
    sfl         SFL (``ued.sfl``)

One chunk (``PPOConfig.num_steps`` = 128 steps of every slot) is one PPO update:

    A. PLR: pre-draw every slot's next assignment from the buffer (frozen for the chunk);
       SFL: every ``refresh_every`` chunks, re-score the pool and replace D
    B. roll out (slots reset to their assigned level, ``ued.pool``)
    C. score the chunk (``ued.scoring``, the audit's estimator code) and run the PPO update;
       PLR-perp masks exploratory windows out of every loss and both normalisers (DEV-16)
    D. PLR: commit the chunk's buffer events in (t, slot) order (``RacingPLR.apply_events``)

Why a dedicated trainer rather than more hooks inside ``make_train`` (one jitted scan over
all updates): step D must run between the rollout and the next chunk's pre-draw, in event
order, on the scores of the audit's own NumPy scorer. So the loop runs on the host, one
chunk at a time, around jitted kernels: the rollout and the learner are ``agents.ppo``'s
own (``make_rollout`` / ``make_learner``; the same networks, losses and normalisers as
E0, with the per-transition loss mask as the only addition), and the buffer events of a
chunk are one jitted scan. The host work per chunk (scoring a [128, N] chunk, grouping
episodes) is small next to the rollout. DR-pool and SFL call the learner without a mask,
which is exactly E0's update.

Learning-rate annealing counts optimiser steps (as in ``make_train`` and JaxUED); under
PLR-perp a minibatch with no replayed transition takes no step, so the schedule ends
slightly above zero (in JaxUED's PLR-perp, every exploratory update skips its steps).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.agents.ppo import (
    Actor,
    AgentState,
    Critic,
    PPOConfig,
    init_agent,
    make_learner,
    make_rollout,
)
from apexzero.audit.estimators import LAMBDAS
from apexzero.audit.rollouts import Snapshot
from apexzero.env import racing
from apexzero.ued import pool, sfl
from apexzero.ued.plr import (
    KIND_DR,
    KIND_EXPLORE,
    KIND_REPLAY,
    KIND_SFL,
    OUT_DROPPED,
    OUT_DUPLICATE,
    OUT_INSERTED,
    OUT_NONE,
    OUT_REJECTED,
    OUT_STALE_DRAW,
    OUT_UPDATED,
    Assignment,
    Events,
    PLRConfig,
    RacingPLR,
    buffer_snapshot,
    event_rows,
)
from apexzero.ued.scoring import ChunkData, ChunkScorer

ARMS = ("dr-pool", "plr-pvl", "plr-maxmc", "plr-oracle", "sfl")


@dataclass(frozen=True)
class UEDConfig:
    arm: str = "plr-pvl"
    plr: PLRConfig = field(default_factory=PLRConfig)
    sfl: sfl.SFLConfig = field(default_factory=sfl.SFLConfig)
    lambdas: tuple[float, ...] = LAMBDAS     # scorer variants (lambda_PPO is the headline)
    modes: tuple[str, ...] = ("F", "B")
    mna: bool = True                         # accumulate MNA's G_hat in the records (O(T^2))

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")

    @property
    def is_plr(self) -> bool:
        return self.arm.startswith("plr-")

    def plr_config(self) -> PLRConfig:
        return replace(self.plr, score=self.arm.removeprefix("plr-"))


@dataclass
class TrainerState:
    agent: AgentState
    env_state: pool.PoolState
    obs: jax.Array
    ret_acc: jax.Array
    ep_ret: jax.Array
    ep_len: jax.Array
    key: jax.Array            # threaded through rollout and learner
    sampler_key: jax.Array    # pre-draws and SFL refreshes
    bank: pool.PoolBank
    chunk: int = 0
    buffer: dict | None = None             # PLR: the LevelSampler state
    cur: Assignment | None = None          # PLR: what each slot is playing now
    refresh: sfl.Refresh | None = None     # SFL: the last refresh
    refresh_chunk: int = -1


class ChunkLog(NamedTuple):
    metrics: dict
    records: dict                  # one row per completed episode (``ued.scoring``), plus
                                   # kind and the write of its group: write_row (row of
                                   # ``event_rows``), write_outcome, write_score (-1 / OUT_NONE
                                   # / nan outside PLR)
    mask: np.ndarray               # [T, N] loss mask of the chunk (all True outside PLR)
    nxt: Assignment | None = None  # PLR: the chunk's pre-draw
    events: Events | None = None   # PLR: the committed buffer operations (padded)
    outcomes: dict | None = None   # PLR: per-event outcome, buffer index, score, R_max
    event_rows: dict | None = None  # PLR: ``plr.event_rows``, the admission log of the chunk


class UEDTrainer:
    """``state = trainer.init(key)``, then ``state, log = trainer.step(state)`` per chunk
    (``step`` updates the state object in place and returns it).

    ``levels`` is the pool's level pytree (``pool.level_pytree``), aligned with ``bank``;
    ``oracle_j`` [L] are the pool's J*_20 labels (plr-oracle only).
    """

    def __init__(self, cfg: PPOConfig, ued: UEDConfig, bank: pool.PoolBank, levels: dict,
                 env_params: racing.EnvParams = racing.EnvParams(), oracle_j=None):
        self.cfg, self.ued, self.env_params = cfg, ued, env_params
        self.bank0 = bank
        self.env = pool.pool_env(env_params)
        self.actor = Actor(cfg, racing.ACTION_DIM, False)
        self.critic = Critic(cfg)
        self.num_updates = max(cfg.num_updates, 1)
        mask_fn = pool.loss_mask if ued.is_plr else None
        self._rollout = jax.jit(make_rollout(cfg, self.env, self.actor, self.critic,
                                             mask_fn=mask_fn, trace_fn=pool.trace))
        self._learn = jax.jit(make_learner(cfg, self.env, self.actor, self.critic, self.num_updates))
        self._obs_update = jax.jit(_obs_update)
        self._diag = jax.jit(_diagnostics)
        self.scorer = ChunkScorer(cfg.num_envs, cfg.gamma, ued.lambdas, cfg.gae_lambda, ued.modes, ued.mna)
        self.plr = RacingPLR(ued.plr_config(), levels, oracle_j) if ued.is_plr else None
        self.sources = (sfl.slot_sources(cfg.num_envs, ued.sfl.rho) if ued.arm == "sfl"
                        else np.zeros(cfg.num_envs, np.int32))

    # ------------------------------------------------------------------ setup

    def init(self, key: jax.Array) -> TrainerState:
        cfg, N, L = self.cfg, self.cfg.num_envs, self.bank0.size
        key, ka, kc, ks, kl = jax.random.split(key, 5)
        # Strong dtypes from the start (TrainState.step and the normaliser counts are weakly
        # typed when created), so no kernel is compiled a second time after the first update.
        agent = jax.tree.map(lambda x: jnp.array(x, dtype=jnp.asarray(x).dtype),
                             init_agent(cfg, self.actor, self.critic, racing.OBS_DIM, ka, kc, self.num_updates))
        z = jnp.zeros(N)
        st = TrainerState(agent, None, None, z, z, z, key, ks, self.bank0)
        if self.plr is not None:
            # Chunk 0 starts on exploratory levels: the buffer is empty.
            cur = Assignment(np.full(N, KIND_EXPLORE, np.int32),
                             np.asarray(jax.random.randint(kl, (N,), 0, L), np.int32),
                             np.full(N, -1, np.int32))
            st.obs, st.env_state = pool.init_slots(st.bank, cur.level_id, cur.level_id, np.full(N, -1),
                                                   cur.kind == KIND_REPLAY, cur.kind == KIND_REPLAY)
            st.buffer, st.cur = self.plr.init(), cur
            return st
        if self.ued.arm == "sfl":
            st = self._sfl_refresh(st)
        # Initial levels: a uniform draw from each slot's source table.
        lens = np.asarray(st.bank.source_len)[self.sources]
        u = np.asarray(jax.random.uniform(kl, (N,)))
        pick = np.minimum((u * lens).astype(np.int64), lens - 1)
        lvl = np.asarray(st.bank.sources)[self.sources, pick]
        st.obs, st.env_state = pool.init_slots(st.bank, lvl, lvl, self.sources, np.ones(N, bool), np.ones(N, bool))
        if cfg.normalize_obs:
            st.agent = agent._replace(obs_rms=agent.obs_rms.update(st.obs))
        return st

    def _sfl_refresh(self, st: TrainerState) -> TrainerState:
        st.sampler_key, k = jax.random.split(st.sampler_key)
        a = st.agent
        snap = Snapshot(a.actor.params, a.critic.params, a.obs_rms, a.ret_rms)
        st.refresh = sfl.refresh(snap, self.cfg, st.bank, k, self.ued.sfl, self.env_params)
        st.refresh_chunk = st.chunk
        st.bank = pool.with_source(st.bank, sfl.SOURCE_BUFFER, st.refresh.buffer)
        return st

    # ------------------------------------------------------------------ one chunk

    def step(self, st: TrainerState) -> tuple[TrainerState, ChunkLog]:
        cfg, N, k = self.cfg, self.cfg.num_envs, st.chunk
        # A. assignments
        nxt = None
        if self.plr is not None:
            st.sampler_key, kp = jax.random.split(st.sampler_key)
            nxt = self.plr.predraw(st.buffer, kp, N)
            st.env_state = st.env_state._replace(next_level=jnp.asarray(nxt.level_id),
                                                 next_mask=jnp.asarray(nxt.kind == KIND_REPLAY))
        elif self.ued.arm == "sfl" and k > 0 and k % self.ued.sfl.refresh_every == 0:
            st = self._sfl_refresh(st)
        sigma = float(np.sqrt(np.asarray(st.agent.ret_rms.var) + 1e-8))   # spec 0.2: at the chunk start

        # B. rollout
        carry, ro = self._rollout(st.agent, st.env_state, st.obs, st.ret_acc, st.ep_ret, st.ep_len,
                                  st.key, st.bank)
        agent, st.env_state, st.obs, st.ret_acc, st.ep_ret, st.ep_len, key = carry

        # C. PPO update (dispatched asynchronously), then score the chunk on the host
        mask = ro.mask if self.plr is not None else None
        agent, st.key, (_, targets, m) = self._learn(agent, ro.traj, ro.last_value, ro.last_heads,
                                                     key, jnp.int32(k), mask)
        if cfg.normalize_obs:
            agent = agent._replace(obs_rms=self._obs_update(agent.obs_rms, ro.raw_obs, mask))
        st.agent = agent
        tr = ro.traj
        data = ChunkData(*jax.device_get((tr.raw_reward, tr.value, tr.heads, tr.terminated, tr.truncated,
                                          tr.final_value, ro.last_value, ro.trace["level"],
                                          ro.trace["progress"], ro.trace["end_of_strip"])))
        records, _ = self.scorer.score(data, sigma, k)
        if records["end_of_strip"].any():
            raise AssertionError("a pool window reached the end of its strip")

        # D. buffer events
        events = outcomes = rows = None
        n_rec = records["slot"].size
        write_row = np.full(n_rec, -1, np.int32)
        write_out = np.full(n_rec, OUT_NONE, np.int32)
        write_score = np.full(n_rec, np.nan, np.float32)
        if self.plr is not None:
            events, kinds, write_row = RacingPLR.chunk_events(records, st.cur, nxt)
            st.buffer, outcomes = self.plr.apply_events(st.buffer, events, k)
            steps = np.bincount(write_row, weights=records["T"], minlength=events.op.size)
            rows = event_rows(events, outcomes, k, steps)
            write_out, write_score = outcomes["outcome"][write_row], outcomes["score"][write_row]
            reset = np.zeros(N, bool)
            reset[records["slot"]] = True
            st.cur = Assignment(*(np.where(reset, n, c) for c, n in zip(st.cur, nxt, strict=True)))
        elif self.ued.arm == "sfl":
            kinds = np.where(self.sources[records["slot"]] == sfl.SOURCE_BUFFER, KIND_SFL, KIND_DR)
        else:
            kinds = np.full(n_rec, KIND_DR)
        records["kind"] = kinds.astype(np.int32)
        records["timesteps"] = np.full(n_rec, (k + 1) * cfg.batch_size, np.int64)
        records.update(write_row=write_row, write_outcome=write_out, write_score=write_score)

        mask_np = np.asarray(ro.mask) if mask is not None else np.ones((cfg.num_steps, N), bool)
        metrics = self._metrics(st, k, records, mask_np, m, self._diag(tr, targets, mask), rows, nxt)
        st.chunk = k + 1
        return st, ChunkLog(metrics, records, mask_np, nxt, events, outcomes, rows)

    def _metrics(self, st, k, rec, mask, m, diag, rows, nxt) -> dict:
        m = jax.device_get(m)
        out = {
            "chunk": k, "timesteps": (k + 1) * self.cfg.batch_size,
            "episodes": int(rec["slot"].size), "train_frac": float(mask.mean()),
            "value_loss": float(np.mean(m[0])), "policy_loss": float(np.mean(m[1])),
            "entropy": float(np.mean(m[2])), "approx_kl": float(np.mean(m[3])),
            "clip_frac": float(np.mean(m[4])),
            "explained_var": float(diag) if mask.any() else float("nan"),
        }
        # The same keys every chunk (a CSV header is written from the first one).
        kinds = ((("explore", KIND_EXPLORE), ("replay", KIND_REPLAY)) if self.plr is not None else
                 (("dr", KIND_DR), ("sfl", KIND_SFL)) if self.ued.arm == "sfl" else (("dr", KIND_DR),))
        for name, kind in kinds:
            sel = rec["kind"] == kind
            mean = lambda a, sel=sel: float(a[sel].mean()) if sel.any() else float("nan")
            out[f"{name}_episodes"] = int(sel.sum())
            out[f"{name}_return"] = mean(rec["G"])
            out[f"{name}_completion"] = mean(rec["s1"])
            out[f"{name}_length"] = mean(rec["T"])
        if rows is not None:
            o = rows["outcome"]
            for name, code in (("inserted", OUT_INSERTED), ("duplicate", OUT_DUPLICATE),
                               ("rejected", OUT_REJECTED), ("updated", OUT_UPDATED),
                               ("dropped", OUT_DROPPED), ("stale_draw", OUT_STALE_DRAW)):
                out[f"buf_{name}"] = int((o == code).sum())
            # Trained-on transitions of the replay windows that ended here, split by whether
            # their score reached the buffer or was dropped because the level had been
            # evicted (DEV-18): PLR-perp trains on the latter but never scores them.
            scored = int(rows["steps"][o == OUT_UPDATED].sum())
            dropped = int(rows["steps"][o == OUT_DROPPED].sum())
            out.update(predraw_replay=int((nxt.kind == KIND_REPLAY).sum()), replay_steps_scored=scored,
                       replay_steps_dropped=dropped,
                       replay_dropped_frac=dropped / (scored + dropped) if scored + dropped else float("nan"))
            size = int(st.buffer["size"])
            scores = np.asarray(st.buffer["scores"][:size])
            out.update(buf_size=size, buf_score_mean=float(scores.mean()) if size else float("nan"),
                       buf_score_max=float(scores.max()) if size else float("nan"))
        if st.refresh is not None:
            out.update(sfl_refresh_chunk=st.refresh_chunk,
                       sfl_learnability_mean=float(st.refresh.learnability[st.refresh.buffer].mean()),
                       sfl_pool_completion=float(st.refresh.success.mean()))
        return out

    # ------------------------------------------------------------------ outputs

    def snapshot(self, st: TrainerState) -> dict:
        """Checkpoint payload in ``train_dr.py``'s format (params and both normalisers)."""
        a = jax.device_get(st.agent)
        return {"actor": a.actor.params, "critic": a.critic.params, "obs_rms": a.obs_rms._asdict(),
                "ret_rms": a.ret_rms._asdict()}

    def sampler_snapshot(self, st: TrainerState) -> dict:
        """The curriculum's state for the in-buffer audit: PLR's buffer, or SFL's D with the
        learnabilities stored at the last refresh (score age = chunks since it)."""
        if self.plr is not None:
            return {"arm": self.ued.arm, "chunk": st.chunk, **buffer_snapshot(st.buffer)}
        if st.refresh is not None:
            return {"arm": self.ued.arm, "chunk": st.chunk, "refresh_chunk": st.refresh_chunk,
                    **st.refresh._asdict()}
        return {"arm": self.ued.arm, "chunk": st.chunk}

    def run(self, st: TrainerState, num_chunks: int,
            callback: Callable[[TrainerState, ChunkLog], Any] | None = None) -> TrainerState:
        for _ in range(num_chunks):
            st, log = self.step(st)
            if callback is not None:
                callback(st, log)
        return st


def _obs_update(rms, raw_obs, mask):
    x = raw_obs.reshape(-1, raw_obs.shape[-1])
    return rms.update(x) if mask is None else rms.update_masked(x, mask.reshape(-1))


def _diagnostics(traj, targets, mask):
    """Explained variance of the critic on the trained-on transitions."""
    if mask is None:
        return 1.0 - jnp.var(targets - traj.value) / (jnp.var(targets) + 1e-8)
    w = mask.astype(targets.dtype)
    n = jnp.maximum(w.sum(), 1.0)

    def var(x):
        mu = (w * x).sum() / n
        return (w * (x - mu) ** 2).sum() / n

    return 1.0 - var(targets - traj.value) / (var(targets) + 1e-8)
