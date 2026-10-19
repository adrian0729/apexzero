"""In-training level scoring: the chunk accumulator of spec section 0.4.

Each chunk (128 steps in curriculum C) is scored right after its rollout, in raw reward
units, with the critic that produced it: V = sigma * V_hat, where sigma is the return
normaliser's std at the start of the chunk (DEV-6). Advantages come from
``audit.estimators.scoring_gae``, the same function the offline audit calls, cut at the
chunk end and bootstrapped there from b = V(s_T) under the same parameters (DEV-2). Mode F
(the window end is terminal, as in every reference codebase) is the headline; mode B
(bootstrap V(final_obs)) and the other lambdas are stored as variants.

A forward pass then adds each step to its slot's running sums. When the slot's episode
ends, one record is emitted and the sums reset. An episode still running at the chunk end
is never scored: it carries into the next chunk (DEV-4), so every record is a completed
episode and no -inf "no episode completed" score can arise.

Records are columnar (dict of 1-D NumPy arrays, one row per completed episode):

    slot, level, chunk, t     where the episode ended (t: step within the chunk)
    T                         steps, including the terminal transition
    crashed, end_of_strip     termination / truncation at the strip end (must never happen)
    s1                        window completed (not crashed, not end of strip; spec 1.5 S1)
    distance                  metres along the reference
    G, G_disc                 undiscounted and discounted (no bootstrap) return
    V_first, V_bar            critic at the first observation, and its episode mean
    pvl, l1, mean_a           headline (lambda_PPO, mode F) per-episode time means
    pvl_{lam}_{mode}, ...     every variant
    dis                       mean_t std_k(V^k_t) over the ensemble heads (nan without heads)

MNA's clipped-GAE term is not accumulated in training (it is O(T^2) per slot and chunk);
the offline audit computes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from apexzero.audit.estimators import LAMBDAS, scoring_gae


class ChunkData(NamedTuple):
    """One chunk on the host, time-major [T, N] (heads [T, N, E])."""
    reward: np.ndarray        # raw reward
    value: np.ndarray         # V_hat(o_t), normalised units
    heads: np.ndarray         # ensemble heads, normalised units
    terminated: np.ndarray
    truncated: np.ndarray
    final_value: np.ndarray   # V_hat(final_obs) (used at truncations, mode B)
    last_value: np.ndarray    # [N] V_hat(o_T): the chunk-end bootstrap
    level: np.ndarray         # pool level of each transition
    progress: np.ndarray      # metres gained per step
    end_of_strip: np.ndarray


def variant_key(stat: str, lam: float, mode: str) -> str:
    return f"{stat}_{lam:g}_{mode}"


@dataclass
class Accumulator:
    """Per-slot running sums of the episode in flight (spec 0.4), carried across chunks."""
    n_slots: int
    n_variants: int
    n: np.ndarray = field(init=False)
    sum_v: np.ndarray = field(init=False)
    v_first: np.ndarray = field(init=False)
    sum_r: np.ndarray = field(init=False)
    sum_rdisc: np.ndarray = field(init=False)
    disc: np.ndarray = field(init=False)       # gamma^tau for the next step
    sum_dis: np.ndarray = field(init=False)
    distance: np.ndarray = field(init=False)
    sum_pos: np.ndarray = field(init=False)    # [N, V] sum max(A, 0) per variant
    sum_abs: np.ndarray = field(init=False)    # [N, V] sum |A|
    sum_a: np.ndarray = field(init=False)      # [N, V] sum A

    def __post_init__(self):
        for name in ("n", "sum_v", "v_first", "sum_r", "sum_rdisc", "sum_dis", "distance"):
            setattr(self, name, np.zeros(self.n_slots))
        for name in ("sum_pos", "sum_abs", "sum_a"):
            setattr(self, name, np.zeros((self.n_slots, self.n_variants)))
        self.disc = np.ones(self.n_slots)

    def reset(self, rows: np.ndarray) -> None:
        for a in (self.n, self.sum_v, self.v_first, self.sum_r, self.sum_rdisc, self.sum_dis,
                  self.distance, self.sum_pos, self.sum_abs, self.sum_a):
            a[rows] = 0.0
        self.disc[rows] = 1.0


class ChunkScorer:
    """Scores consecutive chunks of the same N slots (``score`` keeps the carry)."""

    def __init__(self, n_slots: int, gamma: float = 0.995, lambdas=LAMBDAS, headline_lambda: float = 0.95,
                 modes=("F", "B")):
        if headline_lambda not in lambdas or "F" not in modes:
            raise ValueError("the headline (lambda_PPO, mode F) must be among the variants")
        self.gamma = gamma
        self.variants = [(lam, mode) for lam in lambdas for mode in modes]
        self.headline = self.variants.index((headline_lambda, "F"))
        self.acc = Accumulator(n_slots, len(self.variants))

    def advantages(self, d: ChunkData, sigma: float) -> np.ndarray:
        """[V, N, T] scoring advantages of every variant, in raw units."""
        T, N = d.reward.shape
        done = d.terminated | d.truncated
        # Slot-major with one extra step holding the bootstrap b = V(s_T); the cut after
        # the last real step (unless it ended the episode) makes delta_{T-1} use b and stops
        # the lambda-sum, exactly as at a chunk boundary in the offline audit.
        pad = lambda a, fill: np.concatenate([a.T, np.full((N, 1), fill, a.dtype)], axis=1)
        r = pad(d.reward.astype(float), 0.0)
        v = sigma * np.concatenate([d.value.T, d.last_value[:, None]], axis=1).astype(float)
        vf = pad(sigma * d.final_value.astype(float), 0.0)
        term, trunc = pad(d.terminated, False), pad(d.truncated, False)
        cut = np.zeros((N, T + 1), bool)
        cut[:, T - 1] = ~done[T - 1]
        return np.stack([scoring_gae(r, v, term, trunc, vf, cut, self.gamma, lam, mode)[:, :T]
                         for lam, mode in self.variants])

    def score(self, d: ChunkData, sigma: float, chunk: int) -> tuple[dict, np.ndarray]:
        """Add one chunk; return (records of the episodes that ended in it, headline A [T, N])."""
        T, N = d.reward.shape
        adv = self.advantages(d, sigma)
        v = sigma * d.value.astype(float)
        has_heads = d.heads.shape[-1] > 0
        spread = sigma * d.heads.astype(float).std(-1) if has_heads else np.zeros((T, N))
        done = d.terminated | d.truncated
        acc, out = self.acc, []
        for t in range(T):
            first = acc.n == 0
            acc.v_first[first] = v[t, first]
            acc.n += 1
            acc.sum_v += v[t]
            acc.sum_r += d.reward[t]
            acc.sum_rdisc += acc.disc * d.reward[t]
            acc.disc *= self.gamma
            acc.distance += d.progress[t]
            acc.sum_dis += spread[t]
            a_t = adv[:, :, t].T                      # [N, V]
            acc.sum_pos += np.maximum(a_t, 0.0)
            acc.sum_abs += np.abs(a_t)
            acc.sum_a += a_t
            rows = np.nonzero(done[t])[0]
            if rows.size:
                out.append(self._emit(rows, t, chunk, d, has_heads))
                acc.reset(done[t])
        if not out:
            out.append(self._emit(np.zeros(0, np.int64), 0, chunk, d, has_heads))
        return {k: np.concatenate([p[k] for p in out]) for k in out[0]}, adv[self.headline].T

    def _emit(self, rows, t, chunk, d: ChunkData, has_heads: bool) -> dict:
        acc = self.acc
        n = acc.n[rows]
        crashed = d.terminated[t, rows].astype(bool)
        eos = d.end_of_strip[t, rows].astype(bool)
        rec = {
            "slot": rows.astype(np.int32), "level": d.level[t, rows].astype(np.int32),
            "chunk": np.full(rows.size, chunk, np.int32), "t": np.full(rows.size, t, np.int32),
            "T": n.astype(np.int32), "crashed": crashed, "end_of_strip": eos, "s1": ~crashed & ~eos,
            "distance": acc.distance[rows].copy(), "G": acc.sum_r[rows].copy(),
            "G_disc": acc.sum_rdisc[rows].copy(), "V_first": acc.v_first[rows].copy(),
            "V_bar": acc.sum_v[rows] / n,
            "dis": acc.sum_dis[rows] / n if has_heads else np.full(rows.size, np.nan),
        }
        for k, (lam, mode) in enumerate(self.variants):
            rec[variant_key("pvl", lam, mode)] = acc.sum_pos[rows, k] / n
            rec[variant_key("l1", lam, mode)] = acc.sum_abs[rows, k] / n
            rec[variant_key("mean_a", lam, mode)] = acc.sum_a[rows, k] / n
        h = self.variants[self.headline]
        for stat in ("pvl", "l1", "mean_a"):
            rec[stat] = rec[variant_key(stat, *h)]
        return rec
