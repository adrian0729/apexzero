"""Level scores (UED estimators) computed from audit rollouts.

Conventions follow the reference implementation (JaxUED, itself from minimax), which the
tests diff against: a score is averaged over the time steps of each *completed* episode,
then over episodes. In the audit every window is one complete episode (it ends by
termination or by truncation at the window length), so "completed" always holds; the
-inf "incomplete" convention matters only inside the curriculum's chunked rollouts.

Scores (higher = the sampler wants this level more):
- L1 value loss        mean_t |A_t|
- PVL                  mean_t max(A_t, 0)            (identity: PVL = L1/2 + mean_t A_t / 2)
- MaxMC (implemented)  mean_t (R_max - V_t), R_max the max *undiscounted* return ever seen
                       (floored at 0, as the reference's running max starts at 0)
- MaxMC (unit)         R^gamma_max - V(o_0), with the max *discounted* return
- learnability         p (1 - p), p = success rate over the windows provided
- disagreement         mean_t std_e(V^e_t) over the critic's ensemble heads
A_t is GAE(lambda) with the window's bootstrap: V(o_T) at truncation, 0 at termination.
"""

from __future__ import annotations

import numpy as np

from apexzero.audit.rollouts import WindowRollouts


def gae(ro: WindowRollouts, gamma: float, lam: float) -> np.ndarray:
    """GAE advantages [L, W, T] (zero after the episode ends)."""
    r, v, al = ro.reward, ro.value, ro.alive
    T = r.shape[-1]
    # Next value: V(o_{t+1}) while alive; at the last alive step, the bootstrap value.
    v_next = np.concatenate([v[..., 1:], np.zeros(v.shape[:-1] + (1,))], axis=-1)
    al_next = np.concatenate([al[..., 1:], np.zeros(al.shape[:-1] + (1,))], axis=-1)
    last = (al > 0) & (al_next == 0)
    boot = np.where(ro.terminated[..., None], 0.0, ro.final_value[..., None])
    v_next = np.where(last, boot, v_next)
    delta = (r + gamma * v_next - v) * al
    adv = np.zeros_like(r)
    acc = np.zeros(r.shape[:-1])
    for t in range(T - 1, -1, -1):
        acc = delta[..., t] + gamma * lam * acc * al_next[..., t] * (1 - last[..., t])
        adv[..., t] = acc * al[..., t]
    return adv


def _time_mean(x: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Per-episode mean over alive steps -> [L, W]."""
    return np.sum(x * alive, -1) / np.maximum(np.sum(alive, -1), 1)


def l1_value_loss(ro: WindowRollouts, gamma: float, lam: float) -> np.ndarray:
    return _time_mean(np.abs(gae(ro, gamma, lam)), ro.alive).mean(-1)


def positive_value_loss(ro: WindowRollouts, gamma: float, lam: float) -> np.ndarray:
    return _time_mean(np.maximum(gae(ro, gamma, lam), 0.0), ro.alive).mean(-1)


def undiscounted_returns(ro: WindowRollouts) -> np.ndarray:
    return np.sum(ro.reward * ro.alive, -1)


def maxmc_implemented(ro: WindowRollouts, max_return: np.ndarray | None = None) -> np.ndarray:
    """mean_t (R_max - V_t); R_max = max undiscounted return over these windows and any
    earlier ones passed in ``max_return`` [L] (the buffer's stored maximum).

    Matches the reference exactly, including a quirk: JaxUED's ``compute_max_returns``
    starts its running maximum at 0, so R_max is floored at 0 when every return is
    negative."""
    r_max = np.maximum(undiscounted_returns(ro).max(-1), 0.0)
    if max_return is not None:
        r_max = np.maximum(r_max, max_return)
    return _time_mean(r_max[:, None, None] - ro.value, ro.alive).mean(-1)


def maxmc_unit(ro: WindowRollouts, gamma: float, max_return: np.ndarray | None = None) -> np.ndarray:
    """R^gamma_max - V(o_0): discounted, so it is in the same units as the value."""
    r_max = ro.returns(gamma).max(-1)
    if max_return is not None:
        r_max = np.maximum(r_max, max_return)
    return r_max - ro.value[..., 0].mean(-1)


def success_completed(ro: WindowRollouts) -> np.ndarray:
    """Success = the window ended by truncation, not termination."""
    return ~ro.terminated.astype(bool)


def success_distance(ro: WindowRollouts, qss_distance: np.ndarray, fraction: float = 0.97) -> np.ndarray:
    """Non-saturating success: completed *and* covered >= fraction of the QSS distance."""
    return success_completed(ro) & (ro.distance >= fraction * qss_distance[:, None])


def learnability(success: np.ndarray) -> np.ndarray:
    p = success.mean(-1)
    return p * (1.0 - p)


def disagreement(ro: WindowRollouts) -> np.ndarray:
    if ro.heads.shape[-1] == 0:
        return np.full(ro.reward.shape[0], np.nan)
    return _time_mean(ro.heads.std(-1), ro.alive).mean(-1)
