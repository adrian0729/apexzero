"""Level scores (UED estimators), following docs/paper/score-spec.md (`audit-v1`).

Single-trajectory primitives are shared by the offline audit (E1, below) and the
in-training racing-PLR scorer, so both compute identical numbers:

- ``scoring_gae``: GAE in *raw* reward units (values already multiplied by sigma, DEV-6).
  The window end is terminal in the headline "mode F" (as in every reference codebase)
  and bootstrapped from V(final_obs) in "mode B". Chunk boundaries (kappa_t) cut the
  lambda-sum and bootstrap from V_{t+1} (DEV-2).
- ``mna_terms``: MNA's clipped-GAE G_hat in closed form (exact for lambda = 1, where the
  reference code's 1/(1 - lambda) form divides by zero).

Per-window (episode) records are then aggregated per level (spec section 1.8):

    L1 = mean_e L1_e, PVL = mean_e PVL_e
    MaxMC (as implemented) = max(0, max_e G_e) - mean_e Vbar_e           (undiscounted R_max)
    MaxMC-unit = mean_e (max_e' G^gamma_e' - V_first,e)                  (no floor, DEV-22)
    learnability = p (1 - p),  p = mean_e success_e  (S1 or S2, indicators, DEV-9)
    MNA = C_hat * sum_e sum_t G_hat_t / sum_e T_e,  C_hat = 1[any S1]    (DEV-11, DEV-12)
    disagreement = mean_e mean_t std_k(V^k_t)  (population std, main critic excluded)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from apexzero.audit.rollouts import WindowRollouts

CHUNK = 128
LAMBDAS = (0.9, 0.95, 0.98, 1.0)


# --------------------------------------------------------------------------- primitives


def _next_values(v, term, trunc, vf, mode: str):
    """V used for s_{t+1}: 0 on termination; 0 (mode F) or V^f (mode B) on truncation."""
    nxt = np.concatenate([v[..., 1:], np.zeros_like(v[..., :1])], axis=-1)
    nxt = np.where(trunc, vf if mode == "B" else 0.0, nxt)
    return np.where(term, 0.0, nxt)


def scoring_gae(r, v, term, trunc, vf, cut, gamma, lam, mode: str = "F"):
    """GAE for one trajectory, or a batch of them (time is the last axis; ``vf`` is a
    scalar or broadcasts against ``r``). ``cut[..., t]`` = 1 where a chunk ends after step t
    (and the episode does not): the lambda-sum restarts and V_{t+1} bootstraps it. The
    in-training racing-PLR scorer calls this on all slots of a chunk at once."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    term, trunc, cut = (np.asarray(a, bool) for a in (term, trunc, cut))
    stop = term | trunc | cut
    delta = r + gamma * _next_values(v, term, trunc, vf, mode) - v
    adv = np.zeros_like(r)
    acc = np.zeros(r.shape[:-1])
    for t in range(r.shape[-1] - 1, -1, -1):
        acc = delta[..., t] + gamma * lam * np.where(stop[..., t], 0.0, 1.0) * acc
        adv[..., t] = acc
    return adv


def mna_terms(r, v, term, trunc, vf, cut, gamma, lam, mode: str = "F"):
    """Closed-form G_hat_s (spec 1.6) over segments delimited by episode ends and cuts.

    A_s^(n) = sum_{i<n} gamma^i delta_{s+i};  M_n = max(0, A^(1..n));  G^(n) = M_n - A^(n);
    G_hat_s = sum_{n=1}^{N_s - 1} (1 - lam) lam^{n-1} G^(n) + lam^{N_s - 1} G^(N_s).
    """
    r, v = np.asarray(r, float), np.asarray(v, float)
    term, trunc, cut = (np.asarray(a, bool) for a in (term, trunc, cut))
    delta = r + gamma * _next_values(v, term, trunc, vf, mode) - v
    ends = np.nonzero(term | trunc | cut)[0]
    bounds = list(ends + 1)
    if not bounds or bounds[-1] != len(r):
        bounds.append(len(r))
    out = np.zeros_like(r)
    start = 0
    for stop in bounds:
        d = delta[start:stop]
        N = len(d)
        if N == 0:
            continue
        idx = np.arange(N)
        n = idx[None, :] + 1                                  # horizon n = 1..N
        valid = idx[:, None] + idx[None, :] < N               # s + n - 1 < N
        gidx = np.clip(idx[:, None] + idx[None, :], 0, N - 1)
        terms = np.where(valid, gamma ** idx[None, :] * d[gidx], 0.0)
        A = np.cumsum(terms, axis=1)
        M = np.maximum.accumulate(np.maximum(A, 0.0), axis=1)
        G = M - A
        Ns = N - idx                                           # N_s for each start s
        w = np.where(n < Ns[:, None], (1 - lam) * lam ** (n - 1),
                     np.where(n == Ns[:, None], lam ** (Ns[:, None] - 1), 0.0))
        out[start:stop] = np.sum(np.where(valid, w * G, 0.0), axis=1)
        start = stop
    return out


def mna_terms_batch(r, v, term, trunc, vf, cut, gamma, lam, mode: str = "F", block: int = 64):
    """``mna_terms`` for a batch of trajectories [B, T] at once (the in-training scorer
    calls it on every slot of a chunk). Same closed form, with each start's horizon N_s
    taken to the next episode end, cut or the trajectory end; O(T^2) per row, ``block``
    rows at a time to bound memory."""
    r, v = np.atleast_2d(np.asarray(r, float)), np.atleast_2d(np.asarray(v, float))
    term, trunc, cut = (np.atleast_2d(np.asarray(a, bool)) for a in (term, trunc, cut))
    delta = r + gamma * _next_values(v, term, trunc, vf, mode) - v
    B, T = delta.shape
    stop = term | trunc | cut
    stop[:, -1] = True                                         # the trajectory end closes a segment
    t = np.arange(T)
    nxt = np.minimum.accumulate(np.where(stop, t, T)[:, ::-1], axis=1)[:, ::-1]
    Ns = nxt - t + 1                                            # [B, T] horizon of each start
    n = t + 1                                                   # horizon n = 1..T (offset i = n - 1)
    gidx = np.minimum(t[:, None] + t[None, :], T - 1)          # [T, T] step s + i
    disc = gamma ** t
    out = np.empty((B, T))
    for b0 in range(0, B, block):
        d, ns = delta[b0:b0 + block], Ns[b0:b0 + block, :, None]
        terms = np.where(t < ns, disc * d[:, gidx], 0.0)        # [b, T, T]
        A = np.cumsum(terms, axis=2)
        G = np.maximum.accumulate(np.maximum(A, 0.0), axis=2) - A
        w = np.where(n < ns, (1 - lam) * lam ** (n - 1), np.where(n == ns, lam ** (ns - 1), 0.0))
        out[b0:b0 + block] = np.sum(w * G, axis=2)
    return out


# --------------------------------------------------------------------------- per-window records


class WindowStats(NamedTuple):
    """Per (level, window) quantities in raw units; arrays [L, W] unless noted."""
    T: np.ndarray            # steps (includes the terminal transition)
    G: np.ndarray            # undiscounted return
    G_disc: np.ndarray       # discounted return (no bootstrap)
    V_first: np.ndarray
    V_bar: np.ndarray
    pvl: np.ndarray          # [len(lambdas), L, W] per-episode time means
    l1: np.ndarray
    mean_a: np.ndarray
    pvl_disc: np.ndarray     # discount-weighted variants at lambdas[headline]
    l1_disc: np.ndarray
    mna_sum: np.ndarray      # sum_t G_hat_t
    dis: np.ndarray          # mean_t std_k(V^k_t) (nan without heads)
    completed: np.ndarray    # S1: ended by truncation, not termination
    distance: np.ndarray


def chunk_cuts(T: int, phase: int, chunk: int = CHUNK) -> np.ndarray:
    """cut[t] = 1 where t = phase - 1 (mod chunk): window starts are uniformly phased
    against training chunk boundaries (spec 0.2)."""
    t = np.arange(T)
    return (t % chunk) == ((phase - 1) % chunk)


def window_stats(ro: WindowRollouts, gamma: float = 0.995, lambdas=LAMBDAS, headline: int = 1,
                 mode: str = "F", phases: np.ndarray | None = None, chunked: bool = True) -> WindowStats:
    L, W, _ = ro.reward.shape
    n_lam = len(lambdas)
    out = {k: np.zeros((L, W)) for k in ("T", "G", "G_disc", "V_first", "V_bar", "pvl_disc",
                                         "l1_disc", "mna_sum", "dis")}
    pvl, l1, mean_a = (np.zeros((n_lam, L, W)) for _ in range(3))
    has_heads = ro.heads.shape[-1] > 0
    for i in range(L):
        for w in range(W):
            T = int(ro.alive[i, w].sum())
            r, v = ro.reward[i, w, :T], ro.value[i, w, :T]
            term = np.zeros(T, bool)
            trunc = np.zeros(T, bool)
            if ro.terminated[i, w]:
                term[-1] = True
            else:
                trunc[-1] = True
            if chunked:
                ph = 0 if phases is None else int(phases[i, w])
                cut = chunk_cuts(T, ph) & ~(term | trunc)
            else:
                cut = np.zeros(T, bool)
            vf = ro.final_value[i, w]
            disc = gamma ** np.arange(T)
            for k, lam in enumerate(lambdas):
                a = scoring_gae(r, v, term, trunc, vf, cut, gamma, lam, mode)
                pvl[k, i, w] = np.maximum(a, 0).mean()
                l1[k, i, w] = np.abs(a).mean()
                mean_a[k, i, w] = a.mean()
                if k == headline:
                    out["pvl_disc"][i, w] = np.sum(disc * np.maximum(a, 0)) / disc.sum()
                    out["l1_disc"][i, w] = np.sum(disc * np.abs(a)) / disc.sum()
            g_hat = mna_terms(r, v, term, trunc, vf, cut, gamma, lambdas[headline], mode)
            out["T"][i, w] = T
            out["G"][i, w] = r.sum()
            out["G_disc"][i, w] = np.sum(disc * r)
            out["V_first"][i, w] = v[0]
            out["V_bar"][i, w] = v.mean()
            out["mna_sum"][i, w] = g_hat.sum()
            out["dis"][i, w] = ro.heads[i, w, :T].std(-1).mean() if has_heads else np.nan
    return WindowStats(out["T"], out["G"], out["G_disc"], out["V_first"], out["V_bar"], pvl, l1,
                       mean_a, out["pvl_disc"], out["l1_disc"], out["mna_sum"], out["dis"],
                       ~ro.terminated.astype(bool), ro.distance)


# --------------------------------------------------------------------------- level scores


def level_scores(st: WindowStats, windows, headline: int = 1, qss_distance: np.ndarray | None = None,
                 s2_fraction: float = 0.97) -> dict[str, np.ndarray]:
    """Scores per level from the given window indices (e.g. range(4) for the operational
    score, odd windows for cross-fitted high-budget scores)."""
    w = np.asarray(list(windows))
    sel = lambda a: a[..., w]
    s1 = sel(st.completed)
    scores = {
        "l1": sel(st.l1[headline]).mean(-1),
        "pvl": sel(st.pvl[headline]).mean(-1),
        "mean_a": sel(st.mean_a[headline]).mean(-1),
        "pvl_disc": sel(st.pvl_disc).mean(-1),
        "l1_disc": sel(st.l1_disc).mean(-1),
        "maxmc": np.maximum(sel(st.G).max(-1), 0.0) - sel(st.V_bar).mean(-1),
        "maxmc_unit": (sel(st.G_disc).max(-1)[:, None] - sel(st.V_first)).mean(-1),
        "learnability_s1": s1.mean(-1) * (1 - s1.mean(-1)),
        "mna": np.where(s1.any(-1), sel(st.mna_sum).sum(-1) / sel(st.T).sum(-1), 0.0),
        "mna_lit_gate": (np.maximum(sel(st.G).max(-1), 0.0) > 0).astype(float),
        "disagreement": sel(st.dis).mean(-1),
        "agent_only": -sel(st.G_disc).mean(-1),
    }
    for k, lam in enumerate(LAMBDAS):
        if k != headline:
            scores[f"pvl_lam{lam}"] = sel(st.pvl[k]).mean(-1)
            scores[f"l1_lam{lam}"] = sel(st.l1[k]).mean(-1)
    if qss_distance is not None:
        s2 = s1 & (sel(st.distance) >= s2_fraction * qss_distance[:, None])
        scores["learnability_s2"] = s2.mean(-1) * (1 - s2.mean(-1))
    return scores


def target_returns(st: WindowStats, windows) -> np.ndarray:
    """J^pi estimate per level: mean discounted return over the given (target) windows."""
    return st.G_disc[..., np.asarray(list(windows))].mean(-1)
