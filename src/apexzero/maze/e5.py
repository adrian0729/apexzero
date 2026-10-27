"""Maze E5's statistics (docs/prereg/maze-e5.md Sections 4 and 5): per checkpoint, per arm and
phase, the two-way bootstrap, the tests, the labels and the outcome rows.

Everything here works on stored per-maze arrays (``Run`` / ``Checkpoint``); nothing trains or
audits. The rules, all fixed by the draft:

- **Per checkpoint.** ρ is Spearman's correlation between a score and a target over the mazes,
  with average ranks for ties; z = atanh(ρ) with ρ clipped to ±0.999. A score (or target)
  that is constant over the mazes carries no ranking information: its ρ is 0 (a convention
  this module fixes; the report counts such cases). RC@10% is the expected share of the
  target's total held by the score's top 10%, ties at the cut-off sharing the remaining
  slots (``apexzero.maze.fidelity``); nRC = (RC - 0.1) / (RC* - 0.1), RC* the target's own
  capture, undefined when RC* - 0.1 < 0.02. The absolute target is regret J* - Ĵ^π on every
  maze; the normalised one is (J* - Ĵ^π) / J* on the mazes with J* > 0.
- **Phase** of a checkpoint: p, the share of its target episodes that reach the goal over the
  mazes with J* > 0. Early p < 0.5, mid 0.5 <= p < 0.9, late p >= 0.9. Fixed from the full
  sample. A phase is reached by an arm when at least 3 of its runs have a checkpoint in it.
- **Per arm and phase.** Each run's mean z over its checkpoints in the phase; the mean over
  the arm's runs that have one, runs weighing equally within a view and the two views
  equally (with 5 runs each this is the plain mean); ρ̄ = tanh(Z). nRC is averaged the same
  way, without the transform, over the checkpoints where it is defined.
- **Δ** = ρ̄(score) - ρ̄(best baseline), the best of agent-only, J* and BFS steps taken in
  each bootstrap draw (the point estimate takes the best point ρ̄).
- **The two-way bootstrap.** B = 4,000 draws. Each resamples the mazes (one draw of 2,000 with
  replacement, shared by every run and checkpoint; the normalised target uses the J* > 0
  mazes within the draw) and the runs (with replacement within each arm x view cell). The
  in-buffer level sets differ per run and checkpoint, so each is resampled on its own. Every
  ρ in a draw is the weighted Spearman ρ of the resampled sample (``RankEngine``: average
  ranks under integer weights equal the ranks of the expanded sample). CIs are 95%
  percentile intervals over the finite draws.
- **p values.** Two-sided: min(1, 2(1 + n_opposite)/(B + 1)), n_opposite the draws at or past
  0 on the other side from the point estimate (p = 1 when the point is 0). One-sided against
  a bound b: (1 + n_beyond)/(B + 1), n_beyond the draws at or beyond b on the null's side.
  TOST for |x| < c: the larger of (1 + #{x <= -c})/(B + 1) and (1 + #{x >= c})/(B + 1). The
  trend test is a two-sided one-sample t-test of the runs' OLS slopes of z on log10(update).
- **Holm** over the primary family (36 tests) and the secondary (28). A test that cannot be
  run (a phase not reached, a cell with fewer than 3 runs) keeps its place with p = 1, so the
  family sizes stay 36 and 28.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from scipy import stats

from apexzero.maze.fidelity import holm, rankable, regret_capture

ARMS = ("dr", "plr", "sfl")
ARM_LABEL = {"dr": "DR", "plr": "PLR-MaxMC", "sfl": "SFL"}
VIEWS = (5, 13)
PHASES = ("early", "mid", "late")
SPANS = PHASES + ("all",)
TARGETS = ("abs", "norm")
PRIMARY = ("pvl", "maxmc", "learnability")
HEADLINE = ("pvl", "maxmc", "l1", "maxmc_unit", "learnability", "mna")
NATIVE = ("pvl_native", "maxmc_native", "learnability_native")
BASELINES = ("agent_only", "j_star", "bfs_steps")
SCORES = HEADLINE + NATIVE + BASELINES        # the scores the bootstrap carries
# S2: each curriculum's stored score, and the fresh native score it is compared with.
STORED = {"plr": ("maxmc", "maxmc_native"), "sfl": ("learnability", "learnability_native")}
LABEL = {"pvl": "PVL", "maxmc": "MaxMC", "l1": "L1 value loss", "maxmc_unit": "MaxMC-unit",
         "learnability": "learnability", "mna": "MNA", "pvl_native": "native PVL",
         "maxmc_native": "native MaxMC", "learnability_native": "native learnability",
         "agent_only": "agent-only", "j_star": "J*", "bfs_steps": "BFS steps"}

B = 4000
LEVEL = 0.95
ALPHA = 0.05
TOP = 0.1
RHO_CLIP = 0.999
NRC_MIN = 0.02
FAITHFUL = dict(rho=0.5, nrc=0.6)
UNINFORMATIVE = dict(rho=0.2, nrc=0.2)
MIN_RUNS = 3                 # a phase is reached / an S3 cell is testable with >= 3 runs
ROW_SHIFT = 0.1              # rows B and B': |rho_last - rho_first| >= 0.1
ROW_EFFECT = 0.2             # rows F and G: |difference| >= 0.2
S1_BOUND = 0.2               # S1's TOST margin for |rho_norm|
BOOT_SEED = 2027


def checkpoint_updates() -> list[int]:
    """u_k = 100 round(3 x 100^((k - 1)/15)), k = 1..16: 300, 400, ..., 30000."""
    return [100 * round(3 * 100 ** ((k - 1) / 15)) for k in range(1, 17)]


def phase_of(p: float) -> str:
    return "early" if p < 0.5 else "mid" if p < 0.9 else "late"


# --------------------------------------------------------------------------- data


class Checkpoint(NamedTuple):
    """One audited checkpoint: per-maze arrays over the shared fresh mazes."""
    update: int
    scores: dict                 # score name -> [N] (-inf allowed, ranks last)
    regret: np.ndarray           # [N] J* - J^pi
    regret_norm: np.ndarray      # [N] (J* - J^pi) / J*, NaN where J* = 0
    reachable: np.ndarray        # [N] bool, J* > 0
    success_tgt: np.ndarray      # [N] share of the target episodes that reach the goal
    inbuf: dict | None = None    # {"stored", "regret", "regret_norm", "reachable"}: [K] each
    extra: dict | None = None    # descriptive only: "j_star", "j_pi", "j_pi_odd", "j_pi_even"


class Run(NamedTuple):
    name: str
    arm: str
    view: int
    seed: int
    checkpoints: list            # [Checkpoint], in update order


def checkpoint_phase(ck: Checkpoint) -> str:
    return phase_of(float(np.mean(ck.success_tgt[ck.reachable])))


# --------------------------------------------------------------------------- ranks


def clip_z(rho):
    return np.arctanh(np.clip(rho, -RHO_CLIP, RHO_CLIP))


class RankEngine:
    """Average ranks of many series over one item set under integer weights (multiplicities).

    For weights w, item i in tie group g ranks W(<g) + (W(g) + 1)/2, which is its average rank
    in the expanded sample; items of weight 0 take no part. ``X`` is [S, n] (finite)."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, float)
        S, n = X.shape
        self.order = np.argsort(X, axis=1, kind="stable")
        xs = np.take_along_axis(X, self.order, 1)
        new = np.ones((S, n), bool)
        new[:, 1:] = xs[:, 1:] != xs[:, :-1]
        end = np.ones((S, n), bool)
        end[:, :-1] = new[:, 1:]
        pos = np.broadcast_to(np.arange(n), (S, n))
        self.first = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
        self.last = np.minimum.accumulate(np.where(end, pos, n - 1)[:, ::-1], axis=1)[:, ::-1]
        self.tied = np.nonzero(~new[:, 1:].all(1))[0]          # series with any tie

    def sorted_ranks(self, w: np.ndarray, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
        """w [D, n], shared by every series -> (weights, ranks), each [D, S, n] in each
        series' own sorted order. Without ties a rank is cum - (w - 1)/2; the tie groups are
        resolved only for the series that have them."""
        ws = w[:, self.order].astype(dtype)
        cum = np.cumsum(ws, axis=-1)
        r = cum - (ws - 1) / 2
        t = self.tied
        if t.size:
            c, x = cum[:, t], ws[:, t]
            below = np.take_along_axis(c - x, np.broadcast_to(self.first[t], c.shape), -1)
            upto = np.take_along_axis(c, np.broadcast_to(self.last[t], c.shape), -1)
            r[:, t] = below + (upto - below + 1) / 2
        return ws, r

    def ranks(self, w: np.ndarray) -> np.ndarray:
        """w [..., S, n] (or [S, n]): ranks in the items' own order, same shape."""
        order = np.broadcast_to(self.order, w.shape)
        ws = np.take_along_axis(w, order, -1).astype(float)
        cum = np.cumsum(ws, axis=-1)
        below = np.take_along_axis(cum - ws, np.broadcast_to(self.first, w.shape), -1)
        upto = np.take_along_axis(cum, np.broadcast_to(self.last, w.shape), -1)
        r_sorted = below + (upto - below + 1) / 2
        out = np.empty_like(r_sorted)
        np.put_along_axis(out, order, r_sorted, -1)
        return out


def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Along the last axis; 0 where either side has no spread (the constant-score rule)."""
    sw = w.sum(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = (w * x).sum(-1, keepdims=True) / sw
        my = (w * y).sum(-1, keepdims=True) / sw
        dx, dy = x - mx, y - my
        vx, vy = (w * dx * dx).sum(-1), (w * dy * dy).sum(-1)
        cov = (w * dx * dy).sum(-1)
        den = np.sqrt(vx * vy)
        tol = 1e-9 * np.maximum(sw[..., 0], 1) ** 3
        return np.where((vx > tol) & (vy > tol), cov / np.where(den > 0, den, 1.0), 0.0)


def weighted_spearman(score: np.ndarray, target: np.ndarray, w: np.ndarray | None = None):
    """Spearman ρ of one score and one target (1-D) under weights w (default all 1)."""
    w = np.ones(len(score)) if w is None else np.asarray(w, float)
    rs = RankEngine(score[None]).ranks(w[None])[0]
    rt = RankEngine(target[None]).ranks(w[None])[0]
    return float(weighted_pearson(rs, rt, w))


# --------------------------------------------------------------------------- per run


def _score_matrix(run: Run, scores: Sequence[str]) -> np.ndarray:
    """[K, S, N]: every checkpoint's scores, -inf moved below the finite values."""
    return np.stack([np.stack([rankable(s, ck.scores[s]) for s in scores])
                     for ck in run.checkpoints])


def _targets(run: Run) -> np.ndarray:
    """[K, 2, N]: absolute regret, and normalised regret with 0 where J* = 0 (weight 0)."""
    return np.stack([np.stack([ck.regret, np.where(ck.reachable, ck.regret_norm, 0.0)])
                     for ck in run.checkpoints])


def run_rho(run: Run, maze_w: np.ndarray, scores: Sequence[str] = SCORES,
            buf_w: list | None = None, dtype=np.float64) -> tuple[np.ndarray, np.ndarray | None]:
    """ρ for every checkpoint, score and target under maze weights ``maze_w`` [D, N] (D draws;
    all ones for the point estimate): returns [D, K, S, 2], and, with ``buf_w`` (a list over
    checkpoints of [D, K_c] weights), the in-buffer stored score's ρ [D, K, 2].

    The J* > 0 mazes are the same at every checkpoint (one maze set, one time limit), so one
    weight vector per target serves every series. The weighted mean of average ranks is
    exactly (sum w + 1)/2, and the products are summed in each score's sorted order, with
    the target's ranks gathered into it."""
    X = _score_matrix(run, scores)                         # [K, S, N]
    Y = _targets(run)                                      # [K, 2, N]
    K, S, N = X.shape
    reach = run.checkpoints[0].reachable
    if any(not np.array_equal(ck.reachable, reach) for ck in run.checkpoints):
        raise ValueError(f"{run.name}: the J* > 0 mazes differ between checkpoints")
    D = maze_w.shape[0]
    eng_x = RankEngine(X.reshape(K * S, N))
    kidx = np.repeat(np.arange(K), S)
    out = np.empty((D, K, S, 2))
    for t in range(2):
        w = maze_w if t == 0 else maze_w * reach[None]
        ws, rx = eng_x.sorted_ranks(w, dtype)                          # [D, KS, N]
        eng_y = RankEngine(Y[:, t])
        _, ry_s = eng_y.sorted_ranks(w, dtype)                         # [D, K, N]
        ry = np.empty_like(ry_s)
        np.put_along_axis(ry, np.broadcast_to(eng_y.order, ry_s.shape), ry_s, -1)
        m = ((w.sum(-1) + 1) / 2).astype(dtype)[:, None, None]
        dy = ry - m
        vy = (w[:, None, :] * dy * dy).sum(-1)                         # [D, K]
        dyx = dy[:, kidx[:, None], eng_x.order]                        # [D, KS, N]
        dx = rx - m
        cov, vx = (ws * dx * dyx).sum(-1), (ws * dx * dx).sum(-1)
        vyx = vy[:, kidx]
        tol = 1e-9 * np.maximum(w.sum(-1), 1)[:, None] ** 3
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.where((vx > tol) & (vyx > tol), cov / np.sqrt(vx * vyx), 0.0)
        out[..., t] = rho.reshape(D, K, S)
    buf = None
    if buf_w is not None:
        buf = np.empty((D, K, 2))
        for k, ck in enumerate(run.checkpoints):
            ib = ck.inbuf
            xs = rankable("stored", ib["stored"])
            ys = np.stack([ib["regret"], np.where(ib["reachable"], ib["regret_norm"], 0.0)])
            for t in range(2):
                wt = buf_w[k] * (ib["reachable"][None] if t else 1.0)       # [D, K_c]
                rx = RankEngine(xs[None]).ranks(wt[:, None, :])[:, 0]
                ry = RankEngine(ys[t][None]).ranks(wt[:, None, :])[:, 0]
                buf[:, k, t] = weighted_pearson(rx, ry, wt)
    return out, buf


def capture_stats(ck: Checkpoint, score: str, target: str) -> tuple[float, float, float]:
    """(RC@10%, RC*, nRC) of one score on one target (nRC NaN when RC* - 0.1 < 0.02)."""
    sel = slice(None) if target == "abs" else ck.reachable
    y = (ck.regret if target == "abs" else ck.regret_norm)[sel]
    s = rankable(score, ck.scores[score])[sel]
    rc, rc_star = float(regret_capture(s, y, TOP)), float(regret_capture(y, y, TOP))
    nrc = (rc - TOP) / (rc_star - TOP) if rc_star - TOP >= NRC_MIN else float("nan")
    return rc, rc_star, nrc


# --------------------------------------------------------------------------- aggregation


def arm_mean(values: np.ndarray, counts: np.ndarray, runs: Sequence[Run], arm: str,
             views: Sequence[int] = VIEWS) -> np.ndarray:
    """values [..., R] (NaN where a run has none) and run multiplicities counts [..., R] ->
    the arm's mean: runs equal within a view, the views equal; NaN where no run has one."""
    per_view = []
    for v in views:
        idx = [i for i, r in enumerate(runs) if r.arm == arm and r.view == v]
        if not idx:
            continue
        x, c = values[..., idx], counts[..., idx]
        ok = np.isfinite(x) & (c > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_view.append(np.where(ok, x * c, 0).sum(-1) / np.where(ok, c, 0).sum(-1))
    if not per_view:
        return np.full(values.shape[:-1], np.nan)
    pv = np.stack(per_view)
    ok = np.isfinite(pv)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, pv, 0).sum(0) / ok.sum(0)


def p_two_sided(point: float, draws: np.ndarray) -> float:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if not np.isfinite(point) or d.size == 0:
        return float("nan")
    if point == 0:
        return 1.0
    opposite = np.sum(d <= 0) if point > 0 else np.sum(d >= 0)
    return float(min(1.0, 2 * (1 + opposite) / (d.size + 1)))


def p_one_sided(draws: np.ndarray, bound: float = 0.0, alternative: str = "greater") -> float:
    """H1: x > bound ('greater') counts the draws <= bound; 'less' the draws >= bound."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    n = np.sum(d <= bound) if alternative == "greater" else np.sum(d >= bound)
    return float((1 + n) / (d.size + 1))


def p_tost(draws: np.ndarray, margin: float) -> float:
    """TOST for |x| < margin: the larger of the two one-sided bootstrap p values."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    lo = (1 + np.sum(d <= -margin)) / (d.size + 1)
    hi = (1 + np.sum(d >= margin)) / (d.size + 1)
    return float(max(lo, hi))


def ci(draws: np.ndarray, level: float = LEVEL) -> tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan")
    q = 100 * (1 - level) / 2
    return float(np.percentile(d, q)), float(np.percentile(d, 100 - q))


def trend_test(slopes: Sequence[float]) -> tuple[float, float, int]:
    """(mean slope, two-sided one-sample t-test p against 0, runs)."""
    s = np.asarray([x for x in slopes if np.isfinite(x)], float)
    if s.size < 2:
        return float("nan"), float("nan"), int(s.size)
    if np.all(s == s[0]):
        return float(s[0]), (1.0 if s[0] == 0 else 0.0), int(s.size)
    return float(s.mean()), float(stats.ttest_1samp(s, 0.0).pvalue), int(s.size)


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    dx = x - x.mean()
    return float((dx * (y - y.mean())).sum() / (dx * dx).sum())


def holm_fixed(p: Sequence[float]) -> np.ndarray:
    """Holm over the whole family; an untestable test (NaN) counts as p = 1."""
    p = np.asarray(p, float)
    return holm(np.where(np.isfinite(p), p, 1.0))


# --------------------------------------------------------------------------- the bootstrap


def draw_weights(runs: Sequence[Run], n_mazes: int, n_boot: int = B, seed: int = BOOT_SEED):
    """The draws: maze multiplicities [B, N] (one resample of N with replacement each) and run
    multiplicities [B, R] (with replacement within each arm x view cell)."""
    rng = np.random.default_rng(seed)
    maze = np.stack([np.bincount(rng.integers(0, n_mazes, n_mazes), minlength=n_mazes)
                     for _ in range(n_boot)]).astype(np.int16)
    cells: dict = {}
    for i, r in enumerate(runs):
        cells.setdefault((r.arm, r.view), []).append(i)
    run = np.zeros((n_boot, len(runs)), np.int32)
    for idx in cells.values():
        pick = rng.integers(0, len(idx), (n_boot, len(idx)))
        for j, i in enumerate(idx):
            run[:, i] = (pick == j).sum(1)
    return maze, run


def buffer_weights(draws: range, size: int, seed: int, run_index: int, k: int) -> np.ndarray:
    """In-buffer multiplicities [len(draws), size] for one run's checkpoint k: draw b has
    its own stream (seed, 1, run, k, b), so the result does not depend on the chunking."""
    return np.stack([np.bincount(np.random.default_rng([seed, 1, run_index, k, b])
                                 .integers(0, size, size), minlength=size)
                     for b in draws]).astype(float)


def run_spans(run: Run, z: np.ndarray, phases: Sequence[str]) -> np.ndarray:
    """z [..., K, ...] over checkpoints -> [..., ..., 4]: mean z over the run's checkpoints in
    each of early, mid, late (NaN if none) and over all of them (axis 1 is K)."""
    out = []
    for span in SPANS:
        sel = [k for k, ph in enumerate(phases) if span == "all" or ph == span]
        out.append(z[:, sel].mean(1) if sel else np.full(z[:, 0].shape, np.nan))
    return np.stack(out, axis=-1)


def run_draws(run: Run, run_index: int, maze_w: np.ndarray, seed: int = BOOT_SEED,
              chunk: int | None = None) -> dict[str, np.ndarray]:
    """One run's share of the bootstrap: Z [B, S, 2, 4] (mean z per span) and, if the run has
    in-buffer data, Zbuf [B, 2] (mean z over all checkpoints). ``chunk`` draws at a time
    bound the memory (by default about 2M ranks per array: 5 draws at 16 checkpoints, 12
    scores and 2,000 mazes)."""
    phases = [checkpoint_phase(ck) for ck in run.checkpoints]
    n_boot = maze_w.shape[0]
    if chunk is None:
        chunk = max(1, int(2e6 // (len(run.checkpoints) * len(SCORES) * maze_w.shape[1])))
    has_buf = all(ck.inbuf is not None for ck in run.checkpoints)
    Z = np.empty((n_boot, len(SCORES), 2, len(SPANS)), np.float32)
    Zb = np.empty((n_boot, 2), np.float32) if has_buf else None
    for b0 in range(0, n_boot, chunk):
        sl = slice(b0, min(b0 + chunk, n_boot))
        bw = ([buffer_weights(range(sl.start, sl.stop), len(ck.inbuf["stored"]), seed,
                              run_index, k) for k, ck in enumerate(run.checkpoints)]
              if has_buf else None)
        rho, buf = run_rho(run, maze_w[sl].astype(np.float32), SCORES, bw, np.float32)
        Z[sl] = run_spans(run, clip_z(rho), phases)
        if has_buf:
            Zb[sl] = clip_z(buf).mean(1)
    out = {"Z": Z}
    if has_buf:
        out["Zbuf"] = Zb
    return out


def _worker(args):
    run, i, maze_w, seed = args
    return run_draws(run, i, maze_w, seed)


def bootstrap(runs: Sequence[Run], n_boot: int = B, seed: int = BOOT_SEED,
              workers: int = 1) -> dict[str, np.ndarray]:
    """Every run's draws: Z [B, R, S, 2, 4], Zbuf [B, R, 2] (NaN without in-buffer data),
    and the run multiplicities [B, R]. ``workers`` processes split the runs (at most 4)."""
    n = len(runs[0].checkpoints[0].regret)
    maze_w, run_w = draw_weights(runs, n, n_boot, seed)
    jobs = [(r, i, maze_w, seed) for i, r in enumerate(runs)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=min(int(workers), 4)) as ex:
            parts = list(ex.map(_worker, jobs))
    else:
        parts = [_worker(j) for j in jobs]
    Z = np.stack([p["Z"] for p in parts], axis=1)
    Zb = np.stack([p["Zbuf"] if "Zbuf" in p else np.full((n_boot, 2), np.nan, np.float32)
                   for p in parts], axis=1)
    return {"Z": Z, "Zbuf": Zb, "run_w": run_w}


def point(runs: Sequence[Run]) -> dict[str, np.ndarray]:
    """The full-sample counterparts: rho [R, K, S, 2], Z [R, S, 2, 4], Zbuf [R, 2], nRC
    [R, K, S, 2], phases [R][K], updates [R][K]."""
    rho, zs, zb, nrc, phases = [], [], [], [], []
    for r in runs:
        n = len(r.checkpoints[0].regret)
        has_buf = all(ck.inbuf is not None for ck in r.checkpoints)
        bw = [np.ones((1, len(ck.inbuf["stored"]))) for ck in r.checkpoints] if has_buf else None
        p, buf = run_rho(r, np.ones((1, n)), SCORES, bw)
        ph = [checkpoint_phase(ck) for ck in r.checkpoints]
        rho.append(p[0])
        zs.append(run_spans(r, clip_z(p), ph)[0])
        zb.append(clip_z(buf).mean(1)[0] if has_buf else np.full(2, np.nan))
        nrc.append(np.array([[[capture_stats(ck, s, t)[2] for t in TARGETS] for s in SCORES]
                             for ck in r.checkpoints]))
        phases.append(ph)
    return {"rho": np.stack(rho), "Z": np.stack(zs), "Zbuf": np.stack(zb), "nrc": np.stack(nrc),
            "phases": phases, "updates": [[ck.update for ck in r.checkpoints] for r in runs]}


# --------------------------------------------------------------------------- the analysis


def _tanh(x):
    with np.errstate(invalid="ignore"):
        return np.tanh(x)


def reached_phases(runs: Sequence[Run], phases, arm: str) -> list[str]:
    counts = {ph: sum(1 for r, p in zip(runs, phases) if r.arm == arm and ph in p)
              for ph in PHASES}
    return [ph for ph in PHASES if counts[ph] >= MIN_RUNS]


def label(rho: float, nrc: float, delta: float, d_lo: float, d_hi: float,
          holm_sig: bool | None = None) -> tuple[str, bool]:
    """Section 4's labels: (label, nRC flagged). With ``holm_sig`` (the late-phase labels of
    the primary scores) the Δ condition is PF-late's Holm decision instead of the CI."""
    flagged = not np.isfinite(nrc)
    if holm_sig is None:
        d_pos = delta > 0 and d_lo > 0
        d_not = not (d_lo > 0)
    else:
        d_pos = delta > 0 and holm_sig
        d_not = not d_pos
    faithful = rho >= FAITHFUL["rho"] and (flagged or nrc >= FAITHFUL["nrc"]) and d_pos
    uninformative = (rho <= UNINFORMATIVE["rho"] or (not flagged and nrc <= UNINFORMATIVE["nrc"])
                     or d_not)
    return ("faithful" if faithful else "uninformative" if uninformative else "intermediate",
            flagged)


def analyse(runs: Sequence[Run], boot: dict, pt: dict | None = None) -> dict:
    """Every statistic, test, label and row (Sections 4 and 5) from the bootstrap's draws."""
    pt = point(runs) if pt is None else pt
    R = len(runs)
    Zb, Zp, run_w = boot["Z"], pt["Z"], boot["run_w"]               # [B,R,S,2,4], [R,S,2,4]
    ones = np.ones(R)
    si = {s: i for i, s in enumerate(SCORES)}
    sp = {s: i for i, s in enumerate(SPANS)}
    phases = pt["phases"]

    def rb(arm, s, t, span, draws=True, view=None):
        """ρ̄ (point, or draws [B]) of one arm, score, target and span."""
        views = VIEWS if view is None else (view,)
        if draws:
            return _tanh(arm_mean(np.moveaxis(Zb[:, :, si[s], t, sp[span]], 1, -1), run_w,
                                  runs, arm, views))
        return float(_tanh(arm_mean(Zp[:, si[s], t, sp[span]], ones, runs, arm, views)))

    def nrc_bar(arm, s, t, span):
        vals = np.full(R, np.nan)
        for i, r in enumerate(runs):
            ks = [k for k, ph in enumerate(phases[i]) if span == "all" or ph == span]
            x = pt["nrc"][i, ks, si[s], t] if ks else np.array([])
            x = x[np.isfinite(x)]
            vals[i] = x.mean() if x.size else np.nan
        return float(arm_mean(vals, ones, runs, arm))

    def delta(arm, s, t, span):
        pts = {b: rb(arm, b, t, span, draws=False) for b in BASELINES}
        best = max(pts, key=lambda b: -np.inf if not np.isfinite(pts[b]) else pts[b])
        d_pt = rb(arm, s, t, span, draws=False) - pts[best]
        base = np.stack([rb(arm, b, t, span) for b in BASELINES])
        with np.errstate(invalid="ignore"):
            d_draws = rb(arm, s, t, span) - np.nanmax(np.where(np.isfinite(base), base, -np.inf),
                                                      axis=0)
        d_draws = np.where(np.isfinite(d_draws), d_draws, np.nan)
        return d_pt, d_draws, best

    out: dict = {"arms": {}, "primary": [], "secondary": [], "rows": [], "notes": []}
    constant = sum(int(np.ptp(rankable(s, ck.scores[s])[sel]) == 0) for r in runs
                   for ck in r.checkpoints for s in SCORES for sel in (slice(None), ck.reachable))
    out["notes"].append(f"{constant} score x checkpoint x target cells had a constant score "
                        "(ρ = 0 by the constant-score rule)")
    reached = {a: reached_phases(runs, phases, a) for a in ARMS}
    for arm in ARMS:
        if not any(r.arm == arm for r in runs):
            continue
        cells = {}
        for s in SCORES:
            for ti, t in enumerate(TARGETS):
                for span in SPANS:
                    if span != "all" and span not in reached[arm]:
                        continue
                    d_pt, d_draws, best = delta(arm, s, ti, span)
                    lo, hi = ci(d_draws)
                    rho_pt = rb(arm, s, ti, span, draws=False)
                    r_lo, r_hi = ci(rb(arm, s, ti, span))
                    cells[(s, t, span)] = dict(
                        rho=rho_pt, rho_lo=r_lo, rho_hi=r_hi, nrc=nrc_bar(arm, s, ti, span),
                        delta=d_pt, delta_lo=lo, delta_hi=hi, p_delta=p_two_sided(d_pt, d_draws),
                        best_baseline=best)
        out["arms"][arm] = {"reached": reached[arm], "cells": cells,
                            "runs": sum(r.arm == arm for r in runs)}

    # Primary family: PF-trend and PF-late, 3 arms x 3 scores x 2 targets each.
    prim = []
    for arm in ARMS:
        idx = [i for i, r in enumerate(runs) if r.arm == arm]
        for s in PRIMARY:
            for ti, t in enumerate(TARGETS):
                slopes = []
                for i in idx:
                    log_u = np.log10(pt["updates"][i])
                    slopes.append(ols_slope(log_u, clip_z(pt["rho"][i, :, si[s], ti])))
                m, p, n = trend_test(slopes) if idx else (np.nan, np.nan, 0)
                prim.append(dict(test="PF-trend", arm=arm, score=s, target=t, estimate=m, p=p,
                                 runs=n))
        for s in PRIMARY:
            for ti, t in enumerate(TARGETS):
                late = reached[arm][-1] if reached.get(arm) else None
                if late is None:
                    prim.append(dict(test="PF-late", arm=arm, score=s, target=t, phase=None,
                                     estimate=np.nan, p=np.nan))
                    continue
                c = out["arms"][arm]["cells"][(s, t, late)]
                prim.append(dict(test="PF-late", arm=arm, score=s, target=t, phase=late,
                                 estimate=c["delta"], lo=c["delta_lo"], hi=c["delta_hi"],
                                 p=c["p_delta"]))
    for row, ph in zip(prim, holm_fixed([r["p"] for r in prim])):
        row["p_holm"] = float(ph)
        row["significant"] = bool(np.isfinite(row["p"]) and ph < ALPHA)
    out["primary"] = prim

    # Secondary family: S1 (6), S2 (4), S3 (18).
    sec = []
    for arm in ARMS:
        late = reached[arm][-1] if reached.get(arm) else None
        if late is None:
            sec += [dict(test="S1-diff", arm=arm, p=np.nan),
                    dict(test="S1-tost", arm=arm, p=np.nan)]
            continue
        a_pt, n_pt = rb(arm, "maxmc", 0, late, False), rb(arm, "maxmc", 1, late, False)
        diff = rb(arm, "maxmc", 0, late) - rb(arm, "maxmc", 1, late)
        sec.append(dict(test="S1-diff", arm=arm, phase=late, estimate=a_pt - n_pt, lo=ci(diff)[0],
                        hi=ci(diff)[1], p=p_one_sided(diff, 0.0, "greater")))
        norm = rb(arm, "maxmc", 1, late)
        sec.append(dict(test="S1-tost", arm=arm, phase=late, estimate=n_pt, lo=ci(norm)[0],
                        hi=ci(norm)[1], p=p_tost(norm, S1_BOUND)))
    Zbuf_b, Zbuf_p = boot["Zbuf"], pt["Zbuf"]
    for arm, (stored, fresh) in STORED.items():
        for ti, t in enumerate(TARGETS):
            idx = [i for i, r in enumerate(runs) if r.arm == arm]
            if not idx or not np.isfinite(Zbuf_p[idx, ti]).any():
                sec.append(dict(test="S2", arm=arm, score=stored, target=t, p=np.nan))
                continue
            ib_pt = float(_tanh(arm_mean(Zbuf_p[:, ti], ones, runs, arm)))
            ib_b = _tanh(arm_mean(Zbuf_b[:, :, ti], run_w, runs, arm))
            fr_pt, fr_b = rb(arm, fresh, ti, "all", False), rb(arm, fresh, ti, "all")
            d = ib_b - fr_b
            sec.append(dict(test="S2", arm=arm, score=stored, fresh=fresh, target=t,
                            inbuf=ib_pt, fresh_rho=fr_pt, estimate=ib_pt - fr_pt, lo=ci(d)[0],
                            hi=ci(d)[1], p=p_two_sided(ib_pt - fr_pt, d)))
    for arm in ARMS:
        n_view = {v: sum(r.arm == arm and r.view == v for r in runs) for v in VIEWS}
        for s in PRIMARY:
            for ti, t in enumerate(TARGETS):
                if min(n_view.values()) < MIN_RUNS:
                    sec.append(dict(test="S3", arm=arm, score=s, target=t, p=np.nan,
                                    runs=n_view))
                    continue
                d_pt = (rb(arm, s, ti, "all", False, 13) - rb(arm, s, ti, "all", False, 5))
                d = rb(arm, s, ti, "all", True, 13) - rb(arm, s, ti, "all", True, 5)
                sec.append(dict(test="S3", arm=arm, score=s, target=t, estimate=d_pt,
                                lo=ci(d)[0], hi=ci(d)[1], p=p_two_sided(d_pt, d), runs=n_view))
    for row, ph in zip(sec, holm_fixed([r["p"] for r in sec])):
        row["p_holm"] = float(ph)
        row["significant"] = bool(np.isfinite(row["p"]) and ph < ALPHA)
    out["secondary"] = sec
    s1 = {}
    for arm in ARMS:
        rows_ = [r for r in sec if r["test"].startswith("S1") and r["arm"] == arm]
        s1[arm] = bool(rows_ and all(r["significant"] for r in rows_))
    out["s1_holds"] = s1

    # Labels in every reached phase, then the outcome rows.
    pf_late = {(r["arm"], r["score"], r["target"]): r for r in prim if r["test"] == "PF-late"}
    pf_trend = {(r["arm"], r["score"], r["target"]): r for r in prim if r["test"] == "PF-trend"}
    for arm, a in out["arms"].items():
        labels = {}
        for (s, t, span), c in a["cells"].items():
            if span == "all":
                continue
            holm_sig = None
            if s in PRIMARY and span == a["reached"][-1]:
                holm_sig = pf_late[(arm, s, t)]["significant"]
            lab, flagged = label(c["rho"], c["nrc"], c["delta"], c["delta_lo"], c["delta_hi"],
                                 holm_sig)
            labels[(s, t, span)] = dict(label=lab, nrc_flagged=flagged,
                                        holm_delta=holm_sig is not None)
        a["labels"] = labels
    for arm in ARMS:
        if arm not in out["arms"]:
            continue
        a = out["arms"][arm]
        reached_a = a["reached"]
        g_rows = [r for r in sec if r["test"] == "S2" and r["arm"] == arm]
        g_targets = [r["target"] for r in g_rows
                     if r["significant"] and abs(r["estimate"]) >= ROW_EFFECT]
        for s in PRIMARY:
            labs = [a["labels"][(s, t, ph)]["label"] for t in TARGETS for ph in reached_a]
            row = dict(arm=arm, score=s, rows=[], targets={})
            shift = {}
            for ti, t in enumerate(TARGETS):
                if len(reached_a) < 2:
                    continue
                first, last = reached_a[0], reached_a[-1]
                d_pt = rb(arm, s, ti, last, False) - rb(arm, s, ti, first, False)
                d = rb(arm, s, ti, last) - rb(arm, s, ti, first)
                lo, hi = ci(d)
                tr = pf_trend[(arm, s, t)]
                shift[t] = dict(first=first, last=last, estimate=d_pt, lo=lo, hi=hi,
                                trend=tr["estimate"], trend_significant=tr["significant"])
            row["shift"] = shift
            b = [t for t, x in shift.items() if x["estimate"] <= -ROW_SHIFT and x["hi"] < 0
                 and x["trend_significant"] and x["trend"] < 0]
            bp = [t for t, x in shift.items() if x["estimate"] >= ROW_SHIFT and x["lo"] > 0
                  and x["trend_significant"] and x["trend"] > 0]
            c_row = bool(labs) and all(x == "uninformative" for x in labs)
            f_t = [r["target"] for r in sec if r["test"] == "S3" and r["arm"] == arm
                   and r["score"] == s and r["significant"] and abs(r["estimate"]) >= ROW_EFFECT]
            g_here = bool(g_targets) and STORED.get(arm, (None,))[0] == s
            # Faithful in every reached phase is A, even with a significant shift: B's reading
            # ("only while the agent is incompetent") overstates a score still faithful late,
            # and B's mirror one that was faithful early (decided 27 Oct 2026). The shift stays
            # in row["shift"]. G still blocks A: the arm's row G (for its stored score) blocks A
            # for every score of the arm, and B and B' are still not reported for a score
            # faithful throughout. Such a score is not "mixed": it is labelled A_WITHHELD.
            faithful_all = bool(labs) and all(x == "faithful" for x in labs)
            if faithful_all:
                b, bp = [], []
            a_row = faithful_all and not g_targets and not c_row
            a_withheld = faithful_all and bool(g_targets) and not c_row
            for name, holds, targets in (("A", a_row, list(TARGETS)),
                                         (A_WITHHELD, a_withheld, list(TARGETS)),
                                         ("B", bool(b), b),
                                         ("B'", bool(bp), bp), ("C", c_row, list(TARGETS)),
                                         ("F", bool(f_t), f_t), ("G", g_here, g_targets)):
                if holds:
                    row["rows"].append(name)
                    row["targets"][name] = targets
            if not row["rows"]:
                row["rows"] = ["mixed"]
            row["reading"] = [reading(name, arm, s) for name in row["rows"]]
            out["rows"].append(row)
    return out


A_WITHHELD = "A-withheld-by-G"
READINGS = {
    "A": "On {owner} own {arm} agents, {score} tracks exact regret throughout training",
    A_WITHHELD: "Faithful in every reached phase; A withheld by row G",
    "B": "The score tracks regret only while the agent is incompetent",
    "B'": "The score tracks regret only once the agent solves most mazes",
    "C": "Ground truth confirms No Regrets on the maze",
    "F": "Fidelity depends on observability",
    "G": "Fresh audits misstate what the curriculum actually sees",
    "mixed": "None of the rows holds: reported per phase as mixed",
}


def reading(row: str, arm: str, score: str) -> str:
    owner = "SFL's" if arm == "sfl" else "JaxUED's"
    return READINGS[row].format(owner=owner, arm=ARM_LABEL[arm], score=LABEL[score])


# --------------------------------------------------------------------------- the timing rule


def block_rates(start: float, block_ends: Sequence[float], updates_per_block: int) -> dict:
    """One timing run's per-update time: the median block after the first, over its updates
    (steady state); the start-up is the first block's end minus one steady block."""
    ends = np.asarray(block_ends, float)
    if ends.size < 3:
        raise ValueError("need at least 3 blocks (one start-up block, two steady ones)")
    steady = np.diff(ends)
    per_update = float(np.median(steady)) / updates_per_block
    return {"per_update_s": per_update,
            "startup_s": float(ends[0] - start - per_update * updates_per_block),
            "blocks": int(ends.size)}


def lane_makespan(durations: Sequence[float], gpus: int, lanes_per_gpu: int) -> float:
    """gpu_round.sh's schedule: job j on lane j mod L, each lane in order; the longest lane."""
    lanes = gpus * lanes_per_gpu
    tot = np.zeros(lanes)
    for j, d in enumerate(durations):
        tot[j % lanes] += d
    return float(tot.max()) if len(durations) else 0.0


def project_queue(cells: Sequence[str], rates: dict, updates: int, gpus: int,
                  lanes_per_gpu: int, audit_s: dict | None = None,
                  checkpoints: int = 16, overhead_h: float = 0.0) -> dict:
    """Section 9's projection rule for one box layout and lane count. ``cells`` is the shard's
    run list in order (each run's cell, e.g. "sfl-v13"); ``rates`` maps a cell to its measured
    {"per_update_s", "startup_s"} at this load. A run takes startup + updates x per-update.
    The training wall time is the longest lane (gpu_round.sh's schedule); GPU-hours are that
    wall time times the box's GPUs (idle GPUs included). The audit adds, per run, its
    checkpoints x the cell's measured audit time (``audit_s``), counted as if the audits ran
    one at a time per GPU (an upper bound). ``overhead_h`` is the box's setup and pull-back."""
    missing = sorted({c for c in cells if c not in rates})
    if missing:
        return {"error": f"no measured rate for {missing}"}
    dur = [rates[c]["startup_s"] + updates * rates[c]["per_update_s"] for c in cells]
    wall = lane_makespan(dur, gpus, lanes_per_gpu) / 3600
    audit_h = None
    if audit_s is not None:
        miss = sorted({c for c in cells if c not in audit_s})
        audit_h = (float(sum(checkpoints * audit_s[c] for c in cells)) / 3600
                   if not miss else None)
    train_gpu_h = wall * gpus
    total = train_gpu_h + (audit_h or 0.0) + overhead_h * gpus
    return {"runs": len(cells), "gpus": gpus, "lanes_per_gpu": lanes_per_gpu,
            "train_wall_h": wall, "train_gpu_h": train_gpu_h, "audit_gpu_h": audit_h,
            "overhead_gpu_h": overhead_h * gpus, "total_gpu_h": total,
            "stop_rule_gpu_h": 1.2 * total, "longest_run_h": max(dur) / 3600 if dur else 0.0}


def describe(runs: Sequence[Run], pt: dict) -> dict:
    """Section 5.4's descriptive statistics that need no bootstrap: per-checkpoint ρ and nRC
    per run, the λ variants, ρ against the solve rate and on solved mazes, lagged fidelity,
    P@10% and admission AUROC of the top-10% regret mazes, and the two-loading regression
    rank(S) ~ a rank(J*) + b rank(J^pi). Point estimates only."""
    out = {}
    for i, r in enumerate(runs):
        rows = []
        for k, ck in enumerate(r.checkpoints):
            reg = ck.regret
            top = reg >= np.quantile(reg, 1 - TOP)
            row = {"update": ck.update, "phase": pt["phases"][i][k],
                   "solve_rate": float(np.mean(ck.success_tgt[ck.reachable]))}
            for s in SCORES:
                sc = rankable(s, ck.scores[s])
                row[s] = {"rho_abs": float(pt["rho"][i, k, SCORES.index(s), 0]),
                          "rho_norm": float(pt["rho"][i, k, SCORES.index(s), 1]),
                          "nrc_abs": float(pt["nrc"][i, k, SCORES.index(s), 0]),
                          "rho_solve": weighted_spearman(sc, ck.success_tgt),
                          "p_at_10": _precision_at(sc, top),
                          "auroc_top10": _auroc(sc, top)}
                solved = ck.success_tgt > 0
                row[s]["rho_solved"] = (weighted_spearman(sc[solved], reg[solved])
                                        if solved.sum() > 2 else float("nan"))
                if k > 0:
                    prev = rankable(s, r.checkpoints[k - 1].scores[s])
                    row[s]["rho_lagged"] = weighted_spearman(prev, reg)
            for s in [x for x in ck.scores if x.startswith(("pvl_lam", "l1_lam"))]:
                row[s] = {"rho_abs": weighted_spearman(rankable(s, ck.scores[s]), reg)}
            ex = ck.extra or {}
            if "j_pi" in ex and "j_star" in ex:
                row["two_loading"] = {s: _two_loading(rankable(s, ck.scores[s]), ex["j_star"],
                                                      ex["j_pi"]) for s in HEADLINE}
            if "j_pi_odd" in ex and "j_pi_even" in ex:
                rel = weighted_spearman(ex["j_star"] - ex["j_pi_odd"],
                                        ex["j_star"] - ex["j_pi_even"])
                rel32 = 2 * rel / (1 + rel) if rel > -1 else float("nan")
                row["reliability"] = {"split_half": rel, "spearman_brown_32": rel32}
                if rel32 > 0:
                    for s in SCORES:
                        row[s]["rho_abs_corrected"] = row[s]["rho_abs"] / math.sqrt(rel32)
            rows.append(row)
        out[r.name] = rows
    return out


def _precision_at(score: np.ndarray, top: np.ndarray, frac: float = TOP) -> float:
    from apexzero.maze.fidelity import capture_weights, top_k

    w = capture_weights(score, top_k(len(score), frac))
    return float((w * top).sum() / w.sum())


def _auroc(score: np.ndarray, positive: np.ndarray) -> float:
    pos, neg = score[positive], score[~positive]
    if not pos.size or not neg.size:
        return float("nan")
    r = stats.rankdata(np.r_[pos, neg])
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _two_loading(score, j_star, j_pi) -> dict:
    X = np.c_[stats.rankdata(j_star), stats.rankdata(j_pi), np.ones(len(score))]
    coef, *_ = np.linalg.lstsq(X, stats.rankdata(score), rcond=None)
    return {"a": float(coef[0]), "b": float(coef[1])}


def finite_or_none(x):
    """JSON-safe numbers: NaN and infinities as None."""
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple)
                 else str(k)): finite_or_none(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [finite_or_none(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite_or_none(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
