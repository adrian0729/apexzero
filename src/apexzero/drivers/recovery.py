"""Injection-recovery: what effect size can the pipeline resolve?

For one circuit model, synthetic sessions of ``pairs`` teammate pairs are generated with
known per-phase effects for every "B" driver, run through the real pipeline, and
compared with the truth at three levels:

- **per lap**: estimated minus true phase time (the measurement error of one phase on
  one lap);
- **per pair**: estimated minus true teammate delta over the same laps (measurement
  error of the published quantity; natural lap-to-lap variation is in both and cancels);
- **recovered vs injected**: estimated delta minus the injected effect (adds the
  sampling noise of a few laps per driver).

The smallest claimable effect is 2 sigma of the per-pair measurement error at the lap
counts the real data have (plan: "claim nothing below 2 sigma of the resulting error").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from apexzero.drivers.deltas import aggregate, teammate_deltas
from apexzero.drivers.pipeline import fit_synthetic
from apexzero.drivers.segments import PHASES
from apexzero.drivers.smoother import DEFAULT_CONFIG, SmootherConfig
from apexzero.drivers.synth import DEFAULT_SYNTH, CircuitModel, SynthConfig, synth_session

DEFAULT_EFFECTS = {("slow", "entry"): 0.040, ("medium", "apex"): 0.020, ("fast", "exit"): -0.020}


def effect_vector(model: CircuitModel, effects: dict) -> np.ndarray:
    """Per-phase injected effect (s) from a {(corner type, phase): seconds} spec."""
    return np.array([effects.get((kind, ph), 0.0) for _, ph, kind in model.seg.labels()])


@dataclass
class Recovery:
    circuit: str
    labels: list[tuple[str, str, str]]
    injected: np.ndarray        # [P]
    lap_err: np.ndarray         # [laps, P] estimated - true per lap and phase
    pair_est: np.ndarray        # [pairs, P] estimated teammate delta
    pair_true: np.ndarray       # [pairs, P] true delta over the same laps
    laps_per_driver: int
    top_n: int | None

    @property
    def pair_err(self) -> np.ndarray:
        return self.pair_est - self.pair_true

    def lap_total_err(self) -> np.ndarray:
        """Per-lap error of the whole lap: zero by construction (anchored lap time)."""
        return self.lap_err.sum(axis=1)


def run(model: CircuitModel, effects: dict = DEFAULT_EFFECTS, pairs: int = 10,
        laps_per_driver: int = 3, reps: int = 2, top_n: int | None = 2,
        cfg: SmootherConfig = DEFAULT_CONFIG, synth_cfg: SynthConfig = DEFAULT_SYNTH,
        seed: int = 0) -> Recovery:
    inj = effect_vector(model, effects)
    lap_err, est, true = [], [], []
    for r in range(reps):
        ss = synth_session(model, pairs, laps_per_driver, inj, synth_cfg, seed=seed + r)
        res = fit_synthetic(ss, model, cfg)
        lap_err.append(res.est - res.true)
        d_est = {d.team: d.delta for d in teammate_deltas(res.table, top_n)}
        d_true = {d.team: d.delta for d in teammate_deltas(res.table_true, top_n)}
        for team in sorted(d_est):
            est.append(d_est[team])
            true.append(d_true[team])
    return Recovery(model.name, model.seg.labels(), inj, np.concatenate(lap_err),
                    np.array(est), np.array(true), laps_per_driver, top_n)


@dataclass
class BudgetRow:
    kind: str
    phase: str
    n_cells: int
    lap_sd_ms: float            # per-lap, per-phase measurement error sd
    pair_bias_ms: float         # mean per-pair measurement error
    pair_sd_ms: float           # sd of per-pair measurement error -> 2 sigma claim limit
    injected_ms: float          # mean injected effect in these cells
    recovered_ms: float         # mean estimated delta in these cells
    recovered_sd_ms: float      # sd of estimated - injected

    @property
    def claim_limit_ms(self) -> float:
        return 2.0 * self.pair_sd_ms


def budget(recs: list[Recovery]) -> list[BudgetRow]:
    """Error budget per corner type x phase, pooled over circuits and cells."""
    rows = []
    for kind in ("slow", "medium", "fast"):
        for ph in PHASES:
            le, pe, pest, inj = [], [], [], []
            for rec in recs:
                cols = [i for i, (_, p, k) in enumerate(rec.labels) if k == kind and p == ph]
                if not cols:
                    continue
                le.append(rec.lap_err[:, cols].ravel())
                pe.append(rec.pair_err[:, cols].ravel())
                pest.append(rec.pair_est[:, cols].ravel())
                inj.append(np.broadcast_to(rec.injected[cols], rec.pair_est[:, cols].shape).ravel())
            if not le:
                continue
            le, pe = np.concatenate(le), np.concatenate(pe)
            pest, inj = np.concatenate(pest), np.concatenate(inj)
            rows.append(BudgetRow(kind, ph, len(pe), 1e3 * le.std(), 1e3 * pe.mean(),
                                  1e3 * pe.std(ddof=1), 1e3 * inj.mean(), 1e3 * pest.mean(),
                                  1e3 * (pest - inj).std(ddof=1)))
    return rows


def aggregate_budget(recs: list[Recovery]) -> dict:
    """Same budget for corner type x phase *sums* over a lap (the driver-profile cells)."""
    out: dict = {}
    for rec in recs:
        for est, true, in zip(rec.pair_est, rec.pair_true, strict=True):
            a_est = aggregate(est, rec.labels, "type")
            a_true = aggregate(true, rec.labels, "type")
            a_inj = aggregate(rec.injected, rec.labels, "type")
            for k in a_est:
                out.setdefault(k, []).append((a_est[k] - a_true[k], a_est[k] - a_inj[k], a_inj[k]))
    return {k: {"n": len(v), "pair_sd_ms": 1e3 * float(np.std([x[0] for x in v], ddof=1)),
                "pair_bias_ms": 1e3 * float(np.mean([x[0] for x in v])),
                "recovered_minus_injected_ms": 1e3 * float(np.mean([x[1] for x in v])),
                "recovered_sd_ms": 1e3 * float(np.std([x[1] for x in v], ddof=1)),
                "injected_ms": 1e3 * float(np.mean([x[2] for x in v]))}
            for k, v in out.items()}
