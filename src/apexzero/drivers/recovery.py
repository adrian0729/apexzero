"""Injection-recovery: what effect size can the pipeline resolve?

For one circuit model, synthetic sessions of ``pairs`` teammate pairs are generated with
known per-phase effects for every "B" driver, run through the real pipeline, and
compared with the truth at three levels:

- **per lap**: estimated minus true phase time (the measurement error of one phase on
  one lap);
- **per pair**: estimated minus true teammate delta over the same laps (measurement
  error of the published quantity; natural lap-to-lap variation is in both and cancels);
- **recovered vs injected**: estimated delta minus the injected effect (adds the
  natural lap-to-lap variation of the few laps each driver has).

The smallest claimable effect is 2 sigma of recovered minus injected, at the lap counts
the real data have (plan: "claim nothing below 2 sigma of the resulting error"). That is
only meaningful if the synthetic feed is as noisy as the real one, so ``calibrate`` first
matches two statistics that can be measured on both: the held-out S2 timing error (sets
the line wobble, the dominant measurement-noise knob) and the within-driver spread of
phase times (sets the natural lap-to-lap variation once measurement noise is removed).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from apexzero.drivers.deltas import LapTable, aggregate, teammate_deltas
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
    held_out_s2: np.ndarray | None = None   # synthetic held-out S2 errors (s)

    @property
    def pair_err(self) -> np.ndarray:
        return self.pair_est - self.pair_true

    def lap_total_err(self) -> np.ndarray:
        """Per-lap error of the whole lap: zero by construction (anchored lap time)."""
        return self.lap_err.sum(axis=1)


def run(model: CircuitModel, effects: dict = DEFAULT_EFFECTS, pairs: int = 10,
        laps_per_driver: int = 3, reps: int = 2, top_n: int | None = 2,
        cfg: SmootherConfig = DEFAULT_CONFIG, synth_cfg: SynthConfig = DEFAULT_SYNTH,
        seed: int = 0, held_out: bool = False, natural: np.ndarray | None = None) -> Recovery:
    inj = effect_vector(model, effects)
    lap_err, est, true, held = [], [], [], []
    for r in range(reps):
        ss = synth_session(model, pairs, laps_per_driver, inj, synth_cfg, seed=seed + r,
                           natural=natural)
        res = fit_synthetic(ss, model, cfg, held_out=held_out)
        lap_err.append(res.est - res.true)
        if held_out:
            held.append(res.held_out_s2)
        d_est = {d.team: d.delta for d in teammate_deltas(res.table, top_n)}
        d_true = {d.team: d.delta for d in teammate_deltas(res.table_true, top_n)}
        for team in sorted(d_est):
            est.append(d_est[team])
            true.append(d_true[team])
    return Recovery(model.name, model.seg.labels(), inj, np.concatenate(lap_err),
                    np.array(est), np.array(true), laps_per_driver, top_n,
                    np.concatenate(held) if held else None)


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
    recovered_sd_ms: float      # sd of estimated - injected -> 2 sigma claim limit

    @property
    def claim_limit_ms(self) -> float:
        return 2.0 * self.recovered_sd_ms

    @property
    def measurement_limit_ms(self) -> float:
        """2 sigma of the pipeline's own error (what more laps cannot average away)."""
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


def robust_sd(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def within_driver_sd(tab: LapTable) -> np.ndarray:
    """[P] pooled within-(driver, Q segment) sd of phase times, from laps of one driver
    in one Q segment (so track evolution between segments is not counted)."""
    num = np.zeros(tab.phase_t.shape[1])
    dof = 0
    for d in np.unique(tab.driver):
        for q in np.unique(tab.q_segment):
            idx = np.flatnonzero((tab.driver == d) & (tab.q_segment == q))
            if len(idx) >= 2:
                x = tab.phase_t[idx]
                num += ((x - x.mean(axis=0)) ** 2).sum(axis=0)
                dof += len(idx) - 1
    return np.sqrt(num / dof) if dof else np.full(len(num), np.nan)


@dataclass
class Calibration:
    wobble_sd: float
    natural: np.ndarray          # [P] natural lap-to-lap sd per phase (s)
    real_held_sd: float          # robust sd of real held-out S2 errors (s)
    synth_held_sd: float         # same on synthetic laps at the chosen wobble
    by_cell: dict                # (type, phase) -> (real spread, synthetic measurement) (s)
    tried: dict                  # wobble -> synthetic held-out robust sd

    def config(self, base: SynthConfig) -> SynthConfig:
        return replace(base, wobble_sd=self.wobble_sd)

    @property
    def over_noisy(self) -> list:
        """Cells where synthetic measurement error alone exceeds the real total spread:
        there the synthetic feed is noisier than the real one (a conservative budget)."""
        return [k for k, (real, meas) in self.by_cell.items() if meas > real]


def calibrate(model: CircuitModel, real_held_out: np.ndarray, real_table: LapTable,
              cfg: SmootherConfig = DEFAULT_CONFIG, base: SynthConfig = DEFAULT_SYNTH,
              wobbles: tuple[float, ...] = (0.005, 0.009, 0.015, 0.022, 0.03),
              pairs: int = 4, laps_per_driver: int = 3, seed: int = 7) -> Calibration:
    """Match the synthetic feed to one real session (see the module docstring).

    The wobble whose synthetic held-out S2 robust sd is closest (in ratio) to the real
    one is kept. Natural variation is then set per corner type x phase to whatever
    within-driver spread the synthetic measurement error does not explain,
    sqrt(real^2 - measurement^2), floored at zero (medians over that type's cells).
    """
    target = robust_sd(real_held_out)
    tried, best = {}, None
    for w in wobbles:
        rec = run(model, {}, pairs=pairs, laps_per_driver=laps_per_driver, reps=1, cfg=cfg,
                  synth_cfg=replace(base, wobble_sd=w), seed=seed, held_out=True,
                  natural=np.zeros(len(model.seg.labels())))
        h = robust_sd(rec.held_out_s2)
        tried[w] = h
        score = abs(np.log(h / target))
        if best is None or score < best[0]:
            best = (score, w, h, rec)
    _, w, h, rec = best
    labels = model.seg.labels()
    real = within_driver_sd(real_table)
    meas = rec.lap_err.std(axis=0)
    natural = np.zeros(len(labels))
    by_cell = {}
    for kind in ("slow", "medium", "fast"):
        for ph in PHASES:
            cols = [i for i, (_, p, k) in enumerate(labels) if k == kind and p == ph]
            if not cols:
                continue
            r, m = float(np.nanmedian(real[cols])), float(np.median(meas[cols]))
            by_cell[(kind, ph)] = (r, m)
            natural[cols] = np.sqrt(max(r**2 - m**2, 0.0)) if np.isfinite(r) else 0.0
    return Calibration(w, natural, target, h, by_cell, tried)
