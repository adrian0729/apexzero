"""Step 2's decisions (design Sections 7 and 9) on per-key measures: pure NumPy, no network, no
solver, so every rule here is fixed by the code hash at the tag and tested on synthetic rows.

**Fixed lists** (design Section 9, as constants):
- ``H2_NETWORKS``: the 9 step-1 generalists (``registry.generalists``; H2', finals, A2 keys).
- ``H4_VARIANTS``: per variant (v1, v3), the real-width runs (``dr-realw-<v>-s0..3``) and the
  baseline runs (seed 0 and R18's seeds 1-4 of the variant's pilot), and the seed-0 agent
  whose frozen regime the variant's A2 cells are typed by (G1 for v1, G3 for v3).
- ``H4_TRANSFER``: the 8 leave-out runs (``dr-realw-loo-f{1,2}-s{0..3}``, v3 configuration)
  against the v3 baselines.
- ``H3_CELLS`` (584, 605, 612: G2's three crash units, const_width rungs on R-dev circuits),
  their real-width twins (``h3_pairs``), the lambda grid (step 0.05) and the dose amplitudes
  (``kkt.DOSE_AMPLITUDES``).

**Regimes** (``a2_cells_by_regime``): an A2 cell's regime for an agent is step 1's
``analysis.regime_of`` (imported read-only) of its stimulus's R_seg against the agent's frozen
``support.RegimeTable`` (q1, q10, R_clip^max; outputs/psychophysics/support/support.json). An
H4 network uses its variant's seed-0 regime. H1-typical uses each agent's 'bulk' cells (as P1a).

**H4** (``h4_test``): the statistic is each network's median G; the test is one-sided exact
permutation, real widths > baseline, with labels exchanged within each variant. T = sum over
variants of (mean real - mean baseline); every relabelling of every variant is enumerated
(C(9,4) = 126 per variant, 126^2 = 15,876 jointly). p = share of relabellings with T* >= T
(the observed one included), the opposite p = share with T* <= T. confirm if p <= 0.05,
refute if the opposite p <= 0.05, else inconclusive. The sensitivity analysis drops the seed-0
baselines (C(8,4) = 70 per variant, 4,900 jointly); it is reported, not decided. Per-variant
tests are reported beside the joint one.

**R18** (``variant_rule``): a variant-level claim needs the same decision in at least 2 of the
variant's 3 seeds; otherwise it is 'seed-specific'.

**H1-fast** (``h1_fast``): the predicted signs (s_b,net - s_b* < 0 and v_apex,net - v_apex* < 0
for G1 and G3) on the cluster-bootstrap CI of the median difference over geometries: 'confirm'
if the CI lies below 0, 'refute' if above 0, else 'inconclusive'; G2's 'about 0' is reported
as whether its CI contains 0 (the design gives no band, so it is descriptive). The curvature
gain is two-sided equivalence to [0.8, 1.25] (``equivalence``): TOST at 5% is the 90% CI lying
inside the band ('equivalent'); a 90% CI entirely outside it is 'not equivalent'; else
'inconclusive'.

**H2-speed** (``h2_speed``): r_in's CI below 0.5 'under', within [0.5, 2] 'lawful', above 2
'over', else 'inconclusive'.

**H1-typical** (``h1_typical``): per outcome (s_b, then v_apex), the agent passes when the CI of
its median gain lies within [0.8, 1.25], the CI of its median |location| below the preview
spacing (12 m), and the CI of its median delta_rho (``measures.shape_specificity``) above 0;
each criterion is also paired by key against the blind-pedal control (``paired_rows``: the
per-key difference network - control of the criterion's score; gain as -|g - 1|, location as
-|location|, shape as delta_rho), and the paired CI must lie above 0.

**H3** (descriptive; ``lambda_star``, ``lambda_lin``): lambda* is the smallest lambda of the grid
at which G2 crashes on w(lambda) = w_cw + lambda (w_real - w_cw); lambda*_lin = c_min(0) /
(-sum_j S(c_min; j) dw_j) from the per-sample impulse responses, on the left and right widths
separately; the clearance change splits into the edge's own move (c_min's explicit term) and
the line's move (the rest).

**Exclusions** (``key_measure``): the per-key reason in ``measures.EXCLUSION_REASONS``' order
(``measures.exclusion_reason``); a pipeline failure raises.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from apexzero.fields import measures as MS
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp import registry
from apexzero.interp.analysis import regime_of

# ------------------------------------------------------------------------- fixed lists

H2_NETWORKS = registry.generalists()


def _seeds(run: str, seeds: Iterable[int], zero_plain: bool) -> tuple[str, ...]:
    return tuple(run if (s == 0 and zero_plain) else f"{run}-s{s}" for s in seeds)


H4_VARIANTS = {
    "v1": {"real": _seeds("runs/dr-realw-v1", range(4), False),
           "baseline": _seeds("runs/dr-pilot-200M", range(5), True), "regime_of": "G1"},
    "v3": {"real": _seeds("runs/dr-realw-v3", range(4), False),
           "baseline": _seeds("runs/dr-pilot-v3-tilt50a2", range(5), True), "regime_of": "G3"},
}
H4_TRANSFER = {"real": (_seeds("runs/dr-realw-loo-f1", range(4), False)
                        + _seeds("runs/dr-realw-loo-f2", range(4), False)),
               "baseline": H4_VARIANTS["v3"]["baseline"], "regime_of": "G3"}
H4_ALPHA = 0.05
H1_FAST_SIGNS = {"G1": -1, "G3": -1, "G2": 0}     # predicted sign of net - optimum (s_b, v_apex)
H1_EQUIV = (0.8, 1.25)
EQUIV_LEVEL = 0.90                                  # TOST at 5%: the 90% CI
H2_SPEED_BAND = MS.R_IN_BAND
H3_CELLS = (584, 605, 612)
H3_LAMBDA_STEP = 0.05
H3_LAMBDAS = tuple(np.round(np.arange(0.0, 1.0 + 1e-9, H3_LAMBDA_STEP), 10))
H3_AGENT = "G2"


def is_baseline_seed0(run: str) -> bool:
    """A baseline run of seed 0 (its run name has no -s<k> suffix): left out of H4's
    sensitivity analysis."""
    return re.search(r"-s\d+$", run) is None


# ------------------------------------------------------------------------- regimes


def a2_cells_by_regime(agent: str, bank, cells: dict, regimes: dict, a2_cells: Iterable[int]
                       ) -> dict[str, tuple[int, ...]]:
    """{regime: A2 cell ids} for an agent (module docstring): ``regimes`` maps agent ids to
    ``support.RegimeTable`` (or step 1's ``analysis.Regime``); an H4 network passes its
    variant's seed-0 agent."""
    rt = regimes[agent]
    out: dict[str, list[int]] = {}
    for cid in sorted({int(c) for c in a2_cells}):
        R = float(bank.stimuli[cells[cid].stim_id].geometry.R_seg)
        reg = regime_of(R, rt.q1, rt.q10, getattr(rt, "r_clip_max", float("nan")))
        out.setdefault(reg, []).append(cid)
    return {k: tuple(v) for k, v in sorted(out.items())}


# ------------------------------------------------------------------------- H4


def _diffs(real: Sequence[float], base: Sequence[float]) -> tuple[float, np.ndarray]:
    """(observed mean(real) - mean(base), the same over every relabelling of the pooled
    values into groups of len(real) and len(base))."""
    v = np.concatenate([np.asarray(real, float), np.asarray(base, float)])
    n, k = len(v), len(real)
    tot = v.sum()
    out = np.array([v[list(c)].sum() / k - (tot - v[list(c)].sum()) / (n - k)
                    for c in itertools.combinations(range(n), k)])
    return float(np.mean(real) - np.mean(base)), out


def permutation_test(groups: dict[str, tuple[Sequence[float], Sequence[float]]]) -> dict:
    """The exact within-strata permutation test (module docstring): ``groups`` maps a
    stratum (variant) to (real values, baseline values). T = sum over strata of the mean
    differences; every joint relabelling is enumerated. Returns T, the one-sided p (T* >= T),
    the opposite p (T* <= T), the number of relabellings and the per-stratum tests."""
    if any(not len(r) or not len(b) for r, b in groups.values()):
        raise ValueError("every stratum needs real and baseline values")
    if any(not np.all(np.isfinite(np.r_[r, b])) for r, b in groups.values()):
        raise ValueError("a network's statistic is NaN: decide its exclusion first")
    obs, dist, per = 0.0, np.zeros(1), {}
    for name, (r, b) in groups.items():
        t, d = _diffs(r, b)
        obs += t
        dist = (dist[:, None] + d[None, :]).ravel()
        eps = 1e-12 * max(1.0, float(np.max(np.abs(d))))
        per[name] = {"T": t, "n_relabellings": len(d),
                     "p": float(np.mean(d >= t - eps)), "p_opposite": float(np.mean(d <= t + eps))}
    eps = 1e-12 * max(1.0, float(np.max(np.abs(dist))))
    return {"T": obs, "n_relabellings": len(dist), "p": float(np.mean(dist >= obs - eps)),
            "p_opposite": float(np.mean(dist <= obs + eps)), "strata": per}


def decide(p: float, p_opposite: float, alpha: float = H4_ALPHA) -> str:
    """'confirm' if p <= alpha, 'refute' if the opposite p <= alpha, else 'inconclusive'."""
    if p <= alpha:
        return "confirm"
    if p_opposite <= alpha:
        return "refute"
    return "inconclusive"


def h4_test(stats: dict[str, float], variants: dict = H4_VARIANTS) -> dict:
    """H4 on each network's median G (``stats``: run -> statistic): the joint within-variant
    test and its decision, and the sensitivity analysis without the seed-0 baselines."""
    groups = {v: ([stats[r] for r in d["real"]], [stats[r] for r in d["baseline"]])
              for v, d in variants.items()}
    main = permutation_test(groups)
    main["decision"] = decide(main["p"], main["p_opposite"])
    sens_groups = {v: ([stats[r] for r in d["real"]],
                       [stats[r] for r in d["baseline"] if not is_baseline_seed0(r)])
                   for v, d in variants.items()}
    sens = permutation_test(sens_groups)
    sens["decision"] = decide(sens["p"], sens["p_opposite"])
    return {"main": main, "sensitivity": sens}


def h4_transfer(stats: dict[str, float], spec: dict = H4_TRANSFER) -> dict:
    """H4's statistic and test on the leave-out networks against the v3 baselines (one
    stratum; C(13, 8) = 1,287 relabellings)."""
    res = permutation_test({"transfer": ([stats[r] for r in spec["real"]],
                                         [stats[r] for r in spec["baseline"]])})
    res["decision"] = decide(res["p"], res["p_opposite"])
    return res


# ------------------------------------------------------------------------- R18


def variant_rule(verdicts: dict[str, str], variant: str) -> str:
    """R18's variant-level verdict: the decision at least 2 of the variant's 3 seeds share
    (``registry.variant_seeds``), else 'seed-specific'; seeds without a verdict count as
    disagreeing."""
    seeds = registry.variant_seeds(variant)
    vals = [verdicts.get(a) for a in seeds]
    for v in sorted({v for v in vals if v is not None}):
        if sum(x == v for x in vals) >= 2:
            return v
    return "seed-specific"


# ------------------------------------------------------------------------- bands


def interval(rows: Sequence[MS.KeyMeasure], level: float = MS.LEVEL, n: int = MS.BOOTSTRAP_N,
             directions=(1, -1)) -> tuple[float, float, float]:
    """(median, lo, hi) of an agent's measure over geometries (``measures.agent_summary``,
    whose pair rule and cluster bootstrap apply), at ``level``."""
    pv = MS.pair_values(rows, directions)
    gv = MS.geometry_values(pv)
    vals = np.array(list(gv.values()), float)
    geos = np.array(list(gv.keys()), float).reshape(-1, 2)
    return MS.cluster_bootstrap(vals, geos, np.nanmedian, n, level, MS.stream_rng("bootstrap"))


def sign_decision(lo: float, hi: float, predicted: int) -> str:
    """A predicted sign against a CI: for -1 'confirm' if hi < 0, 'refute' if lo > 0; for +1
    the mirror; for 0 'consistent' if the CI contains 0, else 'inconsistent'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if predicted == 0:
        return "consistent" if lo <= 0.0 <= hi else "inconsistent"
    below, above = hi < 0.0, lo > 0.0
    if (predicted < 0 and below) or (predicted > 0 and above):
        return "confirm"
    if (predicted < 0 and above) or (predicted > 0 and below):
        return "refute"
    return "inconclusive"


def equivalence(lo: float, hi: float, band: tuple[float, float] = H1_EQUIV) -> str:
    """Two-sided equivalence on a (90%) CI: 'equivalent' inside the band, 'not equivalent'
    entirely outside it, else 'inconclusive'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if band[0] <= lo and hi <= band[1]:
        return "equivalent"
    if hi < band[0] or lo > band[1]:
        return "not equivalent"
    return "inconclusive"


def h2_speed(lo: float, hi: float, band: tuple[float, float] = H2_SPEED_BAND) -> str:
    """H2-speed's band rule on r_in's CI (module docstring)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if hi < band[0]:
        return "under"
    if band[0] <= lo and hi <= band[1]:
        return "lawful"
    if lo > band[1]:
        return "over"
    return "inconclusive"


def h1_fast(rows_sb: Sequence[MS.KeyMeasure], rows_v: Sequence[MS.KeyMeasure],
            rows_gain: Sequence[MS.KeyMeasure], agent: str, n: int = MS.BOOTSTRAP_N) -> dict:
    """H1-fast for one agent: the signs of s_b,net - s_b* and v_apex,net - v_apex* (rows hold
    the per-key differences) and the curvature gain's equivalence (rows hold g)."""
    pred = H1_FAST_SIGNS.get(registry.seed0_of(agent), 0)
    out = {"agent": agent, "predicted": pred}
    for name, rows in (("s_b", rows_sb), ("v_apex", rows_v)):
        est, lo, hi = interval(rows, n=n)
        out[name] = {"median": est, "lo": lo, "hi": hi, "decision": sign_decision(lo, hi, pred)}
    est, lo, hi = interval(rows_gain, EQUIV_LEVEL, n)
    out["gain"] = {"median": est, "lo": lo, "hi": hi, "decision": equivalence(lo, hi)}
    return out


# ------------------------------------------------------------------------- H1-typical


def _score(criterion: str, value: float) -> float:
    """The per-key score a criterion is paired on (larger is better)."""
    if criterion == "gain":
        return -abs(value - 1.0)
    if criterion == "location":
        return -abs(value)
    return value                                   # shape: delta_rho


def paired_rows(net: Sequence[MS.KeyMeasure], ctrl: Sequence[MS.KeyMeasure], criterion: str
                ) -> list[MS.KeyMeasure]:
    """Per key, score(net) - score(control) of one criterion (``_score``); a key missing or
    excluded on either side is NaN (excluded)."""
    by = {r.key_id: r for r in ctrl}
    out = []
    for r in net:
        c = by.get(r.key_id)
        v = (_score(criterion, r.value) - _score(criterion, c.value)) if c is not None else np.nan
        out.append(MS.KeyMeasure(r.key_id, r.pair_id, r.approach, r.direction, r.phase,
                                 r.n_phases, r.R, r.theta_deg, float(v),
                                 r.reason or (c.reason if c is not None else "no control")))
    return out


def h1_typical(gain: Sequence[MS.KeyMeasure], location: Sequence[MS.KeyMeasure],
               shape: Sequence[MS.KeyMeasure], control: dict[str, Sequence[MS.KeyMeasure]],
               n: int = MS.BOOTSTRAP_N) -> dict:
    """H1-typical for one agent and outcome (module docstring); ``control`` maps each
    criterion to the blind-pedal control's rows of the same measure."""
    est, lo, hi = interval(gain, n=n)
    crit = {"gain": {"median": est, "lo": lo, "hi": hi,
                     "pass": bool(MS.H1_GAIN[0] <= lo and hi <= MS.H1_GAIN[1])}}
    absloc = [MS.KeyMeasure(r.key_id, r.pair_id, r.approach, r.direction, r.phase, r.n_phases,
                            r.R, r.theta_deg, abs(r.value), r.reason) for r in location]
    est, lo, hi = interval(absloc, n=n)
    crit["location"] = {"median": est, "lo": lo, "hi": hi,
                        "pass": bool(np.isfinite(hi) and hi <= MS.H1_LOCATION)}
    est, lo, hi = interval(shape, n=n)
    crit["shape"] = {"median": est, "lo": lo, "hi": hi,
                     "pass": bool(np.isfinite(lo) and lo > MS.SHAPE_DELTA_MIN)}
    for name, rows in (("gain", gain), ("location", location), ("shape", shape)):
        est, lo, hi = interval(paired_rows(rows, control[name], name), n=n)
        crit[name]["vs_control"] = {"median": est, "lo": lo, "hi": hi,
                                    "pass": bool(np.isfinite(lo) and lo > 0.0)}
    ok = all(c["pass"] and c["vs_control"]["pass"] for c in crit.values())
    return {"criteria": crit, "pass": bool(ok)}


# ------------------------------------------------------------------------- exclusions


def key_measure(meta: dict, value: float, flags: dict) -> MS.KeyMeasure:
    """One key's measure with its exclusion reason (``measures.exclusion_reason``): ``meta``
    holds key_id, pair_id, approach, direction, phase, n_phases, R, theta_deg; ``flags``
    holds invalid (bool), why_invalid, gate3_failed, numerical, inertia_failed, kappa_clipped
    and off_plateau. An excluded key's value is NaN."""
    reason = MS.exclusion_reason(bool(flags.get("invalid", False)),
                                 bool(flags.get("gate3_failed", False)),
                                 bool(flags.get("inertia_failed", False)),
                                 bool(flags.get("kappa_clipped", False)),
                                 bool(flags.get("off_plateau", False)),
                                 str(flags.get("why_invalid", "")),
                                 bool(flags.get("numerical", False)))
    v = float("nan") if reason else float(value)
    return MS.KeyMeasure(str(meta["key_id"]), int(meta["pair_id"]), str(meta["approach"]),
                         int(meta["direction"]), int(meta["phase"]), int(meta["n_phases"]),
                         float(meta["R"]), float(meta["theta_deg"]), v, reason)


def coverage_records(rows: Iterable[MS.KeyMeasure], agent: str, snapshot: str, outcome: str
                     ) -> list[dict]:
    """``measures.coverage``'s records from one agent's key measures of one outcome."""
    return [{"agent": agent, "snapshot": snapshot, "outcome": outcome, "reason": r.reason}
            for r in rows]


# ------------------------------------------------------------------------- H3


def h3_pairs(bank, cells: dict, h3_cells: Sequence[int] = H3_CELLS) -> dict[int, int]:
    """{const_width cell: its real-width stimulus id}: the stimulus named '<key>/real' of the
    const_width cell's key (same curvature samples, asserted)."""
    names = {s.name: i for i, s in enumerate(bank.stimuli)}
    out = {}
    for cid in h3_cells:
        name = bank.stimuli[cells[cid].stim_id].name
        if not name.endswith("/const_width"):
            raise ValueError(f"cell {cid} ({name}) is not a const_width rung")
        sid = names[name[: -len("const_width")] + "real"]
        if not np.array_equal(np.asarray(bank.kappa[sid]), np.asarray(bank.kappa[cells[cid].stim_id])):
            raise ValueError(f"cell {cid}: the real-width twin's curvature differs")
        out[int(cid)] = int(sid)
    return out


def width_change(bank, cw_stim: int, real_stim: int) -> Samples:
    """dw = w_real - w_cw per side (kappa 0): the H3 direction w(lambda) = w_cw + lambda dw.
    float32-cast samples, upcast (the agents' arrays were built from them)."""
    up = PT.samples32(bank, real_stim)
    cw = PT.samples32(bank, cw_stim)
    return Samples(np.zeros_like(up.kappa), up.w_left - cw.w_left, up.w_right - cw.w_right)


def lambda_star(lams: Sequence[float], crashed: Sequence[bool]) -> float:
    """The smallest lambda of the grid at which the run crashes (inf if none)."""
    hits = [float(lam) for lam, c in zip(lams, crashed, strict=True) if c]
    return min(hits) if hits else float("inf")


def lambda_lin(c_min0: float, impulse_c_min: Samples, explicit_c_min: Samples | None,
               dw: Samples) -> dict:
    """lambda*_lin = c_min(0) / (-sum_j S(c_min; j) dw_j), from the impulse responses, in all
    and per side (left and right widths separately), with the first-order clearance change
    per unit lambda split into the edge's own move (``explicit_c_min`` along dw: c_min's read
    of the width, the states held) and the line's move (the rest)."""
    def along(imp, f):
        return float(np.dot(np.asarray(getattr(imp, f), float), np.asarray(getattr(dw, f), float)))

    dc = {f: along(impulse_c_min, f) for f in ("w_left", "w_right")}
    tot = dc["w_left"] + dc["w_right"]

    def lin(d):
        return float(c_min0 / -d) if d < 0 else float("inf")

    out = {"dc_dlambda": tot, "dc_left": dc["w_left"], "dc_right": dc["w_right"],
           "lambda_lin": lin(tot), "lambda_lin_left": lin(dc["w_left"]),
           "lambda_lin_right": lin(dc["w_right"])}
    if explicit_c_min is not None:
        edge = sum(along(explicit_c_min, f) for f in ("w_left", "w_right"))
        out.update(edge_move=edge, line_move=tot - edge)
    return out


def summarise(rows: Sequence[MS.KeyMeasure], decide_fn: Callable[[float, float], str] | None = None,
              n: int = MS.BOOTSTRAP_N) -> dict:
    """One agent's measure per direction and pooled (``measures.by_direction``) with, given
    ``decide_fn(lo, hi)``, the pooled decision."""
    out = {k: v.as_dict() for k, v in MS.by_direction(rows, n).items()}
    if decide_fn is not None:
        out["decision"] = decide_fn(out["pooled"]["lo"], out["pooled"]["hi"])
    return out
