"""Step 2's decisions (design Sections 7, 9 and 15) on per-key measures: pure NumPy, no
network, no solver, so every rule here is fixed by the code hash at the tag and tested on
synthetic rows.

**Fixed lists** (design Sections 9 and 15, as constants):
- ``H2_NETWORKS``: the 9 step-1 generalists (``registry.generalists``; H2', finals, A2 keys).
- ``H4_VARIANTS``: per variant (v1, v3), the real-width runs (``dr-realw-<v>-s0..3``) and the
  baseline runs (seed 0 and R18's seeds 1-4 of the variant's pilot), and the seed-0 agent
  whose frozen regime table types the variant's A2 cells for H4's descriptive per-regime
  breakdown (G1 for v1, G3 for v3; ``regime_of``).
- ``H4_TRANSFER``: the 8 leave-out runs (``dr-realw-loo-f{1,2}-s{0..3}``, v3 configuration)
  against the v3 baselines (typed by G3).
- ``A2_NETWORKS`` (``a2_networks``): every network of H2', H4 and H4-transfer, the networks of
  fields-a2 and gate2-trained; ``DOSE_NETWORKS``: the 9 generalists and the 18 H4 networks.
- ``H3_CELLS`` (584, 605, 612: G2's three crash units, const_width rungs on R-dev circuits),
  their real-width twins (``h3_pairs``), the lambda grid (step 0.05) and the dose amplitudes
  (``kkt.DOSE_AMPLITUDES``).

**Regimes** (``a2_cells_by_regime``): an A2 cell's regime for an agent is step 1's
``analysis.regime_of`` (imported read-only) of its stimulus's R_seg against the agent's frozen
``support.RegimeTable`` (q1, q10, R_clip^max; outputs/psychophysics/support/support.json). The
cell sets are written into the pre-registration record and read from it after the tag.
H1-typical uses each agent's 'bulk' cells (as P1a). H2' and H4 use every regime (design
Section 7): H4's data are the same primary, non-capped A2 keys as H2', and the variant's
seed-0 typing gives only a descriptive per-regime breakdown (``h4_regime_agent``).

**H4** (``h4_test``): the statistic is each network's median G; the test is one-sided exact
permutation, real widths > baseline, with labels exchanged within each variant. T = sum over
variants of (mean real - mean baseline); every joint relabelling is enumerated. p = share of
relabellings with T* >= T (the observed one included), the opposite p = share with T* <= T;
confirm if p <= 0.05, refute if the opposite p <= 0.05, else inconclusive. The decision test
leaves out the seed-0 baselines (4 real vs baseline seeds 1-4 per variant: C(8,4) = 70 per
variant, 4,900 jointly, minimum p 2.0e-4). The sensitivity analysis includes them (4 vs 5:
C(9,4) = 126 per variant, 15,876 jointly); when its decision differs, H4's decision carries
the label 'not robust to including the seed-0 baselines'. Per-variant tests are reported
beside the joint ones. **H4-transfer** (``h4_transfer``) is the same on one stratum: the 8
leave-out runs against v3 baseline seeds 1-4 (C(12,8) = 495, minimum p 0.002) decide, with
seed 0 (C(13,8) = 1,287) the labelled sensitivity. **Coverage** (design Section 15): H4 and
H4-transfer are 'not testable (...)' when any of their networks is descriptive (its share of
excluded keys above 30%, or failing gate2-trained) or has an undefined statistic.

**H2'** (``h2_network``, ``h2_variant``): the band decision on the median G's CI
(``measures.h2_decision``); a network whose excluded share exceeds 30% (every reason) or
that failed gate2-trained is 'descriptive (coverage)' or 'descriptive (gate2-trained)' (its
band decision kept beside it). R18 (``variant_rule``): a variant-level claim needs the same
decision in at least 2 of the variant's 3 seeds; descriptive seeds have no verdict; with two
or more of them the variant is 'descriptive (coverage)'; otherwise 'seed-specific'.

**Asymmetry** (``asymmetry_summary``, descriptive, design Sections 4 and 7): every measure
``summarise`` reports carries its left-right index delta m = m_L - m_R over mirror partners
(``measures.asymmetry_pairs``: per pair the median over its partnered phases), with the
median over pairs and its cluster-bootstrap CI over geometries; H1-typical's location index
is taken on the signed location.

**T** (``t_label``): the corridor-translation gain T = G + G_out (``measures.EdgeMeasures.T``)
is descriptive: 'width-blind' if its CI's upper end is <= 0.5, 'corridor-anchored' if the CI
lies within [0.8, 1.2], else 'mixed'.

**H1-fast** (``h1_fast``, descriptive since v3.2): s_b,net - s_b* and v_apex,net - v_apex* with
their cluster-bootstrap CIs (left, right, pooled, and per approach), each labelled 'below 0',
'above 0' or 'contains 0' and against step 1's direction (G1's and G3's variants < 0, G2's
'about 0'; specialists have none); the s_b field's curvature gain with its 95% CI. No band.

**H2-speed** (``h2_speed``): r_in's CI below 0.5 'under', within [0.5, 2] 'lawful', above 2
'over', else 'inconclusive' (descriptive bands; 'descriptive (coverage)' above 30%).

**H1-typical** (``h1_typical``, descriptive since v3.1): per outcome (s_b, then v_apex), the
median gain, |location| and delta_rho (``measures.shape_specificity``) with their CIs, and
each paired by key against the blind-pedal control (``paired_rows``: the per-key difference
network - control of the criterion's score; gain as -|g - 1|, location as -|location|,
shape as delta_rho). No pass/fail and no band.

**H3** (descriptive; ``lambda_star``, ``lambda_lin``): lambda* is the smallest lambda of the grid
at which G2 crashes on w(lambda) = w_cw + lambda (w_real - w_cw); lambda*_lin = c_min(0) /
(-sum_j S(c_min; j) dw_j) from the per-sample impulse responses, on the left and right widths
separately; the clearance change splits into the edge's own move (c_min's explicit term) and
the line's move (the rest).

**Exclusions** (``key_measure``): the per-key reason in ``measures.EXCLUSION_REASONS``' order
(``measures.exclusion_reason``); since v3.2 a pipeline failure (loop exhausted at the key, no
optimum row, branch consistency) is an exclusion of that key, listed per network.
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
H4_NOT_ROBUST = "not robust to including the seed-0 baselines"
H1_FAST_SIGNS = {"G1": -1, "G3": -1, "G2": 0}     # step 1's direction of net - optimum (s_b, v_apex)
H2_SPEED_BAND = MS.R_IN_BAND
H3_CELLS = (584, 605, 612)
H3_LAMBDA_STEP = 0.05
H3_LAMBDAS = tuple(np.round(np.arange(0.0, 1.0 + 1e-9, H3_LAMBDA_STEP), 10))
H3_AGENT = "G2"


def h4_networks(variants: dict = H4_VARIANTS) -> tuple[str, ...]:
    """H4's 18 networks: each variant's real-width runs, then its baselines."""
    return tuple(r for d in variants.values() for r in d["real"] + d["baseline"])


def transfer_networks(spec: dict = H4_TRANSFER) -> tuple[str, ...]:
    """H4-transfer's 13 networks: the 8 leave-out runs and the 5 v3 baselines."""
    return tuple(spec["real"]) + tuple(spec["baseline"])


def a2_networks() -> tuple[str, ...]:
    """Every network of H2', H4 and H4-transfer (fields-a2 and gate2-trained), in order,
    without repeats."""
    return tuple(dict.fromkeys(H2_NETWORKS + h4_networks() + tuple(H4_TRANSFER["real"])))


A2_NETWORKS = a2_networks()
DOSE_NETWORKS = tuple(dict.fromkeys(H2_NETWORKS + h4_networks()))   # design Section 15: 27


def is_baseline_seed0(run: str) -> bool:
    """A baseline run of seed 0 (its run name has no -s<k> suffix): left out of H4's and
    H4-transfer's decision tests, kept in their sensitivity analyses."""
    return re.search(r"-s\d+$", run) is None


def h4_regime_agent(run: str) -> str | None:
    """The seed-0 agent whose regime table types an H4 or H4-transfer network's A2 cells for
    the descriptive per-regime breakdown (G1 for v1's networks, G3 for v3's and the
    leave-out runs); None for any other network."""
    for d in H4_VARIANTS.values():
        if run in d["real"] or run in d["baseline"]:
            return d["regime_of"]
    if run in H4_TRANSFER["real"]:
        return H4_TRANSFER["regime_of"]
    return None


# ------------------------------------------------------------------------- regimes


def a2_cells_by_regime(agent: str, bank, cells: dict, regimes: dict, a2_cells: Iterable[int]
                       ) -> dict[str, tuple[int, ...]]:
    """{regime: A2 cell ids} for an agent (module docstring): ``regimes`` maps agent ids to
    ``support.RegimeTable`` (or step 1's ``analysis.Regime``). An H4 or H4-transfer network's
    descriptive breakdown uses its variant's seed-0 agent (``h4_regime_agent``)."""
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


def _decided(groups: dict) -> dict:
    res = permutation_test(groups)
    res["decision"] = decide(res["p"], res["p_opposite"])
    return res


def _untestable(networks: Sequence[str], stats: dict[str, float],
                descriptive: dict[str, str] | None) -> dict[str, str]:
    """{network: why} for each network that makes a test not testable: descriptive (its
    reason: 'coverage' or 'gate2-trained'), or its statistic missing or not finite."""
    out = {}
    for r in networks:
        why = (descriptive or {}).get(r)
        if why:
            out[r] = str(why)
        elif r not in stats or not np.isfinite(float(stats[r])):
            out[r] = "statistic undefined"
    return out


def not_testable_label(why: dict[str, str]) -> str:
    """'not testable (coverage: a, b; gate2-trained: c)' from {network: reason}."""
    by: dict[str, list[str]] = {}
    for r, w in sorted(why.items()):
        by.setdefault(w, []).append(r)
    return "not testable (" + "; ".join(f"{w}: {', '.join(rs)}" for w, rs in sorted(by.items())) + ")"


def _with_sensitivity(main: dict, sens: dict) -> dict:
    robust = main["decision"] == sens["decision"]
    label = "" if robust else H4_NOT_ROBUST
    return {"decision": main["decision"], "robust_to_seed0": bool(robust), "label": label,
            "reported": main["decision"] + (f" ({label})" if label else ""),
            "main": main, "sensitivity": sens}


def h4_test(stats: dict[str, float], variants: dict = H4_VARIANTS,
            descriptive: dict[str, str] | None = None) -> dict:
    """H4 on each network's median G (``stats``: run -> statistic; module docstring): the
    decision test without the seed-0 baselines (4,900 relabellings) and the sensitivity with
    them (15,876), labelled when the two disagree. ``descriptive`` maps networks to why they
    are descriptive ('coverage', 'gate2-trained'): any of H4's networks there, or with an
    undefined statistic, makes H4 'not testable (...)' and no test is run."""
    why = _untestable(h4_networks(variants), stats, descriptive)
    if why:
        return {"decision": not_testable_label(why), "not_testable": why}
    main = _decided({v: ([stats[r] for r in d["real"]],
                         [stats[r] for r in d["baseline"] if not is_baseline_seed0(r)])
                     for v, d in variants.items()})
    sens = _decided({v: ([stats[r] for r in d["real"]], [stats[r] for r in d["baseline"]])
                     for v, d in variants.items()})
    return _with_sensitivity(main, sens)


def h4_transfer(stats: dict[str, float], spec: dict = H4_TRANSFER,
                descriptive: dict[str, str] | None = None) -> dict:
    """H4's statistic and test on the leave-out networks (one stratum): against v3 baseline
    seeds 1-4 (C(12, 8) = 495 relabellings) for the decision, with seed 0 (C(13, 8) = 1,287)
    as the labelled sensitivity; 'not testable (...)' as ``h4_test``."""
    why = _untestable(transfer_networks(spec), stats, descriptive)
    if why:
        return {"decision": not_testable_label(why), "not_testable": why}
    real = [stats[r] for r in spec["real"]]
    main = _decided({"transfer": (real, [stats[r] for r in spec["baseline"]
                                         if not is_baseline_seed0(r)])})
    sens = _decided({"transfer": (real, [stats[r] for r in spec["baseline"]])})
    return _with_sensitivity(main, sens)


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


DESCRIPTIVE_COVERAGE = "descriptive (coverage)"
DESCRIPTIVE_GATE2 = "descriptive (gate2-trained)"


def h2_network(summary: dict, coverage_entry: dict | None, gate2_ok: bool | None = None
               ) -> dict:
    """One H2' network's result (module docstring): ``summary`` from ``summarise`` with
    ``measures.h2_decision``; its band decision is kept as 'band_decision', and 'decision'
    becomes 'descriptive (gate2-trained)' when the network failed gate2-trained
    (``gate2_ok`` False; None: not run, e.g. before the tag) or 'descriptive (coverage)' when
    its excluded share of G exceeds 30% (``coverage_entry`` from ``measures.coverage``)."""
    out = dict(summary)
    out["band_decision"] = summary.get("decision")
    out["share_excluded"] = None if coverage_entry is None else coverage_entry["share_excluded"]
    out["descriptive"] = ""
    if gate2_ok is False:
        out["descriptive"] = "gate2-trained"
        out["decision"] = DESCRIPTIVE_GATE2
    elif coverage_entry is None or coverage_entry["descriptive"]:
        out["descriptive"] = "coverage"
        out["decision"] = DESCRIPTIVE_COVERAGE
    return out


def h2_variant(results: dict[str, dict], variant: str) -> str:
    """R18 over a variant's non-descriptive seeds (``h2_network`` results); with two or more
    descriptive seeds the variant is 'descriptive (coverage)' (or '(gate2-trained)')."""
    seeds = registry.variant_seeds(variant)
    desc = [a for a in seeds if a not in results or results[a].get("descriptive")]
    verdicts = {a: results[a]["decision"] for a in seeds if a not in desc}
    v = variant_rule(verdicts, variant)
    if v == "seed-specific" and len(desc) >= 2:
        whys = {results[a].get("descriptive") if a in results else "coverage" for a in desc}
        return DESCRIPTIVE_GATE2 if whys == {"gate2-trained"} else DESCRIPTIVE_COVERAGE
    return v


def t_label(lo: float, hi: float) -> str:
    """T's descriptive label on its CI (module docstring)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if hi <= MS.T_BLIND_HI:
        return "width-blind"
    if MS.T_ANCHORED[0] <= lo and hi <= MS.T_ANCHORED[1]:
        return "corridor-anchored"
    return "mixed"


# ------------------------------------------------------------------------- bands


def sign_label(lo: float, hi: float) -> str:
    """A CI against 0 (descriptive): 'below 0' if hi < 0, 'above 0' if lo > 0, 'contains 0'
    otherwise, 'undefined' when the CI is not finite."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    if hi < 0.0:
        return "below 0"
    if lo > 0.0:
        return "above 0"
    return "contains 0"


def step1_direction(label: str, predicted: int | None) -> str:
    """A ``sign_label`` against step 1's direction (H1-fast, descriptive): for -1 'consistent'
    below 0 and 'inconsistent' above 0; for 0 'consistent' when the CI contains 0, else
    'inconsistent'; '' (no direction: specialists) or 'undetermined' otherwise."""
    if predicted is None:
        return ""
    if label == "undefined":
        return "undetermined"
    if predicted == 0:
        return "consistent" if label == "contains 0" else "inconsistent"
    want, other = ("below 0", "above 0") if predicted < 0 else ("above 0", "below 0")
    return "consistent" if label == want else ("inconsistent" if label == other else "undetermined")


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


def with_coverage(summary: dict, coverage_entry: dict | None) -> dict:
    """A summary's 'decision' (descriptive bands, e.g. H2-speed's) replaced by 'descriptive
    (coverage)' when the outcome's excluded share exceeds 30%; the band kept beside it."""
    out = dict(summary)
    out["band_decision"] = summary.get("decision")
    out["share_excluded"] = None if coverage_entry is None else coverage_entry["share_excluded"]
    if coverage_entry is None or coverage_entry["descriptive"]:
        out["decision"] = DESCRIPTIVE_COVERAGE
    return out


def _approaches(rows: Sequence[MS.KeyMeasure], n: int) -> dict:
    """``summarise`` per approach ('A1', 'B', ...) when the rows hold more than one."""
    apps = sorted({str(r.approach) for r in rows})
    if len(apps) < 2:
        return {}
    return {a: summarise([r for r in rows if str(r.approach) == a], None, n) for a in apps}


def h1_fast(rows_sb: Sequence[MS.KeyMeasure], rows_v: Sequence[MS.KeyMeasure],
            rows_gain: Sequence[MS.KeyMeasure], agent: str, n: int = MS.BOOTSTRAP_N) -> dict:
    """H1-fast for one agent, descriptive (module docstring): the differences s_b,net - s_b*
    and v_apex,net - v_apex* (rows hold the per-key differences) and the s_b field's
    curvature gain (rows hold g), each left, right and pooled (and per approach), with the
    pooled CI's sign label and its agreement with step 1's direction."""
    pred = H1_FAST_SIGNS.get(registry.seed0_of(agent)) if registry.is_generalist(agent) else None
    out: dict = {"agent": agent, "predicted": pred, "descriptive": True}
    for name, rows in (("s_b", rows_sb), ("v_apex", rows_v)):
        res = summarise(rows, None, n)
        lab = sign_label(res["pooled"]["lo"], res["pooled"]["hi"])
        out[name] = {**res, "sign": lab, "step1_direction": step1_direction(lab, pred),
                     "by_approach": _approaches(rows, n)}
    out["gain"] = {**summarise(rows_gain, None, n), "by_approach": _approaches(rows_gain, n)}
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
    """H1-typical for one agent and outcome, descriptive (module docstring): per criterion
    the summary (left, right, pooled) of the network's gain, |location| and delta_rho, and
    'vs_control', the same of the per-key difference against the blind-pedal control's rows
    (``control``: criterion -> rows of the same measure). No pass/fail, no band."""
    absloc = [MS.KeyMeasure(r.key_id, r.pair_id, r.approach, r.direction, r.phase, r.n_phases,
                            r.R, r.theta_deg, abs(r.value), r.reason) for r in location]
    crit = {}
    for name, rows, raw in (("gain", gain, gain), ("location", absloc, location),
                            ("shape", shape, shape)):
        crit[name] = {**summarise(rows, None, n),
                      "vs_control": summarise(paired_rows(raw, control[name], name), None, n,
                                              asym=False)}
    crit["location"]["asymmetry"] = asymmetry_summary(location, n)    # on the signed location
    return {"descriptive": True, "criteria": crit}


# ------------------------------------------------------------------------- exclusions


def key_measure(meta: dict, value: float, flags: dict) -> MS.KeyMeasure:
    """One key's measure with its exclusion reason (``measures.exclusion_reason``): ``meta``
    holds key_id, pair_id, approach, direction, phase, n_phases, R, theta_deg; ``flags``
    holds invalid (bool), why_invalid, gate3_failed, numerical, inertia_failed,
    gate4_failed, kappa_clipped, off_plateau and the pipeline failures loop_exhausted,
    no_optimum and branch_failed. An excluded key's value is NaN."""
    reason = MS.exclusion_reason(bool(flags.get("invalid", False)),
                                 bool(flags.get("gate3_failed", False)),
                                 bool(flags.get("inertia_failed", False)),
                                 bool(flags.get("kappa_clipped", False)),
                                 bool(flags.get("off_plateau", False)),
                                 str(flags.get("why_invalid", "")),
                                 bool(flags.get("numerical", False)),
                                 loop_exhausted=bool(flags.get("loop_exhausted", False)),
                                 no_optimum=bool(flags.get("no_optimum", False)),
                                 branch_failed=bool(flags.get("branch_failed", False)),
                                 gate4_failed=bool(flags.get("gate4_failed", False)))
    v = float("nan") if reason else float(value)
    return MS.KeyMeasure(str(meta["key_id"]), int(meta["pair_id"]), str(meta["approach"]),
                         int(meta["direction"]), int(meta["phase"]), int(meta["n_phases"]),
                         float(meta["R"]), float(meta["theta_deg"]), v, reason)


def coverage_records(rows: Iterable[MS.KeyMeasure], agent: str, snapshot: str, outcome: str
                     ) -> list[dict]:
    """``measures.coverage``'s records from one agent's key measures of one outcome."""
    return [{"agent": agent, "snapshot": snapshot, "outcome": outcome, "reason": r.reason,
             "key_id": r.key_id} for r in rows]


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


def asymmetry_summary(rows: Sequence[MS.KeyMeasure], n: int = MS.BOOTSTRAP_N) -> dict:
    """The left-right asymmetry index of a measure (module docstring): the number of pairs
    and of partnered phases, and the median over pairs of each pair's median delta m with its
    cluster-bootstrap CI over geometries (NaN without a pair). A duplicate mirror partner
    (``measures.asymmetry``) leaves the index undefined, with the reason."""
    try:
        parts = MS.asymmetry(rows)
    except ValueError as e:
        return {"n_pairs": 0, "n_phases": 0, "est": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "error": str(e)}
    pairs = MS.asymmetry_pairs(rows)
    if not pairs:
        return {"n_pairs": 0, "n_phases": len(parts), "est": float("nan"), "lo": float("nan"),
                "hi": float("nan")}
    geo = {(int(r.pair_id), str(r.approach)): (float(r.R), float(r.theta_deg)) for r in rows}
    keys = list(pairs)
    est, lo, hi = MS.cluster_bootstrap([pairs[k] for k in keys], [geo[k] for k in keys],
                                       np.nanmedian, n, MS.LEVEL, MS.stream_rng("bootstrap"))
    return {"n_pairs": len(pairs), "n_phases": len(parts), "est": est, "lo": lo, "hi": hi}


def summarise(rows: Sequence[MS.KeyMeasure], decide_fn: Callable[[float, float], str] | None = None,
              n: int = MS.BOOTSTRAP_N, asym: bool = True) -> dict:
    """One agent's measure per direction and pooled (``measures.by_direction``) with, given
    ``decide_fn(lo, hi)``, the pooled decision, and (``asym``) its left-right asymmetry index
    (``asymmetry_summary``; descriptive)."""
    out = {k: v.as_dict() for k, v in MS.by_direction(rows, n).items()}
    if decide_fn is not None:
        out["decision"] = decide_fn(out["pooled"]["lo"], out["pooled"]["hi"])
    if asym:
        out["asymmetry"] = asymmetry_summary(rows, n)
    return out
