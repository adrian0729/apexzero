"""Analysis: pair values, the pre-registered decision rules, intervals and the dry run.

Protocol 2.2 (errors, band, S), 2.4 (P1-P8), 7 (classifier validation, dry run, power
table) and 9 (units, inference, exclusions). The design is a fixed grid run by a
deterministic agent, so P1-P3 are k-of-n criteria over mirror pairs with leave-one-out
robustness for slopes, without p-values; crash rates use a beta-binomial fit with a
profile-likelihood interval (99% for confirmatory criteria, 95% otherwise; Clopper-Pearson
on pooled counts if the overdispersion estimate is 0); P6a and P1c use permutation nulls.

Tables (``io.Table`` column dicts; the column sets below are the contract):
- key table: one row per cell, the u = 0 keys' values and the cell's geometry, KEY_COLUMNS;
- run table: one row per run (agent, snapshot, cell, phase, mode, sample), RUN_COLUMNS;
- pair table: one row per (agent, snapshot, pair), PAIR_COLUMNS;
- tier D condition table: one row per (agent, snapshot, corner, condition), COND_COLUMNS;
- P8 ladder and twin tables (LADDER_COLUMNS, TWIN_COLUMNS), P7 tables (P7_SD_COLUMNS,
  P7_ALIAS_COLUMNS);
- decision table: one row per (prediction, agent), DECISION_COLUMNS.
Every decision function reads only columns, so a simulated agent (dry run) and a real one
go through identical code.

Pair values (9.1): the pair is (geometry, approach) with 2 directions x 4 phases = 8
deterministic runs. Speed values are the median of the 8 runs, where an overspeed crash
counts as +inf, a crash before s_c0 as missing, a non-overspeed crash after s_c0 keeps its
v_e and loses its v_min; more than 3 missing makes the value undefined (listed, still in
the crash analyses). An error's pair value is the median of the per-run errors, each
against the key solved at that run's phase when one exists, else the u = 0 key. Onset
values are the median over runs with a sustained brake (at least 5 required).

Choices where the protocol leaves room (AD1-AD25), signed off as written by the study
owner before the tag (protocol 12, ruling R13):
- AD1  An overspeed crash is +inf even when it ends before s_c0 (the +inf rule comes first:
       it exists to stop survivorship bias, M4); other crashes before s_c0 are missing.
- AD2  "Crash" is any termination within the window (the run's ``crashed`` flag).
- AD3  The 3-of-8 and 5-of-8 rules scale with the run count (16-phase P7 pairs have 32):
       at most floor(3n/8) missing, at least ceil(5n/8) onsets.
- AD4  Forced onsets are left out of the d_b, Δd_b and v_b pair values (9.4) and still
       count in n_brake and forced_frac.
- AD5  Where e_ve uses the re-anchored key (A1, tier B D = 600), S and r use v_e*RA too
       (column v_e_star_band); elsewhere v_e* (of the phase-matched keys under AD23's V7
       remedy). Ruling R12: P0's re-anchored keys cover every
       non-clipped below-support, tail and bulk A1 cell, so P1b and P2 read e_ve^RA
       throughout A1; a run without one falls back to e_ve, and its e_ve_bo_ra to its
       e_ve_bo, and is counted (ra_key, ra_fallback).
- AD6  "Pooled within-θ range of log v_e*" = max - min of the within-θ-centred values.
- AD7  A leave-one-R-out slope that cannot be computed fails the confirm criterion and does
       not count as "outside" for the refute.
- AD8  An undefined pair (side 'undefined') counts as not in band, and as neither side for
       the same-side refute.
- AD9  P1b and P2's below-support set is R_seg < q1 by design (regimes 'below' and 'clip')
       minus the pairs κ-clipped at run time (the binding definition). Kendall τ is taken
       within each approach; every approach that contributes below-support pairs must meet
       it.
- AD10 P1-tail's contrast is positive only if e_ve^RA(A1) > e_ve(A2) strictly; geometries
       with an undefined value are dropped; +inf at both approaches is not positive.
- AD11 R_dep is +inf when the largest braking R already fails; multi-basin pairs are
       skipped in the walk.
- AD12 P1c gives the ordering decision and one "A1 >= A2" decision per agent.
- AD13 P2(a): a no-brake entry in O counts as Δd_b < 0 (P2b's rule); the Δd_resp sign is
       judged over all of O. P2(b) without any crash cell is not met. (b′) "majority" is the
       most frequent type.
- AD14 P2b decomposes pair medians (v_b, observed v_e, d_b) against the pair's median
       phase-matched key values (v_b*, v_e*, d_b* of the key behind each run's onset error;
       the u = 0 key where no phase key was solved), protocol 6.1.
- AD15 P3's held-out folds are the cells' (v0, D); runs of a geometry with no training runs
       in a fold are not predicted (the same rows for every model).
- AD16 P4's single-factor F1 conditions are the three that change one factor of (a). The
       class 2 rule is applied literally; when no F6 perturbation is as far OOD as the
       failing condition it is vacuous, and details flag it. When a single-factor condition
       fails and its d_OOD or an F6 condition's is missing (no buffer), classes 1 and 2
       cannot be told apart: P4 is 'not testable' (class -1), never class 2 by default.
- AD17 P5 crash concordance is agreement of the crash flag over mirrored (pair, phase) runs.
- AD18 P6: a checkpoint failing the gate counts as not competent; P6a confirm also needs
       the permutation p <= 0.05; t_bulk = max t_R over bulk R levels; P6c is inconclusive
       if the bulk never becomes competent.
- AD19 P7 compares medians of A_alias; 'mixed' = closed-loop excess absent (<= 1 m in >= 75%
       of cells) while the agent's A_alias exceeds Qp's.
- AD20 P8 twin agreement: the twin reproduces the crash or overspeeds (e_ve > b or a
       deterministic overspeed crash).
- AD21 Stability (protocol 8): a claim that does not recur at one of the two previous
       snapshots is reported 'unstable', with the final verdict in details.
- AD22 Clip sensitivity (9.4): runs with any edge, ego or context clipping are removed and
       the pairs recomputed; if the verdict changes, that version is primary.
- AD23 e_ve, e_vmin and e_vmin^line are against the nominal u = 0 key (2.2); Δd_b, Δd_resp,
       regret and Δt_corner against the key solved at the run's phase when one exists, else
       the u = 0 key. V7 remedy (6.5): when the manifest's validation V7 failed on speeds,
       phase-matched keys exist for every braking cell and the speed errors move to the
       onset errors' key as well (``run_table(speed_at_phase=True)``, the verdict read by
       ``v7_speeds_failed``): the key at the run's phase when one exists, else the u = 0
       key. e_ve_bo then uses a backed-off key solved at the run's phase if one exists (none
       is solved, 6.1), else the u = 0 backed-off key. Run columns speed_at_phase and
       bo_at_phase record the choice, key_at_phase and v_e_star_k name the key; S and r
       (v_e_star_band) use the median v_e* of the same keys wherever e_ve is not re-anchored.
- AD24 Protocol 8's gate applies to every snapshot read for P1-P3, the finals and the two
       history snapshots included: a snapshot that fails it (or cannot be judged: no gentle
       or tier G runs, undefined oscillation index) gives 'not testable'. Gentle cells and
       tier G straights must each reach 90%.
- AD25 A key's measures are recomputed from its cached solution (``CachedKeys``), with the
       null key's run for the catch-trial onset d_resp*.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, log_expit

from apexzero.interp import measures
from apexzero.interp.constants import (
    BAND_EXPECTED,
    DT,
    MULTI_BASIN_FLOORS,
    N_PHASES,
    R_CONTROLS,
    REANCHOR_IMMATERIAL_FRACTION,
    REANCHOR_IMMATERIAL_KPH,
    ROOT_SEED,
    STREAM_OFFSET,
    STREAMS,
    THETA_CLASS_EDGE_DEG,
    TWIN_TOL,
    V_MIN_MARGIN,
    VIABILITY_CAP,
    VIABILITY_SUBSET,
    kph,
)
from apexzero.interp.io import (
    ANALYSIS_DIR,
    Table,
    concat_tables,
    save_table,
    select,
    stream_rng,
    write_json,
)
from apexzero.interp.measures import KEY_ERROR_COLUMNS, METRIC_COLUMNS
from apexzero.interp.stimuli import COND_A, COND_ISO, D_CONDITIONS

# ------------------------------------------------------------------------- schemas

RUN_INDEX_COLUMNS = {
    "agent": "str: 'S1'..'G3' or a control name ('Q', 'c1', ...)",
    "timesteps": "int: snapshot timesteps (0 for controls)",
    "cell": "int: cell_id", "stim": "int: stim_id", "pair": "int: pair_id",
    "phase": "int: k of u_k", "sample": "int: -1 deterministic", "mode": "str: det|nat|mat",
    "steer_sign": "int: +1, or -1 on the mirrored member", "W": "int: steps", "s0": "m",
}
RUN_EXTRA_COLUMNS = {
    "e_ve_bo": "v_e / v_e*bo - 1 (backed-off key)",
    "e_ve_ra": "v_e / v_e*RA - 1 (re-anchored key; = e_ve where re-anchoring does not apply)",
    "key_at_phase": "bool: a key was solved at this run's phase",
    "crash_type": "str: overspeed | overbrake | line_steer | other | '' (9.1)",
    "v_e_star_ra": "m/s: v_e* of the key behind e_ve_ra (the key behind e_ve without "
                   "re-anchoring)",
    "v_e_star_k": "m/s: v_e* of the key behind the onset errors (phase-matched where solved)",
    "v_b_star_k": "m/s: its v_b*", "d_b_star_k": "m: its d_b*",
    "ra_key": "bool: e_ve_ra is against a re-anchored key (False: the nominal u = 0 fallback)",
    "speed_at_phase": "bool: e_ve, e_vmin and e_vmin^line are against the key behind the "
                      "onset errors (key_at_phase, v_e_star_k), because V7 failed on speeds "
                      "(6.5, AD23); False: against the nominal u = 0 key (2.2)",
    "bo_at_phase": "bool: e_ve_bo is against a backed-off key solved at this run's phase "
                   "(u = 0 runs: the u = 0 backed-off key); False: the u = 0 one, or none",
}
RUN_COLUMNS = (tuple(RUN_INDEX_COLUMNS) + METRIC_COLUMNS + KEY_ERROR_COLUMNS
               + tuple(RUN_EXTRA_COLUMNS))

KEY_COLUMNS = {
    # identity and design
    "cell": "int", "stim": "int", "pair": "int", "name": "str: stimulus name",
    "geometry": "str: stimulus name without its direction (shared by L, R and approaches)",
    "tiers": "str: comma-joined tiers", "approach": "str", "v0": "m/s", "D": "m (nan: unset)",
    "W": "int (-1: unset)", "n_phases": "int", "direction": "int: +1 left, -1 right, 0",
    "s_c0": "m", "s_a0": "m", "s_apex": "m", "s_c1": "m",
    "R_nom": "m (nan for real roads)", "theta_deg": "deg", "width": "m", "ramp_kind": "str",
    "no_cap": "bool", "R_seg": "m", "R_eff": "m",
    "corner": "int: tier D Hockenheim corner index (-1)", "condition": "str: tier D condition",
    "swap_j": "int: F5 swap A partner (-1)",
    # nominal key at u = 0
    "key_status": "str: verified | unsolved | infeasible_bound | skipped | missing",
    "J_star": "", "v_e_star": "m/s", "v_min_star": "m/s", "d_b_star": "m", "v_b_star": "m/s",
    "has_brake_star": "bool", "tau_vmin": "m/s (tolerance used for braking/flat)",
    "multi_basin": "bool", "lock_limited": "bool", "slip_limited": "bool",
    "braking": "bool: key has a sustained brake and v_min* < v0 - tau (2.3)",
    # backed-off key, pinned s-OCP, QSS
    "v_e_star_bo": "m/s", "v_min_star_bo": "m/s",
    "v_e_socp": "m/s", "v_min_socp": "m/s", "d_b_socp": "m", "v_b_socp": "m/s",
    "braking_socp": "bool: the s-OCP has a sustained brake and v_min < v0 - tau (tier D)",
    "socp_cell": "int: cell whose s-OCP filled the s-OCP columns (tier D: the S1-state cell)",
    "tau_ve": "m/s: the nominal key's v_e tolerance (6.3)", "tau_db": "m: its d_b tolerance",
    "v_min_qss": "m/s: centreline QSS v_min (P2d; nan unless supplied)",
}

PAIR_COLUMNS = {
    # identity
    "agent": "str", "timesteps": "int", "pair": "int: pair_id", "tier": "str: primary tier",
    "tiers": "str: comma-joined tiers of the pair's cells",
    "approach": "str: A1 | A2 | B | E-fast | REC | real | ...", "v0": "m/s", "D": "m", "W": "int",
    # geometry and keys
    "geometry": "str", "R_nom": "m", "theta_deg": "deg", "width": "m", "ramp_kind": "str",
    "no_cap": "bool", "R_seg": "m", "R_eff": "m", "v_min_star": "m/s (nominal key)",
    "v_e_star": "m/s", "v_e_star_bo": "m/s",
    "v_e_star_ra": "m/s (median over the pair's anchored keys)",
    "v_e_star_band": "m/s: v_e*RA where the band uses it (A1, B at D = 600), else v_e* of the "
                     "key behind e_ve (speed_at_phase: the runs' median v_e*_k, AD23)",
    "speed_at_phase": "bool: the runs' speed errors are against their phase-matched keys "
                      "(V7 failed on speeds, AD23)",
    "d_b_star": "m", "v_b_star": "m/s", "braking": "bool: key brakes and v_min* < v0 - tau",
    "v_e_star_k": "m/s: median over the runs' phase-matched keys (P2b; = v_e* without them)",
    "v_b_star_k": "m/s", "d_b_star_k": "m",
    "key_status": "str: verified | unsolved | infeasible_bound", "multi_basin": "bool",
    "lock_limited": "bool", "slip_limited": "bool", "v_min_qss": "m/s",
    # regime and support
    "regime": "str: clip | below | tail | bulk (design, from R_seg)",
    "kappa_clipped": "bool: any deterministic run clipped a preview kappa channel before s_c1",
    "clip_other": "bool: edge, ego or context clipping in any deterministic run",
    "log_support": "joint kNN log density at this approach",
    "log_support_tightfast": "log density of tight (< 20 m) corners reached at >= 250 km/h",
    # deterministic speed values
    "n_runs": "int (8)", "n_missing": "int", "defined": "bool: <= 3 of 8 missing",
    "uses_ra": "bool: the band and S use the re-anchored key (A1, tier B D = 600)",
    "ra_fallback": "int: deterministic runs of a uses_ra pair with no re-anchored key, whose "
                   "e_ve_ra fell back to e_ve (R12: none in P1b's A1 set once P0's anchored "
                   "keys exist)",
    "v_e": "m/s median (+inf for overspeed crashes)",
    "v_e_obs": "m/s median of the observed v_e (no +inf; P2b)",
    "e_ve": "median per-run error", "e_ve_bo": "", "e_ve_ra": "",
    "e_ve_bo_ra": "backed-off error scaled to the re-anchored key: (1 + e_ve_ra) v_e*/v_e*bo - 1 "
                  "on runs with one (ra_key), e_ve_bo on runs without",
    "side": "str: in | over | under | undefined (band b)",
    "v_min": "m/s", "e_vmin": "", "e_vmin_line": "", "e_vmin_line_excess": "",
    "r_extrap": "log residual vs bulk fit",
    # onsets
    "n_brake": "int: runs with a sustained brake", "d_b": "m", "dd_b": "m", "v_b": "m/s",
    "d_resp": "m", "dd_resp": "m", "forced_frac": "", "unsettled_frac": "",
    "time_censored_frac": "",
    # crashes
    "det_crashes": "int of n_runs", "det_crash_frac": "", "det_overspeed": "int",
    "det_complete": "int: deterministic runs completing (no termination before s_c1 + 250)",
    "nat_k": "int", "nat_n": "int", "mat_k": "int", "mat_n": "int",
    "typed_crashes": "int: typed crashes (det + matched)", "typed_overspeed": "int",
    "typed_overbrake": "int", "typed_line_steer": "int", "typed_other": "int",
    "overspeed_frac": "typed crashes (det + matched) classified overspeed",
    "dt_corner": "s median", "regret": "median (u = 0 runs with a key at phase)",
    "osc_index": "median oscillation index of the deterministic runs",
    "excluded": "str: '' or the exclusion reason (9.4)",
}

COND_COLUMNS = {
    "agent": "str", "timesteps": "int", "corner": "int: Hockenheim corner index",
    "condition": "str: stimuli.D_CONDITIONS", "braking": "bool: key brakes in condition (a)",
    "completes": "int: deterministic phase runs completing (of 4)",
    "complete": "bool: >= 3 of 4", "v_min": "m/s median", "v_peak": "m/s",
    "v_min_region": "m/s: min over [s_c - 50, s_c + L_c + 50] (swap B)",
    "d_ood": "median normalised NN distance to the agent's Hockenheim buffer",
    "v_min_star": "m/s: min-time (s-OCP) v_min* in this condition (S_spec)",
    "swap_j": "int: swap A partner corner (-1)",
}

LADDER_COLUMNS = {
    "agent": "str: the crashing agent", "unit": "str: '<agent>/<circuit>/<s_crash>'",
    "set": "str: tail | bulk | below (P8 prediction sets)",
    "version": "str: stimuli.LADDER_VERSIONS", "fidelity_ok": "bool (real rung: gate passed)",
    "fidelity_missing": "bool (real rung: gate not evaluated, its inputs missing; not a "
                        "failure, the unit stays in P8 as unreproduced)",
    "det_crashes": "int (of det_n phases)", "det_n": "int", "nat_k": "int", "nat_n": "int",
    "e_ve": "median e_ve (+inf for overspeed crashes)", "det_overspeed": "int",
}

TWIN_COLUMNS = {
    "agent": "str", "corner": "str", "source": "str: E1 | E2",
    "e_ve_real": "real rung", "e_ve_isolated": "", "e_ve_twin": "",
    "e_ve_curve": "the agent's tier A e_ve curve at the twin's R_seg, theta class, approach",
    "length_shortfall": "m: real L_seg minus the twin's (R4 covariate; > 2 m misses the "
                        "length tolerance)",
}

P7_SD_COLUMNS = {
    "agent": "str", "geometry": "str", "R_nom": "m", "theta_deg": "deg",
    "sd_agent": "m: SD of d_b across the 16 phases (mean over directions)",
    "sd_key": "m: key's SD across its phase-matched solutions (R <= 18.3; nan otherwise)",
    "sd_qp": "m: Qp's SD", "v_b": "m/s", "sd_null": "m: p7_null", "excess": "m",
}

P7_ALIAS_COLUMNS = {
    "agent": "str", "geometry": "str", "R_nom": "m", "theta_deg": "deg",
    "short": "bool: 90 deg and R <= 14.6", "a_agent": "A_alias", "a_q": "", "a_qp": "",
}

DECISION_COLUMNS = {
    "prediction": "str: P1a, P1-tail, P1b, P1c, P2, P2b, P3, P4, P5, P6a, P6c, P7, P8-tail, ...",
    "agent": "str", "family": "str: primary | secondary | exploratory | control",
    "verdict": "str: confirm | refute | inconclusive | not testable | not applicable"
               " | cautious but lawful | not estimable | class <n> | unstable | mixed"
               " | descriptive",
    "stable": "str: yes | no | n/a (holds at the final and >= 1 of the two before it)",
    "details": "str: JSON of the numbers behind the verdict",
}

DRY_RUN_COLUMNS = {"kind": "str", "prediction": "str", "expected": "str", "verdict": "str",
                   "passed": "bool", "details": "str: JSON"}

POWER_COLUMNS = {
    "test": "str", "agent": "str",
    "pairs_A1_clip": "int", "pairs_A1_below": "int", "pairs_A1_tail": "int",
    "pairs_A1_bulk": "int", "pairs_A2_clip": "int", "pairs_A2_below": "int",
    "pairs_A2_tail": "int", "pairs_A2_bulk": "int",
    "min_overspeed": "smallest planted overspeed detected (2.5-point steps; nan: none)",
    "lawful_verdict": "str: the rule's verdict on the lawful agent",
    "planted_verdict": "str: verdict at min_overspeed (P3: on the time-to-corner agent)",
}

# Decision thresholds of section 2.4 (the band b itself comes from V8 via the manifest).
P1A_IN_BAND = 0.80
P1A_S_RANGE, P1A_S_LOO_RANGE = (0.8, 1.2), (0.75, 1.25)
P1A_MAX_DET_CRASH_PAIRS = 1
P1A_PCRASH_UPPER = 0.05
P1A_REFUTE_SAME_SIDE = 0.50
S_MIN_LOG_RANGE = 0.15
P1TAIL_OVERSPEED = 0.60
P1TAIL_CONTRAST = 0.75
P1B_MIN_PAIRS = 4
P1B_FRACTION = 0.75
P1B_KENDALL = -0.3
P2_MIN_SET = 3
P2_ONSET_FRACTION = 0.75
P2_OVERSPEED_MEDIAN = 0.75
P2_REFUTE_ONSET = 0.50
P2_REFUTE_OVERSPEED = 0.50
P2B_SHARE = 0.50
P2B_BETA_TOL = 0.1
P3_MIN_DROP = 30.0 / 3.6     # v_b - v_e >= 30 km/h
P3_FORCED = 0.5              # s: an onset within 0.5 s of the start is forced
P3_B_RANGE, P3_B_LOO_RANGE = (0.9, 1.1), (0.85, 1.15)
P3_REFUTE_MARGIN = 0.10
P3_SEPARATION = 0.25
P3_IDX_TIME, P3_IDX_CLIP = 6.0, (150.0, 500.0)
P3_BRAKE_SHARE = 0.5
P4_GATE, P4_FAIL, P4_PASS = 0.9, 0.5, 0.8
P4_MAIN_EFFECT = 0.4
P4_PHI_MEM, P4_M_MEM, P4_S_MEM = 0.5, 0.5, 0.4
P4_PHI_READ, P4_M_READ, P4_S_READ = 0.25, 0.25, (0.6, 1.4)
P4_TIER_A_R, P4_TIER_A_COMPLETION = 10.7, 0.8
P5_DVE_PP, P5_DDB_M, P5_CONCORDANCE = 0.02, 5.0, 0.90
P6A_RHO, P6_PERMUTATIONS, P6A_ALPHA = -0.5, 10_000, 0.05
P6D_LARGE, P6D_MID = (69.8, 87.3), (28.6, 35.8, 44.7)
P7_EXCESS_M, P7_FLAT_M, P7_FRACTION, P7_ALIAS_RATIO = 2.0, 1.0, 0.75, 2.0
P7_SHORT_R = 14.6
P8_FAIL_PCRASH, P8_REPRO_PCRASH, P8_REPRO_DET = 0.2, 0.5, 2
P8_TWIN_PP, P8_TWIN_FRACTION, P8_SPEARMAN = 0.05, 0.80, 0.5
# P8 thresholds are proportional (ruling R2): of the n tail units, confirm on >= ceil(3n/5)
# reproduced and single-corner with twin agreement, refute on <= floor(n/5); a tail of
# n < 3 is not testable (R2). The bulk set's rule is unchanged: context in >= 2/3 of its n
# reproduced units (>= ceil(2n/3)), not testable only with none reproduced.
P8_TAIL_CONFIRM, P8_TAIL_REFUTE, P8_BULK_CONTEXT = (3, 5), (1, 5), (2, 3)
P8_MIN_UNITS = 3             # P8-tail only
UNSETTLED_REGIME = 0.20      # a regime is flagged if > 20% of its runs are unsettled
GATE_COMPLETION, GATE_OSC = 0.90, 0.10
GENTLE_R = 87.3              # with R_CONTROLS: the checkpoint gate's gentle cells
BOOTSTRAP_N = 20_000

EXCL_ALL = ("infeasible_bound", "unsolved")             # removed from everything (9.4)
EXCL_KEY = EXCL_ALL + ("multi_basin", "no_key")         # removed from key-based errors
GENERALISTS = ("G1", "G2", "G3")
SPECIALISTS = ("S1", "S2", "S3")
CONFIRMATORY_G = ("P1a", "P1-tail", "P1b", "P2", "P3")
EXPLORATORY = ("P2d", "P6b", "P6d", "P8-F")
CLAIMS = ("confirm", "refute", "cautious but lawful")
TIER_ORDER = ("A", "B", "C", "D", "E1", "E2", "E3", "G", "V", "O", "pilot")
MODES = ("det", "nat", "mat")


@dataclass
class Decision:
    prediction: str
    agent: str
    family: str
    verdict: str
    stable: bool | None = None
    details: dict = field(default_factory=dict)


class Regime(NamedTuple):
    """The regime boundaries analysis reads (the fields of ``support.RegimeTable``, which
    works wherever a Regime does)."""

    q1: float
    q10: float
    r_clip_max: float = float("nan")


def family_of(prediction: str, agent: str) -> str:
    """Protocol 8's families: G3, S2, S3 and the recipe claim are primary; G1, G2, S1 and
    G2's P8 tail test secondary; P2d, P6b, P6d exploratory; P5 a control."""
    if prediction in CONFIRMATORY_G and agent in GENERALISTS:
        return "primary" if agent == "G3" else "secondary"
    if prediction == "P4":
        return "primary" if agent in ("S2", "S3") else "secondary"
    if prediction == "P4-recipe":
        return "primary"
    if prediction in EXPLORATORY or (prediction.startswith("P6") and agent in SPECIALISTS):
        return "exploratory"
    if prediction == "P5":
        return "control"
    return "secondary"


def _d(prediction: str, agent: str, verdict: str, **details) -> Decision:
    return Decision(prediction, agent, family_of(prediction, agent), verdict, None, details)


# ------------------------------------------------------------------------- helpers


def _n(t: Table) -> int:
    for v in t.values():
        return len(v)
    return 0


def _get(t: Table, name: str, default) -> np.ndarray:
    if name in t:
        return np.asarray(t[name])
    return np.full(_n(t), default)


def _fl(t: Table, name: str) -> np.ndarray:
    return _get(t, name, np.nan).astype(float)


def _bo(t: Table, name: str) -> np.ndarray:
    a = _get(t, name, False)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=0.0)
    return a.astype(bool)


def _st(t: Table, name: str) -> np.ndarray:
    return _get(t, name, "").astype(str)


def _it(t: Table, name: str, default: int = -1) -> np.ndarray:
    a = _get(t, name, default)
    if a.dtype.kind == "f":
        a = np.where(np.isfinite(a), a, default)
    return a.astype(np.int64)


def _frac(mask: np.ndarray) -> float:
    mask = np.asarray(mask, bool)
    return float(mask.mean()) if mask.size else float("nan")


def _median(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def _column(vals: list) -> np.ndarray:
    if not vals:
        return np.zeros(0)
    if all(isinstance(v, (bool, np.bool_)) for v in vals):
        return np.asarray(vals, dtype=bool)
    if all(isinstance(v, str) for v in vals):
        return np.asarray(vals, dtype=str)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
           for v in vals):
        return np.asarray(vals, dtype=np.int64)
    return np.asarray([np.nan if v is None else v for v in vals], dtype=float)


def _to_table(rows: list[dict], columns) -> Table:
    return {c: _column([r.get(c, np.nan) for r in rows]) for c in columns}


def _in(x: float, lo_hi: tuple[float, float]) -> bool:
    return bool(np.isfinite(x) and lo_hi[0] <= x <= lo_hi[1])


def _out(x: float, lo_hi: tuple[float, float]) -> bool:
    return bool(np.isfinite(x) and not lo_hi[0] <= x <= lo_hi[1])


def has_tier(t: Table, tier: str) -> np.ndarray:
    """Rows whose (comma-joined) ``tiers`` include ``tier``."""
    col = _st(t, "tiers") if "tiers" in t else _st(t, "tier")
    return np.array([tier in s.split(",") for s in col], dtype=bool)


def geometry_name(name: str) -> str:
    """Stimulus name without its direction suffix: the geometry shared by L and R."""
    return name[:-2] if name.endswith(("/L", "/R")) else name


def regime_of(R_seg: float, q1: float, q10: float, r_clip_max: float = float("nan")) -> str:
    """Protocol 2.3 (the rule of ``support.regime``): 'clip' (R_seg <= R_clip^max), 'below'
    (< q1), 'tail' (q1 <= R < q10) or 'bulk' (>= q10); '' for a NaN radius."""
    if not np.isfinite(R_seg):
        return ""
    if np.isfinite(r_clip_max) and R_seg <= r_clip_max:
        return "clip"
    if R_seg < q1:
        return "below"
    if R_seg < q10:
        return "tail"
    return "bulk"


def band_errors(pairs: Table) -> tuple[np.ndarray, np.ndarray]:
    """(e_ve, e_ve^bo) the band uses: the re-anchored pair where ``uses_ra``."""
    ra = _bo(pairs, "uses_ra")
    e = np.where(ra, _fl(pairs, "e_ve_ra"), _fl(pairs, "e_ve"))
    ebo = np.where(ra, _fl(pairs, "e_ve_bo_ra"), _fl(pairs, "e_ve_bo"))
    return e, ebo


def sides(pairs: Table, b: float) -> np.ndarray:
    e, ebo = band_errors(pairs)
    return np.array([band_side(x, y, b) for x, y in zip(e, ebo, strict=True)], dtype=str)


def _mask(pairs: Table, *, tier: str | None = "A", approach: str | None = None,
          regimes: tuple[str, ...] | None = None, braking: bool | None = True,
          no_cap: bool | None = False, exclude: tuple[str, ...] = EXCL_KEY) -> np.ndarray:
    m = np.ones(_n(pairs), bool)
    if tier:
        m &= has_tier(pairs, tier)
    if approach:
        m &= _st(pairs, "approach") == approach
    if regimes is not None:
        m &= np.isin(_st(pairs, "regime"), regimes)
    if braking is not None:
        m &= _bo(pairs, "braking") == braking
    if no_cap is not None:
        m &= _bo(pairs, "no_cap") == no_cap
    if exclude:
        m &= ~np.isin(_st(pairs, "excluded"), exclude)
    return m


def _agent(t: Table) -> str:
    a = _st(t, "agent")
    return str(a[0]) if a.size else ""


def _min_onsets(n_runs: int) -> int:
    return math.ceil(5 * n_runs / 8)


def _max_missing(n_runs: int) -> int:
    return (3 * n_runs) // 8


# ------------------------------------------------------------------------- keys


class KeyView(NamedTuple):
    """One key as analysis sees it: its trajectory (for key_errors and crash typing), its
    measures (``ResponseMetrics`` fields), and its status and flags."""

    traj: Any                    # measures.Trajectory (None if only measures are known)
    metrics: Any                 # ResponseMetrics, or anything with its attribute names
    at_phase: bool = True        # solved at the run's phase (False: the u = 0 key is used)
    status: str = "verified"
    multi_basin: bool = False
    tau: dict | None = None      # measure -> tolerance (answer_key.KeyRecord.tau)
    active: dict | None = None   # slip_limited / lock_limited / corridor_limited
    J: float = float("nan")


class KeySource(Protocol):
    """Where run_table and key_table get keys from. ``CachedKeys`` reads answer_key's
    cache; tests and the dry run use doubles. Optional: ``backoff_at(cell, phase)``, a
    backed-off key solved at that phase (AD23's V7 remedy); a source without it has none."""

    def nominal(self, cell, phase: int) -> KeyView | None: ...
    def backoff(self, cell) -> KeyView | None: ...
    def anchored(self, cell, agent: str, timesteps: int) -> KeyView | None: ...
    def null(self, cell) -> KeyView | None: ...
    def socp(self, cell) -> KeyView | None: ...
    def no_return(self, cell, agent: str, timesteps: int, phase: int, sample: int, traj,
                  key_traj) -> Any: ...


NO_KEY_FLAG = "no_key"     # Cell flag of budget rule 2's cut cells (ruling R7)


def has_no_key(cell) -> bool:
    """Ruling R7: a cell cut from the keys by budget rule 2 carries the flag 'no_key'. It is
    still rolled out, but no key (nominal, phase, backed-off, re-anchored, viability) is
    read for it, so it drops out of every key-based error (excluded = 'no_key')."""
    return NO_KEY_FLAG in tuple(getattr(cell, "flags", ()) or ())


class _NoKeyFilter:
    """A ``KeySource`` that hides every key of a 'no_key' cell (R7) and passes the rest
    through; nulls and the layout s-OCP are not keys of the cut and stay."""

    def __init__(self, keys):
        self._keys = keys

    def __getattr__(self, name):
        return getattr(self._keys, name)

    def nominal(self, cell, phase: int):
        return None if has_no_key(cell) else self._keys.nominal(cell, phase)

    def backoff(self, cell):
        return None if has_no_key(cell) else self._keys.backoff(cell)

    def backoff_at(self, cell, phase: int):
        f = getattr(self._keys, "backoff_at", None)
        return None if has_no_key(cell) or f is None else f(cell, phase)

    def anchored(self, cell, agent: str, timesteps: int):
        return None if has_no_key(cell) else self._keys.anchored(cell, agent, timesteps)

    def null(self, cell):
        return self._keys.null(cell)

    def socp(self, cell):
        return self._keys.socp(cell)

    def no_return(self, cell, *args):
        return None if has_no_key(cell) else self._keys.no_return(cell, *args)


def _attr(m: Any, name: str, default=float("nan")):
    if m is None:
        return default
    if isinstance(m, dict):
        return m.get(name, default)
    return getattr(m, name, default)


def _metrics_from(d: Any):
    if d is None or isinstance(d, measures.ResponseMetrics):
        return d
    if isinstance(d, dict):
        return measures.ResponseMetrics(**{k: v for k, v in d.items() if k in METRIC_COLUMNS})
    return d


class CachedKeys:
    """``KeySource`` over ``answer_key``'s cache (``key_for`` mirrors right-hand keys).

    ``anchor_id(agent, timesteps)`` names the snapshot in ``key_for(..., agent=...)`` for
    re-anchored keys: '<agent>@<timesteps>', the alias tag ``answer_key.anchored_key`` writes
    for ``agent=<agent>, snapshot=str(timesteps)`` (the anchored-keys stage's convention).
    ``viability`` maps (agent, timesteps, cell_id, phase, sample) to that run's viability
    KeyRecords (``answer_key.viability_keys``). Views are cached per key.

    A key's measures are recomputed here from its cached trajectory with the current
    ``measures`` code and ``tables`` (``recompute``, the default), never read from the
    record: the cache is filled before the tag, so a later fix to the measures, landmarks or
    tables would otherwise leave e_ve*, d_b*, v_min* and tau stale. A corner key's catch-trial
    onset d_resp* needs the same driver's null run (protocol 5), so it is measured against the
    null TO-OCP key of the cell's null cell (same v0, width and W); s-OCPs have no null."""

    def __init__(self, bank, cells: list, cache=None,
                 anchor_id: Callable[[str, int], str] = lambda a, t: f"{a}@{int(t)}",
                 viability: dict | None = None, tables=None, recompute: bool = True):
        self.bank, self.cache, self.anchor_id = bank, cache, anchor_id
        self.cells = {c.cell_id: c for c in cells}
        self.viability = viability or {}
        self.tables, self.recompute = tables, recompute
        self._memo: dict = {}

    def _record(self, cell, kind: str, phase: int = 0, agent: str = ""):
        from apexzero.interp import answer_key

        return answer_key.key_for(cell, kind, phase, agent, self.cache, self.bank)

    def _view(self, rec, at_phase: bool = True, cell=None, kind: str = "") -> KeyView | None:
        if rec is None:
            return None
        from apexzero.ocp.window import WindowResult

        status = getattr(rec, "status", "verified")
        traj = None
        if getattr(rec, "x", None) is not None and getattr(rec, "a", None) is not None:
            from apexzero.interp.answer_key import key_ctx

            spec = dict(getattr(rec, "spec", {}) or {})
            res = WindowResult(np.asarray(rec.x), np.asarray(rec.a), float(rec.J), True,
                               status, 0, 0.0, 0, meta={"spec": spec})
            traj = measures.trajectory_from_window(res, self.bank.track(rec.stim_id),
                                                   ctx=key_ctx())
        elif getattr(rec, "result", None) is not None:
            traj = measures.trajectory_from_socp(rec.result)
        metrics = _metrics_from(getattr(rec, "measures", None))
        if self.recompute and traj is not None and status == "verified":
            stim = self.bank.stimuli[rec.stim_id]
            W = int(getattr(rec, "W", 0) or 0)
            null = None
            if traj.dt is not None and cell is not None and kind not in ("null", "socp") \
                    and "G" not in cell.tiers:
                nk = self.null(cell)
                null = nk.traj if nk is not None and nk.status == "verified" else None
            marks = _marks(stim, self.bank, float(traj.x[0, 0]), W)
            metrics = measures.response_metrics(traj, marks, null, self.tables)
        return KeyView(traj, metrics, at_phase, status,
                       bool(getattr(rec, "multi_basin", False)),
                       dict(getattr(rec, "tau", {}) or {}),
                       dict(getattr(rec, "active", {}) or {}),
                       float(getattr(rec, "J", np.nan)))

    def _get(self, cell, kind: str, phase: int = 0, agent: str = "", at_phase: bool = True):
        k = (cell.cell_id, kind, phase, agent, at_phase)
        if k not in self._memo:
            self._memo[k] = self._view(self._record(cell, kind, phase, agent), at_phase, cell,
                                       kind)
        return self._memo[k]

    def nominal(self, cell, phase: int) -> KeyView | None:
        kind = "null" if "G" in cell.tiers else "nominal"
        if phase > 0 and kind == "nominal":
            v = self._get(cell, "phase", phase)
            if v is not None and v.status == "verified":
                return v
        return self._get(cell, kind, 0, at_phase=phase == 0)

    def backoff(self, cell) -> KeyView | None:
        return self._get(cell, "backoff")

    def backoff_at(self, cell, phase: int) -> KeyView | None:
        """A backed-off key solved at phase k (looked up, never solved; 6.1 solves backed-off
        keys at u = 0 only, so this is None unless one was added)."""
        return self._get(cell, "backoff", phase) if phase else self.backoff(cell)

    def anchored(self, cell, agent: str, timesteps: int) -> KeyView | None:
        return self._get(cell, "anchored", 0, self.anchor_id(agent, timesteps))

    def null(self, cell) -> KeyView | None:
        c = self.cells.get(cell.null_cell)
        return None if c is None else self._get(c, "null")

    def socp(self, cell) -> KeyView | None:
        return self._get(cell, "socp")

    def no_return(self, cell, agent, timesteps, phase, sample, traj, key_traj):
        recs = self.viability.get((agent, int(timesteps), cell.cell_id, int(phase), int(sample)))
        if not recs or key_traj is None:
            return None
        from apexzero.interp import answer_key

        return answer_key.no_return(recs, traj, key_traj)


def _strip_width(bank, stim_id: int, s: float) -> float:
    """Total width w_left + w_right at the knot nearest s (straights and real roads)."""
    wl, wr = np.asarray(bank.w_left[stim_id]), np.asarray(bank.w_right[stim_id])
    i = int(np.clip(round(s / getattr(bank, "ds", 2.0)), 0, wl.size - 1))
    return float(wl[i] + wr[i])


def key_table(cells: list, bank, keys: KeySource, extra: dict[int, dict] | None = None
              ) -> Table:
    """KEY_COLUMNS for every cell: geometry from the bank, the u = 0 nominal, backed-off and
    s-OCP keys' values from ``keys``, and the braking/flat class (2.3: a sustained brake and
    v_min* < v0 - tau, tau the key's v_min tolerance, floor 0.5 km/h). ``extra`` adds or
    overrides columns per cell_id (e.g. v_min_qss). A budget rule 2 cut cell has key_status
    'no_key' and no key values (R7)."""
    floor = MULTI_BASIN_FLOORS["v_min"]
    keys = _NoKeyFilter(keys)
    rows = []
    for c in cells:
        st = bank.stimuli[c.stim_id]
        sp, geo, lm = st.spec, st.geometry, st.landmarks
        nom, bo, so = keys.nominal(c, 0), keys.backoff(c), keys.socp(c)
        mm = nom.metrics if nom is not None else None
        status = NO_KEY_FLAG if has_no_key(c) else (nom.status if nom is not None else "missing")
        tau_v = float((nom.tau or {}).get("v_min", floor)) if nom is not None else floor
        v_min_star = float(_attr(mm, "v_min"))
        has_b = bool(_attr(mm, "has_brake", False))
        sm = so.metrics if so is not None else None
        v_min_socp = float(_attr(sm, "v_min"))
        active = (nom.active or {}) if nom is not None else {}
        src = st.source or {}
        swap = src.get("swap_with", -1)
        row = dict(
            cell=int(c.cell_id), stim=int(c.stim_id), pair=int(c.pair_id), name=st.name,
            geometry=geometry_name(st.name), tiers=",".join(c.tiers), approach=c.approach,
            v0=float(c.v0), D=float(c.D) if c.D is not None else np.nan,
            W=int(c.W) if c.W is not None else -1, n_phases=int(c.n_phases),
            direction=int(st.direction), s_c0=float(lm.s_c0), s_a0=float(lm.s_a0),
            s_apex=float(lm.s_apex), s_c1=float(lm.s_c1),
            R_nom=float(sp.R) if sp is not None else np.nan,
            theta_deg=float(sp.theta_deg) if sp is not None else float(geo.theta_seg_deg),
            width=float(sp.width) if sp is not None else _strip_width(bank, c.stim_id, lm.s_c0),
            ramp_kind=str(sp.ramp_kind) if sp is not None else str(st.kind),
            no_cap=bool(sp.no_cap) if sp is not None else False,
            R_seg=float(geo.R_seg), R_eff=float(geo.R_eff),
            corner=int(src.get("corner_index", -1)), condition=str(src.get("condition", "")),
            swap_j=int(swap) if swap is not None else -1,
            key_status=status, J_star=float(nom.J) if nom is not None else np.nan,
            v_e_star=float(_attr(mm, "v_e")), v_min_star=v_min_star,
            d_b_star=float(_attr(mm, "d_b")), v_b_star=float(_attr(mm, "v_b")),
            has_brake_star=has_b, tau_vmin=tau_v,
            multi_basin=bool(nom.multi_basin) if nom is not None else False,
            lock_limited=bool(active.get("lock_limited", False)),
            slip_limited=bool(active.get("slip_limited", False)),
            braking=bool(status == "verified" and has_b and v_min_star < c.v0 - tau_v),
            v_e_star_bo=float(_attr(bo.metrics if bo else None, "v_e")),
            v_min_star_bo=float(_attr(bo.metrics if bo else None, "v_min")),
            v_e_socp=float(_attr(sm, "v_e")), v_min_socp=v_min_socp,
            d_b_socp=float(_attr(sm, "d_b")), v_b_socp=float(_attr(sm, "v_b")),
            braking_socp=bool(_attr(sm, "has_brake", False) and v_min_socp < c.v0 - floor),
            socp_cell=int(c.cell_id) if sm is not None else -1,
            tau_ve=float((nom.tau or {}).get("v_e", MULTI_BASIN_FLOORS["v_e"]))
            if nom is not None else np.nan,
            tau_db=float((nom.tau or {}).get("d_b", MULTI_BASIN_FLOORS["d_b"]))
            if nom is not None else np.nan,
            v_min_qss=np.nan,
        )
        if extra and c.cell_id in extra:
            row.update(extra[c.cell_id])
        rows.append(row)
    _share_tier_d_socp(rows, cells)
    return _to_table(rows, KEY_COLUMNS)


_SOCP_COLS = ("v_e_socp", "v_min_socp", "d_b_socp", "v_b_socp", "braking_socp", "socp_cell")


def _share_tier_d_socp(rows: list[dict], cells: list) -> None:
    """Tier D's min-time s-OCP keys are solved once per (corner, condition), from S1's state
    (protocol 3.3: condition (a) plus the six doses, at most 140): every runner's cell of
    that strip takes the s-OCP columns of the cell run from S1's state (the one holding a
    solved s-OCP), so S2's and S3's conditions get the same braking class and v_min*."""
    by_stim: dict[int, list[int]] = {}
    for i, c in enumerate(cells):
        if "D" in c.tiers:
            by_stim.setdefault(int(c.stim_id), []).append(i)
    for idx in by_stim.values():
        have = [i for i in idx if np.isfinite(rows[i]["v_min_socp"])]
        if not have:
            continue
        s1 = [i for i in have if "S1" in (cells[i].runners or ("S1",))]
        ref = rows[(s1 or have)[0]]
        for i in idx:
            if not np.isfinite(rows[i]["v_min_socp"]):
                rows[i].update({k: ref[k] for k in _SOCP_COLS})


# ------------------------------------------------------------------------- tables


def _marks(stim, bank, s0: float, W: int):
    return measures.Marks(stim.landmarks, int(stim.direction), float(s0), int(W),
                          np.asarray(bank.w_left[stim.stim_id]),
                          np.asarray(bank.w_right[stim.stim_id]),
                          triangle=bool(stim.spec.is_triangle) if stim.spec is not None else False)


def _null_runs(det: dict | None, cells: dict[int, Any]) -> dict[int, Any]:
    """The driver's own tier G runs (phase 0, unmirrored) by null cell id."""
    out: dict[int, Any] = {}
    if not det or "cell" not in det:
        return out
    null_ids = {c.null_cell for c in cells.values() if c.null_cell >= 0}
    for i, cid in enumerate(np.asarray(det["cell"])):
        cid = int(cid)
        if cid in null_ids and cid not in out and int(det["phase"][i]) == 0 \
                and int(det["steer_sign"][i]) > 0:
            out[cid] = measures.trajectory_from_rollout(det, i, "det")
    return out


def v7_speeds_failed(validation: dict | None) -> bool:
    """Protocol 6.5's V7 verdict from the manifest's ``validation`` dict: True when V7 was
    run, failed, and lists the cells that failed (``details['failed']``). V7's pass
    criterion is on speeds only (phases u_1-u_3 within 0.5 km/h of u_0; onsets are
    reported), so a listed failure is a failure on speeds, whose remedy is phase-matched
    keys for every braking cell and speed errors judged against them (AD23). False before
    validation has run, and when V7 could not be judged (no cell with phase keys, or an
    error: passed False with no failed cell), which is a validation fault to fix, not a
    measured phase effect."""
    v = (validation or {}).get("V7")
    if not (isinstance(v, dict) and "passed" in v and not bool(v["passed"])):
        return False
    failed = (v.get("details") or {}).get("failed") or []
    return bool(len(failed))


def run_table(agent: str, timesteps: int, rollouts: dict | tuple, cells: list, bank,
              keys: KeySource, tables, nulls: dict | None = None, check_design: bool = True,
              speed_at_phase: bool = False) -> Table:
    """RUN_COLUMNS for one snapshot's rollout file: per-run ``response_metrics``, key
    errors and crash types. e_ve and e_vmin are against the nominal u = 0 key (protocol
    2.2: the key solved from x0 at u = 0); the onset errors dd_b and dd_resp, regret and
    corner time against the key solved at the run's phase where one exists (protocol 6.1,
    phase-matched keys back Δd_b and P2b), else the u = 0 key; backed-off and re-anchored
    band errors; corner time also against the pinned s-OCP and its null (protocol 5).

    ``speed_at_phase`` (AD23, protocol 6.5's V7 remedy; pass ``v7_speeds_failed`` of the
    manifest's validation): e_ve, e_vmin and e_vmin^line are judged against the onset
    errors' key too (the key at the run's phase when one exists, else the u = 0 key), and
    e_ve_bo against a backed-off key at the run's phase when one exists (else the u = 0
    one). Columns speed_at_phase and bo_at_phase record the choice.

    ``rollouts`` is ``rollout.load_rollouts``'s groups (or its (groups, meta) tuple).
    ``nulls`` maps a null cell id to this driver's tier G trajectory; by default it is
    taken from the det group itself (phase 0, unmirrored run on that cell).
    ``check_design``: every run's recorded s0 and W must be its cell's (u_k phase start and
    window); a run made on another design (a changed D, P or W) raises instead of being
    paired with the wrong keys."""
    groups = rollouts[0] if isinstance(rollouts, tuple) else rollouts
    by_id = {c.cell_id: c for c in cells}
    keys = _NoKeyFilter(keys)          # R7: budget-cut cells are rolled out, never keyed
    if check_design:
        for g in MODES:
            arr = groups.get(g)
            if arr and "cell" in arr:
                check_runs_match_cells(arr, by_id)
    if nulls is None:
        nulls = _null_runs(groups.get("det"), by_id)
    rows = []
    for g in MODES:
        arr = groups.get(g)
        if not arr or "cell" not in arr:
            continue
        for i in range(len(arr["cell"])):
            rows.append(_run_row(agent, timesteps, g, arr, i, by_id, bank, keys, tables, nulls,
                                 bool(speed_at_phase)))
    return _to_table(rows, RUN_COLUMNS)


DESIGN_S0_TOL = 1e-3          # m: a run's s0 vs its cell's phase start (float32 states)


def check_runs_match_cells(arr: dict, by_id: dict) -> None:
    """Raise if a run's recorded W or s0 is not its cell's current W or phase start
    (hygiene: rollouts and keys must come from the same design)."""
    bad = []
    for i in range(len(arr["cell"])):
        c = by_id.get(int(arr["cell"][i]))
        if c is None:
            bad.append((int(arr["cell"][i]), "cell not in the design"))
            continue
        W = int(arr["W"][i]) if "W" in arr else None
        if W is not None and c.W is not None and W != int(c.W):
            bad.append((c.cell_id, f"W {W} != {c.W}"))
            continue
        if "s0" in arr and "phase" in arr:
            try:
                s0 = float(c.initial_state(int(arr["phase"][i]))[0])
            except ValueError:
                continue
            if abs(float(arr["s0"][i]) - s0) > DESIGN_S0_TOL:
                bad.append((c.cell_id, f"s0 {float(arr['s0'][i]):.3f} != {s0:.3f}"))
        if len(bad) >= 5:
            break
    if bad:
        raise ValueError("rollouts do not match the current design (re-run them or restore "
                         f"the design they were made on): {bad}")


def _run_row(agent, timesteps, g, arr, i, by_id, bank, keys, tables, nulls,
             speed_at_phase: bool = False) -> dict:
    cell = by_id[int(arr["cell"][i])]
    stim = bank.stimuli[cell.stim_id]
    traj = measures.trajectory_from_rollout(arr, i, g)
    marks = _marks(stim, bank, float(arr["s0"][i]), int(arr["W"][i]))
    null_traj = nulls.get(cell.null_cell)
    m = measures.response_metrics(traj, marks, null_traj, tables)
    phase, sample = int(arr["phase"][i]), int(arr["sample"][i])
    nom = keys.nominal(cell, phase)
    nom0 = keys.nominal(cell, 0) if phase else nom
    ok = nom is not None and nom.status == "verified" and nom.metrics is not None
    ok0 = nom0 is not None and nom0.status == "verified" and nom0.metrics is not None
    ke = measures.KeyErrors()
    if ok:
        nk = keys.null(cell)
        so = keys.socp(cell)
        null_cell = getattr(keys, "cells", {}).get(cell.null_cell)
        nso = keys.socp(null_cell) if null_cell is not None else None
        ke = measures.key_errors(m, nom.metrics, traj, nom.traj, marks, null_traj,
                                 nk.traj if nk is not None else None, nom.at_phase,
                                 socp=so.traj if so is not None and so.status == "verified"
                                 else None,
                                 null_socp=nso.traj if nso is not None else None)
    # Speed errors against the nominal u = 0 key (protocol 2.2); onsets stay phase-matched.
    # V7 failed on speeds (6.5, AD23): the speed errors use the onset errors' key too, i.e.
    # the key at the run's phase when one exists, else the u = 0 key.
    sk, ok_s = (nom, ok) if speed_at_phase and ok else (nom0, ok0)
    v_e_star_s = float(_attr(sk.metrics, "v_e")) if ok_s else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ke.e_ve = float(m.v_e) / v_e_star_s - 1.0 if ok_s else np.nan
        ke.e_vmin = float(m.v_min) / float(_attr(sk.metrics, "v_min")) - 1.0 if ok_s else np.nan
    ke.e_vmin_line_excess = (float(m.e_vmin_line) - float(_attr(sk.metrics, "e_vmin_line"))
                             if ok_s else np.nan)
    v_e = float(m.v_e)
    bo = keys.backoff(cell)
    bo_at_phase = phase == 0
    if speed_at_phase and phase:
        # A backed-off key at the run's phase where one exists, else the u = 0 one.
        at = getattr(keys, "backoff_at", None)
        bk = at(cell, phase) if at is not None else None
        if bk is not None and bk.status == "verified":
            bo, bo_at_phase = bk, True
    v_bo = float(_attr(bo.metrics, "v_e")) if bo is not None and bo.status == "verified" \
        else np.nan
    bo_at_phase = bool(bo_at_phase and np.isfinite(v_bo))
    ra = keys.anchored(cell, agent, timesteps)
    ra_key = bool(ra is not None and ra.status == "verified"
                  and np.isfinite(_attr(ra.metrics, "v_e")))
    if ra_key:
        v_ra = float(_attr(ra.metrics, "v_e"))
        e_ra = v_e / v_ra - 1.0
    else:
        v_ra = v_e_star_s
        e_ra = float(ke.e_ve)
    # The key behind this run's onset errors (phase-matched where one exists; P2b).
    k_ph = {f"{c}_star_k": float(_attr(nom.metrics, c)) if ok else np.nan
            for c in ("v_e", "v_b", "d_b")}
    ctype = ""
    if m.crashed:
        key_traj = nom.traj if ok else None
        nr = keys.no_return(cell, agent, timesteps, phase, sample, traj, key_traj)
        ctype = str(measures.crash_type(traj, marks, key_traj, nr))
    row = dict(agent=str(agent), timesteps=int(timesteps), cell=int(cell.cell_id),
               stim=int(cell.stim_id), pair=int(cell.pair_id), phase=phase, sample=sample,
               mode=g, steer_sign=int(arr["steer_sign"][i]), W=int(arr["W"][i]),
               s0=float(arr["s0"][i]))
    row.update({f: getattr(m, f) for f in METRIC_COLUMNS})
    row.update({f: getattr(ke, f) for f in KEY_ERROR_COLUMNS})
    row.update(e_ve_bo=v_e / v_bo - 1.0 if np.isfinite(v_bo) else np.nan, e_ve_ra=float(e_ra),
               key_at_phase=bool(nom.at_phase) if ok else False, crash_type=ctype,
               v_e_star_ra=float(v_ra), ra_key=ra_key, speed_at_phase=bool(speed_at_phase),
               bo_at_phase=bo_at_phase, **k_ph)
    return row


_RUN_F = ("s_crash", "v_e", "v_min", "e_ve", "e_ve_bo", "e_ve_ra", "v_e_star_ra", "e_vmin",
          "e_vmin_line", "e_vmin_line_excess", "d_b", "dd_b", "v_b", "d_resp", "dd_resp",
          "clip_frac_edges", "clip_frac_ego", "clip_frac_context", "dt_corner", "regret",
          "osc_index", "v_e_star_k", "v_b_star_k", "d_b_star_k")
_RUN_B = ("crashed", "completed", "has_brake", "forced_onset", "unsettled", "time_censored",
          "kappa_clipped", "ra_key", "speed_at_phase")


def _clip_other_runs(runs: Table) -> np.ndarray:
    """Runs with any clipping in the edge, ego or context channel groups (9.4)."""
    return ((np.nan_to_num(_fl(runs, "clip_frac_edges")) > 0)
            | (np.nan_to_num(_fl(runs, "clip_frac_ego")) > 0)
            | (np.nan_to_num(_fl(runs, "clip_frac_context")) > 0))


def pair_values(runs: Table, keys: Table, regimes: dict | None = None,
                band: float = BAND_EXPECTED, density: Table | None = None) -> Table:
    """PAIR_COLUMNS from a run table (9.1), exclusions marked (9.4).

    ``keys`` is the key table (KEY_COLUMNS); ``regimes`` maps an agent name to its regime
    boundaries (``support.RegimeTable`` or ``Regime``); agents without one get regime ''.
    ``density`` (optional) has columns pair, log_support, log_support_tightfast and,
    optionally, agent. r_extrap uses each (agent, snapshot, approach)'s own bulk fit."""
    n = _n(runs)
    if n == 0:
        return _to_table([], PAIR_COLUMNS)
    cols: dict[str, np.ndarray] = {c: _fl(runs, c) for c in _RUN_F}
    cols.update({c: _bo(runs, c) for c in _RUN_B})
    cols.update(agent=_st(runs, "agent"), mode=_st(runs, "mode"),
                crash_type=_st(runs, "crash_type"), timesteps=_it(runs, "timesteps", 0),
                pair=_it(runs, "pair"), cell=_it(runs, "cell"), phase=_it(runs, "phase", 0),
                steer_sign=_it(runs, "steer_sign", 1))
    kidx = {int(c): i for i, c in enumerate(_it(keys, "cell"))}
    kdir = _it(keys, "direction", 0)
    groups: dict[tuple, list[int]] = {}
    for i, k in enumerate(zip(cols["agent"], cols["timesteps"], cols["pair"], strict=True)):
        groups.setdefault(k, []).append(i)
    rows = [_pair_row(cols, np.asarray(idx), keys, kidx, kdir, regimes or {}, band)
            for _, idx in sorted(groups.items(), key=lambda kv: kv[0])]
    tab = _to_table(rows, PAIR_COLUMNS)
    tab["r_extrap"] = _r_extrap_column(tab)
    if density is not None and _n(density):
        _join_density(tab, density)
    return tab


def _pair_row(cols: dict, idx: np.ndarray, keys: Table, kidx: dict, kdir: np.ndarray,
              regimes: dict, b: float) -> dict:
    mode = cols["mode"][idx]
    det, nat, mat = idx[mode == "det"], idx[mode == "nat"], idx[mode == "mat"]
    cand = [int(c) for c in cols["cell"][det if det.size else idx] if int(c) in kidx]
    kr = None
    for c in sorted(set(cand)):
        kr = kidx[c] if kr is None else kr
        if int(kdir[kidx[c]]) > 0:
            kr = kidx[c]
            break

    def K(name, default=np.nan):
        if kr is None or name not in keys:
            return default
        v = keys[name][kr]
        return v.item() if hasattr(v, "item") else v

    agent = str(cols["agent"][idx[0]])
    nr = int(det.size)
    crashed, ctype = cols["crashed"][det], cols["crash_type"][det]
    s_c0 = float(K("s_c0"))
    ov = crashed & (ctype == "overspeed")
    pre = crashed & ~ov & (cols["s_crash"][det] < s_c0)
    post = crashed & ~ov & ~pre

    def speed(v: np.ndarray, lose_post: bool) -> np.ndarray:
        v = v.copy()
        v[ov] = np.inf
        v[pre] = np.nan
        if lose_post:
            v[post] = np.nan
        return v

    def pv(v: np.ndarray) -> tuple[float, int]:
        miss = int(np.isnan(v).sum())
        if nr == 0 or miss > _max_missing(nr):
            return float("nan"), miss
        return float(np.median(v[~np.isnan(v)])), miss

    v_e_star, v_e_star_bo = float(K("v_e_star")), float(K("v_e_star_bo"))
    ratio_bo = v_e_star / v_e_star_bo if v_e_star_bo > 0 else np.nan
    raw = {c: cols[c][det] for c in ("v_e", "e_ve", "e_ve_bo", "e_ve_ra", "v_min", "e_vmin",
                                     "e_vmin_line", "e_vmin_line_excess")}
    # Scaled to the re-anchored key on runs that have one; a run without one falls back to
    # its own e_ve_bo (the u = 0 backed-off key, or one at its phase: AD23), as e_ve_ra falls
    # back to e_ve. Scaling a fallback by v_e*/v_e*bo would move the backed-off edge by the
    # phase key's ratio v_e*_k / v_e* under the V7 remedy, unlike every other approach.
    raw["e_ve_bo_ra"] = np.where(cols["ra_key"][det], (1.0 + raw["e_ve_ra"]) * ratio_bo - 1.0,
                                 raw["e_ve_bo"])
    vals, missing = {}, {}
    for c, v in raw.items():
        vals[c], missing[c] = pv(speed(v, lose_post=c.startswith(("v_min", "e_vmin"))))
    approach, D = str(K("approach", "")), float(K("D"))
    uses_ra = approach == "A1" or (approach == "B" and np.isfinite(D) and abs(D - 600.0) < 1e-6)
    v_ra = _median(cols["v_e_star_ra"][det])

    def _star_k(c: str, fallback: float) -> float:
        v = _median(cols[c][det])
        return v if np.isfinite(v) else fallback

    # S and r use the key behind e_ve (AD5): the re-anchored key where the band uses it, else
    # the u = 0 key, or the runs' phase-matched keys when V7 failed on speeds (AD23).
    speed_k = bool(cols["speed_at_phase"][det].any()) if det.size else False
    v_e_star_s = _star_k("v_e_star_k", v_e_star) if speed_k else v_e_star
    band_star = v_ra if uses_ra and np.isfinite(v_ra) else v_e_star_s
    e_b = vals["e_ve_ra"] if uses_ra else vals["e_ve"]
    e_bo = vals["e_ve_bo_ra"] if uses_ra else vals["e_ve_bo"]
    # onsets (AD3, AD4)
    has_b, forced = cols["has_brake"][det], cols["forced_onset"][det]
    brake = has_b & ~forced
    need = _min_onsets(nr) if nr else 1

    def onset(c: str, mask: np.ndarray) -> float:
        v = cols[c][det][mask]
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size >= need else float("nan")

    finite_resp = np.isfinite(cols["d_resp"][det])
    # crash typing (det + matched)
    typed_idx = np.concatenate([det, mat])
    typed = cols["crashed"][typed_idx] & (cols["crash_type"][typed_idx] != "")
    tt = cols["crash_type"][typed_idx][typed]
    n_typed = int(typed.sum())
    rt = regimes.get(agent)
    R_seg = float(K("R_seg"))
    reg = regime_of(R_seg, rt.q1, rt.q10, getattr(rt, "r_clip_max", np.nan)) if rt else ""
    status = str(K("key_status", "missing"))
    if kr is None or status in ("missing", NO_KEY_FLAG):
        excluded = NO_KEY_FLAG
    elif status in EXCL_ALL:
        excluded = status
    elif bool(K("multi_basin", False)):
        excluded = "multi_basin"
    else:
        excluded = ""
    tiers = str(K("tiers", ""))
    tier = next((t for t in TIER_ORDER if t in tiers.split(",")), tiers.split(",")[0])
    dtc = cols["dt_corner"][det][~cols["time_censored"][det]]
    clip_other = bool(_clip_other_runs({c: cols[c][det] for c in (
        "clip_frac_edges", "clip_frac_ego", "clip_frac_context")}).any())
    v_e_obs = cols["v_e"][det][~pre]

    return dict(
        agent=agent, timesteps=int(cols["timesteps"][idx[0]]), pair=int(cols["pair"][idx[0]]),
        tier=tier, tiers=tiers, approach=approach, v0=float(K("v0")), D=D, W=int(K("W", -1)),
        geometry=str(K("geometry", "")), R_nom=float(K("R_nom")), theta_deg=float(K("theta_deg")),
        width=float(K("width")), ramp_kind=str(K("ramp_kind", "")),
        no_cap=bool(K("no_cap", False)), R_seg=R_seg, R_eff=float(K("R_eff")),
        v_min_star=float(K("v_min_star")), v_e_star=v_e_star, v_e_star_bo=v_e_star_bo,
        v_e_star_ra=v_ra, v_e_star_band=float(band_star), speed_at_phase=speed_k,
        d_b_star=float(K("d_b_star")),
        v_b_star=float(K("v_b_star")), braking=bool(K("braking", False)),
        v_e_star_k=_star_k("v_e_star_k", v_e_star), v_b_star_k=_star_k("v_b_star_k",
                                                                      float(K("v_b_star"))),
        d_b_star_k=_star_k("d_b_star_k", float(K("d_b_star"))),
        key_status=status, multi_basin=bool(K("multi_basin", False)),
        lock_limited=bool(K("lock_limited", False)), slip_limited=bool(K("slip_limited", False)),
        v_min_qss=float(K("v_min_qss")),
        regime=reg, kappa_clipped=bool(cols["kappa_clipped"][det].any()), clip_other=clip_other,
        log_support=np.nan, log_support_tightfast=np.nan,
        n_runs=nr, n_missing=missing["v_e"], defined=bool(nr > 0 and missing["v_e"]
                                                          <= _max_missing(nr)),
        uses_ra=bool(uses_ra), ra_fallback=int((~cols["ra_key"][det]).sum()) if uses_ra else 0,
        v_e=vals["v_e"], v_e_obs=_median(v_e_obs), e_ve=vals["e_ve"],
        e_ve_bo=vals["e_ve_bo"], e_ve_ra=vals["e_ve_ra"], e_ve_bo_ra=vals["e_ve_bo_ra"],
        side=band_side(e_b, e_bo, b), v_min=vals["v_min"], e_vmin=vals["e_vmin"],
        e_vmin_line=vals["e_vmin_line"], e_vmin_line_excess=vals["e_vmin_line_excess"],
        r_extrap=np.nan,
        n_brake=int(has_b.sum()), d_b=onset("d_b", brake), dd_b=onset("dd_b", brake),
        v_b=onset("v_b", brake), d_resp=onset("d_resp", finite_resp),
        dd_resp=onset("dd_resp", finite_resp),
        forced_frac=_frac(forced[has_b]), unsettled_frac=_frac(cols["unsettled"][det]),
        time_censored_frac=_frac(cols["time_censored"][det]),
        det_crashes=int(crashed.sum()), det_crash_frac=_frac(crashed), det_overspeed=int(ov.sum()),
        det_complete=int(cols["completed"][det].sum()),
        nat_k=int(cols["crashed"][nat].sum()), nat_n=int(nat.size),
        mat_k=int(cols["crashed"][mat].sum()), mat_n=int(mat.size),
        typed_crashes=n_typed, typed_overspeed=int((tt == "overspeed").sum()),
        typed_overbrake=int((tt == "overbrake").sum()),
        typed_line_steer=int((tt == "line_steer").sum()),
        typed_other=int(n_typed - np.isin(tt, ("overspeed", "overbrake", "line_steer")).sum()),
        overspeed_frac=float((tt == "overspeed").sum() / n_typed) if n_typed else np.nan,
        dt_corner=_median(dtc), regret=_median(cols["regret"][det]),
        osc_index=_median(cols["osc_index"][det]), excluded=excluded,
    )


def _r_extrap_column(tab: Table) -> np.ndarray:
    r = np.full(_n(tab), np.nan)
    agent, ts, ap = _st(tab, "agent"), _it(tab, "timesteps", 0), _st(tab, "approach")
    for key in set(zip(agent, ts, ap, strict=True)):
        g = (agent == key[0]) & (ts == key[1]) & (ap == key[2])
        bulk = g & _mask(tab, approach=key[2], regimes=("bulk",))
        if bulk.sum() < 2:
            continue
        fit = sensitivity(select(tab, bulk), x="v_e_star_band")
        if np.isfinite(fit.S):
            r[g] = extrapolation_residual(select(tab, g), fit, x="v_e_star_band")
    return r


def _join_density(tab: Table, density: Table) -> None:
    pair, agent = _it(density, "pair"), _st(density, "agent")
    has_agent = "agent" in density
    look = {}
    for i in range(_n(density)):
        look[(agent[i] if has_agent else "", int(pair[i]))] = i
    ls, lt = _fl(density, "log_support"), _fl(density, "log_support_tightfast")
    for j in range(_n(tab)):
        i = look.get((str(tab["agent"][j]) if has_agent else "", int(tab["pair"][j])))
        if i is not None:
            tab["log_support"][j] = ls[i]
            tab["log_support_tightfast"][j] = lt[i]


def cond_table(runs: Table, keys: Table, d_ood: dict | None = None) -> Table:
    """COND_COLUMNS from the deterministic tier D runs (keys rows with corner >= 0).

    ``completes`` counts completed phase runs (complete: >= 3 of 4, scaled to the count);
    v_min_region is the run's v_min, because a swap B strip keeps the replaced corner's
    landmarks, so v_min's window [s_c0 - 50, s_c1 + 50] is the protocol's region.
    ``braking`` is the s-OCP's braking class in condition (a) of that corner and v_min* the
    s-OCP's v_min in that condition, both from the key rows that hold a solved s-OCP (the
    cell run from S1's state; protocol 3.3 solves one set), never from whichever runner's
    row comes last. ``d_ood`` maps (agent, cell_id) to the run's median normalised NN
    distance (controls.d_ood)."""
    kidx = {int(c): i for i, c in enumerate(_it(keys, "cell"))}
    corner, cond = _it(keys, "corner"), _st(keys, "condition")
    det = _st(runs, "mode") == "det"
    agent, ts, cell = _st(runs, "agent"), _it(runs, "timesteps", 0), _it(runs, "cell")
    completed, v_min, v_peak = _bo(runs, "completed"), _fl(runs, "v_min"), _fl(runs, "v_peak")
    vms, bs = _fl(keys, "v_min_socp"), _bo(keys, "braking_socp")
    brake_a: dict[int, bool] = {}
    vstar: dict[tuple[int, str], float] = {}
    for i in range(_n(keys)):
        if corner[i] < 0 or not np.isfinite(vms[i]):
            continue                      # no solved s-OCP on this runner's row
        vstar.setdefault((int(corner[i]), str(cond[i])), float(vms[i]))
        if cond[i] == COND_A:
            brake_a[int(corner[i])] = brake_a.get(int(corner[i]), False) or bool(bs[i])
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(det):
        k = kidx.get(int(cell[i]))
        if k is None or corner[k] < 0:
            continue
        groups.setdefault((agent[i], int(ts[i]), int(corner[k]), str(cond[k])), []).append(i)
    rows = []
    for (a, t, c, cn), idx in sorted(groups.items()):
        idx = np.asarray(idx)
        k = kidx[int(cell[idx[0]])]
        nr = idx.size
        dd = [d_ood.get((a, int(cell[i])), np.nan) for i in idx] if d_ood else [np.nan]
        rows.append(dict(agent=a, timesteps=t, corner=c, condition=cn,
                         braking=brake_a.get(c, False), completes=int(completed[idx].sum()),
                         complete=bool(completed[idx].sum() >= math.ceil(3 * nr / 4)),
                         v_min=_median(v_min[idx]), v_peak=_median(v_peak[idx]),
                         v_min_region=_median(v_min[idx]), d_ood=_median(dd),
                         v_min_star=vstar.get((c, cn), float("nan")),
                         swap_j=int(_it(keys, "swap_j")[k])))
    return _to_table(rows, COND_COLUMNS)


def band_side(e_ve: float, e_ve_bo: float, b: float) -> str:
    """'in' if e_ve <= +b and e_ve_bo >= -b; 'over' if e_ve > +b; 'under' if e_ve_bo < -b
    (overspeed takes precedence); 'undefined' for NaN (protocol 2.2)."""
    if e_ve is None or np.isnan(e_ve):
        return "undefined"
    if e_ve > b:
        return "over"
    if e_ve_bo is None or np.isnan(e_ve_bo):
        return "undefined"
    if e_ve_bo < -b:
        return "under"
    return "in"


@dataclass
class SlopeFit:
    """log v_e = a_theta + S log v_e* (fixed effects for theta), protocol 2.2."""

    S: float
    intercepts: dict          # theta -> a_theta
    estimable: bool           # pooled within-theta range of log v_e* >= 0.15
    log_range: float
    loo: dict                 # left-out R -> S
    n: int = 0


def _fe_slope(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[float, dict, float]:
    """Within-group OLS slope, intercepts per group and the pooled within-group x range."""
    if x.size < 2:
        return float("nan"), {}, 0.0
    xt, yt, means = x.copy(), y.copy(), {}
    for gv in np.unique(g):
        m = g == gv
        mx, my = float(x[m].mean()), float(y[m].mean())
        xt[m] -= mx
        yt[m] -= my
        means[float(gv) if g.dtype.kind in "fiu" else str(gv)] = (mx, my)
    sxx = float((xt * xt).sum())
    S = float((xt * yt).sum() / sxx) if sxx > 1e-15 else float("nan")
    icpt = {gv: my - S * mx for gv, (mx, my) in means.items()} if np.isfinite(S) else {}
    return S, icpt, float(xt.max() - xt.min())


def sensitivity(pairs: Table, x: str = "v_e_star", y: str = "v_e", by: str = "theta_deg",
                loo_by: str = "R_nom") -> SlopeFit:
    """S with leave-one-R-out values; overspeed (+inf) pairs excluded from the fit."""
    with np.errstate(divide="ignore", invalid="ignore"):
        X, Y = np.log(_fl(pairs, x)), np.log(_fl(pairs, y))
    th, R = _fl(pairs, by), _fl(pairs, loo_by)
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(th)
    X, Y, th, R = X[ok], Y[ok], th[ok], R[ok]
    S, icpt, rng = _fe_slope(X, Y, th)
    loo = {}
    for r in np.unique(R[np.isfinite(R)]):
        keep = R != r
        loo[float(r)] = _fe_slope(X[keep], Y[keep], th[keep])[0]
    return SlopeFit(S, icpt, bool(rng >= S_MIN_LOG_RANGE), rng, loo, int(ok.sum()))


def extrapolation_residual(pairs: Table, fit: SlopeFit, x: str = "v_e_star",
                           y: str = "v_e") -> np.ndarray:
    """r = log v_e - (a_theta + S_bulk log v_e*) per pair (P1b); +inf for overspeed
    crashes, NaN where theta has no bulk intercept."""
    a = np.array([fit.intercepts.get(float(t), np.nan) for t in _fl(pairs, "theta_deg")])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(_fl(pairs, y)) - (a + fit.S * np.log(_fl(pairs, x)))


# ------------------------------------------------------------------------- inference


@dataclass
class RateInterval:
    p_hat: float
    lo: float
    hi: float
    rho: float                # beta-binomial overdispersion (0: Clopper-Pearson used)
    method: str               # 'beta-binomial profile' | 'clopper-pearson' | 'none'


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact two-sided interval for a binomial proportion."""
    k, n = int(k), int(n)
    if n <= 0:
        return float("nan"), float("nan")
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


_EPS = 1e-12
_RHO_GRID = np.concatenate([[0.0], np.geomspace(1e-6, 0.999, 60)])


def _bb_tabs(k: np.ndarray, n: np.ndarray):
    """Counts A_j, B_j, C_j of pairs with k > j, n - k > j and n > j (j < max n): the
    beta-binomial log-likelihood is then sum_j A_j log(mu + j g) + B_j log(1 - mu + j g)
    - C_j log(1 + j g), with g = rho / (1 - rho) (exact, and the binomial at g = 0)."""
    j = np.arange(int(n.max()), dtype=float)
    A = (k[None, :] > j[:, None]).sum(1).astype(float)
    B = ((n - k)[None, :] > j[:, None]).sum(1).astype(float)
    C = (n[None, :] > j[:, None]).sum(1).astype(float)
    return j, A, B, C


def _bb_ll(mu: float, rho, tabs) -> np.ndarray:
    j, A, B, C = tabs
    rho = np.atleast_1d(np.asarray(rho, float))
    g = (rho / (1.0 - rho))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ta = np.where(A > 0, A * np.log(mu + j * g), 0.0)
        tb = np.where(B > 0, B * np.log(1.0 - mu + j * g), 0.0)
        tc = np.where(C > 0, C * np.log1p(j * g), 0.0)
    return (ta + tb - tc).sum(1)


def _max_over_rho(f: Callable[[np.ndarray], np.ndarray]) -> tuple[float, float]:
    """max over rho in [0, 0.999] of a vectorised f: grid, then bounded refinement."""
    ll = f(_RHO_GRID)
    i = int(np.nanargmax(ll))
    best_ll, best_rho = float(ll[i]), float(_RHO_GRID[i])
    lo, hi = _RHO_GRID[max(i - 1, 0)], _RHO_GRID[min(i + 1, _RHO_GRID.size - 1)]
    if hi > lo:
        res = optimize.minimize_scalar(lambda r: -float(f(np.array([r]))[0]), bounds=(lo, hi),
                                       method="bounded", options={"xatol": 1e-10})
        if -res.fun > best_ll:
            best_ll, best_rho = float(-res.fun), float(res.x)
    return best_ll, best_rho


def _bb_profile(mu: float, tabs) -> tuple[float, float]:
    return _max_over_rho(lambda r: _bb_ll(mu, r, tabs))


def _bb_best_mu(rho: float, tabs) -> tuple[float, float]:
    res = optimize.minimize_scalar(lambda m: -float(_bb_ll(m, rho, tabs)[0]),
                                   bounds=(_EPS, 1 - _EPS), method="bounded",
                                   options={"xatol": 1e-12})
    return float(res.x), float(-res.fun)


def beta_binomial(k: np.ndarray, n: np.ndarray, level: float = 0.99) -> RateInterval:
    """ML beta-binomial across pairs with a profile-likelihood interval for the mean rate;
    Clopper-Pearson on pooled counts if the overdispersion estimate is 0 (9.2). All-zero
    (or all-crash) data leave rho unidentified; the ML set then contains rho = 0, so
    Clopper-Pearson is used."""
    k, n = np.asarray(k, np.int64), np.asarray(n, np.int64)
    keep = n > 0
    k, n = k[keep], n[keep]
    if n.size == 0:
        return RateInterval(np.nan, np.nan, np.nan, np.nan, "none")
    K, N = int(k.sum()), int(n.sum())
    if K == 0 or K == N:
        lo, hi = clopper_pearson(K, N, level)
        return RateInterval(K / N, lo, hi, 0.0, "clopper-pearson")
    tabs = _bb_tabs(k, n)

    def outer(rhos: np.ndarray) -> np.ndarray:
        return np.array([_bb_best_mu(float(r), tabs)[1] for r in rhos])

    ll_hat, rho_hat = _max_over_rho(outer)
    ll0 = float(_bb_ll(K / N, 0.0, tabs)[0])
    if ll0 >= ll_hat - 1e-9 or rho_hat < 1e-6:
        lo, hi = clopper_pearson(K, N, level)
        return RateInterval(K / N, lo, hi, 0.0, "clopper-pearson")
    mu_hat = _bb_best_mu(rho_hat, tabs)[0]
    crit = float(stats.chi2.ppf(level, 1))

    def dev(mu: float) -> float:
        return 2.0 * (ll_hat - _bb_profile(mu, tabs)[0]) - crit

    lo = 0.0 if dev(_EPS) <= 0 else optimize.brentq(dev, _EPS, mu_hat, xtol=1e-10)
    hi = 1.0 if dev(1 - _EPS) <= 0 else optimize.brentq(dev, mu_hat, 1 - _EPS, xtol=1e-10)
    return RateInterval(mu_hat, float(lo), float(hi), rho_hat, "beta-binomial profile")


def firth_logistic(x: np.ndarray, y: np.ndarray, n: np.ndarray | None = None,
                   max_iter: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Firth-penalised logistic regression (coef [2], se [2]) of y on x (9.3, R50).

    ``y`` holds successes out of ``n`` trials (n = 1: Bernoulli). Maximises
    l(beta) + 0.5 log|I(beta)| with I = X'WX, W = n p (1 - p), by Newton steps on the
    modified score X'(y - n p + h (1/2 - p)) with step halving; finite under separation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = np.ones_like(y) if n is None else np.asarray(n, float)
    X = np.column_stack([np.ones_like(x), x])

    def info(beta):
        p = expit(X @ beta)
        w = n * p * (1 - p)
        return p, w, X.T @ (w[:, None] * X)

    def pll(beta) -> float:
        eta = X @ beta
        _, _, I = info(beta)
        sign, logdet = np.linalg.slogdet(I)
        if sign <= 0:
            return -np.inf
        return float((y * log_expit(eta) + (n - y) * log_expit(-eta)).sum() + 0.5 * logdet)

    beta = np.zeros(2)
    for _ in range(max_iter):
        p, w, I = info(beta)
        Iinv = np.linalg.pinv(I)
        h = w * np.einsum("ij,jk,ik->i", X, Iinv, X)
        step = Iinv @ (X.T @ (y - n * p + h * (0.5 - p)))
        cur, t = pll(beta), 1.0
        while t > 1e-10 and pll(beta + t * step) < cur - 1e-12:
            t *= 0.5
        beta = beta + t * step
        if np.max(np.abs(t * step)) < tol:
            break
    se = np.sqrt(np.diag(np.linalg.pinv(info(beta)[2])))
    return beta, se


def r50(R_seg: np.ndarray, p_crash: np.ndarray, n: np.ndarray | None = None) -> dict:
    """R50 as the empirical bracket between grid values (primary) and from the Firth fit on
    log R_seg (secondary).

    The bracket is (largest R with P_crash >= 0.5, next grid R above it); 'below_grid' when
    no R reaches 0.5 (bracket (nan, R_min)), 'above_grid' when every R does ((R_max, nan)).
    ``monotone`` says whether the crash set is exactly the R values below the bracket. The
    Firth fit needs ``n`` (trials per R); P_crash * n are the successes."""
    R, p = np.asarray(R_seg, float), np.asarray(p_crash, float)
    ok = np.isfinite(R) & np.isfinite(p)
    R, p = R[ok], p[ok]
    nn = None if n is None else np.asarray(n, float)[ok]
    if R.size == 0:
        return {"bracket": (np.nan, np.nan), "position": "empty", "monotone": False,
                "firth_R50": np.nan, "coef": [np.nan, np.nan], "se": [np.nan, np.nan]}
    levels = np.unique(R)
    if nn is None:
        pl = np.array([p[R == r].mean() for r in levels])
    else:
        pl = np.array([(p[R == r] * nn[R == r]).sum() / max(nn[R == r].sum(), 1e-300)
                       for r in levels])
    crash = pl >= 0.5
    if not crash.any():
        bracket, pos, last = (np.nan, float(levels[0])), "below_grid", -1
    elif crash.all():
        bracket, pos, last = (float(levels[-1]), np.nan), "above_grid", levels.size - 1
    else:
        last = int(np.flatnonzero(crash)[-1])
        if last == levels.size - 1:
            bracket, pos = (float(levels[-1]), np.nan), "above_grid"
        else:
            bracket, pos = (float(levels[last]), float(levels[last + 1])), "inside"
    monotone = bool(np.all(crash[: last + 1]) and not np.any(crash[last + 1:]))
    out = {"bracket": bracket, "position": pos, "monotone": monotone, "firth_R50": np.nan,
           "coef": [np.nan, np.nan], "se": [np.nan, np.nan]}
    if nn is not None and nn.sum() > 0 and np.unique(R).size >= 2:
        coef, se = firth_logistic(np.log(R), p * nn, nn)
        out.update(coef=coef.tolist(), se=se.tolist())
        z = -coef[0] / coef[1] if coef[1] != 0 else np.inf
        if abs(z) < 20.0:                  # a crossing beyond e^20 m is no crossing at all
            out["firth_R50"] = float(np.exp(z))
    return out


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b over the pairs with both values not NaN (+inf ranks above all)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        return float("nan")
    dx = (x[:, None] > x[None, :]).astype(int) - (x[:, None] < x[None, :]).astype(int)
    dy = (y[:, None] > y[None, :]).astype(int) - (y[:, None] < y[None, :]).astype(int)
    iu = np.triu_indices(x.size, 1)
    dx, dy = dx[iu], dy[iu]
    n0 = dx.size
    n1, n2 = int((dx == 0).sum()), int((dy == 0).sum())
    den = math.sqrt((n0 - n1) * (n0 - n2))
    return float((dx * dy).sum() / den) if den > 0 else float("nan")


def _rank_corr(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """Pearson correlation of rx with each row of ry ([m] or [P, m])."""
    rx = rx - rx.mean()
    ry = ry - ry.mean(axis=-1, keepdims=True)
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ry @ rx) / den


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return float("nan")
    return float(_rank_corr(stats.rankdata(x[ok]), stats.rankdata(y[ok])))


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = P6_PERMUTATIONS,
                         rng: np.random.Generator | None = None) -> tuple[float, float]:
    """(rho, one-sided p for rho <= observed) with ``io.stream_rng('permutation')``.
    Ranks are mid-ranks; +inf takes the top ranks (t_R = inf)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    rx, ry = stats.rankdata(x[ok]), stats.rankdata(y[ok])
    rho = float(_rank_corr(rx, ry))
    if not np.isfinite(rho):
        return rho, float("nan")
    rng = stream_rng("permutation") if rng is None else rng
    perms = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
    rp = _rank_corr(rx, perms)
    return rho, float((1 + np.sum(rp <= rho + 1e-12)) / (n_perm + 1))


def bootstrap_band(values: np.ndarray, groups: np.ndarray, n: int = BOOTSTRAP_N,
                   rng: np.random.Generator | None = None,
                   stat: Callable = np.nanmedian, level: float = 0.95
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Descriptive bands: resample pairs within each regime (9.2). Returns (lo, hi) of
    ``stat`` per group, in ``np.unique(groups)`` order."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    rng = stream_rng("bootstrap") if rng is None else rng
    lo, hi = [], []
    q = (100 * (1 - level) / 2, 100 * (1 + level) / 2)
    for g in np.unique(groups):
        v = values[groups == g]
        idx = rng.integers(0, v.size, size=(n, v.size))
        with np.errstate(invalid="ignore"):
            s = stat(v[idx], axis=1)
        a, z = (np.nan, np.nan) if np.isnan(s).all() else \
            np.nanpercentile(s, q, method="nearest")
        lo.append(float(a))
        hi.append(float(z))
    return np.asarray(lo), np.asarray(hi)


# ------------------------------------------------------------------------- P1


def _tracking(pairs: Table, mask_key: np.ndarray, mask_crash: np.ndarray, b: float,
              level: float = 0.99) -> dict:
    """P1a's criteria 1, 3 and 4 (and the side fractions) on a set of pairs."""
    side = sides(pairs, b)[mask_key]
    crash_pairs = int((_it(pairs, "det_crashes", 0)[mask_crash] > 0).sum())
    rate = beta_binomial(_it(pairs, "nat_k", 0)[mask_crash], _it(pairs, "nat_n", 0)[mask_crash],
                         level)
    frac_in = _frac(side == "in")
    return dict(n_pairs=int(mask_key.sum()), n_crash_set=int(mask_crash.sum()),
                frac_in=frac_in, frac_over=_frac(side == "over"),
                frac_under=_frac(side == "under"), frac_undefined=_frac(side == "undefined"),
                crash_pairs=crash_pairs, p_crash=rate,
                c1=bool(np.isfinite(frac_in) and frac_in >= P1A_IN_BAND),
                c3=crash_pairs <= P1A_MAX_DET_CRASH_PAIRS,
                c4=bool(np.isfinite(rate.hi) and rate.hi <= P1A_PCRASH_UPPER),
                unsettled_flag=_unsettled_flag(pairs, mask_crash))


def _unsettled_flag(pairs: Table, mask: np.ndarray) -> bool:
    """9.4: a regime is flagged if more than 20% of its deterministic runs are unsettled."""
    nr = _it(pairs, "n_runs", 0)[mask].astype(float)
    uf = np.nan_to_num(_fl(pairs, "unsettled_frac")[mask])
    return bool(nr.sum() > 0 and (uf * nr).sum() / nr.sum() > UNSETTLED_REGIME)


def _slope_summary(fit: SlopeFit) -> dict:
    return dict(S=fit.S, estimable=fit.estimable, log_range=fit.log_range, loo=fit.loo, n=fit.n)


def _slope_confirms(fit: SlopeFit) -> bool:
    loo = list(fit.loo.values())
    return bool(fit.estimable and _in(fit.S, P1A_S_RANGE)
                and all(_in(v, P1A_S_LOO_RANGE) for v in loo))


def _slope_refutes(fit: SlopeFit) -> bool:
    return bool(fit.estimable and _out(fit.S, P1A_S_RANGE)
                and all(_out(v, P1A_S_RANGE) for v in fit.loo.values()))


def p1a(pairs: Table, b: float) -> Decision:
    """P1a: bulk braking pairs at A2 (criteria 1-4; refute; 'cautious but lawful')."""
    agent = _agent(pairs)
    mk = _mask(pairs, approach="A2", regimes=("bulk",))
    mc = _mask(pairs, approach="A2", regimes=("bulk",), exclude=EXCL_ALL)
    if not mc.any():
        return _d("P1a", agent, "not testable", reason="no bulk braking pairs at A2")
    tr = _tracking(pairs, mk, mc, b)
    fit = sensitivity(select(pairs, mk), x="v_e_star_band")
    c2 = _slope_confirms(fit)
    r_over = bool(np.isfinite(tr["frac_over"]) and tr["frac_over"] >= P1A_REFUTE_SAME_SIDE)
    r_under = bool(np.isfinite(tr["frac_under"]) and tr["frac_under"] >= P1A_REFUTE_SAME_SIDE)
    r_slope = _slope_refutes(fit)
    r_crash = tr["crash_pairs"] >= P1A_MAX_DET_CRASH_PAIRS + 1
    details = dict(tracking=tr, slope=_slope_summary(fit), c2=c2, refute_over=r_over,
                   refute_under=r_under, refute_slope=r_slope, refute_crash=r_crash)
    if tr["c1"] and c2 and tr["c3"] and tr["c4"]:
        verdict = "confirm"
    elif r_over or r_under or r_slope or r_crash:
        only_under = r_under and not (r_over or r_slope or r_crash)
        lawful = fit.estimable and _in(fit.S, P1A_S_RANGE)
        verdict = "cautious but lawful" if only_under and lawful else "refute"
    else:
        verdict = "inconclusive"
    return _d("P1a", agent, verdict, **details)


def p1_tail(pairs: Table, b: float) -> Decision:
    """P1-tail: P1a's criteria 1, 3, 4 on tail pairs at A2; at A1 >= 60% of tail pairs
    overspeed (or crash typed overspeed) and e_ve^RA(A1) - e_ve(A2) > 0 in >= 75% of tail
    geometries. Refuted if the A1 tail pairs track (>= 80% in band, <= 1 crash pair)."""
    agent = _agent(pairs)
    m2k, m2c = (_mask(pairs, approach="A2", regimes=("tail",), exclude=e)
                for e in (EXCL_KEY, EXCL_ALL))
    m1k, m1c = (_mask(pairs, approach="A1", regimes=("tail",), exclude=e)
                for e in (EXCL_KEY, EXCL_ALL))
    if not (m2c.any() or m1c.any()):
        return _d("P1-tail", agent, "not testable", reason="no tail braking pairs")
    a2 = _tracking(pairs, m2k, m2c, b)
    a2_holds = bool(a2["c1"] and a2["c3"] and a2["c4"])
    fit2 = sensitivity(select(pairs, m2k), x="v_e_star_band")
    side = sides(pairs, b)
    over = (side == "over") | (_it(pairs, "det_overspeed", 0) > 0)
    frac_over = _frac(over[m1k])
    e_band, _ = band_errors(pairs)
    geo = _st(pairs, "geometry")
    a1 = {g: e for g, e in zip(geo[m1k], e_band[m1k], strict=True)}
    a2v = {g: e for g, e in zip(geo[m2k], e_band[m2k], strict=True)}
    both = [g for g in a1 if g in a2v and not np.isnan(a1[g]) and not np.isnan(a2v[g])]
    pos = [bool(a1[g] > a2v[g]) for g in both]
    frac_pos = _frac(np.array(pos, bool))
    a1_holds = bool(np.isfinite(frac_over) and frac_over >= P1TAIL_OVERSPEED
                    and np.isfinite(frac_pos) and frac_pos >= P1TAIL_CONTRAST)
    a1_track = _tracking(pairs, m1k, m1c, b)
    details = dict(a2=a2, a2_holds=a2_holds, a2_S=fit2.S if fit2.estimable else "not estimable",
                   a1_frac_overspeed=frac_over, contrast_geometries=len(both),
                   contrast_frac_positive=frac_pos, a1_holds=a1_holds, a1_tracking=a1_track)
    if a2_holds and a1_holds:
        verdict = "confirm"
    elif a1_track["c1"] and a1_track["c3"]:
        verdict = "refute"
    else:
        verdict = "inconclusive"
        details["a2_part"] = "holds" if a2_holds else "fails"
    return _d("P1-tail", agent, verdict, **details)


def kappa_clipped_pairs(runs: Table) -> set[int]:
    """Pair ids kappa-clipped at run time in one snapshot's run table (protocol 2.3's binding
    definition, the pair table's ``kappa_clipped``: any deterministic run of the pair drove a
    preview kappa channel to the clip). The anchored-keys stage uses it to pick P0's
    non-clipped below-support A1 cells (ruling R12)."""
    m = (_st(runs, "mode") == "det") & _bo(runs, "kappa_clipped")
    return {int(p) for p in _it(runs, "pair")[m]}


def _below_mask(pairs: Table, approach: str, exclude=EXCL_KEY) -> np.ndarray:
    """Below-support pairs not kappa-clipped at run time (AD9), no-cap variants excluded."""
    return (_mask(pairs, approach=approach, regimes=("below", "clip"), exclude=exclude)
            & ~_bo(pairs, "kappa_clipped"))


def _bulk_fit(pairs: Table, approach: str) -> SlopeFit:
    return sensitivity(select(pairs, _mask(pairs, approach=approach, regimes=("bulk",))),
                       x="v_e_star_band")


def p1b(pairs: Table, b: float, bulk_fits: dict[str, SlopeFit] | None = None,
        slope_refuted: bool | None = None) -> Decision:
    """P1b over below-support, non-clipped pairs at A1 and A2. r is the residual against
    the agent's bulk fit at the same approach (``bulk_fits``; default fitted here), or
    log(1 + e_ve) if P1a is refuted on its slope (``slope_refuted``; default: P1a's slope
    refute rule on the A2 bulk fit). At A1 every error is e_ve^RA (``band_errors``): ruling
    R12 extends P0's re-anchored keys to the non-clipped below-support A1 cells, so Kendall
    tau does not mix keys; ``ra_fallback_pairs`` in the details counts A1 pairs of its scope
    whose runs still fell back to the nominal key (0 once those keys exist)."""
    agent = _agent(pairs)
    approaches = ("A1", "A2")
    masks = {ap: _below_mask(pairs, ap) for ap in approaches}
    n_below = int(sum(m.sum() for m in masks.values()))
    if n_below < P1B_MIN_PAIRS:
        return _d("P1b", agent, "not testable", n_pairs=n_below)
    fits = bulk_fits or {ap: _bulk_fit(pairs, ap) for ap in approaches}
    if slope_refuted is None:
        slope_refuted = _slope_refutes(fits.get("A2") or _bulk_fit(pairs, "A2"))
    e_band, _ = band_errors(pairs)
    thr = math.log1p(b)

    def resid(m: np.ndarray, ap: str) -> np.ndarray:
        if slope_refuted:
            with np.errstate(invalid="ignore"):
                return np.log1p(e_band[m])
        fit = fits.get(ap)
        if fit is None or not np.isfinite(fit.S):
            return np.full(int(m.sum()), np.nan)
        return extrapolation_residual(select(pairs, m), fit, x="v_e_star_band")

    r_all, ov_all, crash_all, taus = [], [], [], {}
    R_seg = _fl(pairs, "R_seg")
    fallback = _bo(pairs, "uses_ra") & (_it(pairs, "ra_fallback", 0) > 0)
    n_fallback = 0
    for ap in approaches:
        m = masks[ap]
        if not m.any():
            continue
        r = resid(m, ap)
        r_all.append(r)
        ov_all.append(_it(pairs, "det_overspeed", 0)[m] > 0)
        crash_all.append(_it(pairs, "det_crashes", 0)[m] > 0)
        scope = m | _mask(pairs, approach=ap, regimes=("tail",))
        taus[ap] = kendall_tau(resid(scope, ap), R_seg[scope])
        n_fallback += int((fallback & scope).sum())
    r, ov, crash = np.concatenate(r_all), np.concatenate(ov_all), np.concatenate(crash_all)
    with np.errstate(invalid="ignore"):
        frac_dep = _frac((r > thr) | ov)
        frac_lawful = _frac(np.abs(r) <= thr)
    tau_ok = bool(taus) and all(np.isfinite(t) and t <= P1B_KENDALL for t in taus.values())
    details = dict(n_pairs=n_below, slope_refuted=bool(slope_refuted), threshold=thr,
                   frac_departure=frac_dep, frac_lawful=frac_lawful, kendall_tau=taus,
                   det_crash_pairs=int(crash.sum()), ra_fallback_pairs=n_fallback)
    if frac_dep >= P1B_FRACTION and tau_ok:
        verdict = "confirm"
    elif frac_lawful >= P1B_FRACTION and not crash.any():
        verdict = "refute"
    else:
        verdict = "inconclusive"
    return _d("P1b", agent, verdict, **details)


def r_dep(pairs: Table, approach: str, b: float) -> float:
    """Smallest grid R at which this and every larger braking R are in band with no
    deterministic crash (+inf if the largest braking R fails; NaN without pairs)."""
    m = _mask(pairs, approach=approach)
    if not m.any():
        return float("nan")
    side, R = sides(pairs, b), _fl(pairs, "R_nom")
    crashes = _it(pairs, "det_crashes", 0)
    out = float("inf")
    for r in np.unique(R[m])[::-1]:
        rows = m & (R == r)
        if np.all(side[rows] == "in") and np.all(crashes[rows] == 0):
            out = float(r)
        else:
            break
    return out


def _order_ok(v: tuple[float, ...]) -> bool:
    return all(v[i] >= v[i + 1] for i in range(len(v) - 1)) and \
        any(v[i] > v[i + 1] for i in range(len(v) - 1))


def p1c(r_deps: dict[str, dict[str, float]]) -> list[Decision]:
    """Order of departure with the exact null over the 6 label permutations: R_dep(G1) >=
    R_dep(G2) >= R_dep(G3) at A1 with one strict inequality and no reversal (all equal:
    inconclusive), plus R_dep(A1) >= R_dep(A2) per agent. ``r_deps``: agent -> approach ->
    R_dep."""
    out = []
    for a in GENERALISTS:
        d = r_deps.get(a, {})
        a1, a2 = d.get("A1", np.nan), d.get("A2", np.nan)
        if np.isnan(a1) or np.isnan(a2):
            out.append(_d("P1c-approach", a, "not testable", A1=a1, A2=a2))
        else:
            out.append(_d("P1c-approach", a, "confirm" if a1 >= a2 else "refute", A1=a1, A2=a2))
    vals = tuple(float(r_deps.get(a, {}).get("A1", np.nan)) for a in GENERALISTS)
    agent = ",".join(GENERALISTS)
    if any(np.isnan(v) for v in vals):
        out.insert(0, _d("P1c", agent, "not testable", r_dep_A1=vals))
        return out
    p_null = sum(_order_ok(p) for p in permutations(vals)) / 6.0
    if len(set(vals)) == 1:
        verdict = "inconclusive"
    else:
        verdict = "confirm" if _order_ok(vals) else "refute"
    out.insert(0, _d("P1c", agent, verdict, r_dep_A1=vals, p_null=p_null))
    return out


# ------------------------------------------------------------------------- P2


def _p2_scope(pairs: Table, exclude=EXCL_KEY) -> np.ndarray:
    """Tail and non-clipped below-support braking pairs at A1."""
    return _mask(pairs, approach="A1", regimes=("tail",), exclude=exclude) | \
        _below_mask(pairs, "A1", exclude)


def _o_set(pairs: Table, b: float) -> np.ndarray:
    over = (sides(pairs, b) == "over") | (_it(pairs, "det_overspeed", 0) > 0)
    return _p2_scope(pairs) & over


def _no_brake(pairs: Table) -> np.ndarray:
    nr = _it(pairs, "n_runs", 0)
    return np.array([nb < _min_onsets(int(n)) for nb, n in zip(
        _it(pairs, "n_brake", 0), nr, strict=True)], bool) & (nr > 0)


def p2(pairs: Table, runs: Table | None, b: float) -> Decision:
    """P2 on set O (tail and non-clipped below-support A1 pairs that overspeed or crash
    typed overspeed): (a) Δd_b < 0 and Δd_resp < 0 in >= 75% of O; (b) median per-cell
    overspeed share of typed crashes >= 75% over the A1 tail/below cells with crashes.
    ``runs`` (optional) lists the no-brake entries in details."""
    agent = _agent(pairs)
    O = _o_set(pairs, b)
    if O.sum() < P2_MIN_SET:
        return _d("P2", agent, "not applicable", n_O=int(O.sum()))
    ddb, ddr = _fl(pairs, "dd_b")[O], _fl(pairs, "dd_resp")[O]
    nob = _no_brake(pairs)[O] & np.isnan(ddb)
    onset_fail = (ddb < 0) | nob
    a_onset, a_resp = _frac(onset_fail), _frac(ddr < 0)
    a = a_onset >= P2_ONSET_FRACTION and a_resp >= P2_ONSET_FRACTION
    a_ref = _frac(ddb >= 0) >= P2_REFUTE_ONSET
    cells = _p2_scope(pairs, EXCL_ALL) & (_it(pairs, "typed_crashes", 0) > 0)
    med = _median(_fl(pairs, "overspeed_frac")[cells]) if cells.any() else float("nan")
    counts = {t: int(_it(pairs, f"typed_{t}", 0)[cells].sum())
              for t in ("overspeed", "overbrake", "line_steer", "other")}
    top = max(counts, key=lambda t: counts[t]) if any(counts.values()) else ""
    tie = sum(v == counts[top] for v in counts.values()) > 1 if top else False
    b_ok = bool(np.isfinite(med) and med >= P2_OVERSPEED_MEDIAN)
    b_ref = bool(np.isfinite(med) and med < P2_REFUTE_OVERSPEED and top
                 in ("overbrake", "line_steer") and not tie)
    details = dict(n_O=int(O.sum()), frac_onset_late=a_onset, frac_resp_late=a_resp,
                   no_brake_entries=int(nob.sum()), frac_onset_not_late=_frac(ddb >= 0),
                   crash_cells=int(cells.sum()), median_overspeed_share=med,
                   crash_type_counts=counts, a=bool(a), b=b_ok, a_refute=bool(a_ref),
                   b_refute=b_ref)
    if runs is not None and _n(runs):
        pair_ids = set(_it(pairs, "pair")[O].tolist())
        rm = (_st(runs, "mode") == "det") & np.isin(_it(runs, "pair"), list(pair_ids)) \
            & ~_bo(runs, "has_brake")
        details["no_brake_runs"] = [(int(p), int(c), int(k)) for p, c, k in zip(
            _it(runs, "pair")[rm], _it(runs, "cell")[rm], _it(runs, "phase", 0)[rm],
            strict=True)]
    if a and b_ok:
        verdict = "confirm"
    elif a_ref or b_ref:
        verdict = "refute"
    else:
        verdict = "inconclusive"
    return _d("P2", agent, verdict, **details)


def p2b_decompose(v_b, v_e, d_b, v_b_star, v_e_star, d_b_star) -> tuple:
    """(approach, onset, depth) terms of v_e^2 - v_e*^2 = (v_b^2 - v_b*^2) - (abar + abar*)
    (d_b - d_b*) - (d_b + d_b*)(abar - abar*), abar = (v_b^2 - v_e^2) / (2 d_b)."""
    v_b, v_e, d_b = (np.asarray(a, float) for a in (v_b, v_e, d_b))
    v_bs, v_es, d_bs = (np.asarray(a, float) for a in (v_b_star, v_e_star, d_b_star))
    with np.errstate(invalid="ignore", divide="ignore"):
        abar = (v_b**2 - v_e**2) / (2 * d_b)
        abar_s = (v_bs**2 - v_es**2) / (2 * d_bs)
    approach = v_b**2 - v_bs**2
    onset = -(abar + abar_s) * (d_b - d_bs)
    depth = -(d_b + d_bs) * (abar - abar_s)
    return approach, onset, depth


def onset_share(approach, onset, depth) -> np.ndarray:
    """Onset term over the summed positive terms (NaN when no term is positive)."""
    terms = np.stack([np.asarray(t, float) for t in (approach, onset, depth)])
    pos = np.clip(terms, 0, None)
    tot = pos.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, pos[1] / tot, np.nan)


def _key_k(pairs: Table, col: str) -> np.ndarray:
    """The pair's phase-matched key value (column '<col>_k'), else the u = 0 key's."""
    k, base = _fl(pairs, f"{col}_k"), _fl(pairs, col)
    return np.where(np.isfinite(k), k, base)


def p2b(pairs: Table, b: float) -> Decision:
    """P2b: median onset share over O > 50% (no-brake entries that overspeed count as onset
    failures, share 1). The key side is the phase-matched key where one was solved
    (protocol 6.1: phase-matched keys back P2b), else the u = 0 key."""
    agent = _agent(pairs)
    O = _o_set(pairs, b)
    if O.sum() < P2_MIN_SET:
        return _d("P2b", agent, "not applicable", n_O=int(O.sum()))
    terms = p2b_decompose(_fl(pairs, "v_b")[O], _fl(pairs, "v_e_obs")[O], _fl(pairs, "d_b")[O],
                          _key_k(pairs, "v_b_star")[O], _key_k(pairs, "v_e_star")[O],
                          _key_k(pairs, "d_b_star")[O])
    share = onset_share(*terms)
    nob = _no_brake(pairs)[O] & np.isnan(_fl(pairs, "d_b")[O])
    share = np.where(nob, 1.0, share)
    med = _median(share)
    verdict = "not testable" if np.isnan(med) else ("confirm" if med > P2B_SHARE else "refute")
    return _d("P2b", agent, verdict, n_O=int(O.sum()), median_onset_share=med,
              no_brake_entries=int(nob.sum()), approach=_median(terms[0]),
              onset=_median(terms[1]), depth=_median(terms[2]))


def qss_v_min(bank, cells: list) -> dict[int, dict]:
    """P2d's point-mass null per cell: the centreline QSS (``oracle.qss.speed_profile`` on
    the car's GG, from the cell's u = 0 start at v0) minimum over [s_c0 - 50, s_c1 + 50],
    as ``key_table``'s ``extra`` ({cell_id: {'v_min_qss': m/s}}). Cells without a start
    (no D yet) are left out."""
    from apexzero.oracle.qss import GG, speed_profile
    from apexzero.vehicle.params import F1_2025

    gg = GG.from_params(F1_2025)
    ds = float(getattr(bank, "ds", 2.0))
    out: dict[int, dict] = {}
    for c in cells:
        try:
            x0 = np.asarray(c.initial_state(0), float)
        except ValueError:
            continue
        lm = bank.stimuli[c.stim_id].landmarks
        i0 = int(np.floor(x0[0] / ds))
        k = np.asarray(bank.kappa[c.stim_id][i0:], float)
        if k.size < 3:
            continue
        v = speed_profile(k, ds, gg, closed=False, v_start=float(x0[3]))
        s = (i0 + np.arange(k.size)) * ds
        win = (s >= lm.s_c0 - V_MIN_MARGIN) & (s <= lm.s_c1 + V_MIN_MARGIN)
        if win.any():
            out[c.cell_id] = {"v_min_qss": float(np.min(np.asarray(v)[win]))}
    return out


def p2d(pairs: Table, b: float) -> Decision:
    """P2d (exploratory): is hairpin v_min nearer the centreline QSS value than the key's?
    And the line-conditioned excess e_vmin^line (agent minus key)."""
    agent = _agent(pairs)
    m = _mask(pairs, braking=True) & (_fl(pairs, "theta_deg") >= THETA_CLASS_EDGE_DEG)
    v, vq, vs = _fl(pairs, "v_min")[m], _fl(pairs, "v_min_qss")[m], _fl(pairs, "v_min_star")[m]
    ok = np.isfinite(v) & np.isfinite(vq) & np.isfinite(vs)
    if not ok.any():
        return _d("P2d", agent, "not testable", n=0)
    nearer = np.abs(v[ok] - vq[ok]) < np.abs(v[ok] - vs[ok])
    return _d("P2d", agent, "descriptive", n=int(ok.sum()), frac_nearer_qss=_frac(nearer),
              median_line_excess=_median(_fl(pairs, "e_vmin_line_excess")[m]))


# ------------------------------------------------------------------------- P3


@dataclass
class P3Result:
    b: float
    loo_b: dict               # geometry -> b
    rmse: dict                # 'M_phys' | 'M_idx' | 'M_fd' -> held-out RMSE of log d_b
    winner: str
    n_runs: int
    n_predicted: int = 0
    n_geometries: int = 0
    agent: str = ""


def _dphys(tables, v_b: np.ndarray, v_e: np.ndarray) -> np.ndarray:
    return np.array([float(tables.d_b_phys(float(a), float(c))) for a, c in
                     zip(v_b, v_e, strict=True)])


def _join_keys(runs: Table, keys: Table, cols: tuple[str, ...]) -> dict[str, np.ndarray]:
    kidx = {int(c): i for i, c in enumerate(_it(keys, "cell"))}
    rows = np.array([kidx.get(int(c), -1) for c in _it(runs, "cell")])
    out = {}
    for c in cols:
        col = np.asarray(keys[c]) if c in keys else np.full(_n(keys), np.nan)
        fill = "" if col.dtype.kind in "US" else (False if col.dtype.kind == "b" else np.nan)
        out[c] = np.where(rows >= 0, col[np.maximum(rows, 0)] if col.size else fill, fill)
    return out


def _p3_data(runs: Table, keys: Table, tables, max_v0: float | None = None) -> dict:
    k = _join_keys(runs, keys, ("tiers", "geometry", "v0", "D", "key_status"))
    tiers = k["tiers"].astype(str)
    tier_ab = np.array([("A" in t.split(",")) or ("B" in t.split(",")) for t in tiers], bool)
    v_b, v_e, d_b = _fl(runs, "v_b"), _fl(runs, "v_e"), _fl(runs, "d_b")
    ok = ((_st(runs, "mode") == "det") & tier_ab & _bo(runs, "has_brake")
          & ~_bo(runs, "forced_onset") & np.isfinite(v_b) & np.isfinite(v_e)
          & np.isfinite(d_b) & (d_b > 0) & (v_b - v_e >= P3_MIN_DROP)
          & ~np.isin(k["key_status"].astype(str), EXCL_ALL))
    if max_v0 is not None:
        ok &= k["v0"].astype(float) < max_v0
    dp = _dphys(tables, v_b[ok], v_e[ok]) if ok.any() else np.zeros(0)
    good = np.isfinite(dp) & (dp > 0)
    sel = np.flatnonzero(ok)[good]
    v0, D = k["v0"].astype(float)[sel], k["D"].astype(float)[sel]
    return dict(y=np.log(d_b[sel]), x=np.log(dp[good]),
                f_idx=np.log(np.clip(P3_IDX_TIME * v_b[sel], *P3_IDX_CLIP)),
                g=k["geometry"].astype(str)[sel],
                cond=np.array([f"{a:.3f}/{c:.1f}" for a, c in zip(v0, D, strict=True)]),
                v_b=v_b[sel], v0=v0, n=int(sel.size))


def _heldout_rmse(y: np.ndarray, f: np.ndarray, g: np.ndarray, cond: np.ndarray
                  ) -> tuple[float, int]:
    resid = []
    for c in np.unique(cond):
        te, tr = cond == c, cond != c
        cg = {gg: float(np.mean(y[tr & (g == gg)] - f[tr & (g == gg)])) for gg in np.unique(g[tr])}
        resid += [y[i] - (cg[g[i]] + f[i]) for i in np.flatnonzero(te) if g[i] in cg]
    r = np.asarray(resid)
    return (float(np.sqrt(np.mean(r * r))) if r.size else float("nan")), int(r.size)


def p3_fit(runs: Table, keys: Table, tables, max_v0: float | None = None) -> P3Result:
    """log d_b = a_g + b log d_b^phys(v_b, v_e) and the three models' held-out RMSE, one
    (v0, D) condition held out at a time (P3). Deterministic tier A and B runs with a
    sustained, unforced brake, v_b - v_e >= 30 km/h and v_e observed (crashes after s_c0
    included). ``tables.d_b_phys`` is ``measures.ModelTables.d_b_phys``. ``max_v0`` (m/s)
    drops the cells at or above it (the 310 km/h cells are analysed separately)."""
    agent = _agent(runs)
    d = _p3_data(runs, keys, tables, max_v0)
    if d["n"] < 3:
        return P3Result(np.nan, {}, {}, "", d["n"], 0, 0, agent)
    y, x, g, cond = d["y"], d["x"], d["g"], d["cond"]
    b = _fe_slope(x, y, g)[0]
    geos = np.unique(g)
    loo = {str(gg): _fe_slope(x[g != gg], y[g != gg], g[g != gg])[0] for gg in geos} \
        if geos.size > 1 else {}
    rmse, n_pred = {}, 0
    for name, f in (("M_phys", x), ("M_idx", d["f_idx"]), ("M_fd", np.zeros_like(y))):
        rmse[name], n_pred = _heldout_rmse(y, f, g, cond)
    finite = {k: v for k, v in rmse.items() if np.isfinite(v)}
    winner = min(finite, key=lambda k: finite[k]) if finite else ""
    return P3Result(float(b), loo, rmse, winner, d["n"], n_pred, int(geos.size), agent)


def p3_ratio(runs: Table, keys: Table, tables) -> Decision:
    """P3 secondary, descriptive: d_b / d_b^phys against v_b (flat: lawful; falling with v_b:
    a time-to-corner rule), as the within-geometry slope of log(d_b / d_b^phys) on log v_b,
    and b refitted without the 310 km/h cells (d_max saturates at 500 m above 300 km/h)."""
    agent = _agent(runs)
    d = _p3_data(runs, keys, tables)
    if d["n"] < 3:
        return _d("P3-ratio", agent, "not testable", n=d["n"])
    slope = _fe_slope(np.log(d["v_b"]), d["y"] - d["x"], d["g"])[0]
    no310 = p3_fit(runs, keys, tables, max_v0=kph(300.0))
    fast = d["v0"] >= kph(300.0)
    return _d("P3-ratio", agent, "descriptive", ratio_slope_vs_log_vb=slope,
              median_ratio=float(np.exp(np.median(d["y"] - d["x"]))),
              median_ratio_310=float(np.exp(np.median((d["y"] - d["x"])[fast])))
              if fast.any() else np.nan, b_without_310=no310.b, n=d["n"])


def brake_share(runs: Table, keys: Table) -> float:
    """Fraction of deterministic tier B runs with a sustained brake (P3's applicability)."""
    k = _join_keys(runs, keys, ("tiers", "key_status"))
    tier_b = np.array(["B" in t.split(",") for t in k["tiers"].astype(str)], bool)
    m = (_st(runs, "mode") == "det") & tier_b & ~np.isin(k["key_status"].astype(str), EXCL_ALL)
    return _frac(_bo(runs, "has_brake")[m])


def p3(result: P3Result, brake_share: float) -> Decision:
    """Confirm: b in [0.9, 1.1], every leave-one-geometry-out b in [0.85, 1.15] and M_phys
    has the lowest held-out RMSE. Refute naming the winner: M_idx or M_fd beats M_phys by
    >= 10% and b outside [0.85, 1.15]. Only for agents braking in >= 50% of tier B runs."""
    agent = result.agent
    det = dict(b=result.b, loo_b=result.loo_b, rmse=result.rmse, winner=result.winner,
               n_runs=result.n_runs, n_predicted=result.n_predicted,
               n_geometries=result.n_geometries, brake_share=brake_share)
    if not (np.isfinite(brake_share) and brake_share >= P3_BRAKE_SHARE):
        return _d("P3", agent, "not applicable", **det)
    if not np.isfinite(result.b) or not result.rmse:
        return _d("P3", agent, "not testable", **det)
    r_phys = result.rmse.get("M_phys", np.nan)
    alts = {k: v for k, v in result.rmse.items() if k != "M_phys" and np.isfinite(v)}
    best = min(alts, key=lambda k: alts[k]) if alts else ""
    if (_in(result.b, P3_B_RANGE) and all(_in(v, P3_B_LOO_RANGE) for v in result.loo_b.values())
            and result.winner == "M_phys"):
        return _d("P3", agent, "confirm", **det)
    if best and np.isfinite(r_phys) and alts[best] <= (1 - P3_REFUTE_MARGIN) * r_phys \
            and not _in(result.b, P3_B_LOO_RANGE):
        det["winner"] = best
        return _d("P3", agent, "refute", **det)
    return _d("P3", agent, "inconclusive", **det)


def p3_separation(keys_table: Table, tables) -> dict:
    """Pre-tag check: on the key's own (v_b*, v_e*), the slopes M_idx and M_fd predict must
    differ from 1 by >= 0.25, else P3 is demoted to secondary. Within-geometry slopes of
    each model's log d_b on log d_b^phys over tier A and B braking cells."""
    kt = keys_table
    tier_ab = has_tier(kt, "A") | has_tier(kt, "B")
    vb, ve = _fl(kt, "v_b_star"), _fl(kt, "v_e_star")
    m = (tier_ab & _bo(kt, "braking") & (_st(kt, "key_status") == "verified")
         & np.isfinite(vb) & np.isfinite(ve) & (vb - ve >= P3_MIN_DROP))
    if m.sum() < 3:
        return dict(b_idx=np.nan, b_fd=np.nan, passed=False, demote_p3=True, n_cells=int(m.sum()))
    x = np.log(_dphys(tables, vb[m], ve[m]))
    g = _st(kt, "geometry")[m]
    b_idx = _fe_slope(x, np.log(np.clip(P3_IDX_TIME * vb[m], *P3_IDX_CLIP)), g)[0]
    b_fd = _fe_slope(x, np.zeros_like(x), g)[0]
    passed = bool(np.isfinite(b_idx) and abs(b_idx - 1) >= P3_SEPARATION
                  and np.isfinite(b_fd) and abs(b_fd - 1) >= P3_SEPARATION)
    return dict(b_idx=float(b_idx), b_fd=float(b_fd), passed=passed, demote_p3=not passed,
                n_cells=int(m.sum()), n_geometries=int(np.unique(g).size))


def p3_beta(runs: Table, keys: Table) -> list[Decision]:
    """P3 secondary: log d_b = a + beta log(v_b^2 - v_e*^2) by theta class (< 120, >= 120 deg)
    for the agent (v_e* of the key behind each run's e_ve, v_e / (1 + e_ve): the u = 0 key,
    or the key at the run's phase under AD23's V7 remedy) and the min-time
    key (s-OCP columns; the nominal key where the s-OCP is missing). |beta - beta*| <= 0.1."""
    agent = _agent(runs)
    k = _join_keys(runs, keys, ("tiers", "theta_deg", "key_status"))
    tiers = k["tiers"].astype(str)
    tier_ab = np.array([("A" in t.split(",")) or ("B" in t.split(",")) for t in tiers], bool)
    th = k["theta_deg"].astype(float)
    v_b, d_b, e_ve, v_e = (_fl(runs, c) for c in ("v_b", "d_b", "e_ve", "v_e"))
    with np.errstate(invalid="ignore", divide="ignore"):
        v_es = v_e / (1 + e_ve)
        xa = np.log(v_b**2 - v_es**2)
        ya = np.log(d_b)
    ok = ((_st(runs, "mode") == "det") & tier_ab & _bo(runs, "has_brake")
          & ~_bo(runs, "forced_onset") & np.isfinite(xa) & np.isfinite(ya)
          & ~np.isin(k["key_status"].astype(str), EXCL_ALL))
    kb = (has_tier(keys, "A") | has_tier(keys, "B")) & _bo(keys, "braking")
    vbk = np.where(np.isfinite(_fl(keys, "v_b_socp")), _fl(keys, "v_b_socp"), _fl(keys, "v_b_star"))
    vek = np.where(np.isfinite(_fl(keys, "v_e_socp")), _fl(keys, "v_e_socp"), _fl(keys, "v_e_star"))
    dbk = np.where(np.isfinite(_fl(keys, "d_b_socp")), _fl(keys, "d_b_socp"), _fl(keys, "d_b_star"))
    with np.errstate(invalid="ignore", divide="ignore"):
        xk, yk = np.log(vbk**2 - vek**2), np.log(dbk)
    okk = kb & np.isfinite(xk) & np.isfinite(yk)
    out = []
    for name, sel_a, sel_k in (("lt120", th < THETA_CLASS_EDGE_DEG,
                                _fl(keys, "theta_deg") < THETA_CLASS_EDGE_DEG),
                               ("ge120", th >= THETA_CLASS_EDGE_DEG,
                                _fl(keys, "theta_deg") >= THETA_CLASS_EDGE_DEG)):
        ma, mk = ok & sel_a, okk & sel_k
        beta = _ols_slope(xa[ma], ya[ma])
        beta_s = _ols_slope(xk[mk], yk[mk])
        if not (np.isfinite(beta) and np.isfinite(beta_s)):
            verdict = "not testable"
        else:
            verdict = "confirm" if abs(beta - beta_s) <= P2B_BETA_TOL else "refute"
        out.append(_d(f"P3-beta-{name}", agent, verdict, beta=beta, beta_star=beta_s,
                      n=int(ma.sum()), n_key=int(mk.sum())))
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) <= 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


# ------------------------------------------------------------------------- P4

F1_CONDS = tuple(c for c in D_CONDITIONS if c.startswith("F1/"))
F1_SINGLE = {"before": "F1/straight/real/real", "after": "F1/real/straight/real",
             "width": "F1/real/real/const"}
F1_FACTORS = ("before", "after", "width")
LAMBDA_CONDS = tuple(c for c in D_CONDITIONS if c.startswith("F2/lambda"))
F6_CONDS = tuple(c for c in D_CONDITIONS if c.startswith("F6/"))
COND_MIRROR, COND_SWAP_A, COND_SWAP_B = "F4/mirror", "F5/swapA", "F5/swapB"
P4_LABELS = {-1: "not testable: d_OOD missing for a failing F1 or an F6 condition",
             0: "not competent from these states", 1: "off-manifold fragile",
             2: "context-keyed memorisation", 3: "corner-keyed memorisation",
             4: "reads the road, generally", 5: "reads the road, locally", 6: "mixed"}


@dataclass
class P4Result:
    agent: str
    cls: int                  # 0..6 (0 = gate failed)
    label: str
    C: dict                   # condition -> completion fraction
    S_spec: float
    phi: float                # median phantom index
    m: float                  # median swap index
    MI: float                 # memorisation index
    key_factors: tuple[str, ...] = ()
    details: dict = field(default_factory=dict)


def phantom_index(v_peak: float, v_min_region: float, v_min_a: float) -> float:
    """phi = (v_peak - v_min,region) / (v_peak - v_min,i^(a)); 0 reader, 1 memoriser."""
    den = v_peak - v_min_a
    return float((v_peak - v_min_region) / den) if np.isfinite(den) and abs(den) > 1e-9 \
        else float("nan")


def swap_index(v_min_swap: float, v_min_i: float, v_min_j: float) -> float:
    """m = log(v_min^swap / v_min,j^(a)) / log(v_min,i^(a) / v_min,j^(a)); 0 reader, 1
    memoriser."""
    with np.errstate(invalid="ignore", divide="ignore"):
        den = math.log(v_min_i / v_min_j) if v_min_i > 0 and v_min_j > 0 else float("nan")
        num = math.log(v_min_swap / v_min_j) if v_min_swap > 0 and v_min_j > 0 else float("nan")
    return float(num / den) if np.isfinite(den) and abs(den) > 1e-12 else float("nan")


def _cond_lookup(cond: Table) -> dict[tuple[int, str], int]:
    return {(int(c), str(n)): i for i, (c, n) in enumerate(
        zip(_it(cond, "corner"), _st(cond, "condition"), strict=True))}


def completion_by_condition(cond: Table) -> tuple[dict[str, float], dict[str, float], list[int]]:
    """(C, median d_OOD, braking corners) per condition, over the corners whose key brakes
    in condition (a)."""
    corner, cn = _it(cond, "corner"), _st(cond, "condition")
    braking = sorted({int(c) for c, n, b in zip(corner, cn, _bo(cond, "braking"), strict=True)
                      if n == COND_A and b})
    inb = np.isin(corner, braking)
    comp, dood = _bo(cond, "complete"), _fl(cond, "d_ood")
    C, D = {}, {}
    for c in np.unique(cn[inb]):
        m = inb & (cn == c)
        C[str(c)] = _frac(comp[m])
        D[str(c)] = _median(dood[m])
    return C, D, braking


def _spec_indices(cond: Table, braking: list[int]) -> tuple[float, float, float, dict]:
    look = _cond_lookup(cond)
    vmin, vstar, vpeak = _fl(cond, "v_min"), _fl(cond, "v_min_star"), _fl(cond, "v_peak")
    vreg, swap_j = _fl(cond, "v_min_region"), _it(cond, "swap_j")
    slopes, phis, ms = [], [], []
    for i in braking:
        rows = [look[(i, c)] for c in (COND_A, *LAMBDA_CONDS) if (i, c) in look]
        x, y = np.log(vstar[rows]), np.log(vmin[rows])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3 and np.ptp(x[ok]) > 0:
            slopes.append(_ols_slope(x[ok], y[ok]))
        a = look.get((i, COND_A))
        v_ia = vmin[a] if a is not None else np.nan
        if (i, COND_SWAP_B) in look:
            r = look[(i, COND_SWAP_B)]
            phis.append(phantom_index(vpeak[r], vreg[r], v_ia))
        if (i, COND_SWAP_A) in look:
            r = look[(i, COND_SWAP_A)]
            j = int(swap_j[r])
            ja = look.get((j, COND_A))
            if j >= 0 and ja is not None:
                ms.append(swap_index(vmin[r], v_ia, vmin[ja]))
    return _median(slopes), _median(phis), _median(ms), dict(n_slopes=len(slopes),
                                                             n_phi=len(phis), n_m=len(ms))


def _main_effects(C: dict[str, float]) -> dict[str, float]:
    out = {}
    for k, f in enumerate(F1_FACTORS):
        real, changed = [], []
        for c in F1_CONDS:
            if c not in C or not np.isfinite(C[c]):
                continue
            (real if c.split("/")[k + 1] == "real" else changed).append(C[c])
        out[f] = float(np.mean(real) - np.mean(changed)) if real and changed else float("nan")
    return out


def memorisation_index(cond: Table, generalists: Table | None) -> float:
    """MI = [C(a) - C(iso)] - mean over generalist finals of [C_G(a) - C_G(iso)]."""
    C, _, _ = completion_by_condition(cond)
    own = C.get(COND_A, np.nan) - C.get(COND_ISO, np.nan)
    if generalists is None or not _n(generalists):
        return float("nan")
    diffs = []
    for a in np.unique(_st(generalists, "agent")):
        Cg, _, _ = completion_by_condition(select(generalists, _st(generalists, "agent") == a))
        diffs.append(Cg.get(COND_A, np.nan) - Cg.get(COND_ISO, np.nan))
    return float(own - np.nanmean(diffs)) if diffs else float("nan")


def p4_classify(cond: Table, generalists: Table | None, tier_a: Table | None = None,
                tier_g: Table | None = None) -> P4Result:
    """The ordered class rules of P4 (first match wins). ``cond``: one specialist's tier D
    condition table; ``generalists``: the generalist finals' tables (MI); ``tier_a`` /
    ``tier_g``: the specialist's pair tables for class 4 (tier A A2 completion for R_seg
    >= 10.7 m; any crash on a tier G null straight)."""
    agent = _agent(cond)
    C, D, braking = completion_by_condition(cond)
    S_spec, phi, m, nidx = _spec_indices(cond, braking)
    MI = memorisation_index(cond, generalists)
    eff = _main_effects(C)
    det: dict[str, Any] = dict(d_ood=D, main_effects=eff, n_braking=len(braking), **nidx)
    tier_g_crash = bool((_it(tier_g, "det_crashes", 0) > 0).any()) if tier_g is not None \
        and _n(tier_g) else None
    det["tier_g_crash"] = tier_g_crash

    def res(cls: int, keys_: tuple[str, ...] = (), label: str | None = None) -> P4Result:
        return P4Result(agent, cls, label or P4_LABELS[cls], C, S_spec, phi, m, MI, keys_, det)

    def c(name: str) -> float:
        return C.get(name, float("nan"))

    if not (c(COND_A) >= P4_GATE):
        return res(0)
    f6 = [(c(f), D.get(f, np.nan)) for f in F6_CONDS if f in C]
    single = [s for s in F1_SINGLE.values() if c(s) <= P4_FAIL]
    if single:
        # Classes 1 and 2 compare d_OOD across conditions: without it (no buffer, no
        # observations) neither rule can be applied, and class 2 must not hold vacuously.
        missing = [s for s in single if not np.isfinite(D.get(s, np.nan))]
        missing += [f for f in F6_CONDS if f in C and not np.isfinite(D.get(f, np.nan))]
        if missing or not f6:
            det.update(failing_conditions=single, d_ood_missing=missing or list(F6_CONDS))
            return res(-1)
    for s in single:
        if any(df <= D.get(s, np.nan) and cf <= P4_FAIL for cf, df in f6):
            det["failing_condition"] = s
            return res(1)
    for s in single:
        far = [cf for cf, df in f6 if df >= D.get(s, np.nan)]
        if all(cf >= P4_PASS for cf in far):
            keys_ = tuple(f for f in F1_FACTORS if np.isfinite(eff[f]) and eff[f] >= P4_MAIN_EFFECT)
            det.update(failing_condition=s, vacuous=not far, n_f6_at_or_above=len(far))
            label = P4_LABELS[2] + (" (width-keyed)" if keys_ == ("width",) else "")
            return res(2, keys_, label)
    all8 = all(c(x) >= P4_PASS for x in F1_CONDS)
    if all8 and (phi >= P4_PHI_MEM or m >= P4_M_MEM or S_spec < P4_S_MEM
                 or c(COND_MIRROR) <= P4_FAIL):
        return res(3)
    cls5 = (all8 and c(COND_MIRROR) >= P4_PASS and all(c(x) >= P4_PASS for x in LAMBDA_CONDS)
            and _in(S_spec, P4_S_READ) and phi < P4_PHI_READ and m < P4_M_READ)
    if cls5:
        comp_a = float("nan")
        if tier_a is not None and _n(tier_a):
            ma = _mask(tier_a, approach="A2", braking=None, exclude=EXCL_ALL) & \
                (_fl(tier_a, "R_seg") >= P4_TIER_A_R)
            nr = _it(tier_a, "n_runs", 0)[ma].sum()
            comp_a = float(_it(tier_a, "det_complete", 0)[ma].sum() / nr) if nr else np.nan
        det["tier_a_completion"] = comp_a
        if np.isfinite(comp_a) and comp_a >= P4_TIER_A_COMPLETION and tier_g_crash is False:
            return res(4)
        return res(5)
    return res(6)


def p4(result: P4Result) -> Decision:
    verdict = "not testable" if result.cls < 0 else f"class {result.cls}"
    d = _d("P4", result.agent, verdict, label=result.label, C=result.C,
           S_spec=result.S_spec, phi=result.phi, m=result.m, MI=result.MI,
           key_factors=list(result.key_factors), **result.details)
    return d


def p4_recipe(results: list[P4Result]) -> Decision:
    """A class is claimed for the specialist recipe only if >= 2 of the 3 seeds share it
    (a seed that is not testable counts for no class)."""
    classes = [r.cls for r in results]
    agent = ",".join(r.agent for r in results)
    testable = [c for c in classes if c >= 0]
    best = max(set(testable), key=testable.count) if testable else None
    verdict = f"class {best}" if best is not None and classes.count(best) >= 2 else "inconclusive"
    return _d("P4-recipe", agent, verdict, classes=dict(zip([r.agent for r in results], classes,
                                                             strict=True)),
              predicted_class_2_all=bool(classes) and all(c == 2 for c in classes))


# ------------------------------------------------------------------------- P5


def _run_speed(runs: Table, keys: Table, col: str) -> np.ndarray:
    """Per-run speed value under the 9.1 crash rules (+inf overspeed, NaN before s_c0)."""
    s_c0 = _join_keys(runs, keys, ("s_c0",))["s_c0"].astype(float)
    crashed, ct = _bo(runs, "crashed"), _st(runs, "crash_type")
    ov = crashed & (ct == "overspeed")
    pre = crashed & ~ov & (_fl(runs, "s_crash") < s_c0)
    v = _fl(runs, col).copy()
    v[ov] = np.inf
    v[pre] = np.nan
    return v


def tape_errors(runs: Table, keys: Table) -> dict:
    """Control T's 'every error within tau' (protocol 7): on its deterministic u = 0 runs
    with a verified key, |v_e - v_e*|, |v_min - v_min*| and |d_b - d_b*| within the key's
    tolerances tau (6.3: max(floor, spread across the near-equal set)). A measure undefined
    on both run and key (no brake on a straight) is no error."""
    det = (_st(runs, "mode") == "det") & (_it(runs, "phase", 0) == 0)
    k = _join_keys(runs, keys, ("v_e_star", "v_min_star", "d_b_star", "tau_ve", "tau_vmin",
                                "tau_db", "key_status"))
    ok_key = k["key_status"].astype(str) == "verified"
    m = det & ok_key
    floors = MULTI_BASIN_FLOORS
    worst: dict[str, float] = {}
    beyond = np.zeros(_n(runs), bool)
    for col, star, tau_col, floor in (("v_e", "v_e_star", "tau_ve", floors["v_e"]),
                                      ("v_min", "v_min_star", "tau_vmin", floors["v_min"]),
                                      ("d_b", "d_b_star", "tau_db", floors["d_b"])):
        a, b = _fl(runs, col), k[star].astype(float)
        tau = np.where(np.isfinite(k[tau_col].astype(float)), k[tau_col].astype(float), floor)
        both_nan = np.isnan(a) & np.isnan(b)
        err = np.where(both_nan, 0.0, np.abs(a - b))
        bad = m & ~(err <= tau)                 # NaN on one side only counts as beyond
        beyond |= bad
        worst[col] = float(np.nanmax(np.where(m, err / tau, np.nan), initial=0.0)) if m.any() \
            else float("nan")
    return dict(passed=bool(m.any() and not beyond.any()), n=int(m.sum()),
                n_beyond=int(beyond.sum()), worst_over_tau=worst,
                beyond_cells=sorted({int(c) for c in _it(runs, "cell")[beyond]})[:20])


def p5(runs: Table, keys: Table) -> Decision:
    """P5 symmetry (sanity control): median over pairs of |Δe_ve| <= 2 points and |Δd_b| <=
    5 m between the left and right members, and deterministic crash concordance >= 90%
    over mirrored (pair, phase) runs. Also the right-minus-left e_ve at R <= 18.3 m
    (specialist asymmetry, descriptive)."""
    agent = _agent(runs)
    det = _st(runs, "mode") == "det"
    if not det.any():
        return _d("P5", agent, "not testable")
    r = select(runs, det)
    kd = _join_keys(r, keys, ("direction", "R_nom", "key_status"))
    left = kd["direction"].astype(float) > 0
    e = _run_speed(r, keys, "e_ve")
    db = np.where(_bo(r, "has_brake") & ~_bo(r, "forced_onset"), _fl(r, "d_b"), np.nan)
    pair, phase, crashed = _it(r, "pair"), _it(r, "phase", 0), _bo(r, "crashed")
    dve, ddb, conc, conc_c, asym = [], [], [], [], []
    for p in np.unique(pair):
        mp = pair == p
        L, R = mp & left, mp & ~left
        if not (L.any() and R.any()):
            continue
        eL, eR = _median(e[L]), _median(e[R])
        if np.isinf(eL) and np.isinf(eR):
            dve.append(0.0)
        elif not (np.isnan(eL) or np.isnan(eR)):
            dve.append(abs(eL - eR))
            if kd["R_nom"].astype(float)[mp][0] <= 18.3:
                asym.append(eR - eL)
        need = math.ceil(5 * L.sum() / 8)
        if np.isfinite(db[L]).sum() >= need and np.isfinite(db[R]).sum() >= need:
            ddb.append(abs(_median(db[L]) - _median(db[R])))
        for k in np.unique(phase[mp]):
            cl, cr = crashed[L & (phase == k)], crashed[R & (phase == k)]
            if cl.size and cr.size:
                conc.append(bool(cl.any()) == bool(cr.any()))
                if cl.any() or cr.any():
                    conc_c.append(bool(cl.any()) and bool(cr.any()))
    if not dve:                     # no pair with both members' e_ve (no keys): not judged
        return _d("P5", agent, "not testable", n_pairs_concordance=len(conc))
    m_dve, m_ddb, c = _median(dve), _median(ddb), _frac(np.array(conc, bool))
    ok = (np.isfinite(m_dve) and m_dve <= P5_DVE_PP and (np.isnan(m_ddb) or m_ddb <= P5_DDB_M)
          and np.isfinite(c) and c >= P5_CONCORDANCE)
    return _d("P5", agent, "confirm" if ok else "refute", median_abs_dve=m_dve,
              median_abs_ddb=m_ddb, concordance=c,
              concordance_given_crash=_frac(np.array(conc_c, bool)), n_pairs=len(dve),
              right_minus_left_e_ve_tight=_median(asym))


# ------------------------------------------------------------------------- P6


def checkpoint_gate(pairs: Table) -> dict:
    """Protocol 8's gate: deterministic completion >= 90% on the gentle cells (tier A
    controls and R 87.3 at A1 and A2) and, separately, >= 90% on the tier G width-12
    straights, and the median oscillation index of those runs < 0.1. Each set must have
    runs and the oscillation index must be defined, else the gate is not passed
    (``evaluable`` False)."""
    R, W = _fl(pairs, "R_nom"), _fl(pairs, "width")
    gentle = has_tier(pairs, "A") & (np.isin(R, R_CONTROLS) | np.isclose(R, GENTLE_R)) \
        & np.isin(_st(pairs, "approach"), ("A1", "A2"))
    null = has_tier(pairs, "G") & np.isclose(W, 12.0)
    keep = ~np.isin(_st(pairs, "excluded"), EXCL_ALL)
    nr_all, dc_all = _it(pairs, "n_runs", 0), _it(pairs, "det_complete", 0)

    def comp(mask: np.ndarray) -> float:
        nr = nr_all[mask & keep].sum()
        return float(dc_all[mask & keep].sum() / nr) if nr else float("nan")

    c_gentle, c_null = comp(gentle), comp(null)
    m = (gentle | null) & keep
    osc = _median(_fl(pairs, "osc_index")[m])
    evaluable = bool(np.isfinite(c_gentle) and np.isfinite(c_null) and np.isfinite(osc))
    passed = bool(evaluable and c_gentle >= GATE_COMPLETION and c_null >= GATE_COMPLETION
                  and osc < GATE_OSC)
    return dict(completion=comp(gentle | null), completion_gentle=c_gentle,
                completion_null=c_null, osc=osc, passed=passed, evaluable=evaluable,
                n_pairs=int(m.sum()))


def competence(pairs: Table, b: float) -> dict[float, bool]:
    """Per tier A R level at A2: every deterministic run of every theta-pair completes and
    every pair is in band (P6)."""
    m = _mask(pairs, approach="A2", braking=None)
    side, R = sides(pairs, b), _fl(pairs, "R_nom")
    nr, comp = _it(pairs, "n_runs", 0), _it(pairs, "det_complete", 0)
    out = {}
    for r in np.unique(R[m & np.isfinite(R)]):
        rows = m & (R == r)
        out[float(r)] = bool(np.all((nr[rows] > 0) & (comp[rows] == nr[rows])
                                    & (side[rows] == "in")))
    return out


def t_r(competent: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    """First checkpoint from which competence holds at every later one, per R (inf if
    never; ranks with ties at mid-ranks and inf at the maximum rank: ``t_r_ranks``).
    ``competent`` is [n_checkpoints, n_R] bool."""
    comp = np.asarray(competent, bool)
    ck = np.asarray(checkpoints, float)
    order = np.argsort(ck)
    comp, ck = comp[order], ck[order]
    out = np.full(comp.shape[1], np.inf)
    for r in range(comp.shape[1]):
        suffix = np.logical_and.accumulate(comp[::-1, r])[::-1]
        if suffix.any():
            out[r] = ck[int(np.argmax(suffix))]
    return out


def t_r_ranks(t: np.ndarray) -> np.ndarray:
    """Mid-ranks of t_R; never-competent levels (inf) share the top ranks."""
    return stats.rankdata(np.asarray(t, float))


def p6(pairs_by_ckpt: dict[int, Table], b: float, gate: dict[int, bool] | None = None,
       log_std: dict[int, Any] | None = None, v_top: dict[int, float] | None = None
       ) -> list[Decision]:
    """P6a (Spearman rho(R, t_R) <= -0.5, permutation p <= 0.05; refuted if rho >= 0), P6b
    (caution first), P6c (overspeed appears no earlier than t_bulk), P6d (large R before
    mid R), for one generalist's checkpoints."""
    cks = sorted(pairs_by_ckpt)
    agent = _agent(pairs_by_ckpt[cks[-1]]) if cks else ""
    if len(cks) < 2:
        return [_d(p, agent, "not testable") for p in ("P6a", "P6b", "P6c", "P6d")]
    gate = gate or {c: checkpoint_gate(pairs_by_ckpt[c])["passed"] for c in cks}
    comps = {c: competence(pairs_by_ckpt[c], b) for c in cks}
    levels = sorted({r for d in comps.values() for r in d})
    mat = np.array([[bool(gate.get(c, False)) and comps[c].get(r, False) for r in levels]
                    for c in cks], bool)
    t = t_r(mat, np.array(cks, float))
    rho, p = permutation_spearman(np.array(levels), t)
    covs = dict(std={int(c): np.exp(np.asarray(v, float)).tolist() for c, v in
                     (log_std or {}).items()}, v_top={int(c): v for c, v in (v_top or {}).items()})
    tr = {float(r): float(x) for r, x in zip(levels, t, strict=True)}
    if not np.isfinite(rho):
        d6a = _d("P6a", agent, "not testable", t_R=tr)
    else:
        v6a = "confirm" if rho <= P6A_RHO and p <= P6A_ALPHA else ("refute" if rho >= 0
                                                                     else "inconclusive")
        d6a = _d("P6a", agent, v6a, rho=rho, p=p, t_R=tr, ranks=t_r_ranks(t).tolist())
    # P6b
    first = next((c for c in cks if gate.get(c, False)), None)
    if first is None:
        d6b = _d("P6b", agent, "not testable", reason="no checkpoint passes the gate")
    else:
        pf = pairs_by_ckpt[first]
        mb = _mask(pf, approach="A2", regimes=("bulk",))
        med = _median(band_errors(pf)[1][mb])
        d6b = _d("P6b", agent, "not testable" if np.isnan(med) else
                 ("confirm" if med < -b else "refute"), checkpoint=first,
                 median_bulk_e_ve_bo=med, **covs)
    # P6c
    tail = {c: _median(band_errors(pairs_by_ckpt[c])[0][_mask(pairs_by_ckpt[c], approach="A1",
                                                                regimes=("tail",))])
            for c in cks}
    final = pairs_by_ckpt[cks[-1]]
    bulk_R = set(_fl(final, "R_nom")[_mask(final, approach="A2", braking=None,
                                            regimes=("bulk",))].tolist())
    t_bulk = max((tr[r] for r in tr if r in bulk_R), default=np.nan)
    if not (tail[cks[-1]] > b):
        d6c = _d("P6c", agent, "not applicable", tail_e_ve_ra=tail)
    else:
        c_over = next(c for c in cks if tail[c] > b)
        if not np.isfinite(t_bulk):
            v6c = "inconclusive"
        else:
            v6c = "confirm" if c_over >= t_bulk else "refute"
        d6c = _d("P6c", agent, v6c, first_overspeed=c_over, t_bulk=t_bulk, tail_e_ve_ra=tail,
                 **covs)
    # P6d
    large = [tr[r] for r in tr if any(np.isclose(r, P6D_LARGE))]
    mid = [tr[r] for r in tr if any(np.isclose(r, P6D_MID))]
    ml, mm = (float(np.median(x)) if x else np.nan for x in (large, mid))
    d6d = _d("P6d", agent, "descriptive" if large and mid else "not testable",
             median_t_large=ml, median_t_mid=mm,
             large_first=bool(ml < mm) if large and mid else None)
    return [d6a, d6b, d6c, d6d]


# ------------------------------------------------------------------------- P7


def p7_null(key_sd: float | None, qp_sd: float, v_b: float) -> float:
    """max(key SD across its phase-matched solutions (R <= 18.3), Qp's SD, v_b dt / sqrt 12)."""
    vals = [float(v_b) * DT / math.sqrt(12.0)]
    for v in (key_sd, qp_sd):
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return max(vals)


def phase_spread(values) -> float:
    """P7's spread of d_b across a cell's phases: the SD over the fixed phase grid (ddof 0),
    one estimator for the agent's 16 phases, Qp's and the key's 4 phase-matched solutions,
    so the null and the agent are on the same scale. NaN with fewer than 2 values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.std(v, ddof=0)) if v.size >= 2 else float("nan")


def _phase_sd(runs: Table, mask: np.ndarray) -> tuple[float, float]:
    """(SD of d_b across phases, mean over directions; median v_b) over ``mask``'s runs."""
    db = np.where(_bo(runs, "has_brake") & ~_bo(runs, "forced_onset"), _fl(runs, "d_b"), np.nan)
    sds = []
    for c in np.unique(_it(runs, "cell")[mask]):
        v = db[mask & (_it(runs, "cell") == c)]
        v = v[np.isfinite(v)]
        if v.size >= 2:
            sds.append(phase_spread(v))
    return (float(np.mean(sds)) if sds else float("nan")), _median(_fl(runs, "v_b")[mask])


def _p7_runs(runs: Table, keys: Table) -> tuple[np.ndarray, np.ndarray, dict]:
    """(mask of deterministic runs on 16-phase A1 cells, their geometry, joined key columns)."""
    k = _join_keys(runs, keys, ("geometry", "R_nom", "theta_deg", "n_phases", "approach"))
    det = (_st(runs, "mode") == "det") & (k["n_phases"].astype(float) >= 16) \
        & (k["approach"].astype(str) == "A1")
    return det, k["geometry"].astype(str), k


def phase_sd_table(runs: Table, keys: Table, qp_runs: Table | None = None,
                   key_sd: dict[str, float] | None = None) -> Table:
    """P7_SD_COLUMNS for the 16-phase A1 cells: the agent's and Qp's d_b SD across phases
    and the null. ``key_sd``: geometry -> the key's SD across its phase-matched solutions."""
    det, geo, k = _p7_runs(runs, keys)
    if qp_runs is not None:
        dq, gq, _ = _p7_runs(qp_runs, keys)
    rows = []
    for g in np.unique(geo[det]):
        m = det & (geo == g)
        sd_a, v_b = _phase_sd(runs, m)
        sd_q = _phase_sd(qp_runs, dq & (gq == g))[0] if qp_runs is not None else np.nan
        R, th = float(k["R_nom"].astype(float)[m][0]), float(k["theta_deg"].astype(float)[m][0])
        sd_k = (key_sd or {}).get(g, np.nan) if R <= 18.3 else np.nan
        null = p7_null(sd_k, sd_q, v_b if np.isfinite(v_b) else 0.0)
        rows.append(dict(agent=_agent(runs), geometry=str(g), R_nom=R, theta_deg=th,
                         sd_agent=sd_a, sd_key=float(sd_k), sd_qp=float(sd_q), v_b=v_b,
                         sd_null=null, excess=sd_a - null))
    return _to_table(rows, P7_SD_COLUMNS)


def p7(phase_sd: Table, alias: Table) -> Decision:
    """Confirm 'sampling matters': excess >= 2 m in >= 75% of short tight cells (90 deg, R <=
    14.6) while <= 1 m in >= 75% of the 180 deg cells at the same R, and median A_alias on
    short cells >= 2x Qp's. Refute: excess <= 1 m in >= 75% of all cells and A_alias no
    larger than Qp's. Mixed: that excess is absent but A_alias exceeds Qp's."""
    agent = _agent(phase_sd) or _agent(alias)
    R, th, ex = _fl(phase_sd, "R_nom"), _fl(phase_sd, "theta_deg"), _fl(phase_sd, "excess")
    short = np.isclose(th, 90.0) & (R <= P7_SHORT_R)
    long_ = np.isclose(th, 180.0) & (R <= P7_SHORT_R)
    if not short.any() or not _n(alias):
        return _d("P7", agent, "not testable")
    f_short = _frac(ex[short] >= P7_EXCESS_M)
    f_long = _frac(ex[long_] <= P7_FLAT_M)
    f_all = _frac(ex <= P7_FLAT_M)
    s = _bo(alias, "short")
    a_s, q_s = _median(_fl(alias, "a_agent")[s]), _median(_fl(alias, "a_qp")[s])
    a_all, q_all = _median(_fl(alias, "a_agent")), _median(_fl(alias, "a_qp"))
    closed = f_short >= P7_FRACTION and f_long >= P7_FRACTION
    alias_big = bool(np.isfinite(a_s) and np.isfinite(q_s) and a_s >= P7_ALIAS_RATIO * q_s)
    absent = bool(np.isfinite(f_all) and f_all >= P7_FRACTION)
    det = dict(frac_short_excess=f_short, frac_long_flat=f_long, frac_all_flat=f_all,
               alias_short=a_s, alias_qp_short=q_s, alias_all=a_all, alias_qp_all=q_all)
    if closed and alias_big:
        verdict = "confirm"
    elif absent and np.isfinite(a_all) and np.isfinite(q_all):
        verdict = "refute" if a_all <= q_all else "mixed"
    else:
        verdict = "inconclusive"
    return _d("P7", agent, verdict, **det)


# ------------------------------------------------------------------------- P8


def _reproduces(det_crashes: int, nat_k: int, nat_n: int) -> bool:
    return bool(det_crashes >= P8_REPRO_DET or (nat_n > 0 and nat_k / nat_n >= P8_REPRO_PCRASH))


def _suppressed(det_crashes: int, nat_k: int, nat_n: int) -> bool:
    return bool(det_crashes == 0 and (nat_n == 0 or nat_k / nat_n <= P8_FAIL_PCRASH))


def p8_units(ladder: Table, b: float = BAND_EXPECTED, knife_edge: tuple[str, ...] = ()
             ) -> Table:
    """Per (agent, crash unit): reproduced, single-corner or which context factor is
    necessary, and twin agreement (P8). A unit in ``knife_edge`` (3.2's YasMarina diagnosis:
    real-track P_crash < 0.5) stays in P8 as unreproduced. ``fidelity_missing``: the unit's
    fidelity gate was not evaluated (no real rung, or the gate's inputs are missing); it is
    unreproduced but did not fail the gate (``p8`` keeps it in P8-tail's n)."""
    unit, ver = _st(ladder, "unit"), _st(ladder, "version")
    rows = []
    for u in np.unique(unit):
        mu = unit == u

        def rung(v: str, mu: np.ndarray = mu) -> int | None:
            i = np.flatnonzero(mu & (ver == v))
            return int(i[0]) if i.size else None

        def crash(i: int | None, fn=_reproduces) -> bool:
            if i is None:
                return False
            return fn(int(_it(ladder, "det_crashes", 0)[i]), int(_it(ladder, "nat_k", 0)[i]),
                       int(_it(ladder, "nat_n", 0)[i]))

        real = rung("real")
        fid = bool(_bo(ladder, "fidelity_ok")[real]) if real is not None else False
        missing = real is None or (not fid and bool(_bo(ladder, "fidelity_missing")[real]))
        knife = str(u) in set(knife_edge)
        repro = fid and crash(real) and not knife
        single = repro and crash(rung("isolated"))
        necessary = [f for f in ("before_straight", "after_straight", "const_width")
                     if repro and not single and crash(rung(f), _suppressed)]
        tw = rung("twin")
        twin_agree = bool(single and tw is not None and (
            crash(tw) or _fl(ladder, "e_ve")[tw] > b or _it(ladder, "det_overspeed", 0)[tw] > 0))
        if missing:
            label = "fidelity not evaluated (unreproduced)"
        elif not fid:
            label = "fidelity failed"
        elif knife:
            label = "knife-edge (unreproduced)"
        elif not repro:
            label = "not reproduced"
        elif single:
            label = "single-corner"
        else:
            label = "context: " + ",".join(necessary) if necessary else "joint context"
        rows.append(dict(agent=str(_st(ladder, "agent")[mu][0]), unit=str(u),
                         set=str(_st(ladder, "set")[mu][0]), fidelity_ok=fid,
                         fidelity_missing=missing, reproduced=repro,
                         single_corner=single, twin_agree=twin_agree, label=label))
    return _to_table(rows, ("agent", "unit", "set", "fidelity_ok", "fidelity_missing",
                            "reproduced", "single_corner", "twin_agree", "label"))


def tier_a_curve_at(pairs: Table, R_seg: float, theta_deg: float, approach: str) -> float:
    """The agent's tier A e_ve curve (band errors) at a twin's R_seg, theta class and
    approach: linear in log R_seg over the finite pair values of that class, clamped."""
    ge = theta_deg >= THETA_CLASS_EDGE_DEG
    th = _fl(pairs, "theta_deg")
    m = _mask(pairs, approach=approach) & ((th >= THETA_CLASS_EDGE_DEG) == ge)
    e = band_errors(pairs)[0]
    x, y = np.log(_fl(pairs, "R_seg")[m]), e[m]
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    o = np.argsort(x[ok])
    return float(np.interp(math.log(R_seg), x[ok][o], y[ok][o]))


def p8_tail_thresholds(n: int) -> tuple[int, int]:
    """Ruling R2: (confirm minimum ceil(0.6 n), refute maximum floor(0.2 n)) for n tail
    units, in integer arithmetic; n = 5 gives the protocol's original 3 and 1."""
    (cn, cd), (rn, rd) = P8_TAIL_CONFIRM, P8_TAIL_REFUTE
    return -(-cn * n // cd), rn * n // rd


def p8_bulk_minimum(n: int) -> int:
    """Ruling R2's treatment of the bulk set: context in at least ceil(2n/3) of the n
    reproduced bulk units."""
    num, den = P8_BULK_CONTEXT
    return -(-num * n // den)


def p8(ladder: Table, twins: Table | None = None, b: float = BAND_EXPECTED,
       knife_edge: tuple[str, ...] = ()) -> list[Decision]:
    """P8-tail (G2's tail units, n of them after transplants failing the fidelity gate are
    excluded (3.2; knife-edge units and units whose gate could not be evaluated stay, as
    unreproduced, and are listed): confirm if >= ceil(0.6 n) are reproduced and
    >= ceil(0.6 n) are single-corner with twin agreement; refute if at most floor(0.2 n)
    are; not testable if n < 3 (ruling R2)), P8-bulk (context in >= 2/3, i.e.
    >= ceil(2n/3), of the n reproduced bulk units; not testable only when none is
    reproduced: R2 changes the tail rule only), P8-F per agent (descriptive F) and P8-twin
    per agent (twin transfer, with the twins' length shortfall as a covariate, R4). The sets
    are the units' regimes from their crash corners' R_seg (``p8_set``)."""
    out = []
    u = p8_units(ladder, b, knife_edge)
    ag, st = _st(u, "agent"), _st(u, "set")
    rep, sing, tw = _bo(u, "reproduced"), _bo(u, "single_corner"), _bo(u, "twin_agree")
    fid, missing = _bo(u, "fidelity_ok"), _bo(u, "fidelity_missing")
    names = _st(u, "unit")
    tail_all = (ag == "G2") & (st == "tail")
    gate_failed = tail_all & ~fid & ~missing
    tail = tail_all & ~gate_failed
    n_tail = int(tail.sum())
    failed = names[gate_failed].tolist()
    not_evaluated = names[tail & missing].tolist()
    if n_tail < P8_MIN_UNITS:
        out.append(_d("P8-tail", "G2", "not testable", n_units=n_tail,
                      min_units=P8_MIN_UNITS, fidelity_failed=failed,
                      fidelity_missing=not_evaluated))
    else:
        n_rep, n_ok = int(rep[tail].sum()), int((sing & tw)[tail].sum())
        need, refute_max = p8_tail_thresholds(n_tail)
        verdict = "confirm" if n_rep >= need and n_ok >= need else (
            "refute" if n_ok <= refute_max else "inconclusive")
        out.append(_d("P8-tail", "G2", verdict, n_units=n_tail, reproduced=n_rep,
                      single_with_twin=n_ok, confirm_min=need, refute_max=refute_max,
                      fidelity_failed=failed, fidelity_missing=not_evaluated,
                      labels=dict(zip(names[tail].tolist(),
                                      _st(u, "label")[tail].tolist(), strict=True))))
    bulk = (st == "bulk") & rep
    n_bulk = int(bulk.sum())
    if not n_bulk:
        out.append(_d("P8-bulk", "G1,G2", "not testable", n_units=int((st == "bulk").sum()),
                      n_reproduced=0))
    else:
        n_ctx = int((~sing[bulk]).sum())
        need = p8_bulk_minimum(n_bulk)
        out.append(_d("P8-bulk", "G1,G2", "confirm" if n_ctx >= need else "refute",
                      n_reproduced=n_bulk, n_context=n_ctx, confirm_min=need,
                      frac_context=n_ctx / n_bulk))
    for a in np.unique(ag):
        ma = ag == a
        n_rep = int(rep[ma].sum())
        F = float((sing & tw)[ma].sum() / n_rep) if n_rep else float("nan")
        out.append(_d("P8-F", str(a), "descriptive" if n_rep else "not testable", F=F,
                      reproduced=n_rep, units=int(ma.sum())))
    if twins is not None and _n(twins):
        for a in np.unique(_st(twins, "agent")):
            m = _st(twins, "agent") == a
            iso, twn = _fl(twins, "e_ve_isolated")[m], _fl(twins, "e_ve_twin")[m]
            both_inf = np.isinf(iso) & np.isinf(twn) & (np.sign(iso) == np.sign(twn))
            with np.errstate(invalid="ignore"):
                agree = both_inf | (np.abs(iso - twn) <= P8_TWIN_PP)
            defined = both_inf | (np.isfinite(iso) & np.isfinite(twn))
            f_agree = _frac(agree[defined])
            rho = spearman(_fl(twins, "e_ve_curve")[m], _fl(twins, "e_ve_real")[m])
            ok_a = np.isfinite(f_agree) and f_agree >= P8_TWIN_FRACTION
            ok_r = np.isfinite(rho) and rho >= P8_SPEARMAN
            verdict = "confirm" if ok_a and ok_r else ("refute" if not (ok_a or ok_r)
                                                       else "inconclusive")
            ctx = _median(_fl(twins, "e_ve_real")[m] - _fl(twins, "e_ve_isolated")[m])
            out.append(_d("P8-twin", str(a), verdict, frac_agree=f_agree, spearman=rho,
                          n=int(m.sum()), context_effect=ctx,
                          **twin_shortfall_covariate(_fl(twins, "length_shortfall")[m],
                                                     agree, defined, iso - twn)))
    return out


def twin_shortfall_covariate(short: np.ndarray, agree: np.ndarray, defined: np.ndarray,
                             diff: np.ndarray) -> dict:
    """Ruling R4: the twins' length shortfall (real L_seg minus the twin's, m) reported as a
    covariate of the twin transfer: how many twins miss the 2 m length tolerance, the
    agreement fraction among them and among the rest, and the Spearman correlation of the
    shortfall with |e_ve(isolated) - e_ve(twin)| over corners where both are finite. The
    verdict does not use it."""
    short = np.asarray(short, float)
    with np.errstate(invalid="ignore"):
        long_miss = short > TWIN_TOL["length"]
    fin = np.isfinite(short) & np.isfinite(diff)
    rho = spearman(short[fin], np.abs(diff[fin])) if fin.sum() >= 3 else float("nan")
    return dict(length_shortfall_median=_median(short), n_length_short=int(long_miss.sum()),
                frac_agree_length_short=_frac(agree[defined & long_miss]),
                frac_agree_length_ok=_frac(agree[defined & ~long_miss & np.isfinite(short)]),
                spearman_shortfall_disagreement=rho)


# ------------------------------------------------------------------------- descriptive (9.1, 9.3)


def viability_targets(runs: Table, keys: Table, cap: int = VIABILITY_CAP,
                      subset: int = VIABILITY_SUBSET,
                      rng: np.random.Generator | None = None) -> Table:
    """Protocol 9.1: the runs that get viability solves. Per crashed deterministic cell, its
    u = 0 run if that crashed, else its first crashed phase; each costs one solve per tested
    state before the crash (s_b, s_c0 - 50, s_c0, s_a0). If the total exceeds ``cap``, a
    seeded subset (``io.stream_rng('viability')``) of ``subset`` crashed pairs per agent is
    kept. ``runs`` may hold several agents and snapshots."""
    det = (_st(runs, "mode") == "det") & _bo(runs, "crashed")
    k = _join_keys(runs, keys, ("s_c0", "s_a0", "key_status"))
    det &= ~np.isin(k["key_status"].astype(str), EXCL_ALL + (NO_KEY_FLAG,))     # R7
    agent, ts, cell = _st(runs, "agent"), _it(runs, "timesteps", 0), _it(runs, "cell")
    phase, pair, s_crash = _it(runs, "phase", 0), _it(runs, "pair"), _fl(runs, "s_crash")
    sample = _it(runs, "sample")
    s_b, s_c0, s_a0 = _fl(runs, "s_b"), k["s_c0"].astype(float), k["s_a0"].astype(float)
    best: dict[tuple, int] = {}
    for i in np.flatnonzero(det):
        key = (agent[i], int(ts[i]), int(cell[i]))
        if key not in best or phase[i] < phase[best[key]]:
            best[key] = int(i)
    rows = []
    for (a, t, c), i in sorted(best.items()):
        states = [s_b[i], s_c0[i] - 50.0, s_c0[i], s_a0[i]]
        n = int(sum(1 for x in states if np.isfinite(x) and x < s_crash[i]))
        rows.append(dict(agent=a, timesteps=t, cell=c, pair=int(pair[i]), phase=int(phase[i]),
                         sample=int(sample[i]), n_solves=n))
    out = _to_table(rows, ("agent", "timesteps", "cell", "pair", "phase", "sample", "n_solves"))
    if not rows or int(out["n_solves"].sum()) <= cap:
        return out
    rng = stream_rng("viability") if rng is None else rng
    keep = np.zeros(len(rows), bool)
    for a in np.unique(out["agent"]):
        ma = out["agent"] == a
        pairs = np.unique(out["pair"][ma])
        chosen = rng.choice(pairs, size=min(subset, pairs.size), replace=False)
        keep |= ma & np.isin(out["pair"], chosen)
    return select(out, keep)


def cell_rates(pairs: Table, mode: str = "nat", level: float = 0.95) -> Table:
    """Per-cell crash rates with exact intervals (9.3). One cell is one cluster, where the
    beta-binomial reduces to the binomial, so the interval is Clopper-Pearson. ``mode``:
    'nat', 'mat' or 'det'."""
    k = _it(pairs, f"{mode}_k" if mode != "det" else "det_crashes", 0)
    n = _it(pairs, f"{mode}_n" if mode != "det" else "n_runs", 0)
    rows = []
    for i in range(_n(pairs)):
        lo, hi = clopper_pearson(int(k[i]), int(n[i]), level)
        rows.append(dict(agent=str(pairs["agent"][i]), pair=int(pairs["pair"][i]),
                         k=int(k[i]), n=int(n[i]),
                         p=float(k[i] / n[i]) if n[i] else np.nan, lo=lo, hi=hi))
    return _to_table(rows, ("agent", "pair", "k", "n", "p", "lo", "hi"))


def r50_table(pairs: Table, mode: str = "nat") -> Table:
    """R50 per (agent, snapshot, v0, D, theta) over the braking pairs (9.3): the empirical
    bracket between grid values (primary) and the Firth fit on log R_seg (secondary).
    Infeasible and unsolved cells are excluded from P_crash (6.4)."""
    m = ~np.isin(_st(pairs, "excluded"), EXCL_ALL) & np.isfinite(_fl(pairs, "R_seg"))
    k = _it(pairs, f"{mode}_k" if mode != "det" else "det_crashes", 0)
    n = _it(pairs, f"{mode}_n" if mode != "det" else "n_runs", 0)
    m &= n > 0
    groups: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(m):
        key = (str(pairs["agent"][i]), int(_it(pairs, "timesteps", 0)[i]),
               round(float(pairs["v0"][i]), 3), round(float(pairs["D"][i]), 1),
               float(pairs["theta_deg"][i]))
        groups.setdefault(key, []).append(int(i))
    rows = []
    for (a, t, v0, D, th), idx in sorted(groups.items()):
        idx = np.asarray(idx)
        out = r50(_fl(pairs, "R_seg")[idx], k[idx] / n[idx], n[idx])
        rows.append(dict(agent=a, timesteps=t, v0=v0, D=D, theta_deg=th, n_R=int(idx.size),
                         bracket_lo=float(out["bracket"][0]), bracket_hi=float(out["bracket"][1]),
                         position=out["position"], monotone=bool(out["monotone"]),
                         firth_R50=float(out["firth_R50"])))
    return _to_table(rows, ("agent", "timesteps", "v0", "D", "theta_deg", "n_R", "bracket_lo",
                            "bracket_hi", "position", "monotone", "firth_R50"))


def slack(pairs: Table, keys: Table, tables) -> np.ndarray:
    """Tier B's slack axis (9.3): (D - d_b^phys(v0, v_e*(A1))) / v0 in seconds, with v_e*(A1)
    the A1 key's entry speed for the same geometry."""
    a1 = {str(g): float(v) for g, v, ap in zip(_st(keys, "geometry"), _fl(keys, "v_e_star"),
                                               _st(keys, "approach"), strict=True) if ap == "A1"}
    out = np.full(_n(pairs), np.nan)
    for i, (g, v0, D) in enumerate(zip(_st(pairs, "geometry"), _fl(pairs, "v0"),
                                       _fl(pairs, "D"), strict=True)):
        ve = a1.get(str(g), np.nan)
        if np.isfinite(ve) and np.isfinite(v0) and v0 > 0:
            out[i] = (D - float(tables.d_b_phys(float(v0), ve))) / v0
    return out


def support_correlations(pairs: Table, b: float = BAND_EXPECTED) -> Table:
    """Spearman correlation of |e_ve| and native-noise P_crash with the log joint-support
    density, and with the tight-and-fast density, across an agent's tier A braking pairs
    (9.3)."""
    rows = []
    e = np.abs(band_errors(pairs)[0])
    pc = np.where(_it(pairs, "nat_n", 0) > 0,
                  _it(pairs, "nat_k", 0) / np.maximum(_it(pairs, "nat_n", 0), 1), np.nan)
    for a in np.unique(_st(pairs, "agent")):
        m = (_st(pairs, "agent") == a) & _mask(pairs, exclude=EXCL_ALL)
        for cov in ("log_support", "log_support_tightfast"):
            x = _fl(pairs, cov)[m]
            for name, y in (("abs_e_ve", e[m]), ("p_crash", pc[m])):
                ok = np.isfinite(x) & ~np.isnan(y)
                rows.append(dict(agent=str(a), covariate=cov, measure=name,
                                 rho=spearman(x[ok], y[ok]), n=int(ok.sum())))
    return _to_table(rows, ("agent", "covariate", "measure", "rho", "n"))


def v_top(runs: Table, keys: Table, width: float = 12.0) -> float:
    """Tier G top speed of a snapshot (P6 covariate): the largest deterministic vx over the
    whole run (``v_max``) on the width-``width`` null straights. (v_peak on a straight stops
    at the virtual corner start S_C and includes the start state, so it measures v0.)"""
    k = _join_keys(runs, keys, ("tiers", "width"))
    g = np.array(["G" in t.split(",") for t in k["tiers"].astype(str)], bool)
    m = (_st(runs, "mode") == "det") & g & np.isclose(k["width"].astype(float), width)
    v = _fl(runs, "v_max")[m]
    return float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")


def reanchor_immaterial(finals: list[Table]) -> dict:
    """P6's exception: re-anchoring is immaterial if |v_e*RA - v_e*| <= 1 km/h in >= 90% of
    the generalist finals' A1 cells; the checkpoints then use nominal keys. v_e* is the key
    the checkpoints would use instead: the u = 0 key, or the runs' phase-matched keys
    (v_e_star_k) when the speed errors are phase-matched (AD23's V7 remedy)."""
    d = []
    for p in finals:
        ref = np.where(_bo(p, "speed_at_phase"), _fl(p, "v_e_star_k"), _fl(p, "v_e_star"))
        m = (_st(p, "approach") == "A1") & np.isfinite(_fl(p, "v_e_star_ra")) & np.isfinite(ref)
        d.append(np.abs(_fl(p, "v_e_star_ra") - ref)[m])
    dd = np.concatenate(d) if d else np.zeros(0)
    frac = _frac(dd <= kph(REANCHOR_IMMATERIAL_KPH))
    return dict(frac=frac, n=int(dd.size),
                immaterial=bool(np.isfinite(frac) and frac >= REANCHOR_IMMATERIAL_FRACTION))


def descriptive_tables(inp: AnalysisInputs) -> dict[str, Table]:
    """Step 8's descriptive tables per final: R50 (native noise and deterministic), per-cell
    crash intervals and the support correlations."""
    out: dict[str, Table] = {}
    for a, runs in inp.runs.items():
        pairs = _variants(runs, inp)["all"][0]
        out[f"{a}_r50_nat"] = r50_table(pairs, "nat")
        out[f"{a}_r50_det"] = r50_table(pairs, "det")
        out[f"{a}_cell_rates"] = cell_rates(pairs, "nat")
        out[f"{a}_support"] = support_correlations(pairs, inp.b)
        out[f"{a}_pairs"] = pairs
    return out


# ------------------------------------------------------------------------- decision table


@dataclass
class AnalysisInputs:
    """Everything the decision table reads. Run tables are det (+ nat/mat) RUN_COLUMNS
    tables of one snapshot; pairs are recomputed from them so the 9.4 variants match."""

    b: float
    keys: Table
    regimes: dict
    runs: dict[str, Table]                                       # agent -> final snapshot
    history: dict[str, list[Table]] = field(default_factory=dict)  # two snapshots before
    tables: Any = None                                           # measures.ModelTables (P3)
    conds: dict[str, Table] = field(default_factory=dict)        # agent -> tier D (finals)
    checkpoints: dict[str, dict[int, Table]] = field(default_factory=dict)  # P6 run tables
    gates: dict[str, dict[int, bool]] = field(default_factory=dict)
    log_std: dict[str, dict[int, Any]] = field(default_factory=dict)
    v_top: dict[str, dict[int, float]] = field(default_factory=dict)
    p7: dict[str, tuple[Table, Table]] = field(default_factory=dict)  # (phase_sd, alias)
    ladder: Table | None = None
    twins: Table | None = None
    density: Table | None = None
    demote_p3: bool = False          # the pre-tag P3 separation check failed (P3 -> secondary)
    knife_edge: tuple[str, ...] = ()  # P8 units labelled knife-edge (3.2's YasMarina diagnosis)
    _cache: dict = field(default_factory=dict, repr=False)


def _g_rules(inp: AnalysisInputs) -> dict[str, Callable[[Table, Table], Decision]]:
    b = inp.b

    def rule_p3(pairs: Table, runs: Table) -> Decision:
        if inp.tables is None:
            return _d("P3", _agent(pairs), "not testable", reason="no model tables")
        return p3(p3_fit(runs, inp.keys, inp.tables), brake_share(runs, inp.keys))

    return {"P1a": lambda p, r: p1a(p, b), "P1-tail": lambda p, r: p1_tail(p, b),
            "P1b": lambda p, r: p1b(p, b), "P2": lambda p, r: p2(p, r, b), "P3": rule_p3}


def _pv(runs: Table, inp: AnalysisInputs) -> Table:
    return pair_values(runs, inp.keys, inp.regimes, inp.b, inp.density)


def _variants(runs: Table, inp: AnalysisInputs) -> dict[str, tuple[Table, Table]]:
    """(pairs, runs) for all runs and the 9.4 variants without clipped / unsettled runs,
    computed once per run table."""
    key = id(runs)
    if key not in inp._cache:
        v = {"all": (_pv(runs, inp), runs)}
        clip = _clip_other_runs(runs)
        if clip.any():
            r2 = select(runs, ~clip)
            v["noclip"] = (_pv(r2, inp), r2)
        unsettled = _bo(runs, "unsettled")
        if unsettled.any():
            r3 = select(runs, ~unsettled)
            v["settled"] = (_pv(r3, inp), r3)
        inp._cache[key] = (runs, v)          # holding runs keeps its id valid
    return inp._cache[key][1]


def _with_sensitivity(rule: Callable[[Table, Table], Decision], runs: Table,
                      inp: AnalysisInputs) -> Decision:
    """9.4: clipped runs analysed both ways (the version without them is primary when they
    disagree); unsettled runs and lock-limited cells as sensitivity analyses."""
    v = _variants(runs, inp)
    pairs = v["all"][0]
    d = rule(pairs, runs)
    if "noclip" in v:
        d2 = rule(*v["noclip"])
        d.details["without_clipped_runs"] = d2.verdict
        if d2.verdict != d.verdict:
            d2.details.update(with_clipped_runs=d.verdict,
                              primary_version="without clipped runs (9.4)")
            d = d2
    if "settled" in v:
        d.details["without_unsettled_runs"] = rule(*v["settled"]).verdict
    lock = _bo(pairs, "lock_limited")
    if lock.any():
        # Run-based rules (P3) need the runs of lock-limited cells removed too.
        lock_run = _bo(_join_keys(runs, inp.keys, ("lock_limited",)), "lock_limited")
        d.details["without_lock_limited"] = rule(select(pairs, ~lock),
                                                 select(runs, ~lock_run)).verdict
    return d


def _gate_of(runs: Table, inp: AnalysisInputs, agent: str) -> dict:
    """Protocol 8's checkpoint gate of one snapshot's run table: from ``inp.gates`` when the
    stage computed it, else from the pairs."""
    ts = int(_it(runs, "timesteps", 0)[0]) if _n(runs) else 0
    given = inp.gates.get(agent, {}).get(ts)
    if given is not None:
        return dict(passed=bool(given), source="inputs")
    return checkpoint_gate(_variants(runs, inp)["all"][0])


def _gated(d: Decision, gate: dict) -> Decision:
    """A snapshot that fails the gate is not read for P1-P3 (protocol 8; 9.4: crash curves
    only)."""
    if gate.get("passed"):
        return d
    return _d(d.prediction, d.agent, "not testable", reason="checkpoint gate failed",
              gate=gate, ungated_verdict=d.verdict)


def _stabilise(d: Decision, history: list[str]) -> Decision:
    """Protocol 8: a generalist claim holds as a property of the network only if it recurs
    at >= 1 of the two snapshots before the final; else 'unstable'."""
    if not history:
        d.details["stability"] = "n/a"
        return d
    stable = d.verdict in history[:2]
    d.stable = stable
    d.details["history_verdicts"] = history[:2]
    if not stable and d.verdict in CLAIMS:
        d.details["final_verdict"] = d.verdict
        d.verdict = "unstable"
    return d


def _decision_rows(decisions: list[Decision]) -> Table:
    import json

    from apexzero.interp.io import _jsonable

    rows = [dict(prediction=d.prediction, agent=d.agent, family=d.family, verdict=d.verdict,
                 stable="n/a" if d.stable is None else ("yes" if d.stable else "no"),
                 details=json.dumps(_jsonable(d.details), sort_keys=True)) for d in decisions]
    return _to_table(rows, DECISION_COLUMNS)


def decisions(inp: AnalysisInputs, confirmatory: bool = True) -> list[Decision]:
    """Every decision of the protocol that the inputs allow (confirmatory: P1a, P1-tail,
    P1b, P2, P3 per generalist, P4 per specialist and the recipe claim, and G2's P8 tail
    test; otherwise also the secondary, exploratory and control analyses)."""
    out: list[Decision] = []
    rules = _g_rules(inp)
    finals = {a: _variants(r, inp)["all"][0] for a, r in inp.runs.items()}
    for agent in [a for a in GENERALISTS if a in inp.runs]:
        runs = inp.runs[agent]
        hist = inp.history.get(agent, [])
        gate = _gate_of(runs, inp, agent)
        hist_gates = [_gate_of(r, inp, agent) for r in hist]
        for name, rule in rules.items():
            d = _gated(_with_sensitivity(rule, runs, inp), gate)
            past_d = [_gated(_with_sensitivity(rule, r, inp), g)
                      for r, g in zip(hist, hist_gates, strict=True)]
            d = _stabilise(d, [p.verdict for p in past_d])
            if name == "P1b" and past_d:
                # R12 re-anchors the below-support A1 cells at P0 (the finals) only; a
                # history snapshot that is also a P6c checkpoint has re-anchored tail and
                # bulk keys but nominal below-support ones, so its Kendall tau mixes keys.
                # Reported per history snapshot, not acted on (ruling R12's scope).
                d.details["history_ra_fallback_pairs"] = [
                    p.details.get("ra_fallback_pairs") for p in past_d]
            if name == "P3" and inp.demote_p3 and d.family == "primary":
                d.family = "secondary"
                d.details["demoted"] = "pre-tag P3 separation check failed (protocol 2.4 P3)"
            out.append(d)
    gen_conds = [inp.conds[a] for a in GENERALISTS if a in inp.conds]
    gen_table = concat_tables(gen_conds) if gen_conds else None
    p4res = []
    tier_g_crash: dict[str, bool] = {}
    for agent in [a for a in SPECIALISTS if a in inp.conds]:
        own = finals.get(agent)
        tier_a = select(own, has_tier(own, "A")) if own is not None else None
        tier_g = select(own, has_tier(own, "G")) if own is not None else None
        res = p4_classify(inp.conds[agent], gen_table, tier_a, tier_g)
        if res.details.get("tier_g_crash"):
            # Protocol 2.4 P4, tier G rule.
            tier_g_crash[agent] = True
            res.details["tier_g_rule"] = ("crashes on a null straight: evidence for class 1 on "
                                          "synthetic stimuli; tiers A, B, C and E are reported "
                                          "as completion rates only")
        p4res.append(res)
        out.append(p4(res))
    if len(p4res) == len(SPECIALISTS):
        out.append(p4_recipe(p4res))
    if inp.ladder is not None and _n(inp.ladder):
        p8d = p8(inp.ladder, inp.twins, inp.b, knife_edge=inp.knife_edge)
        out += [d for d in p8d if d.prediction == "P8-tail"]
        if not confirmatory:
            out += [d for d in p8d if d.prediction != "P8-tail"]
    if confirmatory:
        return out
    rd = {a: {ap: r_dep(finals[a], ap, inp.b) for ap in ("A1", "A2")}
          for a in GENERALISTS if a in finals}
    out += p1c(rd)
    for agent, pairs in finals.items():
        runs = inp.runs[agent]
        if agent in GENERALISTS:
            out += [p2b(pairs, inp.b), p2d(pairs, inp.b), *p3_beta(runs, inp.keys)]
            if inp.tables is not None:
                out.append(p3_ratio(runs, inp.keys, inp.tables))
        d5 = p5(runs, inp.keys)
        if tier_g_crash.get(agent):
            d5 = _d("P5", agent, "descriptive", reason="tier G rule (P4): completion rates only",
                    rule_verdict=d5.verdict, **d5.details)
        out.append(d5)
    for agent, ck in inp.checkpoints.items():
        out += p6({c: _pv(r, inp) for c, r in ck.items()}, inp.b, inp.gates.get(agent),
                  inp.log_std.get(agent), inp.v_top.get(agent))
    for sd, al in inp.p7.values():
        out.append(p7(sd, al))
    return out


def decision_table(inp: AnalysisInputs, confirmatory: bool = True, out_dir: Path | None = None,
                   overwrite: bool = False) -> Table:
    """Run every confirmatory test once and write analysis/decisions.npz (+ .json).

    With ``confirmatory`` the table is written to decisions.{npz,json} and an existing one is
    never replaced (protocol 10, step 8: one run); otherwise everything is written to
    decisions_all.{npz,json}."""
    out_dir = ANALYSIS_DIR if out_dir is None else Path(out_dir)
    stem = "decisions" if confirmatory else "decisions_all"
    path = out_dir / f"{stem}.npz"
    if confirmatory and path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists: the confirmatory analysis runs once")
    ds = decisions(inp, confirmatory)
    table = _decision_rows(ds)
    save_table(path, table)
    if not confirmatory:
        for name, t in descriptive_tables(inp).items():
            save_table(out_dir / f"{name}.npz", t)
    write_json(out_dir / f"{stem}.json", {"band": inp.b, "confirmatory": confirmatory,
                                          "decisions": ds})
    return table


# ------------------------------------------------------------------------- dry run

DRY_RUN_AGENTS = ("lawful", "overspeed_below_12", "late_onset", "time_to_corner",
                  "context_keyed")
DRY_RUN_EXPECTED = {
    "lawful": {"P1a": "confirm", "P1-tail": "refute", "P1b": "refute", "P2": "not applicable",
               "P3": "confirm"},
    "overspeed_below_12": {"P1a": "confirm", "P1b": "confirm"},
    "late_onset": {"P1a": "confirm", "P1-tail": "confirm", "P2": "confirm", "P2b": "confirm"},
    "time_to_corner": {"P3": "refute:M_idx"},
    "context_keyed": {"P4": "class 2"},
}
DRY_RUN_REGIME = Regime(12.0, 20.0)   # "as if q1 = 12 and q10 = 20" (protocol 7, c2)
SIM_LAWFUL = 0.99                     # lawful: key x 0.99
SIM_SIGMA_V = 0.003                   # phase noise of v_e (relative SD per phase)
SIM_SIGMA_NAT = 0.01                  # stochastic-run noise of v_e (relative SD)
SIM_CRASH_MARGIN = 0.10               # simulated car margin: e_ve above it crashes (overspeed)
SIM_RESP_LEAD = 15.0                  # m: response onset ahead of the brake onset
SIM_OVERSPEED = 0.15                  # 'overspeed_below_12': planted e at R_seg = 12 m
SIM_LATE = 15.0                       # 'late_onset': metres late at A1 (c1's 15 m)
POWER_STEPS = tuple(round(0.025 * i, 4) for i in range(1, 13))


def _dry_rng(*ids: int) -> np.random.Generator:
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + STREAMS["dry_run"], *ids])


def _key_rows(keys: Table) -> list[dict]:
    m = (has_tier(keys, "A") | has_tier(keys, "B")) & (_st(keys, "key_status") == "verified") \
        & np.isfinite(_fl(keys, "v_e_star"))
    sub = select(keys, m)
    names = list(sub)
    return [{c: (sub[c][i].item() if hasattr(sub[c][i], "item") else sub[c][i]) for c in names}
            for i in range(_n(sub))]


def _ttc_constant(rows: list[dict]) -> float:
    """c3's tau_c analogue: d_b = c clip(6 v_b, 150, 500) matches the key at R 22.9, 90 deg,
    A1 (else the median ratio over braking A1 cells)."""
    def ratio(r):
        return r["d_b_star"] / np.clip(P3_IDX_TIME * r["v_b_star"], *P3_IDX_CLIP)

    hit = [ratio(r) for r in rows if r["approach"] == "A1" and r["braking"]
           and np.isclose(r["R_nom"], 22.9) and np.isclose(r["theta_deg"], 90.0)]
    if hit:
        return float(hit[0])
    return _median([ratio(r) for r in rows if r["approach"] == "A1" and r["braking"]])


def _plant_for(kind: str, planted: float | None) -> Callable[[dict], tuple[float, float]]:
    if kind == "overspeed_below_12":
        e0 = SIM_OVERSPEED if planted is None else planted
        return lambda r: (e0 * 12.0 / r["R_seg"] if r["R_seg"] < 12.0 else 0.0, 0.0)
    if kind == "late_onset":
        dl = SIM_LATE if planted is None else planted
        return lambda r: (0.0, dl if r["approach"] == "A1" else 0.0)
    return lambda r: (0.0, 0.0)


def simulate_runs(kind: str, keys_table: Table, rng: np.random.Generator,
                  phase_noise: float = 1.0, plant: Callable[[dict], tuple[float, float]] | None
                  = None, planted: float | None = None, agent: str | None = None,
                  r_clip_max: float = float("nan"), crash_margin: float = SIM_CRASH_MARGIN,
                  n_native: int = 32, n_matched: int = 16) -> Table:
    """A simulated agent's run table from the real keys plus phase noise (protocol 7 dry run).

    Every verified tier A and B cell is run at each of its phases (deterministic) and with
    ``n_native`` native-noise samples (tier A; half on tier B) and ``n_matched`` matched-noise
    samples (tier A). The deterministic agent targets v_e = 0.99 v_e* (``SIM_LAWFUL``) and
    brakes at d_b*; a phase adds a common offset to both mirror members (v_e jitter
    SIM_SIGMA_V, d_b uniform over one step of travel). ``plant(row) -> (e, delta)`` adds an
    overspeed factor (1 + e) or a late onset of delta metres at the key's mean deceleration,
    v_e = 0.99 sqrt(v_e*^2 + 2 abar* delta). A run whose e_ve exceeds ``crash_margin``
    crashes in the corner, typed overspeed. Kinds: 'lawful', 'overspeed_below_12' (e =
    0.15 * 12 / R_seg below 12 m, A1 and A2), 'late_onset' (15 m late at A1),
    'time_to_corner' (d_b = c clip(6 v_b, 150, 500), entry speed lawful), 'planted' (use
    ``plant``)."""
    rows_k = _key_rows(keys_table)
    plant = plant or _plant_for(kind, planted)
    agent = agent or kind
    c_ttc = _ttc_constant(rows_k) if kind == "time_to_corner" else np.nan
    noise: dict[tuple, tuple[float, float]] = {}
    snoise: dict[tuple, float] = {}
    out: list[dict] = []
    for r in sorted(rows_k, key=lambda x: (x["pair"], x["cell"])):
        v_es, v_bs, d_bs = r["v_e_star"], r["v_b_star"], r["d_b_star"]
        brake = bool(r["has_brake_star"]) and np.isfinite(d_bs) and np.isfinite(v_bs)
        e_pl, dl = plant(r)
        L_c = r["s_c1"] - r["s_c0"]
        clipped = bool(np.isfinite(r_clip_max) and r["R_seg"] <= r_clip_max)
        steer = 1 if r["direction"] >= 0 else -1
        n_ph = int(r["n_phases"]) if r["n_phases"] > 0 else N_PHASES
        tier_a = "A" in str(r["tiers"]).split(",")
        e_by_phase = []
        for k in range(n_ph):
            if (r["pair"], k) not in noise:
                noise[(r["pair"], k)] = (float(rng.standard_normal()), float(rng.uniform(-.5, .5)))
            eps, u = noise[(r["pair"], k)]
            d_b = d_bs
            if dl > 0 and brake and d_bs > dl:
                abar = (v_bs**2 - v_es**2) / (2 * d_bs)
                v_e = SIM_LAWFUL * math.sqrt(v_es**2 + 2 * abar * dl)
                d_b = d_bs - dl
            else:
                v_e = v_es * SIM_LAWFUL * (1 + e_pl)
            if kind == "time_to_corner" and brake:
                d_b = c_ttc * float(np.clip(P3_IDX_TIME * v_bs, *P3_IDX_CLIP))
            v_e *= 1 + SIM_SIGMA_V * phase_noise * eps
            d_b = d_b + u * v_bs * DT * phase_noise if brake else np.nan
            e = v_e / v_es - 1
            e_by_phase.append(e)
            crash = bool(e > crash_margin)
            ex = dict(v_e=v_e, e_ve=e, e_ve_bo=v_e / r["v_e_star_bo"] - 1, e_ve_ra=e,
                      v_e_star_ra=v_es, v_min=r["v_min_star"] * (1 + e),
                      e_vmin=e, has_brake=brake, forced_onset=False,
                      d_b=d_b, dd_b=d_b - d_bs if brake else np.nan,
                      v_b=v_bs if brake else np.nan, d_resp=d_b + SIM_RESP_LEAD,
                      dd_resp=(d_b - d_bs) if brake else np.nan)
            out.append(_sim_run(agent, r, k, -1, "det", steer, crash, L_c, clipped, ex))
        n_nat = n_native if tier_a else n_native // 2
        for mode, n_s in (("nat", n_nat), ("mat", n_matched if tier_a else 0)):
            for j in range(n_s):
                key = (mode, r["pair"], j)
                if key not in snoise:
                    snoise[key] = float(rng.standard_normal())
                e = e_by_phase[j % n_ph] + SIM_SIGMA_NAT * snoise[key]
                out.append(_sim_run(agent, r, j % n_ph, j, mode, steer, bool(e > crash_margin),
                                    L_c, clipped, {}))
    cols = ("agent", "timesteps", "cell", "stim", "pair", "phase", "sample", "mode",
            "steer_sign", "W", "s0", "crashed", "completed", "s_crash", "crash_type",
            "kappa_clipped", "unsettled", "time_censored", "key_at_phase", "osc_index", "ra_key",
            "v_e", "e_ve", "e_ve_bo", "e_ve_ra", "v_e_star_ra", "v_min", "e_vmin", "has_brake",
            "forced_onset", "d_b", "dd_b", "v_b", "d_resp", "dd_resp")
    return _to_table(out, cols)


def _sim_run(agent: str, r: dict, k: int, sample: int, mode: str, steer: int, crash: bool,
             L_c: float, clipped: bool, extra: dict) -> dict:
    row = dict(agent=agent, timesteps=0, cell=int(r["cell"]), stim=int(r["stim"]),
               pair=int(r["pair"]), phase=int(k), sample=int(sample), mode=mode,
               steer_sign=int(steer), W=int(r["W"]), s0=float(r["s_c0"] - (r["D"] if
                                                                       np.isfinite(r["D"])
                                                                       else 0.0)),
               crashed=crash, completed=not crash,
               s_crash=float(r["s_c0"] + 0.5 * L_c) if crash else np.nan,
               crash_type="overspeed" if crash else "", kappa_clipped=clipped, unsettled=False,
               time_censored=False, key_at_phase=True, osc_index=0.0, ra_key=True)
    for c in ("v_e", "e_ve", "e_ve_bo", "e_ve_ra", "v_e_star_ra", "v_min", "e_vmin", "d_b",
              "dd_b", "v_b", "d_resp", "dd_resp"):
        row[c] = float(extra.get(c, np.nan))
    row["has_brake"] = bool(extra.get("has_brake", False))
    row["forced_onset"] = False
    return row


def simulate_pairs(kind: str, keys_table: Table, rng: np.random.Generator,
                   phase_noise: float = 1.0, b: float = BAND_EXPECTED,
                   regimes: dict | None = None, **kw) -> Table:
    """A simulated agent's pair table built from the real keys plus phase noise (protocol 7
    dry run): ``simulate_runs`` through ``pair_values``. Default regimes: as if q1 = 12 m
    and q10 = 20 m."""
    runs = simulate_runs(kind, keys_table, rng, phase_noise, **kw)
    agent = kw.get("agent") or kind
    return pair_values(runs, keys_table, regimes or {agent: DRY_RUN_REGIME}, b)


def _cond_design_from_keys(keys: Table) -> Table:
    m = _it(keys, "corner") >= 0
    sub = select(keys, m)
    rows, seen = [], set()
    for i in range(_n(sub)):
        c, cn = int(sub["corner"][i]), str(sub["condition"][i])
        if (c, cn) in seen:
            continue
        seen.add((c, cn))
        rows.append(dict(corner=c, condition=cn, braking=bool(sub["braking_socp"][i]),
                         v_min_star=float(sub["v_min_socp"][i]), swap_j=int(sub["swap_j"][i])))
    design = _to_table(rows, ("corner", "condition", "braking", "v_min_star", "swap_j"))
    # 'braking' is a corner property: the s-OCP brakes in condition (a)
    a = {int(c): bool(bb) for c, n, bb in zip(design["corner"], design["condition"],
                                             design["braking"], strict=True) if n == COND_A}
    design["braking"] = np.array([a.get(int(c), False) for c in design["corner"]], bool)
    return design


SIM_D_OOD = {"context_keyed": {"F6/wobble0.3": 1.0, "F6/wobble0.6": 1.5, "F6/knoise0.5e-3": 2.5,
                               "F6/knoise1e-3": 3.0, "_changed": 2.0, "_other": 1.2},
             "reader": {"F6/wobble0.3": 1.1, "F6/wobble0.6": 1.3, "F6/knoise0.5e-3": 1.6,
                        "F6/knoise1e-3": 2.0, "_changed": 1.5, "_other": 1.1}}


def simulate_conds(kind: str, design: Table, rng: np.random.Generator, agent: str | None = None
                   ) -> Table:
    """A simulated specialist's tier D condition table (protocol 7 dry run).

    ``design``: corner, condition, braking (the s-OCP brakes in (a)), v_min_star, swap_j.
    'context_keyed' completes every condition whose road before the corner is real and
    fails those where it is straightened, completes every F6 perturbation, and reads the
    corner (v_min = 0.98 v_min*, no phantom slowing, swaps read); its F6 d_OOD straddles
    the failing conditions'. 'reader' completes everything with the same reading.
    'corner_keyed' completes everything but slows for the phantom corner (phi = 1)."""
    agent = agent or kind
    corner, cond = _it(design, "corner"), _st(design, "condition")
    vstar, swap_j = _fl(design, "v_min_star"), _it(design, "swap_j")
    look = _cond_lookup(design)
    dd = SIM_D_OOD.get(kind, SIM_D_OOD["reader"])
    rows = []
    v_peak = 80.0
    for i in range(_n(design)):
        c, cn = int(corner[i]), str(cond[i])
        a_row = look.get((c, COND_A))
        v_a = 0.98 * vstar[a_row] if a_row is not None else np.nan
        noise = 1 + 0.002 * float(rng.standard_normal())
        v_min = 0.98 * vstar[i] * noise if np.isfinite(vstar[i]) else v_a * noise
        v_region = v_min
        if cn == COND_SWAP_B:
            v_region = v_a if kind == "corner_keyed" else v_peak
        if cn == COND_SWAP_A and swap_j[i] >= 0 and (int(swap_j[i]), COND_A) in look:
            v_min = 0.98 * vstar[look[(int(swap_j[i]), COND_A)]] * noise
        parts = cn.split("/")
        complete = True
        if kind == "context_keyed" and parts[0] == "F1" and parts[1] == "straight":
            complete = False
        if parts[0] == "F1" and cn != COND_A:
            d_ood = dd["_changed"]
        else:
            d_ood = dd.get(cn, dd["_other"])
        rows.append(dict(agent=agent, timesteps=0, corner=c, condition=cn,
                         braking=bool(_bo(design, "braking")[i]), completes=4 if complete else 0,
                         complete=complete, v_min=float(v_min), v_peak=v_peak,
                         v_min_region=float(v_region), d_ood=float(d_ood),
                         v_min_star=float(vstar[i]), swap_j=int(swap_j[i])))
    return _to_table(rows, COND_COLUMNS)


def _verdict_of(prediction: str, pairs: Table, runs: Table, keys: Table, tables, b: float
                ) -> tuple[str, dict]:
    if prediction == "P1a":
        d = p1a(pairs, b)
    elif prediction == "P1-tail":
        d = p1_tail(pairs, b)
    elif prediction == "P1b":
        d = p1b(pairs, b)
    elif prediction == "P2":
        d = p2(pairs, None, b)
    elif prediction == "P2b":
        d = p2b(pairs, b)
    elif prediction == "P3":
        if tables is None:
            return "not testable", {"reason": "no model tables"}
        d = p3(p3_fit(runs, keys, tables), brake_share(runs, keys))
        if d.verdict == "refute":
            return f"refute:{d.details.get('winner', '')}", d.details
    else:
        raise ValueError(prediction)
    return d.verdict, d.details


def dry_run(keys_table: Table, tables=None, b: float = BAND_EXPECTED,
            regimes: dict | None = None, design: Table | None = None,
            phase_noise: float = 1.0) -> Table:
    """Step 5: each simulated agent of DRY_RUN_AGENTS through the real key design must give
    its planted verdict (DRY_RUN_EXPECTED). ``regimes`` maps a kind to the regime it is
    judged under (default ``DRY_RUN_REGIME`` for all); ``design`` is the tier D design
    (default: from the key table's tier D rows). Returns DRY_RUN_COLUMNS."""
    import json

    from apexzero.interp.io import _jsonable

    rows = []
    for i, kind in enumerate(DRY_RUN_AGENTS):
        if kind == "context_keyed":
            dz = design if design is not None else _cond_design_from_keys(keys_table)
            if not _n(dz):
                verdict, det = "not testable", {"reason": "no tier D design"}
            else:
                spec = simulate_conds(kind, dz, _dry_rng(i, 0), agent="K-sim")
                gen = simulate_conds("reader", dz, _dry_rng(i, 1), agent="G-sim")
                res = p4_classify(spec, gen)
                verdict, det = f"class {res.cls}", dict(label=res.label, C=res.C,
                                                        key_factors=list(res.key_factors))
            exp = DRY_RUN_EXPECTED[kind]["P4"]
            rows.append(dict(kind=kind, prediction="P4", expected=exp, verdict=verdict,
                             passed=verdict == exp, details=json.dumps(_jsonable(det))))
            continue
        reg = (regimes or {}).get(kind, DRY_RUN_REGIME)
        runs = simulate_runs(kind, keys_table, _dry_rng(i, 0), phase_noise,
                             r_clip_max=getattr(reg, "r_clip_max", np.nan))
        pairs = pair_values(runs, keys_table, {kind: reg}, b)
        for pred, exp in DRY_RUN_EXPECTED[kind].items():
            verdict, det = _verdict_of(pred, pairs, runs, keys_table, tables, b)
            rows.append(dict(kind=kind, prediction=pred, expected=exp, verdict=verdict,
                             passed=verdict == exp, details=json.dumps(_jsonable(det),
                                                                       sort_keys=True)))
    return _to_table(rows, DRY_RUN_COLUMNS)


def _regime_counts(keys: Table, rt) -> dict[str, int]:
    out = {}
    m = has_tier(keys, "A") & _bo(keys, "braking") & ~_bo(keys, "no_cap") & \
        (_st(keys, "key_status") == "verified")
    reg = np.array([regime_of(r, rt.q1, rt.q10, getattr(rt, "r_clip_max", np.nan))
                    for r in _fl(keys, "R_seg")])
    for ap in ("A1", "A2"):
        for g in ("clip", "below", "tail", "bulk"):
            sel = m & (_st(keys, "approach") == ap) & (reg == g)
            out[f"pairs_{ap}_{g}"] = int(np.unique(_it(keys, "pair")[sel]).size)
    return out


def _power_plant(test: str, e: float, rt) -> Callable[[dict], tuple[float, float]]:
    def reg(r):
        return regime_of(r["R_seg"], rt.q1, rt.q10, getattr(rt, "r_clip_max", np.nan))

    if test == "P1a":
        return lambda r: (e if r["approach"] == "A2" and reg(r) == "bulk" else 0.0, 0.0)
    if test == "P1-tail":
        return lambda r: (e if r["approach"] == "A1" and reg(r) == "tail" else 0.0, 0.0)
    if test == "P1b":
        return lambda r: (e * rt.q1 / r["R_seg"] if reg(r) == "below"
                          and r["approach"] in ("A1", "A2") else 0.0, 0.0)
    if test == "P2":
        def late(r):
            if r["approach"] != "A1" or reg(r) not in ("tail", "below") or not r["braking"]:
                return 0.0, 0.0
            v_es, v_bs, d_bs = r["v_e_star"], r["v_b_star"], r["d_b_star"]
            abar = (v_bs**2 - v_es**2) / (2 * d_bs)
            dl = v_es**2 * (((1 + e) / SIM_LAWFUL) ** 2 - 1) / (2 * abar)
            return 0.0, float(min(dl, 0.95 * d_bs))
        return late
    raise ValueError(test)


POWER_DETECT = {"P1a": "refute", "P1-tail": "confirm", "P1b": "confirm", "P2": "confirm"}


def power_table(keys_table: Table, support, b: float, tables=None,
                steps: tuple[float, ...] = POWER_STEPS, phase_noise: float = 1.0) -> Table:
    """Per confirmatory test and agent: pairs per regime and the smallest planted overspeed
    (2.5-point steps) the rule detects (P1a: refute when the bulk A2 pairs overspeed;
    P1-tail: confirm when the A1 tail pairs do; P1b: confirm with e q1 / R_seg below
    support; P2: confirm with a late onset sized to give e at the A1 tail and below-support
    cells). P3: the verdicts on the lawful and time-to-corner agents.

    ``support`` is ``support.SupportTables`` (its ``regimes``) or a dict agent -> regime."""
    regimes = support.regimes if hasattr(support, "regimes") else dict(support)
    rows = []
    for ia, (agent, rt) in enumerate(sorted(regimes.items())):
        counts = _regime_counts(keys_table, rt)
        clip = getattr(rt, "r_clip_max", np.nan)
        lawful_runs = simulate_runs("lawful", keys_table, _dry_rng(100 + ia, 0), phase_noise,
                                    agent=agent, r_clip_max=clip)
        lawful_pairs = pair_values(lawful_runs, keys_table, {agent: rt}, b)
        for it, test in enumerate(POWER_DETECT):
            lawful_v = _verdict_of(test, lawful_pairs, lawful_runs, keys_table, tables, b)[0]
            min_e, planted_v = float("nan"), ""
            for ie, e in enumerate(steps):
                runs = simulate_runs("planted", keys_table, _dry_rng(100 + ia, it + 1, ie),
                                     phase_noise, plant=_power_plant(test, e, rt), agent=agent,
                                     r_clip_max=clip)
                pairs = pair_values(runs, keys_table, {agent: rt}, b)
                v = _verdict_of(test, pairs, runs, keys_table, tables, b)[0]
                if v == POWER_DETECT[test]:
                    min_e, planted_v = float(e), v
                    break
            rows.append(dict(test=test, agent=agent, **counts, min_overspeed=min_e,
                             lawful_verdict=lawful_v, planted_verdict=planted_v))
        ttc_runs = simulate_runs("time_to_corner", keys_table, _dry_rng(100 + ia, 99),
                                 phase_noise, agent=agent, r_clip_max=clip)
        ttc_pairs = pair_values(ttc_runs, keys_table, {agent: rt}, b)
        rows.append(dict(test="P3", agent=agent, **counts, min_overspeed=float("nan"),
                         lawful_verdict=_verdict_of("P3", lawful_pairs, lawful_runs,
                                                    keys_table, tables, b)[0],
                         planted_verdict=_verdict_of("P3", ttc_pairs, ttc_runs, keys_table,
                                                     tables, b)[0]))
    return _to_table(rows, POWER_COLUMNS)


# ------------------------------------------------------------------------- input tables (step 8)
#
# The tables AnalysisInputs reads besides run and key tables: P8's ladder and twin tables,
# the joint-support densities per pair, and P7's aliasing table. scripts/psychophysics.py
# (analyse) builds them from the run tables, the design and the probes.

def p8_set(agent: str, R_seg: float, regimes: dict | None = None) -> str:
    """The P8 set of a crash unit (ruling R2): the regime of its crash corner, the
    ``corners.segment`` corner containing the crash (D15, ``stimuli.crash_unit_R_seg``),
    from that corner's R_seg against the crashing agent's frozen q1 / q10 ('below', 'tail'
    or 'bulk'; 'clip' counts as below). The same rule for every agent's units, G3's own R_DEV
    crashes included; no unit carries a radius label of its own. 'other' without the
    agent's frozen regimes or a finite radius."""
    rt = (regimes or {}).get(agent)
    if rt is None or not np.isfinite(R_seg):
        return "other"
    reg = regime_of(float(R_seg), rt.q1, rt.q10, getattr(rt, "r_clip_max", float("nan")))
    return {"clip": "below"}.get(reg, reg) or "other"


def ladder_table(runs: Table, keys: Table, units: list[dict],
                 fidelity: dict[str, bool] | None = None) -> Table:
    """LADDER_COLUMNS for the crash units of one agent's final run table. ``units``: one
    dict per (agent, crash) with 'agent', 'unit', 'set' and 'cells' {version: [cell ids]}
    (the E1 cells built from that agent's state at that corner); ``fidelity`` maps a unit to
    the real rung's fidelity gate (3.2): True (passed), False (evaluated and failed), None or
    absent (not evaluated: its inputs are missing; ``fidelity_missing``). A unit without
    cells keeps one empty real row, so missing data never drops it from P8's n. e_ve is the
    median per-run error with overspeed crashes at +inf (9.1)."""
    fidelity = fidelity or {}
    cell, mode = _it(runs, "cell"), _st(runs, "mode")
    crashed, ct = _bo(runs, "crashed"), _st(runs, "crash_type")
    e = _run_speed(runs, keys, "e_ve")
    rows = []
    for u in units:
        gate = fidelity.get(u["unit"])
        for ver, ids in (u["cells"] or {"real": []}).items():
            m = np.isin(cell, np.asarray(ids, np.int64))
            det, nat = m & (mode == "det"), m & (mode == "nat")
            rows.append(dict(agent=u["agent"], unit=u["unit"], set=u["set"], version=ver,
                             fidelity_ok=bool(gate), fidelity_missing=gate is None,
                             det_crashes=int((crashed & det).sum()), det_n=int(det.sum()),
                             nat_k=int((crashed & nat).sum()), nat_n=int(nat.sum()),
                             e_ve=_median(e[det]) if det.any() else float("nan"),
                             det_overspeed=int((crashed & det & (ct == "overspeed")).sum())))
    return _to_table(rows, LADDER_COLUMNS)


def twin_table(runs: Table, keys: Table, pairs: Table, corners: list[dict]) -> Table:
    """TWIN_COLUMNS for one agent: per real corner (E1 crash corners and E2 matches, every
    state pooled) the median deterministic e_ve on the real, isolated and twin versions
    (+inf for overspeed crashes), and the agent's tier A e_ve curve at the twin's R_seg,
    theta class and approach (A1 when the real rung's v_b is >= 250 km/h, else A2).
    ``corners``: {'corner', 'source', 'R_seg', 'theta_deg', 'cells': {version: [ids]},
    'length_shortfall'} (the twin's length shortfall, R4; NaN when absent)."""
    cell, mode = _it(runs, "cell"), _st(runs, "mode")
    e = _run_speed(runs, keys, "e_ve")
    v_b = _fl(runs, "v_b")
    agent = _agent(runs)
    rows = []
    for c in corners:
        def med(ver: str, arr: np.ndarray = e, c: dict = c) -> float:
            m = np.isin(cell, np.asarray(c["cells"].get(ver, []), np.int64)) & (mode == "det")
            return _median(arr[m]) if m.any() else float("nan")

        vb_real = med("real", v_b)
        approach = "A1" if np.isfinite(vb_real) and vb_real >= kph(250.0) else "A2"
        rows.append(dict(agent=agent, corner=c["corner"], source=c["source"],
                         e_ve_real=med("real"), e_ve_isolated=med("isolated"),
                         e_ve_twin=med("twin"),
                         e_ve_curve=tier_a_curve_at(pairs, c["R_seg"], c["theta_deg"], approach),
                         length_shortfall=float(c.get("length_shortfall", np.nan))))
    return _to_table(rows, TWIN_COLUMNS)


def training_sd(features: np.ndarray) -> np.ndarray:
    """SD (ddof 1) of each support feature over a training sample's finite rows."""
    f = np.asarray(features, float)
    f = f[np.all(np.isfinite(f), axis=1)]
    return np.std(f, axis=0, ddof=1)


def density_table(cell_support: Table, cells: list, agent_grammar: dict[str, str],
                  train: dict[str, np.ndarray], tightfast: dict[str, np.ndarray],
                  feature_sd) -> Table:
    """Per (agent, pair): the joint training-support log density of the pair's stimulus at
    its approach under that agent's grammar (mean over the pair's cells; protocol 2.3), and
    the log density under the grammar's tight-and-fast corners only (R_seg < 20 m, v_peak
    >= 250 km/h; 9.3). ``train`` / ``tightfast``: grammar -> support feature rows;
    ``cell_support``: stimuli.cell_support_table. ``feature_sd``: grammar -> feature SDs
    (each grammar's own training sample, protocol 2.3), or one vector for every grammar;
    None: each grammar's SD from ``train``. Agents without a grammar get no rows
    (specialists: their training corners are real corners with no generator ramp l_r, so
    the density is undefined, S16)."""
    from apexzero.interp.support import knn_log_density, support_features

    cid = _it(cell_support, "cell")
    feats = support_features(_fl(cell_support, "R_seg"), _fl(cell_support, "theta_deg"),
                             _fl(cell_support, "l_r"), _fl(cell_support, "L_c"),
                             _fl(cell_support, "v_peak"))
    pair_of = {int(c.cell_id): int(c.pair_id) for c in cells}

    def sd_of(g: str) -> np.ndarray:
        if feature_sd is None:
            return training_sd(train[g])
        if isinstance(feature_sd, dict):
            return np.asarray(feature_sd[g], float)
        return np.asarray(feature_sd, float)

    rows = []
    for agent, g in sorted(agent_grammar.items()):
        sd = sd_of(g)
        col = f"logdens_{g}"
        ls = (np.asarray(cell_support[col], float) if col in cell_support
              else knn_log_density(train[g], feats, sd))
        lt = knn_log_density(tightfast[g], feats, sd) if len(tightfast.get(g, ())) > 10 \
            else np.full(len(cid), np.nan)
        by: dict[int, list[int]] = {}
        for i, c in enumerate(cid):
            if int(c) in pair_of:
                by.setdefault(pair_of[int(c)], []).append(i)
        for p, idx in sorted(by.items()):
            rows.append(dict(agent=agent, pair=p, log_support=_median(ls[idx]),
                             log_support_tightfast=_median(lt[idx])))
    return _to_table(rows, ("agent", "pair", "log_support", "log_support_tightfast"))


def alias_table(alias: Table, keys: Table) -> Table:
    """P7_ALIAS_COLUMNS from the probes' per-run A_alias rows (columns agent, cell, A_alias;
    controls 'Q' and 'Qp' among the agents): per agent and P7 geometry, the median over the
    pair's members of the agent's, Q's and Qp's A_alias."""
    k = _join_keys(alias, keys, ("geometry", "R_nom", "theta_deg"))
    geo = k["geometry"].astype(str)
    ag, a = _st(alias, "agent"), _fl(alias, "A_alias")
    rows = []
    for agent in sorted(set(ag) - {"Q", "Qp"}):
        for g in sorted(set(geo[ag == agent])):
            m = geo == g
            R, th = float(k["R_nom"].astype(float)[m][0]), float(k["theta_deg"].astype(float)[m][0])
            rows.append(dict(agent=agent, geometry=g, R_nom=R, theta_deg=th,
                             short=bool(th == 90.0 and R <= 14.6),
                             a_agent=_median(a[m & (ag == agent)]), a_q=_median(a[m & (ag == "Q")]),
                             a_qp=_median(a[m & (ag == "Qp")])))
    return _to_table(rows, P7_ALIAS_COLUMNS)
