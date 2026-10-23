"""Idea 2's pre-registration record (design v2.1, Section 7 'Freeze'): what the tag
``costate-prereg-v1`` fixes besides the code.

The record is ``src/apexzero/costate/prereg.json``, inside the frozen paths, so git freezes it
with the code at the tag and scripts/costate.py's freeze check reads it from the tagged commit
(``git show <tag>:<path>``). It is written last before the tag, from a clean tree, by the
'prereg' stage (``build_record``), and holds (modelled on step 2's ``fields.prereg``,
fields/prereg.py:1-60, simplified where idea 2 needs less):

- **the state lists**: the A2 key set (``keys.a2_refs``: 481 sorted (kind, key id) pairs) with
  the content hash of its .json and .npz files (``fields.prereg.content_hash``), N's anchored
  finals per agent (``states.n_refs``) with theirs, and T's (key id, t) list after K2's cut
  with its sha256 (``t_hash``); D is T at t = 0;
- **the frozen values** computed before the tag (``frozen_values``): the winsorisation bounds
  of every target column on T, f_D's one k (``states.KNN_K``, the same for every partition,
  outer fold and K_P component; 'decoy_k', which the post-tag C2 reads, ``decoy_k``), the
  K_P residual shares and the power-floor verdicts, partition 0's fold of each geometry and
  the sha256 of the 10 robustness partitions, t_H, the kappa sensitivity set (the
  geometries with R <= ``KAPPA_SET_R``), the interior share by phase;
- **the constants** the gates, nulls and decisions use (``constants``: a copy for the reader;
  the code hash binds them): tolerances, thresholds (interior k, clearance and slip margins,
  tau_s, the pedal-saturated tolerance), the penalty grid, Huber's constant, the bands, the
  seeds and streams, the null definitions, C3's strata, covariates and pre-signed pairs, the
  value-accuracy threshold;
- **the deferred analyses** (``deferred``, ``DEFERRED``; design Section 13 'Deferred'):
  C3-int, C1-dev, Delta C1, C2-pairs, C1-own on its own-run and N states, N's replication of
  C1, C2 and C2-sw, W1 on the Pi targets, W1's own-policy width price and C2 and C2-sw beyond
  the decision networks are not part of costate-prereg-v1 (exploratory if run later).
  C3-int's permutations and pre-signed pairs (in the constants) and C1-dev's snapshot pins
  stay in the record, labelled there as kept without pre-registering those analyses;
- **every network** a post-tag stage loads (``network_pins``: the 31 generalists of C3-w's
  strata and the 3 specialists; ``snapshot_pins``: the snapshots of the 9 decision networks
  that the deferred C1-dev would read): its checkpoint's repo-relative path (built without
  resolving symlinks, so a worktree whose runs/ links to the main checkout pins runs/...),
  sha256 (which covers the head subtrees Dense_1-Dense_10) and timesteps; post-tag loaders
  load the recorded path and check its hash (``pins_problems`` re-hashes networks and
  snapshots alike);
- **the bound pre-tag outputs** (``bind_outputs``): every ``BOUND_OUTPUTS`` file under
  outputs/costate with its sha256 and its stamp's code hash; each must carry a stamp of this
  code, not be partial, and pass (K1 on its ``pass_with_plants``; the nulls summary with
  every decision network, every C1 null in its layout (the fold-dependent ones per held-out
  fold, ``_null_shapes_ok``) and finite, and the sha256 of each array file; inputs with
  K2's t_H); and the arrays the post-tag analysis reads (``bind_arrays``: inputs/world.npz
  and each decision network's nulls/<net>.npz, with their stamps' sha256);
- **C3-w's inputs** (``c3w_inputs``): per generalist the sha256 of eval_rdev.json and
  metrics.csv and the computed y (completion) and competence, which the post-tag C3-w reads
  from the record, and step 1's stimulus design hash (``stimuli.design_hash``), which the
  post-tag roads are rebuilt from;
- the design document's sha256, ``code_hash`` over src/ (all of it), scripts/costate.py and
  scripts/_bootstrap.py (``fields.prereg.code_hash``'s rule on ``CODE_ROOTS``), the commit the
  record was built on (``prereg_base_commit``), step 1's and step 2's tags and the
  environment (Python and package versions).

``verify_record`` recomputes each item from disk and lists every mismatch as a
``fields.prereg.Problem`` with a code; only 'code_changed' and 'design_doc_changed' may be
waived by a logged deviation (``WAIVABLE``). Post-tag stages iterate ``frozen_ids`` and
``frozen_t``, never a fresh glob.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from apexzero.fields import prereg as FP
from apexzero.fields.prereg import Problem, read_stamp, sha256_file

PREREG_NAME = "prereg.json"
PREREG_PATH = Path(__file__).with_name(PREREG_NAME)
PREREG_REL = f"src/apexzero/costate/{PREREG_NAME}"
DESIGN_DOC = "docs/interp/costate-design.md"
CODE_ROOTS = ("src", "scripts/costate.py", "scripts/_bootstrap.py")
VERSION = 1
WAIVABLE = FP.WAIVABLE
KAPPA_SET_R = 8.94            # m: Section 4's kappa-clipping sensitivity set (R <= this)
STAMP_SUFFIX = FP.STAMP_SUFFIX
# The pre-tag outputs the record binds (relative to outputs/costate) and their pass keys.
BOUND_OUTPUTS = ("inputs/summary.json", "gates/k0.json", "gates/k1.json", "gates/k2.json",
                 "gates/k2b.json", "gates/k3.json", "gates/k3b.json", "gates/k4.json",
                 "gates/k5.json", "gates/k6.json", "nulls/summary.json")
BOUND_ARRAYS = ("inputs/world.npz",)   # plus nulls/<net>.npz per decision network
# Design Section 13 'Deferred': the analyses that are not part of costate-prereg-v1 (if run
# later, they are exploratory), as (name, what, the record's items kept for them, which pin
# their inputs without pre-registering them). The record lists them ('deferred',
# ``deferred``); a later deferral is added here and to Section 13.
DEFERRED = (
    ("C3-int", "the network x geometry interaction test (Section 6; ``analysis.c3_int``)",
     ("constants.analysis.c3_int_perm", "constants.analysis.c3_int_pairs")),
    ("C1-dev", ("C1's R2_c across each decision network's snapshots (Section 6; "
                "``analysis.c1_dev``)"), ("snapshots",)),
    ("DeltaC1", "R2(z + g) - R2(z), conditional probing (Section 6; ``analysis.delta_c1``)",
     ()),
    ("C2-pairs", "matched-decoy row pairs within a geometry (Section 6; ``analysis.c2_pairs``)",
     ()),
    ("C1-own on own-run and N states",
     ("C1-own on every state of each network's own run from each D key and on N (Section 3.3 "
      "(ii), (iii)): it needs critic pullbacks at own-run states; C1-own is decided on T's "
      "interior restarts only"), ()),
    ("N's replication",
     ("N's per-agent replication of C1, C2 and C2-sw (Sections 3.2 and 6, 'N replicates'): "
      "N's states are never built (the inputs stage builds T only)"), ()),
    ("W1 on the Pi targets",
     ("W1's probe R2 and Spearman for Pi^in, Pi^out, Pi^slip and Pi^mu (Section 6 'W1'): W1 "
      "reports only the critic's width derivative against Pi^in, descriptively"), ()),
    ("W1's own-policy width price",
     ("the policy's own width price (Section 3.4) and W1's three comparisons of the critic's "
      "width field with the optimum's and the policy's own, with their calibrated magnitudes"),
     ()),
    ("C2 and C2-sw beyond the decision networks",
     ("the descriptive C2 and C2-sw of the 22 other generalists and of the specialists "
      "(Section 6): the probes stage runs on the 9 decision networks only; their C1 (without "
      "nulls) and the generalists' C1-heads are reported"), ()),
)


def net_file(name: str) -> str:
    """A network's output file stem: its run directory (``analysis.network_of``), '/' -> '_'
    (so 'G1' and 'runs/dr-pilot-200M' name the same file)."""
    from apexzero.costate import analysis as AN

    return AN.network_of(name).replace("/", "_")


def null_networks() -> list[str]:
    """The networks C1's nulls are computed for: the decision networks' run directories."""
    from apexzero.costate import analysis as AN

    return [AN.network_of(a) for a in AN.DECISION_NETWORKS]


def code_hash(repo: Path) -> str:
    return FP.code_hash(Path(repo), CODE_ROOTS)


def tree_code_hash(repo: Path, rev: str) -> str:
    return FP.tree_code_hash(Path(repo), rev, CODE_ROOTS)


def _canon(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_default).encode()).hexdigest()


def t_hash(pairs: list[tuple[str, int]]) -> str:
    """sha256 of T's sorted (key id, t) list."""
    return _canon(sorted([str(k), int(t)] for k, t in pairs))


def key_sets(keys_root: Path | None = None) -> dict[str, list[tuple[str, str]]]:
    """{'a2': A2 keys, 'n:<agent>': each agent's anchored finals} as sorted (kind, id)."""
    from apexzero.costate import states as ST

    out = {"a2": sorted((r.kind, r.key_id) for r in ST.a2_refs(keys_root))}
    for a, refs in ST.n_refs(keys_root).items():
        out[f"n:{a}"] = sorted((r.kind, r.key_id) for r in refs)
    return out


def constants() -> dict:
    """The gates', nulls' and decisions' constants (a copy for the reader)."""
    from apexzero.costate import STREAMS
    from apexzero.costate import analysis as AN
    from apexzero.costate import critic as CR
    from apexzero.costate import gates as GT
    from apexzero.costate import nulls as NL
    from apexzero.costate import own as OW
    from apexzero.costate import probes as PB
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    return {
        "targets": {"active_tol": TG.ACTIVE_TOL, "saturated_tol": TG.SATURATED_TOL,
                    "route_tol": TG.ROUTE_TOL, "x0_tol": TG.X0_TOL, "fold_tol": TG.FOLD_TOL,
                    "sign_tol": TG.SIGN_TOL, "action_tol": TG.ACTION_TOL,
                    "straight_tol": TG.STRAIGHT_TOL, "plant_min": TG.PLANT_MIN,
                    "fd_h": TG.FD_H, "pedal_ladder": list(TG.PEDAL_LADDER),
                    "pedal_up": TG.PEDAL_UP, "pedal_min_h": TG.PEDAL_MIN_H,
                    "k1": [TG.K1_MEDIAN, TG.K1_P95], "plants": list(TG.PLANTS),
                    "kink_factor": TG.KINK_FACTOR, "zero_atol": TG.ZERO_ATOL,
                    "kink_j_noise": TG.KINK_J_NOISE, "shrink_rtol": TG.SHRINK_RTOL,
                    "max_shrink": TG.MAX_SHRINK, "pi_h": TG.PI_H,
                    "pi_plants": list(TG.PI_PLANTS),
                    "k1_rule": "K1's zero threshold and denominator floor eps_t = max(zero_atol, "
                               "6 kink_j_noise / (h gamma^t)) (gates.k1_error; on the pedal-"
                               "saturated ladder the rungs' J spread where it exceeds "
                               "kink_j_noise); zero comparisons counted per component, plant "
                               "and reading, never pooled, and a decided component needs a "
                               "non-zero one; a kink: |forward - backward| > kink_factor "
                               "max(discrepancy, 6 kink_j_noise / (h gamma^t)); c / 10 at most "
                               "max_shrink times while the Richardson pair disagrees by more "
                               "than shrink_rtol |FD| and 6 kink_j_noise / (h gamma^t), and "
                               "still so: unresolved, not counted, in the 30% share; pi_1 "
                               "(targets.fd_pi at pi_h) under the same zero rule with eps = "
                               "max(zero_atol, 6 kink_j_noise / (pi_h gamma)), its plants' "
                               "comparisons too: pi needs a non-zero comparison and each pi "
                               "plant must fail on its non-zero ones; zero_atol alone in the "
                               "reported 'design_rule'"},
        "states": {"every": ST.EVERY, "exit_margin": ST.EXIT_MARGIN,
                   "interior_k": ST.INTERIOR_K, "interior_k_descriptive": ST.INTERIOR_K_DESCRIPTIVE,
                   "clearance_min": ST.CLEARANCE_MIN, "slip_margin_min": ST.SLIP_MARGIN_MIN,
                   "folds": ST.N_FOLDS, "inner": ST.N_INNER, "partitions": ST.N_PARTITIONS,
                   "knn_k": ST.KNN_K, "knn_k_candidates": list(GT.DECOY_K_CANDIDATES),
                   "knn_k_rule": "f_D's one k (knn_k) for every fit, partition, outer fold "
                                 "and K_P component, chosen before the tag on synthetic known "
                                 "truth only (gates.decoy_k_choice): among knn_k_candidates "
                                 "the largest k for which every validation seed "
                                 "(knn_k_validation) of K4's PR-6b and PR-6c worlds gives "
                                 "C2's primary Delta2 <= gates.k4[2], y residualised at k and "
                                 "C2's power floor recomputed at k; none: the smallest, a "
                                 "blocker; the Delta2 read is the units' paired increment "
                                 "over their host (gates.k4_no_gain_rule). Grounds: what "
                                 "leaks into C2 is f_D's bias, which a network holding the "
                                 "decoy function recovers; its variance only costs power, so "
                                 "an out-of-geometry SSE, which trades the two, does not "
                                 "choose k",
                   "knn_k_validation": {
                       "seeds": list(GT.DECOY_K_SEEDS), "stream": "costate_decoy_k",
                       "statistic": GT.K4_PAIRED,
                       "delta2": {str(k): {kind: list(v) for kind, v in d.items()}
                                  for k, d in GT.DECOY_K_VALIDATION.items()},
                       "delta2_unpaired": {str(k): {kind: list(v) for kind, v in d.items()}
                                           for k, d in
                                           GT.DECOY_K_VALIDATION_UNPAIRED.items()}},
                   "power_floor": ST.POWER_FLOOR, "tau_s": ST.TAU_S,
                   "winsor": list(ST.WINSOR), "small_geometry": ST.SMALL_GEOMETRY_KEYS,
                   "sw_stratum": ST.SW_STRATUM},
        "probes": {"grid": list(PB.GRID), "h_grid_inf": True, "min_sd": PB.MIN_SD,
                   "n_boot": PB.N_BOOT, "ci": PB.CI_LEVEL, "c2_band": PB.C2_BAND,
                   "random_actors": PB.N_RANDOM_ACTORS, "random_critics": PB.N_RANDOM_CRITICS,
                   "mlp": [PB.MLP_HIDDEN, list(PB.MLP_ALPHAS), PB.MLP_MAXITER],
                   "control": [PB.CONTROL_HIDDEN, PB.CONTROL_GAIN, PB.CONTROL_TOL,
                               PB.CONTROL_STEPS], "wiring": [PB.WIRING_DRAWS, PB.WIRING_TOL],
                   "wiring_rule": "K5: while each outer fold is held out, the training "
                                  "targets permuted state by state across all training rows "
                                  "(the mirror-odd components within turn direction, the "
                                  "others across both; stream costate_perm, sub-streams "
                                  "(draw, f, j)); the out-of-fold predictions of wiring[0] "
                                  "draws averaged and scored by R2_wd against the unpermuted "
                                  "held-out targets; every K_P component needs a cluster-"
                                  "bootstrap CI lower end <= 0 and a point <= wiring[1]",
                   "streams": [PB.BOOT_STREAM, PB.MLP_STREAM]},
        "analysis": {"huber_c": AN.HUBER_C, "cal_slope": AN.CAL_SLOPE, "k_lev": AN.K_LEV,
                     "c3w_rows": AN.C3W_ROWS, "w1_rows": AN.W1_ROWS,
                     "c1": [AN.C1_CONFIRM, AN.C1_REFUTE],
                     "scale_band": list(AN.SCALE_BAND), "probe_fit": AN.PROBE_FIT,
                     "c3": [AN.C3_PERMUTATIONS, AN.C3_ALPHA, AN.C3_STRONG],
                     "c3_r2c": {"x": "C1's R2_c without nulls (T's interior, the K_P mean)",
                                "test": "C3-w's", "stream": ["costate_c3", AN.C3_R2C_SUBSTREAM],
                                "networks": "the 31 generalists"},
                     "containment": "a descriptive that raises is recorded as {'error': "
                                    "'<type>: <message>'} (analysis.guarded); C1, C2, C2-sw, "
                                    "C3-w, C3-R2c and the decision rules of C1-pattern, "
                                    "C1-own, value accuracy, C1-limit and C1-braking raise; "
                                    "a network that failed K3 after the tag "
                                    "(critic.k3_trained) decides nothing, so its whole "
                                    "analysis or description is contained",
                     # C3-int is deferred (``DEFERRED``): kept, not pre-registered
                     "c3_int_perm": AN.C3_INT_PERMUTATIONS, "c3_strata": AN.C3_STRATA,
                     "c3_int_pairs": [list(p) for p in AN.C3_INT_PAIRS],
                     "value_accuracy": {
                         "r2_min": AN.VALUE_ACCURACY_R2, "target": AN.VALUE_TARGET,
                         "target_is": "the deterministic policy's own discounted return "
                                      "(own.own_return: delta = 0, s_stop = +inf)",
                         "rows": AN.VALUE_ROWS,
                         "rows_are": f"T's interior restarts every {OW.RESTART_EVERY}th step and D "
                                     "(every key's t = 0); crashed runs recorded and dropped",
                         "steps": OW.V_STEPS, "to": "the strip's end at most",
                         "every_row_scored": True,
                         "fit": "out of fold: C1's per-fold affine calibration (slope >= 0) "
                                "on the frozen folds, key weights on the rows",
                         "gates": "C1-own's validity",
                         "descriptive": {"target": AN.VALUE_TARGET_VSTAR,
                                         "rows": AN.VALUE_ROWS_VSTAR}},
                     "strata": {h: list(v) for h, v in AN.STRATA.items()},
                     "strata_per_network_columns": list(AN.PER_NETWORK_COLUMNS),
                     "strata_seed": AN.STRATUM_SEED,
                     "decision_networks": list(AN.DECISION_NETWORKS),
                     "c3_covariates": ["final training episode_return", "step 2's B"]},
        "nulls": {"names": list(AN.NULL_NAMES), "gamma": NL.GAMMA,
                  "barrier": [NL.BARRIER_SCALE, NL.BARRIER_EPS, NL.EDGE_MARGIN],
                  "qss_cv": 1.0, "value_mlp_alphas": list(NL.VALUE_MLP_ALPHAS),
                  "value_target": "horizon-corrected V*inf (value_ridge_window: V*_t)",
                  "value_grid_min": NL.VALUE_GRID_MIN,
                  "per_fold": list(AN.FOLD_NULLS),
                  "per_fold_rule": "[F, n, 8]: slice f built without fold f, cross-fitted "
                                   "as C2's decoys (value fits: fold f's rows through the "
                                   "fit without fold f, a training row of fold g through "
                                   "the fit without folds f and g at fold f's penalty, one "
                                   "fit per pair of folds and penalty; far geometry: "
                                   "neither fold f nor the row's own fold); fold f's "
                                   "calibration fitted on slice f's training rows, "
                                   "predicting fold f from slice f",
                  "value_mlp_nested_seed": "mlp_rng(partition, a, 99, N_INNER + 1 + b) for "
                                           "the fit without folds a < b",
                  "uncalibrated_rows": "C1, C1-pattern, C1-limit, W1 and C3-w's x: a row "
                                       "whose fold has no calibration is dropped and "
                                       "counted, never scored; none left: 'not testable' "
                                       "(C3-w: the network missing)"},
        "critic": {"fd_tol": CR.FD_TOL, "plant_tol": CR.PLANT_TOL, "linear_tol": CR.LINEAR_TOL,
                   "random_seeds": list(CR.RANDOM_SEEDS), "pi_h": CR.PI_H,
                   "k3_min_states": CR.K3_MIN_STATES,
                   # K3's pass rule (design v2.2 Section 13): per form, and the floor every
                   # planted wrong pullback must exceed
                   "fd_tol_manifold": CR.FD_TOL_MANIFOLD, "fd_tols": dict(CR.FD_TOLS),
                   "jz_tol": CR.FD_TOL, "wrong_min": CR.WRONG_MIN,
                   "wrong_pairs": list(CR.WRONG_PAIRS),
                   "wrong_closed_forms": list(CR.WRONG_CLOSED_FORMS),
                   # K3 after the tag (Section 7; ``gates.k3_trained``): every pinned network
                   "k3_trained": {
                       "rows": "D: T's t = 0 rows", "checks": list(GT.K3_TRAINED_CHECKS),
                       "extractor_tol": PB.EXTRACTOR_TOL,
                       "extractor_is": "max |extractor - Flax| in float64 of the actor's mean "
                                       "(Actor.apply), the critic's V and heads "
                                       "(Critic.apply(with_heads=True)), by both extractors",
                       "heads": {"generalists": PB.N_HEADS_GENERALIST, "specialists": 0},
                       "float32_tol": CR.F32_TOL,
                       "float32_is": "max |V64 - V32| in the critic's units; V32: "
                                     "RunningMeanStd.normalize and Critic.apply in float32 "
                                     "(x64 off) on the observation rounded to float32",
                       "fail": "its C1, C1-heads, C1-own (and held form), C1-pattern, "
                               "C1-limit, C1-braking, W1, C2 and C2-sw descriptive ('K3 "
                               "trained check failed: ...'); missing from every decision it "
                               "enters (per network, the variants, C3-w, C3-R2c); its whole "
                               "analysis or description contained (analysis.guarded)"}},
        "own": {"k3b": [OW.K3B_MEDIAN, OW.K3B_P95], "plants": list(OW.OWN_PLANTS),
                "k3b_steps": OW.K3B_STEPS, "v_steps": OW.V_STEPS,
                "restart_every": OW.RESTART_EVERY, "manifold_at": OW.MANIFOLD_AT,
                "accel_weights": OW.ACCEL_WEIGHTS},
        "gates": {"k1_kink_share": GT.K1_KINK_SHARE, "k2_tol": GT.K2_TOL,
                  "k2b": [GT.K2B_MARGIN, GT.K2B_SEVERITIES, list(GT.K2B_RAMP), GT.K2B_TOL],
                  "k4": [list(GT.K4_SNR), GT.K4_TOL, GT.K4_ZERO_TOL, GT.K4_READOUT_TOL],
                  "k4_c1": [list(GT.K4_C1_SNR), list(GT.K4_FLOOR), GT.K4_DIRECTION_FLOOR],
                  "k4_controls": GT.K4_N_CONTROLS,
                  "k4_pr1b_rule": "PR-1b's baseline, plant, probe and random controls on one "
                                  "target array, each row's own-fold lambda_res; every K_P "
                                  "component planted at once; the plant on C2's by-fold "
                                  "targets reported beside it, descriptive",
                  "k4_pr6": {"units": GT.K4_PR6_UNITS, "bias": GT.K4_PR6_BIAS,
                             "gains": list(GT.K4_PR6_GAINS),
                             "calibration_tol": GT.K4_CALIBRATION_TOL,
                             "seeds": list(GT.K4_PR6_SEEDS), "rho": GT.K4_PR6_RHO,
                             "noise": GT.K4_PR6_NOISE, "features": GT.K4_PR6B_FEATURES,
                             "alpha": GT.K4_PR6B_ALPHA,
                             "hold": [GT.K4_HOLD_UNITS, GT.K4_HOLD_SD],
                             "oversmooth_k": GT.K4_OVERSMOOTH_K, "pr6c_k": GT.K4_PR6C_K},
                  "k4_pr6_rule": "PR-6: Delta2 <= k4[2] (one-sided) on C2's by-fold lambda_res "
                                 "with K4's controls, the decoy units' gain the grid's whose "
                                 "probe-alone R2 lies closest to the decoy kNN's, and PR-1b "
                                 "passing; PR-6b (smooth F) and PR-6c (F the leave-one-"
                                 "geometry-out k = 15 kNN of lambda on D1): y = std(F) + "
                                 "noise AR(1) eps, the units holding F appended to h3, y "
                                 "residualised by states.cross_fit at states.knn_k; C2's "
                                 "primary statistic (its decision rows, power floor and 5 "
                                 "random-init controls) <= k4[2] on every seed, > k4[2] on "
                                 "every seed with the residualisation off, and for PR-6b the "
                                 "units' probe-alone R2 >= the kNN's on every seed; K4 needs "
                                 "both; every Delta2 here is k4_no_gain_rule's paired "
                                 "increment",
                  "k4_no_gain_rule": "a plant's effect is the change it causes in the same "
                                     "network: PR-2 (both variants), PR-3's grouped split, "
                                     "PR-6, PR-6b and PR-6c (with their residualisation-off "
                                     "checks and planted faults) are scored by the " + GT.K4_PAIRED
                                     + " (a paired bootstrap for the CI), tolerances, seeds "
                                     "and every-seed rules unchanged; the unpaired values and "
                                     "the host's own offset per plant and per seed are "
                                     "recorded, descriptive; PR-1a, PR-1b, PR-5, PR-8 and "
                                     "PR-3's mis-grouped leak check keep their rules (the "
                                     "leak's paired value reported)",
                  "k4_pr4_rule": "PR-4 one-sided: h2's and h3's R2 each >= the readout's R2 "
                                 "- k4[3] (the probe on either layer also sees its other "
                                 "units and can exceed the readout)",
                  "band_note_margin": AN.BAND_NOTE_MARGIN},
        "streams": dict(STREAMS), "kappa_set_r": KAPPA_SET_R}


def deferred() -> list[dict]:
    """The record's 'deferred' list (``DEFERRED``): each analysis with the record items kept
    for it that do not make it pre-registered."""
    return [{"name": n, "what": w, "kept_not_preregistered": list(kept),
             "status": "not part of costate-prereg-v1: exploratory if run later"}
            for n, w, kept in DEFERRED]


def frozen_values(tab, t_H: int | None, partitions: int | None = None) -> dict:
    """The values computed on T before the tag (module docstring)."""
    from apexzero.costate import states as ST
    from apexzero.costate import targets as TG

    w = ST.weights(tab, "key")
    cols = [c for c in ST.TARGET_COLUMNS if c in tab]
    bounds = {c: list(ST.winsor_bounds(np.asarray(tab[c], float), w)) for c in cols}
    lam = np.stack([ST.winsorise(np.asarray(tab[f"lam_{c}"], float), bounds[f"lam_{c}"])
                    for c in TG.K_P], 1)
    # f_D at its one k (states.KNN_K); partition 0's residuals give the shares and the floor
    res = ST.cross_fit(ST.decoy_d1(tab), lam, ST.row_folds(tab), w, k=ST.KNN_K)
    share, share_int = ST.decoy_shares(tab, lam, res.oof, w)
    inner = np.asarray(tab["interior"], bool)
    from apexzero.costate import analysis as AN
    G = ST.geometry_keys(tab)
    uniq = np.unique(G, axis=0)
    parts = {p: ST.geometry_folds(uniq[:, 0], uniq[:, 1], p).tolist()
             for p in range(partitions or ST.N_PARTITIONS)}
    return {"winsor_bounds": bounds, "residual_share": dict(zip(TG.K_P, share.tolist(),
                                                                strict=True)),
            "residual_share_interior": dict(zip(TG.K_P, share_int.tolist(), strict=True)),
            # C2 is decided on the interior stratum, so its floor is applied there (the
            # T-wide verdicts are descriptive)
            "power_floor": ST.power_floor(share_int), "power_floor_T": ST.power_floor(share),
            "decoy_k": {"k": res.k, "scope": "every partition, outer fold and K_P "
                                              "component (states.KNN_K)"},
            "c3w_rows": AN.C3W_ROWS, "w1_rows": AN.W1_ROWS, "t_H": t_H,
            "geometries": uniq.tolist(), "fold_partition0": parts[0],
            "partitions_sha256": _canon(parts),
            "kappa_set": [g for g in uniq.tolist() if g[0] <= KAPPA_SET_R],
            "interior_share": float(w @ inner.astype(float)),
            "t_count": len(w)}


def decoy_k(frozen: dict) -> int | None:
    """f_D's frozen k (``frozen_values``' 'decoy_k'), or None when the frozen values do not
    carry it (the caller then uses ``states.KNN_K``)."""
    dk = (frozen or {}).get("decoy_k")
    return None if not dk or dk.get("k") is None else int(dk["k"])


def build_record(repo: Path, sets: dict[str, list[tuple[str, str]]], keys_root: Path,
                 t_pairs: list[tuple[str, int]], frozen: dict, extra: dict | None = None) -> dict:
    """The record (module docstring); ``extra`` holds the base commit, network pins, bound
    outputs, environment and the steps' tags."""
    repo = Path(repo)
    if not (repo / DESIGN_DOC).exists():
        raise FileNotFoundError(f"{DESIGN_DOC} is missing: the record registers it")
    rec = {"version": VERSION, "tag": "costate-prereg-v1",
           "key_sets": {k: [list(t) for t in v] for k, v in sets.items()},
           "key_counts": {k: len(v) for k, v in sets.items()},
           "content": {k: FP.content_hash(v, keys_root) for k, v in sets.items()},
           "t_pairs": [[str(k), int(t)] for k, t in sorted(t_pairs)],
           "t_sha256": t_hash(t_pairs), "frozen": frozen, "constants": constants(),
           "deferred": deferred(), "design_doc_sha256": sha256_file(repo / DESIGN_DOC),
           "code_hash": code_hash(repo)}
    rec.update(extra or {})
    return rec


def _repo_rel(path: Path, repo: Path) -> str:
    """``path`` relative to ``repo`` without resolving symlinks (a worktree's runs/ may link
    to the main checkout: the pin stays runs/..., which the post-tag loader opens through the
    same link and whose sha256 guards it); ValueError when it is not under the repo."""
    import os

    p, r = Path(os.path.abspath(path)), Path(os.path.abspath(repo))
    try:
        return p.relative_to(r).as_posix()
    except ValueError:
        try:
            return p.relative_to(r.resolve()).as_posix()
        except ValueError as e:
            raise ValueError(f"{path} is not under the repository {repo}") from e


def pin_of(path: Path, agent_id: str, repo: Path) -> dict:
    """{path (repo-relative), sha256, timesteps} of one checkpoint (reads its pickle only)."""
    from apexzero.interp import agents

    ag = agents.load_agent(path, agent_id)
    return {"path": _repo_rel(path, repo), "sha256": sha256_file(path),
            "timesteps": int(ag.timesteps)}


def network_pins(names, repo: Path) -> tuple[dict, list[str]]:
    """{name: {path, sha256, timesteps}} (``fields.stages.checkpoint_path`` and the pickle;
    evaluates nothing) and the problems: no finished checkpoint, or one outside the
    repository."""
    from apexzero.fields import stages as S

    pins, probs = {}, []
    for n in dict.fromkeys(names):
        try:
            path = S.checkpoint_path(n)
            if path is None:
                raise FileNotFoundError(f"{n}: no finished run with a checkpoint")
            pins[n] = pin_of(path, S.network_agent_id(n), repo)
        except (FileNotFoundError, ValueError) as e:
            probs.append(str(e))
    return pins, probs


def snapshot_files(run: str, repo: Path) -> list[str]:
    """Every checkpoint of a run directory (C1-dev's 16 snapshots), repo-relative, sorted."""
    from apexzero.interp import registry

    spec = registry.AGENTS.get(run)
    d = Path(repo) / (spec.run if spec is not None else run) / "ckpt"
    return sorted(_repo_rel(p, repo) for p in d.glob("*.pkl")) if d.is_dir() else []


def snapshot_pins(run: str, repo: Path) -> tuple[dict, list[str]]:
    """C1-dev's snapshots of one network pinned like the networks: {'<agent>@<timesteps>':
    {path, sha256, timesteps}} (``pins_problems`` re-hashes them; their loader checks the
    hash) and the problems."""
    from apexzero.fields import stages as S

    out, probs = {}, []
    for rel in snapshot_files(run, repo):
        try:
            pin = pin_of(Path(repo) / rel, S.network_agent_id(run), repo)
        except (OSError, ValueError) as e:
            probs.append(f"{run}: {rel}: {e}")
            continue
        out[f"{run}@{pin['timesteps']}"] = pin
    if not out:
        probs.append(f"{run}: no snapshot to pin")
    return out, probs


def c3w_inputs(nets, repo: Path, completion, competence) -> tuple[dict, list[str]]:
    """C3-w's y and covariate per generalist, with the sha256 of the files they come from
    (``completion(net)``, ``competence(net)``: scripts/costate.py's readers)."""
    from apexzero.costate import analysis as AN

    out, probs = {}, []
    for n in nets:
        run = Path(repo) / AN.network_of(n)
        try:
            out[n] = {"eval_rdev_sha256": sha256_file(run / "eval_rdev.json"),
                      "metrics_sha256": sha256_file(run / "metrics.csv"),
                      "completion": float(completion(n)), "competence": float(competence(n))}
        except (OSError, ValueError, KeyError) as e:
            probs.append(f"{n}: C3-w input: {e}")
    return out, probs


def c3w_problems(rec: dict, repo: Path) -> list[str]:
    """C3-w's recorded input files re-hashed."""
    from apexzero.costate import analysis as AN

    out = []
    for n, v in sorted((rec.get("c3w_inputs") or {}).items()):
        run = Path(repo) / AN.network_of(n)
        for f, k in (("eval_rdev.json", "eval_rdev_sha256"), ("metrics.csv", "metrics_sha256")):
            p = run / f
            if not p.exists() or sha256_file(p) != v.get(k):
                out.append(f"{n}: {f} is not the recorded one")
    return out


def stimulus_design_hash() -> str:
    """Step 1's stimulus design hash (``stimuli.design_hash`` of the bank and cells the
    post-tag roads are rebuilt from): the cells as ``stimuli.load_design``'s list, in
    ``keys._design``'s order, as step 2 hashes them (scripts/fields.py:410-414);
    ``keys.design`` gives them as a dict by id, which ``design_hash`` cannot take."""
    from apexzero.costate import stimuli_dir
    from apexzero.interp import stimuli

    bank, cells = stimuli.load_design(stimuli_dir())
    return stimuli.design_hash(bank, cells)


def _null_shapes_ok(shapes) -> bool:
    """A network's nulls are every C1 null (``analysis.NULL_NAMES`` but the random critics,
    which the critic stage computes) over the same rows: the fold-dependent ones
    (``analysis.FOLD_NULLS``) [N_FOLDS, n, 8] per held-out fold, the others [n, 8]."""
    from apexzero.costate import analysis as AN
    from apexzero.costate import states as ST

    if not isinstance(shapes, dict):
        return False
    rows = set()
    for name in AN.NULL_NAMES:
        if name == "random_critics":
            continue
        s = shapes.get(name)
        want = 3 if name in AN.FOLD_NULLS else 2
        if not isinstance(s, list) or len(s) != want or s[-1] != 8 or \
                (want == 3 and s[0] != ST.N_FOLDS):
            return False
        rows.add(s[-2])
    return len(rows) == 1


def _passes(rel: str, d) -> bool:
    if not isinstance(d, dict):
        return False
    if rel == "inputs/summary.json":
        return bool(d.get("n")) and not d.get("failed") and d.get("t_H") is not None
    if rel == "nulls/summary.json":
        nets = d.get("networks") or {}
        need = {net_file(n) for n in null_networks()}
        have = {net_file(n) for n in nets}
        return bool(need <= have and not d.get("failed")
                    and all(all(v.get("finite", {}).values()) and v.get("sha256")
                            and _null_shapes_ok(v.get("shapes")) for v in nets.values()))
    if rel == "gates/k1.json":
        dec = d.get("decision")
        return bool(isinstance(dec, dict) and dec.get("pass_with_plants"))
    if rel in ("gates/k6.json",):
        return bool(d.get("n"))
    for k in ("pass", "summary"):
        if k in d:
            v = d[k]
            return bool(v.get("pass") if isinstance(v, dict) else v)
    dec = d.get("decision")
    return bool(dec.get("pass")) if isinstance(dec, dict) else False


def bind_outputs(out_root: Path, code_hash_now: str) -> tuple[dict, list[str]]:
    """({rel: {sha256, code_hash}}, problems) of the bound pre-tag outputs (module docstring)."""
    root = Path(out_root)
    out: dict = {}
    probs: list[str] = []
    for rel in BOUND_OUTPUTS:
        p = root / rel
        if not p.exists():
            probs.append(f"{rel} is missing")
            continue
        st = read_stamp(p)
        if st is None:
            probs.append(f"{rel} has no provenance stamp")
        else:
            if st.get("code_hash") != code_hash_now:
                probs.append(f"{rel} was written by other code")
            if st.get("partial"):
                probs.append(f"{rel} is a partial run ({st['partial']})")
        try:
            ok = _passes(rel, json.loads(p.read_text()))
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            probs.append(f"{rel} does not pass")
        out[rel] = {"sha256": sha256_file(p), "code_hash": None if st is None
                    else st.get("code_hash")}
    probs += _cross_problems(root)
    return out, probs


def _cross_problems(root: Path) -> list[str]:
    """Consistency between bound outputs: inputs' t_H is K2's, and the nulls summary's array
    hashes are the files on disk."""
    out = []
    try:
        inp = json.loads((root / "inputs/summary.json").read_text())
        k2 = json.loads((root / "gates/k2.json").read_text())
        if inp.get("t_H") is None or inp.get("t_H") != k2["horizon"]["t_H"]:
            out.append(f"inputs/summary.json's t_H {inp.get('t_H')} is not K2's "
                       f"{k2['horizon']['t_H']}: rerun inputs after k2")
    except (OSError, ValueError, KeyError, TypeError):
        pass                       # a missing file is reported by bind_outputs
    try:
        nl = json.loads((root / "nulls/summary.json").read_text())
        for n, v in (nl.get("networks") or {}).items():
            p = root / "nulls" / f"{net_file(n)}.npz"
            if not p.exists() or sha256_file(p) != v.get("sha256"):
                out.append(f"nulls/{net_file(n)}.npz is not the array its summary hashed")
    except (OSError, ValueError, TypeError):
        pass
    return out


def bind_arrays(out_root: Path, code_hash_now: str, nets=None) -> tuple[dict, list[str]]:
    """({rel: {sha256, code_hash}}, problems) of the arrays the post-tag analysis reads
    (``BOUND_ARRAYS`` and nulls/<net>.npz for every decision network, ``nets``): each must
    exist with a sidecar stamp of this code, not partial, whose recorded sha256 is the
    file's."""
    root = Path(out_root)
    nets = null_networks() if nets is None else nets
    rels = list(BOUND_ARRAYS) + [f"nulls/{net_file(n)}.npz" for n in nets]
    out: dict = {}
    probs: list[str] = []
    for rel in rels:
        p = root / rel
        if not p.exists():
            probs.append(f"{rel} is missing")
            continue
        st = read_stamp(p)
        sha = sha256_file(p)
        if st is None:
            probs.append(f"{rel} has no provenance stamp")
        else:
            if st.get("code_hash") != code_hash_now:
                probs.append(f"{rel} was written by other code")
            if st.get("partial"):
                probs.append(f"{rel} is a partial run ({st['partial']})")
            if (st.get("files") or {}).get(p.name) != sha:
                probs.append(f"{rel} changed after it was stamped")
        out[rel] = {"sha256": sha, "code_hash": None if st is None else st.get("code_hash")}
    return out, probs


def verify_record(rec: dict, repo: Path, keys_root: Path,
                  current_sets: Callable[[], dict] | None = None,
                  out_root: Path | None = None,
                  networks: Callable[[dict], list[str]] | None = None,
                  design_hash: str | None = None) -> list[Problem]:
    """Every way the disk differs from the tagged record; [] when it matches (module
    docstring)."""
    out: list[Problem] = []
    repo = Path(repo)
    if int(rec.get("version", -1)) != VERSION:
        out.append(Problem("record", f"record version {rec.get('version')} is not {VERSION}"))
    sets = {k: [tuple(t) for t in v] for k, v in rec.get("key_sets", {}).items()}
    for k, ids in sets.items():
        if FP.content_hash(ids, keys_root) != rec.get("content", {}).get(k):
            out.append(Problem("keys", f"key set {k}: the key files differ from the tagged ones"))
    if current_sets is not None:
        now = current_sets()
        for k in sorted(set(now) | set(sets)):
            if [tuple(t) for t in now.get(k, [])] != sets.get(k, []):
                out.append(Problem("keys", f"key set {k}: the index no longer gives the tagged "
                                   "list"))
    if t_hash([tuple(p) for p in rec.get("t_pairs", [])]) != rec.get("t_sha256"):
        out.append(Problem("record", "T's list does not match its hash"))
    doc = repo / DESIGN_DOC
    if not rec.get("design_doc_sha256"):
        out.append(Problem("record", "the record registers no design document"))
    elif not doc.exists() or sha256_file(doc) != rec["design_doc_sha256"]:
        out.append(Problem("design_doc_changed", f"{DESIGN_DOC} differs from the tagged design"))
    if code_hash(repo) != rec.get("code_hash"):
        out.append(Problem("code_changed", "code changed after the record was written"))
    if out_root is not None:
        for rel, v in (rec.get("outputs") or {}).items():
            p = Path(out_root) / rel
            if not p.exists() or sha256_file(p) != v["sha256"]:
                out.append(Problem("outputs", f"{rel} differs from the tagged output"))
        if not rec.get("outputs"):
            out.append(Problem("record", "the record binds no pre-tag outputs"))
    if networks is not None:
        pins = rec.get("networks")
        if not pins:
            out.append(Problem("record", "the record pins no network checkpoint"))
        else:
            out += [Problem("checkpoints", m) for m in networks(pins)]
        snaps = {k: v for d in (rec.get("snapshots") or {}).values() for k, v in d.items()}
        if snaps:
            out += [Problem("checkpoints", m) for m in networks(snaps)]
        out += [Problem("c3w", m) for m in c3w_problems(rec, repo)]
    if design_hash is not None and design_hash != rec.get("design_hash"):
        out.append(Problem("design", "step 1's stimulus design differs from the recorded one"))
    return out


def pins_problems(pins: dict, repo: Path) -> list[str]:
    """Each recorded checkpoint re-hashed (path and sha256)."""
    out = []
    for name, pin in sorted(pins.items()):
        p = Path(repo) / pin["path"]
        if not p.exists() or sha256_file(p) != pin["sha256"]:
            out.append(f"{name}: the checkpoint {pin['path']} is not the recorded one")
    return out


def frozen_ids(rec: dict, name: str) -> list[tuple[str, str]]:
    return [tuple(t) for t in rec["key_sets"][name]]


def frozen_t(rec: dict) -> list[tuple[str, int]]:
    return [(str(k), int(t)) for k, t in rec["t_pairs"]]


def save(rec: dict, path: Path = PREREG_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=_default) + "\n")
    return path


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")
