"""Fixed choices of the psychophysics protocol, in one place.

Protocol 1.4 (fixed choices), 3.5 (grid and approaches), 4 (rollouts), 5 (measure
thresholds), 6 (key recipe, tolerances, active sets) and 10 (budget, logging). Every module
and every builder reads these names instead of retyping numbers, so a value can only differ
between modules by a deliberate edit here. Nothing in this file may change after the tag
``psychophysics-prereg-v2``; a change is a deviation and is logged.

Units: SI (m, s, m/s, rad, 1/m) unless a name ends in ``_KPH`` (km/h) or ``_DEG`` (degrees).
Thresholds that belong to one decision rule (P1-P8) live next to that rule in analysis.py.
"""

from __future__ import annotations

import math

KPH = 1.0 / 3.6                      # multiply km/h by this to get m/s


def kph(v: float) -> float:
    """km/h -> m/s."""
    return v * KPH


# ----------------------------------------------------------------------- 1.4 strip layout

DS = 2.0                             # knot spacing (m); WindowSolver rejects |ds - 2| > 0.05
N_KNOTS = 2560                       # knots per stimulus strip
STRIP_LENGTH = (N_KNOTS - 1) * DS    # 5118 m (open strip: (n - 1) ds)
STRIP_TAIL = 520.0                   # racing.EnvParams.strip_tail: truncation this far before the end
S_TRUNC = STRIP_LENGTH - STRIP_TAIL  # 4598 m: runs truncate at s > S_TRUNC
S_C = 1000.0                         # corner start s_c0 (start of ramp in) on every strip
COMPLETE_AFTER_EXIT = 250.0          # completion: no termination before s_c1 + 250 m

RAMP_MIN = 10.0                      # abrupt ramp (tier E3) and the generator's ramp_min (m)
INSIDE_CAP_MARGIN = 0.6              # w_in(s) = min(w_in, 1 + 0.6 / |kappa(s)|), generator margin
WIDTH_CORE = 12.0                    # total width of tiers A, B, E and G (m)

# l_typ(R): median generator ramp per R_seg bin, interpolated linearly in log R through the
# bin medians placed at these radii and held constant outside. Values are frozen in step 1
# (support.l_typ_table); the expected values are only for checks and reports.
L_TYP_BIN_EDGES = (11.7, 18.3, 28.6, 55.9)   # R_seg bin edges (m): 5 bins, top one open
L_TYP_R_NODES = (9.4, 14.6, 22.9, 40.0, 75.0)
L_TYP_EXPECTED = (26.0, 33.0, 40.0, 44.0, 44.0)
THETA_CLASS_EDGE_DEG = 120.0         # theta classes: < 120 deg and >= 120 deg (v_typ, E2)

# ------------------------------------------------------------------------- car and seeds

HIDDEN_MU = 1.0
OBS_NOISE = 0.0
ROOT_SEED = 20261019
# Seed streams other than the policy noise. Policy noise uses
# fold_in(fold_in(PRNGKey(ROOT_SEED), noise_id), sample) exactly as protocol 4 states, with
# noise_id < 10**6; every other stream folds in STREAM_OFFSET + its index so the two can
# never collide.
STREAM_OFFSET = 10**6
STREAMS = {
    "support": 1,       # 300 training strips per grammar
    "twin": 2,          # twin-fit initialisation
    "e2_ties": 3,       # E2 matched-corner tie breaks
    "f6": 4,            # tier D F6 kappa-noise and wobble phases
    "knn_buffer": 5,    # S1 Hockenheim laps for K (random s and speed)
    "dry_run": 6,       # simulated agents of the step-5 dry run
    "bootstrap": 7,     # descriptive bootstrap bands
    "permutation": 8,   # P6a permutation test
    "viability": 9,     # seeded subset beyond the viability-solve cap
}
SUPPORT_N_STRIPS = 300

# ------------------------------------------------------------------------- 1.4 noise, keys

DT = 0.05                            # control step (s)
N_PHASES = 4                         # phases u_k = (k / 4) v_sight dt, k = 0..3
N_PHASES_P7 = 16                     # phases on P7 cells
MATCHED_NOISE_STD = (0.04, 0.08)     # matched-noise condition (steer, pedal)
BACKOFF_ALPHA_MAX = math.radians(8.5)  # backed-off key slip limit (rad); nominal 10 deg
BACKOFF_SHRINK = 0.35                # backed-off key: corridor narrowed per side (m)

# Speed band b (V8 rule, protocol 6.5): clipped to [5%, 10%], rounded to 0.5 points.
BAND_LIMITS = (0.05, 0.10)
BAND_EXPECTED = 0.075
BAND_ROUND = 0.005
V8_EXCESS = (0.05, 0.075, 0.10)      # e in vx(s_c0) <= (1 - e) v_e^sOCP
V8_TIME_COST = 0.10                  # s: b is the e where the median far-field cost reaches this

# ------------------------------------------------------------------------- 3.5 grid

R_BRAKING = (6.0, 7.5, 8.4, 9.4, 10.5, 11.7, 13.1, 14.6, 16.4, 18.3, 22.9, 28.6, 35.8,
             44.7, 55.9, 69.8, 87.3)
R_CONTROLS = (109.0, 171.0, 300.0)   # 90 deg only
R_180_MAX = 55.9                     # tier A 180 deg cells: R <= 55.9
R_NOCAP = (6.0, 7.5)                 # no-cap variants, 90 and 180 deg, A2 only
THETA_A_DEG = (90.0, 180.0)

A1_V0_KPH, A1_D = 310.0, 600.0       # fast approach; corner hidden at the start
E_FAST_V0_KPH, E_FAST_D = 280.0, 250.0
V_TYP_CHOICES_KPH = (140.0, 160.0, 180.0, 200.0, 220.0, 250.0)
A2_D_MARGIN = 25.0                   # D = pinned s-OCP brake-demand onset distance + 25 m
A2_D_STEP = 10.0                     # pre-tag adjustment step of D
A2_VB_TOL = 0.10                     # v_b* within +-10% of v_typ ...
A2_VB_FRACTION = 0.90                # ... in at least 90% of braking A2 cells
A2_FLAT_D = 250.0                    # D when the key does not brake at v_typ

TIER_B_R = (7.5, 9.4, 11.7, 14.6, 18.3, 22.9)
TIER_B_THETA_DEG = (90.0, 180.0)
TIER_B_V0_KPH = (160.0, 220.0, 280.0, 310.0)
TIER_B_D = (250.0, 600.0)
RECOVERY_D = 100.0
RECOVERY_V0_KPH = (220.0, 280.0, 310.0)
RECOVERY_R = (9.4, 14.6)

TIER_C_45_R = (14.6, 22.9, 44.7)
TIER_C_135_R = (9.4, 14.6, 22.9, 44.7)
TIER_C_WIDTH_R = (9.4, 14.6, 22.9, 44.7)
TIER_C_WIDTHS = (9.0, 15.0)

E3_R = (9.4, 11.7, 14.6, 18.3)       # 90 deg
E3_W_IN = (4.6, 6.0)                 # inside half-width at the corner, w_in^c (m)
E3_RAMPS = ("abrupt", "typical")

G_V0_KPH = (140.0, 160.0, 180.0, 200.0, 220.0, 250.0, 280.0, 310.0)
G_LENGTH = 4000.0                    # null straight length (m)
G_WIDTHS_EXTRA = (9.0, 15.0)         # at the v0 values tier C uses

O_R_TIGHT = (9.4, 11.7)              # x (90, 180) deg
O_R_WIDE = (14.6, 28.6, 55.9)        # x (45, 90, 180) deg
O_V_KPH = (150.0, 200.0, 250.0, 300.0)
O_D_STEP = 0.25                      # scan step (m)
O_D_MARGIN = 50.0                    # scan from d_b^agent(v) + 50 m to d_max

P7_R = (9.4, 11.7, 14.6, 28.6, 55.9)  # x (90, 180) deg at A1, 16 phases
P7_DELTAS = (0.0, 0.25, 0.5, 0.75)    # corner shifts in units of the preview spacing

VALIDATION_RT = ((13.0, 180.0), (13.0, 90.0), (40.0, 90.0), (40.0, 180.0), (100.0, 60.0),
                 (100.0, 90.0))      # at A1 and A2
VALIDATION_IMPOSED_RT = ((13.0, 90.0), (40.0, 90.0))   # at (310 km/h, 100 m)
V7_N_CELLS = 10

# Pilot mini-grid replication (protocol 1.1, step 2), in the pilot's own layout (see
# stimuli.pilot_minigrid): width 11 (5.5 / 5.5, no cap), 10 m ramps, corner start 800 m on a
# 1280-knot strip, x0 = [100, 0, 0, 80, 0, 0, 0, 0], 500 deterministic steps.
PILOT_R = (9.0, 11.0, 13.0, 16.0, 20.0, 30.0)
PILOT_THETA_DEG = (90.0, 180.0)
PILOT_WIDTH = 11.0
PILOT_N_KNOTS = 1280
PILOT_S_ENTRY = 800.0
PILOT_S0 = 100.0
PILOT_V0 = 80.0
PILOT_STEPS = 500

# Tier D (3.3) and tier E (3.4).
D_GAP_CLIP = (100.0, 600.0)          # D_gap = clip(gap to previous corner, 100, 600)
D_LEAD_IN = 100.0                    # real road, real width, at the start of every condition
D_AFTER = 1200.0                     # strip runs to s_e + 1200 m
D_REGION_MARGIN = 20.0               # corner region [s_c - 20, s_e + 20]
TAPER = 20.0                         # kappa tapers at every cut and substitution (m)
D_CONST_WIDTH = 12.0
D_DOSES = (0.8, 0.9, 0.95, 1.05, 1.1, 1.25)
D_WIDTH_DOSES = (-0.5, 0.5)          # whole-strip width change (m, total)
D_SWAP_LOG_RATIO = 0.35              # |log(v_i*/v_j*)| target ...
D_SWAP_MIN_LOG_RATIO = 0.15          # ... and minimum
F6_WOBBLE_AMPL = (0.3, 0.6)          # m, both edges in phase
F6_WOBBLE_WAVELENGTH = 60.0          # m
F6_KAPPA_NOISE = (0.5e-3, 1e-3)      # 1/m, band-limited, zero net heading
F6_KAPPA_BAND = (80.0, 200.0)        # wavelengths (m)
E_CUT_BEFORE, E_CUT_AFTER = 800.0, 400.0
TWIN_TOL = {"R_seg_rel": 0.02, "angle_deg": 1.0, "length": 2.0}
E2_MIN_DISTANCE = 1000.0             # matched corner at least 1 km from any crash corner

# ------------------------------------------------------------------------- 4 rollouts

CHUNK = 2048                         # episodes per compiled call
OBS_CLIP = 10.0                      # RunningMeanStd.normalize clip
UNSETTLED_N = 1.0                    # |n| > 1 m at first sight ...
UNSETTLED_PEDAL = 0.9                # ... or pedal < 0.9 -> unsettled
REPLICATION = {                      # runs per cell and direction
    "final": {"det_phases": None, "native": {"A": 32, "other": 16}, "matched": {"A": 16, "E": 16}},
    # det_phases None: every phase of the cell (16 on P7 cells); 4: the 4-phase grid
    "last3": {"det_phases": N_PHASES, "det_tiers": ("A", "B"), "native": {"A": 8}},
    "other_generalist": {"det_phases": N_PHASES, "det_tiers": ("A", "G"), "native": {"A": 8}},
    "other_specialist": {"det_phases": N_PHASES, "det_tiers": ("D", "G")},
}

# Observation channel groups for clipping statistics (racing.observe layout, 222 channels).
OBS_GROUPS = ("kappa", "edges", "ego", "context")
OBS_PREVIEW, OBS_EGO, OBS_CONTEXT = 200, 16, 6

# Termination flags as stored in rollout traces (uint8 bitmask per step), and the crash
# cause precedence of protocol 5 (corridor > slip > spin > stall > non-finite).
FLAG_BITS = {"off_track": 0, "slipped": 1, "spun": 2, "stalled": 3, "nonfinite": 4, "truncated": 5}
CAUSE_PRECEDENCE = ("corridor", "slip", "spin", "stall", "nonfinite")

# ------------------------------------------------------------------------- 5 measures

BRAKE_ON = 0.3                       # beta_f threshold of a sustained brake
BRAKE_OFF = 0.1                      # release threshold
BRAKE_MIN_STEPS = 3                  # consecutive steps above BRAKE_ON
FORCED_ONSET = 0.5                   # s: an onset within 0.5 s of the start is forced
RESP_DELTA = 0.1                     # |pedal - pedal_null| threshold of the response onset
RESP_STEPS = 2
LEGACY_PEDAL = -0.2                  # v1 onset: pedal state <= -0.2 ...
LEGACY_STEPS = 3                     # ... for 3 steps
V_MIN_MARGIN = 50.0                  # v_min over [s_c0 - 50, s_c1 + 50]
ETA_OFFSET = 30.0                    # eta at s_c0 - 30 and s_c1 + 30
LINE_RMS_MARGIN = 100.0              # RMS(n - n*) over [s_c0 - 100, s_c1 + 100]
DV_BEFORE, DV_AFTER, DV_STEP = 300.0, 100.0, 1.0   # Delta v(s) grid
V_TOP_KPH = 315.0                    # car top speed (protocol 1.2)
FF_FRACTION = 0.97                   # s_ff: first s > s_c1 with both vx >= 0.97 * V_TOP
OSC_WINDOW = 5.0                     # s, oscillation index window
OSC_GATE = 0.1                       # checkpoint gate: oscillation index < 0.1
GATE_COMPLETION = 0.9                # checkpoint gate: deterministic completion >= 90%
CRASH_UTIL = 0.98                    # crash typing: util >= 0.98 ...
CRASH_UTIL_STEPS = 3                 # ... for >= 3 steps
LINE_STEER_UTIL = 0.9                # line/steer: max util < 0.9 over the last second
CRASH_LOOKBACK = 1.0                 # s
A_DEC_PEDAL_STEP = 0.005             # a_dec(v) table: pedal sweep step on [-1, 0]
A_DEC_V_STEP_KPH = 1.0               # a_dec(v) grid step
A_Y_V_GRID_KPH = (20.0, 315.0, 5.0)  # a_y,max(v) grid (start, stop, step)

# ------------------------------------------------------------------------- 6 keys

W_SHORT, W_LONG = 400, 800           # window lengths (steps)
W_RULE_TIME = 5.0                    # s: W = 400 if the pinned s-OCP reaches s_c1 within 5 s
CONT_STAGES = (200, 400, 800)        # continuation stages up to W
STAGE_SKIP_MARGIN = 300.0            # skip a stage whose window ends before s_c1 + 300 m
SOCP_BACK = 20.0                     # pinned_strip's default lead-in (m) before the s0 node
SOCP_LEAD_IN = 0.0                   # the keys' s-OCP: no free lead-in in the NLP (the s0 node
                                     # is node 0, z[:, 0] = z0); a time-minimised lead-in leaked
                                     # into the pinned solution through the pin's control
SOCP_AFTER_EXIT = 600.0              # s-OCP runs to s_c1 + 600 m
IPOPT_STAGE_MAX_ITER = 800           # per continuation stage
START_MAX_CPU = 400.0                # s of CPU per start; a capped start counts as failed
RETRY_MAX_ITER = 3000                # one retry when every start failed
START_DISAGREE_V_KPH = 2.0           # add the Q start if starts (1) and (2) disagree by > 2 km/h
START_DISAGREE_J = 1e-4              # ... or by > 1e-4 relative in J
NEAR_EQUAL_RTOL = 1e-4               # near-equal set N
LOGGED_RTOL = 1e-3                   # starts within (1e-4, 1e-3] are logged
MULTI_BASIN_FLOORS = {"d_b": 1.0, "v_e": kph(0.5), "v_min": kph(0.5), "s_vmin": 2.0,
                      "line_apex": 2.0}
AGENT_BEATS_KEY_RTOL = 1e-6
REPLAY_RTOL = 1e-6                   # = ocp.replay.REPLAY_RTOL
ANCHOR_LEAD = 1.0                    # s: s_L = s_c0 - (d_b* + v_b* * 1.0 s)
REANCHOR_IMMATERIAL_KPH = 1.0        # P6 rule: |v_e*RA - v_e*| <= 1 km/h ...
REANCHOR_IMMATERIAL_FRACTION = 0.9   # ... in >= 90% of A1 cells
VIABILITY_W = 200
VIABILITY_CAP = 200                  # solves in all
VIABILITY_SUBSET = 10                # crashed pairs per agent beyond the cap
ACTIVE_SLIP = math.radians(0.5)      # slip within 0.5 deg of alpha_max
ACTIVE_STEER = math.radians(2.0)     # steering within 2 deg of delta_max
ACTIVE_CORRIDOR = 0.05               # corridor within 5 cm
ACTIVE_WINDOW = 1.0                  # s around v_min for the cell's limit flags
BOUND_FACTOR = 1.1                   # infeasibility bound: 1.1 a_dec, 1.1 a_y,max, 1.1 rho
UNSOLVED_MAX_FRACTION = 0.05         # > 5% unsolved P0 cells -> investigate before analysis

# ------------------------------------------------------------------------- 10 budget

KEY_WORKERS = 5
ROLLOUT_PROCESSES = 3
XLA_THREADS = 2
BUDGET_TOTAL_CPU_H = 60.0            # drop P2 items above this projection
BUDGET_P0_CPU_H = 35.0               # cut tier A 180 deg R >= 35.8 and phase keys above this
MIN_ROLLOUT_SPS = 15_000             # below: drop checkpoint stochastic and E matched-noise runs
LOG_EVERY = 60.0                     # s between done/total lines in stage logs
