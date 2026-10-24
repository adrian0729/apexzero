"""Idea 2's gates before any trained network (design v2.1, Section 7: K0-K6).

**K0, targets from files** (``k0_key``, ``k0``): ``targets.k0_record`` on every key (the x_0
identity to ``targets.X0_TOL`` = 1e-12, the anchored a_0 companion, the two routes on T's
rows and the cheap route's end gap to ``targets.ROUTE_TOL`` = 1e-10, the pedal fold's
stationarity to ``targets.FOLD_TOL`` and its command signs (a fixed command, an anchored
key's a_0, has no sign), the action rows to ``targets.ACTION_TOL`` with every planted bug
above ``targets.PLANT_MIN``, sigma read from the folded command multipliers, lambda_n = Pi^R -
Pi^L where the road ahead is straight to ``targets.STRAIGHT_TOL``, every target finite), and
the change against the stored (un-re-polished) multipliers (``targets.targets_change``). K0
passes when every key whose targets a pre-registered computation reads (the A2 keys and K1's
sample; ``k0_summary``'s scope, design Section 18) is 'ok' and every key's planted action-row
bugs fail. N's other anchored finals are checked and reported; a failure among them is
listed in 'not_ok_descriptive' (N is deferred, Section 14).

**K1, targets against re-solves** (``k1_sample``, ``k1_key``, ``k1_decision``). The 50 keys
are stratified as step 2's Gate 1 (``fields.gates.GATE1_STRATA``, ``family``; drawn on step
2's 'gate1_sample' stream, gates.py:209-230, from idea 2's key roots, so the sample is step
2's own). Each key is checked at its decision (t = 0; anchored keys t = 1) and at 3 plan
states (approach, braking, corner: the middle T row of each phase with t >= 1), plus the
coverage states sampled explicitly (``k1_states``: a corridor contact, a slip contact and a
pedal-saturated state, the middle of each kind's steps within T's range), every component
(``targets.fd_costate_k1``: Richardson over (c, c/2) of pinned-prefix re-solves at tol 1e-10,
K1's steps ``targets.FD_H``, a smaller c where the pair disagrees; at pedal-saturated states
the pedal component one-sided from below on its own ladder from c = 1e-3,
``targets.fd_pedal_below``, the share of those rows that count reported). Pass per
component: over the comparisons that count (re-solves converged and not a kink, ``FD.kink``:
|forward - backward| > 10 max(discrepancy, ``FD.noise``), ``FD.noise`` = 6 KINK_J_NOISE / (h
gamma^t)), median relative error (``k1_error``: Gate 1's score with its zero rule and
denominator at the difference's floor eps_t = max(1e-6, ``FD.noise``)) < 1e-3 and p95 <
1e-2; a comparison whose Richardson pair still disagrees after the smaller c
(``unresolved``) does not count either. A zero comparison (|lambda| and |FD| both below the
floor) is Gate 1's zero field: counted per component, per plant and per reading ('n_zero')
and never pooled. A component with more than 30% kinks or unresolved comparisons
(``K1_KINK_SHARE``), or with no non-zero comparison that counts, is descriptive. Two
readings are reported beside it and bind nothing ('design_rule' and
'design_rule_literal_kink', each per component and as a verdict over its own decided
components): Section 7's zero rule to the letter (``k1_design_error``: Gate 1's 1e-6, zero
comparisons counted, unresolved comparisons scored) with the coded kink flag, and the same
with Section 7's kink rule to the letter too (``literal_kink``: |forward - backward| > 10
discrepancy, from the one-sided estimates every row stores). Planted bugs must fail the same
rule on the same re-solves (``targets.PLANTS``; the pre-jump form scored on contact states
only, 'unfolded' on the pedal rows whose fold is active; on anchored decisions also 'j_w',
J_W read as J_tail: gamma lambda_1 + e_s/10); none may pass vacuously (every required plant
scored on a non-zero comparison of a decided component). pi on the anchored keys
(``k1_pi``): ``targets.fd_pi`` with its planted bugs (``targets.PI_PLANTS``), required
whenever anchored keys are checked, under the same rule (``k1_error`` at fd_pi's floor
max(1e-6, 6 KINK_J_NOISE / (h gamma)), h = ``targets.PI_H``): a zero comparison of pi or of
a plant is counted ('n_zero') and never pooled, pi needs a non-zero comparison, and each
plant must fail on its non-zero ones. The record binds K1 on ``pass_with_plants``.

**K2, horizon** (``twin_pairs``, ``k2_twins``, ``k2_horizon``, ``k2_extend``). On each tier
V twin pair (the W = 400 and W = 800 nominal keys of one cell), |lambda^400_t -
lambda^800_t| / SD_T(lambda_k) at every t < 400 and on T's steps; t_H is the largest t up to
which the p90 over the pairs stays <= 0.1 for every K_P component (Section 7 'Retention';
expected about W - 130, beyond T's largest t); whether the cut drops any of T's states is
judged on T's own uncut t_end (``k2_retention``: the W = 400 keys that would need the
Fallback are listed). ``k2_extend`` is the insurance solve: a
W = 400 key continued to W = 800 (``window.solve_continuation``, window.py:837-890, the
stored optimum as the first guess, ``answer_key.extend_straight`` beyond it), rebuilt as a
key problem (``keys.rebuild``) and re-polished, its targets compared on the key's T steps; a
different basin (the plans' states on T's range apart by more than ``K2_BASIN_DX``) is
flagged.

**K2b, road beyond the preview** (``k2b_key``). A second corner of the key's own direction
(``second_corner``: a trapezoid of curvature, 10 m ramps, arc R theta) starts
``K2B_MARGIN`` beyond max over T's states of s_t + d_max(vx_t); two severities
(``K2B_SEVERITIES``, fast and slow). The key is re-solved on the variant strip
(``perturb.perturbed_track``) by warm continuation in amplitude (``K2B_RAMP``) from the stored
optimum, with the window centres re-centred on each solution (``_recentre``) until no window
bound binds; |Delta lambda_k| / SD_T per K_P component on T's rows by phase. Rule
(``k2b_rule``): the p90 pooled over every sampled key's rows in a phase stratum, under the
worse severity, above 0.1 excludes component k from decisions there (the record freezes the
rule; ``states.decision_mask`` applies it).

**K3** (``k3_judge``): the critic's pullback dry runs (``python -m apexzero.costate.critic
dry-run plants|random``, the random one on ``critic.K3_MIN_STATES`` = 200 T states) and the
extractor's self-consistency (``python -m apexzero.costate.probes extractor-check``, with
heads and without), judged on their stated tolerances (critic.py's and probes.py's
constants): every closed form and zero the plants report, the composition's autodiff on the
random critics, the sample size, and observability on the exit, before s_c1, at the apex and
on float32-upcast key arrays. **K3's pass rule** (design v2.2 Section 13, fixed after its
full-size pre-tag run, random and planted networks only): the pullbacks' finite differences
(``pullback_fd``: main V and heads, every form present) pass for the held and x_{t-1}-held
forms below ``critic.FD_TOL`` = 1e-6 and for the on-manifold form below
``critic.FD_TOL_MANIFOLD`` = 1e-5 (``critic.FD_TOLS``); and every planted wrong pullback
must exceed ``critic.WRONG_MIN`` = 1e-4, ten times the on-manifold tolerance, or K3 fails:
the accel-reading plant's wrong closed forms g_held, pi_held and g_xprev
(``accel_normal_held_fails``), and each AD form scored on every other form's finite
differences on the accel-normal plants with c > 0 and the random linear critic
(``wrong_pullbacks_fail``, none missing: ``k3_wrong_missing``), with those plants' own forms
matching their finite differences under the same per-form tolerances (``plant_fd``; the
planted linear critic's ``linear_fd`` too, every form present). The record
carries the tolerances, each form's errors and the smallest planted error
('planted_wrong').

**K3 after the tag** (``k3_trained``, ``k3_trained_judge``; Section 7 K3 'trained networks
after it, on D' and 'After the tag'): on each trained network, in the critic stage's float64
process, at D's raw observations: the extractor's self-consistency by both extractors to
``probes.EXTRACTOR_TOL`` = 1e-10 (the critic's V and heads against ``Critic.apply(...,
with_heads=True)``, the actor's mean against ``Actor.apply``), every head count (config,
tree, Flax, probe extractor) equal to ``expected_heads`` (5 for a generalist, 0 for S1-S3),
and the float64 V against the critic run in float32 with x64 off to ``critic.F32_TOL`` =
1e-4. A network that fails is treated as missing in every decision it enters
(``analysis.k3_trained_failed``).

**K3b** (``k3b``): ``own.k3b_start`` on step 2's linear driver, its accel-reading plant
(``own.accel_driver``) and its hybrid control on A2 cell starts (network.py:388-460), with the
on-manifold check, pooled by ``own.k3b_decision``.

**K4** (``k4``): planted hidden layers through the whole probe pipeline on T's rows (z under
G1's obs_rms, random-init actors of G1's own architecture, the frozen folds and key
weights; the host is actor seed 0, the controls seeds 1-3, ``K4_N_CONTROLS``, except
PR-6b and PR-6c): PR-1a (i.i.d. targets; held-out R2 and Delta2 within 0.03 of SNR/(1 +
SNR)), PR-1b (real lambda_res, all four K_P components planted at once; partial R2 within
0.03 of SNR/(1 + SNR) and Delta2 of (1 - R2_b) SNR/(1 + SNR)), PR-2 (m and a*_t in the
layer: the paired Delta2 within 0.01 of 0), PR-3 (one-hot geometry and pair codes: the
paired gain <= 0.01 on the grouped split; dealt by key, on each row's out-of-fold
lambda_res, a leak: the codes' unpaired gain or the base R2's rise above 0.01, the paired
gain reported beside it), PR-4 (``pr4``: a readout of z in h2; h2 and h3 each recover at
least the readout's R2 minus 0.02, one-sided, since the probe on h2 or h3 also sees the
layer's other units and can exceed the readout), PR-5, 5b, 5c, 5d (lambda + noise as the
critic's gradient through C1 at SNR 0.25-4 on C1's decision rows, expected SNR / (1 +
SNR)), PR-5e (the floor: zero, noise and -lambda gradients score about 0 per component, a
direction-only one at most 0.02 on xi; PR-5's rules read C1's decisions as the analysis
records them, ruled by the single-geometry rule, ``probes.LOGO_RULE``, and each row keeps the
full set's decision and the sets), PR-6, PR-6b and PR-6c (below), PR-7 (scale
invariance to 1e-6), PR-8 (C2-sw's gain within 0.03 on C2-sw's stratum), PR-9 (heads sharing
V: D_V = 0 and the planted D_grad recovered). K4 passes when every plant does.

**K4's no-gain plants** (v2.3, the third decision before the tag, on grounds independent of
any result: a plant's effect is the change it causes in the same network). Every K4 plant
whose rule is a no-gain tolerance (PR-2 in both variants, PR-3's grouped split, PR-6, PR-6b
and PR-6c, and PR-6b's and PR-6c's residualisation-off check and planted faults) is scored by
its paired increment (``probes.c2_paired``, ``K4_PAIRED``): Delta2(host + plant) -
Delta2(host), with the same host, random controls, rows, components and bootstrap draws (a
paired bootstrap for the CI), so the controls' median cancels and the statistic is R2(host +
plant) - R2(host). The unpaired statistic compares one random host plus the plant with other
random networks, so it carries the host's own chance offset against the controls (about
0.007 on T at full size, close to the 0.01 tolerance, and re-drawn by any pipeline change).
Tolerances, seeds and every-seed rules are unchanged. Each plant's unpaired value
('unpaired') and the host's own offset ('host_offset': the host alone against that plant's
controls on its rows, with every random network's own gain and their SD; per plant and per
seed in the record's 'host_offset', ``host_offsets``) are descriptive and stay in
gates/k4.json; nothing in the analysis reads them. They are K4's offsets (T, K4's controls,
the plants' rows), not C2's noise floor on its decision rows, which is ``c2_null``'s. The
recovery plants (PR-1a, PR-1b, PR-5, PR-8) keep their rules, and so does PR-3's mis-grouped
leak check (its paired leak 'by_key_paired' reported).

**K4's PR-1b and PR-6 rules** (v2.3, fixed after the full-size pre-tag run of K4 on random
and planted networks only, before the tag; the K4 review's verified write-up):

- PR-1b plants on one target array for its baseline, the probe and the random controls:
  each row's own-fold lambda_res (``states.Residuals.oof``). C2's probe trains, while fold f
  is held out, on the cross-fitted lambda - f_D^(-f,-g), which changes with f; one planted
  column cannot follow it and carried on training rows the held-out fold's share of the
  decoy fit, which under-recovered it (partial R2 0.70-0.75). Its rule is unchanged, so it
  checks the ridge machinery (both penalties, inner CV, scaling), not the cross-fit; the
  plant on C2's by-fold targets made to follow them on training rows (``pr1b_by_fold``) is
  reported beside it, descriptive. All four components are planted at once (the design's
  one at a time differs by at most about 0.008 on T).
- PR-6 (``pr6``, real lambda): 32 units tanh(gain (Ds A + b)) of the decoys (s - s_apex,
  vx, sigma, log R) with random biases (``probes.decoy_units``), the gain from the fixed
  grid ``K4_PR6_GAINS`` whose probe-alone R2 lies closest to the decoy kNN's (the R2 by gain
  and 'calibrated', within ``K4_CALIBRATION_TOL``, recorded; no 32-unit family reaches the
  kNN). Pass: the paired Delta2 <= ``K4_ZERO_TOL`` one-sided on C2's by-fold lambda_res, and
  PR-1b passes. Units carrying nothing beyond z, m and h3 cost out-of-fold R2 on held-out
  geometries, so the null expectation is negative: the negative part is reported as the
  plant's over-fitting cost ('overfit_cost'), the unresidualised Delta2 beside it.
- PR-6b and PR-6c (``pr6_world``, known truth): y = std(F) + 0.5 eps per K_P component,
  eps an AR(1) series (rho 0.9) along each key, independent across keys, 20% of Var y, on
  ``K4_PR6_SEEDS`` (3 draws); 32 units hold F (tanh of random mixes of F, ``hold_units``),
  appended to the host's h3. PR-6b's F is smooth (``pr6b_f``: a ridge of lambda on 256
  biased tanh features of the PR-6 decoys); PR-6c's matches the decoy kNN (``pr6c_f``: the
  leave-one-geometry-out k = 15 kNN of lambda on D1, ``K4_PR6C_K``, the world's own k
  whatever f_D's). y is residualised by the pipeline's own cross-fit (``states.cross_fit``
  at f_D's one k, ``states.KNN_K``). The statistic is
  C2's primary one (``c2_primary``: C2's decision stratum, each component on its decision
  rows, the frozen power floor's components, C2's 5 random-init controls), the units'
  paired increment over the host. K4 keeps the 5 controls (and its other plants the 3):
  the decisions' control pool and two-level CI (``probes`` module docstring 'Two-level
  CI') do not enter K4, whose no-gain plants are paired increments, in which the controls'
  median cancels. Pass per world: on every seed Delta2 <= ``K4_ZERO_TOL``,
  and on every seed Delta2 > ``K4_ZERO_TOL`` with the residualisation turned off; for PR-6b
  also the units' probe-alone R2 at least the kNN's on every seed (mean over K_P on T).
  T-wide values (every row and component, C2's controls) and per-component gains are
  descriptive, as are two planted faults on the first seed (an over-smoothed kNN, k =
  ``K4_OVERSMOOTH_K``, capped at the cross-fit's smallest training set, ``oversmooth_k``,
  which binds only on a T as small as the rehearsal's; D1 without s - s_apex, at f_D's k;
  each fault's k recorded). A stated limitation: at full
  size the second fault was not detected on the gate seeds (C2's decision rows score s on
  approach and exit rows and xi on approach rows only; the T-wide +0.017 and +0.020 the K4
  review saw did not reproduce there).
- ``band_notes``: PR-6's over-fitting cost and PR-6c's leak bound (its largest Delta2 over
  the seeds), both paired, which the analysis prints beside a C2 decision near the band
  (``analysis.band_notes``; copied into the record from the bound gates/k4.json). Both are
  measured on C2's setup (lambda_res through f_D, the base [z, m]), so they are not printed
  beside C2-sw, whose sigma^ped is not residualised.

**f_D's k** (v2.3, the second decision before the tag; ``states`` module docstring for the
grounds): one k for every fit, fold, partition and K_P component, chosen on synthetic known
truth only, on validation seeds disjoint from K4's gate seeds. ``decoy_k_leaks`` builds
PR-6b's and PR-6c's worlds exactly as K4 does (``pr6_world``: the same F, units, C2's
decision rows, controls and statistic) on each of ``DECOY_K_SEEDS`` (6 noise draws,
``decoy_k_rng``: stream ``costate_decoy_k``, sub-stream v, shared by both worlds) with y
residualised at the candidate k and C2's power floor recomputed at k (``at_decoy_k``,
``prereg.frozen_values``' rule); ``decoy_k_choice`` takes among ``DECOY_K_CANDIDATES`` = (3,
5, 8, 15) the largest k whose every validation seed of both worlds gives C2's primary Delta2
<= ``K4_ZERO_TOL`` (none: the smallest, a blocker). The rule was re-applied unchanged once the
no-gain plants became paired increments ('K4's no-gain plants'): the statistic it reads is
the units' paired increment. The full-size run's paired Delta2 per seed is
``DECOY_K_VALIDATION`` (the same fits' unpaired Delta2 ``DECOY_K_VALIDATION_UNPAIRED``,
descriptive) and its choice ``states.KNN_K`` = 8, as under the unpaired statistic: k = 3, 5
and 15 each have a PR-6c or PR-6b seed above 0.01 (largest 0.0130, 0.0165 and 0.0114), k = 8
none (largest 0.0035 on PR-6b, 0.0098 on PR-6c). The host's own offset on C2's decision rows
was at most 0.0067 in absolute value on any validation seed (0.0042 at k = 8). The choice
read the validation seeds alone, and K4's gate seeds are left to K4's own run at the chosen
k. The pre-tag stage 'decoy-k' (``decoy_k``) re-derives the table on the run's own world:
``decoy_k_leaks`` at every candidate, the choice, and whether every per-seed value (paired
and unpaired) reproduces ``DECOY_K_VALIDATION`` and ``DECOY_K_VALIDATION_UNPAIRED`` to
``DECOY_K_REPRODUCE_TOL`` = 1e-6. It passes when the choice is ``states.KNN_K`` (not a
blocker) and every value reproduces, so the record's table is the tagged world's. Its 48
units (candidate k, world, validation seed; ``decoy_k_unit``) are independent: with
``workers`` > 1 they run in a worker pool (``workers.run``), each unit's world rebuilt at its
k from this process' ``at_decoy_k`` arrays, and the record is assembled in the serial order
from the same ``pr6_world`` calls, so it is the serial run's.

**C2's random-only null** (``c2_null``; descriptive, the pre-tag stage 'c2-null'). C2's
primary statistic of networks that hold nothing beyond a random one: each of
``C2_NULL_SEEDS`` (20 random-init actors of G1's architecture, seeds 20-39, disjoint from C2's
control pool) is scored as the network exactly as the probes stage scores a trained one
(``analysis.c2_network``: its own m in the base [z, m], the pool of random-init actors 0-19
as the controls with the two-level CI, ``probes`` module docstring 'Two-level CI', partition
0's by-fold lambda_res at f_D's k, C2's decision stratum with key weights renormalised on
it, the decision mask and the frozen power floor, and each decision ruled by the
single-geometry robustness rule, ``probes.LOGO_RULE``), on z under G1's obs_rms. The spread
of the 20 statistics is the between-network variance; the two-level CI carries the controls'
share of it. Each seed's geometry-only CI and the decision under it, the full set's decision
and the leave-one-geometry-out sets are recorded beside it (descriptive), and the summary
counts the full sets' decisions, the seeds the rule turns inconclusive and the geometries
that change them ('logo'). The summary (``c2_null_frozen``) is copied into the record and printed beside
each network's primary C2 (``analysis.c2_null_note``); no decision reads it. v2.5's null,
before the pool (seeds 5-24 against the 5 controls 0-4, the geometry-only CI: mean -0.0192,
SD 0.0282, 7 of 20 refuting, none confirming), is the constant ``C2_NULL_FIVE_CONTROLS``,
kept for the design's comparison.

**K5** (``k5``): the permuted-target wiring check (``probes.wiring_null``: while each outer
fold is held out the training targets are permuted state by state across all training rows,
within turn direction for the mirror-odd components and across both for s and vx; the
out-of-fold predictions of ``probes.WIRING_DRAWS`` = 10 draws averaged and scored by R2_wd
against the unpermuted held-out lambda; every K_P component needs a CI lower end at or below
0 and a point at or below ``probes.WIRING_TOL`` = ``K4_ZERO_TOL``) and the control task
(``probes.control_task``; descriptive). **K6** (``k6``): coverage per component and reason.

Seeds: K4's planted noise uses ``K4_STREAM`` (310, ``costate.STREAMS``' 'costate_k4';
``k4_rng`` sub-streams: 0 the bootstrap counts, (1, i) and (2, i) PR-1a, 3 PR-1b, 4 and 5
PR-3, 6 PR-7, 7 PR-5, 8 PR-6's units, 9 and 10 PR-8, 11 PR-9, 12 K5's counts, 13 PR-6b's
features, (14, seed) PR-6b and PR-6c's noise, (15, j) the units holding F, j = 0 PR-6b, 1
PR-6c); the validation seeds of f_D's k draw on ``costate_decoy_k`` (311, sub-stream v);
C2's random-only null uses the random-init actors of seeds 20-39 (``probes.random_actor``, as
C2's control pool 0-19 does) and the pool's resampling on ``costate_c2_pool`` (312,
``probes.pool_draws``); the single-geometry rule's sets (c2-null's and PR-5's) resample on
``costate_logo`` (313, ``probes.logo_counts``, sub-stream the geometry); K2b draws nothing
random.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Sequence

import numpy as np

from apexzero.costate import analysis as AN
from apexzero.costate import keys_root, stimuli_dir, stream_rng
from apexzero.costate import probes as PB
from apexzero.costate import states as ST
from apexzero.costate import targets as TG
from apexzero.costate.targets import K_P, STATE_NAMES
from apexzero.fields import gates as FG
from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields import measures as FMS
from apexzero.fields import perturb as PT
from apexzero.interp import io
from apexzero.interp.constants import ROOT_SEED, STREAM_OFFSET

K1_KINK_SHARE = 0.30          # Section 7 K1: a component with more kinks is descriptive
K1_N_PI = 10                  # pi on 10 anchored keys
K2_TOL = 0.1                  # p90 of |delta lambda| / SD_T
K2_BASIN_DX = 0.5             # m / m/s: a W = 800 plan this far from the W = 400 one on T
K2B_MARGIN = 20.0             # m beyond the furthest preview point of T's states
K2B_RAMP_LEN = 10.0           # m, the second corner's curvature ramps
K2B_SEVERITIES = {"fast": (150.0, 60.0), "slow": (30.0, 90.0)}    # (R m, theta deg)
K2B_RAMP = tuple(np.round(np.arange(1, 9) / 8.0, 6))   # amplitude continuation, 8 steps
K2B_RECENTRE = 6              # re-centring rounds per amplitude at most
K2B_MAX_ITER = 1000           # IPOPT iterations per K2b re-solve (Gate 1's solver stops at 100)
K2B_TOL = 0.1
K4_STREAM = 310               # costate.STREAMS['costate_k4']
K4_SNR = (0.25, 1.0, 4.0)
K4_TOL = 0.03
K4_ZERO_TOL = 0.01
K4_READOUT_TOL = 0.02
K4_SCALE_TOL = 1e-6
K4_N_CONTROLS = 3             # K4's random controls: 3 of C2's 5 (PR-6b and PR-6c use C2's 5)
K4_PAIRED = ("paired increment: Delta2(host + plant) - Delta2(host), the same host, random "
             "controls, rows, components and bootstrap draws")   # K4's no-gain plants
K4_PR6_UNITS = 32             # PR-6: decoy units
K4_PR6_BIAS = 1.0             # PR-6: the units' biases b ~ N(0, bias^2)
K4_PR6_GAINS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # PR-6: the gain grid
K4_CALIBRATION_TOL = 0.02     # PR-6: 'calibrated' (descriptive): |decoy R2 - kNN R2| <= this
K4_PR6_SEEDS = (0, 1, 2)      # PR-6b, PR-6c: the noise draws (sub-stream (14, seed))
K4_PR6_RHO = 0.9              # PR-6b, PR-6c: the noise's AR(1) coefficient along each key
K4_PR6_NOISE = 0.5            # y = std(F) + 0.5 eps: eps is 20% of Var y
K4_PR6B_FEATURES = 256        # PR-6b: F a ridge of lambda on this many biased tanh features
K4_PR6B_ALPHA = 1e-2          # PR-6b: that ridge's penalty (weights normalised to sum 1)
K4_HOLD_UNITS = 32            # PR-6b, PR-6c: the units holding F
K4_HOLD_SD = 0.5              # their mixes and biases ~ N(0, 0.5^2)
K4_OVERSMOOTH_K = 300         # PR-6b, PR-6c's planted fault: an over-smoothed decoy kNN
K4_PR6C_K = 15                # PR-6c's F: the k of its leave-one-geometry-out kNN (the world)
DECOY_K_CANDIDATES = (3, 5, 8, 15)   # f_D's k: the candidates (module docstring 'f_D's k')
DECOY_K_SEEDS = (0, 1, 2, 3, 4, 5)   # its validation seeds (stream costate_decoy_k)
DECOY_K_WORLDS = ("smooth", "knn")   # PR-6b's and PR-6c's worlds (``pr6_world``'s kinds)
DECOY_K_REPRODUCE_TOL = 1e-6  # the decoy-k stage: each value against the table (rounded 1e-6)
# C2's random-only null (module docstring): the random-init actors scored as the network,
# disjoint from C2's control pool (seeds 0 to probes.N_POOL - 1)
C2_NULL_SEEDS = tuple(range(PB.N_POOL, PB.N_POOL + 20))
# v2.5's null, before the pool (descriptive, for the design's comparison): seeds 5-24 against
# the 5 controls 0-4 with the geometry-only CI, the fourth full-size run (e14865d; G1's
# obs_rms, C2's 3,951 decision-stratum rows, the floor {s, xi}, f_D at k = 8)
C2_NULL_FIVE_CONTROLS = {
    "seeds": tuple(range(5, 25)), "controls": tuple(range(5)), "ci": "geometry",
    "n": 20, "mean": -0.019198, "sd": 0.028184, "min": -0.090123, "max": 0.031410,
    "p05": -0.063637, "p50": -0.011245, "p95": 0.016757,
    "decisions": {"refute": 7, "inconclusive": 13, "confirm": 0},
    "ci_above_0": 0, "ci_below_0": 4}
# the choice's record: C2's primary Delta2, the units' paired increment over the host alone,
# per candidate k on the validation seeds (PR-6b 'smooth', PR-6c 'knn'; decoy_k_leaks on the
# full-size T, G1's obs_rms, random-init actors, C2's decision rows and the power floor at
# each k; rounded to 1e-6); states.KNN_K = decoy_k_choice(DECOY_K_VALIDATION)['k']
DECOY_K_VALIDATION = {
    3: {"smooth": (0.002829, -0.000504, 0.003479, 0.002554, 0.003095, -0.000619),
        "knn": (0.008749, -0.002521, 0.002347, 0.012951, 0.004918, 0.000282)},
    5: {"smooth": (0.004708, 0.001989, 0.007383, 0.002811, -0.005263, -0.000827),
        "knn": (0.009730, -0.000436, -0.001349, 0.016534, 0.010810, 0.014429)},
    8: {"smooth": (0.000766, -0.000072, 0.001025, 0.003486, -0.000186, -0.005066),
        "knn": (0.007463, 0.000735, -0.002008, 0.007959, 0.008540, 0.009811)},
    15: {"smooth": (0.004982, 0.011340, 0.002244, 0.001530, 0.004287, 0.006061),
         "knn": (-0.011556, -0.014612, -0.004766, 0.011414, -0.018638, -0.021502)}}
# the same fits' unpaired Delta2 (descriptive; the first choice's table, which it reproduces);
# the host's own offset per seed is DECOY_K_VALIDATION_UNPAIRED - DECOY_K_VALIDATION
DECOY_K_VALIDATION_UNPAIRED = {
    3: {"smooth": (0.004384, -0.002206, 0.006054, 0.003679, 0.002618, 0.000524),
        "knn": (0.008771, -0.002498, 0.002644, 0.013044, 0.004397, 0.000703)},
    5: {"smooth": (0.006590, 0.003167, 0.010495, 0.002655, -0.003946, 0.000123),
        "knn": (0.009084, -0.000433, -0.001259, 0.016210, 0.010921, 0.014187)},
    8: {"smooth": (0.004369, -0.000184, 0.005274, 0.003672, -0.001049, -0.005061),
        "knn": (0.007570, 0.000744, -0.001750, 0.008204, 0.007878, 0.009593)},
    15: {"smooth": (0.005319, 0.011415, 0.008983, -0.003051, 0.003993, 0.002600),
         "knn": (-0.010864, -0.016044, -0.003177, 0.008887, -0.016700, -0.024079)}}
SD_KP_PHASES = ST.PHASES


def k4_rng(*ids: int) -> np.random.Generator:
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + K4_STREAM, *map(int, ids)])


# ------------------------------------------------------------------------- K0


def k0_key(ref: K.KeyRef, bank=None) -> dict:
    """K0's record of one key (module docstring), with its T rows' route gap."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    try:
        kt = TG.key_targets(ref, bank=bank)
    except TG.RepolishFailed as e:
        return {"key_id": ref.key_id, "kind": ref.kind, "ok": False, "failed": str(e)}
    prob = kt.problem
    lm = bank.stimuli[ref.stim_id].landmarks
    rows = ST.t_steps(kt.jac.x, lm, kt.jac.W) if ref.kind != "anchored" else np.array([1])
    rec = TG.k0_record(prob, kt.jac, kt.cheap, kt.prices, kt.blocks, rows)
    stored = TG.costate(TG.with_multipliers(prob, np.asarray(kt.repolish.problem.rec.lam_g),
                                            np.asarray(kt.repolish.problem.rec.lam_x)),
                        "jac", blocks=kt.blocks)
    rec["repolish_change"] = TG.targets_change(stored, kt.jac)
    rec.update(key_id=ref.key_id, kind=ref.kind, W=int(ref.W),
               repolish_iterations=int(kt.repolish.iterations))
    return rec


def _number(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def k0_summary(rows: list[dict], scope: set[str] | None = None) -> dict:
    """K0's pass (every key in ``scope`` 'ok', and on every key every planted action-row bug
    fails) and the maxima of each identity over the keys (every numeric field any row carries;
    a row's None, e.g. no straight road ahead, is skipped). ``scope`` is the keys whose targets
    a pre-registered computation reads (``stage_k0``: the A2 keys and K1's sample; design
    Section 18); None is every key. A key outside it that is not 'ok' is listed in
    'not_ok_descriptive' and decides nothing."""
    ok = [r for r in rows if "failed" not in r]
    num = sorted({k for r in ok for k, v in r.items() if _number(v)}
                 - {"W", "repolish_iterations"})
    mx = {}
    for k in num:
        vals = [abs(float(r[k])) for r in ok if _number(r.get(k))]
        if vals:
            mx[k] = float(np.nanmax(vals))
    out = {"n": len(rows), "failed": [r["key_id"] for r in rows if "failed" in r],
           "not_ok": [r["key_id"] for r in rows if not r.get("ok")], "max": mx,
           "action_plants_min": {p: float(min(r["action_plants"][p] for r in ok
                                              if "action_plants" in r))
                                 for p in TG.ACTION_PLANTS} if ok else {},
           "repolish_change_max": {k: float(max(r["repolish_change"][k] for r in ok))
                                   for k in ("lam", "pi", "switch")} if ok else {}}
    out["plants_fail"] = bool(ok and all(r.get("plants_fail") for r in ok))
    deciding = [k for k in out["not_ok"] if scope is None or k in scope]
    out["not_ok_descriptive"] = [k for k in out["not_ok"] if k not in deciding]
    out["scope"] = ({"keys": len(rows), "rule": "every key"} if scope is None else
                    {"keys": sum(r["key_id"] in scope for r in rows),
                     "rule": "the keys a pre-registered computation reads: the A2 keys and K1's "
                             "sample (design Section 18)"})
    out["pass"] = bool(rows and not deciding and out["plants_fail"])
    return out


def k0(refs: Sequence[K.KeyRef], bank=None, progress: Callable | None = None,
       scope: set[str] | None = None) -> dict:
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    rows = []
    for r in refs:
        rows.append(k0_key(r, bank))
        if progress:
            progress(r.key_id)
    return {"rows": rows, "summary": k0_summary(rows, scope)}


# ------------------------------------------------------------------------- K1


def k1_sample(strata: dict[str, int] = FG.GATE1_STRATA, rng: np.random.Generator | None = None,
              root=None) -> list[K.KeyRef]:
    """K1's keys (module docstring): step 2's Gate 1 rule on idea 2's key roots."""
    rng = FMS.stream_rng("gate1_sample") if rng is None else rng
    pools: dict[str, list[K.KeyRef]] = {k: [] for k in strata}
    for r in ST.a2_refs(root):
        if ST.strata(r).w_is_cell and FG.family(r) in pools:
            pools[FG.family(r)].append(r)
    for refs in ST.n_refs(root).values():
        for r in refs:
            if FG.family(r) in pools:
                pools[FG.family(r)].append(r)
    out = []
    for name, n in strata.items():
        pool = sorted(pools[name], key=lambda r: r.key_id)
        if len(pool) < n:
            raise ValueError(f"stratum {name}: {len(pool)} keys < {n}")
        pick = rng.choice(len(pool), size=n, replace=False)
        out.extend(pool[i] for i in sorted(pick))
    return out


def _middle(idx: np.ndarray) -> int | None:
    idx = np.asarray(idx, int)
    return int(idx[len(idx) // 2]) if len(idx) else None


def k1_states(cs: TG.Costate, lm, anchored: bool) -> dict[str, int]:
    """{label: t} of K1's checked states on one key (module docstring)."""
    out: dict[str, int] = {"decision": 1 if anchored else 0}
    try:
        te = ST.t_end(cs.x, lm)
    except ValueError:
        te = cs.W - 1
    t_rng = np.arange(1, min(te, cs.W - 1) + 1)
    if not anchored:
        t = ST.t_steps(cs.x, lm, cs.W)
        t = t[t >= 1]
        ph = ST.phases(cs.x[t, 0], lm, ST.brake_onset(cs, lm))
        for p in ("approach", "braking", "corner"):
            m = _middle(t[ph == p])
            if m is not None:
                out[p] = m
    for label, mask in (("contact_corridor", cs.own_corridor.any(1)),
                        ("contact_slip", cs.own_slip.any(1)),
                        ("pedal_saturated", cs.pedal_saturated)):
        m = _middle(t_rng[mask[t_rng]])
        if m is not None and m not in out.values():
            out[label] = m
    return out


def _plant_lam(prob, cs: TG.Costate, plant: str, t: int) -> np.ndarray:
    """A planted target at state t: ``targets.costate``'s plants, and 'j_w' (J_W read as
    J_tail on an anchored decision: gamma lambda_1 + e_s/10)."""
    if plant == "j_w":
        v = float(prob.spec.gamma) * cs.lam[t].copy()
        v[TG.S] += 1.0 / float(prob.spec.progress_scale)
        return v
    return TG.costate(prob, "jac", plant=plant, fold=cs.fold).lam[t]


def k1_key(ref: K.KeyRef, bank=None, components: Sequence[int] = range(8),
           states: dict[str, int] | None = None, kt: TG.KeyTargets | None = None) -> list[dict]:
    """One row per (state, component) of one key: target, Richardson FD and its one-sided
    estimates (Section 7's literal kink rule, ``literal_kink``), error (None for a zero
    comparison), kink and the planted targets' errors."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    prob, cs = kt.problem, kt.jac
    lm = bank.stimuli[ref.stim_id].landmarks
    anchored = ref.kind == "anchored"
    states = k1_states(cs, lm, anchored) if states is None else states
    plants = list(TG.PLANTS) + (["j_w"] if anchored else [])
    planted = {}
    for p in plants:
        try:
            planted[p] = {t: _plant_lam(prob, cs, p, t) for t in set(states.values())}
        except ValueError:
            continue
    solver = KK.fd_solver(prob)
    rows = []
    for label, t in states.items():
        contact = bool(cs.contact[t])
        for i in components:
            side = "below" if (i == TG.PEDAL and cs.pedal_saturated[t]) else "central"
            fd = TG.fd_costate_k1(prob, int(t), int(i), side, solver=solver)
            target = float(cs.lam[t, i])
            row = {"key_id": ref.key_id, "kind": ref.kind, "family": FG.family(ref),
                   "state": label, "t": int(t), "component": STATE_NAMES[i],
                   "contact": contact, "target": target, "fd": fd.value, "side": side,
                   "saturated": side == "below",
                   "fold_active": bool(t < cs.W and abs(cs.fold.c[t]) > TG.ACTIVE_TOL),
                   "h": fd.h, "discrepancy": fd.discrepancy, "noise": fd.noise, "ok": fd.ok,
                   "forward": fd.forward, "backward": fd.backward,
                   "kink": fd.kink, "unresolved": unresolved(fd),
                   "ladder": [list(v) for v in fd.ladder],
                   "iterations": fd.iterations, "err": k1_error(target, fd),
                   "err_design": k1_design_error(target, fd),
                   "plants": {p: k1_error(float(v[t][i]), fd) for p, v in planted.items()
                              if p != "j_w" or label == "decision"}}
            rows.append(row)
    return rows


def unresolved(fd: TG.FD) -> bool:
    """The Richardson pair still disagrees after K1's smaller c: beyond ``SHRINK_RTOL`` of
    the estimate and beyond the difference's floor (``FD.noise``, ``k1_error``). Such a
    comparison does not count (as a kink does not), and is reported with its share."""
    return bool(fd.discrepancy > max(TG.SHRINK_RTOL * abs(fd.value), fd.noise))


def k1_error(target: float, fd: TG.FD) -> float | None:
    """``targets.fd_error`` with the zero rule and the denominator at the difference's floor
    eps_t = max(ZERO_ATOL, ``FD.noise``) = max(1e-6, 6 eps_J / (h gamma^t)); None for a zero
    comparison (|target| and |FD| both below eps_t), which ``k1_decision`` counts and never
    pools (Gate 1's zero-field rule).

    eps_J = ``targets.KINK_J_NOISE`` = 1e-11 is J's typical spread between warm tol-1e-10
    re-solves, not a bound (on the pedal-saturated ladder, the rungs' own spread). eps_t is
    twice the largest error that J errors of size eps_J at the four stencil points can give
    the central value (3 eps_J / (h gamma^t)); under independent scatter of that size the
    error's p95 is about 3.7 eps_J / (h gamma^t). At small |lambda| the warm re-solves' error
    is a smooth slope bias, not round-off: they settle along nearly flat directions by an
    amount proportional to c (at delta's worst comparison, key cb10d183 at t = 70, about
    5e-12 in J(c) - J(-c) at c = h, about the same slope from c = 1e-7 to 1e-4 and at
    re-solve tolerances 1e-8 to 1e-10). At delta's step that is within eps_J at the stencil,
    so eps_t bounds it; at the 1e-4 steps it would not (lambda_n there fails either way)."""
    floor = max(TG.ZERO_ATOL, float(fd.noise))
    if abs(target) < floor and abs(fd.value) < floor:
        return None
    return abs(target - fd.value) / max(abs(fd.value), 3.0 * fd.discrepancy, floor)


def k1_design_error(target: float, fd: TG.FD) -> float | None:
    """Section 7's error to the letter ('design_rule'): ``targets.fd_error`` with Gate 1's
    ZERO_ATOL; None for a zero comparison (counted, never pooled)."""
    if abs(target) < TG.ZERO_ATOL and abs(fd.value) < TG.ZERO_ATOL:
        return None
    return TG.fd_error(target, fd)


def literal_kink(r: dict) -> bool | None:
    """Section 7's kink rule to the letter on a K1 row: the one-sided estimates disagree by
    more than ``targets.KINK_FACTOR`` x the Richardson discrepancy (``FD.kink`` without its
    noise term; a one-sided row has no kink test). None when the row carries no one-sided
    estimates (a row written before they were stored)."""
    if r.get("side", "central") != "central":
        return False
    f, b = r.get("forward"), r.get("backward")
    if f is None or b is None or not np.isfinite(f) or not np.isfinite(b):
        return None
    return bool(abs(f - b) > TG.KINK_FACTOR * r["discrepancy"])


def k1_pi(ref: K.KeyRef, bank=None, kt: TG.KeyTargets | None = None) -> list[dict]:
    """pi_1 on one anchored key, against ``targets.fd_pi`` and its planted bugs ('r0_prev'
    with the key's anchor['a_prev'], the naming hazard of Section 2), scored as the lambda
    components are (``k1_error``; its floor is fd_pi's ``FD.noise``): 'err' is None for a
    zero comparison (|pi| and |FD| both below the floor), and so is a plant's entry when
    |pi| and the planted FD both are. 'err_design' is Section 7's error to the letter
    (``k1_design_error``), and 'plant_fd' the planted estimates."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    prob, cs = kt.problem, kt.jac
    a_prev = np.asarray(prob.rec.anchor.get("a_prev", np.zeros(2)), float)
    solver = KK.fd_solver(prob)
    rows = []
    for j in range(2):
        fd = TG.fd_pi(prob, j, solver=solver)
        target = float(cs.pi[1, j])
        pl, pfd = {}, {}
        for p in TG.PI_PLANTS:
            fp = TG.fd_pi(prob, j, plant=p, a_prev=a_prev if p == "r0_prev" else None,
                          solver=solver)
            pl[p], pfd[p] = k1_error(target, fp), fp.value
        rows.append({"key_id": ref.key_id, "j": j, "target": target, "fd": fd.value,
                     "ok": fd.ok, "h": fd.h, "discrepancy": fd.discrepancy,
                     "noise": fd.noise, "err": k1_error(target, fd),
                     "err_design": k1_design_error(target, fd), "plants": pl,
                     "plant_fd": pfd})
    return rows


def _pass_rule(errs: Sequence) -> dict:
    """K1's median and p95 rule on the non-zero errors; zero comparisons (None) are counted
    ('n_zero') and never pooled (Gate 1's zero-field rule, fields/gates.py:345-357). With no
    non-zero error nothing passes (a decided component needs one: ``k1_decision``)."""
    errs = list(errs)
    e = np.array([v for v in errs if v is not None], float)
    n_zero = len(errs) - len(e)
    if not len(e):
        return {"n": 0, "n_zero": n_zero, "median": None, "p95": None, "pass": False}
    med, p95 = float(np.median(e)), float(np.quantile(e, 0.95))
    return {"n": len(e), "n_zero": n_zero, "median": med, "p95": p95,
            "pass": bool(med < TG.K1_MEDIAN and p95 < TG.K1_P95)}


def _reading(good: list[dict], kinks: list) -> dict | None:
    """One component under a reported reading (``k1_decision``): Section 7's error to the
    letter ('err_design') over the converged rows that are not kinks by ``kinks`` (unresolved
    ones scored), the kink share, and descriptive above ``K1_KINK_SHARE`` or with no non-zero
    comparison. None when a row's kink cannot be read (``literal_kink``)."""
    if any(k is None for k in kinks):
        return None
    rec = _pass_rule([r.get("err_design", r["err"]) for r, k in zip(good, kinks) if not k])
    share = sum(kinks) / max(len(good), 1)
    rec.update(kink_share=share, descriptive=bool(share > K1_KINK_SHARE or not rec["n"]))
    return rec


def _plant_rows(p: str, c: str, r: dict) -> bool:
    """Where a planted bug is scored: 'pre_jump' on contact states, 'unfolded' on the pedal
    rows whose fold is active (c_t != 0: pedal-saturated states and those upstream)."""
    if p == "pre_jump":
        return bool(r["contact"])
    if p == "unfolded":
        return c == "pedal" and bool(r.get("fold_active"))
    return True


def required_plants(rows: list[dict]) -> list[str]:
    """Every K1 plant that must be scored and fail: ``targets.PLANTS``, and 'j_w' wherever an
    anchored decision is checked."""
    out = list(TG.PLANTS)
    if any(r.get("kind") == "anchored" and r.get("state") == "decision" for r in rows):
        out.append("j_w")
    return out


def k1_decision(rows: list[dict], pi_rows: list[dict] | None = None,
                require_pi: bool | None = None) -> dict:
    """K1's verdict (module docstring). Zero comparisons (None errors) are counted per
    component, per plant and per reading ('n_zero') and never pooled; a component with no
    non-zero comparison that counts is descriptive. The plants cannot pass vacuously: each
    required one (``required_plants``) must be present and fail on at least one decided
    component with non-zero counted rows; pi is required (``require_pi``, default: whenever
    an anchored key is in the rows), under the same zero rule: it passes only on a non-zero
    comparison, and each of its plants must be scored on a non-zero one and fail
    ('design_rule' beside it, binding nothing). The share of the
    pedal-saturated pedal rows that count is reported (``saturated_counted_share``); such
    rows left unresolved count toward pedal's 30% share like any other. The two reported
    readings ('design_rule', 'design_rule_literal_kink'; None for rows without one-sided
    estimates) give each component's numbers and, at the top, their own decided components
    and verdict; they bind nothing."""
    comp = {}
    plants: dict[str, dict] = {}
    names = sorted({p for r in rows for p in r["plants"]})
    for c in STATE_NAMES:
        mine = [r for r in rows if r["component"] == c]
        if not mine:
            continue
        good = [r for r in mine if r["ok"]]
        kinks = [r for r in good if r["kink"]]
        unres = [r for r in good if not r["kink"] and r.get("unresolved")]
        counted = [r for r in good if not r["kink"] and not r.get("unresolved")]
        share = (len(kinks) + len(unres)) / max(len(good), 1)
        rec = _pass_rule([r["err"] for r in counted])
        rec.update(kink_share=len(kinks) / max(len(good), 1),
                   unresolved_share=len(unres) / max(len(good), 1),
                   n_failed_solves=len(mine) - len(good),
                   descriptive=bool(share > K1_KINK_SHARE or not rec["n"]),
                   design_rule=_reading(good, [bool(r["kink"]) for r in good]),
                   design_rule_literal_kink=_reading(good, [literal_kink(r) for r in good]))
        comp[c] = rec
        if c == "pedal":
            sat = [r for r in good if r.get("saturated")]
            rec["saturated_counted_share"] = (
                sum(1 for r in sat if not r["kink"] and not r.get("unresolved"))
                / len(sat) if sat else None)
            rec["n_saturated"] = len(sat)
        comp[c] = rec
        for p in names:
            sel = [r for r in counted if p in r["plants"] and _plant_rows(p, c, r)]
            plants.setdefault(p, {})[c] = _pass_rule([r["plants"][p] for r in sel])
    decided = [c for c, v in comp.items() if not v["descriptive"]]
    out = {"components": comp, "pass": bool(decided and all(comp[c]["pass"] for c in decided)),
           "descriptive": [c for c in comp if c not in decided],
           "plants": {p: {"fails": bool(any(not v["pass"] for c, v in d.items()
                                            if c in decided and v["n"])),
                          "scored": bool(any(v["n"] for c, v in d.items() if c in decided)),
                          "n_zero": sum(v["n_zero"] for v in d.values()),
                          "components": d} for p, d in plants.items()}}
    for name in ("design_rule", "design_rule_literal_kink"):
        recs = {c: v[name] for c, v in comp.items()}
        if not recs or any(v is None for v in recs.values()):
            out[name] = None
            continue
        dec = [c for c, v in recs.items() if not v["descriptive"]]
        out[name] = {"decided": dec, "descriptive": [c for c in recs if c not in dec],
                     "pass": bool(dec and all(recs[c]["pass"] for c in dec))}
    need = required_plants(rows)
    out["plants_missing"] = [p for p in need if not out["plants"].get(p, {}).get("scored")]
    out["plants_fail"] = bool(not out["plants_missing"]
                              and all(out["plants"][p]["fails"] for p in need)
                              and all(v["fails"] for v in out["plants"].values() if v["scored"]))
    if require_pi is None:
        require_pi = any(r.get("kind") == "anchored" for r in rows)
    good_pi = [r for r in (pi_rows or []) if r["ok"]]
    if good_pi:
        pi = _pass_rule([r["err"] for r in good_pi])
        if not pi["n"]:
            pi["reason"] = "no non-zero pi comparison"
        pp = {p: _pass_rule([r["plants"][p] for r in good_pi]) for p in TG.PI_PLANTS}
        pp = {p: {**v, "fails": bool(v["n"] and not v["pass"])} for p, v in pp.items()}
        out["pi"] = {**pi, "n_failed_solves": len(pi_rows) - len(good_pi),
                     "design_rule": _pass_rule([r.get("err_design", r["err"])
                                                for r in good_pi]),
                     "plants": pp}
        out["pass"] = bool(out["pass"] and pi["pass"])
        out["plants_fail"] = bool(out["plants_fail"] and all(v["fails"] for v in pp.values()))
    elif require_pi:
        out["pi"] = {"n": 0, "pass": False, "reason": "no converged pi row"}
        out["pass"] = False
        out["plants_fail"] = False
    out["pass_with_plants"] = bool(out["pass"] and out["plants_fail"])
    return out


# ------------------------------------------------------------------------- K2


def twin_pairs(root=None) -> list[tuple[K.KeyRef, K.KeyRef]]:
    """The tier V horizon twins: (W = 400 key, W = 800 key) of one cell, kind and phase."""
    refs = ST.a2_refs(root)
    twins = [r for r in refs if not ST.strata(r).w_is_cell]
    out = []
    for tw in twins:
        mate = [r for r in refs if r.cell_id == tw.cell_id and r.kind == tw.kind
                and r.phase == tw.phase and r.W != tw.W]
        for m in mate:
            a, b = (m, tw) if m.W < tw.W else (tw, m)
            out.append((a, b))
    return sorted(set(out), key=lambda p: p[0].key_id)


def lam_sd(tab: io.Table, names: Sequence[str] = K_P, scheme: str = "key") -> np.ndarray:
    """SD_T(lambda_k) under the metric's weights (within direction for mirror-odd ones)."""
    w = ST.weights(tab, scheme)
    d = np.asarray(tab["direction"])
    return np.array([np.sqrt(ST.wvar(np.asarray(tab[f"lam_{c}"], float), w,
                                     d if c in TG.MIRROR_ODD else None)) for c in names])


def k2_compare(cs_a: TG.Costate, cs_b: TG.Costate, sd: np.ndarray, t: np.ndarray | None = None,
               names: Sequence[str] = K_P) -> np.ndarray:
    """[len(t), m] |lambda^a_t - lambda^b_t| / SD_T at steps t (all t < min W by default)."""
    idx = [STATE_NAMES.index(c) for c in names]
    t = np.arange(min(cs_a.W, cs_b.W)) if t is None else np.asarray(t, int)
    return np.abs(cs_a.lam[t][:, idx] - cs_b.lam[t][:, idx]) / np.asarray(sd)[None]


def k2_horizon(d: np.ndarray, tol: float = K2_TOL) -> dict:
    """t_H from [pairs, t, m] (module docstring): the largest t such that the p90 over
    pairs is <= tol for every component at every t' <= t."""
    p90 = np.quantile(np.asarray(d, float), 0.9, axis=0)              # [t, m]
    good = np.all(p90 <= tol, axis=1)
    bad = np.flatnonzero(~good)
    t_H = int(bad[0] - 1) if len(bad) else int(len(good) - 1)
    return {"t_H": t_H, "p90_max": p90.max(0).tolist(),
            "first_divergence": None if not len(bad) else int(bad[0])}


def k2_retention(t_H: int, tab: io.Table) -> dict:
    """Whether K2's cut at t_H drops any of T's states, from T's uncut ``t_end`` column (the
    design's Fallback criterion: a W = 400 plan that has not reached s_c1 + 50 m by t_H).
    T_max_t_by_W: the largest t_end on W = 400 keys, and t_end - 400 on W = 800 keys (the cut
    is on remaining horizon); fallback_keys: the W = 400 keys with t_end > t_H (Section 7
    'Fallback': re-solved at W = 800); cut_w800: W = 800 keys with t_end - 400 > t_H."""
    W = np.asarray(tab["W"], int)
    te = np.asarray(tab["t_end"], int)
    key = np.asarray(tab["key_id"])
    by_W = {}
    for w_, off in ((400, 0), (800, 400)):
        m = W == w_
        if m.any():
            by_W[w_] = int(te[m].max()) - off
    fb = sorted(set(key[(W == 400) & (te > int(t_H))].tolist()))
    c8 = sorted(set(key[(W == 800) & (te - 400 > int(t_H))].tolist()))
    drops = bool(any(int(t_H) < v for v in by_W.values()))
    return {"t_H": int(t_H), "T_max_t_by_W": {str(k): v for k, v in by_W.items()},
            "drops_T": drops, "fallback_keys": fb, "cut_w800": c8,
            "pass": bool(not drops and not fb and not c8)}


def k2_twins(pairs: Sequence[tuple[K.KeyRef, K.KeyRef]], sd: np.ndarray, bank=None,
             tab: io.Table | None = None) -> dict:
    """K2 on the stored twins: per pair the T-step and whole-window discrepancies, t_H and
    whether the cut drops any of T's states (``k2_retention`` on T's table ``tab``; without
    it, only against the twins' own T range, reported as ``twins_T_max_t``)."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    per, d_all, t_max = [], [], 0
    for a, b in pairs:
        ka, kb = TG.key_targets(a, bank=bank), TG.key_targets(b, bank=bank)
        lm = bank.stimuli[a.stim_id].landmarks
        tT = ST.t_steps(ka.jac.x, lm, ka.jac.W)
        t_max = max(t_max, int(tT.max()))
        dT = k2_compare(ka.jac, kb.jac, sd, tT)
        d = k2_compare(ka.jac, kb.jac, sd)
        d_all.append(d)
        per.append({"w400": a.key_id, "w800": b.key_id, "stim": a.stim_name,
                    "T_max": dT.max(0).tolist(), "T_p90": np.quantile(dT, 0.9, 0).tolist(),
                    "lam_vx_t1": [float(ka.jac.lam[1, TG.VX]), float(kb.jac.lam[1, TG.VX])]})
    hor = k2_horizon(np.stack(d_all))
    out = {"pairs": per, "horizon": hor, "twins_T_max_t": t_max, "names": list(K_P)}
    if tab is not None:
        ret = k2_retention(hor["t_H"], tab)
        out.update(retention=ret, T_max_t=max(ret["T_max_t_by_W"].values()),
                   drops_T=ret["drops_T"], fallback_keys=ret["fallback_keys"],
                   **{"pass": ret["pass"]})
    else:
        out.update(T_max_t=t_max, drops_T=bool(hor["t_H"] < t_max), fallback_keys=None,
                   **{"pass": bool(hor["t_H"] >= t_max)})
    return out


def _key_record(rec, x, a, lam_g, lam_x, W: int, suffix: str):
    spec = dict(rec.spec, steps=int(W))
    return dataclasses.replace(rec, key_id=f"{rec.key_id}{suffix}", W=int(W), spec=spec,
                               x=np.asarray(x, float), a=np.asarray(a, float),
                               lam_g=np.asarray(lam_g, float), lam_x=np.asarray(lam_x, float),
                               active_steps=None)


def k2_extend(ref: K.KeyRef, bank=None, W: int = 800) -> dict:
    """The insurance solve (module docstring) of one W = 400 A2 key."""
    from apexzero.interp import answer_key as ak
    from apexzero.ocp.window import WindowSpec, solve_continuation

    if bank is None:
        bank, _ = K.design(stimuli_dir())
    rec = K.load_key(ref, root=keys_root())
    track = bank.track(rec.stim_id)
    ak._guard(bank.stimuli[rec.stim_id])
    p = K.key_car(rec)
    ctx = ak.key_ctx(p)
    spec = dataclasses.replace(WindowSpec(**rec.spec), steps=int(W))
    solver = ak.make_solver(int(W), spec=spec)
    res = solve_continuation(lambda _T: solver, track, np.asarray(rec.x0, float), [int(W)],
                             None, lambda s, n: ak.extend_straight(s, n, track, p), ctx=ctx,
                             p=p, previous=_as_result(rec))[0]
    out = {"key_id": ref.key_id, "W": int(W), "status": res.status, "success": bool(res.success),
           "iterations": int(res.iterations)}
    if not res.success:
        return out
    lam_g, lam_x = res.meta["lam"]
    rec8 = _key_record(rec, res.x, res.a, lam_g, lam_x, W, f"-w{W}")
    rp = K.repolish(K.rebuild(rec8, bank))
    cs4 = TG.key_targets(ref, bank=bank).jac
    cs8 = TG.costate(rp.problem, "jac")
    lm = bank.stimuli[ref.stim_id].landmarks
    tT = ST.t_steps(cs4.x, lm, cs4.W)
    dx = float(np.max(np.abs(cs4.x[tT] - cs8.x[tT])))
    out.update(repolish_ok=bool(rp.ok), basin_dx=dx, different_basin=bool(dx > K2_BASIN_DX),
               cs400=cs4, cs800=cs8, t=tT)
    return out


def _as_result(rec):
    """A stored key as the ``previous`` result of a continuation (its optimum and
    multipliers)."""
    from apexzero.ocp.window import WindowResult

    return WindowResult(x=np.asarray(rec.x, float), a=np.asarray(rec.a, float),
                        value=float(rec.J), success=True, status="stored", iterations=0,
                        solve_seconds=0.0, rounds=0,
                        meta={"lam": (np.asarray(rec.lam_g, float), np.asarray(rec.lam_x, float))})


# ------------------------------------------------------------------------- K2b


def second_corner(n: int, s2: float, R: float, theta_deg: float, sigma: int,
                  ramp: float = K2B_RAMP_LEN, ds: float = PT.DS) -> PT.Samples:
    """A curvature trapezoid (ramps ``ramp`` m, total turn theta) starting at s2, of
    direction sigma, as a sample perturbation (widths unchanged)."""
    s = PT.sample_s(n, ds)
    arc = max(np.deg2rad(theta_deg) * R - ramp, 0.0)
    u = s - s2
    L = 2.0 * ramp + arc
    k = np.clip(np.minimum(u / ramp, (L - u) / ramp), 0.0, 1.0) * int(sigma) / R
    return PT.Samples(k, np.zeros(n), np.zeros(n))


def _recentre(prob: K.KeyProblem, w: np.ndarray, lam_g: np.ndarray, lam_x: np.ndarray,
              track) -> K.KeyProblem:
    """The problem with its window centres on the plan in w (``window._window_centres``),
    its P and bounds rebuilt there (keys.py:394-454)."""
    from apexzero.ocp.window import _window_centres

    T = prob.T
    x = w[: 8 * (T + 1)].reshape(T + 1, 8)
    sigma = _window_centres(x[:, 0])
    P = K.key_parameters(prob.solver, track, sigma, prob.ctx, prob.p)
    lbx, ubx, lbg, ubg = K.bounds(prob.rec, prob.spec, sigma, prob.p)
    return dataclasses.replace(prob, track=track, sigma=sigma, P=P, lbx=lbx, ubx=ubx, lbg=lbg,
                               ubg=ubg, w=np.asarray(w, float), lam_g=np.asarray(lam_g, float),
                               lam_x=np.asarray(lam_x, float))


def _window_binds(prob: K.KeyProblem, lam_x: np.ndarray, tol: float = 1e-8) -> bool:
    s_rows = 8 * np.arange(1, prob.T + 1)
    return bool(np.any(np.abs(lam_x[s_rows]) > tol))


_K2B_SOLVERS: dict = {}


def k2b_solver(prob: K.KeyProblem, max_iter: int = K2B_MAX_ITER):
    """Gate 1's re-solver (``kkt.fd_solver``: the key solver's functions and options,
    REFINE_IPOPT's warm start, tol 1e-10) with ``max_iter`` iterations."""
    import casadi as ca

    key = (id(prob.solver), int(max_iter))
    hit = _K2B_SOLVERS.get(key)
    if hit is None or hit[0] is not prob.solver:
        opts = dict(prob.solver._opts)
        opts.update({"ipopt." + k: v for k, v in KK.REFINE_IPOPT.items()})
        opts["ipopt.tol"] = float(KK.GATE1_TOL)
        opts["ipopt.max_iter"] = int(max_iter)
        hit = (prob.solver, ca.nlpsol("to_ocp_k2b", "ipopt", prob.solver.funcs["nlp"], opts))
        _K2B_SOLVERS[key] = hit
    return hit[1]


def k2b_solve(prob: K.KeyProblem, delta: PT.Samples, ramp=K2B_RAMP,
              rounds: int = K2B_RECENTRE) -> tuple[K.KeyProblem | None, dict]:
    """The key re-solved on the variant strip by amplitude continuation with re-centring
    (module docstring): at each amplitude, re-centre the windows on the current plan and
    re-solve (``k2b_solver``) until a solve succeeds with no window bound binding, at most
    ``rounds`` times (a failed solve's iterate is re-centred and continued); returns (the
    problem at the optimum or None, log)."""
    track0 = prob.track
    cur = prob
    solver = k2b_solver(prob)
    log = []
    for amp in ramp:
        track = PT.perturbed_track(track0, KK.scaled(delta, amp))
        done = False
        for r in range(int(rounds)):
            cur = _recentre(cur, cur.w, cur.lam_g, cur.lam_x, track)
            res = KK.resolve_from(cur, None, cur.w, cur.lam_g, cur.lam_x, solver)
            log.append({"amp": float(amp), "round": r, "status": res.status,
                        "iterations": res.iterations})
            cur = dataclasses.replace(cur, w=res.w, lam_g=res.lam_g, lam_x=res.lam_x)
            if res.ok and not _window_binds(cur, res.lam_x):
                done = True
                break
        if not done:
            return None, {"log": log, "ok": False}
    return cur, {"log": log, "ok": True}


def k2b_key(ref: K.KeyRef, sd: np.ndarray, bank=None, severities=K2B_SEVERITIES,
            kt: TG.KeyTargets | None = None) -> dict:
    """K2b on one A2 key (module docstring): per severity, |Delta lambda_k| / SD_T on T's
    rows and its p90 per phase."""
    if bank is None:
        bank, _ = K.design(stimuli_dir())
    kt = TG.key_targets(ref, bank=bank) if kt is None else kt
    prob, cs = kt.problem, kt.jac
    lm = bank.stimuli[ref.stim_id].landmarks
    t = ST.t_steps(cs.x, lm, cs.W)
    from apexzero.env import racing

    reach = float(np.max(cs.x[t, 0] + np.asarray(racing.preview_offsets(
        np.maximum(cs.x[t, 3], 0.0)[:, None]))[:, -1]))
    s2 = reach + K2B_MARGIN
    n = len(np.asarray(prob.track.kappa))
    sigma = int(ST.strata(ref).direction)
    ph = ST.phases(cs.x[t, 0], lm, ST.brake_onset(cs, lm))
    out = {"key_id": ref.key_id, "s2": s2, "reach": reach, "severities": {}}
    for name, (R, th) in severities.items():
        delta = second_corner(n, s2, R, th, sigma)
        new, log = k2b_solve(prob, delta)
        if new is None:
            out["severities"][name] = {"ok": False, **log}
            continue
        cs2 = TG.costate(new, "jac")
        d = k2_compare(cs, cs2, sd, t)
        out["severities"][name] = {
            "ok": True, "max": d.max(0).tolist(),
            "by_phase": {p: np.quantile(d[ph == p], 0.9, axis=0).tolist()
                         for p in ST.PHASES if np.any(ph == p)},
            "rows": d.tolist(), "phase": ph.tolist(),
            "n_solves": len(log["log"])}
    return out


def k2b_rule(keys: list[dict], tol: float = K2B_TOL) -> dict:
    """K2b's exclusions (Section 7 K2b 'Rule'): per severity and phase stratum, the p90 of
    |Delta lambda_k| / SD_T pooled over every sampled key's rows in that phase; a component
    whose p90 exceeds tol under either severity (the worse one) is excluded from decisions in
    that phase. Keys without per-row records fall back to their own p90 (reported)."""
    out: dict[str, list[str]] = {}
    p90: dict[str, dict[str, list[float]]] = {}
    fallback = False
    sev = sorted({n for k in keys for n in k["severities"]})
    for p in ST.PHASES:
        worst = np.zeros(len(K_P))
        for name in sev:
            rows = []
            for k in keys:
                v = k["severities"].get(name, {})
                if not v.get("ok"):
                    continue
                if "rows" in v:
                    ph = np.asarray(v["phase"])
                    if np.any(ph == p):
                        rows.append(np.asarray(v["rows"], float)[ph == p])
                elif p in v.get("by_phase", {}):
                    fallback = True
                    rows.append(np.asarray(v["by_phase"][p], float)[None])
            if rows:
                q = np.quantile(np.vstack(rows), 0.9, axis=0)
                p90.setdefault(p, {})[name] = q.tolist()
                worst = np.maximum(worst, q)
        out[p] = [c for c, x in zip(K_P, worst, strict=True) if x > tol]
    return {"excluded": out, "p90": p90, "n_keys": len(keys), "pooled_rows": not fallback}


# ------------------------------------------------------------------------- K3


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _le(v, tol) -> bool:
    return v is not None and np.isfinite(float(v)) and float(v) <= tol


def _gt(v, tol) -> bool:
    """A finite value above tol (a planted wrong pullback that the check catches)."""
    return bool(np.isfinite(_num(v)) and _num(v) > tol)


def k3_judge(plants: dict | None, random: dict | None, extractor: list[dict] | None,
             min_states: int | None = None) -> dict:
    """K3's pass from the dry-run reports (module docstring). The random report must cover
    ``critic.K3_MIN_STATES`` T states (J_z and the pullbacks' finite differences), its
    composition autodiff must match, s must be flagged on every exit-straight row and on no
    row before s_c1 or at the apex, with the same flags on float32-upcast key arrays. The
    design's 'xi's and vx's statistics with and without the s column agree to 1e-6' and
    'transitional rows unflagged' are reported, not judged (both fail as the code stands:
    the projection ignores the s column's scale, and rows 7.8-9.4 m past s_c1 already sit
    at a ratio of 2e-5 to 5e-5; the user settles the design text)."""
    from apexzero.costate import critic as CR

    min_states = CR.K3_MIN_STATES if min_states is None else int(min_states)
    checks: dict[str, bool] = {}
    if plants is not None:
        checks["env_wiring"] = _le(plants["env_accel_err"], 1e-9) and _le(
            plants["env_step_err"], 1e-9)
        checks["channel_maps"] = all(_le(plants[k], CR.LINEAR_TOL) for k in (
            "J_acc_err", "J_acc_other", "J_prev_err", "J_prev_other", "J_accel_rows"))
        checks["frenet_leak"] = _le(plants["frenet_leak"], 1e-12)
        maps = plants["maps"]
        checks["manifold_maps_fd"] = _le(maps["max_err"], CR.FD_TOL) and _le(maps["newton"],
                                                                            1e-10)
        lin = plants["linear"]
        # every closed form and zero the plants report: per channel g, g_held, pi (0) and
        # accel (0); the planted heads; the composition's autodiff of g_held, pi_held and
        # g_xprev
        lin_vals = [v for k, d in lin.items() if k not in ("fd", "accel_prev")
                    for kk, v in d.items() if kk != "n_heads"]
        ap = lin["accel_prev"]
        lin_vals += [ap[k] for k in ("accel_sens", "pi_held", "g", "pi", "g_held", "g_reset")]
        checks["linear_closed_forms"] = all(_le(v, CR.LINEAR_TOL) for v in lin_vals)
        # the planted critics' own forms against their finite differences under the same
        # per-form rule as the random critics' (Section 13: held and x_{t-1}-held 1e-6, the
        # on-manifold form, whose reference passes through Newton over clipped actions,
        # 1e-5), every form present
        checks["linear_fd"] = set(lin.get("fd") or {}) == set(CR.VARIANTS) and all(
            _le(v["max_err"], CR.FD_TOLS[f]) for f, v in lin["fd"].items())
        pl = plants["plants"]
        checks["state_plant"] = all(_le(pl["state"][k], CR.PLANT_TOL) for k in ("g", "pi"))
        checks["accel_normal_primary"] = all(_le(v["g"], CR.PLANT_TOL) and _le(v["pi"],
                                                                                CR.PLANT_TOL)
                                             for v in pl.values() if "accel_sens" in v)
        live_c = [k for k in pl if k.startswith("c=") and float(k[2:]) != 0.0]
        # K3's pass rule (design v2.2 Section 13): every planted wrong pullback exceeds
        # critic.WRONG_MIN = 1e-4, the wrong closed forms of the accel-reading plant ...
        checks["accel_normal_held_fails"] = bool(live_c) and all(
            _gt(pl[k].get(f), CR.WRONG_MIN) for k in live_c for f in CR.WRONG_CLOSED_FORMS)
        # ... its own finite differences matching each form (the reference the wrong ones
        # are scored on), per form as ``pullback_fd`` and ``linear_fd`` ...
        checks["plant_fd"] = bool(live_c) and all(
            _le((pl[k].get(f"fd_{v}") or {}).get("max_err"), CR.FD_TOLS[v])
            for k in live_c for v in CR.VARIANTS)
        # ... and every AD form scored on every other form's finite differences, on the
        # accel-normal plants and the random linear critic (none may be missing)
        checks["wrong_pullbacks_fail"] = bool(live_c) and not k3_wrong_missing(plants) and all(
            _gt((v or {}).get("max"), CR.WRONG_MIN) for d in plants["wrong"].values()
            for v in d.values())
    if random is not None:
        checks["random_defect"] = _le(random["defect_body"], 1e-9)
        checks["apply_gap"] = all(_le(g[k], PB.EXTRACTOR_TOL) for g in random["apply_gap"]
                                  for k in ("v", "heads", "actor"))
        checks["heads_count"] = all(g["n_heads"] == g["flax_heads"] for g in random["apply_gap"])
        sg = random["stop_gradient"]
        checks["stop_gradient"] = sg["flax"] == 0.0 and sg["extractor"] > 0.0
        checks["jz_fd"] = _le(random["jz"]["max_err"], CR.FD_TOL)
        # Section 13: the held and x_{t-1}-held forms below FD_TOL = 1e-6, the on-manifold
        # form below FD_TOL_MANIFOLD = 1e-5 (main V and heads), every form present
        checks["pullback_fd"] = set(random["fd"]) == set(CR.VARIANTS) and all(
            _le(v["main"], CR.FD_TOLS[f]) and _le(v["heads"], CR.FD_TOLS[f])
            for f, v in random["fd"].items())
        comp = random.get("composed") or []
        checks["composed_random"] = bool(comp) and all(_le(v, 1e-12) for c in comp
                                                       for v in c.values())
        n_jz = random["jz"].get("n_states", random["jz"].get("n", 0) // 8)
        n_fd = min((v.get("n_states", 0) for v in random["fd"].values()), default=0)
        checks["k3_sample"] = bool(n_jz >= min_states and n_fd >= min_states)
        ob = random["observability"]
        checks["observability"] = bool(ob["n_exit"] and ob["exit_flagged"] == ob["n_exit"]
                                       and ob["before_flagged"] == 0
                                       and ob.get("apex_flagged", 0) == 0)
        checks["observability_f32"] = bool(ob.get("f32_same_flags", False))
    if extractor is not None:
        checks["extractor"] = all(bool(e.get("pass")) for e in extractor)
    out = {"checks": checks, "pass": bool(checks and all(checks.values())),
           "tolerances": {"fd": dict(CR.FD_TOLS), "jz": CR.FD_TOL, "wrong_min": CR.WRONG_MIN}}
    if random is not None:
        out["pullback_fd"] = {f: {"main": v.get("main"), "heads": v.get("heads"),
                                  "tol": CR.FD_TOLS.get(f)} for f, v in random["fd"].items()}
    if plants is not None and plants.get("wrong"):
        errs = [(p, k, _num((v or {}).get("max"))) for p, d in plants["wrong"].items()
                for k, v in d.items()]
        p, k, e = min(errs, key=lambda r: r[2] if np.isfinite(r[2]) else -np.inf)
        out["planted_wrong"] = {"min": e, "argmin": f"{p}:{k}", "n": len(errs),
                                "pass_rule": f"each > {CR.WRONG_MIN:g}",
                                "missing": k3_wrong_missing(plants)}
    return out


def k3_wrong_missing(plants: dict) -> list[str]:
    """The planted wrong pullbacks K3's rule needs that the plants report lacks: every
    ordered pair of forms (``critic.WRONG_PAIRS``) on the random linear critic and on every
    accel-normal plant with c > 0, and the latter's wrong closed forms."""
    from apexzero.costate import critic as CR

    wrong = plants.get("wrong") or {}
    need = {"linear": CR.WRONG_PAIRS}
    for k in plants.get("plants", {}):
        if k.startswith("c=") and float(k[2:]) != 0.0:
            need[k] = CR.WRONG_PAIRS + tuple(f"closed:{f}" for f in CR.WRONG_CLOSED_FORMS)
    return [f"{p}:{k}" for p, ks in need.items() for k in ks
            if k not in (wrong.get(p) or {})
            or not np.isfinite(_num((wrong[p][k] or {}).get("max")))]


# ------------------------------------------------------------------------- K3 after the tag

K3_TRAINED_CHECKS = ("extractor", "heads", "float32")


def expected_heads(name: str) -> int | None:
    """Section 7 K3: every generalist has exactly ``probes.N_HEADS_GENERALIST`` = 5 heads,
    S1-S3 none; None for a network outside both (its head check fails)."""
    if name in AN.GENERALISTS:
        return PB.N_HEADS_GENERALIST
    if name in AN.SPECIALISTS:
        return 0
    return None


def k3_trained(agent, ob: np.ndarray, heads: int | None) -> dict:
    """K3's checks on one trained network after the tag (Section 7) at raw observations
    ob [n, 222] (the critic stage passes D's), in a float64 process: the extractor's
    self-consistency in float64 by both extractors (``critic.net_apply_gap``: the critic's V
    and heads and the actor's mean; ``probes.extractor_gap``: the same through the probe
    extractor), each network's head count against ``heads`` (``expected_heads``), and the
    float64 V against the float32 one (``critic.float32_gap``), judged by
    ``k3_trained_judge``. The trained marks refuse it before the tag, and that refusal is
    never contained; after it each of the three measurements is (``analysis.guarded``: one
    that raises fails its checks with the error as the reason). The setup is not: a non-MLP
    config, or a tree whose head count is not the config's, is ``critic.trained_critic``'s
    refusal and stops the stage, as the network's ``critic_of`` would right after."""
    from apexzero.costate import critic as CR

    t0 = time.perf_counter()
    net = CR.trained_critic(agent)
    actor, critic = PB.agent_nets(agent)
    CR._refuse_trained(net, "K3's trained checks")
    PB._refuse_trained(actor, "K3's trained checks")
    o, _ = PB.normalise(ob, agent.obs_rms)
    gap = AN.guarded(CR.net_apply_gap, net, agent.cfg, o, actor.params)
    ext = AN.guarded(PB.extractor_gap, actor, critic, o)
    f32 = AN.guarded(CR.float32_gap, net, agent.cfg, ob)
    out = k3_trained_judge(gap, ext, f32, heads, int(agent.cfg.ensemble_heads))
    out.update(critic=gap, extractor=ext, float32=f32, n_states=len(ob),
               seconds=time.perf_counter() - t0)
    return out


def k3_trained_judge(gap: dict, ext: dict, f32: dict, heads: int | None,
                     cfg_heads: int) -> dict:
    """K3's after-the-tag pass (``k3_trained``'s measurements): 'extractor' every error of
    both extractors at most ``probes.EXTRACTOR_TOL`` = 1e-10, computed in float64; 'heads'
    the config's, the tree's (the config's by construction in ``k3_trained``:
    ``critic.trained_critic`` refuses a mismatch), Flax's and the probe extractor's head
    counts all equal to ``heads``; 'float32' the float32 run computed in float32 and its
    largest |V64 - V32| at most ``critic.F32_TOL`` = 1e-4 (the critic's units). A
    measurement that raised fails its checks. 'decision' holds 'pass', the failed checks
    and, on a fail, the reason 'K3 trained check failed: <check> (<why>); ...'."""
    from apexzero.costate import critic as CR

    tol = PB.EXTRACTOR_TOL
    why: dict[str, list[str]] = {}
    pb_keys = ("actor_mean", "critic_v") + (("critic_heads",) if cfg_heads else ())
    for src, d, keys, dts in (("critic", gap, ("v", "heads", "actor"), ("dtype", "actor_dtype")),
                              ("probe extractor", ext, pb_keys, ("dtype",))):
        if "error" in d:
            why.setdefault("extractor", []).append(f"{src}: {d['error']}")
            continue
        why.setdefault("extractor", []).extend(
            f"{src} {k} {_num(d.get(k)):.3g} > {tol:g}" for k in keys if not _le(_num(d.get(k)),
                                                                                 tol))
        why["extractor"].extend(f"{src} {k} {d.get(k)}" for k in dts if d.get(k) != "float64")
    counts = {"config": cfg_heads, "tree": gap.get("n_heads"), "Flax": gap.get("flax_heads"),
              "probe extractor": ext.get("heads")}
    if heads is None:
        why["heads"] = ["no head rule for this network"]
    elif any(v != heads for v in counts.values()):
        why["heads"] = [f"expected {heads}: " + ", ".join(f"{k} {v}" for k, v in counts.items())]
    if "error" in f32:
        why["float32"] = [f32["error"]]
    elif f32.get("dtype32") != "float32":
        why["float32"] = [f"the float32 run computed in {f32.get('dtype32')}"]
    elif not _le(_num(f32.get("max_abs")), CR.F32_TOL):
        why["float32"] = [f"max |V64 - V32| {_num(f32.get('max_abs')):.3g} > {CR.F32_TOL:g}"]
    failed = [k for k in K3_TRAINED_CHECKS if why.get(k)]
    reason = ("K3 trained check failed: " + "; ".join(f"{k} ({', '.join(why[k])})"
                                                      for k in failed)) if failed else None
    return {"checks": {k: k not in failed for k in K3_TRAINED_CHECKS},
            "decision": {"pass": not failed, "failed": failed, "reason": reason},
            "expected_heads": heads, "tolerances": {"extractor": tol, "float32": CR.F32_TOL}}


# ------------------------------------------------------------------------- K3b


def k3b(cells: Sequence[tuple[str, int]], cfg, obs_rms, steps: int | None = None, bank=None,
        hybrid_seed: int = 0) -> dict:
    """K3b on (policy, A2 cell) pairs, policy 'linear' (step 2's linear driver), 'accel'
    (the linear driver reading the accel channels, ``own.accel_driver``: the on-manifold
    check's plant, its held gradients must fail) or 'hybrid' (the small-gain random-init
    hybrid on G1's architecture and obs_rms), closed loops of ``steps`` (default
    ``own.K3B_STEPS`` = 160): a float64 process."""
    from apexzero.costate import own as OW
    from apexzero.fields import network as NW

    steps = OW.K3B_STEPS if steps is None else int(steps)
    if bank is None:
        bank, cells_ = K.design(stimuli_dir())
    else:
        _, cells_ = K.design(stimuli_dir())
    starts = []
    for kind, cid in cells:
        st = NW.cell_start(cells_[int(cid)], 0, bank)
        rd = NW.road(bank, st.stim_id)
        if kind == "linear":
            net = NW.linear_net(*NW.linear_driver(st.sigma), name=f"linear-{cid}")
        elif kind == "accel":
            net = NW.linear_net(*OW.accel_driver(st.sigma), name=f"accel-{cid}")
        elif kind == "hybrid":
            net = NW.hybrid_net(st.sigma, cfg, obs_rms, hybrid_seed)
        else:
            raise ValueError(f"unknown K3b policy {kind!r}")
        starts.append(OW.k3b_start(net, rd.base, NW.loop_inputs(st, steps), steps,
                                   name=f"{kind}@{cid}"))
    return {"starts": [s.record() for s in starts], "decision": OW.k3b_decision(starts)}


# ------------------------------------------------------------------------- K4


@dataclasses.dataclass
class ProbeWorld:
    """T's rows prepared for the probe pipeline: z (clipped), the folds, the weights, the
    direction, winsorised K_P targets, lambda_res, random-init actors' (h3, m) and the
    decoys; for C2's primary statistic (``c2_primary``) its decision stratum, decision mask
    and frozen power floor (None: every row, every row scored, every component)."""

    tab: io.Table
    z: np.ndarray
    fo: PB.Folds
    w: np.ndarray
    direction: np.ndarray
    lam: np.ndarray            # [n, 4] winsorised K_P targets
    lam_res: np.ndarray        # [F, n, 4]
    h3: list[np.ndarray]       # random-init actors' h3 (seed 0 is the 'network')
    m: list[np.ndarray]
    names: tuple = tuple(f"lam_{c}" for c in K_P)
    rows_c2: np.ndarray | None = None      # [n] C2's decision stratum (T's interior)
    mask: np.ndarray | None = None         # [n, 4] states.decision_mask
    power_floor: dict | None = None        # the frozen power floor's verdicts per component


def probe_world(tab: io.Table, z: np.ndarray, cfg,
                seeds: Sequence[int] = range(1 + PB.N_RANDOM_ACTORS), partition: int = 0,
                mask: np.ndarray | None = None, power_floor: dict | None = None) -> ProbeWorld:
    """The probe pipeline's inputs on T's rows (random-init actors of ``cfg``; seed 0 is the
    plants' host, the others the random controls: K4 uses 3 of C2's 5, ``K4_N_CONTROLS``,
    PR-6b and PR-6c all 5). C2's decision stratum is T's interior where the table has it;
    ``mask`` and ``power_floor`` are C2's decision mask and the frozen floor."""
    w = ST.weights(tab, "key")
    fo = PB.folds(tab, partition)
    lam = np.stack([np.asarray(tab[f"lam_{c}"], float) for c in K_P], 1)
    lam = np.stack([ST.winsorise(lam[:, k], ST.winsor_bounds(lam[:, k], w))
                    for k in range(lam.shape[1])], 1)
    res = ST.cross_fit(ST.decoy_d1(tab), lam, fo.outer, w)
    o = np.clip(z, -10.0, 10.0)
    h3, m = [], []
    for s in seeds:
        hid = PB.hidden(PB.random_actor(cfg, s), o)
        h3.append(hid["h3"])
        m.append(hid["m"])
    rows = np.asarray(tab["interior"], bool) if "interior" in tab else None
    return ProbeWorld(tab, o, fo, w, np.asarray(tab["direction"]), lam, res.by_fold, h3, m,
                      rows_c2=rows, mask=None if mask is None else np.asarray(mask, bool),
                      power_floor=None if power_floor is None else dict(power_floor))


def _std(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (y - ST.wmean(y, w)) / np.sqrt(ST.wvar(y, w))


def _offset(h: dict | None) -> dict | None:
    """The host's own offset (descriptive; module docstring 'K4's no-gain plants') from its
    unpaired gain record (``probes.Gain.record``): its Delta2 against the random controls,
    with the CI and per component, and every random network's own gain over the base
    ('networks': the host's, then each control's, the mean over the statistic's components)
    with their SD, the between-random-network spread C2's statistic carries."""
    if not h or h.get("stat") is None:
        return None
    p = np.asarray(h["passing"], bool)
    base = np.asarray(h["r2_base"], float)
    nets = [float(np.mean(np.asarray(h["raw"], float)[p]))]
    nets += [float(np.mean((np.asarray(r, float) - base)[p])) for r in h["r2_random"]]
    return {"delta2": h["stat"], "ci": h["ci"], "gain": h["gain"], "networks": nets,
            "sd_networks": float(np.std(nets, ddof=1)) if len(nets) > 1 else None}


def k4(pw: ProbeWorld, counts: np.ndarray | None = None, which: Sequence[str] | None = None,
       readout_net=None) -> dict:
    """K4's plants (module docstring) on one probe world; ``which`` limits them."""
    which = set(which or ("PR-1a", "PR-1b", "PR-2", "PR-3", "PR-4", "PR-5", "PR-6", "PR-7",
                          "PR-8", "PR-9"))
    out: dict = {}
    w, fo, z, n = pw.w, pw.fo, pw.z, len(pw.w)
    h0, rand = pw.h3[0], pw.h3[1:1 + K4_N_CONTROLS]
    m0 = pw.m[0]
    G = int(fo.cluster.max()) + 1
    counts = PB.cluster_counts(G, rng=k4_rng(0)) if counts is None else counts
    zm = np.hstack([z, m0])

    def c2(h, Y=None, names=None, passing=None, rand_=None, mm=None):
        return PB.c2(h, rand if rand_ is None else rand_, z, m0 if mm is None else mm,
                     pw.lam_res if Y is None else Y, fo, w, pw.direction,
                     pw.names if names is None else names, passing, None, counts)

    def pair(h, Y=None, folds=None, cnt=None):
        # a no-gain plant's paired increment over the host h0 alone (module docstring)
        return PB.c2_paired(h, h0, rand, z, m0, pw.lam_res if Y is None else Y,
                            fo if folds is None else folds, w, pw.direction, pw.names, None,
                            None, counts if cnt is None else cnt)

    if "PR-1a" in which:
        rows = []
        for i, snr in enumerate(K4_SNR):
            T = k4_rng(1, i).standard_normal((n, 1))
            H = PB.plant_units(h0, T, snr, w, k4_rng(2, i))
            g = PB.c2(H, rand, z, m0, T, fo, w, None, ("target",), None, None, counts)
            exp = PB.expected_r2(snr)
            rows.append({"snr": snr, "expected": exp, "r2_full": float(g.r2_full[0]),
                         "delta2": g.stat, "partial": float(g.partial[0]),
                         "pass": bool(abs(g.partial[0] - exp) <= K4_TOL
                                      and abs(g.stat - exp * (1 - g.r2_base[0])) <= K4_TOL)})
        out["PR-1a"] = {"rows": rows, "pass": all(r["pass"] for r in rows)}
    if "PR-1b" in which or "PR-6" in which:
        snr = 4.0
        # one target array for the plant's baseline, the probe and the random controls: each
        # row's own-fold lambda_res (Residuals.oof). C2's by-fold training targets change
        # with the held-out fold, which one planted column cannot follow (module docstring)
        y1 = np.asarray(pw.lam_res)[fo.outer, np.arange(n)]
        base = PB.ridge_cv(np.hstack([h0, zm]), y1, fo, w, None, pw.names)
        r = base.y - base.pred                       # residualised out of fold on the baseline
        H = PB.plant_units(h0, r, snr, w, k4_rng(3))
        g = c2(H, Y=y1)
        exp = PB.expected_r2(snr)
        r2b = PB.probe_r2(base, w, pw.direction)
        rec = {"snr": snr, "expected": exp, "partial_vs_own": [], "delta2": g.gain.tolist(),
               "r2_b": r2b.tolist(), "target": "each row's own-fold lambda_res",
               "planted": "all K_P components at once", "pass": True}
        for k in range(len(pw.names)):
            pr = (g.r2_full[k] - r2b[k]) / (1.0 - r2b[k])
            rec["partial_vs_own"].append(float(pr))
            ok = abs(pr - exp) <= K4_TOL and abs(g.gain[k] - (1 - r2b[k]) * exp) <= K4_TOL
            rec["pass"] = bool(rec["pass"] and ok)
        rec["by_fold"] = pr1b_by_fold(pw, counts, rand, snr)
        out["PR-1b"] = rec
    if "PR-2" in which:
        a_t = np.stack([np.asarray(pw.tab["a_steer"], float),
                        np.asarray(pw.tab["a_pedal"], float)], 1)
        p1 = pair(np.hstack([h0, m0]))
        # the a*_t variant: a*_t is the base's m and is appended to every layer
        p2 = PB.c2_paired(np.hstack([h0, a_t]), h0, [np.hstack([r_, a_t]) for r_ in rand], z,
                          a_t, pw.lam_res, fo, w, pw.direction, pw.names, None, None, counts)
        out["PR-2"] = {"delta2_m": p1.stat, "delta2_astar": p2.stat,
                       "m": _stat(p1.record()), "astar": _stat(p2.record()),
                       "statistic": K4_PAIRED,
                       "pass": bool(abs(p1.stat) <= K4_ZERO_TOL and abs(p2.stat) <= K4_ZERO_TOL)}
    if "PR-3" in which:
        geo = ST.geometry_index(pw.tab)
        pair_id = np.char.add(geo.astype(str), "/" + np.asarray(pw.tab["pair_id"]).astype(str))
        codes = np.hstack([PB.one_hot(geo), PB.one_hot(pair_id)])
        H = np.hstack([h0, codes])
        p = pair(H)
        g = p.plant
        mis = PB.grouped_folds(np.asarray(pw.tab["key_id"]), k4_rng(4), cluster=geo)
        oof = np.asarray(pw.lam_res)[fo.outer, np.arange(n)]      # each row's own residual
        pm = pair(H, oof, mis, PB.cluster_counts(int(mis.cluster.max()) + 1, rng=k4_rng(5)))
        gm = pm.plant
        # z identifies the geometry as well as the codes do, so under the split by key the
        # leak shows in the base [z, m] (held-out keys of a seen geometry) as much as in the
        # codes' gain: either counts as the leak shown (unpaired, its rule unchanged; the
        # paired leak is reported beside it)
        rise = float(np.mean(gm.r2_base[g.passing]) - np.mean(g.r2_base[g.passing]))
        out["PR-3"] = {"grouped": p.stat, "by_key": gm.stat, "base_rise_by_key": rise,
                       "by_key_paired": pm.stat, "grouped_record": _stat(p.record()),
                       "by_key_record": _stat(pm.record()), "statistic": K4_PAIRED,
                       "pass": bool(p.stat <= K4_ZERO_TOL
                                    and max(gm.stat, rise) > K4_ZERO_TOL)}
    if "PR-4" in which and readout_net is not None:
        out["PR-4"] = readout_net
    if "PR-5" in which:
        out["PR-5"] = pr5(pw, counts)
    if "PR-6" in which:
        out["PR-6"] = pr6(pw, counts, out.get("PR-1b"), rand)
        out["PR-6b"] = pr6_world(pw, "smooth")
        out["PR-6c"] = pr6_world(pw, "knn")
    if "PR-7" in which:
        g0 = c2(h0)
        rz = k4_rng(6).uniform(0.2, 5.0, z.shape[1])
        rows = {}
        for lab, H, Z in (("h3x0.1", 0.1 * h0, z), ("h3x10", 10.0 * h0, z),
                          ("z_scaled", h0, z * rz)):
            gg = PB.c2(H, [r_ * (1.0 if lab == "z_scaled" else (0.1 if lab == "h3x0.1" else 10.0))
                           for r_ in rand], Z, m0, pw.lam_res, fo, w, pw.direction, pw.names,
                       None, None, counts)
            rz2 = PB.probe_r2(PB.ridge_cv(Z, pw.lam, fo, w, None, pw.names), w, pw.direction)
            rows[lab] = {"delta2_change": abs(gg.stat - g0.stat), "r2z": rz2.tolist()}
        r2z0 = PB.probe_r2(PB.ridge_cv(z, pw.lam, fo, w, None, pw.names), w, pw.direction)
        ok = all(v["delta2_change"] <= K4_SCALE_TOL
                 and np.max(np.abs(np.asarray(v["r2z"]) - r2z0)) <= K4_SCALE_TOL
                 for v in rows.values())
        out["PR-7"] = {"rows": rows, "r2z": r2z0.tolist(), "pass": bool(ok)}
    if "PR-8" in which:
        out["PR-8"] = pr8(pw, counts)
    if "PR-9" in which:
        out["PR-9"] = pr9(pw, counts)
    out["pass"] = bool(all(v.get("pass", False) for k, v in out.items() if k.startswith("PR")))
    out["host_offset"] = host_offsets(out)
    out["decoy_k"] = ST.KNN_K                    # f_D's k in lambda_res and PR-6b/6c's y
    return out


def host_offsets(rec: dict) -> dict:
    """The host's own offset per no-gain plant and per seed from a K4 record (descriptive;
    module docstring 'K4's no-gain plants'): the host alone against that plant's random
    controls on that plant's rows (PR-2, PR-3, PR-6: T and K4's controls; PR-6b, PR-6c: C2's
    decision rows and C2's 5 controls, per noise seed), each plant's record carrying its CI,
    components and the networks' spread."""
    def d(r):
        o = (r or {}).get("host_offset") if isinstance(r, dict) else None
        return None if not o else o.get("delta2")

    out: dict = {}
    if isinstance(rec.get("PR-2"), dict):
        out["PR-2"] = {v: d(rec["PR-2"].get(v)) for v in ("m", "astar")}
    if isinstance(rec.get("PR-3"), dict):
        out["PR-3"] = {v: d(rec["PR-3"].get(f"{v}_record")) for v in ("grouped", "by_key")}
    if isinstance(rec.get("PR-6"), dict):
        out["PR-6"] = d(rec["PR-6"])
    for p in ("PR-6b", "PR-6c"):
        if isinstance(rec.get(p), dict):
            out[p] = [d(s) for s in rec[p].get("seeds") or []]
    return {**out, "descriptive": True,
            "is": "Delta2 of the random host alone against the plant's random controls on "
                  "the plant's rows: the between-random-network spread C2 carries"}


K4_C1_SNR = (0.25, 0.5, 1.0, 4.0)   # PR-5's SNRs (0.25 and 0.5: where refute bias showed)
K4_FLOOR = (-0.02, 0.005)            # PR-5e: an uninformative gradient's R2_c per component
K4_DIRECTION_FLOOR = 0.02            # PR-5e: a direction-only gradient on xi


def pr5(pw: ProbeWorld, counts: np.ndarray, rows: np.ndarray | None = None) -> dict:
    """PR-5 and 5b-d through C1 (module docstring): g = lambda + noise of SNR against the
    within-direction variance (reward units), expected R2_c = SNR / (1 + SNR) on every
    component (``pr5_expected``: C1 calibrates within direction for the odd components),
    at SNR 0.25, 0.5, 1 and 4 (never refute at SNR >= 1), on C1's decision rows (the
    interior where the table has it). PR-5e, the floor: a zero, an i.i.d. noise and a -lambda
    gradient each score R2_c in ``K4_FLOOR`` per component, and a direction-only gradient
    at most ``K4_DIRECTION_FLOOR`` on xi."""
    w, fo = pw.w, pw.fo
    lam = pw.lam
    odd = np.array([PB.is_odd(n) for n in pw.names])
    if rows is None:
        rows = (np.asarray(pw.tab["interior"], bool) if "interior" in pw.tab
                else np.ones(len(w), bool))
    rows = np.asarray(rows, bool)
    wr = ST.stratum_weights(pw.tab, rows)
    sd = np.array([np.sqrt(ST.wvar(lam[rows, k], wr[rows],
                                   pw.direction[rows] if odd[k] else None))
                   for k in range(lam.shape[1])])
    bounds = np.stack([lam.min(0), lam.max(0)], 1)
    cl, G = fo.cluster, int(fo.cluster.max()) + 1
    out = {"rows": [], "n_rows": int(rows.sum())}
    rng = k4_rng(7)
    noise = rng.standard_normal(lam.shape)

    labels = AN.geometry_labels(pw.tab)

    def run(g, label, exp=None):
        # C1's decision as the analysis records it: ruled by probes.LOGO_RULE
        c = AN.c1(g, lam, fo.outer, wr, cl, G, pw.direction, rows, None, None, K_P, counts,
                  bounds=bounds, logo="rule", labels=labels)
        rec = {"case": label, "r2c": c.r2c.tolist(), "stat": c.stat, "ci": list(c.ci),
               "decision": c.decision, "label": c.label, "slope": c.slope.tolist(),
               "decision_all_geometries": c.decision_all,
               "geometry_sensitive": list(c.sensitive), "logo": c.logo}
        if exp is not None:
            rec["expected"] = exp
        out["rows"].append(rec)
        return c

    ok = True
    for snr in K4_C1_SNR:
        exp = pr5_expected(snr, lam.shape[1])
        c = run(lam + noise * sd / np.sqrt(snr), f"snr={snr:g}", exp.tolist())
        ok &= bool(np.all(np.abs(c.r2c - exp) <= K4_TOL))
        if snr >= 1.0:
            ok &= c.decision != "refute"
    c = run(-lam, "negated")
    ok &= bool(c.ci[0] <= 0.0 and c.decision != "confirm" and not c.label)
    neg_r2c = c.r2c
    flip = lam.copy()
    flip[:, 1] *= -1.0
    c = run(flip, "one_flipped")
    ok &= c.decision != "confirm"
    c = run(10.0 * lam, "times10")
    ok &= bool(c.decision == "confirm" and not c.label)
    lo, hi = K4_FLOOR
    floor = {"negated": neg_r2c}
    floor["zero"] = run(np.zeros_like(lam), "zero").r2c
    floor["noise"] = run(rng.standard_normal(lam.shape), "noise").r2c
    fl_ok = all(bool(np.all((v >= lo) & (v <= hi))) for v in floor.values())
    dirg = np.repeat(np.asarray(pw.direction, float)[:, None], lam.shape[1], 1)
    d_r2c = run(dirg, "direction_only").r2c
    xi = list(pw.names).index("lam_xi") if "lam_xi" in pw.names else 2
    fl_ok &= bool(d_r2c[xi] <= K4_DIRECTION_FLOOR)
    out["PR-5e"] = {"floor": {k: np.asarray(v).tolist() for k, v in floor.items()},
                    "direction_only": d_r2c.tolist(), "pass": bool(fl_ok)}
    out["pass"] = bool(ok and fl_ok)
    return out


def pr5_expected(snr: float, m: int) -> np.ndarray:
    """PR-5's expected R2_c per component of g = lambda + eta at an SNR against the
    within-direction variance: SNR / (1 + SNR) for every component (C1's calibration is
    within direction for the odd components and least squares in its slope)."""
    return np.full(m, PB.expected_r2(snr))


def pr1b_by_fold(pw: ProbeWorld, counts: np.ndarray, rand: Sequence[np.ndarray] | None = None,
                 snr: float = 4.0) -> dict:
    """PR-1b's fold-consistent plant on C2's own by-fold targets (descriptive; module
    docstring): the baseline [h3, z, m] fitted on lambda_res by fold, the unit u of its
    residual r (``plant_units``, PR-1b's noise draw) and, on fold f's training rows,
    u + alpha d^f / sd_r with d^f = lambda_res^(f) - each row's own-fold lambda_res (0 on fold
    f's rows), so the unit follows the training target that changes with f. Fold f's probe
    is fitted with its own block (one ``ridge_cv`` per fold, fold f's predictions kept);
    partial R2 and Delta2 against SNR/(1 + SNR) and (1 - R2_b) SNR/(1 + SNR), and whether
    they lie within ``K4_TOL`` (binding nothing)."""
    w, fo, n = pw.w, pw.fo, len(pw.w)
    h0, zm = pw.h3[0], np.hstack([pw.z, pw.m[0]])
    rand = pw.h3[1:1 + K4_N_CONTROLS] if rand is None else rand
    Y = np.asarray(pw.lam_res)
    oof = Y[fo.outer, np.arange(n)]
    base = PB.ridge_cv(np.hstack([h0, zm]), Y, fo, w, None, pw.names)
    r = base.y - base.pred
    u = PB.plant_units(h0, r, snr, w, k4_rng(3))[:, h0.shape[1]:]
    scale = float(np.median(np.sqrt(ST.wvar(h0, w))))
    sd_r = np.sqrt(ST.wvar(r, w))
    parts = []
    for f in range(fo.n_folds):
        p = PB.ridge_cv(zm, Y, fo, w, np.hstack([h0, u + scale * (Y[f] - oof) / sd_r]),
                        pw.names)
        parts.append(p)
    held = [fo.outer == f for f in range(fo.n_folds)]
    pred = np.full_like(parts[0].pred, np.nan)
    alpha, kept = np.array(parts[0].alpha), np.array(parts[0].kept)
    for f, p in enumerate(parts):
        pred[held[f]] = p.pred[held[f]]
        alpha[f], kept[f] = p.alpha[f], p.kept[f]
    full = dataclasses.replace(parts[0], pred=pred, alpha=alpha, kept=kept)
    pb, randoms = PB.conditional(rand, zm, Y, fo, w, pw.names)
    g = PB.gain(pb, full, randoms, w, fo, pw.direction, None, None, counts)
    exp = PB.expected_r2(snr)
    r2b = PB.probe_r2(base, w, pw.direction)
    partial = (g.r2_full - r2b) / (1.0 - r2b)
    want = (1.0 - r2b) * exp
    return {"partial_vs_own": partial.tolist(), "delta2": g.gain.tolist(),
            "expected_delta2": want.tolist(), "r2_b": r2b.tolist(),
            "within_tol": bool(np.all(np.abs(partial - exp) <= K4_TOL)
                               and np.all(np.abs(g.gain - want) <= K4_TOL)),
            "descriptive": True}


def decoy4(tab: io.Table) -> np.ndarray:
    """[n, 4] PR-6's decoy variables (s - s_apex, vx, sigma, log R)."""
    return np.stack([np.asarray(tab["ds_apex"], float), np.asarray(tab["vx"], float),
                     np.asarray(tab["direction"], float), np.log(np.asarray(tab["R"], float))],
                    1)


def pr6(pw: ProbeWorld, counts: np.ndarray, pr1b: dict | None,
        rand: Sequence[np.ndarray] | None = None) -> dict:
    """PR-6 (module docstring 'K4's PR-1b and PR-6 rules'): ``K4_PR6_UNITS`` biased tanh
    units of random mixes of the decoys (``decoy4``) appended to the host's h3, the gain from
    ``K4_PR6_GAINS`` whose probe-alone R2 (mean over K_P) lies closest to the decoy kNN's;
    C2's gain on the by-fold lambda_res against K4's controls, paired over the host alone
    (``probes.c2_paired``). Pass: the paired Delta2 <= ``K4_ZERO_TOL`` (one-sided) and PR-1b
    passes; 'overfit_cost' is its negative part, 'delta2_unresidualised' the same units on
    lambda itself, 'unpaired' and 'host_offset' the two unpaired gains (descriptive)."""
    w, fo = pw.w, pw.fo
    rand = pw.h3[1:1 + K4_N_CONTROLS] if rand is None else rand
    D = decoy4(pw.tab)
    knn_pred = pw.lam - np.asarray(pw.lam_res)[fo.outer, np.arange(len(w))]
    target = float(np.mean(PB.r2(pw.lam, knn_pred, w, pw.direction, pw.names)))

    def units(gain):
        return PB.decoy_units(D, w, k4_rng(8), K4_PR6_UNITS, gain, K4_PR6_BIAS)

    r2s = [float(np.mean(PB.probe_r2(PB.ridge_cv(None, pw.lam, fo, w, units(gn), pw.names),
                                     w, pw.direction))) for gn in K4_PR6_GAINS]
    i = int(np.argmin(np.abs(np.asarray(r2s) - target)))
    H = np.hstack([pw.h3[0], units(K4_PR6_GAINS[i])])
    p = PB.c2_paired(H, pw.h3[0], rand, pw.z, pw.m[0], pw.lam_res, fo, w, pw.direction,
                     pw.names, None, None, counts)
    p_raw = PB.c2_paired(H, pw.h3[0], rand, pw.z, pw.m[0], pw.lam, fo, w, pw.direction,
                         pw.names, None, None, counts)
    pr1b_ok = bool(pr1b and pr1b.get("pass", False))
    return {"knn_r2": target, "gains": list(K4_PR6_GAINS), "r2_by_gain": r2s,
            "gain": float(K4_PR6_GAINS[i]), "decoy_r2": r2s[i],
            "calibrated": bool(abs(r2s[i] - target) <= K4_CALIBRATION_TOL),
            "statistic": K4_PAIRED,
            "delta2": p.stat, "ci": list(p.ci), "by_component": p.gain.tolist(),
            "overfit_cost": float(max(0.0, -p.stat)),
            "unpaired": {"delta2": p.plant.stat, "ci": list(p.plant.ci),
                         "by_component": p.plant.gain.tolist()},
            "host_offset": _offset(p.host.record()),
            "delta2_unresidualised": p_raw.stat, "ci_unresidualised": list(p_raw.ci),
            "delta2_unresidualised_unpaired": p_raw.plant.stat,
            "pr1b_pass": pr1b_ok, "pass": bool(p.stat <= K4_ZERO_TOL and pr1b_ok)}


def _ar1(tab: io.Table, w: np.ndarray, m: int, rng: np.random.Generator,
         rho: float = K4_PR6_RHO) -> np.ndarray:
    """[n, m] AR(1) series (coefficient ``rho``, unit innovations) along each key's rows in t
    order, independent across keys and columns, weighted-standardised per column."""
    key, t = np.asarray(tab["key_id"]), np.asarray(tab["t"])
    eps = np.zeros((len(key), int(m)))
    c = np.sqrt(1.0 - rho * rho)
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        idx = idx[np.argsort(t[idx], kind="stable")]
        e = rng.standard_normal((len(idx), int(m)))
        for i in range(1, len(idx)):
            e[i] = rho * e[i - 1] + c * e[i]
        eps[idx] = e
    return _std(eps, w)


def pr6b_f(pw: ProbeWorld) -> np.ndarray:
    """[n, 4] PR-6b's smooth decoy function: the weighted ridge (``K4_PR6B_ALPHA``, the
    weights normalised to sum 1, an unpenalised intercept) of lambda on ``K4_PR6B_FEATURES``
    biased tanh features of ``decoy4`` (``probes.decoy_units``, gain 1, sub-stream 13),
    standardised, fitted on every row: lambda only shapes F."""
    w = np.asarray(pw.w, float)
    RF = PB.decoy_units(decoy4(pw.tab), w, k4_rng(13), K4_PR6B_FEATURES, 1.0, 1.0)
    X = PB.scaler(RF, w).apply(RF)
    wn = w / w.sum()
    mu, yb = wn @ X, wn @ pw.lam
    Xc = X - mu
    G = (Xc * wn[:, None]).T @ Xc
    c = (Xc * wn[:, None]).T @ (pw.lam - yb)
    return Xc @ np.linalg.solve(G + K4_PR6B_ALPHA * np.eye(len(G)), c) + yb


def pr6c_f(pw: ProbeWorld) -> np.ndarray:
    """[n, 4] PR-6c's decoy function, matching the decoy kNN (Section 7's 'matches the decoy
    kNN's R2'): each geometry's rows predicted by the ``K4_PR6C_K`` = 15 kNN of lambda on D1
    fitted on every other geometry (the world's own k, whatever f_D's)."""
    D1, geo = ST.decoy_d1(pw.tab), ST.geometry_index(pw.tab)
    F = np.full(pw.lam.shape, np.nan)
    for g in np.unique(geo):
        s = geo == g
        F[s] = ST.knn_fit(D1[~s], pw.lam[~s], pw.w[~s], K4_PR6C_K).predict(D1[s])
    return F


def hold_units(F: np.ndarray, w: np.ndarray, rng: np.random.Generator,
               n_units: int = K4_HOLD_UNITS, sd: float = K4_HOLD_SD) -> np.ndarray:
    """[n, n_units] units holding F: tanh(F~ A + b), F~ F weighted-standardised, A and b
    ~ N(0, sd^2)."""
    Fz = PB.scaler(F, w).apply(F)
    A = rng.normal(0.0, sd, (Fz.shape[1], int(n_units)))
    b = rng.normal(0.0, sd, int(n_units))
    return np.tanh(Fz @ A + b)


def c2_primary(pw: ProbeWorld, H: np.ndarray, Y: np.ndarray, t_wide: bool = False,
               host: np.ndarray | None = None) -> dict:
    """C2's primary statistic of the layer ``H`` (with its host's m) on the targets ``Y``
    ([F, n, 4] or [n, 4]) as the analysis computes it (``analysis.c2_network``): on C2's
    decision stratum (``ProbeWorld.rows_c2``), each component scored on its decision rows
    (``mask``), the mean over the frozen power floor's components (``power_floor``), against
    C2's 5 random-init controls (``h3[1:6]``; K4's, not the decisions' control pool: module
    docstring 'K4's PR-1b and PR-6 rules'), key weights renormalised on the stratum, C2's
    own bootstrap counts. ``t_wide``: every row of T and every component (descriptive).
    ``host``: ``H`` is that host with a plant, and the record is the plant's paired
    increment over the host alone (module docstring 'K4's no-gain plants')."""
    n = len(pw.w)
    rows = np.ones(n, bool) if (t_wide or pw.rows_c2 is None) else np.asarray(pw.rows_c2,
                                                                                 bool)
    wr = ST.stratum_weights(pw.tab, rows)
    try:
        return AN.c2_network(H, pw.h3[1:1 + PB.N_RANDOM_ACTORS], pw.z, pw.m[0], Y, pw.fo,
                             None, pw.direction, K_P, None if t_wide else pw.power_floor,
                             rows, weights=wr[rows], mask=None if t_wide else pw.mask,
                             host=host)
    except PB.NotTestable as e:          # recorded as the analysis does; the plant fails
        return {"decision": "not testable", "reason": str(e)}


def _stat(rec: dict) -> dict:
    """A gain record's statistic; for a paired increment (``probes.Paired.record``) also its
    unpaired value and the host's own offset."""
    out = {"delta2": rec.get("stat"), "ci": rec.get("ci"), "gain": rec.get("gain"),
           "decision": rec.get("decision")}
    if "unpaired" in rec:
        out["unpaired"] = _stat(rec["unpaired"])
        out["host_offset"] = _offset(rec.get("host"))
    return out


def oversmooth_k(fold: np.ndarray) -> int:
    """The over-smoothed fault's k: ``K4_OVERSMOOTH_K``, capped at the smallest training set
    of ``states.cross_fit``'s fits (the rows outside two outer folds), so a T too small for
    k = 300 (the rehearsal's reduced T) still scores the fault, its kNN then near the
    weighted mean of the training rows. On the full T the cap does not bind."""
    cnt = np.sort(np.bincount(np.asarray(fold, int)))
    return int(max(1, min(K4_OVERSMOOTH_K, int(cnt.sum() - cnt[-2:].sum()))))


def pr6_world(pw: ProbeWorld, kind: str, seeds: Sequence[int] = K4_PR6_SEEDS,
              t_wide: bool = True, faults: bool = True, k: int | None = None,
              noise: Callable[[int], np.random.Generator] | None = None) -> dict:
    """PR-6b (``kind`` 'smooth', ``pr6b_f``) or PR-6c ('knn', ``pr6c_f``): the known-truth
    world of the module docstring on each noise seed (``noise(seed)``; K4's own, the default,
    is sub-stream (14, seed); ``decoy_k_rng`` the validation seeds of f_D's k), the units
    holding F on sub-stream (15, j). Per seed: C2's primary Delta2 (``c2_primary``) as the
    units' paired increment over the host alone (module docstring 'K4's no-gain plants'; the
    unpaired value and the host's offset beside it) on y residualised by ``states.cross_fit``
    at f_D's k (``k``, default ``states.KNN_K``) and on y itself (the residualisation turned
    off), the units' probe-alone R2 and the kNN's on T; with ``t_wide`` the T-wide statistic,
    with ``faults`` (first seed) the over-smoothed kNN (``oversmooth_k``) and D1 without
    s - s_apex (at f_D's k), each paired, with its k. Pass: every seed's Delta2 <=
    ``K4_ZERO_TOL`` and every seed's
    unresidualised Delta2 > it, and for PR-6b every seed's probe-alone R2 >= the kNN's (means
    over K_P). 'leak_bound' is the largest Delta2 over the seeds ('leak_bound_unpaired' the
    unpaired one), 'gain_max' each component's largest gain, 'host_offset' the host's offset
    per seed."""
    j = {"smooth": 0, "knn": 1}[kind]
    k = ST.KNN_K if k is None else int(k)
    noise = (lambda s: k4_rng(14, s)) if noise is None else noise
    w, fo = pw.w, pw.fo
    F = pr6b_f(pw) if kind == "smooth" else pr6c_f(pw)
    U = hold_units(F, w, k4_rng(15, j))
    h0 = pw.h3[0]
    H = np.hstack([h0, U])
    D1 = ST.decoy_d1(pw.tab)
    Fs = _std(F, w)
    out: dict = {"kind": kind, "F": "pr6b_f" if kind == "smooth" else "pr6c_f",
                 "n_controls": len(pw.h3[1:1 + PB.N_RANDOM_ACTORS]),
                 "rows": int(np.sum(pw.rows_c2)) if pw.rows_c2 is not None else len(w),
                 "passing": None if pw.power_floor is None else dict(pw.power_floor),
                 "k": k, "statistic": K4_PAIRED, "seeds": []}
    for s in seeds:
        y = Fs + K4_PR6_NOISE * _ar1(pw.tab, w, F.shape[1], noise(s))
        res = ST.cross_fit(D1, y, fo.outer, w, k=k)
        on, off = c2_primary(pw, H, res.by_fold, host=h0), c2_primary(pw, H, y, host=h0)
        alone = PB.probe_r2(PB.ridge_cv(None, y, fo, w, U, pw.names), w, pw.direction)
        knn = PB.r2(y, y - res.oof, w, pw.direction, pw.names)
        rec = {"seed": int(s), "k": res.k, **_stat(on),
               "scored_rows": on.get("scored_rows"), "off": _stat(off),
               "alone_r2": alone.tolist(), "knn_r2": knn.tolist(),
               "alone_mean": float(np.mean(alone)), "knn_mean": float(np.mean(knn))}
        if t_wide:
            tw = c2_primary(pw, H, res.by_fold, t_wide=True, host=h0)
            rec["T"] = _stat(tw)
        out["seeds"].append(rec)
        if faults and s == seeds[0]:
            bad = {"knn_k300": ST.cross_fit(D1, y, fo.outer, w, k=oversmooth_k(fo.outer)),
                   "d1_without_ds_apex": ST.cross_fit(D1[:, 1:], y, fo.outer, w, k=k)}
            out["faults"] = {}
            for lab, rf in bad.items():
                st = _stat(c2_primary(pw, H, rf.by_fold, host=h0))
                out["faults"][lab] = {**st, "detected": _gt(st["delta2"], K4_ZERO_TOL),
                                      "k": rf.k, "descriptive": True}
    rs = out["seeds"]
    d2 = [_num(r["delta2"]) for r in rs]
    out["leak_bound"] = float(max(d2)) if rs and np.all(np.isfinite(d2)) else None
    d2u = [_num((r.get("unpaired") or {}).get("delta2")) for r in rs]
    out["leak_bound_unpaired"] = float(max(d2u)) if rs and np.all(np.isfinite(d2u)) else None
    out["host_offset"] = [(r.get("host_offset") or {}).get("delta2") for r in rs]
    gains = [[_num(v) for v in (r["gain"] or [])] for r in rs]
    out["gain_max"] = None
    if gains and all(len(g) == len(K_P) for g in gains):
        col = np.asarray(gains).T
        out["gain_max"] = [float(np.max(c[np.isfinite(c)])) if np.isfinite(c).any() else None
                           for c in col]
    out["residualised_ok"] = all(_le(r["delta2"], K4_ZERO_TOL) for r in rs)
    out["off_detected"] = all(_gt(r["off"]["delta2"], K4_ZERO_TOL) for r in rs)
    out["alone_ok"] = all(r["alone_mean"] >= r["knn_mean"] for r in rs)
    out["pass"] = bool(rs and out["residualised_ok"] and out["off_detected"]
                       and (kind != "smooth" or out["alone_ok"]))
    return out


def band_notes(rec: dict) -> dict:
    """K4's two numbers for C2's band-edge notes (``analysis.band_notes``; not C2-sw's,
    module docstring), from a K4 record: PR-6's over-fitting cost (its paired Delta2's negative part, with the Delta2
    and its CI: T, the mean over K_P, K4's controls) and PR-6c's leak bound (its largest
    paired Delta2 over the noise seeds on C2's primary statistic, with each component's
    largest gain). Both are the plant's paired increment in its host (module docstring 'K4's
    no-gain plants'), the host's own offset against the controls taken out."""
    p6, p6c = rec.get("PR-6") or {}, rec.get("PR-6c") or {}
    return {"pr6_cost": {"cost": p6.get("overfit_cost"), "delta2": p6.get("delta2"),
                         "ci": p6.get("ci"),
                         "scope": "K4 PR-6: the paired increment on T, the mean over K_P, "
                                  "K4's random controls"},
            "pr6c_leak": {"bound": p6c.get("leak_bound"), "gain_max": p6c.get("gain_max"),
                          "seeds": len(p6c.get("seeds") or []),
                          "scope": "K4 PR-6c: the paired increment of C2's primary "
                                   "statistic, the largest over its noise seeds"}}


# ------------------------------------------------------------------------- f_D's k


def decoy_k_rng(v: int) -> np.random.Generator:
    """Validation seed ``v``'s noise for the choice of f_D's k (module docstring 'f_D's k'):
    stream ``costate_decoy_k``, sub-stream v, one draw shared by PR-6b and PR-6c as K4's
    gate seeds share theirs (``costate_k4``, sub-stream (14, seed))."""
    return stream_rng("costate_decoy_k", v)


def at_decoy_k(pw: ProbeWorld, k: int) -> tuple[ProbeWorld, dict]:
    """``pw`` with lambda_res cross-fitted at f_D's k = ``k`` and C2's power floor recomputed
    there (``prereg.frozen_values``' rule: partition 0's own-fold residuals, the interior's
    shares, ``states.decoy_shares``), with the shares and both floors (input side only)."""
    res = ST.cross_fit(ST.decoy_d1(pw.tab), pw.lam, pw.fo.outer, pw.w, k=k)
    share, share_int = ST.decoy_shares(pw.tab, pw.lam, res.oof, pw.w)
    floor = ST.power_floor(share_int)
    info = {"k": int(k), "residual_share": dict(zip(K_P, share.tolist(), strict=True)),
            "residual_share_interior": dict(zip(K_P, share_int.tolist(), strict=True)),
            "power_floor": floor, "power_floor_T": ST.power_floor(share)}
    return dataclasses.replace(pw, lam_res=res.by_fold, power_floor=floor), info


def decoy_k_record(pk: ProbeWorld, k: int, kind: str, v: int, faults: bool = False) -> dict:
    """One validation seed's ``pr6_world`` record of one world (``kind``) with f_D at ``k``,
    on ``pk``, ``pw`` at k (``at_decoy_k``): the call ``decoy_k_leaks`` makes in this process
    and ``decoy_k_unit`` in a worker."""
    return pr6_world(pk, kind, (v,), t_wide=False, faults=faults, k=k, noise=decoy_k_rng)


def decoy_k_unit(state, unit) -> dict:
    """A worker's unit of the decoy-k stage (``workers.run``): ``state`` (pw, {k: (lambda_res,
    power floor)} from this process' ``at_decoy_k``), ``unit`` (k, kind, seed); ``pw`` at k
    rebuilt from those arrays exactly as ``at_decoy_k`` builds it, then ``decoy_k_record``."""
    pw, at = state
    k, kind, v = unit
    lam_res, floor = at[k]
    pk = dataclasses.replace(pw, lam_res=lam_res, power_floor=floor)
    return decoy_k_record(pk, k, kind, v)


def decoy_k_leaks(pw: ProbeWorld, k: int, seeds: Sequence[int] = DECOY_K_SEEDS,
                  faults: bool = False, done: tuple | None = None) -> dict:
    """PR-6b's and PR-6c's records on the validation seeds (``decoy_k_rng``) with f_D at
    ``k``, on ``pw`` at k (``at_decoy_k``): per seed C2's primary Delta2 exactly as K4
    computes it on its gate seeds (``pr6_world``: the units' paired increment over the host
    alone), with ``faults`` the planted faults on every seed (each seed its own ``pr6_world``
    call, ``decoy_k_record``). 'delta2' {kind: [per seed]} is what ``decoy_k_choice`` reads;
    the rest ('delta2_unpaired', 'host_offset' and the records) is descriptive. ``done``
    ((pw at k, its info), {(kind, seed): record}) holds them computed already (the worker
    pool's, ``decoy_k``)."""
    pk, info = at_decoy_k(pw, k) if done is None else done[0]
    out: dict = {"k": int(k), "real": info, "seeds": [int(v) for v in seeds], "delta2": {},
                 "delta2_unpaired": {}, "host_offset": {}}
    for kind in DECOY_K_WORLDS:
        recs = [decoy_k_record(pk, k, kind, v, faults) if done is None else done[1][(kind, v)]
                for v in seeds]
        out[kind] = {"seeds": [r["seeds"][0] for r in recs],
                     "faults": [r.get("faults") for r in recs] if faults else None}
        out["delta2"][kind] = [r["seeds"][0]["delta2"] for r in recs]
        out["delta2_unpaired"][kind] = [(r["seeds"][0].get("unpaired") or {}).get("delta2")
                                        for r in recs]
        out["host_offset"][kind] = [(r["seeds"][0].get("host_offset") or {}).get("delta2")
                                    for r in recs]
    return out


def decoy_k_choice(delta2: dict, candidates: Sequence[int] = DECOY_K_CANDIDATES,
                   tol: float = K4_ZERO_TOL) -> dict:
    """f_D's k by the rule (module docstring 'f_D's k'): among ``candidates`` the largest k
    for which every validation seed of PR-6b ('smooth') and PR-6c ('knn') gives C2's primary
    Delta2, the units' paired increment, <= ``tol`` (``delta2`` {k: {kind: [Delta2 per
    seed]}}; a missing or non-finite Delta2 fails); when none does, the smallest candidate,
    flagged as a blocker."""
    ok = {}
    for k in candidates:
        rec = delta2.get(k, delta2.get(str(k), {}))
        vals = [rec.get(kind) or [] for kind in DECOY_K_WORLDS]
        ok[int(k)] = all(len(v) > 0 and all(_le(x, tol) for x in v) for v in vals)
    good = [k for k, v in ok.items() if v]
    return {"k": max(good) if good else min(ok), "qualifying": good, "blocker": not good,
            "candidates": [int(k) for k in candidates], "tol": tol}


def decoy_k_reproduction(delta2: dict, delta2_unpaired: dict,
                         table: dict | None = None, table_unpaired: dict | None = None,
                         tol: float = DECOY_K_REPRODUCE_TOL) -> dict:
    """Whether a run's per-seed Delta2 (``delta2`` and ``delta2_unpaired``, {k: {kind: [per
    seed]}}) reproduce the record's table (``DECOY_K_VALIDATION`` and
    ``DECOY_K_VALIDATION_UNPAIRED`` by default): the largest absolute difference over every
    candidate, world and seed the table holds (None when a value is missing or not finite),
    and 'ok' when both are at most ``tol``."""
    table = DECOY_K_VALIDATION if table is None else table
    table_unpaired = DECOY_K_VALIDATION_UNPAIRED if table_unpaired is None else table_unpaired

    def diff(got: dict, want: dict) -> float | None:
        worst = 0.0
        for k, kinds in want.items():
            rec = got.get(k, got.get(str(k))) or {}
            for kind, vals in kinds.items():
                have = list(rec.get(kind) or [])
                if len(have) != len(vals):
                    return None
                for x, v in zip(have, vals, strict=True):
                    if not np.isfinite(_num(x)):
                        return None
                    worst = max(worst, abs(_num(x) - float(v)))
        return float(worst)

    d, du = diff(delta2, table), diff(delta2_unpaired, table_unpaired)
    return {"tol": tol, "max_abs_diff": d, "max_abs_diff_unpaired": du,
            "ok": bool(d is not None and du is not None and d <= tol and du <= tol)}


def decoy_k(pw: ProbeWorld, candidates: Sequence[int] = DECOY_K_CANDIDATES,
            seeds: Sequence[int] = DECOY_K_SEEDS, progress: Callable | None = None,
            workers: int = 1) -> dict:
    """The pre-tag stage 'decoy-k' (module docstring 'f_D's k'): ``decoy_k_leaks`` on ``pw``
    (C2's decision stratum, mask and frozen power floor) at every candidate on the validation
    seeds, the rule's choice (``decoy_k_choice``) and the reproduction of the record's table
    (``decoy_k_reproduction``), with each k's residual shares and floors ('real'). Pass: the
    choice is ``states.KNN_K``, not a blocker, and every value reproduces the table. With
    ``workers`` > 1 the (k, world, seed) units run in that many spawned processes
    (``decoy_k_unit``; ``at_decoy_k`` once per k here) and the record is the serial one."""
    from apexzero.costate import workers as WK

    by_k = {}
    done: dict = {}
    if workers > 1:
        at = {int(k): at_decoy_k(pw, int(k)) for k in candidates}
        units = [(int(k), kind, int(v)) for k in candidates for kind in DECOY_K_WORLDS
                 for v in seeds]
        got = WK.run(decoy_k_unit, units, workers,
                     state=(pw, {k: (pk.lam_res, pk.power_floor) for k, (pk, _) in at.items()}),
                     names=[f"k = {k}, {kind}, seed {v}" for k, kind, v in units])
        for (k, kind, v), rec in zip(units, got, strict=True):
            done.setdefault(k, (at[k], {}))[1][(kind, v)] = rec
    for k in candidates:
        by_k[int(k)] = (decoy_k_leaks(pw, int(k), seeds) if not done
                        else decoy_k_leaks(pw, int(k), seeds, done=done[int(k)]))
        if progress:
            progress(f"k = {int(k)}: {by_k[int(k)]['delta2']}")
    delta2 = {k: r["delta2"] for k, r in by_k.items()}
    unpaired = {k: r["delta2_unpaired"] for k, r in by_k.items()}
    choice = decoy_k_choice(delta2, candidates)
    rep = decoy_k_reproduction(delta2, unpaired)
    return {"candidates": [int(k) for k in candidates], "seeds": [int(v) for v in seeds],
            "stream": "costate_decoy_k", "statistic": K4_PAIRED,
            "delta2": {str(k): v for k, v in delta2.items()},
            "delta2_unpaired": {str(k): v for k, v in unpaired.items()},
            "host_offset": {str(k): r["host_offset"] for k, r in by_k.items()},
            "real": {str(k): r["real"] for k, r in by_k.items()},
            "records": {str(k): {kind: r[kind]["seeds"] for kind in DECOY_K_WORLDS}
                        for k, r in by_k.items()},
            "choice": choice, "knn_k": ST.KNN_K, "reproduces": rep,
            "pass": bool(choice["k"] == ST.KNN_K and not choice["blocker"] and rep["ok"])}


# ------------------------------------------------------------------------- C2's null


def c2_null(pw: ProbeWorld, cfg, seeds: Sequence[int] = C2_NULL_SEEDS,
            progress: Callable | None = None) -> dict:
    """C2's random-only null (module docstring): each of ``seeds`` a random-init actor of
    ``cfg`` scored as the network through C2's primary statistic as the probes stage scores
    a trained network (``analysis.c2_network`` with its own h3 and m, the pool of random-init
    actors ``probes.POOL_SEEDS`` as the controls with the two-level CI's draws
    ``probes.pool_draws``, on ``pw``'s z, by-fold lambda_res, folds, decision stratum, mask
    and frozen power floor, key weights renormalised on the stratum, each decision ruled by
    ``probes.LOGO_RULE`` with its sets and the full set's decision beside it), and the summary
    (``c2_null_summary``), with v2.5's 5-control summary beside it (``C2_NULL_FIVE_CONTROLS``,
    descriptive). Descriptive; 'pass': every seed's statistic is finite."""
    if pw.rows_c2 is None or pw.mask is None or pw.power_floor is None:
        raise ValueError("C2's random-only null needs C2's decision stratum, mask and power "
                         "floor")
    controls = tuple(PB.POOL_SEEDS)
    clash = sorted(set(map(int, seeds)) & set(controls))
    if clash:
        raise ValueError(f"the null's seeds {clash} are in C2's control pool")
    rows = np.asarray(pw.rows_c2, bool)
    wr = ST.stratum_weights(pw.tab, rows)
    labels = AN.geometry_labels(pw.tab)
    ctrl = [PB.hidden(PB.random_actor(cfg, s), pw.z)["h3"] for s in controls]
    draws = PB.pool_draws(len(ctrl))
    nets = []
    for s in seeds:
        hid = PB.hidden(PB.random_actor(cfg, int(s)), pw.z)
        try:
            rec = AN.c2_network(hid["h3"], ctrl, pw.z, hid["m"], pw.lam_res, pw.fo, None,
                                pw.direction, K_P, pw.power_floor, rows, weights=wr[rows],
                                mask=pw.mask, pool=draws, controls=controls, logo="rule",
                                labels=labels)
        except PB.NotTestable as e:
            rec = {"decision": "not testable", "reason": str(e)}
        ci_g = rec.get("ci_geometry")
        nets.append({"seed": int(s), **{k: rec.get(k) for k in (
            "stat", "ci", "ci_geometry", "gain", "raw", "decision", "decision_all_geometries",
            "geometry_sensitive", "reason", "n_stratum", "scored_rows", "pool", "logo")},
            "decision_geometry_ci": None if ci_g is None else PB.decide(
                float(rec["stat"]), float(ci_g[0]), float(ci_g[1]))})
        if progress:
            progress(f"seed {int(s)}: {rec.get('stat')} {rec.get('ci')} {rec.get('decision')}")
    summ = c2_null_summary(nets)
    return {"seeds": [int(s) for s in seeds], "controls": list(controls), "ci": "two-level",
            "logo": PB.LOGO_RULE,
            "decoy_k": ST.KNN_K, "passing": dict(pw.power_floor), "rows": int(rows.sum()),
            "obs_rms": "G1", "networks": nets, "summary": summ,
            "five_controls_v25": {k: list(v) if isinstance(v, tuple) else v
                                  for k, v in C2_NULL_FIVE_CONTROLS.items()},
            "pass": bool(nets and summ["n_finite"] == len(nets))}


def c2_null_summary(nets: Sequence[dict]) -> dict:
    """The null's distribution over its networks' statistics: their number (and how many are
    finite), mean, SD (ddof 1), minimum, maximum and 5th, 50th and 95th percentiles (linear),
    the decisions' counts, and how many CIs lie above 0 or below 0; under the single-geometry
    rule ('logo') the full sets' decisions' counts, the seeds it turns inconclusive and how
    often each geometry changes one."""
    st = np.array([_num(r.get("stat")) for r in nets], float)
    fin = st[np.isfinite(st)]
    cis = [r.get("ci") for r in nets if r.get("ci") is not None]
    dec: dict = {}
    for r in nets:
        dec[str(r.get("decision"))] = dec.get(str(r.get("decision")), 0) + 1

    def q(p):
        return float(np.quantile(fin, p)) if fin.size else None

    out = {"n": len(nets), "n_finite": int(fin.size),
           "mean": float(fin.mean()) if fin.size else None,
           "sd": float(fin.std(ddof=1)) if fin.size > 1 else None,
           "min": float(fin.min()) if fin.size else None,
           "max": float(fin.max()) if fin.size else None,
           "p05": q(0.05), "p50": q(0.5), "p95": q(0.95), "decisions": dec,
           "ci_above_0": int(sum(_num(c[0]) > 0 for c in cis)),
           "ci_below_0": int(sum(_num(c[1]) < 0 for c in cis))}
    if any("decision_all_geometries" in r for r in nets):
        # the single-geometry robustness rule (probes.LOGO_RULE): the full sets' decisions,
        # the networks it turns inconclusive and the geometries that change them
        da: dict = {}
        flips: dict = {}
        for r in nets:
            d = str(r.get("decision_all_geometries", r.get("decision")))
            da[d] = da.get(d, 0) + 1
            for gm in r.get("geometry_sensitive") or ():
                flips[str(gm["geometry"])] = flips.get(str(gm["geometry"]), 0) + 1
        out["logo"] = {"decisions_all_geometries": da,
                       "geometry_sensitive": int(sum(bool(r.get("geometry_sensitive"))
                                                     for r in nets)),
                       "changing_geometries": flips}
    geo = [r.get("ci_geometry") for r in nets if r.get("ci_geometry") is not None]
    if geo:                   # the two-level null: the same points' geometry-only CIs beside it
        dg: dict = {}
        for r in nets:
            if r.get("decision_geometry_ci") is not None:
                dg[r["decision_geometry_ci"]] = dg.get(r["decision_geometry_ci"], 0) + 1
        out["ci_width_mean"] = float(np.mean([_num(c[1]) - _num(c[0]) for c in cis]))
        out["geometry_ci"] = {"decisions": dg,
                              "ci_above_0": int(sum(_num(c[0]) > 0 for c in geo)),
                              "ci_below_0": int(sum(_num(c[1]) < 0 for c in geo)),
                              "ci_width_mean": float(np.mean([_num(c[1]) - _num(c[0])
                                                              for c in geo]))}
    return out


def c2_null_frozen(rec: dict) -> dict:
    """The record's copy of C2's random-only null (a ``c2_null`` record, the bound
    gates/c2_null.json): the summary, each seed's statistic, CI and decision, and its scope,
    which ``analysis.c2_null_note`` prints beside each network's primary C2."""
    nets = rec.get("networks") or []
    ruled = ({"decisions_all_geometries": [r.get("decision_all_geometries") for r in nets]}
             if rec.get("logo") else {})
    return {"summary": rec.get("summary"), "seeds": [r.get("seed") for r in nets],
            "stats": [r.get("stat") for r in nets], "ci": [r.get("ci") for r in nets],
            "decisions": [r.get("decision") for r in nets], **ruled,
            "scope": (f"C2's primary statistic of {len(nets)} random-init actors (seeds "
                      f"{rec.get('seeds')}) as the network against the random-init controls "
                      f"{rec.get('controls')} ({rec.get('ci', 'geometry')} CI): "
                      f"{rec.get('obs_rms')}'s obs_rms, C2's decision "
                      f"stratum ({rec.get('rows')} rows) and mask, the frozen power floor "
                      f"{rec.get('passing')}, partition 0, key weights, f_D at k = "
                      f"{rec.get('decoy_k')}"
                      + ("; each decision ruled by the single-geometry robustness rule"
                         if rec.get("logo") else "") + "; descriptive")}


def pr8(pw: ProbeWorld, counts: np.ndarray, snr: float = 4.0) -> dict:
    """PR-8: a unit holding sigma^ped on C2-sw's rows (``states.SW_STRATUM``: the folded
    pedal-command multiplier active, a unique quantity) residualised out of fold on random
    h3 + z, plus noise of SNR 4; C2-sw's gain against (1 - R2_b) SNR / (1 + SNR)."""
    rows = np.asarray(pw.tab[ST.SW_STRATUM], bool)
    if rows.sum() < 30:
        return {"pass": False, "reason": "too few pedal-switching rows"}
    sub = pw.fo.subset(rows)
    w = ST.weights({k: np.asarray(v)[rows] for k, v in pw.tab.items()}, "key")
    sig = np.asarray(pw.tab["sigma_pedal"], float)[rows]
    zr, h0 = pw.z[rows], pw.h3[0][rows]
    base = PB.ridge_cv(np.hstack([h0, zr]), sig, sub, w, None, ("sigma_pedal",))
    r = base.y - base.pred
    H = PB.plant_units(h0, r, snr, w, k4_rng(9))
    G = len(np.unique(sub.cluster))         # the stratum's geometries (a subset of T's)
    g = PB.c2_sw(H, [h[rows] for h in pw.h3[1:1 + K4_N_CONTROLS]], zr, sig, sub, w, None,
                 PB.cluster_counts(G, rng=k4_rng(10)))
    r2b = float(PB.probe_r2(base, w)[0])
    exp = (1.0 - r2b) * PB.expected_r2(snr)
    return {"gain": g.stat, "expected": exp, "r2_b": r2b, "n": int(rows.sum()),
            "pass": bool(abs(g.stat - exp) <= K4_TOL)}


def pr9(pw: ProbeWorld, counts: np.ndarray, n_heads: int = 5) -> dict:
    """PR-9: heads sharing V exactly on the rows, their gradients g + c_k grad u with a known
    field grad u; D_V = 0 and D_grad equal to its closed form."""
    n = len(pw.w)
    rng = k4_rng(11)
    g = rng.standard_normal((n, 8))
    du = rng.standard_normal((n, 8))
    c = np.linspace(-1.0, 1.0, n_heads)
    V = rng.standard_normal(n)
    q = {"V": np.repeat(V[:, None], n_heads, 1)}
    for i, nm in enumerate(STATE_NAMES):
        q[f"g_{nm}"] = g[:, i:i + 1] + c[None] * du[:, i:i + 1]
    cl, G = pw.fo.cluster, int(pw.fo.cluster.max()) + 1
    res = AN.c1_heads(q, pw.w, cl, G, np.ones(n, bool), counts)
    w = pw.w / pw.w.sum()
    expect = {}
    for i, nm in enumerate(STATE_NAMES):
        within = np.var(c, ddof=1) * du[:, i] ** 2
        mean = g[:, i] + c.mean() * du[:, i]
        expect[f"g_{nm}"] = float((w @ within) / (w @ (mean - w @ mean) ** 2))
    err = max(abs(res["D"][k] - v) for k, v in expect.items())
    return {"D_V": res["D"]["V"], "max_err": float(err),
            "pass": bool(res["D"]["V"] <= 1e-20 and err <= 1e-10)}


def pr4(cfg, pw: ProbeWorld, component: int = 3) -> dict:
    """PR-4: a random-init actor and critic trunk whose h2 holds tanh of a ridge readout of z
    fitted to lambda_k on all rows (``probes.plant_readout``); the readout's R2 recovered from
    h2 and h3 by the probe pipeline. Read one-sided: each layer's R2 at least the readout's
    minus ``K4_READOUT_TOL`` = 0.02, since the probe on h2 or h3 also sees the layer's other
    units and can exceed the readout alone."""
    w, fo = pw.w, pw.fo
    y = pw.lam[:, component]
    zs = PB.scaler(pw.z, w)
    Z = zs.apply(pw.z)
    beta = np.linalg.lstsq(np.hstack([Z, np.ones((len(w), 1))]) * np.sqrt(w)[:, None],
                           _std(y, w) * np.sqrt(w), rcond=None)[0]
    v = np.zeros(pw.z.shape[1])
    v[zs.keep] = 0.5 * beta[:-1] / zs.sd[zs.keep]
    c0 = 0.5 * beta[-1] - float(np.sum(zs.mean[zs.keep] * v[zs.keep]))
    readout = np.tanh(pw.z @ v + c0)
    name = (pw.names[component],)
    r_read = float(PB.probe_r2(PB.ridge_cv(None, y, fo, w, readout, name), w, pw.direction)[0])
    out = {"readout_r2": r_read}
    ok = True
    for kind, net in (("actor", PB.random_actor(cfg, 0)), ("critic", PB.random_critic(cfg, 0))):
        pl = PB.plant_readout(net, v, c0)
        hid = PB.hidden(pl, pw.z)
        unit_err = float(np.max(np.abs(hid["h2"][:, 0] - readout)))
        r2 = {lay: float(PB.probe_r2(PB.ridge_cv(None, y, fo, w, hid[lay], name), w,
                                     pw.direction)[0]) for lay in ("h2", "h3")}
        out[kind] = {"unit_err": unit_err, **r2}
        ok &= all(r2[lay] >= r_read - K4_READOUT_TOL for lay in r2) and unit_err < 1e-6
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------------- K5, K6


def k5(pw: ProbeWorld, counts: np.ndarray | None = None, ceiling: str = "ridge") -> dict:
    """K5 (module docstring): the permuted-target wiring check on [z, m] + h3 (its verdict is
    K5's) and the control task per K_P component (descriptive; ``ceiling`` 'mlp' is the
    design's MLP-on-z ceiling, 'ridge' a cheaper stand-in for dry runs)."""
    w, fo = pw.w, pw.fo
    G = int(fo.cluster.max()) + 1
    counts = PB.cluster_counts(G, rng=k4_rng(12)) if counts is None else counts
    base = np.hstack([pw.z, pw.m[0]])
    wn = PB.wiring_null(base, pw.lam, fo, w, pw.h3[0], pw.names, pw.direction, counts)

    def ceil_of(y):
        if ceiling == "mlp":
            p = PB.mlp_cv(pw.z, y, fo, w, ("y",))
        else:
            p = PB.ridge_cv(pw.z, y, fo, w, None, ("y",))
        return float(PB.probe_r2(p, w)[0])

    ctrl = {}
    for k, nm in enumerate(pw.names):
        r_star = float(PB.probe_r2(PB.ridge_cv(pw.z, pw.lam[:, k], fo, w, None, (nm,)), w,
                                   pw.direction)[0]) if ceiling != "mlp" else ceil_of(pw.lam[:, k])
        r_star = float(np.clip(r_star, 0.05, 0.95))
        c = PB.control_task(pw.z, w, k, r_star, ceil_of)
        hid = PB.probe_r2(PB.ridge_cv(base, c.y, fo, w, pw.h3[0], ("y",)), w)[0]
        tgt = PB.probe_r2(PB.ridge_cv(base, pw.lam[:, k], fo, w, pw.h3[0], (nm,)), w,
                          pw.direction)[0]
        ctrl[nm] = {"r_star": r_star, "ceiling": c.ceiling, "ok": c.ok, "s": c.s,
                    "selectivity": float(PB.selectivity(tgt, hid))}
    return {"wiring": wn, "control": ctrl, "pass": bool(wn["pass"])}


def k6(tab: io.Table, observable_s: np.ndarray | None = None,
       kappa_clipped: np.ndarray | None = None, k2b_excluded: dict | None = None,
       multi_basin: list[dict] | None = None) -> dict:
    """K6's coverage (Section 7): T's size, per reason and component, s's unobservability by
    distance band past s_c1, the interior share by phase. ``multi_basin``: the keys left out
    of T when it was built (inputs/summary.json's 'excluded': id and row count); T's own
    column counts any left in."""
    n = io.check_table(tab)
    w = ST.weights(tab, "key")
    ds = np.asarray(tab["ds_c1"], float)
    bands = {"before": ds <= 0, "0-10": (ds > 0) & (ds <= 10), "10-25": (ds > 10) & (ds <= 25),
             "25+": ds > 25}
    out = {"n": int(n), "keys": len(np.unique(tab["key_id"])),
           "geometries": len(np.unique(ST.geometry_index(tab))),
           "multi_basin": int(np.sum(np.asarray(tab["multi_basin"], bool))),
           "multi_basin_excluded": None if multi_basin is None else {
               "keys": len(multi_basin), "rows": int(sum(int(m["rows"]) for m in multi_basin))},
           "pedal_switching": int(np.sum(np.asarray(tab[ST.SW_STRATUM], bool)))
           if ST.SW_STRATUM in tab else None,
           "contact": int(np.sum(np.asarray(tab["contact"], bool))),
           "pedal_saturated": int(np.sum(np.asarray(tab["pedal_saturated"], bool))),
           "interior": float(w @ np.asarray(tab["interior"], float)),
           "interior_unweighted": float(np.mean(np.asarray(tab["interior"], bool))),
           "interior_by_phase": {p: float(np.mean(np.asarray(tab["interior"], bool)[
               np.asarray(tab["phase"]) == p])) for p in ST.PHASES
               if np.any(np.asarray(tab["phase"]) == p)}}
    if observable_s is not None:
        uo = ~np.asarray(observable_s, bool)
        out["s_unobservable"] = {b: int(np.sum(uo & m)) for b, m in bands.items()}
    if kappa_clipped is not None:
        out["kappa_clipped"] = int(np.sum(np.asarray(kappa_clipped, bool)))
    if k2b_excluded is not None:
        ph = np.asarray(tab["phase"])
        out["beyond_preview"] = {c: int(sum(np.sum(ph == p) for p, cs in k2b_excluded.items()
                                            if c in cs)) for c in K_P}
    return out
