# Step 2: receptive fields (design v3.3, not frozen)

Status: 21 Oct 2026.
- v1 had one code review.
- v2 had a five-lens adversarial review in which 57 of 59 findings survived independent
  verification. v3 adopted them; Section 13 maps each group of findings to its change.
- v3.1 records the decisions made while implementing v3 in `src/apexzero/fields/` (Section 14).
  An adversarial code review confirmed 22 findings, all fixed.
- v3.2 records the decisions taken after a final pre-tag review: 46 findings survived two
  refuters each, 34 marked essential (Section 15).
- v3.3 records the implementation of Section 15. The implementation was reviewed on three lenses
  (concordance, freeze integrity, tests); 28 findings were confirmed, none rejected, all fixed.
  It also records three decisions taken at that point (Section 16).

Nothing here has been computed on a trained agent's field.

The rules of step 1 carry over:
- design, review, a dry run on controls with known answers, then a tag
  (`receptive-fields-prereg-v1`) before any agent's field is computed;
- R_TEST stays sealed;
- step 1's code (`src/apexzero/interp`, `scripts/psychophysics.py`) stays frozen. Step 2 code lives
  in `src/apexzero/fields/` and `scripts/fields.py`, and imports the frozen modules read-only.

**Departure from protocol 9.6.** Protocol 9.6 prioritises step 1b (two-corner stimuli) after a
low F for G2's tail crashes. Step 2 is taken first because the ladder's necessary factor was the
width profile, not a neighbouring corner. This is logged as a deviation.

## 1. Question

Step 1 ([findings](../findings/psychophysics-step1.md)) found the following:
- The generalists read corner severity lawfully on typical approaches (P1a, all 9 networks).
- Their real-circuit crashes are not single-corner failures (P8-tail refuted).
- 159 of 160 real-road corner crashes leave by the **inside edge**, where that edge moves by
  metres within the last 100 m. Holding the width constant rescues them.
- Training on real width profiles raised R-dev completion. About half of the gain transfers to
  circuits whose widths were not seen ([width test](../findings/width-test.md)).
- The networks are not mirror-symmetric (P5; the re-run gate passes, so the asymmetry is
  learned).

Step 2 asks for the mechanism. How does each network's behaviour through a corner respond to the
road ahead (curvature, inside width, outside width), as a function of where the change is? And
how does that compare with the optimum of its own objective from the same state? The central
question is lateral: **when the inside edge moves, does the network's line follow it?** The
optimum's does: its inside corridor constraint is active at the apex in every key, so its line
moves one-for-one with the edge.

## 2. States

**A2 phase keys (agent-independent; primary state set).**
- 481 verified keys at approach A2 (the typical approach for the corner's size):
  - 142 u = 0 nominal keys (`outputs/psychophysics/keys/nominal`; 99 at W = 400, 43 at W = 800);
  - 339 phase-matched keys u1–u3 (`keys/phase`; 264 at W = 400, 75 at W = 800).
- Each is solved from the cell's designed start state x0, with a_{-1} = 0.
- Every network's deterministic run at that phase starts from the same x0 with prev_action 0.
  So all networks, including networks without step 1 rollouts, are compared at identical
  states.
- The optimum's free decision is a*_0.

**A1 / tier B anchored keys (agent-specific; secondary state set).**
- 9,525 verified keys: 6,452 at A1 (310 km/h from 600 m) and 3,073 on tier B at D = 600 m
  (160, 220, 280 km/h).
- All are fast-approach states: the median brake speed v_b* is 295–307 km/h. None is at A2,
  because s_L lies before the start there. ("typical" in a stimulus name is the ramp kind, not
  the approach.)
- Each is solved from the agent's state x_{L-1} with a_0 fixed to the agent's a_{L-1}
  (`AnchoredSolver`), so the free decision is a_1 at x_1.
- Each key belongs to the network in `anchor.snapshot`.
- **Finals.** G1 701, G2 722, G3 751, G1s1 710, G1s2 705, G2s1 742, G2s2 723, G3s1 760, G3s2 759,
  S1 766, S2 404, S3 393, for 8,136 in all. Per-agent claims use finals only.
- **Earlier snapshots.** 1,389 keys at seed 0's earlier snapshots (step 1's P6c keys). These are
  computed with their own snapshot's network and are descriptive.

**Width within the stimuli.**
- The anchored keys' 60 stimuli are all W 12. They are constant-width except the 8 at R 6.0 and
  7.5 m, where the generator's inside cap narrows the inside to 4.60–5.59 m over 1–7 apex knots.
- The A2 keys also include tier C stimuli at W 9 and W 15. At R 9.4 / W 15 the cap narrows the
  inside to 6.64 m.
- Every stimulus whose geometry has the cap binding (`Geometry.cap_binds`) is a separate,
  descriptive stratum.

**Horizon twins.** 6 tier V cells (the V2 horizon check) have both a W 400 and a W 800 nominal
key.
- The per-pair summaries use the 475 primary A2 keys, those whose W equals the cell's W
  (`keys.a2_primary_refs`).
- Gate 4 runs on all 481.

**Where the pedal is saturated.**
- The optimum's pedal at the anchored decision is full throttle in about 99% of keys.
- The networks' pre-clip pedal is saturated in only 0–12% of steps over the horizon. It is +1
  early and never reaches −1, with a minimum of about −0.45. So the network's closed loop is
  differentiable through about 70 interior steps.
- Instantaneous pedal fields are not used. Plan-level fields are.

**States for H3 (Section 7), new.**
- G2's three crash units have no keys on the constant-width rung.
- H3 needs:
  - full-recipe nominal and phase keys on cells 584, 605 and 612;
  - then W = 400 anchored solves at G2's states.
- Budgeted in Section 11. R-dev circuits only.

## 3. Outcomes

These are defined identically on the network's float64 closed loop and on the optimum's planned
trajectory. Step 1's own definitions are imported read-only from `interp/measures.py` wherever
one exists.

| Outcome | Definition | Role |
|---|---|---|
| **c_min** | Minimum inside clearance w_in(s) − 1 − σ n(s) over [s_c0 − 50, s_c1 + 50], with σ the turn direction. This is the metric form of the protocol's clearance, against the termination boundary (`racing.py`). The derivative is taken at the argmin step (Danskin). | Primary, lateral |
| **v_apex** | Step 1's v_min: min vx over [s_c0 − 50, s_c1 + 50] with interpolated ends (`V_MIN_MARGIN` 50). The derivative is taken at the argmin step. | Secondary |
| **s_b** | Step 1's brake onset: the interpolated 0.3 crossing of beta_f, from `brake_onset` (measures.py). It is differentiated with the onset run fixed from the base trajectory. | Secondary |
| **steer_0** | The steer mean at the decision state (the instantaneous field) | Descriptive |

**Names.** The bump origin is **s_1**:
- for the optimum, the key's x[1, 0] for anchored keys, or x[0, 0] for A2 keys;
- for the network, the matched det_x[L, 0], asserted equal to within 1e-3 m.

Step 1's `measures.s_dec`, `measures.s_b`, `anchor.s_L` and `ModelTables.v_dec` are different
quantities and are never read as step 2 outcomes.

**Validity.** An outcome is defined only if the network's closed loop reaches the end of its
window alive. This is detected from the float64 base run, before any derivative is taken,
because the harness freezes terminated states. Per agent:
- the specialists lose most keys this way (S1 71%, S2 62%, S3 89%);
- the generalists lose at most 1.8%.
Exclusions are counted per agent and per outcome.

**Near-ties.** A key is flagged for an outcome when that outcome's base trajectory has two strict
local minima (separated by a local maximum) within 0.5 m/s for v_apex, or within 0.05 m for
c_min. Applied to both sides. Flagged keys are reported, and they enter the medians.

## 4. Perturbations

**What is perturbed.** The stimulus bank's 2 m samples of curvature, left width and right width.
Each is perturbed separately, and the inside cap is not re-applied.
- **Network.** Perturbations enter as float64 tangents of `strip_arrays`, added to exact upcasts
  of the float32 bank arrays the agents drove (the T64 recipe, `rollout._upcast`). Pose,
  `kappa_c` and widths therefore stay consistent.
- **Optimum.** Perturbations enter through the same samples into P.

**Impulse responses.** One per sample (feature f, sample s_j), by reverse mode:
- one VJP per scalar outcome on the network side;
- one adjoint KKT solve per scalar outcome on the optimum side.
Every perturbation below is a linear combination of impulse responses, so nothing is
truncated. The support is every sample the closed loop can read before the outcome step:
up to max_k (s_k + clip(6 vx_k, 150, 500)) + spline support.

**Primary perturbation: inside-edge shift.**
- Δw_in(s) = Δ · h(s), where h is a trapezoid: 0 before s_c0 − 40, a linear ramp to 1 at
  s_c0 − 20, 1 through s_c1 + 20, and a ramp back to 0 at s_c1 + 40.
- The **edge-tracking gain** is G = 1 − d c_min / dΔ = σ dn(s*)/dΔ at the argmin s*: how far the
  line moves inward per metre the edge moves inward.
  - The optimum's G* is 1 at a key whose inside contact stays active. This is checked per key
    from the KKT field (gate 4).
  - A width-blind line has G = 0.
  - The network's buffer B = c_min(0) is reported with G. To first order, a narrowing of Δ
    crashes the car when (1 − G) Δ > B.
- The outside edge gets the same trapezoid perturbation. Its field is reported descriptively as
  the line's outward move at the inside contact, dc_min/dΔ_out (the optimum's is about 0). No
  outside-edge gain enters a hypothesis. A true one would need an outside-clearance outcome in
  both pipelines, and step 1's crashes are on the inside.

**Distance-resolved fields (secondary).**
- Gaussian bumps of σ_b = 8 m (also 4 and 16 m), centred at s_1 + d. The grid is d = −20 m to the
  support's end, in 2 m steps.
- Curvature reaches the network by three paths:
  - the κ channel;
  - pose, through all 160 body-frame edge channels (a curvature change turns every later edge
    point);
  - in the closed loop, the dynamics.
- Curvature fields are reported split by path. The parts must sum to the total, to 1e-9.
- Width reaches the network pointwise: the preview edges and the ego lateral channel.

**Finite amplitudes (secondary).** Width dose-response on a stratified subsample of 300–500 keys:
- amplitudes Δ = ±0.1, ±0.3, ±1.0 m, and +2 m inside narrowing;
- the optimum by warm continuation in amplitude, with a restart from the stored key if
  iterations exceed 200 or J is non-monotone;
- the network by direct closed-loop rollouts.

Reported: the linearity ratio, the narrowing/widening asymmetry, and the network's crash
amplitude against B / (1 − G). This step is needed because the optimum's local width derivative
is a fixed-active-set derivative, valid only for millimetre-to-centimetre changes. Real width
changes are 10–100 times larger.

**Turn direction.**
- Curvature is signed toward the turn, steer is flipped for right-hand corners, and widths are
  recoded as inside and outside.
- Because P5 found a learned asymmetry, every measure is reported for left keys, right keys and
  pooled. The asymmetry index is Δm = m_L − m_R over mirror partners (same pair, approach and
  phase).

## 5. The two pipelines

**Network** (float64, its own process; `rollout.py` requires x64 off):
- Built from T64-style exact upcasts:
  - the float32 bank arrays;
  - the actor params and obs_rms of the key's snapshot;
  - `default_ctx()` with hidden_mu = 1.
- **Restart.** From det_x[L], det_prev_action[L] and det_accel[L] for anchored keys, and from x0
  for A2 keys, with slow = 0 (asserted vx > 2 m/s) and t = 0.
- **Anchored-key comparisons start one step earlier (option A).** Both sides start from x_{L-1}:
  - the network applies the agent's a_{L-1} on the perturbed road, then runs its own policy;
  - the optimum's problem is exactly the stored one (x_0 and a_0 fixed).
  - So near-car curvature bumps move x_1 identically on both sides, and frozen
    `AnchoredSolver` re-solves are valid finite differences.
- **The loop.** A fixed-length `lax.scan` of 256 steps, run to s_c1 + 50 m. Stored runs need
  up to 210 (A2) and 180 (anchored finals); H3's units need up to 439, so H3 uses 512. State is
  frozen after termination or the horizon with a safe double-`where`. A run still alive at the
  last step is a pipeline failure, never an exclusion.
- **Derivatives.** One `jax.vjp` of the outcome vector with respect to the three sample arrays,
  about 40 ms per key.

**Optimum** (barrier sensitivity at the stored solution):
- **Re-polish.** The float32 track arrays are not bit-identical across the two machines that
  solved step 1's keys. So every key is re-polished once on the computing machine: one
  REFINE_IPOPT solve on the rebuilt P, warm-started from the stored primal and dual, with the
  same bounds and a_0 fixed.
  - The movement is recorded per step (`dx_step`). A key is flagged (informational) when any
    state moves more than 1e-6 anywhere in its window, or J by more than 1e-9 relative. All
    keys are kept.
  - Result on all 10,006 keys: every re-polish succeeded; 5,384 keys are flagged; J moved by at
    most 1.5e-8 relative. The largest movement anywhere is 2.3 cm, late in the window. Within each
    key's outcome horizon (to s_c1 + 50 m) it is at most 2.5e-4 m (p99 2.2e-5 m; 3,247 keys
    above 1e-6, 2 above 1e-4), about float32 resolution in s.
  - A failed re-polish is a 'numerical' exclusion.
- **KKT system.** The condensed barrier KKT matrix at the re-polished point, keeping every bound
  and inequality:
  - Sigma = z_L/s_L + z_U/s_U;
  - slacks from x and g(x);
  - active-side z from lam_x / lam_g, and hidden-side z = mu/s;
  - mu estimated per key as the median s·z over strongly active constraints (expected 1e-11).
- **Hessian.** `hess_lag` returns the upper triangle, so H = U + Uᵀ − diag(U).
- **Parameter derivatives** F_p, from an uncompiled build (`WindowSolver(spec, compile=False)`):
  - **Curvature:** per step, the step function's derivative with respect to its coefficient
    block, by CasADi AD.
  - **Widths:** both terms. These are the corridor-row block −b_k, and the stationarity term on
    s_{t+1}, −ν_side · b_k′/ds. Omitting the second changes width-bump fields by 0.5–25% on
    real A2 keys (Gate 1's planted bug); the trapezoid shift and G* are unaffected.
- **Adjoint.** K μ = [∇_w y; 0] for each scalar outcome y, then dy/dp = −μᵀ F_p.
- **Exclusion.** By inertia of the condensed matrix, (n_free, m, 0), never by the raw condition
  number (about 1e21 on every key because mu = 1e-11).

**Width-only check.** The optimum's G* needs only the inside contact to stay active. Gate 4
checks it per key from the KKT field, and the dose-response measures how far it holds.

## 6. Measures

All are computed per key, then summarised per **mirror pair** (geometry × approach), step 1's
unit (protocol 9.1):
- median over the pair's keys, with both directions weighted equally;
- a pair is undefined if more than 3 of its 8 keys are excluded;
- P7's 16-phase cells first take the median over phases.

Agents are summarised as the median over geometries, with a **cluster bootstrap** that resamples
geometries (R, θ). Approaches are reported separately and pooled.

| Measure | Definition |
|---|---|
| G, B | Edge-tracking gain and buffer (Section 4); outside-shift field descriptive |
| r_in | Δv_apex,net / Δv_apex,opt for the inside-edge shift (speed response to width) |
| Gain g | Signed projection ⟨net, opt⟩ / ⟨opt, opt⟩ of a distance-resolved field |
| Shape ρ | Cosine similarity of the net and opt fields over d |
| Location | Centroid shift of \|field\| from the optimum's, in metres |
| Extent W90 | Width of the shortest interval holding 90% of \|field\| |
| Reach E90 | The smallest d with 90% of \|field\| mass from −20 m to d. Comparisons are only at the same σ_b, since reach includes about 1.28 σ_b of bump blur |

A near-zero network field counts as a low gain, not as missing data.

## 7. Hypotheses and decisions

Each hypothesis has confirm, refute and inconclusive bands on a 95% cluster-bootstrap CI over
geometries, fixed at the tag. Variant-level claims need 2 of 3 seeds (R18). Seed-0 networks
(G1–G3) are decided like the replicates; nothing in step 2 is "pilot-informed".

### Primary family

**H2′, the edge-tracking deficit.**
- Data: A2 keys; the 9 step-1 generalists; finals; non-capped stimuli; keys with no κ clipping
  over the horizon.
- Measure: the median G.
- Decision per network:
  - **confirm** (deficit) if the CI's upper end is ≤ 0.5;
  - **lawful** if the CI lies within [0.8, 1.2];
  - **over-tracking** if its lower end is ≥ 1.2;
  - **inconclusive** otherwise.
- Variant level: 2 of 3 seeds.
- Prior: the crash ladders suggest a deficit.
- **What a confirm means.** A low inside-edge tracking gain, not by itself width-blindness. A
  width-blind (fixed-n) line predicts G ≈ 0 and T ≈ 0. A line anchored to the outside edge, the
  centre, or a fixed fraction of the road predicts low G with T ≈ 1. T is defined in Section 15.

**H4, the width-training intervention.**
- Networks:
  - v1: 4 real-width seeds (`dr-realw-v1-s0..3`) against baseline seeds (`dr-pilot-200M`,
    `-s1..s4`);
  - v3: 4 real-width seeds (`dr-realw-v3-s0..3`) against baseline seeds
    (`dr-pilot-v3-tilt50a2`, `-s1..s4`).
- Data: the same primary, non-capped A2 keys as H2′, the same states for every network. Each
  variant's seed-0 regime typing (G1 for v1, G3 for v3) gives only a descriptive per-regime
  breakdown.
- Statistic: each network's median G.
- **Decision test.** Real-width seeds against baseline seeds 1–4 (4 vs 4 per variant). It is a
  one-sided exact permutation test, real widths > baseline, with labels exchanged within each
  variant: 4,900 relabellings, minimum p 2.0e-4. Baseline seed 0 is left out of the decision
  because it is the most confounded network: older code (before R18's PPO refactor), trained on
  the Mac CPU.
- Decision:
  - **confirm** if p ≤ 0.05;
  - **refute** if the opposite one-sided p ≤ 0.05;
  - **inconclusive** otherwise.
- **Sensitivity.** The same test with baseline seed 0 included (4 vs 5 per variant, 15,876
  relabellings) is reported. If its decision differs, the H4 decision carries the label "not
  robust to including the seed-0 baselines".
- **Coverage.** H4 is not testable if any of its networks fails the coverage rule (Section 15).
- No field of any of these networks has been computed. Their R-dev completion is known.
- **Known confound** (as in the width test). The real-width runs used later code on a rented
  GPU, at 0.56–0.63M steps/s against the L40S's 0.42–0.45M for baseline seeds 1–4. The width
  option itself is bit-identical when off.

**Multiplicity.** The primary family is H2′ (one decision per network, then per variant) and H4
(one decision). They answer different questions and are reported separately, each under its
own rule, with no correction across them. Six of the 9 step 1 generalists are H4 baselines; the
two hypotheses therefore share networks, and this is stated with the results.

### Secondary

**H1-typical, curvature reading (descriptive since v3.1).**
- Data: A2 keys, in each agent's bulk regime (as P1a). Outcome s_b first, then v_apex.
- **Why descriptive.** The dry run showed the planned absolute bands cannot be calibrated. Step
  1's Qp is a lawful QSS reader of preview curvature, yet against the TO-OCP it has a gain of
  only 0.19–0.37 and ρ of 0.50–0.69. A band of [0.8, 1.25] would call a lawful physics reader
  unlawful.
- **What is reported**, each paired by key against the blind-pedal control (the same network with
  its pedal replaced by its recorded pedal tape):
  - g, location and the shape specificity Δρ (Section 14), against the optimum and against Qp;
  - the unlawful readers (c1, c2, c3) judged against Qp, their base law.

**H1-fast (descriptive since v3.2).**
- Data: the A1 and tier B anchored keys.
- Descriptive. Its sign predictions can be computed from step 1's files before the tag, on the
  same states that suggested them, so they are not prospective. Step 1's direction: G1 and G3
  brake early and are slow after fast approaches. Expected signs:
  - s_b,net − s_b* < 0 (earlier) and v_apex,net − v_apex* < 0 for G1 and G3;
  - about 0 for G2.
- The curvature gain is reported with its CI. There is no equivalence band, for the reason
  that made H1-typical descriptive.

**H2-speed.** r_in, the speed response to an inside-edge shift relative to the optimum's, on A2
keys. It is descriptive of how width moves speed, with bands under / lawful / over at [0.5, 2].

**H3, the crash link (descriptive).**
- The three G2 units are no longer prospective: the reviewers saw their rollouts.
- At each unit:
  - λ* is the smallest λ in w(λ) = w_cw + λ (w_real − w_cw) at which G2 crashes, from rollouts
    on a λ grid of 0.05;
  - λ*_lin = c_min(0) / (−Σ_j S_net(c_min; j) Δw_j), from per-sample impulse responses on left
    and right width separately (all three units are right-hand corners);
  - the clearance change is split into the edge's move and the line's move.
- Reported, not decided.

**H4-transfer (secondary).** H4's statistic on the 8 leave-out networks (`dr-realw-loo-*`, v3
configuration) against v3 baseline seeds 1–4: 495 relabellings, minimum p 0.002. With seed 0
included (1,287 relabellings) it is the sensitivity analysis, labelled as for H4.

**Descriptive.**
- The specialists (most keys invalid, and right-turn dominated).
- Other σ_b values, steer_0, and earlier snapshots.
- The capped stimuli.
- The asymmetry indices, on every measure.
- The anchored-key version of H2′, each agent on its own anchored states. The anchored-key version
  of H4 is dropped (Section 16).

## 8. Controls, nulls and gates (all before the tag; no agent touched)

**Gate 0, KKT residual (every key, no solves).** Stationarity grad f + Jᵀ lam_g + lam_x on the
free variables, with the stored multipliers and the rebuilt P, must be ≤ 1e-6. This pins
CasADi's sign convention (f = −J, L = f + lam_gᵀ g), the track and the context.

**Gate 1, KKT against re-solves.**
- Data: 50 keys, stratified by agent family and by approach (A2 and anchored).
- Directions: all three features, at d ∈ {−10, 0, 10, 40, 100} m, plus the trapezoid shifts.
- Reference: the Richardson-extrapolated central difference of warm frozen-solver re-solves
  (option A, so the problems are identical).
- Error: err = ‖KKT − FD_R‖ / max(‖FD_R‖, floor), with floor = 3 × the Richardson-pair
  discrepancy.
- Pass: median < 1e-3 and 95th percentile < 1e-2.
- Planted bugs must fail it: the dropped width mixed term, swapped left/right parameter columns,
  and a sign flip of lam_g.

**Gate 2, network autodiff against finite differences** (float64).
- A comparison counts when the +h, 0 and −h runs share their discrete structure:
  - the same saturated (step, channel) pairs;
  - the same clipped observation channels;
  - the same argmin step;
  - the same onset run.
- FD(h) and FD(h/2) must agree to O(h²).
- h per key is 0.1 × the smallest kink radius upstream of the outcome, converted with the AD
  tangent.
- Pass: field-vector relative error < 1e-3 on comparisons that count. The share that does not
  count is reported per agent.

**Gate 3, wiring** (teacher-forced, every key).
- At every step from L to the horizon, the float64 actor mean on the stored float32 state must
  match det_a_mean to within 1e-4.
- One float64 control step must match det_x[t+1] within substeps × ULP(s)/2 in s, and within
  calibrated bounds on the other states.
- Planted wiring errors must fail on every gate key: G1's obs_rms on G2, prev_action zeroed,
  accel zeroed.
- **Branch consistency.** The free-running float64 loop matches the float32 file in termination
  step, saturation set and outcome events. It also matches the outcome values within 5 × the
  float32/float64 spread measured in the dry run (about 5e-4 m/s for v_apex). Never compared:
  s step by step, since its drift is unobserved and neutral.

**Gate 4, the optimum's lateral answer.** On each A2 key, the KKT inside-shift field must give
G* ∈ [0.98, 1.02]. Keys outside this are listed and excluded from H2′ and H4's lawful band
reference; they are expected to be rare.

**Gate 5, a cross-pipeline check (KKT replay).**
- On the gate 1 keys, feed a* + ε δa* from the KKT solve into the network-side rollout code.
  This bypasses the action clip, and δa ≈ 0 is asserted where |a*| = 1.
- The resulting outcome fields must match S_opt to a relative L2 of 1e-3.
- Separately, propagate the state tangent through the CasADi step Jacobians, against JAX AD of
  `control_step`, to 1e-8.
- This catches misaligned d axes, start states and outcome definitions between the two pipelines.

**Planted fields (positive controls through the network pipeline).**

| Control | Known answer | Pass |
|---|---|---|
| RF-C1 | A linear readout of preview κ at index k* (near, mid and far), acting on the raw observation in an unclipped state. Its field is the closed form (c/√var) · 50 · κ_at(s_1 + d_k*; strip_arrays(G_d)) | Cosine ≥ 0.999; amplitude within 1%; peak within one bin; computed E90 within one bin of d_k* + 1.28 σ_b at σ_b = 4, 8 and 16 m |
| RF-C4 | Width readout: steer = c1 · o[5k1 + 1] + c2 · o[5k2 + 3], with \|steer\| ≤ 0.5. It has known left and right width fields (c/100) cos(θ_k − ψ) B(d − d_k) | Same as RF-C1; also the turn-direction recoding table (inside/outside on L and R keys) |
| RF-C2 | Q's instantaneous steer field against curvature, through the pose path. Its kernel is K(d) = ∫₀^l (l − u) G(u − d) du with l = max(8 m, 0.6 v), and its gain comes from the kinematic and understeer map | Cosine ≥ 0.999; perturbing `kappa_c` alone gives 0. Q is not used for plan-level known answers (its target is precomputed and its law has a bisection) |
| RF-C5 | Planted edge tracker: a scripted line keeper that holds n = w_in − 1 − B0 through the corner (G = 1 by construction), and one that holds a fixed n (G = 0) | G within 0.02 of 1 and of 0 |

**Nulls and ceilings.**
- **Shape specificity** replaces the cross-geometry band. The band's top was the maximum ρ over
  pairs of different geometry, which is about 0.995, so it could never be passed. The
  specificity is Δρ = ρ(net, own key's optimum) − median ρ(net, optimum at far geometries), on
  turn-recoded fields aligned at s_c0. Positive control: another phase of the same cell gives
  Δρ > 0. Negative control: a far geometry's field gives Δρ < 0.
- **Phase ceiling.** The optimum's fields across the phases of one cell.
- **Unlawful readers through the network pipeline,** on the A2 keys:
  - Qp, which reads preview curvature from the observation;
  - c1 (15 m late);
  - c2 (curvature-saturating);
  - c3 (time-to-corner).
  Each is judged against Qp, its base law, on the criterion it violates:
  - c1 by its s_b offset;
  - c3 by its s_b gain ratio;
  - c2 by its v_apex gain ratio on a capped corner.
- **RF-C3,** a random actor. Used only for the instantaneous steer field, since it does not
  survive to s_c1.

**Coverage.** Exclusions are reported per (agent, snapshot) and per reason:
- invalid outcome;
- Gate 3 failure;
- inertia failure;
- κ clipping.
Above 30% for an agent, that agent is descriptive for the affected outcome.

## 9. What is fixed at the tag

The following are fixed at the tag:
- the key lists, each with its re-polish movement;
- the A2 cell set per regime;
- σ_b, the trapezoid h, and the d grid;
- every tolerance above, including the dry-run-calibrated float32/float64 spreads;
- the bands, and the network lists for H4 and H4-transfer;
- the code hash of `src/apexzero/fields/` and `scripts/fields.py`.

The analysis runs once, as in step 1.

## 10. Order of work

1. `src/apexzero/fields/`, tested against small problems with closed-form sensitivities. This
   covers a 20-step window, the network VJP on a linear policy, and the planted controls.
2. Gates 0–5 and the planted controls. Also the float32/float64 spread calibration on the gate
   keys.
3. Review, then the tag.
4. New solves for H3: the base keys, then the anchored keys.
5. The fields for the A2 keys (all networks in Sections 7 and 8), then the anchored finals.
6. The dose-response subsample.
7. The analysis, once.

## 11. Budget

| Item | Estimate |
|---|---|
| Re-polish | One refine per key, about 1–5 CPU-s, so about 4–12 CPU-h for the 8,475 keys |
| Optimum's fields | Per key: F_p blocks (about 0.1–0.3 s in CasADi's VM, more with hess-related terms), one sparse LU (about 8,800 unknowns), and one adjoint solve per outcome. About 2–5 CPU-s, so about 5–12 CPU-h. A Massed Compute box under the standing permission, with R-dev and synthetic data only |
| Network fields | About 40 ms per VJP. A2: 481 keys × about 35 networks (H2′, H4, H4-transfer, controls). Anchored: 8,136 keys. Under 1 h |
| Gate 1 re-solves | 50 keys × about 17 directions × 4 re-solves at 0.2–6 s, about 2–8 CPU-h |
| Dose-response | 300–500 keys × 7 amplitudes, about 5–10 CPU-h |
| H3 solves | Base keys under 0.5 CPU-h; anchored keys a few CPU-h |

## 12. Then idea 2: the costate

The defect multipliers are the discrete costate. With CasADi's convention and f = −J, the
gradient of the window return with respect to x_1 (moved through the defect's right-hand side)
is dJ*_W/dx_1 = +lam_g[0:8]. The probe target is the gradient of the tail value from x_1:

∇V_1 = (lam_g[0:8] − e_s / 10) / γ

plus the prev-action component 0.04 (a*_1 − a_0). J*_tail is a 399-step window with no terminal
value (γ^400 ≈ 0.135), whereas the critic estimates the policy's infinite-horizon value.
Comparisons with the critic's gradient therefore state that difference.

The plan:
- **Checks.** A ±c re-solve with lbg[0:8] = ubg[0:8] = c gives the finite-difference check, on
  J_tail = (J_W − r_0)/γ, not on J_W.
- **Probes.** Linear and small MLP probes from each hidden layer, with held-out stimulus families.
- **Controls.**
  - the raw observation;
  - a random network;
  - the critic's gradient through `observe` with prev_action and accel held.
- Design and gates come in a separate draft, after step 2's tag.

## 13. What changed from v2, and why (review of 21 Oct)

| Group of findings | Change |
|---|---|
| The anchored keys are fast approaches, none at A2 (critical, 2 lenses) | Section 2 rewritten. A2 phase keys become the primary state set; H1 is split into typical and fast |
| Width hypotheses read speed, but the crashes are lateral (critical) | Lateral outcome c_min and edge-tracking gain G become primary (H2′) |
| The float64 reproduction gate was impossible (critical, 3 lenses) | Gate 3: a teacher-forced wiring check plus branch consistency, with tolerances calibrated in the dry run |
| A prospective interventional test was missing | H4 on the real-width networks, A2 states shared by all networks |
| x_1 handling between the KKT system and re-solves | Option A: both sides start from x_{L-1} |
| Width mixed term; Hessian triangle; compiled functions lack p-derivatives; condition number useless; pedal-bound dropping unneeded | Section 5 (optimum) rewritten |
| Rebuilt P differs across machines | Re-polish with recorded movement |
| s_b, s_dec, v_apex definitions and name clashes; near-tie flag | Step 1's definitions imported; new names; tie rule rewritten |
| Gate tolerances 100–10,000 × too loose; no cross-pipeline check; no width control; RF-C2 invalid; RF-C3 null undefined; E90 ambiguous | Gates 0–5, RF-C4, RF-C5, rewritten RF-C1 and RF-C2, structured nulls, E90 and W90 defined |
| Unit of analysis, checkpoint keys pooled, no decision structure, H1-tail direction | Mirror-pair unit and cluster bootstrap; finals only; primary family and bands; H1-fast from step 1's direction |
| Bump fields mis-scale H3's first-order prediction; local width derivative valid only for cm | Impulse-response sums; finite-amplitude dose-response |
| Specialists mostly invalid; κ clipping confounds agents; learned asymmetry | Outcome validity rule; non-clipped confirmatory sets; per-direction reporting |
| Costate target | Section 12 corrected (tail gradient, prev-action component, horizon) |

Refuted by both verifiers: "dropping the pedal-state bounds leaves the stored multipliers
non-stationary" and "mirror pooling cancels every curvature field".

## 14. Implementation decisions (v3.1; fixed at the tag)

Code: `src/apexzero/fields/` and `scripts/fields.py`, tested in `tests/test_fields_*.py` (124 tests
at v3.1; Section 16 gives the count at the tag).
Each decision below is implemented and tested.

**Conventions**
- Bumps have unit peak, G_d(s) = exp(−((s − s_1 − d)/σ_b)²/2).
- Steer fields are in the raw sign until the measures stage recodes them toward the turn.
- **Validity** is stricter than step 1's `reached`: an outcome needs the run alive at some
  s ≥ s_c1 + 50, and s_b also needs an onset before s_apex.
- The optimum's c_min includes the corridor's 3 cm safety margin, so it is 0.03 at the contact.
  The network's c_min is measured against the termination boundary. So B carries a 3 cm offset
  between the two sides; G, a derivative, does not.
- **G's plateau rule.** G = σ dn/dΔ holds only where the trapezoid is flat. A key whose c_min
  argmin has e(s*) < 0.99 is excluded as 'argmin off plateau'. On all 481 A2 optimum keys,
  e(s*) ≥ 0.9999999999.
- **Near-tie flags** follow the literal rule of Section 3. On the optimum they flag c_min on about
  26% of A2 keys, mostly 1e-9-level minima along an active contact arc. This is immaterial
  there, because the optimum's c_min field is zero. Flags are reported and keys kept.
- **κ clipping** is decided per outcome, over the observations that outcome reads. Step 1's
  "before s_c1" window is also recorded. Raw-observation policies never count as clipped.

**Optimum**
- The fixed-active-set map has kinks, so LICQ fails structurally wherever full throttle holds:
  the pedal-state bound is implied by the pedal-command bound. The barrier K stays nonsingular.
- Inertia is certified: rounding-level pivots are recounted from eigenvalues with
  |λ| > n·eps·‖M‖. On all 481 A2 keys no pivot was at rounding level (462 by LDL, 19 by
  Cholesky).
- The adjoint residual must be ≤ 1e-10. On all 481 A2 keys it was at most 1.6e-13, with no
  barrier fallback.

**Gate 1**
- **Directions.** Gates 1, 2 and 5 share one direction set (`perturb.gate_directions`), 26 per
  key:
  - each feature near the car (s_1 − 10, s_1, s_1 + 10) and at the corner (s_c0 − 20, s_c0,
    s_apex, s_c1, s_c1 + 20);
  - plus both trapezoids;
  - near and corner bumps are separate vectors.
  - Every stratum must have scored corner vectors of v_apex and s_b, for curvature and the
    inside width.
- **Re-solve tolerance.** Re-solves use the key solver's own frozen functions and REFINE_IPOPT
  options with tol = 1e-10 (`fd_solver`). The frozen refine tolerance, 1e-7, stops after 0
  iterations on small perturbations.
- **Steps and zero fields.**
  - FD steps per unit-peak direction are 1e-6 for curvature and 1e-4 for widths. A width step of
    1e-3 crosses mm-scale active-set kinks.
  - A vector that both KKT and FD put below 1e-6 is a zero field. It is left out of the pool and
    counted.
  - c_min is scored against its explicit edge term. c_line, the line's own move with the edge
    held, is a separate quantity.
  - The rule applies pooled and within each quantity.
- **Strata.** 50 keys (A2/W400 14, A2/W800 6, v1/v2/v3 anchored 8 each, specialists 6), drawn on
  the seed stream `gate1_sample`.
- **Result** (pre-tag): 1,121 scored vectors and 879 zero fields.
  - Median error 1.4e-7, p95 1.8e-4, max 0.055. The tail is re-solve resolution at
    weakly-active kinks.
  - Planted bugs fail: drop_width_mixed p95 0.38, flip_lam_g median 0.77, swap_widths
    median 1.0.

**Gate 2**
- FD steps are capped at 1e-5 per unit κ and 1e-3 per unit width, with RICH_RTOL 1e-4.
- The zero-field rule applies.
- A valid outcome with no counted comparison fails its row.

**Gate 3 and branch consistency**
- Teacher-forced on every key with a det row.
- STEP_TOL and ACCEL_TOL are about 10× the spreads seen.
- BRANCH_TOL is derived before the tag without running any trained network's free loop
  (`network.branch_tolerance`). Per outcome, it is 5 × the largest float32/float64 outcome
  spread per unit of teacher-forced action error, measured on the controls, times the trained
  agents' largest teacher-forced action error over every key (Gate 3 on all agents).
- Planted wiring errors (wrong obs_rms, prev_action zeroed, accel zeroed) must fail on trained
  agents' rows, where they fail by more than 1000×.

**Gates 4 and 5**
- Gate 4: on all 481 A2 keys, |G* − 1| ≤ 1.7e-9.
- Gate 5 is compared at the optimum's picks:
  - the replay's own plan values must agree to 1e-8;
  - a different pick counts as a tie;
  - δa* ≈ 0 is checked at active bounds to 1e-3 of max |δa*|;
  - the env's action clip is kept.
- The decision steer is reported as a decision check (identity 1e-15), not as cross-pipeline
  evidence.
- Result on 50 keys: v_apex at most 2.2e-10, s_b 1.1e-13, c_line 6.2e-9. A planted 10 m road
  misalignment fails on all 50.

**Controls**
- **AD-safe Q family.** Step 1's Q family is not AD-correct: sign(κ)·us(|κ|) has zero derivative
  at κ = 0, the bisection in `law_a` has none, and `grip_profile` gives NaN. `fields/controls.py`
  holds value-identical AD-safe replacements; the Qp adapter reproduces the frozen Qp bit for
  bit.
- RF-C5's keeper corners at 6 m/s². It reaches s_c1 + 50 within the loop only on the 45° A2
  cells 340, 342, 344 and 345.
- The blind-pedal control reproduces its base run bit for bit.
- **Phase ceiling** (median ρ over 113 cells): s_b 0.984, v_apex 0.978, steer 0.993. v_apex's
  minimum is −0.91, so on some cells the optimum's own v_apex field changes shape across phases.

**Statistics**
- Weighted median: the mean of the lower and upper weighted medians.
- P7 cells: the median over phases within each direction, then the mean of the two directions.
- The H1 location band is 12 m.
- H1-fast's "about 0" for G2 means the CI contains 0 (descriptive).

**Freeze**
- `fields/prereg.py` writes the record:
  - the key lists with content hashes;
  - each key's re-polish movement;
  - the design hash, the design-document hash and the code hash.
- After `receptive-fields-prereg-v1` (and `psychophysics-prereg-v2`) both exist:
  - nothing under `src/apexzero` or `scripts/fields.py` may change;
  - docs-only commits are allowed;
  - the record must match the disk.
- `--post-tag-deviation` logs a deviation instead of refusing.

**Budget updates**
- VJP: 75–85 ms per key.
- Gate 1: 104 re-solves per key, 133 s for 50 keys with 5 workers.
- Re-polish of all keys: about 15 min with 4 workers.

## 15. Decisions after the final pre-tag review (v3.2)

A final review ran on four lenses: design-to-code concordance, pre-registration integrity,
inference validity, and the gate outputs. 46 findings survived two independent refuters, 34 of
them marked essential. These decisions resolve them. Every one is implemented and tested before
the tag.

**Data and exclusions**
- **Pair completeness.** A pair's expected keys are its keys in the frozen A2 list, not the
  cell's phase count. Eleven non-capped pairs have only their two u = 0 nominal keys, and with
  this rule they enter as two-key pairs.
  - A pair is undefined when more than 3/8 of its listed keys are excluded.
  - The two directions are weighted equally. A direction with no valid key leaves the pair to
    the other direction only.
- **Coverage rule, applied.** A network whose share of excluded primary non-capped A2 keys
  exceeds 30%, counting every reason, is descriptive for H2′. H4 and H4-transfer are not
  testable if any of their networks fails the rule.
- **Pipeline failures count as exclusions and are listed per network:**
  - a missing optimum row;
  - a branch-consistency failure;
  - a loop exhausted at one key. This fails that key only; it never aborts the network's stage.
- **Capped stimuli** are excluded from H2-speed and H1-fast as well as from H2′ and H4.

**Measures**
- **T = G + G_out**, the line's move toward the inside when the whole corridor shifts toward the
  inside by Δ: the inside widens by Δ and the outside narrows by Δ. Here G_out = −σ dn/dΔ_out
  is the line's outward move per metre the outside widens. Every corridor-anchored
  line has T = 1, a width-blind line T = 0, and the optimum T* ≈ 1. T is descriptive and is
  reported beside G for every network.
  - Descriptive labels: 'width-blind' if T's CI upper end is ≤ 0.5; 'corridor-anchored' if
    the CI lies within [0.8, 1.2]; 'mixed' otherwise.
- H1-typical's g and location are computed on the s_c0-aligned grid d ∈ [−150, 400] m
  (descriptive).

**Checks on the trained networks after the tag** (pre-registered here; they run before any
analysis)
- **gate2-trained.** Autodiff against finite differences on every H2′, H4 and H4-transfer
  network, on 12 primary non-capped A2 keys drawn on the stream `gate2_trained` and listed in the
  record.
  - Directions: both trapezoids, plus the corner bumps of κ and w_in.
  - Rule: Gate 2's.
  - A network that fails is descriptive.
- **Wiring for networks without step 1 rollouts.** For each such network, its float32 harness run
  on every A2 key is compared with its float64 loop for branch consistency. A failing key is a
  pipeline exclusion.
- **A pre-tag control** puts the trained architecture inside the loop: the linear driver plus a
  small-gain random-init actor reading G1's obs_rms, on cells 344 and 66. It runs through Gate 2,
  and its rows join the branch-tolerance calibration.
- **Dose-response networks** are fixed now: the 9 step 1 generalists and the 18 H4 networks, on
  the subsample drawn on the stream `dose_sample`.

**Freeze and run-once**
- **The record** (`prereg.json`) binds:
  - the key lists;
  - the per-regime A2 cell sets for every agent and variant, with the hash of step 1's
    `support.json`;
  - the gate2-trained and dose key lists;
  - every network's checkpoint path and sha256. Post-tag stages load the recorded path and
    check its hash;
  - the stamped, passing outputs of keys/Gate 0, re-polish, Gates 1–5, controls, calibration and
    opt-fields, with the hash and exact key coverage of opt-fields' rows and arrays;
  - BRANCH_TOL, which must equal the calibrated value.
- **Provenance stamp.** Every stage output carries the code hash (taken at start and before
  writing), HEAD, the changed paths, argv and times.
- **The tag** must be annotated and present on origin with the same object. The first post-tag
  run writes `tag.json` once, and later runs compare against it.
- **Hashes against the tagged copy.** The record is built only from a clean tree, and its
  design-document hash is checked against the document's blob at the tag.
- **Pre-tag stages** refuse after the tag, unless a deviation is logged.
- **Deviations.** `--post-tag-deviation` can waive only changed code or a changed design
  document. It records the diff's hash, HEAD and the tag's commit in the deviation log and in
  every output's stamp. It never waives:
  - a missing tag or record;
  - changed key sets;
  - changed checkpoints;
  - changed step 1 code.
- **Run-once.**
  - The confirmatory analysis refuses unless every required network × frozen key has a row, and
    every key has an optimum row.
  - It writes its decisions once, to a fixed path. Its sha256 is then committed to the findings.
  - Before the confirmatory run, a non-confirmatory analysis computes coverage and plumbing only,
    never the primary decisions.

## 16. Implementation of Section 15, and the last decisions before the tag (v3.3)

Every Section 15 item is implemented and tested. A three-lens review of the implementation
(concordance, freeze integrity, tests) confirmed 28 findings, rejected none, and all are fixed.
The suite (`tests/test_fields_*.py`, with step 1's rollout tests and the dashboard's) passes in
one session.

**Decisions**
- **The anchored-key version of H4 is dropped.** It would run all 18 H4 networks from every
  agent's anchored states: off-policy for most of them, about 146,000 VJPs, and not in the
  Section 11 budget. Its stage exists in the code but is not part of the pre-registration and is
  not run. The anchored-key version of H2′, each agent on its own states, stays descriptive.
- **gate2-trained is strict.** A network passes only if every one of its 12 keys passes Gate 2's
  rule. A key where the trained loop runs out of steps fails the network. No stored run comes
  within 46 steps of the 256-step loop.
- **H1-fast's g and steer_0** are computed on the s_1-anchored grid, to each outcome's support
  end. H1-typical stays on the s_c0-aligned grid (Section 15).

**Pass rules for complete stages.** Gate 4, re-polish and opt-fields pass when they cover every
key. Out-of-band, inertia, adjoint and failed-re-polish keys are listed exclusions, as Sections 5
and 8 say. Only a crash fails the stage.

**Freeze scope**
- All of `src/` and `scripts/fields.py` are frozen; `scripts/_bootstrap.py` is too.
- Package versions are recorded and compared. A difference is a waivable problem.
- Every stage output carries a stamp, and the per-key field arrays that the confirmatory analysis
  reads are hashed in its input manifest.
- The record is bound to the code in the tagged commit, not only to the working tree.

**Post-tag procedure** (enforced by the code)
1. The first post-tag stage (`harness-f32`) writes `outputs/fields/tag.json`. It is committed at
   once (`git add -f`), or every later post-tag stage refuses.
2. Order: harness-f32, gate2-trained, h3-keys, fields-a2, fields-anchored, fields-earlier, dose,
   h3-lambda, analyse (coverage and plumbing only), then `analyse --confirmatory`.
3. The confirmatory `decisions.json` is committed at once (`git add -f`), and its sha256 goes into
   the findings. Until that commit, a non-confirmatory analysis gives coverage and plumbing only;
   after it, a second confirmatory run is refused.
- `qp-fields` computes Qp's fields on the A2 keys before the tag. They are the reference for the
  descriptive H1-typical and for judging the unlawful readers.
- Descriptive definitions fixed in code: the dose linearity and asymmetry ratios, the asymmetry
  index on every measure, and the near-tie flags of Section 3.

