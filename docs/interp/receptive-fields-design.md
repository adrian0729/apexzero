# Step 2: receptive fields (design draft v3, not frozen)

Status: draft, 21 Oct 2026. v1 had one code review, and v2 a five-lens adversarial review in
which 57 of 59 findings survived independent verification. v3 adopts them; Section 13 maps each
group of findings to its change. Nothing here has been computed on an agent.

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
- Widths are 6/6 m and constant on 52 of the 60 keyed stimuli (W 12).
- The 8 stimuli at R 6.0 and 7.5 m (`cap_binds`) carry the generator's inside cap, which narrows
  the inside to 4.60–5.59 m over 1–7 apex knots. They are a separate stratum, and descriptive.

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
- The outside edge gets the same trapezoid perturbation, giving G_out.

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
- **The loop.** A fixed-length `lax.scan` of 150 steps (horizons are 72–142), run to
  s_c1 + 50 m. State is frozen after termination or the horizon with a safe double-`where`.
- **Derivatives.** One `jax.vjp` of the outcome vector with respect to the three sample arrays,
  about 40 ms per key.

**Optimum** (barrier sensitivity at the stored solution):
- **Re-polish.** The float32 track arrays are not bit-identical across the two machines that
  solved step 1's keys. So every key is re-polished once on the computing machine: one
  REFINE_IPOPT solve on the rebuilt P, warm-started from the stored primal and dual, with the
  same bounds and a_0 fixed. The movement is recorded (states and J). A key is flagged if any
  state moves more than 1e-6 or J changes by more than 1e-9 relative. All keys are kept.
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
    s_{t+1}, −ν_side · b_k′/ds. Omitting the second biases the width fields by about 4–5%.
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
| G, G_out, B | Edge-tracking gains and buffer (Section 4) |
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
- Prior: the crash ladders suggest a deficit. The width-blind reading predicts G ≈ 0.

**H4, the width-training intervention.**
- Networks:
  - v1: 4 real-width seeds (`dr-realw-v1-s0..3`) against 5 baseline seeds (`dr-pilot-200M`,
    `-s1..s4`);
  - v3: 4 real-width seeds (`dr-realw-v3-s0..3`) against 5 baseline seeds
    (`dr-pilot-v3-tilt50a2`, `-s1..s4`).
- Data: A2 keys, the same states for every network. The regime is each variant's frozen seed-0
  regime.
- Statistic: each network's median G. One-sided exact permutation test, real widths > baseline,
  with labels exchanged within each variant (15,876 relabellings).
- Decision:
  - **confirm** if p ≤ 0.05;
  - **refute** if the opposite one-sided p ≤ 0.05;
  - **inconclusive** otherwise.
- No field of any of these networks has been computed. Their R-dev completion is known.
- **Known confound** (as in the width test): the two arms differ in more than the width option.
  The real-width runs used later code on a rented GPU. Baseline seed 0 of each variant ran on the
  Mac CPU before R18's PPO refactor. A sensitivity analysis without the seed-0 baselines is
  pre-registered (70 relabellings per variant).

### Secondary

**H1-typical, lawful curvature reading.**
- Data: A2 keys, in each agent's bulk regime (as P1a). Outcome s_b first, then v_apex.
- Pass requires all of the following:
  - gain g in [0.8, 1.25];
  - location within ± the local preview spacing (12–14 m at the corner);
  - ρ above the cross-geometry null band (Section 8).
- Each criterion is also paired by key against the **blind-pedal control**: the same network
  with its pedal replaced by its recorded pedal tape. The criterion must exceed the control's
  CI.

**H1-fast.**
- Data: the A1 and tier B anchored keys.
- Pre-registered from step 1's direction: G1 and G3 brake early and are slow after fast
  approaches. Predicted signs:
  - s_b,net − s_b* < 0 (earlier) and v_apex,net − v_apex* < 0 for G1 and G3;
  - about 0 for G2.
- The curvature gain is tested as two-sided equivalence to [0.8, 1.25].

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

**H4-transfer.** H4's statistic on the 8 leave-out networks (`dr-realw-loo-*`, v3 configuration)
against the v3 baselines. Secondary.

**Descriptive.**
- The specialists (most keys invalid, and right-turn dominated).
- Other σ_b values, steer_0, and earlier snapshots.
- The capped stimuli.
- The asymmetry indices, and the anchored-key versions of H2′ and H4. For H4 these use pooled
  states, off-policy for most networks.

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
- **Cross-geometry null.** ρ, g and location between the optimum's fields at keys of different
  geometry, aligned at s_c0. This is how much any corner-shaped field earns.
- **Phase ceiling.** The optimum's fields across the phases of one cell.
- **Unlawful readers through the network pipeline,** on the A2 keys:
  - Qp, which reads preview curvature from the observation;
  - c1 (15 m late);
  - c2 (curvature-saturating);
  - c3 (time-to-corner).
  Each must land outside H1-typical's pass band on the criterion it violates.
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
