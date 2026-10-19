# Network psychophysics, step 1: pre-registered protocol

**Version:** v2 (final), 2026-10-19. This version replaces the v1 draft. Section 11 lists every review issue and how it was handled.

**Freeze rule.**
- All agent-free work happens before the tag:
  - training-support statistics;
  - stimuli;
  - every answer key that does not depend on an agent, and its validation;
  - the scripted positive controls;
  - the dry run of the analysis on the real key design.
- Then this file is committed and tagged `psychophysics-prereg-v2`, together with the code hash, the stimulus-set hash and the key-cache hash.
- No agent is rolled out on any stimulus of this protocol before the tag. There are two exceptions, and both use only pilot-informed agents:
  - the pilot replication (G1 and G2 on the v1 pilot mini-grid) in step 2;
  - S1's real Hockenheim laps that build the memoriser buffer in step 5.
- Keys that depend on an agent's own states (re-anchored and viability keys) are solved after the tag, by frozen code.
- Any change after the tag is recorded, with its reason, in a dated "Deviations" log in `docs/findings/psychophysics-step1.md`.

**Lockbox.** No agent or optimum is ever evaluated on an `apexzero.tracks.corners.R_TEST` circuit. Every stimulus is either synthetic or cut from an R_DEV track. Every entry point that names a track calls `apexzero.viz.export.lockbox_guard(name)` before loading anything.

## 0. Scope

**In scope:**
- Closed-loop responses of trained PPO policies to single-corner stimuli. These are synthetic corners, or corners cut from R_DEV tracks with their context removed step by step.
- Comparison with optima solved from the same initial state:
  - the training-objective OCP (TO-OCP);
  - a version of it with backed-off limits;
  - the s-domain minimum-time OCP (s-OCP).
- Scripted positive controls run through the same harness.
- Open-loop probes, and on-policy probes that shift the corner by part of a preview spacing.

**Out of scope:**
- Synthetic two-corner stimuli. These are step 1b. The ablation ladder in tier E already attributes each real crash to the corner itself or to its context.
- Jacobians, counterfactual receptive-field edits and KKT sensitivities. These are step 2.
- Any R_TEST circuit.
- Varying the car context for the generalists.
- Observation noise.

## 1. What was known before the tag, and fixed choices

### 1.1 Pilot results

I saw these before writing the protocol. They shaped the predictions and are not reused as confirmatory data.

| Pilot | Result |
|---|---|
| Mini-grid: width 11; R ∈ {9, 11, 13, 16, 20, 30}; 90° and 180°; both directions; 80 m/s from 700 m out; 10 m ramps | v1 completed 24/24 cells. v2 crashed only R9/180/right (corridor exit at 68 km/h). |
| Gentle corners (R100/90°, R50, R25) and R13 hairpins from 80 m/s | v1 and v2 completed all of them. |
| S1 on the mini-grid and on width-12 corners | Crashed 22/24 mini-grid cells, many on the straight at 223–287 km/h. Also crashed R50, R25 and the R13 hairpins at width 12. |
| S1 on transplants of its own Hockenheim corners | Matched the real lap to \|Δv\| ≤ 0.08 m/s. |
| v2 final on transplants of its own crash corners | Reproduced the Norisring and Zandvoort crashes. YasMarina did not reproduce. |
| Answer key on R13/180, R40/90 and R100/60 | 7 solves, all replay-verified. |

### 1.2 Facts established by the two reviews (also seen before the tag)

| Item | Value |
|---|---|
| Radius below which the κ·50 input clips at \|o\| = 10, by agent and preview point | S1 5.2–9.4 m, G1 6.4–8.9 m, G2 5.7–7.6 m. Near points clip at smaller R than far points. |
| Final log_std (steer, pedal) | S1 (−3.72, −2.94); G1 (−3.13, −2.45); G2 (−3.17, −2.54). So σ_steer is 0.024–0.044 and σ_pedal 0.053–0.086. Initial log_std is −0.5. |
| Training settings | obs_noise 0. Hidden grip ±2%. Car context randomised for the generalists. Resets start at 60–100% of v_ref. |
| Real crash corners by regime | G1: 1 below support (9.2 m), 3 tail (13.2, 18.3, 19.3), 4 bulk (21.1, 24.4, 33.6, 47.2). G2: 5 tail (12.1–14.0), 2 bulk (21.1, 47.2). 13 distinct corners in all. |
| Ramp lengths in training | The median ramp is 26, 33, 40 and 44 m in the bins R < 11.7, 11.7–18.3, 18.3–28.6 and 28.6–55.9 m. 92% and 68% of corners in the two tightest bins are triangles. Onset rate (peak κ ÷ distance to peak) for R_eff < 20 is 0.0028–0.0031 m⁻²; an R13 corner with a 10 m ramp has 0.0077. |
| Peak curvature of training corners | Realised R_eff q1/q10 is 11.5/21.2 m (base grammar). |
| Measured key cost (R13/180, A1) | s-OCP 37 CPU-s; one direct 400-step TO-OCP 68 CPU-s, 1,663 iterations. |
| Active limits of the R13/180 key at A1 | Slip limit: \|α_r\| reaches 10.03°. At v_min (47.7 km/h) steering is 22.4° of the 25.8° lock and lateral acceleration is 16.4 m/s². Minimum kinematic turning radius is 7.45 m. |
| Pedal | The pedal that saturates the front axle in straight braking is 0.70 at 310 km/h, 0.46 at 200, 0.30 at 100 and 0.26 at 60. The key never commands −1. |
| Straight-line braking exponent β* | 0.70, 0.72, 0.75 and 0.81 for v_e* of 50, 70, 90 and 120 km/h. A time-to-corner rule gives β/β* of 0.40–0.67. |
| Car limits | Top speed 315.0 km/h. Lateral limit relative to QSS falls from 0.99 at 40 km/h to 0.87 at 300 km/h. Braking at 310 km/h: 65.4 m/s² against QSS's 72.5. |

### 1.3 Prospective tests

These snapshots did not exist when the predictions were written, or have not been evaluated. The manifest records that none of them was rolled out on any stimulus before the tag.
- G3 = `dr-pilot-v3-tilt50a2`: queued, ETA about 06:24 on 19 Oct.
- S3: training, 42% at 03:01, ETA about 03:39.
- S2 snapshots after 122.3M steps, which here means its final at 199.9M.

### 1.4 Fixed choices (cannot change after the tag)

| Item | Value |
|---|---|
| Knot spacing | 2.0 m |
| Strip | N = 2560 knots, so L = 5118 m. Truncation at s > 4598 m. |
| Corner start (start of ramp in) | S_C = 1000.0 m |
| Ramp, core tiers | Training-typical: l_r = max(10, min(l_typ(R), θR)) (Section 3.1). |
| Ramp, abrupt factor | l_r = 10 m, tier E3 only |
| l_typ(R) | Median generator ramp (`l_r`) of corners in each R_seg bin, from 300 base-grammar strips, frozen in step 1. Interpolated linearly in log R through bin medians placed at R = 9.4, 14.6, 22.9, 40 and 75 m; constant outside. Expected 26, 33, 40, 44 m, and ≥ 44 m at 75. |
| Inside-width cap | The generator's cap at every knot: w_in(s) = min(w_in, 1 + 0.6/\|κ(s)\|) |
| Reference line | The centreline. `refine_reference` is never called. |
| Car | `racing.default_context()` as exact float32; hidden_mu = 1.0; obs_noise = 0 |
| Initial auxiliaries | Synthetic stimuli: prev_action = 0, accel = 0. Transplants: the real lap's values. |
| Phases | K = 4: u_k = (k/4)·v_sight·0.05 s for k = 0…3 (Section 4) |
| Matched-noise condition | σ = (0.04 steer, 0.08 pedal) for every agent |
| Backed-off key | α_max = 8.5° instead of 10°; corridor narrowed by 0.35 m on each side |
| Speed band b | Set by the V8 rule (Section 6.5). Clipped to [5%, 10%]; expected about 7.5%. |
| Root seed | 20261019 |
| Single radius estimator | R_seg := 1/max\|κ_smooth\| from `corners.segment` with its defaults. Used for stimuli, training strips, R_DEV corners and crash corners alike. |

## 2. Questions, notation, regimes and predictions

### 2.1 Questions

| Question | Predictions |
|---|---|
| Q1. Is corner severity read lawfully? | P1, P3 |
| Q2. Where and how does the reading depart from physics? | P1, P2, P5 |
| Q3. Does the specialist read the road, or has it memorised Hockenheim? | P4 |
| Q4. Why do the generalists crash in tight corners? | P1-tail, P2, P8 |
| Q5. When during training does each part of the response become correct? | P6 |
| Q6. Does the preview's sparse sampling limit the reading of short, tight corners? | P7 |

### 2.2 Notation, keys and the speed band

**Keys.** These are the optima the agent is compared with (full definitions in Section 6):
- `*` is the nominal TO-OCP key.
- `*bo` is the backed-off TO-OCP key.
- `*RA` is a TO-OCP key re-anchored at the agent's own state.

**Errors:**
- e_ve = v_e/v_e* − 1 and e_ve^bo = v_e/v_e*bo − 1, where v_e = vx(s_c0) is the entry speed.
- At A1, and on tier B free approaches (D = 600 m), e_ve uses the re-anchored key: e_ve^RA = v_e/v_e*RA − 1. The backed-off side is then scaled by the cell's ratio v_e*bo/v_e*.

**Band:**
- A pair is *in band* if e_ve ≤ +b **and** e_ve^bo ≥ −b.
- *Overspeed* means e_ve > +b, judged against the nominal key.
- *Underspeed* or *caution* means e_ve^bo < −b, judged against the backed-off key.
- Expected bias of a lawful agent: about −1%, from robust targeting under the ±2% hidden grip in training. This lies inside the band.

**Sensitivity S.** S is the slope of log v_e on log v_e*, with fixed effects for θ.
- It is *estimable* only if the pooled within-θ range of log v_e* is at least 0.15.
- Otherwise it is reported as "not estimable" and no decision uses it.

**Extrapolation residual.** r = log v_e − (a_θ + S_bulk·log v_e*). It is computed from the agent's own bulk fit at the same approach.

**Onset error.** Δd_b = d_b − d_b*, using the brake-demand onset (Section 5). A negative value means the agent brakes later than the key. Phase-matched keys are used where they exist (Section 6.1).

**Severity axes.** Every curve is drawn against two x-axes:
- R_seg;
- the key's own severity, v_min*.

### 2.3 Regimes

Regimes are assigned per agent and per stimulus, from the stimulus's R_seg. Every boundary is recomputed in step 1, with the single estimator, from 300 generated strips of each agent's own training grammar, and frozen into the manifest. The values below are what is expected.

| Agent | q1 / q10 of R_seg | Clip radius R_clip^max | Clip-saturated | Below support, not clipped | Tail (q1 ≤ R < q10) | Bulk (R ≥ q10, braking per key) |
|---|---|---|---|---|---|---|
| G1 (v1, base grammar) | 11.4 / 20.2 | 8.9 | 6.0, 7.5, 8.4 | 9.4, 10.5 | 11.7–18.3 | 22.9–87.3 |
| G2 (v2, 70/30 mix) | 10.4 / 16.4 | 7.6 | 6.0, 7.5 | 8.4, 9.4 | 10.5–14.6 | 16.4–87.3 |
| G3 (v3, 50/50, α = 2) | 9.4 / 13.4 | From G3's obs_rms | ≤ R_clip^max | Non-clipped values < 9.4 | 9.4–13.1 | 14.6–87.3 |

**Clipping.**
- R_clip(k) = 50/(\|μ_k\| + 10σ_k) is computed for every preview point k from the agent's obs_rms, using the worse sign. R_clip^max is its maximum over k.
- A pair is *κ-clipped* if any of its deterministic runs drives any preview κ channel to \|o\| ≥ 10 before s_c1. This run-time definition is the binding one; the table above is design guidance.
- Reading obs_rms is not an evaluation of the agent. For G3 and S3 it is done when their final snapshot enters the manifest.

**Braking or flat.** A stimulus counts as a braking stimulus at a given approach only if its key has a sustained brake (Section 5) and v_min* < v0 − τ. Large-R cells are therefore classified by the key, not by their nominal R.

**Joint support covariate.** Every stimulus gets a joint training-support density:
- a kNN density (k = 10) over (log R_seg, θ, l_r, L_c, log v_peak^QSS);
- each feature standardised by the training sample's standard deviation;
- the same estimator applied to training corners and to stimuli, with v_peak taken from the QSS profile for both.

### 2.4 Predictions

Section 9.1 defines the unit of analysis (the mirror pair) and the pair values used throughout.

#### P1 – Lawful reading in support; departure at the tail after a fast approach, and below support (G1, G2, G3)

**Data.**
- Tier A, final checkpoint.
- Deterministic pair values decide every criterion about speed or completion.
- Native-noise stochastic runs give crash rates.

**P1a – Bulk tracking at the typical approach (A2).**

Over bulk braking pairs at A2:
- *Confirm* if all of these hold:
  1. at least 80% of pairs are in band;
  2. S_bulk ∈ [0.8, 1.2], and every leave-one-R-out S_bulk ∈ [0.75, 1.25] (S must be estimable);
  3. deterministic crashes occur in at most 1 pair;
  4. the 99% upper bound of the native-noise bulk P_crash is at most 5% (beta-binomial; Section 9.2).
- *Refute* if any of these holds:
  - at least 50% of pairs are out of band on the same side;
  - S_bulk and all of its leave-one-out values lie outside [0.8, 1.2];
  - deterministic crashes occur in at least 2 pairs.
- Otherwise the result is inconclusive.
- A refutation caused only by underspeed, with S_bulk ∈ [0.8, 1.2], is labelled **"cautious but lawful"**.

**P1-tail – Tracks at the typical approach, overspeeds after a fast one.**

| Part | Criterion |
|---|---|
| A2 | P1a criteria 1, 3 and 4 applied to tail pairs. S is reported if estimable. |
| A1 | At least 60% of tail pairs overspeed, or have a deterministic crash classified as overspeed. **And** the within-geometry contrast e_ve^RA(A1) − e_ve(A2) is positive in at least 75% of tail geometries. |

- *Confirm* if both parts hold.
- *Refute* if the A1 tail pairs meet the tracking criteria: at least 80% in band, and deterministic crashes in at most 1 pair.
- Otherwise the result is inconclusive, and the A2 part is reported on its own.

*Rationale.* A1 is a support manipulation. Training contains 1.8, 2.4 and 4.1 tight corners per 100 km reached at 250 km/h or more (v1, v2, v3), against 20 per 100 km on R_DEV.

*Known tension.* v1 completed all 24 fast-approach pilot cells. Completing is compatible with overspeed within the car's margin; P1-tail tests speed, not completion.

**P1b – Departure below support (non-clipped pairs only).**

Below-support, non-clipped pairs at A1 and A2 are used. If there are fewer than 4 such pairs, the result is reported as "not testable".
- *Confirm* if both hold:
  - at least 75% of pairs have r > log(1 + b) or an overspeed crash;
  - Kendall τ(r, R_seg) ≤ −0.3 over the tail and below-support pairs at that approach, meaning the departure grows as R falls.
- *Refute* if at least 75% of pairs have \|r\| ≤ log(1 + b) and none has a deterministic crash.
- If P1a is refuted on its slope, r is replaced by log(1 + e_ve), because a residual from an unlawful fit means nothing.
- Clip-saturated pairs, and the no-cap variants at R 6.0 and 7.5, are reported descriptively only.

**P1c (secondary) – Order of departure.**
- **R_dep** is the smallest grid R at which this R and every larger braking R are in band with no deterministic crash.
- Predictions:
  - R_dep(A1) ≥ R_dep(A2) for each agent;
  - R_dep(G1) ≥ R_dep(G2) ≥ R_dep(G3) at A1, with at least one strict inequality and no reversal.
- The null probability is computed exactly over the 6 permutations of agent labels on the realised R_dep values.
- If all three are equal, the result is inconclusive.

#### P2 – The fast-approach departure is late braking, and its crashes are overspeed (G1, G2, G3)

**Set O:** tail and non-clipped below-support pairs at A1 that overspeed or have an overspeed crash. If O has fewer than 3 pairs, P2 is "not applicable".

- *Confirm* if both hold:
  - (a) Δd_b < 0 in at least 75% of O pairs, and the catch-trial response onset Δd_resp has the same sign in at least 75% of them;
  - (b) over the A1 tail and below-support cells that have crashes (deterministic or matched-noise), the median over cells of the per-cell fraction of crashes classified as overspeed (Section 9.1) is at least 75%.
- *Refute* if either holds:
  - (a′) Δd_b ≥ 0 in at least 50% of O pairs. The agent finds the corner in time; the error is in depth or target.
  - (b′) that median is below 50%, and over-brake or line/steer is the majority crash type.
- Otherwise the result is inconclusive.

**P2b (secondary) – Onset or depth.**

The entry-speed excess decomposes exactly and symmetrically:

v_e² − v_e*² = (v_b² − v_b*²) − (ā + ā*)(d_b − d_b*) − (d_b + d_b*)(ā − ā*), with ā = (v_b² − v_e²)/(2d_b).

The three terms are approach, onset and depth.
- Prediction: the median onset share over O is greater than 50% of the summed positive terms.
- Runs with no sustained brake are listed as "no-brake entries". If they overspeed, they count as onset failures.

**P2d (exploratory) – Point-mass null and line.**
- Does hairpin v_min sit nearer the QSS-centreline value than the key?
- e_vmin^line (Section 5) separates "exceeds grip on its own line" from "took a different line".

#### P3 – The braking law scales like physics (agents with a sustained brake in at least 50% of tier B deterministic runs)

**Data.** Deterministic tier A and tier B runs (all phases) that have:
- a sustained brake;
- an onset that is not forced (more than 0.5 s after the start);
- v_b − v_e ≥ 30 km/h;
- v_e observed. Runs that crash after s_c0 are included.

**Primary fit.**

log d_b = a_g + b·log d_b^phys(v_b, v_e)

- d_b^phys is the model's straight-line braking distance from the agent's own v_b to its own v_e (Section 5).
- a_g is a separate intercept for each geometry.
- A slope b = 1 is lawful braking.

**Model comparison.** Three generative models, each with per-geometry constants, are compared by held-out RMSE of log d_b, holding out one (v0, D) condition at a time:

| Model | Form |
|---|---|
| M_phys | d_b = c_g·d_b^phys |
| M_idx (preview-index rule) | d_b = c_g·clip(6 s·v_b, 150, 500): time-to-corner below 300 km/h, fixed distance above it |
| M_fd (fixed distance) | d_b = c_g |

**Decision:**
- *Confirm* if all hold:
  - b ∈ [0.9, 1.1];
  - every leave-one-geometry-out b ∈ [0.85, 1.15];
  - M_phys has the lowest held-out RMSE.
- *Refute, naming the winning rule*, if both hold:
  - M_idx or M_fd beats M_phys by at least 10% in held-out RMSE;
  - b lies outside [0.85, 1.15].
- Otherwise the result is inconclusive.

**Pre-tag separation check.** In the step-5 dry run, on the key's own design values (v_b*, v_e*), the slopes that M_idx and M_fd predict must differ from 1 by at least 0.25. Otherwise P3 is demoted to secondary before the tag.

**Secondary analyses:**
- Fit log d_b = a + β·log(v_b² − v_e*²) separately for 90° and 180° corners, for the agent and the min-time key (β*). Prediction: \|β − β*\| ≤ 0.1.
- Plot d_b/d_b^phys against v_b. A flat ratio means lawful braking; a ratio falling with v_b means a time-to-corner rule.
- Analyse the 310 km/h cells separately, because d_max saturates at 500 m above 300 km/h.

#### P4 – Specialist: reads the road or memorised (S1, S2, S3, each classified separately)

**Data.** Tier D (Section 3.3).
- **C(cond)** is the fraction of braking Hockenheim corners (the key brakes in condition (a)) whose 4 deterministic phase runs complete in at least 3.
- **Completion** means no termination before s_c1 + 250 m.

**Measures:**
- **S_spec:** the slope of log v_min against the min-time key's log v_min* across the dose series λ ∈ {0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.25}, median over braking corners.
- **Phantom index φ** (on swap B, where the corner is replaced by straight road): φ = (v_peak − v_min,region)/(v_peak − v_min,i^(a)).
  - v_min,region is the minimum over [s_c,i − 50, s_c,i + L_c,i + 50].
  - v_min,i^(a) is the agent's own v_min at corner i in condition (a).
  - A reader gives φ ≈ 0; a memoriser gives φ ≈ 1.
- **Swap index m** (on swap A, where corner i is replaced by corner j): m = log(v_min^swap/v_min,j^(a)) / log(v_min,i^(a)/v_min,j^(a)).
  - A reader gives 0; a memoriser gives 1.
- **OOD distance d_OOD:** the median, over steps from the start to s_c + 50, of the nearest-neighbour distance from the run's normalised observation to that specialist's own Hockenheim buffer. It is divided by the buffer's own leave-one-lap-out median nearest-neighbour distance.
  - The buffer is 40,000 observations subsampled from 50 stochastic laps started at random s, at 60–100% of v_ref, normalised with that seed's obs_rms.
- **Memorisation index:** MI = [C_spec(a) − C_spec(iso)] − mean over G finals of [C_G(a) − C_G(iso)]. "iso" is the factorial cell with both roads straight and constant width. Reported with every class.

**Classes.** Apply in this order; the first match wins.

| Order | Class | Rule |
|---|---|---|
| 0 | Gate | C(a) ≥ 0.9 is required. Otherwise: "not competent from these states". Report and stop. |
| 1 | Off-manifold fragile | Some single-factor F1 condition has C ≤ 0.5, **and** some F6 irrelevant perturbation whose d_OOD is at most that condition's median d_OOD also has C ≤ 0.5. Context keying cannot then be told apart from fragility. |
| 2 | Context-keyed memorisation | Some single-factor F1 condition has C ≤ 0.5, while every F6 perturbation with d_OOD at least that condition's has C ≥ 0.8. The key is named by the F1 main effects: the factor or factors whose change drops mean C by at least 0.4. Width-keyed is the case where the key is width. |
| 3 | Corner-keyed memorisation | All 8 F1 conditions have C ≥ 0.8, and at least one of: median φ ≥ 0.5; median m ≥ 0.5; S_spec < 0.4; C(mirror) ≤ 0.5. |
| 4 | Reads the road, generally | Class 5's criteria, plus deterministic completion ≥ 0.8 on tier A at A2 for R_seg ≥ 10.7 m, plus no crash on tier G. |
| 5 | Reads the road, locally | All F1 conditions and C(mirror) ≥ 0.8; C ≥ 0.8 at every dose λ; S_spec ∈ [0.6, 1.4]; median φ < 0.25; median m < 0.25. |
| 6 | Mixed | None of the above. Reported corner by corner. |

**Claims about the recipe.** A class is claimed for "the specialist recipe" only if at least 2 of the 3 seeds fall into it.

**Prediction:** context-keyed memorisation (class 2) for all three seeds.
- S1 is pilot-informed.
- S2 (final) and S3 are prospective.

**Tier G rule.** A specialist that crashes on a null straight counts as evidence for class 1 on synthetic stimuli. Its tier A, B, C and E results are then reported only as completion rates.

#### P5 – Mirror symmetry (sanity control; not confirmatory)

**Gates**, which must pass before any agent result is read:
- *Harness:* replaying an action sequence on a stimulus, and the steer-negated sequence on its mirror image, gives mirrored states to 1e-5 over 400 steps.
- *Keys:* the left and right solutions agree to a relative return difference of 1e-6.
- Stochastic runs negate the steering noise on the mirrored stimulus, so the two members of a pair see mirrored noise.

**Generalists:**
- across pairs, median \|Δe_ve\| ≤ 2 percentage points and median \|Δd_b\| ≤ 5 m;
- deterministic crash concordance ≥ 90%.
- If this fails, the harness gate is re-run. If the gate still passes, the asymmetry is reported as learned.

**Specialist asymmetry** in favour of right-handers at R ≤ 18.3 m is descriptive only. It rests on Hockenheim's 4 tight corners.

#### P6 – When each part of the curve becomes correct (G1, G2, G3 checkpoints; S2 and S3 descriptive)

**Definitions:**
- **Competence** at checkpoint c and level R, at A2: every deterministic run of every θ-pair at that R completes, and every one of those pairs is in band.
- **t_R** is the first checkpoint from which competence holds at every later checkpoint.
  - If competence is never reached, t_R = ∞ and takes the maximum rank.
  - Ties take mid-ranks.
- Checkpoints that fail the gate (Section 8) get crash curves only.

**Covariates recorded for every checkpoint:**
- exp(log_std);
- the tier G top speed v_top,c.

**Predictions:**

| Prediction | Status | Rule |
|---|---|---|
| P6a, larger R first | Secondary | Spearman ρ(R, t_R) ≤ −0.5 (one-sided permutation test, 10,000 permutations). Refuted if ρ ≥ 0. |
| P6b, caution first | Exploratory, because log_std starts at −0.5 and caution is partly mechanical | At the first checkpoint that passes the gate, median bulk e_ve^bo < −b. Reported against exp(log_std). |
| P6c, overspeed appears late | Secondary | If tail e_ve^RA at A1 exceeds +b at the final checkpoint, the first checkpoint where it exceeds +b is no earlier than t_bulk. Reported with v_top,c as a covariate. |
| P6d, size against frequency | Exploratory | Do R 69.8–87.3 m (rarer in v1 training) become competent earlier than R 28.6–44.7 m (more frequent)? |

**Keys for P6c.** Re-anchored A1 keys are solved at 4 checkpoints per generalist run:
- the first checkpoint that passes the gate;
- two log-spaced middle checkpoints;
- the final checkpoint.

The exception is when the finals show that re-anchoring is immaterial: \|v_e*RA − v_e*\| ≤ 1 km/h in at least 90% of A1 cells. Nominal keys are then used for the checkpoints.

#### P7 – Does preview sampling limit the reading of short, tight corners? (secondary; all final snapshots)

**Measures:**
1. **Closed-loop phase spread.** The standard deviation of d_b across 16 deterministic phases, on A1 cells with R ∈ {9.4, 11.7, 14.6, 28.6, 55.9} × θ ∈ {90°, 180°}. The null is the largest of:
   - the key's own SD across its 4 phase-matched solutions (R ≤ 18.3 only);
   - the Qp controller's SD (Section 7);
   - pure step quantisation, v_b·0.05/√12 (1.24 m at 310 km/h).
2. **On-policy aliasing amplitude, A_alias.**
   - Take the agent's deterministic u = 0 run and the states in [t_b − 1 s, t_b].
   - Re-evaluate the mean pedal command with the corner translated by δ ∈ {0, 0.25, 0.5, 0.75}·Δ(d_c), where Δ(d_c) is the preview spacing at the corner's current distance. The strip is resampled exactly from the trapezoid formula.
   - Fit pedal = α + β·δ at each state. A_alias is the RMS of the residuals.
   - The same is computed for Q, which should be about 0, and for Qp.
3. **Open-loop scans (tier O), descriptive.**
   - Restricted to d ≥ d_b^agent(v) + 50 m, so only states the agent visits are probed.
   - Detrended with a cubic smoothing spline with 6 degrees of freedom.
   - Π and A are reported as in v1, with no thresholds.

**Decision:**
- *Confirm "sampling matters"* if both hold:
  - on short tight cells (90°, R ≤ 14.6) the agent's phase SD exceeds the null by at least 2 m in at least 75% of cells, while on the 180° cells at the same R the excess is at most 1 m in at least 75%;
  - A_alias on short cells is at least 2× Qp's.
- *Refute* if the excess is at most 1 m in at least 75% of all P7 cells and A_alias is no larger than Qp's. The preview then behaves like an ideal observer with the same samples.
- *Mixed:* A_alias exceeds Qp's but the closed-loop excess is absent. A memoryless MLP cannot integrate over frames, so the open-loop aliasing is being filtered by the plant (pedal lag) and by the prev_action and accel inputs. Step 2 must then use closed-loop sensitivities.

#### P8 – Explained-crash fraction (pre-registered endpoint; tier E)

**Units.** The 15 (agent, crash corner) pairs: 8 for G1 and 7 for G2. G3's own R_DEV crash corners, taken from its standard R_DEV evaluation, are added by the same construction rule. That addition is pre-specified and is not a deviation.

**Reproduction.** The real rung reproduces a crash if both hold:
- the transplant passes the fidelity gate (Section 3.2);
- the crashing agent crashes deterministically in at least 2 of 4 phases, or has native-noise P_crash ≥ 0.5 (n = 16).

**Per reproduced pair:**
- **Single-corner:** the isolated rung still crashes, by the same criterion.
- Otherwise **context.** A factor is *necessary* if ablating it alone gives P_crash ≤ 0.2 and no deterministic crash. If no single factor is enough, the label is "joint context".
- **Twin agreement:** for a single-corner pair, the trapezoid twin crashes or overspeeds.
- **F** = (single-corner pairs with twin agreement) / (reproduced pairs).

**Predictions:**

| Set | Prediction | Status |
|---|---|---|
| G2's 5 tail crashes: Norisring 12.6, YasMarina 12.7, Melbourne 14.0, Catalunya 13.9, Zandvoort 12.1 m | *Confirm* if at least 3 are reproduced and at least 3 are single-corner with twin agreement. *Refute* if at most 1 is. Otherwise inconclusive. P1-tail and P2 may be described as explaining G2's real crashes only if this is confirmed. | Confirmatory in the secondary family |
| Bulk crashes: G1 Brands Hatch 33.6, Oschersleben 47.2, Sochi 21.1, Zandvoort 24.4; G2 Oschersleben 47.2, Sochi 21.1 | Context in at least 2/3 of the reproduced ones. This states up front that single-corner reading of R is not expected to explain them. | Secondary |
| Twin transfer (E1 twins + E2) | Isolated real corner and twin agree within 5 percentage points on e_ve in at least 80% of corners. The agent's tier A e_ve curve, evaluated at each twin's R_seg, θ class and approach (A1 if the real v_b ≥ 250 km/h, else A2), rank-correlates with the real-rung e_ve at Spearman ≥ 0.5. The real − isolated difference is reported as the context effect. | Secondary |

## 3. Stimuli

### 3.1 Synthetic construction

**Curvature.** For a corner with radius R, angle θ (radians), direction σ = ±1 (+1 is left) and u = s − S_C:
- l_r = max(10, min(l_typ(R), θR)) in the core tiers; l_r = 10 m in the tier E3 "abrupt" cells.
- κ_peak = min(1/R, θ/l_r).
  - If θR ≥ 10, then κ_peak = 1/R. A corner with θR ≤ l_typ(R) is a triangle whose peak is exactly 1/R.
  - If θR < 10, the corner is the generator's 10 m triangle with peak θ/10. For example R6/90° gives R_eff = 6.37 m.
- Plateau L_p = θ/κ_peak − l_r ≥ 0; corner length L_c = L_p + 2l_r.
- Profile prof(u) = u/l_r on [0, l_r), 1 on [l_r, l_r + L_p), and (2l_r + L_p − u)/l_r after that, clipped at 0.
- κ(s) = σ·κ_peak·prof(u), sampled at s_i = 2i for i = 0…2559.
- Arrays are built with `generator.strip_arrays(κ, w_l, w_r, ds=2.0)`, vmapped. The solvers get the Track from `generator.to_track`.

**Widths.**
- The total width w is split equally between the two sides, and the inside cap is applied at every knot. At width 12 the cap binds for R < 8.3 m; at width 15, for R < 10.8 m. These cells are flagged.
- No-cap variant (tier A, R ∈ {6.0, 7.5}): constant w/2 on each side with no cap. Off-distribution; descriptive only.
- Tier E3: the inside narrows to w_in^c following the corner's own profile, w_in(s) = w/2 − (w/2 − w_in^c)·prof(u), then the cap. The outside stays at w/2.

**Landmarks:**
- s_c0 = S_C (corner start)
- s_a0 = S_C + l_r (arc start)
- s_apex = S_C + L_c/2 (geometric apex)
- s_a1 = S_C + L_c − l_r (arc end)
- s_c1 = S_C + L_c (corner exit)

**Realised geometry,** recorded for every stimulus:
- R_eff = 1/max\|κ\| from `kappa_at` every 0.05 m over [s_c0 − 20, s_c1 + 20];
- θ_eff = ∫κ ds;
- onset rate κ_peak/l_r;
- R_seg;
- any counter-curvature ringing.

**Length checks,** asserted by the builder:
- s_c1 + 250 < 4598 m.
- Every key window ends before 4598 m.
- The longest corner (R 300 m, 90°) has L_c ≤ 521 m. An 800-step window from s0 ≥ 400 m covers at most about 3.5 km, so both checks hold.

### 3.2 Transplants (R_DEV only)

**Common construction:**
- `lockbox_guard(name)` is called first.
- The Track is the one the real-lap evaluation uses (`eval/laps.py`).
- κ, w_l and w_r are excised, and κ is tapered to 0 over 20 m at each cut.
- Width at each cut edge is held constant beyond the cut.
- The strip is padded with straight road to N = 2560, so that the corner start lands at S_C.

**Initial state.** The relevant agent's real deterministic lap state, with the real prev_action and accel.

**Fidelity gate.** On each generalist transplant, the deterministic run from that agent's own state must match its real lap to \|Δv\| ≤ 0.1 m/s up to s_c − 100 m. Transplants that fail are excluded from P8 and listed.

**YasMarina diagnosis** (before YasMarina is used as a positive control):
- Compare the stochastic P_crash (n = 64) on the real closed track and on the transplant, both from the same state.
- Do this with prev_action and accel set to their real values, and set to 0.
- If the real-track P_crash is below 0.5, YasMarina is labelled "knife-edge" and leaves the positive-control set. It stays in P8 as unreproduced.

### 3.3 Tier D: Hockenheim conditions

There are 20 corners, from `corners.segment` on Hockenheim. Each condition strip runs from s_start = s_c − (D_gap + 100) to s_e + 1200 m, where D_gap = clip(gap to the previous corner, 100, 600).

**Lead-in.** The first 100 m after s_start is always the real road, with real width. Every condition therefore starts from the same consistent state:
- each specialist from its own deterministic real-lap state;
- the generalists from S1's state.

**Regions:**
- "before" = [s_start + 100, s_c − 20];
- "corner" = [s_c − 20, s_e + 20];
- "after" = [s_e + 20, s_e + 1200].

Every substitution uses 20 m κ tapers.

| Family | Conditions | Count |
|---|---|---|
| F1 factorial | before ∈ {real, straight} × after ∈ {real, straight} × width ∈ {real, constant 12 m} (width outside the lead-in, with 20 m tapers). (a) is real/real/real; "iso" is straight/straight/constant. | 8 |
| F2 dose (real road) | Corner scaled in place by λ ∈ {0.8, 0.9, 0.95, 1.05, 1.1, 1.25}, κ(s) → κ(s/λ)/λ, with the road after it shifted. Width ±0.5 m over the whole strip. The inside cap is applied and flagged. | 8 |
| F4 mirror | The whole real strip mirrored, starting from the mirrored state | 1 |
| F5 swaps | A: the real approach to corner i, then corner j, then straight. j is the braking corner whose min-time v_min* is closest to \|log(v_i*/v_j*)\| = 0.35, and at least 0.15. B: the real approach, then straight road instead of the corner. | 2 |
| F6 irrelevant perturbations | Width wobble (both edges in phase, 60 m wavelength) at 0.3 and 0.6 m; band-limited κ noise (wavelengths 80–200 m, zero net heading) at 0.5e-3 and 1e-3 m⁻¹. Applied outside the lead-in. | 4 |

That is 23 conditions × 20 corners = 460 stimuli. Min-time s-OCP keys are solved for braking corners only: condition (a) plus the 6 λ doses, at most 140 solves.

### 3.4 Tier E: crash ladders, matched corners, crash shapes

**E1 – Crash ladders.** Each of the 13 distinct crash corners is cut at [s_c − 800, s_c + 400] and built in 7 versions:
1. real road;
2. road before the corner straightened;
3. road after the corner straightened;
4. constant width at the corner's median width;
5. isolated (all three changes);
6. trapezoid twin;
7. mirror of the real road.

The twin's (R, θ, l_r) are fitted by least squares so that `corners.segment` on the twin matches the real corner's R_seg, angle and length to within 2%, 1° and 2 m. The twin's width is the real width at the apex, with the inside cap applied.

Initial states:
- There are 15 crash pairs, each with its own state at s_c − 800, taken from that agent's real lap.
- Sochi and Oschersleben therefore have two states each.
- Every agent runs every ladder from every state of that corner.

**E2 – Matched corners that did not crash.** For each crash corner:
- take the R_DEV braking corner, by `corners.segment`, at which no generalist crashed;
- it must be in the same θ class (below 120° or at least 120°) and nearest in log R_seg;
- it must be at least 1 km from any crash corner on the same circuit;
- ties are broken with the root seed.

Each is built in 3 versions (real, isolated, twin) and starts from G2's real-lap state at s_c − 800.

**E3 – Synthetic crash shapes:**
- R ∈ {9.4, 11.7, 14.6, 18.3} × 90°;
- w_in^c ∈ {4.6, 6.0};
- l_r ∈ {10 (abrupt), l_typ};
- approaches A1 and E-fast (280 km/h, D 250 m);
- L/R.

The cells with w_in^c 6.0, l_typ and A1 are tier A cells and share their keys.

### 3.5 Grid and counts

**Ladder:**
- Braking R (17 values): 6.0, 7.5, 8.4, 9.4, 10.5, 11.7, 13.1, 14.6, 16.4, 18.3, 22.9, 28.6, 35.8, 44.7, 55.9, 69.8, 87.3. The half-steps 8.4–16.4 keep P1b and the tail testable.
- Controls: 109, 171 and 300 m, at 90° only.

**Approaches:**
- **A1** = (310 km/h, D = 600 m). The corner is hidden at the start, since d_max is 500 m.
- **A2** = typical approach for the corner's size:
  - v0 = v_typ, the median v_peak^QSS of base-grammar training corners in the stimulus's R_seg bin and θ class, rounded to {140, 160, 180, 200, 220, 250} km/h. Expected about 160 km/h for R < 18.3.
  - D = the pinned s-OCP's brake-demand onset distance from v0, plus 25 m. Before the tag, D is adjusted in 10 m steps until v_b* is within ±10% of v_typ in at least 90% of braking A2 cells.
  - If the key does not brake at v_typ, D = 250 m and the cell is flat.
  - At A2 the anchor point lies before the start, so the re-anchored key is the nominal key.
- **E-fast** = (280 km/h, 250 m).
- **Tier B grid:** v0 ∈ {160, 220, 280, 310} km/h × D ∈ {250, 600} m.
- **Recovery test:** D = 100 m.

| Tier | Purpose | Design | Stimuli (L/R) | Cells | Keys, one direction (priority) |
|---|---|---|---|---|---|
| A | P1, P2, P5, P6, P7 | 20 geometries at 90° (17 braking + 3 controls) and 15 at 180° (R ≤ 55.9); width 12; A1 and A2. Plus no-cap R {6.0, 7.5} × {90°, 180°} at A2. | 148 | 74 | 74 nominal + 66 backed-off + 60 phase-matched (P0) |
| B | Crash map, P3 | R {7.5, 9.4, 11.7, 14.6, 18.3, 22.9} × θ {90°, 180°} × v0 {160, 220, 280, 310} × D {250, 600}, of which 12 cells are shared with A1. Plus recovery: D 100 × v0 {220, 280, 310} × R {9.4, 14.6} × θ {90°, 180°}. | 192 new | 96 new | 12 recovery (P2). Re-anchored keys for D = 600 (P1). |
| C | θ and width (exploratory) | R {14.6, 22.9, 44.7} × 45° and R {9.4, 14.6, 22.9, 44.7} × 135° at w 12; R {9.4, 14.6, 22.9, 44.7} × θ {90°, 180°} × w {9, 15}; A2 | 46 | 23 | 23 (P2) |
| D | P4 | 20 Hockenheim corners × 23 conditions | 460 | 460 | ≤ 140 s-OCP (P1) |
| E1 | P8 | 13 corners × 7 versions; 15 initial states | 105 runs on 91 strips | 105 | 30 real + isolated (P0); 15 twins (P1) |
| E2 | P8 twin transfer | 13 corners × 3 versions | 39 | 39 | 39 (P1) |
| E3 | Crash shapes, abrupt onset | As above | 64 (56 new) | 32 (28 new) | 28 (P1) |
| G | Null straights and catch trials | 4 km straight at each v0 in {140, 160, 180, 200, 220, 250, 280, 310} at w 12; w 9 and 15 at the v0 values tier C uses | ≤ 14 (symmetric) | ≤ 14 | ≤ 28 (W 400 and 800; P0) |
| O | Open-loop scans | R {9.4, 11.7} × {90°, 180°} and R {14.6, 28.6, 55.9} × {45°, 90°, 180°} at v ∈ {150, 200, 250, 300} km/h | 52 scans (forward passes) | — | 0 |
| **Total** | | | **≈ 1,060 closed-loop** | | **≈ 227 full-recipe + 126 warm-start TO-OCP, ≤ 140 s-OCP, plus agent-anchored keys (Section 6.1)** |

Tier C notes:
- 45° at R 9.4 and 11.7 is dropped, because θR < 10 there.
- The 225° cells are dropped, because the far preview points fall beside or behind the car.
- Regimes are assigned by R_seg.

## 4. Initial states and rollout protocol

**Synthetic initial state.** x0 = [s0, 0, 0, v0, 0, 0, 0, 0], prev_action = 0, accel = 0, t = 0, built with `init_env_state`.

**Car.** `default_context()` as float32, hidden_mu = 1.0. Step 2 asserts that S1's normalised context is within 1e-3 of 0, and that `default_context` lies inside the generalists' `CONTEXT_RANGES`.

**Phases.**
- s0 = S_C − D − u_k, with u_k = (k/4)·P for k = 0…3.
- P = v_sight·0.05 s, where v_sight is the nominal key's vx at the step when the corner start first enters the preview (s_c0 − s ≤ d_max(vx)). Stimuli without a key use the QSS profile instead.
- This spreads the 4 phases evenly over one control-step cycle at the moment the corner comes into view, not at the start.
- The P7 cells use 16 phases.

**First sight.** For every run, n, ξ, pedal and the distance travelled are recorded at first sight. A run is flagged *unsettled* if \|n\| > 1 m or the pedal is below 0.9 at that moment.

**Rollout length** equals the key's window W:
- W = 400 steps if the pinned s-OCP reaches s_c1 within 5 s, which leaves at least 15 s of window after the corner;
- otherwise W = 800.
- A run that has not reached s_c1 + 250 m by W is marked *time-censored*.

**Policy modes:**
- o = obs_rms.normalize(observe(st)), clipped to ±10.
- *Deterministic:* a = mean.
- *Native-noise stochastic:* a = mean + exp(log_std)·ε.
- *Matched-noise stochastic:* a = mean + (0.04, 0.08)·ε.
- The environment clips a to [−1, 1].
- ε comes from fold_in(fold_in(PRNGKey(20261019), stim_id), sample), so every snapshot sees the same noise. On the mirrored stimulus, ε_steer is negated.
- After termination the state is frozen with `jnp.where`.

**Replication per cell:**

| Snapshots | Deterministic | Native-noise stochastic | Matched-noise stochastic |
|---|---|---|---|
| Finals | 4 phases (16 on P7 cells) | 32 (tier A); 16 (all other tiers) | 16 (tiers A and E) |
| Last 3 checkpoints of G1, G2, G3 | 4 phases on A and B | 8 on A | — |
| Other checkpoints | 4 phases on A and G (generalists); D and G (specialists) | 8 on A (generalists) | — |

Deterministic runs are the primary readout of reading. Native-noise crash rates describe the agent as trained. The matched-noise crashes feed P2's crash typing.

**Recorded every step:**
- x[8], prev_action, accel, a_mean, a, log_std, reward, alive;
- every termination flag in `info`;
- util_f, util_r and the front brake demand β_f;
- per channel group (κ, edges, ego, context): the fraction of \|o\| ≥ 10, and max\|o\|.

Deterministic runs keep full traces. Stochastic runs keep the measures plus traces of s, vx, n, pedal, util and β_f.

**Batching.**
- A bank of unique TrackArrays plus a track_id, gathered inside a vmap over (track_id, x0, key).
- The snapshot, or the scripted controller's parameters, is passed as a jit argument.
- Chunks hold at most 2048 episodes.
- float64 is asserted off.

**Open-loop scans (tier O).**
- State: straight-line steady full throttle at v; n = ξ = vy = r = δ = 0; pedal = 1; prev_action = (0, 1); accel = (a_x(v), 0).
- The car is placed at d before s_c0, for d ∈ [d_b^agent(v) + 50, d_max] in 0.25 m steps.

## 5. Response measures

All measures are computed by one function, `response_metrics(traj, marks, null)`, from a common `Trajectory` with adapters for:
- agent rollouts;
- TO-OCP replays (20 Hz);
- the s-OCP (states on the s-grid, with t integrated).

Quantities are interpolated linearly in s. The pedal is always the state x[7]; the command is used only where stated.

**Model tables,** built once in step 3:
- **a_dec(v):** the largest straight-line deceleration of the model, tyres plus drag. The pedal is swept over [−1, 0] in steps of 0.005, with the other states at 0. Tabulated on a 1 km/h grid.
- **a_y,max(v):** the largest steady-state lateral acceleration with \|α_f\|, \|α_r\| ≤ α_max. Found by a grid search over (δ, sideslip), with the pedal set for zero longitudinal acceleration. Tabulated on a 5 km/h grid from 20 to 315 km/h.
- **d_b^phys(v_b, v_e)** = ∫ from v_e to v_b of v/a_dec(v) dv.
- **v_ss(ρ)** solves v²/ρ = a_y,max(v).

| Measure | Definition |
|---|---|
| Front brake demand β_f | max(0, −fx_f,dem)/fx_max,f at each post-step state. `measures.front_brake_demand` re-implements the demand lines of `dynamics.tyre_forces`; a test checks fx_f,dem·scale = `tyre_forces(x).fx_f` to 1e-10. |
| **Brake onset s_b (primary)** | The start of the last run of at least 3 consecutive steps with β_f > 0.3 that begins before s_apex, interpolated at the 0.3 crossing. d_b = s_c0 − s_b; v_b = vx(s_b). Earlier runs are *blips*: their count, positions and speed lost are logged. An onset within 0.5 s of the start is *forced*. The same definition applies to agents and keys. |
| Release s_r; trail | The first step after s_b with β_f < 0.1. Trail braking = s_r − s_c0. |
| Decision point s_dec | The last step before s_b at which the pedal *command* crosses 0 from above |
| **Response onset s_resp (catch trial)** | The first s at which \|pedal(s) − pedal_null(s)\| > 0.1 for at least 2 consecutive steps. pedal_null comes from the same driver (agent or key) on the tier G straight with the same v0, width and W, aligned by distance travelled s − s0. d_resp = s_c0 − s_resp. Defined for lift-only corners too. |
| Legacy onset | Pedal state ≤ −0.2 for at least 3 steps (the v1 definition). Secondary only. |
| Depth | Peak and mean brake-only deceleration, −dvx/dt minus drag/m, over [s_b, s_r]. Mean deceleration ā = (v_b² − v_e²)/(2d_b). |
| Speeds | v_e = vx(s_c0); v_arc = vx(s_a0); v_peak = max vx before s_b; v_min = min vx over [s_c0 − 50, s_c1 + 50], with its position relative to s_apex. \|v\| is reported alongside vx. |
| Friction use | Peak util_f and util_r over [s_c0, s_c1] |
| Path radius | κ_path = (r + β̇)/\|v\|, where β = atan2(vy, vx) and β̇ is a central difference. ρ_min = 1/max κ_path over [s_a0, s_a1], or over [s_c0, s_c1] for triangles. |
| Line-conditioned speed | e_vmin^line = v_min/v_ss(ρ_min) − 1, for agent and key. The *line-conditioned excess* is agent minus key. |
| Line η | σn/(w_in − 1) if σn ≥ 0, else σn/(w_out − 1), so each side is normalised by its own usable width. Reported at s_c0 − 30, s_apex and s_c1 + 30. Also: line apex (argmax η), its offset from s_apex, clearance 1 − max η, and RMS(n − n*) over [s_c0 − 100, s_c1 + 100]. |
| Corner time lost Δt_corner | [t_agent(s_ff) − t_key(s_ff)] minus the same difference on the null straight at the same distance travelled. s_ff is the first s > s_c1 at which both vx ≥ 0.97 × 315 km/h. If that is not reached within W, the last common s is used and flagged. Reported against both TO-OCP and s-OCP. vx(s_c1 + 100) is descriptive only. |
| Crash | Flag; every termination flag that fired; the first cause by precedence (corridor > slip > spin > stall > non-finite); side (σn > 0 is the inside edge); phase (approach, entry, exit, post); s_crash, v_crash, Δv(s_crash); crash type (Section 9.1). Measures after a crash are censored and never imputed. |
| Δv(s) | vx − vx* on a 1 m grid over [s_c0 − 300, s_c1 + 100] |
| Regret | J* − J_agent over W, with the step-0 action-rate term computed against a₋₁ = 0 for both. Computed only where a key exists at that phase. Corner regret = regret(stimulus) − regret(null). |
| Clipping | Per channel group and per step: the fraction of \|o\| ≥ 10, and max\|o\|. κ clipping defines a regime; clipping in the edge, ego or context groups triggers the out-of-distribution sensitivity analysis. |
| Oscillation (gate) | Mean \|Δ steer_mean\| per step over the first 5 s |

## 6. The answer keys

### 6.1 Key types, and which claim each one backs

| Key | Solver | Solved from | Backs |
|---|---|---|---|
| Nominal (*) | TO-OCP, W = 400 or 800 | Stimulus x0 at u = 0 | Regret; "did the agent find its own optimum"; overspeed side of P1 and P2; P5 gate; P8 e_ve |
| Backed-off (*bo) | TO-OCP with α_max 8.5° and the corridor narrowed 0.35 m per side. α_max is a runtime parameter and the widths are data, so no recompile is needed. Lock is unchanged: it is the car's physical limit. | x0, u = 0 | Underspeed and caution side of the band; P6b |
| Phase-matched | Nominal at u_1, u_2, u_3 | x0 at those phases | Δd_b, P2b and key phase SD on A1 cells with R ≤ 18.3 (20 geometries) |
| Null | TO-OCP on a straight | Each (v0, width, W) | Catch-trial onset, corner regret, Δt_corner |
| Re-anchored (*RA) | TO-OCP, W = 400, via `solve_from_state` | The agent's own deterministic u = 0 state at s_L = s_c0 − (d_b* + v_b*·1.0 s) | e_ve at A1 and on tier B free approaches |
| Viability | TO-OCP, W = 200, via `solve_from_state` | The agent's states at s_b, s_c0 − 50, s_c0 and s_a0 (those before the crash) | Point of no return; crash type |
| Pinned min-time | `solve_min_time_pinned`, ds 2.0 m | x0 | "Physics requires" statements: tier D v_min*, A2 layout, V8 band, P3's β*, time lost in seconds; warm start |
| Min-time in the agent's actuation | TO-OCP with `WindowSpec(gamma=1.0, action_cost=1e-4)` (one compile per window length) | Validation cells | Splits V3 into an objective effect and a discretisation effect |
| QSS (`oracle/qss.py`, and a version calibrated to the model's tables) | — | — | Layout, P2d null, controllers. Never used as a key. |

**`solve_from_state`** lives in `interp/answer_key.py`. It is a subclass of `WindowSolver` whose `solve()` copy takes an optional `a0_fixed`, which sets lbx = ubx for a_0.
- An anchored key is solved from x_{L−1}, with a_0 fixed to the agent's action a_{L−1}.
- The free problem therefore starts at x_L, with a₋₁ equal to the agent's real previous action. The step-0 penalty is a constant.
- It reuses the compiled solvers.
- Test: with a0_fixed = None it equals `WindowSolver.solve` to 1e-10.

**Re-anchoring** applies only when the agent reaches s_L alive and the key has a sustained brake. Otherwise the nominal key is used.

### 6.2 Recipe for agent-free keys

1. **Pinned s-OCP.**
   - The strip is cut at s0 − 20 m and pinned at the s0 node: z[:,0] = z0, and `speed_profile(..., v_start=v0)` seeds the guess.
   - It runs to s_c1 + 600 m at ds = 2.0 m.
   - Assert that κ and w equal the full strip's values to 1e-9.
2. **Initial guess.** `min_time_guess`, extended with a straight full-throttle tail.
3. **Window and stages.**
   - W comes from Section 4.
   - Continuation stages are {200, 400, 800} up to W. A stage is skipped if its window ends before s_c1 + 300 m on the s-OCP's time line.
   - If the window is 600 steps, it is used only for V2.
4. **Starts.**
   - (1) The s-OCP guess.
   - (2) The key of the neighbouring larger R (same θ, width and approach), re-timed by aligning in s. Each series is solved in descending R.
   - (3) A rollout of the agent-free Q controller (Section 7). It is added on the validation set, for the largest R in each series, and whenever starts (1) and (2) disagree by more than 2 km/h in v_e or v_min, or by more than 1e-4 relative in J.
   - No agent's rollout is ever used as a start.
5. **Caps.**
   - IPOPT max_iter is 800 per stage, and max_cpu_time is 400 s per start. A capped start counts as failed.
   - If every start fails: one retry with max_iter 3000, after which the cell is "unsolved".
6. **Selection and cache.**
   - The key is chosen with `best_verified`.
   - The cache is keyed by a hash of (stimulus, x0, W, params, spec). It stores the solution, J, the verification result, each start's J and measures, solver statistics, multipliers and the active-set log.

**Warm-start keys:**
- Backed-off and phase-matched keys: one start, warm from the nominal key (shifted by u for phases). If it fails, the full recipe runs.
- Null keys: s-OCP plus one start.
- Anchored keys:
  - start (1) is the nominal key re-timed from the matched s;
  - viability keys add start (2), the backed-off key re-timed.

### 6.3 Tolerance, multiple basins, and agents that beat the key

- **Near-equal set N:** the verified starts with J ≥ J_best − 1e-4·\|J_best\|. Starts within (1e-4, 1e-3] are logged; worse starts are ignored.
- **Multi-basin cells.** A cell is flagged if any of these differs across N by more than twice its floor: d_b (1 m), v_e (0.5 km/h), v_min (0.5 km/h), s_vmin (2 m), line apex (2 m). Such cells are:
  - excluded from confirmatory key-based errors;
  - kept in crash analyses;
  - listed.
- **Tolerance:** τ = max(floor, spread across N).
  - Onsets compared with a key not solved at that phase have a floor of max(1 m, 0.5·v_b*·0.05 s), which is 2.2 m at 310 km/h.
  - Errors are never set to zero. τ is used only in V6, in the braking/flat classification and in reporting.
- **Agent beats the key.** If any agent's deterministic u = 0 return exceeds J* by more than 1e-6·\|J*\|:
  - the cell is re-solved, warm-started from every final agent's u = 0 rollout;
  - the best verified solution is taken;
  - every agent's errors are recomputed;
  - a deviation is logged.

### 6.4 Active sets and infeasibility

**Active-set log.** For every key and step, flag:
- slip within 0.5° of α_max (front or rear);
- steering within 2° of δ_max;
- corridor within 5 cm.

The solver's multipliers (meta "lam") are stored. Each cell is flagged as slip-limited, lock-limited or corridor-limited according to the active set within ±1 s of v_min.

Expected flags:
- R13/180 at A1 is slip-limited and not lock-limited.
- 180° corners at R ≤ about 9 m are lock-limited.

Lock-limited cells stay in the analysis; a sensitivity analysis excludes them.

**Infeasibility certificate.** This is a relaxed point-mass bound:
- v_reach,min(s) is the speed reached when braking from x0 at 1.1·a_dec(v).
- v_cap = sqrt(1.1·a_y,max(v)·ρ_ub), where ρ_ub is 1.1× the minimum radius of the oracle's minimum-curvature line through the corner.
- If v_reach,min(s_apex) > v_cap, the cell is **infeasible (bound)**.
- A cell with no verified key that is not certified is **unsolved**.

Both categories are:
- excluded from error measures and from P_crash;
- listed.

If more than 5% of P0 cells are unsolved, they are investigated before analysis and a deviation is logged.

Imposed D = 100 m cells are screened with the bound before solving.

### 6.5 Validation

All checks must pass before the tag.

**Validation stimuli:**
- (R, θ) ∈ {(13, 180), (13, 90), (40, 90), (40, 180), (100, 60), (100, 90)} at A1 and A2;
- (13, 90) and (40, 90) imposed at (310 km/h, 100 m);
- 10 tier A cells for V7.

| Check | Pass criterion | If it fails |
|---|---|---|
| V1 Replay | `verify_window` passes for every key used (return within 1e-6; alive at every step) | Re-solve; otherwise exclude and list |
| V2 Horizon | A2 and imposed cells: 400 vs 800 steps. A1 cells: 800 vs 600 (one compile). Up to s_c1 + 100: max\|Δv\| ≤ 0.1 m/s and \|Δn\| ≤ 0.05 m. | Use W = 800 for that class of stimulus |
| V3 Objective and discretisation | Objective effect (TO-OCP vs min-time in the agent's actuation) and discretisation effect (that vs the s-OCP), reported for d_b, v_e, v_min and Δt. The expected gaps are estimated here, not fixed in advance. | Blocks only if the objective effect on v_e exceeds b/2. P1 then reports both keys. |
| V4 Spread across starts | Reported. Multi-basin rule as in Section 6.3. | — |
| V5 Mirror | Left and right solves: return within 1e-6 relative; measures within 1e-3 | Blocks the mirror shortcut |
| V6 Monotonicity in R | At fixed (θ, w, approach), v_e* and v_min* do not decrease and d_b* does not increase, beyond τ | Re-solve warm-started from neighbours; list persistent violations |
| V7 Phase | Speeds: phases u_1–u_3 within 0.5 km/h of u_0. Onsets: reported (expected ≤ one control step). | Speeds failing: phase-matched keys for every braking cell |
| V8 Speed band | On the braking validation cells at A1 and A2, solve the pinned s-OCP with vx(s_c0) ≤ (1 − e)·v_e^sOCP for e ∈ {0.05, 0.075, 0.10}. b is the e at which the median far-field time cost reaches 0.10 s (interpolated linearly, rounded to 0.5 percentage points, clipped to [5%, 10%]). | — |
| V9 Backed-off key | v_e*bo ≤ v_e* and v_min*bo ≤ v_min* on every braking cell; fewer slip-active steps | Investigate |
| V10 Solver copies | `solve_min_time_pinned(z_start=None)` equals the repo solver to 1e-8 on a 600 m strip. `solve_from_state(a0_fixed=None)` equals `WindowSolver.solve` to 1e-10. Pins hold to 1e-9. | Blocks |
| V11 Active set | (13, 180) at A1 is flagged slip-limited and not lock-limited | Investigate the logging |

## 7. Positive controls and classifier validation (before the tag)

The controllers live in `interp/controls.py` and are vmappable JAX functions with a policy's signature.

| Control | Definition |
|---|---|
| **T** (tape) | The nominal key's actions at u = 0, replayed open-loop |
| **Q** (QSS, full road) | Pure pursuit on the centreline, lookahead max(8 m, 0.6 s·v). Speed target c_v·v_QSS(s), from a forward-backward pass with a_y,max(v) and a_dec(v) on the full-resolution κ. Pedal from a proportional speed controller (0.5 per m/s) plus feed-forward, clipped. c_v is the largest of {0.95, 0.90, 0.85, 0.80} that gives 100% deterministic completion on tier A and on tier D condition (a). |
| **Qp** (QSS, preview only) | Q, but the κ ahead is rebuilt each step from the 40 preview κ samples alone: linear in d, with the last sample held beyond d_max |
| **c1** (late onset) | Q with the braking branch of its target moved 15 m towards the corner |
| **c2** (saturating target) | Q reading κ capped at 1/12 m⁻¹, so it reads R < 12 m as 12 m |
| **c3** (time-to-corner) | Q that brakes when time to the corner ≤ τ_c, with τ_c fitted so that d_b matches the key at R 22.9 m, 90°, A1. Its target entry speed stays lawful. |
| **K** (kNN memoriser) | 1-nearest-neighbour lookup on S1's Hockenheim buffer (normalised observations → the neighbour's action). Falls back to k = 5 with distance weighting if C(a) < 0.9. |

Each control runs through the whole pipeline: stimulus bank → rollout harness → measures → analysis.

| Control | Tiers | Required verdict |
|---|---|---|
| T | A, G (u = 0) | Regret 0 ± 1e-6 relative; every error within τ; no termination |
| Q | A, B, D | Harness values of e_ve, e_vmin and Δd_b equal the offline predictions from Q's own profile against the key, within 2 percentage points or one control step. P1a "confirm" or "cautious but lawful". P3: b ∈ [0.9, 1.1] and M_phys wins. P4 class 4 or 5. P5 symmetric. |
| Qp | A (A1 short corners), O | Runs cleanly; provides P7's null |
| c1 | A, B | P2 confirm, with onset share > 50% |
| c2 | A | P1b confirm on R < 12 cells, under regimes defined as if q1 = 12 and q10 = 20; P1a confirm |
| c3 | A, B | P3 refuted, with M_idx named |
| K | D | P4 class 2 or 3. If it lands in class 1, the F6 amplitudes are halved once and the check is re-run. |

Any misclassification blocks the tag until it is fixed.

**Dry run on the real key design.** Simulated agents are built from the real keys plus phase noise:
- lawful: key × 0.99;
- overspeed below 12 m;
- late onset;
- time-to-corner braking;
- the tier D completion pattern of context keying.

Each must return its planted verdict. The dry run also writes the **power table**: for each confirmatory test and agent, the pairs per regime, and the smallest planted overspeed (in steps of 2.5 percentage points) that the rule detects.

## 8. Agents, checkpoints and what can be claimed

| ID | Path | Snapshots | Status |
|---|---|---|---|
| S1 | `runs/hockenheim-specialist-mlp-200M/agent.pkl` | Final only | Done; pilot-informed |
| S2 | `runs/hockenheim-specialist-mlp-200M-s2/ckpt/*.pkl` | 16, to 199.9M | Done. Snapshots after 122.3M are prospective. |
| S3 | `runs/hockenheim-specialist-mlp-200M-s3/ckpt/*.pkl` | Log-spaced | Training; prospective |
| G1 (v1) | `runs/dr-pilot-200M/ckpt/*.pkl` | 16 (10M–199.9M) | Done; pilot-informed |
| G2 (v2) | `runs/dr-pilot-v2-tilt30/ckpt/*.pkl` | 16 | Done; pilot-informed |
| G3 (v3) | `runs/dr-pilot-v3-tilt50a2/ckpt/*.pkl` | 16 expected | Queued; prospective |

`runs/hockenheim-specialist-mlp` is excluded.

**Loading.**
- `snapshot_from_pickle`, with `PPOConfig` filtered to its known fields, and `Actor(cfg, 2, False)`.
- The manifest records each file's sha256, timesteps, size and final exp(log_std).
- Files of runs still in progress are listed only once the run is done.
- `UnpicklingError` is caught.

**Checkpoint gate.** A checkpoint is analysed for reading only if both hold:
- deterministic completion ≥ 90% on the gentle cells (tier A controls and R 87.3 at A1 and A2) and on the tier G width-12 straights;
- oscillation index < 0.1.

**Families:**
- **Primary family (prospective):**
  - G3: P1a, P1-tail, P1b, P2, P3;
  - S2 and S3: P4 class;
  - the recipe claim, which needs 2 of the 3 seeds, S1 included.
- **Secondary family (pilot-informed):**
  - G1 and G2: P1a, P1-tail, P1b, P2, P3;
  - G2: P8 tail;
  - S1: P4.
- Step 8 starts only when both G3 and S3 finals are in the manifest. If either run fails, that is a deviation; its tests are dropped, not replaced.

**Limits on claims:**
- **Network-level claims need stability.** A generalist claim is reported as a property of that network only if it holds at the final checkpoint and at least one of the two before it (2 of the last 3). Otherwise it is reported as "unstable".
- **One seed per generalist variant.** Every generalist statement is about a specific network. The ordering of variants (P1c) is suggestive, not proof. Differences between variants smaller than the S1–S3 spread, on tier A and tier D condition (a), are not interpreted.
- **Recommendation for the user, not scheduled:** if the ordering of variants becomes a headline, queue a second v2 seed. That needs the training queue, which is the user's decision.

## 9. Analysis

### 9.1 Units and pair values

**Unit of analysis.** The mirror pair (geometry, approach). For deterministic runs it holds 2 directions × 4 phases = 8 runs.

**Speed measures.** The pair value is the median of the 8 runs, where:
- an overspeed crash counts as +∞;
- a crash before s_c0 counts as missing;
- a non-overspeed crash after s_c0 keeps its observed v_e, but its v_min is missing;
- if more than 3 of the 8 values are missing, the pair value is undefined. It is listed and still counted in the crash analyses.

**Onset measures.** The median over runs that have a sustained brake; at least 5 such runs are required.

**Crash measures:**
- deterministic crash fraction (of 8);
- native-noise fraction (of 64 on tier A);
- matched-noise fraction (of 32).

**Crash type.** Each crashed deterministic or matched-noise run is assigned one type:
- **Overspeed:**
  - the certified point of no return lies at or before s_a0, with Δv > 0 at that point; or
  - if the point of no return is unresolved: util_f or util_r ≥ 0.98 for at least 3 steps with β_f < 1.0 and Δv > 0 at the first such step.
- **Over-brake:** β_f ≥ 1 and util_f ≥ 0.98 for at least 3 steps in the last second, with Δv ≤ 0.
- **Line/steer:** max(util_f, util_r) < 0.9 over the last 1.0 s.
- **Other:** none of the above.

Slip and spin terminations are typed by the same rule. Every termination flag that fired is reported by type.

**Viability solves** are made only for crashed u = 0 runs, or the first crashed phase. The budget is capped at 200 solves in all; beyond the cap a seeded subset of 10 crashed pairs per agent is used.

### 9.2 Inference

- **Deterministic decision rules.** The design is a fixed grid run by a deterministic agent, so P1–P3 use k-of-n criteria over pairs, plus leave-one-R-out (or leave-one-geometry-out) robustness for slopes. No p-values are attached to them.
- **Crash rates.** A beta-binomial model is fitted by maximum likelihood across pairs, with a profile-likelihood interval: 99% for confirmatory criteria, 95% otherwise. If the estimated overdispersion is 0, Clopper–Pearson on the pooled counts is used.
- **Permutation tests.** P6a (10,000 permutations) and P1c (exact).
- **Bootstrap.** Used only for descriptive bands on curves: 20,000 resamples of pairs within each regime.

### 9.3 Curves and descriptive analyses

**Psychometric curves.** For each agent × snapshot × approach × θ × width:
- quantities: v_e, v_min, d_b, d_resp, v_peak, peak util, line-apex offset and Δt_corner;
- x-axes: log R_seg and log v_min*;
- overlays: the nominal key (band = τ), the backed-off key, the s-OCP, and the QSS curves on the centreline and the minimum-curvature line;
- shading by the agent's regimes, with R_DEV R_seg quantiles marked on the axis;
- covariates shown on every curve: v_b/v_b* and v_top.

**Error curves.** e_ve, e_vmin, e_vmin^line, Δd_b and Δt_corner against R, and heatmaps of e_ve over (R × checkpoint).

**Crash probability against R:**
- per-cell beta-binomial intervals;
- R50 for each (agent, v0, D, θ), reported as the empirical bracket between grid values (primary) and from a Firth-penalised logistic fit on log R_seg (secondary);
- tier B maps drawn as v0 × D panels, and against slack (D − d_b^phys(v0, v_e*(A1)))/v0;
- D = 100 m cells labelled "recovery test".

**Support.**
- Spearman correlation of \|e_ve\| and P_crash with the log joint-support density across pairs, for each agent.
- The same for the density of corners that are both tight and reached at 250 km/h or more.

### 9.4 Exclusions and sensitivity analyses (fixed in advance)

- **Removed from everything:** infeasible (bound) and unsolved cells.
- **Removed from confirmatory key-based errors:** multi-basin cells.
- **Removed from P3 and Δd_b:** forced onsets.
- **Analysed both ways:** runs with clipping in the edge, ego or context groups. If the two versions disagree, the version without them is primary.
- **Kept, with a sensitivity analysis that excludes them:**
  - unsettled runs (a regime is flagged if more than 20% of its runs are unsettled);
  - lock-limited cells.
- **Time-censored runs** keep the measures taken before the censoring point; Δt_corner is excluded for them.
- **Checkpoints that fail the gate** get crash curves only.

### 9.5 Power

The step-5 dry run fills in the power table, which is part of the tagged commit.

Expected pairs per approach:

| Agent | Bulk (A2) | Tail | Below support, not clipped | Clip-saturated |
|---|---|---|---|---|
| G1 | About 9 | 10 | 4 | 6 |
| G2 | About 11 | 8 | 4 | 4 |
| G3 | About 13 | 8 | 2–4 | ≥ 4 |

P1b for G3 may be "not testable". That outcome is pre-registered, not a failure.

### 9.6 What each outcome means

| Outcome | Meaning |
|---|---|
| P1a confirmed, P1-tail confirmed | The network reads R lawfully on typical approaches, and departs only for rare fast approaches. The fix is in the data: more tight-and-fast corners. |
| P1a refuted, "cautious but lawful" | A uniform safety margin, not misreading |
| P1a refuted with S < 0.8 | Poor reading even inside support. Step 2 should look for the cue actually used. |
| P1b confirmed | Departure below support, beyond the clipping artefact |
| P1b refuted | The network extrapolates. Crashes must then come from approach or context. |
| P2 confirmed | Late onset produces overspeed crashes; the onset share says whether it is detection or depth. |
| P2 refuted on onset | It finds the corner in time but sets the target too high or brakes too gently |
| P2 refuted on crash type | Over-brake or line/steer failures; ellipse use and line measures carry the explanation |
| P3 confirmed | The braking law is physical. Remaining error lies in the target speed. |
| P3 refuted with M_idx named | A preview-index rule: time-to-corner below 300 km/h, fixed distance above. Fragile when approach speeds are unusual. |
| P4 class 1 | The specialist is fragile off its manifold. Memorisation cannot be diagnosed. |
| P4 class 2 or 3 | Memorised. Its "4.8% from optimal" is a fact about Hockenheim only. Step 2 must use Hockenheim states. |
| P4 class 4 or 5 | It reads the road within its envelope, or beyond it |
| P5 asymmetric | Check the harness; otherwise a learned asymmetry, with its size reported |
| P6a refuted | Competence is not ordered by size; P6d asks whether order follows training frequency |
| P6c confirmed | Speed gains outrun corner reading late in training; judge interventions at late checkpoints |
| P7 confirmed | Short tight corners are partly a perception problem; a denser near-field preview is a candidate fix |
| P7 mixed | Plant or closed-loop filtering; step 2 must use closed-loop sensitivities |
| P7 refuted | The preview behaves like an ideal sampled observer |
| P8: low F for G2's tail crashes | The real crashes are not explained by reading a single corner. The ladder names the necessary context factor; step 1b is prioritised. |

## 10. Implementation (new files only; nothing else in the repo changes)

### Files

| File | Contents |
|---|---|
| `src/apexzero/interp/__init__.py` | — |
| `src/apexzero/interp/support.py` | Training-support statistics from 300 strips per grammar: R_seg quantiles, l_typ, onset rates, v_peak^QSS, kNN density; R_clip from obs_rms |
| `src/apexzero/interp/stimuli.py` | `CornerSpec`, `corner_profile`, `stimulus_bank`, `realised_geometry`, `stimulus_track`, `mirror`, `transplant` (guarded), `ladder`, `twin_fit`, `hockenheim_conditions`, `shift_corner` |
| `src/apexzero/interp/agents.py` | `load_agent`, `freeze_manifest` |
| `src/apexzero/interp/controls.py` | T, Q, Qp, c1–c3, K |
| `src/apexzero/interp/rollout.py` | `approach_state`, `phases`, `make_stimulus_rollout` (deterministic, native-noise or matched-noise; common random numbers with mirrored noise; chunking) |
| `src/apexzero/interp/measures.py` | `Trajectory`, adapters, model tables, `front_brake_demand`, `response_metrics`, `crash_type` |
| `src/apexzero/interp/answer_key.py` | `solve_min_time_pinned` (z_start pin, v_start guess, optional vx bounds), `solve_from_state`, `key_for`, backed-off, phase, null, re-anchored and viability keys, active sets, the infeasibility bound, validation V1–V11 |
| `src/apexzero/interp/probe.py` | Tier O scans; on-policy aliasing |
| `src/apexzero/interp/analysis.py` | Pair values, k-of-n rules, beta-binomial intervals, R50 and Firth fits, P2b decomposition, P3 models, P4 classifier, P6 t_R, P7 nulls, P8 |
| `scripts/psychophysics.py` | Subcommands `manifest`, `support`, `stimuli`, `controls`, `keys`, `validate`, `dryrun`, `rollouts`, `anchored-keys`, `probes`, `analyse [--confirmatory]`, `plots` |
| Tests | `tests/test_interp_{support,stimuli,controls,rollout,measures,answer_key,analysis}.py` |

**Outputs** go to:
- `outputs/psychophysics/manifest.json`
- `outputs/psychophysics/support/`
- `outputs/psychophysics/stimuli/`
- `outputs/psychophysics/keys/{nominal,backoff,phase,null,anchored,viability,socp}/`
- `outputs/psychophysics/controls/`
- `outputs/psychophysics/rollouts/<agent>/<ckpt>.npz`
- `outputs/psychophysics/probes/`
- `outputs/psychophysics/analysis/`

**Findings** go to `docs/findings/psychophysics-step1.md`.

### Steps

Each step's checks take at most about 2 CPU-minutes unless noted.

0. **Manifest.**
   - Every listed snapshot loads and gives a finite forward pass.
   - The manifest records, for every prospective snapshot, that it has not been evaluated.
   - G3 and S3 are added when they finish.
1. **Support and stimuli** (agent-free).
   - Build the support tables and freeze the regimes, l_typ, v_typ and R_clip for the available finals.
   - The stimulus builder's checks:
     - κ matches the trapezoid formula to 1e-12;
     - θ_eff is within 0.3° of nominal;
     - `track_arrays(to_track(·))` equals the bank's arrays;
     - the mirror is exact;
     - (w_in − 1)\|κ\| ≤ 0.6;
     - the length assertions hold;
     - an R_TEST name raises PermissionError;
     - the transplant κ matches the source to 1e-9;
     - twin fits are within tolerance.
2. **Harness.**
   - Mean-action states equal a `racing.step_no_reset` loop to 1e-6.
   - A batch equals the same runs done singly.
   - The mirror gate passes, with mirrored noise.
   - Common random numbers are verified; x64 is asserted off; the context checks pass.
   - **Pilot replication:** the width-11, 10 m-ramp mini-grid gives 24/24 completions for G1 and only R9/180/right crashing for G2. Any mismatch blocks everything after this step.
   - Throughput is benchmarked at B = 2048, T = 800 with 2 XLA threads.
3. **Measures.**
   - Synthetic trajectories with known answers: brake-demand onset, blips, catch-trial onset, censoring after a crash, η on both sides, crash typing.
   - The brake-demand re-implementation passes its test.
   - The model tables are built.
   - The TO-OCP solution and its environment replay give identical measures to 1e-6.
4. **Controls and keys** (agent-free).
   - Tune c_v for Q.
   - Lay out A2 with the pinned s-OCP.
   - Run validation V1–V11. This takes about 3 CPU-hours, in the background on 2 workers. Time it and project the budget.
   - Then solve P0 keys, then P1 keys, then P2 keys, applying the budget rule below.
   - Accept when V1, V4, V6 and V7 hold across all keys. The infeasible, unsolved and multi-basin lists and the τ table go into the manifest.
5. **Positive controls, classifier validation and dry run.**
   - Collect S1's real Hockenheim laps for K.
   - Run the Section 7 table and the dry run, and write the power table.
   - Hash the code, the stimuli and the key cache into the manifest.
   - Commit and **tag `psychophysics-prereg-v2`**.
6. **Rollouts**, with 3 processes × 2 XLA threads. Order:
   1. finals, deterministic;
   2. anchored keys (step 7);
   3. finals, stochastic;
   4. checkpoints;
   5. probes.
7. **Anchored keys,** with 5 workers, never alongside rollouts at full width. Order:
   1. re-anchored keys for A1 tail and bulk on the G1–G3 finals, plus viability solves (P0);
   2. specialists and tier B D = 600 (P1);
   3. checkpoints, under the P6 rule (P2).
8. **Analysis.**
   - Starts only when the G3 and S3 finals are in the manifest.
   - One run of `analyse --confirmatory` writes the decision table.
   - Then the exploratory analyses, the plots, and the findings document with its deviations log.

### Compute budget

There are about 6 free cores out of 12; the training queue runs throughout (S3, then v3, then ued-plr-maxmc-pilot).

| Stage | Items | CPU time | Priority |
|---|---|---|---|
| Pre-tag layout and validation | 33 A2 layout s-OCPs, 36 V8 s-OCPs, validation keys, 2 one-off compiles | About 3 CPU-h | P0 |
| Nominal keys, full recipe | Tier A 74, E1 real + isolated 30, nulls ≤ 28 (cheap) | About 11 CPU-h (about 350 CPU-s each) | P0 |
| Warm-start keys | Backed-off 66, phase-matched 60 | About 3 CPU-h | P0 |
| Anchored keys | Re-anchored A1 for the G finals, about 96; viability ≤ 200 | About 6.5 CPU-h | P0 |
| P1 keys | E1 twins 15, E2 39, E3 28; tier D s-OCPs ≤ 140; re-anchored for specialists ≤ 50 and tier B 108 | About 16 CPU-h | P1 |
| P2 keys | Tier C 23, recovery 12; checkpoint re-anchoring ≤ 216 (only if needed) | About 10 CPU-h | P2 |
| **Keys, total** | | **About 50 CPU-h.** P0 is about 24 CPU-h: 5 h wall on 5 workers, 7–10 h with the trainer running. Everything takes 10–20 h. | |
| Rollouts | 6 finals × about 21M steps, controls about 10M, checkpoints about 100M: about 240M steps | 45 min at 30k steps/s per process × 3; 90 min at 15k | — |
| Probes and analysis | Forward passes; beta-binomial fits; 20k bootstrap resamples | Minutes | — |

**Budget rule,** set from the timing measured in step 4:
1. If the projected total exceeds 60 CPU-h, drop the P2 items.
2. If P0 alone is projected above 35 CPU-h:
   - drop the tier A 180° cells with R ≥ 35.8 (5 geometries × 2 approaches);
   - keep phase-matched keys for 180° cells only.
3. If rollout throughput is below 15k steps/s per process, drop the checkpoint stochastic runs and the tier E matched-noise runs.

Every cut is logged in the manifest. Cuts made before the tag are part of the design.

### Scheduling and monitoring

- **No full-width overlap.** The key stage and the rollout stage never run at full width at the same time.
- **Fixed worker counts.** Load average was about 30 with the trainer alone, so it is not used as a throttle.
- **Commands.** Every stage runs as `uv run python scripts/psychophysics.py <stage>`. The existing terminal dashboard (`uv run python scripts/dashboard.py`) lists it automatically under "Other experiments".
- **Logging.**
  - Each stage logs to `runs/psychophysics-<stage>.log`, with a `done/total` line at least once a minute.
  - While it runs, it keeps a one-line note in `runs/activity.md`.

## 11. Changelog: v1 → v2

M = methodology review, O = optimal-control review.

| # | Issue | Decision | Where |
|---|---|---|---|
| M1 | The confirmatory family tests the wrong regime for the crash story | Accepted. P1-tail is now confirmatory. P8 (explained-crash fraction, using the ablation ladder) is a pre-registered endpoint. P2 is narrowed to the mechanism, and to G2's tail crashes only when P8 confirms. The bulk crashes are stated up front as predicted context. | §2.4 P1-tail, P2, P8 |
| M2 | Below-support saturation is an input-clipping artefact that collides with the OOD rule | Accepted. R_clip per agent; a clip-saturated regime; P1b on non-clipped pairs only; half-steps 8.4 and 10.5 added; 6.0 and 7.5 descriptive plus a no-cap variant; clipping accounted per channel group. | §2.3, §3.1, §3.5, §5, §9.4 |
| M3 | Approach speed confounds every corner measure | Accepted. Re-anchored keys at A1 and on tier B free approaches. A2 redefined so that its re-anchored key equals the nominal key. v_b/v_b* and v_top shown on every curve. | §2.2, §3.5, §6.1 |
| M4 | Survivorship bias | Accepted. v_e is the primary speed; overspeed crashes rank as +∞; a pair rule for partial crashes; crashed runs included in P3. | §2.4 P3, §9.1 |
| M5 | e_vmin confounds line strategy with severity reading | Partly accepted. Primary measures are v_e, Δt_corner and peak ellipse use; v_min is conditioned on the agent's own line with the model's steady-state limit. Rejected: a TO-OCP on the agent's own line, which needs a new path-constrained formulation and new compiles. | §5 |
| M6 | Brake onset is defined differently for agent and optimum | Accepted. Brake-demand onset for both; catch-trial response onset; corner regret and corner time with the null straight subtracted; −0.2 kept as secondary. | §5 |
| M7 | No positive controls through the identical harness | Accepted. Tape, QSS controller and planted-defect controllers with required verdicts that block the tag. | §7 |
| M8 | Synthetic-to-real transfer checked only on crash corners | Accepted. Synthetic twins for 13 crash and 13 matched non-crash corners (26, not about 30, to pair 1:1); pre-registered agreement and correlation criteria. | §3.4, P8 |
| M9 | Context can be tested now | Accepted. A 5-rung ablation ladder plus twin and mirror per crash pair; fidelity gate; YasMarina diagnosis. | §3.2, §3.4, P8 |
| M10 | Memorised and off-distribution cannot be told apart | Accepted, (a)–(e): corner swaps and the phantom index, near-manifold dose series, matched OOD distance with irrelevant perturbations, a memorisation index against the generalists, and classifier validation with the kNN memoriser and the QSS controller. | §2.4 P4, §3.3, §7 |
| M11 | Tier D conditions change several factors at once | Accepted. 2×2×2 factorial; a 100 m real lead-in gives every condition the same initial state; class precedence fixed; the unit of C defined. | §3.3, P4 |
| M12 | The answer key was seeded with G2 | Accepted. The G2 start is replaced by the agent-free Q controller; a symmetric re-solve rule; multi-basin cells excluded instead of inflating τ. Rejected: an s-OCP start on the minimum-curvature line, which needs a second reference frame; Q already gives an agent-free alternative basin. | §6.2, §6.3 |
| M13 | Phase handling, and a key-budget risk | Accepted. Phases defined at sighting speed; K = 4; phase-matched keys on the P2 cells; a half-step onset floor elsewhere; a budgeted plan. | §4, §6.1, §6.3, §10 |
| M14 | P7 thresholds uncalibrated; "integration" misreads an MLP | Accepted. Nulls from the key's phase SD, quantisation and the Qp ideal observer; fixed detrending; scans restricted to visited states, plus on-policy shift probes; "mixed" reworded as plant or closed-loop filtering. | P7, §7 |
| M15 | Problems with the P3 fit | Accepted. Regression on d_b^phys from the agent's own speeds; explicit physics, preview-index (with the d_max clip) and fixed-distance models compared by held-out error; crashed runs included. | P3 |
| M16 | Inference over a fixed deterministic grid | Accepted. k-of-n rules with leave-one-out robustness; beta-binomial intervals; the "refute if any" union replaced by single rules. Rejected: a jittered stimulus population, since every jittered stimulus needs its own key (≥ 10× key cost). | §2.4, §9.2 |
| M17 | S ill-conditioned in the tail and below support | Accepted. An extrapolation residual against the agent's bulk fit; S needs a log range of at least 0.15. | §2.2, P1b |
| M18 | Training support is joint, not marginal in R | Accepted. Joint kNN density covariate; P1a at A2 only, with A1 as a support manipulation; a single radius estimator (R_seg) everywhere. | §1.4, §2.3, P1 |
| M19 | A2 labelled by v0, but the agent chooses v_b | Accepted. A2 is the typical approach per corner size, with D set so that v_b* is within ±10% of v_typ. | §3.5 |
| M20 | Imposed D = 100 m lies outside the reset distribution | Accepted. Labelled a recovery test; slack axis; infeasible cells excluded from P_crash; bound screening. | §3.5, §6.4, §9.3 |
| M21 | The pilot contaminates G1 and G2 | Accepted. G1 and G2 labelled pilot-informed (secondary family); G3, S2 after 122.3M and S3 form the primary family; the manifest records that none was evaluated; step 8 waits for G3 and S3. | §1.3, §8 |
| M22 | P1c's 1/6 chance is wrong once ties are allowed | Accepted. At least one strict inequality, no reversal; exact label-permutation null; all-equal is inconclusive. | P1c |
| M23 | No cheap seed-robustness check for the generalists | Accepted. The 2-of-last-3-checkpoints rule; S1–S3 spread on tiers A and D. A second v2 seed is recommended, not scheduled, because the training queue is the user's call. | §8 |
| M24 | Stochastic crash rates confound reading with noise level | Accepted. Deterministic primary; a matched-noise condition; mirrored steering noise on mirror pairs. | §4, P5 |
| M25 | P6 confounded by overall pace | Accepted. Competence at A2; re-anchored keys at 4 checkpoints (or nominal, by rule); v_top and log_std as covariates; t_R = ∞ takes the maximum rank; P6b made exploratory. | P6 |
| M26 | The P2b decomposition depends on term order | Accepted. Symmetric exact form; a rule for blip-only runs. | P2b |
| M27 | P2's crash taxonomy leaves out slip and spin | Accepted. Typed by viability and ellipse rules whatever the termination cause; per-cell fractions with the median over cells. | §9.1, P2 |
| M28 | Forks to pin down before the tag | Accepted. Unsettled and time-censored rules; R_seg for regimes, with S fitted on log v*; errors never set to zero; V3 gaps estimated, not fixed. | §6.3, §6.5, §9.4 |
| M29 | Minor points | Accepted. 225° cells dropped; obs_noise = 0.01 run dropped; vx and \|v\| both reported; the specialist asymmetry made descriptive. | §3.5, §4, §5, P5 |
| O1 | Stimuli use corner shapes never seen in training | Accepted. Training-typical ramps from frozen l_typ; l_r = 10 m kept as the abrupt factor in E3; onset rate recorded and used in joint support. | §1.4, §3.1 |
| O2 | Centreline R does not measure severity, so S is not estimable | Accepted. v_min* as a second axis; S within θ with a minimum log range of 0.15; P1b uses residuals; P7's S contrast dropped; the dry run on the real design gives power. | §2.2, §7, §9.5 |
| O3 | In hairpins the key rides limits a noisy policy cannot use | Accepted. Active sets and multipliers logged; lock-limited flags; a backed-off key (α_max 8.5°, −0.35 m) judges underspeed and caution, and the nominal key judges overspeed. | §2.2, §6.1, §6.4 |
| O4 | P2's overspeed evidence ignores recoverability | Accepted. Viability solves give the point of no return; crashes typed as overspeed, over-brake or line/steer from ellipse use and brake demand; every termination flag logged. | §6.1, §9.1 |
| O5 | ±7.5% band and τ not calibrated | Accepted. b from the V8 time-cost rule; τ only from verified starts within 1e-4 (logged to 1e-3) of the best J. | §6.3, §6.5 |
| O6 | Braking definitions depend on speed | Accepted. Brake-demand onset and release (0.3 and 0.1); decision point from the command; brake-only depth with drag removed; identical for agents and keys. | §5 |
| O7 | P3's decision bands nearly touch | Accepted. The primary P3 now tests a slope of 1 on d_b^phys, with a pre-tag separation check of at least 0.25. The secondary β vs β* uses a ±0.1 band fitted by θ class. | P3 |
| O8 | Unclear which optimum backs which claim | Accepted. A claim-to-key table; min-time key in the agent's actuation; V3 split into objective and discretisation effects; far-field time lost replaces T100. | §5, §6.1, §6.5 |
| O9 | The window has no terminal value | Accepted. At least 15 s after s_c1; V2 on A2 and imposed cells too; stages ending before s_c1 + 300 m skipped. Rejected: the optional exact tail value, which needs a modified objective and new compiles; the 15 s tail and V2 bound the bias. | §4, §6.2, §6.5 |
| O10 | Solve-time budget unsupported | Accepted. Measured-cost projection with a pre-registered cut rule; IPOPT caps; neighbouring-R start replaces the mirror start; bound screening of D = 100 m. | §6.2, §10 |
| O11 | "No verified solution" treated as "infeasible" | Accepted with a point-mass bound; "unsolved" kept separate and excluded. Rejected: the L1 corridor-slack problem, which is a new formulation needing new compiles. | §6.4 |
| O12 | Some tier C cells are not the corners they are labelled as | Accepted. θR < 10 cells dropped; regimes by R_seg; tier O labels fixed the same way. | §3.5 |
| O13 | Policy noise and hidden grip shift expected responses | Accepted. exp(log_std) recorded per checkpoint and used as a P6b covariate; the backed-off key handles caution; the −1% expected bias stated. | §2.2, P6 |
| O14 | Details of the minimum-time solver | Accepted. ds 2.0 m everywhere (the knot spacing, within the recommended 1–2 m); cut at s0 − 20 m and pinned at the s0 node, with a 1e-9 assert on κ and w. | §6.2 |
| O15 | QSS is used for layout but is off at high speed | Accepted. Braking or flat classified from the key; QSS tables calibrated to the model for the controllers. | §2.3, §7 |
| O16 | Line and regret measures | Accepted. η normalised per side; regret only where a key exists at that phase. | §5 |
| O17 | Tier E arithmetic | Accepted. Every count recomputed: tier E is now 105 + 39 + 64 runs. | §3.5 |
| — | Freeze rule | Changed: agent-free keys and the dry run now come before the tag (needed for O2), and the tag is renamed `psychophysics-prereg-v2`. | Header, §10 |
## 12. Pre-tag rulings (2026-10-19)

The build (`docs/interp/IMPLEMENTATION.md`) raised these questions. Each ruling below is part of the design. Each was made before the tag and before any protocol stimulus was rolled out on a non-pilot agent.

| # | Question | Ruling |
|---|---|---|
| R1 | Real-lap rollouts for initial states (tiers D, E1, E2, the fidelity gate, G3's crash corners) are agent rollouts outside the two exceptions | **Allowed before the tag.** They repeat the standard R_DEV evaluation that training already ran for every agent (`eval_rdev.json` / `eval.json`), under the same convention as `eval/laps.py`. They put no protocol stimulus in front of an agent. Each one is logged in the manifest as a "standard evaluation (R1)" entry. |
| R2 | Crash corners labelled by the crash documents' raw ±60 m radius (YasMarina 12.7, Zandvoort 12.1) differ from the segmented corner that contains the crash (R_seg ≈ 20.4 and 27.5 m) | **The crash corner is the `corners.segment` corner that contains the crash point (D15). Its regime comes from its R_seg.** This is the single-estimator rule of 1.4, which already covers crash corners. Every crash corner's label is recomputed this way, and the P8 sets follow from the recomputed regimes. A neighbouring tighter corner lies inside the ladder's [s_c − 800, s_c + 400] window, so the context rungs still show its effect. **P8 thresholds become proportional:** *confirm* if at least ⌈0.6 n⌉ of the n tail units are reproduced *and* single-corner with twin agreement; *refute* if at most ⌊0.2 n⌋ are. For n = 5 this equals the original 3 and 1. If n < 3 the tail prediction is "not testable". |
| R3 | θ_eff misses 0.3° on 6 of 160 synthetic rows (R 6.0/180°, R 8.4/90°), from 2 m knot sampling of the trapezoid's kinks | **Accepted.** θ_eff is recorded per row and the rows are listed. The failing rows are clip-saturated or below support, which are descriptive regimes. |
| R4 | Twin fits miss the 2 m length tolerance on 5 of 13 crash corners and 10 of 13 E2 corners, because the real corner is longer than the longest symmetric trapezoid with its R_seg and angle | **Accepted:** twins match R_seg and the angle (to < 1e-3) and take the closest length. The length shortfall is listed for each twin and reported as a covariate in the twin-transfer analysis. No new shape is introduced. |
| R5 | l_typ and v_typ use corners turning at least 10°, and v_typ pools bins with fewer than 20 corners | **Accepted.** Without the filter, near-straight kinks at the 10 m ramp floor dominate the top bin, which contradicts the protocol's stated expectation (≥ 44 m at 75 m). |
| R6 | Every A2 run is flagged "unsettled", because the corner is visible from state 0 by design (D < d_max) and the pedal state starts at 0 | **The unsettled flag does not apply to cells whose corner is in view at the start (`sight_at_start`).** Their first-sight state is the designed initial state. The flag keeps its meaning on every other cell. |
| R7 | Budget rule 2 says "5 geometries", but at 180° only R 35.8, 44.7 and 55.9 exist with R ≥ 35.8 | **3 geometries × 2 approaches.** The count was a miscount. Cut cells are still rolled out, carry `no_key`, and are excluded from key-based errors. |
| R8 | The crash-typing ellipse search starts at s_c0 − 50 (M11) | Accepted |
| R9 | R_clip is printed as 50/(\|μ\| + 10σ); the sign that clips first gives 50/(10σ − \|μ\|) | **50/(10σ − \|μ\|)**, which is "the worse sign" as 2.3 intends |
| R10 | P4 uses only single-factor F1 conditions, and class 2 can be met vacuously (AD16) | Accepted. A vacuous class 2 is flagged in the details and reported. |
| R11 | P6a confirmation also needs a permutation p ≤ 0.05 (AD18) | Accepted |
| R12 | P1b's Kendall τ mixes keys at A1: below-support pairs on nominal e_ve, tail pairs on e_ve^RA | **Re-anchored A1 keys are extended to the non-clipped below-support cells (P0)**, so the whole A1 set uses e_ve^RA |
| R13 | Analysis decisions AD1–AD25 (`interp/analysis.py` docstring) | Signed off as written |
| R4a | R4 quoted the fitter's usual precision (< 1e-3) as the acceptance bar. One twin (Brands Hatch 1136 m, real L_seg 206 m against the 154 m triangle) misses the angle by 0.14° because its length cannot be matched | **Amended:** a twin shorter than its corner is accepted if its R_seg and angle are within the protocol's own twin tolerances (3.4: 2%, 1°). Achieved residuals are reported for every twin. |
| R14 | Q tuning (protocol 7) asked for 100% deterministic completion on tier A and on tier D condition (a). From S1's real-lap state at Hockenheim corner 6, Q starts at 295 km/h 85 m before a kink whose centreline radius is 19 m. A centreline driver's own QSS envelope there is 285 km/h, so no c_v completes it; S1 takes the kink on a line that straightens it | **c_v is the largest choice that completes every tier A run and passes P4's competence gate on tier D condition (a)** (C ≥ 0.9: corners completing in at least 3 of 4 phases). That gate is the one Q's verdict uses. Corners Q cannot complete are listed as limits of centreline driving. Implementation fix found along the way: Q's braking clip used only the front axle's friction room, so braking from a mid-corner start spun the car. It now uses the tighter axle. |
| R15 | Validation V1–V11 (step 4): V2, V5, V7 and V9 failed; b = 10% | **V2:** one A1 cell differs by 0.117 m/s (limit 0.1) between 600- and 800-step windows. A1 already uses W = 800, so the remedy is in place; the residual is listed. **V5:** some left/right pairs settle in different near-equal optima (relative J up to 5e-6; v_e up to 0.24 m/s). The mirror shortcut is blocked, so both directions are solved, as 6.5 prescribes. **V7:** v_e* moves by up to 0.67 m/s with the control-grid phase. Phase-matched keys are solved for every braking cell, as 6.5 prescribes, and every error, speed errors included, is judged against the key at the run's own phase when one exists. **V9 (investigated):** the backed-off constraint binds; the nominal key exceeds 8° slip for 4–18 steps in every validation cell. Ordering violations are at most 0.15 m/s on v_e* and 0.21 m/s on v_min*, within the near-equal-optimum spread V5 measured and an order of magnitude below b. "Fewer slip-active steps at 8°" cannot separate a key allowed 8.5° from one allowed 10°, so it is not used. Accepted. **b:** V8 clips the band at its 10% ceiling, because the median far-field cost of entering 10% slow is 0.043 s, below the 0.10 s rule; this is the pre-registered rule's outcome. **Cost:** keys cost about 7.5 CPU-h against the protocol's 50, so no budget cut fires. |
| R16 | Key acceptance (step 4): 523 P0 keys, 0 unsolved, 0 infeasible, 7 multi-basin; V6 lists 3 persistent violations after warm re-solves | **Listed, as 6.5 prescribes.** (1) and (2): v_min* at R 9.4 m exceeds R 10.5 m by 0.28 m/s (90°, A2, both directions). Typical ramps l_typ(R) change the corner's shape with R, so nominal R is not a strict severity order; every curve therefore also uses v_min* (2.2). (3): on one backed-off key (180°, A1, R 22.9 vs 28.6 m), the "last sustained run" onset rule picks a second brake run inside the corner. Backed-off keys set only the underspeed edge of the band, and no confirmatory test reads their onset. |

