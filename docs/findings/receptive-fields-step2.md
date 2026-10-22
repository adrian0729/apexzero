# Receptive fields, step 2: results (22 Oct 2026)

Design: [`docs/interp/receptive-fields-design.md`](../interp/receptive-fields-design.md) v3.3 (hypotheses in Section 7; implementation decisions in Sections 14–16). Pre-registration tag: `receptive-fields-prereg-v1`, on the record commit `0951f1e` (`src/apexzero/fields/prereg.json`). Confirmatory decisions: `outputs/fields/analyse/decisions.json`, sha256 `7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf`, committed in `b839a57`. Step 1: [psychophysics-step1.md](psychophysics-step1.md).

Every number below was read from, or recomputed from, the file named next to it. Files under `outputs/` and `runs/` are local experiment outputs and are not in git. The two exceptions are `outputs/fields/analyse/decisions.json` and `outputs/fields/tag.json`, which are committed. Labels used throughout:

- **Confirmatory**: the primary family (H2′ and H4, design Section 7), decided once by `analyse --confirmatory` and read from `decisions.json`.
- **Secondary**: pre-registered secondary hypotheses decided in the same run (H4-transfer; H2-speed, whose bands are descriptive).
- **Exploratory**: analyses that were not pre-registered as run. The main one is H4's frozen test with the gate2-trained failures ignored (`outputs/fields/exploratory/h4_exploratory.json`).
- **Descriptive**: everything else. These numbers decide nothing. **Computed here** marks numbers computed for this document from per-key output files, with the method stated; no network was run for them.

## 1. Summary

Step 2 asked how a trained racing policy's line and speed through a corner respond to small changes in the road ahead, and how that compares with the optimum of the policy's own training objective from the same state. Both sides were differentiated exactly: the networks by autodiff through their float64 closed loop, and the optimum (the TO-OCP) by barrier KKT sensitivity. The central quantity is the **edge-tracking gain G**: how far the line moves toward the inside per metre the road's inside widens, for a corner-wide trapezoid shift of the inside edge. Equivalently, when the inside edge intrudes by a metre, the line gives way by G metres. The optimum's G* is 1, because its line touches the inside edge. A width-blind line has G = 0. All networks were measured from the same 481 start states (the A2 keys). The tag was placed on 21 Oct at 23:45. The confirmatory analysis ran once, on 22 Oct from 05:43 to 06:06.

**Primary family (confirmatory).**
- **H2′, the edge-tracking deficit: no deficit in any of the 8 decided generalists.** The ninth, G3s2, is descriptive and is also far from the deficit band.
  - Median G is 0.760–0.851, and every 95% CI lies within [0.665, 0.867]. A deficit needed the CI's upper end to be at most 0.5.
  - G3 is **lawful**, with CI [0.825, 0.867] inside [0.8, 1.2].
  - Seven networks are **inconclusive**: their CIs lie between the two bands and straddle 0.8.
  - G3s2 is **descriptive (gate2-trained)**. Its band decision would have been inconclusive.
  - Variant level (2 of 3 seeds): v1 **inconclusive**, v2 **inconclusive**, v3 **seed-specific**.
  - The prior, that the crash ladders implied a deficit, is not borne out for slow, corner-wide edge shifts.
- **H4, the width-training intervention: not testable.**
  - Two of its 18 networks failed the post-tag autodiff check gate2-trained: `dr-pilot-v3-tilt50a2-s2`, which is the same checkpoint as G3s2, and `dr-realw-v1-s1`.
  - Each failed at one key of 12, on the c_min field only, with relative errors of 1.7e-3 and 1.0e-3 against a limit of 1e-3.
  - Under the rules fixed before the tag (design Sections 15–16 and the frozen analysis code; Section 5.3), a gate2-trained failure makes the network descriptive and H4 not testable.
  - H4-transfer (secondary) is not testable for the same reason: it shares the v3 baseline `dr-pilot-v3-tilt50a2-s2`.

**Exploratory, not confirmatory: H4's frozen test with the two gate failures ignored** (Section 6).
- **Real-width training lowered G**, the opposite of H4's prediction.
  - In both variants every real-width network has a lower G than every baseline.
  - The opposite one-sided p is 2.0e-4, the smallest attainable with 4,900 relabellings. That would have been a "refute", and it is unchanged with the seed-0 baselines included (p 6.3e-5).
  - The separation holds with the two failing networks removed, and on the keys common to all 18 networks.
- **Real-width training raised the buffer B**, the line's inside clearance (two-sided p 0.002; 6.3e-4 with seed 0). The B test is not in the note that chose the exploratory H4, so treat the test and its direction as chosen after the confirmatory results were known.
- **The leave-out networks** (real widths from other circuits only) move the same way, by about 40% as much. Neither difference is significant: p 0.12 for G and 0.18 for B. With the seed-0 baselines included, G's p falls to 0.047, so the frozen rule would report it as "inconclusive (not robust to including the seed-0 baselines)". The leave-out shifts cannot be told apart from either no transfer or full transfer of the in-sample shift.
- In these runs the real-width networks kept more inside margin (higher B) and followed corner-wide edge shifts slightly less (lower G). Whether the margin explains their higher completion in the width test is not tested here. The gate rule makes all of this exploratory, and the code and hardware confound of the width test applies in full.

**Most informative descriptive results.**
- **On real crash roads the line barely follows the edge (H3; G2's three tail crash units).**
  - When the real width profile replaces the constant one, the line moves away from the intruding inside edge by a median 19% of the edge's move at the crash point (range −2% to 53%, 12 unit-phases).
  - Only a median 7% is a response to the inside edge's own change; the rest responds to the outside width.
  - G2's gain for corner-wide shifts on synthetic corners is 0.83.
  - The linear crash prediction λ_lin is within 0.1 of the observed crash fraction λ* in 8 of 12 cases.
- **Corner-wide responses are near-linear and symmetric up to 2 m (dose-response).**
  - The networks' c_min responds to inside shifts of ±0.1 to −2 m with median linearity ratios of 0.87–1.14, and narrowing and widening act alike.
  - The optimum's inside contact stays active at every amplitude.
  - So the gap between corner-wide tracking (about 0.8) and real-road following (about 0.2) does not appear to be an amplitude effect, though the dose test covers only synthetic, corner-wide trapezoid shifts and H3's linearity rests on 3 units of one network. It points to the shape or locality of the edge change, or to differences between the real and the synthetic corners. That is a hypothesis for an exploratory follow-up (Section 10), not a finding.
- **The line follows a whole-corridor shift only partly.** T = G + G_out is 0.72–0.82, and every network is labelled "mixed": neither width-blind nor corridor-anchored.
- **The networks brake 11–16 m earlier than the optimum after fast approaches** (H1-fast). All 9 generalists' CIs lie below 0.
- **Asymmetry shows in margin more than in tracking.** The learned left-right asymmetry of step 1 shows up mainly in B (indices up to 0.5 m). G's index is at most 0.08.
- **An implementation gap.** The pre-registered anchored-key version of H2′ and the earlier snapshots' G came out empty, because a filter in the frozen code drops every anchored row (Section 7.9). Per-key medians computed here are given instead.

## 2. Design in brief

**Question.** Step 1 found that 159 of 160 real-road corner crashes leave by the inside edge, where that edge moves by metres within the last 100 m, and that holding the width constant rescues them. Step 2 asks for the mechanism: when the inside edge moves, does the network's line follow it, as the optimum's does?

**States** (design Section 2; `outputs/fields/keys/lists.json`).
- **A2 keys, the primary set.** 481 verified answer keys at approach A2, the typical approach speed for the corner's size: 142 nominal and 339 phase-matched. Every network starts from the same state x0 with prev_action 0, so all networks are compared at identical states.
  - 475 are primary: the key's W is the cell's W, so the 6 tier V horizon twins are left out.
  - 427 of those are not capped. In the other 48, the generator's inside cap narrows the inside.
  - The 427 form 62 mirror pairs over 48 geometries (R, θ).
- **Anchored keys, the secondary set.** 9,525 keys at each agent's own fast-approach states, at A1 and on tier B: 8,136 for the 12 finals and 1,389 for the earlier snapshots of G1, G2 and G3. These are descriptive.
- **H3 keys.** 24 newly solved keys on G2's three crash units: 12 base and 12 anchored, all verified (`outputs/fields/h3-keys/keys.json`). Only R-dev circuits were used.

**Outcomes and edge measures** (design Sections 3, 4 and 15).
- **c_min** is the minimum inside clearance w_in − 1 − σn over [s_c0 − 50, s_c1 + 50] m. For the network it is measured against the termination boundary. For the optimum it includes the corridor's 3 cm margin.
- **v_apex** is the minimum speed, and **s_b** the brake onset; both are step 1's definitions.
- **The trapezoid shift.** The inside edge is moved by Δ·h(s): flat from s_c0 − 20 to s_c1 + 20 m, with 20 m ramps on each side.
- **G = 1 − dc_min/dΔ** is the line's move toward the inside per metre the inside widens.
  - Equivalently, when the inside edge intrudes by Δ, the clearance falls by (1 − G)Δ.
  - It is defined only where the argmin lies on the trapezoid's plateau.
- **B = c_min(0)** is the buffer, in metres. To first order, a corner-wide narrowing Δ crashes the car when (1 − G)Δ > B.
- **G_out** is the line's outward move per metre the outside widens. **T = G + G_out** is the line's move per metre when the whole corridor shifts toward the inside.
  - A line anchored to the corridor has T = 1, and a width-blind line T = 0.
  - Labels on T's CI: "width-blind" if the upper end is ≤ 0.5, "corridor-anchored" if the CI lies within [0.8, 1.2], "mixed" otherwise.
- **Unit of analysis.** Each measure is computed per key, summarised per mirror pair (geometry × approach) with both directions weighted equally, and then per network as the median over geometries. The 95% CI comes from a cluster bootstrap over geometries.

**Two pipelines** (design Section 5).
- **Network.** A float64 closed loop built from exact upcasts of the float32 road and parameters, in a fixed 256-step `lax.scan`. One `jax.vjp` of the outcome vector gives per-sample impulse responses on the 2 m grid of curvature, left width and right width. Every perturbation used here is a linear combination of them.
- **Optimum.** The stored TO-OCP solution, re-polished once on this machine. Its sensitivities come from the condensed barrier KKT system with one adjoint solve per outcome.

**Gates and controls before the tag** (design Section 8; results in Section 3).
- Gate 0: KKT residual of the stored keys.
- Gate 1: KKT sensitivities against finite-difference re-solves, with planted bugs that must fail.
- Gate 2: network autodiff against finite differences, on controls.
- Gate 3: teacher-forced wiring on every key with a stored rollout, with planted wiring errors that must fail.
- Gate 4: G* within [0.98, 1.02] on every A2 key.
- Gate 5: KKT replay through the network pipeline.
- Planted-field controls RF-C1 to RF-C5, and unlawful readers judged against Qp.
- The float32/float64 branch-tolerance calibration.

**Hypotheses** (design Section 7; bands fixed at the tag).

| Hypothesis | Family | Networks and data | Decision rule |
|---|---|---|---|
| H2′ edge-tracking deficit | Primary | 9 step-1 generalists, finals; primary non-capped A2 keys without κ clipping | On median G's CI: deficit if upper ≤ 0.5; lawful if within [0.8, 1.2]; over-tracking if lower ≥ 1.2; else inconclusive. Variant: 2 of 3 seeds (R18) |
| H4 width training | Primary | v1 and v3: 4 real-width seeds against baseline seeds 1–4 per variant (seed 0 in the sensitivity analysis) | One-sided exact permutation test on median G, real > baseline, labels exchanged within variant (4,900 relabellings). Confirm if p ≤ 0.05; refute if the opposite p ≤ 0.05 |
| H4-transfer | Secondary | 8 leave-out runs against v3 baseline seeds 1–4 | As H4 (495 relabellings) |
| H2-speed | Secondary (descriptive bands) | 9 generalists, A2 | r_in = Δv_apex,net / Δv_apex,opt for the inside shift; under / lawful / over at [0.5, 2] |
| H1-typical, H1-fast, H3, T, B, dose, asymmetry, anchored H2′, earlier snapshots, specialists | Descriptive | — | Reported, not decided |

**Coverage rule** (design Section 15). A network whose share of excluded primary non-capped A2 keys exceeds 30%, counting every reason, is descriptive for H2′. A network that fails gate2-trained is descriptive. H4 and H4-transfer are not testable if any of their networks fails the coverage rule, and the frozen analysis code (`analysis.h4_test`) treats a gate2-trained failure the same way (Section 5.3).

**Networks** (`src/apexzero/fields/prereg.json`, `networks`; every checkpoint is loaded by its recorded path and sha256).

| Group | Runs | Role |
|---|---|---|
| Step-1 generalists | G1, G1s1, G1s2 (`dr-pilot-200M`, `-s1`, `-s2`); G2, G2s1, G2s2 (`dr-pilot-v2-tilt30`, …); G3, G3s1, G3s2 (`dr-pilot-v3-tilt50a2`, `-s1`, `-s2`) | H2′. Six of them are also H4 baselines |
| v1 and v3 baselines | `dr-pilot-200M` and `dr-pilot-v3-tilt50a2` seeds 0–4 | H4 (and H4-transfer for v3) |
| Real-width | `dr-realw-v1-s0..3`, `dr-realw-v3-s0..3` | H4 |
| Leave-out | `dr-realw-loo-f1-s0..3`, `dr-realw-loo-f2-s0..3` | H4-transfer |
| Specialists | S1, S2, S3 | Descriptive (anchored keys only) |

That makes 29 distinct trained checkpoints on the A2 keys. The analysis lists 35 entries, because the six generalists that are also H4 baselines appear under both names.

**Pre-registration record and tag.**
- The record binds:
  - the key lists with content hashes, and the gate2-trained and dose key lists;
  - every checkpoint's sha256;
  - the stamped pre-tag outputs;
  - the calibrated branch tolerance;
  - the code hash `651e7eb4…` (all of `src/` and `scripts/fields.py`);
  - the design-document hash `cf00943c…`.
- `receptive-fields-prereg-v1` is an annotated tag (object `9ec48f30…`, tagger date 2026-10-21 23:45:52 −04:00) on commit `0951f1e`. It was pushed with the same object on origin (`runs/activity.md`, 21 Oct 23:46).
- The first post-tag stage pinned it in `outputs/fields/tag.json` at 23:48:05, and the pin was committed at once (`17e347e`).

**Post-tag procedure** (design Section 16; Section 9.2 records how the stages were actually scheduled):
1. harness-f32;
2. tag-pin commit;
3. gate2-trained;
4. h3-keys;
5. the field stages: fields-a2, fields-anchored, fields-earlier;
6. dose and h3-lambda;
7. a non-confirmatory analyse (coverage and plumbing only, `decisions_dry.json`);
8. `analyse --confirmatory`, once;
9. a commit of `decisions.json`.

Every one of the 110 stage stamps under `outputs/fields/` (91 sidecar `*.provenance.json` files and 19 stamps embedded in stage outputs; the `wip/` backup excluded) carries the same code hash `651e7eb4…`, no deviation, no changed paths and no partial run. The 83 post-tag stamps also carry the tag; the other 27 are pre-tag. No code changed after the tag: `git diff receptive-fields-prereg-v1 HEAD -- src scripts` is empty. The post-tag commits are the tag pin, the costate design draft (`705b221`, docs only) and the decisions.

## 3. Pre-tag gates and controls

All of these ran in the stamped pre-tag pass (21 Oct 22:52–23:43, `runs/fields-pretag2.log`). The record binds them.

| Check | Result | Source |
|---|---|---|
| Gate 0, KKT residual | 10,006 of 10,006 keys pass; max free-variable stationarity residual 3.3e-7 (limit 1e-6) | `outputs/fields/keys/gate0.json` |
| Re-polish | 10,006 succeeded (at most 5 iterations); 5,384 flagged (informational); J moved by at most 1.5e-8 relative; largest state movement anywhere in a window 2.3 cm | `outputs/fields/repolish/summary.json` |
| Gate 1, KKT against re-solves | 50 keys, 1,121 scored vectors, 879 zero fields. Error median 1.4e-7, p95 1.8e-4, max 0.055 (pass: median < 1e-3, p95 < 1e-2). All six strata covered. Planted bugs fail: dropped width mixed term p95 0.38; flipped lam_g median 0.77; swapped widths median 1.0 | `outputs/fields/gates/gate1.json` |
| Gate 2, network autodiff against FD (controls) | 8 rows (random actor ×2, edge keeper, linear driver ×3, hybrid linear+random-MLP ×2), all pass; largest relative error 3.3e-6 (hybrid, cell 66, c_min) | `outputs/fields/gates/gate2.json` |
| Gate 3, teacher-forced wiring | 13,908 keys over the 12 step-1 agents (874–1,247 each), 0 failures; largest mean action error 4.9e-5 (limit 1e-4). Planted wiring errors (wrong obs_rms, prev_action zeroed, accel zeroed) fail on every agent | `outputs/fields/gates/gate3.json` |
| Gate 4, the optimum's G* | 481 of 481 A2 keys in [0.98, 1.02]; \|G* − 1\| ≤ 1.7e-9 | `outputs/fields/gates/gate4.json` |
| Gate 5, KKT replay | 50 keys; largest relative errors v_apex 2.2e-10, s_b 1.1e-13, c_line 6.2e-9 (limit 1e-3); state tangent 8.3e-11 (limit 1e-8); a planted 10 m road misalignment fails on all 50 | `outputs/fields/gates/gate5.json` |
| Optimum's fields | 10,006 keys; 0 failures, 0 inertia, adjoint, off-plateau or outside-Gate-4 exclusions | `outputs/fields/opt-fields/summary.json` |
| Qp's fields | 475 primary A2 keys, 0 pipeline failures (s_b valid on 471) | `outputs/fields/qp-fields/summary.json` |
| Controls | All pass: RF-C1, RF-C2, RF-C3, RF-C4, RF-C5 and the unlawful readers. RF-C5's edge keeper gives G = 1.002–1.006 on cells 340, 342, 344 and 345; its fixed-n line gives \|G\| ≤ 1.7e-10. RF-C2's AD-safe Q reproduces its closed form at cosine ≥ 0.99998 and amplitude 0.997–1.000; the frozen Q's amplitude is 0.85, which is why the AD-safe family exists | `outputs/fields/controls/controls.json` |
| Phase ceiling | 113 cells. Median over cells of the per-cell median ρ: s_b 0.984, v_apex 0.978, steer 0.993. Lowest cell median: v_apex −0.91 | `outputs/fields/phase-ceiling/phase_ceiling.json` |
| Branch tolerance | BRANCH_TOL: c_min 3.3e-3 m, v_apex 1.36e-3 m/s, s_b 7.4e-2 m. That is 5 × the controls' largest float32/float64 outcome spread per unit of teacher-forced action error, times the trained agents' largest Gate 3 error, 4.9e-5 (G2s2), over 10 control rows | `outputs/fields/calibration/branch_tol.json` |
| Loop length | The longest stored run needs 210 of the 256 loop steps. H3's units need up to 439 of 512 | `outputs/fields/keys/loop_audit.json` |

Two notes:
- **The c_min tolerance changed before the tag, as pre-registered.** It was 7.7e-4 m at the end of the first, unstamped pre-tag pass (`outputs/fields_prestamp_2026-10-21/calibration/branch_tol.json`), and 5.7e-4 m earlier in that pass (Section 9.1 gives the full history). It rose to 3.3e-3 m when Section 15's hybrid control, the trained architecture inside the loop, joined the calibration. Its c_min slope is 13.4, against at most 3.1 for the other rows.
- **The step-1 Q family is not AD-correct** (design Section 14). Step 2's controls therefore use value-identical AD-safe replacements.

## 4. Post-tag checks

### 4.1 gate2-trained (autodiff against finite differences on every H2′, H4 and H4-transfer network)

- **Method.** Gate 2's rule was applied at 12 primary non-capped A2 keys listed in the record, along 12 directions: 5 corner bumps of κ, 5 corner bumps of the inside width, and both trapezoids. Four outcomes were scored.
  - A comparison counts only when the +h, 0 and −h runs share their discrete structure, and FD(h) and FD(h/2) agree to O(h²).
  - A row passes when the field vector's relative error on the counted comparisons is below 1e-3.
  - Under Section 16's strict rule, a network passes only if all 12 of its rows pass.
- **Run.** 35 network entries (29 distinct checkpoints), 23:48–00:17 (`runs/fields-gate2-trained.log`; `outputs/fields/gate2-trained/gate2_trained.json`). No loop was exhausted.
- **Result: 32 of 35 entries pass, or 27 of 29 checkpoints.** The `runs/activity.md` entry at 00:18 says "33/35", which is wrong: G3s2 and `dr-pilot-v3-tilt50a2-s2` are one checkpoint, run twice with identical results, and both entries fail.

The two failures:

| Network | Key (cell, phase) | Outcome | Relative error (limit 1e-3) | c_min directions counted | Other outcomes at that key |
|---|---|---|---|---|---|
| `dr-pilot-v3-tilt50a2-s2` (= G3s2) | `2f3c4793…` (cell 379, phase 0) | c_min | 1.70e-3 | 9 of 12; 3 failed the FD step-pair check | v_apex 4.7e-6, s_b 1.4e-8, steer 6.7e-12 |
| `dr-realw-v1-s1` | `54d2398d…` (cell 383, phase 2) | c_min | 1.01e-3 | 3 of 12; 9 failed the FD step-pair check | v_apex 1.3e-5, s_b 2.9e-10, steer 4.1e-12 |

What the failures look like:
- **They sit at the top of a continuous distribution.** Over all 420 key-level c_min field errors (35 entries × 12 keys, so the six duplicated checkpoints count twice) the relative error has median 2.6e-7 and p95 1.6e-4. The three failing entries (two checkpoints) are the only ones above 1e-3. The next largest c_min errors are 8.8e-4 (G1s2), 7.1e-4 (`dr-realw-loo-f2-s0`, at the same key `2f3c4793…`) and 5.8e-4 (G1, also at `2f3c4793…`). All of those passed.
- **They fall where c_min is rough.** At both failing keys, most or many of the other c_min directions could not be counted, because FD(h) and FD(h/2) disagreed beyond O(h²). That is a sign of kinks, such as a switching argmin or saturation, near the base point.
- **What the gate cannot say.** It cannot tell whether the excess on the counted directions comes from the autodiff field or from the finite-difference reference at such a point. Either way, the pre-registered rule applies.

### 4.2 Wiring for networks without step-1 rollouts (harness-f32)

- The 20 networks without step-1 rollouts (the H4 and H4-transfer runs other than the six generalists) were run in the float32 harness from the 475 primary A2 starts (23:46:26–23:48:37; `outputs/fields/harness-f32/summary.json` and its provenance stamp; the tag was pinned at 23:48:05 during this run).
- fields-a2 compared each network's float64 loop with these runs for branch consistency: termination step, saturation set, outcome events, and outcome values within BRANCH_TOL. The 9 generalists were compared with step 1's stored float32 rollouts.
- A failing key is a pipeline exclusion for that outcome. On c_min, which feeds G, there are 0–2 per network (Section 4.3). All are branch-consistency failures: the saturation set, an event or an outcome value differs beyond BRANCH_TOL between the float32 and float64 runs (`outputs/fields/fields-a2/*.jsonl`, `branch`).
- **Reproducibility check (computed here).** fields-a2 computed the six shared checkpoints twice, under the generalist name and under the run name. The two runs of each finished 1.8–3.2 h apart (provenance end times), some while other stages ran in parallel. Per-key G, B and the impulse-array hashes are identical on all 481 keys for all six pairs (`outputs/fields/fields-a2/G1.jsonl` against `runs_dr-pilot-200M.jsonl`, and so on).

### 4.3 Coverage of G per network (primary non-capped A2 keys, n = 427)

Source: `decisions.json` (`coverage`, `<network>@final`, `G`). The share counts every reason. No network exceeds the 30% limit. There were no inertia, Gate 3 or Gate 4 exclusions.

| Network | gate2-trained | Max rel. error (all outcomes) | Keys used | Excluded | Invalid outcome | Argmin off plateau | κ clipping | Pipeline (branch) |
|---|---|---|---|---|---|---|---|---|
| G1 (= `dr-pilot-200M`) | pass | 5.8e-4 | 404 | 5.4% | 5 | 2 | 16 | 0 |
| G1s1 (= `-s1`) | pass | 1.2e-4 | 354 | 17.1% | 4 | 53 | 16 | 0 |
| G1s2 (= `-s2`) | pass | 8.8e-4 | 388 | 9.1% | 12 | 14 | 12 | 1 |
| G2 | pass | 4.6e-4 | 414 | 3.0% | 12 | 1 | 0 | 0 |
| G2s1 | pass | 8.9e-4 | 416 | 2.6% | 9 | 1 | 0 | 1 |
| G2s2 | pass | 1.5e-4 | 410 | 4.0% | 12 | 0 | 4 | 1 |
| G3 (= `dr-pilot-v3-tilt50a2`) | pass | 3.0e-4 | 376 | 11.9% | 1 | 48 | 0 | 2 |
| G3s1 (= `-s1`) | pass | 1.1e-4 | 423 | 0.9% | 4 | 0 | 0 | 0 |
| G3s2 (= `-s2`) | **fail** | 1.7e-3 | 418 | 2.1% | 8 | 1 | 0 | 0 |
| `dr-pilot-200M-s3` | pass | 3.6e-4 | 374 | 12.4% | 12 | 33 | 8 | 0 |
| `dr-pilot-200M-s4` | pass | 1.8e-4 | 374 | 12.4% | 8 | 33 | 12 | 0 |
| `dr-pilot-v3-tilt50a2-s3` | pass | 3.9e-5 | 408 | 4.4% | 8 | 10 | 0 | 1 |
| `dr-pilot-v3-tilt50a2-s4` | pass | 2.1e-5 | 417 | 2.3% | 9 | 1 | 0 | 0 |
| `dr-realw-v1-s0` | pass | 2.2e-4 | 399 | 6.6% | 16 | 0 | 12 | 0 |
| `dr-realw-v1-s1` | **fail** | 1.0e-3 | 367 | 14.1% | 16 | 30 | 12 | 2 |
| `dr-realw-v1-s2` | pass | 2.1e-4 | 385 | 9.8% | 13 | 17 | 12 | 0 |
| `dr-realw-v1-s3` | pass | 1.3e-4 | 407 | 4.7% | 8 | 0 | 11 | 1 |
| `dr-realw-v3-s0` | pass | 2.2e-4 | 421 | 1.4% | 4 | 1 | 0 | 1 |
| `dr-realw-v3-s1` | pass | 1.2e-4 | 423 | 0.9% | 4 | 0 | 0 | 0 |
| `dr-realw-v3-s2` | pass | 3.6e-4 | 423 | 0.9% | 4 | 0 | 0 | 0 |
| `dr-realw-v3-s3` | pass | 2.6e-4 | 421 | 1.4% | 5 | 1 | 0 | 0 |
| `dr-realw-loo-f1-s0` | pass | 2.2e-4 | 422 | 1.2% | 4 | 1 | 0 | 0 |
| `dr-realw-loo-f1-s1` | pass | 9.1e-5 | 410 | 4.0% | 0 | 17 | 0 | 0 |
| `dr-realw-loo-f1-s2` | pass | 7.1e-5 | 422 | 1.2% | 4 | 0 | 0 | 1 |
| `dr-realw-loo-f1-s3` | pass | 4.7e-4 | 330 | 22.7% | 4 | 92 | 0 | 1 |
| `dr-realw-loo-f2-s0` | pass | 7.1e-4 | 415 | 2.8% | 12 | 0 | 0 | 0 |
| `dr-realw-loo-f2-s1` | pass | 7.3e-4 | 423 | 0.9% | 4 | 0 | 0 | 0 |
| `dr-realw-loo-f2-s2` | pass | 1.6e-4 | 414 | 3.0% | 4 | 9 | 0 | 0 |
| `dr-realw-loo-f2-s3` | pass | 4.6e-5 | 417 | 2.3% | 8 | 2 | 0 | 0 |

Sources: `outputs/fields/gate2-trained/gate2_trained.json` for gate2-trained; `decisions.json` (`coverage`) for the rest.

- **Exclusion shares** run from 0.9% to 22.7%, and pipeline failures from 0 to 2 per network. `decisions.json` lists nothing missing for H2′, H4 or H4-transfer.
- **"Argmin off plateau"** means the c_min argmin falls where the trapezoid is not flat (e < 0.99), so G is not defined there. It is the largest reason in the networks with the most exclusions.
- **Near-ties** (two local minima of c_min within 0.05 m) are flagged on 0–18 keys per network and kept, as pre-registered (`decisions.json`, `networks`, `near_ties`).

## 5. Confirmatory decisions

Run: `analyse --confirmatory`, 22 Oct 05:43:35–06:05:55 (the provenance stamp in `decisions.json`). `confirmatory` and `decided` are true, the deviation list is empty, and HEAD was `705b221` with no changed paths. The run happened once. The file was committed at 06:06:05 (`b839a57`), and its sha256 is the one given at the top.

### 5.1 H2′ per network

Source: `decisions.json` (`H2`; the optimum's row from `optimum.G_star`). All rows use the primary non-capped A2 keys. The pooled column weights the two turn directions equally. The asymmetry index is m_L − m_R over mirror partners.

| Network | Decision | Median G [95% CI] | Geometries | Pairs defined (of 62) | Left | Right | Asymmetry index [CI] | Excluded |
|---|---|---|---|---|---|---|---|---|
| G1 | inconclusive | 0.790 [0.742, 0.836] | 42 | 56 | 0.803 [0.768, 0.834] | 0.763 [0.714, 0.802] | +0.028 [+0.007, +0.041] | 5.4% |
| G1s1 | inconclusive | 0.811 [0.789, 0.837] | 36 | 45 | 0.785 [0.699, 0.803] | 0.838 [0.803, 0.867] | −0.035 [−0.074, +0.003] | 17.1% |
| G1s2 | inconclusive | 0.825 [0.724, 0.859] | 42 | 54 | 0.807 [0.728, 0.842] | 0.846 [0.720, 0.867] | −0.036 [−0.059, +0.004] | 9.1% |
| G2 | inconclusive | 0.832 [0.791, 0.854] | 45 | 59 | 0.840 [0.807, 0.859] | 0.830 [0.746, 0.870] | +0.020 [−0.013, +0.070] | 3.0% |
| G2s1 | inconclusive | 0.803 [0.767, 0.840] | 45 | 59 | 0.856 [0.818, 0.878] | 0.765 [0.616, 0.804] | +0.075 [+0.031, +0.109] | 2.6% |
| G2s2 | inconclusive | 0.778 [0.764, 0.835] | 44 | 58 | 0.775 [0.725, 0.796] | 0.856 [0.764, 0.881] | +0.017 [−0.069, +0.055] | 4.0% |
| G3 | **lawful** | 0.851 [0.825, 0.867] | 39 | 49 | 0.829 [0.811, 0.849] | 0.847 [0.824, 0.879] | −0.029 [−0.049, −0.013] | 11.9% |
| G3s1 | inconclusive | 0.760 [0.665, 0.806] | 47 | 61 | 0.807 [0.751, 0.850] | 0.659 [0.528, 0.711] | +0.082 [+0.021, +0.142] | 0.9% |
| G3s2 | **descriptive (gate2-trained)**; band decision inconclusive | 0.774 [0.731, 0.799] | 46 | 60 | 0.721 [0.661, 0.803] | 0.779 [0.708, 0.843] | −0.021 [−0.084, +0.016] | 2.1% |
| Optimum G* | (reference) | 1.000 [1 − 6e-12, 1 + 3e-12] | 48 | 62 | 1.000 | 1.000 | 1e-16 | — |

**Reading.**
- No network comes near the deficit band, which needs the CI's upper end at or below 0.5: the lowest upper end is 0.799 (G3s2) and the lowest lower end 0.665 (G3s1).
- Only G3's CI clears 0.8. G2 (lower end 0.791) and G1s1 (0.789) miss the lawful band by about 0.01.
- **G3's lawful decision may partly be selection** (computed here; descriptive, and the decision stands under the frozen rule).
  - "Argmin off plateau" depends on the outcome: a key drops out when the network's own line comes closest where the trapezoid is not flat. G3 loses 48 keys this way and keeps 39 of 48 geometries (only G1s1 keeps fewer, 36).
  - Those 48 keys are tight corners: 44 at R 6.0–14.6 m and 4 at R 22.9 m. On them, every other generalist's per-key median G (0.22–0.82) is below its median over all its own kept keys (0.73–0.83); G1, for example, gives 0.704 against 0.789.
  - On the 306 primary non-capped keys with none of Section 4.3's exclusions in any of the nine generalists, the per-key medians (plain medians over keys, not pair-weighted, no CI; from `outputs/fields/fields-a2/*.jsonl`) run from 0.794 (G3s1) to 0.848 (G2s1), and G3's 0.840 is not distinctive: G2s1 0.848, G2 0.842, G1s2 0.841, G2s2 0.826, G1 0.817, G1s1 0.816, G3s2 0.811, G3s1 0.794.
- In one sentence: the generalists move their line about 80% as far as the inside edge moves, for a corner-wide shift, in all three variants.

### 5.2 Variant decisions (R18: the same decision in at least 2 of 3 seeds)

Source: `decisions.json` (`H2_variants`).

| Variant | Seed 0 | Seed 1 | Seed 2 | Variant decision |
|---|---|---|---|---|
| v1 | inconclusive | inconclusive | inconclusive | **inconclusive** |
| v2 | inconclusive | inconclusive | inconclusive | **inconclusive** |
| v3 | lawful | inconclusive | descriptive (no verdict) | **seed-specific** |

- In v3, a descriptive seed counts as disagreeing (`analysis.variant_rule`). Had G3s2 counted, its band decision (inconclusive) would have made v3 inconclusive.
- Neither label is a claim, so the gate2-trained failure changes no claim.
- **Shared networks.** Six of these nine networks are also H4 baselines (design Section 7, multiplicity).

### 5.3 H4 and H4-transfer: not testable, and exactly why

Source: `decisions.json` (`H4`, `H4_transfer`, `descriptive`, `gate2_trained`).

- **H4:** "not testable (gate2-trained: runs/dr-pilot-v3-tilt50a2-s2, runs/dr-realw-v1-s1)".
- **H4-transfer:** "not testable (gate2-trained: runs/dr-pilot-v3-tilt50a2-s2)".

The chain of rules, all fixed before the tag:
1. Design Section 15 makes gate2-trained a post-tag check on every H2′, H4 and H4-transfer network: "A network that fails is descriptive."
2. Design Section 16 makes it strict: "A network passes only if every one of its 12 keys passes Gate 2's rule."
3. Design Section 15, under the coverage rule: "H4 and H4-transfer are not testable if any of their networks fails the rule." `analysis.h4_test` treats a gate2-trained failure the same way (`_untestable`).

`dr-pilot-v3-tilt50a2-s2` is a v3 baseline (seed 2), so it sits in both H4 and H4-transfer. `dr-realw-v1-s1` is a v1 real-width seed, in H4 only. Section 4.1 shows each failure: one key of 12, the c_min field, a relative error of 1.70e-3 and 1.01e-3 against 1e-3. Apart from this, both networks are within every rule: their excluded shares are 2.1% and 14.1%, both below 30%.

## 6. Exploratory: H4's test with the gate2-trained failures ignored

### 6.1 Status

- **The data are not exploratory; the decision to use them is.** The test, statistic, relabellings, sensitivity analysis and bands are H4's frozen ones (`analysis.h4_test`). The only change is that the two gate2-trained failures are not treated as making H4 untestable.
- **This is post hoc in a specific sense.** The pre-registered rule made H4 not testable, and running the test anyway is a choice made after the gate result was known.
- **The choice to run it was noted before any A2 field existed.** The `runs/activity.md` entry of 22 Oct 00:18 records the gate result and says "Exploratory H4 to be reported separately". That file is a local, untracked notes file with minute-resolution times, which was edited throughout the run, so a reader cannot check the entry from the repository. The outputs corroborate it: fields-a2 started at 00:17:37, its first impulse array (G1, `outputs/fields/fields-a2/arrays/G1_200000000/029d95f0….npz`) was written at 00:19:57, after 00:18:59 at the latest, and its first network (G1) finished at 00:29:16 (`outputs/fields/fields-a2/G1.jsonl.provenance.json`). So the choice was made before any network's G was known.
- **The B version was chosen later.** The same test on the buffer B is not mentioned in that note. Treat it as chosen after the confirmatory results were known.
- **Where the results are and how they were checked.** `outputs/fields/exploratory/h4_exploratory.json` (written 06:07) names `decisions.json` by its sha256 and carries no provenance stamp. The code that wrote it was first run inline; it is now committed as `scripts/fields_exploratory_h4.py`, which checks the decisions file's sha256 and reproduces every number in the file. Every number in it was recomputed here from the per-network medians in `decisions.json` (`edge_descriptive`, pooled), by exact enumeration, and matches to every printed digit.

### 6.2 The per-network medians

Sources: `decisions.json` (`edge_descriptive`, pooled medians). The two gate2-trained failures are marked †.

| Variant | Arm | Network | G | B (m) |
|---|---|---|---|---|
| v1 | real width | `dr-realw-v1-s0`, `-s1`†, `-s2`, `-s3` | 0.757, 0.736, 0.745, 0.738 | 1.59, 1.43, 1.63, 1.43 |
| v1 | baseline | `dr-pilot-200M` (seed 0 = G1), `-s1`, `-s2`, `-s3`, `-s4` | 0.790, 0.811, 0.825, 0.810, 0.855 | 0.99, 1.23, 1.32, 1.15, 1.24 |
| v3 | real width | `dr-realw-v3-s0`, `-s1`, `-s2`, `-s3` | 0.756, 0.696, 0.667, 0.677 | 1.55, 1.55, 2.12, 1.72 |
| v3 | baseline | `dr-pilot-v3-tilt50a2` (seed 0 = G3), `-s1`, `-s2`†, `-s3`, `-s4` | 0.851, 0.760, 0.774, 0.769, 0.800 | 0.96, 1.66, 1.49, 1.35, 1.28 |
| v3 (leave-out) | real width, other fold's circuits | `dr-realw-loo-f1-s0..3` | 0.800, 0.758, 0.742, 0.750 | 1.49, 1.72, 1.55, 1.32 |
| v3 (leave-out) | real width, other fold's circuits | `dr-realw-loo-f2-s0..3` | 0.811, 0.651, 0.723, 0.727 | 1.73, 1.94, 1.39, 1.36 |

**G separates completely in both variants**, with or without seed 0.
- In v1, the real-width range 0.736–0.757 lies below the baseline range 0.790–0.855.
- In v3, the real-width range 0.667–0.756 lies below 0.760–0.851. The v3 gap is thin: 0.756 (`dr-realw-v3-s0`) against 0.760 (baseline s1).

### 6.3 The tests

Source: `outputs/fields/exploratory/h4_exploratory.json`, verified here.
- The statistic is the sum over variants of (mean real-width − mean baseline).
- "p" is the share of relabellings with a statistic at least as large (real > baseline). "p opposite" is the share at least as small.
- The decision test leaves out the seed-0 baselines; the sensitivity analysis includes them.

| Measure | Test | Statistic (per stratum) | Relabellings | p | p opposite | Reading under H4's frozen bands |
|---|---|---|---|---|---|---|
| G | H4, decision test | −0.158 (v1 −0.081, v3 −0.077) | 4,900 | 1.0 | **2.0e-4** (the minimum attainable) | refute |
| G | H4, with seed 0 | −0.166 (v1 −0.074, v3 −0.092) | 15,876 | 1.0 | 6.3e-5 (the minimum) | refute; robust to seed 0 |
| G | per variant, decision test | v1, v3 | 70 each | 1.0 | 0.014 each (the minimum, 1/70) | — |
| B | H4, decision test | +0.575 (v1 +0.283, v3 +0.292) | 4,900 | **0.0010** | 0.999 | real > baseline, p ≤ 0.05 (no pre-specified direction) |
| B | H4, with seed 0 | +0.721 | 15,876 | 3.1e-4 | 0.9997 | robust to seed 0 |
| B | per variant, decision test | v1, v3 | 70 each | v1 0.014, v3 0.057 | — | — |
| G | H4-transfer, decision test | −0.031 | 495 | 0.88 | 0.12 | inconclusive |
| G | H4-transfer, with seed 0 | −0.046 | 1,287 | 0.95 | 0.047 | refute, so reported as "inconclusive (not robust to including the seed-0 baselines)" |
| B | H4-transfer, decision test | +0.120 | 495 | 0.18 | 0.82 | inconclusive |
| B | H4-transfer, with seed 0 | +0.216 | 1,287 | 0.069 | 0.93 | inconclusive |

**Reading.**
- **G.** By H4's own rule, real-width training would have been a **refute**: it lowered the edge-tracking gain, by 0.08 in each variant's mean.
- **B.** It raised the buffer by 0.28–0.29 m in each variant's mean. v3 alone is not significant (p 0.057). B's direction was not pre-specified; read two-sided (the 4-against-4 permutation distribution is symmetric), its p is 0.002. With seed 0 the two-sided p, computed here by the same enumeration, is 6.3e-4.
- **The leave-out networks.** Their shifts go the same way and are smaller, about 40% of the in-sample v3 shift: −0.031 against −0.077 for G, +0.120 against +0.292 m for B. The ratios were computed here from the statistics in the table, with the same baseline seeds.
  - Neither shift is significant against the baselines (p opposite 0.12 for G, p 0.18 for B).
  - Nor can they be told apart from the in-sample real-width v3 shift. Computed here, by an exact permutation of the 8 leave-out against the 4 real-width v3 per-network medians (495 relabellings): leave-out minus real-width v3 is +0.046 for G (two-sided p 0.14) and −0.17 m for B (p 0.26).
  - So the ratio of about 40% is a ratio of two noisy point estimates. The data fit anything from no transfer to full transfer of the in-sample shift.

### 6.4 Why it is still informative, and how far

The gate failures cannot be what produced the direction of the G result:
- **One failure falls in each arm.** `dr-pilot-v3-tilt50a2-s2` is a v3 baseline, and `dr-realw-v1-s1` is a v1 real-width seed. A common bias in the two failing networks' G would widen the gap in one variant and narrow it in the other. It could not by itself produce the same direction in both.
- **The separation does not depend on them.** Computed here: with both networks removed, G still separates completely in both variants.
  - v1: real-width 0.738–0.757 against baseline seeds 1–4 at 0.810–0.855.
  - v3: real-width 0.667–0.756 against baseline seeds 1, 3 and 4 at 0.760–0.800.
  - With complete separation in both strata, the frozen test on that reduced design would give its minimum p opposite, 1/(35 × 35) = 8.2e-4. That is arithmetic; the test was not run.
  - For B, v1 still separates (real 1.43–1.63 m against baseline 1.15–1.32 m). v3 does not: baseline s1's 1.66 m exceeds two real-width values. The means still differ by +0.31 m (1.74 against 1.43 m).
- **Their own values.** `dr-pilot-v3-tilt50a2-s2`'s G (0.774) lies inside its arm's range (0.760–0.851). `dr-realw-v1-s1`'s (0.736) is its arm's lowest, 0.002 below the next seed, so it is the real-width v1 value that widens the gap most. Removing it leaves the separation intact (real-width v1 0.738–0.757, above).
- **The failures are small and look like derivative accuracy at a rough key, not broken wiring** (Section 4.1). On the same 12 keys, both networks' v_apex, s_b and steer fields agreed with finite differences to 1.5e-4 or better, and their other 11 c_min rows passed. On c_min, their float64 loops matched the float32 runs on all but 2 (`dr-realw-v1-s1`) and 0 (G3s2) of the 427 keys.

A further check, on the key sets:
- **The separation does not depend on the networks' different key sets either** (computed here). The arms' exclusions differ: baseline v1 seeds 1, 3 and 4 lose 33–53 keys off the plateau, and the real-width v1 networks 0–30 (Section 4.3).
  - Method: the 285 primary non-capped A2 keys with none of Section 4.3's exclusions in any of the 18 H4 networks (`outputs/fields/fields-a2/*.jsonl`; the per-network kept counts reproduce Section 4.3's "Keys used"); per network, the plain median over those keys (not pair-weighted); then H4's frozen statistic and enumeration.
  - G and B still separate completely in both variants. G: statistic −0.168, p opposite 2.0e-4. B: statistic +0.668, p 2.0e-4. Both are the minimum attainable with 4,900 relabellings. This check is exploratory like the rest of this section.

What this does not do:
- **It does not make H4 confirmatory.** The rule was fixed for a reason. Autodiff agreement was not shown to the pre-registered standard on those two networks, and the decision stands as "not testable".
- **It does not remove the design's known confound** (design Section 7, H4). The real-width runs used later code, probably on a different GPU: the hardware was not recorded, but they logged 0.56–0.63M steps/s against 0.42–0.45M for baseline seeds 1–4 ([width-test.md](width-test.md)).
  - The leave-out runs also used the later code. Their smaller shifts cannot separate a code-only explanation from an effect of real widths: they are neither significant nor distinguishable from the in-sample shift (Section 6.3). With hardware also differing and 4 seeds per arm, nothing here excludes a code-only explanation.
- **The scope is narrow.** G is measured for slow, corner-wide shifts on synthetic A2 corners. Whether real-width training changes how the line follows local edge changes on real roads, the H3 kind of change, is not measured here (Section 10).
- **B's test had no pre-specified direction or role.**
- **Shared networks.** Six of the ten baselines (G1, G1s1, G1s2, G3, G3s1, G3s2) are H2′ networks. This exploratory result and the H2′ decisions are therefore not independent.

## 7. Descriptive results

### 7.1 T and G_out

Source: `decisions.json` (`edge_descriptive`, pooled; `optimum.T_star`).
- **Every network's T is labelled "mixed".** Pooled medians run from 0.722 to 0.816 over all 35 entries. For the 9 generalists they run from 0.746 (G3s1) to 0.816 (G3), and every CI lies within [0.691, 0.847] (lowest lower end G2s1, highest upper end G3).
- **The optimum's T* is 1.000** (CI within 3e-12 of 1). No network is width-blind (T ≤ 0.5), and none is corridor-anchored.
- **G_out is small.** It runs from −0.080 to +0.014 for the ten baselines, where a wider outside moves the line slightly inward. It runs from −0.037 to +0.054 for the eight real-width networks and from −0.052 to +0.107 for the leave-out networks.
- **Real-width networks rearrange tracking without changing T.** They have lower G (Section 6) and higher G_out, so T barely changes (real-width 0.722–0.794, baselines 0.746–0.816). Their line is placed a little less by the inside edge and a little more by the outside, while how far it follows a whole-corridor shift is unchanged.

### 7.2 The buffer B

Source: `decisions.json` (`edge_descriptive`, `B`, pooled medians, metres).

| Group | B (median over geometries) |
|---|---|
| 9 generalists | 0.96 (G3) – 1.66 (G3s1); G1 0.99, G2 1.09 |
| v1 and v3 baselines, seeds 3–4 | 1.15–1.35 |
| Real-width | 1.43–2.12 |
| Leave-out | 1.32–1.94 |

- At its closest point to the inside, a network's line keeps about 1–2 m from the termination boundary.
- The optimum's c_min is 0.03 m, its corridor margin: its inside contact is active (Gate 4).
- **Illustration** (arithmetic on medians, not a measure; a ratio of medians is not a median of ratios). With G ≈ 0.8 and B ≈ 1–1.7 m, the first-order rule (1 − G)Δ > B puts the crash point of a *corner-wide* narrowing at about 5–9 m. That is far beyond anything the A2 corners contain. The real-width networks' lower G and higher B give a similar figure.

### 7.3 H1-typical: curvature reading on typical approaches (descriptive since v3.1)

Source: `decisions.json` (`fields.<network>.H1_typical`, `H1_typical_vs_Qp`, `Qp_vs_optimum`).
- **What is reported.** Pooled medians [CI] of each network's s_b and v_apex curvature fields, against the optimum's, on its bulk A2 cells (13–21 geometries for s_b, 23–32 for v_apex).
  - g is the signed projection gain.
  - |loc| is the centroid offset in metres.
  - Δρ is the shape specificity: ρ with the key's own optimum minus ρ with optima of far geometries.
- **Coverage.** s_b is invalid on 9–13% of bulk keys. v_apex is invalid on at most 0.7%.

| Network | s_b: g | s_b: \|loc\| (m) | s_b: Δρ | v_apex: g | v_apex: \|loc\| (m) | v_apex: Δρ |
|---|---|---|---|---|---|---|
| G1 | 1.16 [0.72, 1.35] | 17.4 | 0.21 [0.10, 0.29] | 0.75 [0.67, 0.77] | 10.3 | 0.36 [0.21, 0.59] |
| G1s1 | 0.85 [0.61, 1.11] | 18.5 | 0.14 [0.06, 0.24] | 0.72 [0.68, 0.81] | 11.3 | 0.35 [0.22, 0.56] |
| G1s2 | 0.71 [0.62, 1.06] | 18.0 | 0.14 [0.10, 0.29] | 0.72 [0.61, 0.82] | 13.2 | 0.42 [0.31, 0.57] |
| G2 | 0.96 [0.64, 1.36] | 20.3 | 0.11 [0.05, 0.26] | 0.68 [0.56, 0.80] | 10.7 | 0.35 [0.21, 0.55] |
| G2s1 | 0.84 [0.56, 1.35] | 20.2 | 0.13 [0.09, 0.27] | 0.71 [0.58, 0.75] | 13.4 | 0.33 [0.20, 0.56] |
| G2s2 | 0.84 [0.63, 1.59] | 16.9 | 0.15 [0.07, 0.27] | 0.71 [0.62, 0.82] | 13.3 | 0.36 [0.21, 0.50] |
| G3 | 0.89 [0.68, 1.24] | 26.9 | 0.12 [0.07, 0.22] | 0.79 [0.59, 0.91] | 10.5 | 0.24 [0.16, 0.38] |
| G3s1 | 0.95 [0.77, 1.31] | 24.7 | 0.10 [−0.01, 0.14] | 0.69 [0.57, 0.74] | 10.9 | 0.24 [0.18, 0.38] |
| G3s2 | 0.88 [0.69, 1.79] | 29.1 | 0.07 [0.04, 0.16] | 0.73 [0.63, 0.79] | 9.9 | 0.24 [0.18, 0.38] |
| Qp, the lawful QSS reader (v1 / v2 / v3 bulk cells) | 0.62 / 0.57 / 0.54 | — | 0.41 / 0.36 / 0.31 | 0.30 / 0.29 / 0.27 | — | 0.41 / 0.39 / 0.33 |

Against Qp directly:
- The networks' s_b gain is 1.16–1.97 times Qp's, with the CI's lower end above 1 in 8 of 9 (G1s2: [0.93, 1.42]).
- Their v_apex gain against Qp is 0.81–1.05, and every CI contains 1.

Against the blind-pedal control (each network with its pedal replaced by its own recorded tape):
- The paired score on gain (−|g − 1|) favours the network: by 0.41–0.56 for s_b (every CI above 0) and by 0.49–0.64 for v_apex (CI above 0 in 7 of 9; G2s1's contains 0 and G2s2's lower end is 0.00).
- The shape score favours it for v_apex (0.27–0.36) but not for s_b (CIs contain 0).

**Reading.**
- On typical corners, brake onset responds to preview curvature with about the optimum's gain (0.71–1.16), more strongly than Qp's.
- Apex speed responds at about 0.7 of the optimum's.
- The s_b fields are centred 17–29 m away from the optimum's, beyond the 12 m descriptive reference.
- Projections are not transitive, because the field shapes differ, so the vs-Qp gains are not ratios of the vs-optimum gains.

### 7.4 H1-fast: fast approaches (anchored keys; descriptive since v3.2)

Source: `decisions.json` (`H1_fast`). Units: s_b,net − s_b* in metres, where negative means the network brakes earlier. v_apex,net − v_apex* is in m/s. "Gain" is the s_b field's curvature gain against the optimum. Pooled over A1 and tier B.

| Agent | Geometries | s_b,net − s_b* | v_apex,net − v_apex* | Gain | Step 1's expected direction |
|---|---|---|---|---|---|
| G1 | 24 | −15.8 [−16.7, −13.3] | −2.7 [−4.9, +0.4] | 1.00 [0.64, 1.26] | s_b consistent; v_apex undetermined |
| G1s1 | 24 | −14.2 [−14.9, −13.9] | −1.8 [−3.7, +0.3] | 1.04 [0.98, 1.36] | as G1 |
| G1s2 | 24 | −11.1 [−12.6, −9.0] | −3.3 [−5.2, +0.1] | 1.03 [0.90, 1.33] | as G1 |
| G2 | 26 | −14.1 [−17.5, −13.1] | −2.4 [−4.0, +0.8] | 1.09 [0.77, 1.27] | s_b **inconsistent** with "about 0"; v_apex consistent |
| G2s1 | 26 | −14.5 [−15.1, −13.0] | −2.9 [−3.9, +1.4] | 1.06 [0.91, 1.26] | as G2 |
| G2s2 | 25 | −12.0 [−12.5, −9.2] | −3.3 [−3.9, +1.2] | 1.16 [0.92, 1.39] | as G2 |
| G3 | 26 | −11.5 [−12.0, −10.3] | −2.4 [−3.8, −0.6] | 1.31 [1.11, 1.55] | both consistent |
| G3s1 | 26 | −15.3 [−16.1, −14.8] | −1.8 [−3.9, +1.6] | 0.95 [0.83, 1.88] | s_b consistent |
| G3s2 | 26 | −14.9 [−15.7, −13.1] | −1.6 [−3.2, +1.2] | 1.29 [1.04, 1.53] | s_b consistent |

- **Every generalist brakes 11–16 m earlier than the optimum after a fast approach.** This matches step 1's direction for G1 and G3: cautious, not fast.
- **It also holds for all three v2 networks.** Their expected "about 0" was not met.
- **Apex speed is lower than the optimum's by 1.6–3.3 m/s.** The CI excludes 0 only for G3.
- **The curvature gain of brake onset is about 1** (0.95–1.31).
- These signs could be computed from step 1's files before the tag, so they are not prospective (design Section 7).

### 7.5 H2-speed: how width moves apex speed (secondary; descriptive bands)

Source: `decisions.json` (`fields.<network>.H2_speed`). r_in is the network's apex-speed response to the inside trapezoid shift divided by the optimum's.

| Network | r_in [CI] | Band label |
|---|---|---|
| G1 | 0.91 [0.71, 1.02] | lawful |
| G1s1 | 0.82 [0.26, 1.00] | inconclusive |
| G1s2 | 0.77 [0.54, 1.13] | lawful |
| G2 | 0.66 [0.23, 0.87] | inconclusive |
| G2s1 | 0.83 [−0.10, 1.05] | inconclusive |
| G2s2 | 0.74 [0.21, 0.94] | inconclusive |
| G3 | 0.73 [−1.43, 1.02] | inconclusive |
| G3s1 | 0.59 [−0.23, 0.98] | inconclusive |
| G3s2† | 0.64 [0.03, 0.78] | inconclusive |

† G3s2 failed gate2-trained, so under design Section 15 it is descriptive. The frozen code's H2-speed label does not apply that gate, so the label shown is the code's band label as computed.

- The apex speed responds to inside width in the optimum's direction, at a median 0.59–0.91 of its size.
- Most CIs reach below 0.5, so most labels are inconclusive. No network is "under" or "over".

### 7.6 Dose-response: finite amplitudes (computed here from `outputs/fields/dose/`)

**Design.** A stratified 400-key subsample was drawn on the `dose_sample` stream: 140 A2 keys and 260 anchored keys (`prereg.json`, `dose_sample`). The inside trapezoid was shifted by +0.1, +0.3, +1.0, −0.1, −0.3, −1.0 and −2.0 m, where + widens.
- The optimum was solved by warm continuation (`dose/optimum.json`).
- The networks were run as closed loops on the shifted road. Each of the 27 dose networks ran on the 140 A2 keys, plus its own anchored keys for the 9 generalists (`dose/networks.json`).
- The code defines the ratios. The linearity ratio is (y(a) − y(0)) / (a · dy/dΔ), and the narrowing/widening asymmetry is (y(−a) − y(0)) / −(y(a) − y(0)). Both are 1 for a linear response.
- The medians below were computed here, over keys whose base c_min is defined and which do not crash at any widening. A key that crashes at a narrowing amplitude has no ratio there, so the medians at −1 and −2 m cover only the keys that survived that amplitude (the crash counts are in the last column).

**The optimum.**
- 2,799 of 2,800 solves converged. One failed (restoration failed at +1 m, key `40925bba…`), and 75 needed a restart.
- In every converged solve but two, the optimum's c_min stays at 0.03 m to within 6e-10: its inside contact stays active, so G* = 1 holds at finite amplitudes from −2 to +1 m.
- The two exceptions are at +1 m widening, where the contact released (c_min 0.22 m, 0.19 m above the corridor margin).

**The networks** (c_min; generalists shown, ranges over all 21 distinct networks below).

| Network | Keys (valid base) | Linearity at −2 m | at −1 m | at +1 m | Asymmetry at 1 m | Crashed at −1 m / −2 m |
|---|---|---|---|---|---|---|
| G1 | 157 | 1.05 | 1.02 | 1.01 | 0.98 | 2 / 15 |
| G1s1 | 160 | 1.06 | 1.02 | 0.98 | 1.02 | 2 / 6 |
| G1s2 | 149 | 0.99 | 0.98 | 1.00 | 1.00 | 1 / 4 |
| G2 | 156 | 1.04 | 1.00 | 1.03 | 0.99 | 2 / 5 |
| G2s1 | 146 | 1.06 | 1.02 | 0.97 | 1.04 | 0 / 3 |
| G2s2 | 156 | 1.14 | 1.06 | 0.93 | 1.15 | 0 / 8 |
| G3 | 157 | 0.92 | 0.99 | 1.04 | 0.89 | 4 / 5 |
| G3s1 | 155 | 1.00 | 0.99 | 1.00 | 0.99 | 1 / 4 |
| G3s2 | 151 | 1.08 | 1.02 | 0.99 | 1.01 | 2 / 9 |

- **Ranges over the 21 distinct networks** (2,972 network-keys). Median linearity is 0.96–1.04 at ±0.1 and ±0.3 m, 0.93–1.07 at ±1 m and 0.87–1.14 at −2 m. Median asymmetry is 0.97–1.00 at 0.1 m, 0.95–1.02 at 0.3 m and 0.86–1.15 at 1 m.
- **Per key the spread is wider.** For example, the 10th–90th percentile range of the −2 m linearity ratio is 0.26–1.34 for G3 and 0.28–2.09 for `dr-pilot-200M-s4`.
- **The first-order crash rule is a rough guide.** Pooled over the 21 networks, B/(1 − G) put the crash within the tested range (≤ 2 m) on 121 keys; 91 keys actually crashed at −1 or −2 m, and 51 did both.
- **Reading.** For corner-wide shifts up to 2 m, a network's clearance responds nearly linearly and about equally to narrowing and widening. The local G is therefore a fair summary of metre-scale corner-wide shifts.

### 7.7 H3: the crash link on G2's real crash corners (descriptive, not prospective)

**Method** (design Section 7; `outputs/fields/h3-lambda/h3.json`). G2 ran from its own lap states on each unit's constant-width rung, with widths w(λ) = w_cw + λ(w_real − w_cw).
- λ* is the smallest λ on the 0.05 grid at which G2 crashes.
- λ_lin = c_min(0) / (−dc_min/dλ) is the first-order prediction, from per-sample impulse responses.
- dc_min/dλ splits into the edge's own move (c_min's explicit term) and the line's move.
- All three corners turn right, so the inside is the right edge.
- **Computed here:** the line's move is split further into the part driven by the right (inside) width and the part driven by the left (outside) width. That is dc_right − edge move and dc_left, which sum to the line's move exactly.

| Unit (cell) | Phase | B0 = c_min at constant width (m) | λ* | λ_lin | Edge move (m per unit λ) | Line move: total | from the inside width | from the outside width | Line / \|edge\| |
|---|---|---|---|---|---|---|---|---|---|
| Norisring 874 (584) | 0 | 0.79 | 0.55 | 0.56 | −3.02 | 1.59 | 1.14 | 0.44 | 0.53 |
| | 1 | 0.81 | 0.50 | 0.39 | −3.16 | 1.09 | 0.45 | 0.63 | 0.34 |
| | 2 | 0.79 | 0.45 | 0.40 | −3.21 | 1.24 | 0.65 | 0.59 | 0.39 |
| | 3 | 0.78 | 0.50 | 0.45 | −3.16 | 1.43 | 0.91 | 0.51 | 0.45 |
| Melbourne 2531 (605) | 0 | 0.67 | 0.95 | 0.95 | −0.69 | −0.01 | −0.03 | 0.02 | −0.02 |
| | 1 | 0.67 | 0.95 | 1.02 | −0.65 | −0.00 | 0.01 | −0.02 | −0.00 |
| | 2 | 0.66 | 1.00 | 1.11 | −0.60 | 0.00 | 0.04 | −0.04 | 0.01 |
| | 3 | 0.67 | 1.00 | 1.24 | −0.56 | 0.02 | 0.10 | −0.08 | 0.03 |
| Catalunya 4168 (612) | 0 | 0.65 | 0.75 | 0.97 | −0.85 | 0.18 | −0.00 | 0.18 | 0.21 |
| | 1 | 0.64 | 0.75 | 0.71 | −1.09 | 0.19 | 0.08 | 0.11 | 0.17 |
| | 2 | 0.64 | 0.75 | 0.75 | −1.04 | 0.19 | 0.04 | 0.15 | 0.18 |
| | 3 | 0.64 | 0.75 | 0.83 | −0.95 | 0.18 | 0.02 | 0.16 | 0.19 |

Unit names and cells: step 1's `outputs/psychophysics/stimuli/cells.json` and `stimuli.json` (the `const_width` rungs). The 24 H3 keys all verified (`outputs/fields/h3-keys/keys.json`).

**Reading.**
- **The line hardly follows the edge.** At the point of closest approach, the real-road inside edge comes in by 0.56–3.21 m per unit λ. The line moves away by a median 19% of that, with a range of −2% to 53%.
  - The part that answers the inside edge's own change is a median 7% (range −4% to 38%).
  - The rest answers the outside width's change, since the real profile changes both sides.
  - For comparison, G2's G for a corner-wide shift on synthetic corners is 0.83 [0.79, 0.85] (Section 5.1).
- **The response stays linear up to the crash.** λ_lin is within 0.1 of λ* in 8 of 12 cases and within 0.25 in all, with a median |λ_lin − λ*| of 0.06. The same holds in the dose-response.
- **Each unit fails differently.**
  - Norisring: the edge intrudes by about 3 m. The line gives way by a third to a half of that, and the 0.8 m buffer is used up at about half the real change.
  - Melbourne: the line barely moves (−0.01 to +0.02 m per unit λ). The real road's intrusion, 0.56–0.69 m, about equals the 0.67 m buffer, so G2 crashes only at λ ≈ 0.95–1.
  - Catalunya: the line gives way by about a fifth, almost all of it answering the outside width.
- **Scope.** This is three units of one network, and the four phases of a unit are not independent. G2's rollouts on these units were seen before the design, so H3 is not prospective (design Section 7).

### 7.8 Earlier snapshots: how the fields develop over training

**From `decisions.json` (`earlier`; H1-fast on each snapshot's own A1 anchored keys).**

| Snapshot | Keys | s_b,net − s_b* (m) | v_apex,net − v_apex* (m/s) | Gain |
|---|---|---|---|---|
| G1 @ 73.7M | 160 | −20.9 [−21.7, −19.4] | −4.4 [−5.1, +0.4] | 0.89 [0.69, 1.34] |
| G1 @ 109.9M | 159 | −15.7 [−18.3, −13.5] | −4.5 [−5.0, +0.4] | 1.04 [0.98, 1.30] |
| G1 @ 134.1M | 160 | −16.0 [−18.4, −15.3] | −5.4 [−5.8, +0.2] | 1.03 [0.90, 1.28] |
| G1 final (A1 and B) | 664 | −15.8 [−16.7, −13.3] | −2.7 [−4.9, +0.4] | 1.00 [0.64, 1.26] |
| G2 @ 134.1M | 168 | −12.5 [−15.5, −10.0] | −4.0 [−4.8, −1.1] | 1.21 [0.99, 1.41] |
| G2 @ 163.8M | 167 | −14.6 [−16.8, −13.5] | −3.5 [−4.2, +0.4] | 1.05 [0.86, 1.42] |
| G2 final | 681 | −14.1 [−17.5, −13.1] | −2.4 [−4.0, +0.8] | 1.09 [0.77, 1.27] |
| G3 @ 109.9M | 192 | −13.4 [−16.1, −12.3] | −4.1 [−4.9, −0.1] | 1.46 [1.17, 1.66] |
| G3 @ 134.1M | 191 | −11.9 [−13.2, −7.0] | −3.7 [−4.3, +0.8] | 1.03 [0.93, 1.24] |
| G3 @ 163.8M | 192 | −12.1 [−12.7, −9.4] | −3.4 [−4.0, −0.8] | 1.28 [1.12, 1.43] |
| G3 final | 682 | −11.5 [−12.0, −10.3] | −2.4 [−3.8, −0.6] | 1.31 [1.11, 1.55] |

**G over training (computed here from `outputs/fields/fields-earlier/*.jsonl` and `fields-anchored/*.jsonl`).**
- The pre-registered summary of G on these keys is empty (Section 7.9). The table below is therefore a plain median over keys: not pair-weighted, no CI, A1 keys only.
- Keys are kept if they are not capped, c_min is valid, the branch and Gate 3 checks pass, c_min reads no κ clipping, and the argmin lies on the plateau.
- The earlier snapshots' keys are step 1's P6c checkpoint keys (37–48 stimuli at A1). The finals' A1 keys cover 44–52 stimuli, so the key sets differ.

| Agent | 73.7M | 109.9M | 134.1M | 163.8M | Final (A1) |
|---|---|---|---|---|---|
| G1 | 0.787 (160) | 0.827 (159) | 0.822 (160) | — | 0.793 (425) |
| G2 | — | — | 0.870 (155) | 0.834 (167) | 0.834 (443) |
| G3 | — | 0.893 (191) | 0.891 (190) | 0.848 (148) | 0.844 (376) |

**Reading.**
- **G is already about 0.8–0.9 at the earliest snapshot tested**, and it does not rise with further training. G3's drifts down slightly, from 0.89 to 0.84.
- **The early-braking offset after fast approaches is there from the first snapshot.** G1's shrinks from 21 m to 16 m by 109.9M.

### 7.9 The anchored-key version of H2′ (descriptive): empty in `decisions.json`

**What happened.**
- `decisions.json` (`anchored_edges`, and `earlier.<snapshot>.anchored_edges`) gives G, B and T as undefined for all 12 finals and all 8 earlier snapshots: 0 pairs, coverage null.
- The cause is in the frozen code. `stages.edge_key_measures` skips any row whose `w_is_cell` is false, a filter meant to drop the six A2 horizon twins. All anchored keys are W = 400 keys on cells whose W differs, so all 9,525 anchored rows in `fields-anchored/` and `fields-earlier/` carry `w_is_cell: false` (checked here), and every one is dropped.
- H1-fast does not use this filter and is unaffected.
- The frozen code is not changed. The item is descriptive and no decision depends on it.

**Per-key values instead** (computed here; plain medians over keys kept by the rule in Section 7.8; A1 and tier B pooled).

| Agent | Keys | G | B (m) | T |
|---|---|---|---|---|
| G1 | 662 | 0.787 | 0.98 | 0.72 |
| G1s1 | 565 | 0.815 | 0.99 | 0.78 |
| G1s2 | 662 | 0.814 | 1.44 | 0.72 |
| G2 | 681 | 0.833 | 0.97 | 0.75 |
| G2s1 | 687 | 0.802 | 1.31 | 0.73 |
| G2s2 | 669 | 0.787 | 1.24 | 0.70 |
| G3 | 567 | 0.851 | 0.72 | 0.79 |
| G3s1 | 687 | 0.744 | 1.43 | 0.72 |
| G3s2 | 687 | 0.730 | 1.22 | 0.75 |

On each agent's own fast-approach states, G is 0.73–0.85, the same range as on the shared A2 states.

### 7.10 Left-right asymmetry

Source: `decisions.json` (the `asymmetry` index of each measure, m_L − m_R over mirror partners, with a cluster-bootstrap CI).
- **G** (Section 5.1). |index| ≤ 0.082. The CI excludes 0 in 4 of 9 generalists, with mixed signs: G1 +0.028, G2s1 +0.075, G3 −0.029, G3s1 +0.082.
- **B.** The index runs from −0.33 to +0.50 m, and the CI excludes 0 in 6 of 9, with mixed signs.
  - Positive: G1s1 +0.50, G2 +0.30, G2s2 +0.32.
  - Negative: G1s2 −0.33, G3s1 −0.29, G3s2 −0.31.
- **T.** The CI excludes 0 in 5 of 9, with |index| ≤ 0.065.
- **r_in** (H2-speed). The CI excludes 0 for G1 (−0.22), G2s2 (−0.22) and G3 (−0.21).
- **Reading.** Step 1 found a learned mirror asymmetry (P5). Here it shows mainly in where the line sits on left and right corners (B, by up to half a metre), and much less in how it tracks the edge (G, by under 0.1). The sign differs between seeds of the same variant.

### 7.11 Specialists

- S1–S3 have no A2 fields: they are in neither H2′ nor H4.
- **Most of their anchored keys are invalid.** The closed loop crashes before the window's end on 71% (S1), 59% (S2) and 94% (S3) of non-capped anchored keys (`decisions.json`, `H1_fast.<agent>.coverage`). That is above the 30% rule, so they are descriptive.
- **H1-fast.** S2 brakes 7.0 m [4.9, 8.7] earlier than the optimum. S1's and S3's CIs contain 0, on 5 and 3 geometries.
- **G on the surviving keys** (computed here; per-key medians as in Section 7.9): S1 0.14 (144 keys), S2 0.48 (151) and S3 0.72 (19). These are heavily selected subsets: the keys a specialist survives. They are not comparable with the generalists'.

### 7.12 Capped stimuli and corner regime

- **Capped stratum** (`decisions.json`, `edge_descriptive.<network>.capped_G`). On the 48 capped A2 keys, where the generator's inside cap narrows the inside over a few apex knots at R 6.0, 7.5 and 9.4 m, the generalists' G is 0.33–0.85, on 3–5 geometries with wide CIs. These are also the tightest corners, so the cap and corner size cannot be separated.
- **H4's per-regime breakdown** (`decisions.json`, `H4_regimes`; per-key medians, A2 cells typed by G1 for v1 and by G3 for v3). G is lower on corners below the training support than on bulk corners in 25 of 26 networks: 0.38–0.76 against 0.66–0.89.

## 8. What it means

**Corner-wide shifts are tracked, though not one-for-one.**
- On synthetic corners, when the inside edge moves for the whole corner, every generalist's line moves about 0.76–0.85 as far (H2′: confirmatory for 8 networks, descriptive for G3s2).
- That is not width-blindness and not a deficit by the pre-registered standard. It is short of the optimum's 1.0, and only G3's CI clears the lawful band's 0.8.
- G3's lawful decision may partly be selection, because it rests on G3's own exclusion set. Its 48 off-plateau keys, an outcome-dependent exclusion, are tight corners where the other generalists track less, and on the keys valid for all nine its per-key median is not distinctive (Section 5.1, computed here). The decision stands under the frozen rule.
- The response is near-linear and symmetric up to 2 m, and it is already present at the earliest training snapshots tested.
- **G* = 1 may be the wrong reference.** The optimum's G* = 1 is the gain of a line that touches the inside edge (its contact is active). The networks keep 1–2 m of buffer, and for a line with a margin, tracking the edge one-for-one is not obviously optimal. The bands compare the networks to an edge-touching reference. The costate design makes the same point (`docs/interp/costate-design.md`, Section 1).

**Local edge changes on real roads are barely followed.**
- On G2's three real crash corners, the line gives way by a median 19% of the inside edge's intrusion at the crash point, and only 7% in answer to the inside edge itself (H3, descriptive).
- With a corner-wide gain of 0.83 for the same network, this gap is the most informative result of step 2. Linearity in both settings (dose-response; λ_lin ≈ λ*) argues against amplitude or nonlinearity as the explanation. The dose test covers only synthetic, corner-wide trapezoid shifts, and H3's λ_lin is within 0.1 of λ* in 8 of 12 cases and at most 0.25 off.
- What differs is the shape of the change: the trapezoid is flat over the corner with 20 m ramps, while the real profiles change within the corner and on both sides. The corners and states differ too: transplanted real corners from G2's own lap states, against synthetic corners at A2.
- **Hypothesis:** the networks' edge tracking depends on the spatial scale or locality of the edge change. They follow slow, corner-wide shifts and largely ignore edge changes local to the apex. That is consistent with their training: step 1 found that width changes of 1 m or more within 100 m occur at 0.4–0.5% of tight-corner knots in the training generator, against 79% on the real crash roads.

**The crash mechanism, to first order.**
- The clearance at the closest point falls by about (1 − r) per metre of local inside-edge intrusion, where r is the share the line follows (≈ 0.2 here).
- A buffer of 0.6–0.8 m (G2 on the constant-width versions of its crash corners) is used up when the real edge comes in by roughly that much: at Melbourne the whole real intrusion (0.6–0.7 m), at Norisring about half of a 3 m intrusion.
- This fits step 1's observations: 159 of 160 crashes leave by the inside edge, and constant width rescues them.

**Real-width training appears to buy margin, not tracking (exploratory).**
- Real-width networks have lower G (by 0.08), higher B (by about 0.3 m), slightly higher G_out and unchanged T. They sit further from the inside edge rather than following it more closely.
- The leave-out networks' shifts point the same way at about 40% of the size, but they are not significant and cannot be told apart from either no transfer or full transfer of the in-sample shift (Section 6.3). Linking them to the width test's finding that about half of the completion gain transfers would be speculation.
- A larger margin would be expected to cost lap time. The width test found exactly that: higher R-dev completion, but a median gap to the optimal lap 0.3 (v1) and 0.6 (v3) points larger on circuits both arms complete ([width-test.md](width-test.md), Section 5.3).
- These links are consistent, not shown: no analysis here connects B to completion or lap time.

**For training.**
- Adding real width profiles did not, in these runs, make the networks track the inside edge more closely. If anything, they tracked corner-wide shifts slightly less, and they kept more margin (exploratory; code and hardware confounded; 4 seeds per arm).
- If the aim is to follow local edge changes rather than to avoid them with margin, a training change should be judged on the local response (an H3-type measure), not on G for corner-wide shifts. That local response has not yet been measured for the real-width networks.

**What is not established.**
- **No confirmatory claim about width training.** H4 and H4-transfer are not testable. Section 6 is exploratory, confounded by code and hardware, and based on 4 seeds per arm.
- **No causal link from low tracking to crashes.** H2′ found no deficit for corner-wide shifts. The real-road contrast (0.8 against 0.2) rests on 3 units of one network, the units were not prospective, and the analysis is descriptive.
- **No frequency or locality dependence yet.** It is a hypothesis. The trapezoid and the real profiles differ in shape, position relative to the apex, amplitude and which sides change, and the real and synthetic corners and their start states differ as well.
- **No proof that G < 1 is a defect.** For a buffered line the edge-touching optimum is not obviously the right reference.
- **No link from margin to the width test's outcomes.** That the higher B explains the width test's completion gain or slower laps is not shown.
- **No pair-weighted anchored-key G.** The pre-registered summary on anchored states is empty (Section 7.9). Only per-key medians are given.
- **Nothing about R_TEST.** It stays sealed and was not used.

## 9. Deviations and procedure log

### 9.1 Before the tag

| Date (EDT) | What | Effect |
|---|---|---|
| 21 Oct (design v1–v3) | **Departure from protocol 9.6**: step 2 was taken before step 1b (two-corner stimuli), because the ladders named width, not a neighbouring corner, as the necessary factor | Logged in the design's preamble. Step 1b is still open |
| 21 Oct 09:13–09:59 | **First pre-tag pass superseded.** Its outputs were unstamped and were moved to `outputs/fields_prestamp_2026-10-21/` at 22:52. The stamped pass (22:52–23:43, `runs/fields-pretag2.log`) is the one the record binds | None on decisions. The design's v3.1 figures for Gates 1, 4 and 5 and for the phase ceiling match the stamped pass. Where Section 15 added checks, the passes differ: the hybrid control raised BRANCH_TOL for c_min from 7.7e-4 to 3.3e-3 m, and Gate 2 gained the hybrid rows |
| 21 Oct 09:31, 09:59, then the stamped pass | **BRANCH_TOL for c_min changed twice before the tag**, 5.7e-4 → 7.7e-4 → 3.3e-3 m. (1) 5.7e-4 m: the first pass's calibrate stage at 09:31, from a trained-agent teacher-forced error of 3.69e-5 and 8 control rows (`runs/fields-pretag.log`). (2) 7.7e-4 m: recomputed at 09:59 with the same 8 control rows (all three tolerances scale by the same factor, 1.34), but with 4.94e-5, Gate 3's largest mean action error (G2s2), as the trained-agent error (`outputs/fields_prestamp_2026-10-21/calibration/branch_tol.json`). The stage log does not show this recomputation. The reason, from the lead's record: the first calibrate run read only Gate 3's 5-row planted-error sample (largest error 3.69e-5) instead of the every-key check (4.94e-5), a bug in the calibrate stage. The stage was fixed to use the every-key maximum and re-run. (3) 3.3e-3 m: the stamped pass, with Section 15's two hybrid control rows added (c_min slope 13.4) | Before the tag, so allowed. Overall the c_min tolerance loosened about 6×. The record binds the last value |
| 21 Oct 09:32 | **Gate 3 count corrected in the notes.** The `runs/activity.md` entry first summed Gate 3's keys as 14,208. The correct total, 13,908, is the sum in `gates/gate3.json` and matches `calibration/branch_tol.json` (`gate3_keys_checked`) | Notes only. The output was never wrong |
| 21 Oct 11:37 → 21:00 | Work paused mid-way through the Section 15 implementation and resumed, with the partial work backed up (`outputs/fields/wip/`) | None. No trained-network field existed, and the tag came later |

### 9.2 After the tag (dated)

| Date (EDT) | What | Effect on results |
|---|---|---|
| 21 Oct 23:45:52 → 23:48:37 | Tag `receptive-fields-prereg-v1` on `0951f1e`; harness-f32 pinned it in `outputs/fields/tag.json` at 23:48:05; the pin was committed at once (`17e347e`) | As designed (Section 16) |
| 22 Oct 00:17:36 | **gate2-trained: two checkpoints fail** (Section 4.1): G3s2 / `dr-pilot-v3-tilt50a2-s2` and `dr-realw-v1-s1`, one key each, on c_min | Under the strict pre-tag rule: G3s2 is descriptive for H2′, v3 becomes "seed-specific" rather than "inconclusive", and H4 and H4-transfer are not testable. The decision to report H4 exploratorily was noted at 00:18 in the local notes file, before any A2 field existed (Section 6.1). The note's "33/35 pass" should read 32 of 35 entries, or 27 of 29 checkpoints |
| 22 Oct 00:17:37 | **Two lanes in parallel.** Lane A ran fields-a2 → fields-anchored → fields-earlier. Lane B ran h3-keys → dose → h3-lambda. Section 16 lists the stages in one sequence (h3-keys before fields-a2; dose and h3-lambda after fields-earlier) | None expected. No field stage reads another post-tag stage's output except fields-a2, which reads harness-f32 (already done); h3-lambda in any case ran after h3-keys in its lane. The six checkpoints that fields-a2 computed twice, at different times and loads, gave bit-identical per-key G, B and arrays (Section 4.2) |
| 22 Oct about 00:47 | **Lane scheduling changed.** fields-a2 ran slower than budgeted, at about 1.1 s per network-key (19,030 s for 16,835) against design Section 11's "under 1 h" for all network fields. Lane A's wrapper shell was stopped, and fields-a2 continued in its own process. fields-anchored and fields-earlier were started at 00:47:11 in two further processes (`runs/fields-posttag-anchored.out`, `runs/fields-posttag-earlier.out`) | No code, rule or input change: same code hash, no deviation, same tag in every stamp. `runs/fields-posttag.log` ends "FAILED a lane (A=137 B=0)" at 02:20:13. That is the stopped wrapper's exit status (killed), not a stage failure. fields-a2 finished all 35 × 481 = 16,835 rows at 05:34:47 (`runs/fields-fields-a2.log`, 315.4 min) |
| 22 Oct 00:47 → 03:00 | fields-earlier finished at 01:18 and fields-anchored at 03:00 (8,136 rows), both before fields-a2. dose ran 00:33–02:18 and h3-lambda 02:18–02:20, all before fields-a2 finished | None. analyse ran only after every stage |
| 22 Oct 05:35:09 → 05:43:07 | Non-confirmatory analyse: coverage and plumbing only (`decisions_dry.json`). Coverage 0.9–22.7%, 0–2 pipeline failures per network, nothing missing | As designed. It contains no H2′ or H4 decisions |
| 22 Oct 05:43:35 → 06:05:55 | **The one confirmatory run** (`decisions.json`), committed 06:06:05 (`b839a57`) | As designed |
| 22 Oct 06:07 | Exploratory H4 file written (`outputs/fields/exploratory/h4_exploratory.json`). It is outside the stage machinery and unstamped, and it names its source by sha256 | Labelled exploratory. Its numbers were verified here (Section 6.1) |
| 22 Oct (this write-up) | **Anchored-key H2′ and the earlier snapshots' G found empty** in `decisions.json`. A filter in the frozen code drops every anchored row (Section 7.9) | Descriptive items only. Per-key medians given instead, labelled "computed here". The frozen code is not changed |

**Checks with nothing to report.**
- No stage used `--post-tag-deviation`. Every post-tag stamp has `deviation: null`, empty `changed_vs_head` and no partial runs.
- No code changed after the tag.
- No loop was exhausted in any stage: gate2-trained `n_exhausted` is 0 for all networks, and no row in `fields-a2/`, `fields-anchored/`, `fields-earlier/` or `dose/networks.json` carries a pipeline flag (`h3.json` has none either).
- No R_TEST circuit was used.
- The analyse runs have no stage log under `runs/`. Their times come from their provenance stamps.

**Records that are not in the repository.**
- `runs/activity.md` is a local, untracked notes file with minute-resolution times, edited during the run. The 00:18 entry is the only record that the exploratory H4 was chosen before any field existed; the first A2 impulse array's write time, 00:19:57, corroborates it (Section 6.1).
- The code that first wrote `outputs/fields/exploratory/h4_exploratory.json` ran inline. It is now committed as `scripts/fields_exploratory_h4.py` (the frozen test functions on the committed decisions, checked by sha256), which reproduces the file.
- The post-tag stages were run by a temporary wrapper shell script outside the repository. Only its output is kept, in `runs/fields-posttag.log` (including the "Killed: 9" line of the stopped lane A). The stages themselves are the frozen `scripts/fields.py`, and each carries its provenance stamp.

## 10. What next

1. **Frequency and locality of edge tracking (exploratory; write and commit the plan first).**
   - The per-key impulse responses are already on disk for all 35 network entries on all 481 A2 keys (`outputs/fields/fields-a2/arrays/<network>_<timesteps>/<key>.npz`, for example `G1_200000000/`; `impulse`: 4 outcomes × {κ, w_left, w_right} × 2,560 samples on the 2 m grid). The line's response to any inside-width profile is a linear combination of them.
   - So, without running a network, one can compute the gain for inside-edge changes of controlled spatial scale and position:
     - the distance-resolved Gaussian bumps already defined for the secondary fields (σ_b = 4, 8 and 16 m, at d along the corner);
     - trapezoids with shorter plateaus;
     - the measured real-minus-constant profiles of the crash corners.
   - Questions:
     - Does the gain fall as the change becomes more local or moves toward the apex?
     - Does it reproduce H3's 0.07–0.19 for real profiles?
     - Do the real-width networks follow local changes any better than the baselines, or only keep more margin?
   - To separate the shape of the change from the corner, also compute G itself (the trapezoid gain) on the H3 units. That needs a few new VJPs of G2 at 12 unit-phases, so it belongs in the plan.
   - Finite-amplitude closed-loop checks, as in the dose stage, on a pre-specified subset. R-dev roads only.
2. **A summary of the anchored-key G with the pre-registered pair weighting and CIs** (exploratory, outside the frozen code). It replaces the per-key medians of Sections 7.8–7.9.
3. **A common-key H2′ summary** (exploratory): the pre-registered pair weighting and CIs on the A2 keys valid for all nine generalists, to check how much of G3's lawful decision is its own exclusion set (Section 5.1).
4. **Idea 2: the costate** ([`docs/interp/costate-design.md`](../interp/costate-design.md), v2.1).
   - Next steps: implementation, gates on controls, a dry run and review, then the tag `costate-prereg-v1` before any activation or gradient is computed.
   - Its C3 hypotheses use step 2's outputs: B on the A2 cells as a covariate in C3-w, and per-geometry G − 1 and B in C3-int. Both were fixed before the confirmatory decisions were computed (v2.1, Section 12; committed at 03:35 in `705b221`). Some post-tag field outputs already existed while it was written (the costate design ran 02:18–03:35, after fields-earlier and dose had finished and while fields-a2 was part done). The costate design cites none of them, but that it read none is not recorded.
   - The costate design should state before its tag how it treats the two gate2-trained failures.
5. **A confirmatory test of real-width training would need a new pre-registration**: new seeds, G, B and a local-tracking measure as outcomes, and a gate rule that says in advance how a single-key failure is handled. The present data cannot be re-decided.
