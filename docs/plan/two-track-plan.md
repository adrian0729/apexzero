# ApexZero final plan: one paper and one demo

*This is the final plan as of 18 Oct 2026. Week 1 (W1) starts Monday 19 Oct 2026. The private dates assume 10 h/week. Three pairs of light weeks run at 5 h: W10–11 (holidays 2026), W44–45 (August 2027) and W62–63 (holidays 2027). At 8 h/week, multiply every remaining duration by 1.25. All venue dates are tracker estimates, so check the official site once each call is posted.*

## 1. Overview

| | Paper | Demo |
|---|---|---|
| Working title | *Regret, Measured: Auditing Curriculum Level Scores Against Optimal Control* | *How far from the limit, and why?* |
| Question | Do the level scores that PLR and SFL use track true regret, both as fresh scores and inside real curriculum buffers? Does that change the curriculum's outcome? | How close does a zero-shot generalist get to the best replayable lap on 12 unseen F1 circuits? Where and why does it lose time, and which plausible circuit is its worst case? What can public data say about humans? |
| Ships | arXiv v1 ~15 Aug 2027, then ICLR 2028 (~late Sep). A pip-installable benchmark | Release 1 ~24 Oct 2027, Release 2 ~16 Jan 2028, then the main video, viewer and repo ~30 Apr 2028 |
| Hours | 411 to arXiv (including the shared core and P0), plus 20 after | 318 |

**Shared asset: the answer key.**
- TO-OCP solves the agent's *own* problem: the same RK2×10 integrator, 20 Hz zero-order hold, constraints and reward.
- Its controls replay in the float64 simulator. So J*_20 is a **best-known, replay-verified optimum**, and "agent minus optimum" is a certified *lower bound* on how much time is left.
- The paper uses it to grade the teacher (the curriculum scores). The demo uses it to grade the driver and the exam.
- E0 (domain randomisation, 5 seeds × 1B steps) is both the paper's main audit subject and the demo's generalist.

**The dependency runs one way.** The paper needs no demo block. The demo reuses E0's checkpoints, TO-OCP and the audit labels, and each of those has a fallback.

---

## 2. The paper

### 2.1 Questions and contributions

ACCEL is dropped from the title and from the claims, because the mutation editor that would test it was cut.

- **RQ1, fidelity.** How well do the PLR and SFL level scores rank levels by regret? This is measured on fresh rollouts and inside the buffers that real PLR⊥ and SFL runs keep.
- **RQ2, dynamics.** How does fidelity change with training phase, critic privilege and observability? Which measured term explains the change: critic bias, dispersion, the gap between best-observed and optimal return (ceiling gap), success saturation, or score staleness?
- **RQ3, consequence.** On the same labelled pool of levels, does PLR⊥ with oracle-latest scoring beat PLR⊥ with PVL or MaxMC, SFL and DR on held-out regret?

**Contributions**
1. A JAX racing benchmark of about 5,400 levels, each labelled with the replay-verified optimum of the agent's own objective.
2. To our knowledge, the first ground-truth audit of curriculum (UED) level scores across training, both fresh and in-buffer.
3. A curriculum comparison on a single shared pool, including an oracle arm.
4. A maze replication that uses JaxUED's own code.

The algebraic identities in E2 are the measurement framework for RQ2, not a contribution.

### 2.2 Definitions

- **Level θ = (g, c, μ, x0).** Every component is fixed per level and checked by assertion in the label schema:
  - g, an 8 km strip;
  - c, the car context;
  - μ, one friction draw;
  - x0, at the start of a straight at least 300 m long: n = 0, ξ = 0, v = 0.8·v_QSS, previous action 0.
  - Windows differ only in policy sampling and the 1% observation noise.
- **Episode:** a 1,200-step window (60 s at 20 Hz). Return G = Σγ^t r_t with γ = 0.995. J^π is the expected return of the *stochastic* policy; the deterministic version is secondary.
- **Targets (both primary):** absolute regret Reg = J*_20 − J^π, and normalised regret Reg/J*_20.
  - J*_20 comes from TO-OCP in window mode.
  - **Every** level gets the same 3-start budget: warm starts from the s-domain OCP, from the QSS line, and from the best rollout of the final checkpoint.
  - Replay in float64 must match within 0.05%, with a 2–5 cm margin inside the corridor.
  - It is reported as the *best-known optimum*, so Reg is a lower bound on true regret.
- **Bounds and references**
  - The best agent return ever observed is a free lower bound on J*_20.
  - A control-rate reference, J*_100: the same window problem at 100 Hz zero-order hold, warm-started from the 20 Hz solution, on 200 levels.
- **Secondary targets**
  - Distance lost in 60 s, against an undiscounted (γ = 1) maximum-distance TO-OCP, on 600 A_DR levels. This removes the mismatch caused by discount weighting.
  - A hurdle split into crash probability and regret given that the window was completed. Total Reg stays the headline, because it is what UED optimises.
- **Label noise**
  - 200 pilot levels get at least 3 extra starts, which gives a label SD.
  - Late-phase degradation is claimed only where median Reg is at least 3× the label SD.
  - Reliability-corrected correlations are reported as well as raw ones.

**Six headline estimators**
1. L1 value loss.
2. Positive value loss (PVL).
3. MaxMC, in two variants: as implemented (undiscounted R_max − V) and unit-consistent (R^γ_max − V(o_0)).
4. Learnability p(1−p), under two success definitions: window completion, and a non-saturating one (complete the window *and* cover at least 97% of the QSS distance).
5. MNA.
6. Disagreement across 5 stop-gradient critic heads.

Stored variants are λ ∈ {0.9, 0.95, 0.98, 1}, plus discount-weighted versions. The appendix covers 1−p, return variance and step-normalised ALP.

**Baselines**
- Agent-only: −Ĵ^π.
- Policy-agnostic: J*_20 and geometric difficulty.
- Random.
- "Oracle-operational" (J*_20 − G_op) is a *reference, not a ceiling*: critic-based scores pool earlier history and can beat it. The actual ceiling is √(reliability).

**Racing-PLR scoring protocol**
- Scores accumulate across 128-step rollout chunks, with GAE bootstrapped at each chunk end, and are written when the episode ends.
- A new level gets one full exploratory window before it can enter the buffer.
- The audit's "operational" score calls the same function.
- This deviates from JaxUED, which scores −inf when no episode completes within num_steps. A table lists every deviation from JaxUED's defaults.

**Window assignment** (frozen in `audit-v1`)
- 32 stochastic windows per level and checkpoint, raised adaptively to 64 when 0.05 < p̂ < 0.95.
- Windows 1–4 feed the operational scores, and windows 5 onward feed the target.
- High-budget scores are cross-fitted: odd windows for the score, even windows for the target.

### 2.3 Endpoints and statistics

**Co-primary endpoints.** Holm correction applies within the family. All are reported by phase (window completion on A_DR: below 50%, 50–90%, 90% and above) and by critic type.
- Spearman ρ against absolute Reg and against normalised Reg.
- RC@10%: the share of regret captured by the score's top 10%.
- **In-buffer ρ:** stored score against current Reg, with score age as a covariate.
- **Admission AUROC:** how well the score admits levels whose true regret is in the top quantile.
- Held-out regret IQM at 1B steps, for RQ3.

**Diagnostics**
- A two-loading regression, rank(S) ~ a·rank(J*_20) + b·rank(Ĵ^π). A faithful score has a > 0 and b < 0.
- Partial ρ.
- P@q and contamination@q.
- **η from simulated PLR replay** (buffer, admission rule, staleness mix, β ∈ {0.1, 0.3, 1}). A static rank^(−1/0.3) ranking puts 87% of its mass on a single level, so it is not used.
- Lagged-score fidelity: the score at checkpoint k−1 against regret at checkpoint k.

**Thresholds** are set at G1 from the pilot's minimum detectable effect. Provisional values:
- **faithful:** ρ ≥ 0.5 and RC@10% ≥ 0.6;
- **uninformative:** ρ ≤ 0.2, or RC@10% ≤ 0.2, or not above the best of the agent-only and policy-agnostic baselines (paired CI).

**Statistics**
- A two-way bootstrap over seeds and levels, or a crossed-effects model on Fisher-z.
- The primary trend test is the seed-slope t-test.
- The TOST margin comes from the pre-computed minimum detectable effect.
- σ, the entropy coefficient and the observation-normalisation statistics enter as covariates.

### 2.4 Experiments

| ID | What | Size | Answers |
|---|---|---|---|
| E0 | DR with the asymmetric critic. Also the demo's generalist | 5 seeds × 1B steps. 16 checkpoints log-spaced from 10M to 1B; the showcase seed also every 25M | RQ1, RQ2 |
| **E0-sym** | Symmetric critic, as in standard UED. **Co-primary** | 3 × 1B, 16 checkpoints | Critic privilege |
| E0-const | Constant learning rate and entropy coefficient | 2 × 1B, 8 checkpoints | Whether the training schedule is confounded with phase |
| E0-short | Preview distance d_max halved | 3 × 1B, 8 checkpoints | Observability |
| E1 | Fresh-score audit of every E0 family | 168 checkpoints × 1,412 levels × 32–64 windows | RQ1, RQ2 |
| E2 | Identities: PVL = ½·L1 + ½·mean(Â); MaxMC_unit − Reg = ceiling gap + critic bias; fraction of tied levels | Logged during E1 | RQ2 |
| **C** | Curricula on the 4,000-level labelled pool, buffer of 1,000, JaxUED defaults (replay 0.8, β 0.3, staleness 0.3). Arms: DR-pool ×5, PLR⊥-PVL ×5, PLR⊥-oracle-latest ×5, PLR⊥-MaxMC ×3, SFL ×3. E0 is the DR arm on the full generator. Seed counts are re-sized at G1 from E0's variance | 21 runs × 1B; buffer snapshots at 8 checkpoints | In-buffer RQ1; RQ3 |
| E5 | JaxUED maze with JaxUED's own DR, PLR-MaxMC and SFL code at its defaults. Optimum from BFS; 5×5 and 13×13 views; a buffer audit as well | 5 seeds × 16 checkpoints × 2,000 mazes | External check |
| A_QSS | Bias of the QSS labels | 20k levels × 1 seed × 8 checkpoints × 4 windows | Secondary; feeds Nemesis |

**Level sets** (hash-partitioned, frozen in `audit-v1`)
- A_DR: 1,000 levels, i.i.d. from the training generator.
- A_tail: 300 levels from one tilted generator (hairpin-heavy and technical), still inside training support.
- A_OOD: 100 levels.
- Pool: 4,000 levels, disjoint from the sets above.
- A_DR, A_tail and A_OOD double as C's held-out evaluation, so no separate A_eval set is needed.
- Track-type strata are tagged from the tokens in the first ~1.4 km, which carries 90% of the discounted weight.
- Real circuits are left out of the paper.

### 2.5 Predictions and the outcome table

**Pre-registered predictions**
- The PVL/L1 ratio tends to 0.5 as critic error shrinks.
- MaxMC as implemented has a high ρ against absolute regret but a ρ near zero against normalised regret.
- The error of unit-consistent MaxMC is dominated by the ceiling gap on hard levels.
- In-buffer fidelity is lower than fresh fidelity.

The learnability tie fraction is reported only as a description, since it follows from how success is defined.

**E4 trigger.** E4 runs only if row B, D or E holds for PVL, MaxMC or learnability at operational budget, *after* Holm correction, the label-noise floor and the reliability correction. Every E4 design includes both the DR-pool arm and the full-generator DR arm, and uses the audit's units.

| Row | Test | Claim | Follow-up |
|---|---|---|---|
| A, faithful | Faithful in every phase, for both critics, fresh and in-buffer; TOST passes | "With dense reward, scores track regret in continuous control." Scoped by E0-sym, E0-short and E5 | None |
| B, degrades late | ρ_late − ρ_early ≤ −0.1 with the CI excluding 0, still present in E0-const | "Scores guide the curriculum only while the agent is incompetent, because of [E2 term]" | **E4-switch:** use the score until completion C*, then the oracle. A control arm scored against J*_100 checks the ReMiDi trap |
| C, uninformative | Uninformative throughout | "Ground truth confirms No Regrets in continuous control." | None: C already compares oracle, PVL and DR |
| D, depends on track type | Δρ ≥ 0.2 between strata | "Score error is structured, so samplers under-replay [type]." | **E4-type:** rank normalisation within each type, plus stratified replay |
| E, faithful but not useful | Oracle-latest ≈ DR-pool in C | "Regret is the wrong target here." | **E4-hybrid:** p(1−p) + λ·Reg_norm, plus a stuck-level diagnostic |
| F, manipulations disagree | Symmetric vs asymmetric critic, full vs short preview, or racing vs maze differ by ≥ 0.2 | "Fidelity depends on [critic privilege / observability]." | None |
| G, fresh ≠ buffer | In-buffer ρ differs from fresh ρ by ≥ 0.2 | "DR-style audits misstate what curricula actually see." | None |

### 2.6 Threats and mitigations

| Threat | Mitigation |
|---|---|
| "It audits a sampler nobody runs" | C runs real PLR⊥ and SFL, and the in-buffer metrics are co-primary |
| Labels are local optima, or noisy late in training | Uniform 3-start budget; repeatability study; noise floor; "lower bound" wording |
| The privileged critic flatters the value-based scores | E0-sym is co-primary, and the size of the drop is reported |
| Training schedule moves with phase | E0-const; schedule covariates; deterministic windows reported separately |
| Discount weighting, and crashed and clean windows mixed together | Straight-start x0; strata and score variants weighted by discount; γ = 1 target; hurdle split |
| One bespoke domain | Maze on JaxUED's own code; critic and observability manipulations inside the racing domain; claims about reward density stated as conjecture |
| Forking paths | Six headline estimators; Holm correction; `audit-v1` frozen before any full-scale result; claim-ledger CSV written by the figure scripts |
| Numerical artefacts | C2 B-spline track lookup in both JAX and CasADi; float64 verification; float32-versus-float64 error bar |

### 2.7 Related work

**Verdict.** No paper from 2024–26 compares UED scores with exact or best-known regret across training.

- **No Regrets** (Rutherford et al., NeurIPS 2024). Correlates PVL and MaxMC with success rate, at one checkpoint. Its Perfect Regret arm is 1−p.
- **ReMiDi** (Beukman et al., ICML 2024). Uses perfect regret as a sampler and finds the irreducible-regret trap.
- **Abdel Sadek et al.** (RLC 2025). Introduces oracle-latest but judges it only by downstream behaviour. C reuses its definition.
- **Monette et al.** (RLC 2025).
- **New scores judged only by downstream performance:** MNA (Mead et al. 2026), TRACED (ICLR 2026), PACE (ICML 2026).
- **Foundations:** PAIRED, PLR, Robust PLR (PLR⊥), ACCEL, SAMPLR, JaxUED.
- **Continuous-control work with no optimal-return oracle:** Hoeppke 2022; Song et al. 2023; Green et al. 2026; Environment as Policy; CENIE.
- **Method precedent:** DOPE for off-policy evaluation ("DOPE for UED scores"); rliable for the statistics.

**Checks**
- W1: search the ICLR 2027 submissions on OpenReview and set a Scholar alert on No Regrets.
- September 2027: re-check the NeurIPS 2027 accepted papers.

### 2.8 Venue

| Venue | Deadline (estimate) | Role |
|---|---|---|
| ICLR 2027 workshops | ~late Jan to early Feb 2027 | **Maze note** (non-archival), posted to arXiv at the same time to establish priority |
| NeurIPS 2027 | ~mid-May 2027 | Off the plan: it needs at least 14 h/week from today |
| **ICLR 2028** | ~late Sep 2027 | **Target.** arXiv v1 ~15 Aug |
| TMLR, or RLC 2028 | Rolling / ~Mar 2028 | Fallback. This is also the public commitment at 8 h/week |

---

## 3. The demo

### 3.1 Story

The answer key changes what you can ask. The demo asks five questions, and each one leads to the next.

1. **How far?** The gap to T*_20 on 12 unseen circuits, plus their mirrored and reversed versions. Each gap carries ±0.5 m width error bars and the distance to the nearest training track.
2. **Where?** Line versus execution, both measured at 20 Hz:
   - G_line = (T_FP20 − T*_20)/T*_20
   - G_exec = (T_RL − T_FP20)/T*_20
   - The control-rate tax, T*_20 − T_cont, is shown separately.
3. **Why?** Place cells in the specialist against distance-to-brake units in the generalist. Then a brake knob injected after LayerNorm, with its dose-response in metres measured against the optimum's braking point.
4. **What is the worst case?** Nemesis I–III.
5. **What about humans?** They have no answer key, so the segment shows what snapped 3.7 Hz data can and cannot support.

A 45-second cameo covers the paper ("grading the teacher").

**Display rules**
- Three badges: **CERTIFIED** (against the replay-verified T*_20), **MODEL-DEPENDENT**, **POST HOC**.
- Human ghosts always carry the caption "position along lap only".
- Humans are compared with humans, and the AI with the optimum.
- Branding is "F1-style", with no logos. Any video that uses F1 data stays non-monetised.

### 3.2 Deliverables

| Release | Contents |
|---|---|
| R0 post, ~early Oct 2027 | Snapping audit: 6 circuits × 2 seasons, with 3 positive controls |
| **Release 1 "It has never seen Suzuka"**, ~24 Oct 2027 | 12-circuit grid; line/execution split by corner type; Short 1; viewer v0 (Rerun's web viewer on exported `.rrd` files) |
| **Release 2 "Inside its head / Nemesis"**, ~16 Jan 2028 | Probes and the brake knob; **Nemesis I–III** A2 poster; Short 2 |
| **Release 3**, ~30 Apr 2028 | ~12-minute video; drivers-lite; viewer v1 (lean Three.js: ghosts, synced v(s) and Δt(s) plots, per-corner receipt, Nemesis tab, Puppeteer render mode); repo v1.0 |

**Nemesis** (35 h, plus 8 h for the arms race)
- ES searches 300–600 m motif windows against a bias-corrected QSS. The bias model is fitted on the audit labels.
- For the last generations, a short-horizon TO-OCP runs inside the search loop on a cloud box ($5–10).
- Only the finalists are closed into F1-plausible circuits, using the Wasserstein penalty fitted on R-dev.
- Each finalist is certified with TO-OCP in lap mode, float64 replay and an RK4 re-check at 1 ms.
- Each is then delta-debugged down to its fear motif, and a reach test asks whether the failure lies beyond d_max.
- Three arms-race rounds of 50M-step fine-tunes follow.
- The search stays in procedural space. Any R-test number from an arms-race model is labelled POST HOC.

### 3.3 Beat sheet (~12 min)

| Time | Beat |
|---|---|
| 0:00 | Cold open: Suzuka Esses, the agent against the white optimum ghost, with a live Δt counter |
| 0:35 | The answer key: same car, same physics, a replayable optimum. The badges |
| 1:20 | The limit (Manim): friction ellipse; downforce equals weight at ~188 km/h; γ braking; my keyboard lap |
| 2:30 | Learning one track: failure montage; ghosts master Hockenheim |
| 3:30 | Track factory, then grading the teacher |
| 4:20 | The lockbox opens: `protocol-v1` on screen; the grid flips; the specialist dies at La Source; Monza reversed |
| 5:50 | Where the time goes: line/execution bars; one annotated corner |
| 7:00 | Inside its head: place units against brake units; knob dose-response |
| 8:20 | Nemesis I–III: search swarm, certificate, fear motif, reach ring, arms race |
| 10:00 | Humans: the grid collapses onto one line; why "99.2% of optimal" cannot be claimed; teammate deltas above the detection limit |
| 11:20 | Close |

**Split rule.** If Release 2 slips past 31 Jan 2028, ship beats 1–8 as Part 1 within 4 weeks.

### 3.4 F1 driver segment (drivers-lite: 45 h, plus 10 h for R0)

**Reframe, used word for word:** "The error in the car model (~1%) is larger than the gap between teammates (0.1–0.4%). So we measure where each driver gains or loses time against their teammate, and we show where the data cannot tell."

**Pipeline**
1. ICP registration onto the TUMFTM centreline.
2. An RTS smoother on [s, v, a], anchored to the sector times.
3. A lap filter: push laps within 102% of the driver's best, SOFT tyres, dry, no car within 1.0 s ahead.
4. Additive corner-complex segments, whose times sum exactly to the lap Δt.
5. Two phases per segment: entry/braking and exit/traction. Corner types come from the observed envelope apex speed.

**Gates**
- Injection-recovery at the empirical OpenF1 sampling distribution. Only cells above 2σ of the resulting error are published.
- Placebo: shuffle driver labels within each car-session.
- Split-half ρ ≥ 0.5. If it fails, publish only 2–3 pair case studies.
- Held-out sector anchor.

**Season 2** (escalate if split-half passes and at least 5 pairs show effects above the error budget):
- tiers and the resistance map;
- driver cards;
- the AI–human bridge;
- per-team calibration and M2;
- the multiverse analysis;
- 2026.

### 3.5 Rabbit holes

**Core** (inside the hours):
- the Nemesis reach test;
- whether the braking direction is the same vector at 120 and 300 km/h;
- the QSS–OCP Goodhart effect, through the bias model;
- the control-rate tax by corner type.

**Side quests.** Start one only while a batch is running or in the week after a release. Each is time-boxed, must produce one figure or a clip of 60 s or less, and is not counted in the hours.
- Shadow-price atlas, which is also the fallback demo if RL fails.
- Price of Tarmac.
- Normal modes, lite version.
- Learning-spacetime figure.
- Robustness at μ −5%.
- Scaling curve; declare it in `protocol-v1` if it is evaluated on R-test.
- Cramér–Rao bound.

**Sequels:** the sandbox, Branching Ghosts, and the 2026 energy OCP.

---

## 4. Shared core

| Phase | h | Contents | Paper uses it for | Demo uses it for |
|---|---|---|---|---|
| K1 | 45 | <ul><li>One `f()` behind an ops shim</li><li>**C2 B-spline track lookup** in both JAX and CasADi</li><li>RK2 at 5 ms × 10 substeps</li><li>Gates: parity < 1e-5; RK4 at 1 ms; coast-down; terminal speed; skidpad</li><li>Track toolkit for 24 circuits</li></ul> | Simulator | Everything |
| K3 | 18 | <ul><li>1-D braking solved by DP</li><li>From-scratch PPO that matches PureJaxRL and rejax on gymnax</li></ul> | A verified PPO | γ chapter |
| K2a | 40 | <ul><li>s-domain minimum-time OCP on Hockenheim, R-dev and open strips</li><li>Duals saved; NLP built once with the track as a parameter</li><li>Batch runner; QSS-lite</li><li>**6-h TO-OCP spike** (a throwaway prototype to measure speed and convergence)</li></ul> | Warm starts; sizing the label budget | Yardsticks |
| K4 | 45 | <ul><li>Hockenheim specialist, tuned on E0's GPU configuration (8,192 envs × 64 steps, 8–16 minibatches)</li><li>Evaluation harness</li></ul> | Removes the risk of PPO failing on the car | MVP-1; place cells |
| K5 | 54 | <ul><li>κ-grammar fitted on R-dev; filters; one tilted generator; P-OOD; straight-start x0</li><li>Ensemble heads; checkpoint and buffer-snapshot format; racing-PLR scorer</li><li>GPU pilot reporting peak VRAM and updates per second; 200M-step DR pilot; backup to object storage</li><li>**`sim-v1`** tag (dynamics, lookup, reward, termination, observations, window, context sampling) and **`generator-v1`** tag</li></ul> | Audit subjects | Generalist; Nemesis search space |
| K2b | 21 | <ul><li>TO-OCP window mode on `sim-v1` at 20 Hz, 100 Hz and γ = 1</li><li>float64 replay</li><li>Label schema stamped with the `sim-v1` hash</li></ul> | Labels | Answer key |

**Deferred core: 34 h, after arXiv**
- TO-OCP lap mode.
- Convergence on all 24 circuits; "at least 23 of 24" is now a demo gate.
- Export schema and Rerun logging.
- JAX FISTA QSS and its bias model.
- ES flat parameter vector and plausibility penalty.
- Activation and injection hooks; blooper logger.
- Pontryagin solution, γ animation and keyboard clip.
- Optional: polishing the specialist to G ≤ 3% (+10–15 h).

**Lockbox rules**
- OCPs run on R-dev only until `generator-v1` is tagged.
- The paper never touches R-test.
- `protocol-v1` is tagged at G1, before the unseal and before any E4 result. It declares:
  - H1 on all 5 E0 seeds at 1B steps, with the median as the headline and the checkpoint chosen on R-dev;
  - G_20, the 20 Hz line/execution split, the per-corner table, and the mirrored and reversed circuits;
  - Nemesis worst-case regret, searched in procedural space only;
  - the R-test lap gaps of the C and E4 arms, as secondary results.

**Cut from the plan**
- H2 and the five-arm curriculum ablation as the headline result.
- The ACCEL editor; OraclePLR scored by QSS; the pure-pursuit tracker (built only if the spike fails).
- P-val; the Q-learning leg of the 1-D braking exercise.
- The plain-MLP critic run; E3 (moves to v2, with a fixed learning rate).
- Estimators 7–11 as headline scores; two of the three tilted generators; A_real and A_eval.
- The atlas as a scheduled release; tiers, resistance map, driver cards and the AI–human bridge.
- Arms-race rounds IV–VIII; Shorts 3–4; the C1 technical report.
- Robustness, the scaling curve and the sandbox, all now optional.

---

## 5. Schedule

### 5.1 Phases (C = shared core, P = paper, D = demo)

| # | Phase | Weeks, ends | h (cumulative) | Track | Background compute | Gate / output |
|---|---|---|---|---|---|---|
| 1 | P0 data rescue, plus a 3-h snapping sanity check | W1–2, 1 Nov | 13 (13) | D | OpenF1 pulls, ≤3 req/s | Data frozen by 25 Oct |
| 2 | K1 | W2–6, 29 Nov | 45 (58) | C | — | Physics gates |
| 3 | K3 | W6–8, 13 Dec | 18 (76) | C | — | PPO parity |
| 4 | Harness, 6 estimators, samplers, E5 maze | W8–13, 17 Jan | 38 (114) | P | Maze runs | First paper figure; sampler parity with JaxUED |
| 5 | Maze note | W13–14, 24 Jan | 10 (124) | P | — | arXiv, plus the workshop if the deadline fits |
| 6 | K2a, then the spike | W14–18, 21 Feb | 40 (164) | C | Mac nights | **Spike gate** |
| 7 | K4 | W18–22, 21 Mar | 45 (209) | C | 4090 tuning sweeps, 10–20 h | **K4 paper gate** |
| 8 | MVP-1 lite | W23, 28 Mar | 4 (213) | D | — | Clip and short post |
| 9 | K5 | W23–28, 2 May | 54 (267) | C/P | GPU pilot, DR pilot, then the E0 families | **`sim-v1`, `generator-v1`** |
| 10 | K2b | W28–30, 16 May | 21 (288) | C | E0 families; 200 pilot labels | Replay within 0.05% in float64 |
| 11 | Pilot audit, label repeatability, power analysis | W30–33, 6 Jun | 25 (313) | P | Pilot rollouts | **G1: `audit-v1` and `protocol-v1`** |
| 12 | Launch labels and curricula; QA | W33–34, 13 Jun | 10 (323) | P | Cloud labels (20–47 h); C runs | QA passed |
| 13 | Analysis: E1, E2, buffer audit, C | W34–38, 11 Jul | 40 (363) | P | E1 and buffer rollouts | **E4 trigger** |
| 14 | Writing | W38–42, 8 Aug | 42 (405) | P | E4 runs launched if triggered | — |
| 15 | Release kit: pip package, data | W43, 15 Aug | 6 (411) | P | — | **arXiv v1** |
| 16 | ICLR formatting (+E4 analysis) | W44–49 | 6 (+16) | P | E4 arms | ICLR 2028 submission, ~late Sep |
| 17 | Deferred core (Release 1 part), unseal, lockbox, 20 Hz fixed-path split, viewer v0, R0, packaging | W44–53, 24 Oct | 82 | C/D | R-test OCPs; ~108 fixed-path solves | **R0, then Release 1** |
| 18 | Deferred core (Release 2 part), probes and knob, Nemesis I–III, poster, Short 2; ICLR rebuttal (8 h) | W54–65, 16 Jan 2028 | 110 | D (+P) | ES search; in-loop TO-OCP; 3 × 50M fine-tunes | **Release 2** |
| 19 | Drivers-lite, viewer v1, video, repo; camera-ready (6 h) | W66–80, 30 Apr 2028 | 140 | D (+P) | Injection-recovery simulations; renders | **Release 3** |

### 5.2 Gates

| Gate | When | Pass | Fail |
|---|---|---|---|
| Data | 25 Oct | Snapshot and manifest complete | Fall back to TracingInsights |
| Spike | ~21 Feb | TO-OCP converges on 20–60 s R-dev windows; minutes and memory per level measured | <ul><li>Over 12 CPU-min per solve: shrink the pool to 2,000 and the buffer to 500</li><li>No convergence: use the s-OCP plus a minimum-time-tail proxy (needs ranking agreement ≥ 0.98 on 200 levels), with the best agent return as the floor, and build the pure-pursuit tracker (3 h)</li></ul> |
| K4 | ~21 Mar | C ≥ 95% over 20 jittered starts; G ≤ 8% against T_cont; 3 seeds learn consistently | Not passed by 25 Apr: switch to a maze-led paper (TMLR or a workshop), and pivot the demo to non-RL content (R0, atlas, drivers) |
| Generalist | End of E0, ~mid-May | Late-phase A_DR completion ≥ 90%; R-dev completion ≥ 80% | <ul><li>Paper: the early-versus-mid fallback written into `audit-v1`</li><li>Demo: extend the showcase seed to 2–3B steps</li><li>Below 50% at 2B: Act 1 becomes adaptation curves</li></ul> |
| G1 | ~6 Jun | Reliability ≥ 0.9 late and ≥ 0.8 mid, within 64 windows; label SD known | More than 4 weeks late: go to TMLR |
| G2 (venue) | 29 Aug | arXiv is out: submit to ICLR 2028 | TMLR or RLC 2028, under the paper cap |
| Nemesis | Release 2 | Certified, and passes the RK4 re-check | Simulator exploits go to the blooper reel |
| Drivers | Release 3 | Split-half ρ ≥ 0.5 and the other gates pass | Pair case studies only |

**Priority rule**
- The paper gets all hours until arXiv or 29 Aug 2027, whichever comes first.
- After that it gets at most 4 h/week until the ICLR deadline, including E4.
- After the deadline it gets at most 2 h/week, except in rebuttal and camera-ready weeks.
- Before arXiv, the only public outputs are the maze note and MVP-1 lite.

**Slip rule.** If any block runs more than 2 weeks over, ship the minimum on time: one figure, a clip of 60 s or less, and a post of 1,500 words or fewer.

### 5.3 Dependency graph

```
LEGEND  ─>  needs        ══>  runs in the background (GPU or CPU)

P0 (W1) ───────────────────────────────────────────────> R0, drivers-lite (after arXiv)
K1 ─> K3 ─> harness + samplers + maze ─> maze note
K1 ─> K2a + spike ─> K2b ══> 200 pilot labels ─┐
K3 ─> K4 ─> MVP-1 lite                         v
K4 ─> K5 ─> sim-v1 / generator-v1 ══> E0 families ─> pilot audit ─> G1 (audit-v1, protocol-v1)
                                                                    │
       ══> labels: 1,412 audit + 4,000 pool ═══> C curricula (oracle arm starts once pool labels exist)
                                                                    v
             analysis (E1, E2, buffer audit, C) ─> writing ─> arXiv ─> ICLR 2028 (══> E4 if triggered)

arXiv ─> deferred core ─> unseal R-test ─> lockbox + 20 Hz fixed-path split ─> Release 1
audit labels ─> QSS bias model + ES ══> Nemesis I–III ─┐
E0 checkpoints + hooks ─> probes + brake knob ─────────┴─> Release 2
P0 ─> pipeline ─> gates ─> drivers-lite ─> video + viewer v1 ─> Release 3
```

### 5.4 Parallelism

| Weeks | You (Mac) | GPU (4090) | CPU |
|---|---|---|---|
| 1–17 | P0, K1, K3, maze, K2a | Maze runs (or on the Mac) | Mac nights: s-OCPs, spike |
| 18–22 | K4 | Tuning sweeps | — |
| 23–30 | K5, K2b | Pilot, then 13 E0-family runs (≤2 at a time) | Mac night: 200 pilot labels |
| 31–38 | Pilot audit, QA, analysis | E1 rollouts; 21 C runs; maze | **Cloud:** ~5,400 labels on a 64-vCPU spot box |
| 39–49 | Writing, ICLR | E4 arms if triggered | — |
| 50–80 | Demo | Lockbox runs, ES, fine-tunes | Mac nights: R-test OCPs, fixed-path solves, certifications; cloud for the in-loop TO-OCP |

---

## 6. Releases timeline

| Date | Track | Release |
|---|---|---|
| ~24 Jan 2027 | Paper | Maze note: arXiv, plus an ICLR 2027 workshop |
| ~28 Mar 2027 | Demo | MVP-1 lite |
| 2 May and 6 Jun 2027 | Internal | `sim-v1` and `generator-v1`; then `audit-v1` and `protocol-v1` |
| **~15 Aug 2027** | Paper | **arXiv v1 and the benchmark:** pip package, labels, checkpoints, per-episode summaries |
| ~late Sep 2027 | Paper | ICLR 2028 submission (arXiv v2 if E4 ran) |
| ~early Oct 2027 | Demo | R0 audit post |
| **~24 Oct 2027** | Demo | **Release 1** (~7 Nov if E4 ran) |
| ~Nov 2027 / ~late Jan 2028 | Paper | ICLR rebuttal / ICLR decision |
| **~16 Jan 2028** | Demo | **Release 2** |
| **~30 Apr 2028** | Both | **Release 3:** main video, viewer v1, repo v1.0 |

**Where to post**
- R0 and drivers: the F1-technical, FastF1 and OpenF1 communities.
- Releases 1–2: RL and ML channels.
- The paper: arXiv.

---

## 7. Failure isolation

| If this stalls | Paper still delivers | Demo still delivers |
|---|---|---|
| Any demo block | **Everything.** The paper needs no demo block | Ship the minimum release |
| PPO on the car (K4 past 25 Apr) | A maze-led paper (E5, the harness and the note), sent to TMLR or a workshop | Non-RL content: R0, the atlas, drivers-lite |
| TO-OCP is slow or will not converge | Smaller pool, or labels bracketed between a proxy and the best agent return | CERTIFIED becomes "best-known feasible lap" |
| Late-phase labels are too noisy | Early and mid-phase claims, plus the buffer audit | Unaffected |
| The paper overruns | It is capped at 4 h/week, then 2 h/week, and moves to TMLR or RLC 2028 | Each demo date slips by at most the capped hours |
| F1 data is lost | — | The P0 snapshot, with TracingInsights as fallback |
| No GPU | Any 24 GB card; a reduced E0 on the Mac (3 seeds × 300M) | Lockbox and probes run on the Mac |

---

## 8. Budget

### Hours

| Bucket | Hours |
|---|---|
| P0 (demo, but must happen in W1) | 13 |
| Shared core: K1 45, K3 18, K2a + spike 40, K4 45, K5 54, K2b 21 | 223 |
| Paper to arXiv: maze block 38, note 10, pilot/G1 25, label QA 10, analysis 40, writing 42, release kit 6 | 171 |
| MVP-1 lite | 4 |
| **Subtotal to arXiv** | **411** |
| Paper after arXiv: ICLR formatting 6, rebuttal 8, camera-ready 6 (E4 adds 16) | 20 (+16) |
| Deferred core | 34 |
| Demo: <ul><li>lockbox 12 · 20 Hz fixed-path split 25 · mechanism 22</li><li>Nemesis 35 · arms race 8 · poster 8</li><li>R0 10 · drivers-lite 45</li><li>viewer v0 5 · viewer v1 28 · video 50</li><li>packaging for Releases 1 and 2: 30 · repo 6</li></ul> | 284 |
| **Total** | **749** (765 with E4) |
| Capacity to 30 Apr 2028 at 10 h/week | ~770, so slack is only ~21 h |

**Cut order** if you fall more than 3 weeks behind. Together these free about 71 h and bring the main video to ~early Mar 2028.
1. Viewer v1 becomes Rerun plus a static page (−20 h).
2. Drivers-lite becomes R0, a detection-limit figure and 2 pair case studies (−30 h).
3. The arms race stops at Nemesis I (−6 h).
4. The video drops to 8 minutes (−15 h).

### Compute

| Item | GPU-h on the 4090 ($0.34/h) |
|---|---|
| K4 tuning sweeps | 10–20 |
| GPU pilot and 200M DR pilot | 1–2 |
| E0, 5 seeds | 6–12 |
| E0-sym, E0-const and E0-short (8 runs) | 10–19 |
| C curricula (21 runs × 1B) | 25–50 |
| Audit inference (E1, buffer audits, A_QSS) | 4–10 |
| E5 maze | 3–6 |
| Demo | 3–8 |
| **Total without E4** | **62–127 (≈ $21–43)** |
| E4 | +12–24 (≈ $4–8) |

| CPU item | Cost |
|---|---|
| ~5,400 labels × 3 starts each (1,200–3,000 CPU-h, unverified until the spike), on a 64-vCPU spot box for 20–47 h | $30–120 |
| Subsets (100 Hz on 200 levels, γ = 1 on 600, repeatability), plus the Nemesis in-loop TO-OCP | $10–25 |
| Mac nights (~8) | $0 |

**Money**
- Expected spend is **≈ $70–190**.
- **Hard cap: $250**, set as a RunPod limit plus a cloud budget alert.
- The pool size shrinks to fit if the spike shows labels are slower than assumed.
- Poster print: $20–40.

---

## 9. What the critics changed

**Reviewer 2**
- **Fatal: the audit used the wrong subjects.** Accepted. A labelled pool of 4,000 levels; real PLR⊥ and SFL runs on it; in-buffer ρ and admission AUROC are now co-primary; lagged-score fidelity added to E0.
- **Partial-ρ endpoint.** Accepted. Raw ρ (absolute and normalised) and RC@10% are the co-primary endpoints. Partial ρ and the two-loading regression are diagnostics. "Oracle-operational" is a reference, not a ceiling.
- **Target and scores shared rollouts.** Accepted. Windows are now disjoint, with 32–64 windows per level and cross-fitting.
- **Static η.** Accepted. η now comes from a simulated replay process, and the real-buffer metrics take its place as primary.
- **JaxUED's operational score.** Accepted. One racing-PLR protocol is used by both the curriculum runs and the audit, with a table of deviations from JaxUED.
- **Discount weighting.** Accepted. Levels start on a straight; strata and score variants are discount-weighted; an undiscounted (γ = 1) target is added.
- **Crashed and clean windows mixed.** Accepted. A hurdle split; p dropped from partialling; adaptive windows. Total regret stays the headline, because it is what UED optimises.
- **The bracket.** Accepted.
  - J*_100 is now the control-rate reference.
  - Every level gets the same 3-start budget, including an agent warm start.
  - Labels are called a "best-known optimum (lower bound on regret)".
  - In the demo, CERTIFIED now means a replay-verified lower bound on the time left.
- **RQ3 missing at submission.** Accepted. C, including an oracle-latest arm, is in the submitted paper. E3 was cut rather than repaired, following the scope critic.
- **Training schedule confounded with phase.** Accepted. E0-const plus schedule covariates.
- **One bespoke domain.** Partly accepted.
  - Added: the maze run with JaxUED's own code at its defaults; critic and preview manipulations inside the racing domain; a pip-installable package.
  - **Rejected for now:** a sparse-reward arm. It would need PPO re-tuned on an untested reward, with no bound on the hours. Claims about reward density are therefore labelled conjecture, and the arm moves to v2.
- **Privileged critic.** Accepted. E0-sym is co-primary, with 3 seeds and all 16 checkpoints.
- **Learnability is near-tautological.** Accepted. It is out of the abstract, has a non-saturating success definition, and is framed as a comparison of targets.
- **Minor issues**
  - Score history is now measured in training steps.
  - **ACCEL is dropped.** The mutant-family audit is rejected, because it costs 1,000 labels for a method the paper does not run.
  - Crossed bootstrap and a pre-computed minimum detectable effect.
  - The IR_info estimate is dropped: receding-horizon solves cost too much, and E0-short covers observability.
  - The E4 trigger needs Holm correction, both DR arms and the audit's units.
  - λ = 0.98 is stored, and JaxUED's defaults are used.
  - Six headline estimators.
  - The maze note is posted first, to establish priority.

**Scope critic**
- **Fatal: hours were about 30% low.** Accepted. The plan is re-baselined to 411 h to arXiv (the critic's 368 h plus 43 h of the reviewer's fixes) and 749 h in total. ICLR 2028 replaces NeurIPS 2027. The main video moves from 21 Nov 2027 to ~30 Apr 2028. Gates are milestone-based, and public commitments use 8 h/week.
- **TO-OCP built before the simulator settled.** Accepted. K2 is split into K2a, the spike and K2b, with the `sim-v1` freeze in between, and labels launch only after G1. The scope critic's second solve pass "only within 0.5%" became a **uniform** agent-warm-start pass on every level, because the reviewer showed the conditional version biases labelling effort.
- **Demo work inside the core.** Accepted, as the 34-h deferred-core block.
- **Paper method tested too late.** Accepted. The maze, harness and samplers now run Dec–Jan.
- **K4.** Accepted. The paper gate is C ≥ 95% and G ≤ 8%, with tuning on the GPU configuration and a 25 Apr pivot date.
- **GPU configuration.** Accepted. 8,192 × 64 batches, checkpoints from 10M steps, at most two seeds at a time, and VRAM reported by the pilot.
- **Label noise.** Accepted. Repeatability study, noise floor and reliability correction, and the E4 trigger is gated on them.
- **Numerical traps.** Accepted. B-spline track lookup, float64 replay, corridor margin.
- **Nemesis chasing QSS error.** Accepted. Bias model, motif windows, TO-OCP in the loop, Nemesis I–III, 35 h.
- **Failure isolation covered content, not schedule.** Accepted, as the priority rule.
- **Production underestimated.** Accepted: the video is now 12 minutes and 50 h, drivers-lite is 45 h, and there are two Shorts. Partly rejected: the Three.js viewer stays, because you asked for a web viewer, but it is first in the cut order and Rerun is the fallback.
- **Release cadence.** Accepted. Three releases plus the R0 post.
- **Padding.** Accepted, except for the symmetric critic, which stays because the reviewer needs it. The estimator set is the union of both critics' lists.
- **Badges.** Changed to CERTIFIED, MODEL-DEPENDENT and POST HOC. FAIR is folded into CERTIFIED.
- **Minor issues**
  - Best agent return as the fallback lower bound.
  - OCPs on R-dev only until `generator-v1`.
  - Car context, μ and x0 fixed per level.
  - A_QSS shrunk.
  - E4 re-costed at 16 h, because C now provides the base arms.
  - 20 Hz fixed-path split, with D8 at 25 h.
- **Money.** The cap rises from $150 to $250, because the pool adds about 4,000 labels.

Source plans: [plan-v1.md](plan-v1.md) and [novelty-and-outputs.md](novelty-and-outputs.md).