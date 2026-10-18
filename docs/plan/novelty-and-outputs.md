> Companion to [plan-v1.md](plan-v1.md). Three ideators proposed unique outputs, a curator shortlisted 17 claims, and three skeptics searched papers, GitHub, YouTube and forums trying to refute each one. All 17 came back "partly done": close work exists, and the novelty lies in a specific combination or framing, spelled out below.

# ApexZero: what's novel, what's cool, what's hard, and what only you can make

## 1. What's actually novel

**Verdict first.** All six core claims held up against prior art, but only in narrower form than the plan states. On its own, none of them is a brand-new idea. What you can defend falls into three kinds:
- **New combinations** that nobody has put together (C1, C4).
- **Classical tools applied to a learned policy for the first time** (C3, C5).
- **One genuinely new experiment** that the curriculum-learning (UED) field has explicitly asked for (C2b).

Say "to our knowledge" every time you claim one of these.

**C1. Zero-shot on real F1 layouts after procedural-only training, scored against an OCP.** *New combination.*
- **Defensible wording:** "The first car-racing RL study to report zero-shot lap times on real F1 layouts (flat, TUMFTM geometry) as a % gap to a minimum-time OCP on the identical vehicle model. The policy is trained only on procedural tracks, the gap is split into line and execution, and a scaling curve over training-track count is included."
- **What you're building on:**
  - **Hoeppke 2022 (Oxford DPhil).** Your dossier missed this one. It used PPO plus collocation on one shared bicycle model and produced valid trajectories on 18 of 20 F1 circuits, 12 of them unseen. But it trained on real layouts and scored completion, not lap-time gap.
  - Jiang et al. 2021, CarRacingF1: toy physics, scored by return.
  - Evans et al. 2024 (F1TENTH, 1/10 scale).
  - Green et al. 2026 (drones).
- **Add:** each R-test circuit's distance to its nearest training tracks. That shows how out-of-distribution "zero-shot" really is.

**C2. Oracle-regret curriculum, plus a ground-truth test of regret estimators.** *Half done before, half new.*
- **Already done:** choosing levels by exact regret, in grid worlds with shortest-path oracles (Beukman et al. 2024, ReMiDi; Abdel Sadek et al. 2025).
- **New (a):** a continuous physics oracle, so regret comes out in seconds.
- **New (b), the real prize:** the first rank-correlation test of positive value loss, MaxMC and learnability against exact regret, on about 2,000 OCP-solved tracks across training. That fills the gap Rutherford et al. 2024 ("No Regrets") left open.
- **Trap:** ReMiDi showed that perfect-regret PLR fixates on regret the agent can never remove. Score against T_OCP@20Hz, not raw T_OCP.
- **Implication:** H3 is a side result. The estimator test is the headline.

**C3. Splitting a learned policy's gap into line and execution.** *Classical method, new application.*
- Fixed-path vs free-path lap simulation is standard practice (Dal Bianco et al. 2019; Massaro & Limebeer 2021).
- Evans et al. (TAL, 2023) only say qualitatively that RL loses time in corners. Nobody has re-solved a fixed-path OCP on an RL agent's own driven path.
- Requirements:
  - publish how sensitive the split is to path smoothing;
  - multi-start the free OCP so G_line ≥ 0;
  - give the fixed-path OCP the same actuator lags.
- Label the "OCP + tracker under −5% grip" experiment as a **replication** of Song et al. 2023.

**C4. Per-corner, per-phase driver time loss from public 3.7 Hz data.** *New combination.*
- **Defensible wording:** "The first open analysis to estimate drivers' time loss by corner type and phase from public timing data, with teammate-identified hierarchical Bayesian effects linked across driver moves, and every claim gated by injection-recovery at the real OpenF1 sampling distribution."
- **Prior work:**
  - AWS "Fastest Driver" 2020: lap level, closed method.
  - van Kesteren & Bergkamp 2023 and Lindner et al. 2026: Bayesian, but at result or lap level.
  - Taylor & Secco 2026: sector-level teammate differences.
  - Community corner-delta scripts; Coach Dave Delta, which already splits corners into phases from full-rate sim data.
- **Don't** lead with "physics-calibrated". The reference cancels in teammate contrasts.

**C5. Steering the braking point in metres, plus a memorisation probe.** *Known method, new domain, new yardstick.*
- **Prior probe-and-steer work:** Bush et al. 2025 and Taufeeque et al. 2024 (Sokoban), Mini/Turner et al. 2023 (maze). The closest continuous-control case is Haon et al. 2025, who found "speed" directions in robot VLA policies.
- **New here:**
  - dose-response measured in metres against the OCP braking point;
  - side effects reported in physical units;
  - absolute-s decodable from the specialist's activations but not the generalist's.
- Policy Gradient Steering (arXiv 2607.27574) shows steering can fail even in gridworlds, so a null or entangled result is still publishable.

**C6. The public F1 position feed has no lateral line.** *Known fact, first numbers.*
- The FastF1 maintainer said this in 2022 (Discussion #116), and the OpenF1 docs carry a caveat.
- **New:** an audit across several circuits and seasons, with positive controls:
  - staggered grid slots, which are metres apart;
  - side-by-side cars taken from overtake data;
  - injected 1–3 m offsets.
- **Consequence:** Shehadeh et al. 2026's "expert racelines" are probably the timing system's track model, not the drivers.

---

## 2. What's cool: the moments people remember

1. **The lockbox opens.** Twelve circuits the agent has never seen, one bar each: % from the optimum, split into line and execution. The finding reads like "right line through 130R, brakes early for the hairpin". Then Monza reversed.
2. **The specialist dies at La Source.** The Hockenheim specialist crashes at Spa's Turn 1 and the generalist gets through. Then the inside view: lap position can be read off the specialist's neurons (place cells) but not the generalist's ("distance-to-braking" cells).
3. **The brake knob.** Add one vector to a 0.58M-parameter network and the car brakes N m earlier, measured against the OCP braking point, with the side effects visible.
4. **Cut grip by 5%.** The OCP's "perfect" trajectory, replayed by a tracker, spins. The policy adapts. This is a replication, but it is the clearest picture of "a trajectory vs a feedback law".
5. **Nemesis I.** A circuit F1 could plausibly build, where the AI loses seconds against a certified optimum. It is shrunk to a 300 m "fear motif", and then the arms race begins.
6. **F1 data can't see the line.** Cars on a staggered starting grid sit metres apart. In the public X/Y feed they collapse onto one polyline.
7. **The racing line changes phase.** Morph a hairpin and the softest Hessian mode goes soft. The optimal line flips from single to double apex, with a kink in T(morph).

---

## 3. What's complex: rabbit holes to explore

**1. One ODE, two compilers (JAX and CasADi).**
- **What:** one f(x,u,κ,p) drives both the RL simulator and the OCP.
- **Why hard:** 1e-5 parity needs every saturation smoothed (the friction-ellipse smooth-min, the kinematic blend below 4 m/s). The Frenet frame is singular at 1 − nκ = 0, which is why the plan has the w·κ ≤ 0.5 filter.
- **You learn:** how integrator error disguises itself as an "optimality gap".
- **Explore:** which fakes more of G, RK2 error at 5 ms or collocation error at 3 m? Bound both.

**2. Minimum time is Pontryagin with walls.**
- **What:** direct collocation in s, solved with IPOPT.
- **Why hard:** the problem is non-convex, with local optima and convergence trouble.
- **You learn:** the multipliers are the costates and shadow prices. The corridor dual is literally how hard the wall pushes back on lap time, and lam_p gives dT*/dClA by the envelope theorem. λ_v = ∂T/∂v behaves like a canonical momentum.
- **Explore:** where does the first-order dual prediction for +1 m of width disagree with a re-solve? Does that line up with an active-set change, such as a new apex touching the wall?

**3. Differentiating through a lap simulator.**
- **What:** QSS = FISTA min-curvature QP + a forward-backward g-g-v pass in `lax.scan`.
- **Why hard:** min and max are non-smooth. The logsumexp temperature trades bias against gradient quality. You also have to choose between unrolled and implicit differentiation.
- **You learn:** the implicit function theorem in practice, and how an optimiser exploits the bias between QSS and the OCP (Goodhart).
- **Explore:** at what soft-min temperature does QSS dT/dκ(s) match OCP finite differences, and does it break at apexes?

**4. Bellman against Pontryagin on one braking zone.**
- **What:** P3's 1-D braking problem, solved three ways.
- **Why hard:** γ = 0.995 at 20 Hz gives a 10 s horizon. Too short and late braking never appears. The critic outputs discounted progress, not seconds.
- **You learn:** HJB as the common parent of both. The value function as time-to-go.
- **Explore:** derive analytically the braking point that the discounted-optimal policy chooses, as a function of γ. Check that PPO lands on it.

**5. Curricula as a minimax-regret game.**
- **Why hard:**
  - regret estimators are noisy proxies;
  - the irreducible-regret trap;
  - QSS error can look like agent error.
- **You learn:** the game theory behind UED, estimator bias, and rliable statistics.
- **Explore:** does Spearman(PVL, true regret) fall as the agent improves? In other words, do the estimators work early in training and fail late?

**6. The racing line as an elastic beam.**
- **What:** the min-curvature line minimises ∫κ², so it is an elastica. The min-time line departs from it. Compute:
  - the kernels ∂v*(s)/∂κ(s');
  - Hessian eigenmodes, by HVP + Lanczos in the free subspace (a reduced-Hessian KKT lesson).
- **Why hard:** the response is non-smooth wherever a constraint switches, and QSS is a point-mass model, not the bicycle.
- **You learn:** Green's functions. Hyperbolic structure (braking propagates upstream at v²/2a) against elliptic structure (line shape propagates both ways). Soft modes and critical slowing down.
- **Explore:** at the single-to-double-apex switch, does the lowest eigenvalue go to zero linearly (a fold), or is it a smooth crossover?

**7. Seeing through a 3.7 Hz keyhole.**
- **What:** ICP registration, then an RTS smoother on [s, v, a] anchored to sector times accurate to 1 ms. Speed arrives in whole km/h and the brake channel is binary.
- **Why hard:** you are chasing 5–20 ms effects with about 270 ms between samples.
- **You learn:** Kalman and RTS smoothing, and simulation-based calibration.
- **Explore:** compare your injection-recovery detection limit with a Cramér–Rao bound computed from the sampling distribution. How close is the pipeline to the information limit?

**8. Statistics of a sparse teammate network.**
- **What:** the driver + team-season model is the AKM model from labour economics. It is identified only through drivers who move teams.
- **Why hard:** with few movers, limited-mobility bias inflates the spread of driver effects.
- **You learn:** NUTS, heteroscedastic hierarchies, and the fact that comparison uncertainty is effective resistance (Laplacian pseudoinverse, Kirchhoff's laws).
- **Explore:** does R_eff predict the posterior SD of θ_a − θ_b? Which hypothetical 2027 driver swap would cut the grid's Kirchhoff index the most?

**9. Interpretability in a small control net.**
- **What:** probes, then causal interventions.
- **Why hard:**
  - the injection site matters: add after LayerNorm, or the vector gets normalised away;
  - directions may be entangled.
- **Key control:** mirrored and warped tracks tell you whether a unit codes place or local shape.
- **Explore:** is the braking direction the same vector at 120 and 300 km/h, or does the network use codes specific to each speed?

---

## 4. Unique outputs

### Flagship: **Nemesis, the certified circuits this AI fears**

**What it looks like**
- **Poster, "Nemesis I–VIII":** eight F1-plausible circuits drawn like official track maps. Each card shows:
  - name and length;
  - OCP-certified lap time;
  - the AI's loss in seconds;
  - the failure point;
  - an inset of the minimal fear motif;
  - its line/execution split (from C3);
  - the probe readout at the failure point (from C5).
- **Lineage strip:** the eight arms-race rounds side by side. Does each new nemesis find a new motif, or the same one in disguise?
- **60–90 s video:** the agent ghost against the OCP ghost on Nemesis I, building to the corner where it goes wrong. Then the fine-tuned agent, then Nemesis II.
- **Science panel:** certified worst-case regret for each curriculum arm. UED theory targets *minimax* regret, yet papers evaluate on hand-picked test levels. Nemesis gives a searched lower bound on the maximum regret over F1-plausible tracks, with every point certified by the OCP.

**Why no one else has it.** The prior-art verdict was "partly done", with high confidence.
- Searching track space to break a frozen agent goes back to Uesato et al. 2019, but their objective was TORCS crash probability.
- ARLPCG (2021), PAIRED, ACCEL and POET produce training curricula, not artefacts.
- Loiacono et al. 2011 evolved tracks for entertainment.
- DiME's AI-generated F1 track video has no physics.
- **What's new:**
  - the objective is lap-time regret against an OCP on identical dynamics;
  - an F1-plausibility penalty (Wasserstein distance to real-circuit statistics);
  - delta-debugged minimal motifs;
  - an arms race in which every round is certified.
- **Wording:** "Nemesis finds F1-plausible circuits that maximise a policy's lap-time regret against an optimum certified by optimal control on the same car model. It shrinks each one to the shortest segment that still costs the time, and iterates the search with fine-tuning to show which motifs a fixed-preview policy keeps failing on."

**Why this one is the flagship.** It is the only output that needs all three of your unusual assets at once: a jittable generator, a zero-shot policy, and an optimum computed on identical physics. Take one away and you get Uesato (no certificate), DiME (no agent) or a curriculum (no artefact). It also feeds three other threads:
- **Reach (rabbit hole 6):** is the failure point beyond the preview horizon d_max?
- **Probes (C5):** what is happening in the network at the failure point?
- **Minimax regret (C2):** each curriculum arm's worst case.

**Needs and effort**
- **Reuses from the plan:** κ-grammar continuous parameters, feasibility filters, the F1-likeness Wasserstein statistics (fitted on R-dev only), JAX QSS, OraclePLR's ρ score, the CasADi OCP, and the Three.js viewer.
- **Adds:**
  - an evosax CMA-ES/OpenAI-ES loop, vmapped over about 1k candidates per generation (optimise open strips, close only the finalists);
  - OCP certification of the top 5 with 20 jittered starts, plus re-verification with RK4 at 1 ms;
  - a delta-debugging motif minimiser;
  - eight fine-tunes of about 50M steps each on the 4090, a few dollars;
  - the poster, via matplotlib to SVG.
- **Total:** about 30 h (3 weeks). **Wow: 9/10.**

**Rules**
- Use T_OCP@20Hz, or an irreducible-gap control, as the denominator. That avoids the ReMiDi trap.
- Every QSS-flagged nemesis gets OCP-certified.
- The first things the optimiser finds will be simulator exploits (w·κ → 0.5, preview aliasing). Those go in the blooper reel, not on the poster.
- Any R-test number from an arms-race model is labelled post hoc.
- **Bonus rabbit hole:** try BPTT through the closed loop on short windows and plot the gradient variance blowing up (Lyapunov exponents; Metz et al., "Gradients are not all you need"). That figure explains why you used ES.

### Supporting outputs

**A. Branching Ghosts: a take-the-wheel coach. Wow 8.**
- **What it looks like:** a browser time trial.
  - Press Space and 8 policy ghosts fan out from your exact car state.
  - After each lap, a per-corner receipt shows entry, apex and exit losses that **sum exactly** to the corner loss (they telescope, as in the performance-difference lemma), plus a line/execution split from a JS fixed-path QSS.
  - A live readout shows "lap time if the AI took over now".
- **Prior work:** Track Titan's Coaching Flows already costs mistakes with an undisclosed simulation, so don't claim the first counterfactual coach. Also MOZA and Trophi. The maths is Kakade & Langford 2002 and DAgger.
- **Claim:** the first *open*, exactly additive coach that forks a policy from the driver's own state.
- **Needs:** the sandbox (ladder #5, 2.5 weeks) plus 2.5 weeks:
  - recovery-start training (large β, r and ξ jitter);
  - a critic-ensemble flag for out-of-distribution human states;
  - a distilled time-to-go head.
- **Nice link:** the nemesis circuits can ship as playable levels here.

**B. The Price of Tarmac. Wow 8.**
- **What it looks like:**
  - one atlas page per circuit, with a price on every corner exit in ms per metre of extra width;
  - circles sized by FIA lap deletions;
  - a cover scatter of price against deletions for about 150–180 corners;
  - a "where the gravel should go" list.
- **Prior work:** RaceFans' 2023 counts (695 infringements), Motorsport Magazine's deleted-vs-clean laps at Austria, and Gulisano et al. 2025, who cost *narrower* corridors on one sector.
- **New:** a physics price, tested against where race control actually deletes laps.
- **Free natural experiment:** the gravel added at Austria T9/T10 in 2024, before vs after.
- **Needs:**
  - race-control text for races and practice (a cheap addition to P0);
  - turn-number → s tables;
  - 360–720 warm-started re-solves (1–2 nights on the Mac);
  - a negative-binomial model with lap exposure and monitored-corner coding.
- **Effort:** 1.5–2 weeks. It sidesteps the snapping problem completely.

**C. Normal modes of a racing line. Wow 7.**
- **What it looks like:**
  - Manim animations of Suzuka's line vibrating in its soft and stiff modes;
  - a stiffness-map poster per circuit;
  - a bifurcation diagram;
  - a "poke the track" page.
- **Prior work:** Aggarwal & Gerdes 2024 (lateral compliance of the racing line). Credit them, because the stiffness map overlaps.
- **New:** full response kernels, eigenmodes, and the bifurcation as a corner morphs.
- **Effort:** about 3 weeks. This is the physicist's signature piece, and it is what tests the reach hypothesis for the nemesis motifs.

**D. Learning spacetime tower. Wow 7.**
- **What it looks like:**
  - a kymograph of the line locking in, corner by corner, on a circuit the agent never trained on;
  - a star-trails still of every checkpoint's path converging on the white OCP line;
  - a 20 cm 3D-printed ribbon tower of the paths stacked by training step, with Suzuka's shape on top.
- **Prior work:** the method is McGrath et al. 2022 (AlphaZero chess). Ghost swarms on *training* tracks are Yosh's signature.
- **New:** an unseen track, measured against the OCP, compared across curricula. The tower is an art claim, not a research claim.
- **Needs:** about 200 checkpoints at 2.3 MB each, a trimesh STL export, and a $40–100 print. 1.5–2.5 weeks. Build it on R-dev first.

### Good ideas already done: avoid them or reframe them
- **"Same dynamics for RL and OCP on F1 tracks"**: Hoeppke 2022. Cite it.
- **Oracle-regret PLR as an idea**: ReMiDi 2024 and Abdel Sadek 2025. Reframe as the continuous oracle plus the estimator test.
- **Adversarial track search**: Uesato 2019. Reframe as certified regret.
- **"The feed has no lateral line" as a discovery**: FastF1, 2022. Reframe as the first quantified audit.
- **A track that equalises the field**: Majid & Crawfis 2024. Evolved F1-style tracks with a lap simulator: Harper 2024.
- **Stiffness or compliance map of the line**: Aggarwal & Gerdes 2024.
- **Circuit power and downforce sensitivity**: F1 Tech Tuesday and Canopy. Only an open, backtested version is new.
- **Per-corner phase coaching**: Coach Dave Delta, Track Titan.
- **Teammate graphs**: racemate.io. Effective-resistance theory: Hendrickx et al. 2019.
- **Place-like units in RL**: Kappel & Cheng 2024, Banino et al. 2018.
- **"AI generated an F1 track"**: DiME. **Converging ghost swarms**: Yosh.
- **Raw track-limit counts per corner**: RaceFans.

---

## 5. How the flagship changes the roadmap

- **P5 (generator):** expose the κ-grammar's continuous parameters as a flat, ES-friendly vector. Make the F1-likeness Wasserstein distance a callable penalty, fitted on R-dev only. About +3 h.
- **P6 (curriculum):** pre-register "certified worst-case regret per arm" in `protocol-v1` as a secondary metric. The search runs on procedural space only, so it never touches the lockbox. Score against T_OCP@20Hz.
- **New P8.5 "Nemesis" (weeks 28–30, about 30 h):** run it after P8, so every nemesis gets a line/execution split and a probe readout. Run the arms-race fine-tunes on the 4090, for under $10.
- **P9 (release):** use the Nemesis poster as a second hero figure next to the 12-circuit grid. Add a video beat between "Lockbox opens" and "Inside its head". Add a fourth short, "The track this AI fears".
- **Budget:** pay for it by cutting ladder #6 (SAC/QR-SAC) and #4 (hybrid score). The total stays at about 330 h. Branching Ghosts and the Price of Tarmac go after release.

---

## Appendix: prior-art verdicts with sources

### C1. Zero-shot real F1 circuits from purely procedural training, scored against an OCP on identical dynamics  (partly-done, medium confidence)

**Defensible claim:** To our knowledge, this is the first car-racing RL study to report zero-shot lap times on real F1 circuit layouts (flat, TUMFTM geometry) as a percentage gap to a minimum-lap-time optimal-control solution computed on the identical vehicle model. The policy is trained exclusively on procedurally generated tracks. The gap is split into line and execution loss, and a curve shows how it scales with the number of training tracks. Prior work trained on real layouts and scored completion (Hoeppke 2022; Evans et al. 2024), scored zero-shot F1 layouts by return in toy physics (Jiang et al. 2021), or compared RL with optimal control on fixed or drone tracks (Song et al. 2023; Green et al. 2026).

**Closest prior work:**
- Hoeppke, 'Optimal Control and Reinforcement Learning for Formula One Lap Simulation' (Oxford DPhil thesis, 2022) (https://ora.ox.ac.uk/objects/uuid:491a5bb1-db1b-4cf6-b6f2-0ec06097ac9d/files/dpr76f389g): Closest single precedent, and it was missing from the dossier. One PPO policy and a direct-collocation minimum-lap-time solver share the same bicycle model and curvilinear track model. The policy trained on 8 real 2022 F1 layouts (constant width) and found valid trajectories on 18 of 20 F1 circuits, 12 of them unseen. It differs in four ways: training used real tracks, not procedural ones (its straight/curve generator served only toy L/Z/S tracks); the score is distance achieved or completion, never a lap-time % gap to the OCP; the best checkpoint was picked per test track; and RL was used to warm-start collocation.
- Jiang et al., Replay-Guided Adversarial Environment Design (Robust PLR), CarRacingF1, NeurIPS 2021 (https://arxiv.org/abs/2110.02439): Procedural Bezier tracks, then zero-shot on 20 real F1 layouts, but in Box2D toy physics with constant width, scored by return. Later UED papers (CLUTR; arXiv 2502.05726) reuse this benchmark, still scored by return.
- Evans et al., 'Unifying F1TENTH Autonomous Racing: Survey, Methods and Benchmarks' (2024) (https://arxiv.org/abs/2402.18558): An end-to-end RL agent trained on one 1:10-scale F1 layout (GBR) was tested on unseen scaled F1 layouts (AUT, ESP, MCO). Its lap times are compared in seconds with an optimisation-plus-pure-pursuit baseline in the same simulator; by my arithmetic from their Table II it is roughly 19-29% slower. Differences: training is not procedural, the baseline is not a min-time OCP with matched actuator constraints, no % gap is reported, and the scale is 1/10.
- Green, Xing, Messikommer, Romero, Scaramuzza, zero-shot generalist drone racing (arXiv 2606.27348, 2026) (https://arxiv.org/abs/2606.27348): A procedural generator plus a learning-progress curriculum, zero-shot on unseen real tracks. It is compared with optimal control (MPCC/MPCC++) on only one track (SplitS, 5.42 s vs 5.38 s) and uses a success-times-speed score, not an OCP gap across a test set. Drones, not cars.
- Song et al., 'Reaching the limit in autonomous racing: OC vs RL' (Science Robotics 2023) (https://arxiv.org/abs/2310.10943): Compares RL with optimal control on the same drone model, on fixed tracks, with no zero-shot test.
- TC-Driver (2022) and Environment as Policy (Wang et al., ICRA 2025) (https://arxiv.org/abs/2410.22308): Unseen-track racing policies compared against MPC or domain randomisation, on 1/10 cars or drones. No real F1 layouts and no matched-dynamics minimum-time gap.

### C2. Oracle-regret UED, and a ground-truth test of regret estimators  (partly-done, high confidence)

**Defensible claim:** "Oracle-regret curricula have so far been tested in grid worlds, where exact optimal returns come from shortest-path or graph solvers (Beukman et al. 2024; Abdel Sadek et al. 2025). We take the idea to continuous-control racing. Here the oracle is a physics minimum-lap-time solver, so regret is a continuous lap-time gap. We also give what we believe is the first direct test of common UED regret estimators (positive value loss, MaxMC, learnability) against ground-truth regret: their rank correlation with OCP-computed lap-time regret on about 2,000 optimally solved tracks during training. Caveats: regret here is measured in lap time, not discounted return, and the OCP optimum is model-specific and locally optimal."

**Closest prior work:**
- Abdel Sadek, Farrugia-Roberts, ..., Dennis 2025, 'Mitigating Goal Misgeneralization via Minimax Regret' (RLC 2025) (https://arxiv.org/abs/2507.03068): This directly refutes the first half of the claim. The paper runs PLR-perp and ACCEL with an 'oracle-latest' regret estimator, which scores levels by exact maximum expected return minus the agent's recent return. A graph-algorithm solver computes the exact return on every procedurally generated level. The setting is discrete grid worlds. It also compares against 'max-latest', a sample-based estimator similar to MaxMC, but only indirectly: by downstream behaviour, with no correlation or error analysis against true regret.
- Beukman et al. 2024, 'Refining Minimax Regret for UED' (ReMiDi, ICML 2024) (https://arxiv.org/abs/2402.12284): Runs PLR-perp with 'perfect Monte Carlo regret' as the level score in mazes. The optimal return comes from the shortest path, and regret is that return minus the agent's average return. It finds that perfect-regret PLR fixates on levels with high irreducible regret (T-mazes, blindfold levels). That pathology bears directly on OraclePLR. It does not compare PVL or MaxMC against exact regret.
- Rutherford et al. 2024, 'No Regrets' / SFL (NeurIPS 2024) (https://arxiv.org/abs/2408.15099): Correlates PVL and MaxMC with success rate and learnability, not with true regret. Its 'Perfect (Oracle) Regret' arm scores levels by 1-p(success) on solvable levels, a binary proxy. The authors say oracle regret 'cannot be easily measured' in multi-agent JaxNav. This confirms the open gap the plan cites.
- Mead, Lacerda, Foerster, Hawes 2025, MNA regret approximation (NeurIPS 2025) (https://arxiv.org/abs/2601.14957): Proposes a new regret estimator, Maximised Negative Advantage, tested on Minigrid, Key-Minigrid and Sokoban. It is judged by zero-shot test performance only, with no ground-truth regret comparison. The same holds for TRACED (arXiv 2506.19997) and PACE (arXiv 2605.01358).
- Wang, Xing, Messikommer, Scaramuzza, 'Environment as Policy' (ICRA 2025); Green et al. arXiv 2606.27348 (https://arxiv.org/abs/2410.22308): These are the closest racing-specific curricula. One uses a learned track-shaping policy rewarded by the racer's performance percentiles, the other learning-progress task switching. Neither scores tracks by a time-optimal planner. Dantu et al. 2025 (arXiv 2510.20008) use point-mass min-time trajectories as a proxy reward, not as a level score.
- Teoh et al. 2025, CENIE (PLR on CarRacing, zero-shot on 20 F1 tracks) (https://arxiv.org/abs/2502.05726): UED on car racing evaluated on real F1 layouts, using approximate (PVL-style) regret plus novelty. It uses no physics oracle.

### C3. Line-vs-execution decomposition of an RL agent's gap via a fixed-path OCP  (partly-done, medium confidence)

**Defensible claim:** To our knowledge, this is the first time a learned racing policy's gap to the minimum-time optimum has been split into line choice and execution. We do it by re-solving a fixed-path OCP on the policy's own driven path, under the same vehicle and actuator model, and report the split per corner type on real F1 layouts. The fixed- vs free-trajectory idea comes from classical lap-time simulation; what is new is applying it to an RL agent.

**Closest prior work:**
- Dal Bianco, Bertolazzi, Biral, Massaro 2019 (VSD), fixed- vs free-trajectory minimum-lap-time methods; Massaro & Limebeer 2021 review 'Minimum-lap-time optimisation and simulation' (https://www.research.unipd.it/retrieve/e14fb26e-b9c2-3de1-e053-1705fe0ac030/2021%20-%20Minimum-lap-time%20optimization%20and%20simulation%20PP.pdf): Solving a fixed-trajectory problem (best speed on a given path) and comparing it with the free-trajectory optimum is standard lap-time-simulation practice. The motorcycle 'optimal manoeuvre' work checks it against measured rider data. The decomposition method itself is therefore not new, but none of this work applies it to a learned policy.
- Kapania, Subosits, Gerdes, sequential two-step algorithm (path, then forward-backward speed), arXiv 1902.00606 (https://arxiv.org/pdf/1902.00606): Splits lap time into path and speed sub-problems and compares the result with a professional driver's path. It is a planner, not a diagnosis of an agent's gap.
- Piazza et al., FBGA forward-backward solver with generic g-g limits (RA-L 2026, arXiv 2509.26428) (https://arxiv.org/abs/2509.26428): Time-optimal speed on a fixed path, within 0.11-0.36% of the OCP. It is the obvious tool for 'best time on the agent's path', but it is used on planned paths, not on RL rollouts.
- Evans, Engelbrecht, Jordaan: TAL (RA-L 2023, arXiv 2306.07003); 'Comparing deep RL architectures for autonomous racing' (MLWA 2023); 'Safe RL for high-speed racing' (2023) (https://arxiv.org/abs/2306.07003): Closest RL-side analysis. They plot RL speed profiles and line deviation against an optimal min-curvature + min-time planner, and say qualitatively that lower corner speeds cause the lap-time gap. There is no quantitative line/execution split and no fixed-path re-solve.
- Song et al., Science Robotics 2023 (arXiv 2310.10943) (https://arxiv.org/abs/2310.10943): RL vs optimal control on the same model, with robustness explaining the difference. The gap is not decomposed into path vs execution. The plan already marks its robustness companion experiment as a replication.
- Remonda, Veas, Luzhnica, PLOS ONE 2021: human vs RL in a professional simulator (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7857611/): Distance-domain comparison of RL and human laps sector by sector. Descriptive only, with no optimal-control reference and no decomposition.

### C4. Physics-calibrated, teammate-identified per-corner driver time loss from public telemetry  (partly-done, medium confidence)

**Defensible claim:** To our knowledge, this is the first open analysis to estimate F1 drivers' qualifying time loss by corner type and phase from public timing data with calibrated uncertainty. It uses teammate-identified hierarchical Bayesian effects linked across driver moves. Every claim must pass an injection-recovery test at the real OpenF1 sampling rate, a placebo test and a split-half reliability gate, and nothing below the resulting error budget is reported. Per-corner deltas (community tools, broadcast graphics) and lap- or result-level Bayesian rankings (van Kesteren & Bergkamp 2023; Lindner et al. 2026) already exist; this work combines the resolution of the first with the statistical rigour of the second. Do not lead with 'physics-calibrated' as the novelty for rankings: the reference cancels in teammate contrasts.

**Closest prior work:**
- AWS/F1 'Fastest Driver' (2020) (https://www.formula1.com/en/latest/article.hamilton-schumacher-senna-machine-learning-reveals-the-fastest-f1-driver-of.3DwwPLW4glCmlunjciH1Cz.html): Teammate-normalised qualifying pace linked across driver moves. Lap level only, method closed, no corner or phase resolution.
- F1/AWS 'Car Performance Scores' and High/Low-Speed Corner Performance graphics (2020) (https://www.formula1.com/en/latest/article.rob-smedley-on-car-performance-scores-and-the-data-behind-the-latest-f1.5QoyFi4KL63EULKM4u5mGL.html): Splits performance by corner speed band (slow <125 km/h, fast >175 km/h) from full team telemetry. Car level, not driver level; closed, with no uncertainty. F1's 2015 Race Performance Ratings scored cornering from lateral g.
- van Kesteren & Bergkamp 2023; Lindner et al. 2026 (Bayesian driver/constructor models) (https://arxiv.org/abs/2608.04629): Hierarchical Bayesian driver-vs-car separation, built on race results (2203.08489) or fastest qualifying laps plus ranks (2608.04629). Lap or result level, no corner resolution, no physics reference.
- Taylor & Secco, 'Can Unsupervised Learning Methods Reveal Distinct Driving Styles Among F1 Drivers?' (IJMO 2026) (https://hira.hope.ac.uk/id/eprint/4949/): FastF1 telemetry with intra-team relative sector differentials, used to remove the car, then clustering into 3-5 driver types. Sector-level, no corner-type x phase effects, no uncertainty model.
- Zahn et al., 'The Curse of the Second Red Bull' (Wharton Sports Analytics Journal, 2025) (https://wsb.wharton.upenn.edu/?p=12273): Locates where a teammate loses lap time (claims high-speed corners), but for one team only, and no hierarchical model was visible.
- Community tools: ppm98dev/F1 CornerTimeComparator, TracingInsights corner analysis, GrandPrix247/PlanetF1 speed-band teammate splits (https://github.com/ppm98dev/F1): Per-corner time and speed deltas between drivers from FastF1. No car model, no uncertainty, no cross-team identification.
- Kegelman, Harbott & Gerdes 2017; Aggarwal & Gerdes 2024 (https://ddl.stanford.edu/publications/journal/insights-vehicle-trajectories-handling-limits-analysing-open-data-race-car): Pro-driver laps compared with minimum-time references using full-rate GPS on vintage and amateur cars. Physics-referenced driver analysis exists, but not for F1 and not from public timing data.
- Coach Dave 'Delta' Auto Insights (sim racing) (https://coachdaveacademy.com/documentation/how-to-use-auto-insights-ai-coaching-in-delta/): Commercial per-corner breakdown into braking, entry, apex and exit time loss for one driver, from full-rate sim telemetry. The corner x phase split is precedented; doing it for F1 from about 3.7 Hz public data with statistics is not.

### C5. Causal braking-point steering and a memorisation probe inside a racing policy  (partly-done, high confidence)

**Defensible claim:** "Probe-and-steer methods from Sokoban and maze agents (Taufeeque et al. 2024; Bush et al. 2025; Turner et al. 2023) and robot VLAs (Haon et al. 2025) carry over to a continuous-control racing policy. Adding a linear-probe 'distance-to-braking-point' direction to the policy's activations moves its braking point by X m per unit, measured against the optimal-control braking point, with side-effects reported in physical units. We also show that absolute lap position is linearly decodable from a single-track specialist's activations but not from the zero-shot generalist's. That is a direct internal signature of memorisation versus generalisation."

**Closest prior work:**
- Bush et al. 2025, 'Interpreting Emergent Planning in Model-Free RL' (ICLR 2025) (https://arxiv.org/abs/2504.01871): The same method: linear probes for concepts, then adding the probe-derived vectors to activations to change behaviour over whole episodes. The agent is a discrete Sokoban planner with no physical yardstick.
- Taufeeque et al. 2024, 'Planning in a recurrent neural network that plays Sokoban' (https://arxiv.org/abs/2407.15421): Probes predict future actions, and steering the hidden state along probe directions changes behaviour. The setting is discrete.
- Turner/Mini et al. 2023, 'Understanding and Controlling a Maze-Solving Policy Network' (https://arxiv.org/abs/2310.08043): Activation-addition steering in an RL policy ('cheese vector'). Its 'top-right vector' exposes a memorised spatial prior from the training distribution, a conceptual cousin of the memorisation probe. The actions are discrete.
- Haon, Stocking, Chuang, Tomlin 2025, 'Mechanistic Interpretability for Steering VLA Models' (CoRL 2025) (https://arxiv.org/abs/2509.00328): Finds 'speed' directions in robot VLA policies and injects them to make motion faster or slower, on a real UR5. This is the closest continuous-motion steering result. It uses a fixed coefficient on the real robot, gives no metric dose-response against an optimal reference, and is not racing.
- Tucker et al. 2022, 'Probe-Based Interventions for Modifying Agent Behavior' (https://arxiv.org/abs/2201.12938): Probe-based edits to a driving agent's representations to change its behaviour (collision avoidance at a simulated junction). It is not racing and has no dose-response.
- Jeon, Kim, Vinitsky, Kim 2026, 'What Probing Reveals about Autonomous Driving' (arXiv 2606.31106, NeurIPS 2026) (https://arxiv.org/abs/2606.31106): Linear probes and targeted perturbations on imitation-learning and RL driving policies, focused on predicting surrounding vehicles. It is not racing and does no metric steering of a control event.
- Raileanu & Fergus 2021, 'Decoupling Value and Policy for Generalization in RL' (https://arxiv.org/abs/2102.10330): Shows that shared representations memorise level-specific information (episode length) on Procgen. This is the precedent for a memorisation diagnostic, though it does not decode absolute position.

### C6. Audit: the public F1 position feed carries no driver lateral-line information  (partly-done, high confidence)

**Defensible claim:** The FastF1 maintainer and the OpenF1 docs already say the public F1 position feed is a 'normalized track position' without lateral detail. As far as we know, we give the first quantitative audit of this across N circuits and 2 seasons. Cross-driver lateral scatter about a reference lap matches within-driver scatter (median ≈5 cm). Positive controls with known lateral offsets, such as staggered grid slots and side-by-side cars, also collapse onto one line. So 'expert racelines' rebuilt from this feed mostly reflect the timing system's track model, not the drivers.

**Closest prior work:**
- FastF1 GitHub Discussion #116 'X and Y coordinates' (Mar 2022); maintainer theOehrly (https://github.com/theOehrly/Fast-F1/discussions/116): The core finding is already stated in public. The maintainer calls X/Y a 'normalized track position' that approximately follows the ideal line, and says it is 'certainly not possible to analyze drivers lines through a corner'. There are no numbers, circuits or seasons in the thread.
- OpenF1 API docs (location endpoint) and the openf1r R wrapper manual (https://coolbutuseless.r-universe.dev/openf1r/doc/manual.html): Documented caveat: location is approximate and 'lacks lateral position', i.e. it does not say which side of the track a car is on. Qualitative only.
- f1technical forum, 2020 FastF1/telemetry thread (https://f1technical.net/forum/viewtopic.php?p=913339): A community member writes that you cannot get the on-track position or line choice in a corner from this data. Folk knowledge, not an audit.
- Shehadeh et al., 'Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization' (ITSC 2026, arXiv 2603.07126) (https://arxiv.org/html/2603.07126v2): The work the audit would critique. It treats FastF1 X/Y as 'noisy GPS telemetry' carrying 'near-optimal driving behavior' and never discusses snapping. Its RMSE to the 'expert' line is about 3 m, and the centreline and min-curvature lines do not differ significantly from it. That pattern fits a track-model line rather than driver lines, but the paper does not test it.

### O1. Nemesis: the circuit the AI fears most  (partly-done, high confidence)

**Defensible claim:** Adversarially searching track space to break a frozen driving policy goes back to Uesato et al. (2019), who looked for TORCS crashes. Nemesis changes the target: it finds F1-plausible circuits that maximise a policy's lap-time regret against an optimum certified by optimal control on the same car model. It shrinks each one to the shortest track segment that still costs the time, and iterates the search with fine-tuning to show which geometric motifs a fixed-preview policy keeps failing on.

**Closest prior work:**
- Uesato et al., 'Rigorous Agent Evaluation: An Adversarial Approach to Uncover Catastrophic Failures' (ICLR 2019) (https://arxiv.org/abs/1812.01647): Very close on the mechanism, and missing from the plan's list. It runs an adversarial search over procedurally generated TORCS tracks, each a 12-d vector of waypoint locations and curvatures, to find failures of a frozen RL driving agent. It also trains on a harder, sharper-turn track distribution to cut failures about 20-fold. Its objective is crash probability, not lap-time regret against a certified optimum.
- Everett, 'Strategically Training and Evaluating Agents in Procedurally Generated Environments' (Oxford DPhil, 2020) (https://www.robots.ox.ac.uk/~sjrob/Theses/richardEverett_2020.pdf): Worst-case analysis of RL agents through adversarial procedurally generated environment settings, including a particle-racing task, followed by adversarial training. Also failure-based, not regret against an optimum.
- Gisslen et al., ARLPCG (arXiv 2103.04847, 2021) (https://arxiv.org/abs/2103.04847): An RL generator builds racing tracks against an RL solver: an arms race, but used as a training curriculum, with no physics optimum.
- PAIRED / ACCEL / POET / Environment as Policy (https://arxiv.org/abs/2203.01302): Regret- or difficulty-driven environment generation and agent-environment arms races (POET arXiv 1901.01753; Environment as Policy arXiv 2410.22308 edits racing tracks). These are curricula with approximate regret, not certified test-time artefacts.
- Togelius, De Nardi & Lucas 2006/2007; Loiacono, Cardamone & Lanzi 2011 (evolved racing tracks) (https://www.bibsonomy.org/bibtex/11932f4657fc0a1fda3c9f021f40f2d72): Tracks evolved against a specific driver model or controller's behaviour, but for entertainment or diversity, not regret.
- Falsification of Autonomous Systems in Rich Environments (2024) (https://arxiv.org/abs/2412.17992): Black-box search over obstacle placements on a sinuous track to make a learned car controller crash.

### O2. Branching Ghosts: take-the-wheel coach  (partly-done, medium confidence)

**Defensible claim:** "Branching Ghosts is an open, in-browser racing coach. Press a key and a learned driving policy takes over from your car's exact state, so you see what you would have gained from that point on. Each corner gets a receipt: entry, apex and exit losses that add up exactly to the corner's total, because the advantages telescope as in the performance-difference lemma, plus a line-versus-execution split from a fixed-path solve on the same physics. Commercial coaches compare you to reference laps (Trophi), replay an RL driver's inputs through your hardware (MOZA), or cost mistakes with undisclosed simulations (Track Titan). We know of no open tool that forks a feedback policy from the driver's own state and accounts for the time exactly."

**Closest prior work:**
- Track Titan 'Coaching Flows' / Timekiller (Nov 2025) (https://www.tracktitan.io/post/november-2025-update-coaching-flows): This is the closest product. A 'proprietary simulation engine' runs 'thousands of different scenarios' to cost each mistake, including the downstream mistakes it causes, and ranks root causes per corner complex. The method is undisclosed. Nothing says it forks a learned policy from your state, splits line from execution loss, or guarantees that the parts sum to the total.
- MOZA Racing Lab AI Coach (GDC/Gamescom 2026) (https://mozaracing.com/blogs/news/moza-ai-coach-motion-gdc-2026): An in-house RL-trained AI driver. The software drives the user's wheel and pedals so they feel the AI's inputs, and compares the user's telemetry with AI reference data. It is a demonstration and reference-lap comparison, not a counterfactual continuation from the human's exact state.
- Trophi.ai (Driver61 / 'Mansell AI') (https://www.trophi.ai/): Compares your laps against an expert reference lap corner by corner, with a live voice coach and post-session reports ranked by time impact. These are reference-lap deltas, not counterfactuals.
- Shen, ..., Subosits, Ersal 2025, 'Cyber Racing Coach: haptic shared control for teaching advanced driving skills' (with TRI authors) (https://arxiv.org/abs/2509.20653): An autonomous controller shares steering with the human, and its help fades as the driver improves. This is real-time human-AI control sharing for coaching, but no per-corner counterfactual time-loss accounting.
- Gopinath, ..., DeCastro, Balachandran, Rosman 2026 (TRI), adaptive computational teaching in driving (https://arxiv.org/abs/2606.25224): A learned teaching model for high-performance driving, validated on a race-coaching simulator dataset. It models the teacher's interventions, not policy continuations.
- Kakade & Langford 2002 performance-difference lemma; Ross et al. 2011 DAgger (https://arxiv.org/abs/1011.0686): The mathematics is old. Querying an expert policy at the learner's own visited states is DAgger. That the phase terms telescope is the performance-difference identity, applied with the AI as the reference policy and the human as the behaviour policy. Credit these; the novelty is in the application.

### O3. The Price of Tarmac: a physics atlas of track limits vs FIA lap deletions  (partly-done, medium confidence)

**Defensible claim:** We produce the first corner-by-corner physics price list for track limits on real F1 circuits: lap time gained per metre of extra exit width, from minimum-time OCP multipliers and checked with re-solves. We then test whether this incentive predicts where race control deletes laps, after accounting for laps run and runoff type. Earlier work counted deletions or compared deleted and clean laps at single events; none priced the incentive from physics.

**Closest prior work:**
- RaceFans (Jan 2024), 'Who are F1's biggest track limits offenders and where are the top trouble spots?' (https://www.racefans.net/2024/01/19/who-are-f1s-biggest-track-limits-offenders-and-where-are-the-top-trouble-spots/): Counts 695 track-limits infringements across 2023 by venue and corner (Red Bull Ring T10 89, T9 60; Losail T5 36). Raw counts only: no exposure normalisation, no time-gain estimate, no physics.
- Motor Sport Magazine / Project F1 (E. Gupta, Jul 2023), 'The time gained by exceeding F1 track limits: is it worth the risk?' (https://motorsportmagazine.com/articles/single-seaters/f1/the-time-gained-by-exceeding-f1-track-limits-is-it-worth-the-risk): Empirical deleted vs clean laps at Austria 2023 (e.g., Albon's deleted lap about 0.5 s quicker, mostly through T9/T10). One circuit, observational, no model.
- F1.com explainer on 2024 Red Bull Ring changes, plus The Race / Mark Hughes coverage (https://www.formula1.com/en/latest/article/explained-how-key-changes-to-the-red-bull-ring-look-to-avoid-last-years.2ubjMYkbT3HHpwuEJ7PoHw): Engineers found that running onto the asphalt verge at the T9/T10 exits gains lap time, and the FIA added 2.5 m gravel strips. 'Where the gravel should go' is already decided case by case from experience, not from a model.
- Autosport, FIA view on track limits 'based on gut feeling and driver feedback' (https://autosport.com/all/news/fia-to-police-f1-track-limits-in-five-corners-at-qatar/6794726/): Supports the framing that the regulator has no quantitative price per corner and chooses monitored corners from experience.
- Gulisano, Masoni, Gabiccini, Guiggiani 2025, 'Disturbance-aware minimum-time planning... with probabilistic safety certificates' (arXiv 2506.13622) (https://arxiv.org/abs/2506.13622): Tightens track-limit constraints in a minimum-time OCP and reports the lap-time penalty, i.e. the cost of having less width, on one Barcelona sector. Same mathematics, opposite direction, with no per-corner atlas and no enforcement data.
- Massaro & Limebeer 2021 review; Perantoni & Limebeer 2014 'Optimal control for a Formula One car with variable parameters' (https://pure.uj.ac.za/en/publications/optimal-control-for-a-formula-one-car-with-variable-parameters/): Standard OCP machinery where multipliers and costates act as sensitivities. Applied to setup parameters, not to track-edge width per corner.

### O4. The Constructor's Circuits: tracks inverse-designed for each team's car  (partly-done, medium confidence)

**Defensible claim:** Track generation for balance between vehicles has been explored in games (Majid & Crawfis 2024), and evolutionary F1-style track generation with a lap simulator exists (Harper 2024). We add three things:
- gradient-based inverse design of closed circuits through a differentiable lap-time model;
- objectives defined on posterior distributions of real teams' cars calibrated from public timing data, such as the probability that a given team is fastest, or the grid's lap-time spread;
- verification of every finalist with a full optimal-control solve for all cars.
All designs use a fixed setup and flat tracks.

**Closest prior work:**
- Majid & Crawfis, 'A Framework to Even-Out Racetrack Bias' (FDG 2024) (https://doi.org/10.1145/3649921.3659849): Direct precedent for the 'Equaliser'. It assembles procedurally generated track segments, using AI-controller simulations of differently specced vehicles, so each vehicle 'operated optimally' gets a similar time. Game vehicles, not calibrated real cars, no gradients and no OCP.
- Harper, 'Evolutionary Design Optimization for a Formula One Car and Track' (UGA MS thesis, 2024) (https://www.ai.uga.edu/sites/default/files/inline-files/theses/harper_daniel_202405_ms.pdf): Evolutionary generation of F1-style tracks (B-splines and OpenLAP segments) with a lap-time simulator in the loop. It explicitly proposes tracks that expose a car's weaknesses as future work. Its fitness is subjective, nothing is calibrated per team, and there are no gradients or posterior.
- Togelius, De Nardi & Lucas 2007 (personalised racing-track evolution); Loiacono et al. 2011 (https://arxiv.org/abs/2311.04710): Search-based track generation optimised against a specific driver or controller model (the search-based PCG survey covers this line).
- Perantoni & Limebeer 2014, 'Optimal control for a Formula One car with variable parameters' (https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f): Optimises setup and driven line jointly by NLP with the track fixed. This is the inverse of designing the track for a fixed car.
- F1 'Tech Tuesday' circuit sensitivity pieces (https://www.formula1.com/en/latest/article.tech-tuesday-why-austria-and-hungary-are-perfect-proving-grounds-for-merc-ferrari-rb-2020.30PZs031OWz7TtPaRgU1Lq.html): Qualitative or numeric per-circuit power and downforce sensitivities, and which tracks suit which car. Analysis of existing tracks, not design.

### O5. Racing-folklore phase diagrams and corner extinction  (partly-done, medium confidence)

**Defensible claim:** "Racing folklore (late apex, sacrifice the first corner, trail braking, 'it's flat in a fast car') becomes phase diagrams. We compute regime boundaries over corner geometry and car parameters (downforce, grip, power) with a minimum-time optimal-control solver. On the same axes we overlay the boundary our car-conditioned RL policy actually uses, including where it extrapolates beyond its training cars. We then check the flat-versus-lift boundary against public F1 qualifying telemetry. Single-corner optimal-control studies exist (Velenis et al.; Tavernini et al. 2013). We have not found these rules mapped as regime boundaries with a learned agent's boundary and a real-data check."

**Closest prior work:**
- Velenis, Tsiotras & Lu 2008, 'Trail-Braking Driver Input Parameterization for General Corner Geometry' (SAE); Velenis & Tsiotras ISIE 2005 (minimum time vs maximum exit velocity) (https://saemobilus.sae.org/papers/trail-braking-driver-input-parameterization-general-corner-geometry-2008-01-2986): Minimum-time OCP studies across corner geometries test whether trail braking and late apexes are optimal, and relate driving techniques to cost functions. The setting is rally and low-friction, with no downforce. They give no regime maps over car parameters and no learned-agent comparison.
- Tavernini, Massaro, Velenis, Katzourakis, Lot 2013, 'Minimum time cornering: the effect of road surface and car transmission layout' (VSD) (https://www.research.unipd.it/handle/11577/2835923): A parametric OCP study of how the optimal cornering technique changes with mu (surface) and drivetrain layout. It is single corner, with no aero, and gives no phase-boundary maps.
- Perantoni & Limebeer 2014, 'Optimal control for a Formula One car with variable parameters' (VSD) (https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f): A full-lap F1 minimum-time OCP that varies and optimises setup parameters on one circuit. It is a sensitivity study, not geometry-by-parameter regime maps.
- Canopy Simulations (Michelin), 'Showing our sensitive side': Secondary Lap-time Sensitivities (https://simulation.michelin.com/canopy/technical-articles/showing-our-sensitive-side): A commercial tool that locates where lap-time sensitivity to downforce, drag, power and grip comes from along a lap (for example, zero power sensitivity mid-corner). It is closed and gives no folklore phase diagrams.
- Coaching literature on late apex and compromise corners (e.g. Paradigm Shift Racing, 'Rules of the Racing Line') (https://www.paradigmshiftracing.com/racing-basics/the-rules-of-the-racing-line): States the heuristics qualitatively: apex timing depends on acceleration versus grip, and you sacrifice corner 1 for corner 2. It gives no quantitative boundaries.

### O6. The Skill Atlas: where physics says drivers matter  (partly-done, medium confidence)

**Defensible claim:** Fans and statisticians have ranked circuits by how far teammates finish apart. As far as we know, this is the first test of whether a physics model with no driver data can predict that ranking. We compute each circuit's sensitivity to driver imprecision from an optimal-control and RL model and pre-register its rank correlation with 1996–2025 teammate qualifying gaps. We also map, segment by segment, how much drivers differ once the car is removed through teammates, and which physical features explain it.

**Closest prior work:**
- 'Hardest circuits as seen in Teammate gaps' (Reddit analysis reposted on Clip The Apex, Aug 2019) (https://cliptheapex.com/threads/difficulty-rating-of-the-circuits.8475/): Median teammate qualifying gap per circuit (Ergast data, 1994 on), normalised by the season average and clustered into 5 groups, on the idea that harder tracks spread teammates further. It is the observational version of S_drv, with no physics and no uncertainty, and is noisy for circuits with few seasons.
- Bell, Smith, Sabel, Jones 2016, 'Formula for success' (JQAS 12(2):99-112) (https://eprints.whiterose.ac.uk/id/eprint/96995/): A multilevel model whose driver and team variance functions vary by track type and weather. Team dominance is reduced on street tracks, which is the heteroscedastic 'where drivers matter' idea at a coarse level, from race points, with no physics features.
- van Kesteren & Bergkamp 2023, Bayesian F1 driver vs constructor (arXiv 2203.08489) (https://arxiv.org/pdf/2203.08489): Constructor advantage varies by circuit type, e.g. high-downforce cars suit street circuits. This is a coarse version of S_car, from results, not physics.
- Gulisano et al. 2025 disturbance-aware minimum-time planning (arXiv 2506.13622); FISITA 2016 F2016-VDCA-019 on time sensitivity to driver limitations (https://arxiv.org/abs/2506.13622): These compute the lap-time cost of disturbances or driver limits inside an OCP, close to the 'imprecision tax' feature. Neither links it to real driver spread.
- F1pace / Formula Dream teammate-gap normalisation; F1MATHS (f1technical) per-corner teammate telemetry (https://f1pace.com/p/2025-f1-season-qualifying-delta-between-teammates-rounds-1-21/): Media and hobby teammate-gap metrics and per-corner case studies. Descriptive; no variance model and no physics predictor.

### O7. The racing line as a physical system: Green's function, normal modes, bifurcations  (partly-done, medium confidence)

**Defensible claim:** Extending Aggarwal & Gerdes' (2024) lateral-compliance analysis of the racing line, we compute the full linear-response kernels of the optimal speed and line to local track curvature, and the Hessian eigenmodes of minimum lap time, on real F1 layouts. We then follow the softest mode through a change in line topology (single to double apex) as a corner is continuously morphed. Results come from a smoothed point-mass model, spot-checked against optimal-control finite differences on the bicycle model, and are shown piecewise where track-limit constraints switch.

**Closest prior work:**
- Aggarwal & Gerdes, 'Computing the Lateral Compliance of the Racing Line Using Trajectory Optimization' (Springer chapter / JSAE 2024) (https://link.springer.com/chapter/10.1007/978-3-031-70392-8_9): The closest precedent, and essentially the 'stiffness map' half of this idea. It computes how lap time responds to lateral perturbations of the minimum-time line on Laguna Seca and finds a wide near-optimal basin. Local and first or second order along the line; no curvature-input kernel, no eigenmode spectrum, no bifurcation study, no F1 circuits.
- Kegelman, Harbott & Gerdes 2017 (Vehicle System Dynamics) (https://ddl.stanford.edu/publications/journal/insights-vehicle-trajectories-handling-limits-analysing-open-data-race-car): Empirical 'anchor points': low path dispersion at high-curvature locations in pro-driver data. This is the test target the plan proposes, not the spectral analysis.
- Perantoni & Limebeer 2014; Massaro & Limebeer review (2021) (https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f): Lap-time sensitivity to vehicle and setup parameters through the NLP. Parameter sensitivities, not the response to track curvature as a function of distance.
- Verschueren et al. 2016 (exact-Hessian SQP for time-optimal race car control) (https://publications.syscop.de/Verschueren2016b.pdf): Uses second derivatives inside the solver, but never analyses them as a physical object (spectrum, modes).
- Disturbance-aware minimum-time planning (arXiv 2506.13622, 2607.13312) (https://arxiv.org/abs/2506.13622): Treats fragility of the min-time line and puts margins where needed. Related to 'where precision is paid for', but robust planning, not response kernels.

### O8. The grid as an electrical circuit: why some driver comparisons are impossible  (partly-done, high confidence)

**Defensible claim:** "We bring a known result from ranking theory and labour economics into F1 (Hendrickx et al. 2019; Jochmans & Weidner 2019). How precisely you can compare two drivers from different teams is roughly the electrical resistance between them in the teammate network. We check this approximation against a full Bayesian posterior. We use it to show which comparisons are effectively impossible from 2023-25 data, and to rank real and hypothetical driver moves by how much comparison information each adds."

**Closest prior work:**
- Hendrickx, Olshevsky & Saligrama 2019, 'Graph Resistance and Learning from Pairwise Comparisons' (ICML 2019) (https://arxiv.org/abs/1902.00141): The core theory: estimation error in pairwise-comparison ranking scales with the square root of graph resistance. The idea that comparison uncertainty is effective resistance is established.
- Jochmans & Weidner 2019, 'Fixed-Effect Regressions on Network Data' (Econometrica); Kline, Saggio & Solvsten 2020 (KSS leave-out) (https://www.econometricsociety.org/publications/econometrica/2019/09/01/fixed-effect-regressions-network-data): Ties the accuracy of two-way fixed-effects (AKM) estimates to the Laplacian and connectivity of the mover network. Limited-mobility bias and its corrections are standard in labour economics.
- Franceschet & Bozzo 2017, 'Massey's method for sport rating: a network science perspective'; Osting, Brune & Osher 2014, 'Optimal data collection for informative rankings' (https://arxiv.org/abs/1701.03363): Massey sports ratings are Laplacian solves; related work reads ratings as potentials on a resistor network. Osting et al. design comparison graphs (schedules) through Laplacian criteria (arXiv 1207.6430). So 'which pairing to add next' is a known design problem in general ranking.
- AWS / F1 'Fastest Driver' (2020) (https://www.formula1.com/en/latest/article/hamilton-schumacher-senna-machine-learning-reveals-the-fastest-f1-driver-of.3DwwPLW4glCmlunjciH1Cz): Links drivers across eras through teammate qualifying comparisons in one network. It publishes point gaps, with no uncertainty and no network analysis.
- racemate.io F1 Driver Network; sijokun/f1-graph; f1-analysis.com teammate model (https://racemate.io/network/): Interactive teammate graphs (connected components, paths between drivers) and teammate-chain rankings. None attach uncertainty to graph structure.
- van Kesteren & Bergkamp 2023; Lindner et al. 2026 (https://arxiv.org/abs/2203.08489): Bayesian F1 driver and constructor models with posterior uncertainty. Neither analyses how teammate-network topology drives that uncertainty.

### O9. Racing brain lab: place cells, lesions and a live knob box  (partly-done, medium confidence)

**Defensible claim:** Place-like units have been seen in navigation RL agents, and activation steering works in maze and robot policies. We bring both into high-speed racing. We test whether memorising one track gives a policy place-like units that remap on new tracks, while procedural training gives track-invariant braking-distance units. We then let viewers push those directions live, with each change measured in metres of braking point and in lap time against the optimal-control lap.

**Closest prior work:**
- Kappel & Cheng, 'Global remapping emerges as the mechanism for renewal of context-dependent behavior in a reinforcement learning model' (Front. Comput. Neurosci. 2024/25) (https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1462110/full): Place-cell-like units arise spontaneously in a DQN agent's hidden layers and globally remap between contexts, with population-vector analysis. Navigation T-maze, discrete actions, with no specialist-vs-generalist contrast and no lesions.
- Banino et al. 2018 (Nature), grid-like units in navigation agents; eLife 'Learning place cells and remapping by decoding the cognitive map' (https://elifesciences.org/articles/99302.pdf): Grid and place-like units, including remapping across trained environments, in navigation networks. Establishes the neuroscience-style analysis; none of it is in racing or continuous control.
- Mini, Grietzer, Sharma, Meek, MacDiarmid, Turner 2023, 'Understanding and Controlling a Maze-Solving Policy Network' (arXiv 2310.08043) (https://arxiv.org/pdf/2310.08043): Activation-steering ('cheese vector') in an RL policy changes behaviour without retraining. Discrete gridworld; the vectors transfer poorly.
- Häon, Stocking, Chuang, Tomlin, 'Mechanistic Interpretability for Steering Vision-Language-Action Models' (CoRL 2025, arXiv 2509.00328); 'Closed-Loop Neural Activation Control in VLAs' (arXiv 2606.00269) (https://arxiv.org/abs/2509.00328): Finds interpretable speed and direction directions in robot-policy activations and steers continuous motion at inference time. The closest continuous-control steering result, in large VLAs rather than a small racing MLP, with no physical-optimum readout.
- Taufeeque et al. 2024 (arXiv 2407.15421); Bush et al. ICLR 2025 (arXiv 2504.01871); Jeon et al. 2026 'What Probing Reveals about Autonomous Driving' (arXiv 2606.31106) (https://arxiv.org/abs/2504.01871): Probes plus causal interventions in model-free RL agents (Sokoban) and probing of driving policies (traffic prediction). The methods are established; nothing here covers racing or braking points.
- Neuronpedia-style LLM steering demos; Yosh (Trackmania) videos showing live network activations (https://www.youtube.com/watch?v=Dw3BZ6O_8LY): Public interactive steering exists for LLMs. Live activation visuals synced to a car exist on YouTube as eye candy, with no interventions and no quantitative readout.

### O10. Learning spacetime: when an AI learns to trail-brake  (partly-done, medium confidence)

**Defensible claim:** We apply the AlphaZero what-when-where methodology (McGrath et al. 2022) to a continuous-control racing policy. Each racing skill is defined against optimal-control targets on a track the agent never trained on, timed across training checkpoints in both behaviour and internal probes, and compared between curricula. To our knowledge this has not been done for driving skills.

**Closest prior work:**
- McGrath et al., 'Acquisition of chess knowledge in AlphaZero' (PNAS 2022) (https://www.pnas.org/doi/10.1073/pnas.2206625119): The method being borrowed: linear probes for human concepts across training checkpoints and layers (what, when, where), paired with behavioural change. Chess, not continuous control.
- Lovering et al., 'Evaluation Beyond Task Performance: Analyzing Concepts in AlphaZero in Hex' (NeurIPS 2022) (https://arxiv.org/abs/2211.14673): Probing plus behavioural tests of concepts over training and layers, in a board game.
- Bush et al., 'Interpreting Emergent Planning in Model-Free RL' (ICLR 2025) (https://arxiv.org/abs/2504.01871): Probes and interventions at training checkpoints of a model-free agent (Sokoban). Methodological precedent for probing over training with causal checks.
- Jeon et al., 'What Probing Reveals about Autonomous Driving' (arXiv 2606.31106, 2026) (https://arxiv.org/abs/2606.31106): Linear probing of RL and behaviour-cloning driving policies, including how prediction and planning signals emerge or plateau with scale and longer training. Urban driving, not racing skills scored against optimal control.
- FastRLAP (2023); Yosh / Linesight training videos; arXiv 2604.09499 (https://arxiv.org/abs/2304.09831): Anecdotes of emergent late braking and racing lines. Ghost-swarm visuals of lines converging over training. One paper tracks deviation from a minimum-curvature line during training. None time named racing skills against OCP targets or compare curricula.

### O11. Circuit fingerprints and the Lost Grands Prix  (partly-done, medium confidence)

**Defensible claim:** "Teams and F1 journalists already talk about circuit 'power sensitivity' and 'downforce sensitivity', and commercial tools map these along a lap (F1 Tech Tuesday; Canopy). We release them openly as per-metre lap-time sensitivity fingerprints for real circuits, computed from a calibrated optimal-control model. We then test them out of sample: leaving each circuit out in turn, do the physics fingerprints predict how the 2023-25 team order reshuffles from track to track better than plain geometry statistics or a flexible statistical model? Only after passing that backtest do we publish calibrated, reliability-checked predictions for Grands Prix this era never held."

**Closest prior work:**
- F1.com 'Tech Tuesday' (Mark Hughes & Giorgio Piola), per-circuit power and downforce sensitivity rankings (e.g. 2019-2020) (https://www.formula1.com/en/latest/article/tech-tuesday-why-austria-and-hungary-are-perfect-proving-grounds-for-merc-ferrari-rb-2020.30PZs031OWz7TtPaRgU1Lq): Publishes per-circuit lap-time sensitivity to +10 bhp and to downforce points (e.g. Red Bull Ring ~0.19 s per 10 bhp vs Hungaroring ~0.12 s). It uses them qualitatively to explain shifts in competitive order. This is probably the source the dossier called the 'Motor Sport 2019 index'. It is journalistic, the data are not open, and it is not validated.
- Canopy Simulations (Michelin), Secondary Lap-time Sensitivities (https://simulation.michelin.com/canopy/technical-articles/showing-our-sensitive-side): Commercial OCP-based sensitivity maps along the lap for downforce, drag, power and grip, showing where on the lap the gains come from. This is the per-metre fingerprint idea, but closed, shown for single circuits, and not tied to results.
- Choi 2025, 'Examining the Correlation Between Circuit Demands and Performance Gaps between F1 Constructors' (non-peer-reviewed preprint) (https://research-archive.org/index.php/rars/preprint/view/2174): Classifies tracks (high-speed, technical, balanced) and relates them to constructor gaps. It has no physics fingerprints, no stated model and no out-of-sample test.
- Perantoni & Limebeer 2014; TUMFTM laptime-simulation (https://github.com/TUMFTM/laptime-simulation): OCP and QSS lap-time tools that compute parameter sensitivities (e.g. mass, about 0.036 s/kg) on chosen tracks. They are not systematised into an open cross-circuit dataset or used to predict team order.
- Lindner et al. 2026 Bayesian state-space driver/constructor model (https://arxiv.org/abs/2608.04629): Models constructor ability over time from qualifying data. It has no circuit-specific interaction terms driven by physics.
