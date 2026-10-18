> **How this plan was made.** Four web researchers (prior art, F1 data, simulation, RL architectures) fed three independent designs: learning-first, showcase-first and research-rigor. A judge merged them, keeping the research-rigor design as the backbone. A fact-checker then verified about 60 concrete claims, and 14 corrections are applied below: track widths and hairpin radii, Shanghai's calendar, the IMS and Zandvoort files, the Fuchs reward detail, action rates, mass attribution, parameter and observation counts, and compute estimates. As of 2026-10-18 the F1 live-timing archive that FastF1 depends on returns HTTP 401, and OpenF1 still responds.

# ApexZero: final plan

## 1. Pitch and title options

You build the whole stack:
- a JAX simulator of an F1-style car;
- a CasADi minimum-lap-time optimal-control solver (OCP) that uses the *same* equations;
- PPO written from scratch;
- a curriculum over procedurally generated circuits.

One policy trains without ever seeing a real track. It then drives 12 real F1 layouts zero-shot. Each lap is scored as a percentage gap to the physics optimum, split into line loss and execution loss.

The same physics is then calibrated per team from public telemetry. It is used to estimate where real drivers lose time relative to their own car. Results carry uncertainty, and the project makes no claim the data cannot support.

- **Repo / project:** ApexZero
- **Paper:** *How Far From the Limit? Zero-Shot Racing on Real F1 Circuits from Procedural Training, Measured Against Optimal Control*
- **Video:** *This AI Has Never Seen Suzuka* / *I Trained an F1 AI on 10,000 Tracks That Don't Exist*

## 2. Central questions

| # | Question | Instrument |
|---|---|---|
| Q1 Generalisation | Trained only on procedural tracks, how close does one policy get zero-shot to the OCP lap on 12 held-out F1 layouts? How does that change with the number of training tracks? | Gap G = T_RL/T_OCP − 1; scaling curve over N ∈ {1, 10, …, 10⁴, ∞} training tracks |
| Q2 Curriculum (novel) | Does choosing training tracks by *computed* regret beat SFL, PLR⊥, ACCEL and plain domain randomisation (DR) at equal compute? | 5-arm ablation with rliable statistics |
| Q3 Anatomy | Does the remaining gap come from line choice or from execution, and in which corner types? Does the network encode braking points, and can we steer them? | Fixed-path OCP split; probes plus interventions; lookahead distance vs v²/2a |
| Q4 Drivers | With the car factored out, where do drivers lose time by corner type and phase? Is the ranking reliable? | Teammate-identified hierarchical Bayesian model, injection-recovery, split-half gate |

**Pre-registered hypotheses.** Commit these before unsealing the test set. They are targets, not predictions.

- **H1.** The DR generalist completes ≥90% of laps on R-test with median G ≤ 4%.
- **H2.** The best curriculum lowers the IQM of G by ≥25% relative to DR.
- **H3.** Oracle-regret PLR beats PLR scored by positive value loss.
- **H4.** G falls monotonically with N and plateaus by N ≈ 10³–10⁴.
- **H5.** Driver effects reach split-half Spearman ρ ≥ 0.5. If not, only teammate-pair results are published.

## 3. System architecture

```
TUMFTM (25 csv) ─┐                                      OpenF1 / jolpica ─► Parquet (P0)
bacinger GeoJSON ┼► trackkit: periodic spline, 2 m ──┐        │
κ-grammar (JAX) ─┘   κ, wL, wR, x,y,θ  f32[B,4096]    │  snapping audit ─► ICP ─► RTS s(t)
                                                      │        ▼
  vehicle/dynamics.py  f(x,u,κ,p; ops)  ◄── ONE ODE   │  lap filter ─► additive segments
    ├ ops=jnp ────► env (jit/vmap/scan) ───────────┐  │        ▼
    └ ops=casadi ─► ocp: min-time, fixed-path,     │  │  calibration θ[team,event] on pooled
                    20 Hz tracker, duals           │  │  teammate envelope (QSS inner loop)
  oracle (JAX): FISTA min-curv QP + QSS fwd-bwd ───┼──┘        ▼
                                                   ▼     refs: R1 envelope / R2 OCP(θ) / R3 RL(θ)
  ued/: DR | PLR⊥ | SFL | ACCEL | OraclePLR ◄─ scores ─ agents/ppo.py (from scratch)
                                                   │           ▼
               eval/: lockbox, G, line/exec, probes, robustness ──► NumPyro M1 / M2
                                                   ▼           ▼
       viz/: Rerun · matplotlib · JSON → Three.js (agent, OCP, driver timing ghost) · Manim
```

**Invariants**
1. `f()` is written once against an `ops` shim, with smooth saturations. A test checks JAX against CasADi on 10⁴ random states and requires a relative difference below 1e-5.
2. Progress `s` is a state variable, never a projection onto the centreline.
3. **Lockbox:** R-test never touches tuning, generator fitting or checkpoint choice.
4. **Claim ledger:** every published number comes from a script tied to a figure and its caveat.

## 4. Key technical decisions

### Simulator

Write your own: a dynamic single-track (bicycle) model in JAX, in Frenet coordinates. Commercial sims are real-time and Windows-bound, f1tenth is 1/10 scale, and CarRacing uses toy physics.

- **State:** [s, n, ξ, vx, vy, r, δ, ux].
- **Tyres:** Pacejka with E = 0, plus a friction ellipse implemented through a smooth min.
- **Loads and forces:** load transfer (ax lagged one substep), per-tyre load sensitivity, ClA·v² downforce split front/rear, CdA·v² drag, a P/v power limit, and a kinematic blend below 4 m/s.
- **Integration:** RK2 at 5 ms, with 10 substeps per 50 ms control step.
- **Gates:** lap time within 0.05% of RK4 at 1 ms, plus a coast-down energy test.
- **Why:** a few thousand FLOPs per 50 ms control step, against about 1 MFLOP for one network forward pass, so substeps cost almost nothing and 10⁹ steps are cheap.

### Car

Baseline is the 2022–25 car, and the policy is conditioned on car parameters.

**TUMFTM-derived values** (except mass: 800 kg is the FIA 2025 minimum; TUMFTM's own file uses 733 kg):

| Parameter | Value |
|---|---|
| m | 800 kg |
| lf / lr | 1.968 / 1.632 m |
| h_cog | 0.335 m |
| CdA | 1.56 m² (DRS ×0.83) |
| ClA front / rear | 2.20 / 2.68 m² |
| ρ | 1.18 kg/m³ |
| P_max | 695 kW |
| μx/μy | front 1.65/1.85, rear 1.95/2.15 at Fz0 = 3 kN |
| kμ | −5e-5 /N |

**Assumed values (each gets a ±30% sensitivity run):**

| Parameter | Value |
|---|---|
| Iz | 1,100 kg·m² |
| Pacejka C, B | 1.6, ≈12 (peak slip about 7°) |
| δmax | 0.45 rad (kinematic radius ≈7.5 m, below the ~8 m tightest centreline radii at Shanghai T14 and Spielberg T3) |
| Steering lag τ | 0.05 s |
| Steering rate limit | 1.5 rad/s |
| Brake bias | 0.58 front |

**Context box (observed by the policy):** m 750–850 kg, P 550–800 kW, ClA 3.0–6.0, CdA 1.0–1.9, μ-scale 0.85–1.10, aero balance ±5%. Add a hidden ±2% μ jitter.

**Why context conditioning:** the policy stays optimal for each car. Blind randomisation made Czechmanowski's policy slower. Conditioning also lets the policy drive each calibrated team car later. Expect the uncalibrated car to pull about 4 g at 200 km/h and over 6 g at 300 km/h, against the quoted ~5 g peak. That is the first lesson, and calibration fixes it.

### Tracks

**Training set: open κ-strips.** Each strip is 8 km, padded to 4,096 samples at 2 m, and generated *on the device in JAX* from a κ-grammar.

- **Tokens:** straight, kink, sweeper, medium, hairpin (R 8–30 m), chicane and esses, joined by 10–60 m clothoids.
- **Token order:** a Markov chain fitted on R-dev only.
- **Widths:** fitted to the TUMFTM width distribution (roughly 7.5–15 m; Monza, Montreal, Spa and Suzuka have a median near 9 m), with low-frequency ±1.5 m variation and left/right asymmetry.
- **Filters:**
  - R_min ≥ 8 m;
  - w_half·κ ≤ 0.5;
  - at least one straight of 500 m or more per strip.
- **Why:** the simulator is local, so closure and self-intersection do not matter. The generator becomes jittable and DR costs nothing.

**Closed circuits** are used only for P-val and visuals. They are closed by Gauss-Newton and checked for self-clearance.

**F1-likeness report:** Wasserstein distances between generator and R-dev for corner radius, turning angle Δθ, straight length and corners per km.

### Observation (actor ~222-d, egocentric)

- **Preview (200):** 40 points at d_k = d_max·(k/40)^1.5, with d_max = clip(6 s·v, 150, 500) m. Per point: left and right edge (x, y) in the body frame divided by 100 m, and κ·50.
- **Ego (16):** vx, vy, r, sideslip β, n/w_half, ξ, δ, pedal state ux, ax, ay, previous action (2), front/rear friction-ellipse utilisation, αf, αr.
- **Car context (6).**
- 1% Gaussian observation noise during training.
- **Excluded:** absolute position, lap progress, track id, rays and any reference-line offset.
- **Critic (asymmetric):** the actor inputs plus κ, wL and wR every 10 m out to 1.5 km, and the true μ.
- **Why:** this matches GT Sophy's speed-scaled ~6 s horizon and contains nothing the policy can memorise.

### Action

- [a_steer, a_pedal] ∈ [−1, 1]² at 20 Hz, passed through the actuator lag and rate limit.
- Combined throttle/brake; no gears.
- The same lags and limits are hard constraints in the OCP.
- **Why:** the two-output layout follows GT Sophy and Fuchs et al. Both acted at 10 Hz; 20 Hz is this plan's own choice, because at 90 m/s the car covers 4.5 m per decision.

### Reward and episodes

```
r_t = Δs_t / 10 m − 0.02·‖a_t − a_{t−1}‖²;   on termination: −(5 + 5·(v/50)²)
```

**Termination:**
- any wheel leaves the corridor, |n| > w_side − 1.0 m (the same constraint as in the OCP);
- |ξ| > 90°;
- vx < 2 m/s for 2 s;
- NaN.

**Training settings:**
- No reference-line term, ever.
- γ = 0.995 at 20 Hz (10 s horizon); GAE λ = 0.95.
- Training windows: 60 s (1,200 steps) from a random s0, with n0 ~ U(±0.5·w_half), ξ0 ~ N(0, 0.03) and v0 ~ U(0.6, 1.0)·v_QSS(s0). Truncated windows bootstrap the value.

**Evaluation:** 2 laps from a rolling start, deterministic mean action, lap 2 timed, 20 jittered starts.

**Why:** Δs is exactly what the OCP integrates. The speed-scaled penalty adapts Fuchs et al. (who used SAC with a non-terminal wall-contact penalty c_w·‖v‖²): without it their cars never learned to brake and ground along walls, and fixed-value penalties were either ignored or made the car stand still. The conservative corridor protects against errors in satellite-measured widths.

### Algorithm

PPO written from scratch: about 400 lines, PureJaxRL-style, with the whole loop jitted.

- **Verification first:** it must match PureJaxRL and rejax on gymnax CartPole and Pendulum within 5-seed CIs before it touches the car.
- **Parallelism:** Mac 2,048 envs × 64 steps; GPU 16,384 envs × 128 steps.
- **Hyperparameters:** 4 epochs × 8 minibatches; Adam 3e-4 with linear decay; clip 0.2; value coefficient 0.5; entropy 1e-3 → 0; gradient norm 0.5; running observation normalisation; return-std reward scaling; log-σ initialised at −0.5.
- Work through the "37 implementation details" checklist.
- **Why:** with a cheap simulator, wall-clock time matters more than sample efficiency, and JaxUED and SFL are both built on PPO.

### Network

- **Default (SimBa-lite):** observation normalisation → Dense 256 → 2 × [LayerNorm → Dense 512 → ReLU → Dense 256 + skip] → LayerNorm → heads. About 0.58M actor parameters; the 384-wide critic is about 1.4M.
- **Baseline:** a plain 3×256 tanh MLP.
- **Ablation:** a 1-D conv encoder over the preview.
- No transformer over a fixed-length preview.

### Oracles

1. **Minimum-time OCP.**
   - CasADi + IPOPT, direct collocation in s with ds = 3 m, periodic boundary conditions, using the same `f()`.
   - Warm-started from min-curvature plus QSS, with μ-homotopy if it fails to converge.
   - Multi-start that includes the RL trajectory. If RL beats the OCP, re-solve from the RL trajectory and report it.
2. **Fixed-path OCP** on the agent's smoothed n(s). This gives the line/execution split on the same bicycle model.
3. **JAX QSS oracle.**
   - Min-curvature as a box-constrained least-squares problem on second differences, solved by fixed-iteration FISTA. Validate to ≤0.5 m RMS against `trajectory_planning_helpers`.
   - Then a forward-backward pass in `lax.scan` using an analytic point-mass g-g-v built from the context parameters.
   - Milliseconds per track and vmappable.
   - Calibrate its bias and Spearman ρ against the full OCP on 200 P-val circuits.
4. **OCP plus a 20 Hz tracker** (pure pursuit + speed PID). Gives T_OCP@20Hz and supports the robustness test.

## 5. Generalisation protocol

| Split | Contents | Use |
|---|---|---|
| P-train | JAX κ-strips, seeds hash-partitioned away from P-val | gradients |
| P-val | 300 closed procedural circuits, 200 with OCP labels | model selection, oracle calibration |
| P-OOD | 100 tracks from a shifted generator: 6.5–8 m widths, R 6–8 m hairpins, 1.5 km straights | stress test |
| R-dev (12) | BrandsHatch, Hockenheim, MoscowRaceway, Norisring, Nuerburgring, Oschersleben, Sepang, Sochi, YasMarina (old layout), Melbourne (old layout), Catalunya (pre-2023), Zandvoort (TUMFTM has the pre-2020 DTM layout, not today's banked F1 track) | generator fit, hyperparameters, early stopping |
| **R-test lockbox (12)** | Austin, Budapest, MexicoCity, Montreal, Monza, Sakhir, SaoPaulo, Shanghai, Silverstone, Spa, Spielberg, Suzuka, each also mirrored and reversed | evaluated once per final model |
| Tier-3 | bacinger GeoJSON (Monaco, Singapore, Baku, Jeddah) with assigned widths | completion only, labelled "approximate geometry" |

All 12 R-test layouts are on the 2024–25 calendar (Shanghai was not raced in 2023, so it has only 2024–25 driver data), so the RL test set is also the driver-study set. TUMFTM's IMS file is the banked Indianapolis oval, so it is left out of every split. All tracks are modelled flat, with no elevation or banking, and the write-up says so.

**Baselines in every table:** OCP, OCP + tracker@20Hz, min-curvature + QSS, a per-track specialist, the DR generalist, and the curriculum arms.

**Metrics**

| Metric | Definition |
|---|---|
| Completion C | Share of laps with no corridor exit |
| Gap G | Against T_OCP and against T_OCP@20Hz |
| Line loss | G_line = (T_FP − T_OCP)/T_OCP, where T_FP is the fixed-path OCP time on the agent's path |
| Execution loss | G_exec = (T_RL − T_FP)/T_OCP |
| Per-corner Δt | By type, using OCP apex speed: slow < 120, medium 120–200, fast > 200 km/h |
| Consistency | Standard deviation of lap time |
| Adaptation | Fine-tune curves at 1M, 10M and 50M steps, against a specialist trained from scratch |
| Geometry error bar | Change in G when the OCP is re-solved with widths ±0.5 m, applied to every G |

**Memorisation diagnostics**
- Track-count scaling curve.
- Hockenheim specialist cross-evaluated on R-test. It should crash, reproducing Fuchs' failure on camera.
- Mirrored and reversed versions of every test track.
- Preview ablation: zeroing the preview must break the generalist.
- Warp test: ±5% κ perturbations. The generalist should degrade smoothly while the specialist collapses.
- Absolute-s probe: s should be decodable from the specialist's activations but not from the generalist's.

**Curriculum arms.** Equal 1B-step budget each, 5 seeds (10 for the headline arm).
- DR.
- PLR⊥ scored by positive value loss.
- SFL, scored by learnability p(1 − p) on window completion.
- ACCEL: token mutations, re-filtered for feasibility.
- **Oracle-regret PLR**, with ρ = Σ(t_agent − t_QSS) / Σ t_QSS over segments starting ≥300 m after the window start.

Buffer settings (starting values, tuned on R-dev): 4,000 levels, replay probability 0.5, rank temperature 0.3, staleness 0.3.

**Statistics.**
- rliable IQM over track × seed, with stratified-bootstrap 95% CIs.
- Performance profiles and probability of improvement between arms.
- Equal step budgets, with wall-clock reported.
- Tag `protocol-v1` (hypotheses, metric code, checkpoints) before unsealing. Any later run is labelled post hoc.

**Pivot rule.** If DR completion on R-dev is below 50% after 2B steps, the headline becomes few-shot adaptation (pretrain, then fine-tune curves).

## 6. F1 driver comparison

### The data constraint

Public F1 X/Y positions appear to be snapped to one track model. At 2025 Monza qualifying, the median cross-driver lateral scatter is 5 cm, and OpenF1 says location "lacks lateral position".

Replicate this snapping audit on ≥6 circuits × 2 seasons. If it holds, **every human comparison lives in the distance domain**: t(s), v(s), throttle(s), brake(s), gear(s). Racing lines are compared only between the agent and the OCP.

### Data: do this this week

- The F1 live-timing archive has returned 401 since about 15 Oct 2026, so FastF1 cannot fill an empty cache (issue #978).
- From OpenF1, pull every 2023–26 qualifying and sprint-qualifying session (2023 calls the latter 'Sprint Shootout'): `sessions, laps, car_data, location, stints, weather, race_control, session_result`.
  - Rate limits: ≤3 requests/s and ≤30/min. About 3 h in total.
  - Store as Parquet with a SHA manifest.
- Get official results from jolpica (custom User-Agent required).
- Never commit raw data; OpenF1 is CC BY-NC-SA.

### Pipeline

1. **Registration.** Convert decimetres to metres. Fit a similarity transform onto the TUMFTM centreline: coarse rotation grid, then ICP. Reject a circuit if the RMS residual is over 3 m or the length mismatch is over 1%.
2. **s(t) estimation.** Windowed projection gives s_obs. An RTS smoother on [s, v, a] fuses it with speed (whole km/h), using a per-lap speed scale. Hard anchors at lap and sector times, which are accurate to 1 ms.
3. **Lap filter:**
   - push laps within 102% of the driver's best in that Q segment;
   - no in-laps or out-laps; not deleted; dry;
   - SOFT compound with a new/used flag (cross-check the compound against a second source);
   - exclude a lap if any car is within 1.0 s ahead in a corner window;
   - tow (a car within 2 s at the start of a straight) is a covariate.
4. **Segmentation (additive).** Cut each lap at straight midpoints into 12–20 corner complexes, so segment Δt values sum exactly to the lap Δt. Split each into entry, apex and exit at the reference's s positions. Types: slow, medium, fast.

### References

- **R1, empirical envelope.** The best time in each segment per car-session, across both teammates. No model involved.
- **R2, calibrated physics.** Per team × event:
  - P_eff and CdA from full-throttle straights, with DRS open and closed fitted separately;
  - μ-scale and ClA by fitting QSS v(s) to R1 over corner windows;
  - a final OCP at the fitted parameters.
  - Always calibrate on the **pooled teammate envelope**, never on one driver.
- **R3, learned.** The context-conditioned agent driving each team's calibrated car zero-shot. Used for style traces and the video only, never in the ranking.

### Model (NumPyro, NUTS on the Mac)

```
y_ℓj = t_ℓj − t^R1_kj            (ms; lap ℓ, segment j, car-event k)
y ~ StudentT(ν, μ, sqrt(σ_type² + σ_obs,ℓj²))       σ_obs comes from the smoother
M1 (reference-free):   μ = α[k,j] + θ_d + θ_d,type + θ_d,phase + βᵀ(Qseg, minutes, newSet, tow, temp)
M2 (physics-anchored): α[k] only; y taken relative to R2, so driver×type reads as "% of the time the car had"
```

- **Identification.** Driver effects are identified within teams and linked across teams by driver moves (Hamilton, Sainz, Ocon, Hülkenberg, Tsunoda, Lawson). Publish the connectivity graph.
- **Agreement between M1 and M2 rankings** is the robustness check.
- **Honesty point.** In a teammate contrast the reference cancels out. R2 adds attribution (where the time goes) and mechanism, not ranking.

### Gates before publishing anything

1. Snapping audit and registration residuals published.
2. **Injection-recovery error budget.** Simulate OCP laps at 100 Hz with known injected effects (for example +40 ms in slow-corner entries). Resample at the empirical OpenF1 sampling-interval distribution, quantise speed, binarise brake, and run the full pipeline. Claim nothing below 2σ of the resulting error.
3. **Held-out sector anchor.** Drop sector 2's anchor, predict the sector-2 time, and report the error.
4. **Placebo.** Shuffle driver labels within each car-session; the effects should collapse to zero.
5. **Split-half reliability.** Odd vs even rounds must reach Spearman ρ ≥ 0.5. Otherwise publish teammate pairs only.
6. **Circularity check.** Calibrate on A only, B only, and both; report Kendall τ between the rankings.
7. **8-specification multiverse:** R1 vs R2 × traffic threshold 0.5/1.5 s × top-1 vs top-2 laps. Publish each driver's rank distribution. Drivers whose 90% credible intervals overlap form one tier.
8. **External validity.** Compare against jolpica head-to-head counts, van Kesteren & Bergkamp 2023 and Lindner et al. 2026.

### Confounders

| Confounder | Handling |
|---|---|
| Car (~88% of variance in race results, per van Kesteren & Bergkamp) | α per car-event-segment, plus teammate identification |
| Fuel | Qualifying only; Q-segment effect |
| Tyres | SOFT only; new-set flag |
| Track evolution | Minutes-into-session and Q-segment covariates |
| Traffic and tow | s-domain gap filter plus covariate |
| Setup | Flag teammate speed-trap gaps > 5 km/h; sensitivity run excluding them |
| Weather | Per-minute weather covariates |
| ERS and engine modes | Assumed equal within a team; 2026 excluded from the core study |
| Elevation and banking | Per-corner flags (Eau Rouge, Austin T1) |

### Defensible vs not defensible claims

| Defensible (with intervals) | Not defensible |
|---|---|
| "Relative to his teammate, A gains 45 ± 20 ms/lap in slow-corner entries, 2023–25 dry qualifying" | Anything about drivers' lateral racing lines |
| Corner-type and phase profiles, linked across teams through driver moves; tiers | "X drives at 99.2% of optimal": calibration error (~1%) is much larger than teammate gaps (0.1–0.4%) |
| "In team T's calibrated model, the agent brakes ~25 m later than the drivers' median at T1", only if the difference is above 2σ and above 15 m | "The AI beats Verstappen": the AI drives a model car with perfect information |
| Car-level extraction relative to R2, labelled model-dependent | Braking differences under 15 m; anything below the injection-recovery budget |
| The data-audit findings themselves | Reward recovery, behaviour cloning, "drive like Leclerc"; any 2026 ranking that ignores energy |

**Display.** Agent and OCP ghosts are drawn at their true (s, n). Real drivers appear as a translucent **timing ghost** spanning the track width, labelled "position along lap only".

**2026** is optional and exploratory: corner phases only, split pre/post Miami.

## 7. Milestone roadmap (~10 h/week)

| Phase | Weeks | Hours | Learning goal | Deliverable / gate |
|---|---|---|---|---|
| **P0 Data rescue (do this week)** | 1 | 8 | Rate-limited APIs, provenance | OpenF1 Parquet for 2023–26 qualifying; snapping audit on 3 circuits; uv environments |
| P1 Physics + trackkit | 2–5 | 35 | Tyres, load transfer, Frenet, functional JAX | All gates pass: terminal speed ≈87.5 m/s with TUMFTM rolling resistance (91.1 m/s without), skidpad, dt convergence, JAX/CasADi parity. κ(s) for all 25 circuits; Mac throughput benchmark; clip of you driving it by keyboard |
| P2 Oracles | 6–9 | 35 | Collocation, KKT conditions and duals, QPs | OCP converges on ≥23/25 circuits; replay within 0.3%; min-curvature vs min-time ≈1–2%; JAX QSS oracle validated; **"most valuable metres" plot** (∂T/∂w; ∂T/∂ClA at Monza vs Budapest) |
| P3 RL fundamentals | 10–11 | 15 | Bellman equations, DP, policy gradient, γ horizon | 1-D braking solved by Pontryagin, value iteration and Q-learning; γ late-braking animation; PPO matches PureJaxRL on gymnax |
| P4 Specialist | 12–14 | 30 | PPO debugging, reward hacking | Hockenheim with C = 100% and G ≤ 3% (otherwise fix the simulator or reward first); swarm time-lapse; blooper reel; **MVP-1 blog post** |
| P5 Generator + DR | 15–18 | 35 | Procedural generation, distribution shift | JAX κ-grammar and statistics report; P-val and P-OOD with OCP labels; DR generalist with C ≥ 80% on R-dev; scaling curve (cloud) |
| P6 Curriculum + lockbox | 19–24 | 50 | UED, experimental design, rliable | 5 arms × 5 seeds; ablations on dev only; `protocol-v1` tag, then one unseal; main results table; **MVP-2 "never seen Monza" clip; arXiv v1** |
| P7 Driver study | 13–31, in parallel | 45 | Kalman/RTS smoothing, system identification, hierarchical Bayes | Pipeline and all 8 gates; M1 and M2; leaderboard data |
| P8 Anatomy + probes | 25–27 | 25 | Probing, causal intervention | Line/execution split by corner type; dose response along the braking direction; lookahead vs v²/2a; robustness against OCP + tracker |
| P9 Release | 28–33 | 45 | Writing, communication | Repo with claim ledger; Three.js viewer; 8-page report; 15–20 min video plus 3 shorts |

**Total: about 320 h over about 33 weeks (~8 months).**

**Fast path (about 180 h, ~4.5 months):**
- P3 reduced to DP only.
- Curriculum arms reduced to DR, SFL and oracle-regret PLR.
- Driver study uses M1 only, with gates 1, 2, 4 and 5.
- Release as a blog post plus a short video.

## 8. Architecture ladder (beyond the core)

Already core in this plan: context conditioning, scaling curve, oracle-regret UED, fixed-path decomposition, probes with interventions, asymmetric critic.

| Rank | Module | Value | Effort (weeks) | Verdict |
|---|---|---|---|---|
| 1 | Lookahead Jacobian vs speed, overlaid on v²/2a_brake(v) | 4 | 0.5 | Do |
| 2 | Robustness: RL vs OCP + tracker under μ −5% and +50 ms latency (Song et al. replicated on a car) | 4 | 1.5 | Do |
| 3 | 1-D conv preview encoder | 3 | 0.5 | Ablation |
| 4 | Hybrid score p(1−p) + λ·p·regret | 3 | 0.5 | Only if P6 is on schedule |
| 5 | Draw-your-track web sandbox (JS port of `f` with parity ≤1e-4, Float32 MLP) | 4 | 2.5 | After MVP-2 |
| 6 | From-scratch SAC, then QR-SAC, vs PPO on generalisation | 3 | 3 | Stretch, for learning |
| 7 | 2026 energy-aware OCP (4 MJ battery, MGU-K taper, Straight Mode) | 3 | 4 | Sequel |
| 8 | TD-MPC2 / DreamerV3 zero-shot comparison | 3 | 4 | Only with spare GPU budget |
| 9 | Multi-car racing (frozen past policies, then a league; DeepSets opponent encoder) | 5 for YouTube | 8+ | Sequel project |
| 10 | Driver-style VAE over corner profiles | 2 | 2 | Skip; it mostly embeds the car |

**Padding: skip, and say why in the README.**
- Transformer over a fixed-length preview
- PBT
- PAIRED
- Pixel or bird's-eye-view inputs
- LLM-written rewards
- Dreamer 4
- Behaviour cloning, GAIL or inverse RL from 3.7 Hz telemetry with on/off brake
- AMP "drive like X"
- jax-metal or jax-mps ports
- Unity or Isaac

## 9. Compute plan (M3 Pro 18 GB, plus a rented RTX 4090)

The Mac is limited by the neural network, not the physics. Measured: NumPy physics ~8M steps/s; PPO-sized MLP training 0.18M samples/s on CPU. Estimated end-to-end PPO: ~40k steps/s on CPU for the 3×256 baseline MLP; expect 3–6× slower with the default SimBa-lite actor and 1.4M-parameter critic.

**Setup rules:**
- JAX on CPU locally; never jax-metal.
- The same code runs with `jax[cuda12]` on RunPod.
- Pin versions and re-benchmark after every upgrade.
- Keep ≥2k envs on CPU.
- `XLA_FLAGS=--xla_force_host_platform_device_count=N` (or the `jax_num_cpu_devices` config) only helps if you shard or pmap the envs across the virtual CPU devices it creates.

| Workload | Where | Time (est.) | Cost |
|---|---|---|---|
| Tests, oracles, 1-D braking, gymnax | Mac | minutes | $0 |
| ~400 OCPs + ~500 fixed-path OCPs (2–5 min each, unverified) | Mac, 10 processes | 2 nights | $0 |
| Specialist, 50–100M steps | Mac | 20–45 min (baseline MLP); 1–4 h (SimBa-lite) | $0 |
| **1-hour GPU pilot (measure steps/s first)** | 4090, $0.34/h | 1 h | <$1 |
| Scaling curve: 6 N × 3 seeds × 0.5B | 4090 | 5–15 h | $2–5 |
| UED: 5 arms × 5 seeds × 1B, plus 5 extra headline seeds | 4090 | 16–40 h | $5–14 |
| Ablations, specialists, fine-tunes (~12B steps) | 4090 | 7–20 h | $2–7 |
| NumPyro fits + multiverse | Mac | overnight | $0 |
| **Total ≈ 50B steps** | | ~27–67 h | **~$10–25 expected; cap $150** (covers debugging and estimate error) |

Vmap seeds on one GPU, checkpoint to object storage, and auto-stop pods. The track buffer is 4,000 tracks × ~98 KB ≈ 0.4 GB.

## 10. Demo and video plan

**Arc** (Yosh's proven structure, aimed also at the Chain Bear / Driver61 explainer audience):

1. **Cold open.** Suzuka's Esses: the agent ghost, the OCP ghost and the pole lap's timing ghost. "This AI has never seen Suzuka."
2. **Stakes.** "Every superhuman car-racing AI so far was trained for specific tracks."
3. **Physics in 90 s (Manim).** Friction ellipse; downforce equals weight at about 188 km/h; g-g-v diagram; the γ braking animation.
4. **You drive your own simulator, badly.**
5. **Failure montage.** Wall-riding, wiggling on straights, spins.
6. **Breakthrough.** 1,000 colour-coded ghosts master Hockenheim.
7. **Twist.** The specialist crashes at Spa's La Source.
8. **Track factory.** The curriculum's "just hard enough" buffer evolving over training.
9. **Lockbox opens.** 12 circuits, delta bar, line vs execution. Bonus: Monza reversed.
10. **Inside its head.** Push the braking direction and the car brakes N m earlier.
11. **Robustness.** Cut grip by 5%: the "perfect" OCP plan spins, the agent adapts.
12. **The grid.** The "F1 data hides the line" reveal, then driver tiers with caveats on screen.
13. **Close.** Open-source link and the draw-your-track site.

**Assets:**
- Three.js viewer with synced v(s), Δt(s), throttle/brake strip and a friction-circle dot.
- Three shorts: "never seen Suzuka", "the brake neuron", "F1 data can't see the line".
- Hero figure: a grid of all 12 circuits.

**Branding:** say "F1-style", with no logos or liveries. Keep the F1-data video non-monetised unless you get legal advice.

## 11. Top risks and mitigations

| Risk | Mitigation |
|---|---|
| OpenF1 gets locked like the F1 archive | Download everything in P0; TracingInsights as a fallback |
| Model mismatch reported as an optimality gap | One shared `f()`; parity test; OCP replay within 0.3%; report against T_OCP@20Hz too |
| Generalist fails on hairpins and chicanes | Hairpins and chicanes in the grammar; P-OOD stress set; curriculum; pivot rule |
| Reward hacking and simulator exploits | Frenet s; conservative corridor; v² penalty; w·κ ≤ 0.5; dt and energy tests; blooper logger |
| Curriculum is just noisy DR | DR and SFL controls; probability of improvement; pre-registration |
| QSS oracle mis-ranks levels | Calibrate against 200 OCPs; publish Spearman ρ; fall back to SFL |
| OCP does not converge | Smooth saturations; variable scaling; warm starts; μ-homotopy; report failures |
| Driver rankings are noise or reflect the car | Teammate identification; injection-recovery; placebo; split-half gate; tiers |
| Calibration circularity | Pooled-teammate envelope; A/B/both check |
| Compute estimates are wrong | P1 benchmark; 1-hour GPU pilot; fast path |
| Scope creep or burnout | Every module must fill an ablation cell; MVP-1 and MVP-2 ship before the big video |
| Licensing | Code under Apache-2.0; TUMFTM (LGPL) and OpenF1 (CC BY-NC-SA) fetched by script, never vendored; do not copy Linesight |

## 12. Starter stack

**Environments.** uv (https://github.com/astral-sh/uv), with three environments:
- `apex-rl`: Python 3.12, JAX 0.11.x
- `apex-ocp`: CasADi, with quadprog overridden to 0.1.13
- `apex-data`: polars, pyarrow, NumPyro

**RL**
- JAX: https://docs.jax.dev
- Flax: https://github.com/google/flax
- Optax: https://github.com/google-deepmind/optax
- gymnax: https://github.com/RobertTLange/gymnax
- References only (do not import in `agents/`):
  - PureJaxRL: https://github.com/luchris429/purejaxrl
  - rejax: https://pypi.org/project/rejax/
  - CleanRL: https://github.com/vwxyzjn/cleanrl
  - sbx: https://github.com/araffin/sbx
  - f1tenth_gym_jax: https://github.com/f1tenth/f1tenth_gym_jax

**Curriculum and statistics**
- JaxUED: https://github.com/DramaCow/jaxued
- SFL: https://github.com/amacrutherford/sampling-for-learnability
- dcd (archived): https://github.com/facebookresearch/dcd
- rliable (archived, still usable): https://github.com/google-research/rliable

**Optimal control and tracks**
- CasADi: https://pypi.org/project/casadi/
- TUMFTM global_racetrajectory_optimization (reference only): https://github.com/TUMFTM/global_racetrajectory_optimization
- trajectory_planning_helpers: https://github.com/TUMFTM/trajectory_planning_helpers
- TUMFTM racetrack-database: https://github.com/TUMFTM/racetrack-database
- TUMFTM laptime-simulation: https://github.com/TUMFTM/laptime-simulation
- TUM-AVS GGGVDiagrams (GPL-3.0: read it, don't copy it into an Apache-2.0 repo): https://github.com/TUM-AVS/GGGVDiagrams
- bacinger/f1-circuits: https://github.com/bacinger/f1-circuits

**F1 data**
- OpenF1: https://openf1.org/docs/
- jolpica-f1: https://github.com/jolpica/jolpica-f1
- TracingInsights: https://github.com/TracingInsights
- FastF1 issue #978: https://github.com/theOehrly/Fast-F1/issues/978
- HumanoidsBonn/f1-optimization (alignment code): https://github.com/HumanoidsBonn/f1-optimization
- NumPyro: https://github.com/pyro-ppl/numpyro
- ArviZ: https://python.arviz.org

**Visualisation**
- Rerun: https://rerun.io
- Three.js: https://threejs.org
- Manim: https://www.manim.community
- raylib / pygame-ce: https://pypi.org/project/raylib/

**Cloud**
- RunPod: https://www.runpod.io/gpu-cloud/pricing

**Reading by phase**

| Phase | Reading |
|---|---|
| P1 | Milliken, *Race Car Vehicle Dynamics*; Pacejka, *Tire and Vehicle Dynamics* |
| P2 | Perantoni & Limebeer 2014; Christ et al. 2019; Betts, *Practical Methods for Optimal Control* |
| P3 | Sutton & Barto, chapters 3–6 and 13 |
| P4 | PPO; GAE; "37 implementation details" |
| P5 | Procgen; Fuchs et al. arXiv 2008.07971; Green et al. arXiv 2606.27348 |
| P6 | Robust PLR arXiv 2110.02439; ACCEL arXiv 2203.01302; SFL arXiv 2408.15099; rliable |
| P7 | van Kesteren & Bergkamp arXiv 2203.08489; Lindner et al. arXiv 2608.04629; Kegelman et al. 2017 |
| P8 | Taufeeque et al. arXiv 2407.15421; Bush et al. arXiv 2504.01871; Song et al. arXiv 2310.10943 |