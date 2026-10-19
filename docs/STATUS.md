# Status (19 Oct 2026)

Phases follow [the two-track plan](plan/two-track-plan.md): K = shared core, P = paper, D = demo.

## Done

| Phase | What exists | Key results |
|---|---|---|
| **P0 data** | OpenF1 archiver (resumable, rate-limited, SHA-256 manifest), jolpica results, snapping audit | 111 sessions 2023-26, 5,196 files, all checksums verified. **The public position feed carries no driver-specific racing line** across 85 qualifying sessions (driver ICC ~0; a planted 0.5 m line is detected at 93-100% of stations). [report](findings/snapping-audit.md) |
| **K1 simulator** | One ODE for JAX and CasADi; C2 B-spline track lookups identical in JAX/NumPy/CasADi; non-singular Frenet references; procedural generator | Parity < 1e-5 (dynamics), 1e-12 (lookups). Gates pass: terminal speed 315 km/h, coast-down, low-speed kinematics, braking 4-7 g, RK2 at 5 ms within 3e-6 of RK4 at 1 ms, skidpad 91-93% of the point-mass envelope (power-limited above ~290 km/h) |
| **K2a lap optimisers** | s-domain minimum-time OCP (CasADi/IPOPT), slip-limited; QSS oracle (NumPy and JAX); TO-OCP window mode | OCP converges on 24/25 circuits (IMS, the banked oval, excluded); simulator reproduces it to < 0.016% (Monza: 0.0003%). TO-OCP on a 10 s window: replay of the optimal controls reproduces J* exactly; 44 s solve; same optimum from two very different warm starts |
| **K3 learning** | PPO from scratch; 1-D braking by Pontryagin and value iteration | PPO matches or beats rejax (CartPole 475 vs 328, Pendulum -158.5 vs -159.5). Horizons of ~0.3 s brake late and crash; from gamma 0.9 every agent brakes one 50 ms control step before Pontryagin's switch point. [ppo](findings/ppo-verification.md), [braking](findings/braking-gamma.md) |
| **K4 specialist** | Training script, timed-lap evaluation | **Gate met.** Hockenheim, 200M steps on the Mac CPU, three seeds: 100% completion at **4.80%, 4.84% and 4.75%** from the optimal lap (gate: 8%) |
| **K5 (part)** | Procedural DR env with on-device track bank; randomised cars and hidden grip; paper level sets (A_DR / A_tail / A_OOD / pool) from disjoint seed streams; tilted and OOD generators; critic ensemble heads with per-head targets; log-spaced checkpoints | Generator matches R-dev corner statistics (radius W1/median 0.07); 1,000 strips in ~2 s |
| **P harness (part)** | Score spec extracted from pinned JaxUED / minimax / MNA / SFL source ([spec](paper/score-spec.md)); estimators; audit rollouts; maze oracle; JaxUED maze example running unmodified | Estimators reproduce every hand-computed test vector in the spec; PVL matches JaxUED's own function; MNA closed form equals a port of the reference code to 1e-10; BFS optimum matches executed optimal plans in JaxUED's env exactly |
| **K5 DR pilots** | 200M steps each on procedural strips, ~3 h each on the Mac | v1: zero-shot it completes 4/12 R-dev circuits at **6.2-8.6% from the optimum**; every crash is a corridor exit in a corner tighter than ~90% of the generator's ([v1](findings/dr-pilot-200M-crashes.md)). v2 (30% hairpin-heavy strips): **5/12**, adding Brands Hatch 7.4% and Hockenheim 6.7%; remaining crashes are R 12-14 m hairpins ([v2](findings/dr-pilot-v2-tilt30-crashes.md)). v3 (50% of strips from a (1/r_min)^2 tilt): **6/12** at 6.4-9.1%, but not a superset of v2: it newly completes Catalunya and newly crashes at Brands Hatch. Gate is 80%; the psychophysics study (below) is diagnosing the crashes before more variants are trained |
| **K2a TO-OCP speed** | Exact floor-indexed lookups, hand-structured derivatives compiled to C, horizon continuation | **Gate met:** a 60 s window solves in 46-232 CPU-s (mean 97 s; gate 12 CPU-min); replay stays exact (< 5e-9). Fixed a slip-limit margin bug the benchmark exposed. Local optima differ by ~1e-4-1e-3 of J* ([report](findings/to-ocp-spike.md)) |
| **P curriculum loop** | `ued/`: racing PLR-perp on JaxUED's own LevelSampler (chunked windows, pre-drawn assignments, per-transition loss masks, writes at episode end) and SFL; `scripts/train_ued.py` | Spec tests P-1..P-9 and S-1..S-4 pass. First 200M PLR-perp (MaxMC score, 1,000-level pool, buffer 250) pilot: **2/12 R-dev circuits** (Brands Hatch 16.7%, Sepang 12.0%) against DR v1's 4/12 at the same steps; one seed, not yet a finding |
| **P maze E5 audit** | `maze/audit.py`: audits JaxUED's own maze checkpoints against exact BFS regret | PVL / MaxMC equal JaxUED's to 2.8e-7. Pilot at 0.7% of the default run: PVL and MaxMC correlate **negatively** with exact regret (rho ~ -0.2) while the agent-only baseline is +0.26. One seed, very early: a pilot, not a finding ([report](findings/maze-audit-pilot.md)) |
| **P pilot audit** | `audit/fidelity.py`: proxy-label targets, Spearman / RC@10% / two-loading regression with bootstrap CIs, split-half reliability and window power analysis | Proxy-label pilot on 13 DR checkpoints, explicitly not a finding: target reliability split-half r ~0.8 at 16 windows per half; ready for real labels via `--j-opt` ([report](findings/pilot-audit.md)) |
| **D drivers-lite** | `drivers/`: registration, anchored RTS s(t), push-lap filter, additive corner phases, teammate deltas within Q segments, R1 envelope, injection-recovery, placebo, split-half | Held-out sector error 10-18 ms; claim limits per phase ~24-100 ms (2 laps per driver); placebo and split-half both run; split-half below the 0.5 gate at pilot scale, so pair case studies only for now ([report](findings/drivers-pilot.md)). Found and fixed: race control's "LAP n" is OpenF1 lap n-1 |
| **Paper draft** | `paper/` (LaTeX, 26 pages): introduction, related work, environment, TO-OCP, estimators and deviations table, protocol, threats, appendices; every number traced to a repo file; unverified citations marked | Builds with latexmk; results are placeholders |
| **D demo tooling** | `viz/`: versioned trajectory export, figures, static Three.js viewer, optional Rerun logger | Hockenheim specialist vs optimum ghost figures in [docs/demo](demo/); trajectory JSON is regenerated by script, not committed (it contains TUMFTM geometry) |

## Deep exploration: inside the racing policies

Idea 1, "where should a racer look?", step 1 is **network psychophysics**: controlled
single-corner stimuli presented to every agent and checkpoint, scored against the exact optimum
of the agent's own objective from the same state. The pre-registered protocol
([protocol](interp/psychophysics-protocol.md), rulings in its Section 12) and harness
(`src/apexzero/interp/`, `scripts/psychophysics.py`, [implementation map](interp/IMPLEMENTATION.md))
are built and reviewed; the pilot replication matches. Pilots seen while designing: the
generalists complete isolated synthetic hairpins (R 9-30 m) that they crash on in real
circuits, and the specialist crashes on a plain synthetic straight. Next: answer keys and
positive controls, the `psychophysics-prereg-v2` tag, then rollouts. Step 2 measures
receptive fields; idea 2 asks whether the networks encode Pontryagin's costate.

## Running now

Answer-key layout (`psychophysics.py keys --layout-only`) and the rulings round; `scripts/dashboard.py`
shows everything running.

## Protocol deviation (recorded)

Optimum computations on R_TEST (lockbox) circuits exist: the first all-circuit s-domain OCP
batch (run before the revised plan restricted OCPs to R-dev) and the TO-OCP speed benchmarks
(Monza, Spa, Silverstone, Budapest; that workflow's instructions lacked the lockbox rule). None
was used for generator fitting, tuning or checkpoint choice. All 24 files are moved to
`outputs/sealed/` (with a README) and must not be read until `protocol-v1` is tagged. Every
later task carries the lockbox rule explicitly.

## Decisions taken (reversible, recorded here)

- **Driver registration gate.** The plan's gate (RMS <= 3 m, length within 1%) rejects 8 of 12 current layouts because the OpenF1 track model behaves like a racing line (inside offsets of 0.6-2.8 m), not a centreline. Adopted instead: excursion beyond the TUMFTM edges at the 99th percentile <= 8 m, with RMS and length reported as diagnostics. It passes all current layouts and no old layout in 2023, 2024 and 2025. Not yet applied in code.
- **Track geometry stays out of git**, including derived trajectory JSON; figures are committed with attribution.
- **Human telemetry from lockbox circuits is allowed** (snapping audit, driver study); the lockbox covers agent and optimum evaluation only.

## Next

1. K4: two more specialist seeds. K5: GPU pilot, `sim-v1` / `generator-v1` tags.
2. Paper: maze E5 audit (checkpoints of JaxUED's DR / PLR / SFL runs scored against BFS regret);
   the racing PLR-perp / SFL curriculum loop (JaxUED's `LevelSampler`, chunked scoring).
3. K2b: production TO-OCP labels (speed is the open question; the plan budgets a 64-vCPU box).

## Decision you will need to make

**A rented GPU.** Everything so far runs on the Mac CPU (~45k PPO steps/s). The paper's
training runs total roughly 30-35B steps (E0 families plus the curricula): about 9 days of
continuous Mac time, versus roughly 10-20 h on an RTX 4090 (~$0.34/h; the plan caps spend at
$250). Renting needs your account and payment method, so it is yours to set up when the
pilots are done; nothing is blocked on it yet.

## Things found along the way (all fixed, each with a test)

- The optimum drifted through Monza's first chicane at 20 deg of rear slip, exploiting the soft
  post-peak tyre curve, for a 0.04 s gain. Fixed with a 10 deg slip limit applied identically
  in the OCP (constraint) and the env (termination).
- Brakes kept pushing after a stop, so the car reversed at -110 m/s. Brakes now fade at standstill.
- The QSS speed cap was a 120 m/s placeholder, so open strips started above top speed.
- The smooth ReLU leaked throttle and brake at neutral pedal (found by the coast-down gate).
- Closed circuits' knot spacing is ~1.9998 m, not 2 m; the TO-OCP now takes it as a parameter.
- Critic values are in normalised units; audit scores now rescale them (spec DEV-6).
- One car's coordinate stream (2023 Jeddah, car 21) is ~3 m off everywhere; screened and reported.
