# Status (21 Oct 2026)

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
| **K5 DR pilots** | 200M steps each on procedural strips; seeds 0–4 per variant ([seeds](findings/generalist-seeds.md)) | Across 5 seeds the variants do not differ (5.4, 5.8, 5.8 of 12). Seed 0: v1: zero-shot it completes 4/12 R-dev circuits at **6.2-8.6% from the optimum**; every crash is a corridor exit in a corner tighter than ~90% of the generator's ([v1](findings/dr-pilot-200M-crashes.md)). v2 (30% hairpin-heavy strips): **5/12**, adding Brands Hatch 7.4% and Hockenheim 6.7%; remaining crashes are R 12-14 m hairpins ([v2](findings/dr-pilot-v2-tilt30-crashes.md)). v3 (50% of strips from a (1/r_min)^2 tilt): **6/12** at 6.4-9.1%, but not a superset of v2: it newly completes Catalunya and newly crashes at Brands Hatch. Gate is 80%; the psychophysics study (below) is diagnosing the crashes before more variants are trained |
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
of the agent's own objective from the same state. The protocol
([protocol](interp/psychophysics-protocol.md), rulings R1–R31 in its Section 12) was frozen at
the tag `psychophysics-prereg-v2` before the protocol rollouts (only pre-registered pilot
runs came earlier); harness in
`src/apexzero/interp/` and `scripts/psychophysics.py` ([implementation map](interp/IMPLEMENTATION.md)).

**Step 1 is done** ([findings](findings/psychophysics-step1.md); the confirmatory
decisions are in `outputs/psychophysics/analysis/decisions.json`, from one run):
- **P1a confirmed** for every generalist: G3 (primary), G1 and G2 (secondary), and all 9
  networks across seeds (replication). On typical approaches the networks brake for a corner
  where the optimum of their own objective does.
- **P3 confirmed** for G3, G1 and variants v1 and v3: braking distance scales like the physics.
- **No fast-approach overspeed was seen** in any network. P1-tail is inconclusive for G3 and G1,
  and refuted for G2 and v2. That refutation is fragile: 1 of 3 seeds without unsettled runs.
- **G2's real tail crashes are not single-corner failures** (P8-tail refuted). In all three, the
  constant-width rung is necessary to rescue the car. Descriptively, 159 of 160 real-road corner
  crashes leave by the inside edge where the inside width moves by metres. Such changes almost
  never occur in a tight training corner.
- **The specialists' memorisation (P4) is not decided.** S1 is class 6. S2 and S3 were not
  testable in the confirmatory run, because their buffers were built after the tag (a logged
  deviation). The rerun puts them in class 2, but vacuously, and they crash on a null straight.
- **The networks are not mirror-symmetric** (P5, a control). The re-run harness gate passes, so
  the asymmetry is learned. It is mostly a 5–12 m difference in where they brake.

**Width test, acting on that result** ([findings](findings/width-test.md); exploratory, not
pre-registered). Generalists trained on strips whose per-side widths are segments of real
circuit width profiles (`train_dr.py --real-widths`) complete more R-dev circuits:
- v1: from 5.4 to 8.25 of 12 (5 vs 4 seeds);
- v3: from 5.8 to 8.5; the v3 seed ranges do not overlap;
- Sochi: from 0 of 10 seeds to 4 of 8; Norisring and Yas Marina stay at 0.

Caveat: the width profiles came from the evaluated circuits. **Running now:** a
leave-circuits-out test, with its design and analysis committed before the runs
([plan](findings/width-test-loo-plan.md), `scripts/width_loo_analysis.py`).

**Next:** step 2, receptive fields ([design draft](interp/receptive-fields-design.md), under
review). It compares the network's closed-loop sensitivity to the road ahead with the optimum's
KKT sensitivity at the same states. Taking step 2 before step 1b departs from protocol 9.6, and
the departure is logged. Then idea 2: does the network encode Pontryagin's costate?

`scripts/dashboard.py` shows everything running: stages, agent workflows, GPU boxes.

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

1. Idea 1 step 2 (receptive fields), then idea 2 (costate), as above.
2. The leave-circuits-out width test decides whether real widths go into the generator by
   default (`generator-v1`).
3. Paused by choice: the paper's curriculum audit (maze E5 at scale, racing PLR / SFL), K2b
   production TO-OCP labels, the drivers study, the demo, and the lockbox evaluation
   (`protocol-v1`).

## Compute

GPU training runs on Brev (Massed Compute). `scripts/gpu_setup.sh` and `scripts/gpu_sync.sh`
set a box up and move R-dev data only; the lockbox never leaves the Mac. At about 430k PPO
steps/s per run (two runs per L40S), a 200M-step run takes minutes instead of 3 h on the Mac.
Boxes are deleted when their results are pulled back.

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
