# Status (18 Oct 2026, end of day 1)

Phases follow [the two-track plan](plan/two-track-plan.md): K = shared core, P = paper, D = demo.

## Done

| Phase | What exists | Key results |
|---|---|---|
| **P0 data** | OpenF1 archiver (resumable, rate-limited, SHA-256 manifest), jolpica results, snapping audit | 111 sessions 2023-26, 5,196 files, all checksums verified. **The public position feed carries no driver-specific racing line** across 85 qualifying sessions (driver ICC ~0; a planted 0.5 m line is detected at 93-100% of stations). [report](findings/snapping-audit.md) |
| **K1 simulator** | One ODE for JAX and CasADi; C2 B-spline track lookups identical in JAX/NumPy/CasADi; non-singular Frenet references; procedural generator | Parity < 1e-5 (dynamics), 1e-12 (lookups). Gates pass: terminal speed 315 km/h, coast-down, low-speed kinematics, braking 4-7 g, RK2 at 5 ms within 3e-6 of RK4 at 1 ms, skidpad 91-93% of the point-mass envelope (power-limited above ~290 km/h) |
| **K2a lap optimisers** | s-domain minimum-time OCP (CasADi/IPOPT), slip-limited; QSS oracle (NumPy and JAX); TO-OCP window mode | OCP converges on 24/25 circuits (IMS, the banked oval, excluded); simulator reproduces it to < 0.016% (Monza: 0.0003%). TO-OCP window mode, 60 s windows (1,200 steps): 53-198 CPU-s on 7 circuit windows; a production-like level (8 km strip, own car and hidden grip) costs 131-520 CPU-s (median 206 s) including its own s-domain guess, on 7 of 8 levels (gate: 720 s); the eighth failed its single start. Every optimum replays in float64 to < 3e-12. Single starts can land 0.3-0.5% low, so labels need 2+ independent starts. [spike](findings/to-ocp-spike.md) |
| **K3 learning** | PPO from scratch; 1-D braking by Pontryagin and value iteration | PPO matches or beats rejax (CartPole 475 vs 328, Pendulum -158.5 vs -159.5). Horizons of ~0.3 s brake late and crash; from gamma 0.9 every agent brakes one 50 ms control step before Pontryagin's switch point. [ppo](findings/ppo-verification.md), [braking](findings/braking-gamma.md) |
| **K4 specialist** | Training script, timed-lap evaluation | Hockenheim, 200M steps on the Mac CPU: 100% completion, **4.8% from the optimal lap** (gate: 8%), seed 1 of 3 |
| **K5 (part)** | Procedural DR env with on-device track bank; randomised cars and hidden grip; paper level sets (A_DR / A_tail / A_OOD / pool) from disjoint seed streams; tilted and OOD generators; critic ensemble heads with per-head targets; log-spaced checkpoints | Generator matches R-dev corner statistics (radius W1/median 0.07); 1,000 strips in ~2 s |
| **P harness (part)** | Score spec extracted from pinned JaxUED / minimax / MNA / SFL source ([spec](paper/score-spec.md)); estimators; audit rollouts; maze oracle; JaxUED maze example running unmodified | Estimators reproduce every hand-computed test vector in the spec; PVL matches JaxUED's own function; MNA closed form equals a port of the reference code to 1e-10; BFS optimum matches executed optimal plans in JaxUED's env exactly |

## Running now

- **DR pilot**, 200M steps on procedural strips (the biggest open risk: does a generalist learn?).

## Next

1. K4: two more specialist seeds. K5: GPU pilot, `sim-v1` / `generator-v1` tags.
2. Paper: maze E5 audit (checkpoints of JaxUED's DR / PLR / SFL runs scored against BFS regret);
   the racing PLR-perp / SFL curriculum loop (JaxUED's `LevelSampler`, chunked scoring).
3. K2b: production TO-OCP labels. The spike meets the 12 CPU-min gate on 7 of 8 production-like
   levels. Each level needs its own s-domain OCP (the continuation guess, 74-132 CPU-s on the
   reachable part of the strip). With 3 starts the estimate is ~1,160 CPU-h for 5,400 levels,
   before the pilot measures how often a start fails or lands in a worse basin. Still to
   build: a retry policy for failed stages, and the label schema.

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
- TO-OCP optima near the limit are open-loop unstable. IPOPT's ~1e-8 defects grew to 4 cm over a
  60 s replay and put the car off the track, and its relaxed bounds left slip-riding optima a
  hair past the limit. Every solve is now refined to ~1e-12 defects without relaxed bounds,
  and the slip limit has a 1e-4 rad margin.
- Critic values are in normalised units; audit scores now rescale them (spec DEV-6).
- One car's coordinate stream (2023 Jeddah, car 21) is ~3 m off everywhere; screened and reported.
