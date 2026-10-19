# TO-OCP spike (K2a): speed

Training-objective OCP in window mode on the RL environment's exact problem (RK2 x 10 at
5 ms, 20 Hz zero-order hold, corridor and slip limits tightened by 3 cm and 1e-4 rad,
discounted reward). Replay: the optimal controls applied open-loop in the float64 JAX
environment, with its termination checks. An optimum is *verified* when the solve
succeeded, the replay survives every step and it reproduces J* within 1e-6
(`replay.verify_window`); every row below reports the best verified optimum of its starts.

**Result: the gate (12 CPU-minutes per 60 s solve) is met on 7 of 8 production-like
levels, including the s-domain guess every level needs: 131-520 CPU-s, median 206 s.
The eighth (pool:5) failed: a continuation stage hit IPOPT's iteration limit, after
791 CPU-s, with no verified optimum.** A production-like level is an 8 km generated strip
with its own random car and hidden grip (`--levels pool:i`). Continuation cannot run
without the level's own s-domain minimum-time OCP. Solved on the part of the strip a 60 s
window can reach, that OCP costs 74-132 CPU-s. On seven circuit windows the TO-OCP alone
takes 53-198 CPU-s (mean 99 s). The CasADi step still equals the simulator (< 1e-10), and
every verified optimum replays to its J* within 3e-12.

**Label budget.** The plan gives every level 3 starts, which share one s-OCP: s-OCP +
3 x TO-OCP is about 775 CPU-s per level at these means (pool:5's failed start included),
or ~1,160 CPU-h for 5,400 levels. That is at the low end of the plan's 1,200-3,000 CPU-h.
Two things are not settled. A single start can land in a clearly worse local optimum: on
Hockenheim at 60 s the policy start is 0.31 (0.45%) below the s-OCP start (below). And a
start can fail outright (pool:5). The pilot labels should measure both rates before the
budget is final.

## Production-like levels: the per-level cost

`--levels pool:0 ... pool:7`, 60 s windows (1,200 steps), x0 at s = 0 with
0.8 v_QSS(0), first guess and extension guess both from the level's s-OCP (no policy is
trained for these strips). The s-OCP is solved on the strip up to where the car could get
in 60 s at its top speed, plus 300 m (`window.reachable_strip`). Its CPU is added to the TO-OCP's to
give the level's cost. Keys from PRNGKey(0) (`--level-root`); the level sets have no
committed root yet.

| Level | Car: P, CLA, mu x hidden | s-OCP on | s-OCP it | s-OCP CPU | TO-OCP iter | TO-OCP CPU | **Level CPU** | J* | Replay diff | Survives |
|---|---|---|---|---|---|---|---|---|---|---|
| pool:0 | 567 kW, 3.89, 0.979 x 1.006 | 5,352 of 8,190 m | 1,020 | 132 s | 1,199 | 74 s | **206 s** | 68.66409 | -2.3e-12 | yes |
| pool:1 | 555 kW, 4.28, 0.990 x 1.019 | 5,914 of 8,190 m | 612 | 91 s | 1,150 | 87 s | **178 s** | 79.74133 | +1.6e-15 | yes |
| pool:2 | 747 kW, 5.28, 1.013 x 0.993 | 6,432 of 8,190 m | 697 | 108 s | 4,870 | 412 s | **520 s** | 75.21653 | +4.7e-15 | yes |
| pool:3 | 638 kW, 5.80, 0.974 x 1.001 | 5,946 of 8,190 m | 514 | 76 s | 2,502 | 160 s | **235 s** | 69.99876 | -1.9e-12 | yes |
| pool:4 | 577 kW, 4.14, 1.096 x 1.002 | 5,100 of 8,190 m | 732 | 96 s | 2,723 | 250 s | **347 s** | 71.01751 | -1.1e-12 | yes |
| pool:5 | 743 kW, 4.19, 1.009 x 1.007 | 6,212 of 8,190 m | 787 | 121 s | 6,339 | 670 s | **791 s** | none verified | - | - |
| pool:6 | 696 kW, 5.42, 1.084 x 0.981 | 5,678 of 8,190 m | 525 | 74 s | 884 | 57 s | **131 s** | 77.51206 | +1.6e-14 | yes |
| pool:7 | 581 kW, 5.17, 0.955 x 0.984 | 5,428 of 8,190 m | 766 | 101 s | 1,523 | 89 s | **190 s** | 66.08384 | +3.9e-15 | yes |
| pool:0, s-OCP on the whole strip | as pool:0 | 8,190 of 8,190 m | 1,249 | 259 s | 782 | 46 s | **305 s** | 68.66397 | +1.4e-13 | yes |

- **pool:5.** The 40 s stage hit max_iter (3,000 iterations, 264 CPU-s). The 60 s stage
  then fell back to the 20 s optimum, as `solve_continuation` now does, and hit max_iter
  too. In the run before the refinement (below), the same level converged at 40 s in
  2,210 iterations, but its optimum did not replay.
  - *A second start* was one direct solve of the 60 s window from the s-OCP guess
    (`--method direct --tag direct`). It failed too: 3 rounds, 8,834 iterations and
    1,068 CPU-s, still at max_iter.
  - *Still open.* These runs do not show whether the level is only slow, or whether
    something about it (a 743 kW car on this strip) defeats IPOPT. A labeller needs a
    retry policy and an iteration budget for such levels.
- **The cut s-OCP.** Solving the s-OCP on the whole strip doubles its cost (259 vs
  132 CPU-s on pool:0) and gives an equivalent guess: the chain lands within 1.2e-4 of the
  same J*. A cut at the centreline-QSS reach is not safe, because the racing line is ~20%
  faster: on pool:0 the s-OCP on such a cut (3.4 km) ended 1.8 s short of 60 s. A looser
  s-OCP tolerance (1e-4) saved nothing (1,018 vs 1,020 iterations).

## Hockenheim, x0 = [0, 0, 0, 0.8 v_QSS(0), 0, 0, 0, 0]

Two independent starts: the Hockenheim specialist's rollout ("policy") and the s-domain
optimum mapped to time ("ocp"). The polish solves the next longer window from the policy
chain's optimum and re-solves the target from it. The "first spike" rows are the original
code. The "before the review" rows are the first version of this work, kept for the
per-iteration speed-up.

| Window | Method | Stages (s) | Iter | Rounds | Solve wall | CPU | J* | Replay diff | Survives |
|---|---|---|---|---|---|---|---|---|---|
| 10 s | first spike's code, single thread | 10 | 160 | 3 | 196 s | 144 s | 48.08917 | -2.0e-09 | yes |
| 10 s | continuation from policy | 10 | 172 | 3 | 4 s | 3 s | 48.08917 | +7.4e-16 | yes |
| 10 s | continuation from ocp | 10 | 105 | 1 | 3 s | 2 s | 48.08917 | -1.6e-14 | yes |
| 10 s | best of policy, ocp, polish | 10 10 20 10 | 391 | 6 | 12 s | 9 s | 48.08917 | +7.4e-16 | yes |
| 20 s | first spike's code, single thread | 20 | 1,268 | 4 | 3,795 s | 2,391 s | 60.28690 | -1.1e-09 | yes |
| 20 s | before the review: direct from policy | 20 | 663 | 4 | 31 s | 25 s | 60.28689 | -1.1e-09 | yes |
| 20 s | continuation from policy | 10 20 | 251 | 4 | 8 s | 7 s | 60.28685 | -3.2e-13 | yes |
| 20 s | continuation from ocp | 10 20 | 207 | 2 | 8 s | 6 s | 60.28689 | +2.1e-15 | yes |
| 20 s | continuation from policy + polish | 10 20 40 20 | 1,313 | 6 | 130 s | 102 s | 60.28685 | -3.2e-13 | yes |
| 20 s | best of policy, ocp, polish | 10 20 10 20 40 20 | 1,520 | 8 | 138 s | 108 s | 60.28689 | +2.1e-15 | yes |
| 60 s | before the review: direct from policy | 60 | 3,400 | 6 | 437 s | 371 s | 69.49897 (bound still active) | -7.8e-10 | yes |
| 60 s | continuation from policy | 10 20 40 60 | 1,391 | 6 | 139 s | 109 s | 69.20988 | -1.6e-15 | yes |
| 60 s | continuation from ocp | 10 20 40 60 | 2,163 | 4 | 236 s | 189 s | **69.51876** | +2.9e-15 | yes |
| 60 s | continuation from policy + polish | 10 20 40 60 80 60 | 2,600 | 8 | 384 s | 306 s | 69.21033 | +2.1e-16 | yes |
| 60 s | best of policy, ocp, polish | 10 20 40 60 10 20 40 60 80 60 | 4,763 | 12 | 621 s | 495 s | **69.51876** | +2.9e-15 | yes |

- **Against the best-known values** (48.0892 at 10 s, 60.2874 at 20 s, 68.6093 at 40 s):
  10 s matches. The 40 s stage of the s-OCP chain reached 68.61011. At 20 s every start
  now lands at 60.28685-60.28689, 5e-4 below the best-known value (worse than the
  1e-4 tolerance; see Local optima).
- **The policy chain at 60 s.** Its 40 s stage converged to 68.29844, a different local
  optimum 0.31 below the s-OCP chain's 68.61011, and the 60 s stage and the polish stayed
  in that basin. The best of the starts is unaffected.
- **Polish.** Here the polish lifted the 60 s policy optimum by only 4.5e-4, still in the
  bad basin. At 20 s it went through a 40 s stage in that basin and came back worse
  (59.96908). It is no longer worth its cost against a second independent start.

## Other 60 s windows (one start)

| Window | First guess | Iter (stages 10 + 20 + 40 + 60 s) | CPU | J* | Replay diff | Survives |
|---|---|---|---|---|---|---|
| Hockenheim, s = 0 | policy | 1,391 (172 + 79 + 1,041 + 99) | 109 s | 69.20988 | -1.6e-15 | yes |
| Hockenheim, s = 0 | s-OCP | 2,163 (105 + 102 + 1,424 + 532) | 189 s | 69.51876 | +2.9e-15 | yes |
| Hockenheim, s = 1,000 m | policy | 1,118 (26 + 128 + 463 + 501) | 106 s | 66.65385 | -4.2e-14 | yes |
| Hockenheim, s = 2,500 m | policy | 1,646 (550 + 408 + 212 + 476) | 106 s | 58.78901 | +2.9e-15 | yes |
| Monza | s-OCP | 830 (29 + 255 + 297 + 249) | 63 s | 70.40997 | -2.6e-14 | yes |
| Spa | s-OCP | 975 (409 + 154 + 113 + 299) | 57 s | 62.90450 | -2.3e-16 | yes |
| Silverstone | s-OCP | 1,217 (781 + 86 + 290 + 60) | 53 s | 65.78710 | -3.0e-15 | yes |
| Budapest | s-OCP | 2,448 (294 + 643 + 544 + 967) | 198 s | 63.88495 | +0.0e+00 | yes |

The circuits' lap s-OCPs are cached (outputs/ocp/; 43-176 s wall each when solved,
`summary.csv`), and one lap s-OCP serves every window on a circuit.

**Machine.** Single-threaded IPOPT 3.14.19 + MUMPS 5.8.2 (the only linear solver in the
CasADi wheel) on an Apple M3 Pro shared with a DR training run and other jobs (load
average 30-180 during these runs, so wall time is 1.2-1.4x CPU time; CPU time is the
budget). Peak RSS 0.8 GB. Compiling the NLP functions to C is a one-off per window length,
cached by source, flags and compiler (`~/.cache/apexzero`, or `$APEXZERO_CACHE`): 1-2 min
for the 60 s window under load, not in the CPU column. Iteration counts move by up to
2-3x with bit-level changes to the guess, so single numbers are indicative.

## Review fixes, and what they changed

The numbers above come from the code after the review of the first version of this work
(pickles, tables and every start's result in `outputs/to_ocp/speed/fixed/`). Earlier
pickles:

- `outputs/to_ocp/speed/*.pkl` and `speed/multi/`: before the review.
- `speed/v2/`: after the review, but before the refinement below. pool:4 and pool:5 fail
  there.

- **The s-domain OCP is part of a level's cost.** Continuation cannot run without it.
  Every production level has its own strip and car, so it needs its own s-OCP. The first
  version loaded cached circuit s-OCPs and left that cost out. It is now solved per level
  and timed (table above), on the reachable part of the strip
  (`window.reachable_strip`). The cut is where the car would get at 1.02x its straight-line
  top speed, plus a 300 m braking margin: 5.1-6.4 km of the 8.2 km strips, depending on
  the car.
- **Refinement (new; the level benchmark found the need).** Before it, pool:4 and pool:5
  ended in Solve_Succeeded, but their replays left the track: J short by 1.1e-3,
  terminated at step 1,128 of 1,200 with the car 1 cm past the corridor. The optimum's
  defects were tiny (<= 1.9e-8, in v_y). But near the limit the trajectory is open-loop
  unstable. The replay drifted from 1e-8 at step 3 to 1e-4 at step 776 and 4 cm at step
  1,128 (growth ~0.28/s). IPOPT's tol = 1e-7 allows such defects. Its default bound
  relaxation (1e-8) also lets states and controls sit a hair outside their bounds.
  - *The fix.* Every converged solve is now re-solved warm from its primal-dual point
    (`REFINE_IPOPT`: mu_init 1e-9, constr_viol_tol 1e-10, no bound relaxation). That takes
    1-5 iterations and leaves J unchanged to 1e-9. All 69 refinements in these runs (every
    converged stage) succeeded. The defects drop to ~1e-12, and pool:4 then replays to 4e-11.
  - *The effect.* Every optimum above replays to 3e-12 or better, against ~1e-9 before.
  - *Why the slip margin stays.* A refined optimum no longer exceeds the slip limit. But
    riding it to the last bit would leave the replay to rounding.
- **Strict success.** `success` now requires all three of:
  - IPOPT's Solve_Succeeded. Solved_To_Acceptable_Level no longer counts: its default
    constraint tolerance is 1e-2, and acceptable_constr_viol_tol is now 1e-7.
  - No window bound active.
  - Every RK stage inside its kappa block.
- **Verified labels.**
  - `replay.verify_window` (solve succeeded, replay survives, J* within 1e-6) and
    `replay.best_verified` (best J* among verified results only) are library code now. The
    driver uses them for every row, so a failed start can no longer win on J.
  - `solve_continuation` never warm-starts from a failed stage. It records each result's
    warm start and the statuses of the chain.
- **Controls inside [-1, 1].** IPOPT 3.14 does not project its result into the original
  bounds by default, so stored controls had |a| = 1 + 1e-8 (the env clips them).
  honor_original_bounds is on, and the refinement does not relax bounds. Max |a| over all
  results above: 1 - 6e-11.
- **Kappa block.**
  - *Coverage.* The 12 coefficients now start one interval behind the node's own, so an RK
    stage that moves backward within a step is looked up exactly. They reach 7 intervals
    ahead; the stages of every optimum above used offsets 0 to 3.
  - *Check.* Every solve recomputes all 20 stage offsets at the optimum and fails if one
    left its block (it would have read kappa = 0).
  - *Strip ends.* Open-strip windows are centred inside the strip, so lookups past either
    end are exact (tested).
- **Window centres** are rate-limited to one control step of travel. A guess that jumps
  by more than 2R + 6 m no longer gives empty bounds (tested with such a guess).
- **Hidden grip.** `solve(hidden_mu=)` and `replay_window(hidden_mu=)` apply a level's
  hidden multiplier as the env does (mu_scale = ctx[4] * hidden_mu). A label replayed
  with the observed context alone would test another car (tested).
- **Compiled libraries.**
  - *Cache key.* It includes the compiler (`$CC` or `cc`, its version banner) and the
    architecture.
  - *Lock.* A file lock makes concurrent workers compile each library once.
  - *-ffp-contract=off.* It stops clang from fusing multiply-adds and costs nothing
    (Hessian 57.6 vs 56.8 us per step). It makes the compiled arithmetic independent of
    the compiler's default; gcc fuses wherever FMA exists. It does not make the library
    bitwise equal to CasADi's VM, whose own matrix kernels fuse: both builds agree with
    the VM to 6e-14.
- **Smaller fixes.**
  - `lam0` without `warm_ipopt` now raises (it was silently ignored).
  - The driver's `--polish` works past the stage schedule.
  - `min_time_guess` warns when a guess runs past an open-strip s-OCP's end.
  - The driver takes `--s0`, `--levels` and `--first policy ocp` (several starts), and it
    saves every start's result.

## What changed, and what each idea was worth

Per-call cost of the NLP derivatives, 10 s window (200 steps), per control step. K is the
window's coefficient count. The first four rows were measured with K = 71 at radius 60 m:
that layout predates the per-step kappa block. The committed layout has K = 81 (block 12
+ 2), measured in the last two rows.

| Variant | Hessian of the Lagrangian | Constraint Jacobian |
|---|---|---|
| Kernel lookup, K = 26 (radius 15 m), CasADi AD in its VM | 1,554 us | 718 us |
| Kernel lookup, K = 71 (radius 60 m), CasADi AD | 3,066 us | 1,467 us |
| Floor-indexed lookup, K = 71, CasADi AD | 773 us | 389 us |
| Floor-indexed lookup, K = 311 (radius 300 m), CasADi AD | 912 us | 567 us |
| + structured derivatives, VM (committed layout, K = 81) | 540 us | 128 us |
| + compiled to C (clang -O2 -ffp-contract=off, K = 81) | **58 us** | **28 us** |

At 60 s an IPOPT iteration costs ~130 ms: Hessian 62, Jacobian 21, f and g 15, and MUMPS
plus IPOPT itself 33 (measured before the review; the refinement adds 1-5 iterations
per solve).

1. **Floor-indexed exact lookup** (kept).
   - *How it works.* i = floor(s/ds), t = s/ds - i, and the four coefficients are picked by
     equality masks. This is the same arithmetic as `eval_jax` (tests < 1e-12 on closed
     tracks and open strips, including eval_jax's clamping at strip ends).
   - *Cost.* Derivatives flow only through t, so the Hessian is 4x cheaper at the same
     window size, and a 300 m window costs only 18% more than a 60 m one. The RK2 stages
     read kappa from a 12-coefficient block picked once per step.
   - *Result.* The 10 s solve went from 144 to 43 CPU-s with the same optimum.
   - *A wide window does not remove re-centring safely.* With a 300 m radius, the policy
     guess (136 m behind the optimum after 10 s, 270 m after 40 s) converged in one round,
     but to worse optima: J* 47.73, 47.34 and 46.96 for three IPOPT settings, against
     48.089. The window bound acts as a trust region. Continuation (3) removes the drift
     instead.
2. **Variable scaling** (implemented as an option, `WindowSolver(scale=...)`, off).
   - *How it works.* s is taken relative to the window centre and the states are divided
     by a scale; the defect rows get the same scale.
   - *On the Hockenheim chain.* One choice (s 10 m, n 1, xi 0.1, vx 10, vy 1, r 0.3,
     delta 0.1, pedal 1) cut CPU from 71 to 46 s. Another (s 100, n 5, vx 50, ...) raised
     it to 224 s.
   - *Over five windows* (measured before the review), the first choice cost more in total
     and never found a better optimum:

   | Window | Unscaled | Scaled | tol 1e-6 |
   |---|---|---|---|
   | Hockenheim, s = 0 | 71 s, 69.51848 | 46 s, 69.51838 | 56 s, 69.51845 |
   | Hockenheim, s = 1,000 m | 50 s, 66.64956 | 90 s, 66.64948 | 75 s, 66.64927 |
   | Hockenheim, s = 2,500 m | 63 s, 58.80731 | 66 s, 58.79743 | 95 s, 58.78831 |
   | Monza | 56 s, 70.41222 | 31 s, 70.41004 | 59 s, 70.41222 |
   | Spa | 35 s, 62.90461 | 164 s, 62.90455 | 34 s, 62.90461 |
   | Total | 275 s | 397 s | 319 s |

3. **Horizon continuation** (kept; the main iteration saving).
   - *Why it works.* Optima of nested windows agree to millimetres except in the last
     second of the shorter window. There is no terminal value, so the 10 s optimum is at
     85 m/s where the 20 s one has braked to 61 m/s.
   - *How it works.* Each stage (10, 20, 40, 60 s) starts from the previous optimum minus
     its last 20 steps, continued by the s-domain minimum-time optimum mapped to time. Only
     the first stage needs re-centring rounds.
   - *Measured.* Continuing with a policy rollout instead took 424/2,864/892 iterations
     (411 CPU-s for the chain). Radius 60 m is the best of those tried for the chain: 40 m
     took 142 CPU-s and 100 m 153 CPU-s (costlier steps, more iterations).
4. **IPOPT options**, on the Hockenheim chain (total CPU and 60 s J*, measured before the
   review):

   | Setting | CPU | 60 s J* |
   |---|---|---|
   | mu_strategy adaptive (kept) | 71 s | 69.51848 |
   | mu_strategy monotone | 113 s | 69.51799 |
   | nlp_scaling_method none | 72 s | 69.51848 (gradient-based scaling was inactive) |
   | stage multipliers passed on, warm_start_init_point, mu_init 1e-3 | 80 s | 69.51822 |
   | same, mu_init 1e-5, bound pushes 1e-8 | 41 s | 69.51799 |
   | tol 1e-6 (replay diff still 6e-11) | 56 s | 69.51845 |
   | acceptable_tol 1e-5, acceptable_iter 5 | 73 s | 69.51848 (never triggered) |
   | mumps_pivot_order AMD / METIS | 153 s / 116 s | 69.51800 / 69.51805 |
   | mumps_pivtol 1e-8 | 91 s | 69.51805 |

   - *MUMPS orderings.* AMD is 15% cheaper per iteration than MUMPS's automatic choice but
     took more iterations. METIS is 3x dearer per iteration.
   - *Warm-started multipliers* halve the cost but stay close to the previous stage and land
     5e-4 lower. They are an option (`warm_multipliers`, `warm_ipopt`), not the default.
   - *tol 1e-6* helped on one window and cost more on three (table in 2).
   - *Bounds on the steering and pedal states.* They are implied by the dynamics, but
     removing them made the direct 20 s solve slower (1,288 vs 849 iterations).
   - *Refinement after convergence* is the one IPOPT setting that changed labels (Review
     fixes).
5. **Other.**
   - *Structured derivatives* (kept, the main per-iteration saving). The step map is a
     composition of 10 RK2 substeps. The Jacobian comes from forward sensitivities, and
     Hess(lam' Phi) from the second-order adjoint sum_k Y_k' Hess(mu_{k+1}' psi) Y_k.
     - It equals CasADi's AD (tests: < 1e-14).
     - Only small per-substep functions are generated, so the C fits in cache; compiled,
       it runs 10x faster than the VM.
     - CasADi's own whole-NLP `jit` wrote 16 MB of C that clang had not compiled after 10
       minutes. Compiling the per-step function whole (190k instructions) gave only 2x.
   - *Slip margin* (a correctness fix). The corridor had a 3 cm margin but the slip limit
     had none, and most 20-60 s optima ride |alpha| = alpha_max.
     - IPOPT left them within ~1e-8 of the bound (relaxed bounds). A 60 s Monza optimum
       replayed 4% short: it terminated, slipped, at step 275.
     - The limit is now tightened by 1e-4 rad (0.06% of alpha_max). On the test case this
       costs 2e-6 of J (relative). A regression test shows the unrefined optimum terminates
       without the margin.

## Local optima

IPOPT finds local optima, and this problem has many. Which one a solve lands in depends
on its path, down to bit-level differences in the guess.

- **Two scales of difference.**
  - *Near-ties.* Many optima lie within ~1e-3 of each other (1e-3 of J is about 1 cm of
    discounted progress): where in the discrete step sequence the line touches an edge,
    or whether a hairpin is taken with an earlier or later apex.
  - *Basins.* Some differ by 0.3 (0.5%), a different line through a whole sequence of
    corners. These are what multi-start is for.
- **20 s values found so far:**
  - before the slip margin: 60.28781 (direct from the s-OCP guess), 60.28738, 60.28729,
    60.28690, 60.28685;
  - after it, before the review: 60.28738 and 60.28775 (polish);
  - after the review: 60.28685-60.28728 from every start, and 59.96908 (a polish through
    the bad 40 s basin).
- **A longer optimum's prefix helps.** Re-solving the 20 s window from the first 380
  steps of a polished 60 s optimum gave 60.28775 in 19 iterations; the 40 s window gave
  68.61011 in 44. That was an ad-hoc test on the code before the refinement, not a
  driver option yet.
- **Bit-level changes alone move the optimum.** Four combinations of `-ffp-contract`
  and honor_original_bounds put the same 20 s chain at 60.28685, 60.28689 or 60.28738.
  Even the import order matters. The driver enables jax_enable_x64 after importing
  `apexzero.env.racing`, whose context-normalisation constants are therefore float32.
  The policy guess then differs in the last bits from a script that enables x64 first,
  and the 10 s solve took 170 instead of 159 iterations.
- **Before vs after the refinement**, one start each, same guesses (v2 vs fixed):
  - higher after: Hockenheim from s = 1,000 m (+8e-5) and 2,500 m (+0.028), Spa,
    Silverstone, Budapest (+1.3e-3), pool:0, pool:3, pool:6;
  - equal to 1e-5: Hockenheim from the s-OCP, pool:1, pool:2, pool:7;
  - lower after: Monza (-2.2e-3) and Hockenheim from the policy (-0.31, the bad basin);
  - no verified optimum before, one after: pool:4. The reverse: pool:5.

The refinement itself does not move J (< 1e-9); it changes later stages' warm starts in
the last bits, which is enough to switch basins. Labels are best-known optima (lower
bounds on regret) either way. The plan's repeat starts on pilot levels measure this
noise; these runs suggest each level needs at least two independent starts. The cheap
polish is not a substitute for an independent start.

## Reproduce

    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 10 20 60 --first policy ocp --polish --out outputs/to_ocp/speed/fixed
    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --s0 1000 2500 --seconds 60 --out outputs/to_ocp/speed/fixed
    uv run python scripts/spike_to_ocp.py --tracks Monza Spa Silverstone Budapest --seconds 60 --first ocp --out outputs/to_ocp/speed/fixed
    uv run python scripts/spike_to_ocp.py --tracks --levels pool:0 pool:1 pool:2 pool:3 --seconds 60 --out outputs/to_ocp/speed/fixed
    uv run python scripts/spike_to_ocp.py --tracks --levels pool:4 pool:5 pool:6 pool:7 --seconds 60 --out outputs/to_ocp/speed/fixed
    uv run python scripts/spike_to_ocp.py --tracks --levels pool:0 --socp-whole --seconds 60 --tag socp-whole --out outputs/to_ocp/speed/fixed

Each run writes a table and the rows (`table-*.md`, `rows-*.pkl`), plus one pickle per
window with the best verified result, every start's final result and their stages.
