# TO-OCP spike (K2a): speed

Training-objective OCP in window mode on the RL environment's exact problem (RK2 x 10 at
5 ms, 20 Hz zero-order hold, corridor and slip limits tightened by 3 cm and 1e-4 rad,
discounted reward). Benchmark: Hockenheim (refined reference), x0 = [0, 0, 0, 0.8 v_QSS(0),
0, 0, 0, 0], first guess = the Hockenheim specialist's rollout from x0. Replay: the optimal
controls applied open-loop in the float64 JAX environment, with its termination checks.

**Result: a 60 s window (1,200 control steps) now solves in 46-232 CPU-seconds (mean 97 s
over seven windows on five circuits); the gate is 12 CPU-minutes.** The CasADi step still
equals the simulator (< 1e-10), and every optimum below replays to its J* within 5e-9
without terminating.

| Window | Method | Stages (s) | Iter | Rounds | Solve wall | CPU | J* | Replay diff | Survives |
|---|---|---|---|---|---|---|---|---|---|
| 10 s | first spike's code, single thread | 10 | 160 | 3 | 196 s | 144 s | 48.08917 | -2.0e-09 | yes |
| 10 s | new solver, direct | 10 | 174 | 3 | 4 s | 3 s | 48.08917 | -2.0e-09 | yes |
| 10 s | **continuation** | 10 | 174 | 3 | 4 s | **3 s** | 48.08917 | -2.0e-09 | yes |
| 10 s | continuation + polish | 10 20 10 | 294 | 5 | 9 s | 7 s | 48.08917 | -2.0e-09 | yes |
| 20 s | first spike's code, single thread | 20 | 1,268 | 4 | 3,795 s | 2,391 s | 60.28690 | -1.1e-09 | yes |
| 20 s | new solver, direct | 20 | 663 | 4 | 31 s | 25 s | 60.28689 | -1.1e-09 | yes |
| 20 s | **continuation** | 10 20 | 261 | 4 | 8 s | **6 s** | 60.28738 | -7.9e-11 | yes |
| 20 s | continuation + polish | 10 20 40 20 | 593 | 6 | 41 s | 31 s | **60.28775** | -6.5e-10 | yes |
| 60 s | new solver, direct | 60 | 3,400 | 6 | 437 s | 371 s | 69.49897 (bound still active) | -7.8e-10 | yes |
| 60 s | **continuation** | 10 20 40 60 | 1,434 | 6 | 165 s | **130 s** | **69.51874** | -2.9e-10 | yes |
| 60 s | continuation + polish | 10 20 40 60 80 60 | 2,128 | 8 | 281 s | 228 s | 69.51874 | -2.9e-10 | yes |

Best-known J* before this work: 48.0892 (10 s), 60.2874 (20 s), 68.6093 (40 s). The
continuation runs match or beat them; the 40 s stage of the 60 s run reached 68.61010. The
first spike's code was not run on 60 s (its 40 s solve took 2,820 iterations and 4,038 s
wall, threaded). "Direct" is one solve from the policy guess with re-centring rounds; at
60 s it stopped after 6 rounds with the window bound still active, at a worse optimum.

Other 60 s windows (continuation, x0 at 0.8 v_QSS; first guess the specialist's rollout on
Hockenheim, the s-domain optimum mapped to time on the other circuits, which have no
policy):

| Window start | Iter | CPU | J* | Replay diff | Survives |
|---|---|---|---|---|---|
| Hockenheim, s = 0 (above) | 1,434 | 130 s | 69.51874 | -2.9e-10 | yes |
| Hockenheim, s = 1,000 m | 1,239 | 81 s | 66.64940 | -5.6e-10 | yes |
| Hockenheim, s = 2,500 m | 928 | 55 s | 58.78827 | -1.1e-09 | yes |
| Monza | 1,164 | 77 s | 70.40998 | -4.4e-09 | yes |
| Spa | 895 | 46 s | 62.90450 | -3.0e-09 | yes |
| Silverstone | 1,435 | 56 s | 65.78779 | -9.8e-10 | yes |
| Budapest | 2,814 | 232 s | 63.87780 | +3.4e-09 | yes |

Single-threaded IPOPT 3.14.19 + MUMPS 5.8.2 (the only linear solver in the CasADi wheel) on
an Apple M3 Pro shared with a DR training run and other jobs (load average 70-170, so wall
time is 1.2-1.6x CPU time; CPU time is the budget). Peak RSS 0.6 GB. Compiling the NLP
functions to C is a one-off per window length, cached by source hash (`~/.cache/apexzero`,
or `$APEXZERO_CACHE`): 1-2 min for the 60 s window under this load, not in the CPU column.
Iteration counts move by up to 2x with bit-level changes to the guess (the 40 s stage took
344 and 566 iterations in two runs whose policy rollouts differed only in the last bits),
so single numbers are indicative; the per-window table is the better guide to a label
budget.

## What changed, and what each idea was worth

Per-call cost of the NLP derivatives, 10 s window (200 steps), per control step:

| Variant | Hessian of the Lagrangian | Constraint Jacobian |
|---|---|---|
| Kernel lookup, K = 26 (radius 15 m), CasADi AD in its VM | 1,554 us | 718 us |
| Kernel lookup, K = 71 (radius 60 m), CasADi AD | 3,066 us | 1,467 us |
| Floor-indexed lookup, K = 71, CasADi AD | 773 us | 389 us |
| Floor-indexed lookup, K = 311 (radius 300 m), CasADi AD | 912 us | 567 us |
| + structured derivatives, VM | 610 us | 147 us |
| + compiled to C (clang -O2) | **66 us** | **34 us** |

At 60 s an IPOPT iteration now costs ~130 ms: Hessian 62, Jacobian 21, f and g 15, MUMPS
and IPOPT itself 33.

1. **Floor-indexed exact lookup** (kept). i = floor(s/ds), t = s/ds - i, and the four
   coefficients picked by equality masks: the same arithmetic as `eval_jax` (tests < 1e-12 on
   closed tracks and open strips, including eval_jax's clamping at strip ends). Derivatives
   flow only through t, so the Hessian is 4x cheaper at the same window size, and a 300 m
   window costs only 18% more than a 60 m one. The RK2 stages read kappa from a
   12-coefficient block picked once per step. 10 s solve: 144 -> 43 CPU-s, same optimum.
   *A wide window does not remove re-centring safely, though.* With a 300 m radius the
   policy guess (136 m behind the optimum after 10 s, 270 m after 40 s) converged in one
   round, but to worse optima (J* 47.73, 47.34 and 46.96 for three IPOPT settings, vs
   48.089): the window bound acts as a trust region. Continuation (3) removes the drift
   instead.
2. **Variable scaling** (implemented as an option, `WindowSolver(scale=...)`, off). s is
   taken relative to the window centre and states divided by a scale; the defect rows get
   the same scale. On the Hockenheim chain one choice (s 10 m, n 1, xi 0.1, vx 10, vy 1, r
   0.3, delta 0.1, pedal 1) cut CPU from 71 to 46 s, another (s 100, n 5, vx 50, ...) raised
   it to 224 s. Over five windows (before the slip margin, see below) the first choice cost
   more in total and never found a better optimum:

   | Window | Unscaled | Scaled | tol 1e-6 |
   |---|---|---|---|
   | Hockenheim, s = 0 | 71 s, 69.51848 | 46 s, 69.51838 | 56 s, 69.51845 |
   | Hockenheim, s = 1,000 m | 50 s, 66.64956 | 90 s, 66.64948 | 75 s, 66.64927 |
   | Hockenheim, s = 2,500 m | 63 s, 58.80731 | 66 s, 58.79743 | 95 s, 58.78831 |
   | Monza | 56 s, 70.41222 | 31 s, 70.41004 | 59 s, 70.41222 |
   | Spa | 35 s, 62.90461 | 164 s, 62.90455 | 34 s, 62.90461 |
   | Total | 275 s | 397 s | 319 s |

3. **Horizon continuation** (kept; the main iteration saving). Optima of nested windows
   agree to millimetres except in the last second of the shorter window (no terminal
   value: the 10 s optimum is at 85 m/s where the 20 s one has braked to 61 m/s). So each
   stage (10, 20, 40, 60 s) starts from the previous optimum minus its last 20 steps,
   continued by the s-domain minimum-time optimum mapped to time. Only the first stage
   needs re-centring rounds, and the 20/40/60 s stages took 61-87, 313-566 and 270-860
   iterations. Continuing with a policy rollout instead took 424/2,864/892 iterations (411
   CPU-s for the chain). Radius 60 m is the best of those tried for the chain: 40 m took
   142 CPU-s, 100 m 153 CPU-s (costlier steps, more iterations).
4. **IPOPT options**, on the Hockenheim chain (total CPU, 60 s J*, before the slip margin):

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

   AMD is 15% cheaper per iteration than MUMPS's automatic choice but took more
   iterations; METIS is 3x dearer per iteration. Warm-started multipliers halve the cost
   but stay close to the previous stage and land 5e-4 lower, so they are an option
   (`warm_multipliers`, `warm_ipopt`), not the default; tol 1e-6 helped on one window and
   cost more on three (table in 2). Removing the redundant bounds on the steering and pedal states
   (implied by the dynamics) made the direct 20 s solve slower (1,288 vs 849 iterations).
5. **Other.**
   - *Structured derivatives* (kept, the main per-iteration saving). The step map is a
     composition of 10 RK2 substeps, so the Jacobian comes from forward sensitivities and
     Hess(lam' Phi) from the second-order adjoint sum_k Y_k' Hess(mu_{k+1}' psi) Y_k. It is
     CasADi's AD (tests: < 1e-14), but only small per-substep functions are generated, so
     the C fits in cache: compiled, 10x faster than the VM. CasADi's own whole-NLP `jit`
     wrote 16 MB of C that clang had not compiled after 10 minutes, and compiling the
     per-step function whole (190k instructions) gave only 2x.
   - *Slip margin* (a correctness fix). The corridor had a 3 cm margin but the slip limit
     none, and most 20-60 s optima ride |alpha| = alpha_max. IPOPT leaves them within ~1e-8
     of the bound, and a 60 s Monza optimum replayed 4% short (terminated, slipped, at step
     275). The Hockenheim optima replayed only because their rounding fell on the safe
     side. The limit is now tightened by 1e-4 rad (0.06% of alpha_max); on the test case
     this costs 2e-6 of J (relative).

## Local optima

IPOPT finds local optima, and this problem has many within ~1e-3 of each other (1e-3 of J is
about 1 cm of discounted progress): where in the discrete step sequence the line touches an
edge, and whether the hairpin after 10 s is taken with an earlier or later apex. Which one a
solve lands in depends on its path, down to bit-level differences in the guess. 20 s
values found during this work (before the slip margin): 60.28781 (direct from the s-OCP
guess), 60.28738, 60.28729, 60.28690, 60.28685; 60 s: 69.51799 to 69.51848 over 15 option
settings. One pair differed by 0.019 (Hockenheim from s = 2,500 m, tol 1e-6 vs 1e-7).
The second start (`--polish`: solve the next longer window, then re-solve the target from
it) is cheap insurance: it lifted the 20 s optimum from 60.28685 to 60.28738 before the
slip margin and from 60.28738 to 60.28775 after it, and left the 60 s optimum unchanged.
Labels are best-known optima either way; the plan's repeat starts on pilot levels measure
this noise.

## Reproduce

    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 10 20 60 --polish
    uv run python scripts/spike_to_ocp.py --tracks Hockenheim --seconds 10 20 60 --method direct

Pickles and tables go to `outputs/to_ocp/speed/` (the other circuits' chains are in
`outputs/to_ocp/speed/multi/`).
