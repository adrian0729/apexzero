# E5 maze audit, pilot: do PVL and MaxMC track exact regret on JaxUED's own agents?

JaxUED's maze example (`third_party/jaxued_examples/maze_plr.py`, vendored unmodified) was
trained twice with its own code and defaults, seed 0, with a 300-update
learning-rate schedule stopped after 200 updates (1.6M env steps;
the default run is 30,000 updates, so this is 0.7% of it):
- **dr**: `--replay_prob 0 --exploratory_grad_updates`: every update trains on fresh generator levels;
- **plr-maxmc**: `--score_function MaxMC`: PLR-perp with replay 0.8, rank prioritisation (beta 0.3),
  staleness 0.3 and a 4,000-level buffer. It takes no gradient step on new levels, and replay starts
  only once the buffer is half full (about 63 updates), so it learns from fewer updates than dr.

Every audited checkpoint plays the same 500 generator mazes (13x13, 25 walls, 5x5 view;
99% have a reachable goal) for 8 episodes each: one AutoReplay
stream per maze with the stochastic policy, step for step as JaxUED's `sample_trajectories_rnn`.
- **Scores** use episodes 1-4: what JaxUED's new-level branch computes on a rollout that
  ends after 4 episodes. PVL and MaxMC come from JaxUED's own `compute_gae` and
  `jaxued.utils` (gamma 0.995, lambda 0.98, time limits terminal). L1, MaxMC-unit,
  learnability p(1-p) (success = goal reached) and MNA use our primitives and the score spec's
  aggregation.
- **Target**: J^pi is the mean discounted return of episodes 5-8, so the scores and
  the target share no sampling noise. J* is the exact BFS optimum (`apexzero.maze.oracle`), and
  regret = J* - J^pi.
- **Baselines**: agent-only (-J^pi on the scored episodes), J* (policy-agnostic), random, and
  oracle-operational (J* - J^pi on the scored episodes), which is a reference, not a ceiling.

Checks on every audited checkpoint:
- our PVL and MaxMC equal JaxUED's to 2.8e-07 (float32 vs float64);
- no episode beat the BFS optimum, so regret >= 0 for every episode, not just on average.
  The 3 mazes without a reachable goal have J* = J^pi = 0; on the rest the smallest
  per-maze regret is 0.004.

## Spearman rho with regret [95% bootstrap CI over levels]

**dr**

| | @50 updates | @100 updates | @150 updates | @200 updates |
|---|---|---|---|---|
| *solve rate* | 12% | 24% | 28% | 30% |
| *mean regret* | 0.853 | 0.801 | 0.775 | 0.758 |
| *regret held by the top 10%* | 0.11 | 0.12 | 0.12 | 0.12 |
| PVL | +0.01 [-0.09, +0.11] | -0.17 [-0.26, -0.07] | -0.27 [-0.36, -0.18] | -0.24 [-0.33, -0.15] |
| MaxMC | -0.09 [-0.19, +0.01] | -0.17 [-0.26, -0.08] | -0.25 [-0.34, -0.16] | -0.21 [-0.30, -0.12] |
| L1 value loss | -0.00 [-0.10, +0.10] | -0.17 [-0.26, -0.08] | -0.24 [-0.33, -0.15] | -0.23 [-0.32, -0.14] |
| MaxMC-unit | +0.02 [-0.08, +0.12] | -0.15 [-0.24, -0.06] | -0.26 [-0.34, -0.17] | -0.21 [-0.30, -0.12] |
| Learnability p(1-p) | -0.02 [-0.12, +0.08] | -0.08 [-0.17, +0.01] | -0.18 [-0.27, -0.09] | -0.08 [-0.16, +0.01] |
| MNA | -0.02 [-0.13, +0.08] | -0.09 [-0.18, -0.00] | -0.11 [-0.20, -0.02] | -0.11 [-0.19, -0.01] |
| Agent-only -J^pi | +0.07 [-0.04, +0.17] | +0.18 [+0.09, +0.27] | +0.29 [+0.20, +0.38] | +0.23 [+0.14, +0.32] |
| J* (policy-agnostic) | +0.18 [+0.06, +0.30] | -0.12 [-0.22, -0.02] | -0.08 [-0.17, +0.02] | -0.22 [-0.31, -0.11] |
| Random | -0.00 [-0.09, +0.08] | -0.01 [-0.09, +0.08] | +0.02 [-0.06, +0.11] | +0.06 [-0.03, +0.14] |
| Oracle-operational | +0.38 [+0.27, +0.48] | +0.22 [+0.13, +0.32] | +0.35 [+0.26, +0.44] | +0.23 [+0.13, +0.33] |

**plr-maxmc**

| | @50 updates | @100 updates | @150 updates | @200 updates |
|---|---|---|---|---|
| *solve rate* | 11% | 12% | 15% | 21% |
| *mean regret* | 0.858 | 0.852 | 0.839 | 0.812 |
| *regret held by the top 10%* | 0.11 | 0.11 | 0.11 | 0.12 |
| PVL | +0.02 [-0.09, +0.12] | +0.01 [-0.09, +0.12] | -0.06 [-0.15, +0.05] | -0.21 [-0.29, -0.12] |
| MaxMC | +0.01 [-0.09, +0.12] | -0.02 [-0.13, +0.09] | -0.12 [-0.22, -0.01] | -0.21 [-0.30, -0.11] |
| L1 value loss | -0.05 [-0.15, +0.05] | -0.12 [-0.21, -0.02] | -0.05 [-0.14, +0.05] | -0.23 [-0.31, -0.14] |
| MaxMC-unit | -0.01 [-0.11, +0.09] | +0.01 [-0.09, +0.11] | -0.05 [-0.14, +0.06] | -0.19 [-0.27, -0.10] |
| Learnability p(1-p) | -0.01 [-0.11, +0.10] | -0.03 [-0.14, +0.08] | -0.01 [-0.10, +0.09] | -0.12 [-0.21, -0.04] |
| MNA | +0.00 [-0.10, +0.11] | -0.05 [-0.16, +0.06] | -0.09 [-0.18, +0.02] | -0.14 [-0.22, -0.05] |
| Agent-only -J^pi | +0.08 [-0.04, +0.19] | +0.08 [-0.04, +0.19] | +0.11 [-0.00, +0.20] | +0.23 [+0.14, +0.32] |
| J* (policy-agnostic) | +0.21 [+0.09, +0.33] | +0.16 [+0.04, +0.28] | +0.06 [-0.05, +0.18] | -0.09 [-0.19, +0.01] |
| Random | +0.00 [-0.09, +0.09] | +0.05 [-0.03, +0.14] | -0.04 [-0.13, +0.05] | -0.02 [-0.11, +0.07] |
| Oracle-operational | +0.45 [+0.35, +0.55] | +0.38 [+0.28, +0.48] | +0.31 [+0.20, +0.41] | +0.33 [+0.23, +0.42] |

## RC@10%: share of total regret in the score's top 50 of 500 levels

**dr**

| | @50 updates | @100 updates | @150 updates | @200 updates |
|---|---|---|---|---|
| PVL | 0.09 [0.08, 0.09] | 0.08 [0.07, 0.09] | 0.07 [0.06, 0.08] | 0.07 [0.06, 0.08] |
| MaxMC | 0.09 [0.08, 0.09] | 0.09 [0.08, 0.09] | 0.09 [0.08, 0.09] | 0.09 [0.08, 0.10] |
| L1 value loss | 0.09 [0.08, 0.09] | 0.08 [0.07, 0.09] | 0.07 [0.06, 0.08] | 0.07 [0.06, 0.08] |
| MaxMC-unit | 0.09 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.08 [0.07, 0.09] | 0.08 [0.07, 0.09] |
| Learnability p(1-p) | 0.09 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.10 [0.09, 0.10] |
| MNA | 0.10 [0.09, 0.10] | 0.09 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.09 [0.09, 0.10] |
| Agent-only -J^pi | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 0.11 [0.10, 0.11] | 0.11 [0.10, 0.11] |
| J* (policy-agnostic) | 0.08 [0.08, 0.09] | 0.08 [0.07, 0.09] | 0.08 [0.07, 0.09] | 0.07 [0.07, 0.08] |
| Random | 0.10 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.10 [0.10, 0.11] | 0.10 [0.10, 0.11] |
| Oracle-operational | 0.10 [0.10, 0.10] | 0.10 [0.09, 0.10] | 0.11 [0.10, 0.11] | 0.10 [0.10, 0.11] |
| Regret itself (ceiling) | 0.11 [0.11, 0.11] | 0.12 [0.12, 0.12] | 0.12 [0.12, 0.12] | 0.12 [0.12, 0.13] |

**plr-maxmc**

| | @50 updates | @100 updates | @150 updates | @200 updates |
|---|---|---|---|---|
| PVL | 0.09 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.09 [0.08, 0.09] | 0.08 [0.07, 0.09] |
| MaxMC | 0.09 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.09 [0.08, 0.10] | 0.09 [0.09, 0.10] |
| L1 value loss | 0.09 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.08 [0.07, 0.09] |
| MaxMC-unit | 0.09 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.08 [0.08, 0.09] | 0.09 [0.08, 0.10] |
| Learnability p(1-p) | 0.09 [0.09, 0.10] | 0.09 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.10 [0.09, 0.10] |
| MNA | 0.10 [0.09, 0.10] | 0.09 [0.09, 0.10] | 0.09 [0.08, 0.09] | 0.09 [0.09, 0.10] |
| Agent-only -J^pi | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.11] |
| J* (policy-agnostic) | 0.08 [0.08, 0.09] | 0.08 [0.07, 0.09] | 0.08 [0.07, 0.09] | 0.08 [0.07, 0.09] |
| Random | 0.10 [0.09, 0.10] | 0.10 [0.09, 0.10] | 0.10 [0.10, 0.10] | 0.10 [0.09, 0.10] |
| Oracle-operational | 0.10 [0.10, 0.11] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.10] | 0.10 [0.10, 0.11] |
| Regret itself (ceiling) | 0.11 [0.11, 0.11] | 0.11 [0.11, 0.11] | 0.11 [0.11, 0.11] | 0.12 [0.11, 0.12] |

*Regret itself (ceiling)* ranks levels by the target regret: no score can capture more.
Ties at the cut-off share the remaining slots, so a constant score captures exactly 0.10.

## Paired difference in rho against the agent-only baseline [95% CI]

| | dr @50 | dr @100 | dr @150 | dr @200 | plr-maxmc @50 | plr-maxmc @100 | plr-maxmc @150 | plr-maxmc @200 |
|---|---|---|---|---|---|---|---|---|
| PVL | -0.05 [-0.26, +0.15] | -0.34 [-0.52, -0.16] | -0.56 [-0.73, -0.39] | -0.47 [-0.65, -0.29] | -0.06 [-0.27, +0.15] | -0.07 [-0.28, +0.16] | -0.16 [-0.35, +0.05] | -0.44 [-0.61, -0.26] |
| MaxMC | -0.15 [-0.36, +0.05] | -0.35 [-0.52, -0.16] | -0.54 [-0.71, -0.36] | -0.44 [-0.62, -0.26] | -0.06 [-0.27, +0.15] | -0.10 [-0.30, +0.13] | -0.22 [-0.41, -0.01] | -0.44 [-0.62, -0.25] |
| L1 value loss | -0.07 [-0.27, +0.14] | -0.35 [-0.52, -0.16] | -0.54 [-0.70, -0.36] | -0.47 [-0.64, -0.28] | -0.13 [-0.33, +0.08] | -0.20 [-0.40, +0.00] | -0.16 [-0.35, +0.04] | -0.46 [-0.63, -0.29] |
| MaxMC-unit | -0.04 [-0.25, +0.16] | -0.33 [-0.51, -0.15] | -0.55 [-0.72, -0.37] | -0.44 [-0.61, -0.25] | -0.09 [-0.28, +0.12] | -0.07 [-0.27, +0.15] | -0.15 [-0.34, +0.06] | -0.42 [-0.59, -0.24] |
| Learnability p(1-p) | -0.09 [-0.29, +0.12] | -0.26 [-0.43, -0.08] | -0.47 [-0.63, -0.31] | -0.31 [-0.46, -0.15] | -0.09 [-0.29, +0.13] | -0.11 [-0.33, +0.11] | -0.12 [-0.30, +0.09] | -0.35 [-0.52, -0.19] |
| MNA | -0.09 [-0.30, +0.12] | -0.27 [-0.44, -0.11] | -0.40 [-0.56, -0.23] | -0.34 [-0.49, -0.17] | -0.07 [-0.28, +0.14] | -0.13 [-0.34, +0.09] | -0.19 [-0.38, +0.02] | -0.37 [-0.53, -0.21] |

## Scores against regret at the last audited checkpoint

![scatter](maze-audit-pilot.png)

Spearman rho across checkpoints (shaded: 95% bootstrap CI):

![rho over training](maze-audit-pilot-rho.png)

## Lambda sensitivity of PVL and L1 (rho)

| | dr @50 | dr @100 | dr @150 | dr @200 | plr-maxmc @50 | plr-maxmc @100 | plr-maxmc @150 | plr-maxmc @200 |
|---|---|---|---|---|---|---|---|---|
| PVL, lambda 0.9 | +0.03 | -0.16 | -0.25 | -0.24 | -0.00 | -0.01 | -0.04 | -0.21 |
| PVL, lambda 0.95 | +0.02 | -0.16 | -0.25 | -0.24 | +0.01 | -0.00 | -0.05 | -0.21 |
| PVL, lambda 0.98 (headline) | +0.01 | -0.17 | -0.27 | -0.24 | +0.02 | +0.01 | -0.06 | -0.21 |
| PVL, lambda 1.0 | -0.05 | -0.18 | -0.29 | -0.23 | +0.01 | -0.08 | -0.07 | -0.23 |
| L1, lambda 0.95 | +0.01 | -0.16 | -0.23 | -0.24 | -0.05 | -0.08 | -0.05 | -0.21 |
| L1, lambda 0.98 (headline) | -0.00 | -0.17 | -0.24 | -0.23 | -0.05 | -0.12 | -0.05 | -0.23 |
| L1, lambda 1.0 | -0.03 | -0.19 | -0.25 | -0.22 | -0.13 | -0.16 | -0.06 | -0.24 |

## Diagnostics: what the scores track

| | dr @50 | dr @100 | dr @150 | dr @200 | plr-maxmc @50 | plr-maxmc @100 | plr-maxmc @150 | plr-maxmc @200 |
|---|---|---|---|---|---|---|---|---|
| rho(PVL, solve rate in episodes 1-4) | +0.82 | +0.89 | +0.90 | +0.88 | +0.79 | +0.80 | +0.85 | +0.90 |
| rho(MaxMC, solve rate in episodes 1-4) | +0.82 | +0.87 | +0.85 | +0.83 | +0.79 | +0.82 | +0.84 | +0.88 |
| rho(L1 value loss, solve rate in episodes 1-4) | +0.79 | +0.87 | +0.79 | +0.82 | +0.75 | +0.62 | +0.81 | +0.75 |
| rho(PVL, regret), mazes solved at least once in 1-4 | -0.24 | -0.23 | -0.27 | -0.26 | -0.33 | -0.27 | -0.26 | -0.31 |
| rho(MaxMC, regret), mazes solved at least once in 1-4 | -0.19 | -0.20 | -0.19 | -0.22 | -0.29 | -0.26 | -0.18 | -0.25 |
| rho(L1 value loss, regret), mazes solved at least once in 1-4 | -0.24 | -0.23 | -0.26 | -0.26 | -0.33 | -0.28 | -0.27 | -0.31 |
| mazes solved at least once in 1-4 | 164 | 300 | 319 | 332 | 145 | 162 | 190 | 276 |
| median / 90th percentile regret | 0.88 / 0.94 | 0.84 / 0.92 | 0.82 / 0.92 | 0.82 / 0.91 | 0.88 / 0.94 | 0.88 / 0.94 | 0.87 / 0.93 | 0.86 / 0.92 |
| reliability of a 4-episode regret estimate | 0.38 | 0.22 | 0.35 | 0.23 | 0.45 | 0.38 | 0.31 | 0.33 |
| projected for 32 episodes (Spearman-Brown) | 0.83 | 0.69 | 0.81 | 0.70 | 0.87 | 0.83 | 0.78 | 0.79 |

The reliability row is the rank correlation of two independent 4-episode estimates
of the same regret (the oracle-operational row above). A score's observable rho with the
target is capped near the square root of it (the plan's sqrt(reliability) ceiling).

## Reading

- **In this phase the value-based scores rank mazes the wrong way round.**
  At the last checkpoint of each run (200 updates; solve rate dr 30%, plr-maxmc 21%), PVL, MaxMC,
  L1 and MaxMC-unit have rho -0.24 to -0.19 with regret, and all six scores are below
  the agent-only baseline -J^pi with every paired CI excluding 0.
  None of the six is positively correlated with regret there; learnability and MNA are
  the least negative.
- **Why: the scores detect success, and success is what lowers regret here.** The agent fails
  most episodes, so most regret is the cost of never reaching the goal. A success surprises the
  critic (a reward after a long run of zeros), so it produces large positive advantages, and
  MaxMC's R_max stays 0 unless an episode succeeded. So PVL, MaxMC and L1 rise with the scored
  episodes' solve rate (rho +0.62 to +0.90 across checkpoints),
  while regret falls with it.
  It is not only the solved versus never-solved split: among mazes solved at least once in
  the scored episodes (checkpoints with 100 or more such mazes), rho with regret is -0.33 to -0.18.
- **RC@10% cannot discriminate yet.** Regret is compressed (median 0.82 to 0.88,
  90th percentile 0.91 to 0.94), because the agent fails most mazes and J*
  is about 0.9 almost everywhere. Ranking by regret itself captures only
  0.11 to 0.12 of it in the top 10%, so the plan's provisional
  'faithful' bar (RC@10% >= 0.6) is out of reach for any score here. RC has to be read against
  its ceiling, and says something only once regret concentrates on a minority of mazes.
- **The target is noisy.** 4 target episodes estimate regret with reliability
  0.22 to 0.45, so even a score equal to the true regret would show a rho of
  only about 0.47 to 0.67 (the square root) against it. The
  full E5 needs about 32 target episodes per maze (projected reliability
  0.69 to 0.87); more mazes do not fix this.
- **Scope.** One seed, under 1% of a default run, and an agent that fails most mazes. The full
  E5 has to show whether the sign changes once the agent solves most mazes, when a solved
  maze's regret is about speed rather than success.

## Compute

- Training: 18.6 min wall, 29.5 CPU-min for the two runs (dr 18.3, plr-maxmc 11.2).
  dr: first launch predates --stop-after: stopped by hand right after checkpoint 3 (200 updates) committed; wall and CPU read from ps at the stop.
- Audits: 3.7 min wall, 3.9 CPU-min for 8 checkpoints (500 mazes x 8 episodes each).
- Total: **33.8 CPU-min**. Every JAX process ran single-threaded XLA at nice 10 on a
  machine shared with other jobs (load average far above the core count), so wall times are
  inflated and CPU times are probably too: niced work lands on the efficiency cores.

## Deviations from JaxUED's scoring, and limits of this pilot

- **Episodes per score.** JaxUED scores whatever completes in one 256-step rollout: one
  episode for an agent that times out, about ten for one that solves fast. We fix it at
  4 for every maze and checkpoint (spec DEV-5), so the scores' noise does not depend
  on the agent's speed. Every counted episode is complete, so JaxUED's -inf 'no episode'
  case cannot occur.
- **MNA** is normalised per step (DEV-11). Its gate 1[R_max > 0] equals 'any success' here,
  because the maze reward is positive exactly when the goal is reached.
- **Fresh-score audit only.** The frozen checkpoint rescores fresh generator levels. The
  scores stored in PLR's buffer (the in-buffer audit) are not analysed yet, although every
  checkpoint carries the buffer.
- **Pilot scale.** One seed, 200 updates, the 5x5 view only. SFL is not run: its maze code is
  not vendored. These numbers size and debug the full E5 (5 seeds x 16 checkpoints x 2,000
  mazes); they are not results.

Reproduce: `uv run python scripts/maze_audit.py` (artefacts in `runs/maze-audit/`).
