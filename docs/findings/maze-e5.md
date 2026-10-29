# Maze E5: none of PVL, MaxMC or learnability ranks regret better than the best baseline on JaxUED's or SFL's own agents (row C in all nine arm × score cells, with B′ in three and G in one); the plan's MaxMC prediction (S1) holds for no arm (pre-registered)

Results of 29 Oct 2026. Pre-registration: [`docs/prereg/maze-e5.md`](../prereg/maze-e5.md), tagged `maze-e5-prereg-v1` (annotated; tag object `e17a8d2192c2dd95445d1be9d0205d5f8e851f08`, on commit `27564685d74f5a0a74e02af7da799c7f9f846586`, the commit of the run manifest) after training and before any audit of an E5 checkpoint.
- **Training.** All 30 runs trained from the tag `maze-e5-train-v1`, on commit `5edb66f81c7389d06d6b22f14fd3a5356a077255` (the prereg text and E5's code). Run manifest: `outputs/maze-e5/manifest.json`, committed in `2756468` (sha256 `c462803b…`).
- **The analysis.** `outputs/maze-e5/analyse/analysis.json` and `decisions.md`, run once on the Mac at the tag by `scripts/maze_e5.py analyse` (B = 4,000, bootstrap seed 2027, 4 workers, 1,391 s) and committed in `4c31c64` with its never-removed lock.
- **What changed after the tag.** Nothing. `git diff 5edb66f 2756468` on E5's code paths (`CODE_PATHS` in `scripts/maze_e5.py`, which include the prereg text) is empty: no amendment and no deviation between training and the tag.
- **No racing circuit.** E5 uses none. The lockbox and R_TEST are untouched, and nothing under `outputs/sealed/` was read.

Every number below was read from `outputs/maze-e5/analyse/decisions.md`, `outputs/maze-e5/analyse/analysis.json` or `outputs/maze-e5/manifest.json`, or is a run fact recorded by the lead (Section 2), except where marked "computed here". Labels used throughout:

- **Confirmatory, primary**: the 36 tests of the primary family (PF-trend, PF-late), Holm at α = 0.05 over 36, and the outcome rows read from them (prereg Sections 5.1, 5.2).
- **Confirmatory, secondary**: S1-diff, S1-tost, S2 and S3, pre-registered, Holm at α = 0.05 over 28 (prereg Section 5.3).
- **Labels**: the faithful / intermediate / uninformative label per arm, score, target and reached phase (prereg Section 4). They are pre-registered rules. Only the primary scores' late-phase Δ condition uses a Holm decision; every other label uses the unadjusted CI.
- **Descriptive**: everything else (phases reached, solve rates, the other scores, reliability, the comparison with the prereg's stated directions, arm-to-arm differences). These decide nothing. "The lead's reading" marks interpretation that the prereg does not define.

Times are UTC. ρ̄ is a mean Spearman correlation between score and target per arm and phase (prereg Section 4). "abs" is absolute regret J\* − Ĵ^π, "norm" is normalised regret.

## 1. The answer in brief

- **Outcome rows (prereg Section 5.2), per arm and primary score, on both targets:**

  | Arm | PVL | MaxMC | learnability |
  |---|---|---|---|
  | DR | B′, C | B′, C | C |
  | PLR-MaxMC | C | B′, C | C |
  | SFL | C | C | C, G |

  - **Row C holds in all 9 cells.** Every primary score is "uninformative" in every reached phase on both targets. The prereg's reading: "Ground truth confirms No Regrets on the maze."
  - **No score is faithful anywhere.** All 168 labels in `analysis.json` (arm × score × target × reached phase, every score and baseline) are "uninformative"; none is faithful or intermediate. So rows A and A-withheld are empty. Row B is empty too, and so is row F on the ρ̄ scale the pinned code uses (on the mean-z scale it would hold for SFL MaxMC; Section 8, item 8).
  - **Row B′ ("The score tracks regret only once the agent solves most mazes") holds for DR PVL, DR MaxMC and PLR-MaxMC MaxMC.** It co-occurs with row C (the prereg allows this). In each, ρ̄ rises from early to late by +0.58 to +1.28 (CIs exclude 0) and PF-trend is positive and significant after Holm. B′ is untestable for SFL: SFL reached only the late phase.
  - **Row G ("Fresh audits misstate what the curriculum actually sees") holds for SFL learnability.** S2 is significant after Holm with |difference| 0.234 (abs) and 0.241 (norm), both at least 0.2. The sign is the opposite of the plan's prediction: SFL's stored learnability on the buffer ranks regret better (ρ̄ +0.33) than the fresh native learnability (+0.10). PLR-MaxMC's S2 is significant but small (−0.113, −0.119), so no row G for PLR-MaxMC.
- **S1 (the plan's MaxMC prediction) holds for no arm.** In the late phase MaxMC's ρ̄ against normalised regret is +0.69 (DR), +0.61 (PLR-MaxMC) and +0.80 (SFL), not near zero. TOST against ±0.2 fails (p = 1.000) for all three. The difference ρ̄_abs − ρ̄_norm is −0.013, +0.000 and −0.007: no gap between the targets.
- **Primary family (Holm over 36): 26 significant.**
  - **PF-late: 17 of 18 significant, all negative.** In the latest reached phase (late, in every arm) the score's ρ̄ is below the best baseline's. The one non-significant test is SFL MaxMC on absolute regret (Δ −0.033, Holm p 0.108).
  - **PF-trend: 9 of 18 significant.** Positive for DR PVL and MaxMC and PLR-MaxMC MaxMC (both targets). Negative for DR learnability (abs only), SFL learnability (both). Not significant for PLR-MaxMC PVL and learnability, DR learnability (norm), SFL PVL and MaxMC.
- **The best baseline is agent-only in every late cell.** Its late-phase ρ̄ is +0.83 to +0.93 on both targets in all three arms (Section 5.1). Among the primary scores, MaxMC is the highest late, at +0.61 to +0.80, below agent-only's in every arm.
- **View size matters, by less than 0.2 on ρ̄ (S3).** 16 of 18 S3 tests are significant after Holm; the largest ρ̄ difference is +0.191 (SFL PVL, norm), so the pinned code assigns no row F. On the mean-z scale SFL MaxMC's difference is −0.246 (abs) and −0.239 (norm), which would cross 0.2 (computed here; Section 8, item 8).
- **What this means (the lead's reading, not a prereg rule).** For contribution 4: on the maze, with JaxUED's and SFL's own agents and an exact optimum, none of PVL, MaxMC or learnability ranks regret better than the best of the three baselines in any reached phase (agent-only −Ĵ^π in every late cell). The pilot left open "whether the sign changes once the agent solves most mazes". For MaxMC in DR and PLR-MaxMC it does (ρ̄ from −0.5 early to +0.6 to +0.7 late). For PVL it moves up but ends at or below 0 (DR late −0.00, PLR-MaxMC −0.34), and learnability stays low (late +0.07 to +0.23). Why late MaxMC correlates with regret is not tested here.

## 2. The runs

**Design (prereg Section 2).** 3 arms × 2 view sizes × 5 seeds (1–5) = 30 runs, 30,000 updates each: DR (JaxUED `maze_plr.py` with replay off), PLR-MaxMC (the same file at its defaults) and SFL (`minigrid_sfl.py` at its defaults except DEV-E5-2 to DEV-E5-4). Views 13 and 5. Each arm runs at its own hyperparameters, so arm-to-arm differences are descriptive. 16 log-spaced checkpoints per run (updates 300 to 30,000). No PLR-PVL arm (decided 27 Oct 2026).

**The boxes.** Five single-GPU `massedcompute_A6000_plus` boxes ($0.68/h), one seed per box (`SHARD=i/5`), 8 lanes per GPU, as the user approved on 28 Oct 2026 (Section 8 step 5's approval note).
- Two of the first five boxes failed to create at the provider and were replaced (`e5-shard3b`, `e5-shard4b`), so shards 3 and 4 started about 12 minutes after the others.
- Queues started 20:21–20:35 on 28 Oct and ended 15:21–15:34 on 29 Oct. Each box's queue was 18.9–19.2 h.
- The first rate reading matched the lead's fair-share model (SFL-13 6.68 s per update with 6 runs alive; the model's 6.69).

**Result of the queue.** 30 runs, 29 done, 1 retried, 0 missing. Every run has its 16 checkpoints (manifest).
- **The retry.** `maze-e5-sfl-v5-s2`'s first attempt failed after 59.8 min with `CUDA_ERROR_ILLEGAL_ADDRESS` (in `jit_render_state`). The queue re-ran it once from scratch and it finished. Under the prereg a run is missing only if it fails twice.

Mean wall time per run, by cell (computed here from `wall_s` in the manifest; the retried run's figure is its successful attempt alone; 8 runs share a GPU, so these are not single-run speeds):

| Cell | Mean wall (h) | Range (h) |
|---|---|---|
| DR view 5 | 5.77 | 5.70–5.83 |
| DR view 13 | 11.52 | 11.48–11.54 |
| PLR-MaxMC view 5 | 5.33 | 5.30–5.40 |
| PLR-MaxMC view 13 | 10.64 | 10.59–10.74 |
| SFL view 5 | 13.38 | 12.79–13.61 |
| SFL view 13 | 19.02 | 18.88–19.18 |

**The manifest (step 7).** Committed as `2756468`. `round_manifest.py check` passed; every run's command, config and 16 checkpoints were checked against sha256. The step 4 dry run's local outputs (`runs/maze-e5-dryrun/`, git-ignored) were renamed `runs/dryrun-maze-e5/`, contents unchanged, because `round_manifest.py check` refuses a directory with the round's prefix outside the run list.

**The audit (step 9).** Round `maze-e5a` at the tag on the same five boxes, at 1 lane per GPU (the load the calibration timed): 30 of 30 jobs, 0.90–1.06 h per box (the calibration projected about 2.1 h). Each box's audit files (143 per box) were checked against a sha256 list taken on the box before it was deleted.

**GPU time and cost.**

| | GPU-hours |
|---|---|
| Five boxes, all work (queue, audit, set-up) | 104.7 |
| If the provider bills the two failed creates | up to +1.26 |
| Approved estimate (prereg Section 8, step 5) | about 108 |
| Stop rule | 130 |
| Timing calibration (step 5, separate) | 1.74 |

The stop rule was not triggered. About $71–72 for the main boxes (computed here: 104.7 GPU-h × $0.68/h = $71.2; the lead's range is $71–72).

**The pull-back** was 6.7 GB, not the ~2.3 GB Section 9 estimated: the trained runs' checkpoints are larger than the pilot's 4.8 MB.

**The analysis.** `analysis.json` records 30 runs, no missing run, B = 4,000 and bootstrap seed 2027. 258 score × checkpoint × target cells had a constant score, and took ρ = 0 by the constant-score rule (`decisions.md`, last line).

## 3. Which phases the arms reached (descriptive)

A phase is "reached" by an arm when at least 3 of its runs have a checkpoint in it. Phase is set by the share of target episodes that reach the goal on the J\* > 0 mazes (early < 0.5, mid 0.5–0.9, late ≥ 0.9). Counts computed here from `analysis.json` (`describe`).

| Arm | Runs with a checkpoint in early / mid / late | Checkpoints in early / mid / late (of 160) | Solve rate at update 300, view 5 / view 13 (mean over seeds) | At update 30,000, view 5 / view 13 |
|---|---|---|---|---|
| DR | 8 / 8 / 10 | 10 / 16 / 134 | 0.468 / 0.380 | 0.993 / 0.998 |
| PLR-MaxMC | 10 / 10 / 10 | 19 / 19 / 122 | 0.357 / 0.331 | 0.998 / 0.999 |
| SFL | 0 / 0 / 10 | 0 / 0 / 160 | 0.965 / 0.985 | 1.000 / 1.000 |

- DR's early phase has 5 runs at view 13 (7 checkpoints) and 3 at view 5 (3 checkpoints). Its mid phase has 3 at view 13 (3 checkpoints) and 5 at view 5 (13 checkpoints). Z is therefore the mean of the two views' means (prereg Section 4).
- DR and PLR-MaxMC first reach late at update 600–1,000 (DR) and 800–1,400 (PLR-MaxMC). Their early and mid phases rest on 1–4 checkpoints per run; late holds 11–14 of the 16.
- SFL is at 0.95 or more solved from its first checkpoint, so it reached only the late phase. Its B and B′ rows are untestable, and S1 and PF-late read its late phase.
- Consequence for the reading: all three arms' late phase is dominated by agents that solve more than 99% of mazes at the end. Regret there is mostly about speed.

## 4. Primary family (confirmatory; Holm over 36)

Estimates: PF-trend is the mean over the arm's 10 run-level OLS slopes of z on log10(update), over all 16 checkpoints (one-sample t-test, df 9, two-sided; no CI by construction). PF-late is Δ = ρ̄(score) − ρ̄(best baseline) in the latest reached phase (late for every arm), two-sided, with its bootstrap 95% CI. The p values of PF-late, S1, S2 and S3 are bootstrap p values with B = 4,000, whose smallest possible two-sided value is 2/4,001 = 0.0005; `decisions.md` prints these as "<0.001". At the floor, Holm gives 0.014 (= 28 × 0.0005, after the 8 PF-trend tests with a smaller p).

| Test | Arm | Score | Target | Estimate | 95% CI | p | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| PF-trend | DR | PVL | abs | +0.256 | n/a | 1.5e-04 | 0.004 | yes |
| PF-trend | DR | PVL | norm | +0.296 | n/a | 5.8e-05 | 0.002 | yes |
| PF-trend | DR | MaxMC | abs | +0.283 | n/a | 4.2e-06 | 1.4e-04 | yes |
| PF-trend | DR | MaxMC | norm | +0.349 | n/a | 2.0e-07 | 7.2e-06 | yes |
| PF-trend | DR | learnability | abs | −0.088 | n/a | 0.0019 | 0.020 | yes |
| PF-trend | DR | learnability | norm | −0.075 | n/a | 0.0071 | 0.071 | no |
| PF-late | DR | PVL | abs | −0.888 | [−0.934, −0.841] | 0.0005 | 0.014 | yes |
| PF-late | DR | PVL | norm | −0.951 | [−0.990, −0.912] | 0.0005 | 0.014 | yes |
| PF-late | DR | MaxMC | abs | −0.211 | [−0.238, −0.182] | 0.0005 | 0.014 | yes |
| PF-late | DR | MaxMC | norm | −0.249 | [−0.268, −0.231] | 0.0005 | 0.014 | yes |
| PF-late | DR | learnability | abs | −0.655 | [−0.679, −0.633] | 0.0005 | 0.014 | yes |
| PF-late | DR | learnability | norm | −0.706 | [−0.721, −0.694] | 0.0005 | 0.014 | yes |
| PF-trend | PLR-MaxMC | PVL | abs | +0.098 | n/a | 0.032 | 0.191 | no |
| PF-trend | PLR-MaxMC | PVL | norm | +0.126 | n/a | 0.014 | 0.108 | no |
| PF-trend | PLR-MaxMC | MaxMC | abs | +0.366 | n/a | 1.6e-06 | 5.3e-05 | yes |
| PF-trend | PLR-MaxMC | MaxMC | norm | +0.414 | n/a | 2.9e-07 | 1.0e-05 | yes |
| PF-trend | PLR-MaxMC | learnability | abs | −0.059 | n/a | 0.012 | 0.107 | no |
| PF-trend | PLR-MaxMC | learnability | norm | −0.035 | n/a | 0.139 | 0.417 | no |
| PF-late | PLR-MaxMC | PVL | abs | −1.216 | [−1.264, −1.163] | 0.0005 | 0.014 | yes |
| PF-late | PLR-MaxMC | PVL | norm | −1.293 | [−1.331, −1.254] | 0.0005 | 0.014 | yes |
| PF-late | PLR-MaxMC | MaxMC | abs | −0.263 | [−0.291, −0.233] | 0.0005 | 0.014 | yes |
| PF-late | PLR-MaxMC | MaxMC | norm | −0.308 | [−0.327, −0.290] | 0.0005 | 0.014 | yes |
| PF-late | PLR-MaxMC | learnability | abs | −0.650 | [−0.674, −0.626] | 0.0005 | 0.014 | yes |
| PF-late | PLR-MaxMC | learnability | norm | −0.695 | [−0.713, −0.681] | 0.0005 | 0.014 | yes |
| PF-trend | SFL | PVL | abs | −0.015 | n/a | 0.730 | 1.000 | no |
| PF-trend | SFL | PVL | norm | −0.015 | n/a | 0.741 | 1.000 | no |
| PF-trend | SFL | MaxMC | abs | −0.040 | n/a | 0.066 | 0.264 | no |
| PF-trend | SFL | MaxMC | norm | −0.045 | n/a | 0.049 | 0.247 | no |
| PF-trend | SFL | learnability | abs | −0.126 | n/a | 3.0e-05 | 9.5e-04 | yes |
| PF-trend | SFL | learnability | norm | −0.127 | n/a | 3.0e-05 | 9.5e-04 | yes |
| PF-late | SFL | PVL | abs | −0.696 | [−0.741, −0.651] | 0.0005 | 0.014 | yes |
| PF-late | SFL | PVL | norm | −0.763 | [−0.802, −0.725] | 0.0005 | 0.014 | yes |
| PF-late | SFL | MaxMC | abs | −0.033 | [−0.055, −0.008] | 0.013 | 0.108 | no |
| PF-late | SFL | MaxMC | norm | −0.075 | [−0.084, −0.066] | 0.0005 | 0.014 | yes |
| PF-late | SFL | learnability | abs | −0.755 | [−0.778, −0.732] | 0.0005 | 0.014 | yes |
| PF-late | SFL | learnability | norm | −0.803 | [−0.816, −0.793] | 0.0005 | 0.014 | yes |

- **Significant after Holm: 26 of 36** (PF-trend 9, PF-late 17).
- **PF-late (prereg: Δ in the latest reached phase).** Every Δ is negative, and 17 of 18 are significant. No score beats the best baseline late, in any arm, on either target. The smallest gaps are MaxMC's: DR −0.21 (abs) and −0.25 (norm), PLR-MaxMC −0.26 and −0.31, SFL −0.03 (abs, not significant after Holm) and −0.07 (norm).
- **PF-trend (prereg: the seed-slope t-test).**
  - Positive and significant: DR PVL (both targets), DR MaxMC (both), PLR-MaxMC MaxMC (both). Slopes +0.26 to +0.41 per decade of updates.
  - Negative and significant: DR learnability (abs; the norm test is p = 0.007, Holm 0.071) and SFL learnability (both targets, −0.13).
  - SFL PVL and MaxMC trends are near zero (−0.015 to −0.045; Holm 0.25–1.0), as expected for an arm that is late throughout.
- Descriptive: for DR learnability the slope over all 16 checkpoints is negative while ρ̄ rises from early (−0.15) to mid (+0.40) and falls to late (+0.23). It is not monotone, which is why its shift and its trend have opposite signs (Section 5).

## 5. Labels and the outcome rows

### 5.1 Labels for the primary scores and the agent-only baseline

A label is faithful if ρ̄ ≥ 0.5, nRC ≥ 0.6 and Δ > 0 with its CI excluding 0; uninformative if ρ̄ ≤ 0.2, or nRC ≤ 0.2, or Δ's CI includes 0 or lies below it; intermediate otherwise (prereg Section 4). "(Holm)" marks the late-phase labels of the primary scores, whose Δ condition uses PF-late's Holm decision. The label is the same on both targets in every row (checked here). The full table (all scores, baselines, natives) is in `decisions.md`.

| Arm | Score | Phase | ρ̄ abs / norm | nRC abs / norm | Δ abs [95% CI] | Δ norm [95% CI] | Label (both targets) |
|---|---|---|---|---|---|---|---|
| DR | PVL | early | −0.59 / −0.67 | −1.24 / −1.33 | −1.18 [−1.24, −1.13] | −1.38 [−1.42, −1.34] | uninformative |
| DR | PVL | mid | −0.75 / −0.79 | −0.84 / −0.84 | −1.60 [−1.65, −1.55] | −1.68 [−1.72, −1.64] | uninformative |
| DR | PVL | late | −0.00 / −0.02 | −0.00 / −0.00 | −0.89 [−0.93, −0.84] | −0.95 [−0.99, −0.91] | uninformative (Holm) |
| DR | MaxMC | early | −0.51 / −0.59 | −0.50 / −0.63 | −1.11 [−1.16, −1.06] | −1.30 [−1.34, −1.26] | uninformative |
| DR | MaxMC | mid | −0.15 / −0.18 | +0.19 / +0.14 | −1.00 [−1.12, −0.86] | −1.07 [−1.19, −0.93] | uninformative |
| DR | MaxMC | late | +0.67 / +0.69 | +0.58 / +0.58 | −0.21 [−0.24, −0.18] | −0.25 [−0.27, −0.23] | uninformative (Holm) |
| DR | learnability | early | −0.15 / −0.21 | −0.01 / −0.05 | −0.75 [−0.80, −0.70] | −0.91 [−0.99, −0.84] | uninformative |
| DR | learnability | mid | +0.40 / +0.39 | +0.43 / +0.41 | −0.45 [−0.54, −0.36] | −0.50 [−0.59, −0.41] | uninformative |
| DR | learnability | late | +0.23 / +0.23 | +0.28 / +0.28 | −0.66 [−0.68, −0.63] | −0.71 [−0.72, −0.69] | uninformative (Holm) |
| DR | agent-only | early | +0.60 / +0.68 | +0.47 / +0.67 | +0.00 [+0.00, +0.00] | −0.03 [−0.06, +0.00] | uninformative |
| DR | agent-only | mid | +0.85 / +0.89 | +0.77 / +0.88 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | uninformative |
| DR | agent-only | late | +0.88 / +0.93 | +0.88 / +0.93 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | uninformative |
| PLR-MaxMC | PVL | early | −0.53 / −0.63 | −1.30 / −1.35 | −1.07 [−1.14, −1.02] | −1.40 [−1.43, −1.36] | uninformative |
| PLR-MaxMC | PVL | mid | −0.66 / −0.69 | −0.61 / −0.60 | −1.55 [−1.61, −1.48] | −1.61 [−1.66, −1.55] | uninformative |
| PLR-MaxMC | PVL | late | −0.34 / −0.38 | −0.24 / −0.25 | −1.22 [−1.26, −1.16] | −1.29 [−1.33, −1.25] | uninformative (Holm) |
| PLR-MaxMC | MaxMC | early | −0.47 / −0.57 | −0.43 / −0.54 | −1.02 [−1.07, −0.96] | −1.34 [−1.37, −1.31] | uninformative |
| PLR-MaxMC | MaxMC | mid | +0.14 / +0.12 | +0.30 / +0.26 | −0.74 [−0.85, −0.65] | −0.80 [−0.91, −0.70] | uninformative |
| PLR-MaxMC | MaxMC | late | +0.61 / +0.61 | +0.48 / +0.47 | −0.26 [−0.29, −0.23] | −0.31 [−0.33, −0.29] | uninformative (Holm) |
| PLR-MaxMC | learnability | early | −0.18 / −0.25 | +0.06 / −0.00 | −0.72 [−0.76, −0.69] | −1.02 [−1.06, −0.97] | uninformative |
| PLR-MaxMC | learnability | mid | +0.50 / +0.50 | +0.54 / +0.52 | −0.39 [−0.42, −0.35] | −0.43 [−0.46, −0.39] | uninformative |
| PLR-MaxMC | learnability | late | +0.22 / +0.22 | +0.27 / +0.27 | −0.65 [−0.67, −0.63] | −0.69 [−0.71, −0.68] | uninformative (Holm) |
| PLR-MaxMC | agent-only | early | +0.54 / +0.63 | +0.50 / +0.67 | +0.00 [−0.02, +0.00] | −0.14 [−0.16, −0.11] | uninformative |
| PLR-MaxMC | agent-only | mid | +0.89 / +0.92 | +0.80 / +0.90 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | uninformative |
| PLR-MaxMC | agent-only | late | +0.87 / +0.92 | +0.85 / +0.91 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | uninformative |
| SFL | PVL | late | +0.13 / +0.11 | −0.07 / −0.08 | −0.70 [−0.74, −0.65] | −0.76 [−0.80, −0.73] | uninformative (Holm) |
| SFL | MaxMC | late | +0.79 / +0.80 | +0.60 / +0.61 | −0.03 [−0.05, −0.01] | −0.07 [−0.08, −0.07] | uninformative (Holm) |
| SFL | learnability | late | +0.07 / +0.07 | +0.07 / +0.07 | −0.75 [−0.78, −0.73] | −0.80 [−0.82, −0.79] | uninformative (Holm) |
| SFL | agent-only | late | +0.83 / +0.87 | +0.72 / +0.80 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | uninformative |

- **Every label is uninformative, and the Δ condition decides it.** No score has a Δ whose CI is above 0 in any reached phase. The best baseline in every cell is agent-only, except the early phase on normalised regret (DR and PLR-MaxMC), where it is BFS steps (`analysis.json`, `best_baseline`).
- Descriptive (the lead's reading): the Δ condition is what makes the labels uninformative where ρ̄ is high. Late MaxMC has ρ̄ +0.67 / +0.69 (DR, nRC +0.58 on both targets), +0.61 (PLR-MaxMC, nRC +0.48 / +0.47) and +0.79 / +0.80 (SFL, nRC +0.597 on abs and +0.614 on norm). Only SFL MaxMC on norm clears the ρ̄ and nRC bars (0.5 and 0.6), and its Δ is −0.07 [−0.08, −0.07]. On abs its nRC is 0.597, just under 0.6. SFL MaxMC is the closest case to a faithful score among the primary scores.
- The agent-only rows are labelled uninformative by the same rule because Δ is 0 against itself (or negative where BFS steps wins a draw). That is a property of the rule, not a finding about agent-only.

### 5.2 Rows B and B′: the first-to-last shift

B and B′ need ρ̄_last − ρ̄_first with its CI excluding 0 and ±0.1 or more, and a significant PF-trend of the same sign (prereg Section 5.2). SFL has one reached phase, so B and B′ are not testable for it. Early is the first phase for DR and PLR-MaxMC, late the last.

| Arm | Score | Target | ρ̄_last − ρ̄_first (early → late) | 95% CI | PF-trend (mean slope per decade) | Trend significant (Holm) |
|---|---|---|---|---|---|---|
| DR | PVL | abs | +0.581 | [+0.531, +0.632] | +0.256 | yes |
| DR | PVL | norm | +0.652 | [+0.603, +0.704] | +0.296 | yes |
| DR | MaxMC | abs | +1.186 | [+1.157, +1.216] | +0.283 | yes |
| DR | MaxMC | norm | +1.277 | [+1.252, +1.306] | +0.349 | yes |
| DR | learnability | abs | +0.380 | [+0.329, +0.428] | −0.088 | yes |
| DR | learnability | norm | +0.434 | [+0.372, +0.492] | −0.075 | no |
| PLR-MaxMC | PVL | abs | +0.187 | [+0.133, +0.243] | +0.098 | no |
| PLR-MaxMC | PVL | norm | +0.251 | [+0.200, +0.301] | +0.126 | no |
| PLR-MaxMC | MaxMC | abs | +1.083 | [+1.052, +1.114] | +0.366 | yes |
| PLR-MaxMC | MaxMC | norm | +1.175 | [+1.148, +1.202] | +0.414 | yes |
| PLR-MaxMC | learnability | abs | +0.402 | [+0.366, +0.435] | −0.059 | no |
| PLR-MaxMC | learnability | norm | +0.470 | [+0.429, +0.506] | −0.035 | no |

- **B′ holds** for DR PVL, DR MaxMC and PLR-MaxMC MaxMC (shift ≥ +0.1, CI above 0, PF-trend positive and significant, on both targets).
- **B′ does not hold** for PLR-MaxMC PVL: the shift is +0.19 (abs) and +0.25 (norm) with CIs above 0, but PF-trend is not significant after Holm (Holm p 0.191, 0.108).
- **B′ does not hold** for DR learnability and PLR-MaxMC learnability. The shift is positive (+0.38 to +0.47), but PF-trend is negative or not significant.
- **B holds nowhere.** No shift is negative.
- **A did not apply.** No score is faithful in any phase, so the rule that reports A rather than B or B′ for a faithful score (prereg Section 5.2, Section 11 item 11) was not used.
- **Row C and B′ together.** The prereg allows this, and both are reported. The lead's reading: B′'s wording ("tracks regret only once the agent solves most mazes") overstates the case for DR PVL, whose late ρ̄ is −0.00 / −0.02: the shift is a recovery from −0.59 / −0.67 to about 0, not a rise to a positive correlation. For the MaxMC cases the late ρ̄ is +0.61 to +0.69, still below agent-only's (row C).

## 6. Secondary family (confirmatory; Holm over 28)

| Test | Arm | Score | Target | Estimate | 95% CI | p | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| S1-diff | DR | MaxMC | abs − norm | −0.013 | [−0.017, −0.010] | 1.000 | 1.000 | no |
| S1-tost | DR | MaxMC | norm (ρ̄_norm) | +0.686 | [+0.666, +0.703] | 1.000 | 1.000 | no |
| S1-diff | PLR-MaxMC | MaxMC | abs − norm | +0.000 | [−0.003, +0.004] | 0.459 | 1.000 | no |
| S1-tost | PLR-MaxMC | MaxMC | norm (ρ̄_norm) | +0.609 | [+0.591, +0.626] | 1.000 | 1.000 | no |
| S1-diff | SFL | MaxMC | abs − norm | −0.007 | [−0.009, −0.005] | 1.000 | 1.000 | no |
| S1-tost | SFL | MaxMC | norm (ρ̄_norm) | +0.800 | [+0.787, +0.812] | 1.000 | 1.000 | no |
| S2 | PLR-MaxMC | MaxMC | abs | −0.113 | [−0.138, −0.088] | 0.0005 | 0.014 | yes |
| S2 | PLR-MaxMC | MaxMC | norm | −0.119 | [−0.144, −0.095] | 0.0005 | 0.014 | yes |
| S2 | SFL | learnability | abs | +0.234 | [+0.218, +0.254] | 0.0005 | 0.014 | yes |
| S2 | SFL | learnability | norm | +0.241 | [+0.223, +0.261] | 0.0005 | 0.014 | yes |
| S3 | DR | PVL | abs | +0.088 | [+0.037, +0.142] | 0.0015 | 0.016 | yes |
| S3 | DR | PVL | norm | +0.082 | [+0.028, +0.138] | 0.0035 | 0.031 | yes |
| S3 | DR | MaxMC | abs | −0.013 | [−0.055, +0.030] | 0.547 | 1.000 | no |
| S3 | DR | MaxMC | norm | +0.012 | [−0.031, +0.057] | 0.573 | 1.000 | no |
| S3 | DR | learnability | abs | −0.164 | [−0.193, −0.138] | 0.0005 | 0.014 | yes |
| S3 | DR | learnability | norm | −0.167 | [−0.197, −0.139] | 0.0005 | 0.014 | yes |
| S3 | PLR-MaxMC | PVL | abs | +0.085 | [+0.042, +0.127] | 0.0005 | 0.014 | yes |
| S3 | PLR-MaxMC | PVL | norm | +0.070 | [+0.027, +0.112] | 0.0030 | 0.030 | yes |
| S3 | PLR-MaxMC | MaxMC | abs | −0.081 | [−0.116, −0.047] | 0.0005 | 0.014 | yes |
| S3 | PLR-MaxMC | MaxMC | norm | −0.079 | [−0.117, −0.042] | 0.0005 | 0.014 | yes |
| S3 | PLR-MaxMC | learnability | abs | −0.139 | [−0.160, −0.119] | 0.0005 | 0.014 | yes |
| S3 | PLR-MaxMC | learnability | norm | −0.145 | [−0.168, −0.122] | 0.0005 | 0.014 | yes |
| S3 | SFL | PVL | abs | +0.180 | [+0.117, +0.244] | 0.0005 | 0.014 | yes |
| S3 | SFL | PVL | norm | +0.191 | [+0.128, +0.256] | 0.0005 | 0.014 | yes |
| S3 | SFL | MaxMC | abs | −0.092 | [−0.111, −0.073] | 0.0005 | 0.014 | yes |
| S3 | SFL | MaxMC | norm | −0.086 | [−0.105, −0.068] | 0.0005 | 0.014 | yes |
| S3 | SFL | learnability | abs | −0.063 | [−0.080, −0.046] | 0.0005 | 0.014 | yes |
| S3 | SFL | learnability | norm | −0.063 | [−0.081, −0.046] | 0.0005 | 0.014 | yes |

- S1-tost's estimate is ρ̄_norm in the late phase, with its CI (`decisions.md` prints its target as "abs-norm"; see Section 8). S1-diff's estimate is ρ̄_abs − ρ̄_norm.
- S2's estimate is the in-buffer ρ̄ minus the fresh native ρ̄, over all 16 checkpoints (the test is on the run-level mean z; the estimate and the row G threshold use the ρ̄ scale, see Section 8).
- S3's estimate is ρ̄(view 13) − ρ̄(view 5), over all 16 checkpoints.

**S1 (prereg: "high ρ against absolute regret but near zero against normalised"). Holds for no arm.** Both parts must pass. The TOST part fails for every arm (p = 1.000): ρ̄_norm is +0.69, +0.61 and +0.80. The one-sided difference part also fails: for DR and SFL the estimate is negative (p = 1.000), and for PLR-MaxMC it is +0.0003 (p = 0.459). Descriptively, the prediction's first half ("high against absolute") holds and its second half does not: MaxMC ranks normalised regret as well as absolute, within 0.013.

**S2 (in-buffer against fresh).** Significant for all four tests after Holm.
- PLR-MaxMC: the stored MaxMC on the buffer's levels has ρ̄ +0.216 (abs) and +0.204 (norm), against +0.328 and +0.324 for fresh native MaxMC. In-buffer is lower, by 0.113 and 0.119: the plan's direction. The difference is under 0.2, so no row G.
- SFL: the stored learnability on the buffer has ρ̄ +0.333 (abs) and +0.339 (norm), against +0.098 and +0.099 for the fresh native learnability. In-buffer is higher, by 0.234 and 0.241: the opposite of the plan's direction. The difference is 0.2 or more on both targets, so row G holds for SFL's stored score (learnability).
- Descriptive: the in-buffer and fresh scores are measured on different levels (the buffer's 1,000 levels against the 2,000 fresh mazes), so the two ρ̄ values come from different populations. That is how the prereg defines S2; the comparison should not be read as a like-for-like change in the same score.

**S3 (view size, ρ̄(13) − ρ̄(5)).** 16 of 18 significant after Holm. The two non-significant are DR MaxMC (−0.013, +0.012).
- PVL is higher at view 13 in all three arms: +0.07 to +0.19.
- MaxMC is lower at view 13 in PLR-MaxMC (−0.08) and SFL (−0.09), and not significantly different in DR.
- Learnability is lower at view 13 in all three arms: −0.06 (SFL) to −0.17 (DR).
- The largest |difference| on the ρ̄ scale is +0.191 (SFL PVL, norm), just under the 0.2 that row F needs, so `analyse` assigns **no row F** and the prereg's reading, "fidelity depends on observability", is not claimed, although the S3 tests are significant. **This depends on the scale:** on the run-level mean z, SFL MaxMC's S3 difference is −0.246 (abs) and −0.239 (norm) (computed here; both significant after Holm), which would give row F for SFL MaxMC. The code's ρ̄ follows S3's own wording, a difference of ρ̄ (Section 8, item 8).

## 7. The prereg's stated directions (Section 6 of the prereg; descriptive)

These were stated before training and held weakly. The comparison below is the lead's reading of the tests above, not a test.

| Direction | Result |
|---|---|
| X1 early PVL, MaxMC, L1, MaxMC-unit uninformative and below the best baseline, DR and PLR-MaxMC | Held. All 16 early labels (2 arms × 4 scores × 2 targets) are uninformative, and each Δ CI is below 0 (checked here) |
| X2 PVL and MaxMC rise with training on absolute regret (B′) | Held for DR PVL, DR MaxMC and PLR-MaxMC MaxMC. Not for PLR-MaxMC PVL (trend not significant). Untestable for SFL |
| X3 learnability uninformative in every phase | Held (row C for all three arms). Its mid-phase ρ̄ is +0.40 (DR) and +0.50 (PLR-MaxMC), so the label rests on Δ, not on ρ̄ or nRC (mid nRC +0.41 to +0.54) |
| X4 S1 holds | Did not hold, for any arm |
| X5 in-buffer fidelity below fresh | Held for PLR-MaxMC (−0.11, −0.12). Did not hold for SFL (+0.23, +0.24) |
| X6 no direction for view size or arm differences | None stated; Section 6 reports them |
| X7 agent-only and BFS steps are the strongest baselines early | Held. Best baseline early: agent-only on abs, BFS steps on norm. Early ρ̄ on abs: agent-only +0.60 (DR), +0.54 (PLR-MaxMC); BFS steps +0.47, +0.53; J\* −0.44, −0.50 |

## 8. Deviations, ambiguities and notes

1. **No change between the tags.** The training tag and the prereg tag have identical E5 code paths; no amendment was appended; no pipeline crash voided the tag. DEV-E5-1 to DEV-E5-7 are the prereg's own, fixed before training.
2. **A pre-registered descriptive item is not reported.** The prereg lists SFL's score on its own generator's mazes (60 walls, against JaxUED's 25) as descriptive. The audit computed it (`out["own"]` in each SFL audit file, `scripts/maze_e5.py`), but the pinned `analyse` never reads it, so no number for it is given here. The data is in the audit files; a read of it now would be post hoc and labelled so.
3. **The retry** (`maze-e5-sfl-v5-s2`, `CUDA_ERROR_ILLEGAL_ADDRESS` after 59.8 min, re-run once, finished). It affects no outcome: 30 of 30 runs are present. The cause was not investigated beyond the logged error.
4. **Two failed box creations.** Shards 3 and 4 started about 12 minutes late on replacement boxes. Cost only. Up to 1.26 GPU-h more if the provider bills them.
5. **The pull-back was 6.7 GB** against the ~2.3 GB estimate. Cost of time only.
6. **The audit was faster than projected** (0.90–1.06 h per box against about 2.1 h). The reason was not examined. It lowers cost only.
7. **The dry run's directory was renamed** (`runs/maze-e5-dryrun/` to `runs/dryrun-maze-e5/`, contents unchanged), which `round_manifest.py check` required. It touched no E5 output.
8. **Where the prereg's wording left a choice, and what `analyse` did.**
   - **The scale of rows F and G ("|difference| ≥ 0.2").** The estimates in `analysis.json` and the 0.2 threshold (`ROW_EFFECT`, `src/apexzero/maze/e5.py`) are on the ρ̄ (tanh) scale, and the rows above use it. For S3 that is the literal reading: the prereg defines S3 as "ρ̄(13) − ρ̄(5) on the run-level mean z". For S2 the prereg leaves the scale open. Recomputed here on mean z, row G does not change (SFL learnability +0.247 and +0.254; PLR-MaxMC −0.122 and −0.129), but **row F would hold for SFL MaxMC** (S3 −0.246 abs and −0.239 norm, against −0.092 and −0.086 on ρ̄; both significant after Holm). So row F's absence depends on the scale. The analysis is run-once, so this is reported, not changed.
   - **S1-tost's label in `decisions.md`.** Lines 49, 51 and 53 print the target as "abs-norm"; the estimate there is ρ̄_norm (for example +0.686 for DR, which equals DR MaxMC norm late ρ̄ at line 92), and the S1-diff rows (lines 48, 50, 52) are the abs − norm difference. `analysis.json` has no target key for these tests.
   - **Row C and B′ together.** The prereg permits it; the prereg is silent on how to read a B′ whose score stays below the baseline. This note reports both and flags the tension (Section 5.2).
9. **`decisions.md` and `analysis.json` agree.** The 64 test rows and 168 label rows of `decisions.md` match `analysis.json` (estimates, Holm p, significance, ρ̄, labels) by a check run here. The only difference is the S1-tost target label above.
10. **Nothing is exploratory.** The descriptive items (Section 3, Section 7, the reliability line in Section 9) are not tests and decide nothing.

## 9. Descriptive context (decides nothing)

- **Target reliability** (odd/even split of the 32 target episodes, Spearman–Brown to 32; unweighted mean over checkpoints in the phase, computed here from `analysis.json`): DR 0.84 early, 0.96 mid, 0.99 late; PLR-MaxMC 0.78, 0.97, 0.99; SFL 0.98 late. The pilot projected 0.68–0.87 for 32 episodes at low competence.
- **J\* as a baseline** has negative ρ̄ against regret in every reached phase of every arm (−0.44 to −0.77; late −0.53 to −0.67).
- **Natives, L1, MaxMC-unit and MNA** are labelled in `decisions.md`; all are uninformative. In DR late, L1 and MNA have ρ̄ +0.65 to +0.70 and nRC +0.75 to +0.82, above MaxMC's, with Δ −0.18 to −0.26.
- **Arm-to-arm differences** are descriptive (hyperparameters differ): late MaxMC ρ̄ is highest in SFL (+0.79) and lowest in PLR-MaxMC (+0.61). Late PVL is near 0 in DR (−0.00) and SFL (+0.13) and −0.34 in PLR-MaxMC.

## 10. Limits

- **One benchmark, one environment.** The maze has an exact optimum, which is why E5 can say anything about regret. It says nothing about the racing task.
- **Early and mid phases are thin.** They rest on 8–10 runs with 1–4 checkpoints each for DR and PLR-MaxMC (DR's views have 3–5 runs per phase), and do not exist for SFL (late from update 300). B and B′ rest on those phases. The run is the unit: each run's phase value is its mean z over its checkpoints in the phase.
- **The late phase is saturated.** Late holds 122–160 of 160 checkpoints per arm, almost all at solve rate above 0.99. The late ρ̄ is largely the correlation of the score with how fast the agent clears each maze.
- **Arms differ in more than the sampler** (batch size, γ, λ, learning rate, generator walls, run structure). Comparisons between arms are descriptive.
- **Views are pooled** in the primary family; S3 is the moderator. Per arm and view results are not tested.
- **The label is relative to agent-only.** "Uninformative" means "not better than the agent's own return", and the best baseline is the largest of three in each bootstrap draw. A score can track regret well (SFL MaxMC late, ρ̄ about +0.8) and still be labelled uninformative.
- **One audit of one set of mazes.** 2,000 mazes from JaxUED's generator, level seed 2027, serve every run. SFL on its own generator (60 walls) was audited but is not analysed (Section 8, item 2).
- **GPU training is not bit-deterministic** (the prereg's capture check was inconclusive on the GPU); the runs are one realisation of each seed.
