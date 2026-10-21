# Width test: training on real-circuit width profiles (21 Oct 2026)

**Status: exploratory, not pre-registered, and in-sample.**
- **Outside the pre-registration.** The test is not part of the `psychophysics-prereg-v2` pre-registration. It was designed after step 1's results were seen.
- **Different code and hardware for the two arms.**
  - The real-width runs used post-tag code (commit `83ebe56`) on a rented GPU whose type was not recorded.
  - The baselines used older code: seed 0 ran on the Mac CPU before the PPO refactor that ruling R18 records, and seeds 1–4 ran on a rented NVIDIA L40S.
- **Possible leakage.** The width profiles used in training come from the same 12 R-dev circuits the agents are evaluated on, so the gain reported below may partly be leakage.
- **Leave-circuits-out test: done** (Section 7). The design and analysis were committed before its runs. About half of the in-sample gain survives on circuits whose widths the agents never saw. The primary test gives p = 0.063, so the evidence is suggestive, not conclusive.

Every number below was read from, or recomputed from, the file named next to it. Files under `runs/` and `outputs/` are local and not in git.

## 1. Summary

The generator gained an opt-in option: training strips take their per-side widths from real R-dev circuits instead of the slowly varying default width model. Four seeds each of v1 and v3 were trained with it at 200M steps and evaluated on the 12 R-dev circuits, against the five-seed baselines of the seed study.

| | v1 | v3 |
|---|---|---|
| Mean circuits completed, out of 12 (baseline, 5 seeds) | 5.4 | 5.8 |
| Mean circuits completed (real widths, 4 seeds) | 8.25 | 8.5 |
| Exact one-sided permutation p | 4/126 = 0.032 | 1/126 = 0.008 |

The leave-out plan ([width-test-loo-plan.md](width-test-loo-plan.md)) and `docs/STATUS.md` round v1's 8.25 to 8.2. It is the same result.

- Sochi went from 0 of 10 baseline seeds to 4 of 8 real-width seeds.
- Norisring and Yas Marina are still completed by no seed.
- The real-width agents are slightly slower on circuits both arms complete: the median gap to the optimal lap rises by 0.3 points (v1) and 0.6 points (v3).
- One v1 seed completes 10 of 12 circuits.

## 2. Motivation: the crash ladders

The psychophysics crash ladders ([psychophysics-step1.md](psychophysics-step1.md), Section 6; descriptive) cut each real crash corner from its circuit and removed its context one factor at a time.

**Constant width removes most of the crashes.** Deterministic completion over all 21 crash states × 4 phases:

| Agent | Real road | Constant width |
|---|---|---|
| G1 | 13/84 | 84/84 |
| G2 | 26/84 | 84/84 |
| G3 | 36/84 | 75/84 |

Straightening the road before or after the corner rarely helped.

**The crashes are on the inside edge where the inside width changes.**
- Of the 160 real-road crashes near the corner, 159 left the road on the inside edge.
- The total width at the crash was typical of the corner.
- The inside width had changed by at least 1 m over the preceding 100 m in 146 of the 160.

**Training almost never shows such changes.** In the default generator, the inside-edge width over 100 m before a tight-corner knot (|κ| ≥ 1/50 m⁻¹) changes by at least 1 m at 0.4–0.5% of knots (300 strips each from the base grammar and from v3's tilted grammar). On the real crash roads it does so at 79% of corner knots.

**The seed study's structural failures.**
- Norisring, Sochi and Yas Marina are completed by 0 of 15 generalists.
- Melbourne is completed by 3 of 15: `dr-pilot-200M-s2`, `dr-pilot-v2-tilt30-s4` and `dr-pilot-v3-tilt50a2-s4` (`runs/dr-pilot-*/eval_rdev.json`).
- The seed study's per-circuit table agrees, with 1, 1 and 1 per variant ([generalist-seeds.md](generalist-seeds.md)). Its prose says "1 of 15", which is wrong.

The test asks a simple question: does showing the agents real width profiles in training remove these failures?

## 3. The generator change

Commit `83ebe56` (`src/apexzero/tracks/generator.py`, `src/apexzero/env/adapters.py`, `scripts/train_dr.py`, `tests/test_real_widths.py`). The change to `adapters.py` is a guard: `generate_bank` refuses a pair of grammars of which only one has real width profiles.

**Default model, unchanged.** Total width is a random R-dev circuit's median width:
- × a lognormal jitter;
- \+ low-frequency variation (wavelengths 300–1,500 m, amplitude 1.5 m);
- clipped to 7–18 m;
- with a 0.5 m low-frequency left-right asymmetry;
- then the inside cap (w_in − 1)·|κ| ≤ 0.6.

**Opt-in real widths** (`train_dr.py --real-widths`, `fit_grammar(..., real_widths=True)`).
- Each strip's (w_left, w_right) is a contiguous segment of one R-dev circuit's per-side width profile, resampled on the generator's 2 m grid.
- The circuit and offset are drawn uniformly, and the segment wraps around the closed circuit.
- The sides are swapped with probability 0.5, scaled by `width_scale` (1.0), and capped by the same inside cap.
- The profiles are measured from `refine_reference`, the geometry every R-dev evaluation drives. On 6 of the 12 circuits that line moves toward hairpin apexes, so one side narrows to about 3 m and the other widens to about 14 m on Norisring.
- Curvature generation is unchanged.
- Lockbox circuits are refused by name.
- A later commit, `2143a59`, added `--real-widths-from` (`fit_grammar(width_tracks=...)`), which restricts the profiles to a subset of the R-dev circuits. The leave-out test uses it. The runs reported here predate it and draw on all 12 circuits.

**Default bit-identical.** With the option off, the generator's outputs are bit-identical to the frozen pre-option copy, for the base, tilted and out-of-distribution grammars (`test_default_is_unchanged`), and the commit message records identical jaxpr and StableHLO. Only training uses the option. Audit level sets and the psychophysics stimuli keep the default model. The psychophysics harness imports this module; [psychophysics-step1.md](psychophysics-step1.md), Section 7.2, logs which of its stages ran on the changed tree.

## 4. Runs

| Arm | Runs | Variant | Seeds |
|---|---|---|---|
| Real widths | `runs/dr-realw-v1-s{0,1,2,3}` | v1 (base grammar) | 0–3 |
| Real widths | `runs/dr-realw-v3-s{0,1,2,3}` | v3 (50% of strips from the (1/r_min)^2 tilt) | 0–3 |
| Baseline | `runs/dr-pilot-200M`, `-s1` … `-s4` | v1 | 0–4 |
| Baseline | `runs/dr-pilot-v3-tilt50a2`, `-s1` … `-s4` | v3 | 0–4 |

- **Configs.** Each real-width run's `config.json` equals the config of the baseline with the same variant and seed (real-width seed k against baseline seed k), except for `real_widths: true` and the name.
  - PPO, environment and every other argument are identical: 2,048 envs, MLP 3 × 256 tanh, `ensemble_heads` 5, γ 0.995, and so on.
  - `dr-pilot-200M`'s config predates the `tilt_fraction` and `tilt_alpha` keys. The real-width v1 runs record 0.0 and 1.0, which are `train_dr.py`'s defaults.
- **Training length.** Every run trained for 199,884,800 steps (`metrics.csv`).
- **Hardware and code.**
  - The run directories do not record the hardware.
  - The real-width runs logged 0.56–0.63M steps/s (`metrics.csv`, `sps`), against 0.42–0.45M for baseline seeds 1–4 on the rented L40S ([generalist-seeds.md](generalist-seeds.md)). So they probably ran on a different GPU or card-sharing arrangement.
  - Baseline seed 0 ran on the Mac CPU (18k and 30k steps/s). Its code predates the golden-tested PPO refactor that ruling R18 records.
- **Evaluation** (`eval_rdev.json`). The default car does 10 timed laps on each of the 12 R-dev circuits, starting at 0.9 × the reference start speed. A circuit counts as completed when more than half the runs finish, as in the seed study. The lockbox circuits are never touched.

## 5. Results

### 5.1 Circuits completed per seed

| | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean ± sd |
|---|---|---|---|---|---|---|
| v1 baseline | 4 | 4 | 8 | 6 | 5 | 5.40 ± 1.67 |
| v1 real widths | 7 | 8 | **10** | 8 | — | 8.25 ± 1.26 |
| v3 baseline | 6 | 5 | 5 | 7 | 6 | 5.80 ± 0.84 |
| v3 real widths | 9 | 8 | 9 | 8 | — | 8.50 ± 0.58 |

- Every v3 real-width seed (8–9 circuits) completes more than every v3 baseline seed (5–7).
- In v1 the lowest real-width count (7, seed 0) beats 4 of the 5 baseline seeds; the best baseline seed completes 8.

### 5.2 Per circuit (seeds completing)

| Circuit | v1 baseline (of 5) | v1 real widths (of 4) | v3 baseline (of 5) | v3 real widths (of 4) | Both variants: baseline → real widths |
|---|---|---|---|---|---|
| Brands Hatch | 3 | 3 | 0 | 3 | 3/10 → 6/8 |
| Hockenheim | 2 | 3 | 3 | 4 | 5/10 → 7/8 |
| Moscow Raceway | 2 | 4 | 3 | 4 | 5/10 → 8/8 |
| Norisring | 0 | 0 | 0 | 0 | 0/10 → 0/8 |
| Nürburgring | 5 | 4 | 4 | 4 | 9/10 → 8/8 |
| Oschersleben | 1 | 3 | 4 | 4 | 5/10 → 7/8 |
| Sepang | 5 | 4 | 5 | 4 | 10/10 → 8/8 |
| Sochi | 0 | 2 | 0 | 2 | 0/10 → 4/8 |
| Yas Marina | 0 | 0 | 0 | 0 | 0/10 → 0/8 |
| Melbourne | 1 | 2 | 1 | 2 | 2/10 → 4/8 |
| Catalunya | 4 | 4 | 5 | 3 | 9/10 → 7/8 |
| Zandvoort | 4 | 4 | 4 | 4 | 8/10 → 8/8 |

- Pooled over both variants, the share of seeds completing rises on 8 circuits, stays the same on 3, and falls on 1.
  - The rises include v3's Brands Hatch (0 of 5 → 3 of 4), and Sochi and Melbourne in both variants.
  - Norisring and Yas Marina stay at 0, and Sepang at 100%. Width is not the whole story for the first two.
  - Catalunya falls slightly (9/10 → 7/8), all of it in v3 (5 of 5 → 3 of 4).
- With 4 or 5 seeds per arm, these per-circuit changes are small samples. Several "rises" come down to a single seed: Nürburgring 9/10 → 8/8 differs by one baseline seed that failed.

### 5.3 Lap time

The gap is the median lap over the optimal lap, minus 1, pooled over completed (seed, circuit) pairs as in the seed study.

| | Baseline | Real widths |
|---|---|---|
| v1 median gap | 7.24% (27 completions) | 7.74% (33) |
| v3 median gap | 7.33% (29) | 7.94% (34) |

- The pooled numbers mix in circuits newly completed, so they compare different sets.
- On circuits both arms complete, the per-circuit median gap rose on 8 of 9 (v1) and 8 of 8 (v3), by a median 0.30 and 0.57 points.
- The real-width agents finish more often but drive slightly slower where both finish.

### 5.4 Permutation test

The test is exact and one-sided (real widths > baseline) on the mean number of circuits completed. It enumerates every split of the pooled seeds into a group of 4 and a group of 5 (126 splits), and p is the fraction with a difference at least as large as the observed one.

| Test | Observed difference | p |
|---|---|---|
| v1 | +2.85 | 4/126 = 0.032 |
| v3 | +2.70 | 1/126 = 0.008 |
| Both, stratified (seeds permuted within each variant; statistic is the sum of the two differences) | +5.55 | 6/15,876 = 0.0004 |
| Sensitivity: baseline seed 0 (Mac CPU) excluded, v1 | +2.50 | 4/70 = 0.057 |
| Sensitivity: baseline seed 0 excluded, v3 | +2.75 | 1/70 = 0.014 |

The test and its one-sided direction were chosen after the result was seen. Read the p-values as a measure of how unusual the seed split is, not as a pre-registered confirmation.

## 6. Caveats, and the next step

**Leakage (the main caveat).**
- The width segments come from the 12 R-dev circuits that the evaluation drives, measured from the same refined reference line.
- Curvature is drawn independently of the widths, so the agent never sees a real corner with its own width profile.
- But it does see the exact width sequences of the evaluation circuits, including those around the crash corners.
- A gain from that is not evidence that the agents learned to handle width changes in general.
- The curvature grammar is also fitted on R-dev: it resamples R-dev corners' angle, minimum radius and length with jitter, but never a circuit's own sequence of corners. The width option reuses the raw per-knot profiles, which is a much more direct form of reuse.

**Other limits.**
- 4 seeds against 5.
- The arms differ in more than the width option:
  - The real-width runs used post-tag code (`83ebe56`) on an unrecorded GPU, at 0.56–0.63M steps/s.
  - The baselines mix the Mac CPU (seed 0, code from before R18's PPO refactor) and a rented L40S (seeds 1–4).
  - Excluding baseline seed 0 weakens v1's test to p = 0.057.
- Per-circuit changes rest on 4–5 seeds per arm (Section 5.2).
- One evaluation suite.
- The test was not pre-registered, and it was designed after step 1's results were seen.

**Leave-circuits-out test (designed and committed before its runs; results in Section 7).**
- **Where it is fixed.** The design is in [width-test-loo-plan.md](width-test-loo-plan.md) (commit `2143a59`, 21 Oct 01:54). The analysis is `scripts/width_loo_analysis.py` (commit `00c1980`, 01:56). The 8 runs started at 02:04, after both commits (`runs/activity.md`). The option it uses, `--real-widths-from`, came with `2143a59`.
- **Folds.** The folds are fixed by alternating through `R_DEV`:
  - fold 1's widths come from `R_DEV[0::2]`: Brands Hatch, Moscow Raceway, Nürburgring, Sepang, Yas Marina, Catalunya;
  - fold 2's come from `R_DEV[1::2]`: Hockenheim, Norisring, Oschersleben, Sochi, Melbourne, Zandvoort.
- **Runs.** The v3 configuration with `--real-widths --real-widths-from <fold>`, 200M steps, seeds 0–3 per fold: 8 runs (`dr-realw-loo-f{1,2}-s{0..3}`). Curvature still comes from all 12 circuits; only the widths are held out.
- **Primary outcome.** Completion on the held-out circuits (the other fold's 6), against the v3 baseline seeds 0–4 on the same circuits. The test is a one-sided exact permutation test, with labels exchanged within each fold only.
- **Secondary outcomes.**
  - The leakage estimate: the in-sample `dr-realw-v3` runs minus the leave-out runs, on the same held-out circuits.
  - Per-circuit completion.
  - The lap-time gap.
- **Reading.** The plan's own reading applies, and this document adds no other rule:
  - a held-out gain over baseline is width robustness;
  - a match with baseline while the in-sample runs beat it is leakage;
  - anything between is reported as the split of the effect.
- **Lockbox.** R_TEST stays sealed. No lockbox circuit is used for widths or evaluation.

A possible follow-up, not part of the plan: if the gain survives, the psychophysics ladders could test whether real-width training removes the inside-edge crashes on the transplanted corners.

## 7. Leave-circuits-out results (21 Oct 2026)

**How it ran.**
- The 8 runs (`runs/dr-realw-loo-f{1,2}-s{0..3}`) trained on one rented NVIDIA RTX A6000, four at a time, at about 160k steps/s each. Each ran 199,884,800 steps, and none failed (`runs/width-loo.log`).
- `scripts/width_loo_analysis.py` was unchanged from its commit before the runs (`00c1980`), and it was run once (`outputs/width-loo/analysis.json`).
- R_TEST was not used.

**Primary outcome: completion on held-out circuits (of 6), per run.**

| Arm | On fold 2's circuits (leave-out: the fold 1 runs) | On fold 1's circuits (leave-out: the fold 2 runs) | Mean |
|---|---|---|---|
| v3 baseline (seeds 0–4, generated widths) | 2, 2, 2, 3, 3 | 4, 3, 3, 4, 3 | 2.90 |
| Leave-out (real widths from the other fold only) | 3, 3, 3, 2 | 6, 5, 3, 4 | 3.62 |
| In-sample (real widths from all 12; `dr-realw-v3`) | 5, 4, 4, 3 | 4, 4, 5, 5 | 4.25 |

- **Primary test.** Leave-out minus baseline is +0.73 circuits. The exact one-sided p is 0.063, from labels exchanged within each fold (15,876 relabellings).
- **Leakage estimate.**
  - In-sample minus leave-out is +0.62.
  - The leave-out runs keep 0.54 of the in-sample gain over baseline without seeing the evaluated circuits' widths.
- **Reading, by the plan's rule.** The result falls between the plan's two clean outcomes, so it is reported as the split of the effect:
  - about half of the in-sample gain is width robustness that transfers to unseen circuits;
  - about half needed the evaluated circuits' own width profiles.
- **Strength of evidence.** The plan set no significance level. At p = 0.063, with 4 seeds per fold, the transferable half is suggestive, not established.

**By fold.** Descriptive, and not in the plan.
- **Fold 1's circuits** (Brands Hatch, Moscow Raceway, Nürburgring, Sepang, Yas Marina, Catalunya), judged with fold 2's widths: baseline 3.4, leave-out 4.5, in-sample 4.5. The whole gain transfers.
- **Fold 2's circuits** (Hockenheim, Norisring, Oschersleben, Sochi, Melbourne, Zandvoort), judged with fold 1's widths: baseline 2.4, leave-out 2.75, in-sample 4.0. Little of the gain transfers.
  - These are the circuits with the study's hardest width cases: Norisring, Sochi and Melbourne.
- **What does not explain it.** The two folds' source profiles vary about equally. The share of 2 m knots where either side's width changes by at least 1 m within the preceding 100 m is 0.26 (fold 1) and 0.27 (fold 2). It is at least 2 m at 0.08 and 0.11 of knots (computed post hoc from the refined profiles).
- So the difference between folds is not simply how much width variation the training saw.

**Per circuit, held out** (share of runs completing it when its widths were not seen): see `analysis.json`, `per_circuit`.
- **Brands Hatch:** 3 of 4 leave-out runs, against 0 of 5 v3 baseline seeds.
- **Yas Marina:** 1 of 4. This is the first time any generalist in this project has completed Yas Marina: 0 of 15 in the seed study and 0 of 8 in the in-sample width runs.
- **Sochi:** 1 of 4 (baseline 0 of 5, in-sample 2 of 4).
- **Melbourne:** 2 of 4 (baseline 1 of 5).
- **Norisring:** 0 of 4.
- **Slight losses** against baseline: Hockenheim 2/4 vs 3/5, Moscow Raceway 2/4 vs 3/5, Oschersleben 3/4 vs 4/5 and Zandvoort 3/4 vs 4/5.

**All 12 circuits, per run** (descriptive; own-fold circuits count, though their widths were seen):

| Run | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|---|
| Fold 1 | 7 | 8 | 8 | 7 |
| Fold 2 | 9 | **11** | 4 | 7 |

- The mean is 7.6, against 5.8 for the v3 baseline and 8.5 for the in-sample runs.
- Fold 2's seed 1 completes 11 of 12. Only Yas Marina is missing. Norisring is completed for the first time, at 4.1% from the optimal lap, though its widths were in that run's training fold.
- Fold 2's seeds range from 4 to 11, so the seed spread is large.

**Lap time.** The median gap on completed held-out circuits is 8.1% for the leave-out runs, against 7.3% for the baseline and 7.9% in-sample. The real-width agents finish more often but drive slightly slower, as in Section 5.3.

**Consequences.**
- Real width profiles in training help on unseen circuits, by about half the in-sample amount, so they are worth keeping. The in-sample number (5.8 → 8.5) overstates the transferable gain.
- Making them the generator's default (`generator-v1`) should wait for one of two things:
  - a larger leave-out run, for example more seeds or three folds;
  - a width model that does not copy any evaluation circuit's profile, such as a statistical model fitted to the profiles.
- Step 2 of the psychophysics study, receptive fields, asks the mechanism directly: how the networks respond to width changes ahead, against the optimum.

