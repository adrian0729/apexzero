# VG-1: the value-gradient term moves the critic's gradient toward λ* on the decided rows; a completion gain is not shown (pre-registered)

Results of 28 Oct 2026. Pre-registration: [`docs/prereg/vg1-value-gradient.md`](../prereg/vg1-value-gradient.md) (sha256 at the tag `eec92d01…`), tagged `vg1-prereg-v1` (annotated; tag object `3a06129e43b2066cafe38db1e3e0ec6f768e28b6`, on commit `6fd82183ba7842a49e09cc1a4bb05ac74ab858fc`, the commit that records β\*) before any main-run network started.
- **The commits.** The first pilot trained from the pilot commit `860939a17709ffb676bcb0b228a5b731fecfc468`. The second pilot (Amendment 1, prereg Section 11) trained from the second pilot commit `495885a2a06536b1358175e5df9cb0dae92dd6df`.
- **The main run.** All 40 networks trained from the tag, clean tree. Run manifest: `outputs/vg1/manifest-main.json`, committed in `3044270`.
- **The analysis.** `outputs/vg1/analysis.json`, run once by `scripts/vg1_analysis.py analyse` and committed in `6939f74`.
- **What changed after the tag.** Neither the tests, the seeds nor the rules changed. One descriptive step, the training snapshots, could not run as tagged and ran through a wrapper. It changes no VG-1 file; it moves one import in memory (Section 8).

Every number below was read from one of these records, or the prereg, except where marked "computed here":
- `outputs/vg1/analysis.json`, `outputs/vg1/manifest-main.json`, `outputs/vg1/pilot.json` and `outputs/vg1/pilot2.json`;
- `runs/vg1-manifest-0of1.json` and `runs/vg1-queue-0of1.log`;
- the M0 records in `outputs/vg1/m0/`, the snapshots in `outputs/vg1/c1/snapshots.json`, and the runs' `metrics.csv` and `eval_rdev.json` (the pilot networks' completion included);
- the box's creation, readiness and deletion times, from the lead's activity log (`runs/activity.md`, not committed). The box's hours, its GPU-hours, VG-1's total, the 6.8% against the projection and the dollar cost are computed here from those times.

Labels used throughout:

- **Primary**: P, decided once by the analysis.
- **Secondary (unadjusted)**: S1–S6, each at 0.05 with no multiplicity adjustment.
- **Descriptive**: the per-component figures, the per-circuit shares, the lap-time gap, the GPU minutes, the snapshots and the pilots. These decide nothing.

Times are UTC.

## 1. The answer in brief

- **P (primary): confirm.** Networks trained with the value-gradient term at β\* = 0.03 (`vgl`) have a higher C1 R²_c than the baseline (`base`), on the frozen costate pipeline's decided rows.
  - Medians −0.294 against −0.683.
  - Hodges–Lehmann shift +0.353 (95% bootstrap CI 0.050 to 0.629).
  - Exact one-sided rank-sum, 20 against 20: W = 495 (expected 410), p = 0.0105.
- **The reading (prereg Section 6, row 2: P and S3 both confirm).** The term moves the critic's gradient toward the window problem's λ on the decided rows (pre-registered, p = 0.0105).
  - S1 says whether that reaches driving, and it shows no difference.
  - S4 confirms for both steer and pedal, so by the prereg's rule "fix the critic's gradient" is **closed as a completion lever at this scale (200M steps, v1)**.
- **S1 (completion, two-sided): no difference shown.** The means are 3.95 (`vgl`) against 4.45 (`base`) circuits of 12.
  - The difference is −0.5 circuits (95% bootstrap CI −1.2 to +0.2), two-sided p = 0.264.
  - The CI leaves out any gain above 0.2 circuits. The prereg's power table gave 0.80 for a 1.5-circuit shift.
- **The term's cost.** S5: the final explained variance is lower with the term (medians 0.921 against 0.939, Hodges–Lehmann −0.018, two-sided p = 6.2e-7).
  - Each `vgl` run took twice the GPU time of a `base` run: 13.3 against 6.6 GPU-minutes per run, or 66 against 33 min of wall time at 5 lanes per GPU (computed here from the per-update figures and the manifest's times).
  - On the 8 circuits both arms complete, `vgl`'s median lap is further from the optimum on every one (Section 7, descriptive).
- **Both arms' R²_c stay negative.** Per component (descriptive; medians, which do not decompose the median of the mean), ξ's median R²_c is −0.61 in `vgl` against −1.33 in `base`, while s is near 0 in both arms (−0.001 and −0.023). The term brings the gradient closer to λ\*; it does not make it a good predictor of λ\* on these rows.
- **E0 does not change** (prereg Section 6 and #5). VG is not in E0's recipe, and any later use of the term is a new study.

## 2. The runs

**Design (prereg Section 4.2).**
- 2 arms × 20 seeds (30–49) = 40 runs, each 199,884,800 steps on v1, with the default PPO and env settings.
- `base` runs with `--vg-coef 0.0`. `vgl` runs with `--vg-coef 0.03` (β\*), `--vg-lambda 0.95` and `--vg-winsor 0.005`.
- Both runs of a seed share a GPU; each GPU runs 5 `base` and 5 `vgl` runs.
- A circuit counts as completed when completion > 0.5 over 10 laps in `eval_rdev.json`.

**The box.** One `massedcompute_A6000_plusx4` (4 × RTX A6000, $2.74/h), created at 04:38:44 and ready at 04:44:53.
- **Setup.** It checked out `vg1-prereg-v1` detached, clean. The dry run passed at 5 lanes per GPU (XLA memory fraction 0.18).
- **The queue** (`scripts/gpu_round.sh vg1 scripts/vg1_runs.py --phase main`) ran from 04:46:21 to 06:26:17: 40 runs, 40 done, 0 retried, 0 failed.
- **Pull-back.** The box's sha256 list (680 files) was taken after the queue, the pull-back verified every file, and the box was deleted at 06:36:08.

**GPU time.**

| | Hours | GPU-hours |
|---|---|---|
| Queue (1.67 h × 4 GPUs) | 1.67 | 6.66 |
| Creation to deletion | 1.96 | 7.83 |
| The cost check's projection for the main run (`outputs/vg1/pilot2.json`) | | 7.33 |
| VG-1 in all: first pilot 1.50 + second pilot 0.98 + main 7.83 | | 10.31 |
| The cost check's limit, and the approved estimate | | 14 (8–14) |
| Stop line | | 16.8 |

The stop rule was not triggered. The main run's box time was 6.8% above its projection (computed here). The box cost about $5.36 (computed here: 1.96 h × $2.74).

**Integrity.**
- Every final and intermediate checkpoint and every `eval_rdev.json` matched the box's sha256 list: 680 of 680. Every final checkpoint is at 199,884,800 steps.
- `round_manifest.py check` passed with nothing tolerated: every run done once at the tag with its arm's arguments, HEAD clean.
- No run failed, so there is no `failures.json`. 0 failed and 0 missing networks in either arm.
- C1's wiring gate passed before any VG-1 network was scored: G1's pullback bit for bit, K3 reproduced, C1 within 4.4·10⁻¹⁶ (tolerance 1e-12). Every network passed K3 at 5 heads.
- M0 ran on all 40 at the tag's code, checked byte for byte before and after each network (40 sidecars).
- R_TEST was never used. Only R-dev data went to the box.

## 3. The pilots (before the tag; exploratory)

The pilot rule (prereg Section 4.1) makes a β eligible if both its runs have finite metrics, EV ≥ 0.90 and M0 below 13.99. β\* is the eligible β with the higher median M1 (C1's R²_c).

| Pilot | β | Seed | EV | M0 | M1 | Completion (of 12) |
|---|---|---|---|---|---|---|
| first | 0.1 | 15 | 0.880 | 0.67 | −0.351 | 3 |
| first | 0.1 | 16 | 0.906 | 0.73 | −0.200 | 4 |
| first | 0.3 | 15 | 0.836 | 0.53 | −0.154 | 2 |
| first | 0.3 | 16 | 0.860 | 0.77 | −0.698 | 2 |
| first | 1.0 | 15 | 0.750 | 0.75 | −0.191 | 4 |
| first | 1.0 | 16 | 0.771 | 0.99 | −0.189 | 3 |
| second | 0.01 | 15 | 0.943 | 0.92 | −0.417 | 2 |
| second | 0.01 | 16 | 0.938 | 0.85 | −0.433 | 5 |
| second | 0.03 | 15 | 0.925 | 0.70 | −0.165 | 4 |
| second | 0.03 | 16 | 0.916 | 0.90 | −0.285 | 3 |

- **The first pilot** (β ∈ {0.1, 0.3, 1.0}; queue 1.25 GPU-hours, box 1.50) had no eligible β: in 5 of 6 runs the EV was below 0.90, and EV fell as β rose. The user chose a second pilot at smaller β (Amendment 1, 28 Oct 2026).
- **The second pilot** (β ∈ {0.01, 0.03}; queue 0.84 GPU-hours, box 0.98) had both βs eligible. β = 0.03 has the higher median M1 (−0.225 against −0.425), so β\* = 0.03. The cost check projected 9.81 GPU-hours (1.50 + 0.98 + 7.33), under 14.
- **Completion** was measured after the tag on the Mac (prereg Section 9, step 5; descriptive). All 10 pilot networks complete 2–5 circuits, close to the main run's two arms.
- M0 in every pilot network is already below 1, while the main run's `base` networks have a median of 13.1 (Section 5). The term fits its own targets at every β tried.

## 4. Primary P: confirm

The test: `vgl` above `base` on each network's C1 R²_c (the mean over s and ξ of the per-component R²_c, frozen pipeline of `costate-prereg-v1`, K2b's decision rows). It is an exact one-sided Wilcoxon rank-sum test, 20 against 20, with mid-ranks for ties.

| Arm | n | Median R²_c | Mean (computed here) | Range (computed here) |
|---|---|---|---|---|
| `base` | 20 | −0.683 | −0.720 | −2.005 to −0.021 |
| `vgl` | 20 | −0.294 | −0.345 | −1.018 to −0.023 |

- W = 495 against an expected 410.
- One-sided p = 0.0105; the opposite one-sided p is 0.990.
- Hodges–Lehmann shift +0.353, 95% percentile bootstrap CI 0.050 to 0.629 (20,000 draws, seed 20261026).
- **Outcome: confirm** (p ≤ 0.05).
- **Power.** The prereg's power line gave 0.78 at a 0.8 SD shift. Measured in `base`'s SD of R²_c (0.52, computed here; `vgl`'s is 0.28), the Hodges–Lehmann shift is about 0.7 SD.

## 5. Secondaries (unadjusted)

| Test | Direction | Medians (`vgl`, `base`) | W (expected 410) | p | Result |
|---|---|---|---|---|---|
| S1 completion | two-sided | 4, 4 (means 3.95, 4.45) | 369.5 | 0.264 | no difference shown; mean difference −0.5 (CI −1.2 to +0.2) |
| S2 Spearman ρ | `vgl` above | 0.313, 0.132 | 584 | 8.4e-8 | confirm |
| S3 M0 | `vgl` below | 0.883, 13.11 | 211 | 1.5e-11 | confirm |
| S4 steer | `vgl` below | 0.882, 6.85 | 211 | 1.5e-11 | confirm |
| S4 pedal | `vgl` below | 0.909, 5.20 | 212 | 2.9e-11 | confirm |
| S5 EV | two-sided | 0.921, 0.939 | 242 | 6.2e-7 | differs; Hodges–Lehmann −0.018 |
| S6 paired R²_c | `vgl` above | mean of 20 differences +0.375 | (2²⁰ sign flips) | 0.0107 | confirm |

- **S3 and S4.** The smallest W possible is 210. Every overlap between the arms involves `base-s48`, whose rollout targets blew up (Section 7), so that its M0 and S4 values are all about 1.000.
  - For M0 and steer, the one overlap is `base-s48` against `vgl-s40`, whose targets also blew up (M0 1.00036 against 1.00040).
  - For pedal (W = 212), the two overlaps are `base-s48` (1.0003) against `vgl-s38` (1.027) and `vgl-s48` (1.0047).
  - The prereg uses a finite statistic as measured, so all stay in.
- **S4's range.** H4R's five v1 baselines had steer 3.4–11.9 and pedal 5.0–7.8. Here `base` has 1.0–13.2 and 1.0–14.7, and `vgl` 0.73–1.00 and 0.60–1.03 (computed here).
- **S6.** No pair was substituted. 14 of the 20 same-seed differences are positive (computed here).

## 6. What counts as the result (prereg Section 6)

P confirms and S3 confirms, so the reading is **row 2**:

> The term moves the critic's gradient toward the window problem's λ on the decided rows (pre-registered, p = …). S1 then says whether that reaches driving. If S1 is null and S4 confirmed for steer or pedal, "fix the critic's gradient" is closed as a completion lever at this scale (200M, v1). […]

(Here p = 0.0105. The omitted last sentence covers the case where S4 did not confirm.)

Here S1 is null and S4 confirmed for both actions. So:
- The actor's one-step signal (S4, the action-projected residual) did change, by a factor of about 6–8 in median.
- The critic's gradient moved toward λ\* (P, S2, S6).
- Completion did not rise (S1).

At 200M steps on v1, making the critic's gradient fit its pathwise targets was not shown to raise completion, and the CI leaves out any gain above 0.2 circuits. By the prereg's rule, that closes the lever at this scale. It does not close it at other scales, grammars or targets (for instance λ\* itself rather than ∇V^π, prereg Sections 3.2–3.3).

**S1 alone** (prereg Section 6) is reported as the completion effect: −0.5 circuits (CI −1.2 to +0.2), at S5's cost of 0.018 in EV and twice the GPU time.

## 7. Descriptive (decides nothing)

**Per component (medians per arm).**

| | `base` | `vgl` |
|---|---|---|
| R²_c, s | −0.023 | −0.001 |
| R²_c, ξ | −1.328 | −0.607 |
| Spearman ρ, s | 0.135 | 0.218 |
| Spearman ρ, ξ | 0.114 | 0.390 |
| M0, s | 5.41 | 0.52 |
| M0, ξ | 3.34 | 0.27 |
| M0, ay (the largest in `base`) | 1,028.6 | 0.89 |
| M0, prev_pedal | 267.9 | 0.89 |
| M0, ax | 245.4 | 0.93 |

In `vgl`, every component's median M0 is below 1.11 (vx is the highest). In `base`, the channels whose targets barely vary dominate, as in H4R's `h4r-v1-base-s10` (prereg Section 5.2).

**The action-projection shares (Section 3.1's form, of the targets).** s's shares are the same in both arms; ξ's steer share is lower in `vgl` (0.158 against 0.193):

| | Steer `base` | Steer `vgl` | Pedal `base` | Pedal `vgl` |
|---|---|---|---|---|
| s | 0.0003 | 0.0003 | 0.0029 | 0.0029 |
| ξ | 0.193 | 0.158 | 0.058 | 0.060 |

So s still carries almost none of either action's projection, as VG-0 measured on H4R's baselines (steer 0.0003, pedal 0.0027–0.0032; ξ 0.181–0.192 and 0.055–0.065). That is consistent with the term moving g_s and g_ξ through the full-state Ω rather than through the actor's one-step channel, but nothing here tests it.

**Per-circuit completion share** (of 20 networks per arm).

| Circuit | `base` | `vgl` |
|---|---|---|
| BrandsHatch | 0.65 | 0.50 |
| Hockenheim | 0.10 | 0.30 |
| MoscowRaceway | 0.45 | 0.05 |
| Norisring | 0.00 | 0.00 |
| Nuerburgring | 1.00 | 0.95 |
| Oschersleben | 0.35 | 0.00 |
| Sepang | 0.90 | 0.90 |
| Sochi | 0.00 | 0.00 |
| YasMarina | 0.05 | 0.00 |
| Melbourne | 0.10 | 0.05 |
| Catalunya | 0.60 | 0.40 |
| Zandvoort | 0.25 | 0.80 |

The totals are close, but the term changes which circuits are completed. `vgl` gains Zandvoort and Hockenheim and loses MoscowRaceway and Oschersleben.

**The lap-time gap** on the circuits both arms complete: each arm's median, over its completing networks, of the gap to the s-OCP lap (IN-4's form, `outputs/ocp/<R-dev>.pkl`).

| Circuit | `base` | `vgl` |
|---|---|---|
| BrandsHatch | 7.7% | 8.7% |
| Hockenheim | 8.3% | 17.8% |
| MoscowRaceway | 8.1% | 17.1% |
| Nuerburgring | 6.9% | 10.5% |
| Sepang | 6.7% | 12.1% |
| Melbourne | 10.3% | 25.7% |
| Catalunya | 6.3% | 10.7% |
| Zandvoort | 7.4% | 11.7% |

`vgl` is slower on all 8. Some medians rest on few networks (Hockenheim's `base` on 2, MoscowRaceway's `vgl` on 1, Melbourne's on 2 and 1).

**GPU minutes per update** (the queue's GPU time shared among a GPU's lanes): `base` 0.0043, `vgl` 0.0087, a ratio of 2.03. Over 1,525 updates that is 6.6 and 13.3 GPU-minutes per run (computed here).

**Training snapshots** (R²_c at each of the 16 checkpoints of seeds 30–33 per arm, 10M to 200M steps; C1-dev's form, the Spearman ρ of R²_c with log timesteps; `outputs/vg1/c1/snapshots.json`, run through the wrapper of Section 8; the wiring gate passed again).
- **Both arms start slightly positive and fall.** At 10M steps every one of the 8 networks has R²_c between 0.04 and 0.14. All 8 C1-dev ρ are negative: `base` −0.93, −0.86, −0.84 and −0.45; `vgl` −0.73, −0.69, −0.97 and −0.92.
- **The early critics are the better aligned.** By R²_c, each critic's gradient is closest to λ\* early in training and drifts away as training goes on. The term does not reverse that drift. In these 4 seeds it leaves `vgl` higher at 200M in two (seeds 30 and 31: −0.11 and −0.29 against −0.94 and −0.62) and lower in two (seeds 32 and 33: −0.41 and −0.58 against −0.18 and −0.02).
- **Four seeds per arm decide nothing**, and the main run's 20 against 20 (Section 4) is the comparison.

**Not pre-registered (computed here from the runs' files).**
- **Final policy entropy** (the last row of `metrics.csv`). `base` ends between −2.87 and −2.51 nats. `vgl` ends between −1.76 and −1.46, except `vgl-s40` at −2.50. The term leaves the policy markedly more stochastic at 200M steps. That fits the slower laps above and the lower EV, though nothing here tests a cause.
- **The two degenerate M0 values.** In `base-s48` and `vgl-s40`, the VGL targets' SD is far above the other networks': on ξ about 23,000× and 600× the largest of the rest, on s about 260× and 30×. On ξ it is 46,647 and 1,205, against 0.40–2.02 in the other 38; on s it is 15.0 and 1.68, against 0.0047–0.057. With targets that large, each normalised residual is about 1 whatever the critic does, and M0 and both S4 values are 1.000.
  - Both end at about −2.5 nats of entropy, where the arms meet: `base-s48` has the highest final entropy of `base` (−2.51), and `vgl-s40` the lowest of `vgl` (−2.50).
  - Their R²_c and ρ are ordinary (`base-s48` −0.821 and 0.138; `vgl-s40` −0.101 and 0.067). The blow-up is in the rollout targets, not in C1.

**Per network.** C is the number of R-dev circuits completed, of 12.

| Seed | `base` R²_c | `vgl` R²_c | `base` ρ | `vgl` ρ | `base` M0 | `vgl` M0 | `base` EV | `vgl` EV | `base` C | `vgl` C |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | −0.941 | −0.112 | 0.180 | 0.218 | 5.51 | 0.78 | 0.944 | 0.919 | 7 | 3 |
| 31 | −0.621 | −0.289 | 0.089 | 0.266 | 19.36 | 0.88 | 0.935 | 0.944 | 5 | 4 |
| 32 | −0.180 | −0.409 | 0.051 | 0.382 | 17.47 | 0.94 | 0.935 | 0.929 | 3 | 3 |
| 33 | −0.021 | −0.584 | 0.027 | 0.408 | 10.60 | 0.70 | 0.940 | 0.922 | 4 | 4 |
| 34 | −0.439 | −0.156 | 0.056 | 0.395 | 6.34 | 0.84 | 0.941 | 0.911 | 6 | 3 |
| 35 | −0.661 | −0.023 | 0.215 | 0.302 | 21.12 | 0.90 | 0.932 | 0.932 | 4 | 5 |
| 36 | −1.435 | −0.459 | 0.203 | 0.253 | 13.94 | 0.89 | 0.944 | 0.925 | 3 | 4 |
| 37 | −0.070 | −0.662 | 0.055 | 0.439 | 16.54 | 0.80 | 0.930 | 0.914 | 3 | 3 |
| 38 | −0.045 | −0.441 | −0.013 | 0.373 | 13.46 | 0.88 | 0.934 | 0.917 | 4 | 6 |
| 39 | −1.213 | −0.105 | 0.153 | 0.328 | 16.29 | 0.89 | 0.921 | 0.927 | 3 | 4 |
| 40 | −2.005 | −0.101 | 0.241 | 0.067 | 10.16 | 1.00 | 0.945 | 0.932 | 5 | 3 |
| 41 | −0.830 | −0.445 | 0.171 | 0.307 | 12.76 | 0.91 | 0.935 | 0.927 | 5 | 4 |
| 42 | −0.315 | −0.093 | 0.125 | 0.332 | 14.77 | 0.70 | 0.934 | 0.914 | 6 | 4 |
| 43 | −0.516 | −0.360 | 0.107 | 0.331 | 10.45 | 0.76 | 0.944 | 0.902 | 4 | 5 |
| 44 | −1.079 | −0.281 | 0.156 | 0.320 | 8.53 | 0.91 | 0.945 | 0.907 | 5 | 3 |
| 45 | −0.856 | −0.135 | 0.152 | 0.244 | 9.40 | 0.90 | 0.949 | 0.929 | 5 | 3 |
| 46 | −1.367 | −0.300 | 0.204 | 0.360 | 11.59 | 0.86 | 0.937 | 0.908 | 4 | 5 |
| 47 | −0.273 | −0.900 | 0.079 | 0.284 | 17.86 | 0.83 | 0.950 | 0.914 | 7 | 3 |
| 48 | −0.821 | −1.018 | 0.138 | 0.276 | 1.00 | 0.94 | 0.938 | 0.934 | 3 | 5 |
| 49 | −0.705 | −0.027 | 0.061 | 0.124 | 14.20 | 0.63 | 0.940 | 0.919 | 3 | 5 |

## 8. Deviations and notes

- **The snapshots ran through a wrapper (28 Oct 2026; descriptive only).**
  - **The defect.** At the tag, `vg1_c1.py snapshots` cannot run. `cmd_snapshots` imports `apexzero.costate.analysis` before `start()` puts the frozen worktree's `src/` on `sys.path`. On main that import fails (`apexzero` is not importable: the venv's editable install is not on `sys.path`), and with the worktree's `src/` on `PYTHONPATH` `enter_worktree` refuses because `apexzero` is already loaded. Every other subcommand imports inside helpers that run after `start()`. The tests never ran `cmd_snapshots` end to end.
  - **The wrapper.** `scripts/vg1_snapshots.py` (not one of `vg1_analysis.CODE_PATHS`) takes `cmd_snapshots`' own source and moves that one import to just after `start()`. It checks that nothing else differs, redefines the function in `vg1_c1`'s namespace in memory, and runs `vg1_c1.main(["snapshots"])`, with the same pin, freeze, import and tag checks, wiring gate and write-once output.
  - **What it did not touch.** No file of VG-1's code changed, and the tag gate passed. It ran from 07:03 to 07:18, after the analysis had been run and committed (07:02), so it could not affect P or S1–S6.
  - **What the record shows.** `snapshots.json` does not show that the wrapper ran: its `provenance.argv` reads `scripts/vg1_c1.py`, because `vg1_c1`'s `safe_argv` writes that name whatever the entry point, and it holds no hash of the wrapper. The wrapper is committed with this document, and the run's log is `runs/vg1-snapshots.log` (not committed).
- **The analysis's lock** (`outputs/vg1/analysis.lock`) records local time, 03:01:53 (UTC−4); that is 07:01:53 UTC.
- **Before the tag.**
  - Amendment 1 (prereg Section 11) added the second pilot after the first had no eligible β. It was written and committed before the second pilot trained, and states what had been seen.
  - The second pilot's M0 first stopped at `os.nice(0)` (a `PermissionError`, started in the background), writing nothing. A foreground rerun succeeded. The main run's M0 ran in the background at niceness 15 without the error.
- **Idle box time.** None to note: the main run's box was pulled back and deleted within 10 minutes of the queue's end.
- **Lane A's branch** (`costate-exploratory`) was not merged while VG-1 ran, so VG-1's code paths on main stayed the tag's throughout.
