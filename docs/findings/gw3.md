# GW-3: a width model fitted on one fold raises completion on the other fold; GW-4's rule falls back to copied widths (pre-registered)

Results of 28 Oct 2026. Pre-registration: [`docs/prereg/gw3-width-round.md`](../prereg/gw3-width-round.md) (sha256 `86436a07…`), tagged `gw3-prereg-v1` (annotated; tag object `0569b467bd324bebdc6076b36423b4efa7d85e68`, on commit `860939a17709ffb676bcb0b228a5b731fecfc468`) before any GW-3 run started. All 120 runs trained from that commit, clean tree. Run manifest: `outputs/gw3/manifest.json`, committed in `8839db3`. Analysis: `outputs/gw3/analysis.json`, run once by `scripts/gw3_analysis.py` and committed in `ab252fd`. Neither the tests, the seeds nor the rules changed after the tag.

Every number below was read from `outputs/gw3/analysis.json`, `outputs/gw3/manifest.json`, `runs/gw3-manifest-0of1.json`, `runs/gw3-queue-0of1.log` or the prereg, except where marked "computed here". Labels used throughout:

- **Primary**: P1, decided once by the analysis.
- **Secondary (unadjusted)**: S1, S1-v1, S1-v3 and S2, exact one-sided tests at 0.05, no multiplicity adjustment.
- **Descriptive**: S3, the lap-time extras, the count tables. These decide nothing, except that GW-4's criteria 2 and 3 read S3's D2 and the pooled lap-time gap as point estimates.

Times are UTC.

## 1. The answer in brief

- **P1 (primary): confirm.** Networks trained with a width model fitted on one fold's circuits, which copies no evaluation circuit, completed more of the other fold's circuits than the baseline did. Means per network pair, out of 12: v1 9.8 against 4.6, v3 9.9 against 5.0. One-sided exact p = 1 / 34,134,779,536 = 2.9e-11, the smallest value possible.
- **Secondaries (unadjusted).** S1, S1-v1, S1-v3 and S2 all confirm. S1-v1 (copied widths above baseline on held-out circuits, v1 alone, #12's transfer condition): 7.4 against 4.6, p 20 / 184,756 = 1.1e-4.
- **GW-4's decision: copied R-dev widths (`--real-widths`), by the fallback rule #12.** Criteria 1, 2 and 4 hold. **Criterion 3 fails**: the lap-time gap Δgap is +1.52 points, against a limit of below 1.0. So the adoption rule fails and #12 applies: S1-v1 confirms, so copied widths are used. Criterion 5 is read by GW-4 at the `generator-v1` commit, not here. The analysis records the same decision (copied) whether it holds or fails.
- **What this means for E0.** E0 (v1) trains on copied R-dev width profiles. Every R-dev number for E0 is therefore labelled **"in-sample for widths"** wherever it is reported. The analysis's `gw4.in_sample_label_covers` lists what the label covers: the 80% gate's result and each seed's count; per-circuit R-dev completion, lap times and gaps to the s-OCP lap; and every comparison or figure that uses E0's R-dev results. R_TEST is unaffected.

## 2. The runs

**Design (prereg Section 2).** 6 arms × 2 variants × 10 seeds (20–29) = 120 runs, 200M steps each, the default PPO and env settings. Arms: `base` (default widths); `cl-f1`, `cl-f2` (copied widths from one fold's 6 circuits); `sl-f1`, `sl-f2` (stat model (2a) fitted on one fold's 6 circuits); `sa` (stat model fitted on all 12). Variants: v1 (the base grammar) and v3 (`--tilt-fraction 0.5 --tilt-alpha 2`). Folds: f1 = `R_DEV[0::2]`, f2 = `R_DEV[1::2]`. A circuit counts as completed when completion > 0.5 over 10 laps from `eval_rdev.json`.

**The box.** One `massedcompute_A6000_plusx4` (4 × RTX A6000, $2.74/h), created about 23:30 on 27 Oct and booted at 23:32:54. The queue (`scripts/gpu_round.sh gw3 scripts/gw3_runs.py`) started its first runs at 23:38:44, 4 GPUs × 4 lanes (16 at a time, XLA memory fraction 0.23), at about 154–160k steps/s per run (the lead's reading of the first runs; not re-derived here). Each run took about 26–27 min (the first v1 base run: 23:38:44 to 00:05:13). The queue ended at 02:51:03 on 28 Oct, its manifest written then: 120 runs, 119 done, 1 retried, 0 failed. `round_manifest.py check` reported "queue 3.21 h × 4 GPUs = 12.8 GPU-h (idle included; setup and pull-back not)". The box's sha256 list was written at 03:02:41, the pull-back finished at 03:10:21 and the box was deleted at 03:10:38.

**GPU time.**

| | Hours | GPU-hours |
|---|---|---|
| Queue (3.21 h × 4 GPUs) | 3.21 | 12.8 |
| Creation to pull-back | about 3.67 | about 14.7 |
| Approved estimate (prereg Section 7) | | about 14 (13.8–15.2 for this box) |
| Stop line (estimate + 20%) | | 16.8 |

The stop rule was not triggered. About 19 min × 4 GPUs (about 1.3 GPU-hours) of the 14.7 was idle, between the queue's end at 02:51:03 and the pull-back at 03:10:21, while the lead waited for the user's decision about another round. The box billed through that wait. The cost at $2.74/h is about $10 (computed here: 3.67 h × $2.74).

**The retry.** `gw3-v3-base-s26`'s first attempt failed at start-up (queue log, 01:34:04) with a JAX CUDA check failure in `gpuMemcpyAsync` (`py_client_gpu.cc:124`). Its log is kept as `runs/gw3-v3-base-s26.log.1`. The queue re-ran it at once in the same lane, from scratch with the same arguments, and it finished. Under the prereg a run is missing only if it fails twice. **120 of 120 runs are present, 0 missing.**

**Integrity.**
- Every config, `eval_rdev.json` and final checkpoint (199,884,800 steps) matched the box's sha256 list: 360 of 360.
- `round_manifest.py check --tag gw3-prereg-v1` returned 0 (every run done once at the tag with the right arguments, HEAD clean). The manifest stage checked each run's config against its cell's flags, `train_dr.py`'s defaults at the tag and H4R's PPO and env blocks.
- R_TEST was never used. Only R-dev data went to the box.

## 3. Primary P1: confirm

The test: `sl` above `base` on held-out circuits, both variants. The statistic is T = Σ over variants of [mean u_s(sl) − mean b_s]. A leave-out unit u_s pairs seed s of `sl-f1` and `sl-f2`, so each R-dev circuit is judged once, by a network that never saw its widths. A baseline unit b_s is the same seed's `base` network on all 12. Exact one-sided permutation test, labels exchanged within each variant; 10 units per arm and variant.

| Variant | n (sl, base) | Mean sl | Mean base | Difference |
|---|---|---|---|---|
| v1 | 10, 10 | 9.8 | 4.6 | +5.2 |
| v3 | 10, 10 | 9.9 | 5.0 | +4.9 |

- T = 10.1.
- p = count_ge / relabellings = 1 / 34,134,779,536 = 2.93e-11 (the observed labelling is the only one with T* ≥ T). The opposite one-sided p is 1.0.
- **Outcome: confirm** (p ≤ 0.05). In wording from prereg Section 4: a width model fitted on one fold's circuits, which copies none, raises completion on the other fold's circuits (pre-registered, one-sided p = 2.9e-11).
- The prereg's power table gave about 0.96 for P1 if the whole in-sample v1 gain transferred (+2.6 v1, +0.6 v3). The observed differences are larger than the gains that table used.

## 4. Secondaries (unadjusted)

All are exact one-sided permutation tests of the same construction at 0.05, unadjusted. T is the sum over variants of the mean difference, or the one variant's mean difference for S1-v1 and S1-v3.

| Test | Question | Mean treated, base (v1) | Mean treated, base (v3) | T | p (count_ge / relabellings) | Outcome |
|---|---|---|---|---|---|---|
| S1 | `cl` above `base`, held-out, both variants | 7.4, 4.6 | 7.6, 5.0 | 5.4 | 4.3e-7 (14,650 / 34,134,779,536) | confirm |
| S1-v1 | `cl` above `base`, held-out, v1 | 7.4, 4.6 | | 2.8 | 1.08e-4 (20 / 184,756) | confirm |
| S1-v3 | `cl` above `base`, held-out, v3 | | 7.6, 5.0 | 2.6 | 1.63e-3 (302 / 184,756) | confirm |
| S2 | `sa` above `base`, all 12, both variants | 9.6, 4.6 | 9.3, 5.0 | 9.3 | 5.9e-11 (2 / 34,134,779,536) | confirm |

- S2 is in-sample for M's summaries (M is fitted on all 12 circuits).
- S1-v1 is #12's transfer condition. S1 and S1-v3 are reported and do not decide #12.
- In v1 the in-sample copied-width gain from H4R was +2.6 of 12 (H4R S1). The held-out copied-width gain here is +2.8 in v1.

## 5. S3 (descriptive): stat model against copied model, held-out

Mean over seeds of C(`sl-f`, H_f) − C(`cl-f`, H_f), in circuits of 6, per (variant, fold) cell, 95% percentile bootstrap over seeds 20–29 (20,000 draws, a fresh `default_rng(0)` per cell, both arms at the same drawn seeds).

| Cell | mean sl | mean cl | Difference | 95% CI |
|---|---|---|---|---|
| v1 / f1 | 4.1 | 2.9 | +1.2 | [0.6, 1.8] |
| v1 / f2 | 5.7 | 4.5 | +1.2 | [0.5, 1.8] |
| v3 / f1 | 4.0 | 3.3 | +0.7 | [−0.5, 1.8] |
| v3 / f2 | 5.9 | 4.3 | +1.6 | [1.2, 2.0] |

**D2 = +1.175** (the mean of the four differences), against GW-4's criterion 2 limit of ≥ −0.5. Three of four CIs exclude 0; v3 / f1 does not.

## 6. Lap time (GW-4's criterion 3; descriptive extras)

Each network's median gap is the median, over its completed R-dev circuits, of (median lap / s-OCP lap − 1), in points. No network was left out (every `sa` and `base` network completed at least 2 circuits), so n is 20 against 20 pooled and 10 against 10 per variant.

| | sa mean (SD) | base mean (SD) | Δgap |
|---|---|---|---|
| Pooled (criterion 3) | 8.81 (0.48) | 7.29 (0.39) | **+1.52** |
| v1 | 9.01 (0.31) | 7.18 (0.35) | +1.83 |
| v3 | 8.60 (0.55) | 7.39 (0.42) | +1.21 |

- Criterion 3 requires Δgap < 1.0 (point estimate, both variants pooled). It is **+1.52**, over the limit by 0.52 points.
- IN-4's copied-width reference for this statistic is +0.30 pooled (+0.52 in v1, +0.09 in v3; prereg Section 5). The stat-model gap is larger in both variants. This round's `cl` arms were not part of the criterion; no Δgap for them is reported here.
- **Paired by seed (descriptive).** On the circuits both same-seed networks complete, the mean of the per-pair median-gap differences (`sa` − `base`) is +0.82 pooled (v1 +1.03, v3 +0.61), and the mean of per-circuit differences is +0.88 (v1 +1.02, v3 +0.73). Pairs left out: 0. The common-circuit count per pair ranges from 2 to 7. Only one of 20 pairs has a negative difference (v3, seed 24: −0.06 on medians, −0.09 on the mean of per-circuit differences).
- **Per circuit (descriptive, `interp-leftovers.md` Section 6.1's form).** On every circuit that both arms complete in at least one network, the `sa` gap is above `base`'s: 9 of 9 circuits in v1 (median delta +1.09) and 8 of 8 in v3 (median delta +0.69). Norisring, Sochi and Yas Marina appear in neither table, and Melbourne only in v1's. The `base` figures rest on few networks on some circuits (v1 Melbourne: 1; Hockenheim: 2).
- The lead's note: the paired figures compare networks on circuits both complete, and are smaller (+0.8 to +0.9) than the pooled figure the criterion reads (+1.52), which also reflects which circuits each arm completes. They are still positive.

## 7. GW-4's criteria 1–5

| # | Criterion | Value | Result |
|---|---|---|---|
| 1 | P1 confirms | P1: confirm | holds |
| 2 | D2 ≥ −0.5 | D2 = +1.175 | holds |
| 3 | Δgap < 1.0 point | Δgap = +1.520 | **fails** |
| 4 | Width report (restated): every v1 check passes in both folds; v3 misses limited to `f1/v3/share_1m`, `f2/v3/share_1m`, `f1/v3/spectrum:8-16`; every no-copy check passes | report sha256 `5871664b…` equals the pinned value; 90 checks; failed: `f1/v3/share_1m`, `f1/v3/spectrum:8-16`, `f2/v3/share_1m`; v1 failures 0; v3 failures outside the allowed set 0; no-copy failures 0 | holds |
| 5 | The default path stays bit-identical (10 listed tests) | not read here | read by GW-4 at the `generator-v1` commit |

- Criteria 1–4 do not all hold (`criteria_1_to_4_hold`: false), because of criterion 3 alone.
- Criterion 4's failed checks are exactly the three allowed ones. The restated criterion was written after (2a)'s report was seen (prereg Section 10, entry 1); the bands' own verdict for (2a) stays "does not meet".
- **Decision.** The adoption rule fails whichever way criterion 5 reads (`if_criterion_5_holds`: copied; `if_criterion_5_fails`: copied). Fallback #12 applies and S1-v1 confirms, so `generator-v1` uses copied R-dev widths. The analysis's text of the rule:

  > #12: copied R-dev widths if and only if (2a) fails the rule and S1-v1 confirms (not testable counts as not confirmed); otherwise the default widths

- E0's curvature variant is v1, decided before the tag without GW-3 data; it was not chosen on these results.
- The lead's note: the rule is deterministic and was fixed before any run. The result does not say copied widths are better than the stat model for generalising; S3 (descriptive) points the other way on held-out completion. The rule chose copied widths because the stat model's lap-time gap exceeded the limit, and copied widths had shown transfer.

## 8. The 80% gate (not read here)

The gate (decision #10) is E0's: 5 seeds × 1B steps in v1 on `generator-v1` and `sim-v1`; it passes if the median seed completes at least 10 of the 12 R-dev circuits at its final checkpoint, with no checkpoint chosen on R-dev. GW-3's 200M networks are not E0, and the analysis marks `read_here: false`. Because `generator-v1` will use copied widths, the gate's result will carry the label "in-sample for widths". The per-cell counts of networks with C ≥ 10 in Section 10 are **not the gate**.

## 9. Deviations and notes

1. **The retry (`gw3-v3-base-s26`).** A start-up CUDA failure, re-run once by the queue as the prereg's retry rule says. It affects no outcome: the run finished, all 120 are present, and the prereg counts a run missing only if it fails twice. The cause was not investigated beyond the logged error.
2. **Idle box time.** About 19 min × 4 GPUs (about 1.3 GPU-hours) between the queue's end and the pull-back, while the lead waited on the user's decision about another round. It affects cost only (14.7 GPU-hours against about 14 approved and a stop line of 16.8), no outcome.
3. **A second box.** VG-1's pilot and P4-B2 ran at the same time on a separate single-GPU box, allowed by the user on 27 Oct 2026 (prereg Section 10, entry 2). It shared nothing with GW-3's box. It affects no GW-3 outcome.
4. **Prereg Section 10, entry 1 (criterion 4 restated).** Decided on 27 Oct 2026, before the tag. Criterion 4 no longer tests (2a) against the width-model bands: the committed report already met the restated version, so GW-4's decision in practice rests on criteria 1, 2, 3 and 5. Here criterion 4 holds with exactly the three allowed misses. The readings offered for the two v3 misses are post hoc and are not tests.
5. **Prereg Section 10, entry 2 (more than one box; one more criterion-5 test).** The x4 box was the plan's first choice, so the single-box fallback was not used. The default-path test for `--skip-eval-rdev` (`f5e300b`) is in criterion 5's list of ten tests, which GW-4 runs at the `generator-v1` commit.
6. **Pre-registered statistic labels.** S1, S1-v1, S1-v3 and S2 are unadjusted secondaries; no Holm or other adjustment was applied or pre-registered. P1 is the only primary test.
7. **Unmeasured.** The throughput (154–160k steps/s) and the box timings other than the queue log and manifest come from the lead, not from files read for this document.
8. **After the tag.** No change to an outcome, test, seed or rule after any run started. Nothing is exploratory.

## 10. Descriptive tables

Mean number of circuits completed per arm (SD in brackets), and the count of networks out of 10 with C ≥ 10 of 12 on all 12 ("not the gate"). Held-out and in-sample-fold counts are out of 6 for the fold-trained arms (`cl-f`, `sl-f`); the in-sample fold is the fold the network's widths came from.

| Cell | All 12 | Held-out (of 6) | In-sample fold (of 6) | C ≥ 10 of 12 |
|---|---|---|---|---|
| v1 base | 4.6 (1.17) | | | 0 |
| v1 cl-f1 | 6.8 (1.32) | 2.9 (0.88) | 3.9 (0.88) | 0 |
| v1 cl-f2 | 7.5 (1.27) | 4.5 (0.85) | 3.0 (1.05) | 0 |
| v1 sl-f1 | 9.8 (0.92) | 4.1 (0.57) | 5.7 (0.67) | 7 |
| v1 sl-f2 | 9.7 (0.82) | 5.7 (0.48) | 4.0 (0.47) | 7 |
| v1 sa | 9.6 (0.52) | | | 6 |
| v3 base | 5.0 (1.56) | | | 0 |
| v3 cl-f1 | 7.7 (1.57) | 3.3 (1.16) | 4.4 (0.97) | 1 |
| v3 cl-f2 | 7.5 (1.65) | 4.3 (0.67) | 3.2 (1.23) | 1 |
| v3 sl-f1 | 9.8 (1.32) | 4.0 (1.05) | 5.8 (0.42) | 7 |
| v3 sl-f2 | 9.9 (0.57) | 5.9 (0.32) | 4.0 (0.82) | 8 |
| v3 sa | 9.3 (1.34) | | | 5 |

- **Per-seed pairs (`sl` against `base`, held-out unit against baseline).** In every one of the 20 same-seed pairs the `sl` unit is above the `base` unit: differences +3 to +7 in v1 and +2 to +8 in v3.
- **Named circuits (share of 10 networks completing).** Norisring stays low everywhere: 0.0–0.2 in every cell. Sochi: 0.0 for `base` and `sa` (v1) and 0.1–0.3 for the other arms. Yas Marina: 0.0 for `base`, 0.0–0.1 for `cl`, 0.8–1.0 for `sl` and 0.6–0.8 for `sa`. Melbourne: 0.0–0.1 for `base`, 0.3–0.6 for `cl`, and 1.0 for `sl` and `sa`.
- The per-circuit shares are in `outputs/gw3/analysis.json` (`descriptive.per_circuit_share`).
