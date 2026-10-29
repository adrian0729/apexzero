# protocol-v1: the lockbox (pre-registration)

**Status.** Approved by the user on 29 Oct 2026 and fixed by the annotated tag `protocol-v1` on the commit of this text, pushed **before any lockbox stage ran** (Section 13). The R-dev dry run's numbers are in Sections 9 and 12. When the tag was placed, no lockbox number existed.
- It was reviewed independently twice before the approval (the harness's fixes and this text), and every finding was applied.
- The tag pins this file, `docs/prereg/protocol-v1-subjects.json`, the predictions (`docs/prereg/protocol-v1-predictions.json` and `scripts/lb1_predictions.py`, which wrote it) and the metric code (`scripts/lockbox.py`, `scripts/_bootstrap.py`, `src/apexzero`, `uv.lock`, `pyproject.toml`). Every lockbox stage refuses to start unless the working tree's metric code is the tag's (`tag_gate`, Section 13).
- Until the tag, nothing under `data/tracks/tumftm/` for an R_TEST circuit and nothing under `outputs/sealed/` is read.

**The user's decision (29 Oct 2026, seventh round; follow-through #14):** the lockbox goes ahead on E0's networks as they are, although E0 missed its 80% gate (median 8 of 12 R-dev circuits; [`docs/findings/e0.md`](../findings/e0.md)). It uses **all 12 R_TEST circuits**, with no 6 + 6 split. It is decoupled from the paper's `audit-v1`. The other defaults are #15's (follow-through Section 10, end notes).

## 1. Scope and decisions

- **One shot.** Each subject is evaluated once on R_TEST, at the tag. Any later R_TEST run is POST HOC (Section 2).
- **Confirmatory:** nine readings, H1 (Section 8, one reading of two parts) and eight calibrated predictions (Section 9), each read on its own. There is no multiplicity correction and no joint pass or fail; the number of predictions held is descriptive. The intervals are marginal 95%, and the eight rows are correlated, so a miss or two is expected even if the model is right. **Everything else is descriptive** and decides nothing.
- **The plan's wording.** [`docs/plan/plan-v1.md`](../plan/plan-v1.md) lists the lockbox as "R-test lockbox (12) ... each also mirrored and reversed ... evaluated once per final model", and fixes: "R-test never touches tuning, generator fitting or checkpoint choice."
- **The code.** `scripts/lockbox.py` (LB-0, with its tests and a dry run on R-dev, Section 12) is the whole analysis. This file cites its fields and constants by name. No constant in `PROTOCOL` is changed from the code at the tag.

| # | Decision | Value | Section |
|---|---|---|---|
| 1 | Subjects | `e0-s50` … `e0-s54`, final checkpoints, in `docs/prereg/protocol-v1-subjects.json` | 2 |
| 2 | Selection | the final checkpoint of each seed; no choice on R-dev | 2 |
| 3 | Variants | 12 circuits × {original, mirrored, reversed} = 36; H1 reads the 12 originals | 3 |
| 4 | Runs | `PROTOCOL["laps"]`: 10 runs per circuit, 2 laps, lap 2 timed, seed 0 | 3 |
| 5 | Circuit completed | more than 5 of 10 runs finish (`completion_min` 0.5) | 3 |
| 6 | Gap reference | the re-solved s-OCP lap, at the tag; no TO-OCP, no G_20 | 4 |
| 7 | Geometry error bar | every variant also at widths −0.5 m and +0.5 m | 4 |
| 8 | Failed references | missing, never filled | 4, 6 |
| 9 | Uncompleted circuits | a failure for completion, left out of the gap; +inf variants as sensitivity | 6 |
| 10 | Aggregation | IQM, stratified bootstrap, 2,000 draws, seed 0, nearest-rank CI | 7 |
| 11 | H1 | plan-v1's wording, read on the median subject | 8 |
| 12 | Predictions | calibrated on the R-dev dry run | 9 |
| 13 | Line / execution split | descriptive, not computed in LB-1 | 5 |

## 2. Subjects and the selection rule

- **Subjects:** `e0-s50`, `e0-s51`, `e0-s52`, `e0-s53`, `e0-s54` (E0, [`docs/plan/e0.md`](../plan/e0.md)): five seeds, 1B steps each on `generator-v1` and `sim-v1`, the actor only.
  - The checkpoint of each is `runs/<name>/ckpt/000999948288.pkl`, at **999,948,288 steps** (7,629 updates of 131,072 steps).
  - The names, paths, steps and sha256s are in `docs/prereg/protocol-v1-subjects.json`. The file is committed, and the tag pins it: `freeze` accepts it only if it is tracked and byte-identical to the tag commit's copy, and records its repo-relative path and sha256 (`freeze.json`, `subjects_file`).
  - `freeze` checks each checkpoint against its sha256 and the declared `timesteps` against the pickle's and the file name's. `laps` re-checks the sha256 before each use (`load_subject`).
  - The lockbox file carries no `eval_rdev` or `eval_rdev_sha256` (those belong to the dry run; `freeze` refuses them in lockbox mode).
- **Selection rule: the final checkpoint of each seed. No checkpoint, seed or subject is chosen on R-dev.**
  - It is the rule E0's gate used (`docs/plan/e0.md` Section 3).
  - GW-1 found that R-dev completion peaks at early, slow checkpoints: 20 of the 21 network-checkpoints with C ≥ 10 are at 60.3M steps or earlier, and their median lap gap is 11–54% (`docs/findings/gate-diagnostics.md` Section 4.4; `docs/findings/e0.md` Section 4). Choosing on R-dev completion would pick slow laps and trade away H1's lap-time half.
  - The lockbox's rule (plan-v1) is that R_TEST never touches checkpoint choice. A final checkpoint needs no choice.
- **All five seeds are subjects,** including seed 50, which completed 4 of 12 R-dev circuits. None is dropped.
- **POST HOC.** Any later R_TEST run is labelled POST HOC wherever it is reported: another checkpoint, seed or network, another circuit set, changed constants or code, or a rerun of `analyse` after its lock (Section 15).

## 3. Variants, runs and completion

- **Circuits:** the 12 circuits of `apexzero.tracks.corners.R_TEST`. `freeze --mode lockbox` takes all of them and refuses a subset or `--circuits`.
- **Variants:** `VARIANTS` = original, mirrored, reversed: **36 tracks**, each `refine_reference(variant(load_tumftm(csv)))`, the variant applied first, as the environment sees it.
  - **H1 is read on the 12 originals.** The mirrored and reversed tracks are descriptive memorisation diagnostics.
- **Laps** (`PROTOCOL["laps"]`, unchanged): the default car (`racing.default_context`), grip 1, no observation noise.
  - A rolling start at s = 0 from `start_fraction` 0.9 × the QSS start speed (`reference_speed(track)[0]`).
  - `runs` 10 per track: run 0 centred (n = 0, the full start speed); runs 1–9 jittered as `eval.laps.timed_laps` does (n uniform on ±1 m, start speed × uniform 0.95–1.05). `seed` 0. `freeze` refuses any other `--runs` in lockbox mode.
  - `laps` 2, lap 2 timed. The harness's rollout equals `timed_laps` bit for bit; each subject's first variant is checked against it (`parity_check`) and the stage stops on a difference.
- **A "lap" in H1 is one run whose timed second lap finishes.** A run finishes when the car reaches the end of lap 2 alive.
- **A circuit is completed when more than 5 of its 10 runs finish** (`completion` > `completion_min`, 0.5). Exactly 5 of 10 does not count, as in E0's gate.

## 4. References and the geometry error bar

- **Reference:** `ocp.min_time.solve_min_time`, re-solved at the tag under the tagged code, in `outputs/lockbox/ocp/` only. `PROTOCOL["ocp"]`: `ds` 3 m, `margin` 1 m, `tol` 1e-6, `max_iter` 3000.
  - **Fallback:** if the solve does not succeed, one retry from the failed iterate at tol 1e-5 (the `fallback` text; `scripts/solve_all_ocp.py`'s rule). No other retry.
  - **A reference is failed** if it still does not succeed, if its solve or widening raises (its pickle then holds None and the stage goes on), or if `abs(shooting_rel_err)` is at or above `P2_SHOOTING_LIMIT` (3e-3, the P2 gate's shooting consistency; a missing error does not pass). **Every gap against a failed reference is missing, never filled.** Nothing is borrowed from `outputs/ocp/` or `outputs/sealed/`, and nothing is re-solved with other settings.
  - `analyse` reads references only from the root's own `ocp/`, each pickle checked against its record's sha256 and the root's `freeze_id`.
- **Geometry error bar** (plan-v1's ±0.5 m): every variant is also solved with both sides' widths moved by each of `PROTOCOL["widths"]["deltas_m"]` = 0.0 (nominal), −0.5 and +0.5 m (`w_left` and `w_right`, each; the reference line stays). A width change that leaves a side at or below `margin` fails that reference.
  - 36 variants × 3 = **108 solves**.
- **G** = median completed lap / T_OCP − 1, with T_OCP the nominal (0.0) reference's `lap_time`. G is a fraction: 0.04 is 4%. The −0.5 and +0.5 m gaps use their own references' lap times and the same median lap.
- **Not built, not declared:** TO-OCP lap mode and a certified G_20 (LB-opt). The gap reference is the s-OCP lap, and the report says so.

## 5. Metrics

All fields are in `outputs/lockbox/analysis.json` unless noted. `<s>` is a subject name; `<f>` is a family: `original`, `mirrored`, `reversed` or `all` (all 36). A row of `rows` is one subject × variant. The gap keys of a row are `"+0.0"`, `"-0.5"` and `"+0.5"` (the width change in m). A non-finite number is written `null` (missing) or `"inf"`.

| Metric | Field | Status |
|---|---|---|
| Circuit completion (per variant) | `rows[].completion` (share of its 10 runs), `rows[].circuit_completed` | descriptive |
| Circuits completed | `per_subject.<s>.<f>.circuits_completed`, `circuits_completed_share`; `headline_median_subject.<f>.circuits_completed` | prediction (Section 9) |
| Lap completion | `per_subject.<s>.<f>.laps_completed_share` (finished runs over all runs of the family, uncompleted circuits included); `headline_median_subject.<f>.laps_completed_share` | **H1**, prediction |
| Gap G (per variant) | `rows[].gap["+0.0"]`, `rows[].median_lap_s`, `rows[].t_ocp_s` | descriptive |
| Median gap (completed circuits) | `per_subject.<s>.<f>.median_gap`; `headline_median_subject.<f>.median_gap` | **H1**, prediction |
| IQM gap | `iqm_gap.<f>.completed_only.iqm` and `.ci95` | prediction |
| Geometry error bar | `per_subject.<s>.<f>.median_gap_narrow` (−0.5 m), `median_gap_wide` (+0.5 m); `rows[].gap["-0.5"]`, `["+0.5"]`; `geometry_error_bar.median_change_in_gap`; `geometry_error_bar.relative_change_in_t_ocp` (median, min, max per delta) | descriptive |
| Per-corner Δt by corner type | `corner_types.<s>.<f>.<type>` with `n_corners`, `dt_sum_s`, `dt_median_s`; `corners.csv` (`dt_median_s` per corner) | descriptive |
| Lap-time SD | `rows[].lap_sd_s` (sample SD, ddof 1, over the variant's completed runs; missing under 2) | descriptive |
| d_OOD | `rows[].d_ood_median`, `d_ood_p90`, `n_ood_steps`; `d_ood.per_variant`; `corners.csv` `d_ood_median`; `d_ood.spearman_completion_vs_d_ood`, `spearman_corner_dt_vs_d_ood`, `crash_corners_vs_others` | descriptive |
| Crashes | `rows[].crashes`, `timeouts`, `causes` (corridor, slip, spin, stall); `corners.csv` `crashes_here` | descriptive |
| Harness checks | `checks` (Section 12) | record |

- **Corner types.** `viz.sectors.corner_sectors` on the variant's track (`PROTOCOL["sectors"]`: `merge_gap_m` 40). The time lost in a sector is the sector delta of t_agent(s) − t_OCP(s) on lap 2 against the nominal reference, so the sectors sum to the lap delta (`checks.sector_sum_closure_s`; the stage stops if it exceeds 1e-6 s). The type is set by the nominal reference's apex speed, its minimum v_x between the complex's entry and exit: slow < 120, medium 120–200 (both ends), fast > 200 km/h (`type_kph`). A corner whose reference failed is `unknown`. Only completed circuits enter `corner_types`.
- **d_OOD** (`PROTOCOL["ood"]`): the generalist analogue of step 1's d_OOD, from the frozen `interp.controls` code. Each subject's buffer holds `n_obs` 40,000 normalised observations from one deterministic `steps` 1,200-step window on each of `levels` 100 generated levels (`root_seed` 7707, stream `a_dr`; subsample seed 7707). The generator is the subjects' own, `generator-v1`: `fit_grammar` on the raw R-dev centrelines with `real_widths` true, so the strips carry copied R-dev widths. The denominator is the buffer's leave-one-level-out median nearest-neighbour distance. A variant's value is the median (and the 90th percentile) over run 0's steps on its two laps; a corner's is the median over run 0's steps in its sector.
- **The line / execution split is descriptive and is not computed in LB-1.** `laps/<subject>/<variant>.npz` stores each completed run's lateral offset n(s) (`path_n`) and speed v_x(s) (`path_v`) on lap 2, at the stations, so the split can be done later. Anything computed from them is POST HOC and says so.

## 6. Uncompleted circuits, crashes and failed references

- **An uncompleted circuit** (10 runs, 5 or fewer finish) is a **failure for completion**. It is **left out of the gap**: its gap is not computed, its corners are not in `corner_types`, and it is missing (not zero, not a worst case) in `median_gap` and in the completed-only IQM.
- **Within a completed circuit,** the gap uses the **median over its completed runs**. Runs that crashed or timed out do not enter the median; they are counted in `crashes`, `timeouts` and `causes`.
- **A completed circuit whose reference failed** counts for completion and has no gap.
- **Sensitivity checks (reported with the headline, never in its place):** the same statistics with an uncompleted circuit counted as +inf.
  - `per_subject.<s>.<f>.median_gap_crash_inf` (and `median_gap_crash_inf_narrow`, `median_gap_crash_inf_wide`): the median over all the family's circuits with +inf for each uncompleted one. It is +inf when half or more of the circuits are uncompleted (with 12, from 6). A completed circuit with a missing gap makes it missing, never smaller.
  - `iqm_gap.<f>.uncompleted_as_inf`: the IQM with +inf entries kept (Section 7).
- **Crash description** (`outcomes`): a run that terminates without finishing is a crash (cause: corridor, slip, spin or stall, in that order, else `nonfinite`); a run that neither finishes nor terminates is a timeout.

## 7. Aggregation

- **IQM** (`iqm`, rliable's, `trim_mean(x, 0.25)` written out; `PROTOCOL["stats"]["iqm_trim"]` 0.25): sort, cut `int(0.25 n)` from each end, average the rest.
- **Matrix:** subjects (5) × tracks of the family (12 for `original`, 36 for `all`), the cell being the track's `rows[].gap["+0.0"]` for that subject.
- **Stratified bootstrap** (`iqm_stratified`): the subjects are resampled with replacement **within each track** (each column), `n_boot` **2,000** draws, `boot_seed` **0**. The interval is `ci` [2.5, 97.5] by **nearest rank** (`_pct`, no interpolation, so +inf entries are safe), written `ci95`.
- **Two versions, both reported** (`iqm_gap.<f>`):
  - `completed_only`: an uncompleted circuit, and a circuit with a missing gap, is missing;
  - `uncompleted_as_inf`: an uncompleted circuit is +inf (`keep_inf`); a missing gap stays missing. The IQM is +inf when +inf reaches the middle half of the scores.
  - each holds `iqm`, `ci95`, `n_scores`, `n_inf`, `n_missing`, `n_boot`, `seed`.
- **The headline** (`headline_median_subject.<f>`): for each statistic in `per_subject.<s>.<f>`, the median over the five subjects (`_med`; missing values dropped). With five subjects it is the third value. Different statistics can come from different subjects.
- Five networks are too few for a subject-level test. No p-value is declared.

## 8. H1

**H1 (plan-v1, Section 2), word for word:** "The DR generalist completes ≥90% of laps on R-test with median G ≤ 4%."
Plan-v1 lists it among the "Pre-registered hypotheses. Commit these before unsealing the test set. They are targets, not predictions."

- **Subject of H1:** E0's five networks stand for "the DR generalist".
- **Reading:** on the 12 originals, on the median subject, as the code does (`PROTOCOL["stats"]["h1"]`: `laps_completed_min` 0.90, `median_gap_max` 0.04). Both are read from `headline_median_subject.h1_as_written_on_median_subject`:
  - `laps_completed_share` at or above 0.90, and
  - `median_gap` finite and at or below 0.04 (a missing median gap fails);
  - `holds` is true only if both hold. `subjects_meeting_both` counts the subjects (of 5) that meet both on their own (`per_subject.<s>.h1_as_written`: `laps_ok`, `gap_ok`).
- **Outcomes:**

| Laps ≥ 0.90 | Gap ≤ 0.04 | Reading |
|---|---|---|
| yes | yes | H1 holds |
| yes | no | H1 refuted on the gap |
| no | yes | H1 refuted on laps |
| no | no | H1 refuted on both counts |

- **What the plan expects:** H1 refuted on both counts with today's agents (follow-through Section 7.7: best R-dev means of 7.6–8.5 circuits of 12, with gaps of 6.7–7.3%; E0 itself completed 8 of 12 at the median, with a median gap of 6.4% on its completions, in-sample for widths, [`docs/findings/e0.md`](../findings/e0.md)). The lockbox is spent to measure this, not to rescue the hypothesis. A refutation is reported as the result.
- **No other hypothesis is tested.** H2–H5 need the curriculum arms, the scaling curve or the drivers study (Section 10).

## 9. Calibrated predictions (from the R-dev dry run)

These are confirmatory with H1. They are written before the tag, from the dry run on R-dev (Section 12), so that a miss on R_TEST is a measured miss.

**How a prediction is formed.**
- The **point value** is the dry run's value of the same field on R-dev (its `analysis.json`), original, mirrored and reversed families, median subject (the IQM row: over subject × circuit).
- The **interval** is a circuit bootstrap of the dry run (`scripts/lb1_predictions.py`, output `docs/prereg/protocol-v1-predictions.json`, both pinned by the tag): the 12 R-dev circuits are resampled with replacement, each draw keeping all five subjects' rows of a circuit and its three variants together (so the differences are paired), every statistic is recomputed with `scripts/lockbox.py`'s own functions, and the interval is the nearest-rank 2.5th–97.5th percentile over 4,000 draws (seed 2029). There is no further widening. The interval models R_TEST as a fresh draw of 12 circuits like R-dev's; the subjects' own bootstrap (`iqm_gap.<f>.completed_only.ci95`) is not used, because it holds the circuits fixed. The script refuses unless every headline input it recomputes (each family's lap share, circuits completed, completed share and median gap) and the IQM equal the dry run's `analysis.json` to within 1e-12.
- A prediction **held** when the R_TEST value of its field lies inside its interval (closed), both rounded to 12 decimal places, so that a value equal to a bound in exact arithmetic is never lost to float rounding. A missing value, or a difference with a missing term, does not hold. The difference rows are computed from the two named fields, because `analysis.json` stores no difference. The completion differences are in circuits (whole numbers), not shares.
- Every R_TEST number is reported next to its prediction, whether it held or not. R-dev and R_TEST are different circuits, so a miss is information, not an error.
- The base is in-sample for widths (Section 11).

| Quantity | `analysis.json` field | Predicted value | Interval | Held when |
|---|---|---|---|---|
| Lap share (median subject, originals) | `headline_median_subject.original.laps_completed_share` | 66.7% | [41.7%, 83.3%] | R_TEST value inside the interval |
| Circuits completed (median subject, originals; 12 circuits in both sets) | `headline_median_subject.original.circuits_completed` | 8 of 12 | [5 of 12, 10 of 12] | R_TEST value inside the interval |
| Median gap (median subject, originals) | `headline_median_subject.original.median_gap` | 6.3% | [5.9%, 7.6%] | R_TEST value inside the interval |
| IQM gap (originals, completed only) | `iqm_gap.original.completed_only.iqm` | 6.5% | [6.0%, 7.4%] | R_TEST value inside the interval |
| Mirrored − original, circuits completed | `headline_median_subject.mirrored.circuits_completed` − `headline_median_subject.original.circuits_completed` | +1 | [0, +4] | R_TEST difference inside the interval |
| Reversed − original, circuits completed | `headline_median_subject.reversed.circuits_completed` − `headline_median_subject.original.circuits_completed` | +3 | [0, +5] | R_TEST difference inside the interval |
| Mirrored − original, gap | `headline_median_subject.mirrored.median_gap` − `headline_median_subject.original.median_gap` | -0.0 pts | [-0.8 pts, +0.7 pts] | R_TEST difference inside the interval |
| Reversed − original, gap | `headline_median_subject.reversed.median_gap` − `headline_median_subject.original.median_gap` | +0.3 pts | [-0.6 pts, +0.8 pts] | R_TEST difference inside the interval |

The table rounds. The exact points and bounds, which decide "held", are in `docs/prereg/protocol-v1-predictions.json` (`predictions.<quantity>.point` and `.interval`).

## 10. Not declared

- **No context sweep.** The lockbox runs the default car at grip 1 only.
- **No scaling curve.** There are no networks with N < ∞ training tracks, so H4 (G falls with N) is untested.
- **No critic comparison.** The asymmetric critic was never built, and E0's critic reads the actor's observation, so E0 is effectively the symmetric arm. Only the actor is evaluated.
- **No curriculum comparison (H2, H3),** no drivers study (H5), no TO-OCP or certified G_20, no line / execution split (Section 5).
- **No Tier-3 (approximate-geometry) circuits.**

## 11. Labels

- **"In-sample for widths"** (`docs/findings/gw3.md` Section 1; `docs/findings/e0.md`): E0 trained on the 12 R-dev circuits' own width profiles. The label covers every R-dev number for E0.
- **It does not apply to any R_TEST number.** `gw3.md` Section 1: "R_TEST is unaffected." No R_TEST circuit's width was in E0's training.
- **It applies to every dry-run number,** and so to the predictions' base (Section 9). A prediction that held on R_TEST does not turn an in-sample number into a held-out one.
- The d_OOD buffer is built from a generator that copies R-dev widths (Section 5). That is the subjects' own training distribution, and it is stated with d_OOD.

## 12. The dry run

`scripts/lockbox.py freeze --mode dryrun` ran with R-dev standing in for R_TEST, on the same five subjects and the same constants. R_TEST names raise at every stage in that mode (`lockbox_guard`).

| Record | Value |
|---|---|
| Root | `outputs/lockbox-dryrun/lb1-rdev-e0/` |
| Commit it ran from | `26583a2` (`26583a2813e104d2106324987706ee06bde34b42`) |
| `freeze_id` (`freeze.json`) | `81866c97de3e4e4e84311d30f7f5ee2a` |
| `git_head` (`freeze.json`, `provenance`) | `26583a2813e104d2106324987706ee06bde34b42` |
| `changed_paths` at freeze (must be empty) | `[]` |
| Circuits frozen | the 12 R-dev circuits × 3 variants (36), 5 subjects |
| Subjects file it froze from | `outputs/lockbox-dryrun/subjects-e0.json`: the same five checkpoints, sha256s and timesteps as `docs/prereg/protocol-v1-subjects.json`, plus each seed's `eval_rdev` (dry-run mode only). The lockbox file itself is exercised in lockbox mode only by the test rehearsal (`tests/test_lockbox.py`) |
| sha256 of its `analysis.json` | `931aa2fee4579d9c89cc62c97a56a56060cbf1bc90bc32d5ec3dab6ac27ac3c7` |

- **The tag's requirement.** `git diff <dry-run commit> protocol-v1 -- scripts/lockbox.py scripts/_bootstrap.py src/apexzero uv.lock pyproject.toml` (the code's `METRIC_PATHS`) must be empty, so the dry run's code is the lockbox's. If it is not empty, the dry run is repeated before the tag.
- **Stored-eval agreement** (`checks.stored_eval_agreement.<s>`): the dry run's original variants against each subject's own end-of-training `eval_rdev.json` (the A6000 run), recorded at freeze for each subject (dry run only). The harness records `completion_equal` out of `circuits` and `max_abs_median_lap_diff_s`. It is descriptive, not a check; GW-1 found completion equal on 334 of 336 calls between the Mac and the A6000 and no lap time equal bit for bit.
  - **Expected:** circuits completed per seed of 4, 10, 8, 8, 6 (`docs/findings/e0.md` Section 1), seeds 50–54.

| Subject | E0 `eval_rdev.json` circuits completed | Dry run `per_subject.<s>.original.circuits_completed` | `completion_equal` / `circuits` | `max_abs_median_lap_diff_s` |
|---|---|---|---|---|
| e0-s50 | 4 | 4 | 12 / 12 | 0.0007 |
| e0-s51 | 10 | 10 | 12 / 12 | 0.0082 |
| e0-s52 | 8 | 8 | 12 / 12 | 0.0006 |
| e0-s53 | 8 | 8 | 12 / 12 | 0.0083 |
| e0-s54 | 6 | 6 | 12 / 12 | 0.0015 |

- **Other `checks` the dry run records,** read the same way on the lockbox: `references_converged`, `unconverged`, `shooting_failed`, `references_failed`, `retried`, `max_abs_shooting_rel_err`, `shooting_over_p2_limit`, `mirror_ocp_lap_rel_diff`, `sector_sum_closure_s`.

## 13. Procedure

Everything runs from the repository root, on the Mac (no GPU box), at the tag. `--workers 3` (at most 4); each worker runs at nice 15 with 2 XLA and BLAS threads. The Mac must be online for each stage: `tag_gate` checks the tag on origin (`git ls-remote`) every time.

1. **Before the tag.** The dry run is complete and its numbers are in Sections 9 and 12. The metric-code diff of Section 12 is empty. The lead has filled this file (Section 16), it has been reviewed, and the user has approved it.
2. **Tag.** Commit this file, `docs/prereg/protocol-v1-predictions.json` and `scripts/lb1_predictions.py` (the subjects file is committed already, at `26583a2`). Place the annotated tag `protocol-v1` on that commit and push it. Record the tag object's sha and the commit's in the findings. `tag_gate` requires, at every stage:
   - the annotated tag `protocol-v1` is on origin, and the local tag is the same tag object;
   - HEAD descends from the tag's commit;
   - the metric code (`METRIC_PATHS`: `scripts/lockbox.py`, `scripts/_bootstrap.py`, `src/apexzero`, `uv.lock`, `pyproject.toml`) in the working tree equals the tag's, with no untracked file among it.

   Behind it, and nowhere else, `lockbox_guard(name, unsealed=True)` admits the R_TEST names.
3. **Run,** in order, each stage behind `tag_gate` (`outputs/lockbox/` must not exist before `freeze`):

   ```
   uv run python scripts/lockbox.py freeze --mode lockbox --root outputs/lockbox --subjects docs/prereg/protocol-v1-subjects.json
   uv run python scripts/lockbox.py ocp --root outputs/lockbox --workers 3
   uv run python scripts/lockbox.py laps --root outputs/lockbox --workers 3
   uv run python scripts/lockbox.py analyse --root outputs/lockbox
   ```

   - `freeze` binds the circuits and their variants, the constants, the subjects (checkpoint sha256s, checked), the circuit files' sha256s and a fresh `freeze_id` into `freeze.json`, before any number exists.
   - `ocp` and `laps` skip what is finished and checked, so an interrupted stage is resumed by running it again. `laps` prints no outcome; `ocp` prints each reference's convergence, error and solve time only.
   - **`analyse` runs once.** It refuses if its lock or any output exists, checks every input, and takes an exclusive lock (`analysis.lock`, never removed by the harness) before computing any number. It writes `analysis.json` and `corners.csv`. It never reads `outputs/ocp/` or `outputs/sealed/`.
   - **A rerun after the lock is a logged deviation, never silent** (Section 15). If `analyse` fails after taking the lock, the lock is removed by hand only with a logged deviation.
4. **Commit** `outputs/lockbox/analysis.json` and `corners.csv` (force-added, as earlier rounds' analysis files were), with their sha256s in the findings.

## 14. Afterwards

- **The findings note** is `docs/findings/lockbox.md`. It reports every Section 5 metric, H1's reading, each prediction held or missed (Section 9), the failed references and every deviation. Each number is read from `analysis.json` or `corners.csv`.
- **Then, and only after the findings are committed,** `outputs/sealed/` may be opened for a **descriptive provenance comparison** of the old optima against the re-solved references. It decides nothing, changes no number in `lockbox.md`, and is labelled as done after the findings.
- Any further analysis of the lockbox data (the line / execution split, other checkpoints, other statistics) is POST HOC.

## 15. Deviations

- Any change after the tag is logged in the findings with its date and reason.
- A change to a metric, constant, subject, reference rule, outcome or reading after the tag makes the affected result POST HOC. The metric code cannot change after the tag without `tag_gate` refusing the stage; if a stage fails on a bug, the lead stops, logs it and asks the user.
- A rerun of `analyse` after its lock exists, a hand-removed lock, a stage rerun other than resuming an interrupted `ocp` or `laps`, or an R_TEST run outside the four commands above is a logged deviation. It is reported with both outputs, never silently replacing the first.
- A failed reference, an uncompleted circuit or a missing gap is a result, not a deviation, and is reported as the rules above say.
- Deviations decided before the tag are logged here, append-only, dated and part of the text the tag pins.

## 16. Items for the lead before the tag

| Item | Where |
|---|---|
| Dry run: commit, `freeze_id`, `git_head`, `changed_paths`, circuits frozen, `analysis.json` sha256 | Section 12 |
| Dry run: stored-eval agreement per subject | Section 12 |
| Empty metric-code diff from the dry-run commit to the tag | Sections 12, 13 |
| Prediction values and intervals: filled from the dry run | Section 9 |
| The interval rule: the circuit bootstrap, no widening | Section 9 |
| `PROTOCOL["stats"]["h1"]["source"]` in `scripts/lockbox.py` still reads "plan-v1 H1 as written; LB-1 may replace it". It is copied into `freeze.json` and changes nothing; this protocol keeps H1 as written (Section 8). It is left as is, because editing it would change the metric code after the dry run | Section 8 |
| Review, then the user's approval | Section 13 |
