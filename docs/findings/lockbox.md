# Lockbox (protocol-v1): H1 is refuted on both counts on R_TEST (median subject 66.7% of laps, median gap 5.2%); six of the eight calibrated predictions held, and the two gap levels missed on the low side (pre-registered)

Results of 29 Oct 2026. Pre-registration: [`docs/prereg/protocol-v1.md`](../prereg/protocol-v1.md), approved by the user and fixed by the annotated tag `protocol-v1` (tag object `5bc35e288a218fe61401e4a77d04461209a30119`, on commit `8c1a8d7f716a995d91688335614e5c153b1d6109`), pushed before any lockbox stage ran.
- **The run.** The four commands of protocol Section 13 ran once each, on the Mac, at the tag: `freeze`, `ocp`, `laps` and `analyse` (Section 9 below). `analyse` wrote `outputs/lockbox/analysis.json` and `corners.csv` under a lock that is never removed. Both are committed in `4d4eaf9` with `freeze.json` and the lock.
- **What changed after the tag.** Nothing. HEAD was the tag's commit at every stage, `tag_gate` passed at each, and `analysis.json`'s `provenance.changed_paths` is `[]`. One deviation is logged, with no effect on any number: each command ran with a `nice` and `--no-sync` prefix (Section 9).
- **Subjects.** E0's five networks, `e0-s50` … `e0-s54`, at their final checkpoints (999,948,288 steps), as `docs/prereg/protocol-v1-subjects.json` lists them. None was chosen or dropped.

Every number below is read from `outputs/lockbox/analysis.json`, `outputs/lockbox/corners.csv` or `docs/prereg/protocol-v1-predictions.json`, except where marked "computed here" (from those files' own rows). Labels:
- **Confirmatory:** H1 (Section 1) and the eight calibrated predictions (Section 2), each read on its own, with no multiplicity correction (protocol Section 1).
- **Descriptive:** everything else. It decides nothing.
- **R_TEST numbers are held out.** No R_TEST circuit, and no R_TEST width profile, was in E0's training (protocol Section 11). **Every R-dev number quoted for comparison is in-sample for widths** (E0's `eval_rdev.json` and the R-dev dry run, `outputs/lockbox-dryrun/lb1-rdev-e0/`).

G is the gap to the s-OCP lap (median completed lap / T_OCP − 1; 0.04 is 4%). "The median subject" is the median over the five subjects of each statistic (`headline_median_subject`), so different statistics can come from different subjects. Times are UTC.

## 1. H1: refuted on both counts

H1 (plan-v1): "The DR generalist completes ≥90% of laps on R-test with median G ≤ 4%." It is read on the 12 originals, on the median subject (`headline_median_subject.h1_as_written_on_median_subject`).

| Part | Value | Bar | Met |
|---|---|---|---|
| `laps_completed_share` | **0.667** (66.7%) | ≥ 0.90 | no |
| `median_gap` | **0.0516** (5.16%) | ≤ 0.04 | no |
| `holds` | **false** | | |
| `subjects_meeting_both` | **0 of 5** | | |

- **The reading** (protocol Section 8): **H1 refuted on both counts.** This is the outcome the protocol said to expect. The lockbox measures it; it was not spent to rescue H1.
- **No subject meets either part alone** (`per_subject.<s>.h1_as_written`: `laps_ok` and `gap_ok` false for all five). The best lap share is 0.833 (seeds 51 and 53) and the smallest median gap is 5.09% (seed 50).
- **The gap part fails across the whole geometry error bar.** The median subject's median gap is 4.47% with the widths narrowed by 0.5 m and 6.16% with them widened by 0.5 m (`median_gap_narrow`, `median_gap_wide`; Section 6).
- **Sensitivity** (counting an uncompleted circuit as +inf; protocol Section 6): the median subject's `median_gap_crash_inf` is 6.59%. The IQM with +inf kept is +inf (`iqm_gap.original.uncompleted_as_inf`), because 19 of the 60 subject × circuit scores are uncompleted.

## 2. The calibrated predictions: six of eight held

Each prediction is the R-dev dry run's value, with a 95% circuit-bootstrap interval (protocol Section 9). It is held when the R_TEST value lies in the closed interval, both rounded to 12 decimals. The difference rows are computed here from the two named `analysis.json` fields, as protocol Section 9 says. The table rounds; the decision used the exact values in `protocol-v1-predictions.json`.

| Quantity (median subject unless noted) | Predicted | Interval | R_TEST | Held |
|---|---|---|---|---|
| Lap share, originals | 66.7% | [41.7%, 83.3%] | 66.7% | **yes** |
| Circuits completed, originals | 8 of 12 | [5, 10] | 8 | **yes** |
| Median gap, originals | 6.31% | [5.87%, 7.65%] | **5.16%** | **no** (below) |
| IQM gap, originals, completed only (subject × circuit) | 6.47% | [5.96%, 7.40%] | **5.38%** | **no** (below) |
| Mirrored − original, circuits completed | +1 | [0, +4] | 0 | **yes** |
| Reversed − original, circuits completed | +3 | [0, +5] | 0 | **yes** |
| Mirrored − original, median gap | −0.03 pts | [−0.81, +0.66] pts | +0.12 pts | **yes** |
| Reversed − original, median gap | +0.34 pts | [−0.59, +0.81] pts | +0.44 pts | **yes** |

- **Six of eight held.** The count is descriptive (protocol Section 1). The two completion rows and all four variant-difference rows held. Both completion differences sit at the bottom of their intervals.
- **The two misses are one finding.** The median gap and the IQM gap both draw on the 41 original completions (the headline median gap is seed 53's median over its own 10; the IQM pools all 41), and both missed on the same side: **E0's gaps on R_TEST are smaller than on R-dev.** The IQM's own subject bootstrap, [5.18%, 5.60%] (`iqm_gap.original.completed_only.ci95`, circuits held fixed), lies entirely below the predicted interval.
- **The lead's reading** (not tested): the interval treated R-dev's 12 circuits as the population that R_TEST is drawn from, and R_TEST's circuits gave the same networks smaller gaps. The in-sample label points the other way, since R-dev's widths were in training and R_TEST's were not, so it does not explain the miss. Two descriptive differences: R_TEST's reference laps are longer (median nominal T_OCP over the originals 82.6 s, against 74.9 s on R-dev; computed here), and the five circuits that every seed completes include the three with the smallest gaps (Spa 4.45%, Silverstone 4.67%, São Paulo 4.80%; Section 4).

## 3. Per subject (descriptive)

| Subject | Originals: circuits, lap share, median G | Mirrored: circuits, lap share, median G | Reversed: circuits, lap share, median G | R-dev circuits (E0, in-sample for widths) |
|---|---|---|---|---|
| e0-s50 | 6, 0.508, 5.09% | 8, 0.667, 4.88% | 7, 0.583, 4.79% | 4 |
| e0-s51 | 10, 0.833, 5.45% | 10, 0.833, 6.55% | 10, 0.833, 6.88% | 10 |
| e0-s52 | 8, 0.667, 5.85% | 7, 0.583, 5.17% | 7, 0.583, 4.91% | 8 |
| e0-s53 | 10, 0.833, 5.16% | 10, 0.875, 5.29% | 8, 0.700, 5.61% | 8 |
| e0-s54 | 7, 0.583, 5.10% | 7, 0.583, 5.31% | 9, 0.750, 5.69% | 6 |
| **Median subject** | **8, 0.667, 5.16%** | **8, 0.667, 5.29%** | **8, 0.700, 5.61%** | **8** |

- **All 36 variants:** the median subject completes 23 of 36 (63.9% of laps), with a median gap of 5.50% (`headline_median_subject.all`).
- **Against R-dev** (computed here, from `docs/findings/e0.md`'s counts): every seed completes at least as many R_TEST originals as it did R-dev circuits. Seeds 50 and 53 complete 2 more, seed 54 1 more, and seeds 51 and 52 the same number. The median is 8 on both.

## 4. By circuit (descriptive)

| Circuit | Originals: seeds completing (of 5) | Originals: median G over them (computed here) | Mirrored | Reversed |
|---|---|---|---|---|
| Montreal | 5 | 7.03% | 4 | 5 |
| São Paulo | 5 | 4.80% | 5 | 5 |
| Silverstone | 5 | 4.67% | 5 | 5 |
| Spa | 5 | 4.45% | 5 | 5 |
| Suzuka | 5 | 5.57% | 5 | 5 |
| Spielberg | 4 | 4.97% | 5 | 5 |
| Budapest | 3 | 6.32% | 4 | 4 |
| Monza | 3 | 5.59% | 3 | **0** |
| Austin | 2 | 5.95% | 3 | 2 |
| Sakhir | 2 | 5.68% | 1 | 2 |
| Mexico City | 1 | 9.18% | 1 | 2 |
| Shanghai | 1 | 5.34% | 1 | 1 |

- **Completion is nearly all-or-nothing.** Of the 60 original subject × circuit results, 59 are 0 or 1 (the exception is seed 50 at Spielberg, 1 run of 10). Over all 180, 177 are 0 or 1. Seed 53's mirrored Sakhir finishes exactly 5 of 10, which does not count (more than 5 is needed), and its reversed Mexico City finishes 4 (computed here, from `rows[].completion`).
- **The gaps** on the 41 original completions run from 3.82% to 10.10%, median 5.38% (computed here). Only 2 of the 124 completions over all variants are under 4%, both seed 50 at Spa (3.82% original, 3.95% mirrored).
- **Reversed Monza** is completed by no seed, although three complete Monza itself.

## 5. Mirrored and reversed (descriptive)

- At the median subject the three families complete the same number of circuits, 8, though not the same set (Monza: 3 seeds as an original, 0 reversed; Section 4). The lap shares are 66.7%, 66.7% and 70.0%. The median gap rises from originals (5.16%) to mirrored (5.29%) to reversed (5.61%).
- **Pooled over subject × circuit** (`iqm_gap.<f>.completed_only`): originals 41 completions, IQM 5.38% [5.18%, 5.60%]; mirrored 42, 5.40% [5.16%, 5.69%]; reversed 41, 5.70% [5.39%, 6.02%]. Every interval is the subjects' bootstrap with the circuits held fixed.
- No R_TEST circuit was trained, in any orientation. These rows describe differences between orientations of unseen circuits, not memorisation of them; both gap-difference intervals in Section 2 contain 0.

## 6. The geometry error bar (descriptive)

| Width change | Median subject's median G, originals | With uncompleted circuits as +inf | Median change in G, all 124 completions (`geometry_error_bar`) | Relative change in T_OCP, all 36 variants: median (min, max) |
|---|---|---|---|---|
| −0.5 m | 4.47% | 5.33% | −0.90 pts | +0.87% (+0.54%, +1.51%) |
| 0 (nominal) | 5.16% | 6.59% | | |
| +0.5 m | 6.16% | 7.94% | +0.87 pts | −0.81% (−1.49%, −0.52%) |

- No width change failed a reference: all 108 solves succeeded (Section 9).
- On R-dev the dry run's median changes were −1.11 and +1.04 pts.

## 7. Where the time goes: corner types (descriptive)

Originals, completed circuits only, pooled over the five subjects (computed here from `corner_types`; the per-corner median from `corners.csv`). Corners are typed by the nominal reference's apex speed: slow < 120, medium 120–200, fast > 200 km/h.

| Type | Subject × corner | Time lost (s, sum of per-corner medians) | Share of time lost | Median per corner (s) | Crashes here (all original circuits, `corners.csv`) |
|---|---|---|---|---|---|
| Slow | 141 | 91.6 | 50.1% | 0.561 | 90 |
| Medium | 101 | 49.9 | 27.3% | 0.461 | 89 |
| Fast | 201 | 41.5 | 22.7% | 0.106 | 10 |

- **Half the time lost is in slow corners**, which are a third of the corners. On R-dev the dry run's shares were 43.2%, 29.7% and 27.1% (in-sample for widths).
- **Crashes are in slow and medium corners:** 179 of the 189 original crashes. No corner is `unknown`, and the sector sums close to within 1.5e-14 s (`checks.sector_sum_closure_s`).

## 8. Crashes, lap-time spread and d_OOD (descriptive)

- **Crashes.** 550 of the 1,800 runs crash (189 of the 600 on originals). Every crash is a corridor exit: no slip, spin or stall, and no timeouts. In the R-dev dry run 538 crashed, 528 of them corridor exits and 10 slips.
- **Lap-time SD.** The median over the 126 subject × variant rows with at least two completed runs is 0.002 s, so the laps are close to deterministic under the start jitter. 13 rows exceed 0.05 s; the largest is 0.53 s (seed 52, mirrored Montreal). Six of the 13 are Montreal in some orientation.
- **d_OOD.** Across subjects and the 12 originals, the variant medians have a median of 1.20 (range 0.80–2.17), against 1.22 (range 0.99–1.90) for R-dev in the dry run (computed here).
  - The buffer comes from `generator-v1` with copied R-dev widths, the subjects' own training distribution (protocol Sections 5 and 11).
  - **Completion falls as d_OOD rises** in every subject: Spearman ρ over the 36 variants ranges from −0.27 to −0.58 (`d_ood.spearman_completion_vs_d_ood`).
  - **Corner Δt rises with d_OOD**, with ρ from +0.13 to +0.23 (`spearman_corner_dt_vs_d_ood`).
  - **Crash corners sit further out** in all five subjects: their median d_OOD is 1.37–2.39, against 1.14–1.18 for the other corners (`crash_corners_vs_others`).

## 9. The run, the checks and deviations

- **The run** (Mac, `--workers 3`, at commit `8c1a8d7`):
  - `freeze` at 22:31 UTC, with `freeze_id` `3ada5b1f4e084707b42060fdf278ba65`. Its tag record names tag object `5bc35e28…`, commit `8c1a8d7…` and HEAD `8c1a8d7…`. The subjects file's sha256 is `dbd9bbcf…`, the tag's copy.
  - `ocp` ran 22:31–23:27 UTC, 108 solves.
  - `laps` ran 23:27–23:31 UTC: 5 subjects × 36 variants × 10 runs. Each subject's parity check against `timed_laps` passed, since the stage stops on a difference.
  - `analyse` ran once, 23:32:24–23:32:34 UTC. Its lock (`analysis.lock`) stays in place.
- **Deviation 1 (29 Oct 2026; logged under protocol Section 15, no effect on any number).** Each of the four commands ran as `nice -n 10 uv run --no-sync python scripts/lockbox.py …`, not the literal `uv run python scripts/lockbox.py …` of Section 13. The reason was to keep the Mac responsive, and to keep uv from touching the environment between stages.
  - **Priority.** The harness still sets its own priority: `analysis.json` records `nice` 15, as Section 13 fixes.
  - **Environment.** `uv sync --locked --dry-run` reports no change, so the environment is `uv.lock`'s, which `tag_gate` pins.
  - **Effect.** The code, inputs, constants and arguments are those of Section 13, so nothing computed differs.
- **Before `analyse`, the lead read only the `ocp` records' convergence fields** (success, status, attempts, error, shooting error, drivable margin) and the `laps` stage's progress lines. No lap time and no agent outcome was read before the lock.
- **`checks`:**
  - `references_converged` is 108/108, all `Solve_Succeeded` on the first attempt (`retried` is `[]`, so the fallback was never used);
  - `references_failed` is 0, and no reference failed on shooting;
  - `max_abs_shooting_rel_err` is 1.8e-4, against the limit of 3e-3;
  - `mirror_ocp_lap_rel_diff` is at most 2.2e-4 (Austin);
  - `stored_eval_agreement` is `null`, by design in lockbox mode.
- **Failed references:** none. **Deviations:** one, Deviation 1 above, with no effect. As the protocol records (Section 16), `PROTOCOL["stats"]["h1"]["source"]` still carries its stale wording. It changes no number.
- **Committed in `4d4eaf9`** (sha256):
  - `analysis.json` `deab99fc3c14b7383845ecf7ec19a015eae41f79ff35780cfa2e7c08e50234af`
  - `corners.csv` `523f46ba48b87c7cd862979f2af8edf830fdea2945738a202dcdb4724a361c69`
  - `freeze.json` `91b02c2fe3f04ec3f583aac6d7e100b7f348dfce42570dae9fbf5f6644ad71bb`
  - `analysis.lock` `e8de983a3d958345219e2d4f0b36ad009db7740dd0c5affa27c9ee66002d1d0b`

  The laps (289 MB) and the reference pickles (24 MB) stay local and untracked. `analysis.json` records the sha256 of every input it read (`provenance.inputs_sha256`, 289 files).

## 10. What it means

- **E0 does not meet H1 on R_TEST.** At the median subject it completes two thirds of its laps and 8 of 12 unseen circuits, and its completed laps are 5.2% slower than the s-OCP lap. Both parts of H1 fail, and they fail for every subject.
- **On these 12 circuits, held-out completion matches R-dev completion, and the gaps are smaller.** R-dev was in-sample for widths, but it was not an optimistic stand-in for R_TEST. The circuit-bootstrap predictions from R-dev held for completion and for the orientation differences, but the completion differences came in at the bottom of their intervals (0, against predicted +1 and +3). The predicted gap level was too high.
- **Limits:**
  - five networks from one recipe, each at its final checkpoint;
  - 10 runs per circuit, with the default car at grip 1;
  - the reference is the s-OCP lap, not a TO-OCP or a certified G_20.
- **Next** (protocol Section 14): `outputs/sealed/` may now be opened for the descriptive provenance comparison, which decides nothing and changes no number here. Any further analysis of the lockbox data is POST HOC.
