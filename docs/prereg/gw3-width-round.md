# GW-3: the combined width round (pre-registration, DRAFT)

**Status.** DRAFT, 26 Oct 2026, for review. Nothing here is tagged and no GW-3 run exists.
- After review the lead commits the final text, the analysis script and its tests, and the queue script. The annotated tag `gw3-prereg-v1` is placed on that commit and pushed **before any GW-3 run starts** (Section 8).
- Each run writes its outcome (`eval_rdev.json`) when its training ends, so the tag must precede training, not just the analysis.
- Only the 12 R-dev circuits are evaluated. R_TEST stays sealed and `outputs/sealed/` is not read.

**Items wait on the user** (Section 9). The main one: neither width model met GW-2's bands ([`width-model.md`](../findings/width-model.md)). This draft therefore fixes the stat-model arms as **M = (2a)**, the recommended option, and words GW-4's criterion 4 to match. The user may choose (2b), or no stat-model arm, before the tag. Everything else here holds for any choice.

## 1. Why

- **The critical path.** GW-3 is the follow-through plan's critical-path round (Section 7.4). It is the one GW-3 round that decision #11 allows, and its result decides `generator-v1`'s width model through GW-4's rule (Section 5), fixed here before any run.
- **What it answers:**
  1. **Primary.** Does a width model that copies no evaluation circuit, trained with one fold's widths only, raise completion on the circuits it never saw, against the baseline?
  2. **The open transfer question** (follow-through Section 9.1). The leave-out test's p 0.063 compared networks from different batches, and in v3, where little width effect existed to transfer. In v1 the in-sample effect replicates (H4R S1: +2.6 of 12), but no leave-out runs exist. GW-3 trains copied leave-out runs in v1 and v3, with all arms in one queue on one commit and one machine.
  3. **GW-4's other inputs:** the stat model against the copied model, and its lap time.
- **The confound it removes.** The arms differ only in their width option: the same commit, the same box, the same seeds, interleaved lanes.

## 2. Runs (fixed now)

**Cells.** 6 arms × 2 variants × 5 seeds = 60 runs.

| Arm | Flags (besides the variant's) | Widths seen in training |
|---|---|---|
| `base` (B) | none | default model |
| `cl-f1`, `cl-f2` (copied leave-out) | `--real-widths --real-widths-from <fold f>` | fold f's 6 circuits' profiles, copied |
| `sl-f1`, `sl-f2` (stat-model leave-out) | `--stat-widths --stat-widths-from <fold f>` | M fitted on fold f's 6 circuits |
| `sa` (stat model, all 12) | `--stat-widths` | M fitted on all 12 (the `generator-v1` candidate) |

- **Folds:** the committed leave-out folds (`width-test-loo-plan.md`): fold f1 = `R_DEV[0::2]`, fold f2 = `R_DEV[1::2]`.
  - `<fold f1>` = `BrandsHatch,MoscowRaceway,Nuerburgring,Sepang,YasMarina,Catalunya`.
  - `<fold f2>` = `Hockenheim,Norisring,Oschersleben,Sochi,Melbourne,Zandvoort`.
- **If M = (2b)** (user's choice), the `sl-*` and `sa` arms add `--reexpress`.
- **Variants:**
  - v1: the base grammar, `train_dr.py`'s default;
  - v3: `--tilt-fraction 0.5 --tilt-alpha 2`.
- **Seeds:** 20, 21, 22, 23 and 24 in every cell.
  - No run on disk uses a seed of 15 or more. Seeds 0–4 and 10–14 are used.
  - The lead must keep VG-1 off seeds 20–24. Its β = 0 arm runs the default path, so a shared seed would duplicate GW-3's baseline networks.
- **Names:** `gw3-<variant>-<arm>-s<seed>`, for example `gw3-v1-base-s20` and `gw3-v3-sl-f2-s23`.
- **Command:** `scripts/train_dr.py --steps 200e6 --seed <s> --name <name> <variant flags> <arm flags>`. Every other argument is the default, as in H4R: 2,048 envs, 64 steps, MLP 3 × 256 tanh, `ensemble_heads` 5, bank 4,096, refresh 256, γ 0.995.
  - Each run writes 16 log-spaced checkpoints, the final one at 199,884,800 steps, and `eval_rdev.json`: the default car, 10 timed laps per R-dev circuit from 0.9 × the reference start speed.
- **One commit:**
  - all 60 runs train from the tagged commit;
  - the queue records `git rev-parse HEAD` at launch and refuses a dirty tree;
  - with the width and VG options off, the training path is bit-identical to H4R's. Since H4R's `612ed7a` it has been touched by GW-2's opt-in commits (`bdfd442`, `1bd85ec`, and `3323a0e`'s batch size for (2b) only) and by VG-0's (`43d68fe`: `ppo.make_rollout`, `make_learner` and `make_train`, `adapters._bank_step`'s `info['final_state']`, `checkpoints.make_checkpoint_fn` and `train_dr.py`). Each option defaults off. The tests that hold the default path:
    - generator and bank: `tests/test_stat_widths.py` (`test_default_is_bit_identical`, `test_default_bank_is_bit_identical`, `test_train_dr_default_path_is_unchanged`) and `tests/test_real_widths.py::test_default_is_unchanged`;
    - PPO and env: `tests/test_value_gradient.py` (`test_default_racing_update_reproduces_the_pre_term_values`, `test_default_rollout_records_no_capture_and_defaults_are_off`, `test_train_dr_default_path_passes_no_term`) and `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update`.
    - The lead re-lists the commits at the tag. Any later option must also default off and come with such a test.
- **One machine:** one rented NVIDIA RTX A6000, the cheapest Massed Compute instance, under the standing permission.
  - Set up with `scripts/gpu_setup.sh`.
  - `scripts/gpu_sync.sh` pushes R-dev data only (centrelines and `outputs/ocp/<R-dev>.pkl` for the gap). No R_TEST circuit leaves the Mac.
- **Queue:** `scripts/gpu_queue_gw3.sh`, to be written on the pattern of `scripts/gpu_queue_h4r.sh`.
  - **4 lanes**, XLA memory fraction 0.22 each.
  - **Job order:** seed-major. For seed index i = s − 20, the cell list below is rotated left by i, so each lane meets every cell:
    `v1-base, v3-base, v1-cl-f1, v3-cl-f1, v1-sl-f1, v3-sl-f1, v1-cl-f2, v3-cl-f2, v1-sl-f2, v3-sl-f2, v1-sa, v3-sa`.
  - Lane k runs jobs k, k + 4, k + 8, …, so every round of 4 concurrent runs mixes arms and variants.
  - **Retry:** a run that fails is re-run once from scratch with the same arguments, as H4R's queue does (its log kept as `.log.1`). It is **missing** only if it fails twice.
- **Manifest:** after pull-back, `outputs/gw3/manifest.json` is committed. It holds every run's config, timesteps, and the sha256 of its final checkpoint and `eval_rdev.json`; the queue log's sha256; the retry record; and the launch commit.
  - The analysis checks it before reading anything, and refuses a run whose config differs from its cell's flags.

## 3. Outcomes and the analysis (fixed now)

- **Completion.** A circuit is completed when `eval_rdev.json` gives completion > 0.5 (more than 5 of 10 runs finish). C(network, circuits) counts completed circuits.
- **Held-out circuits.** For a fold-f run (`cl-f`, `sl-f`), the other fold's 6 circuits, H_f.
- **Units.** One value per network or network pair, all out of 12:
  - **Baseline unit:** b_s = C(`base`, seed s, all 12).
  - **Leave-out unit:** u_s = C(`*-f1`, seed s, H_f1) + C(`*-f2`, seed s, H_f2). Seed s of fold f1 is paired with seed s of fold f2, so every R-dev circuit is judged once, by a network that never saw its widths.
  - Each unit is a different network or pair. Values from one baseline network are not entered twice. The leave-out analysis of 21 Oct 2026 (`width-test.md`, Section 7) did enter them twice: the same baselines sat in both folds' strata and were permuted independently, which understates the variance.
  - **All-12 unit:** a_s = C(`sa`, seed s, all 12).
- **Primary test P1: `sl` above `base` on held-out circuits.**
  - Statistic T = Σ over variants of [mean_s u_s(sl) − mean_s b_s].
  - One-sided exact permutation test, labels exchanged within each variant: C(10, 5)² = 63,504 relabellings. Ties count toward p.
  - **Confirm** if p ≤ 0.05. **Refute** if the opposite one-sided p ≤ 0.05. Otherwise **inconclusive**.
- **Missing runs.**
  - A missing run removes its unit (both halves of a pair are needed). Every relabelling of the remaining units is enumerated.
  - A test is **not testable** with fewer than 3 units in either arm of either variant. Not testable counts as not confirmed.
- **Secondary** (exact one-sided permutation tests of the same construction at 0.05, unadjusted, labelled secondary):
  - **S1:** `cl` above `base` on held-out circuits (u(cl) against b), both variants (T as in P1). This replicates the leave-out test with same-code baselines.
    - **S1-v1 and S1-v3**, also secondary exact tests at 0.05, unadjusted: the same comparison within one variant, T = mean_s u_s(cl) − mean_s b_s, one-sided, with C(10, 5) = 252 relabellings. S1-v1 is reported first: it is the transfer reading follow-through Section 9.1 asks for. The not-testable rule applies per variant.
  - **S2:** `sa` above `base` on all 12 (a against b). It is in-sample for M's summaries.
  - **S3:** `sl` against `cl` on held-out circuits, two-sided, descriptive. The mean difference per (variant, fold) cell, with a 95% percentile bootstrap over seeds (20,000 draws, seed 0). GW-4's criterion 2 reads this point estimate.
- **Lap time (GW-4's criterion 3; IN-4's form).**
  - Each network's median gap is the median over its completed R-dev circuits of (median lap / s-OCP lap − 1), in points.
  - A network with no completed circuit has no median gap. It is left out of Δgap, and the number left out is reported per arm and variant. If every `sa` or every `base` network is left out, Δgap is undefined and criterion 3 does not hold.
  - Δgap = mean over `sa` networks − mean over `base` networks, **both variants pooled** (10 against 10, less any left out).
  - Also reported, descriptively:
    - per variant;
    - on the circuits both networks complete (paired by seed);
    - the per-circuit table of `interp-leftovers.md` Section 6.1.
- **Descriptive:**
  - per variant and per fold, the held-out and all-12 counts per arm;
  - per circuit, the share of networks completing each circuit, with Norisring, Yas Marina, Sochi and Melbourne named;
  - the per-seed pairs (same-seed `sl` and `base`);
  - the number of networks per arm with C ≥ 10 of 12 at 200M, labelled **"not the gate"** (Section 6).
- **The analysis script.** `scripts/gw3_analysis.py`, with tests on planted data, committed before the tag. It reads only the manifest's files, runs once (it refuses if its output exists or was ever committed) and writes `outputs/gw3/analysis.json` with a provenance block.
- **Power** (computed for this draft, by simulation with H4R's baseline means and SDs: v1 5.0 ± 1.58, v3 7.2 ± 1.10; 1,500 draws):

  | True gain (v1, v3), circuits of 12 | P(P1 confirms) |
  |---|---|
  | 0, 0 | 0.03 |
  | +1.3, +0.3 (about half of H4R's in-sample v1 gain transfers) | 0.25 |
  | +1.3, +0.6 | 0.37 |
  | +2.6, +0.6 (the whole in-sample gain transfers) | 0.70 |

  At 5 seeds per cell (#13), an inconclusive P1 is the likeliest result unless most of the in-sample gain transfers. 10 seeds per cell (seeds 20–29) would roughly double the A6000 time (Section 7). That is the user's call before the tag.

## 4. What counts as the result

| Outcome | Wording |
|---|---|
| P1 confirms | A width model fitted on one fold's circuits, which copies none, raises completion on the other fold's circuits (pre-registered, one-sided p = …) |
| P1 inconclusive | Not shown at this power; the observed difference and S1 reported |
| P1 refutes | The stat model lowers held-out completion |

S1 is reported whatever P1 shows. It answers the transfer question of follow-through Section 9.1 on same-code networks.

## 5. GW-4's adoption rule (fixed before GW-3)

GW-4 **adopts M, fitted on all 12 R-dev circuits, as `generator-v1`'s width model** if all five criteria hold (follow-through Section 7.4):
1. **P1 confirms** (Section 3).
2. **Not far below the copied model.** D2 ≥ −0.5, where D2 is the mean over the 4 (variant, fold) cells of [mean_s C(`sl-f`, H_f) − mean_s C(`cl-f`, H_f)], in circuits of 6. This is a point estimate (S3).
3. **Lap time.** Δgap < 1.0 point (Section 3; point estimate, both variants pooled; networks with no completed circuit left out, as Section 3 fixes). IN-4's copied-width reference for this statistic is +0.30 points pooled (+0.52 in v1, +0.09 in v3), with a 95% half-width of about 0.45 points at 10 against 10 (`interp-leftovers.md`, Section 6.3).
4. **The width-report bands. PENDING THE USER** (Section 9). Neither model met them as fixed.
   - Drafted for M = (2a): "(2a)'s committed width report (`outputs/width-report/report.json`, sha256 `5871664be5a6ff6ac882e3a38e131660fe16676f37ddd0a99ffba7294a42f014`) passes every v1 check in both folds and every no-copy check. Its v3 misses are limited to `share_1m` and `spectrum:8-16`."
   - The report is force-added to git (as `outputs/vg0/` is), with (2b)'s `report-2b.json` (sha256 `89ffd2a3a1c49de902e880c44f4e0dfec38ed790ebd2551435778e2c89f37755`). Both sha256s are also in `width-model.md`. GW-4 reads the report only after checking its sha256 against this line; a mismatch stops GW-4 for an amendment.
   - The recorded report already meets this wording. Adopting it is a deviation from the width-model plan's verdict rule, and it must be logged before the tag.
5. **The default stays bit-identical** at the `generator-v1` commit. These tests pass there:
   - `tests/test_stat_widths.py::test_default_is_bit_identical`;
   - `::test_default_bank_is_bit_identical` and `::test_train_dr_default_path_is_unchanged`;
   - `tests/test_real_widths.py::test_default_is_unchanged`;
   - `tests/test_reexpress.py::test_default_and_2a_unchanged_and_2b_fit`, if (2b) is in the tree;
   - and, for the PPO and env path that E0 also trains through, `tests/test_value_gradient.py`'s three default-path tests and `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update` (Section 2).

**What adoption means:**
- **If all five hold:** `generator-v1` sets `stat_widths` (M fitted on all 12) as E0's training width model. The default stays the frozen default.
- **Otherwise:** `generator-v1` is tagged with the default widths. #11 allows no further width-model round. Copied R-dev widths are not a fallback unless the user decides #12 for copying.
- **E0's curvature variant (v1 or v3)** is not GW-4's to choose, and it may not be chosen on GW-3's per-variant results. It is fixed before the tag, without GW-3 data (Section 9, item 5), and recorded here. The rule adopts a width model for either, read pooled over both.
- **GW-5 and GW-6** (#11's optional lever round and 1B pilot) may run after GW-3 on the chosen recipe. They do not change this rule.

## 6. The 80% gate (decision #10; recorded here, not read here)

- **The gate:** E0 (5 seeds × 1B steps, on `generator-v1` and `sim-v1`) passes if its median seed completes at least 10 of the 12 R-dev circuits at its final checkpoint.
- **No checkpoint is chosen on R-dev.** GW-1 showed that R-dev completion peaks at slow early checkpoints.
- **GW-3 does not read it.** Its 200M runs are not E0. The per-arm count of networks with C ≥ 10 is descriptive and labelled "not the gate".

## 7. A6000 time (from recorded rates)

- **Recorded rates.** At 4 lanes on one A6000, a 200M `train_dr.py` run costs about 6.5 min of box time.
  - H4R: 20 runs in 2 h 13 min including one retry, 6.65 min per run.
  - The leave-out round: 8 runs in 51 min, 6.4 min per run (`runs/width-loo.log`).
  - The copied option did not slow training (H4R: 147–179k steps/s against 145–161k for baselines, retry excluded).
  - (2a)'s bank refresh costs what the default's does on the Mac: 0.08 s per 256 strips (`width-model.md`, Section 2).
- **GW-3 with M = (2a):**
  - 60 runs × 6.5–6.65 min = 6.5–6.7 h of training;
  - plus about 10 min of setup and 5 min of pull-back;
  - about **7 A6000 hours** in all, as one box session.
  - The follow-through plan's "about 14 A6000 h" for v1 + v3 assumed 60–70 runs per variant. At #13's 5 seeds per cell the round is 60 runs in all. 10 seeds per cell would be about 14 h.
- **With M = (2b):** unmeasured on the GPU. On the Mac CPU its bank refresh is about 200 times slower (15.9 s per 256 strips).
  - Before its queue starts, one `gw3-v1-sa-s20`-shaped run of 2M steps measures the cost on the box.
  - If it adds more than 50% per run, the lead brings it to the user before continuing.
- **Analysis:** minutes on the Mac.

## 8. Procedure

1. Review this draft. The user decides Section 9's items.
2. Write and test `scripts/gw3_analysis.py` (planted data only) and `scripts/gpu_queue_gw3.sh`. Commit them with the final text.
3. Tag `gw3-prereg-v1` (annotated) on that commit and push it. The tag object's sha is recorded in the findings.
4. Rent the box, set it up at the tagged commit, and run the queue. Pull back, checking every final checkpoint's sha256 against the box's. Delete the box.
5. Commit the manifest.
6. Run the analysis once. Write `docs/findings/gw3.md`.
7. GW-4 applies Section 5's rule.

**Deviations.** Any change after the tag is logged in the findings with its date and reason. A change to an outcome, test, seed or rule after any run has started makes the affected result exploratory.

## 9. Open items before the tag

1. **M and criterion 4 (the user).**
   - (2a), recommended, with criterion 4 as drafted.
   - Or (2b): the `sl` and `sa` arms add `--reexpress`, criterion 4 is restated for (2b)'s report (`report-2b.json`: fold f2 passes fully; fold f1 misses `spectrum:8-16` in both variants and `share_1m` in v3), and its GPU cost is measured first.
   - Or no stat-model arm. GW-3 is then `base`, `cl-f1` and `cl-f2` only: 30 runs, about 3.5 h. P1 becomes S1, and GW-4 falls to default widths unless #12 decides copying.
2. **Seeds per cell (the user).** 5 (#13; about 7 h; power about 0.25–0.37 for a half-transferred gain) or 10 (about 14 h).
3. **#12 (the user):** whether copied R-dev widths may be `generator-v1`'s fallback.
4. **Seed ranges (the lead):** keep VG-1 and any other `train_dr.py` round off seeds 20–24 (20–29 if 10 seeds).
5. **E0's curvature variant (the user, before the tag).** Decided on what exists now, never on GW-3's results; once tagged, a change is a deviation that makes E0's recipe exploratory.
   - Recommended: **v1**, `train_dr.py`'s default. It is the variant where the copied-width effect replicates (H4R S1, +2.6 of 12; follow-through Section 7.4 requires v1 in GW-3 for that reason), and the curvature levers did not differ (v1/v2/v3 at 5.4/5.8/5.8; follow-through Section 8).
   - The alternative is v3 (`--tilt-fraction 0.5 --tilt-alpha 2`), whose H4R baseline was higher (7.2 against 5.0 circuits of 12) and whose width effect was smaller.
6. **To build (the lead):** `scripts/gw3_analysis.py` with tests, `scripts/gpu_queue_gw3.sh`, and the manifest writer.
