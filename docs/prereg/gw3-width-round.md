# GW-3: the combined width round (pre-registration, DRAFT)

**Status.** DRAFT, 26 Oct 2026, for review; revised 27 Oct 2026 with the user's decisions (Section 9), and its machine text (Sections 2, 7 and 8) revised the same day for the boxes the user allowed. Nothing here is tagged and no GW-3 run exists.
- After review the lead commits the final text, the analysis script and its tests, and the queue script. The annotated tag `gw3-prereg-v1` is placed on that commit and pushed **before any GW-3 run starts** (Section 8).
- Each run writes its outcome (`eval_rdev.json`) when its training ends, so the tag must precede training, not just the analysis.
- Only the 12 R-dev circuits are evaluated. R_TEST stays sealed and `outputs/sealed/` is not read.

**The user's items are decided** (27 Oct 2026; Section 9). Neither width model met GW-2's bands ([`width-model.md`](../findings/width-model.md)). The user chose:
- **M = (2a)**, with GW-4's criterion 4 restated (Section 5). That departs from the width-model plan's bands, and the deviation is logged in Section 10, before the tag.
- **10 seeds per cell**, seeds 20–29: 120 runs, about 14 A6000 hours (Section 7).
- **E0's curvature variant: v1** (Section 5).
- **#12:** copied R-dev widths only as a conditional fallback, with in-sample labelling (Section 5).

Item 4 of Section 9 remains for the lead. Item 6, the build, is done (27 Oct 2026).

## 1. Why

- **The critical path.** GW-3 is the follow-through plan's critical-path round (Section 7.4). It is the one GW-3 round that decision #11 allows, and its result decides `generator-v1`'s width model through GW-4's rule (Section 5), fixed here before any run.
- **What it answers:**
  1. **Primary.** Does a width model that copies no evaluation circuit, trained with one fold's widths only, raise completion on the circuits it never saw, against the baseline?
  2. **The open transfer question** (follow-through Section 9.1). The leave-out test's p 0.063 compared networks from different batches, and in v3, where little width effect existed to transfer. In v1 the in-sample effect replicates (H4R S1: +2.6 of 12), but no leave-out runs exist. GW-3 trains copied leave-out runs in v1 and v3, with all arms in one round on one commit and one GPU model.
  3. **GW-4's other inputs:** the stat model against the copied model, and its lap time.
- **The confound it removes.** The arms differ only in their width option: the same commit, the same round of identical GPUs (every box runs all 12 cells of its seeds), the same seeds, interleaved lanes.

## 2. Runs (fixed now)

**Cells.** 6 arms × 2 variants × 10 seeds = 120 runs (decided 27 Oct 2026; Section 9, item 2).

| Arm | Flags (besides the variant's) | Widths seen in training |
|---|---|---|
| `base` (B) | none | default model |
| `cl-f1`, `cl-f2` (copied leave-out) | `--real-widths --real-widths-from <fold f>` | fold f's 6 circuits' profiles, copied |
| `sl-f1`, `sl-f2` (stat-model leave-out) | `--stat-widths --stat-widths-from <fold f>` | M fitted on fold f's 6 circuits |
| `sa` (stat model, all 12) | `--stat-widths` | M fitted on all 12 (the `generator-v1` candidate) |

- **Folds:** the committed leave-out folds (`width-test-loo-plan.md`): fold f1 = `R_DEV[0::2]`, fold f2 = `R_DEV[1::2]`.
  - `<fold f1>` = `BrandsHatch,MoscowRaceway,Nuerburgring,Sepang,YasMarina,Catalunya`.
  - `<fold f2>` = `Hockenheim,Norisring,Oschersleben,Sochi,Melbourne,Zandvoort`.
- **M = (2a)** (decided 27 Oct 2026; Section 9, item 1). The `sl-*` and `sa` arms use `--stat-widths` without `--reexpress`. No arm uses (2b).
- **Variants:**
  - v1: the base grammar, `train_dr.py`'s default;
  - v3: `--tilt-fraction 0.5 --tilt-alpha 2`.
- **Seeds:** 20 to 29 (ten seeds) in every cell.
  - No run on disk uses a seed of 15 or more. Seeds 0–4 and 10–14 are used.
  - The lead must keep VG-1 off seeds 20–29. Its β = 0 arm runs the default path, so a shared seed would duplicate GW-3's baseline networks. VG-1's draft uses seeds 15–16 (pilot) and 30–49 (main run).
- **Names:** `gw3-<variant>-<arm>-s<seed>`, for example `gw3-v1-base-s20` and `gw3-v3-sl-f2-s29`.
- **Command:** `scripts/train_dr.py --steps 200e6 --seed <s> --name <name> <variant flags> <arm flags>`. Every other argument is the default, as in H4R: 2,048 envs, 64 steps, MLP 3 × 256 tanh, `ensemble_heads` 5, bank 4,096, refresh 256, γ 0.995.
  - Each run writes 16 log-spaced checkpoints, the final one at 199,884,800 steps, and `eval_rdev.json`: the default car, 10 timed laps per R-dev circuit from 0.9 × the reference start speed.
- **One commit:**
  - all 120 runs train from the tagged commit, checked out (detached) on every box before launch;
  - each box's manifest (`scripts/round_manifest.py write`, which the queue runs when it ends) records HEAD, whether the tracked tree was clean and the tags at HEAD. `round_manifest.py check` and `scripts/gw3_analysis.py manifest` refuse a box whose HEAD is not the tag's commit or whose tree was not clean;
  - with the width and VG options off, the training path is bit-identical to H4R's. Since H4R's `612ed7a` it has been touched by GW-2's opt-in commits (`bdfd442`, `1bd85ec`, and `3323a0e`'s batch size for (2b) only) and by VG-0's (`43d68fe`: `ppo.make_rollout`, `make_learner` and `make_train`, `adapters._bank_step`'s `info['final_state']`, `checkpoints.make_checkpoint_fn` and `train_dr.py`). Each option defaults off. The tests that hold the default path:
    - generator and bank: `tests/test_stat_widths.py` (`test_default_is_bit_identical`, `test_default_bank_is_bit_identical`, `test_train_dr_default_path_is_unchanged`) and `tests/test_real_widths.py::test_default_is_unchanged`;
    - PPO and env: `tests/test_value_gradient.py` (`test_default_racing_update_reproduces_the_pre_term_values`, `test_default_rollout_records_no_capture_and_defaults_are_off`, `test_train_dr_default_path_passes_no_term`) and `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update`.
    - The lead re-lists the commits at the tag. Any later option must also default off and come with such a test.
- **One round of identical A6000s** (the user allowed more than one box on 27 Oct 2026, under the Phase 3 approval of the same day; Section 7):
  - **the plan:** one `massedcompute_A6000_plusx4` box (4 × NVIDIA RTX A6000 48 GB, 54 vCPU), each run on one GPU, 4 lanes per GPU, 16 runs at a time;
  - **if that type is not available:** K single-GPU `massedcompute_A6000_plus` boxes, K dividing 10 (1, 2, 5 or 10). Box i (0 to K − 1) runs `SHARD=i/K`: the whole seeds whose index s − 20 is i mod K, every cell of each;
  - every box of the round has the same GPU model and trains from the one commit. Each box's manifest records its type, GPU model and count, lanes per GPU and shard, and `round_manifest.py check` refuses boxes whose GPU models differ.
  - Set up with `scripts/gpu_setup.sh`.
  - `scripts/gpu_sync.sh` pushes R-dev data only (centrelines and `outputs/ocp/<R-dev>.pkl` for the gap). No R_TEST circuit leaves the Mac.
- **Queue:** `scripts/gpu_round.sh gw3 scripts/gw3_runs.py`, the shared round queue (`46cdd81`), with GW-3's run list in `scripts/gw3_runs.py`.
  - **4 lanes per GPU**, XLA memory fraction 0.22 each (round(0.9 / 4, 2), as in H4R). Job j of a box's list goes to lane j mod L, with L = 4 × the box's GPUs; lane l runs on GPU l mod GPUs, and each lane runs its jobs in order.
  - **Job order:** seed-major, 12 jobs per seed and 120 in all. For seed index i = s − 20 (0 to 9), the cell list below is rotated left by i:
    `v1-base, v3-base, v1-cl-f1, v3-cl-f1, v1-sl-f1, v3-sl-f1, v1-cl-f2, v3-cl-f2, v1-sl-f2, v3-sl-f2, v1-sa, v3-sa`.
  - **Balance** (corrected 27 Oct 2026 for the boxes; `tests/test_gw3_analysis.py` checks each case):
    - one GPU (all 120 runs on one single-GPU box): lane k runs jobs k, k + 4, k + 8, …, 30 jobs per lane, and meets every cell 2 or 3 times;
    - the x4 box (16 lanes): each GPU runs 30 jobs on its 4 lanes, every cell 2 or 3 times. A lane runs 7 or 8 jobs, each a different cell, so a lane does not meet every cell;
    - two GPUs (8 lanes): each GPU runs every cell 5 times; a lane runs 15 jobs over 11 of the 12 cells;
    - K single boxes: each box runs 10/K seeds, every cell 10/K times;
    - in every case, each wave of L consecutive jobs (the runs that start together) holds both variants and at least 2 arms.
  - **Retry:** a run that fails is re-run once from scratch with the same arguments, as H4R's queue does (its log kept as `.log.1`). It is **missing** only if it fails twice.
- **Manifest:**
  - On each box the queue writes `runs/gw3-manifest-<i>of<K>.json` when it ends: the box's HEAD, tree and tags, GPUs, type, lanes and shard, and per run its status (done, retried or failed), start and end times, lane, GPU and config sha256. The lead then writes the box's sha256 list, `runs/gw3-sums-<i>of<K>.txt`, of every run's `config.json`, `eval_rdev.json` and final checkpoint (Section 8).
  - After pull-back, `round_manifest.py check --tag gw3-prereg-v1` checks the boxes' manifests and every run's recorded arguments against its command. Then `scripts/gw3_analysis.py manifest` writes `outputs/gw3/manifest.json`, which is committed. It holds every run's config (sha256 and arguments), timesteps, and the sha256 of its final checkpoint and `eval_rdev.json`, each checked against the box's list; each box's manifest, queue log, status file and list (sha256), type, GPUs, lanes and shard; the retry record; the check's report; and the launch commit.
  - It refuses a run whose config differs from its cell's flags or from `train_dr.py`'s defaults at the tag. It also refuses a run missing for any reason other than failing twice, unless the lead names it with `--accept-missing RUN=REASON`, which the manifest records and the findings log as a deviation.
  - The analysis checks the manifest before reading anything.

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
  - One-sided exact permutation test, labels exchanged within each variant. Ties count toward p: p is the share of relabellings with T* ≥ T, the observed one included, and the opposite p the share with T* ≤ T.
  - **Exact p by counting.** At 10 units per arm and variant there are C(20, 10)² = 184,756² ≈ 3.4 × 10¹⁰ joint relabellings, too many to list. They are counted instead, which gives exactly the p that listing them would:
    - per variant, a dynamic programme over the integer units (0 to 12) counts the relabellings that give each sum of the `sl` arm's units;
    - T is linear in the two variants' `sl` sums, so the number of relabellings with T* ≥ T (or ≤ T) is a sum, over every pair of sums, of the product of their counts;
    - T is compared in integer arithmetic (scaled by the least common multiple of n_sl · n_base over the variants), so no comparison is approximate.
    - `scripts/gw3_analysis.py`'s tests check the count against full enumeration on planted data with up to 5 units per arm, as `scripts/gw3_power.py` does.
  - **Confirm** if p ≤ 0.05. **Refute** if the opposite one-sided p ≤ 0.05. Otherwise **inconclusive**.
- **Missing runs.**
  - A missing run removes its unit (both halves of a pair are needed). The p is counted over every relabelling of the remaining units, by the same method (arms of unequal size change only the integer weights).
  - A test is **not testable** with fewer than 3 units in either arm of either variant. Not testable counts as not confirmed.
- **Secondary** (exact one-sided permutation tests of the same construction at 0.05, unadjusted, labelled secondary; each p counted as for P1):
  - **S1:** `cl` above `base` on held-out circuits (u(cl) against b), both variants (T as in P1; C(20, 10)² relabellings). This replicates the leave-out test with same-code baselines.
    - **S1-v1 and S1-v3**, also secondary exact tests at 0.05, unadjusted: the same comparison within one variant, T = mean_s u_s(cl) − mean_s b_s, one-sided, with C(20, 10) = 184,756 relabellings (few enough to list; counted by the same method). S1-v1 is reported first: it is the transfer reading follow-through Section 9.1 asks for. The not-testable rule applies per variant.
    - **S1-v1 is also #12's transfer condition** (Section 5). S1 and S1-v3 are reported but do not decide #12.
  - **S2:** `sa` above `base` on all 12 (a against b; C(20, 10)² relabellings). It is in-sample for M's summaries.
  - **S3:** `sl` against `cl` on held-out circuits, two-sided, descriptive. The mean difference per (variant, fold) cell, with a 95% percentile bootstrap over seeds (20,000 draws, seed 0). GW-4's criterion 2 reads this point estimate.
- **Lap time (GW-4's criterion 3; IN-4's form).**
  - Each network's median gap is the median over its completed R-dev circuits of (median lap / s-OCP lap − 1), in points.
  - A network with no completed circuit has no median gap. It is left out of Δgap, and the number left out is reported per arm and variant. If every `sa` or every `base` network is left out, Δgap is undefined and criterion 3 does not hold.
  - Δgap = mean over `sa` networks − mean over `base` networks, **both variants pooled** (20 against 20, less any left out).
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
- **Power**, by simulation with H4R's baseline means and SDs (v1 5.0 ± 1.58, v3 7.2 ± 1.10):
  - **Method.** Each unit is drawn independently as round(N(mean + gain, SD)), clipped to 0–12; the `sl` units take the row's gain. P1 is applied exactly as above.
  - The 26 Oct draft computed the 5-seed column with 1,500 draws and committed no script. `scripts/gw3_power.py` re-implements the method.
  - **It reproduces the 5-seed column within simulation error:** 0.027, 0.260, 0.345 and 0.711 at 1,500 draws (numpy seed 0 plus the row index), and 0.034, 0.272, 0.353 and 0.707 at 20,000 draws (`--draws-5 20000`). Each of the draft's figures is within 2 standard errors of the 20,000-draw value (one standard error of a 1,500-draw estimate is 0.005–0.012 here).
  - **The 10-seed column:** 20,000 draws, numpy seed 20261027 plus the row index; standard error at most 0.004.

  | True gain (v1, v3), circuits of 12 | P(P1 confirms), 5 seeds per cell (26 Oct draft) | P(P1 confirms), 10 seeds per cell |
  |---|---|---|
  | 0, 0 | 0.03 | 0.04 |
  | +1.3, +0.3 (about half of H4R's in-sample v1 gain transfers) | 0.25 | 0.50 |
  | +1.3, +0.6 | 0.37 | 0.63 |
  | +2.6, +0.6 (the whole in-sample gain transfers) | 0.70 | 0.96 |

  At 10 seeds per cell (decided 27 Oct 2026), P1 confirms with about even odds if half of H4R's in-sample v1 gain transfers, and with probability about 0.96 if all of it does. If less than half transfers, an inconclusive P1 is still the likeliest result. The size stays below 0.05 because integer counts tie.
  - S1-v1 (#12's condition, Section 5), from the same script at 10 seeds and 20,000 draws: P(confirms) 0.035 with no gain, 0.48 with a v1 gain of +1.3 and 0.96 with +2.6.

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
3. **Lap time.** Δgap < 1.0 point (Section 3; point estimate, both variants pooled, 20 against 20; networks with no completed circuit left out, as Section 3 fixes). IN-4's copied-width reference for this statistic is +0.30 points pooled (+0.52 in v1, +0.09 in v3), with a 95% half-width of about 0.45 points at 10 against 10 (`interp-leftovers.md`, Section 6.3).
4. **The width report, restated** (decided by the user on 27 Oct 2026; a deviation from the width-model plan's bands, logged in Section 10 before the tag). Every v1 check in (2a)'s committed width report (`outputs/width-report/report.json`, sha256 `5871664be5a6ff6ac882e3a38e131660fe16676f37ddd0a99ffba7294a42f014`) passes in both folds, the v3 misses are limited to `share_1m` (both folds) and `spectrum:8-16` (fold f1), and every no-copy check (f1, f2 and all 12, in both variants) passes.
   - The no-copy condition is kept from the drafted criterion 4: the restatement relaxes only the band checks (the lead, 27 Oct 2026).
   - In the report's check names, those misses are `f1/v3/share_1m`, `f2/v3/share_1m` and `f1/v3/spectrum:8-16`.
   - The report is force-added to git (as `outputs/vg0/` is), with (2b)'s `report-2b.json` (sha256 `89ffd2a3a1c49de902e880c44f4e0dfec38ed790ebd2551435778e2c89f37755`). Both sha256s are also in `width-model.md`. GW-4 reads the report only after checking its sha256 against this line; a mismatch stops GW-4 for an amendment.
   - The committed report already meets this criterion: its failed list is exactly the three checks above, and every other of its 90 checks passes, the six no-copy checks among them (`width-model.md`, Sections 1 and 4).
5. **The default stays bit-identical** at the `generator-v1` commit. These tests pass there:
   - `tests/test_stat_widths.py::test_default_is_bit_identical`;
   - `::test_default_bank_is_bit_identical` and `::test_train_dr_default_path_is_unchanged`;
   - `tests/test_real_widths.py::test_default_is_unchanged`;
   - `tests/test_reexpress.py::test_default_and_2a_unchanged_and_2b_fit`, if (2b) is in the tree;
   - and, for the PPO and env path that E0 also trains through, `tests/test_value_gradient.py`'s three default-path tests and `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update` (Section 2).

**What adoption means:**
- **If all five hold:** `generator-v1` sets `stat_widths` (M fitted on all 12) as E0's training width model. The default stays the frozen default.
- **Otherwise: #12's rule** (decided by the user on 27 Oct 2026; Section 9, item 3). #11 allows no further width-model round.
  - `generator-v1` uses copied R-dev width profiles (`--real-widths`) **if and only if both** of these hold:
    - (2a) fails this adoption rule (any of the five criteria fails); and
    - GW-3 shows that copied widths transfer to held-out circuits: **S1-v1 confirms** (Section 3). That is `cl-f1` and `cl-f2` above `base` on held-out units within v1, one-sided exact p ≤ 0.05. Not testable counts as not confirmed.
  - **Otherwise** (either condition fails) `generator-v1` uses the default widths.
  - The rule is deterministic: the user's answer was to copy only if GW-3 shows transfer, and it is written so that nothing is left to choose after GW-3's data (the lead, 27 Oct 2026).
  - **Why S1-v1, not S1 pooled.** The copied widths would train E0, which is v1 (below). S1-v1 is this draft's pre-specified transfer reading (follow-through Section 9.1). A pooled S1 could confirm on v3's transfer alone. S1 and S1-v3 are reported but do not decide #12. S1-v1's power is in Section 3.
  - **If copied widths are used,** every R-dev number for E0 is labelled **"in-sample for widths"** wherever it is reported (the findings, STATUS and the paper among them). E0 would then have trained on the R-dev circuits' own width profiles. The label covers every such number, among them:
    - the 80% gate's result (#10; the median seed's count of completed R-dev circuits, Section 6) and each seed's count;
    - per-circuit R-dev completion, lap times and gaps to the s-OCP lap;
    - every comparison or figure that uses E0's R-dev results.
  - **The lockbox (R_TEST) is unaffected either way.**
- **E0's curvature variant is v1**, `train_dr.py`'s default (decided by the user on 27 Oct 2026, before the tag and without GW-3 data; Section 9, item 5).
  - It is not GW-4's to choose, and it is not chosen on GW-3's per-variant results. Once tagged, a change is a deviation that makes E0's recipe exploratory.
  - Criteria 1–3 stay pooled over both variants, as fixed before the variant was decided.
- **GW-5 and GW-6** (#11's optional lever round and 1B pilot) may run after GW-3 on the chosen recipe. They do not change this rule.

## 6. The 80% gate (decision #10; recorded here, not read here)

- **The gate:** E0 (5 seeds × 1B steps in v1, on `generator-v1` and `sim-v1`) passes if its median seed completes at least 10 of the 12 R-dev circuits at its final checkpoint.
- **If `generator-v1` uses copied widths** (#12, Section 5), the gate's result is labelled "in-sample for widths" wherever it is reported.
- **No checkpoint is chosen on R-dev.** GW-1 showed that R-dev completion peaks at slow early checkpoints.
- **GW-3 does not read it.** Its 200M runs are not E0. The per-arm count of networks with C ≥ 10 is descriptive and labelled "not the gate".

## 7. A6000 time (from recorded rates)

- **Recorded rates.** At 4 lanes on one A6000, a 200M `train_dr.py` run costs about 6.5 min of box time.
  - H4R: 20 runs in 2 h 13 min including one retry, 6.65 min per run.
  - The leave-out round: 8 runs in 51 min, 6.4 min per run (`runs/width-loo.log`).
  - The copied option did not slow training (H4R: 147–179k steps/s against 145–161k for baselines, retry excluded).
  - (2a)'s bank refresh costs what the default's does on the Mac: 0.08 s per 256 strips (`width-model.md`, Section 2).
- **GW-3 with M = (2a) at 10 seeds per cell** (decided 27 Oct 2026), in GPU-hours: each GPU of a box counts from the box's creation to its deletion, idle GPUs included.
  - Training: 120 runs × 6.4–6.65 min = 768–798 GPU-minutes, 12.8–13.3 GPU-hours, at 4 lanes per GPU.
  - **One x4 box (the plan).** Each GPU runs 30 jobs on its 4 lanes (8, 8, 7 and 7).
    - After 7 rounds of 4 runs per GPU (179–186 min), two runs per GPU remain, on 2 lanes. They take 13–27 min, depending on how much a GPU running 2 runs speeds each of them up.
    - The queue takes 3.2–3.55 h: 12.8–14.2 GPU-hours with the idle lanes.
    - Setup (about 10 min) and pull-back (about 5 min) bill all 4 GPUs: 1.0 GPU-hour.
    - In all **13.8–15.2 GPU-hours, 3.45–3.8 h of box time**, about $9.50–10.40 at $2.74/h.
  - **K single boxes (the fallback).** 12.8–13.3 GPU-hours of queue, plus about 15 min of setup and pull-back per box:
    - K = 1: 13.05–13.55 GPU-hours, 13.05–13.55 h on the box;
    - K = 2: 13.3–13.8 GPU-hours, 6.65–6.9 h per box;
    - K = 5: 14.05–14.55 GPU-hours, 2.8–2.9 h per box;
    - K = 10: 15.3–15.8 GPU-hours, 1.5–1.6 h per box.
    - The account has allowed one instance at a time. If that still holds, the boxes run one after another.
  - About **14 GPU-hours** in all, as approved: every case above is 13.05–15.8 GPU-hours, under the stop rule's 16.8. This matches the follow-through plan's "about 14 A6000 h" (#13). The user's 10 seeds per cell supersede #13's 5.
  - **Not measured on the x4 box.** The rates come from the single-GPU box, with 12 vCPU for 4 runs. The x4 box has 54 vCPU for 16 runs, 3.4 per run against 3. The lead compares the first runs' steps per second (`metrics.csv`) with H4R's 145–179k early in the round.
- **Approved** by the user on 27 Oct 2026: about 14 GPU-hours for this round, within Phase 3's A6000 budget (GW-3 about 14, VG-1 8–14, P4-B2 about 1; about 23–29 GPU-hours in all), on Massed Compute A6000 boxes, more than one allowed (follow-through Section 11, fourth round), each deleted when done. The lead checks the hourly price on Brev before creating a box.
- **The lead's stop rule**, in GPU-hours with idle GPUs included: if the round's projected total exceeds its approved estimate by more than 20% (about 16.8 GPU-hours here; on the x4 box about 4.2 h from creation to deletion), the lead pauses the round and asks the user. `round_manifest.py check` prints each box's queue time × its GPUs; setup and pull-back are added to that.
- **(2b) was not chosen** (Section 9, item 1), so its unmeasured GPU cost (on the Mac CPU its bank refresh is about 200 times slower than (2a)'s) does not arise.
- **Analysis:** minutes on the Mac.

## 8. Procedure

1. Review this draft. The user's items in Section 9 were decided on 27 Oct 2026; the lead's item 4 remains.
2. Write and test `scripts/gw3_runs.py` and `scripts/gw3_analysis.py` (planted data only; done 27 Oct 2026, Section 9 item 6). They run on the shared queue, `scripts/gpu_round.sh` and `scripts/round_manifest.py` (`46cdd81`). Commit them with the final text.
3. Tag `gw3-prereg-v1` (annotated) on that commit and push it. The tag object's sha is recorded in the findings.
4. Rent the box (or the K boxes, Section 2). On each, from the Mac:

   ```
   ssh <box> 'bash -s' < scripts/gpu_setup.sh && scripts/gpu_sync.sh <box>
   ssh <box> 'cd ~/apexzero && git fetch --tags && git checkout --detach gw3-prereg-v1 &&
     git describe --exact-match && test -z "$(git status --porcelain --untracked-files=no)"'
   ssh <box> 'cd ~/apexzero && DRY_RUN=1 bash scripts/gpu_round.sh gw3 scripts/gw3_runs.py'
   ssh <box> 'cd ~/apexzero && BREV_TYPE=massedcompute_A6000_plusx4 nohup bash \
     scripts/gpu_round.sh gw3 scripts/gw3_runs.py > /dev/null 2>&1 &'
   ```

   On K single boxes, box i sets `SHARD=i/K` for the last two commands (`... && SHARD=i/K DRY_RUN=1 bash ...`) and `BREV_TYPE=massedcompute_A6000_plus`. The lead watches `runs/gw3-queue-<i>of<K>.log` and the stop rule (Section 7).
5. When a box's queue has ended, write its sha256 list on the box, pull everything back, and delete the box:

   ```
   ssh <box> 'cd ~/apexzero/runs && sha256sum gw3-v*/config.json gw3-v*/eval_rdev.json \
     gw3-v*/ckpt/000199884800.pkl > gw3-sums-<i>of<K>.txt'
   SHARD=<i>/<K> scripts/gpu_sync.sh <box> pull-round gw3 scripts/gw3_runs.py
   scripts/gpu_sync.sh <box> pull gw3-sums-<i>of<K>.txt
   ```

   On the x4 box, i/K is 0/1. Once every box is back, on the Mac:

   ```
   uv run python scripts/round_manifest.py check --round gw3 --runs-script scripts/gw3_runs.py \
     --tag gw3-prereg-v1
   nice -n 10 uv run python scripts/gw3_analysis.py manifest
   git add -f outputs/gw3/manifest.json && git commit -m 'GW-3: run manifest'
   ```

   The manifest stage accepts the check's problems only for runs that failed twice, or that the lead names as missing (a deviation). It checks every run's config, evaluation and final checkpoint against its box's sha256 list.
6. Run the analysis once: `nice -n 10 uv run python scripts/gw3_analysis.py analyse` (`--check` first runs its input checks alone and writes nothing). Commit `outputs/gw3/analysis.json` with `git add -f`. Write `docs/findings/gw3.md`.
7. GW-4 applies Section 5's rule. The analysis records the rule's decision for both outcomes of criterion 5, which GW-4 checks at the `generator-v1` commit.

**Deviations.** Any change after the tag is logged in the findings with its date and reason. A change to an outcome, test, seed or rule after any run has started makes the affected result exploratory. Deviations from earlier plans that are decided before the tag are logged in Section 10.

## 9. Items before the tag (1, 2, 3 and 5 decided and 6 built on 27 Oct 2026; 4 open)

1. **M and criterion 4: decided by the user on 27 Oct 2026. M = (2a)**, with criterion 4 restated as Section 5 gives it. The deviation from the width-model plan's bands is logged in Section 10.
   - The options were (2a) (recommended); (2b), with `--reexpress` on the `sl` and `sa` arms, criterion 4 restated for `report-2b.json` and its GPU cost measured first; or no stat-model arm (`base`, `cl-f1` and `cl-f2` only). (2b) and dropping the stat-model arm were not chosen.
2. **Seeds per cell: decided by the user on 27 Oct 2026. 10 seeds, 20–29**: 120 runs, about 14 A6000 hours (Section 7). P1's power at 10 seeds is in Section 3. This supersedes #13's 5 seeds per cell.
3. **#12: decided by the user on 27 Oct 2026.** `generator-v1` uses copied R-dev width profiles if and only if (2a) fails GW-4's adoption rule and S1-v1 confirms that copied widths transfer to held-out circuits (Section 5). If copied widths are used, every R-dev number for E0, the 80% gate (#10) included, is labelled "in-sample for widths" wherever it is reported. Otherwise, when (2a) fails the rule and S1-v1 does not confirm, `generator-v1` uses the default widths. The lockbox (R_TEST) is unaffected either way.
4. **Seed ranges (the lead):** keep VG-1 and any other `train_dr.py` round off seeds 20–29. VG-1's draft uses 15–16 and 30–49.
5. **E0's curvature variant: decided by the user on 27 Oct 2026. v1**, `train_dr.py`'s default. It was fixed before the tag, on what existed then, without GW-3 data. Once tagged, a change is a deviation that makes E0's recipe exploratory.
   - The draft's reasons for recommending it (26 Oct 2026): v1 is the variant where the copied-width effect replicates (H4R S1, +2.6 of 12; follow-through Section 7.4 requires v1 in GW-3 for that reason), and the curvature levers did not differ (v1/v2/v3 at 5.4/5.8/5.8; follow-through Section 8).
   - The alternative was v3 (`--tilt-fraction 0.5 --tilt-alpha 2`), whose H4R baseline was higher (7.2 against 5.0 circuits of 12) and whose width effect was smaller.
6. **To build (the lead): done 27 Oct 2026.**
   - `scripts/gw3_runs.py`: the run list. The shared queue `scripts/gpu_round.sh` runs it and `scripts/round_manifest.py` checks it (`46cdd81`); they replace the planned `scripts/gpu_queue_gw3.sh`.
   - `scripts/gw3_analysis.py`: its `manifest` stage is the manifest writer, and its `analyse` stage the analysis.
   - `tests/test_gw3_analysis.py`, among them the counting test of Section 3 against full enumeration.

## 10. Deviations logged before the tag

This list is append-only. Each entry is dated and is part of the text that `gw3-prereg-v1` tags.

1. **27 Oct 2026: GW-4's criterion 4 departs from the width-model plan's bands.**
   - **What the bands required.** The verdict rule of `width-model-plan.md` (Section 4, fixed in `bdfd442` before any report was computed): a model meets the bands only if every judged check passes in both folds and both variants, and the no-copy check passes for f1, f2 and all 12 in both variants. For (2a) that is 90 checks. Criterion 4 was to require that verdict (follow-through Section 7.4), and the plan's branch for "neither meets" left GW-4's criterion 4 unable to hold for either model.
   - **What the bands gave.** (2a) misses 3 of its 90 checks, all in v3: `f1/v3/share_1m`, `f2/v3/share_1m` and `f1/v3/spectrum:8-16`. (2b) misses 3, all in fold f1 (`width-model.md`, Sections 1 and 4).
   - **What changed.** The user chose (2a) on 27 Oct 2026 and restated criterion 4: every v1 check in (2a)'s committed report passes in both folds, and the v3 misses are limited to `share_1m` (both folds) and `spectrum:8-16` (fold f1). The report's sha256 check before GW-4 reads it is kept (Section 5).
   - **Why.** By the plan, the choice went to the user before the tag (`width-model.md`, Section 8). The user took option 1 there, the recommended option. Its stated reasons: (2a) is the direct fit, costs nothing extra to generate and passes every v1 check. (E0's variant, decided the same day and separately, is v1; Section 5.)
     - The readings of its two v3 misses are **post hoc**, made after the results were seen (`width-model.md`, Section 6). They are that the 8–16 m band holds centimetre-scale variation and its rule has no absolute floor, and that the v3 share overshoot is consistent with the tilted grammar's corner mix.
     - They are reasons offered for the choice, not tests. They change no band and no verdict: both models still do not meet the bands.
   - **What it means.** The restated criterion was written after (2a)'s report was seen, and the committed report meets it. Criterion 4 therefore no longer tests (2a) against the bands, and GW-4's decision in practice rests on criteria 1, 2, 3 and 5. The bands' own verdict for (2a) stays "does not meet" (`width-model.md`).
   - **When.** Logged on 27 Oct 2026 in this draft, before the tag `gw3-prereg-v1` and before any GW-3 run.
