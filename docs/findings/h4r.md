# H4R: real-width training lowers edge tracking and raises inside margin (confirmatory)

Results of 22 Oct 2026. Design: [`docs/interp/h4r-design.md`](../interp/h4r-design.md), committed in `612ed7a` before any H4R network was trained; Section 7 (clarifications that change no computation) added in `5453a2e`, before the tag. Pre-registration record: `scripts/h4r_prereg.json` (sha256 `c9d9c5ce…`), committed alone in `eb5637e`. Tag: `h4r-prereg-v1`, annotated (object `368faefc…`), on `eb5637e`, on origin as the same object. Checkpoint manifest: `outputs/h4r/manifest.json` (sha256 `f4731356…`), committed in `593d42e`. Tag pin: `outputs/h4r/tag-h4r-prereg-v1.json`, committed in `fa10eff`. Confirmatory decisions: `outputs/h4r/analyse/decisions.json`, sha256 `f3bcbed6292d19c58f5c6f6c972526bc8bb317540d7314d9f8506bd79207eec9`, committed in `b5f0854`. Code: `scripts/h4r.py` (sha256 `43233a88…`), which drives step 2's frozen code (code hash `651e7eb4…`). Background: [receptive-fields-step2.md](receptive-fields-step2.md) (step 2), [locality.md](locality.md) and [width-test.md](width-test.md).

Every number below was read from, or recomputed from, the file named next to it. Files under `outputs/` and `runs/` are local experiment outputs and are not in git. The exceptions are H4R's `decisions.json`, `manifest.json` and tag pin above, and step 2's `outputs/fields/analyse/decisions.json` (an input of H4R's analysis) and `outputs/fields/tag.json`, which are committed. Labels used throughout:

- **Confirmatory**: the primary family, H4R-G and H4R-B with Holm (design Section 5), decided once by `scripts/h4r.py analyse` and read from `decisions.json`.
- **Secondary**: the pre-registered secondary items S1–S3, computed in the same run. S1 and S2 are decided at 0.05, unadjusted. S3 is descriptive by design, with CIs.
- **Descriptive**: everything else. These numbers decide nothing. **Computed here** marks numbers computed for this document from the files named, with the method stated. No network was run for them.

Times are EDT (UTC−4). The training box logged in UTC. Its queue log's retry line (16:47:40) and the crash's own Unix time stamp (1792687660, 16:47:40 UTC) agree, so its times are converted by −4 h.

## 1. Summary

H4R tested, on 20 networks that did not exist when the hypotheses were written, what training on real-circuit width profiles does to how a racing policy's line follows the inside edge of a corner. Step 2 had not been able to test this (its H4 was "not testable"), and its exploratory analysis pointed to lower edge tracking and more inside margin, on networks whose arms differed in code and hardware. H4R fixed the directions from that result, trained all 20 networks on one machine with one commit, and measured them with step 2's frozen code on step 2's frozen states.

The measures, as in step 2:
- **G**, the edge-tracking gain: how far the line moves toward the inside per metre the inside edge moves, for a corner-wide shift. The optimum's G* is 1; a width-blind line has G = 0.
- **B**, the buffer: the line's minimum clearance to the inside termination boundary, in metres.
- Each network's statistic is its median over geometries of the per-geometry medians of its mirror-pair values, on 427 primary non-capped A2 keys.

**Confirmatory (Holm over two tests, α = 0.05).**
- **H4R-G, real-width training lowers G: confirm.**
  - Statistic −0.118, the sum over variants of mean real-width minus mean baseline (v1 −0.049, v3 −0.069).
  - One-sided exact p 7.6e-4 (48 of 63,504 relabellings), below its Holm level 0.025.
  - Real-width networks 0.716–0.784; baselines 0.729–0.864.
- **H4R-B, real-width training raises B: confirm.**
  - Statistic +0.71 m (v1 +0.44, v3 +0.27).
  - p 1.3e-3 (80 of 63,504), below its Holm level 0.05.
  - Real-width networks 1.23–1.77 m; baselines 0.90–1.71 m.
- Neither test comes near a refute: the opposite one-sided p is 0.9993 for G and 0.9988 for B. The Holm-adjusted p is 0.0015 for both.
- **All 20 networks entered.** Five had exactly one key of 12 failing the post-tag autodiff check gate2-trained: three real-width networks and two baselines. Under the rule fixed before training, each lost that key, and none became descriptive (that needs more than 2 failing keys). Exclusion shares are 0.2–16.2%, all below the 30% limit.

**Secondary.**
- **S1, completion: confirm** (p 0.026). Real-width networks completed a mean 7.6 of 12 R-dev circuits against 5.0 for baselines in v1, and 7.8 against 7.2 in v3. The gain is mostly v1's.
- **S2, paired by seed: confirm for both measures.** G is lower in all 10 same-seed pairs (mean −0.059; p 1/1,024 = 9.8e-4, the smallest possible). B is higher in 9 of 10 (mean +0.36 m; p 0.0029).
- **S3, B against completion within each arm: descriptive (no decision rule); both CIs include 0.** Spearman's ρ is 0.55 [−0.09, 0.93] in the baseline arm and −0.14 [−0.79, 0.65] in the real-width arm. Whether the extra margin explains the extra completion is not established.

**Most informative descriptive results.**
- **Tracking is apparently rearranged.** No shift was detected in T, the line's move when the whole corridor shifts (+0.021, two-sided p 0.32; computed here, descriptive). T is 0.72–0.80 in all 20 networks, all labelled "mixed". Every real-width network's outside-edge response G_out is higher than every baseline's of the same variant (computed here).
- **Same direction as step 2, from networks free of step 2's code and hardware confound.** In both variants and both studies, real-width medians are lower in G and higher in B. H4R's G shift is smaller than step 2's in v1 (−0.049 against −0.081) and similar in v3 (−0.069 against −0.077). Its B shift is larger in v1 (+0.44 against +0.28 m) and similar in v3 (+0.27 against +0.29 m).
- **The shifts sit on typical corners.** On bulk corners real-width training lowers G by about 0.08 in both variants. On corners tighter than the training support there is little G difference (+0.004 in v1, −0.018 in v3; untested).
- **Robustness (computed here).** On the 296 keys valid for all 20 networks, both shifts persist, each at p 3.1e-5. With the five failing-key networks dropped altogether, p is 0.0056 for G and 0.018 for B.

**Reading.** On fresh same-machine, same-code networks, real-width training makes the line follow corner-wide inside-edge shifts less and keep more inside margin. The effect that step 2 found exploratorily does not need its code and hardware differences. H4R does not show that the margin causes the completion gain (S3), and it does not measure how the line follows local edge changes, lap time, or real roads.

## 2. Why H4R

### 2.1 Step 2's H4 was not testable

Source: [receptive-fields-step2.md](receptive-fields-step2.md), Sections 4.1 and 5.3.
- Step 2's H4 predicted that real-width training raises G, on 4 real-width seeds against baseline seeds 1–4 in each of v1 and v3.
- Two of its 18 networks, `dr-pilot-v3-tilt50a2-s2` and `dr-realw-v1-s1`, each failed gate2-trained at one key of 12, on c_min, with relative errors of 1.7e-3 and 1.0e-3 against a 1e-3 limit.
- Under step 2's strict rule, fixed before its tag, a failing network was descriptive and H4 was "not testable". The existing networks cannot be re-decided.

### 2.2 The exploratory result

Step 2 ran H4's frozen test anyway, ignoring the two gate failures (its Section 6, exploratory):
- **Real-width training lowered G**, the opposite of H4's prediction: −0.158 summed over variants (v1 −0.081, v3 −0.077), opposite one-sided p 2.0e-4, the smallest attainable with 4,900 relabellings. G separated completely in both variants.
- **It raised B**: +0.575 m (v1 +0.283, v3 +0.292), two-sided p 0.002.

It carried three weaknesses (H4R design, Section 1):
- it was post hoc in the gate rule;
- B's test and direction were chosen after the confirmatory results were known;
- the arms differed in code and hardware.

### 2.3 The confound it removes

**In step 2** ([width-test.md](width-test.md); step 2, Section 6.4):
- The baselines' seed 0 ran on the Mac CPU before the PPO refactor that ruling R18 records. Seeds 1–4 ran on a rented NVIDIA L40S.
- The real-width runs used later code (commit `83ebe56`) on a rented GPU whose type was not recorded. They logged 0.56–0.63M steps/s against 0.42–0.45M for baseline seeds 1–4.

**In H4R:**
- All 20 runs trained on one rented NVIDIA RTX A6000 box, four at a time, from one checkout at `612ed7a` (`runs/activity.md`; `runs/h4r.log`).
- The training code (`scripts/train_dr.py`, the generator, the environment, the agents) is unchanged since `2143a59`, which added the leave-out option the width test's leave-out runs used (design Section 2 calls it the code of the width-test real-width runs, which it is not; Section 9.1). Checked here: `git diff 2143a59 612ed7a -- scripts/train_dr.py src/` touches only `src/apexzero/fields/`, step 2's analysis code.
- The arms differ only by `--real-widths`. `scripts/h4r.py manifest` checked every run's `config.json` against design Section 2 and found no problems (`manifest.json`, `config_problems` empty).
- The queue ran one lane per arm × variant, so each round trained one network of each arm × variant at the same time (`scripts/gpu_queue_h4r.sh`; Section 4.1).
- Fresh seeds (10–14), not used by any earlier run (design Section 2), and directions fixed from step 2's result on networks that played no part in forming it.

**What it does not remove.**
- The real-width profiles come from the same 12 R-dev circuits that S1 evaluates on, so S1 may partly be leakage, as in the width test. G and B are measured on synthetic A2 corners, not on R-dev roads.
- GPU training is not bit-reproducible, and 5 seeds per arm per variant is a small sample.
- The GPU's load was not identical for all runs. The lanes drifted apart: the v1 real-width lane started its seed 14 at 12:22:05, the other three lanes theirs at 12:26:39–12:27:03 (`runs/h4r.log`), and the retried run trained mostly alone (Section 4.2).

## 3. Design in brief

**Networks** (design Section 2; `manifest.json`).

| | v1 (base grammar) | v3 (tilted grammar: `--tilt-fraction 0.5 --tilt-alpha 2`) |
|---|---|---|
| Baseline | `h4r-v1-base-s10..14` | `h4r-v3-base-s10..14` |
| Real widths (`--real-widths`, all 12 R-dev profiles) | `h4r-v1-realw-s10..14` | `h4r-v3-realw-s10..14` |

- `scripts/train_dr.py --steps 200e6 --seed s`. Every other argument is the default, as in the step 2 networks (`num_envs` 2048, `num_steps` 64, MLP, bank 4096, refresh 256).
- Each run's final checkpoint is at 199,884,800 steps, train_dr.py's last update. Each run also wrote its standard R-dev evaluation (`eval_rdev.json`) at the end of training.
- A failed run is re-run once with the same arguments. It is missing only if it fails twice.

**States and measures: exactly step 2's** (design Section 3; step 2, Section 2).
- **States.** Step 2's frozen A2 key list: all 481 keys were computed. The statistics use the 427 primary non-capped keys, which form 62 mirror pairs over 48 geometries. Every network starts from the same states.
- **c_min** is the minimum inside clearance over [s_c0 − 50, s_c1 + 50] m, against the termination boundary. **B = c_min** on the unshifted road.
- **G = 1 − dc_min/dΔ** for a trapezoid shift Δ·h(s) of the inside edge, flat from s_c0 − 20 to s_c1 + 20 m, with 20 m ramps. It is defined only where the argmin lies on the plateau.
- **T = G + G_out**, the line's move per metre when the whole corridor shifts toward the inside. G_out is the outward move per metre the outside widens.
- **Per-network statistic** (design Section 7): step 2's `stages.median_statistic`, the median over geometries of the per-geometry medians of the mirror-pair values, under step 2's pair-completeness, exclusion and coverage rules. A pair is undefined when more than 3/8 of its keys are excluded.
- All fields were computed with step 2's frozen code, imported read-only: one `jax.vjp` of each network's float64 closed loop per key.

**Gates.**
- **Checkpoint manifest.** Path, timesteps and sha256 of every final checkpoint, recorded and committed before the tag. Every later stage re-hashes it.
- **Wiring.** Each network's float32 harness run (harness-f32) is the reference for the fields stage's teacher-forced Gate 3 and float32/float64 branch-consistency checks, with step 2's calibrated tolerance. A failing key is a pipeline exclusion.
- **gate2-trained, changed from step 2 and fixed in the design before training.** Step 2's 12 keys and per-key rule: autodiff against finite differences along 12 directions, 4 outcomes. A key that fails is excluded for that network as "pipeline: gate2-trained", counted in coverage, after step 2's pipeline reasons and before "invalid outcome". The network becomes descriptive only if more than 2 of its 12 keys fail. The design's reason: step 2's failures were single, marginal keys, and its rule let one such key void a whole hypothesis.
- **Coverage.** Step 2's rule: a network with more than 30% of its keys excluded, counting every reason, is descriptive.

**Tests** (design Section 5; the record's `design` block).
- **H4R-G (primary):** real-width below baseline on median G. **H4R-B (primary):** real-width above baseline on median B.
- **Test:** step 2's exact permutation test (`analysis.permutation_test`). The statistic is the sum over variants of (mean real-width − mean baseline). Labels are exchanged within each variant, giving C(10, 5)² = 63,504 relabellings for 5 against 5.
- **Holm** across the two at α = 0.05 (m = 2): the smaller p is held to 0.025, the other to 0.05. A test that is not testable enters as p = 1.
- **Refute** if the unadjusted opposite one-sided p ≤ 0.05. **Inconclusive** otherwise.
- **Missing or descriptive networks** are dropped and every relabelling of the remaining sets is enumerated. A test is not testable with fewer than 3 networks in any arm of either variant.
- **Secondary:**
  - S1: R-dev circuits completed (completion > 0.5), the same test, real-width above baseline, on every present network.
  - S2: same-seed pairs, the mean of the 10 within-pair differences, all 1,024 sign flips; for G (lower) and B (higher).
  - S3: within each arm, Spearman's ρ of B and circuits completed, both variants pooled (per variant also reported), with a 95% percentile bootstrap CI (20,000 draws, stream 201). Descriptive.
- **Descriptive:** per-network G, B and T with CIs; left and right; the per-regime breakdown; the pooled picture with step 2's 18 H4 networks.

**Record and tag.** The record binds:
- the design document's sha256 (`5b4f985e…`, equal to the file now), `scripts/h4r.py`'s (`43233a88…`) and `tests/test_h4r.py`'s (`c2a772be…`);
- step 2's tag, record and committed decisions by hash, and its frozen A2 and gate2-trained key lists with their content hashes;
- step 2's calibrated branch tolerance (c_min 3.3e-3 m, v_apex 1.36e-3 m/s, s_b 7.4e-2 m);
- the manifest's sha256, and through it every checkpoint and evaluation;
- the permutation design, Holm, the gate rule, the secondary tests and the per-regime cell sets (typed by step 2's G1 for v1 and G3 for v3).

**Procedure and timeline** (design Section 6; commits, logs and provenance stamps).

| Step | When (22 Oct, EDT) | Record |
|---|---|---|
| Design committed, before any H4R network was trained | 10:37:49 | `612ed7a` |
| Box rented under the standing permission and set up at `612ed7a` | 10:38–10:45 | `runs/activity.md` |
| Training: 20 runs, 4 lanes, 5 rounds | 10:44:57–12:58:14 | `runs/h4r.log` |
| `scripts/h4r.py`, its tests and design Section 7 committed. The commit message records an adversarial review: 13 findings fixed, 61 tests, 63 targeted mutants all killed | 12:06:48 | `5453a2e` |
| Runs pulled back (final checkpoints matched the box's sha256), box deleted | 12:59 | `runs/activity.md` |
| Manifest written (12:59:42–12:59:45) and committed | 13:00:15 | `593d42e` |
| Record committed alone; tag placed and pushed | 13:02:21 | `eb5637e`; tag object `368faefc…` |
| harness-f32 (it pinned the tag at 13:04:26) | 13:02:41–13:05:05 | `runs/h4r-posttag.log`; `harness-f32/summary.json` |
| Tag pin committed | 13:05:05 | `fa10eff` |
| gate2-trained (240 network-keys) | 13:05:07–13:22:48 | `runs/h4r-gate2-trained.log` |
| fields: 20 × 481 = 9,620 rows | 13:22:48–16:23:03 | `runs/h4r-posttag.log` (10,812 s); `runs/h4r-fields.log` (rows 13:24:21–16:23:01, 178.7 min) |
| `analyse --check` (input checks only; writes nothing) | about 16:24 | `runs/activity.md` |
| `analyse`, once | 16:24:55–16:27:06 | `decisions.json` (provenance) |
| Decisions committed | 16:27:10 | `b5f0854` |

- The design's budget was about 2.2 h of training (5 rounds of about 26 min) and about 2.9 h of fields. Training took 2 h 13 min (queue start to last run done), and the fields stage 3.0 h (180.2 min, of which 178.7 min computing rows).
- `analyse` refuses to run when its output exists or was ever committed (`scripts/h4r.py` `stage_analyse`, via step 2's `fields.run_once_problems`). It ran once.

## 4. Training and gates

### 4.1 The manifest

Source: `manifest.json` (stage stamp 12:59:42–12:59:45, HEAD `5453a2e`); queue times from `runs/h4r.log` (its sha256 `e040f999…` is bound in the manifest), converted to EDT; training wall time and throughput from each run's `metrics.csv` (last row).

| Run | Start | Done | Training wall time (min) | Steps/s | Final checkpoint sha256 | Circuits completed (of 12) |
|---|---|---|---|---|---|---|
| `h4r-v1-base-s10` | 10:44:57 | 11:11:16 | 22.8 | 146k | `bde4e1b86445…` | 6 |
| `h4r-v1-base-s11` | 11:11:16 | 11:36:20 | 22.1 | 151k | `bce49f90738d…` | 4 |
| `h4r-v1-base-s12` | 11:36:20 | 12:01:51 | 22.1 | 150k | `92f1c2286d15…` | 7 |
| `h4r-v1-base-s13` | 12:01:51 | 12:27:03 | 21.6 | 154k | `d346ab031b5f…` | 5 |
| `h4r-v1-base-s14` | 12:27:03; retry 12:47:40 | 12:58:14 | 8.6 (retry) | 386k (retry) | `733ceb9f0ad7…` | 3 |
| `h4r-v1-realw-s10` | 10:44:57 | 11:10:32 | 22.1 | 151k | `1c13f4ab31e0…` | 4 |
| `h4r-v1-realw-s11` | 11:10:32 | 11:34:18 | 20.1 | 166k | `392dbbf777b6…` | 8 |
| `h4r-v1-realw-s12` | 11:34:18 | 11:58:29 | 19.9 | 167k | `13ded5e232d7…` | 9 |
| `h4r-v1-realw-s13` | 11:58:29 | 12:22:05 | 19.4 | 172k | `f28e387bed30…` | 9 |
| `h4r-v1-realw-s14` | 12:22:05 | 12:45:00 | 18.7 | 179k | `ce7658aa395c…` | 8 |
| `h4r-v3-base-s10` | 10:44:57 | 11:11:17 | 22.9 | 145k | `2ba2621b4b64…` | 6 |
| `h4r-v3-base-s11` | 11:11:17 | 11:36:14 | 22.1 | 151k | `93816e291465…` | 8 |
| `h4r-v3-base-s12` | 11:36:14 | 12:01:40 | 22.1 | 151k | `caf1a0cf08a5…` | 8 |
| `h4r-v3-base-s13` | 12:01:40 | 12:26:39 | 21.3 | 156k | `d50c8941dabc…` | 6 |
| `h4r-v3-base-s14` | 12:26:39 | 12:50:14 | 20.7 | 161k | `d003463216c9…` | 8 |
| `h4r-v3-realw-s10` | 10:44:57 | 11:11:01 | 22.7 | 147k | `3f12e179a961…` | 8 |
| `h4r-v3-realw-s11` | 11:11:01 | 11:35:51 | 21.6 | 154k | `e6f6b295c3b5…` | 8 |
| `h4r-v3-realw-s12` | 11:35:51 | 12:01:45 | 22.5 | 148k | `10c0456acbfc…` | 6 |
| `h4r-v3-realw-s13` | 12:01:45 | 12:26:46 | 21.5 | 155k | `bccc6d7d10da…` | 8 |
| `h4r-v3-realw-s14` | 12:26:46 | 12:50:24 | 20.7 | 161k | `f6ebeffb9dde…` | 9 |

- **20 of 20 present.** Every run reached 199,884,800 steps with 16 checkpoints. `missing`, `unexplained_missing`, `finished_missing` and `config_problems` are empty, and the queue log shows no unexpected run.
- **Hashes checked here.** Every final checkpoint, `eval_rdev.json` and `config.json` on disk matches its sha256 in the manifest, and gate2-trained recorded the same checkpoint hashes.
- **No other errors.** Each training log holds only its evaluation lines. The only error in any training log is the first attempt of `h4r-v1-base-s14` (Section 4.2).

### 4.2 The retry

- **What happened.** `h4r-v1-base-s14` started at 12:27:03. At 12:47:40 it stopped with an XLA check failure: a device-to-host copy failed with "an illegal memory access was encountered" (`runs/h4r-v1-base-s14.log.1`, which holds only that stack trace).
- **What the queue did, as designed** (`scripts/gpu_queue_h4r.sh`; design Section 2). It moved the log to `.log.1`, deleted the run's directory, and re-ran the run once from scratch with the same arguments.
- **The retry** ran 12:47:40–12:58:14 and finished normally. The other three lanes had finished by 12:50:24, so it ran mostly alone on the GPU: 8.6 min of training at 386k steps/s, against 18.7–22.9 min at 145k–179k for the other 19 runs.
- **The record.** The manifest lists it as `retried: true`, `failed_twice: false`, with the first attempt's log kept. The record's commit message calls it "one transient CUDA fault retried, as designed".
- Same box, same code, same arguments. Only the GPU's load differed, which is not expected to change what is learned beyond ordinary GPU non-determinism.
- **The retried run is v1's most atypical baseline**: its lowest G (0.729, below four of the five v1 real-width networks), its fewest circuits (3) and the smallest paired G drop (−0.008; Section 6.2). Without it (31,752 relabellings), G p 9.4e-5, B p 1.6e-3 and S1 p 0.044, each still within its level (computed here by the same exact enumeration; descriptive).

### 4.3 gate2-trained

Source: `outputs/h4r/gate2-trained/gate2_trained.json` (stage stamp 13:05:08–13:22:48; `runs/h4r-gate2-trained.log`, 240 of 240 network-keys in 15.9 min).
- **Method** (step 2, Section 4.1). At each of the 12 keys in the record, along 12 directions (5 corner bumps of κ, 5 corner bumps of the inside width, and both trapezoids), 4 outcomes are scored: c_min, v_apex, s_b and steer.
  - A direction counts only if the +h, 0 and −h runs share their discrete structure and FD(h) and FD(h/2) agree to O(h²).
  - A valid outcome passes when its field vector's relative error over the counted directions is below 1e-3. A valid outcome with no counted direction fails (`network.Gate2.outcome_ok`).
- **Result.** 15 of 20 networks pass all 12 keys. Five fail exactly one key each. None is descriptive. No loop was exhausted (`n_exhausted` 0 for every network).

| Network | Key (cell, phase) | Failing outcome | Why it fails | Directions counted for that outcome | Other outcomes at that key |
|---|---|---|---|---|---|
| `h4r-v1-realw-s10` | `e3b33e62…` (cell 46, phase 1) | c_min | relative error 4.2e-2 | 5 of 12; 7 failed the FD step-pair check | v_apex 1.2e-4, s_b 7.6e-11, steer 1.6e-11 |
| `h4r-v1-realw-s12` | `064f04ca…` (cell 35, phase 3) | s_b | no direction could be counted, so no error is defined | 0 of 12; all 12 failed the FD step-pair check | c_min 1.4e-5, v_apex 1.0e-4, steer 1.0e-10 |
| `h4r-v3-base-s10` | `064f04ca…` (cell 35, phase 3) | c_min | relative error 1.47e-3 | 12 of 12 | v_apex 2.0e-4, s_b 8.5e-11, steer 1.6e-11 |
| `h4r-v3-base-s13` | `17acd94f…` (cell 140, phase 2) | v_apex | relative error 3.5e-3 | 4 of 12; 8 failed the FD step-pair check | c_min 3.7e-5, s_b 7.0e-9, steer 4.5e-11 |
| `h4r-v3-realw-s10` | `578a2d8e…` (cell 342, phase 0) | c_min | relative error 2.2e-3 | 11 of 12; 1 failed the FD step-pair check | v_apex 1.9e-4, steer 1.8e-11; s_b not valid at this key |

What the failures look like (computed here from the same file):
- **Both arms are affected.** Three failures are real-width networks (v1 seeds 10 and 12, v3 seed 10) and two are v3 baselines (seeds 10 and 13). No v1 baseline failed.
- **Four distinct keys.** `064f04ca…` fails for two networks, on different outcomes. None of the four is one of step 2's failing keys (`2f3c4793…`, `54d2398d…`).
- **Not all on c_min.** Three fail on c_min, one on v_apex and one on s_b, where the finite differences could not be counted at all.
- **Three are within a few times the limit, one has nothing countable, one is far above.** The errors are 1.5e-3, 2.2e-3 and 3.5e-3. The s_b failure has no countable direction. The c_min error of `h4r-v1-realw-s10`, 4.2e-2, is 42 times the limit, at a key where 7 of 12 c_min directions could not be counted.
- **The c_min distribution.** Over all 240 network-key c_min errors the median is 3.8e-7 and the 95th percentile 1.8e-4. Three exceed 1e-3 (the three c_min failures). The next largest are 9.2e-4 (`h4r-v3-realw-s11`, cell 134) and 8.1e-4 (`h4r-v1-base-s13` at `2f3c4793…`, where step 2's G3s2 failed), and both pass.
- **Most fall where the outcomes are rough.** At three of the five, most or all directions of the failing outcome failed the step-pair check, the sign of kinks (a switching argmin or saturation) near the base point. The other two (`h4r-v3-base-s10`, `h4r-v3-realw-s10`) counted 12 and 11 of 12 directions and exceed the limit by factors of 1.5 and 2.2. As in step 2, the gate cannot say whether the excess comes from the autodiff field or from the finite-difference reference at such points.
- **The rule mattered.** Under step 2's rules all five would have been descriptive, and a hypothesis with any descriptive network was not testable. Under H4R's rule, fixed before training, each loses one key (Section 4.5), well short of the 3 failing keys that would make a network descriptive. Section 5.4 shows that the decisions survive dropping all five networks.

### 4.4 Wiring (harness-f32, Gate 3 and branch consistency)

Sources: `outputs/h4r/harness-f32/summary.json` and its stamps; `outputs/h4r/fields/*.jsonl` (computed here: counts over rows).
- **harness-f32.** Each network's float32 harness run from the 475 primary A2 starts, in a 256-step loop (13:02:43–13:05:05).
- **Gate 3, teacher-forced wiring, in the fields stage: 9,619 of 9,620 rows pass.**
  - The largest mean action error is 5.1e-5 (`h4r-v1-base-s14`), against a limit of 1e-4.
  - That is slightly above the trained-agent error that step 2's branch tolerance was calibrated with, 4.94e-5 (G2s2). So for this one network the tolerance's safety factor is about 4.8 instead of 5. The other 19 networks' largest errors are 2.9e-5 to 4.9e-5.
  - The one failing row is `h4r-v3-base-s14` at a capped key (`531efcca…`, cell 3, phase 3), outside the 427 keys the statistics use. There one state's one-step error exceeds its float32 rounding tolerance by 11%. Its run crashes before the window end, so c_min is invalid there anyway.
- **Branch consistency** (float64 loop against the float32 harness run: termination step, saturation set, outcome events and values within the calibrated tolerance). 0–6 of 481 rows per network fail on some outcome, and 0–2 on c_min. On the 427 keys this gives 0–2 "pipeline: branch" exclusions per network (Section 4.5).
- **Loop length.** The longest base run takes 203 of the 256 loop steps. No loop was exhausted.

### 4.5 Coverage of G per network (primary non-capped A2 keys, n = 427)

Source: `decisions.json` (`networks.<run>.coverage`, `near_ties`). The share counts every reason. There were no Gate 3, Gate 4 or inertia exclusions on these keys. No network is near the 30% limit, and `decisions.json` lists nothing missing.

| Network | gate2-trained failing keys | Keys used | Excluded | Invalid outcome | Argmin off plateau | κ clipping | Pipeline: branch | Pipeline: gate2-trained | c_min near-ties (kept) |
|---|---|---|---|---|---|---|---|---|---|
| `h4r-v1-base-s10` | 0 | 401 | 6.1% | 4 | 4 | 16 | 2 | 0 | 4 |
| `h4r-v1-base-s11` | 0 | 393 | 8.0% | 16 | 2 | 15 | 1 | 0 | 3 |
| `h4r-v1-base-s12` | 0 | 400 | 6.3% | 12 | 3 | 12 | 0 | 0 | 0 |
| `h4r-v1-base-s13` | 0 | 391 | 8.4% | 24 | 0 | 12 | 0 | 0 | 12 |
| `h4r-v1-base-s14` | 0 | 398 | 6.8% | 16 | 4 | 8 | 1 | 0 | 2 |
| `h4r-v1-realw-s10` | 1 | 400 | 6.3% | 12 | 2 | 12 | 0 | 1 | 4 |
| `h4r-v1-realw-s11` | 0 | 399 | 6.6% | 8 | 8 | 12 | 0 | 0 | 1 |
| `h4r-v1-realw-s12` | 1 | 376 | 11.9% | 20 | 17 | 12 | 1 | 1 | 1 |
| `h4r-v1-realw-s13` | 0 | 398 | 6.8% | 20 | 1 | 8 | 0 | 0 | 0 |
| `h4r-v1-realw-s14` | 0 | 398 | 6.8% | 4 | 9 | 16 | 0 | 0 | 1 |
| `h4r-v3-base-s10` | 1 | 412 | 3.5% | 12 | 1 | 0 | 1 | 1 | 0 |
| `h4r-v3-base-s11` | 0 | 423 | 0.9% | 4 | 0 | 0 | 0 | 0 | 1 |
| `h4r-v3-base-s12` | 0 | 415 | 2.8% | 12 | 0 | 0 | 0 | 0 | 0 |
| `h4r-v3-base-s13` | 1 | 411 | 3.7% | 5 | 10 | 0 | 0 | 1 | 4 |
| `h4r-v3-base-s14` | 0 | 424 | 0.7% | 0 | 3 | 0 | 0 | 0 | 0 |
| `h4r-v3-realw-s10` | 1 | 408 | 4.4% | 16 | 2 | 0 | 0 | 1 | 2 |
| `h4r-v3-realw-s11` | 0 | 358 | 16.2% | 4 | 65 | 0 | 0 | 0 | 4 |
| `h4r-v3-realw-s12` | 0 | 426 | 0.2% | 0 | 0 | 0 | 1 | 0 | 3 |
| `h4r-v3-realw-s13` | 0 | 405 | 5.2% | 20 | 2 | 0 | 0 | 0 | 0 |
| `h4r-v3-realw-s14` | 0 | 422 | 1.2% | 4 | 1 | 0 | 0 | 0 | 1 |

- **Exclusion shares** run from 0.2% to 16.2%.
- **"Argmin off plateau"** (G undefined because the closest point falls where the trapezoid is not flat) depends on the network's own line. It is largest in `h4r-v3-realw-s11` (65 keys, its main reason) and next in `h4r-v1-realw-s12` (17, after 20 invalid outcomes). It is also the main reason in `h4r-v3-base-s13` (10 keys). Section 5.4 checks the result on keys valid for every network.
- **"Invalid outcome"** means c_min is undefined because the run crashes before the window end. Every invalid c_min in the field rows has that reason: 373 of 9,620 rows over all 481 keys (computed here).
- **κ clipping** occurs only in the v1 networks (8–16 keys each), as in step 2, where the v1 networks had 8–16 such keys and the v3 networks none (its Section 4.3).

## 5. Confirmatory decisions

### 5.1 The run

Source: the provenance stamp in `decisions.json`.
- `scripts/h4r.py analyse`, 22 Oct 16:24:55–16:27:06, HEAD `8b4a097`, no changed paths, deviation null, `confirmatory: true`.
- Code hash `651e7eb4…` (step 2's frozen code) and `scripts/h4r.py` sha256 `43233a88…`, both as recorded. Tag `h4r-prereg-v1` (object `368faefc…` on `eb5637e`); record sha256 `c9d9c5ce…`, as committed.
- Its 43 inputs are named by sha256: the 20 field files, gate2-trained, the 20 harness files, step 2's committed decisions and the optimum's field rows. Checked here: all 43 match the files on disk.
- `missing` is empty, every network's status is "kept", and G, B and T are finite for all 20.
- The file was committed at 16:27:10 (`b5f0854`).
- **Reproduced here.** Recomputing each network's G, B and T from its field rows with step 2's `stages.edge_key_measures` and `stages.median_statistic`, with each failing gate2-trained key excluded, gives exactly the values in `decisions.json` (largest difference 0).

### 5.2 Per network

Source: `decisions.json` (`networks.<run>`: `G`, `B`, `T` pooled medians and 95% cluster-bootstrap CIs over geometries; `coverage`; `completed`). The pooled summary weights both turn directions equally. Seeds 10–14 in each row group.

| Variant | Arm | Seed | G [95% CI] | B, m [95% CI] | T [95% CI] | Geometries (of 48) | Pairs defined (of 62) | Excluded | Circuits completed |
|---|---|---|---|---|---|---|---|---|---|
| v1 | baseline | 10 | 0.852 [0.792, 0.870] | 1.37 [1.05, 1.59] | 0.725 [0.716, 0.817] | 45 | 58 | 6.1% | 6 |
| v1 | baseline | 11 | 0.784 [0.743, 0.864] | 0.93 [0.83, 1.23] | 0.778 [0.738, 0.836] | 42 | 55 | 8.0% | 4 |
| v1 | baseline | 12 | 0.832 [0.761, 0.878] | 0.98 [0.86, 1.15] | 0.752 [0.715, 0.791] | 42 | 56 | 6.3% | 7 |
| v1 | baseline | 13 | 0.800 [0.732, 0.844] | 0.90 [0.77, 1.20] | 0.756 [0.698, 0.805] | 44 | 57 | 8.4% | 5 |
| v1 | baseline | 14 | 0.729 [0.707, 0.763] | 1.26 [1.13, 1.47] | 0.724 [0.696, 0.758] | 45 | 57 | 6.8% | 3 |
| v1 | real width | 10 | 0.742 [0.717, 0.777] | 1.61 [1.05, 1.73] | 0.760 [0.738, 0.807] | 43 | 56 | 6.3% | 4 |
| v1 | real width | 11 | 0.753 [0.689, 0.773] | 1.54 [1.35, 1.84] | 0.788 [0.743, 0.821] | 45 | 57 | 6.6% | 8 |
| v1 | real width | 12 | 0.784 [0.739, 0.803] | 1.58 [1.40, 1.85] | 0.802 [0.760, 0.842] | 39 | 52 | 11.9% | 9 |
| v1 | real width | 13 | 0.750 [0.711, 0.759] | 1.49 [1.35, 1.68] | 0.741 [0.715, 0.819] | 43 | 56 | 6.8% | 9 |
| v1 | real width | 14 | 0.722 [0.679, 0.760] | 1.42 [1.08, 1.77] | 0.742 [0.687, 0.811] | 40 | 53 | 6.8% | 8 |
| v3 | baseline | 10 | 0.812 [0.748, 0.845] | 1.32 [1.02, 1.77] | 0.761 [0.726, 0.804] | 46 | 59 | 3.5% | 6 |
| v3 | baseline | 11 | 0.864 [0.807, 0.885] | 1.38 [1.07, 1.81] | 0.752 [0.738, 0.784] | 47 | 61 | 0.9% | 8 |
| v3 | baseline | 12 | 0.785 [0.736, 0.835] | 1.11 [0.88, 1.27] | 0.767 [0.730, 0.836] | 46 | 60 | 2.8% | 8 |
| v3 | baseline | 13 | 0.810 [0.765, 0.824] | 0.96 [0.85, 1.09] | 0.782 [0.739, 0.820] | 44 | 56 | 3.7% | 6 |
| v3 | baseline | 14 | 0.774 [0.653, 0.855] | 1.71 [1.08, 2.37] | 0.768 [0.695, 0.788] | 46 | 60 | 0.7% | 8 |
| v3 | real width | 10 | 0.728 [0.703, 0.751] | 1.50 [1.29, 1.71] | 0.796 [0.747, 0.829] | 43 | 56 | 4.4% | 8 |
| v3 | real width | 11 | 0.779 [0.702, 0.797] | 1.23 [0.92, 1.61] | 0.735 [0.700, 0.811] | 35 | 44 | 16.2% | 8 |
| v3 | real width | 12 | 0.720 [0.678, 0.783] | 1.74 [1.31, 1.94] | 0.772 [0.739, 0.806] | 48 | 62 | 0.2% | 6 |
| v3 | real width | 13 | 0.716 [0.680, 0.747] | 1.60 [1.38, 1.74] | 0.748 [0.699, 0.807] | 43 | 56 | 5.2% | 8 |
| v3 | real width | 14 | 0.758 [0.711, 0.775] | 1.77 [1.20, 2.08] | 0.784 [0.727, 0.854] | 46 | 60 | 1.2% | 9 |
| | Optimum G*, T* (reference) | | 1.000 [1 − 6e-12, 1 + 3e-12] | — | 1.000 [1 − 2e-12, 1 + 2e-12] | 48 | 62 | — | — |

The optimum's row is `descriptive.optimum` (G_star, T_star).

**Arm ranges and means** (computed here from the table).

| Measure | Variant | Real width: range (mean) | Baseline: range (mean) | Mean difference |
|---|---|---|---|---|
| G | v1 | 0.722–0.784 (0.750) | 0.729–0.852 (0.800) | −0.049 |
| G | v3 | 0.716–0.779 (0.740) | 0.774–0.864 (0.809) | −0.069 |
| B (m) | v1 | 1.42–1.61 (1.53) | 0.90–1.37 (1.09) | +0.44 |
| B (m) | v3 | 1.23–1.77 (1.57) | 0.96–1.71 (1.30) | +0.27 |

- **G does not separate completely**, unlike step 2's exploratory result. In v1, baseline seed 14 (0.729) lies below four of the five real-width networks. In v3, real-width seed 11 (0.779) lies above baseline seed 14 (0.774).
- **B separates completely in v1** (real-width 1.42–1.61 m against baselines 0.90–1.37 m). It overlaps in v3: real-width seed 11 (1.23 m) lies below three baselines, and baseline seed 14 (1.71 m) above three real-width networks.

### 5.3 The tests

Source: `decisions.json` (`H4R_G`, `H4R_B`, `holm`). "p" is the share of relabellings with a statistic at least as extreme in the predicted direction; "p opposite" is the other direction's. Per-variant p values (252 relabellings each) are reported by the code but are not decision tests. Checked here: exact enumeration from the per-network statistics reproduces every T and p below.

| Test | Direction | Statistic (v1, v3) | Relabellings | p | p opposite | Per-variant p (v1, v3) | Holm rank, level | Holm-adjusted p | Decision |
|---|---|---|---|---|---|---|---|---|---|
| H4R-G | real < baseline | −0.118 (−0.049, −0.069) | 63,504 | **7.6e-4** (48/63,504) | 0.9993 | 0.040, 0.0079 | 1, 0.025 | 0.0015 | **confirm** |
| H4R-B | real > baseline | +0.712 m (+0.442, +0.270) | 63,504 | **1.3e-3** (80/63,504) | 0.9988 | 0.0040, 0.063 | 2, 0.05 | 0.0015 | **confirm** |

- **Holm.** H4R-G has the smaller p, 7.6e-4, held to 0.025: rejected. H4R-B, 1.3e-3, is then held to 0.05: rejected. Both are confirmed, and neither would be refuted (the opposite p values are far above 0.05).
- **Neither p is at the floor.** The smallest attainable p is 1/63,504 = 1.6e-5, reached only with complete separation in both variants. G's p counts 48 relabellings at least as extreme as the observed one, and B's 80.
- **Per variant.** The G shift reaches p ≤ 0.05 in each variant on its own (0.040 and 0.0079). The B shift does in v1 (0.0040, the floor of 1/252: complete separation) but not in v3 (0.063). These are descriptive.
- **Dropped networks:** none (`dropped` empty).

### 5.4 Robustness (computed here; descriptive)

Neither check was pre-registered, and neither changes a decision.
- **On the keys valid for every network.** The arms' exclusion sets differ, and "argmin off plateau" depends on the network's own line (Section 4.5).
  - Method: the 296 of 427 primary non-capped keys with no exclusion of any kind in any of the 20 networks (step 2's `stages.edge_key_measures`, with the gate2-trained keys excluded; its kept counts reproduce Section 4.5's "keys used"). Per network, the plain median over those keys (not pair-weighted). Then H4R's statistic and its exact enumeration.
  - G: statistic −0.155 (v1 −0.068, v3 −0.086), p 3.1e-5 (2 of 63,504). v3 separates completely (real-width 0.712–0.745, baselines 0.803–0.849). v1 overlaps slightly (real-width 0.735–0.768, baselines 0.758–0.852).
  - B: statistic +0.740 m (v1 +0.368, v3 +0.372), p 3.1e-5. v3 separates completely (1.24–1.37 against 0.76–1.13 m). v1 overlaps slightly (1.12–1.49 against 0.75–1.18 m).
  - So the result does not come from the arms' different key sets. On common keys it is stronger, and the v3 B shift is as large as v1's.
- **Without the five networks with a failing gate2-trained key.** Dropping `h4r-v1-realw-s10`, `h4r-v1-realw-s12`, `h4r-v3-base-s10`, `h4r-v3-base-s13` and `h4r-v3-realw-s10` leaves 3 to 5 networks per arm and variant (1,960 relabellings), testable under the design's minimum of 3.
  - G: statistic −0.122 (v1 −0.058, v3 −0.064), p 0.0056.
  - B: statistic +0.578 m (v1 +0.396, v3 +0.182), p 0.018.
  - Both would pass Holm (0.0056 ≤ 0.025, then 0.018 ≤ 0.05). The decisions do not depend on the networks whose autodiff check failed at one key.

## 6. Secondary results

### 6.1 S1: completion (secondary, decided)

Source: `decisions.json` (`secondary.S1`); circuits completed per network are in Section 5.2. A circuit is completed when more than half of its 10 timed laps finish (the 10 laps: [width-test.md](width-test.md), Section 4; the rule: `scripts/h4r.py`, `COMPLETION_MIN`, applied to each circuit's completion fraction in `eval_rdev.json`).

| Variant | Real width (seeds 10–14) | Mean | Baseline (seeds 10–14) | Mean | Difference | Per-variant p (252) |
|---|---|---|---|---|---|---|
| v1 | 4, 8, 9, 9, 8 | 7.6 | 6, 4, 7, 5, 3 | 5.0 | +2.6 | 0.044 |
| v3 | 8, 8, 6, 8, 9 | 7.8 | 6, 8, 8, 6, 8 | 7.2 | +0.6 | 0.32 |

- **S1: confirm.** Statistic +3.2 circuits, p 0.026 (1,650 of 63,504), opposite p 0.99. All 20 networks entered: field exclusions do not apply to S1.
- **It replicates the width test's direction on fresh seeds, mainly through v1.** The width test (exploratory, 4 real-width seeds against 5 baselines per variant) found 8.25 against 5.4 in v1 and 8.5 against 5.8 in v3 ([width-test.md](width-test.md), Section 1). H4R's v1 gain is similar (+2.6 against +2.85). Its v3 gain is small (+0.6 against +2.7), mainly because H4R's v3 baselines completed more circuits (7.2 against 5.8), and its real-width networks slightly fewer (7.8 against 8.5).

**Per circuit** (descriptive; computed here from `manifest.json`, `completion`): seeds of 5 completing each circuit.

| Circuit | v1 baseline | v1 real width | v3 baseline | v3 real width |
|---|---|---|---|---|
| Brands Hatch | 3 | 3 | 5 | 4 |
| Catalunya | 3 | 5 | 5 | 5 |
| Hockenheim | 2 | 3 | 3 | 4 |
| Melbourne | 2 | 3 | 0 | 2 |
| Moscow Raceway | 2 | 3 | 4 | 4 |
| Norisring | 0 | 1 | 0 | 0 |
| Nürburgring | 4 | 5 | 5 | 5 |
| Oschersleben | 2 | 2 | 4 | 4 |
| Sepang | 5 | 5 | 5 | 5 |
| Sochi | 0 | 2 | 1 | 1 |
| Yas Marina | 0 | 1 | 0 | 0 |
| Zandvoort | 2 | 5 | 4 | 5 |

- Real-width networks complete each circuit at least as often as baselines in each variant, except Brands Hatch in v3 (4 against 5).
- Norisring and Yas Marina, which no in-sample real-width run of the width test completed, are each completed by one v1 real-width seed here (`h4r-v1-realw-s12` and `h4r-v1-realw-s13`). These circuits' widths were in the training profiles, so this is in-sample.

### 6.2 S2: paired by seed (secondary, decided)

Source: `decisions.json` (`secondary.S2`). Each difference is real-width minus baseline for the same seed and variant.

| Pair | ΔG | ΔB (m) |
|---|---|---|
| v1 / s10 | −0.110 | +0.246 |
| v1 / s11 | −0.032 | +0.608 |
| v1 / s12 | −0.048 | +0.598 |
| v1 / s13 | −0.050 | +0.594 |
| v1 / s14 | −0.008 | +0.163 |
| v3 / s10 | −0.084 | +0.183 |
| v3 / s11 | −0.085 | −0.153 |
| v3 / s12 | −0.065 | +0.626 |
| v3 / s13 | −0.094 | +0.639 |
| v3 / s14 | −0.016 | +0.055 |
| **Mean** | **−0.059** | **+0.356** |
| **p** (1,024 sign flips; one-sided) | **9.8e-4** (1/1,024) | **2.9e-3** (3/1,024) |
| Decision | confirm | confirm |

- **G is lower in all 10 pairs.** The smallest drops are seed 14 in both variants (−0.008 and −0.016). The p is the smallest attainable.
- **B is higher in 9 of 10.** The exception is v3 seed 11 (−0.15 m), the network whose G and B rest on the fewest geometries (35; Section 5.2).
- The pooled tests (Section 5.3) do not rely on the pairing; S2 asks whether the same comparison holds seed by seed.

### 6.3 S3: does the margin go with completion? (secondary, descriptive by design)

Source: `decisions.json` (`secondary.S3`). Spearman's ρ of a network's median B and its circuits completed, within each arm. The 95% CI is a percentile bootstrap over networks (20,000 draws, stream 201). Draws in which ρ is undefined (all ranks tied in a resample) are left out; their number is given.

| Arm | Pooled over v1 and v3 (n = 10) | v1 (n = 5) | v3 (n = 5) |
|---|---|---|---|
| Baseline | **0.55 [−0.09, 0.93]** (0 undefined) | 0.10 [−1, 1] (32) | 0.58 [−0.40, 1] (1,732) |
| Real width | **−0.14 [−0.79, 0.65]** (13 undefined) | −0.32 [−1, 1] (412) | 0.22 [−1, 1] (1,550) |

Checked here: the point estimates reproduce from the per-network B and completion values.
- **Descriptive (no decision rule): both pooled CIs include 0.** Both span more than 1.0. With 5 networks the per-variant CIs are uninformative.
- **Nothing distinguishes the arms.** The point estimates differ in sign (0.55 baseline, −0.14 real width), but the CIs overlap almost entirely.
- **Pooling over variants adds a variant effect** (computed here): v3 baselines have higher B (mean 1.30 against 1.09 m) and complete more circuits (7.2 against 5.0) than v1 baselines, so part of the baseline arm's pooled ρ may come from the variant. Within v1 it is 0.10, within v3 0.58.
- **What this does not show.** It does not show that the margin explains the completion gain, nor that it does not. With 10 networks per arm, a correlation of moderate size cannot be resolved: B spans 0.54 m within the real-width arm and 0.82 m within the baseline arm (both variants pooled), and completion 4–9 and 3–8 circuits.

## 7. Descriptive results

### 7.1 T and G_out: tracking is apparently rearranged

Sources: `decisions.json` (`networks.<run>.T`, label); G_out computed here.
- **Every network's T is labelled "mixed".** Medians run from 0.724 to 0.802, and every CI lies within [0.687, 0.854]. No network is width-blind (CI upper end ≤ 0.5) or corridor-anchored (CI within [0.8, 1.2]). The optimum's T* is 1.000.
- **No shift in T was detected** (computed here, by H4R's statistic and exact enumeration on median T; descriptive). Statistic +0.021 (v1 +0.020, v3 +0.001), two-sided p 0.32. Real-width 0.735–0.802, baselines 0.724–0.782, which overlap. This is an absence of evidence for a shift, not evidence that there is none, and no bound on its size was computed.
- **G_out is higher in every real-width network** (computed here: each network's `stages.median_statistic` of its per-key G_out, same keys and rules as G).
  - Baselines: −0.092 to −0.025, all negative (a wider outside moves the line slightly inward).
  - Real width: −0.028 to +0.041.
  - The arms separate completely in both variants (v1: real-width −0.028 to +0.037 against baselines −0.092 to −0.037; v3: −0.003 to +0.041 against −0.074 to −0.025).
- **Reading.** As in step 2 (its Section 7.1), the real-width networks' line is placed a little less by the inside edge and a little more by the outside, while no change was detected in how far it follows a whole-corridor shift. A median of sums is not a sum of medians, so G + G_out is not exactly T here.

### 7.2 Pooled with step 2's networks

Source: `decisions.json` (`descriptive.pooled_with_step2`). Step 2's values are its committed confirmatory decisions (G its H4 statistic; B and T its pooled medians). Group medians:

| Source | Variant | Arm | n | Median G | Median B (m) | Median T |
|---|---|---|---|---|---|---|
| H4R | v1 | baseline | 5 | 0.800 | 0.98 | 0.752 |
| H4R | v1 | real width | 5 | 0.750 | 1.54 | 0.760 |
| H4R | v3 | baseline | 5 | 0.810 | 1.32 | 0.767 |
| H4R | v3 | real width | 5 | 0.728 | 1.60 | 0.772 |
| Step 2 | v1 | baseline (seeds 0–4) | 5 | 0.811 | 1.23 | 0.768 |
| Step 2 | v1 | real width (seeds 0–3)† | 4 | 0.742 | 1.51 | 0.755 |
| Step 2 | v3 | baseline (seeds 0–4)† | 5 | 0.774 | 1.35 | 0.751 |
| Step 2 | v3 | real width (seeds 0–3) | 4 | 0.687 | 1.64 | 0.761 |

† Includes one network that was descriptive in step 2 for failing gate2-trained (`dr-realw-v1-s1`; `dr-pilot-v3-tilt50a2-s2`).

**Shift by study** (computed here; mean real-width minus mean baseline, step 2 against its baseline seeds 1–4 as in its decision test).

| Measure | Variant | H4R | Step 2 (exploratory) |
|---|---|---|---|
| G | v1 | −0.049 | −0.081 |
| G | v3 | −0.069 | −0.077 |
| B (m) | v1 | +0.44 | +0.28 |
| B (m) | v3 | +0.27 | +0.29 |
| T | v1 | +0.020 | −0.015 |
| T | v3 | +0.001 | +0.007 |

- **Same direction in all four source × variant strata**, for both G and B.
- **Sizes: smaller or larger in v1, similar in v3; nothing here separates them.** H4R's G shift is about 60% of step 2's in v1 (smaller) and 90% in v3 (similar). Its B shift is about 155% of step 2's in v1 (larger) and about 90% in v3 (similar). With 4–5 networks per arm, these are noisy point estimates. So H4R does not say how much, if any, of step 2's shift came from its code and hardware confound. It does say that the shift does not need the confound.
- **The baselines' G is similar across studies** (H4R 0.729–0.864, step 2 0.760–0.855). H4R's v1 baselines have less margin than step 2's (median 0.98 against 1.23 m), which makes H4R's v1 B shift larger. The real-width arms' B is similar across studies.

### 7.3 Per regime

Source: `decisions.json` (`networks.<run>.regimes`). The A2 cells are typed by step 2's seed-0 network of each variant (G1 for v1, G3 for v3; the record's `regime_typing` and `regime_cells`), using step 1's regimes, which compare a corner's radius with the q1 and q10 percentiles of the typing network's training grammar: **below** the support (R < q1, not clipped), **tail** (q1 ≤ R < q10), **bulk** (R ≥ q10) and **clip** (tighter than the radius at which the curvature input saturates). Each value is the network's median over geometries within the regime. The table gives the arm means of those values (computed here).

| Variant | Regime (keys) | G: baseline | G: real width | ΔG | B: baseline (m) | B: real width (m) | ΔB (m) |
|---|---|---|---|---|---|---|---|
| v1 | bulk (185) | 0.863 | 0.782 | −0.082 | 1.03 | 1.53 | +0.50 |
| v1 | tail (138) | 0.778 | 0.748 | −0.030 | 0.97 | 1.25 | +0.27 |
| v1 | below (72) | 0.550 | 0.555 | +0.004 | 1.76 | 1.81 | +0.05 |
| v3 | bulk (283) | 0.834 | 0.751 | −0.083 | 1.22 | 1.57 | +0.35 |
| v3 | tail (80) | 0.805 | 0.755 | −0.049 | 1.04 | 1.06 | +0.03 |
| v3 | below (56) | 0.585 | 0.567 | −0.018 | 2.18 | 1.77 | −0.42 |

Differences are computed from the unrounded arm means.
- **The shifts sit in the bulk.** On bulk corners real-width training lowers G by about 0.08 in both variants and raises B by 0.35–0.50 m. On corners below the training support G differs little (+0.004 in v1, −0.018 in v3; untested), and in v3 B is lower in the real-width arm.
- **G is lower below the support than in the bulk in all 20 networks** (0.42–0.70 against 0.72–0.91). Step 2 found the same in 25 of its 26 networks (its Section 7.12: 0.38–0.76 against 0.66–0.89).
- **Clip.** The clip regime (32 keys in v1, 8 in v3) is defined for only 3 networks per arm in v1 and 1 per arm in v3, because many of its keys are excluded: κ clipping (v1 only, 8–16 of 32 keys) and runs that crash before the window end (invalid outcome: 4–16 of 32 in v1; 4–8 of 8 in v3, except `h4r-v3-base-s14` and `h4r-v3-realw-s12`, which keep all 8). That leaves no defined mirror pair in the other networks; the regime has 4 pairs in v1 and 1 in v3, and each network where it is defined rests on one pair (computed here with `scripts/h4r.py`'s `h4r_key_measures` and step 2's pair rule on the record's clip cells). It is left out of the table.
- The regimes are typed by step 2's networks, not by each H4R network's own training support, and the regime sets differ by variant.

### 7.4 Left and right

Source: `decisions.json` (`networks.<run>.<G|B|T>.left`, `.right`, `.asymmetry`). The asymmetry index is m_L − m_R over mirror partners, with a cluster-bootstrap CI.
- **The asymmetry index's CI excludes 0 in 13 of 20 networks** for each of G, B and T, with mixed signs: G 7 positive and 6 negative (|index| ≤ 0.14); B 6 and 7 (−0.51 to +0.63 m); T 6 and 7 (|index| ≤ 0.08). The sign differs between seeds of the same arm and variant. As in step 2, the learned asymmetry shows mainly in where the line sits (B), and much less in how it tracks (G).
- **The arm shifts hold on both turn directions** (computed here; arm means of the per-direction medians).

| Measure | Variant | Left: real width − baseline | Right: real width − baseline |
|---|---|---|---|
| G | v1 | −0.020 (0.762 − 0.782) | −0.083 (0.729 − 0.812) |
| G | v3 | −0.108 (0.721 − 0.829) | −0.044 (0.718 − 0.762) |
| B (m) | v1 | +0.29 (1.44 − 1.15) | +0.63 (1.68 − 1.05) |
| B (m) | v3 | +0.18 (1.39 − 1.21) | +0.42 (1.80 − 1.38) |

- G is lower and B higher on both sides in both variants. The B increase is larger on right turns in both variants. The real-width generator swaps sides with probability 0.5 ([width-test.md](width-test.md), Section 3), so there is no built-in reason for that. No test was run, and the asymmetry indices show no clear pattern by arm (B: real width 4 positive and 6 negative, baselines 5 and 5).

## 8. What it means

**Real-width training lowers corner-wide edge tracking and raises inside margin: confirmed.**
- On 20 networks trained after the hypotheses were fixed, on one machine with one commit of the training code, both pre-registered primary tests confirm under Holm: G is lower (by 0.049 in v1 and 0.069 in v3, arm means) and B is higher (by 0.44 m and 0.27 m).
- The paired comparison agrees: G is lower in all 10 same-seed pairs and B higher in 9.
- It holds on the keys valid for every network and without the five networks whose autodiff check failed at one key (Section 5.4).
- This removes the three weaknesses of step 2's exploratory result. The gate rule was fixed in advance and did not void the test. B's direction was pre-specified. The code and hardware confound is gone. The G and B parts of step 2's exploratory reading (its Section 8) are now confirmed for corner-wide tracking; its T and G_out parts remain descriptive. The two confirmed shifts are separate arm-level results, not a trade-off between networks: within each arm, G and B show no consistent relation (Spearman's ρ +0.19 baseline, −0.16 real width; computed here from `decisions.json`).

**The change shows mainly in where the line sits; no change was detected in how much it follows the corridor.**
- No shift in T was detected (+0.021, two-sided p 0.32; descriptive), and G_out rises (Section 7.1). The real-width networks' line is placed a little less by the inside edge and a little more by the outside, and it sits further from the inside.
- The shifts sit on bulk corners. Below the training support G differs little (+0.004 and −0.018; untested; Section 7.3).
- Lower G is not by itself a defect. The optimum's G* = 1 is the gain of a line that touches the inside edge, and a line that keeps a margin need not follow the edge one-for-one (step 2, Section 8). Here the real-width arm, which tracks less, also completes more circuits (S1). Within each arm the association goes the other way (Spearman's ρ of G with circuits completed +0.37 baseline, +0.56 real width; computed here from `decisions.json`), so this says nothing about G and completion network by network.

**Together with step 2's locality result and the width test.**
- **Locality** ([locality.md](locality.md), exploratory, on step 2's networks). The networks' gain falls from about 0.8 for corner-wide shifts to about 0.1–0.3 for changes a few metres long. At short scales no difference between real-width and baseline networks was detected (two-sided p 0.13–0.81). Their lower corner-wide G came from the slow, far-upstream part of the response.
  - H4R confirms the corner-wide drop on fresh networks. It did not measure local tracking. That the real-width networks follow local edge changes no better than baselines therefore still rests on step 2's networks, unconfirmed.
- **Width test** ([width-test.md](width-test.md), exploratory). Real-width training raised R-dev completion. S1 replicates the direction on fresh seeds (p 0.026), mostly in v1. As there, the evaluated circuits' widths were in the training profiles, so part of the gain may be leakage; the width test's leave-out runs kept about half of it (p 0.063, suggestive).
- **A hypothesis that links the three** (not tested). Step 2's first-order crash rule (its Sections 2 and 8) says a local inside-edge intrusion of Δ crashes the car when (1 − r)Δ > B, where r is the share of the intrusion the line follows (about 0.1–0.3 for changes a few metres long, by the locality result). If r is the same in both arms (locality could not distinguish the arms at short scales for step 2's networks, which is not the same as finding r equal), then a larger B buys tolerance almost one-for-one: arithmetic on arm means with r = 0.15 gives B/(1 − r) of 1.28 m against 1.80 m in v1, and 1.53 m against 1.84 m in v3 (a ratio of means, not a measured quantity). That would fit more completions on real roads, where step 1 found that 159 of 160 corner crashes leave by the inside edge, mostly where the inside width changes within the last 100 m ([width-test.md](width-test.md), Section 2). For corner-wide shifts the lower G offsets part of the extra margin in v1 and more than all of it in v3: B/(1 − G), the corner-wide narrowing that would use up the buffer, is 5.4 m (baseline) against 6.1 m (real width) in v1, and 6.8 against 6.0 m in v3 (same arithmetic on arm means). Either way it is far beyond the A2 corners' range, as in step 2 (its Section 7.2).

**What is not established.**
- **That the margin explains the completion gain.** S3 is descriptive (no decision rule): within-arm correlations of B and completion are 0.55 and −0.14, with CIs that include 0 and span more than 1.0. The hypothesis above is neither shown nor clearly supported by the data. On the keys valid for every network (Section 5.4) the B shift is the same in v1 and v3 (+0.37 m each), while the completion gain is about four times larger in v1 (+2.6 against +0.6 circuits), and within the real-width arm B does not go with completion (ρ −0.14).
- **How the line follows local edge changes in the H4R networks.** Not measured here (Section 10).
- **Lap time.** The width test found real-width networks slightly slower on circuits both arms complete. H4R's laps were not analysed.
- **The mechanism.** Why real-width training lowers corner-wide G and raises B is not addressed. One possibility is that real profiles make the inside edge an unreliable guide, so the line keeps away from it instead of following it. Nothing here tests that.
- **Generality.** G and B are measured on synthetic A2 corners at the typical approach, from shared start states, for corner-wide trapezoid shifts. Two grammars, one PPO configuration, 200M steps, 5 seeds per arm per variant. Not real roads, not the agents' own states.
- **How much of step 2's shift came from its confound.** The H4R shifts are of the same order as step 2's (G smaller and B larger in v1, both similar in v3), and neither study can separate them (Section 7.2).
- **Nothing about R_TEST.** It stays sealed and was not used.

## 9. Deviations and procedure log

### 9.1 Before the tag

| Date (EDT) | What | Effect |
|---|---|---|
| 22 Oct 10:37:49 | Design committed (`612ed7a`), before any H4R network was trained | As designed |
| 22 Oct 10:45:52 → 11:32:16 | **The locality analysis ran while H4R trained.** Its plan (`291128d`) and results (`ea2fbf1`) were committed after the H4R design and before the H4R code and tag. Its E4 reading compares real-width with baseline networks, but on step 2's networks | None on H4R's decisions. H4R's hypotheses, directions and tests were fixed at `612ed7a`, before the locality plan. Design Section 7 (`5453a2e`, 12:06) was written after the locality results; it changes no computation (each item is what step 2's frozen code or `scripts/h4r.py` already did) |
| 22 Oct 12:06:48 | `scripts/h4r.py`, tests and design Section 7 committed (`5453a2e`), "before any H4R run was pulled back or any of its results … was seen" (design Section 7). Training was 12 of 20 done (`runs/activity.md`, 12:08) | As designed |
| 22 Oct 12:08 | The costate build (idea 2) was started in a separate worktree, so the main checkout's `src/` stayed untouched for H4R (`runs/activity.md`) | None. `src/` was unchanged until after the decisions were committed (Section 9.2) |
| 22 Oct 12:27:03 → 12:58:14 | **One transient CUDA fault.** The first attempt of `h4r-v1-base-s14` failed at 12:47:40 and was re-run once from scratch with the same arguments; the retry ran mostly alone on the GPU, at 386k steps/s (Section 4.2) | As designed (design Section 2). The run is present, not missing |
| 22 Oct 12:59 | Runs pulled back; final checkpoints matched the box's sha256; box deleted (`runs/activity.md`) | As designed |
| 22 Oct 13:00 | **One network's R-dev completion was seen before the tag.** Printing the manifest showed it (`runs/activity.md`, 13:00). The committed manifest (`593d42e`, 13:00:15) carries every network's completion, as the record binds every evaluation | None on decisions. S1 and its test were fixed in the design before training, and nothing changed after. S1's inputs exist before any tag by construction, because the evaluation is written at the end of training (design Section 2). No field of any H4R network existed before the tag |
| 22 Oct 13:02:21 | Record committed alone (`eb5637e`); annotated tag `h4r-prereg-v1` placed and pushed; on origin as the same object (checked here with `git ls-remote`) | As designed |

- **A misstatement in the design text.** Design Section 2 describes `2143a59` as "the code of the width-test real-width runs". Those in-sample runs used `83ebe56`; `2143a59` came later and added `--real-widths-from` for the leave-out runs ([width-test.md](width-test.md), Section 3; `git diff 83ebe56 2143a59` changes `scripts/train_dr.py` and the generator). No effect: the claim that matters, that the training code is unchanged since `2143a59` and that both H4R arms share one commit, holds (Section 2.3).

### 9.2 After the tag

| Date (EDT) | What | Effect on results |
|---|---|---|
| 22 Oct 13:02:41 → 13:05:07 | harness-f32; the tag pinned at 13:04:26 and the pin committed at once (`fa10eff`, 13:05:05) | As designed |
| 22 Oct 13:05:07 → 13:22:48 | gate2-trained: 5 networks fail one key each; none descriptive (Section 4.3) | As designed under H4R's rule: one key excluded for each |
| 22 Oct 13:22:48 → 16:23:03 | fields, 9,620 rows, one network after another (stamps 13:22:49 to 16:23:01) | As designed |
| 22 Oct 15:56:59, 15:57:16 | **Two commits during the fields stage**, both to `scripts/dashboard.py` only (`99bf0e4`, `8b4a097`) | None. That file is outside the code the stamps guard (`src/`, `scripts/fields.py`, `scripts/_bootstrap.py`, `scripts/h4r.py`) and is not imported by `scripts/h4r.py`. The analyse stamp's HEAD is `8b4a097` for this reason |
| 22 Oct about 16:24 | `analyse --check`: inputs complete and stamped; nothing written | As designed |
| 22 Oct 16:24:55 → 16:27:10 | **The one confirmatory run**, committed at once (`b5f0854`) | As designed |
| 22 Oct 16:44:05 | **Idea 2 (costate) code merged into main** (`c7d2285`, merge `935112f`), built in a separate worktree from 12:08. It adds new files only (`src/apexzero/costate/`, `scripts/costate.py`, `tests/test_costate_*.py`) | None. It came after the decisions were committed, which is when the freeze on `src/` ends (`scripts/h4r.py`, module docstring). It touches nothing H4R reads: `src/apexzero/fields/`, `scripts/fields.py`, `scripts/_bootstrap.py` and `scripts/h4r.py` are unchanged since the tag |
| 22 Oct 16:47:49 → 16:50:39 and 19:29:20 → 19:30:06 | **Step 2's gate tests run after its tag.** With an uncommitted change to `tests/test_fields_gates.py` (`runs/activity.md`, 16:51), step 2's test runs into a tmp `--out` logged 20 "pre-tag stage after the tag" deviations ("test run after the tag (tmp --out)") into step 2's local `outputs/fields/deviations.jsonl` | None on H4R. They are step 2 test runs into a temporary directory. Nothing under `outputs/h4r/` or `runs/h4r*` is newer than `decisions.json` |
| 22 Oct 19:37:29 | **`9ac2b30`** changes `tests/test_fields_gates.py` (step 2's post-tag test flag, +14/−2) and `docs/interp/costate-design.md` (idea 2's design) | None. Neither is guarded code or read by `scripts/h4r.py`; the guarded paths are unchanged since the tag (below) |
| 22 Oct (this write-up) | Robustness checks, G_out, T's test and per-arm summaries computed for this document from the committed decisions and the field rows, without running any network. The write-up was paused at 16:51 (`runs/activity.md`) and resumed later the same day; every number was re-checked against its source on resuming | Labelled descriptive. No decision changed |

**Checks with nothing to report** (computed here).
- All 44 provenance stamps under `outputs/h4r/` (the manifest, 21 harness-f32, gate2-trained, 20 fields, analyse) carry code hash `651e7eb4…` and `scripts/h4r.py` sha256 `43233a88…`. None has a deviation, a changed path or a partial run. The 43 post-tag stamps carry the tag `h4r-prereg-v1` (object `368faefc…`).
- No guarded code changed after the tag. Up to the decisions commit, `git diff h4r-prereg-v1 b5f0854` touches only `outputs/h4r/analyse/decisions.json`, the tag pin and `scripts/dashboard.py`. After it, the idea 2 merge adds only new costate files, and `9ac2b30` changes only step 2's test helper and the costate design (above). At HEAD `9ac2b30`, `git diff h4r-prereg-v1 HEAD -- src/apexzero/fields scripts/fields.py scripts/_bootstrap.py scripts/h4r.py` is empty. Step 2's frozen code is unchanged since its own tag (`git diff receptive-fields-prereg-v1 HEAD -- src/apexzero/fields scripts/fields.py scripts/_bootstrap.py` is empty).
- No loop was exhausted: gate2-trained `n_exhausted` is 0 for every network, and the longest field run used 203 of 256 steps.
- No network was descriptive and none was missing.
- No R_TEST circuit was used.

**Records that are not in the repository.**
- `runs/activity.md` is a local, untracked notes file with minute-resolution times. It is the only record of the box's GPU type (Massed Compute RTX A6000), of the commit the box was set up at, of the sha256 match on pull-back, of the `analyse --check` run and of the 13:00 note. The run configs record neither commit nor hardware.
- The box was deleted after the pull-back. Its queue log `runs/h4r.log` is local, but its sha256 is bound in the committed manifest.
- The stage logs (`runs/h4r-*.log`) and all outputs other than the five committed files (H4R's three above, and step 2's `outputs/fields/analyse/decisions.json` and `outputs/fields/tag.json`) are local.

## 10. What next

1. **Local tracking on the H4R networks (exploratory; write and commit the plan first).** The per-key impulse responses of all 20 H4R networks on all 481 A2 keys are on disk (`outputs/h4r/fields/arrays/`). The locality analysis's profiles (Gaussian bumps at the closest point, short ramps and plateaus, distance-resolved bumps) can be computed from them without running a network. That would test, on same-machine, same-code networks, whether real-width training changes local following, which locality's E4 could not distinguish on step 2's networks.
2. **Does the margin explain completion?** S3 cannot answer this. Options, each needing a plan before any computation:
   - per-circuit crash analysis of the H4R evaluations: do the real-width networks' extra completions come from inside-edge crashes avoided, and is the closest-approach clearance at the former crash points larger by about the B shift?
   - the step 1 crash ladders, or an H3-type local response, on the H4R networks at the real crash corners (R-dev roads only).
3. **Lap time.** The gap to the optimal lap on circuits both arms complete, from the H4R evaluations (descriptive).
4. **A training target.** If the aim is to follow local edge changes rather than to keep away from them, the outcome to train on and test is the short-scale response (the locality analysis's L/e for bumps of a few metres, or the kernel's peak and lag), not corner-wide G. A new pre-registration of a training change should name it.
