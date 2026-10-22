# H4R: a fresh, pre-registered test of what real-width training does to edge tracking

Status: 22 Oct 2026. This design is committed **before any H4R network is trained**. The
analysis code, the checkpoint manifest and the tag `h4r-prereg-v1` follow before any of the new
networks' fields are computed. R_TEST stays sealed.

## 1. Why

Step 2's H4 asked whether training on real width profiles raises the edge-tracking gain G. It
could not be decided ([findings](../findings/receptive-fields-step2.md), Section 5.3).
- Two of its 18 networks each failed the post-tag autodiff gate (gate2-trained) at a single key,
  on the c_min field only, with errors of 1.7e-3 and 1.0e-3 against a 1e-3 limit.
- Under the strict rule fixed before the tag, a failing network became descriptive, and H4 became
  not testable.
- Re-running the gate gives the same result, and changing the rule after the fact would be post
  hoc. The existing networks cannot give a confirmatory answer.

Step 2's exploratory analysis, H4's frozen test with the two failures ignored, pointed the other
way from H4's prediction:
- real-width training **lowered** G (opposite one-sided p 2.0e-4);
- it **raised** the inside buffer B (two-sided p 0.002).

That analysis carries three weaknesses:
- it is post hoc in the gate rule;
- B's test and direction were chosen after the confirmatory results;
- the arms differ in code and hardware: the baselines' seed 0 ran on the Mac CPU before R18's
  PPO refactor, and the real-width runs ran on a different GPU.

H4R tests the same question on networks that do not exist yet, trained on one machine with one
commit of the training code.

## 2. Networks (fixed now)

| | Variant v1 (base grammar) | Variant v3 (tilted grammar) |
|---|---|---|
| Baseline | `h4r-v1-base-s{10..14}` | `h4r-v3-base-s{10..14}` |
| Real widths | `h4r-v1-realw-s{10..14}` | `h4r-v3-realw-s{10..14}` |

- 20 runs: 2 variants × 2 arms × seeds 10–14. These seeds are not used by any existing run.
- Every run uses `scripts/train_dr.py` at this commit, launched by `scripts/gpu_queue_h4r.sh` (4 lanes; with 4 lanes the job order makes each round one network of each arm × variant). The training code (`train_dr.py`, the
  generator, the env, the agents) is unchanged since `2143a59`, the code of the width-test
  real-width runs.
  - `--steps 200e6 --seed s`;
  - v3 adds `--tilt-fraction 0.5 --tilt-alpha 2`;
  - real widths add `--real-widths`, using the profiles of all 12 R-dev circuits, as
    `dr-realw-v1-*` and `dr-realw-v3-*` did;
  - every other argument is train_dr.py's default, as in the step 2 networks (`num_envs` 2048,
    `num_steps` 64, MLP, bank 4096, refresh 256).
- **One machine.** All 20 run on one rented NVIDIA RTX A6000 box (Massed Compute), 4 runs at a
  time, with one environment set up by `scripts/gpu_setup.sh`.
  - Lanes are assigned so that each round mixes the arms and both variants.
  - A run that fails is re-run once with the same arguments. If it fails again, it is missing
    (Section 5).
- **End of training.** Each run's standard R-dev evaluation (`eval_rdev.json`) is written at the
  end of training, as for every earlier run.

## 3. States, measures and code

- **States.** Step 2's frozen A2 key list (`src/apexzero/fields/prereg.json`, set `a2`): the
  primary non-capped keys, 427 of them, in 62 mirror pairs over 48 geometries. These are
  agent-independent states, the same for every network.
- **Measures.** Exactly step 2's, computed with step 2's frozen code (`src/apexzero/fields` at
  `receptive-fields-prereg-v1`, imported read-only):
  - G, the inside-edge tracking gain;
  - B, the inside buffer;
  - T, as in step 2;
  - each network's statistic is its median over mirror pairs, under step 2's pair-completeness,
    exclusion and coverage rules (design Section 15).
- **H4R's own code** (`scripts/h4r.py`, plus tests) only drives step 2's functions on the new
  networks and runs the tests below. It is written, reviewed and committed before the tag.

## 4. Gates (rules fixed now)

- **Checkpoint manifest.** After training, each run's final checkpoint (path, timesteps,
  sha256) is recorded before the tag. Every later stage loads the recorded file and checks its
  hash.
- **Wiring (harness-f32).** As step 2's: each network's float32 harness run on every A2 start
  is compared with its float64 loop for branch consistency. A failing key is a pipeline
  exclusion.
- **gate2-trained, changed from step 2 and fixed now.** The check runs on the same 12 keys
  (`prereg.json`, `gate2_trained`) with Gate 2's per-key rule. A key that fails is excluded for
  that network, as a pipeline exclusion counted in coverage. The network becomes descriptive only
  if more than 2 of its 12 keys fail.
  - The reason for the change: step 2's failures were single, marginal keys, and step 2's rule
    let one such key void a whole hypothesis.
- **Coverage.** Step 2's 30% rule. A network above it is descriptive.

## 5. Hypotheses and decisions

The direction of each primary test comes from step 2's exploratory result, on networks that
played no part in forming it.

**H4R-G (primary): real-width training lowers G.**
- Statistic: each network's median G.
- Test: one-sided exact permutation test of real-width below baseline, with labels exchanged
  within each variant. With 5 per arm per variant there are C(10, 5)² = 63,504 relabellings.
- **Confirm** if p ≤ its Holm-adjusted level.
- **Refute** if the opposite one-sided p ≤ 0.05. That is real-width training raising G, step 2's
  original H4 direction.
- **Inconclusive** otherwise.

**H4R-B (primary): real-width training raises B.** The same test on each network's median B,
with real-width above baseline.

**Family.** Holm across the two primary tests at α = 0.05.

**Missing or descriptive networks.**
- The tests run on the networks that remain, all relabellings of the remaining sets enumerated.
- A test is not testable if fewer than 3 networks remain in any arm of either variant.
- Every dropped network is listed with its reason.

**Secondary (pre-registered):**
- **S1. Completion.** Real-width training raises R-dev completion (the number of circuits
  completed), with the same test, one-sided. This replicates the width test on fresh seeds.
- **S2. Paired by seed.** The same seed in both arms of a variant forms a pair. A sign-flip
  permutation of the 10 within-pair differences (1,024 flips), for G and for B.
- **S3. Transfer of the margin story.** Within each arm, the Spearman correlation between a
  network's B and its R-dev completion. Descriptive, with CIs.

**Descriptive:**
- per-network G, B and T with CIs;
- the left and right directions;
- the per-regime breakdown;
- the pooled picture with step 2's 18 H4 networks, each source labelled.

## 6. Procedure and budget

1. **This design is committed.** Then the box is rented under the standing permission for the
   cheapest Massed Compute instance, set up, and the 20 runs started.
   - About 2.2 hours, 5 rounds of 4 at about 26 minutes per round.
2. **In parallel:**
   - `scripts/h4r.py` and its tests are written;
   - it gets an adversarial code review, and the fixes are applied.
3. **After training:**
   - the runs are pulled back and the box deleted;
   - the manifest is written and committed;
   - the tag `h4r-prereg-v1` is placed and pushed.
4. **Post-tag, on the Mac, in this order:**
   - harness-f32;
   - gate2-trained;
   - the fields on the A2 keys, about 2.9 hours for 20 networks × 481 keys;
   - the analysis, once;
   - the decisions committed.
5. **The findings** are written and verified as for steps 1 and 2.
