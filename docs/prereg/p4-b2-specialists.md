# P4, option B2: five fresh Hockenheim specialists (pre-registration, DRAFT)

**Status.** DRAFT, 26 Oct 2026, for review. Nothing here is tagged. On 27 Oct 2026 the user
accepted the draft's default for each of its five choices (Section 11) and approved its box time
(Section 8). No specialist below has been trained, and nothing in this document has been computed
on one. After review the lead commits the final text, the five runs are trained, their manifest,
d_OOD buffers and start states are built and committed, and the tag `p4-b2-prereg-v1` is placed
before any of them sees a protocol stimulus (Section 8). Only Hockenheim, an R-dev circuit, is
used. R_TEST stays sealed and `outputs/sealed/` is not read.

**Build (27 Oct 2026).** The code is written, tested and committed (`40ebdf6`): the run list
`scripts/p4b2_runs.py`, the pipeline `scripts/p4_b2.py` and `tests/test_p4_b2.py`. It runs
through the shared GPU-round queue (`scripts/gpu_round.sh`, `scripts/round_manifest.py`), so the
queue of Section 2 is that one, not a new `scripts/gpu_queue_p4b2.sh`. Everything that does not
need S4–S8 is built and checked on the Mac: the bank's strip checks, the S1–S3 cell, start,
buffer and pipeline checks, and the CM dry run (Section 12). The manifest, the five buffers and
start states, the S4–S8 design and the record wait for the trained runs. The box is chosen in
Section 2.

## 1. Why

P4 asks whether a Hockenheim specialist reads the road or has memorised Hockenheim (protocol,
P4). It is still open:
- **Step 1, confirmatory.** S1 class 6. S2 and S3 "not testable": their d_OOD buffers were built
  only after the confirmatory run. The recipe claim "inconclusive"
  ([psychophysics-step1.md](../findings/psychophysics-step1.md), Section 4.1).
- **Step 1, post-tag rerun.** S2 and S3 class 2, but vacuously: no F6 perturbation is as far off
  the manifold as their failing F1 condition (Section 4.4).
- **Option A** ([p4.md](../findings/p4.md), exploratory). 11 road changes were certified
  irrelevant (10 planned, 1 in the post hoc extension). Their condition-level d_OOD reaches only
  0.74–1.53, against 2.01–2.75 for S2's and S3's failing conditions, so class 2 stayed vacuous.
  Option A cannot test P4 within certification.
- **Decision #7** (25 Oct 2026, second round; [follow-through](../plan/follow-through.md),
  Section 11). P4 is settled by option B2: 5 fresh specialists under a new pre-registration,
  before `protocol-v1`.

**What B2 changes:**
1. **Prospective networks.** Five specialists that do not exist yet, trained on one A6000 under
   one commit.
2. **Buffers before the tag.** Each specialist's d_OOD buffer and tier D start states are built,
   hashed and committed before the tag. The "not testable" of step 1 cannot recur.
3. **A fixed F6 set.** The 4 registered F6 conditions plus the 11 certified by option A (Section 3).
4. **A corner-matched reading, fixed now** (Section 4.2). It addresses the reason option A was
   vacuous. P4's D is a median over 7–9 corners, and certified changes move d_OOD far only at
   corners whose approach holds a long straight (p4.md Section 4: K-str reaches per-corner d_OOD
   of 2.75–17.54 at corners 7 and 14). The reading compares, corner by corner, a failing F1 cell
   with the certified changes at the same corner. It decides only when P4's own class is a
   vacuous class 2.
5. **Every outcome has a fixed wording** for `protocol-v1` and the demo (Section 5), so P4 is
   settled whatever the result.

**What to expect** (Section 6). The most likely result is that memorisation and fragility stay
unseparated. This design turns that into a settled answer instead of an open question.

## 2. Specialists (fixed now)

| ID | Run | Seed |
|---|---|---|
| S4 | `runs/hockenheim-specialist-mlp-200M-s4` | 4 |
| S5 | `runs/hockenheim-specialist-mlp-200M-s5` | 5 |
| S6 | `runs/hockenheim-specialist-mlp-200M-s6` | 6 |
| S7 | `runs/hockenheim-specialist-mlp-200M-s7` | 7 |
| S8 | `runs/hockenheim-specialist-mlp-200M-s8` | 8 |

- **Seeds.** No specialist has used seeds 4–8: S1–S3 used 1–3, and the excluded 50M run
  (`runs/hockenheim-specialist-mlp`) used 0. Generalists with seed 4 exist (`dr-pilot-*-s4`).
  A shared seed gives the same initial weights for a different task. That does not matter here,
  as it did not for S1 and S2, whose seeds 1 and 2 the R18 replicates also use.
- **Command.** `scripts/train.py --tracks Hockenheim --steps 200e6 --seed <k> --name
  hockenheim-specialist-mlp-200M-s<k>`. Every other argument is train.py's default, as for S1–S3:
  MLP 3 × 256 (tanh), 2,048 envs, 64 steps, lr 3e-4 annealed, γ 0.995, λ 0.95, entropy 1e-3 to 0.
  `config.json` records the rest.
  - Each run writes 16 log-spaced checkpoints from 5M steps, `agent.pkl` at the end (199,884,800
    steps), and `eval.json`: 20 timed laps at Hockenheim against the s-OCP lap.
- **One commit.** All five train from one commit, the annotated training tag `p4-b2-train-v1`
  that the lead places and pushes before launch (the box runs `git checkout --detach
  p4-b2-train-v1`). The round's manifest records HEAD and whether the tracked tree was clean, and
  `round_manifest.py check --tag p4-b2-train-v1` refuses a manifest whose HEAD is not the tag's
  commit or whose tree was not clean.
  - S1 trained (18 Oct) before the PPO refactor merged in `3aa57d3`. S2 and S3 finished on
    19 Oct, around that merge.
  - Since `3aa57d3`, these commits have touched `train.py`'s path (`agents/`, `env/`,
    `tracks/generator.py`), each adding an option that is off by default:
    - `83ebe56`, `2143a59` and `bdfd442`: copied and statistical width models (the generator;
      `adapters.py` in `83ebe56` and `bdfd442`);
    - `1bd85ec` and `3323a0e`: model (2b) and its batch size (generator, `adapters.py`);
    - `43d68fe`: VG-0's value-gradient term (`ppo.make_rollout`, `make_learner` and
      `make_train`, `adapters._bank_step`, `checkpoints.make_checkpoint_fn`).
  - The default path is held by `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update`,
    `tests/test_value_gradient.py` (`test_default_racing_update_reproduces_the_pre_term_values`,
    `test_default_rollout_records_no_capture_and_defaults_are_off`), and, for the generator and
    bank, `tests/test_stat_widths.py::test_default_bank_is_bit_identical` and
    `tests/test_real_widths.py::test_default_is_unchanged`. The lead re-lists the commits at the
    launch commit. Other lanes may add more such options, each with a default-path test.
  - S4–S8 are therefore a sample of their own. S1–S3 (Mac CPU, earlier or uncertain code) are
    compared with them descriptively only.
- **One machine, one GPU** (decided for the fourth round's boxes, 27 Oct 2026). P4-B2 is its own
  round, `p4b2`, on one single-GPU `massedcompute_A6000_plus` box (one NVIDIA RTX A6000 48 GB,
  $0.68/h), with all five runs on its GPU at once: `GPUS=1 LANES_PER_GPU=5`. It is set up with
  `scripts/gpu_setup.sh`. While the account allows one instance at a time, it is rented before or
  after the other rounds' `massedcompute_A6000_plusx4` box; if the account allows two (the user
  allowed more boxes on 27 Oct 2026), it may run beside that box, since it is a box of its own.
  It may also follow another round on its box (Section 8). Why this and not the x4 box:
  - It keeps the 5 lanes decided on 27 Oct 2026 (Section 11, #4): the five runs train at once on
    one GPU under the same load. `gpu_round.sh` gives each lane XLA memory fraction round(0.9 /
    5, 2) = 0.18, the drafted value.
  - On the x4 box, 5 lanes on one GPU leave 3 GPUs idle. The stop rule counts idle GPUs, and
    `round_manifest.py check` bills queue time x the GPUs `nvidia-smi` lists, so up to about
    0.45 h of queue would count as up to about 1.8 GPU-hours, above the 1.2 the stop rule allows
    (Section 8).
  - Dealt across the x4 box's GPUs at the round's 4 lanes per GPU (`gpu_round.sh`'s default),
    job j goes to GPU j mod 4: S4 and S8 would share GPU 0 and S5-S7 would each train alone.
    The five would no longer train on the same load.
  - Two rounds cannot share the x4 box at once: each queue runs on GPUs 0 to GPUS - 1, so a second
    queue would land on the same GPUs.
  - **Fallback**, only if no single-GPU box can be had: the x4 box after its other round's queue
    has finished, at the defaults (4 GPUs x 4 lanes, memory fraction 0.23). S4 and S8 then share
    GPU 0. This is a deviation from "the same load" and is recorded as one. It bills about 0.2 h
    x 4 GPUs, about 0.8 GPU-hours.
  - `scripts/gpu_sync.sh` pushes R-dev data only, including Hockenheim's centreline and its
    s-OCP lap, which `eval.json` needs. No R_TEST circuit leaves the Mac.
- **Queue.** The shared round queue: `GPUS=1 LANES_PER_GPU=5 BREV_TYPE=massedcompute_A6000_plus
  bash scripts/gpu_round.sh p4b2 scripts/p4b2_runs.py` at `p4-b2-train-v1`. The run list
  `scripts/p4b2_runs.py` gives the five runs in the order of the table above (S4 first). It is
  shown first with `DRY_RUN=1`, and the runs come back with `scripts/gpu_sync.sh <box> pull-round
  p4b2 scripts/p4b2_runs.py`.
  - The round manifest `runs/p4b2-manifest-0of1.json` records HEAD, the tree, the GPU model and
    count, the box type, the lanes, and each run's status, times and config sha256.
  - A run that fails is re-run once, from scratch and with the same arguments, in its own lane as
    soon as it fails. This is the shared queue's rule. It replaces the drafted rule of waiting
    for the other four, which a new queue would have needed. A memory failure can therefore
    recur while the other four still train. That is unlikely: each lane holds 0.18 x 48 GB,
    about 8.6 GB, for a 3 x 256 MLP on 2,048 environments. The first attempt's log is kept as
    `runs/<name>.log.1`. A run that fails twice is missing (Section 9).
  - The run names do not start with `p4b2-`, so `round_manifest.py check` notes that it cannot
    look for stray run directories of theirs. S1–S3's run directories share the names' prefix,
    so a prefix search would flag them wrongly.
- **The final checkpoint only.** P4 reads each run's `agent.pkl`. The 16 checkpoints are kept but
  not analysed: P6 for specialists is out of scope.
- **`eval.json`** is the standard evaluation, not a protocol stimulus. Its completion and gap are
  reported beside K4's gate (100% completion, gap ≤ 8%) and decide nothing.

## 3. Stimuli, measures and code

**Stimuli** (protocol 3.3; all Hockenheim):
- **Starts.** Each specialist starts from its own deterministic real-lap state at s_start = s_c −
  D_gap − 100 on lap 2. This is the rule of `interp.stimuli.real_state_requests`, applied through
  the frozen `agents.real_lap` and `agents.real_state`, and mapped into strip coordinates by the
  frozen tier D builder.
- **The 23 registered tier D conditions** on the 20 corners (swap A on its 9), built by the frozen
  builder at f6_scale 0.5. Their strips must equal step 1's (`outputs/psychophysics/stimuli/bank.npz`,
  sha256 `c124cb39…`) bit for bit.
- **The 11 certified conditions** on their certified corners (table below). They are built by the
  builders in `scripts/p4_exploratory.py`, imported read-only. Each strip's sha256 must equal the
  `strip_sha256_per_corner` in the frozen lists.
- **Tier G** null straights and **tier A** A2 cells (agent-independent starts): the tier G crash
  flag, and P4's class 4.

**The F6 set (fixed now):**

| Condition | Source | Corners (of the 9 braking corners) |
|---|---|---|
| `F6/wobble0.3`, `F6/wobble0.6`, `F6/knoise0.5e-3`, `F6/knoise1e-3` (0.15 m, 0.3 m, 0.25e-3 and 0.5e-3 m⁻¹ as built, R27) | registered, step 1 | all 9, as registered |
| `P4A/F6s/knoise0.5e-3/x1`, `…/x2` | `outputs/p4/certified.json` (`7ad8db2`) | all 9 |
| `P4A/F6s/knoise0.5e-3/x3` | same | 0, 1, 3, 4, 7, 8, 16, 17 |
| `P4A/F6s/knoise1e-3/x1` | same | 0, 1, 4, 7, 8, 14, 17 |
| `P4A/Kstr/1e-3`, `P4A/Kstr/8e-3` | same | all 9 |
| `P4A/Kstr/2e-3`, `P4A/Kstr/4e-3` | same | 0, 1, 3, 4, 7, 8, 16, 17 |
| `P4A/Wplus/0.5m` | same | 0, 1, 3, 4, 7, 8, 17 |
| `P4A/Wplus/1m` | same | 0, 1, 3, 4, 8, 16, 17 |
| `P4A/Kstr/16e-3` | `outputs/p4/extension/certified.json` (`dc0c3c2`) | all 9 |

- The two lists are committed (sha256 `f9ab567a…` and `5d0d9b31…`). They bind every strip's hash.
- K-str at 16e-3 was certified after option A's planned results were seen, so it is post hoc
  there. Certification depends only on the s-OCP and the generalists, not on any specialist, and
  S4–S8 do not exist yet. It enters B2 prospectively.
- No new condition is certified and no OCP is solved.

**Braking corners and v\*_min** come from step 1's stored s-OCP keys at S1's state
(`outputs/psychophysics/analysis/key_table.npz`, sha256 `938fb380…`, recorded again at the record
stage): corners 0, 1, 3, 4, 7, 8, 14, 16 and 17.

**Cells (fixed now).** Built at the 9 braking corners only, as in option A, since P4 reads no
other corner.
- **Registered conditions.** Each specialist's cell starts from its own lap-2 state, mapped into
  strip coordinates through the corner's excision by the frozen `stimuli._strip_state`. This is
  the tier D loop of the frozen `stimuli.build_real_design`, which names S1–S3 only and is
  therefore repeated in `scripts/p4_b2.py`. Condition (a) and the mirror share a pair; every
  other condition has its own.
- **W and P.** No OCP is solved, so W and P are those of step 1's S1-state cell of the same
  strip. W is equal across S1, S2 and S3 at all 449 of step 1's tier D strips. P is read only by
  a state at its run's own start, and every lap-2 state is on ruling R21's lap rule; the build
  refuses any other.
- **Certified conditions.** Each cell copies the specialist's condition (a) cell at that corner
  onto the certified strip, as option A did.
- **Key rows** (braking corners, v\*_min, swap partners) are step 1's S1-state rows of the same
  strip, the values `analysis._share_tier_d_socp` gives every runner.
- **Check.** Applied to S1–S3, the rule gives step 1's 621 registered cells and option A's 270
  certified cells field for field. P is compared for S1 only, since step 1 gave S2 and S3 their
  own (Section 12).
- **Tier G and tier A.** Step 1's own 27 tier G null cells and 78 tier A A2 cells, whose starts
  do not depend on the agent.

**Runs.** 4 deterministic phases per cell, through the frozen rollout code (`interp.rollout`).
Every chunk is padded to 2,048 runs, as in option A (Amendment 1, item 1).

**Measures** (all P4's; option A Amendment 1, items 2 and 3):
- **Corner completed:** at least 3 of its 4 phase runs complete, meaning no termination before
  s_c1 + 250 m.
- **C(cond):** the share of the condition's corners completed. For a registered condition these
  are the 9 braking corners; for a certified one, its certified corners.
- **Per-corner d_OOD:** the frozen `controls.d_ood` on the phase-0 deterministic run, the median
  over steps from the start to s_c + 50 m, against the specialist's own buffer.
- **D(cond):** the median of the per-corner d_OOD over the condition's corners.
- **S_spec, φ, m and MI** as the frozen classifier computes them. MI uses option A's generalist
  reference tables (step 1's nine generalist finals, checked bit for bit against step 1 in
  option A).

**d_OOD buffers.** Built with the frozen `controls.build_knn_buffer`: 50 stochastic laps of real
Hockenheim, 40,000 observations, each specialist's own obs_rms. The agent ids S4–S8 give stream
indices 135–139, distinct from S1–S3's 132–134 (a built-in check).

**Code.**
- `scripts/p4_b2.py` and `tests/test_p4_b2.py`, both new, written, reviewed and committed before
  the tag. Written, tested and committed in `40ebdf6`, with the run list `scripts/p4b2_runs.py`.
  The adversarial review (Section 8, step 3) was done on 27 Oct 2026 and its fixes committed in
  `b3ef979` (Section 12).
  - Pre-tag stages: `manifest`, `buffers`, `starts`, `build`, `reproduce`, `cm-dryrun` and
    `record`. They are refused once the tag exists, their S1–S3 check variants too.
  - Post-tag stages: `rollouts` and `classify`, behind the tag gate (Section 7).
  - Outputs go to `outputs/p4b2/`: `pre/` for the manifest, buffers and starts, `check/` for
    the S1–S3 checks and the dry run, `build/` for the S4–S8 design, `rollouts/`, and
    `classify/decisions.json`. The record is `scripts/p4b2_prereg.json`.
- It imports `src/apexzero/interp`, `scripts/psychophysics.py` and `scripts/p4_exploratory.py`
  read-only. Their versions are pinned in the record that the tag binds (Section 8, step 3):
  the git blob and sha256 of each script, the git tree of `src/apexzero/interp`, and the commit
  they were read at. As of this draft (at `c862a76`, and again at `40ebdf6`, where
  `scripts/p4_b2.py` pins them; none has changed since `f62f273` for `psychophysics.py` and
  `interp/`, or `2fc517b` for `p4_exploratory.py`):
  - `scripts/psychophysics.py`: blob `53d34e79…`, sha256 `40ac5e91…`;
  - `scripts/p4_exploratory.py`: blob `f29eec60…`, sha256 `95183edf…`;
  - `src/apexzero/interp`: tree `e4c456fe…`.
- Every stage refuses to run if any of them differs, in git or byte for byte, from the record.
- It adds no file under `src/` (the costate freeze hash covers `src/`), and writes nothing under
  `outputs/psychophysics/` or `outputs/p4/`. Its outputs go to `outputs/p4b2/`.

## 4. Classification (rules fixed now)

### 4.1 P4 as registered (the primary classification)

The protocol's ordered rules, first match wins, with the frozen thresholds (gate 0.9, fail 0.5,
pass 0.8, main effect 0.4) and the frozen `analysis.p4_classify` logic. The F6 set is the 15
conditions of Section 3, each with its C and D over its corners, as option A's classifier did.

| Order | Class | Rule (protocol P4) |
|---|---|---|
| 0 | Not competent | C(a) < 0.9 |
| −1 | Not testable | a failing F1 or an F6 condition has no d_OOD |
| 1 | Off-manifold fragile | some single-factor F1 condition has C ≤ 0.5, and some F6 condition with D at most that condition's D also has C ≤ 0.5 |
| 2 | Context-keyed memorisation | some single-factor F1 condition has C ≤ 0.5, and every F6 condition with D at least that condition's has C ≥ 0.8. The key is named by the F1 main effects of at least 0.4 |
| 3 | Corner-keyed memorisation | all 8 F1 conditions have C ≥ 0.8, and median φ ≥ 0.5, median m ≥ 0.5, S_spec < 0.4 or C(mirror) ≤ 0.5 |
| 4 | Reads the road, generally | class 5's criteria, tier A completion ≥ 0.8 at A2 for R_seg ≥ 10.7 m, and no tier G crash |
| 5 | Reads the road, locally | all F1 conditions and the mirror at C ≥ 0.8; C ≥ 0.8 at every dose; S_spec in [0.6, 1.4]; median φ < 0.25; median m < 0.25 |
| 6 | Mixed | none of the above |

- **Vacuity** (R10). A class 2 is **non-vacuous** when at least one F6 condition has D at or above
  the failing condition's (`n_f6_at_or_above` ≥ 1). Otherwise it is **vacuous**.
- **Per family.** The class under the registered four plus each family alone (F6s, K-str, W+) is
  reported, as in option A. It is descriptive.
- **The tier G rule** (protocol P4). A specialist that crashes on a null straight counts as
  evidence for class 1 on synthetic stimuli. This is reported, and it does not change the class.
- **The checkpoint gate** of protocol 8 (gentle cells, oscillation) does not apply to P4, as in
  step 1. P4's own gate is class 0.

### 4.2 The corner-matched reading (CM)

It is computed for every specialist with a single-factor F1 condition at C ≤ 0.5. It decides only
when Section 4.1 gives a vacuous class 2 (Section 4.3). Elsewhere it is reported descriptively.

**Inputs** for specialist X, at braking corner i and condition q: done(i, q) = 1 if the corner is
completed (at least 3 of 4 phases), and d(i, q), the phase-0 per-corner d_OOD.

**Cells and comparators:**
- **Failing conditions:** the single-factor F1 conditions with C ≤ 0.5 (P4's list).
- **Failing cell:** a pair (i, f) of a braking corner and a failing condition with done(i, f) = 0
  and a finite d(i, f).
- **Comparators at corner i:** the 11 certified conditions that are certified at corner i (the
  table in Section 3). The four registered F6 conditions are not comparators, because they have no
  frozen per-corner certification.
  - A comparator without a run or a finite d(i, g) is left out of the comparison and listed as
    unused. This fixes an implementation detail (27 Oct 2026, before any S4–S8 run). It does not
    arise when every run has observations before s_c + 50 m, as every tier D run does.

**Each failing cell is typed in this order** (the first match wins, as in P4):
1. **Fragile:** some comparator g with d(i, g) ≤ d(i, f) has done(i, g) = 0.
2. **Keyed:** at least one comparator has d(i, g) ≥ d(i, f), and every such comparator has
   done(i, g) = 1.
3. **Mixed:** some comparator with d(i, g) ≥ d(i, f) fails, and none at or below fails.
4. **Unmatched:** no comparator lies at or above d(i, f), and none at or below fails.

**The specialist's CM class.** Let n be the number of fragile, keyed and mixed cells.
- **Not testable** if n < 3.
- **CM-1, fragile,** if fragile cells are at least half of n.
- **CM-2, context-keyed,** if keyed cells are at least 0.8 of n.
- **CM-6, mixed,** otherwise.

The 0.5 and 0.8 mirror P4's fail and pass thresholds.

**Multiplicity, stated now.** Rule 1 needs one failing comparator out of up to 11 at a corner, so
more comparators make "fragile" easier and "keyed" harder, as in P4 itself (option A plan,
Section 6). No threshold is adjusted for it.

### 4.3 Each specialist's reading

| Section 4.1 result | Reading |
|---|---|
| Class 0 | not competent |
| Class −1 | undecided (P4: not testable). The buffers exist before the tag, so this is a deviation (Section 9) |
| Class 1 | fragile |
| Class 2, non-vacuous | context-keyed, with the key named |
| Class 2, vacuous | CM-1: fragile (CM). CM-2: context-keyed (CM). CM-6: mixed (CM). CM not testable: undecided |
| Class 3 | corner-keyed |
| Class 4 or 5 | reads the road (class 4: generally; class 5: locally) |
| Class 6 | mixed |
| Missing (failed twice) | missing: counts against every reading (Section 4.4) |

- **Class −1** (27 Oct 2026, a wording fix made while building, before any S4–S8 run). This
  table first read "not testable", while Section 9 gives the reading "undecided". The two now
  agree on "undecided", which is the reading Section 4.4 counts; the class is still reported as
  P4's "not testable".

### 4.4 The recipe

- **Reading.** A reading is claimed for the specialist recipe when at least 3 of the 5
  specialists hold it. This is the protocol's "at least 2 of 3" carried to 5 seeds as a majority.
  - A reading reached through CM counts with its kind: fragile with fragile (CM), context-keyed
    with context-keyed (CM), mixed with mixed (CM). The number reached through CM is reported.
  - "Undecided" counts on its own.
  - Classes 4 and 5 count together, as "reads the road". The wording (Section 5) is class 5's,
    "within its envelope", unless at least 3 of the 5 are class 4.
  - Context-keyed through class 2 and through CM-2 count together. The key in the wording is
    every key held by at least one specialist in the majority, each with its count, most held
    first (for example "keyed by constant width (3 of 4) and the straightened road (1 of 4)").
    A specialist whose class 2 names no key counts as "no key named".
    - The phrases (fixed while building, 27 Oct 2026): width is "constant width", after is "the
      straightened road after the corner", before is "the straightened road before the
      corner", and none is "no key named". A specialist holding two keys counts for each. The
      count's denominator is the number of specialists in the majority. Ties go in P4's factor
      order (before, after, width), with "no key named" last. Three or more keys are joined
      as "A (2 of 4), B (2 of 4) and C (1 of 4)".
  - With 5 specialists at most one reading can reach 3. A missing specialist counts against every
    reading, so the denominator stays 5.
- **P4's registered recipe claim.** The same rule on the Section 4.1 classes alone is reported
  beside it, with a vacuous class 2 flagged as such.

## 5. What decides, and the wording it fixes

The recipe reading decides the wording for the specialists in `protocol-v1` and the demo's
specialist beat. Whatever the reading, P4 is then settled, and no further P4 study is planned
before `protocol-v1` (#14).

| Recipe reading | Wording fixed now |
|---|---|
| Fragile | "The Hockenheim specialists are fragile off their training manifold. Certified-irrelevant road changes no further from it break them as the failing single-factor change does, so memorisation cannot be diagnosed." The demo does not call them memorisers |
| Context-keyed | "Context-keyed memorisation, keyed by [factor]. Certified-irrelevant changes at least as far off the manifold leave them intact, while one real-road factor breaks them." [factor] is filled by Section 4.4's key rule. Their gap from the optimum is a fact about Hockenheim only |
| Corner-keyed | "Corner-keyed memorisation" (protocol, interpretation table) |
| Reads the road | "Reads the road within its envelope", or "beyond it" ("Reads the road beyond its envelope") if at least 3 of the 5 are class 4 (Section 4.4) |
| Mixed | "No single label." The specialists are described per network and per corner |
| Undecided | "Tied to Hockenheim's context (the single-factor failures and MI), with memorisation and fragility not separable by certified-irrelevant road changes" |
| Not competent | "The recipe does not reliably give specialists competent from these states." P4 is not answerable |
| No reading reaches 3 of 5 | "No recipe-level description." Each specialist's reading is reported |

The protocol's interpretation table still applies to each specialist's own class.

## 6. Expected directions (stated before training)

From S1–S3 and option A. They are held weakly, and none of them decides anything.
- **EB1, competence.** All five complete Hockenheim in `eval.json` with a gap ≤ 8%, and all pass
  P4's gate C(a) ≥ 0.9. Basis: S1–S3 at 4.75–4.84%, C(a) = 1.0.
- **EB2, a failing F1 condition.** At least 3 of the 5 have a single-factor F1 condition at
  C ≤ 0.5, and it is constant width or the straightened road after the corner. Basis: S2 and S3
  have one each, S1 none (its lowest is 0.56).
- **EB3, P4 as registered.** Those specialists land in class 2, vacuous. Basis: certified D ≤ 1.53,
  against failing D of 2.01–2.75. Class 1 is possible through K-str at 16e-3, where S2's C was
  0.56.
- **EB4.** A specialist without a failing F1 condition lands in class 6, as S1 did.
- **EB5, CM.** CM-6, mixed. Basis: at corners 7 and 14, S2 and S3 complete some certified changes
  at or above their failing d_OOD and fail others (p4.md Section 4, descriptive and post hoc).
- **EB6, the recipe.** No reading reaches 3 of 5, or "undecided" does.
- **EB7.** Every specialist's MI is well above the generalists' (S1–S3: 0.70–1.04, against −0.04),
  and at least 2 of the 5 crash on a tier G null straight (S2 and S3 did).

## 7. Checks (all before the tag unless stated; a failure stops the item for an amendment)

| Check | Rule | Status (27 Oct 2026) |
|---|---|---|
| Manifest | For each run: `config.json`, `agent.pkl` sha256 and timesteps, the 16 checkpoints' sha256, `eval.json`, the training log, the queue log, the commit and the box. Committed | Built (`manifest` stage, with `round_manifest.py check` at `p4-b2-train-v1`, and `agent.pkl` and the checkpoints against the box's sha256 list of Section 8, step 2). It also records the load, all five at once on one GPU, or the deviation from it. Waits for the runs |
| Buffers and starts | The five buffers and the five real-lap state sets, with sha256, in the committed record | Built (`buffers`, `starts`); waits for the runs. On S1–S3 both reproduce step 1's bit for bit (Section 12) |
| Strips | The registered strips equal step 1's bit for bit. The certified strips match the frozen lists' per-corner sha256 | Passed: 449 of 449 and 90 of 90 |
| Pipeline on S1–S3 | (a) With the registered F6 set it reproduces `decisions_all.json`'s P4 rows (sha256 `14c67486…`): class, label, failing condition, `vacuous`, `n_f6_at_or_above` and C exactly, d_OOD to 1e-6. (b) With the registered four plus the 10 planned conditions it reproduces `outputs/p4/classes.json` exactly. (c) With K-str 16e-3 added it reproduces `outputs/p4/extension/classes.json` exactly. (d) States and termination steps equal option A's stored runs bit for bit | Passed, all four (Section 12) |
| CM on S1–S3 | Computed once as a dry run. It is reported as exploratory and post hoc for those networks. It does not change Section 4.2: any change after it is a dated amendment that states what was seen | Computed (Section 12). Nothing in Section 4.2 changed after it |
| Tests | Planted tables for every CM cell type and class (fragile, keyed, mixed, unmatched, not testable); the reading map and the recipe rule; the tag gate; the lockbox refusal; distinct buffer streams | `tests/test_p4_b2.py`: 76 tests, all passing (57 at `40ebdf6`, 19 added by the review) |
| Freeze | No rollout of S4–S8 on any protocol stimulus (tiers A–G) before the tag. Allowed before it: loading and one forward pass, `eval.json`, the deterministic real lap (protocol exception 1, R1) and the buffer laps (exception 2) | Enforced: only `rollouts` drives a stimulus, behind the tag gate |
| Tag gate (after the tag) | The post-tag stages refuse to run unless `p4-b2-prereg-v1` exists on origin and points at the recorded commit. `classify` runs once and refuses if its output exists | Built: as `scripts/lockbox.py`'s gate, with the record's commit as "the recorded commit" (Section 8, step 4). The bound files and the frozen tree are compared byte for byte, so a file marked assume-unchanged or skip-worktree cannot hide an edit. `classify` also refuses rollouts of another design or buffer |

## 8. Procedure, budget and box time

1. **This design is reviewed and committed,** with `scripts/p4b2_runs.py`, `scripts/p4_b2.py`
   and `tests/test_p4_b2.py` (code committed in `40ebdf6`, review fixes in `b3ef979`). The lead
   places and pushes the annotated training tag `p4-b2-train-v1` on that commit.
2. **The box.** The lead checks the A6000's hourly price on Brev before creating it. One
   single-GPU `massedcompute_A6000_plus` box (Section 2) is rented and set up (`ssh <box> 'bash
   -s' < scripts/gpu_setup.sh`, then `scripts/gpu_sync.sh <box>`), and checked out at
   `p4-b2-train-v1`. The queue's schedule is shown with `DRY_RUN=1` first, then it runs the five
   runs (Section 2, Queue). The runs are pulled back with `scripts/gpu_sync.sh <box> pull-round
   p4b2 scripts/p4b2_runs.py`. Before the box is deleted, the box's sha256 of each run's
   `agent.pkl` and checkpoints is written to a file on the Mac, from the repo root
   (`p4_b2.BOX_SHA256_CMD`):
   `ssh <box> 'cd ~/apexzero && sha256sum runs/hockenheim-specialist-mlp-200M-s[4-8]/agent.pkl
   runs/hockenheim-specialist-mlp-200M-s[4-8]/ckpt/*.pkl' > runs/p4b2-box-sha256-0of1.txt`.
   Then the box is deleted.
3. **On the Mac, before the tag** (at most 4 processes, `nice -n 15`; each line is `uv run
   --no-sync python scripts/p4_b2.py <stage>`):
   - `manifest`: `round_manifest.py check --tag p4-b2-train-v1` (the configs' sha256 against
     the box's manifest), each `agent.pkl`'s and checkpoint's sha256 against the box's list
     (step 2; the stage stops without it), then the manifest. Committed with `git add -f
     outputs/p4b2/pre/manifest.json`;
   - `buffers` and `starts`: the five buffers and start states, committed the same way;
   - `build`: the strip build and its checks, and the S4–S8 cells;
   - `build --owners check`, `starts --check`, `buffers --check`, `reproduce`, `cm-dryrun`: the
     S1–S3 checks and the CM dry run (done, Section 12);
   - `scripts/p4_b2.py` and its tests, written, given an adversarial review and committed
     (done: `40ebdf6`, review fixes `b3ef979`);
   - `record`: the record `scripts/p4b2_prereg.json` (every input hash), committed alone.
4. **The tag** `p4-b2-prereg-v1`, annotated, placed on the record's commit and pushed by the lead.
   The gate reads "points at the recorded commit" as that: the tag's commit is the one that last
   changed the record.
5. **After the tag, on the Mac:** `rollouts` (S4–S8 on tiers D, G and A), then `classify` once,
   then `outputs/p4b2/classify/decisions.json` committed.
6. **The findings,** written and verified as for steps 1 and 2.

**Box time (estimated from recorded timings):**

| Part | Basis | Estimate |
|---|---|---|
| Setup | H4R: rented and set up in 7 min (10:38–10:45, `h4r.md`) | 7–10 min |
| Training, 5 lanes | A6000, 200M-step DR runs: 145k–179k steps/s per run at 4 lanes (18.7–22.9 min each), 386k alone (8.6 min; `h4r.md` Section 4.2). With no gain beyond 4 lanes, 5 lanes give about 128k steps/s each, so 26 min. On the Mac the specialist env ran 2.6 times faster per step than DR (47–49k against 18.4k steps/s), so 26 min is the upper end | 10–26 min |
| `eval.json` and pull-back | 20 timed laps of one circuit; 5 small runs | about 5 min |
| A retry, if one is needed | the shared queue restarts it in its lane at once, beside the runs still training: up to one more run's time | +10–26 min |

- **Total: about 25–45 min, about 1 box-hour at most with one retry.**
- **Approved, 27 Oct 2026.** The user approved this box time, about 1 h, as part of Phase 3's
  23–29 A6000 box-hours (GW-3 about 14 h, VG-1 8–14 h, P4-B2 about 1 h), on the cheapest Massed
  Compute A6000, one box at a time, deleted when done.
- **Stop rule** (the lead's; counted in GPU-hours from the fourth round on, idle GPUs included).
  If the round's projected GPU time exceeds its approved estimate by more than 20%, that is, above
  about 1.2 GPU-hours (72 min on the one GPU), the lead pauses the round and asks the user.
  `round_manifest.py check` prints each manifest's queue time x the GPUs listed.
- **Price.** `massedcompute_A6000_plus` was $0.68/h on 27 Oct 2026 (follow-through plan, Section
  11, fourth round): under $1 for the round. The lead checks it again before creating the box.
- B2 is its own round, on its own commit. It may share a rental with another round only if it runs
  from a separate checkout of its own commit, and not while the other round's queue runs (the
  fallback of Section 2).
- **Mac time.** Estimated in the draft; measured on 27 Oct 2026 for the stages that ran on S1–S3
  (3 workers at nice 15, with another workflow on the Mac), on the first run and on the run whose
  outputs are committed (their provenance gives the second):
  - buffers: 46 s and 26 s for S1–S3's three (draft: under a minute for five);
  - real laps and starts: 10 s and 4 s for three;
  - the strip build and cells: 10 s and 3 s;
  - the S1–S3 reproduction: about 5 min and 2.5 min of wall time (each network's 1,188 tier D
    runs in about 3 min and 1.7 min, and its 420 tier G and A runs in about 1 min and 0.6 min,
    the three in parallel; the draft said about 20 min);
  - the post-tag rollouts: about 10 min expected, five networks on at most 4 workers;
  - `classify`: seconds.

  Each job is well under 1 CPU-hour.

## 9. Missing runs, failures and deviations

- **A run that fails twice** is missing. It is listed with its queue events, and the recipe
  denominator stays 5.
- **A specialist whose deterministic real lap ends** before the lap-2 start of any of the 9 braking
  corners is class 0, "not competent from these states". Its stimuli are not built.
- **Class −1** should not occur. If it does, it is a deviation, and the specialist's reading is
  "undecided".
- **A pipeline crash after the tag** voids `p4-b2-prereg-v1`, as for H4R. The fix is re-registered
  as `p4-b2-prereg-v2`, with its own tag, before any decision is computed.
- **Anything else** that departs from this document is a dated deviation in the findings. It says
  what had been seen when it was written.

## 10. Integrity and lockbox

- **Frozen code is unchanged:** `src/apexzero/interp`, `scripts/psychophysics.py`, and the option A
  script and lists, checked at every stage against the hashes the record pins (Section 3, Code).
- **Only Hockenheim is built or driven.** Every track name passes `viz.export.lockbox_guard`, and
  the frozen builder admits Hockenheim only.
- **No R_TEST circuit** and no mirrored or reversed variant of one is used. `outputs/sealed/` is
  not read.
- **The box** receives R-dev data only.

## 11. Choices for review (decided 27 Oct 2026)

On 27 Oct 2026 the user accepted the draft's default for all five. Each is fixed as below. Where
the draft named an alternative, it is kept as history and was not taken.

1. **The CM reading** (decided 27 Oct 2026). It is included and decides only after a vacuous
   class 2 (Sections 4.2 and 4.3). Not taken: reporting it descriptively only, in which case B2
   would most likely have ended "undecided" by Section 4.1 alone.
2. **The recipe threshold** (decided 27 Oct 2026). 3 of 5 specialists to claim a reading
   (Section 4.4). Not taken: 4 of 5, closer to 2 of 3 as a proportion.
3. **Registered F6 conditions** (decided 27 Oct 2026) stay out of the CM comparators, because they
   have no frozen per-corner certification (Section 4.2).
4. **The seeds, run names and lanes** (decided 27 Oct 2026): seeds 4–8, the run names of
   Section 2, and 5 lanes. Not taken: 4 lanes plus 1.
   - The 5 lanes stay on one GPU of a single-GPU box when more boxes were allowed (fourth round,
     27 Oct 2026; Section 2 gives the reasons and the fallback).
5. **No new certified family** (decided 27 Oct 2026). B2 uses option A's families as decided
   (Section 3). A family that reaches far off the manifold at most corners would have needed its
   own certification before the tag: s-OCP solves and generalist runs, about 1 h on the Mac, plus a
   design.

## 12. Build status and the pre-tag artifacts (27 Oct 2026)

Everything here ran on the Mac from the committed code (`40ebdf6`; each output's provenance
gives HEAD and the script's sha256), at nice 15 with at most 3 workers. The JSON files under
`outputs/p4b2/check/` are committed (`git add -f`). Their buffers, rollouts and cells are local,
like option A's.

**Built and checked on S1–S3** (Section 7):

| Item | Result | File (sha256) |
|---|---|---|
| Strips | 449 of 449 registered strips equal step 1's bit for bit; 90 of 90 certified strips match the frozen lists' per-corner sha256. 539 rows; bank hash `02fe2594…`, which the S4–S8 build must equal (the build checks it) | `check/build/build.json` (`651b548f…`; design hash `f81c0294…`) |
| Cells | Under the rule of Section 3 (Cells), S1–S3's 621 registered cells equal step 1's and their 270 certified cells equal option A's, field for field (P for S1 only). Every cell is on R21's lap rule | the same |
| Starts | S1–S3's real laps (states, actions, accelerations, validity, termination) and their 20 tier D states each equal step 1's bit for bit, through the frozen `load_real_states` | `check/starts/starts.json` (`210bdad8…`) |
| Buffers | S1–S3's buffers rebuilt by the frozen `build_knn_buffer` (streams 132–134) equal step 1's arrays bit for bit | `check/buffers/buffers.json` (`c8b3e58b…`) |
| Pipeline (a) | `decisions_all.json`'s P4 rows for S1–S3: class, label, failing condition, `vacuous`, `n_f6_at_or_above` and C equal; d_OOD difference 0 | `check/reproduce.json` (`14b0f335…`) |
| Pipeline (b), (c) | `outputs/p4/classes.json` and `outputs/p4/extension/classes.json`: every set's classes, the recipe and the registered recipe equal | the same |
| Pipeline (d) | 1,188 of 1,188 tier D runs per specialist (828 registered, 324 planned, 36 extension) equal option A's stored runs bit for bit; the d_OOD of all 297 cells per specialist equal option A's (difference 0) | the same |
| Tiers G and A (extra) | 420 of 420 tier G and tier A A2 runs per specialist equal step 1's bit for bit. The 66 pairs' columns that P4 reads equal step 1's pair tables, and so do the tier G flag (S1 none; S2 and S3 crash) and the class 4 statistic (tier A A2 completion: S1 0.63, S2 0.02, S3 0.38) | the same |

**The CM dry run on S1–S3** (`check/cm_dryrun.json`, `0ce3fc27…`). This is a DRY RUN, not a
result: exploratory and post hoc for S1–S3, whose option A results had been seen. It decides
nothing. Section 4.2 is unchanged by it.

| Specialist | Section 4.1 on the 15 conditions | CM cells: fragile, keyed, mixed, unmatched | CM | Reading |
|---|---|---|---|---|
| S1 | 6, mixed (no single-factor F1 condition fails) | – | not computed | mixed |
| S2 | 2, vacuous (road after straightened, constant width) | 4, 2, 4, 4 (n = 10) | CM-6 | mixed (CM) |
| S3 | 2, vacuous (constant width) | 2, 0, 1, 2 (n = 3) | CM-1 | fragile (CM) |

- **S2's cells.** The fragile ones are at corners 3, 4 and 17, where W+ or K-str 16e-3 fails at
  or below the failing d. The keyed ones are at corner 8. The mixed ones are at corners 4, 7 and
  14: certified changes above the failing d fail there (W+ and K-str 16e-3 just above it at
  corner 4, K-str 8e-3 or 16e-3 far above it at corners 7 and 14), and none below it fails.
- **S3's cells.** Fragile at corners 16 and 17 (K-str 8e-3 and 16e-3 fail below the failing d);
  mixed at corner 7 (K-str 4e-3 to 16e-3 fail above 2.75, nothing below fails).
- **S3 rests on n = 3, the minimum.** One fewer typed cell would make it "not testable", and its
  reading "undecided".
- **EB5** expects CM-6 for S4–S8. The dry run gives CM-6 for S2 and CM-1 for S3. That is
  reported and changes nothing.

**Waiting for the training round** (Section 8, steps 2–4). The sha256 of each artifact goes here
and into the record when it is built:

| Artifact | Path | sha256 |
|---|---|---|
| Manifest | `outputs/p4b2/pre/manifest.json` | pending |
| Buffers, S4–S8 (streams 135–139) | `outputs/p4b2/pre/buffers/knn_S{4..8}.npz` and `buffers.json` | pending |
| Start states | `outputs/p4b2/pre/starts/` (`real_states.json`, `real_states_index.json`, `real_laps/S{4..8}_Hockenheim.npz`, `starts.json`) | pending |
| S4–S8 design | `outputs/p4b2/build/` (design hash) | pending |
| Record | `scripts/p4b2_prereg.json` | pending |

- **Size.** Each buffer is about 36 MB (40,000 observations of 222 float32 channels, with their
  actions), so committing all five as Section 1 says adds about 182 MB to the repository. Each
  real lap is about 0.4 MB.
- **Still to do before the tag:** the training round (with the box's sha256 list of Section 8,
  step 2); `manifest`, `buffers`, `starts`, `build` and `record`; the tag.

**The adversarial review (27 Oct 2026).** An independent review of `40ebdf6` and `05f4e22`. Its
fixes are in `b3ef979`; nothing in Sections 4 and 5 changed, and no S4–S8 run exists.
- **Rules.** Section 4 was read against the code: the ordered classes on the 15 conditions, the
  vacuity, the CM cell types, ties and class boundaries, the reading table, the 3-of-5 recipe with
  its pooling, missing runs and key wording, and Section 5's wording. The CM dry run was
  recomputed from the raw check rollouts by separate code (done(i, q) from the 4 phases, d(i, q)
  from the phase-0 runs, C and D over the braking corners): S2 class 2 vacuous, 4, 2, 4, 4,
  CM-6; S3 class 2 vacuous, 2, 0, 1, 2, CM-1; S1 class 6; the same cells and types as
  `check/cm_dryrun.json`.
- **Fixed.** The tag gate used `git diff`, which trusts the index for a file marked
  assume-unchanged or skip-worktree: an edited bound file passed it. The gate now compares every
  bound file's bytes with the tag's blob, and so do the frozen tree pin and the record stage. The
  draft's check of the final checkpoints against the box's sha256 had no code (the box manifest
  records only the configs' sha256): the box's list is now written before the box is deleted
  (Section 8, step 2) and the manifest stage compares every `agent.pkl` and checkpoint with it.
  The manifest also records the load, so a schedule other than all five at once on one GPU is a
  recorded deviation (with the lanes left at `gpu_round.sh`'s default of 4 on one GPU, S8 would
  train after S4). `buffers --check` and `starts --check` now refuse after the tag, and
  `classify` refuses rollouts of another design or buffer.
- **The box.** `DRY_RUN=1` at `GPUS=1 LANES_PER_GPU=5` puts S4–S8 in lanes 0–4 on GPU 0, each at
  position 0. At `GPUS=4 LANES_PER_GPU=4` S4 and S8 share GPU 0. With `SHARD=i/5` each box gets
  one seed, in order. S4's command, run on the Mac with `--steps 262144` in a scratch directory
  (deleted after), trained, wrote `config.json`, a checkpoint, `agent.pkl` and `eval.json`, and
  its config passed `round_manifest.py`'s comparison. On a planted p4b2 round in train.py's
  config format (made by code that reproduces that run's `config.json` byte for byte),
  `round_manifest.py check` passed, and it failed on each of 16 planted faults (seed, steps,
  tracks, envs, learning rate, network, no PPO block, a config not the box's, a run in no
  manifest, a failed run, a missing run directory, HEAD, a dirty tree, the GPU model, a changed
  run list, a run in two manifests).
