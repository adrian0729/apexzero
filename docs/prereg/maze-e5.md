# Maze E5 (PA-3): do level scores track exact regret on JaxUED's own agents? (pre-registration, DRAFT)

**Status.** DRAFT, 26 Oct 2026, for review; updated 27 Oct 2026 for the build. Nothing here is
tagged, and none of the 30 runs below exists. After review the lead commits the final text. On
27 Oct 2026 the user decided three of Section 11's choices (E5's own tag, no PLR-PVL arm, DR
through `maze_plr.py`), approved the SFL download (Section 10), and approved only a short timing
run on the box (Section 9). The same day E5 was built (Section 8, step 3) and SFL's approved files
were vendored, and the user approved SFL's network file, `sfl/train/minigrid_plr.py`, which was
then vendored (Section 11, item 8). That file imports one more SFL file,
`sfl/util/jaxued/jaxued_utils.py`; the user approved it (with the two empty package files) later
the same day and it is vendored, so SFL's source is complete and the network adapter is tested on
SFL's real network (Section 11, item 8). The user also decided that a score faithful in every
reached phase is row A even when it shifts, and that B and B′ are not reported for such a score
when row G blocks A (Section 5.2 and Section 11, item 11). The runs are trained once the user approves the full
queue, their checkpoint manifest is committed, and the tag `maze-e5-prereg-v1` is placed before
any audit of an E5 checkpoint (Section 8). E5 uses no racing circuit, so the lockbox is untouched.
The build had its adversarial review the same day; its fixes are in Sections 2, 3, 5.2, 8, 9,
10 and 11 (item 11).

## 1. Why

- **Its place in the paper.** E5 is the paper's external check, contribution 4 in
  [two-track-plan.md](../plan/two-track-plan.md) Section 2.1: the level scores of curriculum
  methods (PVL, MaxMC, learnability) audited against an exact optimum, on a standard benchmark,
  with the methods' own code.
- **The maze has an exact optimum.** J\* comes from a breadth-first search (`apexzero.maze.oracle`),
  so regret is known exactly, with no OCP labels and no label noise in J\*.
- **The pilot** ([maze-audit-pilot.md](../findings/maze-audit-pilot.md)): one seed, 200 of
  30,000 updates, the 5×5 view, no SFL.
  - At 12–34% solve rates, PVL, MaxMC and L1 ranked mazes the wrong way round. ρ with regret was
    −0.26 to +0.02, and −0.53 to −0.31 with normalised regret. All six headline scores were below
    the best baseline at every checkpoint.
  - It sized E5: 4 target episodes estimate regret with reliability 0.21–0.45, and 32 are
    projected at 0.68–0.87.
  - It left the question open: "whether the sign changes once the agent solves most mazes".
- **The specification** comes from the follow-through ([follow-through.md](../plan/follow-through.md),
  PA-3): JaxUED maze at its defaults, 5 seeds × {DR, PLR-MaxMC, SFL} × 2 view sizes, 30,000
  updates each, 16 log-spaced checkpoints, BFS regret on 2,000 mazes, about 32 target episodes.

## 2. Runs (fixed now)

| Arm | Code | Flags beyond the shared ones |
|---|---|---|
| DR | JaxUED `examples/maze_plr.py`, vendored unmodified (`third_party/jaxued_examples/`) | `--replay_prob 0 --exploratory_grad_updates`, as in the pilot |
| PLR-MaxMC | the same file | none: `--score_function MaxMC` is its default (PLR⊥: replay 0.8, rank β 0.3, staleness 0.3, buffer 4,000) |
| SFL | SFL's `sfl/train/minigrid_sfl.py`, vendored unmodified on 27 Oct 2026 (`third_party/sfl/`, Section 10) | its own config defaults (`minigrid-sfl.yaml`, `learning/ippo-minigrid.yaml`, `env/minigrid.yaml`), except DEV-E5-2 to DEV-E5-4 |

- **No PLR-PVL arm** (decided by the user, 27 Oct 2026; #17). The arms are these three.

**Shared settings:**
- **View sizes:** 5 (the default) and 13. For JaxUED: `--agent_view_size 5|13`. For SFL: the
  override `env.AGENT_VIEW_SIZE=5|13`.
- **Seeds:** 1–5 in every arm × view cell, so 30 runs. The pilot used seed 0 and was seen.
- **Length:** 30,000 PPO updates for every arm. That is JaxUED's default `--num_updates`. SFL's
  default is 4,577 updates (`TOTAL_TIMESTEPS` 3e8 // 256 // 256), so its runs set
  `learning.TOTAL_TIMESTEPS=1966080000` (30,000 × 256 × 256; DEV-E5-3). Its learning rate then
  anneals over the 30,000 updates, by its own schedule.
- **Calling the code:** every run is `scripts/maze_e5.py train --arm A --view V --seed S --updates
  30000 --name N` with N `maze-e5-A-vV-sS` and A `dr`, `plr` or `sfl` (the run list's command
  and names, e.g. `maze-e5-sfl-v13-s1`), which runs the arm's trainer in `runs/<name>/` with
  exactly the flags below (`maze_e5.trainer_cmd`) and records the run (`runs/<name>/e5-run.json`):
  - JaxUED runs go through `scripts/run_jaxued_maze.py`, with its compatibility shim and wandb
    offline: `--run_name N --seed S --agent_view_size V --num_updates 30000 --eval_freq 100
    --checkpoint_save_interval 1 --max_number_of_checkpoints 300`, plus `--replay_prob 0
    --exploratory_grad_updates` for DR;
  - SFL runs go through a shim of the same kind, `scripts/run_sfl_maze.py`: the overrides
    `SEED=S env.AGENT_VIEW_SIZE=V learning.TOTAL_TIMESTEPS=1966080000
    learning.NUM_CHECKPOINTS=300` and `--save-updates` with the 16 checkpoints (DEV-E5-2).
- **Hyperparameters:** each arm at its own defaults, so the arms differ in more than the sampler.
  - JaxUED: 32 envs × 256 steps per update, γ 0.995, λ 0.98, lr 1e-4.
  - SFL's MiniGrid code, read from the vendored config on 27 Oct 2026 (it matches
    `docs/paper/sources-sfl.md` Section 2.3):
    - 256 envs × 256 steps per update, γ 0.99, λ 0.95, lr 2.5e-4 annealed, 4 epochs × 4
      minibatches, clip 0.04, no entropy bonus;
    - every 100 updates, 25,000 fresh mazes (5 batches of 5,000) are scored by learnability from
      2,000-step rollouts, and the top 1,000 form the buffer;
    - half of each batch (128 of 256 levels) is drawn from that buffer, the rest is fresh;
    - its generator places 60 walls (`N_WALLS`); JaxUED's places 25.

  Comparisons between arms are therefore descriptive (Section 5.4).

**Checkpoints.** 16, log-spaced from 1% of the run to its end, at update counts
u_k = 100·round(3·100^((k−1)/15)), k = 1…16:

> 300, 400, 600, 800, 1000, 1400, 1900, 2600, 3500, 4800, 6500, 8800, 11900, 16200, 22100, 30000

- **JaxUED runs** save at every eval block: `--eval_freq 100 --checkpoint_save_interval 1
  --max_number_of_checkpoints 300`. Orbax step e then holds update 100(e + 1). When the run
  exits, `maze_e5.py train` deletes every step but the 16 selected ones, so only those are pulled
  back (DEV-E5-1), and records each kept step's sha256 over its files.
- **SFL runs** save the parameters and the learnable set D at the same 16 update counts
  (DEV-E5-2), in `runs/<name>/sfl-checkpoints/u<update>/`.
  - D is the set in force at the checkpoint: the 1,000 levels the last 100-update block trained
    on, with the learnability computed at the start of that block (score age 100 updates). It is
    captured, not recomputed, and its scores must equal the `learnability_set_scores` the script
    logs for that block, or the run stops.

**Machine and queue** (updated 27 Oct 2026 for the fourth round; `docs/plan/follow-through.md`
Section 11).
- **One commit, the round's boxes.** All 30 runs train from one commit, tagged `maze-e5-train-v1`
  by the lead before the launch (the training tag; `maze-e5-prereg-v1` comes after training), on
  NVIDIA RTX A6000s set up with `scripts/gpu_setup.sh`. No racing data is needed on the boxes.
  - The plan: one `massedcompute_A6000_plusx4` box (4 GPUs, $2.74/h), each run on one GPU. If the
    x4 type is not available, single-GPU `massedcompute_A6000_plus` boxes ($0.68/h), each taking
    whole seeds (`SHARD=i/K`). All boxes of the round are the same GPU model, and each round's
    manifest records the box type, GPU count and shard.
  - The lead checks the hourly price on Brev before creating a box, and deletes each box when its
    work is done.
- **The queue** is the shared one, `scripts/gpu_round.sh maze-e5 scripts/maze_e5_runs.py --phase
  main` (it replaces the `gpu_queue_maze_e5.sh` this draft planned). The run list
  (`scripts/maze_e5_runs.py`) fixes the order: the 10 SFL runs first, then PLR-MaxMC, then DR,
  within an arm seed by seed and view 13 before 5. The expected longest runs start first, so with
  10 or more lanes (or one seed per single-GPU box) no lane holds two SFL runs. With fewer lanes
  some lane holds two or three (at 4 lanes on one GPU, lane 0 runs three).
  - Lane l runs on GPU l mod G, and the list alternates the views, so on the x4 box every view-13
    run is on GPU 0 or 2 and every view-5 run on GPU 1 or 3. At 4 lanes per GPU, GPUs 0 and 1 each
    start three SFL runs at once (lanes 0, 4 and 8; 1, 5 and 9). No calibration GPU at that load
    carries more than two (Section 9), so SFL's measured rate may be optimistic for those GPUs;
    the stop rule (Section 9) is the guard.
  - Job j goes to lane j mod L (L = GPUs × `LANES_PER_GPU`), each lane in order; the lanes per GPU
    are fixed by the timing calibration (Section 9).
  - A failed run is re-run once from scratch (`gpu_round.sh`). A run that fails twice is missing
    (Section 8).
  - Each round's runs are pulled back with `scripts/gpu_sync.sh <box> pull-round` and checked with
    `scripts/round_manifest.py check` (one manifest per box).
- **Integrity.** The vendored files are checked against their git blob hashes before every stage
  (`apexzero.maze.sfl.check_vendored`):
  - `maze_plr.py`, blob `a7b6164d…`, which equals JaxUED's at the pinned commit (checked);
  - SFL's files (Section 10).

**Deviations from the reference code's defaults (fixed now):**

| ID | Item | Default | E5 | Reason |
|---|---|---|---|---|
| DEV-E5-1 | JaxUED checkpoint cadence | eval every 250 updates, save every second eval, keep 60 | eval every 100, save every eval, keep 300; 16 selected | log-spaced checkpoints without editing the file. `maze_plr.py` splits its key at every eval block, so the runs are statistically equivalent to default runs, not bit copies (the pilot's precedent) |
| DEV-E5-2 | SFL checkpoints | the parameters only, one file overwritten every `NUM_UPDATES / EVAL_FREQ / NUM_CHECKPOINTS` blocks (`NUM_CHECKPOINTS` 4); D is never saved or returned from the jitted step | no patch (read 27 Oct 2026). `learning.NUM_CHECKPOINTS=300` makes the script call `save_params` after every block; `run_sfl_maze.py` replaces `save_params` with E5's writer, which keeps the 16 updates. D is captured by a view of `jax` that adds an `io_callback` to `get_learnability_set`'s outputs. A test shows the parameters after a short run are bit-identical with and without the capture (CPU, with SFL's own network); the timing run repeats the check on the GPU (Section 9) | the checkpoints and the in-buffer audit |
| DEV-E5-3 | SFL run length | 3e8 env steps, 4,577 updates (read 27 Oct 2026) | 30,000 PPO updates (`learning.TOTAL_TIMESTEPS=1966080000`) | the follow-through's equal-update rule |
| DEV-E5-4 | View 13 | 5 | 13 | the observability manipulation (outcome row F) |
| DEV-E5-5 | Episodes per score | JaxUED: whatever completes in 256 steps; SFL: up to 10 in 2,000 steps | 4 operational episodes (score spec DEV-5). The native scores are reported beside them | score noise that does not depend on the agent's speed, as in the pilot |
| DEV-E5-6 | DR | JaxUED also ships `examples/maze_dr.py` | `maze_plr.py` with replay 0 and exploratory gradient updates (the pilot's) | DR and PLR then share one script and differ only in the sampler. Decided by the user on 27 Oct 2026; `maze_dr.py` is not downloaded |
| DEV-E5-7 | SFL's environment (engineering, found 27 Oct 2026) | Hydra, OmegaConf, `sfl.train.train_utils`, the JAX and matplotlib of 2024 | `run_sfl_maze.py` stands in without editing the files: it composes the config itself (Hydra's rules for these files, OmegaConf's YAML floats; neither is installed), replaces `save_params` (DEV-E5-2), casts SFL's 0/1 integer `where=` masks to booleans (today's JAX refuses them; older JAX used their truth value), restores `FigureCanvasAgg.tostring_rgb` for the buffer figure and closes each figure once logged, and makes `wandb.Artifact` a no-op. wandb stays in SFL's own mode (`disabled`); the shim logs each block's eval to `sfl-eval.jsonl` | nothing the script computes changes (tested) |

## 3. The audit (fixed now)

This is the pilot's protocol (`apexzero.maze.audit`) at E5's size.

**Mazes.**
- 2,000 from JaxUED's default generator, `make_level_generator(13, 13, 25)`, drawn with level
  seed 2027. The pilot used 2026, with 500 mazes.
- The same 2,000 serve every run and checkpoint.
- SFL's default generator differs from JaxUED's (`env/minigrid.yaml`: 60 walls, read 27 Oct
  2026), so SFL's runs are also audited on 2,000 mazes from `make_level_generator(13, 13, 60)`,
  level seed 2027, as a secondary set (descriptive, Section 5.4).

**The optimum.** J\* by BFS with the run's own γ (0.995 for JaxUED's runs, 0.99 for SFL's) and
time limit (250 steps, the Maze default in both). J\* > 0 exactly when the goal is reachable
within the limit, so the J\* > 0 mazes are the same for every run.

**Settings.** Each run's scores use its own γ, λ and PPO rollout length (256 steps in both
codes), read from its recorded config (`apexzero.maze.e5_audit.settings`). The audit key of run i
(its index in the run list) at update u is `fold_in(fold_in(PRNGKey(20271), i), u)`; the native
streams use `fold_in(key, 1)` (JaxUED's) and `fold_in(key, 2)` (SFL's), the in-buffer streams
`fold_in(key, 3)` and SFL's own-generator set `fold_in(key, 4)`.

**Streams.**
- One AutoReplay stream per maze, with the stochastic policy. The LSTM resets on done. It runs
  until 36 episodes are complete.
- Episodes 1–4 give the operational scores.
- Episodes 5–36 give the target: Ĵ^π is their mean discounted return (32 target episodes).
- Score and target therefore share no sampling noise.

**Native scores** (independent streams):
- JaxUED's own new-level score: one 256-step rollout from a reset with a fresh carry, giving
  native PVL and MaxMC, as in the pilot.
- SFL's own learnability: one 2,000-step AutoReplay stream, counting at most 10 completed episodes
  and dropping the truncated last one, as SFL's MiniGrid code counts them (`sources-sfl.md`
  Section 1.3).

**Scores.**
- **Six headline scores:** PVL, MaxMC, L1 value loss, MaxMC-unit, learnability p(1 − p) (success =
  goal reached) and MNA.
  - All use the run's own γ and λ. PVL and MaxMC are checked against JaxUED's own functions.
- **The natives,** as above.
- **λ variants** 0.9, 0.95, 1.0 and the run's own, for PVL and L1.

**Baselines.**
- agent-only −Ĵ^π on the operational episodes;
- J\* and BFS path length, both policy-agnostic;
- random, on a fixed key;
- oracle-operational, a reference and not a ceiling.

**Targets** (both primary):
- absolute regret J\* − Ĵ^π, on all 2,000 mazes;
- normalised regret (J\* − Ĵ^π)/J\*, on the mazes with J\* > 0.

**Phase** of a checkpoint: p, the share of its target episodes that reach the goal, over the
mazes with J\* > 0. Early is p < 0.5, mid is 0.5 ≤ p < 0.9, late is p ≥ 0.9 (the plan's bands).

**In-buffer audit** (PLR-MaxMC and SFL runs, every checkpoint):
- **PLR.** The valid entries of the sampler state stored in the checkpoint. If there are more than
  1,000, a fixed-key uniform 1,000 are used. Each comes with its stored score and its staleness
  as JaxUED counts it: the sampler's episode counter minus the entry's timestamp. The counter
  advances once per level scored, about 32 per update, so PLR's age is not in updates (corrected
  27 Oct 2026 in review: a 100-update PLR run on the Mac had ages up to 3,199).
- **SFL.** The learnable set D in force at the checkpoint: K = 1,000 levels with the learnability
  of the last refresh, and the score age in updates since that refresh (100: every checkpoint
  ends a block).
- **Current regret** on those levels comes from BFS and the same 36-episode streams.
- The PLR subsample uses a fixed key per run and checkpoint (`AUDIT_SEED`, the run's index, the
  update).

**Built-in checks** (each stops the item for an amendment):
- PVL and MaxMC equal JaxUED's functions to 1e-6. The pilot got 2.8e-7.
- No episode beats J\*: regret ≥ 0 per episode, not only on average.
- BFS matches executed optimal plans (the existing test).
- SFL's native learnability equals a line-for-line port of SFL's MiniGrid outcome counting on test
  vectors, and its top-K follows the score spec's S-1 to S-3.
- **Pilot reproduction.** Run on the pilot's checkpoints with the pilot's settings (500 mazes,
  level seed 2026, 8 episodes), the E5 audit reproduces the pilot's `fidelity.json` exactly. The
  default path of `scripts/maze_audit.py` stays unchanged.

## 4. Estimators and statistics

**Per checkpoint.** For run r (arm, view, seed), checkpoint k, score j and target t:
- **ρ:** the Spearman correlation between score and target over the mazes, with average ranks for
  ties. z = atanh(ρ), with ρ clipped to ±0.999. A score that is constant over the mazes (or over
  a bootstrap resample of them) carries no ranking information, and its ρ is 0 (fixed 27 Oct
  2026; the draft was silent). A level JaxUED scores −∞ (no completed episode) ranks last.
- **RC@10%:** the expected share of total regret in the score's top 10% (200 mazes), with ties at
  the cut-off sharing the remaining slots (`apexzero.maze.fidelity`).
- **nRC = (RC − 0.1)/(RC\* − 0.1):** the share of the achievable excess capture, where RC\* is
  regret's own capture.
  - It is undefined when RC\* − 0.1 < 0.02. The pilot showed RC\* of only 0.11–0.12 at low
    competence, so raw RC must be read against its ceiling.

**Per arm and phase** (the unit of every decision):
- Each run's mean z over its checkpoints in the phase. Then the mean over the arm's runs that
  have one: Z. Then ρ̄ = tanh(Z).
- Runs weigh equally, and so do the two view sizes (5 runs each). When the views hold different
  numbers of such runs (a missing run, or a phase some runs do not reach), Z is the mean of the
  two views' means, runs equal within a view (fixed 27 Oct 2026).
- A phase is **reached** by an arm when at least 3 of its runs have a checkpoint in it.
- nRC is averaged in the same way, without the transform, over the checkpoints where it is
  defined.

**Δ against the best baseline.** ρ̄(score) − ρ̄(best baseline). In each bootstrap draw, the best
baseline is the largest of agent-only, J\* and BFS steps (the pilot's rule).

**Two-way bootstrap.**
- B = 4,000 draws. Each resamples the mazes (2,000 with replacement, one draw shared by every run
  and checkpoint; the normalised target uses the J\* > 0 mazes within the draw), and the runs
  (with replacement within each arm × view cell).
- Phase membership is fixed from the full sample.
- The in-buffer levels differ per run and checkpoint, so each in-buffer set is resampled on its
  own in every draw (with replacement, its own fixed stream).
- In a draw, every ρ is the Spearman ρ of the resampled sample: average ranks under the draw's
  multiplicities, which equal the ranks of the expanded sample (`apexzero.maze.e5.RankEngine`,
  tested against it). A draw in which a statistic is undefined (no resampled run has a
  checkpoint in the phase) is left out of that statistic, and B counts the draws that remain.
- CIs are 95% percentile intervals.
- A two-sided p is min(1, 2(1 + n_opposite)/(B + 1)), where n_opposite counts the draws on the
  other side of 0 from the point estimate, draws at exactly 0 included; p is 1 when the point
  estimate is 0.
- A one-sided p against a bound b is (1 + n_beyond)/(B + 1), where n_beyond counts the draws at
  or beyond b on the null's side. For "difference > 0" the null side is ≤ 0, so n_beyond counts
  the draws ≤ 0.
- **TOST** for |x| < c: p_low = (1 + #draws with x ≤ −c)/(B + 1) and p_high = (1 + #draws with
  x ≥ c)/(B + 1). Its p is the larger of the two.

**The trend test** (the plan's seed-slope t-test):
- Per run, the OLS slope of z on log10(u_k) over all 16 checkpoints.
- Per arm, a two-sided one-sample t-test of its 10 slopes against 0 (df 9).

**Labels**, per arm, score, target and reached phase (the plan's provisional thresholds, fixed
here for the maze):
- **Faithful:** ρ̄ ≥ 0.5, nRC ≥ 0.6, and Δ > 0 with its CI excluding 0.
- **Uninformative:** ρ̄ ≤ 0.2, or nRC ≤ 0.2, or Δ's CI includes 0 or lies below it.
- **Intermediate:** otherwise.
- Where nRC is undefined, its conditions are dropped and the label is flagged.

## 5. Hypotheses, tests and decisions

### 5.1 Primary family (confirmatory; Holm at α = 0.05 over 36 tests)

The primary scores are the three the plan's outcome rows are read on: PVL, MaxMC and learnability.
For each arm (3), primary score (3) and target (2):
- **PF-trend:** the trend test. 18 tests.
- **PF-late:** Δ in the latest reached phase (late if reached), two-sided. 18 tests.

The late-phase labels of the primary scores use PF-late's Holm decision for their Δ condition.
Every other label uses the unadjusted CI.

In both families a test that cannot be run (no reached phase for an arm, a missing arm, an S3 cell
with fewer than 3 runs) keeps its place with p = 1, so Holm always runs over 36 and 28 tests.

### 5.2 Outcome rows (the maze's entry in the plan's outcome table, Section 2.5)

Per arm and primary score, read on both targets:

| Row | Rule | Reading |
|---|---|---|
| A, faithful | Faithful in every reached phase on both targets, and no row G for the arm's stored score | "On JaxUED's own [arm] agents, [score] tracks exact regret throughout training" |
| B, degrades late | On one target, both: ρ̄_last − ρ̄_first ≤ −0.1 with its CI excluding 0, and PF-trend negative and significant after Holm. It holds if they hold together on either target (the target or targets are reported) | "The score tracks regret only while the agent is incompetent" |
| B′, improves with competence (not in the plan's table) | The mirror image, on one target: ≥ +0.1 with its CI excluding 0, and PF-trend positive and significant after Holm | "The score tracks regret only once the agent solves most mazes" |
| C, uninformative | Uninformative in every reached phase on both targets | "Ground truth confirms No Regrets on the maze" |
| F, observability | S3 (Section 5.3) significant after Holm, with \|difference\| ≥ 0.2 | "Fidelity depends on observability" |
| G, fresh ≠ buffer | S2 significant after Holm, with \|difference\| ≥ 0.2 | "Fresh audits misstate what the curriculum actually sees" |
| A withheld by G (not in the plan's table) | Faithful in every reached phase on both targets, and a row G for the arm's stored score, which blocks A (the literal rule of row A) | "Faithful in every reached phase; A withheld by row G" |
| none of these | | reported per phase as mixed |

- "Last" (and "late" elsewhere) means the latest reached phase; "first" means the earliest reached
  phase, which is early when early is reached. ρ̄_last − ρ̄_first is computed on the arm's runs and
  bootstrapped as Section 4 fixes. If only one phase is reached, B and B′ are not testable.
- Rows can co-occur, except A with B, B′ or C. A and C cannot both hold. When A's rule and B's
  or B′'s hold together (faithful in every reached phase, with a significant shift), the build
  reports A and not B or B′ (the user decided 27 Oct 2026; Section 11, item 11). The shift is
  still tabulated. B and B′ are therefore reported only for a score that is not faithful in some
  reached phase. If a row G blocks A for a score faithful in every phase, B and B′ are still not
  reported. The reason (the user's reason for A, decided 27 Oct 2026): such a score is still
  faithful in every reached phase, so B's reading ("only while the agent is incompetent") or B′'s
  ("only once the agent solves most mazes") would overstate it.
- Row A's rule is literal: it needs no row G for the arm's stored score, and that holds for every
  score of the arm, not only the stored one (the stored scores are PLR's MaxMC and SFL's
  learnability; G is an arm-level result). A score faithful in every reached phase whose A is
  withheld only for that reason is not "mixed" ("none of the rows holds" is false for it): the
  build reports it as "A withheld by G", with the reading above. Row G itself is reported for the
  arm's stored score only, as before, so the stored score of such an arm shows both labels when
  it is faithful throughout. "Mixed" is left for a score that holds none of the rows and is not
  faithful in every reached phase.
- Rows D and E of the plan need track types and curricula on a shared pool, and do not apply.

### 5.3 Secondary family (pre-registered; Holm at α = 0.05 over 28 tests)

- **S1, the plan's MaxMC prediction** ("high ρ against absolute regret but near zero against
  normalised"). Per arm, in the latest reached phase, two tests:
  - ρ̄_abs − ρ̄_norm > 0 (one-sided; p against 0, Section 4);
  - |ρ̄_norm| < 0.2, by TOST against ±0.2 (Section 4): the larger of the two one-sided p values.

  The prediction holds for an arm when both pass. 6 tests.
- **S2, in-buffer against fresh** (the plan's prediction that in-buffer fidelity is lower).
  - PLR runs: stored MaxMC against fresh native MaxMC. SFL runs: stored learnability against fresh
    native learnability. Both targets.
  - The run-level mean z over all 16 checkpoints, two-sided.
  - 4 tests.
- **S3, view size.** Per arm, primary score and target: ρ̄(13) − ρ̄(5) on the run-level mean z over
  all 16 checkpoints, two-sided. 18 tests.

### 5.4 Descriptive (no test)

- per-checkpoint ρ and nRC curves per arm × view;
- labels for L1, MaxMC-unit, MNA and the natives;
- λ sensitivity;
- the score against the solve rate, and ρ on solved mazes only;
- reliability: an odd/even split of the 32 target episodes, Spearman–Brown to 32, and
  reliability-corrected ρ;
- the two-loading regression rank(S) ~ a·rank(J\*) + b·rank(Ĵ^π);
- partial ρ, P@10% and contamination;
- admission AUROC for top-10% regret;
- lagged fidelity (the score at checkpoint k − 1 against regret at k);
- in-buffer ρ by score-age tercile;
- arm-to-arm differences;
- SFL on its own generator, if it differs.

## 6. Expected directions (stated before training; from the pilot, held weakly)

- **X1.** In the early phase, PVL, MaxMC, L1 and MaxMC-unit are uninformative and below the best
  baseline on both targets, for DR and PLR-MaxMC. Basis: the pilot, every Holm-adjusted p < 0.05,
  below.
- **X2.** PVL and MaxMC rise with training on absolute regret (row B′). This is held weakly. In the
  pilot the scores track the solve rate (ρ +0.39 to +0.90), which tracks regret negatively while
  most mazes are failed. Once most are solved, regret is about speed.
- **X3.** Learnability is uninformative in every phase. Basis: the pilot (−0.18 to +0.03 absolute,
  −0.47 to −0.09 normalised); p(1 − p) peaks at p = 0.5, while regret peaks at p = 0.
- **X4.** The plan's MaxMC prediction (S1) holds. This is held weakly: its premise failed in the
  pilot's early phase.
- **X5.** In-buffer fidelity is below fresh fidelity (the plan's prediction).
- **X6.** No direction for view size, nor for differences between arms.
- **X7.** Agent-only −Ĵ^π and BFS steps are the strongest baselines early (the pilot).

## 7. What E5 decides, and what it does not

- **It decides** the maze's entry in the paper's outcome table (Section 5.2), and what
  contribution 4 can claim.
- **It changes no racing decision.** E0's recipe, `generator-v1`, the lockbox and the E4 trigger
  (racing only) are untouched.
- **`audit-v1`** may cite E5 as prior information. It is not bound by it.
- **Its tag is its own.** E5 has its own tag, `maze-e5-prereg-v1`, is not folded into
  `protocol-v1` and does not wait for it. The user decided this on 27 Oct 2026 (#17; Section 11).

## 8. Procedure

**Before the tag:**
1. **Approvals.** The user approved the SFL download on 27 Oct 2026 (Section 10). The vendored
   code needs no install: Hydra and OmegaConf are avoided (DEV-E5-7). The user approved SFL's
   `sfl/train/minigrid_plr.py` (38,737 bytes, blob `9537a794…`, same commit and licence), which
   holds SFL's network, on 27 Oct 2026; it is vendored. Read, it imports one more SFL file,
   `sfl/util/jaxued/jaxued_utils.py` (`l1_value_loss`); the user approved that too on 27 Oct 2026
   and it is vendored with its two empty package files (Section 10).
2. **Vendoring.** Done on 27 Oct 2026 (Section 10).
   - SFL's approved files that `minigrid_sfl.py` needs, `LICENSE` included, were downloaded at the
     pinned commit into a new directory outside the repo, each checked against its git blob SHA-1
     in Section 10's table (all matched), and copied byte for byte into `third_party/sfl/`.
   - `third_party/sfl/README.md` records the source, commit, licence, files, hashes, the files
     not vendored, and DEV-E5-2, as `third_party/jaxued_examples/README.md` does.
3. **The build.** Done on 27 Oct 2026; it still needs its adversarial review.
   - `scripts/run_sfl_maze.py`, the SFL shim (DEV-E5-2, DEV-E5-7);
   - no DEV-E5-2 patch: the vendored files are unmodified;
   - `apexzero.maze.sfl`: the vendored blobs, the config, the checkpoint writer and loader, the
     port of SFL's outcome counting and top-K, and the policy adapter (which needs
     `minigrid_plr.py`);
   - `apexzero.maze.e5_audit`: the E5 audit. It calls the pilot's `apexzero.maze.audit` unchanged,
     so `scripts/maze_audit.py`'s default path is untouched (no option was added to it);
   - `apexzero.maze.e5`: Sections 4 and 5 (the two-way bootstrap, the p formulas, Holm, the
     labels, the rows) and Section 9's projection rule;
   - `scripts/maze_e5.py`, with the stages `train`, `timing`, `manifest`, `audit` and `analyse`,
     and the tag gate (`scripts/lockbox.py`'s rule for `maze-e5-prereg-v1`: both refs named in
     full on origin, the same tag object locally, HEAD descended from it, and E5's code, the
     vendored files, the shared code the audit runs, this text and the tests byte for byte the
     tag's, so a file marked assume-unchanged or skip-worktree cannot pass);
   - `scripts/maze_e5_runs.py`, the run lists for `scripts/gpu_round.sh` (it replaces the
     `gpu_queue_maze_e5.sh` planned here): `main`, `timing`, `timing-audit`, `capture-check` and
     `audit`, each its own round;
   - `tests/test_maze_e5.py`. It covers planted scores with a known ρ, the bootstrap (against
     Spearman on the expanded sample, and with 1 or more processes), Holm, the phase assignment,
     the slopes, nRC, the labels and rows (each planted row comes out), S1, the SFL loader's round
     trip, the capture's bit-identity, the shim end to end (with JaxUED's network standing in for
     SFL's), the run lists, the config checks, the train stage, the tag gate (with both index
     flags), run-once outputs, the audit job of a missing run, the run-list options the boxes
     recorded, the audit's equality with the pilot's and the projection rule (38 tests after the
     review, 40 after the decisions of 27 Oct 2026).
4. **A dry run** on the pilot's checkpoints: the pilot reproduction (Section 3), then `analyse`
   end to end on one seed. This is a machinery check only. Not yet run (the test shows the E5
   audit with its additions off equals the pilot's `audit_checkpoint` on a small case).
5. **Timing calibration** on the box (Section 9). It fixes the lanes and the projection. It is the
   only box time approved so far (27 Oct 2026), and its measured projection goes to the user.
6. **Training,** once the user has approved the full queue on the measured estimate (Section 9).
   The lead tags the training commit `maze-e5-train-v1` and launches round `maze-e5`. The 30 runs;
   the 16 selected checkpoints per run are pulled back (`gpu_sync.sh pull-round`) with their
   sha256 checked; the training logs and wandb offline files are kept.
   - Training computes no score-against-regret outcome. The logs hold JaxUED's own eval solve
     rates and the samplers' statistics only.
7. **The manifest,** `scripts/maze_e5.py manifest --train-tag maze-e5-train-v1`: round
   `maze-e5`'s `round_manifest.py check` (a run that failed twice is accepted as missing; the
   check uses the run-list options the boxes recorded when they give the same list, since
   `--phase main` and `--phase main --drop 0` differ word for word), then
   each run's recorded trainer command against Section 2's, its recorded config against that
   command (maze_plr.py's parser rebuilt from the vendored source; SFL's config recomposed), and
   every checkpoint's files against their sha256. It writes `outputs/maze-e5/manifest.json`
   (every checkpoint's path, update and sha256; every run's config and queue events), which is
   committed (with `git add -f`: `outputs/` is ignored).
8. **The tag** `maze-e5-prereg-v1`, placed by the lead.

**Between training and the tag.** The 30 runs train from one commit (the training commit, which
holds this text and step 3's build). Their logs show JaxUED's eval solve rates, and so which phases
the runs reach, before the tag. From the start of training to the tag, any change to this text, E5's
code (step 3's files and `apexzero.maze`) or the audit settings is an amendment appended at the end
of this file, dated, stating what had been seen when it was written, including any training-log
solve rates. At tagging the lead checks `git diff <training commit> HEAD` on those paths: anything
other than the manifest and the recorded amendments is logged as a deviation.

**After the tag:**

9. **The audit,** at the tagged commit: round `maze-e5a` (`gpu_round.sh maze-e5a
   scripts/maze_e5_runs.py --phase audit`), one job per run of the list, each checking the
   committed manifest against the tag, and the run's config file and every checkpoint against
   their sha256, before loading anything. The job of a run the manifest lists as missing records
   that (`audit-run.json`, no scores) and succeeds, so the round's check, which `analyse`
   requires, can pass. It runs on the same
   boxes, kept up while steps 7 and 8 take their few minutes, or on a second rental (the
   checkpoints then go back to it from the Mac; `gpu_sync.sh` does not push them yet).
10. **`analyse`** on the Mac, once (`scripts/maze_e5.py analyse`): the tag gate, round `maze-e5a`'s
    `round_manifest.py check` at the tag, every audit file against its recorded sha256, then a lock
    that is never removed. It refuses to run if the lock or its output exists. B is 4,000; no
    option changes it. It writes
    `outputs/maze-e5/analyse/analysis.json` and `decisions.md`. Then the decisions are committed.
11. **The findings.**

**Missing and failed runs.**
- A run that fails twice is missing. Its arm's statistics use the runs that remain.
- An arm × view cell with fewer than 3 runs drops out of S3 for that arm.
- A pipeline crash after the tag voids it. The fix is re-registered as `maze-e5-prereg-v2` before
  any decision.

## 9. Budget: GPU time from what the repo records

**What is recorded:**

| Source | What it says |
|---|---|
| Two-track plan, Section 8 | E5 3–6 GPU-h (on a 4090; unmeasured). The follow-through carries it to the A6000 as "3–6 (unmeasured)" |
| Pilot timing (`runs/maze-audit/pilot-v2/timing.json`) | Mac CPU, single-thread XLA, nice 10, on a shared, overloaded machine. DR: 642 s wall and 1,015 CPU-s for 200 updates. PLR-MaxMC: 387 s and 726 CPU-s. Audit: 4.0 CPU-min for 8 checkpoints × 500 mazes × 8 episodes plus the native rollouts. Checkpoints are 4.8 MB each |
| A6000, racing PPO (`h4r.md` Section 4.2) | 386k steps/s alone and 145k–179k at 4 lanes, with 2,048 envs and an MLP. This does not transfer to a 32-env LSTM maze agent |
| SFL's MiniGrid code (`sources-sfl.md`; the vendored config, 27 Oct 2026) | 256 levels × 256 steps per update; 25,000 levels × 2,000 steps scored every 100 updates |
| The shim on the Mac (27 Oct 2026, `tests/test_maze_e5.py`) | SFL's loop at a toy size (4 envs, 16 steps, 32 levels scored) with JaxUED's network standing in: about 10–25 s per run of 6 updates, almost all compilation. It says nothing about the A6000 |

**What follows from it:**
- **Steps.** A JaxUED run is 30,000 × 32 × 256 = 246M env steps. An SFL run at its defaults for
  30,000 updates is 1.97B training steps plus 300 refreshes × 50M = 15B scoring steps (inference
  only).
- **Not a Mac job.** At the pilot's rate, a JaxUED run is 30–42 CPU-h, so the 20 JaxUED runs alone
  are about 730 CPU-h.
- **The fresh audit** is bounded by the slowest maze: unsolvable ones time out, so 36 × 250 =
  9,000 steps. That is 18 times the pilot's per-checkpoint work, about 9 CPU-min per checkpoint.
  - Over 480 checkpoints, plus the in-buffer audit on 320 of them, that comes to about 100 CPU-h
    on the Mac.
  - The follow-through's "about 38 h" is low for 36 episodes per maze. The audit runs on the box
    instead.
- **No maze GPU timing exists in the repo.**

**Estimate** (26 Oct 2026, for one A6000; the assumptions are labelled; the calibration
replaces them). Each run uses one GPU, so a box-hour of this table is a GPU-hour; on the x4 box
the same work spread over 4 GPUs costs about as many GPU-hours, plus the GPUs left idle while the
last runs finish (counted by the stop rule).

| Part | Assumption | GPU-h |
|---|---|---|
| 20 JaxUED runs | an update takes 0.1–0.3 s on the A6000 (10–30 times faster than the pilot's 3.2 s wall on the Mac), so 0.8–2.5 h per run; 8 lanes with little slowdown, so 3 rounds | 2.5–7.5 |
| 10 SFL runs | an update costs 1.5–2 times JaxUED's; a refresh is 10,000 sequential steps at a batch of 5,000 mazes, at 0.3–1 ms each, so 3–10 s; 1.5–5.8 h per run; 5 lanes, so 2 rounds | 3–11.6 |
| Audit, 480 checkpoints | 9,000 sequential steps at a batch of 2,000 (1,000 for in-buffer), at 1–3 ms each; 4 lanes | 0.5–1.5 |
| Setup, calibration, pull-backs | H4R's setup took 7 min; 30 × 16 × 4.8 MB is about 2.3 GB to pull back | about 1 |
| **Total** | | **about 7–22 (about 12 at the geometric midpoint)** |

- That is 2–4 times the plan's figure, and SFL dominates it. At $0.68 per GPU-hour (Brev, 27 Oct
  2026, the same per GPU on the x1, x2 and x4 types) it costs about $5–15. The lead checks the
  hourly price on Brev before creating a box.

**Timing calibration (fixed now; decided on time alone; specified 27 Oct 2026 for the boxes).**
Each step is a round of `scripts/gpu_round.sh` with its own run list (`scripts/maze_e5_runs.py`),
on the box the main queue will use (G GPUs; G = 4 on the x4 box):
- **Training, at 4 and at 8 lanes per GPU** (rounds `maze-e5t4` and `maze-e5t8`, `--phase timing
  --lanes-per-gpu L --gpus G`, run with `LANES_PER_GPU=L`): G × L runs of seed 99, 500 updates each
  (SFL through 5 refreshes), the six arm × view cells in the main list's order, repeated, so each
  GPU carries L runs as the main queue would. Below six runs, whole waves are added until every
  cell runs (8 runs on one GPU at 4 lanes). 4 lanes is what the recorded rates assume; 8 lanes
  puts the whole queue in one wave on the x4 box. A load at which a run fails (out of memory, say)
  is out.
- **The audit** (round `maze-e5at`, `--phase timing-audit`): one timing-only audit (it writes no
  scores) of update 400 of the first calibration run of each cell, with everything the real audit
  does: 2,000 fresh mazes, 36 episodes, both native streams, the in-buffer audit, and SFL's
  own-generator set.
- **The capture check** (round `maze-e5c`, `--phase capture-check`): three 200-update SFL runs at
  view 5, seed 99, two with D captured and one without. If the two captured runs are
  bit-identical (the GPU is deterministic here), the run without must be too (DEV-E5-2); if they
  differ, the check is inconclusive and reported as such.
- `--arms dr,plr` leaves SFL out of the first two; SFL's projection is then missing and the full
  queue cannot be projected.
- The calibration runs are discarded and never audited.
- **The projection rule** (`scripts/maze_e5.py timing`, `apexzero.maze.e5.project_queue`):
  - Per calibration run, from the end of each 100-update block (the commit time of JaxUED's orbax
    step, SFL's logged block): the per-update time is the median block after the first, over 100;
    the start-up is the first block's end minus one such block. Per cell and load, the median over
    its runs (the mean of the middle two for an even count). The run list is the one the box
    recorded, so `--arms` carries over.
  - A main run then takes start-up + 30,000 × per-update.
  - The training wall time is the longest lane when job j of the main list goes to lane j mod L,
    each lane in order (`gpu_round.sh`'s schedule); its GPU-hours are that wall time times the
    box's GPUs, idle GPUs included. For single-GPU boxes, each box takes one seed's 6 runs and
    the GPU-hours are summed over the boxes.
  - The audit adds, per run, 16 checkpoints × its cell's measured audit time, counted as if the
    audits ran one at a time per GPU (an upper bound).
  - Setup and pull-back add 0.25 wall hours per box (H4R's setup took 7 min), times its GPUs.
  - The total is the sum. The lane count is the one with the shortest projected total, for each
    box layout (the x4 box and single-GPU boxes are reported separately); the user
    decides the full queue on that total, and the stop rule's threshold is 1.2 times the total the
    user approves.

**Budget rule** (decided by the user, 27 Oct 2026). It replaces this draft's gate, under which a
projection of at most 12 box-hours would have started the queue under the standing permission and
a larger one would have gone to the user.
- **Approved now:** only the timing calibration, a short run on the cheapest Massed Compute A6000
  type (on the fourth round's boxes: the x4 box, or a single-GPU box). This draft gives it no
  separate estimate; the table above puts setup, calibration and pull-backs together at about 1
  GPU-hour on one GPU. As specified above, on the x4 box it is about 30–60 min of wall time, so
  about 2–4 GPU-hours (unmeasured); on one single-GPU box (G = 1) it is 8 runs at each load, about
  1 GPU-hour.
- **The full queue** (the 30 runs and the audit) returns to the user with the measured
  projection, whatever it is. It starts only on the user's OK.
- **Stop rule** (the lead's, 27 Oct 2026). If a round's projected time exceeds its approved
  estimate by more than 20%, in GPU-hours with idle GPUs included (fourth round), the lead pauses
  that round and asks the user. For the full queue, the approved estimate is the one the user
  approves on the measured projection. The timing run has no separate estimate (above).
  `round_manifest.py check` prints each box's queue wall time × GPUs.
- If the user caps the budget, cells are dropped in this fixed order (`maze_e5_runs.py --drop k`,
  k = 1–4, used by every later stage):
  1. SFL at the 13×13 view;
  2. the 13×13 view in every arm (S3 and row F are then not tested);
  3. seed 5, then seed 4, of every remaining cell, so that at least 3 seeds remain.

**Mac.** `analyse` only: bootstrap and tests over stored per-maze arrays, niced, with at most 4
processes. Measured on 27 Oct 2026 on planted arrays of E5's size (2,000 mazes, 16 checkpoints, 12
scores, 1,000 in-buffer levels): 35 ms per draw per run, so about 1.2 CPU-hours for 30 runs ×
4,000 draws, about 20 min of wall time with 4 processes (the 26 Oct estimate was under 1
CPU-hour). BFS on 2,000 mazes takes seconds.

## 10. The SFL source (vendored 27 Oct 2026)

**Approval (27 Oct 2026).** The user approved the download of the files in the table below, at the
commit below, under Apache-2.0: 46,359 bytes for the five core files, or 60,890 bytes with
`sfl/train/common/network.py` and the package files.
- The approval covers the download only. Any install (Hydra, OmegaConf or anything else) needs a
  separate OK from the user.

**Done (27 Oct 2026).** The five core files and the two package files `minigrid_sfl.py` imports
(`sfl/__init__.py`, `sfl/train/__init__.py`) were downloaded from raw.githubusercontent.com at the
commit into a new directory outside the repo, with the commit's tree listing from the GitHub API
(for `sfl/__init__.py`'s full blob, `b3c06d488393abb3b3829e5590d42409c995b4cf`). Every file's
`git hash-object` matched the table. They were read as text only, never run from that directory,
and copied byte for byte into `third_party/sfl/` with a README. `network.py` and
`sfl/train/common/__init__.py` were not downloaded: `minigrid_sfl.py` does not import them.

**The source:**
- **Repository:** `amacrutherford/sampling-for-learnability` on GitHub (Rutherford et al. 2024,
  "No Regrets", arXiv 2408.15099).
- **Commit:** `8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b` (21 Oct 2025, "Fix typo in Sampling For
  Learnability description").
  - It is the commit `docs/paper/score-spec.md` and `sources-sfl.md` pinned, and still the head of
    `main` on 26 Oct 2026 (GitHub API, metadata only).
- **Licence:** Apache-2.0 (GitHub's licence metadata; `LICENSE` is 11,357 bytes). That is the same
  licence as JaxUED.

**Files** (sizes and git blob SHA-1 from the GitHub tree API, each matched by the download):

| Path | Bytes | Blob | Vendored |
|---|---|---|---|
| `sfl/train/minigrid_sfl.py` | 33,866 | `6436f5f659fa5fd0992c3ce479054eacb0353fdd` | yes |
| `sfl/train/config/minigrid-sfl.yaml` | 466 | `41a016967485abcc8001b24ec91ddca783e54e6e` | yes |
| `sfl/train/config/learning/ippo-minigrid.yaml` | 423 | `633e1793c235cce1f3b046075d7f14a1d68ac67d` | yes |
| `sfl/train/config/env/minigrid.yaml` | 247 | `8df39e2c6881321897748e2450b51d43f3ad5b5c` | yes |
| `LICENSE` | 11,357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | yes |
| `sfl/train/minigrid_plr.py` | 38,737 | `9537a794b7df09c76bf816215b366b03eb10b682` | yes (27 Oct 2026, approved) |
| `sfl/util/jaxued/jaxued_utils.py` | 2,942 | `f491626f2164e2b36b06c8e4977b672588c985ab` | yes (27 Oct 2026, approved) |
| `sfl/util/__init__.py`, `sfl/util/jaxued/__init__.py` | 0, 0 | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (both) | yes (27 Oct 2026, approved; imported as packages) |
| `sfl/__init__.py`, `sfl/train/__init__.py` | 21, 0 | `b3c06d488393abb3b3829e5590d42409c995b4cf`, `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | yes (imported as packages) |
| If imported: `sfl/train/common/network.py`, `sfl/train/common/__init__.py` | 14,510, 0 | `b3a951dd184455f51627acc5881add9830849578`, `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | no (not imported) |

- The whole repository (757 entries, about 124 MB of files, mostly data) is not needed.
- `minigrid_sfl.py` was last changed in the repository's first commit (`626971979b0f`, 2 Sep
  2024). `sources-sfl.md` dates the config fix to `09946071de` (6 Mar 2025).

**What `minigrid_sfl.py` imports** (read 27 Oct 2026), and how each is met:

| Import | Here |
|---|---|
| `chex`, `jax` (with `jax.experimental`), `flax`, `optax`, `numpy`, `wandb`, `matplotlib`, `PIL`, `jaxued` (`Maze`, `MazeRenderer`, `AutoReplayWrapper`, `Level`, `make_level_generator`) | installed. The script calls `jax.tree_map` (restored by `apexzero.compat`), passes integer `where=` masks that today's JAX refuses, and calls `FigureCanvasAgg.tostring_rgb`, which matplotlib 3.10 removed (both stood in, DEV-E5-7) |
| `hydra` (`@hydra.main` on `main`), `omegaconf` (`OmegaConf.to_container`) | not installed; not needed: the shim composes the config and calls `main` itself (DEV-E5-7). No install is requested |
| `sfl.train.train_utils.save_params` | not vendored (`sfl/train/train_utils/__init__.py`, 44 bytes, blob `b5437dc2…`, and `common.py`, 6,308 bytes, blob `be4f1f26…`); not needed: the shim's writer replaces it (DEV-E5-2) |
| `sfl.train.minigrid_plr.ActorCritic, evaluate_rnn` | **vendored on 27 Oct 2026** (the user's approval, Section 11 item 8): `sfl/train/minigrid_plr.py`, 38,737 bytes, blob `9537a794b7df09c76bf816215b366b03eb10b682` (size and blob checked after the download, before the copy), the same commit and licence. It is SFL's network and its eval rollout; it differs from JaxUED's `maze_plr.py` (41,764 bytes, another blob), so JaxUED's network cannot stand in for it in the real runs. Its imports (read 27 Oct 2026): `json`, `time`, `os`, `enum`, `typing`, `numpy`, `jax`, `flax`, `optax`, `distrax`, `orbax.checkpoint`, `wandb`, `chex` and `jaxued` (all installed), `hydra` and `omegaconf` (stood in as above), `sfl.train.train_utils.save_params` (stood in as above), and `sfl.util.jaxued.jaxued_utils.l1_value_loss` (next row) |
| `sfl.util.jaxued.jaxued_utils.l1_value_loss` (by `minigrid_plr.py`) | **vendored on 27 Oct 2026** (the user's approval, Section 11 item 8): `sfl/util/jaxued/jaxued_utils.py`, 2,942 bytes, blob `f491626f2164e2b36b06c8e4977b672588c985ab`, and the empty package files `sfl/util/__init__.py` and `sfl/util/jaxued/__init__.py` (0 bytes, blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`), the same commit and licence; sizes and blobs checked after the download, before the copy. Read: it imports only `jax` and `jax.numpy` (it calls `jax.tree_map`, restored by `apexzero.compat`), both installed |

The repo's environment has `jaxued` 0.0.1, whose maze env equals the pinned JaxUED commit's
(blob `34291f1a…`, checked), and wandb, orbax, distrax and chex. The shim's end-to-end test runs
the vendored script with SFL's own network (`minigrid_plr`'s), and the network adapter is tested on
that network (Section 11, item 8). No stand-in network remains in the tests.

**Not downloaded:** JaxUED's `examples/maze_dr.py` (28,606 bytes, blob `d367205a…`, the same
pinned commit `0f8f128`, Apache-2.0). It would have been needed only if review preferred it to
DEV-E5-6, which the user kept on 27 Oct 2026.

## 11. Choices for review (items 1, 4, 8 and 11, and item 7's gate, decided 27 Oct 2026)

1. **E5's own tag now** (this draft), or E5 inside `protocol-v1`. Also whether to add a PLR-PVL arm
   (10 more runs). Both are #17, the user's. **Decided 27 Oct 2026:** E5 has its own tag,
   `maze-e5-prereg-v1`, not folded into `protocol-v1`, and there is no PLR-PVL arm.
2. **Equal updates (30,000) for SFL.** The alternative is equal env steps or SFL's own default
   length.
3. **Arms pooled over the two views** in the primary family, with view as a secondary moderator.
   The alternative is per arm and view: 72 primary tests.
4. **DR through `maze_plr.py`** (DEV-E5-6), or JaxUED's `maze_dr.py` (one more small download).
   **Decided 27 Oct 2026:** DR runs through `maze_plr.py` with replay off, as drafted. `maze_dr.py`
   is not downloaded.
5. **Seeds 1–5.** The alternative is 0–4, which overlaps the pilot's seen seed 0.
6. **The thresholds** (ρ 0.5 / 0.2, nRC 0.6 / 0.2). They are the plan's provisional values, with
   RC read against its ceiling, and fixed here for the maze.
7. **The budget gate** of 12 box-hours, and the drop order. **The gate was replaced on 27 Oct
   2026:** only the timing run is approved, and the user decides the full queue whatever its
   estimate (Section 9). The drop order is still for review.
8. **SFL's network file** (new, 27 Oct 2026; the user's, #2). **Decided 27 Oct 2026:** the user
   approved the download of `sfl/train/minigrid_plr.py` (38,737 bytes, blob `9537a794…`, same
   commit, Apache-2.0). It was downloaded into a new directory outside the repo, its size and git
   blob checked (both matched), and copied unmodified into `third_party/sfl/` with its blob in
   `apexzero.maze.sfl.VENDORED` and the README. Read, it imports `sfl.util.jaxued.jaxued_utils`
   (`l1_value_loss`), which the first approval did not cover. **Decided 27 Oct 2026 (later the
   same day):** the user approved that download too, with the two empty package files
   `sfl/util/__init__.py` and `sfl/util/jaxued/__init__.py`, at the same commit. They were
   downloaded into a new empty directory outside the repo, sizes and `git hash-object` blobs
   checked (2,942 bytes, `f491626f…`; 0 bytes, `e69de29b…` twice; all matched), read (the file
   imports only `jax`), and copied unmodified under `third_party/sfl/`, pinned in
   `apexzero.maze.sfl.VENDORED` and the README. **SFL source complete, adapter tested on the real
   network (27 Oct 2026):** `minigrid_plr.py` imports; the shim's end-to-end test now runs the
   vendored `minigrid_sfl.py` with SFL's own `ActorCritic`; and `tests/test_maze_e5.py` builds
   SFL's `ActorCritic(7)` for the Maze, initialises it, forwards a Maze observation (output
   shapes checked) and checks that `sfl_network_policy` returns valid actions and values, on a
   fresh network and on the shim's checkpointed parameters. Without a vendored network E5 would
   have run DR and PLR-MaxMC only; that fallback is no longer needed.
9. **D as the set the last block trained on** (score age 100 at every checkpoint), captured during
   training; the alternative is the set the checkpoint's own parameters would select next (age 0),
   which would have to be recomputed.
10. **The calibration's loads** (4 and 8 lanes per GPU) and its GPU-side capture check (Section 9).
11. **A with B or B′** (new, 27 Oct 2026, from the review). **Decided 27 Oct 2026:** when an arm's
    score is faithful in every reached phase and also shifts significantly, the label is A
    (faithful), not B or B′. The reason: B's reading ("only while the agent is incompetent")
    overstates a score that is still faithful late, and B′'s one that was faithful early. The
    shift is still reported in the row's shift table. A planted test now covers both directions
    (a shift up and a shift down, faithful throughout: row A only). Not taken: reporting B or B′
    and not A, the draft's first rule.
    - **When row G blocks A** (decided 27 Oct 2026, the same reason): for a score faithful in every
      reached phase, B and B′ are still not reported, because such a score is still faithful in
      every reached phase, so B's or B′'s reading would overstate it. A planted test covers it
      (a shift up and a shift down in an arm with G: no B, no B′).
    - **The label when A is withheld** (a labelling fix, 27 Oct 2026): row A stays literal, so a
      row G for the arm's stored score blocks A for every score of the arm. A score faithful in
      every reached phase whose A is withheld only for that reason is no longer reported as
      "mixed" (which says no row holds); it is reported as "faithful in every reached phase; A
      withheld by row G". G is still reported for the arm's stored score only. A planted test has
      a non-stored faithful score in an arm with G: that label, not mixed.
