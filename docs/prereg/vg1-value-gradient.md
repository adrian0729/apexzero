# VG-1: a value-gradient term in the critic's loss (pre-registration, DRAFT)

**Status.** DRAFT, 26 Oct 2026, for review. Nothing here is tagged and no VG-1 run exists.
- After review the lead commits the final text, the measurement and analysis scripts with their tests, and the queue script.
- A pilot of 6 runs (Section 4.1) then chooses β by a rule fixed here, on manipulation metrics only. The lead records the chosen β in Section 4.2 and places the annotated tag `vg1-prereg-v1` on that commit. It is pushed **before any main-run network is trained** (Section 9).
- Each run writes its completion outcome (`eval_rdev.json`) when its training ends, so the tag must come before training, not just before the analysis.
- Only the 12 R-dev circuits and generated strips are evaluated. R_TEST stays sealed, and `outputs/sealed/` is not read.

**Decision #5 (25 Oct, second round) fixes the frame:**
- the manipulation (C1's R²_c) is primary and completion secondary;
- 40 runs;
- a positive result does not change E0's recipe, so VG-1 stays off the lockbox's critical path.

This draft fixes everything else. Section 10 lists the items it could not settle.

## 1. Why

- **The refute.** Idea 2's C1 refuted in all 9 decision networks ([`costate.md`](../findings/costate.md), Section 5.2). The critics' input gradients, pulled back to the car's state, do not match the window problem's λ_s and λ_ξ: R²_c is −2.575 to 0.010 against a refute band of 0.2. The critics fit their own policies' values (R² 0.80–0.93), so the finding reads "values fit, gradients do not".
- **The matching intervention** (costate design Section 9; findings Section 10, item 5). Fit the critic's gradient directly, with a value-gradient term in its loss (DHP, GDHP, VGL). Test it like H4R: pre-registered, one machine, one commit, with C1's R²_c and R-dev completion as outcomes.
- **What VG-1 answers:**
  1. **Primary.** Does a GDHP-style term, with VGL(λ) targets taken through the JAX simulator, raise C1's frozen R²_c on fresh networks? This is the manipulation.
  2. **Secondary.** Does it change R-dev completion? This is the only causal test of the critic-to-actor path. The prior for a gain is low: the critic reaches the actor only through advantages, C3-w found no strong association, and the crash evidence points to width perception.
- **What it removes.** The arms differ only in `--vg-coef`. They use the same commit, the same box, the same seeds and interleaved lanes.

## 2. The term (fixed by VG-0)

**Code.** `src/apexzero/agents/ppo.py` (`ValueGradient`, `VGModel`, `vgl_scan`, `vg_targets`, `vg_loss_fn`), `src/apexzero/env/adapters.py` (`bank_vg_model`, `info['final_state']`), and `scripts/train_dr.py` (`--vg-coef`, `--vg-lambda`, `--vg-winsor`). Tests: `tests/test_value_gradient.py`.

**The state.** y = (x [8], prev_action [2], accel [2]): every continuous state that the observation and the reward read. V(y) is the critic of the environment's own Markov state. The track, the car context, the hidden grip and the timers are held. C1's scored components are s and ξ. Their columns of C1's on-manifold accel correction vanish (critic.py, `M_x`), so dV/ds and dV/dξ in y are C1's g_s and g_ξ before σ_R.

**The targets: VGL(λ).** They are taken through the simulator, the policy included, over one rollout:

    G'_t = D_y [ r_t / σ_t + γ c_t ],
    c_t = 0 after a termination;
          dV/dy at the pre-reset state after a truncation, and at the rollout's last step;
          λ G'_{t+1} + (1 − λ) dV/dy(y_{t+1}) otherwise.

- D_y is the total derivative of the recorded transition (`racing.step_no_reset` itself). The action depends on y through the policy mean at the recorded noise: a(y) = a_t + μ(o(y)) − stop_gradient(μ(o(y))).
- σ_t is the step's reward divisor, as the value targets use.
- It is computed by jacfwd of the one-step map and a reverse scan, with the rollout's own networks, as GAE's targets are.

**The loss.**

    L_critic = L_value + β · mean_rows mean_k ((dV/dy_k − G'_k) / scale_k)²

- **Rows:** those with finite targets, inside the loss mask.
- **Winsorisation:** each component's targets are clipped at their 0.005 and 0.995 quantiles over the rollout.
- **Scale:** scale_k² = Var(G'_k) + E[(dV/dy_k)²] over the rollout, with dV/dy from the rollout's critic. This is the targets' spread plus the critic's gradient size.
  - Without the second term, a random-init critic's first update scored about 26,000 in VG-0's smoke run. Almost all of it came from the accel channels, whose targets have an SD of about 1e-5 against the critic's 4·10⁻³ there.
  - With it, the first update scored about 21, and the later ones below 1.
- **Ω** is diagonal and equal over the 12 components after standardisation.
- The β-term's parameter gradient is a double backprop. The term reaches the critic only; the actor's loss is unchanged.
- **Fixed settings:** λ = 0.95 (the GAE λ, so both targets look the same distance ahead) and winsor 0.005. β comes from the pilot (Section 4.1).

**Default off, bit-identical.**
- `ValueGradient` is passed to `make_train` separately. It is **not** a `PPOConfig` field: the frozen `scripts/h4r.py` builds its expected PPO block from `asdict(PPOConfig(...))` and compares it with the H4R and step 2 runs' `config.json`, so a new field would change that harness's verdicts.
- Without `--vg-coef`, `train_dr.py` passes no term, and `make_rollout`, `make_learner` and `make_train` trace what they always did.
- **Checked:**
  - A tiny racing-env run reproduces the values recorded at `3323a0e` (before the term existed). On the recording platform it reproduces the sha256 of every final parameter byte.
  - The existing PPO goldens pass.
  - The term on with β = 0 gives the plain update bit for bit.
  - With β > 0 only the critic moves.

**Verified in VG-0** (`tests/test_value_gradient.py`):
- the model's step and observation equal the env's, to float32 rounding;
- dV/dy, the one-step Jacobians (policy path included) and the term's double-backprop gradient agree with central finite differences in float64;
- the VGL targets equal finite differences of the λ-return for λ = 0, 0.5 and 1, across a termination and a truncation;
- **on a 1-D braking problem with a fixed policy**, the critic trained with the term (β = 1) has the smaller state-gradient error in both components (relative error 0.12 and 0.03, against 0.22 and 0.11 for the plain critic on the same rollouts), and its value loss is lower (1.6e-5 against 3.1e-4).

## 3. The design tension, and how the design handles it

### 3.1 The one-step action projection barely touches s and ξ

PPO's actor sees the critic only through advantages. Over one step, an action reaches the value only through B_t = ∂y_{t+1}/∂a_t, so the one-step signal is B_tᵀ dV/dy(y_{t+1}). In 50 ms a steer or pedal input moves δ, pedal, vy, r and vx, and s and ξ only at second order. C1 scores s and ξ.

**Measured in VG-0.** `scripts/vgl.py residual` was run on H4R's five v1 baselines (`runs/h4r-v1-base-s10..14`). They share the recipe and are not among the costate study's 34 networks.
- Each run: 4 rollouts × 64 envs × 64 steps on fresh default-grammar strips, about 16,100 continuing steps per network.
- It gives each component's share of the projection: mean |B_kj G'_k|, normalised over k.
- This is an engineering measurement that decides nothing (`outputs/vg0/residual-h4r-v1-base-s1*.json`, at `43d68fe`, committed with this draft).

| Component | Share of steer's projection (targets) | Share of pedal's (targets) | Steer's, critic's own gradient (s10) | Pedal's, critic (s10) |
|---|---|---|---|---|
| s | 0.0003 in all five | 0.0027–0.0032 | 0.0001 | 0.0013 |
| ξ | 0.181–0.192 | 0.055–0.065 | 0.047 | 0.024 |
| δ, r, vy, vx, pedal, previous action, accel (together) | about 0.81 | about 0.94 | 0.95 | 0.97 |

So s is almost absent from the one-step signal. ξ carries about a fifth of the steer signal on the target side, because λ_ξ is large per radian even though ∂ξ/∂a is small. It carries less on the critic's own side.

**The tension:**
- A term aimed at the action-projected error (Ω = BBᵀ, the costate design's suggestion in Section 9) would leave g_s nearly untouched. The primary manipulation could then barely move by construction.
- A term that fits the full-state gradient moves g_s and g_ξ directly. But it spends much of the critic's capacity on directions the actor does not read in one step, so completion may not move even if the manipulation works.

**How the design handles it:**
1. **Full-state Ω** (Section 2). The primary is the manipulation (#5), so the term targets what C1 scores.
2. **The actor's horizon is longer than one step.** GAE's advantage integrates the critic over about 20 steps (λ 0.95, γ 0.995, about 1 s). Over that span a steer input moves ξ and then n, and a pedal input moves s through vx. So s and ξ reach the actor through the multi-step chain, not the one-step one. Even so, the full-state term is not a test of the action-projected path. It tests the full gradient, of which that path is a part.
3. **The action-projected residual is measured** (S4). On the same rollouts as M0 (Section 5.2), with the per-action normalised residual of B_tᵀ dV/dy(y_{t+1}) against B_tᵀ G'_{t+1}, it says whether the full-state term also improved the signal the actor reads.
4. **Completion is secondary** (#5), and its reading is conditioned on S4 (Section 6). If the manipulation works and S4 falls but completion does not move, "fix the critic's gradient" is closed as a lever at this scale. If S4 does not fall, a completion null says nothing about the actor path.

### 3.2 Pathwise targets miss the jumps at terminations

- **The problem.** The targets are pathwise derivatives along the recorded path. Where the return jumps with the state, dV^π/dy has terms from the jump (the change in crash probability under the stochastic policy) that a pathwise derivative omits. In the racing env, the termination penalty and the loss of all future progress are such jumps, near walls and the slip limit.
- **What VG-0 saw.** On the 1-D braking problem with the penalty charged per step, the return jumped by one penalised step whenever the crossing moved a step later. There the term competed with the value fit:
  - fitted offline to the exact values and their exact pathwise gradients (β = 1), a critic's value explained variance was about 0.5, against 0.996 for values alone;
  - with the penalty charged per metre (a continuous return), the term improved both values and gradients.
- **How the design handles it.** The racing env has such jumps, so the term can cost value accuracy:
  - the pilot's eligibility rule bounds that cost (Section 4.1);
  - S5 reports it on the main run.

### 3.3 V^π, not V*

The targets are the gradients of the stochastic training policy's own return along the recorded noise. C1 compares with the window problem's optimal λ*, from the optimum's states on synthetic corners, and the costate findings list why even an exact critic's gradient would differ (findings Section 2).

So M0 (S3) and the primary answer different questions:
- **M0:** does the term fit what it targets?
- **The primary:** does fitting it bring the gradient closer to λ*?

Section 6 reads them together.

## 4. Runs (fixed now)

**Common to every run:**
- `scripts/train_dr.py --steps 200e6 --seed <s> --name <name> [--vg-coef <β>]`.
- Every other argument is the default, as in H4R and GW-3's baselines: v1 (the base grammar), 2,048 envs, 64 steps, MLP 3 × 256 tanh, `ensemble_heads` 5, bank 4,096, refresh 256, γ 0.995.
- Each run writes 16 log-spaced checkpoints, the final one at 199,884,800 steps. It also writes `eval_rdev.json`: the default car, 10 timed laps per R-dev circuit from 0.9 × the reference start speed.
- `--vg-lambda` and `--vg-winsor` stay at their defaults (0.95, 0.005).

**Seeds.**
- No run on disk uses a seed of 15 or more, and GW-3 reserves 20–24 (20–29 if it runs 10 seeds).
- VG-1 uses 15 and 16 for the pilot and 30–49 for the main run. Seeds 17–19 stay unused.

**One machine.** One rented NVIDIA RTX A6000, the cheapest Massed Compute instance, under the standing permission.
- It is set up with `scripts/gpu_setup.sh`.
- `scripts/gpu_sync.sh` pushes R-dev data only. No R_TEST circuit leaves the Mac.
- The box is deleted when done.

### 4.1 The pilot (before the tag)

- **Runs.** β ∈ {0.1, 0.3, 1.0} × seeds {15, 16} = 6 runs, named `vg1p-b<β>-s<seed>` (for example `vg1p-b0.3-s15`).
  - The grid brackets VG-0's 1-D result: β = 0.1 already lowered the gradient error there, and β = 1 lowered it most.
  - The racing value loss at 200M is about 0.01 (G1's `metrics.csv`). The term measured 0.67 after 6 updates in VG-0's overhead runs (Section 8). So β = 1 lets the term dominate the critic's loss.
- **One commit.** All 6 train from one clean commit that contains VG-0. The queue records `git rev-parse HEAD` and refuses a dirty tree.
- **Lanes.** 3 lanes, one per β: seed 15 in the first round, seed 16 in the second. A run's box time is its wall time divided by 3.
- **Measured per run:**
  - **M1:** C1's R²_c (Section 5.1);
  - **M0:** the median normalised residual (S3, Section 5.2);
  - **EV:** the final explained variance, the last row of `metrics.csv`;
  - non-finite metrics, if any;
  - the box time per run.
  
  No pilot network's `eval_rdev.json` is read before the choice is recorded. Completion is not a pilot metric.
- **Eligibility.** A β is eligible if both its runs:
  1. finished with finite metrics;
  2. have EV ≥ 0.90. The 10 v1 baselines on disk end at 0.926–0.942 (`runs/dr-pilot-200M*`, `runs/h4r-v1-base-s1*`), so 0.90 bounds the value-accuracy cost of Section 3.2;
  3. have an M0 below 13.99, the median M0 of the five H4R v1 baselines (`h4r-v1-base-s10..14`: 10.54, 16.46, 14.30, 13.99 and 10.18), computed in VG-0 with Section 5.2's settings. The term must fit what it targets.
- **The choice.** β\* is the eligible β with the highest median M1 over its two seeds. A tie goes to the smaller β.
- **If no β is eligible,** the main run does not start. The pilot is reported, exploratory, and the lead brings the result to the user.
- **Cost check.** The main run's projected box time is 20 × 6.5 min + 20 × (β\*'s mean pilot box time per run). If it exceeds 11 h, the follow-through plan's upper estimate for 40 runs (r above about 4.1), the lead brings the cost to the user before the tag.
- **Label.** The pilot is exploratory. Its 6 networks never enter the confirmatory tests.

### 4.2 The main run (after the tag)

- **Arms:** `base` (no VG flag; the default path, bit-identical to H4R's and GW-3's baselines) and `vgl` (`--vg-coef β*`).
- **β\*:** to be recorded here after the pilot, before the tag.
- **Runs:** 2 arms × 20 seeds (30–49) = 40, named `vg1-base-s<seed>` and `vg1-vgl-s<seed>`.
- **One commit:** the tagged one. The queue records `git rev-parse HEAD` at launch and refuses a dirty tree.
- **Queue:** `scripts/gpu_queue_vg1.sh`, to be written on the pattern of `scripts/gpu_queue_h4r.sh`.
  - 4 lanes, XLA memory fraction 0.22 each.
  - Lanes 0 and 1 run `base` and lanes 2 and 3 run `vgl`, each pair taking seeds 30, 31, … in turn, so every round has 2 runs of each arm.
  - The `vgl` lanes run longer. The lanes' drift is logged.
- **Retry.** A run that fails is re-run once from scratch with the same arguments (its log kept as `.log.1`). It is **missing** only if it fails twice.
- **Manifest:** `outputs/vg1/manifest.json`, committed after pull-back. It holds every run's config, final timesteps, and the sha256 of its final checkpoint and `eval_rdev.json`; the queue log's sha256; the retry record; and the launch commit.
  - The analysis checks it first.
  - It refuses a run whose config differs from its arm's flags: the `vgl` arm's `config.json` must carry `vg` = {coef β\*, lam 0.95, winsor 0.005}, the `base` arm's no `vg` key.

## 5. Outcomes and the analysis (fixed now)

### 5.1 The primary: C1's frozen R²_c

**The statistic.** Each network's C1 statistic R²_c is C1's `stat`: the mean over s and ξ of the per-component R²_c. It is computed by the frozen pipeline of `costate-prereg-v1` exactly as for the 9 decision networks:
- the main value head's input gradient at T, pulled back to the car's state (`costate.critic.pullback`, the primary on-manifold form) and scaled by σ_R;
- calibrated per outer fold (partition 0) by the leverage-weighted clamped slope with direction intercepts (`costate.analysis.c1`);
- on T's interior stratum, K2b's decision rows only: λ_s on approach and exit rows (1,643 in the decision networks), λ_ξ on approach rows (1,581).
- The row counts are recorded per network. No braking or corner row is scored (decision #8).

**The pipeline.**
- **The script.** `scripts/vg1_c1.py` (to be built) runs in a git worktree whose `src/` is byte-identical to the tag `costate-prereg-v1`, as lane A's follow-ups do (`costate-followups-plan.md`, R2–R4). It reuses that lane's freeze and input checks:
  - the tag;
  - `src/` and the frozen scripts byte-identical;
  - the record's code hash `d22e670c…`;
  - the sha256 of `inputs/world.npz` and every other bound input.
- **Checkpoints.** It reads only the final checkpoints listed in VG-1's manifest, sha256-checked.
- **Wiring gate.** Before any VG-1 network, it recomputes G1's (`runs/dr-pilot-200M`) pullback at T from its pinned checkpoint, and C1 from that. Nothing is scored unless both hold:
  - the pullback equals G1's bound critic file (`post/critic/`), as CS-1's fresh VJP did bit for bit;
  - C1 reproduces `decisions.json`'s `stat`, `r2c` and the per-fold calibration to 1e-12. CS-1 reproduced them to 3.6·10⁻¹⁵.
- **K3 after the tag.** Each network must pass the frozen after-the-tag checks (`k3_trained`, as the costate `critic` stage runs them). A network that fails is missing.

**P (primary): `vgl` above `base` on R²_c.**
- **Test:** a one-sided exact Wilcoxon rank-sum (Mann–Whitney) test, `vgl` against `base`, 20 against 20.
  - The p-value comes from the exact permutation distribution of the rank sum, with mid-ranks for ties.
  - The distribution is enumerated by the standard recursion over ranks. A tie counts toward p.
- **Why ranks.** R²_c is heavy-tailed: the five v1 baselines score 0.010, −0.138, −0.726, −1.359 and −1.403, because a wrong-sign tail of large ξ gradients meets the clamped calibration (findings Section 7.1). A mean-difference test would be decided by its tail.
- **Confirm** if p ≤ 0.05. **Refute** if the opposite one-sided p ≤ 0.05. Otherwise **inconclusive**.
- **Missing networks** are dropped, and the exact test is computed on the rest. P is **not testable** with fewer than 15 networks in either arm, and not testable counts as not confirmed.
- **The point estimate** reported with P is the Hodges–Lehmann shift (the median of the 400 pairwise differences), with a 95% percentile bootstrap CI (20,000 draws, `numpy.random.default_rng(20261026)`).

### 5.2 Secondary (each at 0.05, unadjusted, labelled secondary)

Every secondary test is an exact rank-sum test of the same construction, unless stated.

- **S1, completion (two-sided).** C(network) is the number of R-dev circuits with completion > 0.5 in its `eval_rdev.json` (10 laps each), out of 12.
  - Reported with the mean difference and its 95% bootstrap CI.
  - It is read whatever P shows. Its reading is conditioned on S3 and S4 (Section 6).
- **S2, rank agreement (one-sided, `vgl` above).** C1's Spearman ρ, the frozen within-direction form, averaged over s and ξ, on the same rows.
- **S3 = M0, the term fits what it targets (one-sided, `vgl` below).**
  - **The rollouts:** `scripts/vgl.py residual` at fixed settings: 4 rollouts × 64 envs × 64 steps of the network's own stochastic policy on fresh default-grammar strips (bank 256, seed 20261026, the car randomised as in training).
  - **The targets:** VGL(0.95) from the network's own actor and critic, winsorised at 0.005.
  - **The statistic:** the median over the 12 components of E[(dV/dy_k − G'_k)²] / Var(G'_k).
    - It is a median because the critic's gradient on the channels whose targets barely vary dominates any mean: `h4r-v1-base-s10` scores 834 on ay, 175 on the previous steer and 80 on ax.
    - The same network's median is 10.54, with s 6.9 and ξ 4.5.
    - The per-component values and their mean are descriptive.
  - Both arms are measured identically. Nothing is evaluated on any circuit.
- **S4, the action-projected residual (one-sided, `vgl` below, per action).** From the same rollouts: per action (steer, pedal), E[(B_tᵀ(dV/dy − G')_{t+1})²] / Var(B_tᵀ G'_{t+1}) on continuing steps.
  - H4R's five v1 baselines give steer 3.4–11.9 and pedal 5.0–7.9.
  - These are two tests, steer and pedal, reported separately.
- **S5, value accuracy (two-sided).** The final explained variance (the last row of `metrics.csv`). This is the term's cost.
- **S6, paired by seed (one-sided, `vgl` above).** The 20 same-seed differences in R²_c, tested by their mean over all 2²⁰ sign flips (exact). Missing pairs are dropped.

### 5.3 Descriptive (decides nothing)

- per component, R²_c and Spearman ρ for s and ξ;
- the per-network table;
- M0 per component, and the action-projection shares of Section 3.1 per arm;
- the per-circuit completion share per arm, and the lap-time gap on the circuits both arms complete (IN-4's form, against `outputs/ocp/<R-dev>.pkl`);
- each arm's per-update box time;
- R²_c over the training snapshots of 4 networks per arm (seeds 30–33; C1-dev's form, `costate.analysis.c1_dev`);
- the pilot's 6 networks beside the main run's, labelled pilot.

**Optional:** C1-own's R²_c against λ^own (the frozen `own` stage), whose targets are closest to what the term fits. It costs about 100 s per network on the Mac (the costate `own` stage's 9 networks took 15 min). The lead decides before the tag (Section 10).

### 5.4 Power (computed for this draft)

- **P (rank-sum, 20 against 20, one-sided 0.05; 4,000 simulations):**
  - for a location shift of 0.5, 0.8 and 1.0 SD, power is 0.44, 0.78 and 0.90;
  - resampling the five v1 baselines' R²_c with smoothing (SD 0.66), shifts of +0.25, +0.5 and +0.75 in R²_c give power 0.38, 0.77 and 0.95.
- **S1 (completion, two-sided; H4R's v1 baselines 5.0 ± 1.58 circuits, rounded):** shifts of +1.0, +1.5 and +2.0 circuits give power 0.46, 0.80 and 0.96.

At 40 runs, a completion change of about 1.5 circuits is detectable. A smaller one would read as inconclusive.

## 6. What counts as the result

| P | S3 (M0) | S1 | Reading |
|---|---|---|---|
| confirm | confirm | any | The term moves the critic's gradient toward the window problem's λ on the decided rows (pre-registered, p = …). S1 then says whether that reaches driving. If S1 is null and S4 confirmed for steer or pedal, "fix the critic's gradient" is closed as a completion lever at this scale (200M, v1). If S4 did not confirm, the actor's signal was not shown to change, and the actor path is not tested. |
| not confirm | confirm | any | The term fits the pathwise ∇V^π it targets, but that does not bring the gradient closer to λ*. This is consistent with the gap lying between V^π's pathwise gradient and λ* (Sections 3.2 and 3.3), not only in the regression leaving the gradient undetermined. |
| any | not confirm | any | The manipulation failed: the critic did not fit its targets better. P and S1 are reported but say nothing about the gradient's role. |
| refute | any | any | The term lowers R²_c (pre-registered). Reported with S3–S5. |

- **S1 alone,** whatever P shows, is reported as the completion effect, with its CI and S5's cost.
- **A positive result does not change E0** (#5). E0's recipe is fixed by GW-4 without VG. Any later use of the term is a new study.

## 7. E0 and the lockbox

- VG-1 is off the critical path. GW-3, GW-4 and E0 do not wait for it, and nothing in it changes `generator-v1`, `sim-v1` or E0.
- VG-1's networks are never evaluated on R_TEST. They are not lockbox subjects.
- `train_dr.py`'s end-of-run evaluation reads R-dev only. `scripts/vgl.py` reads generated strips only.
- With `--vg-coef` absent, the training path at the VG-1 commit is the default path bit for bit (Section 2). That keeps any later default-path round, such as GW-3's or E0's, unaffected by the term's code.

## 8. Cost

**Measured on the Mac** (VG-0; `scripts/vgl.py overhead` at `43d68fe`; `outputs/vg0/overhead-e128.json` and `-e512.json`, committed with this draft).
- **What is timed:** one PPO update of the recipe (MLP 3 × 256, 5 heads, 4 epochs × 8 minibatches), bank refresh, rollout, learner and normaliser update included, at a small size.
- **Conditions:** 4 XLA threads at nice 10, with another workflow sharing the machine. Each arm is the median of 3 runs of 6 updates, after compiling.

| Envs × steps (transitions per update) | Default path, s per update | With the term, s per update | Ratio | Compile, default / term (s) |
|---|---|---|---|---|
| 128 × 64 (8,192) | 0.79 | 2.23 | 2.82 | 8.2 / 15.1 |
| 512 × 64 (32,768) | 1.83 | 6.52 | 3.57 | 8.7 / 16.3 |

- **Scaling.** The ratio grows with size: the term's work scales with the transitions, while part of the default update does not. At the full 2,048 × 64 it is likely above 3.6 on this CPU.
- **Where the cost goes.** The term costs a jacfwd of the one-step map (12 tangents through the 10-substep dynamics, the observation map and the actor) per transition, the critic's dV/dy at the pre-reset states, and a double backprop through the observation map and the critic in each of the 32 minibatches.

**A6000.** The overhead on the GPU is unmeasured. The pilot measures it.
- The baseline costs about 6.5 min of box time per 200M run at 4 lanes (H4R: 20 runs in 2 h 13 min).
- With r the GPU ratio, the main run costs 20 × 6.5 × (1 + r) min and the pilot 6 × 6.5 × r min, plus about 15 min of setup and pull-back per session.

  | r | Pilot | Main run | Total |
  |---|---|---|---|
  | 2 | 1.3 h | 6.5 h | about 8 h |
  | 3.6 (the Mac's at 512 envs) | 2.3 h | 10.0 h | about 13 h |
  | 4.1 | 2.7 h | 11.0 h | about 14 h |

- The follow-through plan's estimate for 40 runs was 5–11 h, from a guessed ratio of 1.5–3. Section 4.1's cost check holds the main run to that plan's upper end.

**Mac.**
- `scripts/vg1_c1.py` (the critic stage: one main-V VJP at T's 9,887 states per network) takes about 20 s per network (the costate `critic` stage: 34 networks in 11 min), about 15 min for 46 networks.
- `scripts/vgl.py residual` takes about 30 s per network (measured on `h4r-v1-base-s10`: 26 s wall at 4 threads), about 25 min for 46.
- The descriptive C1 over the training snapshots (8 networks × 16) takes about 45 min.
- The optional C1-own stage would add about 75 min of CPU.

## 9. Procedure

1. **Review this draft.** The user decides Section 10's items marked "the user".
2. **Build and test,** on planted data only:
   - `scripts/vg1_c1.py`, with its wiring gate;
   - `scripts/vg1_analysis.py`: the exact rank-sum, the sign-flip test, the pilot rule and the manifest checks;
   - `scripts/gpu_queue_vg1.sh` (pilot and main modes).
   
   Commit them with this text.
3. **Pilot.** Rent the box, train the 6 pilot runs from that commit, pull back (checking each checkpoint's sha256 against the box's), and delete the box. Run M1, M0 and EV, apply Section 4.1's rule, and record β\* and the per-run box times in Section 4.2.
4. **Tag** `vg1-prereg-v1` (annotated) on the commit that records β\*, and push it. The tag object's sha is recorded in the findings.
5. **Main run.** Rent the box, set it up at the tag, and run the queue. Pull back with sha256 checks, delete the box, and commit the manifest.
6. **Measure:** `scripts/vg1_c1.py` (wiring gate first), then `scripts/vgl.py residual` on all 40.
7. **Analyse** once: `scripts/vg1_analysis.py` refuses if its output exists or was ever committed. Write `docs/findings/vg1.md`.

**Deviations.** Any change after the tag is logged in the findings with its date and reason. A change to an outcome, test, seed or rule after any main-run network has started makes the affected result exploratory.

## 10. Open items before the tag

1. **"With a rank test" (the lead, then the user).** This draft reads decision #5's phrase as the between-arm test being a rank test (P is an exact rank-sum on R²_c), because R²_c's tail would decide a mean test. C1's own rank agreement (Spearman) is S2. If #5 meant Spearman's ρ as a co-primary, P becomes a family of two with Holm (m = 2).
2. **β's grid and the pilot rule (the user).** The grid {0.1, 0.3, 1.0}, the EV floor of 0.90 and the M0 eligibility condition are fixed above. A wider grid costs 2 runs per value.
3. **C1-own (the user).** Include it as a secondary, or keep it optional and descriptive (Section 5.3). Its targets are the closest to what the term fits.
4. **To build (the lead):** `scripts/vg1_c1.py`, `scripts/vg1_analysis.py`, `scripts/gpu_queue_vg1.sh` and the manifest writer. `vg1_c1.py` belongs with lane A's frozen worktree and needs that lane's check helpers (on the `costate-exploratory` branch, not on main).
5. **Seed ranges (the lead).** Keep GW-3 and any other `train_dr.py` round off 15–19 and 30–49.
6. **The box order (the lead).** VG-1 queues behind GW-3 on the one A6000 (follow-through Section 4, Phase 3). The pilot can run in any gap.
