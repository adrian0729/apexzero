# Gate diagnostics (GW-1): plan (exploratory; written before any computation, 25 Oct 2026)

**Status.** This is item GW-1 of the follow-through plan ([`docs/plan/follow-through.md`](../plan/follow-through.md), Sections 3 and 7.4). It is exploratory and is not pre-registered with a tag. This plan is committed before anything below is computed, the timing run included, so the measures, rules and expected directions are fixed in advance. Later changes are dated amendments (Section 10). Nothing is trained. Only R-dev circuits and generated levels are used. R_TEST stays sealed.

## 1. Question

The generalist gate is read at E0 (5 seeds × 1B steps): R-dev completion of at least 80% (in practice 10 of 12 circuits) and late-phase A_DR completion of at least 90% ([`two-track-plan.md`](../plan/two-track-plan.md), Section 5.2). No run on disk reaches the R-dev half reliably: 0 of 25 baselines and 2 of 26 in-sample real-width runs complete 10 or more circuits (follow-through, Section 7.4).

The bounded budget you chose (#11, follow-through Section 11) allows at most one GW-5 lever round and one GW-6 pilot before `generator-v1` is frozen. GW-1 asks which lever the evidence on disk favours:
- **the budget lever** (more steps: GW-6's 1B pilot), or
- **the margin lever** (more inside clearance: GW-5's training-only termination margin).

It does so with three diagnostics, none of which trains anything:
- **D1.** R-dev completion against training steps, from the checkpoints already stored.
- **D2.** Deterministic A_DR completion of the final checkpoints.
- **D3.** H4R Section 10.2's crash comparison ([h4r.md](h4r.md)): are the real-width networks' extra completions avoided inside-edge crashes, and is their clearance at the former crash corners larger by about H4R's B shift?

## 2. Networks (already trained)

The 28 runs that share one training code and one GPU type (h4r.md, Section 2.3; follow-through, Section 9.1). The training code is unchanged since `2143a59`, and all 28 trained on a rented RTX A6000.

| Arm | Runs | Variant | n |
|---|---|---|---|
| v1 baseline | `runs/h4r-v1-base-s10..14` | v1 | 5 |
| v1 real width | `runs/h4r-v1-realw-s10..14` | v1 | 5 |
| v3 baseline | `runs/h4r-v3-base-s10..14` | v3 | 5 |
| v3 real width | `runs/h4r-v3-realw-s10..14` | v3 | 5 |
| v3 leave-out | `runs/dr-realw-loo-f1-s0..3`, `runs/dr-realw-loo-f2-s0..3` | v3 | 8 |

- Every run stores 16 log-spaced checkpoints, from 10,000,000 to 199,884,800 steps.
- The H4R finals are bound by sha256 in the committed `outputs/h4r/manifest.json`, and the leave-out finals in `src/apexzero/fields/prereg.json`. Each is checked before it is read. Intermediate checkpoints are bound nowhere, so the script records the sha256 of every file it reads.
- The learning rate and the entropy coefficient anneal to zero over the 200M schedule (`anneal_lr`, `ent_coef_final` 0 in every `config.json`).

## 3. D0: timing first

Before any other evaluation, one evaluation is timed on the Mac.
- **The timed evaluation** is `h4r-v1-base-s10` at 10.0M steps. It belongs to D1's grid, so its result is kept.
- **How it is timed:**
  - first alone;
  - then as one of 4 concurrent processes (`nice -n 15`, each limited to 2 XLA CPU threads).
  - The times go to `outputs/gate-diagnostics/timing.json`.
- **The schedule rule** uses seconds only, never a completion value.
  - Let t₄ be the wall time per evaluation with 4 concurrent processes. The estimated wall time for D1 is W = 168 × t₄ / 4.
  - **W ≤ 6 h:** everything runs on the Mac.
  - **W > 6 h:** the 168 evaluations run on one A6000 under the standing permission (the cheapest Massed Compute box, deleted when done), at one commit. D3's traces then come from the same box.
  - Either way, every point of every curve, and every D3 trace, comes from one machine.
- **Reported estimates:**
  - about 30 s per evaluation on the Mac, by the lockbox report, which gives W of about 21 min;
  - about 3.5 min of A6000 time, by the generator report.

## 4. D1: R-dev completion against training steps

**Grid.**
- Six checkpoints of the 16: indices 0, 3, 6, 9, 12 and 15 of each run's schedule. That is 10,000,000; 18,205,642; 33,144,540; 60,341,763; 109,856,054; and 199,884,800 steps, about a factor of 1.82 apart.
- 28 runs × 6 checkpoints = 168 evaluations.
- The finals are evaluated again too, so that each curve comes from one machine.

**Protocol.** Exactly `scripts/eval_rdev.py`'s, which is also `train_dr.py`'s end-of-training evaluation:
- the default car;
- `eval.laps.timed_laps` with 10 runs and seed 0: run 0 starts at n = 0 and 0.9 × the reference start speed, and runs 1–9 are jittered;
- two laps, with lap 2 timed, on the refined references of the 12 R-dev circuits;
- a circuit is completed when more than half its runs finish;
- the gap is measured against `outputs/ocp/<circuit>.pkl`.

**Measures per network and checkpoint:**
- **C**, circuits completed (0–12), which is the gate's unit;
- **c̄**, the mean over circuits of the completion fraction (0–1), which is finer-grained;
- per circuit, the completion fraction and the median gap.

**Platform check.**
- The stored final evaluations (`eval_rdev.json`) were written on the A6000 at the end of training. The Mac re-evaluation of each final is compared with them over the 28 × 12 completion calls and over C per network.
- Disagreements are reported, not repaired. The curves use the D1 machine's values throughout.

**Summaries:**
- **Per arm, at each checkpoint:** the mean and range of C and of c̄.
- **Per variant, at each checkpoint:** real width minus baseline.
- **The late change** d_late = C(199.9M) − C(109.9M) per network, and the mid change d_mid = C(199.9M) − C(60.3M). For each:
  - the mean over the 28 networks, with a 95% percentile bootstrap CI over networks (20,000 draws, seed 0);
  - how many networks rise, stay level and fall;
  - the same for c̄.
- **Best checkpoint.** The share of networks whose best C of the six is not the final, with best minus final. A tie with the final counts as the final. This bears on #10, whether an R-dev-selected checkpoint should count.

**Label for the late curve** (fixed now):
- **rising:** the CI of mean d_late lies above 0;
- **flat:** the CI lies within [−0.5, +0.5] circuits;
- **falling:** the CI lies below 0;
- **unresolved:** anything else.

**Caveat fixed now.** The schedule anneals to zero at 200M, and a 1B run anneals over 1B, so a 1B run is not a continued 200M run (follow-through, GW-6).
- A curve that still rises late, despite the anneal, is direct evidence for the budget lever.
- A flat end cannot separate the anneal from a capacity limit.

## 5. D2: deterministic A_DR completion of the finals

**Levels.**
- All 1,000 A_DR levels: `tracks.levels.make_levels(PRNGKey(2026), 'a_dr')`, with the grammar fitted on the 12 R-dev circuits. This is how `scripts/pilot_audit.py` (`pilot_levels`) builds them.
- Each level keeps its own car context, hidden grip and start (s = 0, n = 0, 0.8 v_QSS(0)).
- No level is selected or tuned on.
- At E0 the gate will be read on `generator-v1`'s A_DR, which is regenerated from the same keys.

**Primary measure (deterministic).**
- The mean action, with no observation noise, over one 1,200-step window per level (60 s, the environment's `window_steps`).
- A level is completed when the window ends without termination.
- Per network: the completion share over the 1,000 levels, with a Wilson 95% interval. Per arm: the mean.

**Secondary measure: the audit's window completion.** This is the phase measure the gate names.
- It uses `audit.rollouts.rollout_windows` at its defaults (stochastic actions, 1% observation noise), with 4 windows per level and key `fold_in(PRNGKey(2026), 1)`, as the pilot audit uses.
- The phase is "late" at 90% or more.

**Comparison.** Each final's A_DR completion is set against its R-dev c̄ from D1, network by network.

**Note.** A_DR uses the default width model. The real-width and leave-out arms trained on real width profiles, and v3 trained half on the tilted grammar, so all results are reported by arm.

## 6. D3: the crash comparison (H4R Section 10.2)

**Networks.** H4R's 20 finals: the same-commit, pre-registered arms.

**Traces.**
- The D1 final evaluations are recorded in full: s, n, v_x and the termination flags of every run at every step.
- The new script's rollout must reproduce `timed_laps`'s completion and lap times exactly (test-checked).

**Corners and clearance.**
- **Corners** come from `tracks.corners.segment` on each refined R-dev track, at its default thresholds. σ is the corner's sign. Each corner's window is W_k = [s_start − 50, s_end + 50] m, step 2's c_min window.
- **Clearance** is c(s) = w_in(s) − 1.0 − σ·n(s).
  - w_in is the refined reference's width on the corner's inside: w_left for σ = +1, w_right for σ = −1.
  - 1.0 m is the evaluation's `edge_margin`, the termination boundary that B is measured against.
  - For each passage of a run through a window (lap 1 and lap 2, truncated at termination), c_k is the minimum over the window. A run that leaves by the inside edge in W_k has c_k ≤ 0 there.

**Crash census.** For every run that terminates:
- s and lap;
- the cause: corridor, slip, spin or stall, in `diagnose_crashes.first_crash`'s order;
- the side: inside if σn > 0 at termination, for the corner whose window contains it; outside if σn < 0; "straight" if no window contains it;
- the corner.

**Definitions fixed now:**
- **Former crash corners.** Per variant, the corners where at least 3 of the baseline arm's 50 runs (5 seeds × 10) terminate.
- **Gain circuits.** Per variant, the circuits that more real-width than baseline networks complete (by the more-than-half rule), from the D1 re-evaluation. The stored table (h4r.md, Section 6.1: 9 gain circuits in v1 and 3 in v3) is reported beside it.

**Q1: are the extra completions avoided inside-edge crashes?**
- On each variant's gain circuits, ΔT is baseline terminations minus real-width terminations, pooled over runs.
- The explained share s_in is the same difference counted only over inside-edge corridor terminations at former crash corners, divided by ΔT.
- Also reported: each former crash corner's termination rate per arm (terminations per passage reaching its window).

**Q2: is the clearance larger by about the B shift?**
- At each former crash corner, each network's median c_k over its passages. Δc_v is the mean, over the variant's former crash corners, of the real-width arm's mean minus the baseline arm's mean.
- The 95% CI comes from a bootstrap that resamples networks within each arm (20,000 draws, seed 0), with the corners fixed.
- **The reference** is H4R's confirmatory B shift, as arm means of median B: ΔB_v1 = +0.44 m and ΔB_v3 = +0.27 m (h4r.md, Section 1).
- **"About the B shift"** is fixed now as Δc_v ∈ [0.5, 2] × ΔB_v: 0.22–0.88 m in v1 and 0.14–0.54 m in v3.

**Secondary, labelled as such:**
- Δc on passages that clear the corner only (selection-biased);
- Δc on all corners, not only former crash corners (the general margin on real roads);
- Q1 and Q2 per circuit.

The leave-out runs are not in D3: they have no baseline at their own commit. Their D1 and D2 values are descriptive.

## 7. Expected directions (stated before computing)

- **E1 (D1).** C rises with log steps from 10.0M, with most of the rise before 60.3M. The late curve is "flat" or "unresolved": the median network moves by at most 1 circuit between 109.9M and 199.9M. The v1 real-width advantage over baseline is visible by 60.3M. At least a third of the networks have their best C before the final.
- **E2 (platform).** Mac and stored completion calls agree on at least 90% of the 336 network-circuit calls, and C per network agrees within ±1 for most networks. Disagreements fall on circuits near half completion.
- **E3 (D2).**
  - Deterministic A_DR completion of the finals is at least 0.90 for the median network in every arm, and well above that network's R-dev c̄.
  - The stochastic window completion is lower, about 0.8–0.9. The pilot audit's stochastic window completion was 78–89% at 60–110M on one seed ([pilot-audit.md](pilot-audit.md)).
- **E4 (D3, Q1).**
  - Baseline terminations are mostly (at least 80%) inside-edge corridor exits at corners, as on the transplanted crash corners (159 of 160; [psychophysics-step1.md](psychophysics-step1.md), Section 6.2).
  - In v1, s_in is at least 2/3. In v3 there are few gain circuits, so Q1 is weak there.
- **E5 (D3, Q2).** Δc_v1 > 0 and inside the band. Δc_v3 is unresolved (its CI includes 0).
  - This is held weakly. H4R's S3 found no within-arm link between B and completion. On the common keys, the B shift is the same in v1 and v3, while the completion gain is four times larger in v1 (h4r.md, Section 8).

## 8. Readings: what each outcome means for the two levers

These readings say how to use the bounded slots of #11. They decide nothing confirmatory: the gate is read at E0 (#10).

| Outcome | Reading | Where it points |
|---|---|---|
| D1 "rising" | More steps still pay at 200M, even under the anneal | **Budget lever.** The GW-6 1B pilot is warranted (its own condition, "curves still rising") |
| D1 "flat" or "falling" | The 200M schedule saturates; the anneal and capacity cannot be told apart | GW-6 is not warranted on this evidence. E0's 1B stays as defined |
| D1 "unresolved" | Seed noise is larger than the late change | GW-6 stays optional; D2 and D3 carry the choice |
| D2: median A_DR ≥ 0.90 while R-dev c̄ ≤ 0.7 | The generator's own distribution is mastered; the R-dev shortfall is transfer | More steps on the same generator are unlikely to close the R-dev half alone. GW-2's width model and the margin lever are the candidates |
| D2: median A_DR < 0.90 | The networks have not mastered their own distribution | The budget lever has headroom on both halves of the gate |
| D3 margin-consistent: s_in ≥ 2/3 and Δc in the band, in v1 at least | The extra completions are avoided inside-edge crashes, with about the synthetic B shift in real clearance | **Margin lever.** GW-5's training-margin arm (`edge_margin` 1.3–1.5 m in training only) |
| D3: s_in ≥ 2/3 but Δc below the band | Crashes are avoided without much more clearance: the line is placed or timed differently, not with a uniform margin | GW-5's local-perturbation arm (judged on short-scale L/e, [locality.md](locality.md)) before the training margin; GW-2's corner-coupled dip |
| D3: s_in ≥ 2/3 but Δc above the band | Real roads amplify the margin far beyond the synthetic B | The margin lever is still favoured, but B on A2 corners understates it. Report it before GW-5 is designed |
| D3: s_in < 1/3 | The extra completions are not mainly avoided inside-edge crashes | Neither lever as framed. Report it and reconsider before GW-5 |

- If D1 and D3 each point to their own lever, both slots can be used: #11 allows one of each.
- GW-1 trains nothing and selects no recipe. Its readings steer the bounded lever slots, though, so it counts as one use of R-dev under #11's erosion bound.

## 9. Compute, code, outputs and integrity

**Compute.**
- On the Mac: about 1.5–2 h in all (est.). D1's wall time is set by D0. D2 takes minutes per network for the deterministic pass, and about 4 times that for the stochastic one. D3's analysis takes seconds once the traces exist.
- At most 4 processes at a time, each under `nice -n 15`. Another workflow shares the Mac.
- A6000: none, unless D0's rule sends D1 there (about 1.5 box-h, est.).

**Code.**
- `scripts/gate_diagnostics.py` (new), with stages `time`, `curves`, `adr`, `crashes` and `report`.
- `tests/test_gate_diagnostics.py` (new). It covers:
  - the rollout's equality with `timed_laps`, and with `scripts/eval_rdev.py`'s JSON, on one checkpoint;
  - clearance and side on planted traces;
  - the former-crash-corner and gain-circuit rules;
  - the bootstrap;
  - the lockbox refusal.
- `scripts/diagnose_crashes.py` is not used. It writes into `docs/findings/` and keeps only the first crash of one run. If it is used for a cross-check, it first gets an `--out` option, with its default left unchanged.
- No file under `src/` is added or changed: the costate freeze hash covers all of `src/`. No frozen harness is touched.

**Outputs.**
- `outputs/gate-diagnostics/`: `timing.json`, `curves.json`, `adr.json`, `crashes.json` and `traces/`.
- Each carries a provenance block: the commit, changed paths, and the sha256 of every checkpoint and evaluation read.
- A findings note, `docs/findings/gate-diagnostics.md`. Every number in it is labelled exploratory.

**Lockbox.**
- Every circuit name passes `viz.export.lockbox_guard` before any rollout. R_TEST names and their mirrored or reversed variants are refused.
- Gaps are measured only against `outputs/ocp/<R-dev circuit>.pkl`.
- `outputs/sealed/` is not read.

## 10. Amendments

Any change to this plan after its commit is a dated amendment appended here, and it is committed before the computation it changes. Each amendment states:
- what changed, and why;
- which results, if any, had been seen when it was written.

Results computed under the earlier version are kept and reported beside the amended ones. An amendment written after a result was seen is labelled post hoc.

### Amendment 1 (25 Oct 2026; written before any computation)

**What changed, and why.** Writing `scripts/gate_diagnostics.py` showed points the plan leaves open. Each is fixed here before any evaluation, the timing run included. None changes a measure, a threshold or an expected direction.

**Results seen when it was written.** None. The only rollouts run so far are the script's unit tests: planted traces, and a fixed hand-written policy (no network) on Norisring, which checks the rollout against `timed_laps`.

- **D0.**
  - All timed processes run at nice 15 with 2 XLA threads, the one alone included.
  - The 4 concurrent processes all evaluate the timed checkpoint. Their results must equal the alone run's (a determinism check). Only the alone run's result is kept as the D1 grid point.
  - t₄ is the longest of the 4 evaluation wall times, from loading the checkpoint to the end of the 12th circuit, JIT compilation included and interpreter start excluded. The D1 pool reuses its processes. The process wall times are recorded beside it.
- **D1.**
  - **Two labels at once.** A CI can lie above 0 and inside [−0.5, +0.5] (or below 0 and inside it). Then both labels are reported, and the directional one (rising or falling) is the label read in Section 8. The flat band is meant for a change not resolved from zero.
  - **Bootstraps.** NumPy `default_rng(0)` is restarted for each summary (d_late and d_mid, for C and for c̄; each Δc of D3). The indices are `integers(0, n, (20000, n))`, and in D3 the baseline arm is drawn before the real-width arm.
  - **Rise, level, fall.** d > 0, d = 0 and d < 0 exactly (c̄ moves in steps of 1/120).
  - **Best checkpoint.** The step reported is the earliest checkpoint with the best C.
  - **Platform check.** The completion fractions themselves are also compared exactly (descriptive).
- **D2.**
  - **"Median" in Section 8's D2 rows.** The median over all 28 finals, taken separately for A_DR completion and for R-dev c̄. The per-arm medians, which E3 reads, are reported beside it.
  - **E3's "well above that network's R-dev c̄".** Read on the arm medians, with each network's difference reported.
  - **Strip ends.** A window that ends at the strip's end without termination counts as completed, as in the audit.
- **D3.**
  - **Crashes.** A crash is a run that terminates without completing. A termination after lap 2 is not one, since that run is complete. A run that neither completes nor terminates by the rollout's last step is a time-out, reported separately and not counted as a termination.
  - **Passages.**
    - A passage is a maximal stretch of consecutive post-step samples whose s mod L lies in W_k. Samples run up to the run's end: the terminating step, or for a run that completes, the last sample before s = 2L.
    - The terminating sample always counts. The initial state is not a sample, as in `timed_laps`'s arrays.
    - A window that crosses s = 0 gives a partial passage at the start of lap 1 and one at the end of lap 2. A passage across the line is one passage. Its lap is the lap at entry.
  - **Widths.** The environment's own width spline is used (`widths_at` on the track arrays, float32), so an inside corridor exit has c ≤ 0.
  - **Overlapping windows.** A termination inside two windows goes to the corner whose span [s_start, s_end] is nearest (distance 0 inside it). A tie goes to the earlier corner. At exactly σn = 0 the side is "outside" (this does not occur in practice).
  - **Q1.**
    - ΔT and its inside-edge part are sums over the variant's gain circuits (50 runs per arm per circuit).
    - s_in is undefined when ΔT ≤ 0.
    - The termination rate is the terminations assigned to the corner, divided by the passages through its window.
    - E4's "at corners" means a termination assigned to any corner. The share at former crash corners is reported too.
  - **Q2.**
    - A network's median runs over all its passages at the corner, both laps.
    - A network with no passage at a corner is left out of its arm's mean there.
    - A corner whose arm mean is undefined in a bootstrap draw is left out of that draw's mean over corners. The number of such draws is reported.
    - "All corners" (secondary) means every corner of the 12 circuits.
  - **Leave-out traces.** The leave-out finals' traces are recorded but not used in D3.
- **Outputs.** Besides the files of Section 9, per-evaluation records `evals/<run>/<step>.json` and per-network `adr/<run>.json`. Like the rest of `outputs/`, they stay local.
