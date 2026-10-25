# Interpretability leftovers (IN-1 to IN-4): plan (exploratory and descriptive; written before any computation, 25 Oct 2026)

**Status.** These are items IN-1 to IN-4 of the follow-through plan ([`docs/plan/follow-through.md`](../plan/follow-through.md), Section 7.8). IN-1 and IN-2 are exploratory summaries outside step 2's frozen code; IN-3 and IN-4 are descriptive write-ups. None is pre-registered with a tag, and none can change a confirmatory decision of step 1, step 2 or H4R. This plan is committed before anything below is computed, so the measures and expected directions are fixed in advance; later changes are dated amendments (Section 6). No network, closed loop, gradient or solver is run: every number comes from files already on disk. R_TEST stays sealed.

## 1. Rules common to all four items

- **Inputs, checked before use.**
  - Step 2's committed confirmatory decisions, `outputs/fields/analyse/decisions.json`, by its sha256 `7f5941453539cc0071686b9c041286f783cc3ae8744f7d1f107a76e599656fcf`.
  - Step 2's row files (`outputs/fields/fields-a2/*.jsonl`, `fields-anchored/*.jsonl`, `fields-earlier/*.jsonl`, `opt-fields/rows.jsonl`), each against the sha256 its provenance sidecar records; the key lists `outputs/fields/keys/lists.json`.
  - Step 1's non-confirmatory decisions, `outputs/psychophysics/analysis/decisions_all.json`, by its sha256 `14c674863e78b68cf01986bddcaec34480700ea21811aa0971550711feca4ea1` (written by the non-confirmatory `analyse` of 21 Oct, 01:45–01:48; its modification time is that run's end).
  - H4R's 20 evaluations (`runs/h4r-*/eval_rdev.json`), each against its `eval_sha256` in the committed `outputs/h4r/manifest.json` (sha256 `f4731356…`).
- **Frozen code, read-only.** `src/apexzero/fields` (unchanged since `receptive-fields-prereg-v1`) and `src/apexzero/interp` (unchanged since `psychophysics-prereg-v2`) are imported, never edited. The new script refuses to run when `src/`, `scripts/fields.py` or `scripts/psychophysics.py` differ from HEAD (the `scripts/locality_exploratory.py` precedent). Nothing is written under `outputs/fields/`, `outputs/psychophysics/` or `outputs/h4r/`, and no frozen stage function that stamps provenance there is called.
- **Bootstrap.** Step 2's machinery as frozen: 95% percentile intervals, 20,000 draws, the `bootstrap` stream (101), restarted for every summary (`measures.by_direction`).
- **Code and outputs.** One new script, `scripts/interp_leftovers.py` (subcommands `in1`–`in4`), with `tests/test_interp_leftovers.py`; results in `outputs/interp-leftovers/`; one findings note, `docs/findings/interp-leftovers.md`, with every number labelled exploratory or descriptive. No file under `src/` is added or changed (the costate freeze hash covers `src/`).
- **Compute.** Seconds to minutes on the Mac, one process.

## 2. IN-1: the anchored-key G with the pre-registered pair weighting and CIs

**Why.** Step 2's design pre-registered an anchored-key version of H2′ (descriptive). It came out empty: `stages.edge_key_measures` skips every row whose `w_is_cell` is false, a filter meant for the six A2 horizon twins, and every anchored key is a W = 400 key with `w_is_cell` false, so all 9,525 anchored rows were dropped ([receptive-fields-step2.md](receptive-fields-step2.md), Section 7.9). Only per-key medians exist (Sections 7.8–7.9). This item computes the summary the design intended (step 2 findings, Section 10, item 2).

**Method.**
- For each agent, a copy of each anchored row with `w_is_cell` set to true is passed to the frozen `stages.edge_key_measures`. The six A2 horizon twins (`keys/lists.json`, `a2.w_twins`) are checked to be absent from the anchored rows, so the filter's purpose is kept. Every other rule is unchanged: non-capped keys only, step 2's exclusion reasons in their order (pipeline failures, invalid c_min, Gate 3, numerical, inertia, Gate 4, κ clipping, argmin off plateau), and the optimum's anchored rows from `opt-fields`.
- The summaries are the frozen `anchored_edges` ones, applied to those measures: `analysis.summarise` for G, T and B. Pair values per (pair, approach), with A1 and tier B as separate approaches and both directions weighted equally; a pair is undefined when more than 3/8 of its listed keys are excluded; the median over geometries (R, θ) of the geometry medians; the cluster-bootstrap CI over geometries; left, right and the asymmetry index; coverage under the 30% rule (an agent above it is descriptive); T's label.
- Structure checked before writing this plan: within each agent, no two anchored keys share (pair, approach, direction, phase), so the frozen pair code applies without change.
- H2′'s band rule (`measures.h2_decision`) is applied to each anchored G as a descriptive label only.
- **Agents:** the 12 finals (G1, G1s1, G1s2, G2, G2s1, G2s2, G3, G3s1, G3s2, S1, S2, S3) and the 8 earlier snapshots (G1 at 73.7M, 109.9M and 134.1M; G2 at 134.1M and 163.8M; G3 at 109.9M, 134.1M and 163.8M).
- **Built-in checks.** (i) The same copy-and-call on the A2 rows reproduces `decisions.json`'s `H2` and `edge_descriptive` summaries for all 35 entries exactly (on A2 rows the copy changes nothing). (ii) Plain per-key medians recomputed from the same rows reproduce the tables of step 2's Sections 7.8 and 7.9. Either failure stops the item for an amendment.

**Expected directions.**
- **E1.1.** For the nine generalist finals, the pair-weighted anchored G lies within ±0.03 of its per-key median (0.73–0.85), every CI's upper end is above 0.5 (no deficit on their own fast-approach states), and each CI overlaps the network's A2 CI.
- **E1.2.** The earlier snapshots' G is about 0.78–0.90 and does not rise with training (step 2's Section 7.8 reading).
- **E1.3.** The specialists exceed the 30% coverage limit and are descriptive (most of their anchored runs crash before the window's end; step 2, Section 7.11).

**Reading.** If E1.1 holds, step 2's sentence "on each agent's own fast-approach states, G is the same range as on the shared A2 states" stands with intervals. A generalist whose anchored CI lies wholly below its A2 CI tracks the edge less on its own fast approaches than on the shared states, which would qualify that sentence.

## 3. IN-2: H2′ on the A2 keys valid for all nine generalists

**Why.** G3 is the only generalist whose H2′ verdict is "lawful". It loses 48 tight-corner keys to "argmin off plateau", where every other generalist's G is lower than its own median. On the 306 keys with no exclusion in any of the nine, plain per-key medians run 0.794–0.848, and G3's 0.840 is not distinctive ([receptive-fields-step2.md](receptive-fields-step2.md), Section 5.1). This item asks, with the pre-registered weighting and CIs, how much of G3's verdict comes from its own exclusion set (step 2 findings, Section 10, item 3). H4R's common-key check (296 keys over its 20 networks) is a different set and question.

**Method.**
- **The common set K:** the primary non-capped A2 keys whose G is valid (no exclusion of any reason from the frozen `edge_key_measures`) in all nine step-1 generalists. |K| is expected to be 306; any other count stops the item for an amendment.
- **Primary.** For each generalist, its frozen key measures with every key outside K marked excluded (reason "not common"). Step 2's pair rule and pair weighting then apply unchanged. Then `analysis.summarise(km, measures.h2_decision)`: pooled median G with its CI and band label, left, right, the asymmetry index; R18's variant rule (`analysis.h2_variant`) over the nine labels. All descriptive.
- **Sensitivities.** (a) Keys outside K are dropped from the pairs' listed keys instead, so each pair's denominator is its common keys. (b) G3s2's failing gate2-trained key is also left out of K (H4R's per-key rule; step 2 made G3s2 descriptive instead).
- **Check.** Plain per-key medians on K reproduce Section 5.1's nine values.

**Expected directions.**
- **E2.1.** On K, G3 is not distinctive: its median lies within the range of the other eight, and either its CI no longer clears 0.8 or at least one other generalist's CI does.
- **E2.2.** Most networks' median G on K is higher than their confirmatory value (the low-G tight corners drop out).
- **E2.3.** No deficit: every CI's upper end is above 0.5.

**Reading.** If another generalist becomes lawful on K, or G3 stops being lawful, G3's verdict reflects its key set more than its tracking, and step 2's "about 80% for a corner-wide shift, in all three variants" is the better summary. If G3 alone stays lawful, its verdict is not an artefact of its exclusions. The confirmatory H2′ decisions stand as decided either way.

## 4. IN-3: step 1's outputs never written up (descriptive)

**Why.** [psychophysics-step1.md](psychophysics-step1.md), "Not covered here", lists outputs of the non-confirmatory run that were never written up. This item writes them up. Nothing is recomputed: every number is read from `decisions_all.json` and tabulated.

**Items.**
- **P2d** (exploratory; the 9 generalists): `n`, `frac_nearer_qss` (share of hairpin pairs whose v_min is nearer the centreline QSS value than the key's), `median_line_excess` (e_vmin^line, agent minus key).
- **P3 ratio** (secondary, descriptive; the 9 generalists): the within-geometry slope of log(d_b / d_b^phys) on log v_b, the median ratio overall and at 310 km/h, and b refitted without the 310 km/h cells.
- **P6d** (exploratory; G1–G3): the median t_R of the large radii (69.8–87.3 m) against the mid radii (28.6–44.7 m), and `large_first`.
- **P8-F** (exploratory; G1–G3): F, reproduced units and units.
- **The per-checkpoint P6 curves** (G1–G3, 16 checkpoints each), as `decisions_all.json` holds them: t_R per tier A radius (P6a); exp(log_std) and the tier G top speed v_top per checkpoint (P6b's covariates); tail e_ve^RA at A1 per checkpoint (P6c). One table per agent and one figure.
- The checkpoint × radius competence matrix behind t_R is not stored in `decisions_all.json`. Recomputing it from the per-checkpoint rollouts is not part of this plan; it would need a dated amendment.

**Expected directions.**
- P2d: `frac_nearer_qss` below 0.5 for most of the nine (v_min nearer the key than the centreline QSS, consistent with P1a's confirmations), and the median line excess within ±b.
- P3 ratio: the slope near 0 (|slope| < 0.2, no time-to-corner signature) and b without the 310 km/h cells within 0.05 of P3's b.
- P6d: `large_first` false for G1–G3 (consistent with P6a's refutation: competence is not ordered by size).
- P8-F: F ≤ 0.2 for G1–G3 (consistent with P8-tail's refutation and P8-bulk's context).
- P6 curves: tail e_ve^RA at A1 below +b at every checkpoint; v_top rising and exp(log_std) falling over training.

**Reading.** Descriptive: it completes step 1's record. A result against these expectations is reported, not acted on.

## 5. IN-4: H4R's lap gap on circuits both arms complete (descriptive)

**Why.** H4R did not analyse lap time ([h4r.md](h4r.md), Sections 8 and 10, item 3). The width test found real-width agents slightly slower where both arms finish ([width-test.md](width-test.md), Section 5.3). GW-4's criterion 3 asks that the median lap gap rise by under about 1 point (follow-through, Section 7.4), and this item gives the copied-width reference for it from same-commit networks.

**Data.** The 20 H4R networks' `eval_rdev.json`. Each circuit's `gap` is the median lap of its completed runs (of 10) over the s-OCP lap (`outputs/ocp/<circuit>.pkl`), minus 1. Only circuits counted as completed (completion > 0.5, H4R's rule) enter.

**Measures.**
1. **Primary (width test, Section 5.3).** Per variant, the circuits that both arms complete (at least one network in each). Per circuit, each arm's median gap over its completing networks; Δ = real width minus baseline, in points. The median Δ over those circuits, and the count of circuits where it is positive.
2. **Paired by seed.** Per same-seed pair (10 pairs), the circuits both networks complete and the mean of the per-circuit differences. The mean over pairs, a descriptive two-sided p from all 1,024 sign flips, and a 95% percentile bootstrap CI over pairs (20,000 draws, seed 0).
3. **GW-4's statistic.** Each network's median gap over its completed circuits; the arm means and their difference; the same restricted to the circuits both arms complete. The within-arm SD of the per-network median gap is reported too, to say how well a 1-point criterion can be read with 5 seeds per arm.
4. **The seed study's pooled statistic.** The median over completed (network, circuit) pairs per arm. It mixes circuit sets, and is labelled so.

**Expected directions.**
- **E4.1.** The real-width networks are slower on common circuits: primary Δ between +0.2 and +0.8 points in both variants (the width test's +0.30 and +0.57), and the paired mean above 0.
- **E4.2.** The rise is below 1 point.

**Reading for GW-4.** If the copied-width arm already costs 1 point or more, criterion 3 is tight for any width model and should be restated before GW-3 is pre-registered. If it costs well under 1 point, the criterion stands as written, with this as its reference. Either way the number is descriptive, from 5 seeds per arm.

## 6. Amendments

Any change to this plan after its commit is a dated amendment appended here, committed before the computation it changes. Each states what changed, why, and which results (if any) had been seen when it was written. Results computed under the earlier version are kept and reported beside the amended ones. An amendment written after a result was seen is labelled post hoc.

(None yet.)
