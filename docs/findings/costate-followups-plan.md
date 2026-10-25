# Idea 2's follow-ups: plan (exploratory; written before any computation, 25 Oct 2026)

**Status.** This plan covers the exploratory follow-ups to Idea 2, the costate study
([findings](costate.md); design [`docs/interp/costate-design.md`](../interp/costate-design.md)
v2.8; decisions `outputs/costate/analyse/decisions.json` under the tag `costate-prereg-v1`). They
are lane A of [`docs/plan/follow-through.md`](../plan/follow-through.md) (Sections 7.2 and 7.3,
and decisions #8, #9 and #16 of Section 10). None of them is pre-registered with a tag, and none
can change a confirmatory decision. This plan is committed on its own, before any of its numbers
is computed, so its questions, computations and expected directions are fixed in advance. Every
later change is a dated amendment at the end (Section 7), never an edit in place.

Nothing below has been computed. The only numbers quoted come from `costate.md`, the design, the
record (`src/apexzero/costate/prereg.json`), `decisions.json` and the follow-through plan.

Section numbers refer to this document; the findings' are marked "findings" and the design's
"design".

## 1. Items

| ID | Item | Trained networks | New solves | Cost (Mac) |
|---|---|---|---|---|
| CS-1 | The ξ tail: where it sits, critic against map, other calibrations | yes (a fresh VJP) | – | < 30 min |
| CS-2 | Descriptives pack | – | – | < 15 min |
| CS-3 | C3-int on s and ξ (network × geometry) | – | – | < 15 min |
| CS-4 | ΔC1, R²_z and the MLP-on-z ceiling | – | – | ~20 min pooled |
| CS-5 | W1 on the Π targets | – | – | < 1 h |
| CS-6 | C1-dev over the 16 snapshots | yes | – | ~1 h |
| CS-7 | Critic-layer and head-unit probes | – | – | ~30 min pooled |
| CS-8 | C2-sw's random-only null; C2-sw residualised; C2 with a decoy that fits geometry 6 | – | – | 20 min – 3 h, plus ~2 h |
| CS-9 | N: per-agent replication of C1, C2 and C2-sw, with K2b on N | yes | ~53 min | 45–60 min pooled, plus the build |
| CS-10 | C1-own on own-run states, then on N | yes | – | ~45 min pooled (+4 min per network) |
| CS-11 | The policy's own width price Π^{in,own} | yes | – | 15–50 min pooled |
| CS-12 | C2 and C2-sw beyond the 9 decision networks (primary only) | – | – | 45–60 min |
| CS-13 | C2-pairs | – | – | ~30 min |
| CS-14 | q_c, the action-projected critic signal, with K2b for q_c | yes | ~53 min | ~1.5 h |
| CS-15 | The FWL check, actor h1 and h2 probes, C2 on D | – | – | ~1 h |
| BC-a0 | Braking and corner targets pilot (targets only) | – | 208 + 150 re-solves | 4–8 CPU-h |

"Trained networks: yes" means the item computes a trained network's activation, value or
gradient that is not already on disk. "–" means it reads only stored arrays (or computes
random-init networks). The costs are the follow-through plan's estimates, made for pools of 5
workers; at the 4 workers this lane uses (R9), a pooled stage takes about 25% longer.

## 2. Rules that bind every item

**R1. Labels.** Every number these items produce is exploratory. None is confirmatory on these
34 networks, and none changes a decision in `decisions.json`. Where an item reuses a frozen
statistic and its band (C1's 0.2 and 0.6, C2's 0.05), the band word is a description ("in the
refute band"), not a decision.

**R2. The frozen code.** Nothing under `src/`, `scripts/costate.py`, `scripts/_bootstrap.py`,
`scripts/fields.py` or `docs/interp/costate-design.md` is edited (the costate freeze and step 2's
tag bind them). The code for these items is a new script, `scripts/costate_exploratory.py`, with
`tests/test_costate_exploratory.py`. It runs in a git worktree whose `src/` is byte-identical to
the tag (#9's recommended route), and before any item it checks:
- that the plan is committed and unchanged since its first commit except for appended amendments
  (Section 7);
- that the tag exists as the annotated object `tag.json` pins, and is an ancestor of HEAD;
- that `src/`, `scripts/costate.py` and `scripts/_bootstrap.py` equal the tag's (git and byte by
  byte), that step 1's and step 2's frozen code equal their tags', and that those tags are the
  objects the record registers;
- that the costate code hash of the tree equals the record's (`d22e670c…`);
- that the design document equals the one the record registers, and the package versions are the
  record's;
- that every module loaded from inside the repository is frozen code or the script itself.

**R3. Inputs.** The script checks the sha256 of every bound file it reads:
- against `decisions.json`'s `inputs` (113 files: the 9 null arrays, and the critic and K3 files
  of all 34 networks and the own, probes and activations files of the 9 decision networks);
- or against the record's `outputs` (23 files: the gate records, `inputs/world.npz`,
  `inputs/summary.json`, the null arrays and the nulls' summary).

`decisions.json` itself is checked against its committed sha256 (`8119f91f…`). Post-tag files
that nothing binds (the activations of the 25 non-decision networks) are read only if their
provenance stamp binds their sha256, names the record's code hash and the tag, and is neither
partial nor under a deviation; their sha256 is recorded in the item's output. Checkpoints are
loaded only through the record's pins (`networks`, and `snapshots` for CS-6), sha256 checked. Each
network's decision mask is rebuilt and its sha256 compared with `decisions.json`'s
`networks.<run>.decision_mask_sha256` wherever one is recorded.

**R4. Trained networks.** They may be computed now, after the tag, and only through
`scripts/costate_exploratory.py`. After R2's and R3's checks the script calls
`apexzero.costate.critic.unlock_trained`, `apexzero.costate.probes.unlock_trained` and
`apexzero.fields.network.unlock_trained` itself, as `scripts/costate.py`'s `_post_tag` does; the
worker pools mirror those locks.

**R5. Braking and corner rows (decision #8, adopted).** No exploratory item compares a critic
gradient, or any other network output, against λ on braking or corner rows. This keeps BC-1's
confirmatory option open. Concretely:
- any statistic that scores a network output (trained or random-init) against a target derived
  from the window problem is scored only on approach- and exit-phase rows. Those targets are λ,
  λ⁻, λ_res, π, the switching functions σ^ped and σ^steer (which equal ∂Q*/∂a at the optimum),
  Π^in, Π^out, Π^slip and Π^μ, and also λ^own, the policy's own costate (the follow-through
  applies #8 to CS-10's braking rows);
- for the K_P components it is scored only on K2b's frozen decision mask
  (`frozen.k2b_excluded`, applied by `states.decision_mask`, with s only where observable): s on
  approach and exit rows, ξ on approach rows, n and vx nowhere. On T's interior that is λ_s 1,643
  and λ_ξ 1,581 rows, as in the confirmatory run;
- a score that ignores these rows is not computed here, not even as a labelled sensitivity;
- targets-only computations (BC-a0, K2b on N, K2b for q_c) may read every phase, because they
  compare no network output with anything;
- network-only descriptives with no target (the accel-channel sensitivity, say) may read any
  row;
- the script enforces the rows: every scoring call passes its rows through one guard, which
  refuses a braking- or corner-phase row and, for a K_P component, any row outside its frozen
  mask.

**R6. Decision #16, adopted.**
- The 6 K0 keys (the anchored finals in `gates/k0.json`'s `summary.not_ok_descriptive`) are
  excluded from N.
- The optional new solves are included: K2b on N (CS-9) and K2b for q_c (CS-14), about 53 min
  each.
- C2-sw's random-only null runs under G1's observation scaling first, and per network only if the
  pooled cost of the other 8 scalings, projected from G1's measured time, is under about 3 h
  (CS-8).
- CS-12 runs the primary records only, specialists included.
- C3-int's n and vx pairs are not computed: under R5 they have no decision row (CS-3).

**R7. The lockbox.** No item reads `apexzero.tracks.corners.R_TEST` or anything under
`outputs/sealed/`. Every state is on T's synthetic A2 corners or on N's (also synthetic). BC-a0's
road continuations come from a grammar fitted on the R-dev circuits only.

**R8. Outputs.**
- Each item writes under `outputs/costate/exploratory/<item>/` (for example
  `outputs/costate/exploratory/cs-1/`). Outputs are local; they are not committed.
- Every output carries a provenance stamp: the label `exploratory`, the item, this plan's path,
  first commit, sha256 and amendment count, HEAD, the paths changed against HEAD, the costate code
  hash, the tag, the sha256 of every input, the environment, the arguments (repository-relative),
  and the pool's workers and threads.
- Outputs are write-once: an item never overwrites or deletes an existing output. A re-run goes
  to a new run name, or the old output is moved aside by hand.
- Under `outputs/costate/` the script writes only into `exploratory/`, never into `inputs/`,
  `gates/`, `nulls/`, `post/`, `logs/` or `analyse/`.
- Committed files carry no absolute local paths and no personal data.
- The results go into one findings note, `docs/findings/costate-followups.md`, with every number
  labelled exploratory and each item's expected directions quoted beside its result.

**R9. Compute.**
- Pooled stages use `apexzero.costate.workers` with 4 workers × 2 threads (another workflow runs
  on this Mac), with BLAS pinned to one thread as in `scripts/costate.py`.
- Heavy jobs run under `nice` (the script renices itself, default 10).
- BC-a0 and the two K2b solves are not run alongside each other or alongside another
  CPU-heavy lane.

**R10. Randomness.** Where an item reuses a frozen function, it uses that function's frozen
stream (`costate_boot`, `costate_mlp`, `costate_c2_pool`, `costate_logo`, `costate_c3`). New
randomness (BC-a0's continuations, K2b's key picks on N) comes from
`numpy.random.default_rng([20261025, item number, unit id])`, so each unit's draws depend on the
unit alone.

**R11. Order.** CS-1 comes first. Its result decides which R²_c the later items co-report (CS-1,
'Rule for later items'). Then CS-2 to CS-5 in one session, CS-10 and CS-11, CS-7 and CS-8, CS-9
(with CS-10 (iii), N's CS-11 and N's CS-13), CS-6, CS-12, then CS-14 (only before the VG work,
if wanted) and CS-15 (on request). BC-a0 runs in its own window, alone on the machine.

## 3. Shared definitions

- **Networks.** The 9 decision networks G1, G1s1, G1s2 (v1), G2, G2s1, G2s2 (v2) and G3, G3s1,
  G3s2 (v3) (findings Section 3.1); the 22 other generalists; the specialists S1–S3. Where an item
  says "per network" without a qualifier, it means the 9 decision networks.
- **T, its interior and its phases** are the confirmatory ones (`inputs/world.npz`, `tab/*`): 9,887
  states over 48 geometries, the interior stratum `tab/interior` (k = 10), and the phases
  `tab/phase` from the optimum's brake onset and apex.
- **D** is T's t = 0 rows (the decision states).
- **N** is defined in CS-9.
- **Decision rows.** For a K_P component k, the interior rows of the network's decision mask
  (R5). "C1's rows" means λ_s's 1,643 and λ_ξ's 1,581 rows.
- **Weights, folds and CIs** are the frozen ones:
  - each key weighs equally, and states weigh equally within a key, renormalised on the scored
    rows;
  - outer folds are partition 0 (`frozen.fold_partition0`);
  - CIs are a 2,000-draw cluster bootstrap over the 48 geometries (`probes.N_BOOT`);
  - targets are winsorised at `frozen.winsor_bounds`;
  - mirror-odd components are scored by R²_wd.
- **The frozen calibration** is C1's: per outer fold, direction intercepts for odd components, a
  leverage-weighted least-squares slope (K_LEV = 3) clamped at ≥ 0, predictions clipped to the
  winsor bounds (`analysis.calibrate`).
- **The pool** is C2's 20 random-init actors (seeds 0–19), with the two-level CI
  (`probes.pool_draws`), and the single-geometry rule as a description (`probes.LOGO_RULE`).
- **Reuse.** Wherever an item computes a quantity the confirmatory run already stored (C1's R²_c,
  C2's statistic, `dV_dwin`), the script recomputes it on the same rows first and requires
  agreement with `decisions.json` to 1e-8, or the item stops. This is a wiring check.

## 4. The items

### CS-1. The ξ tail

**Question.** In every critic, 14–21% of the ξ decision rows carry a large ξ gradient whose sign is
opposite to λ_ξ's on 86–92% of them (findings Section 7.1). Which states are they? Is the tail in
the critic's input gradient ∂V_c/∂z, or in the map J_z that pulls it back to the state? And what
does R²_c look like under other calibrations?

**Computation** (the 9 decision networks; λ_ξ's 1,581 decision rows; λ_s's 1,643 rows where the
calibrations are compared):
1. **The tail**, defined as in findings Section 7.1: rows whose g_ξ (`post/critic/<run>.npz`
   `g[:, 0, 2]`), centred on its key-weighted median within each turn direction, lies more than 3
   robust SDs (key-weighted MAD/0.6745) from 0. The script first reproduces the findings' tail
   counts (G1 286, …, G3s2 294) exactly, or stops.
2. **Where it sits.** Per network, the tail's share of rows, of weight and of ξ's squared error
   (under the frozen calibration), cross-tabulated by each of:
   - geometry (48);
   - phase position: s − s_apex in 25 m bins;
   - any observation channel clipped at |z| = 10 (`post/activations/<run>.npz` `clipped`), split
     into the channels ξ moves (the 160 edge-pose channels, 204, 205) and the others;
   - κ clipping (`inputs.flags`);
   - the preview-spacing kinks, |vx − 25| < 1 m/s or |vx − 83.33| < 1 m/s (d_max = clip(6vx,
     150, 500) m);
   - the tyre-channel kinks: the row's piecewise region of the tyre and slip channels (212–215)
     changes within ±10⁻³ rad of ξ (the region signature `critic._signature` uses for K3's
     stencils).

   An enrichment is the tail's share of a flag divided by the bulk's share of it.
3. **Critic against map.** A fresh VJP of the main V with respect to z at each ξ decision row
   gives ∂V_c/∂z (a trained-network computation, R4). Then g_ξ = σ_R Σ_j (∂V_c/∂z_j) J_z[j, ξ]
   over channel groups:
   - the near half and the far half of the edge-pose preview points (by preview index);
   - lateral (204) and ξ (205);
   - everything else ξ moves.

   For tail and bulk rows separately, report:
   - each group's share of |g_ξ|, and of g_ξ's sign;
   - ‖J_z e_ξ‖, the map's size;
   - ‖∂V_c/∂z‖ on the channels ξ moves, the critic's size.

   Check: the groups sum to the stored g_ξ to 10⁻¹⁰ relative, or the item stops.
4. **Other calibrations**, for s and ξ, beside the frozen one, on the same rows, weights, folds and
   bootstrap:
   - **Huber:** design Section 6's original rule, the per-fold Huber affine fit (tuning 1.345 on
     the MAD scale, `analysis.huber_fit`), slope clamped at ≥ 0, direction intercepts, clipped to
     the bounds;
   - **rank:** per outer fold and turn direction, each held-out row's weighted rank of g within
     the training fold mapped to the training fold's weighted λ quantile at that rank. If the
     training fold's weighted Spearman of (g, λ) is ≤ 0, the prediction is the direction
     intercept, the analogue of the slope clamp.

   Each is scored as R²_c (R²_wd for ξ) with its CI, and on the bulk and the tail separately.
   K4's PR-5 plants (zero, noise, −λ, direction-only), re-made on these rows by the gate's own
   recipe (`gates.pr5`), are put through both calibrations first. Both must score them in
   [−0.02, 0.005] (gate PR-5e's interval), or that calibration is reported as failing its plants
   and not co-reported later.

**Expected (stated before computing).**
- **E1.1.** The tail is concentrated: at least half of its squared error lies in at most 6 of the
  48 geometries, with geometry 38 among them in at least 6 of the 9 networks.
- **E1.2.** The tail is enriched for long previews: its rows have higher vx than the bulk, and
  the kink band around 83.3 m/s is enriched (ratio > 1.5) in at least 6 of 9. Neither observation
  clipping nor the tyre kinks are enriched (ratio < 1.5).
- **E1.3.** The tail comes from the critic's sensitivity to the far preview, not from the map
  alone. The far half of the edge-pose channels carries more than half of |g_ξ| on tail rows, and
  less than half on bulk rows. ‖∂V_c/∂z‖ on the ξ channels is larger on tail rows than on bulk
  rows (ratio > 2), while ‖J_z e_ξ‖ differs less (ratio < 2).
- **E1.4.** Under the Huber and rank calibrations, ξ's R²_c rises toward the bulk's (at least
  −0.05 in all 9) but stays below 0.2 in all 9. s moves by less than 0.03.

**What the outcomes mean.**
- A tail that sits at kinks or clips (against E1.2) is mostly a property of the pullback and the
  observation map, so the refute is about the bulk.
- A tail carried by far-preview channels with a large ∂V_c/∂z (E1.3) means the critics' gradients
  with respect to distant road geometry are large and anti-aligned with λ_ξ.
- If either alternative calibration put ξ above 0.2, the refute would still stand, since the
  calibration was fixed before the tag (findings Section 9). The rule below would then carry the
  alternative into the later items.

**Rule for later items.** If, in at least 5 of the 9 networks, an alternative calibration passes
its plants and raises ξ's R²_c over the frozen one by more than 0.1, every later item that
reports R²_c co-reports it under that calibration beside the frozen one. The frozen one stays the
headline everywhere. This rule is fixed here, before CS-1 runs.

**Cost.** < 30 min; one VJP per row per network.

### CS-2. Descriptives pack

**Question.** Design Section 14 ('Not in the confirmatory run') lists descriptives the analysis
did not compute. Which of them are cheap from stored arrays? The interpretive one is the heads'
gradient reliability ρ_k.

**Computation** (from the critic npz files: 9 decision networks, the 22 others and S1–S3 where the
quantity exists; C1's rows):
1. **Unwinsorised R².** The frozen out-of-fold predictions, scored against unwinsorised λ.
2. **Raw R² and the direction-only floor.** Raw R² takes the SST about the overall weighted mean.
   The floor is the per-direction training-fold mean, used as a prediction and scored by raw R².
3. **Per-head R²_c.** Each of the 5 heads' pullbacks (`g[:, 1 + i]`) through the frozen
   calibration. Generalists only, since S1–S3 have no heads.
4. **Inter-head reliability ρ_k** of each gradient component across the 5 heads
   (`critic.head_reliability`, key weights, on C1's rows), with √ρ_k as a ceiling on any head's
   |corr(g^k, λ_k)| under a signal-plus-independent-noise reading.
5. **The spread over folds:** the SD over the six outer folds of the per-fold R²_c.
6. **C1 on D:** C1's statistic on D's rows within the decision mask.
7. **The accel term:** the share of g_vx from the accel term (`g` against `g_held`), and the
   accel-channel sensitivity |∂V_c/∂accel_y| (`accel_sens`) against lateral tyre utilisation
   (channels 212–213). vx is not scored and no target enters these two, so they may read every
   interior row (R5); they are reported on all interior rows and on C1's rows.

**Expected.**
- **E2.1.** ρ_k ≥ 0.9 for s and ξ in every generalist, since D_grad ≤ 0.048 means an ICC near 1.
  So the ceiling √ρ_k does not bind, and head noise does not explain the low agreement.
- **E2.2.** Every head's R²_c is in the refute band (CI upper end < 0.2), and the heads' R²_c
  agree with the main V's to within 0.1.
- **E2.3.** Unwinsorised R² is within 0.05 of R²_c for s, and lower than R²_c for ξ.
- **E2.4.** On ξ, raw R² lies near the direction-only floor (within 0.1).
- **E2.5.** C1 on D is in the refute band.
- **E2.6.** The accel term carries under 10% of |g_vx| on C1's rows.

**What the outcomes mean.** High ρ_k with low head R²_c means the heads agree with each other on a
gradient that is not λ: determined but wrong, given the trunk. Low ρ_k would put part of the
refute down to head-level indeterminacy.

**Cost.** < 15 min.

### CS-3. C3-int on s and ξ

**Question.** Within a network, across geometries, does that network's specific deviation of the
calibrated critic gradient go with its specific deviation in driving? Double-centring over
networks removes T's λ, so this tests network-specific gradient deviation against behaviour, not
mispricing.

**Rows and pairs.** The record keeps C3-int's pre-signed pairs (`constants.analysis.c3_int_pairs`:
n with G − 1 and with B, vx with the brake-onset residual and with the entry-speed ratio). n and
vx have no decision row, so those pairs are not computed (R5, R6). C3-int's machinery is applied
to s and ξ on their decision rows instead. No direction is pre-signed for them, so each pair is
tested two-sided.

**Computation.**
1. **Networks:** the generalists that have step 2 fields (about 29) and S1–S3 descriptively.
2. **Residuals.** For each network k ∈ {s, ξ} and geometry, the signed calibrated critic residual:
   the mean over the geometry's decision rows of (frozen out-of-fold prediction − λ_k), σ-folded
   for ξ.
3. **Behaviour.** Per network and geometry:
   - step 2's G − 1 and its buffer B (the median over the geometry's keys of step 2's per-key
     values, from step 2's rows, read through `outputs/outputs/fields`, checked against step 2's
     stamps);
   - for step 1's agents, the entry-speed ratio − 1 and the brake-onset residual (step 1's
     per-key measures).
4. **Statistic.** Both tables are double-centred (`analysis.double_centre`). For each of the 8
   pairs (2 components × 4 measures), the correlation over present cells is computed, with a
   two-sided p from C3-int's permutation: geometry labels permuted within each network within the
   R tercile × θ class blocks, 10,000 draws (`constants.analysis.c3_int_perm`), stream
   `costate_c3`. Holm's adjustment is applied over the 8 pairs.

**Expected.**
- **E3.1.** Every |r| < 0.15, and no pair has a Holm-adjusted p ≤ 0.05. This is consistent with
  C3-w and C3-R²c, which rule out a strong association between networks.

**What the outcomes mean.** A sizeable pair would mean that where a network's critic is unusually
off, its line or braking is also unusually off. That is design Section 6's path (a) (critic to
actor) or path (b) (policy to value), which this test cannot separate. A null says nothing about
n and vx, the pairs the theory signs.

**Cost.** < 15 min.

### CS-4. ΔC1, R²_z and the MLP-on-z ceiling

**Question.** Are λ_s and λ_ξ decodable at all from what the networks see? And does the critic's
gradient add anything to the input?

**Computation** (per network, on its own z from the bound activations, C1's rows; ridge probes with
the frozen grid, inner CV and weights):
1. **R²_z:** ridge on z for λ_s and λ_ξ (`probes.ridge_cv`, R²_wd for ξ).
2. **The MLP-on-z ceiling:** a one-hidden-layer MLP of 64 tanh units on z (`probes`'
   `MLP_ALPHAS` and `MLP_MAXITER`, stream `costate_mlp`), out of fold, and the normalised fidelity
   R²_c/R²_MLP (`analysis.ratio_to_ceiling`).
3. **ΔC1** = R²(z ⊕ g) − R²(z) with g the network's 8 pulled-back components
   (`analysis.delta_c1`), with a geometry-bootstrap CI.

**Expected.**
- **E4.1.** R²_z is at least 0.3 for ξ and at least 0.2 for s.
- **E4.2.** The MLP ceiling is above R²_z by at least 0.05, and above 0.4 for ξ.
- **E4.3.** R²_c/R²_MLP ≤ 0.1 in every network.
- **E4.4.** ΔC1 ≤ 0.02 in every network, with a CI that covers 0: the gradient adds nothing
  beyond z.

**What the outcomes mean.** A low ceiling (MLP R² < 0.2) would mean λ_s and λ_ξ are not
recoverable from the observation on these rows, so the refute would say as much about the
observation as about the critic. A high ceiling with a low R²_c places the failure in the
critic's gradient. A positive ΔC1 would mean the gradient holds costate information the input
does not, even though it does not calibrate to λ.

**Cost.** ~20 min pooled.

### CS-5. W1 on the Π targets

**Question.** Are the wall, slip-limit and grip prices Π^in, Π^out, Π^slip and Π^μ linearly
accessible in the networks? Design Section 6's W1 asked for their probe R² and Spearman; only the
critic's width derivative against Π^in was computed.

**Rows.** K2b was never measured for the Π targets. They are window-problem targets, so R5
applies, and only the approach- and exit-phase interior rows are scored. That is 1,581 + 72 =
1,653 rows: λ_s's 1,643 without its observability rule, since Π is not s. BC-a0's re-solves
report |ΔΠ|/SD_T on the same keys beside the result, descriptively.

**Computation** (per network; targets winsorised at their frozen bounds; even under mirroring, so
raw R²):
1. Ridge probes on z alone, on z ⊕ actor h3 and on z ⊕ critic h3.
2. The conditional gains beyond z. For the actor's h3, trained minus the pool's median (the pool's
   h3 is stored for the 9 decision networks); for the critic's h3, against the 20 random-init
   critics' h3, computed here.
3. Out-of-fold R² and Spearman with geometry-bootstrap CIs.
4. W1's critic-against-Π^in calibrated R² on the same rows, for reference.

**Expected.**
- **E5.1.** Π^in and Π^out are decodable from z (R² ≥ 0.3), because they track the corner ahead.
- **E5.2.** Neither hidden layer adds more than 0.05 beyond z for any Π target, as C2 found for λ.
- **E5.3.** Π^slip and Π^μ are less decodable than Π^in.

**What the outcomes mean.** A hidden-layer gain above 0.05 with its CI above 0 would be the first
sign that a layer holds a shadow price beyond the input. Gains near 0 extend C2's result from λ to
the road and grip prices.

**Cost.** < 1 h.

### CS-6. C1-dev over the 16 snapshots

**Question.** Does the critic's gradient agreement with λ change with training?

**Computation.**
1. For each decision network, each of its 16 pinned snapshots (the record's `snapshots`, sha256
   checked; 10M to 200M timesteps) is loaded.
2. Its main V's on-manifold pullback is computed at T (R4).
3. C1's statistic without nulls (`c1_bare`: the frozen calibration, the snapshot's own
   observability of s, K2b's frozen phase exclusions, C1's rows) and the per-component Spearman
   are computed.
4. Per network, the Spearman ρ of R²_c with log timesteps is computed (`analysis.c1_dev`), and the
   same for each component's Spearman. Per variant, "ρ > 0 in at least 2 of 3 seeds" is reported
   descriptively.
5. Check: the 200M snapshot reproduces the stored `g` to 10⁻¹⁰.

**Expected.**
- **E6.1.** R²_c stays in the refute band at every snapshot (no snapshot's CI upper end reaches
  0.2).
- **E6.2.** ξ's Spearman rises with training: its trend ρ > 0 in at least 2 of 3 seeds in each
  variant.
- **E6.3.** R²_c's own trend is not predicted, because the heavy ξ tail dominates it (CS-1).

**What the outcomes mean.** Rising rank agreement with no rise in calibrated agreement would say
training improves the gradient's ordering but not its shape. No trend at all would say the
gradient's relation to λ is set early.

**Cost.** ~1 h (144 snapshot pullbacks).

### CS-7. Critic-layer and head-unit probes

**Question.** Do the critics hold λ in another layer, in a form their input gradient does not
express?

**Computation** (per network, C1's rows, from the bound activations: `critic_h1`, `critic_h2`,
`critic_h3`, `head_units`):
1. Conditional ridge probes of λ_s and λ_ξ, and of C2's λ_res (the frozen f̂_D, k = 8), with z ⊕ h
   against z, for h ∈ {critic h1, h2, h3, the 5 heads' units concatenated (320)}.
2. Each gain is taken as trained minus the median over the 20 random-init critics' same layer
   (seeds 0–19, under the network's obs_rms; computed here, since random-init networks need no
   unlock).
3. CIs are geometry bootstraps.
4. The actor's h3 is the comparator (C2's records).

**Expected.**
- **E7.1.** No critic layer gains more than 0.05 beyond z on λ or λ_res with its CI above 0.
- **E7.2.** The critic's h3 gains at least as much as the actor's h3 (since V is linear in h3).
- **E7.3.** The head units gain no more than the critic's h3.

**What the outcomes mean.** A critic layer with a gain above 0.05 would mean λ is linearly present
in the critic even though ∇V does not match it, so C1's refute is about the readout, not the
representation. Gains near 0 extend C2's result to the critic.

**Cost.** ~30 min pooled.

### CS-8. C2-sw's random-only null; C2-sw residualised; C2 with a decoy that fits geometry 6

**Rows.** C2-sw's target σ^ped is a window-problem target, so R5 applies. All three parts are
scored on the pedal-switching steps of the approach and exit phases (976 + 1,531 = 2,507 of the
2,911). For C2 they are C2's decision rows. The confirmatory C2-sw used all 2,911, so these numbers
are not on its rows. Whether random-init hosts fall under #8 at all is a question for the lead
(Section 6). If the lead rules them outside it, an amendment may add the null on all 2,911 rows.

**(a) C2-sw's random-only null.**
- *Question.* How far does the trained h3's C2-sw gain over the pool lie above random-init
  variation? C2-sw has no null of its own (findings Section 5.4).
- *Computation.*
  1. As gates' `c2_null`, but with C2-sw's statistic (`analysis.c2_sw_network`). 20 random-init
     hosts (seeds 20–39) are each scored as the network against the pool (seeds 0–19), with the
     two-level CI and the single-geometry rule described.
  2. The hosts are scored under G1's observation scaling first.
  3. The 9 trained networks' C2-sw is recomputed on the same 2,507 rows.
  4. Reported: the null's mean, SD and range, each trained network's share of hosts at or above
     it, and its pool rank.
  5. Per network (each decision network's own scaling) only under R6's 3 h rule.
- *Expected.* **E8a.1.** The null's points centre near or below 0, with an SD of about 0.01–0.02
  (C2's null had SD 0.020 on its rows). The 7 networks that rank first of 21 lie above every host.

**(b) C2-sw with σ^ped residualised on position and speed.**
- *Question.* Is the trained h3's extra C2-sw gain switching-function information, or the
  position-and-speed decoys that σ^ped carries (design 15.10)?
- *Computation.* σ_res = σ^ped − f̂_D(D₁), with f̂_D the frozen decoy (kNN, k = 8, on (s − s_apex,
  vx, log R, θ, w, σ)), cross-fitted by outer fold as C2's λ_res. Then C2-sw's statistic on σ_res,
  trained minus the pool's median, with the two-level CI.
- *Expected.* **E8b.1.** The gain shrinks to at most half of its unresidualised value in at least 6
  of 9 networks, and the trained h3 ranks first of 21 in at most 4 of 9.
- *What the outcomes mean.* Only this variant removes the named confound. If the rank-1 pattern
  survives (first in at least 7 of 9, CI above 0), the actor holds switching-function information
  beyond position and speed.

**(c) C2 with a decoy model that fits geometry 6.**
- *Question.* Geometry 6 holds 64–71% of C2's base squared error because f̂_D's out-of-fold kNN
  fails there (design 17.6). What does C2 look like when the decoy fits it?
- *Computation.* Candidate decoys are tried in this order:
  1. f̂_D with its neighbour search restricted to training rows of the same θ and turn direction;
  2. that restriction with the log R coordinate weighted × 3;
  3. a ridge on a cubic B-spline basis in (s − s_apex, vx) interacted with (log R, θ class, w, σ).

  The first candidate is kept whose geometry-6 share of λ_res's sum of squares on C2's scored rows
  is at most twice that geometry's share of λ's (s 11.8%, ξ 9.0%). The choice reads targets only,
  before any network is scored under it. Then C2's primary statistic (the pool, the two-level CI,
  the single-geometry rule described) is computed with that λ_res and the frozen power floor
  re-reported.
- *Expected.* **E8c.1.** Geometry 6's share of the base's squared error falls below 0.3. C2's
  two-level CIs narrow by at least a third. No point reaches 0.05, so C2 stays without a confirm.

**Cost.** (a) about 20 min for G1's scaling at 4 workers, and about 2.7 h if run per network.
(b) and (c) about 2 h together.

### CS-9. N: per-agent replication at the networks' own states (with CS-10 (iii), N's CS-11 and N's CS-13)

**Question.** Do C1, C2 and C2-sw replicate at the networks' own states? N is the only set where
the critic's gradient, λ* and λ^own exist at the same state the network itself reached.

**Building N.**
1. For each of step 1's 12 finals, its anchored finals (the record's `key_sets` `n:<agent>`,
   393–766 keys per agent), less the 6 K0 keys (R6).
2. One state per key: x_1, with the agent's own a_0 (the key's anchor) as prev_action and the
   accel of x_0 → x_1.
3. The target λ_1 = γ⁻¹ Φ_x(x_1, a*_1)ᵀ lam_g′[8:16] − e_s/10 (design Section 3.1, anchored
   decision), with the pedal fold, by `targets`' anchored route. Also π_1, σ^ped_1, Π_1 and λ⁻_1.
4. Winsorisation at T's frozen bounds.
5. Phases from the key's landmarks and its own plan's brake onset, and the interior flag from the
   key's plan from x_1, with k = 10 and the frozen thresholds.
6. All of N's geometries are among T's 48, so partition 0's folds, the bootstrap over geometries
   and key weights (one state per key) carry over.

**K2b on N** (new solves, about 53 min, R6). K2b's procedure (`gates.k2b_key`'s second corners,
both severities, placed 20 m beyond x_1's preview reach) is run on 26 of N's keys, picked as K2b
picked T's: stratified by R tercile × θ class over the pooled agents, with R10's stream. K2b's rule
gives N's own exclusions by phase. N's decision mask is T's frozen exclusions together with N's
own (the union, the more conservative), with s only where observable. Braking- and corner-phase
rows are never scored (R5).

**Computation** (per agent; the 9 generalists per network, S1–S3 descriptive):
1. The critic's on-manifold pullback at N, the 20 random-init critics' pullbacks, the actor's h3
   and m, and the pool's h3 and m on N's z (R4).
2. **C1 on N:** C1's statistic (frozen calibration, folds and bootstrap) with the best-of-20
   random-critic floor. The value-fit, QSS, barrier and far-geometry nulls are not rebuilt for N.
3. **C2 on N:** λ_res from f̂_D cross-fitted on N's rows, h3 ⊕ z ⊕ m against z ⊕ m, trained
   minus the pool's median, the two-level CI, on the frozen floor {s, ξ}, with N's own residual
   shares reported.
4. **C2-sw on N:** on N's pedal-switching rows in the approach and exit phases.
5. **CS-10 (iii), C1-own on N:** λ^own from x_1 with a_0 over H = W − 1 by the own loop (the
   record's own constants), and the critic's gradient against it.
6. **N's CS-11:** Π^{in,own} at N's states and the critic's width derivative at N (CS-11's
   computation).
7. **N's CS-13:** C2-pairs between N's states and T's plan states of the same cell (CS-13's
   computation).

**Expected.**
- **E9.1.** C1 on N is in the refute band in all 9 (CI upper end < 0.2).
- **E9.2.** C2 on N is inconclusive or in the refute band in all 9.
- **E9.3.** C2-sw on N is in the refute band.
- **E9.4.** K2b on N excludes n and vx on approach, as on T.
- **E9.5.** C1-own on N is in the refute band.

**What the outcomes mean.** C1's R²_c on N materially above T's (above 0.2) would mean the critics'
gradients are better at the states their own policies reach than at the optimum's states. That
would narrow the confirmatory refute to off-policy states. Agreement with T extends the refute to
on-policy states.

**Cost.** 45–60 min pooled, plus the build, plus K2b on N (about 53 min).

### CS-10. C1-own on own-run states (then on N)

**Question.** Does the critic's gradient match its own policy's costate at the states the policy
actually visits? C1-own has about 162 decision rows per component, and its CIs are wide.

**Computation (ii)** (per decision network):
1. The deterministic policy's closed loop from each of D's 477 analysable keys (x_0, a_{−1} = 0)
   for H = 800 steps or to the strip's end (design Section 3.3).
2. λ^own at every state of the run from the own loop's reverse sweep (K3b's trained analogue
   passed for all 9).
3. The critic's on-manifold pullback at the run's states, thinned every 5th step from t = 0 to the
   run's own t_end (as T).
4. Each own-run state gets its phase from the key's landmarks and the optimum's brake onset
   (T's rule, `states.phases`), and an own interior flag: no wall or slip contact of the run from
   t to t + 10, and the run's clearance and slip margin above the frozen thresholds.
5. Rows (R5): λ_s on approach and exit interior rows (s observable), λ_ξ on approach interior rows.
   Runs that terminate are dropped and counted.
6. C1-own's statistic: the frozen calibration, partition 0's folds by the key's geometry, key
   weights, and the bootstrap over geometries.

**Computation (iii)** is CS-9 step 5, on N.

**Expected.**
- **E10.1.** C1-own on own-run states is in the refute band in all 9, on at least 10 times C1-own's
  rows, with CIs at most half as wide.
- **E10.2.** ξ's Spearman is positive (CI lower end > 0) in at least 6 of 9.

**What the outcomes mean.** A score above 0.2 on own-run states, while C1-own refutes on T's
restarts, would mean the critic's gradient tracks its own policy's costate where the policy drives.
The gap would then lie in the optimum-versus-policy terms of design Section 3.3. A refute here
places the gap in the critic's gradient itself.

**Cost.** ~45 min pooled, plus about 4 min per network.

### CS-11. The policy's own width price Π^{in,own}

**Question.** Does the critic value the inside wall the way its own policy does, and does the
policy value it the way the optimum does? Only this comparison can support "the networks do not
value the inside wall the way the optimum does".

**Computation.**
1. Per decision network, first on one network to measure the cost (the follow-through's estimate
   is 10–15 min per network), then the rest.
2. On D (and on N, CS-9 step 6): Π^{in,own} = d/dΔ of the deterministic policy's discounted return
   for a uniform inside widening Δ ahead. The horizon is design Section 3.3's: H = W − t (W at D)
   for the comparison with Π^in, and H = 800 for the comparison with the critic.
3. It is computed as a VJP through the frozen `network.closed_loop` along the
   `perturb.perturbed_arrays` / `strip_tangent` path (design Section 3.4).
4. Runs are kept if they stay alive through the window and keep a positive inside buffer B (step
   2's B, measured on the run).
5. Rows (R5): D's and N's approach-phase interior rows.
6. W1's three comparisons, each as a calibrated R² (C1's calibration rule), a Spearman and a
   σ_R-scaled pooled slope, with geometry-bootstrap CIs:
   - the critic against its own policy: the stored `dV_dwin` at D, or the critic's derivative at
     N, against Π^{in,own};
   - its own policy against the optimum: Π^{in,own} against Π^in;
   - the critic against the optimum, on the same rows, for reference.

**Expected.**
- **E11.1.** The critic agrees better with its own policy's price than with the optimum's
  (calibrated R² critic–own above critic–optimum) in at least 6 of 9.
- **E11.2.** The policy's price is smaller than the optimum's: the pooled slope of Π^{in,own} on
  Π^in is below 1 with its CI below 1, in at least 6 of 9, because the policies keep a buffer.
- **E11.3.** The own-policy–optimum agreement is low (calibrated R² < 0.2).

**What the outcomes mean.**
- E11.1 with E11.3 supports "the networks do not value the inside wall the way the optimum does",
  at the level of the policy. The critic then prices its own policy's wall correctly.
- If the critic disagrees with its own policy too, the critic misprices the wall even relative to
  what its own driving would earn.

**Cost.** 15–50 min pooled.

### CS-12. C2 and C2-sw beyond the 9 decision networks (primary only)

**Question.** Do the cross-network C2 and C2-sw patterns (C2 mostly inconclusive, the C2-sw gain
ranking first of 21) hold across the other 22 generalists and the specialists?

**Computation** (the 22 other generalists and S1–S3, R6):
1. The primary C2 on C2's decision rows: λ_res, h3 ⊕ z ⊕ m against z ⊕ m, trained minus the
   pool's median, the two-level CI, with the single-geometry rule described.
2. The primary C2-sw on the approach and exit pedal-switching rows (R5; CS-8).
3. The networks' activations are not bound, so R3's stamp check applies. They hold 5 random
   controls, so the pool's 20 actors' h3 and m are computed on each network's own z (random-init,
   no unlock).
4. No sensitivity variants (R6).

**Expected.**
- **E12.1.** C2 is inconclusive or in the refute band in at least 22 of 25, and in the confirm band
  in none.
- **E12.2.** C2-sw's trained h3 ranks first of 21 in at least 15 of the 22 generalists, and in at
  most 1 of the 3 specialists (Hockenheim-trained, scored on synthetic corners).

**Cost.** 45–60 min.

### CS-13. C2-pairs

**Question.** At pairs of states with matched decoys but different λ, does the actor's h3 predict
the difference in λ better than z and m do?

**Computation.**
1. `analysis.c2_pairs` on T's decision rows per component: pairs of rows of one geometry and turn
   direction from different keys, with |Δ(s − s_apex)| < 2.5 m and |Δvx| < 1 m/s.
2. The weighted Spearman of the predicted Δλ_res against the true Δλ_res, for C2's full probe
   (h3 ⊕ z ⊕ m) and for its base (z ⊕ m), both out of fold.
3. The difference between the two, with a paired geometry bootstrap.
4. N's pairs follow in CS-9 step 7.

**Expected.**
- **E13.1.** Both Spearmans are positive and modest (< 0.4).
- **E13.2.** Their difference has a CI that covers 0 in at least 7 of 9: no costate information in
  h3 beyond z and m at matched decoys.

**Cost.** ~30 min.

### CS-14. q_c, the action-projected critic signal

**Question.** PPO's actor sees the critic only through advantages, and a gradient error biases it
only along action-reachable directions (design Section 6, path (a)). Is the critic's
action-projected signal right, even though its full gradient is not?

**Computation** (per decision network; only before the VG work, and only if wanted, R11):
1. **q_c.** The derivative with respect to a_t of the 64-step GAE advantage built from the
   critic, A_t = Σ_{k<64} (γλ_GAE)^k δ_{t+k} with δ_τ = r_τ + γσ_R V_c(o_{τ+1}) − σ_R V_c(o_τ)
   (γ 0.995, λ_GAE 0.95). It is taken along the key's plan, with later actions held at the
   optimum's and the key's own Jacobians (jacfwd of `sim.control_step`). It includes the
   prev-action channel (the critic's π̂ and the next step's smoothness term). This is design
   Section 6's q_c written out. It needs the critic's on-manifold pullback at every plan step of
   the window, not only at T's every-5th.
2. **The target** is ∂Q*/∂a_t = ∂r/∂a_t + γ(B_tᵀλ⁻_{t+1} + π_{t+1}). At the optimum this is the
   switching function σ_t (`tab/sigma_steer`, `tab/sigma_pedal`): 0 off the action's bounds, and
   the folded bound multiplier at saturated steps (design Section 6, C2). Design Section 6
   writes the target without the prev-action term γπ_{t+1}; that form is reported beside it.
   Check on every scored row: the key arrays' ∂r/∂a_t + γ(B_tᵀλ⁻_{t+1} + π_{t+1}) equals the
   stored σ_t to 10⁻⁸ relative.
3. **Check.** On windows with no active corridor, slip or path row, the same code with the
   optimum's λ and π in place of the critic's gradient must reproduce σ_t to 10⁻⁶ relative. If it
   does not, the discrepancy is reported and the item stops until an amendment corrects the
   formula.
4. **Rows (R5).** T's interior rows whose whole window t … t + 64 lies in the approach phase, or
   wholly in the exit phase. Only approach- and exit-phase gradients then enter, and no braking or
   corner row is touched.
5. **K2b for q_c** (new solves, about 53 min, R6). K2b's 26 keys and both severities are re-solved,
   and |Δσ_k|/SD_T per phase is taken. An action component whose approach (or exit) p90 exceeds
   0.1 is not scored in that phase.
6. **Score.**
   - **Pedal:** C1's calibration rule, R²_c and Spearman against σ^ped on the scored rows where
     the pedal command is saturated (σ^ped ≠ 0), with geometry-bootstrap CIs.
   - **Steer:** its target is 0 wherever steer is off its bounds, which is almost every row. So
     the item reports the RMS of σ_R-scaled q_c,steer on those rows, beside pedal's, instead of
     an R²_c. A perfect critic would give 0.
7. **Completion.** q_c's association with completion uses C3-w's test with x = each generalist's
   pedal R²_c. It is computed over the 31 generalists only if the 9's run takes under 1 h.
8. **Stop rule.** If fewer than 200 rows survive the window rule and K2b for q_c, CS-14 reports
   "not scorable under #8" and stops.

**Expected.**
- **E14.1.** Pedal's R²_c is above 0.05, and above C1's s and ξ R²_c, in at least 6 of 9, and
  below 0.6 in all 9.
- **E14.2.** Steer's RMS on unsaturated rows is at least 0.3 of pedal's: the critic's steer signal
  is mostly error.
- **E14.3.** The association with completion is not strong (C3-w's 95% upper bound < 0.3).

**What the outcomes mean.** High agreement would mean the critic is right where PPO's actor reads it
(the action-reachable directions) despite C1's refute. That would weaken the case for a
value-gradient term (#5). Low agreement, or a large steer signal where the optimum's is 0, means
the critic's action signal is off too, which strengthens it.

**Risk.** This item is conceptual and high-risk, because the window rule may leave few rows.

**Cost.** ~1.5 h, plus about 53 min for K2b for q_c.

### CS-15. The FWL check, actor h1 and h2 probes, C2 on D

**Computation** (per decision network; C2's decision rows; low value: machinery and completeness):
1. **FWL:** design Section 4's two-stage check (`probes.fwl` at the full model's penalties),
   against the one-stage nested ridge. The gain difference is reported.
2. **C2 with h1 or h2** in place of h3, against the pool's h1 or h2, computed here (random-init).
3. **C2 on D's rows alone**, where the decision mask leaves enough rows for the inner CV; otherwise
   "not testable".

**Expected.**
- **E15.1.** FWL and one-stage gains agree to within 0.01.
- **E15.2.** h1's and h2's gains are no larger than h3's.
- **E15.3.** C2 on D is not testable or inconclusive.

**Cost.** ~1 h.

### BC-a0. Braking and corner targets pilot (targets only)

**Question.** In the braking and corner phases, how much does realistic road beyond the preview
move λ? This decides BC-1's target route. 86% of λ_s's variance lies in corner states, and K2b's
second corner excluded every K_P component there.

**What it reads.** Targets only: re-solved window problems on K2b's keys. No network output, of
any network, is computed or read, so R5 does not restrict its rows.

**Continuations.**
- The grammar is the v1 training grammar, `tracks.generator.fit_grammar` on the R-dev circuits as
  `scripts/train_dr.py` builds it (the generator refuses lockbox circuits itself).
- Eight continuations j = 1…8 are drawn with R10's stream: each is the curvature of a generated
  strip from the start of its first corner token onward.
- The key's own widths are kept, so only curvature changes, as in K2b.
- Each continuation is re-solved from the stored optimum by K2b's amplitude continuation and
  re-centring (`gates.k2b_solve`).

**Computation.**
1. **Per-key placement** (26 keys × 8 = 208 re-solves). On each of K2b's 26 keys (`gates/k2b.json`
   `keys`), continuation j's first corner starts at s_next = max(s2, s_c1 + g_j). Here s2 is K2b's
   point, 20 m beyond the furthest preview reach of the key's T states, and g_j is the grammar's
   gap draw. So no T state of the key sees any of it.
2. **Per-state placement** (50 states × 3 = 150 re-solves). Two braking- or corner-phase interior
   states are picked on 25 of the 26 keys, stratified by phase. For each, continuations 1–3 start
   20 m beyond that state's own preview reach, and Δλ is read at that state only.
3. **Statistics.**
   - Per row and component, Δλ_k = λ_k(continuation) − λ_k(the stored straight continuation).
   - The unseen-road share per phase p and component k:
     U_{k,p} = Σ_rows w · mean_j(Δλ_k²) / var_{T,p}(λ_k), with var_{T,p} the key-weighted variance
     over T's interior rows of phase p. Its CI is a bootstrap over keys.
   - Beside it, K2b's statistic: the p90 of |Δλ_k|/SD_T.
   - All 8 state components are reported; K_P decides the route.
   - Also |ΔΠ|/SD_T for CS-5 and |Δσ|/SD_T for CS-14, descriptively.
4. Failed re-solves are counted and reported. A key or state with more than half its re-solves
   failed is dropped from the shares and listed.

**Expected.**
- **Ea0.1.** In the corner phase, U is large (above 0.1) for s and vx.
- **Ea0.2.** In the braking phase, U is smaller than in the corner phase for every K_P component.
- **Ea0.3.** The per-state shares exceed the per-key shares.

**What the outcomes mean** (a recommendation to the lead under #6, not a decision). The route is
read from the per-state shares, the estimand BC-1 would need, with the per-key shares as a lower
bound:
- if U's CI upper end is at most 0.1 for a component and phase, BC-1 can be pre-registered on that
  component and phase against the existing λ (option (i)), with that upper end as the stated
  bound;
- if it is above 0.1, that component and phase need the preview-matched average target (option
  (ii)): 2,298 interior states × M draws × about 60–75 s each, that is 150–380 CPU-h.

**Cost.** About 7 CPU-h (358 re-solves at about 1.2 min each, K2b's measured rate). That is about 2
h of wall time at 4 workers, run alone on the machine.

## 5. Order, dependencies and budget

| Order | Items | Needs | Wall time (est., 4 workers) |
|---|---|---|---|
| 1 | CS-1 | this plan, the scaffold | < 30 min |
| 2 | CS-2, CS-3, CS-4, CS-5 | CS-1 (its rule) | ~2 h |
| 3 | CS-10 (ii), CS-11 on D | CS-1 | ~1.5–2 h |
| 4 | CS-7, CS-8 | – | ~1–5 h (CS-8 (a) per network only under R6) |
| 5 | CS-9 with CS-10 (iii), N's CS-11 and N's CS-13 | K2b on N | ~2.5–3 h |
| 6 | CS-6, CS-12, CS-13 on T | – | ~2.5 h |
| 7 | CS-14 (if wanted, before VG), CS-15 (on request) | K2b for q_c | ~3.5 h |
| – | BC-a0 | its own window, alone | ~2 h (7 CPU-h) |

In total that is roughly 15–20 h of Mac wall time over several sessions, with 3 sets of new solves
(K2b on N, K2b for q_c, BC-a0). The follow-through's 10–14 h covered CS-0 to CS-13 at 5 workers;
this adds CS-14, CS-15, BC-a0 and the slower 4-worker pools.

## 6. Open questions for the lead (none blocks an item)

- **Random-init hosts under #8.** R5 reads #8 strictly: CS-8's null hosts are network outputs, so
  they too are scored on approach and exit rows. If the lead rules that random-init networks carry
  no information about BC-1's subject, an amendment may add C2-sw's null on all 2,911 confirmatory
  rows, the rows its trained statistic was decided on.
- **Π under #8.** R5 treats the Π targets as window-problem targets. W1 (confirmatory) already
  scored the critic against Π^in on the whole interior, braking and corner rows included.

## 7. Amendments

The rule:
- This file is append-only after its first commit. Any change to a question, computation, row
  set, expected direction or cost goes here as a new subsection, `### Amendment N (YYYY-MM-DD)`.
- Each amendment states what changed, why, and whether any result of the affected item had been
  computed or seen before it.
- An item computed before an amendment keeps its earlier numbers, labelled with the plan version
  (the sha256 and amendment count in their stamps).
- The script refuses to run when the text above this section, or any earlier amendment, differs
  from its committed form.

Amendments are appended below this line.

### Amendment 1 (2026-10-25)

**What changed.** CS-1's operational details, where Section 4's text leaves a choice, and two
descriptive companions to its tyre-kink flag. No question, row set or expected direction
changes. **Why.** To fix these choices before any CS-1 number exists. **Computed or seen before
it.** No CS-1 number. Only the frozen code and the bound files' layout were read, and the
scaffold's `cs-1 --check` ran, which computes nothing.

1. **The tail** is findings Section 7.1's. On λ_ξ's decision rows, it is the rows where
   `analysis.leverage_weights` < 1 for g_ξ centred on its key-weighted median within each turn
   direction (`analysis._robust_centre`). The weights are C1's `w_in`, the key weights
   renormalised on the interior. Step 1's check also covers the findings table's other columns,
   at their printed rounding: the fold-mean slope, the tail's weight share and share of ξ's
   squared error, bulk R²_c, the sign-opposite shares (on rows) and the variance ratio. Any
   difference stops the item.
2. **Shares and enrichment** (step 2). The weights are `w_in`. ξ's squared error is
   w_in (pred − λ_ξ)² under the frozen calibration. For a flag F, the enrichment is
   P(F | tail) / P(F | bulk). The headline is on rows, and the same ratio on weights is
   co-reported. It is null when both shares are 0, and "inf" when only the bulk's is.
   - **Geometry:** `states.geometry_index` (index 38 is (44.7 m, 180°), `frozen.geometries[38]`).
     E1.1's set is the smallest set of geometries, taken in decreasing order of the tail's
     squared error, that holds at least half of it.
   - **Phase position:** bins ⌊(s − s_apex)/25 m⌋ of `tab/ds_apex`.
   - **Clipping:** four flags. Any channel clipped; a channel of the plan's ξ set clipped (the 160
     edge-pose channels, 204, 205); a channel outside that set clipped; κ clipping (`inputs.flags`
     under the network's obs_rms).
   - **Preview kinks:** |vx − 25| < 1, |vx − 500/6| < 1, and their union.
   - **Tyre kinks, as written:** the region signature of channels 212–215 (util_f ≥ 2,
     util_r ≥ 2, vx ≤ 1, the parts of `critic._signature` for those channels) at x_t ± 10⁻³ e_ξ,
     compared with the one at x_t. Those channels read only (vx, vy, r, δ, pedal) and the context
     (`dynamics.tyre_forces`, `racing.observe`), so the flag cannot be set. It is computed and
     reported as it comes out, with its enrichment undefined if it is never set.
   - **Tyre companions (descriptive):**
     - the row's own tyre-clip region (channel 212 or 213 at 1, that is util ≥ 2, or vx ≤ 1, read
       from the stored raw observation), as a flag;
     - the key-weighted median of max(util_f, util_r) (channels 212–213 plus 1, capped at 2 by the
       observation) on the tail and the bulk.

     They carry no expected direction beyond E1.2's "not enriched".
   - **vx:** E1.2's "higher vx" compares key-weighted medians. Means are co-reported.
3. **Critic against map** (step 3).
   - **The VJP.** The fresh VJP is `critic._grad_jit` of the main V on each key's T rows, in
     `critic._batched`'s chunks, as the critic stage computed it.
   - **The groups.** Let w_j = (∂V_c/∂z_j) when channel j is not clipped, and 0 when it is. A
     group's contribution is c_k = σ_R Σ_{j∈k} w_j J_z[j, ξ]. The groups are:
     - the near half of the edge-pose channels (preview points 1–20);
     - the far half (points 21–40);
     - channels 204 and 205;
     - the other channels with J_z[j, ξ] ≠ 0 on some row;
     - the accel term, σ_R (J_accᵀ w)ᵀ M_x[:, ξ] on rows with t ≥ 1 (of `ObsJac.frenet_leak`'s
       size).

     With the accel term included, the groups sum to the pullback's g_ξ.
   - **The check.** The maximum over rows of |Σ_k c_k − g_ξ| / max(|g_ξ|, 10⁻¹² Σ_k |c_k|) must be
     ≤ 10⁻¹⁰ against the stored g. The fresh `critic.pullback` must also equal the stored g on all
     8 components to the same relative tolerance.
   - **On tail and bulk rows separately:**
     - a group's share of |g_ξ| is Σ w c_k sgn(g_ξ) / Σ w |g_ξ|. It is signed: the shares sum to 1,
       and a group that pulls against the total has a negative share. The unsigned share
       Σ w |c_k| / Σ w Σ_j |c_j| is co-reported;
     - its share of g_ξ's sign is the weighted share of rows with sgn(c_k) = sgn(g_ξ). The share
       with sgn(c_k) = −sgn(λ_ξ) is reported beside it;
     - ‖J_z e_ξ‖ is the norm of J_z[:, ξ] over the unclipped channels. The critic's size is
       σ_R ‖∂V_c/∂z‖ over the unclipped channels with J_z[j, ξ] ≠ 0 at the row;
     - E1.3's ratios are tail over bulk of their key-weighted medians. Means are co-reported, and
       so is the cosine g_ξ / (critic size × map size).
4. **Calibrations** (step 4). The fit rows are each component's decision rows in the training
   folds, with `w_in`, as for the frozen calibration. Predictions are clipped to
   `frozen.winsor_bounds`.
   - **Huber:**
     - s: `analysis.huber_fit` (c = 1.345, slope clamped at ≥ 0: with a negative slope, its
       location-only fit);
     - ξ: the same IRLS with one intercept per turn direction and a shared slope. Each weighted
       least-squares step is solved within direction. The scale is the weighted MAD of the
       residuals about their weighted median, pooled over both directions, / 0.6745, re-estimated
       every iteration. With a negative slope it is the location-only fit per direction. With a
       single direction it is `analysis.huber_fit` exactly (a test checks this);
     - no leverage weights.
   - **Rank:** per fold, component and turn direction (s too, as written), on the training rows T
     of that direction:
     - if T has fewer than 3 rows, or the weighted Spearman of (g, λ) on T (`probes.wspearman`) is
       ≤ 0 or undefined, the prediction is the weighted mean of λ on T;
     - otherwise a held-out g maps to F = (Σ_T w [g_i < g] + ½ Σ_T w [g_i = g]) / Σ_T w, and then
       to `states.wquantile(λ_T, w_T, F)`.
   - **Scores:**
     - R²_c per component (R²_wd for ξ) and their mean, on the frozen bootstrap counts
       (`probes.cluster_counts`, stream costate_boot), with 95% percentile CIs;
     - before the alternatives are scored, the frozen calibration must reproduce `decisions.json`
       to 10⁻⁸ (`r2c`, `stat`, `ci`, and the per-fold `a` and `b`);
     - bulk and tail: ξ on its tail and bulk rows; s on its decision rows inside and outside the
       ξ tail (the exit rows are outside it).
   - **Plants** (on each network's decision rows):
     - zero;
     - noise: the second 9,887 × 4 standard-normal draw of `gates.k4_rng(7)`, as `gates.pr5`
       draws it, its s and ξ columns;
     - −λ (winsorised);
     - direction-only (g = σ).

     A calibration passes its plants in a network when all four score R²_c in [−0.02, 0.005] on
     both s and ξ. The frozen calibration is put through them too, for reference.
5. **Reading the expected directions.**
   - A condition the plan states without a count of networks is read as holding in at least 6 of
     the 9 (as E1.1 and E1.2 count).
   - Every condition's per-network results and count are reported.
   - E1.1 holds when, in at least 6 of 9 networks, the half-error set has at most 6 geometries and
     includes geometry 38. Each part's count is reported too.
   - E1.2's clipping part holds when each of the four clipping flags has a row enrichment below
     1.5 in at least 6 of 9. Its tyre part is read as computed.
   - E1.4 reads point estimates, with CI upper ends co-reported. Per calibration: ξ's R²_c ≥ −0.05
     and < 0.2 in all 9, and |ΔR²_c,s| < 0.03 in all 9.
6. **The rule for later items** is evaluated per alternative calibration on point estimates. It
   fires when, in at least 5 of 9 networks, the calibration passes its plants and its ξ R²_c
   exceeds the frozen one's by more than 0.1.
7. **Outputs.** `outputs/costate/exploratory/cs-1/summary.json` and `rows.npz` (per row: the tail
   flag, the flags, the contributions, the norms, and the three calibrations' predictions). A copy
   of the summary goes to `docs/findings/costate-followups/cs-1-summary.json`.

### Amendment 2 (2026-10-25)

**What changed.** CS-2 to CS-5's operational details, where Section 4's text leaves a choice;
two parts that cannot be done as written (CS-3's step 1 measures have no bound sha256, and
CS-5's |ΔΠ|/SD_T needs BC-a0, which has not run); CS-5 runs in a worker pool. No question, row
set or expected direction changes. **Why.** To fix these choices before any CS-2 to CS-5 number
exists. **Computed or seen before it.** No CS-2 to CS-5 number. Read: the frozen code, the
layouts of the bound files and of step 1's and step 2's outputs, and row counts that involve no
network output: D's 477 rows are all interior and approach-phase; all 477 of T's keys have a
step 1 deterministic run at their (cell, phase), 6 of them horizon twins (`w_is_cell` false);
step 2's rows hold all 477 keys, 48 of them on capped stimuli. CS-1's rule for later items did
not fire, so every R²_c below is the frozen calibration's only.

1. **Shared.**
   - **Networks.** The record's 34 pins: the 9 decision networks, the 22 other generalists
     (`analysis.GENERALISTS` holds all 31) and S1–S3. Each checkpoint is loaded through its pin
     (`scripts/costate.py`'s `load_agent`) for its obs_rms and PPOConfig only. CS-2 to CS-5
     evaluate no trained network and do not unlock; CS-5's random-init critics need no unlock.
   - **The frame.** Per network, `inputs.flags` under its obs_rms, `c1_frame` and its decision
     mask (sha256 against `decisions.json` for the 9). The frozen C1 is recomputed first
     (`analysis.c1`, no nulls, no slope CIs, no leave-one-geometry-out sets) and compared with
     `decisions.json`'s `networks` or `descriptive_networks` record (r2c of s and ξ, stat, ci,
     the per-fold a and b of s and ξ) to 10⁻⁸, or the item stops. Its calibrated predictions
     are "the frozen out-of-fold predictions" below.
   - **Rows, weights, CIs.** C1's rows of component k are `interior & valid[:, k]`, through
     `guard_rows`. CS-2, CS-3 and CS-4 use C1's weights `w_in`; CS-5 the key weights
     renormalised on its rows (`states.stratum_weights`). CIs are 95% percentile intervals on
     the frozen counts (`probes.cluster_counts(48)`, stream costate_boot).
   - **Reading the expected directions** (as amendment 1 item 5): a condition without a count
     holds in at least 6 of the 9 decision networks; "every" and "all" mean every network or
     head the condition names; point estimates unless a CI is named. Counts over the other
     networks are reported beside, descriptively.
   - **Outputs.** `outputs/costate/exploratory/<item>/summary.json` per item (CS-3 also
     `tables.npz`, CS-4 and CS-5 `preds.npz`); a copy of each summary under
     `docs/findings/costate-followups/`.

2. **CS-2.**
   - **Networks.** All 34 for steps 1, 2, 5, 6 and 7; the 31 generalists for steps 3 and 4.
   - **Step 1.** The frozen out-of-fold predictions (clipped to the bounds) against T's
     unwinsorised `lam_s` and `lam_xi`, R² (R²_wd for ξ) with its CI.
   - **Step 2.** Raw R²: the frozen predictions against winsorised λ, SST about the overall
     weighted mean (`analysis.boot_r2` with `odd` false). The floor: per outer fold and turn
     direction, the `w_in`-weighted mean of winsorised λ over the training folds' C1 rows of the
     component (the calibration's fit rows), as a prediction, scored by raw R² on C1's rows.
     Both for s and ξ, with CIs.
   - **Step 3.** `analysis.c1` on each head's pullback `g[:, 1 + i]` with everything else as
     the frozen C1's. Per head: r2c of s and ξ, their mean (C1's statistic) and its CI. E2.2
     reads the statistic: its CI upper end below 0.2, and within 0.1 of the main V's statistic.
   - **Step 4.** ρ_k = `critic.head_reliability(g[rows, 1:6, k], w_in[rows])` on C1's rows of
     s and ξ, and √ρ_k. Beside it: ξ's within-direction ρ (the direction-weight-weighted mean of
     the two directions' ρ); each head's and the main V's weighted Pearson |corr(g_k, λ_k)| on
     the same rows (within direction for ξ, the same mean), and whether any head's exceeds √ρ_k;
     and ρ of all 8 components on the whole interior (network-only, no target, R5). Wiring:
     `analysis.head_disagreement_boot`'s point for g_s and g_ξ on the interior equals
     `decisions.json`'s C1-heads D to 10⁻⁸.
   - **Step 5.** Per outer fold, R² (R²_wd for ξ) of the frozen predictions on that fold's C1
     rows; the SD over the 6 folds (population SD, as `robustness_C1`'s), with the minimum and
     maximum, for s, ξ and their per-fold mean.
   - **Step 6.** D's rows are T's t = 0 rows (`d_rows`), all interior and approach-phase.
     `analysis.c1` refitted on D (fit and scored rows D, each component within its mask), with
     the key weights renormalised on D and the frozen counts (D spans all 48 geometries); its
     band by C1's bands (no nulls). Co-reported: the interior-fitted frozen predictions scored
     on D's rows under the same weights.
   - **Step 7.** The accel term's share of g_vx per state, |g_vx − g_held,vx| / (|g_held,vx| +
     |g_vx − g_held,vx|) (`critic.Pullback.accel_share`'s formula, main V). Rows: the interior;
     C1's rows (the union of s's and ξ's); and each with t ≥ 1 (reset rows carry no accel term).
     Per set, under the key weights renormalised on it: the weighted mean (E2.6 reads it on C1's
     rows), the weighted median and the aggregate Σw|term| / Σw(|g_held| + |term|). The
     accel-channel sensitivity is |`accel_sens[:, 0, 1]`| (σ_R ∂V_c/∂a_y, per m/s²), against
     util_max = max(channel 212, channel 213) + 1 of the stored raw observation (CS-1's
     companion): the weighted Spearman, and the weighted median of |sensitivity| in util_max
     bins [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1) and [1, 2], on the same row sets.

3. **CS-3.**
   - **Residuals** for all 34 networks: per (network, geometry), the `w_in`-weighted mean over
     the geometry's C1 rows of the component of (frozen prediction − winsorised λ_k), times σ
     for ξ; missing where the geometry has no such row.
   - **Step 2's G − 1 and B.** The rows file is read through `outputs/outputs/fields`
     (`scripts/costate.py`'s `fields_out`). Its sha256 must equal the record's
     `c3w_inputs.<run>.overlap_sha256` and its provenance stamp's, `opt-fields/rows.jsonl`'s
     must equal its stamp's, and every row's `ckpt_sha256` must equal the network's pin. Per key,
     `fields.stages.edge_key_measures(rows, opt, "G")` and `(…, "B")`: step 2's own exclusions,
     with horizon twins and capped stimuli left out. Keys are matched to T's by key_id. Per
     (network, geometry): the median over its keys with a finite value; G − 1 is that median of
     G minus 1.
   - **Step 1's measures.** `outputs/outputs/psychophysics/analysis/runs/<agent>_<t>.npz`, t the
     pin's timesteps; its `timesteps` column must equal the pin's. **Cannot be done as written:**
     step 1 binds no content hash for these tables, so their sha256 is recorded, not checked.
     Per T key with `w_is_cell`: the `det` run at (cell_id, key_phase), whose stim and W must
     equal the key's. The entry-speed ratio − 1 is its `e_ve`. The brake-onset residual is
     s_b − s_b* = −`dd_b` (`dd_b` = d_b − d_b*, s_b = s_c0 − d_b), so earlier braking is
     negative. Finite values only (a crashed run keeps a finite value, and is counted). Per
     (network, geometry): the median over its keys.
   - **Networks.** Per measure, the generalists that have it: G − 1 and B, the 29 whose step 2
     rows the record binds; the entry speed and the brake onset, step 1's 9 generalist finals.
     Descriptive: step 1's measures with S1–S3 added (they have no step 2 rows).
   - **The statistic.** Per pair (s, ξ × the 4 measures): both tables restricted to the pair's
     networks and double-centred (`analysis.double_centre`); the Pearson correlation over the
     cells both hold (`analysis._corr_present`).
   - **The permutation.** 10,000 draws (`constants.analysis.c3_int_perm`), stream costate_c3
     sub-stream 1 (`analysis.c3_int`'s). Each draw permutes the geometry labels within each
     network within the R tercile × θ class blocks (`states.r_tercile` and `theta_class` of T's
     48 geometries), network by network as `c3_int` draws them, over the full list (the 31
     generalists in `analysis.GENERALISTS` order, then S1–S3). The same draw serves every
     measure, and the behaviour table is double-centred again in each. The p is two-sided:
     (1 + #{|r_null| ≥ |r| − 10⁻¹²}) / (1 + draws). Holm's adjustment over the 8 primary pairs;
     none for the descriptive ones.
   - **E3.1** holds when every primary |r| < 0.15 and no Holm-adjusted p ≤ 0.05.

4. **CS-4** (the 9 decision networks; per component, its C1 rows).
   - **Inputs.** z = clip(the activations' z, ±10), the input the networks see (as
     `probes_of`); winsorised λ; `w_in`; partition 0's folds subset to the rows.
   - **R²_z** by `probes.ridge_cv` on z; **R²(z ⊕ g)** with g the main V's 8 pulled-back
     components (`post/critic` `g[:, 0]`) as the hidden block (H_GRID: ∞ nests the base). Both
     scored by `probes.probe_r2` (R²_wd for ξ). `analysis.delta_c1` on the same rows must agree
     with them to 10⁻¹², or the item stops.
   - **The MLP ceiling** by `probes.mlp_cv` on z. One component per call, so both components'
     seeds are `mlp_rng(0, f, 0, j)`.
   - **CIs** for R²_z, R²_MLP and R²(z ⊕ g), and the paired CI of ΔC1, on the frozen counts.
   - **The ratio** `analysis.ratio_to_ceiling(R²_c, R²_MLP)` per component, R²_c the frozen one
     (recomputed and checked, item 1).
   - **Pooled:** one unit per (network, component), 18 units on 4 workers.
   - **Readings.** E4.1 and E4.2 in at least 6 of 9, each component's condition separately;
     E4.3 in all 9, both components; E4.4 in all 9, both components (point ≤ 0.02 and the CI
     covering 0).

5. **CS-5** (the 9 decision networks).
   - **Rows.** T's interior rows of the approach and exit phases (1,581 + 72 = 1,653, the same
     for every network), through `guard_rows`. Targets: Π^in, Π^out, Π^slip and Π^μ winsorised
     at `frozen.winsor_bounds`, raw R² (even).
   - **Probes.** `probes.ridge_cv` on z (clipped, as CS-4), on z ⊕ the actor's h3 and on z ⊕
     the critic's h3, the four targets in one call (penalties per target).
   - **Controls.** The actor's h3 against the 20 pool actors' stored h3 (`random0_h3` to
     `random19_h3`); the critic's h3 against 20 random-init critics computed here
     (`probes.random_critic(cfg, s)`, s = 0–19, the network's PPOConfig from its pinned
     checkpoint, h3 on the network's z). Wiring: the pool actor of seed 0, rebuilt by
     `probes.random_actor(cfg, 0)`, gives the stored `random0_h3` on these rows to 10⁻¹², or the
     item stops.
   - **Gains.** Per target, gain = R²(z ⊕ h3) − R²(z); the statistic is the trained gain minus
     the median of the 20 controls' gains. Its CI is two-level, as C2's: each draw takes the
     frozen counts' row and the frozen `probes.pool_draws(20)` row, and the median over the
     drawn controls. The geometry-only CI and the network's rank among the controls are
     co-reported.
   - **Scores.** Each probe's R² and weighted Spearman (`analysis.spearman_boot`) of its
     predictions against the target, with CIs.
   - **W1 for reference.** `analysis.calibrated_agreement` of `dV_dwin` against Π^in on these
     rows, in the confirmatory W1's arithmetic (raw target, predictions clipped to Π^in's
     bounds), with a winsorised-target companion. The confirmatory W1 itself is not re-scored:
     its rows include braking and corner rows (R5).
   - **|ΔΠ|/SD_T. Cannot be done now:** BC-a0 has not run. It is reported with BC-a0.
   - **Pooled:** one unit per network (9 units on 4 workers); CS-5 joins the pooled items.
   - **Readings.** E5.1: Π^in's and Π^out's R²_z each ≥ 0.3, in at least 6 of 9. E5.2: in all
     9, for both layers and all 4 targets, the statistic ≤ 0.05; a statistic above 0.05 with its
     two-level CI's lower end above 0 is named. E5.3: Π^slip's and Π^μ's R²_z each below
     Π^in's, in at least 6 of 9.

### Amendment 3 (2026-10-25)

**What changed.** CS-10's and CS-11's operational details on D, where Section 4's text leaves a
choice; their pooled units; descriptive companions. CS-10 (iii) and N's CS-11 wait for CS-9 and
are not covered here. No question, row set or expected direction changes. **Why.** To fix these
choices before any CS-10 or CS-11 number exists. **Computed or seen before it.** No CS-10 or
CS-11 number, and no trained network. Read: the frozen code, the layouts of the bound files, and
row counts that involve no network output (D's 477 keys: 360 with W = 400, 117 with W = 800; T's
t_end at D 70–199 steps). The cost was measured on a random-init stand-in (step 2's hybrid
control under G1's obs_rms, crashing early, since every scan runs all its steps anyway): about
0.6 s per key for CS-10's loop, sweep and pullbacks and 0.9 s per key for CS-11's two-horizon
VJP, after a compile of about 10 s per worker.

1. **Shared.**
   - **Networks.** The 9 decision networks, each loaded through its pin (`load_agent`); its
     actor through `network.agent_net` and its critic through `critic.trained_critic`, after
     `unlock` (R4).
   - **The runs.** From each of D's 477 keys: x_0 with a_{−1} = 0 and accel 0 (a reset start,
     as D's rows), hidden μ = 1, s_stop = +∞, on the stimulus's float64 road
     (`network.road`). `own.own_return` over 800 steps gives the run's H (800, or fewer where
     the run reaches the strip's end) and whether it terminated within them. A run that
     terminates within its 800 steps is dropped from both items and counted.
   - **Weights, folds, CIs.** Key weights renormalised on the scored rows
     (`states.stratum_weights`); partition 0's folds and the geometry clusters of a table that
     holds every D key, so all 48 of T's geometries are present (its geometry labels and folds
     must equal the record's `frozen.geometries` and `frozen.fold_partition0`, or the item
     stops); the frozen counts `probes.cluster_counts(48)` (stream costate_boot), 95%
     percentile CIs.
   - **Reading the expected directions** (as amendment 1 item 5): a condition without a count
     holds in at least 6 of 9; "all 9" means all 9; point estimates unless a CI is named. Every
     part of a condition is counted and reported.
   - **Pooled:** one unit per (network, quarter of D's keys in T's key order): 36 units on 4
     workers × 2 threads.
   - **Outputs.** `outputs/costate/exploratory/cs-10/` and `cs-11/`: `summary.json` and
     `rows.npz`; a copy of each summary under `docs/findings/costate-followups/`.

2. **CS-10 (ii).**
   - **Requirement.** Each network's K3b trained analogue passed (`post/own/<f>.k3b.json`'s
     decision and the npz's `k3b_pass`), or the item stops. Value accuracy's pass is read from
     `decisions.json` and passed to `analysis.c1_own` for its status field only.
   - **The sweep.** `own.own_costate(net, road, li, 800, H, reset=True, all_states=True)` on a
     kept run: λ^own_k on the manifold for k ≥ 1 (the maps at the run's own (x_k, x_{k−1},
     clip(a_{k−1}))), held at k = 0 (a reset start), the held forms beside them. λ^own_k is
     the gradient of the return from k to H, so its horizon is H − k (at least about 600 steps
     at the states below). There is no K2 horizon cut.
   - **States.** k = 0, 5, 10, … up to the run's own t_end, the first k with s_k ≥ s_c1 + 50 m
     (`states.t_end`'s rule on the run's states, the key's landmarks), and k < H. A kept run
     that never reaches s_c1 + 50 m keeps its states below H and is counted.
   - **The critic.** `critic.pullback` (main V) at those states, from `critic.Inputs`(x_k,
     x_{k−1}, clip(a_{k−1}), reset at k = 0) and their `critic.obs_jacobians` on the same road:
     the on-manifold g, with the accel-held g_held beside it.
   - **Phases.** `states.phases`(s_k, the key's landmarks from the bank, T's `s_b` at the key's
     D row: the optimum's brake onset). The landmarks must equal T's `s_c0`, `s_apex` and
     `s_c1` columns.
   - **The own interior flag.** At every run state j = 0..H:
     - the clearance to the env's termination boundary, w_left(s_j) − 1 − n_j and
       w_right(s_j) − 1 + n_j (racing.py's off-track rule, the road's width splines);
     - the slip margin α_max − max(|α_f|, |α_r|) (`dynamics.tyre_forces` at x_j, μ = 1);
     - contact: a clearance below the OCP's corridor margin (`ocp.window.WindowSpec.safety`,
       0.03 m, T's `edge_safety`) or a slip margin below its slip margin (`slip_safety`,
       10⁻⁴ rad). The thresholds below imply no contact; it is recorded anyway.

     ok_j means no contact, both clearances ≥ `states.CLEARANCE_MIN` (0.25 m) and the slip
     margin ≥ `states.SLIP_MARGIN_MIN` (0.01 rad). The flag at k is ok over j ∈ [k, min(k + 10,
     H)] (`states._window_all`, k = 10).
   - **The mask.** s's observability by `states.observability` on J_z under the network's
     obs_rms (`ObsJac.jz`); the decision mask by `states.decision_mask` on the own-run table
     with the frozen `k2b_excluded`: s on approach and exit rows, ξ on approach rows.
   - **Rows.** A kept run's states with the own interior flag, finite λ^own and g, in the
     approach or exit phase (the stratum, and the weights' rows); each component's rows are
     these within its mask, through `guard_rows` with its mask.
   - **The statistic.** `analysis.c1_own` as the confirmatory C1-own calls it: λ^own raw (not
     winsorised), no bounds, the frozen calibration, the key weights renormalised on the rows,
     partition 0's folds, the frozen counts. **Companions (descriptive):** the same with λ^own
     winsorised at T's frozen `lam_s` and `lam_xi` bounds and the predictions clipped there;
     the accel-held form (λ^own_held against g_held, as `C1-own_held`); the confirmatory
     C1-own on T's restarts (item 2's wiring) beside each.
   - **Wiring (each stops the item).**
     - The confirmatory C1-own recomputed from `post/own` and `post/critic` on T's restarts
       reproduces `decisions.json`'s `C1-own` (r2c of s and ξ, stat, ci, the per-fold a and b
       of s and ξ) to 10⁻⁸.
     - The pullback at every run's state 0 equals the stored g at the key's D row (main V, 8
       components) to 10⁻¹⁰ of the row's largest |g|.
     - In each unit, on its first kept run, restarts at k = ⌊0.25 H⌋ and ⌊0.6 H⌋
       (`own.own_costate` from x_k with x_{k−1}, clip(a_{k−1}), accel_of(x_k, x_{k−1}),
       count H − k) reproduce the sweep's λ^own_k and λ^own_held,k to `own.RESTART_RTOL`
       (10⁻⁹) of the row's largest |λ|.
   - **Readings.**
     - E10.1 holds when, in all 9 networks, the decision is 'refute' (CI upper end < 0.2), both
       s and ξ have at least 10 times the confirmatory C1-own's rows, and the CI is at most half
       as wide as the confirmatory C1-own's. Each part is counted.
     - E10.2 reads ξ's Spearman CI in the c1_own record.
   - **Descriptives.** Runs kept, terminated and reaching the strip's end; H; rows by phase;
     the own interior's share; the own-run states' s − s_apex and vx against T's.

3. **CS-11 on D.**
   - **Π^{in,own}.** The deterministic policy's discounted return J (`own.own_loop`'s rebuilt
     reward, δ = 0, s_stop = +∞) on the road `perturb.perturbed_arrays`(base, samples, d), d
     the perturbation of the strip's samples (κ, w_left, w_right). One forward pass gives J
     over count = W (H = W − t at D, the comparison with Π^in) and count = H (800 or to the
     strip's end, the comparison with the critic), as two `own_loop` scans. Two reverse sweeps
     give ∂J/∂d. Π^{in,own} is the sum over the inside width samples (w_left on a left corner):
     `inputs.inside_widening`, a uniform widening of every inside sample, as the critic's
     `dV_dwin`. `own.own_loop` is costate's frozen rebuild of `network.closed_loop`; K3b showed
     its trace equals `closed_loop`'s to round-off. The widening reaches J through the policy's
     observation (the edges and the lateral channel); the env's termination test is not
     differentiated (a kept run never terminates).
   - **Kept.**
     - At H = W: no termination within W steps, and B_W > 0, with B_W step 2's buffer
       (`outcomes.c_min_plan` on the run's states 0..W: c_min over [s_c0 − 50, s_c1 + 50],
       invalid unless the run is alive through s_c1 + 50 m).
     - At H: the run kept (shared item) and B_H > 0 on its states 0..H.
   - **Rows.** D's rows (all approach-phase interior), through `guard_rows`. The primary rows
     are those kept at both horizons (the common rows): all three comparisons are made on them.
     Companions: the critic–own comparison on the rows kept at H, the own–optimum one on those
     kept at W.
   - **The comparisons** (`analysis.calibrated_agreement`, C1's calibration, raw targets; W1's
     arithmetic):
     - critic–own: x = the stored `dV_dwin` at D, target Π^{in,own} at H (no bounds: it has no
       frozen ones);
     - own–optimum: x = Π^{in,own} at W, target Π^in at D (Π^in's frozen bounds);
     - critic–optimum: x = `dV_dwin`, target Π^in (Π^in's frozen bounds).

     For each: the calibrated R² with its CI, the weighted Spearman with its CI, and the pooled
     slopes (`analysis.pooled_slopes`, the frozen counts): b_lg (target on x) and b_gl (x on
     target), with CIs. `dV_dwin` is σ_R dV_c/dΔ, and every Π is in reward units per metre, so
     the slopes are σ_R-scaled already.
   - **Wiring (each stops the item).**
     - `dV_dwin` recomputed at every D row (`inputs.width_derivative` on the key's reset row)
       equals the stored one to 10⁻¹⁰ relative.
     - Per key, the VJP loop's J at d = 0 and count H equals `own_return`'s J to 10⁻¹²
       relative.
     - In each unit, on its first kept key, the JVP of J along the uniform inside widening
       equals the VJP's contraction to 10⁻⁹ relative, at both horizons.
     - **Descriptive only:** on the same key, Richardson differences (h = 10⁻³ m and h/2) of
       J_env along the widening (secant mode) at count W, with the error against the AD value.
   - **Readings.**
     - E11.1: critic–own's calibrated R² above critic–optimum's, on the common rows.
     - E11.2: own–optimum's b_gl (Π^{in,own} on Π^in) below 1 with its CI's upper end below 1.
     - E11.3: own–optimum's calibrated R² below 0.2 (point).

     Each needs at least 6 of 9.
   - **Descriptives.** Runs kept at each horizon, and the reasons for drops (terminated,
     window end not reached, B ≤ 0); the weighted medians of Π^{in,own} at both horizons, Π^in
     and `dV_dwin`; the share of rows where Π^{in,own} = 0. `rows.npz` also holds the
     inside-width field ∂J/∂w_in(s_j) at both horizons, as float32.
