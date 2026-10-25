# Idea 2, the costate: on approach-phase interior states, the critics' pulled-back value gradients do not match the window problem's λ_s and λ_ξ (C1 refuted, confirmatory)

Results of 24 Oct 2026. Design: [`docs/interp/costate-design.md`](../interp/costate-design.md) v2.8 (the question in Section 1, targets in Section 3, hypotheses in Section 6, gates in Section 7, the decisions and the gate history in Sections 12–18); its sha256 `53fb72d5…` equals the record's `design_doc_sha256`. Pre-registration record: `src/apexzero/costate/prereg.json`, committed alone in `11dc9bc`. Tag: `costate-prereg-v1`, annotated (object `d1f00c93…`), on `11dc9bc`, on origin as the same object. Tag pin: `outputs/costate/tag.json`, committed in `b774472`. Confirmatory decisions: `outputs/costate/analyse/decisions.json` (its stamp's content_sha256 `4c081c42f213c60a066bd52925ec3730cf9cb40cf63fc056a82590bc13c92f05`; file sha256 `8119f91f…`), committed in `9b25cc2`. Code hash `d22e670c…` (src/, `scripts/costate.py`, `scripts/_bootstrap.py`), the record's. Background: [psychophysics-step1.md](psychophysics-step1.md) (step 1), [receptive-fields-step2.md](receptive-fields-step2.md) (step 2), [locality.md](locality.md) and [h4r.md](h4r.md).

Every number below was read from, or recomputed from, the file named next to it. Under `outputs/costate/` only `analyse/decisions.json` and `tag.json` are in git. The pre-tag gate records, the inputs, the nulls and the post-tag stage outputs are local. The record binds 23 files by sha256 (`prereg.json`, `outputs`: the 11 gate records, the 2 input files, the 9 null arrays and the nulls' summary). `decisions.json` binds 113 (`inputs`): the 9 null arrays again, and the 104 post-tag outputs the analysis reads, namely the critic npz and K3 files of all 34 networks, and the own, probes and activations files of the 9 decision networks. That is 127 distinct files (the 9 null arrays are bound by both). Not every post-tag output is bound: of the 206 files under `post/`, the activations of the 25 non-decision networks and every `*.npz.provenance.json` sidecar are not. Checked here: all 127 bound files on disk match. Shorthand: `networks.<run>` is `decisions.json`'s `networks["<run>"]`, `descriptive_networks.<run>` likewise; component lists in the record are ordered (s, n, ξ, vx). Section numbers refer to this document; the design's are marked "design". Labels used throughout:

- **Confirmatory**: the primary family, C1, C2 and C3-w (design Sections 6 and 12), decided once by `scripts/costate.py analyse --confirmatory` and read from `decisions.json`.
- **Secondary**: the named secondaries C2-sw and C3-R²c, and the pre-registered secondaries C1-own, C1-pattern, C1-heads, C1-limit, C1-braking and W1 (design Section 14, 'Labels'), computed in the same run. W1 has no decision band.
- **Descriptive**: everything else in `decisions.json` (strata tables, sensitivity variants, the other 22 generalists, the specialists). These numbers decide nothing.
- **Exploratory (computed here)**: numbers computed for this document from the bound arrays, with the method stated. They were not pre-registered. No network was loaded or run for them.

Times are EDT (UTC−4).

## 1. Summary

Idea 2 asked whether trained racing networks hold Pontryagin's costate. The TO-OCP's window problem has an optimal value V*_W, and along its optimum the costate λ_t is the gradient of the remaining value with respect to the car's state: what one more metre of progress or one more radian of heading is worth. It is exact at every state of every answer key, from the solver's multipliers. Three questions were tested: is the critic's input gradient, pulled back to the state, the costate (C1)? Does the actor's last hidden layer carry costate information beyond its input and output (C2)? Does the critic's price of the inside wall go with driving quality (C3-w)? C1 and C2 were decided on step 1's nine generalists (G1, G2 and G3, two seed replicates each), C3-w across 31 generalists. All networks were scored on the same states, the optimum's own plan states of 477 synthetic corners (T). C1, C2 and C3-w were scored on T's interior stratum, with no wall or slip contact within 10 steps (3,951 states, of which s 1,643 and ξ 1,581 decide C1 and C2), C2-sw on T's 2,911 pedal-switching steps, and C1-own on 790–813 interior restarts per network. The design was reviewed and gated on random-init and planted networks, its gate and analysis rules were revised in rounds before the tag (design Sections 12–18), and the tag was placed at 17:37:59. The analysis ran once, 19:43:10–19:48:43.

**What could be decided.** A second corner placed beyond the networks' preview moves the optimum's λ in the braking and corner phases (gate K2b), so those rows were excluded before the tag. The decisions rest on λ_s (1,643 rows: approach and exit) and λ_ξ (1,581 approach rows). λ_n and λ_vx have no decision row, and C2's power floor also passes only s and ξ.

**Confirmatory (primary family).**
- **C1, the critic's gradient is the costate: refute** in all 9 networks and in each variant (3 of 3 seeds in v1, v2 and v3).
  - R²_c, the calibrated R² averaged over s and ξ, is −2.575 to 0.010. Every CI upper end is at most 0.025, against the refute band's 0.2. Per component it is at most 0.029 for λ_s and 0.006 for λ_ξ.
  - The rank agreement is weak but often positive: Spearman's ρ for ξ is 0.34–0.40 with CI lower ends above 0 in six networks.
  - A phase-matched λ borrowed from another geometry scores 0.194 on ξ, above every network's critic.
  - All 48 leave-one-geometry-out sets refute in all 9.
- **C2, the actor's h3 carries costate information beyond z, m and position-and-speed: inconclusive** in 7 networks and **refute** in 2 (G2s2, G3s2). Every variant is inconclusive. The points are −0.038 to 0.039 against the confirm band's 0.05. G1s1's CI lower end is above 0 (0.009), with a point of 0.039; 2 of the 20 random-init null hosts also have CIs wholly above 0 (Section 6.4). An inconclusive C2 is evidence neither for nor against.
- **C3-w, inside-wall price fidelity goes with completion: rules out a strong association.** The weighted Spearman ρ is −0.167, its 95% upper bound 0.169 (below 0.3), p 0.79, over 31 generalists.

**Secondary.**
- **C2-sw, the actor carries the brake-onset switching function: refute** in all 9 (every CI upper end at most 0.045). Its point, 0.006–0.030 in R², lies below the 0.05 band. Descriptively (the pool rank, design 16.3), the trained h3's gain ranks first of 21 (above all 20 random-init actors) in 7 of 9 networks, and in the same 7 its CI lower end is above 0. σ^ped is not residualised on position and speed, so this does not isolate the switching function.
- **C1-own, the critic's gradient is its own policy's costate: refute** in all 9, each valid.
- **C3-R²c: rules out a strong association**, in the negative direction: ρ −0.394, upper bound −0.057. The test has no reversed-sign refute, so this is descriptive.
- **C1-heads:** the five value heads agree on V more closely than on its gradient in all 9 networks (and, descriptively, in all 22 other generalists). Every gradient component's head disagreement is 7.8 to 23.9 times the value's.
- C1-braking refutes in all 9, but largely repeats C1: under K2b it scores no braking-phase row (Section 5.9). C1-limit is not testable (2 contact rows in one geometry); C1-pattern confirms only for G1, on two R²_c near or below 0; W1 is near 0 except for G3 (0.256) and G3s2 (0.114).

**Validity.** Every pre-tag gate with a verdict passed in the canonical pass at `a616688` (K6 counts coverage and has none), K0 on the scope set in design Section 18 after its failure had been seen. K0 had failed the earlier canonical pass (`291e2bc`), which under v2.4's and v2.7's stated procedure blocks the tag; instead its verdict was narrowed and the pass re-run, a departure from that procedure before the tag (Section 3.5). After the tag, the trained-network checks passed for all 34 networks (K3) and all 9 decision networks (K3b's trained analogue). Value accuracy passes in all 9: on D's states and T's interior restarts (all phases), the calibrated critic value explains 80–93% of the variance of its own deterministic policy's return. No network was missing or descriptive, no deviation was logged and no code changed after the tag. The single-geometry rule withdrew no decision.

**Most informative exploratory result** (Section 7.1, computed here). λ_ξ's R²_c lies far below what a zero or pure-noise gradient scores (−0.003) because 14–21% of the critic's ξ gradients form a heavy tail whose sign is opposite to λ_ξ's on 86–92% of its rows. The calibration fits its slope on the bulk, where g and λ agree weakly or not at all, and applies it to the tail. G1 and G2 score about 0 on ξ only because their fitted slopes are 4 to 11 times smaller. The anti-aligned tail is present in all 9 critics. It explains the very negative magnitudes, not the refute: on the bulk alone ξ's R²_c is at most 0.092.

**Reading.** On the states where the design could decide, the critics fit their own policies' values but their state gradients do not match the window problem's costate in shape: rank agreement is weak, and a calibration fitted on other geometries does not transfer. No costate information beyond z, m and smooth position-and-speed variables was detected in the actor's h3 at the pre-registered band. C2-sw refutes by its band (every CI upper end below 0.05) in all 9; descriptively, the trained h3's gain ranks first of 21 in 7 networks, but σ^ped is not residualised on position and speed, so this does not isolate the switching function. Inside-wall price fidelity does not go strongly with completion. All of this is narrow: λ_s and λ_ξ only, the approach phase (and exit for s), synthetic corners and the optimum's states.

## 2. What was asked

Source: design Sections 1, 3 and 9.

- **The target.** The TO-OCP's window problem: W steps of the agent's own reward, no terminal value, the OCP's margins (3 cm inside the env's edge, 10⁻⁴ rad inside the slip limit), nominal grip and the full road. Its post-jump costate λ_t is the gradient of the value of the future from x_t (design 3.1), with the pedal multipliers folded to unique values. λ_s and λ_n are per metre, λ_ξ per radian, λ_vx per m/s, in reward units (1 = 10 m of progress).
- **The critic's side.** The main value head's input gradient, pulled back to the car's state on the manifold of consistent observations and scaled by σ_R (design 3.2). If V_c∘z equalled V* there, this g_t would equal λ_t.
- **Why even an exact critic would not equal λ** (design 3.3): hidden grip jitter, the preview limit, the termination penalty and the margins, and the critic estimates V^π of the stochastic training policy. So C1 compares shape and sign after a per-fold affine calibration, not raw values.
- **What a refute would mean** (design Section 9), read with C1-own, value accuracy and C1-heads. Accurate values with high gradient disagreement between heads: the regression does not determine the gradient. Low disagreement: a determined gradient, but of V^π rather than V*.

## 3. What was pre-registered, and how it was tested

### 3.1 States, targets and networks

- **T** (design Section 4; K6 in `outputs/costate/gates/k6.json`): the optimum's plan states every 5th step of 477 A2 keys (the 481 verified keys less 4 multi-basin keys, 80 rows), within K2's cut t_H = 267: 9,887 states over 48 geometries (R, θ), the same for every network.
- **The interior stratum** (design Section 12, decision 4): no corridor or slip row active from t to t + 10 and margins above fixed thresholds. 3,951 states, a key-weighted share of 0.385 of T. By phase the interior share is 0.850 of approach rows, 0.255 of braking, 0.636 of corner and 0.030 of exit (K6, `interior_by_phase`).
- **Targets.** K_P = {s, n, ξ, vx}, winsorised at bounds frozen in the record (`frozen.winsor_bounds`). The mirror-odd components (n, ξ) are scored by R²_wd, whose SST is taken within each turn direction.
- **Weights and CIs.** Each key weighs equally and states equally within a key, renormalised on the scored stratum. CIs are a 2,000-draw cluster bootstrap over the 48 geometries. Six outer folds over geometries, partition 0 frozen.
- **Networks** (the record's `networks`, with checkpoint sha256s):

| Label | Run | Variant |
|---|---|---|
| G1, G1s1, G1s2 | `runs/dr-pilot-200M`, `-s1`, `-s2` | v1 |
| G2, G2s1, G2s2 | `runs/dr-pilot-v2-tilt30`, `-s1`, `-s2` | v2 |
| G3, G3s1, G3s2 | `runs/dr-pilot-v3-tilt50a2`, `-s1`, `-s2` | v3 |

The other 22 generalists (seeds 3–4 of each variant, real-width v1 and v3 seeds 0–3, leave-out f1 and f2 seeds 0–3) are descriptive for C1 and enter C3. The specialists S1–S3 (`runs/hockenheim-specialist-mlp-200M`, `-s2`, `-s3`) are descriptive throughout.

### 3.2 What can decide: K2b's exclusions and C2's power floor

**K2b** (design Section 7; `gates/k2b.json`). On 26 keys, a second corner was placed just beyond the largest preview reach of the key's T states and the key re-solved. A component is excluded from decisions in a phase where the p90 of |Δλ_k|/SD_T exceeds 0.1. With the slow second corner:

| Phase | s | n | ξ | vx | Excluded |
|---|---|---|---|---|---|
| approach | 0.075 | 0.120 | 0.051 | 0.255 | n, vx |
| braking | 0.247 | 0.231 | 0.240 | 0.129 | s, n, ξ, vx |
| corner | 0.372 | 0.235 | 0.302 | 0.357 | s, n, ξ, vx |
| exit | 0.052 | 0.262 | 0.131 | 0.442 | n, ξ, vx |

With the fast second corner every p90 is at most 0.0085. The exclusions are frozen in the record (`frozen.k2b_excluded`). Their consequence:
- ξ is scored only on the interior's approach rows (1,581) and s on approach plus 62 exit rows (1,643; s is structurally unobservable on most of the exit straight). These counts are the same under all 9 networks' observation scalings (`networks.<run>.decision_rows`).
- n and vx have no decision row. The interior's braking (871) and corner (1,427) rows have no scored row for any K_P component.
- So C1, C1-own and C1-braking are decided on {s, ξ}.

**C2's power floor** (design Section 12, decision 5, and 15.10). A K_P component decides C2 only if at least 0.10 of its interior variance remains after the decoy model f̂_D. At f̂_D's frozen k = 8 the shares are s 0.224, n 0.091, ξ 0.248 and vx 0.076 (`frozen.residual_share_interior`), so the floor passes s and ξ. n would have no scored row in any case.

### 3.3 Hypotheses and decision rules

| Hypothesis | Statistic | Confirm | Refute | Rows |
|---|---|---|---|---|
| **C1** (primary) | R²_c: R²_wd of the calibrated g against λ, averaged over the scored K_P components. Calibration per outer fold: λ = a_dir + b·g, b a leverage-weighted least-squares slope clamped at ≥ 0, direction intercepts for ξ, predictions clipped to the winsor bounds (design Section 13) | CI lower end > 0.6, every null's paired CI lower end > 0, and Spearman CI lower end > 0 on every component | CI upper end < 0.2 | interior, s 1,643 and ξ 1,581 |
| **C2** (primary) | Δ2 = R²(h3 ⊕ z ⊕ m) − R²(z ⊕ m) on λ_res = λ − f̂_D (cross-fitted kNN, k = 8, on (s − s_apex, vx, log R, θ, w, σ)), trained minus the median over a pool of 20 random-init actors, averaged over the floor's components; two-level CI resampling geometries and controls (design Section 16) | CI lower end > 0 and point ≥ 0.05 | CI upper end < 0.05 | interior, s 1,643 and ξ 1,581 |
| **C2-sw** (named secondary) | Δsw = R²(h3 ⊕ z) − R²(z) on σ^ped, the folded pedal-command multiplier, trained minus the pool's median; two-level CI | as C2 | as C2 | pedal-switching steps, 2,911 |
| **C3-w** (primary) | Size-weighted mean of within-stratum Spearman ρ of x (the calibrated R² of the critic's uniform inside-widening derivative against Π^in on T's interior) with y (mean R-dev completion), 31 generalists in 7 recipe strata; one-sided stratified permutation, 100,000 draws | p ≤ 0.05 | none; "rules out a strong association" if the 95% upper bound < 0.3 | 31 networks |
| **C3-R²c** (named secondary) | C3-w's test with x = each generalist's C1 R²_c | as C3-w | as C3-w | 31 networks |
| **C1-own** (secondary) | C1's statistic, bands and sign rule with λ^own, the gradient of the deterministic policy's own H-step return, in place of λ; valid only if K3b's trained analogue and value accuracy pass | as C1 | as C1 | T's interior restarts |

- **Nulls for C1** (design Section 6): gradients of value fits on z (ridge and a 64-unit MLP to V*_∞, ridge to the window value), a QSS value and a clearance-barrier value, the phase-matched far-geometry λ, and the best of 20 random-init critics. Each is put through the same pullback and calibration.
- **The "at scale" label** needs a ruled confirm and a slope bracket (design Section 13); it is false everywhere and moot.
- **Variants** (R18): a variant claim needs 2 of 3 seeds.
- **The single-geometry rule** (design Section 17): each decision network's C1, C2 and C2-sw is also decided with each geometry left out of scoring, and a confirm or refute stands only if all 48 sets decide the same. The rule can only withdraw a decision.
- **Multiplicity** (design Section 6): C1, C2 and C3-w answer different questions and are reported without correction across them.

The other pre-registered secondaries (design Sections 6 and 14): C1-pattern (R²_c over K_P exceeds R²_c over {vy, r, δ, pedal}), C1-heads (between-head disagreement D_q of values and gradients), C1-limit (at wall contacts the calibrated g_n and g_ξ exceed λ* in magnitude), C1-braking (C1's rule on the braking keys) and W1 (the critic's inside-widening price against Π^in, with no band).

### 3.4 Deferred analyses

Not part of `costate-prereg-v1` (design Sections 13 and 14; the record's `deferred`, nine entries). If run later they are exploratory:
1. C3-int, the network × geometry link between critic mispricing and driving deviation;
2. C1-dev, the trend of R²_c over the 16 training snapshots;
3. ΔC1, whether the gradient adds to the input;
4. C2-pairs, matched-decoy state pairs;
5. C1-own on each network's own-run states and on N;
6. N's per-agent replication of C1, C2 and C2-sw (N is never built);
7. W1's probe R² and Spearman for Π^in, Π^out, Π^slip and Π^μ;
8. the policy's own width price, and W1's three comparisons with it;
9. C2 and C2-sw of the 22 other generalists and the specialists.

### 3.5 The rules were revised in rounds before the tag

The gate rules and several analysis rules were revised in rounds before the tag, recorded in design Sections 12–18. Every revision came before any trained network's activation, value or gradient was computed. Several were chosen after the failure they address had been seen on random-init or planted networks, and the design discloses each. Revisions that bear on the results reported here:
- **v2.1** (design Section 12): C2-sw and C3-int moved from design Section 6's primary family to named secondaries.
- **v2.2** (design Section 13): Section 6's Huber calibration of C1 ('a per-fold Huber affine fit of λ_k on g_k') was replaced by a leverage-weighted least-squares slope (squared leverage weights, K_LEV = 3, slope clamped at ≥ 0); Huber is kept as a coded option, unused. This is the calibration that Section 7.1 shows drives ξ's very negative scores. Among the implementation review's critical findings, Section 13 lists that C1's calibration gave uninformative gradients a negative floor, and it states gate PR-5e: uninformative, zero, noise, direction-only and anti-aligned gradients must score in [−0.02, 0.005]. C2-sw's stratum moved from Section 6's pedal-saturated steps to the pedal-switching steps.
- **v2.3** (design Section 14): C1's fold-dependent nulls were rebuilt per held-out fold (#20). C3-int was deferred. C3-R²c's x went from design Section 6's v1 networks to all 31 generalists.
- **v2.4** (design Section 15; 23 Oct). The full-size smoke run failed K4 (PR-1b, PR-6) and K5, and K1 passed only on a rule design Section 7 did not state. K1's rule was written down; its noise floor had been coded after a development row of K1's own sample was seen (15.2, 'Disclosure'). K5's permutation scheme and tolerance were chosen after its failure (15.3). PR-1b's plant was rebuilt on one target array (15.4). PR-6 became one-sided, and PR-6b and PR-6c were added (15.5, 15.6). f̂_D's k went from 15 to a nested-CV choice, which failed PR-6b at full size, and then to one fixed k = 8 by a rule declared on six validation seeds (15.7). The fixed-k rule was decided with a descriptive k = 15 and k = 5 diagnostic on PR-6b's gate seeds already known, and both worlds' gate seeds had been scored at k = 8 (unpaired) before the rule was re-applied under the paired statistic; the re-application read only the validation seeds (15.1, 15.7). The nested-CV k moved λ_n's interior residual share below the power floor (0.107 to 0.064); at k = 8 it is 0.091. K4's no-gain plants became paired increments after PR-3 failed at 0.0108 against 0.01 (15.1, round 3, 'Disclosure'). C1's value-fit nulls were fully cross-fitted (15.9); on G1's decision rows, in the smoke run's world, the value MLP null moved from −0.0278 to −0.0164, and G1's "not above the value MLP" (Section 5.2) is measured against the cross-fitted null.
- **v2.5** (15.14): C2's random-only null was measured. Its SD on C2's decision rows was about 0.03, not the 0.007 v2.4 had stated.
- **v2.6** (design Section 16): the primary C2 and C2-sw are scored against a pool of 20 random-init actors with a two-level CI, chosen after v2.5's null had been seen (16.2, 'Disclosure').
- **v2.7** (design Section 17): the single-geometry robustness rule, after a review found that one geometry (6) holds about two thirds of C2's base error and that C2's CI covers 0.68–0.79 there (17.2, 'Disclosure').
- **v2.8** (design Section 18): K0 failed the canonical pass on 6 of N's 8,136 anchored finals (pedal-fold stationarity 1.6·10⁻¹⁰ to 1.2·10⁻⁹ against 10⁻¹⁰), as it had unnoticed in the smoke run. Its verdict was narrowed to the 511 keys a pre-registered computation reads. Design Section 18.4 records this as narrowing a gate after its failure was seen.

v2.4 declared the gate rules final, yet v2.6–v2.8 still changed analysis and gate rules before the tag, each recorded in its section. v2.7 again declared the rules final and said a gate failing the canonical pass blocks the tag. K0 failed that pass (`291e2bc`); v2.8 narrowed K0's verdict and the pass was re-run at `a616688`. Design 18.4 records this. It is a departure from v2.4's and v2.7's stated procedure before the tag, so `outputs/costate/deviations.jsonl`, which covers changes after the tag, does not record it. After the tag any change is a logged deviation (design Sections 17 and 18.5); none was logged (Section 6.5 below).

## 4. Procedure and timeline

Sources: commits; the provenance stamps of the bound outputs and of the post-tag stage outputs; `outputs/costate/logs/` (local) for the first inputs run.

| Step | When (24 Oct, EDT) | Record |
|---|---|---|
| The single-geometry rule committed (v2.7) | 08:14:30 | `cd34593`; design `291e2bc` (08:30:59) |
| Canonical pre-tag pass at `291e2bc`: every gate but K0 passed, and prereg refused | 11:42–14:00 | design Section 18.1 |
| K0's scope (v2.8) committed | 14:07:50 | `a616688` |
| Canonical pass re-run at `a616688`: inputs, K2, inputs again (binding t_H), then K2b with K0, K1, K3, K3b, K5 and the nulls beside it, then c2-null with K4, decoy-k and K6 beside it | 15:07:19–17:35:38 | the 23 bound outputs' stamps (all HEAD `a616688`, code hash `d22e670c…`, none partial, none with a deviation) |
| Record committed alone; annotated tag placed and pushed | 17:37:59 | `11dc9bc`; tag object `d1f00c93…` |
| activations, 34 networks (the tag pinned at 17:39:30) | 17:38:36–17:44:49 | `post/activations` stamps; `tag.json` `pinned_at` |
| Tag pin committed | 17:44:50 | `b774472` |
| critic, with K3's after-the-tag checks, 34 networks | 17:44:57–17:55:51 | `post/critic` stamps |
| own, with K3b's trained analogue, 9 networks | 17:55:52–18:10:48 | `post/own` stamps |
| probes (C2 and C2-sw), 9 networks | 18:10:49–19:40:32 | `post/probes` stamps |
| analyse without `--confirmatory`: coverage only, `decided` false | 19:40:33–19:42:04 | `analyse/decisions_dry.json` (local) |
| **analyse `--confirmatory`, once** | 19:43:10–19:48:43 | `decisions.json` provenance |
| Decisions committed | 19:48:58 | `9b25cc2` |

Every post-tag stamp carries the tag `costate-prereg-v1`, the code hash `d22e670c…`, no deviation and no partial option.

## 5. Confirmatory decisions

### 5.1 The run

Source: `decisions.json` (`provenance`, `confirmatory`, `decided`, `deviation`, `unlocked_by`, `k3_trained`).
- `scripts/costate.py analyse --confirmatory`, 19:43:10–19:48:43, HEAD `b774472` (the tag pin), `changed_vs_head` empty, `partial` empty, deviation null. `confirmatory` and `decided` are true; `unlocked_by` is null.
- Tag `costate-prereg-v1` (object `d1f00c93…` on `11dc9bc`, tagger date 17:37:59).
- `k3_trained`: 34 networks, 34 pass, none failed or not run. So no network is descriptive or missing from any decision.

### 5.2 C1: the critic's gradient against the costate

Source: `networks.<run>.C1`. R²_c is the mean of the per-component R²_c (`r2c`) over s and ξ. The Spearman ρ for ξ is the within-direction mean (design Section 14, #1). Decision: the ruled one; in brackets the full set's (`decision_all_geometries`) and the leave-one-geometry-out sets'.

| Network | R²_c [95% CI] | R²_c s | R²_c ξ | Spearman s [CI] | Spearman ξ [CI] | Decision |
|---|---|---|---|---|---|---|
| G1 | 0.010 [−0.060, 0.025] | 0.021 | −0.002 | 0.20 [0.07, 0.33] | 0.05 [−0.06, 0.22] | refute (full set refute; 48 of 48 sets refute) |
| G1s1 | −0.726 [−1.354, −0.389] | −0.102 | −1.351 | 0.10 [−0.01, 0.23] | 0.34 [0.22, 0.46] | refute (refute; 48 of 48) |
| G1s2 | −1.403 [−2.818, −0.593] | −0.101 | −2.706 | 0.15 [0.04, 0.26] | 0.34 [0.22, 0.48] | refute (refute; 48 of 48) |
| G2 | −0.049 [−0.179, 0.012] | −0.103 | 0.006 | 0.14 [−0.03, 0.30] | −0.04 [−0.17, 0.16] | refute (refute; 48 of 48) |
| G2s1 | −2.477 [−4.802, −1.081] | 0.029 | −4.984 | 0.19 [0.06, 0.33] | 0.35 [0.21, 0.49] | refute (refute; 48 of 48) |
| G2s2 | −0.168 [−0.475, −0.036] | −0.015 | −0.320 | 0.01 [−0.12, 0.15] | 0.02 [−0.11, 0.23] | refute (refute; 48 of 48) |
| G3 | −1.359 [−2.932, −0.364] | −0.038 | −2.679 | 0.32 [0.20, 0.44] | 0.39 [0.29, 0.51] | refute (refute; 48 of 48) |
| G3s1 | −2.575 [−5.035, −1.054] | −0.147 | −5.003 | 0.22 [0.10, 0.36] | 0.40 [0.30, 0.50] | refute (refute; 48 of 48) |
| G3s2 | −1.386 [−2.815, −0.480] | −0.063 | −2.709 | 0.16 [0.01, 0.33] | 0.36 [0.23, 0.49] | refute (refute; 48 of 48) |

- **Refute in all 9.** Every CI upper end is below 0.2; the largest are G1's 0.025 and G2's 0.012. No row was dropped for want of a calibration (`n_uncalibrated` 0).
- **The best agreement is about 0.** G1's R²_c is within 0.03 of 0 on each component, and so is G2's on ξ (its λ_s scores −0.103). For comparison, several nulls score exactly −0.015 on s and −0.011 on ξ in every network (`null_r2c`: the value ridges; on ξ also the QSS value and the value MLP). Design Section 15.9 records the ridge nulls' slopes clamping at 0 on these rows, and G2s2's own λ_s, whose slope clamps at 0 in all six folds (`calibration.b`), also scores −0.015. That is what an intercept-only prediction scores here: −0.013 on the mean of s and ξ.
- **The rank agreement is weak, often positive.** On s, ρ is 0.01–0.32, with the CI lower end above 0 in six networks (G1, G1s2, G2s1, G3, G3s1, G3s2). On ξ it is −0.04 to 0.40, with the CI lower end above 0 in six (G1s1, G1s2, G2s1, G3, G3s1, G3s2), where it is 0.34–0.40. Both components' lower ends are above 0 in five networks (G1s2, G2s1, G3, G3s1, G3s2), whose R²_c are all far below 0.
- **No "at scale" label** (`label_at_scale` false everywhere; it needs a confirm). The pooled slopes are far from 1 both ways: λ_ξ on g_ξ 0.003–0.040 (`slope`), g_ξ on λ_ξ 9.1–19.4 (`slope_gl`).
- **Most influential geometry** (`logo.most_influential`): geometry 38 (R 44.7 m, θ 180°) for 7 networks, whose sets move R²_c up by 0.09 to 0.65. Every set still refutes; the largest set CI upper end is G1's 0.029.

**Against the nulls** (`null_r2c`, `null_diff`; the confirm rule needs every paired lower end above 0).

| Null | R²_c s | R²_c ξ | Networks above it (paired CI lower end > 0) | Networks below it (paired CI upper end < 0) |
|---|---|---|---|---|
| value ridge on z, to V*_∞ | −0.015 | −0.011 | G1 | 7 (all but G1, G2) |
| value ridge on z, to the window value | −0.015 | −0.011 | G1 | 7 (all but G1, G2) |
| value MLP on z | −0.067 to −0.018 | −0.011 | none | 6 (all but G1, G2, G2s2) |
| QSS value | −1.489 to −0.974 | −0.011 | G1 | 2 (G2s1, G3s1) |
| clearance barrier | −0.015 | −0.013 to −0.010 | G1 | 7 (all but G1, G2) |
| phase-matched far-geometry λ | −0.015 | 0.194 | none | 9 |
| best of 20 random-init critics | −0.009 to −0.005 | 0.027 to 0.093 | none | 7 (all but G1, G2) |

The paired statistic is R²_c minus the null's, on the K_P mean. Even G1, the only network above any null, is below the far-geometry λ (paired difference −0.080 [−0.105, −0.052]) and not above the value MLP (lower end −0.0017) or the random critics (−0.043). The random-critic null is a selected maximum, the design's sanity floor: its R²_c is that of the seed with the highest K_P-mean R²_c of 20 on these same rows, and its paired CI takes the maximum over seeds in each bootstrap draw (`analysis.c1`). On ξ, this best of 20 random-init critics, pulled back the same way, scores 0.027–0.093, higher than every trained critic (at most 0.006).

### 5.3 C2: the actor's h3 beyond its input, its output and position-and-speed

Source: `networks.<run>.C2` (`stat`, `ci` two-level, `ci_geometry`, `gain` per component, `pool`, `c2_null_note`). "Rank" is the network's rank among the 21 (the network and the 20 pool actors) by its own gain over the base (descriptive, design 16.3). "Null ≥" is the share of the 20 random-only null hosts whose statistic is at or above the network's (descriptive, Section 6.4).

| Network | Δ2 [two-level CI] | Geometry-only CI | Gain s | Gain ξ | Rank of 21 | Null ≥ | Decision |
|---|---|---|---|---|---|---|---|
| G1 | 0.030 [−0.033, 0.203] | [−0.031, 0.179] | 0.015 | 0.045 | 2 | 0.00 | inconclusive |
| G1s1 | 0.039 [0.009, 0.357] | [0.011, 0.340] | 0.070 | 0.008 | 2 | 0.00 | inconclusive |
| G1s2 | 0.031 [−0.008, 0.339] | [−0.005, 0.331] | 0.058 | 0.004 | 3 | 0.00 | inconclusive |
| G2 | 0.023 [−0.143, 0.131] | [−0.130, 0.124] | 0.048 | −0.003 | 3 | 0.10 | inconclusive |
| G2s1 | 0.001 [−0.173, 0.137] | [−0.165, 0.131] | 0.031 | −0.030 | 9 | 0.30 | inconclusive |
| G2s2 | −0.038 [−0.512, −0.017] | [−0.519, −0.021] | −0.057 | −0.019 | 19 | 0.95 | **refute** |
| G3 | −0.006 [−0.089, 0.171] | [−0.079, 0.146] | 0.014 | −0.026 | 15 | 0.35 | inconclusive |
| G3s1 | 0.033 [−0.024, 0.340] | [−0.022, 0.337] | 0.085 | −0.019 | 4 | 0.00 | inconclusive |
| G3s2 | −0.014 [−0.266, 0.020] | [−0.264, 0.014] | 0.025 | −0.053 | 16 | 0.60 | **refute** |

- **No confirm.** No point reaches 0.05. G1s1's CI lower end is above 0, but its point is 0.039.
- **Two refutes**, with CI upper ends −0.017 (G2s2) and 0.020 (G3s2). Their sets all refute (48 of 48).
- **The gain is carried by λ_s.** The trained-minus-pool gain is positive for s in 8 of 9 and negative for ξ in 6 of 9.
- **Geometry 6** (R 9.4 m, θ 90°) holds 0.689–0.709 of the base probe's squared error on s and 0.643–0.646 on ξ in every network, against 0.055 and 0.049 of the weight (`logo.shares`). It is the most influential geometry for 8 of 9 networks. The rule withdrew nothing: every set is inconclusive in 6 networks and refutes in G2s2 and G3s2. In G1s1 one set confirms, the set without geometry 6 (0.102 [0.020, 0.675]); the rule can only withdraw a decision, so the full set's inconclusive stands.
- **K4's band notes** (design Section 15.10): none beside a primary C2 (`k4_band_notes.placed`), since neither refute has a CI upper end at or above 0.03 and nothing confirms.

### 5.4 C2-sw: the switching function on pedal-switching steps (named secondary)

Source: `networks.<run>["C2-sw"]`. The base R²(z) is 0.733 in all 9 (`r2_base`). "Trained gain" is R²(h3 ⊕ z) − R²(z) for the network's own h3; "Pool gain" the same for the 20 random-init actors. "Rank of 21" is the pool rank (descriptive, design 16.3).

| Network | Δsw [two-level CI] | Trained gain | Pool gain: median [min, max] | Rank of 21 | R²(h3 ⊕ z) | Spearman ρ(m_pedal − 1, σ^ped) | Decision |
|---|---|---|---|---|---|---|---|
| G1 | 0.028 [0.012, 0.043] | 0.183 | 0.154 [0.137, 0.167] | 1 | 0.915 | 0.37 | refute |
| G1s1 | 0.030 [0.020, 0.045] | 0.182 | 0.152 [0.127, 0.160] | 1 | 0.915 | 0.38 | refute |
| G1s2 | 0.025 [0.017, 0.037] | 0.177 | 0.152 [0.124, 0.160] | 1 | 0.910 | 0.37 | refute |
| G2 | 0.021 [0.002, 0.036] | 0.190 | 0.169 [0.146, 0.184] | 1 | 0.923 | 0.46 | refute |
| G2s1 | 0.009 [−0.007, 0.020] | 0.174 | 0.164 [0.146, 0.180] | 3 | 0.906 | 0.07 | refute |
| G2s2 | 0.029 [0.021, 0.038] | 0.198 | 0.169 [0.150, 0.183] | 1 | 0.931 | 0.33 | refute |
| G3 | 0.006 [−0.004, 0.016] | 0.184 | 0.178 [0.162, 0.189] | 3 | 0.917 | 0.09 | refute |
| G3s1 | 0.021 [0.012, 0.031] | 0.192 | 0.170 [0.157, 0.187] | 1 | 0.925 | 0.06 | refute |
| G3s2 | 0.028 [0.017, 0.037] | 0.197 | 0.169 [0.156, 0.186] | 1 | 0.930 | 0.50 | refute |

- **Refute in all 9**: every CI upper end is below 0.05 (largest 0.045, G1s1). All 48 sets refute in every network.
- **The pool rank (descriptive, design 16.3).** In 7 of 9 networks the trained h3 adds more than every one of the 20 random actors (rank 1), and in the same 7 the CI lower end is above 0. The 2 others (G2s1, G3) rank 3. σ^ped is not residualised on position and speed, so units that hold those decoys could raise the gain by their whole predictive share (design 15.10). The rank therefore does not isolate the switching function, which design Section 9 claims only with a C2-sw confirm.
- **Size.** σ^ped, the countdown to brake onset on full-throttle stretches, is largely readable from z (R² 0.733). A random h3 adds a median 0.152–0.178. The trained h3 adds 0.006–0.030 beyond that median, so R²(h3 ⊕ z) is 0.906–0.931.
- C2-sw has no random-only null of its own (design 15.10, 16.5). Its reliability is measured against the pool and its two-level CI only, and that CI's coverage on C2-sw's rows was not measured (design 16.3). K4's decoy-leak bounds were measured on C2's setup and do not carry over to it (design 15.10).

### 5.5 Variants (R18)

Source: `variants`. A variant claim needs the same decision in 2 of 3 seeds.

| Variant | C1 | C2 | C2-sw |
|---|---|---|---|
| v1 (G1, G1s1, G1s2) | refute (3 of 3) | inconclusive (3 of 3) | refute (3 of 3) |
| v2 (G2, G2s1, G2s2) | refute (3 of 3) | inconclusive (2 of 3; G2s2 refutes) | refute (3 of 3) |
| v3 (G3, G3s1, G3s2) | refute (3 of 3) | inconclusive (2 of 3; G3s2 refutes) | refute (3 of 3) |

### 5.6 C3-w: inside-wall price fidelity and completion

Source: `C3-w`, `C3-w_sensitivity`. y is each generalist's mean completion over the 12 R-dev circuits (`prereg.json` `c3w_inputs.<run>.completion`). x is not stored per network in `decisions.json`; for the 9 decision networks it is the same quantity as W1's R²_c (Section 5.9) by code path: `c3w_of` and W1 both call `analysis.calibrated_agreement` on the critic's `dV_dwin` against Π^in on T's interior, with the frozen Π^in bounds (`scripts/costate.py`; not re-checked numerically). No row of any network lacked a calibration (`x_n_uncalibrated` 0 for all 31).

| Stratum | n | ρ, C3-w | ρ, C3-R²c |
|---|---|---|---|
| base v1 | 5 | −0.205 | −0.616 |
| base v2 | 5 | −0.872 | −0.667 |
| base v3 | 5 | 0.600 | 0.800 |
| real-width v1 | 4 | −0.400 | −0.800 |
| real-width v3 | 4 | −0.894 | −0.447 |
| leave-out f1 | 4 | 0.200 | −0.400 |
| leave-out f2 | 4 | 0.400 | −0.800 |
| **Weighted** | 31 | **−0.167** | **−0.394** |
| 95% upper bound | | 0.169 | −0.057 |
| p (one-sided, positive) | | 0.788 | 0.974 |
| Critical ρ; null SD | | 0.338; 0.204 | 0.337; 0.204 |
| Partial, competence (31) | | 0.035 | −0.402 |
| Partial, overlap (26; base v2 skipped) | | −0.179 | −0.113 |
| Decision | | **rules out a strong association** | **rules out a strong association** |

- **C3-w rules out a strong association**: the upper bound 0.169 is below 0.3. All 31 networks vary in x and y; none is missing.
- **Sensitivities** (descriptive): with the κ set dropped, ρ −0.242, upper bound 0.095, p 0.880; with the small geometries dropped, ρ −0.167, upper bound 0.168, p 0.788. Both rule out a strong association.
- The overlap partial leaves out `runs/dr-pilot-v2-tilt30-s3` and `-s4`, which have no step 2 rows, and so skips the base-v2 stratum (design Section 14).
- Power (design Section 6): the null SD is about 0.20, and 80% power needs a latent ρ of about 0.6. "Rules out a strong association" fires about 40% of the time under an exact null.

### 5.7 C1-own: the critic's gradient against its own policy's costate (secondary)

Source: `networks.<run>["C1-own"]`. Scored on T's interior restarts with a valid own run (790–813 per network, `n_stratum`); under K2b's exclusions that leaves 161–163 rows per component. Validity (`validity`): K3b's trained analogue and value accuracy pass in all 9, so its status is 'secondary' everywhere.

| Network | R²_c [95% CI] | R²_c s | R²_c ξ | Decision |
|---|---|---|---|---|
| G1 | −0.122 [−0.313, −0.048] | −0.113 | −0.130 | refute |
| G1s1 | −0.054 [−31.190, −0.034] | −0.067 | −0.041 | refute |
| G1s2 | −0.047 [−11.153, −0.034] | −0.029 | −0.065 | refute |
| G2 | 0.029 [−0.284, 0.046] | 0.071 | −0.012 | refute |
| G2s1 | 0.026 [−0.425, 0.146] | 0.020 | 0.031 | refute |
| G2s2 | −0.150 [−0.979, 0.115] | −0.079 | −0.222 | refute |
| G3 | −0.126 [−0.624, 0.035] | −0.119 | −0.133 | refute |
| G3s1 | −0.075 [−0.878, 0.113] | −0.156 | 0.007 | refute |
| G3s2 | −0.031 [−0.716, 0.152] | −0.011 | −0.052 | refute |

**Refute in all 9**: every CI upper end is at most 0.152. The rows are few, and two CIs have very long lower tails (G1s1, G1s2). Unlike C1, no ξ score is far below 0 here.

### 5.8 C3-R²c (named secondary)

Source: `C3-R2c`. x is each generalist's C1 R²_c (`networks.<run>.C1.stat` for the 9, `descriptive_networks.<run>.C1.stat` for the other 22), y as in C3-w. **Rules out a strong association**, with ρ −0.394, upper bound −0.057 and p 0.974 (table above). Within 6 of the 7 strata, generalists with a higher C1 R²_c had a lower mean completion. The one-sided test has no reversed-sign refute (design Section 6), so the negative direction is descriptive. The design expected a null here.

Computed here: the size-weighted within-stratum Spearman ρ recomputed from those x and y is −0.39359, and each stratum's ρ equals the record's.

### 5.9 The other pre-registered secondaries

**C1-heads** (source: `networks.<run>["C1-heads"]`). D_q = mean_t var_k(q^k_t) / var_t(mean_k q^k_t) over the five heads; the comparison D_grad(K_P) − D_V > 0 confirms if its CI lower end is above 0.

| Network | D_V | D of the 8 gradient components | D_grad(K_P) − D_V [CI] | D_grad({vy, r, δ, pedal}) − D_grad(K_P) |
|---|---|---|---|---|
| G1 | 0.0014 | 0.0117–0.0151 | 0.0117 [0.0094, 0.0144] | not confirmed |
| G1s1 | 0.0020 | 0.0183–0.0374 | 0.0250 [0.0158, 0.0360] | not confirmed |
| G1s2 | 0.0012 | 0.0112–0.0239 | 0.0135 [0.0093, 0.0184] | not confirmed |
| G2 | 0.0006 | 0.0045–0.0079 | 0.0055 [0.0034, 0.0083] | not confirmed |
| G2s1 | 0.0015 | 0.0122–0.0327 | 0.0160 [0.0119, 0.0210] | confirm |
| G2s2 | 0.0012 | 0.0107–0.0294 | 0.0140 [0.0098, 0.0188] | confirm |
| G3 | 0.0006 | 0.0082–0.0126 | 0.0091 [0.0072, 0.0118] | not confirmed |
| G3s1 | 0.0025 | 0.0324–0.0484 | 0.0322 [0.0248, 0.0402] | confirm |
| G3s2 | 0.0006 | 0.0064–0.0102 | 0.0082 [0.0063, 0.0113] | not confirmed |

- **Values are pinned more tightly than gradients: confirm in all 9.** Computed here from `D`: every gradient component's D is 7.8 to 23.9 times D_V.
- In absolute terms the heads disagree little: D is at most 0.0025 for V and 0.048 for any gradient. They share a stop-gradient trunk, which hides indeterminacy (design Section 6).
- The other 22 generalists (`descriptive_networks.<run>["C1-heads"]`, descriptive): D_grad(K_P) − D_V confirms in 22 of 22 (diff 0.0073–0.0294; D_V 0.00028–0.0022), and the secondary comparison in 13 of 22. S1–S3 have no heads.

**C1-braking** (`networks.<run>["C1-braking"]`; C1's rule on the braking keys, 3,626 interior rows, s 1,393 and ξ 1,333 scored): **refute** in all 9. R²_c −2.928 to −0.018; CI upper ends at most 0.005 (G1). It is not separate evidence about braking. Under K2b it scores no braking-phase row (`frozen.k2b_excluded.braking` holds every K_P component), so its rows are the approach and exit rows of braking keys. Its stratum is 92% of C1's (3,626 of 3,951 interior rows), and its scored rows are 85% of C1's on s (1,393 of 1,643) and 84% on ξ (1,333 of 1,581). It largely repeats C1 and does not test the braking-phase sign flips its theory is about (design Section 6).

**C1-limit** (`networks.<run>["C1-limit"]`): **not testable** in all 9. T has 482 wall-contact rows. n keeps none (K2b), and ξ keeps 2 approach rows in one geometry, which some bootstrap draws do not hold, so no CI exists ("no component keeps contact rows that every bootstrap draw holds"). Design Section 14 foresaw this.

**C1-pattern** (`networks.<run>["C1-pattern"]`; the K_P side is {s, ξ}, the other side {vy, r, δ, pedal} on 3,231–3,951 rows): **confirm for G1 only**, not confirmed for the other 8.
- G1's difference is 0.150 [0.021, 0.326]. Its K_P mean is 0.010, against −0.140 on the other side (vy −0.047, r −0.292, δ −0.0005, pedal −0.222). So it records that the fast components score worse, not that K_P scores well.
- Elsewhere the difference runs from −2.395 to 0.046, driven by ξ.

**W1** (`networks.<run>.W1`; the critic's uniform inside-widening derivative against Π^in on the interior, 3,951 rows; no band):

| Network | R²_c [CI] | Spearman [CI] | Pooled slope of Π^in on the critic's price [CI] |
|---|---|---|---|
| G1 | 0.010 [−0.074, 0.056] | 0.31 [0.21, 0.40] | 0.22 [0.15, 0.29] |
| G1s1 | 0.032 [−0.072, 0.077] | 0.27 [0.15, 0.38] | 0.15 [0.08, 0.25] |
| G1s2 | 0.021 [−0.101, 0.085] | 0.33 [0.20, 0.44] | 0.19 [0.11, 0.28] |
| G2 | 0.010 [−0.083, 0.040] | 0.23 [0.10, 0.32] | 0.17 [0.10, 0.23] |
| G2s1 | −0.024 [−0.112, 0.022] | 0.22 [0.10, 0.32] | 0.17 [0.07, 0.24] |
| G2s2 | 0.061 [−0.049, 0.134] | 0.37 [0.28, 0.44] | 0.22 [0.16, 0.29] |
| G3 | 0.256 [0.121, 0.337] | 0.52 [0.37, 0.65] | 0.36 [0.24, 0.47] |
| G3s1 | −0.050 [−0.122, −0.026] | 0.11 [−0.00, 0.22] | 0.09 [0.04, 0.16] |
| G3s2 | 0.114 [0.007, 0.176] | 0.37 [0.27, 0.47] | 0.33 [0.22, 0.44] |

The critics' inside-wall price ranks with the optimum's (Spearman 0.22–0.52 with CI lower ends above 0 in 8 of 9), but its calibrated agreement is near 0 except in G3 and G3s2, and G3s1's CI lies below 0.

## 6. Validity evidence

### 6.1 Pre-tag gates (the canonical pass at `a616688`)

Sources: `outputs/costate/gates/*.json`, bound in the record. Every gate's verdict is pass (K0's in `summary.pass`, K1's in `decision.pass`, the others' in their top-level `pass`); K6 counts coverage and has no verdict.

| Gate | What it checks | Result |
|---|---|---|
| K0 (`k0.json` `summary`) | Targets from the key files, no solves: x_0 identity, route agreement, pedal fold, action rows, straight-road identity | Pass on its scope: 511 keys (the 481 A2 keys and K1's sample), every one ok. 8,617 keys checked (481 A2, 8,136 anchored finals). Out of scope, 6 anchored finals of N fail only the pedal-fold stationarity (2.4·10⁻¹⁰, 1.2·10⁻⁹, 1.6·10⁻¹⁰, 7.4·10⁻¹⁰, 2.2·10⁻¹⁰, 2.0·10⁻¹⁰ against 10⁻¹⁰), listed in `not_ok_descriptive`. Every other key is at most 1.0·10⁻¹⁰ (computed here: 9.98·10⁻¹¹). Action-row identity at most 5.8·10⁻¹⁰ (limit 10⁻⁹); route gap 7.0·10⁻¹¹; end gap 5.5·10⁻¹¹; every planted action-row bug fails (smallest 2.3·10⁻⁴, limit 10⁻⁴) |
| K1 (`k1.json` `decision`) | λ against Richardson differences of re-solves: 50 keys, 2,008 comparisons; π on 10 anchored keys | Pass, with every plant failing. Per component, median 1.1·10⁻⁷ to 2.2·10⁻⁵ and p95 2.9·10⁻⁶ to 3.5·10⁻³ (rule: median < 10⁻³, p95 < 10⁻²); kink shares 0.060–0.064; 0 failed solves. π: 12 non-zero and 8 zero comparisons, median 8.9·10⁻⁷, p95 7.0·10⁻⁶. Reported, binding nothing: Gate 1's 10⁻⁶ zero rule fails on δ (p95 0.042), and design Section 7's literal kink test leaves all 8 components descriptive (kink shares 0.40–0.49) |
| K2 (`k2.json`) | Horizon: W = 400 against W = 800 | t_H 267 (first divergence 268); T's largest t is 129, so nothing is dropped; no fallback key. 3 of the 30 insurance solves were problematic (2 in a different basin, 1 at the iteration limit), as in the earlier checks |
| K2b (`k2b.json`) | Road beyond the preview | 26 keys; the exclusions of Section 3.2 |
| K3 (`k3.json`) | Pullback and wiring on random and planted networks | 22 of 22 checks. Pullback against finite differences: held and x_prev forms at most 9.0·10⁻⁸ (limit 10⁻⁶), on-manifold 1.30·10⁻⁶ (limit 10⁻⁵). Every one of 24 planted wrong pullbacks is off by at least 0.030 (limit 10⁻⁴) |
| K3b (`k3b.json`) | The policy's own costate loop | Pass; trace error 2.2·10⁻¹⁴, restart error 1.2·10⁻¹⁴; all three planted bugs fail; share of comparisons counted 0.667 (on-manifold 0.833) |
| K4 (`k4.json`) | Planted representations through the whole probe pipeline (decoy_k 8) | Every plant passes. PR-1a within 0.006 of SNR/(1 + SNR); PR-1b partial R² 0.786–0.803 (0.8 ± 0.03); PR-2 −0.0001; PR-3 grouped 0.0037, while the mis-grouped split shows its leak (0.022); PR-4 0.979 and 0.978 against the readout's 0.916; PR-5: zero, noise, −λ and direction-only gradients refute (R²_c −0.0033 to −0.0027), SNR 4 confirms with the label, SNR 0.25–1 inconclusive; PR-6 −0.0083; PR-6b and PR-6c leak bounds 0.0033 and 0.0069; PR-7 5.4·10⁻¹⁵; PR-8 0.0815 against 0.0860; PR-9 D_V 2.3·10⁻³³ |
| K5 (`k5.json`) | Wiring with permuted targets | Points −0.0028 to −0.0012, every lower end ≤ 0 and point ≤ 0.01. The control task (descriptive) has ceilings 0.946–0.947 |
| K6 (`k6.json`) | Coverage | T 9,887 rows; interior 0.385; contact 831; pedal-saturated 1,911; pedal-switching 2,911; κ-clipped 233; s unobservable 1,983 rows |
| decoy-k (`decoy_k.json`) | f̂_D's k re-derived on the canonical world | Of (3, 5, 8, 15) only 8 qualifies; the validation table reproduces to 4.9·10⁻⁷ (limit 10⁻⁶) |
| c2-null (`c2_null.json`) | C2's statistic on 20 random-init hosts | Section 6.4 |

### 6.2 Trained-network checks after the tag

- **K3 after the tag** (`post/critic/*.k3.json`, 34 files; `decisions.json` `k3_trained`): 34 of 34 pass. The float64 extractor matches Flax to at most 3.6·10⁻¹⁵ (limit 10⁻¹⁰) for the actor's mean, the critic's V and the heads. Float64 V against float32 V on D's 477 states differs by at most 1.4·10⁻⁶ (limit 10⁻⁴; relative 7.7·10⁻⁷). Each of the 31 generalists has 5 heads, S1–S3 none.
- **K3b's trained analogue** (`post/own/*.k3b.json`, 9 files): 9 of 9 pass, and every planted bug fails in each. The share of comparisons counted is 0.55–0.75 (on-manifold 0.50–0.95).

### 6.3 Value accuracy

Source: `networks.<run>.value_accuracy` (decides C1-own's validity) and `value_accuracy_vstar` (descriptive). The target is the deterministic policy's own discounted return from each state, by its own closed loop up to 1,000 steps, on D's 477 states and 813 interior restarts (1,290 asked); runs that crashed are dropped. R² is out of fold, after C1's per-fold affine calibration of the target on σ_R·V_c. Threshold 0.8.

| Network | R² [95% CI] | Scored of 1,290 | Spearman | R² against V*_∞ (descriptive) |
|---|---|---|---|---|
| G1 | 0.898 [0.827, 0.942] | 1,280 | 0.95 | 0.810 |
| G1s1 | 0.932 [0.900, 0.952] | 1,278 | 0.97 | 0.842 |
| G1s2 | 0.924 [0.894, 0.944] | 1,243 | 0.97 | 0.805 |
| G2 | 0.913 [0.878, 0.936] | 1,255 | 0.95 | 0.849 |
| G2s1 | 0.804 [0.609, 0.906] | 1,272 | 0.92 | 0.755 |
| G2s2 | 0.888 [0.843, 0.918] | 1,249 | 0.95 | 0.817 |
| G3 | 0.849 [0.810, 0.881] | 1,288 | 0.94 | 0.885 |
| G3s1 | 0.810 [0.727, 0.863] | 1,282 | 0.91 | 0.729 |
| G3s2 | 0.834 [0.760, 0.880] | 1,262 | 0.92 | 0.835 |

- **Pass in all 9.** Every requested row was scored (`n_missing` 0); 2–47 rows per network were dropped as crashed runs.
- The fit is in shape, not in level. The per-fold calibration slopes are 0.98–1.15 and the intercepts 4.1–18.1 reward units, and the uncalibrated R² is −5.4 to −2.0 (`raw_r2`).
- Against the optimum's V*_∞ on T's interior (3,951 rows, descriptive), R² is 0.729–0.885; G2s1 and G3s1 fall below 0.8.

### 6.4 C2's random-only null

Source: `gates/c2_null.json` (`summary`, `rows`), copied into the record (`frozen.c2_null`); `networks.<run>.C2.c2_null_note`. Each of 20 random-init actors (seeds 20–39) was scored as the network against the pool (0–19), with C2's rows, mask, floor and two-level CI, under G1's observation scaling.
- Mean −0.0090, SD 0.0199, median −0.0107, range −0.0403 to 0.0288.
- Ruled decisions: 3 refute, 17 inconclusive, 0 confirm (full sets 5, 15, 0); two hosts were geometry-sensitive. Two hosts (seeds 26 and 36) have two-level CIs wholly above 0 (`summary.ci_above_0`), and host 26's set without geometry 6 confirms (0.115 [0.049, 1.058]; design 17.4).
- **Where the trained networks sit.** G1, G1s1, G1s2 and G3s1 (points 0.030–0.039) lie above every null host. G2 is above 18 of 20, G2s1 above 14, G3 above 13, G3s2 above 8 and G2s2 above 1. The null is descriptive, and its scope is G1's scaling; each trained network's z is under its own.

### 6.5 Integrity of the run

- `git diff costate-prereg-v1 9b25cc2` touches only `outputs/costate/tag.json` and `outputs/costate/analyse/decisions.json`. No code, test or design text changed after the tag.
- No deviation was logged: `outputs/costate/deviations.jsonl` does not exist, and every post-tag stamp and `decisions.json` record deviation null.
- The 23 pre-tag outputs bound by the record and the 113 inputs bound by `decisions.json`, 127 distinct files (the 9 null arrays are bound by both), all match on disk (checked here). `decisions.json` binds the post-tag outputs the analysis reads (104 of the 206 files under `post/`), not every post-tag output (Section 1).
- Neither R_TEST nor `outputs/sealed/` was read for this document. K2b's strips are synthetic and behind the lockbox guard (design Section 7).

## 7. Descriptive and exploratory results

### 7.1 Why λ_ξ's R²_c lies far below zero (exploratory, computed here)

In 7 of 9 networks λ_ξ's R²_c is between −5.0 and −0.32 (Section 5.2), while K4's zero and noise plants score −0.0031 and −0.0033 on ξ (`gates/k4.json`, PR-5 rows 'zero' and 'noise', `r2c[2]`). Yet ξ's rank agreement is positive in those networks. The following was computed from the bound arrays to see why.

**Method.**
- Rows: the 1,581 ξ decision rows (`tab/interior` and `tab/phase` = approach in `outputs/costate/inputs/world.npz`).
- λ_ξ: `tab/lam_xi`, clipped to the frozen bounds (`frozen.winsor_bounds.lam_xi`, −21.07 to 21.54).
- Weights: key weights renormalised on the interior (each key equal, states equal within a key).
- g_ξ: `post/critic/<run>.npz` `g[:, 0, 2]`, the main V's pullback.
- Predictions: the record's per-fold intercepts by direction and slopes (`networks.<run>.C1.calibration.a`, `.b`), each row by its outer fold (`frozen.fold_partition0[tab/geometry]`), clipped to the bounds.
- Recomputed this way, R²_wd equals `networks.<run>.C1.r2c[2]` to 1.0·10⁻¹⁴ in all 9.
- Tail: rows whose g_ξ, centred on its key-weighted median within each turn direction, lies more than 3 robust SDs (key-weighted MAD/0.6745 of the centred values) from 0. This is the calibration's leverage cut (`analysis.cal_weights`, K_LEV = 3), applied to all 1,581 rows at once rather than per training fold.

**λ_ξ on these rows.** It is positive on 689 of the 745 rows turning one way (direction −1) and negative on 764 of the 836 turning the other way (direction +1); the unweighted means are 0.684 and −0.583.

**g_ξ is heavy-tailed.** Its robust SD is 5.6–8.9, while its 0.5th and 99.5th percentiles lie at −567 to −248 and 290 to 757.

| Network | Slope b_ξ, fold mean | Tail rows (share) | Tail weight share | Tail share of ξ's squared error | R²_c ξ on the bulk | sign(g_ξ) opposite to λ_ξ: tail / bulk | var(prediction)/var(λ_ξ), within direction | Geometry 38's share of ξ's squared error |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.0035 | 286 (0.181) | 0.216 | 0.099 | −0.017 | 0.88 / 0.52 | 0.07 | 0.013 |
| G1s1 | 0.0319 | 247 (0.156) | 0.179 | 0.649 | 0.078 | 0.92 / 0.47 | 1.79 | 0.102 |
| G1s2 | 0.0325 | 259 (0.164) | 0.199 | 0.774 | 0.065 | 0.88 / 0.41 | 3.24 | 0.213 |
| G2 | 0.0037 | 283 (0.179) | 0.226 | 0.093 | −0.017 | 0.90 / 0.56 | 0.05 | 0.009 |
| G2s1 | 0.0359 | 315 (0.199) | 0.239 | 0.862 | 0.065 | 0.89 / 0.36 | 5.37 | 0.193 |
| G2s2 | 0.0146 | 229 (0.145) | 0.171 | 0.313 | −0.012 | 0.89 / 0.41 | 0.54 | 0.146 |
| G3 | 0.0394 | 240 (0.152) | 0.192 | 0.775 | 0.092 | 0.87 / 0.51 | 3.22 | 0.245 |
| G3s1 | 0.0303 | 336 (0.213) | 0.244 | 0.864 | 0.065 | 0.90 / 0.54 | 5.60 | 0.222 |
| G3s2 | 0.0337 | 294 (0.186) | 0.198 | 0.776 | 0.053 | 0.86 / 0.43 | 3.19 | 0.208 |

- **The tail has the wrong sign.** In every network, 14–21% of the rows lie in the tail, and there g_ξ has the sign opposite to λ_ξ on 86–92% of rows (on the bulk 36–56%).
- **The slope is set on the bulk.** On the bulk alone R²_c is −0.017 to 0.092. The leverage weights cap the tail's influence on the slope, so b is fitted mostly on the bulk, where g and λ agree weakly or not at all: bulk R²_c is at or below 0 in G1, G2 and G2s2, and the bulk's signs are opposite to λ_ξ's on 36–56% of rows, above half in G1 (0.52), G2 (0.56), G3 (0.51) and G3s1 (0.54).
- **Applied to the tail, it overshoots.** In the seven networks with slopes 0.015–0.039, the predictions' within-direction variance is 0.54 to 5.6 times λ_ξ's, and the tail carries 0.31–0.86 of ξ's squared error. Up to 4.3% of their predictions reach the clipping bounds (`calibration.clip_share[2]`).
- **G1 and G2 escape only through their small slopes** (0.0035 and 0.0037, 4–11 times smaller than the others'). Their predictions stay near the direction intercepts (variance ratio 0.07 and 0.05), and R²_c is about 0. Their tails have the same opposite sign.
- **Part of the tail is one geometry.** Geometry 38 (R 44.7 m, θ 180°) holds 1.1% of λ_ξ's within-direction total sum of squares (`logo.shares.xi.sst`) and 4.2% of the weight, but 10–25% of ξ's squared error in the seven networks (`networks.<run>.C1.logo.shares.xi`, which equal the last column). It is the most influential geometry of C1 in those seven (Section 5.2).
- **Weight sensitivity** (descriptive; `networks.<run>.by_stratum.C1["interior&phase:approach"]`). Scoring the same 1,581 ξ rows with key weights renormalised on the approach rows, instead of on the interior, changes G1's ξ R²_c from −0.002 to −0.307 and its Spearman from 0.05 to 0.20. The primary statistic is sensitive to how weight is spread over keys on these rows.

So the very negative R²_c does not mean a gradient unrelated to λ. In every critic a minority of states carries a large ξ gradient of the wrong sign. The calibration's slope, fitted on the bulk, turns those into large errors. Where the slope is small, the same tail makes R²_c about 0 instead of positive. The tail explains the very negative magnitudes, not the refute: without it, bulk R²_c (at most 0.092) is still far below the 0.2 band. None of this was pre-registered, and the tail's origin (which states, and why the gradient is large there) was not examined.

### 7.2 The strata tables (descriptive)

Source: `networks.<run>.by_stratum`. Each row is scored on its own stratum with key weights renormalised there; the C2 and C2-sw rows other than the primary use the 5 controls (seeds 0–4) and geometry-only CIs (design 16.3). They decide nothing. Bands in network order (G1 … G3s2): R refute, I inconclusive, C confirm, N not testable.

| Stratum (rows) | C1 | C2 |
|---|---|---|
| interior, the decision stratum (3,951) | RRRRRRRRR | IIIIIRIIR |
| approach phase (1,581) | RRRRRRRRR | IRRRRRRRR |
| braking phase (871) | NNNNNNNNN | NNNNNNNNN |
| corner phase (1,427) | NNNNNNNNN | NNNNNNNNN |
| exit phase (72) | RRRRRRRRR | IIIIICICI |
| braking keys (3,626) | RRRRRRRRR | IIIIIIIII |
| non-braking keys (325) | RRRRRRRRR | ICIIIIIIC |
| W = 400 (2,596) | RRRRRRRRR | IIIIIIIIR |
| W = 800 (1,355) | RRRRRRRRR | IICIIICCC |
| pedal-saturated (720) | RRRRRRRRR | CIICICIII |
| interior with k = 20 (1,938) | RRRRRRRRR | IRRIIIIII |
| near-limit (5,936) | RRRRRIRRR | RIIRRRIIR |
| contact (831) | IIIIIIIII | IIIIIIIII |

- **C1 refutes in every stratum where it can be scored**, except contact (5 scored rows) and G2s2's near-limit row. The braking and corner phases, which the question is most about, cannot be scored (K2b).
- **C2's 11 confirm-band strata rows** lie in small or few-geometry strata: W = 800 (1,355 rows, 19 geometries), pedal-saturated (720, 17), non-braking keys (325, 10) and exit (72, 12). Their CI upper ends run from 1.46 to 147.5, and on s the base probe's R² there is −2.81 to −0.29: the base does not transfer to those strata, so gains are measured against a failing base.
- **C2-sw** refutes on the pedal-switching steps of the approach (976) and exit (1,531) parts in all 9. Confirm-band rows are the non-braking keys (558 rows, 10 geometries) for G1s2, G2s1 and G3, and the braking phase (86 rows) for G1s1.

### 7.3 Sensitivity variants and robustness (descriptive)

Source: `networks.<run>.sensitivity` (16 variants: partitions 1–10, two weight schemes, small geometries dropped, κ set dropped, braking and non-braking keys; C2 and C2-sw with 5 controls and geometry-only CIs), `robustness_C1`, `robustness_C2`, `k4_band_notes`.
- **C1 refutes in all 144** network × variant records (statistic −4.137 to 0.015).
- **C2:** 121 inconclusive, 15 refute, 8 confirm. The confirms are G1 (partition 10, 0.089), G1s1 (non-braking keys, 1.567 on 325 rows; equal weights per geometry, 0.074; equal weights per state, 0.059), G1s2 (small geometries dropped, 0.054), G3s1 (partition 10, 0.080) and G3s2 (non-braking keys, 1.321; partition 2, 0.110). K4's leak-bound note stands beside two of them (G1s1 weights:state, G1s2 drop_small).
- **C2-sw:** 114 refute, 27 inconclusive, 3 confirm, all on the non-braking keys (558 rows): G1s2 0.238, G2s1 0.255 and G3 0.206.
- **Partitions 1–10:** C1's statistic varies with SD 0.011–0.083 per network, C2's over −0.221 to 0.110.

### 7.4 The other 22 generalists and the specialists (descriptive)

Source: `descriptive_networks.<run>.C1` (no nulls, label scope descriptive).
- **Other generalists:** every one is in the refute band. Their R²_c is −2.853 to −0.011, and λ_ξ's R²_c is negative in all 22 (below −0.3 in 19).
- **Specialists** (Hockenheim-trained, scored on the same synthetic corners): S1 0.065 [−0.022, 0.092], S3 0.053 [−0.030, 0.082] and S2 0.007 [−0.073, 0.026]. S1 and S3 have the two highest C1 R²_c of all 34 networks; S2 is fourth, after G1. All three have a positive ξ R²_c (S1 0.159, S2 0.044, S3 0.124; Spearman 0.30, 0.14, 0.29), and all three are in the refute band.

### 7.5 The accel-held forms (descriptive)

The accel-held pullback (`networks.<run>.sensitivity["pullback:held"]`) and C1-own's accel-held form (`["C1-own_held"]`) were pre-registered as sensitivities (design Sections 3.2 and 14). Design Section 14 predicted they cannot differ from the primary here, because the accel term reaches only vx, vy, r, δ and pedal, none of which is scored. Computed here: over the statistic, CI, R²_c, Spearman and the two pooled slopes (`slope`, `slope_gl`) they differ from C1 by at most 3.6·10⁻¹⁵ (slope_gl's; the inverse-slope CI, `inv_slope_lg_ci`, by up to 8.5·10⁻¹⁴) and from C1-own by at most 1.2·10⁻¹⁴.

## 8. What it means

**On the states where it could be decided, the critics' pulled-back value gradients do not match the window problem's λ_s and λ_ξ.**
- All nine critics refute C1, on λ_s and λ_ξ at approach-phase interior states (and 62 exit states for λ_s), with CI upper ends at most 0.025 against a band of 0.2. C1 tested one readout: the main value head's input gradient, pulled back on the manifold and scaled by σ_R, under the per-fold leverage-weighted affine calibration (Section 9, 'Other forms of λ in the critic').
- The best calibrated agreement, G1's 0.010, is close to what an intercept alone scores (−0.013).
- On ξ, a phase-matched λ borrowed from another corner, and the best of 20 random-init critics (selected as the best of 20 on these rows, the design's sanity floor), score higher than every critic. On the K_P mean the far-geometry λ is significantly above all 9 critics and the best random critic above 7. No critic but G1's beats any value-fit, QSS or barrier gradient (paired CI), and G1's does not beat the value MLP.
- The gradients are not unrelated to λ: rank agreement has a CI lower end above 0 in 6 networks per component (ρ up to 0.40). But no per-fold linear calibration fitted on other corners transfers.
- Exploratorily, a wrong-sign tail of large gradients, present in every critic, carries 0.65–0.86 of ξ's squared error in six networks and 0.31 in G2s2 (Section 7.1). The refute does not depend on it, since ξ's bulk R²_c is at most 0.092.

**Values fit, gradients do not.**
- On D's states and T's interior restarts (all phases, not C1's decision rows: for G1, 650 of the 813 restarts asked are braking- or corner-phase rows, computed here from `post/own/<run>.npz` `rows`), the critics predict their own deterministic policy's return with R² 0.80–0.93 after an affine calibration (Section 6.3); G2s1 and G3s1 pass at the point estimate, with CI lower ends 0.61 and 0.73. So design Section 9's "values fitted, gradients not" reading is allowed.
- In the 9 decision networks, the heads' disagreement on each gradient component is 7.8 to 23.9 times their disagreement on V. C1-heads' comparison confirms in all 9 decision networks (secondary), and in all 22 other generalists (descriptive).
- C1-own refutes too: the gradient does not match the critic's own policy's costate either, so the gap is not only the optimum-versus-policy gaps of design Section 3.3. But λ^own still differs from what the critic estimates through the stochastic policy, grip jitter, crash risk, the terminal penalty, the horizon and survivor selection (design 3.3), so C1-own does not isolate gradient indeterminacy. C1-own rests on about 162 rows per component with wide CIs.
- Design Section 9 offers two readings of a refute. "Values accurate and D_grad high: the regression does not determine the gradient" fits the comparison. But the heads' gradient disagreement is small in absolute terms (D at most 0.048), and their shared trunk hides indeterminacy, so the heads cannot settle whether the main critic's gradient is determined. The data are consistent with the design's first reading and do not establish it.

**The actor: nothing beyond the input detected at the band, and C2-sw refutes by its band.**
- C2 confirms nowhere and refutes in 2 of 9. Mostly it is inconclusive, with wide CIs (two-level widths 0.24–0.50). One geometry (6) carries about two thirds of the base's error, which design 17.6 traces to the decoy residualisation.
- G1, G1s1, G1s2 and G3s1 are inconclusive, which is evidence neither for nor against C2. Descriptively, their points (0.030–0.039) lie above every random-only null host (`c2_null_note.share_at_or_above` 0), three of them (G1s1, G1s2, G3s1) mostly through λ_s and G1 through λ_ξ. The null uses G1's observation scaling only (`c2_null_note.scope`), so it fits G1 alone of the four. 2 of its 20 random hosts also have two-level CIs wholly above 0, and host 26's set without geometry 6 confirms (0.115), as G1s1's does (0.102) (Section 6.4). On the approach-phase stratum (descriptive: the same 1,581 ξ rows, s without its 62 exit rows, weights renormalised on approach, 5 controls), the C2 point is negative in all 9 (G1 −0.001, G1s1 −0.006, G1s2 −0.220, G3s1 −0.123) and refutes in 8 (`by_stratum.C2["interior&phase:approach"]`). The 72-row exit stratum (62 scored on s) gives gains of −0.18 to 1.34 against a base R² of −0.91 to −0.82 (`by_stratum.C2["interior&phase:exit"]`). So the positive points do not survive on the approach rows alone.
- C2-sw refutes everywhere by the band (every CI upper end below 0.05). Descriptively (design 16.3), the trained h3's gain ranks first of 21 in 7 of 9 networks; the extra over the pool's median is 0.006–0.030 of R² on a target that z already explains at 0.73. σ^ped is not residualised on position and speed, and units that hold those decoys could raise its gain by their whole predictive share (design 15.10), so this does not isolate the switching function, which design Section 9 claims only with a C2-sw confirm. Under the pre-registered band, "refute" here means a gain below 0.05, not a gain of 0.

**Wall-price fidelity does not go strongly with completion** (C3-w). The critics' inside-wall price ranks with the optimum's, but its calibrated agreement is near 0 except G3 (0.256) and G3s2 (0.114); W1 has no band. C3-R²c's direction is negative.

**The matching intervention** (design Section 9) is a value-gradient term in the critic's loss (DHP, GDHP or VGL), aimed at the action-projected gradient error. Nothing here tests whether that would change driving.

## 9. What it does not mean (limits)

- **A narrow decision set.** C1 and C2 were decided on λ_s and λ_ξ only, at the interior's approach-phase states (and 62 exit rows for s). Not on λ_n, which carries the wall price at the state level; not on λ_vx, the speed price; and not on braking or corner states. K2b excluded them because the window problem's λ there depends on road beyond the preview, which no critic sees (Section 3.2). Whether the critics hold the costate in braking and corner states is not addressed.
- **The states are the optimum's**, on synthetic A2 corners (mostly 12 m wide), not the agents' own states and not real roads. Final checkpoints only; one PPO configuration; 200M steps.
- **Other forms of λ in the critic.** C1 tests the main head's input gradient under this calibration, against the window problem's λ_s and λ_ξ. Probes of the critic's layers, per-head R²_c and the action-projected critic signal q_c were not computed (design Section 14, 'Not in the confirmatory run'), so whether the critics hold λ in another form is untested.
- **Calibration dependence.** R²_c is defined through a per-fold, leverage-weighted affine calibration with a clamped slope, fixed before the tag. It is not the calibration design Section 6 first specified: Section 6's per-fold Huber affine fit was replaced in Section 13 (v2.2), before any trained network was computed, by this leverage-weighted least-squares slope (Section 3.5). Section 7.1 shows that this calibration, applied to a heavy-tailed gradient, makes ξ's score very negative. Gate PR-5e held zero, noise, direction-only and anti-aligned gradients in [−0.02, 0.005] (`gates/k4.json` `PR-5.PR-5e`). Its noise plant is Gaussian (`gates.pr5`), and it did not test a heavy-tailed gradient, or one whose bulk and tail disagree in sign, as in Section 7.1. The scores also move with the key weighting on the same rows. Another reasonable calibration could score the bulk differently. The refute is a statement under this rule, and the rule was fixed in advance.
- **Decoy dependence (C2).** C2's target is λ minus f̂_D, a kNN on position, speed and geometry with k = 8, chosen by a declared rule after k = 15 and a nested-CV k had failed known-truth worlds (15.7). The fixed-k rule was decided with a descriptive k = 15 and k = 5 diagnostic on PR-6b's gate seeds already known, and the gate seeds had been scored at k = 8 before the rule was re-applied (design 15.1, 15.7).
  - PR-6c bounds what f̂_D's smoothing bias can leak into C2 at 0.0069 on the gate seeds; on the validation seeds its largest value at k = 8 is 0.0098, against the rule's 0.01 (design 15.6, 15.7).
  - A layer that encodes the decoy variables, which its random controls lack, loses about 0.01 of Δ2. That makes a confirm harder and a refute slightly easier (15.5).
  - K4 does not detect a D₁ without s − s_apex (15.6).
  - Geometry 6's extreme residual is created by f̂_D's fit, not by the probe (17.6).
- **The power floor narrows C2 to s and ξ.** At k = 8 λ_n's residual share is 0.091, below 0.10, and λ_vx's 0.076. Under k = 15 n would have passed (0.107), but it has no scored row under K2b anyway.
- **C2's probe and gates.** C2 is a linear (ridge) probe of h3 only; probes of h1 and h2 and MLP probes were not run (design Section 14). PR-1b checks the ridge machinery, not the cross-fit, and PR-6's residualisation-off check is marginal (0.00999 against 0.01; design 15.10). K5 does not detect a leak confined to the intercept or the fold means, which is bounded by the between-fold share of the target's variance (at most 0.002 for λ_s; design 15.3). For a network no different from a random one, the pool-median statistic is centred about 0.01 below 0 (the null's mean −0.0090, median −0.0107), because random gains are left-skewed against the median (design 16.5). The coverage figure below comes from two random hosts.
- **Geometry 6.** In every trained network it holds 64–71% of C2's base squared error on 5% of the weight. On two random hosts (seeds 26 and 30, by a double bootstrap) the two-level CI covered only 0.68–0.79 where a decision would fire (16.3). The single-geometry rule guards against a decision resting on it, but only by withdrawing; it cannot turn an inconclusive into a confirm (G1s1's set without geometry 6 confirms). The ruled decisions' coverage was not measured.
- **The one-geometry rule** changed no ruled decision here (`geometry_sensitive` empty in all 27 ruled records). It acts as a margin around each band of about the sets' spread (17.8).
- **C2-sw's band against its reliability.** C2-sw refutes because its gain is below 0.05, while the gain ranks above the random pool in 7 of 9 (descriptive). No between-network null exists for C2-sw. C2's null was measured under G1's observation scaling only. C2-sw's two-level CI has no measured coverage on its rows (on C2's rows it covered 0.68–0.79 on two random hosts). Its target is not residualised on position and speed, and K4's decoy-leak bounds, measured on C2's setup, do not carry over to it (design 15.10, 16.3).
- **K0's scope** was narrowed after its failure in the canonical pass was seen (design Section 18), where v2.4's and v2.7's stated procedure had that failure block the tag (Section 3.5). Six of N's anchored finals fail the pedal-fold stationarity tolerance. No pre-registered computation reads them, because N's analyses are deferred. If N is analysed later, those keys must be left out or re-polished.
- **The other pre-tag revisions** (design Sections 15–17) were made after failures on random-init or planted networks had been seen: K1's floor, K5's scheme, PR-6 one-sided and paired, PR-3 paired, f̂_D's k, the control pool and the one-geometry rule. No trained network had been computed, so no decision here could have shaped them. But the gates' passing rests partly on rules written after their own failures, as the design records.
- **Deferred analyses** (Section 3.4) were not run. Among them are C3-int, the most direct behavioural link, C1-dev and N's replication. Any of them run later is exploratory.
- **C3's power.** With 31 networks in 7 strata, C3-w rules out only a strong association (80% power at a latent ρ of about 0.6). A moderate association is not excluded.
- **Value accuracy is affine.** The per-fold calibrations have intercepts of 4.1–18.1 reward units, and the uncalibrated R² is −5.4 to −2.0. "Values fit" means in shape after a per-fold affine map.
- **Nothing about R_TEST.** It stays sealed and was not used.

## 10. What next

1. **The ξ tail (exploratory; write and commit a plan first).** Find which states carry the large wrong-sign ξ gradients (geometry 38 holds 10–25% of the error), and whether they sit at observation-clip, preview-spacing or tyre-channel kinks of the pullback. Report R²_c under a rank-based calibration, and under the Huber calibration that was design Section 6's original choice (replaced in Section 13), beside the frozen one, descriptively. This needs only the bound arrays.
2. **The deferred analyses that need no new solves** (exploratory; plan first): C1-dev over the 16 snapshots (does gradient agreement change with training?), ΔC1, C2-pairs and C3-int.
3. **A random-only null for C2-sw**, as c2-null is for C2, to say how far the trained h3's gain over the pool lies above random-init variation. For C2, a descriptive score with a decoy model that fits geometry 6.
4. **Braking and corner states.** A target whose window ends within the preview, or a preview-matched window problem, would remove K2b's reason to exclude these phases and λ_n and λ_vx. It needs new solves and a new design, pre-registered if it is to decide anything.
5. **An intervention** (design Section 9). Add a value-gradient term to the critic's loss (DHP, GDHP or VGL), aimed at the action-projected gradient error. Test it like H4R: pre-registered, one machine and one commit, with C1's R²_c and R-dev completion as outcomes.

## Appendix A. Sources of the key numbers

`decisions.json` is `outputs/costate/analyse/decisions.json` (committed in `9b25cc2`); `prereg.json` is `src/apexzero/costate/prereg.json` (committed in `11dc9bc`); gate files are under `outputs/costate/gates/` (local, bound by sha256 in `prereg.json` `outputs`).

| Quantity | File | JSON path |
|---|---|---|
| Run provenance (argv, times, HEAD, code hash, content_sha256, deviation, partial) | `decisions.json` | `provenance` |
| Tag pin | `decisions.json`; `outputs/costate/tag.json` | `tag_pin` |
| Confirmatory, decided, unlocked | `decisions.json` | `confirmatory`, `decided`, `unlocked_by`, `deviation` |
| Per-network decisions | `decisions.json` | `per_network.<G1…G3s2>` |
| Variant decisions | `decisions.json` | `variants.<v1,v2,v3>.<C1,C2,C2-sw>` |
| C1 statistic, CI, decision, full-set decision | `decisions.json` | `networks.<run>.C1.stat`, `.ci`, `.decision`, `.decision_all_geometries` |
| C1 per component, Spearman, slopes | `decisions.json` | `networks.<run>.C1.r2c`, `.spearman`, `.spearman_ci`, `.slope`, `.slope_gl` |
| C1 nulls | `decisions.json` | `networks.<run>.C1.null_r2c.<null>`, `.null_diff.<null>` |
| C1 calibration (per-fold a, b, clip shares) | `decisions.json` | `networks.<run>.C1.calibration` |
| C1 leave-one-geometry-out sets, shares | `decisions.json` | `networks.<run>.C1.logo` |
| Decision rows per component | `decisions.json` | `networks.<run>.decision_rows` |
| C2 statistic, CIs, gains, pool rank | `decisions.json` | `networks.<run>.C2.stat`, `.ci`, `.ci_geometry`, `.gain`, `.pool` |
| C2 null note | `decisions.json` | `networks.<run>.C2.c2_null_note.share_at_or_above` |
| C2 geometry 6 shares and sets | `decisions.json` | `networks.<run>.C2.logo.shares.<lam_s,lam_xi>`, `.logo.stat`, `.logo.ci`, `.logo.decision` |
| K4 band notes placed | `decisions.json` | `networks.<run>.k4_band_notes.placed` |
| C2-sw | `decisions.json` | `networks.<run>["C2-sw"].stat`, `.ci`, `.raw`, `.r2_base`, `.r2_full`, `.pool`, `.spearman_margin` |
| C3-w and its sensitivities | `decisions.json` | `C3-w`, `C3-w_sensitivity.<drop_kappa_set,drop_small>` |
| C3-R²c | `decisions.json` | `C3-R2c` |
| C3 inputs (completion, competence, overlap) | `prereg.json` | `c3w_inputs.<run>` |
| C1-own and its validity | `decisions.json` | `networks.<run>["C1-own"]` |
| C1-heads | `decisions.json` | `networks.<run>["C1-heads"]`; `descriptive_networks.<run>["C1-heads"]` |
| C1-braking, C1-limit, C1-pattern | `decisions.json` | `networks.<run>["C1-braking"]`, `["C1-limit"]`, `["C1-pattern"]` |
| W1 | `decisions.json` | `networks.<run>.W1` |
| Value accuracy | `decisions.json` | `networks.<run>.value_accuracy`, `.value_accuracy_vstar` |
| Trained-network K3 verdicts | `decisions.json`; `outputs/costate/post/critic/<run>.k3.json` | `k3_trained`; `decision`, `extractor`, `float32` |
| K3b's trained analogue | `outputs/costate/post/own/<run>.k3b.json` | `decision` |
| Strata tables | `decisions.json` | `networks.<run>.by_stratum.<C1,C2,C2-sw>.<stratum>` |
| Sensitivity variants, robustness | `decisions.json` | `networks.<run>.sensitivity.<variant>`, `.robustness_C1`, `.robustness_C2` |
| Other generalists and specialists | `decisions.json` | `descriptive_networks.<run>.C1` |
| K2b exclusions; power floor; residual shares; winsor bounds; folds | `prereg.json` | `frozen.k2b_excluded`, `frozen.power_floor`, `frozen.residual_share_interior`, `frozen.winsor_bounds`, `frozen.fold_partition0` |
| K0 | `gates/k0.json` | `summary` (`pass`, `scope`, `not_ok_descriptive`, `max`, `action_plants_min`); `rows[*].stationarity` |
| K1 | `gates/k1.json` | `decision` (`components`, `pi`, `plants`, `design_rule`, `design_rule_literal_kink`) |
| K2 | `gates/k2.json` | `horizon`, `retention`, `T_max_t`, `extend` |
| K2b | `gates/k2b.json` | `rule` (`excluded`, `p90`) |
| K3, K3b | `gates/k3.json`; `gates/k3b.json` | `checks`, `pullback_fd`, `planted_wrong`; `decision` |
| K4 | `gates/k4.json` | `PR-1a` … `PR-9`, `decoy_k` |
| K5 | `gates/k5.json` | `wiring`, `control` |
| K6 | `gates/k6.json` | top level |
| decoy-k | `gates/decoy_k.json` | `choice`, `reproduces` |
| c2-null | `gates/c2_null.json` | `summary`, `rows` |
| ξ tail analysis (Section 7.1, computed here) | `outputs/costate/inputs/world.npz`; `outputs/costate/post/critic/<run>.npz`; `decisions.json`; `prereg.json` | `tab/lam_xi`, `tab/interior`, `tab/phase`, `tab/direction`, `tab/geometry`, `tab/key_id`; `g[:, 0, 2]`; `networks.<run>.C1.calibration`; `frozen.winsor_bounds.lam_xi`, `frozen.fold_partition0` |
