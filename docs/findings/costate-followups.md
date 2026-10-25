# Idea 2's follow-ups: findings (exploratory)

**Status.** Every number here is **exploratory**. Each comes from an item of
[the follow-ups plan](costate-followups-plan.md) (first committed in `eaf1e80`, before any
computation; amendments appended there). None is confirmatory on these 34 networks, and none
changes a decision of `costate-prereg-v1` (`outputs/costate/analyse/decisions.json`). Where a
frozen band is named ("the refute band"), it describes the number; it decides nothing (plan R1).
Each item's expected directions are quoted beside its result.

Sources:
- each item's `outputs/costate/exploratory/<item>/summary.json` (local, stamped);
- the committed copies under [`costate-followups/`](costate-followups/).

## CS-1. The ξ tail (exploratory)

**The run.**
- `scripts/costate_exploratory.py cs-1` at `70979ee`, under the plan with amendment 1 (`320ef22`,
  CS-1's operational details, committed before any CS-1 number).
- The 9 decision networks, 131 s on the Mac, niced, no pool.
- Outputs: `outputs/costate/exploratory/cs-1/summary.json` (committed copy:
  [`costate-followups/cs-1-summary.json`](costate-followups/cs-1-summary.json)) and `rows.npz`
  (per row, local).
- The stamp records an empty `changed_vs_head`, no partial flag, and the sha256 of all 39 inputs:
  the world, the gates' K4 record, the 9 networks' critic, K3 and activations files, and their
  checkpoints through the record's pins.
- A partial development run on G1 alone came first, written outside `outputs/costate`. It
  showed that `probes.wspearman` returns a round-off value, not "undefined", for a constant g.
  So the rank calibration's zero and direction-only plants had taken the rank route instead of
  amendment 1's fallback. The code was fixed to the amendment's text before the canonical run.
  No definition changed.

**Rows.** λ_ξ's 1,581 decision rows (approach phase, interior). λ_s's 1,643 rows enter only
where the calibrations are compared. No braking or corner row is scored (decision #8, plan
R5).

**Wiring checks (all passed).**
- **The tail.** Its counts equal findings Section 7.1's in all 9 (G1 286, G1s1 247, G1s2 259,
  G2 283, G2s1 315, G2s2 229, G3 240, G3s1 336, G3s2 294). The table's other columns agree to
  their printed rounding.
- **The frozen C1.** It reproduces `decisions.json` (`r2c`, `stat`, `ci`, the per-fold `a` and
  `b`) to at most 3.6·10⁻¹⁵. The rebuilt decision masks have the recorded sha256.
- **The fresh VJP.** Its `critic.pullback` equals the stored g bit for bit. The channel groups
  sum to the stored g_ξ within 5.3·10⁻¹² relative, and the accel term is at most 1.1·10⁻¹².

### Where the tail sits

| Network | Geometries holding half the tail's squared error (share) | Geometry 38: share, row enrichment | κ-clipped rows (tail / bulk), enrichment | s − s_apex, median (m), tail / bulk | max tyre util, median (mean), tail / bulk |
|---|---|---|---|---|---|
| G1 | 36 (0.37), 38 (0.11), 31 (0.07) | 0.11, 3.2 | 14 / 40, 1.58 | −75 / −98 | 0.94 (0.84) / <0.001 (0.25) |
| G1s1 | 38 (0.16), 36 (0.15), 0 (0.12), 8 (0.07), 28 (0.07) | 0.16, 2.0 | 14 / 40, 1.89 | −67 / −99 | 0.91 (0.84) / <0.001 (0.28) |
| G1s2 | 38 (0.27), 31 (0.15), 36 (0.13) | 0.27, 3.1 | 11 / 43, 1.31 | −74 / −99 | 0.94 (0.86) / <0.001 (0.26) |
| G2 | 36 (0.43), 38 (0.09) | 0.09, 4.6 | 8 / 16, 2.29 | −64 / −101 | 0.94 (0.85) / <0.001 (0.24) |
| G2s1 | 38 (0.22), 36 (0.18), 31 (0.13) | 0.22, 3.1 | 7 / 16, 1.76 | −72 / −100 | 0.95 (0.84) / <0.001 (0.23) |
| G2s2 | 38 (0.46), 28 (0.15) | 0.46, 5.0 | 0 / 23, 0 | −80 / −96 | 0.90 (0.81) / <0.001 (0.29) |
| G3 | 36 (0.32), 38 (0.32) | 0.32, 4.7 | none | −76 / −97 | 0.95 (0.85) / <0.001 (0.27) |
| G3s1 | 38 (0.26), 36 (0.18), 34 (0.12) | 0.26, 2.9 | none | −77 / −99 | 0.93 (0.81) / <0.001 (0.24) |
| G3s2 | 36 (0.32), 38 (0.27) | 0.27, 3.7 | none | −81 / −95 | 0.87 (0.77) / <0.001 (0.28) |

The geometries are (R, θ): 0 (6.0 m, 90°), 8 (9.4 m, 180°), 28 (22.9 m, 180°), 31 (35.8 m, 90°),
34 (40.0 m, 180°), 36 (44.7 m, 90°) and 38 (44.7 m, 180°). Medians and means are key-weighted.
"Max tyre util" is max(util_f, util_r) as the observation caps it. It is a descriptive companion
added by amendment 1, with no expected direction.

- **Two geometries.** Two to five geometries hold half of the tail's squared error. Geometry 38
  is among them in all 9 networks. Geometry 36, the other 44.7 m corner (θ 90° rather than 180°),
  is among them in 8.
- **The late approach.** 84–99% of the tail's squared error lies 125–50 m before the apex, the
  bins of s − s_apex from −125 to −50 m. The tail's median s − s_apex is −64 to −81 m, against −95
  to −101 m on the bulk.
- **High tyre utilisation.** On the tail rows the larger tyre utilisation has a key-weighted
  median of 0.87–0.95, against below 0.001 on the bulk. No row on either side is at the
  observation's tyre clip (util ≥ 2).
- **No preview-spacing kink.** On these rows vx lies between 30.1 and 81.0 m/s, so no row lies
  within 1 m/s of 25 or 83.3 m/s, and both kink flags are never set. The tail's median vx is about
  1 m/s above the bulk's in all 9 (56.3–56.8 against 55.6 m/s). The weighted means go either way:
  the tail's is lower in 4.
- **Clipping is κ clipping only.** On these rows every clipped channel is a preview κ channel; no
  channel that ξ moves is ever clipped. κ clipping touches 23–54 rows in 6 networks and none in
  G3, G3s1 and G3s2. Its row enrichment exceeds 1.5 in 4 networks, but those rows carry at most
  3.9% of the tail's squared error.
- **No tyre kink.** As amendment 1 anticipated, the tyre-kink flag is never set: channels
  212–215 do not read ξ.

### The critic against the map

g_ξ is split by a fresh VJP of the main V into channel groups. Shares are of |g_ξ| and signed:
a group that pulls against the total has a negative share, and the shares sum to 1. The
lateral channel 204 contributes nothing, since J_z[204, ξ] = 0, so the third column is the ξ
channel 205 alone.

| Network | Tail: near / far half of the edge poses / ξ channel | Bulk: near / far / ξ channel | ‖∂V_c/∂z‖ on ξ's channels, tail / bulk | ‖J_z e_ξ‖, tail / bulk | median \|g_ξ\|, tail / bulk | near group's sign opposite λ_ξ, tail rows |
|---|---|---|---|---|---|---|
| G1 | 0.82 / 0.00 / 0.18 | 0.44 / 0.34 / 0.22 | 3.19 | 0.97 | 90.9 / 2.7 | 0.90 |
| G1s1 | 0.86 / 0.01 / 0.13 | 0.74 / −0.09 / 0.35 | 3.83 | 0.96 | 108.2 / 5.1 | 0.94 |
| G1s2 | 0.88 / 0.04 / 0.09 | 0.74 / −0.14 / 0.41 | 2.42 | 0.97 | 62.0 / 4.2 | 0.91 |
| G2 | 0.84 / 0.00 / 0.16 | 0.41 / 0.42 / 0.17 | 2.84 | 0.94 | 72.2 / 3.3 | 0.92 |
| G2s1 | 0.82 / 0.01 / 0.18 | 0.86 / −0.08 / 0.22 | 3.07 | 0.96 | 108.8 / 3.2 | 0.91 |
| G2s2 | 0.82 / 0.04 / 0.14 | 0.78 / 0.04 / 0.18 | 2.54 | 0.98 | 64.2 / 3.6 | 0.91 |
| G3 | 0.82 / 0.04 / 0.15 | 0.82 / −0.04 / 0.22 | 2.45 | 0.97 | 64.6 / 4.0 | 0.91 |
| G3s1 | 0.82 / 0.02 / 0.16 | 1.37 / −0.34 / −0.02 | 3.40 | 0.98 | 81.7 / 3.3 | 0.92 |
| G3s2 | 0.77 / 0.04 / 0.20 | 1.30 / −0.31 / 0.01 | 3.08 | 1.00 | 100.5 / 3.7 | 0.87 |

The size columns are ratios of key-weighted medians, tail over bulk. The near half is preview
points 1–20; the far half is points 21–40.

- **The critic, not the map.** On tail rows the critic's gradient on ξ's channels is 2.4–3.8
  times its size on bulk rows, while the map's ξ column is 0.94–1.00 times. |g_ξ| itself is 15–34
  times larger. Most of the rest is alignment: the median |cosine| between the critic's gradient
  and the map's ξ column is 0.11–0.13 on the tail, against 0.010–0.019 on the bulk (from
  `rows.npz`).
- **The near preview, not the far.** On tail rows the near half of the edge-pose channels
  carries 77–88% of |g_ξ|, the far half 0–4%, and the ξ channel 9–20%.
- **The wrong sign comes from the near preview.** On 87–94% of tail rows, the near half's
  contribution has the sign opposite to λ_ξ.

### Other calibrations

R²_c is R²_wd for ξ. The rows, weights, folds and bootstrap are the frozen ones, and CIs are
95%. The bulk and tail splits are in the summary.

| Network | ξ: frozen | ξ: Huber [CI] | ξ: rank [CI] | s: frozen | s: Huber | s: rank |
|---|---|---|---|---|---|---|
| G1 | −0.002 | −0.030 [−0.138, 0.006] | −0.664 [−1.125, −0.431] | 0.021 | 0.019 | −0.768 |
| G1s1 | −1.351 | −0.000 [−0.099, 0.037] | −0.274 [−0.771, 0.050] | −0.102 | −0.034 | −0.932 |
| G1s2 | −2.706 | −0.020 [−0.129, 0.018] | −0.268 [−0.621, −0.029] | −0.101 | −0.026 | −0.875 |
| G2 | 0.006 | −0.032 [−0.139, 0.007] | −0.588 [−0.885, −0.458] | −0.103 | −0.052 | −0.955 |
| G2s1 | −4.984 | −0.011 [−0.116, 0.024] | 0.043 [−0.276, 0.252] | 0.029 | 0.020 | −0.701 |
| G2s2 | −0.320 | −0.024 [−0.131, 0.012] | −0.175 [−0.399, −0.043] | −0.015 | −0.035 | −0.854 |
| G3 | −2.679 | −0.026 [−0.130, 0.012] | −0.154 [−0.480, 0.070] | −0.038 | 0.027 | −0.495 |
| G3s1 | −5.003 | 0.002 [−0.094, 0.036] | −0.399 [−0.793, −0.172] | −0.147 | 0.007 | −0.709 |
| G3s2 | −2.709 | −0.032 [−0.138, 0.006] | −0.213 [−0.601, 0.014] | −0.063 | −0.019 | −0.807 |

- **The plants.**
  - The frozen calibration passes its plants in all 9: each plant scores −0.015 on s and −0.011
    on ξ.
  - Huber fails in all 9. Every plant scores −0.055 on ξ, below −0.02, because its intercept is
    the Huber location of the skewed λ_ξ rather than its mean. That is why design Section 13
    moved C1's calibration off Huber.
  - Rank fails in all 9. Its noise plant scores −0.49 on s and −0.40 on ξ: a chance positive
    Spearman in a training fold maps noise onto λ's whole distribution. Its other plants score
    −0.015 and −0.011.
- **Huber's ξ is near 0 because it barely reads g.** Its ξ R²_c is −0.032 to 0.002 in all 9. That
  is above the frozen one in the 7 networks with large slopes, because its per-fold ξ slopes are
  only 0.0007–0.0046, against fold-mean slopes of 0.0035–0.039 for the frozen calibration. So its
  predictions stay near its intercepts. Its ξ scores lie 0.02–0.06 above its own zero plant.
- **Rank is far below.** It scores −0.66 to 0.04 on ξ and −0.96 to −0.50 on s. With weak rank
  agreement, the quantile map hands the predictions λ's full spread.
- **The tail is worse under every calibration.** On the tail rows ξ scores −0.12 to −0.65 under
  Huber, −1.0 to −3.1 under rank, and −0.74 to −88 under the frozen calibration.

### Expected directions against the results

- **E1.1** ("at least half of its squared error lies in at most 6 of the 48 geometries, with
  geometry 38 among them in at least 6 of the 9 networks"): **holds.** Two to five geometries hold
  half, and geometry 38 is among them in 9 of 9.
- **E1.2** ("its rows have higher vx than the bulk, and the kink band around 83.3 m/s is enriched
  (ratio > 1.5) in at least 6 of 9. Neither observation clipping nor the tyre kinks are enriched
  (ratio < 1.5)"): **does not hold.**
  - The tail's median vx is higher in 9 of 9, but only by about 1 m/s, and the means go either
    way.
  - The 83.3 m/s band holds no row: these rows' speeds stop at 81.0 m/s.
  - κ clipping, the only clipping on these rows, is enriched above 1.5 in 4 of 9, so "not
    enriched" holds in 5 of 9, one short of 6. It carries at most 3.9% of the tail's error.
  - The tyre-kink flag is never set.
- **E1.3** ("the far half of the edge-pose channels carries more than half of |g_ξ| on tail rows,
  and less than half on bulk rows. ‖∂V_c/∂z‖ on the ξ channels is larger on tail rows than on bulk
  rows (ratio > 2), while ‖J_z e_ξ‖ differs less (ratio < 2)"): **does not hold, in its preview
  part.**
  - The critic-against-map part holds in 9 of 9: critic ratio 2.4–3.8, map ratio 0.94–1.00.
  - The far half carries more than half on tail rows in 0 of 9: it is the near half, at 77–88%.
- **E1.4** ("Under the Huber and rank calibrations, ξ's R²_c rises toward the bulk's (at least
  −0.05 in all 9) but stays below 0.2 in all 9. s moves by less than 0.03"): **does not hold** for
  either calibration.
  - Huber: ξ is at least −0.05 and below 0.2 in 9 of 9, but its s moves by 0.03 or more in 6 of 9.
  - Rank: ξ is at least −0.05 in 1 of 9, and s moves by 0.03 or more in 9 of 9.

**The rule for later items** (plan CS-1): it **does not fire.** Neither alternative passes its
plants in any network. Huber would otherwise have raised ξ's R²_c by more than 0.1 in 7 of 9. So
the later items report R²_c under the frozen calibration only.

### What CS-1 says (exploratory)

- **The tail is in the critics' gradients, not in the observation map.**
  - The map's ξ column has the same size on tail and bulk rows.
  - None of ξ's channels is clipped on these rows, no row is near a preview-spacing kink, and
    none is at the tyre clip.
- **Where it is.** It sits on the late approach to the two 44.7 m corners (about 125–50 m before
  the apex), at states with high tyre utilisation.
- **What carries it.** The critics' gradients with respect to the near half of the edge preview
  are larger there, much better aligned with the map's ξ column, and of the sign opposite to λ_ξ.
- **The refute stands.** No calibration that passes its plants brings any network's ξ R²_c near
  0.2: only the frozen one passes, and its bulk R²_c is at most 0.092 (findings Section 7.1).
  CS-1 changes no decision.

## CS-2 to CS-5: the runs (exploratory)

- `scripts/costate_exploratory.py cs-2`, `cs-3`, `cs-4` and `cs-5` at `b62ead3`, one after another,
  niced, under the plan with amendment 2 (`f95456f`: CS-2 to CS-5's operational details,
  committed before any of their numbers). CS-4 and CS-5 ran in pools of 4 workers × 2 XLA
  threads, BLAS pinned to one thread. The Mac was shared with another workflow throughout
  (load average 30–60, swap nearly full), so the wall times below are long.
- Every stamp records an empty `changed_vs_head`, no partial flag, the plan's sha256 with 2
  amendments, and the sha256 of every input: CS-2 104 files (the world, all 34 networks' critic
  and K3 files and their checkpoints through the record's pins), CS-3 146 (CS-2's, plus the 29
  step 2 rows files, the optimum's rows and the 12 step 1 run tables), CS-4 and CS-5 38 each.
- **Wiring checks (all passed).** In every item and network the frozen C1 was recomputed first
  and reproduced `decisions.json` (r2c, stat, ci, the per-fold a and b of s and ξ) to at most
  3.6·10⁻¹⁵; the 9 decision networks' rebuilt masks have the recorded sha256; CS-2's heads give
  C1-heads' D of g_s and g_ξ to within 10⁻¹⁶; CS-4's probes agree with `analysis.delta_c1` exactly;
  CS-5's pool actor of seed 0, rebuilt from its PPOConfig, gives its stored h3 exactly.
- **Development runs first.** Partial runs, written outside `outputs/costate`, came before the
  canonical ones: CS-2, CS-4 and CS-5 on G1 alone, and CS-3 on the first 3 networks. They
  found no fault. After them only two things changed, neither a definition: `--limit` for CS-2
  and CS-3, and a filter for a harmless numpy warning in CS-3.
- **No calibration but the frozen one.** CS-1's rule for later items did not fire, so every
  R²_c below is the frozen calibration's.

## CS-2. The descriptives pack (exploratory)

**Run.** 402 s, all 34 networks. Output: `outputs/costate/exploratory/cs-2/summary.json`
(committed copy: [`costate-followups/cs-2-summary.json`](costate-followups/cs-2-summary.json)).
**Rows.** C1's rows (λ_s 1,643 and λ_ξ 1,581 rows in the 9 decision networks), D's 477 rows, and
for the accel term (no target) the interior.

| Network | s: R²_c / unwinsorised | ξ: R²_wd / raw R² / floor | ρ_s / ρ_ξ | heads' C1 statistic (largest CI upper end) | heads − main V | C1 on D [CI] | SD over folds, s / ξ | accel share of g_vx |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.021 / 0.021 | −0.002 / 0.341 / 0.335 | 0.990 / 0.990 | −0.001 to 0.014 (0.034) | −0.01 to 0.00 | −0.017 [−0.112, −0.007] | 0.20 / 0.04 | 0.051 |
| G1s1 | −0.102 / −0.101 | −1.351 / −0.547 / 0.335 | 0.960 / 0.985 | −1.136 to −0.938 (−0.519) | −0.41 to −0.21 | 0.057 [−0.081, 0.123] | 0.08 / 0.98 | 0.042 |
| G1s2 | −0.101 / −0.100 | −2.706 / −1.438 / 0.335 | 0.988 / 0.989 | −1.858 to −1.514 (−0.655) | −0.45 to −0.11 | 0.103 [−0.005, 0.161] | 0.26 / 1.53 | 0.039 |
| G2 | −0.103 / −0.102 | 0.006 / 0.346 / 0.335 | 0.996 / 0.996 | −0.062 to −0.053 (0.015) | −0.01 to −0.00 | −0.017 [−0.112, −0.007] | 0.19 / 0.05 | 0.041 |
| G2s1 | 0.029 / 0.029 | −4.984 / −2.937 / 0.335 | 0.975 / 0.973 | −2.934 to −2.612 (−1.186) | −0.46 to −0.13 | 0.067 [−0.095, 0.140] | 0.19 / 2.37 | 0.039 |
| G2s2 | −0.015 / −0.015 | −0.320 / 0.131 / 0.335 | 0.987 / 0.992 | −0.181 to −0.086 (−0.016) | −0.01 to 0.08 | −0.017 [−0.112, −0.007] | 0.16 / 0.32 | 0.044 |
| G3 | −0.038 / −0.037 | −2.679 / −1.421 / 0.335 | 0.994 / 0.994 | −1.718 to −1.507 (−0.442) | −0.36 to −0.15 | 0.063 [−0.050, 0.115] | 0.38 / 2.10 | 0.047 |
| G3s1 | −0.147 / −0.145 | −5.003 / −2.949 / 0.335 | 0.966 / 0.969 | −3.879 to −2.690 (−1.207) | −1.30 to −0.12 | −0.025 [−0.134, 0.011] | 0.58 / 2.60 | 0.032 |
| G3s2 | −0.063 / −0.062 | −2.709 / −1.440 / 0.335 | 0.991 / 0.994 | −1.703 to −1.606 (−0.591) | −0.32 to −0.22 | −0.030 [−0.150, 0.036] | 0.13 / 1.98 | 0.038 |

ξ's unwinsorised R² equals its R²_wd in every network: λ_ξ lies inside its winsorisation bounds
on all 1,581 rows. The floor is a target-only prediction on rows shared by the 9, so it is the
same 0.335 in each. "Accel share" is the key-weighted mean of the per-state share on C1's rows.

- **The heads agree with each other.** ρ_k is 0.960–0.997 for s and 0.969–0.996 for ξ over all
  31 generalists (ξ within direction: 0.974–0.997 in the 9), so √ρ_k ≥ 0.98. No head's
  |corr(g_k, λ_k)| comes near it: the main V's is −0.05 to 0.22 for s and 0.16 to 0.24 for ξ in
  the 9, and no head's exceeds 0.25 anywhere. On the whole interior, ρ of all 8 components is
  at least 0.959.
- **Every head refutes.** All 155 heads' C1 statistics have CI upper ends below 0.2 (the largest
  is 0.034). But the heads follow the main V only where the main V is near 0. Where the ξ tail
  drives the main V far below 0, the heads lie 0.1 to 1.3 below it. 53 of the 155 heads are
  within 0.1 of their main V.
- **Winsorisation hardly matters.** Unwinsorised R² differs from R²_c by −0.0004 to 0.0014 for s
  over all 34 networks, and not at all for ξ.
- **Raw R² of ξ.** It sits on the floor (0.341 and 0.346 against 0.335) only in G1 and G2, the
  two networks whose ξ R²_c is near 0. In the other 7 it is 0.20 to 3.3 below it: their
  calibrated ξ predictions spread wider than the direction means do.
- **The folds.** The SD over the 6 folds is 0.08–0.58 for s and 0.04–2.60 for ξ; for ξ it
  tracks the size of the tail. G3s1's per-fold ξ R²_wd runs from −9.8 to −1.6.
- **C1 on D.** All 34 networks are in the refute band (CI upper ends 0.161 at most among the 9,
  0.177 among the others). In G1, G2 and G2s2 every fold's slope on D is clamped at 0 for both s
  and ξ, so they score the intercept-only fit's −0.017. The interior-fitted calibration scored on
  D does worse: −0.030 to 0.007 for s and −0.142 to −0.012 for ξ.
- **The accel term is small.** On C1's rows it carries 3.2–5.1% of |g_vx| per state on average in
  the 9 (2.6–5.7% in the others), 6.9–10.0% on the rows with t ≥ 1 (D's reset rows carry none),
  and 5.5–7.9% on the whole interior.
- **The accel-channel sensitivity rises with tyre utilisation.** On C1's rows its weighted
  Spearman with util_max is 0.22 to 0.67 in the 9, and its median in the top util bin (≥ 1) is
  1.3 to 9.2 times that in the bottom one. On the whole interior, braking and corner rows
  included, the Spearman is weaker (−0.13 to 0.37).
- **The specialists** (no heads): R²_c of s −0.03 to −0.02 and of ξ 0.04 to 0.16; C1 on D 0.015
  to 0.111, all in the refute band.

**Expected directions against the results.**
- **E2.1** ("ρ_k ≥ 0.9 for s and ξ in every generalist ... So the ceiling √ρ_k does not bind"):
  **holds**, in 31 of 31.
- **E2.2** ("Every head's R²_c is in the refute band (CI upper end < 0.2), and the heads' R²_c
  agree with the main V's to within 0.1"): **does not hold.** The refute-band part holds for all
  155 heads; the agreement part for 53 of them.
- **E2.3** ("Unwinsorised R² is within 0.05 of R²_c for s, and lower than R²_c for ξ"): **does not
  hold**, in 0 of 9. The s part holds in all 34 networks. ξ's is equal, never lower, because no
  ξ decision row is winsorised.
- **E2.4** ("On ξ, raw R² lies near the direction-only floor (within 0.1)"): **does not hold**:
  2 of 9 (G1, G2), and 4 of the other 25.
- **E2.5** ("C1 on D is in the refute band"): **holds**, in 9 of 9 and in all 25 others.
- **E2.6** ("The accel term carries under 10% of |g_vx| on C1's rows"): **holds**, in 9 of 9 and in
  all 25 others.

**What CS-2 says (exploratory).** The five heads agree on their gradients (ρ about 0.99) and
none of them agrees with λ. The refute is not head-level indeterminacy: in design Section 6's
words, the gradient is determined but wrong, given the trunk. The heads do differ from the main
V in how far the ξ tail drags R²_c below 0, but not in the band. The accel term is too small to
matter for g_vx, and the critic's sensitivity to the accel channels grows with tyre
utilisation.

## CS-3. C3-int on s and ξ (exploratory)

**Run.** 174 s. Outputs: `outputs/costate/exploratory/cs-3/summary.json` (committed copy:
[`costate-followups/cs-3-summary.json`](costate-followups/cs-3-summary.json)) and `tables.npz`
(the residual and behaviour tables and the null draws, local).

**Rows and tables.**
- The residuals: every network's 48 geometries hold C1 rows of both s and ξ.
- Step 2's G − 1 and B: the 29 generalists; 327–419 of T's keys per network survive step 2's own
  exclusions, covering 45–48 geometries.
- Step 1's measures: the 9 generalist finals (and S1–S3); 471 keys each. The entry speed covers
  all 48 geometries. The brake onset covers 37 (445 keys): keys without a sustained brake have
  none.
- The specialists crash on most synthetic keys (289–438 of 471 deterministic runs), the
  generalists on 1–24.

**Results.** The correlation r is Pearson over the cells both double-centred tables hold. p is
two-sided against 10,000 blocked permutations; "null" is the permutation null's 2.5% and 97.5%
quantiles.

| Pair | Networks | Cells | r | p | Holm p | Null |
|---|---|---|---|---|---|---|
| s, G − 1 | 29 | 1,364 | 0.081 | 0.018 | 0.130 | [0.028, 0.079] |
| s, B | 29 | 1,364 | −0.027 | 0.235 | 0.720 | [−0.042, 0.005] |
| s, entry-speed ratio − 1 | 9 | 432 | 0.109 | 0.016 | 0.130 | [−0.025, 0.103] |
| s, brake-onset residual | 9 | 333 | −0.034 | 0.619 | 1 | [−0.127, 0.048] |
| ξ, G − 1 | 29 | 1,364 | −0.078 | 0.127 | 0.720 | [−0.091, −0.030] |
| ξ, B | 29 | 1,364 | 0.033 | 0.120 | 0.720 | [−0.010, 0.043] |
| ξ, entry-speed ratio − 1 | 9 | 432 | −0.078 | 0.165 | 0.720 | [−0.109, 0.016] |
| ξ, brake-onset residual | 9 | 333 | 0.003 | 0.960 | 1 | [−0.042, 0.119] |
| *With S1–S3 (descriptive):* | | | | | | |
| s, entry-speed ratio − 1 | 12 | 576 | 0.034 | 0.158 | – | [−0.036, 0.054] |
| s, brake-onset residual | 12 | 444 | 0.051 | 0.189 | – | [−0.051, 0.085] |
| ξ, entry-speed ratio − 1 | 12 | 576 | −0.018 | 0.495 | – | [−0.058, 0.032] |
| ξ, brake-onset residual | 12 | 444 | −0.047 | 0.183 | – | [−0.070, 0.069] |

The residual is (calibrated prediction − λ), σ-folded for ξ, so a positive s residual is a
calibrated s-gradient above λ_s. The brake-onset residual is s_b − s_b*, negative for earlier
braking.

- **No pair is sizeable.** The largest |r| is 0.109. After Holm's adjustment no pair has p ≤ 0.05;
  the smallest adjusted p is 0.130.
- **Two unadjusted p below 0.05**, both on s: G − 1 (r 0.081, p 0.018) and the entry speed (r
  0.109, p 0.016). Where a network's calibrated s-gradient lies unusually above λ_s, its G and
  its entry speed are a little higher than usual for it.
- **The nulls are not centred on 0.** The permutation keeps each geometry within its R tercile ×
  θ class block, so an association at the level of the blocks survives into the null. s with
  G − 1 has a null band of [0.028, 0.079], and its r of 0.081 lies just above it.
- **With the specialists** every pair is smaller (|r| ≤ 0.051, p ≥ 0.16).

**Expected direction against the result.**
- **E3.1** ("Every |r| < 0.15, and no pair has a Holm-adjusted p ≤ 0.05"): **holds.**

**What CS-3 says (exploratory).** Within a network, the geometries where its critic's
calibrated s- or ξ-gradient is unusually far from λ are not, to any useful degree, the ones
where its line, buffer, entry speed or braking point is unusually far from the optimum's. This
says nothing about n and vx, the pairs the theory signs, which have no decision rows (decision
#8).

## CS-4. ΔC1, R²_z and the MLP-on-z ceiling (exploratory)

**Run.** 653 s, 18 units (network × component) on 4 workers; each unit's MLP took 78–171 s.
Outputs: `outputs/costate/exploratory/cs-4/summary.json` (committed copy:
[`costate-followups/cs-4-summary.json`](costate-followups/cs-4-summary.json)) and `preds.npz`
(the out-of-fold predictions, local). **Rows.** C1's rows (λ_s 1,643, λ_ξ 1,581), C1's weights,
partition 0's folds.

| Network | s: R²_z / R²_MLP / R²(z ⊕ g) | s: ΔC1 [CI] | s: R²_c / R²_MLP | ξ: R²_z / R²_MLP / R²(z ⊕ g) | ξ: ΔC1 [CI] | ξ: R²_c / R²_MLP |
|---|---|---|---|---|---|---|
| G1 | 0.355 / 0.645 / 0.350 | −0.005 [−0.028, 0.012] | 0.033 | 0.663 / 0.830 / 0.660 | −0.003 [−0.023, 0.010] | −0.002 |
| G1s1 | 0.355 / 0.641 / 0.359 | 0.004 [−0.005, 0.021] | −0.159 | 0.668 / 0.819 / 0.668 | −0.000 [−0.000, 0.000] | −1.648 |
| G1s2 | 0.355 / 0.642 / 0.355 | −0.000 [−0.028, 0.033] | −0.157 | 0.669 / 0.821 / 0.668 | −0.001 [−0.002, 0.001] | −3.295 |
| G2 | 0.354 / 0.647 / 0.374 | 0.020 [−0.006, 0.065] | −0.160 | 0.671 / 0.838 / 0.671 | 0.001 [−0.003, 0.005] | 0.007 |
| G2s1 | 0.354 / 0.651 / 0.337 | −0.017 [−0.071, 0.014] | 0.045 | 0.671 / 0.832 / 0.668 | −0.003 [−0.013, 0.005] | −5.991 |
| G2s2 | 0.354 / 0.647 / 0.350 | −0.004 [−0.013, 0.003] | −0.023 | 0.671 / 0.817 / 0.672 | 0.001 [−0.012, 0.013] | −0.392 |
| G3 | 0.354 / 0.645 / 0.353 | −0.000 [−0.013, 0.012] | −0.059 | 0.673 / 0.832 / 0.671 | −0.001 [−0.011, 0.009] | −3.219 |
| G3s1 | 0.354 / 0.641 / 0.376 | 0.022 [−0.012, 0.061] | −0.229 | 0.673 / 0.829 / 0.666 | −0.007 [−0.072, 0.035] | −6.033 |
| G3s2 | 0.354 / 0.644 / 0.342 | −0.012 [−0.039, 0.004] | −0.097 | 0.673 / 0.845 / 0.668 | −0.005 [−0.019, 0.010] | −3.206 |

R² is R²_wd for ξ. g is the main V's 8 pulled-back components.

- **λ_s and λ_ξ are decodable from z.** A ridge on z gives 0.354–0.355 for s (CI about [−0.02,
  0.57]) and 0.663–0.673 for ξ ([0.47, 0.81]). The values hardly differ between networks
  because the probes z-score each feature on the training fold, so a network's obs_rms changes
  only which channels clip at ±10 (as K4's PR-7 intends).
- **The MLP ceiling is higher.** One hidden layer of 64 tanh units on z reaches 0.641–0.651 for s
  (CIs from about 0.35 to 0.86) and 0.817–0.845 for ξ (0.64 to 0.95): 0.29 above the ridge for s
  and 0.15–0.17 for ξ, CIs above 0 in every network.
- **The critic's gradient adds nothing to z.** ΔC1 is −0.017 to 0.022 for s and −0.007 to 0.001
  for ξ, every CI covering 0.
- **The normalised fidelity is near or below 0.** R²_c / R²_MLP is −0.23 to 0.045 for s and −6.0 to
  0.007 for ξ.

**Expected directions against the results.**
- **E4.1** ("R²_z is at least 0.3 for ξ and at least 0.2 for s"): **holds**, in 9 of 9 for both.
- **E4.2** ("The MLP ceiling is above R²_z by at least 0.05, and above 0.4 for ξ"): **holds**, in 9
  of 9.
- **E4.3** ("R²_c/R²_MLP ≤ 0.1 in every network"): **holds**, in 9 of 9 for both components.
- **E4.4** ("ΔC1 ≤ 0.02 in every network, with a CI that covers 0"): **does not hold**, by a
  hair: G3s1's s ΔC1 is 0.022 (CI [−0.012, 0.061]). Every other network and component holds;
  G2's s is 0.0198.

**What CS-4 says (exploratory).** The refute is not about the observation. What the networks
see determines λ_ξ well (R² about 0.67 linearly, 0.83 with one hidden layer) and λ_s moderately
(0.35, 0.65), on the same rows where the critics' calibrated gradients score near or below 0.
Adding the critic's own gradient to z does not improve a probe of λ. The failure is in the
critic's gradient: λ is there to be read from the input, and the critic's input gradient does
not carry it.

## CS-5. W1 on the Π targets (exploratory)

**Run.** 1,195 s, 9 units (one per network) on 4 workers, about 7 min each. Outputs:
`outputs/costate/exploratory/cs-5/summary.json` (committed copy:
[`costate-followups/cs-5-summary.json`](costate-followups/cs-5-summary.json)) and `preds.npz`
(local). **Rows.** T's interior rows of the approach and exit phases, 1,581 + 72 = 1,653, the
same in every network, under the key weights renormalised on them. The targets are winsorised
at their frozen bounds and scored by raw R².

**Decodability from z** (the same in every network to the third decimal, as in CS-4):

| Target | R²_z [CI] | Spearman, z | R²(z ⊕ actor h3) | R²(z ⊕ critic h3) | Pool's median raw gain over z, actor / critic |
|---|---|---|---|---|---|
| Π^in | 0.256 [0.036, 0.387] | 0.53 | 0.841–0.888 | 0.892–0.931 | 0.612–0.621 / 0.612–0.621 |
| Π^out | 0.419 [0.106, 0.557] | 0.62 | 0.724–0.798 | 0.756–0.872 | 0.361–0.378 / 0.361–0.378 |
| Π^slip | 0.742 [0.594, 0.817] | 0.89 | 0.876–0.899 | 0.889–0.915 | 0.150–0.160 / 0.150–0.160 |
| Π^μ | 0.529 [0.331, 0.652] | 0.79 | 0.760–0.822 | 0.784–0.864 | 0.270–0.283 / 0.270–0.283 |

The pool's columns are the median over each network's 20 controls of R²(z ⊕ control h3) −
R²_z: the 20 pool actors for the actor, 20 random-init critics for the critic. Ranges are over
the 9 networks.

**Gains beyond z, against the controls** (the trained h3's gain minus the controls' median;
two-level CI; rank among the 21):

| Network | actor: Π^in | Π^out | Π^slip | Π^μ | critic: Π^in | Π^out | Π^slip | Π^μ |
|---|---|---|---|---|---|---|---|---|
| G1 | 0.012 | −0.003 | −0.006 | −0.020 | 0.050 [0.030, 0.079] 1 | **0.079 [0.043, 0.129] 1** | 0.011 | **0.052 [0.026, 0.097] 1** |
| G1s1 | −0.001 | −0.039 | 0.002 | −0.016 | 0.039 [0.013, 0.070] 1 | 0.045 [0.006, 0.104] 1 | 0.009 | 0.032 [0.002, 0.064] 2 |
| G1s2 | −0.027 | −0.008 | −0.006 | −0.022 | 0.023 | 0.030 [0.012, 0.072] 2 | 0.015 [0.001, 0.036] 2 | 0.033 |
| G2 | −0.032 | −0.044 | −0.002 | −0.041 | 0.034 [0.012, 0.062] 1 | 0.021 | −0.007 | **0.052 [0.013, 0.092] 1** |
| G2s1 | −0.016 | −0.036 | −0.012 | −0.022 | 0.037 [0.013, 0.069] 1 | −0.018 | 0.015 | 0.010 |
| G2s2 | −0.011 | −0.030 | 0.001 | −0.021 | 0.021 [0.000, 0.047] 1 | −0.025 | −0.006 | −0.016 |
| G3 | 0.010 | 0.018 | −0.019 | 0.023 | **0.056 [0.037, 0.085] 1** | **0.092 [0.048, 0.152] 1** | 0.006 | 0.043 |
| G3s1 | −0.019 | −0.056 | −0.004 | −0.023 | 0.038 [0.016, 0.068] 1 | 0.044 [0.000, 0.097] 1 | 0.023 [0.004, 0.049] 1 | 0.041 |
| G3s2 | 0.007 | 0.012 | −0.013 | 0.007 | 0.026 [0.010, 0.051] 1 | 0.021 | 0.016 | 0.025 |

CIs and ranks are shown where the CI's lower end is at or above 0; bold marks a statistic above
0.05 with its CI above 0. The summary holds every CI and rank.

- **Π is decodable from z, and more so nonlinearly.** z alone gives 0.26 for Π^in, 0.42 for
  Π^out, 0.74 for Π^slip and 0.53 for Π^μ. Adding any 256-unit hidden layer computed from z,
  trained or random, raises this a great deal (to 0.84–0.93 for Π^in): most of that is the
  random features' nonlinearity, since the controls gain almost as much.
- **The actor's h3 adds nothing beyond its controls.** Its gain over the pool's median is −0.056
  to 0.023; no CI lies above 0, and it ranks 18th to 21st of 21 in 4–6 networks per target.
- **The critic's h3 adds a little, consistently, on Π^in.** Beyond the random-init critics it
  gains 0.021–0.056 on Π^in, CI above 0 in 8 of 9 and first of 21 in 8. On Π^out it gains
  −0.025 to 0.092 (CI above 0 in 5), on Π^μ −0.016 to 0.052 (3), and on Π^slip −0.007 to 0.023
  (2).
- **Five gains pass 0.05 with their CI above 0**, all in the critic and all in the three seed-0
  networks: G1 on Π^out (0.079) and Π^μ (0.052), G2 on Π^μ (0.052), G3 on Π^in (0.056) and
  Π^out (0.092).
- **W1 on these rows.** The calibrated R² of the critic's inside-width derivative against Π^in is
  −0.046 to 0.176, every CI's lower end below 0 (upper ends 0.07–0.29); its Spearman is 0.37 to
  0.53. With the winsorised target it is the same. The confirmatory W1 on the whole interior
  (braking and corner rows included) was not re-scored here (decision #8).
- **|ΔΠ|/SD_T** awaits BC-a0 (amendment 2). K2b was never measured for the Π targets, so how much
  unseen road moves them on these rows is not yet known.

**Expected directions against the results.**
- **E5.1** ("Π^in and Π^out are decodable from z (R² ≥ 0.3)"): **does not hold.** Π^out is, in 9
  of 9 (0.42); Π^in is not, in 0 of 9 (0.26).
- **E5.2** ("Neither hidden layer adds more than 0.05 beyond z for any Π target"): **does not
  hold.** The actor's h3 holds it for every target in 9 of 9. The critic's h3 exceeds 0.05 in 5
  network-target cases (above), each with its two-level CI above 0.
- **E5.3** ("Π^slip and Π^μ are less decodable than Π^in"): **does not hold**, in 0 of 9: they are
  more decodable (0.74 and 0.53 against 0.26).

**What CS-5 says (exploratory).**
- **The actor's h3 extends C2's result** from λ to the road and grip prices: no linear
  shadow-price information beyond what random-init actors' features of z give.
- **The critic's h3 is the exception, and it is small.** Five network-target cases pass the plan's
  "first sign" (a gain above 0.05 with its CI above 0), all in the critic and all in the seed-0
  network of each variant. On Π^in its gain is smaller (0.02–0.06) but consistent, and its CI is
  above 0 in 8 of 9.
- **What that does and does not show.** V is linear in the critic's h3, and the value of a state
  depends on how much room the road ahead leaves, so a layer that correlates with Π^in beyond z
  need not encode a price. It is still the first layer seen to hold information about a
  window-problem target that z and random features do not.
- **Two caveats.** The random-init critics are the only control. And Π's sensitivity to unseen
  road is unmeasured until BC-a0 reports |ΔΠ|/SD_T.
- **The calibrated agreement stays low.** On these rows the critic's own width derivative still
  agrees poorly with Π^in (calibrated R² at most 0.18, every CI reaching below 0), as W1 found on
  the interior.

## CS-2 to CS-5: where they leave the refute (exploratory)

- **The refute is about the critic's gradient.** It is not about head-level indeterminacy (CS-2:
  ρ ≈ 0.99), the observation (CS-4: λ_ξ is decodable from z at R² ≈ 0.67, 0.83 nonlinearly), the
  winsorisation (CS-2), the accel term (CS-2: under 6% of |g_vx|) or the decision states alone
  (CS-2: C1 on D refutes in all 34).
- **Nothing ties a network's specific gradient errors to its specific driving errors** (CS-3).
- **The critic's last hidden layer holds a little information about the wall and grip prices
  beyond z and random features** (CS-5), most consistently for Π^in, while its input gradient
  does not express it (W1 on these rows). That is the one result here that points away from "no
  shadow price anywhere". It is exploratory, and CS-7 (the critic's layers on λ) is the item that
  tests the same reading for the costate itself.
- None of this changes a decision of `costate-prereg-v1`.
