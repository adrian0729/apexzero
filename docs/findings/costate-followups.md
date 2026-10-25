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
