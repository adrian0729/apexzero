# Locality of edge tracking: results (exploratory, 22 Oct 2026)

Plan: [locality-plan.md](locality-plan.md), committed in `291128d` before anything here was computed. Code: `scripts/locality_exploratory.py`. Results: `outputs/locality/results.json` (checks, per-network summaries, a `readings` block holding every number read below, provenance), `results.csv` (one row per entry and profile, with 95% CIs) and `per_key.npz`. Files under `outputs/` are local and not in git. Step 2: [receptive-fields-step2.md](receptive-fields-step2.md).

**Every number in this note is exploratory** and comes from `outputs/locality/results.json` unless another source is named. Nothing was pre-registered with a tag. No network, closed loop, gradient or solver was run, and R_TEST stays sealed.

**Revised after an independent verification.** A separate re-derivation (its own profiles, exclusion rules, pair aggregation and bootstrap, without the project package) reproduced every per-key value and pooled summary of the first run to about 1e-15, on all 35 entries and all 24 profiles. It found no computational error, but four problems in how the first draft read the results, all corrected here:
- E4 was not borne out, although a first summary said all four directions held.
- E3 holds only in part, and the plan's gloss "preview lead" is the reverse of what the kernel shows.
- For the sharpest profiles G_loc overstates the line's move, which changes one sentence about ramps (Sections 2 and 4.2).
- The trapezoid check confirms the plumbing, not the impulse responses (Section 3).

The script was then extended (the L/e form, group summaries, E3 and E4 span readings, a record of where s\* lies) and re-run. Every G_loc, L and E4 value is identical to the first run.

## 1. Summary

- **E1 is borne out.** The networks' gain falls as the inside-edge change becomes more local. It is monotone in σ_b and in X in all 29 distinct networks, and in L in 28 of 29.
  - **Bumps centred on the critical point s\*:** the median over networks is 0.095 at σ_b = 2 m (range 0.064–0.114), 0.166 at 4 m and 0.305 at 8 m. Step 2's corner-wide trapezoid gives 0.760 (0.651–0.855). At σ_b = 2 and 4 m all 29 networks are at or below 0.2, the plan's real-road band.
  - **Ramps ending at s\* fall less far.** At X = 5 m the line moves 0.238 (0.172–0.312) per metre the edge moves there, and 3 of 29 networks are at or below 0.2. G_loc gives 0.275 (0.216–0.339), but it overstates the line's move for this profile (Section 2).
- **E2 is borne out, by construction.** In families 1–3 the optimum's G_loc is 1 (per key 0.9999996–1.00000001). In family 4 it equals the bump's own height at s\*.
- **E3 is borne out in part.** The response peaks for changes 10 m before s\* (28 of 29 networks), and a change before s\* moves the line more than the mirror change after it (29 of 29). But the weakest response is to changes 40–80 m upstream, not at or after s\*, and the kernel shows the line trailing the edge by about 10 m, not a "preview lead".
- **E4 is not borne out.** At short scales, real-width and baseline networks do not differ (mean differences +0.005 to +0.019, two-sided p 0.13–0.81). They differ at long scales and for changes far upstream, where the real-width networks respond less. Step 2's corner-wide deficit of −0.079 comes from there.
- **For step 2's puzzle:** on the same keys, the same networks span about 0.1 to 0.8, depending only on the length scale of the change. Locality alone produces gains in the plan's real-road band (0.1–0.2), but it does not take G2 as low as its H3 inside-only median of 0.07 (G2's lowest here is 0.10). Whether the real-road changes are that local is not measured.

## 2. What was computed

- **Data.** The stored per-key impulse responses of c_min for all 35 network entries (29 distinct checkpoints) on the A2 keys (step 2's `fields-a2` arrays, each file checked against its row's sha256) and for the optimum (`opt-fields`). By linearity, dc_min/dΔ for an inside-width profile h is the dot product of h's samples with c_min's inside-width impulse row.
- **s\*** is the c_min argmin of each side's base run. The optimum's rows store it. The network rows do not, so it was recovered from the stored explicit term of c_min. That term's inside row is the arrays' cubic-spline basis at s\*, so its weights sum to 1 and their first moment is s\*.
- **Measure (plan).** G_loc(h) = H − dc_min/dΔ, where H is the height of h at s\*. H is 1 in families 1–3. In family 4 it is exp(−d²/2σ_b²), so family 4 is per unit of the bump's peak.
- **Families (plan):** (1) Gaussian bumps at s\*, σ_b ∈ {2, 4, 8, 16, 32, 64} m; (2) ramps from 0 at s\* − X to 1 at s\*, flat to s_c1 + 20 m, X ∈ {5, 10, 20, 40, 80} m; (3) plateaus of length L ∈ {10, 20, 40, 80} m centred at s\* with 20 m ramps, and step 2's trapezoid; (4) σ_b = 8 m bumps at s\* + d, d ∈ {−80, −60, −40, −20, −10, 0, +10, +20} m.
- **Keys and summaries.** Step 2's frozen `stages.edge_key_measures` gives each network's primary non-capped A2 keys and exclusions, "argmin off plateau" included, so every profile uses exactly the keys step 2's G used (330–423 of 427 per network).
  - Each summary is `measures.agent_summary`: mirror-pair values under step 2's rules, then the median over geometries (32–47 per network), with a 20,000-draw cluster bootstrap over geometries on step 2's stream. This is what the plan's "median over mirror pairs, with step 2's pair rules" means in step 2's code.
- **The line-move forms (the plan assumed they equal G_loc).** The plan takes H − dc_min/dΔ to be σ dn(s\*)/dΔ, the line's move.
  - That holds only where c_min sees the profile's full height at s\*. But c_min reads the edge through the arrays' spline of h sampled every 2 m. Where h has a sharp corner or peak at s\*, the spline reads e_h(s\*) < H. G_loc then overstates the line's move by H − e_h(s\*), and a line that did not move at all would score that much.
  - The median gap is 0.048 (max 0.071) at X = 5 m, 0.024 at X = 10 m, 0.012 at X = 20 m and 0.010 at σ_b = 2 m. It is 0.006 at X = 40 m and 0.003 at X = 80 m, and at most 0.0011 for the bumps of σ_b ≥ 4 m and for the plateaus. Step 2 excluded keys with e(s\*) < 0.99 for exactly this reason.
  - Two further forms are therefore reported: **L = e_h(s\*) − dc_min/dΔ = σ dn(s\*)/dΔ**, the line's move per unit of nominal height, and **L/e_h(s\*)**, the line's move per metre that the edge moves at s\* as c_min reads it (H3's line/edge ratio is defined the same way).
  - The forms differ clearly at σ_b = 2 m and for the ramps with X ≤ 20 m (by up to 0.05 in a network's median), slightly at X = 40–80 m (up to 0.008), and not for bumps of σ_b ≥ 4 m, the plateaus or the trapezoid (0.001 or less). Where they differ, L/e is the like-for-like reading; the optimum scores 1 on it in every family 1–3 profile.
- **Choices the plan left open, and departures from it:**
  - Beyond s_c1 + 20 m the ramp falls to 0 over 20 m, as the trapezoid does; the plan does not say. This choice was fixed in the script before it ran. The fallback for s\* beyond s_c1 + 20 m never applies, because every kept s\* lies on the trapezoid's plateau.
  - The optimum is summarised on all 427 primary non-capped keys (48 geometries), not on each network's kept keys. With per-key G_loc\* within 4e-7 of 1, this makes no difference.
  - The plan names no test for E4. H4's frozen permutation test (two-sided) and a leave-out comparison were added (Section 4.3).
  - The L/e form, the group summaries and the E3 and E4 span readings were added after verification.

## 3. Checks (all passed; the script stops and writes nothing on any failure)

| Check | Result |
|---|---|
| G_loc of the trapezoid against step 2's per-key G (required: 1e-9) | max \|difference\| 0 over 16,370 network rows with a finite G; 0 against G\* on all 481 optimum A2 keys |
| Pooled trapezoid summaries against `decisions.json` (35 entries and the optimum; median, lo, hi) | max \|difference\| 3.3e-16 |
| Impulse-array files against their rows' sha256 | 16,835 of 16,835 match |
| s\* recovery | basis sums within 5.6e-16 of 1; basis · trapezoid within 7.8e-16 of each row's e_star; the optimum's recovered s\* within 6.8e-13 m of its row; no kept network s\* off the trapezoid's plateau |
| Frozen code and record | `src/` and `scripts/fields.py` identical to HEAD (`291128d`); `decisions.json` sha256 as committed |

The first check confirms the plumbing only. Step 2's G is the same dot product on the same stored arrays, so the check confirms the outcome index, the inside side, the landmarks, the file-to-key mapping and the bootstrap stream. It does not confirm the impulse responses themselves; step 2 validated those (gate2-trained, its Section 4.1).

## 4. Results

### 4.1 G_loc by group: the median of the networks' pooled medians, with their range

Groups: **generalists**, step 1's nine; **baselines**, H4's baseline arms (seeds 1–4 of v1 and v3; G1s1, G1s2, G3s1 and G3s2 are also generalists); **real-width**, v1 and v3 seeds 0–3; **leave-out**, the eight leave-out runs; **all 29**, the distinct checkpoints. Per-network values with CIs are in `results.csv`. The optimum's CIs lie within 2e-11 of its values.

| Profile | All 29 | Generalists (9) | Baselines (8) | Real-width (8) | Leave-out (8) | Optimum |
|---|---|---|---|---|---|---|
| bump σ_b = 2 m | 0.095 (0.064 to 0.114) | 0.086 (0.064 to 0.110) | 0.085 (0.064 to 0.104) | 0.094 (0.075 to 0.107) | 0.100 (0.080 to 0.114) | 1.000 |
| σ_b = 4 m | 0.166 (0.112 to 0.198) | 0.152 (0.112 to 0.193) | 0.145 (0.112 to 0.186) | 0.166 (0.129 to 0.184) | 0.178 (0.141 to 0.198) | 1.000 |
| σ_b = 8 m | 0.305 (0.227 to 0.359) | 0.286 (0.227 to 0.358) | 0.276 (0.227 to 0.334) | 0.306 (0.253 to 0.336) | 0.326 (0.271 to 0.359) | 1.000 |
| σ_b = 16 m | 0.490 (0.402 to 0.562) | 0.486 (0.402 to 0.562) | 0.482 (0.402 to 0.541) | 0.474 (0.420 to 0.516) | 0.502 (0.431 to 0.560) | 1.000 |
| σ_b = 32 m | 0.647 (0.561 to 0.716) | 0.677 (0.598 to 0.716) | 0.669 (0.598 to 0.697) | 0.618 (0.569 to 0.647) | 0.636 (0.561 to 0.707) | 1.000 |
| σ_b = 64 m | 0.721 (0.604 to 0.808) | 0.763 (0.691 to 0.808) | 0.767 (0.691 to 0.792) | 0.707 (0.634 to 0.719) | 0.715 (0.604 to 0.777) | 1.000 |
| ramp X = 5 m | 0.275 (0.216 to 0.339) | 0.264 (0.229 to 0.315) | 0.257 (0.216 to 0.313) | 0.279 (0.246 to 0.296) | 0.294 (0.264 to 0.339) | 1.000 |
| X = 10 m | 0.314 (0.241 to 0.376) | 0.303 (0.241 to 0.345) | 0.298 (0.241 to 0.338) | 0.304 (0.274 to 0.342) | 0.338 (0.287 to 0.376) | 1.000 |
| X = 20 m | 0.400 (0.325 to 0.472) | 0.394 (0.325 to 0.443) | 0.390 (0.325 to 0.427) | 0.387 (0.373 to 0.431) | 0.429 (0.381 to 0.472) | 1.000 |
| X = 40 m | 0.536 (0.476 to 0.599) | 0.545 (0.476 to 0.595) | 0.537 (0.476 to 0.571) | 0.516 (0.489 to 0.575) | 0.546 (0.486 to 0.599) | 1.000 |
| X = 80 m | 0.646 (0.569 to 0.722) | 0.675 (0.607 to 0.722) | 0.674 (0.607 to 0.689) | 0.630 (0.569 to 0.650) | 0.637 (0.574 to 0.696) | 1.000 |
| plateau L = 10 m | 0.423 (0.337 to 0.500) | 0.408 (0.337 to 0.500) | 0.400 (0.337 to 0.471) | 0.413 (0.358 to 0.462) | 0.450 (0.389 to 0.488) | 1.000 |
| L = 20 m | 0.524 (0.429 to 0.607) | 0.518 (0.429 to 0.607) | 0.516 (0.429 to 0.581) | 0.505 (0.449 to 0.558) | 0.534 (0.467 to 0.594) | 1.000 |
| L = 40 m | 0.650 (0.553 to 0.723) | 0.669 (0.559 to 0.719) | 0.658 (0.559 to 0.708) | 0.616 (0.572 to 0.657) | 0.649 (0.553 to 0.723) | 1.000 |
| L = 80 m | 0.730 (0.645 to 0.825) | 0.779 (0.713 to 0.825) | 0.782 (0.713 to 0.795) | 0.718 (0.650 to 0.738) | 0.724 (0.645 to 0.788) | 1.000 |
| trapezoid (step 2's G) | 0.760 (0.651 to 0.855) | 0.803 (0.760 to 0.851) | 0.805 (0.760 to 0.855) | 0.737 (0.667 to 0.757) | 0.746 (0.651 to 0.811) | 1.000 |
| offset bump d = −80 m | 0.000 (−0.001 to 0.004) | 0.000 (0.000 to 0.004) | 0.000 (0.000 to 0.001) | 0.000 (−0.001 to 0.000) | 0.000 (0.000 to 0.001) | 0.000 |
| d = −60 m | 0.001 (−0.004 to 0.016) | 0.005 (0.000 to 0.016) | 0.006 (0.004 to 0.015) | 0.000 (−0.004 to 0.002) | 0.000 (−0.003 to 0.005) | 0.000 |
| d = −40 m | 0.046 (0.011 to 0.120) | 0.067 (0.044 to 0.111) | 0.070 (0.061 to 0.120) | 0.029 (0.017 to 0.057) | 0.039 (0.011 to 0.050) | 0.000 |
| d = −20 m | 0.267 (0.199 to 0.373) | 0.299 (0.256 to 0.331) | 0.300 (0.249 to 0.373) | 0.251 (0.221 to 0.278) | 0.252 (0.199 to 0.286) | 0.044 |
| d = −10 m | 0.368 (0.295 to 0.463) | 0.368 (0.295 to 0.463) | 0.370 (0.295 to 0.385) | 0.357 (0.311 to 0.383) | 0.363 (0.299 to 0.408) | 0.458 |
| d = 0 | 0.305 (0.227 to 0.359) | 0.286 (0.227 to 0.358) | 0.276 (0.227 to 0.334) | 0.306 (0.253 to 0.336) | 0.326 (0.271 to 0.359) | 1.000 |
| d = +10 m | 0.161 (0.122 to 0.209) | 0.155 (0.123 to 0.177) | 0.152 (0.122 to 0.184) | 0.159 (0.137 to 0.177) | 0.179 (0.151 to 0.209) | 0.458 |
| d = +20 m | 0.058 (0.041 to 0.076) | 0.051 (0.045 to 0.075) | 0.054 (0.042 to 0.072) | 0.056 (0.041 to 0.076) | 0.063 (0.053 to 0.075) | 0.044 |

### 4.2 The sharpest profiles in the three forms (29 networks: median, with the range)

G2 is the H3 network. Its H3 medians are 0.07 for the line's answer to the inside edge alone and 0.19 for its total move ([step 2](receptive-fields-step2.md), Section 7.7).

| Profile | H − e_h(s\*): median (max) | G_loc | L | L/e_h(s\*) | Networks ≤ 0.2 (G_loc / L / L/e) | G2: G_loc, L/e | Optimum: G_loc, L, L/e |
|---|---|---|---|---|---|---|---|
| bump σ_b = 2 m | 0.010 (0.020) | 0.095 (0.064 to 0.114) | 0.083 (0.055 to 0.102) | 0.085 (0.055 to 0.103) | 29 / 29 / 29 | 0.110, 0.100 | 1.000, 0.989, 1.000 |
| σ_b = 4 m | 0.000 (0.001) | 0.166 (0.112 to 0.198) | 0.166 (0.111 to 0.198) | 0.166 (0.111 to 0.198) | 29 / 29 / 29 | 0.193, 0.193 | 1.000, 1.000, 1.000 |
| ramp X = 5 m | 0.048 (0.071) | 0.275 (0.216 to 0.339) | 0.229 (0.164 to 0.296) | 0.238 (0.172 to 0.312) | 0 / 3 / 3 | 0.270, 0.236 | 1.000, 0.952, 1.000 |
| X = 10 m | 0.024 (0.034) | 0.314 (0.241 to 0.376) | 0.292 (0.219 to 0.357) | 0.296 (0.223 to 0.364) | 0 / 0 / 0 | 0.314, 0.296 | 1.000, 0.976, 1.000 |
| X = 20 m | 0.012 (0.017) | 0.400 (0.325 to 0.472) | 0.389 (0.313 to 0.461) | 0.393 (0.317 to 0.465) | 0 / 0 / 0 | 0.413, 0.406 | 1.000, 0.988, 1.000 |

At X = 5 m the three networks at or below 0.2 are `dr-pilot-200M-s4`, G3s2 and G1 (L/e 0.172, 0.189 and 0.198). At L = 10 m the three forms agree to 0.001. The X = 40 and 80 m ramps differ by up to 0.008, and that difference changes no reading.

### 4.3 E4: real-width against baseline networks (descriptive)

**Method** (the plan named no test). H4's arms: real-width v1 and v3 seeds 0–3 against baseline seeds 1–4, on each network's median G_loc, with H4's frozen within-variant permutation statistic (4,900 relabellings). The two-sided p is twice the smaller one-sided p (minimum 4.1e-4). "Diff" is the mean over the two variants of (mean real − mean baseline). The leave-out runs are compared with the v3 baselines (495 relabellings). Both gate2-trained failures are kept.

| Profile | Diff | v1 | v3 | p (two-sided) | Real-width range | Baseline range | Leave-out diff (p) |
|---|---|---|---|---|---|---|---|
| bump σ_b = 2 m | +0.010 | +0.012 | +0.008 | 0.13 | 0.075 to 0.107 | 0.064 to 0.104 | +0.020 (0.048) |
| σ_b = 4 m | +0.019 | +0.022 | +0.016 | 0.13 | 0.129 to 0.184 | 0.112 to 0.186 | +0.035 (0.065) |
| σ_b = 8 m | +0.026 | +0.031 | +0.021 | 0.16 | 0.253 to 0.336 | 0.227 to 0.334 | +0.053 (0.057) |
| σ_b = 16 m | +0.001 | +0.008 | −0.005 | 0.95 | 0.420 to 0.516 | 0.402 to 0.541 | +0.045 (0.18) |
| σ_b = 32 m | −0.047 | −0.044 | −0.050 | 0.0012 | 0.569 to 0.647 | 0.598 to 0.697 | 0.000 (0.97) |
| σ_b = 64 m | −0.066 | −0.068 | −0.064 | 8.2e-4 | 0.634 to 0.719 | 0.691 to 0.792 | −0.022 (0.51) |
| ramp X = 5 m | +0.012 | +0.028 | −0.004 | 0.37 | 0.246 to 0.296 | 0.216 to 0.313 | +0.032 (0.13) |
| X = 10 m | +0.014 | +0.030 | −0.003 | 0.38 | 0.274 to 0.342 | 0.241 to 0.338 | +0.046 (0.081) |
| X = 20 m | +0.015 | +0.027 | +0.003 | 0.36 | 0.373 to 0.431 | 0.325 to 0.427 | +0.052 (0.077) |
| X = 40 m | −0.002 | +0.004 | −0.008 | 0.87 | 0.489 to 0.575 | 0.476 to 0.571 | +0.036 (0.15) |
| X = 80 m | −0.038 | −0.037 | −0.039 | 0.0020 | 0.569 to 0.650 | 0.607 to 0.689 | +0.001 (0.94) |
| plateau L = 10 m | +0.016 | +0.025 | +0.008 | 0.44 | 0.358 to 0.462 | 0.337 to 0.471 | +0.056 (0.10) |
| L = 20 m | +0.005 | +0.013 | −0.003 | 0.81 | 0.449 to 0.558 | 0.429 to 0.581 | +0.049 (0.16) |
| L = 40 m | −0.030 | −0.030 | −0.030 | 0.060 | 0.572 to 0.657 | 0.559 to 0.708 | +0.027 (0.44) |
| L = 80 m | −0.062 | −0.058 | −0.067 | 4.1e-4 | 0.650 to 0.738 | 0.713 to 0.795 | −0.022 (0.45) |
| trapezoid | −0.079 | −0.081 | −0.077 | 4.1e-4 | 0.667 to 0.757 | 0.760 to 0.855 | −0.031 (0.24) |
| d = −60 m | −0.008 | −0.009 | −0.007 | 4.1e-4 | −0.004 to 0.002 | 0.004 to 0.015 | −0.007 (0.0081) |
| d = −40 m | −0.048 | −0.059 | −0.037 | 4.1e-4 | 0.017 to 0.057 | 0.061 to 0.120 | −0.032 (0.0040) |
| d = −20 m | −0.047 | −0.071 | −0.023 | 0.0020 | 0.221 to 0.278 | 0.249 to 0.373 | −0.020 (0.29) |
| d = −10 m | 0.000 | +0.003 | −0.003 | 1.00 | 0.311 to 0.383 | 0.295 to 0.385 | +0.022 (0.36) |
| d = +10 m | +0.005 | +0.012 | −0.002 | 0.62 | 0.137 to 0.177 | 0.122 to 0.184 | +0.028 (0.089) |
| d = +20 m | +0.002 | +0.008 | −0.005 | 0.72 | 0.041 to 0.076 | 0.042 to 0.072 | +0.007 (0.21) |

Two profiles are left out of the table. d = 0 is the same profile as the σ_b = 8 m row. At d = −80 m all values are within 0.002 of zero (diff −0.0005, p 0.014). The trapezoid row reproduces step 2's exploratory H4 statistic (−0.158 summed over the variants; its Section 6.3). In all there are 48 tests (24 profiles × 2), with no correction for multiplicity, and 12 have p below 0.05.

## 5. Readings against the pre-stated directions

- **E1 (G_loc falls as h becomes more local, well below 0.8, toward about 0.1–0.2): borne out.**
  - Monotone in σ_b and in X in 29 of 29 networks. In L it is monotone in 28 of 29: in `dr-realw-v3-s2` the L = 80 m value, 0.673, exceeds the trapezoid's 0.667.
  - Bumps of σ_b = 2–4 m bring every network into the plan's real-road band, in every form (L/e 0.055–0.198). G2 gives 0.100 at 2 m and 0.193 at 4 m (L/e).
  - **Corrected sentence.** The first draft said that ramps "do not get that low (0.22–0.34)". That holds only for G_loc, which overstates the line's move at X = 5 m by about 0.05. In L/e the ramps reach 0.172–0.312, and three networks are at or below 0.2. G2 is at 0.236. So the ramps get close to the real-road band, but most networks stay above it.
- **E2 (the optimum stays at 1): borne out, and by construction.** Its inside contact is active, so to first order its line moves with the edge at s\*, whatever the profile.
  - In families 1–3 its G_loc and L/e are 1.
  - Its L drops to 0.952 at X = 5 m only because the spline reads the edge's move there as 0.952.
  - In family 4 its kernel is the bump itself: 1 at d = 0, 0.458 at ±10 m, 0.044 at ±20 m. This confirms the reference; it is not an independent finding.
- **E3 (most response to changes some way before s\*, "a preview lead"; least at or after s\*): borne out in part.**
  - **Holds: the peak lies before s\*.** It is at d = −10 m in 28 of 29 networks. In `dr-realw-loo-f2-s1`, d = 0 is higher by 0.0004 (0.2994 against 0.2989).
  - **Holds: before beats after.** A change before s\* moves the line more than the mirror change after it in 29 of 29 networks, both at ±10 m and at ±20 m.
  - **Fails as worded: "least at or after s\*".** Every network's minimum is at d = −80 or −60 m, and both lie below d = +20 m in 29 of 29; d = −40 m lies below d = +20 m in 19 of 29. The medians over networks are 0.000, 0.001 and 0.046 at d = −80, −60 and −40 m, against 0.305, 0.161 and 0.058 at d = 0, +10 and +20 m. The before-after asymmetry holds only within about 20 m of s\*. Almost all of the response comes from changes between 40 m before and 20 m after s\*, and it peaks 10 m before.
  - **The gloss is reversed.** A kernel centred 10 m before s\* means the line at s\* carries the edge from about 10 m earlier. So the line's move trails the edge change by about 10 m, while the optimum's line moves with the edge in place (its kernel is centred at d = 0). A preview lead, a line that moves before the edge does, would put the weight at d > 0. Changes after s\* get the least near-s\* weight (0.161 at +10 m against 0.368 at −10 m).
  - **Not separated here:** whether the trail comes from the car's lateral dynamics or from the policy.
- **E4 (real-width and baseline networks differ at short scales; direction not predicted): not borne out.**
  - **Short scales.** At the script's short scales (σ_b = 2–4 m, X = 5–10 m, L = 10–20 m) the mean differences are +0.005 to +0.019 (per variant −0.004 to +0.030), with two-sided p 0.13–0.81.
  - **Wider short scales** (σ_b ≤ 16 m, X ≤ 40 m, L ≤ 20 m): −0.002 to +0.026, with p 0.13–0.95. With 4 seeds per arm, small differences cannot be excluded.
  - **Leave-out runs.** They sit slightly higher at short scales (+0.020 to +0.056, p 0.048–0.16). The one p below 0.05 (σ_b = 2 m) is one of 48 uncorrected tests. The other 11 tests with p below 0.05 are all at long scales or 20–80 m upstream.
  - **Where the arms do differ, the real-width networks are lower.** This shows at σ_b = 32–64 m, X = 80 m, L = 80 m and the trapezoid (−0.038 to −0.079, p ≤ 0.002), and for bumps 20–60 m before s\* (p ≤ 0.002). So their lower corner-wide G comes from the slow, far-upstream part of the response.
  - Real-width training did not change local following. This fits step 2's exploratory reading that it bought margin, not tracking.

## 6. What it means for step 2's puzzle (corner-wide G about 0.8, real-road about 0.2)

- **The corner-wide G comes mostly from slow edge changes.** A σ_b = 64 m bump already gives 0.604–0.808 and an 80 m plateau 0.645–0.825, close to the trapezoid. A change confined to a few metres at s\* gives 0.06–0.20.
  - So on the same synthetic keys the same networks span nearly the whole range from the real-road values to the corner-wide G. Only the length scale of the change differs.
  - Real corner shapes, the outside width and nonlinearity are not needed to get gains in the plan's real-road band (0.1–0.2).
- **It is not shown that locality is the explanation.**
  - The real crash corners' own edge profiles were not used. Where those profiles sit relative to a kernel peaking about 10 m before the closest point is unknown.
  - **G2 does not reach its inside-only median.** The like-for-like number is G2's H3 inside-only median of 0.07, because only the inside width changes here. No profile here takes G2 that low: its lowest is 0.10, for the 2 m bump. The plan's model of a real-road change, the ramp ending at s\*, gives G2 0.24 at X = 5 m. That is near G2's H3 total (0.19), but well above 0.07.
  - G2's H3 inside-only values range from −0.04 to 0.38 over the 12 unit-phases (step 2, Section 7.7). Its values for the most apex-local profiles (σ_b = 2–4 m bumps and the X = 5 m ramp) are 0.10–0.24. They lie inside that range but above its median.
  - For G2, then, the locality of a change closes most of the gap (0.83 down to 0.10–0.24), but not all of it. The rest may lie in the real profiles' shape, the corners or states, or nonlinearity; none of these is tested here.
- **A possible mechanism (hypothesis).** The kernel says the line answers the edge about 10 m late and over a window of a few tens of metres. A shift sustained over the whole corner lets that trailing response catch up; a short change at the apex is over before the line reacts. Step 2's first-order crash rule (Section 8 there) says clearance falls by about 1 − r per metre of local intrusion. Its r ≈ 0.2 is matched here by r ≈ 0.1–0.2 for apex-local changes.

## 7. Limits

- **First order.** Everything is linear and evaluated at the base argmin. Step 2's dose-response supports linearity only for corner-wide shifts up to 2 m. For sharp local changes of finite size, the closest point may move to the perturbed spot, and nothing here covers that.
- **Synthetic corners at A2.** These are the A2 keys: synthetic corners, mostly 12 m wide (W12), at the typical approach and from the shared start states. They are not the real crash corners, real profiles or the agents' own states.
- **Same keys as step 2's G.** The plan keeps the "argmin off plateau" exclusion, which drops 0–92 keys per network (step 2, Section 4.3). For G3 these are tight corners, where the other generalists track less.
- **The form matters only for the sharpest profiles.** Read L/e at σ_b = 2 m and X ≤ 20 m. Elsewhere the forms differ by at most 0.008.
- **E4 is descriptive.** It is confounded by code and hardware as in step 2. There are 4 seeds per arm, the two gate2-trained failures are kept, there are 48 uncorrected tests, and the test itself was not named in the plan.
- **Exploratory throughout.** The directions were fixed before computing. The readings, the forms and the E4 test were not pre-registered.

## 8. Next steps

1. **Real profiles (needs a plan entry).** Compute G_loc and L/e for the measured real-minus-constant inside-width profiles of G2's three crash units, and the trapezoid G on those units. Both need new VJPs at 12 unit-phases. That directly tests whether locality reproduces H3's 0.07.
2. **Finite-amplitude checks.** Run closed-loop checks of a σ_b = 4 m bump and an X = 5 m ramp on a pre-specified key subset (R-dev roads only). They test the first-order values where linearity is least supported.
3. **A training target.** If local following is the aim, the outcome to train on and test is the short-scale response (L/e for σ_b ≈ 4–8 m bumps at s\*, or the family 4 kernel's peak and lag), not corner-wide G. Any new pre-registration of width training (step 2's Section 10, item 5) should name it.
