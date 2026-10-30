# X-1, locality of edge tracking on H4R's 20 same-code networks: plan (exploratory; written before anything here is computed, 30 Oct 2026)

**Status.** This is X-1 of the follow-through plan (`docs/plan/follow-through.md`, Section 9.2), a follow-up to [`locality.md`](locality.md). It is exploratory and not pre-registered with a tag. This plan is committed before anything below is computed, so the measures, the primary outcome and the reading are fixed in advance. Only synthetic A2 keys are used; R_TEST is not involved.

## Question

On step 2's networks, `locality.md` found two things (its E4):
- **Short, local inside-edge changes:** real-width and baseline networks did not differ in how far the line follows them.
- **Long scales and changes far upstream:** they did differ, with real-width training lower.

Those arms differed in code and hardware. H4R trained 20 networks on one machine with one commit: v1 and v3, real-width and baseline, five seeds each. On those networks, H4R confirmed that real-width training lowers corner-wide edge tracking (H4R-G).

**X-1 asks whether real-width training changes short-scale following on these same-code networks.** The answer bears on the follow-through's open question:
- **If short-scale following is unchanged,** width realism does not reach it, and local perturbations are the lever to test (GW-5).
- **If it is changed,** width realism alone may be enough.

## Data (already computed; no network is run)

- **H4R's per-key rows,** `outputs/h4r/fields/runs_<network>.jsonl`.
- **H4R's impulse arrays,** `outputs/h4r/fields/arrays/<network>_<timesteps>/<key>.npz`, each checked against its row's `arrays_sha256`.
- **Step 2's optimum rows** (`outputs/fields/opt-fields/rows.jsonl`), on H4R's 481 frozen A2 keys (`scripts/h4r_prereg.json`).
- **H4R's gate2-trained failing keys** (`outputs/h4r/gate2-trained/gate2_trained.json`, through `h4r.gate_verdict`).
- **H4R's committed `outputs/h4r/analyse/decisions.json`** (sha256 `f3bcbed6…`), for the checks.

The frozen code (`src/apexzero/fields`, `scripts/fields.py`, `scripts/h4r.py`) and `scripts/locality_exploratory.py` are imported read-only and not changed.

## Keys, measures and summaries

- **Keys: H4R's own.** `h4r.h4r_key_measures(rows, opt, "G", failing)` gives step 2's primary non-capped A2 keys and exclusions, plus H4R's gate2-trained exclusion. So every profile uses exactly the keys behind each network's H4R G.
- **Profiles and forms: `locality.md`'s, unchanged.** They come from `scripts/locality_exploratory.py`'s own functions (`profiles`, `key_values`, `network_pass`, `summaries`). There are 24 profiles in four families:
  - Gaussian bumps at s\*, σ_b = 2–64 m;
  - ramps ending at s\*, X = 5–80 m;
  - plateaus, L = 10–80 m, plus the corner-wide trapezoid;
  - σ_b = 8 m bumps at s\* + d, d = −80 to +20 m.

  Each is measured in three forms: G_loc, L and L/e.
- **Each network's statistic** is its pooled median over geometries (`measures.agent_summary`, 20,000 bootstrap draws on step 2's stream), as in `locality.md` and H4R.

## The comparison

- **The test.** For each profile and form, real-width is compared against baseline on the per-network medians. The test is H4R's frozen permutation test (`analysis.permutation_test`), stratified by variant, 5 against 5 in each: 63,504 relabellings.
- **What is reported:**
  - the difference: the mean over the two variants of (mean real-width − mean baseline);
  - the per-variant differences;
  - a two-sided p, twice the smaller one-sided p;
  - the paired-by-seed sign-flip test (`h4r.sign_flip_test`, 10 pairs, 1,024 flips), two-sided the same way;
  - the number of the 10 pairs in which real-width is lower.
- **The primary outcome is L/e for the σ_b = 4 m and 8 m bumps at s\*.** `locality.md` (Section 8, item 3) named this as the target for any width-training change. The two permutation p-values get Holm at 0.05, two-sided.

## The reading, fixed now

- **"Short-scale following is changed by real-width training"** if Holm rejects for either primary outcome; the direction is read from the sign.
- **Otherwise, "no change detected at short scales".** The two differences are reported beside the trapezoid's, which is the corner-wide effect H4R confirmed, so that their size can be judged.
- **Everything else is descriptive:** the other 22 profiles, the G_loc and L forms, and the family 4 kernel's peak.

## Expected directions (stated before computing, from `locality.md`'s confounded E4)

- **X1-a.** No detectable difference at σ_b = 4 and 8 m. On the confounded arms these were +0.019 and +0.026 in G_loc, with p 0.13–0.16.
- **X1-b.** Real-width lower at long scales (σ_b = 32–64 m, X = 80 m, L = 80 m) and for bumps 20–60 m before s\*.
- **X1-c.** In all 20 networks, the gain falls as the change becomes more local: monotone in σ_b and in X.

## Checks (the script stops and writes nothing if one fails)

1. Every arrays file matches its row's sha256.
2. Trapezoid G_loc equals the row's G on every kept key, to 1e-9. This checks the plumbing.
3. Each network's pooled trapezoid summary (median, lo, hi) equals its G in `decisions.json`, to 1e-9.
4. The trapezoid's permutation T and one-sided p equal H4R-G's in `decisions.json`, to 1e-9. This reproduces H4R's confirmed test.
5. The s\* recovery checks of `locality.md` pass, to 1e-9.
6. The frozen code and `scripts/locality_exploratory.py` are unchanged from HEAD.

## Outputs and cost

- **The code:** `scripts/locality_h4r_exploratory.py`.
- **The outputs:** `outputs/locality-h4r/results.json` (committed) and `results.csv`.
- **The findings note:** `docs/findings/locality-h4r.md`.
- **The cost:** minutes on the Mac. No network, solver or GPU is used.
