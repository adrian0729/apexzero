# Line or execution: E0's lost time on R_TEST is mostly in the line, but in small moves. About 30% comes back on its own line, half within 0.37 m of it and 90% within 2 m (POST HOC, descriptive)

Results of 30 Oct 2026. This is the line / execution split that `protocol-v1` left for later (Sections 5 and 14). It is **POST HOC and descriptive**. The plan ([`line-execution-plan.md`](line-execution-plan.md)) was committed and pushed in `b0ecc77` before any R_TEST split number was computed. **It decides nothing and changes no number in [`lockbox.md`](lockbox.md).**
- **The code.** `scripts/line_execution.py` produced the outputs. They are `outputs/posthoc/line-execution/lockbox/split.json` (R_TEST) and `…/lb1-rdev-e0/split.json` (R-dev), both write-once.
- **One version of the code.** All 409 solve records (217 on R_TEST, 192 on R-dev) carry the same script sha256 (`af333919…`), which is that of the file committed with this note. Their `git_head` is the plan's commit, `b0ecc77`, at which the script was not yet committed, so the sha256 is what identifies the code.
- **What is committed.** The two `split.json` files and every solve's JSON record are committed. The solve pickles (79 MB) stay local and untracked, like the lockbox's laps; their sha256s are in the records.
- **Sources.** Every number below is read from those outputs. Section 4's per-circuit table goes beyond the plan's list and is labelled so.

## 1. The method in one paragraph

For each completed lap, the lap's minimum-time problem was solved again with the car's lateral offset held within ε of the line the car actually drove. Everything else stayed the same: car, corridor, slip limit and grid. T_ε is the fastest lap possible within ε of that line. **R(ε) = (T_a − T_ε) / (T_a − T*)** is the share of the car's lost time that better driving recovers without moving the line by more than ε. The rest needs the line itself to move. ε = 0.05 m stands for the car's own line. The rows are the 41 (subject, circuit) pairs behind the headline median gap: completed original circuits, one representative run each.

## 2. The answer

| ε (m) | R, median subject | R, per subject | R, pooled | R, range over rows | Gap on the car's line, G_exec (points) | Gap needing the line to move, G_line (points) |
|---|---|---|---|---|---|---|
| 0.05 | **0.295** | 0.248–0.358 | 0.333 | 0.227–0.561 | 1.55 | 3.68 |
| 0.25 | 0.439 | 0.374–0.476 | 0.466 | 0.349–0.694 | 2.37 | 2.98 |
| 0.5 | 0.571 | 0.491–0.612 | 0.579 | 0.451–0.787 | 3.03 | 2.39 |
| 1 | 0.745 | 0.699–0.796 | 0.745 | 0.588–0.911 | 3.99 | 1.43 |
| 2 | 0.905 | 0.887–0.928 | 0.898 | 0.788–0.973 | 4.69 | 0.53 |

- **By the plan's rule, the loss is "mostly line".** At the median subject, R(0.05) = 0.295, below the rule's 1/3. That is near the rule's edge, though the margin (0.038) is well above the solver's noise in R (about 0.004, Section 6). Pooled over all rows it is 0.3335, just above 1/3, so a pooled reading would be "mixed". Two of the five subjects are also above 1/3 (`e0-s51` 0.337, `e0-s52` 0.358).
- **In gap terms:** of the median subject's 5.16% gap, 1.55 points could be recovered by better driving on the car's own line and 3.68 points need the line to move. These are medians of each row's two parts, so they do not add exactly to 5.16.
- **The line changes needed are small, compared with how far the car's line is from the optimum's.**
  - Half the lost time comes back within **0.37 m** of the car's line (ε₅₀ at the median subject; per subject 0.29–0.52 m). Yet at the median subject the car's line is a median **1.93 m** from the optimum's (each row's median |n_a − n*|; the rows' 90th percentiles have a median of 4.74 m).
  - Across rows, ε₅₀ runs from below the grid to 0.68 m (`e0-s54` Silverstone). In three rows R(0.05) is already above 0.5: `e0-s50` Montreal, `e0-s52` Monza and `e0-s52` Spielberg.
  - Three quarters of the lost time comes back within 1 m, and 90% within 2 m. The 2 m figure says less: 2 m is about the typical distance to the optimum's line, so at that band the optimum can reach its own line at about half the stations.
  - So lines within a few tens of centimetres of the car's are already much faster. The time does not need the optimum's own line.

## 3. By corner type (pooled over the 41 rows)

| Type | Lost time (s) | Recoverable on the car's line, ε = 0.05 m (s) | Share | Recoverable within 0.5 m, share | Distance from the optimum's line, median / p90 over stations (m) |
|---|---|---|---|---|---|
| slow | 91.6 | 33.7 | 0.37 | 0.58 | 2.19 / 5.90 |
| medium | 49.9 | 13.6 | 0.27 | 0.58 | 1.50 / 3.84 |
| fast | 41.5 | 13.7 | 0.33 | 0.58 | 2.18 / 5.33 |

- **Slow corners hold half the lost time** (91.6 of 183.0 s, 50.1%). This is the same share as in `lockbox.md`.
- **The mix is about the same in every type.** On the car's own line, slow corners are a little more recoverable (37%) than medium ones (27%). Within 0.5 m, all three types reach 58%.
- The distances are pooled over the stations in each type's sectors, as the plan says. Over all stations they are 1.97 m (median) and 5.19 m (p90).

## 4. By circuit (beyond the plan's list; medians over each circuit's rows in `split.json`)

| Circuit | Rows | Gap | R(0.05) | R(1 m) | Distance from the optimum's line, median (m) |
|---|---|---|---|---|---|
| Austin | 2 | 5.95% | 0.26 | 0.69 | 2.37 |
| Budapest | 3 | 6.32% | 0.27 | 0.79 | 1.97 |
| Mexico City | 1 | 9.17% | 0.49 | 0.82 | 1.58 |
| Montreal | 5 | 7.03% | 0.45 | 0.83 | 1.89 |
| Monza | 3 | 5.59% | 0.42 | 0.86 | 1.91 |
| Sakhir | 2 | 5.68% | 0.34 | 0.67 | 1.82 |
| São Paulo | 5 | 4.80% | 0.26 | 0.63 | 2.31 |
| Shanghai | 1 | 5.34% | 0.44 | 0.73 | 2.66 |
| Silverstone | 5 | 4.67% | 0.28 | 0.67 | 2.10 |
| Spa | 5 | 4.45% | 0.29 | 0.78 | 1.61 |
| Spielberg | 4 | 4.97% | 0.32 | 0.77 | 2.10 |
| Suzuka | 5 | 5.57% | 0.29 | 0.74 | 2.07 |

- **The two circuits with the largest gaps have the highest R(0.05).** Mexico City (9.2%, one row) and Montreal (7.0%) have 0.49 and 0.45. Shanghai (0.44) and Monza (0.42) come close, and the other eight are 0.26–0.34. Across the 41 rows, R(0.05) rises weakly with the gap (Spearman 0.34; not in the plan).
- **The distance from the optimum's line does not track the gap.** Across the 41 rows the two are weakly and negatively related (Spearman −0.19; not in the plan). Mexico City has the largest gap and one of the closest lines (1.58 m).
- Both correlations are over rows that cluster by circuit (up to five rows each), so they mostly compare circuits.

## 5. R-dev, the same way (in-sample for widths)

| ε (m) | R, median subject | R, pooled | G_exec / G_line (points) |
|---|---|---|---|
| 0.05 | 0.249 | 0.282 | 1.60 / 4.74 |
| 0.25 | 0.399 | 0.420 | 2.55 / 3.76 |
| 0.5 | 0.529 | 0.543 | 3.36 / 2.96 |
| 1 | 0.719 | 0.714 | 4.47 / 1.94 |
| 2 | 0.890 | 0.872 | 5.47 / 0.73 |

- The 36 rows give the same picture: "mostly line" (R(0.05) = 0.249), with ε₅₀ 0.45 m at the median subject. The car's line is 1.78 m from the optimum's (the median subject's median over rows; 1.91 m pooled over all stations).
- **R_TEST's smaller gap is in the line part.** The median subject's gap is 5.16% on R_TEST against 6.31% on R-dev. The part recoverable on the car's own line is about the same (1.55 against 1.60 points); the part needing the line to move is smaller (3.68 against 4.74).
- By corner type at ε = 0.05 m, 30% (slow), 22% (medium) and 31% (fast) is recoverable on the car's line.

## 6. Checks

- **Every solve converged on its first attempt.** That is 205 fixed-line solves and 12 noise-floor solves on R_TEST, and 180 and 12 on R-dev; no fallback was used. Every shooting error was below 2.9e-4, against the lockbox's limit of 3e-3. No row's value is missing.
- **The solver's noise floor.** The 12 references re-solved from their own arrays (inputs reproduced to 2e-13) land within 2.0e-4 of the stored lap times on R_TEST (largest: Mexico City −2.0e-4, Austin −1.9e-4, Monza +1.8e-4) and within 4.2e-4 on R-dev (Hockenheim). At a 5% gap, 2e-4 of a lap is about 0.004 in R.
- **Consistency held everywhere.**
  - No T_ε was below T*.
  - None increased with ε.
  - No car beat the optimum on its own line.
- **The band held the line** at every station; the solutions stayed inside it to within 9e-8 m. The largest jump in the car's line across the start line was 0.035 m (R-dev 0.126 m).
- **The representative runs' gaps** are within 1.3e-4 (0.013 points) of the lockbox's gaps for their rows.
- **Sector closure:** the corner parts sum to the lap totals within 1.8e-15 s.
- **Cost:** 417 solve-minutes on R_TEST and 371 on R-dev. Run side by side on the Mac at 4 workers each, both finished within 1 h 46 min. $0.

## 7. What it means (the lead's reading, not tested)

- **The agents' lines are roughly right, but not precise.** On their own line, better speed, braking and throttle would recover only about 30% of the lost time. At the median subject, moving the line by 0.05–1 m recovers a further 0.45 of it, and 0.26 needs moves of more than 1 m. Where along a corner those small moves are made (turn-in, apex or exit) was not measured.
- **This is not two independent causes.** A car that brakes later must take a different line, so the split counts as "line" anything the optimum gains by moving sideways. ε = 0.05 m also lets the optimum tidy the line by up to 5 cm. The 3 m grid's error, about 0.075% of a lap, is about 1.5% of the lost time.
- **It fits the locality lead** (`receptive-fields-step2.md`, `locality.md`). The networks follow corner-wide edge changes but barely follow short, local ones. Precise local placement is where this split finds the time. That is a consistency, not a test; X-1 would test it.
- **For a write-up:** the largest part of the headline 5.2% gap is line placement within a metre of the car's line (0.45). It is larger than what better speed control on the same line recovers (0.30), and larger than what needs bigger line changes (0.26). The optimum also knows the whole track in advance, while the car sees only its observations.
