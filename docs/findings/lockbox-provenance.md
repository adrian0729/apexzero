# Lockbox provenance: the sealed optima of 18 Oct agree with the lockbox's re-solved references to within 0.06% on all 12 R_TEST circuits (descriptive, done after the findings)

Results of 29 Oct 2026. This is protocol-v1's optional last step ([`docs/prereg/protocol-v1.md`](../prereg/protocol-v1.md) Section 14). It ran after the lockbox findings ([`lockbox.md`](lockbox.md)) were committed and pushed in `4b74084`. **It decides nothing and changes no number in `lockbox.md`.**
- **First read (the lead's account).** In this project's record, no result under `outputs/sealed/` was read before this comparison (one search traversed the folder on 27 Oct and printed nothing; `docs/STATUS.md`). Its README says that "nothing may read these until protocol-v1 is tagged and the lockbox is opened".
- **The code.** `scripts/lockbox_provenance.py` produced the output, `outputs/lockbox-provenance/provenance.json` (write-once). That file also holds the sha256 of each of the 28 sealed files.
- **Sources.** Every number below is read from that output.

## 1. What was sealed

| Path under `outputs/sealed/` | What it is | Compared with |
|---|---|---|
| `ocp/<circuit>.pkl` (12) | the first all-circuit s-OCP batch, 18 Oct 2026 (file times 16:43–16:52 local, Monza 17:19) | the lockbox's nominal original references, `outputs/lockbox/ocp/<circuit>__w+0.0.pkl`, each checked against its record's sha256 |
| `stray-2026-10-25/monza_ocp.pkl`, `monza_ocp_ds2.pkl` | two earlier Monza solves (16:39, and 16:40 at ds 2 m) | the lockbox's Monza reference |
| `stray-2026-10-25/ocp_summary_all_circuits.csv` | that batch's summary table, 25 circuits (the 12 R_TEST, the 12 R-dev and IMS) | the batch pickles' lap times |
| `to_ocp/speed/{fixed,v2}/` (12 files) | TO-OCP speed benchmarks: 60 s windows on Monza, Spa, Silverstone and Budapest | nothing. They are windows, not laps, so no lockbox quantity matches them |

The README says none of these was used for generator fitting, tuning or checkpoint choice, and the lockbox never read them (`analyse` reads only its own root's `ocp/`).

## 2. The batch against the lockbox references

All 12 old solves succeeded (`Solve_Succeeded`). Every one used the lockbox's settings (`meta` equal: `ds` 3 m, `margin` 1 m, `u_rate_weight` 1e-4) and has the same number of grid stations. The old pickles do not record `tol` or `max_iter`.

| Circuit | Old lap (s) | Lockbox lap (s) | Relative difference | Max curvature difference (1/m) | Max width difference (m, left / right) | Max v_x difference (m/s) |
|---|---|---|---|---|---|---|
| Austin | 89.927 | 89.935 | +0.008% | 1.1e-3 | 0.055 / 0.072 | 1.18 |
| Budapest | 75.012 | 75.031 | +0.025% | 6.5e-4 | 0.029 / 0.024 | 1.56 |
| Mexico City | 73.465 | 73.509 | +0.060% | 1.1e-3 | 0.032 / 0.041 | 1.46 |
| Montreal | 71.824 | 71.855 | +0.044% | 2.2e-3 | 0.032 / 0.015 | 0.16 |
| **Monza** | 80.949 | 80.949 | **0 (every array identical)** | 0 | 0 / 0 | 0 |
| Sakhir | 85.476 | 85.478 | +0.003% | 4.8e-4 | 0.023 / 0.041 | 1.21 |
| São Paulo | 67.454 | 67.468 | +0.021% | 9.3e-4 | 0.027 / 0.020 | 0.67 |
| Shanghai | 89.984 | 90.008 | +0.027% | 3.1e-3 | 0.040 / 0.056 | 0.50 |
| Silverstone | 84.325 | 84.334 | +0.011% | 8.1e-4 | 0.017 / 0.016 | 0.60 |
| Spa | 102.186 | 102.200 | +0.014% | 8.5e-4 | 0.025 / 0.016 | 0.18 |
| Spielberg | 65.145 | 65.156 | +0.017% | 2.0e-3 | 0.011 / 0.011 | 0.11 |
| Suzuka | 85.601 | 85.611 | +0.012% | 3.8e-4 | 0.017 / 0.025 | 0.19 |

- **The lap times agree to within 0.06%** (median +0.015%). The lockbox's lap is never faster than the old one.
- **What this does to the lockbox's gaps.** The old optima are slightly faster, so with them as the references every gap would be slightly larger: G becomes (1 + G)(1 + r) − 1, where r is the relative difference above. That is at most about 0.06 percentage points. At the median relative difference, the median subject's median gap would read about 5.18% instead of 5.16%; at the largest, 5.23% (computed here). These are estimates, not recomputations. They change no reading, since H1's gap bar is 4% and the gap is already above it.
- **Monza's solve is reproduced exactly.** Every array (geometry, grid, states, controls and duals), the meta and the iteration count (425) are identical; the two pickles differ only in `solve_seconds`. If the lead's reading below is right, the solver and the track pipeline reproduce exactly when given the same input.
- **The other 11 differ in geometry, not in settings.** The reference line's curvature differs by up to 3.1e-3 1/m and the widths by up to 7 cm, and the solves follow. The lead's reading, not checked: the 11 were solved at 16:43–16:52 on 18 Oct, after `15f3a5f` (16:41, the minimum-time OCP). Monza was re-solved at 17:19, just before `ede2a8a` (17:21, "C2 B-spline track lookup, identical in JAX, NumPy and CasADi"). That fits Monza alone having been solved with the track lookup the tagged code uses.

## 3. The stray files

- **`monza_ocp.pkl`** (16:39) matches the lockbox's Monza to within +0.005% (lap 80.945 s), with curvature differences up to 9.3e-4 1/m. It is not byte-identical to the batch's `Monza.pkl`.
- **`monza_ocp_ds2.pkl`** (ds 2 m) has a lap of 81.006 s, 0.075% slower than `monza_ocp.pkl`, the 3 m solve from the same session (80.945 s). Resampling at 2 m also changes the geometry, so this is the grid's sensitivity on one circuit, not its discretisation error alone.
- **`ocp_summary_all_circuits.csv`** rounds lap times to 1 ms. Its 12 R_TEST lap times match the batch pickles to within 0.5 ms, except Monza (−4.4 ms). Its Monza row matches the 16:39 solve (lap 80.945 s and 392 iterations, both those of `monza_ocp.pkl`), and the table is dated 16:53. So the table was most likely written before Monza was re-solved at 17:19.

## 4. What it means

- **The sealed optima and the lockbox's references are the same quantity, solved the same way.** Their lap times differ by at most 0.06%. The widths differ by at most 7 cm and the curvature by at most 3.1e-3 1/m. Nothing here bears on the lockbox's readings.
- **The lockbox's choice to re-solve rather than reuse** (protocol Section 4) cost the 12 nominal solves: 18.5 of the `ocp` stage's 167.5 solve-minutes, about a ninth, or roughly 6 minutes of its 56 minutes' wall time on 3 workers. The ±0.5 m solves had no sealed counterpart. Re-solving made the references depend only on the tagged code. The comparison shows that the reuse would have given nearly the same numbers.
- `outputs/sealed/` stays where it is. Anything further done with it is POST HOC.
