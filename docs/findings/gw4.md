# GW-4: `generator-v1` uses copied R-dev widths; `generator-v1` and `sim-v1` tagged

Applied 28 Oct 2026. This applies the rule fixed in [`docs/prereg/gw3-width-round.md`](../prereg/gw3-width-round.md) Section 5 before any GW-3 run (follow-through Section 7.4; decisions #12 and #13). GW-3's results are in [`gw3.md`](gw3.md). The rule is deterministic: nothing was chosen here.

## 1. The decision

**`generator-v1` uses copied R-dev width profiles (`train_dr.py --real-widths`).** E0 trains on them, and every R-dev number for E0 is labelled **"in-sample for widths"** wherever it is reported (`gw3.md` Section 1 lists what the label covers). R_TEST is unaffected.

| # | Criterion | Where it was read | Result |
|---|---|---|---|
| 1 | P1 confirms | GW-3's analysis (`outputs/gw3/analysis.json`, `ab252fd`) | holds |
| 2 | D2 ≥ −0.5 | the same (D2 = +1.175) | holds |
| 3 | Δgap < 1.0 point | the same (Δgap = +1.520) | **fails** |
| 4 | The width report, restated | the same; the report's sha256 re-checked here: `5871664b…`, the pinned value | holds |
| 5 | The default path stays bit-identical | here, at `83f1797` (Section 2) | holds |

- **Why copied widths.** Criterion 3 fails, so (2a) fails the adoption rule. #12 then gives copied R-dev widths if and only if S1-v1 confirms, and it does: copied widths above `base` on held-out circuits in v1, 7.4 against 4.6, p = 1.1e-4 (`gw3.md` Section 4).
- **Criterion 5 did not decide anything.** GW-3's analysis recorded "copied" for both of its outcomes. It is read here, as the prereg requires, and it holds.

## 2. Criterion 5 at `83f1797`

`83f1797` is main's head when the tests ran, with a clean tree (`git status` showed only two untracked planning notes outside `src/`, `scripts/` and `tests/`). The tag commit (Section 3) differs from it only by this file. The prereg lists ten tests; the PPO test is parametrised over two environments, so pytest counts eleven items. All pass (11 passed in 134 s; log `runs/gw4-criterion5.log`, not committed):

- `tests/test_stat_widths.py::test_default_is_bit_identical`
- `tests/test_stat_widths.py::test_default_bank_is_bit_identical`
- `tests/test_stat_widths.py::test_train_dr_default_path_is_unchanged`
- `tests/test_real_widths.py::test_default_is_unchanged`
- `tests/test_reexpress.py::test_default_and_2a_unchanged_and_2b_fit` ((2b) is in the tree)
- `tests/test_value_gradient.py::test_default_racing_update_reproduces_the_pre_term_values`
- `tests/test_value_gradient.py::test_default_rollout_records_no_capture_and_defaults_are_off`
- `tests/test_value_gradient.py::test_train_dr_default_path_passes_no_term`
- `tests/test_ppo.py::test_make_train_reproduces_the_pre_split_update` (CartPole-v1 and Pendulum-v1)
- `tests/test_value_gradient.py::test_train_dr_skip_eval_rdev_writes_no_rdev_result_and_the_default_is_unchanged`

## 3. What the tags pin

Both tags are annotated and placed on the commit that adds this file. Each pins that whole commit. The lists below say which code defines each one; E0's pre-registration fixes the checks E0 runs against them.

**`generator-v1`: the training track generator.**
- `scripts/train_dr.py --real-widths`, with no `--real-widths-from` (all 12 R-dev circuits' width profiles), v1 curvature (no `--tilt-*` flag, `train_dr.py`'s default), and `train_dr.py`'s default bank and refresh.
  - The grammar is `apexzero.tracks.generator.fit_grammar(dev, real_widths=True)`, fitted on the 12 R-dev centrelines.
  - Each strip's per-side widths are a wrapped segment of one R-dev circuit's profile, measured from its `refine_reference` line as the evaluation measures it (`train_dr.py`'s docstring).
- The code: `src/apexzero/tracks/` (`generator.py`, `reference.py`, `spline.py`, `track.py`, `corners.py`; `widths.py` and `reexpress.py` are present and unused by this setting) and `train_dr.py`'s grammar and bank set-up.
- The data: the 12 R-dev centreline files `data/tracks/tumftm/<circuit>.csv`. They are not tracked, so their sha256 are recorded here:

| Circuit | sha256 |
|---|---|
| BrandsHatch | `9c69b09cef717d8abffc7c9450e59d7e339ac0dd0c7ed8121d73fe2f373145bb` |
| Hockenheim | `5eb6b7d8ff7b9fc905b38caed9a65d8e76e4bd7af5af9ba82aecdf4a6e0eacc5` |
| MoscowRaceway | `843c0f47e7210b5c9bfb3bf2f55cfee02cced0970f9b711a951b650c5cad3753` |
| Norisring | `8857d3c362ad2923c1f93c8d257498f50459770b9021adcc7969b71085c31d9a` |
| Nuerburgring | `9fea2a9daa25311bb251be416140b6c55cb2e53bb4f69c28f512fbfce25a3f87` |
| Oschersleben | `9ce893ee1fde37cc0626d654b765e65cd7159ffca922496ab881d61cad6ab304` |
| Sepang | `89ee026fd439e9b41311fc742b44efb120c3c1148f55e5963b38b62971352eaf` |
| Sochi | `caf3f00d78a7b54aeab131d2a8f263131c33d5431292a05dcbaaad49a19c351c` |
| YasMarina | `befab5a3be5a6ab320ce16d5ac5ec9016d633c5d6e739b9483f4d6364f0fe017` |
| Melbourne | `f9ea4395a25dc4189d8b8c692ce9656682de62ad7b784dfd7c024ac8d4f245e9` |
| Catalunya | `8d6dc9d50e98bdecd818cb880a4b12bb5cec25d699a6b774e0b593c0f8a9da5e` |
| Zandvoort | `f3f2a82c9f41bcee8d7161a6ad6c0b8e6d4b902dd772459918c9fe7f7451bc59` |

- The audit level sets (`tracks/levels.py`, `pilot_audit.py`, `interp/`) keep the default width model, as `train_dr.py` documents. `generator-v1` changes only training.

**`sim-v1`: the simulator,** at `train_dr.py`'s defaults: dynamics, track lookup, reward, termination, observations, window and context sampling.
- The code: `src/apexzero/vehicle/` (`dynamics.py`, `params.py`, `sim.py`, `ops.py`), `src/apexzero/env/` (`racing.py`, `adapters.py`: the per-reset car context in the plan's ranges, plus hidden ±2% grip) and the track lookup in `src/apexzero/tracks/` (`track.py`, `reference.py`, `spline.py`).
- **No training-only termination margin.** GW-5 did not run, so #13's margin option is not part of `sim-v1`. Training and evaluation use the same `edge_margin`.
- The value-gradient term (VG-0) is off by default (`--vg-coef 0`), and criterion 5's tests show the default path is unchanged. VG-1's positive P does not change E0 (VG-1 prereg Section 6 and #5).

## 4. What this does not decide

- **E0's budget** (5 seeds × 1B steps; about 3.5–4 A6000 hours by follow-through Section 3) is not approved. It comes back to the user with its estimate before it runs.
- **E0's pre-registration**, its checks against these tags and its checkpoint rule (#15) are still to be written.
- **#14's open questions** (whether to go ahead with the lockbox if E0 misses the 80% gate, and whether to split R_TEST 6 + 6) are asked before `protocol-v1`.
