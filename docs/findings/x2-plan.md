# X-2 and X-3, locality at the real crash corners: plan (exploratory; written before anything here is computed, 30 Oct 2026)

**Status.** These are X-2 and X-3 of the follow-through plan (Section 9.2), the next steps that [`locality.md`](locality.md) Section 8 names (items 1 and 2). They are exploratory and not pre-registered with a tag. This plan is committed before anything below is computed.
- **Roads.** Only the three R-dev circuits of step 2's H3 are used (Norisring, Melbourne and Catalunya).
- **Network.** The only network is G2 (`dr-pilot-v2-tilt30`, step 2's pinned checkpoint `G2@199884800`).
- **Code.** The frozen code (`src/apexzero/fields`, `scripts/fields.py`) is imported read-only.

## The question

At G2's three real crash corners (12 unit-phases), step 2's H3 found the line moves away from the real inside edge by a median 7% of the edge's own move (range −4% to 38%). On synthetic corners, G2's corner-wide gain G is 0.83. `locality.md` showed that, on synthetic corners, the networks follow edge changes confined to a few metres at the apex with a gain of only 0.06–0.20. But it could not say whether that explains the real 7%.
- **X-2** asks: **at the real corners and G2's own start states, is the corner-wide G high, so that the real change's shape explains the low ratio (locality)? Or is it low there too, so that the corners or states explain it?**
- **X-3** asks: **do these first-order (linear) numbers hold for changes of finite size,** where the closest point may move to the perturbed spot?

## X-2: first order, at the 12 real unit-phases

- **The units.** Cells 584, 605 and 612 (`analysis.H3_CELLS`), each with phases 0–3. They are Norisring 874, Melbourne 2531 and Catalunya 4168, all right turns.
- **Each unit-phase uses H3's own recipe** (`stages.h3_lambda`): the start `network.cell_start`, the constant-width road, the width change dw = w_real − w_cw (`analysis.width_change`) and a 512-step loop. Then, per unit-phase, the following are computed:
  - G2's base run (`network.run`);
  - one VJP (`network.fields`): the c_min impulse rows;
  - the explicit term (`network.explicit_c_min`).

  That is 12 VJPs on the Mac.
- **The inside** is the right width, checked per unit-phase from the sign of the curvature at s\*.
- **Measures per unit-phase:**
  1. **The real profile's L/e,** for h = dw on the inside side, with e = b · h (the edge's own move at s\*) and dc = (impulse row) · h: L/e = (e − dc)/e. This is H3's inside-only ratio.
  2. **`locality.md`'s 24 profiles, in its three forms** (G_loc, L, L/e), through its own functions (`scripts/locality_exploratory.py`, read-only).
     - s\* comes from the explicit row's spline basis.
     - The ramp's end and the trapezoid use the stimulus's landmarks (s_c0, s_c1).
     - The trapezoid's G_loc is step 2's corner-wide G at this unit-phase.
  3. **The real profile's equivalent scale:** the bump σ_b at s\* whose L/e equals the real profile's. It is a log-linear interpolation on the σ_b grid, reported as below 2 m or above 64 m outside it.
- **Summaries:** the median over the 12 unit-phases, and per unit (the median over its 4 phases).

**The reading of X-2, fixed now.** Let T be the median trapezoid G over the 12 unit-phases, and S the median L/e of the σ_b = 2 and 4 m bumps.
- **"Locality explains the low real-road ratio"** if T ≥ 0.5 and S ≤ 0.2.
- **"The corners or states explain it"** if T < 0.3.
- **Otherwise, "mixed".**

## X-3: finite amplitude, closed loop, at the same 12 unit-phases

- **The changes.** The same base runs, with the inside width changed by A · h, for three profiles:
  - the σ_b = 4 m bump at s\*;
  - the X = 5 m ramp ending at s\*;
  - the trapezoid, as the reference whose linearity step 2's dose-response supports.

  The amplitudes are A ∈ {−1, −0.5, −0.2, +0.2, +0.5, +1} m (negative: the inside edge comes in). That is 216 closed-loop runs (`network.run`, `mode="secant"`, 512 steps, as H3 ran).
- **Measures per run:** c_min(A) and whether the run crashed. Finite L/e(A) = (e_h A − (c_min(A) − c_min(0))) / (e_h A), against the first-order L/e from X-2. Crashed runs are left out of L/e and counted.
- **Summaries:** per profile and amplitude, the median over unit-phases of L/e(A) − L/e_lin and of the ratio L/e(A)/L/e_lin, and the number of crashes.

**The reading of X-3, fixed now.** For each profile, first order "holds" if the median |L/e(A) − L/e_lin| is at most 0.05 at |A| ≤ 0.5 m and at most 0.10 at |A| = 1 m. It "fails" otherwise, with the direction read from the sign.

## Checks (the script stops and writes nothing if one fails)

1. **Each unit-phase's dc on both sides and its edge move reproduce `h3.json`'s** `dc_left`, `dc_right` and `edge_move`, to 1e-6 relative or 1e-9 absolute.
2. **The explicit row's basis:** its weights sum to 1 within 1e-9, and the explicit term on the outside width is 0 within 1e-9.
3. **The base run's c_min equals the run's own c_min(0)** in X-3 (the zero change), to 1e-9.
4. **The frozen code is unchanged from HEAD,** and step 2's tag `receptive-fields-prereg-v1` exists. Only then is `network.unlock_trained` called, as `scripts/costate_exploratory.py` does.

## Limits, stated now

- **One network, three corners.** There are 12 unit-phases, which are not independent (four per corner).
- **G2's own start states.** These are the cells' starts, not each phase's own lap state under every perturbation.
- **X-2 is first order** at the base argmin.

## Outputs

- **The code:** `scripts/x2x3_exploratory.py`.
- **The outputs:** `outputs/x2x3/results.json` (committed).
- **The findings note:** `docs/findings/x2x3.md`.
- **Reuse in GW-5:** the same machinery measures GW-5's networks (`gw5-plan.md`), with their own checkpoints at the same 12 unit-phases and starts.
