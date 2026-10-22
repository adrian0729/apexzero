# Locality of edge tracking: plan (exploratory; written before any computation, 22 Oct 2026)

**Status.** This is an exploratory follow-up to step 2
([findings](receptive-fields-step2.md), Sections 8 and 10). It is not pre-registered with a tag.
This plan is committed before anything below is computed, so the directions and measures are
fixed in advance. R_TEST stays sealed.

## Question

Step 2 found two apparently conflicting facts:
- when the inside edge shifts across the whole corner, the networks' lines follow about 80% of the
  move (G ≈ 0.76–0.85, confirmatory);
- on the real crash roads, the line follows only a median 19% of the edge's local move, and 7% of
  the inside edge's own change (descriptive, H3).

One explanation is that the networks track slow, corner-wide width changes but not short, local
ones, the kind real circuits have near the apex. This plan measures how the tracking gain
depends on the **length scale and placement** of an inside-edge change.

## Data (already computed; no network is run)

- **Per-key impulse responses** for all 35 network entries on the 481 A2 keys:
  - `outputs/fields/fields-a2/arrays/<network>_<timesteps>/<key>.npz`;
  - field `impulse`: 4 outcomes (c_min, v_apex, s_b, steer) × 3 features (κ, w_left, w_right) ×
    2,560 samples on the 2 m grid.
- **The optimum's impulse responses** on the same keys: `outputs/fields/opt-fields/arrays/<key>.npz`.
- **The keys, strata, exclusions and pairs** come from step 2's frozen code and record (primary,
  non-capped A2 keys; step 2's exclusion reasons; mirror-pair aggregation). They are read with
  `src/apexzero/fields` read-only.
- By linearity, the first-order response to any inside-width profile Δw_in(s) is the dot product
  of that profile with the inside-width impulse row. Inside is w_left on left turns and w_right on
  right turns.

## Measure

At each key, s* is the c_min argmin of the base run.
- The **local gain** for a unit-height inside-edge profile h, with h(s*) = 1, is
  G_loc(h) = h(s*) − dc_min/dΔ = σ dn(s*)/dΔ. It is the line's move toward the inside at the
  critical point per metre the inside edge moves there.
- For step 2's corner-wide trapezoid, G_loc equals step 2's G. This is a built-in check; agreement
  to 1e-9 is required.

**Families of h**, all with height 1 at s*:
1. **Gaussian bumps** centred at s*, σ_b ∈ {2, 4, 8, 16, 32, 64} m.
2. **Ramps ending at s*** (the edge moves in over the last X metres, as on the real roads): linear
   from 0 at s* − X to 1 at s*, then flat to s_c1 + 20 m, with X ∈ {5, 10, 20, 40, 80} m.
3. **Plateaus centred at s***, of lengths L ∈ {10, 20, 40, 80} m, with 20 m ramps, and step 2's
   corner-wide trapezoid as L_max.
4. **Bumps of σ_b = 8 m centred at s* + d**, with d ∈ {−80, −60, −40, −20, −10, 0, +10, +20} m
   (where along the corner the edge change happens). Here G_loc is reported per unit of the bump's
   peak, not at s*.

**Summaries.** Each network's statistic is the median over mirror pairs, with step 2's pair rules
and a cluster bootstrap over geometries. The optimum's G_loc on the same keys is the reference.

## Expected directions (stated before computing)

- **E1.** For the networks, G_loc falls as h becomes more local: as σ_b falls, as X falls, as L
  falls. At the shortest scales it falls well below the corner-wide G of about 0.8, toward the
  real-road values of about 0.1–0.2.
- **E2.** The optimum's G_loc stays at about 1 for every h whose height at s* is 1, because its
  inside contact is active.
- **E3.** For family 4, the networks respond most to edge changes some way before s*, a preview
  lead, and least to changes at or after s*.
- **E4, descriptive.** Real-width and baseline networks differ in G_loc at short scales. The
  direction is not predicted, because step 2's exploratory result (lower corner-wide G for
  real-width training) does not fix it.

A result against E1 would mean that locality does not explain the gap between the synthetic and
real-road results. The other candidates would then be the shape of the real corners, the
interaction with the outside width (H3: most of the line's move answers the outside width), and
nonlinearity.

## Outputs

- `scripts/locality_exploratory.py` (committed);
- `outputs/locality/` (results);
- a short findings note, `docs/findings/locality.md`.

Every number is labelled exploratory.
