"""Step 2, receptive fields: how a network's corner behaviour responds to the road ahead.

Design: ``docs/interp/receptive-fields-design.md`` (draft v3; not frozen until the tag
``receptive-fields-prereg-v1``). Step 1's code (``src/apexzero/interp``,
``scripts/psychophysics.py``) is frozen at ``psychophysics-prereg-v2`` and is only imported
here, read-only.

For each stored answer key (design Section 2) both sides are differentiated with respect to
the stimulus bank's 2 m samples of curvature, left width and right width (Section 4): the
network through its float64 closed loop (a VJP per scalar outcome), the optimum through its
barrier KKT system at the re-polished stored solution (an adjoint solve per outcome). The
outcomes are the inside clearance c_min, step 1's v_min (here v_apex) and step 1's brake
onset s_b (Section 3); the headline measure is the edge-tracking gain G = 1 - dc_min/dDelta
for an inside-edge shift (Sections 4 and 7).

Module map:

- ``keys``: step 1's answer keys as read-only data: the A2 state set and the anchored finals
  per agent, strata labels, the rebuilt solver parameters P, Gate 0 (stationarity of the
  stored multipliers under the rebuilt P) and the one-solve re-polish.
- ``outcomes``: c_min, v_apex and s_b on a trajectory, as a NumPy plan on the base run (window,
  validity, argmin / onset fixed, near-tie flags) and JAX functions of the states and the track
  arrays that read the outcome at the fixed pick (Danskin derivatives).
- ``perturb``: sample-space perturbations (the trapezoid inside/outside shift, Gaussian bumps
  and their impulse-response combination, inside/outside recoding by turn direction) and the
  track arrays they act on: exact float64 upcasts of the agents' float32 arrays plus a float64
  tangent of ``generator.strip_arrays``, and the optimum's perturbed Track.
- ``kkt``: the condensed barrier KKT system, parameter derivatives F_p and the adjoint outcome
  fields of the optimum (with the certified inertia and the adjoint check), Gate 1's re-solves
  and scoring, Gate 4's G*, and the optimum's dose-response continuation.
- ``network``: the float64 closed loop (fixed-length ``lax.scan``, two road arrays for the
  path split), its VJPs, the kappa-clipping flags, the blind-pedal control, Gate 2, Gate 3's
  teacher-forced wiring check, branch consistency and the loop-length audit; trained
  networks' fields refuse until ``unlock_trained``.
- ``measures``: edge-tracking gains, buffer and the plateau rule, bump fields and their gain,
  shape, location, extent and reach, the shape-specificity criterion, the cross-geometry null
  and the phase ceiling, mirror-pair values, the cluster bootstrap over geometries, and the
  exclusion reasons and coverage (Section 6).
- ``controls``: planted fields RF-C1..C5 and the unlawful readers through the network pipeline
  (Section 8).
- ``gates``: Gates 0-5 as JSON-ready results (Section 8).
- ``analysis``: the decision rules of Section 7 on per-key measures (H2', H4 and its exact
  permutation test, H4-transfer, R18's 2-of-3 rule, H1-typical with its blind-pedal pairing,
  H1-fast, H2-speed, H3's lambda*), the fixed network lists and the regime cell sets.
- ``stages``: the batch stages' bodies (optimum fields, phase ceiling, network fields, dose,
  H3, analysis assembly), as functions tested before the tag.
- ``prereg``: the pre-registration record ``prereg.json`` the tag fixes (key lists and their
  hashes, re-polish movement, design and code hashes) and its verification.

``scripts/fields.py`` runs the stages (pre-tag: keys, repolish, gate1..gate5, controls,
opt-fields, phase-ceiling, prereg; post-tag stages refuse unless the freeze check passes:
the tag ``receptive-fields-prereg-v1``, no code change since it, and the record matching).

Pre-registration rule (design Section 8, Section 10): before the tag no field of a trained
network is computed; network-side code runs on random-init actors, planted policies and
scripted controllers only (``network.NetPolicy.trained`` enforces it).
"""
