"""Planted fields and unlawful readers through the network pipeline (design Section 8).

Every control here runs through exactly the code an agent's field goes through:
``network.run`` (the float64 closed loop on the exact upcast road, network.py:581) and
``network.fields`` (one VJP of the outcome vector with respect to the three sample arrays,
network.py:653), then ``measures.bump_field`` and the comparison measures. The answers are
computed independently, from the spline code the optimum uses (``packed_coeffs`` and
``spline_np``, spline.py:70 and :141) or from the controller's own law. Run in a float64
process (``python -m apexzero.fields.controls``, or ``scripts/fields.py controls``).

Instantaneous fields (RF-C1, RF-C2, RF-C3, RF-C4) are the steer mean at an A2 start's
decision step 0, from a 1-step loop (the outcome vector's steer, network.py:616-619); the
plan-level control (RF-C5) and the readers use the full loop.

**RF-C1** (``rf_c1``): steer = c o[5k* + 4] on the raw observation (``network.kappa_readout``;
o[5k + 4] = 50 kappa(s_0 + d_k), racing.py:92-107), for k* near, mid and far (preview
distances closest to 20, 80 and 200 m at the start speed). Known field along a bump G_d:
(c / sqrt(var)) 50 kappa_at(s_1 + d_k*; strip_arrays(G_d)) (var = 1 on the raw observation;
with an obs_rms the readout reads the normalised channel, var its running variance + 1e-8).
Pass, at sigma_b = 4, 8 and 16 m: cosine >= 0.999, gain within 1%, peak within one bin (2 m)
of d_k*, E90 within one bin of d_k* + 1.28 sigma_b. ``run_all`` runs it on the raw observation
and, at its first cell, also with G1's obs_rms (read only; ``rf_c1(..., obs_rms=...)``, whose
scale 1 / sqrt(var + 1e-8) is recorded), so the normaliser's scale is part of the test.

**RF-C4** (``rf_c4``): steer = c1 o[5k1 + 1] + c2 o[5k2 + 3] (``network.edge_readout``), the
body-frame y of the left edge at preview point k1 and of the right edge at k2, |steer| <= 0.5.
Known fields (edges_at, sim.py:135-140): on w_left (c1 / 100) cos(theta(s_k1) - psi) times
the spline of G_d at s_k1, on w_right -(c2 / 100) cos(theta(s_k2) - psi) times its spline at
s_k2 (theta linear in s, psi = theta(s_0) + xi_0). Pass as RF-C1 on both width fields, and the
turn-direction recoding table: on a left key w_in is the left field and w_out the right one,
on a right key the reverse, the steer itself flipped (``measures.turn_fields``), on both keys
of a mirror pair.

**RF-C2** (``rf_c2``): Q's steer (``controls.pursuit_steer``, interp/controls.py:551-573) reads
the road only through the pose: pure pursuit to the centreline point at s + l, l =
max(8 m, 0.6 v). A curvature change delta kappa ahead moves that point's body-frame y by
int_0^l (l - u) delta kappa(s + u) du on a straight approach (a change behind the car moves
car and road rigidly), so the field along G_d is gain K(d) with K(d) = int_0^l (l - u)
G(u - d) du. The gain d steer / d y_target (the kinematic and understeer map) is the central
difference of the frozen ``pursuit_steer`` with the centreline knots ahead shifted across the
car's heading (``pursuit_gain``). Pass: cosine >= 0.999 at every sigma_b, the field through
kappa_c alone exactly 0, no width field, and the kappa_c and pose paths summing to the total
to 1e-9 (``path_split``), for the frozen Q; and the same plus the amplitude within 1% for Q
steering with ``pursuit_steer_ad``. On a straight the pursuit curvature is exactly 0, where
JAX differentiates the frozen understeer term sign(kappa) us(|kappa|) to 0: the frozen Q's AD
field misses the understeer slope (14% of the gain at cell 344), reported as
``understeer_share``; ``understeer_ad`` keeps the value and the slope. Q's speed target is
precomputed per stimulus (its aux row), so only its steering is used (design Section 8).

**RF-C3** (``rf_c3``): a random-init actor of G1's architecture with G1's obs_rms (read only;
``network.random_net``, step 1's recipe). Only its instantaneous steer field is used: the full
loop must end before the outcome window's end (its plan-level outcomes are exclusions), and
the steer field must be finite, non-zero and pass Gate 2 (``network.gate2``).

**RF-C5** (``rf_c5``): planted edge trackers (``edge_keeper``), scripted policies that read the
state and the road. Target line n_t(s) = sigma (w - 1 - B(s)), B(s) = B0 + A (1 -
exp(-((s - s_apex) / l)^2 / 2)) with A = W0 - 1 - B0 (so the line is on the centreline far
from the apex and B0 = 1.5 m from the inside termination boundary through the corner); w is the
road's inside width (``reads_width`` 1: G = 1 by construction) or the constant W0, the base
road's inside width at the apex (0: a fixed n, G = 0). Steering: kinematic feed-forward of the
target path's curvature kappa / (1 - n_t kappa) + n_t'' and heading atan(n_t' / (1 - n_t
kappa)) (derivatives of n_t by AD of the width spline), plus k_xi (heading error) + k_n
(n_t - n); speed: the lane keeper's look-ahead speed law (rollout.py:157-162) at 6 m/s^2 of
lateral acceleration (its 12 m/s^2 drives the inside line too hard for the tracker). Pass:
|G - 1| <= 0.02 and |G| <= 0.02, with c_min valid and the edge's own move at the argmin
e(s*) >= 0.99 (``measures.edge_move``, the analysis's own plateau test; the piecewise-linear
h(s*) == 1 would pass a 0.983 spline value at the plateau's corners).

**Unlawful readers** (``readers``): step 1's Q family, on the observation's preview so that the
perturbed road reaches their speed targets (Q, c1, c2 and c3 precompute theirs per stimulus,
interp/controls.py:1117-1139, and would read no field): Qp as step 1 builds it
(``preview_profile``, :619), c1 reading the preview 15 m late, c2 with kappa capped at 1/12,
c3 braking on a time-to-corner trigger (``_c3_target``, :659, with tau_c = step 1's tuned
1.1889 s) at the first preview point whose |kappa| exceeds 1e-3. Steering is Q's pursuit
with ``pursuit_steer_ad`` (see RF-C2). Two pieces of the frozen law
are made differentiable without changing a value: the QSS pass's sqrt(1 - lat^2) at lat = 1
(NaN gradients through ``jnp.clip``) gets a double-where (``grip_profile_ad``), and the pedal
law's bisection (whose AD derivative is that of the bracket, not of the root) gets the
implicit-function derivative of its fixed point (``law_a_ad``).

Each violator is judged for **specificity against Qp**, its base law (``judge_reader``): Qp is
not lawful on these keys (it fails H1-typical's gain and shape against the TO-OCP, reported as
``qp_lawful``), so a violator failing the absolute criteria would say nothing about its
violation. c3: its s_b gain ratio to Qp outside [0.5, 2]; c2: its v_apex gain ratio to Qp
outside [0.5, 2], on capped corners (R < 12 m) only; c1: a constant late read leaves the
field's shape and location unchanged (translation invariance), so it is judged on its base
s_b against Qp's, which must be at least 7.5 m later (H1-fast's value test; ~ +12 to +15 m).
Each must be detected on every cell where it is judged (c1 on at least two). Negative checks
with the same judging code must not be flagged: Qp judged as each violator against itself,
and c2 on corners above its cap (where it is Qp). H1-typical's absolute criteria (gain,
location, and shape as the specificity of each outcome's field against the optimum's fields
at far geometries, ``measures.shape_specificity``, with separate s_b and v_apex nulls) are
reported for every reader, descriptively.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from apexzero.env import racing
from apexzero.fields import keys as K
from apexzero.fields import measures as MS
from apexzero.fields import network as N
from apexzero.fields import perturb as PT
from apexzero.fields.perturb import Samples
from apexzero.interp import controls as QC
from apexzero.interp import measures as M
from apexzero.interp.constants import OBS_PREVIEW
from apexzero.tracks.spline import packed_coeffs, spline_np
from apexzero.vehicle.params import F1_2025
from apexzero.vehicle.sim import global_pose, kappa_at, lookup, widths_at

COS_MIN = 0.999            # RF-C1, RF-C2, RF-C4: cosine with the closed form
AMP_TOL = 0.01             # RF-C1, RF-C4: |gain - 1|
BIN = PT.D_STEP            # m: one bin of the d grid (peak, E90)
PATH_TOL = 1e-9            # RF-C2: kappa_c path + pose path = total (relative)
RF_C5_TOL = 0.02           # RF-C5: |G - 1| and |G|
REACH = {"near": 20.0, "mid": 80.0, "far": 200.0}   # m: RF-C1's k* by preview distance
RF_C1_C = 0.3
RF_C4_ARGS = (8, 3.0, 12, -2.0)      # k1, c1, k2, c2: |steer| ~ 0.3 on a 6 m half-width
Q_CV = 0.9                 # step 1's tuned c_v (outputs/psychophysics/controls/tuning.json);
                           # Q's steering does not read it
TAU_C = 1.1889034489393233   # s, step 1's tuned c3 trigger (tuning.json, ruling R26)
C1_SHIFT = QC.C1_SHIFT     # m, c1 reads the preview this much late
KAPPA_ON = 1e-3            # 1/m, c3: the corner starts where |kappa| first exceeds this
KEEPER = {"B0": 1.5, "k_n": 0.1, "k_xi": 1.2, "a_lat": 6.0, "a_brake": 15.0, "v_cap": 60.0}
KEEPER_STEPS = 200
READERS = ("Qp", "c1", "c2", "c3")
# What each violator is judged on, against Qp (its base law): c1 by the base s_b offset (a
# constant late read leaves the field's shape and location unchanged, by translation
# invariance), c2 by its v_apex gain ratio on capped corners, c3 by its s_b gain ratio.
READER_VIOLATES = {"Qp": None, "c1": ("s_b", "offset"), "c2": ("v_apex", "gain_ratio"),
                   "c3": ("s_b", "gain_ratio")}
READER_RATIO_BAND = (0.5, 2.0)   # a gain ratio to Qp outside this is a detected violation
C1_OFFSET_MIN = 0.5 * QC.C1_SHIFT   # m: c1's base s_b minus Qp's at least this (about +15)
RF_C1_CELLS = (344,)       # A2, D = 250 m (a straight approach), left
RF_C4_CELLS = (344, 345)   # the mirror pair (R 44.7 m / 45 deg)
RF_C2_CELLS = (344, 345)
RF_C3_CELLS = (62,)
RF_C5_CELLS = (344, 345, 342, 340)
READER_CELLS = (46, 110, 94, 134, 343, 50)   # R 28.6 / 90, R 14.6 / 180 (braking A2 starts),
                               # R 9.4 / 180 (below c2's cap), R 44.7 / 180, R 22.9 / 45 and
                               # R 35.8 / 90 (c1 reaches the window end on 46, 134, 343, 50)

# ------------------------------------------------------------------------- starts and fields


def start(cell_id: int, phase: int = 0) -> tuple[N.Start, N.Road]:
    """A designed cell's start (``network.cell_start``) and its road."""
    bank, cells = K.design()
    st = N.cell_start(cells[int(cell_id)], int(phase), bank)
    return st, N.road(bank, st.stim_id)


def steer_field(net: N.NetPolicy, rd: N.Road, st: N.Start) -> tuple[float, Samples]:
    """(steer, d steer / d samples) at the start's decision step 0, through the pipeline with
    a 1-step loop."""
    if st.decision_step != 0:
        raise ValueError("instantaneous fields are read at decision step 0 (an A2 start)")
    r = N.run(net, rd, st, steps=1)
    fl = N.fields(net, rd, r, steps=1, allow_exhausted=True)    # only the decision steer
    return r.steer, fl.of("steer")


@functools.lru_cache(maxsize=32)
def _instant_jit(fn, keep: tuple[str, ...]):
    def steer0(params, obs_rms, base, samples, li, delta):
        tan = PT.strip_tangent(samples, delta)
        tr = base._replace(**{k: jnp.asarray(getattr(base, k), jnp.float64) + getattr(tan, k)
                              for k in keep})
        st = N.env_state(li.x0, li.prev_action, li.accel, li.ctx)
        o, _ = N.policy_input(N.observe(st, tr, li.ctx_obs), obs_rms)
        return fn(params, o, st, tr, jnp.zeros((), jnp.int32))[0][0]

    def field(params, obs_rms, base, samples, li, delta0):
        y, pull = jax.vjp(lambda d: steer0(params, obs_rms, base, samples, li, d), delta0)
        return y, pull(jnp.ones_like(y))[0]

    return jax.jit(field)


def instantaneous(net: N.NetPolicy, rd: N.Road, st: N.Start,
                  keep: tuple[str, ...] = PT.FLOAT_FIELDS) -> Samples:
    """d steer_0 / d samples with the strip_arrays tangent applied to the ``keep`` fields
    of the TrackArrays only (a path of the road into the policy)."""
    N.require_x64()
    N._refuse_trained(net, "instantaneous fields")
    base = jax.tree.map(jnp.asarray, rd.base)
    _, imp = _instant_jit(net.fn, tuple(keep))(net.params, net.obs_rms, base, rd.samples,
                                               N.loop_inputs(st, 1), PT.zeros(rd.n))
    return Samples(*(np.asarray(a) for a in imp))


def path_split(net: N.NetPolicy, rd: N.Road, st: N.Start, total: Samples | None = None
               ) -> dict:
    """The instantaneous steer field split by the road's paths into the policy: the kappa
    channel (kappa_c), the pose (x, y, theta) and the widths (w_left_c, w_right_c). Returns
    the parts' max |field| and the relative error of their sum against ``total`` (the
    pipeline's field)."""
    paths = {"kappa_c": ("kappa_c",), "pose": PT.POSE_FIELDS, "widths": ("w_left_c", "w_right_c")}
    parts = {k: instantaneous(net, rd, st, v) for k, v in paths.items()}
    total = instantaneous(net, rd, st) if total is None else total
    tot = np.concatenate([np.asarray(a) for a in total])
    summed = sum(np.concatenate([np.asarray(a) for a in p]) for p in parts.values())
    scale = max(float(np.abs(tot).max()), 1e-300)
    return {"max_abs": {k: float(max(np.abs(np.asarray(a)).max() for a in p))
                        for k, p in parts.items()},
            "sum_rel_err": float(np.abs(summed - tot).max() / scale), "parts": parts}


def spline_at(samples, s: float, ds: float = PT.DS) -> np.ndarray:
    """The natural cubic spline of each row of ``samples`` [M, N] (or [N]) at s, with the
    optimum's spline code (``packed_coeffs`` of an open strip, ``spline_np``)."""
    rows = np.atleast_2d(np.asarray(samples, float))
    n = rows.shape[1]
    return np.array([spline_np(packed_coeffs(r, False, n), np.array([float(s)]), ds, n, False)[0]
                     for r in rows])


def check_field(F: np.ndarray, cf: np.ndarray, d: np.ndarray, d_k: float, sigma_b: float
                ) -> dict:
    """RF-C1 / RF-C4's criteria for one field against its closed form."""
    cos, g = MS.cosine(F, cf), MS.gain(F, cf)
    pk, reach = MS.peak(F, d), MS.e90(F, d)
    want = float(d_k + MS.E90_BLUR * sigma_b)
    ok = (cos >= COS_MIN and abs(g - 1.0) <= AMP_TOL and abs(pk - d_k) <= BIN
          and abs(reach - want) <= BIN)
    return {"sigma_b": float(sigma_b), "cosine": cos, "gain": g, "peak": pk, "d_k": float(d_k),
            "e90": reach, "e90_want": want, "pass": bool(ok)}


def preview_distances(st: N.Start) -> np.ndarray:
    """d_k of the 40 preview points at the start speed (racing.preview_offsets), float64."""
    v = max(float(st.x0[3]), 0.0)
    k = np.arange(1, racing.N_PREVIEW + 1) / racing.N_PREVIEW
    return np.clip(6.0 * v, 150.0, 500.0) * k ** 1.5


def _d_grid(st: N.Start, extra: float = 0.0) -> np.ndarray:
    """-20 m to the start's support end (``measures.support_end``) plus ``extra``."""
    return MS.d_grid_to(MS.support_end(st.x0[None]) + extra, st.s1)


# ------------------------------------------------------------------------- RF-C1


def rf_c1(cell_id: int = RF_C1_CELLS[0], c: float = RF_C1_C, obs_rms=None,
          sigma_bs=(4.0, 8.0, 16.0)) -> dict:
    """RF-C1 at a cell's phase-0 start: near, mid and far k* (module docstring)."""
    st, rd = start(cell_id)
    dks = preview_distances(st)
    d = _d_grid(st)
    s_bump = PT.sample_s(rd.n)
    out: dict[str, Any] = {"cell": int(cell_id), "c": float(np.float32(c)), "readouts": {}}
    ok = True
    for name, target in REACH.items():
        k = int(np.argmin(np.abs(dks - target)))
        W, b = N.kappa_readout(k, c)
        if obs_rms is None:
            net, scale = N.linear_net(W, b, f"rf-c1-{name}"), 1.0
        else:
            net = N.callable_net(N.linear_policy, (W, b), obs_rms, f"rf-c1-{name}")
            scale = 1.0 / np.sqrt(float(net.obs_rms.var[5 * k + 4]) + 1e-8)
        steer, imp = steer_field(net, rd, st)
        cc = float(np.float32(c)) * scale
        rec = {"k": k, "d_k": float(dks[k]), "steer": steer, "scale": float(scale),
               "width_field_max": float(max(np.abs(imp.w_left).max(), np.abs(imp.w_right).max())),
               "checks": []}
        for sb in sigma_bs:
            F = MS.bump_field(imp.kappa, st.s1, d, sb)
            cf = cc * 50.0 * spline_at(PT.bump_profiles(s_bump, st.s1, d, sb), st.s1 + dks[k])
            rec["checks"].append(check_field(F, cf, d, dks[k], sb))
        rec["pass"] = bool(all(ch["pass"] for ch in rec["checks"])
                           and rec["width_field_max"] == 0.0)
        ok &= rec["pass"]
        out["readouts"][name] = rec
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------------- RF-C4


def _theta(base, s: float) -> float:
    """The pose heading at s (``lookup_angle`` on an open strip: linear in s)."""
    th = np.asarray(base.theta, float)
    return float(np.interp(s, np.arange(len(th)) * float(np.asarray(base.ds)), th))


def rf_c4_closed_form(st: N.Start, rd: N.Road, d: np.ndarray, sigma_b: float,
                      args=RF_C4_ARGS) -> dict[str, np.ndarray]:
    """The width readout's known bump fields on w_left and w_right."""
    k1, c1, k2, c2 = args
    c1, c2 = float(np.float32(c1)), float(np.float32(c2))
    dks = preview_distances(st)
    psi = _theta(rd.base, st.x0[0]) + float(st.x0[2])
    prof = PT.bump_profiles(PT.sample_s(rd.n), st.s1, d, sigma_b)
    s1_, s2_ = st.x0[0] + dks[int(k1)], st.x0[0] + dks[int(k2)]
    left = c1 * np.cos(_theta(rd.base, s1_) - psi) / 100.0 * spline_at(prof, s1_)
    right = -c2 * np.cos(_theta(rd.base, s2_) - psi) / 100.0 * spline_at(prof, s2_)
    return {"w_left": left, "w_right": right}


def rf_c4(cells=RF_C4_CELLS, args=RF_C4_ARGS, sigma_bs=(4.0, 8.0, 16.0)) -> dict:
    """RF-C4 on the cells (a mirror pair), with the recoding table."""
    k1, c1, k2, c2 = args
    out: dict[str, Any] = {"args": [int(k1), float(c1), int(k2), float(c2)], "cells": {}}
    ok = True
    for cid in cells:
        st, rd = start(cid)
        net = N.linear_net(*N.edge_readout(int(k1), c1, int(k2), c2), "rf-c4")
        steer, imp = steer_field(net, rd, st)
        dks = preview_distances(st)
        d = _d_grid(st)
        rec: dict[str, Any] = {"sigma": st.sigma, "steer": steer, "checks": {}, "table": {}}
        for sb in sigma_bs:
            cf = rf_c4_closed_form(st, rd, d, sb, args)
            for side, k in (("w_left", k1), ("w_right", k2)):
                F = MS.bump_field(getattr(imp, side), st.s1, d, sb)
                ch = check_field(F, cf[side], d, dks[int(k)], sb)
                rec["checks"].setdefault(side, []).append(ch)
        # the recoding table at sigma_b = 8: the pipeline's recode against the expected sides
        cf = rf_c4_closed_form(st, rd, d, 8.0, args)
        turn = MS.turn_fields(imp, st.sigma, "steer")
        inside, outside = PT.side_feature(st.sigma, "inside"), PT.side_feature(st.sigma, "outside")
        for name, side in (("w_in", inside), ("w_out", outside)):
            got = MS.bump_field(turn[name], st.s1, d, 8.0)
            want = st.sigma * cf[side]
            err = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-300))
            rec["table"][name] = {"side": side, "rel_err": err, "pass": bool(err <= 1e-6)}
        rec["pass"] = bool(abs(steer) <= 0.5
                           and all(ch["pass"] for v in rec["checks"].values() for ch in v)
                           and all(t["pass"] for t in rec["table"].values()))
        ok &= rec["pass"]
        out["cells"][str(cid)] = rec
    sides = {c["sigma"]: c["table"]["w_in"]["side"] for c in out["cells"].values()}
    out["recoding_table"] = {str(k): v for k, v in sides.items()}
    out["pass"] = bool(ok and sides.get(1) == "w_left" and sides.get(-1, "w_right") == "w_right")
    return out


# ------------------------------------------------------------------------- RF-C2


def q_net(stim_id: int, variant: str = "Q", c_v: float = Q_CV, ad: bool = False) -> N.NetPolicy:
    """Step 1's Q (or another Q variant) for one stimulus: its QParams with one target row
    (``controls.q_params_for_kappa``), so the loop's aux = 0 picks it; ``ad`` (Q only) steers
    with ``pursuit_steer_ad``."""
    bank, _ = K.design()
    car = QC.car_tables(M.model_tables())
    lm = bank.stimuli[int(stim_id)].landmarks
    qp = QC.q_params_for_kappa(np.asarray(bank.kappa)[int(stim_id)][None], np.array([lm.s_c0]),
                               car, c_v, variant)
    if ad and variant != "Q":
        raise ValueError("the AD-safe steering is wired for Q only (readers use it directly)")
    fn = q_ad_policy if ad else QC.make_q_policy(variant)
    return N.callable_net(fn, qp, None, f"{variant}-ad" if ad else variant)


def understeer_ad(v, kappa, car: QC.CarTables):
    """sign(kappa) us(v, u), u = v^2 |kappa| / a_y,max(v): the understeer term of
    ``pursuit_steer`` (interp/controls.py:566-567) with the same values and the derivative
    at kappa = 0 that JAX's sign drops. us(v, 0) = 0, so the term is continuous and odd in
    kappa with slope us_u'(v, 0+) v^2 / a_y there (us is linear on its first u cell)."""
    ay = jnp.maximum(QC._uinterp(v, car.v, car.a_y), QC._AY_FLOOR)
    u = v * v * jnp.abs(kappa) / ay
    f = jnp.sign(kappa) * QC._bilinear(v, u, car.v, car.us_u, car.us_dd)
    du = car.us_u[1] - car.us_u[0]
    slope = QC._bilinear(v, du, car.v, car.us_u, car.us_dd) / du
    return jnp.where(kappa == 0.0, slope * v * v * kappa / ay, f)


def pursuit_steer_ad(state: racing.EnvState, tr, lookahead_min, lookahead_time,
                     car: QC.CarTables, p=F1_2025):
    """``controls.pursuit_steer`` (interp/controls.py:551-573) with ``understeer_ad``: the
    same values, and AD derivatives that keep the understeer map's slope on a straight
    (where the pursuit curvature is exactly 0 and the frozen form differentiates to the
    kinematic term alone)."""
    x = state.x
    vx, vy, r = x[3], x[4], x[5]
    v = jnp.maximum(vx, 0.0)
    X, Y, psi = global_pose(x, tr)
    c, sn = jnp.cos(psi), jnp.sin(psi)
    st = x[0] + jnp.maximum(lookahead_min, lookahead_time * v)
    dx, dy = lookup(tr.x, st, tr) - X, lookup(tr.y, st, tr) - Y
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    kappa = 2.0 * by / jnp.maximum(bx * bx + by * by, 1e-6)
    delta = jnp.arctan((p.lf + p.lr) * kappa) + understeer_ad(v, kappa, car)
    beta_f = jnp.arctan2(vy + p.lf * r, jnp.maximum(vx, 1.0))
    limited = jnp.clip(delta, beta_f - car.alpha_peak, beta_f + car.alpha_peak)
    delta = jnp.where(vx > 4.0, limited, delta)
    return jnp.clip(delta / p.delta_max, -1.0, 1.0)


def q_ad_policy(params: QC.QParams, o, state: racing.EnvState, tr, aux):
    """Step 1's Q with its steering from ``pursuit_steer_ad`` (the pedal is Q's own)."""
    mean, log_std = QC.make_q_policy("Q")(params, o, state, tr, aux)
    steer = pursuit_steer_ad(state, tr, params.lookahead_min, params.lookahead_time, params.car)
    return mean.at[0].set(steer.astype(mean.dtype)), log_std


def pursuit_gain(net: N.NetPolicy, rd: N.Road, st: N.Start, steer_fn=QC.pursuit_steer,
                 h: float | None = None) -> float:
    """d steer / d y_target of a pursuit law at the start, with the centreline knots beyond
    the car's interpolation bracket moved across the car's heading (d y_target = 1,
    d x_target = 0 exactly): by AD (``h`` None) or by a central difference of step h (m)."""
    tr = jax.tree.map(jnp.asarray, rd.base)
    state = N.env_state(jnp.asarray(st.x0), jnp.asarray(st.prev_action), jnp.asarray(st.accel),
                        jnp.asarray(N.ctx64()))
    prm = net.params
    psi = global_pose(state.x, tr)[2]
    j = int(np.floor(float(st.x0[0]) / float(np.asarray(rd.base.ds))))
    mask = (jnp.arange(len(tr.x)) >= j + 2).astype(jnp.float64)

    def steer(eps):
        t2 = tr._replace(x=tr.x - eps * jnp.sin(psi) * mask, y=tr.y + eps * jnp.cos(psi) * mask)
        return steer_fn(state, t2, prm.lookahead_min, prm.lookahead_time, prm.car)

    if h is None:
        return float(jax.grad(steer)(jnp.float64(0.0)))
    return float((steer(jnp.float64(h)) - steer(jnp.float64(-h))) / (2.0 * h))


def pursuit_kernel(d: np.ndarray, l: float, sigma_b: float, n_u: int = 4001) -> np.ndarray:
    """K(d) = int_0^l (l - u) G(u - d) du (trapezoid on a fine u grid)."""
    u = np.linspace(0.0, float(l), int(n_u))
    g = PT.gaussian_bump(u[None, :], np.asarray(d, float)[:, None], sigma_b)
    return np.trapezoid((float(l) - u)[None, :] * g, u, axis=1)


RF_C2_FD_H = 1e-4          # m, the central difference of the pursuit gain


def _rf_c2_one(net: N.NetPolicy, rd: N.Road, st: N.Start, gain_fd: float, l: float,
               d: np.ndarray, sigma_bs) -> dict:
    steer, imp = steer_field(net, rd, st)
    checks = []
    for sb in sigma_bs:
        F = MS.bump_field(imp.kappa, st.s1, d, sb)
        Kd = pursuit_kernel(d, l, sb)
        checks.append({"sigma_b": float(sb), "cosine": MS.cosine(F, Kd),
                       "amplitude": MS.gain(F, gain_fd * Kd)})
    ps = path_split(net, rd, st, imp)
    width = float(max(np.abs(imp.w_left).max(), np.abs(imp.w_right).max()))
    shape_ok = (all(ch["cosine"] >= COS_MIN for ch in checks) and ps["max_abs"]["kappa_c"] == 0.0
                and width == 0.0 and ps["sum_rel_err"] <= PATH_TOL)
    amp_ok = all(abs(ch["amplitude"] - 1.0) <= AMP_TOL for ch in checks)
    return {"policy": net.name, "steer": steer, "checks": checks,
            "kappa_c_path_max": ps["max_abs"]["kappa_c"], "pose_path_max": ps["max_abs"]["pose"],
            "path_sum_rel_err": ps["sum_rel_err"], "width_field_max": width,
            "shape_ok": bool(shape_ok), "amplitude_ok": bool(amp_ok)}


def rf_c2(cell_id: int = RF_C2_CELLS[0], sigma_bs=(4.0, 8.0, 16.0)) -> dict:
    """RF-C2 at a cell's phase-0 start (module docstring), for step 1's Q as frozen and for
    Q steering with ``pursuit_steer_ad``. The known gain is the central difference of the
    frozen ``pursuit_steer`` (kinematic plus understeer map). Pass (the design's criteria):
    the frozen Q's shape (cosine >= 0.999, kappa_c path 0, no width field, paths summing),
    and the AD-safe Q's shape and amplitude (within 1% of the known gain). The frozen Q's
    amplitude is reported: on a straight its pursuit curvature is exactly 0, where JAX's
    derivative of sign(kappa) us(|kappa|) drops the understeer slope."""
    st, rd = start(cell_id)
    frozen, safe = q_net(st.stim_id), q_net(st.stim_id, ad=True)
    gain_fd = pursuit_gain(frozen, rd, st, h=RF_C2_FD_H)
    gains = {"fd": gain_fd, "ad_frozen": pursuit_gain(frozen, rd, st),
             "ad_safe": pursuit_gain(safe, rd, st, pursuit_steer_ad)}
    v = max(float(st.x0[3]), 0.0)
    l = max(QC.LOOKAHEAD_MIN, QC.LOOKAHEAD_TIME * v)
    d = MS.d_grid_to(st.s1 + l + 5 * max(sigma_bs), st.s1)
    res = {"frozen": _rf_c2_one(frozen, rd, st, gain_fd, l, d, sigma_bs),
           "ad_safe": _rf_c2_one(safe, rd, st, gain_fd, l, d, sigma_bs)}
    # the plan-level path split (network.path_fields) on the frozen Q: its decision steer
    # reads the road through the pose alone, so the observation's kappa path gives 0
    r = N.run(frozen, rd, st)
    pf = N.path_fields(frozen, rd, r, parts=("obs_kappa",), allow_exhausted=True)
    k = N.OUTCOMES.index("steer")
    plan_kc = float(max(np.abs(np.asarray(a)[k]).max() for a in pf["parts"]["obs_kappa"]))
    ok = (res["frozen"]["shape_ok"] and res["ad_safe"]["shape_ok"]
          and res["ad_safe"]["amplitude_ok"] and plan_kc == 0.0)
    return {"cell": int(cell_id), "lookahead": l, "gains": gains,
            "plan_obs_kappa_steer_max": plan_kc,
            "understeer_share": 1.0 - gains["ad_frozen"] / gain_fd, **res, "pass": bool(ok)}


# ------------------------------------------------------------------------- RF-C3


def g1_reference():
    """G1's final snapshot (only its config and obs_rms are read: RF-C3 and Gate 2)."""
    return N._final_agent("G1")


def rf_c3(cell_id: int = RF_C3_CELLS[0], seed: int = 0, agent=None) -> dict:
    """RF-C3: a random-init actor (module docstring)."""
    st, rd = start(cell_id)
    g1 = g1_reference() if agent is None else agent
    net = N.random_net(g1.cfg, g1.obs_rms, seed)
    full = N.run(net, rd, st)
    steer, imp = steer_field(net, rd, st)
    dirs = N.gate2_directions(rd, st)
    g2 = N.gate2(net, rd, st, dirs, steps=1, allow_exhausted=True)
    k = N.OUTCOMES.index("steer")
    norm = float(np.linalg.norm(np.concatenate([np.asarray(a) for a in imp])))
    plan_invalid = [o for o in N.OUTCOMES[:3] if not full.plans[o].valid]
    ok = (np.isfinite(norm) and norm > 0 and g2.ok() and len(plan_invalid) == 3)
    return {"cell": int(cell_id), "seed": int(seed), "policy": net.name,
            "term_step": full.term_step, "n_steps": full.n_steps,
            "plan_outcomes_invalid": plan_invalid,
            "reasons": {o: full.plans[o].reason for o in N.OUTCOMES[:3]},
            "steer": steer, "steer_field_norm": norm,
            "gate2_steer_rel_err": float(g2.zero_rule()[k]),
            "gate2_share_counted": float(g2.share_counted[k]), "pass": bool(ok)}


# ------------------------------------------------------------------------- RF-C5


class KeeperParams(NamedTuple):
    """The planted edge tracker's parameters (``edge_keeper``)."""

    sigma: Any                # +1 left corner, -1 right
    reads_width: Any          # 1: n_t from the road's inside width; 0: from W0 (fixed n)
    W0: Any                   # m, the inside width the fixed keeper assumes
    B0: Any                   # m, clearance to the inside termination boundary at the apex
    A: Any                    # m, extra clearance far from the apex
    s_apex: Any
    ell: Any                  # m, the clearance dip's length scale
    k_n: Any                  # rad / m
    k_xi: Any                 # rad / rad
    a_lat: Any                # m/s^2, the speed law's cornering acceleration
    a_brake: Any              # m/s^2
    v_cap: Any                # m/s


_KEEPER_D = np.arange(0.0, 240.0, 4.0)     # m, the speed law's look-ahead (rollout.py:143)


def edge_keeper(params: KeeperParams, o, state: racing.EnvState, tr, aux):
    """RF-C5's planted line keeper (Policy signature; module docstring)."""
    x = state.x
    s, n, xi, vx = x[0], x[1], x[2], x[3]

    def target(s_):
        wl, wr = widths_at(s_, tr)
        w = params.reads_width * jnp.where(params.sigma > 0, wl, wr) \
            + (1.0 - params.reads_width) * params.W0
        B = params.B0 + params.A * (1.0 - jnp.exp(-0.5 * ((s_ - params.s_apex) / params.ell) ** 2))
        return params.sigma * (w - N.ENV.edge_margin - B)

    def slope(s_):
        return jax.jvp(target, (s_,), (jnp.ones_like(s_),))[1]

    nt = target(s)
    dnt, ddnt = jax.jvp(slope, (s,), (jnp.ones_like(s),))
    kap = kappa_at(s, tr)
    one = 1.0 - nt * kap
    L = F1_2025.lf + F1_2025.lr
    delta = (jnp.arctan(L * (kap / one + ddnt)) + params.k_xi * (jnp.arctan(dnt / one) - xi)
             + params.k_n * (nt - n))
    k = jnp.abs(kappa_at(s + jnp.asarray(_KEEPER_D, s.dtype), tr))
    v_corner = jnp.sqrt(params.a_lat / jnp.maximum(k, 1e-4))
    v_target = jnp.minimum(params.v_cap, jnp.min(jnp.sqrt(v_corner ** 2 + 2.0 * params.a_brake
                                                           * jnp.asarray(_KEEPER_D, s.dtype))))
    pedal = jnp.clip(0.4 * (v_target - vx), -1.0, 1.0)
    mean = jnp.stack([delta / F1_2025.delta_max, pedal]).astype(o.dtype)
    return mean, jnp.full((2,), -jnp.inf, o.dtype)


def keeper_params(st: N.Start, rd: N.Road, reads_width: bool, **kw) -> KeeperParams:
    """The keeper for a start: W0 the base road's inside width at the apex."""
    cfg = dict(KEEPER, **kw)
    lm = st.landmarks
    wl, wr = (float(np.asarray(a)) for a in widths_at(jnp.asarray(lm.s_apex),
                                                       jax.tree.map(jnp.asarray, rd.base)))
    W0 = wl if st.sigma > 0 else wr
    return KeeperParams(sigma=float(st.sigma), reads_width=float(bool(reads_width)), W0=W0,
                        B0=cfg["B0"], A=W0 - N.ENV.edge_margin - cfg["B0"], s_apex=lm.s_apex,
                        ell=max(20.0, 0.5 * (lm.s_c1 - lm.s_c0)), k_n=cfg["k_n"],
                        k_xi=cfg["k_xi"], a_lat=cfg["a_lat"], a_brake=cfg["a_brake"],
                        v_cap=cfg["v_cap"])


def keeper_net(st: N.Start, rd: N.Road, reads_width: bool, **kw) -> N.NetPolicy:
    prm = keeper_params(st, rd, reads_width, **kw)
    return N.NetPolicy(edge_keeper, jax.tree.map(np.float64, prm), None,
                       "edge-keeper" if reads_width else "fixed-n-keeper")


def edge_gains(net: N.NetPolicy, rd: N.Road, st: N.Start, steps: int = KEEPER_STEPS) -> dict:
    """G, G_out and B of a policy's run from a start (the pipeline's c_min field)."""
    r = N.run(net, rd, st, steps=steps)
    pl = r.plans["c_min"]
    if not pl.valid:
        return {"valid": False, "reason": pl.reason, "G": float("nan")}
    fl = N.fields(net, rd, r, steps)
    imp_c, imp_v = fl.of("c_min"), fl.of("v_apex")
    em = MS.edge_measures(imp_c, imp_v, float(fl.values[0]), st.landmarks, st.sigma,
                          s_states=r.x[:, 0], pick=pl.pick)
    h = float(PT.trapezoid(np.array([pl.s_at]), st.landmarks.s_c0, st.landmarks.s_c1)[0])
    return {"valid": True, "G": em.G, "G_out": em.G_out, "B": em.B, "s_star": pl.s_at,
            "h_at_s_star": h, "e_star": em.e_star, "on_plateau": em.on_plateau,
            "near_tie": bool(pl.near_tie), "n_steps": r.n_steps, "vjp_seconds": fl.seconds}


def rf_c5(cells=RF_C5_CELLS, steps: int = KEEPER_STEPS) -> dict:
    """RF-C5: G of the edge keeper (-> 1) and of the fixed-n keeper (-> 0) at each cell."""
    out: dict[str, Any] = {"params": dict(KEEPER), "cells": {}}
    ok = True
    for cid in cells:
        st, rd = start(cid)
        e = edge_gains(keeper_net(st, rd, True), rd, st, steps)
        f = edge_gains(keeper_net(st, rd, False), rd, st, steps)
        good = (e["valid"] and f["valid"] and abs(e["G"] - 1.0) <= RF_C5_TOL
                and abs(f["G"]) <= RF_C5_TOL and e["on_plateau"] and f["on_plateau"])
        out["cells"][str(cid)] = {"sigma": st.sigma, "edge": e, "fixed": f, "pass": bool(good)}
        ok &= good
    out["pass"] = bool(ok)
    return out


# ------------------------------------------------------------------------- unlawful readers


def grip_profile_ad(kap_abs, ds, car: QC.CarTables):
    """``controls.grip_profile`` (interp/controls.py:489-504) with the same values and
    finite derivatives: sqrt(1 - lat^2) behind a double-where at lat = 1."""
    vlim = QC._v_ss(kap_abs, car)

    def body(v_next, inp):
        k_next, vl = inp
        ay = jnp.maximum(QC._uinterp(v_next, car.v, car.a_y), QC._AY_FLOOR)
        lat = jnp.clip(v_next * v_next * k_next / ay, 0.0, 1.0)
        q = 1.0 - lat * lat
        room = jnp.where(q > 0.0, jnp.sqrt(jnp.where(q > 0.0, q, 1.0)), 0.0)
        a = QC._uinterp(v_next, car.v, car.a_dec) * room
        v = jnp.minimum(vl, jnp.sqrt(v_next * v_next + 2.0 * a * ds))
        return v, v

    _, vb = jax.lax.scan(body, vlim[-1], (kap_abs[1:], vlim[:-1]), reverse=True)
    return jnp.concatenate([vb, vlim[-1:]])


def _law_h(c, v, pedal_state, v_t, a_t, car, k_v, lead, phi):
    """law_a's monotone residual h(c) (interp/controls.py:466-472)."""
    base = k_v * (v_t - v - lead * (1.0 - phi) * QC._a_ss(v, pedal_state, car)) \
        + QC._pedal_for(v, a_t, car)
    return c - base + k_v * lead * phi * QC._a_ss(v, c, car)


def law_a_ad(v, pedal_state, v_t, a_t, p_lo, p_hi, car, k_v, lead, phi):
    """``controls.law_a`` (R22 A) with its value unchanged and the derivative of the fixed
    point it solves: in the interior, dc = -(dh/dtheta) / (dh/dc) at the bisection's root
    (one Newton correction whose value is cancelled); at a clip, the bound's own derivative."""
    sg = jax.lax.stop_gradient
    c_star = sg(QC.law_a(v, pedal_state, v_t, a_t, p_lo, p_hi, car, k_v, lead, phi))

    def h(c):
        return _law_h(c, v, pedal_state, v_t, a_t, car, k_v, lead, phi)

    corr = -h(c_star) / sg(jax.grad(lambda c: _law_h(c, sg(v), sg(pedal_state), sg(v_t), sg(a_t),
                                                     car, k_v, lead, phi))(c_star))
    c_int = c_star + corr - sg(corr)
    return jnp.where(h(p_lo) >= 0.0, p_lo, jnp.where(h(p_hi) <= 0.0, p_hi, c_int))


def pedal_command_ad(state, v_t, a_t, params: QC.QParams, ff_only):
    """``controls.pedal_command`` (interp/controls.py:592-616) with ``law_a_ad``."""
    car = params.car
    x = state.x
    v = jnp.maximum(x[3], 0.0)
    room_f, room_r = QC.axle_room(x, car)
    p_hi = QC._pedal_for(v, QC._uinterp(v, car.v, car.a_acc) * room_r, car)
    p_lo = QC._pedal_for(v, -QC._uinterp(v, car.v, car.a_dec) * jnp.minimum(room_f, room_r), car)
    c = law_a_ad(v, x[7], v_t, a_t, p_lo, p_hi, car, params.k_v, params.lead_time, params.phi)
    return jnp.where(ff_only, jnp.clip(QC._pedal_for(v, a_t, car), p_lo, p_hi), c)


class ReaderParams(NamedTuple):
    q: QC.QParams             # Qp's parameters (kappa_cap: c2's 1/12)
    shift: Any                # m, the preview read this much late (c1)
    kappa_on: Any             # 1/m, c3's corner-start threshold
    tau_c: Any                # s, c3's trigger (0: off)


def preview_reader(params: ReaderParams, o, state: racing.EnvState, tr, aux):
    """The Q family on the observation's preview (module docstring): Q's pursuit steering,
    Qp's target from the 40 preview kappa samples (``preview_profile``), read ``shift``
    metres late, capped at ``kappa_cap``, and, with tau_c > 0, c3's time-to-corner law with
    the corner start where |kappa| first passes ``kappa_on`` on Qp's 2 m grid."""
    q = params.q
    x = state.x
    v = jnp.maximum(x[3], 0.0)
    steer = pursuit_steer_ad(state, tr, q.lookahead_min, q.lookahead_time, q.car)
    d_k = racing.preview_offsets(v)
    kap_k = o[4:OBS_PREVIEW:5] / 50.0
    d_j = jnp.arange(QC.QP_N, dtype=v.dtype) * QC.QP_DS
    kap = jnp.minimum(jnp.abs(jnp.interp(d_j - params.shift, d_k, kap_k)), q.kappa_cap)
    T = jnp.minimum(q.car.v_top, q.c_v * grip_profile_ad(kap, QC.QP_DS, q.car))
    ahead = q.lead_time * v
    v_t, a_t = QC._at(T, ahead, QC.QP_DS)
    on = kap >= params.kappa_on
    j = jnp.argmax(on)
    jm = jnp.maximum(j - 1, 0)
    den = jnp.where(j > 0, kap[j] - kap[jm], 1.0)
    frac = jnp.where(j > 0, (params.kappa_on - kap[jm]) / den, 0.0)
    d_c = jnp.where(on.any(), d_j[jm] + frac * (d_j[j] - d_j[jm]), 1e9)
    v3, a3, ff3, _ = QC._c3_target(jnp.zeros_like(v), v, ahead, T, d_c, params.tau_c, v_t, a_t,
                                   QC.QP_DS, q.car)
    use3 = params.tau_c > 0.0
    pedal = pedal_command_ad(state, jnp.where(use3, v3, v_t), jnp.where(use3, a3, a_t), q,
                             use3 & ff3)
    return jnp.stack([steer, pedal]).astype(o.dtype), jnp.full((2,), -jnp.inf, o.dtype)


def reader_net(variant: str, stim_id: int) -> N.NetPolicy:
    """One of ``READERS`` as a NetPolicy for one stimulus (float32 parameters, upcast)."""
    if variant not in READERS:
        raise ValueError(f"unknown reader {variant!r}; expected one of {READERS}")
    q = q_net(stim_id, "Qp").params
    if variant == "c2":
        q = q._replace(kappa_cap=np.float64(np.float32(QC.C2_KAPPA_CAP)))
    prm = ReaderParams(q=q, shift=np.float64(C1_SHIFT if variant == "c1" else 0.0),
                       kappa_on=np.float64(KAPPA_ON),
                       tau_c=np.float64(np.float32(TAU_C)) if variant == "c3" else np.float64(0.0))
    return N.NetPolicy(preview_reader, prm, None, f"reader-{variant}")


def optimum_fields(cell_id: int, phase: int = 0):
    """The optimum's fields at the A2 key of (cell, phase) (``kkt.key_fields`` on the
    re-polished key); None when the cell has no A2 key at that phase. Raises when the key
    did not re-polish or its adjoint residuals exceed ``kkt.ADJOINT_RTOL``."""
    from apexzero.fields import kkt as KK

    refs = [r for r in K.a2_refs() if r.cell_id == int(cell_id) and r.phase == int(phase)]
    refs = [r for r in refs if K.strata(r).w_is_cell]
    if not refs:
        return None
    bank, _ = K.design()
    rp = K.repolish(K.rebuild(K.load_key(refs[0]), bank))
    if not rp.ok:
        raise ValueError(f"key {refs[0].key_id}: re-polish {rp.status}, block_ok {rp.block_ok}")
    fl = KK.key_fields(rp.problem, bank=bank)
    if not fl.adjoint_ok:
        raise ValueError(f"key {refs[0].key_id}: adjoint residual {fl.max_adjoint_residual}")
    return fl


def judge_reader(variant: str, row: dict, qp: dict, capped: bool) -> bool | None:
    """Whether a reader's planted violation is detected against Qp's own run at the same cell
    (``READER_VIOLATES``); None when it is not judged there (an outcome missing on either, or
    c2 off a capped corner, where it is Qp). Pure."""
    viol = READER_VIOLATES[variant]
    if viol is None:
        return None
    outcome, how = viol
    if variant == "c2" and not capped:
        return None
    if how == "offset":
        if "values" not in row or "values" not in qp:
            return None
        a, b = row["values"].get(outcome), qp["values"].get(outcome)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return None
        return bool(a - b >= C1_OFFSET_MIN)
    if outcome not in row or outcome not in qp:
        return None
    ga, gb = row[outcome]["g"], qp[outcome]["g"]
    if not (np.isfinite(ga) and np.isfinite(gb)) or gb == 0:
        return None
    ratio = ga / gb
    return bool(not (READER_RATIO_BAND[0] <= ratio <= READER_RATIO_BAND[1]))


def readers(cells=READER_CELLS, sigma_b: float = 8.0, steps: int = N.STEPS) -> dict:
    """The unlawful readers (module docstring) judged for specificity against Qp, their base
    law (``judge_reader``): each violator must be detected on every cell where it is judged
    (and judged on at least one, c1 on at least two), and the negative checks must not be
    flagged (Qp judged as each violator against itself; c2 on corners above its cap, where it
    is Qp). H1-typical's absolute criteria against the optimum (gain, location, and the shape
    specificity against the optimum's fields at far geometries, per outcome) are reported
    for every reader, Qp included; Qp is not lawful on these keys (``qp_lawful``), so those
    flags are descriptive."""
    out: dict[str, Any] = {"cells": {}, "sigma_b": float(sigma_b), "steps": int(steps)}
    opt_fields = {}
    for cid in cells:
        st, _ = start(cid)
        flo = optimum_fields(cid)
        if flo is None:
            raise ValueError(f"cell {cid} has no A2 key at phase 0")
        opt_fields[cid] = (st, flo)
    bank, _ = K.design()
    specs = {cid: bank.stimuli[st.stim_id].spec for cid, (st, _) in opt_fields.items()}
    geo = {cid: (float(sp.R), float(sp.theta_deg)) for cid, sp in specs.items()}

    def opt_map(o):
        return {cid: (geo[cid], st.landmarks.s_c0, st.sigma,
                      flo.impulse[o] if flo.valid(o) else None)
                for cid, (st, flo) in opt_fields.items()}

    out["null"] = {o: MS.cross_geometry_null(opt_map(o), o, sigma_b) for o in ("s_b", "v_apex")}
    runs: dict[tuple[int, str], Any] = {}
    for cid, (st, flo) in opt_fields.items():
        _, rd = start(cid)
        rec: dict[str, Any] = {}
        for variant in READERS:
            net = reader_net(variant, st.stim_id)
            r = N.run(net, rd, st, steps=steps)
            row: dict[str, Any] = {"n_steps": r.n_steps, "term_step": r.term_step,
                                   "valid": r.valid_outcomes().tolist(),
                                   "values": {o: float(r.values()[k]) for k, o in
                                              enumerate(N.OUTCOMES[:3])}}
            if r.valid_outcomes()[:3].all():
                fl = N.fields(net, rd, r, steps)
                runs[cid, variant] = fl
                d = MS.d_grid_to(MS.support_end(r.x[: r.n_steps + 1]), st.s1)
                for o in ("s_b", "v_apex"):
                    if not flo.valid(o):
                        continue
                    Fn = MS.bump_field(fl.of(o).kappa, st.s1, d, sigma_b)
                    Fo = MS.bump_field(np.asarray(flo.impulse[o].kappa), st.s1, d, sigma_b)
                    cmp = MS.compare(Fn, Fo, d)
                    cmp["finite"] = bool(np.isfinite(Fn).all())
                    row[o] = cmp
            rec[variant] = row
        out["cells"][str(cid)] = rec
    # shape specificity per outcome (matched against far-geometry optimum fields)
    for o in ("s_b", "v_apex"):
        for variant in READERS:
            net_map = {cid: (geo[cid], st.landmarks.s_c0, st.sigma, runs[cid, variant].of(o))
                       for cid, (st, _) in opt_fields.items() if (cid, variant) in runs}
            for sp in MS.shape_specificity(net_map, opt_map(o), o, sigma_b):
                row = out["cells"][str(sp["label"])][variant]
                if o in row:
                    row[o]["rho_far"] = sp["rho_far"]
                    row[o]["delta_rho"] = sp["delta_rho"]
                    row[o]["h1"] = MS.h1_criteria(row[o], sp["rho_far"])
    judged: dict[str, list[bool]] = {v: [] for v in READERS if READER_VIOLATES[v]}
    negative: dict[str, list[bool]] = {v: [] for v in judged}
    for cid, rec in out["cells"].items():
        capped = geo[int(cid)][0] < 1.0 / QC.C2_KAPPA_CAP
        for variant, verdicts in judged.items():
            for o in ("s_b", "v_apex"):          # specificity numbers beside the verdict
                if o in rec[variant] and o in rec["Qp"]:
                    a, b = rec[variant][o], rec["Qp"][o]
                    a["vs_Qp"] = {"g_ratio": a["g"] / b["g"] if b["g"] else np.nan,
                                  "location_shift": a["location"] - b["location"]}
            if all(rec[variant]["valid"][:3]) and all(rec["Qp"]["valid"][:3]):
                v = judge_reader(variant, rec[variant], rec["Qp"], capped)
                if v is not None:
                    rec[variant]["violation_detected"] = v
                    verdicts.append(v)
                # Qp relabelled as this violator must not be detected
                nq = judge_reader(variant, rec["Qp"], rec["Qp"], capped)
                if nq is not None:
                    negative[variant].append(nq)
            if variant == "c2" and not capped and all(rec["c2"]["valid"][:3]) \
                    and all(rec["Qp"]["valid"][:3]):
                # c2 above its cap is Qp: judged as if capped, it must not be detected
                nc = judge_reader("c2", rec["c2"], rec["Qp"], True)
                if nc is not None:
                    negative["c2"].append(nc)
                    rec["c2"]["uncapped_detected"] = nc
    out["judged"] = judged
    out["negative"] = negative
    qp_h1 = [rec["Qp"][o]["h1"] for rec in out["cells"].values() for o in ("s_b", "v_apex")
             if o in rec["Qp"] and "h1" in rec["Qp"][o]]
    out["qp_lawful"] = bool(qp_h1 and all(all(h.values()) for h in qp_h1))
    need = {v: (2 if v == "c1" else 1) for v in judged}
    out["pass"] = bool(all(len(j) >= need[v] and all(j) for v, j in judged.items())
                       and all(not any(n) for n in negative.values())
                       and all(len(n) for n in negative.values()))
    return out


# ------------------------------------------------------------------------- all controls


CONTROLS = ("RF-C1", "RF-C2", "RF-C3", "RF-C4", "RF-C5", "readers")
CELLS = {"RF-C1": RF_C1_CELLS, "RF-C2": RF_C2_CELLS, "RF-C3": RF_C3_CELLS, "RF-C4": RF_C4_CELLS,
         "RF-C5": RF_C5_CELLS, "readers": READER_CELLS}
QUICK_CELLS = {"RF-C2": (344,), "RF-C5": (344, 340), "readers": (46, 94, 134, 343)}   # tests


def run_all(which=CONTROLS, cells: dict | None = None) -> dict:
    """Every control in ``which`` (a float64 process) on its cells (``CELLS``, or the given
    override per control), with its pass flag and seconds."""
    N.require_x64()
    cs = {**CELLS, **(cells or {})}
    def rf_c1_all():
        res = {str(c): rf_c1(c) for c in cs["RF-C1"]}
        c0 = str(cs["RF-C1"][0])
        res[c0]["obs_rms"] = rf_c1(int(c0), obs_rms=g1_reference().obs_rms)
        res[c0]["pass"] = bool(res[c0]["pass"] and res[c0]["obs_rms"]["pass"])
        return res

    fns = {"RF-C1": rf_c1_all,
           "RF-C2": lambda: {str(c): rf_c2(c) for c in cs["RF-C2"]},
           "RF-C3": lambda: {str(c): rf_c3(c) for c in cs["RF-C3"]},
           "RF-C4": lambda: rf_c4(cs["RF-C4"]), "RF-C5": lambda: rf_c5(cs["RF-C5"]),
           "readers": lambda: readers(cs["readers"])}
    out = {}
    for name in which:
        t0 = time.perf_counter()
        res = fns[name]()
        if name in ("RF-C1", "RF-C2", "RF-C3"):
            res = {"cells": res, "pass": all(v["pass"] for v in res.values())}
        res["seconds"] = time.perf_counter() - t0
        out[name] = res
    out["pass"] = {k: bool(v["pass"]) for k, v in out.items() if isinstance(v, dict)}
    return out


def jsonable(o):
    """A JSON-ready copy: NumPy scalars and arrays as Python values and lists, NaN and inf
    as strings, ``Samples`` as dicts of lists, tuples as lists."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, Samples):
        return {k: jsonable(np.asarray(v)) for k, v in o._asdict().items()}
    if isinstance(o, list | tuple):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return repr(o)
    return o


def main(argv: list[str] | None = None) -> int:
    """``python -m apexzero.fields.controls --out FILE [--which RF-C1 ...] [--quick]``: the
    controls in a float64 process (x64 switched on here), written as JSON; ``--quick`` runs
    the multi-cell controls on ``QUICK_CELLS``."""
    ap = argparse.ArgumentParser(prog="python -m apexzero.fields.controls")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--which", nargs="+", default=list(CONTROLS), choices=CONTROLS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    res = run_all(tuple(args.which), QUICK_CELLS if args.quick else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
