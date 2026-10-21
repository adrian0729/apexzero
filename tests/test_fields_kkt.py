"""Step 2's optimum fields (fields.kkt): the barrier sensitivity of a TO-OCP optimum against
warm re-solves of the frozen solver (design Section 5 'Optimum', Gates 1 and 4).

- A synthetic window solves a left corner on a short strip with the uncompiled solver
  (T = 76: a braking zone, an inside contact, and 50 m past the corner so that c_min, v_apex
  and s_b are all defined). There:
  - the barrier point is consistent;
  - F_p equals central differences of the KKT residual in P;
  - the inertia equals a dense eigenvalue count;
  - every impulse response (three features, four outcomes, and c_min's line part) matches
    Richardson finite differences of warm re-solves to <= 1e-3;
  - the planted bugs fail by orders of magnitude;
  - G* = 1.
- On step 1's A2 keys at W = 400, the inertia is right, G* lies in [0.98, 1.02], and a reduced
  Gate 1 passes.

float64 JAX runs inside ``jax.enable_x64`` blocks in fields.kkt; the module restores the global
flag. No agent network is evaluated."""

import dataclasses
import itertools
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from apexzero.fields import keys as K
from apexzero.fields import kkt as KK
from apexzero.fields import perturb as PT
from apexzero.interp import answer_key as ak
from apexzero.interp import io
from apexzero.interp.stimuli import Landmarks
from apexzero.ocp.window import NA, NX, WindowSpec
from apexzero.tracks.spline import packed_coeffs
from apexzero.tracks.track import Track
from apexzero.vehicle.params import F1_2025

HAVE_KEYS = (io.KEYS_DIR / "nominal").exists()
SIGMA = 1                      # the synthetic corner turns left
T_SYN = 76
SOLVE_OPTS = {"mu_strategy": "monotone", "mu_init": 1e-2, "bound_push": 1e-6}
TOL = 1e-3                     # the task's bound on KKT vs Richardson FD


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


def corner_strip(n: int = 200, ds: float = 2.0, s_c0: float = 100.0, ramp: float = 6.0,
                 arc: float = 18.0, R: float = 15.0) -> tuple[Track, Landmarks]:
    """A left corner (trapezoid curvature to 1/R) on a 400 m strip, 9 m wide with widths that
    vary along s, and its landmarks."""
    s = np.arange(n) * ds
    L = 2 * ramp + arc
    u = s - s_c0
    kappa = np.clip(np.minimum(u / ramp, (L - u) / ramp), 0.0, 1.0) / R
    wl = 4.5 + 0.3 * np.sin(s / 23.0)
    wr = 4.5 + 0.3 * np.cos(s / 19.0)
    th = np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(th[1:]) + np.cos(th[:-1])) * ds)])
    y = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(th[1:]) + np.sin(th[:-1])) * ds)])
    lm = Landmarks(s_c0, s_c0 + ramp, s_c0 + L / 2, s_c0 + ramp + arc, s_c0 + L)
    tr = Track(name="synthetic-corner", s=s, x=x, y=y, theta=th, kappa=kappa, w_left=wl,
               w_right=wr, closed=False)
    return tr, lm


def kinematic_guess(tr: Track, lm: Landmarks, x0: np.ndarray, T: int, v_c: float = 22.0,
                    a_b: float = 25.0, a_a: float = 8.0, dt: float = 0.05):
    """Centreline states along a brake-to-v_c speed profile (not dynamically consistent)."""
    p = F1_2025
    xs, acts = np.zeros((T + 1, NX)), np.zeros((T, NA))
    xs[0] = x0
    s, v = float(x0[0]), float(x0[3])
    for t in range(T):
        v_new = v + a_a * dt
        if s < lm.s_apex:
            v_new = min(v_new, max(np.sqrt(v_c ** 2 + 2 * a_b * max(lm.s_c0 - s, 0.0)),
                                   v - a_b * dt))
        pedal = 0.5 if v_new >= v else -0.6
        s, v = s + 0.5 * (v + v_new) * dt, v_new
        kap = np.interp(s, tr.s, tr.kappa)
        delta = np.arctan((p.lf + p.lr) * kap)
        xs[t + 1] = [s, 0.0, 0.0, v, 0.0, v * kap, delta, pedal]
        acts[t] = [np.clip(delta / p.delta_max, -1.0, 1.0), pedal]
    return xs, acts


@pytest.fixture(scope="module")
def synth():
    tr, lm = corner_strip()
    spec = WindowSpec(steps=T_SYN, radius=30.0)
    p = F1_2025
    ctx = ak.key_ctx(p)
    x0 = np.array([lm.s_c0 - 45.0, 0.0, 0.0, 42.0, 0.0, 0.0, 0.0, 0.3])
    # a monotone barrier finds the optimum in ~190 iterations from this guess; the problem
    # itself (re-polish, re-solves) uses the keys' default options, so mu ends at ~1e-11
    fast = ak.AnchoredSolver(spec, compile=False, ipopt=SOLVE_OPTS)
    res = fast.solve(tr, x0, init=kinematic_guess(tr, lm, x0, T_SYN), ctx=ctx, p=p)
    assert res.success, res.status
    solver = ak.AnchoredSolver(spec, compile=False)
    # one tight warm re-solve with the default (adaptive) barrier: mu ends at ~1e-11 as on
    # step 1's keys (the refine inside ``fast.solve`` stops at the monotone barrier's ~1e-8)
    rs = KK.resolve(KK.problem(solver, tr, res, ctx, p, x0), None)
    assert rs.ok, rs.status
    T = spec.steps
    res = dataclasses.replace(res, x=rs.w[: NX * (T + 1)].reshape(T + 1, NX),
                              a=rs.w[NX * (T + 1):].reshape(T, NA),
                              meta=dict(res.meta, lam=(rs.lam_g, rs.lam_x)))
    prob = K.repolish(KK.problem(solver, tr, res, ctx, p, x0, key_id="synthetic")).problem
    # option A's anchored structure on the same optimum: a_0 fixed to its optimal value
    anch = K.repolish(KK.problem(solver, tr, res, ctx, p, x0, kind="anchored",
                                 key_id="synthetic-anchored")).problem
    return SimpleNamespace(tr=tr, lm=lm, prob=prob, anch=anch)


@pytest.fixture(scope="module")
def synth_fields(synth):
    return KK.fields(synth.prob, synth.lm, SIGMA, 0)


def _contacts(prob: K.KeyProblem, side: int) -> np.ndarray:
    """s of the states whose corridor row (0: left / inside here, 1: right) is active."""
    T = prob.T
    lam = prob.lam_g[NX * T: 10 * T].reshape(T, 2)[:, side]
    return prob.x[1:, 0][lam > 1e-4]


def _directions(prob: K.KeyProblem, lm: Landmarks) -> dict[str, PT.Samples]:
    """Unit-peak bumps (sigma_b = 8 m) of each feature at the braking zone, the inside and
    outside contacts and the apex, plus the inside and outside trapezoid shifts."""
    n = prob.track.n_points
    s = PT.sample_s(n)
    inside, outside = _contacts(prob, 0), _contacts(prob, 1)
    assert len(inside) and len(outside), "the synthetic optimum touches both edges"
    centres = {"brake": lm.s_c0 - 10.0, "in": float(inside[0]), "out": float(outside[0]),
               "apex": lm.s_apex}
    out = {}
    for f in PT.FEATURES:
        for nm, c in centres.items():
            z = PT.zeros(n)._asdict()
            z[f] = PT.gaussian_bump(s, c, 8.0)
            out[f"{f}@{nm}"] = PT.Samples(**z)
    for side in ("inside", "outside"):
        out[f"trapezoid_{side}"] = PT.edge_shift(n, lm, SIGMA, side)
    return out


@pytest.fixture(scope="module")
def synth_gate1(synth, synth_fields):
    dirs = _directions(synth.prob, synth.lm)
    return dirs, KK.gate1(synth.prob, synth_fields, dirs, SIGMA)


# ------------------------------------------------------------------------- building blocks


def test_coefficient_map_and_its_transpose():
    rng = np.random.default_rng(0)
    for n in (5, 37, 400):
        f = rng.standard_normal(n)
        c = KK.coeffs(f)
        np.testing.assert_array_equal(c, packed_coeffs(f, False, n))
        cb = rng.standard_normal(n + 2)
        assert abs(c @ cb - f @ KK.coeffs_transpose(cb, n)) <= 1e-12 * np.abs(c).sum() * np.abs(cb).max()
        # linear: the transpose of the dense matrix, column by column
        Cm = np.stack([KK.coeffs(e) for e in np.eye(n)], axis=1)
        np.testing.assert_allclose(KK.coeffs_transpose(cb, n), Cm.T @ cb, rtol=0, atol=1e-12)


def test_problem_from_a_solve_is_a_kkt_point(synth):
    for prob in (synth.prob, synth.anch):
        g0 = K.gate0(prob)
        assert g0.ok and g0.r_free < 1e-9 and g0.defect < 1e-9, g0
    assert synth.prob.free.sum() == synth.prob.solver.nw - NX
    assert synth.anch.free.sum() == synth.anch.solver.nw - NX - NA        # a_0 held too


def test_barrier_point(synth):
    prob = synth.prob
    bar = KK.barrier(prob)
    assert 1e-12 <= bar.mu <= 1e-10 and bar.n_strong > 10 and bar.fallback == 0
    free = prob.free
    ineq = prob.lbg < prob.ubg
    # z_U - z_L = lam wherever lam points to a side that exists
    for lam, zl, zu, sl, su, use in ((prob.lam_x, bar.zl, bar.zu, bar.sl, bar.su, free),
                                     (prob.lam_g, bar.gzl, bar.gzu, bar.gsl, bar.gsu, ineq)):
        m = use & (((lam > 0) & np.isfinite(su)) | ((lam < 0) & np.isfinite(sl)))
        assert m.sum() > 10
        np.testing.assert_allclose((zu - zl)[m], lam[m], rtol=1e-12, atol=0)
    assert np.all(bar.sigma_x[free] >= 0) and np.all(bar.sigma_x[~free] == 0)
    assert np.all(bar.sigma_g[ineq] > 0) and np.all(bar.sigma_g[~ineq] == 0)
    # the strongly active sides sit on the central path: s z = mu
    act = (bar.gzu > 1e-4) & ineq
    np.testing.assert_allclose(bar.gzu[act] * bar.gsu[act], bar.mu, rtol=0.05)


def test_param_blocks_equal_differences_of_the_kkt_residual(synth):
    """F_p delta against central differences in P of [grad f + J'lam; g] at the fixed
    primal-dual point (both width terms, the curvature step derivatives), and the transpose."""
    prob = synth.prob
    blocks = KK.param_blocks(prob)
    n = prob.track.n_points
    s = PT.sample_s(n)

    def residual(delta):
        P = K.key_parameters(prob.solver, prob.track, prob.sigma, prob.ctx, prob.p, delta)
        _, gf = prob.solver.funcs["grad_f"](prob.w, P)
        g, jac = prob.solver.funcs["jac_g"](prob.w, P)
        rw = np.asarray(gf).ravel() + jac.sparse().T @ prob.lam_g
        return rw, np.asarray(g).ravel()

    rng = np.random.default_rng(1)
    for f, h in (("kappa", 1e-6), ("w_left", 1e-4), ("w_right", 1e-4)):
        z = PT.zeros(n)._asdict()
        z[f] = PT.gaussian_bump(s, synth.lm.s_apex, 8.0) + 0.3 * PT.gaussian_bump(s, 60.0, 5.0)
        d = PT.Samples(**z)
        (wp, gp), (wm, gm) = residual(KK.scaled(d, h)), residual(KK.scaled(d, -h))
        rw, rg = blocks.apply(d)
        fw, fg = (wp - wm) / (2 * h), (gp - gm) / (2 * h)
        free = prob.free
        scale = max(np.abs(fw[free]).max(), np.abs(fg).max())
        assert scale > 1e-3, f
        assert np.abs(rw[free] - fw[free]).max() <= 1e-6 * scale, f
        assert np.abs(rg - fg).max() <= 1e-6 * scale, f
        # transpose: <mu, F_p d> = <F_p' mu, d>
        mw, mg = rng.standard_normal(prob.solver.nw), rng.standard_normal(prob.solver.ng)
        tr_ = blocks.transpose(mw, mg)
        lhs = mw @ rw + mg @ rg
        rhs = sum(np.asarray(getattr(tr_, k)) @ np.asarray(getattr(d, k)) for k in PT.FEATURES)
        assert abs(lhs - rhs) <= 1e-10 * max(abs(lhs), 1.0)
    # the width stationarity term is there (the design's -nu b'/ds on s_{t+1}) and is what
    # 'drop_width_mixed' removes
    assert np.abs(blocks.path_w[:, 0, :]).max() > 1e-3
    assert np.abs(KK.param_blocks(prob, "drop_width_mixed").path_w).max() == 0.0


def test_null_space_and_inertia_against_a_dense_count(synth):
    for prob in (synth.prob, synth.anch):
        kkt = KK.kkt_system(prob)
        Z = KK.null_space(kkt)
        Je = kkt.J[: NX * prob.T][:, kkt.free]
        assert np.abs(Je @ Z).max() <= 1e-12 * max(1.0, np.abs(Z).max())
        inr = KK.inertia(kkt)
        assert inr.ok and inr.n_small == 0, inr
        # dense count on the Jacobi-scaled matrix (a congruence: the same inertia)
        Ks = kkt.scale[:, None] * kkt.matrix.toarray() * kkt.scale[None, :]
        ev = np.linalg.eigvalsh(Ks)
        assert (int(np.sum(ev > 0)), int(np.sum(ev < 0))) == (kkt.n_free, kkt.m)
        assert (inr.n_pos, inr.n_neg, inr.n_zero) == (kkt.n_free, kkt.m, 0)
    # the reduced matrix's LDL' count equals eigenvalue signs on an indefinite matrix
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 40))
    A = A + A.T
    pos, neg, zero, _ = KK._ldl_counts(A)
    ev = np.linalg.eigvalsh(A)
    assert (pos, neg, zero) == (int(np.sum(ev > 0)), int(np.sum(ev < 0)), 0)
    # a wrong Lagrangian Hessian (L = f - lam'g) breaks the inertia
    assert not KK.inertia(KK.kkt_system(synth.prob, "flip_lam_g")).ok


# ------------------------------------------------------------------------- Gate 1 (synthetic)


def test_synthetic_fields_are_defined(synth, synth_fields):
    fl = synth_fields
    assert all(fl.valid(k) for k in KK.OUTCOMES), {k: r.reason for k, r in fl.reads.items()}
    assert fl.reads["c_min"].value == pytest.approx(0.03, abs=1e-6)       # the 3 cm margin
    assert fl.reads["s_b"].plan.pick.k >= 3                              # braking starts later
    assert fl.inertia.ok and 1e-12 <= fl.mu <= 1e-10
    assert max(fl.adjoint_residual.values()) <= 1e-10
    assert fl.adjoint_ok and fl.max_adjoint_residual <= KK.ADJOINT_RTOL
    assert fl.barrier_fallback == 0 and fl.inertia.determined
    n = synth.prob.track.n_points
    for k in KK.OUTCOMES:
        assert all(np.asarray(getattr(fl.impulse[k], f)).shape == (n,) for f in PT.FEATURES)
    # only c_min reads the samples directly, and only its inside (left) width
    assert np.abs(fl.explicit["c_min"].w_left).max() > 0.1
    assert np.abs(fl.explicit["c_min"].w_right).max() == 0.0
    assert all(np.abs(np.asarray(v)).max() == 0.0 for k in ("v_apex", "s_b", "steer")
               for v in fl.explicit[k])


def test_kkt_fields_match_richardson_resolves(synth, synth_fields, synth_gate1):
    """Gate 1 on the synthetic window: every (quantity, feature group) field vector within
    1e-3 of the Richardson FD of warm re-solves, and the comparisons are not vacuous."""
    dirs, g1 = synth_gate1
    assert g1.ok_solves, [f.statuses for f in g1.fd]
    bad = {k: e for k, e in g1.err.items() if not e <= TOL}
    assert not bad, bad
    groups = {KK.direction_group(nm) for nm in g1.names}
    assert groups == {"kappa", "w_left", "w_right", "trapezoid_inside", "trapezoid_outside"}
    for q in ("v_apex", "s_b", "steer"):
        for g in ("kappa", "w_left", "w_right"):
            ix = [i for i, nm in enumerate(g1.names) if KK.direction_group(nm) == g]
            assert np.abs(g1.fd_r[q][ix]).max() > 1e-3, (q, g)            # a real field
    # c_min's total is 0 on an active contact (G* = 1); its line part is not
    assert np.abs(g1.kkt["c_min"]).max() < 1e-6
    i_tr = g1.names.index("trapezoid_inside")
    assert g1.kkt["c_line"][i_tr] == pytest.approx(-1.0, abs=1e-6)
    assert g1.fd_r["c_line"][i_tr] == pytest.approx(-1.0, abs=1e-6)
    # the primal tangent itself, against a central difference of the re-solves
    kkt = KK.kkt_system(synth.prob)
    blocks = KK.param_blocks(synth.prob)
    for nm, h in (("kappa@apex", 1e-6), ("w_left@in", 1e-4), ("trapezoid_outside", 1e-4)):
        dw, _, res = KK.forward(kkt, blocks, dirs[nm])
        wp = KK.resolve(synth.prob, KK.scaled(dirs[nm], h)).w
        wm = KK.resolve(synth.prob, KK.scaled(dirs[nm], -h)).w
        fd = (wp - wm) / (2 * h)
        assert res <= 1e-10 and np.linalg.norm(dw - fd) <= TOL * np.linalg.norm(fd), nm


@pytest.mark.parametrize("plant", KK.PLANTS)
def test_planted_bugs_fail_gate1(synth, synth_gate1, plant):
    """Each planted bug moves the field error far above 1e-3, on the groups it touches."""
    dirs, g1 = synth_gate1
    fl = KK.fields(synth.prob, synth.lm, SIGMA, 0, plant=plant, check_inertia=False)
    err = {}
    for q in KK.GATE1_OUTCOMES:
        kk = np.array([fl.along(q, dirs[nm]) for nm in g1.names])
        for g in {KK.direction_group(nm) for nm in g1.names}:
            ix = [i for i, nm in enumerate(g1.names) if KK.direction_group(nm) == g]
            err[q, g] = KK.gate1_error(kk[ix], g1.fd_r[q][ix], g1.disc[q][ix])
    worst = max(err.values())
    assert worst > 100 * TOL, (plant, worst)
    width = max(err[q, g] for q in ("v_apex", "s_b", "steer") for g in ("w_left", "w_right"))
    assert width > 10 * TOL, (plant, width)
    kappa = max(err[q, "kappa"] for q in ("v_apex", "s_b", "steer"))
    if plant == "flip_lam_g":
        assert kappa > 10 * TOL
    else:                       # the width bugs leave the curvature fields alone
        assert kappa <= TOL


def test_gate4_on_the_synthetic_window(synth, synth_fields):
    g = KK.edge_gain(synth_fields, synth.lm, SIGMA)
    assert KK.gate4(g) and abs(g - 1.0) < 1e-6
    swapped = KK.fields(synth.prob, synth.lm, SIGMA, 0, plant="swap_widths", check_inertia=False)
    assert not KK.gate4(KK.edge_gain(swapped, synth.lm, SIGMA))
    # the edge's own move is exactly the trapezoid at the contact
    n = synth.prob.track.n_points
    h = PT.edge_shift(n, synth.lm, SIGMA, "inside")
    assert synth_fields.along("c_min", h) - synth_fields.along("c_line", h) == pytest.approx(1.0, abs=1e-9)


def test_anchored_structure_reads_a1(synth):
    """Option A: x_0 and a_0 held; the decision is a*_1 and its steer field matches FD."""
    prob = synth.anch
    fl = KK.fields(prob, synth.lm, SIGMA, 1)
    assert fl.inertia.ok and fl.inertia.expected[0] == prob.solver.nw - NX - NA
    ia = NX * (prob.T + 1) + NA
    assert np.argmax(np.abs(fl.reads["steer"].grad_w)) == ia
    assert not KK.readouts(prob, synth.lm, SIGMA, 0)["steer"].valid       # a_0 is fixed
    dirs = _directions(prob, synth.lm)
    sub = {k: dirs[k] for k in ("kappa@brake", "w_left@in", "trapezoid_inside")}
    g1 = KK.gate1(prob, fl, sub, SIGMA)
    assert g1.ok_solves and all(e <= TOL for e in g1.err.values()), g1.err
    assert np.abs(g1.fd_r["steer"]).max() > 1e-3


def test_gate1_error_rules():
    assert KK.gate1_error([1.0, 2.0], [1.0, 2.0], [0.0, 0.0]) == 0.0
    assert KK.gate1_error([1.1], [1.0], [0.0]) == pytest.approx(0.1)
    assert KK.gate1_error([1.1], [1.0], [1.0]) == pytest.approx(0.1 / 3)    # the floor
    assert KK.gate1_error([1e-9], [0.0], [0.0]) == 0.0                      # zero field
    assert KK.gate1_error([1e-3], [0.0], [0.0]) == np.inf
    # zero fields are flagged (never pooled); a real one is not
    assert KK.gate1_score([1e-9], [0.0], [0.0]) == (0.0, True)
    assert KK.gate1_score([1.1], [1.0], [0.0])[1] is False
    # c_min against its explicit edge term: FD noise of 1.5e-6 on a cancelling field (above
    # ZERO_ATOL) is ~1e-6 of the edge's move, not an error of order 1
    e, z = KK.gate1_score([7e-12], [1.5e-6], [1e-7], scale=[1.0])
    assert not z and e == pytest.approx(1.5e-6, rel=1e-3)
    e, z = KK.gate1_score([7e-12], [1.5e-6], [1e-7])
    assert not z and e > 0.9                                           # without the scale
    # a wrong cancellation (KKT field 1 where FD is 0) fails against the scale
    assert KK.gate1_score([1.0], [0.0], [0.0], scale=[1.0])[0] == pytest.approx(1.0)


def test_inertia_certification_of_rounding_level_pivots():
    """An LDL' count with a pivot at the rounding level is not 'ok' unless the eigenvalue
    recount certifies it: a duplicated bound row with D_C ~ 1e-17 makes an eigenvalue of
    that size, whose sign cannot be certified; a well-separated indefinite matrix is."""
    d = 1e-17
    M = np.array([[1.0, 0.5, 0.5], [0.5, -d, 0.0], [0.5, 0.0, -d]])
    pos, neg, zero, n_small, det, mn, method = KK.certified_counts(M)
    assert n_small > 0 and not det and method == "ldl+eig" and mn < 1e-15
    inr = KK.Inertia(n_pos=1, n_neg=2, n_zero=0, expected=(1, 2, 0), method=method,
                     reduced_size=3, n_active_rows=2, n_bound_rows=0, r_min_eig=1.0,
                     n_small=n_small, seconds=0.0, determined=det, eig_min_abs=mn)
    assert (inr.n_pos, inr.n_neg, inr.n_zero) == inr.expected and not inr.ok
    rng = np.random.default_rng(5)
    A = rng.standard_normal((30, 30))
    A = A + A.T + np.diag(np.r_[np.full(15, 8.0), np.full(15, -8.0)])
    pos, neg, zero, n_small, det, _, method = KK.certified_counts(A)
    ev = np.linalg.eigvalsh(A)
    assert det and method == "ldl" and (pos, neg) == (int((ev > 0).sum()), int((ev < 0).sum()))


def _toy_resolve(J, ok=True, it=10):
    return KK.Resolve(w=np.zeros(1), lam_g=np.zeros(1), lam_x=np.zeros(1),
                      status="Solve_Succeeded" if ok else "Maximum_Iterations_Exceeded",
                      iterations=it, seconds=0.0, J=J)


def test_dose_restart_rule_on_a_toy_problem():
    """Warm continuation (design Section 4): each branch steps out from the stored key; a
    step that takes more than 200 iterations, fails, or breaks J's monotonicity is re-solved
    from the stored key (warm None), and the branch continues from the last good point."""
    calls = []

    def solve(a, warm):
        calls.append((a, None if warm is None else warm.J))
        if warm is not None and a == 0.3:
            return _toy_resolve(100.0 + a, it=250)          # too many iterations: restart
        if warm is not None and a == -1.0:
            return _toy_resolve(100.5)                       # narrowing raised J: restart
        return _toy_resolve(100.0 + a)

    steps = KK.continuation(solve, 100.0)
    by = {s.amplitude: s for s in steps}
    assert [s.amplitude for s in steps] == [0.1, 0.3, 1.0, -0.1, -0.3, -1.0, -2.0]
    assert by[0.1].warm_from == 0.0 and not by[0.1].restarted
    assert by[0.3].restarted and "iterations" in by[0.3].reasons[0]
    assert by[-1.0].restarted and "non-monotone" in by[-1.0].reasons[0]
    assert by[1.0].warm_from == 0.3 and by[-2.0].warm_from == -1.0
    assert (0.3, None) in calls and (-1.0, None) in calls      # the restarts start at the key
    assert KK.dose_branches() == ((0.1, 0.3, 1.0), (-0.1, -0.3, -1.0, -2.0))


def test_dose_continuation_on_the_synthetic_window(synth):
    """The optimum's dose-response at the synthetic key: every amplitude converges, J is
    monotone in the widening, and the inside contact keeps c_min at the 3 cm margin on the
    re-solved plans (G* = 1 to the end of the linear range)."""
    res = KK.dose_optimum(synth.prob, synth.lm, SIGMA, 0)
    assert res["all_ok"], [(r["amplitude"], r["status"]) for r in res["steps"]]
    J = {r["amplitude"]: r["J"] for r in res["steps"]}
    order = sorted(J)
    assert all(J[a] <= J[b] + 1e-9 * abs(res["J0"]) for a, b in itertools.pairwise(order))
    for r in res["steps"]:
        if abs(r["amplitude"]) <= 0.3:
            assert r["plans"]["c_min"] == pytest.approx(0.03, abs=1e-4), r
    assert KK.direction_h(PT.Samples(np.ones(3), np.zeros(3), np.zeros(3))) == KK.FD_H["kappa"]
    with pytest.raises(ValueError):
        KK.param_blocks(None, plant="no such bug")


# ------------------------------------------------------------------------- step 1's A2 keys


def _a2_problem(i: int):
    bank, _ = K.design()
    ref = [r for r in K.a2_refs() if r.W == 400][i]
    prob = K.repolish(K.rebuild(K.load_key(ref), bank)).problem
    stim = bank.stimuli[ref.stim_id]
    return ref, prob, stim.landmarks, int(stim.direction)


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's answer keys are not on this machine")
def test_gate4_and_inertia_on_a2_keys():
    """G* in [0.98, 1.02] (measured: 1 to ~1e-9) and inertia (n_free, m, 0) on three W = 400
    A2 keys of both turn directions."""
    seen = set()
    for i in (5, 120, 250):
        ref, prob, lm, sig = _a2_problem(i)
        fl = KK.key_fields(prob)
        assert fl.inertia.ok, (ref.key_id, fl.inertia)
        assert 5e-12 <= fl.mu <= 5e-11
        assert all(fl.valid(k) for k in KK.OUTCOMES), ref.key_id
        assert max(fl.adjoint_residual.values()) <= 1e-10
        g = KK.edge_gain(fl, lm, sig)
        assert KK.gate4(g) and abs(g - 1.0) < 1e-6, (ref.key_id, g)
        seen.add(sig)
    assert seen == {1, -1}


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's answer keys are not on this machine")
def test_reduced_gate1_on_an_a2_key():
    """Gate 1's pass rule (median < 1e-3, every group < 1e-2) on one A2 key with a subset of
    the design's directions."""
    ref, prob, lm, sig = _a2_problem(120)
    fl = KK.key_fields(prob)
    dirs = KK.gate1_directions(prob.track.n_points, K.bump_origin(prob.rec), lm, sig,
                               near=(0.0,), corner=PT.GATE_CORNER[1:4])
    sub = {k: v for k, v in dirs.items() if not k.startswith("w_right")}
    assert {KK.direction_group(k) for k in sub} >= {"kappa@near", "kappa@corner"}
    g1 = KK.gate1(prob, fl, sub, sig)
    assert g1.ok_solves
    err = np.array(list(g1.err.values()))
    assert np.median(err) < KK.GATE1_MEDIAN and err.max() < KK.GATE1_P95, g1.err
    assert np.abs(g1.fd_r["v_apex"]).max() > 1.0
