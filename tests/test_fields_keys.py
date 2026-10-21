"""Step 2's key loader (fields.keys): the state sets, strata, the rebuilt NLP, Gate 0 and the
re-polish, on step 1's stored keys (read only; nothing is written).

Counts are the design's (receptive-fields-design.md, Section 2). Gate 0's keys span the kinds
and both machines that solved step 1's keys (rebuilt defects ~1e-12 here, ~1e-8 for keys from
the rented machine). No agent network is evaluated: final snapshots are opened only for their
timesteps."""

import copy

import numpy as np
import pytest

from apexzero.fields import keys as K
from apexzero.fields import perturb as PT
from apexzero.interp import io
from apexzero.ocp.window import NA, NX

pytestmark = pytest.mark.skipif(not (io.KEYS_DIR / "anchored").exists(),
                                reason="step 1's answer keys are not on this machine")

FINALS = {"G1": 701, "G2": 722, "G3": 751, "G1s1": 710, "G1s2": 705, "G2s1": 742, "G2s2": 723,
          "G3s1": 760, "G3s2": 759, "S1": 766, "S2": 404, "S3": 393}


def _spread(refs, n):
    """n refs evenly spaced through a (sorted) list."""
    refs = list(refs)
    return [refs[i] for i in np.linspace(0, len(refs) - 1, n).round().astype(int)]


@pytest.fixture(scope="module")
def gate_refs():
    a2 = K.a2_refs()
    fin = K.anchored_finals()
    nom = [r for r in a2 if r.kind == "nominal"]
    pha = [r for r in a2 if r.kind == "phase"]
    return (_spread(nom, 3) + _spread(pha, 3) + _spread(fin["G1"], 1) + _spread(fin["S3"], 2)
            + _spread(fin["G2s2"], 1))


# ------------------------------------------------------------------------- the state sets


def test_a2_state_set():
    refs = K.a2_refs()
    _, cells = K.design()
    nom = [r for r in refs if r.kind == "nominal"]
    pha = [r for r in refs if r.kind == "phase"]
    assert (len(nom), len(pha), len(refs)) == (142, 339, 481)
    assert sorted(np.unique([r.W for r in nom], return_counts=True)[1]) == [43, 99]
    assert [sum(r.W == w for r in nom) for w in (400, 800)] == [99, 43]
    assert [sum(r.W == w for r in pha) for w in (400, 800)] == [264, 75]
    assert all(r.verified and cells[r.cell_id].approach == "A2" for r in refs)
    assert {r.phase for r in nom} == {0} and {r.phase for r in pha} == {1, 2, 3}
    # tier V's horizon twins: six cells hold a W = 400 and a W = 800 nominal key
    st = [K.strata(r) for r in refs]
    assert sum(not s.w_is_cell for s in st) == 6
    assert len({(r.cell_id, r.phase) for r in refs}) == 475
    # the per-pair summaries' set: one key per (cell, phase); the twins listed apart
    prim, twins = K.a2_primary_refs(), K.a2_twins()
    assert len(prim) == 475 and len(twins) == 6
    assert len({(r.cell_id, r.phase) for r in prim}) == 475
    assert {r.key_id for r in prim} | {r.key_id for r in twins} == {r.key_id for r in refs}


def test_anchored_finals_and_earlier():
    fin = K.anchored_finals()
    assert {a: len(v) for a, v in fin.items()} == FINALS
    assert sum(len(v) for v in fin.values()) == 8136
    assert K.final_timesteps("G1") == 200_000_000
    assert {K.final_timesteps(a) for a in FINALS if a != "G1"} == {199_884_800}
    for aid, refs in fin.items():
        assert all(r.agent == aid and r.verified and r.snapshot == str(K.final_timesteps(aid))
                   for r in refs)
    early = K.anchored_earlier()
    assert sum(len(v) for v in early.values()) == 1389
    assert {a for a, _ in early} == {"G1", "G2", "G3"}
    assert all(s != str(K.final_timesteps(a)) for a, s in early)


def test_strata():
    bank, cells = K.design()
    fin = K.anchored_finals()
    first = {}
    for v in fin.values():
        for r in v:
            first.setdefault(r.stim_id, r)
    an = {s: K.strata(r) for s, r in first.items()}
    assert len(an) == 60 and {st.width for st in an.values()} == {12.0}
    assert {st.R for st in an.values() if st.cap_binds} == {6.0, 7.5}
    assert sum(st.cap_binds for st in an.values()) == 8
    assert {st.approach for st in an.values()} <= {"A1", "B"}
    st = [K.strata(r) for r in K.a2_refs()]
    assert {s.R for s in st if s.cap_binds} == {6.0, 7.5, 9.4}   # tier C's W 15 at R 9.4 too
    assert {s.direction for s in st} == {1, -1} and {s.approach for s in st} == {"A2"}
    assert {s.width for s in st} == {9.0, 12.0, 15.0}
    # mirror pairs: each pair id carries both directions
    by_pair: dict[int, set] = {}
    for s in st:
        by_pair.setdefault(s.pair_id, set()).add(s.direction)
    assert all(v == {1, -1} for v in by_pair.values())
    rec = K.load_key(fin["G2"][0])
    assert K.decision_step(rec) == 1 and K.bump_origin(rec) == rec.x[1, 0]
    rec2 = K.load_key(K.a2_refs()[0])
    assert K.decision_step(rec2) == 0 and K.bump_origin(rec2) == rec2.x[0, 0]


# ------------------------------------------------------------------------- the rebuilt NLP


class _Stop(Exception):
    pass


def _recorder(seen: dict):
    """An IPOPT stand-in that records the call's arguments and stops the solve."""
    def call(**kw):
        seen.update(kw)
        raise _Stop

    return call


def test_rebuild_matches_the_frozen_solver(gate_refs):
    """P and the bounds are the ones AnchoredSolver.solve builds from the same warm start (its
    IPOPT call is intercepted before it runs)."""
    for ref in (gate_refs[0], gate_refs[6]):          # an A2 nominal key and an anchored one
        rec = K.load_key(ref)
        prob = K.rebuild(rec)
        seen: dict = {}
        s = copy.copy(prob.solver)
        s.solver = _recorder(seen)
        with pytest.raises(_Stop):
            s.solve(prob.track, rec.x0, (rec.x, rec.a), ctx=prob.ctx, p=prob.p,
                    a0_fixed=rec.a[0] if K.is_anchored(rec) else None)
        for k in ("lbx", "ubx", "lbg", "ubg"):
            np.testing.assert_array_equal(seen[k], getattr(prob, k))
        np.testing.assert_array_equal(seen["p"], prob.P)
        n_fixed = NX + (NA if K.is_anchored(rec) else 0)
        assert int((~prob.free).sum()) == n_fixed


def test_window_centring_does_not_change_the_constraints(gate_refs):
    """The lookups are exact inside a window, so centres moved by a few metres give the same
    g at the stored point (bit for bit)."""
    prob = K.rebuild(K.load_key(gate_refs[3]))
    sigma2 = prob.sigma + 4.0
    P2 = K.key_parameters(prob.solver, prob.track, sigma2, prob.ctx, prob.p)
    assert not np.array_equal(P2, prob.P)
    np.testing.assert_array_equal(K.constraints(prob, P=P2)[1], K.constraints(prob)[1])


def test_gate0(gate_refs):
    kinds = {r.kind for r in gate_refs}
    assert kinds == {"nominal", "phase", "anchored"} and len(gate_refs) == 10
    res = [K.gate0(K.rebuild(K.load_key(r))) for r in gate_refs]
    for r, g in zip(gate_refs, res, strict=True):
        assert g.ok and g.r_free <= 1e-6, (r.key_id, g)
        assert g.defect <= 1e-6 and g.corridor_violation <= 1e-6 and g.slip_violation <= 1e-6
        assert abs(g.J_rebuilt - g.J_stored) <= 1e-9 * abs(g.J_stored)
    # both machines are represented: some keys rebuild exactly, some to ~1e-8
    d = np.array([g.defect for g in res])
    assert d.min() < 1e-10 and d.max() > 1e-9


def test_gate0_has_power(gate_refs):
    """Planted errors fail Gate 0: lam_g's sign flipped, lam_x dropped, and the other car."""
    prob = K.rebuild(K.load_key(gate_refs[1]))
    assert K.gate0(prob).ok
    flipped = copy.copy(prob)
    flipped.lam_g = -prob.lam_g
    assert K.gate0(flipped).r_free > 1e-2
    no_bounds = copy.copy(prob)
    no_bounds.lam_x = np.zeros_like(prob.lam_x)
    assert K.gate0(no_bounds).r_free > 1e-4
    other_ctx = copy.copy(prob)
    other_ctx.P = K.key_parameters(prob.solver, prob.track, prob.sigma,
                                   prob.ctx * np.array([1, 1, 1.01, 1, 1, 1]), prob.p)
    assert K.gate0(other_ctx).r_free > 1e-4


def test_repolish(gate_refs):
    """An A2 key (solved here) does not move; a key from the rented machine (rebuilt defects
    ~1e-8) is polished back to 1e-10 defects with a tiny J change and stays stationary."""
    a2 = K.rebuild(K.load_key(gate_refs[0]))
    cands = [K.rebuild(K.load_key(r)) for r in gate_refs[7:9]]
    other = max(cands, key=lambda q: K.gate0(q).defect)
    assert K.gate0(other).defect > 1e-9
    for prob, moves in ((a2, False), (other, True)):
        rp = K.repolish(prob)
        assert rp.ok, rp.status
        assert rp.defect_before <= 1e-6 and rp.defect_after <= 1e-9
        assert abs(rp.dJ_rel) <= 1e-8 and rp.max_da <= 1e-5 and rp.max_dx <= 1e-2
        assert K.gate0(rp.problem).r_free <= 1e-9
        assert rp.flagged == (rp.max_dx > K.MOVE_X_TOL or abs(rp.dJ_rel) > K.MOVE_J_RTOL)
        g0, g1 = K.constraints(prob)[1], K.constraints(rp.problem)[1]
        T = prob.T
        if not moves:
            assert rp.max_dx <= 1e-9 and not rp.flagged
            assert np.max(np.abs(g1 - g0)) <= 1e-6            # P reproduces the stored g
        else:
            assert rp.iterations >= 1 and rp.max_dx > 0.0
            assert rp.dx_step.shape == (T + 1,) and rp.dx_step.max() == rp.max_dx
            assert rp.dx_step[:50].max() <= 1e-4                # early steps barely move
            assert np.max(np.abs(g0[: NX * T])) <= 1e-6        # stored primal nearly feasible
            assert np.max(np.abs(g1[: NX * T])) <= 1e-9
        assert set(K.summary(rp)) >= {"key_id", "max_dx", "dJ_rel", "flagged", "ok"}


def test_parameters_follow_sample_perturbations(gate_refs):
    """P of the track plus a sample perturbation = P plus the coefficient windows of the
    perturbation's natural-spline coefficients (packed_coeffs is linear): the optimum-side
    entry of design Section 4."""
    prob = K.rebuild(K.load_key(gate_refs[6]))
    n = prob.track.n_points
    rng = np.random.default_rng(0)
    delta = PT.Samples(1e-4 * rng.standard_normal(n), 0.1 * rng.standard_normal(n),
                       0.1 * rng.standard_normal(n))
    P1 = K.key_parameters(prob.solver, prob.track, prob.sigma, prob.ctx, prob.p, delta)
    sl = K.parameter_slices(prob.T, prob.spec.n_coef)
    idx = prob.coefficient_index()
    from apexzero.tracks.spline import packed_coeffs
    for name, d in zip(("kappa", "w_left", "w_right"), delta, strict=True):
        dc = packed_coeffs(d, False, n)[idx]                      # [T, K]
        np.testing.assert_allclose(P1[sl[name]] - prob.P[sl[name]], dc.reshape(-1),
                                   rtol=0, atol=1e-12 * max(1.0, np.abs(dc).max()) + 1e-15)
    for name in ("j0", "q", "ctx", "alpha_max"):
        np.testing.assert_array_equal(P1[sl[name]], prob.P[sl[name]])
    assert sl["alpha_max"].stop == prob.solver.n_param
