"""Stimuli (protocol 3, step-1 builder checks of protocol 10): synthetic corners, the bank,
cells, transplants, crash ladders, twins, tier D conditions and the design."""

import collections
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.interp import io
from apexzero.interp import stimuli as T
from apexzero.interp import support as S
from apexzero.interp.constants import DS, N_KNOTS, S_C, S_TRUNC, TWIN_TOL
from apexzero.tracks.generator import strip_arrays
from apexzero.vehicle.sim import track_arrays


@pytest.fixture(scope="module", autouse=True)
def _float32():
    """Rollouts, controls and stimuli arrays run in float32. tests/test_dynamics.py and
    tests/test_env.py switch x64 on globally when they are imported, so in a full-suite run
    this module switches it off for its own tests and restores it afterwards."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", prev)


LT = S.LTypTable(l_nodes=(26.0, 33.0, 40.0, 44.0, 46.0), counts=(1, 1, 1, 1, 1))
VT = S.VTypTable(choice_kph={f"{b}|{c}": v for b, v in enumerate((160.0, 160.0, 180.0, 220.0, 250.0))
                             for c in ("lt120", "ge120")})


def _support(tmp_path=None) -> S.SupportTables:
    tables, paths = {}, {}
    if tmp_path is not None:
        rng = np.random.default_rng(0)
        for g in S.GRAMMARS:
            n = 400
            t = {"R_seg": np.exp(rng.uniform(np.log(8), np.log(400), n)),
                 "theta_deg": rng.uniform(5, 200, n), "l_r": rng.uniform(10, 60, n),
                 "L_c": rng.uniform(15, 300, n), "v_peak": rng.uniform(25, 87, n)}
            paths[g] = str(io.save_table(tmp_path / f"corners_{g}.npz", t))
            tables[g] = t
    sd = tuple(np.std(S.table_features(tables["base"]), axis=0)) if tables else (1.0,) * 5
    return S.SupportTables(LT, VT, {}, sd, paths, "")


@pytest.fixture(scope="module")
def design():
    return T.build_design(_support())


# ------------------------------------------------------------------------- synthetic


def test_corner_spec_ramp_rules():
    abrupt = T.corner_spec(6.0, 90.0, 1, ramp="abrupt")
    assert abrupt.l_r == 10.0 and abrupt.is_triangle
    assert 1.0 / abrupt.kappa_peak == pytest.approx(6.366, abs=1e-3)     # generator triangle
    tri = T.corner_spec(13.1, 90.0, 1, l_typ_table=LT)                    # theta R < l_typ
    assert tri.l_r == pytest.approx(math.pi / 2 * 13.1) and tri.kappa_peak == pytest.approx(1 / 13.1)
    arc = T.corner_spec(40.0, 90.0, -1, l_typ_table=LT)                   # plateau
    assert arc.l_r == pytest.approx(S.l_typ(40.0, LT)) and arc.L_p > 0
    assert arc.kappa_peak == pytest.approx(1 / 40.0)
    small = T.corner_spec(6.0, 90.0, 1, l_typ_table=LT)                   # theta R < 10
    assert small.l_r == 10.0
    assert T.corner_spec(9.4, 90.0, 1, ramp="abrupt", w_in_corner=6.0).w_in_corner is None
    with pytest.raises(ValueError):
        T.corner_spec(13.0, 90.0, 1)                                      # typical needs l_typ


def test_corner_profile_formula_widths_and_landmarks():
    spec = T.corner_spec(13.1, 180.0, -1, l_typ_table=LT)
    p = T.corner_profile(spec)
    s = np.arange(N_KNOTS) * DS
    xs = [spec.s_c, spec.s_c + spec.l_r, spec.s_c + spec.l_r + spec.L_p, spec.s_c + spec.L_c]
    ref = -spec.kappa_peak * np.interp(s, xs, [0.0, 1.0, 1.0, 0.0])
    assert np.max(np.abs(p.kappa - ref)) <= 1e-12
    assert np.all(p.w_left == 6.0) and np.all(p.w_right == 6.0)        # R 13.1: cap slack
    lm = p.landmarks
    assert (lm.s_c0, lm.s_c1) == (S_C, pytest.approx(S_C + spec.L_c))
    assert lm.s_a0 == pytest.approx(S_C + spec.l_r) and lm.s_apex == pytest.approx(S_C + spec.L_c / 2)
    # E3 narrowing follows the corner's own profile on the inside only, then the cap.
    e3 = T.corner_profile(T.corner_spec(9.4, 90.0, 1, ramp="abrupt", w_in_corner=4.6))
    apex = round(e3.landmarks.s_apex / DS)
    assert e3.w_left[apex] == pytest.approx(4.6) and np.all(e3.w_right == 6.0)
    assert e3.w_left[0] == 6.0
    nocap = T.corner_profile(T.corner_spec(6.0, 90.0, 1, l_typ_table=LT, no_cap=True))
    assert np.all(nocap.w_left == 6.0) and not nocap.capped.any()


@pytest.mark.parametrize(("width", "r_bind"), [(12.0, 8.33), (15.0, 10.83)])
def test_inside_cap_binds_where_the_protocol_says(width, r_bind):
    # A plateau puts knots at kappa = 1/R (a triangle's peak can fall between knots, and
    # the cap is applied knot by knot, as in the generator).
    for R in (r_bind - 0.1, r_bind + 0.1):
        p = T.corner_profile(T.corner_spec(R, 90.0, 1, width, ramp="abrupt"))
        inside = np.where(p.kappa > 0, p.w_left, p.w_right)
        assert np.max((inside - 1.0) * np.abs(p.kappa)) <= 0.6 + 1e-12
        assert p.capped.any() == (R < r_bind)
    wl, wr, capped = T.apply_inside_cap(np.array([0.2, -0.2, 0.0]), np.full(3, 6.0), np.full(3, 6.0))
    assert wl[0] == pytest.approx(4.0) and wr[1] == pytest.approx(4.0) and capped.tolist() == [True, True, False]


def test_mirror_is_exact_on_samples_and_arrays():
    p = T.corner_profile(T.corner_spec(9.4, 90.0, 1, ramp="abrupt", w_in_corner=4.6))
    m = T.mirror(p)
    np.testing.assert_array_equal(m.kappa, -p.kappa)
    np.testing.assert_array_equal(m.w_left, p.w_right)
    a = strip_arrays(*(jnp.asarray(x, jnp.float32) for x in (p.kappa, p.w_left, p.w_right)), DS)
    b = strip_arrays(*(jnp.asarray(x, jnp.float32) for x in (m.kappa, m.w_left, m.w_right)), DS)
    np.testing.assert_array_equal(np.asarray(b.kappa_c), -np.asarray(a.kappa_c))
    np.testing.assert_array_equal(np.asarray(b.theta), -np.asarray(a.theta))
    np.testing.assert_array_equal(np.asarray(b.y), -np.asarray(a.y))
    np.testing.assert_array_equal(np.asarray(b.x), np.asarray(a.x))
    x = np.arange(8.0) + 1.0
    np.testing.assert_array_equal(T.mirror_state(T.mirror_state(x)), x)
    assert T.mirror_state(x).tolist() == [1, -2, -3, 4, -5, -6, -7, 8]
    pa, ac = T.mirror_aux(np.array([0.3, 0.9]), np.array([2.0, -1.0]))
    assert pa.tolist() == [-0.3, 0.9] and ac.tolist() == [2.0, 1.0]


def test_shift_corner_is_an_exact_translation():
    spec = T.corner_spec(11.7, 90.0, 1, l_typ_table=LT)
    p = T.shift_corner(spec, 0.75)
    s = np.arange(N_KNOTS) * DS
    ref = spec.kappa_peak * T.trapezoid(s - (S_C + 0.75), spec.l_r, spec.L_p)
    assert np.max(np.abs(p.kappa - ref)) <= 1e-12
    assert p.landmarks.s_c0 == S_C + 0.75 and p.landmarks.s_c1 == pytest.approx(S_C + 0.75 + spec.L_c)
    assert "dc+0.750" in T.stimulus_name(T.replace(spec, s_c=S_C + 0.75))


def test_realised_geometry_of_ramps():
    # Abrupt 10 m ramps overshoot (brief: +1.7%); a typical triangle whose peak falls
    # between knots undershoots 1/R by up to a few percent (recorded as R_eff).
    for ramp, overshoot in (("abrupt", (0.005, 0.03)), ("typical", (-0.03, 0.01))):
        spec = T.corner_spec(13.0, 90.0, 1, ramp=ramp, l_typ_table=LT)
        p = T.corner_profile(spec)
        tr = strip_arrays(*(jnp.asarray(x, jnp.float32) for x in (p.kappa, p.w_left, p.w_right)), DS)
        g = T.realised_geometry(tr, p.landmarks, p.kappa, p.w_left, p.w_right, p.capped,
                                spec.l_r, spec.kappa_peak, 1)
        excess = (1.0 / spec.kappa_peak) / g.R_eff - 1.0
        assert overshoot[0] <= excess <= overshoot[1]
        assert abs(g.theta_eff_deg - 90.0) < 0.3 and g.n_segments == 1
        assert g.onset_rate == pytest.approx(spec.kappa_peak / spec.l_r)
        assert g.ringing < 0.02 * spec.kappa_peak and g.R_seg >= g.R_eff   # spline ringing


# ------------------------------------------------------------------------- bank and cells


def test_bank_arrays_track_roundtrip_and_save(tmp_path):
    specs = [T.corner_spec(R, 90.0, d, l_typ_table=LT) for R in (9.4, 44.7) for d in (1, -1)]
    profs = [T.corner_profile(s) for s in specs]
    rows = [T._Row(T.stimulus_name(s), p, "synthetic", s.direction, s, {"A"}, l_r=s.l_r,
                   kappa_peak=s.kappa_peak) for s, p in zip(specs, profs)]
    stims, _ = T._assemble(rows)
    bank = T.bank_from_profiles(profs, stims)
    tr_b = bank.arrays()
    for i in range(len(bank)):
        a = track_arrays(bank.track(i))
        b = jax.tree.map(lambda x, i=i: x[i], tr_b)
        for f in ("kappa_c", "w_left_c", "w_right_c", "x", "y", "theta"):
            np.testing.assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)))
        assert bank.track(i).name == stims[i].name
    assert T.stimulus_track(tr_b, 1, "x").kappa.shape == (N_KNOTS,)
    bank.save(tmp_path)
    back = T.StimulusBank.load(tmp_path)
    assert back.content_hash() == bank.content_hash() and back.stimuli == bank.stimuli
    assert back.by_name(stims[2].name).stim_id == 2
    with pytest.raises(ValueError):
        T.StimulusBank(bank.kappa, bank.w_left, bank.w_right, stims[::-1])


def test_cell_initial_states(tmp_path):
    c = T.Cell(cell_id=0, stim_id=0, pair_id=0, tiers=("A",), approach="A1", v0=310 / 3.6, D=600.0)
    x0 = c.initial_state(0)
    assert x0.dtype == np.float32 and x0[0] == 400.0 and x0[3] == np.float32(310 / 3.6)
    with pytest.raises(ValueError):
        c.initial_state(1)                       # no phase cycle before the key layout
    c.P = 4.0
    assert c.initial_state(2)[0] == pytest.approx(398.0)
    np.testing.assert_allclose(c.phase_offsets(), [0.0, 1.0, 2.0, 3.0])
    r = T.Cell(cell_id=1, stim_id=0, pair_id=0, tiers=("E1",), approach="real", v0=60.0, D=800.0,
               P=8.0, x0=(200.0, 0.3, 0.01, 60.0, 0.1, 0.02, 0.01, 0.9), runners=("S1",))
    assert r.initial_state(1)[0] == pytest.approx(198.0) and r.initial_state(1)[1] == np.float32(0.3)
    a2 = T.Cell(cell_id=2, stim_id=0, pair_id=1, tiers=("A",), approach="A2", v0=44.4, D=None)
    with pytest.raises(ValueError):
        a2.initial_state(0)
    T.save_cells(tmp_path / "cells.json", [c, r, a2])
    assert T.load_cells(tmp_path / "cells.json") == [c, r, a2]


# ------------------------------------------------------------------------- real roads


def test_lockbox_and_r_dev_only():
    with pytest.raises(PermissionError):
        T.real_track("Monza")
    with pytest.raises(PermissionError):
        T.real_track("spa_reversed")
    fake = T.Corner(100.0, 140.0, 1.0, 20.0, 42.0)
    with pytest.raises(PermissionError):
        T.transplant("Suzuka", fake)
    with pytest.raises(PermissionError):
        T.ladder("Silverstone", fake)
    with pytest.raises(PermissionError):           # a Track carrying a lockbox name
        T.hockenheim_conditions(0, track=T.replace(T.real_track("Hockenheim"), name="Monza"))
    with pytest.raises(ValueError):
        T.real_track("IMS")


def test_transplant_matches_its_source():
    tr = T.real_track("Norisring")
    c = T.crash_corner(tr, 910.0)
    p, exc = T.transplant("Norisring", c)
    i = np.arange(N_KNOTS)
    j = i - exc.i_c + exc.j_c
    inside = (j >= exc.j_a) & (j <= exc.j_b)
    np.testing.assert_allclose(p.kappa[inside], tr.kappa[np.mod(j[inside], tr.n_points)],
                               atol=1e-12, rtol=0)
    assert np.all(p.kappa[(j < exc.j_a - 10) | (j > exc.j_b + 10)] == 0.0)       # padding
    np.testing.assert_array_equal(p.w_left[j < exc.j_a], tr.w_left[exc.j_a % tr.n_points])
    assert p.landmarks.s_c0 == S_C and p.landmarks.L_c == pytest.approx(c.length / tr.ds * DS)
    assert exc.to_strip(c.s_start) == pytest.approx(S_C)
    assert exc.to_strip(c.s_start + tr.length) == pytest.approx(S_C)          # unwraps
    # A corner within 800 m of s = 0 wraps to the end of the lap.
    h = T.real_track("Hockenheim")
    ch = T.crash_corner(h, 265.0)
    ph, eh = T.transplant("Hockenheim", ch)
    assert eh.j_a < 0
    assert eh.to_strip(0.0) == pytest.approx(S_C - ch.s_start / h.ds * DS)
    k0 = round(eh.to_strip(0.0) / DS) - 5
    assert ph.kappa[k0] == h.kappa[(k0 - eh.i_c + eh.j_c) % h.n_points]


def test_crash_corners_carry_no_radius_labels():
    dc = T.distinct_crash_corners()
    assert len(dc) == 13
    assert sum(len(v[2]) for v in dc.values()) == 15
    shared = {k for k, v in dc.items() if len(v[2]) == 2}
    assert {k.split("/")[1] for k in shared} == {"Sochi", "Oschersleben"}
    # R2: no unit carries a radius label; each crash corner is the segment corner that
    # contains the crash (D15), and its R_seg is what the regimes read.
    assert not any(hasattr(p, "label_R") for _, _, ps in dc.values() for p in ps)
    r_seg = {(p.agent, p.track): c.r_min for _, c, ps in dc.values() for p in ps}
    assert 19.0 < r_seg[("G2", "YasMarina")] < 22.0 and 26.0 < r_seg[("G2", "Zandvoort")] < 30.0


def test_twin_fit_recovers_a_trapezoid_and_flags_long_corners():
    spec = T.corner_spec(16.0, 100.0, -1, l_typ_table=LT)
    meas = T._measure_twin(spec)
    tw = T.twin_fit(meas["R_seg"], meas["theta_deg"], meas["L_seg"], -1, 12.0)
    assert tw.ok and tw.spec.ramp_kind == "twin" and tw.spec.s_c == S_C
    assert abs(tw.achieved["R_seg"] / meas["R_seg"] - 1) <= TWIN_TOL["R_seg_rel"]
    long = T.twin_fit(meas["R_seg"], meas["theta_deg"], 3.0 * meas["L_seg"], -1, 12.0)
    assert not long.ok and "exceeds" in long.achieved["note"]
    # R_seg and the angle are still matched: only the length is out of reach.
    assert abs(long.achieved["R_seg"] / meas["R_seg"] - 1) <= TWIN_TOL["R_seg_rel"]
    assert abs(long.achieved["theta_deg"] - meas["theta_deg"]) <= TWIN_TOL["angle_deg"]


def test_ladder_versions():
    tr = T.real_track("Norisring")
    c = T.crash_corner(tr, 910.0)
    profs, exc = T.ladder("Norisring", c, tr)
    assert set(profs) == set(T.LADDER_VERSIONS)
    real = profs["real"]
    i_a = exc.i_c - (exc.j_c - exc.j_a)
    lead = slice(i_a, i_a + T.LEAD_KNOTS + 1)
    for v, p in profs.items():
        if v != "mirror":                                    # the lead-in is always real
            np.testing.assert_array_equal(p.kappa[lead], real.kappa[lead])
            np.testing.assert_array_equal(p.w_left[lead], real.w_left[lead])
    mid_before = slice(i_a + 200, i_a + 300)
    assert np.all(profs["before_straight"].kappa[mid_before] == 0.0)
    corner = slice(T.I_C, T.I_C + round(c.length / tr.ds))
    np.testing.assert_array_equal(profs["before_straight"].kappa[corner], real.kappa[corner])
    assert np.all(profs["after_straight"].kappa[T.I_C + 200:] == 0.0)
    cw = profs["const_width"]
    far = slice(i_a + T.LEAD_KNOTS + T.TAPER_KNOTS + 1, N_KNOTS)
    assert np.ptp((cw.w_left + cw.w_right)[far][~cw.capped[far]]) < 1e-12
    np.testing.assert_array_equal(profs["mirror"].kappa, -real.kappa)
    tw = profs["twin"]
    assert tw.landmarks.s_c0 == S_C and np.all(tw.kappa[i_a + 100:T.I_C - 1] == 0.0)
    e2, _ = T.ladder("Norisring", c, tr, versions=T.E2_VERSIONS)
    assert set(e2) == {"real", "isolated", "twin"}


def test_hockenheim_conditions():
    h = T.real_track("Hockenheim")
    dcs = T.hockenheim_corners(h)
    assert len(dcs) == 20 and all(100.0 <= d.D_gap <= 600.0 for d in dcs)
    hair = min(dcs, key=lambda d: d.corner.r_min)
    conds, exc = T.hockenheim_conditions(hair.index, h)
    assert set(conds) == set(T.D_CONDITIONS) - {"F5/swapA"}
    conds, exc = T.hockenheim_conditions(hair.index, h, swap_j=(hair.index + 2) % 20)
    assert set(conds) == set(T.D_CONDITIONS)
    a = conds[T.COND_A]
    i_a = exc.i_c - (exc.j_c - exc.j_a)
    assert i_a * DS == pytest.approx(S_C - hair.before, abs=DS / 2)
    lead = slice(i_a, i_a + T.LEAD_KNOTS + 1)
    for name, p in conds.items():
        if name == "F4/mirror":
            continue
        np.testing.assert_array_equal(p.kappa[lead], a.kappa[lead], err_msg=name)
        np.testing.assert_array_equal(p.w_left[lead], a.w_left[lead], err_msg=name)
        np.testing.assert_array_equal(p.w_right[lead], a.w_right[lead], err_msg=name)
    iso = conds[T.COND_ISO]
    assert np.all(iso.kappa[T.I_C + round(hair.corner.length / DS) + 30:] == 0.0)
    far = slice(i_a + T.LEAD_KNOTS + T.TAPER_KNOTS + 1, N_KNOTS)
    assert np.allclose((iso.w_left + iso.w_right)[far][~iso.capped[far]], 12.0)
    # Dose: the corner is scaled about the region start; landmarks follow.
    for lam in (0.8, 1.25):
        p = conds[f"F2/lambda{lam:g}"]
        assert p.landmarks.s_c0 == pytest.approx(S_C - 20 + 20 * lam)
        assert p.landmarks.L_c == pytest.approx(lam * a.landmarks.L_c)
        reg = slice(T.I_C - 10, round(p.landmarks.s_c1 / DS) + 10)
        ref = slice(T.I_C - 10, round(a.landmarks.s_c1 / DS) + 10)
        assert np.max(np.abs(p.kappa[reg])) == pytest.approx(np.max(np.abs(a.kappa[ref])) / lam,
                                                             rel=0.05)
    wplus = conds["F2/width+0.5"]
    assert (wplus.w_left - a.w_left)[far][~wplus.capped[far]] == pytest.approx(0.25)
    np.testing.assert_array_equal(conds["F4/mirror"].kappa, -a.kappa)
    # F6 curvature noise: peak level and zero net heading.
    for tag, level in (("0.5e-3", 0.5e-3), ("1e-3", 1e-3)):
        d = conds[f"F6/knoise{tag}"].kappa - a.kappa
        assert np.max(np.abs(d)) == pytest.approx(level) and abs(np.sum(d)) < 1e-12
    wob = conds["F6/wobble0.6"].w_left - a.w_left
    assert np.max(np.abs(wob)) == pytest.approx(0.6, abs=0.02)
    np.testing.assert_allclose(conds["F6/wobble0.6"].w_right - a.w_right, wob)
    assert np.all(conds["F5/swapB"].kappa[T.I_C:] == 0.0)


def test_swap_partner():
    v = {0: 50.0, 1: 70.0, 2: 52.0, 3: 100.0, 4: 80.0}
    assert T.swap_partner(0, v, {0, 1, 2, 3, 4}) == 1        # |log(50/70)| = 0.336
    assert T.swap_partner(0, v, {0, 2}) is None              # 0.039 < 0.15
    assert T.swap_partner(9, v, {0, 1}) is None


def test_matched_corners_are_distinct_and_far_from_crashes():
    crash = [(t, c) for (t, c, _) in T.distinct_crash_corners().values()]
    rdev = {n: T.real_track(n) for n in T.R_DEV}
    m = T.matched_corners(rdev, crash, {n: float("inf") for n in T.R_DEV})
    assert len(m) == 13 and len({(t, c.s_start) for t, c, _ in m}) == 13
    for (tc, cc), (tm, cm, lap) in zip(crash, m):
        assert S.theta_class(math.degrees(abs(cc.angle))) == S.theta_class(math.degrees(abs(cm.angle)))
        assert lap == 2
        for t2, c2 in crash:
            if t2 == tm:
                assert T._cyclic_gap(cm, c2, rdev[tm].ds, rdev[tm].length) >= 1000.0
    short = T.matched_corners(rdev, crash, {n: 1500.0 for n in T.R_DEV})
    assert all(c.s_start + c.length + 250.0 <= 1500.0 and lap == 1 for _, c, lap in short)


# ------------------------------------------------------------------------- the design


def test_design_counts(design):
    bank, cells = design
    pairs = collections.defaultdict(set)
    for c in cells:
        for t in c.tiers:
            pairs[t].add(c.pair_id)
    assert {t: len(p) for t, p in pairs.items()} == {
        "A": 74, "B": 108, "C": 23, "E3": 32, "V": 14, "G": len(pairs["G"])}
    assert 8 <= len(pairs["G"]) <= 20
    both = {c.pair_id for c in cells if "A" in c.tiers and "B" in c.tiers}
    assert len(both) == 12
    assert len({c.pair_id for c in cells if "A" in c.tiers and "E3" in c.tiers}) == 4
    assert sum(1 for c in cells if "recovery" in c.flags) == 24
    by_pair = collections.defaultdict(list)
    for c in cells:
        by_pair[c.pair_id].append(c)
    for members in by_pair.values():
        if len(members) == 2:
            L, R = members
            assert not L.mirrored and R.mirrored
            assert bank.stimuli[L.stim_id].mirror_id == R.stim_id
    p7 = [c for c in cells if "p7" in c.flags]
    assert len(p7) == 20 and all(c.n_phases == 16 and c.approach == "A1" for c in p7)
    assert all(c.D is None for c in cells if c.approach == "A2")
    assert all(c.null_cell >= 0 for c in cells if c.approach != "G")
    names = [s.name for s in bank.stimuli]
    assert len(names) == len(set(names))
    assert {s.kind for s in bank.stimuli} == {"synthetic", "straight"}


def test_design_checks(design):
    bank, cells = design
    res = T.check_design_details(bank, cells)
    for k in ("kappa_formula", "track_arrays_roundtrip", "mirror_exact", "inside_cap", "lengths",
              "lockbox"):
        assert res[k]["ok"], (k, res[k])
    # A synthetic-only design has no transplant or twin rows: those checks are vacuous and
    # say so (step 1 checks them on the stand-in real build), they do not pass.
    for k in ("transplant_source", "twin_fit"):
        assert res[k]["vacuous"] and res[k]["rows"] == 0, (k, res[k])
    assert res["lengths"]["longest_corner"] <= 521.0
    assert res["lengths"]["max_window_reach"] < S_TRUNC
    # theta_eff: the protocol's 0.3 deg check misses exactly the known rows, each by the
    # samples' own area (kinks between knots); ruling R3 accepts and lists them.
    assert res["theta_eff"]["ok"] and not res["theta_eff"]["fails"]
    assert res["theta_eff"]["accepted_by"] == "R3"
    failing = sorted(f["name"] for f in res["theta_eff"]["accepted"])
    assert len(failing) == 6 and all(("/R6/" in n and "/T180/" in n) or ("/R8.4/" in n and "/T90/" in n)
                                     for n in failing), failing
    for f in res["theta_eff"]["accepted"]:
        assert f["theta_eff"] == pytest.approx(math.copysign(abs(f["sample_area_deg"]), f["theta_eff"]),
                                               abs=0.01)
        # R3 accepts a sampling miss only: the spec's trapezoid turns its nominal angle and
        # the samples' area lies within the kinks' sampling bound of it.
        nominal = 180.0 if "/T180/" in f["name"] else 90.0
        assert f["spec_area_deg"] == pytest.approx(nominal, abs=1e-6)
        assert 0.3 < nominal - abs(f["sample_area_deg"]) <= f["sampling_bound_deg"]
    assert res["theta_eff"]["spec_area_max_deg"] <= 1e-6


def test_pilot_minigrid():
    bank, cells = T.pilot_minigrid()
    assert len(bank) == 24 and bank.kappa.shape == (24, 1280)
    assert np.all(bank.w_left == 5.5) and np.all(bank.w_right == 5.5)
    assert all(c.x0 == (100.0, 0.0, 0.0, 80.0, 0.0, 0.0, 0.0, 0.0) and c.W == 500 for c in cells)
    assert all(s.landmarks.s_c0 == 800.0 and s.spec.l_r == 10.0 for s in bank.stimuli)
    assert all(T.check_design(bank, cells).values())


def test_build_real_design_with_stand_in_states(design):
    """Real-lap states come from agents in the Integrate phase; stand-ins exercise the
    mapping, the ladders and tier D here without any agent."""
    bank, cells = design
    keys = {r["key"]: r for r in T.real_state_requests() if r["agent"] in ("G2", "S1")}
    e1 = next(k for k in keys if k.startswith("E1/Norisring/874"))
    d14 = "D/14"

    def st(r, s):
        return T.RealState(r["agent"], r["track"], r["lap"], s, (s, 0.4, 0.02, 70.0, 0.1, 0.03, 0.02, 1.0),
                           (0.1, 1.0), (2.0, 0.5))

    r_e1 = keys[e1]
    hk = T.real_track("Hockenheim")
    r_d = next(r for r in T.real_state_requests() if r["key"] == d14 and r["agent"] == "S1")
    states = {"G2": {e1: st(r_e1, r_e1["s_target"])},
              "S1": {d14: st(r_d, r_d["s_target"] % hk.length)}}
    bank2, cells2 = T.build_real_design(bank, cells, states)
    new = cells2[len(cells):]
    assert len(bank2) == len(bank) + 7 + 22 and len(new) == 7 + 22
    e1_cells = [c for c in new if c.tiers == ("E1",)]
    real = next(c for c in e1_cells if bank2.stimuli[c.stim_id].name.endswith("/real"))
    mir = next(c for c in e1_cells if bank2.stimuli[c.stim_id].name.endswith("/mirror"))
    assert real.pair_id == mir.pair_id and mir.mirrored and not real.mirrored
    assert real.x0[0] == pytest.approx(S_C - 800.0, abs=DS) and real.D == pytest.approx(800.0, abs=DS)
    assert mir.x0[1] == -real.x0[1] and mir.prev_action0 == (-0.1, 1.0) and mir.accel0 == (2.0, -0.5)
    assert len({c.pair_id for c in e1_cells}) == 6
    assert not any(f.startswith("label_mismatch") for f in real.flags)     # R2: no labels
    d_cells = [c for c in new if c.tiers == ("D",)]
    assert all(c.runners == T.RUNNERS_FROM_S1 for c in d_cells)
    assert d_cells[0].D == pytest.approx(T.hockenheim_corners(hk)[14].before, abs=DS)
    res = T.check_design_details(bank2, cells2)
    assert res["transplant_source"]["ok"] and res["transplant_source"]["rows"] == 29
    assert res["lockbox"]["ok"] and res["mirror_exact"]["ok"] and res["track_arrays_roundtrip"]["ok"]
    # R4: Norisring 874 is 76 m long, longer than any trapezoid with its R_seg and angle; its
    # twin matches both to < 1e-3, so it is accepted with its shortfall and flagged as short,
    # not as a failed fit.
    twin = next(s for s in bank2.stimuli if s.name.startswith("E1/Norisring/874")
                and s.name.endswith("/twin"))
    assert "twin_length_short" in twin.flags and "twin_fit_fail" not in twin.flags
    tw = res["twin_fit"]
    assert tw["ok"] and [r["name"] for r in tw["accepted"]] == [twin.name]
    assert tw["accepted_max_R_rel"] < 1e-3 and tw["accepted_max_angle_deg"] < 1e-3
    # the twin's spec turns its angle exactly; only the 6 known synthetic rows miss 0.3 deg
    th = res["theta_eff"]
    assert th["ok"] and th["spec_area_max_deg"] <= 1e-6 and len(th["accepted"]) == 6


def test_cell_support_table(tmp_path, design):
    bank, cells = design
    sup = _support(tmp_path)
    tab = T.cell_support_table(bank, cells, sup)
    assert set(tab) == set(T.CELL_SUPPORT_COLUMNS) and len(tab["cell"]) == len(cells)
    a1 = np.array([c.approach == "A1" for c in cells])
    a2 = np.array([c.approach == "A2" for c in cells])
    assert np.all(np.isfinite(tab["logdens_base"][a1])) and np.all(np.isnan(tab["v_peak"][a2]))
    assert np.all(tab["v_peak"][a1] > 300 / 3.6)
