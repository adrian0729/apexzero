"""Step 2's planted fields and unlawful readers through the network pipeline
(fields.controls; design Section 8).

The float64 work runs in two processes at once (``python -m apexzero.fields.controls --quick``,
spawned by a module fixture); this process keeps x64 off and only uses float64 inside
``jax.enable_x64`` blocks (restored by a fixture). No trained network is evaluated: the
controls are linear readouts of the raw observation, step 1's scripted Q family, planted edge
keepers and a random-init actor (G1's architecture and obs_rms, read only).

- RF-C1: the kappa readout's bump fields equal (c / sqrt(var)) 50 kappa_at(s_1 + d_k*; G_d) to
  cosine 1 and gain 1, peak and E90 within a bin, near / mid / far, sigma_b = 4, 8, 16 m.
- RF-C4: the edge readout's width fields have their closed forms; the recoding table puts the
  left field inside on a left key and the right field inside on a right key.
- RF-C2: Q's steer field is gain x K(d) through the pose path only; the frozen pursuit's AD
  drops the understeer slope on a straight, the AD-safe one does not.
- RF-C3: a random actor crashes before the window end; its instantaneous steer field passes
  Gate 2.
- RF-C5: a planted edge keeper has G = 1 and a fixed-n keeper G = 0, within 0.02.
- Readers: Qp, c1 (15 m late), c2 (kappa cap) and c3 (time to corner) on the preview land
  outside H1-typical's band on the criteria they violate; the AD-safe pieces keep the frozen
  values and give the derivatives of the frozen law.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.fields import controls as C
from apexzero.interp import agents, io
from apexzero.interp import controls as QC
from apexzero.interp import measures as M

SRC = Path(C.__file__).resolve().parents[2]


def _have() -> bool:
    try:
        return ((io.STIMULI_DIR / "cells.json").exists() and (io.KEYS_DIR / "nominal").exists()
                and agents.final_snapshot(agents.AGENTS["G1"]) is not None)
    except (KeyError, OSError):
        return False


HAVE = _have()


@pytest.fixture(autouse=True)
def _restore_x64():
    prev = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not HAVE:
        pytest.skip("step 1's design, keys or G1's final are not here")
    out = tmp_path_factory.mktemp("controls")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    groups = {"instant": ["RF-C1", "RF-C4", "RF-C2"], "plan": ["RF-C3", "RF-C5", "readers"]}
    procs = {k: subprocess.Popen([sys.executable, "-m", "apexzero.fields.controls", "--quick",
                                  "--out", str(out / f"{k}.json"), "--which", *w], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for k, w in groups.items()}
    res = {}
    for k, p in procs.items():
        log, _ = p.communicate(timeout=600)
        assert p.returncode == 0, f"controls {k} failed:\n{log[-4000:]}"
        res.update(json.loads((out / f"{k}.json").read_text()))
    for name, v in res.items():
        if isinstance(v, dict) and "seconds" in v:
            print(f"\n{name}: {v['seconds']:.1f} s, pass {v['pass']}")
    return res


# ------------------------------------------------------------------------- in-process pieces


def test_pursuit_kernel_closed_form():
    """K(d) = int_0^l (l - u) G(u - d) du: for a narrow bump well inside (0, l) it is
    (l - d) sigma_b sqrt(2 pi); beyond l + 5 sigma_b it vanishes; behind the car it decays."""
    l, sb = 40.0, 0.5
    d = np.array([5.0, 20.0, 35.0, 60.0])
    K = C.pursuit_kernel(d, l, sb, n_u=40001)
    np.testing.assert_allclose(K[:3], (l - d[:3]) * sb * np.sqrt(2 * np.pi), rtol=1e-6)
    assert K[3] < 1e-12
    assert C.pursuit_kernel(np.array([-30.0]), l, 8.0)[0] < 1e-3 * C.pursuit_kernel(
        np.array([10.0]), l, 8.0)[0]


@pytest.fixture(scope="module")
def car():
    return QC.car_tables(M.model_tables())


def test_understeer_term_keeps_values_and_its_slope(car):
    with jax.enable_x64(True):
        cj = QC.CarTables(*(jnp.asarray(a, jnp.float64) for a in car))
        v = jnp.float64(40.0)
        for k in (-0.02, -1e-4, 0.0, 3e-3, 0.05):
            kk = jnp.float64(k)
            ay = jnp.maximum(QC._uinterp(v, cj.v, cj.a_y), QC._AY_FLOOR)
            u = v * v * jnp.abs(kk) / ay
            frozen = jnp.sign(kk) * QC._bilinear(v, u, cj.v, cj.us_u, cj.us_dd)
            assert float(C.understeer_ad(v, kk, cj)) == float(frozen)
        g_safe = float(jax.grad(lambda k: C.understeer_ad(v, k, cj))(jnp.float64(0.0)))

        def frozen_term(k):
            ay = jnp.maximum(QC._uinterp(v, cj.v, cj.a_y), QC._AY_FLOOR)
            return jnp.sign(k) * QC._bilinear(v, v * v * jnp.abs(k) / ay, cj.v, cj.us_u, cj.us_dd)

        h = 1e-7
        fd = float((frozen_term(jnp.float64(h)) - frozen_term(jnp.float64(-h))) / (2 * h))
        assert float(jax.grad(frozen_term)(jnp.float64(0.0))) == 0.0     # JAX's sign drops it
        assert g_safe > 0 and g_safe == pytest.approx(fd, rel=1e-6)


def test_pedal_law_and_qss_pass_keep_values_with_true_derivatives(car):
    with jax.enable_x64(True):
        cj = QC.CarTables(*(jnp.asarray(a, jnp.float64) for a in car))
        args = dict(car=cj, k_v=jnp.float64(QC.K_V), lead=jnp.float64(QC.LEAD_TIME),
                    phi=jnp.float64(QC.PHI))
        v, ps, at = jnp.float64(50.0), jnp.float64(0.2), jnp.float64(-5.0)
        lo, hi = jnp.float64(-1.0), jnp.float64(1.0)

        def frozen(vt):
            return QC.law_a(v, ps, vt, at, lo, hi, **args)

        def safe(vt):
            return C.law_a_ad(v, ps, vt, at, lo, hi, **args)

        for vt in (45.0, 49.0, 50.5, 80.0, 10.0):          # interior and both clips
            assert float(safe(jnp.float64(vt))) == float(frozen(jnp.float64(vt)))
        vt = jnp.float64(49.0)
        h = 1e-3
        fd = float((frozen(vt + h) - frozen(vt - h)) / (2 * h))
        assert abs(float(jax.grad(frozen)(vt))) < 1e-6 * abs(fd)    # the bisection's AD: ~0
        assert float(jax.grad(safe)(vt)) == pytest.approx(fd, rel=1e-4)
        assert float(jax.grad(safe)(jnp.float64(80.0))) == 0.0      # clipped at p_hi
        # the QSS pass: same values, finite derivatives where lat saturates
        kap = jnp.abs(jnp.linspace(0.0, 0.2, 60))
        np.testing.assert_array_equal(np.asarray(C.grip_profile_ad(kap, 2.0, cj)),
                                      np.asarray(QC.grip_profile(kap, 2.0, cj)))
        g = jax.grad(lambda k: C.grip_profile_ad(k, 2.0, cj).sum())(kap)
        g0 = jax.grad(lambda k: QC.grip_profile(k, 2.0, cj).sum())(kap)
        assert np.isfinite(np.asarray(g)).all() and not np.isfinite(np.asarray(g0)).all()


def test_keeper_and_reader_parameters():
    if not HAVE:
        pytest.skip("step 1's design is not here")
    st, rd = C.start(344)
    kp = C.keeper_params(st, rd, True)
    assert kp.W0 == pytest.approx(6.0, abs=1e-6) and kp.A == pytest.approx(6.0 - 1.0 - 1.5)
    assert C.keeper_params(st, rd, False).reads_width == 0.0
    with pytest.raises(ValueError):
        C.reader_net("Q", st.stim_id)


# ------------------------------------------------------------------------- the controls


def test_rf_c1_kappa_readout(runs):
    r = runs["RF-C1"]
    assert r["pass"]
    for cell in r["cells"].values():
        ds = [ro["d_k"] for ro in cell["readouts"].values()]
        assert ds[0] < 30 < ds[1] < 120 < ds[2]                   # near, mid, far
        for ro in cell["readouts"].values():
            assert ro["width_field_max"] == 0.0
            for ch in ro["checks"]:
                assert ch["cosine"] >= 1 - 1e-9 and abs(ch["gain"] - 1) <= 1e-9
                assert abs(ch["peak"] - ch["d_k"]) <= 1.0
                assert abs(ch["e90"] - ch["e90_want"]) <= 0.5
    # the normalised-observation closed form (c / sqrt(var)): G1's obs_rms at the first cell,
    # with a scale far from 1, so a wrong scale could not pass
    c0 = next(iter(r["cells"].values()))
    nr = c0["obs_rms"]
    assert nr["pass"]
    assert any(abs(ro["scale"] - 1.0) > 0.1 for ro in nr["readouts"].values())
    for ro in nr["readouts"].values():
        for ch in ro["checks"]:
            assert ch["cosine"] >= 1 - 1e-9 and abs(ch["gain"] - 1) <= 1e-9


def test_rf_c4_width_readout_and_recoding_table(runs):
    r = runs["RF-C4"]
    assert r["pass"] and r["recoding_table"] == {"1": "w_left", "-1": "w_right"}
    for cell in r["cells"].values():
        assert abs(cell["steer"]) <= 0.5
        for side in ("w_left", "w_right"):
            for ch in cell["checks"][side]:
                assert ch["pass"] and ch["cosine"] >= 1 - 1e-9 and abs(ch["gain"] - 1) <= 1e-9
        assert cell["table"]["w_in"]["rel_err"] < 1e-12


def test_rf_c2_q_pose_path(runs):
    r = runs["RF-C2"]
    assert r["pass"]
    for cell in r["cells"].values():
        for which in ("frozen", "ad_safe"):
            c = cell[which]
            assert c["shape_ok"] and c["kappa_c_path_max"] == 0.0 and c["width_field_max"] == 0.0
            assert c["pose_path_max"] > 0 and c["path_sum_rel_err"] <= 1e-9
            assert min(ch["cosine"] for ch in c["checks"]) >= 0.999
        assert cell["ad_safe"]["amplitude_ok"]
        assert cell["gains"]["ad_safe"] == pytest.approx(cell["gains"]["fd"], rel=1e-7)
        # the frozen pursuit's AD misses the understeer slope on a straight (~14% here)
        assert 0.05 < cell["understeer_share"] < 0.3
        assert not cell["frozen"]["amplitude_ok"]
        # the plan-level split: the decision steer's observation-kappa path is exactly 0
        assert cell["plan_obs_kappa_steer_max"] == 0.0


def test_rf_c3_random_actor(runs):
    r = runs["RF-C3"]
    assert r["pass"]
    for cell in r["cells"].values():
        assert cell["plan_outcomes_invalid"] == ["c_min", "v_apex", "s_b"]
        assert cell["term_step"] >= 0 and cell["gate2_steer_rel_err"] < 1e-3
        assert cell["steer_field_norm"] > 0


def test_rf_c5_edge_trackers(runs):
    r = runs["RF-C5"]
    assert r["pass"]
    for cell in r["cells"].values():
        e, f = cell["edge"], cell["fixed"]
        assert abs(e["G"] - 1) <= C.RF_C5_TOL and abs(f["G"]) <= 1e-8   # spline ringing of h
        assert e["on_plateau"] and f["on_plateau"] and e["B"] > 0.5
        assert e["e_star"] >= 0.99 and f["e_star"] >= 0.99
        assert e["vjp_seconds"] < 1.0


def test_unlawful_readers(runs):
    """Each violator detected against Qp (its base law) on every cell where it is judged
    (c1 on at least two); the negative checks (Qp judged as each violator, c2 above its cap)
    never flagged; Qp itself is not lawful here, so the absolute criteria are descriptive."""
    r = runs["readers"]
    assert r["pass"], (r["judged"], r["negative"])
    assert set(r["judged"]) == {"c1", "c2", "c3"} and all(r["judged"][v] for v in r["judged"])
    assert len(r["judged"]["c1"]) >= 2
    assert all(len(n) and not any(n) for n in r["negative"].values()), r["negative"]
    assert not r["qp_lawful"]
    for cell in r["cells"].values():
        for v in C.READERS:
            row = cell[v]
            for o in ("s_b", "v_apex"):
                if o in row:
                    assert row[o]["finite"] and "delta_rho" in row[o]
    # c2's capped corner reads no curvature in the corner: its apex speed field vanishes
    capped = r["cells"]["94"]["c2"]["v_apex"]
    assert abs(capped["g"]) < 0.05 and abs(r["cells"]["94"]["Qp"]["v_apex"]["g"]) > 0.1
    # c1 brakes later than Qp by about its 15 m late read
    for cid in ("46", "134", "343"):
        c1, qp = r["cells"][cid]["c1"], r["cells"][cid]["Qp"]
        if c1.get("violation_detected") is not None:
            assert c1["values"]["s_b"] - qp["values"]["s_b"] >= C.C1_OFFSET_MIN
    assert set(r["null"]) == {"s_b", "v_apex"}                 # one null per outcome


def test_reader_judging_rules():
    qp = {"values": {"s_b": 950.0}, "s_b": {"g": 0.3}, "v_apex": {"g": 0.4}}
    late = {"values": {"s_b": 962.0}, "s_b": {"g": 0.29}, "v_apex": {"g": 0.4}}
    assert C.judge_reader("c1", late, qp, False) is True
    assert C.judge_reader("c1", qp, qp, False) is False            # Qp relabelled
    assert C.judge_reader("c3", {**qp, "s_b": {"g": 0.05}}, qp, False) is True
    assert C.judge_reader("c3", qp, qp, False) is False
    assert C.judge_reader("c2", {**qp, "v_apex": {"g": 1e-6}}, qp, True) is True
    assert C.judge_reader("c2", {**qp, "v_apex": {"g": 1e-6}}, qp, False) is None   # uncapped
    assert C.judge_reader("Qp", qp, qp, True) is None
    assert C.judge_reader("c3", {"values": {}}, qp, False) is None   # not judged there
