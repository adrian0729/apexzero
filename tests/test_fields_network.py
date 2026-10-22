"""Step 2's network side (fields.network): the float64 closed loop, its VJPs, Gates 2 and 3.

float64 network work runs in its own processes (``python -m apexzero.fields.network dry-run``,
three at once, spawned by a module fixture); this process keeps x64 off and runs the frozen
float32 harness for the references. Pre-registration: no trained network is differentiated or
perturbed. A random-init actor (G1's architecture and obs_rms, read only) and planted linear
readouts drive the loop; a trained agent (G2) enters only Gate 3's teacher-forced wiring
check on its stored rows.

- Branch consistency: the float64 loop and the float32 harness end on the same step with the
  same saturation set and outcome events; their outcome values agree to the float32 spread.
- Gate 2: VJP impulse responses match finite differences (field-vector error < 1e-3) where
  the discrete structure is shared; the tangent of strip_arrays moves pose, kappa and widths
  consistently (second order against the secant through the closed loop).
- Known answers: a centre-tracking readout's inside and outside edge gains sum to 1; mirror
  partners have mirrored fields; RF-C1 / RF-C4 shaped readouts have their closed forms.
- Gate 3: G2's stored rows pass the teacher-forced check; every planted wiring error fails.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.fields import keys as K
from apexzero.fields import network as N
from apexzero.fields import perturb as PT
from apexzero.interp import agents, io, rollout
from apexzero.tracks.generator import strip_arrays
from apexzero.tracks.spline import packed_coeffs, spline_np
from apexzero.vehicle.params import CONTEXT_RANGES

SRC = Path(N.__file__).resolve().parents[2]
HAVE_DESIGN = (io.STIMULI_DIR / "cells.json").exists()
HAVE_KEYS = (io.KEYS_DIR / "anchored").exists()


def _have_finals() -> bool:
    try:
        return all(agents.final_snapshot(agents.AGENTS[a]) is not None for a in ("G1", "G2"))
    except (KeyError, OSError):
        return False


HAVE_ALL = HAVE_DESIGN and HAVE_KEYS and _have_finals() and (io.ROLLOUTS_DIR / "G2").exists()
RANDOM_STARTS = ((62, 0), (345, 1), (102, 0), (51, 2))
LINEAR_CELLS = (344, 345, 62, 63)          # A2, D = 250 m; 344/345 and 62/63 are mirror pairs
LINEAR_STEPS = 200                         # the planted driver is slower than the agents
KEEPER_STARTS = ((380, 0), (51, 1))        # step 1's lane keeper (generic callable), slower still


# ------------------------------------------------------------------------- in-process (x64 off)


def test_x64_guard_and_upcasts():
    assert not jax.config.jax_enable_x64
    with pytest.raises(RuntimeError):
        N.require_x64()
    up = N.upcast32({"w": np.array([0.1, 1 / 3]), "i": np.array([3], np.int32)})
    assert up["w"].dtype == np.float64 and up["i"].dtype == np.int32
    np.testing.assert_array_equal(up["w"], np.array([0.1, 1 / 3], np.float32).astype(np.float64))
    W, b = N.linear_driver(-1)
    assert W.dtype == np.float32 and W.shape == (2, 222) and b[1] == np.float32(11.2)
    # row-stacked readouts: the run's aux picks the row (one harness call, several readouts)
    Ws = np.stack([W, 2 * W])
    bs = np.stack([b, b])
    o = jnp.ones(222, jnp.float32)
    m1 = N.linear_policy((Ws, bs), o, None, None, jnp.int32(1))[0]
    np.testing.assert_allclose(np.asarray(m1), 2 * W @ np.ones(222) + b, rtol=1e-6)


def test_context_channels_are_the_harness_float32_values():
    """The loop's constant context channels are bit for bit the float32 harness's (racing.py:119
    with x64 off), upcast: for the default car (the protocol's) and for other contexts."""
    k = np.zeros(400, np.float32)
    tr = strip_arrays(jnp.asarray(k), jnp.full(400, 6.0, jnp.float32),
                      jnp.full(400, 6.0, jnp.float32), 2.0)
    x = rollout.approach_state(100.0, 50.0)[None]
    lo, hi = (np.array([v[i] for v in CONTEXT_RANGES.values()], np.float32) for i in (0, 1))
    rng = np.random.default_rng(3)
    for ctx in [None] + [rng.uniform(lo, hi).astype(np.float32) for _ in range(5)]:
        ob = rollout.recompute_observations(x, np.zeros((1, 2)), np.zeros((1, 2)), tr, ctx=ctx)
        assert ob.dtype == np.float32
        np.testing.assert_array_equal(ob[0, N.CTX].astype(np.float64), N.context_channels(ctx))
    assert N.context_channels()[0] != 0.0          # XLA's float32 value, not IEEE's exact 0


@pytest.mark.skipif(not HAVE_ALL, reason="step 1's design, keys or G2's rollouts are not here")
def test_restarts():
    """A2 starts are the keys' x0 bit for bit; an anchored key restarts from its agent's row
    (x_{L-1} with a_{L-1} under option A, x_L under option L), and a tampered key is refused."""
    bank, cells = K.design()
    for ref in K.a2_refs()[::97]:
        rec = K.load_key(ref)
        st = N.a2_start(rec, bank, cells)
        assert st.decision_step == 0 and st.s1 == K.bump_origin(rec) and not len(st.a_forced)
        assert st.s_stop == bank.stimuli[rec.stim_id].landmarks.s_c1 + 50.0
    g2 = N._final_agent("G2")
    rec = N._anchored_refs("G2", g2.timesteps, 1)[0]
    det = N.load_det(io.rollout_path("G2", g2.timesteps))
    L = int(rec.anchor["step"])
    a = N.anchored_start(rec, det, bank, "A")
    i = N.det_row(det, rec.cell_id, rec.phase)
    assert a.offset == L - 1 and a.decision_step == 1
    np.testing.assert_array_equal(a.x0, rec.x0)
    np.testing.assert_array_equal(a.a_forced[0], rec.anchor["a_prev"])
    assert abs(a.s1 - rec.x[1][0]) <= N.S1_TOL
    lo = N.anchored_start(rec, det, bank, "L")
    assert lo.offset == L and lo.decision_step == 0 and not len(lo.a_forced)
    np.testing.assert_array_equal(lo.x0, det["x"][i][L].astype(np.float64))
    bad = dataclasses.replace(rec, anchor={**rec.anchor, "a_prev": [0.0, 0.5]})
    with pytest.raises(ValueError, match="a_prev"):
        N.anchored_start(bad, det, bank)
    with pytest.raises(ValueError, match="anchored"):
        N.a2_start(rec, bank, cells)


# ------------------------------------------------------------------------- float64 dry runs


def _references(out: Path) -> None:
    """The float32 harness's runs (this x64-off process) for the two dry runs."""
    bank, cells = K.design()
    g1 = N._final_agent("G1")
    starts = [(cells[c], k) for c, k in RANDOM_STARTS]
    det = N.float32_reference(rollout.agent_policy(g1.cfg), N.random_actor_params(g1.cfg, 0),
                              g1.obs_rms, bank, starts, 200)
    N.save_det(out / "random.npz", det, {"policy": "random", "seed": 0, "loop_steps": N.STEPS})
    starts = [(cells[c], 0) for c in LINEAR_CELLS]
    Ws, bs = zip(*(N.linear_driver(bank.stimuli[c.stim_id].direction) for c, _ in starts),
                 strict=True)
    det = N.float32_reference(N.linear_policy, (np.stack(Ws), np.stack(bs)), None, bank, starts,
                              LINEAR_STEPS, aux=list(range(len(starts))))
    det["lin_W"], det["lin_b"] = np.stack(Ws), np.stack(bs)
    N.save_det(out / "linear.npz", det, {"policy": "linear_driver", "loop_steps": LINEAR_STEPS})
    starts = [(cells[c], k) for c, k in KEEPER_STARTS]
    det = N.float32_reference(rollout._lane_keeper, N.KEEPER_PARAMS, None, bank, starts, 300)
    N.save_det(out / "keeper.npz", det, {"policy": "lane_keeper", "loop_steps": 300})
    starts = [(cells[c], 0) for c in N.HYBRID_CELLS]
    prm = N.hybrid_reference_params([bank.stimuli[c.stim_id].direction for c, _ in starts],
                                    g1.cfg, g1.obs_rms)
    det = N.float32_reference(N.hybrid_policy(g1.cfg), prm, g1.obs_rms, bank, starts,
                              N.HYBRID_STEPS, aux=list(range(len(starts))))
    det["hyb_Wn"] = prm[0]
    N.save_det(out / "hybrid.npz", det, {"policy": "hybrid", "loop_steps": N.HYBRID_STEPS})


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    if not HAVE_ALL:
        pytest.skip("step 1's design, keys, finals or G2's rollouts are not here")
    out = tmp_path_factory.mktemp("network")
    _references(out)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]))
    cmds = {"random": ["--f32", str(out / "random.npz"), "--keeper", str(out / "keeper.npz")],
            "linear": ["--f32", str(out / "linear.npz")], "gate3": ["--agent", "G2"],
            "hybrid": ["--f32", str(out / "hybrid.npz")]}
    procs = {k: subprocess.Popen([sys.executable, "-m", "apexzero.fields.network", "dry-run", k,
                                  *a, "--out", str(out / k)], env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
             for k, a in cmds.items()}
    res = {}
    for k, p in procs.items():
        log, _ = p.communicate(timeout=600)
        assert p.returncode == 0, f"dry-run {k} failed:\n{log[-4000:]}"
        rep = json.loads((out / f"{k}.json").read_text())
        with np.load(out / f"{k}.npz") as z:
            rep["arrays"] = {n: z[n] for n in z.files}
        res[k] = rep
    for k, rep in res.items():
        vjp = [r.get("vjp_seconds") for r in rep["rows"] if "vjp_seconds" in r]
        print(f"\n{k}: {rep['seconds']:.1f} s; VJP (forward + 4 pullbacks) "
              f"{', '.join(f'{v * 1e3:.0f}' for v in vjp)} ms")
    return res


def test_random_actor_matches_float32_harness(dry):
    """Same random actor, same starts: the float64 loop ends where the float32 harness does,
    saturates nothing it does not, reads the same means to float32 rounding, and passes the
    teacher-forced wiring check (planted errors stand out of its own error)."""
    rows = dry["random"]["rows"]
    assert len(rows) == len(RANDOM_STARTS)
    for r in rows:
        b = r["branch"]
        assert b["ok"], (r["source"], b)
        assert b["end64"][0] and b["end64"] == b["end32"]       # a random actor crashes (RF-C3)
        assert r["diag"]["max_dmean"] < 1e-5
        assert r["wiring"]["ok"], r["wiring"]
        clean = r["wiring"]["mean_err"]
        assert r["planted"]["obs_rms"]["fails"]
        for name in ("prev_action", "accel"):          # a random actor's means are ~0.01
            assert r["planted"][name]["mean_err"] > 30 * clean, (name, r["planted"])
        assert np.isfinite(r["steer"]) and r["steer"] == pytest.approx(r["steer_vjp"], abs=1e-12)
        assert r["vjp_seconds"] < 1.0
        # a crashed run's invalid outcomes have NaN impulse rows, so its G is NaN (not 1)
        assert not r["exhausted"]
        assert r["impulse_nan_rows"] == [not v for v in r["valid"]], r
        assert r["edge_gain"] == "nan" or np.isnan(r["edge_gain"])
        assert set(r["kappa_clip"]["flag"]) == set(N.OUTCOMES)
    g = rows[0]["gate2"]
    assert g["rel_err"][3] < N.GATE2_TOL and g["share_counted"][3] >= 0.5
    assert g["ok"] and g["outcome_ok"] == [True] * 4


def test_hybrid_control_matches_float32_harness(dry):
    """The pre-tag control with the trained architecture inside the loop (design Section 15):
    the linear driver plus a small-gain random-init actor through G1's obs_rms drives cells 344
    and 66 through the corner in float64 as the float32 harness does (branch consistency per
    outcome, the teacher-forced check), and the MLP takes a small, nonzero share of the
    plan-level fields."""
    rows = dry["hybrid"]["rows"]
    assert len(rows) == len(N.HYBRID_CELLS) and dry["hybrid"]["eps"] == N.HYBRID_GAIN
    for r in rows:
        assert all(r["valid"]), r["source"]
        b = r["branch"]
        assert b["ok"] and all(b["ok_by"].values()), (r["source"], b)
        assert r["wiring"]["ok"], r["wiring"]
        share = r["mlp_share"]
        assert 0.0 < share["c_min"] < 0.5 and 0.0 < share["v_apex"] < 0.5, share
    # the per-outcome verdict: a value outside the tolerance fails that outcome only
    b = N.Branch((False, -1), (False, -1), frozenset(), frozenset(), ((True, (1,)),) * 3,
                 ((True, (1,)),) * 3, {"c_min": (1.0, 1.0), "v_apex": (2.0, 2.5),
                                       "s_b": (3.0, 3.0)}, {"c_min": 1e-3, "v_apex": 1e-3,
                                                            "s_b": 1e-2})
    assert b.ok_by() == {"c_min": True, "v_apex": False, "s_b": True, "steer": True}
    b.saturated64 = frozenset({(3, 1)})
    assert b.ok_by() == {"c_min": False, "v_apex": False, "s_b": False, "steer": True}


def test_generic_callable_matches_float32_harness(dry):
    """Step 1's scripted lane keeper through the generic-callable path: same end, saturation
    set, outcome events and values (to the float32 spread) as the float32 harness."""
    rows = dry["random"]["keeper"]
    assert len(rows) == len(KEEPER_STARTS)
    for r in rows:
        assert r["branch"]["ok"], (r["source"], r["branch"])
        assert r["wiring"]["step_ok"], r["wiring"]
    assert any(all(r["valid"][:3]) for r in rows)


def test_linear_driver_fields_and_gate2(dry):
    """A planted closed-loop readout with valid outcomes: branch consistency, Gate 2 on Gate 1's
    directions for all four outcomes, and the tangent against the secant."""
    rep = dry["linear"]
    for r in rep["rows"]:
        assert all(r["valid"]), r
        assert r["branch"]["ok"], (r["source"], r["branch"])
        assert r["wiring"]["ok"], r["wiring"]
        np.testing.assert_allclose(r["vjp_values"], r["values"], rtol=1e-12, atol=1e-15)
        assert r["vjp_seconds"] < 1.0
    for r in rep["rows"][:2]:
        g = r["gate2"]
        assert all(e < N.GATE2_TOL for e in g["rel_err"]), g["rel_err"]
        assert min(g["share_counted"]) >= 0.5, g["share_counted"]
        # the readout tracks the corridor's centre: widening both edges moves c_min 1:1, so
        # the inside and outside edge gains sum to 1 (G_in + G_out = 1)
        ad = dict(zip(g["names"], g["ad"], strict=True))
        assert ad["trapezoid_inside"][0] + ad["trapezoid_outside"][0] == pytest.approx(1.0, abs=1e-9)
        assert 0.2 < 1.0 - ad["trapezoid_inside"][0] < 0.8      # it partly follows the edge
        tc = r["tangent"]
        k1, k2 = tc["kappa@near+10"]
        assert 3.5 < k1["tangent_vs_secant"] / k2["tangent_vs_secant"] < 4.5   # second order
        assert k1["tangent_vs_secant"] < 1e-3 * k1["tangent_vs_base"]
        for name in ("w_left@near+10", "w_right@corner:c0"):      # widths: linear, exact
            for d in tc[name]:
                assert d["tangent_vs_secant"] < 1e-12 and d["tangent_vs_base"] > 1e-8
        assert g["ok"] and len(g["names"]) == 3 * 8 + 2         # near and corner bumps
    r0 = rep["rows"][0]
    # the plan-level path split: the parts sum to the full VJP; the decision steer at an A2
    # start (x_0 fixed) has no dynamics part; widths reach it only through the observation
    pa = r0["paths"]
    assert pa["sum_rel_err"] <= N.PATH_TOL and pa["steer_dynamics_max"] == 0.0
    assert pa["max_abs"]["obs_kappa"][2] > 0 and pa["max_abs"]["dynamics"][0] > 0
    # the blind-pedal control: the network's own pedal tape reproduces its base run bit for
    # bit; on a policy that reads no road but through the pedal, the blind fields' observation
    # paths are exactly 0, kappa's field is the dynamics path, and v_apex's and s_b's width
    # fields vanish, while the unblinded policy's s_b reads the preview
    bp = r0["blind_pedal"]
    assert bp["reproduces"] and bp["planted_max_dx"] < 1e-5
    assert bp["blind_obs_max"] == 0.0 and bp["blind_width_vs_max"] == 0.0
    assert bp["blind_kappa_is_dynamics"] <= 1e-12 and bp["blind_kappa_max"] > 0
    assert bp["pedal_reading_share"] > 0.5
    # a loop too short for the run is a pipeline failure, never an exclusion
    ex = r0["exhaustion"]
    assert ex["exhausted"] and set(ex["reasons"].values()) == {N.EXHAUSTED}
    assert ex["fields_refused"] and ex["branch_refused"]


def test_mirror_partners_have_mirrored_fields(dry):
    """Cells 344 (left) and 345 (right) are mirror images and the readout is mirror-symmetric:
    equal outcome values; even outcomes' fields mirror (kappa negated, left and right widths
    swapped), steer's with the opposite sign; the recoded fields are equal."""
    bank, cells = K.design()
    L, R = (bank.stimuli[cells[c].stim_id] for c in LINEAR_CELLS[:2])
    assert L.direction == 1 and R.direction == -1
    np.testing.assert_array_equal(bank.kappa[R.stim_id], -bank.kappa[L.stim_id])
    rows, arr = dry["linear"]["rows"], dry["linear"]["arrays"]
    np.testing.assert_allclose(rows[0]["values"][:3], rows[1]["values"][:3], rtol=1e-12)
    for o in N.OUTCOMES:
        fl, fr = arr[f"{o}_0"], arr[f"{o}_1"]
        sign = 1.0 if o == "steer" else -1.0
        scale = np.max(np.abs(fl))
        np.testing.assert_allclose(fr[0], sign * fl[0], rtol=0, atol=1e-9 * scale)
        np.testing.assert_allclose(fr[1], -sign * fl[2], rtol=0, atol=1e-9 * scale)
        np.testing.assert_allclose(fr[2], -sign * fl[1], rtol=0, atol=1e-9 * scale)
        if o != "steer":
            a, b = PT.recode(PT.Samples(*fl), 1), PT.recode(PT.Samples(*fr), -1)
            for k in a:
                np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-9 * scale)


def _cardinal(n: int, s: float, ds: float = 2.0) -> np.ndarray:
    """L_j(s): the natural cubic spline of the unit sample e_j, evaluated at s, for every j
    (NumPy, the optimum's spline code: packed_coeffs and spline_np)."""
    out = np.zeros(n)
    j0 = int(s / ds)
    for j in range(max(0, j0 - 60), min(n, j0 + 60)):    # |L_j| < 1e-30 further out
        e = np.zeros(n)
        e[j] = 1.0
        out[j] = spline_np(packed_coeffs(e, False, n), np.array([s]), ds, n, False)[0]
    return out


def test_readout_fields_have_their_closed_forms(dry):
    """Instantaneous steer fields at an A2 start (x0 fixed): RF-C1's kappa readout gives
    c 50 L_j(s0 + d_k); RF-C4's edge readout gives c1 cos(theta_k1 - psi) / 100 L_j(s_k1) on
    w_left and -c2 cos(theta_k2 - psi) / 100 L_j(s_k2) on w_right, nothing on kappa's channel."""
    bank, cells = K.design()
    st0 = dry["linear"]["readout_start"]
    cell = cells[st0["cell"]]
    x0 = cell.initial_state(st0["phase"]).astype(np.float64)
    base = PT.upcast_row(bank, cell.stim_id)
    n = bank.kappa.shape[1]
    d = np.clip(6.0 * max(x0[3], 0.0), 150.0, 500.0) * (np.arange(1, 41) / 40) ** 1.5

    def theta(s):
        i = s / 2.0
        i0 = int(np.floor(i))
        return base.theta[i0] + (i - i0) * (base.theta[i0 + 1] - base.theta[i0])

    k, c = dry["linear"]["readouts"]["kappa"]["args"]
    c = float(np.float32(c))                     # the readouts' weights are float32 values
    f = dry["linear"]["arrays"]["readout_kappa"]
    want = c * 50.0 * _cardinal(n, x0[0] + d[int(k)])
    np.testing.assert_allclose(f[0], want, rtol=0, atol=1e-12 * np.abs(want).max())
    assert np.all(f[1] == 0) and np.all(f[2] == 0)

    k1, c1, k2, c2 = dry["linear"]["readouts"]["edges"]["args"]
    c1, c2 = float(np.float32(c1)), float(np.float32(c2))
    f = dry["linear"]["arrays"]["readout_edges"]
    psi = theta(x0[0]) + x0[2]
    s1, s2 = x0[0] + d[int(k1)], x0[0] + d[int(k2)]
    want_l = c1 * np.cos(theta(s1) - psi) / 100.0 * _cardinal(n, s1)
    want_r = -c2 * np.cos(theta(s2) - psi) / 100.0 * _cardinal(n, s2)
    np.testing.assert_allclose(f[1], want_l, rtol=0, atol=1e-12 * np.abs(want_l).max())
    np.testing.assert_allclose(f[2], want_r, rtol=0, atol=1e-12 * np.abs(want_r).max())


def test_gate3_wiring_on_a_trained_agent(dry):
    """G2's stored rows (anchored from x_{L-1}, and A2 from x0) pass the teacher-forced check
    to the outcome horizon; G1's obs_rms, a zeroed prev_action and a zeroed accel each fail on
    every row. Option A's restart (random actor): the forced float64 step lands on det_x[L],
    s_1 on the key's x[1, 0], and the decision-step steer field passes Gate 2."""
    rows = dry["gate3"]["rows"]
    assert {r["kind"] for r in rows} >= {"anchored", "nominal"}
    for r in rows:
        w = r["wiring"]
        assert w["ok"], (r["source"], w)
        assert w["mean_err"] <= N.MEAN_TOL and w["steps"][1] - w["steps"][0] > 20
        assert all(p["fails"] and p["mean_err"] > 10 * N.MEAN_TOL for p in r["planted"].values()), \
            r["planted"]
        if r["kind"] == "anchored":
            rr = r["restart_random"]
            assert rr["forced_step_excess"] <= 1.0 and rr["s1_err"] <= N.S1_TOL
            assert rr["applied"] == rr["a_prev"]
            assert rr["steer_field_norm"] > 0 and rr["vjp_seconds"] < 1.0
            assert rr["gate2"]["rel_err"][3] < N.GATE2_TOL
            # under option A the forced step moves x_1: the decision steer has a dynamics part
            assert rr["steer_dynamics_max"] > 0


# ------------------------------------------------------------------------- in-process rules


def _gate2(ad, fd, counted, valid):
    M, K_ = np.shape(ad)
    reason = [["" if valid[k] else "outcome not valid" for k in range(K_)] for _ in range(M)]
    z = np.zeros((M, K_))
    return N.Gate2([f"d{m}" for m in range(M)], np.asarray(ad, float), np.asarray(fd, float), z,
                   z, z, z, np.asarray(counted, bool), reason, np.full(K_, np.nan), 0.0)


def test_gate2_pass_rule():
    """One rule for the gates and the dry runs: a valid outcome passes only with a finite
    zero-rule error below 1e-3 (FD exactly 0 where AD is order 1: inf, fails; AD 1e-18 on a
    zero FD: a zero field, passes; nothing counted: fails); at least one valid outcome."""
    ad = [[1.0, 1.0, 1e-18, 1.0], [2.0, 2.0, 0.0, 1.0]]
    fd = [[1.0 + 1e-7, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]
    cnt = [[True, True, True, False], [True, True, True, False]]
    g = _gate2(ad, fd, cnt, [True, True, True, True])
    err = g.zero_rule()
    assert err[0] < 1e-6 and np.isinf(err[1]) and err[2] == 0.0 and np.isnan(err[3])
    assert g.outcome_ok() == [True, False, True, False] and not g.ok()
    assert _gate2(ad, fd, cnt, [True, False, True, False]).ok()       # invalid ones skipped
    assert not _gate2(ad, fd, cnt, [False, False, False, False]).ok()  # nothing valid
    assert np.array_equal(N.zero_rule_error(g), err, equal_nan=True)


def _synthetic_run(z, valid_outcomes=True, decision=0):
    """A NumPy Run of 10 steps along s (for the kappa-clipping rule)."""
    from apexzero.fields import outcomes as O
    from apexzero.interp.stimuli import Landmarks

    S = len(z)
    x = np.zeros((S + 1, 8))
    x[:, 0] = 100.0 + 10.0 * np.arange(S + 1)
    x[:, 3] = 20.0
    lm = Landmarks(100.0, 105.0, 110.0, 115.0, 120.0)   # window end s = 170: state 7
    st = N.Start(x0=x[0], prev_action=np.zeros(2), accel=np.zeros(2), a_forced=np.zeros((0, 2)),
                 decision_step=decision, s1=x[0, 0], stim_id=0, sigma=1, landmarks=lm)
    pick = O.Pick(5, 5, 0.0, False)
    plans = {o: O.Plan(o, valid_outcomes, "", 0.0 if valid_outcomes else np.nan,
                       (O.OnsetPick(4) if o == "s_b" else pick) if valid_outcomes else None)
             for o in O.OUTCOMES}
    valid = np.ones(S + 1, bool)
    return N.Run(start=st, x=x, valid=valid, mean=np.zeros((S, 2)), a=np.zeros((S, 2)),
                 z=np.asarray(z, float), term_step=-1, trunc_step=-1, stop_step=S - 1,
                 plans=plans, arrays=None)


def test_kappa_clipping_rule():
    """A kappa channel at the obs clip counts only over the observations the outcome reads
    (before its read end; the decision step's own for steer); raw-observation runs (z NaN)
    never clip; other channels never count."""
    S = 10
    z = np.zeros((S, 222))
    r = _synthetic_run(z)
    assert not any(N.kappa_clip_record(r)["flag"].values())
    z2 = z.copy()
    z2[8, N.KAPPA_CHANNELS[3]] = -10.0                  # after c_min's window end (state 6)
    r2 = _synthetic_run(z2)
    assert N.kappa_clip_steps(r2) == [(8, 3)]
    assert not N.kappa_clipped(r2, "c_min") and not N.kappa_clipped(r2, "steer")
    z3 = z.copy()
    z3[1, N.KAPPA_CHANNELS[10]] = 12.0                  # before every read end (and s_c1)
    r3 = _synthetic_run(z3)
    assert N.kappa_clipped(r3, "c_min") and N.kappa_clipped(r3, "v_apex")
    assert not N.kappa_clipped(r3, "steer")             # steer reads step 0 only
    assert N.kappa_clipped(_synthetic_run(z3, decision=2), "steer")
    z4 = z.copy()
    z4[2, 1] = 50.0                                     # an edge channel: not kappa
    assert not N.kappa_clipped(_synthetic_run(z4), "c_min")
    assert not N.kappa_clipped(_synthetic_run(np.full((S, 222), np.nan)), "c_min")
    assert N.kappa_clip_record(r3)["before_s_c1"]


def test_branch_tolerance_derivation():
    """5 x the controls' float32/float64 spread per unit teacher-forced action error, scaled
    to the trained networks' largest teacher-forced error."""
    rows = [{"mean_err": 1e-7, "values": {"c_min": (1.0, 1.0 + 1e-6), "v_apex": (20.0, 20.0),
                                          "s_b": (900.0, 900.0 + 1e-4)}},
            {"mean_err": 2e-7, "values": {"c_min": (1.0, 1.0 + 1e-5), "v_apex": (20.0, 20.0 + 2e-6),
                                          "s_b": (float("nan"), 900.0)}}]
    res = N.branch_tolerance(rows, 3e-5)
    assert res["tol"]["c_min"] == pytest.approx(5 * (1e-5 / 2e-7) * 3e-5)
    assert res["tol"]["v_apex"] == pytest.approx(5 * (2e-6 / 2e-7) * 3e-5)
    assert res["tol"]["s_b"] == pytest.approx(5 * (1e-4 / 1e-7) * 3e-5)
    assert N.branch_tolerance(rows, 3e-5, floor={"v_apex": 1.0})["tol"]["v_apex"] == 1.0


def test_normaliser_scale_and_clip_enter_the_policy_input():
    """policy_input's Jacobian is diagonal: 1 / sqrt(var + 1e-8) on unclipped channels and
    exactly 0 on a channel pushed past |z| = 10 (RF-C1's c / sqrt(var) closed form)."""
    from apexzero.agents.ppo import RunningMeanStd

    with jax.enable_x64(True):
        var = np.linspace(0.5, 4.0, 222)
        mean = np.zeros(222)
        mean[7] = -100.0                                 # channel 7 clips at z = +10
        rms = RunningMeanStd(mean=jnp.asarray(mean), var=jnp.asarray(var), count=jnp.float64(1.0))
        ob = jnp.zeros(222, jnp.float64)
        J = np.asarray(jax.jacfwd(lambda o: N.policy_input(o, rms)[0])(ob))
        d = np.diag(J)
        assert np.abs(J - np.diag(d)).max() == 0.0
        assert d[7] == 0.0
        keep = np.arange(222) != 7
        np.testing.assert_allclose(d[keep], 1.0 / np.sqrt(var[keep] + 1e-8), rtol=1e-12)


def test_trained_networks_refused_before_the_tag_and_the_lockbox():
    """A policy marked trained (as agent_net marks every snapshot) cannot have fields, path
    fields, kink radii, Gate 2, tangents or any closed-loop run (base or perturbed: no trained
    network's free loop before the tag) before ``unlock_trained``; the teacher-forced check
    (``policy_means`` / ``wiring_check``) stays open. ``road`` applies step 1's lockbox."""
    from types import SimpleNamespace

    W, b = N.linear_driver(1)
    net = dataclasses.replace(N.linear_net(W, b), trained=True)
    assert not N._TRAINED_OK
    with jax.enable_x64(True):
        for call in (lambda: N.fields(net, None, None),
                     lambda: N.path_fields(net, None, None),
                     lambda: N.kink_radius(net, None, None, None),
                     lambda: N.gate2(net, None, None, {}),
                     lambda: N.tangent_consistency(net, None, None, None, 1e-3),
                     lambda: N.run(net, None, None, PT.zeros(3)),
                     lambda: N.run(net, None, None)):
            with pytest.raises(PermissionError, match="waits for the tag"):
                call()
    bank = SimpleNamespace(stimuli=[SimpleNamespace(name="E1/Monza/1200/real",
                                                    source={"track": "Monza"})])
    with pytest.raises(Exception, match="(?i)monza|lockbox|sealed|R_TEST"):
        N.road(bank, 0)


def test_loop_length_covers_the_stored_horizons(dry):
    """File reads only (in the gate3 dry run's process): every stored det run of G2 (A2 set
    and anchored finals) ends within STEPS - HORIZON_MARGIN steps of the network's restart,
    and none runs out of its own stored steps alive."""
    audit = dry["gate3"]["loop_audit"]
    g2 = audit["agents"]["G2"]
    assert g2["a2"]["n"] > 400 and g2["finals"]["n"] > 600
    assert audit["ok"], audit
    assert audit["max"] <= N.STEPS - N.HORIZON_MARGIN and audit["max"] > 150
