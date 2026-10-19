"""Trajectory export: geometry, the OCP-to-time mapping, cutting policy laps, corner
sectors, the schema and its validation, and the lockbox guard."""

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from _viz_fixtures import circle, make_doc, rounded_square, synthetic_ocp

from apexzero.vehicle.params import F1_2025 as P
from apexzero.viz import export
from apexzero.viz.sectors import corner_sectors, sector_deltas

HOCKENHEIM_OCP = Path("outputs/ocp/Hockenheim.pkl")
TUMFTM = Path("data/tracks/tumftm")


# --------------------------------------------------------------------------- geometry


def test_frenet_to_world_matches_the_simulator_pose():
    import jax.numpy as jnp

    from apexzero.vehicle.sim import global_pose, track_arrays

    tr = rounded_square()
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 2.5 * tr.length, 500)       # includes later laps
    n = rng.uniform(-5, 5, 500)
    xi = rng.uniform(-0.2, 0.2, 500)
    x, y, h = export.frenet_to_world(tr, s, n, xi)
    state = np.zeros((500, 8))
    state[:, 0], state[:, 1], state[:, 2] = s, n, xi
    X, Y, H = (np.asarray(a) for a in global_pose(jnp.asarray(state), track_arrays(tr)))
    assert np.max(np.hypot(x - X, y - Y)) < 2e-3
    assert np.max(np.abs(np.angle(np.exp(1j * (h - H))))) < 1e-4


def test_numpy_tyre_model_matches_jax():
    import jax.numpy as jnp

    from apexzero.vehicle.dynamics import tyre_forces
    from apexzero.vehicle.ops import JAX

    rng = np.random.default_rng(1)
    x = np.c_[np.zeros(200), rng.normal(0, 2, 200), rng.normal(0, 0.05, 200),
              rng.uniform(5, 90, 200), rng.normal(0, 1, 200), rng.normal(0, 0.3, 200),
              rng.normal(0, 0.1, 200), rng.uniform(-1, 1, 200)]
    a = tyre_forces(x, P, export.NUMPY)
    b = tyre_forces(jnp.asarray(x, jnp.float32), P, JAX)
    np.testing.assert_allclose(a.util_f, np.asarray(b.util_f), rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(a.util_r, np.asarray(b.util_r), rtol=2e-4, atol=1e-5)


# --------------------------------------------------------------------------- OCP -> time


def test_ocp_ghost_maps_stations_to_a_monotone_time_grid():
    tr = rounded_square()
    res = synthetic_ocp(tr)
    g = export.ocp_ghost(res, tr, rate_hz=20.0)
    t, s = g.t, g.s
    assert t[0] == 0.0 and s[0] == 0.0
    assert t[-1] == res.lap_time == g.lap_time        # the lap time is the OCP's own
    assert s[-1] == pytest.approx(tr.length, abs=1e-9)
    assert np.all(np.diff(t) > export.MIN_GAP)
    assert np.all(np.diff(s) > 0)
    np.testing.assert_allclose(np.diff(t)[:-1], 0.05, atol=1e-9)
    # Distance at the OCP's own station times is reproduced (piecewise-linear error only).
    np.testing.assert_allclose(np.interp(res.t, t, s), res.s, atol=0.02)
    # And the speed channel integrates back to the lap time.
    lap = np.sum(np.diff(s) / (0.5 * (g.channels["speed"][1:] + g.channels["speed"][:-1])))
    assert lap == pytest.approx(res.lap_time, rel=1e-4)


def test_ocp_ghost_can_keep_the_stations():
    tr = rounded_square()
    res = synthetic_ocp(tr)
    g = export.ocp_ghost(res, tr, rate_hz=None)
    np.testing.assert_array_equal(g.t, np.r_[res.t, res.lap_time])
    np.testing.assert_allclose(g.s, np.r_[res.s, tr.length])
    np.testing.assert_allclose(g.channels["speed"][:-1], res.z[:, 2])
    np.testing.assert_allclose(g.channels["n"][:-1], res.z[:, 0])
    assert g.channels["throttle"].max() == pytest.approx(0.3)


def test_ocp_ghost_positions_lie_on_the_line():
    tr = circle()
    res = synthetic_ocp(tr)
    res.z[:, 0] = 0.0
    g = export.ocp_ghost(res, tr)
    r = np.hypot(g.channels["x"], g.channels["y"])
    # On the (spline-smoothed, so +/-0.1 m) centreline wherever it is between stations.
    r_track = np.interp(g.s, np.r_[tr.s, tr.length], np.r_[np.hypot(tr.x, tr.y),
                                                            np.hypot(tr.x[0], tr.y[0])])
    np.testing.assert_allclose(r, r_track, atol=0.01)
    np.testing.assert_allclose(r, 150.0, atol=0.15)
    assert np.all(np.abs(np.diff(g.channels["heading"])) < 0.2)       # unwrapped


def test_ocp_ghost_rejects_a_mismatched_track():
    tr = rounded_square()
    res = synthetic_ocp(tr)
    with pytest.raises(ValueError, match="OCP is for"):
        export.ocp_ghost(res, circle())
    res.meta["track"] = "circle"
    with pytest.raises(ValueError, match="spans"):
        export.ocp_ghost(res, circle())


@pytest.mark.skipif(not (HOCKENHEIM_OCP.exists() and TUMFTM.exists()),
                    reason="needs outputs/ocp/Hockenheim.pkl and the TUMFTM tracks")
def test_hockenheim_ocp_ghost_keeps_the_solved_lap_time():
    from apexzero.tracks.reference import refine_reference
    from apexzero.tracks.track import load_tumftm

    tr = refine_reference(load_tumftm(TUMFTM / "Hockenheim.csv"))
    with open(HOCKENHEIM_OCP, "rb") as fh:
        res = pickle.load(fh)
    g = export.ocp_ghost(res, tr)
    assert g.lap_time == res.lap_time and g.t[-1] == res.lap_time
    assert np.all(np.diff(g.t) > 0) and np.all(np.diff(g.s) > 0)
    assert g.channels["ellipse_use"].max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------- policy laps


def test_lap_window_times_laps_like_the_harness():
    L, v, dt = 1000.0, 37.0, 0.05
    t = np.arange(4000) * dt
    s = v * t
    tl, ch = export.lap_window(t, {"s": s, "speed": np.full_like(t, v)}, L, 2 * L)
    assert tl[0] == 0.0 and ch["s"][0] == 0.0 and ch["s"][-1] == L
    assert tl[-1] == pytest.approx(L / v, abs=1e-9)
    assert np.all(np.diff(tl) > export.MIN_GAP)
    with pytest.raises(ValueError, match="never reaches"):
        export.lap_window(t, {"s": s}, L, 1e6)


def test_ghost_from_a_timed_laps_trace():
    tr = circle()
    v, dt = 30.0, 0.05
    k = np.arange(1, 2500)
    trace = {"t": k * dt, "s": v * k * dt, "n": np.zeros(len(k)), "vx": np.full(len(k), v),
             "action": np.tile([0.05, 0.2], (len(k), 1))}
    g = export.ghost_from_lap_trace(trace, tr, lap=2)
    assert g.lap_time == pytest.approx(tr.length / v, abs=1e-6)
    assert "util_front" not in g.channels and "ellipse_use" not in g.channels
    # The course angle of a centred car is the track heading.
    _, _, th = export.frenet_to_world(tr, g.s, 0 * g.s)
    d = np.angle(np.exp(1j * (g.channels["heading"] - th)))
    assert np.max(np.abs(d[1:-1])) < 0.02


def _ring_policy(obs):
    """A hand-written controller for the ring test: curvature feed-forward plus feedback
    on the lateral position and heading error, and a speed hold at 30 m/s."""
    import jax.numpy as jnp

    from apexzero.env.racing import OBS_PREVIEW

    kappa = obs[:, 4] / 50.0
    lateral, xi, vx = obs[:, OBS_PREVIEW + 4], obs[:, OBS_PREVIEW + 5], 50.0 * obs[:, OBS_PREVIEW]
    steer = kappa * (P.lf + P.lr) / P.delta_max - 0.6 * lateral - 1.5 * xi
    pedal = 0.1 * (30.0 - vx)
    return jnp.clip(jnp.stack([steer, pedal], axis=-1), -1.0, 1.0)


def test_policy_ghost_reproduces_the_harness_lap():
    import jax

    from apexzero.eval.laps import timed_laps

    tr = circle()
    # The harness is float32-only, and earlier tests (the OCP checks) switch x64 on globally.
    with jax.enable_x64(False):
        harness = timed_laps(_ring_policy, tr, 30.0, runs=1)
    assert harness.completed[0]
    with jax.enable_x64(True):          # the export forces float32 itself
        ro = export.rollout_policy(_ring_policy, tr, 30.0)
    g = export.policy_ghost(ro, tr)
    assert g.lap_time == pytest.approx(float(harness.lap_time[0]), abs=1e-3)
    assert set(export.CHANNELS) <= set(g.channels)
    assert g.channels["ellipse_use"].max() <= 1.0 + 1e-6
    assert g.s[0] == 0.0 and g.s[-1] == pytest.approx(tr.length)
    with pytest.raises(ValueError, match="not completed"):
        export.policy_ghost(ro, tr, lap=50)


# --------------------------------------------------------------------------- sectors


@pytest.mark.parametrize("start_frac", [0.0, 0.3, 0.62])
def test_sectors_tile_the_lap_and_sum_to_the_lap_delta(start_frac):
    tr = rounded_square(start_frac=start_frac)
    secs = corner_sectors(tr)
    L = tr.length
    assert len(secs) == 4
    assert sum(s.s_end - s.s_start for s in secs) == pytest.approx(L)
    for a, b in zip(secs, secs[1:] + secs[:1], strict=True):
        gap = np.mod(a.s_end - b.s_start, L)       # each sector ends where the next starts
        assert min(gap, L - gap) < 1e-6
    for s in secs:
        assert s.s_start <= s.s_entry < s.s_exit <= s.s_end
        assert abs(s.angle) == pytest.approx(np.pi / 2, abs=0.1)
        assert 0 <= s.s_apex < L
    grid = np.r_[tr.s, L]
    d = np.cumsum(np.random.default_rng(2).normal(0.0, 0.01, len(grid)))
    d -= d[0]
    assert sector_deltas(grid, d, secs, L).sum() == pytest.approx(d[-1] - d[0], abs=1e-12)


def test_close_corners_merge_into_one_complex():
    from apexzero.tracks.corners import segment

    tr = rounded_square(side=40.0, radius=40.0)     # 40 m straights
    seg = segment(tr)
    gaps = seg.straights
    assert len(corner_sectors(tr, merge_gap=gaps.min() - 1.0)) == len(seg.corners)
    merged = corner_sectors(tr, merge_gap=gaps.max() + 3.0)
    assert len(merged) == 1 and merged[0].s_end - merged[0].s_start == pytest.approx(tr.length)
    assert merged[0].angle == pytest.approx(2 * np.pi, abs=0.1)


# --------------------------------------------------------------------------- document


def test_document_round_trip(tmp_path):
    doc = make_doc()
    p = export.write_document(doc, tmp_path / "x.json")
    back = export.read_document(p)
    assert back["schema"] == export.SCHEMA and back["version"] == export.SCHEMA_VERSION
    assert [g["id"] for g in back["ghosts"]] == ["agent", "ocp"]
    comp = back["comparisons"][0]
    assert comp["lap_delta_s"] > 0 and comp["delta_t"][0] == 0.0
    assert comp["delta_t"][-1] == pytest.approx(comp["lap_delta_s"], abs=1e-4)
    assert sum(comp["sector_delta_s"]) == pytest.approx(comp["lap_delta_s"], abs=1e-3)
    g = export.ghost_arrays(back, "ocp")
    assert g["t"][-1] == pytest.approx(back["ghosts"][1]["lap_time_s"], abs=1e-4)
    assert "NaN" not in p.read_text()


def _broken(mutate):
    doc = json.loads(json.dumps(make_doc()))
    mutate(doc)
    return doc


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema="other"), "schema"),
    (lambda d: d.update(version="2.0.0"), "major version"),
    (lambda d: d["ghosts"][0]["channels"].pop("heading"), "heading"),
    (lambda d: d["ghosts"][0]["channels"]["speed"].pop(), "lengths differ"),
    (lambda d: d["ghosts"][0]["channels"]["t"].reverse(), "strictly increase"),
    (lambda d: d["ghosts"][0].update(lap_time_s=1.0), "lap_time_s"),
    (lambda d: d["ghosts"][0].update(kind="dream"), "kind"),
    (lambda d: d["ghosts"][1].update(id="agent"), "unique"),
    (lambda d: d.update(reference="nobody"), "reference"),
    (lambda d: d["ghosts"][0]["channels"]["x"].__setitem__(3, None), "numeric|non-finite"),
    (lambda d: d["comparisons"][0]["sector_delta_s"].pop(), "sector"),
])
def test_validation_rejects_broken_documents(mutate, match):
    with pytest.raises(export.SchemaError, match=match):
        export.validate_document(_broken(mutate))


def test_validation_accepts_additive_changes():
    doc = _broken(lambda d: d.update(version="1.7.0", future_key={"a": 1}))
    doc["ghosts"][0]["channels"]["new_channel"] = [0.0] * len(doc["ghosts"][0]["channels"]["t"])
    export.validate_document(doc)


def test_ghosts_need_required_channels():
    with pytest.raises(ValueError, match="required"):
        export.Ghost("g", "g", "policy", 1.0, {"t": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="kind"):
        export.Ghost("g", "g", "human", 1.0, {})


def test_lockbox_guard():
    for name in ("Monza", "Suzuka-mirrored", "Spa-reversed"):
        with pytest.raises(PermissionError):
            export.lockbox_guard(name)
        export.lockbox_guard(name, unsealed=True)
    for name in ("Hockenheim", "Catalunya-reversed", "square"):
        export.lockbox_guard(name)
