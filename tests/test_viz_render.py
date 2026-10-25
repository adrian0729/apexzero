"""Render mode: frame timing, sampling and the live delta, the frames themselves (camera,
badges, attribution), the writers, and the lockbox guard."""

import copy
import json

import numpy as np
import pytest
from _viz_fixtures import make_doc
from PIL import Image, features

from apexzero.viz import export, render

TINY = dict(width=320, height=180, fps=2.0, rate=4.0, hold=1.0)


@pytest.fixture(scope="module")
def doc():
    return make_doc()


def texts(r: render.Renderer) -> list[str]:
    return [t.get_text() for t in r.fig.texts]


# --------------------------------------------------------------------------- timing


def test_frame_times_step_at_the_rate_and_end_exactly():
    t = render.frame_times(0.0, 10.5, fps=4.0, rate=2.0)
    np.testing.assert_allclose(np.diff(t)[:-1], 0.5)
    assert t[0] == 0.0 and t[-1] == 10.5 and len(t) == 22
    held = render.frame_times(1.0, 3.0, fps=10.0, hold=0.5)
    assert len(held) == 21 + 5 and np.all(held[-6:] == 3.0)
    with pytest.raises(ValueError):
        render.frame_times(2.0, 2.0, fps=30.0)
    with pytest.raises(ValueError):
        render.frame_times(0.0, 1.0, fps=0.0)


def test_sample_interpolates_in_time_and_parks_a_finished_car(doc):
    ch = export.ghost_arrays(doc, "agent")
    k = len(ch["t"]) // 3
    p = render.sample(ch, ch["t"][k])
    assert p["x"] == pytest.approx(ch["x"][k]) and p["s"] == pytest.approx(ch["s"][k])
    mid = render.sample(ch, 0.5 * (ch["t"][k] + ch["t"][k + 1]))
    assert mid["s"] == pytest.approx(0.5 * (ch["s"][k] + ch["s"][k + 1]))
    after = render.sample(ch, ch["t"][-1] + 5.0)
    assert after["s"] == ch["s"][-1] and after["x"] == ch["x"][-1]


def test_live_delta_follows_the_comparison_and_ends_at_the_lap_delta(doc):
    g, ref = export.ghost_arrays(doc, "agent"), export.ghost_arrays(doc, "ocp")
    comp = doc["comparisons"][0]
    for t in np.linspace(0.5, g["t"][-1] - 0.5, 7):
        s = np.interp(t, g["t"], g["s"])
        # The readout is the export's Δt(s) at the distance the ghost has reached.
        assert render.live_delta(g, ref, t) == pytest.approx(
            np.interp(s, comp["s"], comp["delta_t"]), abs=2e-3)
    # Exported times are rounded to 0.1 ms; lap_delta_s is not.
    assert render.live_delta(g, ref, g["t"][-1] + 3.0) == pytest.approx(
        comp["lap_delta_s"], abs=2e-4)
    assert render.live_delta(ref, ref, 2.0) == pytest.approx(0.0, abs=1e-9)
    assert render.live_delta(g, ref, 0.0) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- labels


def test_badges_tag_an_uncertified_optimum_and_show_the_badge(doc):
    ocp = next(g for g in doc["ghosts"] if g["id"] == "ocp")
    agent = next(g for g in doc["ghosts"] if g["id"] == "agent")
    assert render.badges(ocp) == ["NOT CERTIFIED"]
    assert render.badges(agent) == []
    assert render.badges(dict(agent, badge="POST HOC")) == ["POST HOC"]
    assert render.badges(dict(ocp, badge="CERTIFIED")) == ["CERTIFIED"]
    assert render.badges(dict(ocp, badge="MODEL-DEPENDENT")) == ["MODEL-DEPENDENT",
                                                                 "NOT CERTIFIED"]


def test_colours_match_the_figures(doc):
    from apexzero.viz.figures import SERIES

    assert render.ghost_colours(doc) == {"agent": SERIES[0], "ocp": SERIES[1]}


def test_attribution_comes_from_the_track_source_or_notes(doc):
    assert render.default_attribution(doc) is None
    d = copy.deepcopy(doc)
    d["notes"].append("Track geometry: TUMFTM racetrack-database (LGPL-3.0).")
    assert render.default_attribution(d) == render.TUMFTM_ATTRIBUTION


def test_still_time_tokens(doc):
    lap_agent = next(g["lap_time_s"] for g in doc["ghosts"] if g["id"] == "agent")
    assert render.still_time(doc, "end") == lap_agent       # the slower ghost finishes last
    assert render.still_time(doc, "12.5") == 12.5
    comp = doc["comparisons"][0]
    worst = doc["track"]["sectors"][int(np.argmax(comp["sector_split"]["charged_s"]))]
    t = render.still_time(doc, "worst")
    g = export.ghost_arrays(doc, "agent")
    assert np.interp(t, g["t"], g["s"]) == pytest.approx(worst["s_apex"], abs=0.05)
    with pytest.raises(ValueError, match="worst"):
        render.still_time(doc, "apex")


# --------------------------------------------------------------------------- frames


def test_frames_have_the_requested_size_and_are_deterministic(doc):
    r = render.Renderer(doc, render.RenderSpec(width=320, height=180))
    a, b = r.frame(3.0), r.frame(3.0)
    assert a.shape == (180, 320, 3) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, r.frame(6.0))
    odd = render.Renderer(doc, render.RenderSpec(width=333, height=211)).frame(1.0)
    assert odd.shape == (211, 333, 3)


def test_follow_camera_centres_on_the_followed_ghost(doc):
    r = render.Renderer(doc, render.RenderSpec(width=320, height=180, window_m=120.0))
    assert r.follow == "agent"
    r.frame(4.0)
    p = render.sample(r.ch["agent"], 4.0)
    (x0, x1), (y0, y1) = r.ax.get_xlim(), r.ax.get_ylim()
    assert 0.5 * (x0 + x1) == pytest.approx(p["x"]) and 0.5 * (y0 + y1) == pytest.approx(p["y"])
    assert y1 - y0 == pytest.approx(120.0)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(r.aspect)
    r2 = render.Renderer(doc, render.RenderSpec(width=320, height=180, follow="ocp"))
    r2.frame(4.0)
    q = render.sample(r2.ch["ocp"], 4.0)
    assert 0.5 * sum(r2.ax.get_xlim()) == pytest.approx(q["x"])
    with pytest.raises(KeyError):
        render.Renderer(doc, render.RenderSpec(follow="nobody"))


def test_full_camera_shows_the_whole_circuit(doc):
    r = render.Renderer(doc, render.RenderSpec(width=320, height=180, camera="full"))
    r.frame(2.0)
    (x0, x1), (y0, y1) = r.ax.get_xlim(), r.ax.get_ylim()
    lo, hi = r.box
    assert x0 < lo[0] and x1 > hi[0] and y0 < lo[1] and y1 > hi[1]


def test_hud_shows_badges_delta_and_attribution(doc):
    d = copy.deepcopy(doc)
    next(g for g in d["ghosts"] if g["id"] == "agent")["badge"] = "POST HOC"
    r = render.Renderer(d, render.RenderSpec(width=320, height=180, attribution="Licence X"))
    end = render.still_time(d, "end")
    r.frame(end)
    shown = texts(r)
    assert "POST HOC" in shown and "NOT CERTIFIED" in shown and "Licence X" in shown
    lap = d["comparisons"][0]["lap_delta_s"]
    assert f"Δt {lap:+.3f} s" in shown
    laps = {g["id"]: g["lap_time_s"] for g in d["ghosts"]}
    assert f"lap {laps['agent']:.3f} s" in shown and f"lap {laps['ocp']:.3f} s" in shown
    r.frame(1.0)
    assert any(s.endswith("km/h") for s in texts(r))


def test_a_document_without_a_reference_renders_without_a_delta(doc):
    d = copy.deepcopy(doc)
    d["reference"], d["comparisons"] = None, []
    r = render.Renderer(d, render.RenderSpec(width=320, height=180))
    assert r.frame(1.0).shape == (180, 320, 3)
    assert r.delta is None and not any(s.startswith("Δt") for s in texts(r))


def test_render_refuses_lockbox_circuits_and_invalid_documents(doc):
    d = copy.deepcopy(doc)
    d["track"]["name"] = "Monza-reversed"
    with pytest.raises(PermissionError):
        render.Renderer(d)
    bad = copy.deepcopy(doc)
    bad["ghosts"][0]["channels"]["t"][1] = -1.0
    with pytest.raises(export.SchemaError):
        render.Renderer(bad)


def test_spec_rejects_bad_settings():
    for kw in (dict(fps=0), dict(rate=-1), dict(camera="chase"), dict(width=10),
               dict(window_m=0), dict(hold=-1)):
        with pytest.raises(ValueError):
            render.RenderSpec(**kw)


# --------------------------------------------------------------------------- writers


def test_write_frames_writes_the_sequence_and_a_manifest(doc, tmp_path):
    spec = render.RenderSpec(**TINY)
    man = json.loads(render.write_frames(doc, tmp_path / "f", spec).read_text())
    frames = sorted((tmp_path / "f").glob("frame_*.png"))
    expected = render.Renderer(doc, spec).times()
    assert len(frames) == man["frames"] == len(expected)
    np.testing.assert_allclose(man["lap_times_s"], np.round(expected, 4))
    assert man["duration_s"] == pytest.approx(len(expected) / 2.0)
    assert man["ghosts"]["ocp"]["badges"] == ["NOT CERTIFIED"]
    with Image.open(frames[0]) as im:
        assert im.size == (320, 180)
    with pytest.raises(FileExistsError):
        render.write_frames(doc, tmp_path / "f", spec)
    shorter = render.RenderSpec(**dict(TINY, hold=0.0))
    render.write_frames(doc, tmp_path / "f", shorter, overwrite=True)
    assert len(list((tmp_path / "f").glob("frame_*.png"))) == len(expected) - 2


@pytest.mark.parametrize("suffix", [".gif", ".webp", ".png"])
def test_write_animation_keeps_the_frame_count_and_clock(doc, tmp_path, suffix):

    if suffix == ".webp" and not features.check("webp"):
        pytest.skip("Pillow built without WebP")
    spec = render.RenderSpec(**TINY)
    n = len(render.Renderer(doc, spec).times())
    held = round(spec.hold * spec.fps)
    path = render.write_animation(doc, tmp_path / f"clip{suffix}", spec)
    with Image.open(path) as im:
        # Encoders may merge the identical held frames into one longer frame.
        assert n - held <= im.n_frames <= n
        total = 0
        for k in range(im.n_frames):
            im.seek(k)
            im.load()                   # WebP sets the duration when the frame loads
            total += im.info["duration"]
    assert total == pytest.approx(1000.0 * n / spec.fps, abs=10)


def test_gif_refuses_a_frame_rate_it_cannot_time(doc, tmp_path):
    with pytest.raises(ValueError, match="1/100"):
        render.write_animation(doc, tmp_path / "c.gif", render.RenderSpec(**dict(TINY, fps=30)))
    with pytest.raises(ValueError, match="unsupported"):
        render.write_animation(doc, tmp_path / "c.avi", render.RenderSpec(**TINY))


def test_animations_refuse_to_exhaust_memory(doc, tmp_path, monkeypatch):
    monkeypatch.setattr(render, "PIL_MEMORY_LIMIT", 1e6)
    with pytest.raises(ValueError, match="GB of frames"):
        render.write_animation(doc, tmp_path / "c.webp", render.RenderSpec(**TINY))


def test_mp4_without_ffmpeg_says_so(doc, tmp_path, monkeypatch):
    monkeypatch.setattr(render.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        render.write_animation(doc, tmp_path / "c.mp4", render.RenderSpec(**TINY))


def test_write_still_matches_the_frame(doc, tmp_path):
    spec = render.RenderSpec(width=320, height=180)
    path = render.write_still(doc, 5.0, tmp_path / "still.png", spec)
    with Image.open(path) as im:
        np.testing.assert_array_equal(np.asarray(im.convert("RGB")),
                                      render.Renderer(doc, spec).frame(5.0))
