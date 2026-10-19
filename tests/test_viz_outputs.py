"""Renderers of a trajectory export: the matplotlib figures, the static viewer and the
optional Rerun recording."""

import json
import re

import pytest
from _viz_fixtures import make_doc

from apexzero.viz import export


def test_viewer_pins_three_and_embeds_the_document(tmp_path):
    html = export.VIEWER_PATH.read_text()
    # Every CDN import names one exact version.
    urls = re.findall(r"https://cdn\.jsdelivr\.net/npm/three@([^/]+)/", html)
    assert urls and all(re.fullmatch(r"\d+\.\d+\.\d+", v) for v in urls) and len(set(urls)) == 1
    doc = make_doc()
    out = export.embed_in_viewer(doc, tmp_path / "v.html").read_text()
    m = re.search(r'<script type="application/json" id="apexzero-data">(.*?)</script>', out,
                  re.DOTALL)
    assert m, "embedded data block missing"
    assert "</script> inside" not in m.group(1)              # cannot close the tag early
    back = json.loads(m.group(1))
    export.validate_document(back)
    assert back["notes"] == doc["notes"]


def test_figures_render(tmp_path):
    from apexzero.viz import figures

    paths = figures.save_figures(make_doc(), tmp_path, "demo", dpi=60)
    assert [p.name for p in paths] == ["demo-map.png", "demo-speed-delta.png", "demo-sectors.png"]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 5_000


def test_an_uncertified_optimum_is_labelled_so():
    from apexzero.viz import figures

    doc = make_doc()
    assert figures._label(doc, "ocp").endswith("not certified")
    assert "not certified" not in figures._label(doc, "agent")
    doc["ghosts"][1]["badge"] = "CERTIFIED"
    assert figures._label(doc, "ocp").endswith("CERTIFIED")
    # The viewer tags it too, and reads the carry-over split when the export has one.
    html = export.VIEWER_PATH.read_text()
    assert "not certified" in html and "sector_split" in html and "sector_speed" in html


def test_sector_figure_bars_are_the_charged_costs():
    from apexzero.viz import figures

    doc = make_doc()
    comp = doc["comparisons"][0]
    comp["sector_split"]["charged_s"] = [0.0] * len(comp["sector_delta_s"])
    comp["sector_split"]["charged_s"][0] = comp["lap_delta_s"]
    fig = figures.sector_losses(doc)
    filled, outline = fig.axes[0].containers          # charged bars, then measured outlines
    assert [p.get_height() for p in filled] == pytest.approx(comp["sector_split"]["charged_s"])
    assert [p.get_height() for p in outline] == pytest.approx(comp["sector_delta_s"])
    assert all(p.get_facecolor()[3] == 0.0 for p in outline)


def test_rerun_log(tmp_path):
    pytest.importorskip("rerun")   # optional dependency group: uv sync --group rerun
    from apexzero.viz.rerun_log import log_document

    p = log_document(make_doc(), tmp_path / "demo.rrd")
    assert p.read_bytes()[:4] == b"RRF2" and p.stat().st_size > 10_000


def test_figures_leave_the_pyplot_state_alone():
    import matplotlib
    import matplotlib.pyplot as plt

    from apexzero.viz import figures

    backend = matplotlib.get_backend()
    before = plt.get_fignums()
    figures.sector_losses(make_doc())
    assert plt.get_fignums() == before and matplotlib.get_backend() == backend
