"""Renderers of a trajectory export: the matplotlib figures and the static viewer."""

import json
import re

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


def test_figures_leave_the_pyplot_state_alone():
    import matplotlib
    import matplotlib.pyplot as plt

    from apexzero.viz import figures

    backend = matplotlib.get_backend()
    before = plt.get_fignums()
    figures.sector_losses(make_doc())
    assert plt.get_fignums() == before and matplotlib.get_backend() == backend
