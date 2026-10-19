"""Renderers of a trajectory export: the matplotlib figures."""

from _viz_fixtures import make_doc


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
