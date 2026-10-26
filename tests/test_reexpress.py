"""Model 2b (GW-2): the JAX port of refine_reference for open strips (tracks/reexpress.py)
and its opt-in generator path.

Each piece of the port is checked against the numpy original it replaces (the open second
difference, the box QP, the trapezoid centreline, the ray search), and the whole step
against a numpy reference built from refine_reference's own pieces with the score taken on
the output part of the strip (numpy's rebase wraps its hint modulo n, which on an open
strip puts a spurious width of hundreds of metres at the last knot). The default path stays
bit-identical; the 2b path yields valid strips, runs in the bank and in train_dr.py."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.tracks import reexpress as R
from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import fit_grammar, generate, generate_strip, strip_arrays, tilted
from apexzero.tracks.reference import (
    _offsets_qp,
    _ray_distances,
    drivable_margin,
    rebase,
    second_difference,
)
from apexzero.tracks.track import Track, from_centreline, load_tumftm
from apexzero.tracks.widths import fit_width_model

REPO = Path(__file__).resolve().parents[1]
TUMFTM = REPO / "data" / "tracks" / "tumftm"
needs_tracks = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
N = 4096


@pytest.fixture(autouse=True)
def float32():
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture(scope="module")
def dev():
    return [load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV]


@pytest.fixture(scope="module")
def g2b(dev):
    return fit_grammar(dev, stat_widths=True, reexpress=True)


def _raw(g, key, n=N + R.EXTRA_KNOTS):
    raw = g._replace(reexpress_lams=None, margin=jnp.full_like(g.margin, jnp.inf))
    return [np.asarray(a, float) for a in jax.jit(lambda k: generate(k, raw, n))(key)]


def _track(kap, wl, wr):
    th, dx, dy, cx, cy = (np.asarray(a, float) for a in R.centreline(jnp.asarray(kap, jnp.float32), 2.0))
    return Track("strip", np.arange(len(kap)) * 2.0, cx, cy, th, kap, wl, wr, closed=False)


def _margin(k, wl, wr):
    return float(np.max(np.clip(np.where(k > 0, wl, wr) - 1.0, 0, None) * np.abs(k)))


def _numpy_reference(tr, n_out=N):
    """refine_reference's ladder and choice from its own pieces, scored on [:n_out]."""
    base = _margin(tr.kappa[:n_out], tr.w_left[:n_out], tr.w_right[:n_out])
    if base <= R.TARGET:
        return -1, None
    best = (base, -1, None)
    for i, lam in enumerate(R.LAMS):
        a = _offsets_qp(tr, lam, R.INNER_MARGIN)
        xy = np.stack([tr.x - a * np.sin(tr.theta), tr.y + a * np.cos(tr.theta)], 1)
        ref = from_centreline("s", xy, np.ones(len(xy)), np.ones(len(xy)), ds=2.0, smoothing=0.01,
                              closed=False)
        c = rebase(tr, ref)
        k, wl, wr = c.kappa[:n_out], c.w_left[:n_out], c.w_right[:n_out]
        if len(k) < n_out or 1.0 / np.abs(k).max() < R.MIN_RADIUS:
            continue
        sc = _margin(k, wl, wr)
        if sc < best[0]:
            best = (sc, i, (k, wl, wr))
        if sc <= R.TARGET:
            break
    return best[1], best[2]


# --------------------------------------------------------------------------- the pieces


def test_second_difference_and_its_transpose():
    n = 9
    d2 = second_difference(n, closed=False).toarray()
    v = np.random.default_rng(0).normal(size=n).astype(np.float32)
    assert np.allclose(np.asarray(R._d2(jnp.asarray(v))), d2 @ v, atol=1e-6)
    assert np.allclose(np.asarray(R._d2t(jnp.asarray(v))), d2.T @ v, atol=1e-6)


def test_offsets_solve_the_box_qp():
    """FISTA reaches lsq_linear's optimum of _offsets_qp's problem on a curved strip."""
    rng = np.random.default_rng(1)
    n = 600
    kap = np.zeros(n)
    kap[200:260] = np.r_[np.linspace(0, 1 / 15, 30), np.linspace(1 / 15, 0, 30)]
    kap[400:440] = -np.r_[np.linspace(0, 1 / 30, 20), np.linspace(1 / 30, 0, 20)]
    wl, wr = 6 + rng.uniform(-1, 1, n), 6 + rng.uniform(-1, 1, n)
    tr = _track(kap, wl, wr)
    f32 = lambda a: jnp.asarray(a, jnp.float32)
    th, dx, dy, _, _ = R.centreline(f32(kap), 2.0)
    nx, ny = -jnp.sin(th), jnp.cos(th)
    z = jnp.zeros(1)
    d2x, d2y = jnp.concatenate([z, jnp.diff(dx), z]), jnp.concatenate([z, jnp.diff(dy), z])
    for lam in (3e-2, 1e-2):
        a_np = _offsets_qp(tr, lam, 3.0)
        a_j = np.asarray(R.offsets(d2x, d2y, nx, ny, f32(-wr + 3.0), f32(wl - 3.0), lam,
                                   jnp.zeros(n, jnp.float32), 3000), float)

        def obj(a, lam=lam):
            px, py = tr.x - a * np.sin(tr.theta), tr.y + a * np.cos(tr.theta)
            return np.sum(np.diff(px, 2) ** 2) + np.sum(np.diff(py, 2) ** 2) + lam * np.sum(a**2)

        assert obj(a_j) <= obj(a_np) * (1 + 1e-4)
        assert np.abs(a_j - a_np).max() < 0.02
        assert np.all(a_j >= -wr + 3.0 - 1e-5) and np.all(a_j <= wl - 3.0 + 1e-5)


def test_centreline_matches_strip_arrays():
    kap = np.zeros(300, np.float32)
    kap[100:160] = 1 / 40
    th, _, _, cx, cy = R.centreline(jnp.asarray(kap), 2.0)
    arr = strip_arrays(jnp.asarray(kap), jnp.ones(300), jnp.ones(300))
    for a, b in ((th, arr.theta), (cx, arr.x), (cy, arr.y)):
        assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_ray_distance_matches_numpy():
    rng = np.random.default_rng(2)
    m = 200
    s = np.linspace(0, 400, m)
    poly = np.stack([s, 5.0 + 0.5 * np.sin(s / 30)], 1)
    origins = np.stack([s[::4] + rng.uniform(-1, 1, 50), rng.uniform(-1, 1, 50)], 1)
    normals = np.tile([[0.0, 1.0]], (50, 1))
    hint = np.arange(0, m, 4)
    ref = _ray_distances(origins, normals, poly, hint, R.WINDOW, closed=False)
    got = R._ray_distance(*(jnp.asarray(a, jnp.float32) for a in (origins[:, 0], origins[:, 1], normals[:, 0],
                                                                  normals[:, 1], poly[:, 0], poly[:, 1])),
                          jnp.asarray(hint))
    assert np.allclose(np.asarray(got), ref, atol=1e-3)
    # A ray that misses every segment in the window falls back to the nearest vertex.
    far = R._ray_distance(jnp.asarray([1000.0]), jnp.asarray([0.0]), jnp.asarray([0.0]), jnp.asarray([1.0]),
                          jnp.asarray(poly[:, 0], jnp.float32), jnp.asarray(poly[:, 1], jnp.float32),
                          jnp.asarray([m - 1]))
    assert np.isclose(float(far[0]), np.hypot(*(poly[-1] - [1000.0, 0.0])), rtol=1e-4)


# ---------------------------------------------------------------- the step against numpy


@needs_tracks
@pytest.mark.parametrize("tilt, k", [(False, 4), (True, 5), (True, 8)])
def test_reexpress_matches_the_numpy_reference(g2b, tilt, k):
    g = tilted(g2b, alpha=2.0) if tilt else g2b
    kap, wl, wr = _raw(g, jax.random.PRNGKey(k))
    ch_np, res = _numpy_reference(_track(kap, wl, wr))
    out = jax.jit(lambda a, b, c: R.reexpress(a, b, c, jnp.asarray(R.LAMS, jnp.float32), N))(
        *(jnp.asarray(a, jnp.float32) for a in (kap, wl, wr)))
    kj, wlj, wrj, ch = (np.asarray(a, float) for a in out)
    assert ch_np >= 0 and int(ch) == ch_np                    # triggered, same rung
    k_np, wl_np, wr_np = res
    dw = np.abs(np.r_[wlj - wl_np, wrj - wr_np])
    assert np.quantile(dw, 0.99) < 0.1 and dw.max() < 0.25
    dk = np.abs(kj - k_np)
    assert np.quantile(dk, 0.99) < 0.005 and dk.max() < 0.02
    assert 1.0 / np.abs(kj).max() >= R.MIN_RADIUS


@needs_tracks
def test_untriggered_strip_is_kept(g2b):
    for k in range(6):
        kap, wl, wr = _raw(g2b, jax.random.PRNGKey(k))
        if _margin(kap[:N], wl[:N], wr[:N]) <= R.TARGET:
            break
    out = R.reexpress(*(jnp.asarray(a, jnp.float32) for a in (kap, wl, wr)),
                      jnp.asarray(R.LAMS, jnp.float32), N)
    assert int(out[3]) == -1
    for a, b in zip(out[:3], (kap, wl, wr), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b[:N], np.float32))


# ----------------------------------------------------------------- the generator path


@needs_tracks
def test_default_and_2a_unchanged_and_2b_fit(dev, g2b):
    g0 = fit_grammar(dev)
    assert g0.reexpress_lams is None and len(jax.tree.leaves(g0)) == 14
    g2a = fit_grammar(dev, stat_widths=True)
    assert g2a.reexpress_lams is None
    assert np.array_equal(np.asarray(g2b.reexpress_lams), np.asarray(R.LAMS, np.float32))
    ref = fit_width_model(dev)                                 # 2b fits the raw profiles
    for a, b in zip(jax.tree.leaves(g2b.width_model), jax.tree.leaves(ref), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError, match="needs stat_widths"):
        fit_grammar(dev[:2], reexpress=True)
    from apexzero.tracks.reference import refine_reference
    with pytest.raises(ValueError, match="raw centrelines"):
        fit_grammar(dev[:2], stat_widths=True, reexpress=True,
                    width_tracks=[refine_reference(dev[3])])          # Norisring, refined


@needs_tracks
def test_2b_strips_are_valid(g2b):
    for gg in (g2b, tilted(g2b, alpha=2.0)):
        kap, wl, wr = (np.asarray(a) for a in jax.jit(
            lambda ks, gg=gg: jax.lax.map(lambda k: generate(k, gg), ks))(jax.random.split(jax.random.PRNGKey(3), 6)))
        assert kap.shape == (6, N)
        inside = np.where(kap > 0, wl, np.where(kap < 0, wr, 1.0))
        assert np.all((inside - 1.0) * np.abs(kap) <= float(gg.margin) + 1e-5)     # the cap
        assert np.all(np.isfinite(kap)) and np.all(wl > 1.5) and np.all(wr > 1.5)
        assert np.all(np.abs(kap) <= 1.0 / R.MIN_RADIUS + 1e-4)
        arr = jax.jit(lambda k, gg=gg: generate_strip(k, gg))(jax.random.PRNGKey(3))
        assert all(np.all(np.isfinite(np.asarray(a))) for a in jax.tree.leaves(arr))


@needs_tracks
def test_2b_bank_batches_and_refuses_mixing(dev, g2b):
    from apexzero.env.adapters import generate_bank

    bank = generate_bank(jax.random.PRNGKey(3), g2b, 5, tilted(g2b, alpha=2.0), alt_fraction=0.5)
    assert np.all(np.isfinite(np.asarray(bank.v_ref))) and bank.v_ref.shape[0] == 5
    g2a = fit_grammar(dev, stat_widths=True)
    for a, b in ((g2b, tilted(g2a)), (g2a, tilted(g2b))):
        with pytest.raises(ValueError, match="re-express"):
            generate_bank(jax.random.PRNGKey(3), a, 4, b, alt_fraction=0.5)


@needs_tracks
def test_train_dr_reexpress_flags(tmp_path, monkeypatch):
    scripts = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("train_dr_2b", scripts / "train_dr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "bad", "--reexpress"])
    with pytest.raises(SystemExit, match="needs --stat-widths"):
        mod.main()
    seen = {}

    def spy_env(g, **kw):
        seen["g"], seen["g_alt"] = g, kw["g_alt"]
        raise SystemExit("stop after the grammar")

    monkeypatch.setattr(mod, "procedural_env", spy_env)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "ok", "--stat-widths", "--reexpress",
                                      "--stat-widths-from", ",".join(R_DEV[0::2])])
    with pytest.raises(SystemExit, match="stop after"):
        mod.main()
    assert seen["g"].reexpress_lams is not None and seen["g_alt"].reexpress_lams is not None
    assert seen["g"].width_model.median_width.shape == (6,)


def test_drivable_margin_port():
    kap = np.r_[np.zeros(10), np.full(10, 0.1), np.full(10, -0.05)].astype(np.float32)
    wl = np.full(30, 7.0, np.float32)
    wr = np.full(30, 9.0, np.float32)
    tr = Track("t", np.arange(30) * 2.0, *(np.zeros(30),) * 3, kap, wl, wr, closed=False)
    assert np.isclose(float(R.drivable_margin(jnp.asarray(kap), jnp.asarray(wl), jnp.asarray(wr))),
                      drivable_margin(tr), rtol=1e-6)
