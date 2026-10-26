"""Opt-in statistical widths (GW-2, model 2a): ``tracks/widths.py`` and its generator hook.

Default behaviour must stay bit-identical (the frozen pre-option copy
``interp.support.generate_with_corners``, the default grammar's pytree, the default
train_dr.py path); with the option on, curvature is unchanged, the model stores no profile
(no-copy, structural), the fit recovers planted corner terms and band spectra on synthetic
circuits, the sampler reproduces its band SDs and quantiles, strips are valid under the
inside cap, the bank, env and train_dr.py run with it, and lockbox circuits are refused."""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from apexzero.tracks.corners import R_DEV
from apexzero.tracks.generator import (
    N_POINTS,
    fit_grammar,
    generate,
    generate_strip,
    out_of_distribution,
    tilted,
    to_track,
)
from apexzero.tracks.reference import drivable_margin, refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.tracks.widths import (
    BAND_EDGES,
    N_BANDS,
    N_RADIUS,
    SIDE_MAX,
    SIDE_MIN,
    WidthModel,
    band_noise,
    band_variance,
    corner_terms,
    fit_width_model,
    max_leaf_size,
    stat_widths,
    two_piece,
)
from apexzero.vehicle.sim import track_arrays

REPO = Path(__file__).resolve().parents[1]
TUMFTM = REPO / "data" / "tracks" / "tumftm"
needs_tracks = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
N_BASE_FIELDS = 14


@pytest.fixture
def float32():
    """Some test modules switch JAX to float64 when imported; training runs in float32."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.fixture(scope="module")
def dev():
    return [load_tumftm(TUMFTM / f"{n}.csv") for n in R_DEV]


@pytest.fixture(scope="module")
def grammars(dev):
    return fit_grammar(dev), fit_grammar(dev, stat_widths=True)


# ------------------------------------------------------------------ synthetic circuits


def _synthetic(n=2000, corners=((300, 40, 1 / 30), (900, 60, -1 / 80), (1500, 30, 1 / 20)),
               base=(6.0, 5.0)):
    """A closed profile with triangular corners (start knot, half-length in knots, peak
    kappa) and flat widths."""
    kappa = np.zeros(n)
    for start, half, kp in corners:
        kappa[start:start + 2 * half] = kp * np.r_[np.linspace(0, 1, half), np.linspace(1, 0, half)]
    return kappa, np.full(n, base[0]), np.full(n, base[1])


def _plant(kappa, wl, wr, a_in, a_out, ds=2.0):
    """Add the corner terms on exactly the knots and shape ``corner_terms`` fits."""
    from apexzero.tracks.corners import segment, smooth
    from apexzero.tracks.widths import profile_track

    ks = smooth(kappa, ds, 10.0, True)
    wl, wr = wl.copy(), wr.copy()
    for c in segment(profile_track(kappa, wl, wr, True, ds)).corners:
        a, b = round(c.s_start / ds), round(c.s_end / ds)
        idx = np.arange(a, b + 1) % len(kappa)
        g = np.abs(ks[idx]) / np.abs(ks[idx]).max()
        if c.sign > 0:
            wl[idx] += a_in * g
            wr[idx] += a_out * g
        else:
            wr[idx] += a_in * g
            wl[idx] += a_out * g
    return wl, wr


def test_corner_terms_recover_planted_amplitudes():
    kappa, wl, wr = _synthetic()
    wl2, wr2 = _plant(kappa, wl, wr, a_in=-2.0, a_out=1.5)
    rows, comp_l, comp_r = corner_terms(kappa, wl2, wr2)
    assert rows.shape == (3, 3)
    # delta = (a_out - a_in) / 2 = 1.75, tau = a_in + a_out = -0.5, at every corner.
    assert np.allclose(rows[:, 1], 1.75, atol=1e-9) and np.allclose(rows[:, 2], -0.5, atol=1e-9)
    assert np.allclose(np.sort(rows[:, 0]), [20.0, 30.0, 80.0], rtol=0.15)   # r_min, smoothed
    # Removing the fitted terms gives back the flat base exactly.
    assert np.allclose(wl2 - comp_l, wl) and np.allclose(wr2 - comp_r, wr)


def test_band_variance_and_band_noise_round_trip():
    n, ds = 4096, 2.0
    # A sinusoid of wavelength 100 m (64-128 m band) and amplitude 1: variance 0.5 there.
    s = np.arange(n) * ds
    x = np.sin(2 * np.pi * s / (n * ds / 82))       # 82 cycles: wavelength ~99.9 m
    v = band_variance(x, ds)
    band = int(np.searchsorted(BAND_EDGES, n * ds / 82, side="right") - 1)
    assert np.isclose(v[band], 0.5, rtol=1e-9) and np.isclose(v.sum(), 0.5, rtol=1e-9)
    # Band noise: the expected band variance is sd^2 (checked over many draws).
    sd = jnp.asarray(np.linspace(0.1, 1.0, N_BANDS), jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    xs = np.asarray(jax.vmap(lambda k: band_noise(k, sd, n, ds))(keys), float)
    got = np.mean([band_variance(x, ds) for x in xs], axis=0)
    # Bands with a few FFT bins (the longest) are noisier; all within 15%.
    assert np.allclose(got, np.asarray(sd, float) ** 2, rtol=0.15)
    assert np.allclose(xs.mean(axis=1), 0.0, atol=1e-5)


def test_two_piece_quantiles():
    q = jnp.asarray([[-3.0, -1.0, 0.5]])
    z = jax.random.normal(jax.random.PRNGKey(1), (200_000,))
    x = np.asarray(two_piece(z, q[0]))
    assert np.allclose(np.quantile(x, [0.1587, 0.5, 0.8413]), [-3.0, -1.0, 0.5], atol=0.03)


def test_fit_on_synthetic_circuits_and_no_profile_is_stored():
    from apexzero.tracks.widths import profile_track

    rng = np.random.default_rng(0)
    tracks = []
    for i in range(3):
        kappa, wl, wr = _synthetic()
        wl2, wr2 = _plant(kappa, wl + 0.1 * i, wr, a_in=-2.0, a_out=1.5)
        # Short-scale noise on both sides, the same on each circuit (variance 0.04 per side).
        e = rng.normal(0, 0.2, (2, len(kappa)))
        tr = profile_track(kappa, wl2 + e[0], wr2 + e[1], True, name=f"syn{i}")
        tracks.append(tr)
    wm = fit_width_model(tracks)
    assert isinstance(wm, WidthModel)
    assert wm.delta_q.shape == (N_RADIUS, 3) and wm.tau_q.shape == (N_RADIUS, 3)
    assert wm.common_sd.shape == (N_BANDS,) and wm.diff_sd.shape == (N_BANDS,)
    # Only 9 corners in all: every bin pools all of them, so every row is the pooled fit.
    assert np.allclose(np.asarray(wm.delta_q)[:, 1], 1.75, atol=0.1)
    assert np.allclose(np.asarray(wm.tau_q)[:, 1], -0.5, atol=0.15)
    # The residual is the planted white noise: common and differential each carry about
    # half the per-side variance (0.02 each), mostly in the shortest bands.
    assert np.isclose(float(np.sum(np.asarray(wm.common_sd) ** 2)), 0.02, rtol=0.25)
    assert np.isclose(float(np.sum(np.asarray(wm.diff_sd) ** 2)), 0.02, rtol=0.25)
    assert np.allclose(np.asarray(wm.median_width), [11.0, 11.1, 11.2], atol=0.1)
    # No-copy (structural): the largest array is a 5 x 3 table, far from a profile.
    assert max_leaf_size(wm) == 15


@needs_tracks
def test_lockbox_names_are_refused(dev):
    with pytest.raises(PermissionError):
        fit_width_model([refine_reference(dev[0]), dataclasses.replace(dev[1], name="Monza")])
    with pytest.raises(ValueError):
        fit_width_model([dataclasses.replace(dev[0], closed=False)])


# ------------------------------------------------------------- default stays identical


@needs_tracks
def test_default_is_bit_identical(dev, grammars, float32):
    from apexzero.interp.support import generate_with_corners  # frozen pre-option copy

    g0, g2 = grammars
    assert g0.width_model is None and g0.real_w_left is None
    assert len(jax.tree.leaves(g0)) == N_BASE_FIELDS                 # same pytree as before
    for f in g0._fields[:N_BASE_FIELDS]:                             # the other fields: equal
        a, b = np.asarray(getattr(g0, f)), np.asarray(getattr(g2, f))
        assert a.dtype == b.dtype and np.array_equal(a, b), f
    for gg in (g0, tilted(g0, alpha=2.0), out_of_distribution(g0)):
        on = gg._replace(width_model=g2.width_model)
        for k in range(8):
            key = jax.random.PRNGKey(k)
            new = jax.jit(generate)(key, gg)
            old = jax.jit(lambda kk, gg=gg: generate_with_corners(kk, gg)[:3])(key)
            for x, y in zip(new, old, strict=True):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
            # Curvature is the same with the width model on; the widths are not. Op by op it
            # is bit-identical; under jit, XLA fuses the shared corner layout differently, so
            # it agrees to float32 rounding (measured: at most 2.6e-7 relative).
            stat = jax.jit(generate)(key, on)
            assert np.allclose(np.asarray(stat[0]), np.asarray(new[0]), rtol=1e-6, atol=0.0)
            assert not np.array_equal(np.asarray(stat[1]), np.asarray(new[1]))
            if k == 0:
                eager_on, eager_off = generate(key, on), generate(key, gg)
                assert np.asarray(eager_on[0]).tobytes() == np.asarray(eager_off[0]).tobytes()


@needs_tracks
def test_default_bank_is_bit_identical(grammars, float32):
    """The default training bank (generate_bank with the tilted mixture) equals one built
    from the frozen copy, strip for strip."""
    from apexzero.env.adapters import _strip_arrays, generate_bank
    from apexzero.interp.support import generate_with_corners
    from apexzero.oracle.qss import speed_profile_jax
    from apexzero.vehicle.params import F1_2025

    g0, _ = grammars
    g_alt = tilted(g0, alpha=2.0)
    bank = generate_bank(jax.random.PRNGKey(3), g0, 8, g_alt, alt_fraction=0.5)

    def one(k):
        k, kc = jax.random.split(k)
        use_alt = jax.random.uniform(kc) < 0.5
        gg = jax.tree.map(lambda a, b: jnp.where(use_alt, b, a), g0, g_alt)
        kappa, wl, wr, _ = generate_with_corners(k, gg)
        return _strip_arrays(kappa, wl, wr), speed_profile_jax(kappa, 2.0, F1_2025)

    tracks, v = jax.vmap(one)(jax.random.split(jax.random.PRNGKey(3), 8))
    for a, b in zip(jax.tree.leaves((bank.tracks, bank.v_ref)), jax.tree.leaves((tracks, v)),
                    strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@needs_tracks
def test_options_are_exclusive(dev, grammars):
    from apexzero.env.adapters import generate_bank

    g0, g2 = grammars
    with pytest.raises(ValueError, match="alternatives"):
        fit_grammar(dev[:2], real_widths=True, stat_widths=True)
    g1 = fit_grammar(dev[:2], real_widths=True)
    with pytest.raises(ValueError, match="not both"):
        generate(jax.random.PRNGKey(0), g1._replace(width_model=g2.width_model))
    for a, b in ((g2, tilted(g0)), (g0, tilted(g2))):
        with pytest.raises(ValueError, match="width model"):
            generate_bank(jax.random.PRNGKey(3), a, 4, b, alt_fraction=0.5)


# ------------------------------------------------------------------ the option itself


@needs_tracks
def test_model_fitted_on_evaluated_profiles_and_subsets(dev, grammars):
    _, g2 = grammars
    wm = g2.width_model
    assert wm.median_width.shape == (len(R_DEV),)
    assert max_leaf_size(wm) == 15                       # no-copy: summaries only
    evaluated = [refine_reference(t) for t in dev]
    ref = fit_width_model(evaluated)
    for a, b in zip(jax.tree.leaves(wm), jax.tree.leaves(ref), strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    # width_tracks restricts the fit to a fold (leave-circuits-out); curvature uses all.
    fold = dev[0::2]
    sub = fit_grammar(dev, stat_widths=True, width_tracks=fold)
    assert sub.width_model.median_width.shape == (len(fold),)
    assert not np.array_equal(np.asarray(sub.width_model.delta_q), np.asarray(wm.delta_q))
    for f in ("corner_angle", "corner_rmin", "corner_len", "straight_len", "width"):
        assert np.array_equal(np.asarray(getattr(sub, f)), np.asarray(getattr(g2, f)))


def _corner_arrays(key, g):
    """The generator's own corner layout for a key (from the frozen copy's record)."""
    from apexzero.interp.support import generate_with_corners

    kappa, _, _, gc = generate_with_corners(key, g)
    return kappa, gc


@needs_tracks
def test_corner_coupling_follows_the_corner_profile(grammars, float32):
    """With the noise off and fixed corner terms, the widths before the cap are exactly
    level / 2 + (inside: -delta + tau / 2, outside: delta + tau / 2) x the trapezoid."""
    g0, g2 = grammars
    wm = g2.width_model._replace(
        median_width=jnp.asarray([12.0], jnp.float32), level_jitter=jnp.asarray(0.0, jnp.float32),
        common_sd=jnp.zeros(N_BANDS, jnp.float32), diff_sd=jnp.zeros(N_BANDS, jnp.float32),
        roughness=jnp.asarray(0.0, jnp.float32),
        delta_q=jnp.full((N_RADIUS, 3), 1.5, jnp.float32), tau_q=jnp.full((N_RADIUS, 3), 1.0, jnp.float32))
    on = g2._replace(width_model=wm, margin=jnp.asarray(jnp.inf, jnp.float32))   # no cap
    key = jax.random.PRNGKey(11)
    kappa, wl, wr = (np.asarray(a, float) for a in jax.jit(generate)(key, on))
    # On a corner knot, |kappa| / k_peak is the trapezoid; off corners it is 0.
    _, gc = _corner_arrays(key, g0)
    s = np.arange(N_POINTS) * 2.0
    j = np.clip(np.searchsorted(np.asarray(gc.start), s, side="right") - 1, 0, None)
    shape = np.where(kappa != 0, np.abs(kappa) / np.asarray(gc.k_peak)[j], 0.0)
    d_in, d_out = (-1.5 + 0.5) * shape, (1.5 + 0.5) * shape
    exp_l = np.clip(6.0 + np.where(kappa > 0, d_in, d_out), SIDE_MIN, SIDE_MAX)
    exp_r = np.clip(6.0 + np.where(kappa > 0, d_out, d_in), SIDE_MIN, SIDE_MAX)
    assert np.allclose(wl, exp_l, atol=1e-4) and np.allclose(wr, exp_r, atol=1e-4)
    assert wl.min() < 5.5 and wr.min() < 5.5 and max(wl.max(), wr.max()) > 7.5


@needs_tracks
def test_stat_width_strips_are_valid(grammars):
    g0, g2 = grammars
    for gg in (g2, tilted(g2, alpha=2.0), out_of_distribution(g2)):
        kap, wl, wr = (np.asarray(a) for a in jax.jit(jax.vmap(lambda k, gg=gg: generate(k, gg)))(
            jax.random.split(jax.random.PRNGKey(2), 40)))
        inside = np.where(kap > 0, wl, np.where(kap < 0, wr, 1.0))
        assert np.all((inside - 1.0) * np.abs(kap) <= float(gg.margin) + 1e-5)       # the cap
        scale = float(gg.width_scale)
        assert wl.min() >= SIDE_MIN * scale - 1e-5 and wl.max() <= SIDE_MAX * scale + 1e-5
        assert wr.min() >= SIDE_MIN * scale - 1e-5 and wr.max() <= SIDE_MAX * scale + 1e-5
        arr = jax.jit(jax.vmap(lambda k, gg=gg: generate_strip(k, gg)))(jax.random.split(jax.random.PRNGKey(2), 40))
        for i in range(0, 40, 4):
            tr = to_track(jax.tree.map(lambda x, i=i: x[i], arr))
            assert np.all(np.isfinite(tr.w_left)) and np.all(np.isfinite(tr.w_right))
            assert drivable_margin(tr) <= float(gg.margin) + 1e-3
            assert np.all(tr.w_left > 1.5) and np.all(tr.w_right > 1.5)
            assert all(np.all(np.isfinite(np.asarray(a))) for a in jax.tree.leaves(track_arrays(tr)))
    # Strips differ in width within themselves by more than the default model's.
    rng = lambda g: np.median(np.ptp(np.asarray(jax.vmap(lambda k: generate(k, g)[1])(
        jax.random.split(jax.random.PRNGKey(9), 40))), axis=1))
    assert rng(g2) > rng(g0)


@needs_tracks
def test_stat_width_bank_and_env_rollout(grammars, float32):
    from apexzero.env.adapters import generate_bank, procedural_env

    g0, g2 = grammars
    bank = generate_bank(jax.random.PRNGKey(3), g2, 16, tilted(g2, alpha=2.0), alt_fraction=0.5)
    plain = generate_bank(jax.random.PRNGKey(3), g0, 16, tilted(g0, alpha=2.0), alt_fraction=0.5)
    assert np.all(np.isfinite(np.asarray(bank.v_ref)))
    assert np.allclose(np.asarray(bank.tracks.kappa_c), np.asarray(plain.tracks.kappa_c), rtol=1e-5, atol=1e-9)
    assert not np.array_equal(np.asarray(bank.tracks.w_left_c), np.asarray(plain.tracks.w_left_c))

    env = procedural_env(g2, bank_size=8, refresh_per_update=4, g_alt=tilted(g2), alt_fraction=0.5)
    params = env.init_params(jax.random.PRNGKey(0))
    n = 6
    obs, st = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(1), n), params)

    @jax.jit
    def run(obs, st, params):
        def body(c, k):
            obs, st = c
            ka, ks = jax.random.split(k)
            a = jnp.clip(jax.random.normal(ka, (n, env.action_dim)) * 0.3 + jnp.array([0.3, 0.0]), -1, 1)
            obs, st, r, term, trunc, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(ks, n), st, a, params)
            return (obs, st), (r, term | trunc)

        return jax.lax.scan(body, (obs, st), jax.random.split(jax.random.PRNGKey(2), 200))

    (obs, st), (r, done) = run(obs, st, params)
    assert np.all(np.isfinite(np.asarray(obs))) and np.all(np.isfinite(np.asarray(r)))
    params2 = env.refresh(jax.random.PRNGKey(4), params, jnp.asarray(1))
    assert np.all(np.isfinite(np.asarray(params2.v_ref)))


def test_stat_widths_is_jittable_in_float64():
    """The sampler runs under x64 too (some analysis code enables it)."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        wm = WidthModel(jnp.asarray([12.0]), jnp.asarray(0.08), jnp.zeros((N_RADIUS, 3)),
                        jnp.zeros((N_RADIUS, 3)), jnp.full(N_BANDS, 0.1), jnp.full(N_BANDS, 0.1),
                        jnp.asarray(0.2))
        n, m = 256, 4
        out = jax.jit(lambda k: stat_widths(k, wm, jnp.asarray(1.0), jnp.zeros(n, jnp.int32),
                                            jnp.zeros(n), jnp.ones(m), jnp.full(m, 0.02)))(
            jax.random.PRNGKey(0))
        assert all(np.all(np.isfinite(np.asarray(o))) for o in out)
    finally:
        jax.config.update("jax_enable_x64", old)


# ----------------------------------------------------------------------- train_dr.py


def _load_train_dr(monkeypatch):
    scripts = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("train_dr_stat", scripts / "train_dr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_train_dr(mod, tmp_path, monkeypatch, extra):
    seen = {}
    real_env, real_fit = mod.procedural_env, mod.fit_grammar

    def spy_env(g, **kw):
        seen["g"], seen["g_alt"] = g, kw["g_alt"]
        return real_env(g, **kw)

    def spy_fit(*a, **kw):
        seen["fit_kwargs"] = kw
        return real_fit(*a, **kw)

    class Laps:
        completion_rate, median_lap = 0.0, float("nan")

    monkeypatch.setattr(mod, "procedural_env", spy_env)
    monkeypatch.setattr(mod, "fit_grammar", spy_fit)
    monkeypatch.setattr(mod, "timed_laps", lambda *a, **k: Laps())
    monkeypatch.setattr(mod, "refine_reference", lambda tr: tr)
    if not (tmp_path / "data").exists():
        (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "smoke", "--steps", "128",
                                      "--num-envs", "8", "--num-steps", "8", "--bank", "8",
                                      "--refresh", "4", "--ensemble-heads", "2", *extra])
    mod.main()
    return seen


@needs_tracks
def test_train_dr_default_path_is_unchanged(tmp_path, monkeypatch, float32):
    """With no width option, train_dr.py fits the grammar exactly as before (same call,
    no width model), so its bank is the default one."""
    mod = _load_train_dr(monkeypatch)
    seen = _run_train_dr(mod, tmp_path, monkeypatch, [])
    assert seen["fit_kwargs"] == {"real_widths": False, "width_tracks": None}
    assert seen["g"].width_model is None and seen["g"].real_w_left is None
    assert seen["g_alt"].width_model is None
    cfg = json.loads((tmp_path / "runs" / "smoke" / "config.json").read_text())
    assert cfg["args"]["stat_widths"] is False and cfg["args"]["stat_widths_from"] == ""


@needs_tracks
def test_train_dr_stat_widths_smoke(tmp_path, monkeypatch, float32):
    """Two PPO updates at 8 envs on CPU with the stat model fitted on one fold."""
    mod = _load_train_dr(monkeypatch)
    fold = R_DEV[0::2]
    seen = _run_train_dr(mod, tmp_path, monkeypatch,
                         ["--stat-widths", "--stat-widths-from", ",".join(fold), "--tilt-fraction", "0.3"])
    assert seen["fit_kwargs"]["stat_widths"] is True
    assert [t.name for t in seen["fit_kwargs"]["width_tracks"]] == fold
    assert seen["g"].width_model is not None and seen["g_alt"].width_model is not None
    assert seen["g"].width_model.median_width.shape == (len(fold),)
    out = tmp_path / "runs" / "smoke"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["args"]["stat_widths"] is True and cfg["args"]["stat_widths_from"] == ",".join(fold)
    assert (out / "metrics.csv").read_text().count("\n") >= 2
    assert list((out / "ckpt").glob("*.pkl"))
    assert set(json.loads((out / "eval_rdev.json").read_text())) == set(R_DEV)


@needs_tracks
@pytest.mark.parametrize("extra, msg", [
    (["--stat-widths-from", "Norisring"], "needs --stat-widths"),
    (["--stat-widths", "--stat-widths-from", "Monza"], "not R-dev"),
    (["--stat-widths", "--real-widths"], "alternatives"),
])
def test_train_dr_refuses_bad_width_flags(tmp_path, monkeypatch, extra, msg):
    mod = _load_train_dr(monkeypatch)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--name", "bad", *extra])
    with pytest.raises(SystemExit, match=msg):
        mod.main()
