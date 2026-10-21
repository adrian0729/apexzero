"""Opt-in real widths: strip widths cut from the R-dev circuits' own per-side profiles.

Default behaviour must be bit-identical to the pre-option generator (checked against the
frozen copy ``interp.support.generate_with_corners``); with the option on, the profiles are
the R-dev circuits' widths as evaluated (from ``refine_reference``), each strip's widths are
an exact wrapped segment of one profile (before the inside cap), the cap holds, sides swap
symmetrically, strips stay valid, and train_dr.py runs with the flag."""

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
    DS,
    N_POINTS,
    _real_width_draw,
    _refuse_lockbox,
    fit_grammar,
    generate,
    generate_strip,
    out_of_distribution,
    real_width_segment,
    tilted,
    to_track,
    width_profiles,
)
from apexzero.tracks.reference import drivable_margin, refine_reference
from apexzero.tracks.track import load_tumftm
from apexzero.vehicle.sim import track_arrays

REPO = Path(__file__).resolve().parents[1]
TUMFTM = REPO / "data" / "tracks" / "tumftm"
pytestmark = pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
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
def evaluated(dev):
    """The R-dev circuits as every evaluation drives them (train_dr.py, eval_rdev.py)."""
    return [refine_reference(t) for t in dev]


@pytest.fixture(scope="module")
def grammars(dev):
    return fit_grammar(dev), fit_grammar(dev, real_widths=True)


def _find_segment(wl, wr, g):
    """Every (circuit, offset, swapped) whose wrapped profile segment equals (wl, wr) exactly."""
    PL, PR, lens = (np.asarray(a) for a in (g.real_w_left, g.real_w_right, g.real_len))
    hits = []
    for c, n in enumerate(lens):
        for swapped, (a, b) in ((False, (PL[c], PR[c])), (True, (PR[c], PL[c]))):
            win = (np.arange(n)[:, None] + np.arange(32)[None, :]) % n    # cheap pre-filter
            for o in np.flatnonzero((a[win] == wl[None, :32]).all(1) & (b[win] == wr[None, :32]).all(1)):
                idx = (o + np.arange(len(wl))) % n
                if np.array_equal(a[idx], wl) and np.array_equal(b[idx], wr):
                    hits.append((c, int(o), swapped))
    return hits


def test_default_is_unchanged(dev, grammars):
    from apexzero.interp.support import generate_with_corners  # frozen pre-option copy

    g0, g1 = grammars
    assert g0.real_w_left is None and g0.real_w_right is None and g0.real_len is None
    assert len(jax.tree.leaves(g0)) == N_BASE_FIELDS              # same pytree as before
    for f in g0._fields[:N_BASE_FIELDS]:                          # existing fields untouched
        a, b = np.asarray(getattr(g0, f)), np.asarray(getattr(g1, f))
        assert a.dtype == b.dtype and np.array_equal(a, b), f
    for gg in (g0, tilted(g0, alpha=2.0), out_of_distribution(g0)):
        for k in range(4):
            key = jax.random.PRNGKey(k)
            new = jax.jit(generate)(key, gg)
            old = jax.jit(lambda kk, gg=gg: generate_with_corners(kk, gg)[:3])(key)
            for x, y in zip(new, old, strict=True):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
            # Curvature is the same with real widths on.
            on = gg._replace(real_w_left=g1.real_w_left, real_w_right=g1.real_w_right,
                             real_len=g1.real_len)
            assert np.array_equal(np.asarray(new[0]), np.asarray(jax.jit(generate)(key, on)[0]))


def test_profiles_are_the_dev_circuits_widths(dev, evaluated, grammars):
    _, g1 = grammars
    lens = np.asarray(g1.real_len)
    assert g1.real_w_left.shape == g1.real_w_right.shape == (len(R_DEV), lens.max())
    assert g1.real_w_left.dtype == jnp.float32 and g1.real_len.dtype == jnp.int32
    for i, tr in enumerate(evaluated):
        n = lens[i]
        assert n == round(tr.length / DS)
        s = np.arange(n) * DS
        for prof, w in ((g1.real_w_left, tr.w_left), (g1.real_w_right, tr.w_right)):
            p = np.asarray(prof[i])
            assert np.array_equal(p[:n], np.interp(s, tr.s, w, period=tr.length).astype(np.float32))
            assert np.all(p[n:] == 0.0)
            # The 2 m grid misses sharp extremes by < 0.1 m (refined YasMarina: 0.064 m).
            assert abs(p[:n].min() - w.min()) < 0.1 and abs(p[:n].max() - w.max()) < 0.1
    # Lockbox names are refused (a renamed R-dev track, so no R_TEST data is read).
    with pytest.raises(PermissionError):
        width_profiles([dev[0], dataclasses.replace(dev[1], name="Monza")])
    with pytest.raises(ValueError):
        width_profiles([dataclasses.replace(dev[0], closed=False)])


def test_profiles_are_the_evaluated_geometry(dev, evaluated, grammars):
    """Review finding: the profiles must be the widths the agent is evaluated on, i.e.
    measured from refine_reference, not from the raw centreline (much more symmetric)."""
    _, g1 = grammars
    raw = width_profiles(dev)
    lens = np.asarray(g1.real_len)
    changed = [i for i, (t, r) in enumerate(zip(dev, evaluated, strict=True)) if r is not t]
    assert len(changed) == 6                       # the circuits refine_reference re-expresses
    for i in range(len(R_DEV)):
        same = all(np.array_equal(np.asarray(p[i, : lens[i]]), q[i, : raw[2][i]].astype(np.float32))
                   for p, q in ((g1.real_w_left, raw[0]), (g1.real_w_right, raw[1])))
        assert same == (i not in changed), R_DEV[i]
    i = R_DEV.index("Norisring")
    wl, wr = (np.asarray(p[i, : lens[i]]) for p in (g1.real_w_left, g1.real_w_right))
    assert min(wl.min(), wr.min()) < 3.2 and max(wl.max(), wr.max()) > 13.5   # raw: 4.5-11.1 m
    assert np.abs(wl - wr).max() > 7.0                                          # raw: 2.6 m
    # Already-refined input is used as is (refine_reference is not idempotent on Norisring).
    g_ref = fit_grammar(evaluated, real_widths=True)
    for f in ("real_w_left", "real_w_right", "real_len"):
        assert np.array_equal(np.asarray(getattr(g_ref, f)), np.asarray(getattr(g1, f))), f


def test_lockbox_backstop_matches_the_repo_guard(dev):
    """The generator's backstop refuses exactly the names ``viz.export.is_lockbox`` flags."""
    from apexzero.tracks.corners import R_TEST
    from apexzero.viz.export import is_lockbox

    names = list(R_DEV) + list(R_TEST) + ["monza", "Monza-mirrored", "Mexico City", "Spa_reversed",
                                          "SpaFrancorchamps", "IMS", "strip"]
    for name in names:
        refused = True
        try:
            _refuse_lockbox([dataclasses.replace(dev[0], name=name)])
            refused = False
        except PermissionError:
            pass
        assert refused == is_lockbox(name), name


def test_widths_are_an_exact_real_segment_before_the_cap(grammars):
    g0, g1 = grammars
    no_cap = g1._replace(margin=jnp.asarray(jnp.inf, jnp.float32))
    gen_raw = jax.jit(lambda k: generate(k, no_cap))
    gen = jax.jit(lambda k: generate(k, g1))
    swaps = []
    for k in range(24):
        key = jax.random.PRNGKey(100 + k)
        kap, wl_raw, wr_raw = (np.asarray(a) for a in gen_raw(key))
        hits = _find_segment(wl_raw, wr_raw, g1)
        ti, off, swap = (int(v) for v in _real_width_draw(jax.random.split(key, 10)[9], g1))
        assert (ti, off, bool(swap)) in hits
        swaps.append(bool(swap))
        assert np.array_equal(kap, np.asarray(jax.jit(generate)(key, g0)[0]))   # curvature unchanged
        # The capped strip is exactly the raw segment under today's inside cap.
        kap_c, wl, wr = gen(key)
        cap = 1.0 + g1.margin / jnp.maximum(jnp.abs(kap_c), 1e-6)
        assert np.array_equal(np.asarray(wl), np.asarray(jnp.where(kap_c > 0, jnp.minimum(wl_raw, cap), wl_raw)))
        assert np.array_equal(np.asarray(wr), np.asarray(jnp.where(kap_c < 0, jnp.minimum(wr_raw, cap), wr_raw)))
    assert any(swaps) and not all(swaps)
    # width_scale multiplies the segment (the OOD grammar's 0.65).
    ood = out_of_distribution(g1)
    key = jax.random.PRNGKey(7)
    a, b = real_width_segment(key, g1), real_width_segment(key, ood)
    for x, y in zip(a, b, strict=True):
        assert np.allclose(0.65 * np.asarray(x), np.asarray(y), rtol=1e-6)


def test_side_swap_is_symmetric(grammars):
    _, g1 = grammars
    keys = jax.random.split(jax.random.PRNGKey(5), 2000)
    ti, off, swap = jax.vmap(lambda k: _real_width_draw(k, g1))(keys)
    assert abs(float(np.mean(swap)) - 0.5) < 0.05
    assert np.all(np.bincount(np.asarray(ti), minlength=len(R_DEV)) > 100)    # every circuit drawn
    assert np.all(np.asarray(off) < np.asarray(g1.real_len)[np.asarray(ti)])
    # Exchanging the stored sides exchanges the strip's sides, key for key.
    mirror = g1._replace(real_w_left=g1.real_w_right, real_w_right=g1.real_w_left)
    for k in keys[:20]:
        wl, wr = real_width_segment(k, g1)
        ml, mr = real_width_segment(k, mirror)
        assert np.array_equal(np.asarray(wl), np.asarray(mr)) and np.array_equal(np.asarray(wr), np.asarray(ml))


def test_real_width_strips_are_valid(grammars):
    g0, g1 = grammars
    for gg in (g1, tilted(g1, alpha=2.0), out_of_distribution(g1)):
        kap, wl, wr = (np.asarray(a) for a in jax.jit(jax.vmap(lambda k, gg=gg: generate(k, gg)))(
            jax.random.split(jax.random.PRNGKey(2), 40)))
        inside = np.where(kap > 0, wl, np.where(kap < 0, wr, 1.0))
        assert np.all((inside - 1.0) * np.abs(kap) <= float(gg.margin) + 1e-5)     # the cap
        arr = jax.jit(jax.vmap(lambda k, gg=gg: generate_strip(k, gg)))(jax.random.split(jax.random.PRNGKey(2), 40))
        assert arr.x.shape == (40, N_POINTS)
        for i in range(0, 40, 4):
            tr = to_track(jax.tree.map(lambda x, i=i: x[i], arr))
            assert np.all(np.isfinite(tr.kappa)) and np.all(np.isfinite(tr.x))
            assert np.all(np.isfinite(tr.w_left)) and np.all(np.isfinite(tr.w_right))
            assert 1.0 / np.abs(tr.kappa).max() >= float(gg.r_floor) - 1e-3
            assert drivable_margin(tr) <= float(gg.margin) + 1e-3
            assert np.all(tr.w_left > 1.5) and np.all(tr.w_right > 1.5)
            ta = track_arrays(tr)
            assert all(np.all(np.isfinite(np.asarray(a))) for a in jax.tree.leaves(ta))
    # The point of the option: widths vary by metres within a strip.
    rng = lambda g: np.median(np.ptp(np.asarray(jax.vmap(lambda k: generate(k, g)[1])(
        jax.random.split(jax.random.PRNGKey(9), 40))), axis=1))
    assert rng(g1) > rng(g0) + 1.0


def test_real_width_bank_and_env_rollout(grammars, float32):
    from apexzero.env.adapters import generate_bank, procedural_env

    g0, g1 = grammars
    bank = generate_bank(jax.random.PRNGKey(3), g1, 16, tilted(g1, alpha=2.0), alt_fraction=0.5)
    assert np.all(np.isfinite(np.asarray(bank.v_ref)))
    plain = generate_bank(jax.random.PRNGKey(3), g0, 16, tilted(g0, alpha=2.0), alt_fraction=0.5)
    assert np.array_equal(np.asarray(bank.tracks.kappa_c), np.asarray(plain.tracks.kappa_c))
    assert not np.array_equal(np.asarray(bank.tracks.w_left_c), np.asarray(plain.tracks.w_left_c))
    # Mixing a real-width grammar with a default one is refused with a clear error.
    for a, b in ((g1, tilted(g0)), (g0, tilted(g1))):
        with pytest.raises(ValueError, match="real width profiles"):
            generate_bank(jax.random.PRNGKey(3), a, 4, b, alt_fraction=0.5)

    env = procedural_env(g1, bank_size=8, refresh_per_update=4, g_alt=tilted(g1), alt_fraction=0.5)
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


def test_train_dr_real_widths_smoke(tmp_path, monkeypatch, float32):
    """Two PPO updates at 8 envs on CPU, in a temporary runs/ (the R-dev lap evaluation at
    the end is stubbed: it does not depend on the option)."""
    scripts = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("train_dr_smoke", scripts / "train_dr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    seen = {}
    real_env = mod.procedural_env

    def spy(g, **kw):
        seen["g"], seen["g_alt"] = g, kw["g_alt"]
        return real_env(g, **kw)

    class Laps:
        completion_rate, median_lap = 0.0, float("nan")

    monkeypatch.setattr(mod, "procedural_env", spy)
    monkeypatch.setattr(mod, "timed_laps", lambda *a, **k: Laps())
    monkeypatch.setattr(mod, "refine_reference", lambda tr: tr)
    (tmp_path / "data").symlink_to(REPO / "data")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train_dr.py", "--real-widths", "--name", "smoke", "--steps", "128",
                                      "--num-envs", "8", "--num-steps", "8", "--bank", "8", "--refresh", "4",
                                      "--ensemble-heads", "2", "--tilt-fraction", "0.3"])
    mod.main()

    out = tmp_path / "runs" / "smoke"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["args"]["real_widths"] is True and cfg["ppo"]["num_envs"] == 8
    assert seen["g"].real_w_left is not None and seen["g_alt"].real_w_left is not None
    assert (out / "metrics.csv").read_text().count("\n") >= 2
    assert list((out / "ckpt").glob("*.pkl"))
    assert set(json.loads((out / "eval_rdev.json").read_text())) == set(R_DEV)
