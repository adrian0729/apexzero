"""scripts/train_ued.py end to end on a tiny pool, and its pieces: the shared checkpoint
schedule and naming, the oracle-label checks, and the on-disk admission log."""

import csv
import importlib.util
import pickle
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

from apexzero.agents.checkpoints import checkpoint_schedule, list_checkpoints
from apexzero.audit.rollouts import snapshot_from_pickle
from apexzero.tracks import levels as paper_levels
from apexzero.ued import pool
from apexzero.ued.plr import OP_INSERT, OP_UPDATE, OUT_NONE

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
TUMFTM = Path("data/tracks/tumftm")


def _launcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))                 # for ``import _bootstrap``
    spec = importlib.util.spec_from_file_location("train_ued", SCRIPTS / "train_ued.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def float32():
    """Some test modules switch JAX to float64 when imported; training runs in float32."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    yield
    jax.config.update("jax_enable_x64", old)


def test_checkpoint_schedule_for_curriculum_c():
    batch = 2048 * 128
    at = checkpoint_schedule(1e9, batch, 10e6, 8)
    assert len(at) == 8 and sorted(at) == list(at)
    assert at[min(at)] == 10_000_000 and min(at) == 10_000_000 // batch - 1
    last = int(1e9) // batch
    assert at[last - 1] == last * batch                     # the final update, by timesteps done
    for u, s in at.items():                                 # saved after the last update not past s
        assert (u + 1) * batch <= s < (u + 2) * batch or u == last - 1
    # A short run keeps distinct targets and always ends on the final update.
    assert checkpoint_schedule(61_440, 2048, 2048 * 3, 3) == {2: 6144, 8: 19429, 29: 61440}
    assert checkpoint_schedule(61_440, 2048, 10e6, 8) == {29: 61440}


def test_plr_oracle_needs_the_frozen_pool_and_its_own_labels(tmp_path):
    mod = _launcher()
    with pytest.raises(SystemExit):
        mod.parse_args(["--arm", "plr-oracle", "--name", "x", "--labels", str(tmp_path / "j.npz")])
    with pytest.raises(SystemExit):
        mod.parse_args(["--arm", "plr-oracle", "--name", "x", "--pool-file", str(tmp_path / "p.npz")])

    lv = paper_levels.make_levels(jax.random.PRNGKey(0), "pool", n=12)
    other = paper_levels.make_levels(jax.random.PRNGKey(1), "pool", n=12)   # same size, other levels
    j = np.linspace(10.0, 20.0, 12)
    pool.save_oracle_labels(tmp_path / "j.npz", lv, j)
    paper_levels.save(lv, tmp_path / "pool.npz")
    assert np.array_equal(pool.load_oracle_labels(tmp_path / "j.npz", paper_levels.load(tmp_path / "pool.npz")), j)
    with pytest.raises(ValueError, match="different pool"):
        pool.load_oracle_labels(tmp_path / "j.npz", other)
    shuffled = lv._replace(keys=lv.keys[::-1], ctx=lv.ctx[::-1], hidden_mu=lv.hidden_mu[::-1])
    with pytest.raises(ValueError, match="different pool"):
        pool.load_oracle_labels(tmp_path / "j.npz", shuffled)
    np.save(tmp_path / "bare.npy", j)
    np.savez(tmp_path / "bare.npz", jstar=j)
    with pytest.raises(ValueError, match="need"):
        pool.load_oracle_labels(tmp_path / "bare.npz", lv)
    with pytest.raises(ValueError):
        pool.save_oracle_labels(tmp_path / "short.npz", lv, j[:5])


@pytest.mark.skipif(not TUMFTM.exists(), reason="run scripts/fetch_tracks.py")
def test_launcher_writes_checkpoints_snapshots_records_and_the_admission_log(tmp_path, float32):
    mod = _launcher()
    envs, steps, chunks = 16, 8, 6
    batch = envs * steps
    argv = ["--arm", "plr-maxmc", "--name", "tiny", "--runs-dir", str(tmp_path), "--steps", str(batch * chunks),
            "--num-envs", str(envs), "--num-steps", str(steps), "--pool-size", "16", "--window-steps", "12",
            "--capacity", "8", "--width", "16", "--ensemble-heads", "2", "--checkpoints", "3",
            "--checkpoint-first", str(2 * batch), "--records-every", "4"]
    out = mod.main(argv)
    sched = checkpoint_schedule(batch * chunks, batch, 2 * batch, 3)
    assert len(sched) == 3

    # Checkpoints: the shared format and naming, loadable by the audit.
    files = list_checkpoints(out)
    assert [int(f.stem) for f in files] == sorted(sched.values())
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        assert d["timesteps"] == int(f.stem) and d["arm"] == "plr-maxmc"
        assert d["timesteps_done"] <= d["timesteps"] < d["timesteps_done"] + batch
        assert float(snapshot_from_pickle(d).sigma) > 0
        with open(out / "sampler" / f.name, "rb") as fh:
            s = pickle.load(fh)
        assert s["timesteps"] == d["timesteps"] and np.all(np.isfinite(s["scores"]))

    # Metrics: one row per chunk.
    with open(out / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == chunks and "replay_steps_dropped" in rows[0] and "cpu_s" in rows[0]

    # Records and the admission log, two shards (chunks 0-3 and 4-5).
    shards = sorted((out / "records").glob("*.npz"))
    assert [s.stem for s in shards] == ["0000000", "0000004"]
    data = [dict(np.load(s)) for s in shards]
    rec = {k: np.concatenate([d[k] for d in data]) for k in data[0] if not k.startswith("ev_")}
    ev = {k: np.concatenate([d[k] for d in data]) for k in data[0] if k.startswith("ev_")}
    assert rec["slot"].size > 0 and ev["ev_row"].size > 0
    assert set(ev) >= {"ev_chunk", "ev_row", "ev_op", "ev_level_id", "ev_outcome", "ev_index", "ev_score",
                       "ev_max_return", "ev_steps"}
    key = {(c, r): i for i, (c, r) in enumerate(zip(ev["ev_chunk"], ev["ev_row"], strict=True))}
    assert len(key) == ev["ev_row"].size                        # (chunk, row) identifies an event
    hit = np.array([key[(c, r)] for c, r in zip(rec["chunk"], rec["write_row"], strict=True)])
    assert np.all(np.isin(ev["ev_op"][hit], (OP_INSERT, OP_UPDATE)))
    assert np.array_equal(rec["write_outcome"], ev["ev_outcome"][hit]) and np.all(rec["write_outcome"] != OUT_NONE)
    assert np.allclose(rec["write_score"], ev["ev_score"][hit])
    assert np.array_equal(rec["level"], ev["ev_level_id"][hit])
    writes = np.isin(ev["ev_op"], (OP_INSERT, OP_UPDATE))
    assert np.all(np.isfinite(ev["ev_score"][writes]))
    assert ev["ev_steps"].sum() == rec["T"].sum()
