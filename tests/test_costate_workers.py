"""The worker pools of the long stages (``apexzero.costate.workers``; scripts/costate.py's
probes, own, nulls and decoy-k, ``POOLED``).

- ``workers.run``: the results in the units' order whatever order they finish in; every
  worker shares its parent's thread pins, x64 setting and trained-network locks and gets
  the parent's state bit for bit; one worker is the serial path, in this process; a unit
  that raises, or a worker that dies, fails the run promptly and returns nothing; a script's
  function reaches a worker by its file and name.
- Each pooled stage through ``main()`` in a fresh process (its thread pins set before numpy
  loads, as a real run sets them) at reduced size, with --workers 1 and --workers 3: the
  outputs are identical (every JSON output's content_sha256, recomputed and as stamped, and
  every npz byte for byte and array for array), and the stamps differ in the pool only
  ('workers' and argv; the thread pins and partial options the same). decoy-k on a synthetic
  probe world (two candidates, two seeds: 8 units), probes on a toy T with toy activations
  for three networks, the nulls on a world of 12 stored keys under three synthetic obs_rms,
  own on a world of two stored keys with three random-init stand-ins (G1's config and
  obs_rms, random params). The stage's parent-side inputs (its probe world, the record, the
  world,
  the checkpoints) are patched in that process only, as the rehearsal patches its reduced
  lists, so every worker computes from what its parent hands it. A unit that raises in a
  worker (a network's activations without its h3, an obs_rms of the wrong size) fails the
  stage, and no output is written, serial or pooled.
- The options: --workers on the pooled stages only, at least 1, never partial; the default
  pool; ``savez``'s bytes depend on the arrays alone.
"""

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apexzero.costate import data_dirs
from apexzero.costate import workers as WK

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "costate.py"
KEYS, STIM = data_dirs()
HAVE_KEYS = (KEYS / "nominal").exists() and (STIM / "cells.json").exists()
VOLATILE = ("start", "end", "argv", "workers")     # a stamp's fields a rerun may change


def _script():
    spec = importlib.util.spec_from_file_location("costate_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return mod


def _have_g1() -> bool:
    from apexzero.interp import agents

    try:
        return agents.final_snapshot(agents.AGENTS["G1"]) is not None
    except (KeyError, OSError):
        return False


# ------------------------------------------------------------------------- workers.run


def _probe(state, unit):
    """A unit that reports the worker it ran in (module level: a worker loads this file)."""
    time.sleep(unit.get("sleep", 0.0))
    if unit.get("raise"):
        raise ValueError(f"unit {unit['i']} raised")
    if unit.get("die"):
        os._exit(3)
    return {"i": unit["i"], "pid": os.getpid(), "state": state, "env": WK.thread_env(),
            "x64": WK._x64_now(), "locks": WK._locks_now()}


class _Unloadable:
    """Pickles here, raises when a worker unpickles it."""

    def __reduce__(self):
        return (int, ("not a number",))


def test_run_gathers_in_the_units_order_and_each_worker_mirrors_its_parent():
    state = {"a": np.random.default_rng(0).normal(size=(50, 3)), "k": 7}
    units = [{"i": i, "sleep": s} for i, s in enumerate((3.0, 1.0, 0.0, 0.2))]
    got = WK.run(_probe, units, 3, state=state)
    assert [g["i"] for g in got] == [0, 1, 2, 3]
    pids = {g["pid"] for g in got}
    assert os.getpid() not in pids and 2 <= len(pids) <= 3
    for g in got:
        assert np.array_equal(g["state"]["a"], state["a"]) and g["state"]["k"] == 7
        assert g["env"] == WK.thread_env()
        assert g["x64"] == WK._x64_now() and g["locks"] == WK._locks_now()
    # one worker, or one unit: the serial path, in this process (the state itself)
    for w, us in ((1, units), (3, units[:1])):
        got = WK.run(_probe, us, w, state=state)
        assert {g["pid"] for g in got} == {os.getpid()} and got[0]["state"] is state
    assert WK.run(_probe, [], 3) == []


def test_a_unit_that_raises_or_a_worker_that_dies_fails_the_run_promptly(tmp_path,
                                                                         monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))   # where the state's file goes
    units = [{"i": 0}, {"i": 1, "sleep": 0.5, "raise": True}, {"i": 2, "sleep": 120.0}]
    t0 = time.time()
    with pytest.raises(WK.WorkerFailed, match="unit b failed in a worker: ValueError: unit 1"):
        WK.run(_probe, units, 3, names=["a", "b", "c"])
    assert time.time() - t0 < 60          # the 120 s unit was ended, not waited for
    with pytest.raises(WK.WorkerFailed, match=r"a worker died with units (0, )?1 unfinished: "
                                              "BrokenProcessPool"):
        WK.run(_probe, [{"i": 0}, {"i": 1, "die": True}], 2)
    # a large state that a worker cannot load (it fails in the initializer): the run fails,
    # the parent never blocks writing it
    t0 = time.time()
    with pytest.raises(WK.WorkerFailed, match="a worker died with units 0, 1 unfinished"):
        WK.run(_probe, [{"i": 0}, {"i": 1}], 2, state={"big": np.zeros(10_000_000),
                                                      "bad": _Unloadable()})
    assert time.time() - t0 < 60
    with pytest.raises(ValueError, match="unit 1 raised"):     # the serial path: its own
        WK.run(_probe, units[:2], 1)
    assert not list(tmp_path.glob("costate-workers-*"))     # the state's file is removed


def test_a_script_function_reaches_a_worker_by_file_and_name():
    cs = _script()
    from apexzero.costate import gates as GT

    assert WK._ref(GT.decoy_k_unit) is GT.decoy_k_unit
    ref = WK._ref(cs.probes_unit)
    assert ref == WK.ScriptFn(str(SCRIPT.resolve()), "probes_unit")
    assert WK._ref(_probe).path == str(Path(__file__).resolve())
    with pytest.raises(TypeError, match="module-level"):
        WK._ref(lambda s, u: u)
    assert pickle.loads(pickle.dumps(ref)) == ref


def test_the_options_the_default_pool_and_the_thread_pins(monkeypatch):
    cs = _script()
    assert cs.POOLED == WK.POOLED and cs.BLAS_THREADS == WK.BLAS_THREADS
    assert WK.THREAD_ENV == ("XLA_FLAGS", *WK.BLAS_THREADS)
    assert WK.default_workers(2, cores=12) == 5 and WK.default_workers(1, cores=12) == 11
    assert WK.default_workers(8, cores=4) == 1
    ap = cs.build_parser()
    for stage in cs.STAGES:
        args = [stage, "--workers", "3"]
        if stage in WK.POOLED:
            assert ap.parse_args(args).workers == 3
            assert ap.parse_args([stage]).workers is None
            with pytest.raises(SystemExit):
                ap.parse_args([stage, "--workers", "0"])
        else:
            with pytest.raises(SystemExit):
                ap.parse_args(args)
    # never partial, alone or beside the options that are
    for argv in (["probes"], ["own", "--network", "G1"], ["nulls", "--mlp-maxiter", "5"],
                 ["decoy-k", "--out", "/elsewhere"], ["nulls"]):
        a, b = ap.parse_args(argv), ap.parse_args([*argv, "--workers", "4"])
        assert cs._partial(a) == cs._partial(b)
    assert cs._partial(ap.parse_args(["decoy-k", "--workers", "4"])) == []
    # the pool of a stage: --workers, else the default, at most one per unit
    a = ap.parse_args(["--threads", "2", "nulls"])
    monkeypatch.setattr(cs.F, "_PROV", [cs.Provenance(a, "nulls")])
    assert cs._workers(a) == WK.default_workers(2) and cs._workers(a, 2) == min(
        2, WK.default_workers(2))
    assert cs._workers(ap.parse_args(["nulls", "--workers", "7"]), 3) == 3
    assert cs.F._prov().workers == 3                  # what the stamps record
    assert cs._stage_arg(["x", "--threads", "2", "nulls", "--workers", "3"]) == "nulls"
    assert cs._stage_arg(["x", "--threads=4", "probes"]) == "probes"
    assert cs._stage_arg(["x"]) is None


def test_savez_bytes_depend_on_the_arrays_alone(tmp_path):
    cs = _script()
    arrays = {"a": np.arange(12.0).reshape(3, 4), "b": np.array([True, False]),
              "c": np.float64(2.5), "s": np.array(["x", "yz"])}
    p, q, r = tmp_path / "p.npz", tmp_path / "q.npz", tmp_path / "r.npz"
    cs.savez(p, arrays)
    time.sleep(2.1)                          # a zip timestamp's resolution is 2 s
    cs.savez(q, arrays)
    np.savez_compressed(r, **arrays)
    assert p.read_bytes() == q.read_bytes()
    with np.load(p) as d, np.load(r) as e:
        assert d.files == e.files == list(arrays)
        for k in arrays:
            assert d[k].dtype == e[k].dtype and np.array_equal(d[k], e[k])
    import zipfile

    with zipfile.ZipFile(p) as a, zipfile.ZipFile(r) as b:
        assert [(i.filename, i.compress_type, i.file_size, i.CRC) for i in a.infolist()] == \
            [(i.filename, i.compress_type, i.file_size, i.CRC) for i in b.infolist()]
        assert {i.date_time for i in a.infolist()} == {(1980, 1, 1, 0, 0, 0)}


# ------------------------------------------------------------------------- the stages

# Runs one pooled stage through main() in a fresh process (its thread pins set before numpy
# loads), its parent-side inputs patched from the test's files (``conf``).
_DRIVER = r"""
import json, pickle, sys
from pathlib import Path

script, conf = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
sys.argv = [str(script), *sys.argv[3:]]           # what every stamp records
sys.path[:0] = [str(script.parent), str(script.parent.parent / "src")]
import costate as cs                              # the thread pins, before numpy loads
import numpy as np
from apexzero.costate import critic as CR, gates as GT, inputs as IN, probes as PB
from apexzero.fields import network as NW


def load(key):
    with open(conf[key], "rb") as fh:
        return pickle.load(fh)


stage = conf["stage"]
if stage == "decoy-k":
    pw = load("pw")
    cs._probe_world = lambda args, c2_setup=False: (pw, None, {"world_sha256": "toy"})
    _dk = GT.decoy_k

    def decoy_k(pw_, *a, **k):
        k.update(candidates=tuple(conf["candidates"]), seeds=tuple(conf["seeds"]))
        return _dk(pw_, *a, **k)

    GT.decoy_k = decoy_k
if stage == "nulls":
    rms = load("rms")
    IN.obs_rms_of = lambda name: (rms[name], None)
if stage in ("probes", "own"):
    rec = conf["rec"]
    agents = load("agents")

    def post_tag(args, what):
        cs._x64()
        if stage == "own":                 # random-init stand-ins, unlocked as after the tag
            CR.unlock_trained(); PB.unlock_trained(); NW.unlock_trained()
        return rec

    cs._post_tag = post_tag
    cs.load_agent = lambda name, pin: agents[name]
if stage == "probes":
    world, flags = load("world"), load("flags")
    cs._load_world_frozen = lambda args, rec_: world
    NW.upcast32 = lambda x: x
    IN.flags = lambda world_, rms_: flags
if stage == "own":
    cs._load_world_frozen = lambda args, rec_: IN.load_world(conf["world"])
cs.main(sys.argv[1:])
"""


def _run(tmp: Path, conf: dict, stage: str, workers: int, *extra: str, ok: bool = True):
    """The stage through ``_DRIVER`` into ``tmp``/w<workers> (2 threads); returns its root."""
    out = tmp / f"w{workers}"
    cp = tmp / "conf.json"
    cp.write_text(json.dumps({**conf, "stage": stage}, default=str))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", *WK.THREAD_ENV)}
    r = subprocess.run([sys.executable, "-c", _DRIVER, str(SCRIPT), str(cp), "--threads", "2",
                        stage, "--out", str(out), "--log-dir", str(out / "logs"),
                        "--workers", str(workers), *extra], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=900, check=False)
    if ok:
        assert r.returncode == 0, r.stderr[-4000:]
    else:
        assert r.returncode != 0, "the stage ran where a unit raises"
    return out, r


def _content_sha256(obj: dict) -> str:
    """A JSON output's content hash without its stamp (scripts/fields.py's
    ``content_sha256``, which its stamp binds)."""
    from apexzero.fields.prereg import sha256_bytes
    from apexzero.interp import io

    return sha256_bytes(io.json_bytes({k: v for k, v in obj.items() if k != "provenance"}))


def _outputs(root: Path, inputs: tuple[str, ...] = ()) -> dict:
    """Every output under ``root`` (the logs and the ``inputs`` directories aside): a JSON
    object's content_sha256 (recomputed) and stamp, an npz's bytes' sha256 and arrays, a
    sidecar's stamp."""
    from apexzero.fields.prereg import sha256_file

    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or rel.parts[0] == "logs" or str(rel).startswith(inputs):
            continue
        if p.name.endswith(".provenance.json"):
            out[str(rel)] = {"stamp": json.loads(p.read_text())}
        elif p.suffix == ".json":
            d = json.loads(p.read_text())
            out[str(rel)] = {"content": _content_sha256(d), "stamp": d["provenance"]}
        elif p.suffix == ".npz":
            with np.load(p) as z:
                out[str(rel)] = {"sha256": sha256_file(p), "arrays": {k: z[k] for k in z.files}}
    return out


def _same(a: Path, b: Path, files: set[str], inputs: tuple[str, ...] = ()) -> dict:
    """The two runs' outputs are ``files`` and identical in content (module docstring); the
    stamps differ in the pool alone. Returns the first run's outputs."""
    x, y = _outputs(a, inputs), _outputs(b, inputs)
    assert set(x) == set(y) == files
    for rel, v in x.items():
        w = y[rel]
        if "content" in v:
            assert v["content"] == w["content"] == v["stamp"]["content_sha256"] == \
                w["stamp"]["content_sha256"], rel
        if "sha256" in v:
            assert v["sha256"] == w["sha256"], rel
            assert list(v["arrays"]) == list(w["arrays"]), rel
            for k, arr in v["arrays"].items():
                other = w["arrays"][k]
                assert arr.dtype == other.dtype and arr.shape == other.shape, (rel, k)
                assert np.array_equal(arr, other, equal_nan=arr.dtype.kind in "fc"), (rel, k)
        if "stamp" in v:
            sa, sb = v["stamp"], w["stamp"]
            assert (sa["workers"], sb["workers"]) == (1, 3), rel
            assert sa["threads"] == sb["threads"] and sa["threads"]["VECLIB_MAXIMUM_THREADS"] \
                == "1" and "intra_op_parallelism_threads=2" in sa["threads"]["XLA_FLAGS"], rel
            assert {k: s for k, s in sa.items() if k not in VOLATILE} == \
                {k: s for k, s in sb.items() if k not in VOLATILE}, rel
    return x


def _dump(path: Path, obj) -> str:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    return str(path)


def test_decoy_k_with_three_workers_writes_the_serial_record(tmp_path):
    import dataclasses

    from test_costate_gates import _decoy_world

    pw = _decoy_world(n_geo=8, keys=4, steps=12)
    inner = pw.tab["ds_apex"] < 40.0
    pw = dataclasses.replace(pw, tab={**pw.tab, "interior": inner}, rows_c2=inner,
                             power_floor=dict.fromkeys(("s", "n", "xi", "vx"), True))
    conf = {"pw": _dump(tmp_path / "pw.pkl", pw), "candidates": [3, 8], "seeds": [0, 1]}
    a, _ = _run(tmp_path, conf, "decoy-k", 1)
    b, _ = _run(tmp_path, conf, "decoy-k", 3)
    _same(a, b, {"gates/decoy_k.json"})
    rec = json.loads((a / "gates" / "decoy_k.json").read_text())
    assert rec["candidates"] == [3, 8] and rec["seeds"] == [0, 1]
    assert all(len(v) == 2 for d in rec["delta2"].values() for v in d.values())
    assert rec["provenance"]["partial"] == ["--out"]


def _toy_activations(n: int, seed: int) -> dict:
    from apexzero.costate.probes import POOL_SEEDS

    rng = np.random.default_rng(seed)
    return {"z": rng.normal(size=(n, 6)), "m": rng.normal(size=(n, 2)),
            "actor_h3": rng.normal(size=(n, 5)),
            **{f"random{s}_h3": rng.normal(size=(n, 5)) for s in POOL_SEEDS}}


def test_probes_with_three_workers_writes_the_serial_records(tmp_path):
    from test_costate_stages import _toy_world

    from apexzero.costate import prereg as PR

    world, frozen = _toy_world(n_geo=6)
    n = world.n
    rng = np.random.default_rng(4)
    flags = {"observability": SimpleNamespace(s_unobservable=rng.random(n) < 0.1),
             "kappa_clipped": rng.random(n) < 0.2}
    names = ["runs/toy-a", "runs/toy-b", "runs/toy-c"]
    conf = {"world": _dump(tmp_path / "world.pkl", world),
            "flags": _dump(tmp_path / "flags.pkl", flags),
            "agents": _dump(tmp_path / "agents.pkl",
                            {m: SimpleNamespace(obs_rms=None) for m in names}),
            "rec": {"frozen": frozen, "networks": {m: {} for m in names}}}
    for w in (1, 3):
        d = tmp_path / f"w{w}" / "post" / "activations"
        d.mkdir(parents=True)
        for i, m in enumerate(names):
            np.savez(d / f"{PR.net_file(m)}.npz", **_toy_activations(n, i))
    net = [a for m in names for a in ("--network", m)]
    a, _ = _run(tmp_path, conf, "probes", 1, *net)
    b, _ = _run(tmp_path, conf, "probes", 3, *net)
    got = _same(a, b, {f"post/probes/{PR.net_file(m)}.json" for m in names},
                inputs=("post/activations",))
    recs = [json.loads((a / r).read_text()) for r in sorted(got) if "probes" in r]
    assert len({json.dumps(r["C2"], sort_keys=True) for r in recs}) == 3    # three networks
    assert all(r["C2"]["controls"]["n"] == 20 for r in recs)
    # a unit that raises in a worker (toy-b's activations without its h3) fails the stage:
    # no probes file at all, serial or pooled
    for w in (1, 3):
        p = tmp_path / f"w{w}" / "post" / "activations" / f"{PR.net_file(names[1])}.npz"
        np.savez(p, **{k: v for k, v in _toy_activations(n, 1).items() if k != "actor_h3"})
        for f in (tmp_path / f"w{w}" / "post" / "probes").glob("*.json"):
            f.unlink()
        _, r = _run(tmp_path, conf, "probes", w, *net, ok=False)
        assert "actor_h3" in r.stderr
        assert ("unit runs/toy-b failed in a worker" in r.stderr) == (w == 3), r.stderr[-2000:]
        assert not list((tmp_path / f"w{w}" / "post" / "probes").glob("*.json"))


# T's world on a few stored keys (rows, inputs and observation maps; no network), one key per
# geometry, built in a float64 process as the inputs stage builds it
_WORLD = r"""
import sys
import jax
jax.config.update("jax_enable_x64", True)
from apexzero.costate import inputs as IN, states as ST
bank, _ = IN.design()
by = {}
for q in sorted(ST.a2_refs(), key=lambda r: r.key_id):
    s = ST.strata(q)
    if s.w_is_cell:
        by.setdefault((s.R, s.theta_deg), q)
IN.save_world(sys.argv[1], IN.build_world([IN.key_data(q, bank) for q in
                                           [by[g] for g in sorted(by)][::4][:int(sys.argv[2])]]))
"""


def _world(path: Path, keys: int) -> Path:
    r = subprocess.run([sys.executable, "-c", _WORLD, str(path), str(keys)], cwd=REPO,
                       capture_output=True, text=True, timeout=600, check=False,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert r.returncode == 0, r.stderr[-3000:]
    return path


@pytest.fixture(scope="module")
def two_keys(tmp_path_factory):
    """Two keys: the own stage's restarts and returns (its cost is per restart)."""
    return _world(tmp_path_factory.mktemp("two_keys") / "world.npz", 2)


@pytest.fixture(scope="module")
def twelve_keys(tmp_path_factory):
    """12 keys on 12 geometries: enough for the six geometry folds of the value fits."""
    return _world(tmp_path_factory.mktemp("twelve_keys") / "world.npz", 12)


@pytest.mark.skipif(not HAVE_KEYS, reason="step 1's keys or stimuli are missing")
def test_nulls_with_three_workers_write_the_serial_arrays(tmp_path, twelve_keys):
    import shutil

    from apexzero.agents.ppo import RunningMeanStd
    from apexzero.costate import critic as CR
    from apexzero.costate import prereg as PR

    names = ["runs/toy-a", "runs/toy-b", "runs/toy-c"]
    rms = {m: RunningMeanStd(np.full(CR.OBS_DIM, 0.05 * i), np.full(CR.OBS_DIM, 1.0 + 0.2 * i),
                             np.float64(1e4)) for i, m in enumerate(names)}
    conf = {"rms": _dump(tmp_path / "rms.pkl", rms)}
    for w in (1, 3):
        (tmp_path / f"w{w}" / "inputs").mkdir(parents=True)
        shutil.copy(twelve_keys, tmp_path / f"w{w}" / "inputs" / "world.npz")
    opts = [*(a for m in names for a in ("--network", m)), "--mlp-maxiter", "20"]
    a, _ = _run(tmp_path, conf, "nulls", 1, *opts)
    b, _ = _run(tmp_path, conf, "nulls", 3, *opts)
    files = {f"nulls/{PR.net_file(m)}.npz{s}" for m in names for s in ("", ".provenance.json")}
    got = _same(a, b, files | {"nulls/summary.json"}, inputs=("inputs",))
    summ = json.loads((a / "nulls" / "summary.json").read_text())
    assert list(summ["networks"]) == names        # the serial order
    for m in names:                               # the summary hashes the bytes written
        assert summ["networks"][m]["sha256"] == got[f"nulls/{PR.net_file(m)}.npz"]["sha256"]
    assert got["nulls/runs_toy-a.npz"]["arrays"]["qss"].shape[1] == 8
    assert not np.array_equal(got["nulls/runs_toy-a.npz"]["arrays"]["value_ridge"],
                              got["nulls/runs_toy-b.npz"]["arrays"]["value_ridge"])
    # a unit that raises (toy-b's obs_rms of the wrong size) fails the stage: nothing written
    bad = {**rms, names[1]: RunningMeanStd(np.zeros(5), np.ones(5), np.float64(1.0))}
    conf = {"rms": _dump(tmp_path / "bad.pkl", bad)}
    for w in (1, 3):
        shutil.rmtree(tmp_path / f"w{w}" / "nulls")
        _, r = _run(tmp_path, conf, "nulls", w, *opts, ok=False)
        assert ("unit runs/toy-b failed in a worker" in r.stderr) == (w == 3), r.stderr[-2000:]
        assert not (tmp_path / f"w{w}" / "nulls").exists() or not any(
            (tmp_path / f"w{w}" / "nulls").iterdir())


@pytest.mark.skipif(not (HAVE_KEYS and _have_g1()), reason="step 1's keys or G1 are missing")
def test_own_with_three_workers_writes_the_serial_arrays(tmp_path, two_keys):
    from apexzero.agents.ppo import RunningMeanStd
    from apexzero.costate import critic as CR
    from apexzero.costate import inputs as IN
    from apexzero.costate import prereg as PR
    from apexzero.fields import network as NW

    rms, cfg = IN.obs_rms_of("G1")            # G1's config and obs_rms; random-init params
    names = ["runs/toy-a", "runs/toy-b", "runs/toy-c"]
    agents = {}
    for i, m in enumerate(names):
        snap = SimpleNamespace(actor=NW.random_actor_params(cfg, 101 + i),
                               critic=CR.random_critic_params(cfg, 101 + i),
                               ret_rms=RunningMeanStd(np.zeros(()), np.float32(4.0),
                                                      np.float32(1.0)))
        agents[m] = SimpleNamespace(agent_id=f"stand-in-{i}", timesteps=0, cfg=cfg,
                                    obs_rms=rms, snapshot=snap, actor_params=snap.actor)
    conf = {"agents": _dump(tmp_path / "agents.pkl", agents), "world": str(two_keys),
            "rec": {"frozen": {}, "networks": {m: {} for m in names}}}
    net = [a for m in names for a in ("--network", m)]
    a, _ = _run(tmp_path, conf, "own", 1, *net)
    b, _ = _run(tmp_path, conf, "own", 3, *net)
    files = {f"post/own/{PR.net_file(m)}{s}" for m in names
             for s in (".k3b.json", ".npz", ".npz.provenance.json")}
    got = _same(a, b, files)
    arr = [got[f"post/own/{PR.net_file(m)}.npz"]["arrays"] for m in names]
    assert all(bool(x["v_rows"].any()) and "k3b_pass" in x for x in arr)
    assert not np.array_equal(arr[0]["v_own_det"], arr[1]["v_own_det"], equal_nan=True)
