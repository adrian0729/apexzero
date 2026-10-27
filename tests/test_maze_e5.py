"""Maze E5 (docs/prereg/maze-e5.md): the vendored SFL source and its shim, the run lists, the
train stage, the config and round checks, the tag gate, the audit's additions, and the
statistics on planted data.

SFL's own network file (sfl/train/minigrid_plr.py) is not vendored, so the shim's end-to-end
test runs the vendored minigrid_sfl.py with JaxUED's maze_plr.py network as a stand-in
(``network_module``), at a toy size. The planted analyses use 120 mazes and fewer bootstrap
draws than the pre-registered 4,000, enough for every planted effect to clear Holm.
"""

import importlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import jax
import numpy as np
import pytest

import apexzero.compat  # noqa: F401  (must precede jaxued calls)
from apexzero.maze import audit as maze
from apexzero.maze import e5, sfl
from apexzero.maze import e5_audit as ea
from apexzero.maze.fidelity import spearman

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CHECKPOINTS = [300, 400, 600, 800, 1000, 1400, 1900, 2600, 3500, 4800, 6500, 8800, 11900,
               16200, 22100, 30000]


@pytest.fixture(scope="module")
def mods():
    sys.path.insert(0, str(SCRIPTS))
    try:
        yield {n: importlib.import_module(n) for n in ("maze_e5", "maze_e5_runs",
                                                        "round_manifest")}
    finally:
        sys.path.remove(str(SCRIPTS))


# --------------------------------------------------------------------------- vendoring


def test_vendored_files_match_their_pinned_blobs(tmp_path):
    blobs = sfl.check_vendored()
    assert blobs["third_party/sfl/sfl/train/minigrid_sfl.py"] == sfl.VENDORED[
        "sfl/train/minigrid_sfl.py"]
    assert blobs["third_party/jaxued_examples/maze_plr.py"].startswith("a7b6164d")
    lic = sfl.VENDOR / "LICENSE"
    git = subprocess.run(["git", "hash-object", "--no-filters", str(lic)], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert git == sfl.git_blob_sha1(lic.read_bytes()) == sfl.VENDORED["LICENSE"]
    (tmp_path / "x").write_bytes(b"abc")
    with pytest.raises(sfl.VendorMismatch, match="blob"):
        sfl.check_files({"x": sfl.git_blob_sha1(b"abd")}, tmp_path)
    with pytest.raises(sfl.VendorMismatch, match="missing"):
        sfl.check_files({"y": "0" * 40}, tmp_path)


def test_readme_records_source_commit_and_every_blob():
    text = (sfl.VENDOR / "README.md").read_text()
    assert sfl.SFL_COMMIT in text and "Apache" in text
    for path, blob in sfl.VENDORED.items():
        assert path in text and blob in text
    for path, (blob, _) in sfl.NOT_VENDORED.items():
        assert path in text and blob in text


def test_sfls_network_file_is_required_until_it_is_vendored():
    if (sfl.VENDOR / sfl.MINIGRID_PLR).exists():
        pytest.skip("minigrid_plr.py is vendored")
    with pytest.raises(sfl.MissingSFLSource, match="needs the user's OK"):
        sfl.check_minigrid_plr()
    with pytest.raises(sfl.MissingSFLSource):
        sfl.check_vendored(network=True)


# --------------------------------------------------------------------------- the config


def test_compose_config_is_hydras_composition_with_omegaconfs_yaml():
    cfg = sfl.compose_config()
    assert set(cfg) == {"SEED", "GROUP_NAME", "ENTITY", "PROJECT", "WANDB_MODE", "SAVE_PATH",
                        "BATCH_SIZE", "NUM_BATCHES", "ROLLOUT_STEPS", "NUM_TO_SAVE",
                        "EVAL_SAMPLED_SET_PATH", "env", "learning"}
    assert cfg["env"] == {"env_name": "Minigrid-v0", "N_WALLS": 60, "AGENT_VIEW_SIZE": 5,
                          "EVAL_NUM_ATTEMPTS": 5,
                          "EVAL_LEVELS": ["SixteenRooms", "SixteenRooms2", "Labyrinth",
                                          "LabyrinthFlipped", "Labyrinth2", "StandardMaze",
                                          "StandardMaze2", "StandardMaze3"]}
    lc = cfg["learning"]
    assert lc["TOTAL_TIMESTEPS"] == 3e8 and isinstance(lc["TOTAL_TIMESTEPS"], float)
    assert lc["EVAL_TIMESTEPS"] == 2.5e7 and lc["LR"] == 2.5e-4
    assert lc["NUM_ENVS"] == 256
    assert lc["NUM_ENVS_FROM_SAMPLED"] == lc["NUM_ENVS_TO_GENERATE"] == 128
    assert (lc["GAMMA"], lc["GAE_LAMBDA"], lc["NUM_STEPS"], lc["EVAL_FREQ"],
            lc["NUM_CHECKPOINTS"], lc["UPDATE_EPOCHS"], lc["NUM_MINIBATCHES"]) == (
        0.99, 0.95, 256, 100, 4, 4, 4)
    assert (cfg["SEED"], cfg["BATCH_SIZE"], cfg["NUM_BATCHES"], cfg["ROLLOUT_STEPS"],
            cfg["NUM_TO_SAVE"], cfg["WANDB_MODE"]) == (2, 5000, 5, 2000, 1000, "disabled")
    assert sfl.num_updates(cfg) == 4577          # SFL's own length, 3e8 // 256 // 256
    import yaml

    assert yaml.safe_load("x: 3e8")["x"] == "3e8"   # plain PyYAML: a string


def test_e5_overrides_change_seed_view_length_and_save_cadence_only():
    base = sfl.compose_config()
    ov = sfl.e5_overrides(3, 13, 30000)
    assert ov == ["SEED=3", "env.AGENT_VIEW_SIZE=13", "learning.TOTAL_TIMESTEPS=1966080000",
                  "learning.NUM_CHECKPOINTS=300"]
    cfg = sfl.compose_config(ov)
    assert sfl.num_updates(cfg) == 30000
    assert cfg["learning"]["NUM_CHECKPOINTS"] == 300
    changed = {("SEED",), ("env", "AGENT_VIEW_SIZE"), ("learning", "TOTAL_TIMESTEPS"),
               ("learning", "NUM_CHECKPOINTS")}
    for top, v in base.items():
        if isinstance(v, dict):
            for k, x in v.items():
                assert (top, k) in changed or cfg[top][k] == x
        else:
            assert (top,) in changed or cfg[top] == v
    assert sfl.run_settings(cfg) == dict(n_walls=60, gamma=0.99, gae_lambda=0.95, num_steps=256,
                                         agent_view_size=13, rollout_steps=2000, eval_freq=100)
    with pytest.raises(KeyError):
        sfl.compose_config(["learning.NOT_A_KEY=1"])
    with pytest.raises(ValueError):
        sfl.compose_config(["SEED"])
    with pytest.raises(ValueError):
        sfl.e5_overrides(1, 5, 250)


# --------------------------------------------------------------------------- outcomes


def test_native_learnability_equals_the_port_of_sfls_counting():
    rng = np.random.default_rng(0)
    for p_done in (0.0015, 0.01, 0.05):          # few, some and more than 10 episodes
        done = rng.random((2000, 40)) < p_done
        reward = np.where(done & (rng.random((2000, 40)) < 0.5), rng.uniform(0.1, 1, (2000, 40)),
                          0.0)
        ref = sfl.sfl_outcomes_reference(2000, done, reward)
        nat = sfl.native_learnability(done, reward)
        np.testing.assert_array_equal(ref["num_episodes"], nat["episodes"])
        np.testing.assert_allclose(ref["success_rate"], nat["success_rate"], atol=1e-6,
                                   equal_nan=True)
    done, reward = np.zeros((2000, 3), bool), np.zeros((2000, 3))
    ends = np.arange(99, 1200, 100)              # 12 episodes; the first 10 count
    done[ends, 0] = True
    reward[ends[[0, 1, 10]], 0] = 0.9            # goals in episodes 1, 2 and 11
    done[[199, 499], 1] = True
    reward[199, 1] = 0.5
    reward[1500, 1] = 0.7                        # in the open tail: dropped
    nat = sfl.native_learnability(done, reward)
    assert nat["episodes"].tolist() == [10, 2, 0]
    np.testing.assert_allclose(nat["success_rate"][:2], [0.2, 0.5])
    np.testing.assert_allclose(nat["learnability"][:2], [0.16, 0.25])
    assert np.isnan(nat["success_rate"][2])
    ref = sfl.sfl_outcomes_reference(2000, done, reward)
    np.testing.assert_allclose(ref["success_rate"][:2], [0.2, 0.5])
    with pytest.raises(ValueError):
        sfl.native_learnability(done[:100], reward[:100])


def test_top_k_follows_the_score_specs_s1_to_s3():
    import jax.numpy as jnp

    assert sfl.reference_top_k([0.25, 0.0, 0.1875, 0.0, 0.24], 2).tolist() == [4, 0]
    assert sfl.reference_top_k([0.0, 0.0, 0.0], 1).tolist() == [2]
    with pytest.raises(ValueError):
        sfl.reference_top_k([0.25, np.nan, 0.1], 1)
    x = np.random.default_rng(1).integers(0, 5, 50) / 16
    assert np.array_equal(np.asarray(jnp.argsort(jnp.asarray(x))[-10:]),
                          sfl.reference_top_k(x, 10))      # minigrid_sfl.py L289


# --------------------------------------------------------------------------- checkpoints


def _level_dict(levels) -> dict:
    return {k: np.asarray(getattr(levels, k)) for k in sfl.LEVEL_FIELDS}


def test_sfl_checkpoint_round_trip_and_its_learnable_set(tmp_path):
    params = {"params": {"a": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)},
                         "b": np.ones(4, np.float32)}}
    lv = maze.generate_levels(5, jax.random.PRNGKey(0))
    learnable = {**_level_dict(lv), "scores": np.linspace(0, 0.25, 5), "refresh_update": 200,
                 "update": 300}
    (tmp_path / "config.json").write_text(json.dumps({"config": sfl.compose_config()}))
    sfl.save_checkpoint(tmp_path, 300, params, learnable)
    with pytest.raises(FileExistsError):
        sfl.save_checkpoint(tmp_path, 300, params, learnable)
    ck = sfl.load_sfl_checkpoint(tmp_path, 300)
    np.testing.assert_array_equal(ck.params["params"]["a"]["kernel"],
                                  params["params"]["a"]["kernel"])
    jax.tree.map(np.testing.assert_array_equal, sfl.learnable_levels(ck.learnable), lv)
    assert sfl.checkpoint_updates(tmp_path) == [300] and ck.update == 300
    levels, stored, age = ea.inbuffer_levels("sfl", ck, 0, 300)
    np.testing.assert_array_equal(stored, learnable["scores"])
    assert age.tolist() == [100.0] * 5
    assert ea.settings("sfl", ck.config)["gamma"] == 0.99


def test_inbuffer_plr_entries_are_the_valid_ones_or_a_fixed_1000():
    lv = maze.generate_levels(1500, jax.random.PRNGKey(1))
    sampler = {"levels": _level_dict(lv), "scores": np.arange(1500.0),
               "timestamps": np.arange(1500), "size": 1200, "episode_count": 5000}
    ck = maze.MazeCheckpoint(None, {}, 0, 300, sampler)
    levels, stored, age = ea.inbuffer_levels("plr", ck, 3, 300)
    assert len(stored) == 1000 and stored.max() < 1200 and np.all(np.diff(stored) > 0)
    np.testing.assert_array_equal(age, 5000 - stored)
    np.testing.assert_array_equal(np.asarray(levels.wall_map),
                                  np.asarray(lv.wall_map)[stored.astype(int)])
    again = ea.inbuffer_levels("plr", ck, 3, 300)[1]
    np.testing.assert_array_equal(again, stored)
    sampler["size"] = 800
    assert ea.inbuffer_levels("plr", ck, 3, 300)[1].tolist() == list(range(800))
    assert ea.inbuffer_levels("dr", ck, 3, 300) is None


# --------------------------------------------------------------------------- the shim

SMALL = ["SEED=3", "learning.NUM_ENVS=4", "learning.NUM_ENVS_FROM_SAMPLED=2",
         "learning.NUM_ENVS_TO_GENERATE=2", "learning.NUM_STEPS=16", "learning.EVAL_FREQ=2",
         "learning.TOTAL_TIMESTEPS=384", "learning.NUM_CHECKPOINTS=3",
         "learning.NUM_MINIBATCHES=2", "learning.UPDATE_EPOCHS=1", "BATCH_SIZE=16",
         "NUM_BATCHES=2", "ROLLOUT_STEPS=300", "NUM_TO_SAVE=20", "env.EVAL_NUM_ATTEMPTS=1"]


@pytest.fixture(scope="module")
def shim_runs(tmp_path_factory):
    sys.path.insert(0, str(SCRIPTS))
    try:
        shim = importlib.import_module("run_sfl_maze")
    finally:
        sys.path.remove(str(SCRIPTS))
    out = {}
    for capture in (True, False):
        d = tmp_path_factory.mktemp("capture" if capture else "plain")
        with jax.enable_x64(False):
            saved = shim.run(SMALL, [4, 6], capture=capture, out=d,
                             network_module=maze.maze_plr())
        out[capture] = (d, saved)
    return shim, out


def test_sfl_shim_runs_the_vendored_loop_and_saves_params_and_d(shim_runs):
    _, out = shim_runs
    d, saved = out[True]
    assert saved == [4, 6]
    rec = json.loads((d / "config.json").read_text())
    assert rec["overrides"] == SMALL and rec["config"] == sfl.compose_config(SMALL)
    assert rec["network"] == "stand-in" and rec["capture_learnable_set"] is True
    log = [json.loads(x) for x in (d / "sfl-eval.jsonl").read_text().splitlines()]
    assert [x["update"] for x in log] == [2, 4, 6]
    for u, line in zip((4, 6), log[1:]):
        ck = sfl.load_sfl_checkpoint(d, u)
        assert int(ck.learnable["refresh_update"]) == u - 2 and int(ck.learnable["update"]) == u
        assert len(ck.learnable["scores"]) == 20
        assert np.all(np.diff(ck.learnable["scores"]) >= 0)          # argsort's top-K order
        assert ck.learnable["scores"].mean() == pytest.approx(
            line["learnability_set_mean_score"], abs=1e-7)
        sfl.learnable_levels(ck.learnable)
    assert not (out[False][0] / sfl.CKPT_DIR / "u00004" / "learnable.npz").exists()


def test_capturing_d_leaves_the_parameters_bit_identical(shim_runs):
    _, out = shim_runs
    for u in (4, 6):
        a = (sfl.checkpoint_dir(out[True][0], u) / "params.msgpack").read_bytes()
        b = (sfl.checkpoint_dir(out[False][0], u) / "params.msgpack").read_bytes()
        assert a == b


def test_sfl_shim_refuses_saves_it_cannot_make(shim_runs, tmp_path):
    shim, _ = shim_runs
    with pytest.raises(SystemExit, match="multiples"):
        shim.run(SMALL, [2], out=tmp_path, network_module=maze.maze_plr())    # the first block
    with pytest.raises(SystemExit, match="multiples"):
        shim.run(SMALL, [5], out=tmp_path, network_module=maze.maze_plr())
    bad = [x for x in SMALL if not x.startswith("learning.NUM_CHECKPOINTS")]
    with pytest.raises(SystemExit, match="NUM_CHECKPOINTS"):
        shim.run(bad, [4], out=tmp_path, network_module=maze.maze_plr())
    if not (sfl.VENDOR / sfl.MINIGRID_PLR).exists():
        with pytest.raises(sfl.MissingSFLSource):
            shim.run(SMALL, [4], out=tmp_path)


def test_where_mask_shim_reduces_over_the_mask(shim_runs):
    import jax.numpy as jnp

    x, m = jnp.arange(6.0), jnp.array([0, 1, 1, 0, 1, 0])
    assert float(x.mean(where=m)) == float(x.mean(where=m.astype(bool))) == pytest.approx(7 / 3)
    assert float(x.std(where=m)) == float(x.std(where=m.astype(bool)))


# --------------------------------------------------------------------------- run lists


def test_main_run_list(mods, monkeypatch):
    runs, rm = mods["maze_e5_runs"].runs, mods["round_manifest"]
    monkeypatch.chdir(ROOT)
    r = runs("main")
    assert len(r) == 30 and len({x["name"] for x in r}) == 30
    assert [x["arm"] for x in r] == ["sfl"] * 10 + ["plr"] * 10 + ["dr"] * 10
    assert [x["name"] for x in r[:3]] == ["maze-e5-sfl-v13-s1", "maze-e5-sfl-v5-s1",
                                          "maze-e5-sfl-v13-s2"]
    assert set(Counter((x["arm"], x["view"]) for x in r).values()) == {5}
    assert {x["seed"] for x in r} == {1, 2, 3, 4, 5}
    assert rm.run_list_problems(r) == []
    assert all(x["cmd"][0] == "scripts/maze_e5.py" and "config" not in x for x in r)
    lanes = Counter(lane for lane, _, _, run in rm.schedule(r, 4, 4) if run["arm"] == "sfl")
    assert max(lanes.values()) == 1               # no lane holds two SFL runs at 16 lanes
    assert [len(runs("main", drop=d)) for d in range(5)] == [30, 25, 15, 12, 9]
    assert not any(x["arm"] == "sfl" and x["view"] == 13 for x in runs("main", drop=1))
    assert {x["view"] for x in runs("main", drop=2)} == {5}
    assert {x["seed"] for x in runs("main", drop=4)} == {1, 2, 3}
    with pytest.raises(ValueError):
        runs("main", drop=5)


def test_the_other_rounds(mods, monkeypatch):
    R, rm = mods["maze_e5_runs"], mods["round_manifest"]
    monkeypatch.chdir(ROOT)
    t = R.runs("timing", lanes_per_gpu=4, gpus=4)
    cells = [(a, v) for a in ("sfl", "plr", "dr") for v in (13, 5)]
    assert len(t) == 16 and [(x["arm"], x["view"]) for x in t[:6]] == cells
    assert {x["seed"] for x in t} == {99} and {x["updates"] for x in t} == {500}
    assert all(x["name"].startswith("maze-e5t4-") for x in t)
    assert Counter(gpu for _, gpu, _, _ in rm.schedule(t, 4, 4)) == {0: 4, 1: 4, 2: 4, 3: 4}
    ta = R.runs("timing-audit", lanes_per_gpu=4, gpus=4)
    assert [x["cmd"][x["cmd"].index("--source") + 1] for x in ta] == [x["name"] for x in t[:6]]
    assert all("--timing-only" in x["cmd"] for x in ta)
    au = R.runs("audit")
    assert [x["source"] for x in au] == [x["name"] for x in R.runs("main")]
    cc = R.runs("capture-check")
    assert sum("--no-capture" in x["cmd"] for x in cc) == 1 and all("200" in x["cmd"] for x in cc)
    assert not any(x["arm"] == "sfl" for x in R.runs("timing", arms=("dr", "plr")))
    one = R.runs("timing", lanes_per_gpu=4, gpus=1)        # one GPU: two waves of 4
    assert len(one) == 8 and {(x["arm"], x["view"]) for x in one} == set(cells)
    every = {"maze-e5": R.runs("main"), "maze-e5t4": t, "maze-e5t8": R.runs("timing", 0, 8, 4),
             "maze-e5at": ta, "maze-e5a": au, "maze-e5c": cc}
    for name, rl in every.items():
        assert rm.run_list_problems(rl) == []
        assert all(x["name"].startswith(name + "-") for x in rl)
        for other, rl2 in every.items():
            assert other == name or not any(x["name"].startswith(name + "-") for x in rl2)
    out = subprocess.run([sys.executable, str(SCRIPTS / "maze_e5_runs.py"), "--phase", "main"],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    assert [json.loads(x)["name"] for x in out] == [x["name"] for x in R.runs("main")]
    bad = subprocess.run([sys.executable, str(SCRIPTS / "maze_e5_runs.py"), "--arms", "dr"],
                         capture_output=True, text=True, check=False)
    assert bad.returncode != 0


# --------------------------------------------------------------------------- trainers


PILOT_ARGS = ["--run_name", "dr", "--seed", "0", "--num_updates", "30000", "--eval_freq", "50",
              "--checkpoint_save_interval", "1", "--replay_prob", "0",
              "--exploratory_grad_updates"]
PILOT_GROUP = ("5True0.20.50.0015True0.980.99540000.00010.50.5255None12563230000rank0.0MaxMC0.3"
               "0.34False")


def test_maze_plr_parser_reproduces_the_pilots_config(mods):
    m = mods["maze_e5"]
    cfg = m.expected_jaxued_config(PILOT_ARGS)
    assert cfg["group_name"] == PILOT_GROUP
    assert cfg["replay_prob"] == 0 and cfg["exploratory_grad_updates"] is True
    assert cfg["max_number_of_checkpoints"] == 60 and cfg["eval_levels"][0] == "SixteenRooms"
    pilot = ROOT / "runs" / "maze-audit" / "pilot-v2" / "checkpoints" / "dr" / "0" / "config.json"
    if pilot.is_file():                           # runs/ is local: compared when present
        got = json.loads(pilot.read_text())
        assert set(got) == set(cfg) and all(m._same(got[k], cfg[k]) for k in got)


def test_trainer_commands_are_section_2s(mods):
    m = mods["maze_e5"]
    dr = m.trainer_cmd("x", "dr", 13, 2, 30000)
    assert dr == ["scripts/run_jaxued_maze.py", "--run_name", "x", "--seed", "2",
                  "--agent_view_size", "13", "--num_updates", "30000", "--eval_freq", "100",
                  "--checkpoint_save_interval", "1", "--max_number_of_checkpoints", "300",
                  "--replay_prob", "0", "--exploratory_grad_updates"]
    plr = m.trainer_cmd("x", "plr", 5, 2, 30000)
    assert plr == [*dr[:6], "5", *dr[7:-3]]
    cfg = m.expected_jaxued_config(plr[1:])
    assert (cfg["score_function"], cfg["replay_prob"], cfg["exploratory_grad_updates"],
            cfg["eval_freq"], cfg["max_number_of_checkpoints"], cfg["num_updates"],
            cfg["agent_view_size"]) == ("MaxMC", 0.8, False, 100, 300, 30000, 5)
    s = m.trainer_cmd("x", "sfl", 13, 2, 30000)
    assert s[1:5] == sfl.e5_overrides(2, 13, 30000)
    assert s[5:] == ["--save-updates", ",".join(map(str, CHECKPOINTS))]
    assert m.trainer_cmd("x", "sfl", 5, 99, 200, [200], capture=False)[-3:] == [
        "--save-updates", "200", "--no-capture"]
    assert m.checkpoint_updates() == e5.checkpoint_updates() == CHECKPOINTS
    assert m.selected(500) == [300, 400]


FAKE_JAXUED = """
import os, sys, time
from pathlib import Path
a = sys.argv[1:]
get = lambda f: a[a.index(f) + 1]
d = Path("checkpoints") / get("--run_name") / get("--seed")
for k in range(int(get("--num_updates")) // 100):
    s = d / "models" / str(k)
    (s / "default").mkdir(parents=True)
    (s / "default" / "x").write_bytes(os.urandom(8))
    time.sleep(0.01)
    (s / "commit_success.txt").write_text("ok")
(d / "config.json").write_text("{}")
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


def test_train_stage_keeps_the_selected_checkpoints_and_records_them(mods, tmp_path,
                                                                     monkeypatch):
    m = mods["maze_e5"]
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE_JAXUED)
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "RUNS", tmp_path / "runs")
    real = m.trainer_cmd
    monkeypatch.setattr(m, "trainer_cmd", lambda *a, **k: [str(fake), *real(*a, **k)[1:]])
    argv = ["train", "--arm", "dr", "--view", "5", "--seed", "3", "--updates", "1000"]
    assert m.main([*argv, "--name", "t-dr"]) == 0
    run = tmp_path / "runs" / "t-dr"
    rec = json.loads((run / "e5-run.json").read_text())
    assert [c["update"] for c in rec["checkpoints"]] == [300, 400, 600, 800, 1000]
    steps = m.jaxued_steps(m.jaxued_dir(run, "t-dr", 3))
    assert sorted(steps) == [2, 3, 5, 7, 9] and rec["pruned"] == 5
    assert len(rec["block_ends_unix"]) == 10 and rec["block_ends_unix"] == sorted(
        rec["block_ends_unix"])
    for c in rec["checkpoints"]:
        assert m.tree_sha256(m.ROOT / c["path"]) == c["sha256"]
    assert rec["trainer_cmd"][1:] == real("t-dr", "dr", 5, 3, 1000)[1:] and rec["problems"] == []
    with pytest.raises(SystemExit, match="not empty"):
        m.main([*argv, "--name", "t-dr"])
    monkeypatch.setenv("FAKE_EXIT", "3")
    assert m.main([*argv, "--name", "t-fail"]) == 1
    rec = json.loads((tmp_path / "runs" / "t-fail" / "e5-run.json").read_text())
    assert rec["exit"] == 3 and rec["problems"] and "pruned" not in rec
    if not (sfl.VENDOR / sfl.MINIGRID_PLR).exists():
        with pytest.raises(sfl.MissingSFLSource):
            m.main(["train", "--arm", "sfl", "--view", "5", "--seed", "1", "--name", "t-sfl"])


def test_manifest_config_checks(mods, tmp_path, monkeypatch):
    m = mods["maze_e5"]
    monkeypatch.setattr(m, "RUNS", tmp_path)
    run = {"name": "maze-e5-plr-v5-s1", "arm": "plr", "view": 5, "seed": 1, "updates": 30000}
    cmd = m.trainer_cmd(run["name"], "plr", 5, 1, 30000)
    ckdir = m.jaxued_dir(tmp_path / run["name"], run["name"], 1)
    ckdir.mkdir(parents=True)
    cfg = m.expected_jaxued_config(cmd[1:])
    (ckdir / "config.json").write_text(json.dumps(cfg))
    rec = {"arm": "plr", "view": 5, "seed": 1, "updates": 30000, "trainer_cmd": cmd, "exit": 0,
           "problems": [], "selected": CHECKPOINTS}
    assert m.config_problems(run, rec)[0] == []
    (ckdir / "config.json").write_text(json.dumps({**cfg, "replay_prob": 0.0}))
    assert any("replay_prob" in p for p in m.config_problems(run, rec)[0])
    (ckdir / "config.json").write_text(json.dumps(cfg))
    assert m.config_problems(run, {**rec, "trainer_cmd": cmd[:-2]})[0]
    assert m.config_problems(run, {**rec, "seed": 2})[0]
    srun = {**run, "name": "maze-e5-sfl-v5-s1", "arm": "sfl"}
    ov = sfl.e5_overrides(1, 5, 30000)
    (tmp_path / srun["name"]).mkdir()
    good = {"overrides": ov, "config": sfl.compose_config(ov), "network": "vendored",
            "save_updates": CHECKPOINTS, "capture_learnable_set": True}
    (tmp_path / srun["name"] / "config.json").write_text(json.dumps(good))
    srec = {**rec, "arm": "sfl", "trainer_cmd": m.trainer_cmd(srun["name"], "sfl", 5, 1, 30000)}
    assert m.config_problems(srun, srec)[0] == []
    bad = {**good, "network": "stand-in"}
    (tmp_path / srun["name"] / "config.json").write_text(json.dumps(bad))
    assert m.config_problems(srun, srec)[0]


def test_round_check_accepts_only_runs_that_failed_twice(mods):
    m, rm = mods["maze_e5"], mods["round_manifest"]
    # the line round_manifest.py check prints for such a run
    assert "f\"{name}: status {rec.get('status')!r} in {path}\"" in Path(rm.__file__).read_text()
    assert m.parse_check("maze-e5", 0, "OK: every run", "", True) == []
    fail = ("runs: 28 done\nFAILED: 2 problem(s)\n"
            "  maze-e5-dr-v5-s1: status 'failed' in runs/maze-e5-manifest-0of1.json\n"
            "  maze-e5-dr-v5-s2: status 'failed' in runs/maze-e5-manifest-0of1.json\n")
    assert m.parse_check("maze-e5", 1, fail, "", True) == ["maze-e5-dr-v5-s1", "maze-e5-dr-v5-s2"]
    with pytest.raises(SystemExit):
        m.parse_check("maze-e5a", 1, fail, "", False)
    mixed = fail + "  runs/maze-e5-x: has the round's prefix but is not in the run list\n"
    with pytest.raises(SystemExit, match="prefix"):
        m.parse_check("maze-e5", 1, mixed, "", True)
    with pytest.raises(SystemExit, match="not found"):
        m.parse_check("maze-e5", 1, "", "tag 'x' not found", True)


# --------------------------------------------------------------------------- the tag gate


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
                          cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_tag_gate_and_the_tagged_manifest(mods, tmp_path, monkeypatch):
    m = mods["maze_e5"]
    origin, repo = tmp_path / "origin.git", tmp_path / "repo"
    _git(tmp_path, "init", "--bare", "-q", str(origin))
    _git(tmp_path, "clone", "-q", str(origin), str(repo))
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "code").mkdir()
    (repo / "code" / "a.py").write_text("x = 1\n")
    (repo / "manifest.json").write_text("{}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "c")
    _git(repo, "push", "-q", "origin", "HEAD")
    monkeypatch.setattr(m, "ROOT", repo)
    gate = lambda: m.tag_gate("t-v1", ("code",))
    with pytest.raises(PermissionError, match="not on origin"):
        gate()
    _git(repo, "tag", "t-v1")
    _git(repo, "push", "-q", "origin", "t-v1")
    with pytest.raises(PermissionError, match="annotated"):
        gate()
    _git(repo, "push", "-q", "origin", ":refs/tags/t-v1")
    _git(repo, "tag", "-d", "t-v1")
    _git(repo, "tag", "-a", "t-v1", "-m", "v1")
    _git(repo, "push", "-q", "origin", "t-v1")
    head = _git(repo, "rev-parse", "HEAD")
    assert gate()["commit"] == head
    (repo / "code" / "a.py").write_text("x = 2\n")
    with pytest.raises(PermissionError, match="differs"):
        gate()
    (repo / "code" / "a.py").write_text("x = 1\n")
    (repo / "code" / "b.py").write_text("")
    with pytest.raises(PermissionError, match="untracked"):
        gate()
    (repo / "code" / "b.py").unlink()
    assert m.tagged_file(repo / "manifest.json", head) == b"{}\n"
    (repo / "manifest.json").write_text("{ }\n")
    with pytest.raises(SystemExit, match="byte-identical"):
        m.tagged_file(repo / "manifest.json", head)


def test_outputs_are_written_once(mods, tmp_path, monkeypatch):
    m = mods["maze_e5"]
    m.write_new(tmp_path / "a.json", b"1")
    with pytest.raises(SystemExit, match="never overwrites"):
        m.write_new(tmp_path / "a.json", b"2")
    monkeypatch.setattr(m, "tag_gate", lambda *a, **k: {"commit": "c", "head": "h"})
    monkeypatch.setattr(m, "OUT", tmp_path)
    (tmp_path / "analyse").mkdir()
    (tmp_path / "analyse" / "analysis.lock").write_text("")
    with pytest.raises(SystemExit, match="runs once"):
        m.main(["analyse"])

    def no_tag(*a, **k):
        raise PermissionError("maze-e5-prereg-v1 is not on origin")

    monkeypatch.setattr(m, "tag_gate", no_tag)
    (tmp_path / "analyse" / "analysis.lock").unlink()
    with pytest.raises(PermissionError):
        m.main(["analyse"])
    monkeypatch.setattr(m, "RUNS", tmp_path / "runs")
    with pytest.raises(PermissionError):
        m.main(["audit", "--run", "maze-e5-dr-v5-s1", "--name", "maze-e5a-dr-v5-s1"])


# --------------------------------------------------------------------------- the audit


@maze.float32
def _random_params(seed: int = 0, value_bias: float = 0.3):
    key = jax.random.PRNGKey(seed)
    mod = maze.maze_plr()
    env, ep = maze.make_env()
    lvl = maze.generate_levels(1, key)
    obs, _ = jax.vmap(env.reset_to_level, (0, 0, None))(jax.random.split(key, 1), lvl, ep)
    x = (jax.tree.map(lambda a: a[None], obs), jax.numpy.zeros((1, 1), bool))
    params = mod.ActorCritic(7).init(key, x, mod.ActorCritic.initialize_carry((1,)))
    params["params"]["critic1"]["bias"] = params["params"]["critic1"]["bias"] + value_bias
    return params


PILOT_CFG = dict(gamma=0.995, gae_lambda=0.98, num_steps=256, agent_view_size=5, n_walls=25)


def test_e5_audit_with_its_additions_off_is_the_pilots_audit():
    params = _random_params()
    ck = maze.MazeCheckpoint(params, PILOT_CFG, 0, 200, {})
    levels = maze.generate_levels(12, jax.random.PRNGKey(4))
    oracle = maze.bfs_optimum(levels, 0.995, 250)
    key = jax.random.PRNGKey(7)
    ref = maze.audit_checkpoint(ck, levels, key, 8, 4, oracle=oracle)
    st = ea.settings("plr", PILOT_CFG)
    assert st == dict(gamma=0.995, lam=0.98, num_steps=256, agent_view_size=5, n_walls=25,
                      max_steps=250)
    pol = maze.network_policy(params)
    got = ea.audit_policy(pol, st, levels, oracle, key, 8, 4, sfl_native=False)
    for k, v in ref.items():
        np.testing.assert_array_equal(got[k], v)
    assert set(got) - set(ref) == {"success_rate_tgt", "j_pi_odd", "j_pi_even"}
    np.testing.assert_allclose(got["success_rate_tgt"],
                               2 * ref["success_rate"] - ref["success_rate_op"])
    np.testing.assert_allclose((got["j_pi_odd"] + got["j_pi_even"]) / 2, ref["j_pi"])
    nat = ea.audit_policy(pol, st, levels, oracle, key, 8, 4, sfl_native=True)
    np.testing.assert_array_equal(nat["s_pvl"], ref["s_pvl"])
    assert np.all((nat["s_learnability_native"] >= 0) & (nat["s_learnability_native"] <= 0.25))
    assert np.all((nat["native_sfl_episodes"] >= 8) & (nat["native_sfl_episodes"] <= 10))
    sub = jax.tree.map(lambda x: x[:5], levels)
    ib = ea.audit_inbuffer(pol, st, sub, np.arange(5.0), np.ones(5), key, 6, 2)
    assert np.all(ib["inbuf_regret"] >= -1e-6) and ib["inbuf_stored"].tolist() == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- statistics


def test_rank_engine_is_spearman_on_the_expanded_sample():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 20, 300).astype(float)
    y = rng.normal(size=300) + 0.1 * x
    for _ in range(5):
        idx = rng.integers(0, 300, 300)
        w = np.bincount(idx, minlength=300)
        assert e5.weighted_spearman(x, y, w) == pytest.approx(float(spearman(x[idx], y[idx])),
                                                              abs=1e-12)
    m = rng.random(300) < 0.6
    assert e5.weighted_spearman(x, y, m.astype(float)) == pytest.approx(
        float(spearman(x[m], y[m])), abs=1e-12)
    assert e5.weighted_spearman(np.ones(300), y) == 0.0           # the constant-score rule
    assert e5.weighted_spearman(y, y) == pytest.approx(1.0)
    assert e5.weighted_spearman(-y, y) == pytest.approx(-1.0)
    assert e5.clip_z(1.0) == pytest.approx(np.arctanh(0.999))


def test_p_values_tost_holm_and_the_trend_test():
    d = np.array([-1.0, 1, 2, 3])
    assert e5.p_two_sided(2.0, d) == pytest.approx(2 * 2 / 5)
    assert e5.p_two_sided(-1.0, d) == pytest.approx(1.0)
    assert e5.p_two_sided(0.0, d) == 1.0
    assert e5.p_two_sided(1.0, np.ones(4000)) == pytest.approx(2 / 4001)
    assert e5.p_one_sided(d, 0.0) == pytest.approx(2 / 5)
    assert e5.p_one_sided(d, 0.0, "less") == pytest.approx(4 / 5)
    assert e5.p_tost(np.array([-0.3, 0.0, 0.1, 0.25]), 0.2) == pytest.approx(2 / 5)
    assert e5.p_tost(np.zeros(9), 0.2) == pytest.approx(1 / 10)
    assert np.isnan(e5.p_two_sided(1.0, np.array([np.nan])))
    adj = e5.holm_fixed([0.01, np.nan, 0.04])
    np.testing.assert_allclose(adj, [0.03, 1.0, 0.08])
    x = np.log10(CHECKPOINTS)
    assert e5.ols_slope(x, 2 * x + 1) == pytest.approx(2.0)
    m, p, n = e5.trend_test([-0.5, -0.4, -0.6, -0.55, -0.45])
    assert m == pytest.approx(-0.5) and p < 1e-3 and n == 5
    assert e5.ci(np.arange(1001.0)) == pytest.approx((25.0, 975.0))


def test_phase_capture_and_labels():
    assert [e5.phase_of(p) for p in (0.0, 0.49, 0.5, 0.89, 0.9, 1.0)] == [
        "early", "early", "mid", "mid", "late", "late"]
    rng = np.random.default_rng(3)
    reg = rng.exponential(1.0, 200)
    ck = e5.Checkpoint(300, {"s": reg, "c": np.ones(200), "r": -reg}, reg, reg / 0.5,
                       np.ones(200, bool), np.full(200, 0.95))
    rc, rc_star, nrc = e5.capture_stats(ck, "s", "abs")
    assert rc == rc_star and nrc == pytest.approx(1.0)
    assert e5.capture_stats(ck, "c", "abs")[0] == pytest.approx(0.1)        # ties share
    assert e5.capture_stats(ck, "c", "abs")[2] == pytest.approx(0.0)
    flat = ck._replace(regret=np.ones(200))
    assert np.isnan(e5.capture_stats(flat, "s", "abs")[2])                 # RC* - 0.1 < 0.02
    assert e5.checkpoint_phase(ck) == "late"
    assert e5.label(0.6, 0.7, 0.3, 0.1, 0.5) == ("faithful", False)
    assert e5.label(0.6, np.nan, 0.3, 0.1, 0.5) == ("faithful", True)
    assert e5.label(0.6, 0.7, 0.3, -0.1, 0.5) == ("uninformative", False)  # CI includes 0
    assert e5.label(0.6, 0.7, 0.3, 0.1, 0.5, holm_sig=False) == ("uninformative", False)
    assert e5.label(0.4, 0.7, 0.3, 0.1, 0.5) == ("intermediate", False)
    assert e5.label(0.6, 0.15, 0.3, 0.1, 0.5) == ("uninformative", False)
    assert e5.label(0.15, 0.9, 0.3, 0.1, 0.5) == ("uninformative", False)


def test_arm_mean_weighs_views_equally():
    runs = [e5.Run(f"r{i}", "dr", v, i, []) for i, v in enumerate([5, 5, 5, 13])]
    vals, ones = np.array([1.0, 2.0, 3.0, 10.0]), np.ones(4)
    assert float(e5.arm_mean(vals, ones, runs, "dr")) == pytest.approx((2.0 + 10.0) / 2)
    assert float(e5.arm_mean(np.array([1.0, np.nan, 3.0, np.nan]), ones, runs, "dr")) == 2.0
    w = np.array([[2, 0, 1, 1]])
    assert float(e5.arm_mean(vals[None], w, runs, "dr")[0]) == pytest.approx(
        ((2 * 1 + 3) / 3 + 10) / 2)
    assert np.isnan(float(e5.arm_mean(vals, ones, runs, "plr")))


# Planted runs: 3 seeds per arm x view (the fewest that reach a phase), 120 mazes (12
# unreachable), 16 checkpoints, phases early (checkpoints 1-6), mid (7-11) and late (12-16).
# A "tracking" score is regret plus small noise; every other score is independent noise.
N = 120


def _faithful(reg, rng):
    return reg + 1e-3 * rng.normal(size=reg.shape)


def _plan(arm, view, k, reg, rng):
    if arm == "dr":
        out = {"pvl": _faithful(reg, rng)}
        if view == 13:
            out["maxmc"] = _faithful(reg, rng)                  # F: view 13 only
        return out
    if arm == "plr":
        out = {"maxmc_native": _faithful(reg, rng)}            # fresh native tracks: G
        if k < 6:
            out["maxmc"] = _faithful(reg, rng)                  # B: early only
        return out
    return {}


def _run(rng, arm, view, seed, plan=_plan, j_star=None, regret_fn=None):
    reach = np.ones(N, bool)
    reach[:12] = False
    js = np.where(reach, rng.uniform(0.59, 0.61, N), 0.0) if j_star is None else j_star
    cks = []
    for k, u in enumerate(CHECKPOINTS):
        reg = (np.where(reach, rng.exponential(0.1, N), 0.0) if regret_fn is None
               else regret_fn(js, rng))
        p = 0.3 if k < 6 else 0.7 if k < 11 else 0.95
        scores = {s: rng.normal(size=N) for s in e5.SCORES}
        scores.update(plan(arm, view, k, reg, rng))
        inbuf = None
        if arm in ("plr", "sfl"):
            breg = rng.exponential(0.1, 40)
            stored = rng.normal(size=40) if arm == "plr" else _faithful(breg, rng)
            inbuf = dict(stored=stored, regret=breg, regret_norm=breg / 0.6,
                         reachable=np.ones(40, bool))
        cks.append(e5.Checkpoint(u, scores, reg, np.where(reach, reg / np.where(reach, js, 1),
                                                          np.nan), reach,
                                 np.where(reach, p, 0.0), inbuf))
    return e5.Run(f"maze-e5-{arm}-v{view}-s{seed}", arm, view, seed, cks)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(2027)
    runs = [_run(rng, a, v, s) for a in ("sfl", "plr", "dr") for s in range(1, 4)
            for v in (13, 5)]
    pt = e5.point(runs)
    boot = e5.bootstrap(runs, n_boot=2000, workers=3)
    return runs, pt, boot, e5.analyse(runs, boot, pt)


def test_point_estimates_are_spearman_per_checkpoint(planted):
    runs, pt, _, _ = planted
    r, k = runs[12], 3                            # a DR run
    ck = r.checkpoints[k]
    for s in ("pvl", "maxmc", "j_star"):
        j = e5.SCORES.index(s)
        assert pt["rho"][12, k, j, 0] == pytest.approx(float(spearman(ck.scores[s], ck.regret)))
        m = ck.reachable
        assert pt["rho"][12, k, j, 1] == pytest.approx(float(spearman(ck.scores[s][m],
                                                                      ck.regret_norm[m])))
    assert pt["phases"][0] == ["early"] * 6 + ["mid"] * 5 + ["late"] * 5


def test_bootstrap_is_deterministic_and_does_not_depend_on_workers(planted):
    runs = planted[0][::4]
    a = e5.bootstrap(runs, n_boot=30, workers=1)
    b = e5.bootstrap(runs, n_boot=30, workers=2)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert a["Z"].shape == (30, len(runs), len(e5.SCORES), 2, 4)
    maze_w, run_w = e5.draw_weights(runs, N, 30)
    assert np.all(maze_w.sum(1) == N)
    for (arm, view) in {(r.arm, r.view) for r in runs}:
        idx = [i for i, r in enumerate(runs) if (r.arm, r.view) == (arm, view)]
        assert np.all(run_w[:, idx].sum(1) == len(idx))


def test_planted_effects_give_the_rows(planted):
    runs, pt, boot, res = planted
    rows = {(r["arm"], r["score"]): r["rows"] for r in res["rows"]}
    assert rows[("dr", "pvl")] == ["A"]
    assert rows[("dr", "maxmc")] == ["F"]
    assert rows[("plr", "maxmc")] == ["B", "G"]
    assert rows[("sfl", "learnability")] == ["C", "G"]
    for key in [("dr", "learnability"), ("plr", "pvl"), ("plr", "learnability"),
                ("sfl", "pvl"), ("sfl", "maxmc")]:
        assert rows[key] == ["C"]
    b = next(r for r in res["rows"] if (r["arm"], r["score"]) == ("plr", "maxmc"))
    assert b["targets"]["B"] == ["abs", "norm"] and b["shift"]["abs"]["estimate"] < -0.9
    assert all(res["arms"][a]["reached"] == ["early", "mid", "late"] for a in e5.ARMS)
    labels = res["arms"]["dr"]["labels"]
    assert {labels[("pvl", t, ph)]["label"] for t in e5.TARGETS for ph in e5.PHASES} == {
        "faithful"}
    assert labels[("pvl", "abs", "late")]["holm_delta"] is True
    assert labels[("pvl", "abs", "mid")]["holm_delta"] is False


def test_planted_families_sizes_and_holm(planted):
    *_, res = planted
    assert len(res["primary"]) == 36 and len(res["secondary"]) == 28
    assert Counter(r["test"] for r in res["primary"]) == {"PF-trend": 18, "PF-late": 18}
    assert Counter(r["test"] for r in res["secondary"]) == {"S1-diff": 3, "S1-tost": 3,
                                                             "S2": 4, "S3": 18}
    sig = {(r["test"], r["arm"], r.get("score"), r.get("target")) for r in res["primary"]
           if r["significant"]}
    assert ("PF-late", "dr", "pvl", "abs") in sig and ("PF-trend", "plr", "maxmc", "abs") in sig
    assert not any(t == "PF-trend" and a == "dr" for t, a, _, _ in sig)
    s2 = {(r["arm"], r["target"]): r for r in res["secondary"] if r["test"] == "S2"}
    assert s2[("plr", "abs")]["estimate"] < -0.8 and s2[("plr", "abs")]["significant"]
    assert s2[("sfl", "norm")]["estimate"] > 0.8 and s2[("sfl", "norm")]["significant"]
    s3 = [r for r in res["secondary"] if r["test"] == "S3" and r["significant"]]
    assert {(r["arm"], r["score"]) for r in s3} == {("dr", "maxmc")}
    assert res["s1_holds"] == {"dr": False, "plr": False, "sfl": False}
    for fam in ("primary", "secondary"):
        p = np.array([r["p"] for r in res[fam]])
        np.testing.assert_allclose([r["p_holm"] for r in res[fam]], e5.holm_fixed(p))


def test_s1_the_maxmc_prediction_on_planted_data():
    """MaxMC tracks J* and regret = J* x u with u independent: high against absolute regret,
    zero against normalised (u)."""
    rng = np.random.default_rng(5)
    js = np.r_[np.zeros(12), rng.uniform(0.1, 0.9, N - 12)]

    def plan(arm, view, k, reg, rng):
        return {"maxmc": js + 1e-3 * rng.normal(size=N)}

    runs = [_run(rng, "dr", v, s, plan, js, lambda j, r: j * r.uniform(0, 1, N))
            for s in range(1, 4) for v in (13, 5)]
    res = e5.analyse(runs, e5.bootstrap(runs, n_boot=2000, workers=3))
    s1 = {r["test"]: r for r in res["secondary"] if r["arm"] == "dr" and r["test"][:2] == "S1"}
    assert s1["S1-diff"]["estimate"] > 0.3 and abs(s1["S1-tost"]["estimate"]) < 0.1
    assert res["s1_holds"]["dr"] is True
    assert not any(a in res["arms"] for a in ("plr", "sfl"))


def test_describe_and_json(planted, mods):
    runs, pt, _, res = planted
    d = e5.describe(runs, pt)
    row = d[runs[12].name][0]
    assert row["phase"] == "early" and row["pvl"]["rho_abs"] > 0.95
    assert 0.9 < row["pvl"]["auroc_top10"] <= 1 and row["pvl"]["p_at_10"] > 0.8
    text = json.dumps(e5.finite_or_none(res), allow_nan=False)
    assert "pvl|abs|late" in text
    md = mods["maze_e5"].decisions_md(res)
    assert "| DR | PVL | A |" in md and "Holm over 36" in md
    assert res["notes"][0].startswith("0 score x checkpoint x target cells")


# --------------------------------------------------------------------------- the timing rule


def test_block_rates_schedule_and_projection(mods):
    rm = mods["round_manifest"]
    assert e5.block_rates(100.0, [160, 170, 180, 190, 200], 100) == {
        "per_update_s": 0.1, "startup_s": 50.0, "blocks": 5}
    with pytest.raises(ValueError):
        e5.block_rates(0.0, [1.0, 2.0], 100)
    jobs = [{"name": str(j)} for j in range(6)]
    lanes = [lane for lane, _, _, _ in rm.schedule(jobs, 2, 2)]
    assert lanes == [0, 1, 2, 3, 0, 1]                      # what lane_makespan assumes
    assert e5.lane_makespan([5, 1, 1, 1, 5, 1], 2, 2) == 10
    rates = {"sfl-v13": {"per_update_s": 0.4, "startup_s": 60.0},
             "dr-v5": {"per_update_s": 0.1, "startup_s": 30.0}}
    p = e5.project_queue(["sfl-v13", "dr-v5", "dr-v5"], rates, 30000, 1, 2,
                         {"sfl-v13": 90.0, "dr-v5": 45.0}, overhead_h=0.25)
    assert p["train_wall_h"] == pytest.approx((60 + 12000 + 30 + 3000) / 3600)
    assert p["train_gpu_h"] == pytest.approx(p["train_wall_h"])
    assert p["audit_gpu_h"] == pytest.approx(16 * 180 / 3600)
    assert p["total_gpu_h"] == pytest.approx(p["train_gpu_h"] + p["audit_gpu_h"] + 0.25)
    assert p["stop_rule_gpu_h"] == pytest.approx(1.2 * p["total_gpu_h"])
    assert "error" in e5.project_queue(["plr-v5"], rates, 30000, 1, 2)


def test_timing_stage_projects_and_picks_the_lane_count(mods, tmp_path, monkeypatch):
    m, R = mods["maze_e5"], mods["maze_e5_runs"]
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(m, "RUNS", runs_dir)
    slow = {"sfl": 0.5, "plr": 0.2, "dr": 0.2}
    for lanes, factor in ((4, 1.0), (8, 1.1)):          # 8 lanes: 10% slower per update
        t = R.runs("timing", lanes_per_gpu=lanes, gpus=4)
        recs = []
        for run in t:
            d = runs_dir / run["name"]
            d.mkdir(parents=True)
            per = slow[run["arm"]] * factor * 100
            ends = [1000 + 40 + per * (b + 1) for b in range(5)]
            (d / "e5-run.json").write_text(json.dumps({"start_unix": 1000.0,
                                                       "block_ends_unix": ends}))
            recs.append({"name": run["name"], "status": "done"})
        (runs_dir / f"maze-e5t{lanes}-manifest-0of1.json").write_text(json.dumps(
            {"gpus": 4, "lanes_per_gpu": lanes, "runs": recs}))
    for run in R.runs("timing-audit"):
        d = runs_dir / run["name"]
        d.mkdir()
        (d / "audit-timing.json").write_text(json.dumps({"arm": run["arm"], "view": run["view"],
                                                         "audit_s": 120.0}))
    m.main(["timing", "--rounds", "maze-e5t4", "maze-e5t8", "--audit-round", "maze-e5at"])
    out = json.loads((runs_dir / "maze-e5-timing.json").read_text())
    assert out["rates"]["4"]["sfl-v13"]["per_update_s"] == pytest.approx(0.5)
    assert out["rates"]["4"]["sfl-v13"]["startup_s"] == pytest.approx(40.0)
    x4, x8 = out["projection"]["4"]["x4"], out["projection"]["8"]["x4"]
    # 4 lanes: two waves (SFL then PLR or DR on the longest lane); 8: one wave, 10% slower
    assert x4["train_wall_h"] == pytest.approx((40 + 15000 + 40 + 6000) / 3600)
    assert x8["train_wall_h"] == pytest.approx((40 + 16500) / 3600)
    assert out["lanes_per_gpu"] == 8
    assert x4["audit_gpu_h"] == pytest.approx(30 * 16 * 120 / 3600)
    assert out["projection"]["4"]["single_gpu_boxes"]["boxes"] == 5


def test_load_audits_reads_the_audit_files_and_checks_them(mods, tmp_path, monkeypatch):
    m = mods["maze_e5"]
    monkeypatch.setattr(m, "RUNS", tmp_path)
    rng = np.random.default_rng(0)
    entry = {"arm": "plr", "view": 5, "seed": 1, "status": "trained",
             "checkpoints": [{"update": u, "sha256": f"{u:064x}"} for u in (300, 400)]}
    manifest = {"runs": {"maze-e5-plr-v5-s1": entry,
                         "maze-e5-dr-v5-s1": {"status": "missing", "checkpoints": []}}}
    d = tmp_path / "maze-e5a-plr-v5-s1"
    d.mkdir()
    files = {}
    for ck in entry["checkpoints"]:
        j_star = np.r_[0.0, rng.uniform(0.3, 0.9, 9)]
        arrays = {f"s_{s}": rng.normal(size=10) for s in e5.SCORES}
        arrays.update(s_pvl_lam0_9=rng.normal(size=10), j_star=j_star, j_pi=j_star * 0.5,
                      j_pi_odd=j_star * 0.4, j_pi_even=j_star * 0.6, regret=j_star * 0.5,
                      regret_norm=np.where(j_star > 0, 0.5, np.nan),
                      success_rate_tgt=np.full(10, 0.6), inbuf_stored=rng.normal(size=4),
                      inbuf_regret=rng.random(4), inbuf_regret_norm=rng.random(4),
                      inbuf_reachable=np.ones(4, bool), inbuf_age=np.ones(4),
                      update=ck["update"], checkpoint_sha256=ck["sha256"])
        path = d / f"u{ck['update']:05d}-fresh.npz"
        files[path.name] = m.write_new(path, m.npz_bytes(**arrays))
    (d / "audit-run.json").write_text(json.dumps({"files": files}))
    names = {"maze-e5-plr-v5-s1": "maze-e5a-plr-v5-s1"}
    runs = m.load_audits(manifest, names)
    assert len(runs) == 1 and runs[0].arm == "plr" and len(runs[0].checkpoints) == 2
    ck = runs[0].checkpoints[1]
    assert ck.update == 400 and ck.reachable.sum() == 9 and set(e5.SCORES) <= set(ck.scores)
    assert ck.inbuf["stored"].shape == (4,) and ck.extra["j_pi_odd"].shape == (10,)
    assert e5.checkpoint_phase(ck) == "mid"
    (d / "audit-run.json").write_text(json.dumps({"files": {**files, "u00300-fresh.npz": "0"}}))
    with pytest.raises(SystemExit, match="does not match"):
        m.load_audits(manifest, names)
    (d / "audit-run.json").write_text(json.dumps({"files": files}))
    entry["checkpoints"][0]["sha256"] = "f" * 64
    with pytest.raises(SystemExit, match="another checkpoint"):
        m.load_audits(manifest, names)
