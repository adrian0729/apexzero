"""The terminal dashboard's parsing helpers (scripts/dashboard.py)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "dashboard", Path(__file__).resolve().parent.parent / "scripts" / "dashboard.py")
dashboard = importlib.util.module_from_spec(_spec)
sys.modules["dashboard"] = dashboard          # dataclasses look their module up
_spec.loader.exec_module(dashboard)


@pytest.mark.parametrize("s, sec", [("00:07", 7), ("12:34", 754), ("1:02:03", 3723),
                                    ("2-01:00:00", 2 * 86400 + 3600)])
def test_parse_etime(s, sec):
    assert dashboard.parse_etime(s) == sec


def test_parse_queue_joins_continuations_and_skips_function_body():
    text = """run() {
  local name="$1"; shift
  uv run python "$@" > "runs/$name.log" 2>&1
}
run a-job scripts/train.py --tracks X --steps 200e6 \\
    --name a-job --seed 2
# comment
run b-job scripts/train_dr.py --steps 1e6 --name b-job
"""
    jobs = dashboard.parse_queue(text)
    assert [j.name for j in jobs] == ["a-job", "b-job"]
    assert jobs[0].script == "scripts/train.py"
    assert "--seed 2" in jobs[0].args
    assert dashboard._steps_arg(jobs[0].args) == 200e6


def test_tail_rows_reads_last_rows_of_large_csv(tmp_path):
    p = tmp_path / "metrics.csv"
    rows = ["timesteps,wall_s,sps"] + [f"{i * 1000},{i},{1000.0}" for i in range(5000)]
    p.write_text("\n".join(rows) + "\n")
    last = dashboard.tail_rows(p, 2)
    assert [r["timesteps"] for r in last] == [4998000.0, 4999000.0]
    small = tmp_path / "small.csv"
    small.write_text("timesteps,sps\n10,5\n")
    assert dashboard.tail_rows(small, 2) == [{"timesteps": 10.0, "sps": 5.0}]


def test_job_state_running_and_done(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUNS", tmp_path)
    job = dashboard.Job("j", "scripts/train.py", "--steps 1000 --name j")
    run = tmp_path / "j"
    run.mkdir()
    (run / "config.json").write_text('{"ppo": {"total_timesteps": 1000, "num_envs": 10, "num_steps": 10}}')
    (run / "metrics.csv").write_text("timesteps,wall_s,sps\n200,10,20\n400,20,20\n")
    proc = dashboard.Proc(1, 0, 60, 500.0, 100.0, "python scripts/train.py --name j --seed 1")
    st = dashboard.job_state(job, [proc], True, {})
    assert st.status == "running" and st.steps == 400 and st.sps == 20 and st.eta_s == 30
    (run / "metrics.csv").write_text("timesteps,wall_s,sps\n900,45,20\n1000,50,20\n")
    assert dashboard.job_state(job, [], True, {}).status == "done"
    other = dashboard.Job("j2", "scripts/train.py", "--steps 1000 --name j2")
    st = dashboard.job_state(other, [proc], True, {"scripts/train.py": [100.0]})
    assert st.status == "pending" and st.estimated and st.eta_s == 10


def test_fit_keeps_ansi_and_truncates_visible_width():
    s = dashboard.GREEN + "abcdef" + dashboard.RESET + "ghij"
    out = dashboard.fit(s, 4)
    assert dashboard.ANSI.sub("", out) == "abcd"


def test_stage_state_reads_the_latest_stage_run(tmp_path):
    log = tmp_path / "psychophysics-keys.log"
    log.write_text(
        "2026-10-19 10:13:40 start keys: 0/1 (P0, 9 workers)\n"
        "2026-10-19 10:20:00 done keys: 1/1 in 6.0 min\n"
        "2026-10-19 21:40:00 start keys: 0/831 (P2, 10 workers)\n"
        "2026-10-19 21:41:08 384/831 phase cell 109: verified (225 CPU-s)\n"
        "2026-10-19 22:09:26 acceptance: V1 PASS, 2141 keys\n")
    st = dashboard.stage_state(log, running=True)
    assert (st.stage, st.status, st.done, st.total) == ("keys", "running", 384, 831)
    assert st.summary == "acceptance: V1 PASS, 2141 keys" and st.when == "22:09"
    with open(log, "a") as fh:
        fh.write("2026-10-19 22:09:26 done keys: 831/831 in 35.4 min\n")
    st = dashboard.stage_state(log, running=False)
    assert (st.status, st.done, st.total) == ("done", 831, 831)
    with open(log, "a") as fh:
        fh.write("2026-10-19 23:00:00 start keys: 0/9 (P0)\n"
                 "2026-10-19 23:00:01 failed keys: 1/9 in 0.0 min (ValueError: boom)\n")
    st = dashboard.stage_state(log, running=False)
    assert st.status == "failed" and st.summary == "in 0.0 min (ValueError: boom)"
    log.write_text("2026-10-19 23:00:00 start keys: 0/9 (P0)\n")
    assert dashboard.stage_state(log, running=False).status == "stopped"
