from pathlib import Path

from kaggle_agent.state_md import AgentState, RunLock, load_state, parse_kv_markdown, save_state


def test_parse_kv_markdown():
    d = parse_kv_markdown("# t\n\n- phase: PLAN\n- paused: false\n")
    assert d["phase"] == "PLAN"
    assert d["paused"] == "false"


def test_save_load_state(tmp_path: Path):
    save_state(AgentState(phase="RESEARCH", competition="rsna_knee"), tmp_path)
    loaded = load_state(tmp_path)
    assert loaded.phase == "RESEARCH"
    assert loaded.competition == "rsna_knee"


def test_run_lock(tmp_path: Path):
    a, b = RunLock(tmp_path), RunLock(tmp_path)
    assert a.acquire() is True
    assert b.acquire() is False
    a.release()
    assert b.acquire() is True
    b.release()
