"""Telegram command contract — every advertised command must behave."""

from __future__ import annotations

from pathlib import Path

from kaggle_agent.notify.commands import HELP, handle_command, process_updates
from kaggle_agent.notify.run_agent import start_agent_cycle, start_agent_cycle_async
from kaggle_agent.state_md import AgentState, RunLock, load_state, save_state
from kaggle_agent.submit.pending import load_pending, request_approval


def test_help_and_start_list_advertised_commands():
    for cmd in ("/help", "/start"):
        r = handle_command(cmd)
        assert r.ok
        assert r.start_cycle is False
        for needle in (
            "/run",
            "/run live",
            "/status",
            "/yes",
            "/no",
            "/approve",
            "/reject",
            "/pause",
            "/resume",
            "/budget",
            "/help",
        ):
            assert needle in r.reply, f"{cmd} missing {needle}"
        assert HELP in r.reply or r.reply == HELP


def test_status_reports_phase_budget_and_pending(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(
        AgentState(
            paused=False,
            competition="rsna_knee",
            phase="IDLE",
            public_best="0.500",
            proposals_used="1",
            max_proposals="2",
            budget_date="2026-08-13",
            note="ready",
        ),
        tmp_path,
    )
    request_approval(
        exp_id="exp-status",
        csv_path="/tmp/s.csv",
        competition="rsna-knee-abnormality-detection",
        root=tmp_path,
    )
    r = handle_command("/status", root=tmp_path)
    assert r.ok
    assert "Phase: IDLE" in r.reply
    assert "Paused: no" in r.reply
    assert "rsna_knee" in r.reply
    assert "1 used / 2 max" in r.reply
    assert "0.500" in r.reply
    assert "exp-status" in r.reply
    assert "pending" in r.reply.lower()


def test_yes_and_ok_approve_latest(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    request_approval(exp_id="e1", csv_path="/tmp/a.csv", competition="rsna", root=tmp_path)
    r = handle_command("/ok", root=tmp_path)
    assert r.ok
    assert load_pending(tmp_path).status == "approved"
    assert "/run live" in r.reply


def test_no_rejects_latest(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    request_approval(exp_id="e2", csv_path="/tmp/b.csv", competition="rsna", root=tmp_path)
    r = handle_command("/no", root=tmp_path)
    assert r.ok
    assert load_pending(tmp_path).status == "rejected"
    assert load_state(tmp_path).pending_approve == "none"


def test_yes_without_pending_does_not_approve(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=False), tmp_path)
    r = handle_command("/yes", root=tmp_path)
    assert r.ok is False
    assert load_pending(tmp_path).status in {"none", ""}
    assert "pending" in r.reply.lower() or "could not" in r.reply.lower()


def test_approve_wrong_exp_fails(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    request_approval(exp_id="real-exp", csv_path="/tmp/c.csv", competition="rsna", root=tmp_path)
    r = handle_command("/approve other-exp", root=tmp_path)
    assert r.ok is False
    assert load_pending(tmp_path).status == "pending"


def test_run_bot_suffix_and_live_flag(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=False), tmp_path)
    dry = handle_command("/run@KaggleAgentBot", root=tmp_path)
    assert dry.ok and dry.start_cycle and dry.cycle_dry_run is True
    live = handle_command("/run@KaggleAgentBot live", root=tmp_path)
    assert live.ok and live.start_cycle and live.cycle_dry_run is False


def test_run_while_paused_does_not_start_cycle(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=True), tmp_path)
    r = handle_command("/run", root=tmp_path)
    assert r.ok is False
    assert r.start_cycle is False
    assert "pause" in r.reply.lower() or "/resume" in r.reply


def test_unknown_and_non_command(tmp_path: Path):
    r = handle_command("/foobar")
    assert r.ok is False
    assert "/help" in r.reply
    r2 = handle_command("hello")
    assert r2.ok is False
    assert r2.start_cycle is False


def test_process_updates_run_live_sets_start_flags(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=False), tmp_path)
    updates = [
        {"update_id": 1, "message": {"chat": {"id": 42}, "text": "/run live"}},
    ]
    out = process_updates(updates, root=tmp_path, allowed_chat_id="42")
    assert len(out) == 1
    assert out[0].ok and out[0].start_cycle and out[0].cycle_dry_run is False
    assert out[0].chat_id == "42"


def test_process_updates_ignores_non_commands(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    updates = [
        {"update_id": 1, "message": {"chat": {"id": 1}, "text": "not a command"}},
        {"update_id": 2, "edited_message": {"chat": {"id": 1}, "text": "/help"}},
    ]
    out = process_updates(updates, root=tmp_path, allowed_chat_id="1")
    assert len(out) == 1
    assert out[0].ok and " /run" in out[0].reply or "/run" in out[0].reply


def test_start_agent_cycle_refuses_paused(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=True, competition="rsna_knee"), tmp_path)
    res = start_agent_cycle(root=tmp_path, dry_run=True, background=False)
    assert res.ok is False
    assert "paused" in res.message.lower()


def test_start_agent_cycle_async_refuses_lock(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    save_state(AgentState(paused=False, competition="rsna_knee"), tmp_path)
    lock = RunLock(tmp_path)
    assert lock.acquire()
    try:
        res = start_agent_cycle_async(root=tmp_path, dry_run=True)
        assert res.ok is False
        assert "already running" in res.message.lower()
    finally:
        lock.release()
