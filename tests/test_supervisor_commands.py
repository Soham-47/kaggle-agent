from pathlib import Path

from kaggle_agent.notify import commands
from kaggle_agent.state_md import AgentState, save_state
from kaggle_agent.supervisor.commands import SupervisorCommandQueue
from kaggle_agent.supervisor.state import RuntimeLayout


def test_telegram_run_pause_resume_enqueue_supervisor_commands(tmp_path: Path, monkeypatch):
    save_state(AgentState(paused=False, competition="demo"), tmp_path)
    queue = SupervisorCommandQueue(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    monkeypatch.setattr(commands, "_supervisor_queue", lambda root: queue)

    run = commands.handle_command("/run dry", root=tmp_path)
    pause = commands.handle_command("/pause", root=tmp_path)
    resume = commands.handle_command("/resume", root=tmp_path)

    assert run.ok is True
    assert run.start_cycle is False
    assert [item.command for item in queue.pending()] == ["run", "pause", "resume"]
    assert "supervisor" in run.reply.lower()


def test_command_queue_is_durable_and_control_is_safe_by_default(tmp_path: Path):
    queue = SupervisorCommandQueue(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    queue.enqueue("pause")
    assert queue.drain()[0].command == "pause"
    queue.set_paused(True)
    assert queue.paused() is True
    queue.set_paused(False)
    assert queue.paused() is False
