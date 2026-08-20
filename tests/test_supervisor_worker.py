import json
import time
from types import SimpleNamespace
from pathlib import Path

from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore
from kaggle_agent.supervisor.protocol import WorkerExit, WorkerRequest, WorkerResult
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.resume import ResumeRequest


def _revision() -> RuntimeRevision:
    return RuntimeRevision("a" * 40, "b" * 40, "generation-0001")


def test_worker_protocol_round_trips_revision_and_resume_fields():
    resume = ResumeRequest("cycle-1", "inc-1", "generation-0001", "generation-0002", "KERNEL_TRAIN", "KERNEL_TRAIN", ("RESEARCH", "PLAN", "CODE"), ("KERNEL_TRAIN", "VALIDATE_SUB"), ("kernel-1",), (("KERNEL_TRAIN", 1), ("VALIDATE_SUB", 1)))
    request = WorkerRequest("w1", "generation-0001", "demo", "cycle-1", "observe", "CODE", "inc-1", _revision(), resume)
    restored = WorkerRequest.from_dict(request.to_dict())
    assert restored == request
    assert restored.resume_request is not None
    assert restored.resume_request.epoch_for("KERNEL_TRAIN") == 1
    result = WorkerResult("w1", "generation-0001", WorkerExit.RECOVERABLE_FAILURE.value, None, None, "CODE", "inc-1", "stage failure", _revision())
    assert WorkerResult.from_dict(result.to_dict()).revision == _revision()


def test_heartbeat_store_writes_and_detects_stale(tmp_path: Path):
    store = HeartbeatStore(tmp_path)
    store.write(Heartbeat("w1", 123, "generation-0001", "cycle", "CODE", "progress", time.time()))
    assert json.loads((tmp_path / "workers" / "w1" / "heartbeat.json").read_text())["stage"] == "CODE"
    assert store.is_fresh("w1", timeout_seconds=10) is True
    assert store.is_fresh("w1", timeout_seconds=0) is False


def test_heartbeat_store_missing_is_stale(tmp_path: Path):
    assert HeartbeatStore(tmp_path).is_fresh("missing", timeout_seconds=10) is False


def test_worker_passes_resume_request_to_orchestrator(tmp_path: Path, monkeypatch):
    state_root = tmp_path / "state"
    request = WorkerRequest(
        "w1", "generation-0002", "demo", "cycle-1", "repair_only", "KERNEL_TRAIN", "inc-1",
        _revision(),
        ResumeRequest(
            "cycle-1", "inc-1", "generation-0001", "generation-0002", "KERNEL_TRAIN",
            "KERNEL_TRAIN", ("RESEARCH", "PLAN"), ("KERNEL_TRAIN",), (), (("KERNEL_TRAIN", 1),),
        ),
    )
    request_path = state_root / "workers" / "w1" / "request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    captured = {}

    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state_root))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("kaggle_agent.config.load_settings", lambda root: SimpleNamespace(dry_run=True))
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.run_daily",
        lambda *args, **kwargs: captured.update(kwargs) or SimpleNamespace(hard_errors=[], experiment_id="cycle-1"),
    )

    from kaggle_agent.supervisor.worker import run_worker

    assert run_worker(request_path) == 0
    assert captured["resume_request"] == request.resume_request
    assert json.loads((state_root / "workers" / "w1" / "result.json").read_text())["status"] == "SUCCESS"


def test_worker_launcher_adds_generation_source_to_pythonpath(tmp_path: Path, monkeypatch):
    from kaggle_agent.supervisor.state import RuntimeLayout
    from kaggle_agent.supervisor.worker import WorkerLauncher

    calls = []

    class _Process:
        pid = 123

    monkeypatch.setattr("kaggle_agent.supervisor.worker.subprocess.Popen", lambda *args, **kwargs: calls.append((args, kwargs)) or _Process())
    request = WorkerRequest("w1", "g1", "demo", None, "observe", None, None, _revision())
    generation = tmp_path / "generation"
    generation.mkdir()
    WorkerLauncher(RuntimeLayout.for_repo(tmp_path, tmp_path / "state")).start(request, cwd=generation)
    environment = calls[0][1]["env"]
    assert environment["PYTHONPATH"].split(":")[0] == str(generation / "src")
