import json
import time
from types import SimpleNamespace
from pathlib import Path

from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore
from kaggle_agent.supervisor.protocol import WorkerExit, WorkerRequest, WorkerResult
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.worker import run_worker


def _revision() -> RuntimeRevision:
    return RuntimeRevision("a" * 40, "b" * 40, "generation-0001")


def test_worker_protocol_round_trips_revision_and_resume_fields():
    request = WorkerRequest("w1", "generation-0001", "demo", None, "observe", "CODE", "inc-1", _revision())
    restored = WorkerRequest.from_dict(request.to_dict())
    assert restored == request
    result = WorkerResult("w1", "generation-0001", WorkerExit.RECOVERABLE_FAILURE.value, None, None, "CODE", "inc-1", "stage failure", _revision())
    assert WorkerResult.from_dict(result.to_dict()).revision == _revision()


def test_worker_request_round_trips_explicit_dry_run_intent():
    request = WorkerRequest(
        "w1", "generation-0001", "demo", None, "observe", None, None,
        _revision(), False,
    )
    assert WorkerRequest.from_dict(request.to_dict()).dry_run is False


def test_heartbeat_store_writes_and_detects_stale(tmp_path: Path):
    store = HeartbeatStore(tmp_path)
    store.write(Heartbeat("w1", 123, "generation-0001", "cycle", "CODE", "progress", time.time()))
    assert json.loads((tmp_path / "workers" / "w1" / "heartbeat.json").read_text())["stage"] == "CODE"
    assert store.is_fresh("w1", timeout_seconds=10) is True
    assert store.is_fresh("w1", timeout_seconds=0) is False


def test_heartbeat_store_missing_is_stale(tmp_path: Path):
    assert HeartbeatStore(tmp_path).is_fresh("missing", timeout_seconds=10) is False


def test_worker_runs_exact_requested_generation_root(tmp_path: Path, monkeypatch):
    state_root = tmp_path / "state"
    generation = tmp_path / "generation"
    generation.mkdir()
    request_path = state_root / "workers" / "w1" / "request.json"
    request_path.parent.mkdir(parents=True)
    request = WorkerRequest("w1", "generation-1", "demo", None, "observe", None, None, _revision(), True)
    request_path.write_text(json.dumps({**request.to_dict(), "generation_path": str(generation)}), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state_root))
    seen = {}

    def fake_run_daily(*args, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(hard_errors=[], experiment_id="exp-1")

    monkeypatch.setattr("kaggle_agent.orchestrator.run_daily", fake_run_daily)
    assert run_worker(request_path) == 0
    assert seen["root"] == generation.resolve()
    result = json.loads((state_root / "workers" / "w1" / "result.json").read_text())
    assert result["status"] == WorkerExit.SUCCESS.value
