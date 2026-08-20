import json
import time
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
