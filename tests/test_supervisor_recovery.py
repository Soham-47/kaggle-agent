import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from kaggle_agent.autonomy.outbox import ExternalActionOutbox
from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.audit import AuditLog
from kaggle_agent.supervisor.classifier import FailureClass, classify_after_reconciliation
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeGeneration, RuntimeRevision
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore
from kaggle_agent.supervisor.loop import Supervisor
from kaggle_agent.supervisor.promote import GenerationPromotion
from kaggle_agent.supervisor.recovery import SupervisorRecovery
from kaggle_agent.supervisor.resume import ResumeRequest
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


def test_uncertain_external_action_stays_pending_before_classification(tmp_path: Path):
    outbox = ExternalActionOutbox(tmp_path)
    item = outbox.enqueue(action="kernel_push", idempotency_key="k", payload={"kernel_ref": "u/k"})
    incident = Incident(
        incident_id="i", worker_id="w", generation_id="g", cycle_id=None, experiment_id=None,
        competition="c", stage="KERNEL_TRAIN", stage_attempt=1,
        revision=RuntimeRevision("a", "b", "g"),
        outcome_state="recoverable_failure", exception_type="TimeoutError", exception_message="timeout",
        traceback=None, failure_signature="s", evidence=(), artifacts=(), external_job=item.action_id,
        kernel_ref="u/k", candidate_csv=None, recent_logs=(), created_at="now",
    )
    result = classify_after_reconciliation(incident, outbox, lambda current: current)
    assert result.failure_class is FailureClass.PENDING_EXTERNAL


def test_audit_log_tolerates_partial_final_json_line(tmp_path: Path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("start", worker_id="w")
    log.path.write_text(log.path.read_text() + '{"partial":', encoding="utf-8")
    assert len(log.read()) == 1


def test_restart_adopts_only_owned_live_worker_with_fresh_heartbeat(tmp_path: Path):
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path / "repo", tmp_path / "state"))
    state.write_json("workers/w1/metadata.json", {"pid": os.getpid(), "worker_id": "w1", "supervisor_token": "owner", "generation_id": "g1"})
    HeartbeatStore(state.layout.state_root).write(Heartbeat("w1", os.getpid(), "g1", "c1", "CODE", "progress", time.time()))
    item = SupervisorRecovery(state).inspect_worker("w1", timeout_seconds=30, owner_token="owner")
    assert item.action == "ADOPT"
    assert item.owned is True


def test_restart_marks_dead_worker_interrupted_without_result(tmp_path: Path):
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path / "repo", tmp_path / "state"))
    state.write_json("workers/w1/metadata.json", {"pid": 99999999, "worker_id": "w1", "supervisor_token": "owner", "generation_id": "g1"})
    recovered = SupervisorRecovery(state).recover_workers(timeout_seconds=30, owner_token="owner")
    assert recovered[0].action == "START_OR_RESUME"
    assert state.read_json("workers/w1/result.json")["status"] == "INTERRUPTED"


def test_sigkill_worker_is_not_adopted_while_zombie(tmp_path: Path):
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path / "repo", tmp_path / "state"))
    process = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(30)"))
    state.write_json(
        "workers/w1/metadata.json",
        {
            "pid": process.pid,
            "worker_id": "w1",
            "supervisor_token": "owner",
            "generation_id": "g1",
        },
    )
    HeartbeatStore(state.layout.state_root).write(
        Heartbeat("w1", process.pid, "g1", "c1", "CODE", "progress", time.time())
    )
    process.kill()
    time.sleep(0.1)
    recovery = SupervisorRecovery(state)
    try:
        item = recovery.inspect_worker("w1", timeout_seconds=30, owner_token="owner")
        assert item.action == "START_OR_RESUME"
    finally:
        process.wait(timeout=5)


def test_interrupted_promotion_recovers_to_old_or_new_pointer(tmp_path: Path):
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path / "repo", tmp_path / "state"))
    old = RuntimeGeneration("g1", RuntimeRevision("a", "b", "g1"), str(tmp_path / "old"))
    new = RuntimeGeneration("g2", RuntimeRevision("c", "d", "g2"), str(tmp_path / "new"))
    promotion = GenerationPromotion(state)
    state.write_json("promotion.json", {"status": "PREPARED", "old_generation": "g1", "new_generation": "g2"})
    assert promotion.recover_interrupted(old, new) == "ROLLED_BACK"
    assert state.read_json("active-generation.json")["generation_id"] == "g1"
    state.write_json("promotion.json", {"status": "PREPARED", "old_generation": "g1", "new_generation": "g2"})
    state.write_json("active-generation.json", new.to_dict())
    assert promotion.recover_interrupted(old, new) == "COMMITTED"


def test_restart_resumes_promoted_worker_with_same_durable_request(tmp_path: Path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": True, "mode": "auto_safe"}}),
        encoding="utf-8",
    )
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    settings = load_settings(tmp_path)
    old = RuntimeGeneration("g1", RuntimeRevision("a", "b", "g1"), str(tmp_path), created_at="now")
    new = RuntimeGeneration("g2", RuntimeRevision("c", "d", "g2"), str(tmp_path), parent_generation="g1", repair_id="r1", created_at="now")
    GenerationStore(state).save(old)
    GenerationStore(state).save(new)
    state.write_json("active-generation.json", new.to_dict())
    resume = ResumeRequest("c1", "i1", "g1", "g2", "CODE", "CODE", ("RESEARCH",), ("CODE",), (), (("CODE", 1),))
    state.write_json("resume-requests/i1.json", resume.to_dict())
    state.write_json(
        "promotion.json",
        {
            "schema_version": 2,
            "status": "PROMOTED",
            "old_generation": "g1",
            "new_generation": "g2",
            "resume_request_path": "resume-requests/i1.json",
            "replacement_worker_id": "worker-replacement",
        },
    )
    supervisor = Supervisor(settings, tmp_path)
    state.write_json(
        "workers/worker-replacement/metadata.json",
        {"pid": 99999999, "worker_id": "worker-replacement", "generation_id": "g2", "supervisor_token": supervisor.lock.owner_token},
    )
    requests = []

    class FakeProcess:
        pid = 123

        def wait(self):
            return 0

    def start(_launcher, request, *, cwd=None):
        requests.append(request)
        return FakeProcess()

    monkeypatch.setattr("kaggle_agent.supervisor.loop.WorkerLauncher.start", start)
    result = supervisor._resume_promoted_if_needed("demo", "auto_safe", 30)

    assert result is not None
    assert result.status == "WORKER_STARTED"
    assert len(requests) == 1
    assert requests[0].worker_id == "worker-replacement"
    assert requests[0].resume_request == resume


def test_failed_resumed_worker_rolls_back_promoted_generation(tmp_path: Path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": True, "mode": "auto_safe"}}),
        encoding="utf-8",
    )
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    settings = load_settings(tmp_path)
    old = RuntimeGeneration("g1", RuntimeRevision("a", "b", "g1"), str(tmp_path), created_at="now")
    new = RuntimeGeneration("g2", RuntimeRevision("c", "d", "g2"), str(tmp_path), parent_generation="g1", repair_id="r1", created_at="now")
    GenerationStore(state).save(old)
    GenerationStore(state).save(new)
    state.write_json("active-generation.json", new.to_dict())
    resume = ResumeRequest("c1", "i1", "g1", "g2", "CODE", "CODE", (), ("CODE",), (), (("CODE", 1),))
    state.write_json("resume-requests/i1.json", resume.to_dict())
    state.write_json(
        "promotion.json",
        {
            "schema_version": 2,
            "status": "PROMOTED",
            "old_generation": "g1",
            "new_generation": "g2",
            "resume_request_path": "resume-requests/i1.json",
            "replacement_worker_id": "worker-replacement",
        },
    )
    state.write_json(
        "workers/worker-replacement/result.json",
        {"status": "FATAL", "exit_reason": "startup import failed"},
    )
    result = Supervisor(settings, tmp_path)._resume_promoted_if_needed("demo", "auto_safe", 30)

    assert result is not None
    assert result.status == "ROLLED_BACK"
    assert state.read_json("active-generation.json")["generation_id"] == "g1"
    assert state.read_json("promotion.json")["status"] == "ROLLED_BACK"


def test_successful_resumed_worker_closes_pending_promotion(tmp_path: Path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": True, "mode": "auto_safe"}}),
        encoding="utf-8",
    )
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    supervisor = Supervisor(load_settings(tmp_path), tmp_path)
    transaction = {"schema_version": 2, "status": "PROMOTED", "old_generation": "g1", "new_generation": "g2"}
    state.write_json("promotion.json", transaction)

    result = supervisor._settle_promoted_result(transaction, "worker-replacement", {"status": "SUCCESS"})

    assert result.status == "SUCCESS"
    assert state.read_json("promotion.json")["status"] == "RESUMED"


def test_restart_blocks_ambiguous_replacement_launch_without_pid(tmp_path: Path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": True, "mode": "auto_safe"}}),
        encoding="utf-8",
    )
    state_root = tmp_path / "state"
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state_root))
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, state_root))
    settings = load_settings(tmp_path)
    generation = RuntimeGeneration("g2", RuntimeRevision("c", "d", "g2"), str(tmp_path), created_at="now")
    GenerationStore(state).save(generation)
    state.write_json("resume-requests/i1.json", ResumeRequest("c1", "i1", "g1", "g2", "CODE", "CODE", (), ("CODE",), ()).to_dict())
    state.write_json(
        "promotion.json",
        {"schema_version": 2, "status": "PROMOTED", "old_generation": "g1", "new_generation": "g2", "resume_request_path": "resume-requests/i1.json", "replacement_worker_id": "worker-replacement"},
    )
    state.write_json("workers/worker-replacement/metadata.json", {"pid": None, "launch_state": "STARTING"})

    result = Supervisor(settings, tmp_path)._resume_promoted_if_needed("demo", "auto_safe", 30)

    assert result is not None
    assert result.status == "RECOVERY_BLOCKED"
