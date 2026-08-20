import os
import signal
import subprocess
import sys
from pathlib import Path

from kaggle_agent.autonomy.outbox import ExternalAction, ExternalActionOutbox
from kaggle_agent.supervisor.audit import AuditLog
from kaggle_agent.supervisor.classifier import FailureClass, classify_after_reconciliation
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.generation import RuntimeGeneration
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore
from kaggle_agent.supervisor.promote import GenerationPromotion
from kaggle_agent.supervisor.recovery import SupervisorRecovery
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
import time


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
