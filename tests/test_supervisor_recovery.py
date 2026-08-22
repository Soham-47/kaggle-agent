from pathlib import Path

from kaggle_agent.autonomy.outbox import ExternalAction, ExternalActionOutbox
from kaggle_agent.supervisor.audit import AuditLog
from kaggle_agent.supervisor.classifier import FailureClass, classify_after_reconciliation
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.generation import RuntimeRevision


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
