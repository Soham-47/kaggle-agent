from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.classifier import FailureClass, classify_failure
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident, IncidentStore, failure_signature, sanitize_text
from kaggle_agent.supervisor.spec import RepairSpec


def test_sanitizer_removes_credentials_and_signature_is_stable():
    text = "Authorization: Bearer abc123 API_KEY=secret-value File /tmp/run.py, line 42"
    clean = sanitize_text(text)
    assert "abc123" not in clean and "secret-value" not in clean
    assert failure_signature("File /tmp/a.py, line 1: NameError: stale") == failure_signature("File /other/b.py, line 9: NameError: stale")


def test_incident_store_persists_structured_sanitized_incident(tmp_path: Path):
    incident = Incident.from_outcome(
        worker_id="worker-1", generation_id="generation-0001", competition="demo",
        outcome=StageOutcome.failure("CODE", "NameError: token=secret", "sig"),
        stage_attempt=2, revision=RuntimeRevision("a", "b", "generation-0001"),
        traceback="Authorization: Bearer secret\nFile src/a.py, line 4",
    )
    path = IncidentStore(tmp_path).save(incident)
    assert path == tmp_path / "incidents" / "sig.json"
    assert "secret" not in path.read_text(encoding="utf-8")


def test_classifier_prefers_external_and_known_kernel_repairs():
    base = dict(worker_id="w", generation_id="g", competition="c", cycle_id=None, experiment_id=None,
                stage_attempt=1, revision=RuntimeRevision("a", "b", "g"), exception_type="RuntimeError",
                traceback=None, evidence=(), artifacts=(), external_job=None, kernel_ref=None,
                candidate_csv=None, recent_logs=(), created_at="now")
    transient = Incident(incident_id="i", stage="KERNEL_TRAIN", outcome_state="recoverable_failure",
                         exception_message="HTTP 503", failure_signature="s", **base)
    assert classify_failure(transient).failure_class is FailureClass.TRANSIENT_EXTERNAL
    pin = Incident(incident_id="i2", stage="KERNEL_TRAIN", outcome_state="recoverable_failure",
                   exception_message="Model instance version-number is required", failure_signature="s2", **base)
    assert classify_failure(pin).failure_class is FailureClass.KNOWN_RUNTIME_REPAIR


def test_repair_spec_serializes_markdown(tmp_path: Path):
    spec = RepairSpec(
        repair_id="r1", incident_id="i1", base_generation="g1",
        base_revision=RuntimeRevision("a", "b", "g1"), title="fix",
        failed_stage="CODE", observed_failure="NameError", root_cause="missing import",
        current_behavior="fails", expected_behavior="works", likely_files=("src/a.py",),
        reproduction_mode="EXISTING_TEST_REPRO", reproduction_commands=("uv run pytest",),
        invariants=("approval remains required",), forbidden_changes=(".env",),
        required_tests=("tests/test_a.py",), verification_commands=("uv run pytest -q",),
        allowed_paths=("src", "tests"), max_changed_source_files=8,
        max_changed_test_files=5, max_changed_lines=500, proposed_resume_stage="CODE",
        risk_level="low", acceptance_criteria=("test passes",),
    )
    paths = spec.save(tmp_path)
    assert paths[0].is_file() and paths[1].is_file()
    assert RepairSpec.from_dict(spec.to_dict()) == spec
