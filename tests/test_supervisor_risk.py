from pathlib import Path

import pytest

from kaggle_agent.config import ConfigError, load_settings
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.risk import (
    ExternalStateCertainty,
    RepairRiskTier,
    ReproductionStrength,
    evaluate_repair_risk,
)
from kaggle_agent.supervisor.spec import RepairSpec


def _incident(stage: str = "CODE", *, external_job: str | None = None) -> Incident:
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    return Incident(
        "incident-1", "worker-1", "generation-0001", "cycle-1", None, "demo", stage, 1,
        revision, "recoverable_failure", "NameError", "NameError: broken", None, "signature-1",
        (), (), external_job, None, None, (), "now",
    )


def _spec(*, stage: str = "CODE", paths: tuple[str, ...] = ("src/example/parser.py",),
          mode: str = "EXISTING_TEST_REPRO", allowed: tuple[str, ...] | None = None) -> RepairSpec:
    incident = _incident(stage)
    return RepairSpec(
        "repair-1", incident.incident_id, incident.generation_id, incident.revision,
        "repair", stage, "NameError", "undefined local", "raises", "returns", paths,
        mode, ("uv run pytest -q tests/test_parser.py",), (), (),
        ("tests/test_parser.py",), ("uv run pytest -q tests/test_parser.py",),
        allowed or paths, 4, 2, 250, stage, "low", ("focused test passes",),
    )


def test_existing_deterministic_parser_repair_is_low_and_promotable():
    decision = evaluate_repair_risk(
        _incident(),
        FailureClassification(FailureClass.CODE_DEFECT, 0.95, True),
        _spec(),
        changed_paths=["src/example/parser.py"],
        changed_lines=20,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )

    assert decision.tier is RepairRiskTier.LOW
    assert decision.reproduction_strength is ReproductionStrength.EXISTING_DETERMINISTIC_TEST
    assert decision.candidate_generation_allowed is True
    assert decision.automatic_promotion_allowed is True


def test_replay_logic_is_high_but_candidate_only():
    incident = _incident("KERNEL_TRAIN")
    spec = _spec(stage="KERNEL_TRAIN", paths=("src/kaggle_agent/supervisor/resume.py",))
    decision = evaluate_repair_risk(
        incident,
        FailureClassification(FailureClass.CODE_DEFECT, 0.95, True),
        spec,
        changed_paths=["src/kaggle_agent/supervisor/resume.py"],
        changed_lines=20,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )

    assert decision.tier is RepairRiskTier.HIGH
    assert decision.candidate_generation_allowed is True
    assert decision.automatic_promotion_allowed is False
    assert decision.authority_required is True


def test_external_action_identity_is_prohibited_even_for_one_line_diff():
    spec = _spec(paths=("src/kaggle_agent/autonomy/outbox.py",))
    decision = evaluate_repair_risk(
        _incident(),
        FailureClassification(FailureClass.CODE_DEFECT, 0.99, True),
        spec,
        changed_paths=["src/kaggle_agent/autonomy/outbox.py"],
        changed_lines=2,
        diff="+def submission_key(value):\n+    return external_action_key(value)\n",
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )

    assert decision.tier is RepairRiskTier.PROHIBITED
    assert decision.candidate_generation_allowed is False
    assert decision.automatic_promotion_allowed is False


def test_ambiguous_external_state_blocks_automatic_repair_and_promotion():
    decision = evaluate_repair_risk(
        _incident("KERNEL_TRAIN", external_job="action-1"),
        FailureClassification(FailureClass.CODE_DEFECT, 0.95, True),
        _spec(stage="KERNEL_TRAIN"),
        changed_paths=["src/example/parser.py"],
        changed_lines=20,
        external_state=ExternalStateCertainty.AMBIGUOUS,
    )

    assert decision.automatic_promotion_allowed is False
    assert decision.candidate_generation_allowed is False
    assert decision.external_reconciliation_required is True


def test_medium_scope_is_not_rejected_only_for_exceeding_old_canary_size():
    paths = tuple(f"competitions/demo/pipeline/file{i}.py" for i in range(4))
    decision = evaluate_repair_risk(
        _incident(),
        FailureClassification(FailureClass.CODE_DEFECT, 0.95, True),
        _spec(paths=paths, allowed=paths),
        changed_paths=list(paths),
        changed_lines=400,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )

    assert decision.tier is RepairRiskTier.MEDIUM
    assert decision.automatic_promotion_allowed is True
    assert decision.max_changed_lines >= 400
    assert "old canary" not in " ".join(decision.reasons).lower()


def test_external_and_authentication_classes_are_not_code_repairable():
    for failure_class in (FailureClass.TRANSIENT_EXTERNAL, FailureClass.PENDING_EXTERNAL, FailureClass.AUTHENTICATION):
        decision = evaluate_repair_risk(
            _incident(),
            FailureClassification(failure_class, 0.99, False),
            None,
            external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
        )
        assert decision.tier is RepairRiskTier.PROHIBITED
        assert decision.candidate_generation_allowed is False


def test_risk_adaptive_configuration_is_strict_and_defaults_disabled(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        "default_competition: demo\nsupervisor:\n  mode: auto_safe\n  promotion:\n    automatic: true\n",
        encoding="utf-8",
    )
    assert load_settings(tmp_path).supervisor_config().auto_safe.enabled is False

    (config / "settings.yaml").write_text(
        "default_competition: demo\nsupervisor:\n  auto_safe:\n    enabled: true\n    profiles:\n      high:\n        automatic_promotion: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="high.automatic_promotion"):
        load_settings(tmp_path)


@pytest.mark.parametrize(
    ("stage", "path", "expected_tier", "candidate", "promotion"),
    [
        ("RESEARCH", "src/kaggle_agent/research/parser.py", "LOW", True, True),
        ("PLAN", "src/kaggle_agent/agents/plan.py", "LOW", True, True),
        ("CODE", "src/kaggle_agent/orchestrator.py", "MEDIUM", True, True),
        ("KERNEL_TRAIN", "src/kaggle_agent/train/kernel_runner.py", "MEDIUM", True, True),
        ("KERNEL_TRAIN", "src/kaggle_agent/supervisor/generation.py", "HIGH", True, False),
        ("CODE", "src/kaggle_agent/supervisor/recovery.py", "HIGH", True, False),
        ("SUBMIT", "src/kaggle_agent/autonomy/outbox.py", "PROHIBITED", False, False),
        ("SUBMIT", "src/kaggle_agent/autonomy/approval.py", "PROHIBITED", False, False),
        ("CODE", "pyproject.toml", "HIGH", True, False),
    ],
)
def test_risk_scenario_matrix(stage: str, path: str, expected_tier: str, candidate: bool, promotion: bool):
    decision = evaluate_repair_risk(
        _incident(stage),
        FailureClassification(FailureClass.CODE_DEFECT, 0.95, True),
        _spec(stage=stage, paths=(path,), allowed=(path,)),
        changed_paths=[path], changed_lines=20,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )
    assert decision.tier.value == expected_tier
    assert decision.candidate_generation_allowed is candidate
    assert decision.automatic_promotion_allowed is promotion


def test_risk_escalates_after_failed_history_and_rejects_test_weakening():
    normal = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), _spec(),
        changed_paths=["src/example/parser.py"], changed_lines=20,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )
    repeated = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), _spec(),
        changed_paths=["src/example/parser.py"], changed_lines=20,
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
        failed_attempts=2, same_signature_failures=1,
    )
    weakened = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), _spec(),
        changed_paths=["src/example/parser.py"], changed_lines=2,
        diff="- assert result == 3\n+ assert True\n",
        external_state=ExternalStateCertainty.NO_EXTERNAL_ACTION,
    )

    assert normal.tier is RepairRiskTier.LOW
    assert repeated.tier in {RepairRiskTier.MEDIUM, RepairRiskTier.HIGH}
    assert weakened.tier is RepairRiskTier.PROHIBITED
    assert weakened.candidate_generation_allowed is False


def test_risk_decision_round_trip_is_durable():
    decision = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), _spec(),
        changed_paths=["src/example/parser.py"], changed_lines=20,
    )
    restored = type(decision).from_dict(decision.to_dict())
    assert restored == decision


def test_reviewer_findings_escalate_a_low_candidate():
    decision = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), _spec(),
        changed_paths=["src/example/parser.py"], changed_lines=20,
        reviewer_findings=("medium:adapter contract needs review",),
    )
    assert decision.tier is RepairRiskTier.MEDIUM
    assert decision.automatic_promotion_allowed is True
