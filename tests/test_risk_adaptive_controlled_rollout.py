"""Provider-independent controlled rollout certification scenarios."""

from __future__ import annotations

import subprocess
import sys
import time
import os
from pathlib import Path

import pytest

from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeGeneration, RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.promote import GenerationPromotion
from kaggle_agent.supervisor.risk import ExternalStateCertainty, RepairRiskTier, evaluate_repair_risk
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.spec import RepairSpec


def _incident(stage: str = "CODE", *, external_job: str | None = None) -> Incident:
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    return Incident(
        incident_id="controlled-incident", worker_id="worker-1", generation_id="generation-0001",
        cycle_id="cycle-1", experiment_id=None, competition="demo", stage=stage, stage_attempt=1,
        revision=revision, outcome_state="recoverable_failure", exception_type="NameError",
        exception_message="NameError: broken", traceback=None, failure_signature="controlled-signature",
        evidence=(), artifacts=(), external_job=external_job, kernel_ref=None, candidate_csv=None,
        recent_logs=(), created_at="now",
    )


def _spec(stage: str, paths: tuple[str, ...], mode: str = "EXISTING_TEST_REPRO") -> RepairSpec:
    incident = _incident(stage)
    return RepairSpec(
        repair_id="controlled-repair", incident_id=incident.incident_id,
        base_generation=incident.generation_id, base_revision=incident.revision,
        title="controlled test repair", failed_stage=stage,
        observed_failure="NameError", root_cause="undefined local", current_behavior="fails",
        expected_behavior="works", likely_files=paths, reproduction_mode=mode,
        reproduction_commands=("uv run pytest -q tests/test_controlled.py",), invariants=(),
        forbidden_changes=("approval", "outbox", "credentials"), required_tests=(),
        verification_commands=("uv run pytest -q tests/test_controlled.py",), allowed_paths=paths,
        max_changed_source_files=12, max_changed_test_files=6, max_changed_lines=1000,
        proposed_resume_stage=stage, risk_level="medium", acceptance_criteria=("focused test passes",),
    )


@pytest.mark.parametrize(
    ("name", "stage", "paths", "lines", "external", "tier", "candidate", "promotion"),
    (
        ("LOW deterministic repair", "CODE", ("src/example/parser.py",), 20, ExternalStateCertainty.NO_EXTERNAL_ACTION, RepairRiskTier.LOW, True, True),
        ("MEDIUM multi-file repair", "CODE", tuple(f"competitions/demo/pipeline/file{i}.py" for i in range(4)), 400, ExternalStateCertainty.NO_EXTERNAL_ACTION, RepairRiskTier.MEDIUM, True, True),
        ("HIGH replay repair", "KERNEL_TRAIN", ("src/kaggle_agent/supervisor/resume.py",), 20, ExternalStateCertainty.NO_EXTERNAL_ACTION, RepairRiskTier.HIGH, True, False),
        ("PROHIBITED trust-base repair", "SUBMIT", ("src/kaggle_agent/autonomy/outbox.py",), 2, ExternalStateCertainty.NO_EXTERNAL_ACTION, RepairRiskTier.PROHIBITED, False, False),
        ("LOW defect with ambiguous external state", "KERNEL_TRAIN", ("src/example/parser.py",), 20, ExternalStateCertainty.AMBIGUOUS, RepairRiskTier.LOW, False, False),
    ),
)
def test_controlled_rollout_canary_matrix(name, stage, paths, lines, external, tier, candidate, promotion):
    decision = evaluate_repair_risk(
        _incident(stage, external_job="action-1" if external is not ExternalStateCertainty.NO_EXTERNAL_ACTION else None),
        FailureClassification(FailureClass.CODE_DEFECT, 0.99, True),
        _spec(stage, paths), changed_paths=paths, changed_lines=lines, external_state=external,
    )

    assert decision.tier is tier, name
    assert decision.candidate_generation_allowed is candidate, name
    assert decision.automatic_promotion_allowed is promotion, name


def test_low_provisional_risk_escalates_when_candidate_touches_trust_base():
    provisional = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.99, True),
        _spec("CODE", ("src/example/parser.py",)), changed_paths=(), changed_lines=0,
    )
    post_diff = evaluate_repair_risk(
        _incident(), FailureClassification(FailureClass.CODE_DEFECT, 0.99, True),
        _spec("CODE", ("src/example/parser.py",)),
        changed_paths=("src/kaggle_agent/autonomy/outbox.py",), changed_lines=1,
        minimum_tier=provisional.tier,
    )

    assert provisional.tier is RepairRiskTier.LOW
    assert post_diff.tier is RepairRiskTier.PROHIBITED
    assert post_diff.automatic_promotion_allowed is False


@pytest.mark.parametrize("pointer_is_new", (False, True))
def test_real_subprocess_kill_at_promotion_boundary_recovers_deterministically(tmp_path: Path, pointer_is_new: bool):
    repo = tmp_path / "repo"
    state = SupervisorStateStore(RuntimeLayout.for_repo(repo, tmp_path / "state"))
    old = RuntimeGeneration("generation-0001", RuntimeRevision("a", "b", "generation-0001"), str(repo))
    new = RuntimeGeneration("generation-0002", RuntimeRevision("c", "d", "generation-0002"), str(repo), parent_generation=old.generation_id, repair_id="r1")
    generations = GenerationStore(state)
    generations.save(old)
    generations.save(new)
    state.write_json("active-generation.json", new.to_dict() if pointer_is_new else old.to_dict())
    marker = tmp_path / "ready"
    script = """
from pathlib import Path
import sys, time
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
state = SupervisorStateStore(RuntimeLayout.for_repo(Path(sys.argv[1]), Path(sys.argv[2])))
state.write_json('promotion.json', {'schema_version': 2, 'status': 'PREPARED', 'old_generation': 'generation-0001', 'new_generation': 'generation-0002'})
Path(sys.argv[3]).write_text('ready', encoding='utf-8')
time.sleep(30)
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    process = subprocess.Popen(
        (sys.executable, "-c", script, str(repo), str(state.layout.state_root), str(marker)),
        env=environment,
    )
    try:
        for _ in range(100):
            if marker.is_file():
                break
            time.sleep(0.01)
        assert marker.is_file()
        process.kill()
        process.wait(timeout=5)
        recovered = GenerationPromotion(state).recover_interrupted(old, new)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    expected = "PROMOTED" if pointer_is_new else "ROLLED_BACK"
    assert recovered == expected
    assert state.read_json("active-generation.json")["generation_id"] == (new.generation_id if pointer_is_new else old.generation_id)
