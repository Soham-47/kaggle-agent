import hashlib
import json
from pathlib import Path

import pytest

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.llm.zen_client import ZenError
from kaggle_agent.supervisor.agents import AgentProtocolError, DeepSeekSupervisorAgents
from kaggle_agent.supervisor.repair_agent import RepairAgentBoundary
from kaggle_agent.supervisor.classifier import FailureClass
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.review import ReviewVerdict
from kaggle_agent.supervisor.spec import SpecReviewVerdict
from kaggle_agent.supervisor.verify import VerificationResult


class _FakeRouter:
    def __init__(self, responses):
        self.client = self
        self.responses = list(responses)
        self.last_response = None
        self.calls = []

    def model(self, role):
        return f"model-for-{role}"

    def chat_text(self, model, system, user, **kwargs):
        self.calls.append((model, system, user, kwargs))
        if self.responses:
            self.last_response = self.responses.pop(0)
        if self.last_response is None:
            raise AssertionError("fake router has no response")
        return self.last_response


def _incident() -> Incident:
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    return Incident.from_outcome(
        worker_id="worker-1",
        generation_id="generation-0001",
        competition="demo",
        outcome=StageOutcome.failure("CODE", "NameError: missing_name", "sig-1"),
        stage_attempt=1,
        revision=revision,
    )


def _classification_response(**overrides):
    value = {
        "failure_class": "CODE_DEFECT",
        "confidence": 0.92,
        "repairable": True,
        "likely_files": ["src/x.py"],
        "reason": "the failure is a deterministic source defect",
    }
    value.update(overrides)
    return json.dumps(value)


def _spec_response(**overrides):
    value = {
        "title": "Fix missing name",
        "failed_stage": "CODE",
        "observed_failure": "NameError: missing_name",
        "root_cause": "the source references a missing name",
        "current_behavior": "the CODE stage raises NameError",
        "expected_behavior": "the CODE stage completes without NameError",
        "likely_files": ["src/x.py"],
        "reproduction_mode": "STATIC_REPRO",
        "reproduction_commands": ["uv run pytest -q tests/test_x.py"],
        "invariants": ["submission approval remains required"],
        "forbidden_changes": [".env", "src/kaggle_agent/supervisor/policy.py"],
        "required_tests": ["tests/test_x.py"],
        "verification_commands": ["uv run pytest -q tests/test_x.py"],
        "allowed_paths": ["src", "tests"],
        "max_changed_source_files": 2,
        "max_changed_test_files": 1,
        "max_changed_lines": 80,
        "proposed_resume_stage": "CODE",
        "risk_level": "low",
        "acceptance_criteria": ["the focused test passes"],
    }
    value.update(overrides)
    return json.dumps(value)


def _spec_review_response(**overrides):
    value = {
        "verdict": "APPROVE",
        "blocking_findings": [],
        "non_blocking_findings": [],
        "rationale": "the scope and verification plan are bounded",
    }
    value.update(overrides)
    return json.dumps(value)


def _code_review_response(**overrides):
    value = {
        "verdict": "APPROVE",
        "root_cause_fixed": True,
        "tests_sufficient": True,
        "idempotency_safe": True,
        "checkpoint_safe": True,
        "policy_safe": True,
        "blocking_findings": [],
        "non_blocking_findings": [],
    }
    value.update(overrides)
    return json.dumps(value)


def test_production_roles_use_fresh_sessions_and_explicit_typed_artifacts():
    router = _FakeRouter([
        _classification_response(),
        _spec_response(),
        _spec_review_response(),
        _code_review_response(),
    ])
    agents = DeepSeekSupervisorAgents(router)
    incident = _incident()

    classification = agents.classify(incident)
    spec = agents.author_spec(incident, classification, repair_id="repair-1")
    spec_review = agents.review_spec(incident, classification, spec)
    code_review = agents.review_code(
        incident,
        spec,
        "diff --git a/src/x.py b/src/x.py\n+fixed\n",
        VerificationResult(True, (("uv", "run", "pytest"),), ()),
    )

    assert classification.failure_class is FailureClass.CODE_DEFECT
    assert spec.repair_id == "repair-1"
    assert spec.base_revision == incident.revision
    assert spec_review.verdict is SpecReviewVerdict.APPROVE
    assert code_review.verdict is ReviewVerdict.APPROVE
    sessions = [json.loads(call[2])["session_id"] for call in router.calls]
    assert len(set(sessions)) == 4
    assert [call[0] for call in router.calls] == [
        "model-for-supervisor_classifier",
        "model-for-repair_spec_author",
        "model-for-spec_reviewer",
        "model-for-code_reviewer",
    ]
    assert "root_cause" not in json.loads(router.calls[0][2])["artifact"]
    assert json.loads(router.calls[1][2])["artifact"]["classification"]["failure_class"] == "CODE_DEFECT"
    assert "previous_response" not in json.loads(router.calls[2][2])


def test_classifier_prompt_enumerates_allowed_failure_classes():
    router = _FakeRouter([_classification_response()])
    agents = DeepSeekSupervisorAgents(router)

    agents.classify(_incident())

    prompt = router.calls[0][1]
    assert "CODE_DEFECT" in prompt
    assert "TRANSIENT_EXTERNAL" in prompt
    assert "PENDING_EXTERNAL" in prompt
    assert "use CODE_DEFECT, never NameError" in prompt


def test_malformed_role_output_gets_one_bounded_fresh_session_retry():
    router = _FakeRouter(["{\"failure_class\": \"NameError\"}", _classification_response()])
    agents = DeepSeekSupervisorAgents(router)

    classification = agents.classify(_incident())

    assert classification.failure_class is FailureClass.CODE_DEFECT
    assert len(router.calls) == 2
    assert json.loads(router.calls[0][2])["session_id"] != json.loads(router.calls[1][2])["session_id"]


def test_provider_failure_retries_once_then_fails_closed():
    class FailingRouter(_FakeRouter):
        def chat_text(self, model, system, user, **kwargs):
            self.calls.append((model, system, user, kwargs))
            raise ZenError("provider unavailable")

    router = FailingRouter([])
    agents = DeepSeekSupervisorAgents(router)

    with pytest.raises(AgentProtocolError, match="provider failure after bounded retry"):
        agents.classify(_incident())

    assert len(router.calls) == 2


def test_implementer_prompt_requires_tool_action_envelope():
    router = _FakeRouter([
        _classification_response(),
        _spec_response(),
        json.dumps({"action": "done", "tool": "done", "args": {}, "reason": "complete"}),
    ])
    agents = DeepSeekSupervisorAgents(router)

    incident = _incident()
    spec = agents.author_spec(incident, agents.classify(incident), repair_id="repair-1")
    agents.implement(Path("."), spec)

    prompt = next(call[1] for call in router.calls if call[0] == "model-for-repair_implementer")
    assert 'action must be exactly "tool" or "done"' in prompt
    assert 'action "tool"' in prompt
    assert 'Successful tool results are in messages' in prompt
    assert 'never read that same file again' in prompt


@pytest.mark.parametrize(
    "response, message",
    [
        (_classification_response(confidence="high"), "classifier"),
        (_classification_response(extra="reject"), "classifier"),
        ("[]", "classifier"),
    ],
)
def test_classifier_rejects_non_strict_model_output(response, message):
    agents = DeepSeekSupervisorAgents(_FakeRouter([response]))
    with pytest.raises(AgentProtocolError, match=message):
        agents.classify(_incident())


def test_spec_author_rejects_unbounded_or_malformed_model_output():
    response = _spec_response(max_changed_lines=501)
    agents = DeepSeekSupervisorAgents(_FakeRouter([_classification_response(), response]))
    incident = _incident()
    classification = agents.classify(incident)
    with pytest.raises(AgentProtocolError, match="spec author"):
        agents.author_spec(incident, classification, repair_id="repair-1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"acceptance_criteria": "one string, not an array"},
        {"reproduction_mode": "command"},
        {"max_changed_test_files": -1},
        {"allowed_paths": ["src/**"]},
    ],
)
def test_spec_author_rejects_loosely_typed_or_broad_specs(overrides):
    agents = DeepSeekSupervisorAgents(_FakeRouter([
        _classification_response(),
        _spec_response(**overrides),
    ]))
    incident = _incident()
    classification = agents.classify(incident)

    with pytest.raises(AgentProtocolError, match="spec author"):
        agents.author_spec(incident, classification, repair_id="repair-1")


def test_spec_reviewer_rejects_non_string_findings():
    agents = DeepSeekSupervisorAgents(_FakeRouter([
        _classification_response(),
        _spec_response(),
        _spec_review_response(blocking_findings=[{"issue": "not a string"}]),
    ]))
    incident = _incident()
    classification = agents.classify(incident)
    spec = agents.author_spec(incident, classification, repair_id="repair-1")

    with pytest.raises(AgentProtocolError, match="spec reviewer"):
        agents.review_spec(incident, classification, spec)


def test_repair_implementer_uses_only_restricted_tools(tmp_path: Path):
    source = tmp_path / "src" / "x.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    responses = [_classification_response(), _spec_response(),
        json.dumps({"action": "tool", "tool": "read_file", "args": {"path": "src/x.py"}, "reason": "inspect"}),
        json.dumps({
            "action": "tool",
            "tool": "write_file",
            "args": {"path": "src/x.py", "content": "x = 2\n", "expected_sha256": digest},
            "reason": "apply minimal fix",
        }),
        json.dumps({"action": "done", "tool": "done", "args": {}, "reason": "repair complete"}),
    ]
    router = _FakeRouter(responses)
    agents = DeepSeekSupervisorAgents(router)
    spec = agents.author_spec(
        _incident(),
        agents.classify(_incident()),
        repair_id="repair-1",
    )

    result = agents.implement(tmp_path, spec)

    assert result.stopped is True
    assert result.reason == "repair complete"
    assert source.read_text(encoding="utf-8") == "x = 2\n"
    assert all("DEEPSEEK_API_KEY" not in call[2] for call in router.calls)


def test_repair_boundary_accepts_model_file_path_alias(tmp_path: Path):
    source = tmp_path / "src" / "x.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")

    assert RepairAgentBoundary(tmp_path).call("read_file", file_path="src/x.py") == "x = 1\n"


def test_repair_implementer_stops_on_unavailable_shell_tool(tmp_path: Path):
    router = _FakeRouter([
        _classification_response(),
        _spec_response(),
        json.dumps({"action": "tool", "tool": "shell", "args": {}, "reason": "try shell"}),
    ])
    agents = DeepSeekSupervisorAgents(router)
    incident = _incident()
    spec = agents.author_spec(
        incident,
        agents.classify(incident),
        repair_id="repair-1",
    )

    result = agents.implement(tmp_path, spec)

    assert result.stopped is True
    assert "policy" in result.reason


def test_repair_implementer_recovers_from_invalid_action_envelope(tmp_path: Path):
    router = _FakeRouter([
        _classification_response(),
        _spec_response(),
        json.dumps({"action": "read_file", "tool": "read_file", "args": {"path": "src/x.py"}, "reason": "inspect"}),
        json.dumps({"action": "done", "tool": "done", "args": {}, "reason": "complete"}),
    ])
    agents = DeepSeekSupervisorAgents(router)
    incident = _incident()
    spec = agents.author_spec(incident, agents.classify(incident), repair_id="repair-1")

    result = agents.implement(tmp_path, spec)

    assert result.reason == "complete"
