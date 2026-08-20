"""Independent DeepSeek role sessions with strict typed artifact parsing."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from kaggle_agent.llm.zen_client import ZenClient, ZenError
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.review import Review, ReviewFinding, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec, SpecReview, SpecReviewVerdict
from kaggle_agent.supervisor.repair_agent import RepairAgentBoundary


@dataclass(frozen=True)
class IndependentSession:
    role: str
    session_id: str
    parent_history: tuple[str, ...] = ()

    @classmethod
    def fresh(cls, role: str) -> "IndependentSession":
        return cls(role, f"{role}-{uuid.uuid4().hex}")


class AgentProtocolError(ValueError):
    pass


def _object(raw: str, role: str, required: set[str], optional: set[str] = set()) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentProtocolError(f"{role} returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError(f"{role} must return a JSON object")
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise AgentProtocolError(f"{role} omitted fields: {', '.join(sorted(missing))}")
    if unknown:
        raise AgentProtocolError(f"{role} returned unknown fields: {', '.join(sorted(unknown))}")
    return value


def _str(value: Any, role: str, field: str) -> str:
    if not isinstance(value, str):
        raise AgentProtocolError(f"{role}.{field} must be a string")
    return value


def _bool(value: Any, role: str, field: str) -> bool:
    if type(value) is not bool:
        raise AgentProtocolError(f"{role}.{field} must be a boolean")
    return value


def _list(value: Any, role: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AgentProtocolError(f"{role}.{field} must be a list of strings")
    return tuple(value)


def _finding_text_list(value: Any, role: str, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AgentProtocolError(f"{role}.{field} must be a list of strings")
    return tuple(value)


def _confidence(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentProtocolError(f"{role}.confidence must be numeric")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise AgentProtocolError(f"{role}.confidence must be between 0 and 1")
    return value


def run_structured_session(role: str, call: Callable[[str, str], str], system: str, artifact: dict[str, Any]) -> dict[str, Any]:
    session = IndependentSession.fresh(role)
    raw = call(system, json.dumps({"session_id": session.session_id, "artifact": artifact}, sort_keys=True))
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentProtocolError(f"{role} returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError(f"{role} must return a JSON object")
    return value


_CLASSIFIER_FIELDS = {"failure_class", "confidence", "repairable", "likely_files", "reason"}
_SPEC_FIELDS = {
    "title", "failed_stage", "observed_failure", "root_cause", "current_behavior", "expected_behavior",
    "likely_files", "reproduction_mode", "reproduction_commands", "invariants", "forbidden_changes",
    "required_tests", "verification_commands", "allowed_paths", "max_changed_source_files",
    "max_changed_test_files", "max_changed_lines", "proposed_resume_stage", "risk_level", "acceptance_criteria",
}
_SPEC_REVIEW_FIELDS = {"verdict", "blocking_findings", "non_blocking_findings", "rationale"}
_CODE_REVIEW_FIELDS = {
    "verdict", "root_cause_fixed", "tests_sufficient", "idempotency_safe", "checkpoint_safe", "policy_safe",
    "blocking_findings", "non_blocking_findings",
}


class DeepSeekSupervisorAgents:
    """Classifier/spec/reviewer/implementer roles backed by fresh sessions."""

    def __init__(self, router: Any, *, max_tool_turns: int = 30, max_attempts: int = 2) -> None:
        self.router = router
        self.max_tool_turns = max_tool_turns
        self.max_attempts = max(1, max_attempts)

    @classmethod
    def from_env(cls, *, model: str = "deepseek-chat") -> "DeepSeekSupervisorAgents | None":
        client = ZenClient.from_env()
        if client is None:
            return None

        class _Router:
            def __init__(self, inner: ZenClient) -> None:
                self.client = inner

            def model(self, role: str) -> str:
                return model

        return cls(_Router(client))

    def _chat(self, role: str, system: str, artifact: Mapping[str, Any]) -> str:
        session = IndependentSession.fresh(role)
        user = json.dumps({"session_id": session.session_id, "artifact": dict(artifact)}, sort_keys=True)
        client = self.router.client
        return client.chat_text(self.router.model(role), system, user, temperature=0, max_tokens=4096)

    def _request(
        self,
        role: str,
        system: str,
        artifact: Mapping[str, Any],
        parse: Callable[[str], Any],
    ) -> Any:
        """Call a role with bounded fresh-session retries and strict parsing."""
        last_error: AgentProtocolError | None = None
        for attempt in range(self.max_attempts):
            retry_hint = ""
            if attempt:
                retry_hint = (
                    " The previous response was invalid. Return only a JSON object that "
                    "matches the exact schema and enum constraints in this instruction."
                )
            try:
                return parse(self._chat(role, system + retry_hint, artifact))
            except AgentProtocolError as exc:
                last_error = exc
            except ZenError:
                last_error = AgentProtocolError(
                    f"{role} provider failure after bounded retry"
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _incident(incident: Incident) -> dict[str, Any]:
        value = incident.to_dict()
        # The classifier must not receive conclusions generated by a later role.
        value.pop("root_cause", None)
        return value

    def classify(self, incident: Incident) -> FailureClassification:
        allowed = ", ".join(item.value for item in FailureClass)

        def parse(raw_text: str) -> FailureClassification:
            raw = _object(raw_text, "classifier", _CLASSIFIER_FIELDS)
            try:
                failure_class = FailureClass(_str(raw["failure_class"], "classifier", "failure_class"))
            except ValueError as exc:
                raise AgentProtocolError("classifier.failure_class is invalid") from exc
            return FailureClassification(
                failure_class, _confidence(raw["confidence"], "classifier"),
                _bool(raw["repairable"], "classifier", "repairable"),
                _list(raw["likely_files"], "classifier", "likely_files"),
                _str(raw["reason"], "classifier", "reason"),
            )

        return self._request(
            "supervisor_classifier",
            f"You are a strict JSON classifier. Classify this sanitized incident. Return exactly one JSON object and nothing else, with exactly these five keys: failure_class, confidence, repairable, likely_files, reason. "
            f"failure_class MUST be one of [{allowed}]. For a deterministic Python NameError in project code, use CODE_DEFECT, never NameError. "
            "confidence must be a number from 0 to 1. repairable must be true or false. "
            "likely_files must be a JSON array of strings. reason must be a JSON string. "
            "Do not include session_id, markdown, or any extra key.",
            {"incident": self._incident(incident)},
            parse,
        )

    def author_spec(self, incident: Incident, classification: FailureClassification, *, repair_id: str) -> RepairSpec:
        def parse(raw_text: str) -> RepairSpec:
            raw = _object(raw_text, "spec author", _SPEC_FIELDS)
            for field in (
                "title", "failed_stage", "observed_failure", "root_cause",
                "current_behavior", "expected_behavior", "reproduction_mode",
                "risk_level", "proposed_resume_stage",
            ):
                _str(raw[field], "spec author", field)
            for field in (
                "likely_files", "reproduction_commands", "invariants",
                "forbidden_changes", "required_tests", "verification_commands",
                "allowed_paths", "acceptance_criteria",
            ):
                _list(raw[field], "spec author", field)
            if raw["reproduction_mode"] not in {
                "NEW_REGRESSION_TEST", "EXISTING_TEST_REPRO", "STATIC_REPRO", "NO_CODE_REPAIR",
            }:
                raise AgentProtocolError("spec author.reproduction_mode is invalid")
            if any(
                not path.strip() or "*" in path or path in {".", "/"}
                for path in raw["allowed_paths"]
            ):
                raise AgentProtocolError("spec author.allowed_paths must be narrow")
            limits = (raw["max_changed_source_files"], raw["max_changed_test_files"], raw["max_changed_lines"])
            if not all(isinstance(item, int) and not isinstance(item, bool) for item in limits):
                raise AgentProtocolError("spec author change limits must be integers")
            if limits[0] < 1 or limits[0] > 8 or limits[1] < 0 or limits[1] > 5 or limits[2] < 1 or limits[2] > 500:
                raise AgentProtocolError("spec author change limits exceed supervisor policy")
            value = dict(raw)
            value.update({
                "repair_id": repair_id,
                "incident_id": incident.incident_id,
                "base_generation": incident.generation_id,
                "base_revision": asdict(incident.revision),
            })
            try:
                return RepairSpec.from_dict(value)
            except (KeyError, TypeError, ValueError) as exc:
                raise AgentProtocolError(f"spec author returned invalid RepairSpec: {exc}") from exc

        return self._request(
            "repair_spec_author",
            "Inspect the explicit incident and classification, then author one minimal RepairSpec. "
            "Return only one JSON object containing exactly these fields: "
            + ", ".join(sorted(_SPEC_FIELDS))
            + ". Every field ending in _files, _commands, _changes, _criteria, invariants, "
            "forbidden_changes, required_tests, verification_commands, likely_files, or allowed_paths "
            "must be a JSON array where applicable; acceptance_criteria must be an array of strings. "
            "reproduction_mode must be NEW_REGRESSION_TEST, EXISTING_TEST_REPRO, STATIC_REPRO, or NO_CODE_REPAIR. "
            "Use integer limits of 1..8 source files, 0..5 test files, and 1..500 changed lines. "
            "Keep allowed_paths narrow with no wildcards or repository-wide paths.",
            {"incident": self._incident(incident), "classification": {"failure_class": classification.failure_class.value, "confidence": classification.confidence, "repairable": classification.repairable, "likely_files": list(classification.likely_files), "reason": classification.reason}},
            parse,
        )

    def review_spec(self, incident: Incident, classification: FailureClassification, spec: RepairSpec) -> SpecReview:
        def parse(raw_text: str) -> SpecReview:
            raw = _object(raw_text, "spec reviewer", _SPEC_REVIEW_FIELDS)
            try:
                verdict = SpecReviewVerdict(_str(raw["verdict"], "spec reviewer", "verdict").upper())
            except ValueError as exc:
                raise AgentProtocolError("spec reviewer verdict is invalid") from exc
            return SpecReview(
                verdict,
                _finding_text_list(raw["blocking_findings"], "spec reviewer", "blocking_findings"),
                _finding_text_list(raw["non_blocking_findings"], "spec reviewer", "non_blocking_findings"),
                _str(raw["rationale"], "spec reviewer", "rationale"),
            )

        return self._request(
            "spec_reviewer",
            "Independently review the incident, classification, and RepairSpec. "
            "Return exactly one JSON object with fields verdict, blocking_findings, "
            "non_blocking_findings, rationale. verdict must be APPROVE, REVISE, "
            "NOT_CODE_DEFECT, or NEEDS_AUTHORITY.",
            {"incident": self._incident(incident), "classification": classification.failure_class.value, "spec": spec.to_dict()},
            parse,
        )

    def review_code(self, incident: Incident, spec: RepairSpec, diff: str, verification: Any) -> Review:
        return self._request(
            "code_reviewer",
            "Independently review the candidate diff and verification. Return exactly one JSON object with "
            "the required typed fields: verdict, root_cause_fixed, tests_sufficient, idempotency_safe, "
            "checkpoint_safe, policy_safe, blocking_findings, non_blocking_findings. "
            "verdict must be APPROVE, REJECT, REVISE, NOT_CODE_DEFECT, or NEEDS_AUTHORITY. "
            "Each finding must contain severity, file, issue, and required_fix.",
            {"incident": self._incident(incident), "spec": spec.to_dict(), "diff": diff, "verification": verification.to_dict() if hasattr(verification, "to_dict") else str(verification)},
            lambda raw: _parse_review(_object(raw, "code reviewer", _CODE_REVIEW_FIELDS)),
        )

    def implement(self, root: Path, spec: RepairSpec) -> "RepairImplementerResult":
        boundary = RepairAgentBoundary(root)
        system = (
            "Implement exactly this approved RepairSpec. Read before writing. Use only the provided tools. "
            "Return exactly one JSON action object with action, tool, args, reason. "
            'action must be exactly "tool" or "done"; when invoking a tool, use action "tool" '
            'and put the tool name in the separate tool field. Do not use action "read_file" or any other tool name. '
            "Successful tool results are in messages as {tool, result}; use those results, do not repeat a successful read, "
            "and apply the smallest patch or write after inspecting the source. Use args.path (not file_path) for file tools. "
            "After a successful read of an allowed file, never read that same file again: use its returned content and the approved spec to edit it, or stop with done if the required fix is not provable. "
            "Prefer apply_patch for edits. Do not use write_file unless an expected_sha256 value is available from the tool context. "
            "Allowed tools are list_files, read_file, search_code, git_diff, write_file, apply_patch, "
            "run_reproduction, run_focused_test, run_compile_check, done. "
            "Do not use shell, network, credentials, Git mutation, dependencies, or protected files."
        )
        messages: list[dict[str, Any]] = []
        for _ in range(self.max_tool_turns):
            action = self._request(
                "repair_implementer",
                system,
                {"spec": spec.to_dict(), "messages": messages},
                lambda raw: _object(raw, "repair implementer", {"action", "tool", "args", "reason"}),
            )
            kind = _str(action["action"], "repair implementer", "action")
            reason = _str(action["reason"], "repair implementer", "reason")
            if kind == "done":
                return RepairImplementerResult(True, reason)
            if kind != "tool":
                messages.append({"error": f"unsupported action {kind}; use action tool or done"})
                continue
            tool = _str(action["tool"], "repair implementer", "tool")
            args = action["args"]
            if not isinstance(args, dict):
                return RepairImplementerResult(True, "policy: tool args must be an object")
            try:
                result = boundary.call(tool, **args)
            except Exception as exc:  # noqa: BLE001
                if tool not in boundary.tools_allowed:
                    return RepairImplementerResult(True, f"policy: {exc}")
                result = {"error": str(exc)}
            messages.append({"tool": tool, "result": str(result)[:12000]})
        return RepairImplementerResult(True, "policy: implementer turn budget exhausted")


@dataclass(frozen=True)
class RepairImplementerResult:
    stopped: bool
    reason: str
    writes: tuple[str, ...] = ()


def _parse_review(raw: Mapping[str, Any]) -> Review:
    try:
        verdict = ReviewVerdict(_str(raw["verdict"], "code reviewer", "verdict").upper())
    except ValueError as exc:
        raise AgentProtocolError("code reviewer verdict is invalid") from exc

    def findings(field: str) -> tuple[ReviewFinding, ...]:
        rows = raw[field]
        if not isinstance(rows, list):
            raise AgentProtocolError(f"code reviewer.{field} must be a list")
        result = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"severity", "file", "issue", "required_fix"}:
                raise AgentProtocolError(f"code reviewer.{field} contains an invalid finding")
            result.append(ReviewFinding(_str(row["severity"], "code reviewer", "severity"), _str(row["file"], "code reviewer", "file"), _str(row["issue"], "code reviewer", "issue"), _str(row["required_fix"], "code reviewer", "required_fix")))
        return tuple(result)

    return Review(verdict, *(_bool(raw[name], "code reviewer", name) for name in ("root_cause_fixed", "tests_sufficient", "idempotency_safe", "checkpoint_safe", "policy_safe")), findings("blocking_findings"), findings("non_blocking_findings"))
