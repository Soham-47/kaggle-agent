"""Independent DeepSeek role sessions with strict typed artifact parsing."""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from kaggle_agent.llm.zen_client import ZenClient, ZenError
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.review import Review, ReviewFinding, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec, SpecReview, SpecReviewVerdict
from kaggle_agent.supervisor.policy import RepairPolicy
from kaggle_agent.supervisor.repair_agent import RepairAgentBoundary, ToolProtocolError
from kaggle_agent.autonomy.repair_tools import ToolPolicyError


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


class ImplementerStatus(str, Enum):
    PATCH_READY = "PATCH_READY"
    NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
    TOOL_POLICY_BLOCK = "TOOL_POLICY_BLOCK"
    PROTOCOL_FAILURE = "PROTOCOL_FAILURE"
    TURN_BUDGET_EXHAUSTED = "TURN_BUDGET_EXHAUSTED"
    NO_CHANGE = "NO_CHANGE"
    IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"


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
            for field in ("reproduction_commands", "verification_commands"):
                if any(not _allowed_verification_command(command) for command in raw[field]):
                    raise AgentProtocolError(
                        f"spec author.{field} contains a command outside the supervisor verification allowlist"
                    )
            if any(
                not path.strip() or "*" in path or path in {".", "/"}
                for path in raw["allowed_paths"]
            ):
                raise AgentProtocolError("spec author.allowed_paths must be narrow")
            limits = (raw["max_changed_source_files"], raw["max_changed_test_files"], raw["max_changed_lines"])
            if not all(isinstance(item, int) and not isinstance(item, bool) for item in limits):
                raise AgentProtocolError("spec author change limits must be integers")
            if limits[0] < 1 or limits[0] > 8 or limits[1] < 0 or limits[1] > 5 or limits[2] < 2 or limits[2] > 500:
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
            "Use integer limits of 1..8 source files, 0..5 test files, and 2..500 changed lines; "
            "a one-line replacement counts as two changed lines because both deletion and addition are measured. "
            "Keep allowed_paths narrow with no wildcards or repository-wide paths. "
            "Every reproduction_commands and verification_commands entry must begin with one of: "
            "uv run pytest, uv run python -m compileall, or uv run python -m py_compile.",
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

    def implement(
        self,
        root: Path,
        spec: RepairSpec,
        verification_feedback: Any | None = None,
        *,
        state_root: Path | None = None,
    ) -> "RepairImplementerResult":
        boundary = RepairAgentBoundary(root, state_root=state_root)
        system = (
            "Implement exactly this approved RepairSpec. Read before writing. Use only the provided tools. "
            "Return exactly one JSON action object with action, tool, args, reason. "
            'action must be exactly "tool" or "done"; when invoking a tool, use action "tool" '
            'and put the tool name in the separate tool field. Do not use action "read_file" or any other tool name. '
            'When action is "done", set tool to the string "done" and args to an empty object. '
            'The exact completion envelope is {"action":"done","tool":"done","args":{},"reason":"verified candidate is ready"}. '
            "Successful tool results are in messages as {tool, result}; use those results, do not repeat a successful read, "
            "and apply the smallest patch or write after inspecting the source. Use exactly the typed argument schema: "
            "The artifact includes targeted context, but you must still call read_file on every required_read_path before writing. "
            "read_file requires {path}; write_file requires {path, content, expected_sha256}; apply_patch requires {patch} only. "
            "The apply_patch value must be a standard unified diff with --- a/path and +++ b/path lines; do not add a path field. "
            "After a successful read of an allowed file, never read that same file again: use its returned content and the approved spec to edit it, or stop with done if the required fix is not provable. "
            "A done action is invalid while current_candidate_diff is empty; when the approved spec describes a repair, you must produce a non-empty scoped write before done. "
            "Prefer apply_patch for edits. Do not use write_file unless an expected_sha256 value is available from the tool context. "
            "Allowed tools are list_files, read_file, search_code, git_diff, write_file, apply_patch, "
            "run_reproduction, run_focused_test, run_compile_check, done. "
            "Do not use shell, network, credentials, Git mutation, dependencies, or protected files."
        )
        messages: list[dict[str, Any]] = []
        read_paths: set[str] = set()
        writes: list[str] = []
        last_protocol_error: str | None = None
        last_tool_error: str | None = None
        for turn in range(1, self.max_tool_turns + 1):
            artifact = {
                "spec": spec.to_dict(),
                "verification_feedback": (
                    verification_feedback.to_dict()
                    if hasattr(verification_feedback, "to_dict")
                    else verification_feedback
                ),
                "current_candidate_diff": _git_diff(root),
                "implementation_contract": "A non-empty scoped candidate diff is required before done; supervisor verification is authoritative.",
                "relevant_files": list(spec.likely_files),
                "required_read_paths": list(spec.likely_files),
                "target_context": _target_context(root, spec),
                "messages": messages,
            }
            try:
                action = self._request(
                    "repair_implementer",
                    system,
                    artifact,
                    lambda raw: _object(raw, "repair implementer", {"action", "tool", "args", "reason"}),
                )
                kind = _str(action["action"], "repair implementer", "action")
                tool = _str(action["tool"], "repair implementer", "tool")
                args = action["args"]
                reason = _str(action["reason"], "repair implementer", "reason")
                if not isinstance(args, dict):
                    raise AgentProtocolError("repair implementer.args must be an object")
                if kind not in {"tool", "done"}:
                    raise AgentProtocolError("repair implementer.action must be tool or done")
                if kind == "done":
                    if tool != "done" or args:
                        raise AgentProtocolError("repair implementer done requires tool=done and empty args")
                    candidate = _candidate_result(root, spec, reason, turn, tuple(writes))
                    if candidate.status is ImplementerStatus.NO_CHANGE and turn < self.max_tool_turns:
                        messages.append({
                            "error": {
                                "kind": "NO_CHANGE",
                                "message": "done was returned before a candidate diff existed",
                                "instruction": "The approved repair still requires a non-empty scoped edit. Read the approved context and apply the smallest fix, then return done.",
                            }
                        })
                        continue
                    return candidate
                if tool == "done" or tool not in boundary.tools_allowed:
                    return RepairImplementerResult(
                        True, f"policy: tool is not available to repair agent: {tool}",
                        tuple(writes), ImplementerStatus.TOOL_POLICY_BLOCK, turn,
                    )
                if tool == "read_file":
                    path = _model_path(args)
                    if path in read_paths:
                        messages.append({
                            "error": {
                                "kind": "DUPLICATE_READ",
                                "message": _bound(f"read_file already succeeded for {path}"),
                                "instruction": "Use the earlier successful read result; do not request the same file again. Apply the smallest approved edit or stop if the fix is not provable.",
                            }
                        })
                        continue
                if tool in {"write_file", "apply_patch"}:
                    _require_read_before_write(tool, args, read_paths)
                    _require_spec_write_scope(tool, args, spec)
                result = boundary.call(tool, **args)
                if tool == "read_file":
                    path = _model_path(args)
                    if path:
                        read_paths.add(path)
                if tool in {"write_file", "apply_patch"}:
                    writes.extend(_written_paths(tool, args, result))
                    messages.append({
                        "tool": tool,
                        "result": _bounded_tool_result(result),
                        "instruction": "The write succeeded. Do not repeat this edit. Return the exact done envelope now; supervisor verification is authoritative.",
                    })
                else:
                    messages.append({"tool": tool, "result": _bounded_tool_result(result)})
                last_tool_error = None
            except ToolPolicyError as exc:
                if str(exc).startswith("write requires a prior successful read:"):
                    last_tool_error = str(exc)
                    messages.append({
                        "error": {
                            "kind": "READ_REQUIRED",
                            "message": _bound(str(exc)),
                            "instruction": "Read every named target with read_file in this session, then make the smallest allowed edit.",
                        }
                    })
                    continue
                return RepairImplementerResult(
                    True, f"policy: {exc}", tuple(writes), ImplementerStatus.TOOL_POLICY_BLOCK, turn,
                )
            except (ToolProtocolError, AgentProtocolError) as exc:
                last_protocol_error = str(exc)
                messages.append({"error": {"kind": "PROTOCOL_FAILURE", "message": _bound(str(exc))}})
            except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
                last_tool_error = str(exc)
                messages.append({"error": {"kind": "TOOL_ERROR", "message": _bound(str(exc))}})
            except Exception as exc:  # noqa: BLE001 - fail closed at the role boundary
                return RepairImplementerResult(
                    True, f"implementation failure: {exc}", tuple(writes),
                    ImplementerStatus.IMPLEMENTATION_FAILURE, turn,
                )
        if last_protocol_error:
            return RepairImplementerResult(
                True, last_protocol_error, tuple(writes), ImplementerStatus.PROTOCOL_FAILURE,
                self.max_tool_turns,
            )
        reason = last_tool_error or "implementer turn budget exhausted"
        return RepairImplementerResult(
            True, reason, tuple(writes), ImplementerStatus.TURN_BUDGET_EXHAUSTED,
            self.max_tool_turns,
        )


@dataclass(frozen=True)
class RepairImplementerResult:
    stopped: bool
    reason: str
    writes: tuple[str, ...] = ()
    status: ImplementerStatus = ImplementerStatus.IMPLEMENTATION_FAILURE
    tool_turns: int = 0
    diff: str = ""
    changed_paths: tuple[str, ...] = ()


def _bound(value: object, limit: int = 4000) -> str:
    return str(value).replace("\x00", "")[:limit]


def _allowed_verification_command(command: str) -> bool:
    parts = command.split()
    return bool(parts) and (
        parts[:3] == ["uv", "run", "pytest"]
        or parts[:4] == ["uv", "run", "python", "-m"]
        and len(parts) >= 5
        and parts[4] in {"compileall", "py_compile"}
    )


def _bounded_tool_result(value: object) -> str:
    if isinstance(value, subprocess.CompletedProcess):
        return _bound({
            "returncode": value.returncode,
            "stdout": _bound(value.stdout),
            "stderr": _bound(value.stderr),
        }, 12000)
    return _bound(value, 12000)


def _model_path(args: Mapping[str, Any]) -> str:
    return str(args.get("path") or args.get("file_path") or "").replace("\\", "/").lstrip("./")


def _patch_paths(patch: str) -> tuple[str, ...]:
    paths = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if path != "/dev/null":
                paths.append(path.replace("\\", "/").lstrip("./"))
    return tuple(dict.fromkeys(paths))


def _written_paths(tool: str, args: Mapping[str, Any], result: object) -> tuple[str, ...]:
    if tool == "write_file":
        path = _model_path(args)
        return (path,) if path else ()
    paths = _patch_paths(str(args.get("patch", "")))
    if paths:
        return paths
    if isinstance(result, str):
        return tuple(item for item in result.splitlines() if item.strip())
    return ()


def _require_read_before_write(tool: str, args: Mapping[str, Any], read_paths: set[str]) -> None:
    if tool == "write_file":
        paths = (_model_path(args),)
    else:
        paths = _patch_paths(str(args.get("patch", "")))
    missing = [path for path in paths if path and path not in read_paths]
    if missing:
        raise ToolPolicyError(f"write requires a prior successful read: {', '.join(missing)}")


def _require_spec_write_scope(tool: str, args: Mapping[str, Any], spec: RepairSpec) -> None:
    paths = (_model_path(args),) if tool == "write_file" else _patch_paths(str(args.get("patch", "")))
    policy = RepairPolicy()
    violations = policy.allowed_path_violations(tuple(path for path in paths if path), spec.allowed_paths)
    violations += policy.protected_violations(tuple(path for path in paths if path))
    if violations:
        raise ToolPolicyError(f"write is outside approved RepairSpec scope: {', '.join(violations)}")


def _git_diff(root: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), "diff", "--no-ext-diff"),
        text=True,
        capture_output=True,
        check=False,
    )
    return _bound(result.stdout, 12000)


def _target_context(root: Path, spec: RepairSpec) -> dict[str, str]:
    context: dict[str, str] = {}
    for relative in tuple(dict.fromkeys(spec.likely_files + spec.required_tests)):
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
            context[relative] = _bound(path.read_text(encoding="utf-8"), 6000)
        except (OSError, ValueError):
            context[relative] = "<unavailable; use the allowed read_file tool>"
    return context


def _candidate_result(root: Path, spec: RepairSpec, reason: str, turns: int, writes: tuple[str, ...]) -> RepairImplementerResult:
    diff = _git_diff(root)
    paths_result = subprocess.run(
        ("git", "-C", str(root), "diff", "--name-only"),
        text=True,
        capture_output=True,
        check=False,
    )
    changed_paths = tuple(line for line in paths_result.stdout.splitlines() if line)
    if not diff.strip() and writes and not changed_paths:
        # Unit callers may exercise the role boundary without a Git checkout.
        # The coordinator still performs the authoritative Git diff gate.
        changed_paths = tuple(dict.fromkeys(writes))
        diff = "\n".join(f"modified: {path}" for path in changed_paths)
    if not diff.strip():
        return RepairImplementerResult(True, reason, writes, ImplementerStatus.NO_CHANGE, turns, "", changed_paths)
    policy = RepairPolicy()
    findings = (
        policy.allowed_path_violations(list(changed_paths), spec.allowed_paths)
        + policy.protected_violations(list(changed_paths))
        + policy.scan_test_diff(diff)
        + policy.semantic_violations(diff)
    )
    if findings:
        return RepairImplementerResult(
            True, f"policy: candidate rejected: {', '.join(findings)}", writes,
            ImplementerStatus.TOOL_POLICY_BLOCK, turns, diff, changed_paths,
        )
    return RepairImplementerResult(
        True, reason, writes, ImplementerStatus.PATCH_READY, turns, diff, changed_paths,
    )


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
