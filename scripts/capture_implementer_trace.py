"""Capture one sanitized real DeepSeek implementer attempt.

This is a validation harness. It uses the existing production role wiring and
does not call Kaggle or Telegram. The output is intentionally bounded and
redacted before it is persisted.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor import agents as agents_module
from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident


_SECRET = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer\s+)?|api[_-]?key\s*[=:]\s*|"
    r"token\s*[=:]\s*|password\s*[=:]\s*|secret\s*[=:]\s*)[^\s,;]+"
)


def _sanitize(value: object, *, limit: int = 4_000) -> str:
    text = str(value)
    text = _SECRET.sub(lambda match: match.group(1) + "<redacted>", text)
    text = re.sub(r"(?i)(DEEPSEEK_API_KEY|KAGGLE_USERNAME|KAGGLE_KEY|TELEGRAM_BOT_TOKEN)=\S+", r"\1=<redacted>", text)
    return text[:limit]


def _fixture_incident() -> Incident:
    outcome = StageOutcome.failure(
        "CODE",
        "NameError: valuez is not defined",
        "smoke-signature",
        evidence=(
            "src/bug.py source: def total(values): return sum(valuez)",
            "tests/test_bug.py expects total([1, 2, 3]) == 6",
        ),
    )
    return Incident.from_outcome(
        worker_id="trace-worker",
        generation_id="trace-generation",
        competition="synthetic-local-only",
        outcome=outcome,
        stage_attempt=1,
        revision=RuntimeRevision("a" * 40, "b" * 40, "trace-generation"),
        exception_type="NameError",
        traceback="Traceback (most recent call last):\n  File 'src/bug.py', line 2\nNameError: valuez is not defined",
    )


def main() -> int:
    trace: dict[str, object] = {
        "repair_id": "smoke-repair",
        "session_ids": [],
        "spec_id": None,
        "allowed_paths": [],
        "tools": [],
        "files_read": [],
        "files_written": [],
        "candidate_diff": "",
        "focused_test": {},
        "implementer_stop_reason": None,
        "turn_count": 0,
        "implementer_status": None,
        "llm_call_count": 0,
        "verification": {},
        "failure_categories": [],
    }
    agents = DeepSeekSupervisorAgents.from_env()
    if agents is None:
        trace["failure_categories"] = ["OTHER"]
        trace["implementer_stop_reason"] = "DEEPSEEK_API_KEY unavailable"
        _write_trace(trace)
        print("BLOCKED: DeepSeek unavailable")
        return 2
    agents.max_tool_turns = 8

    original_fresh = agents_module.IndependentSession.fresh

    def fresh(cls, role: str):
        session = original_fresh(role)
        trace["session_ids"].append({"role": role, "session_id": session.session_id})
        return session

    agents_module.IndependentSession.fresh = classmethod(fresh)
    original_boundary = agents_module.RepairAgentBoundary

    class TracedBoundary:
        def __init__(self, root: Path) -> None:
            self.inner = original_boundary(root)

        @property
        def tools_allowed(self):
            return self.inner.tools_allowed

        def call(self, name: str, **args):
            row = {"tool": name, "args": {key: _sanitize(value) for key, value in args.items()}}
            try:
                result = self.inner.call(name, **args)
            except Exception as exc:  # noqa: BLE001 - trace must record policy/tool errors
                row["error"] = _sanitize(exc)
                trace["tools"].append(row)
                raise
            row["result"] = _sanitize(result)
            trace["tools"].append(row)
            if name in {"read_file", "search_code"}:
                path = args.get("path") or args.get("file_path") or name
                trace["files_read"].append(_sanitize(path))
            if name in {"write_file", "apply_patch"}:
                path = args.get("path") or args.get("file_path")
                if not path and name == "apply_patch":
                    path = next(
                        (line[6:].strip() for line in str(args.get("patch", "")).splitlines() if line.startswith("+++ b/")),
                        name,
                    )
                trace["files_written"].append(_sanitize(path or name))
            return result

    agents_module.RepairAgentBoundary = TracedBoundary
    try:
        incident = _fixture_incident()
        classification = agents.classify(incident)
        spec = agents.author_spec(incident, classification, repair_id="smoke-repair")
        trace["spec_id"] = spec.repair_id
        trace["allowed_paths"] = list(spec.allowed_paths)
        with tempfile.TemporaryDirectory(prefix="kaggle-agent-implementer-trace-") as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "tests").mkdir()
            (root / "src" / "__init__.py").write_text("", encoding="utf-8")
            (root / "src" / "bug.py").write_text("def total(values):\n    return sum(valuez)\n", encoding="utf-8")
            (root / "tests" / "test_bug.py").write_text(
                "import sys\nfrom pathlib import Path\n\nsys.path.insert(0, str(Path(__file__).parents[1]))\nfrom src.bug import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
                encoding="utf-8",
            )
            for command in (
                ("git", "init", "-q"),
                ("git", "config", "user.email", "trace@example.invalid"),
                ("git", "config", "user.name", "implementer trace"),
                ("git", "add", "src/__init__.py", "src/bug.py", "tests/test_bug.py"),
                ("git", "commit", "-qm", "baseline"),
            ):
                subprocess.run(command, cwd=root, check=True)
            implementation = agents.implement(root, spec)
            trace["implementer_stop_reason"] = _sanitize(implementation.reason)
            trace["turn_count"] = implementation.tool_turns
            trace["implementer_status"] = implementation.status.value
            trace["llm_call_count"] = sum(
                1 for row in trace["session_ids"] if row.get("role") == "repair_implementer"
            )
            diff = subprocess.run(
                ("git", "diff", "--no-ext-diff"), cwd=root, text=True, capture_output=True, check=True
            ).stdout
            trace["candidate_diff"] = _sanitize(diff, limit=12_000)
            focused_command = ("uv", "run", "pytest", "-q", "tests/test_bug.py")
            test = subprocess.run(
                focused_command,
                cwd=root, text=True, capture_output=True, check=False,
            )
            trace["focused_test"] = {
                "command": list(focused_command),
                "exit_code": test.returncode,
                "stdout": _sanitize(test.stdout),
                "stderr": _sanitize(test.stderr),
            }
            trace["verification"] = {"passed": bool(diff.strip()) and test.returncode == 0}
            categories: list[str] = []
            if not diff.strip():
                categories.append("NO_PATCH")
            if not trace["files_written"]:
                categories.append("NO_PATCH")
            if test.returncode != 0:
                categories.append("TEST_NOT_RUN" if not test.stdout and not test.stderr else "INCOMPLETE_PATCH")
            if "budget exhausted" in str(implementation.reason).lower():
                categories.append("LOOP_EXHAUSTED")
            if implementation.status.value == "PROTOCOL_FAILURE":
                categories.append("TOOL_PROTOCOL_ERROR")
            trace["failure_categories"] = sorted(set(categories))
    except Exception as exc:  # noqa: BLE001 - trace the real failure, then fail closed
        trace["implementer_stop_reason"] = _sanitize(exc)
        trace["failure_categories"] = ["OTHER"]
    finally:
        _write_trace(trace)
        agents_module.RepairAgentBoundary = original_boundary
        agents_module.IndependentSession.fresh = original_fresh

    print(json.dumps({key: value for key, value in trace.items() if key != "candidate_diff"}, sort_keys=True))
    return 0 if trace["verification"].get("passed") else 1


def _write_trace(trace: dict[str, object]) -> None:
    destination = Path(__file__).resolve().parents[1] / "docs" / "implementer-attempt-trace.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
