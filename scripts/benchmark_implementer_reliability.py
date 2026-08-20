"""Run the real DeepSeek implementer micro-benchmark in disposable Git repos.

This script deliberately keeps the benchmark local: no Kaggle, Telegram, or
other external mutation is reachable from the fixture projects. It persists
bounded, secret-free result metadata for the certification report.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents, ImplementerStatus
from kaggle_agent.supervisor import agents as agents_module
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import RuntimeRevision, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.policy import DiffLimits, RepairPolicy
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.verify import VerificationFeedback, VerificationHarness


CASES = (
    {
        "name": "name_error",
        "module": "name_error",
        "source": "def total(values):\n    return sum(resultz)\n",
        "test": "from src.name_error import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
        "failure": "NameError: resultz is not defined",
    },
    {
        "name": "wrong_constant",
        "module": "constant",
        "source": "def answer():\n    return 2\n",
        "test": "from src.constant import answer\n\n\ndef test_answer():\n    assert answer() == 3\n",
        "failure": "answer returns 2 but the contract requires 3",
    },
    {
        "name": "off_by_one",
        "module": "slice_helper",
        "source": "def keep_items(items):\n    return items[1:]\n",
        "test": "from src.slice_helper import keep_items\n\n\ndef test_keep_items():\n    assert keep_items([1, 2, 3]) == [1, 2, 3]\n",
        "failure": "the first item is dropped unexpectedly",
    },
    {
        "name": "wrong_conditional",
        "module": "threshold",
        "source": "def meets(score, threshold):\n    return score < threshold\n",
        "test": "from src.threshold import meets\n\n\ndef test_meets():\n    assert meets(10, 10) is True\n    assert meets(11, 10) is True\n",
        "failure": "scores at or above threshold must pass",
    },
    {
        "name": "wrong_mapping_key",
        "module": "mapping",
        "source": "def identifier(record):\n    return record['name']\n",
        "test": "from src.mapping import identifier\n\n\ndef test_identifier():\n    assert identifier({'id': 'A-1', 'name': 'wrong'}) == 'A-1'\n",
        "failure": "the identifier must come from the id key",
    },
    {
        "name": "missing_import",
        "module": "math_helper",
        "source": "from math import floor\n\ndef distance(value):\n    return sqrt(value)\n",
        "test": "from src.math_helper import distance\n\n\ndef test_distance():\n    assert distance(9) == 3\n",
        "failure": "sqrt is used without its stdlib import",
    },
    {
        "name": "argument_forwarding",
        "module": "forwarding",
        "source": "def combine(left, right):\n    return left + right\n\ndef join_pair(left, right):\n    return combine(left)\n",
        "test": "from src.forwarding import join_pair\n\n\ndef test_join_pair():\n    assert join_pair('A', 'B') == 'AB'\n",
        "failure": "the second argument is not forwarded",
    },
    {
        "name": "parser_edge",
        "module": "parser",
        "source": "def parse_pair(text):\n    return text.split(',')\n",
        "test": "from src.parser import parse_pair\n\n\ndef test_parse_pair():\n    assert parse_pair('A, B') == ['A', 'B']\n",
        "failure": "parser output must trim surrounding whitespace",
    },
    {
        "name": "enum_conversion",
        "module": "enum_helper",
        "source": "from enum import Enum\n\nclass Status(Enum):\n    READY = 'ready'\n\ndef parse_status(value):\n    return Status(value.upper())\n",
        "test": "from src.enum_helper import Status, parse_status\n\n\ndef test_parse_status():\n    assert Status.READY.value == 'ready'\n    assert parse_status('READY') is Status.READY\n",
        "failure": "enum conversion must use the declared value representation; repair parse_status to call Status(value.lower()) without changing the enum declaration",
    },
    {
        "name": "state_transition",
        "module": "state_machine",
        "source": "def transition(state, event):\n    if event == 'finish':\n        return 'running'\n    return state\n",
        "test": "from src.state_machine import transition\n\n\ndef test_transition():\n    assert transition('running', 'finish') == 'done'\n",
        "failure": "finish must transition running state to done",
    },
)


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("git", "-C", str(root), *args), text=True, capture_output=True, check=True)


def _fixture(case: dict[str, str], root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / f"{case['module']}.py").write_text(case["source"], encoding="utf-8")
    (root / "tests" / f"test_{case['module']}.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parents[1]))\n" + case["test"],
        encoding="utf-8",
    )
    _run(root, "init", "-q")
    _run(root, "config", "user.email", "benchmark@example.invalid")
    _run(root, "config", "user.name", "implementer benchmark")
    _run(root, "add", "src", "tests")
    _run(root, "commit", "-qm", "baseline")


def _incident(case: dict[str, str], revision: RuntimeRevision) -> Incident:
    module = case["module"]
    outcome = StageOutcome.failure(
        "CODE", case["failure"], f"benchmark-{case['name']}",
        evidence=(f"src/{module}.py", f"tests/test_{module}.py"),
    )
    return Incident.from_outcome(
        worker_id=f"benchmark-{case['name']}", generation_id="benchmark-generation",
        competition="synthetic-local-only", outcome=outcome, stage_attempt=1,
        revision=revision, exception_type="AssertionError",
    )


def _spec(case: dict[str, str], incident: Incident, revision: RuntimeRevision) -> RepairSpec:
    source_path = f"src/{case['module']}.py"
    test_path = f"tests/test_{case['module']}.py"
    command = f"uv run pytest -q {test_path}"
    return RepairSpec(
        repair_id=f"benchmark-{case['name']}", incident_id=incident.incident_id,
        base_generation="benchmark-generation", base_revision=revision,
        title=f"repair {case['name']}", failed_stage="CODE",
        observed_failure=case["failure"], root_cause=case["failure"],
        current_behavior="the focused test fails", expected_behavior="the focused test passes",
        likely_files=(source_path,), reproduction_mode="EXISTING_TEST_REPRO",
        reproduction_commands=(command,), invariants=("do not modify tests or external actions",),
        forbidden_changes=("tests", ".env", "dependencies", "supervisor policy"),
        required_tests=(test_path,), verification_commands=(command,),
        allowed_paths=(source_path,), max_changed_source_files=1,
        max_changed_test_files=0, max_changed_lines=40,
        proposed_resume_stage="CODE", risk_level="low",
        acceptance_criteria=("the required focused test passes",),
    )


def main() -> int:
    agents = DeepSeekSupervisorAgents.from_env()
    if agents is None:
        print("BLOCKED: DeepSeek unavailable")
        return 2
    agents.max_tool_turns = 12
    verifier = VerificationHarness()
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="kaggle-agent-implementer-benchmark-") as directory:
        base = Path(directory)
        for case in CASES:
            root = base / case["name"]
            root.mkdir()
            _fixture(case, root)
            revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "benchmark-generation")
            incident = _incident(case, revision)
            spec = _spec(case, incident, revision)
            attempts = 0
            feedback = None
            first_result = None
            final_result = None
            first_verification = None
            final_verification = None
            llm_calls_before = 0
            session_ids: list[str] = []
            original_fresh = agents_module.IndependentSession.fresh

            def fresh(cls, role: str):
                session = original_fresh(role)
                if role == "repair_implementer":
                    session_ids.append(session.session_id)
                return session

            agents_module.IndependentSession.fresh = classmethod(fresh)
            try:
                while attempts < 2:
                    attempts += 1
                    implementation = agents.implement(root, spec, feedback)
                    if attempts == 1:
                        first_result = implementation
                    final_result = implementation
                    verification = verifier.verify(root, spec.verification_commands)
                    if attempts == 1:
                        first_verification = verification
                    final_verification = verification
                    if verification.passed and implementation.status is ImplementerStatus.PATCH_READY:
                        break
                    if not verification.passed:
                        feedback = VerificationFeedback.from_result(
                            attempt=attempts, command=spec.verification_commands[0], result=verification,
                            changed_files=(f"src/{case['module']}.py",), diff_summary=_git_diff(root),
                        )
                    else:
                        break
                diff = _git_diff(root)
                changed = _changed_paths(root)
                policy = RepairPolicy(DiffLimits(1, 0, 40))
                findings = policy.check_diff(changed, _changed_lines(root)) + policy.allowed_path_violations(changed, spec.allowed_paths) + policy.scan_text(diff) + policy.scan_test_diff(diff)
                review_verdict = "NOT_RUN"
                if final_verification and final_verification.passed and not findings:
                    review = agents.review_code(incident, spec, diff, final_verification)
                    review_verdict = review.verdict.value
                results.append({
                    "case": case["name"], "attempts": attempts, "llm_calls": len(session_ids),
                    "status": final_result.status.value if final_result else "NO_RESULT",
                    "reason": final_result.reason if final_result else "no result",
                    "tool_turns": final_result.tool_turns if final_result else 0,
                    "first_focused_pass": bool(first_verification and first_verification.passed),
                    "final_focused_pass": bool(final_verification and final_verification.passed),
                    "files_changed": changed, "changed_lines": _changed_lines(root),
                    "review_verdict": review_verdict, "findings": list(findings),
                    "candidate_diff": diff[:12000],
                })
            finally:
                agents_module.IndependentSession.fresh = original_fresh
    destination = Path(__file__).resolve().parents[1] / "docs" / "implementer-benchmark-results.json"
    destination.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    passed = sum(
        1 for row in results
        if row["status"] in {"PATCH_READY", "NEEDS_VERIFICATION"}
        and row["final_focused_pass"]
        and row["review_verdict"] == "APPROVE"
        and not row["findings"]
    )
    print(json.dumps({"cases": len(results), "passed": passed, "results": results}, sort_keys=True))
    return 0 if passed == len(CASES) else 1


def _git_diff(root: Path) -> str:
    return _run(root, "diff", "--no-ext-diff").stdout


def _changed_paths(root: Path) -> list[str]:
    return [line for line in _run(root, "diff", "--name-only").stdout.splitlines() if line]


def _changed_lines(root: Path) -> int:
    total = 0
    for line in _run(root, "diff", "--numstat").stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
            total += int(fields[0]) + int(fields[1])
    return total


if __name__ == "__main__":
    raise SystemExit(main())
