"""Exercise one real DeepSeek REPAIR_ONLY lifecycle in a disposable repo."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents
from kaggle_agent.supervisor.classifier import FailureClass
from kaggle_agent.supervisor.generation import RuntimeRevision, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


def _git(root: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)


def _create_fixture(root: Path) -> None:
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "bug.py").write_text(
        "def total(values):\n    return sum(valuez)\n", encoding="utf-8"
    )
    (root / "tests" / "test_bug.py").write_text(
        "import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path(__file__).parents[1]))\n"
        "from src.bug import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "repair-only@example.invalid")
    _git(root, "config", "user.name", "repair-only validation")
    _git(root, "add", "src", "tests")
    _git(root, "commit", "-qm", "baseline")


def main() -> int:
    agents = DeepSeekSupervisorAgents.from_env()
    if agents is None:
        print("BLOCKED: DeepSeek unavailable")
        return 2
    agents.max_tool_turns = 12
    with tempfile.TemporaryDirectory(prefix="kaggle-agent-repair-only-") as directory:
        root = Path(directory) / "repo"
        root.mkdir()
        _create_fixture(root)
        state_root = Path(directory) / "state"
        state = SupervisorStateStore(RuntimeLayout.for_repo(root, state_root))
        state.write_json("active-generation.json", {"generation_id": "generation-0001"})
        revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
        incident = Incident.from_outcome(
            worker_id="repair-only-worker", generation_id="generation-0001",
            competition="synthetic-local-only",
            outcome=StageOutcome.failure(
                "CODE", "NameError: valuez is not defined", "repair-only-name-error",
                evidence=("src/bug.py", "tests/test_bug.py"),
            ),
            stage_attempt=1, revision=revision, exception_type="NameError",
            traceback="Traceback (most recent call last): NameError: valuez is not defined",
        )
        classification = agents.classify(incident)
        spec = agents.author_spec(incident, classification, repair_id="repair-only-real")
        spec_review = agents.review_spec(incident, classification, spec)
        if classification.failure_class is not FailureClass.CODE_DEFECT or spec_review.verdict.value != "APPROVE":
            result = {
                "classification": classification.failure_class.value,
                "confidence": classification.confidence,
                "spec_review": spec_review.verdict.value,
                "status": "BLOCKED_BEFORE_REPAIR",
            }
            _write_result(result)
            print(json.dumps(result, sort_keys=True))
            return 1
        coordinator = RepairCoordinator(root, state)
        flow = coordinator.execute(
            incident,
            classification,
            spec,
            spec_approved=True,
            implementer=lambda worktree, approved, feedback=None: agents.implement(worktree, approved, feedback, state_root=state.layout.state_root),
            reviewer=lambda inc, approved, diff, verification: agents.review_code(inc, approved, diff, verification),
            cycle_id="repair-only-cycle",
            mode="repair_only",
        )
        active_after = state.read_json("active-generation.json")["generation_id"]
        result = {
            "classification": classification.failure_class.value,
            "confidence": classification.confidence,
            "spec_review": spec_review.verdict.value,
            "repair_id": spec.repair_id,
            "allowed_paths": list(spec.allowed_paths),
            "spec_limits": {
                "max_changed_source_files": spec.max_changed_source_files,
                "max_changed_test_files": spec.max_changed_test_files,
                "max_changed_lines": spec.max_changed_lines,
            },
            "verification_commands": list(spec.verification_commands),
            "status": flow.status,
            "candidate_revision": flow.candidate_revision,
            "candidate_path": flow.candidate_path,
            "implementer_attempts": flow.implementer_attempts,
            "verification_feedback": list(flow.verification_feedback),
            "active_before": "generation-0001",
            "active_after": active_after,
            "active_generation_unchanged": active_after == "generation-0001",
            "review_verdict": flow.review.verdict.value if flow.review else None,
            "findings": list(flow.findings),
        }
    _write_result(result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "CANDIDATE_ACCEPTED" and result["active_generation_unchanged"] else 1


def _write_result(result: dict[str, object]) -> None:
    destination = Path(__file__).resolve().parents[1] / "docs" / "repair-only-certification.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
