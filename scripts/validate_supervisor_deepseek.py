"""Run the five supervisor DeepSeek roles against a disposable local defect.

This script never loads competition state, Kaggle credentials, or Telegram
credentials. It is intentionally opt-in and returns a non-zero status when
the production key is unavailable or a role returns an invalid artifact.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.verify import VerificationResult


def main() -> int:
    agents = DeepSeekSupervisorAgents.from_env()
    if agents is None:
        print("BLOCKED: DEEPSEEK_API_KEY is unavailable; no production role was called")
        return 2
    agents.max_tool_turns = 8

    incident = Incident.from_outcome(
        worker_id="smoke-worker",
        generation_id="smoke-generation",
        competition="synthetic-local-only",
        outcome=StageOutcome.failure(
            "CODE",
            "NameError: valuez is not defined",
            "smoke-signature",
            evidence=(
                "src/bug.py source: def total(values): return sum(valuez)",
                "tests/test_bug.py expects total([1, 2, 3]) == 6",
            ),
        ),
        stage_attempt=1,
        revision=RuntimeRevision("a" * 40, "b" * 40, "smoke-generation"),
        exception_type="NameError",
        traceback="Traceback (most recent call last):\n  File 'src/bug.py', line 2\nNameError: valuez is not defined",
    )
    classification = agents.classify(incident)
    spec = agents.author_spec(incident, classification, repair_id="smoke-repair")
    spec_review = agents.review_spec(incident, classification, spec)
    if spec_review.verdict.value != "APPROVE":
        print(f"REJECTED: spec review verdict={spec_review.verdict.value}")
        return 1

    with tempfile.TemporaryDirectory(prefix="kaggle-agent-supervisor-smoke-") as directory:
        root = Path(directory)
        (root / "src").mkdir()
        (root / "tests").mkdir()
        (root / "src" / "bug.py").write_text(
            "def total(values):\n    return sum(valuez)\n",
            encoding="utf-8",
        )
        (root / "tests" / "test_bug.py").write_text(
            "from src.bug import total\n\n\ndef test_total():\n    assert total([1, 2, 3]) == 6\n",
            encoding="utf-8",
        )
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)
        subprocess.run(("git", "config", "user.email", "smoke@example.invalid"), cwd=root, check=True)
        subprocess.run(("git", "config", "user.name", "supervisor smoke"), cwd=root, check=True)
        subprocess.run(("git", "add", "src/bug.py", "tests/test_bug.py"), cwd=root, check=True)
        subprocess.run(("git", "commit", "-qm", "baseline"), cwd=root, check=True)
        before = subprocess.run(
            (sys.executable, "-m", "pytest", "-q", "tests/test_bug.py"),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if before.returncode == 0:
            print("REJECTED: synthetic defect did not reproduce before implementation")
            return 1
        implementation = agents.implement(root, spec)
        diff = subprocess.run(("git", "diff", "--no-ext-diff"), cwd=root, text=True, capture_output=True, check=True).stdout
        after = subprocess.run(
            (sys.executable, "-m", "pytest", "-q", "tests/test_bug.py"),
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        verification = VerificationResult(
            bool(diff) and after.returncode == 0,
            ((sys.executable, "-m", "pytest", "-q", "tests/test_bug.py"),),
            ()
            if bool(diff) and after.returncode == 0
            else ("synthetic focused test failed:\n" + (after.stdout + after.stderr)[-4000:],),
        )
        review = agents.review_code(incident, spec, diff, verification)

    print(
        "RESULT: classifier=%s spec=%s spec_review=%s implementer=%s code_review=%s"
        % (classification.failure_class.value, spec.repair_id, spec_review.verdict.value, implementation.reason, review.verdict.value)
    )
    if review.blocking_findings or review.non_blocking_findings:
        print("CODE_REVIEW_FINDINGS:")
        for finding in review.blocking_findings + review.non_blocking_findings:
            print(f"- {finding.severity}: {finding.file}: {finding.issue} -> {finding.required_fix}")
    return 0 if review.verdict.value == "APPROVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
