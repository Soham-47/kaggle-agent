"""Run the five supervisor DeepSeek roles against a disposable local defect.

This script never loads competition state, Kaggle credentials, or Telegram
credentials. It is intentionally opt-in and returns a non-zero status when
the production key is unavailable or a role returns an invalid artifact.
"""

from __future__ import annotations

import subprocess
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

    incident = Incident.from_outcome(
        worker_id="smoke-worker",
        generation_id="smoke-generation",
        competition="synthetic-local-only",
        outcome=StageOutcome.failure("CODE", "NameError: missing_name", "smoke-signature"),
        stage_attempt=1,
        revision=RuntimeRevision("a" * 40, "b" * 40, "smoke-generation"),
        exception_type="NameError",
        traceback="Traceback (most recent call last):\n  File 'src/bug.py', line 1\nNameError: missing_name",
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
        (root / "src" / "bug.py").write_text("value = missing_name\n", encoding="utf-8")
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)
        subprocess.run(("git", "config", "user.email", "smoke@example.invalid"), cwd=root, check=True)
        subprocess.run(("git", "config", "user.name", "supervisor smoke"), cwd=root, check=True)
        subprocess.run(("git", "add", "src/bug.py"), cwd=root, check=True)
        subprocess.run(("git", "commit", "-qm", "baseline"), cwd=root, check=True)
        implementation = agents.implement(root, spec)
        diff = subprocess.run(("git", "diff", "--no-ext-diff"), cwd=root, text=True, capture_output=True, check=True).stdout
        verification = VerificationResult(bool(diff), (("synthetic", "local"),), () if diff else ("implementer produced no diff",))
        review = agents.review_code(incident, spec, diff, verification)

    print(
        "PASS: classifier=%s spec=%s spec_review=%s implementer=%s code_review=%s"
        % (classification.failure_class.value, spec.repair_id, spec_review.verdict.value, implementation.reason, review.verdict.value)
    )
    return 0 if review.verdict.value == "APPROVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
