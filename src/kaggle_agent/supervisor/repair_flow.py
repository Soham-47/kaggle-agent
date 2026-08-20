"""Bounded deterministic coordinator for an approved repair attempt."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from kaggle_agent.supervisor.budgets import RepairBudgetStore
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeRevision, read_git_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.impact import StageImpactAnalyzer
from kaggle_agent.supervisor.policy import RepairPolicy
from kaggle_agent.supervisor.promote import GenerationPromotion, RepairAcceptance
from kaggle_agent.supervisor.resume import ResumeRequest, invalidated_stages
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.state import SupervisorStateStore
from kaggle_agent.supervisor.verify import VerificationHarness
from kaggle_agent.supervisor.worktree import WorktreeManager


@dataclass(frozen=True)
class RepairFlowResult:
    status: str
    acceptance: RepairAcceptance
    generation_id: str | None = None
    resume_request: ResumeRequest | None = None
    review: Review | None = None
    findings: tuple[str, ...] = ()
    candidate_path: str | None = None
    candidate_revision: str | None = None


class RepairCoordinator:
    def __init__(self, source_root: Path, state: SupervisorStateStore) -> None:
        self.source_root = source_root.resolve()
        self.state = state
        self.worktrees = WorktreeManager(self.source_root, state.layout.state_root)
        self.generations = GenerationStore(state)
        self.policy = RepairPolicy()
        self.verifier = VerificationHarness()
        self.budgets = RepairBudgetStore(state.layout.state_root)

    def execute(self, incident: Incident, classification: FailureClassification, spec: RepairSpec, *, spec_approved: bool, implementer: Callable[[Path, RepairSpec], None], reviewer: Callable[[Incident, RepairSpec, str, object], Review], cycle_id: str | None = None, mode: str = "auto_safe") -> RepairFlowResult:
        if mode not in {"repair_only", "auto_safe"}:
            raise ValueError(f"unsupported repair mode: {mode}")
        if classification.failure_class not in {FailureClass.CODE_DEFECT, FailureClass.TEST_FAILURE} or not classification.repairable:
            return RepairFlowResult("NOT_REPAIRABLE", RepairAcceptance())
        if not spec_approved or not self.budgets.available(incident.incident_id, incident.failure_signature, cycle_id):
            return RepairFlowResult("REJECTED", RepairAcceptance(spec_approved=spec_approved))
        attempt = len(list((self.state.layout.state_root / "worktrees" / incident.incident_id).glob("a*"))) + 1
        worktree = self.worktrees.create(incident.incident_id, attempt, spec.base_revision.git_sha)
        try:
            implementer(worktree, spec)
            diff = self.worktrees.diff(worktree)
            paths = self._changed_paths(worktree)
            changed_lines = self._changed_lines(worktree)
            policy_findings = self.policy.check_diff(paths, changed_lines) + self.policy.semantic_violations(diff)
            test_findings = self.policy.scan_test_diff(diff)
            verification = self.verifier.verify(worktree, spec.verification_commands)
            review = reviewer(incident, spec, diff, verification)
            acceptance = RepairAcceptance(
                classification_allows_repair=True, spec_approved=True,
                base_revision_matches=read_git_revision(worktree) == spec.base_revision.git_sha,
                protected_paths_pass=not self.policy.protected_violations(paths),
                protected_semantics_pass=not self.policy.semantic_violations(diff),
                reproduction_pass=verification.passed, focused_tests_pass=verification.passed,
                subsystem_tests_pass=verification.passed, full_tests_pass=verification.passed,
                diff_limits_pass=not any(item in policy_findings for item in ("source_file_limit", "test_file_limit", "changed_line_limit", "dependency_change")),
                static_safety_pass=not self.policy.scan_text(diff), test_integrity_pass=not test_findings,
                review_approved=review.verdict is ReviewVerdict.APPROVE and not review.blocking_findings,
                external_state_safe=True, repair_budget_available=True,
                resume_plan_valid=bool(spec.proposed_resume_stage),
            )
            self.budgets.record(incident.incident_id, incident.failure_signature, cycle_id, accepted=acceptance.accepted)
            if not acceptance.accepted:
                return RepairFlowResult("REJECTED", acceptance, review=review, findings=tuple(policy_findings) + tuple(test_findings))
            repair_sha = self.worktrees.commit(worktree, f"repair({incident.incident_id}): {spec.title}")
            if mode == "repair_only":
                self.state.write_json(
                    f"accepted/{spec.repair_id}.json",
                    {
                        "repair_id": spec.repair_id,
                        "incident_id": incident.incident_id,
                        "base_revision": spec.base_revision.git_sha,
                        "candidate_revision": repair_sha,
                        "candidate_path": str(worktree),
                        "diff": diff,
                        "review": asdict(review),
                        "status": "CANDIDATE_ACCEPTED",
                    },
                )
                return RepairFlowResult(
                    "CANDIDATE_ACCEPTED", acceptance, review=review,
                    candidate_path=str(worktree), candidate_revision=repair_sha,
                )
            generation = self.generations.create_from_revision(self.source_root, repair_sha, parent_generation=incident.generation_id, repair_id=spec.repair_id)
            invalidated = invalidated_stages(spec.proposed_resume_stage)
            resume = ResumeRequest(
                cycle_id or "", incident.incident_id, incident.generation_id,
                generation.generation_id, incident.stage, spec.proposed_resume_stage,
                tuple(s for s in invalidated if s != spec.proposed_resume_stage),
                invalidated, tuple(x for x in (incident.external_job, incident.kernel_ref) if x),
                tuple((stage, 1) for stage in invalidated),
            )
            self.state.write_json(f"repairs/{spec.repair_id}/acceptance.json", {"acceptance": acceptance.__dict__, "generation": generation.to_dict(), "resume": resume.to_dict()})
            return RepairFlowResult("ACCEPTED", acceptance, generation.generation_id, resume, review)
        finally:
            if worktree.exists() and mode != "repair_only":
                self.worktrees.destroy(worktree)

    def _changed_paths(self, worktree: Path) -> list[str]:
        result = subprocess.run(("git", "-C", str(worktree), "diff", "--name-only", "HEAD"), text=True, capture_output=True, check=False)
        return [line for line in result.stdout.splitlines() if line]

    def _changed_lines(self, worktree: Path) -> int:
        result = subprocess.run(("git", "-C", str(worktree), "diff", "--numstat", "HEAD"), text=True, capture_output=True, check=False)
        total = 0
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[0].isdigit() and fields[1].isdigit():
                total += int(fields[0]) + int(fields[1])
        return total
