"""Bounded deterministic coordinator for an approved repair attempt."""

from __future__ import annotations

import hashlib
import inspect
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from kaggle_agent.supervisor.budgets import RepairBudgetStore
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeRevision, read_git_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.impact import StageImpactAnalyzer
from kaggle_agent.supervisor.policy import DiffLimits, RepairPolicy
from kaggle_agent.supervisor.promote import GenerationPromotion, RepairAcceptance
from kaggle_agent.supervisor.resume import ResumeRequest, invalidated_stages
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.state import SupervisorStateStore
from kaggle_agent.supervisor.verify import VerificationFeedback, VerificationHarness
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
    implementer_attempts: int = 0
    verification_feedback: tuple[dict[str, object], ...] = ()


class RepairCoordinator:
    def __init__(self, source_root: Path, state: SupervisorStateStore) -> None:
        self.source_root = source_root.resolve()
        self.state = state
        self.worktrees = WorktreeManager(self.source_root, state.layout.state_root)
        self.generations = GenerationStore(state)
        self.policy = RepairPolicy()
        self.verifier = VerificationHarness()
        self.budgets = RepairBudgetStore(state.layout.state_root)

    def execute(self, incident: Incident, classification: FailureClassification, spec: RepairSpec, *, spec_approved: bool, implementer: Callable[..., object], reviewer: Callable[[Incident, RepairSpec, str, object], Review], cycle_id: str | None = None, mode: str = "auto_safe", max_implementation_attempts: int = 2) -> RepairFlowResult:
        if mode not in {"repair_only", "auto_safe"}:
            raise ValueError(f"unsupported repair mode: {mode}")
        if classification.failure_class not in {FailureClass.CODE_DEFECT, FailureClass.TEST_FAILURE} or not classification.repairable:
            return RepairFlowResult("NOT_REPAIRABLE", RepairAcceptance())
        if not spec_approved or not self.budgets.available(incident.incident_id, incident.failure_signature, cycle_id):
            return RepairFlowResult("REJECTED", RepairAcceptance(spec_approved=spec_approved))
        attempt = len(list((self.state.layout.state_root / "worktrees" / incident.incident_id).glob("a*"))) + 1
        worktree = self.worktrees.create(incident.incident_id, attempt, spec.base_revision.git_sha)
        try:
            feedback: VerificationFeedback | None = None
            feedback_rows: list[dict[str, object]] = []
            previous_fingerprint: str | None = None
            verification = None
            diff = ""
            paths: list[str] = []
            policy_findings: tuple[str, ...] = ()
            test_findings: tuple[str, ...] = ()
            static_findings: tuple[str, ...] = ()
            attempts = 0
            max_attempts = max(1, max_implementation_attempts)
            while attempts < max_attempts:
                attempts += 1
                implementation = self._call_implementer(implementer, worktree, spec, feedback)
                status = getattr(getattr(implementation, "status", None), "value", getattr(implementation, "status", None))
                diff = self.worktrees.diff(worktree)
                paths = self._changed_paths(worktree)
                changed_lines = self._changed_lines(worktree)
                candidate_policy = RepairPolicy(DiffLimits(
                    max_changed_source_files=spec.max_changed_source_files,
                    max_changed_test_files=spec.max_changed_test_files,
                    max_changed_lines=spec.max_changed_lines,
                ))
                policy_findings = (
                    candidate_policy.check_diff(paths, changed_lines)
                    + candidate_policy.allowed_path_violations(paths, spec.allowed_paths)
                    + candidate_policy.semantic_violations(diff)
                )
                test_findings = candidate_policy.scan_test_diff(diff)
                static_findings = candidate_policy.scan_text(diff)
                if status not in {None, "PATCH_READY", "NEEDS_VERIFICATION"}:
                    return RepairFlowResult(
                        "REJECTED", RepairAcceptance(), findings=(f"implementer:{status or 'unknown'}",),
                        implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
                    )
                if not diff.strip():
                    return RepairFlowResult(
                        "REJECTED", RepairAcceptance(), findings=("no_candidate_diff",),
                        implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
                    )
                if policy_findings or test_findings or static_findings:
                    return RepairFlowResult(
                        "REJECTED", RepairAcceptance(),
                        findings=tuple(policy_findings) + tuple(test_findings) + tuple(static_findings),
                        implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
                    )
                fingerprint = hashlib.sha256(diff.encode("utf-8")).hexdigest()
                if previous_fingerprint == fingerprint:
                    return RepairFlowResult(
                        "REJECTED", RepairAcceptance(), findings=("REPEATED_BAD_PATCH",),
                        implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
                    )
                previous_fingerprint = fingerprint
                verification = self.verifier.verify(worktree, spec.verification_commands)
                if verification.passed:
                    break
                command = " && ".join(spec.verification_commands) or "uv run python -m compileall src"
                feedback = VerificationFeedback.from_result(
                    attempt=attempts, command=command, result=verification,
                    changed_files=tuple(paths), diff_summary=diff,
                )
                feedback_rows.append(feedback.to_dict())
                self.state.write_json(
                    f"repairs/{spec.repair_id}/attempt-{attempts}-feedback.json", feedback.to_dict()
                )
            else:
                return RepairFlowResult(
                    "REJECTED", RepairAcceptance(), findings=("implementation_attempt_budget_exhausted",),
                    implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
                )
            assert verification is not None
            review = reviewer(incident, spec, diff, verification)
            acceptance = RepairAcceptance(
                classification_allows_repair=True, spec_approved=True,
                base_revision_matches=read_git_revision(worktree) == spec.base_revision.git_sha,
                protected_paths_pass=not self.policy.protected_violations(paths),
                protected_semantics_pass=not self.policy.semantic_violations(diff),
                reproduction_pass=verification.passed, focused_tests_pass=verification.passed,
                subsystem_tests_pass=verification.passed, full_tests_pass=verification.passed,
                diff_limits_pass=bool(diff.strip()) and not policy_findings,
                static_safety_pass=not static_findings, test_integrity_pass=not test_findings,
                review_approved=review.verdict is ReviewVerdict.APPROVE and not review.blocking_findings,
                external_state_safe=True, repair_budget_available=True,
                resume_plan_valid=bool(spec.proposed_resume_stage),
            )
            self.budgets.record(incident.incident_id, incident.failure_signature, cycle_id, accepted=acceptance.accepted)
            if not acceptance.accepted:
                return RepairFlowResult("REJECTED", acceptance, review=review, findings=tuple(policy_findings) + tuple(test_findings), implementer_attempts=attempts, verification_feedback=tuple(feedback_rows))
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
                    implementer_attempts=attempts, verification_feedback=tuple(feedback_rows),
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
            return RepairFlowResult("ACCEPTED", acceptance, generation.generation_id, resume, review, implementer_attempts=attempts, verification_feedback=tuple(feedback_rows))
        finally:
            if worktree.exists() and mode != "repair_only":
                self.worktrees.destroy(worktree)

    @staticmethod
    def _call_implementer(implementer: Callable[..., object], worktree: Path, spec: RepairSpec, feedback: VerificationFeedback | None) -> object:
        try:
            parameters = inspect.signature(implementer).parameters.values()
            accepts_feedback = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                or parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            ) or len(inspect.signature(implementer).parameters) >= 3
        except (TypeError, ValueError):
            accepts_feedback = False
        if accepts_feedback:
            return implementer(worktree, spec, feedback)
        return implementer(worktree, spec)

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
