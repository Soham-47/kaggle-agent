"""Supervisor lifecycle coordinator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.config import Settings
from kaggle_agent.supervisor.generation import GenerationStore, read_git_revision
from kaggle_agent.supervisor.commands import SupervisorCommandQueue
from kaggle_agent.supervisor.agents import AgentProtocolError, DeepSeekSupervisorAgents
from kaggle_agent.supervisor.classifier import FailureClass, classify_after_reconciliation
from kaggle_agent.supervisor.incidents import IncidentStore
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.spec import SpecReviewVerdict
from kaggle_agent.autonomy.outbox import ExternalActionOutbox
from kaggle_agent.supervisor.lock import SupervisorLock
from kaggle_agent.supervisor.policy import RepairPolicy, SafetyViolation
from kaggle_agent.supervisor.protocol import WorkerRequest
from kaggle_agent.supervisor.recovery import SupervisorRecovery
from kaggle_agent.supervisor.resume import ResumeRequest
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.worker import WorkerLauncher


@dataclass(frozen=True)
class SupervisorRun:
    status: str
    worker_id: str | None = None
    reason: str = ""


class Supervisor:
    def __init__(self, settings: Settings, root: Path | None = None) -> None:
        self.settings = settings
        self.root = (root or settings.root).resolve()
        self.layout = RuntimeLayout.for_repo(self.root)
        self.store = SupervisorStateStore(self.layout)
        self.lock = SupervisorLock(self.layout.state_root / "supervisor.lock")

    def run_once(self, *, competition: str | None = None, wait: bool = True) -> SupervisorRun:
        config = self.settings.supervisor_config()
        if not config.enabled or config.mode == "off":
            return SupervisorRun("OFF", reason="supervisor disabled")
        if not self.lock.acquire():
            return SupervisorRun("LOCKED", reason="supervisor lock held")
        try:
            command_queue = SupervisorCommandQueue(self.layout)
            for command in command_queue.drain():
                if command.command == "pause":
                    command_queue.set_paused(True)
                elif command.command == "resume":
                    command_queue.set_paused(False)
            if command_queue.paused():
                return SupervisorRun("PAUSED", reason="supervisor command queue is paused")
            recovery = self.recover_startup(timeout_seconds=config.heartbeat_timeout_seconds)
            adopted = next((item for item in recovery if item.action == "ADOPT"), None)
            if adopted is not None:
                return SupervisorRun("ADOPTED", worker_id=adopted.worker_id, reason="owned worker heartbeat is fresh")
            blocked = next((item for item in recovery if item.action == "TERMINATE_OR_RECONCILE"), None)
            if blocked is not None:
                return SupervisorRun(
                    "RECOVERY_BLOCKED",
                    worker_id=blocked.worker_id,
                    reason="live worker ownership or heartbeat is unsafe; reconcile before launching another worker",
                )
            selected_competition = competition or self.settings.default_competition
            if not selected_competition:
                return SupervisorRun("NO_COMPETITION", reason="pass --competition or initialize a competition")
            resumed = self._resume_promoted_if_needed(selected_competition, config.mode, config.heartbeat_timeout_seconds)
            if resumed is not None:
                return resumed
            if config.mode == "auto_safe":
                try:
                    RepairPolicy().require_clean_auto_safe(self.root)
                except SafetyViolation as exc:
                    return SupervisorRun("DIRTY_SOURCE_BASELINE", reason=str(exc))
            generation = self._active_generation(managed=config.mode == "auto_safe")
            worker_id, result = self._start_worker(
                generation, selected_competition, config.mode, wait=wait,
            )
            if result is not None and result.get("status"):
                follow_up = self._handle_worker_result(result, config.mode, selected_competition)
                if follow_up is not None:
                    return SupervisorRun(follow_up[0], worker_id=worker_id, reason=follow_up[1])
                return SupervisorRun(str(result["status"]), worker_id=worker_id, reason=str(result.get("exit_reason") or ""))
            return SupervisorRun("WORKER_STARTED", worker_id=worker_id)
        finally:
            self.lock.release()

    def _active_generation(self, *, managed: bool = False):
        from kaggle_agent.supervisor.generation import RuntimeGeneration

        value = self.store.read_json("active-generation.json")
        if isinstance(value, dict):
            return RuntimeGeneration.from_dict(value)
        generation = GenerationStore(self.store).create_managed(self.root) if managed else GenerationStore(self.store).create(self.root)
        if managed:
            self.store.write_json("active-generation.json", generation.to_dict())
        return generation

    def recover_startup(self, *, timeout_seconds: float) -> tuple[object, ...]:
        """Adopt safe workers and settle an interrupted promotion before launch."""
        recovery = SupervisorRecovery(self.store)
        transaction = self.store.read_json("promotion.json")
        if isinstance(transaction, dict) and transaction.get("status") == "PREPARED":
            generations = GenerationStore(self.store)
            new = generations.load(str(transaction.get("new_generation") or ""))
            old = generations.load(str(transaction.get("old_generation") or "")) if transaction.get("old_generation") else None
            if new is not None:
                from kaggle_agent.supervisor.promote import GenerationPromotion

                GenerationPromotion(self.store).recover_interrupted(old, new)
        return recovery.recover_workers(timeout_seconds=timeout_seconds, owner_token=self.lock.owner_token)

    def _handle_worker_result(self, result: dict[str, object], mode: str, competition: str) -> tuple[str, str] | None:
        if str(result.get("status")) != "RECOVERABLE_FAILURE":
            return None
        incident_id = str(result.get("incident_id") or "")
        incident = IncidentStore(self.layout.state_root).load(incident_id) if incident_id else None
        if incident is None:
            return ("NEEDS_AUTHORITY", "recoverable worker result has no durable incident")
        outbox = ExternalActionOutbox(self.root, state_root=self.layout.state_root)
        classification = classify_after_reconciliation(incident, outbox, lambda item: item)
        sessions = DeepSeekSupervisorAgents.from_env()
        if classification.failure_class is FailureClass.UNKNOWN:
            if sessions is None:
                return ("NEEDS_AUTHORITY", "DEEPSEEK_API_KEY is unavailable for ambiguous classification")
            try:
                classification = sessions.classify(incident)
            except AgentProtocolError as exc:
                return ("NEEDS_AUTHORITY", str(exc))
        self.store.write_json(
            f"incidents/{incident.incident_id}/classification.json",
            {
                "failure_class": classification.failure_class.value,
                "confidence": classification.confidence,
                "repairable": classification.repairable,
                "likely_files": list(classification.likely_files),
                "reason": classification.reason,
            },
        )
        if classification.failure_class not in {FailureClass.CODE_DEFECT, FailureClass.TEST_FAILURE} or not classification.repairable:
            return (classification.failure_class.value, classification.reason)
        supervisor_config = self.settings.supervisor_config()
        if not supervisor_config.repair.enabled:
            return ("NEEDS_AUTHORITY", "supervisor repair is disabled")
        if classification.confidence < supervisor_config.repair.classification_min_confidence:
            return ("NEEDS_AUTHORITY", "classification confidence is below repair threshold")
        if sessions is None:
            return ("NEEDS_AUTHORITY", "DEEPSEEK_API_KEY is unavailable for repair roles")
        repair_id = f"repair-{incident.incident_id[:12]}-a1"
        try:
            spec = sessions.author_spec(incident, classification, repair_id=repair_id)
            spec.save(self.layout.state_root)
            spec_review = sessions.review_spec(incident, classification, spec)
            self.store.write_json(
                f"reviews/{repair_id}-spec.json",
                {
                    "verdict": spec_review.verdict.value,
                    "blocking_findings": list(spec_review.blocking_findings),
                    "non_blocking_findings": list(spec_review.non_blocking_findings),
                    "rationale": spec_review.rationale,
                },
            )
        except (AgentProtocolError, OSError, ValueError) as exc:
            return ("NEEDS_AUTHORITY", str(exc))
        limits = supervisor_config.repair
        if (
            spec.max_changed_source_files > limits.max_changed_source_files
            or spec.max_changed_test_files > limits.max_changed_test_files
            or spec.max_changed_lines > limits.max_changed_lines
        ):
            return ("REJECTED", "RepairSpec exceeds configured repair limits")
        if mode == "observe":
            return ("SPEC_READY", repair_id)
        if spec_review.verdict is not SpecReviewVerdict.APPROVE:
            return ("REJECTED", "independent spec review did not approve")
        if mode == "auto_safe" and not supervisor_config.promotion_automatic:
            return ("NEEDS_AUTHORITY", "automatic promotion is disabled unless supervisor.promotion.automatic is true")
        if mode not in {"repair_only", "auto_safe"}:
            return ("NEEDS_AUTHORITY", "automatic activation is disabled for this supervisor mode")
        coordinator = RepairCoordinator(
            self.root,
            self.store,
            max_attempts_per_incident=limits.max_attempts_per_incident,
            max_repairs_per_cycle=limits.max_repairs_per_cycle,
            max_repairs_per_day=limits.max_repairs_per_day,
        )
        try:
            outcome = coordinator.execute(
                incident, classification, spec, spec_approved=True,
                implementer=lambda worktree, approved, feedback=None: self._run_implementer(sessions, worktree, approved, feedback, self.store.layout.state_root),
                reviewer=lambda inc, approved, diff, verification: sessions.review_code(inc, approved, diff, verification),
                cycle_id=incident.cycle_id,
                mode=mode,
                max_implementation_attempts=limits.max_attempts_per_incident,
            )
        except (AgentProtocolError, OSError, RuntimeError, ValueError) as exc:
            return ("REJECTED", str(exc))
        if mode == "auto_safe" and outcome.status == "ACCEPTED":
            return self._promote_and_resume(outcome, incident, selected_competition=competition)
        return (outcome.status, outcome.candidate_revision or "candidate stored")

    def _start_worker(
        self,
        generation,
        competition: str,
        mode: str,
        *,
        wait: bool,
        worker_id: str | None = None,
        resume_request: ResumeRequest | None = None,
        incident_id: str | None = None,
    ) -> tuple[str, dict[str, object] | None]:
        worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        request = WorkerRequest(
            worker_id,
            generation.generation_id,
            competition,
            resume_request.cycle_id if resume_request else None,
            mode,
            resume_request.resume_from_stage if resume_request else None,
            incident_id,
            generation.revision,
            resume_request,
        )
        process = WorkerLauncher(self.layout).start(request, cwd=Path(generation.path))
        self.store.write_json(
            f"workers/{worker_id}/metadata.json",
            {
                "pid": process.pid,
                "worker_id": worker_id,
                "generation_id": generation.generation_id,
                "supervisor_token": self.lock.owner_token,
                "resume_request": resume_request.to_dict() if resume_request else None,
            },
        )
        if not wait:
            return worker_id, None
        process.wait()
        return worker_id, self.store.read_json(f"workers/{worker_id}/result.json", {}) or {}

    def _promote_and_resume(self, outcome, incident, *, selected_competition: str) -> tuple[str, str]:
        from kaggle_agent.supervisor.promote import GenerationPromotion, PromotionError

        generation = GenerationStore(self.store).load(outcome.generation_id or "")
        resume = outcome.resume_request
        if generation is None or resume is None or not outcome.acceptance.accepted:
            return ("NEEDS_AUTHORITY", "accepted repair did not produce a promotable generation and resume request")
        try:
            if read_git_revision(Path(generation.path)) != generation.revision.git_sha:
                return ("REJECTED", "candidate generation revision does not match its recorded commit")
            RepairPolicy().require_clean_auto_safe(Path(generation.path))
        except (RuntimeError, SafetyViolation) as exc:
            return ("REJECTED", f"candidate generation is not immutable: {exc}")
        try:
            RepairPolicy().require_clean_auto_safe(self.root)
        except SafetyViolation as exc:
            self.store.write_json("promotion.json", {"status": "BLOCKED", "reason": str(exc), "new_generation": generation.generation_id})
            return ("DIRTY_SOURCE_BASELINE", str(exc))
        outbox = ExternalActionOutbox(self.root, state_root=self.layout.state_root)
        if incident.external_job:
            external = outbox.get(incident.external_job)
            if external is None or external.status in {"prepared", "sent", "unknown"}:
                reason = "external action remains unresolved before promotion"
                self.store.write_json("promotion.json", {"status": "BLOCKED", "reason": reason, "new_generation": generation.generation_id})
                return ("PENDING_EXTERNAL", reason)
        promotion = GenerationPromotion(self.store)
        health = promotion.health_check(generation, selected_competition)
        resume_path = f"resume-requests/{incident.incident_id}.json"
        self.store.write_json(resume_path, resume.to_dict())
        replacement_worker_id = f"worker-{uuid.uuid4().hex[:12]}"
        if not health.healthy:
            self.store.write_json(
                "promotion.json",
                {
                    "status": "HEALTH_CHECK_FAILED",
                    "new_generation": generation.generation_id,
                    "resume_request_path": resume_path,
                    "failures": list(health.failures),
                },
            )
            return ("REJECTED", "generation startup health check failed")
        try:
            promotion.activate(
                generation,
                outcome.acceptance,
                health=health,
                resume_request_path=resume_path,
                replacement_worker_id=replacement_worker_id,
            )
        except PromotionError as exc:
            return ("REJECTED", str(exc))
        worker_id, result = self._start_worker(
            generation,
            selected_competition,
            "auto_safe",
            wait=True,
            worker_id=replacement_worker_id,
            resume_request=resume,
            incident_id=incident.incident_id,
        )
        if not result or not result.get("status"):
            return ("WORKER_STARTED", worker_id)
        return (str(result["status"]), str(result.get("exit_reason") or "resumed worker completed"))

    def _resume_promoted_if_needed(self, competition: str, mode: str, timeout_seconds: float) -> SupervisorRun | None:
        if mode != "auto_safe":
            return None
        transaction = self.store.read_json("promotion.json")
        if not isinstance(transaction, dict) or transaction.get("status") != "PROMOTED":
            return None
        worker_id = str(transaction.get("replacement_worker_id") or "")
        resume_path = str(transaction.get("resume_request_path") or "")
        generation_id = str(transaction.get("new_generation") or "")
        raw_resume = self.store.read_json(resume_path) if resume_path else None
        generation = GenerationStore(self.store).load(generation_id)
        if not worker_id or not isinstance(raw_resume, dict) or generation is None:
            return SupervisorRun("RECOVERY_BLOCKED", reason="promoted generation is missing durable resume state")
        metadata_path = self.store.path(f"workers/{worker_id}/metadata.json")
        result_path = self.store.path(f"workers/{worker_id}/result.json")
        if result_path.is_file():
            result = self.store.read_json(f"workers/{worker_id}/result.json", {}) or {}
            return self._settle_promoted_result(transaction, worker_id, result)
        if metadata_path.is_file():
            item = SupervisorRecovery(self.store).inspect_worker(worker_id, timeout_seconds=timeout_seconds, owner_token=self.lock.owner_token)
            if item.action == "ADOPT":
                return SupervisorRun("ADOPTED", worker_id=worker_id, reason="replacement worker heartbeat is fresh")
            if item.action == "TERMINATE_OR_RECONCILE":
                return SupervisorRun("RECOVERY_BLOCKED", worker_id=worker_id, reason="replacement worker ownership or heartbeat is unsafe")
        resume = ResumeRequest.from_dict(raw_resume)
        _, result = self._start_worker(
            generation, competition, "auto_safe", wait=True, worker_id=worker_id,
            resume_request=resume, incident_id=resume.incident_id,
        )
        return self._settle_promoted_result(transaction, worker_id, result or {})

    def _settle_promoted_result(
        self, transaction: dict[str, object], worker_id: str, result: dict[str, object]
    ) -> SupervisorRun:
        status = str(result.get("status") or "WORKER_STARTED")
        reason = str(result.get("exit_reason") or "")
        if status == "SUCCESS":
            self.store.write_json(
                "promotion.json",
                {**transaction, "status": "RESUMED", "resumed_worker_id": worker_id},
            )
            return SupervisorRun(status, worker_id=worker_id, reason=reason)
        if status in {"FATAL", "INTERRUPTED", "HUNG"}:
            old_id = str(transaction.get("old_generation") or "")
            old = GenerationStore(self.store).load(old_id) if old_id else None
            if old is not None:
                from kaggle_agent.supervisor.promote import GenerationPromotion

                GenerationPromotion(self.store).rollback(old)
                self.store.write_json(
                    "promotion.json",
                    {
                        **transaction,
                        "status": "ROLLED_BACK",
                        "rollback_reason": reason or status,
                    },
                )
                return SupervisorRun("ROLLED_BACK", worker_id=worker_id, reason=reason or status)
        return SupervisorRun(status, worker_id=worker_id, reason=reason)

    @staticmethod
    def _run_implementer(sessions: DeepSeekSupervisorAgents, worktree: Path, spec, feedback=None, state_root: Path | None = None):
        return sessions.implement(worktree, spec, feedback, state_root=state_root)
