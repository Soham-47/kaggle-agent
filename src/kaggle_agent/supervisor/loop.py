"""Supervisor lifecycle coordinator."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.config import Settings
from kaggle_agent.supervisor.generation import GenerationStore
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
            if config.mode == "auto_safe":
                try:
                    RepairPolicy().require_clean_auto_safe(self.root)
                except SafetyViolation as exc:
                    return SupervisorRun("DIRTY_SOURCE_BASELINE", reason=str(exc))
            generation = self._active_generation(managed=config.mode == "auto_safe")
            selected_competition = competition or self.settings.default_competition
            if not selected_competition:
                return SupervisorRun("NO_COMPETITION", reason="pass --competition or initialize a competition")
            worker_id = f"worker-{uuid.uuid4().hex[:12]}"
            request = WorkerRequest(worker_id, generation.generation_id, selected_competition, None, config.mode, None, None, generation.revision)
            process = WorkerLauncher(self.layout).start(request, cwd=Path(generation.path))
            self.store.write_json(f"workers/{worker_id}/metadata.json", {"pid": process.pid, "worker_id": worker_id, "generation_id": generation.generation_id, "supervisor_token": self.lock.owner_token})
            if wait:
                process.wait()
                result = self.store.read_json(f"workers/{worker_id}/result.json", {}) or {}
                if result.get("status"):
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
        return GenerationStore(self.store).create_managed(self.root) if managed else GenerationStore(self.store).create(self.root)

    def recover_startup(self, *, timeout_seconds: float) -> tuple[object, ...]:
        """Adopt safe workers and settle an interrupted promotion before launch."""
        recovery = SupervisorRecovery(self.store)
        workers = recovery.recover_workers(timeout_seconds=timeout_seconds, owner_token=self.lock.owner_token)
        transaction = self.store.read_json("promotion.json")
        if isinstance(transaction, dict) and transaction.get("status") == "PREPARED":
            generations = GenerationStore(self.store)
            new = generations.load(str(transaction.get("new_generation") or ""))
            old = generations.load(str(transaction.get("old_generation") or "")) if transaction.get("old_generation") else None
            if new is not None:
                from kaggle_agent.supervisor.promote import GenerationPromotion

                GenerationPromotion(self.store).recover_interrupted(old, new)
        return workers

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
        if classification.confidence < self.settings.supervisor_config().repair.classification_min_confidence:
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
        if mode == "observe":
            return ("SPEC_READY", repair_id)
        if mode != "repair_only":
            return ("NEEDS_AUTHORITY", "automatic activation is disabled during stabilization")
        if spec_review.verdict is not SpecReviewVerdict.APPROVE:
            return ("REJECTED", "independent spec review did not approve")
        coordinator = RepairCoordinator(self.root, self.store)
        try:
            outcome = coordinator.execute(
                incident, classification, spec, spec_approved=True,
                implementer=lambda worktree, approved: self._run_implementer(sessions, worktree, approved),
                reviewer=lambda inc, approved, diff, verification: sessions.review_code(inc, approved, diff, verification),
                cycle_id=incident.cycle_id,
                mode="repair_only",
            )
        except (AgentProtocolError, OSError, RuntimeError, ValueError) as exc:
            return ("REJECTED", str(exc))
        return (outcome.status, outcome.candidate_revision or "candidate stored")

    @staticmethod
    def _run_implementer(sessions: DeepSeekSupervisorAgents, worktree: Path, spec) -> None:
        result = sessions.implement(worktree, spec)
        if not result.stopped or result.reason.startswith("policy:"):
            raise RuntimeError(result.reason or "repair implementer stopped without completion")
