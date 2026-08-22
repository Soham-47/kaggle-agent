"""Supervisor lifecycle coordinator."""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.config import Settings
from kaggle_agent.supervisor.generation import GenerationStore
from kaggle_agent.supervisor.budgets import RepairBudgetStore
from kaggle_agent.supervisor.classifier import classify_failure
from kaggle_agent.supervisor.incidents import Incident, IncidentStore
from kaggle_agent.supervisor.lock import SupervisorLock
from kaggle_agent.supervisor.policy import RepairPolicy, SafetyViolation
from kaggle_agent.supervisor.protocol import WorkerRequest
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.worker import WorkerLauncher
from kaggle_agent.supervisor.heartbeat import HeartbeatStore
from kaggle_agent.autonomy.outcomes import OutcomeState, failure_signature
from kaggle_agent.autonomy.outcomes import StageOutcome


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

    def run_once(
        self,
        *,
        competition: str | None = None,
        wait: bool = True,
        dry_run: bool | None = None,
    ) -> SupervisorRun:
        config = self.settings.supervisor_config()
        if not config.enabled or config.mode == "off":
            return SupervisorRun("OFF", reason="supervisor disabled")
        if not self.lock.acquire():
            return SupervisorRun("LOCKED", reason="supervisor lock held")
        try:
            if config.mode == "auto_safe":
                try:
                    RepairPolicy().require_clean_auto_safe(self.root)
                except SafetyViolation as exc:
                    return SupervisorRun("DIRTY_SOURCE_BASELINE", reason=str(exc))
            generation = self._active_generation(managed=config.mode == "auto_safe")
            worker_id = f"worker-{uuid.uuid4().hex[:12]}"
            request = WorkerRequest(
                worker_id,
                generation.generation_id,
                competition or self.settings.default_competition,
                None,
                config.mode,
                None,
                None,
                generation.revision,
                self.settings.dry_run if dry_run is None else bool(dry_run),
                str(Path(generation.path)),
                None,
                getattr(generation, "repair_id", None),
                getattr(generation, "parent_generation", None),
            )
            launcher = WorkerLauncher(self.layout)
            started_at = time.time()
            try:
                process = launcher.start(request, cwd=Path(generation.path), heartbeat_seconds=config.heartbeat_seconds)
            except TypeError as exc:
                # Keep injected/legacy launchers that predate the optional
                # heartbeat setting source-compatible.
                if "heartbeat_seconds" not in str(exc):
                    raise
                process = launcher.start(request, cwd=Path(generation.path))
            self.store.write_json(f"workers/{worker_id}/metadata.json", {"pid": process.pid, "worker_id": worker_id, "generation_id": generation.generation_id, "supervisor_token": self.lock.owner_token, "started_at": started_at})
            if wait:
                hung = launcher.monitor_until_exit(
                    process,
                    worker_id,
                    timeout_seconds=config.heartbeat_timeout_seconds,
                    progress_timeout_seconds=config.progress_timeout_seconds,
                    grace_seconds=config.worker_terminate_grace_seconds,
                    poll_seconds=min(float(config.heartbeat_seconds), 1.0),
                    started_at=started_at,
                )
                if hung:
                    reason = "heartbeat stale or missing beyond timeout"
                    beat = HeartbeatStore(self.layout.state_root).read(worker_id)
                    outcome = StageOutcome(OutcomeState.FATAL, (beat.stage if beat else request.resume_from_stage) or "SUPERVISOR", reason, failure_signature=failure_signature(reason))
                    incident = Incident.from_outcome(worker_id=worker_id, generation_id=generation.generation_id, competition=request.competition, outcome=outcome, stage_attempt=1, revision=generation.revision, cycle_id=request.cycle_id, parent_occurrence_id=request.parent_occurrence_id, originating_repair_id=request.originating_repair_id, originating_generation_id=request.originating_generation_id)
                    IncidentStore(self.layout.state_root).save(incident)
                    result = {
                        "worker_id": worker_id,
                        "generation_id": generation.generation_id,
                        "status": "HUNG",
                        "cycle_id": request.cycle_id,
                        "experiment_id": None,
                        "current_stage": beat.stage if beat else request.resume_from_stage,
                        "incident_id": incident.incident_id,
                        "exit_reason": reason,
                        "revision": generation.revision.__dict__,
                    }
                    self.store.write_json(f"workers/{worker_id}/result.json", result)
                    return SupervisorRun("HUNG", worker_id=worker_id, reason=reason)
                result = self.store.read_json(f"workers/{worker_id}/result.json", {}) or {}
                if result.get("status"):
                    status = str(result["status"])
                    reason = str(result.get("exit_reason") or "")
                    if status == "RECOVERABLE_FAILURE":
                        reason = self._route_recoverable_failure(result, reason)
                    return SupervisorRun(status, worker_id=worker_id, reason=reason)
                reason = "worker exited without result"
                outcome = StageOutcome(OutcomeState.FATAL, request.resume_from_stage or "SUPERVISOR", reason, failure_signature=failure_signature(reason))
                incident = Incident.from_outcome(worker_id=worker_id, generation_id=generation.generation_id, competition=request.competition, outcome=outcome, stage_attempt=1, revision=generation.revision, cycle_id=request.cycle_id, parent_occurrence_id=request.parent_occurrence_id, originating_repair_id=request.originating_repair_id, originating_generation_id=request.originating_generation_id)
                IncidentStore(self.layout.state_root).save(incident)
                incident_id = incident.incident_id
                self.store.write_json(f"workers/{worker_id}/result.json", {"worker_id": worker_id, "generation_id": generation.generation_id, "status": "FATAL", "cycle_id": request.cycle_id, "experiment_id": None, "current_stage": request.resume_from_stage, "incident_id": incident_id, "exit_reason": reason, "revision": generation.revision.__dict__})
                return SupervisorRun("FATAL", worker_id=worker_id, reason=reason)
            return SupervisorRun("WORKER_STARTED", worker_id=worker_id)
        finally:
            self.lock.release()

    def _route_recoverable_failure(self, result: dict, reason: str) -> str:
        """Record one bounded post-worker incident decision.

        The supervisor does not restart a failed worker inline.  It classifies
        the durable incident and records whether a repair budget is available;
        a later repair coordinator can consume that decision exactly once.
        """
        incident_id = str(result.get("incident_id") or "")
        if not incident_id:
            return f"incident missing; {reason}".strip("; ")
        incident = IncidentStore(self.layout.state_root).load(incident_id)
        if incident is None:
            return f"incident {incident_id} missing; {reason}".strip("; ")
        classification = classify_failure(incident)
        repair = self.settings.supervisor_config().repair
        budget_available = RepairBudgetStore(
            self.layout.state_root,
            max_attempts_per_incident=repair.max_attempts_per_incident,
            max_repairs_per_cycle=repair.max_repairs_per_cycle,
            max_repairs_per_day=repair.max_repairs_per_day,
        ).available(incident.incident_id, incident.failure_signature, incident.cycle_id, incident.lineage_key)
        self.store.write_json(
            f"audit/incident-{incident_id}.json",
            {
                "incident_id": incident_id,
                "failure_signature": incident.failure_signature,
                "classification": classification.failure_class.value,
                "repairable": classification.repairable,
                "budget_available": budget_available,
                "bounded": True,
            },
        )
        return (
            f"incident={incident_id} class={classification.failure_class.value} "
            f"repair_budget={'available' if budget_available else 'exhausted'}; {reason}"
        ).strip("; ")

    def _active_generation(self, *, managed: bool = False):
        from kaggle_agent.supervisor.generation import RuntimeGeneration

        value = self.store.read_json("active-generation.json")
        if isinstance(value, dict):
            return RuntimeGeneration.from_dict(value)
        return (
            GenerationStore(self.store).create_managed(self.root)
            if managed
            else GenerationStore(self.store).create_snapshot(self.root)
        )
