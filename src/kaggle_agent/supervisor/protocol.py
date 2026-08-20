"""Explicit supervisor/worker wire protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.resume import ResumeRequest


class WorkerExit(str, Enum):
    SUCCESS = "SUCCESS"
    PENDING_EXTERNAL = "PENDING_EXTERNAL"
    RECOVERABLE_FAILURE = "RECOVERABLE_FAILURE"
    NEEDS_AUTHORITY = "NEEDS_AUTHORITY"
    EXHAUSTED = "EXHAUSTED"
    FATAL = "FATAL"
    INTERRUPTED = "INTERRUPTED"
    HUNG = "HUNG"


@dataclass(frozen=True)
class WorkerRequest:
    worker_id: str
    generation_id: str
    competition: str
    cycle_id: str | None
    mode: str
    resume_from_stage: str | None
    incident_id: str | None
    revision: RuntimeRevision | None = None
    resume_request: ResumeRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.revision is not None:
            value["revision"] = asdict(self.revision)
        if self.resume_request is not None:
            value["resume_request"] = self.resume_request.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkerRequest":
        raw = dict(value)
        revision = raw.get("revision")
        raw["revision"] = RuntimeRevision(**revision) if isinstance(revision, dict) else None
        resume = raw.get("resume_request")
        if isinstance(resume, dict):
            for field in ("preserved_stages", "invalidated_stages", "external_refs"):
                resume[field] = tuple(resume.get(field, ()))
            resume = ResumeRequest(
                **{**resume, "replay_epochs": tuple(tuple(item) for item in resume.get("replay_epochs", ()))},
            )
        raw["resume_request"] = resume
        return cls(**raw)


@dataclass(frozen=True)
class WorkerResult:
    worker_id: str
    generation_id: str
    status: str
    cycle_id: str | None
    experiment_id: str | None
    current_stage: str | None
    incident_id: str | None
    exit_reason: str
    revision: RuntimeRevision | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.revision is not None:
            value["revision"] = asdict(self.revision)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkerResult":
        raw = dict(value)
        revision = raw.get("revision")
        raw["revision"] = RuntimeRevision(**revision) if isinstance(revision, dict) else None
        return cls(**raw)
