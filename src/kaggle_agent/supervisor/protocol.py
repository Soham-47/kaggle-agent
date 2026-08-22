"""Explicit supervisor/worker wire protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from kaggle_agent.supervisor.generation import RuntimeRevision


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
    # The supervisor owns the exact run intent.  Do not let the worker infer
    # this from mutable repository settings after a Telegram command.
    dry_run: bool = True
    # Exact immutable source tree selected by the supervisor.  Optional for
    # request files written by older supervisors; the worker then uses cwd.
    generation_path: str | None = None
    parent_occurrence_id: str | None = None
    originating_repair_id: str | None = None
    originating_generation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.revision is not None:
            value["revision"] = asdict(self.revision)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkerRequest":
        raw = dict(value)
        revision = raw.get("revision")
        raw["revision"] = RuntimeRevision(**revision) if isinstance(revision, dict) else None
        # Ignore unknown fields from newer supervisors while preserving the
        # old wire format's defaults.
        allowed = {field for field in cls.__dataclass_fields__}
        raw = {key: item for key, item in raw.items() if key in allowed}
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
