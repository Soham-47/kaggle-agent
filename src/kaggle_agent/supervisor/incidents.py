"""Durable, redacted incident records."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.generation import RuntimeRevision


_SECRET = re.compile(r"(?i)(authorization\s*:\s*(?:bearer\s+)?|api[_-]?key\s*[=:]\s*|token\s*[=:]\s*|password\s*[=:]\s*|secret\s*[=:]\s*)[^\s,;]+")
_PATH = re.compile(r"(?:[a-z]:)?[/\\][\w./\\-]+")


def sanitize_text(value: str | None, *, limit: int = 20_000) -> str:
    value = value or ""
    value = _SECRET.sub(lambda match: match.group(0).split("=")[0].split(":")[0] + "=<redacted>", value)
    value = re.sub(r"(?i)(DEEPSEEK_API_KEY|KAGGLE_USERNAME|KAGGLE_KEY|TELEGRAM_BOT_TOKEN)=\S+", r"\1=<redacted>", value)
    return value[:limit]


def failure_signature(stage: str, exception_type: str | None = None, message: str | None = None, traceback: str | None = None) -> str:
    if message is None:
        message, stage, exception_type = stage, "", None
    text = "\n".join((stage, exception_type or "", message, traceback or ""))
    text = _PATH.sub("<path>", text.lower())
    text = re.sub(r"\bline\s+\d+\b", "line <n>", text)
    text = re.sub(r"0x[0-9a-f]+", "<addr>", text)
    text = re.sub(r"\b\d{4,}\b", "<n>", text)
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Incident:
    incident_id: str
    worker_id: str
    generation_id: str
    cycle_id: str | None
    experiment_id: str | None
    competition: str
    stage: str
    stage_attempt: int
    revision: RuntimeRevision
    outcome_state: str
    exception_type: str | None
    exception_message: str
    traceback: str | None
    failure_signature: str
    evidence: tuple[str, ...]
    artifacts: tuple[str, ...]
    external_job: str | None
    kernel_ref: str | None
    candidate_csv: str | None
    recent_logs: tuple[str, ...]
    created_at: str
    lifecycle: str = "OPEN"
    # ``incident_id`` is the occurrence identity.  ``occurrence_id`` is an
    # explicit alias for newer callers; old records/callers only had
    # ``incident_id`` and continue to round-trip unchanged.
    occurrence_id: str | None = None
    parent_occurrence_id: str | None = None
    originating_repair_id: str | None = None
    originating_generation_id: str | None = None

    def __post_init__(self) -> None:
        if self.occurrence_id is None:
            object.__setattr__(self, "occurrence_id", self.incident_id)

    @classmethod
    def from_outcome(cls, *, worker_id: str, generation_id: str, competition: str,
                     outcome: StageOutcome, stage_attempt: int,
                     revision: RuntimeRevision, traceback: str | None = None,
                     cycle_id: str | None = None, experiment_id: str | None = None,
                     exception_type: str | None = None, kernel_ref: str | None = None,
                     candidate_csv: str | None = None, recent_logs: tuple[str, ...] = (),
                     parent_occurrence_id: str | None = None,
                     originating_repair_id: str | None = None,
                     originating_generation_id: str | None = None) -> "Incident":
        signature = outcome.failure_signature or failure_signature(outcome.stage, exception_type, outcome.summary, traceback)
        occurrence_id = f"incident-{uuid.uuid4().hex}"
        return cls(
            incident_id=occurrence_id,
            worker_id=worker_id, generation_id=generation_id, cycle_id=cycle_id,
            experiment_id=experiment_id, competition=competition, stage=outcome.stage,
            stage_attempt=stage_attempt, revision=revision, outcome_state=outcome.state.value,
            exception_type=exception_type, exception_message=sanitize_text(outcome.summary),
            traceback=sanitize_text(traceback), failure_signature=signature,
            evidence=tuple(sanitize_text(x) for x in outcome.evidence),
            artifacts=tuple(sanitize_text(x) for x in outcome.artifacts), external_job=outcome.external_job,
            kernel_ref=sanitize_text(kernel_ref), candidate_csv=sanitize_text(candidate_csv),
            recent_logs=tuple(sanitize_text(x) for x in recent_logs),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            parent_occurrence_id=parent_occurrence_id,
            originating_repair_id=originating_repair_id,
            originating_generation_id=originating_generation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["revision"] = asdict(self.revision)
        return value

    @property
    def lineage_key(self) -> str:
        return ":".join((self.originating_generation_id or self.generation_id, self.originating_repair_id or ""))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Incident":
        raw = dict(value)
        raw["revision"] = RuntimeRevision(**raw["revision"])
        for field in ("evidence", "artifacts", "recent_logs"):
            raw[field] = tuple(raw.get(field) or ())
        return cls(**raw)


class IncidentStore:
    def __init__(self, state_root: Path) -> None:
        self.root = state_root

    def save(self, incident: Incident) -> Path:
        destination = self.root / "incidents" / f"{incident.incident_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(incident.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        # Compatibility alias for callers that historically addressed an
        # incident by its stable signature.  Never overwrite an existing alias
        # because multiple occurrences may share one signature.
        alias = self.root / "incidents" / f"{incident.failure_signature}.json"
        if alias != destination and not alias.exists():
            alias.write_text(json.dumps(incident.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return alias
        return destination

    def load(self, incident_id: str) -> Incident | None:
        path = self.root / "incidents" / f"{incident_id}.json"
        if not path.is_file():
            return None
        return Incident.from_dict(json.loads(path.read_text(encoding="utf-8")))
