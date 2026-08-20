"""Normalized outcomes shared by every autonomous runtime stage."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum


class OutcomeState(str, Enum):
    SUCCESS = "success"
    PENDING_EXTERNAL = "pending_external"
    RECOVERABLE_FAILURE = "recoverable_failure"
    NEEDS_AUTHORITY = "needs_authority"
    EXHAUSTED = "exhausted"
    FATAL = "fatal"


_FAILURE_STATES = {
    OutcomeState.RECOVERABLE_FAILURE,
    OutcomeState.EXHAUSTED,
    OutcomeState.FATAL,
}


def failure_signature(message: str) -> str:
    """Return a stable signature after removing volatile runtime details."""
    normalized = message.lower()
    normalized = re.sub(r"(?:[a-z]:)?[/\\][\w./\\-]+", "<path>", normalized)
    normalized = re.sub(r"\bline\s+\d+\b", "line <n>", normalized)
    normalized = re.sub(r"0x[0-9a-f]+", "<addr>", normalized)
    normalized = re.sub(r"\b\d{4,}\b", "<n>", normalized)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class StageOutcome:
    state: OutcomeState
    stage: str
    summary: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    failure_signature: str | None = None
    external_job: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.state in _FAILURE_STATES and not self.failure_signature:
            raise ValueError("failure_signature is required for failure outcomes")
        if self.state is OutcomeState.PENDING_EXTERNAL and not self.external_job:
            raise ValueError("external_job is required for pending_external")

    @classmethod
    def success(cls, stage: str, summary: str, **kwargs) -> "StageOutcome":
        return cls(OutcomeState.SUCCESS, stage, summary, **kwargs)

    @classmethod
    def failure(
        cls, stage: str, summary: str, signature: str | None = None, **kwargs
    ) -> "StageOutcome":
        return cls(
            OutcomeState.RECOVERABLE_FAILURE,
            stage,
            summary,
            failure_signature=signature or failure_signature(summary),
            retryable=True,
            **kwargs,
        )

