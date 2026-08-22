"""Checkpoint invalidation and durable resume requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


_STAGES = ("RESEARCH", "PLAN", "CODE", "LOCAL_SMOKE", "KERNEL_TRAIN", "VALIDATE_SUB", "TELEGRAM_APPROVE", "SUBMIT", "FEEDBACK", "HEAL", "REPORT")


def invalidated_stages(from_stage: str) -> tuple[str, ...]:
    try:
        index = _STAGES.index(from_stage)
    except ValueError:
        index = 0
    return _STAGES[index:]


@dataclass(frozen=True)
class ResumeRequest:
    cycle_id: str
    incident_id: str
    old_generation: str
    new_generation: str
    failed_stage: str
    resume_from_stage: str
    preserved_stages: tuple[str, ...]
    invalidated_stages: tuple[str, ...]
    external_refs: tuple[str, ...]
    parent_occurrence_id: str | None = None
    originating_repair_id: str | None = None
    originating_generation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
