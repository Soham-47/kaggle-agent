"""Test-only fault injection seam; production is disabled by default."""

from __future__ import annotations

from enum import Enum


class FaultPoint(str, Enum):
    STAGE_ENTRY = "stage_entry"
    OUTBOX_PREPARED = "outbox_prepared"
    EXTERNAL_SENT = "external_sent"
    HANG = "hang"
    WORKER_KILL = "worker_kill"
    PARTIAL_JSONL = "partial_jsonl"
    REPAIR_REJECTED = "repair_rejected"
    REVIEW_REJECTED = "review_rejected"
    PROMOTION_INTERRUPTED = "promotion_interrupted"


class FaultInjected(RuntimeError):
    pass


class FaultInjector:
    def __init__(self, enabled: bool = False, points: set[FaultPoint] | None = None) -> None:
        self.enabled = enabled
        self.points = frozenset(points or ())

    def hit(self, point: FaultPoint) -> None:
        if self.enabled and point in self.points:
            raise FaultInjected(point.value)
