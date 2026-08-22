"""Typed supervisor configuration.

Validation remains owned by :mod:`kaggle_agent.config`; this module only
converts the already validated mapping into immutable runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepairSettings:
    enabled: bool = True
    classification_min_confidence: float = 0.80
    max_attempts_per_incident: int = 3
    max_repairs_per_cycle: int = 5
    max_repairs_per_day: int = 20
    max_changed_source_files: int = 8
    max_changed_test_files: int = 5
    max_changed_lines: int = 500
    allow_dependency_changes: bool = False
    require_spec_review: bool = True
    require_code_review: bool = True
    require_full_tests: bool = True


@dataclass(frozen=True)
class SupervisorSettings:
    enabled: bool = False
    mode: str = "observe"
    heartbeat_seconds: int = 30
    heartbeat_timeout_seconds: int = 180
    progress_timeout_seconds: int = 3600
    worker_terminate_grace_seconds: int = 20
    repair: RepairSettings = RepairSettings()
    promotion_automatic: bool = False
    protected_strict: bool = True

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SupervisorSettings":
        raw = raw or {}
        repair = raw.get("repair") or {}
        return cls(
            enabled=raw.get("enabled", False),
            mode=raw.get("mode", "observe"),
            heartbeat_seconds=raw.get("heartbeat_seconds", 30),
            heartbeat_timeout_seconds=raw.get("heartbeat_timeout_seconds", 180),
            progress_timeout_seconds=raw.get("progress_timeout_seconds", 3600),
            worker_terminate_grace_seconds=raw.get("worker_terminate_grace_seconds", 20),
            repair=RepairSettings(**{
                field: repair.get(field, getattr(RepairSettings(), field))
                for field in RepairSettings.__dataclass_fields__
            }),
            promotion_automatic=(raw.get("promotion") or {}).get("automatic", False),
            protected_strict=(raw.get("protected") or {}).get("strict", True),
        )
