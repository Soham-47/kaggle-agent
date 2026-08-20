"""Typed supervisor configuration.

Validation remains owned by :mod:`kaggle_agent.config`; this module only
converts the already validated mapping into immutable runtime settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskProfileSettings:
    automatic_promotion: bool
    allow_candidate_generation: bool
    max_source_files: int
    max_test_files: int
    max_changed_lines: int
    max_attempts: int
    required_reproduction_strength: str
    require_focused_tests: bool
    require_adjacent_tests: bool
    require_full_tests: bool
    require_static_safety: bool
    require_spec_review: bool
    require_code_review: bool


def _default_profiles() -> dict[str, RiskProfileSettings]:
    common = dict(
        require_focused_tests=True,
        require_adjacent_tests=True,
        require_static_safety=True,
        require_spec_review=True,
        require_code_review=True,
    )
    return {
        "low": RiskProfileSettings(
            automatic_promotion=True, allow_candidate_generation=True,
            max_source_files=4, max_test_files=2, max_changed_lines=250,
            max_attempts=2, required_reproduction_strength="STATIC_REPRO",
            require_full_tests=False, **common,
        ),
        "medium": RiskProfileSettings(
            automatic_promotion=True, allow_candidate_generation=True,
            max_source_files=8, max_test_files=4, max_changed_lines=600,
            max_attempts=2, required_reproduction_strength="STATIC_REPRO",
            require_full_tests=True, **common,
        ),
        "high": RiskProfileSettings(
            automatic_promotion=False, allow_candidate_generation=True,
            max_source_files=12, max_test_files=6, max_changed_lines=1000,
            max_attempts=2, required_reproduction_strength="EXISTING_DETERMINISTIC_TEST",
            require_full_tests=True, **common,
        ),
        "prohibited": RiskProfileSettings(
            automatic_promotion=False, allow_candidate_generation=False,
            max_source_files=0, max_test_files=0, max_changed_lines=0,
            max_attempts=0, required_reproduction_strength="EXISTING_DETERMINISTIC_TEST",
            require_focused_tests=False, require_adjacent_tests=False,
            require_full_tests=False, require_static_safety=True,
            require_spec_review=True, require_code_review=True,
        ),
    }


@dataclass(frozen=True)
class AutoSafeSettings:
    enabled: bool = False
    policy: str = "risk_adaptive"
    global_max_source_files: int = 12
    global_max_test_files: int = 6
    global_max_changed_lines: int = 1000
    low: RiskProfileSettings = _default_profiles()["low"]
    medium: RiskProfileSettings = _default_profiles()["medium"]
    high: RiskProfileSettings = _default_profiles()["high"]
    prohibited: RiskProfileSettings = _default_profiles()["prohibited"]

    def __post_init__(self) -> None:
        if self.policy not in {"risk_adaptive", "conservative"}:
            raise ValueError("unsupported AUTO_SAFE risk policy")
        if self.high.automatic_promotion:
            raise ValueError("HIGH risk cannot allow automatic promotion")
        if self.prohibited.allow_candidate_generation:
            raise ValueError("PROHIBITED risk cannot allow candidate generation")
        if any(
            not profile.require_spec_review or not profile.require_code_review
            for profile in (self.low, self.medium, self.high)
        ):
            raise ValueError("automatic risk profiles must require independent review")
        for profile in (self.low, self.medium, self.high, self.prohibited):
            if (
                profile.max_source_files > self.global_max_source_files
                or profile.max_test_files > self.global_max_test_files
                or profile.max_changed_lines > self.global_max_changed_lines
            ):
                raise ValueError("risk profile exceeds global AUTO_SAFE hard ceiling")

    def profile(self, tier: str) -> RiskProfileSettings:
        return {
            "LOW": self.low,
            "MEDIUM": self.medium,
            "HIGH": self.high,
            "PROHIBITED": self.prohibited,
        }.get(str(tier).upper(), self.prohibited)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "AutoSafeSettings":
        raw = raw or {}
        defaults = _default_profiles()
        profiles = raw.get("profiles") or {}
        values: dict[str, RiskProfileSettings] = {}
        for name, default in defaults.items():
            values[name] = RiskProfileSettings(**{
                field: (profiles.get(name) or {}).get(field, getattr(default, field))
                for field in RiskProfileSettings.__dataclass_fields__
            })
        ceiling = raw.get("global_hard_ceiling") or {}
        return cls(
            enabled=raw.get("enabled", False),
            policy=raw.get("policy", "risk_adaptive"),
            global_max_source_files=ceiling.get("max_source_files", 12),
            global_max_test_files=ceiling.get("max_test_files", 6),
            global_max_changed_lines=ceiling.get("max_changed_lines", 1000),
            **values,
        )


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
    worker_terminate_grace_seconds: int = 20
    repair: RepairSettings = RepairSettings()
    promotion_automatic: bool = False
    protected_strict: bool = True
    auto_safe: AutoSafeSettings = AutoSafeSettings()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> "SupervisorSettings":
        raw = raw or {}
        repair = raw.get("repair") or {}
        return cls(
            enabled=raw.get("enabled", False),
            mode=raw.get("mode", "observe"),
            heartbeat_seconds=raw.get("heartbeat_seconds", 30),
            heartbeat_timeout_seconds=raw.get("heartbeat_timeout_seconds", 180),
            worker_terminate_grace_seconds=raw.get("worker_terminate_grace_seconds", 20),
            repair=RepairSettings(**{
                field: repair.get(field, getattr(RepairSettings(), field))
                for field in RepairSettings.__dataclass_fields__
            }),
            promotion_automatic=(raw.get("promotion") or {}).get("automatic", False),
            protected_strict=(raw.get("protected") or {}).get("strict", True),
            auto_safe=AutoSafeSettings.from_mapping(raw.get("auto_safe")),
        )
