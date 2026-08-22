"""Deterministic acceptance and atomic active-generation promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kaggle_agent.supervisor.generation import RuntimeGeneration
from kaggle_agent.supervisor.state import SupervisorStateStore


@dataclass(frozen=True)
class RepairAcceptance:
    classification_allows_repair: bool = False
    spec_approved: bool = False
    base_revision_matches: bool = False
    protected_paths_pass: bool = False
    protected_semantics_pass: bool = False
    reproduction_pass: bool = False
    focused_tests_pass: bool = False
    subsystem_tests_pass: bool = False
    full_tests_pass: bool = False
    diff_limits_pass: bool = False
    static_safety_pass: bool = False
    test_integrity_pass: bool = False
    review_approved: bool = False
    external_state_safe: bool = False
    repair_budget_available: bool = False
    resume_plan_valid: bool = False

    @property
    def accepted(self) -> bool:
        return all(asdict(self).values())

    @classmethod
    def all_passed(cls, **overrides: bool) -> "RepairAcceptance":
        values = {field: True for field in cls.__dataclass_fields__}
        values.update(overrides)
        return cls(**values)


class PromotionError(RuntimeError):
    pass


class GenerationPromotion:
    def __init__(self, store: SupervisorStateStore) -> None:
        self.store = store

    def activate(self, generation: RuntimeGeneration, acceptance: RepairAcceptance) -> None:
        if not acceptance.accepted:
            raise PromotionError("repair acceptance is incomplete")
        self.store.write_json("active-generation.json", generation.to_dict())

    def rollback(self, generation: RuntimeGeneration) -> None:
        self.store.write_json("active-generation.json", generation.to_dict())

    def health_check(self, generation: RuntimeGeneration, competition: str):
        from kaggle_agent.supervisor.health import startup_health_check

        return startup_health_check(Path(generation.path), competition, state_root=self.store.layout.state_root)
