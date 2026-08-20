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

    def activate(
        self,
        generation: RuntimeGeneration,
        acceptance: RepairAcceptance,
        *,
        health: object | None = None,
        resume_request_path: str | None = None,
        replacement_worker_id: str | None = None,
    ) -> None:
        if not acceptance.accepted:
            raise PromotionError("repair acceptance is incomplete")
        if health is not None and not bool(getattr(health, "healthy", False)):
            raise PromotionError("health check failed before promotion")
        current = self.store.read_json("active-generation.json")
        transaction = {
            "schema_version": 2,
            "status": "PREPARED",
            "old_generation": current.get("generation_id") if isinstance(current, dict) else None,
            "new_generation": generation.generation_id,
            "resume_request_path": resume_request_path,
            "replacement_worker_id": replacement_worker_id,
        }
        if health is not None:
            transaction["health"] = {
                "healthy": bool(getattr(health, "healthy", False)),
                "checks": list(getattr(health, "checks", ())),
                "failures": list(getattr(health, "failures", ())),
            }
        self.store.write_json(
            "promotion.json",
            transaction,
        )
        self.store.write_json("active-generation.json", generation.to_dict())
        self.store.write_json(
            "promotion.json",
            {**transaction, "status": "PROMOTED"},
        )

    def rollback(self, generation: RuntimeGeneration) -> None:
        self.store.write_json("active-generation.json", generation.to_dict())

    def recover_interrupted(self, old: RuntimeGeneration | None, new: RuntimeGeneration) -> str:
        """Resolve a prepared promotion to exactly old or new."""
        transaction = self.store.read_json("promotion.json")
        if not isinstance(transaction, dict) or transaction.get("status") in {"COMMITTED", "PROMOTED"}:
            return "NOOP"
        if transaction.get("new_generation") != new.generation_id:
            raise PromotionError("promotion transaction candidate does not match recovery candidate")
        pointer = self.store.read_json("active-generation.json")
        pointer_id = pointer.get("generation_id") if isinstance(pointer, dict) else None
        if pointer_id == new.generation_id:
            status = "PROMOTED" if transaction.get("schema_version") == 2 else "COMMITTED"
        elif old is not None and pointer_id in {None, old.generation_id}:
            self.rollback(old)
            status = "ROLLED_BACK"
        else:
            raise PromotionError("active generation pointer is neither old nor new")
        self.store.write_json(
            "promotion.json",
            {**transaction, "status": status},
        )
        return status

    def health_check(self, generation: RuntimeGeneration, competition: str):
        from kaggle_agent.supervisor.health import startup_health_check

        return startup_health_check(Path(generation.path), competition, state_root=self.store.layout.state_root)
