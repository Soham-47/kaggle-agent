"""Task-adapter selection without competition-specific core logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from kaggle_agent.autonomy.contracts import CompetitionContract


@dataclass(frozen=True)
class SupportResult:
    supported: bool
    reason: str


@runtime_checkable
class TaskAdapter(Protocol):
    family: str

    def supports(self, contract: CompetitionContract) -> SupportResult: ...


class AdapterRegistry:
    def __init__(self, adapters: list[TaskAdapter] | None = None) -> None:
        self._adapters = list(adapters or [])

    def register(self, adapter: TaskAdapter) -> None:
        self._adapters.append(adapter)

    def select(self, contract: CompetitionContract) -> TaskAdapter:
        matches = [a for a in self._adapters if a.supports(contract).supported]
        if not matches:
            raise LookupError(f"no adapter supports task family {contract.task_family}")
        if len(matches) > 1:
            raise LookupError(f"ambiguous adapters for task family {contract.task_family}")
        return matches[0]


@dataclass(frozen=True)
class FamilyAdapter:
    """Core routing adapter; competition-local CODE owns model implementation."""

    family: str

    def supports(self, contract: CompetitionContract) -> SupportResult:
        return SupportResult(
            supported=contract.task_family == self.family,
            reason=f"task.family={contract.task_family}",
        )


INITIAL_FAMILIES = (
    "tabular_classification",
    "tabular_regression",
    "image_classification",
    "image_multilabel_classification",
    "image_segmentation",
    "image_detection",
    "text_classification",
    "text_regression",
    "text_generation",
    "time_series_forecasting",
    "ranking",
    "recommendation",
    "multimodal",
)


def default_registry() -> AdapterRegistry:
    return AdapterRegistry([FamilyAdapter(family) for family in INITIAL_FAMILIES])
