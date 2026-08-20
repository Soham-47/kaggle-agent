"""Stage interface: uniform phase dispatch for the orchestrator cycle.

Each phase handler runs as ``Stage.run(state, dry, result) -> AgentState``.
The registry maps phase names to Stages; the orchestrator dispatches through
it, so adding a phase means adding one Stage entry instead of editing the
dispatch body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.state_md import AgentState

PhaseFn = Callable[..., AgentState]


@dataclass(frozen=True)
class StageRun:
    """Typed stage result; ``outcome=None`` denotes a legacy handler."""

    state: AgentState
    outcome: StageOutcome | None = None


@dataclass(frozen=True)
class Stage:
    """A named cycle phase with a uniform run signature."""

    name: str
    fn: PhaseFn
    uses_dry: bool = False

    def run(self, state: AgentState, dry: bool, result: Any) -> AgentState:
        return self.run_typed(state, dry, result).state

    def run_typed(self, state: AgentState, dry: bool, result: Any) -> StageRun:
        if self.uses_dry:
            returned = self.fn(state, dry, result)
        else:
            returned = self.fn(state, result)
        if isinstance(returned, StageRun):
            if returned.outcome is not None and returned.outcome.stage != self.name:
                raise ValueError(
                    f"stage outcome mismatch: expected {self.name}, got {returned.outcome.stage}"
                )
            return returned
        return StageRun(returned or state)


_PHASE_ATTRS: dict[str, tuple[str, bool]] = {
    "RESEARCH": ("_research", False),
    "PLAN": ("_plan", True),
    "CODE": ("_code", False),
    "LOCAL_SMOKE": ("_local_smoke", False),
    "KERNEL_TRAIN": ("_kernel_train", True),
    "VALIDATE_SUB": ("_validate_sub", False),
    "TELEGRAM_APPROVE": ("_telegram_approve", True),
    "SUBMIT": ("_submit", True),
    "FEEDBACK": ("_feedback", True),
    "REPORT": ("_report", True),
    "HEAL": ("_heal", False),
}


def build_stage_registry(orch: Any) -> dict[str, Stage]:
    """Bind the orchestrator's phase handlers into a uniform Stage registry."""
    return {
        name: Stage(name=name, fn=getattr(orch, attr), uses_dry=uses_dry)
        for name, (attr, uses_dry) in _PHASE_ATTRS.items()
    }
