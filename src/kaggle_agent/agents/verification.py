"""Shared execution records and stage-specific verification rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentExecution:
    """Evidence emitted by one agent run."""

    agent: str
    stop_reason: str = ""
    turns: int = 0
    tool_calls: list[str] = field(default_factory=list)
    source_reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    rejected_writes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Verification:
    ok: bool
    detail: str


def verify_research_fleet(
    executions: list[AgentExecution], required_agents: list[str]
) -> Verification:
    """Require one successful, agent-owned card write per roster member."""
    by_agent = {item.agent: item for item in executions}
    missing = [
        name
        for name in required_agents
        if not _has_owned_write(by_agent.get(name, AgentExecution(name)))
    ]
    if missing:
        detail = {
            name: {
                "source_reads": by_agent.get(name, AgentExecution(name)).source_reads,
                "writes": by_agent.get(name, AgentExecution(name)).writes,
            }
            for name in missing
        }
        return Verification(False, f"agents without verified card writes: {detail}")
    return Verification(True, f"verified card writes for {len(required_agents)} agents")


def _has_owned_write(execution: AgentExecution) -> bool:
    if not execution.source_reads:
        return False
    for raw_path in execution.writes:
        path = Path(raw_path)
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if (
            f"- agent: {execution.agent}" in lines
            and not any("stall-recovery" in line for line in lines)
            and any("- copyable next step:" in line for line in lines)
            and any("- do not copy:" in line for line in lines)
        ):
            return True
    return False


def verify_plan_stage(*, wrote: bool, judge_ready: bool) -> Verification:
    if not wrote:
        return Verification(False, "plan agent wrote no plan")
    if not judge_ready:
        return Verification(False, "plan judge did not accept the plan")
    return Verification(True, "plan artifact and judge accepted")


def verify_code_stage(
    *,
    wrote_recipe: bool,
    wrote_custom_infer: bool,
    artifact_ok: bool = True,
    smoke_ok: bool,
) -> Verification:
    if not (wrote_recipe or wrote_custom_infer):
        return Verification(False, "code agent wrote no implementation artifact")
    if not artifact_ok:
        return Verification(False, "code artifact failed recipe validation")
    if not smoke_ok:
        return Verification(False, "code artifact failed local smoke verification")
    return Verification(True, "code artifact and smoke verification passed")
