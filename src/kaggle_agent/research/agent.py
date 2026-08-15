"""Research stage agent: LLM chooses tools until done, time, or turn cap."""

from __future__ import annotations

from typing import Callable

from kaggle_agent.agents.loop import (
    StageAgent,
    StageAgentResult,
    parse_tool_call,
)
from kaggle_agent.config import ResearchAgentSettings
from kaggle_agent.llm.zen_client import ZenClient

ResearchAgentConfig = ResearchAgentSettings
ResearchAgentResult = StageAgentResult
LogFn = Callable[[str], None]
ToolFn = Callable[..., str]

_SYSTEM = (
    "You research one Kaggle contest. Call one tool per turn. "
    "Tools: list_kernels, pull_kernel, fetch_url, search, write_card, "
    "harvest_cards, deep_research, judge_cards, done. "
    "Prefer harvest_cards or write_card before done. "
    "Call judge_cards before done. "
    "Call done when cards are implementable. Do not invent slugs. "
    'If you cannot call a tool, output {"tool": name, "args": {}}.'
)

__all__ = [
    "ResearchAgent",
    "ResearchAgentConfig",
    "ResearchAgentResult",
    "parse_tool_call",
]


class ResearchAgent(StageAgent):
    """while not done and under budget: LLM → tool → observe."""

    def __init__(
        self,
        zen: ZenClient | None,
        model: str,
        tools: dict[str, ToolFn],
        config: ResearchAgentConfig,
        *,
        log: LogFn | None = None,
        accept_done: Callable[[], bool] | None = None,
        tracer: object | None = None,
        must_first: list[str] | None = None,
        must_first_args: dict | None = None,
        reject_msg: str = "done rejected: cards not ready",
    ) -> None:
        super().__init__(
            zen,
            model,
            tools,
            config,
            system=_SYSTEM,
            log=log,
            accept_done=accept_done,
            must_first=must_first if must_first is not None else ["harvest_cards"],
            must_first_args=must_first_args,
            name="research",
            reject_msg=reject_msg,
            tracer=tracer,
        )
