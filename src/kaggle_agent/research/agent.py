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
    "Reply with ONLY JSON: {\"tool\": name, \"args\": {}}. "
    "Tools: list_kernels, pull_kernel, fetch_url, search, write_card, "
    "harvest_cards, deep_research, judge_cards, done. "
    "Call done when cards are implementable (datasets/models, hidden test IDs, "
    "ensemble rule). Do not invent slugs."
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
    ) -> None:
        super().__init__(
            zen,
            model,
            tools,
            config,
            system=_SYSTEM,
            log=log,
            accept_done=accept_done,
            no_zen_sequence=["harvest_cards", "deep_research"],
            name="research",
            reject_msg="done rejected: cards not ready",
        )
