"""StageAgent must not burn 40 turns on invalid-JSON LLM replies.

DeepSeek-flash often returns prose instead of {"tool": ...} JSON. After a
small streak of invalid replies the loop must run a deterministic fallback
tool or bail (no_llm) so the orchestrator safety net takes over.
"""

from __future__ import annotations

import json

from kaggle_agent.agents.loop import StageAgent
from kaggle_agent.config import ResearchAgentSettings


class _ProseZen:
    """LLM that never returns a parseable tool call."""

    last_tool_calls: list = []

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        return "We need answer final JSON with reportMarkdown string. No prose."


class _MixedZen:
    """Scripted replies; falls back to garbage once the script is exhausted."""

    last_tool_calls: list = []

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if self._replies:
            return self._replies.pop(0)
        return "garbage fallback"


def _agent(zen, tools, *, fallback_tool=None, max_invalid=2, turns=10):  # noqa: ANN001
    return StageAgent(
        zen,
        "m",
        tools,
        ResearchAgentSettings(max_minutes=5, max_tool_turns=turns),
        system="test system",
        max_invalid=max_invalid,
        fallback_tool=fallback_tool,
    )


def test_invalid_json_twice_then_fallback_tool():
    zen = _ProseZen()
    seen: list[str] = []

    def list_kernels(**_a: object) -> str:
        seen.append("list_kernels")
        return "owner/kernel-a"

    agent = _agent(zen, {"list_kernels": list_kernels}, fallback_tool="list_kernels", turns=3)
    out = agent.run("contest")
    assert seen == ["list_kernels"]
    assert zen.calls == 4  # two invalid turns x (initial + zero-temp retry)
    assert out.stop_reason == "turn_cap"


def test_invalid_json_bails_no_llm_when_no_fallback():
    zen = _ProseZen()
    agent = _agent(zen, {}, turns=10)
    out = agent.run("contest")
    assert out.stop_reason == "no_llm"
    assert out.turns == 2
    assert zen.calls == 4


def test_valid_tool_call_resets_invalid_streak():
    zen = _MixedZen(
        [
            "garbage one",
            "garbage two",
            json.dumps({"tool": "search", "args": {"query": "knee"}}),
            "garbage three",
            "garbage four",
        ]
    )
    seen: list[str] = []

    def search(**_a: object) -> str:
        seen.append("search")
        return "hits"

    agent = _agent(zen, {"search": search}, fallback_tool="search", turns=5)
    out = agent.run("contest")
    assert seen == ["search", "search"]
    assert out.stop_reason == "turn_cap"


def test_invalid_streak_reset_on_native_tool_calls():
    zen = _ProseZen()
    calls: list[str] = []

    def search(**_a: object) -> str:
        calls.append("search")
        return "hits"

    class NativeZen(_ProseZen):
        def __init__(self) -> None:  # noqa: D107
            super().__init__()
            self._gave_native = False

        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            self.calls += 1
            if not self._gave_native:
                self._gave_native = True
                self.last_tool_calls = [("search", {})]
                return "tool call"
            self.last_tool_calls = []
            return "We need answer final JSON. No prose."

    agent = _agent(NativeZen(), {"search": search}, fallback_tool="search", turns=4)
    out = agent.run("contest")
    assert calls == ["search", "search"]
    assert out.stop_reason == "turn_cap"