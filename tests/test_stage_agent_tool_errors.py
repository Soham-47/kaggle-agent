from __future__ import annotations

import json

from kaggle_agent.agents.loop import StageAgent
from kaggle_agent.config import ResearchAgentSettings


class _ScriptedZen:
    last_tool_calls: list = []

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        return json.dumps(self.replies.pop(0) if self.replies else {"tool": "done", "args": {}})


def _agent(zen, tools, *, max_tool_turns: int = 5, fallback_tool: str | None = None):  # noqa: ANN001
    return StageAgent(
        zen,
        "model",
        tools,
        ResearchAgentSettings(max_minutes=5, max_tool_turns=max_tool_turns),
        system="test",
        fallback_tool=fallback_tool,
    )


def test_internal_type_error_executes_tool_once():
    calls = 0

    def write_file(path: str) -> str:
        nonlocal calls
        calls += 1
        raise TypeError("failure after side effect")

    out = _agent(
        _ScriptedZen([{"tool": "write_file", "args": {"path": "out.txt"}}]),
        {"write_file": write_file},
    ).run("context")

    assert calls == 1
    assert any("tool error: failure after side effect" in error for error in out.errors)


def test_invalid_arguments_are_rejected_before_tool_execution():
    calls = 0

    def write_file(path: str) -> str:
        nonlocal calls
        calls += 1
        return path

    out = _agent(
        _ScriptedZen(
            [
                {"tool": "write_file", "args": {"wrong": "out.txt"}},
                {"tool": "done", "args": {}},
            ]
        ),
        {"write_file": write_file},
    ).run("context")

    assert calls == 0
    assert any(error.startswith("invalid tool arguments:") for error in out.errors)


def test_no_arg_tool_executes_once():
    calls = 0

    def inspect_state() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    out = _agent(
        _ScriptedZen([{"tool": "inspect_state", "args": {}}]),
        {"inspect_state": inspect_state},
    ).run("context")

    assert calls == 1
    assert out.tool_calls == ["inspect_state"]


def test_optional_argument_and_kwargs_tools_keep_working():
    optional_seen: list[str] = []
    kwargs_seen: list[dict] = []

    def optional_tool(value: str = "default") -> str:
        optional_seen.append(value)
        return value

    def kwargs_tool(**kwargs: object) -> str:
        kwargs_seen.append(dict(kwargs))
        return "ok"

    out = _agent(
        _ScriptedZen(
            [
                {"tool": "optional_tool", "args": {}},
                {"tool": "kwargs_tool", "args": {"limit": 3}},
                {"tool": "done", "args": {}},
            ]
        ),
        {"optional_tool": optional_tool, "kwargs_tool": kwargs_tool},
    ).run("context")

    assert optional_seen == ["default"]
    assert kwargs_seen == [{"limit": 3}]
    assert out.stop_reason == "done"


def test_forced_tool_uses_same_single_invocation_guard():
    calls = 0

    def forced_tool(required: str) -> str:
        nonlocal calls
        calls += 1
        return required

    agent = StageAgent(
        None,
        "model",
        {"forced_tool": forced_tool},
        ResearchAgentSettings(max_minutes=5, max_tool_turns=3),
        system="test",
        must_first=["forced_tool"],
        must_first_args={"forced_tool": {"wrong": "value"}},
    )
    out = agent.run("context")

    assert calls == 0
    assert any(error.startswith("invalid tool arguments:") for error in out.errors)
