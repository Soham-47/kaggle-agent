"""Research agent: LLM tool loop with done / time / turn-cap stops."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.research.agent import ResearchAgent, ResearchAgentConfig


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if not self.replies:
            return json.dumps({"tool": "done", "args": {"reason": "empty script"}})
        return json.dumps(self.replies.pop(0))


def test_done_rejected_until_cards_ready():
    zen = _ScriptedZen(
        [
            {"tool": "done", "args": {"reason": "early"}},
            {"tool": "write_card", "args": {"ref": "a", "markdown": "x"}},
            {"tool": "done", "args": {"reason": "ok"}},
        ]
    )
    ready = {"ok": False}

    def write_card(**_a: object) -> str:
        ready["ok"] = True
        return "wrote"

    agent = ResearchAgent(
        zen,
        "m",
        {"write_card": write_card},
        ResearchAgentConfig(max_minutes=5, max_tool_turns=10),
        accept_done=lambda: ready["ok"],
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert ready["ok"] is True
    assert out.turns >= 2


def test_agent_stops_on_done_before_turn_cap(tmp_path: Path):
    log: list[str] = []
    seen: list[str] = []

    def list_kernels(**_a: object) -> str:
        seen.append("list_kernels")
        return "owner/kernel-a"

    zen = _ScriptedZen(
        [
            {"tool": "list_kernels", "args": {"limit": 3}},
            {"tool": "done", "args": {"reason": "enough"}},
        ]
    )
    agent = ResearchAgent(
        zen,
        "m",
        {"list_kernels": list_kernels},
        ResearchAgentConfig(max_minutes=5, max_tool_turns=40),
        log=log.append,
    )
    out = agent.run("contest x")
    assert out.stop_reason == "done"
    assert out.turns == 1
    assert seen == ["list_kernels"]
    assert zen.calls == 2


def test_agent_stops_on_turn_cap(tmp_path: Path):
    def search(**_a: object) -> str:
        return "hit"

    zen = _ScriptedZen(
        [{"tool": "search", "args": {"query": "knee"}}] * 20
    )
    agent = ResearchAgent(
        zen,
        "m",
        {"search": search},
        ResearchAgentConfig(max_minutes=5, max_tool_turns=3),
    )
    out = agent.run("contest")
    assert out.stop_reason == "turn_cap"
    assert out.turns == 3


def test_no_zen_harvests_once_then_done():
    seen: list[str] = []

    def harvest_cards(**_a: object) -> str:
        seen.append("harvest")
        return "wrote 1"

    agent = ResearchAgent(
        None,
        "m",
        {"harvest_cards": harvest_cards},
        ResearchAgentConfig(max_minutes=5, max_tool_turns=10),
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert seen == ["harvest"]
    assert out.turns == 1


def test_no_zen_harvests_then_deep():
    seen: list[str] = []

    def harvest_cards(**_a: object) -> str:
        seen.append("harvest")
        return "ok"

    def deep_research(**_a: object) -> str:
        seen.append("deep")
        return "ok"

    agent = ResearchAgent(
        None,
        "m",
        {"harvest_cards": harvest_cards, "deep_research": deep_research},
        ResearchAgentConfig(max_minutes=5, max_tool_turns=10),
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert seen == ["harvest", "deep"]


def test_agent_stops_on_time(tmp_path: Path, monkeypatch):
    ticks = {"n": 0}

    def monotonic() -> float:
        ticks["n"] += 1
        return 0.0 if ticks["n"] < 4 else 1000.0

    monkeypatch.setattr("kaggle_agent.research.agent.time.monotonic", monotonic)

    zen = _ScriptedZen([{"tool": "search", "args": {"query": "x"}}] * 20)
    agent = ResearchAgent(
        zen,
        "m",
        {"search": lambda **_a: "ok"},
        ResearchAgentConfig(max_minutes=0.001, max_tool_turns=40),
    )
    out = agent.run("contest")
    assert out.stop_reason == "time"
    assert out.turns < 40
