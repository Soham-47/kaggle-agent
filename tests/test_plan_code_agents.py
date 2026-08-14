"""PLAN and CODE stage agents: tool loop with done / turn cap."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.agents.plan import make_plan_agent, plan_is_ready, write_plan_text
from kaggle_agent.agents.code import methods_payload_ok


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        if not self.replies:
            return json.dumps({"tool": "done", "args": {}})
        return json.dumps(self.replies.pop(0))


def test_plan_write_then_done():
    stored: dict[str, str] = {}

    def write_plan(hypothesis: str = "", approach: str = "tune", steps: str = "", **_: object) -> str:
        stored["h"] = hypothesis
        stored["a"] = approach
        return "ok"

    zen = _ScriptedZen(
        [
            {
                "tool": "write_plan",
                "args": {
                    "hypothesis": "rank-mean DINOv2 slots",
                    "approach": "recipe",
                    "steps": "attach weights; rank-mean",
                },
            },
            {"tool": "done", "args": {}},
        ]
    )
    agent = StageAgent(
        zen,
        "m",
        {"write_plan": write_plan},
        StageAgentConfig(max_minutes=5, max_tool_turns=10),
        system="plan",
        accept_done=lambda: bool(stored.get("h")),
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert stored["h"] == "rank-mean DINOv2 slots"
    assert stored["a"] == "recipe"


def test_make_plan_agent_rejects_done_until_write_plan(tmp_path: Path):
    zen = _ScriptedZen(
        [
            {"tool": "done", "args": {}},
            {
                "tool": "write_plan",
                "args": {"hypothesis": "use cards", "approach": "recipe", "steps": "x"},
            },
            {"tool": "done", "args": {}},
        ]
    )
    agent, state = make_plan_agent(
        zen, "m", tmp_path, StageAgentConfig(max_minutes=5, max_tool_turns=10)
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert state["wrote"] == "1"
    assert state["hypothesis"] == "use cards"


def test_plan_ready_needs_hypothesis_and_approach():
    assert plan_is_ready("use grouped folds", "tune") is True
    assert plan_is_ready("", "tune") is False
    assert plan_is_ready("x", "maybe") is False


def test_write_plan_text_has_three_lines():
    text = write_plan_text("use grouped folds", "tune", "copy winner step")
    assert "hypothesis:" in text
    assert "approach: tune" in text
    assert "steps:" in text


def test_methods_payload_rejects_junk_pin():
    ok, err = methods_payload_ok(
        dataset_sources=["owner/public-weights"],
        model_sources=["dataset/model"],
        implement_steps=["attach public weights"],
    )
    assert ok is False
    assert "pin" in err.lower() or "model" in err.lower()


def test_methods_payload_accepts_valid():
    ok, err = methods_payload_ok(
        dataset_sources=["pilkwang/rsna-knee-weights"],
        model_sources=["metaresearch/dinov2/PyTorch/small/1"],
        implement_steps=["attach pilkwang weights; rank-mean"],
    )
    assert ok is True
    assert err == ""


def test_code_done_rejected_until_brief(tmp_path: Path):
    brief = tmp_path / "code_brief.md"
    zen = _ScriptedZen(
        [
            {"tool": "done", "args": {}},
            {"tool": "write_brief", "args": {"text": "attach pins; folders; rank-mean"}},
            {"tool": "done", "args": {}},
        ]
    )

    def write_brief(text: str = "", **_: object) -> str:
        brief.write_text(text, encoding="utf-8")
        return "ok"

    agent = StageAgent(
        zen,
        "m",
        {"write_brief": write_brief},
        StageAgentConfig(max_minutes=5, max_tool_turns=10),
        system="code",
        accept_done=lambda: brief.is_file(),
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert "rank-mean" in brief.read_text(encoding="utf-8")
