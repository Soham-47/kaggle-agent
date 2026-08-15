"""PLAN and CODE stage agents: tool loop with done / turn cap."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.agents.plan import build_plan_tools, make_plan_agent, plan_is_ready, write_plan_text
from kaggle_agent.agents.code import (
    make_code_agent,
    methods_payload_ok,
    plan_to_methods_args,
    replace_kernel_recipe,
    splice_custom_infer,
)


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


def test_loop_stalls_again_after_write(tmp_path: Path):
    brief = tmp_path / "code_brief.md"
    zen = _ScriptedZen(
        [{"tool": "write_brief", "args": {"text": "attach pins; rank-mean"}}]
        + [{"tool": "read_file", "args": {"rel": "x"}}] * 30
    )
    calls: list[str] = []

    def write_brief(text: str = "", **_: object) -> str:
        calls.append("write_brief")
        brief.write_text(text, encoding="utf-8")
        return "ok"

    agent = StageAgent(
        zen,
        "m",
        {"write_brief": write_brief, "read_file": lambda **_: "contents"},
        StageAgentConfig(max_minutes=5, max_tool_turns=30),
        system="code",
        accept_done=lambda: brief.is_file(),
        stall_after=3,
        stall_force=("done", {}),
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert out.turns >= 4
    assert calls.count("write_brief") == 1


def test_loop_stops_when_stall_force_returns_none():
    zen = _ScriptedZen([{"tool": "read_file", "args": {"rel": "x"}}] * 30)
    agent = StageAgent(
        zen,
        "m",
        {"read_file": lambda **_: "contents"},
        StageAgentConfig(max_minutes=5, max_tool_turns=30),
        system="code",
        stall_after=3,
        stall_force=lambda episode: None,
    )
    out = agent.run("contest")
    assert out.stop_reason == "stalled"
    assert out.turns < 20


def test_code_agent_stops_on_second_stall_episode(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n",
        encoding="utf-8",
    )
    (pipe / "methods.json").write_text(json.dumps({"implement_steps": ["old step"]}), encoding="utf-8")

    new_recipe = (
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub.rank(method='average', pct=True)\n"
        "sub = sub.copy()\n"
        "sub.to_csv('submission.csv')\n"
    )
    from kaggle_agent.agents.loop import StageAgentConfig

    zen = _ScriptedZen(
        [{"tool": "write_kernel_recipe", "args": {"source": new_recipe}}]
        + [{"tool": "read_file", "args": {"rel": "pipeline/kernel_recipe.py"}}] * 30
    )
    agent, state = make_code_agent(
        zen,
        "m",
        root,
        workspace=ws,
        config=StageAgentConfig(max_minutes=5, max_tool_turns=30),
        plan_text="steps: unrelated plan text",
    )
    out = agent.run("test")
    assert out.stop_reason == "stalled"
    assert state.get("wrote_recipe") == "1"
    recipe = (pipe / "kernel_recipe.py").read_text(encoding="utf-8")
    assert "# === CUSTOM_INFER START ===" in recipe
    assert "rank" in recipe


def test_replace_kernel_recipe_works():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    out = replace_kernel_recipe(wrapper, "# === CUSTOM_INFER START ===\n" "def CUSTOM_INFER(sub, ctx):\n" "    return sub\n" "# === CUSTOM_INFER END ===\n" "sub = CUSTOM_INFER(sub, ctx)\n" "sub.to_csv('submission.csv')\n")
    assert "CUSTOM_INFER" in out
    assert "submission.csv" in out


def test_replace_kernel_recipe_rejects_invalid():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    try:
        replace_kernel_recipe(wrapper, "'''")
        raise AssertionError("should reject")
    except ValueError:
        pass
    try:
        replace_kernel_recipe(wrapper, "pass")
        raise AssertionError("should reject")
    except ValueError:
        pass


def test_replace_kernel_recipe_accepts_double_quote_docstring():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    out = replace_kernel_recipe(
        wrapper,
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        '"""docstring with "quotes"\n"""\n'
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n",
    )
    assert "docstring" in out


def test_any_plan_word_in_recipe():
    from kaggle_agent.agents.code import _any_plan_word_in_recipe

    assert _any_plan_word_in_recipe("rank-mean ensemble grouped_cv", "sub.rank('average')") is True
    assert _any_plan_word_in_recipe("convnext backbone dinov2", "print('hello')") is False


def test_done_ok_rejects_when_wrote_methods_but_no_recipe_word(tmp_path: Path):
    import json

    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n",
        encoding="utf-8",
    )
    (pipe / "methods.json").write_text(json.dumps({"implement_steps": ["train convnext model"]}), encoding="utf-8")

    from kaggle_agent.agents.loop import StageAgentConfig

    zen = _ScriptedZen([
        {"tool": "write_methods", "args": {"implement_steps": ["train convnext with dinov2 backbone"], "infer_hints": ["grouped_cv"]}},
        {"tool": "write_custom_infer", "args": {"source": "return sub"}},
        {"tool": "done", "args": {}},
    ])
    agent, _ = make_code_agent(zen, "m", root, workspace=ws, config=StageAgentConfig(max_minutes=5, max_tool_turns=5), plan_text="steps: train a convnext model with dinov2 backbone")
    out = agent.run("test")
    assert out.stop_reason != "done"


def test_replace_kernel_recipe_auto_inserts_glue():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    body = (
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = sub.copy()\n"
        "sub.to_csv('submission.csv')\n"
    )
    out = replace_kernel_recipe(wrapper, body)
    assert "sub = CUSTOM_INFER(sub, ctx)" in out


def test_replace_kernel_recipe_auto_inserts_markers():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    body = (
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "sub = sub.copy()\n"
        "sub.to_csv('submission.csv')\n"
    )
    out = replace_kernel_recipe(wrapper, body)
    assert "# === CUSTOM_INFER START ===" in out
    assert "# === CUSTOM_INFER END ===" in out
    assert "def CUSTOM_INFER(sub, ctx):" in out
    assert "sub = CUSTOM_INFER(sub, ctx)" in out
    ast.parse(out)


def test_write_kernel_recipe_unlocks_done(tmp_path: Path):
    import json

    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n",
        encoding="utf-8",
    )
    (pipe / "methods.json").write_text(json.dumps({"implement_steps": ["old step"]}), encoding="utf-8")

    new_recipe = (
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub.rank(method='average', pct=True)\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
    )

    from kaggle_agent.agents.loop import StageAgentConfig

    zen = _ScriptedZen([
        {"tool": "write_kernel_recipe", "args": {"source": new_recipe}},
        {"tool": "done", "args": {}},
    ])
    agent, state = make_code_agent(zen, "m", root, workspace=ws, config=StageAgentConfig(max_minutes=5, max_tool_turns=5), plan_text="steps: unrelated plan text")
    out = agent.run("test")
    assert out.stop_reason == "done"
    assert state.get("wrote_recipe") == "1"
    assert "rank" in (pipe / "kernel_recipe.py").read_text(encoding="utf-8")


def test_write_plan_judge_rejects_until_novel(tmp_path: Path):
    verdicts = [(False, "generic tuning"), (True, "novel step")]
    judged: list[str] = []

    def judge(hypothesis: str, approach: str, steps: str) -> tuple[bool, str]:
        judged.append(steps)
        return verdicts.pop(0)

    zen = _ScriptedZen(
        [
            {
                "tool": "write_plan",
                "args": {"hypothesis": "h1", "approach": "tune", "steps": "grouped 5-fold CV"},
            },
            {
                "tool": "write_plan",
                "args": {"hypothesis": "h2", "approach": "new", "steps": "second-stage stacker"},
            },
            {"tool": "done", "args": {}},
        ]
    )
    agent, state = make_plan_agent(
        zen,
        "m",
        tmp_path,
        StageAgentConfig(max_minutes=5, max_tool_turns=10),
        judge=judge,
    )
    out = agent.run("contest")
    assert out.stop_reason == "done"
    assert state["wrote"] == "1"
    assert state["hypothesis"] == "h2"
    assert state["approach"] == "new"
    assert judged == ["grouped 5-fold CV", "second-stage stacker"]


def test_write_plan_judge_rejection_mentions_reason():
    tools, state = build_plan_tools(
        Path("."), None, judge=lambda h, a, s: (False, "re-run of implemented step")
    )
    msg = tools["write_plan"]("h", "tune", "grouped 5-fold CV")
    assert "judge" in msg
    assert "re-run of implemented step" in msg
    assert state["wrote"] == ""
