"""PLAN and CODE stage agents: tool loop with done / turn cap."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig, StallControl
from kaggle_agent.agents.plan import build_plan_tools, make_plan_agent, plan_is_ready, write_plan_text
from kaggle_agent.agents.code import (
    ValidationPipeline,
    calls_custom_infer,
    extract_recipe_string,
    make_code_agent,
    methods_payload_ok,
    min_length,
    no_out_hook,
    plan_to_methods_args,
    replace_kernel_recipe,
    splice_custom_infer,
    valid_python,
    writes_submission_csv,
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


def test_code_agent_stops_after_fallback_when_model_ignores_forced_write(tmp_path: Path):
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
        "T.Resize((224, 224))\n"
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
        plan_text="steps: rank average predictions",
    )
    out = agent.run("test")
    assert out.stop_reason == "stalled"
    assert state.get("wrote_recipe") == "1"
    recipe = (pipe / "kernel_recipe.py").read_text(encoding="utf-8")
    assert "# === CUSTOM_INFER START ===" in recipe
    assert "rank" in recipe


def test_code_agent_forces_model_recipe_write_after_stall(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub.to_csv('submission.csv')\n'''")
    (pipe / "methods.json").write_text(json.dumps({"implement_steps": ["old"]}))

    class _ChoiceZen:
        def __init__(self) -> None:
            self.choices = []

        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            self.choices.append(kwargs.get("tool_choice"))
            if isinstance(kwargs.get("tool_choice"), dict):
                return json.dumps(
                    {
                        "tool": "write_kernel_recipe",
                        "args": {
                            "source": (
                                "def CUSTOM_INFER(sub, ctx):\n"
                                "    return sub.rank(method='average', pct=True)\n"
                                "sub.to_csv('submission.csv')\n"
                            )
                        },
                    }
                )
            return json.dumps({"tool": "read_plan", "args": {}})

    zen = _ChoiceZen()
    agent, state = make_code_agent(
        zen,
        "m",
        root,
        workspace=ws,
        config=StageAgentConfig(max_minutes=5, max_tool_turns=8),
        plan_text="steps: rank average predictions",
    )

    agent.run("code")

    assert state["wrote_recipe"] == "1"
    assert any(isinstance(choice, dict) for choice in zen.choices)


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
        "T.Resize((224, 224))\n"
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


def test_code_agent_fallback_writes_distinct_recipe_variant(tmp_path: Path):
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
        "T.Resize((224, 224))\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n",
        encoding="utf-8",
    )
    (pipe / "methods.json").write_text(
        json.dumps({"implement_steps": ["old: copy ID discovery + rank-average"]}),
        encoding="utf-8",
    )

    from kaggle_agent.agents.loop import StageAgentConfig

    zen = _ScriptedZen([
        {"tool": "read_plan", "args": {}},
        {"tool": "read_plan", "args": {}},
        {"tool": "read_plan", "args": {}},
        {"tool": "read_plan", "args": {}},
    ])
    plan = "steps: raise DINOv2 input resolution to 336px; write submission"
    agent, _ = make_code_agent(
        zen, "m", root, workspace=ws,
        config=StageAgentConfig(max_minutes=5, max_tool_turns=20),
        plan_text=plan,
    )
    out = agent.run("test")
    assert out.stop_reason == "done"
    recipe = (pipe / "kernel_recipe.py").read_text(encoding="utf-8")
    assert "T.Resize((336, 336))" in recipe


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


def test_replace_kernel_recipe_rejects_hook_call_with_undefined_sub():
    wrapper = """KERNEL_RECIPE_SOURCE = r'''
def main():
    sub = make_submission()
    return sub
def CUSTOM_INFER(sub, ctx):
    return sub
sub = CUSTOM_INFER(sub, ctx)
sub.to_csv('submission.csv')
'''
    """

    try:
        replace_kernel_recipe(
            wrapper,
            "def main():\n    sub = make_submission()\n    return sub\n"
            "def CUSTOM_INFER(sub, ctx):\n    return sub\n"
            "sub = CUSTOM_INFER(sub, ctx)\n"
            "sub.to_csv('submission.csv')\n",
        )
        raise AssertionError("should reject undefined top-level sub")
    except ValueError as exc:
        assert "sub" in str(exc)


def test_code_write_recipe_rejects_missing_plan_tokens(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "def CUSTOM_INFER(sub, ctx):\n    return sub\n"
        "sub = None\n"
        "sub = CUSTOM_INFER(sub, ctx)\nsub.to_csv('submission.csv')\n'''")

    tools, _, _ = __import__(
        "kaggle_agent.agents.code", fromlist=["build_code_tools"]
    ).build_code_tools(
        root, ws, plan_text="steps: add fold rank aggregation"
    )
    out = tools["write_kernel_recipe"](
        source=(
            "def CUSTOM_INFER(sub, ctx):\n    return sub\n"
            "sub = None\n"
            "sub = CUSTOM_INFER(sub, ctx)\nsub = sub.copy()\n"
            "sub.to_csv('submission.csv')\n"
        )
    )

    assert out.startswith("rejected: plan steps")


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


def test_validation_pipeline_rules():
    assert valid_python("def f(): pass") is None
    assert valid_python("'''") == "recipe is not valid Python"
    assert writes_submission_csv("sub.to_csv('submission.csv')") is None
    assert writes_submission_csv("x = 1") == "recipe must write submission.csv"
    assert no_out_hook("out = CUSTOM_INFER(sub, ctx)") == "do not hook Path out"
    assert no_out_hook("sub = CUSTOM_INFER(sub, ctx)") is None
    assert calls_custom_infer("sub = CUSTOM_INFER(sub, ctx)") is None
    assert calls_custom_infer("sub = sub.copy()") == "recipe must call CUSTOM_INFER(sub, ctx)"


def test_submitted_fold_rank_recipe_preserves_study_rows():
    from pathlib import Path

    wrapper = Path("competitions/rsna_knee/pipeline/kernel_recipe.py").read_text(
        encoding="utf-8"
    )
    recipe = extract_recipe_string(wrapper) or ""
    assert "ranked.loc[group.index, df.columns]" in recipe
    assert "fold_ranks[name] = rank_transform(df)" in recipe


def test_validation_pipeline_min_length_rule():
    rule = min_length("x" * 100)
    assert rule("x" * 31) is None
    assert rule("x" * 30) is None
    assert rule("x" * 29) == "recipe too short (29 chars vs 100 before)"
    assert min_length("y" * 10)("y" * 3) is None


def test_validation_pipeline_composition():
    good = "sub = CUSTOM_INFER(sub, ctx)\nsub.to_csv('submission.csv')\n"
    pipe = ValidationPipeline([calls_custom_infer, writes_submission_csv])
    assert pipe(good) is None
    assert pipe.run("sub.to_csv('submission.csv')\n") == "recipe must call CUSTOM_INFER(sub, ctx)"
    assert pipe.run("sub = CUSTOM_INFER(sub, ctx)\n") == "recipe must write submission.csv"


def test_splice_custom_infer_shared_rules():
    wrapper = "KERNEL_RECIPE_SOURCE = r'''\n" "sub = CUSTOM_INFER(s,c)\n" "sub.to_csv('sub.csv')\n" "'''\n"
    try:
        splice_custom_infer(wrapper, "out = CUSTOM_INFER(sub, ctx)\nreturn sub")
        raise AssertionError("should reject out hook")
    except ValueError:
        pass
    try:
        splice_custom_infer(wrapper, "return sub\nsub.to_csv('x.csv')")
        raise AssertionError("should reject missing call")
    except ValueError:
        pass
    try:
        splice_custom_infer(wrapper, "'''")
        raise AssertionError("should reject invalid python")
    except ValueError:
        pass


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
    agent, state = make_code_agent(zen, "m", root, workspace=ws, config=StageAgentConfig(max_minutes=5, max_tool_turns=5), plan_text="steps: rank average predictions")
    out = agent.run("test")
    assert out.stop_reason == "done"
    assert state.get("wrote_recipe") == "1"
    assert "rank" in (pipe / "kernel_recipe.py").read_text(encoding="utf-8")


def test_write_kernel_recipe_rejects_identical_recipe(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: IDLE\n")
    ws = root / "competitions" / "rsna_knee"
    pipe = ws / "pipeline"
    pipe.mkdir(parents=True)
    wrapper = (
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n"
    )
    (pipe / "kernel_recipe.py").write_text(wrapper, encoding="utf-8")

    from kaggle_agent.agents.code import build_code_tools

    tools, _, _ = build_code_tools(root, ws, plan_text="steps: keep baseline")
    result = tools["write_kernel_recipe"](
        source=(
            "# === CUSTOM_INFER START ===\n"
            "def CUSTOM_INFER(sub, ctx):\n"
            "    return sub\n"
            "# === CUSTOM_INFER END ===\n"
            "sub = CUSTOM_INFER(sub, ctx)\n"
            "sub.to_csv('submission.csv')\n"
        )
    )

    assert result.startswith("rejected: recipe is identical")


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


# ---------------------------------------------------------------------------
# StallControl unit tests — one seam, 7 scenarios
# ---------------------------------------------------------------------------


def _sc(*, stall_after: int = 3, stall_nudge: str = "", stall_force=None):
    return StallControl(stall_after=stall_after, stall_nudge=stall_nudge, stall_force=stall_force)


def test_stallcontrol_no_stall_returns_continue():
    sc = StallControl(stall_after=None, stall_nudge="", stall_force=("done", {}))
    d = sc.evaluate(turns=10)
    assert d.action == "continue"


def test_stallcontrol_within_window_returns_continue():
    sc = _sc(stall_after=5)
    sc.mark_write(turn=3)
    d = sc.evaluate(turns=6)
    assert d.action == "continue"


def test_stallcontrol_stalled_forces_tool():
    sc = _sc(stall_after=3, stall_force=("write_plan", {"steps": "a"}))
    d = sc.evaluate(turns=5)
    assert d.action == "force_tool"
    assert d.tool_name == "write_plan"
    assert d.tool_args == {"steps": "a"}
    assert sc.force_count == 1
    assert sc.stall_forced is True


def test_stallcontrol_force_returns_none_stops():
    sc = _sc(stall_after=2, stall_force=lambda ep: None)
    d = sc.evaluate(turns=5)
    assert d.action == "stop_stalled"
    assert sc.stall_forced is True


def test_stallcontrol_force_done():
    sc = _sc(stall_after=2, stall_force=("done", {}))
    d = sc.evaluate(turns=5)
    assert d.action == "force_done"
    assert d.tool_name == "done"


def test_stallcontrol_force_exhausted_nudges():
    sc = _sc(stall_after=2, stall_force=("write_plan", {"steps": "x"}))
    sc.stall_forced = True
    sc.last_force_turn = 4
    sc.last_write_turn = 0
    d = sc.evaluate(turns=6)
    assert d.action == "nudge"


def test_stallcontrol_second_force_after_gap():
    sc = _sc(stall_after=3, stall_force=("write_plan", {"steps": "x"}))
    sc.last_force_turn = 2
    sc.last_write_turn = 0
    d = sc.evaluate(turns=8)
    assert d.action == "force_tool"
    assert sc.force_count == 1
    sc.stall_forced = False
    sc.last_write_turn = 0
    d = sc.evaluate(turns=14)
    assert d.action == "force_tool"
    assert sc.force_count == 2


def test_stallcontrol_mark_write_resets_window():
    sc = _sc(stall_after=3)
    sc.stall_forced = True
    sc.mark_write(turn=5)
    assert sc.last_write_turn == 5
    assert sc.stall_forced is False


def test_stallcontrol_callable_receives_episode():
    seen: list[int] = []

    def force(ep: int):
        seen.append(ep)
        return ("write_plan", {"steps": str(ep)})

    sc = _sc(stall_after=2, stall_force=force)
    sc.evaluate(turns=5)
    assert seen == [1]
    sc.stall_forced = False
    sc.last_write_turn = 0
    sc.evaluate(turns=10)
    assert seen == [1, 2]


def test_stallcontrol_callable_returns_none_stops():
    def force(ep: int):
        return ("write_plan", {"steps": "x"}) if ep == 1 else None

    sc = _sc(stall_after=2, stall_force=force)
    d1 = sc.evaluate(turns=5)
    assert d1.action == "force_tool"
    sc.stall_forced = False
    sc.last_write_turn = 0
    d2 = sc.evaluate(turns=10)
    assert d2.action == "stop_stalled"


def test_stallcontrol_no_force_nudges():
    sc = _sc(stall_after=2, stall_nudge="write now")
    d = sc.evaluate(turns=5)
    assert d.action == "nudge"
    assert d.nudge_text == "write now"


def test_stallcontrol_default_nudge_text():
    sc = _sc(stall_after=2)
    d = sc.evaluate(turns=5)
    assert d.action == "nudge"
    assert "stall" in d.nudge_text.lower()
