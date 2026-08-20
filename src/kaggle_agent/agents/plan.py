"""PLAN stage: read memory/cards/methods, write a shippable plan, done."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.memory.ingest import build_context_pack, retrieve
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.source_cards import load_methods, step_is_junk, steps_implemented
from kaggle_agent.state_md import parse_kv_markdown

_PLAN_APPROACHES = frozenset({"baseline", "tune", "recipe", "new"})
DEFAULT_HYPOTHESIS = "dry-run default: schema-valid 0.5 baseline then improve"

PLAN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_memory": {
        "description": "Read the memory pack or one memory file.",
        "properties": {
            "name": {
                "type": "string",
                "description": "Memory file name without .md (e.g. MEMORY, COMPETITION)",
            }
        },
    },
    "read_cards": {
        "description": "Read the copyable next step from the latest method cards."
    },
    "read_methods": {
        "description": "Read the current pipeline/methods.json. Compare with your plan: "
        "if your steps are already implemented, propose something new."
    },
    "retrieve": {
        "description": "Semantic search over memory, method cards, and experiments.",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "scope": {
                "type": "string",
                "description": "Search scope: cards, experiments, or another memory key",
            },
        },
    },
    "write_plan": {
        "description": "Write the next experiment plan. The step must not already be "
        "implemented in methods.json.",
        "properties": {
            "hypothesis": {"type": "string", "description": "What you test and why"},
            "approach": {
                "type": "string",
                "enum": ["baseline", "tune", "recipe", "new"],
                "description": "baseline, tune, recipe, or new",
            },
            "steps": {
                "type": "string",
                "description": "Concrete copyable step(s), semicolon separated",
            },
        },
        "required": ["hypothesis", "approach", "steps"],
    },
    "done": {
        "description": "Finish planning when the plan is written and not yet implemented."
    },
}

PLAN_SYSTEM = (
    "You plan the next Kaggle experiment. Call one tool per turn. "
    "Tools: read_memory, read_cards, read_methods, retrieve, write_plan, done. "
    "write_plan args: hypothesis, approach (baseline|tune|recipe|new), steps. "
    "Prefer the copyable next step from method cards. Do not use the dry-run default. "
    "Call done when the plan is written. "
    "Rule: after at most two reads per tool, call write_plan. "
    "If methods.json already implements the copyable steps, write a NEW plan "
    "(approach new or tune) with a concrete step that is NOT yet implemented: "
    "enable GPU, a different backbone, grouped CV, or an inference change. "
    "Repeatedly reading without writing fails the stage. "
    'If you cannot call a tool, output {"tool": name, "args": {}}.'
)


def plan_is_ready(hypothesis: str, approach: str) -> bool:
    return bool((hypothesis or "").strip()) and (approach or "").strip() in _PLAN_APPROACHES


def write_plan_text(hypothesis: str, approach: str, steps: str) -> str:
    return (
        f"hypothesis: {hypothesis.strip()}\n"
        f"approach: {approach.strip()}\n"
        f"steps: {steps.strip()}"
    )


def _heal_next(root: Path) -> str:
    path = memory_dir(root) / "heal.md"
    if not path.is_file():
        return ""
    return parse_kv_markdown(path.read_text(encoding="utf-8")).get("decision_next", "")


def build_plan_tools(
    root: Path,
    workspace: Path | None = None,
    *,
    on_plan: Callable[[str, str, str], None] | None = None,
    judge: Callable[[str, str, str], tuple[bool, str]] | None = None,
) -> tuple[dict[str, Callable[..., str]], dict[str, str]]:
    state: dict[str, str] = {
        "hypothesis": "",
        "approach": "",
        "steps": "",
        "wrote": "",
    }
    heal_next = _heal_next(root)

    def read_memory(name: str = "", **_: Any) -> str:
        pack = build_context_pack(root, view="plan", workspace=workspace)
        if name:
            key = name if name.endswith(".md") else f"{name}.md"
            text = pack.get(key) or pack.get(name)
            if text:
                return text[:4000]
            return "missing"
        return pack.as_prompt_block()

    def read_cards(**_: Any) -> str:
        return retrieve(root, "copyable", scope="cards") or "no cards"

    def read_methods(**_: Any) -> str:
        if workspace is None:
            return "{}"
        return str(load_methods(workspace))

    def retrieve_tool(query: str = "", scope: str = "cards", **_: Any) -> str:
        return retrieve(root, query, scope=scope)

    def write_plan(
        hypothesis: str = "",
        approach: str = "",
        steps: str = "",
        **_: Any,
    ) -> str:
        hyp = (hypothesis or "").strip()
        app = (approach or "").strip()
        step = (steps or "").strip()
        if not hyp:
            return "rejected: empty hypothesis"
        if DEFAULT_HYPOTHESIS.lower() in hyp.lower() or DEFAULT_HYPOTHESIS.lower() in step.lower():
            return "rejected: dry-run default"
        if not step or step_is_junk(step):
            return "rejected: need a non-junk step"
        if app not in _PLAN_APPROACHES:
            return "rejected: need hypothesis and approach baseline|tune|recipe|new"
        if workspace is not None:
            try:
                if steps_implemented(step, load_methods(workspace)):
                    return (
                        "rejected: these steps are already implemented in methods.json. "
                        "Propose a different step (new model, GPU, CV, or inference change)."
                    )
            except Exception:  # noqa: BLE001
                pass
        if heal_next in {"recipe", "new"} and app == "baseline":
            return f"rejected: heal says {heal_next}, not baseline"
        if judge is not None:
            ready, reason = judge(hyp, app, step)
            if not ready:
                return (
                    f"rejected: judge says {reason}; "
                    "write a more novel, concrete plan"
                )
        state["hypothesis"] = hyp
        state["approach"] = app
        state["steps"] = step
        state["wrote"] = "1"
        if on_plan is not None:
            on_plan(hyp, app, step)
        if not plan_is_ready(hyp, app):
            state["wrote"] = ""
            return "rejected: need hypothesis and approach baseline|tune|recipe|new"
        return write_plan_text(hyp, app, step)

    tools = {
        "read_memory": read_memory,
        "read_cards": read_cards,
        "read_methods": read_methods,
        "retrieve": retrieve_tool,
        "write_plan": write_plan,
    }
    return tools, state


def make_plan_agent(
    zen: Any,
    model: str,
    root: Path,
    config: StageAgentConfig,
    *,
    workspace: Path | None = None,
    log: Callable[[str], None] | None = None,
    on_plan: Callable[[str, str, str], None] | None = None,
    judge: Callable[[str, str, str], tuple[bool, str]] | None = None,
    tracer: Any | None = None,
) -> tuple[StageAgent, dict[str, str]]:
    tools, state = build_plan_tools(root, workspace, on_plan=on_plan, judge=judge)
    agent = StageAgent(
        zen,
        model,
        tools,
        config,
        system=PLAN_SYSTEM,
        log=log,
        accept_done=lambda: bool(state.get("wrote"))
        and plan_is_ready(state["hypothesis"], state["approach"]),
        must_first=[],
        name="plan",
        reject_msg="done rejected: write_plan first",
        tracer=tracer,
        tool_schemas=PLAN_TOOL_SCHEMAS,
        stall_after=5,
        stall_nudge=(
            "STALL NUDGE: you have read enough. methods.json already implements the "
            "copyable steps, so write a NEW plan now: call write_plan with approach "
            "new or tune and a concrete step that is NOT yet implemented — e.g. "
            "enable_gpu true, a different backbone from the cards, grouped CV, or an "
            "inference change. A plan that just re-copies methods.json steps is rejected."
        ),
        stall_force=(
            "write_plan",
            {"hypothesis": "improve the rsna_knee public score", "approach": "tune", "steps": ""},
        ),
    )
    return agent, state
