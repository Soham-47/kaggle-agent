"""PLAN stage: read memory/cards/methods, write plan lines, done."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.memory.ingest import build_context_pack
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.source_cards import load_methods

_PLAN_APPROACHES = frozenset({"baseline", "tune", "recipe", "new"})

PLAN_SYSTEM = (
    "You plan the next Kaggle experiment. Call one tool per turn. "
    "Tools: read_memory, read_cards, read_methods, write_plan, done. "
    "write_plan args: hypothesis, approach (baseline|tune|recipe|new), steps. "
    "Prefer the copyable next step from method cards. Call done when the plan is written. "
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


def build_plan_tools(
    root: Path,
    workspace: Path | None = None,
    *,
    on_plan: Callable[[str, str, str], None] | None = None,
) -> tuple[dict[str, Callable[..., str]], dict[str, str]]:
    state: dict[str, str] = {
        "hypothesis": "",
        "approach": "",
        "steps": "",
        "wrote": "",
    }

    def read_memory(name: str = "", **_: Any) -> str:
        pack = build_context_pack(root)
        if name:
            key = name if name.endswith(".md") else f"{name}.md"
            text = pack.get(key) or pack.get(name)
            if text:
                return text[:4000]
            deep = memory_dir(root) / "research-deep"
            if deep.is_dir():
                matches = list(deep.glob(f"*{name}*"))
                if matches:
                    return matches[0].read_text(encoding="utf-8")[:4000]
            return "missing"
        return pack.as_prompt_block(max_chars_per_section=1200)

    def read_cards(**_: Any) -> str:
        deep = memory_dir(root) / "research-deep"
        if not deep.is_dir():
            return "no cards"
        cards = sorted(
            deep.glob("source-*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:2]
        if not cards:
            return "no cards"
        return "\n\n".join(p.read_text(encoding="utf-8")[:2000] for p in cards)

    def read_methods(**_: Any) -> str:
        if workspace is None:
            return "{}"
        data = load_methods(workspace)
        return str(data)

    def write_plan(
        hypothesis: str = "",
        approach: str = "baseline",
        steps: str = "",
        **_: Any,
    ) -> str:
        hyp = hypothesis or "dry-run default: schema-valid 0.5 baseline then improve"
        app = approach or "baseline"
        state["hypothesis"] = hyp
        state["approach"] = app
        state["steps"] = steps
        state["wrote"] = "1"
        if on_plan is not None:
            on_plan(hyp, app, steps)
        if not plan_is_ready(hyp, app):
            return "rejected: need hypothesis and approach baseline|tune|recipe|new"
        return write_plan_text(hyp, app, steps)

    tools = {
        "read_memory": read_memory,
        "read_cards": read_cards,
        "read_methods": read_methods,
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
    tracer: Any | None = None,
) -> tuple[StageAgent, dict[str, str]]:
    tools, state = build_plan_tools(root, workspace, on_plan=on_plan)
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
    )
    return agent, state
