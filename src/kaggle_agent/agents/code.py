"""CODE stage: read cards/plan, write brief + valid methods sidecar, done."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.memory.ingest import build_context_pack
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.source_cards import (
    load_methods,
    step_is_junk,
    valid_model_pin,
)

CODE_SYSTEM = (
    "You are the coding agent for this Kaggle cycle. Call one tool per turn. "
    "Tools: read_cards, read_plan, write_brief, write_methods, done. "
    "write_methods args: dataset_sources, model_sources, implement_steps "
    "(lists). Model pins must be owner/slug/framework/instance/version. "
    "Never write dataset/model. Call write_methods then done. "
    'If you cannot call a tool, output {"tool": name, "args": {}}.'
)


def methods_payload_ok(
    dataset_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
    implement_steps: list[str] | None = None,
) -> tuple[bool, str]:
    steps = [str(s).strip() for s in (implement_steps or []) if str(s).strip()]
    if not steps or any(step_is_junk(s) for s in steps):
        return False, "need a non-junk implement step"
    for pin in model_sources or []:
        if pin and not valid_model_pin(str(pin)):
            return False, f"invalid model pin: {pin}"
    for ds in dataset_sources or []:
        if not ds or "/" not in str(ds) or str(ds).lower() in {"dataset/model"}:
            return False, f"invalid dataset: {ds}"
    return True, ""


def build_code_tools(
    root: Path,
    workspace: Path,
    *,
    plan_text: str = "",
) -> tuple[dict[str, Callable[..., str]], Path]:
    brief_path = workspace / "pipeline" / "code_brief.md"

    def read_cards(**_: Any) -> str:
        pack = build_context_pack(root)
        parts = [pack.get("research.md", "")[:2000]]
        deep = memory_dir(root) / "research-deep"
        if deep.is_dir():
            cards = sorted(
                deep.glob("source-*.md"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:2]
            for path in cards:
                parts.append(path.read_text(encoding="utf-8")[:2000])
        return "\n\n".join(p for p in parts if p) or "no cards"

    def read_plan(**_: Any) -> str:
        if plan_text.strip():
            return plan_text[:3000]
        exp = memory_dir(root) / "experiments"
        if exp.is_dir():
            files = sorted(exp.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                return files[0].read_text(encoding="utf-8")[:3000]
        return "no plan"

    def write_brief(text: str = "", **_: Any) -> str:
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(text or "attach listed datasets; discover test dirs; rank-mean", encoding="utf-8")
        return str(brief_path)

    def write_methods(
        dataset_sources: list[str] | str | None = None,
        model_sources: list[str] | str | None = None,
        implement_steps: list[str] | str | None = None,
        infer_hints: list[str] | str | None = None,
        **_: Any,
    ) -> str:
        def _as_list(v: list[str] | str | None) -> list[str]:
            if v is None:
                return []
            if isinstance(v, str):
                return [x.strip() for x in v.split(",") if x.strip()]
            return [str(x).strip() for x in v if str(x).strip()]

        ds, ms, steps, hints = (
            _as_list(dataset_sources),
            _as_list(model_sources),
            _as_list(implement_steps),
            _as_list(infer_hints),
        )
        ok, err = methods_payload_ok(ds, ms, steps)
        if not ok:
            return f"rejected: {err}"
        current = load_methods(workspace)
        payload = {
            "dataset_sources": ds or current.get("dataset_sources") or [],
            "model_sources": ms or current.get("model_sources") or [],
            "implement_steps": steps or current.get("implement_steps") or [],
            "infer_hints": hints or current.get("infer_hints") or [],
        }
        out = workspace / "pipeline" / "methods.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return str(out)

    tools = {
        "read_cards": read_cards,
        "read_plan": read_plan,
        "write_brief": write_brief,
        "write_methods": write_methods,
    }
    return tools, brief_path


def make_code_agent(
    zen: Any,
    model: str,
    root: Path,
    workspace: Path,
    config: StageAgentConfig,
    *,
    plan_text: str = "",
    log: Callable[[str], None] | None = None,
    tracer: Any | None = None,
) -> StageAgent:
    tools, brief_path = build_code_tools(root, workspace, plan_text=plan_text)
    return StageAgent(
        zen,
        model,
        tools,
        config,
        system=CODE_SYSTEM,
        log=log,
        accept_done=lambda: brief_path.is_file(),
        must_first=[],
        name="code",
        reject_msg="done rejected: write_brief first",
        tracer=tracer,
    )
