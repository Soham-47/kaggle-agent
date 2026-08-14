"""Assemble the dashboard payload from memory, traces, and evals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggle_agent.memory.ingest import CORE, build_context_pack
from kaggle_agent.ops.evals import collect_events, evaluate_cycle
from kaggle_agent.ops.terminal import relevant_lines
from kaggle_agent.ops.tracing import read_jsonl, usage_path
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.source_cards import load_methods
from kaggle_agent.state_md import parse_kv_markdown

PHASE_NODE = {
    "LOCK": "lock",
    "RESEARCH": "research",
    "PLAN": "plan",
    "CODE": "code",
    "LOCAL_SMOKE": "smoke",
    "KERNEL_TRAIN": "kernel",
    "VALIDATE_SUB": "kernel",
    "TELEGRAM_APPROVE": "submit",
    "SUBMIT": "submit",
    "FEEDBACK": "submit",
    "HEAL": "heal",
    "REPORT": "ops",
    "IDLE": "",
}

ARCHITECTURE = {
    "nodes": [
        {"id": "gateway", "label": "Gateway", "file": "notify/telegram.py · cli.py"},
        {"id": "lock", "label": "LOCK", "file": "orchestrator.py"},
        {"id": "research", "label": "RESEARCH loop", "file": "research/agent.py"},
        {"id": "plan", "label": "PLAN loop", "file": "agents/plan.py"},
        {"id": "code", "label": "CODE loop", "file": "agents/code.py"},
        {"id": "smoke", "label": "LOCAL_SMOKE", "file": "pipeline/smoke.py"},
        {"id": "kernel", "label": "KERNEL_TRAIN", "file": "train/kernel_runner.py"},
        {"id": "submit", "label": "SUBMIT", "file": "kaggle_api/submit_ops.py"},
        {"id": "heal", "label": "HEAL", "file": "heal/policy.py"},
        {"id": "memory", "label": "Memory pack", "file": "memory/ingest.py"},
        {"id": "ops", "label": "LLM Ops", "file": "ops/tracing.py"},
    ],
    "edges": [
        ["gateway", "lock"],
        ["lock", "research"],
        ["research", "plan"],
        ["plan", "code"],
        ["code", "smoke"],
        ["smoke", "kernel"],
        ["kernel", "submit"],
        ["submit", "heal"],
        ["memory", "research"],
        ["memory", "plan"],
        ["memory", "code"],
        ["research", "ops"],
        ["plan", "ops"],
        ["code", "ops"],
    ],
}

NOT_IN_PACK = ("heal.md", "daily/", "pending_submit.md", "older source cards")

STAGE_TOOLS = {
    "research": [
        "list_kernels",
        "pull_kernel",
        "fetch_url",
        "search",
        "write_card",
        "harvest_cards",
        "deep_research",
        "judge_cards",
        "done",
    ],
    "plan": ["read_memory", "read_cards", "read_methods", "write_plan", "done"],
    "code": ["read_cards", "read_plan", "write_brief", "write_methods", "done"],
}


def _is_running(state: dict[str, str]) -> bool:
    if str(state.get("lock_held") or "").lower() == "true":
        return True
    if str(state.get("last_result") or "") == "running":
        return True
    phase = state.get("phase") or "IDLE"
    return phase not in {"IDLE", "none", ""}


def _read_kv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return parse_kv_markdown(path.read_text(encoding="utf-8"))


def _workspace(root: Path, competition: str) -> Path:
    return root / "competitions" / competition


def _card_summaries(root: Path) -> list[dict[str, str]]:
    deep = memory_dir(root) / "research-deep"
    if not deep.is_dir():
        return []
    cards = sorted(deep.glob("source-*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[dict[str, str]] = []
    for path in cards[:8]:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.name
        out.append(
            {
                "name": path.name,
                "title": title[:120],
                "chars": str(len(text)),
                "in_pack": "yes" if path in cards[:2] else "no",
            }
        )
    return out


def _memory_view(root: Path) -> dict[str, Any]:
    pack = build_context_pack(root)
    sections = []
    for name, text in pack.sections.items():
        sections.append(
            {
                "name": name,
                "chars": len(text),
                "preview": text[:400],
            }
        )
    return {
        "core": list(CORE),
        "in_pack": list(pack.sections.keys()),
        "not_in_pack": list(NOT_IN_PACK),
        "missing": pack.missing,
        "sections": sections,
        "notes": [
            "PLAN/CODE only see the last 2 source cards by mtime.",
            "StageAgent keeps the last 8 transcript turns.",
            "heal.md and daily logs are never in the prompt pack.",
            "CODE writes a brief then always runs apply_recipe (metadata-ranker).",
        ],
    }


def _loop_view(events: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [e for e in events if e.get("type") == "tool"]
    return {
        "tools": tools[-200:],
        "counts": _counts(tools),
        "stops": [e for e in events if e.get("type") == "agent_stop"][-12:],
        "catalog": STAGE_TOOLS,
    }


def _counts(tools: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in tools:
        key = f"{e.get('stage') or '?'}:{e.get('tool') or '?'}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _submit_view(root: Path) -> dict[str, Any]:
    research = memory_dir(root) / "research.md"
    text = research.read_text(encoding="utf-8") if research.is_file() else ""
    pending = _read_kv(memory_dir(root) / "pending_submit.md")
    return {
        "research_excerpt": text[:2500],
        "pending": pending,
    }


def build_snapshot(root: Path) -> dict[str, Any]:
    state = _read_kv(memory_dir(root) / "state.md")
    heal = _read_kv(memory_dir(root) / "heal.md")
    competition = state.get("competition") or "rsna_knee"
    workspace = _workspace(root, competition)
    events = collect_events(root)
    evals = evaluate_cycle(root, events, workspace=workspace if workspace.is_dir() else None)
    usage = read_jsonl(usage_path(root))
    methods = load_methods(workspace) if workspace.is_dir() else {}
    phase = state.get("phase") or "IDLE"
    running = _is_running(state)
    return {
        "state": {**state, "heal": heal},
        "running": running,
        "active_node": PHASE_NODE.get(phase, "") if running else "",
        "terminal": relevant_lines(events),
        "architecture": ARCHITECTURE,
        "memory": _memory_view(root),
        "loop": _loop_view(events),
        "research": {
            "methods": methods,
            "cards": _card_summaries(root),
        },
        "submit": _submit_view(root),
        "evals": evals,
        "traces": [e for e in events if e.get("type") != "log"][-300:],
        "usage": {
            "calls": len(usage),
            "tokens_in": sum(int(u.get("in") or 0) for u in usage),
            "tokens_out": sum(int(u.get("out") or 0) for u in usage),
            "recent": usage[-40:],
        },
    }
