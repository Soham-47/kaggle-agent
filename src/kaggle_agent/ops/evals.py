"""Deterministic cycle gates (Waku eval split: 0/1 here, no judge mixed in)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.agents.code import extract_recipe_string
from kaggle_agent.memory.ingest import build_context_pack
from kaggle_agent.ops.log_parse import parse_daily_log
from kaggle_agent.ops.tracing import day_trace_path, read_jsonl
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.source_cards import (
    _valid_attach_ref,
    cards_feasible,
    load_methods,
    valid_model_pin,
)

DEFAULT_HYPOTHESIS = "dry-run default: schema-valid 0.5 baseline then improve"

MAX_INVALID_JSON_RATE = 0.30
RESEARCH_WRITE_TOOLS = frozenset({"write_card", "harvest_cards"})


def _latest_daily_log(root: Path) -> str:
    daily = memory_dir(root) / "daily"
    if not daily.is_dir():
        return ""
    files = sorted(daily.glob("????-??-??.md"))
    if not files:
        return ""
    return files[-1].read_text(encoding="utf-8")


def collect_events(root: Path, extra: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    events = list(extra or [])
    events.extend(read_jsonl(day_trace_path(root)))
    events.extend(parse_daily_log(_latest_daily_log(root)))
    return events


def _check(cid: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"id": cid, "ok": ok, "detail": detail}


def evaluate_cycle(
    root: Path,
    events: list[dict[str, Any]] | None = None,
    *,
    workspace: Path | None = None,
) -> dict[str, Any]:
    ev = list(events) if events is not None else collect_events(root)
    cycle_ids = [str(e.get("cycle_id")) for e in ev if e.get("cycle_id")]
    scoped_events = (
        [e for e in ev if e.get("cycle_id") == cycle_ids[-1]]
        if cycle_ids
        else ev
    )
    ev = scoped_events
    tools = [e for e in ev if e.get("type") == "tool"]
    research_tools = [e for e in tools if e.get("stage") == "research"]
    research_fleets = [e for e in research_tools if e.get("tool") == "fleet"]
    stage_verifications = [
        e
        for e in ev
        if e.get("type") == "agent_verification"
    ]
    executed_stages = {
        str(e.get("stage"))
        for e in ev
        if e.get("type") == "agent_execution"
        and e.get("stage") in {"plan", "code"}
    }
    verified_stages = {str(e.get("stage")) for e in stage_verifications}
    invalid = [e for e in tools if e.get("tool") == "invalid_json"]
    n_tools = len(tools)
    n_research = len(research_tools)
    rate = (len(invalid) / n_tools) if n_tools else 0.0
    wrote = any(
        e.get("tool") in RESEARCH_WRITE_TOOLS
        or (e.get("tool") == "fleet" and bool(e.get("verified")))
        for e in research_tools
    )

    ws = workspace
    if ws is None:
        comps = root / "competitions"
        kids = [p for p in comps.iterdir() if p.is_dir()] if comps.is_dir() else []
        ws = kids[0] if kids else root
    methods = load_methods(ws)
    junk_ds = [
        x
        for x in (methods.get("dataset_sources") or [])
        if x and not _valid_attach_ref(str(x))
    ]
    junk_md = [
        x
        for x in (methods.get("model_sources") or [])
        if x and not valid_model_pin(str(x))
    ]
    research_md = memory_dir(root) / "research.md"
    feasible = cards_feasible(ws, research_md) if research_md.is_file() else False
    pack = build_context_pack(root, view="plan", workspace=ws if ws.is_dir() else None)
    plan_ok = any(
        e.get("stage") == "plan" and e.get("tool") == "write_plan" for e in tools
    ) and not any(
        DEFAULT_HYPOTHESIS.lower() in str(e.get("result") or "").lower()
        for e in tools
        if e.get("tool") == "write_plan"
    )
    code_changed = any(
        e.get("stage") == "code" and e.get("tool") in {"write_methods", "write_custom_infer"}
        for e in tools
    )
    recipe_path = ws / "pipeline" / "kernel_recipe.py"
    hook_ok = not recipe_path.is_file()
    if recipe_path.is_file():
        try:
            import ast

            wrapper = recipe_path.read_text(encoding="utf-8")
            extracted = extract_recipe_string(wrapper)
            hook_ok = bool(
                extracted
                and "CUSTOM_INFER" in extracted
                and ast.parse(wrapper)
                and ast.parse(extracted)
            )
        except SyntaxError:
            hook_ok = False
    pack_keys = list(pack.sections)
    has_cards = any(
        k.startswith("research-deep/") or "Method cards" in pack.sections.get("research.md", "")
        for k in pack_keys
    ) or "## Method cards" in pack.sections.get("research.md", "")

    checks = [
        _check(
            "invalid_json_rate",
            n_research == 0 or rate <= MAX_INVALID_JSON_RATE,
            f"{len(invalid)}/{n_tools} = {rate:.0%} (max {MAX_INVALID_JSON_RATE:.0%})",
        ),
        _check(
            "research_wrote_card",
            wrote or n_research == 0,
            "write_card/harvest_cards called" if wrote else "no card-write tool in research loop",
        ),
        _check(
            "research_agents_verified",
            not research_fleets
            or all(bool(e.get("verified")) for e in research_fleets),
            "all research agents wrote verified cards"
            if research_fleets and all(bool(e.get("verified")) for e in research_fleets)
            else "research agent verification missing",
        ),
        _check(
            "stage_agent_verification",
            not executed_stages
            or executed_stages.issubset(verified_stages)
            and all(bool(e.get("verified")) for e in stage_verifications),
            "stage artifacts verified"
            if executed_stages.issubset(verified_stages)
            and all(bool(e.get("verified")) for e in stage_verifications)
            else "stage artifact verification failed",
        ),
        _check(
            "methods_pins_valid",
            not junk_ds and not junk_md,
            f"junk datasets={junk_ds} junk models={junk_md}"
            if (junk_ds or junk_md)
            else "pins ok",
        ),
        _check("cards_feasible", feasible, "methods.json + digest ready" if feasible else "cards thin"),
        _check(
            "context_has_method_cards",
            has_cards,
            "method cards in pack" if has_cards else "no method cards in context pack",
        ),
        _check(
            "plan_shippable",
            plan_ok or not any(e.get("stage") == "plan" for e in tools),
            "write_plan shippable" if plan_ok else "no shippable write_plan",
        ),
        _check(
            "code_changed_artifact",
            code_changed or not any(e.get("stage") == "code" for e in tools),
            "methods or hook written" if code_changed else "code wrote nothing",
        ),
        _check("code_hook_compiles", hook_ok, "hook compiles" if hook_ok else "hook missing or bad"),
    ]
    passed = all(c["ok"] for c in checks)
    return {
        "passed": passed,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checks": checks,
        "tool_counts": _tool_counts(tools),
    }


def _tool_counts(tools: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in tools:
        key = f"{e.get('stage') or '?'}:{e.get('tool') or '?'}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def persist_report(root: Path, report: dict[str, Any]) -> Path:
    daily = memory_dir(root) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    path = daily / "eval_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (daily / "eval_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    return path
