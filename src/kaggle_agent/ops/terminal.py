"""Turn raw cycle events into a short terminal feed."""

from __future__ import annotations

from typing import Any

_NOISE_TOOLS = frozenset(
    {
        "invalid_json",
        "read_memory",
        "read_cards",
        "read_plan",
        "read_methods",
        "fetch_url",
        "search",
        "list_kernels",
    }
)


def relevant_lines(
    events: list[dict[str, Any]], *, limit: int = 40
) -> list[dict[str, str]]:
    lines: list[dict[str, str]] = []
    bad_n = 0
    bad_stage = "loop"
    bad_ts = ""

    def flush() -> None:
        nonlocal bad_n
        if bad_n <= 0:
            return
        lines.append(
            {
                "ts": str(bad_ts),
                "level": "warn",
                "text": f"{bad_stage} · {bad_n}× invalid_json",
            }
        )
        bad_n = 0

    for ev in events:
        kind = ev.get("type") or ""
        ts = str(ev.get("ts") or "")
        if kind == "tool" and ev.get("tool") in _NOISE_TOOLS:
            bad_n += 1
            bad_stage = str(ev.get("stage") or "loop")
            bad_ts = ts
            continue
        flush()
        if kind == "phase":
            lines.append({"ts": ts, "level": "info", "text": str(ev.get("phase") or "")})
        elif kind == "cycle_start":
            lines.append(
                {
                    "ts": ts,
                    "level": "info",
                    "text": f"start {ev.get('competition') or ''} dry={ev.get('dry')}",
                }
            )
        elif kind == "cycle_end":
            detail = str(ev.get("detail") or ev.get("status") or "end")
            level = "error" if "error" in detail.lower() else "ok"
            lines.append({"ts": ts, "level": level, "text": detail[:240]})
        elif kind == "agent_stop":
            stage = ev.get("stage") or "agent"
            reason = ev.get("reason") or "stop"
            lines.append(
                {"ts": ts, "level": "warn", "text": f"{stage} stop · {reason}"}
            )
        elif kind == "recipe":
            detail = str(ev.get("detail") or "recipe")
            if "metadata-ranker" in detail:
                detail = "recipe · metadata-ranker"
            lines.append({"ts": ts, "level": "info", "text": detail[:120]})
        elif kind == "tool":
            lines.append(
                {
                    "ts": ts,
                    "level": "ok",
                    "text": f"{ev.get('stage') or 'loop'} · {ev.get('tool')}",
                }
            )
        elif kind == "log":
            detail = str(ev.get("detail") or "")
            low = detail.lower()
            if any(tok in low for tok in ("error", "fail", "cards still", "feasible")):
                level = "error" if "error" in low or "fail" in low else "warn"
                lines.append({"ts": ts, "level": level, "text": detail[:240]})
    flush()
    compact: list[dict[str, str]] = []
    for line in lines:
        if compact and compact[-1]["text"] == line["text"]:
            continue
        compact.append(line)
    return compact[-limit:]
