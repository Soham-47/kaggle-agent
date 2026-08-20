"""Turn daily markdown logs into the same event shapes as JSONL traces."""

from __future__ import annotations

import re
from typing import Any

_LINE = re.compile(
    r"^-\s*(\d{2}:\d{2}:\d{2})\s+UTC:\s*(.+)$"
)
_AGENT_TURN = re.compile(
    r"^(research|plan|code)\s+agent turn=(\d+)\s+tool=(\S+)"
)
_AGENT_STOP = re.compile(
    r"^(research|plan|code)\s+agent stop(?:=|:)\s*(\S+)"
)
_START = re.compile(r"^start\s+(\S+)\s+dry=(\S+)")
_PHASES = {
    "LOCK",
    "RESEARCH",
    "PLAN",
    "CODE",
    "LOCAL_SMOKE",
    "KERNEL_TRAIN",
    "VALIDATE_SUB",
    "TELEGRAM_APPROVE",
    "SUBMIT",
    "FEEDBACK",
    "HEAL",
    "REPORT",
}


def parse_daily_log(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in text.splitlines():
        m = _LINE.match(raw.strip())
        if not m:
            continue
        ts, body = m.group(1), m.group(2).strip()
        start = _START.match(body)
        if start:
            events.append(
                {
                    "type": "cycle_start",
                    "ts": ts,
                    "competition": start.group(1),
                    "dry": start.group(2),
                }
            )
            continue
        if body in _PHASES:
            events.append({"type": "phase", "ts": ts, "phase": body})
            continue
        turn = _AGENT_TURN.match(body)
        if turn:
            events.append(
                {
                    "type": "tool",
                    "ts": ts,
                    "stage": turn.group(1),
                    "turn": int(turn.group(2)),
                    "tool": turn.group(3),
                    "source": "daily",
                }
            )
            continue
        stop = _AGENT_STOP.match(body)
        if stop:
            events.append(
                {
                    "type": "agent_stop",
                    "ts": ts,
                    "stage": stop.group(1),
                    "reason": stop.group(2).rstrip(":"),
                    "detail": body,
                }
            )
            continue
        if body.startswith("code recipe"):
            events.append({"type": "recipe", "ts": ts, "detail": body})
            continue
        if body.startswith("end ") or body.startswith("error:"):
            events.append({"type": "cycle_end", "ts": ts, "detail": body})
            continue
        events.append({"type": "log", "ts": ts, "detail": body})
    return events
