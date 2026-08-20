"""JSONL traces + usage ledger (Waku-style, no extra deps).

Writes under memory/daily/ so they stay out of the PLAN/CODE context pack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.paths import memory_dir


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def traces_dir(root: Path | None = None) -> Path:
    path = memory_dir(root) / "daily" / "traces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def usage_path(root: Path | None = None) -> Path:
    daily = memory_dir(root) / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    return daily / "usage.jsonl"


def day_trace_path(root: Path | None = None, when: datetime | None = None) -> Path:
    when = when or datetime.now(timezone.utc)
    return traces_dir(root) / f"{when.strftime('%Y-%m-%d')}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


class Tracer:
    """Append-only events for one repo. Safe to construct per cycle."""

    def __init__(self, root: Path, cycle_id: str = "") -> None:
        self.root = root
        self.cycle_id = cycle_id

    def emit(self, kind: str, **fields: Any) -> None:
        record = {"ts": _now(), "type": kind, **fields}
        if self.cycle_id and "cycle_id" not in record:
            record["cycle_id"] = self.cycle_id
        path = day_trace_path(self.root)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if kind == "llm":
            self._record_usage(record)

    def _record_usage(self, event: dict[str, Any]) -> None:
        row = {
            "ts": event.get("ts") or _now(),
            "cycle_id": event.get("cycle_id") or self.cycle_id,
            "stage": event.get("stage") or "",
            "model": event.get("model") or "",
            "in": int(event.get("tokens_in") or 0),
            "out": int(event.get("tokens_out") or 0),
        }
        with usage_path(self.root).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def read_day(self, when: datetime | None = None) -> list[dict[str, Any]]:
        return read_jsonl(day_trace_path(self.root, when))

    def read_usage(self) -> list[dict[str, Any]]:
        return read_jsonl(usage_path(self.root))
