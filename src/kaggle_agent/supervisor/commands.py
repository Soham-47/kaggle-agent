"""Durable command queue owned and consumed by the supervisor."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.supervisor.state import RuntimeLayout


@dataclass(frozen=True)
class SupervisorCommand:
    command_id: str
    command: str
    payload: dict[str, object]
    created_at: str


class SupervisorCommandQueue:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout
        self.path = layout.state_root / "commands.jsonl"
        self.cursor_path = layout.state_root / "commands.cursor"
        self.control_path = layout.state_root / "control.json"

    def enqueue(self, command: str, payload: dict[str, object] | None = None) -> SupervisorCommand:
        if command not in {"run", "pause", "resume"}:
            raise ValueError(f"unsupported supervisor command: {command}")
        item = SupervisorCommand(
            uuid.uuid4().hex, command, dict(payload or {}),
            datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item.__dict__, sort_keys=True) + "\n")
            handle.flush()
        return item

    def pending(self) -> tuple[SupervisorCommand, ...]:
        rows = []
        if not self.path.is_file():
            return ()
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                rows.append(SupervisorCommand(str(value["command_id"]), str(value["command"]), dict(value.get("payload") or {}), str(value["created_at"])))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        try:
            cursor = int(self.cursor_path.read_text(encoding="utf-8")) if self.cursor_path.is_file() else 0
        except ValueError:
            cursor = 0
        return tuple(rows[cursor:])

    def drain(self) -> tuple[SupervisorCommand, ...]:
        items = self.pending()
        if self.path.is_file():
            total = len(self.path.read_text(encoding="utf-8").splitlines())
            self.cursor_path.write_text(str(total), encoding="utf-8")
        return items

    def paused(self) -> bool:
        if not self.control_path.is_file():
            return False
        try:
            return bool(json.loads(self.control_path.read_text(encoding="utf-8")).get("paused", False))
        except (json.JSONDecodeError, AttributeError):
            return True

    def set_paused(self, value: bool) -> None:
        temporary = self.control_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"paused": bool(value)}) + "\n", encoding="utf-8")
        temporary.replace(self.control_path)
