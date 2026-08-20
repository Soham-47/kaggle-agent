"""Durable state outside managed code generations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DIRECTORIES = (
    "workers", "incidents", "repairs", "reviews", "generations", "worktrees",
    "accepted", "rejected", "audit", "logs",
)


@dataclass(frozen=True)
class RuntimeLayout:
    code_root: Path
    state_root: Path

    @classmethod
    def for_repo(cls, root: Path, state_root: Path | None = None) -> "RuntimeLayout":
        configured = os.environ.get("KAGGLE_AGENT_SUPERVISOR_DIR")
        selected = Path(configured) if configured else (root.resolve().parent / ".kaggle-agent-supervisor")
        return cls(root.resolve(), (state_root or selected).resolve())

    @property
    def memory_root(self) -> Path:
        return self.state_root / "memory"

    @property
    def agent_root(self) -> Path:
        return self.state_root / ".agent"

    def initialize(self) -> None:
        self.state_root.mkdir(parents=True, exist_ok=True)
        for name in _DIRECTORIES:
            (self.state_root / name).mkdir(parents=True, exist_ok=True)


class SupervisorStateStore:
    """Atomic JSON state store owned by the supervisor process."""

    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout
        self.layout.initialize()

    def path(self, relative: str) -> Path:
        candidate = (self.layout.state_root / relative).resolve()
        try:
            candidate.relative_to(self.layout.state_root)
        except ValueError as exc:
            raise ValueError("supervisor state path escapes state root") from exc
        return candidate

    def read_json(self, relative: str, default: Any = None) -> Any:
        path = self.path(relative)
        if not path.is_file():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid supervisor state JSON: {path}") from exc

    def write_json(self, relative: str, value: Any) -> Path:
        destination = self.path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        return destination
