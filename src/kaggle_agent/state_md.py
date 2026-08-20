"""Single memory/state.md for runtime KV state + file lock."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.paths import memory_dir

_KV_RE = re.compile(r"^-\s*([a-zA-Z0-9_]+):\s*(.*)$")
_BOOL_FIELDS = frozenset({"paused", "dry_run", "lock_held"})


def parse_kv_markdown(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _KV_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def format_kv_markdown(title: str, data: dict[str, str]) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in data.items())
    lines.append("")
    return "\n".join(lines)


def _as_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    return str(raw).lower() in {"true", "yes", "1"}


@dataclass
class AgentState:
    phase: str = "IDLE"
    paused: bool = False
    dry_run: bool = True
    competition: str = "rsna_knee"
    pending_approve: str = "none"
    active_experiment: str = "none"
    lock_held: bool = False
    public_best: str = "none"
    budget_date: str = "none"
    proposals_used: str = "0"
    max_proposals: str = "2"
    last_cycle_start: str = "never"
    last_cycle_end: str = "never"
    last_result: str = "none"
    last_error: str = "none"
    note: str = "none"
    loop_last_score: str = "none"
    loop_prev_score: str = "none"
    loop_last_n: str = "none"
    loop_next_n: str = "3"
    loop_note: str = "none"

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> AgentState:
        values: dict[str, object] = {}
        for f in fields(cls):
            raw = d.get(f.name)
            if f.name in _BOOL_FIELDS:
                default = f.name == "dry_run"
                values[f.name] = _as_bool(raw, default)
            else:
                values[f.name] = f.default if raw is None else raw
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in _BOOL_FIELDS:
                out[f.name] = "true" if val else "false"
            elif val is None or val == "":
                out[f.name] = "none"
            else:
                out[f.name] = str(val)
        return out


def state_file(root: Path | None = None) -> Path:
    return memory_dir(root) / "state.md"


def load_state(root: Path | None = None) -> AgentState:
    path = state_file(root)
    if not path.is_file():
        return AgentState()
    return AgentState.from_dict(parse_kv_markdown(path.read_text(encoding="utf-8")))


def save_state(state: AgentState, root: Path | None = None) -> Path:
    path = state_file(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_kv_markdown("state", state.to_dict()), encoding="utf-8")
    return path


class RunLock:
    STALE_AGE_SECONDS = 12 * 3600

    def __init__(self, root: Path | None = None) -> None:
        self.path = memory_dir(root) / "run.lock"
        self._held = False
        self.took_over = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.is_stale():
            return False
        if self.path.exists():
            self.took_over = True
        self.path.write_text(
            f"pid={os.getpid()} at={datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )
        self._held = True
        return True

    def is_stale(self) -> bool:
        """True if the lock file cannot belong to a live run."""
        text = self.path.read_text(encoding="utf-8").strip()
        pid = self._read_pid(text)
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            return False
        age = time.time() - self.path.stat().st_mtime
        return age > self.STALE_AGE_SECONDS

    @staticmethod
    def _read_pid(text: str) -> int | None:
        for tok in text.split():
            if tok.startswith("pid="):
                try:
                    return int(tok[4:])
                except ValueError:
                    return None
        return None

    def release(self) -> None:
        if self._held and self.path.exists():
            self.path.unlink(missing_ok=True)
        self._held = False
        self.took_over = False
