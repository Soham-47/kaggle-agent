"""Resolve repo root and key directories."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the kaggle-agent repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def runtime_state_root(root: Path | None = None) -> Path | None:
    """Return the externally managed mutable-state root when configured."""
    value = os.environ.get("KAGGLE_AGENT_STATE_ROOT")
    return Path(value).resolve() if value else None


def agent_dir(root: Path | None = None) -> Path:
    configured = runtime_state_root(root)
    return (configured if configured is not None else (root or repo_root())) / ".agent"


def memory_dir(root: Path | None = None) -> Path:
    configured = runtime_state_root(root)
    return (configured if configured is not None else (root or repo_root())) / "memory"


def config_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "config"
