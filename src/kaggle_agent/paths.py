"""Resolve repo root and key directories."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the kaggle-agent repository root (parent of src/)."""
    return Path(__file__).resolve().parents[2]


def memory_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "memory"


def config_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "config"
