"""Ensure competition workspace has the files CODE phase needs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WorkspaceCheck:
    ok: bool
    workspace: Path
    missing: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)


REQUIRED_PIPELINE = (
    "pipeline/schema.py",
    "pipeline/baseline.py",
    "pipeline/__init__.py",
    "pipeline/reports.py",
    "pipeline/ranker.py",
    "pipeline/recipe.py",
)
def ensure_pipeline_ready(
    workspace: Path, *, required: tuple[str, ...] = REQUIRED_PIPELINE
) -> WorkspaceCheck:
    missing: list[str] = []
    present: list[str] = []
    for rel in required:
        path = workspace / rel
        if path.is_file():
            present.append(rel)
        else:
            missing.append(rel)
    return WorkspaceCheck(ok=not missing, workspace=workspace, missing=missing, present=present)
