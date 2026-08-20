"""Git revisions and immutable runtime-generation records."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_agent.supervisor.state import SupervisorStateStore


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(("git", "-C", str(root), *args), text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git command failed: {' '.join(args)}")
    return result.stdout.strip()


def read_git_revision(root: Path) -> str:
    return _git(root, "rev-parse", "HEAD")


def read_tree_revision(root: Path) -> str:
    try:
        return _git(root, "rev-parse", "HEAD^{tree}")
    except RuntimeError:
        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".git" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeRevision:
    git_sha: str
    tree_sha: str
    generation_id: str = ""


@dataclass(frozen=True)
class RuntimeGeneration:
    generation_id: str
    revision: RuntimeRevision
    path: str
    parent_generation: str | None = None
    repair_id: str | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["revision"] = asdict(self.revision)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeGeneration":
        return cls(revision=RuntimeRevision(**value["revision"]), **{key: value[key] for key in (
            "generation_id", "path", "parent_generation", "repair_id", "created_at"
        )})


class GenerationStore:
    def __init__(self, state: SupervisorStateStore) -> None:
        self.state = state

    def save(self, generation: RuntimeGeneration) -> Path:
        return self.state.write_json(f"generations/{generation.generation_id}.json", generation.to_dict())

    def load(self, generation_id: str) -> RuntimeGeneration | None:
        value = self.state.read_json(f"generations/{generation_id}.json")
        return RuntimeGeneration.from_dict(value) if isinstance(value, dict) else None

    def new_id(self) -> str:
        rows = list((self.state.layout.state_root / "generations").glob("generation-*.json"))
        return f"generation-{len(rows) + 1:04d}"

    def create(self, path: Path, *, parent_generation: str | None = None, repair_id: str | None = None) -> RuntimeGeneration:
        generation_id = self.new_id()
        revision = RuntimeRevision(read_git_revision(path), read_tree_revision(path), generation_id)
        generation = RuntimeGeneration(
            generation_id=generation_id,
            revision=revision,
            path=str(path.resolve()),
            parent_generation=parent_generation,
            repair_id=repair_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        self.save(generation)
        return generation

    def create_managed(self, source_root: Path, *, parent_generation: str | None = None, repair_id: str | None = None) -> RuntimeGeneration:
        """Import a clean committed revision into an immutable detached worktree."""
        generation_id = self.new_id()
        destination = self.state.layout.state_root / "generations" / "code" / generation_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(("git", "-C", str(source_root), "worktree", "add", "--detach", str(destination), "HEAD"), text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "unable to import managed generation")
        revision = RuntimeRevision(read_git_revision(destination), read_tree_revision(destination), generation_id)
        generation = RuntimeGeneration(generation_id, revision, str(destination.resolve()), parent_generation, repair_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.save(generation)
        return generation

    def create_from_revision(self, source_root: Path, revision_sha: str, *, parent_generation: str | None = None, repair_id: str | None = None) -> RuntimeGeneration:
        generation_id = self.new_id()
        destination = self.state.layout.state_root / "generations" / "code" / generation_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(("git", "-C", str(source_root), "worktree", "add", "--detach", str(destination), revision_sha), text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "unable to materialize repaired generation")
        revision = RuntimeRevision(read_git_revision(destination), read_tree_revision(destination), generation_id)
        generation = RuntimeGeneration(generation_id, revision, str(destination.resolve()), parent_generation, repair_id, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.save(generation)
        return generation
