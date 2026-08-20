"""Supervisor-owned isolated Git worktrees."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


class WorktreeError(RuntimeError):
    pass


class WorktreeManager:
    def __init__(self, source_root: Path, state_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.state_root = state_root.resolve()
        self.worktrees_root = self.state_root / "worktrees"

    def _git(self, worktree: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = ("git", "-C", str(worktree), *args)
        if args and args[0] == "commit":
            command = ("git", "-C", str(worktree), "-c", "user.name=kaggle-agent supervisor", "-c", "user.email=supervisor@localhost", *args)
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if check and result.returncode:
            raise WorktreeError(result.stderr.strip() or "git worktree operation failed")
        return result

    def create(self, incident_id: str, attempt: int, base_revision: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", incident_id)
        destination = self.worktrees_root / safe / f"a{attempt}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        branch = f"repair/{safe}-a{attempt}"
        result = subprocess.run(("git", "-C", str(self.source_root), "worktree", "add", "-b", branch, str(destination), base_revision), text=True, capture_output=True, check=False)
        if result.returncode:
            raise WorktreeError(result.stderr.strip() or "unable to create repair worktree")
        return destination

    def status(self, worktree: Path) -> str:
        return self._git(worktree, "status", "--porcelain").stdout

    def diff(self, worktree: Path) -> str:
        return self._git(worktree, "diff", "--no-ext-diff").stdout

    def commit(self, worktree: Path, message: str) -> str:
        self._git(worktree, "add", "--all")
        self._git(worktree, "commit", "-m", message)
        return self._git(worktree, "rev-parse", "HEAD").stdout.strip()

    def destroy(self, worktree: Path) -> None:
        result = subprocess.run(("git", "-C", str(self.source_root), "worktree", "remove", "--force", str(worktree)), text=True, capture_output=True, check=False)
        if result.returncode:
            raise WorktreeError(result.stderr.strip() or "unable to remove repair worktree")
