"""Persistent record of submitted kernel package fingerprints."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import fcntl

from kaggle_agent.paths import memory_dir


def history_path(root: Path | None = None) -> Path:
    return memory_dir(root) / "kernel_history.jsonl"


def package_fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in folder.rglob("*") if path.is_file() and "output" not in path.parts
    )
    required = {"agent_baseline.ipynb", "kernel-metadata.json"}
    if not required.issubset({path.name for path in files}):
        missing = sorted(required - {path.name for path in files})
        raise ValueError(f"kernel package missing {', '.join(missing)}")
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def kernel_push_lock(root: Path | None) -> Iterator[None]:
    """Serialize duplicate checks and pushes for one workspace."""
    if root is None:
        yield
        return
    path = memory_dir(root) / "kernel_history.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def seen_kernel(root: Path | None, fingerprint: str) -> bool:
    path = history_path(root)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("fingerprint") == fingerprint:
            return True
    return False


def record_kernel(root: Path | None, kernel_ref: str, fingerprint: str) -> Path:
    path = history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"kernel_ref": kernel_ref, "fingerprint": fingerprint},
                sort_keys=True,
            )
            + "\n"
        )
    return path
