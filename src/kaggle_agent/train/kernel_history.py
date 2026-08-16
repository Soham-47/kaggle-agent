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


def package_recipe_hash(folder: Path) -> str:
    """Read the recipe hash from a kernel package manifest."""
    path = folder / "kernel-metadata.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    manifest = metadata.get("experiment_manifest")
    return str(manifest.get("recipe_sha256") or "") if isinstance(manifest, dict) else ""


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


def seen_recipe(root: Path | None, recipe_hash: str) -> bool:
    """Detect a recipe that already reached Kaggle, including old records."""
    if root is None or not recipe_hash:
        return False
    history = history_path(root)
    refs: set[str] = set()
    if history.is_file():
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("recipe_hash") == recipe_hash:
                return True
            ref = str(row.get("kernel_ref") or "")
            if ref:
                refs.add(ref)
    for metadata_path in root.glob("competitions/*/notebooks/*/kernel-metadata.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(metadata.get("id") or "") not in refs:
            continue
        manifest = metadata.get("experiment_manifest")
        if isinstance(manifest, dict) and manifest.get("recipe_sha256") == recipe_hash:
            return True
    return False


def record_kernel(
    root: Path | None, kernel_ref: str, fingerprint: str, recipe_hash: str = ""
) -> Path:
    path = history_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kernel_ref": kernel_ref,
                    "fingerprint": fingerprint,
                    "recipe_hash": recipe_hash,
                },
                sort_keys=True,
            )
            + "\n"
        )
    return path
