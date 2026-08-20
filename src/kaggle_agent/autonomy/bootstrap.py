"""Create a competition-neutral local workspace from checked-in templates."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


class InitializationError(ValueError):
    """Raised when initialization would overwrite an existing user file."""


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def initialize_competition(root: Path, competition_id: str, slug: str | None = None) -> Path:
    """Create config, pipeline, and memory files without overwriting anything."""
    cid = competition_id.strip().lower()
    if not _ID_RE.fullmatch(cid):
        raise InitializationError("competition id must use lowercase letters, numbers, '_' or '-'")
    contest_slug = (slug or cid).strip()
    if not contest_slug:
        raise InitializationError("competition slug must not be empty")

    config_dir = root / "config" / "competitions"
    template = config_dir / "_template.yaml"
    config_path = config_dir / f"{cid}.yaml"
    workspace = root / "competitions" / cid / "pipeline"
    memory_dir = root / "memory"
    memory_templates = memory_dir / "templates"
    if not template.is_file():
        raise InitializationError(f"missing competition template: {template}")
    if config_path.exists():
        raise InitializationError(f"already exists: {config_path}")
    if workspace.parent.exists():
        raise InitializationError(f"already exists: {workspace.parent}")
    memory_targets = [memory_dir / name for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md")]
    existing_memory = [path for path in memory_targets if path.exists()]
    if existing_memory:
        names = ", ".join(str(path.relative_to(root)) for path in existing_memory)
        raise InitializationError(f"refusing to overwrite existing runtime memory: {names}")
    missing_templates = [memory_templates / path.name for path in memory_targets if not (memory_templates / path.name).is_file()]
    if missing_templates:
        raise InitializationError("missing memory templates: " + ", ".join(str(path) for path in missing_templates))

    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        template.read_text(encoding="utf-8").replace("my_contest", cid).replace("the-kaggle-url-slug", contest_slug),
        encoding="utf-8",
    )
    workspace.mkdir(parents=True)
    files = {
        "__init__.py": '"""Competition-local pipeline scaffold."""\n',
        "schema.py": '"""Submission schema scaffold; replace with the verified contract."""\nID_COLUMN = "id"\nLABELS = ["target"]\nSUBMISSION_HEADER = [ID_COLUMN, *LABELS]\n',
        "baseline.py": '"""Small deterministic baseline for local smoke tests."""\nfrom .schema import ID_COLUMN, LABELS\n\ndef predict_constant(ids, value=0.5):\n    return [{ID_COLUMN: item, **{label: value for label in LABELS}} for item in ids]\n',
        "recipe.py": '"""Placeholder; CODE replaces this after research and validation."""\n',
        "ranker.py": '"""Optional competition-specific ranker scaffold."""\n',
        "reports.py": '"""Optional report extraction scaffold."""\n',
    }
    for name, content in files.items():
        (workspace / name).write_text(content, encoding="utf-8")
    memory_dir.mkdir(parents=True, exist_ok=True)
    for target in memory_targets:
        shutil.copyfile(memory_templates / target.name, target)
    return config_path
