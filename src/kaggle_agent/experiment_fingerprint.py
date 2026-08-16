"""Stable fingerprints for experiment and kernel duplicate checks."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
from typing import Any


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash JSON data without depending on mapping key order."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _digest(encoded)


def recipe_hash(recipe: str) -> str:
    """Hash recipe source while ignoring only trailing whitespace."""
    lines = recipe.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    normalized = "\n".join(line.rstrip() for line in lines)
    return _digest(normalized)


def recipe_logic_hash(recipe: str) -> str:
    """Hash executable recipe logic while ignoring comments and variant markers."""
    tree = ast.parse(recipe)
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "EXPERIMENT_VARIANT"
                for target in node.targets
            )
        )
    ]
    return _digest(ast.dump(tree, annotate_fields=True, include_attributes=False))


def submission_output_hash(path: Path, id_column: str) -> str:
    """Hash a submission CSV's data, ignoring row order and line endings.

    Two runs that predict the same labels for the same ids hash identically
    even if the writer emitted rows in a different order.
    """
    import csv

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    rows.sort(key=lambda r: r.get(id_column, ""))
    return canonical_hash({"columns": fieldnames, "rows": rows})


def experiment_fingerprint(
    plan_text: str,
    methods: dict[str, Any],
    recipe: str,
    brief_path: Path,
    *,
    seed: int,
) -> str:
    brief = brief_path.read_text(encoding="utf-8") if brief_path.is_file() else ""
    payload = {
        "plan": plan_text.strip(),
        "methods": methods,
        "recipe": recipe_hash(recipe),
        "brief": brief.strip(),
        "seed": int(seed),
    }
    return canonical_hash(payload)
