"""Static safety gate for competition-local generated adapters."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AdapterVerdict:
    ok: bool
    violations: tuple[str, ...]


_NETWORK_MODULES = {"requests", "urllib", "httpx", "socket", "aiohttp"}
_SECRET_MARKERS = {"token", "secret", "password", "api_key", "credential"}
_SUBMIT_CALLS = {"competition_submit", "competition_submit_code", "submit"}
_DESTRUCTIVE_CALLS = {"rmtree", "unlink", "remove", "rmdir"}


def validate_generated_adapter(path: Path) -> AdapterVerdict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return AdapterVerdict(False, ("compile",))
    violations: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] in _NETWORK_MODULES for name in names):
                violations.add("network")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and any(mark in target.id.lower() for mark in _SECRET_MARKERS):
                    violations.add("secret")
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _SUBMIT_CALLS:
                violations.add("submission")
            if name in _DESTRUCTIVE_CALLS:
                violations.add("destructive")
    return AdapterVerdict(not violations, tuple(sorted(violations)))
