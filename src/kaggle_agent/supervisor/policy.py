"""Deterministic safety gates for autonomous repair candidates."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class SafetyViolation(RuntimeError):
    pass


PROTECTED_PATHS = frozenset({
    "src/kaggle_agent/supervisor/policy.py", "src/kaggle_agent/supervisor/verify.py",
    "src/kaggle_agent/supervisor/promote.py", "src/kaggle_agent/supervisor/rollback.py",
    "src/kaggle_agent/autonomy/runtime.py", "src/kaggle_agent/autonomy/outbox.py",
    "src/kaggle_agent/autonomy/stage_outputs.py", "src/kaggle_agent/autonomy/approval.py",
    "src/kaggle_agent/state_md.py", ".env", ".git",
})

_PROTECTED_NAMES = frozenset({".env", "kaggle.json", "credentials.json", "service-account.json"})


@dataclass(frozen=True)
class DiffLimits:
    max_changed_source_files: int = 8
    max_changed_test_files: int = 5
    max_changed_lines: int = 500
    allow_dependency_changes: bool = False


class RepairPolicy:
    def __init__(self, limits: DiffLimits | None = None) -> None:
        self.limits = limits or DiffLimits()

    def require_clean_auto_safe(self, root: Path) -> None:
        result = subprocess.run(("git", "-C", str(root), "status", "--porcelain"), text=True, capture_output=True, check=False)
        if result.returncode:
            raise SafetyViolation("unable to inspect source baseline")
        if result.stdout.strip():
            raise SafetyViolation("DIRTY_SOURCE_BASELINE")

    def protected_violations(self, paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        violations = []
        for path in paths:
            normalized = path.replace("\\", "/")
            if normalized.startswith("./"):
                normalized = normalized[2:]
            normalized = normalized.lstrip("/")
            name = normalized.rsplit("/", 1)[-1].lower()
            if normalized in PROTECTED_PATHS or normalized == ".git" or normalized.startswith(".git/") or name in _PROTECTED_NAMES or name.startswith(".env.") or name.endswith(".pem") or name.endswith(".key") or name.startswith("credentials"):
                violations.append(normalized)
        return tuple(violations)

    def scan_text(self, text: str) -> tuple[str, ...]:
        checks = {
            "eval": "eval(", "exec": "exec(", "os.system": "os.system(",
            "shell": "shell=True", "credential_read": ".env",
            "network": "requests.", "broad_swallow": "except Exception: pass",
            "prompt_injection": "ignore previous instructions",
        }
        return tuple(name for name, marker in checks.items() if marker.lower() in text.lower())

    def scan_test_diff(self, diff: str) -> tuple[str, ...]:
        violations = []
        if "+ assert True" in diff or "+pytest.skip" in diff or "+    pytest.skip" in diff:
            violations.append("test_weakening")
        if any(line.startswith("-") and "assert" in line for line in diff.splitlines()):
            violations.append("test_weakening")
        return tuple(dict.fromkeys(violations))

    def semantic_violations(self, diff: str) -> tuple[str, ...]:
        markers = {
            "approval_bypass": ("first_submission_approved = true", "require_telegram_approve = false", "assume_approved=True"),
            "submission_reconciliation": ("reconcile_with_kaggle", "submission_marker"),
            "target_semantics": ("metric.direction", "target_columns", "competition target"),
            "browser_submission": ("browser_submit", "submit via browser"),
        }
        return tuple(name for name, values in markers.items() if any(value.lower() in diff.lower() for value in values))

    def check_diff(self, paths: list[str], changed_lines: int) -> tuple[str, ...]:
        violations = list(self.protected_violations(paths))
        source = [p for p in paths if p.startswith("src/") and p.endswith(".py")]
        tests = [p for p in paths if p.startswith("tests/") and p.endswith(".py")]
        if len(source) > self.limits.max_changed_source_files:
            violations.append("source_file_limit")
        if len(tests) > self.limits.max_changed_test_files:
            violations.append("test_file_limit")
        if changed_lines > self.limits.max_changed_lines:
            violations.append("changed_line_limit")
        if not self.limits.allow_dependency_changes and any(p in {"pyproject.toml", "uv.lock"} for p in paths):
            violations.append("dependency_change")
        return tuple(dict.fromkeys(violations))
