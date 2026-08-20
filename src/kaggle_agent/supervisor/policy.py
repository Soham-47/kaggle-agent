"""Deterministic safety gates for autonomous repair candidates."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SafetyViolation(RuntimeError):
    pass


PROTECTED_PATHS = frozenset({
    "src/kaggle_agent/supervisor/policy.py", "src/kaggle_agent/supervisor/risk.py",
    "src/kaggle_agent/supervisor/config.py", "src/kaggle_agent/supervisor/verify.py",
    "src/kaggle_agent/supervisor/promote.py", "src/kaggle_agent/supervisor/rollback.py",
    "src/kaggle_agent/config.py",
    "config/settings.yaml", "config/profiles",
    "src/kaggle_agent/autonomy/runtime.py", "src/kaggle_agent/autonomy/outbox.py",
    "src/kaggle_agent/autonomy/stage_outputs.py", "src/kaggle_agent/autonomy/approval.py",
    "src/kaggle_agent/state_md.py", ".env", ".git",
})


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
            normalized = path.replace("\\", "/").lstrip("./")
            if normalized in PROTECTED_PATHS or normalized == ".git" or normalized.startswith(".git/"):
                violations.append(normalized)
        return tuple(violations)

    def allowed_path_violations(
        self, paths: list[str] | tuple[str, ...], allowed_paths: tuple[str, ...]
    ) -> tuple[str, ...]:
        allowed = tuple(path.replace("\\", "/").lstrip("./").rstrip("/") for path in allowed_paths)
        violations = []
        for path in paths:
            normalized = path.replace("\\", "/").lstrip("./")
            if not any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in allowed):
                violations.append(normalized)
        return tuple(violations)

    def scan_text(self, text: str) -> tuple[str, ...]:
        source_lines = "\n".join(
            line[1:] if line.startswith(("+", "-")) else line
            for line in text.splitlines()
        )
        normalized = re.sub(r"\s+", " ", source_lines).lower()
        checks = {
            "eval": "eval(", "exec": "exec(", "os.system": "os.system(",
            "shell": "shell=true", "credential_read": ".env",
            "network": "requests.", "http_client": "httpx.", "subprocess": "subprocess",
            "broad_swallow": "except exception: pass", "bare_swallow": "except: pass",
            "unbounded_loop": "while true:",
            "prompt_injection": "ignore previous instructions",
        }
        return tuple(name for name, marker in checks.items() if marker in normalized)

    def scan_test_diff(self, diff: str) -> tuple[str, ...]:
        violations = []
        if "+ assert True" in diff or "+pytest.skip" in diff or "+    pytest.skip" in diff:
            violations.append("test_weakening")
        if any(marker in diff for marker in ("+@pytest.mark.xfail", "+ @pytest.mark.xfail", "+@pytest.mark.skip", "+ @pytest.mark.skip")):
            violations.append("test_weakening")
        if any(line.startswith("-") and "assert" in line for line in diff.splitlines()):
            violations.append("test_weakening")
        return tuple(dict.fromkeys(violations))

    def semantic_violations(self, diff: str) -> tuple[str, ...]:
        markers = {
            "approval_bypass": ("first_submission_approved = true", "require_telegram_approve = false", "assume_approved=True"),
            "submission_reconciliation": ("reconcile_with_kaggle", "submission_marker", "external_action_key", "kernel_push_key", "submission_key"),
            "target_semantics": ("metric.direction", "target_columns", "competition target"),
            "browser_submission": ("browser_submit", "submit via browser"),
            "approval_policy": ("first_submission_approved", "approval_required", "telegram_approve"),
            "credential_policy": ("deepseek_api_key", "telegram_bot_token", "kaggle.json", "credential_loader"),
            "promotion_policy": ("automatic_promotion", "repairacceptance", "protected_paths"),
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
        if not self.limits.allow_dependency_changes and any(
            p in {"pyproject.toml", "uv.lock"} or p == "requirements.txt" or p.startswith("requirements-")
            for p in paths
        ):
            violations.append("dependency_change")
        return tuple(dict.fromkeys(violations))
