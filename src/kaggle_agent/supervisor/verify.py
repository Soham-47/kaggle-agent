"""Supervisor-owned deterministic verification harness."""

from __future__ import annotations

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path


_ALLOWED_PREFIXES = (
    ("uv", "run", "pytest"),
    ("uv", "run", "python", "-m", "compileall"),
    ("uv", "run", "python", "-m", "py_compile"),
)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    commands: tuple[tuple[str, ...], ...] = ()
    failures: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "commands": [list(command) for command in self.commands],
            "failures": list(self.failures),
            "stdout": _bounded(self.stdout),
            "stderr": _bounded(self.stderr),
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class VerificationFeedback:
    """Bounded supervisor-owned evidence passed to a fresh revision session."""

    attempt: int
    command: str
    exit_code: int
    failure_kind: str
    failing_tests: tuple[str, ...] = ()
    stderr_excerpt: str = ""
    stdout_excerpt: str = ""
    changed_files: tuple[str, ...] = ()
    diff_summary: str = ""

    @classmethod
    def from_result(
        cls,
        *,
        attempt: int,
        command: str,
        result: VerificationResult,
        changed_files: tuple[str, ...] = (),
        diff_summary: str = "",
    ) -> "VerificationFeedback":
        combined = "\n".join(result.failures) + "\n" + result.stdout + "\n" + result.stderr
        failing = tuple(
            line.strip()[:400]
            for line in combined.splitlines()
            if line.strip().startswith(("FAILED ", "ERROR "))
        )[:20]
        exit_code = result.exit_code if not result.passed else 0
        kind = "PASS" if result.passed else ("TEST_FAILURE" if failing or "pytest" in command else "VERIFICATION_FAILURE")
        stderr = result.stderr or "\n".join(result.failures)
        return cls(
            attempt=attempt,
            command=_bounded(command),
            exit_code=exit_code,
            failure_kind=kind,
            failing_tests=failing,
            stderr_excerpt=_bounded(stderr),
            stdout_excerpt=_bounded(result.stdout),
            changed_files=tuple(_bounded(path, 400) for path in changed_files)[:50],
            diff_summary=_bounded(diff_summary),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "command": self.command,
            "exit_code": self.exit_code,
            "failure_kind": self.failure_kind,
            "failing_tests": list(self.failing_tests),
            "stderr_excerpt": self.stderr_excerpt,
            "stdout_excerpt": self.stdout_excerpt,
            "changed_files": list(self.changed_files),
            "diff_summary": self.diff_summary,
        }


def _bounded(value: object, limit: int = 4000) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value))[:limit]


class VerificationHarness:
    def _allowed(self, command: list[str]) -> bool:
        return any(tuple(command[:len(prefix)]) == prefix for prefix in _ALLOWED_PREFIXES)

    def run_commands(self, root: Path, commands: list[list[str]] | tuple[list[str], ...]) -> VerificationResult:
        failures: list[str] = []
        seen: list[tuple[str, ...]] = []
        stdout: list[str] = []
        stderr: list[str] = []
        exit_code = 0
        for command in commands:
            seen.append(tuple(command))
            if not self._allowed(command):
                failures.append(f"allowlist rejected: {' '.join(command)}")
                if exit_code == 0:
                    exit_code = 126
                continue
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
            if result.returncode and exit_code == 0:
                exit_code = result.returncode
            stdout.append(result.stdout)
            stderr.append(result.stderr)
            if result.returncode:
                failures.append(f"failed ({result.returncode}): {' '.join(command)}\n{(result.stdout + result.stderr)[-4000:]}")
        return VerificationResult(not failures, tuple(seen), tuple(failures), _bounded("\n".join(stdout)), _bounded("\n".join(stderr)), exit_code)

    def verify(self, root: Path, commands: tuple[str, ...] = ()) -> VerificationResult:
        selected = [["uv", "run", "python", "-m", "compileall", "src"]]
        selected.extend([command.split() for command in commands])
        return self.run_commands(root, selected)
