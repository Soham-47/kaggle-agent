"""Supervisor-owned deterministic verification harness."""

from __future__ import annotations

import subprocess
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


class VerificationHarness:
    def _allowed(self, command: list[str]) -> bool:
        return any(tuple(command[:len(prefix)]) == prefix for prefix in _ALLOWED_PREFIXES)

    def run_commands(self, root: Path, commands: list[list[str]] | tuple[list[str], ...]) -> VerificationResult:
        failures: list[str] = []
        seen: list[tuple[str, ...]] = []
        for command in commands:
            seen.append(tuple(command))
            if not self._allowed(command):
                failures.append(f"allowlist rejected: {' '.join(command)}")
                continue
            result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
            if result.returncode:
                failures.append(f"failed ({result.returncode}): {' '.join(command)}\n{(result.stdout + result.stderr)[-4000:]}")
        return VerificationResult(not failures, tuple(seen), tuple(failures))

    def verify(self, root: Path, commands: tuple[str, ...] = ()) -> VerificationResult:
        selected = [["uv", "run", "python", "-m", "compileall", "src"]]
        selected.extend([command.split() for command in commands])
        return self.run_commands(root, selected)
