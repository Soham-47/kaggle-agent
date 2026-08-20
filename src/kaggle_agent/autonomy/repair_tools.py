"""Auditable, path-scoped tools exposed to the DEBUG coding agent."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.autonomy.outcomes import StageOutcome


class ToolPolicyError(RuntimeError):
    pass


class RepairToolbox:
    READ_PREFIXES = ("src", "tests", "competitions", "config", "memory", ".agent")
    WRITE_PREFIXES = ("src", "tests", "competitions", ".agent")
    VERIFY_PREFIXES = (
        ("uv", "run", "pytest"),
        ("uv", "run", "python", "-m", "py_compile"),
    )

    def __init__(self, root: Path, *, timeout_seconds: int = 300) -> None:
        self.root = root.resolve()
        self.timeout_seconds = timeout_seconds
        self.audit_path = self.root / ".agent" / "debug-tools.jsonl"

    def _resolve(self, relative: str, prefixes: tuple[str, ...]) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            rel = candidate.relative_to(self.root)
        except ValueError as exc:
            raise ToolPolicyError("path is outside the repair workspace") from exc
        if not rel.parts or rel.parts[0] not in prefixes:
            raise ToolPolicyError("path is outside the repair envelope")
        return candidate

    def _audit(self, action: str, **payload) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"at": datetime.now(timezone.utc).isoformat(), "action": action, **payload}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def read_file(self, relative: str, *, max_bytes: int = 200_000) -> str:
        path = self._resolve(relative, self.READ_PREFIXES)
        data = path.read_bytes()
        if len(data) > max_bytes:
            raise ToolPolicyError(f"file exceeds read limit: {len(data)} bytes")
        self._audit("read_file", path=relative, sha256=hashlib.sha256(data).hexdigest())
        return data.decode("utf-8")

    def search_code(self, query: str, *, limit: int = 80) -> str:
        if not query or len(query) > 200:
            raise ToolPolicyError("invalid search query")
        hits: list[str] = []
        for prefix in ("src", "tests", "competitions"):
            base = self.root / prefix
            if not base.is_dir():
                continue
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".json", ".yaml", ".md"}:
                    continue
                try:
                    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                        if query.lower() in line.lower():
                            hits.append(f"{path.relative_to(self.root)}:{number}:{line[:300]}")
                            if len(hits) >= limit:
                                self._audit("search_code", query=query, hits=len(hits))
                                return "\n".join(hits)
                except (UnicodeDecodeError, OSError):
                    continue
        self._audit("search_code", query=query, hits=len(hits))
        return "\n".join(hits)

    def write_file(self, relative: str, content: str, *, expected_sha256: str) -> str:
        path = self._resolve(relative, self.WRITE_PREFIXES)
        old = path.read_bytes() if path.exists() else b""
        actual = hashlib.sha256(old).hexdigest()
        if actual != expected_sha256:
            raise ToolPolicyError("file changed since read; refusing overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        new_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self._audit("write_file", path=relative, before=actual, after=new_hash)
        return new_hash

    def run_verification(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = tuple(command)
        if not any(cmd[: len(prefix)] == prefix for prefix in self.VERIFY_PREFIXES):
            raise ToolPolicyError("verification command is not allowlisted")
        result = subprocess.run(
            command,
            cwd=self.root,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        self._audit(
            "run_verification",
            command=command,
            returncode=result.returncode,
            output_sha256=hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest(),
        )
        return result


class IncidentStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def record(
        self,
        failure: StageOutcome,
        *,
        experiment_id: str,
        package_fingerprint: str,
    ) -> Path:
        dest = self.root / ".agent" / "incidents" / experiment_id
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{failure.stage.lower()}-{failure.failure_signature}.json"
        payload = asdict(failure)
        payload["state"] = failure.state.value
        payload["package_fingerprint"] = package_fingerprint
        payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
