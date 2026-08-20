"""Restricted implementer tool boundary for an approved RepairSpec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.autonomy.repair_tools import RepairToolbox, ToolPolicyError


REPAIR_AGENT_TOOLS = frozenset({
    "list_files", "read_file", "search_code", "git_diff", "write_file",
    "apply_patch", "run_reproduction", "run_focused_test", "run_compile_check", "done",
})


@dataclass(frozen=True)
class RepairAgentResult:
    stopped: bool
    reason: str
    writes: tuple[str, ...] = ()


class RepairAgentBoundary:
    """Only supervisor-created toolbox methods are exposed to an implementer."""

    def __init__(self, root: Path, *, state_root: Path | None = None) -> None:
        self.root = root.resolve()
        self.tools = RepairToolbox(self.root, state_root=state_root)

    @property
    def tools_allowed(self) -> frozenset[str]:
        return REPAIR_AGENT_TOOLS

    def call(self, name: str, **args):
        if name not in REPAIR_AGENT_TOOLS:
            raise ToolPolicyError(f"tool is not available to repair agent: {name}")
        self._validate_args(name, args)
        if name == "read_file":
            return self.tools.read_file(str(args.get("path") or args.get("file_path") or ""))
        if name == "search_code":
            return self.tools.search_code(str(args.get("query", "")))
        if name == "write_file":
            return self.tools.write_file(str(args.get("path") or args.get("file_path") or ""), str(args.get("content", "")), expected_sha256=str(args.get("expected_sha256", "")))
        if name == "apply_patch":
            try:
                return self.tools.apply_patch(str(args.get("patch", "")))
            except ToolPolicyError as exc:
                message = str(exc)
                if any(marker in message for marker in ("patch has no scoped", "patch rejected", "patch failed", "corrupt patch")):
                    raise ToolProtocolError(message) from exc
                raise
        if name == "run_reproduction":
            return self.tools.run_reproduction(str(args.get("target", "")))
        if name == "run_focused_test":
            return self.tools.run_focused_test(str(args.get("target", "")))
        if name == "run_compile_check":
            return self.tools.run_compile_check()
        if name == "git_diff":
            import subprocess
            return subprocess.run(("git", "-C", str(self.root), "diff", "--no-ext-diff"), text=True, capture_output=True, check=False).stdout
        if name == "list_files":
            return "\n".join(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts)
        return "done"

    @staticmethod
    def _validate_args(name: str, args: dict[str, object]) -> None:
        schemas: dict[str, tuple[set[str], set[str]]] = {
            "list_files": (set(), set()),
            "read_file": ({"path", "file_path"}, set()),
            "search_code": ({"query"}, {"query"}),
            "git_diff": (set(), set()),
            "write_file": ({"path", "file_path", "content", "expected_sha256"}, {"content", "expected_sha256"}),
            "apply_patch": ({"patch"}, {"patch"}),
            "run_reproduction": ({"target"}, {"target"}),
            "run_focused_test": ({"target"}, {"target"}),
            "run_compile_check": (set(), set()),
            "done": (set(), set()),
        }
        allowed, required = schemas[name]
        unknown = set(args) - allowed
        if unknown:
            raise ToolProtocolError(f"{name} received unknown arguments: {', '.join(sorted(unknown))}")
        missing = required - set(args)
        if missing:
            raise ToolProtocolError(f"{name} omitted arguments: {', '.join(sorted(missing))}")
        if name == "read_file" and not (args.get("path") or args.get("file_path")):
            raise ToolProtocolError("read_file requires a non-empty path")
        if name == "write_file" and not (args.get("path") or args.get("file_path")):
            raise ToolProtocolError("write_file requires a non-empty path")


class ToolProtocolError(ValueError):
    """The model used a known tool with an invalid typed argument envelope."""
