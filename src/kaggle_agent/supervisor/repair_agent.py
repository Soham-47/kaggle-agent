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

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.tools = RepairToolbox(self.root)

    def call(self, name: str, **args):
        if name not in REPAIR_AGENT_TOOLS:
            raise ToolPolicyError(f"tool is not available to repair agent: {name}")
        if name == "read_file":
            return self.tools.read_file(str(args.get("path", "")))
        if name == "search_code":
            return self.tools.search_code(str(args.get("query", "")))
        if name == "write_file":
            return self.tools.write_file(str(args.get("path", "")), str(args.get("content", "")), expected_sha256=str(args.get("expected_sha256", "")))
        if name == "run_focused_test":
            return self.tools.run_focused_test(str(args.get("target", "")))
        if name == "run_compile_check":
            return self.tools.run_compile_check()
        if name == "git_diff":
            import subprocess
            return subprocess.run(("git", "-C", str(self.root), "diff", "--no-ext-diff"), text=True, capture_output=True, check=False).stdout
        if name == "list_files":
            return "\n".join(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts)
        if name in {"run_reproduction", "apply_patch"}:
            raise ToolPolicyError(f"{name} requires an explicit supervisor implementation")
        return "done"
