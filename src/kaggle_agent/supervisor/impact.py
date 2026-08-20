"""Conservative mapping from changed files to earliest invalid stage."""

from __future__ import annotations


class StageImpactAnalyzer:
    _STAGES = ("RESEARCH", "PLAN", "CODE", "LOCAL_SMOKE", "KERNEL_TRAIN", "VALIDATE_SUB", "TELEGRAM_APPROVE", "SUBMIT", "FEEDBACK", "HEAL", "REPORT")

    def earliest_affected_stage(self, changed_files: list[str] | tuple[str, ...]) -> str:
        earliest = len(self._STAGES) - 1
        for raw in changed_files:
            path = raw.replace("\\", "/")
            if path.startswith(("research/", "src/kaggle_agent/research/")):
                index = 0
            elif "agents/plan.py" in path or path.endswith("/plan.py"):
                index = 1
            elif "agents/code.py" in path or "pipeline/" in path:
                index = 2
            elif "local_smoke" in path:
                index = 3
            elif "kernel_" in path or "notebook_builder" in path:
                index = 4
            elif "validate" in path:
                index = 5
            elif path.startswith("heal/") or "/heal/" in path:
                index = 9
            else:
                index = 0
            earliest = min(earliest, index)
        return self._STAGES[earliest]
