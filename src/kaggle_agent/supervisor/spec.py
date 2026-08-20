"""Persisted, reviewable repair specifications."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from kaggle_agent.supervisor.generation import RuntimeRevision


class SpecReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    NOT_CODE_DEFECT = "NOT_CODE_DEFECT"
    NEEDS_AUTHORITY = "NEEDS_AUTHORITY"


@dataclass(frozen=True)
class SpecReview:
    verdict: SpecReviewVerdict
    blocking_findings: tuple[object, ...] = ()
    non_blocking_findings: tuple[object, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class RepairSpec:
    repair_id: str
    incident_id: str
    base_generation: str
    base_revision: RuntimeRevision
    title: str
    failed_stage: str
    observed_failure: str
    root_cause: str
    current_behavior: str
    expected_behavior: str
    likely_files: tuple[str, ...]
    reproduction_mode: str
    reproduction_commands: tuple[str, ...]
    invariants: tuple[str, ...]
    forbidden_changes: tuple[str, ...]
    required_tests: tuple[str, ...]
    verification_commands: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    max_changed_source_files: int
    max_changed_test_files: int
    max_changed_lines: int
    proposed_resume_stage: str
    risk_level: str
    acceptance_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["base_revision"] = asdict(self.base_revision)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RepairSpec":
        raw = dict(value)
        raw["base_revision"] = RuntimeRevision(**raw["base_revision"])
        for field in ("likely_files", "reproduction_commands", "invariants", "forbidden_changes", "required_tests", "verification_commands", "allowed_paths", "acceptance_criteria"):
            raw[field] = tuple(raw.get(field) or ())
        return cls(**raw)

    def save(self, state_root: Path) -> tuple[Path, Path]:
        directory = state_root / "repairs" / self.repair_id
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "spec.json"
        md_path = directory / "spec.md"
        json_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return json_path, md_path

    def to_markdown(self) -> str:
        lines = [f"# RepairSpec: {self.title}", "", f"- repair_id: `{self.repair_id}`", f"- incident_id: `{self.incident_id}`", f"- base_generation: `{self.base_generation}`", f"- failed_stage: `{self.failed_stage}`", "", "## Root cause", self.root_cause, "", "## Observed failure", self.observed_failure, "", "## Acceptance criteria"]
        lines.extend(f"- {item}" for item in self.acceptance_criteria)
        lines.extend(("", "## Invariants"))
        lines.extend(f"- {item}" for item in self.invariants)
        lines.extend(("", "## Verification commands", "```text"))
        lines.extend(self.verification_commands)
        lines.extend(("```", ""))
        return "\n".join(lines)
