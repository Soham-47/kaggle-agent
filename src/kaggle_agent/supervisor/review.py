"""Independent review result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REVISE = "REVISE"
    NOT_CODE_DEFECT = "NOT_CODE_DEFECT"
    NEEDS_AUTHORITY = "NEEDS_AUTHORITY"


@dataclass(frozen=True)
class ReviewFinding:
    severity: str
    file: str
    issue: str
    required_fix: str


@dataclass(frozen=True)
class Review:
    verdict: ReviewVerdict
    root_cause_fixed: bool = False
    tests_sufficient: bool = False
    idempotency_safe: bool = False
    checkpoint_safe: bool = False
    policy_safe: bool = False
    blocking_findings: tuple[ReviewFinding, ...] = ()
    non_blocking_findings: tuple[ReviewFinding, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "Review":
        try:
            verdict = ReviewVerdict(str(value.get("verdict", "REJECT")).upper())
        except ValueError:
            verdict = ReviewVerdict.REJECT

        def findings(key: str) -> tuple[ReviewFinding, ...]:
            rows = value.get(key) or []
            return tuple(ReviewFinding(
                severity=str(row.get("severity", "high")), file=str(row.get("file", "")),
                issue=str(row.get("issue", "")), required_fix=str(row.get("required_fix", "")),
            ) for row in rows if isinstance(row, dict))

        return cls(verdict, bool(value.get("root_cause_fixed", False)), bool(value.get("tests_sufficient", False)), bool(value.get("idempotency_safe", False)), bool(value.get("checkpoint_safe", False)), bool(value.get("policy_safe", False)), findings("blocking_findings"), findings("non_blocking_findings"))
