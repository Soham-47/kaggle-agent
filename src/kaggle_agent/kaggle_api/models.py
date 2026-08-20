"""Plain value types for the Kaggle adapter (no SDK types leak out)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SubmissionLimits:
    """Team submission allowance.

    Mapped from ApiSubmissionLimits (kaggle package).
    Fields: num_today, num_total, num_allowed_now, limited_by_total
    """

    num_today: int
    num_total: int
    num_allowed_now: int
    limited_by_total: bool = False

    @property
    def can_submit(self) -> bool:
        return self.num_allowed_now > 0


@dataclass(frozen=True)
class MetaFile:
    name: str
    total_bytes: int
    ref: str = ""


@dataclass(frozen=True)
class CompetitionInfo:
    slug: str
    title: str
    url: str
    deadline: str
    evaluation_metric: str
    kernels_only: bool
    max_daily_submissions: int
    tags: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class LeaderboardRow:
    team_name: str
    score: str
    team_id: int | None = None
    submission_date: str = ""


@dataclass(frozen=True)
class KernelRow:
    ref: str
    title: str
    author: str = ""
    total_votes: int | None = None
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url and self.ref:
            object.__setattr__(self, "url", f"https://www.kaggle.com/code/{self.ref}")


@dataclass(frozen=True)
class SubmissionRow:
    ref: str
    file_name: str
    status: str
    public_score: str = ""
    date: str = ""
    description: str = ""


@dataclass(frozen=True)
class SubmitResult:
    dry_run: bool
    message: str
    success: bool = True
    raw_status: str = ""


@dataclass
class ResearchSnapshot:
    """Everything RESEARCH needs from Kaggle in one structure."""

    competition: str
    limits: SubmissionLimits | None = None
    meta_files: list[MetaFile] = field(default_factory=list)
    leaderboard: list[LeaderboardRow] = field(default_factory=list)
    kernels: list[KernelRow] = field(default_factory=list)
    my_submissions: list[SubmissionRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_research_markdown(self) -> str:
        lines = [
            "# research",
            "",
            "Distilled from Kaggle API (not browser). Keep short.",
            "",
            f"## Competition `{self.competition}`",
            "",
        ]
        if self.limits:
            lines += [
                "### Submission limits",
                "",
                f"- today: {self.limits.num_today}",
                f"- total: {self.limits.num_total}",
                f"- allowed_now: {self.limits.num_allowed_now}",
                f"- limited_by_total: {self.limits.limited_by_total}",
                f"- can_submit: {self.limits.can_submit}",
                "",
            ]
        if self.meta_files:
            lines += ["### Meta files (root CSV/JSON)", ""]
            for f in self.meta_files:
                lines.append(f"- `{f.name}` ({f.total_bytes} bytes)")
            lines.append("")
        if self.leaderboard:
            lines += ["### Leaderboard (public top)", ""]
            for i, row in enumerate(self.leaderboard, 1):
                lines.append(f"{i}. {row.team_name} — {row.score}")
            lines.append("")
        if self.kernels:
            lines += ["### Top public kernels", ""]
            for k in self.kernels:
                votes = f" votes={k.total_votes}" if k.total_votes is not None else ""
                lines.append(f"- [{k.title}]({k.url}){votes}")
            lines.append("")
        if self.my_submissions:
            lines += ["### Our recent submissions", ""]
            for s in self.my_submissions:
                lines.append(
                    f"- {s.date or s.ref}: status={s.status} score={s.public_score or 'n/a'} file={s.file_name}"
                )
            lines.append("")
        if self.errors:
            lines += ["### Errors", ""]
            for e in self.errors:
                lines.append(f"- {e}")
            lines.append("")
        return "\n".join(lines)
