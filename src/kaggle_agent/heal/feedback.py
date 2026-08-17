"""Score ingestion helpers for the feedback loop (pure logic, no I/O)."""

from __future__ import annotations

import re
from pathlib import Path

from kaggle_agent.kaggle_api.models import SubmissionRow
from kaggle_agent.loop import parse_loop_score

_EXP_PATTERN = re.compile(r"\bagent\s+([0-9]{8}-[0-9]{6}(?:-dry)?)\b")


def first_scored(subs: list[SubmissionRow]) -> SubmissionRow | None:
    """First submission whose public_score parses as a number.

    Never returns a status string: a PENDING submission has an empty score
    and is skipped, so a status can never leak into heal/loop state.
    """
    for row in subs:
        if parse_loop_score(row.public_score) is not None:
            return row
    return None


def exp_id_from_description(description: object) -> str | None:
    """Extract the experiment id from a submission description.

    The submit message is ``agent <experiment_id>``; match that pattern so
    a late score can be attached to the right experiment file.
    """
    text = str(description or "").strip()
    match = _EXP_PATTERN.search(text)
    return match.group(1) if match else None


def exp_public_score(exp_file: Path) -> str | None:
    """Read the ``public_score`` line of an experiment file, if present."""
    if not exp_file.is_file():
        return None
    for line in exp_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- public_score:"):
            value = stripped.split(":", 1)[1].strip()
            return None if value in {"", "none"} else value
    return None


def already_recorded(exp_file: Path, public_score: object) -> bool:
    """True when the experiment file already shows this exact score."""
    current = exp_public_score(exp_file)
    return current is not None and current == str(public_score).strip()
