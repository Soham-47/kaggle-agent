"""Append daily log + experiment files (not loaded fully into every prompt)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.paths import memory_dir


def append_daily_log(line: str, root: Path | None = None, when: datetime | None = None) -> Path:
    when = when or datetime.now(timezone.utc)
    path = memory_dir(root) / "daily" / f"{when.strftime('%Y-%m-%d')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(f"# daily {when.strftime('%Y-%m-%d')}\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"- {when.strftime('%H:%M:%S UTC')}: {line}\n")
    return path


def write_experiment(
    exp_id: str,
    *,
    hypothesis: str,
    approach: str = "baseline",
    notes: str = "",
    root: Path | None = None,
) -> Path:
    path = memory_dir(root) / "experiments" / f"{exp_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# {exp_id}

- hypothesis: {hypothesis}
- approach: {approach}
- code_ref: none
- kernel: none
- local_smoke: pending
- cv_auc: none
- submission: none
- public_score: none
- telegram: none
- decision_next: none
- notes: {notes or "none"}
""",
        encoding="utf-8",
    )
    return path
