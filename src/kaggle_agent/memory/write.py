"""Append daily log + experiment files (not loaded fully into every prompt)."""

from __future__ import annotations

import re
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


def patch_experiment(
    exp_id: str,
    *,
    root: Path | None = None,
    public_score: str | None = None,
    submission: str | None = None,
    kernel: str | None = None,
    judge: str | None = None,
) -> Path | None:
    path = memory_dir(root) / "experiments" / f"{exp_id}.md"
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    pairs = {
        "public_score": public_score,
        "submission": submission,
        "kernel": kernel,
    }
    lines = []
    saw_judge = False
    for line in text.splitlines():
        replaced = False
        for key, val in pairs.items():
            if val is None:
                continue
            if line.strip().startswith(f"- {key}:"):
                lines.append(f"- {key}: {val}")
                replaced = True
                break
        if not replaced:
            if judge is not None and line.strip().startswith("- judge:"):
                lines.append(f"- judge: {judge}")
                saw_judge = True
                continue
            lines.append(line)
    if judge is not None and not saw_judge:
        lines.append(f"- judge: {judge}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def patch_memory_public_score(score: str, root: Path | None = None) -> None:
    path = memory_dir(root) / "MEMORY.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(
        r"(^- public_score:\s*).*$",
        rf"\g<1>{score}",
        text,
        count=1,
        flags=re.M,
    )
    if n:
        path.write_text(new, encoding="utf-8")
