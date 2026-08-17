"""Append daily log + experiment files (not loaded fully into every prompt)."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.paths import memory_dir

_SCORE_RE = re.compile(r"^-\s*public_score:\s*(\S+)", re.M)
_SUPERSEDED_RE = re.compile(r"^-\s*superseded:\s*\S+", re.M)


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


def supersede_experiment(
    exp_id: str,
    *,
    root: Path | None = None,
    by: str | None = None,
) -> None:
    """Mark an experiment file as superseded (drops it from pick_experiments)."""
    path = memory_dir(root) / "experiments" / f"{exp_id}.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if _SUPERSEDED_RE.search(text):
        return  # already superseded
    line = "- superseded: yes"
    if by:
        line += f" (by {by})"
    path.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")


def _parse_score_from_text(text: str) -> float | None:
    m = _SCORE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    if raw in {"none", "n/a", "nan", ""}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def supersede_worse_experiments(root: Path | None, best_score: str) -> None:
    """Supersede every experiment whose public_score is strictly worse."""
    best = _parse_score_from_text(f"- public_score: {best_score}")
    if best is None:
        return
    exp_dir = memory_dir(root) / "experiments"
    if not exp_dir.is_dir():
        return
    for p in exp_dir.glob("*.md"):
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        if _SUPERSEDED_RE.search(text):
            continue
        exp_score = _parse_score_from_text(text)
        if exp_score is not None and exp_score < best:
            exp_id = p.stem
            supersede_experiment(exp_id, root=root)
