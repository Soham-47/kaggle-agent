"""Self-heal policy: tune → recipe → new → pause."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import format_kv_markdown, parse_kv_markdown

APPROACHES = ("tune", "recipe", "new", "pause")


@dataclass
class HealState:
    approach: str = "baseline"  # current approach family
    tune_attempts: str = "0"
    no_improve_days: str = "0"
    last_score: str = "none"
    best_score: str = "none"
    decision_next: str = "baseline"
    last_improved: str = "never"
    note: str = "none"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> HealState:
        return cls(
            **{
                name: d.get(name, field.default)  # type: ignore[misc]
                for name, field in cls.__dataclass_fields__.items()
            }
        )


def heal_path(root: Path | None = None) -> Path:
    return memory_dir(root) / "heal.md"


def load_heal(root: Path | None = None) -> HealState:
    path = heal_path(root)
    if not path.is_file():
        return HealState()
    return HealState.from_dict(parse_kv_markdown(path.read_text(encoding="utf-8")))


def save_heal(state: HealState, root: Path | None = None) -> Path:
    path = heal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_kv_markdown("heal", state.to_dict()), encoding="utf-8")
    return path


def _parse_score(raw: str | None) -> float | None:
    if raw is None or raw in {"", "none", "n/a"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def decide_next(
    heal: HealState,
    *,
    public_score: str | None,
    metric_direction: str = "max",
    max_tune_attempts: int = 3,
    max_no_improve_days: int = 5,
    cycle_ok: bool = True,
) -> HealState:
    """Update heal state and set decision_next.

    Improved = score strictly better than best_score (direction-aware).
    """
    score = _parse_score(public_score)
    best = _parse_score(heal.best_score)
    improved = False

    if score is not None:
        heal.last_score = f"{score}"
        if best is None:
            improved = True
        elif metric_direction == "min":
            improved = score < best
        else:
            improved = score > best
        if improved:
            heal.best_score = f"{score}"
            heal.last_improved = date.today().isoformat()
            heal.no_improve_days = "0"
            heal.tune_attempts = "0"
            heal.decision_next = "tune"
            heal.approach = "tune"
            heal.note = "new best — continue tuning"
            return heal

    # No improvement this cycle (or no score)
    if best is None:
        # Never scored a real submission: count nothing, never pause.
        heal.decision_next = "tune"
        heal.approach = "tune"
        heal.note = "no public score yet — keep tuning"
        return heal

    try:
        flat = int(heal.no_improve_days or "0") + 1
    except ValueError:
        flat = 1
    heal.no_improve_days = str(flat)

    try:
        tunes = int(heal.tune_attempts or "0")
    except ValueError:
        tunes = 0

    if not cycle_ok:
        heal.decision_next = "tune"
        heal.note = "cycle errors — retry small fix"
        return heal

    if flat >= max_no_improve_days:
        heal.decision_next = "pause"
        heal.approach = "pause"
        heal.note = f"no improve {flat}d — pause"
        return heal

    if tunes < max_tune_attempts:
        heal.tune_attempts = str(tunes + 1)
        heal.decision_next = "tune"
        heal.approach = "tune"
        heal.note = f"tune attempt {heal.tune_attempts}/{max_tune_attempts}"
        return heal

    # Exhausted tunes this streak → recipe, then new
    if heal.approach in {"baseline", "tune"}:
        heal.decision_next = "recipe"
        heal.approach = "recipe"
        heal.tune_attempts = "0"
        heal.note = "move to recipe change"
        return heal

    if heal.approach == "recipe":
        heal.decision_next = "new"
        heal.approach = "new"
        heal.tune_attempts = "0"
        heal.note = "try new approach from research"
        return heal

    heal.decision_next = "pause"
    heal.approach = "pause"
    heal.note = "exhausted ladder — pause"
    return heal
