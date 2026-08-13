"""Adaptive inner-loop count and persist memory/loop.md."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import format_kv_markdown, parse_kv_markdown


@dataclass
class LoopState:
    last_score: str = "none"
    prev_score: str = "none"
    last_n: str = "none"
    next_n: str = "3"
    note: str = "none"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> LoopState:
        return cls(
            **{
                name: d.get(name, field.default)  # type: ignore[misc]
                for name, field in cls.__dataclass_fields__.items()
            }
        )


def loop_path(root: Path | None = None) -> Path:
    return memory_dir(root) / "loop.md"


def load_loop(root: Path | None = None) -> LoopState:
    path = loop_path(root)
    if not path.is_file():
        return LoopState()
    return LoopState.from_dict(parse_kv_markdown(path.read_text(encoding="utf-8")))


def save_loop(state: LoopState, root: Path | None = None) -> Path:
    path = loop_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_kv_markdown("loop", state.to_dict()), encoding="utf-8")
    return path


def next_loop_count(
    rate: float | None,
    *,
    n_min: int,
    n_max: int,
    typical_gain: float,
    default_n: int = 3,
) -> int:
    """Map last score-gain rate to next slice count. None rate = first run."""
    if rate is None:
        return int(default_n)
    gain = max(float(rate), 0.0)
    denom = 1.0 + (gain / typical_gain if typical_gain else 0.0)
    n = round(n_min + (n_max - n_min) / denom)
    return int(min(n_max, max(n_min, n)))
