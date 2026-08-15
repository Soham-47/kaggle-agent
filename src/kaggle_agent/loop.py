"""Adaptive inner-loop count, stored on memory/state.md."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import load_state, parse_kv_markdown, save_state


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


def load_loop(root: Path | None = None) -> LoopState:
    st = load_state(root)
    legacy = memory_dir(root) / "loop.md"
    if legacy.is_file() and st.loop_last_score == "none" and st.loop_next_n == "3":
        old = LoopState.from_dict(parse_kv_markdown(legacy.read_text(encoding="utf-8")))
        save_loop(old, root)
        legacy.unlink(missing_ok=True)
        st = load_state(root)
    return LoopState(
        last_score=st.loop_last_score,
        prev_score=st.loop_prev_score,
        last_n=st.loop_last_n,
        next_n=st.loop_next_n,
        note=st.loop_note,
    )


def save_loop(state: LoopState, root: Path | None = None) -> Path:
    st = load_state(root)
    st.loop_last_score = state.last_score
    st.loop_prev_score = state.prev_score
    st.loop_last_n = state.last_n
    st.loop_next_n = state.next_n
    st.loop_note = state.note
    return save_state(st, root)


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
        return int(min(n_max, max(n_min, default_n)))
    gain = max(float(rate), 0.0)
    denom = 1.0 + (gain / typical_gain if typical_gain > 0 else 0.0)
    n = round(n_min + (n_max - n_min) / denom)
    return int(min(n_max, max(n_min, n)))


def score_is_better(new: object, old: object, direction: str = "max") -> bool:
    n = parse_loop_score(new)
    if n is None:
        return False
    o = parse_loop_score(old)
    if o is None:
        return True
    return n < o if str(direction).lower() == "min" else n > o


def parse_loop_score(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if text.lower() in {"", "none", "n/a", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def record_run_score(
    loop: LoopState,
    new_score: object,
    *,
    n_used: int,
    n_min: int,
    n_max: int,
    typical_gain: float,
    default_n: int,
    direction: str = "max",
) -> LoopState:
    """Update last/prev scores and next_n from this run's public score."""
    old = parse_loop_score(loop.last_score)
    new = parse_loop_score(new_score)
    if new is None or old is None:
        rate: float | None = None
    elif str(direction).lower() == "min":
        rate = max(old - new, 0.0)
    else:
        rate = max(new - old, 0.0)
    nxt = next_loop_count(
        rate,
        n_min=n_min,
        n_max=n_max,
        typical_gain=typical_gain,
        default_n=default_n,
    )
    if new is None:
        last = loop.last_score
        prev = loop.prev_score
        note = "no score"
    else:
        last = str(new_score).strip()
        prev = loop.last_score
        note = "first score" if rate is None else f"rate {rate}"
    return LoopState(
        last_score=last,
        prev_score=prev,
        last_n=str(n_used),
        next_n=str(nxt),
        note=note,
    )


def update_loop_from_score(
    root: Path | None,
    new_score: object,
    *,
    n_used: int,
    n_min: int,
    n_max: int,
    typical_gain: float,
    default_n: int,
    direction: str = "max",
) -> LoopState:
    updated = record_run_score(
        load_loop(root),
        new_score,
        n_used=n_used,
        n_min=n_min,
        n_max=n_max,
        typical_gain=typical_gain,
        default_n=default_n,
        direction=direction,
    )
    save_loop(updated, root)
    return updated
