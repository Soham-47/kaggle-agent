"""Shared tmp workspace setup for cycle tests."""

from __future__ import annotations

from pathlib import Path

from kaggle_agent.loop import LoopState, save_loop
from kaggle_agent.state_md import AgentState, save_state


def write_min_study_csv(root: Path) -> Path:
    """Tmp fixtures do not include study IDs; kernel package needs some."""
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    path = data / "sample_submission.csv"
    path.write_text("StudyInstanceUID\ns1\ns2\n", encoding="utf-8")
    return path


def copy_min_workspace(
    root: Path, real: Path, *, competition: str = "rsna_knee"
) -> None:
    import shutil

    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    write_min_study_csv(root)
    # Real state may be paused; tests need a clean agent
    save_state(AgentState(paused=False, competition=competition), root)
    # Keep existing cycle tests at N=1; production missing loop.md still defaults to 3
    save_loop(LoopState(next_n="1"), root)
