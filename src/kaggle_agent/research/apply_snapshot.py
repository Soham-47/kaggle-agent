"""Write Kaggle research snapshot into lean memory files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from kaggle_agent.kaggle_api.models import ResearchSnapshot
from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import AgentState, load_state, save_state


def apply_kaggle_research(
    snap: ResearchSnapshot,
    root: Path | None = None,
    *,
    agent_max_proposals: int | None = None,
) -> AgentState:
    """Persist snapshot to research.md and refresh budget fields on state.md."""
    base = memory_dir(root)
    base.mkdir(parents=True, exist_ok=True)
    (base / "research.md").write_text(snap.to_research_markdown(), encoding="utf-8")

    state = load_state(root)
    today = date.today().isoformat()
    if state.budget_date != today:
        state.budget_date = today
        state.proposals_used = "0"
    if snap.limits is not None:
        # Agent config may cap lower than Kaggle allowance.
        kaggle_left = snap.limits.num_allowed_now
        if agent_max_proposals is not None:
            state.max_proposals = str(min(int(agent_max_proposals), kaggle_left))
        else:
            state.max_proposals = str(kaggle_left)
        # Live allowance for status / heal (not a secret). public_best stays personal only.
        state.note = f"kaggle_allowed_now={kaggle_left}; today_used={snap.limits.num_today}"
    save_state(state, root)
    return state
