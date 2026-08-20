from pathlib import Path

from helpers import copy_min_workspace
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.heal.policy import HealState, load_heal, save_heal
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.paths import repo_root
from kaggle_agent.state_md import AgentState


def test_pending_score_does_not_advance_heal_or_pause(tmp_path: Path):
    root = tmp_path / "ka"
    copy_min_workspace(root, repo_root())
    heal = HealState(no_improve_days=99, decision_next="pause", note="old")
    save_heal(heal, root)
    orch = Orchestrator(load_settings(root), load_competition("rsna_knee", root), root=root)
    state = AgentState(paused=False)
    result = CycleResult("rsna_knee", False, submit_ok=True, feedback_pending=True)
    returned = orch._heal(state, result)
    assert returned.paused is False
    assert result.heal_decision == "pending_external"
    assert str(load_heal(root).no_improve_days) == "99"
