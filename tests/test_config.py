from pathlib import Path

from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.paths import repo_root


def test_load_settings():
    s = load_settings(repo_root())
    assert s.default_competition == "rsna_knee"
    assert "RESEARCH" in s.phases
    assert s.dry_run is True
    assert s.mcp_submit is False
    assert s.api_submit is True
    assert s.kernel_push is True
    assert s.loop_n_min == 2
    assert s.loop_n_max == 8
    assert s.loop_typical_gain == 0.01
    assert s.loop_default_n == 3
    assert s.loop_max_minutes == 90
    assert s.research_loop_passes == 3
    agent = s.research_agent_config()
    assert agent.max_tool_turns == 40
    assert agent.max_minutes == 15.0
    assert s.plan_agent_config().max_tool_turns == 20
    assert s.code_agent_config().max_minutes == 10.0


def test_load_rsna_competition():
    c = load_competition("rsna_knee", repo_root())
    assert c.slug == "rsna-knee-abnormality-detection"
    assert "Baker's" in c.labels
    assert len(c.labels) == 12
    assert c.metric_direction == "max"


def test_model_fallback_to_settings():
    s = load_settings(repo_root())
    c = load_competition("rsna_knee", repo_root())
    # null in yaml → default from settings
    assert c.model_for("plan", s) == s.zen_model("plan")
