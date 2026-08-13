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
