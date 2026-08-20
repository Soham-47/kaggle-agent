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
    assert agent.max_tokens == 2048
    assert s.block_submit is True
    assert s.plan_agent_config().max_tool_turns == 20
    assert s.code_agent_config().max_minutes == 10.0
    assert s.llm_provider() == "deepseek"
    assert s.zen_model("plan") == "deepseek-v4-flash"


def test_env_example_documents_supported_llm_key():
    example = (repo_root() / ".env.example").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=" in example
    assert "OPENCODE_API_KEY" not in example


def test_dotenv_sets_missing_key_only(tmp_path: Path, monkeypatch):
    from kaggle_agent.config import load_dotenv

    monkeypatch.delenv("DOTENV_SMOKE_KEY", raising=False)
    monkeypatch.setenv("DOTENV_KEEP", "already")
    (tmp_path / ".env").write_text(
        "DOTENV_SMOKE_KEY=from-file\nDOTENV_KEEP=ignored\n",
        encoding="utf-8",
    )
    load_dotenv(tmp_path)
    import os

    assert os.environ["DOTENV_SMOKE_KEY"] == "from-file"
    assert os.environ["DOTENV_KEEP"] == "already"


def test_load_rsna_competition():
    c = load_competition("rsna_knee", repo_root())
    assert c.slug == "rsna-knee-abnormality-detection"
    assert "Baker's" in c.labels
    assert len(c.labels) == 12
    assert c.metric_direction == "max"
    assert c.submission_min_rows == 1000


def test_model_fallback_to_settings():
    s = load_settings(repo_root())
    c = load_competition("rsna_knee", repo_root())
    # null in yaml → default from settings
    assert c.model_for("plan", s) == s.zen_model("plan")


def test_research_fleet_config_defaults():
    s = load_settings(repo_root())
    fleet = s.research_fleet_config()
    assert fleet.enabled is False
    assert len(fleet.agents) == 6
    assert fleet.max_tool_turns == 24
    assert fleet.max_minutes == 15.0


def test_rsna_competition_enables_fleet():
    c = load_competition("rsna_knee", repo_root())
    assert c.fleet_enabled is True
    assert c.fleet_agents == [
        "notebooks",
        "papers",
        "github",
        "web",
        "discussions",
        "datasets",
    ]


def test_competition_fleet_requires_bool_or_roster_list():
    c = load_competition("rsna_knee", repo_root())
    c.raw["research"]["fleet"] = {"enabled": True}
    assert c.fleet_enabled is False
    assert c.fleet_agents == []
    c.raw["research"]["fleet"] = ["notebooks"]
    assert c.fleet_enabled is True
    assert c.fleet_agents == ["notebooks"]
    c.raw["research"]["fleet"] = []
    assert c.fleet_enabled is False
    assert c.fleet_agents == []
