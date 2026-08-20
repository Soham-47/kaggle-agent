from pathlib import Path

import yaml

from kaggle_agent.config import ConfigError, load_competition, load_settings
from kaggle_agent.paths import repo_root


def test_load_settings():
    s = load_settings(repo_root())
    assert s.default_competition is None
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


def test_competition_fleet_false_is_valid_in_yaml(tmp_path: Path):
    config = tmp_path / "config" / "competitions"
    config.mkdir(parents=True)
    text = (repo_root() / "config" / "competitions" / "rsna_knee.yaml").read_text(
        encoding="utf-8"
    )
    text = text.replace(
        "fleet: [notebooks, papers, github, web, discussions, datasets]",
        "fleet: false",
    )
    (config / "rsna_knee.yaml").write_text(text, encoding="utf-8")
    competition = load_competition("rsna_knee", tmp_path)
    assert competition.fleet_enabled is False
    assert competition.fleet_agents == []


def _write_settings(tmp_path: Path, **overrides: object) -> None:
    raw: dict[str, object] = {
        "default_competition": "rsna_knee",
        "loop": {"n_min": 1, "n_max": 3, "default_n": 2, "typical_gain": 0.01},
    }
    for section, values in overrides.items():
        raw[section] = values
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_invalid_integer_fails_at_settings_load(tmp_path: Path):
    _write_settings(tmp_path, kernel={"poll_attempts": "many"})
    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "config/settings.yaml" in str(exc)
        assert "kernel.poll_attempts" in str(exc)
    else:
        raise AssertionError("malformed integer was accepted")


def test_quoted_false_is_rejected_instead_of_truthy(tmp_path: Path):
    _write_settings(tmp_path, orchestrator={"dry_run": "false"})
    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "orchestrator.dry_run" in str(exc)
        assert "boolean" in str(exc)
    else:
        raise AssertionError("quoted boolean was accepted")


def test_invalid_loop_range_fails_at_settings_load(tmp_path: Path):
    _write_settings(tmp_path, loop={"n_min": 5, "n_max": 2, "default_n": 3, "typical_gain": 0.01})
    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "loop.n_max" in str(exc)
        assert "n_min" in str(exc)
    else:
        raise AssertionError("invalid loop range was accepted")


def test_malformed_float_fails_at_settings_load(tmp_path: Path):
    _write_settings(tmp_path, loop={"typical_gain": "fast"})
    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "loop.typical_gain" in str(exc)
        assert "number" in str(exc)
    else:
        raise AssertionError("malformed float was accepted")


def test_negative_poll_timeout_fails_at_settings_load(tmp_path: Path):
    _write_settings(tmp_path, kernel={"poll_seconds": -1})
    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "kernel.poll_seconds" in str(exc)
        assert ">= 1" in str(exc)
    else:
        raise AssertionError("negative poll timeout was accepted")


def test_invalid_competition_metric_and_submit_mode_fail_at_load(tmp_path: Path):
    config = tmp_path / "config" / "competitions"
    config.mkdir(parents=True)
    (config / "bad.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "bad",
                "slug": "bad",
                "metric": {"direction": "sideways"},
                "submission": {"id_column": "id"},
                "workspace": {"relative": "competitions/bad"},
                "submit": {"mode": "browser"},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_competition("bad", tmp_path)
    except ConfigError as exc:
        assert "metric.direction" in str(exc)
        assert "min or max" in str(exc)
    else:
        raise AssertionError("invalid competition config was accepted")


def test_invalid_competition_submit_mode_fails_at_load(tmp_path: Path):
    config = tmp_path / "config" / "competitions"
    config.mkdir(parents=True)
    (config / "bad.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "bad",
                "slug": "bad",
                "metric": {"direction": "max"},
                "submission": {"id_column": "id"},
                "workspace": {"relative": "competitions/bad"},
                "submit": {"mode": "browser"},
            }
        ),
        encoding="utf-8",
    )
    try:
        load_competition("bad", tmp_path)
    except ConfigError as exc:
        assert "submit.mode" in str(exc)
        assert "file" in str(exc) and "notebook" in str(exc)
    else:
        raise AssertionError("invalid submit mode was accepted")


def test_valid_minimal_settings_keep_optional_defaults(tmp_path: Path):
    _write_settings(tmp_path)
    settings = load_settings(tmp_path)
    assert settings.kernel_poll_seconds == 30
    assert settings.require_telegram_approve is True


def test_default_competition_may_be_unset(tmp_path: Path):
    _write_settings(tmp_path)
    path = tmp_path / "config" / "settings.yaml"
    path.write_text("default_competition: null\n", encoding="utf-8")

    assert load_settings(tmp_path).default_competition is None


def test_default_competition_rejects_non_string_value(tmp_path: Path):
    path = tmp_path / "config"
    path.mkdir()
    (path / "settings.yaml").write_text("default_competition: 42\n", encoding="utf-8")

    try:
        load_settings(tmp_path)
    except ConfigError as exc:
        assert "default_competition" in str(exc)
    else:
        raise AssertionError("numeric default competition was accepted")
