from pathlib import Path

import yaml

from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.loop import Supervisor
from kaggle_agent.supervisor.policy import SafetyViolation


def _settings(tmp_path: Path, mode: str, enabled: bool = True):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": enabled, "mode": mode}}), encoding="utf-8")
    return load_settings(tmp_path)


def test_supervisor_off_is_safe_noop(tmp_path: Path):
    result = Supervisor(_settings(tmp_path, "off"), tmp_path).run_once(wait=False)
    assert result.status == "OFF"


def test_auto_safe_refuses_dirty_checkout(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "auto_safe")
    def refuse(self, root):
        raise SafetyViolation("DIRTY_SOURCE_BASELINE")

    monkeypatch.setattr("kaggle_agent.supervisor.policy.RepairPolicy.require_clean_auto_safe", refuse)
    result = Supervisor(settings, tmp_path).run_once(wait=False)
    assert result.status == "DIRTY_SOURCE_BASELINE"
