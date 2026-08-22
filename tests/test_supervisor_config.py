from pathlib import Path

import pytest
import yaml

from kaggle_agent.config import ConfigError, load_settings


def _write_settings(tmp_path: Path, supervisor: dict) -> None:
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": supervisor}),
        encoding="utf-8",
    )


def test_supervisor_settings_are_validated_and_exposed(tmp_path: Path):
    _write_settings(tmp_path, {
        "enabled": True,
        "mode": "repair_only",
        "heartbeat_seconds": 5,
        "heartbeat_timeout_seconds": 20,
        "worker_terminate_grace_seconds": 3,
        "repair": {
            "classification_min_confidence": 0.8,
            "max_attempts_per_incident": 2,
            "max_repairs_per_cycle": 4,
            "max_repairs_per_day": 9,
        },
        "promotion": {"automatic": True},
        "protected": {"strict": True},
    })
    settings = load_settings(tmp_path)
    supervisor = settings.supervisor_config()
    assert supervisor.enabled is True
    assert supervisor.mode == "repair_only"
    assert supervisor.heartbeat_timeout_seconds == 20
    assert supervisor.progress_timeout_seconds == 3600
    assert supervisor.repair.max_changed_lines == 500


@pytest.mark.parametrize(
    "supervisor, message",
    [
        ({"mode": "unsafe"}, "mode"),
        ({"repair": {"classification_min_confidence": 2}}, "confidence"),
        ({"heartbeat_seconds": 20, "heartbeat_timeout_seconds": 20}, "timeout"),
        ({"progress_timeout_seconds": 0}, "progress_timeout_seconds"),
        ({"enabled": "false"}, "boolean"),
        ({"repair": {"max_repairs_per_day": -1}}, "max_repairs_per_day"),
    ],
)
def test_invalid_supervisor_settings_fail_at_load(tmp_path: Path, supervisor, message):
    _write_settings(tmp_path, supervisor)
    with pytest.raises(ConfigError, match=message):
        load_settings(tmp_path)
