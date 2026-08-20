from pathlib import Path

import yaml

from kaggle_agent.config import load_settings
from kaggle_agent.paths import repo_root


def test_controlled_auto_safe_profile_is_explicit_and_restricted():
    path = Path(__file__).parents[1] / "config" / "profiles" / "controlled-auto-safe.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    supervisor = raw["supervisor"]

    assert supervisor["enabled"] is True
    assert supervisor["mode"] == "auto_safe"
    assert supervisor["promotion"]["automatic"] is True
    assert supervisor["protected"]["strict"] is True
    repair = supervisor["repair"]
    assert repair["max_repairs_per_cycle"] == 5
    assert repair["max_attempts_per_incident"] == 3
    assert repair["max_repairs_per_day"] == 20
    assert repair["max_changed_source_files"] == 8
    assert repair["max_changed_test_files"] == 1
    assert repair["max_changed_lines"] == 500
    assert repair["allow_dependency_changes"] is False
    assert repair["require_spec_review"] is True
    assert repair["require_code_review"] is True
    assert repair["require_full_tests"] is True


def test_checked_in_defaults_do_not_enable_auto_safe():
    path = Path(__file__).parents[1] / "config" / "settings.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    supervisor = raw["supervisor"]

    assert supervisor["enabled"] is False
    assert supervisor["mode"] == "observe"
    assert supervisor["promotion"]["automatic"] is False


def test_controlled_profile_is_explicitly_loadable_without_changing_defaults():
    settings = load_settings(repo_root(), profile="controlled-auto-safe")
    supervisor = settings.supervisor_config()

    assert supervisor.enabled is True
    assert supervisor.mode == "auto_safe"
    assert supervisor.promotion_automatic is True
    assert supervisor.repair.max_repairs_per_cycle == 5
