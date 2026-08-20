from pathlib import Path

import pytest

from kaggle_agent.autonomy.repair_tools import RepairToolbox, ToolPolicyError
from kaggle_agent.supervisor.budgets import RepairBudgetStore


def test_repair_toolbox_cannot_write_agent_state_and_supports_compile_check(tmp_path: Path):
    tools = RepairToolbox(tmp_path)
    with pytest.raises(ToolPolicyError):
        tools.write_file(".agent/supervisor.json", "{}", expected_sha256="" )
    result = tools.run_compile_check(["src"])
    assert result.returncode == 0


def test_repair_budgets_persist_and_exhaust_same_signature(tmp_path: Path):
    store = RepairBudgetStore(tmp_path, max_attempts_per_incident=2, max_repairs_per_cycle=3, max_repairs_per_day=5)
    assert store.available("i", "sig", "cycle") is True
    store.record("i", "sig", "cycle", accepted=False)
    store.record("i", "sig", "cycle", accepted=False)
    assert store.available("i", "sig", "cycle") is False
    assert store.available("other", "other", "cycle") is True


def test_apply_patch_is_scoped_and_audited(tmp_path: Path):
    source = tmp_path / "src" / "x.py"
    source.parent.mkdir()
    source.write_text("x = 1\n", encoding="utf-8")
    tools = RepairToolbox(tmp_path)
    tools.apply_patch("diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")
    assert source.read_text(encoding="utf-8") == "x = 2\n"
    with pytest.raises(ToolPolicyError):
        tools.apply_patch("diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-x\n+y\n")
