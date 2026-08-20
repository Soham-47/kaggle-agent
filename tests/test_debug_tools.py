import hashlib
from pathlib import Path

import pytest

from kaggle_agent.autonomy.repair_tools import IncidentStore, RepairToolbox, ToolPolicyError
from kaggle_agent.autonomy.outcomes import StageOutcome


def test_repair_toolbox_requires_expected_hash_and_stays_in_scope(tmp_path: Path):
    source = tmp_path / "competitions/demo/pipeline/recipe.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    tools = RepairToolbox(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    tools.write_file("competitions/demo/pipeline/recipe.py", "VALUE = 2\n", expected_sha256=digest)
    assert source.read_text() == "VALUE = 2\n"
    with pytest.raises(ToolPolicyError, match="changed since read"):
        tools.write_file("competitions/demo/pipeline/recipe.py", "VALUE = 3\n", expected_sha256=digest)
    with pytest.raises(ToolPolicyError, match="outside"):
        tools.read_file("../secret")


def test_repair_toolbox_only_runs_allowlisted_verification(tmp_path: Path):
    tools = RepairToolbox(tmp_path)
    with pytest.raises(ToolPolicyError, match="allowlisted"):
        tools.run_verification(["bash", "-c", "echo nope"])
    result = tools.run_verification(["uv", "run", "python", "-m", "py_compile", "missing.py"])
    assert result.returncode != 0


def test_incident_store_persists_evidence_without_using_ingested_memory(tmp_path: Path):
    store = IncidentStore(tmp_path)
    failure = StageOutcome.failure("LOCAL_SMOKE", "NameError: stale sampler", "sampler")
    path = store.record(failure, experiment_id="exp-1", package_fingerprint="pkg")
    assert path.parent == tmp_path / ".agent/incidents/exp-1"
    assert "stale sampler" in path.read_text()
    assert not (tmp_path / "memory").exists()
