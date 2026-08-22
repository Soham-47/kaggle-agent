from pathlib import Path

import pytest

from kaggle_agent.supervisor.faults import FaultInjected, FaultInjector, FaultPoint
from kaggle_agent.supervisor.recovery import SupervisorRecovery
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


def test_fault_injection_is_disabled_by_default():
    FaultInjector().hit(FaultPoint.STAGE_ENTRY)


def test_enabled_fault_injection_is_explicit():
    injector = FaultInjector(True, {FaultPoint.REVIEW_REJECTED})
    with pytest.raises(FaultInjected):
        injector.hit(FaultPoint.REVIEW_REJECTED)


def test_recovery_does_not_adopt_without_owned_fresh_worker(tmp_path: Path):
    store = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    result = SupervisorRecovery(store).inspect_worker("w1", timeout_seconds=30)
    assert result.action == "START_OR_RESUME"
