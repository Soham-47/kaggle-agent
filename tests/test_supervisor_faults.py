from pathlib import Path

import pytest

from kaggle_agent.supervisor.faults import FaultInjected, FaultInjector, FaultPoint
from kaggle_agent.supervisor.recovery import SupervisorRecovery
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.autonomy.outbox import ExternalActionOutbox, reconcile_with_kaggle
from kaggle_agent.kaggle_api.models import SubmissionRow


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


@pytest.mark.parametrize("point", tuple(FaultPoint))
def test_every_declared_fault_point_requires_explicit_enablement(point: FaultPoint):
    with pytest.raises(FaultInjected, match=point.value):
        FaultInjector(True, {point}).hit(point)


def test_kernel_and_submission_faults_reconcile_without_duplicate_keys(tmp_path: Path):
    outbox = ExternalActionOutbox(tmp_path)
    kernel = outbox.enqueue(action="kernel_push", idempotency_key="k", payload={"kernel_ref": "u/k"})
    injector = FaultInjector(True, {FaultPoint.EXTERNAL_SENT})
    outbox.mark_sent(kernel.action_id)
    with pytest.raises(FaultInjected):
        injector.hit(FaultPoint.EXTERNAL_SENT)
    assert reconcile_with_kaggle(outbox, kernel, kernel_status=lambda _: "RUNNING", submissions=lambda _: []).status == "accepted"
    assert outbox.enqueue(action="kernel_push", idempotency_key="k", payload={}).status == "accepted"

    submit = outbox.enqueue(action="submit", idempotency_key="s", payload={"competition": "demo", "reconciliation_marker": "ka:demo:x"})
    outbox.mark_sent(submit.action_id)
    assert reconcile_with_kaggle(outbox, submit, kernel_status=lambda _: "", submissions=lambda _: [SubmissionRow("s1", "x", "complete", description="ka:demo:x")]).status == "accepted"
    assert outbox.enqueue(action="submit", idempotency_key="s", payload={}).status == "accepted"
