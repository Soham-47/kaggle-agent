from __future__ import annotations

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome
from kaggle_agent.autonomy.runtime import StageExecutor, StageInput, StageLedger


def test_executor_records_success_and_replays_same_idempotency_key(tmp_path):
    calls = []
    executor = StageExecutor(StageLedger(tmp_path))
    request = StageInput.create(
        stage="RESEARCH",
        cycle_id="cycle-1",
        competition="demo",
        inputs={"contract": "abc"},
    )

    def run(_: StageInput) -> StageOutcome:
        calls.append("run")
        return StageOutcome.success("RESEARCH", "sources verified")

    first = executor.execute(request, run)
    second = executor.execute(request, run)

    assert calls == ["run"]
    assert first.outcome.state is OutcomeState.SUCCESS
    assert second.replayed is True
    assert second.outcome.state is OutcomeState.SUCCESS
    records = StageLedger(tmp_path).records()
    assert [record["event"] for record in records] == ["stage_started", "stage_finished"]
    assert records[-1]["idempotency_key"] == request.idempotency_key


def test_pending_outcome_is_recorded_but_reentered_for_reconciliation(tmp_path):
    executor = StageExecutor(StageLedger(tmp_path))
    request = StageInput.create(
        stage="KERNEL_TRAIN",
        cycle_id="cycle-1",
        competition="demo",
        inputs={"kernel_ref": "owner/kernel"},
    )
    calls = []

    def run(_: StageInput) -> StageOutcome:
        calls.append("poll")
        return StageOutcome(
            OutcomeState.PENDING_EXTERNAL,
            "KERNEL_TRAIN",
            "kernel queued",
            external_job="owner/kernel",
        )

    assert executor.execute(request, run).replayed is False
    assert executor.execute(request, run).replayed is False
    assert calls == ["poll", "poll"]
    assert StageLedger(tmp_path).latest(request.idempotency_key)["state"] == "pending_external"


def test_failed_attempt_is_recorded_with_stable_failure_signature(tmp_path):
    executor = StageExecutor(StageLedger(tmp_path))
    request = StageInput.create(
        stage="CODE", cycle_id="cycle-1", competition="demo", inputs={"plan": "v1"}
    )
    result = executor.execute(
        request, lambda _: StageOutcome.failure("CODE", "recipe did not compile")
    )

    assert result.attempt == 1
    latest = StageLedger(tmp_path).latest(request.idempotency_key)
    assert latest["state"] == "recoverable_failure"
    assert latest["failure_signature"] == result.outcome.failure_signature
