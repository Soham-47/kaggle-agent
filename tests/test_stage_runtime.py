from __future__ import annotations

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome
from kaggle_agent.autonomy.runtime import StageExecutor, StageInput, StageLedger, StageResult
from kaggle_agent.orchestrator import CycleResult, Orchestrator


def test_executor_records_success_and_replays_same_idempotency_key(tmp_path):
    calls = []
    executor = StageExecutor(StageLedger(tmp_path))
    request = StageInput.create(
        stage="RESEARCH",
        cycle_id="cycle-1",
        competition="demo",
        inputs={"contract": "abc"},
    )
    assert request.stage_execution_key == request.idempotency_key

    def run(_: StageInput) -> StageResult:
        calls.append("run")
        return StageResult(
            StageOutcome.success("RESEARCH", "sources verified"),
            {"research_verified": True, "source_count": 2},
        )

    first = executor.execute(request, run)
    second = executor.execute(request, run)

    assert calls == ["run"]
    assert first.outcome.state is OutcomeState.SUCCESS
    assert second.replayed is True
    assert second.outcome.state is OutcomeState.SUCCESS
    assert second.outputs == {"research_verified": True, "source_count": 2}
    records = StageLedger(tmp_path).records()
    assert [record["event"] for record in records] == ["stage_started", "stage_finished"]
    assert records[-1]["idempotency_key"] == request.idempotency_key
    assert records[-1]["schema_version"] == 1
    assert records[-1]["outputs"] == {"research_verified": True, "source_count": 2}


def test_legacy_success_without_outputs_is_rerun_safely(tmp_path):
    ledger = StageLedger(tmp_path)
    request = StageInput.create(stage="PLAN", cycle_id="cycle-1", competition="demo", inputs={})
    ledger.append("stage_started", request, attempt=1)
    ledger.append(
        "stage_finished",
        request,
        outcome=StageOutcome.success("PLAN", "legacy"),
        attempt=1,
    )
    calls = []
    execution = StageExecutor(ledger).execute(
        request,
        lambda _: (calls.append("run") or StageOutcome.success("PLAN", "new")),
    )
    assert execution.replayed is False
    assert calls == ["run"]


def test_outputs_exclude_secret_like_fields(tmp_path):
    executor = StageExecutor(StageLedger(tmp_path))
    request = StageInput.create(stage="CODE", cycle_id="cycle-1", competition="demo", inputs={})
    result = executor.execute(
        request,
        lambda _: StageResult(
            StageOutcome.success("CODE", "ok"),
            {
                "recipe_path": "/tmp/recipe.py",
                "api_key": "secret",
                "nested": {"token": "secret", "ok": True},
            },
        ),
    )
    assert result.outputs == {"recipe_path": "/tmp/recipe.py", "nested": {"ok": True}}
    assert "api_key" not in StageLedger(tmp_path).latest(request.idempotency_key)["outputs"]


def test_orchestrator_rehydrates_only_allowlisted_stage_outputs():
    result = CycleResult("demo", dry_run=False)
    Orchestrator._restore_stage_outputs(
        result,
        "KERNEL_TRAIN",
        {
            "kernel_ref": "owner/kernel",
            "kernel_pending": True,
            "kernel_path": "/tmp/kernel",
            "submit_message": "must not be restored here",
        },
    )
    assert result.kernel_ref == "owner/kernel"
    assert result.kernel_pending is True
    assert result.kernel_path == "/tmp/kernel"
    assert result.submit_message is None


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
