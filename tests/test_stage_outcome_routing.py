from pathlib import Path

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.ops.tracing import Tracer
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.paths import repo_root
from kaggle_agent.state_md import AgentState
from kaggle_agent.stages import Stage, StageRun
from helpers import copy_min_workspace


def _orch(tmp_path: Path, debug_runner=None):
    root = tmp_path / "ka"
    copy_min_workspace(root, repo_root())
    return root, Orchestrator(
        load_settings(root), load_competition("rsna_knee", root), root=root,
        debug_runner=debug_runner,
    )


def test_failed_stage_records_incident_and_debug_retries_once(tmp_path: Path):
    calls = {"stage": 0, "debug": 0}

    def debug(failure, incident):
        calls["debug"] += 1
        assert incident.is_file()
        return StageOutcome.success("DEBUG", "patched and verified")

    root, orch = _orch(tmp_path, debug)

    def smoke(state, result):
        calls["stage"] += 1
        result.smoke_ok = calls["stage"] == 2
        if not result.smoke_ok:
            result.errors.append("smoke: NameError stale sampler")
        return state

    orch._stages["LOCAL_SMOKE"] = Stage("LOCAL_SMOKE", smoke)
    result = CycleResult("rsna_knee", False, experiment_id="exp")
    orch._run_named_phases(("LOCAL_SMOKE",), AgentState(), False, result)
    assert calls == {"stage": 2, "debug": 1}
    assert [o.state for o in result.stage_outcomes] == [
        OutcomeState.RECOVERABLE_FAILURE, OutcomeState.SUCCESS, OutcomeState.SUCCESS
    ]


def test_pending_external_never_invokes_debug(tmp_path: Path):
    called = []
    root, orch = _orch(tmp_path, lambda *args: called.append(args))

    def kernel(state, dry, result):
        result.kernel_pending = True
        result.kernel_ref = "owner/job"
        return state

    orch._stages["KERNEL_TRAIN"] = Stage("KERNEL_TRAIN", kernel, uses_dry=True)
    result = CycleResult("rsna_knee", False, experiment_id="exp")
    orch._run_named_phases(("KERNEL_TRAIN",), AgentState(), False, result)
    assert result.stage_outcomes[-1].state is OutcomeState.PENDING_EXTERNAL
    assert called == []


def test_unrepaired_failure_blocks_submission_phases(tmp_path: Path):
    root, orch = _orch(tmp_path, lambda failure, incident: StageOutcome(
        OutcomeState.EXHAUSTED, "DEBUG", "budget exhausted",
        failure_signature=failure.failure_signature,
    ))
    result = CycleResult("rsna_knee", False, experiment_id="exp", smoke_ok=False)
    result.stage_outcomes.append(StageOutcome.failure("LOCAL_SMOKE", "boom"))
    assert orch._submission_blocked(result) is True


def test_typed_stage_outcome_is_durable_and_not_inferred_from_mutable_result(tmp_path: Path):
    root, orch = _orch(tmp_path)
    orch._tracer = Tracer(root, cycle_id="exp")
    expected = StageOutcome.success("LOCAL_SMOKE", "adapter validation passed")
    orch._stages["LOCAL_SMOKE"] = Stage(
        "LOCAL_SMOKE", lambda state, result: StageRun(state, expected)
    )
    result = CycleResult("rsna_knee", False, experiment_id="exp")

    orch._run_named_phases(("LOCAL_SMOKE",), AgentState(), False, result)

    assert result.stage_outcomes == [expected]
    ledger = root / ".agent" / "stage-ledger.jsonl"
    assert ledger.is_file()
    assert '"event": "stage_finished"' in ledger.read_text(encoding="utf-8")
    assert '"state": "success"' in ledger.read_text(encoding="utf-8")
    trace = next((root / "memory" / "daily" / "traces").glob("*.jsonl"))
    assert '"type": "stage_outcome"' in trace.read_text(encoding="utf-8")


def test_unverified_research_or_plan_requires_authority_without_an_llm(tmp_path: Path):
    root, orch = _orch(tmp_path)
    orch.router.client = None
    result = CycleResult("rsna_knee", False, research_verified=False, plan_verified=False)

    research = orch._stage_outcome("RESEARCH", result, 0)
    plan = orch._stage_outcome("PLAN", result, 0)

    assert research.state is OutcomeState.NEEDS_AUTHORITY
    assert plan.state is OutcomeState.NEEDS_AUTHORITY
    result.stage_outcomes.extend((research, plan))
    assert orch._submission_blocked(result) is True


def test_duplicate_kernel_recipe_requires_new_candidate_not_debug_retry(tmp_path: Path):
    root, orch = _orch(tmp_path)
    result = CycleResult("rsna_knee", False, kernel_ok=False, kernel_duplicate=True)

    outcome = orch._stage_outcome("KERNEL_TRAIN", result, 0)

    assert outcome.state is OutcomeState.NEEDS_AUTHORITY


def test_keyboard_interrupt_clears_persisted_lock_and_records_error(tmp_path: Path):
    root, orch = _orch(tmp_path)

    def interrupt(state, result):
        raise KeyboardInterrupt("operator stopped run")

    orch._stages["RESEARCH"] = Stage("RESEARCH", interrupt)
    import pytest

    with pytest.raises(KeyboardInterrupt):
        orch.run_cycle(dry_run=True)

    state = orch._sa.load_state()
    assert state.lock_held is False
    assert state.last_result == "error"
    assert "operator stopped run" in state.last_error
