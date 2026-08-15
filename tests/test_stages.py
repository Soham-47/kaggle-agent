"""Stage registry: uniform phase dispatch for the orchestrator."""

from kaggle_agent.stages import Stage, build_stage_registry
from kaggle_agent.state_md import AgentState


class _Stub:
    def _research(self, state, result):  # noqa: ANN001
        return state

    def _plan(self, state, dry, result):  # noqa: ANN001
        return state

    def _code(self, state, result):  # noqa: ANN001
        return state

    def _local_smoke(self, state, result):  # noqa: ANN001
        return state

    def _kernel_train(self, state, dry, result):  # noqa: ANN001
        return state

    def _validate_sub(self, state, result):  # noqa: ANN001
        return state

    def _telegram_approve(self, state, dry, result):  # noqa: ANN001
        return state

    def _submit(self, state, dry, result):  # noqa: ANN001
        return state

    def _feedback(self, state, dry, result):  # noqa: ANN001
        return state

    def _report(self, state, dry, result):  # noqa: ANN001
        return state

    def _heal(self, state, result):  # noqa: ANN001
        return state


def test_stage_run_forwards_dry_when_flagged():
    seen: list[tuple] = []

    def fn(state, dry, result):  # noqa: ANN001
        seen.append((state, dry, result))
        return state

    stage = Stage("PLAN", fn, uses_dry=True)
    st = AgentState()
    out = stage.run(st, True, "res")
    assert out is st
    assert seen == [(st, True, "res")]


def test_stage_run_omits_dry_when_not_flagged():
    seen: list[tuple] = []

    def fn(state, result):  # noqa: ANN001
        seen.append((state, result))
        return state

    stage = Stage("CODE", fn)
    st = AgentState()
    out = stage.run(st, False, "res")
    assert out is st
    assert seen == [(st, "res")]


def test_registry_binds_all_eleven_phases():
    reg = build_stage_registry(_Stub())
    assert set(reg) == {
        "RESEARCH",
        "PLAN",
        "CODE",
        "LOCAL_SMOKE",
        "KERNEL_TRAIN",
        "VALIDATE_SUB",
        "TELEGRAM_APPROVE",
        "SUBMIT",
        "FEEDBACK",
        "REPORT",
        "HEAL",
    }
    assert [s.name for s in reg.values()] == list(reg)


def test_registry_dry_flags():
    reg = build_stage_registry(_Stub())
    assert reg["PLAN"].uses_dry is True
    assert reg["SUBMIT"].uses_dry is True
    assert reg["REPORT"].uses_dry is True
    assert reg["RESEARCH"].uses_dry is False
    assert reg["CODE"].uses_dry is False
    assert reg["HEAL"].uses_dry is False


def test_registry_unknown_phase_is_absent():
    reg = build_stage_registry(_Stub())
    assert reg.get("LOCK") is None
    assert reg.get("NOPE") is None