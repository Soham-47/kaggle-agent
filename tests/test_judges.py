"""Shared stage judges: verdict tracking, deterministic-first gates, LLM layer."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.judge import (
    judge_kernel,
    judge_plan,
    judge_stage,
    judge_train_llm,
    new_judge_state,
    record_verdict,
)
from kaggle_agent.pipeline.validate import ValidationResult
from kaggle_agent.research.source_cards import judge_cards_ready


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if not self.replies:
            return json.dumps({"ready": True, "reason": "default"})
        return json.dumps(self.replies.pop(0))


def _state() -> dict[str, object]:
    return new_judge_state()


def _plan_text(steps: str = "grouped 5-fold CV") -> str:
    return f"hypothesis: h\napproach: tune\nsteps: {steps}"


def test_record_verdict_streaks_identical_verdicts():
    st = _state()
    record_verdict(st, False, "a")
    record_verdict(st, False, "a")
    assert st["streak"] == 2
    record_verdict(st, True, "b")
    assert st["streak"] == 1
    assert st["ready"] is True
    assert st["last_reason"] == "b"


def test_judge_stage_deterministic_pass_then_llm():
    st = _state()
    lines: list[str] = []
    ready, reason = judge_stage(
        "plan",
        state=st,
        deterministic=lambda: (True, "det ok"),
        llm=lambda: (False, "llm says generic"),
        log=lines.append,
    )
    assert (ready, reason) == (False, "llm says generic")
    assert st["streak"] == 1
    assert st["ready"] is False
    assert lines == ["judge plan ready=False reason=llm says generic"]


def test_judge_stage_deterministic_fail_is_hard_floor():
    st = _state()
    llm_called = False

    def llm() -> tuple[bool, str]:
        nonlocal llm_called
        llm_called = True
        return (True, "llm ok")

    ready, reason = judge_stage(
        "plan",
        state=st,
        deterministic=lambda: (False, "det fail"),
        llm=llm,
    )
    assert (ready, reason) == (False, "det fail")
    assert llm_called is False
    assert st["last_reason"] == "det fail"
    assert st["ready"] is False


def test_judge_stage_without_llm_keeps_deterministic_verdict():
    st = _state()
    ready, reason = judge_stage(
        "plan",
        state=st,
        deterministic=lambda: (True, "det ok"),
    )
    assert (ready, reason) == (True, "det ok")
    assert st["ready"] is True


def test_judge_plan_rejects_implemented_steps():
    zen = _ScriptedZen([])
    ready, reason = judge_plan(
        zen, "m", _plan_text("rank-mean the outputs"), {"implement_steps": ["rank-mean the outputs"]}, "0.526"
    )
    assert ready is False
    assert "implemented" in reason
    assert zen.calls == 0


def test_judge_plan_deterministic_pass_without_zen():
    ready, reason = judge_plan(None, "m", _plan_text(), {}, "0.526")
    assert (ready, reason) == (True, "deterministic")


def test_judge_plan_llm_rejects_generic():
    zen = _ScriptedZen([{"ready": False, "reason": "generic tuning"}])
    ready, reason = judge_plan(zen, "m", _plan_text(), {}, "0.526")
    assert ready is False
    assert reason == "generic tuning"


def test_judge_plan_llm_accepts_novel():
    zen = _ScriptedZen([{"ready": True, "reason": "new backbone"}])
    ready, reason = judge_plan(zen, "m", _plan_text("swin-v2 backbone"), {}, "0.526")
    assert ready is True
    assert reason == "new backbone"


def test_judge_plan_fail_open_on_llm_error():
    class Boom:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            raise RuntimeError("boom")

    ready, reason = judge_plan(Boom(), "m", _plan_text(), {}, "0.526")
    assert ready is True
    assert "fail-open" in reason


def test_judge_kernel_rejects_failed_job():
    ok = ValidationResult(ok=True, path=Path("x.csv"))
    ready, reason = judge_kernel("error", ok)
    assert ready is False
    assert "error" in reason


def test_judge_kernel_rejects_bad_csv():
    ok = ValidationResult(ok=False, path=Path("x.csv"))
    ok.fail("header mismatch")
    ready, reason = judge_kernel("success", ok)
    assert ready is False
    assert "header mismatch" in reason


def test_judge_kernel_accepts_good_job_and_csv():
    ok = ValidationResult(ok=True, path=Path("x.csv"), n_rows=50)
    ready, reason = judge_kernel("success", ok)
    assert ready is True
    assert "rows=50" in reason


def test_judge_kernel_unknown_status_left_to_csv():
    ok = ValidationResult(ok=True, path=Path("x.csv"), n_rows=9)
    ready, reason = judge_kernel("none", ok)
    assert ready is True
    assert "rows=9" in reason


def test_judge_kernel_records_verdict_and_logs():
    st = _state()
    lines: list[str] = []
    ok = ValidationResult(ok=True, path=Path("x.csv"), n_rows=9)
    ready, _reason = judge_kernel("success", ok, state=st, log=lines.append)
    assert ready is True
    assert st["ready"] is True
    assert st["streak"] == 1
    assert lines == ["judge kernel ready=True reason=job success rows=9"]


def test_judge_train_llm_accepts_plausible(tmp_path: Path):
    zen = _ScriptedZen([{"ready": True, "reason": "outputs plausible"}])
    csv = tmp_path / "submission.csv"
    csv.write_text(
        "StudyInstanceUID,abnormal\ns1,0.9\ns2,0.1\ns3,0.5\n", encoding="utf-8"
    )
    st = _state()
    lines: list[str] = []
    ready, reason = judge_train_llm(
        zen, "m", "success", csv, ["abnormal"], "0.526", state=st, log=lines.append
    )
    assert ready is True
    assert reason == "outputs plausible"
    assert st["ready"] is True
    assert any("judge train ready=True" in line for line in lines)


def test_judge_train_llm_fail_open_on_error(tmp_path: Path):
    class Boom:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            raise RuntimeError("boom")

    csv = tmp_path / "submission.csv"
    csv.write_text("StudyInstanceUID,abnormal\ns1,0.9\n", encoding="utf-8")
    ready, reason = judge_train_llm(Boom(), "m", "success", csv, ["abnormal"], "0.526")
    assert ready is True
    assert "fail-open" in reason


def test_judge_train_llm_reports_nan_column(tmp_path: Path):
    seen: list[str] = []

    class _CapturingZen:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            seen.append(messages[-1]["content"])
            return json.dumps({"ready": True, "reason": "ok"})

    csv = tmp_path / "submission.csv"
    csv.write_text("StudyInstanceUID,abnormal\ns1,NaN\ns2,NaN\n", encoding="utf-8")
    judge_train_llm(_CapturingZen(), "m", "success", csv, ["abnormal"], "0.526")
    assert "nan=2" in seen[0]
    assert "constant=True" in seen[0]


def test_judge_train_llm_accepts_empty_labels(tmp_path: Path):
    csv = tmp_path / "submission.csv"
    csv.write_text("StudyInstanceUID\ns1\ns2\n", encoding="utf-8")
    ready, reason = judge_train_llm(None, "m", "success", csv, [], "0.526")
    assert ready is True
    assert reason == "judge-fail-open"


# ---------------------------------------------------------------------------
# Cards judge: unified with judge_stage — streak tracking, deterministic gate,
# fail-open all in the one seam.
# ---------------------------------------------------------------------------


def _card(path: Path, *, step: str = "attach public weights") -> Path:
    path.write_text(
        "# t\n"
        "- ref: u/r\n"
        f"- copyable next step: {step} Our score=0.526.\n"
        "- do not copy: H-flip\n",
        encoding="utf-8",
    )
    return path


def test_judge_cards_records_verdict_into_state(tmp_path: Path):
    st = _state()
    card = _card(tmp_path / "a.md")
    ready, reason = judge_cards_ready(None, "m", [card], "0.526", state=st)
    assert (ready, reason) == (True, "deterministic")
    assert st["ready"] is True
    assert st["streak"] == 1
    assert st["last_reason"] == "deterministic"


def test_judge_cards_no_actionable_step_rejects(tmp_path: Path):
    st = _state()
    card = _card(tmp_path / "a.md", step="use dataset/model refs")
    ready, reason = judge_cards_ready(None, "m", [card], "0.526", state=st)
    assert ready is False
    assert "no actionable step" in reason
    assert st["streak"] == 1
    assert st["ready"] is False


def test_judge_cards_empty_cards_reject():
    ready, reason = judge_cards_ready(None, "m", [], "0.526")
    assert (ready, reason) == (False, "no cards")


def test_judge_cards_llm_rejects_generic(tmp_path: Path):
    zen = _ScriptedZen([{"ready": False, "reason": "generic steps"}])
    card = _card(tmp_path / "a.md", step="improve the model")
    ready, reason = judge_cards_ready(zen, "m", [card], "0.526")
    assert ready is False
    assert reason == "generic steps"


def test_judge_cards_fail_open_on_llm_error(tmp_path: Path):
    class Boom:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            raise RuntimeError("boom")

    card = _card(tmp_path / "a.md")
    ready, reason = judge_cards_ready(Boom(), "m", [card], "0.526")
    assert ready is True
    assert "fail-open" in reason