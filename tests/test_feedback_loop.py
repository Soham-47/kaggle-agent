from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.heal.feedback import (
    already_recorded,
    exp_id_from_description,
    exp_public_score,
    first_scored,
)
from kaggle_agent.heal.policy import load_heal
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.kaggle_api.models import SubmissionRow
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.state_md import AgentState, load_state, save_state


def _orch(root: Path, api=None) -> Orchestrator:
    settings = load_settings(root)
    comp = load_competition("rsna_knee", root)
    client = KaggleClient(api=api or FakeKaggleApi()).connect()

    class _Router:
        def __init__(self, client) -> None:  # noqa: ANN001
            self.client = client

    return Orchestrator(settings, comp, root=root, router=_Router(client), kaggle=client)


def _setup(root: Path) -> Path:
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    (root / "memory" / "experiments").mkdir(parents=True, exist_ok=True)
    return root


def _row(ref: str, score: str, status: str = "complete", desc: str = "") -> SubmissionRow:
    return SubmissionRow(
        ref=ref, file_name="sub.csv", status=status, public_score=score,
        date="2026-08-01", description=desc,
    )


def _exp(root: Path, exp_id: str, score: str = "none") -> Path:
    path = root / "memory" / "experiments" / f"{exp_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {exp_id}\n\n- hypothesis: h\n- approach: tune\n- public_score: {score}\n",
        encoding="utf-8",
    )
    return path


# --- pure helpers ---


def test_first_scored_skips_pending_and_status_strings():
    rows = [
        _row("s1", "", status="pending"),
        _row("s2", "SubmissionStatus.PENDING", status="pending"),
        _row("s3", "0.526"),
    ]
    assert first_scored(rows) is rows[2]
    assert first_scored([_row("s1", "", status="pending")]) is None


def test_exp_id_from_description():
    assert exp_id_from_description("agent 20260817-090501") == "20260817-090501"
    assert exp_id_from_description("agent 20260817-090501-dry") == "20260817-090501-dry"
    assert exp_id_from_description("kaggle submit") is None
    assert exp_id_from_description("") is None


def test_exp_public_score_and_already_recorded(tmp_path: Path):
    exp = _exp(tmp_path, "20260817-090501", score="0.526")
    assert exp_public_score(exp) == "0.526"
    assert already_recorded(exp, "0.526")
    assert not already_recorded(exp, "0.520")
    exp2 = _exp(tmp_path, "20260817-090502", score="none")
    assert exp_public_score(exp2) is None
    assert not already_recorded(exp2, "0.526")


# --- feedback phase ---


def test_feedback_uses_scored_submission_not_pending_status(tmp_path: Path):
    root = _setup(tmp_path)
    api = FakeKaggleApi()

    class _Api(FakeKaggleApi):
        def competition_submissions(self, competition, **kwargs):
            return [
                SimpleNamespace(
                    ref="s-new", fileName="sub.csv", status="complete",
                    publicScore="0.527", date="2026-08-17",
                    description="agent 20260817-090501",
                )
            ]

    from types import SimpleNamespace

    class _Api2(_Api):
        pass

    orch = _orch(root, _Api2())
    orch.settings.raw["feedback"] = {"wait_minutes": 0, "poll_seconds": 5}
    _exp(root, "20260817-090501")
    result = CycleResult(
        competition="rsna_knee", dry_run=False, experiment_id="20260817-090501",
        submit_ok=True,
    )
    orch._feedback(AgentState(), dry=False, result=result)
    assert result.feedback_score == "0.527"
    assert "SubmissionStatus" not in str(result.feedback_score)
    exp_text = (root / "memory" / "experiments" / "20260817-090501.md").read_text(
        encoding="utf-8"
    )
    assert "- public_score: 0.527" in exp_text


def test_feedback_waits_for_pending_score(tmp_path: Path, monkeypatch):
    root = _setup(tmp_path)
    calls = {"n": 0}

    class _Api(FakeKaggleApi):
        def competition_submissions(self, competition, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return [
                    SimpleNamespace(
                        ref="s-new", fileName="sub.csv", status="pending",
                        publicScore="", date="2026-08-17",
                        description="agent 20260817-090501",
                    )
                ]
            return [
                SimpleNamespace(
                    ref="s-new", fileName="sub.csv", status="complete",
                    publicScore="0.527", date="2026-08-17",
                    description="agent 20260817-090501",
                )
            ]

    from types import SimpleNamespace

    monkeypatch.setattr("time.sleep", lambda s: None)
    orch = _orch(root, _Api())
    orch.settings.raw["feedback"] = {"wait_minutes": 5, "poll_seconds": 5}
    _exp(root, "20260817-090501")
    result = CycleResult(
        competition="rsna_knee", dry_run=False, experiment_id="20260817-090501",
        submit_ok=True,
    )
    orch._feedback(AgentState(), dry=False, result=result)
    assert result.feedback_score == "0.527"
    assert calls["n"] == 2


def test_feedback_gives_up_when_never_scored(tmp_path: Path, monkeypatch):
    root = _setup(tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)
    ticks = {"now": 1_000.0}
    monkeypatch.setattr(
        "time.monotonic", lambda: (ticks.__setitem__("now", ticks["now"] + 120.0) or ticks["now"])
    )

    class _Api(FakeKaggleApi):
        def competition_submissions(self, competition, **kwargs):
            return [
                SimpleNamespace(
                    ref="s-new", fileName="sub.csv", status="pending",
                    publicScore="", date="2026-08-17",
                    description="agent 20260817-090501",
                )
            ]

    from types import SimpleNamespace

    orch = _orch(root, _Api())
    orch.settings.raw["feedback"] = {"wait_minutes": 1, "poll_seconds": 5}
    _exp(root, "20260817-090501")
    result = CycleResult(
        competition="rsna_knee", dry_run=False, experiment_id="20260817-090501",
        submit_ok=True,
    )
    orch._feedback(AgentState(), dry=False, result=result)
    assert result.feedback_score is None


# --- catch-up at cycle start ---


def test_catch_up_ingests_late_scores_and_advances_heal(tmp_path: Path):
    root = _setup(tmp_path)
    _exp(root, "20260817-090501", score="none")

    class _Api(FakeKaggleApi):
        def competition_submissions(self, competition, **kwargs):
            return [
                SimpleNamespace(
                    ref="s1", fileName="sub.csv", status="complete",
                    publicScore="0.526", date="2026-08-17",
                    description="agent 20260817-090501",
                )
            ]

    from types import SimpleNamespace

    orch = _orch(root, _Api())
    state = load_state(root)
    orch._catch_up_scores(state)
    exp_text = (root / "memory" / "experiments" / "20260817-090501.md").read_text(
        encoding="utf-8"
    )
    assert "- public_score: 0.526" in exp_text
    heal = load_heal(root)
    assert heal.best_score == "0.526"
    assert heal.last_score == "0.526"
    assert heal.decision_next == "tune"


def test_catch_up_is_idempotent(tmp_path: Path):
    root = _setup(tmp_path)
    _exp(root, "20260817-090501", score="0.526")

    class _Api(FakeKaggleApi):
        def competition_submissions(self, competition, **kwargs):
            return [
                SimpleNamespace(
                    ref="s1", fileName="sub.csv", status="complete",
                    publicScore="0.526", date="2026-08-17",
                    description="agent 20260817-090501",
                )
            ]

    from types import SimpleNamespace

    orch = _orch(root, _Api())
    state = load_state(root)
    orch._catch_up_scores(state)
    orch._catch_up_scores(state)
    heal = load_heal(root)
    assert heal.no_improve_days == "0"
    exp_text = (root / "memory" / "experiments" / "20260817-090501.md").read_text(
        encoding="utf-8"
    )
    assert "- public_score: 0.526" in exp_text