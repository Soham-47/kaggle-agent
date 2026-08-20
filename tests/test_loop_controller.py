"""Repeat train slices N times, then submit the best candidate once."""

from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.loop import LoopState, save_loop
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import Orchestrator, run_daily
from kaggle_agent.config import load_competition, load_settings


def _fake_browser(url: str, max_chars: int = 12000) -> str:
    return "Overview knee MRI macro AUC discussion. " * 6


def _root_with_n(tmp_path: Path, n: int) -> Path:
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    save_loop(LoopState(next_n=str(n)), root)
    return root


def _code_subs(api: FakeKaggleApi) -> list:
    return [
        c
        for c in api.submit_calls
        if isinstance(c, tuple) and c and c[0] == "submit_code"
    ]


def test_next_n_two_trains_once_submit(tmp_path: Path):
    root = _root_with_n(tmp_path, 2)
    api = FakeKaggleApi()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        assume_approved=True,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=_fake_browser,
        telegram=FakeTelegram(),
    )
    assert result.phases_run.count("PLAN") == 2
    assert result.train_slices == 2
    assert result.research_passes >= 1
    notebooks = root / "competitions" / "rsna_knee" / "notebooks"
    pkgs = (
        [
            p
            for p in notebooks.iterdir()
            if p.is_dir() and "-s" in p.name and "-submit-offline" not in p.name
        ]
        if notebooks.is_dir()
        else []
    )
    assert len(pkgs) == 2
    assert result.submit_ok is True
    assert len(_code_subs(api)) == 1


def test_first_slice_fail_submits_second(tmp_path: Path, monkeypatch):
    root = _root_with_n(tmp_path, 2)
    hits = {"n": 0}
    orig = Orchestrator._validate_sub

    def flaky(self, state, result):  # noqa: ANN001
        hits["n"] += 1
        if hits["n"] == 1:
            result.validate_ok = False
            result.candidate_csv = None
            result.errors.append("validate: no candidate CSV")
            return state
        return orig(self, state, result)

    monkeypatch.setattr(Orchestrator, "_validate_sub", flaky)
    api = FakeKaggleApi()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        assume_approved=True,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=_fake_browser,
        telegram=FakeTelegram(),
    )
    assert hits["n"] == 2
    assert result.validate_ok is True
    assert result.submit_ok is True
    assert result.candidate_csv
    assert "-s2" in (result.experiment_id or "")
    assert "-s2" in Path(result.candidate_csv).name or "-s2" in (
        result.kernel_path or ""
    )
    assert len(_code_subs(api)) == 1


def test_both_slices_fail_skips_submit(tmp_path: Path, monkeypatch):
    root = _root_with_n(tmp_path, 2)

    def always_fail(self, state, result):  # noqa: ANN001
        result.validate_ok = False
        result.candidate_csv = None
        result.errors.append("validate: no candidate CSV")
        return state

    monkeypatch.setattr(Orchestrator, "_validate_sub", always_fail)
    api = FakeKaggleApi()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        assume_approved=True,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=_fake_browser,
        telegram=FakeTelegram(),
    )
    assert result.validate_ok is False
    assert result.submit_ok is False
    assert _code_subs(api) == []


def test_assume_approved_submits_once(tmp_path: Path):
    root = _root_with_n(tmp_path, 2)
    settings = load_settings(root)
    competition = load_competition("rsna_knee", root)
    api = FakeKaggleApi()
    result = Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=_fake_browser,
        telegram=FakeTelegram(),
    ).run_cycle(dry_run=False, assume_approved=True)
    assert result.submit_ok is True
    assert result.waiting_approve is False
    assert len(_code_subs(api)) == 1
