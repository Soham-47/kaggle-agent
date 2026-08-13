from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.notify.commands import handle_command, process_updates
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.state_md import load_state, save_state, AgentState
from kaggle_agent.submit.pending import load_pending, request_approval


def _fake_browser(url: str, max_chars: int = 12000) -> str:
    return ("Overview text about knee MRI abnormalities and macro AUC metric. " * 4)


def test_commands_approve_reject(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    request_approval(
        exp_id="exp1",
        csv_path="/tmp/a.csv",
        competition="rsna",
        root=tmp_path,
    )
    r = handle_command("/approve exp1", root=tmp_path)
    assert r.ok
    assert load_pending(tmp_path).status == "approved"

    request_approval(
        exp_id="exp2",
        csv_path="/tmp/b.csv",
        competition="rsna",
        root=tmp_path,
    )
    r2 = handle_command("/reject latest", root=tmp_path)
    assert r2.ok
    assert load_pending(tmp_path).status == "rejected"


def test_pause_resume_budget(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    assert handle_command("/pause", root=tmp_path).ok
    assert load_state(tmp_path).paused is True
    assert handle_command("/resume", root=tmp_path).ok
    assert load_state(tmp_path).paused is False
    assert handle_command("/budget 3", root=tmp_path).ok
    assert load_state(tmp_path).max_proposals == "3"


def test_run_command_flags(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    dry = handle_command("/run", root=tmp_path)
    assert dry.ok and dry.start_cycle and dry.cycle_dry_run is True
    live = handle_command("/run live", root=tmp_path)
    assert live.ok and live.start_cycle and live.cycle_dry_run is False
    bad = handle_command("/run please", root=tmp_path)
    assert not bad.ok


def test_process_updates_filters_chat(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    updates = [
        {
            "update_id": 1,
            "message": {"chat": {"id": 999}, "text": "/status"},
        }
    ]
    out = process_updates(updates, root=tmp_path, allowed_chat_id="1")
    assert out and out[0].ok is False


def test_dry_cycle_approve_submit_with_fake_telegram(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    tg = FakeTelegram()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    assert result.approve_ok is True
    assert result.submit_ok is True
    assert result.candidate_csv
    pending = load_pending(root)
    assert pending.status == "pending"
    assert pending.exp_id == result.experiment_id
    assert tg.sent  # approve + report at least
    assert any(
        "Approval needed" in m or "Cycle finished" in m or "Done" in m for m in tg.sent
    )


def test_live_submit_blocked_without_approve(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    # force require approve + non-dry via settings edit
    settings = (root / "config" / "settings.yaml").read_text(encoding="utf-8")
    settings = settings.replace("dry_run: true", "dry_run: false")
    settings = settings.replace("enabled: false", "enabled: true")
    (root / "config" / "settings.yaml").write_text(settings, encoding="utf-8")

    tg = FakeTelegram()
    api = FakeKaggleApi()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    # Without /approve, real submit must not call competition_submit
    assert result.submit_ok is False
    assert result.waiting_approve is True
    assert result.hard_errors == []
    # competition_submit records 3-tuples (file, message, competition)
    real_submits = [c for c in api.submit_calls if isinstance(c, tuple) and len(c) == 3]
    assert real_submits == []



def test_approve_then_second_live_submits(tmp_path: Path):
    """/yes then /run live must not ask for approval again; it should submit."""
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    settings = (root / "config" / "settings.yaml").read_text(encoding="utf-8")
    settings = settings.replace("dry_run: true", "dry_run: false")
    (root / "config" / "settings.yaml").write_text(settings, encoding="utf-8")

    tg = FakeTelegram()
    api = FakeKaggleApi()
    kaggle = KaggleClient(api=api).connect()

    r1 = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=kaggle,
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    assert r1.waiting_approve is True
    exp = r1.experiment_id
    assert exp
    handle_command("/yes", root=root)
    assert load_pending(root).status == "approved"
    approved_csv = load_pending(root).csv_path

    r2 = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=kaggle,
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    assert r2.waiting_approve is False
    assert r2.submit_ok is True, r2.errors
    assert load_pending(root).status == "submitted"
    # Notebook-only comps use submit_code after kernels_push
    code_subs = [
        c
        for c in api.submit_calls
        if isinstance(c, tuple) and c and c[0] == "submit_code"
    ]
    file_subs = [
        c for c in api.submit_calls if isinstance(c, tuple) and len(c) == 3
    ]
    assert code_subs or file_subs
    assert any("previous /yes" in m or "Using your previous" in m for m in tg.sent)
