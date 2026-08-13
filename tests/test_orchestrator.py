from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import DEFAULT_PHASES
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.state_md import AgentState, load_state, save_state


def _fake_kaggle() -> KaggleClient:
    return KaggleClient(api=FakeKaggleApi()).connect()


def _fake_browser(url: str, max_chars: int = 12000) -> str:
    return (
        "Competition overview: detect twelve knee abnormalities from multimodal MRI. "
        "Evaluation uses macro-averaged ROC AUC. "
        "Discussion tip: study-level 2D CNN baselines are strong starters. "
    ) * 2


def test_dry_run_cycle(tmp_path: Path):
    from kaggle_agent.notify.telegram import FakeTelegram

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    tg = FakeTelegram()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
        telegram=tg,
    )
    assert not result.skipped
    assert result.kaggle_ok is True
    assert result.browser_ok is True
    assert result.code_ok is True
    assert result.smoke_ok is True
    assert result.smoke_path
    assert Path(result.smoke_path).is_file()
    assert result.kernel_ok is True
    assert result.kernel_path
    assert (Path(result.kernel_path) / "kernel-metadata.json").is_file()
    assert result.validate_ok is True
    assert result.candidate_csv
    assert result.approve_ok is True
    assert result.submit_ok is True  # dry submit
    assert result.phases_run == list(DEFAULT_PHASES)
    st = load_state(root)
    assert st.phase == "IDLE"
    assert st.lock_held is False
    research = (root / "memory" / "research.md").read_text(encoding="utf-8")
    assert "Browser (read-only)" in research
    assert "allowed_now" in research or "Alpha" in research


def test_skip_when_paused(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    save_state(AgentState(paused=True), root)
    r = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
    )
    assert r.skipped and r.skip_reason == "paused"


def test_dropped_submit_phase_is_skipped(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    path = root / "config" / "settings.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("    - SUBMIT\n", ""), encoding="utf-8")
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=_fake_kaggle(),
        browser_fetch=_fake_browser,
    )
    assert "SUBMIT" not in result.phases_run
    assert "RESEARCH" in result.phases_run
    assert result.phases_run.index("RESEARCH") < result.phases_run.index("PLAN")
    assert "VALIDATE_SUB" in result.phases_run
    assert "REPORT" in result.phases_run
