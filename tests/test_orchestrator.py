from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.state_md import AgentState, load_state, save_state


def _copy_min(root: Path, real: Path) -> None:
    import shutil

    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    # Kernel package needs real IDs; /data is gitignored and not in worktrees.
    (root / "data").mkdir()
    (root / "data" / "sample_submission.csv").write_text(
        "StudyInstanceUID\ns1\ns2\n", encoding="utf-8"
    )
    # Real state may be paused; tests need a clean agent
    save_state(AgentState(paused=False, competition="rsna_knee"), root)


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
    assert "PLAN" in result.phases_run
    assert "LOCAL_SMOKE" in result.phases_run
    assert "KERNEL_TRAIN" in result.phases_run
    assert "VALIDATE_SUB" in result.phases_run
    assert "TELEGRAM_APPROVE" in result.phases_run
    assert "SUBMIT" in result.phases_run
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
