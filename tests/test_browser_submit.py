"""Browser submit fallback unit tests (no real browser)."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeKaggleApi, successful_kernel_train
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.state_md import load_state
from kaggle_agent.submit.browser_submit import BrowserSubmitRequest, submit_via_browser
from kaggle_agent.submit.pending import load_pending, request_approval, set_decision


def _copy_min(root: Path, real: Path) -> None:
    import shutil

    from kaggle_agent.state_md import AgentState, save_state

    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    save_state(AgentState(paused=False, competition="rsna_knee"), root)


def _fake_browser(url: str, max_chars: int = 12000) -> str:
    return "Overview knee MRI macro AUC discussion. " * 6


def _write_live_submission(path: Path, header: list[str]) -> None:
    rows = "\n".join(
        f"study-{i}," + ",".join([str(0.5 + (i % 2) * 0.1)] * (len(header) - 1))
        for i in range(1000)
    )
    path.write_text(",".join(header) + "\n" + rows + "\n", encoding="utf-8")


class _FailSubmitApi(FakeKaggleApi):
    """API that rejects competition_submit / submit_code."""

    def competition_submit(self, *args, **kwargs):
        raise RuntimeError("400 CreateSubmission forced")

    def competition_submit_code(self, *args, **kwargs):
        raise RuntimeError("submit_code forced fail")


def test_browser_submit_dry_run(tmp_path: Path):
    csv = tmp_path / "s.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    r = submit_via_browser(
        BrowserSubmitRequest(
            competition="rsna-knee",
            message="m",
            mode="file",
            csv_path=csv,
            dry_run=True,
        )
    )
    assert r.dry_run and r.success
    assert "would browser-submit" in r.message


def test_browser_submit_injected(tmp_path: Path):
    csv = tmp_path / "s.csv"
    csv.write_text("a\n1\n", encoding="utf-8")

    def fake(req: BrowserSubmitRequest) -> SubmitResult:
        assert req.competition == "rsna-knee"
        return SubmitResult(dry_run=False, message="ui ok", success=True)

    r = submit_via_browser(
        BrowserSubmitRequest(
            competition="rsna-knee",
            message="m",
            csv_path=csv,
            dry_run=False,
        ),
        run_fn=fake,
    )
    assert r.success and r.message == "ui ok"


def test_orchestrator_browser_fallback_on_api_fail(tmp_path: Path, monkeypatch):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    # Ensure fallback enabled in copied settings
    settings = root / "config" / "settings.yaml"
    text = settings.read_text(encoding="utf-8")
    if "browser_fallback" not in text:
        settings.write_text(
            text.replace(
                "require_telegram_approve: true",
                "require_telegram_approve: true\n  browser_fallback: true",
            ),
            encoding="utf-8",
        )

    # Pre-approve so live submit runs
    csv = root / "competitions" / "rsna_knee" / "submissions" / "pre.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid-ish CSV for validate path — cycle will make its own smoke
    request_approval(
        exp_id="pre-exp",
        csv_path=str(csv),
        competition="rsna-knee-abnormality-detection",
        root=root,
        kernel_path="none",
        kernel_ref="user/rsna-agent-pre",
    )
    set_decision("latest", approved=True, root=root)
    # Ensure approved csv exists so usable_approval / submit path can prefer it
    # Write after smoke may overwrite preference; give real header matching labels
    from kaggle_agent.config import load_competition

    comp = load_competition("rsna_knee", root)
    header = [comp.id_column, *comp.labels]
    _write_live_submission(csv, header)
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.Orchestrator._kernel_train", successful_kernel_train(root)
    )

    calls: list[str] = []

    def browser_ok(req: BrowserSubmitRequest) -> SubmitResult:
        calls.append(req.competition)
        return SubmitResult(dry_run=False, message="browser recovered", success=True)

    def mcp_fail(name: str, arguments: dict):
        raise RuntimeError("mcp forced fail")

    tg = FakeTelegram()
    client = KaggleClient(api=_FailSubmitApi()).connect()
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=client,
        browser_fetch=_fake_browser,
        telegram=tg,
        browser_submit=browser_ok,
        mcp_submit_fn=mcp_fail,
    )
    assert result.submit_ok is True
    assert calls, "browser fallback should run"
    assert "browser recovered" in (result.submit_message or "")
    assert load_pending(root).status == "submitted"
    st = load_state(root)
    assert st.lock_held is False
