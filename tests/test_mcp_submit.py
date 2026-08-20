"""MCP submit cascade unit tests (no real Kaggle write)."""

from __future__ import annotations

from pathlib import Path

from kaggle_agent.kaggle_api.mcp_submit import submit_via_mcp
from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref, split_kernel_ref
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import run_daily
from kaggle_agent.submit.pending import load_pending, request_approval, set_decision
from fakes import FakeKaggleApi, successful_kernel_train
from kaggle_agent.kaggle_api import KaggleClient


def _enable_mcp_in_settings(root: Path) -> None:
    path = root / "config" / "settings.yaml"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("mcp: false", "mcp: true"), encoding="utf-8")


def _disable_research_fleet(root: Path) -> None:
    path = root / "config" / "competitions" / "rsna_knee.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "fleet: [notebooks, papers, github, web, discussions, datasets]",
            "fleet: false",
        ),
        encoding="utf-8",
    )


def _write_live_submission(path: Path, header: list[str]) -> None:
    rows = "\n".join(
        f"study-{i}," + ",".join([str(0.5 + (i % 2) * 0.1)] * (len(header) - 1))
        for i in range(1000)
    )
    path.write_text(",".join(header) + "\n" + rows + "\n", encoding="utf-8")


def test_normalize_kernel_ref():
    assert normalize_kernel_ref("/code/user/slug") == "user/slug"
    assert normalize_kernel_ref("code/user/slug") == "user/slug"
    assert normalize_kernel_ref("https://www.kaggle.com/code/user/slug") == "user/slug"
    assert normalize_kernel_ref("user/slug") == "user/slug"
    assert split_kernel_ref("user/my-kernel") == ("user", "my-kernel")


def test_mcp_file_submit_injected(tmp_path: Path):
    csv = tmp_path / "submission.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    calls: list[str] = []

    def call(name: str, arguments: dict):
        calls.append(name)
        if name == "start_competition_submission_upload":
            return {"token": "tok123", "create_url": "https://example.invalid/upload"}
        if name == "submit_to_competition":
            return {"status": "ok"}
        raise AssertionError(name)

    # blob PUT will fail to example.invalid — inject full path by mocking upload:
    # exercise dry_run and code path instead
    r = submit_via_mcp(
        competition="titanic",
        message="m",
        mode="file",
        csv_path=csv,
        dry_run=True,
        call_tool=call,
    )
    assert r.success and r.dry_run
    assert not calls


def test_mcp_code_submit_injected():
    def call(name: str, arguments: dict):
        assert name == "create_code_competition_submission"
        req = arguments["request"]
        assert req["kernelOwner"] == "u"
        assert req["kernelSlug"] == "s"
        assert req["competitionName"] == "rsna-knee-abnormality-detection"
        return {"ref": 1}

    r = submit_via_mcp(
        competition="rsna-knee-abnormality-detection",
        message="m",
        mode="notebook",
        kernel_ref="/code/u/s",
        dry_run=False,
        call_tool=call,
    )
    assert r.success
    assert "mcp code submit ok" in r.message


def test_mcp_and_api_failure_never_falls_back_to_browser(tmp_path: Path, monkeypatch):
    import shutil

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    _disable_research_fleet(root)
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    from kaggle_agent.state_md import AgentState, save_state

    save_state(AgentState(paused=False, competition="rsna_knee"), root)
    settings_path = root / "config" / "settings.yaml"
    settings_path.write_text(
        settings_path.read_text(encoding="utf-8").replace("block_submit: true", "block_submit: false"),
        encoding="utf-8",
    )

    from kaggle_agent.config import load_competition

    comp = load_competition("rsna_knee", root)
    csv = root / "competitions" / "rsna_knee" / "submissions" / "pre.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = [comp.id_column, *comp.labels]
    _write_live_submission(csv, header)
    request_approval(
        exp_id="pre-exp",
        csv_path=str(csv),
        competition="rsna-knee-abnormality-detection",
        root=root,
        kernel_ref="user/rsna-agent-pre",
        kernel_path="none",
    )
    set_decision("latest", approved=True, root=root)
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.Orchestrator._kernel_train", successful_kernel_train(root)
    )

    def mcp_fail(name: str, arguments: dict):
        raise RuntimeError("mcp forced fail")

    def browser_ok(req):
        return SubmitResult(dry_run=False, message="browser ok", success=True)

    class FailApi(FakeKaggleApi):
        def competition_submit_code(self, *a, **k):
            raise RuntimeError("api submit_code fail")

        def competition_submit(self, *a, **k):
            raise RuntimeError("api file fail")

    r = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=KaggleClient(api=FailApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
        browser_submit=browser_ok,
        mcp_submit_fn=mcp_fail,
    )
    assert r.submit_ok is False
    assert r.submission_pending is False
    assert load_pending(root).status != "submitted"


def test_mcp_success_skips_api(tmp_path: Path, monkeypatch):
    import shutil

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    _disable_research_fleet(root)
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    from kaggle_agent.state_md import AgentState, save_state

    save_state(AgentState(paused=False, competition="rsna_knee"), root)
    _enable_mcp_in_settings(root)

    from kaggle_agent.config import load_competition

    comp = load_competition("rsna_knee", root)
    csv = root / "competitions" / "rsna_knee" / "submissions" / "pre.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = [comp.id_column, *comp.labels]
    _write_live_submission(csv, header)
    request_approval(
        exp_id="pre-exp",
        csv_path=str(csv),
        competition="rsna-knee-abnormality-detection",
        root=root,
        kernel_ref="user/k",
    )
    set_decision("latest", approved=True, root=root)
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.Orchestrator._kernel_train", successful_kernel_train(root)
    )

    submit_code_hits: list[str] = []

    class TrackApi(FakeKaggleApi):
        def competition_submit_code(self, *a, **k):
            submit_code_hits.append("submit_code")
            return super().competition_submit_code(*a, **k)

    def mcp_ok(name: str, arguments: dict):
        return {"ok": True}

    r = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=KaggleClient(api=TrackApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
        mcp_submit_fn=mcp_ok,
    )
    assert r.submit_ok is True
    assert r.submit_message and r.submit_message.startswith("mcp:")
    assert submit_code_hits == []


def test_live_submit_uses_api_not_mcp(tmp_path: Path, monkeypatch):
    import shutil

    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    _disable_research_fleet(root)
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    from kaggle_agent.state_md import AgentState, save_state

    save_state(AgentState(paused=False, competition="rsna_knee"), root)

    from kaggle_agent.config import load_competition, load_settings

    assert load_settings(root).mcp_submit is False
    assert load_settings(root).kernel_push is True

    comp = load_competition("rsna_knee", root)
    csv = root / "competitions" / "rsna_knee" / "submissions" / "pre.csv"
    csv.parent.mkdir(parents=True, exist_ok=True)
    header = [comp.id_column, *comp.labels]
    _write_live_submission(csv, header)
    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / "pre-exp"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel-metadata.json").write_text("{}", encoding="utf-8")
    request_approval(
        exp_id="pre-exp",
        csv_path=str(csv),
        competition="rsna-knee-abnormality-detection",
        root=root,
        kernel_ref="tester/fake-kernel",
        kernel_path=str(kernel_dir),
    )
    set_decision("latest", approved=True, root=root)
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.Orchestrator._kernel_train", successful_kernel_train(root)
    )

    mcp_hits: list[str] = []

    def mcp_must_not_run(name: str, arguments: dict):
        mcp_hits.append(name)
        return {"ok": True}

    api = FakeKaggleApi()
    r = run_daily(
        "rsna_knee",
        root=root,
        dry_run=False,
        kaggle=KaggleClient(api=api).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
        mcp_submit_fn=mcp_must_not_run,
    )
    assert r.submit_ok is True
    assert r.submit_message and r.submit_message.startswith("api:")
    assert mcp_hits == []
    kinds = [c[0] for c in api.submit_calls if isinstance(c, tuple) and c]
    assert "submit_code" in kinds
    assert "kernels_push" in kinds  # variant push (internet-off) + explicit version
    assert load_pending(root).status == "submitted"
