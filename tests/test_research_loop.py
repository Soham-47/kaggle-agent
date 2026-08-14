"""RESEARCH retries source cards until methods.json is implementable."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import Orchestrator
from kaggle_agent.research.deep import DeepResearchResult
from kaggle_agent.research.source_cards import cards_feasible


class _EmptyKernelsApi(FakeKaggleApi):
    def kernels_list(self, **kwargs):  # noqa: ANN003
        return []


def _methods_path(root: Path) -> Path:
    return root / "competitions" / "rsna_knee" / "pipeline" / "methods.json"


def _setup(tmp_path: Path, *, drop_methods: bool = False) -> Path:
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    if drop_methods:
        path = _methods_path(root)
        if path.is_file():
            path.unlink()
    return root


def _orch(
    root: Path,
    kaggle: KaggleClient,
    *,
    loop_passes: int | None = None,
) -> Orchestrator:
    settings = load_settings(root)
    settings.raw.setdefault("orchestrator", {})["phases"] = ["LOCK", "RESEARCH"]
    if loop_passes is not None:
        settings.raw.setdefault("research", {})["loop_passes"] = loop_passes
    class _NoZen:
        client = None

        def available(self) -> bool:
            return False

    return Orchestrator(
        settings,
        load_competition("rsna_knee", root),
        root=root,
        kaggle=kaggle,
        browser_fetch=lambda u, m=12000: "overview " * 20,
        router=_NoZen(),
    )


def _count_method(monkeypatch, name: str) -> dict[str, int]:  # noqa: ANN001
    calls = {"n": 0}
    orig = getattr(Orchestrator, name)

    def wrapped(self, *args, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, name, wrapped)
    return calls


def _count_source_cards(monkeypatch) -> dict[str, int]:  # noqa: ANN001
    return _count_method(monkeypatch, "_source_cards")


def _stub_deep(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.DeepResearcher.run",
        lambda self, prompt, md: DeepResearchResult(
            learnings=[], sources=[], queries_run=0
        ),
    )


def test_cards_feasible_needs_sidecar_step_and_section(tmp_path: Path):
    workspace = tmp_path / "comp"
    (workspace / "pipeline").mkdir(parents=True)
    research = tmp_path / "research.md"
    assert cards_feasible(workspace, research) is False
    (workspace / "pipeline" / "methods.json").write_text("{}", encoding="utf-8")
    research.write_text("## Method cards\n", encoding="utf-8")
    assert cards_feasible(workspace, research) is False
    (workspace / "pipeline" / "methods.json").write_text(
        '{"implement_steps": ["attach public weights"], '
        '"dataset_sources": ["owner/public-weights"]}\n',
        encoding="utf-8",
    )
    research.write_text("notes only; no method or digest heading\n", encoding="utf-8")
    assert cards_feasible(workspace, research) is False
    research.write_text("## Method cards\n", encoding="utf-8")
    assert cards_feasible(workspace, research) is True
    research.write_text("## Deep research digest\n- x\n", encoding="utf-8")
    assert cards_feasible(workspace, research) is True


def test_one_kernel_one_pass_then_stop(tmp_path: Path, monkeypatch):
    _stub_deep(monkeypatch)
    calls = _count_source_cards(monkeypatch)
    root = _setup(tmp_path, drop_methods=True)
    result = _orch(
        root, KaggleClient(api=FakeKaggleApi()).connect()
    ).run_cycle(dry_run=True)
    assert not result.skipped
    assert calls["n"] == 1
    assert _methods_path(root).is_file()
    assert cards_feasible(
        root / "competitions" / "rsna_knee",
        root / "memory" / "research.md",
    )


def test_missing_methods_json_harvest_then_done(tmp_path: Path, monkeypatch):
    _stub_deep(monkeypatch)
    calls = _count_source_cards(monkeypatch)
    root = _setup(tmp_path, drop_methods=True)
    result = _orch(
        root,
        KaggleClient(api=_EmptyKernelsApi()).connect(),
    ).run_cycle(dry_run=True)
    assert not result.skipped
    assert not _methods_path(root).is_file()
    assert calls["n"] == 1
    logs = "".join(
        p.read_text(encoding="utf-8")
        for p in (root / "memory" / "daily").glob("*.md")
    )
    assert "research agent stop=" in logs
    assert "research cards still thin; continuing" in logs


def test_snapshot_and_browser_once_then_agent(tmp_path: Path, monkeypatch):
    _stub_deep(monkeypatch)
    snaps = _count_method(monkeypatch, "_kaggle_snapshot")
    browsers = _count_method(monkeypatch, "_browser_research")
    root = _setup(tmp_path, drop_methods=True)
    result = _orch(
        root,
        KaggleClient(api=_EmptyKernelsApi()).connect(),
    ).run_cycle(dry_run=True)
    assert not result.skipped
    assert result.kaggle_ok is True
    assert snaps["n"] == 1
    assert browsers["n"] == 1
