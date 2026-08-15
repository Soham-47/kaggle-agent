"""RESEARCH retries source cards until methods.json is implementable."""

from __future__ import annotations

import json
from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.research.deep import DeepResearchResult
from kaggle_agent.research.source_cards import cards_feasible


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if not self.replies:
            return json.dumps({"tool": "done", "args": {"reason": "empty script"}})
        return json.dumps(self.replies.pop(0))


def _fake_judge(monkeypatch, verdicts):  # noqa: ANN001
    """Patch orchestrator.judge_cards_ready; verdicts is callable or iterable.

    Honors the interface: the real judge records each verdict into ``state``
    (streak tracking lives in ``judge_stage``), so the double does the same.
    """
    from kaggle_agent.judge import record_verdict

    calls = {"n": 0}

    def fake(zen, model, cards, our, state=None, **kwargs):  # noqa: ANN001
        calls["n"] += 1
        if callable(verdicts):
            v = verdicts(calls["n"])
        else:
            seq = iter(verdicts)
            v = next(seq, (False, f"reason {calls['n']}"))
        if state is not None:
            record_verdict(state, *v)
        return v

    monkeypatch.setattr("kaggle_agent.orchestrator.judge_cards_ready", fake)
    return calls


class _ZenRouter:
    def __init__(self, zen) -> None:  # noqa: ANN001
        self.client = zen

    def available(self) -> bool:
        return True


def _daily_logs(root: Path) -> str:
    return "".join(
        p.read_text(encoding="utf-8")
        for p in (root / "memory" / "daily").glob("*.md")
    )


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
    zen=None,  # noqa: ANN001
) -> Orchestrator:
    settings = load_settings(root)
    settings.raw.setdefault("orchestrator", {})["phases"] = ["LOCK", "RESEARCH"]
    if loop_passes is not None:
        settings.raw.setdefault("research", {})["loop_passes"] = loop_passes
    if zen is not None:
        return Orchestrator(
            settings,
            load_competition("rsna_knee", root),
            root=root,
            kaggle=kaggle,
            browser_fetch=lambda u, m=12000: "overview " * 20,
            router=_ZenRouter(zen),
        )

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


def _stub_cards(monkeypatch) -> None:  # noqa: ANN001
    """Harvest writes cards without consuming scripted zen replies."""

    def card(*, title: str, ref: str, **kwargs) -> str:  # noqa: ANN001, ANN003
        return (
            f"# {title}\n"
            f"- ref: {ref}\n"
            f"- claimed_public: unknown\n"
            f"- backbone / input: see source\n"
            "- labels: see source\n"
            "- CV: prefer grouped splits\n"
            "- inference: discover hidden test IDs from study folders\n"
            "- copyable next step: attach public weights Our score=unknown.\n"
            "- do not copy: H-flip\n"
            f"- kind: {kwargs.get('kind') or 'kernel'}\n"
        )

    monkeypatch.setattr(
        "kaggle_agent.research.source_cards.card_from_source_llm", card
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
    assert "research fleet" in logs
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


def test_fleet_disabled_uses_sequential_research(tmp_path: Path, monkeypatch):
    _stub_deep(monkeypatch)
    root = _setup(tmp_path, drop_methods=True)
    settings = load_settings(root)
    settings.raw.setdefault("research", {}).setdefault("fleet", {})["enabled"] = False
    settings.raw.setdefault("orchestrator", {})["phases"] = ["LOCK", "RESEARCH"]
    competition = load_competition("rsna_knee", root)
    competition.raw.setdefault("research", {})["fleet"] = False

    class _NoZen:
        client = None

        def available(self) -> bool:
            return False

    result = Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=_EmptyKernelsApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        router=_NoZen(),
    ).run_cycle(dry_run=True)
    assert not result.skipped
    logs = "".join(
        p.read_text(encoding="utf-8")
        for p in (root / "memory" / "daily").glob("*.md")
    )
    assert "research agent stop=" in logs
    assert "research fleet" not in logs


def _judge_gate_orch(
    root: Path,
    zen,  # noqa: ANN001
    *,
    drop_methods: bool = True,
) -> Orchestrator:
    settings = load_settings(root)
    settings.raw.setdefault("research", {}).setdefault("fleet", {})["enabled"] = False
    settings.raw.setdefault("orchestrator", {})["phases"] = ["LOCK", "RESEARCH"]
    settings.raw.setdefault("browser_research", {})["enabled"] = False
    competition = load_competition("rsna_knee", root)
    competition.raw.setdefault("research", {})["fleet"] = False
    return Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        router=_ZenRouter(zen),
    )


def test_sequential_judge_rejects_until_convergence(tmp_path: Path, monkeypatch):
    """Two identical not-ready verdicts accept (convergence); the gate re-judges done."""
    _stub_deep(monkeypatch)
    _stub_cards(monkeypatch)
    calls = _fake_judge(monkeypatch, [(False, "generic steps"), (False, "generic steps")])
    root = _setup(tmp_path, drop_methods=True)
    zen = _ScriptedZen(
        [
            {"tool": "judge_cards"},
            {"tool": "done", "args": {"reason": "ok"}},
        ]
    )
    result = _judge_gate_orch(root, zen).run_cycle(dry_run=True)
    assert not result.skipped
    assert zen.calls == 2
    assert calls["n"] == 2
    logs = _daily_logs(root)
    assert "research agent stop=done" in logs
    assert "research judge" in logs


def test_sequential_judge_accepts_when_ready(tmp_path: Path, monkeypatch):
    _stub_deep(monkeypatch)
    _stub_cards(monkeypatch)
    _fake_judge(monkeypatch, [(True, "ready"), (True, "ready")])
    root = _setup(tmp_path, drop_methods=True)
    zen = _ScriptedZen(
        [
            {"tool": "judge_cards"},
            {"tool": "done", "args": {"reason": "ok"}},
        ]
    )
    result = _judge_gate_orch(root, zen).run_cycle(dry_run=True)
    assert not result.skipped
    logs = _daily_logs(root)
    assert "research agent stop=done" in logs
    assert "research judge" in logs


def test_sequential_done_without_judge_gets_judged_by_gate(tmp_path: Path, monkeypatch):
    """done before judge_cards: the gate judges each done; tool and gate share streak."""
    _stub_deep(monkeypatch)
    _stub_cards(monkeypatch)
    calls = _fake_judge(
        monkeypatch,
        [(False, "generic steps"), (False, "generic steps"), (False, "generic steps")],
    )
    root = _setup(tmp_path, drop_methods=True)
    zen = _ScriptedZen(
        [
            {"tool": "done", "args": {"reason": "first"}},
            {"tool": "judge_cards"},
            {"tool": "done", "args": {"reason": "second"}},
        ]
    )
    result = _judge_gate_orch(root, zen).run_cycle(dry_run=True)
    assert not result.skipped
    assert zen.calls == 3
    assert calls["n"] == 3
    logs = _daily_logs(root)
    assert "research agent stop=done" in logs
    assert "research judge" in logs


def test_fleet_polish_skipped_when_judge_ready(tmp_path: Path, monkeypatch):
    """Fleet ends with a ready verdict: no polish pass runs."""
    _stub_deep(monkeypatch)
    polish = _count_method(monkeypatch, "_fleet_polish")
    _fake_judge(monkeypatch, [(True, "ready")])
    root = _setup(tmp_path, drop_methods=True)
    result = _orch(
        root, KaggleClient(api=FakeKaggleApi()).connect()
    ).run_cycle(dry_run=True)
    assert not result.skipped
    assert polish["n"] == 0
    logs = _daily_logs(root)
    assert "research judge post-fleet" in logs


def test_fleet_polish_improves_cards_when_judge_not_ready(tmp_path: Path, monkeypatch):
    """Not-ready verdict triggers one polish agent; improved card is merged."""
    _stub_deep(monkeypatch)
    verdicts = [(False, "generic steps"), (True, "better")]
    calls = _fake_judge(monkeypatch, verdicts)
    root = _setup(tmp_path, drop_methods=True)
    zen = _ScriptedZen(
        [
            {"tool": "read_cards"},
            {"tool": "write_card", "args": {"ref": "u/polish", "markdown": "# x\n- copyable next step: attach u/weights\n- do not copy: H-flip\n"}},
            {"tool": "judge_cards"},
            {"tool": "done", "args": {"reason": "ok"}},
        ]
    )
    orch = _orch(root, KaggleClient(api=FakeKaggleApi()).connect(), zen=zen)
    orch._fleet_polish(CycleResult(competition="rsna_knee", dry_run=True), "generic steps")  # noqa: SLF001
    assert calls["n"] == 2
    logs = _daily_logs(root)
    assert "research polish stop=" in logs
    assert "research judge post-polish" in logs
    cards = list((root / "memory" / "research-deep").glob("source-polish-*.md"))
    assert cards
