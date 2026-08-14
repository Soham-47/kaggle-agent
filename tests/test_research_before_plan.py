"""Research must finish (Kaggle + papers + notebooks + rules) before PLAN."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace as _copy_min
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import Orchestrator, run_daily
from kaggle_agent.research.deep import DeepResearchResult


class RecordingZen:
    """Records chat messages; one scripted tool per stage."""

    def __init__(self) -> None:
        self.users: list[str] = []
        self._n: dict[str, int] = {"research": 0, "plan": 0, "code": 0}

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        system = ""
        user = ""
        for m in messages:
            if m.get("role") == "system":
                system = str(m.get("content") or "")
            if m.get("role") == "user":
                user = str(m.get("content") or "")
        self.users.append(user)
        if "plan the next" in system:
            self._n["plan"] += 1
            if self._n["plan"] == 1:
                return json.dumps(
                    {
                        "tool": "write_plan",
                        "args": {
                            "hypothesis": "use grouped folds from public notebooks",
                            "approach": "tune",
                            "steps": "pull winner notebook; keep Baker's header",
                        },
                    }
                )
            return json.dumps({"tool": "done", "args": {}})
        if "coding agent" in system:
            self._n["code"] += 1
            if self._n["code"] == 1:
                return json.dumps(
                    {
                        "tool": "write_brief",
                        "args": {
                            "text": "attach listed datasets; discover test dirs; rank-mean"
                        },
                    }
                )
            return json.dumps({"tool": "done", "args": {}})
        self._n["research"] += 1
        if self._n["research"] == 1:
            return json.dumps({"tool": "deep_research", "args": {}})
        if self._n["research"] == 2:
            return json.dumps({"tool": "harvest_cards", "args": {}})
        return json.dumps({"tool": "done", "args": {}})


@dataclass
class RecordingRouter:
    """Stand-in ModelRouter: records plan/code agent chat."""

    plan_calls: list[tuple[str, str]] = field(default_factory=list)
    client: RecordingZen = field(default_factory=RecordingZen)

    def available(self) -> bool:
        return True

    def plan(self, system: str, user: str) -> str:
        self.plan_calls.append((system, user))
        return (
            "hypothesis: use grouped folds from public notebooks\n"
            "approach: tune\n"
            "steps: pull winner notebook; keep Baker's header"
        )

    def code(self, system: str, user: str) -> str:
        return "attach listed datasets; discover test dirs; rank-mean"


def test_research_phase_runs_before_plan(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    result = run_daily(
        "rsna_knee",
        root=root,
        dry_run=True,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
    )
    assert "RESEARCH" in result.phases_run
    assert "PLAN" in result.phases_run
    assert result.phases_run.index("RESEARCH") < result.phases_run.index("PLAN")
    assert result.phases_run.index("PLAN") < result.phases_run.index("CODE")
    assert result.phases_run.index("CODE") < result.phases_run.index("SUBMIT")


def test_plan_prompt_includes_memory_competition_and_research(tmp_path: Path, monkeypatch):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    research = root / "memory" / "research.md"
    research.write_text(
        research.read_text(encoding="utf-8")
        + "\n## Deep research digest\n"
        "- paper: 2.5D MRI CNN (arXiv:1234.5678)\n"
        "- notebook: winner EfficientNet study-level\n",
        encoding="utf-8",
    )

    def fake_run(self, prompt, research_md):  # noqa: ANN001
        text = research_md.read_text(encoding="utf-8") if research_md.is_file() else ""
        if "## Deep research digest" not in text:
            research_md.write_text(
                text + "\n## Deep research digest\n- paper: 2.5D MRI CNN\n",
                encoding="utf-8",
            )
        return DeepResearchResult(
            learnings=["2.5D MRI CNN", "EfficientNet study-level"],
            sources=["https://arxiv.org/abs/1234.5678", "https://www.kaggle.com/code/u/nb"],
            queries_run=3,
        )

    monkeypatch.setattr("kaggle_agent.orchestrator.DeepResearcher.run", fake_run)

    rec = RecordingRouter()
    settings = load_settings(root)
    competition = load_competition("rsna_knee", root)
    orch = Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
        router=rec,
    )
    result = orch.run_cycle(dry_run=True)
    assert result.deep_ok is True
    users = "\n".join(rec.client.users)
    assert rec.client.users, "PLAN/CODE must chat after research"
    assert "## MEMORY.md" in users
    assert "## COMPETITION.md" in users
    assert "## research.md" in users
    assert "Deep research digest" in users
    assert "copyable next step" in users or "Must implement" in users
    assert "Baker" in users  # contest header rule lives in COMPETITION.md


def test_deep_research_wires_kaggle_arxiv_github_web(tmp_path: Path, monkeypatch):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    kinds_seen: list[list[str]] = []

    orig = None

    def capturing_init(self, client, model, config, sources, root, **kwargs):  # noqa: ANN001
        kinds_seen.append([getattr(s, "kind", type(s).__name__) for s in sources])
        return orig(self, client, model, config, sources, root, **kwargs)

    import kaggle_agent.research.deep as deep_mod

    orig = deep_mod.DeepResearcher.__init__
    monkeypatch.setattr(deep_mod.DeepResearcher, "__init__", capturing_init)
    monkeypatch.setattr(
        "kaggle_agent.orchestrator.DeepResearcher.run",
        lambda self, prompt, md: DeepResearchResult(learnings=["x"], sources=["s"], queries_run=1),
    )

    rec = RecordingRouter()
    settings = load_settings(root)
    competition = load_competition("rsna_knee", root)
    Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        browser_fetch=lambda u, m=12000: "overview " * 20,
        telegram=FakeTelegram(),
        router=rec,
    ).run_cycle(dry_run=True)

    assert kinds_seen, "DeepResearcher must be constructed during RESEARCH"
    kinds = kinds_seen[0]
    for need in ("kaggle", "arxiv", "github", "web"):
        assert need in kinds, f"missing source {need}: {kinds}"


def test_deep_prompt_uses_active_competition_not_hardcoded_host(tmp_path: Path):
    root = tmp_path / "kaggle-agent"
    real = Path(__file__).resolve().parents[1]
    _copy_min(root, real)
    rec = RecordingRouter()
    settings = load_settings(root)
    competition = load_competition("rsna_knee", root)
    orch = Orchestrator(
        settings,
        competition,
        root=root,
        kaggle=KaggleClient(api=FakeKaggleApi()).connect(),
        router=rec,
    )
    prompt = orch._deep_prompt()
    assert competition.slug in prompt
    assert "site:kaggle.com/code" in prompt
    assert "implementable" in prompt.lower() or "coding agent" in prompt.lower()
