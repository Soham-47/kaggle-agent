"""Research must finish (Kaggle + papers + notebooks + rules) before PLAN."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.llm.router import ModelRouter
from kaggle_agent.notify.telegram import FakeTelegram
from kaggle_agent.orchestrator import Orchestrator, run_daily
from kaggle_agent.research.deep import DeepResearchResult
from kaggle_agent.state_md import AgentState, save_state


def _copy_min(root: Path, real: Path) -> None:
    import shutil

    shutil.copytree(real / "config", root / "config")
    shutil.copytree(real / "competitions", root / "competitions")
    (root / "memory").mkdir()
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        shutil.copy(real / "memory" / name, root / "memory" / name)
    (root / "memory" / "experiments").mkdir()
    (root / "memory" / "daily").mkdir()
    save_state(AgentState(paused=False, competition="rsna_knee"), root)


@dataclass
class RecordingRouter:
    """Stand-in ModelRouter: records plan prompts; exposes a dummy Zen client."""

    plan_calls: list[tuple[str, str]] = field(default_factory=list)
    client: object = field(default_factory=lambda: object())

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
    assert rec.plan_calls, "PLAN must call the router after research"
    user = rec.plan_calls[0][1]
    assert "## MEMORY.md" in user
    assert "## COMPETITION.md" in user
    assert "## research.md" in user
    assert "Deep research digest" in user
    assert "copyable next step" in user or "Must implement" in user
    assert "Baker" in user  # contest header rule lives in COMPETITION.md


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
