"""Load settings.yaml and competition YAML configs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.paths import config_dir, repo_root

DEFAULT_PHASES = (
    "LOCK",
    "RESEARCH",
    "PLAN",
    "CODE",
    "LOCAL_SMOKE",
    "KERNEL_TRAIN",
    "VALIDATE_SUB",
    "TELEGRAM_APPROVE",
    "SUBMIT",
    "FEEDBACK",
    "HEAL",
    "REPORT",
)


@dataclass(frozen=True)
class ResearchAgentSettings:
    max_minutes: float = 15.0
    max_tool_turns: int = 40
    max_tokens: int = 2048


@dataclass(frozen=True)
class ResearchFleetSettings:
    enabled: bool = False
    agents: tuple[str, ...] = (
        "notebooks",
        "papers",
        "github",
        "web",
        "discussions",
        "datasets",
    )
    max_minutes: float = 15.0
    max_tool_turns: int = 24
    max_tokens: int = 2048


@dataclass(frozen=True)
class DeepResearchSettings:
    enabled: bool = True
    breadth: int = 3
    depth: int = 2
    max_queries: int = 12
    per_query_limit: int = 5
    max_learnings: int = 4
    max_followups: int = 3
    max_fetches: int = 40
    max_minutes: float = 15.0
    report_dir: str = "memory/research-deep"


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    root: Path

    @property
    def default_competition(self) -> str:
        return str(self.raw.get("default_competition", "rsna_knee"))

    @property
    def dry_run(self) -> bool:
        return bool(self.raw.get("orchestrator", {}).get("dry_run", True))

    @property
    def phases(self) -> list[str]:
        phases = self.raw.get("orchestrator", {}).get("phases")
        return [str(p) for p in (phases or DEFAULT_PHASES)]

    @property
    def max_proposals_per_day(self) -> int:
        return int(self.raw.get("submit", {}).get("max_proposals_per_day", 2))

    def llm_provider(self) -> str:
        raw = str((self.raw.get("llm") or {}).get("provider") or "deepseek").lower()
        return "deepseek"

    def zen_model(self, role: str) -> str:
        block = self.raw.get("deepseek") or {}
        return str(block.get(f"default_{role}_model", "deepseek-v4-flash"))

    @property
    def browser_research_enabled(self) -> bool:
        return bool(self.raw.get("browser_research", {}).get("enabled", True))

    @property
    def browser_prefer_harness(self) -> bool:
        return bool(self.raw.get("browser_research", {}).get("prefer_browser_harness", True))

    @property
    def browser_pages(self) -> list[str]:
        pages = self.raw.get("browser_research", {}).get("pages") or ["overview", "discussion"]
        return [str(p) for p in pages]

    @property
    def research_loop_passes(self) -> int:
        raw = (self.raw.get("research") or {}).get("loop_passes", 3)
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 3
        return max(1, n)

    def _agent_config(self, section: str, *, minutes: float = 15, turns: int = 40) -> ResearchAgentSettings:
        raw = (self.raw.get(section) or {}).get("agent") or {}
        return ResearchAgentSettings(
            max_minutes=float(raw.get("max_minutes", minutes)),
            max_tool_turns=max(1, int(raw.get("max_tool_turns", turns))),
            max_tokens=max(256, int(raw.get("max_tokens", 2048))),
        )

    def research_agent_config(self) -> ResearchAgentSettings:
        return self._agent_config("research", minutes=15, turns=40)

    def research_fleet_config(self) -> ResearchFleetSettings:
        fleet = self.raw.get("research", {}).get("fleet") or {}
        agent = fleet.get("agent") or {}
        agents = fleet.get("agents") or ResearchFleetSettings.agents
        return ResearchFleetSettings(
            enabled=bool(fleet.get("enabled", False)),
            agents=tuple(str(a) for a in agents),
            max_minutes=float(agent.get("max_minutes", 15.0)),
            max_tool_turns=max(1, int(agent.get("max_tool_turns", 24))),
            max_tokens=max(256, int(agent.get("max_tokens", 2048))),
        )

    def plan_agent_config(self) -> ResearchAgentSettings:
        return self._agent_config("plan", minutes=10, turns=20)

    def code_agent_config(self) -> ResearchAgentSettings:
        return self._agent_config("code", minutes=10, turns=20)

    def deep_research_config(self) -> "DeepResearchSettings":
        deep = self.raw.get("research", {}).get("deep", {}) or {}

        def _num(key: str, default: int | float) -> int | float:
            return deep.get(key, default)

        return DeepResearchSettings(
            enabled=bool(deep.get("enabled", True)),
            breadth=int(_num("breadth", 3)),
            depth=int(_num("depth", 2)),
            max_queries=int(_num("max_queries", 12)),
            per_query_limit=int(_num("per_query_limit", 5)),
            max_learnings=int(_num("max_learnings", 4)),
            max_followups=int(_num("max_followups", 3)),
            max_fetches=int(_num("max_fetches", 40)),
            max_minutes=float(_num("max_minutes", 15.0)),
            report_dir=str(deep.get("report_dir", "memory/research-deep")),
        )

    @property
    def kernel_push(self) -> bool:
        """Push notebook to Kaggle (only when orchestrator dry_run is false)."""
        return bool(self.raw.get("kernel", {}).get("push", False))

    @property
    def kernel_enable_gpu(self) -> bool:
        return bool(self.raw.get("kernel", {}).get("enable_gpu", False))

    @property
    def kernel_poll_seconds(self) -> int:
        return max(5, int(self.raw.get("kernel", {}).get("poll_seconds", 30)))

    @property
    def kernel_poll_attempts(self) -> int:
        return max(1, int(self.raw.get("kernel", {}).get("poll_attempts", 40)))

    @property
    def kernel_username(self) -> str | None:
        raw = self.raw.get("kernel", {}).get("username")
        return str(raw) if raw else None

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.raw.get("telegram", {}).get("enabled", False))

    @property
    def require_telegram_approve(self) -> bool:
        return bool(self.raw.get("submit", {}).get("require_telegram_approve", True))

    @property
    def judge_train(self) -> bool:
        """LLM-judge TRAIN/SUBMIT output behind a flag (mechanical checks always run)."""
        return bool((self.raw.get("judges") or {}).get("train", False))

    @property
    def browser_submit_fallback(self) -> bool:
        """If MCP+API submit fail, try browser-harness UI (needs logged-in Chrome)."""
        return bool(self.raw.get("submit", {}).get("browser_fallback", True))

    @property
    def mcp_submit(self) -> bool:
        """Try Kaggle official MCP submit first. Off by default (notebook comps)."""
        return bool(self.raw.get("submit", {}).get("mcp", False))

    @property
    def api_submit(self) -> bool:
        """Try Kaggle Python API submit (after MCP if enabled)."""
        return bool(self.raw.get("submit", {}).get("api", True))

    @property
    def max_tune_attempts(self) -> int:
        return int(self.raw.get("heal", {}).get("max_tune_attempts", 3))

    @property
    def max_no_improve_days(self) -> int:
        return int(self.raw.get("heal", {}).get("max_no_improve_days", 5))

    @property
    def loop_n_min(self) -> int:
        return int(self.raw.get("loop", {}).get("n_min", 2))

    @property
    def loop_n_max(self) -> int:
        return int(self.raw.get("loop", {}).get("n_max", 8))

    @property
    def loop_typical_gain(self) -> float:
        return float(self.raw.get("loop", {}).get("typical_gain", 0.01))

    @property
    def loop_default_n(self) -> int:
        return int(self.raw.get("loop", {}).get("default_n", 3))

    @property
    def loop_max_minutes(self) -> float:
        return float(self.raw.get("loop", {}).get("max_minutes", 90))

    @property
    def block_submit(self) -> bool:
        return bool(self.raw.get("eval", {}).get("block_submit", False))

    @property
    def cron_hour(self) -> int:
        return int(self.raw.get("cron", {}).get("hour_utc", 6))


@dataclass(frozen=True)
class CompetitionConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def slug(self) -> str:
        return str(self.raw["slug"])

    @property
    def labels(self) -> list[str]:
        return [str(x) for x in self.raw.get("labels", [])]

    @property
    def metric_direction(self) -> str:
        return str(self.raw.get("metric", {}).get("direction", "max"))

    @property
    def id_column(self) -> str:
        return str(self.raw.get("submission", {}).get("id_column", "StudyInstanceUID"))

    @property
    def workspace_relative(self) -> str:
        return str(
            self.raw.get("workspace", {}).get("relative", f"competitions/{self.id}")
        )

    @property
    def submit_mode(self) -> str:
        """file = CSV upload API; notebook = kernels-only submit_code."""
        return str(self.raw.get("submit", {}).get("mode", "file")).lower()

    @property
    def submit_output_file(self) -> str:
        return str(self.raw.get("submit", {}).get("output_file", "submission.csv"))

    @property
    def fleet_enabled(self) -> bool:
        """True when research.fleet is `true` (default roster) or a roster list."""
        raw = self.raw.get("research", {}).get("fleet")
        if isinstance(raw, list):
            return bool(raw)
        return raw is True

    @property
    def fleet_agents(self) -> list[str]:
        raw = self.raw.get("research", {}).get("fleet")
        if isinstance(raw, list):
            return [str(a) for a in raw]
        return []

    def model_for(self, role: str, settings: Settings) -> str:
        models = self.raw.get("models") or {}
        value = models.get(role)
        return str(value) if value else settings.zen_model(role)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing config: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def load_dotenv(root: Path | None = None) -> None:
    """Load KEY=value from repo .env without overwriting a set env var."""
    path = (root or repo_root()) / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def load_settings(root: Path | None = None) -> Settings:
    root = root or repo_root()
    path = config_dir(root) / "settings.yaml"
    return Settings(raw=_read_yaml(path), root=root)


def load_competition(competition_id: str, root: Path | None = None) -> CompetitionConfig:
    root = root or repo_root()
    path = config_dir(root) / "competitions" / f"{competition_id}.yaml"
    return CompetitionConfig(raw=_read_yaml(path), path=path)
