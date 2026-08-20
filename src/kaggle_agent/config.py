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


class ConfigError(ValueError):
    """Raised when a YAML configuration violates its runtime contract."""


class _Section(dict[str, Any]):
    """Mapping view that retains its dotted path for precise errors."""

    def __init__(self, values: dict[str, Any], config_path: str) -> None:
        super().__init__(values)
        self.config_path = config_path


def _config_error(path: Path, field: str, reason: str, value: Any) -> ConfigError:
    return ConfigError(f"{path}: {field} {reason}; got {value!r}")


def _section(raw: dict[str, Any], name: str, path: Path) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise _config_error(path, name, "must be a mapping", value)
    parent = getattr(raw, "config_path", "")
    dotted = f"{parent}.{name}" if parent else name
    return _Section(value, dotted)


def _field(section: dict[str, Any], key: str) -> str:
    parent = getattr(section, "config_path", "")
    return f"{parent}.{key}" if parent else key


def _bool(section: dict[str, Any], key: str, path: Path, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise _config_error(path, _field(section, key), "must be a boolean", value)
    return value


def _int(
    section: dict[str, Any], key: str, path: Path, default: int, *, minimum: int | None = None
) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _config_error(path, _field(section, key), "must be an integer", value)
    if minimum is not None and value < minimum:
        raise _config_error(path, _field(section, key), f"must be an integer >= {minimum}", value)
    return value


def _float(
    section: dict[str, Any], key: str, path: Path, default: float, *, minimum: float | None = None
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _config_error(path, _field(section, key), "must be a number", value)
    number = float(value)
    if minimum is not None and number < minimum:
        raise _config_error(path, _field(section, key), f"must be a number >= {minimum}", value)
    return number


def _str(section: dict[str, Any], key: str, path: Path, default: str, *, nonempty: bool = False) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        expected = "a non-empty string" if nonempty else "a string"
        raise _config_error(path, _field(section, key), f"must be {expected}", value)
    return value


def _list(section: dict[str, Any], key: str, path: Path, default: list[Any]) -> list[Any]:
    value = section.get(key, default)
    if not isinstance(value, list):
        raise _config_error(path, _field(section, key), "must be a list", value)
    return value


def _validate_agent(section: dict[str, Any], path: Path, *, minutes: float, turns: int) -> None:
    agent = _section(section, "agent", path)
    _float(agent, "max_minutes", path, minutes, minimum=0.001)
    _int(agent, "max_tool_turns", path, turns, minimum=1)
    _int(agent, "max_tokens", path, 2048, minimum=256)


def _validate_settings(raw: dict[str, Any], path: Path) -> None:
    default_competition = raw.get("default_competition", "rsna_knee")
    if not isinstance(default_competition, str) or not default_competition.strip():
        raise _config_error(path, "default_competition", "must be a non-empty string", default_competition)

    orchestrator = _section(raw, "orchestrator", path)
    _bool(orchestrator, "dry_run", path, True)
    phases = _list(orchestrator, "phases", path, list(DEFAULT_PHASES))
    if not phases or any(not isinstance(item, str) or not item.strip() for item in phases):
        raise _config_error(path, "orchestrator.phases", "must contain non-empty strings", phases)

    submit = _section(raw, "submit", path)
    _bool(submit, "require_telegram_approve", path, True)
    _bool(submit, "api", path, True)
    _bool(submit, "mcp", path, False)
    _int(submit, "max_proposals_per_day", path, 2, minimum=0)

    kernel = _section(raw, "kernel", path)
    _bool(kernel, "push", path, False)
    _bool(kernel, "enable_gpu", path, False)
    _bool(kernel, "enable_internet", path, False)
    _int(kernel, "poll_seconds", path, 30, minimum=1)
    _int(kernel, "poll_attempts", path, 40, minimum=1)
    if "machine_shape" in kernel and kernel["machine_shape"] is not None:
        _str(kernel, "machine_shape", path, "")
    if "username" in kernel and kernel["username"] is not None:
        _str(kernel, "username", path, "")

    research = _section(raw, "research", path)
    _int(research, "loop_passes", path, 3, minimum=1)
    _validate_agent(research, path, minutes=15.0, turns=40)
    fleet = _section(research, "fleet", path)
    _bool(fleet, "enabled", path, False)
    agents = _list(fleet, "agents", path, list(ResearchFleetSettings.agents))
    if any(not isinstance(item, str) or not item.strip() for item in agents):
        raise _config_error(path, "research.fleet.agents", "must contain non-empty strings", agents)
    _validate_agent(fleet, path, minutes=15.0, turns=24)
    deep = _section(research, "deep", path)
    _bool(deep, "enabled", path, True)
    for key, default in (("breadth", 3), ("depth", 2), ("max_queries", 12), ("per_query_limit", 5),
                         ("max_learnings", 4), ("max_followups", 3), ("max_fetches", 40)):
        _int(deep, key, path, default, minimum=1)
    _float(deep, "max_minutes", path, 15.0, minimum=0.001)
    _str(deep, "report_dir", path, "memory/research-deep", nonempty=True)
    _validate_agent(_section(raw, "plan", path), path, minutes=10.0, turns=20)
    _validate_agent(_section(raw, "code", path), path, minutes=10.0, turns=20)

    browser = _section(raw, "browser_research", path)
    _bool(browser, "enabled", path, True)
    _bool(browser, "prefer_browser_harness", path, True)
    pages = _list(browser, "pages", path, ["overview", "discussion"])
    if any(not isinstance(item, str) or not item.strip() for item in pages):
        raise _config_error(path, "browser_research.pages", "must contain non-empty strings", pages)
    _bool(_section(raw, "telegram", path), "enabled", path, False)
    _bool(_section(raw, "judges", path), "train", path, False)
    _bool(_section(raw, "eval", path), "block_submit", path, False)

    heal = _section(raw, "heal", path)
    _int(heal, "max_tune_attempts", path, 3, minimum=0)
    _int(heal, "max_no_improve_days", path, 5, minimum=0)
    feedback = _section(raw, "feedback", path)
    _int(feedback, "wait_minutes", path, 12, minimum=0)
    _int(feedback, "poll_seconds", path, 30, minimum=1)
    loop = _section(raw, "loop", path)
    n_min = _int(loop, "n_min", path, 2, minimum=1)
    n_max = _int(loop, "n_max", path, 8, minimum=1)
    default_n = _int(loop, "default_n", path, 3, minimum=1)
    _float(loop, "typical_gain", path, 0.01, minimum=0.0)
    _float(loop, "max_minutes", path, 90.0, minimum=0.001)
    if n_max < n_min:
        raise _config_error(path, "loop.n_max", "must be >= loop.n_min", n_max)
    if not n_min <= default_n <= n_max:
        raise _config_error(path, "loop.default_n", "must be between loop.n_min and loop.n_max", default_n)
    cron = _section(raw, "cron", path)
    hour = _int(cron, "hour_utc", path, 6, minimum=0)
    if hour > 23:
        raise _config_error(path, "cron.hour_utc", "must be between 0 and 23", hour)


def _validate_competition(raw: dict[str, Any], path: Path) -> None:
    for key in ("id", "slug"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _config_error(path, key, "must be a non-empty string", value)
    metric = _section(raw, "metric", path)
    direction = _str(metric, "direction", path, "max").lower()
    if direction not in {"min", "max"}:
        raise _config_error(path, "metric.direction", "must be min or max", direction)
    labels = raw.get("labels", [])
    if not isinstance(labels, list) or any(not isinstance(item, str) or not item.strip() for item in labels):
        raise _config_error(path, "labels", "must be a list of non-empty strings", labels)
    submission = _section(raw, "submission", path)
    _str(submission, "id_column", path, "StudyInstanceUID", nonempty=True)
    if "min_rows" in submission and submission["min_rows"] is not None:
        _int(submission, "min_rows", path, 0, minimum=0)
    probability_columns = _list(submission, "probability_columns", path, [])
    if any(not isinstance(item, str) or not item.strip() for item in probability_columns):
        raise _config_error(path, "submission.probability_columns", "must contain non-empty strings", probability_columns)
    workspace = _section(raw, "workspace", path)
    _str(workspace, "relative", path, f"competitions/{raw['id']}", nonempty=True)
    submit = _section(raw, "submit", path)
    mode = _str(submit, "mode", path, "file").lower()
    if mode not in {"file", "notebook"}:
        raise _config_error(path, "submit.mode", "must be one of {'file', 'notebook'}", mode)
    _str(submit, "output_file", path, "submission.csv", nonempty=True)
    models = _section(raw, "models", path)
    for role in ("plan", "code", "distill", "vision"):
        if role in models and models[role] is not None and not isinstance(models[role], str):
            raise _config_error(path, f"models.{role}", "must be a string or null", models[role])
    train = _section(raw, "train", path)
    _str(train, "backend", path, "kaggle_kernel", nonempty=True)
    _bool(train, "local_smoke_only", path, True)
    budget = _section(raw, "submit_budget", path)
    _int(budget, "max_proposals_per_day", path, 2, minimum=0)
    fleet = raw.get("research", {}).get("fleet") if isinstance(raw.get("research"), dict) else None
    if fleet is not None and fleet is not True and not isinstance(fleet, list):
        raise _config_error(path, "research.fleet", "must be true, false, or a roster list", fleet)
    if isinstance(fleet, list) and any(not isinstance(item, str) or not item.strip() for item in fleet):
        raise _config_error(path, "research.fleet", "must contain non-empty strings", fleet)


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
        """Return the only production provider currently supported."""
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
    def kernel_machine_shape(self) -> str | None:
        """Accelerator name (e.g. NvidiaTeslaT4). Overrides boolean GPU flags."""
        raw = self.raw.get("kernel", {}).get("machine_shape")
        return str(raw) if raw else None

    @property
    def kernel_enable_internet(self) -> bool:
        return bool(self.raw.get("kernel", {}).get("enable_internet", False))

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
        """Deprecated: browser submission is forbidden; always disabled."""
        return False

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
    def feedback_wait_minutes(self) -> int:
        """How long FEEDBACK polls the LB for the just-submitted score (0 = no wait)."""
        return max(0, int(self.raw.get("feedback", {}).get("wait_minutes", 12)))

    @property
    def feedback_poll_seconds(self) -> int:
        return max(5, int(self.raw.get("feedback", {}).get("poll_seconds", 30)))

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
    def submission_min_rows(self) -> int | None:
        raw = self.raw.get("submission", {}).get("min_rows")
        return int(raw) if raw is not None else None

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
    raw = _read_yaml(path)
    _validate_settings(raw, path)
    return Settings(raw=raw, root=root)


def load_competition(competition_id: str, root: Path | None = None) -> CompetitionConfig:
    root = root or repo_root()
    path = config_dir(root) / "competitions" / f"{competition_id}.yaml"
    raw = _read_yaml(path)
    _validate_competition(raw, path)
    return CompetitionConfig(raw=raw, path=path)
