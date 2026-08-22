"""Read-only startup health checks before a generation is resumed."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.autonomy.outbox import ExternalActionOutbox
from kaggle_agent.autonomy.runtime import StageLedger
from kaggle_agent.config import load_competition, load_settings


@dataclass(frozen=True)
class HealthResult:
    healthy: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...] = ()


def startup_health_check(root: Path, competition: str, *, state_root: Path | None = None) -> HealthResult:
    checks = []
    failures = []
    try:
        import kaggle_agent
        _ = kaggle_agent
        checks.append("import")
        load_settings(root)
        checks.append("settings")
        load_competition(competition, root)
        checks.append("competition")
        StageLedger(root, state_root=state_root).records()
        ExternalActionOutbox(root, state_root=state_root).pending()
        checks.append("runtime_state")
    except (OSError, ValueError, RuntimeError) as exc:
        failures.append(str(exc))
    return HealthResult(not failures, tuple(checks), tuple(failures))
