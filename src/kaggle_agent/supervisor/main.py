"""Command-line supervisor entrypoint."""

from __future__ import annotations

from pathlib import Path

from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.loop import Supervisor


def run_supervisor(
    root: Path,
    *,
    competition: str | None = None,
    wait: bool = True,
    mode: str | None = None,
    profile: str | None = None,
) -> int:
    settings = load_settings(root, profile=profile)
    if mode is not None:
        raw = dict(settings.raw)
        section = dict(raw.get("supervisor") or {})
        section["mode"] = mode
        section["enabled"] = mode != "off"
        raw["supervisor"] = section
        from kaggle_agent.config import Settings

        settings = Settings(raw=raw, root=root)
    result = Supervisor(settings, root).run_once(competition=competition, wait=wait)
    print(f"supervisor={result.status} worker={result.worker_id or 'none'} reason={result.reason}")
    return 0 if result.status in {"OFF", "WORKER_STARTED", "SUCCESS", "SPEC_READY", "CANDIDATE_ACCEPTED", "PAUSED", "ADOPTED"} else 1
