"""Start one daily agent cycle (used by Telegram /run and cron wrappers)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.paths import memory_dir, repo_root
from kaggle_agent.state_md import RunLock, load_state


@dataclass
class RunStartResult:
    ok: bool
    message: str
    dry_run: bool = True


def _load_dotenv(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def start_agent_cycle(
    *,
    root: Path | None = None,
    dry_run: bool = True,
    competition: str | None = None,
    background: bool = True,
) -> RunStartResult:
    """Launch one orchestrator cycle. background=True detaches a subprocess."""
    root = root or repo_root()
    _load_dotenv(root)

    state = load_state(root)
    if state.paused:
        return RunStartResult(
            ok=False,
            message="Agent is paused. Send /resume, then /run again.",
            dry_run=dry_run,
        )

    # Quick lock probe (orchestrator also locks)
    probe = RunLock(root)
    if not probe.acquire():
        return RunStartResult(
            ok=False,
            message="A cycle is already running. Check /status or wait for it to finish.",
            dry_run=dry_run,
        )
    probe.release()

    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        py = Path(sys.executable)
    cmd = [
        str(py),
        str(root / "scripts" / "run_daily.py"),
        "--competition",
        competition or "rsna_knee",
    ]
    if dry_run:
        cmd.append("--dry-run")
    else:
        cmd.append("--no-dry-run")

    log_dir = memory_dir(root) / "daily"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "manual_run.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    if not background:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n--- start dry={dry_run} ---\n")
            proc = subprocess.run(cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT)
        ok = proc.returncode == 0
        mode = "dry" if dry_run else "live"
        if ok:
            msg = (
                f"Cycle finished ({mode}).\n\n"
                f"Log: {log_path}\n\n"
                "If it was a live run, check the report: you may need to "
                "send /yes (approve) then /run live to submit."
            )
        else:
            msg = (
                f"Cycle finished with errors ({mode}, exit {proc.returncode}).\n\n"
                f"Log: {log_path}\n"
                "Send /status for current state."
            )
        return RunStartResult(ok=ok, message=msg, dry_run=dry_run)

    # Background: open log and detach
    log_f = log_path.open("a", encoding="utf-8")
    log_f.write(f"\n--- start dry={dry_run} background ---\n")
    log_f.flush()
    subprocess.Popen(
        cmd,
        cwd=str(root),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    mode = "dry (safe)" if dry_run else "live"
    return RunStartResult(
        ok=True,
        message=(
            f"Background cycle started ({mode}).\n\n"
            f"Log: {log_path}\n"
            "Use /status while it runs."
        ),
        dry_run=dry_run,
    )


def start_agent_cycle_async(
    *,
    root: Path | None = None,
    dry_run: bool = True,
    competition: str | None = None,
    on_done=None,
) -> RunStartResult:
    """Start in a daemon thread (keeps Telegram poll loop free)."""
    root = root or repo_root()
    state = load_state(root)
    if state.paused:
        return RunStartResult(
            ok=False,
            message="Agent is paused. Send /resume, then /run again.",
            dry_run=dry_run,
        )

    probe = RunLock(root)
    if not probe.acquire():
        return RunStartResult(
            ok=False,
            message="A cycle is already running. Check /status or wait for it to finish.",
            dry_run=dry_run,
        )
    probe.release()

    def worker() -> None:
        res = start_agent_cycle(
            root=root, dry_run=dry_run, competition=competition, background=False
        )
        if on_done:
            try:
                on_done(res)
            except Exception:
                pass

    threading.Thread(target=worker, name="kaggle-agent-run", daemon=True).start()
    mode = "dry (safe)" if dry_run else "live"
    return RunStartResult(
        ok=True,
        message=(
            f"Cycle started in the background ({mode}).\n\n"
            "You will get a full report when it finishes.\n"
            "Meanwhile: /status"
        ),
        dry_run=dry_run,
    )
