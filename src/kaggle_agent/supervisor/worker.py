"""Managed worker process entrypoint and launcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore, HeartbeatThread
from kaggle_agent.supervisor.protocol import WorkerExit, WorkerRequest, WorkerResult
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.incidents import Incident, IncidentStore


class WorkerLauncher:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout

    def start(self, request: WorkerRequest, *, cwd: Path | None = None) -> subprocess.Popen[str]:
        store = SupervisorStateStore(self.layout)
        request_path = store.write_json(f"workers/{request.worker_id}/request.json", request.to_dict())
        environment = os.environ.copy()
        environment["KAGGLE_AGENT_SUPERVISOR_DIR"] = str(self.layout.state_root)
        environment["KAGGLE_AGENT_STATE_ROOT"] = str(self.layout.state_root)
        command = [sys.executable, "-m", "kaggle_agent.supervisor.worker", "--request", str(request_path)]
        return subprocess.Popen(command, cwd=str(cwd or self.layout.code_root), env=environment, text=True)

    def terminate_hung(self, process: subprocess.Popen[str], grace_seconds: float) -> None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace_seconds)

    def monitor(self, process: subprocess.Popen[str], worker_id: str, *, timeout_seconds: float, grace_seconds: float) -> bool:
        """Return True when a stale heartbeat caused bounded termination."""
        heartbeat = HeartbeatStore(self.layout.state_root)
        if heartbeat.is_fresh(worker_id, timeout_seconds=timeout_seconds):
            return False
        self.terminate_hung(process, grace_seconds)
        return True


def run_worker(request_path: Path) -> int:
    request = WorkerRequest.from_dict(json.loads(request_path.read_text(encoding="utf-8")))
    layout = RuntimeLayout.for_repo(Path.cwd(), Path(os.environ["KAGGLE_AGENT_SUPERVISOR_DIR"]))
    store = SupervisorStateStore(layout)
    revision = request.revision or RuntimeRevision("", "", request.generation_id)
    heartbeat = HeartbeatStore(layout.state_root)
    beat = HeartbeatThread(heartbeat, Heartbeat(request.worker_id, os.getpid(), request.generation_id, request.cycle_id, request.resume_from_stage, "started", time.time()), 30)
    beat.start()
    try:
        from kaggle_agent.config import load_settings
        from kaggle_agent.orchestrator import run_daily

        result = run_daily(
            request.competition,
            dry_run=load_settings(Path.cwd()).dry_run,
            cycle_id=request.cycle_id,
            resume_request=request.resume_request,
        )
        hard_errors = list(getattr(result, "hard_errors", []))
        status = WorkerExit.SUCCESS.value if not hard_errors else WorkerExit.RECOVERABLE_FAILURE.value
        incident_id = None
        if hard_errors:
            outcomes = list(getattr(result, "stage_outcomes", []))
            outcome = next((item for item in reversed(outcomes) if isinstance(item, StageOutcome) and item.failure_signature), None)
            outcome = outcome or StageOutcome.failure("UNKNOWN", "; ".join(hard_errors))
            incident = Incident.from_outcome(worker_id=request.worker_id, generation_id=request.generation_id, competition=request.competition, outcome=outcome, stage_attempt=1, revision=revision, cycle_id=request.cycle_id, experiment_id=getattr(result, "experiment_id", None))
            IncidentStore(layout.state_root).save(incident)
            incident_id = incident.incident_id
        worker_result = WorkerResult(request.worker_id, request.generation_id, status, request.cycle_id, getattr(result, "experiment_id", None), None, incident_id, "; ".join(hard_errors) or "cycle completed", revision)
    except KeyboardInterrupt:
        worker_result = WorkerResult(request.worker_id, request.generation_id, WorkerExit.INTERRUPTED.value, request.cycle_id, None, None, None, "interrupted", revision)
    except Exception as exc:  # noqa: BLE001
        worker_result = WorkerResult(request.worker_id, request.generation_id, WorkerExit.FATAL.value, request.cycle_id, None, request.resume_from_stage, None, str(exc), revision)
    finally:
        beat.stop()
    store.write_json(f"workers/{request.worker_id}/result.json", worker_result.to_dict())
    return 0 if worker_result.status == WorkerExit.SUCCESS.value else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kaggle-agent worker")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    return run_worker(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
