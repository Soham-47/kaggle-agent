"""Managed worker process entrypoint and launcher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore, HeartbeatThread
from kaggle_agent.supervisor.protocol import WorkerExit, WorkerRequest, WorkerResult
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome, failure_signature
from kaggle_agent.supervisor.incidents import Incident, IncidentStore, sanitize_text


class WorkerLauncher:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout

    def start(
        self,
        request: WorkerRequest,
        *,
        cwd: Path | None = None,
        heartbeat_seconds: float = 30,
    ) -> subprocess.Popen[str]:
        store = SupervisorStateStore(self.layout)
        request_path = store.write_json(f"workers/{request.worker_id}/request.json", request.to_dict())
        environment = os.environ.copy()
        environment["KAGGLE_AGENT_SUPERVISOR_DIR"] = str(self.layout.state_root)
        environment["KAGGLE_AGENT_STATE_ROOT"] = str(self.layout.state_root)
        environment["KAGGLE_AGENT_HEARTBEAT_SECONDS"] = str(heartbeat_seconds)
        generation_root = (cwd or self.layout.code_root).resolve()
        environment["KAGGLE_AGENT_GENERATION_ROOT"] = str(generation_root)
        generation_src = generation_root / "src"
        if generation_src.is_dir():
            existing_pythonpath = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = str(generation_src) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        command = [sys.executable, "-m", "kaggle_agent.supervisor.worker", "--request", str(request_path)]
        return subprocess.Popen(command, cwd=str(cwd or self.layout.code_root), env=environment, text=True)

    def terminate_hung(self, process: subprocess.Popen[str], grace_seconds: float) -> None:
        process.terminate()
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=grace_seconds)

    def monitor(
        self,
        process: subprocess.Popen[str],
        worker_id: str,
        *,
        timeout_seconds: float,
        grace_seconds: float,
        progress_timeout_seconds: float | None = None,
    ) -> bool:
        """Return True when a stale heartbeat caused bounded termination."""
        heartbeat = HeartbeatStore(self.layout.state_root)
        beat = heartbeat.read(worker_id)
        if beat is not None and not self._heartbeat_stale(
            beat,
            current=time.time(),
            timeout_seconds=timeout_seconds,
            progress_timeout_seconds=progress_timeout_seconds,
        ):
            return False
        self.terminate_hung(process, grace_seconds)
        return True

    @staticmethod
    def _heartbeat_stale(
        heartbeat: Heartbeat | None,
        *,
        current: float,
        timeout_seconds: float,
        progress_timeout_seconds: float | None,
    ) -> bool:
        if heartbeat is None:
            return False
        if current - heartbeat.timestamp >= timeout_seconds:
            return True
        progress_timestamp = heartbeat.progress_timestamp
        return (
            progress_timeout_seconds is not None
            and progress_timestamp is not None
            and current - progress_timestamp >= progress_timeout_seconds
        )

    def monitor_until_exit(
        self,
        process: subprocess.Popen[str],
        worker_id: str,
        *,
        timeout_seconds: float,
        grace_seconds: float,
        poll_seconds: float = 1.0,
        started_at: float | None = None,
        now: float | None = None,
        progress_timeout_seconds: float | None = None,
    ) -> bool:
        """Wait for a worker while continuously checking its heartbeat."""
        heartbeat = HeartbeatStore(self.layout.state_root)
        started = time.time() if started_at is None else started_at
        current = time.time() if now is None else now
        if not callable(getattr(process, "poll", None)):
            process.wait()
            return False
        while process.poll() is None:
            beat = heartbeat.read(worker_id)
            stale = self._heartbeat_stale(
                beat,
                current=current,
                timeout_seconds=timeout_seconds,
                progress_timeout_seconds=progress_timeout_seconds,
            )
            missing_too_long = beat is None and current - started >= timeout_seconds
            if stale or missing_too_long:
                self.terminate_hung(process, grace_seconds)
                return True
            if poll_seconds > 0:
                time.sleep(poll_seconds)
                if now is None:
                    current = time.time()
                else:
                    current += max(0.001, poll_seconds)
        return False


def run_worker(request_path: Path) -> int:
    request: WorkerRequest | None = None
    worker_id = request_path.parent.name
    state_root = Path(os.environ.get("KAGGLE_AGENT_SUPERVISOR_DIR", request_path.parents[2]))
    try:
        request = WorkerRequest.from_dict(json.loads(request_path.read_text(encoding="utf-8")))
        worker_id = request.worker_id
    except Exception as exc:  # noqa: BLE001
        # Request parsing is itself a durable fatal path; continue through the
        # same result/incident writer instead of exiting with no result.
        parse_error = exc
    else:
        parse_error = None
    generation_root = Path(
        (request.generation_path if request is not None else None)
        or os.environ.get("KAGGLE_AGENT_GENERATION_ROOT", str(Path.cwd()))
    ).resolve()
    layout = RuntimeLayout.for_repo(generation_root, state_root)
    store = SupervisorStateStore(layout)
    if request is None:
        request = WorkerRequest(worker_id, "unknown", "unknown", None, "observe", None, None, RuntimeRevision("", "", "unknown"), True, str(generation_root))
    revision = request.revision or RuntimeRevision("", "", request.generation_id)
    heartbeat = HeartbeatStore(layout.state_root)
    try:
        beat_seconds = float(os.environ.get("KAGGLE_AGENT_HEARTBEAT_SECONDS", "30"))
    except ValueError:
        beat_seconds = 30.0
    beat = HeartbeatThread(heartbeat, Heartbeat(request.worker_id, os.getpid(), request.generation_id, request.cycle_id, request.resume_from_stage, "started", time.time()), max(0.1, beat_seconds))
    beat.start()
    try:
        if parse_error is not None:
            raise ValueError(f"invalid worker request: {parse_error}") from parse_error
        from kaggle_agent.orchestrator import run_daily

        # The request is an immutable supervisor decision.  Settings remain
        # the fallback for old request files that predate the field.
        result = run_daily(
            request.competition,
            root=generation_root,
            dry_run=request.dry_run,
            progress_reporter=lambda stage, progress: beat.update(stage=stage, progress=progress),
        )
        hard_errors = list(getattr(result, "hard_errors", []))
        status = WorkerExit.SUCCESS.value if not hard_errors else WorkerExit.RECOVERABLE_FAILURE.value
        incident_id = None
        if hard_errors:
            outcomes = list(getattr(result, "stage_outcomes", []))
            outcome = next((item for item in reversed(outcomes) if isinstance(item, StageOutcome) and item.failure_signature), None)
            outcome = outcome or StageOutcome.failure("UNKNOWN", "; ".join(hard_errors))
            incident = Incident.from_outcome(worker_id=request.worker_id, generation_id=request.generation_id, competition=request.competition, outcome=outcome, stage_attempt=1, revision=revision, cycle_id=request.cycle_id, experiment_id=getattr(result, "experiment_id", None), parent_occurrence_id=request.parent_occurrence_id, originating_repair_id=request.originating_repair_id, originating_generation_id=request.originating_generation_id)
            IncidentStore(layout.state_root).save(incident)
            incident_id = incident.incident_id
        worker_result = WorkerResult(request.worker_id, request.generation_id, status, request.cycle_id, getattr(result, "experiment_id", None), None, incident_id, "; ".join(hard_errors) or "cycle completed", revision)
    except KeyboardInterrupt:
        worker_result = WorkerResult(request.worker_id, request.generation_id, WorkerExit.INTERRUPTED.value, request.cycle_id, None, None, None, "interrupted", revision)
    except Exception as exc:  # noqa: BLE001
        outcome = StageOutcome(
            OutcomeState.FATAL,
            request.resume_from_stage or "SUPERVISOR",
            str(exc),
            failure_signature=failure_signature(str(exc)),
        )
        incident = Incident.from_outcome(worker_id=request.worker_id, generation_id=request.generation_id, competition=request.competition, outcome=outcome, stage_attempt=1, revision=revision, cycle_id=request.cycle_id, parent_occurrence_id=request.parent_occurrence_id, originating_repair_id=request.originating_repair_id, originating_generation_id=request.originating_generation_id, traceback=traceback.format_exc(), exception_type=type(exc).__name__)
        IncidentStore(layout.state_root).save(incident)
        incident_id = incident.incident_id
        worker_result = WorkerResult(request.worker_id, request.generation_id, WorkerExit.FATAL.value, request.cycle_id, None, request.resume_from_stage, incident_id, sanitize_text(str(exc)), revision)
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
