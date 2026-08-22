from pathlib import Path

import yaml

from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.loop import Supervisor
from kaggle_agent.supervisor.policy import SafetyViolation
from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.supervisor.incidents import Incident, IncidentStore
from kaggle_agent.supervisor.generation import RuntimeRevision


def _settings(tmp_path: Path, mode: str, enabled: bool = True):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text(yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": enabled, "mode": mode}}), encoding="utf-8")
    return load_settings(tmp_path)


def test_supervisor_off_is_safe_noop(tmp_path: Path):
    result = Supervisor(_settings(tmp_path, "off"), tmp_path).run_once(wait=False)
    assert result.status == "OFF"


def test_auto_safe_refuses_dirty_checkout(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "auto_safe")
    def refuse(self, root):
        raise SafetyViolation("DIRTY_SOURCE_BASELINE")

    monkeypatch.setattr("kaggle_agent.supervisor.policy.RepairPolicy.require_clean_auto_safe", refuse)
    result = Supervisor(settings, tmp_path).run_once(wait=False)
    assert result.status == "DIRTY_SOURCE_BASELINE"


def test_supervisor_passes_explicit_run_intent_to_worker(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "observe")
    seen = {}
    from kaggle_agent.supervisor.generation import RuntimeGeneration, RuntimeRevision

    monkeypatch.setattr(
        Supervisor,
        "_active_generation",
        lambda self, managed=False: RuntimeGeneration(
            "generation-0001",
            RuntimeRevision("git", "tree", "generation-0001"),
            str(tmp_path),
        ),
    )

    class Process:
        pid = 123

        def wait(self):
            return 0

    def start(self, request, *, cwd=None):
        seen["request"] = request
        return Process()

    monkeypatch.setattr("kaggle_agent.supervisor.loop.WorkerLauncher.start", start)
    result = Supervisor(settings, tmp_path).run_once(wait=False, dry_run=False)
    assert result.status == "WORKER_STARTED"
    assert seen["request"].dry_run is False


def test_supervisor_routes_recoverable_worker_failure_to_bounded_incident_handling(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "observe")
    supervisor = Supervisor(settings, tmp_path)
    incident = Incident.from_outcome(
        worker_id="worker-1",
        generation_id="generation-0001",
        competition="demo",
        outcome=StageOutcome.failure("CODE", "NameError: missing recipe"),
        stage_attempt=1,
        revision=RuntimeRevision("git", "tree", "generation-0001"),
    )
    IncidentStore(supervisor.layout.state_root).save(incident)
    result_path = supervisor.store.write_json(
        "workers/worker-1/result.json",
        {
            "status": "RECOVERABLE_FAILURE",
            "incident_id": incident.incident_id,
            "exit_reason": "stage failed",
        },
    )
    assert result_path.is_file()
    reason = supervisor._route_recoverable_failure(
        {"incident_id": incident.incident_id, "status": "RECOVERABLE_FAILURE"}, "stage failed"
    )
    assert "CODE_DEFECT" in reason
    handling = supervisor.store.read_json(f"audit/incident-{incident.incident_id}.json")
    assert handling["bounded"] is True
    assert handling["budget_available"] is True


def test_supervisor_classifies_worker_exit_without_result_as_fatal(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "observe")
    from kaggle_agent.supervisor.generation import RuntimeGeneration, RuntimeRevision
    monkeypatch.setattr(
        Supervisor,
        "_active_generation",
        lambda self, managed=False: RuntimeGeneration(
            "generation-0001", RuntimeRevision("git", "tree", "generation-0001"), str(tmp_path)
        ),
    )

    class Process:
        pid = 123
        def poll(self):
            return 0

    monkeypatch.setattr(
        "kaggle_agent.supervisor.loop.WorkerLauncher.start",
        lambda self, request, **kwargs: Process(),
    )
    result = Supervisor(settings, tmp_path).run_once(wait=True)
    assert result.status == "FATAL"
    assert "without result" in result.reason
    incidents = list((Supervisor(settings, tmp_path).layout.state_root / "incidents").glob("incident-*.json"))
    assert incidents
