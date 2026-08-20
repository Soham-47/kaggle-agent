from pathlib import Path

import yaml

from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeGeneration, RuntimeRevision
from kaggle_agent.supervisor.health import HealthResult
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.loop import Supervisor
from kaggle_agent.supervisor.policy import SafetyViolation
from kaggle_agent.supervisor.repair_flow import RepairFlowResult
from kaggle_agent.supervisor.resume import ResumeRequest
from kaggle_agent.supervisor.promote import RepairAcceptance
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.spec import SpecReview, SpecReviewVerdict
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


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


def test_auto_safe_requires_explicit_automatic_promotion(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "auto_safe")
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    revision = RuntimeRevision("a", "b", "generation-0001")
    incident = Incident(
        "i1", "w1", "generation-0001", "cycle-1", None, "demo", "CODE", 1, revision,
        "recoverable_failure", "NameError", "NameError", None, "sig", (), (), None, None,
        None, (), "now",
    )
    state.write_json("incidents/i1.json", incident.to_dict())

    class FakeSpec:
        max_changed_source_files = 1
        max_changed_test_files = 0
        max_changed_lines = 20

        def save(self, _state_root):
            return None

    class FakeSessions:
        def author_spec(self, *_args, **_kwargs):
            return FakeSpec()

        def review_spec(self, *_args, **_kwargs):
            return SpecReview(SpecReviewVerdict.APPROVE)

    monkeypatch.setattr(
        "kaggle_agent.supervisor.loop.classify_after_reconciliation",
        lambda *_args: FailureClassification(FailureClass.CODE_DEFECT, 0.99, True),
    )
    monkeypatch.setattr("kaggle_agent.supervisor.loop.DeepSeekSupervisorAgents.from_env", lambda: FakeSessions())
    monkeypatch.setattr(
        "kaggle_agent.supervisor.loop.RepairCoordinator.execute",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not execute without automatic promotion")),
    )

    result = Supervisor(settings, tmp_path)._handle_worker_result(
        {"status": "RECOVERABLE_FAILURE", "incident_id": "i1"}, "auto_safe", "demo"
    )

    assert result == ("NEEDS_AUTHORITY", "automatic promotion is disabled unless supervisor.promotion.automatic is true")


def test_supervisor_does_not_start_replacement_for_live_unadoptable_worker(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "observe")
    monkeypatch.setattr(
        Supervisor,
        "recover_startup",
        lambda self, timeout_seconds: (type("Recovery", (), {"action": "TERMINATE_OR_RECONCILE", "worker_id": "w1"})(),),
    )
    started = []
    monkeypatch.setattr("kaggle_agent.supervisor.loop.WorkerLauncher.start", lambda *args, **kwargs: started.append(True))

    result = Supervisor(settings, tmp_path).run_once(wait=False)

    assert result.status == "RECOVERY_BLOCKED"
    assert started == []


def test_auto_safe_promotes_accepted_generation_and_starts_one_resume_worker(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path, "auto_safe")
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    state = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    old = RuntimeGeneration("generation-0001", RuntimeRevision("a", "b", "generation-0001"), str(tmp_path), created_at="now")
    new = RuntimeGeneration("generation-0002", RuntimeRevision("c", "d", "generation-0002"), str(tmp_path), parent_generation=old.generation_id, repair_id="r1", created_at="now")
    GenerationStore(state).save(old)
    GenerationStore(state).save(new)
    state.write_json("active-generation.json", old.to_dict())
    incident = Incident(
        "i1", "w1", old.generation_id, "cycle-1", None, "demo", "KERNEL_TRAIN", 1,
        old.revision, "recoverable_failure", "NameError", "NameError", None, "sig", (), (),
        None, None, None, (), "now",
    )
    resume = ResumeRequest(
        "cycle-1", "i1", old.generation_id, new.generation_id, "KERNEL_TRAIN", "KERNEL_TRAIN",
        ("RESEARCH", "PLAN", "CODE", "LOCAL_SMOKE"), ("KERNEL_TRAIN",), (), (("KERNEL_TRAIN", 1),),
    )
    outcome = RepairFlowResult("ACCEPTED", RepairAcceptance.all_passed(), new.generation_id, resume)
    supervisor = Supervisor(settings, tmp_path)
    monkeypatch.setattr("kaggle_agent.supervisor.loop.RepairPolicy.require_clean_auto_safe", lambda *_: None)
    monkeypatch.setattr("kaggle_agent.supervisor.loop.read_git_revision", lambda *_: "c")
    monkeypatch.setattr("kaggle_agent.supervisor.promote.GenerationPromotion.health_check", lambda *_: HealthResult(True, ("import",)))
    requests = []

    class FakeProcess:
        pid = 12345

        def wait(self):
            return 0

    def start(_launcher, request, *, cwd=None):
        requests.append((request, cwd))
        return FakeProcess()

    monkeypatch.setattr("kaggle_agent.supervisor.loop.WorkerLauncher.start", start)
    status, reason = supervisor._promote_and_resume(outcome, incident, selected_competition="demo")

    assert status == "WORKER_STARTED"
    assert reason.startswith("worker-")
    assert state.read_json("active-generation.json")["generation_id"] == new.generation_id
    assert state.read_json("promotion.json")["status"] == "PROMOTED"
    assert len(requests) == 1
    assert requests[0][0].generation_id == new.generation_id
    assert requests[0][0].resume_request == resume
