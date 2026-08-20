"""Controlled validation flows for supervisor stabilization."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from kaggle_agent.autonomy.outcomes import StageOutcome
from kaggle_agent.autonomy.runtime import StageExecutor, StageInput, StageLedger, StageResult
from kaggle_agent.config import load_settings
from kaggle_agent.supervisor.agents import DeepSeekSupervisorAgents
from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import RuntimeRevision, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.resume import ResumeRequest
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.spec import SpecReview, SpecReviewVerdict
from kaggle_agent.supervisor.loop import Supervisor


def _git_repo(root: Path) -> RuntimeRevision:
    root.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=root, check=True)
    (root / "src").mkdir()
    (root / "src" / "bug.py").write_text("value = missing_name\n", encoding="utf-8")
    subprocess.run(("git", "add", "src/bug.py"), cwd=root, check=True)
    subprocess.run(("git", "-c", "user.email=test@example.invalid", "-c", "user.name=test", "commit", "-qm", "baseline"), cwd=root, check=True)
    return RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")


def _observe_settings(root: Path):
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "settings.yaml").write_text(
        yaml.safe_dump({"default_competition": "demo", "supervisor": {"enabled": True, "mode": "observe"}}),
        encoding="utf-8",
    )
    return load_settings(root)


def _observe_incident(revision: RuntimeRevision, message: str, signature: str) -> Incident:
    return Incident.from_outcome(
        worker_id="worker-observe", generation_id="generation-0001", competition="demo",
        outcome=StageOutcome.failure("CODE", message, signature), stage_attempt=1,
        revision=revision, exception_type="NameError" if "NameError" in message else None,
    )


class _ObserveAgents:
    classify_calls = 0

    def classify(self, incident):
        self.classify_calls += 1
        return FailureClassification(FailureClass.CODE_DEFECT, 0.91, True, ("src/bug.py",), "fallback")

    def author_spec(self, incident, classification, *, repair_id):
        return RepairSpec(
            repair_id, incident.incident_id, incident.generation_id, incident.revision, "observe spec",
            incident.stage, incident.exception_message, "synthetic root cause", "fails", "works",
            ("src/bug.py",), "STATIC_REPRO", (), (), (), (), (), ("src",), 8, 5, 500,
            "CODE", "low", ("the focused check passes",),
        )

    def review_spec(self, *args):
        return SpecReview(SpecReviewVerdict.APPROVE, (), (), "bounded synthetic repair")


def test_observe_captures_incident_and_persists_spec_without_repair(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    settings = _observe_settings(root)
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    supervisor = Supervisor(settings, root)
    incident = _observe_incident(revision, "NameError: missing_name", "observe-deterministic")
    supervisor.store.write_json(f"incidents/{incident.incident_id}.json", incident.to_dict())
    agents = _ObserveAgents()
    monkeypatch.setattr(DeepSeekSupervisorAgents, "from_env", classmethod(lambda cls: agents))

    result = supervisor._handle_worker_result(
        {"status": "RECOVERABLE_FAILURE", "incident_id": incident.incident_id}, "observe", "demo"
    )

    assert result == ("SPEC_READY", f"repair-{incident.incident_id[:12]}-a1")
    assert agents.classify_calls == 0
    assert supervisor.store.read_json(f"incidents/{incident.incident_id}/classification.json")["failure_class"] == "CODE_DEFECT"
    assert supervisor.store.read_json(f"repairs/repair-{incident.incident_id[:12]}-a1/spec.json")["incident_id"] == incident.incident_id


def test_observe_uses_deepseek_only_for_unknown_classification(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    settings = _observe_settings(root)
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(tmp_path / "state"))
    supervisor = Supervisor(settings, root)
    incident = _observe_incident(revision, "opaque provider response", "observe-unknown")
    supervisor.store.write_json(f"incidents/{incident.incident_id}.json", incident.to_dict())
    agents = _ObserveAgents()
    monkeypatch.setattr(DeepSeekSupervisorAgents, "from_env", classmethod(lambda cls: agents))

    result = supervisor._handle_worker_result(
        {"status": "RECOVERABLE_FAILURE", "incident_id": incident.incident_id}, "observe", "demo"
    )

    assert result[0] == "SPEC_READY"
    assert agents.classify_calls == 1


def test_repair_only_candidate_is_verified_and_active_generation_is_unchanged(tmp_path: Path):
    root = tmp_path / "repo"
    revision = _git_repo(root)
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    state.write_json("active-generation.json", {"generation_id": "generation-0001"})
    incident = Incident.from_outcome(
        worker_id="worker-1", generation_id="generation-0001", competition="demo",
        outcome=StageOutcome.failure("CODE", "NameError: missing_name", "sig-repair-only"),
        stage_attempt=1, revision=revision, exception_type="NameError",
    )
    spec = RepairSpec(
        "repair-repair-only", incident.incident_id, "generation-0001", revision, "fix missing name",
        "CODE", "NameError: missing_name", "the module references an undefined name",
        "importing src/bug.py fails", "the module imports", ("src/bug.py",), "STATIC_REPRO",
        ("uv run python -m py_compile src/bug.py",), ("do not change external actions",),
        (".env",), (), ("uv run python -m py_compile src/bug.py",), ("src",), 8, 5, 500,
        "CODE", "low", ("py_compile succeeds",),
    )
    result = RepairCoordinator(root, state).execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), spec,
        spec_approved=True,
        implementer=lambda worktree, _: (worktree / "src" / "bug.py").write_text("value = 1\n", encoding="utf-8"),
        reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True),
        mode="repair_only",
    )

    assert result.status == "CANDIDATE_ACCEPTED"
    assert result.candidate_revision
    assert result.candidate_path
    assert state.read_json("active-generation.json")["generation_id"] == "generation-0001"
    accepted = state.read_json("accepted/repair-repair-only.json")
    assert accepted["status"] == "CANDIDATE_ACCEPTED"
    assert Path(result.candidate_path, "src", "bug.py").read_text(encoding="utf-8") == "value = 1\n"


def test_replay_epochs_execute_only_invalidated_stages(tmp_path: Path):
    ledger = StageLedger(tmp_path, state_root=tmp_path / "runtime")
    executor = StageExecutor(ledger)
    calls = {"RESEARCH": 0, "KERNEL_TRAIN": 0}

    def run(stage: str):
        def execute(_: StageInput):
            calls[stage] += 1
            return StageResult(StageOutcome.success(stage, "completed"), {"count": calls[stage]})
        return execute

    research = StageInput.create(stage="RESEARCH", cycle_id="cycle-1", competition="demo", inputs={}, replay_epoch=0)
    kernel = StageInput.create(stage="KERNEL_TRAIN", cycle_id="cycle-1", competition="demo", inputs={}, replay_epoch=0)
    assert executor.execute(research, run("RESEARCH")).replayed is False
    assert executor.execute(kernel, run("KERNEL_TRAIN")).replayed is False

    resume = ResumeRequest(
        "cycle-1", "incident-1", "generation-0001", "generation-0002", "KERNEL_TRAIN", "KERNEL_TRAIN",
        ("RESEARCH",), ("KERNEL_TRAIN",), (), (("KERNEL_TRAIN", 1),),
    )
    replay_research = StageInput.create(stage="RESEARCH", cycle_id="cycle-1", competition="demo", inputs={}, replay_epoch=resume.epoch_for("RESEARCH"))
    rerun_kernel = StageInput.create(stage="KERNEL_TRAIN", cycle_id="cycle-1", competition="demo", inputs={}, replay_epoch=resume.epoch_for("KERNEL_TRAIN"))
    assert executor.execute(replay_research, run("RESEARCH")).replayed is True
    assert executor.execute(rerun_kernel, run("KERNEL_TRAIN")).replayed is False
    assert calls == {"RESEARCH": 1, "KERNEL_TRAIN": 2}


def test_repair_flow_rejects_noop_candidate(tmp_path: Path):
    root = tmp_path / "repo"
    revision = _git_repo(root)
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    incident = Incident.from_outcome(
        worker_id="worker-1", generation_id="generation-0001", competition="demo",
        outcome=StageOutcome.failure("CODE", "NameError: missing_name", "sig-noop"),
        stage_attempt=1, revision=revision, exception_type="NameError",
    )
    spec = RepairSpec(
        "repair-noop", incident.incident_id, "generation-0001", revision, "noop",
        "CODE", "NameError", "root cause", "fails", "works", ("src/bug.py",), "STATIC_REPRO",
        ("uv run python -m py_compile src/bug.py",), (), (), (), (), ("src",), 8, 5, 500,
        "CODE", "low", ("compile succeeds",),
    )
    result = RepairCoordinator(root, state).execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.95, True), spec,
        spec_approved=True, implementer=lambda *_: None,
        reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True),
        mode="repair_only",
    )

    assert result.status == "REJECTED"
