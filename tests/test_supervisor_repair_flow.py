import subprocess
from pathlib import Path

from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import RuntimeRevision, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


def test_approved_repair_flow_materializes_generation_and_resume(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
    incident = Incident("i1", "w", "generation-0001", "c", None, "demo", "CODE", 1, revision, "recoverable_failure", "NameError", "NameError", None, "sig", (), (), None, None, None, (), "now")
    spec = RepairSpec("r1", "i1", "generation-0001", revision, "fix", "CODE", "NameError", "bad value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("src",), 8, 5, 500, "CODE", "low", ("x works",))
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    coordinator = RepairCoordinator(root, state)
    result = coordinator.execute(incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec, spec_approved=True, implementer=lambda worktree, _: (worktree / "x.py").write_text("x = 2\n", encoding="utf-8"), reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True))
    assert result.status == "ACCEPTED"
    assert result.generation_id
    assert result.resume_request is not None
