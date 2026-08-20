import subprocess
from pathlib import Path

from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.generation import RuntimeRevision, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.spec import RepairSpec
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.verify import VerificationResult
from kaggle_agent.supervisor.verify import VerificationResult


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
    spec = RepairSpec("r1", "i1", "generation-0001", revision, "fix", "CODE", "NameError", "bad value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("x.py",), 8, 5, 500, "CODE", "low", ("x works",))
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    coordinator = RepairCoordinator(root, state)
    result = coordinator.execute(incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec, spec_approved=True, implementer=lambda worktree, _: (worktree / "x.py").write_text("x = 2\n", encoding="utf-8"), reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True))
    assert result.status == "ACCEPTED"
    assert result.generation_id
    assert result.resume_request is not None


def test_repair_only_keeps_inspectable_candidate_without_promotion(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
    incident = Incident("i2", "w", "generation-0001", "c", None, "demo", "CODE", 1, revision, "recoverable_failure", "NameError", "NameError", None, "sig-2", (), (), None, None, None, (), "now")
    spec = RepairSpec("r2", "i2", "generation-0001", revision, "fix", "CODE", "NameError", "bad value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("x.py",), 8, 5, 500, "CODE", "low", ("x works",))
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    result = RepairCoordinator(root, state).execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec,
        spec_approved=True,
        implementer=lambda worktree, _: (worktree / "x.py").write_text("x = 2\n", encoding="utf-8"),
        reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True),
        mode="repair_only",
    )
    assert result.status == "CANDIDATE_ACCEPTED"
    assert result.candidate_path and Path(result.candidate_path).is_dir()
    assert result.candidate_revision
    assert not (state.layout.state_root / "active-generation.json").exists()
    assert (state.layout.state_root / "accepted" / "r2.json").is_file()


def test_noop_candidate_is_rejected_before_independent_review(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
    incident = Incident("i3", "w", "generation-0001", "c", None, "demo", "CODE", 1, revision, "recoverable_failure", "NameError", "NameError", None, "sig-3", (), (), None, None, None, (), "now")
    spec = RepairSpec("r3", "i3", "generation-0001", revision, "noop", "CODE", "NameError", "bad value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("x.py",), 8, 5, 500, "CODE", "low", ("x works",))
    reviewed = False

    def reviewer(*_args):
        nonlocal reviewed
        reviewed = True
        return Review(ReviewVerdict.APPROVE, True, True, True, True, True)

    result = RepairCoordinator(root, SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))).execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec,
        spec_approved=True, implementer=lambda *_: None, reviewer=reviewer, mode="repair_only",
    )

    assert result.status == "REJECTED"
    assert not reviewed


def test_failed_focused_verification_gets_one_fresh_revision(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
    incident = Incident("i4", "w", "generation-0001", "c", None, "demo", "CODE", 1, revision, "recoverable_failure", "NameError", "NameError", None, "sig-4", (), (), None, None, None, (), "now")
    spec = RepairSpec("r4", "i4", "generation-0001", revision, "fix", "CODE", "NameError", "bad value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("x.py",), 8, 5, 500, "CODE", "low", ("x works",))
    calls = []
    coordinator = RepairCoordinator(root, SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state")))
    verification = [
        VerificationResult(False, (("uv", "run", "pytest", "-q", "tests/test_x.py"),), ("failed (1): test_x",)),
        VerificationResult(True, (("uv", "run", "pytest", "-q", "tests/test_x.py"),), ()),
    ]
    coordinator.verifier.verify = lambda *_args: verification.pop(0)

    def implementer(worktree, _spec, feedback=None):
        calls.append(feedback)
        (worktree / "x.py").write_text("x = 2\n" if feedback is not None else "x = 3\n", encoding="utf-8")

    result = coordinator.execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec,
        spec_approved=True, implementer=implementer,
        reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True),
        mode="repair_only",
    )

    assert result.status == "CANDIDATE_ACCEPTED"
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] is not None
    assert calls[1].attempt == 1
    assert (coordinator.state.layout.state_root / "repairs" / "r4" / "attempt-1-feedback.json").is_file()


def test_same_failed_patch_is_not_retried_indefinitely(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    revision = RuntimeRevision(read_git_revision(root), read_tree_revision(root), "generation-0001")
    incident = Incident("i5", "w", "generation-0001", "c", None, "demo", "CODE", 1, revision, "recoverable_failure", "AssertionError", "AssertionError", None, "sig-5", (), (), None, None, None, (), "now")
    spec = RepairSpec("r5", "i5", "generation-0001", revision, "bad", "CODE", "AssertionError", "wrong value", "fails", "works", ("x.py",), "STATIC_REPRO", (), (), (), (), (), ("x.py",), 8, 5, 500, "CODE", "low", ("x works",))
    coordinator = RepairCoordinator(root, SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state")))
    coordinator.verifier.verify = lambda *_args: VerificationResult(False, (("uv", "run", "pytest"),), ("FAILED test",), exit_code=1)
    calls = 0

    def implementer(worktree, _spec, _feedback=None):
        nonlocal calls
        calls += 1
        (worktree / "x.py").write_text("x = 2\n", encoding="utf-8")

    result = coordinator.execute(
        incident, FailureClassification(FailureClass.CODE_DEFECT, 0.9, True), spec,
        spec_approved=True, implementer=implementer,
        reviewer=lambda *_: Review(ReviewVerdict.APPROVE, True, True, True, True, True),
        mode="repair_only",
    )

    assert result.status == "REJECTED"
    assert result.findings == ("REPEATED_BAD_PATCH",)
    assert calls == 2
