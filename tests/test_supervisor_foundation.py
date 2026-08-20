import json
import subprocess
from pathlib import Path

from kaggle_agent.supervisor.generation import GenerationStore, read_git_revision, read_tree_revision
from kaggle_agent.supervisor.lock import SupervisorLock
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.autonomy.outbox import ExternalActionOutbox
from kaggle_agent.autonomy.runtime import StageLedger


def test_runtime_layout_uses_external_state_root(tmp_path: Path, monkeypatch):
    state = tmp_path / "state"
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state))
    layout = RuntimeLayout.for_repo(tmp_path)
    assert layout.code_root == tmp_path.resolve()
    assert layout.state_root == state
    assert layout.memory_root == state / "memory"


def test_state_store_writes_json_atomically_and_creates_layout(tmp_path: Path):
    store = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "supervisor"))
    store.write_json("supervisor.json", {"status": "observe"})
    assert json.loads((tmp_path / "supervisor" / "supervisor.json").read_text()) == {"status": "observe"}
    assert not list((tmp_path / "supervisor").glob("*.tmp"))
    for name in ("workers", "incidents", "repairs", "reviews", "generations", "worktrees", "accepted", "rejected", "audit", "logs"):
        assert (tmp_path / "supervisor" / name).is_dir()


def test_supervisor_lock_is_exclusive_and_owner_safe(tmp_path: Path):
    first = SupervisorLock(tmp_path / "supervisor.lock")
    second = SupervisorLock(tmp_path / "supervisor.lock")
    assert first.acquire() is True
    assert second.acquire() is False
    second.release()
    first.release()
    assert second.acquire() is True
    first.release()
    second.release()


def test_supervisor_lock_recovers_dead_owner(tmp_path: Path):
    path = tmp_path / "supervisor.lock"
    path.write_text("pid=999999 token=dead at=now\n", encoding="utf-8")
    lock = SupervisorLock(path)
    assert lock.acquire() is True
    assert lock.took_over is True
    lock.release()


def test_revision_helpers_read_sha_and_tree(tmp_path: Path):
    def git(*args: str) -> None:
        subprocess.run(("git", "-C", str(tmp_path), *args), check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "README").write_text("x\n", encoding="utf-8")
    git("add", "README")
    git("-c", "user.email=test@example.com", "-c", "user.name=test", "commit", "-qm", "init")
    revision = read_git_revision(tmp_path)
    assert len(revision) == 40
    assert read_tree_revision(tmp_path) == read_tree_revision(tmp_path)


def test_existing_runtime_ledgers_can_use_shared_state_root(tmp_path: Path):
    shared = tmp_path / "shared"
    assert StageLedger(tmp_path, state_root=shared).path == shared / ".agent" / "stage-ledger.jsonl"
    assert ExternalActionOutbox(tmp_path, state_root=shared).path == shared / ".agent" / "external-outbox.jsonl"


def test_generation_store_imports_clean_code_as_detached_worktree(tmp_path: Path):
    import subprocess
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "x.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "x.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    generation = GenerationStore(state).create_managed(root)
    assert Path(generation.path).is_dir()
    assert Path(generation.path) != root
