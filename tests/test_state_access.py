"""StateAccessor: one seam over state.md, run.lock, pending_submit.md, kernel_job.md."""

from pathlib import Path

from kaggle_agent.state_access import DiskStateAccessor, MemoryStateAccessor
from kaggle_agent.state_md import AgentState
from kaggle_agent.submit.pending import PendingSubmit
from kaggle_agent.train.kernel_job import KernelJob


def test_memory_accessor_roundtrip():
    sa = MemoryStateAccessor()
    assert sa.load_state().phase == "IDLE"
    st = AgentState(phase="PLAN", public_best="0.526")
    sa.save_state(st)
    assert sa.load_state() is st
    assert sa.load_state().phase == "PLAN"


def test_memory_accessor_lock():
    sa = MemoryStateAccessor()
    assert sa.acquire_lock() is True
    assert sa.acquire_lock() is False
    assert sa.lock_took_over() is False
    sa.release_lock()
    assert sa.acquire_lock() is True


def test_memory_accessor_pending_and_job():
    sa = MemoryStateAccessor()
    assert sa.load_pending().status == "none"
    p = PendingSubmit(exp_id="e1", status="pending")
    sa.save_pending(p)
    assert sa.load_pending() is p
    assert sa.load_kernel_job().kernel_ref == "none"
    j = KernelJob(kernel_ref="r1", status="queued")
    sa.save_kernel_job(j)
    assert sa.load_kernel_job() is j
    sa.clear_kernel_job()
    assert sa.load_kernel_job().kernel_ref == "none"


def test_disk_accessor_writes_files(tmp_path: Path):
    root = tmp_path / "ka"
    sa = DiskStateAccessor(root)
    assert sa.acquire_lock() is True
    assert (root / "memory" / "run.lock").is_file()
    sa.release_lock()
    assert not (root / "memory" / "run.lock").exists()

    st = AgentState(phase="CODE", public_best="0.534")
    sa.save_state(st)
    assert (root / "memory" / "state.md").is_file()
    loaded = sa.load_state()
    assert loaded.phase == "CODE"
    assert loaded.public_best == "0.534"

    p = PendingSubmit(exp_id="e2", status="approved")
    sa.save_pending(p)
    assert (root / "memory" / "pending_submit.md").is_file()
    assert sa.load_pending().status == "approved"

    j = KernelJob(kernel_ref="r9", status="complete")
    sa.save_kernel_job(j)
    assert (root / "memory" / "kernel_job.md").is_file()
    assert sa.load_kernel_job().kernel_ref == "r9"
    sa.clear_kernel_job()
    assert sa.load_kernel_job().kernel_ref == "none"


def test_disk_accessor_lock_took_over(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "run.lock").write_text(
        "pid=99999999 at=now\n", encoding="utf-8"
    )
    sa = DiskStateAccessor(root)
    assert sa.acquire_lock() is True
    assert sa.lock_took_over() is True
    sa.release_lock()
    assert sa.lock_took_over() is False


def test_memory_accessor_implements_protocol():
    from kaggle_agent.state_access import StateAccessor

    def use(sa: StateAccessor) -> AgentState:
        st = sa.load_state()
        st.phase = "REPORT"
        sa.save_state(st)
        return sa.load_state()

    sa = MemoryStateAccessor()
    assert use(sa).phase == "REPORT"
    assert use(DiskStateAccessor(Path("/tmp/opencode/state-access-test"))).phase == "REPORT"