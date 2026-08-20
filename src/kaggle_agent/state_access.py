"""Runtime state access: one seam over the four state files.

The orchestrator reads and writes cycle state through a StateAccessor, never
through the module-level load/save functions directly.  DiskStateAccessor is
the production adapter; MemoryStateAccessor keeps everything in memory for
tests (no tmp_path, no disk I/O).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from kaggle_agent.state_md import AgentState, RunLock, load_state, save_state
from kaggle_agent.submit.pending import PendingSubmit, load_pending, save_pending
from kaggle_agent.train.kernel_job import (
    KernelJob,
    clear_kernel_job,
    load_kernel_job,
    save_kernel_job,
)


class StateAccessor(Protocol):
    def load_state(self) -> AgentState: ...

    def save_state(self, state: AgentState) -> None: ...

    def acquire_lock(self) -> bool: ...

    def lock_took_over(self) -> bool: ...

    def release_lock(self) -> None: ...

    def load_pending(self) -> PendingSubmit: ...

    def save_pending(self, pending: PendingSubmit) -> None: ...

    def load_kernel_job(self) -> KernelJob: ...

    def save_kernel_job(self, job: KernelJob) -> None: ...

    def clear_kernel_job(self) -> None: ...


class DiskStateAccessor:
    """Production adapter: reads and writes the markdown files under root."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root
        self._lock = RunLock(root)

    def load_state(self) -> AgentState:
        return load_state(self._root)

    def save_state(self, state: AgentState) -> None:
        save_state(state, self._root)

    def acquire_lock(self) -> bool:
        return self._lock.acquire()

    def lock_took_over(self) -> bool:
        return self._lock.took_over

    def release_lock(self) -> None:
        self._lock.release()

    def load_pending(self) -> PendingSubmit:
        return load_pending(self._root)

    def save_pending(self, pending: PendingSubmit) -> None:
        save_pending(pending, self._root)

    def load_kernel_job(self) -> KernelJob:
        return load_kernel_job(self._root)

    def save_kernel_job(self, job: KernelJob) -> None:
        save_kernel_job(job, self._root)

    def clear_kernel_job(self) -> None:
        clear_kernel_job(self._root)


class MemoryStateAccessor:
    """Test adapter: keeps all state in memory, no files on disk.

    ``load_state`` returns the stored object itself, so mutating the
    returned state then calling ``save_state`` is a no-op that still works.
    """

    def __init__(self) -> None:
        self._state = AgentState()
        self._pending = PendingSubmit()
        self._job = KernelJob()
        self._locked = False
        self._took_over = False

    def load_state(self) -> AgentState:
        return self._state

    def save_state(self, state: AgentState) -> None:
        self._state = state

    def acquire_lock(self) -> bool:
        if self._locked:
            return False
        self._locked = True
        return True

    def lock_took_over(self) -> bool:
        return self._took_over

    def release_lock(self) -> None:
        self._locked = False

    def load_pending(self) -> PendingSubmit:
        return self._pending

    def save_pending(self, pending: PendingSubmit) -> None:
        self._pending = pending

    def load_kernel_job(self) -> KernelJob:
        return self._job

    def save_kernel_job(self, job: KernelJob) -> None:
        self._job = job

    def clear_kernel_job(self) -> None:
        self._job = KernelJob()