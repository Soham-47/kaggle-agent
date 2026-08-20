"""Restart inspection without blindly starting duplicate workers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.supervisor.heartbeat import HeartbeatStore
from kaggle_agent.supervisor.state import SupervisorStateStore


@dataclass(frozen=True)
class WorkerRecovery:
    worker_id: str
    pid: int | None
    owned: bool
    heartbeat_fresh: bool
    action: str


class SupervisorRecovery:
    def __init__(self, store: SupervisorStateStore) -> None:
        self.store = store

    def inspect_worker(self, worker_id: str, *, timeout_seconds: float, owner_token: str | None = None) -> WorkerRecovery:
        metadata = self.store.read_json(f"workers/{worker_id}/metadata.json", {}) or {}
        pid = metadata.get("pid")
        owner = str(metadata.get("supervisor_token") or "")
        alive = False
        if isinstance(pid, int):
            alive = self._pid_is_live(pid)
        fresh = HeartbeatStore(self.store.layout.state_root).is_fresh(worker_id, timeout_seconds=timeout_seconds)
        owned = bool(owner) and (owner_token is None or owner == owner_token)
        if alive and owned and fresh:
            action = "ADOPT"
        elif alive:
            action = "TERMINATE_OR_RECONCILE"
        else:
            action = "START_OR_RESUME"
        return WorkerRecovery(worker_id, pid if isinstance(pid, int) else None, owned, fresh, action)

    @staticmethod
    def _pid_is_live(pid: int) -> bool:
        """Treat an unreaped zombie as exited, not as an adoptable worker."""
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                contents = proc_stat.read_text(encoding="utf-8")
                state_start = contents.rfind(")") + 2
                if state_start > 1 and contents[state_start:state_start + 1] == "Z":
                    return False
            except OSError:
                pass
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def recover_workers(self, *, timeout_seconds: float, owner_token: str) -> tuple[WorkerRecovery, ...]:
        workers = self.store.layout.state_root / "workers"
        promotion = self.store.read_json("promotion.json", {}) or {}
        promoted_worker = (
            str(promotion.get("replacement_worker_id") or "")
            if promotion.get("status") == "PROMOTED" else ""
        )
        recovered: list[WorkerRecovery] = []
        for metadata in sorted(workers.glob("*/metadata.json")):
            worker_id = metadata.parent.name
            item = self.inspect_worker(worker_id, timeout_seconds=timeout_seconds, owner_token=owner_token)
            if (
                item.action == "START_OR_RESUME"
                and worker_id != promoted_worker
                and not (metadata.parent / "result.json").is_file()
            ):
                self.store.write_json(
                    f"workers/{worker_id}/result.json",
                    {
                        "worker_id": worker_id,
                        "generation_id": self.store.read_json(f"workers/{worker_id}/metadata.json", {}).get("generation_id"),
                        "status": "INTERRUPTED",
                        "exit_reason": "supervisor restarted after worker exit",
                    },
                )
            recovered.append(item)
        return tuple(recovered)
