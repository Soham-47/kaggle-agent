"""Independent worker heartbeat persistence."""

from __future__ import annotations

import os
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Heartbeat:
    worker_id: str
    pid: int
    generation_id: str
    cycle_id: str | None
    stage: str | None
    last_progress_event: str
    timestamp: float


class HeartbeatStore:
    def __init__(self, state_root: Path) -> None:
        self.root = state_root

    def path(self, worker_id: str) -> Path:
        return self.root / "workers" / worker_id / "heartbeat.json"

    def write(self, heartbeat: Heartbeat) -> Path:
        destination = self.path(heartbeat.worker_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(asdict(heartbeat), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        return destination

    def read(self, worker_id: str) -> Heartbeat | None:
        path = self.path(worker_id)
        if not path.is_file():
            return None
        try:
            return Heartbeat(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None

    def is_fresh(self, worker_id: str, *, timeout_seconds: float, now: float | None = None) -> bool:
        heartbeat = self.read(worker_id)
        return heartbeat is not None and (now or time.time()) - heartbeat.timestamp < timeout_seconds


class HeartbeatThread:
    def __init__(self, store: HeartbeatStore, heartbeat: Heartbeat, interval_seconds: float) -> None:
        self.store = store
        self.heartbeat = heartbeat
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{heartbeat.worker_id}", daemon=True)

    def start(self) -> None:
        self.store.write(self.heartbeat)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.store.write(Heartbeat(**{**asdict(self.heartbeat), "timestamp": time.time()}))
