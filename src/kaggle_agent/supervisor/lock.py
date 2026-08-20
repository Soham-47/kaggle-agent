"""Exclusive supervisor lock, separate from the worker's RunLock."""

from __future__ import annotations

import fcntl
import os
import time
import uuid
from pathlib import Path


class SupervisorLock:
    STALE_AGE_SECONDS = 12 * 3600

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None
        self._held = False
        self._token = uuid.uuid4().hex
        self.took_over = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            existing = os.read(fd, 4096).decode("utf-8", errors="replace").strip()
            if existing and not self._stale(existing):
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False
            self.took_over = bool(existing)
            if self.took_over:
                for token in existing.split():
                    if token.startswith("token=") and token[6:]:
                        # Preserve the ownership token across a supervisor
                        # restart so a still-live worker can be adopted.
                        self._token = token[6:]
                        break
            data = f"pid={os.getpid()} token={self._token} at={time.time()}\n".encode()
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, data)
            os.fsync(fd)
        except Exception:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise
        self._fd, self._held = fd, True
        return True

    @property
    def owner_token(self) -> str:
        return self._token

    def _stale(self, text: str) -> bool:
        pid = None
        for token in text.split():
            if token.startswith("pid="):
                try:
                    pid = int(token[4:])
                except ValueError:
                    pass
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            return False
        try:
            return time.time() - self.path.stat().st_mtime > self.STALE_AGE_SECONDS
        except FileNotFoundError:
            return True

    def release(self) -> None:
        if not self._held:
            return
        try:
            current = self.path.read_text(encoding="utf-8").strip() if self.path.exists() else ""
            if f"token={self._token}" in current:
                self.path.unlink(missing_ok=True)
        finally:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            self._fd, self._held, self.took_over = None, False, False
