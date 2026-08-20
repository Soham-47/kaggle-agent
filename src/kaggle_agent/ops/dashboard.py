"""Local dashboard: 127.0.0.1 only. Same process reads memory + traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from kaggle_agent.notify.commands import handle_command
from kaggle_agent.notify.run_agent import start_agent_cycle_async
from kaggle_agent.ops.snapshot import build_snapshot
from kaggle_agent.paths import repo_root

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7777
StartCycle = Callable[..., str]


def _safe_static(name: str) -> Path | None:
    if not name or name.startswith(".") or "/" in name or "\\" in name:
        return None
    path = (STATIC_DIR / name).resolve()
    if not str(path).startswith(str(STATIC_DIR.resolve())):
        return None
    return path if path.is_file() else None


def _default_start(root: Path, *, dry_run: bool, command: str) -> str:
    del command
    res = start_agent_cycle_async(root=root, dry_run=dry_run)
    return res.message


def make_handler(
    root: Path,
    *,
    start_cycle: StartCycle | None = None,
) -> type[BaseHTTPRequestHandler]:
    starter = start_cycle

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, default=str).encode("utf-8")
            self._send(code, data, "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            if path in {"/", "/index.html"}:
                page = STATIC_DIR / "index.html"
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/snapshot":
                self._json(200, build_snapshot(root))
                return
            if path.startswith("/static/"):
                file_path = _safe_static(path.removeprefix("/static/"))
                if file_path is None:
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                ctype = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "text/javascript; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(file_path.suffix, "application/octet-stream")
                self._send(200, file_path.read_bytes(), ctype)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/api/command":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json(400, {"ok": False, "reply": "invalid json"})
                return
            text = str((body or {}).get("text") or "").strip()
            if not text.startswith("/"):
                text = f"/{text}" if text else "/help"
            result = handle_command(text, root=root)
            reply = result.reply
            if result.start_cycle:
                fn = starter or (
                    lambda **kw: _default_start(root, **kw)
                )
                reply = fn(dry_run=result.cycle_dry_run, command=text)
            self._json(
                200 if result.ok else 409,
                {
                    "ok": result.ok,
                    "reply": reply,
                    "started": result.start_cycle,
                    "dry_run": result.cycle_dry_run,
                },
            )

    return Handler


def serve(
    root: Path | None = None,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> None:
    root = root or repo_root()
    httpd = ThreadingHTTPServer((host, port), make_handler(root))
    print(f"kaggle-agent dashboard  http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
