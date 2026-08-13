"""Minimal HTTP client for Kaggle's official MCP server.

Auth: prefer ``~/.kaggle/access_token`` (KGAT_…), else ``kaggle.json`` key.
Write tools (submit) need the access_token; API key alone is often read-only.

Endpoint: https://www.kaggle.com/mcp
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MCP_URL = "https://www.kaggle.com/mcp"
DEFAULT_TIMEOUT = 90


class KaggleMcpError(RuntimeError):
    pass


def load_mcp_bearer() -> str:
    """Bearer for MCP Authorization header."""
    import os

    env = (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_MCP_TOKEN") or "").strip()
    if env:
        return env
    access = Path.home() / ".kaggle" / "access_token"
    if access.is_file():
        tok = access.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        raw = json.loads(kaggle_json.read_text(encoding="utf-8"))
        key = str(raw.get("key") or raw.get("token") or "").strip()
        if key:
            return key
    raise KaggleMcpError(
        "no MCP token: set KAGGLE_API_TOKEN or write ~/.kaggle/access_token "
        "(Kaggle Settings → Generate New Token)"
    )


class KaggleMcpClient:
    """JSON-RPC + SSE-ish responses from Kaggle.Web MCP."""

    def __init__(self, bearer: str | None = None, *, url: str = MCP_URL) -> None:
        self.url = url
        self._bearer = bearer
        self._initialized = False

    @property
    def bearer(self) -> str:
        if self._bearer is None:
            self._bearer = load_mcp_bearer()
        return self._bearer

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.bearer}",
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise KaggleMcpError(f"MCP HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise KaggleMcpError(f"MCP network: {exc}") from exc

        data = _parse_rpc_body(raw)
        if "error" in data and data["error"]:
            raise KaggleMcpError(f"MCP error: {data['error']}")
        return data.get("result", data)

    def ensure_init(self) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "kaggle-agent", "version": "0.1"},
            },
        )
        self._initialized = True

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.ensure_init()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        return _unwrap_tool_result(result, tool=name)


def _parse_rpc_body(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("{"):
        return json.loads(text)
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise KaggleMcpError(f"unparseable MCP body: {text[:200]}")


def _unwrap_tool_result(result: Any, *, tool: str) -> Any:
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        msg = _content_text(result) or str(result)
        raise KaggleMcpError(f"{tool}: {msg}")
    # content: [{type:text, text:"..."}] often JSON string
    text = _content_text(result)
    if text is None:
        return result
    low = text.strip().lower()
    if low in {"unauthenticated", "unauthorized"} or "permission" in low and "denied" in low:
        raise KaggleMcpError(f"{tool}: {text.strip()}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _content_text(result: dict[str, Any]) -> str | None:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            return str(first.get("text") or first.get("value") or "")
    return None
