"""Telegram Bot API client (stdlib only).

Auth: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID env vars.
API: https://core.telegram.org/bots/api
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class TelegramError(RuntimeError):
    pass


class SupportsTelegram(Protocol):
    def send_message(self, text: str, *, chat_id: str | None = None) -> dict[str, Any]: ...

    def get_updates(self, *, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]: ...


@dataclass
class TelegramClient:
    token: str
    default_chat_id: str = ""
    base_url: str = "https://api.telegram.org"

    @classmethod
    def from_env(cls) -> TelegramClient | None:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            return None
        chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        return cls(token=token, default_chat_id=chat)

    def set_default_chat(self, chat_id: str) -> None:
        self.default_chat_id = str(chat_id)

    def _call(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/bot{self.token}/{method}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise TelegramError(f"Telegram HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise TelegramError(f"Telegram network error: {e}") from e
        if not body.get("ok"):
            raise TelegramError(f"Telegram API error: {body}")
        return body.get("result")

    def send_message(self, text: str, *, chat_id: str | None = None) -> dict[str, Any]:
        # Telegram limit ~4096 chars
        target = chat_id or self.default_chat_id
        if not target:
            raise TelegramError("no chat_id (message the bot once, or set TELEGRAM_CHAT_ID)")
        chunk = text if len(text) <= 4000 else text[:3990] + "\n…"
        result = self._call(
            "sendMessage",
            {
                "chat_id": target,
                "text": chunk,
                "disable_web_page_preview": True,
            },
        )
        return result if isinstance(result, dict) else {"result": result}

    def get_updates(self, *, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        return list(result or [])


class FakeTelegram:
    """Test double — records outbound messages."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.chat_id = "1"
        self.updates: list[dict[str, Any]] = []
        self._update_id = 1

    def send_message(self, text: str, *, chat_id: str | None = None) -> dict[str, Any]:
        self.sent.append(text)
        return {"message_id": len(self.sent), "chat": {"id": chat_id or self.chat_id}}

    def get_updates(self, *, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]:
        return list(self.updates)

    def push_command(self, text: str, chat_id: str = "1") -> None:
        self.updates.append(
            {
                "update_id": self._update_id,
                "message": {
                    "message_id": self._update_id,
                    "chat": {"id": int(chat_id) if chat_id.isdigit() else chat_id},
                    "text": text,
                },
            }
        )
        self._update_id += 1
