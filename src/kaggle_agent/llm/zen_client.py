"""Minimal OpenCode Zen client (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class ZenError(RuntimeError):
    pass


@dataclass
class ZenClient:
    """Call https://opencode.ai/zen/v1/chat/completions with OPENCODE_API_KEY."""

    api_key: str
    base_url: str = "https://opencode.ai/zen/v1"
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls, base_url: str = "https://opencode.ai/zen/v1") -> ZenClient | None:
        key = os.environ.get("OPENCODE_API_KEY", "").strip()
        if not key:
            return None
        return cls(api_key=key, base_url=base_url.rstrip("/"))

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(body).encode("utf-8")
        # Cloudflare on opencode.ai returns 403/1010 to bare urllib without a browser UA.
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Origin": "https://opencode.ai",
                "Referer": "https://opencode.ai/zen",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ZenError(f"Zen HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ZenError(f"Zen network error: {e}") from e

        try:
            return str(payload["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as e:
            raise ZenError(f"Unexpected Zen response: {payload!r}") from e

    def chat_text(self, model: str, system: str, user: str, **kwargs: Any) -> str:
        return self.chat(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

    def chat_vision(
        self,
        model: str,
        system: str,
        user_text: str,
        image_urls: list[str],
        **kwargs: Any,
    ) -> str:
        """Multimodal message: text + image_url parts (for research screenshots)."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return self.chat(
            model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
