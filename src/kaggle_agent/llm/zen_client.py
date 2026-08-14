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


def _usage_from_payload(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return {"tokens_in": 0, "tokens_out": 0}
    return {
        "tokens_in": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "tokens_out": int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        ),
    }


@dataclass
class ZenClient:
    """Call https://opencode.ai/zen/v1/chat/completions with OPENCODE_API_KEY."""

    api_key: str
    base_url: str = "https://opencode.ai/zen/v1"
    timeout_s: float = 120.0

    @classmethod
    def from_env(cls, base_url: str | None = None) -> ZenClient | None:
        key = os.environ.get("OPENCODE_API_KEY", "").strip()
        if key:
            url = (base_url or "https://opencode.ai/zen/v1").rstrip("/")
            return cls(api_key=key, base_url=url)
        nvidia = os.environ.get("NVIDIA_API_KEY", "").strip()
        if nvidia:
            url = (base_url or "https://integrate.api.nvidia.com/v1").rstrip("/")
            return cls(api_key=nvidia, base_url=url, timeout_s=180.0)
        return None

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "kaggle-agent/1.0",
        }
        # Cloudflare on opencode.ai returns 403/1010 to bare urllib without a browser UA.
        if "opencode.ai" in self.base_url:
            headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Origin": "https://opencode.ai",
                    "Referer": "https://opencode.ai/zen",
                }
            )
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise ZenError(f"Zen HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise ZenError(f"Zen network error: {e}") from e
        except TimeoutError as e:
            raise ZenError(f"Zen network error: timed out: {e}") from e

        if isinstance(payload, dict) and payload.get("error"):
            raise ZenError(f"Zen provider error: {payload['error']}")
        self.last_usage = _usage_from_payload(payload)
        try:
            msg = payload["choices"][0]["message"]
            content = msg.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                content = msg.get("reasoning_content") or msg.get("reasoning") or ""
            return str(content).strip()
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
