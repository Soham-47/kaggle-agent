"""Try LLM providers in order: NVIDIA, Zen, then Hetznez on timeout/limits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from kaggle_agent.llm.zen_client import ZenClient, ZenError

_RETRY_MARKERS = (
    "HTTP 408",
    "HTTP 429",
    "HTTP 502",
    "HTTP 503",
    "HTTP 504",
    "HTTP 529",
    "HTTP 401",
    "HTTP 402",
    "UNAUTHENTICATED",
    "timed out",
    "timeout",
    "network error",
    "504",
    "rate limit",
    "provider error",
    "server_error",
)


def is_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(m.lower() in text for m in _RETRY_MARKERS)


@dataclass
class ProviderSpec:
    name: str
    client: ZenClient
    models: dict[str, str]


class FallbackClient:
    """Same chat API as ZenClient; walks the provider chain on retryable errors."""

    def __init__(self, providers: list[ProviderSpec]) -> None:
        if not providers:
            raise ValueError("FallbackClient needs at least one provider")
        self.providers = providers
        self.last_usage: dict[str, int] = {"tokens_in": 0, "tokens_out": 0}
        self.last_tool_calls: list[tuple[str, dict[str, Any]]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        last: BaseException | None = None
        for spec in self.providers:
            mid = spec.models.get(model) or spec.models.get("*") or model
            try:
                text = spec.client.chat(
                    mid,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                self.last_usage = getattr(spec.client, "last_usage", None) or {
                    "tokens_in": 0,
                    "tokens_out": 0,
                }
                self.last_tool_calls = list(
                    getattr(spec.client, "last_tool_calls", None) or []
                )
                return text
            except ZenError as exc:
                last = exc
                if not is_retryable(exc):
                    raise
                continue
        assert last is not None
        raise last

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
        return self.providers[0].client.chat_vision(
            model, system, user_text, image_urls, **kwargs
        )


def _role_models(block: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for role in ("plan", "code", "distill", "vision"):
        key = f"default_{role}_model"
        if block.get(key):
            out[str(block[key])] = str(block[key])
            out[role] = str(block[key])
    if block.get("default_plan_model"):
        out["*"] = str(block["default_plan_model"])
    return out


def build_llm_client(settings: Any) -> FallbackClient | ZenClient | None:
    raw = getattr(settings, "raw", {}) or {}
    provider = str((raw.get("llm") or {}).get("provider") or "auto").lower()
    nvidia = raw.get("nvidia") or {}
    zen = raw.get("zen") or {}
    hetz = raw.get("hetznez") or {}
    arouter = raw.get("agentrouter") or {}
    dseek = raw.get("deepseek") or {}
    specs: list[ProviderSpec] = []
    zen_only = provider == "zen"
    deepseek_only = provider == "deepseek"

    ds_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if deepseek_only:
        if not ds_key:
            return None
        return ZenClient(
            api_key=ds_key,
            base_url=str(dseek.get("base_url", "https://api.deepseek.com")),
            timeout_s=float(dseek.get("timeout_s", 180)),
        )
    if ds_key and not zen_only:
        specs.append(
            ProviderSpec(
                "deepseek",
                ZenClient(
                    api_key=ds_key,
                    base_url=str(dseek.get("base_url", "https://api.deepseek.com")),
                    timeout_s=float(dseek.get("timeout_s", 180)),
                ),
                _role_models(dseek) or {"*": "deepseek-v4-flash"},
            )
        )

    ar_key = os.environ.get("AGENTROUTER_API_KEY", "").strip()
    if ar_key and not zen_only:
        specs.append(
            ProviderSpec(
                "agentrouter",
                ZenClient(
                    api_key=ar_key,
                    base_url=str(
                        arouter.get("base_url", "https://agentrouter.org/v1")
                    ),
                    timeout_s=float(arouter.get("timeout_s", 90)),
                ),
                _role_models(arouter),
            )
        )

    nv_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    zen_key = os.environ.get("OPENCODE_API_KEY", "").strip()
    if zen_key and not deepseek_only:
        zen_client = ZenClient(
            api_key=zen_key,
            base_url=str(zen.get("base_url", "https://opencode.ai/zen/v1")),
            timeout_s=float(zen.get("timeout_s", 90)),
        )
        free = zen.get("free_models") or []
        if zen_only and free:
            for mid in free:
                mid_s = str(mid).strip()
                if not mid_s:
                    continue
                specs.append(ProviderSpec(f"zen:{mid_s}", zen_client, {"*": mid_s}))
        else:
            specs.append(ProviderSpec("zen", zen_client, _role_models(zen)))

    if nv_key and not zen_only:
        specs.append(
            ProviderSpec(
                "nvidia",
                ZenClient(
                    api_key=nv_key,
                    base_url=str(nvidia.get("base_url", "https://integrate.api.nvidia.com/v1")),
                    timeout_s=float(nvidia.get("timeout_s", 90)),
                ),
                _role_models(nvidia),
            )
        )

    hz_key = os.environ.get("HETZNEZ_API_KEY", "").strip()
    hz_url = (
        os.environ.get("HETZNEZ_BASE_URL", "").strip()
        or str(hetz.get("base_url") or "").strip()
    )
    if hz_key and hz_url and not zen_only:
        specs.append(
            ProviderSpec(
                "hetznez",
                ZenClient(
                    api_key=hz_key,
                    base_url=hz_url.rstrip("/"),
                    timeout_s=float(hetz.get("timeout_s", 90)),
                ),
                _role_models(hetz),
            )
        )

    if not specs:
        return None
    if len(specs) == 1:
        return specs[0].client
    return FallbackClient(specs)
