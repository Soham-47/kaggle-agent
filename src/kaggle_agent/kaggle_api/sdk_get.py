"""Read fields from Kaggle SDK objects (mix of snake_case / camelCase / dict)."""

from __future__ import annotations

from typing import Any


def get(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def get_str(obj: Any, *names: str, default: str = "") -> str:
    val = get(obj, *names, default=default)
    return default if val is None else str(val)


def http_detail(exc: BaseException) -> str:
    """Prefer API error body over generic 'Bad Request for url'."""
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            text = getattr(resp, "text", None) or ""
            if text:
                return f"{exc} | body={text[:500]}"
        except Exception:  # noqa: BLE001
            pass
    return str(exc)
