"""Classify Kaggle submit/push errors by recovery class."""

from __future__ import annotations

import re

_NETWORK_PATTERNS = re.compile(
    r"connection refused|connection reset|connection timed out|"
    r"urlopen error|getaddrinfo|name or service not known|"
    r"temporary failure in name resolution|"
    r"timed out|timeout|network is unreachable|"
    r"no route to host|no address",
    re.I,
)


def is_network_error(message: str) -> bool:
    """True when the message indicates a transient network failure."""
    return bool(_NETWORK_PATTERNS.search(message or ""))


def is_409_title_conflict(message: str) -> bool:
    """True when the message indicates a 409 title-conflict on push."""
    low = (message or "").lower()
    return "409" in low and ("title" in low or "conflict" in low)


def is_403_submit(message: str) -> bool:
    """True when the message indicates a 403 permission denial on submit."""
    low = (message or "").lower()
    if "403" in low or "forbidden" in low:
        return True
    if "permission" in low and ("denied" in low or "forbidden" in low):
        return True
    return False


def classify_submit_error(message: str) -> str | None:
    """Classify a submit/push error for heal notes. Returns the first match."""
    if is_network_error(message):
        return "network"
    if is_409_title_conflict(message):
        return "409"
    if is_403_submit(message):
        return "403"
    return None
