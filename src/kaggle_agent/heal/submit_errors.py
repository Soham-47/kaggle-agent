"""Classify Kaggle submit/push errors by recovery class."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NETWORK_PATTERNS = re.compile(
    r"connection refused|connection reset|connection timed out|"
    r"urlopen error|getaddrinfo|name or service not known|"
    r"temporary failure in name resolution|"
    r"timed out|timeout|network is unreachable|"
    r"no route to host|no address",
    re.I,
)


@dataclass(frozen=True)
class SubmitFailure:
    """A stable, non-secret recovery decision for a Kaggle API failure."""

    category: str
    retryable: bool
    detail: str


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


def classify_submit_failure(message: str) -> SubmitFailure:
    """Classify a failure without relying on brittle exception types.

    Kaggle's SDK wraps HTTP and DNS errors inconsistently, so recovery is
    deliberately based on the server/client message.  Only transport failures
    are safe to retry automatically: retrying permission, validation, or
    duplicate-title errors consumes time without changing the outcome.
    """
    detail = (message or "unknown Kaggle API failure").strip()
    if is_network_error(detail):
        return SubmitFailure("network", True, detail)
    if is_409_title_conflict(detail):
        return SubmitFailure("title_conflict", False, detail)
    if is_403_submit(detail):
        return SubmitFailure("permission", False, detail)
    low = detail.lower()
    if "429" in low or "rate limit" in low or "too many requests" in low:
        return SubmitFailure("rate_limit", False, detail)
    if "failed_precondition" in low or "enable_internet" in low or "validation" in low:
        return SubmitFailure("precondition", False, detail)
    return SubmitFailure("unknown", False, detail)
