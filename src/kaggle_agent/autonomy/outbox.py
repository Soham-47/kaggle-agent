"""Crash-safe intent ledger for externally visible mutations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class ExternalAction:
    action_id: str
    action: str
    idempotency_key: str
    payload: Mapping[str, Any]
    status: str
    external_ref: str | None = None


def external_action_key(operation: str, competition: str, **facts: str) -> str:
    """Build a stable identity for one intended external mutation."""
    canonical = json.dumps(
        {"operation": operation, "competition": competition, **facts},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def kernel_push_key(competition: str, kernel_ref: str, package_fingerprint: str) -> str:
    return external_action_key(
        "kernel_push",
        competition,
        kernel_ref=kernel_ref,
        package_fingerprint=package_fingerprint,
    )


def submission_key(competition: str, mode: str, artifact_hash: str) -> str:
    return external_action_key(
        "submit", competition, mode=mode, artifact_hash=artifact_hash
    )


def submission_marker(competition: str, artifact_hash: str) -> str:
    """Return a concise machine-readable marker stable for one artifact."""
    slug = re.sub(r"[^a-z0-9-]+", "-", competition.lower()).strip("-")[:32]
    return f"ka:{slug}:{artifact_hash[:16]}"


def submission_description(competition: str, artifact_hash: str, experiment_id: str) -> str:
    marker = submission_marker(competition, artifact_hash)
    return f"{marker} | agent {experiment_id}"[:100]


class ExternalActionOutbox:
    """Append-only intent state; only reconciliation resolves an action."""

    _PENDING = frozenset({"prepared", "sent", "unknown"})

    def __init__(self, root: Path) -> None:
        self.path = root / ".agent" / "external-outbox.jsonl"

    def enqueue(self, *, action: str, idempotency_key: str, payload: Mapping[str, Any]) -> ExternalAction:
        existing = next(
            (item for item in self._items().values()
             if item.action == action and item.idempotency_key == idempotency_key and item.status in self._PENDING),
            None,
        )
        if existing is not None:
            return existing
        action_id = hashlib.sha256(f"{action}:{idempotency_key}".encode("utf-8")).hexdigest()[:24]
        item = ExternalAction(action_id, action, idempotency_key, dict(payload), "prepared")
        self._append("enqueued", item)
        return item

    def mark_sent(self, action_id: str) -> ExternalAction:
        return self._transition(action_id, "sent")

    def mark_unknown(self, action_id: str) -> ExternalAction:
        return self._transition(action_id, "unknown")

    def reconcile(self, action_id: str, *, status: str, external_ref: str | None = None) -> ExternalAction:
        if status not in {"accepted", "rejected", "unknown"}:
            raise ValueError("reconciliation status must be accepted, rejected, or unknown")
        return self._transition(action_id, status, external_ref=external_ref, event="reconciled")

    def get(self, action_id: str) -> ExternalAction | None:
        return self._items().get(action_id)

    def pending(self) -> list[ExternalAction]:
        return [item for item in self._items().values() if item.status in self._PENDING]

    def _transition(self, action_id: str, status: str, *, external_ref: str | None = None, event: str = "updated") -> ExternalAction:
        current = self.get(action_id)
        if current is None:
            raise KeyError(f"unknown outbox action: {action_id}")
        item = ExternalAction(
            current.action_id, current.action, current.idempotency_key, current.payload,
            status, external_ref if external_ref is not None else current.external_ref,
        )
        self._append(event, item)
        return item

    def _items(self) -> dict[str, ExternalAction]:
        items: dict[str, ExternalAction] = {}
        if not self.path.is_file():
            return items
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                item = ExternalAction(**row["item"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            items[item.action_id] = item
        return items

    def _append(self, event: str, item: ExternalAction) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": _now(), "event": event, "item": item.__dict__}, sort_keys=True) + "\n")


def reconcile_with_kaggle(
    outbox: ExternalActionOutbox,
    item: ExternalAction,
    *,
    kernel_status: Any,
    submissions: Any,
) -> ExternalAction:
    """Resolve only when Kaggle exposes exact evidence of the intent.

    A failed read or a nonmatching record deliberately leaves the action in its
    existing pending state.  Callers must never infer that a send was absent.
    """
    item = outbox.get(item.action_id) or item
    if item.action == "kernel_push":
        ref = str(item.payload.get("kernel_ref") or "")
        try:
            status = str(kernel_status(ref)).strip().lower()
        except Exception:  # noqa: BLE001
            return item
        if ref and status not in {"", "none", "not found", "notfound", "404"}:
            return outbox.reconcile(item.action_id, status="accepted", external_ref=ref)
        return item
    if item.action == "submit":
        competition = str(item.payload.get("competition") or "")
        message = str(item.payload.get("message") or "")
        marker = str(item.payload.get("reconciliation_marker") or "")
        try:
            rows = submissions(competition)
        except Exception:  # noqa: BLE001
            return item
        matches = []
        for row in rows or []:
            description = str(getattr(row, "description", ""))
            if (marker and marker in description) or (not marker and description == message):
                ref = str(getattr(row, "ref", "") or "")
                if ref:
                    matches.append(ref)
        if len(matches) == 1:
            return outbox.reconcile(item.action_id, status="accepted", external_ref=matches[0])
        return item
    raise ValueError(f"unsupported external action: {item.action}")
