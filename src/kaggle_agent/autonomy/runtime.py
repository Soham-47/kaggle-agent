"""Durable, idempotent execution seam for autonomous cycle stages.

The ledger is operational state under ``.agent/`` rather than competition
memory, so it is safe to retain detailed attempts without polluting prompts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome

_LEDGER_SCHEMA_VERSION = 1
_SECRET_KEY = re.compile(
    r"(?:secret|token|password|api[_-]?key|authorization|credential|header|environment|env)",
    re.IGNORECASE,
)


def _safe_value(value: Any, *, key: str = "") -> Any:
    """Keep durable outputs JSON-safe and exclude secret-like fields."""
    if key and _SECRET_KEY.search(key):
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(child_key): safe
            for child_key, child_value in value.items()
            if (safe := _safe_value(child_value, key=str(child_key))) is not None
        }
    if isinstance(value, (list, tuple)):
        return [safe for child in value if (safe := _safe_value(child)) is not None]
    return None


def _safe_outputs(outputs: Mapping[str, Any] | None) -> dict[str, Any]:
    if not outputs:
        return {}
    return {
        str(key): safe
        for key, value in outputs.items()
        if (safe := _safe_value(value, key=str(key))) is not None
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class StageInput:
    """All stable facts a stage needs to decide whether work is duplicate."""

    stage: str
    cycle_id: str
    competition: str
    idempotency_key: str
    inputs: Mapping[str, Any]

    @property
    def stage_execution_key(self) -> str:
        """Cycle-scoped identity used only for replaying this stage execution."""
        return self.idempotency_key

    @classmethod
    def create(
        cls, *, stage: str, cycle_id: str, competition: str, inputs: Mapping[str, Any]
    ) -> "StageInput":
        canonical = json.dumps(
            {"stage": stage, "cycle_id": cycle_id, "competition": competition, "inputs": inputs},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return cls(
            stage=stage,
            cycle_id=cycle_id,
            competition=competition,
            idempotency_key=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            inputs=dict(inputs),
        )


class StageLedger:
    """Append-only execution records with a compact query interface."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".agent" / "stage-ledger.jsonl"

    def append(
        self,
        event: str,
        request: StageInput,
        *,
        outcome: StageOutcome | None = None,
        outputs: Mapping[str, Any] | None = None,
        attempt: int = 0,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "ts": _now(), "event": event, "stage": request.stage,
            "cycle_id": request.cycle_id, "competition": request.competition,
            "idempotency_key": request.idempotency_key, "attempt": attempt,
        }
        if outcome is not None:
            record.update({
                "state": outcome.state.value, "summary": outcome.summary,
                "failure_signature": outcome.failure_signature,
                "external_job": outcome.external_job, "evidence": list(outcome.evidence),
                "artifacts": list(outcome.artifacts),
                "retryable": outcome.retryable,
            })
            if outputs is not None:
                record["outputs"] = _safe_outputs(outputs)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def latest(self, idempotency_key: str) -> dict[str, Any] | None:
        return next(
            (row for row in reversed(self.records()) if row.get("event") == "stage_finished" and row.get("idempotency_key") == idempotency_key),
            None,
        )


@dataclass(frozen=True)
class StageExecution:
    outcome: StageOutcome
    attempt: int
    replayed: bool = False
    outputs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageResult:
    outcome: StageOutcome
    outputs: Mapping[str, Any] = field(default_factory=dict)


StageCallable = Callable[[StageInput], StageOutcome | StageResult]


class StageExecutor:
    """Runs one stage and prevents unsafe duplicate terminal work."""

    def __init__(self, ledger: StageLedger) -> None:
        self.ledger = ledger

    def execute(self, request: StageInput, run: StageCallable) -> StageExecution:
        previous = self.ledger.latest(request.idempotency_key)
        if (
            previous
            and previous.get("state") == OutcomeState.SUCCESS.value
            and "outputs" in previous
        ):
            return StageExecution(
                self._outcome_from_record(previous),
                int(previous["attempt"]),
                True,
                dict(previous.get("outputs") or {}),
            )
        attempt = int(previous["attempt"]) + 1 if previous else 1
        self.ledger.append("stage_started", request, attempt=attempt)
        returned = run(request)
        result = returned if isinstance(returned, StageResult) else StageResult(returned)
        outcome = result.outcome
        if outcome.stage != request.stage:
            raise ValueError(f"stage outcome mismatch: expected {request.stage}, got {outcome.stage}")
        outputs = _safe_outputs(result.outputs)
        self.ledger.append(
            "stage_finished", request, outcome=outcome, outputs=outputs, attempt=attempt
        )
        return StageExecution(outcome, attempt, outputs=outputs)

    @staticmethod
    def _outcome_from_record(record: Mapping[str, Any]) -> StageOutcome:
        return StageOutcome(
            state=OutcomeState(str(record["state"])), stage=str(record["stage"]),
            summary=str(record.get("summary") or "completed"),
            evidence=tuple(str(x) for x in record.get("evidence", ())),
            artifacts=tuple(str(x) for x in record.get("artifacts", ())),
            failure_signature=record.get("failure_signature"),
            external_job=record.get("external_job"),
            retryable=bool(record.get("retryable", False)),
        )
