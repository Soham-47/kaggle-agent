"""Durable, idempotent execution seam for autonomous cycle stages.

The ledger is operational state under ``.agent/`` rather than competition
memory, so it is safe to retain detailed attempts without polluting prompts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome


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

    def append(self, event: str, request: StageInput, *, outcome: StageOutcome | None = None, attempt: int = 0) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
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
            })
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


StageCallable = Callable[[StageInput], StageOutcome]


class StageExecutor:
    """Runs one stage and prevents unsafe duplicate terminal work."""

    def __init__(self, ledger: StageLedger) -> None:
        self.ledger = ledger

    def execute(self, request: StageInput, run: StageCallable) -> StageExecution:
        previous = self.ledger.latest(request.idempotency_key)
        if previous and previous.get("state") == OutcomeState.SUCCESS.value:
            return StageExecution(self._outcome_from_record(previous), int(previous["attempt"]), True)
        attempt = int(previous["attempt"]) + 1 if previous else 1
        self.ledger.append("stage_started", request, attempt=attempt)
        outcome = run(request)
        if outcome.stage != request.stage:
            raise ValueError(f"stage outcome mismatch: expected {request.stage}, got {outcome.stage}")
        self.ledger.append("stage_finished", request, outcome=outcome, attempt=attempt)
        return StageExecution(outcome, attempt)

    @staticmethod
    def _outcome_from_record(record: Mapping[str, Any]) -> StageOutcome:
        return StageOutcome(
            state=OutcomeState(str(record["state"])), stage=str(record["stage"]),
            summary=str(record.get("summary") or "completed"),
            evidence=tuple(str(x) for x in record.get("evidence", ())),
            artifacts=tuple(str(x) for x in record.get("artifacts", ())),
            failure_signature=record.get("failure_signature"),
            external_job=record.get("external_job"),
        )
