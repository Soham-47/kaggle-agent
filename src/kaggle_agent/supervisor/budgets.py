"""Durable repair budgets and loop detection outside worker state."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore


class RepairBudgetStore:
    def __init__(self, state_root: Path, *, max_attempts_per_incident: int = 3,
                 max_repairs_per_cycle: int = 5, max_repairs_per_day: int = 20) -> None:
        self.store = SupervisorStateStore(RuntimeLayout.for_repo(state_root, state_root))
        self.max_attempts_per_incident = max_attempts_per_incident
        self.max_repairs_per_cycle = max_repairs_per_cycle
        self.max_repairs_per_day = max_repairs_per_day

    def _data(self) -> dict:
        return self.store.read_json("repairs/budgets.json", {"attempts": [], "accepted_failures": []})

    def available(self, incident_id: str, signature: str, cycle_id: str | None, lineage: str | None = None) -> bool:
        data = self._data()
        lineage_key = lineage or ""
        def matches(row: dict) -> bool:
            stored_lineage = row.get("lineage")
            # Legacy rows had no lineage. Count them conservatively for the
            # first lineage that encounters the signature rather than letting
            # an occurrence reset the recurrence budget.
            return row.get("signature") == signature and (stored_lineage in (None, "") or stored_lineage == lineage_key)
        attempts = [
            row for row in data.get("attempts", [])
            if matches(row)
        ]
        if len(attempts) >= self.max_attempts_per_incident:
            return False
        if cycle_id and sum(row.get("cycle_id") == cycle_id for row in data.get("attempts", [])) >= self.max_repairs_per_cycle:
            return False
        day = datetime.now(timezone.utc).date().isoformat()
        if sum(row.get("day") == day for row in data.get("attempts", [])) >= self.max_repairs_per_day:
            return False
        failures = [
            row for row in data.get("accepted_failures", [])
            if matches(row)
        ]
        return len(failures) < 2

    def record(self, incident_id: str, signature: str, cycle_id: str | None, *, accepted: bool, lineage: str | None = None) -> None:
        data = self._data()
        lineage_key = lineage or ""
        data.setdefault("attempts", []).append({
            "incident_id": incident_id, "signature": signature, "cycle_id": cycle_id,
            "lineage": lineage_key, "day": datetime.now(timezone.utc).date().isoformat(), "accepted": accepted,
        })
        if accepted:
            data.setdefault("accepted_failures", []).append({"signature": signature, "lineage": lineage_key})
        self.store.write_json("repairs/budgets.json", data)
