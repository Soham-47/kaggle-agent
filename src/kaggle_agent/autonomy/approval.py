"""Competition-scoped transition from first approval to budgeted autonomy."""

from __future__ import annotations

from pathlib import Path

import yaml


class SubmissionAutonomy:
    def __init__(self, config_path: Path) -> None:
        self.path = config_path

    def _load(self) -> dict:
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    def can_submit_without_approval(self, *, proposals_used: int) -> bool:
        raw = self._load()
        autonomy = raw.get("autonomy") or {}
        return bool(
            autonomy.get("first_submission_approved")
            and autonomy.get("approved_contract_hash") == raw.get("contract_hash")
            and proposals_used < int(autonomy.get("max_submissions_per_day", 0))
        )

    def record_approved_submission(self) -> None:
        raw = self._load()
        autonomy = raw.setdefault("autonomy", {})
        autonomy["first_submission_approved"] = True
        autonomy["approved_contract_hash"] = raw.get("contract_hash")
        self.path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
