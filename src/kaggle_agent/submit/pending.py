"""Pending Kaggle submission approval (lean markdown)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import format_kv_markdown, parse_kv_markdown


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingSubmit:
    exp_id: str = "none"
    csv_path: str = "none"
    kernel_path: str = "none"
    kernel_ref: str = "none"
    status: str = "none"  # none | pending | approved | rejected | submitted
    competition: str = "none"
    message: str = "none"
    requested_at: str = "never"
    decided_at: str = "never"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> PendingSubmit:
        fields = cls.__dataclass_fields__
        return cls(
            **{
                name: d.get(name, field.default)  # type: ignore[misc]
                for name, field in fields.items()
            }
        )


def pending_path(root: Path | None = None) -> Path:
    return memory_dir(root) / "pending_submit.md"


def load_pending(root: Path | None = None) -> PendingSubmit:
    path = pending_path(root)
    if not path.is_file():
        return PendingSubmit()
    return PendingSubmit.from_dict(parse_kv_markdown(path.read_text(encoding="utf-8")))


def save_pending(pending: PendingSubmit, root: Path | None = None) -> Path:
    path = pending_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_kv_markdown("pending submit", pending.to_dict()), encoding="utf-8")
    return path


def request_approval(
    *,
    exp_id: str,
    csv_path: str,
    competition: str,
    message: str = "please review",
    root: Path | None = None,
    kernel_path: str = "none",
    kernel_ref: str = "none",
) -> PendingSubmit:
    pending = PendingSubmit(
        exp_id=exp_id,
        csv_path=csv_path,
        kernel_path=kernel_path or "none",
        kernel_ref=kernel_ref or "none",
        status="pending",
        competition=competition,
        message=message,
        requested_at=_now(),
    )
    save_pending(pending, root)
    return pending


def set_decision(
    exp_id: str,
    *,
    approved: bool,
    root: Path | None = None,
) -> PendingSubmit:
    pending = load_pending(root)
    if pending.exp_id in {"none", ""} or pending.status in {"none", ""}:
        raise ValueError("no pending submit to approve or reject")
    target = pending.exp_id if exp_id == "latest" else exp_id
    if pending.exp_id not in {"none", "", target}:
        raise ValueError(f"exp mismatch: pending={pending.exp_id} requested={exp_id}")
    pending.exp_id = target if target != "none" else pending.exp_id
    pending.status = "approved" if approved else "rejected"
    pending.decided_at = _now()
    save_pending(pending, root)
    return pending


def mark_submitted(root: Path | None = None) -> PendingSubmit:
    pending = load_pending(root)
    pending.status = "submitted"
    pending.decided_at = _now()
    save_pending(pending, root)
    return pending


def usable_approval(
    root: Path | None = None,
    *,
    competition: str | None = None,
) -> PendingSubmit | None:
    """Return pending if already approved and CSV still on disk (ready to submit)."""
    pending = load_pending(root)
    if pending.status != "approved":
        return None
    if competition and pending.competition not in {competition, "none", ""}:
        return None
    if pending.csv_path in {"none", ""} or not Path(pending.csv_path).is_file():
        return None
    return pending
