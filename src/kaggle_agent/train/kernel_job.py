"""Track in-flight Kaggle kernel so cron can resume instead of double-pushing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from kaggle_agent.paths import memory_dir
from kaggle_agent.state_md import format_kv_markdown, parse_kv_markdown

# Statuses we treat as "still running" (Kaggle API varies casing).
RUNNING = frozenset(
    {
        "running",
        "queued",
        "pending",
        "cancelrequested",
        "cancel_requested",
        "pushed",  # just pushed, status not yet complete
    }
)
DONE = frozenset({"complete", "completed", "success", "error", "failed", "cancelled", "canceled"})


@dataclass
class KernelJob:
    kernel_ref: str = "none"
    folder: str = "none"
    status: str = "none"
    competition: str = "none"
    exp_id: str = "none"
    updated_at: str = "never"

    def to_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> KernelJob:
        return cls(
            **{
                name: d.get(name, field.default)  # type: ignore[misc]
                for name, field in cls.__dataclass_fields__.items()
            }
        )

    @property
    def is_active(self) -> bool:
        if self.kernel_ref in {"none", ""} or self.status in {"none", ""}:
            return False
        st = self.status.lower().replace(" ", "")
        if st.startswith("kernelworkerstatus."):
            st = st.split(".", 1)[1]
        if st in DONE:
            return False
        if st in RUNNING:
            return True
        # Unknown but not complete → treat as active to avoid double push
        return st not in DONE


def kernel_job_path(root: Path | None = None) -> Path:
    return memory_dir(root) / "kernel_job.md"


def load_kernel_job(root: Path | None = None) -> KernelJob:
    path = kernel_job_path(root)
    if not path.is_file():
        return KernelJob()
    return KernelJob.from_dict(parse_kv_markdown(path.read_text(encoding="utf-8")))


def save_kernel_job(job: KernelJob, root: Path | None = None) -> Path:
    path = kernel_job_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    job.updated_at = datetime.now(timezone.utc).isoformat()
    path.write_text(format_kv_markdown("kernel job", job.to_dict()), encoding="utf-8")
    return path


def clear_kernel_job(root: Path | None = None) -> None:
    path = kernel_job_path(root)
    if path.is_file():
        path.unlink()
