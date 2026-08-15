"""Competition submit helpers (file CSV vs notebook-only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.kaggle_api.sdk_get import get, get_str, http_detail

_DONE = frozenset({"complete", "completed", "success"})
_FAIL = frozenset({"error", "failed", "cancelled", "canceled"})


def normalize_kernel_ref(ref: str | None) -> str:
    """Kaggle push sometimes returns ``/code/user/slug``; status wants ``user/slug``."""
    s = (ref or "").strip()
    if not s:
        return ""
    s = s.replace("https://www.kaggle.com/", "").replace("http://www.kaggle.com/", "")
    s = s.lstrip("/")
    if s.startswith("code/"):
        s = s[len("code/") :]
    if "/code/" in s:
        s = s.split("/code/", 1)[1]
    return s.strip("/")


def split_kernel_ref(ref: str | None) -> tuple[str, str]:
    ref = normalize_kernel_ref(ref)
    if "/" not in ref:
        return "", ref
    owner, slug = ref.split("/", 1)
    return owner, slug.split("/")[0]


def submit_file(api: Any, competition: str, path: Path, message: str) -> SubmitResult:
    try:
        resp = api.competition_submit(str(path), message, competition, quiet=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"submit failed: {http_detail(exc)}") from exc
    status = get_str(resp, "message", "status", default=str(resp) if resp else "")
    return SubmitResult(
        dry_run=False, message=status or "submitted", success=True, raw_status=status
    )


def _fix_metadata_owner(api: Any, kernel_folder: Path) -> None:
    """Rewrite kernel-metadata.json id owner to the authenticated user.

    A stale owner (e.g. "local-user") makes the API treat the re-push as a
    new kernel and fail with 409 title conflict.
    """
    meta = kernel_folder / "kernel-metadata.json"
    if not meta.is_file():
        return
    owner = getattr(api, "username", None) or ""
    if not owner:
        cfg = getattr(api, "config_values", None) or {}
        owner = str(cfg.get("username") or "")
    if not owner:
        return
    try:
        import json

        data = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    k_id = str(data.get("id") or "")
    if "/" not in k_id:
        return
    if k_id.split("/", 1)[0] != owner:
        data["id"] = f"{owner}/{k_id.split('/', 1)[1]}"
        meta.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def submit_notebook(
    api: Any,
    *,
    competition: str,
    message: str,
    kernel_folder: Path,
    kernel_ref: str | None,
    output_file: str,
    status_fn: Callable[[str], str],
    poll_seconds: int = 45,
    poll_attempts: int = 60,
) -> SubmitResult:
    """Push kernel, wait COMPLETE, then competition_submit_code(submission.csv)."""
    if not kernel_folder.is_dir():
        return SubmitResult(
            dry_run=False, message=f"kernel folder missing: {kernel_folder}", success=False
        )
    _fix_metadata_owner(api, kernel_folder)
    try:
        push = api.kernels_push(str(kernel_folder))
    except Exception as exc:  # noqa: BLE001
        from kaggle_agent.heal.pins import apply_pin_heal, is_pin_error

        detail = http_detail(exc)
        if is_pin_error(str(detail)) or is_pin_error(str(exc)):
            apply_pin_heal(kernel_folder.parent.parent, kernel_folder)
            try:
                push = api.kernels_push(str(kernel_folder))
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(f"kernels_push failed: {http_detail(exc2)}") from exc2
        else:
            raise RuntimeError(f"kernels_push failed: {detail}") from exc

    ref = normalize_kernel_ref(get_str(push, "ref", default=kernel_ref or ""))
    if not ref:
        ref = normalize_kernel_ref(kernel_ref)
    version = get(push, "version_number", "versionNumber")
    err = get_str(push, "error")
    if err and err not in {"none", "None", ""}:
        return SubmitResult(dry_run=False, message=f"kernels_push error: {err}", success=False)
    if not ref:
        return SubmitResult(dry_run=False, message="kernels_push returned no ref", success=False)

    last = "pushed"
    for _ in range(max(1, poll_attempts)):
        try:
            last = status_fn(ref)
        except Exception as exc:  # noqa: BLE001
            last = f"status_error:{exc}"
        st = str(last).lower().replace(" ", "").replace("kernelworkerstatus.", "")
        if st in _DONE or st.endswith("complete") or st.endswith("completed"):
            break
        if st in _FAIL or any(st.endswith(x) for x in _FAIL):
            return SubmitResult(
                dry_run=False,
                message=f"kernel {ref} ended with status={last}",
                success=False,
                raw_status=str(last),
            )
        time.sleep(max(1, poll_seconds))
    else:
        return SubmitResult(
            dry_run=False,
            message=f"kernel {ref} still {last} after polling",
            success=False,
            raw_status=str(last),
        )

    kwargs: dict[str, Any] = {
        "file_name": output_file,
        "message": message,
        "competition": competition,
        "kernel": ref,
        "quiet": True,
    }
    if version is not None:
        kwargs["kernel_version"] = int(version)
    try:
        resp = api.competition_submit_code(**kwargs)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"submit_code failed: {http_detail(exc)}") from exc

    status = get_str(resp, "message", "ref", "status", default=str(resp))
    return SubmitResult(
        dry_run=False,
        message=f"notebook submit ok ref={ref} v={version} status={status}",
        success=True,
        raw_status=status,
    )
