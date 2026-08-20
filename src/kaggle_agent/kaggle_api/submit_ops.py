"""Competition submit helpers (file CSV vs notebook-only)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.kaggle_api.sdk_get import get, get_str, http_detail
from kaggle_agent.heal.submit_errors import classify_submit_failure

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


def _read_meta(kernel_folder: Path) -> dict[str, Any] | None:
    meta = kernel_folder / "kernel-metadata.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _response_value(response: Any, *names: str) -> Any:
    if isinstance(response, dict):
        for name in names:
            if name in response:
                return response[name]
        return None
    for name in names:
        value = getattr(response, name, None)
        if value is not None:
            return value
    return None


def _fix_metadata_owner(api: Any, kernel_folder: Path) -> None:
    """Rewrite kernel-metadata.json id owner to the authenticated user.

    A stale owner (e.g. "local-user") makes the API treat the re-push as a
    new kernel and fail with 409 title conflict.
    """
    data = _read_meta(kernel_folder)
    if not data:
        return
    owner = getattr(api, "username", None) or ""
    if not owner:
        cfg = getattr(api, "config_values", None) or {}
        owner = str(cfg.get("username") or "")
    if not owner:
        return
    k_id = str(data.get("id") or "")
    if "/" not in k_id:
        return
    if k_id.split("/", 1)[0] != owner:
        data["id"] = f"{owner}/{k_id.split('/', 1)[1]}"
        (kernel_folder / "kernel-metadata.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )


def _disable_internet(kernel_folder: Path) -> None:
    """Force enable_internet=false so the submitted kernel meets the rules.

    Kaggle now enforces this server-side: CreateCodeSubmission returns 400
    ``FAILED_PRECONDITION`` when the kernel metadata has enable_internet=true.
    """
    data = _read_meta(kernel_folder)
    if not data:
        return
    if data.get("enable_internet") is not False:
        data["enable_internet"] = False
        (kernel_folder / "kernel-metadata.json").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )


def _artifact_sha256(folder: Path) -> str:
    """Return a deterministic digest for the exact notebook package pushed."""
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _retry_api(
    operation: Callable[[], Any], *, attempts: int, seconds: float
) -> Any:
    """Retry only a bounded set of transient network failures."""
    for attempt in range(max(1, attempts)):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001
            failure = classify_submit_failure(http_detail(exc) or str(exc))
            if not failure.retryable or attempt >= max(1, attempts) - 1:
                raise
            time.sleep(max(0.0, seconds) * (2**attempt))
    raise AssertionError("unreachable")


def _failure_result(operation: str, exc: Exception) -> SubmitResult:
    failure = classify_submit_failure(http_detail(exc) or str(exc))
    return SubmitResult(
        dry_run=False,
        success=False,
        message=f"{operation} failed category={failure.category}: {failure.detail}",
        raw_status=json.dumps(
            {
                "operation": operation,
                "category": failure.category,
                "retryable": failure.retryable,
            },
            sort_keys=True,
        ),
    )


def _submit_variant_folder(
    api: Any,
    kernel_folder: Path,
    kernel_ref: str,
    *,
    retry_attempts: int,
    retry_seconds: float,
) -> tuple[Path, str, str, int | None, str]:
    """Build + push an internet-off variant of the train kernel.

    Returns (variant_folder, ref, status, version_number, error). The variant
    is a copy of the completed train package with enable_internet forced off,
    so the scored re-run cannot download weights over the network.
    """
    variant = kernel_folder.parent / f"{kernel_folder.name}-submit-offline"
    import shutil

    if variant.is_dir():
        shutil.rmtree(variant)
    shutil.copytree(kernel_folder, variant)
    _fix_metadata_owner(api, variant)
    _disable_internet(variant)
    try:
        push = _retry_api(
            lambda: api.kernels_push(str(variant)),
            attempts=retry_attempts,
            seconds=retry_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        from kaggle_agent.heal.pins import apply_pin_heal, is_pin_error
        from kaggle_agent.heal.submit_errors import is_409_title_conflict

        detail = http_detail(exc)
        detail_str = f"{detail} {exc}"
        if is_pin_error(str(detail)) or is_pin_error(str(exc)):
            apply_pin_heal(variant.parent.parent, variant)
            try:
                push = _retry_api(
                    lambda: api.kernels_push(str(variant)),
                    attempts=retry_attempts,
                    seconds=retry_seconds,
                )
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(f"kernels_push failed: {http_detail(exc2)}") from exc2
        elif is_409_title_conflict(detail_str):
            _fix_metadata_owner(api, variant)
            try:
                push = _retry_api(
                    lambda: api.kernels_push(str(variant)),
                    attempts=retry_attempts,
                    seconds=retry_seconds,
                )
            except Exception as exc2:  # noqa: BLE001
                raise RuntimeError(f"kernels_push failed: {http_detail(exc2)}") from exc2
        else:
            raise RuntimeError(f"kernels_push failed: {detail}") from exc

    ref = normalize_kernel_ref(str(_response_value(push, "ref") or ""))
    if not ref:
        ref = normalize_kernel_ref(kernel_ref)
    version = _response_value(push, "version_number", "versionNumber")
    err = get_str(push, "error")
    return variant, ref, str(err), version, ""


def submit_notebook(
    api: Any,
    *,
    competition: str,
    message: str,
    kernel_folder: Path,
    kernel_ref: str | None,
    kernel_version: int | None = None,
    output_file: str,
    status_fn: Callable[[str], str],
    poll_seconds: int = 45,
    poll_attempts: int = 60,
    retry_attempts: int = 3,
    retry_seconds: float = 2.0,
) -> SubmitResult:
    """Submit the output of a completed kernel via an internet-off variant.

    Kaggle enforces two rules server-side:
    1. CreateCodeSubmission without an explicit kernel_version returns 403
       (Permission 'kernelSessions.get' denied) for API tokens.
    2. A kernel with enable_internet=true returns 400 FAILED_PRECONDITION
       ("Your Notebook cannot use internet access in this competition").

    So we push an internet-off copy of the train package (a new version of the
    same kernel), wait for it to complete, then submit with the explicit
    version number from the push response.
    """
    if not kernel_folder.is_dir():
        return SubmitResult(
            dry_run=False, message=f"kernel folder missing: {kernel_folder}", success=False
        )

    meta = _read_meta(kernel_folder)
    metadata_ref = normalize_kernel_ref(str((meta or {}).get("id") or ""))
    effective_ref = normalize_kernel_ref(kernel_ref) or metadata_ref
    if meta and meta.get("enable_internet") is False and kernel_version is not None:
        variant, ref, err, version = kernel_folder, effective_ref, "", kernel_version
    else:
        try:
            variant, ref, err, version, _ = _submit_variant_folder(
                api,
                kernel_folder,
                effective_ref,
                retry_attempts=retry_attempts,
                retry_seconds=retry_seconds,
            )
        except RuntimeError as exc:
            return _failure_result("kernels_push", exc)
    if err and err not in {"none", "None", ""}:
        return SubmitResult(dry_run=False, message=f"kernels_push error: {err}", success=False)
    if not ref:
        return SubmitResult(
            dry_run=False, message="kernels_push returned no ref", success=False
        )
    if version is None:
        return SubmitResult(
            dry_run=False,
            message="kernels_push returned no kernel version; refusing ambiguous submit_code",
            success=False,
            raw_status=json.dumps(
                {"operation": "kernels_push", "category": "missing_kernel_version"},
                sort_keys=True,
            ),
        )

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
    provenance = {
        "artifact_sha256": _artifact_sha256(variant),
        "kernel_ref": ref,
        "kernel_version": version,
        "output_file": output_file,
    }
    try:
        resp = _retry_api(
            lambda: api.competition_submit_code(**kwargs),
            attempts=retry_attempts,
            seconds=retry_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        return _failure_result("submit_code", exc)

    status = get_str(resp, "message", "ref", "status", default=str(resp))
    return SubmitResult(
        dry_run=False,
        message=f"notebook submit ok ref={ref} v={version} status={status}",
        success=True,
        raw_status=json.dumps({"status": status, "provenance": provenance}, sort_keys=True),
    )
