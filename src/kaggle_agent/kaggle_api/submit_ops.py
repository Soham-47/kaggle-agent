"""Competition submit helpers (file CSV vs notebook-only)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.autonomy.outbox import ExternalActionOutbox, kernel_push_key, submission_key, reconcile_with_kaggle
from kaggle_agent.kaggle_api.sdk_get import get_str, http_detail
from kaggle_agent.heal.submit_errors import classify_submit_failure

_DONE = frozenset({"accepted", "complete", "completed", "success", "succeeded"})
_FAIL = frozenset({"cancelled", "canceled", "error", "failed", "failure", "rejected"})


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
    authoritative = get_str(resp, "status")
    normalized = authoritative.lower().replace(" ", "")
    if (
        normalized in _FAIL
        or normalized.startswith(("error", "fail", "cancel", "reject"))
        or normalized.endswith(("failed", "failure", "cancelled", "canceled"))
    ):
        return SubmitResult(dry_run=False, message=authoritative or "submission failed", success=False, raw_status=authoritative)
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


def _version_number(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
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
        # This is an externally visible mutation.  The durable outbox owns
        # retries/reconciliation; this function performs exactly one call.
        push = api.kernels_push(str(variant))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"kernels_push failed: {http_detail(exc)}") from exc

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
    outbox: ExternalActionOutbox | None = None,
    push_idempotency_key: str | None = None,
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
    push_action = None
    variant: Path = kernel_folder
    ref = effective_ref
    err = ""
    version = kernel_version
    reusable_variant = bool(meta and meta.get("enable_internet") is False and kernel_version is not None)
    if not reusable_variant:
        if outbox is not None:
            push_action = outbox.enqueue(
                action="kernel_push",
                idempotency_key=push_idempotency_key or kernel_push_key(competition, effective_ref, _artifact_sha256(kernel_folder)),
                payload={
                    "competition": competition,
                    "kernel_ref": effective_ref,
                    "package_fingerprint": _artifact_sha256(kernel_folder),
                    "nested_submission_push": True,
                },
            )
            if push_action.status in {"sent", "unknown"}:
                push_action = reconcile_with_kaggle(outbox, push_action, kernel_status=status_fn, submissions=lambda _: [])
            if push_action.status == "accepted":
                ref = push_action.external_ref or effective_ref
                version = push_action.external_version
                if version is None:
                    return SubmitResult(dry_run=False, message="accepted kernel push has no durable version", success=False)
            elif push_action.status == "rejected":
                return SubmitResult(dry_run=False, message="kernel push intent was rejected", success=False)
            elif push_action.status == "prepared":
                outbox.mark_sent(push_action.action_id)
            else:
                return SubmitResult(
                    dry_run=False,
                    message="kernel push intent awaits reconciliation",
                    success=False,
                    raw_status=json.dumps({"action_id": push_action.action_id, "status": push_action.status}),
                )
        if push_action is None or push_action.status != "accepted":
            try:
                variant, ref, err, version, _ = _submit_variant_folder(api, kernel_folder, effective_ref)
                if outbox is not None and push_action is not None:
                    push_action = outbox.record_delivery(push_action.action_id, external_ref=ref, external_version=_version_number(version))
            except RuntimeError as exc:
                if outbox is not None and push_action is not None:
                    outbox.mark_unknown(push_action.action_id)
                return _failure_result("kernels_push", exc)
    if err and err not in {"none", "None", ""}:
        if outbox is not None and push_action is not None:
            # A push response containing an error may still race with remote
            # acceptance; preserve the intent as unknown for reconciliation.
            outbox.mark_unknown(push_action.action_id)
        return SubmitResult(dry_run=False, message=f"kernels_push error: {err}", success=False)
    if not ref:
        if outbox is not None and push_action is not None:
            outbox.mark_unknown(push_action.action_id)
        return SubmitResult(
            dry_run=False, message="kernels_push returned no ref", success=False
        )
    version = _version_number(version)
    if version is None:
        if outbox is not None and push_action is not None:
            outbox.mark_unknown(push_action.action_id)
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
        st = str(getattr(last, "status", last)).lower().replace(" ", "").replace("kernelworkerstatus.", "")
        if st in _DONE or st.endswith("complete") or st.endswith("completed"):
            if outbox is not None and push_action is not None:
                outbox.reconcile(push_action.action_id, status="accepted", external_ref=ref, external_version=_version_number(version))
            break
        if st in _FAIL or any(st.endswith(x) for x in _FAIL):
            if outbox is not None and push_action is not None:
                outbox.reconcile(push_action.action_id, status="rejected", external_ref=ref, external_version=_version_number(version))
            return SubmitResult(
                dry_run=False,
                message=f"kernel {ref} ended with status={last}",
                success=False,
                raw_status=str(last),
            )
        time.sleep(max(1, poll_seconds))
    else:
        if outbox is not None and push_action is not None:
            outbox.mark_unknown(push_action.action_id)
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
        kwargs["kernel_version"] = version
    provenance = {
        "artifact_sha256": _artifact_sha256(variant),
        "kernel_ref": ref,
        "kernel_version": version,
        "output_file": output_file,
    }
    code_action = None
    if outbox is not None:
        code_key = submission_key(competition, "notebook", hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest())
        code_action = outbox.enqueue(
            action="submit_code",
            idempotency_key=code_key,
            payload={"competition": competition, "message": message, "reconciliation_marker": message},
        )
        if code_action.status in {"sent", "unknown"}:
            code_action = reconcile_with_kaggle(outbox, code_action, kernel_status=status_fn, submissions=lambda comp: api.competition_submissions(comp))
        if code_action.status == "accepted":
            return SubmitResult(dry_run=False, message=f"notebook submit already accepted ref={code_action.external_ref or ref}", success=True)
        if code_action.status != "prepared":
            return SubmitResult(dry_run=False, message="submit_code intent awaits reconciliation", success=False)
        outbox.mark_sent(code_action.action_id)
    try:
        resp = api.competition_submit_code(**kwargs)
    except Exception as exc:  # noqa: BLE001
        if outbox is not None and code_action is not None:
            outbox.mark_unknown(code_action.action_id)
        return _failure_result("submit_code", exc)

    authoritative = get_str(resp, "status")
    normalized = authoritative.strip().lower().replace(" ", "").replace("_", "")
    response_ref = get_str(resp, "ref", "submission_ref", "submissionRef") or ref
    if (
        normalized in _FAIL
        or normalized.startswith(("error", "fail", "cancel", "reject"))
        or normalized.endswith(("failed", "failure", "cancelled", "canceled"))
    ):
        if outbox is not None and code_action is not None:
            outbox.reconcile(code_action.action_id, status="rejected", external_ref=response_ref)
        return SubmitResult(dry_run=False, message=f"submit_code failed status={authoritative}", success=False, raw_status=authoritative)
    if normalized not in _DONE and not normalized.endswith("complete") and not normalized.endswith("success"):
        if outbox is not None and code_action is not None:
            outbox.reconcile(code_action.action_id, status="unknown", external_ref=response_ref)
        return SubmitResult(
            dry_run=False,
            message="submit_code intent awaits reconciliation",
            success=False,
            raw_status=authoritative or get_str(resp, "message", default=str(resp)),
        )
    status = get_str(resp, "message", "ref", "status", default=str(resp))
    if outbox is not None and code_action is not None:
        outbox.reconcile(code_action.action_id, status="accepted", external_ref=response_ref)
    return SubmitResult(
        dry_run=False,
        message=f"notebook submit ok ref={ref} v={version} status={status}",
        success=True,
        raw_status=json.dumps({"status": status, "provenance": provenance}, sort_keys=True),
    )
