"""Submit via Kaggle MCP: file upload or code (notebook) competition submit."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.kaggle_api.mcp_client import KaggleMcpClient, KaggleMcpError
from kaggle_agent.kaggle_api.models import SubmitResult
from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref, split_kernel_ref

McpCallFn = Callable[[str, dict[str, Any]], Any]


def submit_via_mcp(
    *,
    competition: str,
    message: str,
    mode: str = "file",
    csv_path: Path | None = None,
    kernel_ref: str | None = None,
    kernel_version: int | None = None,
    output_file: str = "submission.csv",
    dry_run: bool = False,
    client: KaggleMcpClient | None = None,
    call_tool: McpCallFn | None = None,
) -> SubmitResult:
    """MCP first-line submit. Does not mark memory; caller handles success."""
    mode = (mode or "file").lower()
    if dry_run:
        target = kernel_ref or (csv_path.name if csv_path else "none")
        return SubmitResult(
            dry_run=True,
            message=f"dry_run: would mcp-{mode}-submit {target} → {competition}",
            success=True,
        )

    def _call(name: str, arguments: dict[str, Any]) -> Any:
        if call_tool is not None:
            return call_tool(name, arguments)
        mcp = client or KaggleMcpClient()
        return mcp.call_tool(name, arguments)

    try:
        if mode == "notebook":
            return _mcp_code_submit(
                _call,
                competition=competition,
                message=message,
                kernel_ref=kernel_ref,
                kernel_version=kernel_version,
                output_file=output_file,
            )
        if csv_path is None or not Path(csv_path).is_file():
            return SubmitResult(
                dry_run=False,
                message=f"mcp file submit: missing csv {csv_path}",
                success=False,
            )
        return _mcp_file_submit(
            _call,
            competition=competition,
            message=message,
            csv_path=Path(csv_path),
        )
    except KaggleMcpError as exc:
        return SubmitResult(dry_run=False, message=f"mcp: {exc}", success=False)
    except Exception as exc:  # noqa: BLE001
        return SubmitResult(dry_run=False, message=f"mcp: {exc}", success=False)


def _mcp_code_submit(
    call: McpCallFn,
    *,
    competition: str,
    message: str,
    kernel_ref: str | None,
    kernel_version: int | None,
    output_file: str,
) -> SubmitResult:
    ref = normalize_kernel_ref(kernel_ref or "")
    owner, slug = split_kernel_ref(ref)
    if not owner or not slug:
        return SubmitResult(
            dry_run=False,
            message=f"mcp code submit: bad kernel_ref {kernel_ref!r}",
            success=False,
        )
    req: dict[str, Any] = {
        "competitionName": competition,
        "kernelOwner": owner,
        "kernelSlug": slug,
        "fileName": output_file,
        "hasFileName": True,
        "submissionDescription": message,
        "hasSubmissionDescription": True,
    }
    if kernel_version is not None:
        req["kernelVersion"] = int(kernel_version)
        req["hasKernelVersion"] = True
    raw = call("create_code_competition_submission", {"request": req})
    return SubmitResult(
        dry_run=False,
        message=f"mcp code submit ok ref={owner}/{slug} resp={_short(raw)}",
        success=True,
        raw_status=str(raw)[:300],
    )


def _mcp_file_submit(
    call: McpCallFn,
    *,
    competition: str,
    message: str,
    csv_path: Path,
) -> SubmitResult:
    data = csv_path.read_bytes()
    start = call(
        "start_competition_submission_upload",
        {
            "request": {
                "competitionName": competition,
                "hasCompetitionName": True,
                "contentLength": len(data),
                "lastModifiedEpochSeconds": int(csv_path.stat().st_mtime),
                "fileName": csv_path.name,
            }
        },
    )
    if not isinstance(start, dict):
        return SubmitResult(
            dry_run=False, message=f"mcp upload start bad resp: {start!r}", success=False
        )
    token = start.get("token") or start.get("blobFileToken") or start.get("blob_file_token")
    create_url = start.get("create_url") or start.get("createUrl") or start.get("url")
    if not token or not create_url:
        return SubmitResult(
            dry_run=False,
            message=f"mcp upload start missing token/url: {start}",
            success=False,
        )

    # Resumable GCS-style upload: PUT body to create_url
    try:
        put = urllib.request.Request(
            str(create_url),
            data=data,
            method="PUT",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(data)),
            },
        )
        with urllib.request.urlopen(put, timeout=120) as resp:
            _ = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        return SubmitResult(
            dry_run=False,
            message=f"mcp blob upload HTTP {exc.code}: {body}",
            success=False,
        )
    except Exception as exc:  # noqa: BLE001
        return SubmitResult(
            dry_run=False, message=f"mcp blob upload failed: {exc}", success=False
        )

    finish = call(
        "submit_to_competition",
        {
            "request": {
                "competitionName": competition,
                "blobFileTokens": str(token),
                "submissionDescription": message,
                "hasSubmissionDescription": True,
            }
        },
    )
    return SubmitResult(
        dry_run=False,
        message=f"mcp file submit ok resp={_short(finish)}",
        success=True,
        raw_status=str(finish)[:300],
    )


def _short(obj: Any, n: int = 180) -> str:
    s = str(obj).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"
