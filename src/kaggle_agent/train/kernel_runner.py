"""Push / status / pull Kaggle kernel packages (official API)."""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.kaggle_api.client import KaggleClient
from kaggle_agent.train.kernel_job import (
    DONE,
    KernelJob,
    clear_kernel_job,
    load_kernel_job,
    save_kernel_job,
)
from kaggle_agent.train.kernel_history import (
    kernel_push_lock,
    package_fingerprint,
    package_recipe_hash,
    record_kernel,
    seen_kernel,
    seen_recipe,
)
from kaggle_agent.train.notebook_builder import KernelPackage


def _browser_traceback(kernel_ref: str) -> str:
    """Return a short traceback from the Kaggle page, or an empty string."""
    try:
        from kaggle_agent.research.browser import fetch_via_browser_harness

        text = fetch_via_browser_harness(f"https://www.kaggle.com/code/{kernel_ref}")
    except Exception:  # noqa: BLE001
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = ("traceback (most recent call last):", "error:", "exception:")
    for index, line in enumerate(lines):
        if line.lower() in markers or "traceback (most recent call last):" in line.lower():
            return "\n".join(lines[index : index + 8])[:1200]
    return ""


@dataclass
class KernelRunResult:
    ok: bool
    package: KernelPackage | None
    pending: bool = False
    pushed: bool = False
    resumed: bool = False
    status: str = "local_only"
    message: str = ""
    output_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    kernel_ref: str = "none"
    kernel_version: int | None = None


class KernelPushRepair:
    """Classify and apply bounded repairs for a failed kernel push."""

    @staticmethod
    def classify(message: str) -> str | None:
        from kaggle_agent.heal.pins import is_pin_error
        from kaggle_agent.heal.submit_errors import is_409_title_conflict

        if is_pin_error(message):
            return "pin"
        if is_409_title_conflict(message):
            return "title_conflict"
        return None

    @staticmethod
    def apply(
        action: str,
        *,
        client: KaggleClient,
        package_folder: Path,
        root: Path | None,
    ) -> bool:
        if action == "pin":
            if root is None:
                return False
            from kaggle_agent.heal.pins import apply_pin_heal

            apply_pin_heal(package_folder.parent.parent, package_folder)
            return True
        if action == "title_conflict":
            from kaggle_agent.kaggle_api.submit_ops import _fix_metadata_owner

            _fix_metadata_owner(client.api, package_folder)
            return True
        return False


def _push_result(client: KaggleClient, folder: Path):
    method = getattr(client, "kernels_push_result", None)
    return method(folder) if method is not None else client.kernels_push(folder)


def _push_version(response) -> int | None:
    if isinstance(response, dict):
        value = response.get("version_number") or response.get("versionNumber")
    else:
        value = getattr(response, "version_number", None) or getattr(response, "versionNumber", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def run_kernel_phase(
    client: KaggleClient | None,
    package: KernelPackage | None,
    *,
    push: bool,
    pull_output_dir: Path | None = None,
    root: Path | None = None,
    competition: str = "",
    exp_id: str = "",
    poll_seconds: int = 30,
    poll_attempts: int = 40,
) -> KernelRunResult:
    """Build is already done. Resume in-flight job, or push new; pull if complete.

    dry-run / push=False → local package only (no API).
    """
    # 1) Resume path
    existing = load_kernel_job(root)
    if push and client is not None and existing.is_active:
        return _resume_job(
            client,
            existing,
            pull_output_dir=pull_output_dir,
            root=root,
            package=package,
            poll_seconds=poll_seconds,
            poll_attempts=poll_attempts,
        )

    if package is None:
        return KernelRunResult(
            ok=False, package=None, errors=["no package and no active job"]
        )

    result = KernelRunResult(
        ok=True, package=package, message="package ready", kernel_ref=package.kernel_ref
    )
    if not package.notebook_path.is_file() or not package.metadata_path.is_file():
        result.ok = False
        result.errors.append("kernel package incomplete")
        return result

    if not push:
        result.status = "local_only"
        result.message = f"local package at {package.folder}"
        return result

    if client is None:
        result.ok = False
        result.errors.append("push requested but no KaggleClient")
        return result

    try:
        with kernel_push_lock(root):
            package_fp = package_fingerprint(package.folder)
            recipe_fp = package_recipe_hash(package.folder)
            if root is not None and seen_kernel(root, package_fp):
                result.ok = False
                result.errors.append(
                    "duplicate kernel package: identical package was already recorded"
                )
                return result
            if root is not None and seen_recipe(root, recipe_fp):
                result.ok = False
                result.errors.append(
                    "duplicate recipe: identical recipe was already submitted"
                )
                return result
            try:
                push_resp = _push_result(client, package.folder)
            except Exception as exc2:  # noqa: BLE001
                err_str = str(exc2)
                repair = KernelPushRepair.classify(err_str)
                if repair and KernelPushRepair.apply(
                    repair,
                    client=client,
                    package_folder=package.folder,
                    root=root,
                ):
                    push_resp = _push_result(client, package.folder)
                else:
                    raise
            if root is not None:
                record_kernel(root, package.kernel_ref, package_fp, recipe_fp)
    except Exception as exc:  # noqa: BLE001
        result.ok = False
        result.errors.append(f"push: {exc}")
        return result
    result.pushed = True
    result.kernel_version = _push_version(push_resp)
    result.status = "pushed"
    result.message = str(push_resp)[:300]
    job = KernelJob(
        kernel_ref=package.kernel_ref,
        folder=str(package.folder),
        status="pushed",
        competition=competition,
        exp_id=exp_id,
        kernel_version=str(result.kernel_version) if result.kernel_version is not None else "none",
    )
    save_kernel_job(job, root)

    return _poll_and_maybe_pull(
        client,
        result,
        package.kernel_ref,
        package.folder,
        pull_output_dir=pull_output_dir,
        root=root,
        competition=competition,
        exp_id=exp_id,
        poll_seconds=poll_seconds,
        poll_attempts=poll_attempts,
    )


def package_matches_existing(package: KernelPackage, existing: KernelJob) -> bool:
    """True when the new package trains the same kernel as the last job.

    Only training-relevant artifacts count: the notebook (the kernel body)
    and the kernel env metadata. methods.json is agent-side bookkeeping and
    must not block a resume of an identical notebook.
    """
    if existing.kernel_ref in {"none", ""} or existing.folder in {"none", ""}:
        return False
    folder = Path(existing.folder)
    if not folder.is_dir():
        return False
    nb = "agent_baseline.ipynb"
    if not (package.folder / nb).is_file() or not (folder / nb).is_file():
        return False
    if _normalized_notebook(package.folder / nb) != _normalized_notebook(folder / nb):
        return False
    meta = "kernel-metadata.json"
    if (package.folder / meta).is_file() and (folder / meta).is_file():
        import json

        try:
            new = json.loads((package.folder / meta).read_text(encoding="utf-8"))
            old = json.loads((folder / meta).read_text(encoding="utf-8"))
        except ValueError:
            return False
        for key in ("id", "title"):
            new.pop(key, None)
            old.pop(key, None)
        new.pop("experiment_manifest", None)
        old.pop("experiment_manifest", None)
        if new != old:
            return False
    return True


def _normalized_notebook(path: Path) -> bytes:
    """Ignore per-experiment manifest IDs while retaining the seed."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path.read_bytes()
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source") or [])
        match = re.search(r"^EXPERIMENT_MANIFEST = (\{.*\})$", source, re.M)
        if not match:
            continue
        try:
            manifest = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            continue
        if isinstance(manifest, dict):
            manifest.pop("experiment_id", None)
            replacement = "EXPERIMENT_MANIFEST = " + repr(manifest)
            source = source[: match.start()] + replacement + source[match.end() :]
            cell["source"] = [source]
    return json.dumps(notebook, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _resume_job(
    client: KaggleClient,
    job: KernelJob,
    *,
    pull_output_dir: Path | None,
    root: Path | None,
    package: KernelPackage | None,
    poll_seconds: int = 30,
    poll_attempts: int = 40,
) -> KernelRunResult:
    folder = Path(job.folder) if job.folder not in {"none", ""} else None
    result = KernelRunResult(
        ok=True,
        package=package,
        resumed=True,
        kernel_ref=job.kernel_ref,
        kernel_version=int(job.kernel_version) if job.kernel_version not in {"none", ""} else None,
        message=f"resuming {job.kernel_ref}",
        status=job.status,
    )
    return _poll_and_maybe_pull(
        client,
        result,
        job.kernel_ref,
        folder,
        pull_output_dir=pull_output_dir,
        root=root,
        competition=job.competition,
        exp_id=job.exp_id,
        poll_seconds=poll_seconds,
        poll_attempts=poll_attempts,
    )


def _poll_and_maybe_pull(
    client: KaggleClient,
    result: KernelRunResult,
    kernel_ref: str,
    folder: Path | None,
    *,
    pull_output_dir: Path | None,
    root: Path | None,
    competition: str,
    exp_id: str,
    poll_seconds: int,
    poll_attempts: int,
) -> KernelRunResult:
    try:
        st = "unknown"
        for attempt in range(max(1, poll_attempts)):
            st = client.kernels_status(kernel_ref)
            result.status = st
            result.message = f"status={st}"
            plain_st = str(st).split(".")[-1]
            job = KernelJob(
                kernel_ref=kernel_ref,
                folder=str(folder) if folder else "none",
                status=plain_st,
                competition=competition,
                exp_id=exp_id,
                kernel_version=(
                    str(result.kernel_version)
                    if result.kernel_version is not None
                    else "none"
                ),
            )
            save_kernel_job(job, root)
            st_now = str(st or "").split(".")[-1].lower().replace(" ", "")
            if st_now in DONE or st_now in {"error", "failed"}:
                break
            if attempt == 0 and st_now in {"complete", "completed", "success"}:
                break
            # Tests and hosts that return a terminal status on the first call exit here.
            if st_now not in {
                "running",
                "queued",
                "pending",
                "pushed",
                "cancelrequested",
                "cancel_requested",
            }:
                break
            if attempt < max(1, poll_attempts) - 1:
                time.sleep(max(5, poll_seconds))
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"status: {exc}")
        return result

    st_norm = str(result.status or "").split(".")[-1].lower().replace(" ", "")
    if st_norm in {
        "running",
        "queued",
        "pending",
        "pushed",
        "cancelrequested",
        "cancel_requested",
    }:
        result.ok = False
        result.pending = True
        result.message = f"pending status={result.status}"
        return result
    if st_norm in {"error", "failed"}:
        fail = ""
        try:
            fail = client.kernels_failure_message(kernel_ref)
        except Exception:  # noqa: BLE001
            fail = ""
        if folder is not None and _retry_cpu_after_gpu_ban(client, folder, fail):
            try:
                push_resp = client.kernels_push(folder)
                result.pushed = True
                result.kernel_version = _push_version(push_resp)
                result.message = f"retried cpu after: {fail[:160]}"
                job = KernelJob(
                    kernel_ref=kernel_ref,
                    folder=str(folder),
                    status="pushed",
                    competition=competition,
                    exp_id=exp_id,
                    kernel_version=result.kernel_version,
                )
                save_kernel_job(job, root)
                st = client.kernels_status(kernel_ref)
                result.status = st
                st_norm = str(st or "").split(".")[-1].lower().replace(" ", "")
            except Exception as exc:  # noqa: BLE001
                result.ok = False
                diagnosis = _browser_traceback(kernel_ref)
                detail = fail or result.status
                if diagnosis:
                    detail = f"{detail}\ntraceback:\n{diagnosis}"
                result.errors.append(f"cpu-retry: {exc}\nkernel error: {detail}")
                clear_kernel_job(root)
                return result
            if st_norm in {"error", "failed"}:
                result.ok = False
                diagnosis = _browser_traceback(kernel_ref)
                detail = fail or result.status
                if diagnosis:
                    detail = f"{detail}\ntraceback:\n{diagnosis}"
                result.errors.append(f"kernel error: {detail}")
                clear_kernel_job(root)
                return result
        else:
            result.ok = False
            diagnosis = _browser_traceback(kernel_ref)
            detail = fail or result.status
            if diagnosis:
                detail = f"{detail}\ntraceback:\n{diagnosis}"
            result.errors.append(f"kernel error: {detail}")
            clear_kernel_job(root)
            return result

    if st_norm in DONE:
        out_dir = pull_output_dir
        if out_dir is None and folder is not None:
            out_dir = folder / "output"
        if out_dir is not None:
            try:
                files = client.kernels_output(kernel_ref, out_dir)
                result.output_files = files
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"output: {exc}")
        clear_kernel_job(root)
        result.message = f"complete status={result.status}"
        if st_norm in {"error", "failed"}:
            result.ok = False
    return result


def gpu_forbidden(message: str) -> bool:
    low = (message or "").lower()
    return any(
        token in low
        for token in ("p100", "cannot use gpu", "gpu is not allowed", "cannot use p100")
    )


def _retry_cpu_after_gpu_ban(client: KaggleClient, folder: Path, fail: str) -> bool:
    if not gpu_forbidden(fail):
        return False
    meta_path = folder / "kernel-metadata.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not meta.get("enable_gpu") and not meta.get("machine_shape"):
        return False
    meta["enable_gpu"] = False
    meta.pop("machine_shape", None)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return True
