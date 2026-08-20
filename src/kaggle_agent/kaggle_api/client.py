"""Deep Kaggle Public API client.

Official docs: https://www.kaggle.com/docs/api
Python package: `kaggle` (KaggleApi) — methods used (verified on kaggle 2.2.4):

- authenticate()
- competition_get_submission_limits(competition_name)
- competition_list_files(competition, page_token, page_size)
- competition_download_file(competition, file_name, path, force, quiet)
- competition_leaderboard_view(competition, page_size, page_token)
- competition_submissions(competition, ...)
- competition_submit(file_name, message, competition, quiet, sandbox)
- kernels_list(competition=..., page_size=..., sort_by=...)

Auth: ~/.kaggle/kaggle.json (see docs above). Never log key contents.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from kaggle_agent.kaggle_api import submit_ops
from kaggle_agent.kaggle_api.models import (
    CompetitionInfo,
    KernelRow,
    LeaderboardRow,
    MetaFile,
    ResearchSnapshot,
    SubmissionLimits,
    SubmissionRow,
    SubmitResult,
)
from kaggle_agent.kaggle_api.sdk_get import get as _g
from kaggle_agent.kaggle_api.sdk_get import get_str as _s
from kaggle_agent.heal.submit_errors import is_network_error

_META_SUFFIXES = (".csv", ".json", ".md", ".txt", ".parquet")
_MAX_META_BYTES = 50 * 1024 * 1024

_NETWORK_MAX_ATTEMPTS = 3
_NETWORK_BASE_DELAY = 2.0


def _retry_network(
    fn: Callable,
    *,
    attempts: int = _NETWORK_MAX_ATTEMPTS,
    base_delay: float = _NETWORK_BASE_DELAY,
    _sleep: Callable = time.sleep,
) -> Any:
    """Retry *fn* on transient network errors with exponential backoff."""
    delay = base_delay
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i >= attempts - 1 or not is_network_error(str(exc)):
                raise
            _sleep(delay)
            delay *= 2


class KaggleApiError(RuntimeError):
    """Raised when the official client cannot complete a call."""


class SupportsKaggleApi(Protocol):
    """Minimal seam for the official KaggleApi (or a test double)."""

    def authenticate(self) -> None: ...

    def competition_get_submission_limits(self, competition_name: str) -> Any: ...

    def competitions_list(self, **kwargs: Any) -> Any: ...

    def competition_list_files(
        self, competition: str, page_token: str | None = None, page_size: int = 20
    ) -> Any: ...

    def competition_download_file(
        self,
        competition: str,
        file_name: str,
        path: str | None = None,
        force: bool = False,
        quiet: bool = False,
    ) -> None: ...

    def competition_leaderboard_view(
        self,
        competition: str,
        page_size: int | None = 20,
        page_token: str | None = None,
    ) -> list[Any] | None: ...

    def competition_submissions(self, competition: str, **kwargs: Any) -> list[Any] | None: ...

    def competition_submit(
        self,
        file_name: str | None,
        message: str,
        competition: str | None,
        quiet: bool = False,
        sandbox: bool = False,
    ) -> Any: ...

    def competition_submit_code(
        self,
        file_name: str,
        message: str,
        competition: str | None = None,
        kernel: str | None = None,
        kernel_version: int | None = None,
        quiet: bool = False,
    ) -> Any: ...

    def kernels_list(self, **kwargs: Any) -> list[Any] | None: ...

    def kernels_push(
        self, folder: str, timeout: str | None = None, acc: str | None = None
    ) -> Any: ...

    def kernels_status(self, kernel: str) -> Any: ...

    def kernels_output(
        self,
        kernel: str,
        path: str,
        file_pattern: str | None = None,
        force: bool = False,
        quiet: bool = True,
        page_token: str | None = None,
        page_size: int = 20,
    ) -> Any: ...

    def kernels_pull(
        self, kernel: str, path: str, metadata: bool = False, quiet: bool = True
    ) -> None: ...


def _default_api() -> SupportsKaggleApi:
    from kaggle.api.kaggle_api_extended import KaggleApi

    return KaggleApi()


def _is_root_meta(name: str, size: int) -> bool:
    if not name or "/" in name or size > _MAX_META_BYTES:
        return False
    lower = name.lower()
    return any(lower.endswith(sfx) for sfx in _META_SUFFIXES)


class KaggleClient:
    """Small interface over a large official API surface."""

    def __init__(self, api: SupportsKaggleApi | None = None) -> None:
        self._api = api

    @property
    def api(self) -> SupportsKaggleApi:
        if self._api is None:
            self._api = _default_api()
        return self._api

    def connect(self) -> KaggleClient:
        try:
            self.api.authenticate()
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"Kaggle authenticate failed: {exc}") from exc
        return self

    def submission_limits(self, competition: str) -> SubmissionLimits:
        raw = self.api.competition_get_submission_limits(competition)
        return SubmissionLimits(
            num_today=int(_g(raw, "num_today", "numToday", default=0) or 0),
            num_total=int(_g(raw, "num_total", "numTotal", default=0) or 0),
            num_allowed_now=int(_g(raw, "num_allowed_now", "numAllowedNow", default=0) or 0),
            limited_by_total=bool(_g(raw, "limited_by_total", "limitedByTotal", default=False)),
        )

    def competition_info(self, slug: str) -> CompetitionInfo:
        """Return exact competition metadata from the official API search."""
        response = self.api.competitions_list(search=slug)
        rows = _g(response, "competitions", default=response) or []
        wanted = slug.rstrip("/").split("/")[-1]
        matches = []
        for row in rows:
            ref = _s(row, "ref", "url")
            row_slug = ref.rstrip("/").split("/")[-1]
            if row_slug == wanted:
                matches.append(row)
        if len(matches) != 1:
            raise KaggleApiError(
                f"competition metadata must match exactly once for {slug!r}; found {len(matches)}"
            )
        row = matches[0]
        tags = tuple(
            _s(tag, "name", "ref").lower()
            for tag in (_g(row, "tags", default=[]) or [])
            if _s(tag, "name", "ref")
        )
        raw = row.to_dict() if hasattr(row, "to_dict") else {}
        return CompetitionInfo(
            slug=wanted,
            title=_s(row, "title"),
            url=_s(row, "url", "ref"),
            deadline=_s(row, "deadline"),
            evaluation_metric=_s(row, "evaluationMetric", "evaluation_metric"),
            kernels_only=bool(_g(row, "isKernelsSubmissionsOnly", "is_kernels_submissions_only", default=False)),
            max_daily_submissions=int(_g(row, "maxDailySubmissions", "max_daily_submissions", default=1) or 1),
            tags=tags,
            raw=raw,
        )

    def list_meta_files(
        self,
        competition: str,
        *,
        max_pages: int = 5,
        page_size: int = 100,
    ) -> list[MetaFile]:
        """List root-level small text/tabular files (not DICOM series paths)."""
        out: list[MetaFile] = []
        token: str | None = None
        for _ in range(max_pages):
            resp = self.api.competition_list_files(
                competition, page_token=token, page_size=page_size
            )
            for f in _g(resp, "files", default=[]) or []:
                name = _s(f, "name")
                size = int(_g(f, "total_bytes", "totalBytes", "size", default=0) or 0)
                if not _is_root_meta(name, size):
                    continue
                out.append(MetaFile(name=name, total_bytes=size, ref=_s(f, "ref", default=name)))
            token = _g(resp, "next_page_token", "nextPageToken", default=None)
            if not token:
                break
        return out

    def download_file(
        self,
        competition: str,
        file_name: str,
        dest_dir: Path,
        *,
        force: bool = False,
    ) -> Path:
        if "/" in file_name:
            raise KaggleApiError(
                f"Refusing nested path download (DICOM/tree risk): {file_name!r}"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.api.competition_download_file(
                competition, file_name, path=str(dest_dir), force=force, quiet=True
            )
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"download_file failed for {file_name}: {exc}") from exc

        path = dest_dir / file_name
        if path.is_file():
            return path
        zipped = dest_dir / f"{file_name}.zip"
        if zipped.is_file():
            return zipped
        raise KaggleApiError(f"Expected file missing after download: {path}")

    def leaderboard(self, competition: str, *, top: int = 10) -> list[LeaderboardRow]:
        rows = self.api.competition_leaderboard_view(competition, page_size=top) or []
        out: list[LeaderboardRow] = []
        for r in rows[:top]:
            if r is None:
                continue
            out.append(
                LeaderboardRow(
                    team_name=_s(r, "team_name", "teamName"),
                    score=_s(r, "score"),
                    team_id=_g(r, "team_id", "teamId"),
                    submission_date=_s(r, "submission_date", "submissionDate"),
                )
            )
        return out

    def kernels(
        self,
        competition: str,
        *,
        top: int = 10,
        sort_by: str = "voteCount",
    ) -> list[KernelRow]:
        rows = self.api.kernels_list(
            competition=competition, page_size=top, sort_by=sort_by
        ) or []
        out: list[KernelRow] = []
        for k in rows[:top]:
            if k is None:
                continue
            ref = _s(k, "ref")
            out.append(
                KernelRow(
                    ref=ref,
                    title=_s(k, "title", default=ref) or ref,
                    author=_s(k, "author"),
                    total_votes=_g(k, "total_votes", "totalVotes"),
                )
            )
        return out

    def submissions(self, competition: str, *, top: int = 20) -> list[SubmissionRow]:
        rows = _retry_network(
            lambda: self.api.competition_submissions(competition)
        ) or []
        out: list[SubmissionRow] = []
        for s in rows[:top]:
            if s is None:
                continue
            out.append(
                SubmissionRow(
                    ref=_s(s, "ref"),
                    file_name=_s(s, "fileName", "file_name"),
                    status=_s(s, "status"),
                    public_score=_s(s, "publicScore", "public_score"),
                    date=_s(s, "date"),
                    description=_s(s, "description"),
                )
            )
        return out

    def submit(
        self,
        competition: str,
        file_path: Path | None,
        message: str,
        *,
        dry_run: bool = True,
        mode: str = "file",
        kernel_folder: Path | None = None,
        kernel_ref: str | None = None,
        kernel_version: int | None = None,
        output_file: str = "submission.csv",
        poll_seconds: int = 30,
        poll_attempts: int = 40,
    ) -> SubmitResult:
        """file = CSV upload; notebook = push kernel + submit_code (kernels-only comps)."""
        mode = (mode or "file").lower()
        if dry_run:
            target = (
                f"kernel {kernel_ref or kernel_folder} → {output_file}"
                if mode == "notebook"
                else (Path(file_path).name if file_path else "csv")
            )
            return SubmitResult(
                dry_run=True,
                message=f"dry_run: would {mode}-submit {target} to {competition}: {message}",
            )
        try:
            if mode == "notebook":
                return submit_ops.submit_notebook(
                    self.api,
                    competition=competition,
                    message=message,
                    kernel_folder=Path(kernel_folder or "."),
                    kernel_ref=kernel_ref,
                    kernel_version=kernel_version,
                    output_file=output_file,
                    status_fn=self.kernels_status,
                    poll_seconds=poll_seconds,
                    poll_attempts=poll_attempts,
                )
            path = Path(file_path) if file_path else None
            if path is None or not path.is_file():
                return SubmitResult(
                    dry_run=False, message=f"file missing: {file_path}", success=False
                )
            return submit_ops.submit_file(self.api, competition, path, message)
        except RuntimeError as exc:
            raise KaggleApiError(str(exc)) from exc

    def username(self) -> str:
        """Kaggle username from config (after authenticate)."""
        # KaggleApi stores config; also readable from ~/.kaggle/kaggle.json
        cfg_user = _g(self.api, "config_values", default=None)
        if isinstance(cfg_user, dict) and cfg_user.get("username"):
            return str(cfg_user["username"])
        try:
            import json
            from pathlib import Path

            raw = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
            return str(raw["username"])
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"cannot resolve kaggle username: {exc}") from exc

    def kernels_push(self, folder: Path | str) -> str:
        """Push kernel folder (must contain kernel-metadata.json + code_file).

        Source: KaggleApi.kernels_push
        """
        try:
            resp = _retry_network(lambda: self.api.kernels_push(str(folder)))
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_push failed: {exc}") from exc
        return _s(resp, "message", "ref", "errorMessage", default=str(resp))

    def kernels_push_result(self, folder: Path | str) -> Any:
        """Push a kernel and preserve the API response, including its version."""
        try:
            return _retry_network(lambda: self.api.kernels_push(str(folder)))
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_push failed: {exc}") from exc

    def kernels_status(self, kernel_ref: str) -> str:
        """Return status string e.g. COMPLETE / RUNNING.

        Source: KaggleApi.kernels_status
        """
        from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref

        ref = normalize_kernel_ref(kernel_ref)
        try:
            resp = _retry_network(lambda: self.api.kernels_status(ref))
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_status failed: {exc}") from exc
        return _s(resp, "status", default=str(resp))

    def kernels_failure_message(self, kernel_ref: str) -> str:
        """Host failure text for ERROR kernels (P100 ban, OOM, traceback)."""
        from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref

        ref = normalize_kernel_ref(kernel_ref)
        try:
            resp = self.api.kernels_status(ref)
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_status failed: {exc}") from exc
        return _s(resp, "failureMessage", "errorMessage", "failure_message", default="")

    def kernels_output(self, kernel_ref: str, dest_dir: Path) -> list[str]:
        """Download kernel output files into dest_dir.

        Source: KaggleApi.kernels_output → (files, token)
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            resp = self.api.kernels_output(kernel_ref, str(dest_dir), quiet=True)
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_output failed: {exc}") from exc
        if isinstance(resp, tuple) and resp:
            files = resp[0] or []
            return [str(f) for f in files]
        if isinstance(resp, list):
            return [str(f) for f in resp]
        return []

    def kernels_pull(self, kernel_ref: str, dest_dir: Path) -> Path:
        """Download the kernel's source notebook (.ipynb) into dest_dir.

        Returns the pulled notebook path. Deep research uses this to read
        top kernel code. Source: KaggleApi.kernels_pull.
        """
        from kaggle_agent.kaggle_api.submit_ops import normalize_kernel_ref

        ref = normalize_kernel_ref(kernel_ref)
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.api.kernels_pull(ref, str(dest_dir), metadata=False, quiet=True)
        except Exception as exc:  # noqa: BLE001
            raise KaggleApiError(f"kernels_pull failed for {ref}: {exc}") from exc
        slug = ref.rsplit("/", 1)[-1]
        candidates = [dest_dir / f"{slug}.ipynb", dest_dir / f"{slug}.py"]
        for cand in candidates:
            if cand.is_file():
                return cand
        raise KaggleApiError(f"kernels_pull produced no notebook for {ref} in {dest_dir}")

    def research_snapshot(
        self,
        competition: str,
        *,
        lb_top: int = 10,
        kernel_top: int = 8,
    ) -> ResearchSnapshot:
        """One call site for RESEARCH: limits + meta + LB + kernels + our subs."""
        snap = ResearchSnapshot(competition=competition)
        steps: list[tuple[str, Callable[[], None]]] = [
            ("limits", lambda: setattr(snap, "limits", self.submission_limits(competition))),
            ("meta_files", lambda: setattr(snap, "meta_files", self.list_meta_files(competition))),
            ("leaderboard", lambda: setattr(snap, "leaderboard", self.leaderboard(competition, top=lb_top))),
            ("kernels", lambda: setattr(snap, "kernels", self.kernels(competition, top=kernel_top))),
            ("submissions", lambda: setattr(snap, "my_submissions", self.submissions(competition, top=10))),
        ]
        for label, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 — partial snapshot is useful
                snap.errors.append(f"{label}: {exc}")
        return snap
