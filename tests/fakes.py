"""Shared test doubles (no network)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


def successful_kernel_train(root: Path):
    """Return a kernel-stage stub with a validated candidate artifact."""
    def run(orchestrator, _state, _dry, result):
        source = next((root / "competitions").glob("**/submissions/*.csv"))
        experiment_tag = (result.experiment_id or "run").replace("/", "-")
        kernel_path = source.parent / f"successful-kernel-{experiment_tag}"
        output = kernel_path / "output"
        output.mkdir(parents=True, exist_ok=True)
        header = source.read_text(encoding="utf-8").splitlines()[0]
        columns = len(header.split(","))
        base = 0.2 + (sum(map(ord, experiment_tag)) % 9) * 0.05
        rows = "\n".join(
            f"study-{i}," + ",".join(
                [str(base + (i % 2) * 0.1)] * (columns - 1)
            )
            for i in range(1000)
        )
        submission = output / "submission.csv"
        submission.write_text(header + "\n" + rows + "\n", encoding="utf-8")
        artifact = root / "competitions" / "rsna_knee" / "pipeline" / "artifact_manifest.json"
        if artifact.is_file():
            (kernel_path / "artifact_manifest.json").write_text(
                artifact.read_text(encoding="utf-8"), encoding="utf-8"
            )
            evidence = _synthetic_image_evidence()
            (output / "semantic_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
            (output / "artifact_manifest.runtime.json").write_text(
                json.dumps({
                    "template_version": "rsna-2d-dino-mil-v1",
                    "fold_outputs": evidence["fold_outputs"],
                    "prediction_hashes": evidence["prediction_hashes"],
                }),
                encoding="utf-8",
            )
        result.kernel_ok = True
        result.kernel_ref = "tester/fake-kernel"
        result.kernel_version = 1
        result.kernel_path = str(kernel_path)
        result.candidate_csv = str(submission)
        return orchestrator._sa.load_state()

    return run


def _synthetic_image_evidence() -> dict[str, object]:
    return {
        "mounted_weights_loaded": True,
        "series_mapping_loaded": True,
        "mapped_series_count": 2,
        "mapped_study_count": 1,
        "decoded_non_empty_tensors": True,
        "report_labels_joined": True,
        "group_overlap": False,
        "optimizer_stepped": True,
        "checkpoints_written": True,
        "fold_predictions_written": True,
        "hidden_ids_from_folders": True,
        "submission_rows_match_hidden_ids": True,
        "resumed_folds": [0],
        "newly_trained_folds": [1, 2, 3, 4],
        "resume_checkpoint_source": "/kaggle/input/resume/fold_0_checkpoint.pt",
        "resume_checkpoint_sha256": "synthetic",
        "optimizer_steps": 4,
        "fold_outputs": [f"fold_{i}_predictions.csv" for i in range(5)],
        "prediction_hashes": [f"h{i}" for i in range(5)],
    }


@dataclass
class _Lim:
    num_today: int = 1
    num_total: int = 3
    num_allowed_now: int = 4
    limited_by_total: bool = False


class FakeKaggleApi:
    def __init__(
        self,
        *,
        status_queue: list[str] | None = None,
        failure_message: str | None = None,
    ) -> None:
        self.authenticated = False
        self.submit_calls: list[tuple] = []
        self._status_queue = list(status_queue or [])
        self._status_idx = 0
        self._seen_status = False
        self.failure_message = failure_message

    def authenticate(self) -> None:
        self.authenticated = True

    def competition_get_submission_limits(self, competition_name: str) -> _Lim:
        assert competition_name
        return _Lim()

    def competition_list_files(self, competition: str, page_token=None, page_size: int = 20):
        files = [
            SimpleNamespace(
                name="sample_submission.csv", total_bytes=470, ref="sample_submission.csv"
            ),
            SimpleNamespace(name="test.csv", total_bytes=212, ref="test.csv"),
            SimpleNamespace(
                name="test_series/foo.dcm", total_bytes=1_000_000, ref="test_series/foo.dcm"
            ),
        ]
        return SimpleNamespace(files=files, next_page_token=None)

    def competition_download_file(
        self, competition, file_name, path=None, force=False, quiet=False
    ):
        dest = Path(path or ".")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / file_name).write_text("StudyInstanceUID,ACL\n1,0.5\n", encoding="utf-8")

    def competition_leaderboard_view(self, competition, page_size=20, page_token=None):
        return [
            SimpleNamespace(
                team_name="Alpha", score="0.94", team_id=1, submission_date="2026-08-11"
            ),
            SimpleNamespace(
                team_name="Beta", score="0.93", team_id=2, submission_date="2026-08-10"
            ),
        ]

    def competition_submissions(self, competition, **kwargs):
        return [
            SimpleNamespace(
                ref="s1",
                fileName="sub.csv",
                status="complete",
                publicScore="0.5",
                date="2026-08-01",
                description="test",
            )
        ]

    def competition_submit(
        self, file_name, message, competition, quiet=False, sandbox=False
    ):
        self.submit_calls.append((file_name, message, competition))
        return SimpleNamespace(message="ok")

    def competition_submit_code(
        self,
        file_name,
        message,
        competition=None,
        kernel=None,
        kernel_version=None,
        quiet=False,
    ):
        self.submit_calls.append(
            ("submit_code", file_name, message, competition, kernel, kernel_version)
        )
        return SimpleNamespace(message="ok", ref=kernel or "user/kernel")

    def kernels_list(self, **kwargs):
        return [
            SimpleNamespace(
                ref="user/baseline", title="Baseline CNN", author="user", total_votes=12
            )
        ]

    def kernels_push(self, folder, timeout=None, acc=None):
        self.submit_calls.append(("kernels_push", folder))
        if self._seen_status and self._status_queue:
            self._status_idx = min(self._status_idx + 1, len(self._status_queue) - 1)
        # version_number used by notebook submit path
        return SimpleNamespace(
            message="ok", ref="tester/fake-kernel", versionNumber=1, error=None
        )

    def kernels_status(self, kernel):
        self._seen_status = True
        if self._status_queue:
            idx = min(self._status_idx, len(self._status_queue) - 1)
            status = self._status_queue[idx]
        else:
            status = "COMPLETE"
        return SimpleNamespace(status=status, failureMessage=self.failure_message)

    def kernels_output(
        self,
        kernel,
        path,
        file_pattern=None,
        force=False,
        quiet=True,
        page_token=None,
        page_size=20,
    ):
        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / "submission.csv"
        # Minimal RSNA-shaped header for tests that pull outputs
        kernel_tag = str(kernel or "kernel")
        base = 0.2 + (sum(map(ord, kernel_tag)) % 9) * 0.05
        rows = "\n".join(
            f"k{i}," + ",".join([str(base + (i % 2) * 0.1)] * 12)
            for i in range(1000)
        )
        out.write_text(
            "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,"
            "Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture\n"
            + rows
            + "\n",
            encoding="utf-8",
        )
        evidence = _synthetic_image_evidence()
        (dest / "semantic_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
        (dest / "artifact_manifest.runtime.json").write_text(
            json.dumps({
                "template_version": "rsna-2d-dino-mil-v1",
                "fold_outputs": evidence["fold_outputs"],
                "prediction_hashes": evidence["prediction_hashes"],
            }),
            encoding="utf-8",
        )
        return ([str(out)], "")
