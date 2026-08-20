"""submit_ops notebook-submit: push internet-off variant, explicit version."""

import json

from fakes import FakeKaggleApi
from kaggle_agent.kaggle_api.submit_ops import submit_notebook


def _write_meta(folder, **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": "tester/kernel",
        "title": "kernel",
        "code_file": "agent_baseline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
    }
    meta.update(overrides)
    (folder / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def test_submit_notebook_pushes_offline_variant_and_passes_version(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder)
    api = FakeKaggleApi()

    result = submit_notebook(
        api,
        competition="rsna-knee-abnormality-detection",
        message="agent test",
        kernel_folder=folder,
        kernel_ref="tester/kernel",
        output_file="submission.csv",
        status_fn=lambda ref: "COMPLETE",
        poll_seconds=1,
        poll_attempts=3,
    )

    assert result.success
    pushes = [c for c in api.submit_calls if c and c[0] == "kernels_push"]
    assert len(pushes) == 1
    subs = [c for c in api.submit_calls if c and c[0] == "submit_code"]
    assert len(subs) == 1
    _, file_name, message, competition, kernel, version = subs[0]
    assert file_name == "submission.csv"
    assert kernel == "tester/fake-kernel"  # ref from the variant push response
    assert version == 1

    # The variant folder (not the original) has internet off
    variant = folder.parent / "pkg-submit-offline"
    assert variant.is_dir()
    meta = json.loads((variant / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is False
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    # original train package untouched
    orig = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert orig["enable_internet"] is True


def test_submit_notebook_uses_existing_version_when_already_offline(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder, enable_internet=False)
    api = FakeKaggleApi()

    result = submit_notebook(
        api,
        competition="rsna-knee-abnormality-detection",
        message="agent test",
        kernel_folder=folder,
        kernel_ref="tester/kernel",
        output_file="submission.csv",
        status_fn=lambda ref: "COMPLETE",
        poll_seconds=1,
        poll_attempts=3,
    )

    assert result.success
    subs = [c for c in api.submit_calls if c and c[0] == "submit_code"]
    assert len(subs) == 1
    assert subs[0][5] == 1


def test_submit_notebook_submits_offline_kernel_version_without_push(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder, enable_internet=False)
    api = FakeKaggleApi()
    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="agent test",
        kernel_folder=folder, kernel_ref="tester/kernel", kernel_version=3,
        output_file="submission.csv", status_fn=lambda ref: "COMPLETE",
        poll_seconds=1, poll_attempts=3,
    )
    assert result.success
    assert not [c for c in api.submit_calls if c[0] == "kernels_push"]
    assert [c for c in api.submit_calls if c[0] == "submit_code"][0][5] == 3


def test_submit_notebook_uses_metadata_ref_when_ref_omitted(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder, id="meta-owner/meta-kernel", enable_internet=False)
    api = FakeKaggleApi()
    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="m",
        kernel_folder=folder, kernel_ref=None, kernel_version=4,
        output_file="submission.csv", status_fn=lambda ref: "COMPLETE",
        poll_seconds=1, poll_attempts=1,
    )
    assert result.success
    assert [c for c in api.submit_calls if c[0] == "submit_code"][0][4] == "meta-owner/meta-kernel"


def test_submit_notebook_reads_mapping_push_response(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder)
    api = FakeKaggleApi()
    original_push = api.kernels_push
    def push(path):
        original_push(path)
        return {"ref": "/code/u/mapped", "versionNumber": 7}
    api.kernels_push = push
    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="m",
        kernel_folder=folder, kernel_ref="u/old", output_file="submission.csv",
        status_fn=lambda ref: "COMPLETE", poll_seconds=1, poll_attempts=1,
    )
    assert result.success
    call = [c for c in api.submit_calls if c[0] == "submit_code"][0]
    assert call[4:] == ("u/mapped", 7)


def test_submit_notebook_retries_transient_push_and_records_artifact_provenance(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder)
    api = FakeKaggleApi()
    attempts = 0
    original_push = api.kernels_push

    def push(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary failure in name resolution")
        return original_push(path)

    api.kernels_push = push
    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="m",
        kernel_folder=folder, kernel_ref="tester/kernel", output_file="submission.csv",
        status_fn=lambda ref: "COMPLETE", poll_seconds=0, poll_attempts=1,
        retry_attempts=2, retry_seconds=0,
    )

    assert result.success
    assert attempts == 2
    assert '"artifact_sha256"' in result.raw_status
    assert '"kernel_ref": "tester/fake-kernel"' in result.raw_status


def test_submit_notebook_returns_structured_nonretryable_submit_failure(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder, enable_internet=False)
    api = FakeKaggleApi()

    def denied(**kwargs):
        raise RuntimeError("403 Permission 'kernelSessions.get' denied")

    api.competition_submit_code = denied
    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="m",
        kernel_folder=folder, kernel_ref="tester/kernel", kernel_version=3,
        output_file="submission.csv", status_fn=lambda ref: "COMPLETE",
        poll_seconds=0, poll_attempts=1,
    )

    assert not result.success
    assert 'category=permission' in result.message


def test_submit_notebook_refuses_push_response_without_explicit_version(tmp_path):
    folder = tmp_path / "pkg"
    _write_meta(folder)
    api = FakeKaggleApi()
    api.kernels_push = lambda path: {"ref": "tester/kernel"}

    result = submit_notebook(
        api, competition="rsna-knee-abnormality-detection", message="m",
        kernel_folder=folder, kernel_ref="tester/kernel", output_file="submission.csv",
        status_fn=lambda ref: "COMPLETE", poll_seconds=0, poll_attempts=1,
    )

    assert not result.success
    assert "no kernel version" in result.message
    assert not [c for c in api.submit_calls if c[0] == "submit_code"]
