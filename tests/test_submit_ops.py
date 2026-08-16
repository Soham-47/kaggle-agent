"""submit_ops notebook-submit: submit the completed train kernel output."""

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


def test_submit_notebook_no_repush_submits_completed_kernel(tmp_path):
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
    assert pushes == []
    subs = [c for c in api.submit_calls if c and c[0] == "submit_code"]
    assert len(subs) == 1
    _, file_name, message, competition, kernel, version = subs[0]
    assert file_name == "submission.csv"
    assert kernel == "tester/kernel"
    assert version is None
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"