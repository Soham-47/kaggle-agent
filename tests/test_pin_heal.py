"""Drop junk Kaggle attach pins and retry push (real heal loop)."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.heal.pins import (
    apply_pin_heal,
    is_pin_error,
    sanitize_datasets,
    sanitize_models,
    sanitize_methods_payload,
)


def test_sanitize_drops_junk_keeps_full_pin():
    models = sanitize_models(
        [
            "dataset/model",
            "metaresearch/dinov2",
            "dinov2/pytorch",
            "wguesdon/rsna-knee-dinov2-at-meniscus-resolution",
            "metaresearch/dinov2/PyTorch/small/1",
        ]
    )
    assert models == ["metaresearch/dinov2/PyTorch/small/1"]
    datasets = sanitize_datasets(
        ["dataset/model", "wguesdon/rsna-knee-llm-report-labels-opus", "metaresearch/dinov2"]
    )
    assert datasets == ["sohamgawd47foden/rsna-knee-dinov2-vits14"]


def test_sanitize_methods_deduplicates_steps():
    payload = sanitize_methods_payload(
        {"implement_steps": ["same step", "same step", "bad Our score=0.5"]}
    )
    assert payload["implement_steps"] == ["same step"]


def test_sanitize_methods_wraps_string_step():
    payload = sanitize_methods_payload({"implement_steps": "one step"})
    assert payload["implement_steps"] == ["one step"]


def test_should_not_wait_approve_when_already_approved_or_pin_error():
    from kaggle_agent.heal.pins import should_wait_approve

    assert (
        should_wait_approve(
            validate_ok=True,
            submit_ok=False,
            dry_run=False,
            assume_approved=True,
            errors=["kernel:push: version-number"],
        )
        is False
    )
    assert (
        should_wait_approve(
            validate_ok=True,
            submit_ok=False,
            dry_run=False,
            assume_approved=False,
            errors=["kernel:push: Model instance version must be specified"],
        )
        is False
    )
    assert (
        should_wait_approve(
            validate_ok=True,
            submit_ok=False,
            dry_run=False,
            assume_approved=False,
            errors=[],
        )
        is True
    )


def test_is_pin_error():
    assert is_pin_error(
        "kernels_push failed: Model instance version must be specified "
        "in the form of '{owner}/{model-slug}/{framework}/{instance-slug}/{version-number}'"
    )
    assert is_pin_error("version-number") is True
    assert is_pin_error("quota exceeded") is False


def test_apply_pin_heal_rewrites_methods_and_metadata(tmp_path: Path):
    workspace = tmp_path / "comp"
    pipe = workspace / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "methods.json").write_text(
        json.dumps(
            {
                "dataset_sources": ["dataset/model", "owner/weights"],
                "model_sources": ["metaresearch/dinov2", "metaresearch/dinov2/PyTorch/small/1"],
                "implement_steps": ["attach public weights"],
            }
        ),
        encoding="utf-8",
    )
    folder = tmp_path / "nb"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "dataset_sources": ["dataset/model"],
                "model_sources": ["dinov2/pytorch"],
            }
        ),
        encoding="utf-8",
    )
    out = apply_pin_heal(workspace, folder)
    assert out["changed"] is True
    methods = json.loads((pipe / "methods.json").read_text(encoding="utf-8"))
    assert methods["dataset_sources"] == ["owner/weights"]
    assert methods["model_sources"] == ["metaresearch/dinov2/PyTorch/small/1"]
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["model_sources"] == ["metaresearch/dinov2/PyTorch/small/1"]
    assert "dataset/model" not in meta["dataset_sources"]


def test_kernel_push_retries_after_pin_error(tmp_path: Path):
    from fakes import FakeKaggleApi
    from helpers import copy_min_workspace
    from kaggle_agent.config import load_competition
    from kaggle_agent.kaggle_api import KaggleClient
    from kaggle_agent.train.kernel_runner import run_kernel_phase
    from kaggle_agent.train.notebook_builder import write_kernel_package

    real = Path(__file__).resolve().parents[1]
    root = tmp_path / "ka"
    copy_min_workspace(root, real)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    (pipe / "methods.json").write_text(
        json.dumps(
            {
                "dataset_sources": ["dataset/model"],
                "model_sources": ["metaresearch/dinov2"],
            }
        ),
        encoding="utf-8",
    )
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="heal1")
    # builder already strips junk; put junk back to force first-push fail
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    meta["model_sources"] = ["metaresearch/dinov2"]
    pkg.metadata_path.write_text(json.dumps(meta), encoding="utf-8")

    class PinThenOk(FakeKaggleApi):
        def kernels_push(self, folder, timeout=None, acc=None):
            meta_p = Path(folder) / "kernel-metadata.json"
            models = json.loads(meta_p.read_text(encoding="utf-8")).get("model_sources") or []
            if any(m.count("/") < 3 for m in models):
                raise RuntimeError(
                    "Model instance version must be specified in the form of "
                    "'{owner}/{model-slug}/{framework}/{instance-slug}/{version-number}'"
                )
            return super().kernels_push(folder, timeout=timeout)

    api = PinThenOk()
    client = KaggleClient(api=api).connect()
    run = run_kernel_phase(
        client, pkg, push=True, root=root, competition=comp.slug, exp_id="heal1"
    )
    assert run.pushed is True
    assert run.ok is True
    assert run.errors == []
