import json
from pathlib import Path

from fakes import FakeKaggleApi
from helpers import write_kernel_fixture_data
from kaggle_agent.config import load_competition
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.paths import repo_root
from kaggle_agent.train.kernel_runner import run_kernel_phase
from kaggle_agent.train.notebook_builder import (
    build_baseline_notebook,
    write_kernel_package,
)


def test_build_notebook_has_baker_and_submission():
    nb = build_baseline_notebook(
        competition_slug="rsna-knee-abnormality-detection",
        labels=["ACL", "Baker's", "Fracture"],
        study_ids=["s1", "s2"],
    )
    assert nb["nbformat"] == 4
    src = "".join(nb["cells"][1]["source"])
    assert "Baker's" in src
    assert "submission.csv" in src
    assert "kernel-smoke" not in src


def _root_with_data(tmp_path: Path) -> Path:
    root = tmp_path / "ka"
    real = repo_root()
    import shutil

    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    write_kernel_fixture_data(root)
    return root


def test_write_kernel_package(tmp_path: Path):
    root = _root_with_data(tmp_path)
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(
        comp, root=root, username="tester", exp_id="20260101-test"
    )
    assert pkg.notebook_path.is_file()
    assert pkg.metadata_path.is_file()
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta["language"] == "python"
    assert meta["kernel_type"] == "notebook"
    assert meta["competition_sources"] == ["rsna-knee-abnormality-detection"]
    assert meta["id"].startswith("tester/")
    assert "Baker's" in pkg.notebook_path.read_text(encoding="utf-8")
    notebook = pkg.notebook_path.read_text(encoding="utf-8")
    assert "EXPERIMENT_MANIFEST" in notebook
    assert "20260101-test" in notebook


def test_run_kernel_phase_local_only(tmp_path: Path):
    root = _root_with_data(tmp_path)
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="x")
    run = run_kernel_phase(None, pkg, push=False)
    assert run.ok
    assert run.pushed is False
    assert run.status == "local_only"


def test_run_kernel_phase_push_fake(tmp_path: Path):
    root = _root_with_data(tmp_path)
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="y")
    client = KaggleClient(api=FakeKaggleApi()).connect()
    out = pkg.folder / "output"
    run = run_kernel_phase(client, pkg, push=True, pull_output_dir=out, root=root)
    assert run.ok
    assert run.pushed is True
    assert run.status == "COMPLETE"
    assert (out / "submission.csv").is_file()


def test_run_kernel_phase_rejects_recorded_duplicate(tmp_path: Path):
    root = _root_with_data(tmp_path)
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="duplicate")
    client = KaggleClient(api=FakeKaggleApi()).connect()

    first = run_kernel_phase(client, pkg, push=True, root=root)
    second = run_kernel_phase(client, pkg, push=True, root=root)

    assert first.ok
    assert second.ok is False
    assert any("duplicate kernel package" in error for error in second.errors)


def test_write_kernel_package_uses_methods_json(tmp_path: Path):
    root = _root_with_data(tmp_path)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "methods.json").write_text(
        '{"dataset_sources": ["owner/public-weights"], "model_sources": []}',
        encoding="utf-8",
    )
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="methods")
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["owner/public-weights"]
    assert meta["model_sources"] == []
    assert pkg.notebook_path.name == "agent_baseline.ipynb"


def test_write_kernel_package_carries_image_artifact_manifest(tmp_path: Path):
    root = _root_with_data(tmp_path)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True)
    (pipe / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "template_version": "image-2d-dino-mil-v1",
                "source_card_refs": ["source-dino"],
                "model_sources": ["owner/dinov2/PyTorch/base/1"],
                "prediction_hashes": ["abc"],
            }
        ),
        encoding="utf-8",
    )
    (pipe / "image_contract.json").write_text('{"template": "image-2d-dino-mil-v1"}', encoding="utf-8")

    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="image-manifest")

    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    artifact = meta["experiment_manifest"]["artifact_manifest"]
    assert artifact["template_version"] == "image-2d-dino-mil-v1"
    assert (pkg.folder / "artifact_manifest.json").is_file()
    assert (pkg.folder / "image_contract.json").is_file()


def test_write_kernel_package_attaches_resume_dataset_outside_image_contract(tmp_path: Path):
    root = _root_with_data(tmp_path)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True)
    contract = '{"template": "image-2d-dino-mil-v1", "dataset_sources": ["owner/base"]}'
    (pipe / "image_contract.json").write_text(contract, encoding="utf-8")
    (pipe / "methods.json").write_text(
        '{"dataset_sources": ["owner/base"], "model_sources": []}', encoding="utf-8"
    )
    (pipe / "resume_manifest.json").write_text(
        json.dumps(
            {
                "dataset_source": "owner/private-fold-zero",
                "checkpoint_filename": "fold_0_checkpoint.pt",
                "sidecar_filename": "fold_0_checkpoint.json",
            }
        ),
        encoding="utf-8",
    )
    comp = load_competition("rsna_knee", root)

    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="resume")

    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta["dataset_sources"] == ["owner/private-fold-zero", "owner/base"]
    assert (pkg.folder / "resume_manifest.json").is_file()
    assert (pipe / "image_contract.json").read_text(encoding="utf-8") == contract


def test_resume_dataset_keeps_a_reserved_slot_when_cards_list_six_sources(tmp_path: Path):
    root = _root_with_data(tmp_path)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True)
    sources = [f"owner/data-{i}" for i in range(6)]
    (pipe / "methods.json").write_text(
        json.dumps({"dataset_sources": sources, "model_sources": []}), encoding="utf-8"
    )
    (pipe / "resume_manifest.json").write_text(
        json.dumps({"dataset_source": "owner/private-fold-zero"}), encoding="utf-8"
    )
    comp = load_competition("rsna_knee", root)

    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="resume-slots")

    datasets = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))["dataset_sources"]
    assert datasets == ["owner/private-fold-zero", *sources[:5]]


def test_write_kernel_package_requires_study_ids(tmp_path: Path):
    root = tmp_path / "ka"
    import shutil

    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    comp = load_competition("rsna_knee", root)
    import pytest

    with pytest.raises(ValueError):
        write_kernel_package(comp, root=root, username="tester", exp_id="noid")
