import json
from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.config import load_competition
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.paths import repo_root
from kaggle_agent.train.kernel_runner import run_kernel_phase
from kaggle_agent.train.notebook_builder import (
    build_rsna_baseline_notebook,
    write_kernel_package,
)


def test_build_notebook_has_baker_and_submission():
    nb = build_rsna_baseline_notebook(
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
    shutil.copytree(real / "data", root / "data")
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
