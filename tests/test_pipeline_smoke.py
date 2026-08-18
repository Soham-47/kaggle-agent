from pathlib import Path

from kaggle_agent.pipeline.smoke import run_local_smoke, write_constant_submission
from kaggle_agent.pipeline.validate import validate_submission_csv
from kaggle_agent.train.local_smoke import run_competition_smoke
from kaggle_agent.config import load_competition
from kaggle_agent.paths import repo_root
from kaggle_agent.code.workspace import ensure_pipeline_ready


LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def test_validate_accepts_good_csv(tmp_path: Path):
    path = write_constant_submission(
        tmp_path / "s.csv",
        ["study-a", "study-b"],
        id_column="StudyInstanceUID",
        labels=LABELS,
        value=0.5,
    )
    r = validate_submission_csv(path, id_column="StudyInstanceUID", labels=LABELS)
    assert r.ok
    assert r.n_rows == 2


def test_validate_rejects_bad_header(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("StudyInstanceUID,ACL\n1,0.5\n", encoding="utf-8")
    r = validate_submission_csv(path, id_column="StudyInstanceUID", labels=LABELS)
    assert not r.ok
    assert any("header" in e for e in r.errors)


def test_validate_rejects_out_of_range(tmp_path: Path):
    path = write_constant_submission(
        tmp_path / "s.csv",
        ["s1"],
        id_column="StudyInstanceUID",
        labels=LABELS,
        value=1.5,
    )
    r = validate_submission_csv(path, id_column="StudyInstanceUID", labels=LABELS)
    assert not r.ok


def test_validate_rejects_constant_kernel_output(tmp_path: Path):
    path = write_constant_submission(
        tmp_path / "kernel.csv",
        ["s1", "s2"],
        id_column="StudyInstanceUID",
        labels=LABELS,
        value=0.525,
    )

    r = validate_submission_csv(
        path,
        id_column="StudyInstanceUID",
        labels=LABELS,
        require_prediction_variation=True,
    )

    assert not r.ok
    assert any("constant" in error for error in r.errors)


def test_validate_rejects_output_shorter_than_required_minimum(tmp_path: Path):
    path = write_constant_submission(
        tmp_path / "short.csv",
        ["s1", "s2", "s3"],
        id_column="StudyInstanceUID",
        labels=LABELS,
        value=0.5,
    )

    r = validate_submission_csv(
        path,
        id_column="StudyInstanceUID",
        labels=LABELS,
        require_min_rows=1000,
    )

    assert not r.ok
    assert any("1000" in error for error in r.errors)


def test_run_local_smoke(tmp_path: Path):
    out = run_local_smoke(
        study_ids=["a", "b", "c"],
        out_path=tmp_path / "smoke.csv",
        id_column="StudyInstanceUID",
        labels=LABELS,
    )
    assert out.ok
    assert out.submission_path and out.submission_path.is_file()
    # Baker's must be in header
    header = out.submission_path.read_text(encoding="utf-8").splitlines()[0]
    assert "Baker's" in header


def test_competition_smoke_and_workspace():
    root = repo_root()
    comp = load_competition("rsna_knee", root)
    check = ensure_pipeline_ready(root / "competitions" / "rsna_knee")
    assert check.ok, check.missing

    outcome = run_competition_smoke(comp, root=root, exp_id="test-smoke")
    assert outcome.ok
    assert outcome.smoke and outcome.smoke.submission_path
    assert outcome.smoke.submission_path.is_file()


def test_competition_smoke_reports_cv_auc(tmp_path: Path, monkeypatch):
    import shutil

    from kaggle_agent.paths import repo_root

    root = tmp_path / "kaggle-agent"
    workspace = root / "competitions" / "rsna_knee"
    shutil.copytree(
        repo_root() / "competitions" / "rsna_knee" / "pipeline",
        workspace / "pipeline",
    )
    data = root / "data"
    data.mkdir(parents=True)
    (data / "sample_submission.csv").write_text(
        "StudyInstanceUID," + ",".join(LABELS) + "\ns1," + ",".join(["0.5"] * 12) + "\n",
        encoding="utf-8",
    )
    (data / "train.csv").write_text("train\n", encoding="utf-8")
    (data / "train_series.csv").write_text("series\n", encoding="utf-8")
    monkeypatch.setattr(
        "kaggle_agent.pipeline.cv.evaluate_ranker_cv",
        lambda *args, **kwargs: {"macro_auc": 0.5312, "n_studies": 10, "folds": 5},
    )
    comp = load_competition("rsna_knee", repo_root())
    outcome = run_competition_smoke(comp, root=root, exp_id="cv-smoke")
    assert outcome.ok
    assert outcome.cv_auc == 0.5312
