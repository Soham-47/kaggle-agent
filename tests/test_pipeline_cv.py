from pathlib import Path

from kaggle_agent.pipeline.cv import evaluate_ranker_cv, grouped_folds, macro_auc


LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def test_macro_auc_orders_scores():
    assert macro_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert macro_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0


def test_macro_auc_handles_ties_and_missing_class():
    assert macro_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert macro_auc([1, 1], [0.1, 0.2]) == 0.5


def test_grouped_folds_do_not_split_duplicate_ids():
    ids = ["a", "a", "b", "c", "d", "e"]
    folds = grouped_folds(ids, n_folds=3, seed=7)
    seen = []
    for train, valid in folds:
        assert not set(train) & set(valid)
        assert not ({"a"} <= set(train) and {"a"} <= set(valid))
        seen.extend(valid)
    assert sorted(seen) == list(range(len(ids)))


def _write_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    data.mkdir(parents=True)
    train = data / "train.csv"
    series = data / "train_series.csv"
    rows = ["StudyInstanceUID," + ",".join(LABELS)]
    for i in range(10):
        values = ",".join(str((i + j) % 2) for j in range(len(LABELS)))
        rows.append(f"s{i},{values}")
    train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    series_rows = [
        "StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane"
    ]
    for i in range(10):
        plane = "Axial" if i % 2 else "Sagittal"
        series_rows.append(f"s{i},series-{i},1,1,{plane}")
    series.write_text("\n".join(series_rows) + "\n", encoding="utf-8")
    return train, series


def test_evaluate_ranker_cv_returns_out_of_fold_metrics(tmp_path: Path):
    import shutil

    from kaggle_agent.paths import repo_root

    train, series = _write_fixture(tmp_path)
    workspace = tmp_path / "competitions" / "rsna_knee"
    shutil.copytree(repo_root() / "competitions" / "rsna_knee" / "pipeline", workspace / "pipeline")
    result = evaluate_ranker_cv(train, series, workspace, n_folds=5, seed=0)
    assert result is not None
    assert result["n_studies"] == 10
    assert result["folds"] == 5
    assert set(result["per_label"]) == set(LABELS)
    assert 0.0 <= result["macro_auc"] <= 1.0


def test_evaluate_ranker_cv_skips_missing_inputs(tmp_path: Path):
    assert evaluate_ranker_cv(tmp_path / "train.csv", tmp_path / "series.csv", tmp_path) is None
