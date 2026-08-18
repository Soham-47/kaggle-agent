import numpy as np
import pandas as pd
import pytest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SPEC = spec_from_file_location(
    "kernel_recipe", Path(__file__).parents[1] / "competitions/rsna_knee/pipeline/kernel_recipe.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_grouped_cv_validate_handles_original_index_labels():
    namespace = {}
    exec(_MODULE.KERNEL_RECIPE_SOURCE, namespace)

    ids = [f"study-{i}" for i in range(20)]
    train = pd.DataFrame(
        {
            "StudyInstanceUID": ids,
            "ACL": [i % 2 for i in range(20)],
        },
        index=np.arange(100, 120),
    )
    sft = pd.DataFrame({"feature": np.arange(20, dtype=float)}, index=ids)

    result = namespace["grouped_cv_validate"](train, sft)

    assert result is not None


def _recipe_namespace():
    namespace = {}
    exec(_MODULE.KERNEL_RECIPE_SOURCE, namespace)
    return namespace


def test_discover_test_ids_reads_test_series_folders(tmp_path):
    namespace = _recipe_namespace()
    root = tmp_path
    (root / "test_series" / "study-a" / "series-1").mkdir(parents=True)
    (root / "test_series" / "study-a" / "series-2").mkdir()
    (root / "test_series" / "study-b" / "series-1").mkdir(parents=True)
    for path in root.glob("test_series/*/*"):
        (path / "slice.dcm").touch()
    test_df = pd.DataFrame({"StudyInstanceUID": ["study-a", "stale"]})

    result = namespace["discover_test_ids"](root, test_df)

    assert result["StudyInstanceUID"].tolist() == ["study-a", "study-b"]


def test_folder_study_features_counts_series_and_slices(tmp_path):
    namespace = _recipe_namespace()
    root = tmp_path
    for series, slices in {"series-1": 2, "series-2": 3}.items():
        folder = root / "test_series" / "study-a" / series
        folder.mkdir(parents=True)
        for i in range(slices):
            (folder / f"{i}.dcm").touch()

    result = namespace["folder_study_features"](root)

    assert result.loc["study-a", "n_series"] == 2
    assert result.loc["study-a", "n_slices"] == 5


def test_main_does_not_write_fallback_for_empty_inputs(tmp_path, monkeypatch):
    namespace = _recipe_namespace()
    namespace["train"] = pd.DataFrame()
    namespace["test"] = pd.DataFrame()
    namespace["sample"] = pd.DataFrame({"StudyInstanceUID": ["study-a"]})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="discovered test IDs"):
        namespace["main"]()

    assert not (tmp_path / "submission.csv").exists()


def test_main_rejects_incomplete_discovery_before_output(tmp_path, monkeypatch):
    namespace = _recipe_namespace()
    namespace["train"] = pd.DataFrame({"StudyInstanceUID": ["train-a"]})
    namespace["test"] = pd.DataFrame({"StudyInstanceUID": ["test-a"]})
    namespace["sample"] = pd.DataFrame()
    namespace["build_dinov2_member"] = lambda *args: pd.DataFrame(index=["test-a"])
    namespace["build_radimagenet_member"] = lambda *args: pd.DataFrame(index=["test-a"])
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="discovered test IDs"):
        namespace["main"]()

    assert not (tmp_path / "submission.csv").exists()


def test_main_output_has_schema_and_full_discovered_row_count(tmp_path, monkeypatch):
    namespace = _recipe_namespace()
    ids = [f"study-{i}" for i in range(1000)]
    namespace["train"] = pd.DataFrame({"StudyInstanceUID": ["train-a"]})
    namespace["test"] = pd.DataFrame({"StudyInstanceUID": ids})
    namespace["sample"] = pd.DataFrame()
    predictions = pd.DataFrame(0.5, index=ids, columns=namespace["LABELS"])
    namespace["build_dinov2_member"] = lambda *args: predictions
    namespace["build_radimagenet_member"] = lambda *args: predictions
    monkeypatch.chdir(tmp_path)

    result = namespace["main"]()

    assert result.shape == (1000, 13)
    assert result.columns.tolist() == [namespace["ID_COL"]] + namespace["LABELS"]
    assert pd.read_csv("submission.csv").shape == (1000, 13)


def test_dinov2_member_handles_asymmetric_feature_columns_and_varies_predictions():
    namespace = _recipe_namespace()
    train = pd.DataFrame({
        "StudyInstanceUID": [f"train-{i}" for i in range(12)],
        "Report": ["ACL tear" if i % 2 else "normal" for i in range(12)],
    })
    test = pd.DataFrame({"StudyInstanceUID": ["test-a", "test-b"]})
    sft = pd.DataFrame({
        "shared": np.arange(12, dtype=float),
        "train_only": np.arange(12, dtype=float),
    }, index=train["StudyInstanceUID"])
    sfe = pd.DataFrame({
        "shared": [0.0, 100.0],
        "test_only": [0.0, 100.0],
    }, index=test["StudyInstanceUID"])

    result = namespace["build_dinov2_member"](train, test, sft, sfe)

    assert result.shape == (2, 12)
    assert result["ACL"].nunique() > 1
