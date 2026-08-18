import numpy as np
import pandas as pd
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
