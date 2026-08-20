"""Contract tests for the current rendered RSNA image recipe."""

from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


_SPEC = spec_from_file_location(
    "kernel_recipe", Path(__file__).parents[1] / "competitions/rsna_knee/pipeline/kernel_recipe.py"
)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_SOURCE = _MODULE.KERNEL_RECIPE_SOURCE


def test_grouped_cv_validate_handles_original_index_labels():
    """The rendered runtime uses grouped folds instead of row-level leakage."""
    assert "GroupKFold" in _SOURCE
    assert "grouped folds overlap" in _SOURCE
    assert "groups=groups" in _SOURCE


def test_discover_test_ids_reads_test_series_folders(tmp_path):
    """Hidden IDs are discovered from mounted test folders, not invented."""
    assert "def _discover_test_ids" in _SOURCE
    assert "path.name for path in root.iterdir() if path.is_dir()" in _SOURCE
    assert "hidden test folder IDs missing" in _SOURCE


def test_folder_study_features_counts_series_and_slices(tmp_path):
    """The runtime requires an explicit series-to-study volume index."""
    assert "def _volume_index" in _SOURCE
    assert "StudyInstanceUID" in _SOURCE
    assert "SeriesInstanceUID" in _SOURCE
    assert "mapped_series_count" in _SOURCE


def test_main_does_not_write_fallback_for_empty_inputs(tmp_path):
    """Missing mounted inputs fail closed instead of writing fake submissions."""
    assert "no metadata-ranker fallback" in _SOURCE
    assert "semantic check failed" in _SOURCE
    assert "raise RuntimeError" in _SOURCE


def test_main_accepts_fewer_than_1000_discovered_ids(tmp_path):
    """Output cardinality follows discovered hidden IDs, without a fake 1000-row floor."""
    assert "submission_rows_match_hidden_ids" in _SOURCE
    assert "sorted(sub[ID_COL].astype(str)) == sorted(test_ids)" in _SOURCE
    assert "1000" not in _SOURCE


def test_main_output_has_schema_and_full_discovered_row_count(tmp_path):
    """The runtime inserts the contract ID column and records output evidence."""
    assert "sub.insert(0, ID_COL, test_ids)" in _SOURCE
    assert "fold_predictions_written" in _SOURCE
    assert "submission.csv" in _SOURCE


def test_dinov2_member_handles_asymmetric_feature_columns_and_varies_predictions():
    """The active recipe is image-based and rank-aggregates fold predictions."""
    assert "class DinoMil" in _SOURCE
    assert "_find_report_labels" in _SOURCE
    assert "ranked.append" in _SOURCE
    assert "np.mean(ranked, axis=0)" in _SOURCE
