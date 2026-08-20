import json
from pathlib import Path

import pytest

from kaggle_agent.pipeline.image_template import (
    Image2dDinoMilTemplate,
    index_study_volumes,
    grouped_splits_have_no_overlap,
    hidden_ids_from_folders,
    image_2d_dino_mil_contract,
    submission_ids_match_folders,
    validate_contract,
    validate_image_runtime_evidence,
    validate_rendered_recipe,
    validate_resume_artifact,
)


LABELS = [f"label_{i}" for i in range(12)]


def _contract():
    return image_2d_dino_mil_contract(
        competition_id="example",
        labels=LABELS,
        dataset_sources=["owner/example-labels"],
        model_sources=["owner/dinov2/PyTorch/base/1"],
        source_card_refs=["source-dino", "source-report-labels"],
    )


def test_rsna_contract_requires_pins_and_12_labels():
    contract = _contract()

    assert validate_contract(contract) == []
    assert contract.encoder == "2d_pretrained_dino"
    assert contract.pooling == "series_mil_attention_pooling"


def test_rsna_contract_rejects_missing_model_pin():
    contract = image_2d_dino_mil_contract(
        competition_id="example",
        labels=LABELS,
        dataset_sources=["owner/labels"],
        model_sources=[],
        source_card_refs=["source-dino"],
    )

    assert "model_sources required" in validate_contract(contract)


def test_template_renders_kaggle_recipe_and_manifest():
    rendered = Image2dDinoMilTemplate().render(_contract())

    assert "submission.csv" in rendered.recipe_source
    assert "CUSTOM_INFER(sub, ctx)" in rendered.recipe_source
    assert "metadata-ranker fallback" in rendered.recipe_source
    assert "AutoModel.from_pretrained" in rendered.recipe_source
    assert "pydicom.dcmread" in rendered.recipe_source
    assert "GroupKFold" in rendered.recipe_source
    assert "optimizer.step()" in rendered.recipe_source
    assert "torch.save" in rendered.recipe_source
    assert "(candidate / \"test.csv\").is_file()" in rendered.recipe_source
    assert 'f"{kind}_series.zip"' in rendered.recipe_source
    assert "zipfile.ZipFile" in rendered.recipe_source
    assert "with np.load(path) as archive" in rendered.recipe_source
    assert 'archive["data"]' in rendered.recipe_source
    assert "train_folds.csv" in rendered.recipe_source
    assert "train_series.csv" in rendered.recipe_source
    assert "SERIES_ID_COL" in rendered.recipe_source
    assert "series_mapping_loaded" in rendered.recipe_source
    assert "label_join_count" in rendered.recipe_source
    assert "supplied folds are incomplete" in rendered.recipe_source
    assert "fold_source" in rendered.recipe_source
    assert "hash(tuple(row))" not in rendered.recipe_source
    assert "np.linspace(0.05, 0.95" not in rendered.recipe_source
    assert rendered.manifest["template_version"] == "image-2d-dino-mil-v1"
    assert rendered.manifest["dataset_sources"] == ["owner/example-labels"]
    assert rendered.manifest["model_sources"] == ["owner/dinov2/PyTorch/base/1"]
    assert len(rendered.manifest["contract_sha256"]) == 64


def test_rendered_recipe_uses_only_configured_per_series_sampler_limit():
    rendered = Image2dDinoMilTemplate().render(_contract())

    assert "SLICES_PER_STUDY" not in rendered.recipe_source
    assert validate_rendered_recipe(rendered.recipe_source) == []
    sample_paths = rendered.recipe_source.split("def _sample_paths", 1)[1].split(
        "def _study_tensor", 1
    )[0]
    assert "min(SLICES_PER_SERIES, len(files))" in sample_paths


def test_rendered_recipe_rejects_stale_or_undefined_sampler_constants():
    stale = "SLICES_PER_SERIES = 8\ndef _sample_paths(files):\n    return files[:SLICES_PER_STUDY]\n"

    errors = validate_rendered_recipe(stale)

    assert any("SLICES_PER_STUDY" in error for error in errors)


def test_template_rejects_invalid_contract():
    contract = _contract()
    bad = contract.__class__(**{**contract.to_dict(), "head_labels": []})

    with pytest.raises(ValueError, match="at least one label"):
        Image2dDinoMilTemplate().render(bad)


def test_series_volumes_are_indexed_by_study_uid(tmp_path: Path):
    first = tmp_path / "series-a.npz"
    second = tmp_path / "series-b.npz"
    orphan = tmp_path / "series-orphan.npz"
    rows = [
        {"id": "study-1", "series_id": "series-a"},
        {"id": "study-1", "series_id": "series-b"},
        {"id": "study-2", "series_id": "missing-series"},
    ]

    indexed, mapped_series = index_study_volumes([first, second, orphan], rows)

    assert indexed == {"study-1": [first, second]}
    assert mapped_series == 2


def test_hidden_ids_and_submission_matching(tmp_path: Path):
    test_root = tmp_path / "test_images"
    (test_root / "study-b").mkdir(parents=True)
    (test_root / "study-a").mkdir()
    submission = tmp_path / "submission.csv"
    submission.write_text("id,label\nstudy-a,0.1\nstudy-b,0.2\n", encoding="utf-8")

    assert hidden_ids_from_folders(test_root) == ["study-a", "study-b"]
    assert submission_ids_match_folders(
        submission,
        id_column="id",
        test_root=test_root,
    )


def test_grouped_splits_have_no_overlap():
    assert grouped_splits_have_no_overlap(
        [{"train_groups": ["p1", "p2"], "valid_groups": ["p3"]}]
    )
    assert not grouped_splits_have_no_overlap(
        [{"train_groups": ["p1", "p2"], "valid_groups": ["p2"]}]
    )


def test_semantic_evidence_requires_real_image_training_proofs():
    result = validate_image_runtime_evidence(
        {
            "mounted_weights_loaded": True,
            "series_mapping_loaded": True,
            "mapped_series_count": 2,
            "mapped_study_count": 1,
            "decoded_non_empty_tensors": True,
            "report_labels_joined": True,
            "group_overlap": False,
            "optimizer_stepped": True,
            "checkpoints_written": True,
            "fold_predictions_written": True,
            "hidden_ids_from_folders": True,
            "submission_rows_match_hidden_ids": True,
            "resumed_folds": [0],
            "newly_trained_folds": [1, 2, 3, 4],
            "resume_checkpoint_source": "/kaggle/input/resume/fold_0_checkpoint.pt",
            "resume_checkpoint_sha256": "abc",
            "optimizer_steps": 4,
            "fold_outputs": [f"fold_{fold}_predictions.csv" for fold in range(5)],
            "prediction_hashes": [f"hash-{fold}" for fold in range(5)],
        }
    )
    assert result.ok

    failed = validate_image_runtime_evidence({"mounted_weights_loaded": True})
    assert not failed.ok
    assert "non-empty image tensors decoded" in failed.errors
    assert "prediction hashes missing" in failed.errors


def _resume_manifest(contract, checkpoint_sha: str) -> dict:
    import hashlib

    contract_sha = hashlib.sha256(
        json.dumps(contract.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "source_experiment": "20260819-123221",
        "fold": 0,
        "sha256": checkpoint_sha,
        "template_version": "image-2d-dino-mil-v1",
        "model_pin": contract.model_sources[0],
        "labels": contract.head_labels,
        "training_parameters": contract.parameters,
        "source_contract_sha256": contract_sha,
    }


def test_resume_artifact_accepts_matching_manifest_and_checkpoint(tmp_path: Path):
    import hashlib

    checkpoint = tmp_path / "fold_0_checkpoint.pt"
    checkpoint.write_bytes(b"trusted checkpoint")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = _contract()
    expected = _resume_manifest(contract, sha)

    assert validate_resume_artifact(expected, dict(expected), contract, checkpoint) == []


def test_resume_manifest_is_embedded_outside_contract_hash(tmp_path: Path):
    import hashlib

    checkpoint = tmp_path / "fold_0_checkpoint.pt"
    checkpoint.write_bytes(b"trusted checkpoint")
    contract = _contract()
    expected = _resume_manifest(
        contract, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )

    without_resume = Image2dDinoMilTemplate().render(contract)
    with_resume = Image2dDinoMilTemplate().render(contract, resume_manifest=expected)

    assert with_resume.manifest["contract_sha256"] == without_resume.manifest["contract_sha256"]
    assert "20260819-123221" in with_resume.recipe_source
    assert 'Path("resume_manifest.json")' not in with_resume.recipe_source


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"fold": 1}, "fold"),
        ({"source_contract_sha256": "bad"}, "source_contract_sha256"),
        ({"model_pin": "owner/other/model/1"}, "model_pin"),
        ({"labels": ["wrong"]}, "labels"),
    ],
)
def test_resume_artifact_rejects_incompatible_sidecar(
    tmp_path: Path, mutation: dict, message: str
):
    import hashlib

    checkpoint = tmp_path / "fold_0_checkpoint.pt"
    checkpoint.write_bytes(b"trusted checkpoint")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract = _contract()
    expected = _resume_manifest(contract, sha)
    actual = {**expected, **mutation}

    errors = validate_resume_artifact(expected, actual, contract, checkpoint)

    assert any(message in error for error in errors)


def test_resume_artifact_rejects_checkpoint_sha_mismatch(tmp_path: Path):
    checkpoint = tmp_path / "fold_0_checkpoint.pt"
    checkpoint.write_bytes(b"tampered checkpoint")
    contract = _contract()
    expected = _resume_manifest(contract, "0" * 64)

    errors = validate_resume_artifact(expected, dict(expected), contract, checkpoint)

    assert "checkpoint SHA-256 mismatch" in errors


def test_contract_round_trips_as_json():
    contract = _contract()
    encoded = json.loads(json.dumps(contract.to_dict()))

    assert contract.__class__.from_dict(encoded) == contract
