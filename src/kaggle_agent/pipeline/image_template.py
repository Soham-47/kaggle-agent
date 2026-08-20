"""Typed image experiment templates and semantic runtime evidence checks."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any


TEMPLATE_VERSION = "rsna-2d-dino-mil-v1"


@dataclass(frozen=True)
class ImageExperimentContract:
    competition_id: str
    template: str
    source_card_refs: list[str]
    dataset_sources: list[str]
    model_sources: list[str]
    encoder: str
    slice_sampler: str
    pooling: str
    head_labels: list[str]
    target_source: str
    fold_group_key: str
    hidden_id_strategy: str
    inference_aggregation: str
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageExperimentContract":
        required = [
            "competition_id",
            "template",
            "source_card_refs",
            "dataset_sources",
            "model_sources",
            "encoder",
            "slice_sampler",
            "pooling",
            "head_labels",
            "target_source",
            "fold_group_key",
            "hidden_id_strategy",
            "inference_aggregation",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing contract fields: {', '.join(missing)}")
        return cls(
            competition_id=str(data["competition_id"]),
            template=str(data["template"]),
            source_card_refs=_string_list(data["source_card_refs"]),
            dataset_sources=_string_list(data["dataset_sources"]),
            model_sources=_string_list(data["model_sources"]),
            encoder=str(data["encoder"]),
            slice_sampler=str(data["slice_sampler"]),
            pooling=str(data["pooling"]),
            head_labels=_string_list(data["head_labels"]),
            target_source=str(data["target_source"]),
            fold_group_key=str(data["fold_group_key"]),
            hidden_id_strategy=str(data["hidden_id_strategy"]),
            inference_aggregation=str(data["inference_aggregation"]),
            parameters=dict(data.get("parameters") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "competition_id": self.competition_id,
            "template": self.template,
            "source_card_refs": list(self.source_card_refs),
            "dataset_sources": list(self.dataset_sources),
            "model_sources": list(self.model_sources),
            "encoder": self.encoder,
            "slice_sampler": self.slice_sampler,
            "pooling": self.pooling,
            "head_labels": list(self.head_labels),
            "target_source": self.target_source,
            "fold_group_key": self.fold_group_key,
            "hidden_id_strategy": self.hidden_id_strategy,
            "inference_aggregation": self.inference_aggregation,
            "parameters": dict(self.parameters),
        }


def rsna_2d_dino_mil_contract(
    *,
    labels: list[str],
    dataset_sources: list[str],
    model_sources: list[str],
    source_card_refs: list[str],
    parameters: dict[str, Any] | None = None,
) -> ImageExperimentContract:
    return ImageExperimentContract(
        competition_id="rsna_knee",
        template=TEMPLATE_VERSION,
        source_card_refs=source_card_refs,
        dataset_sources=dataset_sources,
        model_sources=model_sources,
        encoder="2d_pretrained_dino",
        slice_sampler="per_series_balanced_slice_sampler",
        pooling="series_mil_attention_pooling",
        head_labels=labels,
        target_source="report_derived_mounted_targets",
        fold_group_key="patient_id",
        hidden_id_strategy="discover_study_ids_from_test_folders",
        inference_aggregation="rank_average_folds",
        parameters=parameters or {"image_size": 336, "folds": 5, "epochs": 1},
    )


def validate_contract(contract: ImageExperimentContract) -> list[str]:
    errors: list[str] = []
    if contract.template != TEMPLATE_VERSION:
        errors.append(f"unsupported template: {contract.template}")
    if contract.competition_id != "rsna_knee":
        errors.append("rsna image template only supports rsna_knee")
    if len(contract.head_labels) != 12:
        errors.append("rsna contract requires 12 labels")
    if not contract.source_card_refs:
        errors.append("source_card_refs required")
    if not contract.dataset_sources:
        errors.append("dataset_sources required")
    if not contract.model_sources:
        errors.append("model_sources required")
    if "dino" not in contract.encoder.lower():
        errors.append("encoder must be DINO-based")
    required_terms = {
        "slice_sampler": "series",
        "pooling": "mil",
        "target_source": "report",
        "hidden_id_strategy": "folder",
        "inference_aggregation": "rank",
    }
    for attr, term in required_terms.items():
        if term not in str(getattr(contract, attr)).lower():
            errors.append(f"{attr} must mention {term}")
    return errors


@dataclass(frozen=True)
class TemplateRenderResult:
    recipe_source: str
    manifest: dict[str, Any]


class ImageExperimentTemplate:
    def render(
        self,
        contract: ImageExperimentContract,
        *,
        resume_manifest: Mapping[str, Any] | None = None,
    ) -> TemplateRenderResult:
        raise NotImplementedError


class Rsna2dDinoMilTemplate(ImageExperimentTemplate):
    """Render a Kaggle-ready recipe constrained by the typed RSNA contract."""

    def render(
        self,
        contract: ImageExperimentContract,
        *,
        resume_manifest: Mapping[str, Any] | None = None,
    ) -> TemplateRenderResult:
        errors = validate_contract(contract)
        if errors:
            raise ValueError("; ".join(errors))
        recipe = (
            _RSNA_RECIPE.replace("__LABELS__", repr(contract.head_labels))
            .replace("__CONTRACT_JSON__", repr(json.dumps(contract.to_dict(), sort_keys=True)))
            .replace(
                "__RESUME_MANIFEST_JSON__",
                repr(json.dumps(dict(resume_manifest or {}), sort_keys=True)),
            )
        )
        rendered_errors = validate_rendered_recipe(recipe)
        if rendered_errors:
            raise ValueError("; ".join(rendered_errors))
        manifest = build_artifact_manifest(contract, recipe_source=recipe)
        return TemplateRenderResult(recipe_source=recipe, manifest=manifest)


def build_artifact_manifest(
    contract: ImageExperimentContract,
    *,
    recipe_source: str,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = contract.to_dict()
    return {
        "template_version": TEMPLATE_VERSION,
        "competition_id": contract.competition_id,
        "source_card_refs": list(contract.source_card_refs),
        "dataset_sources": list(contract.dataset_sources),
        "model_sources": list(contract.model_sources),
        "model_load_path": "/kaggle/input",
        "contract_sha256": _sha256(json.dumps(payload, sort_keys=True)),
        "recipe_sha256": _sha256(recipe_source),
        "fold_outputs": list((evidence or {}).get("fold_outputs") or []),
        "prediction_hashes": list((evidence or {}).get("prediction_hashes") or []),
        "semantic_evidence": evidence or {},
    }


def validate_rendered_recipe(recipe_source: str) -> list[str]:
    """Reject stale sampler names and prove `_sample_paths` uses its contract limit."""
    errors: list[str] = []
    sampler_names = set(re.findall(r"\bSLICES_PER_[A-Z_]+\b", recipe_source))
    stale = sorted(sampler_names - {"SLICES_PER_SERIES"})
    if stale:
        errors.append(f"undefined sampler constants: {', '.join(stale)}")
    try:
        tree = ast.parse(recipe_source)
    except SyntaxError as exc:
        return [*errors, f"rendered recipe is invalid Python: {exc}"]
    sample = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_sample_paths"
        ),
        None,
    )
    sample_source = ast.get_source_segment(recipe_source, sample) if sample is not None else None
    if not sample_source:
        errors.append("rendered recipe is missing _sample_paths")
    elif "min(SLICES_PER_SERIES, len(files))" not in sample_source:
        errors.append("_sample_paths does not use SLICES_PER_SERIES for each series")
    return errors


def validate_resume_artifact(
    expected: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    contract: ImageExperimentContract,
    checkpoint_path: Path,
) -> list[str]:
    """Validate fold checkpoint provenance without mutating the image contract."""
    errors: list[str] = []
    required = (
        "source_experiment",
        "fold",
        "sha256",
        "template_version",
        "model_pin",
        "labels",
        "training_parameters",
        "source_contract_sha256",
    )
    for key in required:
        if key not in expected or key not in sidecar:
            errors.append(f"resume manifest missing {key}")
        elif sidecar[key] != expected[key]:
            errors.append(f"resume sidecar {key} mismatch")
    contract_sha = _sha256(json.dumps(contract.to_dict(), sort_keys=True))
    compatibility = {
        "fold": 0,
        "template_version": contract.template,
        "model_pin": contract.model_sources[0] if contract.model_sources else "",
        "labels": contract.head_labels,
        "training_parameters": contract.parameters,
        "source_contract_sha256": contract_sha,
    }
    for key, value in compatibility.items():
        if expected.get(key) != value:
            errors.append(f"resume manifest {key} is incompatible with current contract")
    if not checkpoint_path.is_file():
        errors.append("resume checkpoint missing")
    elif hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != expected.get("sha256"):
        errors.append("checkpoint SHA-256 mismatch")
    return errors


@dataclass
class SemanticEvidenceResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_image_runtime_evidence(evidence: dict[str, Any]) -> SemanticEvidenceResult:
    errors: list[str] = []
    checks = {
        "mounted_weights_loaded": "mounted weights found and loaded",
        "series_mapping_loaded": "series volumes mapped to study IDs",
        "decoded_non_empty_tensors": "non-empty image tensors decoded",
        "report_labels_joined": "report-label table joined",
        "optimizer_stepped": "model optimizer executes",
        "checkpoints_written": "checkpoints written",
        "fold_predictions_written": "fold predictions written",
        "hidden_ids_from_folders": "hidden test IDs discovered from folders",
        "submission_rows_match_hidden_ids": "submission rows match discovered IDs",
    }
    for key, message in checks.items():
        if not evidence.get(key):
            errors.append(message)
    for key, message in {
        "mapped_series_count": "mapped series count missing",
        "mapped_study_count": "mapped study count missing",
    }.items():
        if int(evidence.get(key) or 0) <= 0:
            errors.append(message)
    if evidence.get("group_overlap"):
        errors.append("grouped splits contain overlapping groups")
    if evidence.get("resumed_folds") != [0]:
        errors.append("fold 0 resume proof missing")
    if evidence.get("newly_trained_folds") != [1, 2, 3, 4]:
        errors.append("newly trained folds must be 1-4")
    if not evidence.get("resume_checkpoint_source"):
        errors.append("resume checkpoint source missing")
    if not evidence.get("resume_checkpoint_sha256"):
        errors.append("resume checkpoint hash missing")
    if int(evidence.get("optimizer_steps") or 0) <= 0:
        errors.append("optimizer step count missing")
    fold_outputs = evidence.get("fold_outputs") or []
    prediction_hashes = evidence.get("prediction_hashes") or []
    if not fold_outputs:
        errors.append("fold outputs missing")
    elif len(fold_outputs) != 5:
        errors.append("five fold outputs required")
    if not prediction_hashes:
        errors.append("prediction hashes missing")
    elif len(prediction_hashes) != 5:
        errors.append("five prediction hashes required")
    return SemanticEvidenceResult(ok=not errors, errors=errors)


def hidden_ids_from_folders(test_root: Path) -> list[str]:
    if not test_root.is_dir():
        return []
    return sorted(path.name for path in test_root.iterdir() if path.is_dir())


def submission_ids_match_folders(
    submission_csv: Path,
    *,
    id_column: str,
    test_root: Path,
) -> bool:
    folder_ids = hidden_ids_from_folders(test_root)
    if not folder_ids or not submission_csv.is_file():
        return False
    with submission_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        ids = sorted(str(row.get(id_column, "")) for row in reader if row.get(id_column))
    return ids == folder_ids


def grouped_splits_have_no_overlap(folds: list[dict[str, list[str]]]) -> bool:
    for fold in folds:
        train = set(fold.get("train_groups") or [])
        valid = set(fold.get("valid_groups") or [])
        if train & valid:
            return False
    return True


def index_study_volumes(
    volume_paths: Iterable[Path],
    mapping_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[Path]], int]:
    """Map series-keyed volume files to study IDs using RSNA metadata."""
    volumes_by_series = {path.stem: path for path in volume_paths}
    volumes_by_study: dict[str, list[Path]] = {}
    mapped_series = 0
    for row in mapping_rows:
        study_id = str(row.get("StudyInstanceUID", "")).strip()
        series_id = str(row.get("SeriesInstanceUID", "")).strip()
        path = volumes_by_series.get(series_id)
        if not study_id or path is None:
            continue
        volumes_by_study.setdefault(study_id, []).append(path)
        mapped_series += 1
    return {
        study_id: sorted(paths)
        for study_id, paths in sorted(volumes_by_study.items())
    }, mapped_series


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError("expected list of strings")
    return [str(item) for item in value if str(item).strip()]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_RSNA_RECIPE = r'''
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from transformers import AutoModel

LABELS = __LABELS__
IMAGE_CONTRACT = json.loads(__CONTRACT_JSON__)
RESUME_MANIFEST = json.loads(__RESUME_MANIFEST_JSON__)
ID_COL = "StudyInstanceUID"
WORK = Path(".")
# This template deliberately has no metadata-ranker fallback.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = int(IMAGE_CONTRACT["parameters"].get("image_size", 336))
FOLDS = int(IMAGE_CONTRACT["parameters"].get("folds", 5))
EPOCHS = int(IMAGE_CONTRACT["parameters"].get("epochs", 1))
SLICES_PER_SERIES = int(IMAGE_CONTRACT["parameters"].get("slices_per_series", 8))


def _competition_root():
    base = Path("/kaggle/input")
    expected = base / "rsna-knee-abnormality-detection"
    candidates = [expected]
    if base.is_dir():
        candidates.extend(sorted(path for path in base.iterdir() if path.is_dir()))
        candidates.extend(sorted(path.parent for path in base.rglob("test.csv")))
    for candidate in candidates:
        if (candidate / "test.csv").is_file() and (candidate / "sample_submission.csv").is_file():
            return candidate
    raise RuntimeError("RSNA competition files are not mounted")


def _study_root(kind):
    for name in (f"{kind}_images", f"{kind}_series", kind):
        root = _competition_root() / name
        if root.is_dir():
            return root
    archive = _competition_root() / f"{kind}_series.zip"
    extracted = Path("/kaggle/working") / f"{kind}_series"
    if archive.is_file():
        if not extracted.is_dir():
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall("/kaggle/working")
        if extracted.is_dir():
            return extracted
    raise RuntimeError(f"{kind} image folders are not mounted")


def _discover_test_ids():
    root = _study_root("test")
    ids = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not ids:
        raise RuntimeError("semantic check failed: hidden test folder IDs missing")
    return ids, root


def _load_dino():
    expected = Path("/kaggle/input/dinov2/pytorch/small/1")
    candidates = [expected] + [path.parent for path in Path("/kaggle/input").rglob("config.json")]
    for path in candidates:
        if (path / "config.json").is_file():
            return AutoModel.from_pretrained(str(path), local_files_only=True).to(DEVICE), path
    raise RuntimeError("semantic check failed: mounted DINOv2 model files missing")


def _find_report_labels(train_volumes):
    competition = _competition_root().resolve()
    volume_ids = set(train_volumes)
    candidates = []
    aliases = (ID_COL, "study_id", "StudyID", "study_uid", "StudyUID")
    paths = sorted(
        Path("/kaggle/input").rglob("*.csv"),
        key=lambda path: (path.name not in {"train_folds.csv", "train_folds_with_pseudo.csv"}, str(path)),
    )
    for path in paths:
        if competition in path.resolve().parents:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        id_column = next((name for name in aliases if name in frame.columns), None)
        if id_column is None or not set(LABELS).issubset(frame.columns):
            continue
        frame = frame.rename(columns={id_column: ID_COL}).copy()
        frame[ID_COL] = frame[ID_COL].astype(str)
        for label in LABELS:
            frame[label] = pd.to_numeric(frame[label], errors="coerce")
        frame = frame.dropna(subset=[ID_COL, *LABELS])
        overlap = int(frame[ID_COL].isin(volume_ids).sum())
        if overlap:
            candidates.append((overlap, path.name == "train_folds.csv", frame, path, id_column))
    if not candidates:
        raise RuntimeError("semantic check failed: no mounted label table joins train volumes")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    overlap, _preferred, frame, path, id_column = candidates[0]
    minimum = max(FOLDS, 5)
    if overlap < minimum:
        raise RuntimeError(f"semantic check failed: report labels join only {overlap} train volumes")
    joined = frame[frame[ID_COL].isin(volume_ids)].reset_index(drop=True)
    return joined, path, {
        "label_source": str(path),
        "label_id_column": id_column,
        "label_join_count": len(joined),
        "label_candidates_considered": len(candidates),
    }


def _sample_paths(study_dir):
    selected = []
    for series in sorted(path for path in study_dir.iterdir() if path.is_dir()):
        files = sorted(series.rglob("*.dcm"))
        if files:
            selected.extend(
                files[index]
                for index in np.linspace(
                    0, len(files) - 1, min(SLICES_PER_SERIES, len(files)), dtype=int
                )
            )
    if not selected:
        files = sorted(study_dir.rglob("*.dcm"))
        selected = [
            files[index]
            for index in np.linspace(
                0, len(files) - 1, min(SLICES_PER_SERIES, len(files)), dtype=int
            )
        ] if files else []
    if not selected:
        raise RuntimeError(f"no DICOM slices for study {study_dir.name}")
    return selected


def _study_tensor(root, study_id):
    images = []
    for path in _sample_paths(root / study_id):
        pixels = pydicom.dcmread(path, force=True).pixel_array.astype("float32")
        if pixels.ndim > 2:
            pixels = pixels[0]
        low, high = np.percentile(pixels, (1, 99))
        pixels = np.clip((pixels - low) / max(high - low, 1e-6), 0.0, 1.0)
        image = torch.from_numpy(pixels)[None, None]
        image = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        images.append(image.repeat(1, 3, 1, 1).squeeze(0))
    if not images:
        raise RuntimeError(f"decoded zero image tensors for {study_id}")
    return torch.stack(images)


def _volume_index():
    volumes_by_series = {path.stem: path for path in Path("/kaggle/input").rglob("*.npz")}
    if not volumes_by_series:
        raise RuntimeError("semantic check failed: mounted train volumes missing")
    candidates = []
    mapping_paths = sorted(
        Path("/kaggle/input").rglob("*.csv"),
        key=lambda path: (path.name != "train_series.csv", str(path)),
    )
    for mapping_path in mapping_paths:
        try:
            with mapping_path.open(newline="", encoding="utf-8") as handle:
                rows = csv.DictReader(handle)
                if not {"StudyInstanceUID", "SeriesInstanceUID"}.issubset(rows.fieldnames or []):
                    continue
                volumes_by_study = {}
                mapped_series = 0
                for row in rows:
                    study_id = str(row.get("StudyInstanceUID", "")).strip()
                    series_id = str(row.get("SeriesInstanceUID", "")).strip()
                    path = volumes_by_series.get(series_id)
                    if not study_id or path is None:
                        continue
                    volumes_by_study.setdefault(study_id, []).append(path)
                    mapped_series += 1
        except (OSError, UnicodeError, csv.Error):
            continue
        if mapped_series:
            candidates.append((mapped_series, mapping_path.name == "train_series.csv", mapping_path, volumes_by_study))
    if not candidates:
        raise RuntimeError(
            "semantic check failed: no StudyInstanceUID/SeriesInstanceUID table maps mounted train volumes"
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    mapped_series, _preferred, mapping_path, volumes_by_study = candidates[0]
    volumes_by_study = {
        study_id: sorted(paths)
        for study_id, paths in sorted(volumes_by_study.items())
    }
    return volumes_by_study, {
        "series_mapping_loaded": True,
        "series_mapping_path": str(mapping_path),
        "mapped_series_count": mapped_series,
        "mapped_study_count": len(volumes_by_study),
    }


def _volume_tensor(volumes, study_id):
    paths = volumes.get(study_id)
    if not paths:
        raise RuntimeError(f"missing mounted train volume for {study_id}")
    images = []
    for path in paths:
        with np.load(path) as archive:
            volume = archive["data"].astype("float32")
        if volume.ndim != 3 or not volume.size:
            raise RuntimeError(f"invalid train volume {path.name} for {study_id}")
        indices = np.linspace(0, len(volume) - 1, min(SLICES_PER_SERIES, len(volume)), dtype=int)
        for index in indices:
            pixels = volume[index]
            low, high = np.percentile(pixels, (1, 99))
            pixels = np.clip((pixels - low) / max(high - low, 1e-6), 0.0, 1.0)
            image = torch.from_numpy(pixels)[None, None]
            image = F.interpolate(image, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
            images.append(image.repeat(1, 3, 1, 1).squeeze(0))
    if not images:
        raise RuntimeError(f"decoded zero train image tensors for {study_id}")
    return torch.stack(images)


class DinoMil(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        hidden = int(encoder.config.hidden_size)
        self.attention = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.head = nn.Linear(hidden, len(LABELS))

    def forward(self, images):
        with torch.no_grad():
            tokens = self.encoder(pixel_values=images).last_hidden_state[:, 0]
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=0)
        return self.head((tokens * weights[:, None]).sum(dim=0, keepdim=True))


def _contract_sha256():
    return hashlib.sha256(json.dumps(IMAGE_CONTRACT, sort_keys=True).encode("utf-8")).hexdigest()


def _validate_resume_values(expected, sidecar):
    required = (
        "source_experiment",
        "fold",
        "sha256",
        "template_version",
        "model_pin",
        "labels",
        "training_parameters",
        "source_contract_sha256",
    )
    for key in required:
        if key not in expected or key not in sidecar or sidecar[key] != expected[key]:
            raise RuntimeError(f"resume checkpoint sidecar {key} mismatch")
    compatible = {
        "fold": 0,
        "template_version": IMAGE_CONTRACT["template"],
        "model_pin": IMAGE_CONTRACT["model_sources"][0],
        "labels": LABELS,
        "training_parameters": IMAGE_CONTRACT["parameters"],
        "source_contract_sha256": _contract_sha256(),
    }
    for key, value in compatible.items():
        if expected.get(key) != value:
            raise RuntimeError(f"resume checkpoint {key} is incompatible")


def _load_resume_artifact():
    expected = dict(RESUME_MANIFEST)
    if not expected:
        raise RuntimeError("resume manifest missing from rendered recipe")
    sidecar_name = expected.get("sidecar_filename", "fold_0_checkpoint.json")
    checkpoint_name = expected.get("checkpoint_filename", "fold_0_checkpoint.pt")
    matches = []
    for sidecar_path in Path("/kaggle/input").rglob(sidecar_name):
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if sidecar.get("source_experiment") == expected.get("source_experiment"):
            matches.append((sidecar_path, sidecar))
    if len(matches) != 1:
        raise RuntimeError(f"expected one resume checkpoint sidecar, found {len(matches)}")
    sidecar_path, sidecar = matches[0]
    checkpoint_path = sidecar_path.parent / checkpoint_name
    _validate_resume_values(expected, sidecar)
    if not checkpoint_path.is_file():
        raise RuntimeError("resume checkpoint file missing")
    actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if actual_sha != expected["sha256"]:
        raise RuntimeError("resume checkpoint SHA-256 mismatch")
    return expected, checkpoint_path


def _load_component_state(component, state, name):
    current = component.state_dict()
    if set(state) != set(current):
        raise RuntimeError(f"resume checkpoint {name} tensor keys mismatch")
    for key, tensor in current.items():
        if tuple(state[key].shape) != tuple(tensor.shape):
            raise RuntimeError(f"resume checkpoint {name}.{key} tensor shape mismatch")
    component.load_state_dict(state, strict=True)


def _predict(model, root, study_ids):
    model.eval()
    rows = []
    with torch.no_grad():
        for study_id in study_ids:
            rows.append(torch.sigmoid(model(_study_tensor(root, study_id).to(DEVICE))).cpu().numpy()[0])
    return np.asarray(rows, dtype=np.float32)


def _train_and_predict(train_volumes, test_root, labels, test_ids):
    fold_column = next((name for name in ("fold", "Fold") if name in labels.columns), None)
    group_col = next((name for name in ("patient_id", "PatientID", "study_id", "StudyID") if name in labels.columns), ID_COL)
    groups = labels[group_col].astype(str).to_numpy()
    if fold_column is not None:
        fold_values = sorted(value for value in labels[fold_column].dropna().unique())
        if len(fold_values) < FOLDS:
            raise RuntimeError("semantic check failed: supplied folds are incomplete")
        splits = []
        for value in fold_values[:FOLDS]:
            valid_idx = np.flatnonzero(labels[fold_column].to_numpy() == value)
            train_idx = np.flatnonzero(labels[fold_column].to_numpy() != value)
            splits.append((train_idx, valid_idx))
        fold_source = fold_column
    else:
        splitter = GroupKFold(n_splits=FOLDS)
        splits = list(splitter.split(labels, groups=groups))
        fold_source = "GroupKFold"
    resume_manifest, resume_path = _load_resume_artifact()
    fold_outputs, hashes, ranked, decoded, steps = [], [], [], 0, 0
    resumed_folds, newly_trained_folds = [], []
    for fold, (train_idx, valid_idx) in enumerate(splits):
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError("semantic check failed: grouped folds overlap")
        encoder, model_path = _load_dino()
        model = DinoMil(encoder).to(DEVICE)
        checkpoint = WORK / f"fold_{fold}_checkpoint.pt"
        if fold == 0:
            saved = torch.load(resume_path, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict) or "attention" not in saved or "head" not in saved:
                raise RuntimeError("resume checkpoint components missing")
            _load_component_state(model.attention, saved["attention"], "attention")
            _load_component_state(model.head, saved["head"], "head")
            shutil.copy2(resume_path, checkpoint)
            resumed_folds.append(fold)
        else:
            optimizer = torch.optim.AdamW(
                [*model.attention.parameters(), *model.head.parameters()], lr=2e-4
            )
            for _ in range(EPOCHS):
                model.train()
                for index in train_idx:
                    row = labels.iloc[index]
                    images = _volume_tensor(train_volumes, str(row[ID_COL])).to(DEVICE)
                    decoded += len(images)
                    target = torch.tensor(
                        row[LABELS].to_numpy(dtype="float32"), device=DEVICE
                    )[None]
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.binary_cross_entropy_with_logits(model(images), target)
                    loss.backward()
                    optimizer.step()
                    steps += 1
            torch.save(
                {
                    "attention": model.attention.state_dict(),
                    "head": model.head.state_dict(),
                    "dino_path": str(model_path),
                },
                checkpoint,
            )
            newly_trained_folds.append(fold)
        frame = pd.DataFrame(_predict(model, test_root, test_ids), columns=LABELS)
        frame.insert(0, ID_COL, test_ids)
        output = WORK / f"fold_{fold}_predictions.csv"
        frame.to_csv(output, index=False)
        fold_outputs.append(str(output))
        hashes.append(hashlib.sha256(output.read_bytes()).hexdigest())
        ranked.append(frame[LABELS].rank(pct=True, method="average").to_numpy())
    if not decoded or not steps:
        raise RuntimeError("semantic check failed: image decode or optimizer proof missing")
    return (
        np.mean(ranked, axis=0),
        fold_outputs,
        hashes,
        decoded,
        steps,
        fold_source,
        resumed_folds,
        newly_trained_folds,
        str(resume_path),
        resume_manifest["sha256"],
    )


# === CUSTOM_INFER START ===
def CUSTOM_INFER(sub, ctx):
    return sub
# === CUSTOM_INFER END ===


test_ids, test_root = _discover_test_ids()
train_volumes, volume_evidence = _volume_index()
labels, report_label_path, label_evidence = _find_report_labels(train_volumes)
(
    prediction,
    fold_outputs,
    prediction_hashes,
    decoded,
    optimizer_steps,
    fold_source,
    resumed_folds,
    newly_trained_folds,
    resume_checkpoint_source,
    resume_checkpoint_sha256,
) = _train_and_predict(train_volumes, test_root, labels, test_ids)
sub = pd.DataFrame(prediction, columns=LABELS)
sub.insert(0, ID_COL, test_ids)
ctx = {"labels": LABELS, "id_col": ID_COL, "work": str(WORK), "contract": IMAGE_CONTRACT}
sub = CUSTOM_INFER(sub, ctx)
sub.to_csv("submission.csv", index=False)
evidence = {
    "mounted_weights_loaded": True,
    "model_load_path": "/kaggle/input/dinov2/pytorch/small/1",
    "decoded_non_empty_tensors": decoded > 0,
    "decoded_tensor_count": decoded,
    **volume_evidence,
    "report_labels_joined": True,
    "report_label_path": str(report_label_path),
    **label_evidence,
    "fold_source": fold_source,
    "group_overlap": False,
    "optimizer_stepped": optimizer_steps > 0,
    "optimizer_steps": optimizer_steps,
    "resumed_folds": resumed_folds,
    "newly_trained_folds": newly_trained_folds,
    "resume_checkpoint_source": resume_checkpoint_source,
    "resume_checkpoint_sha256": resume_checkpoint_sha256,
    "checkpoints_written": all((WORK / f"fold_{fold}_checkpoint.pt").is_file() for fold in range(FOLDS)),
    "fold_predictions_written": len(fold_outputs) == FOLDS,
    "hidden_ids_from_folders": bool(test_ids),
    "submission_rows_match_hidden_ids": sorted(sub[ID_COL].astype(str)) == sorted(test_ids),
    "fold_outputs": fold_outputs,
    "prediction_hashes": prediction_hashes,
}
Path("semantic_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
Path("artifact_manifest.runtime.json").write_text(json.dumps({"template_version": "rsna-2d-dino-mil-v1", "fold_outputs": fold_outputs, "prediction_hashes": prediction_hashes, "semantic_evidence": evidence}, indent=2), encoding="utf-8")
print("wrote verified DINO MIL submission", len(sub))
'''
