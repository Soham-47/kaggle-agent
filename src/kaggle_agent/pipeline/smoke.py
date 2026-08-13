"""Local smoke: write a schema-valid submission without training on DICOMs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.pipeline.validate import ValidationResult, validate_submission_csv


@dataclass
class SmokeResult:
    ok: bool
    submission_path: Path | None
    validation: ValidationResult | None = None
    n_studies: int = 0
    errors: list[str] = field(default_factory=list)


def read_study_ids(sample_csv: Path, id_column: str = "StudyInstanceUID") -> list[str]:
    with sample_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"{sample_csv}: missing column {id_column}")
        return [row[id_column].strip() for row in reader if row.get(id_column, "").strip()]


def write_constant_submission(
    path: Path,
    study_ids: list[str],
    *,
    id_column: str,
    labels: list[str],
    value: float = 0.5,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [id_column, *labels]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for sid in study_ids:
            w.writerow([sid, *[f"{value:.6g}" for _ in labels]])
    return path


def run_local_smoke(
    *,
    study_ids: list[str],
    out_path: Path,
    id_column: str,
    labels: list[str],
    value: float = 0.5,
) -> SmokeResult:
    if not study_ids:
        return SmokeResult(
            ok=False,
            submission_path=None,
            errors=["no study ids for smoke"],
        )
    path = write_constant_submission(
        out_path,
        study_ids,
        id_column=id_column,
        labels=labels,
        value=value,
    )
    validation = validate_submission_csv(path, id_column=id_column, labels=labels)
    return SmokeResult(
        ok=validation.ok,
        submission_path=path,
        validation=validation,
        n_studies=len(study_ids),
        errors=list(validation.errors),
    )
