"""Constant-probability baseline for RSNA Knee.

Local smoke and first kernel seed only — not a competitive model.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .schema import ID_COLUMN, LABELS, SUBMISSION_HEADER


def predict_constant(
    study_ids: list[str], value: float = 0.5
) -> list[dict[str, float | str]]:
    return [{ID_COLUMN: sid, **{lab: value for lab in LABELS}} for sid in study_ids]


def write_submission(path: Path, rows: list[dict[str, float | str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_study_ids(sample_csv: Path) -> list[str]:
    with sample_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            row[ID_COLUMN].strip()
            for row in reader
            if row.get(ID_COLUMN, "").strip()
        ]
