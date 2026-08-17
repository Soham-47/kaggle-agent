"""Validate Kaggle submission CSV shape against competition labels."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    ok: bool
    path: Path
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors) or "invalid submission")


def validate_submission_csv(
    path: Path,
    *,
    id_column: str,
    labels: list[str],
    require_rows: bool = True,
    require_prediction_variation: bool = False,
) -> ValidationResult:
    result = ValidationResult(ok=True, path=path)
    if not path.is_file():
        result.fail(f"missing file: {path}")
        return result

    expected = [id_column, *labels]
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            result.fail("empty CSV / no header")
            return result

        header = list(reader.fieldnames)
        if header != expected:
            result.fail(f"header mismatch\n  expected: {expected}\n  got:      {header}")

        n = 0
        prediction_values: list[float] = []
        for i, row in enumerate(reader, start=2):
            n += 1
            if not (row.get(id_column) or "").strip():
                result.fail(f"line {i}: empty {id_column}")
            for lab in labels:
                raw = row.get(lab, "")
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    result.fail(f"line {i}: {lab} not a float ({raw!r})")
                    continue
                if not 0.0 <= val <= 1.0:
                    result.fail(f"line {i}: {lab}={val} outside [0,1]")
                prediction_values.append(val)

        result.n_rows = n
        if require_rows and n == 0:
            result.fail("no data rows")
        if require_prediction_variation and n > 1 and len(set(prediction_values)) < 2:
            result.fail("prediction output is constant across all labels and rows")

    return result
