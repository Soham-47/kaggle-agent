"""Verified, hashable description of a Kaggle competition."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ContractError(ValueError):
    pass


@dataclass
class ContractValidationResult:
    """Submission checks derived from one verified competition contract."""

    ok: bool
    path: Path
    n_rows: int = 0
    errors: list[str] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    def fail(self, message: str) -> None:
        self.ok = False
        assert self.errors is not None
        self.errors.append(message)


_REQUIRED_PATHS = (
    ("id",),
    ("slug",),
    ("title",),
    ("task", "family"),
    ("metric", "name"),
    ("metric", "direction"),
    ("data", "identifier_columns"),
    ("data", "target_columns"),
    ("data", "hidden_id_strategy"),
    ("submission", "mode"),
    ("submission", "output_file"),
    ("submission", "columns"),
)


def _get(raw: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = raw
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


@dataclass
class CompetitionContract:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "CompetitionContract":
        value = copy.deepcopy(raw)
        for path in _REQUIRED_PATHS:
            found = _get(value, path)
            if found is None or found == "" or found == []:
                raise ContractError(f"missing or ambiguous {'.'.join(path)}")
        direction = _get(value, ("metric", "direction"))
        if direction not in {"min", "max"}:
            raise ContractError("metric.direction must be min or max")
        if not isinstance(_get(value, ("task", "family")), str) or not _get(value, ("task", "family")).strip():
            raise ContractError("task.family must be a non-empty string")
        if not isinstance(_get(value, ("metric", "name")), str) or not _get(value, ("metric", "name")).strip():
            raise ContractError("metric.name must be a non-empty string")
        raw_columns = _get(value, ("submission", "columns"))
        raw_identifiers = _get(value, ("data", "identifier_columns"))
        raw_targets = _get(value, ("data", "target_columns"))
        if not all(isinstance(items, list) for items in (raw_columns, raw_identifiers, raw_targets)):
            raise ContractError("submission columns and data columns must be lists")
        columns = list(raw_columns)
        identifiers = list(raw_identifiers)
        targets = list(raw_targets)
        if any(not isinstance(item, str) or not item.strip() for item in columns + identifiers + targets):
            raise ContractError("contract column names must be non-empty strings")
        if set(identifiers) & set(targets):
            raise ContractError("identifier and target columns must be disjoint")
        if len(set(columns)) != len(columns):
            raise ContractError("submission.columns must be unique and ordered")
        if columns != identifiers + targets:
            raise ContractError("submission.columns must exactly preserve identifier/target order")
        mode = _get(value, ("submission", "mode"))
        if mode not in {"file", "notebook"}:
            raise ContractError("submission.mode must be file or notebook")
        if not isinstance(_get(value, ("submission", "output_file")), str) or not _get(value, ("submission", "output_file")).strip():
            raise ContractError("submission.output_file must be a non-empty string")
        column_types = _get(value, ("submission", "column_types"))
        if column_types is not None:
            if not isinstance(column_types, dict) or set(column_types) != set(columns):
                raise ContractError("submission.column_types must declare every submission column")
            allowed_types = {"string", "float", "number", "integer", "boolean"}
            if any(str(kind).lower() not in allowed_types for kind in column_types.values()):
                raise ContractError("submission.column_types contains an unsupported type")
        validation = value.get("validation") or {}
        if not isinstance(validation, dict):
            raise ContractError("validation must be a mapping")
        minimum_rows = validation.get("minimum_rows")
        if minimum_rows is not None and (isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows < 0):
            raise ContractError("validation.minimum_rows must be a non-negative integer")
        require_variation = validation.get("require_variation")
        if require_variation is not None and not isinstance(require_variation, bool):
            raise ContractError("validation.require_variation must be a boolean")
        constraints = _get(value, ("submission", "value_constraints"))
        if constraints is not None and not isinstance(constraints, dict):
            raise ContractError("submission.value_constraints must be a mapping")
        return cls(value)

    @property
    def task_family(self) -> str:
        return str(self.raw["task"]["family"])

    @property
    def compatibility_hash(self) -> str:
        compatible = copy.deepcopy(self.raw)
        compatible.pop("contract_hash", None)
        runtime = compatible.get("runtime")
        if isinstance(runtime, dict):
            runtime.pop("resume_datasets", None)
        encoded = json.dumps(compatible, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

    def validate_submission(
        self, path: Path, *, expected_ids: Iterable[tuple[str, ...]] | None = None
    ) -> ContractValidationResult:
        return validate_submission_contract(path, self, expected_ids=expected_ids)


def validate_submission_contract(
    path: Path,
    contract: CompetitionContract | dict[str, Any],
    *,
    expected_ids: Iterable[tuple[str, ...]] | None = None,
) -> ContractValidationResult:
    """Validate a CSV against contract fields, with tabular task semantics.

    Classification predictions are probabilities in ``[0, 1]``; regression
    predictions remain numeric but are not incorrectly constrained to that
    interval.  Unknown task families use only the explicit contract type and
    column/value constraints, so onboarding remains conservative.
    """
    raw = contract.raw if isinstance(contract, CompetitionContract) else contract
    result = ContractValidationResult(True, path)
    if not path.is_file():
        result.fail(f"missing file: {path}")
        return result
    data = raw.get("data") or {}
    submission = raw.get("submission") or {}
    validation = raw.get("validation") or {}
    identifiers = [str(x) for x in data.get("identifier_columns") or []]
    targets = [str(x) for x in data.get("target_columns") or []]
    expected = [str(x) for x in submission.get("columns") or [*identifiers, *targets]]
    family = str((raw.get("task") or {}).get("family") or "")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != expected:
            result.fail(f"header mismatch\n  expected: {expected}\n  got:      {header}")
        rows = list(reader)
    result.n_rows = len(rows)
    expected_identifier_rows = [
        tuple(str(value).strip() for value in item) for item in (expected_ids or ())
    ]
    if expected_identifier_rows:
        actual_identifier_rows = [
            tuple(str(row.get(identifier) or "").strip() for identifier in identifiers)
            for row in rows
        ]
        if len(rows) != len(expected_identifier_rows):
            result.fail(
                f"submission row count {len(rows)} does not match sample row count "
                f"{len(expected_identifier_rows)}"
            )
        if actual_identifier_rows != expected_identifier_rows:
            result.fail(
                "submission identifiers do not match sample order or values"
            )
    minimum = validation.get("minimum_rows")
    if minimum is not None and result.n_rows < int(minimum):
        result.fail(f"only {result.n_rows} data rows; require at least {minimum}")
    if not rows:
        result.fail("no data rows")
    target_values: list[float] = []
    seen_identifiers: set[tuple[str, ...]] = set()
    constraints = submission.get("value_constraints") or {}
    declared_types = submission.get("column_types") or {}
    for index, row in enumerate(rows, start=2):
        identifier_values = tuple(str(row.get(identifier) or "").strip() for identifier in identifiers)
        if identifiers and all(identifier_values):
            if identifier_values in seen_identifiers:
                result.fail(f"line {index}: duplicate submission identifier")
            seen_identifiers.add(identifier_values)
        for identifier in identifiers:
            raw_identifier = row.get(identifier, "")
            declared = str(declared_types.get(identifier) or "").lower()
            if not str(raw_identifier or "").strip():
                result.fail(f"line {index}: empty {identifier}")
            elif declared == "integer":
                try:
                    if str(int(str(raw_identifier).strip())) != str(raw_identifier).strip():
                        raise ValueError
                except (TypeError, ValueError):
                    result.fail(f"line {index}: {identifier} must be an integer")
            elif declared == "float" or declared == "number":
                try:
                    if not math.isfinite(float(raw_identifier)):
                        raise ValueError
                except (TypeError, ValueError):
                    result.fail(f"line {index}: {identifier} must be numeric")
        for target in targets:
            raw_value = row.get(target, "")
            declared = str(declared_types.get(target) or "").lower()
            if declared == "string":
                if not str(raw_value).strip():
                    result.fail(f"line {index}: {target} must be a non-empty string")
                continue
            if declared == "boolean":
                if str(raw_value).strip().lower() not in {"true", "false", "0", "1"}:
                    result.fail(f"line {index}: {target} must be boolean")
                continue
            if declared == "integer":
                try:
                    if str(int(str(raw_value).strip())) != str(raw_value).strip():
                        raise ValueError
                except (TypeError, ValueError):
                    result.fail(f"line {index}: {target} must be an integer")
                    continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                result.fail(f"line {index}: {target} not numeric ({raw_value!r})")
                continue
            if not math.isfinite(value):
                result.fail(f"line {index}: {target} must be finite")
                continue
            target_values.append(value)
            bound = constraints.get(target) or {}
            if isinstance(bound, dict):
                if bound.get("min") is not None and value < float(bound["min"]):
                    result.fail(f"line {index}: {target} below minimum")
                if bound.get("max") is not None and value > float(bound["max"]):
                    result.fail(f"line {index}: {target} above maximum")
            if family in {"tabular_classification", "image_classification", "image_multilabel_classification", "text_classification"} and not 0.0 <= value <= 1.0:
                result.fail(f"line {index}: {target}={value} outside [0,1]")
    if validation.get("require_variation") and len(rows) > 1 and len(set(target_values)) < 2:
        result.fail("prediction output is constant across all labels and rows")
    return result
