"""Verified, hashable description of a Kaggle competition."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    pass


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
        columns = list(_get(value, ("submission", "columns")))
        identifiers = list(_get(value, ("data", "identifier_columns")))
        targets = list(_get(value, ("data", "target_columns")))
        if not set(identifiers + targets).issubset(columns):
            raise ContractError("submission.columns must include identifiers and targets")
        return cls(value)

    @property
    def task_family(self) -> str:
        return str(self.raw["task"]["family"])

    @property
    def compatibility_hash(self) -> str:
        compatible = copy.deepcopy(self.raw)
        runtime = compatible.get("runtime")
        if isinstance(runtime, dict):
            runtime.pop("resume_datasets", None)
        encoded = json.dumps(compatible, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(self.raw)

