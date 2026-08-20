"""Synthetic deterministic baseline."""

from .schema import ID_COLUMN, LABELS


def predict_constant(ids: list[str], value: float = 0.5) -> list[dict[str, object]]:
    return [{ID_COLUMN: item, **{label: value for label in LABELS}} for item in ids]
