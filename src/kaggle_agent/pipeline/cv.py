"""Local grouped-fold CV for a competition workspace ranker."""

from __future__ import annotations

import random
import csv
import sys
from pathlib import Path
from typing import Any


def macro_auc(y_true: list[int], scores: list[float]) -> float:
    """Return tie-aware rank AUC, or 0.5 when a class is absent."""
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    positives = [i for i, value in enumerate(y_true) if value == 1]
    negatives = [i for i, value in enumerate(y_true) if value == 0]
    if not positives or not negatives:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        rank = (start + end) / 2.0 + 1.0
        for position in range(start, end + 1):
            ranks[order[position]] = rank
        start = end + 1

    positive_rank_sum = sum(ranks[index] for index in positives)
    positive_count = len(positives)
    negative_count = len(negatives)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def grouped_folds(
    study_ids: list[str], n_folds: int = 5, seed: int = 0
) -> list[tuple[list[int], list[int]]]:
    """Create deterministic folds that never split one study across folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    study_groups = sorted(set(study_ids))
    if not study_groups:
        return []
    rng = random.Random(seed)
    rng.shuffle(study_groups)
    fold_count = min(n_folds, len(study_groups))
    folds: list[tuple[list[int], list[int]]] = []
    for fold_number in range(fold_count):
        validation_groups = set(study_groups[fold_number::fold_count])
        validation = [
            index for index, study_id in enumerate(study_ids) if study_id in validation_groups
        ]
        training = [
            index for index, study_id in enumerate(study_ids) if study_id not in validation_groups
        ]
        folds.append((training, validation))
    return folds


def evaluate_ranker_cv(
    train_csv: Path,
    train_series_csv: Path,
    workspace: Path,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Evaluate the workspace ranker with deterministic out-of-fold predictions."""
    if not train_csv.is_file() or not train_series_csv.is_file():
        return None
    if not (workspace / "pipeline" / "ranker.py").is_file():
        return None

    workspace_text = str(workspace)
    inserted_path = workspace_text not in sys.path
    if inserted_path:
        sys.path.insert(0, workspace_text)
    try:
        from pipeline.ranker import (  # type: ignore[import-not-found]
            fit_label,
            load_series_by_study,
            predict_one,
            series_feature_row,
            LABELS,
        )

        series_by_study = load_series_by_study(train_series_csv)
        labels_by_study: dict[str, dict[str, int]] = {}
        with train_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                study_id = (row.get("StudyInstanceUID") or "").strip()
                if study_id and all((row.get(label) or "").strip() for label in LABELS):
                    labels_by_study[study_id] = {
                        label: int(float(row[label])) for label in LABELS
                    }
        study_ids = [study_id for study_id in labels_by_study if study_id in series_by_study]
        if len(study_ids) < 2:
            return None

        features = [series_feature_row(series_by_study[study_id]) for study_id in study_ids]
        labels = list(next(iter(labels_by_study.values())))
        out_of_fold: dict[str, list[tuple[int, float]]] = {label: [] for label in labels}
        folds = grouped_folds(study_ids, n_folds=n_folds, seed=seed)
        for training, validation in folds:
            training_features = [features[index] for index in training]
            weights = {
                label: fit_label(
                    training_features,
                    [int(labels_by_study[study_ids[index]][label]) for index in training],
                    steps=40,
                )
                for label in labels
            }
            for index in validation:
                for label in labels:
                    out_of_fold[label].append(
                        (
                            int(labels_by_study[study_ids[index]][label]),
                            predict_one(weights[label], features[index]),
                        )
                    )

        per_label = {
            label: macro_auc(
                [truth for truth, _ in values],
                [score for _, score in values],
            )
            for label, values in out_of_fold.items()
        }
        return {
            "macro_auc": round(sum(per_label.values()) / len(per_label), 4),
            "per_label": {label: round(value, 4) for label, value in per_label.items()},
            "n_studies": len(study_ids),
            "folds": len(folds),
        }
    finally:
        if inserted_path:
            sys.path.remove(workspace_text)
