"""Study-level metadata ranker (no pandas). Varying ranks beat constant 0.5 AUC."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from .schema import ID_COLUMN, LABELS
except ImportError:
    from schema import ID_COLUMN, LABELS  # type: ignore

FEATURE_NAMES = (
    "n_series",
    "n_axial",
    "n_sagittal",
    "n_coronal",
    "n_other_plane",
    "frac_axial",
    "frac_sagittal",
    "frac_coronal",
    "n_fluid",
    "frac_fluid",
    "fluid_equals_fat",
    "bias",
)


def series_feature_row(rows: list[dict[str, str]]) -> list[float]:
    n = len(rows)
    if n == 0:
        return [0.0] * (len(FEATURE_NAMES) - 1) + [1.0]
    ax = sag = cor = other = fluid = fat = same = 0
    for r in rows:
        plane = (r.get("Anatomical_Plane") or "").strip().title()
        if plane == "Axial":
            ax += 1
        elif plane == "Sagittal":
            sag += 1
        elif plane == "Coronal":
            cor += 1
        else:
            other += 1
        fl = int(float(r.get("Fluid_Sensitive") or 0) or 0)
        fa = int(float(r.get("Fat_Suppression") or 0) or 0)
        fluid += fl
        fat += fa
        if fl == fa:
            same += 1
    return [
        float(n),
        float(ax),
        float(sag),
        float(cor),
        float(other),
        ax / n,
        sag / n,
        cor / n,
        float(fluid),
        fluid / n,
        1.0 if same == n else 0.0,
        1.0,
    ]


def load_series_by_study(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get(ID_COLUMN) or "").strip()
            if sid:
                out.setdefault(sid, []).append(row)
    return out


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_label(xs: list[list[float]], ys: list[int], steps: int = 80, lr: float = 0.08) -> list[float]:
    dim = len(xs[0])
    w = [0.0] * dim
    n = max(1, len(xs))
    for _ in range(steps):
        grad = [0.0] * dim
        for x, y in zip(xs, ys):
            p = _sigmoid(sum(a * b for a, b in zip(w, x)))
            err = p - y
            for j in range(dim):
                grad[j] += err * x[j]
        for j in range(dim):
            w[j] -= lr * (grad[j] / n + 1e-3 * w[j])
    return w


def predict_one(w: list[float], x: list[float]) -> float:
    p = _sigmoid(sum(a * b for a, b in zip(w, x)))
    return min(1.0 - 1e-6, max(1e-6, p))


def gold_or_report_labels(train_csv: Path) -> dict[str, dict[str, int]]:
    from .reports import extract_labels

    out: dict[str, dict[str, int]] = {}
    with train_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get(ID_COLUMN) or "").strip()
            if not sid:
                continue
            if all((row.get(lab) or "").strip() != "" for lab in LABELS):
                out[sid] = {lab: int(float(row[lab])) for lab in LABELS}
            else:
                out[sid] = extract_labels(row.get("Report") or "")
    return out


def fit_ranker(train_csv: Path, train_series_csv: Path) -> dict:
    series = load_series_by_study(train_series_csv)
    ymap = gold_or_report_labels(train_csv)
    xs: list[list[float]] = []
    ys: dict[str, list[int]] = {lab: [] for lab in LABELS}
    for sid, labs in ymap.items():
        xs.append(series_feature_row(series.get(sid, [])))
        for lab in LABELS:
            ys[lab].append(int(labs[lab]))
    weights = {lab: fit_label(xs, ys[lab]) for lab in LABELS}
    priors = {lab: (sum(ys[lab]) / max(1, len(ys[lab]))) for lab in LABELS}
    return {
        "feature_names": list(FEATURE_NAMES),
        "weights": weights,
        "priors": priors,
        "n_train": len(xs),
    }


def save_ranker(model: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model), encoding="utf-8")
    return path


def load_ranker(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_studies(
    study_ids: list[str],
    series_by_study: dict[str, list[dict[str, str]]],
    model: dict,
) -> list[dict[str, float | str]]:
    weights = model["weights"]
    rows: list[dict[str, float | str]] = []
    for sid in study_ids:
        x = series_feature_row(series_by_study.get(sid, []))
        rec: dict[str, float | str] = {ID_COLUMN: sid}
        for lab in LABELS:
            rec[lab] = predict_one(weights[lab], x)
        rows.append(rec)
    return rows
