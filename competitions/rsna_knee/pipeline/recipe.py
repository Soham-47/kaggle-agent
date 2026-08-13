"""Apply the PLAN recipe: report labels + metadata ranker files and weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

try:
    from .ranker import fit_ranker, save_ranker
    from .schema import LABELS
except ImportError:
    from ranker import fit_ranker, save_ranker  # type: ignore
    from schema import LABELS  # type: ignore


@dataclass
class RecipeResult:
    ok: bool
    n_train: int = 0
    weights_path: str = ""
    message: str = ""


def apply_recipe(workspace: Path, *, data_dir: Path | None = None) -> RecipeResult:
    """Fit the metadata ranker from train.csv + train_series.csv."""
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir / "train.csv")
    candidates.append(workspace / "data" / "train.csv")
    if len(workspace.parents) >= 2:
        candidates.append(workspace.parents[1] / "data" / "train.csv")
    train_csv = next((p for p in candidates if p.is_file()), candidates[0])
    series_csv = train_csv.parent / "train_series.csv"
    if not train_csv.is_file() or not series_csv.is_file():
        return RecipeResult(ok=False, message=f"missing train tables under {train_csv.parent}")
    model = fit_ranker(train_csv, series_csv)
    out = workspace / "pipeline" / "weights.json"
    save_ranker(model, out)
    return RecipeResult(
        ok=True,
        n_train=int(model["n_train"]),
        weights_path=str(out),
        message=f"recipe metadata-ranker n_train={model['n_train']} labels={len(LABELS)}",
    )


def apply_from_cards(workspace: Path) -> RecipeResult:
    """Turn method cards into files the kernel package can attach and follow."""
    methods_path = workspace / "pipeline" / "methods.json"
    if not methods_path.is_file():
        return RecipeResult(ok=False, message="no methods.json from source cards")
    try:
        data = json.loads(methods_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return RecipeResult(ok=False, message=f"methods.json invalid: {exc}")
    if not isinstance(data, dict):
        return RecipeResult(ok=False, message="methods.json not an object")
    datasets = [str(x) for x in (data.get("dataset_sources") or [])]
    models = [str(x) for x in (data.get("model_sources") or [])]
    steps = [str(x) for x in (data.get("implement_steps") or [])]
    hints = [str(x) for x in (data.get("infer_hints") or [])]
    applied = workspace / "pipeline" / "methods_applied.md"
    lines = [
        "# Methods applied by CODE",
        "",
        f"dataset_sources: {', '.join(datasets) or 'none'}",
        f"model_sources: {', '.join(models) or 'none'}",
        f"infer_hints: {', '.join(hints) or 'none'}",
        "",
        "Kernel must:",
        "1. Attach the dataset_sources / model_sources listed above.",
        "2. Discover hidden test IDs from study folders, not only test.csv.",
        "3. Rank-average any mounted prediction CSVs with the local ranker.",
        "",
    ]
    if steps:
        lines.append("Copyable steps from cards:")
        lines.extend(f"- {s}" for s in steps)
        lines.append("")
    applied.write_text("\n".join(lines), encoding="utf-8")
    return RecipeResult(
        ok=True,
        weights_path=str(methods_path),
        message=f"cards datasets={len(datasets)} models={len(models)} hints={hints}",
    )
