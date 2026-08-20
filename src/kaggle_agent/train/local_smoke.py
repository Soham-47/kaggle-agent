"""Competition-aware local smoke runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.config import CompetitionConfig
from kaggle_agent.pipeline.smoke import SmokeResult, read_study_ids, run_local_smoke
from kaggle_agent.pipeline.validate import validate_submission_csv

SYNTHETIC_STUDY_IDS = (
    "smoke-study-0001",
    "smoke-study-0002",
    "smoke-study-0003",
)


@dataclass
class LocalSmokeOutcome:
    ok: bool
    smoke: SmokeResult | None = None
    cv_auc: float | None = None
    errors: list[str] = field(default_factory=list)


def ensure_sample_csv(
    competition: CompetitionConfig,
    root: Path,
    sample_csv: Path | None = None,
) -> Path | None:
    """Return path to sample_submission.csv, downloading via Kaggle if needed."""
    candidates = [
        sample_csv,
        root / "data" / "sample_submission.csv",
        root / competition.workspace_relative / "data" / "sample_submission.csv",
    ]
    for path in candidates:
        if path and path.is_file():
            return path

    # Download small meta file only
    dest = root / "data"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.competition_download_file(
            competition.slug,
            "sample_submission.csv",
            path=str(dest),
            force=True,
            quiet=True,
        )
    except Exception:
        return None

    got = dest / "sample_submission.csv"
    if got.is_file():
        return got
    # sometimes zip
    import zipfile

    for z in dest.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
    return got if got.is_file() else None


def _ranker_smoke(
    competition: CompetitionConfig,
    root: Path,
    *,
    study_ids: list[str],
    out_path: Path,
) -> SmokeResult | None:
    """Use fitted metadata ranker when weights + train_series exist."""
    import csv
    import sys

    workspace = root / competition.workspace_relative
    weights = workspace / "pipeline" / "weights.json"
    series_csv = root / "data" / "test_series.csv"
    if not (workspace / "pipeline" / "ranker.py").is_file() or not weights.is_file():
        return None
    if not series_csv.is_file():
        series_csv = workspace / "data" / "test_series.csv"
    if not series_csv.is_file():
        return None
    sys.path.insert(0, str(workspace))
    from pipeline.ranker import load_ranker, load_series_by_study, predict_studies  # type: ignore

    model = load_ranker(weights)
    by_study = load_series_by_study(series_csv)
    rows = predict_studies(study_ids, by_study, model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [competition.id_column, *competition.labels]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    validation = validate_submission_csv(
        out_path, id_column=competition.id_column, labels=competition.labels
    )
    return SmokeResult(
        ok=validation.ok,
        submission_path=out_path,
        validation=validation,
        n_studies=len(study_ids),
        errors=list(validation.errors),
    )


def run_competition_smoke(
    competition: CompetitionConfig,
    *,
    root: Path,
    exp_id: str,
    sample_csv: Path | None = None,
    value: float = 0.5,
) -> LocalSmokeOutcome:
    out_path = root / competition.workspace_relative / "submissions" / f"{exp_id}_smoke.csv"
    notes: list[str] = []
    study_ids: list[str] = []

    sample = ensure_sample_csv(competition, root, sample_csv)
    if sample and sample.is_file():
        try:
            study_ids = read_study_ids(sample, id_column=competition.id_column)
            notes.append(f"info: study_ids from {sample.name} (n={len(study_ids)})")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"info: sample_csv read failed: {exc}")

    if not study_ids:
        study_ids = list(SYNTHETIC_STUDY_IDS)
        notes.append(
            "info: used synthetic study ids (download sample_submission.csv for real IDs)"
        )

    smoke = _ranker_smoke(
        competition,
        root,
        study_ids=study_ids,
        out_path=out_path,
    )
    if smoke is None:
        smoke = run_local_smoke(
            study_ids=study_ids,
            out_path=out_path,
            id_column=competition.id_column,
            labels=competition.labels,
            value=value,
        )
    cv_auc: float | None = None
    train_csv = _first_file(
        root / "data" / "train.csv",
        root / competition.workspace_relative / "data" / "train.csv",
    )
    train_series_csv = _first_file(
        root / "data" / "train_series.csv",
        root / competition.workspace_relative / "data" / "train_series.csv",
    )
    if train_csv is not None and train_series_csv is not None:
        from kaggle_agent.pipeline.cv import evaluate_ranker_cv

        try:
            result = evaluate_ranker_cv(
                train_csv,
                train_series_csv,
                root / competition.workspace_relative,
            )
        except Exception as exc:  # noqa: BLE001
            result = None
            notes.append(f"info: cv skipped ({exc})")
        if result is not None:
            cv_auc = float(result["macro_auc"])
            notes.append(
                f"info: cv_auc={cv_auc:.4f} n={result['n_studies']} folds={result['folds']}"
            )

    # notes are informational; only validation failures are real errors
    return LocalSmokeOutcome(
        ok=smoke.ok,
        smoke=smoke,
        cv_auc=cv_auc,
        errors=notes + list(smoke.errors),
    )


def _first_file(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None
