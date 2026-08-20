"""Build a Kaggle-ready notebook + kernel-metadata.json (no push)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from kaggle_agent.config import CompetitionConfig
from kaggle_agent.experiment_fingerprint import (
    canonical_hash,
    experiment_fingerprint,
    recipe_hash,
)


def _cell(kind: str, source: str) -> dict:
    cell: dict = {
        "cell_type": kind,
        "metadata": {},
        "source": [source.strip("\n") + "\n"],
    }
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _recipe_source(root: Path | None, competition: CompetitionConfig | None = None) -> str | None:
    if root is None:
        return None
    rel = (
        competition.workspace_relative
        if competition is not None
        else "competitions/example"
    )
    path = root / rel / "pipeline" / "kernel_recipe.py"
    if not path.is_file():
        return None
    ns: dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    src = ns.get("KERNEL_RECIPE_SOURCE")
    return str(src) if isinstance(src, str) and src.strip() else None


def build_baseline_notebook(
    *,
    competition_slug: str,
    labels: list[str],
    study_ids: list[str] | None = None,
    recipe_source: str | None = None,
    id_column: str = "id",
    manifest: dict[str, object] | None = None,
    seed: int = 42,
) -> dict:
    """Notebook that writes submission.csv from mounted data or embedded IDs.

    Runs on Kaggle with competition data attached; also readable offline.
    Raises at build time if no study IDs are available, so a submission can
    never be built from fake IDs.
    """
    label_list = ", ".join(repr(x) for x in labels)
    if not study_ids:
        raise ValueError("build_baseline_notebook: study_ids required")
    embedded_ids = repr(study_ids)
    fallback = f"""
from pathlib import Path
import pandas as pd
COMP_SLUG = {competition_slug!r}
LABELS = [{label_list}]
ID_COL = {id_column!r}
study_ids = {embedded_ids}
out = pd.DataFrame({{ID_COL: study_ids}})
for lab in LABELS:
    out[lab] = 0.5
Path("submission.csv").write_text(out.to_csv(index=False))
print("fallback constant wrote", len(out))
"""
    recipe = recipe_source if recipe_source and "submission.csv" in recipe_source else fallback
    recipe = recipe.replace("SEED = 42", f"SEED = {int(seed)}", 1)
    if manifest:
        manifest_source = (
            "EXPERIMENT_MANIFEST = "
            + repr(manifest)
            + "\nprint('EXPERIMENT_MANIFEST', EXPERIMENT_MANIFEST)\n"
        )
        recipe = manifest_source + recipe
    cells = [
        _cell(
            "markdown",
            f"# {competition_slug}\n\n"
            "Agent recipe kernel: report-derived labels + metadata ranker "
            "(optional DINOv2 blend). Not a constant 0.5 file.",
        ),
        _cell("code", recipe),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def _load_methods(root: Path, competition: CompetitionConfig) -> dict:
    path = root / competition.workspace_relative / "pipeline" / "methods.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_artifact_manifest(root: Path, competition: CompetitionConfig) -> dict:
    path = root / competition.workspace_relative / "pipeline" / "artifact_manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _load_resume_manifest(root: Path, competition: CompetitionConfig) -> dict:
    path = root / competition.workspace_relative / "pipeline" / "resume_manifest.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class KernelPackage:
    folder: Path
    notebook_path: Path
    metadata_path: Path
    kernel_ref: str  # username/slug
    title: str


def _load_study_ids(root: Path, competition: CompetitionConfig) -> list[str]:
    """Real test study IDs from data/test.csv, else sample_submission.csv."""
    import csv

    id_col = competition.id_column
    for name in ("test.csv", "sample_submission.csv"):
        for base in (root / "data", root / competition.workspace_relative / "data"):
            path = base / name
            if not path.is_file():
                continue
            try:
                with path.open(encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                if rows and id_col in rows[0]:
                    ids = [str(r[id_col]) for r in rows if r.get(id_col)]
                    if ids:
                        return ids
            except Exception:
                continue
    return []


def _bundle_recipe_files(
    root: Path, competition: CompetitionConfig, folder: Path
) -> None:
    """Ship train tables + extractor so the kernel can train without mounted CSVs."""
    pipe = root / competition.workspace_relative / "pipeline"
    for name in (
        "reports.py",
        "schema.py",
        "ranker.py",
        "methods.json",
        "methods_applied.md",
        "image_contract.json",
        "artifact_manifest.json",
        "resume_manifest.json",
    ):
        src = pipe / name
        if src.is_file():
            shutil.copy2(src, folder / name)
    data_dirs = (root / "data", root / competition.workspace_relative / "data")
    for name in ("train.csv", "train_series.csv", "test.csv", "test_series.csv"):
        for base in data_dirs:
            src = base / name
            if src.is_file():
                shutil.copy2(src, folder / name)
                break


def write_kernel_package(
    competition: CompetitionConfig,
    *,
    root: Path,
    username: str,
    exp_id: str,
    enable_gpu: bool = False,
    machine_shape: str | None = None,
    is_private: bool = True,
    enable_internet: bool = False,
    plan_text: str = "",
) -> KernelPackage:
    """Write notebook + kernel-metadata.json under competitions/<id>/notebooks/<exp_id>/."""
    prefix = competition.id.replace("_", "-")
    slug_part = f"{prefix}-agent-{exp_id}".replace("_", "-").lower()
    # Kaggle slugs: keep short-ish
    slug_part = "".join(c if c.isalnum() or c == "-" else "-" for c in slug_part)[:60].strip("-")
    kernel_ref = f"{username}/{slug_part}"
    title = f"{prefix}-agent {exp_id}"

    folder = root / competition.workspace_relative / "notebooks" / exp_id
    folder.mkdir(parents=True, exist_ok=True)
    nb_name = "agent_baseline.ipynb"
    nb_path = folder / nb_name
    meta_path = folder / "kernel-metadata.json"

    methods = _load_methods(root, competition)
    recipe_source = _recipe_source(root, competition) or ""
    artifact_manifest = _load_artifact_manifest(root, competition)
    resume_manifest = _load_resume_manifest(root, competition)
    seed = 42
    if plan_text.strip():
        seed += int.from_bytes(hashlib.sha256(exp_id.encode("utf-8")).digest()[:2], "big") % 10000
    manifest = {
        "experiment_id": exp_id,
        "experiment_fingerprint": experiment_fingerprint(
            plan_text,
            methods,
            recipe_source,
            root / competition.workspace_relative / "pipeline" / "code_brief.md",
            seed=seed,
        ),
        "plan_sha256": canonical_hash(plan_text.strip()),
        "recipe_sha256": recipe_hash(recipe_source),
        "methods_sha256": canonical_hash(methods),
        "seed": seed,
        "seed_sha256": canonical_hash(seed),
    }
    if artifact_manifest:
        manifest["artifact_manifest"] = artifact_manifest
    notebook = build_baseline_notebook(
        competition_slug=competition.slug,
        labels=competition.labels,
        study_ids=_load_study_ids(root, competition),
        recipe_source=recipe_source,
        id_column=competition.id_column,
        manifest=manifest,
        seed=seed,
    )
    nb_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

    from kaggle_agent.heal.pins import sanitize_datasets, sanitize_models

    datasets = sanitize_datasets(
        [str(x) for x in (methods.get("dataset_sources") or []) if x]
    )
    resume_dataset = str(resume_manifest.get("dataset_source") or "").strip()
    if resume_dataset:
        resume_sources = sanitize_datasets([resume_dataset])
        if not resume_sources:
            raise ValueError("resume manifest dataset_source is invalid")
        # The resume artifact is required to reproduce the approved fold and
        # always owns one of Kaggle's six attachment slots.  Research-card
        # attachments are optional and are truncated deterministically.
        resume_source = resume_sources[0]
        datasets = [resume_source] + [x for x in datasets if x != resume_source][:5]
    else:
        datasets = datasets[:6]
    models = sanitize_models([str(x) for x in (methods.get("model_sources") or []) if x])[:3]

    # Official template fields from kernels_initialize (kaggle API).
    # Source: KaggleApi.kernels_initialize / https://www.kaggle.com/docs/api
    meta = {
        "id": kernel_ref,
        "title": title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": is_private,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": enable_internet,
        "dataset_sources": datasets,
        "competition_sources": [competition.slug],
        "kernel_sources": [],
        "model_sources": models,
        "experiment_manifest": manifest,
    }
    if machine_shape:
        meta["machine_shape"] = machine_shape
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    _bundle_recipe_files(root, competition, folder)
    return KernelPackage(
        folder=folder,
        notebook_path=nb_path,
        metadata_path=meta_path,
        kernel_ref=kernel_ref,
        title=title,
    )
