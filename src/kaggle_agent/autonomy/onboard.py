"""Slug-only, fail-closed competition onboarding."""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kaggle_agent.autonomy.contracts import CompetitionContract, ContractError
from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome, failure_signature
from kaggle_agent.kaggle_api.client import KaggleClient
from kaggle_agent.kaggle_api.models import CompetitionInfo
from kaggle_agent.state_md import AgentState, save_state


@dataclass(frozen=True)
class OnboardResult:
    outcome: StageOutcome
    contract: CompetitionContract | None = None


def _competition_id(slug: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")


def _task_family(info: CompetitionInfo) -> str | None:
    tags = " ".join(info.tags)
    if "tabular" in tags and ("classification" in tags or "auc" in info.evaluation_metric.lower()):
        return "tabular_classification"
    if "tabular" in tags and ("regression" in tags or "mean squared" in info.evaluation_metric.lower()):
        return "tabular_regression"
    if "image classification" in tags:
        return "image_classification"
    if "segmentation" in tags:
        return "image_segmentation"
    if "time series" in tags or "forecasting" in tags:
        return "time_series_forecasting"
    if "text" in tags and "classification" in tags:
        return "text_classification"
    return None


def _metric_direction(metric: str) -> str | None:
    value = metric.lower()
    if any(x in value for x in ("auc", "accuracy", "f1", "dice", "map", "average precision")):
        return "max"
    if any(x in value for x in ("error", "loss", "rmse", "mae", "log loss")):
        return "min"
    return None


class CompetitionBootstrapper:
    def __init__(self, root: Path, kaggle: KaggleClient) -> None:
        self.root = root
        self.kaggle = kaggle

    def onboard(self, slug: str) -> OnboardResult:
        try:
            info = self.kaggle.competition_info(slug)
            family = _task_family(info)
            direction = _metric_direction(info.evaluation_metric)
            if family is None or direction is None:
                return OnboardResult(
                    StageOutcome(
                        OutcomeState.NEEDS_AUTHORITY,
                        "BOOTSTRAP",
                        "task family or metric direction is ambiguous",
                        evidence=(info.evaluation_metric, *info.tags),
                    )
                )
            files = self.kaggle.list_meta_files(slug)
            names = {f.name for f in files}
            sample_name = next((n for n in names if n.lower() == "sample_submission.csv"), None)
            train_name = next((n for n in names if n.lower() == "train.csv"), None)
            test_name = next((n for n in names if n.lower() == "test.csv"), None)
            if not sample_name:
                return OnboardResult(
                    StageOutcome(OutcomeState.NEEDS_AUTHORITY, "BOOTSTRAP", "sample submission is unavailable")
                )
            probe = self.root / ".agent" / "onboard" / _competition_id(slug)
            sample_path = self.kaggle.download_file(slug, sample_name, probe, force=True)
            with sample_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                sample_rows = list(reader)
            identifiers = [columns[0]] if columns else []
            rows = len(sample_rows)
            identifier_values = [str(row.get(columns[0]) or "").strip() for row in sample_rows] if columns else []
            malformed_rows = any(None in row or any(key not in row for key in columns) for row in sample_rows)
            if len(columns) < 2 or rows < 1 or malformed_rows or any(not str(column).strip() for column in columns) or len(set(columns)) != len(columns):
                return OnboardResult(
                    StageOutcome(OutcomeState.NEEDS_AUTHORITY, "BOOTSTRAP", "sample submission schema is ambiguous")
                )
            if not all(identifier_values) or len(set(identifier_values)) != len(identifier_values):
                return OnboardResult(
                    StageOutcome(OutcomeState.NEEDS_AUTHORITY, "BOOTSTRAP", "sample submission contains blank or duplicate identifiers")
                )
            targets = columns[1:]
            try:
                for row in sample_rows:
                    for target in targets:
                        value = float(row.get(target, ""))
                        if value != value or value in {float("inf"), float("-inf")}:
                            raise ValueError
            except (TypeError, ValueError):
                return OnboardResult(
                    StageOutcome(OutcomeState.NEEDS_AUTHORITY, "BOOTSTRAP", "sample submission target values are not finite numeric values")
                )
            contract = CompetitionContract.from_mapping(
                {
                    "id": _competition_id(slug),
                    "slug": slug,
                    "title": info.title,
                    "deadline": info.deadline,
                    "url": info.url,
                    "task": {"family": family, "modalities": [family.split("_")[0]]},
                    "metric": {
                        "name": info.evaluation_metric,
                        "direction": direction,
                        "implementation_source": info.url,
                    },
                    "data": {
                        "train_sources": [train_name] if train_name else [],
                        "test_sources": [test_name] if test_name else [],
                        "identifier_columns": identifiers,
                        "target_columns": targets,
                        "group_columns": [],
                        "time_columns": [],
                        "hidden_id_strategy": "sample_submission",
                    },
                    "submission": {
                        "mode": "notebook" if info.kernels_only else "file",
                        "output_file": "submission.csv",
                        "columns": columns,
                        "column_types": {columns[0]: "string", **{c: "float" for c in targets}},
                        "value_constraints": {},
                    },
                    "validation": {
                        "minimum_rows": rows,
                        "require_variation": True,
                        "leakage_rules": [],
                    },
                    "runtime": {"accelerator": "cpu", "internet": False, "dataset_slots": 0},
                    "autonomy": {
                        "first_submission_approved": False,
                        "max_submissions_per_day": min(info.max_daily_submissions, 2),
                        "max_kernel_retries": 1,
                        "max_gpu_hours": 2,
                    },
                }
            )
            self._activate(contract, sample_path)
            return OnboardResult(
                StageOutcome.success(
                    "BOOTSTRAP",
                    f"verified and activated {slug}",
                    evidence=(info.url, sample_name),
                    artifacts=(f"config/competitions/{contract.raw['id']}.yaml",),
                ),
                contract,
            )
        except (ContractError, OSError, ValueError) as exc:
            return OnboardResult(
                StageOutcome(
                    OutcomeState.FATAL,
                    "BOOTSTRAP",
                    str(exc),
                    failure_signature=failure_signature(str(exc)),
                )
            )

    def _activate(self, contract: CompetitionContract, sample_path: Path) -> None:
        cid = str(contract.raw["id"])
        raw: dict[str, Any] = contract.to_mapping()
        raw["contract_hash"] = contract.compatibility_hash
        raw["labels"] = list(raw["data"]["target_columns"])
        raw["workspace"] = {"relative": f"competitions/{cid}"}
        raw["submit"] = {
            "mode": raw["submission"]["mode"],
            "output_file": raw["submission"]["output_file"],
        }
        raw["submission"].update(
            {
                "id_column": raw["data"]["identifier_columns"][0],
                "probability_columns": list(raw["data"]["target_columns"]),
                "min_rows": int(raw["validation"]["minimum_rows"]),
            }
        )
        config_path = self.root / "config" / "competitions" / f"{cid}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        workspace = self.root / "competitions" / cid / "pipeline"
        workspace.mkdir(parents=True, exist_ok=True)
        data_dir = workspace.parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample_path, data_dir / "sample_submission.csv")
        self._write_baseline_scaffold(workspace, raw)
        settings_path = self.root / "config" / "settings.yaml"
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
        settings["default_competition"] = cid
        settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
        memory = self.root / "memory" / "COMPETITION.md"
        memory.write_text(
            "# COMPETITION\n\n"
            f"- id: {cid}\n- slug: {contract.raw['slug']}\n"
            f"- title: {contract.raw['title']}\n"
            f"- metric: {contract.raw['metric']['name']} ({contract.raw['metric']['direction']})\n"
            f"- task: {contract.task_family}\n"
            f"- contract_hash: {contract.compatibility_hash}\n",
            encoding="utf-8",
        )
        save_state(AgentState(competition=cid, paused=False), self.root)

    @staticmethod
    def _write_baseline_scaffold(pipeline: Path, raw: dict[str, Any]) -> None:
        identifier = raw["data"]["identifier_columns"][0]
        targets = raw["data"]["target_columns"]
        files = {
            "__init__.py": "\"\"\"Competition-local generated pipeline.\"\"\"\n",
            "schema.py": (
                "\"\"\"Generated from the verified competition contract.\"\"\"\n"
                f"ID_COLUMN = {identifier!r}\nLABELS = {targets!r}\n"
                "SUBMISSION_HEADER = [ID_COLUMN, *LABELS]\n"
            ),
            "baseline.py": (
                "\"\"\"Schema baseline; CODE replaces this after research.\"\"\"\n"
                "from .schema import ID_COLUMN, LABELS\n\n"
                "def predict_constant(ids, value=0.5):\n"
                "    return [{ID_COLUMN: item, **{label: value for label in LABELS}} for item in ids]\n"
            ),
            "ranker.py": "\"\"\"Adapter-owned ranking placeholder for CODE.\"\"\"\n",
            "reports.py": "\"\"\"No report extraction in the generated baseline.\"\"\"\n",
            "recipe.py": (
                "\"\"\"Generated marker; live training requires CODE verification.\"\"\"\n"
                "from dataclasses import dataclass\n\n"
                "@dataclass\nclass RecipeResult:\n    ok: bool\n    message: str = ''\n\n"
                "def apply_recipe(*args, **kwargs):\n"
                "    return RecipeResult(False, 'CODE has not generated a verified training recipe')\n"
            ),
            "kernel_recipe.py": (
                "\"\"\"Generated Kaggle wrapper. The payload is the string below.\"\"\"\n"
                "KERNEL_RECIPE_SOURCE = r'''\n"
                "from pathlib import Path\n"
                "import csv\n"
                f"ID_COLUMN = {identifier!r}\n"
                f"LABELS = {targets!r}\n"
                "candidates = [Path('sample_submission.csv'), Path('data/sample_submission.csv'), Path('../data/sample_submission.csv'), Path('test.csv'), Path('data/test.csv'), Path('../data/test.csv')]\n"
                "sample = next((p for p in candidates if p.is_file()), None)\n"
                "if sample is None:\n"
                "    raise FileNotFoundError('contract baseline requires mounted sample_submission.csv or test.csv')\n"
                "with sample.open(newline='', encoding='utf-8-sig') as handle:\n"
                "    rows = list(csv.DictReader(handle))\n"
                "if not rows:\n"
                "    raise ValueError('contract baseline found zero test rows')\n"
                "with Path('submission.csv').open('w', newline='', encoding='utf-8') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=[ID_COLUMN, *LABELS])\n"
                "    writer.writeheader()\n"
                "    for index, row in enumerate(rows):\n"
                "        value = 0.25 + 0.5 * (index % 2)\n"
                "        writer.writerow({ID_COLUMN: row.get(ID_COLUMN, ''), **{label: value for label in LABELS}})\n"
                "'''\n"
            ),
        }
        for name, content in files.items():
            (pipeline / name).write_text(content, encoding="utf-8")
