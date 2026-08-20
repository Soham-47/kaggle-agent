"""Fix junk Kaggle attach pins so kernels_push can succeed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kaggle_agent.heal.submit_errors import is_409_title_conflict, is_network_error
from kaggle_agent.research.source_cards import _valid_attach_ref, valid_model_pin


def should_wait_approve(
    *,
    validate_ok: bool | None,
    submit_ok: bool | None,
    dry_run: bool,
    assume_approved: bool,
    errors: list[str] | None = None,
) -> bool:
    """True only when a live CSV is ready and a human yes is still required."""
    if dry_run or assume_approved:
        return False
    if any(is_pin_error(e) or is_network_error(e) or is_409_title_conflict(e)
           for e in (errors or [])):
        return False
    return bool(validate_ok) and not submit_ok


def is_pin_error(message: str) -> bool:
    low = (message or "").lower()
    return "model instance version" in low or "version-number" in low


def sanitize_models(refs: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in refs or []:
        ref = str(raw).strip().strip("/")
        if valid_model_pin(ref) and ref not in out:
            out.append(ref)
    return out


_NOT_DATASET_SLUG = frozenset({"dinov2", "pytorch", "transformers", "model", "keras"})

def sanitize_datasets(refs: list[str] | None) -> list[str]:
    out: list[str] = []
    for raw in refs or []:
        ref = str(raw).strip().strip("/")
        if valid_model_pin(ref):
            continue
        parts = [p for p in ref.split("/") if p]
        if len(parts) != 2:
            continue
        if parts[1].lower() in _NOT_DATASET_SLUG:
            continue
        if _valid_attach_ref(ref) and ref not in out:
            out.append(ref)
    return out


def sanitize_methods_payload(data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(data)
    cleaned["dataset_sources"] = sanitize_datasets(list(data.get("dataset_sources") or []))
    cleaned["model_sources"] = sanitize_models(list(data.get("model_sources") or []))
    if cleaned.get("implement_steps"):
        from kaggle_agent.research.source_cards import dedupe_steps

        raw_steps = cleaned["implement_steps"]
        if isinstance(raw_steps, str):
            raw_steps = [raw_steps]
        cleaned["implement_steps"] = dedupe_steps(list(raw_steps))
    return cleaned


def apply_pin_heal(workspace: Path, kernel_folder: Path | None) -> dict[str, Any]:
    """Rewrite methods.json and kernel-metadata.json. Empty model list is OK."""
    changed = False
    methods_path = workspace / "pipeline" / "methods.json"
    payload: dict[str, Any] = {}
    if methods_path.is_file():
        try:
            payload = json.loads(methods_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        cleaned = sanitize_methods_payload(payload)
        if cleaned != payload:
            methods_path.parent.mkdir(parents=True, exist_ok=True)
            methods_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")
            changed = True
        payload = cleaned

    datasets = list(payload.get("dataset_sources") or [])
    models = list(payload.get("model_sources") or [])
    if kernel_folder is not None:
        meta_path = kernel_folder / "kernel-metadata.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            new_ds = sanitize_datasets(list(meta.get("dataset_sources") or []) or datasets)
            new_md = sanitize_models(list(meta.get("model_sources") or []))
            if not new_md:
                new_md = list(models)
            if new_ds != list(meta.get("dataset_sources") or []) or new_md != list(
                meta.get("model_sources") or []
            ):
                meta["dataset_sources"] = new_ds
                meta["model_sources"] = new_md
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                changed = True
            datasets, models = new_ds, new_md

    return {"changed": changed, "dataset_sources": datasets, "model_sources": models}
