"""Explicit, safe downstream outputs for durable stage replay."""

from __future__ import annotations

from typing import Any, Mapping

_STAGE_OUTPUT_FIELDS: dict[str, tuple[str, ...]] = {
    "RESEARCH": (
        "research_verified", "research_verification_detail", "research_passes",
        "deep_ok", "deep_learnings", "deep_sources", "kaggle_ok", "browser_ok",
    ),
    "PLAN": ("plan_text", "plan_verified"),
    "CODE": (
        "code_ok", "code_verified", "code_agent", "wrote_custom_infer",
        "wrote_methods", "wrote_recipe",
    ),
    "LOCAL_SMOKE": ("smoke_ok", "smoke_path", "code_ok", "code_verified"),
    "KERNEL_TRAIN": (
        "kernel_ok", "kernel_pending", "kernel_duplicate", "kernel_ref",
        "kernel_version", "kernel_path", "kernel_resumed", "kernel_judge_ok",
    ),
    "VALIDATE_SUB": ("validate_ok", "candidate_csv", "output_duplicate"),
    "TELEGRAM_APPROVE": ("approve_ok", "waiting_approve"),
    "SUBMIT": ("submit_ok", "submit_message", "submission_pending", "candidate_csv"),
    "FEEDBACK": ("feedback_score", "feedback_pending"),
    "HEAL": ("heal_decision",),
}


def capture(stage: str, result: Any) -> dict[str, Any]:
    """Capture only the allowlisted fields for a completed stage."""
    return {
        field: getattr(result, field)
        for field in _STAGE_OUTPUT_FIELDS.get(stage, ())
        if hasattr(result, field)
    }


def restore(stage: str, result: Any, outputs: Mapping[str, Any]) -> None:
    """Restore only known fields; ignore unknown/legacy ledger values."""
    allowed = set(_STAGE_OUTPUT_FIELDS.get(stage, ()))
    for field, value in outputs.items():
        if field in allowed and hasattr(result, field):
            setattr(result, field, value)
