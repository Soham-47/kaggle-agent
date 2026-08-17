"""Evals → plan/code/heal prompt routing (smallest seam: ingest.py helper)."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.memory.ingest import build_context_pack


def _write_eval(root: Path, *, passed: bool = False, checks: list[dict] | None = None) -> None:
    (root / "memory" / "daily").mkdir(parents=True, exist_ok=True)
    (root / "memory" / "daily" / "eval_report.json").write_text(
        json.dumps({
            "passed": passed,
            "ran_at": "2026-08-18T10:00:00+00:00",
            "checks": checks or [],
            "tool_counts": {},
        }),
        encoding="utf-8",
    )


def test_plan_pack_has_evals_when_report_exists(tmp_path: Path):
    _write_eval(tmp_path, checks=[
        {"id": "invalid_json_rate", "ok": True, "detail": "0%"},
        {"id": "code_changed_artifact", "ok": False, "detail": "code wrote nothing"},
    ])
    pack = build_context_pack(tmp_path, view="plan")
    assert "Evals (last cycle)" in pack.sections
    assert "code_changed_artifact" in pack.sections["Evals (last cycle)"]
    assert "passed=False" in pack.sections["Evals (last cycle)"]


def test_heal_pack_has_evals_when_report_exists(tmp_path: Path):
    _write_eval(tmp_path, checks=[
        {"id": "plan_shippable", "ok": False, "detail": "no write_plan"},
    ])
    pack = build_context_pack(tmp_path, view="heal")
    assert "Evals (last cycle)" in pack.sections
    assert "plan_shippable" in pack.sections["Evals (last cycle)"]


def test_code_pack_has_evals_when_report_exists(tmp_path: Path):
    _write_eval(tmp_path, checks=[
        {"id": "code_changed_artifact", "ok": False, "detail": "code wrote nothing"},
    ])
    pack = build_context_pack(tmp_path, view="code")
    assert "Evals (last cycle)" in pack.sections
    assert "code_changed_artifact" in pack.sections["Evals (last cycle)"]


def test_no_evals_section_without_report(tmp_path: Path):
    pack = build_context_pack(tmp_path, view="plan")
    assert "Evals (last cycle)" not in pack.sections


def test_all_passed(tmp_path: Path):
    _write_eval(tmp_path, passed=True, checks=[
        {"id": "invalid_json_rate", "ok": True, "detail": "0%"},
    ])
    pack = build_context_pack(tmp_path, view="plan")
    body = pack.sections["Evals (last cycle)"]
    assert "passed=True" in body
    assert "failed" not in body
