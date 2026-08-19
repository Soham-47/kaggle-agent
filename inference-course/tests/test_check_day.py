from pathlib import Path

import pytest

from scripts.check_day import validate_day

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "fixture-day"

DEFAULT_SECTIONS = [
    "Goal",
    "Prerequisites",
    "Concept",
    "Experiment",
    "Expected observations",
    "Metric",
    "Sources",
    "Hardware notes",
    "Reflection prompts",
]
DEFAULT_LESSON = (
    "import argparse\n"
    "parser = argparse.ArgumentParser()\n"
    'parser.add_argument("--smoke", action="store_true")\n'
    "args = parser.parse_args()\n"
    'print(f"mode={args.smoke and \'smoke\' or \'full\'} tps=99.9")\n'
)


def readme_body(sections=None, title="# Day 1 — Example"):
    parts = [title]
    for name in sections or DEFAULT_SECTIONS:
        parts.append(f"## {name}")
        parts.append("placeholder text for the section.")
    return "\n".join(parts) + "\n"


def make_day(tmp_path, name="day", readme=None, lesson=None, extra_files=()):
    day = tmp_path / name
    day.mkdir()
    (day / "README.md").write_text(readme or readme_body())
    (day / "lesson.py").write_text(lesson or DEFAULT_LESSON)
    (day / "exercise.md").write_text("# Exercise\n")
    (day / "reflection.md").write_text("# Reflection\n")
    for name in extra_files:
        (day / name).write_text("junk")
    return day


def test_fixture_day_passes_in_smoke_mode(tmp_path):
    ok, report = validate_day(FIXTURE, smoke=True)
    assert ok
    assert any(line == "[ok] files: 4 required files present" for line in report)
    assert any("[ok] readme sections:" in line for line in report)
    assert any(line == "[ok] lesson: exited 0" for line in report)
    assert any(line == "[output] mode=smoke ttft_ms=12.5" for line in report)


def test_fixture_day_contract_only_when_not_running_lesson(tmp_path):
    ok, report = validate_day(FIXTURE, smoke=True, run_lesson=False)
    assert ok
    assert not any(line.startswith("[output]") for line in report)


def test_missing_required_file_fails(tmp_path):
    day = make_day(tmp_path)
    (day / "reflection.md").unlink()
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any(line == "[fail] missing required file: reflection.md" for line in report)


def test_extra_file_fails(tmp_path):
    day = make_day(tmp_path, extra_files=("junk.txt",))
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any(line == "[fail] unexpected file: junk.txt" for line in report)


def test_missing_readme_section_fails(tmp_path):
    day = make_day(tmp_path, readme=readme_body(sections=DEFAULT_SECTIONS[:6]))
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any("Sources" in line for line in report if line.startswith("[fail]"))


def test_readme_sections_out_of_order_fails(tmp_path):
    reordered = DEFAULT_SECTIONS[:6] + ["Hardware notes", "Sources"] + DEFAULT_SECTIONS[8:]
    day = make_day(tmp_path, readme=readme_body(sections=reordered))
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any("out of order" in line for line in report)


def test_readme_title_must_be_day_heading(tmp_path):
    day = make_day(tmp_path, readme=readme_body(title="# Not a day"))
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any(line.startswith("[fail] readme title") for line in report)


def test_failing_lesson_fails(tmp_path):
    day = make_day(tmp_path, lesson="import sys\nsys.exit(1)\n")
    ok, report = validate_day(day, smoke=True)
    assert not ok
    assert any(line == "[fail] lesson: exited 1" for line in report)


def test_missing_folder_fails(tmp_path):
    ok, report = validate_day(tmp_path / "nope", smoke=True)
    assert not ok
    assert any("missing day folder" in line for line in report)