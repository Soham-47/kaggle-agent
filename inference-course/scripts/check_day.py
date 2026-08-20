"""Validate a course day folder against the contract and run its lesson.

Usage:
    python scripts/check_day.py --day N [--smoke]

Checks the day folder contract (exactly four files, README title and
sections in order), then runs days/day-XXX/lesson.py with --smoke when
the flag is given, and prints the lesson's captured output. Exits 0 only
when every check passes.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parent.parent
DAYS_DIR = COURSE_ROOT / "days"

REQUIRED_FILES = ["README.md", "lesson.py", "exercise.md", "reflection.md"]
REQUIRED_SECTIONS = [
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
LESSON_TIMEOUT_S = 600


def validate_day(day_dir, smoke: bool = True, run_lesson: bool = True):
    """Validate day_dir against the contract and run its lesson.

    Returns (ok, report) where report is a list of report lines:
    "[ok] ..." and "[fail] ..." for each check, "[output] ..." for the
    lesson's captured stdout, and, when the lesson fails, "[stderr] ...".
    run_lesson=False skips executing lesson.py.
    """
    day_dir = Path(day_dir)
    if not day_dir.is_dir():
        return False, [f"[fail] missing day folder: {day_dir}"]

    report = ["[ok] folder exists"]
    ok = True

    files = {p.name for p in day_dir.iterdir() if p.is_file()}
    for name in REQUIRED_FILES:
        if name not in files:
            ok = False
            report.append(f"[fail] missing required file: {name}")
    for name in sorted(files - set(REQUIRED_FILES)):
        ok = False
        report.append(f"[fail] unexpected file: {name}")
    if ok:
        report.append(f"[ok] files: {len(REQUIRED_FILES)} required files present")

    readme = day_dir / "README.md"
    if readme.is_file():
        ok = _check_readme(readme, day_dir.name, ok, report)
    else:
        ok = False
        report.append("[fail] readme sections: cannot check, README.md is missing")

    if run_lesson and (day_dir / "lesson.py").is_file():
        cmd = [sys.executable, "lesson.py"]
        if smoke:
            cmd.append("--smoke")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(day_dir),
                capture_output=True,
                text=True,
                timeout=LESSON_TIMEOUT_S,
                env={**os.environ, "PYTHONPATH": str(COURSE_ROOT)},
            )
        except subprocess.TimeoutExpired:
            ok = False
            report.append(f"[fail] lesson: timed out after {LESSON_TIMEOUT_S}s")
        else:
            if proc.returncode == 0:
                report.append("[ok] lesson: exited 0")
            else:
                ok = False
                report.append(f"[fail] lesson: exited {proc.returncode}")
            for line in proc.stdout.splitlines():
                report.append(f"[output] {line}")
            if proc.returncode != 0:
                for line in proc.stderr.splitlines():
                    report.append(f"[stderr] {line}")

    return ok, report


def _check_readme(readme: Path, folder_name: str, ok: bool, report: list) -> bool:
    """Append README title and section checks to report; return their combined result."""
    lines = [ln for ln in readme.read_text().splitlines() if ln.strip()]
    title = lines[0] if lines else ""
    title_ok = bool(re.match(r"^# Day \d+\b", title))
    folder_day = re.match(r"day-(\d+)", folder_name)
    if folder_day and not re.match(rf"^# Day {int(folder_day.group(1))}\b", title):
        title_ok = False
    if title_ok:
        report.append(f"[ok] readme title: {title}")
    else:
        ok = False
        report.append(f"[fail] readme title: expected '# Day N ...', got: {title!r}")

    headings = [line[3:] for line in lines if line.startswith("## ")]
    idx = 0
    for name in REQUIRED_SECTIONS:
        try:
            idx = headings.index(name, idx) + 1
        except ValueError:
            ok = False
            report.append(f"[fail] readme sections: {name} missing or out of order")
            break
    else:
        report.append(f"[ok] readme sections: {len(REQUIRED_SECTIONS)} sections in order")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate a course day folder and run its lesson")
    parser.add_argument("--day", type=int, required=True, help="day number between 1 and 100")
    parser.add_argument("--smoke", action="store_true", help="run lesson.py in smoke mode")
    args = parser.parse_args(argv)
    if not 1 <= args.day <= 100:
        parser.error("--day must be between 1 and 100")
    day_dir = DAYS_DIR / f"day-{args.day:03d}"
    ok, report = validate_day(day_dir, smoke=args.smoke)
    print(f"checking {day_dir.relative_to(COURSE_ROOT)}")
    for line in report:
        print(f"  {line}")
    status = "PASS" if ok else "FAIL"
    print(f"day-{args.day:03d}: {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
