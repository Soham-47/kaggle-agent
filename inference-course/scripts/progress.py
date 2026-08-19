"""Print course progress from ROADMAP.md checkboxes.

Usage:
    python scripts/progress.py

Parses the "## Phase N — Title (days X-Y)" headers and the day-table rows
below them, then prints per-phase done/total and the overall done/total.
"""

import re
import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parent.parent
ROADMAP = COURSE_ROOT / "ROADMAP.md"

PHASE = re.compile(r"^## Phase (\d+) — (.+) \(days (\d+)-(\d+)\)$")
ROW = re.compile(r"^\| - \[([ x])\] \| \d{3} \|")


def main() -> int:
    if not ROADMAP.is_file():
        print(f"error: {ROADMAP} not found", file=sys.stderr)
        return 1
    phases = []
    current = None
    for line in ROADMAP.read_text().splitlines():
        match = PHASE.match(line)
        if match:
            current = {
                "label": f"Phase {match.group(1)} — {match.group(2)}",
                "done": 0,
                "total": 0,
            }
            phases.append(current)
            continue
        match = ROW.match(line)
        if match and current is not None:
            current["total"] += 1
            if match.group(1) == "x":
                current["done"] += 1
    if not phases:
        print("error: no phase headers found in ROADMAP.md", file=sys.stderr)
        return 1
    done = sum(p["done"] for p in phases)
    total = sum(p["total"] for p in phases)
    for phase in phases:
        print(f"{phase['label']}: {phase['done']}/{phase['total']} done")
    print(f"Overall: {done}/{total} done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
