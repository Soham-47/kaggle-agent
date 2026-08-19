import re
from pathlib import Path

ROADMAP = Path(__file__).resolve().parent.parent / "ROADMAP.md"
ROW = re.compile(r"^\| - \[([ x])\] \| (\d{3}) \|")


def _rows():
    matches = []
    for line in ROADMAP.read_text().splitlines():
        match = ROW.match(line)
        if match:
            matches.append(match)
    return matches


def test_roadmap_has_exactly_100_rows():
    assert len(_rows()) == 100


def test_all_rows_unchecked():
    assert all(match.group(1) == " " for match in _rows())


def test_day_numbers_are_001_to_100_in_order():
    days = [int(match.group(2)) for match in _rows()]
    assert days == list(range(1, 101))


def test_day_numbers_are_zero_padded():
    assert all(len(match.group(2)) == 3 for match in _rows())