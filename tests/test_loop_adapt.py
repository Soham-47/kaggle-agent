from pathlib import Path

from kaggle_agent.config import load_settings
from kaggle_agent.loop import LoopState, load_loop, next_loop_count, save_loop
from kaggle_agent.paths import repo_root


def test_rate_zero_is_n_max():
    assert next_loop_count(0, n_min=2, n_max=8, typical_gain=0.01) == 8


def test_large_rate_near_n_min():
    n = next_loop_count(1.0, n_min=2, n_max=8, typical_gain=0.01)
    assert n == 2


def test_typical_gain_midpoint():
    assert next_loop_count(0.01, n_min=2, n_max=8, typical_gain=0.01) == 5


def test_first_run_no_scores_default():
    assert next_loop_count(None, n_min=2, n_max=8, typical_gain=0.01) == 3


def test_first_run_default_clamped_to_range():
    assert next_loop_count(None, n_min=2, n_max=8, typical_gain=0.01, default_n=10) == 8
    assert next_loop_count(None, n_min=2, n_max=8, typical_gain=0.01, default_n=1) == 2


def test_negative_rate_treated_as_zero():
    assert next_loop_count(-0.05, n_min=2, n_max=8, typical_gain=0.01) == 8


def test_nonpositive_typical_gain_is_n_max():
    assert next_loop_count(0.01, n_min=2, n_max=8, typical_gain=0) == 8
    assert next_loop_count(0.01, n_min=2, n_max=8, typical_gain=-0.01) == 8


def test_n_min_greater_than_n_max_does_not_raise():
    assert next_loop_count(0.01, n_min=8, n_max=2, typical_gain=0.01) == 2


def test_loop_round_trip(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    state = LoopState(
        last_score="0.52",
        prev_score="0.50",
        last_n="3",
        next_n="5",
        note="gain 0.02",
    )
    save_loop(state, tmp_path)
    loaded = load_loop(tmp_path)
    assert loaded.last_score == "0.52"
    assert loaded.prev_score == "0.50"
    assert loaded.last_n == "3"
    assert loaded.next_n == "5"
    assert loaded.note == "gain 0.02"


def test_load_loop_missing_file_defaults(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    loaded = load_loop(tmp_path)
    assert loaded.last_score == "none"
    assert loaded.prev_score == "none"
    assert loaded.next_n == "3"


def test_settings_loop_defaults():
    s = load_settings(repo_root())
    assert s.loop_n_min == 2
    assert s.loop_n_max == 8
    assert s.loop_typical_gain == 0.01
    assert s.loop_default_n == 3
    assert s.loop_max_minutes == 90
