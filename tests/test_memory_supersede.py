"""Expiry/supersede tracking for stale memory notes."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from kaggle_agent.memory.ingest import pick_cards, pick_experiments, retrieve
from kaggle_agent.memory.write import supersede_experiment


# --- pick_cards: skip superseded + expired, newest fallback ---


def _make_card(deep: Path, name: str, suffix: str = "") -> Path:
    deep.mkdir(parents=True, exist_ok=True)
    p = deep / f"source-{name}.md"
    p.write_text(f"# {name}\n- title: {name}{suffix}\n", encoding="utf-8")
    return p


def test_pick_cards_skips_expired(tmp_path: Path):
    deep = tmp_path / "research-deep"
    good = _make_card(deep, "a-good")
    old = _make_card(deep, "b-old")
    old.write_text(
        "# b-old\n- title: b\n- expires: 2020-01-01\n", encoding="utf-8"
    )
    cards = pick_cards(deep, n=2)
    assert good in cards
    assert old not in cards


def test_pick_cards_skips_superseded(tmp_path: Path):
    deep = tmp_path / "research-deep"
    good = _make_card(deep, "c-good")
    dead = _make_card(deep, "d-dead")
    dead.write_text(
        "# d-dead\n- title: d\n- superseded: yes\n", encoding="utf-8"
    )
    cards = pick_cards(deep, n=2)
    assert good in cards
    assert dead not in cards


def test_pick_cards_newest_fallback_when_all_filtered(tmp_path: Path):
    deep = tmp_path / "research-deep"
    a = _make_card(deep, "a")
    a.write_text("# a\n- title: a\n- superseded: yes\n", encoding="utf-8")
    b = _make_card(deep, "b")
    b.write_text(
        "# b\n- title: b\n- expires: 2020-01-01\n", encoding="utf-8"
    )
    cards = pick_cards(deep, n=2)
    # both are superseded/expired: fall back to the newest by mtime
    assert len(cards) == 1


# --- pick_experiments: skip superseded + expired ---


def _make_exp(exp_dir: Path, exp_id: str, score: str = "none", extra: str = "") -> Path:
    exp_dir.mkdir(parents=True, exist_ok=True)
    p = exp_dir / f"{exp_id}.md"
    p.write_text(
        f"# {exp_id}\n\n- hypothesis: h\n- public_score: {score}\n{extra}\n",
        encoding="utf-8",
    )
    return p


def test_pick_experiments_skips_superseded(tmp_path: Path):
    exp_dir = tmp_path / "experiments"
    good = _make_exp(exp_dir, "e1", score="0.526")
    dead = _make_exp(exp_dir, "e2", score="0.520")
    dead.write_text(
        dead.read_text(encoding="utf-8").rstrip()
        + "\n- superseded: yes\n",
        encoding="utf-8",
    )
    exps = pick_experiments(exp_dir, n=5)
    assert good in exps
    assert dead not in exps


def test_pick_experiments_skips_expired(tmp_path: Path):
    exp_dir = tmp_path / "experiments"
    good = _make_exp(exp_dir, "e1", score="0.526")
    old = _make_exp(exp_dir, "e3", score="0.510")
    old.write_text(
        old.read_text(encoding="utf-8").rstrip()
        + "\n- expires: 2020-01-01\n",
        encoding="utf-8",
    )
    exps = pick_experiments(exp_dir, n=5)
    assert good in exps
    assert old not in exps


# --- retrieve: skip superseded + expired ---


def test_retrieve_skips_superseded(tmp_path: Path):
    deep = tmp_path / "memory" / "research-deep"
    good = _make_card(deep, "good-leak", "leak warning train overlap")
    dead = _make_card(deep, "dead-leak", "leak warning confirmed")
    dead.write_text(
        "# dead-leak\n- superseded: yes\n- leak warning confirmed\n",
        encoding="utf-8",
    )
    result = retrieve(tmp_path, "leak warning", scope="cards")
    assert "good-leak" in result
    assert "dead-leak" not in result


def test_retrieve_skips_expired(tmp_path: Path):
    deep = tmp_path / "memory" / "research-deep"
    _make_card(deep, "fresh-aug", "augment strategy v2")
    old = _make_card(deep, "old-aug", "augment strategy v1")
    old.write_text(
        "# old-aug\n- expires: 2020-01-01\n- augment strategy v1\n",
        encoding="utf-8",
    )
    result = retrieve(tmp_path, "augment strategy", scope="cards")
    assert "fresh-aug" in result
    assert "old-aug" not in result


# --- supersedes write flag ---


def test_supersede_experiment_writes_flag(tmp_path: Path):
    exp_dir = tmp_path / "memory" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    path = exp_dir / "test-exp.md"
    path.write_text(
        "# test-exp\n\n- hypothesis: h\n- public_score: 0.520\n",
        encoding="utf-8",
    )
    supersede_experiment("test-exp", root=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "- superseded: yes" in text


# --- best_score supersedes worse experiments ---


def _make_root_with_exps(tmp_path: Path) -> None:
    exp_dir = tmp_path / "memory" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    _make_exp(exp_dir, "e-high", score="0.528")
    _make_exp(exp_dir, "e-low", score="0.520")
    _make_exp(exp_dir, "e-mid", score="0.526")


def test_supersede_worse_experiments(tmp_path: Path):
    _make_root_with_exps(tmp_path)
    from kaggle_agent.memory.write import supersede_worse_experiments
    supersede_worse_experiments(tmp_path, "0.528")
    high = (tmp_path / "memory" / "experiments" / "e-high.md").read_text(encoding="utf-8")
    low = (tmp_path / "memory" / "experiments" / "e-low.md").read_text(encoding="utf-8")
    mid = (tmp_path / "memory" / "experiments" / "e-mid.md").read_text(encoding="utf-8")
    assert "superseded: yes" not in high
    assert "- superseded: yes" in low
    assert "- superseded: yes" in mid
