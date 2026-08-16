from pathlib import Path

from kaggle_agent.experiment_fingerprint import (
    canonical_hash,
    experiment_fingerprint,
    recipe_hash,
    recipe_logic_hash,
)


def test_canonical_hash_ignores_mapping_order():
    left = {"b": [2, 1], "a": "x"}
    right = {"a": "x", "b": [2, 1]}

    assert canonical_hash(left) == canonical_hash(right)


def test_recipe_hash_ignores_trailing_whitespace():
    assert recipe_hash("x = 1\n") == recipe_hash("x = 1")
    assert recipe_hash(" x = 1") != recipe_hash("x = 1")


def test_recipe_logic_hash_ignores_comments_and_variant_marker():
    first = "x = 1\nprint(x)\n"
    second = "# comment\nEXPERIMENT_VARIANT = 'abc'\nx = 1\nprint(x)\n"

    assert recipe_logic_hash(first) == recipe_logic_hash(second)


def test_experiment_fingerprint_changes_when_recipe_changes(tmp_path: Path):
    methods = {"implement_steps": ["new encoder"]}

    first = experiment_fingerprint(
        "steps: new encoder",
        methods,
        "sub = 1\n",
        tmp_path / "brief.md",
        seed=42,
    )
    second = experiment_fingerprint(
        "steps: new encoder",
        methods,
        "sub = 2\n",
        tmp_path / "brief.md",
        seed=42,
    )

    assert first != second
