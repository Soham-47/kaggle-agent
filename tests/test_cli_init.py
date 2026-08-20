from pathlib import Path

import yaml

from kaggle_agent import cli


def test_init_creates_generic_competition_scaffold(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)
    config = tmp_path / "config" / "competitions"
    config.mkdir(parents=True)
    (tmp_path / "config" / "settings.yaml").write_text(
        "default_competition: null\n", encoding="utf-8"
    )
    (tmp_path / "memory" / "templates").mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "memory" / "templates"
    for name in ("MEMORY.md", "COMPETITION.md", "state.md", "research.md"):
        (tmp_path / "memory" / "templates" / name).write_text(
            (source / name).read_text(encoding="utf-8"), encoding="utf-8"
        )
    (tmp_path / "config" / "competitions" / "_template.yaml").write_text(
        (source.parent.parent / "config" / "competitions" / "_template.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert cli.main(["init", "--competition", "demo", "--slug", "demo-contest"]) == 0

    contract = yaml.safe_load((config / "demo.yaml").read_text(encoding="utf-8"))
    assert contract["id"] == "demo"
    assert contract["slug"] == "demo-contest"
    assert (tmp_path / "competitions/demo/pipeline/schema.py").is_file()
    assert "rsna" not in (tmp_path / "competitions/demo/pipeline/schema.py").read_text().lower()
    assert (tmp_path / "memory/COMPETITION.md").is_file()


def test_init_refuses_to_overwrite_existing_competition(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)
    config = tmp_path / "config" / "competitions"
    config.mkdir(parents=True)
    (tmp_path / "config" / "competitions" / "_template.yaml").write_text(
        "id: my_contest\nslug: the-kaggle-url-slug\n", encoding="utf-8"
    )
    (tmp_path / "config" / "competitions" / "demo.yaml").write_text("existing\n", encoding="utf-8")

    assert cli.main(["init", "--competition", "demo"]) == 2
