from pathlib import Path

from kaggle_agent.memory.write import patch_experiment, write_experiment


def test_patch_experiment_writes_cv_auc(tmp_path: Path):
    write_experiment("exp-1", hypothesis="h", root=tmp_path)
    patch_experiment("exp-1", cv_auc="0.5234", root=tmp_path)
    text = (tmp_path / "memory" / "experiments" / "exp-1.md").read_text()
    assert "- cv_auc: 0.5234" in text


def test_patch_experiment_writes_local_smoke(tmp_path: Path):
    write_experiment("exp-2", hypothesis="h", root=tmp_path)
    patch_experiment("exp-2", local_smoke="ok", root=tmp_path)
    text = (tmp_path / "memory" / "experiments" / "exp-2.md").read_text()
    assert "- local_smoke: ok" in text
