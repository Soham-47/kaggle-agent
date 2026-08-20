"""Report labels + metadata ranker must produce varying ranks, not constant 0.5."""

from pathlib import Path

from kaggle_agent.paths import repo_root
from helpers import write_kernel_fixture_data


def test_extract_labels_spanish_meniscus():
    import sys

    ws = repo_root() / "competitions" / "rsna_knee"
    sys.path.insert(0, str(ws))
    from pipeline.reports import extract_labels

    labs = extract_labels("Rotura de menisco interno. Derrame articular. Sin rotura ACL.")
    assert labs["Medial Meniscus"] == 1
    assert labs["Effusion"] == 1


def test_fit_ranker_varies_on_real_test(tmp_path: Path):
    import sys

    root = tmp_path / "fixture"
    ws = root / "competitions" / "rsna_knee"
    (ws / "pipeline").mkdir(parents=True)
    write_kernel_fixture_data(root)
    sys.path.insert(0, str(ws))
    from pipeline.recipe import apply_recipe
    from pipeline.ranker import load_ranker, load_series_by_study, predict_studies

    r = apply_recipe(ws, data_dir=root / "data")
    assert r.ok
    assert r.n_train >= 58
    model = load_ranker(Path(r.weights_path))
    ids = ["test-a", "test-b", "test-c"]
    series = load_series_by_study(root / "data" / "test_series.csv")
    rows = predict_studies(ids, series, model)
    assert len(rows) == 3
    # not a constant 0.5 file
    acl = [float(r["ACL"]) for r in rows]
    assert max(acl) - min(acl) > 1e-6 or any(abs(v - 0.5) > 1e-4 for v in acl)


def test_kernel_package_embeds_recipe(tmp_path: Path):
    import json
    import shutil

    from kaggle_agent.config import load_competition
    from kaggle_agent.train.notebook_builder import write_kernel_package

    root = tmp_path / "ka"
    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    write_kernel_fixture_data(root)
    shutil.copytree(real / "competitions" / "rsna_knee" / "pipeline", root / "competitions" / "rsna_knee" / "pipeline")
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="recipe1")
    text = pkg.notebook_path.read_text(encoding="utf-8")
    assert "_find_report_labels" in text
    assert "discover" in text.lower()
    assert "constant probabilities from sample header" not in text
    assert (pkg.folder / "train.csv").is_file()
    assert (pkg.folder / "reports.py").is_file()
    assert (pkg.folder / "ranker.py").is_file()
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is False


def test_apply_from_cards_writes_applied_file(tmp_path: Path):
    import json
    import sys

    ws = tmp_path / "comp"
    (ws / "pipeline").mkdir(parents=True)
    (ws / "pipeline" / "methods.json").write_text(
        json.dumps(
            {
                "dataset_sources": ["owner/weights"],
                "model_sources": [],
                "infer_hints": ["rank_mean_ensemble"],
                "implement_steps": ["attach owner/weights and rank-mean"],
            }
        ),
        encoding="utf-8",
    )
    sys.path.insert(0, str(repo_root() / "competitions" / "rsna_knee"))
    from pipeline.recipe import apply_from_cards

    r = apply_from_cards(ws)
    assert r.ok
    applied = (ws / "pipeline" / "methods_applied.md").read_text(encoding="utf-8")
    assert "owner/weights" in applied
    assert "rank-mean" in applied
