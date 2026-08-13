"""Report labels + metadata ranker must produce varying ranks, not constant 0.5."""

from pathlib import Path

from kaggle_agent.paths import repo_root


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

    root = repo_root()
    ws = root / "competitions" / "rsna_knee"
    sys.path.insert(0, str(ws))
    from pipeline.recipe import apply_recipe
    from pipeline.ranker import load_ranker, load_series_by_study, predict_studies

    r = apply_recipe(ws, data_dir=root / "data")
    assert r.ok
    assert r.n_train >= 58
    model = load_ranker(Path(r.weights_path))
    ids = [
        "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191",
        "1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776",
        "1.2.826.0.1.3680043.8.498.10067514707072572280263481548497591402",
    ]
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
    shutil.copytree(real / "data", root / "data")
    shutil.copytree(real / "competitions" / "rsna_knee" / "pipeline", root / "competitions" / "rsna_knee" / "pipeline")
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="recipe1")
    text = pkg.notebook_path.read_text(encoding="utf-8")
    assert "def extract_labels" in text
    assert "from reports import" not in text
    assert "LGBMClassifier" in text or "LogisticRegression" in text
    assert "0.5" in text  # prior fallback exists
    assert "constant probabilities from sample header" not in text
    assert (pkg.folder / "train.csv").is_file()
    assert (pkg.folder / "reports.py").is_file()
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
