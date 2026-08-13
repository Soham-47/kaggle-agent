from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.kaggle_api.models import KernelRow
from kaggle_agent.research.source_cards import (
    card_from_notebook,
    extract_datasets,
    extract_infer_hints,
    skip_kernel,
    write_methods_sidecar,
    run_source_card_research,
)


def test_skip_efficiency_lb():
    assert skip_kernel(KernelRow(ref="host/efficiency-lb", title="Efficiency LB"))
    assert not skip_kernel(KernelRow(ref="user/dino", title="DINOv2 [0.89]"))


def test_extract_datasets_and_hints():
    text = """
    https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights
    kaggle.com/models/metaresearch/dinov2/PyTorch/small/1
    also mention input/rsna-knee-abnormality-detection and user/some-kernel-title
    rank-mean the folds; iterdir on test study folders
    """
    ds = extract_datasets(text)
    assert any("rsna-knee-weights" in x for x in ds)
    assert not any(x.startswith("input/") for x in ds)
    assert "user/some-kernel-title" not in ds
    hints = extract_infer_hints(text)
    assert "rank_mean_ensemble" in hints
    assert "discover_test_ids_from_folders" in hints


def test_card_lists_copyable_step():
    row = KernelRow(ref="user/nb", title="Winner 0.89", url="https://www.kaggle.com/code/user/nb")
    card = card_from_notebook(row, "use dinov2 and rank-mean", "0.50")
    assert "copyable next step" in card
    assert "0.50" in card


def test_write_methods_sidecar_no_forced_dino(tmp_path: Path):
    card = tmp_path / "source-a.md"
    card.write_text(
        "# A\n- ref: u/a\n- claimed_public: 0.8\n"
        "- copyable next step: attach u/weights\n"
        "datasets_mentioned: owner/public-weights\n",
        encoding="utf-8",
    )
    out = write_methods_sidecar([card], tmp_path)
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert "dataset_sources" in data
    assert data["model_sources"] == [] or isinstance(data["model_sources"], list)
    assert "metaresearch/dinov2" not in data.get("model_sources", [])


def test_run_source_cards_writes_digest(tmp_path: Path):
    root = tmp_path / "ka"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "research.md").write_text("# research\n", encoding="utf-8")
    cache = root / "cache"
    cache.mkdir()
    client = KaggleClient(api=FakeKaggleApi()).connect()
    cards = run_source_card_research(
        client=client,
        competition="any-comp",
        cache_dir=cache,
        root=root,
        our_score="0.5",
        max_kernels=3,
    )
    assert cards
    research = (root / "memory" / "research.md").read_text(encoding="utf-8")
    assert "Method cards" in research or "Deep research digest" in research
