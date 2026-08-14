from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.kaggle_api.models import KernelRow
from kaggle_agent.research.source_cards import (
    card_from_notebook,
    card_from_source_llm,
    cards_feasible,
    extra_source_urls,
    extract_datasets,
    extract_infer_hints,
    judge_cards_ready,
    skip_kernel,
    step_is_junk,
    valid_model_pin,
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


def test_extract_skips_junk_dataset_model():
    text = "attach dataset/model and https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights"
    ds = extract_datasets(text)
    assert "dataset/model" not in ds
    assert any("rsna-knee-weights" in x for x in ds)


def test_valid_model_pin_needs_five_or_four_parts():
    assert valid_model_pin("metaresearch/dinov2/PyTorch/small/1")
    assert not valid_model_pin("metaresearch/dinov2")
    assert not valid_model_pin("dinov2/pytorch")


def test_step_is_junk_dataset_model():
    assert step_is_junk("Attach datasets ['dataset/model'] and reuse their infer path.")
    assert not step_is_junk("Attach pilkwang/rsna-knee-weights and rank-average members.")


def test_cards_feasible_rejects_junk_sidecar(tmp_path: Path):
    workspace = tmp_path / "comp"
    (workspace / "pipeline").mkdir(parents=True)
    research = tmp_path / "research.md"
    research.write_text("## Method cards\n", encoding="utf-8")
    (workspace / "pipeline" / "methods.json").write_text(
        '{"implement_steps": ["Attach datasets [\'dataset/model\']"], '
        '"dataset_sources": ["dataset/model"]}\n',
        encoding="utf-8",
    )
    assert cards_feasible(workspace, research) is False


def test_extra_source_urls(tmp_path: Path):
    p = tmp_path / "research.md"
    p.write_text(
        "see https://www.kaggle.com/competitions/foo/discussion/733343 "
        "and https://arxiv.org/abs/2304.07193\n",
        encoding="utf-8",
    )
    urls = extra_source_urls(p)
    kinds = {k for k, _ in urls}
    assert "discussion" in kinds
    assert "paper" in kinds


def test_card_from_source_llm():
    class _Zen:
        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            import json

            return json.dumps(
                {
                    "claimed_public": "0.89",
                    "backbone": "DINOv2 ViT-S/14",
                    "labels": "report LLM table",
                    "cv": "GroupKFold report hash",
                    "inference": "study folders",
                    "copyable_next_step": "Attach pilkwang/rsna-knee-weights; rank-average.",
                    "do_not_copy": "H-flip",
                    "dataset_sources": ["pilkwang/rsna-knee-weights"],
                    "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
                }
            )

    card = card_from_source_llm(
        zen=_Zen(),  # type: ignore[arg-type]
        model="gpt-5.5",
        title="t",
        ref="u/n",
        url="https://www.kaggle.com/code/u/n",
        source_text="dinov2 weights",
        our_score="0.5",
        kind="kernel",
    )
    assert "pilkwang/rsna-knee-weights" in card
    assert "metaresearch/dinov2/PyTorch/small/1" in card


def test_judge_cards_ready_no_zen(tmp_path: Path):
    p = tmp_path / "source-a.md"
    p.write_text(
        "# A\n- copyable next step: Attach pilkwang/rsna-knee-weights\n",
        encoding="utf-8",
    )
    ok, reason = judge_cards_ready(None, "m", [p], "0.5")
    assert ok is True
    assert reason == "deterministic"


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
