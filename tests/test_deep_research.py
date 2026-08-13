"""Tests for the deep-research stage (dzhng/deep-research pattern)."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.research.deep import (
    ArxivSource,
    DeepResearcher,
    GithubSource,
    KaggleSource,
    SourceHit,
    WebSource,
    _notebook_text,
)


class FakeLLM:
    """Strict-JSON callable stand-in for ZenClient.chat."""

    def __init__(self, plan: list[dict]) -> None:
        self._plan = iter(plan)
        self.calls: list[str] = []

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls.append(messages[-1]["content"])
        return json.dumps(next(self._plan))


class FakeKaggle:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def kernels_list(self, search=None, page_size=5, sort_by="voteCount"):  # noqa: ANN001
        return self._rows

    def kernels_pull(self, ref, path):  # noqa: ANN001
        (Path(path) / f"{ref.split('/')[-1]}.ipynb").write_text(
            json.dumps(
                {
                    "cells": [
                        {"cell_type": "markdown", "source": ["# Winner\n\nUses EfficientNet."]},
                        {"cell_type": "code", "source": ["import torch\nx = 1\n"]},
                    ]
                }
            ),
            encoding="utf-8",
        )


class FakeSource:
    kind = "fake"

    def __init__(self, hits: list[SourceHit]) -> None:
        self._hits = hits

    def search(self, query, limit=5):  # noqa: ANN001
        return self._hits[:limit]

    def content(self, hit, max_chars=25000):  # noqa: ANN001
        return f"Content of {hit.title}: deep learning wins. " * 20


def _researcher(llm: FakeLLM | None, hits: list[SourceHit], root: Path) -> DeepResearcher:
    cfg = DeepResearcher.__new__(DeepResearcher)  # placeholder; use real config below
    from kaggle_agent.research.deep import DeepResearchConfig

    config = DeepResearchConfig(depth=1, breadth=1, max_queries=5, max_fetches=10)
    return DeepResearcher(
        client=llm,
        model="gpt-5.5",
        config=config,
        sources=[FakeSource(hits)],
        root=root,
    )


def test_generate_queries_parses_json():
    llm = FakeLLM(
        [
            {
                "queries": [
                    {"query": "winning approach RSNA knee", "researchGoal": "find top method"}
                ]
            }
        ]
    )
    r = _researcher(llm, [], Path("/tmp"))
    queries = r._generate_queries("RSNA knee", [], 3)
    assert queries[0]["query"] == "winning approach RSNA knee"
    assert queries[0]["researchGoal"].startswith("find")


def test_generate_queries_fallback_without_llm():
    r = _researcher(None, [], Path("/tmp"))
    queries = r._generate_queries("some query", [], 2)
    assert queries == [{"query": "some query", "researchGoal": "some query"}]


def test_distill_without_llm_uses_snippets():
    hits = [SourceHit(url="u1", title="t1", snippet="snippet one", kind="fake")]
    r = _researcher(None, hits, Path("/tmp"))
    out = r._distill("q", hits, 2, 2)
    assert out["learnings"] == ["snippet one"]


def test_distill_with_llm_and_fetched_content():
    llm = FakeLLM([{"learnings": ["EfficientNet wins"], "followUpQuestions": ["why"]}])
    hits = [SourceHit(url="u1", title="t1", kind="fake")]
    r = _researcher(llm, hits, Path("/tmp"))
    out = r._distill("q", hits, 2, 2)
    assert out["learnings"] == ["EfficientNet wins"]
    assert "deep learning wins" in llm.calls[-1]


def test_research_loop_recurses_and_deduplicates():
    llm = FakeLLM(
        [
            {"queries": [{"query": "q1", "researchGoal": "g1"}]},
            {"learnings": ["A method", "A method"], "followUpQuestions": ["follow"]},
            {"queries": [{"query": "q2", "researchGoal": "g2"}]},
            {"learnings": ["B method"], "followUpQuestions": []},
        ]
    )
    hits = [SourceHit(url=f"u{i}", title=f"t{i}", kind="fake") for i in range(3)]
    r = _researcher(llm, hits, Path("/tmp"))
    learnings, visited = r.research("start", 1, 2, [], [])
    assert learnings == ["A method", "B method"]
    assert len(visited) >= 2


def test_research_respects_depth_zero():
    r = _researcher(FakeLLM([]), [], Path("/tmp"))
    learnings, visited = r.research("q", 2, 0, ["seed"], [])
    assert learnings == ["seed"]
    assert visited == []


def test_run_writes_report_and_digest(tmp_path: Path):
    llm = FakeLLM(
        [
            {"queries": [{"query": "q1", "researchGoal": "g1"}]},
            {"learnings": ["key insight"], "followUpQuestions": []},
            {"reportMarkdown": "# Deep report\n\nInsights here."},
        ]
    )
    hits = [SourceHit(url="https://x.com/1", title="x1", kind="fake")]
    r = _researcher(llm, hits, tmp_path)
    research_md = tmp_path / "memory" / "research.md"
    out = r.run("research the competition", research_md)
    assert not out.error
    assert out.learnings == ["key insight"]
    assert out.report_path is not None
    assert "# Deep report" in out.report_path.read_text()
    assert research_md.is_file()
    text = research_md.read_text()
    assert "## Deep research digest" in text
    assert "key insight" in text


def test_run_without_llm_errors_cleanly(tmp_path: Path):
    r = _researcher(None, [], tmp_path)
    out = r.run("q", tmp_path / "memory" / "research.md")
    assert out.error  # no llm and no real sources


def test_notebook_text_flattens_cells(tmp_path: Path):
    nb = tmp_path / "k.ipynb"
    nb.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title", " text"]},
                    {"cell_type": "code", "source": ["a=1"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    text = _notebook_text(nb)
    assert "# Title" in text
    assert "```python" in text
    assert "a=1" in text


def test_kaggle_source_pulls_and_parses_notebook(tmp_path: Path):
    class Wrapper:
        def __init__(self, rows):  # noqa: ANN001
            self.api = FakeKaggle(rows)
            self.last_kwargs: dict = {}

        def kernels_list(self, **kwargs):  # noqa: ANN001
            self.last_kwargs = kwargs
            return self.api.kernels_list(**kwargs)

        def kernels_pull(self, ref, path):  # noqa: ANN001
            return self.api.kernels_pull(ref, path)

    class TrackApi(FakeKaggle):
        def kernels_list(self, search=None, page_size=5, sort_by="voteCount", **kwargs):  # noqa: ANN001
            self.seen = {
                "search": search,
                "page_size": page_size,
                "sort_by": sort_by,
                **kwargs,
            }
            return super().kernels_list(search=search, page_size=page_size, sort_by=sort_by)

    api = TrackApi(
        [
            {
                "ref": "owner/winner-kernel",
                "title": "Winner Kernel",
                "total_votes": 100,
            }
        ]
    )

    class Wrap:
        def __init__(self, api):  # noqa: ANN001
            self.api = api

    wrap = Wrap(api)
    src = KaggleSource(wrap, "rsna-knee-abnormality-detection", tmp_path / "cache")
    hits = src.search("winning knee", 5)
    assert hits[0].url == "https://www.kaggle.com/code/owner/winner-kernel"
    assert getattr(api, "seen", {}).get("competition") == "rsna-knee-abnormality-detection"
    content = src.content(hits[0])
    assert "EfficientNet" in content
    assert "import torch" in content


def test_digest_lists_kaggle_notebooks_before_unrelated_arxiv():
    from kaggle_agent.research.deep import DeepResearchConfig, DeepResearcher

    r = DeepResearcher(
        None,
        "gpt-5.5",
        DeepResearchConfig(),
        [],
        Path("/tmp"),
    )
    md = r.digest_markdown(
        ["use DINOv2 at 336px"],
        [
            "http://arxiv.org/abs/1405.0546v2",
            "https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1",
            "http://arxiv.org/abs/2209.10033v1",
            "https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution",
        ],
    )
    kaggle_pos = md.find("kaggle.com/code/")
    arxiv_pos = md.find("arxiv.org")
    assert kaggle_pos != -1
    assert kaggle_pos < arxiv_pos
    assert "pilkwang/rsna-knee-baseline-v1" in md


def test_kaggle_source_search_failure_empty():
    class Boom:
        def kernels_list(self, **kwargs):  # noqa: ANN001
            raise RuntimeError("boom")

        def kernels_pull(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("boom")

    src = KaggleSource(Boom(), "c", Path("/tmp/cache-nope"))
    assert src.search("q") == []
    assert src.content(SourceHit(url="https://www.kaggle.com/code/o/s", title="t")) == ""


def test_arxiv_source_search_failure_empty(monkeypatch):
    import kaggle_agent.research.deep as mod

    def boom(*a, **k):  # noqa: ANN001
        raise OSError("offline")

    monkeypatch.setattr(mod.urllib.request, "urlopen", boom)
    src = ArxivSource()
    assert src.search("knee mri") == []


def test_github_source_content_falls_back_to_snippet():
    src = GithubSource()
    hit = SourceHit(url="https://github.com/a/b", title="a/b", snippet="desc only")
    assert src.content(hit) == "desc only"
