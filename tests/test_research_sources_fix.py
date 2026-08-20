"""Real Kaggle sources for the fleet: discussions + datasets, query dedup,
nudge-spam control, and sequential-research stall control."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.research.deep import (
    DeepResearchConfig,
    DeepResearcher,
    DatasetSource,
    DiscussionSource,
    SourceHit,
)
from kaggle_agent.research.fleet import AGENT_SPECS, make_fleet_tools


# --- real Kaggle sources ---


class FakeDiscussionsApi:
    def __init__(self) -> None:
        self.shown: list[int] = []

    def competition_list_topics(self, competition, sort_by=None, page=None):  # noqa: ANN001
        assert competition
        return SimpleNamespace(
            topics=[
                SimpleNamespace(
                    id=11,
                    title="Leak warning: train/test overlap",
                    topic_url="https://www.kaggle.com/c/x/discussion/11",
                    votes=42,
                    comment_count=7,
                ),
                SimpleNamespace(
                    id=12,
                    title="About the data",
                    topic_url="https://www.kaggle.com/c/x/discussion/12",
                    votes=5,
                    comment_count=1,
                ),
            ]
        )

    def forums_topic_show(self, topic_id, page_size=None, page_token=None):  # noqa: ANN001
        self.shown.append(topic_id)
        topic = SimpleNamespace(
            id=topic_id,
            title="Leak warning",
            content="The test set overlaps the train set by 3%. Do not trust CV.",
        )
        comments = [
            SimpleNamespace(id=1, content="Confirmed, same series appear twice."),
            SimpleNamespace(id=2, content=""),
        ]
        return topic, comments, ""


def _with_api(fake: object) -> SimpleNamespace:
    return SimpleNamespace(api=fake)


def test_discussion_source_search_and_content():
    src = DiscussionSource(_with_api(FakeDiscussionsApi()), "rsna-knee-abnormality-detection")
    hits = src.search("leak", 5)
    assert len(hits) == 2
    assert hits[0].kind == "discussion"
    assert "discussion/11" in hits[0].url
    assert "42" in hits[0].snippet
    body = src.content(hits[0])
    assert "overlaps the train set" in body
    assert "Confirmed, same series" in body


def test_discussion_source_fails_soft_without_api():
    class _Broken:
        def competition_list_topics(self, *a, **k):  # noqa: ANN001
            raise RuntimeError("offline")

    src = DiscussionSource(_with_api(_Broken()), "x")
    assert src.search("q", 5) == []


class FakeDatasetsApi:
    def dataset_list_with_response(self, search=None, sort_by=None, page_size=None, **kwargs):  # noqa: ANN001
        return SimpleNamespace(
            datasets=[
                SimpleNamespace(
                    ref="owner/rsna-knee-preprocessed",
                    title="RSNA Knee Preprocessed PNGs",
                    subtitle="Resized 256px images",
                    description="All train/test DICOM converted to PNG at 256px.",
                    url="https://www.kaggle.com/datasets/owner/rsna-knee-preprocessed",
                    download_count=1234,
                    vote_count=56,
                ),
                SimpleNamespace(
                    ref="other/rsna-knee-preprocessed",
                    title="Decoy with same title",
                    subtitle="Unrelated mirror",
                    description="WRONG dataset description.",
                    url="https://www.kaggle.com/datasets/other/rsna-knee-preprocessed",
                    download_count=1,
                    vote_count=1,
                ),
            ]
        )


def test_dataset_source_search_and_content():
    src = DatasetSource(_with_api(FakeDatasetsApi()), "rsna-knee-abnormality-detection")
    hits = src.search("rsna knee png", 5)
    assert len(hits) == 2
    assert hits[0].kind == "dataset"
    assert "owner/rsna-knee-preprocessed" in hits[0].url
    assert "1234" in hits[0].snippet
    body = src.content(hits[0])
    assert "PNG at 256px" in body
    assert "WRONG dataset" not in body
    decoy_body = src.content(hits[1])
    assert "WRONG dataset description" in decoy_body


def test_fleet_discussions_datasets_use_real_kinds():
    assert AGENT_SPECS["discussions"].search_kinds == ("discussion",)
    assert AGENT_SPECS["datasets"].search_kinds == ("dataset",)
    for name in ("discussions", "datasets"):
        tools = make_fleet_tools(
            AGENT_SPECS[name],
            search_fn=lambda q, k, l: f"hit {k}",
            fetch_fn=lambda u: "body",
            write_fn=lambda ref, md: "/tmp/c.md",
        )
        assert "search" in tools
        out = tools["search"](query="knee", kind="web")
        kind = AGENT_SPECS[name].search_kinds[0]
        assert f"hit {kind}" in out


# --- query dedup in deep research ---


class FakeLLM:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if not self._replies:
            return json.dumps({"queries": [], "learnings": []})
        return json.dumps(self._replies.pop(0))


class FakeSource:
    kind = "fake"

    def __init__(self, hits: list[SourceHit]) -> None:
        self._hits = hits
        self.searches: list[str] = []

    def search(self, query, limit=5):  # noqa: ANN001
        self.searches.append(query)
        return self._hits

    def content(self, hit, max_chars=25000):  # noqa: ANN001
        return f"Content {hit.title}" * 10


def _researcher(llm, hits, root: Path) -> DeepResearcher:
    return DeepResearcher(
        client=llm,
        model="m",
        config=DeepResearchConfig(depth=2, breadth=2, max_queries=6, max_fetches=10),
        sources=[FakeSource(hits)],
        root=root,
    )


def test_deep_research_skips_repeated_queries():
    """A repeated query runs once; new queries still run."""
    llm = FakeLLM(
        [
            {"queries": [{"query": "knee mri", "researchGoal": "g"}]},
            {"learnings": ["A"], "followUpQuestions": ["more on knee mri"]},
            {
                "queries": [
                    {"query": "knee dataset", "researchGoal": "g"},
                    {"query": "Knee MRI", "researchGoal": "g"},
                ]
            },
            {"learnings": ["B"], "followUpQuestions": []},
        ]
    )
    hits = [SourceHit(url="u1", title="t1", kind="fake")]
    src = FakeSource(hits)
    r = DeepResearcher(
        client=llm,
        model="m",
        config=DeepResearchConfig(depth=2, breadth=2, max_queries=6, max_fetches=10),
        sources=[src],
        root=Path("/tmp"),
    )
    learnings, _ = r.research("start", 2, 2, [], [])
    assert "A" in learnings and "B" in learnings
    assert src.searches == ["knee mri", "knee dataset"]
    assert r._query_count <= 3


def test_deep_research_all_duplicates_stops_node():
    llm = FakeLLM(
        [
            {"queries": [{"query": "same", "researchGoal": "g"}]},
            {"learnings": ["A"], "followUpQuestions": ["more"]},
            {"queries": [{"query": "same", "researchGoal": "g"}]},
            {"learnings": ["B"], "followUpQuestions": []},
        ]
    )
    hits = [SourceHit(url="u1", title="t1", kind="fake")]
    src = FakeSource(hits)
    r = DeepResearcher(
        client=llm,
        model="m",
        config=DeepResearchConfig(depth=2, breadth=2, max_queries=6, max_fetches=10),
        sources=[src],
        root=Path("/tmp"),
    )
    learnings, _ = r.research("start", 2, 2, [], [])
    assert src.searches == ["same"]


# --- nudge spam: one nudge per stall episode, not per turn ---


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self._replies = list(replies)

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        if not self._replies:
            return json.dumps({"tool": "done", "args": {}})
        return json.dumps(self._replies.pop(0))


def test_nudge_logged_once_per_episode():
    zen = _ScriptedZen([{"tool": "search", "args": {"query": "q"}}] * 12)
    logs: list[str] = []
    agent = StageAgent(
        zen,
        "m",
        {"search": lambda **_a: "hit"},
        StageAgentConfig(max_minutes=5, max_tool_turns=12),
        system="s",
        log=lambda msg: logs.append(msg),
        name="research",
        stall_after=3,
        stall_nudge="Stall: write now.",
    )
    out = agent.run("ctx")
    nudge_logs = [l for l in logs if "nudge" in l]
    assert out.turns == 12
    assert len(nudge_logs) == 1


def test_nudge_resets_after_write_between_episodes():
    """A write ends one stall episode; the next episode nudges again."""
    zen = _ScriptedZen([{"tool": "search", "args": {"query": "q"}}] * 12)
    logs: list[str] = []
    agent = StageAgent(
        zen,
        "m",
        {"search": lambda **_a: "hit", "write_card": lambda **_a: "ok"},
        StageAgentConfig(max_minutes=5, max_tool_turns=18),
        system="s",
        log=lambda msg: logs.append(msg),
        name="research",
        stall_after=2,
        stall_nudge="Stall: write now.",
        stall_force=("write_card", {"ref": "c", "md": "x"}),
    )
    out = agent.run("ctx")
    nudge_logs = [l for l in logs if "nudge" in l]
    assert out.stop_reason == "turn_cap"
    assert len(nudge_logs) >= 2


def test_research_agent_forwards_stall_control():
    """Sequential research accepts stall params: nudges once, no force."""
    from kaggle_agent.config import ResearchAgentSettings
    from kaggle_agent.research.agent import ResearchAgent

    zen = _ScriptedZen([{"tool": "search", "args": {"query": "q"}}] * 12)
    logs: list[str] = []
    agent = ResearchAgent(
        zen,
        "m",
        {"search": lambda **_a: "hit"},
        ResearchAgentSettings(max_minutes=5, max_tool_turns=12),
        log=lambda msg: logs.append(msg),
        stall_after=3,
        stall_nudge="Stall: write now.",
    )
    out = agent.run("ctx")
    assert out.stop_reason == "turn_cap"
    nudge_logs = [l for l in logs if "nudge" in l]
    assert len(nudge_logs) == 1
