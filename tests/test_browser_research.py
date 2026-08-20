import json
from pathlib import Path

from kaggle_agent.research.browser import (
    BrowserResearcher,
    competition_page_urls,
    default_serp,
    merge_browser_into_research_md,
    search_via_browser_harness,
)


def test_competition_urls():
    urls = competition_page_urls("rsna-knee-abnormality-detection")
    assert "overview" in urls
    assert urls["discussion"].endswith("/discussion")


def test_collect_with_fake_fetch():
    def fake(url: str, max_chars: int = 12000) -> str:
        if "overview" in url:
            return "Detect twelve knee abnormalities. Metric is macro ROC AUC. " * 3
        if "discussion" in url:
            return "Discussion: baseline 2D CNN works well for study-level labels. " * 3
        return "other"

    notes = BrowserResearcher(fetch=fake).collect(
        "rsna-knee-abnormality-detection", pages=("overview", "discussion")
    )
    assert "overview" in notes.pages
    assert "discussion" in notes.pages
    assert not notes.errors
    md = notes.to_markdown_section()
    assert "Browser (read-only)" in md
    assert "macro ROC AUC" in md or "abnormalities" in md


def test_merge_browser_section(tmp_path: Path):
    research = tmp_path / "research.md"
    research.write_text("# research\n\n## Schema\n\n- x\n", encoding="utf-8")

    def fake(url: str, max_chars: int = 12000) -> str:
        return "Page text about evaluation metric and rules. " * 5

    notes = BrowserResearcher(fetch=fake).collect("slug", pages=("overview",))
    merge_browser_into_research_md(research, notes)
    text = research.read_text(encoding="utf-8")
    assert "## Schema" in text
    assert "## Browser (read-only)" in text

    # Second merge replaces browser section, keeps schema
    notes2 = BrowserResearcher(fetch=lambda u, m=12000: "Updated browser content here. " * 5).collect(
        "slug", pages=("overview",)
    )
    merge_browser_into_research_md(research, notes2)
    text2 = research.read_text(encoding="utf-8")
    assert text2.count("## Browser (read-only)") == 1
    assert "Updated browser content" in text2


class _FakeProc:
    def __init__(self, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def test_search_via_browser_harness_parses_google_results(monkeypatch):
    import kaggle_agent.research.browser as br

    payload = json.dumps(
        [
            {
                "title": "RSNA Knee | Kaggle",
                "url": "https://www.kaggle.com/c/rsna-knee-abnormality-detection",
                "snippet": "competition page",
            },
            {
                "title": "A paper",
                "url": "https://arxiv.org/abs/2401.00000",
                "snippet": "abstract text",
            },
            {
                "title": "Accessibility help",
                "url": "https://support.google.com/websearch",
                "snippet": "",
            },
            {"title": "", "url": "https://example.com/bad", "snippet": ""},
        ]
    )
    monkeypatch.setattr(br.shutil, "which", lambda name: "/usr/local/bin/browser-use")
    monkeypatch.setattr(br.subprocess, "run", lambda *a, **k: _FakeProc(payload + "\n"))

    hits = search_via_browser_harness("rsna knee", limit=5)
    assert len(hits) == 2
    assert hits[0][0] == "RSNA Knee | Kaggle"
    assert hits[0][1].startswith("https://www.kaggle.com/c/")
    assert hits[1][2] == "abstract text"


def test_search_via_browser_harness_skips_non_http(monkeypatch):
    import kaggle_agent.research.browser as br

    payload = json.dumps(
        [{"title": "T", "url": "javascript:void(0)", "snippet": ""}]
    )
    monkeypatch.setattr(br.shutil, "which", lambda name: "/usr/local/bin/browser-use")
    monkeypatch.setattr(br.subprocess, "run", lambda *a, **k: _FakeProc(payload + "\n"))
    assert search_via_browser_harness("q", 5) == []


def test_default_serp_prefers_harness_then_falls_back(monkeypatch):
    import kaggle_agent.research.browser as br

    calls = {"harness": 0, "ddg": 0}

    def fake_harness(query: str, limit: int = 5):
        calls["harness"] += 1
        return [("K", "https://www.kaggle.com/c/rsna-knee-abnormality-detection", "")]

    def fake_ddg(query: str, limit: int = 5):
        calls["ddg"] += 1
        return [("D", "https://example.com/d", "")]

    monkeypatch.setattr(br, "search_via_browser_harness", fake_harness)
    monkeypatch.setattr(br, "search_via_ddg_http", fake_ddg)
    monkeypatch.setattr(br.shutil, "which", lambda name: "/usr/local/bin/browser-use")

    serp = default_serp(prefer_browser_harness=True)
    assert serp("rsna knee") == fake_harness("rsna knee")
    assert calls["ddg"] == 0

    monkeypatch.setattr(br, "search_via_browser_harness", lambda q, l=5: [])
    assert serp("rsna knee") == [("D", "https://example.com/d", "")]
    assert calls["ddg"] == 1

    monkeypatch.setattr(br.shutil, "which", lambda name: None)
    assert default_serp(prefer_browser_harness=True)("rsna knee") == [
        ("D", "https://example.com/d", "")
    ]
    assert calls["ddg"] == 2
