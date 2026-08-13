from pathlib import Path

from kaggle_agent.research.browser import (
    BrowserResearcher,
    competition_page_urls,
    merge_browser_into_research_md,
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
