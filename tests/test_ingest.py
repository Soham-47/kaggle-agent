from kaggle_agent.memory.ingest import CORE, build_context_pack
from kaggle_agent.paths import repo_root


def test_context_pack_is_lean():
    pack = build_context_pack(repo_root())
    assert CORE == ("MEMORY.md", "COMPETITION.md", "state.md", "research.md")
    for name in CORE:
        assert name in pack.sections
    # no old bloated files
    for bad in ("USER.md", "SKILLS.md", "GOALS.md", "HEARTBEAT.md", "index.md"):
        assert bad not in pack.sections
    assert pack.missing == []


def test_prompt_block():
    block = build_context_pack(repo_root()).as_prompt_block()
    assert "## MEMORY.md" in block
    assert "rsna" in block.lower() or "RSNA" in block


def test_context_pack_prefers_digest_and_source_cards(tmp_path):
    from pathlib import Path

    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("user prefs", encoding="utf-8")
    (mem / "COMPETITION.md").write_text("active contest", encoding="utf-8")
    (mem / "state.md").write_text("phase: IDLE", encoding="utf-8")
    (mem / "research.md").write_text(
        "long snapshot " * 200 + "\n## Deep research digest\n- next: attach owner/weights\n",
        encoding="utf-8",
    )
    deep = mem / "research-deep"
    deep.mkdir()
    (deep / "source-winner.md").write_text(
        "# winner\n- copyable next step: rank-mean folds\n", encoding="utf-8"
    )
    pack = build_context_pack(tmp_path)
    block = pack.as_prompt_block()
    assert "Deep research digest" in block
    assert block.find("Deep research digest") < block.find("long snapshot")
    assert any(k.startswith("research-deep/") for k in pack.sections)
    assert "rank-mean folds" in block
