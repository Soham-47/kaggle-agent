"""Load the few memory files that matter for a cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.paths import memory_dir

# Keep this list tiny — every file costs tokens every cycle.
CORE = ("MEMORY.md", "COMPETITION.md", "state.md", "research.md")
_DIGEST = "## Deep research digest"


@dataclass
class ContextPack:
    sections: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    experiment_paths: list[Path] = field(default_factory=list)

    def get(self, name: str, default: str = "") -> str:
        return self.sections.get(name, default)

    def as_prompt_block(self, max_chars_per_section: int = 3000) -> str:
        parts: list[str] = []
        for name, text in self.sections.items():
            body = text if len(text) <= max_chars_per_section else text[: max_chars_per_section - 15] + "\n...[cut]"
            parts.append(f"## {name}\n\n{body.strip()}\n")
        return "\n".join(parts)


def _research_with_digest_first(text: str) -> str:
    """Put the method-card digest before the long Kaggle snapshot so PLAN sees it."""
    if _DIGEST not in text:
        return text
    before, after = text.split(_DIGEST, 1)
    digest = (_DIGEST + after).strip()
    return digest + "\n\n" + before.strip()


def build_context_pack(root: Path | None = None, *, last_experiments: int = 2) -> ContextPack:
    base = memory_dir(root)
    pack = ContextPack()
    for rel in CORE:
        path = base / rel
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
            pack.sections[rel] = (
                _research_with_digest_first(raw) if rel == "research.md" else raw
            )
        else:
            pack.missing.append(rel)

    deep = base / "research-deep"
    if deep.is_dir():
        cards = sorted(
            deep.glob("source-*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:2]
        for path in cards:
            pack.sections[f"research-deep/{path.name}"] = path.read_text(encoding="utf-8")

    exp_dir = base / "experiments"
    if exp_dir.is_dir():
        files = sorted(
            [p for p in exp_dir.glob("*.md")],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:last_experiments]
        pack.experiment_paths = files
        for p in files:
            pack.sections[f"experiments/{p.name}"] = p.read_text(encoding="utf-8")
    return pack
