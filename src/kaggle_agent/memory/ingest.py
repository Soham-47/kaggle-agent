"""Load the few memory files that matter for a cycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.paths import memory_dir

# Keep this list tiny — every file costs tokens every cycle.
CORE = ("MEMORY.md", "COMPETITION.md", "state.md", "research.md")
_DIGEST_HEADINGS = ("## Method cards", "## Deep research digest")


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
    """Put implementable card/digest sections before the long Kaggle snapshot."""
    chunks: list[str] = []
    rest = text
    for heading in _DIGEST_HEADINGS:
        if heading not in rest:
            continue
        before, after = rest.split(heading, 1)
        # keep only this section body until the next ## heading
        body, _, tail = after.partition("\n## ")
        chunks.append((heading + body).strip())
        rest = before + (("## " + tail) if tail else "")
    if not chunks:
        return text
    return "\n\n".join(chunks) + "\n\n" + rest.strip()


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
