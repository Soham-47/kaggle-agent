"""Stage-view context packs from the one markdown store."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from kaggle_agent.paths import memory_dir

CORE = ("MEMORY.md", "COMPETITION.md", "state.md", "research.md")
_DIGEST_HEADINGS = ("## Method cards", "## Deep research digest")
_SCORE_RE = re.compile(r"^-\s*public_score:\s*(\S+)", re.M)
_EXPIRES_RE = re.compile(r"^-\s*expires:\s*(\S+)", re.M)
_SUPERSEDED_RE = re.compile(r"^-\s*superseded:\s*\S+", re.M)
_VIEWS = frozenset({"research", "plan", "code", "heal", "ops"})
_RETRIEVE_SCOPES = {
    "cards": ("research-deep", "source-*.md"),
    "research": (".", "research.md"),
    "experiments": ("experiments", "*.md"),
    "memory": (".", None),
}
_MEMORY_NAMES = ("MEMORY.md", "COMPETITION.md", "state.md")


@dataclass
class ContextPack:
    sections: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    experiment_paths: list[Path] = field(default_factory=list)
    view: str = "plan"
    caps: dict[str, int] = field(default_factory=dict)
    pack_cap: int = 14000

    def get(self, name: str, default: str = "") -> str:
        return self.sections.get(name, default)

    def as_prompt_block(
        self,
        max_chars_per_section: int | None = None,
        *,
        order: list[str] | None = None,
    ) -> str:
        keys = list(self.sections)
        if order:
            ordered = [k for k in order if k in self.sections]
            keys = ordered + [k for k in keys if k not in ordered]
        parts: list[str] = []
        used = 0
        for name in keys:
            text = self.sections[name]
            cap = self.caps.get(name, max_chars_per_section or 3000)
            body = text if len(text) <= cap else text[: cap - 15] + "\n...[cut]"
            chunk = f"## {name}\n\n{body.strip()}\n"
            if used + len(chunk) > self.pack_cap:
                remain = self.pack_cap - used
                if remain < 40:
                    break
                chunk = chunk[: remain - 15] + "\n...[cut]\n"
            parts.append(chunk)
            used += len(chunk)
        return "\n".join(parts)


def _research_with_digest_first(text: str) -> str:
    chunks: list[str] = []
    rest = text
    for heading in _DIGEST_HEADINGS:
        if heading not in rest:
            continue
        before, after = rest.split(heading, 1)
        body, _, tail = after.partition("\n## ")
        chunks.append((heading + body).strip())
        rest = before + (("## " + tail) if tail else "")
    if not chunks:
        return text
    return "\n\n".join(chunks) + "\n\n" + rest.strip()


def _heading_excerpt(text: str, *headings: str) -> str:
    found: list[str] = []
    for heading in headings:
        if heading not in text:
            continue
        after = text.split(heading, 1)[1]
        body, _, _ = after.partition("\n## ")
        found.append((heading + body).strip())
    return "\n\n".join(found)


def _kv_lines(text: str, keys: tuple[str, ...]) -> str:
    keep: list[str] = []
    wanted = set(keys)
    for line in text.splitlines():
        m = re.match(r"^-\s*([a-zA-Z0-9_]+):", line.strip())
        if m and m.group(1) in wanted:
            keep.append(line.strip())
    return "\n".join(keep)


def _note_is_expired(text: str) -> bool:
    """True when the note has an expires date in the past."""
    m = _EXPIRES_RE.search(text)
    if not m:
        return False
    try:
        exp = date.fromisoformat(m.group(1).strip())
    except (ValueError, TypeError):
        return False
    return exp < date.today()


def _note_is_superseded(text: str) -> bool:
    """True when the note carries a truthy superseded field."""
    m = _SUPERSEDED_RE.search(text)
    if not m:
        return False
    val = m.group(0).split(":", 1)[1].strip().lower()
    return val in {"yes", "true", "1"}


def _note_is_stale(text: str) -> bool:
    return _note_is_expired(text) or _note_is_superseded(text)


def _evals_section(base: Path, cap: int = 600) -> str:
    """Compact summary of the last eval report, or '' if absent."""
    path = base / "daily" / "eval_report.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    checks = data.get("checks") or []
    failed = [c for c in checks if isinstance(c, dict) and not c.get("ok")]
    lines = [f"passed={data.get('passed')} ran_at={data.get('ran_at', '')}"]
    if failed:
        lines.append("failed checks:")
        for c in failed[:6]:
            lines.append(f"- {c.get('id')}: {str(c.get('detail', ''))[:140]}")
    return "\n".join(lines)[:cap]


def _exp_score(path: Path) -> float | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _SCORE_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    if raw in {"none", "n/a", "nan", ""}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def pick_experiments(exp_dir: Path, n: int) -> list[Path]:
    files = [p for p in exp_dir.glob("*.md") if p.is_file()]
    files.sort(
        key=lambda p: (
            _exp_score(p) is None,
            -(_exp_score(p) or 0.0),
            -p.stat().st_mtime,
        )
    )
    fresh = [p for p in files if not _note_is_stale(_safe_read(p))]
    return fresh[:n] if fresh else files[:n]


def pick_cards(deep: Path, n: int = 2) -> list[Path]:
    cards = [p for p in deep.glob("source-*.md") if p.is_file()]
    cards.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    fresh = [p for p in cards if not _note_is_stale(_safe_read(p))]
    if fresh:
        return fresh[:n]
    # Fallback: newest card when all are stale (avoids empty pack section)
    return cards[:1] if cards else []


def build_context_pack(
    root: Path | None = None,
    *,
    view: str = "plan",
    workspace: Path | None = None,
    last_experiments: int | None = None,
    plan_text: str = "",
) -> ContextPack:
    if view not in _VIEWS:
        view = "plan"
    base = memory_dir(root)
    pack = ContextPack(view=view)
    if view == "research":
        _fill_research(pack, base)
    elif view == "code":
        _fill_code(pack, base, workspace, plan_text)
    elif view in {"heal", "ops"}:
        _fill_plan(pack, base, workspace, last_experiments if last_experiments is not None else 2)
        pack.view = view
    else:
        _fill_plan(pack, base, workspace, last_experiments if last_experiments is not None else 2)
    return pack


def _read(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _add(pack: ContextPack, key: str, text: str, cap: int) -> None:
    pack.sections[key] = text
    pack.caps[key] = cap


def _fill_research(pack: ContextPack, base: Path) -> None:
    pack.pack_cap = 12000
    mem = _read(base / "MEMORY.md")
    if mem is None:
        pack.missing.append("MEMORY.md")
    else:
        excerpt = _heading_excerpt(mem, "## Lessons", "## Active contest") or mem
        _add(pack, "MEMORY.md", excerpt, 1500)
    for rel, cap in (("COMPETITION.md", 1500),):
        raw = _read(base / rel)
        if raw is None:
            pack.missing.append(rel)
        else:
            _add(pack, rel, raw, cap)
    st = _read(base / "state.md")
    if st is None:
        pack.missing.append("state.md")
    else:
        _add(
            pack,
            "state.md",
            _kv_lines(st, ("public_best", "proposals_used", "max_proposals", "note")) or st,
            800,
        )
    research = _read(base / "research.md")
    if research is None:
        pack.missing.append("research.md")
    else:
        _add(pack, "research.md", _research_with_digest_first(research), 1200)
    deep = base / "research-deep"
    if deep.is_dir():
        for path in pick_cards(deep, 2):
            _add(pack, f"research-deep/{path.name}", path.read_text(encoding="utf-8"), 2000)
    exp_dir = base / "experiments"
    if exp_dir.is_dir():
        files = pick_experiments(exp_dir, 2)
        pack.experiment_paths = files
        for p in files:
            _add(pack, f"experiments/{p.name}", p.read_text(encoding="utf-8"), 1500)


def _fill_plan(
    pack: ContextPack,
    base: Path,
    workspace: Path | None,
    last_n: int,
) -> None:
    pack.pack_cap = 14000
    for rel, cap in (
        ("MEMORY.md", 2000),
        ("COMPETITION.md", 1500),
        ("state.md", 1000),
    ):
        raw = _read(base / rel)
        if raw is None:
            pack.missing.append(rel)
        else:
            _add(pack, rel, raw, cap)
    research = _read(base / "research.md")
    if research is None:
        pack.missing.append("research.md")
    else:
        digest = _heading_excerpt(research, *_DIGEST_HEADINGS) or _research_with_digest_first(
            research
        )
        _add(pack, "research.md", digest if digest.strip() else research, 2500)
    deep = base / "research-deep"
    if deep.is_dir():
        for path in pick_cards(deep, 2):
            _add(pack, f"research-deep/{path.name}", path.read_text(encoding="utf-8"), 2000)
    exp_dir = base / "experiments"
    if exp_dir.is_dir():
        files = pick_experiments(exp_dir, last_n)
        pack.experiment_paths = files
        for p in files:
            _add(pack, f"experiments/{p.name}", p.read_text(encoding="utf-8"), 1500)
    heal = _read(base / "heal.md")
    if heal:
        excerpt = _kv_lines(heal, ("decision_next", "note")) or heal
        _add(pack, "heal.md", excerpt, 400)
    evals = _evals_section(base)
    if evals:
        _add(pack, "Evals (last cycle)", evals, 800)
    if workspace is not None:
        methods = workspace / "pipeline" / "methods.json"
        raw = _read(methods)
        if raw:
            _add(pack, "methods.json", raw, 1500)


def _fill_code(
    pack: ContextPack,
    base: Path,
    workspace: Path | None,
    plan_text: str,
) -> None:
    pack.pack_cap = 12000
    comp = _read(base / "COMPETITION.md")
    if comp is None:
        pack.missing.append("COMPETITION.md")
    else:
        _add(pack, "COMPETITION.md", comp, 1500)
    st = _read(base / "state.md")
    if st:
        _add(pack, "state.md", _kv_lines(st, ("public_best",)) or st, 200)
    if plan_text.strip():
        _add(pack, "plan_text", plan_text, 3000)
    deep = base / "research-deep"
    if deep.is_dir():
        for path in pick_cards(deep, 2):
            _add(pack, f"research-deep/{path.name}", path.read_text(encoding="utf-8"), 2000)
    if workspace is not None:
        methods = _read(workspace / "pipeline" / "methods.json")
        if methods:
            _add(pack, "methods.json", methods, 1500)
        recipe = _read(workspace / "pipeline" / "kernel_recipe.py")
        if recipe:
            _add(pack, "kernel_recipe.py", recipe, 4000)
    evals = _evals_section(base)
    if evals:
        _add(pack, "Evals (last cycle)", evals, 800)
    exp_dir = base / "experiments"
    if exp_dir.is_dir():
        files = pick_experiments(exp_dir, 1)
        pack.experiment_paths = files
        for p in files:
            _add(pack, f"experiments/{p.name}", p.read_text(encoding="utf-8"), 1500)


def retrieve(
    root: Path | None,
    query: str,
    scope: str = "cards",
    *,
    max_hits: int = 4,
    window: int = 800,
) -> str:
    q = (query or "").strip().lower()
    if not q:
        return "empty query"
    if scope not in _RETRIEVE_SCOPES:
        return f"unknown scope {scope}"
    base = memory_dir(root)
    sub, pattern = _RETRIEVE_SCOPES[scope]
    folder = (base / sub).resolve()
    if not str(folder).startswith(str(base.resolve())):
        return "refuse: path"
    files: list[Path] = []
    if scope == "memory":
        files = [base / n for n in _MEMORY_NAMES if (base / n).is_file()]
    elif folder.is_dir() or folder.is_file():
        if folder.is_file():
            files = [folder]
        else:
            files = sorted(folder.glob(pattern or "*"), key=lambda p: p.stat().st_mtime, reverse=True)
    hits: list[str] = []
    for path in files:
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("deep-") or "secret" in name or "daily" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if _note_is_stale(text):
            continue
        idx = text.lower().find(q)
        if idx < 0:
            continue
        start = max(0, idx - window // 4)
        chunk = text[start : start + window]
        rel = path.relative_to(base)
        hits.append(f"## {rel}\n{chunk.strip()}")
        if len(hits) >= max_hits:
            break
    return "\n\n".join(hits) if hits else "no hits"
