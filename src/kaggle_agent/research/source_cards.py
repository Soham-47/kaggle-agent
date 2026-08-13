"""Competition-agnostic method cards from top public kernels.

One in-process worker per source (ThreadPoolExecutor). PLAN/CODE read the
cards and methods.json. This is not a second memory store.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable  # noqa: I001

from kaggle_agent.kaggle_api.models import KernelRow
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.browser import merge_section_into_research_md
from kaggle_agent.research.deep import KaggleSource, SourceHit

LogFn = Callable[[str], None]

_SKIP_DATASET = (
    "competitions/",
    "kaggle/input",
    "kaggle/working",
)


def _slug(ref: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (ref or "src").lower()).strip("-")[:60]


def skip_kernel(row: KernelRow) -> bool:
    title = (row.title or "").lower()
    ref = (row.ref or "").lower()
    return "efficiency lb" in title or "efficiency-lb" in ref


_NOT_A_DATASET_OWNER = {
    "input",
    "data",
    "mnt",
    "tmp",
    "home",
    "torch",
    "kaggle",
    "working",
    "output",
}


def extract_datasets(text: str) -> list[str]:
    """Only keep explicit Kaggle dataset/model refs, not kernel slugs or local paths."""
    found = re.findall(
        r"kaggle\.com/(datasets|models)/([a-z0-9_-]+/[a-z0-9_-]+(?:/[A-Za-z0-9._-]+){0,3})",
        text,
        flags=re.I,
    )
    out: list[str] = []
    kinds: dict[str, str] = {}
    for kind, ref in found:
        ref = ref.strip("/")
        owner = ref.split("/", 1)[0].lower()
        if owner in _NOT_A_DATASET_OWNER or "competitions/" in ref:
            continue
        if ref not in out:
            out.append(ref)
            kinds[ref] = kind.lower()
    # second pass: owner/name next to 'dataset' / 'weights' / 'labels' words
    for ref in re.findall(r"\b([a-z0-9_-]+/[a-z0-9_-]+)\b", text, flags=re.I):
        low = ref.lower()
        owner = low.split("/", 1)[0]
        if owner in _NOT_A_DATASET_OWNER:
            continue
        if not any(tok in low for tok in ("weight", "label", "dataset", "dinov2")):
            continue
        if ref not in out:
            out.append(ref)
            kinds[ref] = "dataset"
    return out[:8]


def extract_infer_hints(text: str) -> list[str]:
    hints: list[str] = []
    if re.search(r"iterdir|rglob|study.?dir|test_series|hidden test", text, re.I):
        hints.append("discover_test_ids_from_folders")
    if re.search(r"rank.?mean|rank.?avg|scipy\.stats\.rankdata", text, re.I):
        hints.append("rank_mean_ensemble")
    if re.search(r"llm.?label|report|weak.?label", text, re.I):
        hints.append("train_on_report_or_llm_labels")
    if re.search(r"group.?kfold|group.?split|site|scanner", text, re.I):
        hints.append("grouped_cv")
    return hints


def card_from_notebook(row: KernelRow, notebook_text: str, our_score: str) -> str:
    scores = re.findall(r"0\.\d{2,3}", notebook_text[:8000])
    claimed = scores[0] if scores else "unknown"
    if re.search(r"dinov2|vit-s", notebook_text, re.I):
        backbone = "DINOv2 / ViT mentioned in source"
    elif re.search(r"lightgbm|lgbm|metadata", notebook_text, re.I):
        backbone = "tabular / metadata"
    else:
        backbone = "see notebook"
    datasets = extract_datasets(notebook_text)
    hints = extract_infer_hints(notebook_text)
    next_step = (
        f"Attach datasets {datasets} and reuse their infer path."
        if datasets
        else "Pull this kernel and copy its inference ID discovery + rank-average."
    )
    if "rank_mean_ensemble" in hints:
        next_step += " Rank-average member scores; do not probability-mean."
    return "\n".join(
        [
            f"# {row.title or row.ref}",
            f"- ref: {row.ref} ({row.url})",
            f"- claimed_public: {claimed}",
            f"- backbone / input: {backbone}",
            "- labels: see notebook (prefer mounted LLM/report tables over gold-only)",
            "- CV: prefer grouped splits (report or site); avoid random folds",
            "- inference: discover hidden test IDs from study folders, not only sample test.csv",
            f"- infer_hints: {', '.join(hints) or 'none'}",
            f"- copyable next step: {next_step} Our score={our_score}.",
            "- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.",
            "",
            f"votes: {row.total_votes}",
            f"datasets_mentioned: {', '.join(datasets) or 'none'}",
            "",
        ]
    )


METHOD_CARDS_HEADING = "## Method cards"


def merge_digest(cards: list[Path], research_md: Path, our_score: str) -> None:
    lines = [
        METHOD_CARDS_HEADING,
        "",
        f"Method cards for PLAN/CODE. Our public best: {our_score}.",
        "",
        "**Must implement**",
        "",
        "1. Use public kernel methods (imaging or published weight packs), not constant scores.",
        "2. Find test IDs from hidden study folders.",
        "3. Rank-average members. AUC-style metrics only read order.",
        "4. Train labels from reports / mounted label tables. Gold subsets are too small for priors.",
        "",
        "**Sources**",
        "",
    ]
    for path in cards:
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else path.stem
        ref_m = re.search(r"- ref:\s*(\S+)", text)
        score_m = re.search(r"- claimed_public:\s*(.+)", text)
        step_m = re.search(r"- copyable next step:\s*(.+)", text)
        ref = ref_m.group(1) if ref_m else path.stem
        url = f"https://www.kaggle.com/code/{ref}" if "/" in ref and "http" not in ref else ref
        lines.append(f"- source: {url} — {title}; public {score_m.group(1) if score_m else '?'}")
        if step_m:
            lines.append(f"  next: {step_m.group(1)[:240]}")
        lines.append("")
    merge_section_into_research_md(research_md, METHOD_CARDS_HEADING, "\n".join(lines))


def write_methods_sidecar(cards: list[Path], workspace: Path) -> Path:
    datasets: list[str] = []
    models: list[str] = []
    hints: list[str] = []
    steps: list[str] = []
    for path in cards:
        text = path.read_text(encoding="utf-8")
        for ref in extract_datasets(text):
            if "dinov2" in ref.lower() or "/pytorch/" in ref.lower():
                if ref not in models:
                    models.append(ref)
            elif ref not in datasets:
                datasets.append(ref)
        for h in extract_infer_hints(text):
            if h not in hints:
                hints.append(h)
        step_m = re.search(r"- copyable next step:\s*(.+)", text)
        if step_m:
            steps.append(step_m.group(1).strip()[:240])
    payload = {
        "dataset_sources": datasets[:6],
        "model_sources": models[:3],
        "infer_hints": hints,
        "implement_steps": steps[:6],
        "n_cards": len(cards),
    }
    out = workspace / "pipeline" / "methods.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def load_methods(workspace: Path) -> dict[str, Any]:
    path = workspace / "pipeline" / "methods.json"
    if not path.is_file():
        return {"dataset_sources": [], "model_sources": [], "infer_hints": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"dataset_sources": [], "model_sources": [], "infer_hints": []}
    return data if isinstance(data, dict) else {}


_DEEP_DIGEST_HEADING = "## Deep research digest"


def _implement_or_copyable_steps(data: dict[str, Any]) -> list[str]:
    for key in ("implement_steps", "copyable_steps", "copyable_next_steps"):
        raw = data.get(key)
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        if isinstance(raw, list):
            steps = [str(x).strip() for x in raw if str(x).strip()]
            if steps:
                return steps
    return []


def cards_feasible(workspace: Path, research_md: Path) -> bool:
    """True when PLAN/CODE have a methods sidecar plus at least one copyable step.

    Claimed public > our best is intentionally not required — that bar can stall
    research forever on a new contest.
    """
    methods_path = workspace / "pipeline" / "methods.json"
    if not methods_path.is_file():
        return False
    if not _implement_or_copyable_steps(load_methods(workspace)):
        return False
    if not research_md.is_file():
        return False
    text = research_md.read_text(encoding="utf-8")
    return METHOD_CARDS_HEADING in text or _DEEP_DIGEST_HEADING in text


_PULL_LOCK = threading.Lock()


def _write_one_card(
    *,
    row: KernelRow,
    src: KaggleSource,
    dest: Path,
    our_score: str,
) -> Path:
    hit = SourceHit(url=row.url, title=row.title, kind="kaggle")
    with _PULL_LOCK:
        text = src.content(hit) or row.title or ""
    card = card_from_notebook(row, text, our_score)
    path = dest / f"source-{_slug(row.ref)}.md"
    path.write_text(card, encoding="utf-8")
    return path


def run_source_card_research(
    *,
    client: Any,
    competition: str,
    cache_dir: Path,
    root: Path,
    our_score: str = "unknown",
    max_kernels: int = 6,
    log: LogFn | None = None,
) -> list[Path]:
    """Pull top kernels and write one method card per source in parallel."""
    rows = client.kernels(competition, top=max_kernels + 2)
    src = KaggleSource(client, competition, cache_dir)
    dest = memory_dir(root) / "research-deep"
    dest.mkdir(parents=True, exist_ok=True)
    selected = [row for row in rows if row.ref and not skip_kernel(row)][:max_kernels]
    written: list[Path] = []
    workers = min(6, max(1, len(selected)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                _write_one_card, row=row, src=src, dest=dest, our_score=our_score
            )
            for row in selected
        ]
        for fut in as_completed(futs):
            try:
                path = fut.result()
            except Exception:  # noqa: BLE001
                continue
            written.append(path)
            if log:
                log(f"source card -> {path.name}")
    written.sort(key=lambda p: p.name)
    if written:
        merge_digest(written, memory_dir(root) / "research.md", our_score)
    return written
