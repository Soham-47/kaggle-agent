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
from kaggle_agent.llm.zen_client import ZenClient
from kaggle_agent.paths import memory_dir
from kaggle_agent.research.browser import merge_section_into_research_md
from kaggle_agent.research.deep import KaggleSource, SourceHit, _json_completion

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
    "com",
    "www.kaggle.com",
    "input",
    "data",
    "mnt",
    "tmp",
    "home",
    "torch",
    "kaggle",
    "working",
    "output",
    "dataset",
    "model",
    "dinov2",
    "pytorch",
    "transformers",
}

_JUNK_REFS = frozenset(
    {
        "dataset/model",
        "data/raw",
        "torch/gpu",
        "torch/GPU",
    }
)

_CARD_SYSTEM = (
    "You write one implementable Kaggle method card from a single primary source. "
    "Use only facts in the source text. Never invent dataset slugs. "
    "A Kaggle model pin is owner/model-slug/framework/instance/version. "
    "Never write dataset/model as a path."
)

_CARD_JUDGE_SYSTEM = (
    "You are the method-cards judge for one Kaggle experiment cycle. "
    "Return JSON {\"ready\": bool, \"reason\": str}. "
    "ready=false when the steps are generic (\"improve the model\"), "
    "refs are fake dataset/model slugs, or inference IDs are missing. "
    "ready=true when the cards give a coding agent concrete copyable steps "
    "with real kernel/dataset refs."
)


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
        if not _valid_attach_ref(ref):
            continue
        if ref not in out:
            out.append(ref)
            kinds[ref] = kind.lower()
    # second pass: owner/name next to 'dataset' / 'weights' / 'labels' words
    for ref in re.findall(r"\b([a-z0-9_-]+/[a-z0-9_-]+)\b", text, flags=re.I):
        low = ref.lower()
        if not _valid_attach_ref(ref):
            continue
        if not any(tok in low for tok in ("weight", "label", "dataset", "dinov2")):
            continue
        if ref not in out:
            out.append(ref)
            kinds[ref] = "dataset"
    return out[:8]


def _valid_attach_ref(ref: str) -> bool:
    ref = (ref or "").strip("/")
    if not ref or ref.lower() in {x.lower() for x in _JUNK_REFS}:
        return False
    if "competitions/" in ref:
        return False
    parts = [p for p in ref.split("/") if p]
    if not parts:
        return False
    owner = parts[0].lower()
    if owner in _NOT_A_DATASET_OWNER:
        return False
    if len(parts) == 2 and parts[1].lower() in {"dinov2", "pytorch", "transformers", "model"}:
        return False
    return True


def valid_model_pin(ref: str) -> bool:
    """Kaggle model attach needs owner/slug/framework/instance[/version]."""
    parts = [p for p in (ref or "").strip("/").split("/") if p]
    return len(parts) >= 4 and _valid_attach_ref(ref)


def step_is_junk(step: str) -> bool:
    low = (step or "").lower()
    if "dataset/model" in low:
        return True
    if "source unavailable" in low:
        return True
    if re.search(r"attach datasets \['dataset/", low):
        return True
    return False


def _slice_words(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text.strip()
    return text[:limit].rsplit(" ", 1)[0].strip()


def dedupe_steps(steps: list[str]) -> list[str]:
    """Keep unique, usable implementation steps in source order."""
    out: list[str] = []
    seen: set[str] = set()
    for step in steps:
        raw = str(step).strip()
        if not raw or step_is_junk(raw) or re.search(r"our score\s*=", raw, re.I):
            continue
        key = re.sub(r"\W+", " ", raw.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(raw)
    return out


_STOP_TOKENS = frozenset(
    "a an the of for with and or in on to from at by is are was were be been it its "
    "this that these those use using used new our we you your as per via vs do does "
    "not no so but then then next then next".split()
)


def plan_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-_/]*", (text or "").lower())
    return {w for w in words if len(w) > 1 and w not in _STOP_TOKENS}


_plan_tokens = plan_tokens  # legacy alias


def steps_implemented(steps_text: str, methods: dict[str, Any]) -> bool:
    """True when the plan steps are already covered by methods.json steps."""
    plan_tokens = _plan_tokens(steps_text)
    if not plan_tokens:
        return False
    impl_steps = [s for s in (methods.get("implement_steps") or []) if not step_is_junk(s)]
    for step in impl_steps:
        impl_tokens = _plan_tokens(step)
        overlap = len(plan_tokens & impl_tokens)
        if overlap >= 0.5 * len(plan_tokens) or plan_tokens <= impl_tokens:
            return True
    return False


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
            f"- copyable next step: {next_step}",
            "- do not copy: H-flip on laterality labels; probability-mean ensembles; P100 if host forbids it.",
            "",
            f"votes: {row.total_votes}",
            f"datasets_mentioned: {', '.join(datasets) or 'none'}",
            "",
        ]
    )


def card_from_source_llm(
    *,
    zen: ZenClient,
    model: str,
    title: str,
    ref: str,
    url: str,
    source_text: str,
    our_score: str,
    kind: str,
) -> str:
    """One Zen job per source. Falls back to empty string if JSON is unusable."""
    user = (
        f"kind: {kind}\nref: {ref}\nurl: {url}\ntitle: {title}\n"
        f"our_public_best: {our_score}\n\n"
        "Return JSON with keys: claimed_public, backbone, labels, cv, inference, "
        "copyable_next_step, do_not_copy, dataset_sources (list), model_sources (list).\n"
        "copyable_next_step is one kernel change we can ship. It must state the "
        "exact model, input resolution, loss, epochs, fold scheme, and weight pin. "
        "model_sources must be full pins or [].\n\n"
        f"<source>\n{source_text[:18000]}\n</source>"
    )
    parsed = _json_completion(zen, model, _CARD_SYSTEM, user, max_tokens=2400)
    step = str(parsed.get("copyable_next_step") or "").strip()
    if not step or step_is_junk(step):
        return ""
    models = [str(x) for x in (parsed.get("model_sources") or []) if valid_model_pin(str(x))]
    datasets = [
        str(x)
        for x in (parsed.get("dataset_sources") or [])
        if _valid_attach_ref(str(x)) and not valid_model_pin(str(x))
    ]
    return "\n".join(
        [
            f"# {title}",
            f"- ref: {ref} ({url})",
            f"- claimed_public: {parsed.get('claimed_public') or 'unknown'}",
            f"- backbone / input: {parsed.get('backbone') or 'see source'}",
            f"- labels: {parsed.get('labels') or 'see source'}",
            f"- CV: {parsed.get('cv') or 'prefer grouped splits'}",
            f"- inference: {parsed.get('inference') or 'discover hidden test IDs from study folders'}",
            f"- copyable next step: {step}",
            f"- do not copy: {parsed.get('do_not_copy') or 'H-flip; probability-mean; P100 if forbidden.'}",
            "",
            f"kind: {kind}",
            f"datasets_mentioned: {', '.join(datasets) or 'none'}",
            f"models_mentioned: {', '.join(models) or 'none'}",
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
            lines.append(f"  next: {_slice_words(step_m.group(1), 240)}")
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
            if valid_model_pin(ref):
                if ref not in models:
                    models.append(ref)
            elif ref not in datasets:
                datasets.append(ref)
        pin_m = re.search(r"models_mentioned:\s*(.+)", text)
        if pin_m and pin_m.group(1).strip() not in {"none", ""}:
            for ref in [x.strip() for x in pin_m.group(1).split(",")]:
                if valid_model_pin(ref) and ref not in models:
                    models.append(ref)
        for h in extract_infer_hints(text):
            if h not in hints:
                hints.append(h)
        step_m = re.search(r"- copyable next step:\s*(.+)", text)
        if step_m:
            raw_step = _slice_words(step_m.group(1).strip(), 240)
            if not step_is_junk(raw_step):
                steps.append(raw_step)
    from kaggle_agent.heal.pins import sanitize_methods_payload

    payload = sanitize_methods_payload(
        {
            "dataset_sources": datasets[:6],
            "model_sources": models[:3],
            "infer_hints": hints,
            "implement_steps": dedupe_steps(steps)[:6],
            "n_cards": len(cards),
        }
    )
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
    """True when PLAN/CODE have a sidecar, a non-junk step, and a digest heading.

    Public score beating our best is not required. That bar can stall a new contest.
    """
    methods_path = workspace / "pipeline" / "methods.json"
    if not methods_path.is_file():
        return False
    data = load_methods(workspace)
    steps = [s for s in _implement_or_copyable_steps(data) if not step_is_junk(s)]
    if not steps:
        return False
    junk_ds = [x for x in (data.get("dataset_sources") or []) if not _valid_attach_ref(str(x))]
    junk_md = [x for x in (data.get("model_sources") or []) if str(x) and not valid_model_pin(str(x))]
    if junk_ds or junk_md:
        return False
    if not research_md.is_file():
        return False
    text = research_md.read_text(encoding="utf-8")
    return METHOD_CARDS_HEADING in text or _DEEP_DIGEST_HEADING in text


def judge_cards_ready(
    zen: ZenClient | None,
    model: str,
    cards: list[Path],
    our_score: str,
    *,
    state: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Cards judge: delegates to the shared ``judge_stage`` scaffold so streak
    tracking and fail-open logic live in one seam with the other judges."""
    from kaggle_agent.judge import judge_stage, new_judge_state  # noqa: PLC0415

    if not cards:
        return False, "no cards"
    texts = [p.read_text(encoding="utf-8") for p in cards]

    def deterministic() -> tuple[bool, str]:
        steps = []
        for text in texts:
            m = re.search(r"- copyable next step:\s*(.+)", text)
            if m and not step_is_junk(m.group(1)):
                steps.append(m.group(1))
        if not steps:
            return False, "no actionable step"
        return True, "deterministic"

    def llm() -> tuple[bool, str]:
        joined = "\n\n".join(t[:1200] for t in texts[:2])
        user = (
            f"our_public_best={our_score}\n"
            "Are these method cards good enough for a coding agent to change a kernel "
            "without inventing dataset slugs? Return JSON "
            '{"ready": bool, "reason": str}. ready=false if steps are generic, '
            "refs are fake (dataset/model), or inference IDs are missing.\n\n"
            f"{joined}"
        )
        try:
            parsed = _json_completion(zen, model, _CARD_JUDGE_SYSTEM, user, max_tokens=400)
        except Exception:  # noqa: BLE001
            return True, "judge-fail-open"
        ready = bool(parsed.get("ready"))
        return ready, str(parsed.get("reason") or ("ready" if ready else "not ready"))

    return judge_stage(
        "cards",
        state=state if state is not None else new_judge_state(),
        deterministic=deterministic,
        llm=llm if zen is not None else None,
        log=log,
    )


_PULL_LOCK = threading.Lock()

_DISCUSSION_RE = re.compile(
    r"https?://www\.kaggle\.com/competitions/[^/\s]+/discussion/\d+", re.I
)
_ARXIV_RE = re.compile(r"https?://arxiv\.org/(?:abs|pdf)/\d+\.\d+(?:v\d+)?", re.I)


def extra_source_urls(research_md: Path, limit: int = 4) -> list[tuple[str, str]]:
    """Discussion and paper URLs from the current research file."""
    if not research_md.is_file():
        return []
    text = research_md.read_text(encoding="utf-8")
    out: list[tuple[str, str]] = []
    for url in _DISCUSSION_RE.findall(text):
        item = ("discussion", url)
        if item not in out:
            out.append(item)
    for url in _ARXIV_RE.findall(text):
        item = ("paper", url)
        if item not in out:
            out.append(item)
    return out[:limit]


def reset_source_cards(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for path in dest.glob("source-*.md"):
        path.unlink(missing_ok=True)


def _write_one_card(
    *,
    row: KernelRow,
    src: KaggleSource,
    dest: Path,
    our_score: str,
    zen: ZenClient | None = None,
    model: str = "",
) -> Path:
    hit = SourceHit(url=row.url, title=row.title, kind="kaggle")
    with _PULL_LOCK:
        text = src.content(hit) or row.title or ""
    card = ""
    if zen is not None and model:
        try:
            card = card_from_source_llm(
                zen=zen,
                model=model,
                title=row.title or row.ref,
                ref=row.ref,
                url=row.url,
                source_text=text,
                our_score=our_score,
                kind="kernel",
            )
        except Exception:  # noqa: BLE001
            card = ""
    if not card:
        card = card_from_notebook(row, text, our_score)
    path = dest / f"source-{_slug(row.ref)}.md"
    path.write_text(card, encoding="utf-8")
    return path


def _write_url_card(
    *,
    kind: str,
    url: str,
    dest: Path,
    our_score: str,
    zen: ZenClient | None,
    model: str,
    fetch: Callable[[str], str] | None,
) -> Path | None:
    body = ""
    if fetch is not None:
        try:
            body = fetch(url) or ""
        except Exception:  # noqa: BLE001
            body = ""
    title = url.rsplit("/", 1)[-1]
    card = ""
    if zen is not None and model and body:
        try:
            card = card_from_source_llm(
                zen=zen,
                model=model,
                title=title,
                ref=url,
                url=url,
                source_text=body,
                our_score=our_score,
                kind=kind,
            )
        except Exception:  # noqa: BLE001
            card = ""
    if not card:
        card = (
            f"# {kind} {title}\n"
            f"- ref: {url}\n"
            f"- claimed_public: unknown\n"
            f"- backbone / input: see source\n"
            "- labels: see source\n"
            "- CV: prefer grouped splits (report or site); avoid random folds\n"
            "- inference: discover hidden test IDs from study folders, not only sample test.csv\n"
            "- copyable next step: source unavailable; no implementation step.\n"
            "- do not copy: H-flip; probability-mean ensembles; P100 if host forbids it.\n"
        )
    path = dest / f"source-{kind}-{_slug(title)}.md"
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
    zen: ZenClient | None = None,
    model: str = "",
    reset: bool = True,
    url_fetch: Callable[[str], str] | None = None,
) -> list[Path]:
    """One worker per kernel / discussion / paper. New files each pass."""
    dest = memory_dir(root) / "research-deep"
    dest.mkdir(parents=True, exist_ok=True)
    if reset:
        reset_source_cards(dest)
    rows = client.kernels(competition, top=max_kernels + 2)
    src = KaggleSource(client, competition, cache_dir)
    selected = [row for row in rows if row.ref and not skip_kernel(row)][:max_kernels]
    extras = extra_source_urls(memory_dir(root) / "research.md")
    written: list[Path] = []
    workers = min(8, max(1, len(selected) + len(extras)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                _write_one_card,
                row=row,
                src=src,
                dest=dest,
                our_score=our_score,
                zen=zen,
                model=model,
            )
            for row in selected
        ]
        futs += [
            pool.submit(
                _write_url_card,
                kind=kind,
                url=url,
                dest=dest,
                our_score=our_score,
                zen=zen,
                model=model,
                fetch=url_fetch,
            )
            for kind, url in extras
        ]
        for fut in as_completed(futs):
            try:
                path = fut.result()
            except Exception:  # noqa: BLE001
                continue
            if path is None:
                continue
            written.append(path)
            if log:
                log(f"source card -> {path.name}")
    written.sort(key=lambda p: p.name)
    if written:
        merge_digest(written, memory_dir(root) / "research.md", our_score)
    return written
