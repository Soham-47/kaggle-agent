"""CODE stage: write methods and/or CUSTOM_INFER hook."""

from __future__ import annotations

import ast
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig, StageAgentResult
from kaggle_agent.paths import memory_dir
from kaggle_agent.experiment_fingerprint import (
    canonical_hash,
    recipe_hash,
    recipe_logic_hash,
)
from kaggle_agent.heal.pins import sanitize_methods_payload
from kaggle_agent.memory.ingest import build_context_pack, retrieve
from kaggle_agent.pipeline.image_template import (
    Image2dDinoMilTemplate,
    image_2d_dino_mil_contract,
)
from kaggle_agent.research.source_cards import (
    extract_infer_hints,
    load_methods,
    step_is_junk,
    steps_implemented,
    valid_model_pin,
)

HOOK_START = "# === CUSTOM_INFER START ==="
HOOK_END = "# === CUSTOM_INFER END ==="
_RECIPE_RE = re.compile(
    r"(KERNEL_RECIPE_SOURCE\s*=\s*r?('''|\"\"\"))(.*?)\2", re.S
)
ALLOWED_READ = frozenset(
    {
        "pipeline/methods.json",
        "pipeline/kernel_recipe.py",
        "pipeline/code_brief.md",
        "pipeline/methods_applied.md",
        "pipeline/weights.json",
        "pipeline/ranker.py",
        "pipeline/schema.py",
        "pipeline/image_contract.json",
        "pipeline/artifact_manifest.json",
    }
)

DEFAULT_BRIEF = (
    "Implement the plan steps in the kernel recipe. State the exact model, "
    "input resolution, loss, epochs, and fold scheme. Attach listed datasets; "
    "discover test dirs; rank-mean members."
)


class CodeOutcome(str, Enum):
    """Supervisor-visible, bounded outcomes for the CODE stage."""

    CODE_READY = "CODE_READY"
    NO_IMPLEMENTABLE_PLAN = "NO_IMPLEMENTABLE_PLAN"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    TOOL_PROTOCOL_FAILURE = "TOOL_PROTOCOL_FAILURE"
    TURN_BUDGET_EXHAUSTED = "TURN_BUDGET_EXHAUSTED"


def classify_code_outcome(
    result: StageAgentResult,
    *,
    artifact_valid: bool = False,
    provider_error: bool = False,
) -> CodeOutcome:
    """Map an agent stop to a bounded CODE result.

    The model's ``done`` action is deliberately absent from this decision.  A
    CODE stage is ready only after the caller has independently validated the
    required artifact.
    """
    if artifact_valid:
        return CodeOutcome.CODE_READY
    if provider_error:
        return CodeOutcome.PROVIDER_FAILURE
    evidence = "\n".join(
        [*result.errors, *result.rejected_writes, *result.observations]
    ).lower()
    if any(
        marker in evidence
        for marker in (
            "invalid_json",
            "unknown tool",
            "invalid tool arguments",
            "tool protocol",
        )
    ) or result.stop_reason == "no_llm":
        return CodeOutcome.TOOL_PROTOCOL_FAILURE
    if result.stop_reason in {"turn_cap", "time"}:
        return CodeOutcome.TURN_BUDGET_EXHAUSTED
    return CodeOutcome.NO_IMPLEMENTABLE_PLAN

CODE_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "read_cards": {
        "description": "Read the latest method cards (research-deep digest) for copyable steps."
    },
    "read_plan": {"description": "Read the current cycle plan text."},
    "retrieve": {
        "description": "Semantic search over memory, method cards, and experiments.",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "scope": {
                "type": "string",
                "description": "Search scope: cards, experiments, or another memory key",
            },
        },
    },
    "read_file": {
        "description": "Read a pipeline file. Paths allowed: methods.json, kernel_recipe.py, "
        "code_brief.md, methods_applied.md, weights.json, ranker.py, schema.py.",
        "properties": {
            "rel": {
                "type": "string",
                "description": "File name or pipeline/relative path",
            }
        },
    },
    "write_brief": {
        "description": "Write the code brief the kernel builder follows. Default: "
        "state the model, input resolution, loss, epochs, and fold scheme; "
        "attach listed datasets; discover test dirs; rank-mean.",
        "properties": {
            "text": {
                "type": "string",
                "description": "Brief text for the kernel builder",
            }
        },
    },
    "write_methods": {
        "description": "Write pipeline/methods.json for the next kernel build. "
        "Every proposed method must cite one or more existing source-card IDs or refs. "
        "model_sources accepts ONLY 4-part Kaggle model pins owner/slug/framework/instance "
        "(for example owner/model/framework/instance). "
        "Kernel references like romanrozen/... are NOT valid model pins; drop them.",
        "properties": {
            "dataset_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Kaggle dataset pins in owner/dataset-name form",
            },
            "model_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "4-part Kaggle model pins only: owner/slug/framework/instance",
            },
            "implement_steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete copyable implement steps, one string per step",
            },
            "infer_hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional inference hints for the kernel builder",
            },
            "source_card_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Existing source-card filenames (without .md) or card refs that ground these changes",
            },
        },
    },
    "write_custom_infer": {
        "description": "Splice a CUSTOM_INFER hook into the kernel recipe. The hook runs "
        "after the ranker on the submission table `sub`. Never hook Path `out`.",
        "properties": {
            "source": {
                "type": "string",
                "description": "Function body only (no def line, no triple quotes)",
            }
        },
    },
    "write_kernel_recipe": {
        "description": "Replace the full kernel recipe source (KERNEL_RECIPE_SOURCE body). "
        "Must write submission.csv, call CUSTOM_INFER(sub, ctx), and keep "
        "the CUSTOM_INFER markers. Cite existing source-card IDs/refs; the implementation "
        "must contain concrete evidence from those cards, not only plan tokens.",
        "properties": {
            "source": {
                "type": "string",
                "description": "Full recipe source code (the KERNEL_RECIPE_SOURCE body, "
                "no triple quotes)",
            },
            "source_card_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Existing source-card filenames (without .md) or card refs that ground this recipe change",
            },
        },
    },
    "write_image_contract": {
        "description": "Write the supported 2D DINO MIL image contract and render "
        "the owned image template. CODE may tune only dataset/model pins, cited cards, "
        "and bounded parameters such as image_size, folds, and epochs.",
        "properties": {
            "dataset_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete Kaggle dataset pins for labels/folds/weights.",
            },
            "model_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "4-part Kaggle model pins owner/slug/framework/instance.",
            },
            "source_card_refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Existing source-card IDs or refs grounding the contract.",
            },
            "parameters": {
                "type": "object",
                "description": "Bounded template parameters: image_size, folds, epochs.",
            },
        },
        "required": ["dataset_sources", "model_sources", "source_card_refs"],
    },
    "done": {
        "description": "Finish the code stage. Accepted when you wrote the recipe "
        "(write_kernel_recipe) with plan tokens in it, or wrote methods+custom_infer "
        "and the recipe already contains the plan tokens."
    },
}

CODE_SYSTEM = (
    "You are the coding agent for this Kaggle cycle. Call one tool per turn. "
    "Tools: read_cards, read_plan, retrieve, read_file, write_brief, write_methods, "
    "write_custom_infer, write_kernel_recipe, write_image_contract, done. "
    "write_methods args: dataset_sources, model_sources, implement_steps, source_card_refs (lists). "
    "write_custom_infer args: source = the CUSTOM_INFER function body only (no triple quotes). "
    "write_kernel_recipe args: source = full recipe body (no triple quotes), source_card_refs = cited cards. "
    "The recipe must write submission.csv and call CUSTOM_INFER(sub, ctx). "
    "Hook runs after the ranker on the submission table `sub`. Never hook Path `out`. "
    "Do not treat plan words as evidence: cite real source-card IDs/refs and implement their concrete terms. "
    "Call write_kernel_recipe or write_methods + write_custom_infer then done. "
    "If the recipe source already implements the plan steps, call done immediately. "
    "Rule: after at most two reads per tool, call write_kernel_recipe (or done when "
    "the plan steps are already in the recipe). Repeatedly reading without writing fails "
    "the stage. "
    'If you cannot call a tool, output {"tool": name, "args": {}}.'
)


def methods_payload_ok(
    dataset_sources: list[str] | None = None,
    model_sources: list[str] | None = None,
    implement_steps: list[str] | None = None,
) -> tuple[bool, str]:
    steps = [str(s).strip() for s in (implement_steps or []) if str(s).strip()]
    if not steps or any(step_is_junk(s) for s in steps):
        return False, "need a non-junk implement step"
    for pin in model_sources or []:
        if pin and not valid_model_pin(str(pin)):
            return False, (
                f"invalid model pin: {pin} — model_sources need 4-part pins "
                "owner/slug/framework/instance (e.g. owner/model/framework/instance)"
            )
    for ds in dataset_sources or []:
        if not ds or "/" not in str(ds) or str(ds).lower() in {"dataset/model"}:
            return False, f"invalid dataset: {ds}"
    return True, ""


def extract_recipe_string(wrapper: str) -> str | None:
    m = _RECIPE_RE.search(wrapper)
    return m.group(3) if m else None


def _as_list(v: list[str] | str | None) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.split(",") if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()]


def _indent_body(source: str) -> str:
    text = source.strip()
    if text.startswith("def CUSTOM_INFER"):
        lines = text.splitlines()[1:]
        text = "\n".join(ln[4:] if ln.startswith("    ") else ln for ln in lines).strip()
    if not text:
        text = "return sub"
    return "\n".join("    " + ln if ln.strip() else ln for ln in text.splitlines())


# ---------------------------------------------------------------------------
# Recipe validation pipeline: ordered predicates, each independently testable.
# A rule returns a failure reason or None on pass; the pipeline returns the
# first failure.  splice_custom_infer and replace_kernel_recipe share the
# syntactic/semantic rules, so a rule change fixes both callers at once.
# ---------------------------------------------------------------------------

RecipeRule = Callable[[str], str | None]


class ValidationPipeline:
    """Ordered recipe checks; ``run`` returns the first failure reason or None."""

    def __init__(self, rules: list[RecipeRule]) -> None:
        self._rules = list(rules)

    def run(self, recipe: str) -> str | None:
        for rule in self._rules:
            reason = rule(recipe)
            if reason is not None:
                return reason
        return None

    def __call__(self, recipe: str) -> str | None:
        return self.run(recipe)


def valid_python(recipe: str) -> str | None:
    try:
        ast.parse(recipe)
    except SyntaxError:
        return "recipe is not valid Python"
    return None


def writes_submission_csv(recipe: str) -> str | None:
    if "submission.csv" not in recipe:
        return "recipe must write submission.csv"
    return None


def no_out_hook(recipe: str) -> str | None:
    if "out = CUSTOM_INFER" in recipe:
        return "do not hook Path out"
    return None


def calls_custom_infer(recipe: str) -> str | None:
    if "sub = CUSTOM_INFER(sub, ctx)" not in recipe:
        return "recipe must call CUSTOM_INFER(sub, ctx)"
    return None


def valid_custom_infer_scope(recipe: str) -> str | None:
    """Reject a top-level hook call when ``sub`` only exists inside ``main``."""
    try:
        tree = ast.parse(recipe)
    except SyntaxError:
        return "recipe is not valid Python"
    global_sub = False
    has_main = any(
        isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body
    )
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign) else None
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "CUSTOM_INFER" and not global_sub and has_main:
                return "recipe calls CUSTOM_INFER with undefined top-level sub"
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "sub" for target in targets):
                global_sub = True
    return None


def min_length(old: str) -> RecipeRule:
    """Length guard against summarized/truncated recipe replacements."""

    def rule(recipe: str) -> str | None:
        if len(recipe) < len(old) * 0.3:
            return f"recipe too short ({len(recipe)} chars vs {len(old)} before)"
        return None

    return rule


_PRE_FIX_RULES: list[RecipeRule] = [valid_python, writes_submission_csv]
_POST_FIX_RULES: list[RecipeRule] = [
    no_out_hook,
    calls_custom_infer,
    valid_custom_infer_scope,
]


def splice_custom_infer(wrapper: str, source: str) -> str:
    m = _RECIPE_RE.search(wrapper)
    if not m:
        raise ValueError("KERNEL_RECIPE_SOURCE string missing")
    quote = m.group(2)
    if quote and quote in source:
        raise ValueError(f"hook must not contain {quote}")
    recipe = m.group(3)
    if HOOK_START not in recipe or HOOK_END not in recipe:
        raise ValueError("CUSTOM_INFER markers missing inside KERNEL_RECIPE_SOURCE")
    block = f"{HOOK_START}\ndef CUSTOM_INFER(sub, ctx):\n{_indent_body(source)}\n{HOOK_END}"
    before, rest = recipe.split(HOOK_START, 1)
    _, after = rest.split(HOOK_END, 1)
    new_recipe = before + block + after
    reason = ValidationPipeline(_POST_FIX_RULES + [writes_submission_csv, valid_python])(new_recipe)
    if reason is not None:
        raise ValueError(reason)
    ast.parse(wrapper)
    return wrapper[: m.start(3)] + new_recipe + wrapper[m.end(3) :]


def replace_kernel_recipe(wrapper: str, source: str) -> str:
    """Replace the KERNEL_RECIPE_SOURCE body with source (validated).

    Glue lines the agent routinely drops at the end of the recipe are
    re-inserted when the CUSTOM_INFER def is present: the ctx dict and the
    sub = CUSTOM_INFER(sub, ctx) call. Missing CUSTOM_INFER marker lines
    are re-inserted around the def the same way.
    """
    m = _RECIPE_RE.search(wrapper)
    if not m:
        raise ValueError("KERNEL_RECIPE_SOURCE string missing")
    quote = m.group(2)
    if quote and quote in source:
        raise ValueError(f"recipe must not contain {quote}")
    old = m.group(3)
    new = source.strip()
    reason = ValidationPipeline(_PRE_FIX_RULES)(new)
    if reason is not None:
        raise ValueError(reason)
    if "def CUSTOM_INFER" in new and "sub = CUSTOM_INFER(sub, ctx)" not in new:
        new = new.rstrip() + "\nctx = {\"labels\": LABELS, \"id_col\": ID_COL, \"work\": str(WORK)}\nsub = CUSTOM_INFER(sub, ctx)\n"
    if HOOK_START not in new or HOOK_END not in new:
        new = _wrap_markers(new)
    reason = ValidationPipeline(_POST_FIX_RULES + [min_length(old)])(new)
    if reason is not None:
        raise ValueError(reason)
    return wrapper[: m.start(3)] + new + wrapper[m.end(3) :]


def _wrap_markers(recipe: str) -> str:
    """Insert the CUSTOM_INFER marker lines around the hook def if missing."""
    tree = ast.parse(recipe)
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "CUSTOM_INFER"
        ),
        None,
    )
    if node is None:
        raise ValueError("recipe must keep CUSTOM_INFER markers")
    lines = recipe.splitlines(keepends=True)
    block = f"{HOOK_START}\n" + "".join(lines[node.lineno - 1 : node.end_lineno]) + f"{HOOK_END}\n"
    return "".join(lines[: node.lineno - 1]) + block + "".join(lines[node.end_lineno :])


def _recipe_text(workspace: Path) -> str:
    """Extract the KERNEL_RECIPE_SOURCE body from the workspace recipe file."""
    path = workspace / "pipeline" / "kernel_recipe.py"
    if not path.is_file():
        return ""
    ns: dict[str, object] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    src = ns.get("KERNEL_RECIPE_SOURCE")
    return str(src) if isinstance(src, str) else ""


def _any_plan_word_in_recipe(steps_text: str, recipe_text: str) -> bool:
    """True when at least one content word (3+ chars) from the plan appears in the recipe."""
    p = {w.lower() for w in re.findall(r"\w{3,}", (steps_text or ""))}
    if not p:
        return True
    r = {w.lower() for w in re.findall(r"\w+", (recipe_text or ""))}
    return bool(p & r)


_CARD_REF_RE = re.compile(r"^- ref:\s*(\S+)", re.M)
_CARD_STEP_RE = re.compile(r"^- copyable next step:\s*(.+)$", re.M)
_CARD_PROVENANCE_RE = re.compile(r"^\s*#\s*source_cards:\s*.+$", re.M)
_GROUNDING_STOP_WORDS = frozenset(
    "attach cards copyable do from inference kernel members next source step "
    "steps the this to use weights with write".split()
)


def _source_card_catalog(root: Path) -> dict[str, str]:
    """Map stable source-card IDs and refs to their card text.

    The file stem is the durable local ID. ``- ref: owner/kernel`` is an
    equivalent citation because it is what PLAN commonly carries forward.
    """
    catalog: dict[str, str] = {}
    cards_dir = memory_dir(root) / "research-deep"
    if not cards_dir.is_dir():
        return catalog
    for path in sorted(cards_dir.glob("source-*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        catalog[path.stem.lower()] = text
        ref = _CARD_REF_RE.search(text)
        if ref:
            catalog[ref.group(1).lower()] = text
    return catalog


def _resolve_source_cards(root: Path, refs: list[str] | str | None) -> tuple[list[str], list[str], str | None]:
    requested = _as_list(refs)
    if not requested:
        return [], [], "missing source_card_refs"
    catalog = _source_card_catalog(root)
    resolved: list[str] = []
    texts: list[str] = []
    for ref in requested:
        key = ref.lower().removesuffix(".md")
        text = catalog.get(key)
        if text is None:
            return [], [], f"unknown source card: {ref}"
        if key not in resolved:
            resolved.append(key)
            texts.append(text)
    return resolved, texts, None


def _card_evidence_tokens(cards: list[str]) -> set[str]:
    """Return implementation-specific vocabulary, excluding card boilerplate."""
    evidence: set[str] = set()
    for card in cards:
        fragments = _CARD_STEP_RE.findall(card)
        fragments.extend(re.findall(r"^(?:datasets|models)_mentioned:\s*(.+)$", card, re.M))
        for fragment in fragments:
            normalized = fragment.lower().replace("-", " ").replace("_", " ")
            evidence.update(re.findall(r"[a-z0-9][a-z0-9_-]*", normalized))
    return {token for token in evidence if len(token) >= 3 and token not in _GROUNDING_STOP_WORDS}


def _is_grounded_in_cards(change_text: str, cards: list[str]) -> bool:
    """Require two concrete method terms from cited cards, not plan-word overlap."""
    candidate = _CARD_PROVENANCE_RE.sub("", change_text)
    candidate = candidate.lower().replace("-", " ").replace("_", " ")
    candidate_tokens = set(re.findall(r"[a-z0-9][a-z0-9_-]*", candidate))
    return len(candidate_tokens & _card_evidence_tokens(cards)) >= 2


def _add_source_card_citation(source: str, refs: list[str]) -> str:
    """Persist tool-level card provenance with the generated recipe artifact."""
    return f"# source_cards: {', '.join(refs)}\n{source.lstrip()}"


def build_code_tools(
    root: Path,
    workspace: Path,
    *,
    plan_text: str = "",
) -> tuple[dict[str, Callable[..., str]], Path, dict[str, str]]:
    brief_path = workspace / "pipeline" / "code_brief.md"
    state = {"wrote_methods": "", "wrote_custom_infer": "", "wrote_recipe": ""}

    def read_cards(**_: Any) -> str:
        pack = build_context_pack(root, view="code", workspace=workspace, plan_text=plan_text)
        parts = [pack.get(k) for k in pack.sections if k.startswith("research-deep/")]
        return "\n\n".join(p for p in parts if p) or retrieve(root, "copyable", "cards")

    def read_plan(**_: Any) -> str:
        if plan_text.strip():
            return plan_text[:3000]
        return retrieve(root, "hypothesis", "experiments") or "no plan"

    def retrieve_tool(query: str = "", scope: str = "cards", **_: Any) -> str:
        return retrieve(root, query, scope=scope)

    def read_file(rel: str = "", **_: Any) -> str:
        name = (rel or "").replace("\\", "/").lstrip("/")
        if ".." in name or name not in ALLOWED_READ:
            return "refuse: path"
        path = workspace / name
        if not path.is_file():
            return "missing"
        cap = 40000 if name == "pipeline/kernel_recipe.py" else 8000
        return path.read_text(encoding="utf-8")[:cap]

    def write_brief(text: str = "", **_: Any) -> str:
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        brief_path.write_text(
            text or DEFAULT_BRIEF,
            encoding="utf-8",
        )
        return str(brief_path)

    def write_methods(
        dataset_sources: list[str] | str | None = None,
        model_sources: list[str] | str | None = None,
        implement_steps: list[str] | str | None = None,
        infer_hints: list[str] | str | None = None,
        source_card_refs: list[str] | str | None = None,
        **_: Any,
    ) -> str:
        ds, ms, steps, hints = (
            _as_list(dataset_sources),
            _as_list(model_sources),
            _as_list(implement_steps),
            _as_list(infer_hints),
        )
        from kaggle_agent.research.source_cards import dedupe_steps

        steps = dedupe_steps(steps)
        ok, err = methods_payload_ok(ds, ms, steps)
        if not ok:
            return f"rejected: {err}"
        card_refs, cards, err = _resolve_source_cards(root, source_card_refs)
        if err:
            return f"rejected: {err}"
        method_claim = "\n".join([*steps, *ds, *ms, *hints])
        if not _is_grounded_in_cards(method_claim, cards):
            return "rejected: methods are not grounded in cited source cards"
        current = load_methods(workspace)
        payload = sanitize_methods_payload(
            {
                "dataset_sources": ds or current.get("dataset_sources") or [],
                "model_sources": ms or current.get("model_sources") or [],
                "implement_steps": steps or current.get("implement_steps") or [],
                "infer_hints": hints or current.get("infer_hints") or [],
                "source_card_refs": card_refs,
            }
        )
        if canonical_hash(payload) == canonical_hash(current):
            return "rejected: methods are identical to the previous experiment"
        out = workspace / "pipeline" / "methods.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            return f"rejected: could not write methods.json: {exc}"
        state["wrote_methods"] = "1"
        return str(out)

    def write_custom_infer(source: str = "", **_: Any) -> str:
        path = workspace / "pipeline" / "kernel_recipe.py"
        if not path.is_file():
            return "rejected: no kernel_recipe.py"
        try:
            before = path.read_text(encoding="utf-8")
            text = splice_custom_infer(before, source or "return sub")
        except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
            return f"rejected: {exc}"
        old_recipe = _recipe_text_from_wrapper(before)
        new_recipe = _recipe_text_from_wrapper(text)
        if recipe_hash(old_recipe) == recipe_hash(new_recipe):
            return "rejected: recipe is identical to the previous experiment"
        if recipe_logic_hash(old_recipe) == recipe_logic_hash(new_recipe):
            return "rejected: recipe has no semantic logic change"
        path.write_text(text, encoding="utf-8")
        state["wrote_custom_infer"] = "1"
        return "hook written"

    def write_kernel_recipe(
        source: str = "",
        source_card_refs: list[str] | str | None = None,
        **_: Any,
    ) -> str:
        path = workspace / "pipeline" / "kernel_recipe.py"
        if not path.is_file():
            return "rejected: no kernel_recipe.py"
        card_refs, cards, err = _resolve_source_cards(root, source_card_refs)
        if err:
            return f"rejected: {err}"
        if not _is_grounded_in_cards(source, cards):
            return "rejected: recipe is not grounded in cited source cards"
        try:
            before = path.read_text(encoding="utf-8")
            text = replace_kernel_recipe(before, _add_source_card_citation(source, card_refs))
        except (OSError, UnicodeError, ValueError, SyntaxError) as exc:
            return f"rejected: {exc}"
        old_recipe = _recipe_text_from_wrapper(before)
        new_recipe = _recipe_text_from_wrapper(text)
        if recipe_hash(old_recipe) == recipe_hash(new_recipe):
            return "rejected: recipe is identical to the previous experiment"
        if recipe_logic_hash(old_recipe) == recipe_logic_hash(new_recipe):
            return "rejected: recipe has no semantic logic change"
        plan_steps = _plan_steps(plan_text)
        if plan_steps and not _any_plan_word_in_recipe(plan_steps, new_recipe):
            return "rejected: plan steps are missing from the recipe"
        path.write_text(text, encoding="utf-8")
        state["wrote_recipe"] = "1"
        return "recipe written"

    def write_image_contract(
        dataset_sources: list[str] | str | None = None,
        model_sources: list[str] | str | None = None,
        source_card_refs: list[str] | str | None = None,
        parameters: dict[str, Any] | None = None,
        **_: Any,
    ) -> str:
        ds = _as_list(dataset_sources)
        ms = _as_list(model_sources)
        ok, err = methods_payload_ok(
            dataset_sources=ds,
            model_sources=ms,
            implement_steps=["render the 2D DINO MIL image template"],
        )
        if not ok:
            return f"rejected: {err}"
        card_refs, cards, err = _resolve_source_cards(root, source_card_refs)
        if err:
            return f"rejected: {err}"
        contract_claim = "\n".join(
            [
                "2D pretrained DINO encoder",
                "series MIL attention pooling",
                "report derived labels",
                "grouped folds",
                "rank average folds",
                *ds,
                *ms,
            ]
        )
        if not _is_grounded_in_cards(contract_claim, cards):
            return "rejected: image contract is not grounded in cited source cards"
        labels = _labels_from_workspace(workspace)
        try:
            image_parameters = _bounded_image_parameters(parameters or {})
            contract = image_2d_dino_mil_contract(
                competition_id=workspace.name,
                labels=labels,
                dataset_sources=ds,
                model_sources=ms,
                source_card_refs=card_refs,
                parameters=image_parameters,
            )
            rendered = Image2dDinoMilTemplate().render(contract)
        except (TypeError, ValueError) as exc:
            return f"rejected: {exc}"
        pipe = workspace / "pipeline"
        pipe.mkdir(parents=True, exist_ok=True)
        wrapper = (
            '"""Rendered 2D DINO MIL image template."""\n\n'
            "KERNEL_RECIPE_SOURCE = r'''\n"
            + rendered.recipe_source.strip()
            + "\n'''\n"
        )
        recipe_path = pipe / "kernel_recipe.py"
        old_recipe = _recipe_text(workspace)
        existing_contract = pipe / "image_contract.json"
        if (
            old_recipe
            and recipe_logic_hash(old_recipe) == recipe_logic_hash(rendered.recipe_source)
            and existing_contract.is_file()
            and existing_contract.read_text(encoding="utf-8")
            == json.dumps(contract.to_dict(), indent=2) + "\n"
        ):
            return "rejected: recipe has no semantic logic change"
        recipe_path.write_text(wrapper, encoding="utf-8")
        (pipe / "image_contract.json").write_text(
            json.dumps(contract.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        (pipe / "artifact_manifest.json").write_text(
            json.dumps(rendered.manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        methods = sanitize_methods_payload(
            {
                "dataset_sources": ds,
                "model_sources": ms,
                "implement_steps": [
                    "Train the 2D DINO MIL template on mounted labels with grouped folds."
                ],
                "infer_hints": [
                    "discover_test_ids_from_folders",
                    "rank_mean_ensemble",
                ],
                "source_card_refs": card_refs,
            }
        )
        (pipe / "methods.json").write_text(
            json.dumps(methods, indent=2) + "\n",
            encoding="utf-8",
        )
        state["wrote_methods"] = "1"
        state["wrote_recipe"] = "1"
        return "image contract rendered"

    tools = {
        "read_cards": read_cards,
        "read_plan": read_plan,
        "retrieve": retrieve_tool,
        "read_file": read_file,
        "write_brief": write_brief,
        "write_methods": write_methods,
        "write_custom_infer": write_custom_infer,
        "write_kernel_recipe": write_kernel_recipe,
        "write_image_contract": write_image_contract,
    }
    return tools, brief_path, state


def _recipe_text_from_wrapper(wrapper: str) -> str:
    """Extract recipe text without executing a workspace file."""
    return extract_recipe_string(wrapper) or ""


def _plan_steps(plan_text: str) -> str:
    for line in (plan_text or "").splitlines():
        if line.lower().startswith("steps:"):
            return line.split(":", 1)[1].strip()
    return (plan_text or "").strip()


def _labels_from_workspace(workspace: Path) -> list[str]:
    path = workspace / "pipeline" / "schema.py"
    if path.is_file():
        ns: dict[str, Any] = {}
        try:
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        except Exception:  # noqa: BLE001
            ns = {}
        labels = ns.get("LABELS")
        if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
            return labels
    return ["target"]


def _bounded_image_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"image_size", "folds", "epochs", "slices_per_series", "batch_size"}
    bounded: dict[str, Any] = {}
    for key, value in parameters.items():
        if key not in allowed:
            continue
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if key == "image_size":
            bounded[key] = min(max(number, 224), 512)
        elif key == "folds":
            bounded[key] = min(max(number, 2), 10)
        elif key == "epochs":
            bounded[key] = min(max(number, 1), 20)
        elif key == "slices_per_series":
            bounded[key] = min(max(number, 1), 64)
        elif key == "batch_size":
            bounded[key] = min(max(number, 1), 64)
    return bounded or {"image_size": 336, "folds": 5, "epochs": 1}


_DATASET_PIN_RE = re.compile(r"['\"]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)['\"]")


def plan_to_methods_args(
    plan_text: str, current: dict[str, Any]
) -> dict[str, Any]:
    """Deterministic plan-to-methods fallback when the code model stalls."""
    steps = [s.strip() for s in _plan_steps(plan_text).split(";") if s.strip()]
    steps = [s for s in steps if not step_is_junk(s)]
    datasets = [
        d
        for d in _DATASET_PIN_RE.findall(plan_text or "")
        if "/" in d and d.lower() not in {"dataset/model"}
    ]
    return {
        "dataset_sources": datasets or current.get("dataset_sources") or [],
        "model_sources": current.get("model_sources") or [],
        "implement_steps": steps[:6] or current.get("implement_steps") or [],
        "infer_hints": extract_infer_hints(plan_text or "")
        or current.get("infer_hints")
        or [],
    }


def make_code_agent(
    zen: Any,
    model: str,
    root: Path,
    workspace: Path,
    config: StageAgentConfig,
    *,
    plan_text: str = "",
    log: Callable[[str], None] | None = None,
    tracer: Any | None = None,
) -> tuple[StageAgent, dict[str, str]]:
    tools, _brief, state = build_code_tools(root, workspace, plan_text=plan_text)

    def done_ok() -> bool:
        plan_steps = _plan_steps(plan_text)
        if state.get("wrote_recipe"):
            return _any_plan_word_in_recipe(plan_steps, _recipe_text(workspace))
        if state.get("wrote_custom_infer"):
            recipe = _recipe_text(workspace)
            if recipe and not _any_plan_word_in_recipe(plan_steps, recipe):
                return False
            try:
                methods = load_methods(workspace)
            except Exception:  # noqa: BLE001
                return False
            return steps_implemented(plan_steps, methods)
        return False

    try:
        current_methods = load_methods(workspace)
    except Exception:  # noqa: BLE001
        current_methods = {}

    agent = StageAgent(
        zen,
        model,
        tools,
        config,
        system=CODE_SYSTEM,
        log=log,
        accept_done=done_ok,
        name="code",
        reject_msg=(
            "done rejected: write_kernel_recipe or write_custom_infer must "
            "change the recipe before kernel training"
        ),
        tracer=tracer,
        tool_schemas=CODE_TOOL_SCHEMAS,
        stall_after=3,
        stall_nudge=(
            "STALL NUDGE: you have read enough. If the plan steps are already in "
            "the kernel recipe source (check with read_file kernel_recipe.py), "
            "call done. Otherwise write the implementation NOW: "
            "write_kernel_recipe with a concrete recipe source that implements "
            "every plan step (the recipe must write submission.csv and keep "
            "CUSTOM_INFER markers). Then call done."
        ),
        # Ask the model for one bounded write turn after the nudge.  The write
        # tool still applies all normal recipe/card/no-op validation; this only
        # prevents a read-only prefix from being mistaken for a completed CODE
        # attempt.  If the model still cannot produce a valid artifact, the
        # existing turn cap and supervisor failure path remain authoritative.
        force_after_stall="write_kernel_recipe",
    )
    return agent, state
