"""CODE stage: write methods and/or CUSTOM_INFER hook."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.heal.pins import sanitize_methods_payload
from kaggle_agent.memory.ingest import build_context_pack, retrieve
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
    }
)

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
        "model_sources accepts ONLY 4-part Kaggle model pins owner/slug/framework/instance "
        "(e.g. byi8552/rsna-keras3-effnet-b0-pretrain-trainin/densenet121/pretrain). "
        "Kernel references like romanrozen/... are NOT valid model pins; drop them.",
        "properties": {
            "dataset_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Kaggle dataset pins owner/dataset-name, e.g. "
                "wguesdon/rsna-knee-dinov2-at-meniscus-resolution",
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
        "the CUSTOM_INFER markers. Plan step tokens must appear in the recipe.",
        "properties": {
            "source": {
                "type": "string",
                "description": "Full recipe source code (the KERNEL_RECIPE_SOURCE body, "
                "no triple quotes)",
            }
        },
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
    "write_custom_infer, write_kernel_recipe, done. "
    "write_methods args: dataset_sources, model_sources, implement_steps (lists). "
    "write_custom_infer args: source = the CUSTOM_INFER function body only (no triple quotes). "
    "write_kernel_recipe args: source = full recipe body (no triple quotes). "
    "The recipe must write submission.csv and call CUSTOM_INFER(sub, ctx). "
    "Hook runs after the ranker on the submission table `sub`. Never hook Path `out`. "
    "The plan step key tokens MUST appear in the recipe source for done to pass. "
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
                "owner/slug/framework/instance (e.g. byi8552/rsna-keras3-effnet-b0"
                "-pretrain-trainin/densenet121/pretrain)"
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
    if "out = CUSTOM_INFER" in new_recipe:
        raise ValueError("do not hook Path out")
    if "sub = CUSTOM_INFER(sub, ctx)" not in new_recipe:
        raise ValueError("need sub = CUSTOM_INFER(sub, ctx)")
    if "submission.csv" not in new_recipe or "CUSTOM_INFER" not in new_recipe:
        raise ValueError("extracted recipe missing submission.csv or CUSTOM_INFER")
    ast.parse(wrapper)
    ast.parse(new_recipe)
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
    ast.parse(new)
    if "submission.csv" not in new:
        raise ValueError("recipe must write submission.csv")
    if "def CUSTOM_INFER" in new and "sub = CUSTOM_INFER(sub, ctx)" not in new:
        new = new.rstrip() + "\nctx = {\"labels\": LABELS, \"id_col\": ID_COL, \"work\": str(WORK)}\nsub = CUSTOM_INFER(sub, ctx)\n"
    if "out = CUSTOM_INFER" in new:
        raise ValueError("do not hook Path out")
    if HOOK_START not in new or HOOK_END not in new:
        new = _wrap_markers(new)
    if "sub = CUSTOM_INFER(sub, ctx)" not in new:
        raise ValueError("recipe must call CUSTOM_INFER(sub, ctx)")
    if len(new) < len(old) * 0.3:
        raise ValueError(f"recipe too short ({len(new)} chars vs {len(old)} before)")
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
            text or "attach listed datasets; discover test dirs; rank-mean",
            encoding="utf-8",
        )
        return str(brief_path)

    def write_methods(
        dataset_sources: list[str] | str | None = None,
        model_sources: list[str] | str | None = None,
        implement_steps: list[str] | str | None = None,
        infer_hints: list[str] | str | None = None,
        **_: Any,
    ) -> str:
        ds, ms, steps, hints = (
            _as_list(dataset_sources),
            _as_list(model_sources),
            _as_list(implement_steps),
            _as_list(infer_hints),
        )
        ok, err = methods_payload_ok(ds, ms, steps)
        if not ok:
            return f"rejected: {err}"
        current = load_methods(workspace)
        payload = sanitize_methods_payload(
            {
                "dataset_sources": ds or current.get("dataset_sources") or [],
                "model_sources": ms or current.get("model_sources") or [],
                "implement_steps": steps or current.get("implement_steps") or [],
                "infer_hints": hints or current.get("infer_hints") or [],
            }
        )
        out = workspace / "pipeline" / "methods.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        state["wrote_methods"] = "1"
        return str(out)

    def write_custom_infer(source: str = "", **_: Any) -> str:
        path = workspace / "pipeline" / "kernel_recipe.py"
        if not path.is_file():
            return "rejected: no kernel_recipe.py"
        try:
            text = splice_custom_infer(path.read_text(encoding="utf-8"), source or "return sub")
        except (ValueError, SyntaxError) as exc:
            return f"rejected: {exc}"
        path.write_text(text, encoding="utf-8")
        state["wrote_custom_infer"] = "1"
        return "hook written"

    def write_kernel_recipe(source: str = "", **_: Any) -> str:
        path = workspace / "pipeline" / "kernel_recipe.py"
        if not path.is_file():
            return "rejected: no kernel_recipe.py"
        try:
            text = replace_kernel_recipe(path.read_text(encoding="utf-8"), source)
        except ValueError as exc:
            return f"rejected: {exc}"
        path.write_text(text, encoding="utf-8")
        state["wrote_recipe"] = "1"
        return "recipe written"

    tools = {
        "read_cards": read_cards,
        "read_plan": read_plan,
        "retrieve": retrieve_tool,
        "read_file": read_file,
        "write_brief": write_brief,
        "write_methods": write_methods,
        "write_custom_infer": write_custom_infer,
        "write_kernel_recipe": write_kernel_recipe,
    }
    return tools, brief_path, state


def _plan_steps(plan_text: str) -> str:
    for line in (plan_text or "").splitlines():
        if line.lower().startswith("steps:"):
            return line.split(":", 1)[1].strip()
    return (plan_text or "").strip()


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
            return True
        if state.get("wrote_methods") or state.get("wrote_custom_infer"):
            recipe = _recipe_text(workspace)
            if recipe and not _any_plan_word_in_recipe(plan_steps, recipe):
                return False
            try:
                methods = load_methods(workspace)
            except Exception:  # noqa: BLE001
                return False
            return steps_implemented(plan_steps, methods)
        try:
            methods = load_methods(workspace)
        except Exception:  # noqa: BLE001
            return False
        methods_ok = steps_implemented(plan_steps, methods)
        recipe = _recipe_text(workspace)
        if recipe and not _any_plan_word_in_recipe(plan_steps, recipe):
            return False
        return methods_ok

    try:
        current_methods = load_methods(workspace)
    except Exception:  # noqa: BLE001
        current_methods = {}

    def code_stall_force(episode: int) -> tuple[str, dict[str, Any]] | None:
        if episode > 1:
            return None
        if _any_plan_word_in_recipe(_plan_steps(plan_text), _recipe_text(workspace) or ""):
            return (
                "write_methods",
                {"implement_steps": current_methods.get("implement_steps") or []},
            )
        return "read_file", {"rel": "pipeline/kernel_recipe.py"}

    agent = StageAgent(
        zen,
        model,
        tools,
        config,
        system=CODE_SYSTEM,
        log=log,
        accept_done=done_ok,
        must_first=[],
        name="code",
        reject_msg=(
            "done rejected: write_kernel_recipe to unlock done, or "
            "write_methods + write_custom_infer with the plan's key terms "
            "already present in the recipe source"
        ),
        tracer=tracer,
        tool_schemas=CODE_TOOL_SCHEMAS,
        stall_after=5,
        stall_nudge=(
            "STALL NUDGE: you have read enough. If the plan steps are already in "
            "the kernel recipe source (check with read_file kernel_recipe.py), "
            "call done. Otherwise write the implementation NOW: "
            "write_kernel_recipe with a concrete recipe source that implements "
            "every plan step (the recipe must write submission.csv and keep "
            "CUSTOM_INFER markers). Then call done."
        ),
        stall_force=code_stall_force,
    )
    return agent, state
