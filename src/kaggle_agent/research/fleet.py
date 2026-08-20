"""Parallel per-source research fleet: one StageAgent loop per source kind.

Each subagent is a StageAgent with a restricted tool set and its own card
namespace (source-<kind>-*.md). The orchestrator converges the written cards
into methods.json / research.md after the fleet completes.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.agents.verification import AgentExecution
from kaggle_agent.llm.fallback import FallbackClient, ProviderSpec


@dataclass(frozen=True)
class SubagentSpec:
    name: str
    card_kind: str
    tools: tuple[str, ...]
    search_kinds: tuple[str, ...] = ()


AGENT_SPECS: dict[str, SubagentSpec] = {
    "notebooks": SubagentSpec(
        "notebooks", "notebook", ("list_kernels", "pull_kernel", "write_card")
    ),
    "papers": SubagentSpec(
        "papers", "paper", ("search", "fetch_url", "write_card"), ("arxiv",)
    ),
    "github": SubagentSpec(
        "github", "github", ("search", "fetch_url", "write_card"), ("github",)
    ),
    "web": SubagentSpec(
        "web", "web", ("search", "fetch_url", "write_card"), ("web",)
    ),
    "discussions": SubagentSpec(
        "discussions",
        "discussion",
        ("search", "fetch_url", "write_card"),
        ("discussion",),
    ),
    "datasets": SubagentSpec(
        "datasets", "dataset", ("search", "fetch_url", "write_card"), ("dataset",)
    ),
}

_ROLE_LINES = {
    "notebooks": (
        "You extract the winning recipe from public Kaggle notebooks: attachable "
        "datasets, model pins, CV scheme, inference ID discovery, claimed scores."
    ),
    "papers": (
        "You find arXiv papers that the competition or its top notebooks cite, "
        "and distill them into implementable method cards."
    ),
    "github": (
        "You find GitHub repositories that implement this competition's winning "
        "approach (or its backbones), and distill them."
    ),
    "web": (
        "You search the web for competition write-ups, data guides, and community "
        "solutions beyond Kaggle itself."
    ),
    "discussions": (
        "You mine Kaggle discussion threads for this competition: leak warnings, "
        "label sources, CV advice, hidden-test findings."
    ),
    "datasets": (
        "You find public Kaggle datasets and preprocessed versions for this "
        "competition via the Kaggle dataset search API."
    ),
}

_COMMON_SYSTEM = (
    "Call one tool per turn. Search and read, then write at least one card "
    "via write_card before done. write_card takes ref (source url/ref) and "
    "markdown (the card body). Prefer harvest_cards or write_card before done. "
    "Never repeat a normalized search query. Resolve every search before another "
    "search: fetch one returned source, write a card, or call reject_source with "
    "the source/ref and a concrete reason. "
    "Card body format:\n"
    "# title\n"
    "- ref: <url or ref>\n"
    "- claimed_public: <score or unknown>\n"
    "- backbone / input: <model or input type>\n"
    "- labels: <label source>\n"
    "- CV: prefer grouped splits; avoid random folds\n"
    "- inference: discover hidden test IDs from study folders, not only sample test.csv\n"
    "- copyable next step: <one implementable change> Our score={our_score}.\n"
    "- do not copy: H-flip; probability-mean ensembles; P100 if host forbids it.\n"
    "Never invent dataset slugs or URLs. Prefer primary sources over summaries."
)


def subagent_system(name: str, slug: str, our_score: str) -> str:
    """Per-agent charter: role, allowed tools, card format, write-before-done."""
    spec = AGENT_SPECS[name]
    kinds = ", ".join(spec.search_kinds) if spec.search_kinds else "none"
    notebook_rule = (
        "For notebooks: list public kernels, then pull one listed kernel before "
        "writing its card. A list alone is not source evidence.\n"
        if name == "notebooks"
        else ""
    )
    return (
        f"You are the {name} research agent for the Kaggle contest '{slug}'.\n"
        f"Your job: {_ROLE_LINES[name]}\n"
        f"Your tools: {', '.join(spec.tools)}. search kind is restricted to: {kinds}.\n"
        + notebook_rule
        + _COMMON_SYSTEM.format(our_score=our_score or "unknown")
    )


def make_write_card(
    dest: Path,
    kind: str,
    *,
    agent: str = "",
    run_id: str = "",
) -> Callable[[str, str], str]:
    """write_card closure that namespaces files as source-<kind>-<slug>.md."""

    def write_card(ref: str = "", markdown: str = "") -> str:
        body = (markdown or f"# {ref}\n- ref: {ref}\n").strip()
        if "copyable next step:" not in body or "do not copy:" not in body:
            return (
                "rejected: card body must contain both a 'copyable next step:' line "
                "and a 'do not copy:' line; write a shorter, complete card"
            )
        dest.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", (ref or kind).lower()).strip("-")[:60] or "src"
        path = dest / f"source-{kind}-{slug}.md"
        provenance = []
        if agent:
            provenance.append(f"- agent: {agent}")
        if run_id:
            provenance.append(f"- run_id: {run_id}")
        if provenance:
            lines = body.splitlines()
            insert_at = 1 if lines and lines[0].startswith("#") else 0
            lines[insert_at:insert_at] = provenance
            body = "\n".join(lines)
        path.write_text(body + "\n", encoding="utf-8")
        return str(path)

    return write_card


def fleet_tool_schemas(spec: SubagentSpec) -> dict[str, dict[str, Any]]:
    """Structured tool schemas so the model learns each tool's real parameters."""
    schemas: dict[str, dict[str, Any]] = {}
    if "search" in spec.tools:
        kind_prop: dict[str, Any] = {"type": "string", "description": "source kind"}
        if spec.search_kinds:
            kind_prop["enum"] = list(spec.search_kinds)
        schemas["search"] = {
            "description": "Search one source kind for this competition.",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "kind": kind_prop,
                "limit": {"type": "integer", "description": "max hits"},
            },
            "required": ["query"],
        }
    if "fetch_url" in spec.tools:
        schemas["fetch_url"] = {
            "description": "Fetch one http(s) URL returned by search.",
            "properties": {"url": {"type": "string", "description": "http(s) url"}},
            "required": ["url"],
        }
    if "list_kernels" in spec.tools:
        schemas["list_kernels"] = {
            "description": "List top public kernels for this competition.",
            "properties": {"query": {"type": "string", "description": "filter"}},
        }
    if "pull_kernel" in spec.tools:
        schemas["pull_kernel"] = {
            "description": "Pull one kernel's source by ref.",
            "properties": {"ref": {"type": "string", "description": "owner/kernel"}},
            "required": ["ref"],
        }
    if "write_card" in spec.tools:
        schemas["write_card"] = {
            "description": (
                "Write one method card. Call after you read at least one source."
            ),
            "properties": {
                "ref": {"type": "string", "description": "source url or ref"},
                "markdown": {"type": "string", "description": "full card body"},
            },
            "required": ["ref", "markdown"],
        }
    if "search" in spec.tools:
        schemas["reject_source"] = {
            "description": "Record why the current search/source cannot produce a method card.",
            "properties": {
                "ref": {"type": "string", "description": "search or source reference"},
                "reason": {"type": "string", "description": "specific rejection reason"},
            },
            "required": ["ref", "reason"],
        }
    return schemas


def make_fleet_tools(
    spec: SubagentSpec,
    *,
    search_fn: Callable[[str, str, int], str],
    fetch_fn: Callable[[str], str],
    write_fn: Callable[[str, str], str],
    kernel_list_fn: Callable[..., str] | None = None,
    kernel_pull_fn: Callable[..., str] | None = None,
    max_searches: int = 2,
) -> dict[str, Callable[..., str]]:
    """Tool dict for one subagent; every tool enforces the spec's restrictions."""

    search_calls = 0
    seen_queries: set[str] = set()
    pending_search: str | None = None
    listed_kernel_refs: list[str] = []
    pulled_kernel_ref = ""

    def _normalized_query(query: str) -> str:
        return " ".join(str(query).lower().split())

    def _is_low_yield(result: str) -> bool:
        return result.strip().lower().startswith(
            ("no source", "no hits", "none", "empty", "0 results")
        )

    def search(query: str = "", kind: str = "", limit: int = 5, **_a: Any) -> str:
        nonlocal search_calls, pending_search
        query = str(query)
        normalized = _normalized_query(query)
        if normalized in seen_queries:
            return "rejected: duplicate query; fetch a source or use a materially different query"
        if pending_search is not None:
            return (
                "rejected: resolve the previous search before another search; "
                "fetch_url, write_card, or reject_source first"
            )
        if search_calls >= max(1, int(max_searches)):
            return "search budget exhausted; fetch a returned source or write_card now"
        search_calls += 1
        seen_queries.add(normalized)
        kind = str(kind).strip()
        # A single-source agent defaults to (and is coerced to) its one kind,
        # so a kind-less or mismatched call never wastes a turn on a refusal.
        if len(spec.search_kinds) == 1:
            kind = spec.search_kinds[0]
        elif not kind and spec.search_kinds:
            kind = spec.search_kinds[0]
        if kind not in spec.search_kinds:
            allowed = ", ".join(spec.search_kinds) or "none"
            return f"refuse: kind={kind} not allowed for {spec.name} (allowed: {allowed})"
        result = search_fn(query, str(kind), int(limit))
        pending_search = f"search:{normalized or 'empty'}"
        if _is_low_yield(result):
            return (
                "rejected: low-yield search; call reject_source with a concrete reason "
                "before trying another query"
            )
        return result

    def fetch_url(url: str = "", **_a: Any) -> str:
        nonlocal pending_search
        result = fetch_fn(str(url))
        if result.strip() and not result.startswith(("rejected:", "tool error:", "no source", "none")):
            pending_search = None
        return result

    def write_card(ref: str = "", markdown: str = "", **_a: Any) -> str:
        nonlocal pending_search
        if spec.name == "notebooks" and not pulled_kernel_ref:
            return "rejected: pull a notebook source before writing a card"
        if spec.name == "notebooks" and ref and ref != pulled_kernel_ref:
            return "rejected: card ref must match the pulled notebook source"
        result = write_fn(str(ref), str(markdown))
        if result.strip() and not result.startswith(("rejected:", "tool error:")):
            pending_search = None
        return result

    def reject_source(ref: str = "", reason: str = "", **_a: Any) -> str:
        nonlocal pending_search
        if pending_search is None:
            return "rejected: no unresolved search to reject"
        if not str(reason).strip():
            return "rejected: rejection reason is required"
        pending_search = None
        return f"rejected: source {str(ref).strip() or 'unknown'} recorded: {str(reason).strip()}"

    tools: dict[str, Callable[..., str]] = {}
    if "search" in spec.tools:
        tools["search"] = search
    if "fetch_url" in spec.tools:
        tools["fetch_url"] = fetch_url
    if "write_card" in spec.tools:
        tools["write_card"] = write_card
    if "search" in spec.tools:
        tools["reject_source"] = reject_source
    if "list_kernels" in spec.tools and kernel_list_fn is not None:
        def list_kernels(query: str = "", limit: int = 6, **_a: Any) -> str:
            nonlocal listed_kernel_refs
            result = kernel_list_fn(query=query, limit=limit)
            listed_kernel_refs = [
                line.strip()
                for line in str(result).splitlines()
                if "/" in line and not line.lower().startswith(("none", "no source"))
            ]
            return str(result)

        tools["list_kernels"] = list_kernels
    if "pull_kernel" in spec.tools and kernel_pull_fn is not None:
        def pull_kernel(ref: str = "", **_a: Any) -> str:
            nonlocal pulled_kernel_ref
            selected = str(ref).strip() or (listed_kernel_refs[0] if listed_kernel_refs else "")
            if not selected:
                return "missing ref; list public kernels first"
            result = str(kernel_pull_fn(ref=selected))
            if result.strip() and not result.lower().startswith(
                ("missing", "none", "no source", "tool error:", "rejected:")
            ):
                pulled_kernel_ref = selected
            return result

        tools["pull_kernel"] = pull_kernel
    return tools


@dataclass
class FleetResult:
    agents: int = 0
    turns: int = 0
    stops: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    executions: list[AgentExecution] = field(default_factory=list)


def run_fleet(
    agents: list[tuple[str, StageAgent]],
    *,
    log: Callable[[str], None] | None = None,
) -> FleetResult:
    """Run every subagent concurrently; one failure never kills the fleet."""

    def _one(item: tuple[str, StageAgent]) -> tuple[str, Any]:
        name, agent = item
        try:
            return name, agent.run("")
        except Exception as exc:  # noqa: BLE001
            return name, exc

    stops: dict[str, str] = {}
    errors: dict[str, str] = {}
    executions: dict[str, AgentExecution] = {}
    turns = 0
    with ThreadPoolExecutor(max_workers=max(1, len(agents))) as pool:
        futs = {pool.submit(_one, item): item[0] for item in agents}
        for fut in as_completed(futs):
            name = futs[fut]
            _, out = fut.result()
            if isinstance(out, BaseException):
                errors[name] = f"{name}: {out}"
                stops[name] = "error"
            else:
                stops[name] = out.stop_reason
                turns += out.turns
                execution = out.execution()
                executions[name] = execution
            if log is not None:
                log(f"research fleet {name} stop={stops[name]}")
    return FleetResult(
        agents=len(agents),
        turns=turns,
        stops=[stops.get(name, "?") for name, _ in agents],
        errors=[errors[name] for name, _ in agents if name in errors],
        executions=[executions.get(name, AgentExecution(name, "error")) for name, _ in agents],
    )


def clone_client_for_agent(client: Any) -> Any:
    """Fresh LLM client per subagent so last_usage / last_tool_calls never race."""
    if client is None:
        return None
    if isinstance(client, FallbackClient):
        providers = [
            ProviderSpec(p.name, clone_client_for_agent(p.client), dict(p.models))
            for p in client.providers
        ]
        return FallbackClient(providers)
    try:
        return replace(client)
    except TypeError:
        return client
