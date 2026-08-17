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
    return (
        f"You are the {name} research agent for the Kaggle contest '{slug}'.\n"
        f"Your job: {_ROLE_LINES[name]}\n"
        f"Your tools: {', '.join(spec.tools)}. search kind is restricted to: {kinds}.\n"
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


def make_fleet_tools(
    spec: SubagentSpec,
    *,
    search_fn: Callable[[str, str, int], str],
    fetch_fn: Callable[[str], str],
    write_fn: Callable[[str, str], str],
    kernel_list_fn: Callable[..., str] | None = None,
    kernel_pull_fn: Callable[..., str] | None = None,
) -> dict[str, Callable[..., str]]:
    """Tool dict for one subagent; every tool enforces the spec's restrictions."""

    def search(query: str = "", kind: str = "web", limit: int = 5, **_a: Any) -> str:
        if kind not in spec.search_kinds:
            allowed = ", ".join(spec.search_kinds) or "none"
            return f"refuse: kind={kind} not allowed for {spec.name} (allowed: {allowed})"
        return search_fn(str(query), str(kind), int(limit))

    def fetch_url(url: str = "", **_a: Any) -> str:
        return fetch_fn(str(url))

    def write_card(ref: str = "", markdown: str = "", **_a: Any) -> str:
        return write_fn(str(ref), str(markdown))

    tools: dict[str, Callable[..., str]] = {}
    if "search" in spec.tools:
        tools["search"] = search
    if "fetch_url" in spec.tools:
        tools["fetch_url"] = fetch_url
    if "write_card" in spec.tools:
        tools["write_card"] = write_card
    if "list_kernels" in spec.tools and kernel_list_fn is not None:
        tools["list_kernels"] = kernel_list_fn
    if "pull_kernel" in spec.tools and kernel_pull_fn is not None:
        tools["pull_kernel"] = kernel_pull_fn
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
