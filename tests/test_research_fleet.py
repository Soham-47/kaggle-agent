"""Parallel per-source research fleet: specs, tool restriction, runner."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.llm.fallback import FallbackClient, ProviderSpec
from kaggle_agent.llm.zen_client import ZenClient
from kaggle_agent.research.fleet import (
    AGENT_SPECS,
    clone_client_for_agent,
    make_fleet_tools,
    make_write_card,
    run_fleet,
    subagent_system,
)


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        self.calls += 1
        if not self.replies:
            return json.dumps({"tool": "done", "args": {"reason": "empty script"}})
        return json.dumps(self.replies.pop(0))


def test_roster_has_six_specialist_agents():
    assert set(AGENT_SPECS) == {
        "notebooks",
        "papers",
        "github",
        "web",
        "discussions",
        "datasets",
    }
    notebooks = AGENT_SPECS["notebooks"]
    assert notebooks.card_kind == "notebook"
    assert notebooks.tools == ("list_kernels", "pull_kernel", "write_card")
    assert AGENT_SPECS["papers"].search_kinds == ("arxiv",)
    assert AGENT_SPECS["github"].search_kinds == ("github",)
    assert AGENT_SPECS["web"].search_kinds == ("web",)
    assert AGENT_SPECS["discussions"].search_kinds == ("web",)
    assert AGENT_SPECS["datasets"].search_kinds == ("web",)
    for name, spec in AGENT_SPECS.items():
        assert "search" not in spec.tools or "write_card" in spec.tools


def test_system_prompt_scoped_to_agent():
    text = subagent_system("papers", "rsna-knee-abnormality-detection", "0.526")
    assert "rsna-knee-abnormality-detection" in text
    assert "write_card" in text
    assert "0.526" in text


def _tools(spec_name: str, search_calls: dict | None = None, **extra):  # noqa: ANN001
    spec = AGENT_SPECS[spec_name]
    calls = search_calls if search_calls is not None else {"n": 0}

    def search_fn(query: str, kind: str, limit: int) -> str:
        calls["n"] += 1
        return f"hit {kind}"

    return make_fleet_tools(
        spec,
        search_fn=search_fn,
        fetch_fn=lambda u: "body",
        write_fn=lambda ref, md: "/tmp/card.md",
        kernel_list_fn=lambda **_a: "owner/kernel-a",
        kernel_pull_fn=lambda **_a: "notebook text",
        **extra,
    )


def test_search_refuses_disallowed_kinds():
    calls = {"n": 0}
    tools = _tools("papers", search_calls=calls)
    out = tools["search"](query="knee", kind="github")
    assert "refuse" in out
    assert calls["n"] == 0
    out = tools["search"](query="knee", kind="arxiv", limit=3)
    assert calls["n"] == 1
    assert "hit arxiv" in out


def test_notebooks_agent_gets_kernel_tools_only():
    tools = _tools("notebooks")
    assert tools["list_kernels"]() == "owner/kernel-a"
    assert tools["pull_kernel"]() == "notebook text"
    assert "search" not in tools
    assert "fetch_url" not in tools


def test_write_card_namespaces_by_kind(tmp_path: Path):
    dest = tmp_path / "cards"
    web_card = make_write_card(dest, "web")
    web_card("https://example.com/a", "# a\n- ref: https://example.com/a\n")
    github_card = make_write_card(dest, "github")
    github_card("owner/repo", "# repo\n- ref: https://github.com/owner/repo\n")
    names = sorted(p.name for p in dest.glob("source-*.md"))
    assert names == ["source-github-owner-repo.md", "source-web-https-example-com-a.md"]


def _stage_agent(zen, tools, accept_done=None):  # noqa: ANN001
    return StageAgent(
        zen,
        "m",
        tools,
        StageAgentConfig(max_minutes=5, max_tool_turns=10),
        system="s",
        accept_done=accept_done,
        reject_msg="write first",
        name="research",
    )


def test_agent_writes_card_before_done():
    wrote = {"n": 0}

    def write_card(**_a: object) -> str:
        wrote["n"] += 1
        return "/tmp/card.md"

    zen = _ScriptedZen(
        [
            {"tool": "done", "args": {"reason": "early"}},
            {"tool": "write_card", "args": {"ref": "a", "markdown": "x"}},
            {"tool": "done", "args": {"reason": "ok"}},
        ]
    )
    out = _stage_agent(zen, {"write_card": write_card}, accept_done=lambda: wrote["n"] > 0).run("ctx")
    assert out.stop_reason == "done"
    assert wrote["n"] == 1


def test_stall_pressure_forces_write_or_rejected_done():
    wrote = {"n": 0}

    def write_card(**_a: object) -> str:
        wrote["n"] += 1
        return "/tmp/card.md"

    zen = _ScriptedZen(
        [{"tool": "search", "args": {"query": "q"}}] * 8
        + [{"tool": "done", "args": {"reason": "finally"}}]
    )
    logs: list[str] = []
    agent = StageAgent(
        zen,
        "m",
        {"search": lambda **_a: "hit", "write_card": write_card},
        StageAgentConfig(max_minutes=5, max_tool_turns=8),
        system="s",
        log=lambda msg: logs.append(msg),
        accept_done=lambda: wrote["n"] > 0,
        reject_msg="done rejected: write at least one card first",
        stall_after=2,
        stall_nudge="Stall: call write_card now with your best finding.",
        stall_force=("done", {}),
        name="research",
    )
    out = agent.run("ctx")
    assert out.stop_reason == "turn_cap"
    assert out.turns == 8
    assert any("done rejected" in o for o in out.observations)
    assert any("nudge: stall" in m for m in logs)


def test_run_fleet_collects_stops_and_is_fail_soft():
    class _BrokenAgent:
        def run(self, context):  # noqa: ANN001
            raise RuntimeError("boom")

    ok_agent = _stage_agent(None, {})
    res = run_fleet(
        [("ok", ok_agent), ("bad", _BrokenAgent())],
        log=lambda msg: None,
    )
    assert res.agents == 2
    assert res.stops == ["no_llm", "error"]
    assert len(res.errors) == 1
    assert "bad" in res.errors[0]


def test_clone_client_isolates_runtime_state():
    client = ZenClient(api_key="k", base_url="https://api.deepseek.com", timeout_s=180.0)
    clone = clone_client_for_agent(client)
    assert clone is not client
    assert clone.api_key == "k"
    assert clone.base_url == "https://api.deepseek.com"
    assert clone_client_for_agent(None) is None
    opaque = object()
    assert clone_client_for_agent(opaque) is opaque


def test_clone_client_rebuilds_fallback_clients():
    zen = ZenClient(api_key="k", base_url="https://api.deepseek.com", timeout_s=180.0)
    fb = FallbackClient([ProviderSpec("zen", zen, {"chat": "model-x"})])
    clone = clone_client_for_agent(fb)
    assert clone is not fb
    assert clone.providers[0].client is not zen
    assert clone.providers[0].name == "zen"