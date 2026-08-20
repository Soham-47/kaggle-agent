"""Shared agent execution and artifact verification contracts."""

from __future__ import annotations

import json
from pathlib import Path

from kaggle_agent.agents.loop import StageAgent, StageAgentConfig
from kaggle_agent.agents.verification import (
    AgentExecution,
    SourceEvidence,
    verify_code_stage,
    verify_plan_stage,
    verify_research_fleet,
)
from kaggle_agent.ops.evals import evaluate_cycle
from kaggle_agent.research.fleet import make_write_card, run_fleet


class _ScriptedZen:
    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)

    def chat(self, model, messages, **kwargs):  # noqa: ANN001
        return json.dumps(self.replies.pop(0))


def _agent(name: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(**_):
        path.write_text(
            f"# card\n- agent: {name}\n- copyable next step: inspect\n"
            "- do not copy: unknown\n",
            encoding="utf-8",
        )
        return str(path)

    return StageAgent(
        _ScriptedZen(
            [
                {"tool": "search", "args": {}},
                {"tool": "write_card", "args": {}},
                {"tool": "done"},
            ]
        ),
        "model",
        {"search": lambda **_: "source result", "write_card": write},
        StageAgentConfig(max_minutes=1, max_tool_turns=4),
        system="system",
        name="research",
        agent_id=name,
        accept_done=lambda: True,
        source_tools={"search": "source_search"},
    )


def test_stage_result_records_agent_tools_and_writes(tmp_path: Path):
    card = tmp_path / "source-notebook-a.md"
    result = _agent("notebooks", card).run("context")

    assert result.agent == "notebooks"
    assert result.stop_reason == "done"
    assert result.tool_calls == ["search", "write_card"]
    assert result.writes == [str(card)]
    assert result.rejected_writes == []


def test_stall_nudge_forces_model_write_tool(tmp_path: Path):
    card = tmp_path / "card.md"

    class _ChoiceZen:
        def __init__(self) -> None:
            self.choices = []

        def chat(self, model, messages, **kwargs):  # noqa: ANN001
            self.choices.append(kwargs.get("tool_choice"))
            if isinstance(kwargs.get("tool_choice"), dict):
                return json.dumps(
                    {
                        "tool": "write_card",
                        "args": {
                            "ref": "owner/source",
                            "markdown": "# source\n- copyable next step: inspect\n- do not copy: unknown\n",
                        },
                    }
                )
            return json.dumps({"tool": "search", "args": {}})

    zen = _ChoiceZen()

    def write(**_: object) -> str:
        card.write_text("# source\n- agent: research\n", encoding="utf-8")
        return str(card)

    agent = StageAgent(
        zen,
        "model",
        {"search": lambda **_: "source result", "write_card": write},
        StageAgentConfig(max_minutes=1, max_tool_turns=6),
        system="system",
        name="research",
        stall_after=2,
        stall_force=None,
        force_after_stall="write_card",
        accept_done=lambda: card.is_file(),
    )

    result = agent.run("context")

    assert result.writes == [str(card)]
    assert any(isinstance(choice, dict) for choice in zen.choices)


def test_fleet_returns_one_execution_record_per_agent(tmp_path: Path):
    agents = [
        ("notebooks", _agent("notebooks", tmp_path / "n.md")),
        ("papers", _agent("papers", tmp_path / "p.md")),
    ]

    result = run_fleet(agents)

    assert [item.agent for item in result.executions] == ["notebooks", "papers"]
    assert all(item.writes for item in result.executions)
    assert verify_research_fleet(result.executions, ["notebooks", "papers"]).ok


def test_research_verification_rejects_fallback_only_cards(tmp_path: Path):
    card = tmp_path / "source-pilkwang.md"
    card.write_text("fallback", encoding="utf-8")
    executions = [AgentExecution(agent="notebooks", stop_reason="no_llm")]

    verdict = verify_research_fleet(executions, ["notebooks"])

    assert not verdict.ok
    assert "notebooks" in verdict.detail


def test_transient_tool_error_does_not_fail_owned_write(tmp_path: Path):
    card = tmp_path / "source-github-x.md"
    card.write_text(
        "# card\n- agent: github\n- ref: owner/source\n"
        "- copyable next step: inspect\n- do not copy: unknown\n",
        encoding="utf-8",
    )
    executions = [
        AgentExecution(
            agent="github",
            stop_reason="done",
            source_reads=["search"],
            source_evidence=[SourceEvidence("search", "source_search", uri="https://example.test")],
            writes=[str(card)],
            errors=["tool error: http fetch failed: HTTP Error 404: Not Found"],
        )
    ]

    verdict = verify_research_fleet(executions, ["github"])

    assert verdict.ok


def test_typed_source_evidence_requires_registered_source_content(tmp_path: Path):
    card = tmp_path / "source-github-x.md"
    card.write_text(
        "# card\n- agent: github\n- ref: owner/source\n"
        "- copyable next step: inspect\n- do not copy: unknown\n",
        encoding="utf-8",
    )
    no_source = AgentExecution(
        agent="github",
        writes=[str(card)],
        source_evidence=[SourceEvidence("search", "source_search", uri="https://example.test")],
    )
    assert verify_research_fleet([no_source], ["github"]).ok
    assert not verify_research_fleet(
        [
            AgentExecution(
                agent="github",
                writes=[str(card)],
                source_evidence=[SourceEvidence("search", "source_search")],
            )
        ],
        ["github"],
    ).ok
    assert not verify_research_fleet(
        [AgentExecution(agent="github", writes=[str(card)], source_evidence=[])],
        ["github"],
    ).ok


def test_write_only_activity_never_counts_as_research_evidence(tmp_path: Path):
    card = tmp_path / "source-github-x.md"
    card.write_text(
        "# card\n- agent: github\n- ref: owner/source\n"
        "- copyable next step: inspect\n- do not copy: unknown\n",
        encoding="utf-8",
    )
    verdict = verify_research_fleet(
        [AgentExecution(agent="github", writes=[str(card)])], ["github"]
    )
    assert not verdict.ok


def test_research_verification_rejects_card_without_source_evidence(tmp_path: Path):
    card = tmp_path / "source-github-x.md"
    card.write_text(
        "# card\n- agent: github\n- ref: owner/source\n"
        "- copyable next step: inspect\n- do not copy: unknown\n",
        encoding="utf-8",
    )

    verdict = verify_research_fleet(
        [AgentExecution(agent="github", writes=[str(card)])], ["github"]
    )

    assert not verdict.ok


def test_harvested_fallback_card_requires_typed_source_evidence(tmp_path: Path):
    card = tmp_path / "source-user-baseline.md"
    card.write_text(
        "# card\n- agent: fallback\n- ref: kaggle/baseline\n"
        "- copyable next step: inspect\n- do not copy: unknown\n",
        encoding="utf-8",
    )
    evidence = SourceEvidence(
        "harvest_cards", "source_harvest", content_hash="abc123"
    )
    assert verify_research_fleet(
        [
            AgentExecution(
                agent="notebooks",
                writes=[str(card)],
                source_evidence=[evidence],
            )
        ],
        ["notebooks"],
    ).ok
    assert not verify_research_fleet(
        [AgentExecution(agent="notebooks", writes=[str(card)])], ["notebooks"]
    ).ok


def test_plan_and_code_verification_require_stage_artifacts():
    assert verify_plan_stage(wrote=True, judge_ready=True).ok
    assert not verify_plan_stage(wrote=False, judge_ready=True).ok
    assert verify_code_stage(wrote_recipe=True, wrote_custom_infer=False, smoke_ok=True).ok
    assert not verify_code_stage(
        wrote_recipe=True,
        wrote_custom_infer=False,
        artifact_ok=False,
        smoke_ok=True,
    ).ok
    assert not verify_code_stage(wrote_recipe=False, wrote_custom_infer=False, smoke_ok=True).ok


def test_fleet_card_writer_records_agent_provenance(tmp_path: Path):
    write = make_write_card(tmp_path, "notebook", agent="notebooks", run_id="run-1")

    path = Path(
        write(
            "owner/kernel",
            "# Kernel\n- ref: owner/kernel\n"
            "- copyable next step: attach weights\n"
            "- do not copy: P100\n",
        )
    )

    text = path.read_text(encoding="utf-8")
    assert path.name.startswith("source-notebook-")
    assert "- agent: notebooks" in text
    assert "- run_id: run-1" in text


def test_cycle_eval_rejects_unverified_research_fleet(tmp_path: Path):
    workspace = tmp_path / "competition"
    (workspace / "pipeline").mkdir(parents=True)
    (workspace / "pipeline" / "methods.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "memory").mkdir()
    events = [
        {
            "type": "tool",
            "stage": "research",
            "tool": "fleet",
            "verified": False,
        }
    ]

    report = evaluate_cycle(tmp_path, events=events, workspace=workspace)

    check = next(item for item in report["checks"] if item["id"] == "research_agents_verified")
    assert check["ok"] is False


def test_cycle_eval_accepts_verified_research_fleet(tmp_path: Path):
    workspace = tmp_path / "competition"
    (workspace / "pipeline").mkdir(parents=True)
    (workspace / "pipeline" / "methods.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "memory").mkdir()
    events = [
        {
            "type": "tool",
            "stage": "research",
            "tool": "fleet",
            "verified": True,
        }
    ]

    report = evaluate_cycle(tmp_path, events=events, workspace=workspace)

    check = next(item for item in report["checks"] if item["id"] == "research_wrote_card")
    assert check["ok"] is True
