"""Benchmark the CODE stage in disposable local workspaces.

Normal cases use the configured DeepSeek client. Fault cases use a local
scripted client to prove malformed/stalled responses terminate safely. This
script never reaches Kaggle, Telegram, or an external mutation path.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from kaggle_agent.agents.code import (
    calls_custom_infer,
    classify_code_outcome,
    extract_recipe_string,
    make_code_agent,
    valid_python,
    writes_submission_csv,
)
from kaggle_agent.agents.loop import StageAgentConfig
from kaggle_agent.llm.zen_client import ZenClient


REAL_CASES = (
    ("simple_pipeline", "rank average predictions"),
    ("multi_step_recipe", "attach the public weights and rank average predictions"),
    ("existing_method", "apply grouped cross validation and rank mean members"),
    ("parser_edge", "discover hidden test IDs from folders and write submission"),
    ("custom_infer", "apply custom inference after the ranker and write submission"),
)


def _fixture(root: Path) -> Path:
    (root / "memory" / "research-deep").mkdir(parents=True)
    (root / "memory" / "state.md").write_text("# state\n- phase: CODE\n", encoding="utf-8")
    (root / "memory" / "research-deep" / "source-rank.md").write_text(
        "# Synthetic source card\n"
        "- ref: owner/rank-kernel (https://example.invalid/rank)\n"
        "- copyable next step: attach public weights, grouped cross validation, "
        "hidden test IDs from folders, rank average predictions, and apply custom "
        "inference after the ranker.\n",
        encoding="utf-8",
    )
    workspace = root / "competitions" / "synthetic"
    pipeline = workspace / "pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "methods.json").write_text(
        json.dumps({"implement_steps": ["baseline submission"]}), encoding="utf-8"
    )
    (pipeline / "kernel_recipe.py").write_text(
        "KERNEL_RECIPE_SOURCE = r'''\n"
        "# === CUSTOM_INFER START ===\n"
        "def CUSTOM_INFER(sub, ctx):\n"
        "    return sub\n"
        "# === CUSTOM_INFER END ===\n"
        "sub = CUSTOM_INFER(sub, ctx)\n"
        "sub.to_csv('submission.csv')\n"
        "'''\n",
        encoding="utf-8",
    )
    return workspace


class _ScriptedClient:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat(self, *_args: Any, **_kwargs: Any) -> str:
        return self.response


def _fault_case(root: Path, response: str) -> dict[str, Any]:
    workspace = _fixture(root)
    agent, state = make_code_agent(
        _ScriptedClient(response),
        "local-fault-model",
        root,
        workspace,
        StageAgentConfig(max_minutes=0.1, max_tool_turns=5, max_tokens=512),
        plan_text="steps: rank average predictions",
    )
    out = agent.run("synthetic fault case")
    outcome = classify_code_outcome(out)
    return {
        "code_outcome": outcome.value,
        "outcome": out.stop_reason,
        "writes": bool(state.get("wrote_recipe") or state.get("wrote_custom_infer")),
        "turns": out.turns,
        "bounded": out.turns <= 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/code-agent-benchmark.json"))
    args = parser.parse_args()
    client = ZenClient.from_env()
    if client is None:
        print("DeepSeek: UNAVAILABLE")
        return 2

    rows: list[dict[str, Any]] = []
    config = StageAgentConfig(max_minutes=3, max_tool_turns=8, max_tokens=2048)
    with tempfile.TemporaryDirectory(prefix="kaggle-agent-code-benchmark-") as directory:
        base = Path(directory)
        for name, plan in REAL_CASES:
            root = base / name
            root.mkdir()
            workspace = _fixture(root)
            feedback = ""
            attempts: list[dict[str, Any]] = []
            for attempt in range(1, 3):
                agent, state = make_code_agent(
                    client,
                    "deepseek-v4-flash",
                    root,
                    workspace,
                    config,
                    plan_text=f"steps: {plan}{feedback}",
                )
                out = agent.run("Synthetic local CODE benchmark; no external actions.")
                wrapper = (workspace / "pipeline" / "kernel_recipe.py").read_text(
                    encoding="utf-8"
                )
                recipe = extract_recipe_string(wrapper) or ""
                artifact_valid = bool(
                    state.get("wrote_recipe") or state.get("wrote_custom_infer")
                ) and valid_python(recipe) is None and calls_custom_infer(recipe) is None and writes_submission_csv(recipe) is None
                attempts.append(
                    {
                        "attempt": attempt,
                        "code_outcome": classify_code_outcome(
                            out, artifact_valid=artifact_valid
                        ).value,
                        "stop_reason": out.stop_reason,
                        "turns": out.turns,
                        "tool_calls": out.tool_calls,
                        "writes": bool(state.get("wrote_recipe") or state.get("wrote_custom_infer")),
                        "artifact_valid": artifact_valid,
                    }
                )
                if artifact_valid:
                    break
                feedback = (
                    "\nPrevious attempt failed artifact validation. Write the complete "
                    "recipe now, preserve the required markers, and call done only "
                    "after the artifact is valid."
                )
            rows.append(
                {
                    "case": name,
                    "provider": "DeepSeek",
                    "attempts": attempts,
                    "final_artifact_valid": attempts[-1]["artifact_valid"],
                    "bounded": out.turns <= config.max_tool_turns,
                }
            )
        for name, response in (
            ("malformed_response", "not a tool action"),
            ("premature_done", json.dumps({"tool": "done", "args": {}})),
            ("repeated_noop", json.dumps({"tool": "read_plan", "args": {}})),
        ):
            root = base / name
            root.mkdir()
            row = _fault_case(root, response)
            row.update({"case": name, "provider": "scripted fault probe"})
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(rows), "rows": rows, "output": str(args.output)}, sort_keys=True))
    passed = all(
        (
            bool(row.get("final_artifact_valid"))
            if row.get("provider") == "DeepSeek"
            else bool(row.get("bounded")) and not bool(row.get("writes"))
        )
        for row in rows
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
