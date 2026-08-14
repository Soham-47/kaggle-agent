"""Research stage agent: LLM chooses tools until done, time, or turn cap."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from kaggle_agent.config import ResearchAgentSettings
from kaggle_agent.llm.zen_client import ZenClient

ResearchAgentConfig = ResearchAgentSettings

LogFn = Callable[[str], None]
ToolFn = Callable[..., str]

_SYSTEM = (
    "You research one Kaggle contest. Call one tool per turn. "
    "Reply with ONLY JSON: {\"tool\": name, \"args\": {}}. "
    "Tools: list_kernels, pull_kernel, fetch_url, search, write_card, "
    "harvest_cards, deep_research, judge_cards, done. "
    "Call done when cards are implementable (datasets/models, hidden test IDs, "
    "ensemble rule). Do not invent slugs."
)


@dataclass
class ResearchAgentResult:
    stop_reason: str
    turns: int
    observations: list[str] = field(default_factory=list)


def parse_tool_call(raw: str) -> tuple[str, dict[str, Any]]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json", {}
    if not isinstance(parsed, dict):
        return "invalid_json", {}
    tool = str(parsed.get("tool") or "done").strip()
    args = parsed.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return tool, args


class ResearchAgent:
    """while not done and under budget: LLM → tool → observe."""

    def __init__(
        self,
        zen: ZenClient | None,
        model: str,
        tools: dict[str, ToolFn],
        config: ResearchAgentConfig,
        *,
        log: LogFn | None = None,
        accept_done: Callable[[], bool] | None = None,
    ) -> None:
        self._zen = zen
        self._model = model
        self._tools = dict(tools)
        self._config = config
        self._log = log
        self._accept_done = accept_done

    def _logmsg(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def run(self, context: str) -> ResearchAgentResult:
        deadline = time.monotonic() + self._config.max_minutes * 60
        transcript: list[str] = [context]
        observations: list[str] = []
        turns = 0
        while True:
            if time.monotonic() >= deadline:
                self._logmsg("research agent stop: time")
                return ResearchAgentResult("time", turns, observations)
            if turns >= self._config.max_tool_turns:
                self._logmsg("research agent stop: turn_cap")
                return ResearchAgentResult("turn_cap", turns, observations)
            tool, args = self._next_action(transcript)
            if tool == "done":
                if self._accept_done is None or self._accept_done():
                    self._logmsg(f"research agent stop: done {args}")
                    return ResearchAgentResult("done", turns, observations)
                obs = "done rejected: cards not ready"
                turns += 1
                observations.append(obs)
                transcript.append(obs)
                self._logmsg(f"research agent turn={turns} tool=done rejected")
                continue
            fn = self._tools.get(tool)
            if fn is None:
                obs = f"unknown tool {tool}"
            else:
                try:
                    obs = str(fn(**args))
                except TypeError:
                    try:
                        obs = str(fn())
                    except Exception as exc:  # noqa: BLE001
                        obs = f"tool error: {exc}"
                except Exception as exc:  # noqa: BLE001
                    obs = f"tool error: {exc}"
            turns += 1
            observations.append(obs[:4000])
            transcript.append(f"tool={tool} args={args} result={obs[:2000]}")
            self._logmsg(f"research agent turn={turns} tool={tool}")

    def _next_action(self, transcript: list[str]) -> tuple[str, dict[str, Any]]:
        if self._zen is None or not hasattr(self._zen, "chat"):
            harvested = any(
                "tool=harvest_cards" in t or "tool=write_card" in t for t in transcript
            )
            if not harvested and "harvest_cards" in self._tools:
                return "harvest_cards", {}
            deepened = any("tool=deep_research" in t for t in transcript)
            if not deepened and "deep_research" in self._tools:
                return "deep_research", {}
            return "done", {"reason": "no_zen"}
        user = "Transcript:\n" + "\n---\n".join(transcript[-8:])
        raw = self._zen.chat(
            self._model,
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user + "\n\nReply with ONLY valid JSON."},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return parse_tool_call(raw)
