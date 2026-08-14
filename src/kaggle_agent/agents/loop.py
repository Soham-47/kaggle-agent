"""Generic stage agent: LLM → tool → observe until done, time, or turn cap."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from kaggle_agent.config import ResearchAgentSettings
from kaggle_agent.llm.zen_client import ZenClient

StageAgentConfig = ResearchAgentSettings
LogFn = Callable[[str], None]
ToolFn = Callable[..., str]


@dataclass
class StageAgentResult:
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


class StageAgent:
    def __init__(
        self,
        zen: ZenClient | None,
        model: str,
        tools: dict[str, ToolFn],
        config: StageAgentConfig,
        *,
        system: str,
        log: LogFn | None = None,
        accept_done: Callable[[], bool] | None = None,
        no_zen_sequence: list[str] | None = None,
        name: str = "stage",
        reject_msg: str = "done rejected",
    ) -> None:
        self._zen = zen
        self._model = model
        self._tools = dict(tools)
        self._config = config
        self._system = system
        self._log = log
        self._accept_done = accept_done
        self._no_zen_sequence = list(no_zen_sequence or [])
        self._name = name
        self._reject_msg = reject_msg

    def _logmsg(self, msg: str) -> None:
        if self._log is not None:
            self._log(msg)

    def run(self, context: str) -> StageAgentResult:
        deadline = time.monotonic() + self._config.max_minutes * 60
        transcript: list[str] = [context]
        observations: list[str] = []
        turns = 0
        while True:
            if time.monotonic() >= deadline:
                self._logmsg(f"{self._name} agent stop: time")
                return StageAgentResult("time", turns, observations)
            if turns >= self._config.max_tool_turns:
                self._logmsg(f"{self._name} agent stop: turn_cap")
                return StageAgentResult("turn_cap", turns, observations)
            tool, args = self._next_action(transcript)
            if tool == "done":
                if self._accept_done is None or self._accept_done():
                    self._logmsg(f"{self._name} agent stop: done {args}")
                    return StageAgentResult("done", turns, observations)
                obs = self._reject_msg
                turns += 1
                observations.append(obs)
                transcript.append(obs)
                self._logmsg(f"{self._name} agent turn={turns} tool=done rejected")
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
            self._logmsg(f"{self._name} agent turn={turns} tool={tool}")

    def _next_action(self, transcript: list[str]) -> tuple[str, dict[str, Any]]:
        if self._zen is None or not hasattr(self._zen, "chat"):
            used = {t.split(" ", 1)[0][5:] for t in transcript if t.startswith("tool=")}
            for name in self._no_zen_sequence:
                if name in self._tools and name not in used:
                    return name, {}
            return "done", {"reason": "no_zen"}
        user = "Transcript:\n" + "\n---\n".join(transcript[-8:])
        raw = self._zen.chat(
            self._model,
            [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user + "\n\nReply with ONLY valid JSON."},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return parse_tool_call(raw)
