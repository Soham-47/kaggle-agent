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
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return "invalid_json", {}
        else:
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
        must_first: list[str] | None = None,
        must_first_args: dict[str, dict[str, Any]] | None = None,
        name: str = "stage",
        reject_msg: str = "done rejected",
        tracer: Any | None = None,
    ) -> None:
        self._zen = zen
        self._model = model
        self._tools = dict(tools)
        self._config = config
        self._system = system
        self._log = log
        self._accept_done = accept_done
        self._must_first = list(must_first or no_zen_sequence or [])
        self._must_first_args = dict(must_first_args or {})
        self._name = name
        self._reject_msg = reject_msg
        self._tracer = tracer

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
                self._trace("agent_stop", reason="time", turns=turns)
                return StageAgentResult("time", turns, observations)
            if turns >= self._config.max_tool_turns:
                self._logmsg(f"{self._name} agent stop: turn_cap")
                self._trace("agent_stop", reason="turn_cap", turns=turns)
                return StageAgentResult("turn_cap", turns, observations)
            tool, args = self._next_action(transcript)
            self._trace("tool", tool=tool, turn=turns + 1, args_keys=sorted(args.keys()))
            if tool == "done":
                if self._accept_done is None or self._accept_done():
                    self._logmsg(f"{self._name} agent stop: done {args}")
                    self._trace("agent_stop", reason="done")
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

    def _used_tools(self, transcript: list[str]) -> set[str]:
        return {t.split(" ", 1)[0][5:] for t in transcript if t.startswith("tool=")}

    def _forced_next(self, transcript: list[str]) -> tuple[str, dict[str, Any]] | None:
        used = self._used_tools(transcript)
        for name in self._must_first:
            if name in self._tools and name not in used:
                return name, dict(self._must_first_args.get(name) or {})
        return None

    def _user_blob(self, transcript: list[str]) -> str:
        pack = transcript[0] if transcript else ""
        tail = [t for t in transcript[1:] if t.startswith("tool=")][-12:]
        return "Context:\n" + pack + "\n\nTranscript:\n" + "\n---\n".join(tail)

    def _schemas(self) -> list[dict[str, Any]]:
        names = list(self._tools) + (["done"] if "done" not in self._tools else [])
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object", "additionalProperties": True},
                },
            }
            for name in names
        ]

    def _next_action(self, transcript: list[str]) -> tuple[str, dict[str, Any]]:
        forced = self._forced_next(transcript)
        if forced is not None:
            return forced
        if self._zen is None or not hasattr(self._zen, "chat"):
            return "done", {"reason": "no_zen"}
        writes = {"harvest_cards", "write_card", "write_plan", "write_methods", "write_custom_infer"}
        used = self._used_tools(transcript)
        choice = "auto" if used & writes else "required"
        raw = self._zen.chat(
            self._model,
            [
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": self._user_blob(transcript)
                    + '\n\nIf you cannot call a tool, output {"tool": name, "args": {}}.',
                },
            ],
            temperature=0.2,
            max_tokens=2048,
            tools=self._schemas(),
            tool_choice=choice,
        )
        usage = getattr(self._zen, "last_usage", None) or {}
        self._trace(
            "llm",
            model=self._model,
            tokens_in=int(usage.get("tokens_in") or 0),
            tokens_out=int(usage.get("tokens_out") or 0),
            chars=len(raw or ""),
        )
        native = list(getattr(self._zen, "last_tool_calls", None) or [])
        if native:
            return native[0]
        tool, args = parse_tool_call(raw)
        if tool != "invalid_json":
            return tool, args
        raw2 = self._zen.chat(
            self._model,
            [
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": self._user_blob(transcript)
                    + '\n\nPrevious reply was not a tool. Output {"tool": name, "args": {}}.',
                },
            ],
            temperature=0.0,
            max_tokens=2048,
            tools=self._schemas(),
            tool_choice=choice,
        )
        native = list(getattr(self._zen, "last_tool_calls", None) or [])
        if native:
            return native[0]
        return parse_tool_call(raw2)

    def _trace(self, kind: str, **fields: Any) -> None:
        if self._tracer is None:
            return
        self._tracer.emit(kind, stage=self._name, **fields)
