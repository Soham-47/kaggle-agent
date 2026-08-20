"""Generic stage agent: LLM → tool → observe until done, time, or turn cap."""

from __future__ import annotations

import json
import inspect
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from kaggle_agent.config import ResearchAgentSettings
from kaggle_agent.llm.zen_client import ZenClient
from kaggle_agent.agents.verification import AgentExecution

StageAgentConfig = ResearchAgentSettings
LogFn = Callable[[str], None]
ToolFn = Callable[..., str]

WRITE_TOOLS = frozenset(
    {
        "harvest_cards",
        "write_card",
        "write_plan",
        "write_methods",
        "write_custom_infer",
        "write_kernel_recipe",
        "write_brief",
    }
)


@dataclass
class StageAgentResult:
    stop_reason: str
    turns: int
    loop_iterations: int = 0
    llm_calls: int = 0
    control_actions: int = 0
    observations: list[str] = field(default_factory=list)
    agent: str = ""
    tool_calls: list[str] = field(default_factory=list)
    source_reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    rejected_writes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def execution(self) -> AgentExecution:
        return AgentExecution(
            agent=self.agent,
            stop_reason=self.stop_reason,
            turns=self.turns,
            loop_iterations=self.loop_iterations,
            llm_calls=self.llm_calls,
            control_actions=self.control_actions,
            tool_calls=list(self.tool_calls),
            source_reads=list(self.source_reads),
            writes=list(self.writes),
            rejected_writes=list(self.rejected_writes),
            errors=list(self.errors),
        )


@dataclass
class StallDecision:
    """One outcome of stall evaluation: continue, stop, force a tool, or nudge."""

    action: Literal["continue", "stop_stalled", "force_tool", "force_done", "nudge"]
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    nudge_text: str = ""


@dataclass
class StallControl:
    """Stall detection, force exhaustion, and nudge logic extracted from the agent loop.

    Call ``evaluate(turns)`` each turn.  The caller interprets the returned
    ``StallDecision``: execute a forced tool, append a nudge, stop, or continue
    to a normal LLM turn.  After any tool write (forced or LLM-driven), call
    ``mark_write(turn)`` to reset the stall window.

    The module is in-process and tested at its single seam.  All 7 parameters that
    previously threaded through the while-true loop now live in one place.
    """

    stall_after: int | None
    stall_nudge: str
    stall_force: (
        tuple[str, dict[str, Any]]
        | Callable[[int], tuple[str, dict[str, Any]] | None]
        | None
    )
    reject_msg: str = "done rejected"

    # --- mutable evaluation state ---
    stall_forced: bool = False
    force_count: int = 0
    last_force_turn: int | None = None
    last_write_turn: int = 0

    def __post_init__(self) -> None:
        if self.last_force_turn is None:
            self.last_force_turn = -(self.stall_after or 2)

    # --- methods ---

    def _force_for(self, episode: int) -> tuple[str, dict[str, Any]] | None:
        if callable(self.stall_force):
            return self.stall_force(episode)
        return self.stall_force

    def evaluate(self, turns: int) -> StallDecision:
        if self.stall_after is None:
            return StallDecision("continue")
        if turns - self.last_write_turn < self.stall_after:
            return StallDecision("continue")
        if (
            self.stall_force is not None
            and not self.stall_forced
            and turns - self.last_force_turn >= self.stall_after + 2
        ):
            self.stall_forced = True
            self.force_count += 1
            self.last_force_turn = turns
            forced = self._force_for(self.force_count)
            if forced is None:
                return StallDecision("stop_stalled")
            name, fargs = forced
            if name == "done":
                return StallDecision("force_done", tool_name=name, tool_args=fargs)
            return StallDecision("force_tool", tool_name=name, tool_args=fargs)
        nudge = self.stall_nudge or (
            "Stall: you have read enough. Call a write tool or done now."
        )
        return StallDecision("nudge", nudge_text=nudge)

    def mark_write(self, turn: int) -> None:
        self.last_write_turn = turn
        self.stall_forced = False


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
        agent_id: str | None = None,
        reject_msg: str = "done rejected",
        tracer: Any | None = None,
        max_invalid: int = 2,
        fallback_tool: str | None = None,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        stall_after: int | None = None,
        stall_nudge: str = "",
        stall_force: tuple[str, dict[str, Any]]
        | Callable[[int], tuple[str, dict[str, Any]] | None]
        | None = None,
        force_after_stall: str | None = None,
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
        self._agent_id = agent_id or name
        self._reject_msg = reject_msg
        self._tracer = tracer
        self._max_invalid = max(1, int(max_invalid))
        self._fallback_tool = fallback_tool
        self._invalid_streak = 0
        self._tool_schemas = dict(tool_schemas or {})
        self._stall = StallControl(
            stall_after=stall_after,
            stall_nudge=stall_nudge,
            stall_force=None if force_after_stall and stall_force else stall_force,
            reject_msg=reject_msg,
        )
        self._nudges: list[str] = []
        self._force_after_stall = force_after_stall
        self._forced_tool_choice: str | None = None
        self._active_forced_tool: str | None = None
        self._stall_fallback_force = stall_force if force_after_stall else None
        self._tool_calls: list[str] = []
        self._source_reads: list[str] = []
        self._writes: list[str] = []
        self._rejected_writes: list[str] = []
        self._errors: list[str] = []
        self._loop_iterations = 0
        self._llm_calls = 0
        self._control_actions = 0

    def _logmsg(self, msg: str) -> None:
        if self._log is not None:
            prefix = self._name if self._agent_id == self._name else f"{self._name}[{self._agent_id}]"
            self._log(f"{prefix}: {msg}")

    def run(self, context: str) -> StageAgentResult:
        deadline = time.monotonic() + self._config.max_minutes * 60
        transcript: list[str] = [context]
        observations: list[str] = []
        turns = 0
        stall = self._stall
        while True:
            self._loop_iterations += 1
            if time.monotonic() >= deadline:
                self._logmsg(f"{self._name} agent stop: time")
                self._trace("agent_stop", reason="time", turns=turns)
                return self._result("time", turns, observations)
            if turns >= self._config.max_tool_turns:
                self._logmsg(f"{self._name} agent stop: turn_cap")
                self._trace("agent_stop", reason="turn_cap", turns=turns)
                return self._result("turn_cap", turns, observations)
            decision = stall.evaluate(turns)
            if decision.action == "stop_stalled":
                self._logmsg(f"{self._name} agent stop: stalled")
                self._trace("agent_stop", reason="stalled", turns=turns)
                return self._result("stalled", turns, observations)
            if decision.action in ("force_tool", "force_done"):
                self._control_actions += 1
                name, args = decision.tool_name, dict(decision.tool_args)
                if decision.action == "force_done":
                    if self._accept_done is None or self._accept_done():
                        self._logmsg(f"{self._name} agent stop: done (forced) {args}")
                        self._trace("agent_stop", reason="done")
                        return self._result("done", turns, observations)
                    obs = self._stall.reject_msg
                else:
                    fn = self._tools.get(name)
                    if fn is None:
                        obs = f"unknown tool {name}"
                    else:
                        obs = self._invoke_tool(fn, args)
                turns += 1
                observations.append(obs[:4000])
                transcript.append(f"tool={name} args={args} result={obs[:2000]}")
                self._record_tool(name, obs)
                self._trace(
                    "tool",
                    tool=name,
                    turn=turns,
                    args_keys=sorted(args.keys()),
                    forced=True,
                )
                self._logmsg(f"{self._name} agent forced tool={name} turns={turns}")
                if name in WRITE_TOOLS:
                    stall.mark_write(turns)
                    self._nudges = []
                stall.stall_forced = False
                continue
            if decision.action == "nudge":
                self._control_actions += 1
                nudge_text = decision.nudge_text
                self._forced_tool_choice = self._force_after_stall
                if nudge_text not in self._nudges:
                    self._nudges.append(nudge_text)
                    self._logmsg(f"{self._name} agent nudge: stall turns={turns}")
                    transcript.append(f"nudge: {nudge_text}")
                if self._force_after_stall:
                    # Give the model one forced write turn after the nudge.
                    stall.last_write_turn = turns
                turns += 1
                continue
            tool, args = self._next_action(transcript)
            self._trace("tool", tool=tool, turn=turns + 1, args_keys=sorted(args.keys()))
            if tool == "no_llm":
                self._logmsg(f"{self._name} agent stop: no_llm")
                self._trace("agent_stop", reason="no_llm", turns=turns)
                return self._result("no_llm", turns, observations)
            if tool == "done":
                if self._accept_done is None or self._accept_done():
                    self._logmsg(f"{self._name} agent stop: done {args}")
                    self._trace("agent_stop", reason="done")
                    return self._result("done", turns, observations)
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
                obs = self._invoke_tool(fn, args)
            turns += 1
            observations.append(obs[:4000])
            transcript.append(f"tool={tool} args={args} result={obs[:2000]}")
            self._record_tool(tool, obs)
            if self._active_forced_tool is not None:
                forced_tool = self._active_forced_tool
                self._active_forced_tool = None
                if tool != forced_tool or obs.startswith(("rejected:", "tool error:")):
                    stall.stall_force = self._stall_fallback_force
            self._logmsg(f"{self._name} agent turn={turns} tool={tool}")
            if obs.startswith(("rejected:", "tool error:")):
                self._logmsg(f"{self._name} agent turn={turns} result={obs[:300]}")
            if tool in WRITE_TOOLS:
                stall.mark_write(turns)
                self._nudges = []

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
        head = ("\n".join(self._nudges) + "\n\n" if self._nudges else "")
        return (
            head
            + "Context:\n"
            + pack
            + "\n\nTranscript:\n"
            + "\n---\n".join(tail)
        )

    def _schemas(self) -> list[dict[str, Any]]:
        names = list(self._tools) + (["done"] if "done" not in self._tools else [])
        out: list[dict[str, Any]] = []
        for name in names:
            spec = self._tool_schemas.get(name) or {}
            fn: dict[str, Any] = {
                "name": name,
                "description": spec.get("description") or name,
            }
            props = spec.get("properties") or {}
            if props:
                fn["parameters"] = {
                    "type": "object",
                    "properties": props,
                    "additionalProperties": False,
                }
                if spec.get("required"):
                    fn["parameters"]["required"] = list(spec["required"])
            else:
                fn["parameters"] = {"type": "object", "additionalProperties": True}
            out.append({"type": "function", "function": fn})
        return out

    def _reset_invalid_streak(self) -> None:
        self._invalid_streak = 0

    def _next_action(self, transcript: list[str]) -> tuple[str, dict[str, Any]]:
        used = self._used_tools(transcript)
        forced = self._forced_next(transcript)
        if forced is not None:
            self._reset_invalid_streak()
            return forced
        if self._invalid_streak >= self._max_invalid:
            self._reset_invalid_streak()
            if self._fallback_tool and self._fallback_tool in self._tools:
                return self._fallback_tool, {}
            return "no_llm", {}
        if self._zen is None or not hasattr(self._zen, "chat"):
            return "no_llm", {}
        forced_choice = self._forced_tool_choice
        self._forced_tool_choice = None
        if (
            forced_choice
            and forced_choice in self._tools
            and (forced_choice != "write_card" or self._source_reads)
        ):
            self._active_forced_tool = forced_choice
            choice: str | dict[str, Any] = {
                "type": "function",
                "function": {"name": forced_choice},
            }
        else:
            choice = "auto" if used & WRITE_TOOLS else "required"
        self._llm_calls += 1
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
            max_tokens=getattr(self._config, "max_tokens", 2048),
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
            self._reset_invalid_streak()
            return native[0]
        tool, args = parse_tool_call(raw)
        if tool != "invalid_json":
            self._reset_invalid_streak()
            return tool, args
        self._llm_calls += 1
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
            max_tokens=getattr(self._config, "max_tokens", 2048),
            tools=self._schemas(),
            tool_choice=choice,
        )
        native = list(getattr(self._zen, "last_tool_calls", None) or [])
        if native:
            self._reset_invalid_streak()
            return native[0]
        tool2, args2 = parse_tool_call(raw2)
        if tool2 != "invalid_json":
            self._reset_invalid_streak()
            return tool2, args2
        self._invalid_streak += 1
        return tool2, args2

    def _trace(self, kind: str, **fields: Any) -> None:
        if self._tracer is None:
            return
        self._tracer.emit(kind, stage=self._name, agent=self._agent_id, **fields)

    @staticmethod
    def _invoke_tool(fn: ToolFn, args: dict[str, Any]) -> str:
        """Validate a call contract before executing a tool exactly once."""
        try:
            signature = inspect.signature(fn)
        except (TypeError, ValueError):
            # Some extension/builtin callables have no inspectable signature.
            # Execute once; an exception is an execution error, never a reason
            # to retry with a different argument set.
            signature = None
        if signature is not None:
            try:
                signature.bind(**args)
            except TypeError as exc:
                return f"invalid tool arguments: {exc}"
        try:
            return str(fn(**args))
        except Exception as exc:  # noqa: BLE001
            return f"tool error: {exc}"

    def _record_tool(self, tool: str, result: str) -> None:
        if tool in {"done", "no_llm", "invalid_json"}:
            return
        self._tool_calls.append(tool)
        if result.startswith(("tool error:", "invalid tool arguments:")):
            self._errors.append(result)
        if tool not in WRITE_TOOLS and result.strip() and not result.startswith(
            (
                "tool error:",
                "invalid tool arguments:",
                "refuse:",
                "no source",
                "no kaggle",
                "missing",
                "none",
                "search budget exhausted",
            )
        ):
            self._source_reads.append(tool)
        if result.startswith("search budget exhausted") and self._source_reads:
            self._forced_tool_choice = "write_card"
        if tool in WRITE_TOOLS and tool != "harvest_cards":
            if result.startswith(("rejected:", "tool error:", "invalid tool arguments:")):
                self._rejected_writes.append(result[:400])
            elif result.strip():
                self._writes.append(result.strip()[:4000])

    def _result(
        self, stop_reason: str, turns: int, observations: list[str]
    ) -> StageAgentResult:
        return StageAgentResult(
            stop_reason,
            turns,
            self._loop_iterations,
            self._llm_calls,
            self._control_actions,
            observations,
            agent=self._agent_id,
            tool_calls=list(self._tool_calls),
            source_reads=list(self._source_reads),
            writes=list(self._writes),
            rejected_writes=list(self._rejected_writes),
            errors=list(self._errors),
        )
