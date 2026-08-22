"""Independent, explicit artifact handoffs between repair-agent roles."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class IndependentSession:
    role: str
    session_id: str
    parent_history: tuple[str, ...] = ()

    @classmethod
    def fresh(cls, role: str) -> "IndependentSession":
        return cls(role=role, session_id=f"{role}-{uuid.uuid4().hex}")


class AgentProtocolError(ValueError):
    pass


def run_structured_session(role: str, call: Callable[[str, str], str], system: str, artifact: dict[str, Any]) -> dict[str, Any]:
    """Call one fresh session and accept only a JSON object artifact."""
    session = IndependentSession.fresh(role)
    prompt = json.dumps({"session_id": session.session_id, "artifact": artifact}, sort_keys=True)
    raw = call(system, prompt)
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentProtocolError(f"{role} returned malformed JSON") from exc
    if not isinstance(value, dict):
        raise AgentProtocolError(f"{role} must return a JSON object")
    return value
