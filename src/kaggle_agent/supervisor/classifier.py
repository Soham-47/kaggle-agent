"""Deterministic failure classification before any source repair."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.train.kernel_runner import KernelPushRepair
from kaggle_agent.autonomy.outbox import ExternalActionOutbox


class FailureClass(str, Enum):
    TRANSIENT_EXTERNAL = "TRANSIENT_EXTERNAL"
    PENDING_EXTERNAL = "PENDING_EXTERNAL"
    KNOWN_RUNTIME_REPAIR = "KNOWN_RUNTIME_REPAIR"
    CODE_DEFECT = "CODE_DEFECT"
    TEST_FAILURE = "TEST_FAILURE"
    CONFIGURATION = "CONFIGURATION"
    DATA_CONTRACT = "DATA_CONTRACT"
    DEPENDENCY = "DEPENDENCY"
    ENVIRONMENT = "ENVIRONMENT"
    LLM_PROTOCOL = "LLM_PROTOCOL"
    LLM_PROVIDER = "LLM_PROVIDER"
    AGENT_REASONING = "AGENT_REASONING"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    POLICY_BLOCK = "POLICY_BLOCK"
    CORRUPTED_STATE = "CORRUPTED_STATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FailureClassification:
    failure_class: FailureClass
    confidence: float
    repairable: bool
    likely_files: tuple[str, ...] = ()
    reason: str = ""


def classify_failure(incident: Incident) -> FailureClassification:
    message = f"{incident.exception_message} {' '.join(incident.evidence)}".lower()
    if incident.external_job and any(word in message for word in ("unknown", "timeout", "uncertain", "not confirmed")):
        return FailureClassification(FailureClass.PENDING_EXTERNAL, 0.99, False, reason="external action is unresolved")
    if any(code in message for code in ("http 408", "http 429", "http 502", "http 503", "http 504", "dns", "connection reset", "network timeout")):
        return FailureClassification(FailureClass.TRANSIENT_EXTERNAL, 0.99, False, reason="bounded external retry condition")
    if KernelPushRepair.classify(incident.exception_message):
        return FailureClassification(FailureClass.KNOWN_RUNTIME_REPAIR, 0.99, False, reason="existing KernelPushRepair policy applies")
    if any(word in message for word in ("api key", "credentials", "credential", "missing deepseek", "telegram token")):
        return FailureClassification(FailureClass.AUTHENTICATION, 0.99, False, reason="required credential is unavailable")
    if any(word in message for word in ("not joined", "terms not accepted", "permission denied", "competition closed", "gpu not allowed")):
        return FailureClassification(FailureClass.AUTHORIZATION, 0.98, False, reason="external authority is required")
    if any(word in message for word in ("assert", "pytest", "test failed")):
        return FailureClassification(FailureClass.TEST_FAILURE, 0.93, True, reason="deterministic test failure")
    if any(word in message for word in ("nameerror", "attributeerror", "typeerror", "indexerror", "keyerror", "syntaxerror", "importerror")):
        return FailureClassification(FailureClass.CODE_DEFECT, 0.94, True, reason="deterministic project implementation failure")
    if any(word in message for word in ("out of memory", "oom", "disk full", "no space left")):
        return FailureClassification(FailureClass.RESOURCE_EXHAUSTION, 0.98, False, reason="resource exhaustion is not a source defect")
    return FailureClassification(FailureClass.UNKNOWN, 0.0, False, reason="deterministic rules did not establish a safe class")


def classify_after_reconciliation(incident: Incident, outbox: ExternalActionOutbox, reconcile) -> FailureClassification:
    """Reconcile an uncertain mutation before applying source-repair rules."""
    if incident.external_job:
        item = outbox.get(incident.external_job)
        if item is not None and item.status in {"prepared", "sent", "unknown"}:
            resolved = reconcile(item)
            if resolved.status in {"prepared", "sent", "unknown"}:
                return FailureClassification(FailureClass.PENDING_EXTERNAL, 1.0, False, reason="external action remains unresolved")
    return classify_failure(incident)
