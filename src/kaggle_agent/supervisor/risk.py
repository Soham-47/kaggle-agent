"""Deterministic, semantic risk evaluation for autonomous repairs.

This module is deliberately independent of the model sessions.  A model may
describe a likely file or root cause, but only durable incident/spec data,
the candidate diff, and hard-coded trust-boundary checks affect the decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from kaggle_agent.supervisor.classifier import FailureClass, FailureClassification
from kaggle_agent.supervisor.incidents import Incident
from kaggle_agent.supervisor.policy import PROTECTED_PATHS
from kaggle_agent.supervisor.spec import RepairSpec


class RepairRiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    PROHIBITED = "PROHIBITED"


_RISK_TIER_ORDER = {
    RepairRiskTier.LOW: 0,
    RepairRiskTier.MEDIUM: 1,
    RepairRiskTier.HIGH: 2,
    RepairRiskTier.PROHIBITED: 3,
}


class ExternalStateCertainty(str, Enum):
    NO_EXTERNAL_ACTION = "NO_EXTERNAL_ACTION"
    RECONCILED_EXACT = "RECONCILED_EXACT"
    RECONCILED_ABSENT = "RECONCILED_ABSENT"
    PENDING = "PENDING"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class ReproductionStrength(str, Enum):
    EXISTING_DETERMINISTIC_TEST = "EXISTING_DETERMINISTIC_TEST"
    NEW_REGRESSION_TEST = "NEW_REGRESSION_TEST"
    DETERMINISTIC_COMMAND_REPRO = "DETERMINISTIC_COMMAND_REPRO"
    STATIC_REPRO = "STATIC_REPRO"
    LOG_ONLY = "LOG_ONLY"
    NO_REPRO = "NO_REPRO"


@dataclass(frozen=True)
class RepairRiskDecision:
    tier: RepairRiskTier
    score: int
    reasons: tuple[str, ...]
    positive_factors: tuple[str, ...]
    negative_factors: tuple[str, ...]
    candidate_generation_allowed: bool
    automatic_promotion_allowed: bool
    max_source_files: int
    max_test_files: int
    max_changed_lines: int
    max_implementation_attempts: int
    required_reproduction_strength: ReproductionStrength
    require_focused_tests: bool
    require_adjacent_tests: bool
    require_full_tests: bool
    require_static_safety: bool
    require_spec_review: bool
    require_code_review: bool
    external_reconciliation_required: bool
    authority_required: bool
    external_state: ExternalStateCertainty = ExternalStateCertainty.NO_EXTERNAL_ACTION
    reproduction_strength: ReproductionStrength = ReproductionStrength.NO_REPRO
    subsystem: str = "unknown"
    changed_source_files: int = 0
    changed_test_files: int = 0
    changed_lines: int = 0
    risk_floor: RepairRiskTier | None = None

    @property
    def accepted_for_candidate(self) -> bool:
        return self.candidate_generation_allowed

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("tier", "risk_floor", "required_reproduction_strength", "reproduction_strength", "external_state"):
            if value[field] is None:
                continue
            value[field] = value[field].value
        for field in ("reasons", "positive_factors", "negative_factors"):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RepairRiskDecision":
        raw = dict(value)
        raw["tier"] = RepairRiskTier(raw["tier"])
        raw["risk_floor"] = RepairRiskTier(raw["risk_floor"]) if raw.get("risk_floor") else None
        raw["external_state"] = ExternalStateCertainty(raw.get("external_state", "NO_EXTERNAL_ACTION"))
        raw["required_reproduction_strength"] = ReproductionStrength(raw["required_reproduction_strength"])
        raw["reproduction_strength"] = ReproductionStrength(raw.get("reproduction_strength", "NO_REPRO"))
        for field in ("reasons", "positive_factors", "negative_factors"):
            raw[field] = tuple(raw.get(field) or ())
        return cls(**raw)


_PROHIBITED_CLASSES = frozenset({
    FailureClass.TRANSIENT_EXTERNAL,
    FailureClass.PENDING_EXTERNAL,
    FailureClass.AUTHENTICATION,
    FailureClass.AUTHORIZATION,
    FailureClass.POLICY_BLOCK,
    FailureClass.CORRUPTED_STATE,
    FailureClass.KNOWN_RUNTIME_REPAIR,
})
_PROTECTED_MARKERS = (
    "external_action_key", "kernel_push_key", "submission_key", "submission_marker",
    "reconcile_with_kaggle", "first_submission_approved", "require_telegram_approve",
    "approved_contract_hash", "daily_submission", "duplicate submission", "duplicate kernel",
    "credential", "api_key", ".env", "browser_submit", "approval", "protected_paths",
    "automatic_promotion", "repair acceptance",
)
_TEST_WEAKENING_MARKERS = (
    "pytest.skip", "pytest.mark.skip", "pytest.mark.xfail", "assert true",
    "except exception: pass", "except: pass",
)
_HIGH_PATH_MARKERS = (
    "supervisor/recovery", "supervisor/generation", "supervisor/resume", "supervisor/rollback",
    "supervisor/promote", "supervisor/worker", "supervisor/impact", "replay", "generation",
    "outbox", "approval", "credential", "submission", "kernel_push",
)


def _default_settings():
    from kaggle_agent.supervisor.config import AutoSafeSettings

    return AutoSafeSettings()


def _reproduction(spec: RepairSpec | None) -> ReproductionStrength:
    if spec is None:
        return ReproductionStrength.LOG_ONLY
    try:
        mode = getattr(spec, "reproduction_mode", "NO_REPRO")
        if mode == "EXISTING_TEST_REPRO":
            return ReproductionStrength.EXISTING_DETERMINISTIC_TEST
        return ReproductionStrength(mode)
    except ValueError:
        return ReproductionStrength.NO_REPRO


def _subsystem(incident: Incident, paths: Iterable[str], diff: str) -> str:
    text = " ".join((*paths, incident.stage, diff)).lower().replace("\\", "/")
    if any(marker in text for marker in _PROTECTED_MARKERS):
        return "trust_base"
    if any(marker in text for marker in _HIGH_PATH_MARKERS):
        return "lifecycle_or_external_runtime"
    if "orchestrator" in text or "worker" in text:
        return "orchestrator"
    if any(marker in text for marker in ("competition/", "competitions/", "pipeline/", "adapter")):
        return "competition_adapter"
    if any(marker in text for marker in ("parser", "format", "helper", "utility")):
        return "pure_utility"
    if incident.stage in {"RESEARCH", "PLAN", "CODE", "LOCAL_SMOKE"}:
        return "stage_local"
    if incident.stage in {"KERNEL_TRAIN", "VALIDATE_SUB", "FEEDBACK"}:
        return "external_capable_stage"
    return "unknown"


def _base_score(subsystem: str) -> int:
    return {
        "pure_utility": 0,
        "stage_local": 1,
        "competition_adapter": 2,
        "orchestrator": 3,
        "external_capable_stage": 3,
        "lifecycle_or_external_runtime": 4,
        "trust_base": 5,
        "unknown": 2,
    }.get(subsystem, 2)


def _tier(score: int, *, prohibited: bool) -> RepairRiskTier:
    if prohibited:
        return RepairRiskTier.PROHIBITED
    if score >= 6:
        return RepairRiskTier.HIGH
    if score >= 3:
        return RepairRiskTier.MEDIUM
    return RepairRiskTier.LOW


def _max_tier(left: RepairRiskTier, right: RepairRiskTier) -> RepairRiskTier:
    return left if _RISK_TIER_ORDER[left] >= _RISK_TIER_ORDER[right] else right


def _reproduction_rank(value: ReproductionStrength) -> int:
    return {
        ReproductionStrength.NO_REPRO: 0,
        ReproductionStrength.LOG_ONLY: 1,
        ReproductionStrength.STATIC_REPRO: 2,
        ReproductionStrength.DETERMINISTIC_COMMAND_REPRO: 3,
        ReproductionStrength.NEW_REGRESSION_TEST: 4,
        ReproductionStrength.EXISTING_DETERMINISTIC_TEST: 4,
    }[value]


def evaluate_repair_risk(
    incident: Incident,
    classification: FailureClassification,
    spec: RepairSpec | None = None,
    *,
    changed_paths: Iterable[str] = (),
    changed_lines: int = 0,
    diff: str = "",
    external_state: ExternalStateCertainty = ExternalStateCertainty.NO_EXTERNAL_ACTION,
    failed_attempts: int = 0,
    same_signature_failures: int = 0,
    reviewer_findings: Iterable[str] = (),
    minimum_tier: RepairRiskTier | None = None,
    settings=None,
) -> RepairRiskDecision:
    """Evaluate one repair using only deterministic, explicit evidence."""
    settings = settings or _default_settings()
    paths = tuple(path.replace("\\", "/").lstrip("./") for path in changed_paths)
    hint_paths = paths or tuple(getattr(spec, "likely_files", ()) if spec else ()) or tuple(classification.likely_files)
    subsystem = _subsystem(incident, hint_paths, diff)
    reproduction = _reproduction(spec)
    reasons: list[str] = [f"subsystem={subsystem}"]
    positive: list[str] = []
    negative: list[str] = []
    score = _base_score(subsystem)
    if subsystem in {"orchestrator", "external_capable_stage"}:
        score = max(score, 3)

    if reproduction is ReproductionStrength.EXISTING_DETERMINISTIC_TEST:
        score -= 2
        positive.append("existing deterministic reproduction")
    elif reproduction is ReproductionStrength.NEW_REGRESSION_TEST:
        score -= 1
        positive.append("new deterministic regression test")
    elif reproduction in {ReproductionStrength.LOG_ONLY, ReproductionStrength.NO_REPRO}:
        score += 2
        negative.append("weak or missing reproduction")
    else:
        positive.append("deterministic reproduction")

    source_files = sum(path.endswith(".py") and not path.startswith("tests/") for path in paths)
    test_files = sum(path.startswith("tests/") and path.endswith(".py") for path in paths)
    if len(paths) > 1:
        score += 1
        negative.append("multiple changed files")
    if source_files > 2 or test_files > 2 or changed_lines > 250:
        score += 1
        negative.append("candidate exceeds low-risk scope")
    if len(paths) > 3:
        score += 1
        negative.append("candidate has broad file scope")
    if len({path.split("/")[0] for path in paths}) > 1:
        score += 1
        negative.append("candidate spans multiple top-level areas")
    if failed_attempts or same_signature_failures:
        score += failed_attempts + same_signature_failures
        score = max(score, 3)
        negative.append("repair history contains failed attempts")
    finding_count = len(tuple(reviewer_findings))
    if finding_count:
        score = max(score + min(2, finding_count), 3)
        negative.append("review findings increase uncertainty")

    if subsystem in {"orchestrator", "external_capable_stage"}:
        score = max(score, 3)

    text = " ".join((diff, *paths)).lower()
    protected_path = any(
        path in PROTECTED_PATHS
        or path.startswith("config/profiles/")
        or path.startswith(".git/")
        for path in paths
    )
    protected_semantics = protected_path or any(marker in text for marker in _PROTECTED_MARKERS)
    test_weakening = any(marker in text for marker in _TEST_WEAKENING_MARKERS)
    prohibited_class = classification.failure_class in _PROHIBITED_CLASSES
    dependency_change = any(
        path in {"pyproject.toml", "uv.lock", "requirements.txt"} or path.startswith("requirements-")
        for path in paths
    )
    if dependency_change:
        score = max(score + 3, 6)
        negative.append("dependency change requires authority")
    if external_state in {ExternalStateCertainty.PENDING, ExternalStateCertainty.AMBIGUOUS, ExternalStateCertainty.UNKNOWN}:
        score += 4
        negative.append(f"external state is {external_state.value}")
    if protected_semantics:
        reasons.append("protected semantic detected")
    if prohibited_class:
        reasons.append(f"failure class {classification.failure_class.value} is not a source-repair class")

    prohibited = protected_semantics or prohibited_class or test_weakening
    if test_weakening:
        reasons.append("test weakening pattern detected")
    tier = _tier(score, prohibited=prohibited)
    if subsystem == "lifecycle_or_external_runtime" and not prohibited:
        tier = RepairRiskTier.HIGH
    if minimum_tier is not None:
        floored_tier = _max_tier(tier, minimum_tier)
        if floored_tier is not tier:
            reasons.append(f"same-repair risk floor retained {minimum_tier.value}")
            negative.append("risk cannot de-escalate within the same repair")
        tier = floored_tier
    profile = settings.profile(tier.value)
    unresolved_external = external_state in {
        ExternalStateCertainty.PENDING, ExternalStateCertainty.AMBIGUOUS, ExternalStateCertainty.UNKNOWN,
    }
    sufficient_reproduction = _reproduction_rank(reproduction) >= _reproduction_rank(
        ReproductionStrength(profile.required_reproduction_strength)
    )
    if not sufficient_reproduction:
        negative.append("reproduction is weaker than the tier requirement")
    candidate_allowed = (
        classification.repairable
        and profile.allow_candidate_generation
        and not unresolved_external
        and not prohibited
        and sufficient_reproduction
    )
    automatic_allowed = (
        candidate_allowed
        and tier in {RepairRiskTier.LOW, RepairRiskTier.MEDIUM}
        and profile.automatic_promotion
        and not unresolved_external
        and not dependency_change
        and not protected_semantics
    )
    if getattr(settings, "policy", "risk_adaptive") == "conservative":
        automatic_allowed = False
        negative.append("conservative policy requires authority before promotion")
    if automatic_allowed:
        positive.append("tier envelope permits automatic promotion")
    else:
        negative.append("automatic promotion is not permitted by the deterministic envelope")
    if unresolved_external:
        reasons.append("authoritative external reconciliation is required before autonomy")
    reasons.extend((f"failure_class={classification.failure_class.value}", f"reproduction={reproduction.value}"))
    max_source_files = min(profile.max_source_files, getattr(settings, "global_max_source_files", profile.max_source_files))
    max_test_files = min(profile.max_test_files, getattr(settings, "global_max_test_files", profile.max_test_files))
    max_changed_lines = min(profile.max_changed_lines, getattr(settings, "global_max_changed_lines", profile.max_changed_lines))
    return RepairRiskDecision(
        tier=tier,
        score=score,
        reasons=tuple(dict.fromkeys(reasons)),
        positive_factors=tuple(dict.fromkeys(positive)),
        negative_factors=tuple(dict.fromkeys(negative)),
        candidate_generation_allowed=candidate_allowed,
        automatic_promotion_allowed=automatic_allowed,
        max_source_files=max_source_files,
        max_test_files=max_test_files,
        max_changed_lines=max_changed_lines,
        max_implementation_attempts=profile.max_attempts,
        required_reproduction_strength=ReproductionStrength(profile.required_reproduction_strength),
        require_focused_tests=profile.require_focused_tests,
        require_adjacent_tests=profile.require_adjacent_tests,
        require_full_tests=profile.require_full_tests,
        require_static_safety=profile.require_static_safety,
        require_spec_review=profile.require_spec_review,
        require_code_review=profile.require_code_review,
        external_reconciliation_required=unresolved_external or bool(incident.external_job),
        authority_required=not automatic_allowed,
        external_state=external_state,
        reproduction_strength=reproduction,
        subsystem=subsystem,
        changed_source_files=source_files,
        changed_test_files=test_files,
        changed_lines=changed_lines,
        risk_floor=minimum_tier,
    )


def external_state_for_incident(incident: Incident, outbox) -> ExternalStateCertainty:
    """Map durable outbox state to the policy's conservative certainty enum."""
    if not incident.external_job:
        return ExternalStateCertainty.NO_EXTERNAL_ACTION
    item = outbox.get(incident.external_job)
    if item is None:
        return ExternalStateCertainty.UNKNOWN
    if item.status == "accepted":
        return ExternalStateCertainty.RECONCILED_EXACT
    if item.status == "rejected":
        return ExternalStateCertainty.RECONCILED_ABSENT
    if item.status in {"prepared", "sent", "unknown"}:
        return ExternalStateCertainty.PENDING
    return ExternalStateCertainty.UNKNOWN


def record_risk_decision(
    state,
    decision: RepairRiskDecision,
    *,
    phase: str,
    failure_class: str = "UNKNOWN",
    incident_id: str | None = None,
    previous_tier: RepairRiskTier | None = None,
) -> None:
    """Persist bounded, non-sensitive policy counters and the latest decision."""
    metrics = state.read_json("risk-metrics.json", {}) or {}
    metrics.setdefault("decisions", 0)
    metrics["decisions"] += 1
    tiers = metrics.setdefault("tiers", {})
    tiers[decision.tier.value] = tiers.get(decision.tier.value, 0) + 1
    failure_classes = metrics.setdefault("failure_classes", {})
    failure_classes[failure_class] = failure_classes.get(failure_class, 0) + 1
    phases = metrics.setdefault("phases", {})
    phases[phase] = phases.get(phase, 0) + 1
    from_tier = previous_tier.value if previous_tier is not None else "NONE"
    transition = f"{from_tier}->{decision.tier.value}"
    transitions = metrics.setdefault("transitions", {})
    transitions[transition] = transitions.get(transition, 0) + 1
    if previous_tier is not None:
        if _RISK_TIER_ORDER[decision.tier] > _RISK_TIER_ORDER[previous_tier]:
            metrics["risk_escalations"] = metrics.get("risk_escalations", 0) + 1
        elif _RISK_TIER_ORDER[decision.tier] < _RISK_TIER_ORDER[previous_tier]:
            metrics["risk_deescalation_attempts"] = metrics.get("risk_deescalation_attempts", 0) + 1
    transition_reasons = metrics.setdefault("transition_reasons", {})
    if previous_tier is not None and _RISK_TIER_ORDER[decision.tier] > _RISK_TIER_ORDER[previous_tier]:
        transition_reasons.setdefault(transition, []).extend(
            reason for reason in decision.negative_factors if reason not in transition_reasons[transition]
        )
    if decision.automatic_promotion_allowed:
        metrics["automatic_promotion_eligible"] = metrics.get("automatic_promotion_eligible", 0) + 1
    if decision.authority_required:
        metrics["authority_required"] = metrics.get("authority_required", 0) + 1
    if decision.external_reconciliation_required:
        metrics["external_state_blocks"] = metrics.get("external_state_blocks", 0) + 1
    state.write_json("risk-metrics.json", metrics)
    state.write_json(
        "risk-latest.json",
        {
            "incident_id": incident_id,
            "phase": phase,
            "failure_class": failure_class,
            "previous_tier": previous_tier.value if previous_tier is not None else None,
            "transition": transition,
            "decision": decision.to_dict(),
        },
    )
