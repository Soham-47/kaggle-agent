"""Bounded policy layer for coding-capable repair episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome


@dataclass(frozen=True)
class RepairEnvelope:
    allowed_contract_prefixes: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]

    @classmethod
    def default(cls) -> "RepairEnvelope":
        return cls(
            allowed_contract_prefixes=("runtime.resume_datasets",),
            allowed_path_prefixes=("src/", "tests/", "competitions/"),
        )


@dataclass(frozen=True)
class RepairLimits:
    max_episodes: int = 3
    identical_signature_retries: int = 1


@dataclass(frozen=True)
class RepairProposal:
    summary: str
    changed_paths: tuple[str, ...]
    changed_contract_fields: tuple[str, ...]
    regression_test: str
    verification_commands: tuple[str, ...]
    package_fingerprint: str


@dataclass
class DebugController:
    root: Path
    envelope: RepairEnvelope
    limits: RepairLimits
    _episodes: int = 0
    _signatures: dict[str, int] = field(default_factory=dict)

    def evaluate(
        self,
        failure: StageOutcome,
        proposal: RepairProposal,
        *,
        previous_package_fingerprint: str,
    ) -> StageOutcome:
        if failure.state is not OutcomeState.RECOVERABLE_FAILURE:
            raise ValueError("DEBUG only accepts recoverable failures")
        signature = failure.failure_signature or "unknown"
        if any(
            not any(field == prefix or field.startswith(prefix + ".") for prefix in self.envelope.allowed_contract_prefixes)
            for field in proposal.changed_contract_fields
        ):
            return StageOutcome(
                OutcomeState.NEEDS_AUTHORITY,
                "DEBUG",
                "repair changes protected competition semantics",
                evidence=proposal.changed_contract_fields,
            )
        if any(
            not any(path.startswith(prefix) for prefix in self.envelope.allowed_path_prefixes)
            or ".." in Path(path).parts
            for path in proposal.changed_paths
        ):
            return StageOutcome(
                OutcomeState.NEEDS_AUTHORITY,
                "DEBUG",
                "repair path is outside the workspace envelope",
                evidence=proposal.changed_paths,
            )
        seen = self._signatures.get(signature, 0)
        if self._episodes >= self.limits.max_episodes or seen >= self.limits.identical_signature_retries:
            return StageOutcome(
                OutcomeState.EXHAUSTED,
                "DEBUG",
                "repair retry budget exhausted",
                failure_signature=signature,
            )
        if proposal.package_fingerprint == previous_package_fingerprint:
            return StageOutcome(
                OutcomeState.EXHAUSTED,
                "DEBUG",
                "repair did not change the package fingerprint",
                failure_signature=signature,
            )
        if not proposal.regression_test or not proposal.verification_commands:
            return StageOutcome(
                OutcomeState.NEEDS_AUTHORITY,
                "DEBUG",
                "repair lacks test-first verification evidence",
            )
        self._episodes += 1
        self._signatures[signature] = seen + 1
        return StageOutcome.success(
            "DEBUG",
            proposal.summary,
            evidence=(proposal.regression_test, *proposal.verification_commands),
            artifacts=proposal.changed_paths,
        )
