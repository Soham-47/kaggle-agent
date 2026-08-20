from pathlib import Path
import subprocess

import pytest

from kaggle_agent.supervisor.generation import RuntimeGeneration, RuntimeRevision
from kaggle_agent.supervisor.promote import GenerationPromotion, PromotionError, RepairAcceptance
from kaggle_agent.supervisor.health import HealthResult
from kaggle_agent.supervisor.review import Review, ReviewVerdict
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.verify import VerificationHarness


def _generation() -> RuntimeGeneration:
    revision = RuntimeRevision("a" * 40, "b" * 40, "generation-0001")
    return RuntimeGeneration("generation-0001", revision, "/tmp/generation-0001", created_at="now")


def test_acceptance_requires_every_gate():
    acceptance = RepairAcceptance.all_passed()
    assert acceptance.accepted is True
    assert RepairAcceptance.all_passed(review_approved=False).accepted is False


def test_review_response_is_structured_and_defaults_to_reject():
    review = Review.from_mapping({"verdict": "APPROVE", "root_cause_fixed": True, "blocking_findings": []})
    assert review.verdict is ReviewVerdict.APPROVE
    assert Review.from_mapping({}).verdict is ReviewVerdict.REJECT


def test_verification_rejects_unallowlisted_commands(tmp_path: Path):
    harness = VerificationHarness()
    result = harness.run_commands(tmp_path, [["bash", "-c", "echo nope"]])
    assert result.passed is False
    assert "allowlist" in result.failures[0]


def test_verification_preserves_quoted_arguments(monkeypatch, tmp_path: Path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = VerificationHarness().verify(
        tmp_path,
        ("uv run pytest tests/test_bug.py -k 'test_total' --no-header -q",),
    )

    assert result.passed is True
    assert calls[1] == ["uv", "run", "pytest", "tests/test_bug.py", "-k", "test_total", "--no-header", "-q"]


def test_promotion_pointer_is_atomic_and_rollback_is_supported(tmp_path: Path):
    store = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    promotion = GenerationPromotion(store)
    generation = _generation()
    promotion.activate(generation, RepairAcceptance.all_passed())
    assert store.read_json("active-generation.json")["generation_id"] == generation.generation_id
    promotion.activate(RuntimeGeneration("generation-0002", generation.revision, "/tmp/g2", created_at="now"), RepairAcceptance.all_passed())
    promotion.rollback(generation)
    assert store.read_json("active-generation.json")["generation_id"] == generation.generation_id


def test_promotion_requires_read_only_health_check(tmp_path: Path):
    store = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    promotion = GenerationPromotion(store)
    generation = _generation()

    with pytest.raises(PromotionError, match="health check"):
        promotion.activate(
            generation,
            RepairAcceptance.all_passed(),
            health=HealthResult(False, ("import",), ("config failed",)),
        )

    assert store.read_json("active-generation.json") is None
    assert store.read_json("promotion.json") is None


def test_successful_promotion_records_promoted_transaction(tmp_path: Path):
    store = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    promotion = GenerationPromotion(store)
    generation = _generation()

    promotion.activate(
        generation,
        RepairAcceptance.all_passed(),
        health=HealthResult(True, ("import", "settings")),
    )

    assert store.read_json("active-generation.json")["generation_id"] == generation.generation_id
    assert store.read_json("promotion.json")["status"] == "PROMOTED"
