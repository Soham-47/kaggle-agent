from pathlib import Path

import pytest

from kaggle_agent.autonomy.contracts import CompetitionContract, ContractError
from kaggle_agent.autonomy.debug import (
    DebugController,
    RepairEnvelope,
    RepairLimits,
    RepairProposal,
)
from kaggle_agent.autonomy.outcomes import OutcomeState, StageOutcome, failure_signature
from kaggle_agent.autonomy.adapters import AdapterRegistry, SupportResult, TaskAdapter, default_registry
from kaggle_agent.autonomy.generated_adapter import validate_generated_adapter


def _contract() -> CompetitionContract:
    return CompetitionContract.from_mapping(
        {
            "id": "demo",
            "slug": "owner-demo",
            "title": "Demo",
            "task": {"family": "tabular_classification", "modalities": ["tabular"]},
            "metric": {"name": "auc", "direction": "max", "implementation_source": "competition"},
            "data": {
                "train_sources": ["train.csv"],
                "test_sources": ["test.csv"],
                "identifier_columns": ["id"],
                "target_columns": ["target"],
                "hidden_id_strategy": "sample_submission",
            },
            "submission": {
                "mode": "file",
                "output_file": "submission.csv",
                "columns": ["id", "target"],
                "column_types": {"id": "string", "target": "float"},
            },
            "validation": {"minimum_rows": 1, "require_variation": True, "leakage_rules": []},
            "runtime": {"accelerator": "cpu", "internet": False, "dataset_slots": 1},
            "autonomy": {
                "first_submission_approved": False,
                "max_submissions_per_day": 1,
                "max_kernel_retries": 1,
                "max_gpu_hours": 1,
            },
        }
    )


def test_stage_outcome_requires_failure_signature_for_failure():
    with pytest.raises(ValueError, match="failure_signature"):
        StageOutcome(state=OutcomeState.RECOVERABLE_FAILURE, stage="CODE", summary="boom")


def test_failure_signature_ignores_paths_addresses_and_line_numbers():
    a = failure_signature("File /tmp/a.py, line 17, at 0xABC: NameError: x")
    b = failure_signature("File /kaggle/b.py, line 99, at 0x123: NameError: x")
    assert a == b


def test_contract_hash_is_stable_and_resume_attachments_are_excluded():
    contract = _contract()
    first = contract.compatibility_hash
    contract.raw["runtime"]["resume_datasets"] = ["owner/checkpoint-v2"]
    assert contract.compatibility_hash == first


def test_contract_rejects_unknown_submission_semantics():
    raw = _contract().to_mapping()
    raw["submission"]["columns"] = []
    with pytest.raises(ContractError, match="submission.columns"):
        CompetitionContract.from_mapping(raw)


class _Adapter(TaskAdapter):
    family = "tabular_classification"

    def supports(self, contract):
        return SupportResult(supported=contract.task_family == self.family, reason="family")


def test_adapter_registry_selects_supported_adapter_and_rejects_ambiguous():
    registry = AdapterRegistry([_Adapter()])
    assert registry.select(_contract()).family == "tabular_classification"
    registry.register(_Adapter())
    with pytest.raises(LookupError, match="ambiguous"):
        registry.select(_contract())


def test_debug_controller_rejects_patch_outside_envelope(tmp_path: Path):
    controller = DebugController(
        root=tmp_path,
        envelope=RepairEnvelope.default(),
        limits=RepairLimits(max_episodes=3, identical_signature_retries=1),
    )
    failure = StageOutcome.failure("VALIDATE_SUB", "wrong target", "sig-1")
    proposal = RepairProposal(
        summary="change target",
        changed_paths=("config/competitions/demo.yaml",),
        changed_contract_fields=("data.target_columns",),
        regression_test="tests/test_demo.py::test_target",
        verification_commands=("uv run pytest -q tests/test_demo.py",),
        package_fingerprint="new",
    )
    outcome = controller.evaluate(failure, proposal, previous_package_fingerprint="old")
    assert outcome.state is OutcomeState.NEEDS_AUTHORITY


def test_debug_controller_exhausts_unchanged_or_repeated_repairs(tmp_path: Path):
    controller = DebugController(
        root=tmp_path,
        envelope=RepairEnvelope.default(),
        limits=RepairLimits(max_episodes=3, identical_signature_retries=1),
    )
    failure = StageOutcome.failure("KERNEL_TRAIN", "name error", "sig-2")
    proposal = RepairProposal(
        summary="fix sampler",
        changed_paths=("competitions/demo/pipeline/kernel_recipe.py",),
        changed_contract_fields=(),
        regression_test="tests/test_recipe.py::test_sampler",
        verification_commands=("uv run pytest -q tests/test_recipe.py",),
        package_fingerprint="same",
    )
    unchanged = controller.evaluate(failure, proposal, previous_package_fingerprint="same")
    assert unchanged.state is OutcomeState.EXHAUSTED

    proposal = RepairProposal(**{**proposal.__dict__, "package_fingerprint": "new"})
    assert controller.evaluate(failure, proposal, previous_package_fingerprint="old").state is OutcomeState.SUCCESS
    assert controller.evaluate(failure, proposal, previous_package_fingerprint="old").state is OutcomeState.EXHAUSTED


@pytest.mark.parametrize("family", [
    "tabular_classification", "tabular_regression", "image_classification",
    "image_multilabel_classification", "image_segmentation", "image_detection",
    "text_classification", "text_regression", "text_generation",
    "time_series_forecasting", "ranking", "recommendation", "multimodal",
])
def test_default_registry_declares_every_initial_task_family(family):
    contract = _contract()
    contract.raw["task"]["family"] = family
    assert default_registry().select(contract).family == family


def test_generated_adapter_static_gate_rejects_network_secrets_and_submission(tmp_path: Path):
    source = tmp_path / "adapter.py"
    source.write_text("import requests\nTOKEN = 'x'\nrequests.post('https://x')\n")
    verdict = validate_generated_adapter(source)
    assert verdict.ok is False
    assert {"network", "secret"}.issubset(set(verdict.violations))
