from pathlib import Path

from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api.models import SubmissionRow
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.pipeline.validate import validate_submission_csv
from kaggle_agent.experiment_fingerprint import submission_output_hash
from kaggle_agent.state_md import AgentState
from kaggle_agent.train.notebook_builder import KernelPackage
from helpers import copy_min_workspace
from kaggle_agent.paths import repo_root


class _KaggleWithoutSubmit:
    def submissions(self, competition, *, top=20):  # noqa: ANN001
        return [SubmissionRow("other", "submission.csv", "complete", description="agent another")]

    def kernels_status(self, kernel_ref):  # noqa: ANN001
        return ""


def test_unreconciled_submission_intent_is_never_sent_again(tmp_path: Path):
    root = tmp_path / "ka"
    copy_min_workspace(root, repo_root())
    orch = Orchestrator(
        load_settings(root), load_competition("rsna_knee", root), root=root,
        kaggle=_KaggleWithoutSubmit(),
    )
    candidate = root / "candidate.csv"
    candidate.write_text(
        orch.competition.id_column + "," + ",".join(orch.competition.labels) + "\n"
        + "study-1," + ",".join(["0.5"] * len(orch.competition.labels)) + "\n",
        encoding="utf-8",
    )
    assert validate_submission_csv(
        candidate, id_column=orch.competition.id_column, labels=orch.competition.labels
    ).ok
    result = CycleResult("rsna_knee", False, experiment_id="exp-1", candidate_csv=str(candidate))
    orch._assume_approved = True
    message = "agent exp-1"
    output_hash = submission_output_hash(candidate, orch.competition.id_column)
    action = orch._outbox.enqueue(
        action="submit",
        idempotency_key=f"{orch.competition.slug}:{orch.competition.submit_mode}:{output_hash}",
        payload={"competition": orch.competition.slug, "message": message},
    )
    orch._outbox.mark_sent(action.action_id)

    orch._submit(AgentState(), False, result)

    assert result.submission_pending is True
    assert result.submit_ok is False
    assert orch._outbox.get(action.action_id).status == "sent"


def test_unreconciled_kernel_push_intent_is_never_pushed_again(tmp_path: Path, monkeypatch):
    root = tmp_path / "ka"
    copy_min_workspace(root, repo_root())
    orch = Orchestrator(
        load_settings(root), load_competition("rsna_knee", root), root=root,
        kaggle=_KaggleWithoutSubmit(),
    )
    folder = root / "kernel"
    folder.mkdir()
    notebook = folder / "agent_baseline.ipynb"
    metadata = folder / "kernel-metadata.json"
    notebook.write_text("{}", encoding="utf-8")
    metadata.write_text("{}", encoding="utf-8")
    package = KernelPackage(folder, notebook, metadata, "owner/demo", "Demo")
    monkeypatch.setattr("kaggle_agent.orchestrator.write_kernel_package", lambda *a, **k: package)
    from kaggle_agent.train.kernel_history import package_fingerprint

    action = orch._outbox.enqueue(
        action="kernel_push", idempotency_key=package_fingerprint(folder),
        payload={"kernel_ref": package.kernel_ref},
    )
    orch._outbox.mark_sent(action.action_id)
    result = CycleResult("rsna_knee", False, experiment_id="exp-2")

    orch._kernel_train(AgentState(), False, result)

    assert result.kernel_pending is True
    assert result.kernel_ref == "owner/demo"
    assert orch._outbox.get(action.action_id).status == "sent"
