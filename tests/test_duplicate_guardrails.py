"""Guardrails: stop the run when the kernel or its output is a repeat.

Root cause of the historical bug: kernel_runner._poll_and_maybe_pull compared
an un-stripped enum-repr status ("KernelWorkerStatus.COMPLETE") against the
plain DONE set, so the completion branch (output pull + clear_kernel_job)
never ran. The stale kernel_job.md then made every later cycle silently
"reuse" the same old kernel via package_matches_existing, which the
orchestrator treated as success. See test_heal_kernel_job.py for the
status-parsing regression test. These tests cover the second layer: even if
a kernel or its predictions end up identical for any other reason, the cycle
must stop with a clear error instead of continuing quietly.
"""

from pathlib import Path

from fakes import FakeKaggleApi
from helpers import copy_min_workspace, write_min_study_csv
from kaggle_agent.config import load_competition, load_settings
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.orchestrator import CycleResult, Orchestrator
from kaggle_agent.paths import repo_root
from kaggle_agent.state_md import AgentState
from kaggle_agent.train.kernel_history import record_output
from kaggle_agent.train.kernel_job import KernelJob, save_kernel_job
from kaggle_agent.train.notebook_builder import write_kernel_package


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "ka"
    copy_min_workspace(root, repo_root())
    write_min_study_csv(root)
    return root


def test_kernel_train_stops_on_identical_kernel(tmp_path: Path):
    root = _root(tmp_path)
    comp = load_competition("rsna_knee", root)
    settings = load_settings(root)

    # A previous kernel completed, but its job record was never cleared
    # (crash, or the pre-fix status-prefix bug). This is the same shape of
    # state that caused the "same kernel forever" bug.
    prior = write_kernel_package(
        comp,
        root=root,
        username="tester",
        exp_id="prior",
        enable_gpu=settings.kernel_enable_gpu,
        machine_shape=settings.kernel_machine_shape,
        enable_internet=settings.kernel_enable_internet,
    )
    save_kernel_job(
        KernelJob(
            kernel_ref=prior.kernel_ref,
            folder=str(prior.folder),
            status="COMPLETE",
            competition=comp.slug,
            exp_id="prior",
        ),
        root,
    )

    api = FakeKaggleApi()
    client = KaggleClient(api=api).connect()
    orch = Orchestrator(settings, comp, root=root, kaggle=client)
    result = CycleResult(competition=comp.slug, dry_run=False, experiment_id="next", plan_text="")
    state = AgentState(paused=False, competition="rsna_knee")

    orch._kernel_train(state, False, result)

    assert result.kernel_duplicate is True
    assert result.kernel_ok is False
    assert any("identical to previous experiment" in e for e in result.errors)
    pushes = [c for c in api.submit_calls if c and c[0] == "kernels_push"]
    assert pushes == []


def test_validate_stops_on_identical_output(tmp_path: Path):
    root = _root(tmp_path)
    comp = load_competition("rsna_knee", root)
    settings = load_settings(root)

    from kaggle_agent.experiment_fingerprint import submission_output_hash

    header = [comp.id_column, *comp.labels]
    rows = [
        f"study-{i}," + ",".join([str(0.5 + (i % 2) * 0.1)] * len(comp.labels))
        for i in range(1000)
    ]
    csv_text = ",".join(header) + "\n" + "\n".join(rows) + "\n"

    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / "e2"
    out_dir = kernel_dir / "output"
    out_dir.mkdir(parents=True)
    csv_path = out_dir / "submission.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    prior_hash = submission_output_hash(csv_path, comp.id_column)
    record_output(root, "prior-exp", prior_hash)

    orch = Orchestrator(settings, comp, root=root)
    result = CycleResult(
        competition=comp.slug,
        dry_run=False,
        experiment_id="e2",
        kernel_path=str(kernel_dir),
        kernel_ok=True,
    )
    state = AgentState(paused=False, competition="rsna_knee")

    orch._validate_sub(state, result)

    assert result.output_duplicate is True
    assert result.validate_ok is False
    assert any("predictions identical to previous experiment prior-exp" in e for e in result.errors)


def test_validate_records_and_accepts_new_output(tmp_path: Path):
    root = _root(tmp_path)
    comp = load_competition("rsna_knee", root)
    settings = load_settings(root)

    header = [comp.id_column, *comp.labels]
    rows = [
        f"study-{i}," + ",".join([str(0.7 + (i % 2) * 0.1)] * len(comp.labels))
        for i in range(1000)
    ]
    csv_text = ",".join(header) + "\n" + "\n".join(rows) + "\n"

    kernel_dir = root / "competitions" / "rsna_knee" / "notebooks" / "e3"
    out_dir = kernel_dir / "output"
    out_dir.mkdir(parents=True)
    csv_path = out_dir / "submission.csv"
    csv_path.write_text(csv_text, encoding="utf-8")

    orch = Orchestrator(settings, comp, root=root)
    result = CycleResult(
        competition=comp.slug,
        dry_run=False,
        experiment_id="e3",
        kernel_path=str(kernel_dir),
        kernel_ok=True,
    )
    state = AgentState(paused=False, competition="rsna_knee")

    orch._validate_sub(state, result)

    assert result.output_duplicate is False
    assert result.validate_ok is True
    from kaggle_agent.train.kernel_history import output_history_path

    assert output_history_path(root).is_file()
