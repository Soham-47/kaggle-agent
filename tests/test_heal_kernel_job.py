from pathlib import Path

from fakes import FakeKaggleApi
from kaggle_agent.heal.policy import decide_next, load_heal, save_heal, HealState
from kaggle_agent.kaggle_api import KaggleClient
from kaggle_agent.train.kernel_job import (
    KernelJob,
    load_kernel_job,
    save_kernel_job,
)
from kaggle_agent.train.kernel_runner import (
    package_matches_existing,
    run_kernel_phase,
)
from kaggle_agent.train.notebook_builder import write_kernel_package
from kaggle_agent.config import load_competition
from kaggle_agent.paths import repo_root


def test_heal_improves_resets_streak(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    h = HealState(best_score="0.5", no_improve_days="3", tune_attempts="2")
    h = decide_next(h, public_score="0.6", metric_direction="max")
    assert h.best_score == "0.6"
    assert h.no_improve_days == "0"
    assert h.decision_next == "tune"


def test_heal_ladder_to_pause(tmp_path: Path):
    h = HealState(
        best_score="0.9",
        last_score="0.9",
        no_improve_days="4",
        tune_attempts="3",
        approach="new",
    )
    h = decide_next(
        h, public_score="0.89", metric_direction="max", max_no_improve_days=5
    )
    assert h.decision_next == "pause"
    assert int(h.no_improve_days) >= 5


def test_heal_never_pauses_without_a_score():
    h = HealState(best_score="none", no_improve_days="0", approach="baseline")
    for _ in range(30):
        h = decide_next(
            h, public_score=None, metric_direction="max", max_no_improve_days=5
        )
    assert h.decision_next == "tune"
    assert int(h.no_improve_days) == 0
    assert h.approach == "tune"


def test_heal_tune_then_recipe():
    h = HealState(tune_attempts="2", no_improve_days="0", approach="tune")
    h = decide_next(h, public_score=None, max_tune_attempts=3)
    assert h.decision_next == "tune"
    h = decide_next(h, public_score=None, max_tune_attempts=3)
    # after 3 tunes exhausted on next flat cycle
    assert h.decision_next in {"tune", "recipe"}


def test_kernel_job_resume_no_second_push(tmp_path: Path):
    import shutil

    root = tmp_path / "ka"
    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    (root / "memory").mkdir()
    if (real / "data").is_dir():
        shutil.copytree(real / "data", root / "data")
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="e1")

    api = FakeKaggleApi()
    client = KaggleClient(api=api).connect()
    # Seed active job as if previous cron pushed
    save_kernel_job(
        KernelJob(
            kernel_ref=pkg.kernel_ref,
            folder=str(pkg.folder),
            status="RUNNING",
            competition=comp.slug,
            exp_id="e1",
        ),
        root,
    )
    push_calls_before = list(api.submit_calls)
    run = run_kernel_phase(
        client,
        None,
        push=True,
        pull_output_dir=pkg.folder / "output",
        root=root,
        competition=comp.slug,
        exp_id="e1",
    )
    assert run.resumed is True
    # Fake status is COMPLETE → should pull and clear job
    assert run.status == "COMPLETE"
    assert (pkg.folder / "output" / "submission.csv").is_file()
    assert load_kernel_job(root).kernel_ref == "none" or not load_kernel_job(root).is_active
    # No new kernels_push
    new_pushes = [
        c for c in api.submit_calls if isinstance(c, tuple) and c and c[0] == "kernels_push"
    ]
    old_pushes = [
        c for c in push_calls_before if isinstance(c, tuple) and c and c[0] == "kernels_push"
    ]
    assert len(new_pushes) == len(old_pushes)


def test_kernel_retries_cpu_after_p100_ban(tmp_path: Path):
    import json
    import shutil

    root = tmp_path / "ka"
    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    (root / "memory").mkdir()
    if (real / "data").is_dir():
        shutil.copytree(real / "data", root / "data")
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(
        comp, root=root, username="tester", exp_id="gpu", enable_gpu=True
    )
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta["enable_gpu"] is True
    api = FakeKaggleApi(
        status_queue=["ERROR", "COMPLETE"],
        failure_message="You cannot use P100 on this competition",
    )
    client = KaggleClient(api=api).connect()
    run = run_kernel_phase(
        client,
        pkg,
        push=True,
        pull_output_dir=pkg.folder / "output",
        root=root,
        competition=comp.slug,
        exp_id="gpu",
    )
    assert run.ok
    assert run.status == "COMPLETE"
    meta2 = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta2["enable_gpu"] is False
    pushes = [c for c in api.submit_calls if c and c[0] == "kernels_push"]
    assert len(pushes) == 2


def test_kernel_job_enum_repr_status_is_done():
    job = KernelJob(
        kernel_ref="u/k",
        folder="/tmp/x",
        status="KernelWorkerStatus.COMPLETE",
    )
    assert job.is_active is False


def test_kernel_job_plain_running_status_is_active():
    job = KernelJob(
        kernel_ref="u/k",
        folder="/tmp/x",
        status="running",
    )
    assert job.is_active is True


def test_package_matches_existing_identical(tmp_path: Path):
    import shutil

    root = tmp_path / "ka"
    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    (root / "memory").mkdir()
    if (real / "data").is_dir():
        shutil.copytree(real / "data", root / "data")
    comp = load_competition("rsna_knee", root)
    pkg1 = write_kernel_package(comp, root=root, username="tester", exp_id="e1")
    pkg2 = write_kernel_package(comp, root=root, username="tester", exp_id="e2")
    job = KernelJob(
        kernel_ref=pkg1.kernel_ref,
        folder=str(pkg1.folder),
        status="COMPLETE",
    )
    assert package_matches_existing(pkg2, job) is True


def test_package_matches_existing_true_when_only_methods_change(tmp_path: Path):
    import shutil

    root = tmp_path / "ka"
    real = repo_root()
    shutil.copytree(real / "config", root / "config")
    (root / "competitions" / "rsna_knee").mkdir(parents=True)
    (root / "memory").mkdir()
    if (real / "data").is_dir():
        shutil.copytree(real / "data", root / "data")
    comp = load_competition("rsna_knee", root)
    pkg1 = write_kernel_package(comp, root=root, username="tester", exp_id="e1")
    pkg2 = write_kernel_package(comp, root=root, username="tester", exp_id="e2")
    (pkg1.folder / "methods.json").write_text('{"implement_steps": ["old step"]}')
    (pkg2.folder / "methods.json").write_text('{"implement_steps": ["new step"]}')
    job = KernelJob(
        kernel_ref=pkg1.kernel_ref,
        folder=str(pkg1.folder),
        status="COMPLETE",
    )
    assert package_matches_existing(pkg2, job) is True


def test_heal_persist(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    h = HealState(decision_next="recipe", note="test")
    save_heal(h, tmp_path)
    loaded = load_heal(tmp_path)
    assert loaded.decision_next == "recipe"
