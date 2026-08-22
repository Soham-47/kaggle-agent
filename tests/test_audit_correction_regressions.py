import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_agent.autonomy.contracts import CompetitionContract
from kaggle_agent.autonomy.onboard import CompetitionBootstrapper
from kaggle_agent.autonomy.outbox import ExternalActionOutbox, reconcile_with_kaggle
import kaggle_agent.autonomy.outbox as outbox_module
from kaggle_agent.kaggle_api.submit_ops import submit_notebook
from fakes import FakeKaggleApi
from kaggle_agent.config import CompetitionConfig, load_competition, load_settings
from kaggle_agent.supervisor.generation import GenerationStore, RuntimeRevision
from kaggle_agent.supervisor.incidents import Incident, IncidentStore
from kaggle_agent.supervisor.loop import Supervisor
from kaggle_agent.supervisor.state import RuntimeLayout, SupervisorStateStore
from kaggle_agent.supervisor.worker import run_worker
from kaggle_agent.supervisor.protocol import WorkerExit, WorkerRequest
from kaggle_agent.supervisor.repair_flow import RepairCoordinator
from kaggle_agent.supervisor.policy import RepairPolicy
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore, HeartbeatThread


def _revision() -> RuntimeRevision:
    return RuntimeRevision("a" * 40, "b" * 40, "generation-1")


def _contract(**overrides):
    raw = {
        "id": "demo", "slug": "demo", "title": "Demo",
        "task": {"family": "tabular_regression"}, "metric": {"name": "RMSE", "direction": "min"},
        "data": {"identifier_columns": ["id"], "target_columns": ["target"], "hidden_id_strategy": "sample"},
        "submission": {"mode": "file", "output_file": "submission.csv", "columns": ["id", "target"]},
    }
    raw.update(overrides)
    return CompetitionContract.from_mapping(raw)


def test_worker_runs_exact_requested_generation_root(tmp_path: Path, monkeypatch):
    state_root = tmp_path / "state"
    generation = tmp_path / "generation"
    generation.mkdir()
    request_path = state_root / "workers" / "w1" / "request.json"
    request_path.parent.mkdir(parents=True)
    request = WorkerRequest("w1", "generation-1", "demo", None, "observe", None, None, _revision(), True)
    request_path.write_text(json.dumps({**request.to_dict(), "generation_path": str(generation)}), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state_root))
    seen = {}
    monkeypatch.setattr("kaggle_agent.orchestrator.run_daily", lambda *args, **kwargs: (seen.update(kwargs) or SimpleNamespace(hard_errors=[], experiment_id="exp-1")))
    assert run_worker(request_path) == 0
    assert seen["root"] == generation.resolve()


def test_supervisor_classifies_worker_exit_without_result_as_fatal(tmp_path: Path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.yaml").write_text("default_competition: demo\nsupervisor:\n  enabled: true\n  mode: observe\n", encoding="utf-8")
    settings = load_settings(tmp_path)

    class Process:
        pid = 123
        def poll(self):
            return 0

    monkeypatch.setattr("kaggle_agent.supervisor.loop.WorkerLauncher.start", lambda self, request, **kwargs: Process())
    monkeypatch.setattr(Supervisor, "_active_generation", lambda self, managed=False: SimpleNamespace(generation_id="g", path=str(tmp_path), revision=_revision()))
    supervisor = Supervisor(settings, tmp_path)
    result = supervisor.run_once(wait=True)
    assert result.status == WorkerExit.FATAL.value
    assert "without result" in result.reason
    result_json = SupervisorStateStore(supervisor.layout).read_json(f"workers/{result.worker_id}/result.json")
    incident_id = result_json["incident_id"]
    assert incident_id.startswith("incident-")
    assert IncidentStore(supervisor.layout.state_root).load(incident_id) is not None


def test_worker_fatal_result_uses_loadable_occurrence_id(tmp_path: Path, monkeypatch):
    state_root = tmp_path / "state"
    request_path = state_root / "workers" / "w-fatal" / "request.json"
    request_path.parent.mkdir(parents=True)
    request = WorkerRequest("w-fatal", "generation-1", "demo", None, "observe", None, None, _revision(), True)
    request_path.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    monkeypatch.setenv("KAGGLE_AGENT_SUPERVISOR_DIR", str(state_root))

    def fail(*args, **kwargs):
        raise RuntimeError("fatal worker test")

    monkeypatch.setattr("kaggle_agent.orchestrator.run_daily", fail)
    assert run_worker(request_path) == 1
    payload = SupervisorStateStore(RuntimeLayout.for_repo(tmp_path, state_root)).read_json("workers/w-fatal/result.json")
    incident_id = payload["incident_id"]
    assert incident_id.startswith("incident-")
    assert IncidentStore(state_root).load(incident_id) is not None


def test_contract_hash_ignores_persisted_hash_and_authoritative_order(tmp_path: Path):
    base = _contract()
    persisted = CompetitionContract.from_mapping(dict(base.to_mapping(), contract_hash=base.compatibility_hash))
    assert persisted.compatibility_hash == base.compatibility_hash
    path = tmp_path / "reordered.csv"
    path.write_text("id,target\nb,1\na,2\n", encoding="utf-8")
    result = base.validate_submission(path, expected_ids=[("a",), ("b",)])
    assert not result.ok
    assert any("order" in error.lower() for error in result.errors)


def test_contract_enforces_declared_column_types(tmp_path: Path):
    contract = _contract(submission={"mode": "file", "output_file": "submission.csv", "columns": ["id", "target"], "column_types": {"id": "integer", "target": "float"}})
    path = tmp_path / "wrong-types.csv"
    path.write_text("id,target\nnot-an-int,1.0\n", encoding="utf-8")
    result = contract.validate_submission(path)
    assert not result.ok
    assert any("integer" in error.lower() for error in result.errors)


def test_malformed_present_contract_fails_closed_instead_of_legacy_fallback(tmp_path: Path):
    config = CompetitionConfig({"id": "demo", "slug": "demo", "task": {"family": "tabular_regression"}}, tmp_path / "demo.yaml")
    with pytest.raises(ValueError):
        _ = config.contract


def test_outbox_terminal_intent_is_idempotent_and_error_status_is_not_accepted(tmp_path: Path):
    outbox = ExternalActionOutbox(tmp_path)
    first = outbox.enqueue(action="kernel_push", idempotency_key="same", payload={"kernel_ref": "u/k"})
    accepted = outbox.reconcile(first.action_id, status="accepted", external_ref="u/k-v2")
    assert outbox.enqueue(action="kernel_push", idempotency_key="same", payload={"kernel_ref": "u/k"}) == accepted
    failed = outbox.enqueue(action="kernel_push", idempotency_key="different", payload={"kernel_ref": "u/k2"})
    reconciled = reconcile_with_kaggle(outbox, failed, kernel_status=lambda _: SimpleNamespace(status="ERROR"), submissions=lambda _: [])
    assert reconciled.status == "rejected"


def test_generation_snapshot_excludes_secret_files(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    (root / "credentials.json").write_text("{}\n", encoding="utf-8")
    run("add", "app.py", ".env", "credentials.json")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    generation = GenerationStore(state).create_snapshot(root)
    assert not Path(generation.path, ".env").exists()
    assert not Path(generation.path, "credentials.json").exists()


def test_incident_lineage_is_preserved_and_occurrences_remain_distinct(tmp_path: Path):
    from kaggle_agent.autonomy.outcomes import StageOutcome
    first = Incident.from_outcome(worker_id="w", generation_id="g2", competition="demo", outcome=StageOutcome.failure("CODE", "same"), stage_attempt=1, revision=_revision(), parent_occurrence_id="parent", originating_repair_id="repair-1", originating_generation_id="g1")
    second = Incident.from_outcome(worker_id="w", generation_id="g2", competition="demo", outcome=StageOutcome.failure("CODE", "same"), stage_attempt=1, revision=_revision(), parent_occurrence_id="parent", originating_repair_id="repair-1", originating_generation_id="g1")
    assert first.incident_id != second.incident_id and first.failure_signature == second.failure_signature
    IncidentStore(tmp_path).save(first)
    assert IncidentStore(tmp_path).load(first.incident_id) == first


def test_repair_diff_includes_untracked_files_and_protects_credentials(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    run = lambda *args: subprocess.run(("git", "-C", str(root), *args), check=True, capture_output=True)
    run("init", "-q")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    run("add", "app.py")
    run("-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "init")
    (root / "new.py").write_text("x = 2\n", encoding="utf-8")
    (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    state = SupervisorStateStore(RuntimeLayout.for_repo(root, tmp_path / "state"))
    coordinator = RepairCoordinator(root, state)
    paths = coordinator._changed_paths(root)
    assert "new.py" in paths and ".env" in paths
    assert ".env" in RepairPolicy().protected_violations(paths)


def test_nested_notebook_push_and_submit_are_exactly_once(tmp_path: Path):
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "tester/kernel", "enable_internet": True}), encoding="utf-8")
    api = FakeKaggleApi()
    outbox = ExternalActionOutbox(tmp_path)
    first = submit_notebook(api, competition="demo", message="m", kernel_folder=folder, kernel_ref="tester/kernel", output_file="submission.csv", status_fn=lambda _: "COMPLETE", poll_attempts=1, poll_seconds=0, outbox=outbox)
    second = submit_notebook(api, competition="demo", message="m", kernel_folder=folder, kernel_ref="tester/kernel", output_file="submission.csv", status_fn=lambda _: "COMPLETE", poll_attempts=1, poll_seconds=0, outbox=outbox)
    assert first.success and second.success
    assert len([call for call in api.submit_calls if call and call[0] == "kernels_push"]) == 1
    assert len([call for call in api.submit_calls if call and call[0] == "submit_code"]) == 1
    push = next(item for item in outbox._items().values() if item.action == "kernel_push")
    assert push.external_ref == "tester/fake-kernel" and push.external_version == 1


class _SubmitStatusApi(FakeKaggleApi):
    def __init__(self, status: str):
        super().__init__()
        self.submit_status = status

    def competition_submit_code(self, *args, **kwargs):
        self.submit_calls.append(("submit_code", kwargs.get("file_name"), kwargs.get("message"), kwargs.get("competition"), kwargs.get("kernel"), kwargs.get("kernel_version")))
        return SimpleNamespace(message="ok", ref="submission-ref", status=self.submit_status)


@pytest.mark.parametrize(
    "status, expected_success, expected_outbox",
    [("", False, "unknown"), ("ok", False, "unknown"), ("ERROR", False, "rejected"), ("SUCCESS", True, "accepted")],
)
def test_submit_code_requires_authoritative_status(tmp_path: Path, status: str, expected_success: bool, expected_outbox: str):
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "tester/kernel", "enable_internet": True}), encoding="utf-8")
    outbox = ExternalActionOutbox(tmp_path)
    result = submit_notebook(
        _SubmitStatusApi(status), competition="demo", message="m", kernel_folder=folder,
        kernel_ref="tester/kernel", output_file="submission.csv", status_fn=lambda _: "COMPLETE",
        poll_attempts=1, poll_seconds=0, outbox=outbox,
    )
    item = next(item for item in outbox._items().values() if item.action == "submit_code")
    assert result.success is expected_success
    assert item.status == expected_outbox
    if expected_outbox == "unknown":
        assert item.external_ref == "submission-ref"


def test_liveness_beats_do_not_refresh_stale_progress(tmp_path: Path, monkeypatch):
    class Process:
        pid = 321
        def __init__(self):
            self.terminated = False
        def poll(self):
            return None if not self.terminated else -15
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return -15

    layout = RuntimeLayout.for_repo(tmp_path, tmp_path / "state")
    store = HeartbeatStore(layout.state_root)
    store.write(Heartbeat("w-live", 321, "g1", None, "CODE", "stalled", 19.0, 1.0))
    thread = HeartbeatThread(store, Heartbeat("w-live", 321, "g1", None, "CODE", "stalled", 19.0, 1.0), 60)
    waits = iter((False, True))
    monkeypatch.setattr(thread._stop, "wait", lambda _: next(waits))
    thread._run()
    refreshed = store.read("w-live")
    assert refreshed is not None and refreshed.timestamp > 19.0
    assert refreshed.progress_timestamp == 1.0
    launcher = __import__("kaggle_agent.supervisor.worker", fromlist=["WorkerLauncher"]).WorkerLauncher(layout)
    process = Process()
    assert launcher.monitor_until_exit(
        process, "w-live", timeout_seconds=10, progress_timeout_seconds=5,
        grace_seconds=0, poll_seconds=0, now=refreshed.timestamp + 2.0,
    ) is True
    assert process.terminated


def test_one_shot_monitor_treats_missing_heartbeat_as_stale(tmp_path: Path):
    class Process:
        terminated = False
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return -15

    layout = RuntimeLayout.for_repo(tmp_path, tmp_path / "state")
    launcher = __import__("kaggle_agent.supervisor.worker", fromlist=["WorkerLauncher"]).WorkerLauncher(layout)
    process = Process()
    assert launcher.monitor(process, "missing", timeout_seconds=5, grace_seconds=0) is True
    assert process.terminated


def test_heartbeat_progress_update_resets_progress_timestamp(tmp_path: Path):
    store = HeartbeatStore(tmp_path / "state")
    heartbeat = Heartbeat("w-progress", 321, "g1", None, "CODE", "started", 10.0, 10.0)
    thread = HeartbeatThread(store, heartbeat, interval_seconds=60)
    thread.update(stage="TRAIN", progress="batch-2")
    refreshed = store.read("w-progress")
    assert refreshed is not None
    assert refreshed.progress_timestamp >= refreshed.timestamp - 0.001
    assert refreshed.last_progress_event == "batch-2"


def test_outbox_append_flushes_and_fsyncs(tmp_path: Path, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(outbox_module, "os", SimpleNamespace(fsync=lambda fd: calls.append(fd)), raising=False)
    ExternalActionOutbox(tmp_path).enqueue(action="submit", idempotency_key="one", payload={})
    assert calls


def test_onboarding_rejects_blank_sample_identifiers(tmp_path: Path):
    class Api(FakeKaggleApi):
        def competitions_list(self, **kwargs):
            return SimpleNamespace(competitions=[SimpleNamespace(ref="demo", title="Demo", url="u", deadline="d", evaluationMetric="AUC", isKernelsSubmissionsOnly=False, maxDailySubmissions=1, tags=[SimpleNamespace(name="tabular classification")])])
        def competition_list_files(self, *args, **kwargs):
            return SimpleNamespace(files=[SimpleNamespace(name="sample_submission.csv", total_bytes=10, ref="sample_submission.csv")], next_page_token=None)
        def competition_download_file(self, competition, file_name, path=None, **kwargs):
            destination = Path(path)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / file_name).write_text("id,target\n,0.5\n", encoding="utf-8")
    (tmp_path / "config" / "competitions").mkdir(parents=True)
    (tmp_path / "config" / "settings.yaml").write_text("default_competition: old\n", encoding="utf-8")
    (tmp_path / "memory").mkdir()
    result = CompetitionBootstrapper(tmp_path, __import__("kaggle_agent.kaggle_api.client", fromlist=["KaggleClient"]).KaggleClient(api=Api()).connect()).onboard("demo")
    assert result.outcome.state.value == "needs_authority"
    assert not (tmp_path / "config" / "competitions" / "demo.yaml").exists()
