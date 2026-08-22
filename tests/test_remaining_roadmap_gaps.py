import json
from pathlib import Path
from types import SimpleNamespace

from kaggle_agent.autonomy.contracts import CompetitionContract
from kaggle_agent.autonomy.onboard import CompetitionBootstrapper
from kaggle_agent.autonomy.outbox import ExternalActionOutbox
from kaggle_agent.config import load_competition
from kaggle_agent.kaggle_api.client import KaggleClient
from kaggle_agent.kaggle_api.submit_ops import submit_notebook
from kaggle_agent.pipeline.validate import validate_submission_csv
from kaggle_agent.supervisor.generation import RuntimeRevision
from kaggle_agent.supervisor.heartbeat import Heartbeat, HeartbeatStore
from kaggle_agent.supervisor.incidents import Incident, IncidentStore
from kaggle_agent.supervisor.worker import WorkerLauncher
from fakes import FakeKaggleApi


def test_worker_monitor_terminates_stale_process_and_writes_deterministic_hung_result(tmp_path):
    class Process:
        pid = 321

        def __init__(self):
            self.terminated = False
            self.polls = 0

        def poll(self):
            self.polls += 1
            return None if not self.terminated else -15

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return -15

    launcher = WorkerLauncher(__import__("kaggle_agent.supervisor.state", fromlist=["RuntimeLayout"]).RuntimeLayout.for_repo(tmp_path, tmp_path / "state"))
    process = Process()
    store = HeartbeatStore(tmp_path / "state")
    store.write(Heartbeat("w1", 321, "g1", None, "CODE", "stalled", 10.0))
    assert launcher.monitor_until_exit(process, "w1", timeout_seconds=5, grace_seconds=0, poll_seconds=0, now=20.0) is True
    assert process.terminated is True


def test_incident_occurrences_are_distinct_while_signature_is_stable(tmp_path):
    kwargs = dict(
        worker_id="w", generation_id="g", competition="demo",
        outcome=__import__("kaggle_agent.autonomy.outcomes", fromlist=["StageOutcome"]).StageOutcome.failure("CODE", "same"),
        stage_attempt=1, revision=RuntimeRevision("a", "b", "g"),
    )
    first = Incident.from_outcome(**kwargs)
    second = Incident.from_outcome(**kwargs)
    assert first.incident_id != second.incident_id
    assert first.occurrence_id == first.incident_id
    assert first.failure_signature == second.failure_signature
    IncidentStore(tmp_path).save(first)
    IncidentStore(tmp_path).save(second)
    assert IncidentStore(tmp_path).load(first.incident_id) == first
    assert IncidentStore(tmp_path).load(second.incident_id) == second


def test_contract_validation_distinguishes_tabular_classification_and_regression(tmp_path):
    base = {
        "id": "demo", "slug": "demo", "title": "Demo",
        "task": {"family": "tabular_classification"},
        "metric": {"name": "AUC", "direction": "max"},
        "data": {"identifier_columns": ["id"], "target_columns": ["target"], "hidden_id_strategy": "sample"},
        "submission": {"mode": "file", "output_file": "submission.csv", "columns": ["id", "target"]},
        "validation": {"minimum_rows": 2, "require_variation": True},
    }
    classification = tmp_path / "classification.csv"
    classification.write_text("id,target\na,0.1\nb,0.9\n", encoding="utf-8")
    assert validate_submission_csv(classification, contract=CompetitionContract.from_mapping(base)).ok
    regression = dict(base)
    regression["task"] = {"family": "tabular_regression"}
    regression["metric"] = {"name": "RMSE", "direction": "min"}
    values = tmp_path / "regression.csv"
    values.write_text("id,target\na,-12.5\nb,4.0\n", encoding="utf-8")
    assert validate_submission_csv(values, contract=CompetitionContract.from_mapping(regression)).ok


def test_contract_validation_rejects_duplicate_submission_ids(tmp_path):
    contract = CompetitionContract.from_mapping(
        {
            "id": "demo",
            "slug": "demo",
            "title": "Demo",
            "task": {"family": "tabular_classification"},
            "metric": {"name": "AUC", "direction": "max"},
            "data": {
                "identifier_columns": ["id"],
                "target_columns": ["target"],
                "hidden_id_strategy": "sample",
            },
            "submission": {
                "mode": "file",
                "output_file": "submission.csv",
                "columns": ["id", "target"],
            },
            "validation": {"minimum_rows": 2},
        }
    )
    path = tmp_path / "duplicate.csv"
    path.write_text("id,target\na,0.1\na,0.9\n", encoding="utf-8")
    result = validate_submission_csv(path, contract=contract)
    assert not result.ok
    assert any("duplicate" in error.lower() for error in result.errors)


def test_contract_validation_can_enforce_authoritative_sample_ids(tmp_path):
    contract = CompetitionContract.from_mapping(
        {
            "id": "demo",
            "slug": "demo",
            "title": "Demo",
            "task": {"family": "tabular_regression"},
            "metric": {"name": "RMSE", "direction": "min"},
            "data": {
                "identifier_columns": ["id"],
                "target_columns": ["target"],
                "hidden_id_strategy": "sample_submission",
            },
            "submission": {
                "mode": "file",
                "output_file": "submission.csv",
                "columns": ["id", "target"],
            },
        }
    )
    path = tmp_path / "wrong-ids.csv"
    path.write_text("id,target\nextra,1.0\n", encoding="utf-8")
    result = validate_submission_csv(
        path, contract=contract, expected_ids=[("expected",)]
    )
    assert not result.ok
    assert any("identifier" in error.lower() or "row" in error.lower() for error in result.errors)


def test_notebook_submit_tracks_nested_variant_push_in_outbox(tmp_path):
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(json.dumps({"id": "tester/kernel", "enable_internet": True}), encoding="utf-8")
    api = FakeKaggleApi()
    original = api.kernels_push
    def uncertain(path):
        raise ConnectionError("connection reset")
    api.kernels_push = uncertain
    outbox = ExternalActionOutbox(tmp_path)
    result = submit_notebook(
        api, competition="demo", message="m", kernel_folder=folder, kernel_ref="tester/kernel",
        output_file="submission.csv", status_fn=lambda _: "COMPLETE", poll_attempts=1,
        poll_seconds=0, outbox=outbox,
    )
    assert not result.success
    assert len(outbox.pending()) == 1
    api.kernels_push = original
    second = submit_notebook(
        api, competition="demo", message="m", kernel_folder=folder, kernel_ref="tester/kernel",
        output_file="submission.csv", status_fn=lambda _: "COMPLETE", poll_attempts=1,
        poll_seconds=0, outbox=outbox,
    )
    assert not second.success
    assert not [call for call in api.submit_calls if call and call[0] == "kernels_push"]


def test_nested_push_response_error_does_not_settle_outbox(tmp_path):
    folder = tmp_path / "pkg"
    folder.mkdir()
    (folder / "kernel-metadata.json").write_text(
        json.dumps({"id": "tester/kernel", "enable_internet": True}), encoding="utf-8"
    )
    api = FakeKaggleApi()
    api.kernels_push = lambda path: {"ref": "tester/kernel", "version_number": 1, "error": "rejected"}
    outbox = ExternalActionOutbox(tmp_path)
    result = submit_notebook(
        api,
        competition="demo",
        message="m",
        kernel_folder=folder,
        kernel_ref="tester/kernel",
        output_file="submission.csv",
        status_fn=lambda _: "COMPLETE",
        poll_attempts=1,
        poll_seconds=0,
        outbox=outbox,
    )
    assert not result.success
    assert outbox.pending()[0].status == "unknown"


def test_onboarding_scaffold_contains_executable_kernel_recipe_wrapper(tmp_path):
    root = tmp_path
    (root / "config" / "competitions").mkdir(parents=True)
    (root / "config" / "settings.yaml").write_text("default_competition: old\n", encoding="utf-8")
    (root / "memory").mkdir()
    class Api(FakeKaggleApi):
        def competitions_list(self, **kwargs):
            return SimpleNamespace(competitions=[SimpleNamespace(
                ref="demo", title="Demo", url="https://kaggle.com/c/demo", deadline="d",
                evaluationMetric="AUC", isKernelsSubmissionsOnly=False, maxDailySubmissions=2,
                tags=[SimpleNamespace(name="tabular classification")],
            )])
        def competition_list_files(self, *args, **kwargs):
            return SimpleNamespace(files=[SimpleNamespace(name="sample_submission.csv", total_bytes=10, ref="sample_submission.csv")], next_page_token=None)
        def competition_download_file(self, competition, file_name, path=None, **kwargs):
            destination = Path(path); destination.mkdir(parents=True, exist_ok=True)
            (destination / file_name).write_text("id,target\na,0.5\n", encoding="utf-8")
    result = CompetitionBootstrapper(root, KaggleClient(api=Api()).connect()).onboard("demo")
    assert result.contract is not None
    wrapper = root / "competitions/demo/pipeline/kernel_recipe.py"
    namespace: dict[str, object] = {}
    exec(compile(wrapper.read_text(encoding="utf-8"), str(wrapper), "exec"), namespace)
    assert isinstance(namespace.get("KERNEL_RECIPE_SOURCE"), str)
    compile(namespace["KERNEL_RECIPE_SOURCE"], "kernel_recipe_payload", "exec")
    assert load_competition("demo", root).contract is not None
