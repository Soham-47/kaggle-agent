from kaggle_agent.train import kernel_runner
from kaggle_agent.train.kernel_runner import KernelPushRepair, KernelRunResult
from kaggle_agent.train.kernel_job import KernelJob, load_kernel_job, save_kernel_job


class ErrorClient:
    def kernels_status(self, kernel_ref):
        return "ERROR"

    def kernels_failure_message(self, kernel_ref):
        return "worker failed"


def test_kernel_push_repair_policy_classifies_without_network_calls():
    assert KernelPushRepair.classify("Model instance version-number is required") == "pin"
    assert KernelPushRepair.classify("HTTP 409 title conflict") == "title_conflict"
    assert KernelPushRepair.classify("HTTP 500 worker failed") is None


def test_browser_traceback_extracts_short_traceback(monkeypatch):
    monkeypatch.setattr(
        "kaggle_agent.research.browser.fetch_via_browser_harness",
        lambda url: "Kernel output\nTraceback (most recent call last):\n  line 1\nValueError: bad input\nmore",
    )

    assert kernel_runner._browser_traceback("alice/demo") == (
        "Traceback (most recent call last):\nline 1\nValueError: bad input\nmore"
    )


def test_browser_traceback_failure_is_best_effort(monkeypatch):
    monkeypatch.setattr(
        "kaggle_agent.research.browser.fetch_via_browser_harness",
        lambda url: (_ for _ in ()).throw(RuntimeError("no Chrome")),
    )

    assert kernel_runner._browser_traceback("alice/demo") == ""


def test_poll_error_appends_traceback_for_folderless_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_runner, "_browser_traceback", lambda ref: "Traceback (most recent call last):\nboom")

    result = kernel_runner._poll_and_maybe_pull(
        ErrorClient(),
        KernelRunResult(ok=True, package=None, resumed=True, kernel_ref="alice/demo"),
        "alice/demo",
        None,
        pull_output_dir=None,
        root=tmp_path,
        competition="comp",
        exp_id="exp",
        poll_seconds=0,
        poll_attempts=1,
    )

    assert result.ok is False
    assert "worker failed\ntraceback:\nTraceback (most recent call last):\nboom" in result.errors[0]


class RunningThenCompleteClient:
    def __init__(self):
        self.status = "RUNNING"
        self.output_calls = 0

    def kernels_status(self, kernel_ref):
        return self.status

    def kernels_output(self, kernel_ref, dest_dir):
        self.output_calls += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        output = dest_dir / "submission.csv"
        output.write_text("id,prediction\n1,0.5\n", encoding="utf-8")
        return [str(output)]


def test_poll_exhaustion_is_pending_and_preserves_active_job(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_runner.time, "sleep", lambda _seconds: None)
    client = RunningThenCompleteClient()
    folder = tmp_path / "kernel"
    save_kernel_job(
        KernelJob(kernel_ref="alice/demo", folder=str(folder), status="RUNNING"),
        tmp_path,
    )

    result = kernel_runner.run_kernel_phase(
        client,
        None,
        push=True,
        pull_output_dir=folder / "output",
        root=tmp_path,
        poll_attempts=1,
    )

    assert result.pending is True
    assert result.ok is False
    assert load_kernel_job(tmp_path).kernel_ref == "alice/demo"
    assert client.output_calls == 0


def test_later_cycle_resumes_pending_job_and_pulls_after_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(kernel_runner.time, "sleep", lambda _seconds: None)
    client = RunningThenCompleteClient()
    folder = tmp_path / "kernel"
    save_kernel_job(
        KernelJob(kernel_ref="alice/demo", folder=str(folder), status="RUNNING"),
        tmp_path,
    )
    first = kernel_runner.run_kernel_phase(
        client, None, push=True, root=tmp_path, poll_attempts=1
    )
    client.status = "COMPLETE"

    second = kernel_runner.run_kernel_phase(
        client, None, push=True, root=tmp_path, poll_attempts=1
    )

    assert first.pending is True
    assert second.resumed is True
    assert second.pending is False
    assert second.ok is True
    assert client.output_calls == 1
    assert load_kernel_job(tmp_path).kernel_ref == "none"
