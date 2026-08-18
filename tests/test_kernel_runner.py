from kaggle_agent.train import kernel_runner
from kaggle_agent.train.kernel_runner import KernelRunResult


class ErrorClient:
    def kernels_status(self, kernel_ref):
        return "ERROR"

    def kernels_failure_message(self, kernel_ref):
        return "worker failed"


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
