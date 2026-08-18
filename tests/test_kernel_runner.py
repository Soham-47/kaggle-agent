from kaggle_agent.train import kernel_runner


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
