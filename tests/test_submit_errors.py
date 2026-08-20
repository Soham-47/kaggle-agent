"""Submit-error classes: 409 title conflict, 403 submit, DNS backoff."""

from __future__ import annotations

import json
import time
from pathlib import Path

from kaggle_agent.heal.submit_errors import (
    classify_submit_error,
    classify_submit_failure,
    is_409_title_conflict,
    is_403_submit,
    is_network_error,
)
from kaggle_agent.heal.pins import should_wait_approve


# --- classifier tests ---


def test_409_title_conflict():
    assert is_409_title_conflict("kernels_push failed: 409 title conflict")
    assert is_409_title_conflict("HTTPError: 409 Conflict: title already exists")
    assert not is_409_title_conflict("500 internal error")


def test_403_submit():
    assert is_403_submit("Permission 'kernelSessions.get' denied")
    assert is_403_submit("403 Forbidden")
    assert is_403_submit("Access denied: user does not have permission")
    assert not is_403_submit("200 OK")


def test_network_error():
    assert is_network_error("urlopen error [Errno -2] Name or service not known")
    assert is_network_error("Connection reset by peer")
    assert is_network_error("getaddrinfo failed")
    assert is_network_error("timed out")
    assert is_network_error("Connection refused")
    assert is_network_error("Network is unreachable")
    assert not is_network_error("409 title conflict")
    assert not is_network_error("random other error")


def test_classify_submit_error_ordering():
    assert classify_submit_error("urlopen error [Errno -2] Name or service not known") == "network"
    assert classify_submit_error("kernels_push failed: 409 title conflict") == "409"
    assert classify_submit_error("403 Permission denied") == "403"
    assert classify_submit_error("random other error") is None


def test_structured_submit_failure_marks_only_network_errors_retryable():
    network = classify_submit_failure("kernels_push: name or service not known")
    assert network.category == "network"
    assert network.retryable is True

    permission = classify_submit_failure("403 Permission 'kernelSessions.get' denied")
    assert permission.category == "permission"
    assert permission.retryable is False


# --- should_wait_approve: network error exemption ---


def test_should_wait_approve_ignores_network_errors():
    assert not should_wait_approve(
        validate_ok=True,
        submit_ok=False,
        dry_run=False,
        assume_approved=False,
        errors=["urlopen error [Errno -2] Name or service not known"],
    )


def test_should_wait_approve_ignores_409():
    assert not should_wait_approve(
        validate_ok=True,
        submit_ok=False,
        dry_run=False,
        assume_approved=False,
        errors=["kernels_push failed: 409 title conflict"],
    )


# --- 409 push retry in kernel_runner ---


def test_kernel_push_retries_after_409(tmp_path: Path):
    from fakes import FakeKaggleApi
    from helpers import copy_min_workspace
    from kaggle_agent.config import load_competition
    from kaggle_agent.kaggle_api import KaggleClient
    from kaggle_agent.train.kernel_runner import run_kernel_phase
    from kaggle_agent.train.notebook_builder import write_kernel_package

    real = Path(__file__).resolve().parents[1]
    root = tmp_path / "ka"
    copy_min_workspace(root, real)
    pipe = root / "competitions" / "rsna_knee" / "pipeline"
    pipe.mkdir(parents=True, exist_ok=True)
    (pipe / "methods.json").write_text(
        json.dumps({"dataset_sources": [], "model_sources": []}),
        encoding="utf-8",
    )
    comp = load_competition("rsna_knee", root)
    pkg = write_kernel_package(comp, root=root, username="tester", exp_id="409heal")
    # Write a bad owner into metadata so _fix_metadata_owner has something to fix
    meta = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    meta["id"] = "local-user/test-kernel"
    pkg.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    class FailThenOk(FakeKaggleApi):
        username = "tester"

        def __init__(self):
            super().__init__()
            self.push_attempts = []

        def kernels_push(self, folder, timeout=None, acc=None):
            self.push_attempts.append(folder)
            meta_path = Path(folder) / "kernel-metadata.json"
            if meta_path.is_file():
                mid = json.loads(meta_path.read_text(encoding="utf-8")).get("id", "")
                if mid.startswith("local-user/"):
                    raise RuntimeError("409 title conflict: kernel name already taken")
            return super().kernels_push(folder, timeout=timeout)

    api = FailThenOk()
    client = KaggleClient(api=api).connect()
    run = run_kernel_phase(
        client, pkg, push=True, root=root, competition=comp.slug, exp_id="409heal"
    )
    assert run.pushed is True
    assert run.ok is True
    assert run.errors == []
    # metadata owner was rewritten to "tester"
    meta_after = json.loads(pkg.metadata_path.read_text(encoding="utf-8"))
    assert meta_after["id"].startswith("tester/")
    assert len(api.push_attempts) == 2


# --- network retry in client ---


def test_client_retries_network_on_submissions(tmp_path: Path):
    from kaggle_agent.kaggle_api.client import KaggleClient

    call_count = 0

    class FakeApi:
        def __init__(self):
            self.username = "tester"

        def authenticate(self):
            pass

        def competition_submissions(self, competition):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("urlopen error: Name or service not known")
            return []

    client = KaggleClient(api=FakeApi()).connect()
    rows = client.submissions("test-comp")
    assert rows == []
    assert call_count == 3


def test_client_gives_up_after_max_retries(tmp_path: Path):
    from kaggle_agent.kaggle_api.client import KaggleClient

    class AlwaysFailApi:
        def __init__(self):
            self.username = "tester"
            self.attempts = 0

        def authenticate(self):
            pass

        def kernels_push(self, folder):
            self.attempts += 1
            raise ConnectionError("urlopen error: Name or service not known")

    api = AlwaysFailApi()
    client = KaggleClient(api=api).connect()
    try:
        client.kernels_push("/tmp/fake")
        assert False, "should have raised"
    except Exception:
        pass
    assert api.attempts == 3  # default attempts
