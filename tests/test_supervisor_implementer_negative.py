"""Fail-closed gates for unsafe implementer candidates."""

from __future__ import annotations

import pytest

from kaggle_agent.supervisor.policy import DiffLimits, RepairPolicy


@pytest.mark.parametrize(
    ("name", "diff", "expected"),
    [
        ("broad exception swallow", "+    except Exception:\n+    pass\n", "broad_swallow"),
        ("bare exception swallow", "+except:\n+    pass\n", "bare_swallow"),
        ("unbounded retry loop", "+while True:\n+    retry()\n", "unbounded_loop"),
        ("new subprocess", "+subprocess.run(['curl'])\n", "subprocess"),
        ("new network client", "+httpx.get(url)\n", "http_client"),
        ("credential read", "+Path('.env').read_text()\n", "credential_read"),
        ("approval bypass marker", "+first_submission_approved = true\n", "approval_bypass"),
        ("test weakening", "-    assert result == 1\n+    assert True\n", "test_weakening"),
    ],
)
def test_unsafe_candidate_is_rejected_by_deterministic_gate(name: str, diff: str, expected: str):
    policy = RepairPolicy()
    findings = policy.scan_test_diff(diff) + policy.scan_text(diff) + policy.semantic_violations(diff)

    assert expected in findings, name


def test_protected_and_dependency_paths_are_rejected():
    policy = RepairPolicy(DiffLimits(2, 0, 120, allow_dependency_changes=False))

    assert policy.protected_violations(["src/kaggle_agent/autonomy/outbox.py"])
    assert "dependency_change" in policy.check_diff(["pyproject.toml"], 1)
    assert "source_file_limit" in policy.check_diff(["src/a.py", "src/b.py", "src/c.py"], 3)
    assert "changed_line_limit" in policy.check_diff(["src/a.py"], 121)
