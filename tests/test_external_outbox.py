from kaggle_agent.autonomy.outbox import (
    ExternalActionOutbox,
    kernel_push_key,
    reconcile_with_kaggle,
    submission_description,
    submission_marker,
    submission_key,
)
from kaggle_agent.kaggle_api.models import SubmissionRow


def test_outbox_deduplicates_unresolved_external_action_and_records_reconciliation(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    first = outbox.enqueue(
        action="kernel_push", idempotency_key="package-sha", payload={"kernel_ref": "owner/demo"}
    )
    second = outbox.enqueue(
        action="kernel_push", idempotency_key="package-sha", payload={"kernel_ref": "owner/demo"}
    )

    assert first.action_id == second.action_id
    assert outbox.pending() == [first]

    outbox.reconcile(first.action_id, status="accepted", external_ref="owner/demo/3")

    assert outbox.pending() == []
    resolved = outbox.get(first.action_id)
    assert resolved is not None
    assert resolved.status == "accepted"
    assert resolved.external_ref == "owner/demo/3"


def test_outbox_keeps_unknown_result_pending_instead_of_reissuing(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    item = outbox.enqueue(action="submit", idempotency_key="output-sha", payload={"file": "submission.csv"})
    outbox.mark_sent(item.action_id)

    assert outbox.pending()[0].status == "sent"
    assert outbox.enqueue(action="submit", idempotency_key="output-sha", payload={}).action_id == item.action_id


def test_reconciliation_accepts_only_exact_kernel_and_submission_evidence(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    kernel = outbox.enqueue(
        action="kernel_push", idempotency_key="kernel-sha", payload={"kernel_ref": "owner/demo"}
    )
    outbox.mark_sent(kernel.action_id)
    reconciled = reconcile_with_kaggle(
        outbox, kernel, kernel_status=lambda ref: "RUNNING", submissions=lambda _: []
    )
    assert reconciled.status == "accepted"
    assert reconciled.external_ref == "owner/demo"

    submit = outbox.enqueue(
        action="submit", idempotency_key="output-sha",
        payload={"competition": "demo", "message": "agent exp-1"},
    )
    outbox.mark_sent(submit.action_id)
    unresolved = reconcile_with_kaggle(
        outbox, submit, kernel_status=lambda _: "", submissions=lambda _: [
            SubmissionRow("s1", "x.csv", "complete", description="agent other")
        ],
    )
    assert unresolved.status == "sent"
    accepted = reconcile_with_kaggle(
        outbox, submit, kernel_status=lambda _: "", submissions=lambda _: [
            SubmissionRow("s2", "x.csv", "pending", description="agent exp-1")
        ],
    )
    assert accepted.status == "accepted"
    assert accepted.external_ref == "s2"


def test_external_keys_are_stable_across_stage_cycles():
    first = kernel_push_key("demo", "owner/kernel", "package-sha")
    second = kernel_push_key("demo", "owner/kernel", "package-sha")
    assert first == second
    assert first == kernel_push_key("demo", "owner/kernel", "package-sha")


def test_external_keys_change_for_competition_or_artifact():
    base = submission_key("demo", "file", "output-sha")
    assert submission_key("other", "file", "output-sha") != base
    assert submission_key("demo", "file", "other-output") != base
    assert submission_key("demo", "notebook", "output-sha") != base


def test_pending_external_key_reuses_one_action_across_cycles(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    key = kernel_push_key("demo", "owner/kernel", "package-sha")
    first = outbox.enqueue(
        action="kernel_push", idempotency_key=key, payload={"cycle_id": "cycle-1"}
    )
    second = outbox.enqueue(
        action="kernel_push", idempotency_key=key, payload={"cycle_id": "cycle-2"}
    )
    assert first.action_id == second.action_id


def test_submission_marker_is_stable_and_unique_per_artifact():
    marker = submission_marker("rsna-knee-abnormality-detection", "abcdef0123456789deadbeef")
    assert marker == submission_marker("rsna-knee-abnormality-detection", "abcdef0123456789deadbeef")
    assert marker.startswith("ka:rsna-knee-abnormality-detection:abcdef0123456789")
    assert marker != submission_marker("rsna-knee-abnormality-detection", "0000000123456789deadbeef")
    assert submission_description("demo", "abcdef0123456789", "cycle-2").startswith(
        "ka:demo:abcdef0123456789"
    )
    assert submission_description("demo", "abcdef0123456789", "cycle-1").split(" | ")[0] == submission_description(
        "demo", "abcdef0123456789", "cycle-2"
    ).split(" | ")[0]


def test_marker_reconciliation_selects_exact_unique_row(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    marker = submission_marker("demo", "abcdef0123456789")
    item = outbox.enqueue(
        action="submit",
        idempotency_key="stable",
        payload={
            "competition": "demo",
            "message": f"{marker} | agent cycle-1",
            "reconciliation_marker": marker,
        },
    )
    outbox.mark_sent(item.action_id)
    rows = [
        SubmissionRow("old", "x.csv", "complete", description="ka:demo:other | agent"),
        SubmissionRow("new", "x.csv", "complete", description=f"{marker} | agent cycle-2"),
    ]
    accepted = reconcile_with_kaggle(
        outbox,
        item,
        kernel_status=lambda _: "",
        submissions=lambda _: rows,
    )
    assert accepted.status == "accepted"
    assert accepted.external_ref == "new"


def test_ambiguous_marker_remains_pending(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    marker = submission_marker("demo", "abcdef0123456789")
    item = outbox.enqueue(
        action="submit",
        idempotency_key="stable",
        payload={"competition": "demo", "reconciliation_marker": marker},
    )
    outbox.mark_sent(item.action_id)
    rows = [
        SubmissionRow("a", "x.csv", "complete", description=marker),
        SubmissionRow("b", "x.csv", "complete", description=f"{marker} | duplicate"),
    ]
    unresolved = reconcile_with_kaggle(
        outbox, item, kernel_status=lambda _: "", submissions=lambda _: rows
    )
    assert unresolved.status == "sent"


def test_kernel_push_crash_after_external_send_reconciles_without_second_push(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    item = outbox.enqueue(action="kernel_push", idempotency_key="kernel-key", payload={"kernel_ref": "owner/kernel"})
    outbox.mark_sent(item.action_id)  # crash window: request may already be remote
    accepted = reconcile_with_kaggle(
        outbox, item, kernel_status=lambda ref: "RUNNING", submissions=lambda _: []
    )
    again = outbox.enqueue(action="kernel_push", idempotency_key="kernel-key", payload={"kernel_ref": "owner/kernel"})
    assert accepted.status == "accepted"
    assert again.action_id == item.action_id
    assert again.status == "accepted"


def test_submission_crash_before_or_after_send_never_reissues_same_key(tmp_path):
    outbox = ExternalActionOutbox(tmp_path)
    item = outbox.enqueue(
        action="submit", idempotency_key="submission-key",
        payload={"competition": "demo", "reconciliation_marker": "ka:demo:abc"},
    )
    outbox.mark_sent(item.action_id)  # crash before local result is persisted
    pending = reconcile_with_kaggle(
        outbox, item, kernel_status=lambda _: "", submissions=lambda _: []
    )
    same = outbox.enqueue(action="submit", idempotency_key="submission-key", payload={})
    assert pending.status == "sent"
    assert same.action_id == item.action_id
    assert same.status == "sent"
    accepted = reconcile_with_kaggle(
        outbox, item, kernel_status=lambda _: "",
        submissions=lambda _: [SubmissionRow("s1", "x", "complete", description="ka:demo:abc")],
    )
    assert accepted.status == "accepted"
    assert outbox.enqueue(action="submit", idempotency_key="submission-key", payload={}).status == "accepted"
