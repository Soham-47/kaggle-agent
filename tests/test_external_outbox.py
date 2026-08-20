from kaggle_agent.autonomy.outbox import (
    ExternalActionOutbox,
    kernel_push_key,
    reconcile_with_kaggle,
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
