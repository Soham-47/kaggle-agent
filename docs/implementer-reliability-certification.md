# DeepSeek Implementer Reliability Certification

## Original failure

The clean baseline was `922f3427fde9f1a5d034e34f42a4dbc6dae25a62` on branch
`supervisor-implementer-reliability`.

The first real implementer trace reproduced the prior failure. DeepSeek read
`src/bug.py`, then emitted an `apply_patch` action with an extra `path`
argument and an `*** Begin Patch` envelope. The existing tool accepts only a
typed `{patch}` argument containing a standard unified diff. Each edit was
rejected with `patch has no scoped writable files`; no source file was written
and the session ended at the implementer turn budget. The sanitized trace
classified the result as no patch/incomplete patch, tool argument/protocol
failure, and loop exhaustion.

The first diagnostic harness also used the temporary environment's Python to
invoke pytest, which did not contain pytest. That was corrected to the
allowlisted `uv run pytest` command before certification. The successful final
trace is persisted at
[`docs/implementer-attempt-trace.json`](implementer-attempt-trace.json).

## Changes

Only reliability-boundary changes were made:

- Added explicit `ImplementerStatus` outcomes and retained compatibility
  fields on `RepairImplementerResult`.
- Made action and per-tool argument envelopes strict.
- Enforced approved-path writes and read-before-write with optimistic hash
  protection; denied writes do not mutate the worktree.
- Made a real candidate diff mandatory before coordinator verification or
  independent review.
- Added supervisor-owned `VerificationFeedback` with bounded, sanitized
  output and durable per-attempt persistence.
- Added a maximum-two-attempt fresh-session implementation revision loop,
  including same-diff detection.
- Moved repair-tool audit output to the supervisor state root so audit JSONL
  cannot become a candidate source diff.
- Enforced the existing verification command allowlist in model-authored
  RepairSpecs.
- Strengthened static safety normalization for multiline exception swallowing,
  unbounded loops, subprocess/network additions, and credential reads.
- Added deterministic negative-gate tests, the ten-case local benchmark, and
  the real disposable REPAIR_ONLY validation harness.

## Attempt state machine

| State | Meaning |
| --- | --- |
| `PATCH_READY` | Non-empty, in-scope candidate exists; deterministic verification is still authoritative. |
| `NEEDS_VERIFICATION` | Candidate exists but must be checked outside the model. |
| `TOOL_POLICY_BLOCK` | Protected, out-of-scope, or otherwise denied action. |
| `PROTOCOL_FAILURE` | Malformed action or typed response after bounded retries. |
| `TURN_BUDGET_EXHAUSTED` | No acceptable completion within the implementer turn budget. |
| `NO_CHANGE` | Model stopped without a candidate diff. |
| `IMPLEMENTATION_FAILURE` | Unexpected role-boundary failure; fail closed. |

The model's `done` action only ends its editing turn. It does not establish
verification, acceptance, review, promotion, or safety.

## VerificationFeedback schema

`VerificationFeedback` contains:

```text
attempt
command
exit_code
failure_kind
failing_tests
stderr_excerpt
stdout_excerpt
changed_files
diff_summary
```

Text is bounded and control characters are removed. A failed focused check is
persisted under `repairs/<repair-id>/attempt-<n>-feedback.json` and supplied to
a fresh implementer session. The required verification command remains
supervisor-owned.

## Benchmark

The benchmark used ten disposable Git repositories, one writable source file,
zero writable test files, no dependencies, no Kaggle calls, and no Telegram
calls. Every case was handled by a real DeepSeek implementer and an
independent real DeepSeek code reviewer.

| Case | Attempts | First focused result | Final focused result | Files / lines | Reviewer |
| --- | ---: | --- | --- | --- | --- |
| NameError | 1 | pass | pass | 1 / 2 | APPROVE |
| Wrong constant | 1 | pass | pass | 1 / 2 | APPROVE |
| Off-by-one | 1 | pass | pass | 1 / 2 | APPROVE |
| Wrong conditional | 1 | pass | pass | 1 / 2 | APPROVE |
| Wrong mapping key | 1 | pass | pass | 1 / 2 | APPROVE |
| Missing import | 1 | pass | pass | 1 / 2 | APPROVE |
| Argument forwarding | 1 | pass | pass | 1 / 2 | APPROVE |
| Parser edge case | 1 | pass | pass | 1 / 2 | APPROVE |
| Enum conversion | 2 | fail, then pass | pass | 1 / 2 | APPROVE |
| State transition | 1 | pass | pass | 1 / 2 | APPROVE |

Result: **10/10 final focused-test-passing repairs**, **10/10 review
approvals**, zero protected-path violations, zero test weakenings, zero
dependency changes, and all candidates within the one-file/40-line benchmark
envelope. Full machine-readable results are in
[`docs/implementer-benchmark-results.json`](implementer-benchmark-results.json).

## Unsafe cases

Deterministic negative tests reject:

- broad and bare exception swallowing;
- unbounded retry loops;
- new subprocess or HTTP client use;
- credential and `.env` reads;
- approval-bypass markers;
- weakened or skipped tests;
- protected outbox/policy paths;
- dependency changes;
- changed-file, test-file, and line-budget excess;
- no-op candidates;
- repeated failed candidate diffs.

The test coverage is in
[`tests/test_supervisor_implementer_negative.py`](../tests/test_supervisor_implementer_negative.py),
[`tests/test_supervisor_agents.py`](../tests/test_supervisor_agents.py), and
[`tests/test_supervisor_repair_flow.py`](../tests/test_supervisor_repair_flow.py).

## Provider retries vs implementation revisions

Provider retries happen inside one role request when DeepSeek returns malformed
JSON or the provider fails. They use fresh model sessions and the bounded
`max_attempts` provider budget.

Implementation revisions happen only after a candidate exists and the
supervisor-owned focused verification fails. They use a fresh implementer
session, the approved original RepairSpec, the current candidate diff, and
typed `VerificationFeedback`. The implementation budget is independent and
defaults to two attempts. An identical effective diff is stopped as
`REPEATED_BAD_PATCH`.

## REPAIR_ONLY

**PASS.** The real disposable lifecycle completed:

```text
real classifier → CODE_DEFECT
real spec author → bounded RepairSpec
real spec reviewer → APPROVE
real implementer → PATCH_READY
deterministic pytest/compile verification → PASS
real independent code reviewer → APPROVE
supervisor commit → 1a40ac3e623130b0cef7aa189d3dc470f89ef68
```

The candidate was stored in an isolated worktree. The active generation was
`generation-0001` before and after the run. No promotion occurred. The
candidate worktree and state root were disposable and were removed with the
temporary validation directory after the result artifact was recorded.

## AUTO_SAFE canary

**NOT RUN.** The current supervisor loop explicitly returns
`NEEDS_AUTHORITY` for any mode other than `observe` or `repair_only` during
stabilization, and automatic activation is intentionally disabled. This phase
did not bypass that guard or claim an unattended promotion/resume canary.

## Full test suite

Baseline before changes:

```text
uv run python -m compileall -q src examples
uv run pytest -q -m "not integration"
619 passed, 1 deselected
```

Final verification:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                    644 passed, 1 deselected
git diff --check                                          PASS
```

Focused supervisor and safety tests after the final changes:

```text
uv run pytest -q tests/test_supervisor_agents.py tests/test_supervisor_repair_flow.py tests/test_supervisor_validation.py tests/test_supervisor_safety.py tests/test_supervisor_implementer_negative.py
55 passed
```

The real REPAIR_ONLY validation used no Kaggle or Telegram mutation.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: NOT READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

## Remaining risks

- The ten-case benchmark is deterministic and local; it does not establish
  reliability across all production repository shapes or future model
  versions.
- AUTO_SAFE promotion/resume remains unvalidated in this phase because the
  existing stabilization guard correctly blocks automatic activation.
- No Kaggle mutation was performed. External exactly-once guarantees remain
  covered by the existing read/reconciliation and outbox test suites, not by a
  new live mutation.
- Telegram was not needed for this local certification.
