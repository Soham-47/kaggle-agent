# Live Kaggle Boundary Certification

## Scope and baseline

- Starting `origin/main`: `1ac3c9213fb1baad4d42473667911d4eab447867`
- Branch: `live-kaggle-boundary-certification`
- Worktree: `/home/soham/kaggle-agent-live-boundary-certification`
- State was isolated under a disposable `KAGGLE_AGENT_STATE_ROOT`.
- Kaggle authentication and read-only API access passed.
- DeepSeek was loaded through the existing dotenv path; no credential value was
  logged or persisted.

Baseline and focused verification before the live run:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     698 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_external_outbox.py \
  tests/test_replay_epoch.py tests/test_risk_adaptive_controlled_rollout.py \
  tests/test_telegram*.py                                 196 passed
uv run python scripts/run_full_system_harness.py           PASS
uv run python scripts/benchmark_code_agent.py              10/10 normal;
  3/3 bounded negative probes
git diff --check                                           PASS
```

## Live kernel certification

Competition: `rsna-knee-abnormality-detection` (active, kernels-only).
The normal agent path ran RESEARCH → PLAN → CODE → LOCAL_SMOKE → one-slice
KERNEL_TRAIN. Downstream submission phases were skipped to guarantee that this
run could issue only one kernel action.

The first kernel-only attempt failed before package creation because the
normal builder requires research-produced `study_ids`; it created no outbox
entry and made no Kaggle call. The corrected full path produced the following
single logical action:

```text
cycle:             live-kernel-boundary-20260821-02
kernel ref:        sohamgawd47foden/rsna-knee-agent-live-kernel-boundary-20260821-02
kernel version:    1
action id:         cf25383bb4aba5d4813a11e5
idempotency key:   18fbae8ed4c4f3f5e462ab8184cf5d195f311ec71733d9201b6adf2eb23680cf
package fingerprint: 755f425a2059949713ea836de25bfce34e93e33a636093d6bd131082bf104309
recipe hash:       990572c97565d8f3c065a0bcee02a5967bd6a55efa414f1ee8e3b32f7ad72b3a
```

Outbox lifecycle:

```text
prepared  2026-08-21T15:56:55.433Z
sent      2026-08-21T15:56:55.938Z
accepted  2026-08-21T16:11:03.676Z
```

Authoritative Kaggle polling observed:

```text
QUEUED → RUNNING → ERROR
```

The remote kernel ended in `KernelWorkerStatus.ERROR`; Kaggle returned no
failure-message text. The worker persisted this as a retryable external
`KERNEL_TRAIN` failure, not a `CODE_DEFECT`. It did not retry the push. The
outbox's `accepted` state records that the remote logical request was found;
the stage result separately records the terminal kernel failure.

The kernel history contains one package fingerprint and one kernel reference.
No duplicate push occurred.

## Kernel lost-response recovery

The real accepted kernel action was locally moved to `unknown` to model a lost
client response. The existing reconciler queried the same authoritative
Kaggle kernel reference and settled it back to `accepted`:

```text
local state: unknown → accepted
external ref: same kernel ref
additional kernel pushes: 0
```

This was a production-shaped fault injection around the real reconciler; no
second Kaggle mutation was issued.

## Training result handling

The real run observed `QUEUED`, `RUNNING`, and terminal `ERROR`. The agent did
not treat the remote error as a repository defect, did not blind-retry, and
persisted the result in the stage ledger and daily audit log. A successful
remote training result and output artifact were not available, so submission
through that kernel was correctly not attempted.

## Submission preparation

The only joined account-visible competition was the RSNA kernels-only
competition. Its failed kernel could not provide a valid submission output.

For a separate file-submission path, the official Titanic
`gender_submission.csv` was downloaded read-only and transformed into a
valid harmless artifact:

```text
competition:       titanic
competition mode:  file submission
rows:              418
columns:           PassengerId, Survived
validation:        PASS; all predictions in [0, 1]
artifact hash:     e848b685665ffefc8ce26e48e892e01f5cf6d7301c7c6ba59007e2c0166fb457
marker:            ka:titanic:e848b685665ffefc
submission limit:  10 available
```

No Titanic submission was made. The authoritative submission-history API
returned `You do not have a Team in this Competition`; joining/accepting
another competition would require external authority. The RSNA notebook path
would require a successful kernel output and was unavailable after the single
authorized kernel failed. No additional kernel or submission mutation was
created to work around this constraint.

## Submission lost-response recovery

The real reconciler was exercised against an existing authoritative RSNA
submission (`55627303`) using a local `sent` outbox action. It settled:

```text
sent → accepted
external ref: 55627303
additional submit calls: 0
```

This validates the lost-response reconciliation path, but it is not a live
mutation certification for a new submission.

## External effects and assertions

```text
live kernel mutation attempts:       1
logical kernel pushes:                1
duplicate kernel pushes:              0
live submission mutations:            0
duplicate submissions:                0
unsafe retries:                       0
duplicate workers/promotions:         0
protected changes:                    0
Telegram messages:                    0
Git revision in external action keys: no
```

## Defects found and fixed

No repository defect was fixed during this certification. The initial
kernel-only probe was a harness invocation that skipped the research stage;
the repository correctly rejected it before external mutation. The actual
full path exposed a real remote Kaggle kernel error and handled it as an
external failure without source repair or retry.

## Final verification

No tracked source files were changed. Generated competition files were
restored after the disposable run. The baseline and focused verification
listed above remained green; `git diff --check` passed.

## Readiness

```text
CODE_AGENT: READY
LIVE_KERNEL_MUTATION: CONDITIONAL
LIVE_SUBMISSION_MUTATION: NOT_READY
EXACTLY_ONCE_EXTERNAL_ACTIONS: CONDITIONAL
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

The kernel boundary demonstrated one real, reconciled logical mutation and
zero duplicate pushes, but the remote execution failed. Submission mutation
was not performed because no safe joined file-submission competition was
available and the only joined competition required successful kernel output.
