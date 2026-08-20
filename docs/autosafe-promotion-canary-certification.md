# AUTO_SAFE Promotion & Resume Certification

## Baseline

- Starting merged-main SHA: `b11ddefd4fd40cc0daebd13dea3b6570f8133a68`
- Branch: `supervisor-autosafe-canary-current`
- Implementation candidate revision: `03dc7fe`
- Original dirty checkout: preserved outside this worktree.
- Checked-in AUTO_SAFE defaults: unchanged (`supervisor.enabled: false`, automatic promotion disabled).

Baseline before this phase:

```text
uv run python -m compileall -q src examples scripts  PASS
uv run pytest -q -m "not integration"                 644 passed, 1 deselected
git diff --check                                       PASS
```

## Promotion implementation

The temporary guard was in `Supervisor._handle_worker_result`: `observe`
stopped after specification, `repair_only` retained a candidate, and other
modes returned `NEEDS_AUTHORITY`. The explicit `auto_safe` path now requires
`supervisor.promotion.automatic: true` and all existing deterministic
acceptance gates before it can activate a repair.

The path reuses `RepairCoordinator`, `GenerationPromotion`,
`RuntimeGeneration`, `ResumeRequest`, `StageLedger`, and
`ExternalActionOutbox`. It adds no second generation, outbox, approval, or
checkpoint system.

Additional stabilization fixes found during the live canary:

- malformed but scoped unified diffs are returned as retryable tool feedback;
- duplicate successful reads and `done` before a candidate diff receive bounded
  supervisor feedback;
- quoted verification arguments are parsed with `shlex.split` while retaining
  the existing command allowlist;
- successful replacement-worker completion settles the durable transaction as
  `RESUMED`;
- `ResumeRequest.preserved_stages` records stages before the resume point.

## Real DeepSeek certification

The credential-safe preflight returned `DeepSeek: AVAILABLE` using the normal
provider path. The five independent roles completed with typed outputs:

| Role | Provider | Fresh session | Result |
| --- | --- | --- | --- |
| classifier | configured DeepSeek | yes | `CODE_DEFECT` |
| RepairSpec author | configured DeepSeek | yes | bounded spec |
| spec reviewer | configured DeepSeek | yes | `APPROVE` |
| implementer | configured DeepSeek | yes | `PATCH_READY` |
| code reviewer | configured DeepSeek | yes | `APPROVE` |

The existing real-provider smoke and REPAIR_ONLY validation completed without
Kaggle or Telegram calls. No key or authorization material is stored here.

## REPAIR_ONLY certification

The real disposable REPAIR_ONLY lifecycle passed:

```text
classification: CODE_DEFECT (confidence 0.95)
spec review: APPROVE
candidate: CANDIDATE_ACCEPTED
implementer attempts: 1
code review: APPROVE
active generation before: generation-0001
active generation after:  generation-0001
```

The candidate commit and verification metadata are recorded in
`docs/repair-only-certification.json`; REPAIR_ONLY did not promote.

## AUTO_SAFE canary incident

The canary used worktree `/tmp/kaggle-agent-autosafe-real-canary` and a fresh
disposable state root. The fixture was committed before the supervisor started;
no source edits were made during the successful run.

```text
initial generation: generation-0001
base revision: 0087ea4...
incident: 74e24d24c28040544919
stage: CODE
failure: NameError: name 'valuez' is not defined
failure signature: 74e24d24c28040544919
repair: repair-74e24d24c280-a1
allowed source paths: src/kaggle_agent/bug.py
candidate: 73587bd4e331f16033fb768c092226c7295c3755
candidate diff: 1 source file, 2 changed lines, no test/dependency changes
focused verification: PASS
broader verification: PASS
spec review: APPROVE
code review: APPROVE
```

The repair changed `return sum(valuez)` to `return sum(values)`.

## Promotion

```text
old generation: generation-0001
new generation: generation-0002
health check: PASS (import, settings, competition, runtime_state)
pointer lifecycle: PREPARED → atomic switch → PROMOTED → RESUMED
replacement workers: exactly one
active generation after: generation-0002
```

The repaired generation is committed and immutable. The developer checkout
was not modified.

## Resume

`ResumeRequest` was persisted and consumed by the replacement worker:

```text
resume stage: CODE
preserved stages: RESEARCH, PLAN
invalidated stages: CODE and downstream stages
replay epochs: CODE/downstream epoch 1; preserved stages epoch 0
stage calls: RESEARCH=1, PLAN=1, CODE=2
replacement result: SUCCESS
```

The preserved stage callables were not invoked again. The failed stage ran
again and completed after the repair.

## Crash and rollback coverage

The existing recovery suite passed for prepared promotion recovery, durable
replacement-worker adoption, ambiguous launch blocking, successful resumed
worker settlement, and fatal replacement rollback. The live canary itself had
no process termination or external action boundary injected.

## Negative promotion gates

Existing deterministic gates remain fail-closed for failed health checks,
disabled automatic promotion, protected paths/semantics, invalid or empty
candidates, diff/test/dependency policy violations, rejected reviews, base
revision mismatch, exhausted budgets, and unresolved external actions.

## External effects

```text
Kaggle mutations: 0
Telegram messages: 0
```

The canary used only local synthetic stages. No competition submission or
kernel mutation was performed.

## Verification

```text
uv run python -m compileall -q src examples scripts  PASS
uv run pytest -q -m "not integration"                 655 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_external_outbox.py tests/test_replay_epoch.py
                                                       142 passed
git diff --check                                       PASS
```

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`AUTO_SAFE_CANARY` is ready for disposable/local canary use because the real
DeepSeek repair, deterministic gates, atomic promotion, checkpoint resume,
and successful continuation all passed. Unrestricted production AUTO_SAFE
remains disabled and is not certified by this report.

## Remaining risks

- The successful canary was local and synthetic; it performed no real Kaggle
  mutation and does not certify exactly-once behavior against live mutation.
- Telegram live ownership was not needed for this canary and was not exercised.
- The DeepSeek key supplied for validation was exposed in chat; rotate it before
  any further use.
