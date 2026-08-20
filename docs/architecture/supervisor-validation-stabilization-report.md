# Supervisor Validation & Controlled Rollout Report

This report covers the stabilization worktree
`supervisor-completion-stabilization`. No push or merge was performed.

## Candidate revisions

```text
implementation candidate: 170dd9e4ec0de8599c0441f78e0068b9615e7748
latest stabilization revision: f1ae6c1
clean-main baseline: b489498d8f24cd9fd3dad7f1ee7241b32407acf4
```

The original dirty checkout at `/home/soham/kaggle-agent` was preserved and
not modified.

## Ticket 1 — Baseline failure manifest

### Root cause/design

The previous 22-failure claim was incomplete. A fresh clean-main run produced
38 failures. The canonical manifest accounts for all 38 by exact test name:
[clean-main-failure-manifest.md](clean-main-failure-manifest.md).

### Files changed

- `docs/architecture/clean-main-failure-manifest.md`

### Tests and results

```text
UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run python -m compileall -q src
PASS

UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -m "not integration"
clean main: 478 passed, 38 failed, 1 deselected, 2 warnings in 23.04s
```

### Code review findings

The first review check found that the old report did not enumerate all 38
tests. The canonical manifest was added and checked for exactly 38 numbered
rows. No test behavior was changed.

### Findings fixed and remaining risks

Fixed: discrepancy resolved as 16 baseline-existing plus 22
missing-environment/dependency, with zero supervisor regressions.

Risk: the baseline is not healthy. The 22 prerequisite failures need either
the ignored dataset/artifacts and `pydicom`, or a separately approved baseline
policy. They are not suppressed by the supervisor.

## Ticket 2 — Production DeepSeek smoke validation

### Root cause/design

`DeepSeekSupervisorAgents` already uses separate UUID sessions and typed JSON
boundaries. The validation script exercises classifier, spec author, spec
reviewer, implementer, and code reviewer on a disposable local defect only.
The spec-review findings parser was tightened to reject non-string findings.

### Files changed

- `scripts/validate_supervisor_deepseek.py`
- `src/kaggle_agent/supervisor/agents.py`
- `tests/test_supervisor_agents.py`

### Tests and results

```text
UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/test_supervisor_agents.py
8 passed in 0.21s

UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run python scripts/validate_supervisor_deepseek.py
BLOCKED: DEEPSEEK_API_KEY is unavailable; no production role was called
exit_status=2
```

### Code review findings

Required finding: spec-review list elements were not type-checked. Fixed with
a strict list-of-strings parser and a regression test. The script contains no
Kaggle, Telegram, environment-dump, or mutation path.

### Findings fixed and remaining risks

Fixed: malformed spec-review findings cannot cross the typed boundary.

Risk: real DeepSeek role validation is not complete because the configured
production key is absent. Synthetic router tests are not a substitute for
provider smoke validation.

## Ticket 3 — OBSERVE end-to-end

### Root cause/design

The controlled path consumes a durable worker incident, applies deterministic
classification first, calls the DeepSeek classifier only for `UNKNOWN`, then
persists a spec and independent spec review without creating a worktree.

### Files changed

- `tests/test_supervisor_validation.py`

### Tests and results

```text
UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/test_supervisor_validation.py
5 passed in 0.69s
```

The deterministic synthetic `NameError` classification did not call the
fallback classifier. The opaque synthetic failure called it once and then
produced `SPEC_READY`. No synthetic false positive or false negative was
observed.

### Code review findings

No blocking findings. The tests assert persisted classification/spec artifacts
and not only method calls.

### Findings fixed and remaining risks

Fixed: none required for this ticket.

Risk: this is controlled local validation, not a live worker cycle. No Kaggle
mutation or live DeepSeek classification was attempted.

## Ticket 4 — REPAIR_ONLY end-to-end

### Root cause/design

The existing `RepairCoordinator` creates an exact-base worktree, runs the
supervisor-owned verification harness, calls an independent review callback,
commits an accepted candidate, and writes `accepted/<repair-id>.json` without
activating a generation.

### Files changed

- `tests/test_supervisor_validation.py`
- existing `tests/test_supervisor_repair_flow.py`
- `src/kaggle_agent/supervisor/repair_flow.py`
- `src/kaggle_agent/supervisor/worktree.py`
- `src/kaggle_agent/supervisor/policy.py`

### Tests and results

The focused repair-only test fixes a deterministic source defect with
`py_compile`, confirms a candidate revision and accepted artifact, and proves
the prior active-generation pointer is unchanged. The focused stabilization
slice passed `20 passed` before the final report-only additions.

### Code review findings

Required findings fixed:

- New untracked files were absent from policy/reviewer diffs. Worktree diff
  now uses Git intent-to-add before inspection.
- `RepairSpec.allowed_paths` was not enforced. The coordinator now rejects
  changed paths outside the approved scope.
- A no-op implementation could be accepted. Empty candidate diffs now fail
  the acceptance gate.
- Added xfail/skip test weakening is rejected.

### Findings fixed and remaining risks

Fixed: active generation is unchanged in REPAIR_ONLY and candidate state is
durable/inspectable.

Risk: the implementer in this run was deterministic local code, not a real
DeepSeek session, because the provider key was unavailable.

## Ticket 5 — Checkpoint replay E2E

### Root cause/design

`ResumeRequest.replay_epochs` is consumed by `StageInput`; preserved stages
retain epoch 0 and replay their durable outputs, while invalidated stages use
the incremented epoch and execute again. Git SHA is not inserted directly into
old stage identities.

### Files changed

- `tests/test_supervisor_validation.py`
- `tests/test_supervisor_worker.py`
- `src/kaggle_agent/supervisor/protocol.py`

### Tests and results

The replay test counted stage calls: preserved `RESEARCH` ran once total and
invalidated `KERNEL_TRAIN` ran twice (initial execution plus rerun). Worker
protocol validation also confirms ResumeRequest reaches `run_daily` and
round-trips all tuple fields.

### Code review findings

Required finding: protocol deserialization left preserved/invalidated stage
lists and external refs as lists. Fixed by normalizing all tuple fields.

### Findings fixed and remaining risks

Fixed: durable replay epoch and worker-side ResumeRequest consumption are
covered locally.

Risk: the complete production stage graph was not run because the baseline
requires unavailable competition data/dependencies.

## Ticket 6 — Process crash recovery

### Root cause/design

Restart recovery uses ownership token plus heartbeat freshness. A fresh owned
worker is adopted; a live worker with unsafe ownership/freshness blocks a new
launch; dead workers without results are marked `INTERRUPTED`; prepared
promotions resolve to old or new, never a half-pointer.

### Files changed

- `src/kaggle_agent/supervisor/loop.py`
- `src/kaggle_agent/supervisor/worker.py`
- `tests/test_supervisor_loop.py`
- `tests/test_supervisor_recovery.py`
- `tests/test_supervisor_worker.py`

### Tests and results

```text
TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py tests/test_external_outbox.py
95 passed in 1.91s
```

### Code review findings

Required finding: a live but unadoptable worker did not prevent a replacement
launch. Fixed with `RECOVERY_BLOCKED`. Required finding: managed-generation
workers lacked an explicit generation `src/` import path. Fixed in the launcher.

### Findings fixed and remaining risks

Fixed: duplicate replacement launch is blocked and generation imports are
explicit.

Risk: a real kill-at-each-boundary subprocess campaign was not run in this
environment. Process-level crash certification remains incomplete.

## Ticket 7 — External exactly-once validation

### Root cause/design

The outbox remains the canonical identity ledger. Kernel push and submission
call sites now reconcile `prepared`, `sent`, and `unknown` intents before any
second send. Exact remote evidence changes an intent to `accepted`; uncertainty
remains pending.

### Files changed

- `src/kaggle_agent/orchestrator.py`
- `tests/test_external_outbox.py`

### Tests and results

```text
UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/test_external_outbox.py
20 passed in 0.12s
```

Eight matrix cases cover four boundaries for both kernel push and submission.
The matrix produces one logical send when no authoritative remote record
exists, and zero additional sends when the remote record exists after a crash.

### Code review findings

Required finding: only `sent`/`unknown` intents were reconciled; a crash after
remote send but before local persistence left `prepared` and could duplicate.
Fixed at both production call sites and with prepared-intent tests.

### Findings fixed and remaining risks

Fixed: duplicate prevention is proven against fake authoritative callbacks.

Risk: no live Kaggle action was performed. Exactly-once is logically tested,
not externally certified against Kaggle transport behavior.

## Ticket 8 — Telegram ownership E2E

### Root cause/design

When supervisor mode is enabled, Telegram `/run` enqueues a durable supervisor
command instead of launching a worker. `/pause` and `/resume` enqueue control
commands; `/status` reports the supervisor queue and control state.

### Files changed

- `tests/test_supervisor_commands.py`

### Tests and results

```text
UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests/test_supervisor_commands.py
4 passed in 0.10s
```

Covered: `/run`, duplicate `/run`, `/status`, `/pause`, `/resume`, and queue
survival across a new queue instance.

### Code review findings

No blocking findings in the controlled queue review.

### Findings fixed and remaining risks

Fixed: duplicate commands are durable and remain supervisor-owned.

Risk: Telegram transport, real bot polling, and live supervisor restart were
not exercised because credentials/network were unavailable.

## Ticket 9 — AUTO_SAFE canary readiness

### Minimal envelope defined

The intended canary envelope is documented and remains disabled:

```text
mode: auto_safe only in a disposable canary checkout
max repairs per cycle: 1
max attempts per incident: 2
max changed source files: 2
max changed test files: 1
max changed lines: 120
allow dependency changes: false
promotion: manual/disabled until the canary evidence is accepted
protected trust-base paths: unchanged and denied
```

### Tests and results

The acceptance, protected-path, no-op, test-integrity, and dirty-baseline
checks passed in the controlled stabilization slice. Production
`supervisor.enabled` remains `false`; no AUTO_SAFE run was started.

### Code review findings

No new production behavior was enabled for this ticket. The envelope is a
rollout decision, not a permission for unrestricted autonomy.

### Findings fixed and remaining risks

Fixed: canary limits and trust-base constraints are explicit in this report.

Risk: canary readiness is not approved until real DeepSeek roles, subprocess
crash recovery, and live-independent external reconciliation are validated.

## Final verification snapshot

After the source changes, the broad candidate suite was:

```text
uv run python -m compileall -q src: passed
uv run pytest -q -m "not integration": 565 passed, 38 failed, 1 deselected, 2 warnings in 24.50s
```

The 38 candidate failures matched the clean-main manifest exactly. The two
additional local validation tests added afterward are isolated supervisor
tests and passed; no source behavior changed afterward.

## Readiness decisions

| Mode | Decision | Reason |
|---|---|---|
| OBSERVE | CONDITIONAL READY | Local incident/classification/spec flow passes, but real worker-cycle and DeepSeek smoke are blocked by missing credentials/data. |
| REPAIR_ONLY | CONDITIONAL READY | Disposable candidate lifecycle and safety gates pass locally; real DeepSeek implementer/reviewer and process crash campaign remain outstanding. |
| AUTO_SAFE_CANARY | NOT READY | The envelope is defined, but the required real-role, process-crash, and external-action evidence is incomplete. |
| unrestricted AUTO_SAFE | NOT READY / DISABLED | Baseline has 38 failures, real provider roles are unvalidated, and live external/process E2E is incomplete. |

The next milestone remains `AUTO_SAFE_CANARY ready`. It must include one
intentionally introduced low-risk defect repaired, reviewed, promoted, and
resumed without human intervention before the canary decision changes.
