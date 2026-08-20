# Self-healing supervisor ticket reports

This report covers the implemented foundation slices. Tickets that require
live DeepSeek repair sessions, full generation promotion, Telegram/cron
routing, or fault injection remain explicitly incomplete; they are not marked
complete by the implementation plan.

## Ticket 1 — Supervisor package

### Existing behavior
No supervisor package existed; repair was inline in the orchestrator.

### Problem
There was no process boundary or durable supervisor contract.

### Implementation
Added `kaggle_agent.supervisor` modules for state, lock, generations, worker
protocol, heartbeat, incidents, classification, specs, policy, worktrees,
verification, review, promotion, rollback, resume, audit, budgets, and CLI
coordination.

### Files changed
`src/kaggle_agent/supervisor/`, focused supervisor tests, and CLI/config seams.

### Tests added / results
38 supervisor tests pass; source compilation passes.

### Code review findings / fixed
Fixed a first-pass auto-safe generation bug so clean code is imported into a
detached managed worktree instead of running from the developer checkout.

### Invariants preserved
Existing `RunLock`, stage ledger, outbox identities, and kernel repair policy
remain in use.

### Remaining risks
The full repair coordinator is not yet wired to live DeepSeek sessions.

## Ticket 2 — Supervisor configuration

### Existing behavior
The YAML loader validated existing settings only.

### Problem
Supervisor modes, heartbeat limits, repair budgets, and promotion policy were
not configurable.

### Implementation
Added strict validation and typed `SupervisorSettings`/`RepairSettings`, with
safe disabled/observe defaults in `config/settings.yaml`.

### Tests added / results
Valid, invalid mode/confidence/heartbeat/boolean/budget cases pass.

### Invariants preserved
The existing YAML loader and error style are reused.

## Tickets 3–4 — State root and lock

### Implementation
Added `KAGGLE_AGENT_SUPERVISOR_DIR`, atomic JSON state layout, separate
ownership-safe `supervisor.lock`, stale-owner takeover, and shared-state path
seams.

### Tests / review
Exclusive ownership, stale recovery, non-owner release, atomic writes, and
legacy lock/outbox compatibility pass.

## Tickets 5–8 — Baseline, revisions, generations, shared state

### Implementation
AUTO_SAFE rejects dirty source baselines. `RuntimeRevision`,
`RuntimeGeneration`, detached managed generation import, and explicit
`RuntimeLayout` were added. Stage ledger, outbox, memory, run lock, and debug
audit can resolve through `KAGGLE_AGENT_STATE_ROOT`.

### Tests / review
Managed detached worktree and replay epoch tests pass. The current checkout is
dirty, so AUTO_SAFE is intentionally not enabled.

## Tickets 9–13 — Worker lifecycle

### Implementation
Added typed worker request/result states, separate worker process launcher,
atomic request/result persistence, independent heartbeat thread, stale heartbeat
termination, and supervisor mode boundary in the orchestrator.

### Tests / review
Protocol, heartbeat, launcher, and inline-debug suppression tests pass.

### Remaining risks
Resume execution and full supervisor restart adoption are not yet wired.

## Tickets 14–25 — Incidents, classification, and specs

### Implementation
Added redacted structured incidents, stable signatures, deterministic failure
classes, known `KernelPushRepair` precedence, reconciliation-before-classification
helper, persisted `RepairSpec` JSON/Markdown, and fresh-session structured-agent
contracts.

### Tests / review
Credential redaction, signature stability, external pending behavior,
classification precedence, spec round-trip, and malformed-agent output tests
pass.

### Remaining risks
The DeepSeek classifier/spec/reviewer calls are represented by explicit
callback/session boundaries but are not yet connected to production orchestration.

## Tickets 26–42 — Repair safety foundations

### Implementation
Added exact-base worktree manager, diff limits, protected paths, static and
test-integrity scans, restricted repair toolbox, independent review result
contract, deterministic verification allowlist, acceptance object, durable
repair budgets, and loop detection.

### Tests / review
Worktree commit/destroy, unsafe construct rejection, prompt-injection rejection,
test weakening, budget exhaustion, verification allowlisting, and acceptance
gates pass.

### Remaining risks
Candidate revision loops and supervisor-owned implementer/reviewer execution
are not yet connected end-to-end.

## Tickets 43–53 — Impact, promotion, rollback, resume

### Implementation
Added conservative stage impact mapping, replay epochs, `ResumeRequest`,
atomic active-generation pointer, acceptance-gated promotion, and rollback
helper.

### Tests / review
Impact, epoch invalidation, atomic pointer, and rollback tests pass.

### Remaining risks
The worker does not yet consume `ResumeRequest` to replay only preserved stages;
startup health checks and automatic rollback on failed resume remain to implement.

## Tickets 54–59 — Recovery and integrations

### Implementation
Added append-only audit recovery, ownership metadata inspection, CLI
`supervisor` command, ordinary CLI/Telegram routing, and cron routing through
the supervisor when enabled.

### Remaining risks
Telegram status/pause/resume are not yet supervisor queue commands, and
restart adoption does not yet resume a live repair transaction automatically.

## Tickets 60–72 — Fault injection

### Implementation
Added a disabled-by-default fault injection seam and basic recovery tests.

### Remaining risks
The complete required external-action crash matrix is not yet integrated with
live Kaggle fakes.

## Tickets 73–75 — Rollout modes

### Status
Not complete. Observe and disabled-mode command paths exist; repair-only
candidate storage and the full AUTO_SAFE readiness suite still require
implementation. The implementation deliberately reports AUTO_SAFE as not
ready.

## Stabilization Ticket 1 — Production DeepSeek role wiring

### Root cause / design

The foundation exposed only a callback that accepted any JSON object. It did
not select the configured DeepSeek backend, enforce role-specific schemas, or
provide a real implementer tool boundary. The stabilization change adds
`DeepSeekSupervisorAgents`, which resolves the existing `ZenClient` provider
from `DEEPSEEK_API_KEY` and creates a separate explicit session for classifier,
spec author, spec reviewer, implementer, and code reviewer calls. Only
persisted incident/classification/spec/diff/verification artifacts cross role
boundaries.

### Files changed

* `src/kaggle_agent/supervisor/agents.py`
* `src/kaggle_agent/supervisor/spec.py`
* `src/kaggle_agent/supervisor/repair_agent.py`
* `tests/test_supervisor_agents.py`

### Tests and results

* Focused: `uv run pytest -q tests/test_supervisor_agents.py --tb=short` —
  `7 passed`.
* Affected subsystem: `uv run pytest -q tests/test_supervisor_*.py tests/test_replay_epoch.py` —
  `46 passed`.

### Code review findings

The first review found that the existing boundary exposed a nominal
`apply_patch` tool but rejected it without a production session path, and that
the old generic helper did not enforce role schemas. The ticket implementation
keeps arbitrary shell unavailable, validates all production role fields and
limits, and routes implementer actions through the restricted boundary.

### Findings fixed

Fixed missing strict enum/type/unknown-field checks, missing provider/session
wiring, missing base identity injection into authored specs, and shell-tool
handling in the implementer loop. No protected path, dependency, or
AUTO_SAFE policy was changed.

### Invariants preserved

DeepSeek receives no environment or secret artifact; model output cannot alter
acceptance policy; each role has a fresh session; repair writes remain scoped to
the repair workspace; malformed output is a protocol failure.

### Remaining risks

The live role facade is not yet the supervisor loop's end-to-end coordinator;
worker resume and bounded repair lifecycle wiring are the next tickets.

## Stabilization Ticket 2 — Worker ResumeRequest and replay epochs

### Root cause / design

The worker protocol carried only a stage name and the worker discarded the
resume artifact. `run_daily` also generated a fresh cycle ID on every process
start, preventing the existing durable stage ledger from finding successful
outputs. The protocol now serializes `ResumeRequest` with per-stage epochs, the
worker passes it into `run_daily`, and the orchestrator preserves the cycle ID.
`StageInput` continues to use the explicit replay epoch rather than embedding a
Git SHA in every identity.

### Files changed

* `src/kaggle_agent/supervisor/resume.py`
* `src/kaggle_agent/supervisor/protocol.py`
* `src/kaggle_agent/supervisor/worker.py`
* `src/kaggle_agent/supervisor/repair_flow.py`
* `src/kaggle_agent/orchestrator.py`
* `tests/test_supervisor_worker.py`

### Tests and results

* Focused: `uv run pytest -q tests/test_supervisor_worker.py tests/test_supervisor_repair_flow.py tests/test_replay_epoch.py --tb=short` — `5 passed`.
* Affected supervisor suite after ticket 1 and 2 — `46 passed`.

### Code review findings

Review confirmed that preserved stages retain epoch 0 and the failed stage and
downstream stages receive epoch 1. The cycle ID is reused, and legacy worker
requests without a resume artifact still deserialize with `None`.

### Findings fixed

Fixed the dropped resume payload, fresh-cycle replay miss, and missing durable
epoch map in repair-flow output.

### Invariants preserved

Stage output allowlists and `StageLedger` remain the replay authority; external
action identities are unchanged; no Git SHA was added to existing stage keys;
uncertain external actions remain outside source-repair decisions.

### Remaining risks

The supervisor loop does not yet launch a resumed worker from an accepted
repair or handle repair-only candidates end-to-end.

## Stabilization Ticket 3 — repair_only candidate lifecycle

### Root cause / design

`RepairCoordinator` always materialized a runtime generation after acceptance
and removed the worktree. That made `repair_only` unsafe for human inspection
and coupled candidate creation to activation. The coordinator now accepts an
explicit mode. In `repair_only`, it commits through the supervisor-owned
worktree manager, writes the candidate revision/diff/review under
`accepted/<repair-id>.json`, retains the candidate worktree, and never writes
`active-generation.json`.

### Files changed

* `src/kaggle_agent/supervisor/repair_flow.py`
* `tests/test_supervisor_repair_flow.py`

### Tests and results

* Focused: `uv run pytest -q tests/test_supervisor_repair_flow.py --tb=short` —
  `2 passed`.
* Existing accepted-generation flow remains covered by the same test module.

### Code review findings

Review found that candidate metadata needed JSON-safe review serialization and
that unrecognized modes could accidentally take the promotion path. Both were
fixed before completion.

### Findings fixed

Added `repair_only` mode validation, durable candidate metadata, candidate
revision/path reporting, and JSON-safe review persistence.

### Invariants preserved

Verification, protected-path/static/test-integrity gates, review, budgets, and
supervisor-owned commits run before a candidate is accepted. No active runtime
pointer or developer branch is modified.

### Remaining risks

The supervisor loop still needs to route its configured mode into this
coordinator and recover interrupted candidate/promotion transactions.

## Stabilization Ticket 4 — Restart adoption and promotion recovery

### Root cause / design

Restart inspection existed but was not connected to startup, ownership tokens
were replaced during stale-lock takeover, and promotion had no durable phase
record. The lock now preserves a stale owner token during safe takeover.
`Supervisor.recover_startup()` inspects workers before launch and resolves a
prepared promotion using the existing atomic generation pointer; recovery can
only adopt, interrupt, commit-new, or roll back-old.

### Files changed

* `src/kaggle_agent/supervisor/lock.py`
* `src/kaggle_agent/supervisor/recovery.py`
* `src/kaggle_agent/supervisor/promote.py`
* `src/kaggle_agent/supervisor/loop.py`
* `tests/test_supervisor_recovery.py`

### Tests and results

* Focused: `uv run pytest -q tests/test_supervisor_recovery.py tests/test_supervisor_foundation.py tests/test_supervisor_loop.py --tb=short` — `14 passed`.

### Code review findings

Review required owner-token continuity across restart and rejected any
recovery path that could leave an indeterminate active-generation pointer.
Prepared promotions now record old/new IDs and recovery rejects unknown pointer
values.

### Findings fixed

Connected startup recovery, preserved ownership tokens on lock takeover,
persisted promotion PREPARED/COMMITTED states, and added deterministic
interrupted-promotion tests.

### Invariants preserved

No second worker starts while an owned heartbeat is fresh; worker RunLock is
untouched; active generation is written with atomic JSON replacement; old/new
generation choice is explicit.

### Remaining risks

External-action crash reconciliation and supervisor-owned Telegram command
routing are still incomplete.

## Stabilization Ticket 5 — Kernel-push and submission crash matrix

### Root cause / design

The outbox already recorded intent, but accepted actions were not deduplicated
on a later enqueue and orchestrator branches could treat an accepted action as
sendable. Accepted actions now remain the terminal identity for the same
logical key; kernel and submission paths reconcile `sent`/`unknown` first and
short-circuit `accepted` without invoking the mutation API.

### Files changed

* `src/kaggle_agent/autonomy/outbox.py`
* `src/kaggle_agent/orchestrator.py`
* `tests/test_external_outbox.py`

### Tests and results

* Crash matrix: `uv run pytest -q tests/test_external_outbox.py --tb=short` —
  `11 passed`.
* The existing direct submission-outbox test remains a baseline pipeline
  failure on clean `origin/main` because its earlier evaluation gate exits
  before the outbox assertion; it is not a supervisor regression.

### Code review findings

Review specifically tested crash before local result persistence, crash after
remote acceptance, repeated enqueue, exact kernel evidence, exact submission
marker evidence, and ambiguous evidence. The accepted-action re-enqueue bug was
blocking and was fixed.

### Findings fixed

Accepted actions are terminally deduplicated and orchestrator mutation branches
now handle accepted/sent/unknown distinctly.

### Invariants preserved

Existing `kernel_push_key`, `submission_key`, markers, outbox reconciliation,
and “timeout is not proof of no mutation” semantics remain authoritative.

### Remaining risks

Live Kaggle API crash injection was not run; the deterministic fake matrix is
the available non-integration proof.

## Stabilization Ticket 6 — Telegram supervisor ownership

### Root cause / design

Telegram and dashboard command handling could directly mutate `memory/state.md`
or request a worker start. A durable `SupervisorCommandQueue` now receives run,
pause, and resume requests when supervisor mode is enabled. Supervisor startup
drains the queue and owns pause state before worker launch; status is read-only
and includes queue/control ownership.

### Files changed

* `src/kaggle_agent/supervisor/commands.py`
* `src/kaggle_agent/supervisor/loop.py`
* `src/kaggle_agent/notify/commands.py`
* `tests/test_supervisor_commands.py`

### Tests and results

* Focused and affected command tests: `uv run pytest -q tests/test_supervisor_commands.py tests/test_telegram_commands.py tests/test_supervisor_loop.py --tb=short` — `17 passed`.

### Code review findings

Review confirmed command callers no longer receive `start_cycle=True` in
enabled supervisor mode, command records are durable, and malformed control
state fails closed as paused.

### Findings fixed

Added queue persistence/cursoring, supervisor-owned pause control, run routing,
and status visibility.

### Invariants preserved

Telegram cannot bypass the supervisor lock or launch an ordinary worker;
submission approval semantics remain unchanged; disabled mode retains existing
command behavior.

### Remaining risks

The long-running Telegram polling process still needs deployment-level
supervisor polling/notification integration; command ownership itself is now
supervisor-only.

## Stabilization Ticket 7 — Fault-injection suite

### Root cause / design

The fault injector had declared points but only two smoke tests. The suite now
exercises every declared point, including stage entry, outbox preparation,
external send, hangs, worker kill, partial JSONL, repair/review rejection, and
promotion interruption. External faults use the real outbox/reconciliation
implementation and assert accepted logical keys cannot be re-enqueued as new
mutations.

### Files changed

* `tests/test_supervisor_faults.py`

### Tests and results

* Focused: `uv run pytest -q tests/test_supervisor_faults.py --tb=short` —
  `13 passed`.

### Code review findings

Review confirmed fault injection is disabled by default and that tests cannot
turn it on through production configuration. External and promotion recovery
tests use durable state, not mocked assertions alone.

### Findings fixed

Added complete fault-point coverage and duplicate-mutation assertions for both
kernel and submission action classes.

### Invariants preserved

Fault injection remains test-only; no production default, dependency, policy,
approval, or protected path was broadened.

### Remaining risks

Live Kaggle/GPU/Telegram process faults remain unavailable in this environment;
the deterministic non-integration suite is the validated boundary.

## Stabilization Ticket 8 — Supervisor lifecycle wiring and final regression review

### Root cause / design

The supervisor process could launch a worker but did not consume a recoverable
incident. It now loads the durable incident, reconciles unresolved external
actions before classification, invokes the production role facade for ambiguous
classification/spec/review, persists those artifacts, and routes `repair_only`
through the bounded coordinator. OBSERVE stops at `SPEC_READY`; automatic
activation remains refused during stabilization.

### Files changed

* `src/kaggle_agent/supervisor/loop.py`
* `src/kaggle_agent/supervisor/main.py`
* `src/kaggle_agent/supervisor/agents.py`
* `src/kaggle_agent/autonomy/repair_tools.py`
* `tests/test_supervisor_agents.py`
* `tests/test_supervisor_budgets_tools.py`

### Tests and results

* Focused stabilization suite: `uv run pytest -q tests/test_supervisor_*.py tests/test_replay_epoch.py` — `63 passed`.
* Compile: `uv run python -m compileall src` — passed.
* Full non-integration suite: `543 passed, 38 failed, 1 deselected, 2 warnings`.

### Code review findings

The final review found and fixed the advertised-but-rejected `apply_patch`
tool. The implementation now validates unified patch paths against the repair
write prefixes and uses Git's check/apply path without exposing arbitrary shell
or Git-history commands. Full-suite comparison found no supervisor-only
failure; all 38 failures reproduce on clean `origin/main`.

### Findings fixed

Connected incident lifecycle role wiring, added strict implementer patch
application, and removed generated bytecode from the worktree after compile
verification.

### Invariants preserved

AUTO_SAFE remains disabled; protected paths and acceptance policy are unchanged;
no new dependency, push, merge, developer-branch mutation, or live external
mutation was performed.

### Remaining risks

OBSERVE/REPAIR_ONLY are structurally ready for local deterministic operation,
but live DeepSeek/Kaggle/Telegram process validation was not available. AUTO_SAFE
requires the remaining full fault-injection/live integration confidence and a
clean managed baseline.
