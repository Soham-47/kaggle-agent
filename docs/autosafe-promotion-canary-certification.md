# AUTO_SAFE Promotion & Resume Certification

## Baseline

- Starting merged-main SHA: `b11ddefd4fd40cc0daebd13dea3b6570f8133a68`
- Branch: `supervisor-autosafe-canary-current`
- Final local revision: `c29a513`
- The original dirty checkout was preserved outside this worktree.

Baseline verification before the change:

```text
uv run python -m compileall -q src examples scripts  PASS
uv run pytest -q -m "not integration"                 644 passed, 1 deselected
git diff --check                                       PASS
```

## Promotion implementation

The temporary guard was in `Supervisor._handle_worker_result`. It allowed
`observe` to stop at a reviewed specification and `repair_only` to retain an
accepted candidate, but returned `NEEDS_AUTHORITY` for every other mode. That
guard was intentionally conservative while promotion and resume recovery were
being stabilized.

The new path is still opt-in:

```text
mode != auto_safe                         → no automatic activation
auto_safe + promotion.automatic != true  → NEEDS_AUTHORITY
auto_safe + all deterministic gates       → health → promote → resume
```

The implementation reuses `RepairCoordinator`, `GenerationPromotion`,
`RuntimeGeneration`, `ResumeRequest`, `StageLedger`, and `ExternalActionOutbox`.
It does not add a second promotion or checkpoint system.

Promotion now records:

```text
PREPARED → active-generation atomic replace → PROMOTED
```

The accepted candidate must have a matching committed revision and a clean
immutable worktree. A read-only startup health check runs before the pointer is
changed. The replacement worker receives the original cycle ID, generation,
incident, and serialized `ResumeRequest`; its stage ledger and outbox use the
shared `KAGGLE_AGENT_STATE_ROOT`.

## Resume and recovery

The supervisor persists the resume request and replacement worker ID before
promotion. On restart it:

- adopts a fresh owned replacement worker;
- restarts the same replacement request if promotion completed but the worker
  result was not persisted;
- blocks rather than guesses if a process dies between replacement launch and
  durable PID recording;
- resolves an interrupted `PREPARED` transaction to exactly the old or new
  pointer;
- rolls back the new generation after fatal/interrupted replacement startup;
- marks successful replacement completion as `RESUMED`, allowing later cycles.

Replay epochs remain stage-specific. The Git SHA is not inserted into existing
external-action identities.

## Tests added and results

Focused promotion/resume/recovery tests cover:

- failed health check leaves the active pointer unchanged;
- successful promotion records `PROMOTED`;
- default `auto_safe` without `promotion.automatic` fails closed;
- accepted generation promotion starts one worker with the exact
  `ResumeRequest`;
- restart reuses the durable replacement worker ID and request;
- fatal replacement startup rolls back to the prior generation;
- existing interrupted-promotion recovery remains compatible.

Results:

```text
uv run pytest -q tests/test_supervisor*.py
116 passed

uv run pytest -q tests/test_external_outbox.py tests/test_replay_epoch.py tests/test_stage_runtime.py
27 passed

uv run python -m compileall -q src examples scripts
PASS

uv run pytest -q -m "not integration"
652 passed, 1 deselected

git diff --check
PASS
```

## Real-provider certification

The credential-safe dotenv preflight returned:

```text
DeepSeek: UNAVAILABLE
```

No DeepSeek session was faked, and no production-model canary was run.
Therefore the required real DeepSeek repair chain has not been certified in
this worktree.

## Canary incident

No unattended AUTO_SAFE canary was executed because the real provider was
unavailable. No synthetic defect was introduced into a managed generation.

## External effects

```text
Kaggle mutations: 0
Telegram messages: 0
```

No external mutation or Telegram command was used for validation.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: NOT READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`AUTO_SAFE_CANARY` remains blocked until the real DeepSeek implementer,
reviewer, deterministic verification, promotion, replacement-worker launch,
and checkpoint-resume chain completes unattended in a disposable state root.
Checked-in defaults remain safe: supervisor disabled and automatic promotion
disabled.
