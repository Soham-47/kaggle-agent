# Production Self-Repair Certification & AUTO_SAFE Canary

## Baseline

- Starting merged-main SHA: `89cf64b95fb6c232f856cf94cdbcfec0ec737faf`
- Branch: `supervisor-autosafe-canary`
- Worktree: `/home/soham/kaggle-agent-autosafe-canary`
- Starting working tree: clean
- Starting baseline: `609 passed, 1 deselected`
- `AUTO_SAFE` default: disabled

The cleanup branch was confirmed merged into `origin/main` before this worktree
was created. The prior cleanup worktree was not modified.

## Ticket 1 — Baseline gate

### Existing behavior

Merged main already contained the genericized framework and supervisor
stabilization work.

### Result

Passed before changes:

```text
uv run python -m compileall -q src examples       PASS
uv run pytest -q -m "not integration"            609 passed, 1 deselected
git diff --check                                 PASS
```

## Ticket 2 — Production DeepSeek role wiring

### Root cause / goal

The existing production role path reached DeepSeek, but its prompts did not
constrain enum values, several artifact fields were not strictly type-checked,
provider errors were not retried in a bounded way, and implementer action
envelopes were underspecified.

### Implementation

The smallest existing-wiring fix added:

- bounded fresh-session retries for malformed/provider responses;
- explicit classifier enum and JSON-schema prompts;
- strict RepairSpec field, list, reproduction-mode, path, and budget checks;
- explicit spec-review and code-review verdict contracts;
- explicit implementer action/tool envelope and tool-result protocol;
- a safe `file_path` compatibility alias at the existing repair boundary;
- a disposable smoke defect with a real failing/passing focused test.

### Focused tests

```text
uv run pytest -q tests/test_supervisor_agents.py
18 passed
```

### Real provider result

DeepSeek credentials were available through the preserved repository `.env`.
No key or authorization value was printed.

The real classifier, RepairSpec author, and spec reviewer produced typed
results. The implementer/reviewer sequence did not complete successfully:

```text
classifier: CODE_DEFECT                         PASS
spec author: typed bounded spec                  PASS
spec reviewer: APPROVE                           PASS
implementer: bounded but no accepted candidate  FAIL
code reviewer: REJECT                            PASS (fail-closed)
```

The real smoke therefore remains a failed certification, not a passed role
smoke. The provider was reached; no successful five-role repair certification
is claimed.

## Ticket 3 — Real REPAIR_ONLY lifecycle

`REPAIR_ONLY` was not certified. The precondition failed because the real
implementer did not produce a focused-test-passing candidate and the
independent reviewer rejected it. No candidate was promoted and no active
generation was changed by the failed smoke runs.

## Ticket 4 — Provider failure behavior

Added a local regression test for provider failure:

```text
provider error → one fresh retry → typed failure → fail closed
```

This behavior passed in `tests/test_supervisor_agents.py`. No provider outage
was allowed to trigger source mutation.

## Ticket 5 — Provider-independent supervisor validation

The existing supervisor, replay, outbox, and CLI validation suites passed:

```text
uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py \
  tests/test_external_outbox.py tests/test_cli_init.py
107 passed
```

The full non-integration suite after the local wiring changes passed:

```text
uv run python -m compileall -q src examples
uv run pytest -q -m "not integration"
618 passed, 1 deselected
```

## Kaggle and Telegram external state

- DeepSeek credentials: available.
- Telegram credentials: available; no live command or message was sent in this phase.
- Kaggle credentials: available.
- Read-only Kaggle authentication and competition metadata lookup: passed.
- Kaggle mutations performed: `0`.

The read-only check used the existing `KaggleClient`; no kernel push or
submission was attempted.

## Security review

Reviewed new occurrences from the required searches. The only new subprocess
usage is inside the disposable local DeepSeek smoke fixture and existing
repair tooling. No new arbitrary shell, network, credential access, Git push,
approval bypass, or protected-path bypass was added.

Existing `subprocess`, `exec`, polling, and broad exception occurrences were
reviewed as pre-existing runtime behavior or policy scan strings; they were
not broadened by this phase.

## Code review findings

### Findings fixed

- Classifier could receive a Python exception name instead of a valid enum.
  Fixed with explicit enum prompt and strict parsing.
- Provider and malformed-output failures had no bounded retry. Fixed with two
  fresh-session attempts and fail-closed protocol errors.
- RepairSpec lists, reproduction modes, budgets, and wildcard paths were not
  fully validated. Fixed with strict field validation.
- Implementer action envelopes were ambiguous. Fixed with explicit `tool` or
  `done` protocol and bounded invalid-action recovery.
- Model file path spelling differed from the repair boundary. Fixed with a
  path-scoped alias without widening permissions.
- The original smoke fixture had no inferable correct value. Replaced with a
  deterministic `sum(valuez)` test fixture and real focused-test execution.

### Remaining blocking finding

The real DeepSeek implementer did not reliably produce a candidate that passed
the disposable focused test. The independent reviewer rejected the resulting
candidate. This blocks REPAIR_ONLY and AUTO_SAFE_CANARY certification.

## Canary status

The unattended AUTO_SAFE canary was not started. Its preconditions were not
met, so no source was manually repaired, no generation was promoted, and no
Kaggle or Telegram mutation occurred.

Required unverified items remain:

- successful real REPAIR_ONLY candidate lifecycle;
- successful unattended AUTO_SAFE promotion and ResumeRequest execution;
- exact stage-call-count proof for the promoted canary;
- post-acceptance supervisor crash recovery during the canary;
- repeat-failure exhaustion in the real provider path.

## Readiness

```text
OBSERVE: CONDITIONAL

REPAIR_ONLY: NOT READY

AUTO_SAFE_CANARY: NOT READY

UNRESTRICTED_AUTO_SAFE: NOT READY
```

OBSERVE is conditional on using deterministic classifications or treating
DeepSeek provider/typed-output failures as `NEEDS_AUTHORITY`. REPAIR_ONLY and
AUTO_SAFE_CANARY must remain disabled for real autonomous repairs until the
implementer produces a passing isolated candidate and the reviewer approves
it in the complete end-to-end flow.
