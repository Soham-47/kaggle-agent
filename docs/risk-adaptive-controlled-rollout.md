# Risk-Adaptive AUTO_SAFE Controlled Rollout Certification

## Baseline

- Starting merged `origin/main`: `973486c90aff2ca7e2ce811cec83a818f3dcf813`
- Branch: `supervisor-risk-adaptive-controlled-rollout`
- Original dirty developer checkout: preserved and untouched.
- Baseline before changes: `678 passed, 1 deselected`.
- Final suite: `692 passed, 1 deselected`.
- AUTO_SAFE checked-in default: disabled.

Commands:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                    692 passed, 1 deselected
git diff --check                                          PASS
```

## Validation changes

The risk tier set and policy architecture were not expanded. The validation
defects fixed were:

- Same-repair risk floors are monotonic. A HIGH or PROHIBITED provisional
  decision cannot become a lower tier during spec, diff, or review evaluation.
- Risk decisions preserve `risk_floor` through JSON serialization.
- Durable metrics now record `NONE->tier` and phase-to-phase transitions,
  escalation counts, de-escalation attempts, and escalation reasons.
- OBSERVE persists a pre-spec decision for every classified incident,
  including transient and non-repairable failures. When no spec is authored it
  persists a hypothetical `risk-post-spec.json` decision instead.
- The actual supervisor and repair coordinator pass the previous tier into
  later evaluations, so a narrow candidate diff cannot regain lower promotion
  authority after a sensitive earlier decision.

No generation, replay, outbox, promotion, rollback, or LLM architecture was
replaced.

## Risk transition evidence

The deterministic tests cover:

- LOW deterministic parser repair: candidate and automatic promotion allowed.
- MEDIUM four-file, 400-line adapter-style repair: candidate and automatic
  promotion allowed with the stronger MEDIUM envelope.
- HIGH replay/resume repair: candidate allowed, automatic promotion denied.
- PROHIBITED outbox/external-identity repair: candidate generation denied.
- LOW local defect with AMBIGUOUS external state: risk remains local in tier,
  but candidate generation and promotion are denied by the hard external-state
  barrier.

Transition metrics include `NONE->LOW`, `LOW->MEDIUM`, escalation counters,
and bounded reasons. A candidate unexpectedly touching `outbox.py` escalates
from a LOW provisional decision to PROHIBITED; the one-line diff does not make
the trust-base change safe.

## Controlled canaries

The provider-independent canary matrix passed in
`tests/test_risk_adaptive_controlled_rollout.py` (`8 passed`). The PROHIBITED
case also exercises `RepairCoordinator` and verifies that the implementer
callback is never invoked and no candidate is created.

The prior real DeepSeek synthetic AUTO_SAFE canary remains the basis for
`AUTO_SAFE_CANARY: READY`. With the replacement credential loaded only in the
validation subprocess, the production role smoke passed for classifier,
spec author, spec reviewer, implementer, and code reviewer. The result was:

```text
classifier=CODE_DEFECT
spec=smoke-repair
spec_review=APPROVE
implementer=verified candidate is ready
code_review=APPROVE
```

The real disposable `REPAIR_ONLY` coordinator path also passed against a
repository-shaped synthetic project: classification and spec review approved,
the implementer produced a one-file patch on its first bounded attempt,
focused verification passed, the independent code review approved, and a
candidate commit was persisted. The active generation remained unchanged.
The initial fixture attempts failed closed on generated `uv.lock`/bytecode,
missing `pythonpath`, and missing package metadata; these were corrected in
the disposable fixture only and did not require production permission changes.

## Process-kill and rollback validation

Actual subprocess termination tests passed at both durable promotion boundaries:

| Boundary | Kill/restart result |
| --- | --- |
| `PREPARED`, old pointer | Recovery rolled back to exactly the old generation |
| `PREPARED`, new pointer already switched | Recovery retained exactly the new generation |

The existing process-level worker kill/adoption tests also passed. Existing
rollback coverage passed for startup/import failure of a resumed generation:
the replacement is marked rolled back, the old generation is restored, and the
candidate is not blindly retried.

Focused recovery/acceptance/replay/outbox/Telegram tests were included in:

```text
uv run pytest -q tests/test_supervisor*.py tests/test_external_outbox.py \
  tests/test_replay_epoch.py tests/test_telegram_commands.py \
  tests/test_telegram_submit.py tests/test_controlled_rollout_profile.py
190 passed
```

## OBSERVE and REPAIR_ONLY operational validation

The clean merged repository has no default competition selected, and this
worktree has no DeepSeek credential. Therefore no real worker cycle was
started merely to manufacture an incident. Synthetic OBSERVE tests exercised
deterministic classification, pre-spec/post-spec persistence, non-repairable
incident handling, and risk metrics. REPAIR_ONLY tests exercised candidate
creation, verification, independent review, and the unchanged active
generation.

DeepSeek credential: **AVAILABLE** through the normal local dotenv path; the
value was not printed or persisted by validation.
Telegram credentials: **UNAVAILABLE**.

Live Telegram ownership tests were not run. This is a provider/environment
limitation, not a simulated pass.

## Kaggle read-side validation

Authenticated read-only calls through the existing `KaggleClient` passed for
the configured optional competition:

- authentication: PASS;
- competition metadata: PASS;
- kernel history: PASS;
- submission history: PASS;
- Kaggle mutations: **0**.

Live exactly-once mutation behavior is **NOT TESTED**. No competition
submission was created for this certification.

## Security and scope review

- No protected paths were relaxed.
- No user configuration can make HIGH or PROHIBITED automatic.
- Unresolved external state still blocks autonomous candidate generation and
  promotion.
- No dependencies were added.
- No secrets, authorization headers, or environment values were persisted.
- No Telegram messages were sent.
- No Kaggle mutation was performed.

## Test and code-review results

Focused risk/repair/validation tests: `35 passed`.
Real DeepSeek role smoke: `PASS` (all five roles, separate sessions).
Real disposable REPAIR_ONLY coordinator: `PASS` (candidate persisted, no promotion).
Controlled canary and subprocess tests: `8 passed`.
Supervisor/recovery/replay/outbox/Telegram suite: `190 passed`.
Full non-integration suite: `692 passed, 1 deselected`.
Compile and `git diff --check`: passed.

Review findings fixed during this phase:

1. Risk could de-escalate between phases of one repair. Fixed with a typed
   monotonic risk floor and production-flow propagation.
2. `risk_floor` was initially omitted from deserialization. Fixed with a
   round-trip regression test.
3. OBSERVE did not persist hypothetical post-spec risk for non-repairable or
   provider-blocked incidents. Fixed with durable status-bearing artifacts.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains CONDITIONAL because live Telegram ownership
and real configured worker shadow cycles were not run in this validation
worktree. DeepSeek provider execution is no longer the blocker: all five role
smoke sessions and the disposable real REPAIR_ONLY coordinator path passed.
The deterministic policy, canaries, process recovery, rollback, and read-only
Kaggle boundary are validated. Controlled rollout may proceed after the
Telegram credential and a real competition configuration are supplied through
the normal local paths.
