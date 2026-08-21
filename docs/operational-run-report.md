# Kaggle-Agent Operational Run Report

## Scope and baseline

- Repository: `kaggle-agent`
- Branch: `operational-run`
- Starting `origin/main`: `34464cdbc10af329bea1a845f580d45390ada47c`
- Final commit: `24fd8e12ad049d5b06da0c948a93fbb3269092a0`
- Competition: `rsna-knee-abnormality-detection` (`rsna_knee`)
- Runtime state: disposable `/tmp/kaggle-agent-operational-run-20260821`
- Kaggle credentials: available; values were not logged
- DeepSeek credentials: available through the existing dotenv path; values were not logged
- Telegram credentials: available; no messages were sent

Baseline and final repository verification:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     693 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py tests/test_external_outbox.py
                                                            168 passed
git diff --check                                          PASS
```

## Real OBSERVE cycles

Two real supervisor-owned worker cycles were run against Kaggle read APIs and
the configured DeepSeek-backed research/planning path. Both used
`orchestrator.dry_run: true`; no submission or kernel mutation path was
invoked.

| Cycle | Worker | Request cycle ID | Result | Incident | External mutation |
|---|---|---|---|---|---|
| 1 | `worker-0c3bbdb145a0` | missing in the pre-fix request | `RECOVERABLE_FAILURE` | `4e79d7cd24031bb46dd7` | 0 |
| 2 | `worker-bf2fa7253578` | `cycle-ce19549e0270` | `RECOVERABLE_FAILURE` | same stable signature | 0 |

Each cycle completed:

- Kaggle competition snapshot, metadata-file, leaderboard, kernel-listing,
  submission-history, and limits reads;
- browser/research fleet with verified card writes;
- DeepSeek-backed research planning;
- three planned slices, each reaching CODE and stopping when the coding agent
  produced no verified recipe change.

The worker correctly failed closed. It did not classify the agent stall as a
source `CODE_DEFECT`, did not create a repair worktree, and did not promote or
retry any external action. The durable classification was `UNKNOWN`, with
`MEDIUM` provisional risk and authority required. The normalized failure was
the same on both cycles, so the existing signature-keyed incident store
retained one incident record while risk metrics recorded both cycles' decision
transitions.

Operational risk metrics recorded:

```text
failure classes: UNKNOWN = 4 decisions
risk tiers: MEDIUM = 4 decisions
transitions: NONE -> MEDIUM = 2; MEDIUM -> MEDIUM = 2
false CODE_DEFECT classifications: 0
repair attempts: 0
promotions: 0
rollbacks: 0
external-state blocks: 0
duplicate workers/promotions: 0
```

No natural deterministic implementation defect occurred, so REPAIR_ONLY was
not invoked against the real competition. No failure was manufactured merely
to force a repair.

## Production-shaped harness

The existing production-shaped harness was run against the committed branch
using a disposable synthetic generation and real DeepSeek repair roles. It
passed:

- real initial and replacement worker subprocesses;
- one incident, accepted repair, candidate commit, new generation, and
  promotion;
- `PREPARED -> PROMOTED -> RESUMED`;
- stage calls `RESEARCH=1`, `PLAN=1`, `CODE=2`;
- restart-after-promotion and unhealthy-replacement rollback probes;
- zero Kaggle mutations and zero Telegram messages.

The harness script was used as a temporary validation artifact and was not
added to this operational branch because it is not present in current
`origin/main`.

## Genuine production fix

The first OBSERVE worker request had `cycle_id: null`. The orchestrator later
created its own experiment ID, but the supervisor-owned request, heartbeat,
result, and incident envelope lacked durable cycle correlation. This weakens
audit and resume diagnostics.

The minimal fix assigns `cycle-<token>` to initial workers and preserves the
existing `ResumeRequest.cycle_id` for replacement workers. External-action
keys and stage idempotency keys were not changed. A regression test verifies
the initial request identity.

Focused test for the fix: `6 passed` in `tests/test_supervisor_loop.py`.

The repeated CODE-agent stall was not changed: it produced no actionable
source traceback or deterministic reproduction and was classified `UNKNOWN`,
so the supervisor correctly refused to invent a source repair.

## External integrations

Kaggle reads and reconciliation preflight passed for competition metadata,
metadata files, submissions, kernels, and submission limits. All operational
cycles used dry-run configuration.

```text
Kaggle mutations: 0
Kaggle submissions: 0
Telegram messages: 0
```

## Defaults and safety

Checked-in defaults remain unchanged and safe:

```text
default_competition: null
supervisor.enabled: false
supervisor.promotion.automatic: false
supervisor.auto_safe.enabled: false
orchestrator.dry_run: true
submit.dry_run_default: true
```

Protected paths, external reconciliation, review gates, and risk policy were
not weakened.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY (synthetic certification; no natural real CODE_DEFECT occurred)
AUTO_SAFE_CANARY: READY (synthetic certification)
FULL_AGENT_HARNESS: READY (synthetic production-shaped harness passed)
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains conditional because this operational run did
not produce a natural real source defect eligible for autonomous repair, and
the real competition path stopped at the configured CODE-agent no-change
guard. Controlled synthetic certification remains green; unrestricted
autonomy remains disabled.

## Remaining risks

- The configured competition CODE agent repeatedly stalled without writing a
  recipe. This is an agent/task-quality issue requiring a separate actionable
  reproduction before changing production behavior.
- The operational cycle did not reach Kaggle kernel training, validation, or
  feedback because CODE produced no verified recipe.
- Live external mutation exactly-once behavior remains untested by policy;
  only read-side reconciliation and synthetic safety paths were exercised.
