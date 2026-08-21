# Kaggle-Agent Operational Run Report

## Scope and baseline

- Repository: `kaggle-agent`
- Branch: `operational-run`
- Starting `origin/main` after reconciliation: `83e8ce12079ae1daddf6868efa5998398d93596f`
- Implementation baseline: `83e8ce1` (current merged main)
- Competition: `rsna-knee-abnormality-detection` (`rsna_knee`)
- Runtime state: disposable `/tmp/kaggle-agent-operational-run-20260821`
- Kaggle credentials: available; values were not logged
- DeepSeek credentials: available through the existing dotenv path; values were not logged
- Telegram credentials: available; no messages were sent

Baseline and final repository verification:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     696 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py tests/test_external_outbox.py
                                                            233 passed
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

The production-shaped harness is present in current `origin/main` and was run
without modifications from this branch.

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

The earlier CODE-agent stall is now fixed as documented below. It remains
outside source-defect classification when it produces no actionable traceback
or deterministic reproduction.

## Branch reconciliation

The earlier local `24fd8e1 fix: persist initial supervisor cycle identity`
overlapped the fix already merged in current `origin/main` through the full
system certification merge. Its source and regression test are already present
on `83e8ce1`; rebasing with
`git rebase --onto origin/main 24fd8e1 operational-run` removed the duplicate
code commit and preserved the operational report. The branch adds no duplicate
cycle-ID implementation.

## CODE-agent reliability fix

The exact trace showed DeepSeek returning three valid read actions
(`read_cards`/`read_plan`/`read_file`) and then `StallControl` stopping at
`stall_after=3`. The cause was the CODE agent's
`stall_force=lambda _episode: None`: unlike the existing bounded forced-tool
path, this converted a read-only prefix directly into `stalled`. There was no
provider error, malformed tool payload, write-policy denial, or source
traceback.

The fix keeps generic stall behavior unchanged and uses the existing
`force_after_stall="write_kernel_recipe"` path for CODE. A forced write still
passes the existing card-grounding, syntax, marker, no-op, and recipe
validation gates. If no usable artifact exists, the orchestrator makes exactly
one fresh-session retry with bounded verification feedback; it never treats
the model's `done` action as acceptance.

CODE now records one of these bounded outcomes:

```text
CODE_READY
NO_IMPLEMENTABLE_PLAN
PROVIDER_FAILURE
TOOL_PROTOCOL_FAILURE
TURN_BUDGET_EXHAUSTED
```

The deterministic artifact verifier owns `CODE_READY`; provider and protocol
failures remain fail-closed and are not classified as `CODE_DEFECT`.

### Real DeepSeek CODE benchmark

`uv run python scripts/benchmark_code_agent.py` ran eight disposable cases:
five real DeepSeek cases and three local fault probes. All five normal cases
produced valid recipe artifacts after at most two attempts; two needed the
single fresh-session revision. Malformed response, premature `done`, and
repeated no-op probes all produced no writes and terminated within the bounded
turn limits. No Kaggle or Telegram integration was reachable from the
benchmark.

## Post-fix operational rerun

The prior two completed real OBSERVE cycles remain the primary competition
evidence above. A fresh post-fix supervisor run was also started with isolated
state. Its real worker reached `RESEARCH` and remained blocked on an external
HTTPS research request for approximately nine minutes before the disposable
supervisor/worker were terminated safely. It never reached CODE, created no
repair candidate, and performed no mutation. This is recorded as an external
research-timeout limitation, not a CODE defect. The local real DeepSeek CODE
benchmark and production-shaped harness exercised the post-fix CODE path
successfully.

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
CODE_AGENT: READY (5/5 real DeepSeek local artifacts; bounded fault probes)
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains conditional because no natural real source
defect was eligible for autonomous repair, and the fresh competition rerun was
blocked in external research before CODE. Controlled synthetic certification
and the local real-provider CODE benchmark remain green; unrestricted autonomy
remains disabled.

## Remaining risks

- The fresh post-fix real competition rerun was externally blocked in RESEARCH
  before CODE; long-running research/browser requests need separate timeout
  observation.
- The earlier operational cycles did not reach Kaggle kernel training,
  validation, or feedback because CODE produced no verified recipe; the
  disposable benchmark and production-shaped harness did reach and pass the
  repaired CODE path.
- Live external mutation exactly-once behavior remains untested by policy;
  only read-side reconciliation and synthetic safety paths were exercised.
