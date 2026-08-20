# Full System + Harness Certification

## Baseline

```text
starting origin/main: 973486c90aff2ca7e2ce811cec83a818f3dcf813
branch: full-system-certification
working tree: clean at start
baseline: 678 passed, 1 deselected
final: 679 passed, 1 deselected
```

The one additional passing test covers the worker cycle-ID regression found by
the production-shaped harness.

## Fixture architecture

`scripts/run_full_system_harness.py` creates a disposable non-local Git clone,
adds a synthetic competition and data fixture, injects one pure `CODE`-stage
`NameError`, and runs the existing production components:

```text
Supervisor
  -> WorkerLauncher subprocess
  -> worker entrypoint
  -> run_daily / Orchestrator / StageLedger
  -> durable Incident
  -> real DeepSeek supervisor roles
  -> RepairCoordinator / worktree / verification / review
  -> RuntimeGeneration / atomic promotion
  -> replacement WorkerLauncher subprocess
  -> ResumeRequest / replay epochs
```

The fixture hooks only replace external reads and worker-stage model calls with
local deterministic behavior. The supervisor’s classifier, spec author, spec
reviewer, implementer, and code reviewer use the real configured DeepSeek
provider in separate production sessions. No Kaggle or Telegram mutation is
reachable from the fixture.

## Unattended self-healing result

The real harness passed without source edits after fault injection:

```text
incident: b1f95916190585058cb3
repair: repair-b1f959161905-a1
initial generation: generation-0001
candidate generation: generation-0002
candidate revision: 1db75254d4d502ad69b25f18feeb8cda5fa91a02
promotion: PREPARED -> PROMOTED -> RESUMED
first worker PID: 4064879
replacement worker PID: 4065521
accepted repairs: 1
new generations: 1
promotions: 1
```

The exact durable stage evidence was:

```text
RESEARCH callable count: 1
PLAN callable count: 1
CODE callable count: 2
preserved stages: RESEARCH, PLAN
invalidated stages: CODE onward
replacement worker result: SUCCESS
Kaggle mutations: 0
Telegram messages: 0
```

The initial worker’s `RESEARCH` and `PLAN` results were replayed from the
original cycle. `CODE` reran under replay epoch 1 and the downstream synthetic
stages completed successfully.

## DeepSeek roles

The live harness exercised these roles with the real provider and typed
artifacts:

| Role | Result |
| --- | --- |
| failure classifier | `CODE_DEFECT` |
| RepairSpec author | valid bounded spec |
| independent spec reviewer | `APPROVE` |
| repair implementer | focused verification passed |
| independent code reviewer | `APPROVE` |

The separate implementer benchmark also passed `10/10` deterministic defects
on bounded first attempts. No credentials, headers, or hidden model reasoning
were persisted.

## Recovery and rollback

The harness ran fresh supervisor subprocesses from durable state:

| Scenario | Result |
| --- | --- |
| restart after `PROMOTED` with replacement result absent | fresh supervisor reused the same worker identity, launched one real replacement subprocess, and closed `RESUMED` |
| unhealthy replacement with durable `FATAL` result | supervisor returned `ROLLED_BACK`, restored `generation-0001`, and did not retry the candidate |

The focused recovery suite also passed real worker SIGKILL/zombie handling,
owned-heartbeat adoption, interrupted-promotion resolution, replacement launch
blocking, and rollback tests.

## Full clean cycles

Five isolated full dry-run cycles passed through the real orchestrator and
stage sequence:

```text
RESEARCH -> PLAN -> CODE -> LOCAL_SMOKE -> KERNEL_TRAIN
-> VALIDATE_SUB -> TELEGRAM_APPROVE -> SUBMIT -> FEEDBACK -> HEAL -> REPORT
```

Each cycle completed with no hard errors, no incidents, no promotion, and no
external mutation. The tests used synthetic/fake external boundaries and did
not touch developer state.

## Bugs found and fixed

The production-shaped run found one real integration defect. Initial workers
were created with `cycle_id=None`, while resumed workers used the persisted
cycle ID. That could change durable stage identities and experiment identity
across a repair/resume boundary. Initial worker requests now receive a durable
`cycle-<token>` ID; resume requests continue to reuse the original ID.

Added regression test:

```text
tests/test_supervisor_loop.py::test_initial_worker_request_gets_durable_cycle_id
```

The harness also corrected its own acceptance-artifact accounting to read the
supervisor’s durable `repairs/*/acceptance.json` records. No safety gate or
production repair limit was weakened.

## Verification

```text
uv run python -m compileall -q src examples scripts  PASS
uv run pytest -q -m "not integration"                 679 passed, 1 deselected
uv run pytest -q tests/test_supervisor_loop.py tests/test_supervisor_recovery.py tests/test_supervisor_acceptance.py tests/test_supervisor_worker.py tests/test_replay_epoch.py tests/test_external_outbox.py
                                                        49 passed
uv run pytest -q tests/test_orchestrator.py::test_dry_run_cycle (x5)
                                                        5 passed
uv run python scripts/benchmark_implementer_reliability.py
                                                        10/10 passed
git diff --check                                       PASS
```

## External integrations

```text
Kaggle authenticated mutation/submission: 0
Telegram messages: 0
Telegram live certification: not required for this local harness
```

Read-only external reconciliation remains covered by the existing repository
tests and prior certification. Live exactly-once mutation behavior remains
**NOT TESTED**.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
FULL_AGENT_HARNESS: READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`FULL_AGENT_HARNESS` is READY because the complete unattended worker -> repair
-> promotion -> replacement -> replay -> success chain passed using real worker
subprocesses and the real DeepSeek repair path. Risk-adaptive controlled
rollout remains conditional on operational deployment evidence. Checked-in
AUTO_SAFE defaults remain disabled, and unrestricted AUTO_SAFE remains out of
scope.
