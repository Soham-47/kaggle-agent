# Full System + Harness Certification

## Baseline

- Fresh worktree: `/home/soham/kaggle-agent-full-system-certification`
- Starting `origin/main`: `973486c90aff2ca7e2ce811cec83a818f3dcf813`
- Branch: `full-system-certification`
- Working tree was clean at start.
- Actual merged-main baseline: `678 passed, 1 deselected`.

The previously reported `692 passed` count belongs to the later risk-validation
branch and is not reproducible from this fresh `origin/main`. No baseline
failure was found.

## DeepSeek roles

Real provider smoke passed after installing the existing development extra in
the clean worktree. All roles used independent sessions and typed artifacts:

```text
classifier: CODE_DEFECT
spec author: smoke-repair
spec reviewer: APPROVE
implementer: verified candidate is ready
code reviewer: APPROVE
```

No key, authorization header, or environment dump was persisted.

## REPAIR_ONLY

The existing real disposable coordinator harness passed:

```text
classification: CODE_DEFECT (confidence 0.95)
spec review: APPROVE
candidate: CANDIDATE_ACCEPTED
implementer attempts: 1
focused verification: PASS
code review: APPROVE
candidate revision: 071245f504934be1c07d58360240c6fb33ce7680
active generation before: generation-0001
active generation after:  generation-0001
```

The active generation did not change. Evidence is recorded in
`docs/repair-only-certification.json`.

## Risk-adaptive cases

The existing deterministic risk matrix and supervisor safety suites passed.

| Case | Decision |
| --- | --- |
| LOW deterministic parser repair | Candidate and promotion eligible |
| MEDIUM four-file/400-line adapter repair | Stronger envelope; promotion eligible |
| HIGH replay/generation repair | Candidate allowed; promotion blocked |
| PROHIBITED outbox/external identity repair | Implementer not invoked; no candidate |
| Ambiguous external state | Candidate and promotion blocked |
| Test weakening/protected semantics/dependency change | Rejected fail-closed |

Focused risk, repair, replay, outbox, fault, and recovery tests: `73 passed`.
Negative safety and acceptance tests: `40 passed`.

The merged repository already contains the prior successful synthetic
AUTO_SAFE promotion/resume certification. This run revalidated its component
contracts and the risk decisions, but did not claim a new medium-risk
promotion without an existing end-to-end medium-risk worker fixture.

## Full clean agent cycle

The real `run_daily` orchestrator ran five disposable dry cycles using the
existing stage ledger, state handling, pipeline, validation, feedback, heal,
and report paths. External services were isolated behind the repository’s
existing test boundary fakes; no source or state from the developer checkout
was used.

One representative cycle completed:

```text
LOCK → RESEARCH → PLAN → CODE → LOCAL_SMOKE → KERNEL_TRAIN
→ VALIDATE_SUB → TELEGRAM_APPROVE → SUBMIT → FEEDBACK → HEAL → REPORT
```

Results:

```text
cycles: 5
successful cycles: 5
hard errors: 0
incidents: 0
duplicate flags: 0
promotions: 0
final state: IDLE
run lock held: false
```

This is a safe dry-run certification. It is not a live submission.

## Self-healing lifecycle

The prior merged certification documents a complete synthetic chain through
incident, real DeepSeek repair, independent review, generation promotion,
ResumeRequest consumption, preserved-stage replay, affected-stage rerun, and
successful continuation. This fresh-main run revalidated the real DeepSeek
REPAIR_ONLY path and the local promotion/replay/rollback components.

The current repository does not provide a production-shaped, dependency-
injected worker fixture that can run the entire real `run_daily` process with
synthetic external boundaries while also exercising an unattended real
worker-to-supervisor repair transition. Creating such a fixture would be new
harness architecture, which was outside this certification scope. Therefore a
new full unattended self-healing cycle is not claimed here.

## Recovery and rollback

The supervisor worker and recovery suites passed, including real subprocess
SIGKILL handling for a worker, owned-heartbeat adoption rules, interrupted
promotion resolution to exactly one old/new pointer, durable ResumeRequest
recovery, replacement-worker launch blocking, and rollback after a fatal
resumed worker.

No duplicate workers, promotions, or active-generation pointers were observed
in the exercised cases.

## Long-run metrics

The repeated disposable dry-run campaign recorded:

```text
successful cycles: 5
incidents: 0
repair attempts: 0
promotions: 0
rollbacks: 0
duplicate workers: 0
duplicate promotions: 0
false CODE_DEFECT classifications: 0
```

The real DeepSeek implementer benchmark separately passed `10/10` local
deterministic defects on the first bounded attempt, with no policy findings,
no test weakening, and independent review approval for every case.

## External integrations

Kaggle read-only validation passed through the existing client:

```text
authentication: PASS
competition metadata read: PASS
kernel history read: PASS
submission history read: PASS
Kaggle mutations: 0
```

Live exactly-once mutation behavior remains **NOT TESTED**. No submission or
kernel mutation was performed for certification.

Telegram credentials were unavailable. Live Telegram ownership tests are
therefore **BLOCKED**; no Telegram message was sent.

## Bugs found and fixes

No production source integration defect was found, so no source fix was
required. Validation-only issues were corrected without repository changes:

- installed the existing `dev` extra so the disposable role verifier had
  `pytest`;
- used the repository’s `pythonpath = ["src", "tests"]` contract for the
  direct orchestrator probe;
- used the correct competition argument for Kaggle history methods.

## Final verification

```text
uv run python -m compileall -q src examples scripts  PASS
uv run pytest -q -m "not integration"                 678 passed, 1 deselected
git diff --check                                       PASS
```

The focused suites and five-cycle harness listed above also passed. No new
dependencies or production defaults were added. `supervisor.enabled: false`
and risk-adaptive AUTO_SAFE remain disabled by default.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
FULL_AGENT_HARNESS: NOT READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains CONDITIONAL because live Telegram ownership,
real worker shadow cycles, and a fresh unattended full worker-to-repair-to-
resume run were not certified in this worktree. Unrestricted AUTO_SAFE remains
disabled.
