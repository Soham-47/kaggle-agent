# Controlled Production Rollout & Restricted AUTO_SAFE Certification

## Baseline

- Starting `origin/main`: `5aefaef97cfb8051eef5dd38ec47b8d7db46c5a0`
- Ending local revision before final report metadata update: `47302dd131057324f5d8966452297a8d4caabc94`
- Branch: `supervisor-controlled-rollout`
- Working tree: clean after the certification commits
- Baseline: `655 passed, 1 deselected`
- Compile check: passed
- `git diff --check`: passed

The original dirty checkout at `/home/soham/kaggle-agent` was preserved and
was not used for rollout validation.

## Security precondition

The previously exposed DeepSeek credential was not used. The replacement
credential was loaded through the normal dotenv path and validated by the
existing synthetic role smoke; its value is not recorded here:

```text
DeepSeek credential: AVAILABLE
Provider calls: synthetic smoke and one disposable REPAIR_ONLY attempt
```

No authorization header, environment dump, or credential value was persisted.

## CI

Added `.github/workflows/ci.yml` with one Python 3.11 job. It runs:

```text
uv sync --extra dev
uv run python -m compileall -q src examples scripts
uv run pytest -q -m "not integration"
git diff --check
```

The workflow has no Kaggle, Telegram, or DeepSeek secrets and was syntax-
validated locally. A hosted GitHub run was not available from this environment.

## Restricted AUTO_SAFE envelope

Added the explicit, non-default profile:

```text
config/profiles/controlled-auto-safe.yaml
```

It is selected only with:

```bash
kaggle-agent supervisor --profile controlled-auto-safe --competition <id>
```

The profile currently sets:

```text
mode: auto_safe
automatic promotion: true
max repairs/cycle: 5
max repairs/day: 20
max attempts/incident: 3
max changed source files: 8
max changed test files: 1
max changed lines: 500
dependency changes: disabled
spec review: required
code review: required
full tests: required
strict protected paths: enabled
```

The checked-in defaults remain `supervisor.enabled: false`, `mode: observe`,
and `promotion.automatic: false`. Profile loading is an explicit CLI opt-in;
it deep-merges only the selected profile into the normal settings and validates
the merged result.

## OBSERVE shadow

Fresh real configured-competition shadow cycles were not run because the
replacement provider credential was unverified. Therefore this phase records:

```text
cycles: 0
incidents: 0
classifications: 0
false CODE_DEFECT count: not measured in this phase
```

The merged main branch retains the prior OBSERVE certification and the full
provider-independent suite remains green. No new live certification is claimed.

## Telegram

Live Telegram transport and command ownership were not exercised in this run;
no replacement Telegram credential was authorized or verified.

```text
/status: not live-tested
/pause: not live-tested
/resume: not live-tested
/run dry: not live-tested
duplicate /run: not live-tested live
restart persistence: covered by existing durable command tests
Telegram messages sent: 0
```

The existing command and ownership tests passed without contacting Telegram.

## DeepSeek provider smoke

The existing five-role synthetic smoke was run with independent production
sessions. It returned:

```text
classifier: CODE_DEFECT
spec review: APPROVE
implementer: verified candidate is ready
code review: APPROVE
```

The reviewer reported only an informational disposable-environment note; no
blocking finding was returned.

## REPAIR_ONLY

Two real disposable provider-backed REPAIR_ONLY attempts were run. The first
rejected an implementer candidate that exceeded its approved
`max_changed_lines: 4` limit. The second produced a one-file, two-line patch,
passed focused verification, received an independent `APPROVE`, and produced
a candidate commit. Both attempts kept the active generation unchanged. The
sanitized successful result is retained in `docs/repair-only-certification.json`.

The provider-independent supervisor, recovery, safety, command, and acceptance
tests passed. No active generation was changed by this validation work.

## Process kill matrix

The existing disposable-process recovery tests were rerun. They cover fresh
heartbeat adoption, dead-worker interruption, SIGKILL handling, interrupted
promotion resolution, replacement-worker restart, and rollback of a failed
resumed worker.

The full operational kill campaign requested for this phase was not run as a
new live campaign, so no stronger certification is claimed:

| Boundary | Result |
| --- | --- |
| worker recovery with real subprocess | PASS in existing tests |
| interrupted `PREPARED` promotion | PASS in existing tests |
| active pointer already switched | PASS in existing tests |
| `PROMOTED` before replacement launch | PASS in existing tests |
| replacement worker immediate failure | PASS in existing tests |
| every requested supervisor termination boundary | NOT RUN as a fresh campaign |

## Rollback

Existing rollback tests pass for a failed resumed worker and restore the prior
active generation. A new live rollout rollback campaign was not executed.

## Real-cycle AUTO_SAFE dry run

Not run. No provider-backed real cycle was started, and no Kaggle mutation was
performed.

## Controlled repair canary

The synthetic AUTO_SAFE promotion/resume canary is already merged in
`origin/main` and remains covered by the existing certification artifacts.
This branch did not repeat it because the required replacement DeepSeek
credential was not verified. No new generation, promotion, worker, or repair
was created by this phase.

## Kaggle

No live Kaggle client call was made in this phase.

```text
Kaggle mutations: 0
read-side reconciliation: not rerun in this phase
live mutation exactly-once: NOT TESTED
```

The repository's prior read-only reconciliation certification remains the
reference result; it must not be interpreted as mutation-side certification.

## Metrics

Controlled-rollout phase counters:

```text
worker launches: 0 live
incidents: 0 live
CODE_DEFECT classifications: 0 live
UNKNOWN classifications: 0 live
false CODE_DEFECTs: not measured
repair attempts: 0 live
repair acceptances: 0 live
reviewer rejections: 0 live
promotions: 0 live
rollbacks: 0 live
exhausted incidents: 0 live
external reconciliation blocks: 0 live
provider protocol failures: 0 live
```

## Verification

Exact commands and results:

```text
uv run python -m compileall -q src examples scripts
PASS

uv run pytest -q -m "not integration"
658 passed, 1 deselected

uv run pytest -q tests/test_supervisor*.py tests/test_external_outbox.py tests/test_replay_epoch.py tests/test_telegram_commands.py tests/test_telegram_submit.py tests/test_cli_init.py
164 passed

uv run pytest -q tests/test_controlled_rollout_profile.py tests/test_supervisor_config.py tests/test_supervisor_loop.py tests/test_supervisor_recovery.py tests/test_supervisor_acceptance.py tests/test_supervisor_safety.py tests/test_supervisor_commands.py
41 passed

git diff --check
PASS
```

## Code review

Blocking findings: none.

Implemented changes are limited to:

- one deterministic baseline CI workflow;
- one explicit restricted AUTO_SAFE profile;
- profile loading and a constrained supervisor CLI selector;
- profile tests and operational documentation.

No supervisor architecture, outbox, replay, generation, approval, rollback,
or protected-path semantics were redesigned. Checked-in defaults remain safe.

## Readiness

```text
OBSERVE: READY (prior certification retained; fresh live shadow CONDITIONAL)
REPAIR_ONLY: READY (fresh disposable provider-backed candidate accepted; no promotion)
AUTO_SAFE_CANARY: READY (prior merged synthetic certification retained)
RESTRICTED_PRODUCTION_AUTO_SAFE: NOT READY
UNRESTRICTED_AUTO_SAFE: NOT READY
```

Restricted production AUTO_SAFE is not ready because fresh real OBSERVE,
Telegram, and real-cycle validations remain incomplete. The profile remains an
explicit opt-in, with the existing upper repair budgets but a one-test-file
cap, and unrestricted AUTO_SAFE remains disabled.
