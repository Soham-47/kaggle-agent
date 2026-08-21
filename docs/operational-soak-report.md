# Operational Soak Report

## Scope and baseline

- Branch: `final-controlled-rollout`
- Starting SHA: `69c326639d924bb61acd18416e5723be1f6b6f6f`
- Competition: `rsna_knee` / `rsna-knee-abnormality-detection`
- State root: disposable `/tmp/kaggle-agent-final-controlled-gHWnEv/state`
- DeepSeek: available through the existing dotenv path; no key was logged or persisted
- Kaggle: authenticated read path used; no mutation was enabled
- Telegram: credentials were available, but no command or message was sent
- AUTO_SAFE: no real-cycle promotion was triggered; unrestricted AUTO_SAFE remained disabled

Fresh baseline before this run:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     697 passed, 1 deselected
git diff --check                                          PASS
```

## Real soak campaign

Five consecutive real worker cycles ran in OBSERVE mode against the configured
RSNA Knee competition with `dry_run=True`. Every run used a real supervisor,
worker subprocess, stage ledger, managed state root, Kaggle read path, and
DeepSeek-backed RESEARCH, PLAN, and CODE stages.

| Cycle | Worker result | Slices | Incidents | Promotions | External mutations |
| --- | --- | ---: | ---: | ---: | ---: |
| `cycle-3078b0dde417` | SUCCESS | 3 | 0 | 0 | 0 |
| `cycle-ddb465a2efaa` | SUCCESS | 3 | 0 | 0 | 0 |
| `cycle-7e8961dc4542` | SUCCESS | 3 | 0 | 0 | 0 |
| `cycle-44343691d223` | SUCCESS | 1 | 0 | 0 | 0 |
| `cycle-3913adab4a04` | SUCCESS | 1 | 0 | 0 | 0 |

Aggregate result: 5/5 successful cycles and 11 attempted dry-run slices;
9 slices had successful CODE stage outcomes and 2 recorded bounded CODE
recoverable failures before a later slice completed the cycle.
All slices reached the safe local path through RESEARCH, PLAN, CODE,
LOCAL_SMOKE, local-only KERNEL_TRAIN, VALIDATE_SUB, dry submission
preparation, FEEDBACK, HEAL, and REPORT where the slice produced a usable
candidate.

Two slices in the third multi-slice cycle recorded bounded CODE recoverable
failures before the later slice succeeded. The failures were caused by the
model proposing stale or fabricated source-card references; deterministic tool
policy rejected them, the bounded session ended, and no source repair or
external action was attempted. This is recorded as CODE reliability evidence,
not as a false `CODE_DEFECT` classification.

## CODE and RESEARCH reliability

Operational counts:

| Metric | Result |
| --- | ---: |
| CODE stage successes | 9 |
| Bounded CODE recoverable slice failures | 2 |
| CODE provider failures | 0 |
| CODE infinite/no-op loops | 0 |
| RESEARCH stage failures | 0 |
| PLAN stage failures | 0 |
| Fresh CODE revision sessions | bounded existing retry path |
| False `CODE_DEFECT` classifications | 0 |

The real implementer reliability benchmark passed 10/10 cases, including
focused verification and independent review.

The separate real DeepSeek CODE benchmark was bounded in all 5 normal cases,
but only 2/5 produced a valid final artifact in this run:

```text
simple_pipeline   bounded, final artifact: fail
multi_step_recipe bounded, final artifact: fail
existing_method   bounded, final artifact: pass
parser_edge       bounded, final artifact: pass after retry
custom_infer      bounded, final artifact: fail
```

Malformed-response, premature-done, and repeated-no-op probes all failed
closed without infinite loops or writes. The benchmark therefore does not
support declaring `CODE_AGENT: READY`; the correct result remains
`CONDITIONAL` pending better provider consistency.

## External integrations and safety

- Kaggle authentication and read/research calls completed during every cycle.
- Kernel training remained local-only (`push=False`).
- Submission preparation remained dry-run; no competition submission occurred.
- Kaggle mutations/submissions: **0**.
- Telegram messages/commands: **0**.
- Unresolved external state was not converted into a source repair.
- Duplicate workers: **0**.
- Duplicate promotions: **0**.
- Duplicate logical external actions: **0**.
- Protected changes accepted: **0**.
- Repair attempts, accepted repairs, promotions, and rollbacks in real cycles:
  **0 / 0 / 0 / 0**.

The previously certified synthetic full-system harness was rerun with the real
DeepSeek path and passed: one accepted repair, generation `0001 → 0002`,
atomic promotion and resume, rollback recovery, stage calls
`RESEARCH=1`, `PLAN=1`, `CODE=2`, Kaggle mutations `0`, and Telegram messages
`0`.

## Genuine defects found or fixed

No new production source defect was fixed during this soak. The merged
state-root fix was present in the starting SHA and remained green under the
five-cycle run. Generated competition recipe/method files produced by the
disposable cycles were restored and are not part of this branch change.

The operational evidence did identify a remaining reliability limitation:
DeepSeek CODE artifact production is bounded and fail-closed, but provider
output is not yet consistent enough for an unconditional CODE readiness claim.
No safety gate was weakened to improve the benchmark result.

## Required regression verification

```text
uv run python -m compileall -q src examples scripts                         PASS
uv run pytest -q -m "not integration"                                       697 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py \
  tests/test_external_outbox.py tests/test_risk_adaptive_controlled_rollout.py \
  tests/test_plan_code_agents.py                                             227 passed
uv run python scripts/run_full_system_harness.py                             PASS
uv run python scripts/benchmark_implementer_reliability.py                  10/10 PASS
uv run python scripts/benchmark_code_agent.py --output /tmp/...              bounded; 2/5 normal artifacts
git diff --check                                                              PASS
```

The repository remained clean until this report was added; no generated
benchmark output or live competition artifact is included.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
FULL_AGENT_HARNESS: READY
CODE_AGENT: CONDITIONAL
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains conditional because this real campaign had no
natural repairable incident and therefore did not exercise a real risk-based
promotion. The synthetic canary and full-system harness remain green, while
the real CODE benchmark still shows provider-dependent artifact reliability.
Continue in OBSERVE/REPAIR_ONLY until that reliability is improved and more
real cycles establish stable risk metrics.
