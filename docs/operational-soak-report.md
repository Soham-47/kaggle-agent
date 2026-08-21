# Operational Soak Report

## Scope and baseline

- Branch: `operational-soak`
- Starting SHA: `87596a557c771897de6426aea456b9246f67a9b0`
- Competition: `rsna_knee` / `rsna-knee-abnormality-detection`
- DeepSeek: available through the existing dotenv path
- Kaggle credentials: available; authenticated read preflight passed
- Telegram credentials: available, but no Telegram command or message was sent
- AUTO_SAFE remained bounded and no real Kaggle mutation was enabled

Fresh baseline:

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     696 passed, 1 deselected
git diff --check                                          PASS
```

The final suite contains one additional regression test for the discovered
managed-state bug, so the final count is `697 passed, 1 deselected`.

## Real soak cycles

Three disposable attempts were made across two isolated state roots
(`/tmp/kaggle-agent-soak-20260821-asiCKi` for the failing run and
`/tmp/kaggle-agent-soak-fixed-gYj9qp` for the two reruns):

1. The first attempt reached real RESEARCH and CODE, then was stopped after
   its bounded CODE retry path repeatedly rejected valid card references.
2. After the fix below, cycle `cycle-975520268f0d` completed all three dry-run
   slices successfully.
3. Cycle `cycle-e3522a1cd7c5` completed all three dry-run slices successfully.

The two completed cycles therefore produced six successful dry-run slices.
Each slice reached RESEARCH, PLAN, CODE, LOCAL_SMOKE, local-only
KERNEL_TRAIN, VALIDATE_SUB, dry submission preparation, FEEDBACK, HEAL, and
REPORT. The worker results were `SUCCESS`; there were no worker incidents,
repairs, promotions, rollbacks, exhausted incidents, duplicate workers, or
duplicate promotions.

The normal research loop remained bounded. Some research agents reached their
turn cap and some source-card writes were rejected by the existing provenance
guards; these were recorded as research/tool outcomes and never classified as
`CODE_DEFECT`. No false `CODE_DEFECT` classification occurred.

## Genuine defect found and fixed

Managed workers correctly redirected mutable memory and research cards to the
external `KAGGLE_AGENT_STATE_ROOT`, but the CODE source-card catalog still
looked only at `<code_root>/memory/research-deep`. In a real worker this made
valid card references fail with `unknown source card`, causing the bounded CODE
loop to spend both attempts without a usable artifact.

The smallest fix was to resolve the catalog through the existing `memory_dir`
path helper, preserving the state-root architecture. A regression test places
cards only in the managed state root and verifies that `write_methods` resolves
them successfully.

Files changed:

- `src/kaggle_agent/agents/code.py`
- `tests/test_plan_code_agents.py`

The operational run did not uncover an unsafe external-action retry, duplicate
worker, replay error, protected-path acceptance, or policy bypass.

## DeepSeek and CODE reliability

The real 10-case implementer benchmark passed `10/10`; every patch passed its
focused test and independent review.

The separate 5-case CODE benchmark was provider-variable: two runs produced
`3/5` and `4/5` final valid normal artifacts. All malformed, premature-done,
and repeated-no-op probes remained bounded and failed closed. The real soak
cycles completed all six production CODE slices after deterministic validation;
some intermediate model attempts were rejected for stale/fabricated card
references or invalid `CUSTOM_INFER` shape and were safely corrected or
accepted only after artifact validation.

This is sufficient to confirm bounded operational behavior, but the separate
benchmark variance means CODE reliability remains `CONDITIONAL` for a stronger
provider-certification claim.

## Risk and repair metrics

| Metric | Result |
| --- | ---: |
| Completed real dry-run cycles | 2 |
| Completed dry-run slices | 6 |
| Worker incidents | 0 |
| `CODE_DEFECT` incidents | 0 |
| False `CODE_DEFECT` classifications | 0 |
| UNKNOWN incidents | 0 |
| Repairs / accepted repairs | 0 / 0 |
| Promotions / rollbacks | 0 / 0 |
| Risk decisions in real cycles | 0 (no repair incident reached policy evaluation) |
| Duplicate workers | 0 |
| Duplicate promotions | 0 |
| Duplicate logical external actions | 0 |
| Protected changes accepted | 0 |
| Kaggle mutations/submissions | 0 |
| Telegram messages | 0 |

The existing synthetic certification still covers LOW/MEDIUM/HIGH/PROHIBITED
risk behavior, promotion, resume, and rollback. This real soak did not create
a natural repair incident, so it does not independently re-certify those
synthetic paths.

## Required regression verification

```text
uv run python -m compileall -q src examples scripts       PASS
uv run pytest -q -m "not integration"                     697 passed, 1 deselected
uv run pytest -q tests/test_supervisor*.py tests/test_replay_epoch.py \
  tests/test_external_outbox.py tests/test_risk_adaptive_controlled_rollout.py \
  tests/test_plan_code_agents.py                           227 passed
uv run python scripts/run_full_system_harness.py           PASS
  accepted repairs=1, generation 0001→0002, RESUMED, rollback PASS,
  stage calls RESEARCH=1 PLAN=1 CODE=2, Kaggle=0, Telegram=0
uv run python scripts/benchmark_implementer_reliability.py 10/10 PASS
uv run python scripts/benchmark_code_agent.py               bounded; provider-variable
git diff --check                                           PASS
```

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

`RISK_ADAPTIVE_AUTO_SAFE` remains conditional because this campaign observed
no genuine repairable production incident and therefore did not validate a
real risk decision, promotion, or resume. The synthetic canary and full
system harness remain green, but controlled rollout should continue in
OBSERVE/REPAIR_ONLY until more real cycles establish stable CODE and risk
metrics.
