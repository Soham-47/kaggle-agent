# CODE Agent Reliability Certification

## Scope

This certification covers the CODE execution path only. The supervisor,
risk policy, promotion logic, and AUTO_SAFE defaults were not changed.

## Original failure

The clean-branch five-case benchmark reproduced a low completion rate (one
valid artifact in the first run; earlier operational evidence reported two of
five). The sanitized traces showed two contributing causes:

1. The benchmark supplied only a one-line user prompt, while production CODE
   receives the existing targeted `build_context_pack(..., view="code")`.
   The model therefore had to discover plan/cards by spending tool turns.
2. With `stall_after=3` and the existing eight-turn cap, a forced write could
   arrive on the last turn. If deterministic validation rejected it, there
   was no remaining turn for the model to consume the feedback. One concrete
   rejection was an invalid `CUSTOM_INFER` reference (`sub` undefined at the
   top level).

The failure was not treated as a source-code defect in the generated recipe.
It was a bounded CODE-agent execution problem.

## Changes

- `make_code_agent` now nudges the bounded write at `stall_after=2`. The
  configured turn budget is unchanged; the earlier write leaves a feedback
  turn when validation rejects a candidate.
- The real benchmark now uses the production targeted CODE context pack.
- The benchmark has ten normal cases and records sanitized attempt evidence:
  model-call count, tool turns, reads, rejected writes, errors, and bounded
  observations.
- One regression test proves that an invalid bounded write can be corrected
  within the existing eight-turn cap.

No artifact validation, tool permission, protected-path rule, retry budget, or
AUTO_SAFE default was weakened.

## Attempt outcomes

| Outcome | Meaning |
| --- | --- |
| `CODE_READY` | Required recipe/artifact passed deterministic validation. |
| `NO_IMPLEMENTABLE_PLAN` | The plan cannot produce an allowed implementation. |
| `PROVIDER_FAILURE` | Provider failure after bounded handling. |
| `TOOL_PROTOCOL_FAILURE` | Malformed or disallowed tool protocol. |
| `TURN_BUDGET_EXHAUSTED` | The bounded loop ended without a valid artifact. |

The model's `done` signal is not acceptance. Deterministic artifact
validation remains authoritative.

## Sanitized failure traces

The benchmark output records, per attempt, only non-secret operational fields;
the compact certification trace is persisted in
`docs/code-agent-reliability-traces.json`:

```text
case, attempt, outcome, turns, llm_calls, tool_calls,
source_reads, rejected_writes, errors, observations, artifact_valid
```

The traces contain no API keys, authorization headers, environment dumps, or
credential files. The original failing trace pattern was:

```text
read_plan → read_plan/read_cards → read_file → write_kernel_recipe
→ rejected candidate → turn cap
```

The corrected benchmark supplies the targeted context before the loop and
validates the bounded early-write feedback path.

## Real DeepSeek benchmark

The expanded benchmark contains ten normal repairable cases:

| Cases | Result |
| --- | ---: |
| Normal cases with valid final artifacts | **10/10** |
| First-attempt valid artifacts | **10/10** |
| Bounded fault probes | **3/3 fail closed** |
| Infinite loops | **0** |
| Protected-path or dependency changes | **0** |

The fault probes were malformed response, premature `done`, and repeated
no-op action. They did not produce a valid artifact or an unbounded loop.

## Operational dry slices

Two real managed-state dry runs completed three slices each against
`rsna_knee`. All six slices reached RESEARCH, PLAN, CODE, LOCAL_SMOKE,
local-only kernel preparation, VALIDATE, dry submission preparation, FEEDBACK,
HEAL, and REPORT. CODE artifacts were consumed downstream, including on the
second cycle using the same state root.

Observed mutations:

```text
Kaggle mutations/submissions: 0
Telegram messages: 0
Duplicate workers: 0
Duplicate promotions: 0
False CODE_DEFECT classifications: 0
```

## Verification

The final report records the exact compile, full-suite, focused-suite,
full-system harness, implementer benchmark, CODE benchmark, and diff-check
commands and results.

## Readiness

```text
CODE_AGENT: READY
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

`RISK_ADAPTIVE_AUTO_SAFE` remains conditional because the real dry slices did
not produce a natural repair incident; synthetic promotion/replay coverage
continues to pass, but no real operational repair decision was observed in
this run.
