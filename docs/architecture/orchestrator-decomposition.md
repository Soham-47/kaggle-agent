# Orchestrator decomposition design

## Current responsibility surface

`src/kaggle_agent/orchestrator.py` currently coordinates the complete cycle:

- cycle lifecycle, lock/state persistence, and daily/trace events;
- Kaggle snapshot and browser/deep research;
- research verification and fleet execution;
- PLAN/CODE agents and artifact verification;
- local smoke, kernel package/push/poll/reconciliation, and output validation;
- Telegram approval, submission outbox/API/MCP submission, and feedback polling;
- heal policy, debug-agent repair, durable stage execution, and final reports.

The large file is intentional legacy orchestration. Its externally visible
contract is the fixed phase order and conservative submission policy.

## Bounded controller proposal

The next safe boundaries are:

1. `CycleCoordinator`: lifecycle, lock ownership, phase order, and stage routing.
2. `ResearchController`: Kaggle/browser/deep research, fleet execution, and
   typed evidence verification.
3. `ExperimentController`: PLAN, CODE, local smoke, kernel training, and
   validation for one train slice.
4. `SubmissionController`: approval, outbox reconciliation, API/MCP submission,
   and feedback.
5. `RecoveryController`: debug incidents, kernel repair policy, and HEAL.

Each controller should receive explicit state/result contracts and return a
typed result. Controllers must not acquire a second lock, submit through a
browser, or invent external success when Kaggle state is uncertain.

## Extraction completed in this ticket

The durable stage-output contract is now isolated in
`kaggle_agent.autonomy.stage_outputs`. It allowlists downstream-visible fields,
captures them after a stage executes, and restores them on replay. The
orchestrator remains the coordinator but no longer owns the field map or its
serialization/re-hydration rules.

## Sequencing constraints

- Preserve `LOCK → RESEARCH → PLAN → CODE → LOCAL_SMOKE → KERNEL_TRAIN →
  VALIDATE_SUB → approval → SUBMIT → FEEDBACK → HEAL → REPORT`.
- Keep one active cycle lock and the existing stale-lock takeover rules.
- Treat pending/unknown Kaggle operations as pending; authoritative Kaggle
  reads resolve uncertainty.
- Keep Telegram approval, duplicate kernel/output guards, typed research
  verification, and API-only submission unchanged.

Further controller extraction should land one boundary at a time with focused
tests and no simultaneous behavior or policy changes.
