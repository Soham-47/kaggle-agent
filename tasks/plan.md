# Implementation Plan: Autonomous self-healing supervisor

## Overview

Add a conservative, out-of-process supervisor around the existing Kaggle
worker. The first implementation slices establish durable state, typed
protocols, policy gates, and safe repair artifacts before wiring live worker
promotion. Supervisor operation remains disabled by default.

## Architecture decisions

* Reuse `StageLedger`, durable stage outputs, `ExternalActionOutbox`,
  `KernelPushRepair`, and existing repair tooling.
* Put supervisor state outside code generations, using
  `KAGGLE_AGENT_SUPERVISOR_DIR` or a repository-parent default.
* Treat generations as immutable committed worktrees and make promotion an
  atomic pointer update.
* Keep `auto_safe` conservative: clean baseline, no protected-path changes,
  no dependency changes, bounded diff, independent verification and review.
* Keep external mutation uncertainty pending until authoritative reconciliation.
* Do not add Git SHA to existing stage idempotency keys; represent checkpoint
  invalidation explicitly.

## Ordered ticket slices

### Phase 1 — baseline and foundations

- [x] Baseline audit and architecture record.
- [ ] Tickets 1–4: package, validated configuration, state root, supervisor lock.
- [ ] Tickets 5–8: clean baseline, revisions, generations, runtime layout.
- [ ] Tickets 9–13: worker protocol, exit states, mode boundary, heartbeat and
  hung-worker handling.

### Checkpoint: foundations

- [ ] Focused supervisor tests pass.
- [ ] Existing non-integration failures are separated from new failures.
- [ ] No existing lock/outbox/stage tests regress.

### Phase 2 — incidents and repair contract

- [ ] Tickets 14–21: structured incidents, redaction, stable signatures,
  deterministic classification, known repair precedence and external
  reconciliation.
- [ ] Tickets 22–25: `RepairSpec`, reproduction modes, spec author boundary,
  and independent spec review contract.

### Phase 3 — isolated repair and acceptance

- [ ] Tickets 26–37: worktree manager, scoped toolbox, implementer boundary,
  verification gates, test integrity, static safety, code review and bounded
  revision loop.
- [ ] Tickets 38–42: protected trust base, prompt-injection resistance and
  durable repair budgets/loop detection.

### Phase 4 — resume, promotion and operations

- [ ] Tickets 43–53: impact analysis, replay epochs, resume requests,
  acceptance, promotion, health check, rollback and resume execution.
- [ ] Tickets 54–59: atomic state, supervisor restart recovery, ownership,
  CLI, cron and Telegram routing.

### Phase 5 — fault injection and rollout

- [ ] Tickets 60–72: fault injection and required safety scenarios.
- [ ] Tickets 73–75: observe, repair-only and auto-safe rollout gates.

## Verification policy

Each ticket records existing behavior, problem, implementation, files, tests,
focused and broader results, review findings, invariants, and risks. Tests are
written before behavior changes where appropriate. Full verification uses:

```bash
uv run python -m compileall src
uv run pytest -q -m "not integration"
```

## Risks

| Risk | Mitigation |
| --- | --- |
| Dirty developer checkout | Refuse AUTO_SAFE; preserve all unrelated changes. |
| External mutation ambiguity | Outbox reconciliation before retry or repair. |
| Repair changes trust base | Protected paths, semantic scans, independent review. |
| Supervisor crash | Atomic JSON writes, ownership tokens, restart inspection. |
| Disk exhaustion | Keep tests local/temporary and report blocked verification. |
