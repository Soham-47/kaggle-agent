# Implementation Plan: Risk-Adaptive AUTO_SAFE Controlled Rollout

## Overview

Validate and minimally harden the existing deterministic risk-evaluation layer
against controlled rollout scenarios. The policy and tier set are frozen; only
concrete validation defects may change implementation behavior. The work must
preserve the existing generation, replay, outbox, verification, review, and
promotion architecture.

## Architecture decisions

- Risk policy remains a pure evaluation layer in `supervisor/risk.py`.
- Protected paths and semantic checks remain deterministic and cannot be
  overridden by configuration or model output.
- Risk is evaluated before spec authoring, after spec approval, after the
  candidate diff, and after review; same-repair HIGH/PROHIBITED floors cannot
  silently de-escalate.
- Existing `RepairAcceptance` and `GenerationPromotion` remain the final
  enforcement points.
- Checked-in defaults keep AUTO_SAFE disabled; live validation uses disposable
  state/generation roots only.

## Tasks

1. Audit merged-main behavior and reproduce the green baseline.
2. Enforce monotonic same-repair risk floors and persist transition metrics.
3. Add scenario, canary, process-recovery, rollback, and external-state gate tests.
4. Run controlled OBSERVE/REPAIR_ONLY/live read-only validation where available.
5. Run full verification, review the complete diff, and write certification docs.

## Invariants

- PROHIBITED and unresolved external-state repairs cannot generate autonomous
  candidates or promote.
- HIGH may produce an isolated candidate but cannot automatically promote.
- Reviewer, verification, protected-path, protected-semantic, dependency, and
  budget gates remain mandatory.
- External logical action keys do not depend on Git revisions.
- AUTO_SAFE remains disabled by default.
- HIGH and PROHIBITED decisions cannot de-escalate within one repair attempt.
- Ambiguous or unknown external state blocks autonomous candidate generation and
  promotion until reconciliation is exact.
