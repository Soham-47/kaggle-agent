# Implementation Plan: Risk-Adaptive AUTO_SAFE

## Overview

Add a deterministic risk-evaluation layer around the existing supervisor repair
artifacts. The layer will assign LOW, MEDIUM, HIGH, or PROHIBITED, derive
tier-specific envelopes, persist decisions, and gate candidate generation and
automatic promotion without changing the existing generation, replay, outbox,
verification, or review implementations.

## Architecture decisions

- Risk policy is a pure evaluation layer in `supervisor/risk.py`.
- Protected paths and semantic checks remain deterministic and cannot be
  overridden by configuration or model output.
- Risk is evaluated before spec authoring, after spec approval, and after the
  actual candidate diff exists; the safest final result wins.
- Existing `RepairAcceptance` and `GenerationPromotion` remain the final
  enforcement points.
- Checked-in defaults keep AUTO_SAFE disabled; the explicit controlled profile
  opts into the risk-adaptive policy.

## Tasks

1. Add typed risk tiers, inputs, decisions, and deterministic evaluation.
2. Add strict risk-adaptive configuration and explicit profile defaults.
3. Persist and enforce provisional/final risk decisions in repair flow.
4. Add scenario and invariant tests, status/metrics artifacts, and docs.
5. Run full verification, review the complete diff, and record findings.

## Invariants

- PROHIBITED and unresolved external-state repairs cannot generate autonomous
  candidates or promote.
- HIGH may produce an isolated candidate but cannot automatically promote.
- Reviewer, verification, protected-path, protected-semantic, dependency, and
  budget gates remain mandatory.
- External logical action keys do not depend on Git revisions.
- AUTO_SAFE remains disabled by default.
