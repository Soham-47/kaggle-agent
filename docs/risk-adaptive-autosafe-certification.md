# Risk-Adaptive AUTO_SAFE Certification

## Baseline

- Starting `origin/main`: `2b49f4d0bca15eed43b96eae75f6f8a4a073a03b`
- Branch: `supervisor-risk-adaptive-autosafe`
- Clean worktree created from merged main; the original dirty checkout was
  preserved separately.
- Baseline verification: `659 passed, 1 deselected`.

## Policy model

The existing repair coordinator, acceptance object, verification harness,
generation store, promotion transaction, replay epochs, and outbox remain the
canonical implementations. `supervisor/risk.py` adds a deterministic
evaluation layer with these tiers:

| Tier | Candidate generation | Automatic promotion | Typical scope |
| --- | --- | --- | --- |
| LOW | allowed | allowed after all gates | reproducible pure/local defects |
| MEDIUM | allowed | allowed with stronger verification | adapters and broader stage work |
| HIGH | allowed | denied; authority required | replay, generation, worker lifecycle |
| PROHIBITED | denied | denied | trust base, credentials, approval, external identity, unresolved uncertainty |

Decisions include score, reasons, evidence factors, tier budgets, reproduction
strength, verification requirements, external reconciliation status, and
authority requirement. Decisions are stored in `risk-pre-spec.json`,
`risk-spec.json`, `risk-post-diff.json`, `risk-latest.json`, and the repair
acceptance artifact. Counters are stored in `risk-metrics.json`.

Risk is computed before spec authoring, after spec approval, and after the
actual candidate diff. The final diff can escalate a repair; it cannot lower a
hard trust-boundary decision. Global ceilings are enforced in addition to each
tier profile.

## Hard invariants

- Protected paths and protected semantic markers override the score.
- `PROHIBITED` never creates an autonomous candidate.
- `HIGH` never promotes automatically.
- Pending, ambiguous, or unknown external state blocks candidate generation
  and promotion.
- Dependencies remain authority-required and are rejected by the existing
  deterministic diff policy.
- Spec review, code review, verification, test integrity, and acceptance gates
  remain mandatory.
- AUTO_SAFE is disabled in checked-in defaults.

## Scenario results

The deterministic scenario suite covers pure parser and stage-local defects,
competition adapters, orchestrator and kernel-stage defects, replay and
generation changes, outbox and approval changes, dependency edits, external
uncertainty, test weakening, and repeated failed repair history. It also
demonstrates that a 400-line bounded adapter repair can be MEDIUM while a
two-line external-action identity change is PROHIBITED.

## Verification

Commands run on this branch:

```text
uv run python -m compileall -q src examples scripts
uv run pytest -q -m "not integration"
git diff --check
```

Result: `677 passed, 1 deselected`; compile and diff checks passed.

Focused risk and supervisor flow tests: `33 passed` in the final focused run.

No Kaggle mutations or Telegram messages were performed. No live production
rollout was attempted in this policy implementation phase.

## Readiness

```text
OBSERVE: READY
REPAIR_ONLY: READY
AUTO_SAFE_CANARY: READY
RISK_ADAPTIVE_AUTO_SAFE: CONDITIONAL
UNRESTRICTED_AUTO_SAFE: NOT READY
```

Risk-adaptive AUTO_SAFE is conditional on controlled operational validation of
the new policy against real incident traffic. The unrestricted mode remains
disabled and is not a target of this change.

## Code-review findings and fixes

- Enforced global hard ceilings at decision time, not only during config load.
- Protected configuration-contract paths and requirements files.
- Aligned MEDIUM reproduction defaults between validator, runtime settings, and
  checked-in profiles.
- Added regression coverage for tier assignment, hard gates, escalation,
  serialization, and default-disabled configuration.
