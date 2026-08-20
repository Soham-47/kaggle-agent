# Self-healing supervisor

The supervisor is an opt-in process around the existing worker. It launches a
managed runtime generation, records worker heartbeats and typed exit states,
persists incidents, classifies failures, and handles only bounded repair paths.

Source repairs use separate DeepSeek sessions for classification, specification,
spec review, implementation, and code review. The implementer receives an
isolated Git worktree and allowlisted tools. The supervisor independently runs
verification, diff and protected-path gates, and review acceptance before a
candidate can become a new generation.

Modes:

- `off`: no supervisor lifecycle.
- `observe`: capture and classify incidents without repair activation.
- `repair_only`: create and verify candidate repairs without activating them.
- `auto_safe`: conservative promotion mode; disabled by default.

## Risk-adaptive AUTO_SAFE

When explicitly enabled, automatic repair uses four deterministic tiers:

- `LOW` covers bounded, reproducible local implementation defects. Candidate
  generation and promotion may be automatic after verification and review.
- `MEDIUM` covers broader stage or adapter changes. It uses a larger bounded
  envelope and requires the full non-integration suite before promotion.
- `HIGH` covers lifecycle, replay, generation, or worker-sensitive changes.
  A candidate may be prepared for inspection, but activation requires
  authority.
- `PROHIBITED` covers trust-base, credential, approval, external-action
  identity, reconciliation, policy, and test-weakening changes. No autonomous
  model-authored candidate is allowed.

The decision uses failure class, stage/subsystem, semantic markers, external
state certainty, reproduction quality, candidate scope, repair history, and
verification evidence. Diff size is only one guardrail. An unresolved kernel
push or submission always blocks automatic promotion, and Git revisions are
never added to external-action identity keys.

The explicit profile is opt-in:

```bash
kaggle-agent supervisor --profile controlled-auto-safe --competition <id>
```

User configuration can tune tier envelopes within the global hard ceiling, but
cannot make `HIGH` or `PROHIBITED` trust-base work automatic, remove review,
allow dependencies, or override protected semantics. Missing or invalid policy
configuration fails closed.

See the source-level contracts in `src/kaggle_agent/supervisor/` and the tests
under `tests/test_supervisor*.py`.
