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

See the source-level contracts in `src/kaggle_agent/supervisor/` and the tests
under `tests/test_supervisor*.py`.
