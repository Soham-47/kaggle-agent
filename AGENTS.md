# Agent instructions: Kaggle Agent

This is a generic framework for one Kaggle competition at a time. Use
`config/competitions/_template.yaml` and `kaggle-agent init --competition …`
to create a competition contract and pipeline scaffold. Do not assume a
particular competition, dataset, target, metric, or identifier column.

## Architecture and safety

- `StageLedger`, `StageExecutor`, durable stage outputs, `RunLock`, and
  `ExternalActionOutbox` are canonical. Do not build parallel replacements.
- The supervisor and worker are separate processes. The worker never replaces
  its own running source tree.
- Kernel pushes and submissions must record durable intent before mutation and
  reconcile authoritative Kaggle state after uncertainty.
- Never submit through a browser.
- Keep credentials, private data, generated memory, datasets, and competition
  outputs outside tracked source.
- `auto_safe` is disabled by default. Do not weaken protected paths, approval,
  outbox, replay, promotion, rollback, or validation policy.

## Runtime state

Checked-in `memory/templates/` files are sanitized starters. Runtime memory is
generated under the configured state root. `KAGGLE_AGENT_STATE_ROOT` and
`KAGGLE_AGENT_SUPERVISOR_DIR` may relocate mutable state.

## Research and code

Research evidence must be collected before planning. Plans and code must use
the active competition contract and its source cards. Local execution is smoke
testing; full training belongs on Kaggle Kernels.

## Verification

```bash
uv run python -m compileall -q src
uv run pytest -q -m "not integration"
git diff --check
```

For changes affecting supervisor or external actions, run the focused
supervisor, replay, outbox, and approval tests as well.
