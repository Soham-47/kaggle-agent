# Operations

Run a dry cycle first:

```bash
kaggle-agent run dry
kaggle-agent --dry-run --competition demo
```

Cron should invoke the supervisor when supervisor mode is enabled. Telegram
commands are routed through the supervisor command queue in that mode, so a
duplicate `/run` cannot launch a second worker. `/status`, `/pause`, and
`/resume` operate on durable state.

Read-only Kaggle reconciliation is safe to exercise. Do not create a real
competition submission for framework validation. Inspect logs and state under
the configured mutable state root rather than committing them.

For a controlled rollout, use the explicit `controlled-auto-safe` profile only
after the provider credential, process-recovery, and rollback checks have
passed. The profile is inert unless selected on the supervisor command line;
the repository defaults keep automatic promotion disabled.
