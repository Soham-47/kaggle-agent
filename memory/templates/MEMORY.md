# MEMORY

Durable facts only. Keep this file short and free of secrets.

## User

- Train on Kaggle Kernels; local execution is smoke testing.
- Submit through the Kaggle API. Never submit through a browser.
- Require the configured approval flow before a real submission.
- Keep research evidence and experiment results in generated runtime state.

## Goals

1. Produce a valid schema baseline before optimizing a competition metric.
2. Preserve external-action idempotency and durable checkpoints.
3. Keep self-healing conservative and disabled unless explicitly enabled.

## Active contest

Replace this section when a competition is initialized.

- id:
- slug:
- metric:
- public_score: none

## Lessons

- Honor the host accelerator and internet policy.
- Generate competition code from the verified competition contract.
- Keep credentials and private data outside the repository.
