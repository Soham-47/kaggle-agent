# Task 5 Report

Status: implemented.

- Added direct submission for offline kernels with a known push version.
- Preserved the internet-off variant fallback.
- Threaded `kernel_version` through the client, runner, and orchestrator.
- Set `kernel.enable_internet` to `false`.
- Added a no-push direct-submit test.

Tests:

- `pytest tests/test_submit_ops.py tests/test_submit_errors.py`: passed.
- The wider submit fallback set has four failures because the new global offline setting changes existing cycle setup before submission.
