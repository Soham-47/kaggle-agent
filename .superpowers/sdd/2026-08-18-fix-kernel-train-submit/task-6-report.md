# Task 6 Fix Report

Status: corrected.

Fixes:

- `_poll_and_maybe_pull` now extracts traceback details for folderless resumed failures.
- Failed CPU retries now include the same best-effort traceback diagnostics.
- Browser-harness failures remain non-fatal and preserve the original error message.
- Added an integration test for a folderless resumed `ERROR` status.

Verification:

```text
python3 -m pytest tests/test_kernel_runner.py
3 passed
```
