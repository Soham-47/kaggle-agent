# Task 5 Report

Status: corrected.

Fixes:

- Added the required report path.
- Kept submit tests compatible with `kernel.enable_internet=false`.
- Derived `kernel_ref` from `kernel-metadata.json` when the caller omitted it.
- Parsed `version_number` and `versionNumber` from object and mapping responses.
- Updated `kernel_version` after a CPU retry push.
- Preserved `kernel_version` in resume job state when available.
- Added focused tests for metadata references and mapping push responses.

Verification:

Command:

```text
pytest tests/test_submit_ops.py tests/test_kernel_history.py tests/test_orchestrator.py
```

Output:

```text
============================== 18 passed in 1.28s ==============================
```

Command:

```text
pytest tests/test_submit*.py tests/test_mcp_submit.py tests/test_kaggle_client.py
```

Output:

```text
26 passed, 3 failed in 14.80s
```

The three failures are existing wider-cycle failures. They stop before submit because the generated kernel reports `study_ids required`. The failure is unrelated to submit response parsing or version handling.

## Follow-up Fix

The status polling path created a new `KernelJob` without the existing kernel version. This caused a resume poll to replace the saved version with `none`.

The poll now copies `result.kernel_version` into each saved `KernelJob`. A regression test polls an active job, resumes it, and verifies that version `7` remains saved.
