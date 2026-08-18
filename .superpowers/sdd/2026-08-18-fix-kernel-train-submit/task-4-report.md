# Task 4 Report

## Status

Updated tests that model live kernel output.

## Changes

- Changed the constant-output kernel fixture to contain 1000 rows.
- Kept the three-row fixture for the explicit minimum-row rejection test.
- Kept local smoke fixtures short because production does not apply the kernel-only row guard to smoke output.
- Added this complete report at the required task path.

## Tests

Targeted validation and orchestrator tests: 28 passed.

Full suite: 380 passed, 1 skipped, 8 failed.

## Concerns

- The current worktree contains unrelated user changes and untracked files. They remain untouched.
- Four pre-existing browser and MCP end-to-end tests fail before submit because their copied workspace has no study IDs. Their errors are unrelated to the minimum-row contract.
- Three pre-existing research-loop tests fail because the scripted judge call count and stop log differ. Their errors are unrelated to the minimum-row contract.
- One pre-existing Telegram live-submit test fails because the second run rejects a duplicate recipe. Its error is unrelated to the minimum-row contract.
