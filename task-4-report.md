# Task 4 Report

## Status

Implemented the minimum-row guard for kernel output.

## Changes

- Added `require_min_rows` to CSV validation.
- Added `CompetitionConfig.submission_min_rows`.
- Set RSNA Knee `submission.min_rows` to `1000`.
- Applied the limit only when the orchestrator validates kernel output.
- Updated affected test fixtures to represent valid kernel output.

## Tests

`python3 -m pytest tests/test_pipeline_smoke.py tests/test_config.py tests/test_orchestrator.py tests/test_duplicate_guardrails.py`

Result: 28 passed.

## Concerns

- The environment does not provide `uv`, so tests used `python3 -m pytest`.
- Unrelated dirty files remain unchanged.
