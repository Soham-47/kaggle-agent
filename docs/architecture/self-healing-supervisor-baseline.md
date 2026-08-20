# Self-healing supervisor baseline

## Reconciliation date

2026-08-20 (Asia/Kolkata)

## Revisions

The preserved dirty developer checkout is on
`feature/autonomous-agent-loop` at `d05caf91cd3e240ef45066d20042c715405235d2`.
The requested reference SHA (`45cbd031e59860b62b8ee2137b0936fa16ee8afb`) is
not the latest main revision. `origin/main` was fetched and is now
`b489498d8f24cd9fd3dad7f1ee7241b32407acf4`.

The clean baseline worktree is:

`/home/soham/kaggle-agent-baseline-main`

## Working-tree state

The working tree is dirty before this project. It contains tracked edits,
large deletions under `inference-course/`, memory changes, and untracked
runtime/research artifacts. These changes are preserved. AUTO_SAFE must refuse
to import this developer checkout until it is clean.

## Baseline checks

The clean baseline worktree had an empty `git status --porcelain=v1`.

* `TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run python -m compileall src`:
  passed.
* `TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -m "not integration"`:
  `478 passed, 38 failed, 1 deselected, 2 warnings`.

The current dirty checkout was rerun with the same environment and command:

* `534 passed, 22 failed, 1 deselected, 2 warnings`.

The 22 current failures are not all pre-existing in the same sense: the clean
baseline has 38 failures, including 23 failures that do not occur in the
current checkout. Each current failure was compared against the clean run.

## Current failure classification

| Current failing test | Classification | Evidence / cause |
| --- | --- | --- |
| `tests/test_browser_submit.py::test_orchestrator_does_not_submit_via_browser_when_api_fails` | baseline-existing | Fails on clean `origin/main` with the same `submit_ok is None` behavior. |
| `tests/test_duplicate_guardrails.py::test_kernel_train_stops_on_identical_kernel` | baseline-existing | Fails on clean `origin/main`; duplicate state is reported but the result flag remains false. |
| The seven tests in `tests/test_kernel_recipe.py` | missing-environment/dependency | Both runs fail before the test body with `ModuleNotFoundError: No module named 'pydicom'`. |
| The four tests in `tests/test_loop_controller.py` | baseline-existing | All four fail on clean `origin/main` with the same submit/result expectations. |
| The three tests in `tests/test_mcp_submit.py` | baseline-existing | All three fail on clean `origin/main` with the same `submit_ok is None` behavior. |
| `tests/test_recipe_ranker.py::test_kernel_package_embeds_recipe` | baseline-existing | Fails on clean `origin/main`; the generated notebook does not contain the expected embedded recipe text. |
| The three tests in `tests/test_research_loop.py` | baseline-existing | All three fail on clean `origin/main`; judge-loop call/log expectations do not match current behavior. |
| The two tests in `tests/test_telegram_submit.py` | baseline-existing | Both fail on clean `origin/main` before approval assertions because the cycle returns an earlier failure. |

Totals for the current 22 failures:

* 15 baseline-existing;
* 7 missing-environment/dependency (`pydicom`);
* 0 supervisor regressions.

The comparison is test-by-test, rather than an inference from the aggregate
failure count. The clean baseline's additional failures include kernel-package
tests that require the repository's ignored local data and existing generated
artifacts; they are not current supervisor failures.

## Existing repair architecture

The repository already has `RunLock`, `StageInput`, `StageOutcome`,
`StageResult`, `StageExecution`, `StageLedger`, `StageExecutor`, durable stage
outputs, `ExternalActionOutbox`, external action keys and reconciliation,
`KernelPushRepair`, `IncidentStore`, `RepairToolbox`, `DebugController`,
`RepairEnvelope`, `RepairLimits`, and `CodingDebugAgent`. The supervisor
extends these seams instead of creating competing checkpoint or mutation
identities.

## Existing autonomy components

`orchestrator.py` currently performs inline coding-debug repair after a
recoverable stage failure. Runtime state is primarily rooted at `.agent/` and
`memory/`; generation switching therefore needs an explicit layout seam before
managed generations can safely be activated.

## Deviations and constraints

* The preserved developer checkout is dirty and is not at the requested
  reference SHA. No dirty files were discarded.
* `origin/main` is newer than the requested reference SHA and is the baseline
  used for stabilization.
* The host filesystem reached 100% usage during setup. Tests were run with
  `TMPDIR=/dev/shm`; live Kaggle, Telegram, GPU, and unavailable Python
  dependencies remain environment-dependent.
* The current configuration loader is YAML-based and will be extended rather
  than replaced.

## Stabilization worktree

The current supervisor implementation and its supporting changes were copied
onto a dedicated worktree and branch based on `origin/main`:

* worktree: `/home/soham/kaggle-agent-supervisor-stabilization`;
* branch: `supervisor-completion-stabilization`;
* base: `b489498d8f24cd9fd3dad7f1ee7241b32407acf4`.

The original dirty checkout remains at its original path and branch as the
preservation copy. No branch was pushed or merged.

## Stabilization verification

* Focused supervisor/fault/replay suite: `63 passed`.
* `TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run python -m compileall src`:
  passed.
* Full branch suite:
  `543 passed, 38 failed, 1 deselected, 2 warnings`.

The 38 branch failures are the same 38 clean-baseline failures, with no new
supervisor-only regression. Seven remain dependency failures from missing
`pydicom`; the rest are baseline pipeline/data/evolved-behavior failures
documented by the clean run.
