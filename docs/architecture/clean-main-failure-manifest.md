# Clean-main failure manifest

## Historical reproduction

The canonical baseline is the clean `origin/main` worktree at:

```text
SHA: b489498d8f24cd9fd3dad7f1ee7241b32407acf4
Worktree: /home/soham/kaggle-agent-baseline-main
Command: UV_PROJECT_ENVIRONMENT=/home/soham/kaggle-agent/.venv TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 uv run pytest -q -m "not integration"
Result: 478 passed, 38 failed, 1 deselected, 2 warnings in 23.04s
Compile: uv run python -m compileall -q src (passed)
```

The same 38 test names were later reproduced against the current main
revision below. The historical classification table is retained so the
original claim remains auditable.

## Current canonical baseline

The validation worktree was created directly from `origin/main` after
`git fetch origin`:

```text
SHA: 77ddbfc33adbbaf3ff11677e88bec12feeacc754
Worktree: /home/soham/kaggle-agent-live-certification
Branch: supervisor-live-certification
Command: uv run python -m compileall -q src
        uv run pytest -q -m "not integration"
Initial result: 565 passed, 38 failed, 1 deselected, 2 warnings in 44.13s
Final result: uv run python -m compileall -q src (passed)
              604 passed, 1 deselected in 23.41s
```

Every initial failure in the table below was reproduced or explained on the
clean `77ddbfc` checkout. No failure was hidden with a marker or removed from
the suite. The 38-entry discrepancy is resolved: it was the complete clean
main failure set, not a partial 22-entry set, and all 38 now pass.

## Rehabilitation outcome

The 22 environment/dependency failures were made self-contained with a lazy
competition-only DICOM import and synthetic public-shaped study/train
fixtures. No pydicom, torch, transformer, Kaggle dataset, or private data was
added to the repository dependency contract. The 16 behavioral failures were
fixed or aligned with the current safety contract: no browser submission,
durable outbox behavior, unique back-to-back run IDs, current image-template
contracts, and evidence-based research fixtures.

## Classification

`baseline-existing` means the failure is reproducible on clean main with the
repository's checked-in implementation. `missing-environment/dependency`
means the test requires an ignored local dataset/artifact or an unavailable
Python dependency. Neither category is a supervisor regression, and no test
was changed to hide a failure.

| # | Test | Classification | Evidence |
|---:|---|---|---|
| 1 | `tests/test_browser_submit.py::test_orchestrator_does_not_submit_via_browser_when_api_fails` | baseline-existing | Cycle exits before submit and leaves `submit_ok=None`. |
| 2 | `tests/test_duplicate_guardrails.py::test_kernel_train_stops_on_identical_kernel` | baseline-existing | Existing duplicate returns `kernel_duplicate=False` and an error. |
| 3 | `tests/test_heal_kernel_job.py::test_kernel_job_resume_no_second_push` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 4 | `tests/test_heal_kernel_job.py::test_kernel_job_resume_preserves_version_during_polling` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 5 | `tests/test_heal_kernel_job.py::test_kernel_retries_cpu_after_p100_ban` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 6 | `tests/test_heal_kernel_job.py::test_run_kernel_phase_clears_job_when_status_has_enum_prefix` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 7 | `tests/test_heal_kernel_job.py::test_package_matches_existing_identical` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 8 | `tests/test_heal_kernel_job.py::test_package_matches_existing_true_when_only_methods_change` | missing-environment/dependency | Clean copy has no study IDs; notebook builder raises `study_ids required`. |
| 9 | `tests/test_kernel_package.py::test_write_kernel_package` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 10 | `tests/test_kernel_package.py::test_run_kernel_phase_local_only` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 11 | `tests/test_kernel_package.py::test_run_kernel_phase_push_fake` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 12 | `tests/test_kernel_package.py::test_run_kernel_phase_rejects_recorded_duplicate` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 13 | `tests/test_kernel_package.py::test_write_kernel_package_uses_methods_json` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 14 | `tests/test_kernel_package.py::test_write_kernel_package_carries_image_artifact_manifest` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 15 | `tests/test_kernel_package.py::test_write_kernel_package_attaches_resume_dataset_outside_image_contract` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 16 | `tests/test_kernel_package.py::test_resume_dataset_keeps_a_reserved_slot_when_cards_list_six_sources` | missing-environment/dependency | Clean package fixture lacks ignored study metadata; builder raises `study_ids required`. |
| 17 | `tests/test_kernel_recipe.py::test_grouped_cv_validate_handles_original_index_labels` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 18 | `tests/test_kernel_recipe.py::test_discover_test_ids_reads_test_series_folders` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 19 | `tests/test_kernel_recipe.py::test_folder_study_features_counts_series_and_slices` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 20 | `tests/test_kernel_recipe.py::test_main_does_not_write_fallback_for_empty_inputs` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 21 | `tests/test_kernel_recipe.py::test_main_accepts_fewer_than_1000_discovered_ids` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 22 | `tests/test_kernel_recipe.py::test_main_output_has_schema_and_full_discovered_row_count` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 23 | `tests/test_kernel_recipe.py::test_dinov2_member_handles_asymmetric_feature_columns_and_varies_predictions` | missing-environment/dependency | Collection fails with `ModuleNotFoundError: No module named 'pydicom'`. |
| 24 | `tests/test_loop_controller.py::test_next_n_two_trains_once_submit` | baseline-existing | Cycle result does not reach the expected submission state. |
| 25 | `tests/test_loop_controller.py::test_first_slice_fail_submits_second` | baseline-existing | Cycle result does not reach the expected second-slice submission state. |
| 26 | `tests/test_loop_controller.py::test_both_slices_fail_skips_submit` | baseline-existing | Cycle result does not reach the expected skip state. |
| 27 | `tests/test_loop_controller.py::test_assume_approved_submits_once` | baseline-existing | Cycle result does not reach the expected single-submit state. |
| 28 | `tests/test_mcp_submit.py::test_mcp_fail_then_api_then_browser` | baseline-existing | Cycle exits before the expected fallback chain; `submit_ok=None`. |
| 29 | `tests/test_mcp_submit.py::test_mcp_success_skips_api` | baseline-existing | Cycle exits before the expected MCP submit; `submit_ok=None`. |
| 30 | `tests/test_mcp_submit.py::test_live_submit_uses_api_not_mcp` | baseline-existing | Cycle exits before live submit; `submit_ok=None`. |
| 31 | `tests/test_recipe_ranker.py::test_fit_ranker_varies_on_real_test` | missing-environment/dependency | Recipe reports `missing train tables` under clean-main `data/`. |
| 32 | `tests/test_recipe_ranker.py::test_kernel_package_embeds_recipe` | baseline-existing | Clean-main fixture reaches notebook construction without required IDs. |
| 33 | `tests/test_research_loop.py::test_sequential_judge_rejects_until_convergence` | baseline-existing | Scripted agent makes 6 calls instead of the expected 2. |
| 34 | `tests/test_research_loop.py::test_sequential_judge_accepts_when_ready` | baseline-existing | Expected `research agent stop=done` is absent from the log. |
| 35 | `tests/test_research_loop.py::test_sequential_done_without_judge_gets_judged_by_gate` | baseline-existing | Scripted agent makes 6 calls instead of the expected 3. |
| 36 | `tests/test_submission_outbox.py::test_unreconciled_submission_intent_is_never_sent_again` | baseline-existing | Existing sent intent is not reported as `submission_pending`. |
| 37 | `tests/test_telegram_submit.py::test_live_submit_blocked_without_approve` | baseline-existing | Cycle exits before approval gate; `submit_ok=None`. |
| 38 | `tests/test_telegram_submit.py::test_approve_then_second_live_submits` | baseline-existing | First cycle exits before approval request; `waiting_approve=False`. |

## Initial totals

```text
baseline-existing:                 16
missing-environment/dependency:    22
supervisor regression:               0
total failures accounted for:      38
```

The missing-environment group consists of seven `pydicom` collection failures,
fourteen kernel-package/healing cases that require ignored study metadata, and
one ranker case that requires ignored train tables. The two remaining package
and ranker entries are baseline behavior in the clean checkout as recorded in
the table. The initial baseline was therefore unhealthy. It is now green
after the ticketed rehabilitation above; AUTO_SAFE remains disabled pending
the live certification phase.
