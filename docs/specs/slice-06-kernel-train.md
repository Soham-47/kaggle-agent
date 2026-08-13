# Spec: Slice 6 — Kaggle Kernel train package

## Objective

Prepare a Kaggle Kernel notebook package for the pilot competition, optionally push/poll/download via the official API. Local cycle always **builds** the package; **push** only when not dry-run and config allows.

## Success criteria

1. `notebook_builder` writes a valid `.ipynb` + `kernel-metadata.json` under `competitions/<id>/notebooks/`.
2. Notebook writes `submission.csv` with RSNA 12-label header (including `Baker's`).
3. `KERNEL_TRAIN` phase: build package; if `kernel.push` and not dry_run → push via API.
4. `VALIDATE_SUB` validates best local candidate CSV (smoke or kernel output).
5. Unit tests offline with fake kernel API; no real push in dry-run.

## Boundaries

- Always: competition data only on Kaggle for real train; local = package + smoke CSV.
- Never: browser submit; silent real push in dry-run.
