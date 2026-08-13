# Review: method-card workflow (2026-08-13)

Five-axis review after wiring source cards into RESEARCH → PLAN → CODE.

## Context

Deep research must produce implementable method cards. CODE attaches listed datasets, finds hidden test IDs, and rank-averages prediction tables. Docs are written for any contest.

## Correctness

- Cards run in RESEARCH and write `methods.json`. PLAN sees last two cards and the Method cards section first.
- Required: do not rank-mean LLM/label CSVs (fixed in `kernel_recipe.py`).
- Required: skip DINOv2 on CPU so we do not blend random embeddings (fixed).
- Required: dataset extractor only keeps `kaggle.com/datasets|models` refs plus weight/label names (fixed).
- Required: method cards use `## Method cards` so DeepResearcher does not overwrite them (fixed).
- Kernel runner now polls past QUEUED and retries a P100 ban on status ERROR.

## Readability

- Orchestrator is still one large file. `_source_cards` is a separate helper, which is enough for this slice.
- `apply_from_cards` writes a note; the kernel body is the implementation.

## Architecture

- Research workers are in-process (one thread per kernel, lock around Kaggle pull). That is the daily workflow, not a second memory store.
- Contest-specific extractor/ranker stay under `competitions/<id>/`.
- Residual RSNA names remain in the inlined kernel recipe (expected: that file belongs to the contest workspace).

## Security

- `.env` and kaggle secrets are not in the pack or git.
- Kernel is private, internet off.
- Public notebook text goes to Zen for PLAN/brief only.

## Performance

- Six card workers share one API client behind a lock.
- Deep research still re-lists kernels after cards. Acceptable for this slice.

## Verification

- `uv run pytest`: 101 passed after the first wiring pass; targeted re-runs after digest/extractor fixes passed.
- Live cycle: 6 source cards, deep research 14 learnings / 25 sources, CODE applied 3 datasets + 1 model, kernel pushed CPU with those attachments.

## Verdict

Approve the wiring slice after the required kernel/extractor/digest fixes above. Further work (grouped CV, real DINOv2 infer, GitHub token for a remote) is optional.
