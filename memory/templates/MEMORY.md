# MEMORY

Durable facts only. Keep short. No secrets.

## User

- Train on Kaggle Kernels. Local = smoke only.
- Submit through the Kaggle API. Never browser-submit.
- Real submit needs `memory/pending_submit.md` status=approved.
- Research: Kaggle API first, then headed research Chrome, then method cards, then DeepResearcher.

## Goals

1. Daily loop with a small surface area.
2. Valid schema baseline, then beat personal best without burning quota.
3. Self-heal: tune → recipe → new → pause.

## Active contest

Swap this block when you change `default_competition`.

- id:
- slug:
- metric:
- public_score: none
- host accelerator notes:

## Lessons (any contest)

- Honor the host GPU policy (`kernel.enable_gpu`).
- Sibling `.py` files are not importable inside Kaggle notebooks. Inline the recipe.
- Do not import pandas in local agent paths unless you add it to local deps.
- Stale `memory/run.lock` after a killed run blocks the next cycle.
