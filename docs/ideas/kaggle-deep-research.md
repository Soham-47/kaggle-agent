# Parallel source research for any Kaggle contest

## Problem

PLAN/CODE need the actual methods from public notebooks that already beat us, not a title list and not a constant-score file.

## Direction

One worker per named source (top kernels, host discussion, cited paper). Each writes a method card. Merge only steps CODE can ship into `research.md` and `pipeline/methods.json`. Do not rely on a generic web crawl for this.

This is part of the daily RESEARCH phase.

## Assumptions to check per contest

- [ ] Titled scores in notebook titles are public LB, not only CV.
- [ ] We can ship a slice of those methods under the host accelerator rules.
- [ ] Hidden test IDs may live in study folders, not only the public `test.csv`.

## Scope

Skill + in-cycle source cards + digest merge + CODE applying `methods.json`.

## Not this

- Another generic crawl for leaderboard names.
- Pasting full notebooks into PLAN.
