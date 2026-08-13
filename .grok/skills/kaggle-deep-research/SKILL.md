---
name: kaggle-deep-research
description: Parallel primary-source research for the active Kaggle contest. One subagent per top notebook, discussion thread, or paper. Writes method cards the PLAN/CODE stages can implement. Use when public kernels beat our LB score, when PLAN ignores notebook methods, or the user runs /kaggle-deep-research or /deep-research.
---

# Kaggle deep research (one agent per source)

The daily `DeepResearcher` lists kernel titles and often distills off-topic arXiv. PLAN only sees a cut of `research.md`. This skill (and the in-cycle `source_cards` workers) pull methods from named sources that CODE can ship.

The daily orchestrator already runs this during RESEARCH. You do not wait for a separate command.

## When to run

- Our public score is below a titled public notebook.
- PLAN or CODE ignored notebook methods on the last cycle.
- Before a live submit that is supposed to beat the current personal best.

## Procedure

1. Read our best public score from `memory/MEMORY.md` / `memory/state.md`.
2. List top kernels via `KaggleClient.kernels(slug)` filtered to this contest. Skip host Efficiency LB notebooks (scoreboard, not a model).
3. Add pinned discussion URLs from `memory/research.md` if present.
4. Add at most two papers that the notebooks actually cite. Do not search raw arXiv for leaderboard names.
5. Launch one worker per source in parallel. Cap at 8.
6. Each worker writes one method card to `memory/research-deep/source-<slug>.md`.
7. Merge a short "Must implement" list into `memory/research.md` under `## Deep research digest`. Prefer Kaggle notebook URLs first. Write `competitions/<id>/pipeline/methods.json` for CODE.

## Method card template

```markdown
# <title>
- ref: owner/slug or URL
- claimed_public: <score or unknown>
- backbone / input: ...
- labels: gold / report-rules / LLM
- CV: random vs grouped (site / scanner / patient)
- inference: how test IDs are found (csv vs study dirs)
- copyable next step: one kernel change we can ship
- do not copy: ...
```

## Hard rules

- Do not submit. Do not browser-submit.
- Train stays on Kaggle Kernels; local = smoke.
- Sibling `.py` files are not importable inside Kaggle notebooks. Inline code only.
- Honor the host accelerator policy in `config/settings.yaml` (`kernel.enable_gpu`). If the host rejects P100 or GPU, keep GPU off and retry CPU on that error.
- A constant probability file scores chance on rank metrics (AUC, MAP). Do not ship one as the live idea.

## After merge

CODE reads the digest, the last two source cards, and `methods.json`. It attaches listed datasets, discovers hidden test IDs from folders, and rank-averages members. Do not wait for another "implement" message when the user already authorized a live cycle.
