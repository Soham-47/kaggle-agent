---
name: kaggle-deep-research
description: Parallel primary-source research for the active Kaggle contest. One subagent per top notebook, discussion thread, or paper. Writes method cards the PLAN/CODE stages can implement. Use when public kernels beat our LB score, when PLAN ignores notebook methods, or the user runs /kaggle-deep-research or /deep-research.
---

# Kaggle deep research (one agent per source)

The daily cycle already runs this in RESEARCH. RESEARCH is an in-process agent: Zen picks tools (list/pull kernels, fetch, search, write_card, harvest_cards, judge_cards, done) until it calls done, hits the time budget, or hits max_tool_turns. harvest_cards still launches one Zen job per source.

You do not wait for a separate command.

## When to run

- Our public score is below a titled public notebook.
- PLAN or CODE ignored notebook methods on the last cycle.
- Before a live submit that is supposed to beat the current personal best.

## Procedure

1. Read our best public score from `memory/MEMORY.md` / `memory/state.md`.
2. List top kernels via `KaggleClient.kernels(slug)` filtered to this contest. Skip host Efficiency LB notebooks (scoreboard, not a model).
3. Add pinned discussion URLs from `memory/research.md` if present.
4. Add at most two papers that the notebooks actually cite. Do not search raw arXiv for leaderboard names.
5. Launch one Zen worker per source in parallel. Cap at 8. Delete old `source-*.md` first.
6. Each worker writes one method card to `memory/research-deep/source-<slug>.md`.
7. Merge a short "Must implement" list into `memory/research.md` under `## Method cards`. Write `competitions/<id>/pipeline/methods.json` for CODE.
8. The research agent calls judge_cards and done when cards look implementable. Hard stop is time plus `research.agent.max_tool_turns`.

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
