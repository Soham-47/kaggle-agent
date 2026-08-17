# Agent Architecture Audit — 2026-08-17

Audit of the kaggle-agent harness against the aman.ai agentic design patterns
primer. Sources: codebase walkthrough, past-run reconstruction from
`memory/daily/`, `memory/kernel_job.md`, submissions list, and the primer
section map (core loop, prompt chaining, routing, parallelization, reflection,
tool use, planning, prioritization, multi-agent, memory, learning, MCP, goal
setting, exception handling, HITL, guardrails, evaluation).

## Verdict summary

| Area | Verdict | Evidence |
|---|---|---|
| Core loop | GOOD | 12-phase cycle, ReAct-style tool loops, state persisted in `memory/state.md` / `memory/kernel_job.md` outside context |
| Context control | GOOD | Per-section char caps, pack rebuilt per phase, in-loop trimming (`loop.py` loop trim ~297) |
| Tool use | GOOD | JSON schemas, invalid-call retry, fallback client, stall → nudge → force → stop control |
| Guardrails | GOOD | `seen_recipe` / `seen_kernel` / `seen_output`, byte-compare re-push guard, eval gates — caught 3 real bugs in 3 days |
| HITL | GOOD | 4 explicit modes: dry-run, Telegram `/yes`, `/run` assume-approved, command veto |
| Harness | GOOD | PID lock with stale takeover, cron, resume via kernel_job, dashboard, Telegram bot |
| Memory | PARTIAL | Per-phase views, caps, `daily/` never in pack; no vector memory, no expiry/supersede; `retrieve` is substring search |
| Evals | PARTIAL | 8 deterministic artifact gates work; plan/code/heal never see eval results; 303 tests use fake LLMs only |
| LLMOps | PARTIAL | Token counts per agent; no cost ($), no latency, no per-phase spend |
| LLM-as-judge | GOOD | PLAN/CODE judge with concrete rejection reasons; works (rejected 2 bad plans) |
| Planning | GOOD | PLAN phase + judge + code brief distillation |
| Self-healing | BROKEN | heal ladder cannot advance: FEEDBACK records `SubmissionStatus.PENDING` as score; `best_score` stuck at `none` |
| Research loops | BROKEN | Fleet agents always hit turn_cap/nudge spam; "discussions" and "datasets" agents are web-search aliases; sequential research has no stall control |
| Submit failure healing | PARTIAL | Pin (403) errors auto-healed; 409 title-conflict and 403 submit errors have no handler; DNS outage kills the cycle |

## Score history (reconstructed)

| Time | Run | Result |
|---|---|---|
| 08-16 | 172109 | 0.526 (offline variant, no DINOv2 weights) |
| 08-16 | 184615 | validate rejected: output identical to 172109 |
| 08-16 | 190915 | stale-kernel bug → submitted stale artifact, 0.526, wasted slot (commit 9128801) |
| 08-17 | 054649 | submit DNS failure; manual submit → 0.526 |
| 08-17 | 090501 | 0.520 regression (wguesdon datasets 403 + n_estimators 300) — fixed via pins commit 5c8da90 |
| 08-17 | 132537 | same unimplemented weighted-rank-vote hypothesis; weights 0 |

Learning works: 3 failures were root-caused and fixed with regression tests
within hours (same-kernel, duplicate output, 403 datasets). But score feedback
never reaches HEAL, so the self-healing ladder never advances.

## Root cause of broken self-healing

`_feedback` (`src/kaggle_agent/orchestrator.py:1808`) samples `subs[0]`
immediately after submit:

```python
subs = self._kaggle.submissions(self.competition.slug, top=3)
latest = subs[0]
result.feedback_score = latest.public_score or latest.status
```

At that moment the submission is still `SubmissionStatus.PENDING`, so
`latest.public_score` is empty and `feedback_score` becomes the string
`"SubmissionStatus.PENDING"`. Every feedback line in the daily logs confirms:

```
feedback status=SubmissionStatus.PENDING score=
heal decision_next=tune note=no public score yet — keep tuning flat_days=0 best=none
```

`decide_next` (`src/kaggle_agent/heal/policy.py:66`) parses the score with
`_parse_score`; the string fails to parse → `best is None` → the function
returns early at line 102-107 with "keep tuning" and never increments
`tune_attempts` or `no_improve_days`. The ladder (tune → recipe → new →
pause) cannot advance, so the agent never pauses and never escalates after a
regression.

`_update_loop_after_feedback` (`orchestrator.py:456`) has the same guard:
`parse_loop_score("SubmissionStatus.PENDING")` is None → return. `loop_n`
never adapts either.

Secondary issue: a score that lands minutes after the loop exits is never
ingested. FEEDBACK runs once, right after submit; the loop then ends. The
late score stays on Kaggle, unseen by heal.md / loop.md / experiments.

## Fix plan

1. FEEDBACK samples the first submission with a numeric `public_score` in the
   fetched list, never records a status string as a score.
2. FEEDBACK waits for the just-submitted ref to score (bounded poll, ~10-15
   min), so the loop records the score it caused.
3. Cycle start ingests late scores: if previous cycles submitted and the LB
   now has numeric scores, patch experiments, advance heal/loop state.
4. Cost/latency observability in tracing (later item).

Items 1-3 landed (2026-08-17): `src/kaggle_agent/heal/feedback.py`,
`_feedback` rewrite + `_catch_up_scores` in `orchestrator.py`, `feedback.*`
settings, tests in `tests/test_feedback_loop.py`.

## Out-of-scope findings for a later pass

- Research: real Kaggle discussions/datasets sources, query dedup, stall
  control on sequential research, nudge spam. **Landed (2026-08-17)**:
  `DiscussionSource` / `DatasetSource` in `research/deep.py` use real Kaggle
  API endpoints; fleet specs now restrict kinds to `("discussion",)` /
  `("dataset",)`; `DeepResearcher._mark_query` dedups queries across nodes;
  stall control is wired into the sequential research agent; a nudge is
  logged once per stall episode. Tests in `tests/test_research_sources_fix.py`.
- Evals: route eval results to plan/code/heal prompts.
- Memory: expiry/supersede tracking for stale notes.
- Submit-error classes: 409 title conflict, 403 submit, DNS backoff.
