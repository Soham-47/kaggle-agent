# wayfinder:map — Memory, context, and stronger loops

## Destination

A small, contest-agnostic memory and context pack, plus RESEARCH/PLAN/CODE loops that emit valid tool calls, write implementable method cards, and change the kernel. Waku-inspired. Simple English, few files.

## Notes

- Skills: design, implement, kaggle-deep-research, code-simplification.
- Execution is in-scope after the design doc is approved (user asked /design then /implement).
- Keep one memory store: `memory/`. No second database unless the design proves markdown cannot work.
- Train stays on Kaggle Kernels. Local = smoke. No secrets in memory files.
- Prefer fewer files and one StageAgent, not a new framework.

## Decisions so far

- [Waku memory pillars we can steal](issues/01-waku-memory.md) — keep markdown pack; do not add SQLite or a retrieval LLM.
- [Why our tool loop wastes turns](issues/02-loop-tool-calls.md) — 400-token thinking fills the reply; use native `tools=` and force harvest.
- Design approved: [docs/specs/memory-and-loops.md](../../docs/specs/memory-and-loops.md) — Protocol A, harvest reset when thin, CUSTOM_INFER later, eval observe-only.

## Not yet specified

- Whether a closed eval gate should hard-block SUBMIT.
- How far CODE may edit `kernel_recipe.py` vs only a brief + sidecar.

## Out of scope

- Cloning Waku chat/calendar/voice.
- A second memory product (mem0, Zep) in this effort.
- Changing the contest metric or host GPU policy.
