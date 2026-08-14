# Waku memory pillars we can steal

Type: research
Status: resolved
Blocked by:

## Question

What does Waku actually do for semantic / episodic / procedural memory and the retrieval gate, and which of those maps to our `memory/*.md` pack without adding SQLite?

## Answer

Waku is a personal-assistant harness. Queryable memory lives in one SQLite file (`.waku/state.db`). A generated `.waku/MEMORY.md` is only a human mirror. kaggle-agent already has the markdown-side of that idea, and `AGENTS.md` forbids a second store.

### 1. Waku pillars (files + data model)

Working memory is rebuilt every turn in `waku/runtime/session.py`: `SOUL.md` (persona) + gated facts/episodes + matching `SKILL.md` bodies + chat history. It is thrown away after the turn.

| Pillar | What it is | Store |
|--------|------------|--------|
| Semantic | Durable facts: `{subject, content, source}` | `facts` + FTS5 `facts_fts` (or Supabase/mem0/Zep/LangMem) |
| Episodic | Dated events: `{happened_at, summary}` | `episodes` + FTS5; search = FTS rank then recency |
| Procedural | How to act | `SKILL.md` files; scan frontmatter always; load body only on keyword overlap ≥ 2 |
| Chat log | Raw turns for later distill | `chat_log` rows (`consolidated` flag) |
| Consolidation | After N exchanges, a cheap model writes facts + one episode | `consolidation.py`; fail-safe: keep log unconsolidated |

Facade: `waku/memory/__init__.py` class `Memory`. Tools: `save_note`, `manage_memory`, `update_soul`, `create_skill`. Calendar ICS / Apple / Google sit next to this; they are not a memory pillar.

### 2. What the retrieval gate decides

`should_retrieve(client, small_model, message) → (retrieve, query, reason)`.

A cheap model answers one question: does this user message need stored life-facts? Output JSON only: `retrieve`, `query`, `reason`. Math / small talk / self-contained → skip. People, plans, history → retrieve, then FTS top-k facts + 3 episodes.

Why: default-on retrieval is slow and biases answers. Fail-open: gate error or no JSON → retrieve. Skills are **not** gated; they match by keyword independently.

kaggle-agent has no chat gate. Each RESEARCH/PLAN/CODE/HEAL cycle already knows it needs the pack. The stealable idea is “do not dump the archive,” which `ingest.py` already does (CORE + last 2 cards + last 2 experiments). Do not add an extra LLM call before PLAN.

### 3. Maps 1:1 vs do not copy

| Waku | Our file | Copy? |
|------|----------|--------|
| Semantic facts | `memory/MEMORY.md` (prefs, goals, best score, lessons) | Yes, keep as source of truth |
| Domain facts | `memory/COMPETITION.md` | Yes (contest-only) |
| Generated `MEMORY.md` mirror | — | No (we already edit markdown by hand) |
| Episodes | `memory/experiments/*.md` (last 2 in pack) | Yes, as dated run notes |
| Chat log | `memory/daily/` | Logs only; never ingest |
| Consolidation | Distill into `research.md` + experiment files | Idea yes; no chat-log summarizer |
| Procedural `SKILL.md` | `AGENTS.md` + method cards `research-deep/source-*.md` | Rules + last 2 cards already; no skill installer |
| Working memory | `build_context_pack()` in `ingest.py` | Yes |
| `state.db` / FTS5 / pgvector | — | **No** |
| `SOUL.md` | — | **No** (assistant persona) |
| Calendar / ICS / GCal / Apple Mail brief | — | **No** |
| mem0 / Zep / LangMem / Notion episodes | — | **No** (map.md: out of scope) |
| Waku `state.db` heartbeat | `memory/state.md` | Different thing: phase/budget/lock, not facts |

Machine files we already have and should **not** stuff into every prompt: `heal.md`, `pending_submit.md`, `kernel_job.md`. Read them only in the phase that owns them.

### 4. Recommended tiny layout (keep; do not grow)

Stay on the current ingest list. Every extra file costs tokens every cycle (`ingest.py` comment).

| File | Role | Who reads |
|------|------|-----------|
| `memory/MEMORY.md` | Semantic: user, goals, active-contest scores, lessons | PLAN, CODE, HEAL, ops snapshot |
| `memory/COMPETITION.md` | Semantic: slug, metric, labels, submit header | same |
| `memory/state.md` | Cycle machine state | orchestrator + pack |
| `memory/research.md` | Distilled research (cards/digest first) | RESEARCH writes; PLAN/CODE read |
| `memory/research-deep/source-*.md` | Last 2 method cards (how to implement) | PLAN/CODE via ingest |
| `memory/experiments/*.md` | Last 2 episodes | PLAN/HEAL via ingest |
| `memory/daily/` | Raw log | humans / evals; **not** the pack |
| `AGENTS.md` | Standing procedure | humans + agent instructions |

Do not add: `SOUL.md`, `state.db`, FTS, a retrieval-gate model call, calendar, or a second product (mem0/Zep).

Optional later (only if a cycle proves CORE is too big): split “always load” vs “phase load.” Not a Waku gate — just `ingest.py` choosing files by phase.

### 5. Sources

Remote:

- https://github.com/ShenSeanChen/waku-agent
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/docs/architecture.md
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/__init__.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/retrieval_gate.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/consolidation.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/semantic/store.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/episodic/store.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/memory/procedural/loader.py
- https://raw.githubusercontent.com/ShenSeanChen/waku-agent/main/waku/runtime/session.py
- https://api.github.com/repos/ShenSeanChen/waku-agent/contents/waku/memory

Local:

- `/home/soham/kaggle-agent/src/kaggle_agent/memory/ingest.py`
- `/home/soham/kaggle-agent/src/kaggle_agent/memory/write.py`
- `/home/soham/kaggle-agent/memory/MEMORY.md`
- `/home/soham/kaggle-agent/AGENTS.md`
- `/home/soham/kaggle-agent/.scratch/memory-and-loops/map.md`
