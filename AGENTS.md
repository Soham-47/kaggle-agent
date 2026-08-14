# Agent instructions: kaggle-agent

This repo runs one Kaggle competition at a time. Set `default_competition` in `config/settings.yaml` and keep that contest's facts in `memory/COMPETITION.md` plus `config/competitions/<id>.yaml`.

## Channels

| Action | Tool |
|--------|------|
| Download meta CSV, list kernels, submit, LB, limits | `kaggle_agent.kaggle_api.KaggleClient` + `~/.kaggle/kaggle.json` |
| Discussions / HTML pages the API cannot give | browser-harness (research only; headed Chrome from `scripts/start_research_chrome.sh`) |
| Deep research (notebooks, papers, repos, web) | `research/agent.py` tool loop; `source_cards.py` / `deep.py` as tools |
| LLM plan / code brief / distill / vision | OpenCode Zen (`OPENCODE_API_KEY`) |
| Approve a real submit | Telegram `/yes` (when enabled) |

Never submit via the browser. Never invent a second memory store.

RESEARCH always runs in this order: Kaggle snapshot, then browser pages, then the research tool-loop. PLAN and CODE are the same kind of tool-loop. Source cards and `pipeline/methods.json` are what CODE implements.

## Memory (only these are ingested)

| File | Role |
|------|------|
| `memory/MEMORY.md` | user prefs, goals, best score, lessons |
| `memory/COMPETITION.md` | active contest only |
| `memory/state.md` | phase, budget, heartbeat fields |
| `memory/research.md` | distilled research (Kaggle snapshot + browser + deep digest) |
| `memory/research-deep/source-*.md` | last 2 method cards (PLAN/CODE) |
| `memory/experiments/*.md` | last 2 loaded into context |
| `memory/daily/` | logs only; not in the context pack |

## Rules

1. Read memory before PLAN/HEAL.
2. Smallest code change that can improve score.
3. Train on Kaggle Kernels; local = smoke.
4. No secrets in memory files.
5. One approved slice at a time unless the user grants autonomy.
6. CODE follows method cards: attach named datasets, find hidden test IDs from folders, rank-average members.

## Commands

```bash
cd ~/kaggle-agent
uv run pytest
uv run python scripts/run_daily.py --competition <id>
# with Zen plan:
OPENCODE_API_KEY=... uv run python scripts/run_daily.py
```
