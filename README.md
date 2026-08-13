# kaggle-agent

A daily loop that researches a Kaggle contest, writes a small experiment, trains it on Kaggle Kernels, and submits only after you approve.

Point it at any contest. The active one lives in `config/settings.yaml` (`default_competition`) plus `config/competitions/<id>.yaml` and `memory/COMPETITION.md`. The first contest we ran this on was RSNA Knee Abnormality Detection. The agent code is not tied to that name.

## What a cycle does

RESEARCH runs first, always, and in this order:

1. Kaggle API snapshot (limits, leaderboard, kernels, meta files)
2. Browser pages the API does not give (overview, discussion)
3. One method card per top public kernel, in parallel (`research/source_cards.py`)
4. Recursive DeepResearcher over Kaggle, arXiv, GitHub, and the web

Cards go to `memory/research-deep/source-*.md`. A short digest is merged into `memory/research.md`. Attachable datasets land in `competitions/<id>/pipeline/methods.json`.

PLAN reads that pack (digest first, plus the last two cards). CODE applies the local recipe and the cards: attach listed datasets, find hidden test IDs from study folders, rank-average members. If Zen is available it also writes a short `code_brief.md`. Then local smoke, kernel package, optional push. If the host rejects the GPU (for example P100), the runner turns GPU off and pushes again.

Live submit still waits for Telegram `/yes` unless you already approved.

## Tools

| Need | What we use |
|------|-------------|
| Limits, LB, kernels, submit | `kaggle_agent.kaggle_api.KaggleClient` and `~/.kaggle/kaggle.json` |
| Discussion HTML | browser-harness (research only) |
| Plan / code brief / distill | OpenCode Zen (`OPENCODE_API_KEY`) |
| Human gate | Telegram `/yes` |

Do not submit through the browser. Notebook contests use `kernels_push` then `competition_submit_code`. File contests upload a CSV. MCP submit stays off by default because it cannot push a kernel you do not already own.

## Memory

Only a few files go into the LLM pack:

```text
memory/MEMORY.md
memory/COMPETITION.md
memory/state.md
memory/research.md
memory/research-deep/source-*.md   # last 2
memory/experiments/                # last 2
memory/daily/                      # logs, not in context
```

Do not put secrets in those files.

## Setup

```bash
cd ~/kaggle-agent
uv sync --extra dev
# put OPENCODE_API_KEY and Telegram vars in .env (not committed)
# ~/.kaggle/kaggle.json for the API
# ~/.kaggle/access_token (KGAT_...) if you turn MCP on
```

## Run

```bash
uv run pytest
uv run python scripts/run_daily.py --competition <id>
# dry-run is the default in settings.yaml
uv run python scripts/run_daily.py --no-dry-run
```

Telegram (separate process):

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
uv run python scripts/telegram_bot.py
```

Useful bot commands: `/run`, `/run live`, `/yes`, `/no`, `/status`, `/pause`, `/resume`.

## Add another contest

1. Write `config/competitions/<id>.yaml` (slug, metric, labels, submit mode, workspace).
2. Create `competitions/<id>/pipeline/` with at least `schema.py`, `baseline.py`, `recipe.py`.
3. Put a short `memory/COMPETITION.md` for that contest.
4. Set `default_competition: <id>` or pass `--competition <id>`.
5. Honor the host accelerator rules in `kernel.enable_gpu`.

## Submit safety

1. The cycle writes a candidate and `memory/pending_submit.md`.
2. You send `/yes`.
3. The next live cycle may call the Kaggle API.
4. Dry-run never spends a submission.

If you reuse an old `/yes`, check that `kernel_path` / `kernel_ref` are not leftover from a broken package.

## Cron and heal

```bash
bash scripts/install_cron.sh 6   # daily 06:00 UTC
```

Heal ladder: tune → recipe → new → pause (`memory/heal.md`). If a kernel is still RUNNING, the next cron polls it instead of pushing again. Details: `docs/cron.md`.

## Layout

```text
src/kaggle_agent/     agent library
competitions/<id>/    per-contest pipeline and notebooks
config/               settings + contest yaml
memory/               lean files the loop reads
scripts/run_daily.py  cron / CLI entry
tests/                pytest
```
