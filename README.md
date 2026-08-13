# kaggle-agent

A daily loop that researches the Kaggle contest you point it at, writes a small experiment, trains on Kaggle Kernels, and submits only after you approve.

The agent code is contest-agnostic. A contest is three files: a YAML under `config/competitions/`, a short `memory/COMPETITION.md`, and a pipeline under `competitions/<id>/`. `competitions/rsna_knee/` is one worked example, not a hard-wired target.

## What a cycle does

RESEARCH always runs first:

1. Kaggle API snapshot (limits, leaderboard, kernels, meta files)
2. Browser pages the API does not give (overview, discussion)
3. One method card per top public kernel
4. Recursive search over Kaggle, arXiv, GitHub, and the web

PLAN reads those cards. CODE applies the contest recipe plus listed datasets. Then local smoke, kernel package, optional push. Live submit waits for Telegram `/yes`.

## Setup

You need Python 3.11+, [uv](https://github.com/astral-sh/uv), and a Kaggle account.

```bash
git clone https://github.com/Soham-47/kaggle-agent.git
cd kaggle-agent
uv sync --extra dev
cp .env.example .env
# edit .env: OPENCODE_API_KEY, optional Telegram tokens
# put ~/.kaggle/kaggle.json in place (Kaggle account → Settings → API)
```

Optional, for discussion HTML:

```bash
bash scripts/start_research_chrome.sh
# sign in to Kaggle once in that window; leave it running
```

## Point it at a contest

```bash
bash scripts/new_competition.sh my_id the-kaggle-url-slug
# edit config/competitions/my_id.yaml (metric, labels, submit.mode)
# edit competitions/my_id/pipeline/ (schema, baseline, recipe)
# set default_competition: my_id in config/settings.yaml
```

`submit.mode` is `notebook` when the host only accepts kernel submits, or `file` for a CSV upload.

Honor the host GPU rules in `kernel.enable_gpu`. Some contests reject P100.

## Run

```bash
uv run pytest
uv run python scripts/run_daily.py --competition my_id
# dry-run is the default
uv run python scripts/run_daily.py --competition my_id --no-dry-run
```

Telegram (separate process):

```bash
uv run python scripts/telegram_bot.py
```

Commands: `/run` (loops then submits once; counts as approval), `/run dry`, `/status`, `/pause`, `/resume`.

## Submit safety

1. `/run` is a full live loop: research until cards, N train slices, one submit.
2. Sending `/run` counts as approval. Cron still writes `memory/pending_submit.md` and can wait for `/yes` unless you pass `--assume-approved`.
3. `/run dry` never spends a submission.

Do not submit through the browser.

## Tools

| Need | What we use |
|------|-------------|
| Limits, LB, kernels, submit | `kaggle_agent.kaggle_api.KaggleClient` + `~/.kaggle/kaggle.json` |
| Discussion HTML | browser-use on `scripts/start_research_chrome.sh` |
| Plan / code brief | OpenCode Zen (`OPENCODE_API_KEY`) |
| Human gate | Telegram `/yes` |

## Memory

Only these files go into the LLM pack:

```text
memory/MEMORY.md
memory/COMPETITION.md
memory/state.md
memory/research.md
memory/research-deep/source-*.md   # last 2
memory/experiments/                # last 2
```

Starter copies live in `memory/templates/`. Do not put secrets in memory files.

## Cron

```bash
bash scripts/install_cron.sh 6
```

Heal ladder: tune → recipe → new → pause. Details: `docs/cron.md`.

## Layout

```text
src/kaggle_agent/           shared agent
competitions/<id>/pipeline  per-contest training recipe
config/competitions/        contest YAML
memory/                     lean files the loop reads
scripts/run_daily.py        entry
scripts/new_competition.sh  scaffold a new contest
```

MIT license. See `LICENSE`.
