# kaggle-agent

kaggle-agent is a daily loop for one Kaggle contest at a time.
It reads public kernels, writes a small experiment, and trains on Kaggle Kernels.
It submits a file only after you approve.

The agent code does not bind to one contest.
A contest is three files: YAML in `config/competitions/`, `memory/COMPETITION.md`, and a pipeline in `competitions/<id>/`.
`competitions/rsna_knee/` is one example. It is not the only contest.

## What a cycle does

RESEARCH always runs first:

1. Kaggle API snapshot (limits, leaderboard, kernels, meta files)
2. Browser pages that the API does not give (overview, discussion)
3. One method card per top public kernel
4. Recursive search over Kaggle, arXiv, GitHub, and the web

PLAN reads those cards.
CODE applies the contest recipe and the listed datasets.
Then the cycle does a local smoke test and builds a kernel package.
It can push the package.
A live submit waits for Telegram `/yes`.

## Setup

You need Python 3.11+, [uv](https://github.com/astral-sh/uv), and a Kaggle account.

```bash
git clone https://github.com/Soham-47/kaggle-agent.git
cd kaggle-agent
uv sync --extra dev
cp .env.example .env
# edit .env: DEEPSEEK_API_KEY, optional Telegram tokens
# put ~/.kaggle/kaggle.json in place (Kaggle account → Settings → API)
```

Copy `.env.example` to `.env`.
Set `DEEPSEEK_API_KEY` in `.env`. Telegram tokens are optional.
Put `kaggle.json` in `~/.kaggle/` (Kaggle account, Settings, API).

For discussion HTML, run this command:

```bash
bash scripts/start_research_chrome.sh
# sign in to Kaggle once in that window; leave it running
```

Sign in to Kaggle one time in that window.
Leave the window running.

## Point it at a contest

```bash
bash scripts/new_competition.sh my_id the-kaggle-url-slug
# edit config/competitions/my_id.yaml (metric, labels, submit.mode)
# edit competitions/my_id/pipeline/ (schema, baseline, recipe)
# set default_competition: my_id in config/settings.yaml
```

Edit `config/competitions/my_id.yaml` (metric, labels, `submit.mode`).
Edit `competitions/my_id/pipeline/` (schema, baseline, recipe).
Set `default_competition: my_id` in `config/settings.yaml`.

`submit.mode` is `notebook` when the host accepts only kernel submits.
`submit.mode` is `file` for a CSV upload.

Obey the host GPU rules in `kernel.enable_gpu`.
Some contests reject P100.

## Run

```bash
uv run pytest
uv run python scripts/run_daily.py --competition my_id
# dry-run is the default
uv run python scripts/run_daily.py --competition my_id --no-dry-run
```

Dry-run is the default.
If you want a live cycle, add `--no-dry-run`.

Telegram is a separate process:

```bash
uv run python scripts/telegram_bot.py
```

Commands:

- `/run` does a full live loop. It counts as approval.
- `/run dry` does a practice loop. It does not spend a submission.
- `/status`, `/pause`, `/resume`

## Submit safety

1. `/run` is a full live loop: research until cards, N train slices, one submit.
2. `/run` counts as approval.
3. Cron still writes `memory/pending_submit.md`.
4. Cron can wait for `/yes` unless you pass `--assume-approved`.
5. `/run dry` does not spend a submission.

CAUTION: Do not submit through the browser.

## Tools

| Need | What we use |
|------|-------------|
| Limits, LB, kernels, submit | `kaggle_agent.kaggle_api.KaggleClient` + `~/.kaggle/kaggle.json` |
| Discussion HTML | browser-use on `scripts/start_research_chrome.sh` |
| Plan / code brief | Official DeepSeek API (`DEEPSEEK_API_KEY`) |
| Approve submit | Telegram `/yes` |

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

Starter copies are in `memory/templates/`.
Do not put secrets in memory files.

## Cron

```bash
bash scripts/install_cron.sh 6
```

The heal ladder is tune, then recipe, then new, then pause.
Read `docs/cron.md` for details.

## Layout

```text
src/kaggle_agent/           shared agent
competitions/<id>/pipeline  per-contest training recipe
config/competitions/        contest YAML
memory/                     lean files the loop reads
scripts/run_daily.py        entry
scripts/new_competition.sh  scaffold a new contest
```

The project uses the MIT license. See `LICENSE`.
