# Daily cron for kaggle-agent

The daily cycle uses `scripts/run_daily.py`. Overlap is blocked by `memory/run.lock`.

## Install (user crontab)

```bash
cd ~/kaggle-agent
bash scripts/install_cron.sh
# or with hour (UTC):
bash scripts/install_cron.sh 6
```

Cron sources `.env` so Zen/Telegram keys work. Log: `memory/daily/cron.log`.

Manual / Telegram:

```text
/run          # full live loop; /run counts as submit approval
/run dry      # practice loop, no real submit
```

`/run` loops then submits once (research until cards, N train slices, one submit).

## What a cycle does

1. RESEARCH: loop until method cards are implementable (Kaggle snapshot, browser, source cards, DeepResearcher)
2. Repeat N train slices: PLAN, CODE, LOCAL_SMOKE, KERNEL_TRAIN, VALIDATE
3. One submit of the best candidate (Telegram `/run` counts as approval)
4. Feedback, heal, report

## Research Chrome (Kaggle sign-in)

Browser research uses a dedicated headed Chrome, not your daily window.

```bash
bash scripts/start_research_chrome.sh
# first time: Sign In on the new window (Google SSO / MFA)
# leave it running. Profile: ~/.local/share/kaggle-agent/chrome
# CDP: http://127.0.0.1:9224  (BU_CDP_URL)
```

Do not attach to daily Chrome port 9222. That port often dies after a Chrome upgrade.

## Telegram bot (separate process)

Cron does not run the bot. Keep it as a user service or `screen`/`tmux`:

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_CHAT_ID=...
uv run python scripts/telegram_bot.py
```

## Kernel resume

If `kernel.push: true` and a job is still RUNNING, the next cron polls that kernel (`memory/kernel_job.md`) instead of pushing a second notebook. If the host rejects the accelerator, the runner flips `enable_gpu` off and pushes again.

## Uninstall

```bash
crontab -l | grep -v 'kaggle-agent.*run_daily' | crontab -
```
