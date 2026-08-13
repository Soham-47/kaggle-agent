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
/run          # dry-run cycle
/run live     # live cycle (submit still needs /yes)
```

## What a cycle does

1. RESEARCH: Kaggle snapshot, browser pages, parallel source cards, DeepResearcher
2. PLAN: Zen reads memory + method cards
3. CODE: apply recipe + apply method cards (datasets, rank-mean, ID discovery)
4. LOCAL_SMOKE, KERNEL_TRAIN (retry CPU if the host rejects GPU), VALIDATE
5. Telegram approve, submit, feedback, heal

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
