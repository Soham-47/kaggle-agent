#!/usr/bin/env bash
# Install daily kaggle-agent cron job (UTC hour, default 6).
# Loads .env so DeepSeek / Telegram work under cron.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOUR="${1:-6}"
PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

MARKER="kaggle-agent supervisor"
# source .env then run; append to cron.log
LINE="0 ${HOUR} * * * cd ${ROOT} && set -a && [ -f ${ROOT}/.env ] && . ${ROOT}/.env; set +a && ${PYTHON} -m kaggle_agent.cli supervisor --competition rsna_knee >> ${ROOT}/memory/daily/cron.log 2>&1  # ${MARKER}"

mkdir -p "${ROOT}/memory/daily"

EXISTING="$(crontab -l 2>/dev/null || true)"
if echo "$EXISTING" | grep -qF "$MARKER"; then
  FILTERED="$(echo "$EXISTING" | grep -vF "$MARKER" || true)"
  { echo "$FILTERED"; echo "$LINE"; } | grep -v '^$' | crontab -
  echo "updated cron: $LINE"
else
  { echo "$EXISTING"; echo "$LINE"; } | grep -v '^$' | crontab -
  echo "installed cron: $LINE"
fi

crontab -l | grep -F "$MARKER" || true
