#!/usr/bin/env bash
# Load .env and run the Telegram command bot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
  echo "TELEGRAM_BOT_TOKEN missing (set in .env)" >&2
  exit 1
fi
if [[ -z "${TELEGRAM_CHAT_ID:-}" ]]; then
  echo "TELEGRAM_CHAT_ID not set yet — bot will learn it on first /start or /help"
fi
exec "$ROOT/.venv/bin/python" scripts/telegram_bot.py
