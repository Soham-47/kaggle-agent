#!/usr/bin/env bash
# Backward-compatible wrapper for the generic CLI scaffold command.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID="${1:-}"
SLUG="${2:-}"
if [[ -z "$ID" || -z "$SLUG" ]]; then
  echo "usage: bash scripts/new_competition.sh <id> <kaggle-slug>" >&2
  echo "  example: bash scripts/new_competition.sh titan_v2 titanic" >&2
  exit 1
fi

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
PYTHON_BIN="${KAGGLE_AGENT_PYTHON:-python3}"
exec "$PYTHON_BIN" -m kaggle_agent.cli init --competition "$ID" --slug "$SLUG"
