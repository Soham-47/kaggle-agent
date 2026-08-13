#!/usr/bin/env bash
# Source this:  source scripts/export_kaggle_mcp_token.sh
if [[ -f "${HOME}/.kaggle/access_token" ]]; then
  export KAGGLE_API_TOKEN="$(tr -d '\n' < "${HOME}/.kaggle/access_token")"
  echo "KAGGLE_API_TOKEN set from ~/.kaggle/access_token (len=${#KAGGLE_API_TOKEN})"
elif [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  export KAGGLE_API_TOKEN="$(python3 -c 'import json;print(json.load(open(__import__("pathlib").Path.home()/".kaggle"/"kaggle.json"))["key"])')"
  echo "KAGGLE_API_TOKEN set from kaggle.json key (write tools may fail; prefer access_token)"
else
  echo "No ~/.kaggle/access_token or kaggle.json" >&2
  return 1 2>/dev/null || exit 1
fi
