#!/usr/bin/env bash
# Scaffold a new contest workspace from the templates.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ID="${1:-}"
SLUG="${2:-}"
if [[ -z "$ID" || -z "$SLUG" ]]; then
  echo "usage: bash scripts/new_competition.sh <id> <kaggle-slug>" >&2
  echo "  example: bash scripts/new_competition.sh titan_v2 titanic" >&2
  exit 1
fi

yaml="$ROOT/config/competitions/${ID}.yaml"
ws="$ROOT/competitions/${ID}/pipeline"
if [[ -e "$yaml" ]]; then
  echo "already exists: $yaml" >&2
  exit 1
fi

sed -e "s/my_contest/${ID}/g" -e "s/the-kaggle-url-slug/${SLUG}/g" \
  "$ROOT/config/competitions/_template.yaml" > "$yaml"

mkdir -p "$ws"
if [[ ! -f "$ws/__init__.py" ]]; then
  printf '%s\n' '"""Contest pipeline (runs on Kaggle Kernels)."""' > "$ws/__init__.py"
fi
for name in schema.py baseline.py recipe.py; do
  if [[ ! -f "$ws/$name" && -f "$ROOT/competitions/rsna_knee/pipeline/$name" ]]; then
    # starter copies; edit labels / id column for the new contest
    cp "$ROOT/competitions/rsna_knee/pipeline/$name" "$ws/$name"
  fi
done

cp "$ROOT/memory/templates/COMPETITION.md" "$ROOT/memory/COMPETITION.md"
sed -i "s/my_contest/${ID}/g; s/the-kaggle-url-slug/${SLUG}/g" "$ROOT/memory/COMPETITION.md"

echo "wrote $yaml"
echo "wrote $ws"
echo "updated memory/COMPETITION.md"
echo "set default_competition: ${ID} in config/settings.yaml or pass --competition ${ID}"
