#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
TARGET_DM=${TARGET_DM:-cuda:0}
DRAFT_DM=${DRAFT_DM:-cuda:1}
DRAFTS=${DRAFTS:-"qwen3.5-2b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"math500 wikitext"}
GAMMAS=${GAMMAS:-"8"}
LIMIT=${LIMIT:-10}
MAX_NEW=${MAX_NEW:-128}
mkdir -p out/loop out/logs

for DOM in $DOMAINS; do
  DATA="data/processed/${DOM}_target.jsonl"
  for D in $DRAFTS; do
    for G in $GAMMAS; do
      OUT="out/loop/${DOM}_${D}_g${G}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      if [ -f "$DATA" ]; then
        python -m infra.spec_loop --target "$TARGET" --draft "$D" --domain "$DOM" \
          --data "$DATA" --limit "$LIMIT" --gamma "$G" --max-new "$MAX_NEW" \
          --device-map-draft "$DRAFT_DM" --device-map-target "$TARGET_DM" --out "$OUT"
      else
        python -m infra.spec_loop --target "$TARGET" --draft "$D" --domain "$DOM" \
          --limit "$LIMIT" --gamma "$G" --max-new "$MAX_NEW" \
          --device-map-draft "$DRAFT_DM" --device-map-target "$TARGET_DM" --out "$OUT"
      fi
    done
  done
done
