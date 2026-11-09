#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DRAFT=${DRAFT:-qwen3.5-9b}
DEV=${DEV:-cuda:2}
DOMAINS=${DOMAINS:-"math500 wikitext"}
BITS=${BITS:-"8 4"}
mkdir -p out/accept out/logs
for DOM in $DOMAINS; do
  for B in $BITS; do
    OUT="out/accept/quant_${DOM}_9b_w${B}.json"
    [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
    python -m infra.quant_accept --draft "$DRAFT" --bits "$B" --group 128 --skip-lm-head \
      --data "data/processed/${DOM}_target.jsonl" --cache "out/target_cache/${DOM}_k256.npz" \
      --batch-size 2 --device-map "$DEV" --out "$OUT"
  done
done
