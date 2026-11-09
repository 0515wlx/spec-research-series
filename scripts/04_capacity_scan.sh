#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
TARGET=${TARGET:-qwen3.5-35b-a3b}
TARGET_DM=${TARGET_DM:-balanced}
DRAFT_DM=${DRAFT_DM:-cuda:0}
DRAFTS=${DRAFTS:-"qwen3.5-0.8b qwen3.5-2b qwen3.5-4b qwen3.5-9b qwen3.5-27b"}
DOMAINS=${DOMAINS:-"math500 gsm8k humaneval wikitext mtbench"}
TOPK=${TOPK:-256}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p out/target_cache out/accept out/logs

for DOM in $DOMAINS; do
  DATA="data/processed/${DOM}_target.jsonl"
  [ -f "$DATA" ] || { echo "missing $DATA (run scripts/03_target.sh)"; continue; }
  CACHE="out/target_cache/${DOM}_k${TOPK}.npz"
  if [ ! -f "$CACHE" ]; then
    python -m infra.cache_target --target "$TARGET" --data "$DATA" \
      --topk "$TOPK" --batch-size 4 --device-map "$TARGET_DM" --out "$CACHE"
  fi
  for D in $DRAFTS; do
    OUT="out/accept/${DOM}_${D}.json"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    case "$D" in
      *0.8b|*2b) DBS=8 ;;
      *4b|*9b)   DBS=4 ;;
      *27b)      DBS=1 ;;
      *)         DBS=2 ;;
    esac
    python -m infra.accept_eval --target "$TARGET" --draft "$D" --data "$DATA" \
      --cache "$CACHE" --gammas 1,2,4,8,16,32,64 --batch-size "$DBS" \
      --device-map "$DRAFT_DM" --out "$OUT"
  done
done
